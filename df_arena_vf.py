"""DF-Arena 1B VF — v1식 믹스 인코딩 + Demucs 마스크 풀링 (frozen, LoRA 없음)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


AUDIO_SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_600
SILENCE_RMS = 1e-5
MASK_SMOOTH_SAMPLES = 320

MODEL_DIR = Path(__file__).resolve().parent / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"


def load_df_arena(device: torch.device, model_dir: Path | None = None):
    model_dir = Path(model_dir) if model_dir is not None else DF_ARENA_DIR
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    previous = Path.cwd()
    os.chdir(model_dir)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(model_dir),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    finally:
        os.chdir(previous)

    model = model.to(device).eval()
    fake_idx = int(model.config.label2id["spoof"])
    return model, fake_idx


def _smooth_envelope(audio: np.ndarray, win: int = MASK_SMOOTH_SAMPLES) -> np.ndarray:
    magnitude = np.abs(audio.astype(np.float64))
    if magnitude.size == 0:
        return magnitude.astype(np.float32)
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(magnitude, kernel, mode="same").astype(np.float32)


def _downsample_weights(values: np.ndarray, num_frames: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    if values.size == 0:
        return np.zeros(num_frames, dtype=np.float32)
    source = np.linspace(0.0, 1.0, num=values.size, dtype=np.float64)
    target = np.linspace(0.0, 1.0, num=num_frames, dtype=np.float64)
    return np.interp(target, source, values).astype(np.float32)


def _segment_starts(n: int) -> list[int]:
    if n <= SEGMENT_SAMPLES:
        return [0]
    starts = list(range(0, n - SEGMENT_SAMPLES + 1, SEGMENT_SAMPLES))
    last = n - SEGMENT_SAMPLES
    if starts[-1] != last:
        starts.append(last)
    return starts


def _extract_segment(audio: np.ndarray, start: int) -> np.ndarray:
    chunk = audio[start : start + SEGMENT_SAMPLES]
    if chunk.size < SEGMENT_SAMPLES:
        chunk = np.pad(chunk, (0, SEGMENT_SAMPLES - chunk.size))
    return chunk.astype(np.float32, copy=False)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _pooled_spoof(frames, weights, classifier, fake_label_index: int) -> float:
    weights = torch.as_tensor(weights, device=frames.device, dtype=frames.dtype).clamp(min=0)
    if float(weights.sum()) < 1e-6:
        return 0.0
    weights = weights / weights.sum()
    pooled = (frames[0] * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
    logits = classifier(pooled)
    probs = torch.softmax(logits.float(), dim=-1)
    return float(probs[0, fake_label_index])


@torch.inference_mode()
def predict_voice_fake_df(model, fake_label_index, mix, voice_stem, device) -> float:
    length = min(mix.size, voice_stem.size)
    mix = mix[:length]
    voice_env = _smooth_envelope(voice_stem[:length])
    classifier = model.backbone.conformer.fc5

    scores: list[float] = []
    for start in _segment_starts(mix.size):
        segment = _extract_segment(mix, start)
        voice_seg = _extract_segment(voice_env, start)
        if _rms(segment) < SILENCE_RMS:
            continue
        segment_tensor = torch.from_numpy(segment).to(device)
        frames = model.encode_frames(segment_tensor)
        num_frames = int(frames.shape[1])
        weights = _downsample_weights(voice_seg, num_frames)
        scores.append(_pooled_spoof(frames, weights, classifier, fake_label_index))
    return max(scores) if scores else 0.0
