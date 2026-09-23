#!/usr/bin/env python3
"""v3.1 추론: v1 뼈대 + AntiDeepfake VF (+ optional LoRA).

VP/MP: 원본 믹스 → PANNs
VF:    믹스 + Demucs 마스크 풀링 → AntiDeepfake (model/ad_lora.pt 있으면 LoRA)
       옵션: --vf-mode df|ensemble 로 DF / max 앙상블 A/B
MF:    원본 믹스 → PANNs 임베딩 → mf_head
FILE:  max(VP×VF, MP×MF)  # v1
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from tqdm import tqdm

from antideepfake_vf import load_antideepfake, predict_voice_fake_masked


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
ANTIDEFP_DIR = MODEL_DIR / "antideepfake_xlsr_1b"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"

DEFAULT_TEST_DIR = Path("data") / "test"
DEFAULT_SAMPLE_SUBMISSION = Path("data") / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = Path("output") / "submission.csv"
DEFAULT_MF_CKPT = MODEL_DIR / "mf_head.pt"
DEFAULT_AD_LORA = MODEL_DIR / "ad_lora.pt"

AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600
SILENCE_RMS = 1e-5

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]

TASK_HEAD_COLUMNS = (
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}

EMBED_DIM = 2048


class MusicFakeHead(nn.Module):
    """PANNs Cnn14 임베딩 위의 음악 fake 분류기."""

    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, embedding):
        return self.net(embedding).squeeze(-1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="v3.1: AntiDeepfake (+LoRA) VF mask-pooled + PANNs VP/MP/MF."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--mf-ckpt", type=Path, default=DEFAULT_MF_CKPT)
    parser.add_argument(
        "--antideepfake-dir",
        type=Path,
        default=ANTIDEFP_DIR,
        help="Local AntiDeepfake XLS-R-1B directory.",
    )
    parser.add_argument(
        "--ad-lora",
        type=Path,
        default=DEFAULT_AD_LORA,
        help="Optional AntiDeepfake LoRA from train_ad_adapt.py. Missing = frozen AD.",
    )
    parser.add_argument(
        "--df-arena-dir",
        type=Path,
        default=DF_ARENA_DIR,
        help="Local DF-Arena 1B directory (vf-mode df|ensemble only).",
    )
    parser.add_argument(
        "--vf-mode",
        choices=["antideepfake", "df", "ensemble"],
        default="ensemble",
        help="VF backbone. default ensemble=max(AD+LoRA, DF). antideepfake/df for A/B.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--vf-device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="VF backbone device. auto = same as --device.",
    )
    parser.add_argument(
        "--file-fuse",
        choices=["max", "noisy_or"],
        default="max",
        help="FILE fusion. v1 default is max.",
    )
    return parser.parse_args()


def select_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(device_name)


def find_audio_files(test_dir):
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    audio_files = []
    for path in test_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.append(path)
    audio_files.sort(key=lambda path: path.stem)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")

    audio_ids = [path.stem for path in audio_files]
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("Audio IDs must be unique")
    return audio_files


def read_sample_submission(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sample submission not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        column_names = reader.fieldnames
        rows = list(reader)

    if column_names is None or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")

    required_columns = ["ID"] + PREDICTION_COLUMNS
    missing_columns = [name for name in required_columns if name not in column_names]
    if missing_columns:
        raise ValueError(f"Sample submission is missing columns: {missing_columns}")

    seen_ids = set()
    for row in rows:
        audio_id = str(row["ID"]).strip()
        if not audio_id:
            raise ValueError("Sample submission contains an empty ID")
        if audio_id in seen_ids:
            raise ValueError(f"Duplicate ID in sample submission: {audio_id}")
        seen_ids.add(audio_id)
        row["ID"] = audio_id

    return column_names, rows


def order_audio_files(audio_files, submission_rows):
    audio_by_id = {path.stem: path for path in audio_files}
    submission_ids = [row["ID"] for row in submission_rows]

    missing_ids = [audio_id for audio_id in submission_ids if audio_id not in audio_by_id]
    extra_ids = [audio_id for audio_id in audio_by_id if audio_id not in submission_ids]
    if missing_ids or extra_ids:
        raise ValueError(
            "Test audio and sample submission IDs do not match. "
            f"Missing: {missing_ids[:5]}, Extra: {extra_ids[:5]}"
        )

    return [audio_by_id[audio_id] for audio_id in submission_ids]


def load_audio(audio_path):
    """읽기 실패·빈 파일은 짧은 무음으로 대체."""
    try:
        path = Path(audio_path)
        if not path.is_file() or path.stat().st_size == 0:
            return np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
        audio, _ = librosa.load(
            str(path), sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32
        )
    except Exception:
        return np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        return np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
    return audio


def load_track(track, audio_channels, samplerate):
    audio, _ = librosa.load(str(track), sr=samplerate, mono=False, dtype=np.float32)
    if audio.ndim == 1:
        wav = torch.from_numpy(audio).unsqueeze(0)
    else:
        wav = torch.from_numpy(np.ascontiguousarray(audio))
    if wav.shape[0] == audio_channels:
        return wav
    if audio_channels == 1:
        return wav.mean(0, keepdim=True)
    if wav.shape[0] == 1:
        return wav.expand(audio_channels, -1).contiguous()
    return wav[:audio_channels]


def get_segment_starts(audio_length):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]
    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio, start):
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[:SEGMENT_SAMPLES].astype(np.float32)
    end = start + SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


def prepare_panns_labels():
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_panns_model(device):
    prepare_panns_labels()
    from panns_inference import AudioTagging, labels

    model = AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device.type,
    )
    config_path = PANNS_DIR / "component_labels.json"
    label_groups = json.loads(config_path.read_text(encoding="utf-8"))
    label_to_index = {label: index for index, label in enumerate(labels)}
    voice_indices = [label_to_index[label] for label in label_groups["voice"]]
    music_indices = [label_to_index[label] for label in label_groups["music"]]
    return model, voice_indices, music_indices


def make_panns_segments(audio):
    segments = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment = librosa.resample(
            segment,
            orig_sr=AUDIO_SAMPLE_RATE,
            target_sr=PANNS_SAMPLE_RATE,
            res_type="soxr_hq",
        )
        segments.append(segment.astype(np.float32))
    return np.stack(segments)


def predict_presence_heads(model, voice_indices, music_indices, audio):
    segments = make_panns_segments(audio)
    predictions, _ = model.inference(segments)
    voice_present = float(predictions[:, voice_indices].max())
    music_present = float(predictions[:, music_indices].max())
    return voice_present, music_present


def load_htdemucs_model():
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = original_torch_load
    return model.cpu().eval()


def estimate_source_stems(audio_path, model, device):
    waveform = load_track(
        audio_path, model.audio_channels, model.samplerate
    ).float()
    mono_waveform = waveform.mean(0)
    mean = mono_waveform.mean()
    std = mono_waveform.std()

    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * AUDIO_SAMPLE_RATE / model.samplerate)
        silence = np.zeros(max(1, length), dtype=np.float32)
        return silence, silence.copy()

    normalized_waveform = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model,
            normalized_waveform[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + mean

    vocal_index = model.sources.index("vocals")
    voice_audio = sources[vocal_index].mean(0, keepdim=True)

    music_sources = []
    for index, source_name in enumerate(model.sources):
        if source_name != "vocals":
            music_sources.append(sources[index])
    music_audio = torch.stack(music_sources).sum(0).mean(0, keepdim=True)

    voice_audio = torchaudio.functional.resample(
        voice_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    music_audio = torchaudio.functional.resample(
        music_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    return (
        voice_audio.cpu().numpy().astype(np.float32),
        music_audio.cpu().numpy().astype(np.float32),
    )


def load_task_head(ckpt_path, device):
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dim = int(payload.get("embed_dim", EMBED_DIM))
    model = MusicFakeHead(dim)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def panns_segment_embedding(model, segment):
    resampled = librosa.resample(
        segment,
        orig_sr=AUDIO_SAMPLE_RATE,
        target_sr=PANNS_SAMPLE_RATE,
        res_type="soxr_hq",
    ).astype(np.float32)
    _clipwise, embedding = model.inference(resampled[None])
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim == 2:
        vector = vector[0]
    return torch.from_numpy(vector.copy())


def predict_music_fake(panns_model, mf_head, audio, device):
    scores = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        if float(np.sqrt(np.mean(np.square(segment, dtype=np.float64)))) < SILENCE_RMS:
            continue
        vector = panns_segment_embedding(panns_model, segment).to(device)
        with torch.inference_mode():
            scores.append(float(torch.sigmoid(mf_head(vector)).item()))
    return max(scores) if scores else 0.0


def release_cuda(*models):
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def empty_head_outputs():
    return {column: 0.0 for column in TASK_HEAD_COLUMNS}


def fuse_file_fake(head_outputs, mode="max"):
    voice_risk = (
        head_outputs["VOICE_PRESENT_PROB"] * head_outputs["VOICE_FAKE_PROB"]
    )
    music_risk = (
        head_outputs["MUSIC_PRESENT_PROB"] * head_outputs["MUSIC_FAKE_PROB"]
    )
    if mode == "noisy_or":
        return 1.0 - (1.0 - voice_risk) * (1.0 - music_risk)
    return max(voice_risk, music_risk)


def write_prediction_row(row, head_outputs, file_fuse="max"):
    for column, value in head_outputs.items():
        row[column] = round(value, 10)
    row["FILE_FAKE_PROB"] = round(fuse_file_fake(head_outputs, file_fuse), 10)


def fuse_vf_scores(ad_score: float, df_score: float, mode: str) -> float:
    if mode == "antideepfake":
        return float(ad_score)
    if mode == "df":
        return float(df_score)
    # ensemble: max — 한쪽이라도 spoof면 ADS에 유리 (v1 CPS 여유 가정)
    return max(float(ad_score), float(df_score))


class FourHeadPredictor:
    """v3.1: VP/MP PANNs, VF AntiDeepfake(+LoRA)+mask, MF PANNs head, FILE max."""

    def __init__(
        self,
        device,
        mf_ckpt,
        antideepfake_dir,
        df_arena_dir=None,
        vf_mode="antideepfake",
        vf_device=None,
        file_fuse="max",
        ad_lora=None,
    ):
        self.device = device
        self.mf_ckpt = mf_ckpt
        self.antideepfake_dir = antideepfake_dir
        self.df_arena_dir = df_arena_dir or DF_ARENA_DIR
        self.vf_mode = vf_mode
        self.vf_device = device if vf_device is None else vf_device
        self.file_fuse = file_fuse
        self.ad_lora = ad_lora

    def run_presence_heads(self, audio_files):
        model, voice_indices, music_indices = load_panns_model(self.device)
        presence_scores = {}
        for audio_path in tqdm(audio_files, desc="Heads VP/MP"):
            try:
                audio = load_audio(audio_path)
                voice_present, music_present = predict_presence_heads(
                    model, voice_indices, music_indices, audio
                )
                presence_scores[audio_path.stem] = {
                    "VOICE_PRESENT_PROB": voice_present,
                    "MUSIC_PRESENT_PROB": music_present,
                }
            except Exception:
                presence_scores[audio_path.stem] = {
                    "VOICE_PRESENT_PROB": 0.0,
                    "MUSIC_PRESENT_PROB": 0.0,
                }
        del model
        release_cuda()
        return presence_scores

    def _estimate_all_stems(self, audio_files):
        htdemucs_model = load_htdemucs_model()
        stems = {}
        for audio_path in tqdm(audio_files, desc="Stems"):
            try:
                stems[audio_path.stem] = estimate_source_stems(
                    audio_path, htdemucs_model, self.device
                )
            except Exception:
                stems[audio_path.stem] = (
                    np.zeros(1, dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                )
        del htdemucs_model
        release_cuda()
        return stems

    def _run_voice_fake_ad(self, audio_files, stems):
        lora = self.ad_lora if self.ad_lora and Path(self.ad_lora).is_file() else None
        model = load_antideepfake(
            self.antideepfake_dir, self.vf_device, lora_path=lora
        )
        print(f"Loaded AntiDeepfake from {self.antideepfake_dir}")
        scores = {}
        for audio_path in tqdm(audio_files, desc="Head VF (AntiDeepfake)"):
            try:
                mix = load_audio(audio_path)
                voice_stem, _ = stems[audio_path.stem]
                scores[audio_path.stem] = predict_voice_fake_masked(
                    model, mix, voice_stem, self.vf_device
                )
            except Exception:
                scores[audio_path.stem] = 0.0
        del model
        release_cuda()
        return scores

    def _run_voice_fake_df(self, audio_files, stems):
        from df_arena_vf import load_df_arena, predict_voice_fake_df

        model, fake_idx = load_df_arena(self.vf_device, self.df_arena_dir)
        print(f"Loaded DF-Arena from {self.df_arena_dir}")
        scores = {}
        for audio_path in tqdm(audio_files, desc="Head VF (DF-Arena)"):
            try:
                mix = load_audio(audio_path)
                voice_stem, _ = stems[audio_path.stem]
                scores[audio_path.stem] = predict_voice_fake_df(
                    model, fake_idx, mix, voice_stem, self.vf_device
                )
            except Exception:
                scores[audio_path.stem] = 0.0
        del model
        release_cuda()
        return scores

    def _run_voice_fake(self, audio_files, stems):
        need_ad = self.vf_mode in {"antideepfake", "ensemble"}
        need_df = self.vf_mode in {"df", "ensemble"}
        ad_scores = (
            self._run_voice_fake_ad(audio_files, stems)
            if need_ad
            else {p.stem: 0.0 for p in audio_files}
        )
        df_scores = (
            self._run_voice_fake_df(audio_files, stems)
            if need_df
            else {p.stem: 0.0 for p in audio_files}
        )
        fused = {}
        for audio_path in audio_files:
            key = audio_path.stem
            fused[key] = fuse_vf_scores(
                ad_scores.get(key, 0.0),
                df_scores.get(key, 0.0),
                self.vf_mode,
            )
        return fused

    def _run_music_fake(self, audio_files):
        panns_model, _, _ = load_panns_model(self.device)
        mf_head = load_task_head(self.mf_ckpt, self.device)
        music_scores = {}
        for audio_path in tqdm(audio_files, desc="Head MF"):
            try:
                mix = load_audio(audio_path)
                music_scores[audio_path.stem] = predict_music_fake(
                    panns_model, mf_head, mix, self.device
                )
            except Exception:
                music_scores[audio_path.stem] = 0.0
        del panns_model
        del mf_head
        release_cuda()
        return music_scores

    def run_fake_heads(self, audio_files, presence_scores):
        music_fakes = self._run_music_fake(audio_files)
        stems = self._estimate_all_stems(audio_files)
        voice_fakes = self._run_voice_fake(audio_files, stems)

        head_outputs_by_id = {}
        for audio_path in audio_files:
            head_outputs = empty_head_outputs()
            head_outputs.update(presence_scores.get(audio_path.stem, {}))
            head_outputs["VOICE_FAKE_PROB"] = voice_fakes.get(audio_path.stem, 0.0)
            head_outputs["MUSIC_FAKE_PROB"] = music_fakes.get(audio_path.stem, 0.0)
            head_outputs_by_id[audio_path.stem] = head_outputs
        return head_outputs_by_id

    def predict(self, audio_files, submission_rows):
        presence_scores = self.run_presence_heads(audio_files)
        head_outputs_by_id = self.run_fake_heads(audio_files, presence_scores)
        for index, audio_path in enumerate(audio_files):
            write_prediction_row(
                submission_rows[index],
                head_outputs_by_id[audio_path.stem],
                self.file_fuse,
            )
        return submission_rows


def save_submission(output_path, column_names, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    device = select_device(args.device)

    audio_files = find_audio_files(args.test_dir)
    column_names, submission_rows = read_sample_submission(args.sample_submission)
    audio_files = order_audio_files(audio_files, submission_rows)

    vf_name = args.device if args.vf_device == "auto" else args.vf_device
    vf_device = torch.device("cpu") if vf_name == "cpu" else select_device(vf_name)
    predictor = FourHeadPredictor(
        device,
        args.mf_ckpt,
        args.antideepfake_dir,
        df_arena_dir=args.df_arena_dir,
        vf_mode=args.vf_mode,
        vf_device=vf_device,
        file_fuse=args.file_fuse,
        ad_lora=args.ad_lora,
    )
    submission_rows = predictor.predict(audio_files, submission_rows)
    save_submission(args.output, column_names, submission_rows)
    print(f"Saved {len(submission_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
