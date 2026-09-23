"""AntiDeepfake XLS-R-1B — transformers 전용 로더 + v1식 마스크 풀링.

fairseq/hydra 는 Python 3.12 대회 이미지와 충돌하므로 사용하지 않는다.
safetensors(fairseq 키) → HuggingFace Wav2Vec2Model 로 키를 매핑해 로드한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Wav2Vec2Config, Wav2Vec2Model


AUDIO_SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_600
SILENCE_RMS = 1e-5
MASK_SMOOTH_SAMPLES = 320
FAKE_CLASS_INDEX = 0


def _xlsr_1b_config() -> Wav2Vec2Config:
    return Wav2Vec2Config(
        hidden_size=1280,
        num_hidden_layers=48,
        num_attention_heads=16,
        intermediate_size=5120,
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=True,
        feat_extract_norm="layer",
        feat_extract_activation="gelu",
        layer_norm_eps=1e-5,
        do_stable_layer_norm=True,
        apply_spec_augment=False,
    )


def _map_fairseq_to_hf(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """m_ssl.model.* fairseq 키 → Wav2Vec2Model 키."""
    mapped: dict[str, torch.Tensor] = {}
    unused: list[str] = []

    for name, value in state.items():
        if not name.startswith("m_ssl.model."):
            continue
        key = name[len("m_ssl.model.") :]

        # feature encoder convs: conv_layers.{i}.0.{weight,bias}
        if key.startswith("feature_extractor.conv_layers."):
            rest = key[len("feature_extractor.conv_layers.") :]
            parts = rest.split(".")
            layer_id, type_id = int(parts[0]), int(parts[1])
            if type_id == 0:
                # conv
                suffix = parts[2]  # weight / bias
                mapped[f"feature_extractor.conv_layers.{layer_id}.conv.{suffix}"] = value
                continue
            if type_id == 2:
                # layer norm (layer_norm mode): conv_layers.{i}.2.1.{weight,bias}
                suffix = parts[-1]
                mapped[f"feature_extractor.conv_layers.{layer_id}.layer_norm.{suffix}"] = value
                continue
            unused.append(name)
            continue

        if key.startswith("post_extract_proj."):
            mapped["feature_projection.projection." + key.split(".", 1)[1]] = value
            continue

        if key.startswith("layer_norm."):
            # post-extract layer norm (before projection in fairseq = feature_projection.layer_norm)
            mapped["feature_projection.layer_norm." + key.split(".", 1)[1]] = value
            continue

        if key == "mask_emb":
            mapped["masked_spec_embed"] = value
            continue

        if key.startswith("encoder.pos_conv.0."):
            suffix = key[len("encoder.pos_conv.0.") :]
            # transformers weight_norm parametrization or weight_g/weight_v
            if suffix in {"weight_g", "weight_v", "bias"}:
                mapped[f"encoder.pos_conv_embed.conv.{suffix}"] = value
            continue

        if key.startswith("encoder.layer_norm."):
            mapped["encoder.layer_norm." + key.split(".", 2)[2]] = value
            continue

        if key.startswith("encoder.layers."):
            # encoder.layers.{i}.self_attn.q_proj.weight → encoder.layers.{i}.attention.q_proj.weight
            rest = key[len("encoder.layers.") :]
            layer_id, rem = rest.split(".", 1)
            prefix = f"encoder.layers.{layer_id}."
            replacements = {
                "self_attn.k_proj.": "attention.k_proj.",
                "self_attn.v_proj.": "attention.v_proj.",
                "self_attn.q_proj.": "attention.q_proj.",
                "self_attn.out_proj.": "attention.out_proj.",
                "self_attn_layer_norm.": "layer_norm.",
                "fc1.": "feed_forward.intermediate_dense.",
                "fc2.": "feed_forward.output_dense.",
                "final_layer_norm.": "final_layer_norm.",
            }
            done = False
            for src, dst in replacements.items():
                if rem.startswith(src):
                    mapped[prefix + dst + rem[len(src) :]] = value
                    done = True
                    break
            if not done:
                unused.append(name)
            continue

        # quantizer / final_proj / project_q — classification에 불필요
        if key.startswith(("quantizer.", "final_proj.", "project_q.")):
            continue

        unused.append(name)

    if unused:
        print(f"AntiDeepfake map: {len(unused)} unused ssl keys (ok if quantizer/etc)")
    return mapped


class AntiDeepfakeVF(nn.Module):
    """XLS-R-1B + Linear(1280→2). HF 카드와 동일 헤드."""

    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model(_xlsr_1b_config())
        self.proj_fc = nn.Linear(1280, 2)

    def extract_feat(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, T]
        out = self.backbone(wav)
        return out.last_hidden_state  # [B, T, D]


def load_antideepfake(
    model_dir: Path,
    device: torch.device,
    lora_path: Path | None = None,
) -> AntiDeepfakeVF:
    model_dir = Path(model_dir)
    weight_path = model_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing {weight_path}")

    raw = load_file(str(weight_path))
    model = AntiDeepfakeVF()
    hf_ssl = _map_fairseq_to_hf(raw)
    missing, unexpected = model.backbone.load_state_dict(hf_ssl, strict=False)
    # pos_conv weight_norm 이름 차이 등은 missing에 남을 수 있음 — 치명만 검사
    critical = [
        m
        for m in missing
        if "parametrizations" not in m and "weight_g" not in m and "weight_v" not in m
    ]
    # allow missing pos_conv parametrizations; try alternate keys
    if any("pos_conv_embed" in m for m in missing):
        alt = {}
        for k, v in list(hf_ssl.items()):
            if "pos_conv_embed.conv.weight_g" in k:
                alt["encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = v
            elif "pos_conv_embed.conv.weight_v" in k:
                alt["encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = v
        if alt:
            model.backbone.load_state_dict(alt, strict=False)

    if "proj_fc.weight" in raw:
        model.proj_fc.weight.data.copy_(raw["proj_fc.weight"])
        model.proj_fc.bias.data.copy_(raw["proj_fc.bias"])
    else:
        raise KeyError("proj_fc.* missing in AntiDeepfake weights")

    # 너무 많은 critical missing 이면 경고
    missing2, _ = model.backbone.load_state_dict(hf_ssl, strict=False)
    critical_missing = [
        m
        for m in missing2
        if "parametrizations" not in m and "num_batches_tracked" not in m
    ]
    if len(critical_missing) > 20:
        print(
            f"WARNING: many missing backbone keys ({len(critical_missing)}), "
            f"e.g. {critical_missing[:5]}"
        )

    model = model.to(device)
    if lora_path is not None and Path(lora_path).is_file():
        from ad_lora import apply_saved_ad_adapter

        info = apply_saved_ad_adapter(model, Path(lora_path), device=device)
        print(f"Loaded AntiDeepfake LoRA {lora_path}: {info}")
    return model.eval()


def preprocess_waveform(wav: np.ndarray, device: torch.device) -> torch.Tensor:
    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)
    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    with torch.no_grad():
        tensor = F.layer_norm(tensor, tensor.shape)
    return tensor.unsqueeze(0).to(device)


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


def _segment_starts(audio_length: int, segment: int = SEGMENT_SAMPLES) -> list[int]:
    if audio_length <= segment:
        return [0]
    starts = list(range(0, audio_length - segment + 1, segment))
    last = audio_length - segment
    if starts[-1] != last:
        starts.append(last)
    return starts


def _extract_segment(audio: np.ndarray, start: int, segment: int = SEGMENT_SAMPLES) -> np.ndarray:
    end = start + segment
    chunk = audio[start:end]
    if chunk.size < segment:
        chunk = np.pad(chunk, (0, segment - chunk.size))
    return chunk.astype(np.float32, copy=False)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


@torch.inference_mode()
def predict_voice_fake_masked(
    model: AntiDeepfakeVF,
    mix: np.ndarray,
    voice_stem: np.ndarray,
    device: torch.device,
) -> float:
    length = min(mix.size, voice_stem.size)
    mix = mix[:length]
    voice_env = _smooth_envelope(voice_stem[:length])

    scores: list[float] = []
    for start in _segment_starts(mix.size):
        segment = _extract_segment(mix, start)
        voice_seg = _extract_segment(voice_env, start)
        if _rms(segment) < SILENCE_RMS:
            continue

        wav = preprocess_waveform(segment, device)
        emb = model.extract_feat(wav)
        num_frames = int(emb.shape[1])
        weights = torch.as_tensor(
            _downsample_weights(voice_seg, num_frames),
            device=emb.device,
            dtype=emb.dtype,
        ).clamp(min=0)
        if float(weights.sum()) < 1e-6:
            continue
        weights = weights / weights.sum()
        pooled = (emb[0] * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        logits = model.proj_fc(pooled)
        probs = torch.softmax(logits.float(), dim=-1)
        scores.append(float(probs[0, FAKE_CLASS_INDEX]))

    return max(scores) if scores else 0.0
