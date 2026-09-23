#!/usr/bin/env python3
"""AntiDeepfake(XLS-R-1B)용 경량 LoRA (peft 없이 오프라인 제출 가능).

SSL 앞단은 고정하고 encoder 마지막 N층 Linear + proj_fc 만 적응한다.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from df_lora import LoRALinear, _wrap_module_linears, load_lora_state_dict, lora_state_dict


def inject_antideepfake_lora(
    model: nn.Module,
    last_n_layers: int = 4,
    rank: int = 8,
    alpha: float = 16.0,
    adapt_proj_fc: bool = True,
):
    """AntiDeepfakeVF: backbone.encoder.layers[-N:] + proj_fc."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    encoder = model.backbone.encoder
    layers = list(encoder.layers)
    n = min(max(int(last_n_layers), 0), len(layers))
    targets: list[str] = []
    for index, layer in enumerate(layers[-n:] if n else []):
        layer_targets: list[str] = []
        _wrap_module_linears(layer, rank, alpha, "", layer_targets)
        abs_index = len(layers) - n + index
        targets.extend(f"encoder.layers.{abs_index}.{t}" for t in layer_targets)

    if adapt_proj_fc and isinstance(model.proj_fc, nn.Linear):
        wrapped = LoRALinear(model.proj_fc, rank=rank, alpha=alpha)
        model.proj_fc = wrapped
        targets.append("proj_fc")
    elif adapt_proj_fc and isinstance(model.proj_fc, LoRALinear):
        model.proj_fc.lora_A.requires_grad = True
        model.proj_fc.lora_B.requires_grad = True
        targets.append("proj_fc")

    trainable = [p for p in model.parameters() if p.requires_grad]
    return {
        "targets": targets,
        "trainable_tensors": len(trainable),
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "last_n_layers": n,
    }


def enable_last_encoder_layers(
    model: nn.Module,
    last_n_layers: int = 4,
    adapt_proj_fc: bool = True,
):
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = list(model.backbone.encoder.layers)
    n = min(max(int(last_n_layers), 0), len(layers))
    for layer in layers[-n:] if n else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if adapt_proj_fc:
        for parameter in model.proj_fc.parameters():
            parameter.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    return {
        "targets": [f"encoder.layers[-{n}:]"] + (["proj_fc"] if adapt_proj_fc else []),
        "trainable_tensors": len(trainable),
        "trainable_params": int(sum(p.numel() for p in trainable)),
    }


def save_ad_adapter(path: Path, model: nn.Module, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # proj_fc may be LoRALinear — include full proj if adapted as LoRA via state
    payload = {
        "adapter": "antideepfake_xlsr_lora",
        "state_dict": lora_state_dict(model),
        "meta": meta or {},
    }
    torch.save(payload, path)


def apply_saved_ad_adapter(model: nn.Module, ckpt_path: Path, device=None):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    mode = str(meta.get("mode", "lora"))
    if mode != "lora":
        raise ValueError(f"unsupported AD adapter mode={mode}")
    info = inject_antideepfake_lora(
        model,
        last_n_layers=int(meta.get("last_n_layers", 4)),
        rank=int(meta.get("rank", 8)),
        alpha=float(meta.get("alpha", 16.0)),
        adapt_proj_fc=bool(meta.get("adapt_proj_fc", True)),
    )
    loaded = load_lora_state_dict(model, payload["state_dict"])
    if device is not None:
        model.to(device)
    info["loaded_tensors"] = loaded
    return info
