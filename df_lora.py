#!/usr/bin/env python3
"""DF-Arena용 경량 LoRA (peft 없이 오프라인 제출 가능).

대상: Conformer 마지막 N블록의 Linear + fc5.
SSL(XLS-R 1B)은 고정한다 — VRAM·망각 비용을 줄이기 위함.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """기존 Linear에 low-rank 잔차만 학습한다."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        in_features = base.in_features
        out_features = base.out_features
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def _replace_linear(parent: nn.Module, name: str, rank: int, alpha: float):
    child = getattr(parent, name)
    if isinstance(child, LoRALinear):
        return child
    if not isinstance(child, nn.Linear):
        return None
    wrapped = LoRALinear(child, rank=rank, alpha=alpha)
    setattr(parent, name, wrapped)
    return wrapped


def _wrap_module_linears(module: nn.Module, rank: int, alpha: float, path: str, out: list):
    for child_name, child in list(module.named_children()):
        child_path = f"{path}.{child_name}" if path else child_name
        if isinstance(child, nn.Linear):
            wrapped = _replace_linear(module, child_name, rank, alpha)
            if wrapped is not None:
                out.append(child_path)
        elif isinstance(child, LoRALinear):
            continue
        else:
            _wrap_module_linears(child, rank, alpha, child_path, out)


def inject_conformer_lora(
    antispoof_model: nn.Module,
    last_n_blocks: int = 2,
    rank: int = 8,
    alpha: float = 16.0,
    adapt_fc5: bool = True,
):
    """DF_Arena_1B_Antispoofing.backbone.conformer 에 LoRA를 꽂는다."""
    backbone = antispoof_model.backbone
    conformer = backbone.conformer
    for parameter in antispoof_model.parameters():
        parameter.requires_grad = False

    targets = []
    blocks = list(conformer.encoder_blocks)
    n = min(max(int(last_n_blocks), 0), len(blocks))
    for block in blocks[-n:] if n else []:
        _wrap_module_linears(block, rank, alpha, "", targets)

    if adapt_fc5 and isinstance(conformer.fc5, nn.Linear):
        _replace_linear(conformer, "fc5", rank, alpha)
        targets.append("fc5")
    elif adapt_fc5 and isinstance(conformer.fc5, LoRALinear):
        for parameter in conformer.fc5.parameters():
            if parameter is conformer.fc5.lora_A or parameter is conformer.fc5.lora_B:
                parameter.requires_grad = True

    trainable = [p for p in antispoof_model.parameters() if p.requires_grad]
    return {
        "targets": targets,
        "trainable_tensors": len(trainable),
        "trainable_params": int(sum(p.numel() for p in trainable)),
    }


def enable_last_blocks(
    antispoof_model: nn.Module,
    last_n_blocks: int = 2,
    adapt_fc5: bool = True,
):
    """LoRA 대신 마지막 Conformer 블록(+fc5)만 full fine-tune."""
    for parameter in antispoof_model.parameters():
        parameter.requires_grad = False
    conformer = antispoof_model.backbone.conformer
    blocks = list(conformer.encoder_blocks)
    n = min(max(int(last_n_blocks), 0), len(blocks))
    for block in blocks[-n:] if n else []:
        for parameter in block.parameters():
            parameter.requires_grad = True
    if adapt_fc5:
        for parameter in conformer.fc5.parameters():
            parameter.requires_grad = True
    trainable = [p for p in antispoof_model.parameters() if p.requires_grad]
    return {
        "targets": [f"encoder_blocks[-{n}:]"] + (["fc5"] if adapt_fc5 else []),
        "trainable_tensors": len(trainable),
        "trainable_params": int(sum(p.numel() for p in trainable)),
    }


def lora_state_dict(antispoof_model: nn.Module):
    state = {}
    for name, module in antispoof_model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(antispoof_model: nn.Module, state: dict):
    modules = dict(antispoof_model.named_modules())
    loaded = 0
    for key, tensor in state.items():
        if key.endswith(".lora_A"):
            module_name = key[: -len(".lora_A")]
            attr = "lora_A"
        elif key.endswith(".lora_B"):
            module_name = key[: -len(".lora_B")]
            attr = "lora_B"
        else:
            continue
        module = modules.get(module_name)
        if not isinstance(module, LoRALinear):
            raise KeyError(f"LoRA module missing for {module_name}; inject LoRA first")
        getattr(module, attr).data.copy_(tensor.to(getattr(module, attr).device))
        loaded += 1
    return loaded


def save_adapter(path: Path, antispoof_model: nn.Module, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter": "df_conformer_lora",
        "state_dict": lora_state_dict(antispoof_model),
        "meta": meta or {},
    }
    torch.save(payload, path)


def apply_saved_adapter(
    antispoof_model: nn.Module,
    ckpt_path: Path,
    device=None,
):
    """추론용: 체크포인트 meta로 LoRA를 inject한 뒤 가중치를 로드한다."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    rank = int(meta.get("rank", 8))
    alpha = float(meta.get("alpha", 16.0))
    last_n = int(meta.get("last_n_blocks", 2))
    mode = str(meta.get("mode", "lora"))
    if mode != "lora":
        raise ValueError(
            f"adapter mode={mode} is not a LoRA file; merge weights into df_arena instead"
        )
    info = inject_conformer_lora(
        antispoof_model,
        last_n_blocks=last_n,
        rank=rank,
        alpha=alpha,
        adapt_fc5=bool(meta.get("adapt_fc5", True)),
    )
    loaded = load_lora_state_dict(antispoof_model, payload["state_dict"])
    if device is not None:
        antispoof_model.to(device)
    info["loaded_tensors"] = loaded
    return info
