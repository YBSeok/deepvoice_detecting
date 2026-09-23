#!/usr/bin/env python3
"""AntiDeepfake XLS-R-1B 도메인 적응 (v3.1 / v2.5식 aug).

SSL 앞단 고정, encoder 마지막 N층 + proj_fc 에 LoRA.
목표: fake recall 우선, real_hit 하한 유지 (ADS 가중 총점).

예 (Docker/Colab GPU):
  python train_ad_adapt.py \\
    --train-csv /mnt/d/manifests/train.csv \\
    --valid-csv /mnt/d/manifests/valid.csv \\
    --rewrite-prefix 'D:\\=/mnt/d/' \\
    --out model/ad_lora.pt \\
    --max-per-source 200 --voice-per-source 800 --overlays 400 \\
    --phone-frac 0.12 --codec-frac 0.08 --band-frac 0 \\
    --gain-frac 0.1 --noise-frac 0.1 --domain-repeat 1 \\
    --last-n-layers 4 --epochs 3 --batch-size 1 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from ad_lora import (
    enable_last_encoder_layers,
    inject_antideepfake_lora,
    save_ad_adapter,
)
from antideepfake_vf import (
    AUDIO_SAMPLE_RATE,
    FAKE_CLASS_INDEX,
    SEGMENT_SAMPLES,
    load_antideepfake,
)
from learning_data import (
    build_split_rows,
    is_domain_focus_row,
    load_row_audio,
    parse_rewrites,
    row_has_voice,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_AD_DIR = MODEL_DIR / "antideepfake_xlsr_1b"
DEFAULT_OUT = MODEL_DIR / "ad_lora.pt"


class VoiceFakeDataset(Dataset):
    def __init__(self, rows):
        self.rows = [row for row in rows if row_has_voice(row)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio = load_row_audio(row).astype(np.float32)
        if audio.size < SEGMENT_SAMPLES:
            audio = np.pad(audio, (0, SEGMENT_SAMPLES - audio.size))
        else:
            audio = audio[:SEGMENT_SAMPLES]
        label = float(row.get("VOICE_FAKE", 0))
        return torch.from_numpy(audio.copy()), torch.tensor(label, dtype=torch.float32)


def collate_batch(batch):
    waves, labels = zip(*batch)
    return torch.stack(waves, dim=0), torch.stack(labels)


def spoof_logits(model, waves: torch.Tensor) -> torch.Tensor:
    """waves [B, T] → spoof logits [B] (class FAKE_CLASS_INDEX)."""
    out = []
    for wave in waves:
        wav = F.layer_norm(wave, wave.shape).unsqueeze(0)
        emb = model.extract_feat(wav)
        pooled = emb.mean(dim=1)
        logits = model.proj_fc(pooled)
        out.append(logits[0, FAKE_CLASS_INDEX])
    return torch.stack(out, dim=0)


def evaluate_hits(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    pred = (y_prob >= thr).astype(np.float64)
    fake = y_true >= 0.5
    real = ~fake
    return {
        "acc": float((pred == y_true).mean()) if y_true.size else 0.0,
        "fake_hit": float((pred[fake] >= 0.5).mean()) if fake.any() else 0.0,
        "real_hit": float((pred[real] < 0.5).mean()) if real.any() else 0.0,
    }


def selection_score(m: dict) -> float:
    """ADS 가중 대회: fake 우선, real_hit 0.90 하한."""
    if m["real_hit"] < 0.90:
        return 0.35 * m["fake_hit"] + 0.25 * m["real_hit"]
    return 0.75 * m["fake_hit"] + 0.25 * m["real_hit"]


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    pos_weight: float = 2.0,
    checkpoint_fn=None,
    checkpoint_every: int = 0,
):
    train = optimizer is not None
    model.train(train)
    # spoof logit → BCE (DF adapt와 동일 패턴)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    total_loss = 0.0
    total = 0
    all_labels: list[float] = []
    all_probs: list[float] = []
    step = 0

    for waves, labels in tqdm(loader, desc="train" if train else "valid"):
        waves = waves.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = spoof_logits(model, waves)
            loss = loss_fn(logits, labels)
            if train:
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                params = [p for p in model.parameters() if p.requires_grad]
                total_norm = nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(total_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                step += 1
                if (
                    checkpoint_fn is not None
                    and checkpoint_every > 0
                    and step % checkpoint_every == 0
                ):
                    checkpoint_fn(step)
        probs = torch.sigmoid(logits.detach())
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        total_loss += float(loss.item()) * int(labels.numel())
        total += int(labels.numel())

    metrics = evaluate_hits(np.asarray(all_labels), np.asarray(all_probs))
    metrics["loss"] = total_loss / max(total, 1)
    metrics["score"] = selection_score(metrics)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="v3.1: AntiDeepfake LoRA with v2.5-style aug"
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--ad-dir", type=Path, default=DEFAULT_AD_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mode", choices=["lora", "last_layers"], default="lora")
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument("--max-per-source", type=int, default=200)
    parser.add_argument("--voice-per-source", type=int, default=800)
    parser.add_argument("--overlays", type=int, default=400)
    parser.add_argument("--phone-frac", type=float, default=0.12)
    parser.add_argument("--codec-frac", type=float, default=0.08)
    parser.add_argument("--band-frac", type=float, default=0.0)
    parser.add_argument("--gain-frac", type=float, default=0.1)
    parser.add_argument("--noise-frac", type=float, default=0.1)
    parser.add_argument("--channel-fake-share", type=float, default=0.25)
    parser.add_argument("--domain-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--rewrite-prefix", action="append", default=[])
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=1000,
        help="Save interim LoRA every N train steps (0=off). Survives CUDA drops.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rewrites = parse_rewrites(args.rewrite_prefix)
    train_rows = build_split_rows(
        args.train_csv,
        max_per_source=args.max_per_source or None,
        voice_per_source=args.voice_per_source or None,
        overlays=args.overlays,
        overlay_seed=args.seed,
        phone_frac=args.phone_frac,
        codec_frac=args.codec_frac,
        band_frac=args.band_frac,
        gain_frac=args.gain_frac,
        noise_frac=args.noise_frac,
        channel_fake_share=args.channel_fake_share,
        rewrites=rewrites,
    )
    valid_rows = build_split_rows(
        args.valid_csv,
        max_per_source=args.max_per_source or None,
        voice_per_source=args.voice_per_source or None,
        overlays=max(args.overlays // 10, 0),
        overlay_seed=args.seed + 1,
        phone_frac=args.phone_frac,
        codec_frac=args.codec_frac * 0.5,
        band_frac=args.band_frac * 0.5,
        gain_frac=args.gain_frac * 0.5,
        noise_frac=args.noise_frac * 0.5,
        channel_fake_share=args.channel_fake_share,
        rewrites=rewrites,
    )

    train_ds = VoiceFakeDataset(train_rows)
    valid_ds = VoiceFakeDataset(valid_rows)
    if len(train_ds) < 50:
        raise SystemExit(f"too few voice train rows: {len(train_ds)}")

    weights = [
        float(args.domain_repeat) if is_domain_focus_row(row) else 1.0
        for row in train_ds.rows
    ]
    # fake 행 추가 가중 (ADS)
    for i, row in enumerate(train_ds.rows):
        if float(row.get("VOICE_FAKE", 0)) >= 0.5:
            weights[i] *= 1.4

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    print(f"train voice={len(train_ds)} valid={len(valid_ds)}")
    print(
        f"aug phone={args.phone_frac} codec={args.codec_frac} band={args.band_frac} "
        f"gain={args.gain_frac} noise={args.noise_frac} overlays={args.overlays}"
    )

    model = load_antideepfake(args.ad_dir, device)
    if args.mode == "lora":
        info = inject_antideepfake_lora(
            model,
            last_n_layers=args.last_n_layers,
            rank=args.rank,
            alpha=args.alpha,
            adapt_proj_fc=True,
        )
    else:
        info = enable_last_encoder_layers(
            model, last_n_layers=args.last_n_layers, adapt_proj_fc=True
        )
    print(f"adapt mode={args.mode} {info}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    def save_lora(tag: str, valid_m: dict | None = None):
        meta = {
            "mode": "lora",
            "rank": args.rank,
            "alpha": args.alpha,
            "last_n_layers": args.last_n_layers,
            "adapt_proj_fc": True,
            "valid": valid_m or {},
            "sample_rate": AUDIO_SAMPLE_RATE,
            "objective": "fake_priority_real_floor_0.90",
            "tag": tag,
            "aug": {
                "phone_frac": args.phone_frac,
                "codec_frac": args.codec_frac,
                "band_frac": args.band_frac,
                "gain_frac": args.gain_frac,
                "noise_frac": args.noise_frac,
                "overlays": args.overlays,
            },
        }
        save_ad_adapter(args.out, model, meta=meta)
        print(f"saved {args.out} ({tag})")

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        def _mid_ckpt(step: int, ep=epoch):
            if args.mode == "lora":
                save_lora(f"epoch{ep}_step{step}")

        train_m = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            pos_weight=args.pos_weight,
            checkpoint_fn=_mid_ckpt if args.mode == "lora" else None,
            checkpoint_every=args.ckpt_every,
        )
        valid_m = run_epoch(
            model, valid_loader, device, optimizer=None, pos_weight=args.pos_weight
        )
        print(
            f"epoch {epoch}: "
            f"train loss={train_m['loss']:.4f} fake={train_m['fake_hit']:.3f} "
            f"real={train_m['real_hit']:.3f} | "
            f"valid loss={valid_m['loss']:.4f} fake={valid_m['fake_hit']:.3f} "
            f"real={valid_m['real_hit']:.3f} score={valid_m['score']:.3f}"
        )
        if valid_m["score"] < best_score:
            continue
        best_score = valid_m["score"]
        if args.mode == "lora":
            save_lora(f"best_epoch{epoch}_score{best_score:.3f}", valid_m)
        else:
            out = args.out.with_name(args.out.stem + "_last_layers.pt")
            partial = {
                name: parameter.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            torch.save(
                {
                    "adapter": "antideepfake_last_layers",
                    "state_dict": partial,
                    "meta": {
                        "mode": "last_layers",
                        "last_n_layers": args.last_n_layers,
                        "valid": valid_m,
                    },
                },
                out,
            )
            print(f"saved {out}")

    print("done")


if __name__ == "__main__":
    main()
