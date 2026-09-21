#!/usr/bin/env python3
"""DF-Arena 가벼운 도메인 적응 (Colab A100 권장).

SSL(XLS-R 1B)은 고정하고 Conformer 마지막 블록만 LoRA(또는 full)로 학습한다.
학습 분포는 phone / codec / band aug + 한국어·실믹스·TTS 비중을 올린다.

예:
  python train_df_adapt.py \\
    --train-csv data/manifests/train.csv \\
    --valid-csv data/manifests/valid.csv \\
    --out model/df_arena_lora.pt \\
    --max-per-source 200 --voice-per-source 800 --overlays 200 \\
    --phone-frac 0.35 --codec-frac 0.25 --band-frac 0.2 \\
    --domain-repeat 2 --epochs 3 --batch-size 2 --mode lora
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from df_lora import (
    enable_last_blocks,
    inject_conformer_lora,
    save_adapter,
)
from learning_data import (
    AUDIO_SAMPLE_RATE,
    SEGMENT_SAMPLES,
    build_split_rows,
    is_domain_focus_row,
    load_row_audio,
    row_has_voice,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
DEFAULT_OUT = MODEL_DIR / "df_arena_lora.pt"

# 학습(Colab)은 SSL config hub 조회가 필요할 수 있어 기본은 온라인 허용.
# 제출 추론(script.py)만 오프라인 강제.


def load_df_arena(device):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    previous = Path.cwd()
    os.chdir(DF_ARENA_DIR)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(DF_ARENA_DIR),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    finally:
        os.chdir(previous)
    return model.to(device)


class VoiceFakeDataset(Dataset):
    """VOICE_PRESENT인 클립만 → spoof/bonafide (VOICE_FAKE)."""

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
        focus = 1.0 if is_domain_focus_row(row) else 0.0
        return (
            torch.from_numpy(audio.copy()),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(focus, dtype=torch.float32),
        )


def collate_batch(batch):
    waves, labels, focus = zip(*batch)
    return torch.stack(waves, dim=0), torch.stack(labels), torch.stack(focus)


def spoof_index(model):
    return int(model.config.label2id["spoof"])


def run_epoch(model, loader, device, spoof_idx, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for waves, labels, _focus in tqdm(loader, desc="train" if train else "valid"):
        waves = waves.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            # DF_Arena_1B.forward expects 1-D wave; returns (B=1, 2) logits.
            batch_logits = []
            for wave in waves:
                logits = model.backbone(wave)
                if logits.dim() == 1:
                    batch_logits.append(logits[spoof_idx])
                else:
                    batch_logits.append(logits.reshape(-1, logits.shape[-1])[0, spoof_idx])
            logits = torch.stack(batch_logits, dim=0)
            loss = loss_fn(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        probs = torch.sigmoid(logits.detach())
        preds = (probs >= 0.5).float()
        total_correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        total_loss += float(loss.item()) * int(labels.numel())
    return total_loss / max(total, 1), total_correct / max(total, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Light DF-Arena domain adaptation")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mode", choices=["lora", "last_blocks"], default="lora")
    parser.add_argument("--last-n-blocks", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-per-source", type=int, default=200)
    parser.add_argument("--voice-per-source", type=int, default=800)
    parser.add_argument("--overlays", type=int, default=200)
    parser.add_argument("--phone-frac", type=float, default=0.35)
    parser.add_argument("--codec-frac", type=float, default=0.25)
    parser.add_argument("--band-frac", type=float, default=0.2)
    parser.add_argument(
        "--domain-repeat",
        type=int,
        default=2,
        help="phone/Zeroth/TTS/실믹스 행 샘플링 가중치 배수",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--rewrite-prefix",
        action="append",
        default=[],
        help="경로 치환 SRC=DST (Colab Drive 마운트용)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from learning_data import parse_rewrites

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
        rewrites=rewrites,
    )

    train_ds = VoiceFakeDataset(train_rows)
    valid_ds = VoiceFakeDataset(valid_rows)
    if len(train_ds) == 0:
        raise SystemExit("VOICE_PRESENT 학습 행이 없습니다.")

    weights = []
    for row in train_ds.rows:
        w = float(args.domain_repeat) if is_domain_focus_row(row) else 1.0
        weights.append(w)
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

    print(f"train voice clips: {len(train_ds)}  valid: {len(valid_ds)}")
    model = load_df_arena(device)
    if args.mode == "lora":
        info = inject_conformer_lora(
            model,
            last_n_blocks=args.last_n_blocks,
            rank=args.rank,
            alpha=args.alpha,
            adapt_fc5=True,
        )
    else:
        info = enable_last_blocks(
            model, last_n_blocks=args.last_n_blocks, adapt_fc5=True
        )
    print(f"adapt mode={args.mode}  {info}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    spoof_idx = spoof_index(model)

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, device, spoof_idx, optimizer
        )
        valid_loss, valid_acc = run_epoch(
            model, valid_loader, device, spoof_idx, optimizer=None
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.3f} "
            f"valid_loss={valid_loss:.4f} acc={valid_acc:.3f}"
        )
        if valid_acc < best_acc:
            continue
        best_acc = valid_acc
        if args.mode == "lora":
            save_adapter(
                args.out,
                model,
                meta={
                    "mode": "lora",
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "last_n_blocks": args.last_n_blocks,
                    "adapt_fc5": True,
                    "valid_acc": valid_acc,
                    "sample_rate": AUDIO_SAMPLE_RATE,
                },
            )
            print(f"saved LoRA adapter {args.out}")
        else:
            out = args.out.with_name(args.out.stem + "_last_blocks.pt")
            partial = {
                name: parameter.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            torch.save(
                {
                    "adapter": "df_last_blocks",
                    "state_dict": partial,
                    "meta": {
                        "mode": "last_blocks",
                        "last_n_blocks": args.last_n_blocks,
                        "valid_acc": valid_acc,
                    },
                },
                out,
            )
            print(f"saved last-block weights {out}")

    print("done")


if __name__ == "__main__":
    main()
