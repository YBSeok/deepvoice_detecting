#!/usr/bin/env python3
"""학습 VF fusion (비대칭: fake recall 우선, real FP 하한 유지).

  python train_fusion.py \\
    --train-csv /mnt/d/manifests/train.csv \\
    --valid-csv /mnt/d/manifests/valid.csv \\
    --rewrite-prefix 'D:\\=/mnt/d/' \\
    --df-lora model/df_arena_lora.pt \\
    --max-per-source 100 --voice-per-source 250 --overlays 80 \\
    --skip-demucs --epochs 40 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from heads.vf_fusion import FUSION_FEATURE_DIM, VoiceFakeFusion, fusion_features
from learning_data import (
    build_split_rows,
    is_domain_focus_row,
    parse_rewrites,
    row_has_voice,
    strip_aug_prefix,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_OUT = MODEL_DIR / "vf_fusion.pt"
DEFAULT_CACHE = BASE_DIR / "data" / "cache" / "vf_fusion_scores"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def gate(p, d):
    p, d = float(p), float(d)
    if p >= d:
        return d + (p - d) * d
    return p + (d - p) * p


def score_key(path_text: str) -> str:
    return hashlib.md5(str(path_text).encode("utf-8")).hexdigest()[:16] + ".pt"


def sample_weight(row, label: float) -> float:
    weight = 1.0
    source = str(row.get("source_folder", "")).lower()
    if label >= 0.5:
        weight *= 2.2
        if any(tok in source for tok in ("tts", "fake_voice", "mlaad")):
            weight *= 1.3
        if "mix" in source or str(row.get("rule", "")).startswith("overlay"):
            weight *= 1.2
    else:
        if is_domain_focus_row(row) or "phone" in source or "call" in source:
            weight *= 1.4
    return weight


def evaluate(y_true, y_prob, thr=0.5):
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


def selection_score(m):
    if m["real_hit"] < 0.88:
        return m["fake_hit"] * 0.3 + m["real_hit"] * 0.2
    return 0.7 * m["fake_hit"] + 0.3 * m["real_hit"]


def bare_path(row):
    _kind, bare = strip_aug_prefix(row.get("path", ""))
    return bare


def unique_bare_rows(rows):
    out = {}
    for row in rows:
        path = bare_path(row)
        if str(path).startswith("overlay::"):
            continue
        if path not in out:
            clone = dict(row)
            clone["path"] = path
            out[path] = clone
    return list(out.values())


def cache_scores(rows, cache_dir: Path, args, device):
    import script as infer

    cache_dir.mkdir(parents=True, exist_ok=True)
    uniq, seen = [], set()
    for row in rows:
        path = Path(str(row["path"]))
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if not (cache_dir / score_key(key)).is_file():
            uniq.append(path)

    missing = [p for p in uniq if not p.is_file()]
    if missing:
        print(f"warning: {len(missing)} audio paths missing on disk (first: {missing[0]})")

    if not uniq:
        print(f"fusion cache complete ({len(seen)} files known)")
        return

    print(f"caching fusion scores for {len(uniq)} files")
    lora = args.df_lora if args.df_lora.is_file() else None
    predictor = infer.FourHeadPredictor(
        device,
        args.mf_ckpt,
        args.vf_ckpt,
        vf_device=device,
        df_lora=lora,
        vf_fusion=None,
    )
    presence = predictor.run_presence_heads(uniq)
    panns_vf, _ = predictor._run_panns_fakes(uniq)

    if args.skip_demucs:
        # 균일 마스크만 필요. 깨진/미존재 파일은 DF 루프에서 0점으로 떨어지게 둔다.
        stems = {}
        skipped = 0
        for audio_path in uniq:
            try:
                if not audio_path.is_file() or audio_path.stat().st_size == 0:
                    raise FileNotFoundError(audio_path)
                mix = infer.load_audio(audio_path)
                stems[audio_path.stem] = np.ones_like(mix, dtype=np.float32)
            except Exception as exc:
                skipped += 1
                stems[audio_path.stem] = np.ones(1, dtype=np.float32)
                print(f"skip demucs-mask {audio_path}: {exc}")
        if skipped:
            print(f"skip-demucs: {skipped}/{len(uniq)} files unreadable")
    else:
        raw = predictor._estimate_all_stems(uniq)
        stems = {stem: voice for stem, (voice, _) in raw.items()}

    df_scores = {}
    model, fake_idx = infer.load_df_arena_model(device, lora_path=lora)
    for audio_path in tqdm(uniq, desc="DF VF"):
        try:
            mix = infer.load_audio(audio_path)
            voice = stems.get(audio_path.stem)
            if voice is None or getattr(voice, "size", 0) < 8:
                voice = np.ones_like(mix, dtype=np.float32)
            df_scores[audio_path.stem] = infer.predict_voice_fake(
                model, fake_idx, mix, voice, device
            )
        except Exception:
            df_scores[audio_path.stem] = 0.0
    del model
    infer.release_cuda()

    for audio_path in uniq:
        stem = audio_path.stem
        torch.save(
            {
                "path": audio_path.as_posix(),
                "vp": float(presence.get(stem, {}).get("VOICE_PRESENT_PROB", 0.0)),
                "panns_vf": float(panns_vf.get(stem, 0.0)),
                "df_vf": float(df_scores.get(stem, 0.0)),
            },
            cache_dir / score_key(audio_path.as_posix()),
        )


def expand_payloads(rows, cache_dir: Path):
    payloads = []
    for row in rows:
        if str(row.get("path", "")).startswith("overlay::"):
            continue
        bare = bare_path(row)
        bare_norm = str(bare).replace("\\", "/")
        ckpt = cache_dir / score_key(bare_norm)
        if not ckpt.is_file():
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        label = float(row.get("VOICE_FAKE", 0))
        payloads.append(
            {
                **blob,
                "label": label,
                "weight": sample_weight(row, label),
                "source_folder": row.get("source_folder", ""),
            }
        )
    return payloads


def train_fusion(train_p, valid_p, epochs, lr, device):
    train_x = torch.stack(
        [fusion_features(p["panns_vf"], p["df_vf"], p.get("vp", 0.0)) for p in train_p]
    )
    train_y = torch.tensor([p["label"] for p in train_p], dtype=torch.float32)
    train_w = torch.tensor([p["weight"] for p in train_p], dtype=torch.float32)
    valid_x = torch.stack(
        [fusion_features(p["panns_vf"], p["df_vf"], p.get("vp", 0.0)) for p in valid_p]
    )
    valid_y = torch.tensor([p["label"] for p in valid_p], dtype=torch.float32)

    model = VoiceFakeFusion().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor([2.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    loader = DataLoader(
        TensorDataset(train_x, train_y, train_w), batch_size=256, shuffle=True
    )
    best_state, best_score, history = None, -1.0, []

    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            logits = model(xb)
            loss = (loss_fn(logits, yb) * wb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            count += xb.size(0)

        model.eval()
        with torch.inference_mode():
            logits = model(valid_x.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
        m = evaluate(valid_y.numpy(), probs)
        score = selection_score(m)
        history.append(
            {"epoch": epoch, "loss": total / max(count, 1), **m, "score": score}
        )
        print(
            f"epoch {epoch}: loss={total / max(count, 1):.4f} "
            f"fake_hit={m['fake_hit']:.3f} real_hit={m['real_hit']:.3f} "
            f"score={score:.3f}"
        )
        if score >= best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), history


def compare(valid_p, model):
    labels = [p["label"] for p in valid_p]
    panns = [p["panns_vf"] for p in valid_p]
    dfs = [p["df_vf"] for p in valid_p]
    gates = [gate(a, b) for a, b in zip(panns, dfs)]
    fused = [
        model.predict_proba(p["panns_vf"], p["df_vf"], p.get("vp", 0.0))
        for p in valid_p
    ]
    print("valid gate       :", evaluate(labels, gates))
    print("valid fusion     :", evaluate(labels, fused))
    print("valid panns-only :", evaluate(labels, panns))
    print("valid df-only    :", evaluate(labels, dfs))


def parse_args():
    parser = argparse.ArgumentParser(description="Train asymmetric VF fusion")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--vf-ckpt", type=Path, default=MODEL_DIR / "vf_head.pt")
    parser.add_argument("--mf-ckpt", type=Path, default=MODEL_DIR / "mf_head.pt")
    parser.add_argument("--df-lora", type=Path, default=MODEL_DIR / "df_arena_lora.pt")
    parser.add_argument("--max-per-source", type=int, default=100)
    parser.add_argument("--voice-per-source", type=int, default=250)
    parser.add_argument("--overlays", type=int, default=80)
    parser.add_argument("--phone-frac", type=float, default=0.12)
    parser.add_argument("--codec-frac", type=float, default=0.08)
    parser.add_argument("--band-frac", type=float, default=0.0)
    parser.add_argument("--gain-frac", type=float, default=0.1)
    parser.add_argument("--noise-frac", type=float, default=0.1)
    parser.add_argument("--channel-fake-share", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--rewrite-prefix", action="append", default=[])
    parser.add_argument("--skip-demucs", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not args.vf_ckpt.is_file():
        raise FileNotFoundError(args.vf_ckpt)
    if not args.mf_ckpt.is_file():
        args.mf_ckpt = args.vf_ckpt

    device = torch.device(args.device)
    rewrites = parse_rewrites(args.rewrite_prefix)

    train_rows = [
        r
        for r in build_split_rows(
            args.train_csv,
            max_per_source=args.max_per_source,
            voice_per_source=args.voice_per_source,
            overlays=args.overlays,
            phone_frac=args.phone_frac,
            codec_frac=args.codec_frac,
            band_frac=args.band_frac,
            gain_frac=args.gain_frac,
            noise_frac=args.noise_frac,
            channel_fake_share=args.channel_fake_share,
            rewrites=rewrites,
        )
        if row_has_voice(r)
    ]
    valid_rows = [
        r
        for r in build_split_rows(
            args.valid_csv,
            max_per_source=args.max_per_source,
            voice_per_source=args.voice_per_source,
            overlays=max(args.overlays // 4, 1),
            phone_frac=args.phone_frac,
            codec_frac=args.codec_frac,
            band_frac=args.band_frac,
            gain_frac=args.gain_frac,
            noise_frac=args.noise_frac,
            channel_fake_share=args.channel_fake_share,
            rewrites=rewrites,
            overlay_seed=99,
        )
        if row_has_voice(r)
    ]
    print(f"voice rows train={len(train_rows)} valid={len(valid_rows)}")

    train_bare = unique_bare_rows(train_rows)
    valid_bare = unique_bare_rows(valid_rows)
    cache_scores(train_bare + valid_bare, args.cache_dir, args, device)

    train_p = expand_payloads(train_rows, args.cache_dir)
    valid_p = expand_payloads(valid_rows, args.cache_dir)
    print(f"payloads train={len(train_p)} valid={len(valid_p)}")
    if len(train_p) < 50 or len(valid_p) < 20:
        raise SystemExit("cached payloads too few")

    if args.cache_only:
        print("cache-only done")
        return

    model, history = train_fusion(train_p, valid_p, args.epochs, args.lr, device)
    compare(valid_p, model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": "vf_fusion",
            "state_dict": model.state_dict(),
            "in_dim": FUSION_FEATURE_DIM,
            "hidden": int(getattr(model, "hidden", 32)),
            "meta": {
                "df_lora": str(args.df_lora),
                "vf_ckpt": str(args.vf_ckpt),
                "train_n": len(train_p),
                "valid_n": len(valid_p),
                "history_tail": history[-5:],
                "objective": "asymmetric_fake_priority_real_floor_0.88",
            },
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
