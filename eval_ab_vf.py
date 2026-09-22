#!/usr/bin/env python3
"""VF A/B: LoRA on/off + gate / asymmetric-gate / fusion.

  python eval_ab_vf.py \\
    --valid-csv /mnt/d/manifests/valid.csv \\
    --rewrite-prefix 'D:\\=/mnt/d/' \\
    --per-class 100 --skip-demucs --device cuda

기본 샘플링은 LB(fake recall) 방향으로 소스 쿼터를 씀
(TTS↑, 전화·실믹스 FP 비중↓). --no-quotas 면 예전 균등 round-robin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from learning_data import parse_rewrites, rewrite_row, row_has_voice


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_CACHE = BASE_DIR / "data" / "cache" / "vf_ab_scores"

# relative weights (normalized to --per-class)
DEFAULT_FAKE_QUOTAS = {
    "Fake_Voice_Only_TTS_ko": 0.9,
    "Music_FakeVocal_Mix": 0.1,
}
DEFAULT_REAL_QUOTAS = {
    "Voice_Only_LibriSpeech": 0.4,
    "Voice_Only_Zeroth_Korean": 0.4,
    "Voice_Only_call_aihub": 0.1,
    "Voice_and_Music_FMA": 0.1,
}

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def gate(p, d):
    p, d = float(p), float(d)
    if p >= d:
        return d + (p - d) * d
    return p + (d - p) * p


def asymmetric_gate(p, d, fake_boost=0.35):
    base = gate(p, d)
    hi = max(float(p), float(d))
    return float(base + fake_boost * (hi - base))


def metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    pred = (y_prob >= thr).astype(np.float64)
    fake = y_true >= 0.5
    real = ~fake
    return {
        "n": int(y_true.size),
        "n_fake": int(fake.sum()),
        "n_real": int(real.sum()),
        "acc": float((pred == y_true).mean()) if y_true.size else 0.0,
        "fake_hit": float((pred[fake] >= 0.5).mean()) if fake.any() else 0.0,
        "real_hit": float((pred[real] < 0.5).mean()) if real.any() else 0.0,
        "fake_mean": float(y_prob[fake].mean()) if fake.any() else 0.0,
        "real_mean": float(y_prob[real].mean()) if real.any() else 0.0,
    }


def cache_key(path_text: str) -> str:
    return hashlib.md5(path_text.encode("utf-8")).hexdigest()[:18] + ".pt"


def diversify_sample(pool, n, rng):
    by = {}
    for row in pool:
        by.setdefault(row.get("source_folder", "?"), []).append(row)
    for bucket in by.values():
        rng.shuffle(bucket)
    keys = list(by.keys())
    rng.shuffle(keys)
    out = []
    i = 0
    while len(out) < n and keys:
        key = keys[i % len(keys)]
        if by[key]:
            out.append(by[key].pop())
            i += 1
        else:
            keys.remove(key)
            if not keys:
                break
            i = 0
    return out


def parse_quotas(text: str | None) -> dict[str, float] | None:
    """Parse 'Folder=0.9,Other=0.1'. None → use defaults; empty/'off' → no quota."""
    if text is None:
        return None
    raw = text.strip()
    if raw.lower() in ("", "none", "off"):
        return {}
    out: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"quota entry must be name=weight, got {part!r}")
        name, weight = part.split("=", 1)
        out[name.strip()] = float(weight.strip())
    return out


def _alloc_counts(weights: dict[str, float], n: int) -> dict[str, int]:
    total = sum(weights.values())
    if total <= 0 or n <= 0:
        return {k: 0 for k in weights}
    raw = {k: (w / total) * n for k, w in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    rem = n - sum(counts.values())
    for key, _ in sorted(
        raw.items(), key=lambda kv: (kv[1] - int(kv[1]), kv[0]), reverse=True
    ):
        if rem <= 0:
            break
        counts[key] += 1
        rem -= 1
    return counts


def _match_folders(quota_key: str, folders: list[str]) -> list[str]:
    exact = [f for f in folders if f == quota_key]
    if exact:
        return exact
    return [f for f in folders if quota_key in f or f in quota_key]


def quota_sample(pool, n, quotas: dict[str, float], rng):
    """Sample up to n rows using relative source quotas; shortfall → diversify."""
    if not quotas:
        return diversify_sample(pool, n, rng)

    by: dict[str, list] = {}
    for row in pool:
        by.setdefault(row.get("source_folder", "?"), []).append(row)
    for bucket in by.values():
        rng.shuffle(bucket)

    folders = list(by.keys())
    counts = _alloc_counts(quotas, n)
    out = []
    for key, want in counts.items():
        matched = _match_folders(key, folders)
        if not matched or want <= 0:
            continue
        taken = 0
        while taken < want:
            progressed = False
            for folder in matched:
                if by[folder] and taken < want:
                    out.append(by[folder].pop())
                    taken += 1
                    progressed = True
            if not progressed:
                break

    if len(out) < n:
        leftover = [row for bucket in by.values() for row in bucket]
        out.extend(diversify_sample(leftover, n - len(out), rng))

    rng.shuffle(out)
    return out[:n]


def _print_composition(tag: str, rows):
    counts = Counter(row.get("source_folder", "?") for row in rows)
    parts = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    print(f"  {tag} n={len(rows)} | {parts}")


def sample_voice_rows(
    csv_path: Path,
    per_class: int,
    rewrites,
    seed: int,
    fake_quotas: dict[str, float] | None = None,
    real_quotas: dict[str, float] | None = None,
):
    import csv

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [rewrite_row(row, rewrites) for row in csv.DictReader(handle)]
    rows = [row for row in rows if row_has_voice(row)]
    fake = [row for row in rows if int(row.get("VOICE_FAKE", 0)) == 1]
    real = [row for row in rows if int(row.get("VOICE_FAKE", 0)) == 0]
    rng = random.Random(seed)

    if fake_quotas is None:
        fake_quotas = dict(DEFAULT_FAKE_QUOTAS)
    if real_quotas is None:
        real_quotas = dict(DEFAULT_REAL_QUOTAS)

    fake_rows = quota_sample(fake, per_class, fake_quotas, rng)
    real_rows = quota_sample(real, per_class, real_quotas, rng)
    mode = "quota" if (fake_quotas or real_quotas) else "diversify"
    print(f"sample mode={mode} per_class={per_class}")
    _print_composition("fake", fake_rows)
    _print_composition("real", real_rows)
    return fake_rows + real_rows


def score_and_cache(rows, cache_dir: Path, args, device):
    import script as infer

    cache_dir.mkdir(parents=True, exist_ok=True)
    unique = []
    seen = set()
    for row in rows:
        path = Path(str(row["path"]))
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if not (cache_dir / cache_key(key)).is_file():
            unique.append(path)

    if not unique:
        print(f"A/B cache complete ({len(seen)} files)")
        return

    print(f"A/B scoring {len(unique)} files")
    predictor = infer.FourHeadPredictor(
        device,
        args.mf_ckpt,
        args.vf_ckpt,
        vf_device=device,
        df_lora=None,
        vf_fusion=None,
    )
    presence = predictor.run_presence_heads(unique)
    panns_vf, _ = predictor._run_panns_fakes(unique)

    stems = {}
    if args.skip_demucs:
        skipped = 0
        for audio_path in unique:
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
            print(f"skip-demucs: {skipped}/{len(unique)} files unreadable")
    else:
        raw = predictor._estimate_all_stems(unique)
        for stem, (voice, _music) in raw.items():
            stems[stem] = voice

    def run_df(lora_path, desc):
        scores = {}
        model, fake_idx = infer.load_df_arena_model(device, lora_path=lora_path)
        for audio_path in tqdm(unique, desc=desc):
            try:
                mix = infer.load_audio(audio_path)
                voice = stems.get(audio_path.stem)
                if voice is None or getattr(voice, "size", 0) < 8:
                    voice = np.ones_like(mix, dtype=np.float32)
                scores[audio_path.stem] = infer.predict_voice_fake(
                    model, fake_idx, mix, voice, device
                )
            except Exception:
                scores[audio_path.stem] = 0.0
        del model
        infer.release_cuda()
        return scores

    df_frozen = run_df(None, "DF frozen")
    lora = args.df_lora if args.df_lora.is_file() else None
    df_lora_scores = run_df(lora, "DF LoRA")

    for audio_path in unique:
        stem = audio_path.stem
        torch.save(
            {
                "path": audio_path.as_posix(),
                "vp": float(presence.get(stem, {}).get("VOICE_PRESENT_PROB", 0.0)),
                "panns_vf": float(panns_vf.get(stem, 0.0)),
                "df_frozen": float(df_frozen.get(stem, 0.0)),
                "df_lora": float(df_lora_scores.get(stem, 0.0)),
            },
            cache_dir / cache_key(audio_path.as_posix()),
        )


def load_payloads(rows, cache_dir: Path):
    out = []
    for row in rows:
        path = Path(str(row["path"])).as_posix()
        ckpt = cache_dir / cache_key(path)
        if not ckpt.is_file():
            continue
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        payload["label"] = float(row.get("VOICE_FAKE", 0))
        payload["source_folder"] = row.get("source_folder", "")
        out.append(payload)
    return out


def print_table(payloads, fusion_model=None):
    labels = [p["label"] for p in payloads]
    variants = {
        "panns_only": [p["panns_vf"] for p in payloads],
        "df_frozen": [p["df_frozen"] for p in payloads],
        "df_lora": [p["df_lora"] for p in payloads],
        "gate+frozen": [gate(p["panns_vf"], p["df_frozen"]) for p in payloads],
        "gate+lora (v2.3)": [gate(p["panns_vf"], p["df_lora"]) for p in payloads],
        "asym_gate+frozen": [
            asymmetric_gate(p["panns_vf"], p["df_frozen"]) for p in payloads
        ],
        "asym_gate+lora": [
            asymmetric_gate(p["panns_vf"], p["df_lora"]) for p in payloads
        ],
    }
    if fusion_model is not None:
        variants["fusion+lora"] = [
            fusion_model.predict_proba(p["panns_vf"], p["df_lora"], p.get("vp", 0.0))
            for p in payloads
        ]
        variants["fusion+frozen"] = [
            fusion_model.predict_proba(
                p["panns_vf"], p["df_frozen"], p.get("vp", 0.0)
            )
            for p in payloads
        ]

    print(
        f"\n=== A/B n={len(payloads)} "
        f"fake={sum(y >= 0.5 for y in labels)} "
        f"real={sum(y < 0.5 for y in labels)} ==="
    )
    print(
        f"{'variant':<22} {'fake_hit':>8} {'real_hit':>8} "
        f"{'acc':>7} {'f_mean':>7} {'r_mean':>7}"
    )
    table = {}
    for name, probs in variants.items():
        m = metrics(labels, probs)
        table[name] = m
        print(
            f"{name:<22} {m['fake_hit']:8.3f} {m['real_hit']:8.3f} "
            f"{m['acc']:7.3f} {m['fake_mean']:7.3f} {m['real_mean']:7.3f}"
        )
    return table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--vf-ckpt", type=Path, default=MODEL_DIR / "vf_head.pt")
    parser.add_argument("--mf-ckpt", type=Path, default=MODEL_DIR / "mf_head.pt")
    parser.add_argument("--df-lora", type=Path, default=MODEL_DIR / "df_arena_lora.pt")
    parser.add_argument("--vf-fusion", type=Path, default=MODEL_DIR / "vf_fusion.pt")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--rewrite-prefix", action="append", default=[])
    parser.add_argument("--skip-demucs", action="store_true")
    parser.add_argument(
        "--fake-quotas",
        default=None,
        help=(
            "fake source weights, e.g. "
            "Fake_Voice_Only_TTS_ko=0.9,Music_FakeVocal_Mix=0.1 "
            "(default LB-like; 'off' = diversify)"
        ),
    )
    parser.add_argument(
        "--real-quotas",
        default=None,
        help=(
            "real source weights, e.g. "
            "Voice_Only_LibriSpeech=0.4,Voice_Only_Zeroth_Korean=0.4,"
            "Voice_Only_call_aihub=0.1,Voice_and_Music_FMA=0.1 "
            "(default LB-like; 'off' = diversify)"
        ),
    )
    parser.add_argument(
        "--no-quotas",
        action="store_true",
        help="use equal-per-folder diversify for both classes",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=BASE_DIR / "output" / "ab_vf_metrics.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not args.mf_ckpt.is_file():
        args.mf_ckpt = args.vf_ckpt

    device = torch.device(args.device)
    rewrites = parse_rewrites(args.rewrite_prefix)
    if args.no_quotas:
        fake_quotas: dict[str, float] = {}
        real_quotas: dict[str, float] = {}
    else:
        parsed_fake = parse_quotas(args.fake_quotas)
        parsed_real = parse_quotas(args.real_quotas)
        fake_quotas = (
            dict(DEFAULT_FAKE_QUOTAS) if parsed_fake is None else parsed_fake
        )
        real_quotas = (
            dict(DEFAULT_REAL_QUOTAS) if parsed_real is None else parsed_real
        )
    rows = sample_voice_rows(
        args.valid_csv,
        args.per_class,
        rewrites,
        args.seed,
        fake_quotas=fake_quotas,
        real_quotas=real_quotas,
    )
    print(f"sampled voice rows: {len(rows)}")
    score_and_cache(rows, args.cache_dir, args, device)
    payloads = load_payloads(rows, args.cache_dir)
    if len(payloads) < 20:
        raise SystemExit(f"payloads too few: {len(payloads)}")

    fusion = None
    if args.vf_fusion.is_file():
        from heads.vf_fusion import VoiceFakeFusion

        blob = torch.load(args.vf_fusion, map_location="cpu", weights_only=False)
        fusion = VoiceFakeFusion(
            in_dim=int(blob.get("in_dim", 5)),
            hidden=int(blob.get("hidden", 32)),
        )
        fusion.load_state_dict(blob["state_dict"])
        fusion = fusion.to(device).eval()
        print(f"loaded fusion {args.vf_fusion}")

    table = print_table(payloads, fusion)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
