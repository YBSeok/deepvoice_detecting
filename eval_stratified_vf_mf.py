#!/usr/bin/env python3
"""valid 층화 프로브: voice-only fake vs music-only fake.

DF vs AntiDeepfake VF + MF hit/평균 점수로 병목·백본 선택.

  python eval_stratified_vf_mf.py \
    --valid-csv /mnt/d/manifests/valid.csv \
    --rewrite-prefix 'D:\\=/mnt/d/' \
    --per-class 80 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from learning_data import parse_rewrites, rewrite_row
import script as infer
from antideepfake_vf import load_antideepfake, predict_voice_fake_masked
from df_arena_vf import load_df_arena, predict_voice_fake_df


def stratum_of(row: dict) -> str | None:
    vp = int(row.get("VOICE_PRESENT", 0))
    mp = int(row.get("MUSIC_PRESENT", 0))
    vf = int(row.get("VOICE_FAKE", 0))
    mf = int(row.get("MUSIC_FAKE", 0))
    if vp == 1 and mp == 0 and vf == 1:
        return "voice_only_fake"
    if mp == 1 and vp == 0 and mf == 1:
        return "music_only_fake"
    if vp == 1 and mp == 0 and vf == 0:
        return "voice_only_real"
    if mp == 1 and vp == 0 and mf == 0:
        return "music_only_real"
    return None


def sample_rows(csv_path: Path, rewrites, per_class: int, seed: int):
    buckets: dict[str, list] = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row = rewrite_row(row, rewrites)
            kind = stratum_of(row)
            if kind is None:
                continue
            path = Path(row["path"])
            if not path.is_file() or path.stat().st_size < 1000:
                continue
            buckets[kind].append(row)

    rng = random.Random(seed)
    picked = []
    for kind, rows in buckets.items():
        rng.shuffle(rows)
        take = rows[:per_class]
        print(f"stratum {kind}: available={len(rows)} use={len(take)}")
        for r in take:
            r = dict(r)
            r["_stratum"] = kind
            picked.append(r)
    return picked


def summarize(name: str, values: list[float], labels: list[int], thr: float = 0.5):
    if not values:
        print(f"  {name}: empty")
        return
    arr = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int32)
    mean = float(arr.mean())
    if lab.size and set(lab.tolist()) <= {0, 1} and lab.max() >= 0:
        # for fake strata labels are 1; for real 0 — hit = correct side of thr
        pred = (arr >= thr).astype(np.int32)
        hit = float((pred == lab).mean())
        # also report mean among positives / negatives if mixed — here usually pure
        print(
            f"  {name}: n={len(arr)} mean={mean:.4f} "
            f"hit@{thr:.1f}={hit:.3f} "
            f"p50={np.median(arr):.4f} "
            f"min={arr.min():.4f} max={arr.max():.4f}"
        )
    else:
        print(f"  {name}: n={len(arr)} mean={mean:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--rewrite-prefix", action="append", default=[])
    parser.add_argument("--per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--skip-df", action="store_true")
    parser.add_argument("--skip-ad", action="store_true")
    args = parser.parse_args()

    rewrites = parse_rewrites(args.rewrite_prefix)
    rows = sample_rows(args.valid_csv, rewrites, args.per_class, args.seed)
    device = infer.select_device(args.device)

    # stems once
    paths = [Path(r["path"]) for r in rows]
    stems = {}
    htd = infer.load_htdemucs_model()
    for p in tqdm(paths, desc="Stems"):
        try:
            stems[p.stem] = infer.estimate_source_stems(p, htd, device)
        except Exception:
            stems[p.stem] = (
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            )
    del htd
    infer.release_cuda()

    # MF
    panns, _, _ = infer.load_panns_model(device)
    mf_head = infer.load_task_head(infer.DEFAULT_MF_CKPT, device)
    mf_scores = {}
    for r in tqdm(rows, desc="MF"):
        p = Path(r["path"])
        try:
            mix = infer.load_audio(p)
            mf_scores[p.stem] = infer.predict_music_fake(panns, mf_head, mix, device)
        except Exception:
            mf_scores[p.stem] = 0.0
    del panns, mf_head
    infer.release_cuda()

    ad_scores = {}
    if not args.skip_ad:
        ad = load_antideepfake(infer.ANTIDEFP_DIR, device)
        for r in tqdm(rows, desc="VF AntiDeepfake"):
            p = Path(r["path"])
            try:
                mix = infer.load_audio(p)
                vs, _ = stems[p.stem]
                ad_scores[p.stem] = predict_voice_fake_masked(ad, mix, vs, device)
            except Exception:
                ad_scores[p.stem] = 0.0
        del ad
        infer.release_cuda()

    df_scores = {}
    if not args.skip_df:
        df, fake_idx = load_df_arena(device, infer.DF_ARENA_DIR)
        for r in tqdm(rows, desc="VF DF-Arena"):
            p = Path(r["path"])
            try:
                mix = infer.load_audio(p)
                vs, _ = stems[p.stem]
                df_scores[p.stem] = predict_voice_fake_df(
                    df, fake_idx, mix, vs, device
                )
            except Exception:
                df_scores[p.stem] = 0.0
        del df
        infer.release_cuda()

    # report per stratum
    by = defaultdict(list)
    for r in rows:
        by[r["_stratum"]].append(r)

    print("\n=== Stratified results (thr=0.5) ===")
    for kind in [
        "voice_only_fake",
        "voice_only_real",
        "music_only_fake",
        "music_only_real",
    ]:
        group = by.get(kind, [])
        if not group:
            continue
        print(f"\n[{kind}] n={len(group)}")
        stems_ids = [Path(r["path"]).stem for r in group]

        if kind.startswith("voice_only"):
            lab = 1 if "fake" in kind else 0
            if ad_scores:
                summarize(
                    "VF AntiDeepfake",
                    [ad_scores[s] for s in stems_ids],
                    [lab] * len(stems_ids),
                )
            if df_scores:
                summarize(
                    "VF DF-Arena",
                    [df_scores[s] for s in stems_ids],
                    [lab] * len(stems_ids),
                )
            if ad_scores and df_scores:
                mx = [max(ad_scores[s], df_scores[s]) for s in stems_ids]
                summarize("VF max(AD,DF)", mx, [lab] * len(stems_ids))

        if kind.startswith("music_only"):
            lab = 1 if "fake" in kind else 0
            summarize(
                "MF PANNs-head",
                [mf_scores[s] for s in stems_ids],
                [lab] * len(stems_ids),
            )
            # VF on music-only should be low ideally
            if ad_scores:
                summarize(
                    "VF AD (should be low on music-only)",
                    [ad_scores[s] for s in stems_ids],
                    [0] * len(stems_ids),
                )
            if df_scores:
                summarize(
                    "VF DF (should be low on music-only)",
                    [df_scores[s] for s in stems_ids],
                    [0] * len(stems_ids),
                )

    print("\nDONE_STRATIFIED")


if __name__ == "__main__":
    main()
