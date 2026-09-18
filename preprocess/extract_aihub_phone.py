#!/usr/bin/env python3
"""AI-Hub 저음질 전화망 Validation 원천에서 학습용 8 kHz 클립만 뽑는다.

원본 zip은 그대로 두고, 4초 이상·세션당 1개·최대 N개만 푼다.

  python -m preprocess.extract_aihub_phone
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path


LABEL_DIR = Path(r"D:\007.저음질 전화망 음성인식 데이터\01.데이터\2.Validation\라벨링데이터_230316")
SOURCE_DIR = Path(r"D:\007.저음질 전화망 음성인식 데이터\01.데이터\2.Validation\원천데이터_230316")
OUT_DIR = Path(r"D:\Voice_Only_Phone")


def domain_of(audio_path: str) -> str:
    return audio_path.split("/", 1)[0]


def session_of(audio_path: str) -> str:
    return str(Path(audio_path).parent).replace("\\", "/")


def collect_candidates(label_dir: Path, min_sec: float, max_sec: float):
    grouped = defaultdict(list)
    for zpath in sorted(label_dir.glob("VL_D*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                payload = json.loads(zf.read(name))
                info = payload["dataSet"]["typeInfo"]
                speakers = {
                    str(speaker["id"]): speaker.get("telephone_network", "")
                    for speaker in info.get("speakers", [])
                }
                for dialog in payload["dataSet"]["dialogs"]:
                    net = speakers.get(str(dialog.get("speaker", "")), "")
                    if net != "8k":
                        continue
                    duration = float(dialog.get("duration") or 0)
                    if duration < min_sec or duration > max_sec:
                        continue
                    audio_path = str(dialog["audioPath"]).replace("\\", "/")
                    grouped[session_of(audio_path)].append((duration, audio_path))
    return grouped


def pick_clips(grouped, limit: int, seed: int):
    sessions = []
    for session, clips in grouped.items():
        clips.sort(reverse=True)
        duration, audio_path = clips[0]
        sessions.append((domain_of(audio_path), session, audio_path, duration))
    rng = random.Random(seed)
    rng.shuffle(sessions)

    by_domain = defaultdict(list)
    for item in sessions:
        by_domain[item[0]].append(item)

    domains = sorted(by_domain)
    per_domain = max(limit // max(len(domains), 1), 1)
    chosen = []
    leftover = []
    for domain in domains:
        pool = by_domain[domain]
        chosen.extend(pool[:per_domain])
        leftover.extend(pool[per_domain:])
    rng.shuffle(leftover)
    if len(chosen) < limit:
        chosen.extend(leftover[: limit - len(chosen)])
    return chosen[:limit]


def extract_clips(clips, source_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = {}
    written = 0
    missing = 0
    for _domain, _session, audio_path, _duration in clips:
        zip_name = f"VS_{domain_of(audio_path)}.zip"
        zpath = source_dir / zip_name
        if zip_name not in zips:
            zips[zip_name] = zipfile.ZipFile(zpath)
        zf = zips[zip_name]
        try:
            info = zf.getinfo(audio_path)
        except KeyError:
            missing += 1
            continue
        target = out_dir / Path(audio_path)
        if target.is_file() and target.stat().st_size == info.file_size:
            written += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(audio_path))
        written += 1
    for zf in zips.values():
        zf.close()
    return written, missing


def parse_args():
    parser = argparse.ArgumentParser(description="Extract a phone-channel training subset.")
    parser.add_argument("--label-dir", type=Path, default=LABEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--min-sec", type=float, default=4.0)
    parser.add_argument("--max-sec", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.source_dir.is_dir():
        raise SystemExit(f"원천 폴더가 없습니다: {args.source_dir}")
    grouped = collect_candidates(args.label_dir, args.min_sec, args.max_sec)
    clips = pick_clips(grouped, args.limit, args.seed)
    print(f"세션 {len(grouped)} / 뽑을 클립 {len(clips)}")
    written, missing = extract_clips(clips, args.source_dir, args.out)
    print(f"저장 {written} -> {args.out}")
    if missing:
        print(f"zip에 없는 경로 {missing}")


if __name__ == "__main__":
    main()
