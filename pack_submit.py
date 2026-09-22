#!/usr/bin/env python3
"""v2.4 제출용 submit.zip 생성.

포함: script.py, df_lora.py, requirements.txt, heads/, model/
      (df_arena_lora.pt + vf_fusion.pt = fusion+lora)
제외: 학습 로그, data/cache, manifests, 노트북

  python pack_submit.py --out submit.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


BASE = Path(__file__).resolve().parent

# 코드
CODE_FILES = [
    "script.py",
    "df_lora.py",
    "requirements.txt",
    "heads/__init__.py",
    "heads/music_fake.py",
    "heads/vf_fusion.py",
]

# model 트리에서 넣을 확장자 / 이름
MODEL_KEEP_SUFFIXES = {".pt", ".pth", ".th", ".bin", ".json", ".yaml", ".txt", ".csv", ".py"}
MODEL_SKIP_NAMES = set()
MODEL_SKIP_DIR_PARTS = {".cache", "__pycache__", ".git"}


def should_keep_model_file(path: Path, model_root: Path) -> bool:
    rel = path.relative_to(model_root)
    if any(part in MODEL_SKIP_DIR_PARTS for part in rel.parts):
        return False
    if path.name in MODEL_SKIP_NAMES:
        return False
    if path.suffix.lower() not in MODEL_KEEP_SUFFIXES and path.name not in {
        ".gitattributes",
        "LICENSE.txt",
        "README.md",
    }:
        # 확장자 없는 건 스킵, 단 LICENSE/README 허용
        if path.suffix == "" and path.name not in {"LICENSE", "LICENSE.txt"}:
            return False
    return True


def required_files():
    missing = []
    for rel in CODE_FILES:
        if not (BASE / rel).is_file():
            missing.append(rel)
    must = [
        "model/df_arena_lora.pt",
        "model/vf_fusion.pt",
        "model/vf_head.pt",
        "model/mf_head.pt",
        "model/panns/Cnn14_mAP=0.431.pth",
        "model/htdemucs/955717e8-8726e21a.th",
        "model/df_arena_1b/pytorch_model.bin",
        "model/df_arena_1b/config.json",
    ]
    for rel in must:
        if not (BASE / rel).is_file():
            missing.append(rel)
    return missing


def pack(out_path: Path, dry_run: bool = False):
    missing = required_files()
    if missing:
        raise SystemExit("제출에 필요한 파일이 없습니다:\n  - " + "\n  - ".join(missing))

    entries = []
    for rel in CODE_FILES:
        entries.append((BASE / rel, rel.replace("\\", "/")))

    model_root = BASE / "model"
    for path in model_root.rglob("*"):
        if not path.is_file():
            continue
        if not should_keep_model_file(path, model_root):
            continue
        arc = path.relative_to(BASE).as_posix()
        entries.append((path, arc))

    total = sum(p.stat().st_size for p, _ in entries)
    print(f"files: {len(entries)}")
    print(f"raw bytes: {total / 1e9:.2f} GB")
    for path, arc in entries:
        if path.stat().st_size > 50_000_000:
            print(f"  {arc}: {path.stat().st_size / 1e6:.1f} MB")

    if dry_run:
        print("dry-run: zip not written")
        return

    out_path = out_path.resolve()
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path, arc in entries:
            zf.write(path, arcname=arc)
    zip_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({zip_mb:.1f} MB)")
    if out_path.stat().st_size > 10 * 1024**3:
        print("WARNING: zip exceeds 10GB contest limit")


def parse_args():
    parser = argparse.ArgumentParser(description="Pack v2.4 submit.zip (fusion+lora)")
    parser.add_argument("--out", type=Path, default=BASE / "submit.zip")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pack(args.out, dry_run=args.dry_run)
