#!/usr/bin/env python3
"""v3.1 제출용 submit.zip (AntiDeepfake + LoRA).

포함: script.py, antideepfake_vf.py, ad_lora.py, df_lora.py, requirements.txt, heads/
      model/ antideepfake_xlsr_1b + ad_lora.pt + panns + htdemucs + mf_head.pt
제외: df_arena_1b(용량), vf_fusion, vf_head, df_arena_lora, cache

  python pack_submit.py --out submit.zip
  python pack_submit.py --with-df   # A/B용 DF 포함(용량↑)
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


BASE = Path(__file__).resolve().parent

CODE_FILES = [
    "script.py",
    "antideepfake_vf.py",
    "ad_lora.py",
    "df_lora.py",  # LoRALinear 공유
    "requirements.txt",
    "heads/__init__.py",
    "heads/music_fake.py",
]

MODEL_KEEP_SUFFIXES = {
    ".pt",
    ".pth",
    ".th",
    ".bin",
    ".json",
    ".yaml",
    ".txt",
    ".csv",
    ".py",
    ".safetensors",
    ".model",
}
MODEL_SKIP_NAMES = {
    "df_arena_lora.pt",
    "vf_fusion.pt",
    "vf_head.pt",
}
MODEL_SKIP_DIR_PARTS = {".cache", "__pycache__", ".git"}


def should_keep_model_file(path: Path, model_root: Path, with_df: bool) -> bool:
    rel = path.relative_to(model_root)
    if any(part in MODEL_SKIP_DIR_PARTS for part in rel.parts):
        return False
    if not with_df and rel.parts and rel.parts[0] == "df_arena_1b":
        return False
    if path.name in MODEL_SKIP_NAMES:
        return False
    if path.suffix.lower() not in MODEL_KEEP_SUFFIXES and path.name not in {
        ".gitattributes",
        "LICENSE",
        "LICENSE.txt",
        "LICENSE-CHECKPOINT",
        "README.md",
    }:
        if path.suffix == "" and path.name not in {
            "LICENSE",
            "LICENSE.txt",
            "LICENSE-CHECKPOINT",
        }:
            return False
    return True


def required_files(with_df: bool):
    missing = []
    code = list(CODE_FILES)
    if with_df:
        code.append("df_arena_vf.py")
    for rel in code:
        if not (BASE / rel).is_file():
            missing.append(rel)
    must_files = [
        "model/mf_head.pt",
        "model/ad_lora.pt",
        "model/panns/Cnn14_mAP=0.431.pth",
        "model/htdemucs/955717e8-8726e21a.th",
        "model/antideepfake_xlsr_1b/model.safetensors",
    ]
    if with_df:
        must_files.extend(
            [
                "model/df_arena_1b/pytorch_model.bin",
                "model/df_arena_1b/config.json",
            ]
        )
    for rel in must_files:
        if not (BASE / rel).is_file():
            missing.append(rel)
    return missing


def pack(out_path: Path, dry_run: bool = False, with_df: bool = False):
    missing = required_files(with_df)
    if missing:
        raise SystemExit("제출에 필요한 파일이 없습니다:\n  - " + "\n  - ".join(missing))

    entries = []
    code = list(CODE_FILES)
    if with_df:
        code.append("df_arena_vf.py")
    for rel in code:
        entries.append((BASE / rel, rel.replace("\\", "/")))

    model_root = BASE / "model"
    for path in model_root.rglob("*"):
        if not path.is_file():
            continue
        if not should_keep_model_file(path, model_root, with_df):
            continue
        entries.append((path, path.relative_to(BASE).as_posix()))

    total = sum(p.stat().st_size for p, _ in entries)
    print(f"files: {len(entries)}  with_df={with_df}")
    print(f"raw bytes: {total / 1e9:.2f} GB")
    for path, arc in entries:
        if path.stat().st_size > 50_000_000:
            print(f"  {arc}: {path.stat().st_size / 1e6:.1f} MB")
    if total > 10 * 1024**3:
        print("WARNING: uncompressed payload exceeds 10GB contest limit")

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
    parser = argparse.ArgumentParser(description="Pack v3.1 submit.zip (AD+LoRA)")
    parser.add_argument("--out", type=Path, default=BASE / "submit.zip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--with-df",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pack DF-Arena for default --vf-mode ensemble (default: on). --no-with-df to omit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pack(args.out, dry_run=args.dry_run, with_df=args.with_df)
