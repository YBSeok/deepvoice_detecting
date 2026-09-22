#!/usr/bin/env python3
"""폴더 규칙으로 5개 필드 라벨 매니페스트(train/valid/test)를 만든다.

원본 데이터는 복사하지 않고 경로만 붙인다.

Drive `pjs` 루트 예 (폴더명 접두사로 라벨):

  Fake_Music_Only_MusicGen / Fake_Music_Only_Suno
  Fake_Voice_Only_MLAAD_ko / Fake_Voice_Only_TTS_ko
  Music_Only_FMA
  Music_FakeVocal_Mix          # 진짜 반주 + 가짜 보컬
  Voice_Only_LibriSpeech / Voice_Only_Zeroth_Korean / Voice_Only_call_aihub
  Voice_and_Music_FMA
  manifests/                   # 출력용, 스캔에서 제외

  python -m preprocess.build_manifest \\
    --root "/content/drive/.../pjs" \\
    --out "/content/drive/.../pjs/manifests" \\
    --valid-ratio 0.1 --test-ratio 0.1

  python -m preprocess.build_manifest \\
    --root "D:/datasets/pjs" \\
    --alias Voice_Only_Zeroth_Korean="/other/zeroth"
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}

LABEL_COLUMNS = (
    "VOICE_PRESENT",
    "MUSIC_PRESENT",
    "VOICE_FAKE",
    "MUSIC_FAKE",
    "FILE_FAKE",
)

# 매니페스트/캐시 등 — 오디오 소스가 아님.
SKIP_DIR_NAMES = {
    "manifests",
    "cache",
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
}

# 더 구체적인 이름을 먼저 검사한다. (vp, mp, vf, mf)
FOLDER_RULES = (
    ("Fake_Voice_Only", (1, 0, 1, 0)),
    ("Fake_Music_Only", (0, 1, 0, 1)),
    ("Music_FakeVocal", (1, 1, 1, 0)),  # Music_FakeVocal_Mix 등
    ("Voice_and_Music", (1, 1, 0, 0)),
    ("Voice_Only", (1, 0, 0, 0)),
    ("Music_Only", (0, 1, 0, 0)),
)


def labels_from_folder(folder_name: str):
    for prefix, (vp, mp, vf, mf) in FOLDER_RULES:
        if folder_name.startswith(prefix):
            file_fake = 1 if vf or mf else 0
            return {
                "VOICE_PRESENT": vp,
                "MUSIC_PRESENT": mp,
                "VOICE_FAKE": vf,
                "MUSIC_FAKE": mf,
                "FILE_FAKE": file_fake,
                "rule": prefix,
            }
    return None


def iter_audio_files(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def parse_alias(raw: str):
    if "=" not in raw:
        raise ValueError(f"--alias는 Name=경로 형식이어야 합니다: {raw}")
    name, path_text = raw.split("=", 1)
    name = name.strip()
    path = Path(path_text.strip()).expanduser()
    if not name:
        raise ValueError(f"--alias 이름이 비었습니다: {raw}")
    return name, path


def format_path(audio_path: Path, root: Path, path_style: str):
    if path_style == "relative":
        try:
            return audio_path.relative_to(root).as_posix()
        except ValueError:
            return str(audio_path.resolve())
    return str(audio_path.resolve())


def rows_from_folder(folder: Path, source_name: str, relative_root: Path, path_style: str):
    labels = labels_from_folder(source_name)
    if labels is None:
        return [], source_name
    rows = []
    for audio_path in iter_audio_files(folder):
        try:
            relative = audio_path.relative_to(relative_root).as_posix()
        except ValueError:
            relative = audio_path.as_posix()
        rows.append(
            {
                "ID": audio_path.stem,
                "path": format_path(audio_path, relative_root, path_style),
                "relative_path": relative,
                "source_folder": source_name,
                "rule": labels["rule"],
                **{column: labels[column] for column in LABEL_COLUMNS},
            }
        )
    return rows, None


def collect_rows(root: Path, aliases, path_style: str):
    rows = []
    skipped = []
    alias_names = {name for name, _ in aliases}

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if folder.name in SKIP_DIR_NAMES or folder.name in alias_names:
            continue
        if folder.name.startswith("."):
            continue
        folder_rows, skip_name = rows_from_folder(folder, folder.name, root, path_style)
        if skip_name:
            skipped.append(skip_name)
            continue
        rows.extend(folder_rows)

    for source_name, folder in aliases:
        if not folder.is_dir():
            raise FileNotFoundError(f"alias 경로가 없습니다: {source_name}={folder}")
        folder_rows, skip_name = rows_from_folder(folder, source_name, folder, path_style)
        if skip_name:
            skipped.append(skip_name)
            continue
        if not folder_rows:
            print(f"경고: alias에 오디오가 없습니다: {source_name} -> {folder}")
        rows.extend(folder_rows)
    return rows, skipped


def _split_counts(n: int, valid_ratio: float, test_ratio: float):
    """소스 폴더별 train/valid/test 개수. train이 최소 1개 남도록 조정."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0

    n_test = int(n * test_ratio) if test_ratio > 0 else 0
    n_valid = int(n * valid_ratio) if valid_ratio > 0 else 0
    if test_ratio > 0 and n >= 3 and n_test < 1:
        n_test = 1
    if valid_ratio > 0 and n >= 2 and n_valid < 1:
        n_valid = 1

    while n_test + n_valid >= n and n_test > 0:
        n_test -= 1
    while n_test + n_valid >= n and n_valid > 0:
        n_valid -= 1

    n_train = n - n_valid - n_test
    return n_train, n_valid, n_test


def stratified_split(rows, valid_ratio, test_ratio, seed):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_folder"], []).append(row)

    rng = random.Random(seed)
    train, valid, test = [], [], []
    for folder_rows in grouped.values():
        folder_rows = list(folder_rows)
        rng.shuffle(folder_rows)
        n_train, n_valid, n_test = _split_counts(len(folder_rows), valid_ratio, test_ratio)
        i = 0
        test.extend(folder_rows[i : i + n_test])
        i += n_test
        valid.extend(folder_rows[i : i + n_valid])
        i += n_valid
        train.extend(folder_rows[i : i + n_train])
    return train, valid, test


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ID",
        "path",
        "relative_path",
        "source_folder",
        "rule",
        *LABEL_COLUMNS,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, title="총"):
    print(f"{title} 클립: {len(rows)}")
    if not rows:
        return
    counts = {}
    for row in rows:
        key = (
            row["source_folder"],
            row["VOICE_PRESENT"],
            row["MUSIC_PRESENT"],
            row["VOICE_FAKE"],
            row["MUSIC_FAKE"],
            row["FILE_FAKE"],
        )
        counts[key] = counts.get(key, 0) + 1
    print("folder\tcount\tVP MP VF MF FILE")
    for (folder, vp, mp, vf, mf, file_fake), count in sorted(counts.items()):
        print(f"{folder}\t{count}\t{vp} {mp} {vf} {mf} {file_fake}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/valid/test manifests from pjs-style dataset folders."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="데이터 루트 (Fake_*/Voice_*/Music_* 폴더가 있는 디렉터리)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV 저장 디렉터리 (기본: <root>/manifests)",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="복사 없이 외부 폴더를 붙인다. 예: Voice_Only_Zeroth_Korean=/path/to/zeroth",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--path-style",
        choices=["absolute", "relative"],
        default="absolute",
        help="CSV path 컬럼: absolute(기본) 또는 root 기준 relative",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"데이터 루트가 없습니다: {root}")

    out = (args.out or (root / "manifests")).expanduser().resolve()
    if args.valid_ratio < 0 or args.test_ratio < 0:
        raise SystemExit("valid/test ratio는 0 이상이어야 합니다.")
    if args.valid_ratio + args.test_ratio >= 1.0:
        raise SystemExit("valid-ratio + test-ratio 합은 1 미만이어야 합니다.")

    aliases = [parse_alias(item) for item in args.alias]
    rows, skipped = collect_rows(root, aliases, args.path_style)
    if skipped:
        print("규칙에 없는 폴더 (건너뜀):", ", ".join(skipped))
    if not rows:
        raise SystemExit("오디오를 찾지 못했습니다.")

    print_summary(rows, "전체")
    train_rows, valid_rows, test_rows = stratified_split(
        rows, args.valid_ratio, args.test_ratio, args.seed
    )

    write_csv(out / "all.csv", rows)
    write_csv(out / "train.csv", train_rows)
    write_csv(out / "valid.csv", valid_rows)
    write_csv(out / "test.csv", test_rows)

    print()
    print_summary(train_rows, "train")
    print_summary(valid_rows, "valid")
    print_summary(test_rows, "test")
    print(f"저장: {out}")
    print(
        f"train {len(train_rows)} / valid {len(valid_rows)} / test {len(test_rows)} "
        f"(path-style={args.path_style})"
    )


if __name__ == "__main__":
    main()
