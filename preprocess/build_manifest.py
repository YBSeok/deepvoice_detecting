#!/usr/bin/env python3
"""폴더 규칙으로 5개 필드 라벨 매니페스트를 만든다.

원본 데이터는 복사하지 않고 경로만 붙인다.

  python -m preprocess.build_manifest \\
    --root "/content/drive/Shareddrives/Korean Voice Datasets/pjs" \\
    --alias Voice_Only_Zeroth_Korean="/content/drive/Shareddrives/Korean Voice Datasets/Zeroth-Korean/zeroth_korean"
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

# 더 구체적인 이름을 먼저 검사한다.
FOLDER_RULES = (
    ("Fake_Voice_Only", (1, 0, 1, 0)),
    ("Fake_Music_Only", (0, 1, 0, 1)),
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


def rows_from_folder(folder: Path, source_name: str, relative_root: Path):
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
                "path": str(audio_path.resolve()),
                "relative_path": relative,
                "source_folder": source_name,
                "rule": labels["rule"],
                **{column: labels[column] for column in LABEL_COLUMNS},
            }
        )
    return rows, None


def collect_rows(root: Path, aliases):
    rows = []
    skipped = []
    alias_names = {name for name, _ in aliases}

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if folder.name in alias_names:
            continue
        folder_rows, skip_name = rows_from_folder(folder, folder.name, root)
        if skip_name:
            skipped.append(skip_name)
            continue
        rows.extend(folder_rows)

    for source_name, folder in aliases:
        if not folder.is_dir():
            raise FileNotFoundError(f"alias 경로가 없습니다: {source_name}={folder}")
        folder_rows, skip_name = rows_from_folder(folder, source_name, folder)
        if skip_name:
            skipped.append(skip_name)
            continue
        if not folder_rows:
            print(f"경고: alias에 오디오가 없습니다: {source_name} -> {folder}")
        rows.extend(folder_rows)
    return rows, skipped


def stratified_split(rows, valid_ratio, seed):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_folder"], []).append(row)

    rng = random.Random(seed)
    train, valid = [], []
    for folder_rows in grouped.values():
        folder_rows = list(folder_rows)
        rng.shuffle(folder_rows)
        cut = max(1, int(len(folder_rows) * valid_ratio)) if len(folder_rows) > 1 else 0
        valid.extend(folder_rows[:cut])
        train.extend(folder_rows[cut:])
    return train, valid


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


def print_summary(rows):
    print(f"총 클립: {len(rows)}")
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
        description="Build 5-field labels from pjs dataset folders."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Korean_Voice_Datasets/pjs 같은 데이터 루트",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "manifests",
        help="CSV를 저장할 디렉터리",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="복사 없이 외부 폴더를 붙인다. 예: Voice_Only_Zeroth_Korean=/path/to/zeroth_korean",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"데이터 루트가 없습니다: {root}")

    aliases = [parse_alias(item) for item in args.alias]
    rows, skipped = collect_rows(root, aliases)
    if skipped:
        print("규칙에 없는 폴더 (건너뜀):", ", ".join(skipped))
    if not rows:
        raise SystemExit("오디오를 찾지 못했습니다.")

    print_summary(rows)
    write_csv(args.out / "all.csv", rows)
    train_rows, valid_rows = stratified_split(rows, args.valid_ratio, args.seed)
    write_csv(args.out / "train.csv", train_rows)
    write_csv(args.out / "valid.csv", valid_rows)
    print(f"저장: {args.out / 'all.csv'}")
    print(f"train {len(train_rows)} / valid {len(valid_rows)}")


if __name__ == "__main__":
    main()
