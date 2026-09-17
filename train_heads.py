#!/usr/bin/env python3
"""매니페스트로 4개 헤드를 학습하고, 같은 체크포인트로 추론한다.

학습 전에 manifests/train.csv, valid.csv가 있어야 한다.
  python -m preprocess.build_manifest --root <pjs> --alias Voice_Only_Zeroth_Korean=<zeroth> --out <pjs>/manifests

# 로컬 학습 (폴더당 400클립, 첫 실험용)
python train_heads.py --train-csv manifests/train.csv --valid-csv manifests/valid.csv --ckpt model/task_heads.pt --epochs 8 --batch-size 16 --max-per-source 400

# 로컬 학습 (매니페스트 전체)
python train_heads.py --train-csv manifests/train.csv --valid-csv manifests/valid.csv --ckpt model/task_heads.pt --epochs 8 --batch-size 16

# Colab 학습
# python train_heads.py --train-csv "/content/drive/Shareddrives/Korean Voice Datasets/pjs/manifests/train.csv" --valid-csv "/content/drive/Shareddrives/Korean Voice Datasets/pjs/manifests/valid.csv" --ckpt "/content/drive/Shareddrives/Korean Voice Datasets/pjs/model/task_heads.pt" --epochs 8 --batch-size 16 --max-per-source 400

# 학습한 헤드로 추론
python train_heads.py --predict --ckpt model/task_heads.pt --test-dir data/test --sample-submission data/sample_submission.csv --output output/submission.csv

# 대회 베이스라인 추론 (이 파일이 아님, 모은 데이터를 쓰지 않음)
# python script.py --test-dir data/test --sample-submission data/sample_submission.csv --output output/submission.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_600
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
LABEL_COLUMNS = (
    "VOICE_PRESENT",
    "MUSIC_PRESENT",
    "VOICE_FAKE",
    "MUSIC_FAKE",
)
PROB_COLUMNS = (
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
)
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


class LogmelDataset(Dataset):
    def __init__(self, rows, max_per_source=None):
        if max_per_source:
            rows = _cap_per_source(rows, max_per_source)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features = load_logmel(row["path"])
        labels = torch.tensor(
            [float(row[column]) for column in LABEL_COLUMNS],
            dtype=torch.float32,
        )
        return features, labels, row["ID"]


def _cap_per_source(rows, limit):
    counts = {}
    selected = []
    for row in rows:
        source = row.get("source_folder", "")
        counts[source] = counts.get(source, 0) + 1
        if counts[source] <= limit:
            selected.append(row)
    return selected


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_logmel(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        audio = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // max(audio.size, 1) + 1
        audio = np.tile(audio, repeat_count)[:SEGMENT_SAMPLES]
    else:
        audio = audio[:SEGMENT_SAMPLES]
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    logmel = np.clip((logmel + 80.0) / 80.0, 0.0, 1.0)
    return torch.from_numpy(logmel).unsqueeze(0)


class TaskHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.heads = nn.Linear(64, 4)

    def forward(self, features):
        embedding = self.encoder(features).flatten(1)
        return self.heads(embedding)


def fuse_file_fake(vp, mp, vf, mf):
    return max(vp * vf, mp * mf)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for features, labels, _ids in tqdm(loader, desc="train"):
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * features.size(0)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    correct = 0
    count = 0
    for features, labels, _ids in tqdm(loader, desc="valid"):
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = loss_fn(logits, labels)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += int((preds == labels).all(dim=1).sum().item())
        total += float(loss.item()) * features.size(0)
        count += features.size(0)
    return total / max(count, 1), correct / max(count, 1)


def save_checkpoint(path: Path, model, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_columns": LABEL_COLUMNS,
        },
        path,
    )
    print(f"saved {path}")


def load_model(ckpt: Path, device):
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = TaskHeads().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def collate_train(batch):
    features, labels, ids = zip(*batch)
    return torch.stack(features), torch.stack(labels), ids


def parse_args():
    parser = argparse.ArgumentParser(description="Train or predict 4 task heads.")
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument("--ckpt", type=Path, default=Path("model") / "task_heads.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--test-dir", type=Path, default=Path("data") / "test")
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data") / "sample_submission.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("output") / "submission.csv")
    return parser.parse_args()


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(name)


def train(args, device):
    if not args.train_csv or not args.valid_csv:
        raise ValueError("학습에는 --train-csv 와 --valid-csv 가 필요합니다")
    cap = args.max_per_source or None
    train_set = LogmelDataset(read_csv_rows(args.train_csv), cap)
    valid_set = LogmelDataset(read_csv_rows(args.valid_csv), cap)
    if len(train_set) == 0:
        raise SystemExit("train.csv에 오디오가 없습니다")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_train,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_train,
    )

    model = TaskHeads().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} exact_acc={valid_acc:.4f}"
        )
        if valid_acc >= best_acc:
            best_acc = valid_acc
            save_checkpoint(args.ckpt, model, args)


def find_audio_files(test_dir: Path):
    files = [
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    files.sort(key=lambda path: path.stem)
    if not files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")
    return files


def predict(args, device):
    model = load_model(args.ckpt, device)
    column_names, submission_rows = _read_submission(args.sample_submission)
    audio_files = {
        path.stem: path for path in find_audio_files(args.test_dir)
    }

    for row in tqdm(submission_rows, desc="predict"):
        audio_id = str(row["ID"]).strip()
        audio_path = audio_files[audio_id]
        features = load_logmel(audio_path).unsqueeze(0).to(device)
        with torch.inference_mode():
            probs = torch.sigmoid(model(features))[0].cpu().tolist()
        for column, value in zip(PROB_COLUMNS, probs):
            row[column] = round(float(value), 10)
        row["FILE_FAKE_PROB"] = round(
            fuse_file_fake(probs[0], probs[1], probs[2], probs[3]),
            10,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(submission_rows)
    print(f"Saved {len(submission_rows)} predictions to {args.output}")


def _read_submission(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        return reader.fieldnames, rows


def main():
    args = parse_args()
    device = select_device(args.device)
    if args.predict:
        predict(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
