#!/usr/bin/env python3
"""V1 실험용. V2 학습/추론은 아래를 쓴다.

python train_v2.py --train-csv data/manifests/train.csv --valid-csv data/manifests/valid.csv --ckpt model/mf_head.pt --vf-ckpt model/vf_head.pt --max-per-source 400 --overlays 400

python script.py --test-dir data/test --sample-submission data/sample_submission.csv --output output/submission.csv --mf-ckpt model/mf_head.pt --vf-ckpt model/vf_head.pt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_600
EMBED_DIM = 1280
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


class TaskHeads(nn.Module):
    """고정된 DF-Arena 1280차원 임베딩 → 4개 필드."""

    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.heads = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

    def forward(self, features):
        return self.heads(features)


class EmbeddingDataset(Dataset):
    def __init__(self, rows, embeddings):
        self.rows = rows
        self.embeddings = embeddings

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        labels = torch.tensor(
            [float(row[column]) for column in LABEL_COLUMNS],
            dtype=torch.float32,
        )
        return self.embeddings[index], labels, row["ID"]


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


def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        return np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // max(audio.size, 1) + 1
        return np.tile(audio, repeat_count)[:SEGMENT_SAMPLES]
    return audio[:SEGMENT_SAMPLES]


def cache_key(path: Path):
    digest = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"{path.stem}_{digest}.pt"


def load_df_arena(device):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    previous_directory = Path.cwd()
    os.chdir(DF_ARENA_DIR)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(DF_ARENA_DIR),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    finally:
        os.chdir(previous_directory)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def extract_embedding(backbone, audio, device):
    segment = torch.from_numpy(audio).to(device)
    ssl_features = backbone._ssl_features(segment)
    cls_embedding, _frames, _attn = backbone.conformer.forward_tokens(ssl_features)
    return cls_embedding.squeeze(0).float().cpu()


def embeddings_for_rows(rows, backbone, device, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = []
    for row in tqdm(rows, desc="DF-Arena embed"):
        audio_path = Path(row["path"])
        cache_path = cache_dir / cache_key(audio_path)
        if cache_path.is_file():
            embeddings.append(torch.load(cache_path, map_location="cpu", weights_only=True))
            continue
        audio = load_audio(audio_path)
        vector = extract_embedding(backbone, audio, device)
        torch.save(vector, cache_path)
        embeddings.append(vector)
    return torch.stack(embeddings)


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


def save_checkpoint(path: Path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_columns": LABEL_COLUMNS,
            "embed_dim": EMBED_DIM,
            "backbone": "df_arena_1b",
        },
        path,
    )
    print(f"saved {path}")


def load_heads(ckpt: Path, device):
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = TaskHeads(payload.get("embed_dim", EMBED_DIM)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def collate_train(batch):
    features, labels, ids = zip(*batch)
    return torch.stack(features), torch.stack(labels), ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune 4 task heads on frozen DF-Arena 1B embeddings."
    )
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument("--ckpt", type=Path, default=MODEL_DIR / "task_heads.pt")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=BASE_DIR / "data" / "cache" / "df_arena_emb",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
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
    train_rows = read_csv_rows(args.train_csv)
    valid_rows = read_csv_rows(args.valid_csv)
    if cap:
        train_rows = _cap_per_source(train_rows, cap)
        valid_rows = _cap_per_source(valid_rows, cap)
    if not train_rows:
        raise SystemExit("train.csv에 오디오가 없습니다")

    backbone = load_df_arena(device)
    print("DF-Arena 1B frozen. extracting embeddings...")
    train_emb = embeddings_for_rows(train_rows, backbone.backbone, device, args.cache_dir)
    valid_emb = embeddings_for_rows(valid_rows, backbone.backbone, device, args.cache_dir)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_loader = DataLoader(
        EmbeddingDataset(train_rows, train_emb),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_train,
    )
    valid_loader = DataLoader(
        EmbeddingDataset(valid_rows, valid_emb),
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
            save_checkpoint(args.ckpt, model)


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
    heads = load_heads(args.ckpt, device)
    backbone = load_df_arena(device)
    column_names, submission_rows = _read_submission(args.sample_submission)
    audio_files = {path.stem: path for path in find_audio_files(args.test_dir)}

    for row in tqdm(submission_rows, desc="predict"):
        audio_id = str(row["ID"]).strip()
        audio = load_audio(audio_files[audio_id])
        embedding = extract_embedding(backbone.backbone, audio, device).unsqueeze(0).to(device)
        with torch.inference_mode():
            probs = torch.sigmoid(heads(embedding))[0].cpu().tolist()
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
