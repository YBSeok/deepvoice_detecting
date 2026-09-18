#!/usr/bin/env python3
"""V2 학습: PANNs 임베딩 위 MF/VF 헤드를 학습한다.

백본 역할
  VP/MP : PANNs 그대로 (학습 안 함)
  VF    : 이 스크립트의 VF 헤드 (+ 추론 시 DF-Arena와 max)
  MF    : 이 스크립트의 MF 헤드
  FILE  : max(VP x VF, MP x MF)

학습 데이터
  Music_Only / Voice_and_Music / Voice_Only / Fake_Voice_Only -> MUSIC_FAKE, VOICE_FAKE 라벨 사용
  Fake_Music_Only / 오버레이 -> 부분 fake 혼합 포함
  VF 헤드: 음성 있는 클립만 (Libri vs TTS_ko, 오버레이 포함)

# 로컬 학습 (폴더당 400클립, Zeroth/Phone 1200, 전화 왜곡 30%)
python train_v2.py --train-csv data/manifests/train.csv --valid-csv data/manifests/valid.csv --ckpt model/mf_head.pt --vf-ckpt model/vf_head.pt --max-per-source 400 --voice-per-source 1200 --overlays 400 --phone-frac 0.3
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from heads.music_fake import EMBED_DIM, MusicFakeHead
from learning_data import (
    AUDIO_SAMPLE_RATE,
    add_all_overlays,
    add_phone_rows,
    cache_key,
    load_row_audio,
    music_rows,
    read_csv_rows,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
PANNS_DIR = MODEL_DIR / "panns"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PANNS_SAMPLE_RATE = 32_000


def prepare_panns_labels():
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_panns(device):
    prepare_panns_labels()
    from panns_inference import AudioTagging

    return AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device.type,
    )


def panns_embedding(model, audio):
    resampled = librosa.resample(
        audio,
        orig_sr=AUDIO_SAMPLE_RATE,
        target_sr=PANNS_SAMPLE_RATE,
        res_type="soxr_hq",
    ).astype(np.float32)
    _clipwise, embedding = model.inference(resampled[None])
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim == 2:
        vector = vector[0]
    return torch.from_numpy(vector.copy())


def embed_rows(rows, model, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = []
    music_labels = []
    voice_labels = []
    has_voice = []
    for row in tqdm(rows, desc="PANNs embed"):
        cache_path = cache_dir / cache_key(row)
        if cache_path.is_file():
            vector = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            try:
                vector = panns_embedding(model, load_row_audio(row))
            except Exception as exc:
                print(f"skip {row.get('path')}: {exc}")
                continue
            torch.save(vector, cache_path)
        embeddings.append(vector)
        music_labels.append(float(row["MUSIC_FAKE"]))
        voice_labels.append(float(row.get("VOICE_FAKE", 0)))
        present = int(row.get("VOICE_PRESENT", 0)) == 1 or str(row.get("rule", "")).startswith("overlay")
        has_voice.append(present)
    return (
        torch.stack(embeddings),
        torch.tensor(music_labels, dtype=torch.float32),
        torch.tensor(voice_labels, dtype=torch.float32),
        torch.tensor(has_voice, dtype=torch.bool),
    )


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for features, labels in tqdm(loader, desc="train"):
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
    for features, labels in tqdm(loader, desc="valid"):
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = loss_fn(logits, labels)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += int((preds == labels).sum().item())
        total += float(loss.item()) * features.size(0)
        count += features.size(0)
    return total / max(count, 1), correct / max(count, 1)


def save_checkpoint(path: Path, model, task):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embed_dim": EMBED_DIM,
            "task": task,
            "backbone": "panns_cnn14",
        },
        path,
    )
    print(f"saved {path}")


@torch.no_grad()
def hard_real_indices(model, features, labels, device, min_prob=0.25):
    model.eval()
    loader = DataLoader(
        EmbeddingDataset(features, labels),
        batch_size=256,
        shuffle=False,
    )
    probs = []
    for batch_x, _batch_y in loader:
        logits = model(batch_x.to(device))
        probs.append(torch.sigmoid(logits).cpu())
    probs = torch.cat(probs)
    mask = (labels < 0.5) & (probs >= min_prob)
    return torch.where(mask)[0]


def fit_head(
    train_x,
    train_y,
    valid_x,
    valid_y,
    ckpt,
    task,
    device,
    epochs,
    batch_size,
    lr,
    init_ckpt=None,
):
    if int(train_y.numel()) == 0:
        raise SystemExit(f"{task}: 학습 샘플이 없습니다.")
    train_loader = DataLoader(
        EmbeddingDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        EmbeddingDataset(valid_x, valid_y),
        batch_size=batch_size,
        shuffle=False,
    )
    model = MusicFakeHead(train_x.shape[1])
    if init_ckpt and Path(init_ckpt).is_file():
        payload = torch.load(init_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"])
        print(f"resume {init_ckpt}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_acc = -1.0
    if init_ckpt:
        _valid_loss, best_acc = evaluate(model, valid_loader, loss_fn, device)
        print(f"{task} start: valid acc={best_acc:.4f}")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, device)
        print(
            f"{task} epoch {epoch}: train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} acc={valid_acc:.4f}"
        )
        if valid_acc >= best_acc:
            best_acc = valid_acc
            save_checkpoint(ckpt, model, task)
    return best_acc


def parse_args():
    parser = argparse.ArgumentParser(description="V2: train music-fake head on frozen PANNs.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=MODEL_DIR / "mf_head.pt")
    parser.add_argument("--vf-ckpt", type=Path, default=MODEL_DIR / "vf_head.pt")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=BASE_DIR / "data" / "cache" / "panns_mf",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument(
        "--voice-per-source",
        type=int,
        default=1200,
        help="Zeroth·Phone 한국어 실음성 캡. TTS/전화 오탐을 줄이려고 영어 Libri보다 많이 넣는다.",
    )
    parser.add_argument("--overlays", type=int, default=400)
    parser.add_argument(
        "--phone-frac",
        type=float,
        default=0.3,
        help="이미 전화인 Phone 폴더를 제외하고, 그 비율만큼 8 kHz 왕복 복사본을 넣는다.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(name)


def main():
    args = parse_args()
    device = select_device(args.device)
    cap = args.max_per_source or None
    voice_cap = args.voice_per_source or None
    train_rows = music_rows(read_csv_rows(args.train_csv), cap, voice_cap)
    valid_rows = music_rows(read_csv_rows(args.valid_csv), cap, voice_cap)
    train_rows.extend(add_all_overlays(train_rows, args.overlays, args.seed))
    valid_rows.extend(add_all_overlays(valid_rows, max(args.overlays // 10, 0), args.seed + 1))
    train_rows.extend(add_phone_rows(train_rows, args.phone_frac, args.seed + 99))
    valid_rows.extend(add_phone_rows(valid_rows, args.phone_frac, args.seed + 100))
    if not train_rows:
        raise SystemExit("학습 클립이 없습니다. 매니페스트 규칙을 확인하세요.")

    print(f"train clips: {len(train_rows)}")
    print(f"valid clips: {len(valid_rows)}")

    panns = load_panns(device)
    train_x, train_mf, train_vf, train_has_voice = embed_rows(train_rows, panns, args.cache_dir)
    valid_x, valid_mf, valid_vf, valid_has_voice = embed_rows(valid_rows, panns, args.cache_dir)
    del panns
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("train MF")
    fit_head(
        train_x,
        train_mf,
        valid_x,
        valid_mf,
        args.ckpt,
        "music_fake",
        device,
        args.epochs,
        args.batch_size,
        args.lr,
    )
    print("train VF")
    vf_train_x = train_x[train_has_voice]
    vf_train_y = train_vf[train_has_voice]
    vf_valid_x = valid_x[valid_has_voice]
    vf_valid_y = valid_vf[valid_has_voice]
    n_real = int((vf_train_y < 0.5).sum().item())
    n_fake = int((vf_train_y >= 0.5).sum().item())
    print(f"VF samples: real={n_real} fake={n_fake}")
    fit_head(
        vf_train_x,
        vf_train_y,
        vf_valid_x,
        vf_valid_y,
        args.vf_ckpt,
        "voice_fake",
        device,
        args.epochs,
        args.batch_size,
        args.lr,
    )

    payload = torch.load(args.vf_ckpt, map_location="cpu", weights_only=True)
    probe_model = MusicFakeHead(vf_train_x.shape[1]).to(device)
    probe_model.load_state_dict(payload["state_dict"])
    hard_idx = hard_real_indices(probe_model, vf_train_x, vf_train_y, device)
    print(f"hard real negatives (VF>=0.25): {int(hard_idx.numel())} / {n_real}")
    del probe_model
    if int(hard_idx.numel()) == 0:
        return
    print("refine VF on hard real negatives")
    extra_x = vf_train_x[hard_idx].repeat(4, 1)
    extra_y = vf_train_y[hard_idx].repeat(4)
    fit_head(
        torch.cat([vf_train_x, extra_x], dim=0),
        torch.cat([vf_train_y, extra_y], dim=0),
        vf_valid_x,
        vf_valid_y,
        args.vf_ckpt,
        "voice_fake_hardneg",
        device,
        max(args.epochs // 2, 4),
        args.batch_size,
        args.lr * 0.3,
        init_ckpt=args.vf_ckpt,
    )


if __name__ == "__main__":
    main()
