#!/usr/bin/env python3
"""경진대회 테스트 데이터에 대한 5개 확률값을 생성한다.

VP/MP는 원본 믹스의 PANNs. VF/MF는 원본 믹스를 DF-Arena에 넣은 뒤
프레임 임베딩을 소스 마스크로만 나눠 풀링한다. 파형은 수정하지 않는다.
FILE_FAKE_PROB만 네 헤드의 융합이다.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

# 추론에는 model 폴더에 포함된 로컬 파일만 사용한다.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import librosa
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track
from tqdm import tqdm


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"

DEFAULT_TEST_DIR = Path("data") / "test"
DEFAULT_SAMPLE_SUBMISSION = Path("data") / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = Path("output") / "submission.csv"

# 오디오 처리 설정
AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600
SILENCE_RMS = 1e-5
MASK_SMOOTH_SAMPLES = 320

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]

# 4개 헤드가 직접 예측하는 필드. FILE_FAKE_PROB는 이 네 값의 융합이다.
TASK_HEAD_COLUMNS = (
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}


# -----------------------------------------------------------------------------
# 1. 입력 파일 및 제출 양식 확인
# -----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Encode the original mix, pool frames with source masks, fuse FILE_FAKE_PROB."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def select_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(device_name)


def find_audio_files(test_dir):
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    audio_files = []
    for path in test_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.append(path)
    audio_files.sort(key=lambda path: path.stem)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")

    audio_ids = [path.stem for path in audio_files]
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("Audio IDs must be unique")
    return audio_files


def read_sample_submission(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sample submission not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        column_names = reader.fieldnames
        rows = list(reader)

    if column_names is None or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")

    required_columns = ["ID"] + PREDICTION_COLUMNS
    missing_columns = [name for name in required_columns if name not in column_names]
    if missing_columns:
        raise ValueError(f"Sample submission is missing columns: {missing_columns}")

    seen_ids = set()
    for row in rows:
        audio_id = str(row["ID"]).strip()
        if not audio_id:
            raise ValueError("Sample submission contains an empty ID")
        if audio_id in seen_ids:
            raise ValueError(f"Duplicate ID in sample submission: {audio_id}")
        seen_ids.add(audio_id)
        row["ID"] = audio_id

    return column_names, rows


def order_audio_files(audio_files, submission_rows):
    audio_by_id = {path.stem: path for path in audio_files}
    submission_ids = [row["ID"] for row in submission_rows]

    missing_ids = [audio_id for audio_id in submission_ids if audio_id not in audio_by_id]
    extra_ids = [audio_id for audio_id in audio_by_id if audio_id not in submission_ids]
    if missing_ids or extra_ids:
        raise ValueError(
            "Test audio and sample submission IDs do not match. "
            f"Missing: {missing_ids[:5]}, Extra: {extra_ids[:5]}"
        )

    return [audio_by_id[audio_id] for audio_id in submission_ids]


def load_audio(audio_path):
    audio, _ = librosa.load(
        audio_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32
    )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {audio_path}")
    return audio


# -----------------------------------------------------------------------------
# 2. 오디오 구간 분할
# -----------------------------------------------------------------------------

def get_segment_starts(audio_length):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]

    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio, start):
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[:SEGMENT_SAMPLES].astype(np.float32)

    end = start + SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# 3. Presence heads (VP, MP)
# -----------------------------------------------------------------------------

def prepare_panns_labels():
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_panns_model(device):
    prepare_panns_labels()
    from panns_inference import AudioTagging, labels

    model = AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device.type,
    )

    config_path = PANNS_DIR / "component_labels.json"
    label_groups = json.loads(config_path.read_text(encoding="utf-8"))
    label_to_index = {label: index for index, label in enumerate(labels)}
    voice_indices = [label_to_index[label] for label in label_groups["voice"]]
    music_indices = [label_to_index[label] for label in label_groups["music"]]
    return model, voice_indices, music_indices


def make_panns_segments(audio):
    segments = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment = librosa.resample(
            segment,
            orig_sr=AUDIO_SAMPLE_RATE,
            target_sr=PANNS_SAMPLE_RATE,
            res_type="soxr_hq",
        )
        segments.append(segment.astype(np.float32))
    return np.stack(segments)


def predict_presence_heads(model, voice_indices, music_indices, audio):
    """Presence heads: PANNs 태그에서 음성·음악 그룹 최댓값을 확률로 쓴다."""
    segments = make_panns_segments(audio)
    predictions, _ = model.inference(segments)
    voice_present = float(predictions[:, voice_indices].max())
    music_present = float(predictions[:, music_indices].max())
    return voice_present, music_present


# -----------------------------------------------------------------------------
# 4. 소스 마스크 (파형은 바꾸지 않고 프레임 가중치만 만듦)
# -----------------------------------------------------------------------------

def load_htdemucs_model():
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = original_torch_load
    return model.cpu().eval()


def estimate_source_stems(audio_path, model, device):
    waveform = load_track(
        audio_path, model.audio_channels, model.samplerate
    ).float()
    mono_waveform = waveform.mean(0)
    mean = mono_waveform.mean()
    std = mono_waveform.std()

    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * AUDIO_SAMPLE_RATE / model.samplerate)
        silence = np.zeros(max(1, length), dtype=np.float32)
        return silence, silence.copy()

    normalized_waveform = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model,
            normalized_waveform[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + mean

    vocal_index = model.sources.index("vocals")
    voice_audio = sources[vocal_index].mean(0, keepdim=True)

    music_sources = []
    for index, source_name in enumerate(model.sources):
        if source_name != "vocals":
            music_sources.append(sources[index])
    music_audio = torch.stack(music_sources).sum(0).mean(0, keepdim=True)

    voice_audio = torchaudio.functional.resample(
        voice_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    music_audio = torchaudio.functional.resample(
        music_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    return (
        voice_audio.cpu().numpy().astype(np.float32),
        music_audio.cpu().numpy().astype(np.float32),
    )


def smooth_envelope(audio, win=MASK_SMOOTH_SAMPLES):
    magnitude = np.abs(audio.astype(np.float64, copy=False))
    if magnitude.size == 0 or magnitude.size < win:
        return magnitude
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(magnitude, kernel, mode="same")


def downsample_weights(values, num_frames):
    values = np.asarray(values, dtype=np.float64)
    if num_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 0:
        return np.zeros(num_frames, dtype=np.float64)
    if values.size == 1:
        return np.full(num_frames, float(values[0]), dtype=np.float64)
    source = np.linspace(0.0, 1.0, num=values.size)
    target = np.linspace(0.0, 1.0, num=num_frames)
    return np.interp(target, source, values)


def pooled_spoof_probability(frames, weights, classifier, fake_label_index):
    weights = torch.as_tensor(
        weights, device=frames.device, dtype=frames.dtype
    ).clamp(min=0)
    if float(weights.sum()) < 1e-6:
        return 0.0
    weights = weights / weights.sum()
    pooled = (frames[0] * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
    logits = classifier(pooled)
    probabilities = torch.softmax(logits.float(), dim=-1)
    return float(probabilities[0, fake_label_index])


# -----------------------------------------------------------------------------
# 5. Fake heads (VF, MF)
# -----------------------------------------------------------------------------

def load_df_arena_model(device):
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
    fake_label_index = int(model.config.label2id["spoof"])
    return model, fake_label_index


def calculate_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def predict_component_fake(
    model, fake_label_index, mix, voice_stem, music_stem, device
):
    """원본 믹스만 인코딩하고, 마스크는 프레임 풀링에만 쓴다."""
    length = min(mix.size, voice_stem.size, music_stem.size)
    mix = mix[:length]
    voice_env = smooth_envelope(voice_stem[:length])
    music_env = smooth_envelope(music_stem[:length])
    classifier = model.backbone.conformer.fc5

    voice_scores = []
    music_scores = []
    for start in get_segment_starts(mix.size):
        segment = extract_segment(mix, start)
        voice_seg = extract_segment(voice_env.astype(np.float32), start)
        music_seg = extract_segment(music_env.astype(np.float32), start)
        if calculate_rms(segment) < SILENCE_RMS:
            continue

        segment_tensor = torch.from_numpy(segment).to(device)
        with torch.inference_mode():
            frames = model.encode_frames(segment_tensor)
            num_frames = int(frames.shape[1])
            voice_weights = downsample_weights(voice_seg, num_frames)
            music_weights = downsample_weights(music_seg, num_frames)
            voice_scores.append(
                pooled_spoof_probability(
                    frames, voice_weights, classifier, fake_label_index
                )
            )
            music_scores.append(
                pooled_spoof_probability(
                    frames, music_weights, classifier, fake_label_index
                )
            )

    voice_fake = max(voice_scores) if voice_scores else 0.0
    music_fake = max(music_scores) if music_scores else 0.0
    return voice_fake, music_fake


# -----------------------------------------------------------------------------
# 6. 헤드 예측 및 제출 파일 저장
# -----------------------------------------------------------------------------

def empty_head_outputs():
    return {column: 0.0 for column in TASK_HEAD_COLUMNS}


def fuse_file_fake(head_outputs):
    """FILE_FAKE_PROB는 5번째 헤드가 아니라 4개 헤드 출력의 융합이다."""
    voice_risk = (
        head_outputs["VOICE_PRESENT_PROB"] * head_outputs["VOICE_FAKE_PROB"]
    )
    music_risk = (
        head_outputs["MUSIC_PRESENT_PROB"] * head_outputs["MUSIC_FAKE_PROB"]
    )
    return max(voice_risk, music_risk)


def write_prediction_row(row, head_outputs):
    for column, value in head_outputs.items():
        row[column] = round(value, 10)
    row["FILE_FAKE_PROB"] = round(fuse_file_fake(head_outputs), 10)


class FourHeadPredictor:
    """원본 믹스에서 5개 필드를 만든다.

    Head VP: PANNs 음성 라벨 그룹 → VOICE_PRESENT_PROB
    Head MP: PANNs 음악 라벨 그룹 → MUSIC_PRESENT_PROB
    Head VF/MF: 원본 믹스 인코딩 후 프레임을 보컬/음악 마스크로 풀링

    FILE_FAKE_PROB = max(VP × VF, MP × MF)
    """

    def __init__(self, device):
        self.device = device

    def run_presence_heads(self, audio_files):
        model, voice_indices, music_indices = load_panns_model(self.device)
        presence_scores = {}

        for audio_path in tqdm(audio_files, desc="Heads VP/MP"):
            try:
                audio = load_audio(audio_path)
                voice_present, music_present = predict_presence_heads(
                    model, voice_indices, music_indices, audio
                )
                presence_scores[audio_path.stem] = {
                    "VOICE_PRESENT_PROB": voice_present,
                    "MUSIC_PRESENT_PROB": music_present,
                }
            except Exception:
                presence_scores[audio_path.stem] = {
                    "VOICE_PRESENT_PROB": 0.0,
                    "MUSIC_PRESENT_PROB": 0.0,
                }

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return presence_scores

    def run_fake_heads(self, audio_files, presence_scores):
        df_arena_model, fake_label_index = load_df_arena_model(self.device)
        htdemucs_model = load_htdemucs_model()
        head_outputs_by_id = {}

        for audio_path in tqdm(audio_files, desc="Heads VF/MF"):
            head_outputs = empty_head_outputs()
            head_outputs.update(presence_scores.get(audio_path.stem, {}))
            try:
                mix = load_audio(audio_path)
                voice_stem, music_stem = estimate_source_stems(
                    audio_path, htdemucs_model, self.device
                )
                voice_fake, music_fake = predict_component_fake(
                    df_arena_model,
                    fake_label_index,
                    mix,
                    voice_stem,
                    music_stem,
                    self.device,
                )
                head_outputs["VOICE_FAKE_PROB"] = voice_fake
                head_outputs["MUSIC_FAKE_PROB"] = music_fake
            except Exception:
                pass
            head_outputs_by_id[audio_path.stem] = head_outputs

        del df_arena_model
        del htdemucs_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return head_outputs_by_id

    def predict(self, audio_files, submission_rows):
        presence_scores = self.run_presence_heads(audio_files)
        head_outputs_by_id = self.run_fake_heads(audio_files, presence_scores)

        for index, audio_path in enumerate(audio_files):
            write_prediction_row(
                submission_rows[index],
                head_outputs_by_id[audio_path.stem],
            )
        return submission_rows


def save_submission(output_path, column_names, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    device = select_device(args.device)

    # 1. 테스트 파일을 제출 양식의 ID 순서에 맞춘다.
    audio_files = find_audio_files(args.test_dir)
    column_names, submission_rows = read_sample_submission(args.sample_submission)
    audio_files = order_audio_files(audio_files, submission_rows)

    # 2. VP/MP는 믹스, VF/MF는 원본 인코딩 + 프레임 풀링, FILE_FAKE만 융합한다.
    predictor = FourHeadPredictor(device)
    submission_rows = predictor.predict(audio_files, submission_rows)

    # 3. 5개 예측값을 제출 파일로 저장한다.
    save_submission(args.output, column_names, submission_rows)
    print(f"Saved {len(submission_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
