#!/usr/bin/env python3
"""매니페스트 → 오버레이/전화 채널 학습 행.

train_v2(PANNs MF)와 embed_df_arena(VF 1280 추출)가 같은 행 구성을 쓴다.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import librosa
import numpy as np


AUDIO_SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_600
PHONE_SAMPLE_RATE = 8_000

REAL_RULES = {"Music_Only", "Voice_and_Music", "Voice_Only", "Fake_Voice_Only"}
FAKE_RULES = {"Fake_Music_Only"}
TRAIN_RULES = REAL_RULES | FAKE_RULES


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_rewrites(items):
    pairs = []
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--rewrite-prefix 형식은 SRC=DST 입니다: {item}")
        src, dst = item.split("=", 1)
        pairs.append((src, dst))
    return pairs


def rewrite_path(path_text, rewrites):
    if not rewrites or not path_text:
        return path_text
    if str(path_text).startswith("phone::"):
        return "phone::" + rewrite_path(path_text[len("phone::") :], rewrites)
    if str(path_text).startswith("overlay::"):
        return path_text
    text = str(path_text).replace("\\", "/")
    for src, dst in rewrites:
        src_n = src.replace("\\", "/").rstrip("/")
        dst_n = dst.replace("\\", "/").rstrip("/")
        if text.lower().startswith(src_n.lower()):
            return dst_n + text[len(src_n) :]
    return path_text


def rewrite_row(row, rewrites):
    out = dict(row)
    for key in ("path", "voice_path", "music_path"):
        if out.get(key):
            out[key] = rewrite_path(out[key], rewrites)
    return out


def folder_contains(row, needle):
    if not needle:
        return True
    return needle.lower() in str(row.get("source_folder", "")).lower()


def row_limit(row, max_per_source, voice_per_source, mix_per_source=None):
    source = row.get("source_folder", "")
    rule = row.get("rule", "")
    if voice_per_source and ("Zeroth" in source or "Phone" in source):
        return voice_per_source
    if mix_per_source and rule == "Voice_and_Music":
        return mix_per_source
    return max_per_source


def music_rows(rows, max_per_source=None, voice_per_source=None, mix_per_source=None):
    selected = [row for row in rows if row.get("rule") in TRAIN_RULES]
    if not max_per_source:
        return selected
    counts = {}
    capped = []
    for row in selected:
        source = row.get("source_folder", "")
        limit = row_limit(row, max_per_source, voice_per_source, mix_per_source)
        counts[source] = counts.get(source, 0) + 1
        if counts[source] <= limit:
            capped.append(row)
    return capped


def load_audio(path):
    audio, _ = librosa.load(str(path), sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        return np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // max(audio.size, 1) + 1
        return np.tile(audio, repeat_count)[:SEGMENT_SAMPLES]
    return audio[:SEGMENT_SAMPLES]


def peak_norm(audio, peak=0.8):
    mag = float(np.max(np.abs(audio)))
    if mag < 1e-6:
        return audio
    return audio / mag * peak


def overlay_audio(voice, music, voice_gain=1.0, music_gain=1.0):
    length = min(voice.size, music.size)
    mix = peak_norm(voice[:length]) * float(voice_gain) + peak_norm(music[:length]) * float(
        music_gain
    )
    return peak_norm(mix, 0.9).astype(np.float32)


def telephone_channel(audio, sr=AUDIO_SAMPLE_RATE, seed=0):
    """8 kHz 왕복 + 약한 히스. 대회 평가셋의 전화 채널을 흉내낸다."""
    rng = np.random.default_rng(int(seed) % (2**32))
    narrow = librosa.resample(
        audio, orig_sr=sr, target_sr=PHONE_SAMPLE_RATE, res_type="soxr_hq"
    )
    band = librosa.resample(
        narrow, orig_sr=PHONE_SAMPLE_RATE, target_sr=sr, res_type="soxr_hq"
    )
    if band.size < audio.size:
        band = np.pad(band, (0, audio.size - band.size))
    else:
        band = band[: audio.size]
    noise = rng.normal(0.0, 0.003, size=band.size).astype(np.float32)
    return peak_norm(band + noise, 0.7).astype(np.float32)


def make_overlay_rows(
    rows,
    count,
    seed,
    voice_rule,
    music_rule,
    voice_fake,
    music_fake,
    name,
    voice_folder=None,
    music_folder=None,
    voice_gain=1.0,
    music_gain=1.0,
):
    voices = [row for row in rows if row.get("rule") == voice_rule]
    if voice_folder:
        voices = [row for row in voices if folder_contains(row, voice_folder)]
    musics = [row for row in rows if row.get("rule") == music_rule]
    if music_folder:
        musics = [row for row in musics if folder_contains(row, music_folder)]
    if count <= 0 or not voices or not musics:
        return []
    rng = np.random.default_rng(seed)
    overlays = []
    for _ in range(count):
        voice = voices[int(rng.integers(0, len(voices)))]
        music = musics[int(rng.integers(0, len(musics)))]
        overlays.append(
            {
                "path": f"overlay::{name}::{voice['path']}::{music['path']}",
                "voice_path": voice["path"],
                "music_path": music["path"],
                "VOICE_PRESENT": 1,
                "MUSIC_PRESENT": 1,
                "VOICE_FAKE": voice_fake,
                "MUSIC_FAKE": music_fake,
                "source_folder": name,
                "rule": name,
                "voice_gain": voice_gain,
                "music_gain": music_gain,
            }
        )
    return overlays


def add_all_overlays(rows, count, seed):
    extras = []
    extras.extend(
        make_overlay_rows(rows, count, seed, "Voice_Only", "Fake_Music_Only", 0, 1, "overlay_rv_fm")
    )
    extras.extend(
        make_overlay_rows(
            rows, count, seed + 1, "Fake_Voice_Only", "Music_Only", 1, 0, "overlay_fv_rm"
        )
    )
    extras.extend(
        make_overlay_rows(
            rows, count, seed + 2, "Fake_Voice_Only", "Fake_Music_Only", 1, 1, "overlay_fv_fm"
        )
    )
    extras.extend(
        make_overlay_rows(
            rows,
            count,
            seed + 3,
            "Voice_Only",
            "Fake_Music_Only",
            0,
            1,
            "overlay_ko_fm",
            voice_folder="Voice_Only_Zeroth",
        )
    )
    extras.extend(
        make_overlay_rows(
            rows,
            count,
            seed + 4,
            "Voice_Only",
            "Music_Only",
            0,
            0,
            "overlay_ko_rm",
            voice_folder="Voice_Only_Zeroth",
        )
    )
    extras.extend(
        make_overlay_rows(
            rows,
            max(count // 2, 0),
            seed + 5,
            "Voice_Only",
            "Fake_Music_Only",
            0,
            1,
            "overlay_rv_fm_quiet",
            voice_gain=0.45,
            music_gain=1.0,
        )
    )
    extras.extend(
        make_overlay_rows(
            rows,
            max(count // 2, 0),
            seed + 6,
            "Fake_Voice_Only",
            "Music_Only",
            1,
            0,
            "overlay_fv_rm_quiet",
            voice_gain=0.45,
            music_gain=1.0,
        )
    )
    mg_count = min(count, 80) if count else 0
    extras.extend(
        make_overlay_rows(
            rows,
            mg_count,
            seed + 7,
            "Voice_Only",
            "Fake_Music_Only",
            0,
            1,
            "overlay_rv_mg",
            music_folder="Fake_Music_Only_MusicGen",
        )
    )
    extras.extend(
        make_overlay_rows(
            rows,
            mg_count,
            seed + 8,
            "Fake_Voice_Only",
            "Fake_Music_Only",
            1,
            1,
            "overlay_fv_mg",
            music_folder="Fake_Music_Only_MusicGen",
        )
    )
    extras.extend(
        make_overlay_rows(
            rows,
            mg_count,
            seed + 9,
            "Voice_Only",
            "Fake_Music_Only",
            0,
            1,
            "overlay_ko_mg",
            voice_folder="Voice_Only_Zeroth",
            music_folder="Fake_Music_Only_MusicGen",
        )
    )
    return extras


def add_phone_rows(rows, frac, seed):
    if frac <= 0 or not rows:
        return []
    rng = np.random.default_rng(seed)
    n = int(round(len(rows) * float(frac)))
    n = min(max(n, 0), len(rows))
    if n == 0:
        return []
    eligible = [
        i
        for i, row in enumerate(rows)
        if "Phone" not in str(row.get("source_folder", ""))
        and not str(row.get("path", "")).startswith("phone::")
    ]
    if not eligible:
        return []
    n = min(n, len(eligible))
    chosen = rng.choice(eligible, size=n, replace=False)
    extras = []
    for index in chosen:
        row = dict(rows[int(index)])
        row["path"] = "phone::" + str(row["path"])
        row["source_folder"] = str(row.get("source_folder", "")) + "_phone"
        extras.append(row)
    return extras


def row_has_voice(row):
    if int(row.get("VOICE_PRESENT", 0)) == 1:
        return True
    return str(row.get("rule", "")).startswith("overlay")


def is_phone_row(row):
    return str(row.get("path", "")).startswith("phone::")


def load_row_audio(row):
    phone = is_phone_row(row)
    work = dict(row)
    if phone:
        work["path"] = work["path"][len("phone::") :]
    if str(work.get("rule", "")).startswith("overlay"):
        audio = overlay_audio(
            load_audio(work["voice_path"]),
            load_audio(work["music_path"]),
            voice_gain=float(work.get("voice_gain", 1.0) or 1.0),
            music_gain=float(work.get("music_gain", 1.0) or 1.0),
        )
    else:
        audio = load_audio(work["path"])
    if phone:
        seed = int(hashlib.md5(str(row["path"]).encode("utf-8")).hexdigest()[:8], 16)
        audio = telephone_channel(audio, seed=seed)
    return audio


def cache_key(row):
    raw = row["path"]
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    name = Path(str(row.get("voice_path", raw))).stem
    return f"{name}_{digest}.pt"


def build_split_rows(
    csv_path,
    max_per_source=None,
    voice_per_source=None,
    overlays=0,
    overlay_seed=42,
    phone_frac=0.0,
    rewrites=None,
):
    rows = [rewrite_row(row, rewrites) for row in read_csv_rows(csv_path)]
    rows = music_rows(rows, max_per_source, voice_per_source)
    rows.extend(add_all_overlays(rows, overlays, overlay_seed))
    rows.extend(add_phone_rows(rows, phone_frac, overlay_seed + 99))
    return rows
