#!/usr/bin/env python3
"""매니페스트 → 오버레이 + 채널/전송 aug 학습 행.

train_v2(PANNs) · train_df_adapt(DF LoRA) · train_fusion 이 같은 구성을 쓴다.

## Whispeak(ASVspoof5) vs 이 대회

Whispeak 채널 aug (논문): silence / time-stretch / pitch / MUSAN noise / RIR /
RawBoost / 16k·8k codec / bit-crush / gain / SpecAug — **온라인 직렬**, 각
변환 확률 p_DA≈0.05~0.2. 목적은 **음성 단일 트랙** 전송 강건성.

이 대회 요구: 음성·음악 **단독/공존**, PRESENT + FAKE 4축.
→ Whispeak에 없는 **오버레이(음성×음악 조합)** 가 핵심.
→ 전화·코덱은 CPS용으로 **약하게·실음성 위주**만.
→ 4 kHz band·고비율 오프라인 복제·fake에 동일 채널 삭감은 ADS를 깎음
  (spoof 고주파 단서 제거 = “깨끗한 생성 음성”과 반대 방향).
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
# band는 기본 OFF. 켤 때도 4 kHz(너무 셈) 대신 6 kHz.
BAND_LIMIT_HZ = 6_000
# Whispeak 16k codec에 가깝게: 예전 12 kHz+8bit → 14 kHz+10bit
CODEC_SAMPLE_RATE = 14_000
CODEC_BITS = 10
AUG_PREFIXES = ("phone::", "codec::", "band::", "gain::", "noise::")

REAL_RULES = {"Music_Only", "Voice_and_Music", "Voice_Only", "Fake_Voice_Only"}
FAKE_RULES = {"Fake_Music_Only"}
# 진짜 반주 + 가짜 보컬 믹스 (Drive: Music_FakeVocal_Mix)
MIX_FAKE_VOICE_RULES = {"Music_FakeVocal"}
TRAIN_RULES = REAL_RULES | FAKE_RULES | MIX_FAKE_VOICE_RULES


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
    for prefix in AUG_PREFIXES:
        if str(path_text).startswith(prefix):
            return prefix + rewrite_path(path_text[len(prefix) :], rewrites)
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
    source_l = str(source).lower()
    # Zeroth·전화·AI Hub 콜 — 한국어/채널 도메인 비중
    if voice_per_source and any(
        token in source_l for token in ("zeroth", "phone", "call", "aihub")
    ):
        return voice_per_source
    if mix_per_source and rule in {"Voice_and_Music", "Music_FakeVocal"}:
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
    """읽기 실패·0바이트·깨진 파일은 무음으로 대체 (학습이 죽지 않게)."""
    try:
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            return np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        audio, _ = librosa.load(str(path), sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
    except Exception:
        return np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
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


def band_limit_channel(audio, sr=AUDIO_SAMPLE_RATE, cutoff_hz=BAND_LIMIT_HZ, seed=0):
    """저역 통과(대역 제한). 전화·저품질 업로드에서 고주파 아티팩트가 사라지는 경우를 흉내낸다."""
    rng = np.random.default_rng(int(seed) % (2**32))
    target_sr = max(int(cutoff_hz * 2), 2_000)
    narrow = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    band = librosa.resample(narrow, orig_sr=target_sr, target_sr=sr, res_type="soxr_hq")
    if band.size < audio.size:
        band = np.pad(band, (0, audio.size - band.size))
    else:
        band = band[: audio.size]
    noise = rng.normal(0.0, 0.0015, size=band.size).astype(np.float32)
    return peak_norm(band + noise, 0.75).astype(np.float32)


def codec_channel(audio, sr=AUDIO_SAMPLE_RATE, seed=0):
    """경량 코덱 손상: 14 kHz 왕복 + 10-bit 양자화 (Whispeak codec/bit-crush 약화 이식).

    예전 12 kHz+8bit는 spoof 단서를 과하게 지워 ADS↓ 유발.
    """
    rng = np.random.default_rng(int(seed) % (2**32))
    levels = float(2 ** (CODEC_BITS - 1) - 1)
    narrow = librosa.resample(
        audio, orig_sr=sr, target_sr=CODEC_SAMPLE_RATE, res_type="soxr_hq"
    )
    peak = float(np.max(np.abs(narrow))) + 1e-6
    quantized = np.round(narrow / peak * levels) / levels * peak
    restored = librosa.resample(
        quantized.astype(np.float32),
        orig_sr=CODEC_SAMPLE_RATE,
        target_sr=sr,
        res_type="soxr_hq",
    )
    if restored.size < audio.size:
        restored = np.pad(restored, (0, audio.size - restored.size))
    else:
        restored = restored[: audio.size]
    noise = rng.normal(0.0, 0.0015, size=restored.size).astype(np.float32)
    return peak_norm(restored + noise, 0.75).astype(np.float32)


def gain_channel(audio, sr=AUDIO_SAMPLE_RATE, seed=0):
    """Whispeak Gain(0.25~2.0)의 온화판. 스펙트럼 단서는 거의 유지."""
    del sr  # API 통일
    rng = np.random.default_rng(int(seed) % (2**32))
    gain = float(rng.uniform(0.5, 1.6))
    return peak_norm(audio.astype(np.float32) * gain, 0.95).astype(np.float32)


def noise_channel(audio, sr=AUDIO_SAMPLE_RATE, seed=0):
    """Whispeak MUSAN noise의 대용: SNR 18~30 dB 백색 잡음 (외부 코퍼스 불필요)."""
    del sr
    rng = np.random.default_rng(int(seed) % (2**32))
    snr_db = float(rng.uniform(18.0, 30.0))
    power = float(np.mean(audio.astype(np.float64) ** 2) + 1e-9)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.size).astype(np.float32)
    return peak_norm(audio.astype(np.float32) + noise, 0.9).astype(np.float32)


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


def is_bonafide_content(row):
    """생성·변조가 없는 클립 (채널 삭감을 주로 여기 적용 → CPS, ADS 보호)."""
    return int(row.get("VOICE_FAKE", 0)) == 0 and int(row.get("MUSIC_FAKE", 0)) == 0


def add_channel_aug_rows(
    rows,
    frac,
    seed,
    prefix,
    suffix,
    fake_share=0.25,
):
    """채널 복사본 추가.

    Whispeak는 온라인 p_DA로 약하게 섞음. 우리는 오프라인 복제라 비율을 낮추고,
    fake_share로 **실(bonafide) 위주**에만 채널 삭감을 걸어 spoof 단서 파괴를 막는다.
    """
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
        and not any(str(row.get("path", "")).startswith(p) for p in AUG_PREFIXES)
    ]
    if not eligible:
        return []
    bona = [i for i in eligible if is_bonafide_content(rows[i])]
    fake = [i for i in eligible if not is_bonafide_content(rows[i])]
    n_fake = min(int(round(n * float(fake_share))), len(fake), n)
    n_bona = min(n - n_fake, len(bona))
    # 풀이 부족하면 반대쪽·전체에서 보충
    chosen = []
    if n_bona > 0:
        chosen.extend(rng.choice(bona, size=n_bona, replace=False).tolist())
    if n_fake > 0:
        chosen.extend(rng.choice(fake, size=n_fake, replace=False).tolist())
    remain = n - len(chosen)
    if remain > 0:
        leftover = [i for i in eligible if i not in set(chosen)]
        if leftover:
            take = min(remain, len(leftover))
            chosen.extend(rng.choice(leftover, size=take, replace=False).tolist())
    extras = []
    for index in chosen:
        row = dict(rows[int(index)])
        row["path"] = prefix + str(row["path"])
        row["source_folder"] = str(row.get("source_folder", "")) + suffix
        extras.append(row)
    return extras


def add_phone_rows(rows, frac, seed, fake_share=0.25):
    return add_channel_aug_rows(rows, frac, seed, "phone::", "_phone", fake_share=fake_share)


def add_codec_rows(rows, frac, seed, fake_share=0.25):
    return add_channel_aug_rows(rows, frac, seed, "codec::", "_codec", fake_share=fake_share)


def add_band_rows(rows, frac, seed, fake_share=0.15):
    # band는 spoof 고주파를 가장 많이 지움 → fake_share 더 낮게
    return add_channel_aug_rows(rows, frac, seed, "band::", "_band", fake_share=fake_share)


def add_gain_rows(rows, frac, seed, fake_share=0.5):
    """Whispeak gain: 단서 보존형 → fake에도 상대적으로 더 허용."""
    return add_channel_aug_rows(rows, frac, seed, "gain::", "_gain", fake_share=fake_share)


def add_noise_rows(rows, frac, seed, fake_share=0.35):
    return add_channel_aug_rows(rows, frac, seed, "noise::", "_noise", fake_share=fake_share)


def row_has_voice(row):
    if int(row.get("VOICE_PRESENT", 0)) == 1:
        return True
    return str(row.get("rule", "")).startswith("overlay")


def strip_aug_prefix(path_text):
    text = str(path_text)
    for prefix in AUG_PREFIXES:
        if text.startswith(prefix):
            return prefix.rstrip(":"), text[len(prefix) :]
    return None, text


def is_phone_row(row):
    kind, _ = strip_aug_prefix(row.get("path", ""))
    return kind == "phone" or str(row.get("path", "")).startswith("phone::")


def load_row_audio(row):
    kind, bare_path = strip_aug_prefix(row.get("path", ""))
    work = dict(row)
    work["path"] = bare_path
    if str(work.get("rule", "")).startswith("overlay"):
        audio = overlay_audio(
            load_audio(work["voice_path"]),
            load_audio(work["music_path"]),
            voice_gain=float(work.get("voice_gain", 1.0) or 1.0),
            music_gain=float(work.get("music_gain", 1.0) or 1.0),
        )
    else:
        audio = load_audio(work["path"])
    if kind is not None:
        seed = int(hashlib.md5(str(row["path"]).encode("utf-8")).hexdigest()[:8], 16)
        if kind == "phone":
            audio = telephone_channel(audio, seed=seed)
        elif kind == "codec":
            audio = codec_channel(audio, seed=seed)
        elif kind == "band":
            audio = band_limit_channel(audio, seed=seed)
        elif kind == "gain":
            audio = gain_channel(audio, seed=seed)
        elif kind == "noise":
            audio = noise_channel(audio, seed=seed)
    return audio


def cache_key(row):
    raw = row["path"]
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    name = Path(str(row.get("voice_path", raw))).stem
    return f"{name}_{digest}.pt"


def is_domain_focus_row(row):
    """DF 샘플 가중: 오버레이·한국어·실전화·TTS.

    채널 aug 복사본 전체에 domain_repeat를 걸면(=예전) 삭감 분포가 2배로 증폭됨.
    채널 접두사는 focus에서 제외한다.
    """
    source = str(row.get("source_folder", "")).lower()
    rule = str(row.get("rule", ""))
    path = str(row.get("path", ""))
    if any(path.startswith(p) for p in AUG_PREFIXES):
        return False
    if any(token in source for token in ("phone", "call", "zeroth", "tts", "mix")):
        return True
    if rule == "Voice_and_Music" or rule.startswith("overlay"):
        return True
    if rule in {"Fake_Voice_Only", "Music_FakeVocal"}:
        return True
    return False


def build_split_rows(
    csv_path,
    max_per_source=None,
    voice_per_source=None,
    overlays=0,
    overlay_seed=42,
    phone_frac=0.0,
    codec_frac=0.0,
    band_frac=0.0,
    gain_frac=0.0,
    noise_frac=0.0,
    channel_fake_share=0.25,
    rewrites=None,
):
    rows = [rewrite_row(row, rewrites) for row in read_csv_rows(csv_path)]
    rows = music_rows(rows, max_per_source, voice_per_source)
    # 1) 대회 과제 핵심: 음성×음악 조합 (Whispeak에 없음)
    rows.extend(add_all_overlays(rows, overlays, overlay_seed))
    # 2) 전송 강건성: 약·실음성 위주 (Whispeak online DA의 오프라인 근사)
    rows.extend(
        add_phone_rows(rows, phone_frac, overlay_seed + 99, fake_share=channel_fake_share)
    )
    rows.extend(
        add_codec_rows(rows, codec_frac, overlay_seed + 199, fake_share=channel_fake_share)
    )
    rows.extend(
        add_band_rows(
            rows, band_frac, overlay_seed + 299, fake_share=min(channel_fake_share, 0.15)
        )
    )
    rows.extend(
        add_gain_rows(rows, gain_frac, overlay_seed + 399, fake_share=0.5)
    )
    rows.extend(
        add_noise_rows(rows, noise_frac, overlay_seed + 499, fake_share=0.35)
    )
    return rows
