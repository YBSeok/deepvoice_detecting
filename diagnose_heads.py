#!/usr/bin/env python3
"""VF / MF / FILE 중 어디가 죽는지 가른다.

정답이 있는 작은 프로브셋을 만들고 현재 script.py 파이프라인으로 추론한다.

  .\\.venv\\Scripts\\python.exe diagnose_heads.py --per-case 8

판독
  fake_music에서 MF가 낮음          -> MF 헤드가 죽음
  fake_music에서 MF는 높은데 FILE낮음 -> 합치기(MP x MF)가 죽음
  mix_rv_fm에서 MF가 낮음           -> 분리 후 MF가 죽음
  mix_rv_fm에서 MF는 높은데 FILE낮음 -> 합치기가 죽음
  real_* 에서 VF/MF/FILE이 높음     -> 오탐
  fake_voice에서 VF가 낮음          -> VF 헤드가 죽음
  mix_fv_rm에서 VF가 낮음           -> 가짜음성+진짜음악에서 VF가 죽음
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from script import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_MF_CKPT,
    DEFAULT_VF_CKPT,
    FourHeadPredictor,
    SUPPORTED_AUDIO_EXTENSIONS,
    fuse_file_fake,
    select_device,
)


PROBE_DIR = Path("data") / "probe"

SOURCES = {
    "real_voice": Path(r"D:\Voice_Only_LibriSpeech"),
    "real_voice_ko": Path(r"D:\Voice_Only_Zeroth"),
    "real_music": Path(r"D:\Music_Only_FMA"),
    "fake_music": Path(r"D:\Fake_Music_Only_Suno"),
    "real_mix": Path(r"D:\Voice_and_Music_FMA"),
    "fake_voice": Path(r"D:\Fake_Voice_Only_TTS_ko"),
}

# 기대값: 높음=1, 낮음=0. FILE은 규칙상 한쪽 fake면 1.
EXPECTED = {
    "real_voice": {"VF": 0, "MF": 0, "FILE": 0},
    "real_voice_ko": {"VF": 0, "MF": 0, "FILE": 0},
    "real_music": {"VF": 0, "MF": 0, "FILE": 0},
    "fake_music": {"VF": 0, "MF": 1, "FILE": 1},
    "real_mix": {"VF": 0, "MF": 0, "FILE": 0},
    "fake_voice": {"VF": 1, "MF": 0, "FILE": 1},
    "mix_rv_fm": {"VF": 0, "MF": 1, "FILE": 1},
    "mix_ko_fm": {"VF": 0, "MF": 1, "FILE": 1},
    "mix_fv_rm": {"VF": 1, "MF": 0, "FILE": 1},
}


def list_audio(folder: Path):
    files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    files.sort()
    return files


def pick(files, count, rng):
    if len(files) <= count:
        return list(files)
    return rng.sample(files, count)


def load_mono(path: Path, seconds=8.0):
    audio, _ = librosa.load(path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
    if audio.size == 0:
        return None
    limit = int(seconds * AUDIO_SAMPLE_RATE)
    if audio.size > limit:
        audio = audio[:limit]
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak * 0.8
    return audio


def overlay(voice, music):
    length = min(voice.size, music.size)
    mix = voice[:length] + music[:length]
    peak = float(np.max(np.abs(mix)))
    if peak > 1e-6:
        mix = mix / peak * 0.9
    return mix.astype(np.float32)


def write_wav(path: Path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, AUDIO_SAMPLE_RATE)


def build_probe(per_case, seed):
    rng = random.Random(seed)
    cases = defaultdict(list)

    for name in ("real_voice", "real_voice_ko", "real_music", "fake_music", "real_mix", "fake_voice"):
        folder = PROBE_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        chosen = pick(list_audio(SOURCES[name]), per_case, rng)
        for index, source in enumerate(chosen):
            dest = folder / f"{name}_{index:02d}{source.suffix.lower()}"
            if not dest.is_file():
                audio = load_mono(source)
                if audio is None:
                    continue
                write_wav(dest.with_suffix(".wav"), audio)
                dest = dest.with_suffix(".wav")
            cases[name].append(dest)

    voice_files = pick(list_audio(SOURCES["real_voice"]), per_case, rng)
    music_files = pick(list_audio(SOURCES["fake_music"]), per_case, rng)
    mix_folder = PROBE_DIR / "mix_rv_fm"
    mix_folder.mkdir(parents=True, exist_ok=True)
    for index, (voice_path, music_path) in enumerate(zip(voice_files, music_files)):
        dest = mix_folder / f"mix_rv_fm_{index:02d}.wav"
        if not dest.is_file():
            voice = load_mono(voice_path)
            music = load_mono(music_path)
            if voice is None or music is None:
                continue
            write_wav(dest, overlay(voice, music))
        cases["mix_rv_fm"].append(dest)

    fake_voices = pick(list_audio(SOURCES["fake_voice"]), per_case, rng)
    real_musics = pick(list_audio(SOURCES["real_music"]), per_case, rng)
    mix_fv_folder = PROBE_DIR / "mix_fv_rm"
    mix_fv_folder.mkdir(parents=True, exist_ok=True)
    for index, (voice_path, music_path) in enumerate(zip(fake_voices, real_musics)):
        dest = mix_fv_folder / f"mix_fv_rm_{index:02d}.wav"
        if not dest.is_file():
            voice = load_mono(voice_path)
            music = load_mono(music_path)
            if voice is None or music is None:
                continue
            write_wav(dest, overlay(voice, music))
        cases["mix_fv_rm"].append(dest)

    ko_files = pick(list_audio(SOURCES["real_voice_ko"]), per_case, rng)
    ko_music = pick(list_audio(SOURCES["fake_music"]), per_case, rng)
    mix_ko_folder = PROBE_DIR / "mix_ko_fm"
    mix_ko_folder.mkdir(parents=True, exist_ok=True)
    for index, (voice_path, music_path) in enumerate(zip(ko_files, ko_music)):
        dest = mix_ko_folder / f"mix_ko_fm_{index:02d}.wav"
        if not dest.is_file():
            voice = load_mono(voice_path)
            music = load_mono(music_path)
            if voice is None or music is None:
                continue
            write_wav(dest, overlay(voice, music))
        cases["mix_ko_fm"].append(dest)

    return cases


def mean_scores(rows):
    if not rows:
        return {key: 0.0 for key in ("VP", "MP", "VF", "MF", "FILE")}
    keys = ("VP", "MP", "VF", "MF", "FILE")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def hit_rate(rows, expected, threshold=0.5):
    vf_ok = mf_ok = file_ok = 0
    for row in rows:
        vf_pred = int(row["VF"] >= threshold)
        mf_pred = int(row["MF"] >= threshold)
        file_pred = int(row["FILE"] >= threshold)
        vf_ok += vf_pred == expected["VF"]
        mf_ok += mf_pred == expected["MF"]
        file_ok += file_pred == expected["FILE"]
    n = max(len(rows), 1)
    return vf_ok / n, mf_ok / n, file_ok / n


def diagnose(summary):
    print("\n판독")
    fake = summary.get("fake_music")
    mix = summary.get("mix_rv_fm")
    if fake:
        if fake["MF"] < 0.5:
            print("- MF 헤드: 가짜 악기만 있는 클립을 못 잡음")
        elif fake["FILE"] < 0.5:
            print("- 합치기: MF는 맞는데 FILE이 안 올라감 (MP x MF)")
        else:
            print("- 가짜 악기만 있는 클립: MF/FILE은 동작")
    if mix:
        if mix["MF"] < 0.5:
            print("- 혼합(진짜음성+가짜음악): 분리 후 MF가 죽음")
        elif mix["FILE"] < 0.5:
            print("- 혼합: MF는 맞는데 FILE 합치기가 죽음")
        else:
            print("- 혼합(진짜음성+가짜음악): MF/FILE은 동작")
    real_fp = []
    for name in ("real_voice", "real_voice_ko", "real_music", "real_mix"):
        row = summary.get(name)
        if row and row["FILE"] >= 0.5:
            real_fp.append(name)
    if real_fp:
        print("- 오탐: 진짜 클립에서 FILE이 높음 ->", ", ".join(real_fp))
    ko = summary.get("real_voice_ko")
    if ko:
        if ko["VF"] >= 0.5:
            print("- 오탐: 한국어 실음성(Zeroth)에서 VF가 높음")
        else:
            print("- 한국어 실음성(Zeroth): VF 오탐은 낮음")
    mix_ko = summary.get("mix_ko_fm")
    if mix_ko:
        if mix_ko["MF"] < 0.5:
            print("- 혼합(한국어실음성+가짜음악): MF가 죽음")
        elif mix_ko["VF"] >= 0.5:
            print("- 혼합(한국어실음성+가짜음악): VF 오탐")
        else:
            print("- 혼합(한국어실음성+가짜음악): MF/VF는 동작")
    fake_voice = summary.get("fake_voice")
    if fake_voice:
        if fake_voice["VF"] < 0.5:
            print("- VF 헤드: 가짜 음성만 있는 클립을 못 잡음")
        elif fake_voice["FILE"] < 0.5:
            print("- 합치기: VF는 맞는데 FILE이 안 올라감 (VP x VF)")
        else:
            print("- 가짜 음성만 있는 클립: VF/FILE은 동작")
    mix_fv = summary.get("mix_fv_rm")
    if mix_fv:
        if mix_fv["VF"] < 0.5:
            print("- 혼합(가짜음성+진짜음악): VF가 죽음")
        elif mix_fv["FILE"] < 0.5:
            print("- 혼합: VF는 맞는데 FILE 합치기가 죽음")
        else:
            print("- 혼합(가짜음성+진짜음악): VF/FILE은 동작")


def parse_args():
    parser = argparse.ArgumentParser(description="Probe VF/MF/FILE failure mode.")
    parser.add_argument("--per-case", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mf-ckpt", type=Path, default=DEFAULT_MF_CKPT)
    parser.add_argument("--vf-ckpt", type=Path, default=DEFAULT_VF_CKPT)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--vf-device",
        choices=["cuda", "cpu"],
        default="cpu",
        help="5060 Ti에서 DF-Arena CUDA 로드가 죽어서 기본은 CPU",
    )
    parser.add_argument(
        "--skip-vf",
        action="store_true",
        default=True,
        help="로컬에서 DF-Arena forward가 죽으면 VF=0으로 두고 MF/FILE만 본다",
    )
    parser.add_argument("--run-vf", action="store_true", help="VF도 시도한다")
    return parser.parse_args()


def main():
    args = parse_args()
    cases = build_probe(args.per_case, args.seed)
    audio_files = []
    owner = {}
    for name, paths in cases.items():
        for path in paths:
            audio_files.append(path)
            owner[path.stem] = name

    print("프로브 클립")
    for name, paths in cases.items():
        print(f"  {name}: {len(paths)}")

    predictor = FourHeadPredictor(
        select_device(args.device),
        args.mf_ckpt,
        args.vf_ckpt,
        vf_device=torch.device(args.vf_device),
    )
    skip_vf = not args.run_vf
    if skip_vf:
        print("DF-Arena는 이 PC에서 죽어서 건너뜀. PANNs VF/MF로 측정.")
    presence = predictor.run_presence_heads(audio_files)
    outputs = predictor.run_fake_heads(audio_files, presence, skip_vf=skip_vf)

    grouped = defaultdict(list)
    for path in audio_files:
        heads = outputs[path.stem]
        row = {
            "id": path.stem,
            "VP": heads["VOICE_PRESENT_PROB"],
            "MP": heads["MUSIC_PRESENT_PROB"],
            "VF": heads["VOICE_FAKE_PROB"],
            "MF": heads["MUSIC_FAKE_PROB"],
            "FILE": fuse_file_fake(heads),
        }
        grouped[owner[path.stem]].append(row)

    print("\nVF mismatch")
    for name in EXPECTED:
        expected_vf = EXPECTED[name]["VF"]
        for row in grouped.get(name, []):
            pred = int(row["VF"] >= 0.5)
            if pred != expected_vf:
                print(
                    f"  {row['id']}\t{name}\tVF={row['VF']:.3f}\texpected={expected_vf}"
                )

    print("\ncase\tn\tVP\tMP\tVF\tMF\tFILE\tVF맞음\tMF맞음\tFILE맞음")
    summary = {}
    for name in EXPECTED:
        rows = grouped.get(name, [])
        avg = mean_scores(rows)
        summary[name] = avg
        vf_hit, mf_hit, file_hit = hit_rate(rows, EXPECTED[name])
        print(
            f"{name}\t{len(rows)}\t"
            f"{avg['VP']:.3f}\t{avg['MP']:.3f}\t"
            f"{avg['VF']:.3f}\t{avg['MF']:.3f}\t{avg['FILE']:.3f}\t"
            f"{vf_hit:.2f}\t{mf_hit:.2f}\t{file_hit:.2f}"
        )
    diagnose(summary)


if __name__ == "__main__":
    main()
