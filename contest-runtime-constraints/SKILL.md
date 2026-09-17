---
name: contest-runtime-constraints
description: Enforces the AI-generated audio detection contest runtime, size, hardware, offline, input, and output constraints. Use when implementing, modifying, reviewing, or optimizing the deepvoice baseline, script.py, submit.zip, model loading, inference, PANNs, HTDemucs, DF-Arena, fusion, or submission.csv.
---

# Contest Runtime Constraints

이 프로젝트의 모든 구현·수정·리뷰는 아래 제약에 맞춰야 한다. 정확도보다 먼저 제출 가능 여부(시간·용량·VRAM·오프라인)를 확인한다.

## Hard limits

| 항목 | 한도 | 설계 함의 |
|------|------|-----------|
| 추론 시간 | 1,200파일 ≤ 60분 | 파일당 평균 ≤ 3.0초. 여유를 보면 목표 ≤ 2.5초 |
| 패키지 설치 | ≤ 10분 | `requirements`는 휠/로컬 설치 위주. 소스 빌드·거대 의존성 추가 금지 |
| 제출 zip | ≤ 10GB | 가중치·모델이 용량 대부분. 중복 체크포인트 금지 |
| 압축 해제 후 | ≤ 32GB | zip보다 커질 수 있는 언팩 산출물도 한도 안 |
| 네트워크 | 패키지 설치 외 오프라인 | 추론 중 Hub/URL 다운로드 금지 |
| CPU | 6 vCPU | DataLoader worker·프로세스 폭증 금지 |
| RAM | 28GB | 전 파일 파형 상주 금지. 파일 단위 스트리밍 |
| GPU | L4 22.4GiB VRAM | 대형 모델 동시 상주 금지. 순차 로드·즉시 해제 |

평가 환경 기본 장치는 CUDA다. CPU 폴백을 전제로 설계하지 않는다.

## Evaluation data (do not assume otherwise)

- 파일 수: 1,200
- 길이: 4초 이상 1분 이하
- 샘플링레이트: 16 kHz로 표준화됨
- 채널: 샘플별 모노/스테레오 모두 존재
- 확장자: MP3, WAV, FLAC 등 다양. 평가 데이터도 다양
- 일부 샘플은 전화채널 오디오

입력 로더는 최소 `.aac .flac .m4a .mp3 .ogg .opus .wav .wma`를 처리해야 한다.

## Outputs

파일마다 0~1 확률 5개:

- `FILE_FAKE_PROB`
- `VOICE_FAKE_PROB`
- `MUSIC_FAKE_PROB`
- `VOICE_PRESENT_PROB`
- `MUSIC_PRESENT_PROB`

제출 양식은 `data/sample_submission.csv`의 `ID` 순서를 그대로 따른다. 출력은 `output/submission.csv`.

## Label rules that constrain fusion

- 음성: 사람 발화 또는 보컬만
- 음악: 보컬 없는 반주·악기음만
- 혼합: 음성+음악 동시 또는 순차. 보컬+반주 노래는 혼합
- AI 생성 음성 또는 음악 → FAKE
- 실제 원천 → REAL
- 음성·음악 중 하나라도 FAKE이면 파일 전체 FAKE
- 품질 개선·잡음 제거·음량 조정 등 성분을 새로 생성하지 않는 후처리만 있으면 REAL

`FILE_FAKE_PROB`는 “한쪽이라도 FAKE이면 파일 FAKE”와 맞아야 한다. 존재 확률이 낮은 성분의 fake 점수가 파일 점수를 지배하지 않게 결합한다. 베이스라인은 `max(VP×VF, MP×MF)`다.

## Offline loading

추론 코드는 로컬 `model/`만 사용한다.

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

- `from_pretrained(..., local_files_only=True)`
- Hugging Face/torch hub 실시간 다운로드 경로를 남기지 않는다
- 새 가중치는 제출물에 포함하여 용량 한도를 다시 계산한다

경로 규약:

- 입력: `data/test/`
- 양식: `data/sample_submission.csv`
- 모델: `model/` (`panns/`, `htdemucs/`, `df_arena_1b/`)
- 출력: `output/submission.csv`

## VRAM / RAM budget

L4 22.4GiB에서 모델 3개를 동시에 올리지 않는다. 베이스라인 패턴을 유지한다.

1. PANNs Cnn14로 전 파일 presence 추론 → `del model` + `torch.cuda.empty_cache()`
2. HTDemucs + DF-Arena 1B로 성분 분리·fake 추론
3. HTDemucs 가중치는 CPU에 두고 `apply_model(..., device=cuda)`로 구간만 GPU에 올린다
4. DF-Arena는 GPU 상주, 세그먼트 단위 추론
5. 파일 하나 처리가 끝나면 해당 파형·분리 결과를 버린다

금지:

- 1,200개 파형을 리스트에 모아 두기
- 스테레오 고샘플레이트 전체를 VRAM에 유지
- 배치를 VRAM 한도 확인 없이 키우기
- 분리 결과를 디스크에 전량 dump (용량·IO 위험). 디버그 외 금지

세그먼트: 16 kHz 기준 `SEGMENT_SAMPLES = 64600` (약 4.04초). 최대 1분 파일은 약 15세그먼트. 마지막 구간은 겹치게 붙인다.

침묵 성분(`RMS < 1e-5`)의 fake 확률은 0.0으로 둔다.

## Time budget

60분 / 1,200파일 = 파일당 3.0초. 모델 로드·워밍업을 빼면 실제 목표 ≈ 2.5초/파일.

현재 파이프라인에서 비싼 구간:

1. HTDemucs 분리 (가장 큼)
2. DF-Arena 세그먼트 추론 × (voice + music)
3. PANNs (32 kHz 리샘플 + 태그)

최적화 우선순위: 분리 비용 감소 → fake 모델 중복 전방 계산 감소 → presence 모델 경량화. 정확도만 올리고 시간 한도를 깨는 변경은 거부한다.

허용 예시:

- Demucs `shifts=0`, `split=True` 유지 또는 더 싸게
- 존재 확률이 매우 낮은 성분은 분리/fake 추론 스킵 (라벨 규칙과 모순되지 않게)
- 모델을 더 작게, 또는 한 패스 공유 백본

금지 예시:

- Demucs `shifts≥1` 앙상블
- 파일당 여러 분리 모델
- 초당 수백 번 리샘플이 반복되는 파이썬 루프
- 설치 10분을 넘는 conda 소스 빌드

## Size budget

zip ≤ 10GB, 언팩 ≤ 32GB. 새 가중치를 넣기 전에 추정한다.

```
추정 zip ≈ sum(모델 파일) + 코드 + 의존성 휠
언팩 ≈ zip 압축률 역산. 안전하게 zip의 2~3배로 본다
```

- 같은 모델을 fp32/fp16 중복 저장하지 않는다
- 토크나이저·데이터셋 캐시·학습 로그를 제출에 넣지 않는다
- `model/` 밖 캐시(`~/.cache/huggingface`)에 의존하지 않는다. 오프라인에서 없다

## Package install budget

설치 ≤ 10분, 이후 인터넷 없음.

- CUDA 휠이 있는 torch / torchaudio만
- 컴파일 필수 패키지 추가 시 설치 시간 근거를 먼저 댄다
- 런타임 `pip install` / `huggingface_hub snapshot_download` 금지

## Change checklist

코드를 바꾸기 전에 전부 통과해야 한다.

- [ ] 1,200 × 최악 1분 오디오로 60분 안인가?
- [ ] 피크 VRAM이 22.4GiB 아래인가? (동시 상주 모델 합)
- [ ] 피크 RAM이 28GB 아래인가?
- [ ] 추론 중 네트워크 호출이 0인가?
- [ ] zip ≤ 10GB, 언팩 ≤ 32GB인가?
- [ ] 신규 패키지 설치 ≤ 10분인가?
- [ ] MP3/WAV/FLAC 등 다중 확장자·모노/스테레오·전화채널을 깨지 않는가?
- [ ] 출력 5개 컬럼과 sample_submission ID 순서가 유지되는가?
- [ ] 한쪽 성분 FAKE → 파일 FAKE 규칙과 fusion이 맞는가?

하나라도 실패하면 해당 변경을 적용하지 말고, 한도 안으로 줄인 대안을 제시한다.

## Baseline invariants

현재 제출 파이프라인의 기본값. 바꿀 때는 위 체크리스트를 다시 돈다.

- 존재: PANNs Cnn14, 32 kHz, AudioSet voice/music 라벨 그룹 max
- 분리: HTDemucs, vocals vs 나머지 합(accompaniment)
- fake: DF-Arena 1B, spoof 라벨 softmax, 세그먼트 max
- fusion: `FILE_FAKE_PROB = max(VP * VF, MP * MF)`
- 리샘플 목표: 존재/fake 모두 16 kHz 원본 로드 후 모델별 변환
- 장치는 CUDA. CUDA 없으면 실패시키는 것이 맞다 (평가 환경에 GPU가 있다)
