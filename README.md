# deepvoice_detecting

AI 생성 음성·음악 탐지. 제출 필드 5개: `FILE_FAKE`, `VOICE_FAKE`, `MUSIC_FAKE`, `VOICE_PRESENT`, `MUSIC_PRESENT`.

## 리더보드

최신이 위. 점수는 제출 후 기입.

| 버전 | 날짜 | 점수 | 비고 |
|------|------|------|------|
| v2.2 | 2026-09-18 | | VF 양방향 게이트 + FILE noisy-OR. 전화·실믹스 DF 오탐 억제 |
| v2.1 | 2026-09-18 | | VF: DF 기본, PANNs는 DF도 높을 때만 가산 (DF 오탐은 못 깎음) |
| v2 | 2026-09-18 | 0.684 | PANNs VF/MF + DF-Arena VF `max`. ADS 하락 |
| v1 | 2026-09-17 | | 원본 믹스 인코딩 + 마스크 풀링 |
| v0 | | | 베이스라인 (PANNs + Demucs 스템 + DF-Arena) |

## 제출

```text
submit.zip  ← script.py + requirements.txt + model/ (PANNs, HTDemucs, DF-Arena, mf_head.pt, vf_head.pt)
```

로컬 학습·추론:

```bash
.\.venv\Scripts\python.exe train_v2.py --train-csv data/manifests/train.csv --valid-csv data/manifests/valid.csv --ckpt model/mf_head.pt --vf-ckpt model/vf_head.pt --max-per-source 400 --voice-per-source 1200 --overlays 400 --phone-frac 0.3

.\.venv\Scripts\python.exe script.py --test-dir data/test --sample-submission data/sample_submission.csv --output output/submission.csv
```

프로브 진단:

```bash
.\.venv\Scripts\python.exe diagnose_heads.py --per-case 8 --run-vf --vf-device cuda
```

## 변경 이력

### v2.2 — DF 오탐 억제 (fusion만 변경)

DF-Arena 가중치는 고정(오픈 체크포인트). 재학습 대신 **점수 결합**만 바꿈.

- VF: 양방향 게이트
  - `panns ≥ df` → `df + (panns−df)·df` (이전과 동일, 가산)
  - `df > panns` → `panns + (df−panns)·panns` (**DF만 높은 오탐 억제**)
- FILE: `1 − (1−VP×VF)(1−MP×MF)` (noisy-OR). 혼합에서 `max`만 쓸 때 과소평가되던 경우 보완
- 학습 데이터: `Voice_Only_Phone` + Zeroth를 `voice-per-source`로 더 넣고, `phone-frac`으로 8 kHz 왕복 증강. PANNs VF는 전화·실믹스에서 이미 낮게 잘 봄 → fusion이 DF 오탐을 깎아 줘야 효과가 남

로컬 프로브(케이스당 8클립) 요약: 전화 실음성 VF hit 0.50→0.88, 실믹스 VF hit 0.00→1.00. 가짜 음성 VF hit는 1.00→0.75로 소폭 희생.

### v2.1 — 단방향 게이트

- VF: `df + (max(panns, df)−df)·df`. PANNs는 DF가 높을 때만 가산
- FILE: `max(VP×VF, MP×MF)`
- DF 단독 오탐(전화·실믹스)은 그대로 통과함

### v2 — 학습된 VF/MF 헤드

추론은 원본 믹스 기준. Demucs 줄기는 DF-Arena 마스크용으로만 쓴다.

- VP/MP: 믹스 → PANNs (v1과 동일)
- MF: 믹스 → PANNs 임베딩 → `mf_head.pt`
- VF: DF-Arena + PANNs `vf_head`. v2는 `max(panns, df)` → 실음성 오탐 합집합으로 ADS 0.694→0.650
- FILE_FAKE: `max(VP×VF, MP×MF)`

학습에서 v1 대비 바꾼 점:

- 가짜 악기만이 아니라 진짜 음성·진짜 노래도 MF=0으로 학습 (실음성/실곡 오탐 감소)
- `Fake_Voice_Only_TTS_ko`로 VF=1, `Voice_Only_Zeroth`로 한국어 실음성 VF=0
- 부분 fake 오버레이: 진짜음성+가짜음악, 가짜음성+진짜/가짜음악, 한국어실음성+음악
- MF는 반주 줄기가 아니라 믹스로 학습·추론을 맞춤

### v1 — 원본 인코딩 + 프레임 풀링

- VP/MP: 믹스 → PANNs (v0과 동일)
- Demucs는 재합성 스템을 탐지에 넣지 않고, 프레임 가중치만 생성
- VF/MF: 원본 믹스를 DF-Arena에 1회 인코딩한 뒤 마스크로 풀링
- FILE_FAKE: `max(VP×VF, MP×MF)` (v0과 동일)

지문을 지키기 위해 인코더 입력 파형은 수정하지 않음.

### v0 — 베이스라인

- VP/MP: 믹스 → PANNs
- VF/MF: Demucs 보컬/반주 스템을 DF-Arena에 각각 입력
- FILE_FAKE: `max(VP×VF, MP×MF)`
