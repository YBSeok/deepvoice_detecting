# deepvoice_detecting

AI 생성 음성·음악 탐지. 제출 필드 5개: `FILE_FAKE`, `VOICE_FAKE`, `MUSIC_FAKE`, `VOICE_PRESENT`, `MUSIC_PRESENT`.

## 리더보드

최신이 위. 점수는 제출 후 기입.

| 버전 | 날짜 | 점수 | 비고 |
|------|------|------|------|
| v2.3 | 2026-09-21 | | phone/codec/band aug + DF Conformer LoRA 도메인 적응 |
| v2.2 | 2026-09-18 | | VF 양방향 게이트 + FILE noisy-OR. 전화·실믹스 DF 오탐 억제 |
| v2.1 | 2026-09-18 | | VF: DF 기본, PANNs는 DF도 높을 때만 가산 (DF 오탐은 못 깎음) |
| v2 | 2026-09-18 | 0.684 | PANNs VF/MF + DF-Arena VF `max`. ADS 하락 |
| v1 | 2026-09-17 | | 원본 믹스 인코딩 + 마스크 풀링 |
| v0 | | | 베이스라인 (PANNs + Demucs 스템 + DF-Arena) |

## 제출

```text
submit.zip  ← script.py + df_lora.py + requirements.txt
            + model/ (PANNs, HTDemucs, DF-Arena, mf_head.pt, vf_head.pt, 선택 df_arena_lora.pt)
```

```bash
python train_v2.py --train-csv data/manifests/train.csv --valid-csv data/manifests/valid.csv --ckpt model/mf_head.pt --vf-ckpt model/vf_head.pt --max-per-source 400 --voice-per-source 1200 --overlays 400 --phone-frac 0.3 --codec-frac 0.2 --band-frac 0.15

python train_df_adapt.py --train-csv data/manifests/train.csv --valid-csv data/manifests/valid.csv --out model/df_arena_lora.pt --max-per-source 200 --voice-per-source 800 --overlays 200 --phone-frac 0.35 --codec-frac 0.25 --band-frac 0.2 --domain-repeat 2 --epochs 3 --batch-size 2 --mode lora

python script.py --test-dir data/test --sample-submission data/sample_submission.csv --output output/submission.csv
```

## 변경 이력

### v2.3 — 채널 aug + DF LoRA 도메인 적응

- `learning_data.py`: phone(8 kHz)에 더해 **codec**(12 kHz+8bit), **band**(4 kHz) 복사본 행
- `train_v2.py`: `--codec-frac` / `--band-frac`으로 PANNs 헤드 재학습 분포 확장
- `train_df_adapt.py` + `df_lora.py`: SSL 고정, Conformer 마지막 N블록(+fc5) **LoRA**
- `script.py`: `model/df_arena_lora.pt`가 있으면 VF 경로에 자동 로드. 없으면 기존 frozen DF
- Colab 노트북: `colab_train_df_lora.ipynb`

### v2.2 — DF 오탐 억제 (fusion만 변경)

DF-Arena 가중치는 고정(오픈 체크포인트). 재학습 대신 **점수 결합**만 바꿈.

- VF: 양방향 게이트
  - `panns ≥ df` → `df + (panns−df)·df` (이전과 동일, 가산)
  - `df > panns` → `panns + (df−panns)·panns` (**DF만 높은 오탐 억제**)
- FILE: `1 − (1−VP×VF)(1−MP×MF)` (noisy-OR)
- 학습 데이터: `Voice_Only_Phone` + Zeroth를 `voice-per-source`로 더 넣고, `phone-frac`으로 8 kHz 왕복 증강

로컬 프로브(케이스당 8클립): 전화 실음성 VF hit 0.50→0.88, 실믹스 VF hit 0.00→1.00. 가짜 음성 VF hit 1.00→0.75.

### v2.1 — 단방향 게이트

- VF: `df + (max(panns, df)−df)·df`. PANNs는 DF가 높을 때만 가산
- FILE: `max(VP×VF, MP×MF)`
- DF 단독 오탐(전화·실믹스)은 그대로 통과함

### v2 — 학습된 VF/MF 헤드

- VP/MP: 믹스 → PANNs
- MF: 믹스 → PANNs 임베딩 → `mf_head.pt`
- VF: DF-Arena + PANNs `vf_head` (`max` → 실음성 오탐으로 ADS 0.694→0.650)
- FILE: `max(VP×VF, MP×MF)`
- 가짜/진짜 음성·음악·오버레이·한국어 TTS/Zeroth로 헤드 학습. MF는 믹스 기준

### v1 — 원본 인코딩 + 프레임 풀링

- Demucs는 재합성 스템이 아니라 프레임 마스크만
- VF/MF: 원본 믹스 1회 인코딩 후 마스크 풀링
- FILE: `max(VP×VF, MP×MF)`

### v0 — 베이스라인

- VP/MP: PANNs
- VF/MF: Demucs 보컬/반주 스템 → DF-Arena
- FILE: `max(VP×VF, MP×MF)`
