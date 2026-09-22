# deepvoice_detecting

AI 생성 음성·음악 탐지. 제출 필드 5개: `FILE_FAKE`, `VOICE_FAKE`, `MUSIC_FAKE`, `VOICE_PRESENT`, `MUSIC_PRESENT`.

## 리더보드

최신이 위. 점수는 제출 후 기입.

| 버전 | 날짜 | 총점 | ADS | CPS | 비고 |
|------|------|------|-----|-----|------|
| v2.5 | 2026-09-22 | | | | aug 재설계(Whispeak 대비·과제 정합). fusion은 **재학습 후** 사용 |
| v2.4 | 2026-09-22 | 0.654 | 0.617 | 0.989 | 학습 VF fusion + v2.3 LoRA. v2.3 대비 ADS 소폭↑, 총점↑ (CPS 동일) |
| v2.3 | 2026-09-22 | 0.652 | 0.615 | 0.989 | LoRA + 강한 채널 aug. **ADS↓ → 총점↓** (CPS는 v2.2와 동일) |
| v2.2 | 2026-09-18 | 0.661 | 0.625 | 0.989 | VF 양방향 게이트 + FILE noisy-OR. 전화·실믹스 DF 오탐 억제 |
| v2.1 | 2026-09-18 | | | | VF: DF 기본, PANNs는 DF도 높을 때만 가산 (DF 오탐은 못 깎음) |
| v2 | 2026-09-18 | 0.684 | | | PANNs VF/MF + DF-Arena VF `max`. ADS 하락 |
| v1 | 2026-09-17 | 0.724 | 0.694 | 0.989 | 원본 믹스 인코딩 + 마스크 풀링 |
| v0 | | | | | 베이스라인 (PANNs + Demucs 스템 + DF-Arena) |

## 제출

```text
submit.zip  ← script.py + df_lora.py + requirements.txt + heads/
            + model/ (PANNs, HTDemucs, DF-Arena, mf_head.pt, vf_head.pt,
                      df_arena_lora.pt, vf_fusion.pt)
            # v2.5: 새 aug로 헤드·LoRA·fusion 재학습 후 zip
            # v2.4: 구 LoRA 위 fusion (재학습 전 패키지)
```

```bash
# pjs 폴더 기준 train/valid/test 매니페스트 (기본 출력: <root>/manifests)
python -m preprocess.build_manifest --root /path/to/pjs --valid-ratio 0.1 --test-ratio 0.1

# v2.5 권장 학습 순서
python train_v2.py --train-csv data/manifests/train.csv --valid-csv data/manifests/valid.csv --ckpt model/mf_head.pt --vf-ckpt model/vf_head.pt --max-per-source 400 --voice-per-source 1200 --overlays 500 --phone-frac 0.12 --codec-frac 0.08 --band-frac 0 --gain-frac 0.1 --noise-frac 0.1

python train_df_adapt.py --train-csv data/manifests/train.csv --valid-csv data/manifests/valid.csv --out model/df_arena_lora.pt --max-per-source 200 --voice-per-source 800 --overlays 300 --phone-frac 0.12 --codec-frac 0.08 --band-frac 0 --gain-frac 0.1 --noise-frac 0.1 --domain-repeat 1 --epochs 3 --batch-size 2 --mode lora

python train_fusion.py --train-csv data/manifests/train.csv --valid-csv data/manifests/valid.csv --df-lora model/df_arena_lora.pt --vf-ckpt model/vf_head.pt --out model/vf_fusion.pt --phone-frac 0.12 --codec-frac 0.08 --band-frac 0 --gain-frac 0.1 --noise-frac 0.1

python pack_submit.py --out submit.zip

python script.py --test-dir data/test --sample-submission data/sample_submission.csv --output output/submission.csv
# fusion 없이 asym gate만: --vf-gate asym  (vf_fusion.pt 제거/미포함 시)
```

## 변경 이력

### v2.5 — aug 재설계 (Whispeak 대비 · 과제 정합)

음성·음악 **단독/공존** + PRESENT/FAKE 4축에 맞게 학습 분포를 다시 잡음.
v2.3~2.4의 강한 채널 aug가 ADS를 깎은 문제를 직접 고친다.

#### Aug: Whispeak vs 과제 vs 우리

| | Whispeak (ASVspoof5) | 이 대회 과제 | v2.3~2.4 (旧) | v2.5 (新) |
|--|---------------------|-------------|---------------|-----------|
| 목표 | 음성 단일·전송 강건 | 음성/음악 단독·공존 + PRESENT/FAKE | 전화 CPS | 과제 정합 + ADS 보호 |
| 핵심 | online 직렬 p_DA≈0.05~0.2 | **오버레이(음성×음악)** | phone/codec/**band 고비율** | **overlays↑**, band **OFF** |
| 채널 | codec/bitcrush/gain/noise/RIR… | 전화는 CPS용 보조 | fake에도 동일 삭감 | **실(bonafide) 위주** 채널 삭감 |
| 위험 | OOD 일반화 한계(논문도 인정) | 고주파 spoof 단서 | band 4kHz·~50% 복제 → ADS↓ | gain/noise는 단서 보존형만 |

조치:
- **제거/기본 OFF**: `band-frac=0`
- **축소**: phone 0.35→**0.12**, codec 0.25→**0.08**(14 kHz+10bit), domain-repeat 2→**1**
- **추가**: Whispeak식 **gain / 약한 noise**
- **강화**: overlays 기본 ↑
- 채널 aug 복사본은 domain-focus에서 **제외**

#### `vf_fusion`이 v2.5에 어울리는가?

**구조적으로는 맞다. 체크포인트는 당장 그대로 쓰면 안 된다.**

| 관점 | 판정 | 이유 |
|------|------|------|
| 역할 | 적합 | PANNs VF ↔ DF VF 비대칭 결합(ADS↑, 실음성 FP↓)은 v2.5 목표와 동일 |
| 입력 | 적합 | `[panns, df, vp, panns·df, \|panns−df\|]` — aug가 바뀌어도 스칼라 결합기는 그대로 유효 |
| 과제(음성+음악) | 부분 적합 | fusion은 **VF만** 담당. 음악·오버레이는 MF/FILE noisy-OR + 데이터 쪽. fusion이 음악 경로를 대체하진 않음 |
| 기존 `vf_fusion.pt` | **부적합(재학습 필요)** | v2.3 LoRA·강한 채널 점수 분포로 학습됨. v2.5 LoRA/헤드 재학습 후 점수 스케일이 바뀌면 게이트가 어긋남 |
| 대안 | 임시 OK | 재학습 전: `--vf-gate asym` (로컬 A/B에서 CPS 여유↑). 재학습 후: fusion 재학습 → 제출 |

권장 파이프라인: **헤드(v2) → DF LoRA → vf_fusion** 을 v2.5 aug로 한 세트 재학습.
재학습 전 제출이면 fusion을 빼고 **asym_gate + (새/구) LoRA**가 더 안전.

### v2.4 — 학습 VF fusion + LoRA (제출)

- `heads/vf_fusion.py` + `train_fusion.py`: PANNs VF · DF VF · VP → 작은 MLP
- `script.py`: `model/vf_fusion.pt`가 있으면 gate 대신 학습 fusion
- `pack_submit.py`: **vf_fusion.pt 포함**
- LB형 A/B (구 LoRA 기준): fusion+lora fake_hit **0.91** / real_hit 0.92  
  asym_gate+lora는 fake 0.89 / real 0.98 → CPS 여유 더 큼

리더보드: 총점 **0.654** / ADS **0.617** / CPS **0.989** (v2.3 대비 ADS +0.002, 총점 +0.002. CPS는 v2.2·v2.3과 동일).
v2.2(0.661)에는 아직 못 미침 → fusion만으로는 ADS 회복이 제한적.
### v2.3 — 채널 aug + DF LoRA 도메인 적응

- `learning_data.py`: phone(8 kHz)에 더해 **codec**(12 kHz+8bit), **band**(4 kHz) 복사본 행
- `train_v2.py` / `train_df_adapt.py`: 강한 채널 비중 + LoRA
- VF: **v2.2 양방향 게이트** (`vf_fusion.pt` 없음)

리더보드: 총점 **0.652** / ADS **0.615** / CPS **0.989** (v2.2 대비 ADS −0.010).
CPS는 v2.2와 동일 → **실음성 억제는 유지**, **가짜 탐지(ADS)만 악화**.

로컬 A/B (`eval_ab_vf.py`, n=200, LB형 쿼터):

| variant | fake_hit | real_hit | acc |
|---------|----------|----------|-----|
| gate+lora (v2.3) | 0.840 | **1.000** | 0.920 |
| **asym_gate+lora** | 0.890 | **0.980** | **0.935** |
| fusion+lora | **0.910** | 0.920 | 0.915 |
| fusion+frozen | **0.920** | 0.870 | 0.895 |

### v2.2 — DF 오탐 억제 (fusion만 변경)

DF-Arena 가중치는 고정. 재학습 대신 **점수 결합**만 바꿈.

- VF: 양방향 게이트
  - `panns ≥ df` → `df + (panns−df)·df`
  - `df > panns` → `panns + (df−panns)·panns` (**DF만 높은 오탐 억제**)
- FILE: `1 − (1−VP×VF)(1−MP×MF)` (noisy-OR)

### v2.1 — 단방향 게이트

- VF: `df + (max(panns, df)−df)·df`
- FILE: `max(VP×VF, MP×MF)`

### v2 — 학습된 VF/MF 헤드

- VP/MP: 믹스 → PANNs
- MF: 믹스 → PANNs 임베딩 → `mf_head.pt`
- VF: DF-Arena + PANNs `vf_head` (`max`)
- FILE: `max(VP×VF, MP×MF)`

### v1 — 원본 인코딩 + 프레임 풀링

- Demucs는 프레임 마스크만
- FILE: `max(VP×VF, MP×MF)`

### v0 — 베이스라인

- VP/MP: PANNs · VF/MF: Demucs 스템 → DF-Arena · FILE: `max(...)`
