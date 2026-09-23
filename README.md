# deepvoice_detecting

AI 생성 음성·음악 탐지. 제출 필드 5개: `FILE_FAKE`, `VOICE_FAKE`, `MUSIC_FAKE`, `VOICE_PRESENT`, `MUSIC_PRESENT`.

## 리더보드

최신이 위. 점수는 제출 후 기입.

| 버전 | 날짜 | 총점 | ADS | CPS | 비고 |
|------|------|------|-----|-----|------|
| v3.1 | 2026-09-23 | 0.641 | 0.602 | 0.989 | AD LoRA + DF max 앙상블. **v3 대비 ADS −0.087 → 총점↓** (CPS 동일) |
| v3 | 2026-09-23 | 0.719 | 0.689 | 0.989 | AntiDeepfake frozen + v1 마스크. **v2.5 대비 +0.052**, v1(0.724)에는 **−0.005** |
| v2.5 | 2026-09-22 | 0.667 | 0.631 | 0.989 | aug 재설계 + 헤드·LoRA·fusion 재학습. v2.x 최고 |
| v2.4 | 2026-09-22 | 0.654 | 0.617 | 0.989 | 학습 VF fusion + v2.3 LoRA |
| v2.3 | 2026-09-22 | 0.652 | 0.615 | 0.989 | LoRA + 강한 채널 aug. ADS↓ |
| v2.2 | 2026-09-18 | 0.661 | 0.625 | 0.989 | VF 양방향 게이트 + FILE noisy-OR |
| v2.1 | 2026-09-18 | | | | VF 단방향 게이트 |
| v2 | 2026-09-18 | 0.684 | | | PANNs VF/MF + DF-Arena `max` |
| v1 | 2026-09-17 | 0.724 | 0.694 | 0.989 | 원본 믹스 인코딩 + 마스크 풀링 (DF-Arena) — **역대 최고** |
| v0 | | | | | 베이스라인 (스템→DF) |

## 제출 (v3.1)

```text
submit.zip  ← script.py + antideepfake_vf.py + ad_lora.py + df_lora.py + df_arena_vf.py
            + requirements.txt + heads/
            + model/ (PANNs, HTDemucs, antideepfake_xlsr_1b, ad_lora.pt, mf_head.pt, df_arena_1b)
            # 기본: --vf-mode ensemble = max(AD+LoRA, DF)
```

```bash
# 데이터: 구 폴더 + deepfake_dataset → D:/deepvoice_dataset (junction)
# 매니페스트: D:/deepvoice_dataset/manifests (= D:/manifests junction)

# v3.1 학습 (v2.5식 mild aug + AntiDeepfake LoRA)
python train_ad_adapt.py \
  --train-csv /mnt/d/manifests/train.csv \
  --valid-csv /mnt/d/manifests/valid.csv \
  --rewrite-prefix 'D:\\=/mnt/d/' \
  --out model/ad_lora.pt \
  --max-per-source 200 --voice-per-source 800 --overlays 400 \
  --phone-frac 0.12 --codec-frac 0.08 --band-frac 0 \
  --gain-frac 0.1 --noise-frac 0.1 --domain-repeat 1 \
  --last-n-layers 4 --epochs 3 --batch-size 1 --device cuda

python pack_submit.py --out submit.zip          # DF 포함(기본)
# python pack_submit.py --no-with-df             # AD only

python script.py --test-dir data/test --sample-submission data/sample_submission.csv --output output/submission.csv
# A/B: --vf-mode antideepfake | df
```

## 체크리스트 (0–8)

| # | 항목 | 상태 | 메모 (LB 반영) |
|---|------|------|----------------|
| 0 | DF vs AntiDeepfake | 완료 | LB: v1 DF 0.724 ≳ v3 AD 0.719. 제출 기본 **max 앙상블** |
| 1 | Validation | 완료 | 폴더별 층화 train/valid/test (`build_manifest`) |
| 2 | Voice Branch LoRA | 완료(실패) | LB: v3.1 ADS **0.602** ≪ v3 frozen **0.689** → LoRA가 ADS 붕괴 |
| 3 | Codec Aug | 완료 | codec 0.08, 실음성 위주 |
| 4 | Telephone Band Aug | 완료 | phone 0.12 / **band 0** (강한 band는 ADS↓) |
| 5 | Noise Aug | 완료 | noise/gain 0.1 |
| 6 | Pseudo Label | **스킵** | CPS/ADS 리스크, LB 근거 없음 |
| 7 | DF-Arena 앙상블 | 완료(의심) | max+LoRA 제출이 v3만도 못함. **다음: v3 frozen 재제출/AD only** |
| 8 | TTA | **스킵** | 지연·채널 삭감이 ADS에 불리 (v2.x) |

의존: `transformers` + `safetensors`. `fairseq` 불필요.

## 변경 이력

### v3.1 — AntiDeepfake LoRA + DF max 앙상블 (실패)

리더보드: 총점 **0.641** / ADS **0.602** / CPS **0.989**.
v3(0.719 / 0.689) 대비 ADS **−0.087**, 총점 **−0.078**. CPS만 동일.
v2.3(0.615 ADS)보다도 낮음 → **역대 최악권 ADS**.

가설(로컬 valid fake≈0.92 → LB↑)은 **기각**. 가능한 원인:
1. **LoRA가 frozen AD의 일반화를 붕괴** (학습 mean-pool vs 추론 마스크 풀링 불일치, CSV≠LB 생성기)
2. **max(AD_LoRA, DF)** 가 망가진 AD 점수를 살리지 못함 — DF만으로는 v1급이 안 됨(앙상블이 LoRA 피해를 못 메움)
3. 로컬 valid는 매니페스트 분포에 과적합 → LB ADS와 괴리

**다음:** `ad_lora.pt` 없이 **v3 frozen AD만** 재확인(이미 0.719). LoRA/앙상블은 A/B 후에만.

| 항목 | 설정 |
|------|------|
| 백본 | AntiDeepfake XLS-R-1B + LoRA + DF max |
| 적응 | encoder 마지막 4층 + `proj_fc` (rank 8) |
| aug | phone 0.12 / codec 0.08 / band 0 / gain·noise 0.1 / overlays |
| 산출 | `model/ad_lora.pt` |

### v3 — AntiDeepfake frozen

리더보드: 총점 **0.719** / ADS **0.689** / CPS **0.989**.
v1 뼈대 + AD only. **현재 사실상 최고 제출 후보**(v1 −0.005).

### v2.5 — aug 재설계

총점 **0.667** / ADS **0.631** / CPS **0.989**.
band OFF·채널 축소·overlays↑ → FT가 다시 이득을 냄. v1에는 미달 → 백본 교체(v3).

### v2.4 / v2.3 / v2.2

fusion·강한 aug·대칭 게이트. ADS 정체/하락 구간.

### v1 — 원본 인코딩 + 마스크 풀링

총점 **0.724** / ADS **0.694** / CPS **0.989** — 넘어야 할 목표선.
