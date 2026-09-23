---
license: other
tags:
  - audio
  - deepfake-detection
  - antispoofing
  - lora
library_name: transformers
base_model: nii-yamagishilab/xls-r-1b-anti-deepfake
---

# deepvoice-pjs-v3.1-ad-lora

SSAFY / deepvoice_detecting **v3.1** AntiDeepfake LoRA adapter.

## Base

- [`nii-yamagishilab/xls-r-1b-anti-deepfake`](https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake)
- Adapter only (`ad_lora.pt`). Does **not** include full XLS-R-1B weights.

## Training (summary)

- Target: encoder last **4** layers + `proj_fc` LoRA (rank 8, alpha 16)
- Aug (v2.5-style): phone 0.12, codec 0.08, **band 0**, gain/noise 0.1, overlays
- Objective: fake-priority, real_hit floor ~0.90
- Valid (internal): fake_hit ≈ 0.92, real_hit ≈ 0.91 after epoch 3

## Usage

Place as `model/ad_lora.pt` in the contest repo. Inference loads it automatically:

```bash
python script.py --ad-lora model/ad_lora.pt --vf-mode antideepfake
# or ensemble with DF-Arena:
python script.py --ad-lora model/ad_lora.pt --vf-mode ensemble
```

## Files

- `ad_lora.pt` — LoRA A/B tensors + meta (`adapter: antideepfake_xlsr_lora`)
