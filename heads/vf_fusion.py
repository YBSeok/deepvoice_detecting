"""학습된 VF fusion: PANNs VF + DF VF (+ VP) → VOICE_FAKE_PROB."""

from __future__ import annotations

import torch
import torch.nn as nn


FUSION_FEATURE_DIM = 5


def fusion_features(panns_vf, df_vf, vp=0.0):
    """스칼라 점수 → fusion 입력 벡터.

    [panns, df, vp, panns*df, |panns-df|]
    마지막 두 항은 합의/불일치 패턴을 바로 학습하게 한다.
    """
    p = float(panns_vf)
    d = float(df_vf)
    v = float(vp)
    return torch.tensor(
        [p, d, v, p * d, abs(p - d)],
        dtype=torch.float32,
    )


class VoiceFakeFusion(nn.Module):
    """작은 MLP. 제출 zip에 넣기 쉬운 수 KB 단위."""

    def __init__(self, in_dim=FUSION_FEATURE_DIM, hidden=32, dropout=0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)

    @torch.inference_mode()
    def predict_proba(self, panns_vf, df_vf, vp=0.0):
        device = next(self.parameters()).device
        x = fusion_features(panns_vf, df_vf, vp).unsqueeze(0).to(device)
        return float(torch.sigmoid(self.forward(x)).item())
