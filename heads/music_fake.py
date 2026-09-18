import torch
import torch.nn as nn


EMBED_DIM = 2048


class MusicFakeHead(nn.Module):
    """PANNs Cnn14 임베딩 위의 이진 fake 분류기. VF/MF 공용."""

    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, embedding):
        return self.net(embedding).squeeze(-1)


VoiceFakeHead = MusicFakeHead
