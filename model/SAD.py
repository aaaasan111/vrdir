# models/sad.py
import torch.nn as nn


class SAD(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        # 构建反卷积解码器（上采样还原图像）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        return self.decoder(feat)

