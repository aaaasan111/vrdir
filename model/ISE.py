# models/ise.py
import torch
import torch.nn as nn
import torch.fft as fft


class DNC(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)  # 输出形状为[B, C, 1, 1]
        std = x.std(dim=[2, 3], keepdim=True)
        fn = self.gamma.view(1, -1, 1, 1) * (x - mu) / (std + 1e-6) + self.beta.view(1, -1, 1, 1)
        cat = torch.cat([x, fn], dim=1)
        weight = self.se(cat)
        b, c, h, w = x.shape
        w = weight.view(b, 2, c, 1, 1)  # 拆成两个权重矩阵
        out = w[:, 0] * x + w[:, 1] * fn
        return out


class FGM(nn.Module):
    def __init__(self, in_channels, num_branches=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.conv_amp = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_guidance = nn.Conv2d(in_channels, num_branches, kernel_size=3, padding=1)

    def forward(self, fused_feats, f_input):
        B, total_C, H, W = fused_feats.shape
        C = self.in_channels
        N = self.num_branches
        # 将特征还原为 [B, N, C, H, W]
        branch_feats = fused_feats.view(B, N, total_C // N, H, W)
        f_input = f_input.contiguous()

        try:
            x_fft = torch.fft.rfft2(f_input, dim=(-2, -1))  # 傅里叶变换：空间域 -> 频域
        except RuntimeError as e:
            if "CUFFT_INTERNAL_ERROR" in str(e):
                x_fft = torch.fft.fft2(f_input.cpu(), dim=(-2, -1))
                x_fft = x_fft.to(f_input.device)
            else:
                raise
        real = x_fft.real
        imag = x_fft.imag
        amp = torch.sqrt(real ** 2 + imag ** 2)
        amp_adjusted = self.conv_amp(amp)

        adjusted_real = amp_adjusted * (real / (amp + 1e-8))
        adjusted_imag = amp_adjusted * (imag / (amp + 1e-8))
        adjusted_fft = torch.complex(adjusted_real, adjusted_imag)
        try:
            adjusted_spatial = torch.fft.ifft2(adjusted_fft, dim=(-2, -1)).real  # 还原为时域图
        except RuntimeError as e:
            if "CUFFT_INTERNAL_ERROR" in str(e):
                adjusted_spatial = torch.fft.ifft2(adjusted_fft.cpu(), dim=(-2, -1)).real
                adjusted_spatial = adjusted_spatial.to(adjusted_fft.device)
            else:
                raise
        # 生成调制引导图
        guidance_map = self.conv_guidance(adjusted_spatial)
        guidance_map = torch.softmax(guidance_map, dim=1)
        guidance_map = guidance_map.view(B, N, 1, H, W)
        # 对每个分支按权重进行加权求和
        weighted_sum = (branch_feats * guidance_map).sum(dim=1)
        return weighted_sum


class ISE(nn.Module):
    def __init__(self, dncs, in_channels, num_branches=3):
        super().__init__()
        self.dncs = nn.ModuleList(dncs)
        self.fgm = FGM(in_channels, num_branches)
        self.num_branches = num_branches
        self.in_channels = in_channels

    def forward(self, fdeg):
        outs = [dnc(fdeg) for dnc in self.dncs]
        fused = torch.cat(outs, dim=1)  # 拼接后交给 FGM 处理.[B, N*C, H, W]
        out = self.fgm(fused, fdeg)
        return out
