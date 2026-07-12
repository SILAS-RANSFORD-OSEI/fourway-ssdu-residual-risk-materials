from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReliabilityUNetSmall(nn.Module):
    """
    Lightweight U-Net for residual-risk prediction.

    Input:
        (B, 6, H, W)

    Output:
        (B, 1, H, W)

    The model outputs an unconstrained regression map. Nonnegative clamping
    should be applied only for evaluation if needed, not inside the training loss.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_channels: int = 16,
    ):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape (B,C,H,W), got {tuple(x.shape)}")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self._upsample_to(b, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self._upsample_to(d3, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self._upsample_to(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)



class ReliabilityUNetWithPSFSkip(nn.Module):
    """
    Reliability U-Net with explicit global PSF/gain skip connection.

    Output:
        y_hat = UNet(X) + w * X[:, psf_channel_index] + b

    This preserves the smooth physics prior from q_psf while allowing the CNN
    branch to learn nonlinear local corrections.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_channels: int = 16,
        psf_channel_index: int = 5,
        init_psf_weight: float = 1.0,
        init_psf_bias: float = 0.0,
    ):
        super().__init__()

        if out_channels != 1:
            raise ValueError("ReliabilityUNetWithPSFSkip currently supports out_channels=1.")

        if psf_channel_index < 0 or psf_channel_index >= in_channels:
            raise ValueError(
                f"psf_channel_index={psf_channel_index} is invalid for "
                f"in_channels={in_channels}."
            )

        self.psf_channel_index = int(psf_channel_index)

        self.unet = ReliabilityUNetSmall(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )

        self.psf_weight = nn.Parameter(torch.tensor(float(init_psf_weight)))
        self.psf_bias = nn.Parameter(torch.tensor(float(init_psf_bias)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape (B,C,H,W), got {tuple(x.shape)}")

        if x.shape[1] <= self.psf_channel_index:
            raise ValueError(
                f"Input has {x.shape[1]} channels, but psf_channel_index="
                f"{self.psf_channel_index}."
            )

        cnn_out = self.unet(x)
        q_psf = x[:, self.psf_channel_index : self.psf_channel_index + 1, :, :]

        return cnn_out + self.psf_weight * q_psf + self.psf_bias
