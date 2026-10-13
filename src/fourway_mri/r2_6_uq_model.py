import torch
import torch.nn as nn

from fourway_mri.reliability_model import ReliabilityUNetSmall


A4_ZERO_CHANNELS = (3, 4, 5)


def apply_a4_input(x: torch.Tensor) -> torch.Tensor:
    """
    Construct the Paper-1 A4 image-only input representation.

    The physical tensor remains six-channel:
        channels 0-2: active image-derived channels
        channels 3-5: deterministically zeroed

    Parameters
    ----------
    x : torch.Tensor
        Tensor with shape (B, 6, H, W).

    Returns
    -------
    torch.Tensor
        Six-channel A4 tensor with channels 3-5 zeroed.
    """
    if x.ndim != 4:
        raise ValueError(
            f"x must have shape (B,6,H,W), got {tuple(x.shape)}"
        )

    if x.shape[1] != 6:
        raise ValueError(
            f"A4 requires exactly 6 physical channels, got {x.shape[1]}"
        )

    x_a4 = x.clone()
    x_a4[:, 3:6, :, :] = 0.0

    return x_a4


class R26MCDropoutUNet(ReliabilityUNetSmall):
    """
    MC-dropout model for predictive-variance comparison.

    Base architecture:
        ReliabilityUNetSmall topology used for the A4 comparator.

    Difference from deterministic A4:
        spatial Dropout2d is applied after each feature block.

    No dropout is applied to the final 1x1 output convolution.

    Dropout introduces no trainable parameters.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_channels: int = 8,
        dropout_probability: float = 0.10,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )

        p = float(dropout_probability)

        if not (0.0 < p < 1.0):
            raise ValueError(
                f"dropout_probability must be in (0,1), got {p}"
            )

        self.dropout_probability = p
        self.mc_dropout = nn.Dropout2d(p=p)

    def _drop(self, x: torch.Tensor) -> torch.Tensor:
        return self.mc_dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"x must have shape (B,C,H,W), got {tuple(x.shape)}"
            )

        e1 = self._drop(self.enc1(x))
        e2 = self._drop(self.enc2(self.pool(e1)))
        e3 = self._drop(self.enc3(self.pool(e2)))

        b = self._drop(
            self.bottleneck(self.pool(e3))
        )

        d3 = self._upsample_to(b, e3)
        d3 = self._drop(
            self.dec3(
                torch.cat([d3, e3], dim=1)
            )
        )

        d2 = self._upsample_to(d3, e2)
        d2 = self._drop(
            self.dec2(
                torch.cat([d2, e2], dim=1)
            )
        )

        d1 = self._upsample_to(d2, e1)
        d1 = self._drop(
            self.dec1(
                torch.cat([d1, e1], dim=1)
            )
        )

        return self.out_conv(d1)


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enable stochastic Dropout2d layers while leaving the remainder of
    the network in evaluation mode.

    Intended sequence:
        model.eval()
        enable_mc_dropout(model)
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            module.train()
