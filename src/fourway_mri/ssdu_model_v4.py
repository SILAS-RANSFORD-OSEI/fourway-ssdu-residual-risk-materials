from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fourway_mri.operators import multicoil_adjoint, multicoil_forward
from fourway_mri.torch_fft import chan_to_complex, complex_to_chan


class ZeroInitComplexResidualCNN(nn.Module):
    """
    Complex-channel residual CNN with zero-initialized terminal convolution.

    Input and output are real-channel complex images with shape:

        (B, 2, H, W)

    At initialization, the final convolution outputs zero, so:

        output = input

    This makes the model conservative at step zero.
    """

    def __init__(self, features: int = 32):
        super().__init__()

        self.conv1 = nn.Conv2d(2, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(features, 2, kernel_size=3, padding=1)

        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = F.relu(residual, inplace=True)
        residual = self.conv2(residual)
        residual = F.relu(residual, inplace=True)
        residual = self.final_conv(residual)

        return x + residual


class ScaleConstrainedSingleStepMoDL(nn.Module):
    """
    One-step MoDL-style SSDU model with zero-initialized conservative residual CNN.

    The data-consistency step is computed in physical normalized k-space scale.

    The CNN branch receives sqrt(R_theta)-compensated image data, but the
    zero-initialized residual CNN begins as identity, preventing immediate
    brightness amplification at initialization.
    """

    def __init__(self, features: int = 32, init_step_size: float = 0.1):
        super().__init__()

        self.denoiser = ZeroInitComplexResidualCNN(features=features)
        self.log_step_size = nn.Parameter(torch.log(torch.tensor(float(init_step_size))))

    @property
    def step_size(self) -> torch.Tensor:
        return F.softplus(self.log_step_size)

    def forward(
        self,
        x0: torch.Tensor,
        y_theta: torch.Tensor,
        sensitivities: torch.Tensor,
        theta_mask: torch.Tensor,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not torch.is_complex(x0):
            raise TypeError("x0 must be complex.")
        if not torch.is_complex(y_theta):
            raise TypeError("y_theta must be complex.")
        if not torch.is_complex(sensitivities):
            raise TypeError("sensitivities must be complex.")

        predicted_theta = multicoil_forward(x0, sensitivities, theta_mask)
        residual_theta = predicted_theta - y_theta

        dc_gradient = multicoil_adjoint(
            residual_theta,
            sensitivities,
            theta_mask,
        )

        z = x0 - self.step_size * dc_gradient

        if input_scale is None:
            input_scale = torch.ones(
                z.shape[0],
                device=z.device,
                dtype=z.real.dtype,
            )

        input_scale = input_scale.to(device=z.device, dtype=z.real.dtype)
        scale = input_scale[:, None, None]

        z_comp = z * scale
        z_chan = complex_to_chan(z_comp)

        out_chan = self.denoiser(z_chan)

        out_comp = chan_to_complex(out_chan)
        x_out = out_comp / scale

        return x_out
