from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fourway_mri.operators import multicoil_adjoint, multicoil_forward
from fourway_mri.torch_fft import chan_to_complex, complex_to_chan


class ComplexResidualCNN(nn.Module):
    def __init__(self, features: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SingleStepMoDL(nn.Module):
    """
    One-step MoDL-style complex SSDU model.

    DC step is computed in physical scale.
    CNN receives sqrt(R_theta)-compensated image.
    Output is converted back to physical scale before k-space loss.
    """

    def __init__(self, features: int = 32, init_step_size: float = 0.1):
        super().__init__()
        self.denoiser = ComplexResidualCNN(features=features)
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


def ssdu_kspace_loss(
    prediction: torch.Tensor,
    target_kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    lambda_rec_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_kspace = multicoil_forward(
        prediction,
        sensitivities,
        lambda_rec_mask,
    )

    target_lambda = target_kspace * lambda_rec_mask.to(
        device=target_kspace.device,
        dtype=target_kspace.real.dtype,
    )[None, None, None, :]

    numerator = torch.sum(torch.abs(pred_kspace - target_lambda) ** 2)
    denominator = torch.sum(torch.abs(target_lambda) ** 2) + eps

    return numerator / denominator
