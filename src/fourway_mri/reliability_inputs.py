from __future__ import annotations

from typing import Dict

import torch


def robust_normalize_per_sample(
    image: torch.Tensor,
    percentile: float = 99.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Robustly normalize each sample in a real image batch by its percentile.

    Input:
        image: (B,H,W) real tensor

    Output:
        normalized image with same shape
    """
    if not torch.is_floating_point(image):
        raise TypeError("image must be a real floating tensor.")

    if image.ndim != 3:
        raise ValueError(f"image must have shape (B,H,W), got {tuple(image.shape)}")

    b = image.shape[0]
    flat = image.detach().reshape(b, -1)

    q = torch.quantile(
        flat,
        float(percentile) / 100.0,
        dim=1,
    ).to(device=image.device, dtype=image.dtype)

    return image / (q[:, None, None] + eps)


def analytical_cartesian_psf_from_mask(
    risk_mask: torch.Tensor,
    height: int,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute analytical image-domain PSF magnitude from a 1D Cartesian line mask.

    The risk mask is interpreted as a centered k-space phase-encoding mask.
    It is repeated along the readout/frequency dimension to form a 2D mask.

    Output:
        psf magnitude with shape (H,W), centered and real.
    """
    if risk_mask.ndim != 1:
        raise ValueError(f"risk_mask must have shape (W,), got {tuple(risk_mask.shape)}")

    if height < 1:
        raise ValueError("height must be >= 1.")

    device = risk_mask.device
    dtype = risk_mask.dtype if torch.is_floating_point(risk_mask) else torch.float32

    mask_1d = (risk_mask > 0).to(device=device, dtype=dtype)
    width = mask_1d.shape[0]

    mask_2d = mask_1d[None, :].repeat(int(height), 1)
    mask_2d_complex = torch.complex(mask_2d, torch.zeros_like(mask_2d))

    # Convert centered k-space mask to uncentered FFT order, apply IFFT,
    # then shift the PSF to image center.
    uncentered = torch.fft.ifftshift(mask_2d_complex, dim=(-2, -1))
    psf = torch.fft.ifft2(uncentered, norm="ortho")
    psf = torch.fft.fftshift(psf, dim=(-2, -1))

    psf_mag = torch.abs(psf)

    if normalize:
        psf_mag = psf_mag / (torch.max(psf_mag) + eps)

    return psf_mag.to(dtype=dtype)


def image_gradient_magnitude(
    image: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute simple finite-difference gradient magnitude.

    Input:
        image: (B,H,W) real tensor

    Output:
        gradient magnitude with shape (B,H,W)
    """
    if not torch.is_floating_point(image):
        raise TypeError("image must be a real floating tensor.")

    if image.ndim != 3:
        raise ValueError(f"image must have shape (B,H,W), got {tuple(image.shape)}")

    dx = torch.zeros_like(image)
    dy = torch.zeros_like(image)

    dx[:, :, 1:] = image[:, :, 1:] - image[:, :, :-1]
    dy[:, 1:, :] = image[:, 1:, :] - image[:, :-1, :]

    return torch.sqrt(dx**2 + dy**2 + eps)


def build_reliability_input_tensor(
    x_hat: torch.Tensor,
    x0: torch.Tensor,
    support_mask: torch.Tensor,
    risk_mask: torch.Tensor,
    psf_gain: torch.Tensor,
    normalize_percentile: float = 99.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Build the 6-channel PSF-aware ReliabilityCNN input tensor.

    Channels:
        0: |x_hat|
        1: |x0|
        2: |x_hat - x0|
        3: M_soft
        4: analytical PSF from Lambda_risk
        5: sensitivity-aware PSF/gain map q_psf

    The residual target itself is not included.
    """
    if not torch.is_complex(x_hat):
        raise TypeError("x_hat must be complex.")

    if not torch.is_complex(x0):
        raise TypeError("x0 must be complex.")

    if x_hat.shape != x0.shape:
        raise ValueError(f"x_hat and x0 shape mismatch: {x_hat.shape} vs {x0.shape}")

    if x_hat.ndim != 3:
        raise ValueError(f"x_hat must have shape (B,H,W), got {tuple(x_hat.shape)}")

    if support_mask.shape != x_hat.shape:
        raise ValueError(
            f"support_mask must have shape {tuple(x_hat.shape)}, "
            f"got {tuple(support_mask.shape)}"
        )

    if psf_gain.shape != x_hat.shape:
        raise ValueError(
            f"psf_gain must have shape {tuple(x_hat.shape)}, "
            f"got {tuple(psf_gain.shape)}"
        )

    b, h, w = x_hat.shape

    if risk_mask.ndim != 1 or risk_mask.shape[0] != w:
        raise ValueError(
            f"risk_mask must have shape ({w},), got {tuple(risk_mask.shape)}"
        )

    x_hat_mag = torch.abs(x_hat)
    x0_mag = torch.abs(x0)
    intervention_mag = torch.abs(x_hat - x0)

    x_hat_ch = robust_normalize_per_sample(x_hat_mag, normalize_percentile, eps)
    x0_ch = robust_normalize_per_sample(x0_mag, normalize_percentile, eps)
    intervention_ch = robust_normalize_per_sample(
        intervention_mag,
        normalize_percentile,
        eps,
    )

    mask_ch = torch.clamp(support_mask.to(dtype=x_hat_mag.dtype), 0.0, 1.0)

    psf = analytical_cartesian_psf_from_mask(
        risk_mask=risk_mask.to(device=x_hat.device),
        height=h,
        normalize=True,
        eps=eps,
    ).to(device=x_hat.device, dtype=x_hat_mag.dtype)

    psf_ch = psf[None, :, :].repeat(b, 1, 1)

    psf_gain_ch = robust_normalize_per_sample(
        psf_gain.to(dtype=x_hat_mag.dtype),
        normalize_percentile,
        eps,
    )

    input_tensor = torch.stack(
        [
            x_hat_ch,
            x0_ch,
            intervention_ch,
            mask_ch,
            psf_ch,
            psf_gain_ch,
        ],
        dim=1,
    )

    return {
        "input": input_tensor,
        "x_hat_mag": x_hat_mag,
        "x0_mag": x0_mag,
        "intervention_mag": intervention_mag,
        "support_mask": mask_ch,
        "analytical_psf": psf_ch,
        "psf_gain": psf_gain_ch,
    }
