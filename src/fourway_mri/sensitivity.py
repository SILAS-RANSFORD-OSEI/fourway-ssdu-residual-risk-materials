from __future__ import annotations

from typing import Iterable

import torch

from fourway_mri.torch_fft import ifft2c


def indices_to_torch_mask(
    width: int,
    indices: Iterable[int],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mask = torch.zeros(width, device=device, dtype=dtype)
    idx = torch.as_tensor(list(indices), device=device, dtype=torch.long)

    if idx.numel() > 0:
        if torch.any(idx < 0) or torch.any(idx >= width):
            raise ValueError("Mask index out of bounds.")
        mask[idx] = 1.0

    return mask


def apply_1d_phase_mask(kspace: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(kspace):
        raise TypeError(f"kspace must be complex, got {kspace.dtype}")

    if mask_1d.ndim != 1:
        raise ValueError(f"mask_1d must have shape (W,), got {tuple(mask_1d.shape)}")

    if kspace.shape[-1] != mask_1d.shape[0]:
        raise ValueError(
            f"Width mismatch: kspace width={kspace.shape[-1]}, mask width={mask_1d.shape[0]}"
        )

    mask_1d = mask_1d.to(device=kspace.device, dtype=kspace.real.dtype)

    if kspace.ndim == 3:
        return kspace * mask_1d[None, None, :]

    if kspace.ndim == 4:
        return kspace * mask_1d[None, None, None, :]

    raise ValueError(f"Expected kspace shape (C,H,W) or (B,C,H,W), got {tuple(kspace.shape)}")


def acs_soft_body_mask(
    acs_coil_images: torch.Tensor,
    threshold_fraction: float = 0.03,
    softness_fraction: float = 0.02,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Smooth ACS-derived support mask.

    This avoids the hard discontinuity caused by binary thresholding.
    """
    if not torch.is_complex(acs_coil_images):
        raise TypeError("acs_coil_images must be complex.")

    rss = torch.sqrt(torch.sum(torch.abs(acs_coil_images) ** 2, dim=0) + eps)
    rss_max = torch.max(rss)

    threshold = float(threshold_fraction) * rss_max
    softness = float(softness_fraction) * rss_max + eps

    soft_mask = torch.sigmoid((rss - threshold) / softness)
    return soft_mask.to(dtype=acs_coil_images.real.dtype)


def estimate_sensitivities_from_acs(
    kspace: torch.Tensor,
    acs_indices: Iterable[int],
    eps: float = 1e-8,
    body_mask_threshold: float | None = None,
    body_mask_softness: float = 0.02,
) -> torch.Tensor:
    """
    Estimate coil sensitivity maps from ACS-only k-space.

    If body_mask_threshold is provided, a smooth ACS-derived support mask is
    applied to suppress noisy background phase without hard edges.
    """
    if not torch.is_complex(kspace):
        raise TypeError(f"kspace must be complex, got {kspace.dtype}")

    if kspace.ndim != 3:
        raise ValueError(f"Expected kspace shape (C,H,W), got {tuple(kspace.shape)}")

    width = kspace.shape[-1]
    acs_mask = indices_to_torch_mask(
        width=width,
        indices=acs_indices,
        device=kspace.device,
        dtype=kspace.real.dtype,
    )

    acs_kspace = apply_1d_phase_mask(kspace, acs_mask)
    acs_coil_images = ifft2c(acs_kspace)

    rss = torch.sqrt(torch.sum(torch.abs(acs_coil_images) ** 2, dim=0) + eps)
    sensitivities = acs_coil_images / rss[None, :, :]

    if body_mask_threshold is not None:
        support = acs_soft_body_mask(
            acs_coil_images,
            threshold_fraction=float(body_mask_threshold),
            softness_fraction=float(body_mask_softness),
            eps=eps,
        )
        sensitivities = sensitivities * support[None, :, :]

    return sensitivities.to(torch.complex64)


def sensitivity_rss_norm(sensitivities: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(sensitivities):
        raise TypeError("sensitivities must be complex.")

    if sensitivities.ndim != 3:
        raise ValueError(f"Expected sensitivities shape (C,H,W), got {tuple(sensitivities.shape)}")

    return torch.sum(torch.abs(sensitivities) ** 2, dim=0)
