from __future__ import annotations

from typing import Iterable

import numpy as np

from fourway_mri.fft import ifft2c, rss_combine


def indices_to_mask(width: int, indices: Iterable[int]) -> np.ndarray:
    """
    Convert phase-encoding line indices into a binary 1D mask.
    """
    mask = np.zeros(width, dtype=np.float32)
    mask[np.asarray(list(indices), dtype=int)] = 1.0
    return mask


def apply_phase_mask(kspace: np.ndarray, mask_1d: np.ndarray) -> np.ndarray:
    """
    Apply a 1D phase-encoding mask to complex multicoil k-space.

    Expected kspace shape:
        (coils, height, width)

    Expected mask shape:
        (width,)

    The mask is broadcast as:
        (1, 1, width)
    """
    if kspace.ndim != 3:
        raise ValueError(f"Expected kspace shape (C, H, W), got {kspace.shape}")

    if mask_1d.ndim != 1:
        raise ValueError(f"Expected mask shape (W,), got {mask_1d.shape}")

    if kspace.shape[-1] != mask_1d.shape[0]:
        raise ValueError(
            f"Width mismatch: kspace width={kspace.shape[-1]}, mask width={mask_1d.shape[0]}"
        )

    return kspace * mask_1d[None, None, :]


def rss_from_kspace(kspace_slice: np.ndarray) -> np.ndarray:
    """
    Compute RSS magnitude image from one multicoil k-space slice.

    Parameters
    ----------
    kspace_slice:
        Complex-valued k-space with shape (coils, height, width).

    Returns
    -------
    np.ndarray
        RSS image with shape (height, width).
    """
    coil_images = ifft2c(kspace_slice)
    rss = rss_combine(coil_images, coil_axis=0)
    return rss.astype(np.float32)


def zero_filled_rss_from_theta(
    kspace_slice: np.ndarray,
    theta_mask_1d: np.ndarray,
) -> np.ndarray:
    """
    Compute zero-filled RSS reconstruction using only Theta_v lines.
    """
    masked = apply_phase_mask(kspace_slice, theta_mask_1d)
    return rss_from_kspace(masked)
