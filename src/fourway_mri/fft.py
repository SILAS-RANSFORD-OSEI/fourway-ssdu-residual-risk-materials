from __future__ import annotations

import numpy as np


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """
    Centered orthonormal 2D inverse FFT over the last two dimensions.

    Parameters
    ----------
    kspace:
        Complex-valued k-space array with shape (..., H, W).

    Returns
    -------
    np.ndarray
        Complex-valued image array with shape (..., H, W).
    """
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image


def rss_combine(coil_images: np.ndarray, coil_axis: int = 0) -> np.ndarray:
    """
    Root-sum-of-squares coil combination.

    Parameters
    ----------
    coil_images:
        Complex-valued coil images.

    coil_axis:
        Axis corresponding to receiver coils.

    Returns
    -------
    np.ndarray
        RSS magnitude image.
    """
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=coil_axis))
