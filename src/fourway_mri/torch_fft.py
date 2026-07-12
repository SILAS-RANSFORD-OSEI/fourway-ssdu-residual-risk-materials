from __future__ import annotations

import torch


def fft2c(image: torch.Tensor) -> torch.Tensor:
    """
    Centered orthonormal 2D FFT over the last two dimensions.
    Expected input: complex tensor (..., H, W).
    """
    if not torch.is_complex(image):
        raise TypeError(f"fft2c expects a complex tensor, got dtype={image.dtype}")

    shifted = torch.fft.ifftshift(image, dim=(-2, -1))
    kspace = torch.fft.fft2(shifted, dim=(-2, -1), norm="ortho")
    kspace = torch.fft.fftshift(kspace, dim=(-2, -1))
    return kspace


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """
    Centered orthonormal 2D inverse FFT over the last two dimensions.
    Expected input: complex tensor (..., H, W).
    """
    if not torch.is_complex(kspace):
        raise TypeError(f"ifft2c expects a complex tensor, got dtype={kspace.dtype}")

    shifted = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
    image = torch.fft.fftshift(image, dim=(-2, -1))
    return image


def complex_to_chan(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex image tensor to two-channel real tensor.

    Input:
        (B, H, W) complex

    Output:
        (B, 2, H, W) real
    """
    if not torch.is_complex(x):
        raise TypeError(f"complex_to_chan expects complex tensor, got {x.dtype}")

    if x.ndim != 3:
        raise ValueError(f"Expected shape (B, H, W), got {tuple(x.shape)}")

    return torch.stack([x.real, x.imag], dim=1)


def chan_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Convert two-channel real tensor to complex image tensor.

    Input:
        (B, 2, H, W)

    Output:
        (B, H, W) complex
    """
    if torch.is_complex(x):
        raise TypeError("chan_to_complex expects a real tensor.")

    if x.ndim != 4:
        raise ValueError(f"Expected shape (B, 2, H, W), got {tuple(x.shape)}")

    if x.shape[1] != 2:
        raise ValueError(f"Expected channel dimension 2, got {x.shape[1]}")

    return torch.complex(x[:, 0], x[:, 1])


def rss_combine_torch(
    coil_images: torch.Tensor,
    coil_dim: int = 1,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Root-sum-of-squares coil combination.

    Typical input:
        (B, C, H, W) complex

    Output:
        (B, H, W) real
    """
    if not torch.is_complex(coil_images):
        raise TypeError(f"rss_combine_torch expects complex tensor, got {coil_images.dtype}")

    return torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=coil_dim) + eps)
