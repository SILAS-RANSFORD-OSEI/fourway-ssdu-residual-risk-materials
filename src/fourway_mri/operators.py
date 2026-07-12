from __future__ import annotations

import torch

from fourway_mri.torch_fft import fft2c, ifft2c


def broadcast_mask(mask_1d: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a 1D phase-encoding mask to complex k-space shape.

    mask_1d:
        (W,)

    target:
        (B, C, H, W)

    output:
        (1, 1, 1, W)
    """
    if mask_1d.ndim != 1:
        raise ValueError(f"mask_1d must have shape (W,), got {tuple(mask_1d.shape)}")

    if target.ndim != 4:
        raise ValueError(f"target must have shape (B,C,H,W), got {tuple(target.shape)}")

    if target.shape[-1] != mask_1d.shape[0]:
        raise ValueError(
            f"Width mismatch: target width={target.shape[-1]}, mask width={mask_1d.shape[0]}"
        )

    return mask_1d.to(device=target.device, dtype=target.real.dtype)[None, None, None, :]


def apply_kspace_mask(kspace: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
    """
    Apply phase-encoding mask to batched multicoil k-space.

    kspace:
        (B, C, H, W) complex

    mask_1d:
        (W,)
    """
    if not torch.is_complex(kspace):
        raise TypeError("kspace must be complex.")

    mask = broadcast_mask(mask_1d, kspace)
    return kspace * mask


def multicoil_forward(
    image: torch.Tensor,
    sensitivities: torch.Tensor,
    mask_1d: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Multicoil forward operator.

    image:
        (B, H, W) complex

    sensitivities:
        (B, C, H, W) complex

    mask_1d:
        optional (W,) phase-encoding mask

    returns:
        (B, C, H, W) complex k-space
    """
    if not torch.is_complex(image):
        raise TypeError("image must be complex.")

    if not torch.is_complex(sensitivities):
        raise TypeError("sensitivities must be complex.")

    if image.ndim != 3:
        raise ValueError(f"image must have shape (B,H,W), got {tuple(image.shape)}")

    if sensitivities.ndim != 4:
        raise ValueError(
            f"sensitivities must have shape (B,C,H,W), got {tuple(sensitivities.shape)}"
        )

    if image.shape[0] != sensitivities.shape[0]:
        raise ValueError("Batch size mismatch between image and sensitivities.")

    if image.shape[-2:] != sensitivities.shape[-2:]:
        raise ValueError("Spatial shape mismatch between image and sensitivities.")

    coil_images = sensitivities * image[:, None, :, :]
    kspace = fft2c(coil_images)

    if mask_1d is not None:
        kspace = apply_kspace_mask(kspace, mask_1d)

    return kspace


def multicoil_adjoint(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    mask_1d: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Multicoil adjoint operator.

    kspace:
        (B, C, H, W) complex

    sensitivities:
        (B, C, H, W) complex

    mask_1d:
        optional (W,) phase-encoding mask

    returns:
        (B, H, W) complex image
    """
    if not torch.is_complex(kspace):
        raise TypeError("kspace must be complex.")

    if not torch.is_complex(sensitivities):
        raise TypeError("sensitivities must be complex.")

    if kspace.ndim != 4:
        raise ValueError(f"kspace must have shape (B,C,H,W), got {tuple(kspace.shape)}")

    if sensitivities.shape != kspace.shape:
        raise ValueError("sensitivities and kspace must have identical shapes.")

    if mask_1d is not None:
        kspace = apply_kspace_mask(kspace, mask_1d)

    coil_images = ifft2c(kspace)
    image = torch.sum(torch.conj(sensitivities) * coil_images, dim=1)

    return image


def normal_operator(
    image: torch.Tensor,
    sensitivities: torch.Tensor,
    mask_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Apply E^H E to an image.
    """
    return multicoil_adjoint(
        multicoil_forward(image, sensitivities, mask_1d),
        sensitivities,
        mask_1d,
    )
