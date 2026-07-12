from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from fourway_mri.operators import apply_kspace_mask, multicoil_adjoint, multicoil_forward


def _validate_line_mask(mask_1d: torch.Tensor) -> torch.Tensor:
    if mask_1d.ndim != 1:
        raise ValueError(f"mask_1d must have shape (W,), got {tuple(mask_1d.shape)}")

    return (mask_1d > 0).to(dtype=torch.float32)


def _broadcast_line_weights(
    weights_1d: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if target.ndim != 4:
        raise ValueError(f"target must have shape (B,C,H,W), got {tuple(target.shape)}")

    if weights_1d.ndim != 1:
        raise ValueError(f"weights_1d must have shape (W,), got {tuple(weights_1d.shape)}")

    if target.shape[-1] != weights_1d.shape[0]:
        raise ValueError(
            f"Width mismatch: target width={target.shape[-1]}, "
            f"weights width={weights_1d.shape[0]}"
        )

    return weights_1d.to(
        device=target.device,
        dtype=target.real.dtype,
    )[None, None, None, :]


def cartesian_line_density_compensation(
    mask_1d: torch.Tensor,
    kernel_size: int = 9,
    eps: float = 1e-8,
    normalize_mean: bool = True,
) -> torch.Tensor:
    """
    Estimate a simple 1D Cartesian density compensation function.

    The mask is interpreted along the phase-encoding direction.

    Dense local clusters receive smaller weights. Sparse local regions receive
    larger weights. The weights are zero outside the mask.

    Parameters
    ----------
    mask_1d:
        Binary phase-encoding mask with shape (W,).

    kernel_size:
        Local window used to estimate sampling density.

    eps:
        Numerical stabilizer.

    normalize_mean:
        If True, normalize sampled-line weights so their mean is 1.

    Returns
    -------
    weights:
        Density compensation weights with shape (W,).
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1.")

    mask = _validate_line_mask(mask_1d)

    device = mask.device
    dtype = mask.dtype

    x = mask[None, None, :]

    pad_left = (kernel_size - 1) // 2
    pad_right = kernel_size // 2

    x_pad = F.pad(x, (pad_left, pad_right), mode="replicate")
    local_density = F.avg_pool1d(x_pad, kernel_size=kernel_size, stride=1)[0, 0]

    weights = mask / (local_density + eps)
    weights = weights * mask

    if normalize_mean and torch.sum(mask) > 0:
        sampled_mean = torch.sum(weights) / (torch.sum(mask) + eps)
        weights = weights / (sampled_mean + eps)

    return weights.to(device=device, dtype=dtype)


def apply_line_weights_to_kspace(
    kspace: torch.Tensor,
    weights_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Apply 1D phase-encoding weights to multicoil k-space.

    kspace:
        (B, C, H, W) complex

    weights_1d:
        (W,)
    """
    if not torch.is_complex(kspace):
        raise TypeError("kspace must be complex.")

    weights = _broadcast_line_weights(weights_1d, kspace)
    return kspace * weights


def risk_kspace_residual(
    prediction: torch.Tensor,
    target_kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    risk_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the held-out k-space residual on Lambda_risk.

    r_Lambda = P_Lambda y - P_Lambda F S x_hat
    """
    if not torch.is_complex(prediction):
        raise TypeError("prediction must be complex.")

    if not torch.is_complex(target_kspace):
        raise TypeError("target_kspace must be complex.")

    pred_risk = multicoil_forward(
        prediction,
        sensitivities,
        risk_mask,
    )

    target_risk = apply_kspace_mask(
        target_kspace,
        risk_mask,
    )

    return target_risk - pred_risk


def complex_residual_backprojection(
    residual_kspace: torch.Tensor,
    sensitivities: torch.Tensor,
) -> torch.Tensor:
    """
    Backproject a masked residual through the multicoil adjoint.

    The residual is assumed to already be zero outside the selected k-space lines.
    """
    if not torch.is_complex(residual_kspace):
        raise TypeError("residual_kspace must be complex.")

    return multicoil_adjoint(
        residual_kspace,
        sensitivities,
        mask_1d=None,
    )


def residual_energy(
    complex_residual_image: torch.Tensor,
) -> torch.Tensor:
    """
    Convert complex residual image to squared-magnitude residual energy.

    Input:
        (B, H, W) complex

    Output:
        (B, H, W) real
    """
    if not torch.is_complex(complex_residual_image):
        raise TypeError("complex_residual_image must be complex.")

    return torch.abs(complex_residual_image) ** 2


def same_avg_pool2d(
    image: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Average-pool a 2D image while preserving spatial size.

    Supports:
        (B, H, W) or (B, 1, H, W)
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1.")

    squeeze_channel = False

    if image.ndim == 3:
        image = image[:, None, :, :]
        squeeze_channel = True
    elif image.ndim != 4:
        raise ValueError(f"image must have shape (B,H,W) or (B,1,H,W), got {tuple(image.shape)}")

    pad_top = (kernel_size - 1) // 2
    pad_bottom = kernel_size // 2
    pad_left = (kernel_size - 1) // 2
    pad_right = kernel_size // 2

    padded = F.pad(
        image,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="replicate",
    )

    pooled = F.avg_pool2d(
        padded,
        kernel_size=kernel_size,
        stride=1,
    )

    if squeeze_channel:
        pooled = pooled[:, 0, :, :]

    return pooled


def estimate_psf_gain_map(
    sensitivities: torch.Tensor,
    risk_mask: torch.Tensor,
    dcf_weights: torch.Tensor,
    num_probes: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """
    Estimate a PSF/sensitivity gain map for the Lambda_risk backprojection.

    Random complex probes are injected on Lambda_risk, density compensated,
    backprojected through E^H, squared, and averaged.

    This is a deterministic Monte Carlo approximation controlled by seed.
    """
    if num_probes < 1:
        raise ValueError("num_probes must be >= 1.")

    if not torch.is_complex(sensitivities):
        raise TypeError("sensitivities must be complex.")

    device = sensitivities.device
    real_dtype = sensitivities.real.dtype

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    b, c, h, w = sensitivities.shape
    energy_sum = torch.zeros((b, h, w), device=device, dtype=real_dtype)

    for _ in range(num_probes):
        real = torch.randn(
            (b, c, h, w),
            device=device,
            dtype=real_dtype,
            generator=generator,
        )
        imag = torch.randn(
            (b, c, h, w),
            device=device,
            dtype=real_dtype,
            generator=generator,
        )

        probe = torch.complex(real, imag)
        probe = apply_kspace_mask(probe, risk_mask)
        probe = apply_line_weights_to_kspace(probe, dcf_weights)

        probe_backproj = complex_residual_backprojection(
            probe,
            sensitivities,
        )

        energy_sum = energy_sum + residual_energy(probe_backproj)

    return energy_sum / float(num_probes)


def robust_target_normalization(
    target: torch.Tensor,
    percentile: float = 99.0,
    log_alpha: float = 10.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Produce robust-normalized and log-compressed residual-risk targets.
    """
    if not torch.is_floating_point(target):
        raise TypeError("target must be a real floating tensor.")

    q = torch.quantile(
        target.detach().flatten(),
        float(percentile) / 100.0,
    )

    target_norm = target / (q + eps)
    target_log = torch.log1p(float(log_alpha) * target_norm)

    return {
        "target_norm": target_norm,
        "target_log": target_log,
        "target_p99": q,
    }


def generate_residual_risk_target(
    prediction: torch.Tensor,
    target_kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    risk_mask: torch.Tensor,
    patch_size: int = 16,
    dcf_kernel_size: int = 9,
    psf_num_probes: int = 4,
    psf_seed: int = 0,
    normalize_percentile: float = 99.0,
    log_alpha: float = 10.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Generate a PSF-aware, patch-aggregated residual-risk target.

    Pipeline:
        1. residual on Lambda_risk,
        2. density compensation,
        3. complex adjoint backprojection,
        4. squared-magnitude residual energy,
        5. patch-level energy envelope,
        6. PSF/gain normalization,
        7. robust normalization and log compression.
    """
    dcf_weights = cartesian_line_density_compensation(
        risk_mask,
        kernel_size=dcf_kernel_size,
        eps=eps,
        normalize_mean=True,
    )

    residual = risk_kspace_residual(
        prediction=prediction,
        target_kspace=target_kspace,
        sensitivities=sensitivities,
        risk_mask=risk_mask,
    )

    residual_comp = apply_line_weights_to_kspace(
        residual,
        dcf_weights,
    )

    residual_complex = complex_residual_backprojection(
        residual_comp,
        sensitivities,
    )

    residual_mag = residual_energy(residual_complex)

    residual_envelope = same_avg_pool2d(
        residual_mag,
        kernel_size=patch_size,
    )

    psf_gain = estimate_psf_gain_map(
        sensitivities=sensitivities,
        risk_mask=risk_mask,
        dcf_weights=dcf_weights,
        num_probes=psf_num_probes,
        seed=psf_seed,
    )

    psf_envelope = same_avg_pool2d(
        psf_gain,
        kernel_size=patch_size,
    )

    target_psf_normalized = residual_envelope / (psf_envelope + eps)

    norm = robust_target_normalization(
        target_psf_normalized,
        percentile=normalize_percentile,
        log_alpha=log_alpha,
        eps=eps,
    )

    return {
        "risk_residual_kspace": residual,
        "dcf_weights": dcf_weights,
        "residual_complex": residual_complex,
        "residual_energy": residual_mag,
        "residual_envelope": residual_envelope,
        "psf_gain": psf_gain,
        "psf_envelope": psf_envelope,
        "target": target_psf_normalized,
        "target_norm": norm["target_norm"],
        "target_log": norm["target_log"],
        "target_p99": norm["target_p99"],
    }



def soft_anatomy_mask_from_magnitude(
    magnitude: torch.Tensor,
    threshold: float = 0.05,
    softness: float = 0.02,
    smooth_kernel_size: int = 9,
    percentile: float = 99.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build a soft anatomy/support mask from an image magnitude.

    The mask is derived only from reconstructed/acquired-data images, such as
    the initial adjoint RSS or the frozen reconstruction magnitude.
    """
    if not torch.is_floating_point(magnitude):
        raise TypeError("magnitude must be a real floating tensor.")

    if magnitude.ndim != 3:
        raise ValueError(f"magnitude must have shape (B,H,W), got {tuple(magnitude.shape)}")

    b = magnitude.shape[0]
    flat = magnitude.reshape(b, -1)

    q = torch.quantile(
        flat.detach(),
        float(percentile) / 100.0,
        dim=1,
    ).to(device=magnitude.device, dtype=magnitude.dtype)

    mag_norm = magnitude / (q[:, None, None] + eps)

    soft = torch.sigmoid((mag_norm - float(threshold)) / (float(softness) + eps))

    if smooth_kernel_size > 1:
        soft = same_avg_pool2d(soft, kernel_size=int(smooth_kernel_size))

    return torch.clamp(soft, 0.0, 1.0)


def apply_support_mask_to_target(
    target: torch.Tensor,
    support_mask: torch.Tensor,
    power: float = 1.0,
) -> torch.Tensor:
    """
    Apply a soft support/anatomy mask to a residual-risk target.
    """
    if target.shape != support_mask.shape:
        raise ValueError(
            f"target and support_mask must have same shape, got "
            f"{tuple(target.shape)} and {tuple(support_mask.shape)}"
        )

    mask = torch.clamp(support_mask, 0.0, 1.0)

    if power != 1.0:
        mask = mask ** float(power)

    return target * mask
