from __future__ import annotations

import torch

from fourway_mri.operators import multicoil_forward


def masked_target_kspace(
    target_kspace: torch.Tensor,
    mask_1d: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_complex(target_kspace):
        raise TypeError("target_kspace must be complex.")

    mask = mask_1d.to(
        device=target_kspace.device,
        dtype=target_kspace.real.dtype,
    )[None, None, None, :]

    return target_kspace * mask


def ssdu_kspace_diagnostics(
    prediction: torch.Tensor,
    target_kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    lambda_rec_mask: torch.Tensor,
    eps: float = 1e-8,
) -> dict:
    pred_lambda = multicoil_forward(
        prediction,
        sensitivities,
        lambda_rec_mask,
    )

    target_lambda = masked_target_kspace(target_kspace, lambda_rec_mask)
    residual = pred_lambda - target_lambda

    target_l1 = torch.sum(torch.abs(target_lambda))
    target_l2_sq = torch.sum(torch.abs(target_lambda) ** 2)

    residual_l1 = torch.sum(torch.abs(residual))
    residual_l2_sq = torch.sum(torch.abs(residual) ** 2)

    rel_l1 = residual_l1 / (target_l1 + eps)
    rel_l2_sq = residual_l2_sq / (target_l2_sq + eps)

    return {
        "target_l1": target_l1,
        "target_l2_sq": target_l2_sq,
        "residual_l1": residual_l1,
        "residual_l2_sq": residual_l2_sq,
        "relative_l1": rel_l1,
        "relative_l2_sq": rel_l2_sq,
    }


def ssdu_loss(
    prediction: torch.Tensor,
    target_kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    lambda_rec_mask: torch.Tensor,
    loss_type: str = "relative_l1",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    diagnostics = ssdu_kspace_diagnostics(
        prediction=prediction,
        target_kspace=target_kspace,
        sensitivities=sensitivities,
        lambda_rec_mask=lambda_rec_mask,
        eps=eps,
    )

    if loss_type == "relative_l1":
        loss = diagnostics["relative_l1"]
    elif loss_type == "relative_l2_sq":
        loss = diagnostics["relative_l2_sq"]
    else:
        raise ValueError(
            f"Unknown loss_type={loss_type}. Use 'relative_l1' or 'relative_l2_sq'."
        )

    return loss, diagnostics
