import torch

from fourway_mri.operators import multicoil_forward
from fourway_mri.residuals import (
    apply_line_weights_to_kspace,
    cartesian_line_density_compensation,
    generate_residual_risk_target,
    risk_kspace_residual,
    same_avg_pool2d,
)


def make_complex(shape):
    real = torch.randn(*shape)
    imag = torch.randn(*shape)
    return torch.complex(real, imag).to(torch.complex64)


def test_cartesian_line_density_compensation_shape_and_mean():
    mask = torch.zeros(20)
    mask[[2, 3, 10, 17]] = 1.0

    weights = cartesian_line_density_compensation(mask, kernel_size=5)

    assert weights.shape == mask.shape
    assert torch.all(weights[mask == 0] == 0)

    sampled_mean = weights[mask > 0].mean()
    assert torch.allclose(sampled_mean, torch.tensor(1.0), atol=1e-5)


def test_apply_line_weights_to_kspace_shape():
    kspace = make_complex((1, 4, 16, 20))
    weights = torch.ones(20)

    out = apply_line_weights_to_kspace(kspace, weights)

    assert out.shape == kspace.shape
    assert torch.is_complex(out)


def test_same_avg_pool2d_preserves_shape_for_even_and_odd_kernels():
    x = torch.randn(1, 16, 20)

    y8 = same_avg_pool2d(x, kernel_size=8)
    y9 = same_avg_pool2d(x, kernel_size=9)

    assert y8.shape == x.shape
    assert y9.shape == x.shape


def test_risk_kspace_residual_zero_for_perfect_prediction():
    torch.manual_seed(0)

    image = make_complex((1, 16, 20))
    sensitivities = make_complex((1, 3, 16, 20))
    target_kspace = multicoil_forward(image, sensitivities, mask_1d=None)

    risk_mask = torch.zeros(20)
    risk_mask[::4] = 1.0

    residual = risk_kspace_residual(
        prediction=image,
        target_kspace=target_kspace,
        sensitivities=sensitivities,
        risk_mask=risk_mask,
    )

    assert torch.max(torch.abs(residual)).item() < 1e-5


def test_generate_residual_risk_target_outputs_are_finite():
    torch.manual_seed(0)

    prediction = make_complex((1, 16, 20))
    sensitivities = make_complex((1, 3, 16, 20))

    target_image = prediction + 0.05 * make_complex((1, 16, 20))
    target_kspace = multicoil_forward(target_image, sensitivities, mask_1d=None)

    risk_mask = torch.zeros(20)
    risk_mask[[1, 5, 9, 13, 17]] = 1.0

    out = generate_residual_risk_target(
        prediction=prediction,
        target_kspace=target_kspace,
        sensitivities=sensitivities,
        risk_mask=risk_mask,
        patch_size=4,
        dcf_kernel_size=5,
        psf_num_probes=2,
        psf_seed=123,
    )

    expected_keys = {
        "risk_residual_kspace",
        "dcf_weights",
        "residual_complex",
        "residual_energy",
        "residual_envelope",
        "psf_gain",
        "psf_envelope",
        "target",
        "target_norm",
        "target_log",
        "target_p99",
    }

    assert expected_keys.issubset(set(out.keys()))
    assert out["target"].shape == prediction.shape
    assert out["target_norm"].shape == prediction.shape
    assert out["target_log"].shape == prediction.shape

    assert torch.isfinite(out["target"]).all()
    assert torch.isfinite(out["target_norm"]).all()
    assert torch.isfinite(out["target_log"]).all()
    assert out["target_p99"].item() >= 0.0



def test_soft_anatomy_mask_shape_range_and_object_response():
    mag = torch.zeros(1, 32, 32)
    mag[:, 10:22, 10:22] = 1.0

    from fourway_mri.residuals import soft_anatomy_mask_from_magnitude

    mask = soft_anatomy_mask_from_magnitude(
        mag,
        threshold=0.1,
        softness=0.02,
        smooth_kernel_size=5,
    )

    assert mask.shape == mag.shape
    assert torch.isfinite(mask).all()
    assert mask.min().item() >= 0.0
    assert mask.max().item() <= 1.0

    center_mean = mask[:, 12:20, 12:20].mean()
    corner_mean = mask[:, :5, :5].mean()

    assert center_mean > corner_mean


def test_apply_support_mask_to_target_suppresses_background():
    from fourway_mri.residuals import apply_support_mask_to_target

    target = torch.ones(1, 16, 16)
    mask = torch.zeros(1, 16, 16)
    mask[:, 4:12, 4:12] = 1.0

    masked = apply_support_mask_to_target(target, mask)

    assert masked.shape == target.shape
    assert masked[:, :2, :2].max().item() == 0.0
    assert masked[:, 6:10, 6:10].mean().item() == 1.0
