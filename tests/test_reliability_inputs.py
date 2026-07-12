import torch

from fourway_mri.reliability_inputs import (
    analytical_cartesian_psf_from_mask,
    build_reliability_input_tensor,
    image_gradient_magnitude,
    robust_normalize_per_sample,
)


def make_complex(shape):
    real = torch.randn(*shape)
    imag = torch.randn(*shape)
    return torch.complex(real, imag).to(torch.complex64)


def test_robust_normalize_per_sample_shape_and_finiteness():
    x = torch.rand(2, 16, 20)

    y = robust_normalize_per_sample(x, percentile=99.0)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_analytical_cartesian_psf_shape_range_and_center_peak():
    risk_mask = torch.zeros(20)
    risk_mask[[2, 5, 9, 13, 17]] = 1.0

    psf = analytical_cartesian_psf_from_mask(
        risk_mask=risk_mask,
        height=16,
        normalize=True,
    )

    assert psf.shape == (16, 20)
    assert torch.isfinite(psf).all()
    assert psf.min().item() >= 0.0
    assert psf.max().item() <= 1.0 + 1e-6

    center_value = psf[psf.shape[0] // 2, psf.shape[1] // 2]
    assert center_value > 0.5


def test_image_gradient_magnitude_shape_and_nonnegative():
    x = torch.zeros(1, 16, 20)
    x[:, 8:, :] = 1.0

    grad = image_gradient_magnitude(x)

    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()
    assert grad.min().item() >= 0.0
    assert grad.max().item() > 0.0


def test_build_reliability_input_tensor_shape_finiteness_and_channels():
    torch.manual_seed(0)

    x_hat = make_complex((1, 16, 20))
    x0 = make_complex((1, 16, 20))

    support_mask = torch.zeros(1, 16, 20)
    support_mask[:, 4:12, 5:15] = 1.0

    psf_gain = torch.rand(1, 16, 20)

    risk_mask = torch.zeros(20)
    risk_mask[[1, 5, 9, 13, 17]] = 1.0

    out = build_reliability_input_tensor(
        x_hat=x_hat,
        x0=x0,
        support_mask=support_mask,
        risk_mask=risk_mask,
        psf_gain=psf_gain,
    )

    x = out["input"]

    assert x.shape == (1, 6, 16, 20)
    assert torch.isfinite(x).all()

    # support mask channel should remain in [0,1]
    assert x[:, 3].min().item() >= 0.0
    assert x[:, 3].max().item() <= 1.0

    # analytical PSF channel should be deterministic and normalized
    assert x[:, 4].max().item() <= 1.0 + 1e-6
