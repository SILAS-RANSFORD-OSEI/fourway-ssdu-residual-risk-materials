import torch

from fourway_mri.operators import (
    apply_kspace_mask,
    multicoil_adjoint,
    multicoil_forward,
    normal_operator,
)


def make_complex(shape):
    real = torch.randn(*shape)
    imag = torch.randn(*shape)
    return torch.complex(real, imag).to(torch.complex64)


def test_apply_kspace_mask_shape():
    kspace = torch.ones(2, 4, 16, 20, dtype=torch.complex64)
    mask = torch.zeros(20)
    mask[::2] = 1

    masked = apply_kspace_mask(kspace, mask)

    assert masked.shape == kspace.shape
    assert torch.all(masked[..., 1] == 0)
    assert torch.all(masked[..., 0] == 1)


def test_forward_shape():
    image = make_complex((2, 16, 20))
    sens = make_complex((2, 4, 16, 20))
    mask = torch.ones(20)

    kspace = multicoil_forward(image, sens, mask)

    assert kspace.shape == (2, 4, 16, 20)
    assert torch.is_complex(kspace)


def test_adjoint_shape():
    kspace = make_complex((2, 4, 16, 20))
    sens = make_complex((2, 4, 16, 20))
    mask = torch.ones(20)

    image = multicoil_adjoint(kspace, sens, mask)

    assert image.shape == (2, 16, 20)
    assert torch.is_complex(image)


def test_adjoint_inner_product_property():
    torch.manual_seed(0)

    image = make_complex((1, 16, 20))
    sens = make_complex((1, 4, 16, 20))
    y = make_complex((1, 4, 16, 20))

    mask = torch.zeros(20)
    mask[::2] = 1

    ex = multicoil_forward(image, sens, mask)
    ehy = multicoil_adjoint(y, sens, mask)

    lhs = torch.sum(torch.conj(ex) * y)
    rhs = torch.sum(torch.conj(image) * ehy)

    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


def test_normal_operator_shape_and_gradient():
    torch.manual_seed(0)

    image = make_complex((1, 16, 20))
    image.requires_grad_(True)

    sens = make_complex((1, 4, 16, 20))
    mask = torch.zeros(20)
    mask[::2] = 1

    out = normal_operator(image, sens, mask)
    loss = torch.mean(torch.abs(out) ** 2)
    loss.backward()

    assert out.shape == image.shape
    assert image.grad is not None
    assert torch.isfinite(image.grad.real).all()
    assert torch.isfinite(image.grad.imag).all()
