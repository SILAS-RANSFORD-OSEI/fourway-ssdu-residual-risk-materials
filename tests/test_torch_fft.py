import torch

from fourway_mri.torch_fft import (
    chan_to_complex,
    complex_to_chan,
    fft2c,
    ifft2c,
    rss_combine_torch,
)


def test_fft_ifft_roundtrip_complex64():
    torch.manual_seed(0)

    real = torch.randn(2, 4, 32, 40)
    imag = torch.randn(2, 4, 32, 40)
    image = torch.complex(real, imag).to(torch.complex64)

    kspace = fft2c(image)
    recon = ifft2c(kspace)

    assert kspace.shape == image.shape
    assert recon.shape == image.shape
    assert torch.is_complex(kspace)
    assert torch.is_complex(recon)
    assert torch.allclose(recon, image, atol=1e-5, rtol=1e-5)


def test_complex_channel_conversion_roundtrip():
    torch.manual_seed(0)

    real = torch.randn(3, 32, 40)
    imag = torch.randn(3, 32, 40)
    x = torch.complex(real, imag).to(torch.complex64)

    x_chan = complex_to_chan(x)
    x_back = chan_to_complex(x_chan)

    assert x_chan.shape == (3, 2, 32, 40)
    assert x_back.shape == x.shape
    assert torch.is_complex(x_back)
    assert torch.allclose(x_back, x)


def test_rss_combine_torch_shape_and_value():
    coil_images = torch.ones(2, 4, 16, 16, dtype=torch.complex64)

    rss = rss_combine_torch(coil_images, coil_dim=1)

    assert rss.shape == (2, 16, 16)
    assert torch.allclose(rss, torch.full((2, 16, 16), 2.0))


def test_fft2c_rejects_real_tensor():
    x = torch.randn(1, 16, 16)

    try:
        fft2c(x)
        raised = False
    except TypeError:
        raised = True

    assert raised is True


def test_ifft2c_rejects_real_tensor():
    x = torch.randn(1, 16, 16)

    try:
        ifft2c(x)
        raised = False
    except TypeError:
        raised = True

    assert raised is True
