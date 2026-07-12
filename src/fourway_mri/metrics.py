from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_laplace
from skimage.metrics import structural_similarity


def nmse(reference: np.ndarray, prediction: np.ndarray, eps: float = 1e-12) -> float:
    numerator = np.sum((reference - prediction) ** 2)
    denominator = np.sum(reference ** 2) + eps
    return float(numerator / denominator)


def mse(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((reference - prediction) ** 2))


def psnr(
    reference: np.ndarray,
    prediction: np.ndarray,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> float:
    err = mse(reference, prediction)
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(err + eps))


def ssim(
    reference: np.ndarray,
    prediction: np.ndarray,
    data_range: float = 1.0,
) -> float:
    return float(
        structural_similarity(
            reference,
            prediction,
            data_range=data_range,
        )
    )


def hfen(
    reference: np.ndarray,
    prediction: np.ndarray,
    sigma: float = 1.5,
    eps: float = 1e-12,
) -> float:
    ref_log = gaussian_laplace(reference, sigma=sigma)
    pred_log = gaussian_laplace(prediction, sigma=sigma)
    numerator = np.linalg.norm(ref_log - pred_log)
    denominator = np.linalg.norm(ref_log) + eps
    return float(numerator / denominator)
