from typing import Any

import numpy as np


def clean_1d_array(x: Any | np.ndarray) -> list[float]:
    if isinstance(x, np.ndarray):
        assert x.ndim == 1, "Input numpy array must be 1D"
        return [float(v) for v in x.tolist()]
    return [float(v) for v in x]


def weighted_std(x: np.ndarray, weights: np.ndarray, unbiased: bool = True) -> float:
    average = np.average(x, weights=weights)
    variance = np.average((x - average) ** 2, weights=weights)
    if unbiased:
        eff_n = (weights.sum()) ** 2 / (np.sum(weights**2))
        if eff_n > 1:
            variance *= eff_n / (eff_n - 1)
    return np.sqrt(variance)
