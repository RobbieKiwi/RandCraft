from typing import Any

import numpy as np


def clean_1d_array(x: Any | np.ndarray) -> list[float]:
    if isinstance(x, np.ndarray):
        assert x.ndim == 1, "Input numpy array must be 1D"
        return [float(v) for v in x.tolist()]
    return [float(v) for v in x]
