import numpy as np
from scipy.fft import irfftn as ifft
from scipy.fft import next_fast_len
from scipy.fft import rfftn as fft


def fftconvolve(arrs: list[np.ndarray]) -> np.ndarray:
    N = sum([len(arr) for arr in arrs]) - (len(arrs) - 1)
    fshape = [next_fast_len(N, True)]
    axes = [0]

    sps = np.stack([fft(arr, fshape, axes=axes) for arr in arrs])  # type: ignore
    # Use reduce to avoid complex number overflow issues
    prod = np.prod(sps, axis=0, dtype=np.complex128)

    ret = ifft(prod, fshape, axes=axes)

    # Ensure we get a proper numpy array and take real part
    ret_array = np.asarray(ret)
    return np.real(ret_array[:N])
