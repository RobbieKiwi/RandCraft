import numpy as np
from scipy import fft as sp_fft


def _freq_domain_conv(in1: np.ndarray, in2: np.ndarray, N: int) -> np.ndarray:
    """
    Convolve two arrays in the frequency domain.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    N: int
        The length of the output array.

    Returns
    -------
    out : array
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.

    """
    fshape = [sp_fft.next_fast_len(N, True)]
    axes = [0]
    shape = [N]

    fft, ifft = sp_fft.rfftn, sp_fft.irfftn

    sp1 = fft(in1, fshape, axes=axes)
    sp2 = fft(in2, fshape, axes=axes)

    ret = ifft(sp1 * sp2, fshape, axes=axes)

    fslice = tuple([slice(sz) for sz in shape])
    ret = ret[fslice]

    return ret


def fftconvolve(in1, in2):
    """
    Convolve two N-dimensional arrays using FFT.


    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.


    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    """

    N = len(in1) + len(in2) - 1
    return _freq_domain_conv(in1=in1, in2=in2, N=N)
