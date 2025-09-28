from scipy import fft as sp_fft
from scipy._lib._array_api import xp_copy
from scipy.signal._signaltools import _centered


def _freq_domain_conv(in1, in2, axes, shape):
    """
    Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    axes : array_like of ints
        Axes over which to compute the FFTs.
    shape : array_like of ints
        The sizes of the FFTs.

    Returns
    -------
    out : array
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.

    """
    fshape = [sp_fft.next_fast_len(shape[a], True) for a in axes]

    fft, ifft = sp_fft.rfftn, sp_fft.irfftn

    sp1 = fft(in1, fshape, axes=axes)
    sp2 = fft(in2, fshape, axes=axes)

    ret = ifft(sp1 * sp2, fshape, axes=axes)

    fslice = tuple([slice(sz) for sz in shape])
    ret = ret[fslice]

    return ret


def _apply_conv_mode(ret, s1, s2, mode, axes):
    """
    Calculate the convolution result shape based on the `mode` argument.

    Returns the result sliced to the correct size for the given mode.

    Parameters
    ----------
    ret : array
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.

    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.

    """
    if mode == "full":
        return xp_copy(ret)
    elif mode == "same":
        return xp_copy(_centered(ret, s1))
    elif mode == "valid":
        shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1 for a in range(ret.ndim)]
        return xp_copy(_centered(ret, shape_valid))
    else:
        raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")


def fftconvolve(in1, in2, mode="full", axes=None):
    """
    Convolve two N-dimensional arrays using FFT.


    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    """

    s1 = in1.shape
    s2 = in2.shape
    axes = [0]
    shape = [len(in1) + len(in2) - 1]

    ret = _freq_domain_conv(in1, in2, axes, shape)

    return _apply_conv_mode(ret, s1, s2, mode, axes)
