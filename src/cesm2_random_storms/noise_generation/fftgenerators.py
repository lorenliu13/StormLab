# The FFT code here was obtained from the pysteps package: https://pysteps.readthedocs.io/en/stable/

"""
pysteps.noise.fftgenerators
===========================

Methods for noise generators based on FFT filtering of white noise.

The methods in this module implement the following interface for filter
initialization depending on their parametric or nonparametric nature::

  initialize_param_2d_xxx_filter(field, **kwargs)

or::

  initialize_nonparam_2d_xxx_filter(field, **kwargs)

where field is an array of shape (m, n) or (t, m, n) that defines the target field
and optional parameters are supplied as keyword arguments.

The output of each initialization method is a dictionary containing the keys field
and input_shape. The first is a two-dimensional array of shape (m, int(n/2)+1)
that defines the filter. The second one is the shape of the input field for the
filter.

The methods in this module implement the following interface for the generation
of correlated noise::

  generate_noise_2d_xxx_filter(field, randstate=np.random, seed=None, **kwargs)

where field (m, n) is a filter returned from the corresponding initialization
method, and randstate and seed can be used to set the random generator and
its seed. Additional keyword arguments can be included as a dictionary.

The output of each generator method is a two-dimensional array containing the
field of correlated noise cN of shape (m, n).

.. autosummary::
    :toctree: ../generated/

    initialize_param_2d_fft_filter
    initialize_nonparam_2d_fft_filter
    initialize_nonparam_2d_nested_filter
    initialize_nonparam_2d_ssft_filter
    generate_noise_2d_fft_filter
    generate_noise_2d_ssft_filter
"""

import numpy as np
from scipy import optimize
import sys
import numpy.fft as fft



def _hann(R):
    W = np.ones_like(R)
    N = min(R.shape[0], R.shape[1])
    mask = R > int(N / 2)

    W[mask] = 0.0
    W[~mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * (R[~mask] + int(N / 2)) / N))

    return W


def _tukey(R, alpha):
    W = np.ones_like(R)
    N = min(R.shape[0], R.shape[1])

    mask1 = R < int(N / 2)
    mask2 = R > int(N / 2) * (1.0 - alpha)
    mask = np.logical_and(mask1, mask2)
    W[mask] = 0.5 * (
        1.0 + np.cos(np.pi * (R[mask] / (alpha * 0.5 * N) - 1.0 / alpha + 1.0))
    )
    mask = R >= int(N / 2)
    W[mask] = 0.0

    return W


def compute_window_function(m, n, func, **kwargs):
    """
    Compute window function for a two-dimensional rectangular region. Window
    function-specific parameters are given as keyword arguments.

    Parameters
    ----------
    m: int
        Height of the array.
    n: int
        Width of the array.
    func: str
        The name of the window function.
        The currently implemented functions are
        'hann' and 'tukey'.

    Other Parameters
    ----------------
    alpha: float
        Applicable if func is 'tukey'.

    Notes
    -----
    Two-dimensional tapering weights are computed from one-dimensional window
    functions using w(r), where r is the distance from the center of the
    region.

    Returns
    -------
    out: array
        Array of shape (m, n) containing the tapering weights.
    """
    X, Y = np.meshgrid(np.arange(n), np.arange(m))
    R = np.sqrt((X - int(n / 2)) ** 2 + (Y - int(m / 2)) ** 2)

    if func == "hann":
        return _hann(R)
    elif func == "tukey":
        alpha = kwargs.get("alpha", 0.2)

        return _tukey(R, alpha)
    else:
        raise ValueError("invalid window function '%s'" % func)


def initialize_nonparam_2d_fft_filter(field, **kwargs):
    """
    Takes one ore more 2d input fields and produces one non-parametric, global
    and anisotropic fourier filter.

    Parameters
    ----------
    field: array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite. If more than one field are passed,
        the average fourier filter is returned. It assumes that fields are stacked
        by the first axis: [nr_fields, y, x].

    Other Parameters
    ----------------
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    donorm: bool
        Option to normalize the real and imaginary parts.
        Default: False
    rm_rdisc: bool
        Whether or not to remove the rain/no-rain disconituity (default True).
        It assumes no-rain pixels are assigned with lowest value.
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".

    Returns
    -------
    out: dict
        A dictionary containing the keys field and input_shape. The first is a
        two-dimensional array of shape (m, int(n/2)+1) that defines the filter.
        The second one is the shape of the input field for the filter.

        It can be passed to
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_fft_filter`.
    """
    if len(field.shape) < 2 or len(field.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(~np.isfinite(field)):
        raise ValueError("field contains non-finite values")

    # defaults
    win_fun = kwargs.get("win_fun", "tukey")
    donorm = kwargs.get("donorm", False)
    rm_rdisc = kwargs.get("rm_rdisc", True)
    use_full_fft = kwargs.get("use_full_fft", False)
    # fft = kwargs.get("fft_method", "numpy")
    # if type(fft) == str:
    #     fft_shape = field.shape if len(field.shape) == 2 else field.shape[1:]
    #     fft = utils.get_method(fft, shape=fft_shape)

    field = field.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        field[field > field.min()] -= field[field > field.min()].min() - field.min()

    # dims
    if len(field.shape) == 2:
        field = field[None, :, :]
    nr_fields = field.shape[0]
    field_shape = field.shape[1:]
    if use_full_fft:
        fft_shape = (field.shape[1], field.shape[2])
    else:
        fft_shape = (field.shape[1], int(field.shape[2] / 2) + 1)

    # make sure non-rainy pixels are set to zero
    field -= field.min(axis=(1, 2))[:, None, None]

    if win_fun is not None:
        tapering = compute_window_function(
            field_shape[0], field_shape[1], win_fun
        )
    else:
        tapering = np.ones(field_shape)

    F = np.zeros(fft_shape, dtype=complex)
    for i in range(nr_fields):
        if use_full_fft:
            F += fft.fft2(field[i, :, :] * tapering)
        else:
            F += fft.rfft2(field[i, :, :] * tapering)
    F /= nr_fields

    # normalize the real and imaginary parts
    if donorm:
        if np.std(F.imag) > 0:
            F.imag = (F.imag - np.mean(F.imag)) / np.std(F.imag)
        if np.std(F.real) > 0:
            F.real = (F.real - np.mean(F.real)) / np.std(F.real)

    return {
        "field": np.abs(F),
        "input_shape": field.shape[1:],
        "use_full_fft": use_full_fft,
    }


def initialize_nonparam_2d_ssft_filter(field, **kwargs):
    """
    Function to compute the local Fourier filters using the Short-Space Fourier
    filtering approach.

    Parameters
    ----------
    field: array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite. If more than one field are passed,
        the average fourier filter is returned. It assumes that fields are stacked
        by the first axis: [nr_fields, y, x].

    Other Parameters
    ----------------
    win_size: int or two-element tuple of ints
        Size-length of the window to compute the SSFT (default (128, 128)).
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    overlap: float [0,1[
        The proportion of overlap to be applied between successive windows
        (default 0.3).
    war_thr: float [0,1]
        Threshold for the minimum fraction of rain needed for computing the FFT
        (default 0.1).
    rm_rdisc: bool
        Whether or not to remove the rain/no-rain disconituity. It assumes no-rain
        pixels are assigned with lowest value.
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".

    Returns
    -------
    field: array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.
        It can be passed to
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter`.

    References
    ----------
    :cite:`NBSG2017`
    """

    if len(field.shape) < 2 or len(field.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(np.isnan(field)):
        raise ValueError("field must not contain NaNs")

    # defaults
    win_size = kwargs.get("win_size", (128, 128))
    if type(win_size) == int:
        win_size = (win_size, win_size)
    win_fun = kwargs.get("win_fun", "tukey")
    overlap = kwargs.get("overlap", 0.3)
    war_thr = kwargs.get("war_thr", 0.1)
    rm_rdisc = kwargs.get("rm_disc", True)
    # fft = kwargs.get("fft_method", "numpy")
    # if type(fft) == str:
    #     fft_shape = field.shape if len(field.shape) == 2 else field.shape[1:]
    #     fft = utils.get_method(fft, shape=fft_shape)

    field = field.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        field[field > field.min()] -= field[field > field.min()].min() - field.min()

    # dims
    if len(field.shape) == 2:
        field = field[None, :, :]
    nr_fields = field.shape[0]
    dim = field.shape[1:]
    dim_x = dim[1]
    dim_y = dim[0]

    # make sure non-rainy pixels are set to zero
    field -= field.min(axis=(1, 2))[:, None, None]

    # SSFT algorithm

    # prepare indices
    idxi = np.zeros(2, dtype=int)
    idxj = np.zeros(2, dtype=int)

    # number of windows
    num_windows_y = np.ceil(float(dim_y) / win_size[0]).astype(int)
    num_windows_x = np.ceil(float(dim_x) / win_size[1]).astype(int)

    # domain fourier filter
    # the fourier filter for the entire domain, no need to use windows function
    F0 = initialize_nonparam_2d_fft_filter(
        field, win_fun=None, donorm=True, use_full_fft=True, fft_method=fft
    )["field"]
    # and allocate it to the final grid
    F = np.zeros((num_windows_y, num_windows_x, F0.shape[0], F0.shape[1]))
    F += F0[np.newaxis, np.newaxis, :, :]

    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):

            # compute indices of local window
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0))) # get the upper y index
            idxi[1] = int(
                np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y)) # get the lower y index
            )
            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0))) # get the left x index
            idxj[1] = int(
                np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x)) # get the right x index
            )

            # build localization mask
            # TODO: the 0.01 rain threshold must be improved
            mask = _get_mask(dim, idxi, idxj, win_fun)
            war = float(np.sum((field * mask[None, :, :]) > 0.01)) / (
                (idxi[1] - idxi[0]) * (idxj[1] - idxj[0]) * nr_fields
            ) # count the number of pixels with value > 0.01 after masking
            # then compute the ratio of the number to total pixels in the window

            if war > war_thr: # if the wet area ratio is greater than 0.1 (default)
                # the new filter
                F[i, j, :, :] = initialize_nonparam_2d_fft_filter(
                    field * mask[None, :, :], # this is already applied a window function
                    win_fun=None,
                    donorm=True,
                    use_full_fft=True,
                    fft_method=fft,
                )["field"]
            # the function returns a matrix of (window_number_y, window_number_x, y_dim, x_dim),
            # each element is a matrix of F_field generated by the FFT the rainfall field * mask in the window.
            # if the WAR ratio is not meet,
            # the element is the global FFT field
            # note that the value is the absolute value rather than the complex value for each element...(What's the implication?) 11/23/2022
    return {"field": F, "input_shape": field.shape[1:], "use_full_fft": True}


def generate_noise_2d_ssft_filter(F, randstate=None, seed=None, **kwargs):
    """
    Function to compute the locally correlated noise using a nested approach.

    Parameters
    ----------
    F: array-like
        A filter object returned by
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter` or
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_nested_filter`.
        The filter is a four-dimensional array containing the 2d fourier filters
        distributed over a 2d spatial grid.
    randstate: mtrand.RandomState
        Optional random generator to use. If set to None, use numpy.random.
    seed: int
        Value to set a seed for the generator. None will not set the seed.

    Other Parameters
    ----------------
    overlap: float
        Percentage overlap [0-1] between successive windows (default 0.2).
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".

    Returns
    -------
    N: array-like
        A two-dimensional numpy array of non-stationary correlated noise.
    """
    input_shape = F["input_shape"]
    use_full_fft = F["use_full_fft"]
    F = F["field"]

    if len(F.shape) != 4:
        raise ValueError("the input is not four-dimensional array")
    if np.any(~np.isfinite(F)):
        raise ValueError("field contains non-finite values")

    if "domain" in kwargs.keys() and kwargs["domain"] == "spectral":
        raise NotImplementedError(
            "SSFT-based noise generator is not implemented in the spectral domain"
        )

    # defaults
    overlap = kwargs.get("overlap", 0.2)
    win_fun = kwargs.get("win_fun", "tukey")
    # fft = kwargs.get("fft_method", "numpy")
    # if type(fft) == str:
    #     fft = utils.get_method(fft, shape=input_shape)

    if randstate is None:
        randstate = np.random

    # set the seed
    if seed is not None:
        randstate.seed(seed)

    dim_y = F.shape[2]
    dim_x = F.shape[3]
    dim = (dim_y, dim_x)

    # produce fields of white noise
    N = randstate.randn(dim_y, dim_x)
    fN = fft.fft2(N) # get the white noise FFT field

    # initialize variables
    cN = np.zeros(dim) # intialize a zero field of rainfall field size, this saves the noise
    sM = np.zeros(dim) # this saves the tapering window?

    idxi = np.zeros(2, dtype=int)
    idxj = np.zeros(2, dtype=int)

    # get the window size
    win_size = (float(dim_y) / F.shape[0], float(dim_x) / F.shape[1])

    # loop the windows and build composite image of correlated noise

    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):

            # apply fourier filtering with local filter
            lF = F[i, j, :, :] # get the rainfall window FFT field
            flN = fN * lF # compute F_rain * F_noise
            flN = np.array(fft.ifft2(flN).real) # get the noise field from inverse FFT

            # compute indices of local window
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))
            idxi[1] = int(
                np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y))
            )
            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0)))
            idxj[1] = int(
                np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x))
            )

            # build mask and add local noise field to the composite image
            M = _get_mask(dim, idxi, idxj, win_fun) # get the mask with a tapering window
            cN += flN * M # add the noise in that window with weights of the tapering
            sM += M # add the tapering window as weights

    # normalize the field
    cN[sM > 0] /= sM[sM > 0] # 计算加权平均 (noise_1 * mask_1 + noise_2*mask_2) / (mask_1 + mask_2)
    cN = (cN - cN.mean()) / cN.std() # normalize

    return cN


def generate_noise_2d_ssft_filter_new(F, N, **kwargs):
    """
    Function to compute the locally correlated noise using a nested approach.
    The noise array is an additional input

    Parameters
    ----------
    F: array-like
        A filter object returned by
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter` or
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_nested_filter`.
        The filter is a four-dimensional array containing the 2d fourier filters
        distributed over a 2d spatial grid.
    randstate: mtrand.RandomState
        Optional random generator to use. If set to None, use numpy.random.
    seed: int
        Value to set a seed for the generator. None will not set the seed.

    Other Parameters
    ----------------
    overlap: float
        Percentage overlap [0-1] between successive windows (default 0.2).
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".

    Returns
    -------
    N: array-like
        A two-dimensional numpy array of non-stationary correlated noise.
    """
    input_shape = F["input_shape"]
    use_full_fft = F["use_full_fft"]
    F = F["field"]

    if len(F.shape) != 4:
        raise ValueError("the input is not four-dimensional array")
    if np.any(~np.isfinite(F)):
        raise ValueError("field contains non-finite values")

    if "domain" in kwargs.keys() and kwargs["domain"] == "spectral":
        raise NotImplementedError(
            "SSFT-based noise generator is not implemented in the spectral domain"
        )

    # defaults
    overlap = kwargs.get("overlap", 0.2)
    win_fun = kwargs.get("win_fun", "tukey")
    # fft = kwargs.get("fft_method", "numpy")
    # if type(fft) == str:
    #     fft = utils.get_method(fft, shape=input_shape)

    dim_y = F.shape[2]
    dim_x = F.shape[3]
    dim = (dim_y, dim_x)

    # produce fields of white noise
    # N = randstate.randn(dim_y, dim_x)
    fN = fft.fft2(N) # get the white noise FFT field

    # initialize variables
    cN = np.zeros(dim) # intialize a zero field of rainfall field size, this saves the noise
    sM = np.zeros(dim) # this saves the tapering window?

    idxi = np.zeros(2, dtype=int)
    idxj = np.zeros(2, dtype=int)

    # get the window size
    win_size = (float(dim_y) / F.shape[0], float(dim_x) / F.shape[1])

    # loop the windows and build composite image of correlated noise

    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):

            # apply fourier filtering with local filter
            lF = F[i, j, :, :] # get the rainfall window FFT field
            flN = fN * lF # compute F_rain * F_noise
            flN = np.array(fft.ifft2(flN).real) # get the noise field from inverse FFT

            # compute indices of local window
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))
            idxi[1] = int(
                np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y))
            )
            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0)))
            idxj[1] = int(
                np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x))
            )

            # build mask and add local noise field to the composite image
            M = _get_mask(dim, idxi, idxj, win_fun) # get the mask with a tapering window
            cN += flN * M # add the noise in that window with weights of the tapering
            sM += M # add the tapering window as weights

    # normalize the field
    cN[sM > 0] /= sM[sM > 0] # 计算加权平均 (noise_1 * mask_1 + noise_2*mask_2) / (mask_1 + mask_2)
    cN = (cN - cN.mean()) / cN.std() # normalize

    return cN



def _split_field(idxi, idxj, Segments):
    """Split domain field into a number of equally sapced segments."""

    sizei = idxi[1] - idxi[0]
    sizej = idxj[1] - idxj[0]

    winsizei = int(sizei / Segments)
    winsizej = int(sizej / Segments)

    Idxi = np.zeros((Segments**2, 2))
    Idxj = np.zeros((Segments**2, 2))

    count = -1
    for i in range(Segments):
        for j in range(Segments):
            count += 1
            Idxi[count, 0] = idxi[0] + i * winsizei
            Idxi[count, 1] = np.min((Idxi[count, 0] + winsizei, idxi[1]))
            Idxj[count, 0] = idxj[0] + j * winsizej
            Idxj[count, 1] = np.min((Idxj[count, 0] + winsizej, idxj[1]))

    Idxi = np.array(Idxi).astype(int)
    Idxj = np.array(Idxj).astype(int)

    return Idxi, Idxj


def _get_mask(Size, idxi, idxj, win_fun):
    """Compute a mask of zeros with a window at a given position."""

    idxi = np.array(idxi).astype(int)
    idxj = np.array(idxj).astype(int)

    win_size = (idxi[1] - idxi[0], idxj[1] - idxj[0])
    if win_fun is not None:
        wind = compute_window_function(win_size[0], win_size[1], win_fun)
        wind += 1e-6  # avoid zero values

    else:
        # wind = np.ones(win_size) # this function is wrong, wrong array index, 11/23/2022
        wind = np.ones((win_size[0], win_size[1])) # this is right

    mask = np.zeros(Size) # create a zero array with shape of the rainfall field
    mask[idxi.item(0) : idxi.item(1), idxj.item(0) : idxj.item(1)] = wind

    return mask



if __name__ == "__main__":

    print('Test')