"""Matrix operation utility functions."""
import warnings

import numpy as np
from numpy.lib.stride_tricks import as_strided


def sliding_window(data, window, step=1, padded=False, axis=-1, copy=True):
    """Calculate a sliding window over a signal.

    Parameters
    ----------
    data : array
        The array to be slided over.
    window : int
        The sliding window size.
    step : int
        The sliding window stepsize (default=1).
    axis : int
        The axis to slide over (defaults=-1).
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.

    Returns
    -------
    data :  array, shape=(..., n_windows, window_size)
        A matrix whose last dimension corresponds to the window size, and the
        second-to-last dimension corresponds to the number of slices.

    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occur.

    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])

    """
    if axis >= data.ndim:
        raise ValueError("Axis value out of range")
    if step < 1:
        raise ValueError("Stepsize may not be zero or negative")
    if window > data.shape[axis]:
        print("Sliding window size exceeds size of selected axis")
        return data[..., None]

    shape = list(data.shape)
    shape[axis] = np.floor(
        data.shape[axis] / step - window / step + 1).astype(int)
    shape.append(window)

    strides = list(data.strides)
    strides[axis] *= step
    strides.append(data.strides[axis])
    strided = as_strided(data, shape=shape, strides=strides)

    if copy:
        return strided.copy()
    else:
        return strided


def widen_mask(mask, widen=4, axis=0):
    """Widen each 'bad' section of a binary mask by `n` cells.

    Parameters
    ----------
    mask : array
        Masking array. If an element of the mask is True (or 1), the
        corresponding element of the associated array is masked (marked as
        invalid).
    widen : int
        Number of cells to widen mask by.
    axis : int
        Axis to operate on.

    Returns
    -------
    out : array
        Widened mask, of same shape as input mask.

    Examples
    --------
    >> test = widen_mask(np.array([False, False, False, True, False], 1)
    >> print(test)
    [False False False True True]]

    """
    dims = mask.ndim
    dtype = mask.dtype

    if axis > dims - 1:
        raise AttributeError('Invalid `axis` value.')

    if widen < 0:
        # This places the desired axis at the front of the shape tuple, then
        # reverses that first axis, and then returns it to its original
        # position.
        if dims > 1:
            mask = np.swapaxes(np.swapaxes(mask, 0, axis)[::-1], 0, axis)
        else:
            mask = mask[::-1]

    def repeat_or(a, n=1):
        """Take a mask and applies logical OR to shifted versions of itself."""
        m = a.copy().astype(bool)
        k = m.copy()
        if n == 0:
            return k
        n = n + 1  # trick so that n=1 means "expand mask by one"

        # lenM and lenK : how many subsequent Trues there are at least
        lenM, lenK = 1, 1

        # run until a combination of both masks has n or more subsequent Trues
        while lenM + lenK < n:
            # append what we have in k to the end of what we have in m
            m[lenM:] |= k[:-lenM]
            m, k = k, m  # swap so that m is again the small one

            # update the lengths
            lenM, lenK = lenK, lenM + lenK

        # see how much m has to be shifted in order to append the missing Trues
        k[n - lenM:] |= m[:-n + lenM]

        return k

    if dims > 1:  # apply along given axis
        out = np.apply_along_axis(repeat_or, axis, mask, n=np.abs(widen))
    else:
        out = repeat_or(mask, n=np.abs(widen))

    if widen < 0:  # un-reverse axis
        if dims > 1:
            out = np.swapaxes(np.swapaxes(out, 0, axis)[::-1], 0, axis)
        else:
            out = out[::-1]

    return out.astype(dtype)


def relshift(X, ref, shifts, fill_value=0, axis=0):
    """Create shifted versions of X relative to ref with padding.

    `ref` is replicated to have the same shape as `X` and padded accordingly.

    Parameters
    ----------
    X : array, shape=(n_samples[, n_epochs][, n_trials])
        Array to shift.
    ref : array, shape=(n_samples[, n_epochs][, n_trials])
        Reference array against which `X` is shifted.
    shifts : array | int
        Array of shifts. Positive shifts mean that X is 'delayed' in time (i.e.
        `y[shift] = X[0]`). Conversely, a negative shift means that X is
        'advanced' (i.e. y[0] =
    fill_value : float
        Value to pad output axis by.
    axis : int
        The axis along which elements are shifted.

    Returns
    -------
    y : array, shape=(n_samples[, n_epochs][, n_trials], n_shifts)
        Shifted array.
    y_ref : array, shape=(n_samples[, n_epochs][, n_trials], n_shifts)
        Reference array, repeated to match `y.shape`. Padding matches that of
        `y`.

    See Also
    --------
    multishift, shift, shiftnd

    """
    shifts, n_shifts = _check_shifts(shifts)
    X = _check_data(X)
    ref = _check_data(ref)

    if X.shape[0] != ref.shape[0]:
        raise AttributeError('X and ref must have same n_times')

    # First we delay X
    y = multishift(X, shifts=shifts, axis=axis, fill_value=fill_value)

    # Then we create as many copies of ref as there are lags
    y_ref = multishift(ref, shifts=np.zeros(n_shifts), axis=axis)

    # We need to find out the indices of the padded values in `y`. For this we
    # use a hack where we feed in an array of ones to multishift(), with a
    # known `fill_value`.
    temp = multishift(np.ones_like(ref), shifts=shifts, axis=axis,
                      fill_value=0)
    mask = temp == 0
    if mask.any():
        y_ref[mask] = fill_value

    return y, y_ref


def multishift(X, shifts, fill_value=0, axis=0, keep_dims=False,
               reshape=False, solution='full'):
    """Apply several shifts along specified axis.

    If `shifts` has multiple values, the output will contain one shift per
    page. Shifted data are padded with `fill_value`.

    Parameters
    ----------
    X : array, shape=(n_samples[, n_chans][, n_trials])
        Array to shift.
    shifts : array, shape=(n_shifts,)
        Array of shifts. Positive shifts mean that X is 'delayed' in time (i.e.
        `y[shift] = X[0]`). Conversely, a negative shift means that X is
        'advanced' (i.e. y[0] =
    fill_value : float | np.nan
        Value to pad output axis by.
    axis : int, optional
        The axis along which elements are shifted.
    keep_dims : bool
        If True, keep singleton dimensions in output.
    reshape : bool
        If True, concatenate channels and lags, yielding an array of shape
        (n_samples, n_chans*n_shifts[, n_trials])
    solution : {'valid', 'full'}
        If `valid`, the output's is cropped along `axis` by `n_shifts` in order
        to remove edge artifacts. If `full`, the output has the same size as X.

    Returns
    -------
    y : array, shape=(n_samples[, n_chans][, n_trials], n_shifts)
        Shifted array.

    See Also
    --------
    relshift, shift, shiftnd

    """
    shifts, n_shifts = _check_shifts(shifts)
    X = _check_data(X)
    n_samples, n_chans, n_trials = theshapeof(X)

    if np.array_equal(shifts, [0]):
        return X.copy()

    # Loop over shifts
    y = np.zeros(X.shape + (n_shifts,))
    for i, s in enumerate(shifts):
        y[..., i] = shift(X, shift=s, fill_value=fill_value, axis=axis)

    if reshape is True:
        if X.ndim == 3:  # n_samples, n_chans, n_trials, n_shifts
            y = np.swapaxes(y, 2, 3)
            y = np.reshape(y, (n_samples, n_chans * n_shifts, n_trials))
        elif X.ndim == 2:  # n_samples, n_chans, n_shifts
            y = np.reshape(y, (n_samples, n_chans * n_shifts))

    if n_shifts == 1 and not keep_dims:
        y = np.squeeze(y, axis=-1)

    if solution == 'valid':
        max_neg_shift = np.abs(np.min(np.min(shifts), 0))
        max_pos_shift = np.max((np.max(shifts), 0))
        y = y[max_pos_shift:-max_neg_shift, ...]

    return y


def multismooth(X, smooths, axis=0, keep_dims=False):
    """Apply several shifts along specified axis.

    If `shifts` has multiple values, the output will contain one shift per
    trial. Shifted data are padded with `fill_value`.

    Parameters
    ----------
    X : array, shape=(n_samples[, n_epochs][, n_trials])
        Array to shift.
    smooths : array
        Array of smoothing values (in samples).
    axis : int, optional
        The axis along which elements are shifted (default=0).
    keep_dims : bool
        If True, keep singleton dimensions in output.

    Returns
    -------
    y : array, shape=(n_samples[, n_epochs][, n_trials], n_shifts)
        Shifted array.

    See Also
    --------
    multishift, smooth

    """
    from .sig import smooth

    smooths, n_smooths = _check_shifts(smooths, allow_floats=True)
    X = _check_data(X)

    # Loop over shifts
    y = np.zeros(X.shape + (n_smooths,))
    for i, s in enumerate(smooths):
        y[..., i] = smooth(X, window_len=s, axis=axis)

    if n_smooths == 1 and not keep_dims:
        y = np.squeeze(y, axis=-1)

    return y


def shift(X, shift, fill_value=0, axis=0):
    """Shift array along its first, second or last dimension.

    Output is padded by `fill_value`.

    Parameters
    ----------
    X : array, shape=(n_samples[, n_epochs][, n_trials])
        Multidimensional input array.
    shift : int
        The number of places by which elements are shifted along axis. Positive
        shifts mean that X is 'delayed' in time (i.e. `y[shift] = X[0]`).
        Conversely, a negative shift means that X is 'advanced' (i.e. y[0] =
        X[shift]).
    fill_value : float
        Value to pad output axis by.
    axis : int, optional
        The axis along which elements are shifted.

    Returns
    -------
    y : array
        Output array, with the same shape as `X`.

    See Also
    --------
    relshift, multishift, shiftnd

    """
    if not np.equal(np.mod(shift, 1), 0):
        raise AttributeError('shift must be a single int')

    # reallocate empty array and assign slice.
    y = np.empty_like(X)

    if shift == 0:
        y[:] = X
    else:
        if axis == 0:
            if shift > 0:
                y[:shift, ...] = fill_value
                y[shift:, ...] = X[:-shift]
            elif shift < 0:
                y[shift:,  ...] = fill_value
                y[:shift,  ...] = X[-shift:]

        elif axis == 1:
            if shift > 0:
                y[:, :shift, ...] = fill_value
                y[:, shift:, ...] = X[:, :-shift]
            elif shift < 0:
                y[:, shift:,  ...] = fill_value
                y[:, :shift,  ...] = X[:, -shift:]

        elif axis == -1:
            if shift > 0:
                y[..., :shift] = fill_value
                y[..., shift:] = X[..., :-shift]
            elif shift < 0:
                y[..., shift:] = fill_value
                y[..., :shift] = X[..., -shift:]

        else:
            raise NotImplementedError('Axis must be 0, 1 or -1.')

    return y


def shiftnd(X, shift, fill_value=0, axis=None):
    """Roll array elements along a given axis with padding.

    Elements off the end of the array are treated as zeros. This function is
    slower than function:`shift`, so prefer the latter when possible.

    Parameters
    ----------
    X : array
        Multidimensional input array.
    shift : int
        The number of places by which elements are shifted along axis.
    fill_value : float
        Value to pad output axis by.
    axis : int, optional
        The axis along which elements are shifted. By default, the array is
        flattened before shifting, after which the original shape is restored.

    Returns
    -------
    y : array, (n_samples, [n_epochs, ][n_trials, ])
        Output array, with the same shape as `X`.

    See Also
    --------
    np.roll     : Elements that roll off one end come back on the other.
    np.rollaxis : Roll the specified axis backwards, until it lies in a given
                  position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> shiftnd(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> shiftnd(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> shiftnd(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> shiftnd(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> shiftnd(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> shiftnd(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])

    """
    X = np.asanyarray(X)
    if shift == 0:
        return X

    if axis is None:
        n = X.size
        reshape = True
    else:
        n = X.shape[axis]
        reshape = False

    if np.abs(shift) > n:
        y = np.ones_like(X) * fill_value
    elif shift < 0:
        shift += n
        pad = np.ones_like(X.take(np.arange(n - shift), axis)) * fill_value
        y = np.concatenate((X.take(np.arange(n - shift, n), axis), pad), axis)
    else:
        pad = np.ones_like(X.take(np.arange(n - shift, n), axis)) * fill_value
        y = np.concatenate((pad, X.take(np.arange(n - shift), axis)), axis)

    if reshape:
        return y.reshape(X.shape)
    else:
        return y


def theshapeof(X):
    """Return the shape of X."""
    X = _check_data(X)
    # if not isinstance(X, np.ndarray):
    #     raise AttributeError('X must be a numpy array')

    if X.ndim == 3:
        return X.shape[0], X.shape[1], X.shape[2]
    elif X.ndim == 2:
        return X.shape[0], X.shape[1], 1
    elif X.ndim == 1:
        return X.shape[0], 1, 1
    else:
        raise ValueError("Array contains more than 3 dimensions")


def unsqueeze(X):
    """Append singleton dimensions to an array."""
    X = _check_data(X)
    if X.shape != theshapeof(X):
        return X.reshape(theshapeof(X))
    else:
        return X


def fold(X, epoch_size):
    """Fold 2D X into 3D."""
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if X.ndim > 2:
        raise AttributeError('X must be 2D at most')

    n_chans = X.shape[0] // epoch_size
    if X.shape[0] / epoch_size >= 1:
        X = np.transpose(np.reshape(X, (epoch_size, n_chans, X.shape[1]),
                                    order="F").copy(), [0, 2, 1])
    return X


def unfold(X):
    """Unfold 3D X into 2D (concatenate trials)."""
    n_samples, n_chans, n_trials = theshapeof(X)
    if X.size == 0:
        return X

    if X.shape == (n_samples,):
        X = X[:, None]

    if n_trials > 1 or X.ndim == 3:
        return np.reshape(
            np.transpose(X, (0, 2, 1)),
            (n_samples * n_trials, n_chans), order="F")
    else:
        return X


def normcol(X, weights=None, return_norm=False):
    """Normalize each column so that its weighted mean square value is 1.

    If X is 3D, pages are concatenated vertically before calculating the
    norm.

    Weight should be either a column vector, or a matrix (2D or 3D) of same
    size as X.

    Parameters
    ----------
    X : array
        X to normalize.
    weights : array
        Weights.
    return_norm : bool
        If True, also return norm vector.

    Returns
    -------
    X_norm : array
        Normalized X.
    norm : array
        Norm.

    """
    if X.ndim == 3:
        n_samples, n_chans, n_trials = theshapeof(X)
        weights = _check_weights(weights, X)
        X = unfold(X)
        if not weights.any():  # no weights
            X_norm, N = normcol(X, return_norm=True)
            N = N ** 2
            X_norm = fold(X_norm, n_samples)

        else:  # weights
            if weights.shape[0] != n_samples:
                raise ValueError("Weight array should have same number of' \
                                 'columns as X")

            if weights.ndim == 2 and weights.shape[1] == 1:
                weights = np.tile(weights, (1, n_samples, n_trials))

            weights = unfold(weights)
            X_norm, N = normcol(X, weights, return_norm=True)
            N = N ** 2
            X_norm = fold(X_norm, n_samples)
            return X_norm

    else:
        n_samples, n_chans, n_trials = theshapeof(X)
        weights = _check_weights(weights, X)
        if not weights.any():
            with np.errstate(divide='ignore'):
                N = ((np.sum(X ** 2, axis=0) / n_samples) ** -0.5)[np.newaxis]

            N[np.isinf(N)] = 0
            N[np.isnan(N)] = 0
            X_norm = X * N

        else:
            if weights.shape[0] != X.shape[0]:
                raise ValueError('Weight array should have same number of ' +
                                 'columns as X')
            if weights.shape[1] == 1:
                weights = np.tile(weights, (1, n_chans))
            if weights.shape != X.shape:
                raise ValueError('Weight array should have be same shape as X')

            N = (np.sum(X ** 2 * weights, axis=0) /
                 np.sum(weights, axis=0)) ** -0.5
            N[np.isinf(N)] = 0
            N[np.isnan(N)] = 0
            X_norm = X * N

    if return_norm:
        return X_norm, np.sqrt(N)
    else:
        return X_norm


def _check_shifts(shifts, allow_floats=False):
    """Check shifts."""
    types = (int, np.int_)
    if allow_floats:
        types += (float, np.float_)
    if not isinstance(shifts, (np.ndarray, list, type(None)) + types):
        raise AttributeError('shifts should be a list, an array or an int')
    if isinstance(shifts, (list, ) + types):
        shifts = np.array(shifts).flatten()
    if shifts is None or len(shifts) == 0:
        shifts = np.array([0])

    n_shifts = np.size(shifts)

    return shifts, n_shifts


def _check_data(X):
    """Check data is numpy array and has the proper dimensions."""
    if not isinstance(X, (np.ndarray, list)):
        raise AttributeError('data should be a list or a numpy array')

    dtype = np.complex128 if np.any(np.iscomplex(X)) else np.float64
    X = np.asanyarray(X, dtype=dtype)
    if X.ndim > 3:
        raise ValueError('Data must be 3D at most')

    return X


def _check_weights(weights, X):
    """Check weights dimensions against X."""
    if not isinstance(weights, (np.ndarray, list)):
        if weights is not None:
            warnings.warn('weights should be a list or a numpy array.')
        weights = np.array([])

    weights = np.asanyarray(weights)
    if weights.size > 0:
        dtype = np.complex128 if np.any(np.iscomplex(weights)) else np.float64
        weights = np.asanyarray(weights, dtype=dtype)
        if weights.ndim > 3:
            raise ValueError('Weights must be 3D at most')
        if weights.shape[0] != X.shape[0]:
            raise ValueError("Weights should be the same n_times as X.")

        if X.ndim == 2 and weights.ndim == 1:
            weights = weights[:, np.newaxis]
        if X.ndim == 3:
            if weights.ndim == 2:
                weights = weights[:, np.newaxis, :]
            elif weights.ndim == 1:
                weights = weights[:, np.newaxis, np.newaxis]

            if weights.shape[-1] != X.shape[-1]:
                weights = np.tile(weights, (1, 1, X.shape[-1]))

        if weights.ndim > 1:
            if weights.shape[1] > 1 and weights.shape[1] != X.shape[1]:
                raise ValueError("Weights array should have a single column.")

        if np.any(np.abs(weights) > 1.):
            warnings.warn('weights should be between 0 and 1.')
            weights[np.abs(weights) > 1.] = 1.

    return weights


def _times_to_delays(lags, sfreq):
    """Convert a lags in seconds to delays."""
    if lags is None:
        return np.array([0])
    if not isinstance(sfreq, (int, float, np.int_)):
        raise ValueError('`sfreq` must be an integer or float')
    sfreq = float(sfreq)

    if not all([isinstance(ii, (int, float, np.int_)) for ii in lags]):
        raise ValueError('lags must be an integer or float')

    if len(lags) == 2 and sfreq != 1:
        tmin = lags[0]
        tmax = lags[1]

        if not tmin <= tmax:
            raise ValueError('tmin must be <= tmax')

        # Convert seconds to samples
        delays = np.arange(int(np.round(tmin * sfreq)),
                           int(np.round(tmax * sfreq) + 1), step=2)

    else:
        delays = lags

    return delays
