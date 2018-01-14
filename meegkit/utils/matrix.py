import numpy as np


def theshapeof(data):
    """Return the shape of data."""
    if not isinstance(data, np.ndarray):
        raise AttributeError('data must be a numpy array')

    if data.ndim == 3:
        return data.shape[0], data.shape[1], data.shape[2]
    elif data.ndim == 2:
        return data.shape[0], data.shape[1], 1
    elif data.ndim == 1:
        return data.shape[0], 1, 1
    else:
        raise ValueError("Array contains more than 3 dimensions")


def unsqueeze(data):
    """Add singleton dimensions to an array."""
    return data.reshape(theshapeof(data))


def fold(data, epochsize):
    """Fold 2D data into 3D."""
    n_chans = data.shape[0] // epochsize
    data = np.transpose(
        np.reshape(data, (epochsize, n_chans, data.shape[1]),
                   order="F").copy(), [0, 2, 1])

    return data


def unfold(data):
    """Unfold 3D data."""
    n_samples, n_chans, n_trials = theshapeof(data)

    if n_trials > 1:
        return np.reshape(
            np.transpose(data, (0, 2, 1)),
            (n_samples * n_trials, n_chans), order="F").copy()
    else:
        return data
