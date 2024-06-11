"""Buffer class for real-time processing."""
import warnings

import numpy as np


class Buffer:
    """Circular buffer for real-time processing.

    Parameters
    ----------
    size : int
        The number of samples of the buffer.
    n_channels : int
        The number of channels of the buffer.

    Attributes
    ----------
    size : int
        The number of samples of the buffer.
    n_channels : int
        The number of channels of the buffer.
    counter : int
        The number of samples in the buffer.
    _data : ndarray, shape (size, n_channels)
        Data buffer.

    """

    def __init__(self, size, n_channels=1):
        self.size = size
        self.n_channels = n_channels
        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.counter = 0  # number of samples in the buffer
        self._head = 0  # most recent sample in the buffer
        self._tail = 0  # most recent read sample in the buffer
        self._data = np.zeros((self.size, self.n_channels))

    def push(self, X):
        """Update the buffer with new data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels)
            The input signal to be transformed.
        """
        # Check input
        if X.ndim == 0:
            # X = np.array([X])
            n_samples = 1
        elif X.ndim == 1:
            X = X[:, None]
            n_samples = X.shape[0]
        elif X.ndim > 2:
            raise ValueError("X must be 1- or 2-dimensional")

        # Check size
        if n_samples > self.size:
            warnings.warn("Buffer overflow: some old data has been lost")
            X = X[-self.size:]  # keep only the last self.size samples
            n_samples = X.shape[0]

        # Warn when data has not been read for a long time
        if self.counter + n_samples - self._tail > self.size:
            warnings.warn("Buffer overflow: some old data has been discarded")
            self._tail = self._head + n_samples - self.size

        # Update buffer
        end = self.head + n_samples
        if end <= self.size:
            # If the new data fits in the remaining space
            self._data[self.head:end] = X
        else:
            # If the new data wraps around to the beginning of the buffer
            overflow = end - self.size
            self._data[self.head:] = X[:-overflow]
            self._data[:overflow] = X[-overflow:]

        self._head += n_samples
        self.counter += n_samples

        return self

    def get_new_samples(self, n_samples=None):
        """Consume n_samples."""
        if n_samples is None:
            n_samples = min(self.counter, self.size)
        elif n_samples > self.counter:
            raise ValueError("Not enough samples in the buffer")
        elif n_samples > self.size:
            raise ValueError("Requested more samples than buffer size")

        start = self.tail
        end = (self.tail + n_samples) % self.size

        if start < end:
            samples = self._data[start:end]
        else:
            samples = np.concatenate((self._data[start:], self._data[:end]))

        self._tail += n_samples
        return samples

    def view(self, n_samples):
        """View the last n_samples, without consuming them."""
        start = (self.head - n_samples) % self.size
        end = self.head
        if start < end:
            return self._data[start:end]
        else:
            return np.concatenate((self._data[start:], self._data[:end]))

    def __repr__(self) -> str:
        """Return complete string representation."""
        repr = f"Buffer({self.size}, {self.n_channels})\n"
        repr += f"> counter: {self.counter}\n"
        repr += f"> head: {self.head}\n"
        repr += f"> tail: {self.tail}\n"
        return repr

    @property
    def tail(self):
        """Get the index of the tail."""
        return self._tail % self.size

    @property
    def head(self):
        """Get the index of the head."""
        return self._head % self.size
