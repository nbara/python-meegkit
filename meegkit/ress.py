"""Rhythmic Entrainment Source Separation."""
import functools
import warnings

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance


class RESS(TransformerMixin, BaseEstimator):
    """Rhythmic Entrainment Source Separation.

    As described in [1]_.

    Parameters
    ----------
    sfreq : int
        Sampling frequency.
    peak_freq : float
        Peak frequency.
    neig_freq : float
        Distance of neighbouring frequencies away from peak frequency, +/- in
        Hz (default=1).
    peak_width : float
        FWHM of the peak frequency (default=.5).
    neig_width : float
        FWHM of the neighboring frequencies (default=1).
    n_keep : int
        Number of components to keep (default=1). -1 keeps all components.
    gamma : float
        Regularization coefficient, between 0 and 1 (default=0.01, which
        corresponds to 1 % regularization and helps reduce numerical problems
        for noisy or reduced-rank matrices [2]_).

    References
    ----------
    .. [1] Cohen, M. X., & Gulbinaite, R. (2017). Rhythmic entrainment source
       separation: Optimizing analyses of neural responses to rhythmic sensory
       stimulation. Neuroimage, 147, 43-56.
    .. [2] Cohen, M. X. (2021). A tutorial on generalized eigendecomposition
       for source separation in multichannel electrophysiology.
       ArXiv:2104.12356 [Eess, q-Bio].
    """

    def __init__(
        self,
        sfreq: int,
        peak_freq: float,
        neig_freq: float = 1,
        peak_width: float = 0.5,
        neig_width: float = 1,
        n_keep: int = 1,
        gamma: float = 0.01,
    ):

        self.sfreq = sfreq
        self.peak_freq = peak_freq
        self.neig_freq = neig_freq
        self.peak_width = peak_width
        self.neig_width = neig_width
        self.n_keep = n_keep
        self.gamma = gamma

    def fit(self, X, y=None):
        """Learn a RESS filter.

        X : np.array (n_trials, n_chans, n_samples)
            Follow MNE format
        y : (ignored)
            Ignored parameter.

        Returns
        -------
        self : object
            RESS class instance.
        """
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
            warnings.warn("Fitting the RESS on only one sample !")

        _, n_chans, _ = X.shape

        # Compute mean along epoch + trial and remove it
        X -= np.mean(X, axis=(0, 2), keepdims=True)

        if self.n_keep == -1:
            self.n_keep = n_chans

        # Covariance of  neighboor frequencies (noise)
        c01 = self._avg_cov(
            self._gaussfilt(
                X,
                self.sfreq,
                self.peak_freq + self.neig_freq,
                fwhm=self.neig_width,
                n_harm=1,
            )
        )
        c02 = self._avg_cov(
            self._gaussfilt(
                X,
                self.sfreq,
                self.peak_freq - self.neig_freq,
                fwhm=self.neig_width,
                n_harm=1,
            )
        )
        # Covariance of the signal
        c1 = self._avg_cov(
            self._gaussfilt(
                X, self.sfreq, self.peak_freq, fwhm=self.peak_width, n_harm=1
            )
        )

        # add 1% regularization to avoid numerical precision problems in the GED
        c0 = (c01 + c02) / 2
        c0 = c0 * (1 - self.gamma) + self.gamma * np.trace(c0) / len(c0) * np.eye(
            len(c0)
        )

        # Perform generalized eigendecomposition
        # It solve a Generalized Eigenvalue Problem:
        # find a vector `to_ress` that maximize the ratio
        # c0^{-1} c1 (max c1 and min c0)
        d, to_ress = linalg.eigh(c1, c0)
        # Keep only the real part as c1 and c0
        # are symetric (then PSD) the imaginary
        # part is only a numerical error
        d = d.real
        to_ress = to_ress.real

        # Sort eigenvectors by decreasing eigenvalues
        # We are looking for the eigenvectors associated
        # with the larger eigenvalue
        idx = np.argsort(d)[::-1]
        d = d[idx]
        to_ress = to_ress[:, idx]

        # Normalize components (yields mixing matrix)
        to_ress /= np.sqrt(np.sum(to_ress, axis=0) ** 2)
        self.to_ress = to_ress[:, np.arange(self.n_keep)]

        # Compute unmixing matrix
        A = np.matmul(c1, to_ress)
        B = np.matmul(to_ress.T, np.matmul(c1, to_ress))
        try:
            # Note: we must use overwrite_a=False in order to be able to
            # use the fall-back solution below in case a LinAlgError is raised
            from_ress = linalg.solve(B.T, A.T, overwrite_a=False)
        except linalg.LinAlgError:
            # A or B is not full rank so exact solution is not tractable.
            #  Using least-squares solution instead.
            from_ress = linalg.lstsq(B.T, A.T, lapack_driver="gelsy")[0]
        self.from_ress = from_ress[: self.n_keep, :]

        return self

    def transform(self, X):
        """Project data using the learned RESS filter."""
        if X.ndim == 2:
            out = np.matmul(X, self.to_ress)
        else:
            out = np.zeros((X.shape[0], self.n_keep, X.shape[2]))
            for idx_trial in range(len(X)):
                out[idx_trial] = np.matmul(X[idx_trial].T, self.to_ress).T

        return out

    def fit_transform(self, X, y=None):
        """Compute RESS filter and apply it on data."""
        self.fit(X)

        return self.transform(X)

    def inverse_transform(self, X):
        """Backproject the RESS filtered data to the sensors space."""
        return np.matmul(X, self.from_ress)

    def _avg_cov(self, X):
        """Compute average covariance matrix along trials."""
        X_transpose = X.transpose([0, 2, 1])
        s = list(X_transpose.shape)
        combined = functools.reduce(lambda x, y: x * y, s[0: 0 + 1 + 1])
        X_combined = np.reshape(X_transpose, s[:0] + [combined] + s[0 + 1 + 1:])
        cov = EmpiricalCovariance().fit(X_combined)
        return cov.covariance_

    def _gaussfilt(self, data, srate, f, fwhm, n_harm=1, shift=0, return_empvals=False,
                   show=False):
        """FIR filter with the window method and a gaussian window.

        Parameters
        ----------
        data : ndarray
            EEG data, shape=(n_samples, n_channels[, ...])
        srate : int
            Sampling rate in Hz.
        f : float
            Break frequency of filter.
        fhwm : float
            Standard deviation of filter, defined as full-width at half-maximum
            in Hz.
        n_harm : int
            Number of harmonics of the frequency to consider.
        shift : int
            Amount shift peak frequency by (only useful when considering harmonics,
            otherwise leave to 0).
        return_empvals : bool
            Return empirical values (default: False).

        Returns
        -------
        filtdat : ndarray
            Filtered data.
        empVals : float
            The empirical frequency and FWHM.

        References
        ----------
        https://ccrma.stanford.edu/~jos/sasp/Window_Method_FIR_Filter.html

        """
        # input check
        assert data.shape[1] <= data.shape[2], "n_channels must be less than n_samples"
        assert (f - fwhm) >= 0, "increase frequency or decrease FWHM"
        assert fwhm >= 0, "FWHM must be greater than 0"

        # frequencies
        hz = np.fft.fftfreq(data.shape[2], 1.0 / srate)
        empVals = np.zeros((2,))

        # compute empirical frequency and standard deviation
        idx_p = np.searchsorted(hz[hz >= 0], f, "left")

        # create Gaussian
        fx = np.zeros_like(hz)
        for i_harm in range(1, n_harm + 1):  # make one gaussian per harmonic
            s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # normalized width
            x = hz.copy()
            x -= f * i_harm - shift
            gauss = np.exp(-0.5 * (x / s) ** 2)  # gaussian
            gauss = gauss / np.max(gauss)  # gain-normalized
            fx = fx + gauss

        # filter
        if data.ndim == 2:
            filtdat = 2 * np.real(
                np.fft.ifft(np.fft.fft(data, axis=2) * fx[None, :], axis=2)
            )
        elif data.ndim == 3:
            filtdat = 2 * np.real(
                np.fft.ifft(np.fft.fft(data, axis=2) * fx[None, None, :], axis=2)
            )

        if return_empvals:
            empVals[0] = hz[idx_p]
            # find values closest to .5 after MINUS before the peak
            empVals[1] = (
                hz[idx_p - 1 + np.searchsorted(fx[:idx_p], 0.5)]
                - hz[np.searchsorted(fx[:idx_p + 1], 0.5)]
            )

        if return_empvals:
            return filtdat, empVals
        else:
            return filtdat
