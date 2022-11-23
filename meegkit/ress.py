"""Rhythmic Entrainment Source Separation."""
import functools
import warnings

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin

from meegkit.utils import gaussfilt


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
    compute_unmixing : bool
        If True, also computing unmixing matrices (from_ress), used to
        transform the data back into sensor space.

    Examples
    --------
    To project the RESS components back into sensor space, one can proceed as
    follows:
    >>> # First create RESS estimator and fit_transform the data
    >>> r = ress.RESS(sfreq, peak_freq, compute_unmixing=True)
    >>> out = r.fit_transform(data)
    >>> # Then matrix multiply each trial by the unmixing matrix:
    >>> fromRESS = r.from_ress
    >>> proj = matmul3d(out, fromRESS)

    To transform a new observation into RESS component space (e.g. in the
    context of a cross-validation, with separate train/test sets) use the
    `transform` method:
    >>> new_comp = r.transform(newdata)

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
        compute_unmixing: bool = False
    ):

        self.sfreq = sfreq
        self.peak_freq = peak_freq
        self.neig_freq = neig_freq
        self.peak_width = peak_width
        self.neig_width = neig_width
        self.n_keep = n_keep
        self.gamma = gamma
        self.compute_unmixing = compute_unmixing

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

        # Back to Matlab formatting for gaussfilt
        X = X.transpose([2,1,0])

        # Covariance of  neighboor frequencies (noise)
        c01 = self._avg_cov(
            gaussfilt(
                X,
                self.sfreq,
                self.peak_freq + self.neig_freq,
                fwhm=self.neig_width,
                n_harm=1,
            )
        )
        c02 = self._avg_cov(
            gaussfilt(
                X,
                self.sfreq,
                self.peak_freq - self.neig_freq,
                fwhm=self.neig_width,
                n_harm=1,
            )
        )
        # Covariance of the signal
        c1 = self._avg_cov(
            gaussfilt(
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

        if self.compute_unmixing:
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

