"""Signal coherence tools.

Compute 2D, 1D, and 0D bicoherence, polycoherence, bispectrum, and polyspectrum.

Bicoherence is a measure of the degree of phase coupling between different
frequency components in a signal. It's essentially a normalized form of the
bispectrum, which itself is a Fourier transform of the third-order cumulant of
a time series:
- 2D bicoherence is the most common form, where one looks at a
  two-dimensional representation of the phase coupling between different
  frequencies. It's a function of two frequency variables.
- 1D bicoherence would imply a slice or a specific condition in the 2D
  bicoherence, reducing it to a function of a single frequency variable. It
  simplifies the analysis by looking at the relationship between a particular
  frequency and its harmonics or other relationships
- 0D bicoherence would imply a single value representing some average or
  overall measure of phase coupling in the signal. It's a highly condensed
  summary, which might represent the average bicoherence over all frequency
  pairs, for instance.


"""
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.fftpack import next_fast_len
from scipy.signal import spectrogram


def cross_coherence(x1, x2, sfreq, norm=2, **kwargs):
    """Compute the bispectral cross-coherence between two signals of same length.

    Code adapted from [2]_.

    Parameters
    ----------
    x1: array-like, shape=([n_channels, ]n_samples)
        First signal.
    x2: array-like, shape=([n_channels, ]n_samples)
        Second signal.
    sfreq: float
        Sampling sfreq.
    norm: int | None
        Norm (default=2). If None, return bispectrum.

    Returns
    -------
    f1: array-like
        Frequency axis.
    B: array-like
        Bicoherence between s1 and s2.

    References
    ----------
    .. [1] http://wiki.fusenet.eu/wiki/Bicoherence
    .. [2] https://stackoverflow.com/a/36725871

    """
    N = x1.shape[-1]
    kwargs.setdefault("nperseg", N // 20)
    kwargs.setdefault("nfft", next_fast_len(N // 10))

    # compute the stft
    f1, _, S1 = spectrogram(x1, fs=sfreq, mode="complex", **kwargs)
    _, _, S2 = spectrogram(x2, fs=sfreq, mode="complex", **kwargs)

    # transpose (f, t) -> (t, f)
    S1 = np.swapaxes(S1, -1, -2)
    S2 = np.swapaxes(S2, -1, -2)

    # compute the bicoherence
    ind = np.arange(f1.size // 2)
    indsum = ind[:, None] + ind[None, :]

    P1 = S1[..., ind, None]
    P2 = S2[..., None, ind]
    P12 = S1[..., indsum]

    B = np.mean(P1 * P2 * np.conj(P12), axis=-3)

    if norm is not None: # Bispectrum -> Bicoherence
        B = norm_spectrum(B, P1, P2, P12, time_axis=-3)

    return f1[ind], B


def polycoherence_0d(X, sfreq, freqs, norm=2, synthetic=None, **kwargs):
    """Polycoherence between freqs and sum of freqs.

    Parameters
    ----------
    X: ndarray, shape=(n_channels, n_samples)
        Input data.
    sfreq: float
        Sampling rate.
    freqs: list[float]
        Fixed frequencies.
    norm: int | None
        Norm (default=2).
    synthetic: tuple(float, float, float)
        Used for synthetic signal for some frequencies (freq, amplitude,
        phase), freq must coincide with the first fixed frequency.
    **kwargs: dict
        Additional parameters passed to scipy.signal.spectrogram. Important
        parameters are nperseg, noverlap, nfft.

    Returns
    -------
    B: ndarray, shape=(n_channels,)
        Polycoherence

    """
    assert isinstance(freqs, Iterable), "freqs must be a list"
    N = X.shape[-1]
    kwargs.setdefault("nperseg", N // 20)
    kwargs.setdefault("nfft", next_fast_len(N // 10))

    freq, t, spec = spectrogram(X, fs=sfreq, mode="complex", **kwargs)

    ind = _freq_ind(freq, freqs)
    indsum = _freq_ind(freq, np.sum(freqs))
    spec = np.swapaxes(spec, -1, -2)

    Pi = _product_other_freqs(spec, ind, synthetic, t)
    Psum = spec[..., indsum]

    B = np.mean(Pi * np.conj(Psum), axis=-1)

    if norm is not None:
        # Bispectrum -> Bicoherence
        B = norm_spectrum(B, Pi, 1., Psum, time_axis=-1)

    return B


def polycoherence_1d(X, sfreq, f2, norm=2, synthetic=None, **kwargs):
    """1D polycoherence as a function of f1 and at least one fixed frequency f2.

    Parameters
    ----------
    X: ndarray
        Input data.
    sfreq: float
        Sampling rate
    f2: list[float]
        Fixed frequencies.
    norm: int | None
        Norm (default=2).
    synthetic: tuple(float, float, float)
        Used for synthetic signal for some frequencies (freq, amplitude,
        phase), freq must coincide with the first fixed frequency.
    **kwargs:
        Additional parameters passed to scipy.signal.spectrogram. Important
        parameters are `nperseg`, `noverlap`, `nfft`.

    Returns
    -------
    freq: ndarray, shape=(n_freqs_f1,)
        Frequencies
    B: ndarray, shape=(n_channels, n_freqs_f1)
        1D polycoherence.

    """
    assert isinstance(f2, Iterable), "f2 must be a list"

    N = X.shape[-1]
    kwargs.setdefault("nperseg", N // 20)
    kwargs.setdefault("nfft", next_fast_len(N // 10))

    f1, t, S = spectrogram(X, fs=sfreq, mode="complex", **kwargs)
    S = np.swapaxes(S, -1, -2)  # transpose (f, t) -> (t, f)

    ind2 = _freq_ind(f1, f2)
    ind1 = np.arange(len(f1) - sum(ind2))
    indsum = ind1 + sum(ind2)

    P1 = S[..., ind1]
    Pother = _product_other_freqs(S, ind2, synthetic, t)[..., None]
    Psum = S[..., indsum]

    B = np.mean(P1 * Pother * np.conj(Psum), axis=-2)

    if norm is not None:
        B = norm_spectrum(B, P1, Pother, Psum, time_axis=-2)

    return f1[ind1], B


def polycoherence_1d_sum(X, sfreq, fsum, *ofreqs, norm=2, synthetic=None, **kwargs):
    """1D polycoherence with fixed frequency sum fsum as a function of f1.

    Returns polycoherence for frequencies ranging from 0 up to fsum.

    Parameters
    ----------
    X: ndarray
        Input data.
    sfreq: float
        Sampling rate.
    fsum : float
        Fixed frequency sum.
    ofreqs: list[float]
        Fixed frequencies.
    norm: int or None
        If 2 - return polycoherence, n0 = n1 = n2 = 2 (default)
    synthetic: tuple(float, float, float) | None
        Used for synthetic signal for some frequencies (freq, amplitude,
        phase), freq must coincide with the first fixed frequency.

    Returns
    -------
    freq: ndarray, shape=(n_freqs,)
        Frequencies.
    B: ndarray, shape=(n_channels, n_freqs)
        Polycoherence for f1+f2=fsum.

    """
    N = X.shape[-1]
    kwargs.setdefault("nperseg", N // 20)
    kwargs.setdefault("nfft", next_fast_len(N // 10))

    freq, t, S = spectrogram(X, fs=sfreq, mode="complex", **kwargs)
    S = np.swapaxes(S, -1, -2)  # transpose (f, t) -> (t, f)

    indsum = _freq_ind(freq, fsum)
    ind1 = np.arange(np.searchsorted(freq, fsum - np.sum(ofreqs)))
    ind3 = _freq_ind(freq, ofreqs)
    ind2 = indsum - ind1 - sum(ind3)

    P1 = S[..., ind1]
    P2 = S[..., ind2]
    Pother = _product_other_freqs(S, ind3, synthetic, t)[..., None]
    Psum = S[..., [indsum]]

    B = np.mean(P1 * P2 * Pother * np.conj(Psum), axis=-2)

    if norm is not None:
        B = norm_spectrum(B, P1, P2 * Pother, Psum, time_axis=-2)

    return freq[ind1], B


def polycoherence_2d(X, sfreq, ofreqs=None, norm=2, flim1=None, flim2=None,
                     synthetic=None, **kwargs):
    """2D polycoherence between freqs and their sum as a function of f1 and f2.

    2D bicoherence is the most common form, where one looks at a
    two-dimensional representation of the phase coupling between different
    frequencies. It is a function of two frequency variables.

    2D polycoherence as a function of f1 and f2, ofreqs are additional fixed
    frequencies.

    Parameters
    ----------
    X: ndarray
        Input data.
    sfreq: float
        Sampling rate.
    ofreqs: list[float]
        Fixed frequencies.
    norm: int or None
        If 2 - return polycoherence (default), else return polyspectrum.
    flim1: tuple | None
        Frequency limits for f1. If None, it is set to (0, nyquist / 2)
    flim2: tuple | None
        Frequency limits for f2.

    Returns
    -------
    freq1: ndarray, shape=(n_freqs_f1,)
        Frequencies for f1.
    freq2: ndarray, shape=(n_freqs_f2,)
        Frequencies for f2.
    B: ndarray, shape=([n_chans, ]n_freqs_f1, n_freqs_f2)
        Polycoherence.

    """
    N = X.shape[-1]
    kwargs.setdefault("nperseg", N // 20)
    kwargs.setdefault("nfft", next_fast_len(N // 10))

    freq, t, S = spectrogram(X, fs=sfreq, mode="complex", **kwargs)
    freq = freq[:len(freq) // 2 + 1]  # only positive frequencies
    S = S[:len(freq) // 2 + 1]  # only positive frequencies
    S = np.require(S, "complex64")

    S = np.swapaxes(S, -1, -2)  # transpose (f, t) -> (t, f)

    if ofreqs is None:
        ofreqs = []
    if flim1 is None:
        flim1 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)
    if flim2 is None:
        flim2 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)

    # indices ranges matching flim1 and flim2
    ind1 = np.arange(*np.searchsorted(freq, flim1))
    ind2 = np.arange(*np.searchsorted(freq, flim2))
    ind3 = _freq_ind(freq, ofreqs)
    indsum = ind1[:, None] + ind2[None, :] + sum(ind3)

    Pother = _product_other_freqs(S, ind3, synthetic, t)[..., None, None]

    P1 = S[..., ind1, None]
    P2 = S[..., None, ind2] * Pother
    P12 = S[..., indsum]

    # Average over time to get the bispectrum
    B = np.mean(P1 * P2 * np.conj(P12), axis=-3)

    if norm is not None: # Bispectrum -> Bicoherence
        B = norm_spectrum(B, P1, P2, P12, time_axis=-3)

    return freq[ind1], freq[ind2], B


def norm_spectrum(spec, P1, P2, P12, time_axis=-2):
    """Compute bicoherence from bispectrum.

    For formula see [1]_.

    Parameters
    ----------
    spec: ndarray, shape=(n_chans, n_freqs)
        Polyspectrum.
    P1: array-like, shape=(n_chans, n_times, n_freqs_f1)
        Spectrum evaluated at f1.
    P2: array-like, shape=(n_chans, n_times, n_freqs_f2)
        Spectrum evaluated at f2.
    P12: array-like, shape=(n_chans, n_times, n_freqs)
        Spectrum evaluated at f1 + f2.
    time_axis: int
        Time axis.

    Returns
    -------
    coh: ndarray, shape=(n_chans,)
        Polycoherence.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bicoherence

    """
    coh = np.abs(spec) ** 2
    norm = np.mean(np.abs(P1 * P2) ** 2, axis=time_axis)
    norm *= np.mean(np.abs(np.conj(P12)) ** 2, axis=time_axis)
    coh /= norm
    coh **= 0.5
    return coh


def plot_polycoherence(freq1, freq2, bicoh, ax=None):
    """Plot polycoherence."""
    df1 = freq1[1] - freq1[0]  # resolution
    df2 = freq2[1] - freq2[0]
    freq1 = np.append(freq1, freq1[-1] + df1) - 0.5 * df1
    freq2 = np.append(freq2, freq2[-1] + df2) - 0.5 * df2

    if ax is None:
        f, ax = plt.subplots()

    ax.pcolormesh(freq2, freq1, np.abs(bicoh))
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("freq (Hz)")
    # ax.colorbar()
    return ax

def plot_polycoherence_1d(freq, coh):
    """Plot polycoherence for fixed frequencies."""
    plt.figure()
    plt.plot(freq, coh)
    plt.xlabel("freq (Hz)")


def plot_signal(t, signal, ax=None):
    """Plot signal and spectrum."""
    if ax is None:
        f, ax = plt.subplots(2, 1)

    ax[0].plot(t, signal)
    ax[0].set_xlabel("time (s)")

    ndata = len(signal)
    nfft = next_fast_len(ndata)
    freq = rfftfreq(nfft, t[1] - t[0])
    spec = rfft(signal, nfft) * 2 / ndata
    ax[1].plot(freq, np.abs(spec))
    ax[1].set_xlabel("freq (Hz)")
    return ax

def _freq_ind(freq, f0):
    """Find the index of the frequency closest to f0."""
    if isinstance(f0, Iterable):
        return [np.argmin(np.abs(freq - f)) for f in f0]
    else:
        return np.argmin(np.abs(freq - f0))


def _product_other_freqs(spec, indices, synthetic=None, t=None):
    """Product of all frequencies."""
    if synthetic is None:
        synthetic = ()

    p1 = synthetic_signal(t, synthetic)
    p2 = np.prod(spec[..., indices[len(synthetic):]], axis=-1)
    return p1 * p2


def synthetic_signal(t, synthetic):
    """Create a synthetic complex signal spectrum.

    Parameters
    ----------
    t: array-like
        Time.
    synthetic: list[tuple(float, float, float)]
        List of tuples with (freq, amplitude, phase).

    Returns
    -------
    Complex signal.

    """
    return np.prod([amp * np.exp(2j * np.pi * freq * t + phase)
                    for (freq, amp, phase) in synthetic], axis=0)