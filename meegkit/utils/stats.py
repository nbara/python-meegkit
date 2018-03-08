"""Benchmark utility functions."""
from __future__ import division, print_function

import random

import numpy as np

try:
    import mne
except ImportError:
    mne = None


def rms(x, axis=0):
    """Root-mean-square along given axis."""
    return np.sqrt(np.mean(x ** 2, axis=axis, keepdims=True))


def robust_mean(x, axis=0, percentile=[5, 95]):
    """Do robust mean based on JR Kings implementation."""
    x = np.array(x)
    axis_ = axis
    # force axis to be 0 for facilitation
    if axis is not None and axis != 0:
        x = np.transpose(x, [axis] + range(0, axis) + range(axis + 1, x.ndim))
        axis_ = 0
    mM = np.percentile(x, percentile, axis=axis_)
    indices_min = np.where((x - mM[0][np.newaxis, ...]) < 0)
    indices_max = np.where((x - mM[1][np.newaxis, ...]) > 0)
    x[indices_min] = np.nan
    x[indices_max] = np.nan
    m = np.nanmean(x, axis=axis_)
    return m


def bootstrap_ci_chan(average, n_bootstrap=2000):
    """Get confidence intervals from non-parametric bootstrap resampling."""
    indices = np.arange(len(average.ch_names), dtype=int)
    gfps_bs = np.empty((n_bootstrap, len(average.times)))

    for i in range(n_bootstrap):
        bs_indices = np.random.choice(indices, replace=True, size=len(indices))
        gfps_bs[i] = np.sum(average.data[bs_indices] ** 2, 0)
    gfps_bs = mne.baseline.rescale(gfps_bs, average.times, baseline=(None, 0))
    ci_low, ci_up = np.percentile(gfps_bs, (2.5, 97.5), axis=0)
    return ci_low, ci_up


def bootstrap_ci(X, n_bootstrap=2000, ci=(5, 95), axis=0):
    """Confidence intervals from non-parametric bootstrap resampling.

    Bootstrap is computed *with* replacement.

    Parameters
    ----------
    X : 2D array, shape = (n_chans, n_times)
        Input data.
    n_bootstrap : int
        Number of bootstrap resamplings.
    ci : length-2 tuple
        Confidence interval.
    axis : int
        Axis to operate on.

    Returns
    -------
    ci_low, ci_up : confidence intervals

    """
    indices = np.arange(X.shape[axis], dtype=int)

    gfps_bs = np.nan * np.ones((n_bootstrap, X.shape[~axis]))
    for i in range(n_bootstrap):
        bs_indices = np.random.choice(indices, replace=True, size=len(indices))
        gfps_bs[i] = np.mean(np.take(X, bs_indices, axis=axis), axis=axis)

    ci_low, ci_up = np.percentile(gfps_bs, ci, axis=0)

    return ci_low, ci_up


def bootstrap_snr(epochs, n_bootstrap=2000, baseline=None, window=None):
    """Get bootstrap SNR from non-parametric bootstrap resampling.

    The lower bound SNR value can serve as a minimum threshold of signal
    quality. Parks et al. (2016) suggest using a threshold of 3 dB when
    assessing whether to remove subjects from ERP studies.

    Parameters
    ----------
    epochs : mne.Epochs instance
        Epochs instance to compute ERP from.
    n_bootstrap : int
        Number of bootstrap iterations (should be > 10000 for publication
        quality).
    baseline : tuple or list of length 2, or None
        The time interval to apply rescaling / baseline correction. If None do
        not apply it. If baseline is ``(bmin, bmax)`` the interval is between
        ``bmin`` (s) and ``bmax`` (s). If ``bmin is None`` the beginning of the
        data is used and if ``bmax is None`` then ``bmax`` is set to the end of
        the interval. If baseline is ``(None, None)`` the entire time interval
        is used. If baseline is None, no correction is applied.
    window : tuple or list of length 2, or None
        The time interval used to compute the numerator of the SNR. If window
        is ``(bmin, bmax)`` the interval is between ``bmin`` (s) and ``bmax``
        (s). If ``bmin is None`` the beginning of the data is used and if
        ``bmax is None`` then ``bmax`` is set to the end of the interval. If
        window is ``(None, None)`` the entire time interval is used. If None
        use all positive times.

    Returns
    -------
    ERP : length-3 tuple
        Mean bootstrap ERP, with 90% confidence intervals.
    SNR : length-3 tuple
        Mean bootstrap SNR, with 90% confidence intervals.

    References
    ----------
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4751267/

    """
    indices = np.arange(len(epochs.selection), dtype=int)
    erp_bs = np.empty((n_bootstrap, len(epochs.times)))
    gfp_bs = np.empty((n_bootstrap, len(epochs.times)))

    for i in range(n_bootstrap):
        if 100 * i / n_bootstrap == np.floor(100 * i / n_bootstrap):
            print('Bootstrapping... {}%'.format(round(100 * i / n_bootstrap)),
                  end='\r'),
        bs_indices = np.random.choice(indices, replace=True, size=len(indices))
        erp_bs[i] = np.mean(epochs._data[bs_indices, 0, :], 0)

        # Baseline correct mean waveform
        erp_bs[i] = mne.baseline.rescale(erp_bs[i], epochs.times,
                                         baseline=baseline, verbose='ERROR')

        # Rectify waveform
        gfp_bs[i] = np.sqrt(erp_bs[i] ** 2)

    # Bootstrap ERP confidence intervals
    ci_low, ci_up = np.percentile(erp_bs, (10, 90), axis=0)

    # Calculate SNR for each bootstrapped ERP; form distribution in `snr_dist`
    snr_dist = np.zeros((n_bootstrap,))
    if window is not None:
        if window[0] is None:
            window[0] = 0
        if window[1] is None:
            window[1] = np.inf
        post = np.logical_and(epochs.times >= window[0],
                              epochs.times <= window[1])
    else:
        post = epochs.times >= 0
    pre = epochs.times <= 0

    # SNR for each bootstrap iteration
    for i in range(n_bootstrap):
        snr_dist[i] = 20 * np.log10(
            np.mean(gfp_bs[i, post]) / np.mean(gfp_bs[i, pre]))

    print('Bootstrapping... OK!   ')

    # Mean, lower, and upper bound SNR
    snr_low, snr_up = np.percentile(snr_dist, (10, 90), axis=0)
    snr_mean = np.mean(snr_dist)
    mean_bs_erp = np.mean(erp_bs, axis=0)

    return (mean_bs_erp, ci_low, ci_up), (snr_mean, snr_low, snr_up)


def cronbach(epochs, K=None, n_bootstrap=2000, tmin=None, tmax=None):
    r"""Compute reliability of ERP.

    Internal reliability of the ERN and Pe as a function of increasing trial
    numbers can be quantified with Cronbach's alpha:

    $$\alpha = \frac{K}{K-1} \left(1-\frac{\sum_{i=1}^K
    \sigma^2_{Y_i}}{ \sigma^2_X}\right)$$

    Hinton, Brownlow, McMurray, and Cozens (2004) have suggested that
    Cronbach's alpha exceeding .90 indicates excellent internal reliability,
    between .70 and .90 indicates high internal reliability, from .50 to .70
    indicates moderate internal reliability, and below .50 is low.

    Parameters
    ----------
    epochs: instance of mne.Epochs
        Epochs to compute alpha from.
    K: int
        Number of trials to use for alpha computation.
    n_bootstrap: int
        Number of bootstrap resamplings.
    tmin: float
        Start time of epoch.
    tmax: float
        End of epoch.

    Returns
    -------
    alpha:
        Cronbach alpha value
    bounds: length-2 tuple
        Lower and higher bound of CI.

    """
    if tmin:
        tmin = epochs.time_as_index(tmin)[0]
    else:
        tmin = 0

    if tmax:
        tmax = epochs.time_as_index(tmax)[0]
    else:
        tmax = -1

    if K is None:
        K = epochs._data.shape[0]

    alpha = np.empty(n_bootstrap)
    for b in np.arange(n_bootstrap):  # take K trials randomly
        idx = random.sample(range(epochs._data.shape[0]), K)
        X = epochs._data[idx, 0, tmin:tmax]
        sigmaY = X.var(axis=1)  # var over time
        sigmaX = X.sum(axis=0).var()  # var of average
        alpha[b] = K / (K - 1) * (1 - sigmaY.sum() / sigmaX)

    ci_lb, ci_ub = np.percentile(alpha, (10, 90))
    return np.max([alpha.mean(), 0]), (ci_lb, ci_ub)


def snr_spectrum(data, f, n_avg=1, n_harm=1, skipbins=1):
    """Compute Signal-to-Noise-corrected spectrum.

    Parameters
    ----------
    data : ndarray , shape = ([n_trials, ]n_chans, n_samples)
        Power spectrum.
    n_avg : int
        Number of neighbour bins to estimate noise over. Make sure that this
        value doesn't overlap with neighbouring target frequencies
    n_harm : int
        Compute SNR at each frequency bin as a pooled RMS over this bin and
        n_harm harmonics (see references below)

    Returns
    -------
        SNR : ndarray, shape = (Nconds, Nchans, Nsamples) or (Nchans, Nsamples)
            Signal-to-Noise-corrected spectrum

    References
    ----------
    .. [1] Cottereau, B. R., McKee, S. P., Ales, J. M., & Norcia, A. M. (2011).
       Disparity-tuned population responses from human visual cortex. The
       Journal of Neuroscience, 31(3), 954-965.
    .. [2] Cottereau, B. R., McKee, S. P., & Norcia, A. M. (2014). Dynamics and
       cortical distribution of neural responses to 2D and 3D motion in human.
       Journal of neurophysiology, 111(3), 533-543.
    .. [3] de Heering, A., & Rossion, B. (2015). Rapid categorization of
       natural face images in the infant right hemisphere. Elife, 4.

    """
    if data.ndim == 3:
        n_trials = data.shape[0]
        n_channels = data.shape[1]
        n_samples = data.shape[-1]
    elif data.ndim == 2:
        n_trials = 1
        n_channels = data.shape[0]
        n_samples = data.shape[-1]
    else:
        raise ValueError('Data must have shape (n_trials, n_channels, n_times)'
                         ' or (n_channels, n_times)')

    # Number of points to get desired resolution
    data = np.reshape(data, (n_trials * n_channels, n_samples))
    SNR = np.zeros_like(data)

    for i_bin in range(n_samples):

        # Indices of bins to average over
        # ----------------------------------------------------------------------
        # bin_peaks = np.zeros(n_harm, dtype=int)
        bin_noise = []
        bin_peaks = []
        for h in range(n_harm):  # loop over harmonics

            # make sure freq <= fmax
            if f[i_bin] > 0 and f[i_bin] * (h + 1) <= f[-1]:

                # Get indices of harmonics
                bin_peaks.append(int(np.argmin(np.abs(f[i_bin] * (h + 1) -
                                                      f))))

                # Now get indices of noise (i.e., neighbouring FFT bins)
                # eg if currentbin=54, navg=3, skipbins=1 :
                # bin_noise = 51, 52, 56, 57
                tmp = np.stack((np.arange(bin_peaks[h] - skipbins - n_avg,
                                          bin_peaks[h] - skipbins),
                                np.arange(bin_peaks[h] + skipbins + 1,
                                          bin_peaks[h] + skipbins + 1 + n_avg)
                                ))
                tmp = tmp.flatten().astype(int)

                # Remove impossible bin values (eg <1 or >n_samp)
                tmp = [t for t in tmp if t >= 0 and t < n_samples]
                bin_noise.append(tmp)
                del tmp

        # SNR at central bin is ratio between (power at central
        # bin) to (average of N surrounding bins)
        # --------------------------------------------------------------------------
        for i_trial in range(n_trials * n_channels):

            # RMS of signal over fundamental+harmonics
            A = data[i_trial, bin_peaks]

            # Noise around fundamental+harmonics
            B = np.zeros(len(bin_noise))
            for h in range(len(B)):

                if n_trials > 1:
                    # Mean over samples and median over trials
                    B[h] = np.median(np.mean(data[i_trial::n_channels,
                                                  bin_noise[h]],
                                             1))
                else:
                    # Mean over samples
                    B[h] = np.mean(data[i_trial, bin_noise[h]])

            # Ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                SNR[i_trial, i_bin] = np.sqrt(np.sum(A)) / np.sqrt(np.sum(B))
                SNR[np.abs(SNR) == np.inf] = 1
                SNR[SNR == 0] = 1
                SNR[np.isnan(SNR)] = 1

            del A
            del B

    # Reshape matrix if necessary
    if np.min((n_trials, n_channels)) > 1:
        SNR = np.reshape(SNR, (n_trials, n_channels, n_samples))

    return SNR
