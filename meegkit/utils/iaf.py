"""Individual Alpha Frequency estimation method."""
from collections import namedtuple  # noqa: I100

import numpy as np
from numpy.polynomial import Polynomial
from scipy.fft import next_fast_len
from scipy.integrate import trapz

# from scipy.ndimage import center_of_mass
from scipy.signal import savgol_filter, welch

IafEst = namedtuple('IAFEstimate',
                    ['PeakAlphaFrequency', 'CenterOfGravity', 'AlphaBand'])


def restingIAF(data, cmin, frange, sfreq, w, Fw, order=10, mpow=1, mdiff=0.20,
               taper="boxcar", tlen=None, noverlap=None, nfft=None, norm=True,
               labels=None, show=True):
    """
    Primary function for running `restingIAF` analysis routine for estimating
    two indices of individual alpha frequency (IAF): Peak alpha frequency (PAF)
    and the alpha centre of gravity (CoG) or mean frequency.

    Implements method described in [1]_.

    Parameters
    ----------
    data : ndarray, shape=(n_channels, n_samples)
        Continuous EEG channel data.
    cmin : int
        Minimum number of channel estimates that must be resolved in
        order to calculate average PAF/CoG estimates.
    frange : ndarray
        Frequency range to be included in analysis (e.g., [1, 40] Hz).
    sfreq : int
        EEG sampling rate.
    w : ndarray
        Bounds of alpha peak search window (e.g., [7 13]).
    Fw : int
        Frame width, Savitzky-Golay filter (corresponds to number of
        freq. bins spanned by filter; must be odd).
    order : int
        Polynomial order, Savitzky-Golay filter (must be < Fw).
    mpow : float, optional
        Error bound (s.d.) used to determine threshold differentiating
        substantive peaks from background spectral noise (default=1).
    mdiff : float, optional
        Minimal height difference distinguishing a primary peak from
        competing peaks (default=0.20; i.e. 20% peak height).
    taper : str, optional
        Taper window function applied by `welch` (default='hamming').
    tlen : int, optional
        Length of taper window applied by `welch` (default=4 sec).
    noverlap : int, optional
        Length of taper window overlap in samples (default=50% window length).
    nfft : int, optional
        Specify number of FFT points used to calculate PSD (default=next power
        of 2 above window length).
    norm : bool, optional
        Normalize power spectra (default=True).
    show : bool, optional
        Whether to plot the PSD estimates (default=True).

    Returns
    -------
    pSum : dict
        Structure containing summary statistics of alpha-band parameters.
    p : dict
        Structure containing channel-wise spectral and alpha parameter data.
    f : ndarray
        Trimmed vector of frequency bins resolved by `welch`.

    References
    ----------
    .. [1] Corcoran, A. W., Alday, P. M., Schlesewsky, M., &
       Bornkessel-Schlesewsky, I. (2018). Toward a reliable, automated method
       of individual alpha frequency (IAF) quantification. Psychophysiology,
       e13064. doi:10.1111/psyp.13064
    """
    if tlen is None:
        tlen = sfreq * 8
    if noverlap is None:
        noverlap = tlen // 4
    if nfft is None:
        nfft = next_fast_len(tlen)

    n_chans = data.shape[-2]
    p = [dict() for _ in range(n_chans)]

    # calculate power spectral density estimates with Welch's method
    if not np.isfinite(data).all(axis=0).all():
        mask = np.isfinite(data).all(axis=0)
        print(f"Warning: {np.sum(~mask)/mask.size * 100:.1f}% of data is NaN. "
              "Removing...")
        data = data[:, mask]

    f, pxx = welch(data, sfreq, taper, nperseg=tlen, noverlap=noverlap, nfft=nfft,
                   axis=-1)

    # if pxx.ndim == 3 and labels is not None:
    #     pxx = pxx[labels == 0].mean(0) - pxx[labels == 1].mean(0)

    if norm:
        pxx /= np.nanmean(pxx, axis=-1, keepdims=True)

    # Only consider frequencies within the specified range
    frex = np.logical_and(f >= frange[0], f <= frange[1])
    f = f[frex]
    pxx = pxx[..., frex]


    if data.ndim == 3 and labels is not None:
        nogo = pxx[labels == 0].mean(0)
        go = pxx[labels == 1].mean(0)
        logx = np.log10(nogo) - np.log10(go)
        pxx = nogo / go
    else:
        logx = np.log10(pxx)


    for kx in range(n_chans):
        p[kx]['pxx'] = pxx[kx]

        poly = Polynomial.fit(f, logx[kx], 1)
        yval = poly(f)
        err = np.sqrt(np.mean((logx[kx] - yval) ** 2))
        p[kx]['minPow'] = yval + (mpow * err)  # takes [minPowThresh * Std dev] as upper error bound on background spectral noise

        fres = sfreq / nfft

        p[kx]['d0'] = savgol_filter(p[kx]['pxx'], Fw, order, deriv=0, delta=fres)
        p[kx]['d1'] = savgol_filter(p[kx]['pxx'], Fw, order, deriv=1, delta=fres)
        p[kx]['d2'] = savgol_filter(p[kx]['pxx'], Fw, order, deriv=2, delta=fres)

        # Placeholder for peakBounds function
        (p[kx]['peaks'],
         p[kx]['pos1'], p[kx]['pos2'],
         p[kx]['f1'], p[kx]['f2'],
         p[kx]['inf1'], p[kx]['inf2'],
         p[kx]['Q'], p[kx]['Qf']) = peakBounds(
            p[kx]['d0'],
            p[kx]['d1'],
            p[kx]['d2'],
            f,
            w,
            p[kx]['minPow'],
            mdiff, fres)

    # estimate gravities for smoothed spectra (average IAF window across channels)
    gravs, selG, iaw = chan_gravs(np.array([pc['d0'] for pc in p]),
                                  f,
                                  np.array([pc['f1'] for pc in p]),
                                  np.array([pc['f2'] for pc in p]))

    # calculate average pt estimates/spectra across k-th channels for each j-th recording
    selP, pSum = chan_means(gravs,
                            selG,
                            np.array([pc['peaks'] for pc in p]),
                            np.array([pc['d0'] for pc in p]),
                            np.array([pc['Qf'] for pc in p]),
                            cmin)

    # retain gravity estimates and selected channels in channel data struct
    # (only loop through trimmed channels)
    for cx in range(n_chans):
        p[cx]['gravs'] = gravs[cx]
        p[cx]['selP'] = selP[cx]
        p[cx]['selG'] = selG[cx]

    # get total number of chans that contributed to PAF/CoG estimation
    pSum['pSel'] = sum(selP)
    pSum['gSel'] = sum(selG)
    pSum['iaw'] = iaw

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(n_chans//2, 2, sharex=True, sharey=True, figsize=(8, 8))
        for i, a in enumerate(ax.flat):
            a.plot(f, p[i]["pxx"], label="Raw PSD", lw=.75)
            a.plot(f, p[i]["d0"], label="Smoothed PSD", lw=3, alpha=0.7)
            a.grid(True, ls=":")
            if i % 2 == 0:
                a.set_ylabel("Power (norm.)")
            if i >= n_chans - 2:
                a.set_xlabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()


    return pSum, p, f


def chan_means(chan_cogs, sel_g, peaks, specs, qf, cmin):
    """Compute weighted mean over channels.

    Takes channel-wise estimates of peak alpha frequency (PAF) / centre of
    gravity (CoG) and calculates mean and standard deviation if cmin
    condition satisfied. PAFs are weighted in accordance with qf, which aims
    to quantify the relative strength of each channel peak.

    Parameters
    ----------
    chan_cogs : ndarray
        Vector of channel-wise CoG estimates.
    sel_g : ndarray
        Vector (logical) of channels selected for individual alpha band estimation.
    peaks : ndarray
        Vector of channel-wise PAF estimates.
    specs : ndarray
        Matrix of smoothed spectra (helpful for plotting weighted estimates).
    qf : ndarray
        Vector of peak quality estimates (area bounded by inflection points).
    cmin : int
        Min number of channels required to generate cross-channel mean.

    Returns
    -------
    sel_p : ndarray
        Channels contributing peak estimates to calculation of mean PAF.
    sums : dict
        Structure containing summary estimates (m, std) for PAF and CoG.
    """

    # channel selection and weights
    sel_p = ~np.isnan(peaks)  # evaluate whether channel provides estimate of PAF

    # channel weightings scaled in proportion to Qf value of channel
    # manifesting highest Qf
    chan_wts = qf / np.nanmax(qf)

    sums = {}

    # average across peaks
    if np.sum(sel_p) < cmin:
        # if number of viable channels < cmin threshold, don't calculate
        # cross-channel mean & std
        sums['paf'] = np.nan
        sums['paf_std'] = np.nan
        sums['mu_spec'] = np.nan
    else:
        # else compute (weighted) cross-channel average PAF estimate and
        # corresponding std of channel PAFs
        sums['paf'] = np.nansum(peaks * chan_wts) / np.nansum(chan_wts)
        sums['paf_std'] = np.nanstd(peaks)
        # estimate averaged spectra for plotting
        wt_spec = specs * chan_wts[:, np.newaxis]
        sums['mu_spec'] = np.nansum(wt_spec, axis=1) / np.nansum(chan_wts)

    # now for the gravs (no channel weighting, all channels included if cmin satisfied)
    if np.sum(sel_g) < cmin:
        sums['cog'] = np.nan
        sums['cog_std'] = np.nan
    else:
        sums['cog'] = np.nanmean(chan_cogs)
        sums['cog_std'] = np.nanstd(chan_cogs)

    return sel_p, sums


def chan_gravs(d0, f, f1, f2):
    """Compute centre of gravity (CoG) across channels.

    Takes smoothed channel spectra and associated estimates of individual
    alpha bandwidth [f1:f2], calculate mean bandwidth, estimate CoG across
    all channels (as per Klimesch's group; e.g, 1990, 1993, & 1997 papers).

    Parameters
    ----------
    d0 : ndarray
        Vector / matrix of smoothed PSDs.
    f : ndarray
        Frequency bin vector.
    f1 : ndarray
        Vector of f1 bin indices (lower bound, individual alpha window).
    f2 : ndarray
        Vector of f2 bin indices (upper bound, individual alpha window).

    Returns
    -------
    cogs : ndarray, shape=(n_channels,)
        Centre of gravity derived from averaged f1:f2 frequency window.
    sel : ndarray
        Channels contributing estimates of alpha window bandwidth.
    iaw : ndarray
        Bounds of individual alpha window.
    """

    # trim off any NaNs in f1/f2 vectors
    trim_f1 = f1[~np.isnan(f1)].astype(int)
    trim_f2 = f2[~np.isnan(f2)].astype(int)

    # derive average f1 & f2 values across chans, then look for nearest freq bin
    mean_f1 = np.argmin(np.abs(f - np.mean(f[trim_f1])))
    mean_f2 = np.argmin(np.abs(f - np.mean(f[trim_f2])))
    iaw = [f[mean_f1], f[mean_f2]]

    # calculate CoG for each channel spectra on basis of averaged alpha window
    if len(trim_f1) == 0 or len(trim_f2) == 0:
        cogs = np.full((d0.shape[0]), np.nan)
    else:
        cogs = np.nansum(d0[:, mean_f1:mean_f2] * f[mean_f1:mean_f2], axis=1)
        cogs /= np.nansum(d0[:, mean_f1:mean_f2], axis=1)

    # report which channels contribute to averaged window
    sel = ~np.isnan(f1)

    return cogs, sel, iaw


def peakBounds(d0, d1, d2, f, w, minPow, minDiff, fres):
    """
    Take derivatives from Savitzky-Golay curve-fitting and differentiation
    function sgfDiff, pump out estimates of alpha-band peak & bounds.

    Also calculates primary peak area Qf via integration between inflections.

    Parameters
    ----------
    d0 : ndarray
        Smoothed PSD estimate vector.
    d1 : ndarray
        1st derivative vector.
    d2 : ndarray
        2nd derivative vector.
    f : ndarray
        Frequency bin vector.
    w : ndarray
        Bounds of initial alpha window.
    minPow : ndarray
        Vector of minimum power threshold values defining candidate peaks.
    minDiff : float
        Minimum difference required to distinguish peak as dominant (proportion
        of primary peak height).
    fres : float
        Frequency resolution (determine how many bins to search to establish
        shallow rolloff in d1).

    Returns
    -------
    peakF : float
        Peak frequency estimate.
    posZ1 : float
        Frequency of 1st positive zero-crossing (lower bound alpha interval).
    posZ2 : float
        Frequency of 2nd positive zero-crossing (upper bound alpha interval).
    f1 : float
        Frequency bin for posZ1.
    f2 : float
        Frequency bin for posZ2.
    inf1 : float
        Inflection point, ascending edge.
    inf2 : float
        Inflection point, descending edge.
    Q : float
        Area under peak between inf1 & inf2.
    Qf : float
        Q divided by bandwidth of Q.

    """

    # evaluate derivative for zero-crossings
    lower_alpha = np.argmin(np.abs(f - w[0]))  # set lower bound for alpha band
    upper_alpha = np.argmin(np.abs(f - w[1]))  # set upper bound for alpha band

    negZ = []  # initialise for zero-crossing count & frequency bin
    cnt = 0  # start counter at 0
    for k in range(lower_alpha - 1, upper_alpha + 2):  # step through frequency bins in alpha band
        if np.sign(d1[k]) > np.sign(d1[k + 1]):  # look for switch from positive to negative derivative values
            maxim = k if d0[k] > d0[k + 1] else k + 1  # find larger of two values either side of crossing
            cnt += 1  # advance counter by 1
            negZ.append(
                [cnt,  # zero-crossing count
                 maxim,  # keep bin index for later
                 f[maxim],  # zero-crossing frequency
                 d0[maxim],  # power estimate
            ])

    negZ = np.array(negZ)

    # sort out appropriate estimates for output
    if len(negZ) == 0:  # if no zero-crossing detected
        peakF = np.nan
        subBin = np.nan
    elif negZ.shape[0] == 1:  # if singular crossing
        bin = int(negZ[0, 1])
        if np.log10(negZ[0, 3]) > minPow[bin]:  # and peak power is > minimum threshold
            peakBin = int(negZ[0, 1])
            peakF = negZ[0, 2]
        else:
            peakF = np.nan
            subBin = np.nan
    else:  # if >1 crossing
        negZ = negZ[negZ[:, 3].argsort()][::-1]  # re-sort from largest to smallest peak
        bin = int(negZ[0, 1])
        if np.log10(negZ[0, 3]) > minPow[bin]:
            if negZ[0, 3] * (1 - minDiff) > negZ[1, 3]:
                peakF = negZ[0, 2]
                peakBin = int(negZ[0, 1])
            else:
                peakF = np.nan
                subBin = int(negZ[0, 1])
        else:
            peakF = np.nan
            subBin = np.nan

    # search for positive (upward going) zero-crossings
    slen = round(1 / fres)  # define number of bins included in shallow slope search

    if np.isnan(peakF) and np.isnan(subBin):  # if no evidence of peak activity, no parameter estimation indicated
        posZ1, posZ2, f1, f2, inf1, inf2, Q, Qf = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    elif np.isnan(peakF):  # deal with spectra lacking a clear primary peak
        f1, posZ1 = find_f1(f, d0, d1, negZ, minPow, slen, subBin)
        f2, posZ2 = find_f2(f, d0, d1, negZ, minPow, slen, subBin)
        inf1, inf2, Q, Qf = np.nan, np.nan, np.nan, np.nan
    else:  # for the primary peak spectra
        f1, posZ1 = find_f1(f, d0, d1, negZ, minPow, slen, peakBin)
        f2, posZ2 = find_f2(f, d0, d1, negZ, minPow, slen, peakBin)

        # define boundaries by inflection points
        inf1, inf2 = np.nan, np.nan
        for k in range(peakBin):
            if np.sign(d2[k]) > np.sign(d2[k + 1]):  # look for switch from positive to negative derivative values
                min1 = k if d2[k] < d2[k + 1] else k + 1
                inf1 = f[min1]
                break

        for k in range(peakBin, len(d2) - 1):
            if np.sign(d2[k]) < np.sign(d2[k + 1]):  # look for upward zero-crossing
                min2 = k if d2[k] > d2[k + 1] else k + 1
                inf2 = f[min2]
                break

        # estimate area under curve between inflection points
        Q = trapz(d0[min1:min2], f[min1:min2])
        Qf = Q / (min2 - min1)

    return peakF, posZ1, posZ2, f1, f2, inf1, inf2, Q, Qf


def find_f1(f, d0, d1, negZ, minPow, slen, bin):
    """
    Searches 1st derivative for evidence of local minima or near horizontal
    function prior to alpha peak. This location will be taken as the lower
    bound of the individual alpha band used to calculate CoG (f1).

    Parameters
    ----------
    f : ndarray
        Frequency bin vector.
    d0 : ndarray
        Smoothed PSD estimate vector.
    d1 : ndarray
        1st derivative vector.
    negZ : ndarray, shape=(n_peaks, 4)
        Vector / matrix of negative zero-crossings detected within alpha window.
    minPow : ndarray
        Vector of minimum power threshold values defining candidate peaks.
    slen : int
        Number of bins examined for evidence of shallow slope (~1 Hz interval).
    bin : int
        Frequency bin indexing peak / subpeak.

    Returns
    -------
    f1 : float
        Frequency bin indexing f1.
    posZ1 : float
        Frequency of f1.

    """

    # contingency for multiple peaks
    if negZ.shape[0] > 1:
        negZ = negZ[negZ[:, 2].argsort()]
        for z in range(negZ.shape[0]):
            if np.log10(negZ[z, 3]) > minPow[int(negZ[0, 1])] or negZ[z, 3] > (0.5 * d0[bin]):
                leftPeak = int(negZ[z, 1])
                break
            else:
                leftPeak = bin
    else:
        leftPeak = bin

    posZ1 = []
    cnt = 0

    for k in range(1, leftPeak):
        if np.sign(d1[k]) < np.sign(d1[k + 1]):
            mink = np.argmin(np.abs([d0[k - 1], d0[k], d0[k + 1]]))
            if mink == 0:
                minim = k - 1
            elif mink == 1:
                minim = k
            else:
                minim = k + 1

            cnt += 1
            posZ1.append([cnt, minim, f[minim]])
        elif np.abs(d1[k]) < 1 and less_than_1(d1[k + 1:k + slen]):
            minim = k
            cnt += 1
            posZ1.append([cnt, minim, f[minim]])

    posZ1 = np.array(posZ1)

    if posZ1.shape[0] == 1:
        f1 = int(posZ1[0, 1])
        posZ1 = posZ1[0, 2]
    else:
        posZ1 = posZ1[np.argsort(posZ1[:, 2])[::-1]]
        f1 = int(posZ1[0, 1])
        posZ1 = posZ1[0, 2]

    return f1, posZ1


def find_f2(f, d0, d1, negZ, minPow, slen, bin):
    """
    Searches 1st derivative for evidence of local minima or near horizontal
    function post alpha peak. This location will be taken as the lower
    bound of the individual alpha band used to calculate CoG (f2).

    Parameters
    ----------
    f : ndarray
        Frequency bin vector.
    d0 : ndarray
        Smoothed PSD estimate vector.
    d1 : ndarray
        1st derivative of d0.
    negZ : ndarray
        Vector / matrix of negative zero-crossings detected within alpha window.
    minPow : ndarray
        Vector of minimum power threshold values defining candidate peaks.
    slen : int
        Number of bins examined for evidence of shallow slope (~1 Hz interval).
    bin : int
        Frequency bin indexing peak / subpeak.

    Returns
    -------
    f2 : float
        Frequency bin indexing f2.
    posZ2 : float
        Frequency of f2.

    """

    # contingency for multiple peaks
    if negZ.shape[0] > 1:
        negZ = negZ[negZ[:, 2].argsort()[::-1]]
        for z in range(negZ.shape[0]):
            if (np.log10(negZ[z, 3]) > minPow[int(negZ[0, 1])]
                    or negZ[z, 3] > (0.5 * d0[bin])):
                right_peak = int(negZ[z, 1])
                break
            else:
                right_peak = bin
    else:
        right_peak = bin

    posZ2 = []
    cnt = 0

    for k in range(right_peak + 1, len(d1) - slen):
        if np.sign(d1[k]) < np.sign(d1[k + 1]):
            mink = np.argmin(np.abs([d0[k - 1], d0[k], d0[k + 1]]))
            if mink == 0:
                minim = k - 1
            elif mink == 1:
                minim = k
            else:
                minim = k + 1

            cnt += 1
            posZ2.append([cnt, minim, f[minim]])
        elif np.abs(d1[k]) < 1 and less_than_1(d1[k + 1:k + slen]):
            minim = k
            cnt += 1
            posZ2.append([cnt, minim, f[minim]])

    posZ2 = np.array(posZ2)

    f2 = int(posZ2[0, 1])
    posZ2 = posZ2[0, 2]

    return f2, posZ2


def less_than_1(d1):
    """
    When encounter 1st derivative absolute value < 1, eval whether following
    values (within segment) remain < +/- 1.

    Parameters
    ----------
    d1 : ndarray
        Segment of 1st derivative spanning approx. 1 Hz.

    Returns
    -------
    tval : bool
        True if all values within segment remain < +/- 1, False otherwise.

    """
    if len(d1) < 2:
        raise ValueError("Length of 1st derivative segment < 2")

    t = np.zeros_like(d1)
    for kx in range(len(d1)):
        t[kx] = np.abs(d1[kx]) < 1

    return np.all(t)