"""Zapline-plus for automatic removal of frequency-specific noise artifacts.

This module implements Zapline-plus, an extension of the Zapline algorithm
that enables fully automatic removal of line noise and other frequency-specific
artifacts from M/EEG data.

Based on:
Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for
automatic and adaptive removal of frequency-specific noise artifacts in M/EEG.
Human Brain Mapping, 43(9), 2743-2758.

Original Zapline by:
de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
power line artifacts. NeuroImage, 207, 116356.


Differences from Matlab implementation:

Finding noise frequencies:
- one iteration returning all frequencies

Adaptive chunking:
- merged chunks at edges if too short

Plotting:
- only once per frequency after cleaning



"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from meegkit.dss import dss_line


def zapline_plus(
    data: np.ndarray,
    sfreq: float,
    fline: float | list[float] | None = None,
    nkeep: int = 0,
    adaptiveNremove: bool = True,
    fixedNremove: int = 1,
    minfreq: float = 17.0,
    maxfreq: float = 99.0,
    chunkLength: float = 0.0,
    minChunkLength: float = 30.0,
    noiseCompDetectSigma: float = 3.0,
    adaptiveSigma: bool = True,
    minsigma: float = 2.5,
    maxsigma: float = 4.0,
    detectionWinsize: float = 6.0,
    coarseFreqDetectPowerDiff: float = 4.0,
    coarseFreqDetectLowerPowerDiff: float = 1.76,
    searchIndividualNoise: bool = True,
    freqDetectMultFine: float = 2.0,
    detailedFreqBoundsUpper: tuple[float, float] = (0.05, 0.05),
    detailedFreqBoundsLower: tuple[float, float] = (0.4, 0.1),
    maxProportionAboveUpper: float = 0.005,
    maxProportionBelowLower: float = 0.005,
    plotResults: bool = False,
    figsize: tuple[int, int] = (14, 10),
    vanilla_mode: bool = False,
) -> tuple[np.ndarray, dict]:
    """Remove line noise and other frequency-specific artifacts using Zapline-plus.

    Parameters
    ----------
        data : array, shape=(n_times, n_chans)
        Input data.
    sfreq : float
        Sampling frequency in Hz.
    fline : float | list of float | None
        Noise frequency or frequencies to remove. If None, frequencies are
        detected automatically. Defaults to None.
    nkeep : int
        Number of principal components to keep in DSS. If 0, no dimensionality
        reduction is applied. Defaults to 0.
    adaptiveNremove : bool
        If True, automatically detect the number of components to remove.
        If False, use fixedNremove for all chunks. Defaults to True.
    fixedNremove : int
        Fixed number of components to remove per chunk. Used when
        adaptiveNremove=False, or as minimum when adaptiveNremove=True.
        Defaults to 1.
    minfreq : float
        Minimum frequency (Hz) to consider when detecting noise automatically.
        Defaults to 17.0.
    maxfreq : float
        Maximum frequency (Hz) to consider when detecting noise automatically.
        Defaults to 99.0.
    chunkLength : float
        Length of chunks (seconds) for cleaning. If 0, adaptive chunking based
        on noise covariance stability is used. Set to -1 via vanilla_mode to
        process the entire recording as a single chunk. Defaults to 0.0.
    minChunkLength : float
        Minimum chunk length (seconds) when using adaptive chunking.
        Defaults to 30.0.
    noiseCompDetectSigma : float
        Initial SD threshold for iterative outlier detection of noise components.
        Defaults to 3.0.
    adaptiveSigma : bool
        If True, automatically adapt noiseCompDetectSigma and fixedNremove
        based on cleaning results. Defaults to True.
    minsigma : float
        Minimum SD threshold when adapting noiseCompDetectSigma.
        Defaults to 2.5.
    maxsigma : float
        Maximum SD threshold when adapting noiseCompDetectSigma.
        Defaults to 4.0.
    detectionWinsize : float
        Window size (Hz) for noise frequency detection. Defaults to 6.0.
    coarseFreqDetectPowerDiff : float
        Threshold (10*log10) above center power to detect a peak as noise.
        Defaults to 4.0.
    coarseFreqDetectLowerPowerDiff : float
        Threshold (10*log10) above center power to detect end of noise peak.
        Defaults to 1.76.
    searchIndividualNoise : bool
        If True, search for individual noise peaks in each chunk.
        Defaults to True.
    freqDetectMultFine : float
        Multiplier for fine noise frequency detection threshold. Defaults to 2.0.
    detailedFreqBoundsUpper : tuple of float
        Frequency boundaries (Hz) for fine threshold of too weak cleaning.
        Defaults to (0.05, 0.05).
    detailedFreqBoundsLower : tuple of float
        Frequency boundaries (Hz) for fine threshold of too strong cleaning.
        Defaults to (0.4, 0.1).
    maxProportionAboveUpper : float
        Maximum proportion of samples above upper threshold before adapting.
        Defaults to 0.005.
    maxProportionBelowLower : float
        Maximum proportion of samples below lower threshold before adapting.
        Defaults to 0.005.
    plotResults : bool
        If True, generate diagnostic plots for each cleaned frequency.
        Defaults to False.
    figsize : tuple of int
        Figure size for diagnostic plots. Defaults to (14, 10).
    vanilla_mode : bool
        If True, disable all Zapline-plus features and use vanilla Zapline behavior:
        - Process entire dataset as single chunk
        - Use fixed component removal (no adaptive detection)
        - No individual chunk frequency detection
        - No adaptive parameter tuning
        Requires fline to be specified (not None). Defaults to False.

    Returns
    -------
    clean_data : array, shape=(n_times, n_chans)
        Cleaned data.
    config : dict
        Configuration dictionary containing all parameters and analytics.

    Notes
    -----
    The algorithm proceeds as follows:
    1. Detect noise frequencies (if not provided)
    2. Segment data into chunks with stable noise topography
    3. Apply Zapline to each chunk
    4. Automatically detect and remove noise components
    5. Adapt parameters if cleaning is too weak or too strong

    Examples
    --------
    Remove 50 Hz line noise automatically:
    >>> clean_data, config = zapline_plus(data, sfreq=500, fline=50)

    Remove line noise with automatic frequency detection:
    >>> clean_data, config = zapline_plus(data, sfreq=500)

    """
    n_times, n_chans = data.shape

    # Handle vanilla mode (ZapLine without plus)
    if vanilla_mode:
        logging.warning(
            "vanilla_mode=True: Using vanilla Zapline behavior. "
            "All adaptive features disabled."
        )
        if fline is None:
            raise ValueError("vanilla_mode requires fline to be specified (not None)")

        for param_name in [
            "adaptiveNremove",
            "adaptiveSigma",
            "searchIndividualNoise",
        ]:
            if locals()[param_name]:
                logging.warning(f"vanilla_mode=True: Overriding {param_name} to False.")

        # Override all adaptive features
        adaptiveNremove = False
        adaptiveSigma = False
        searchIndividualNoise = False
        chunkLength = -1  # Zapline vanilla deals with single chunk

    # if nothing is adaptive, only one iteration per frequency
    if not (adaptiveNremove and adaptiveSigma):
        max_iterations = 1

    # check for globally flat channels
    # will be omitted during processing and reintroduced later
    diff_data = np.diff(data, axis=0)
    global_flat = np.where(np.all(diff_data == 0, axis=0))[0]
    if len(global_flat) > 0:
        logging.warning(
            f"Detected {len(global_flat)} globally flat channels: {global_flat}. "
            f"Removing for processing, will add back after."
        )
        flat_data = data[:, global_flat]
        active_channels = np.setdiff1d(np.arange(n_chans), global_flat)
        data = data[:, active_channels]
    else:
        active_channels = np.arange(n_chans)
        flat_data = None

    # initialize configuration
    config = {
        "sfreq": sfreq,
        "fline": fline,
        "nkeep": nkeep,
        "adaptiveNremove": adaptiveNremove,
        "fixedNremove": fixedNremove,
        "minfreq": minfreq,
        "maxfreq": maxfreq,
        "chunkLength": chunkLength,
        "minChunkLength": minChunkLength,
        "noiseCompDetectSigma": noiseCompDetectSigma,
        "adaptiveSigma": adaptiveSigma,
        "minsigma": minsigma,
        "maxsigma": maxsigma,
        "detectionWinsize": detectionWinsize,
        "coarseFreqDetectPowerDiff": coarseFreqDetectPowerDiff,
        "coarseFreqDetectLowerPowerDiff": coarseFreqDetectLowerPowerDiff,
        "searchIndividualNoise": searchIndividualNoise,
        "freqDetectMultFine": freqDetectMultFine,
        "detailedFreqBoundsUpper": detailedFreqBoundsUpper,
        "detailedFreqBoundsLower": detailedFreqBoundsLower,
        "maxProportionAboveUpper": maxProportionAboveUpper,
        "maxProportionBelowLower": maxProportionBelowLower,
        "analytics": {},
    }

    # detect noise frequencies if not provided
    if fline is None:
        fline = _detect_noise_frequencies(
            data,
            sfreq,
            minfreq,
            maxfreq,
            detectionWinsize,
            coarseFreqDetectPowerDiff,
            coarseFreqDetectLowerPowerDiff,
        )
    elif not isinstance(fline, list):
        fline = [fline]

    if len(fline) == 0:
        logging.info("No noise frequencies detected. Returning original data.")
        return data.copy(), config

    config["detected_fline"] = fline

    # retain input data
    clean_data = data.copy()

    # Process each noise frequency
    for freq_idx, target_freq in enumerate(fline):
        print(f"Processing noise frequency: {target_freq:.2f} Hz")

        if chunkLength == -1:
            # single chunk
            chunks = [(0, n_times)]
        elif chunkLength == 0:
            # adaptive chunking
            chunks = _adaptive_chunking(clean_data, sfreq, target_freq, minChunkLength)
        else:
            # fixed-length chunks
            chunk_samples = int(chunkLength * sfreq)
            chunks = [
                (i, min(i + chunk_samples, n_times))
                for i in range(0, n_times, chunk_samples)
            ]

        # initialize tracking variables
        current_sigma = noiseCompDetectSigma
        current_fixed = fixedNremove
        too_strong_once = False
        iteration = 0
        max_iterations = 20

        while iteration < max_iterations:
            iteration += 1

            # Clean each chunk
            chunk_results = []
            for chunk_start, chunk_end in chunks:
                chunk_data = clean_data[chunk_start:chunk_end, :]

                # Detect chunk-specific noise frequency
                if searchIndividualNoise:
                    chunk_freq, has_noise = _detect_chunk_noise_frequency(
                        chunk_data,
                        sfreq,
                        target_freq,
                        detectionWinsize,
                        freqDetectMultFine,
                        detailed_freq_bounds=detailedFreqBoundsUpper,
                    )
                else:
                    chunk_freq = target_freq
                    has_noise = True

                # Apply Zapline to chunk
                if has_noise:
                    if adaptiveNremove:
                        n_remove = _detect_noise_components(
                            chunk_data, sfreq, chunk_freq, current_sigma, nkeep
                        )
                        n_remove = max(n_remove, current_fixed)
                    else:
                        n_remove = current_fixed

                    # Cap at 1/5 of components
                    n_remove = min(n_remove, n_chans // 5)
                else:
                    n_remove = current_fixed

                # clean chunk
                cleaned_chunk = _apply_zapline_to_chunk(
                    chunk_data, sfreq, chunk_freq, n_remove, nkeep
                )

                chunk_results.append(
                    {
                        "start": chunk_start,
                        "end": chunk_end,
                        "freq": chunk_freq,
                        "n_remove": n_remove,
                        "has_noise": has_noise,
                        "data": cleaned_chunk,
                    }
                )

            # reconstruct cleaned data
            temp_clean = clean_data.copy()
            for result in chunk_results:
                temp_clean[result["start"] : result["end"], :] = result["data"]

            # check if cleaning is optimal
            cleaning_status = _check_cleaning_quality(
                data,
                temp_clean,
                sfreq,
                target_freq,
                detectionWinsize,
                freqDetectMultFine,
                detailedFreqBoundsUpper,
                detailedFreqBoundsLower,
                maxProportionAboveUpper,
                maxProportionBelowLower,
            )

            # store analytics
            config["analytics"][f"freq_{freq_idx}"] = {
                "target_freq": target_freq,
                "iteration": iteration,
                "sigma": current_sigma,
                "fixed_nremove": current_fixed,
                "n_chunks": len(chunks),
                "chunk_results": chunk_results,
                "cleaning_status": cleaning_status,
            }

            # check if we need to adapt
            if cleaning_status == "good":
                clean_data = temp_clean
                break

            elif cleaning_status == "too_weak" and not too_strong_once:
                if current_sigma > minsigma:
                    current_sigma = max(current_sigma - 0.25, minsigma)
                    current_fixed += 1
                    logging.info(
                        f"Cleaning too weak. Adjusting sigma to {current_sigma:.2f}, "
                        f"fixed removal to {current_fixed}"
                    )
                else:
                    logging.info("At minimum sigma, accepting result")
                    clean_data = temp_clean
                    break

            elif cleaning_status == "too_strong":
                too_strong_once = True
                if current_sigma < maxsigma:
                    current_sigma = min(current_sigma + 0.25, maxsigma)
                    current_fixed = max(current_fixed - 1, fixedNremove)
                    logging.info(
                        f"Cleaning too strong. Adjusting sigma to {current_sigma:.2f}, "
                        f"fixed removal to {current_fixed}"
                    )
                else:
                    logging.info("At maximum sigma, accepting result")
                    clean_data = temp_clean
                    break

            else:
                # Too strong takes precedence, or we can't improve further
                clean_data = temp_clean
                break

        # Generate diagnostic plot
        if plotResults:
            _plot_cleaning_results(
                data,
                clean_data,
                sfreq,
                target_freq,
                config["analytics"][f"freq_{freq_idx}"],
                figsize,
            )

    # add flat channels back to data, if present
    if flat_data is not None:
        full_clean = np.zeros((n_times, n_chans))
        full_clean[:, active_channels] = clean_data
        full_clean[:, global_flat] = flat_data
        clean_data = full_clean

    return clean_data, config


def _detect_noise_frequencies(
    data, sfreq, minfreq, maxfreq, winsize, power_diff_high, power_diff_low
):
    """
    Detect noise frequencies.

    This is an exact implementation of find_next_noisefreq.m with the only difference
    that all peaks are returned instead of this being called iteratively.

    How it works
    ------------
    1. Compute PSD and log-transform.
    2. Slide a window across frequencies from minfreq to maxfreq.
    3. For each frequency, compute center power as mean of left and right thirds.
    4. Use a state machine to detect peaks:
        - SEARCHING: If current power - center power > power_diff_high,
            mark peak start and switch to IN_PEAK.
        - IN_PEAK: If current power - center power <= power_diff_low,
            mark peak end, find max within peak, record frequency,
            and switch to SEARCHING.
    5. Return list of detected noise frequencies.
    """
    # Compute PSD
    freqs, psd = _compute_psd(data, sfreq)
    log_psd = 10 * np.log10(np.mean(psd, axis=1))

    # State machine variables
    in_peak = False
    peak_start_idx = None
    noise_freqs = []

    # Search bounds
    start_idx = np.searchsorted(freqs, minfreq)
    end_idx = np.searchsorted(freqs, maxfreq)

    # Window size in samples
    freq_resolution = freqs[1] - freqs[0]
    win_samples = int(winsize / freq_resolution)

    idx = start_idx
    while idx < end_idx:
        # Get window around current frequency
        win_start = max(0, idx - win_samples // 2)
        win_end = min(len(freqs), idx + win_samples // 2)
        win_psd = log_psd[win_start:win_end]

        if len(win_psd) < 3:
            idx += 1
            continue

        # Compute center power (mean of left and right thirds)
        n_third = len(win_psd) // 3
        if n_third < 1:
            idx += 1
            continue

        left_third = win_psd[:n_third]
        right_third = win_psd[-n_third:]
        center_power = np.mean([np.mean(left_third), np.mean(right_third)])

        current_power = log_psd[idx]

        # State machine logic
        if not in_peak:
            # State: SEARCHING - Check for peak start
            if current_power - center_power > power_diff_high:
                in_peak = True
                peak_start_idx = idx

        else:
            # State: IN_PEAK - Check for peak end
            if current_power - center_power <= power_diff_low:
                in_peak = False
                peak_end_idx = idx

                # Find the actual maximum within the peak
                if peak_start_idx is not None and peak_end_idx > peak_start_idx:
                    peak_region = log_psd[peak_start_idx:peak_end_idx]
                    max_offset = np.argmax(peak_region)
                    max_idx = peak_start_idx + max_offset
                    noise_freqs.append(freqs[max_idx])

                    # Skip past this peak to avoid re-detection
                    idx = peak_end_idx
                    continue

        idx += 1

    return noise_freqs


def _adaptive_chunking(
    data,
    sfreq,
    target_freq,
    min_chunk_length,
    detection_winsize=6.0,
    prominence_quantile=0.95,
):
    """Segment data into chunks with stable noise topography."""
    n_times, n_chans = data.shape

    if n_times < sfreq * min_chunk_length:
        logging.warning("Data too short for adaptive chunking. Using single chunk.")
        return [(0, n_times)]

    # Narrow-band filter around target frequency
    bandwidth = detection_winsize / 2.0
    filtered = _narrowband_filter(data, sfreq, target_freq, bandwidth=bandwidth)

    # Compute covariance matrices for 1-second epochs
    epoch_length = int(sfreq)
    n_epochs = n_times // epoch_length

    distances = np.zeros(n_epochs)
    prev_cov = None

    for i in range(n_epochs):
        start = i * epoch_length
        end = start + epoch_length
        epoch = filtered[start:end, :]
        cov = np.cov(epoch, rowvar=False)

        if prev_cov is not None:
            # Frobenius norm of difference
            distances[i] = np.linalg.norm(cov - prev_cov, "fro")
        # else: distance[i] already 0 from initialization

        prev_cov = cov

    if len(distances) < 2:
        return [(0, n_times)]

    # find all peaks to get prominence distribution
    peaks_all, properties_all = signal.find_peaks(distances, prominence=0)

    if len(peaks_all) == 0 or "prominences" not in properties_all:
        # No peaks found
        logging.warning("No peaks found in distance signal. Using single chunk.")
        return [(0, n_times)]

    prominences = properties_all["prominences"]

    # filter by prominence quantile
    min_prominence = np.quantile(prominences, prominence_quantile)
    min_distance_epochs = int(min_chunk_length)  # Convert seconds to epochs

    peaks, properties = signal.find_peaks(
        distances, prominence=min_prominence, distance=min_distance_epochs
    )

    # cconvert peak locations (in epochs) to sample indices
    chunk_starts = [0]
    for peak in peaks:
        chunk_start_sample = peak * epoch_length
        chunk_starts.append(chunk_start_sample)
    chunk_starts.append(n_times)

    # create chunk list
    chunks = []
    for i in range(len(chunk_starts) - 1):
        start = chunk_starts[i]
        end = chunk_starts[i + 1]
        chunks.append((start, end))

    # ensure minimum chunk length at edges
    min_chunk_samples = int(min_chunk_length * sfreq)

    if len(chunks) > 1:
        # check first chunk
        if chunks[0][1] - chunks[0][0] < min_chunk_samples:
            # merge with next
            chunks[1] = (chunks[0][0], chunks[1][1])
            chunks.pop(0)

    if len(chunks) > 1:
        # check last chunk
        if chunks[-1][1] - chunks[-1][0] < min_chunk_samples:
            # merge with previous
            chunks[-2] = (chunks[-2][0], chunks[-1][1])
            chunks.pop(-1)

    return chunks


def _detect_chunk_noise_frequency(
    data,
    sfreq,
    target_freq,
    winsize,
    mult_fine,
    detailed_freq_bounds=(-0.05, 0.05),  # ← Add this parameter
):
    """Detect chunk-specific noise frequency around target."""
    freqs, psd = _compute_psd(data, sfreq)
    log_psd = 10 * np.log10(np.mean(psd, axis=1))

    # get frequency mask
    search_mask = (freqs >= target_freq + detailed_freq_bounds[0]) & (
        freqs <= target_freq + detailed_freq_bounds[1]
    )

    if not np.any(search_mask):
        return target_freq, False

    search_freqs = freqs[search_mask]
    search_psd = log_psd[search_mask]

    # find peak
    peak_idx = np.argmax(search_psd)
    peak_freq = search_freqs[peak_idx]
    peak_power = search_psd[peak_idx]

    # Compute threshold (uses broader window)
    win_mask = (freqs >= target_freq - winsize / 2) & (freqs <= target_freq + winsize / 2)
    win_psd = log_psd[win_mask]

    n_third = len(win_psd) // 3
    left_third = win_psd[:n_third]
    right_third = win_psd[-n_third:]
    center = np.mean([np.mean(left_third), np.mean(right_third)])

    # Compute deviation (lower 5% quantiles)
    lower_quant_left = np.percentile(left_third, 5)
    lower_quant_right = np.percentile(right_third, 5)
    deviation = center - np.mean([lower_quant_left, lower_quant_right])

    threshold = center + mult_fine * deviation

    has_noise = peak_power > threshold

    return peak_freq, has_noise


def _detect_noise_components(data, sfreq, target_freq, sigma, nkeep):
    """Detect number of noise components to remove using outlier detection."""
    # Apply DSS to get component scores
    _, scores = dss_line(data, target_freq, sfreq, nkeep=nkeep)

    if scores is None or len(scores) == 0:
        return 1

    # Sort scores in descending order
    sorted_scores = np.sort(scores)[::-1]

    # Iterative outlier detection
    n_remove = 0
    remaining = sorted_scores.copy()

    while len(remaining) > 1:
        mean_val = np.mean(remaining)
        std_val = np.std(remaining)
        threshold = mean_val + sigma * std_val

        if remaining[0] > threshold:
            n_remove += 1
            remaining = remaining[1:]
        else:
            break

    return max(n_remove, 1)


def _apply_zapline_to_chunk(chunk_data, sfreq, chunk_freq, n_remove, nkeep):
    """Apply Zapline to a single chunk, handling flat channels."""
    n_samples, n_chans = chunk_data.shape

    # Detect flat channels (zero variance)
    diff_chunk = np.diff(chunk_data, axis=0)
    flat_channels = np.where(np.all(diff_chunk == 0, axis=0))[0]

    if len(flat_channels) > 0:
        logging.warning(
            f"Detected {len(flat_channels)} flat channels in chunk: {flat_channels}. "
            f"Removing temporarily for processing."
        )

        # store flat channel data
        flat_channel_data = chunk_data[:, flat_channels]

        # remove flat channels from processing
        active_channels = np.setdiff1d(np.arange(n_chans), flat_channels)
        chunk_data_active = chunk_data[:, active_channels]

        # process only active channels
        cleaned_active, _ = dss_line(
            chunk_data_active,
            fline=chunk_freq,
            sfreq=sfreq,
            nremove=n_remove,
            nkeep=nkeep,
        )

        # Reconstruct full data with flat channels
        cleaned_chunk = np.zeros_like(chunk_data)
        cleaned_chunk[:, active_channels] = cleaned_active
        cleaned_chunk[:, flat_channels] = (
            flat_channel_data  # Add flat channels back unchanged
        )

    else:
        # no flat channels, process normally
        cleaned_chunk, _ = dss_line(
            chunk_data,
            fline=chunk_freq,
            sfreq=sfreq,
            nremove=n_remove,
            nkeep=nkeep,
        )

    return cleaned_chunk


def _check_cleaning_quality(
    original_data,
    cleaned_data,
    sfreq,
    target_freq,
    winsize,
    mult_fine,
    bounds_upper,
    bounds_lower,
    max_prop_above,
    max_prop_below,
):
    """Check if cleaning is too weak, too strong, or good."""
    # Compute PSDs
    freqs, psd_clean = _compute_psd(cleaned_data, sfreq)
    log_psd_clean = 10 * np.log10(np.mean(psd_clean, axis=1))

    # Compute fine thresholds
    win_mask = (freqs >= target_freq - winsize / 2) & (freqs <= target_freq + winsize / 2)
    win_psd = log_psd_clean[win_mask]

    n_third = len(win_psd) // 3
    left_third = win_psd[:n_third]
    right_third = win_psd[-n_third:]
    center = np.mean([np.mean(left_third), np.mean(right_third)])

    # Deviation from lower quantiles
    lower_quant_left = np.percentile(left_third, 5)
    lower_quant_right = np.percentile(right_third, 5)
    deviation = center - np.mean([lower_quant_left, lower_quant_right])

    # Upper threshold (too weak cleaning)
    upper_mask = (freqs >= target_freq - bounds_upper[0]) & (
        freqs <= target_freq + bounds_upper[1]
    )
    upper_threshold = center + mult_fine * deviation
    upper_psd = log_psd_clean[upper_mask]
    prop_above = np.mean(upper_psd > upper_threshold)

    # Lower threshold (too strong cleaning)
    lower_mask = (freqs >= target_freq - bounds_lower[0]) & (
        freqs <= target_freq + bounds_lower[1]
    )
    lower_threshold = center - mult_fine * deviation
    lower_psd = log_psd_clean[lower_mask]
    prop_below = np.mean(lower_psd < lower_threshold)

    if prop_below > max_prop_below:
        return "too_strong"
    elif prop_above > max_prop_above:
        return "too_weak"
    else:
        return "good"


def _compute_psd(data, sfreq, nperseg=None):
    """Compute power spectral density using Welch's method."""
    if nperseg is None:
        nperseg = int(sfreq * 4)  # 4-second windows

    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        axis=0,
    )

    return freqs, psd


def _narrowband_filter(data, sfreq, center_freq, bandwidth=3.0):
    """Apply narrow-band filter around center frequency."""
    nyq = sfreq / 2
    low = (center_freq - bandwidth) / nyq
    high = (center_freq + bandwidth) / nyq

    # Ensure valid frequency range
    low = max(low, 0.001)
    high = min(high, 0.999)

    sos = signal.butter(4, [low, high], btype="band", output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=0)

    return filtered


def _plot_cleaning_results(original, cleaned, sfreq, target_freq, analytics, figsize):
    """Generate diagnostic plots for cleaning results."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Compute PSDs
    freqs, psd_orig = _compute_psd(original, sfreq)
    _, psd_clean = _compute_psd(cleaned, sfreq)

    log_psd_orig = 10 * np.log10(np.mean(psd_orig, axis=1))
    log_psd_clean = 10 * np.log10(np.mean(psd_clean, axis=1))

    # 1. Zoomed spectrum around noise frequency
    ax1 = fig.add_subplot(gs[0, 0])
    zoom_mask = (freqs >= target_freq - 1.1) & (freqs <= target_freq + 1.1)
    ax1.plot(freqs[zoom_mask], log_psd_orig[zoom_mask], "k-", label="Original")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power (dB)")
    ax1.set_title(f"Detected frequency: {target_freq:.2f} Hz")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Number of removed components per chunk
    ax2 = fig.add_subplot(gs[0, 1])
    chunk_results = analytics["chunk_results"]
    n_removes = [cr["n_remove"] for cr in chunk_results]
    ax2.bar(range(len(n_removes)), n_removes)
    ax2.set_xlabel("Chunk")
    ax2.set_ylabel("# Components removed")
    ax2.set_title(f"Removed components (mean={np.mean(n_removes):.1f})")
    ax2.grid(True, alpha=0.3)

    # 3. Individual noise frequencies per chunk
    ax3 = fig.add_subplot(gs[0, 2])
    chunk_freqs = [cr["freq"] for cr in chunk_results]
    time_min = np.array([cr["start"] for cr in chunk_results]) / sfreq / 60
    ax3.plot(time_min, chunk_freqs, "o-")
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_title("Individual noise frequencies")
    ax3.grid(True, alpha=0.3)

    # 4. Component scores (would need actual scores from DSS)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.text(
        0.5,
        0.5,
        "Component scores\n(requires DSS output)",
        ha="center",
        va="center",
        transform=ax4.transAxes,
    )
    ax4.set_title("Mean artifact scores")

    # 5. Cleaned spectrum (zoomed)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(freqs[zoom_mask], log_psd_clean[zoom_mask], "g-", label="Cleaned")
    ax5.set_xlabel("Frequency (Hz)")
    ax5.set_ylabel("Power (dB)")
    ax5.set_title("Cleaned spectrum")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Full spectrum
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(freqs, log_psd_orig, "k-", alpha=0.5, label="Original")
    ax6.plot(freqs, log_psd_clean, "g-", label="Cleaned")
    ax6.axvline(target_freq, color="r", linestyle="--", alpha=0.5)
    ax6.set_xlabel("Frequency (Hz)")
    ax6.set_ylabel("Power (dB)")
    ax6.set_title("Full power spectrum")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, 100])

    # 7. Removed power (ratio)
    ax7 = fig.add_subplot(gs[1, 2])
    noise_mask = (freqs >= target_freq - 0.05) & (freqs <= target_freq + 0.05)
    ratio_orig = np.mean(psd_orig[noise_mask, :]) / np.mean(psd_orig)
    ratio_clean = np.mean(psd_clean[noise_mask, :]) / np.mean(psd_clean)

    ax7.text(
        0.5,
        0.6,
        f"Original ratio: {ratio_orig:.2f}",
        ha="center",
        transform=ax7.transAxes,
    )
    ax7.text(
        0.5,
        0.4,
        f"Cleaned ratio: {ratio_clean:.2f}",
        ha="center",
        transform=ax7.transAxes,
    )
    ax7.set_title("Noise/surroundings ratio")
    ax7.axis("off")

    # 8. Below noise frequencies
    ax8 = fig.add_subplot(gs[1, 3])
    below_mask = (freqs >= target_freq - 11) & (freqs <= target_freq - 1)
    ax8.plot(
        freqs[below_mask], log_psd_orig[below_mask], "k-", alpha=0.5, label="Original"
    )
    ax8.plot(freqs[below_mask], log_psd_clean[below_mask], "g-", label="Cleaned")
    ax8.set_xlabel("Frequency (Hz)")
    ax8.set_ylabel("Power (dB)")
    ax8.set_title("Power below noise frequency")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.suptitle(
        f"Zapline-plus cleaning results: {target_freq:.2f} Hz "
        f"(iteration {analytics['iteration']})",
        fontsize=14,
        y=0.98,
    )

    plt.show()

    return fig


# Convenience function with simpler interface
def remove_line_noise(
    data: np.ndarray,
    sfreq: float,
    fline: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Remove line noise from data using Zapline-plus.

    This is a simplified interface to zapline_plus() that returns only
    the cleaned data.

    Parameters
    ----------
    data : array, shape=(n_times, n_chans)
        Input data.
    sfreq : float
        Sampling frequency in Hz.
    fline : float | None
        Line noise frequency. If None, automatically detected.
    **kwargs
        Additional arguments passed to zapline_plus().

    Returns
    -------
    clean_data : array, shape=(n_times, n_chans)
        Cleaned data.

    Examples
    --------
    >>> clean = remove_line_noise(data, sfreq=500, fline=50)

    """
    clean_data, _ = zapline_plus(data, sfreq, fline=fline, **kwargs)
    return clean_data
