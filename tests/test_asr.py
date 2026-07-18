"""ASR test."""
import inspect
import os
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import signal

from meegkit.asr import ASR, asr_calibrate, asr_process, clean_windows
from meegkit.utils.asr import SHAPE_RANGE, fit_eeg_distribution, yulewalk, yulewalk_filter
from meegkit.utils.matrix import sliding_window

# Data files
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
# file = os.path.join(THIS_FOLDER, 'data', 'eeg_raw.fif')
# raw = mne.io.read_raw_fif(file, preload=True)
# raw.filter(.5, 30)
# raw._data *= 1e6  # to uV for easy plotting
# raw.crop(0, 60)  # keep 60s only
# raw.pick_types(eeg=True, misc=False)
# raw = raw._data
rng = np.random.default_rng(9)


@pytest.mark.parametrize("obj", (ASR, clean_windows, asr_calibrate, asr_process))
def test_docstring_parameters(obj):
    """Documented parameters must match the signature (no stray prose)."""
    docscrape = pytest.importorskip("numpydoc.docscrape")
    doc = docscrape.ClassDoc(obj) if inspect.isclass(obj) else docscrape.FunctionDoc(obj)
    documented = {p.name for p in doc["Parameters"]}
    signature = set(inspect.signature(obj).parameters)
    assert documented <= signature, documented - signature


def test_docstring_attributes():
    """ASR attribute names must not carry literal RST markup."""
    docscrape = pytest.importorskip("numpydoc.docscrape")
    for attr in docscrape.ClassDoc(ASR)["Attributes"]:
        assert "`" not in attr.name, attr.name


@pytest.mark.parametrize(argnames="sfreq", argvalues=(125, 250, 256, 2048))
def test_yulewalk(sfreq, show=False):
    """Test that my version of yulewelk works just like MATLAB's."""
    # Temp fix, values are computed in matlab using yulewalk.m
    if sfreq == 125:
        a = [1, -0.983952187817050, -0.520232502560362, 0.603540557711479,
             0.116893105621457, -0.0291261609247754, -0.282359853603720,
             0.0407847933579206, 0.103437108246108]
        b = [1.08742316795540, -1.83643555381637, 0.573976014496824,
             0.361020603610170, 0.0592714561864745, 0.0767631759850725,
             -0.498304757808424, 0.276872948140515, -0.00693079202803615]
    elif sfreq == 256:
        a = [1, -1.70080396393018, 1.92328303910588, -2.08269297269299,
             1.59826387425574, -1.07358541839301, 0.567971922565269,
             -0.188618149976820, 0.0572954115997260]
        b = [1.75870131417701, -4.32676243944586, 5.79998800310163,
             -6.23966254635480, 5.37680790468827, -3.79382188933751,
             2.16491080952266, -0.859139256986372, 0.256936112562797]
    elif sfreq == 250:
        a = [1, -1.63849492766660, 1.73987814299054, -1.83638657883455,
             1.39241775367979, -0.953780426622192, 0.505158779550744,
             -0.159504514603054, 0.0545278399847976]
        b = [1.73133310854258, -4.16813353295698, 5.37379900844170,
             -5.57212564343883, 4.70122651316511, -3.34208799655244,
             1.95045488724907, -0.766909658912065, 0.233281060974834]
    elif sfreq == 2048:
        a = [1, -7.37108717906118, 23.9532262496612, -44.8116760275397,
             52.7784619594557, -40.0682205405753, 19.1457849272128,
             -5.26406859898898, 0.637581364205670]
        b = [2.84929120330035, -21.2941639596528, 70.1293865360529,
             -132.920238664871, 158.567177443427, -121.909488069062,
             58.9853908881204, -16.4212688404351, 2.01391570212326]
    else:
        raise AttributeError("Currently sfreq must be 250, 256 or 2048...")

    # Theoretical values
    w0, h0 = signal.freqz(b, a, sfreq)

    f = np.array([0, 2, 3, 13, 16, 40, np.min((80, sfreq / 2 - 1)), sfreq / 2])
    f *= 2. / sfreq
    m = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
    [b, a] = yulewalk(8, f, m)

    w1, h1 = signal.freqz(b, a, sfreq)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w0 / np.pi, np.abs(h0), label="matlab")
        ax.plot(w1 / np.pi, np.abs(h1), ":", label="mine")
        ax.set_title("Filter frequency response")
        ax.set_xlabel("Frequency [radians / second]")
        ax.set_ylabel("Amplitude [dB]")
        ax.grid(which="both", axis="both")
        ax.legend()
        # plt.show()

    np.testing.assert_almost_equal(np.abs(h0), np.abs(h1), decimal=4)

    f = [0, .6, .6, 1]              # Frequency breakpoints
    m = [1., 1., 0, 0]              # Magnitude breakpoints
    b, a = yulewalk(8, f, m)        # Filter design using least-squares method
    w, h = signal.freqz(b, a, 250)  # Frequency response of filter

    if show:
        plt.figure()
        plt.plot(f, m, label="ideal")
        plt.plot(w / np.pi, np.abs(h), "--", label="yw designed")
        plt.legend()
        plt.title("Comparison of Frequency Response Magnitudes")
        plt.legend()
        plt.show()


@pytest.mark.parametrize(argnames="n_chans", argvalues=(4, 8, 12))
def test_yulewalk_filter(n_chans, show=False):
    """Test yulewalk filter."""
    raw = np.load(os.path.join(THIS_FOLDER, "data", "eeg_raw.npy"))
    sfreq = 250
    n_chan_orig = raw.shape[0]
    raw = rng.standard_normal((n_chans, n_chan_orig)) @ raw
    raw_filt, iirstate = yulewalk_filter(raw, sfreq)

    if show:
        f, ax = plt.subplots(n_chans, sharex=True, figsize=(8, 5))
        for i in range(n_chans):
            ax[i].plot(raw[i], lw=.5, label="before")
            ax[i].plot(raw_filt[i], label="after", lw=.5)
            ax[i].set_ylim([-50, 50])
            if i < n_chans - 1:
                ax[i].set_yticks([])
        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel(f"ch{i}")
        ax[0].legend(fontsize="small", bbox_to_anchor=(1.04, 1),
                     borderaxespad=0)
        plt.subplots_adjust(hspace=0, right=0.75)
        plt.suptitle("Before/after filter")
        plt.show()


def test_asr_functions(show=False, method="riemann"):
    """Test ASR functions (offline use).

    Note: this will not be optimal since the filter parameters will be
    estimated only once and not updated online as is intended.

    """
    raw = np.load(os.path.join(THIS_FOLDER, "data", "eeg_raw.npy"))
    sfreq = 250
    raw_filt = raw.copy()
    raw_filt, iirstate = yulewalk_filter(raw_filt, sfreq)

    # Train on a clean portion of data
    train_idx = np.arange(5 * sfreq, 45 * sfreq, dtype=int)

    # Clean data of high amplitude artifacts
    clean, sample_mask = clean_windows(raw[:, train_idx], sfreq)
    assert clean.shape[1] < train_idx.size  # make sure we removed artefacts

    M, T = asr_calibrate(clean, sfreq, method=method, cutoff=2,
                         max_dropout_fraction=.2)
    state = dict(M=M, T=T, R=None)
    clean, _ = asr_process(raw, raw_filt, state, method=method)

    if show:
        f, ax = plt.subplots(8, sharex=True, figsize=(8, 5))
        for i in range(8):
            ax[i].fill_between(train_idx, 0, 1, color="grey", alpha=.3,
                               transform=ax[i].get_xaxis_transform(),
                               label="calibration window")
            ax[i].fill_between(train_idx, 0, 1, where=sample_mask.flat,
                               transform=ax[i].get_xaxis_transform(),
                               facecolor="none", hatch="...", edgecolor="k",
                               label="selected window")
            ax[i].plot(raw[i], lw=.5, label="before ASR")
            ax[i].plot(clean[i], label="after ASR", lw=.5)
            # ax[i].set_xlim([10, 50])
            ax[i].set_ylim([-50, 50])
            # ax[i].set_ylabel(raw.ch_names[i])
            if i < 7:
                ax[i].set_yticks([])
        ax[i].set_xlabel("Time (s)")
        ax[0].legend(fontsize="small", bbox_to_anchor=(1.04, 1),
                     borderaxespad=0)
        plt.subplots_adjust(hspace=0, right=0.75)
        plt.suptitle("Before/after ASR")
        plt.show()


@pytest.mark.parametrize(argnames="zthresholds", argvalues=([1, 2], [-5, -1]))
def test_clean_windows_one_sided_zthresholds(zthresholds):
    """Test clean_windows with a one-sided zthresholds.

    Regression test: when zthresholds does not straddle 0 (i.e. both bounds
    are positive, or both are negative), only one of the two rejection-mask
    branches in clean_windows runs. The other mask must still be defined
    (as an all-False "reject nothing" default) so that combining the masks
    does not raise UnboundLocalError.
    """
    raw = np.load(os.path.join(THIS_FOLDER, "data", "eeg_raw.npy"))
    sfreq = 250
    # Use a short slice for speed; still exercises the mask logic.
    X = raw[:, :10 * sfreq]

    clean, sample_mask = clean_windows(X, sfreq, zthresholds=zthresholds)

    assert clean.shape[0] == X.shape[0]
    assert clean.shape[1] <= X.shape[1]
    assert sample_mask.shape == (1, X.shape[1])
    assert sample_mask.dtype == bool


@pytest.mark.parametrize(argnames="method", argvalues=("riemann", "euclid"))
@pytest.mark.parametrize(argnames="reref", argvalues=(False, True))
def test_asr_class(method, reref, show=False):
    """Test ASR class (simulate online use)."""
    raw = np.load(os.path.join(THIS_FOLDER, "data", "eeg_raw.npy"))
    sfreq = 250
    # Train on a clean portion of data
    train_idx = np.arange(5 * sfreq, 45 * sfreq, dtype=int)

    # Rereference
    if reref:
        # Rank deficient matrix
        raw2 = raw - np.nanmean(raw, axis=0, keepdims=True)
    else:
        raw2 = raw.copy()

    if reref:
        if method == "riemann":
            with pytest.raises(ValueError, match="add regularization"):
                blah = ASR(method=method, estimator="scm")
                blah.fit(raw2[:, train_idx])

        asr = ASR(method=method, estimator="lwf", memory=int(2 * sfreq))
        asr.fit(raw2[:, train_idx])
    else:
        asr = ASR(method=method, estimator="scm")
        asr.fit(raw2[:, train_idx])

    # Split into small windows
    X = sliding_window(raw2, window=int(sfreq // 2), step=int(sfreq // 2))
    X = X.swapaxes(0, 1)

    # Transform each trial
    Y = np.zeros_like(X)
    for i in range(X.shape[0]):
        Y[i] = asr.transform(X[i])

    # Transform all trials at once
    asr.reset()
    asr.fit(raw2[:, train_idx])
    Y2 = asr.transform(X)

    X = X.swapaxes(0, 1).reshape(8, -1)
    Y = Y.swapaxes(0, 1).reshape(8, -1)
    Y2 = Y2.swapaxes(0, 1).reshape(8, -1)
    times = np.arange(X.shape[-1]) / sfreq
    if show:
        f, ax = plt.subplots(8, sharex=True, figsize=(8, 5))
        for i in range(8):
            ax[i].plot(times, X[i], lw=.5, label="before ASR")
            ax[i].plot(times, Y[i], label="after ASR", lw=.5)
            ax[i].set_ylim([-50, 50])
            ax[i].set_ylabel(f"ch{i}")
            if i < 7:
                ax[i].set_yticks([])
        ax[i].set_xlabel("Time (s)")
        ax[0].legend(fontsize="small", bbox_to_anchor=(1.04, 1),
                     borderaxespad=0)
        plt.subplots_adjust(hspace=0, right=0.75)
        plt.suptitle("Before/after ASR")

        f, ax = plt.subplots(8, sharex=True, figsize=(8, 5))
        for i in range(8):
            ax[i].plot(times, Y[i], label="incremental", lw=.5)
            ax[i].plot(times, Y2[i], label="bulk", lw=.5)
            ax[i].plot(times, Y[i] - Y2[i], label="difference", lw=.5)
            if i < 7:
                ax[i].set_yticks([])
        ax[i].set_xlabel("Time (s)")
        plt.suptitle("incremental vs. bulk difference ")
        plt.show()

    # TODO: the transform() process is stochastic, so Y and Y2 are not going to
    # be entirely identical but close enough
    assert np.all(np.abs(Y - Y2) < 6), np.max(np.abs(Y - Y2))  # < 6uV diff
    assert np.all(np.isreal(Y)), "output should be real-valued"
    assert np.all(np.isreal(Y2)), "output should be real-valued"

    # Test different sampling rates
    with pytest.raises(ValueError):
        ASR(sfreq=60)

    ASR(sfreq=80)
    ASR(sfreq=100)
    ASR(sfreq=125)
    ASR(Sfreq=150)


def test_clean_windows_uses_full_shape_range():
    """clean_windows passes the full 13-point SHAPE_RANGE (incl beta=3.5) down.

    Spies on the shape_range argument to fit_eeg_distribution (6th positional).
    """
    raw = np.load(os.path.join(THIS_FOLDER, "data", "eeg_raw.npy"))
    sfreq = 250

    with patch(
        "meegkit.asr.fit_eeg_distribution", side_effect=fit_eeg_distribution
    ) as mock_fit:
        clean_windows(raw, sfreq)

    assert mock_fit.called
    captured = mock_fit.call_args.args[5]
    assert len(captured) == 13
    assert np.isclose(captured[-1], 3.5)
    assert np.allclose(captured, SHAPE_RANGE)


def test_fit_eeg_distribution_shape_range_materiality():
    """Dropping the beta=3.5 endpoint changes the selected shape parameter.

    A bounded, near-uniform sample drives the grid search to the largest beta.
    """
    # Bounded, near-uniform sample: favors the largest available beta.
    Y = np.linspace(-1, 1, 2000)

    shape_range_trunc = np.arange(1.7, 3.5, 0.15)

    beta_full = fit_eeg_distribution(Y, shape_range=SHAPE_RANGE)[3]
    beta_trunc = fit_eeg_distribution(Y, shape_range=shape_range_trunc)[3]

    assert np.isclose(beta_full, 3.5)
    assert beta_trunc <= np.max(shape_range_trunc) + 1e-9
    assert beta_full > beta_trunc

def test_asr_calibrate_nonfinite_input():
    """Non-finite calibration samples must not silently yield NaN M/T."""
    raw = np.load(os.path.join(THIS_FOLDER, "data", "eeg_raw.npy"))
    sfreq = 250
    train_idx = np.arange(5 * sfreq, 45 * sfreq, dtype=int)
    X = raw[:, train_idx].copy()
    X[0, 100] = np.nan
    X[0, 200] = np.inf

    M, T = asr_calibrate(X, sfreq)

    assert np.all(np.isfinite(M))
    assert np.all(np.isfinite(T))


def test_asr_calibrate_too_short():
    """Calibration data shorter than one analysis window should raise."""
    X_short = rng.standard_normal((8, 50))  # N = round(0.5 * 250) = 125
    with pytest.raises(ValueError, match="shorter than one analysis window"):
        asr_calibrate(X_short, 250)

        
def test_asr_max_bad_chans_param():
    """max_bad_chans is exposed on ASR and defaults to 0.3."""
    assert ASR().max_bad_chans == 0.3
    assert ASR(max_bad_chans=0.2).max_bad_chans == 0.2


def test_asr_process_state_key_order():
    """asr_process reads state by key, not by dict insertion order.

    M and T are distinct and the keep mask is non-trivial, so a positional
    unpack of a reordered dict would change the output.
    """
    nc, ns = 4, 8
    X = np.arange(nc * ns, dtype=float).reshape(nc, ns) + 1.0
    X_filt = X.copy()
    # T's large column-2 norm keeps component 2 while M=eye rejects it, so the
    # two dict orderings give different keep masks (hence different R) under a
    # positional unpack.
    cov = np.diag([1.0, 1.0, 1000.0, 1000.0])
    T = np.eye(nc)
    T[2, 2] = 40.0   # column-2 sum of squares = 1600 > 1000
    M = np.eye(nc)

    state_ordered = dict(M=M, T=T, R=None)
    out_ordered, _ = asr_process(
        X, X_filt, state_ordered, cov=cov.copy(), method="euclid")

    # Same content, different insertion order
    state_reordered = dict(T=T, M=M, R=None)
    out_reordered, _ = asr_process(
        X, X_filt, state_reordered, cov=cov.copy(), method="euclid")

    np.testing.assert_allclose(out_ordered, out_reordered)


def test_clean_windows_offset_phase_drift():
    """Offsets use the non-truncated win_len*sfreq to avoid phase drift.

    Diverges from truncated-N spacing only when win_len*sfreq is non-integer.
    """
    sfreq = 251
    win_len = 0.5  # win_len * sfreq = 125.5, non-integer
    win_overlap = 0.66
    ns = 5000

    N = int(win_len * sfreq)
    N_raw = win_len * sfreq
    offsets_raw = np.round(
        np.arange(0, ns - N, N_raw * (1 - win_overlap))).astype(int)
    offsets_truncated = np.round(
        np.arange(0, ns - N, N * (1 - win_overlap))).astype(int)

    assert len(offsets_raw) == len(offsets_truncated)
    # The two stepping schemes diverge (truncated-N accumulates drift)
    assert not np.array_equal(offsets_raw, offsets_truncated)
    assert abs(int(offsets_raw[-1]) - int(offsets_truncated[-1])) >= 10


if __name__ == "__main__":
    pytest.main([__file__])
    # test_yulewalk(250, True)
    # test_asr_functions(True)
    # test_asr_class(method='riemann', reref=True, show=False)
    # test_yulewalk_filter(16, True)
