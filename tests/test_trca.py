"""TRCA tests."""
import os

import numpy as np
import pytest
import scipy.io
from meegkit.trca import TRCA
from meegkit.utils.trca import itr, normfit, round_half_up

##########################################################################
# Load data
# -----------------------------------------------------------------------------
path = os.path.join('.', 'tests', 'data', 'trcadata.mat')
mat = scipy.io.loadmat(path)
eeg = mat["eeg"]

n_trials = eeg.shape[0]
n_chans = eeg.shape[1]
n_samples = eeg.shape[2]
n_blocks = eeg.shape[3]

list_freqs = np.concatenate(
    [[x + 8 for x in range(8)],
        [x + 8.2 for x in range(8)],
        [x + 8.4 for x in range(8)],
        [x + 8.6 for x in range(8)],
        [x + 8.8 for x in range(8)]])  # list of stimulus frequencies
n_targets = len(list_freqs)  # The number of stimuli

# Convert dummy Matlab format to (sample, channels, trials) and construct
# vector of labels
eeg = np.reshape(eeg.transpose([2, 1, 3, 0]),
                 (n_samples, n_chans, n_trials * n_blocks))
labels = np.array([x for x in range(n_targets)] * n_blocks)

# We use the filterbank specification described in [2]_.
filterbank = [[[6, 90], [4, 100]],  # passband freqs, stopband freqs (Wp, Ws)
              [[14, 90], [10, 100]],
              [[22, 90], [16, 100]],
              [[30, 90], [24, 100]],
              [[38, 90], [32, 100]]]


@pytest.mark.parametrize('ensemble', [True, False])
@pytest.mark.parametrize('method', ['original', 'riemann'])
@pytest.mark.parametrize('regularization', ['schaefer', 'scm'])
def test_trca(ensemble, method, regularization):
    """Test TRCA."""
    if method == 'original' and regularization == 'schaefer':
        pytest.skip("regularization only used for riemann version")

    len_gaze_s = 0.5  # data length for target identification [s]
    len_delay_s = 0.13  # visual latency being considered in the analysis [s]
    alpha_ci = 0.05   # 100*(1-alpha_ci): confidence interval for accuracy
    sfreq = 250  # sampling rate [Hz]
    len_shift_s = 0.5  # duration for gaze shifting [s]

    # useful variables
    len_gaze_smpl = round_half_up(len_gaze_s * sfreq)  # data length [samples]
    len_delay_smpl = round_half_up(len_delay_s * sfreq)  # visual latency [samples]
    len_sel_s = len_gaze_s + len_shift_s  # selection time [s]
    ci = 100 * (1 - alpha_ci)  # confidence interval

    crop_data = np.arange(len_delay_smpl, len_delay_smpl + len_gaze_smpl)

    ##########################################################################
    # TRCA classification
    # -----------------------------------------------------------------------------
    # Estimate classification performance with a Leave-One-Block-Out
    # cross-validation approach
    trca = TRCA(sfreq, filterbank, ensemble=ensemble, method=method,
                estimator=regularization)
    accs = np.zeros(2)
    itrs = np.zeros(2)
    for i in range(2):

        # Training stage
        traindata = eeg.copy()[crop_data]

        # Select all folds except one for training
        traindata = np.concatenate(
            (traindata[..., :i * n_trials],
             traindata[..., (i + 1) * n_trials:]), 2)
        y_train = np.concatenate(
            (labels[:i * n_trials], labels[(i + 1) * n_trials:]), 0)

        # Construction of the spatial filter and the reference signals
        trca.fit(traindata, y_train)

        # Test stage
        testdata = eeg[crop_data, :, i * n_trials:(i + 1) * n_trials]
        y_test = labels[i * n_trials:(i + 1) * n_trials]
        estimated = trca.predict(testdata)

        # Evaluation of the performance for this fold (accuracy and ITR)
        is_correct = estimated == y_test
        accs[i] = np.mean(is_correct) * 100
        itrs[i] = itr(n_targets, np.mean(is_correct), len_sel_s)
        print(f"Block {i}: accuracy = {accs[i]:.1f}, \tITR = {itrs[i]:.1f}")

    # Mean accuracy and ITR computation
    mu, _, muci, _ = normfit(accs, alpha_ci)
    print(f"Mean accuracy = {mu:.1f}%\t({ci:.0f}% CI: {muci[0]:.1f}-{muci[1]:.1f}%)")  # noqa
    if method != 'riemann' or (regularization == 'scm' and ensemble):
        assert mu > 95
    mu, _, muci, _ = normfit(itrs, alpha_ci)
    print(f"Mean ITR = {mu:.1f}\t({ci:.0f}% CI: {muci[0]:.1f}-{muci[1]:.1f}%)")
    if method != 'riemann' or (regularization == 'scm' and ensemble):
        assert mu > 300


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
    # test_trcacode()
