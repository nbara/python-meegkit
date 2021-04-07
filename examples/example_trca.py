"""
Task-related component analysis (TRCA)-based SSVEP detection
============================================================

Sample code for the task-related component analysis (TRCA)-based steady
-state visual evoked potential (SSVEP) detection method [1]_. The filter
bank analysis [2, 3]_ can also be combined to the TRCA-based algorithm.

Uses meegkit.trca.TRCA()

References:

.. [1] M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung,
   "Enhancing detection of SSVEPs for a high-speed brain speller using
   task-related component analysis", IEEE Trans. Biomed. Eng, 65(1): 104-112,
   2018.
.. [2] X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao, "Filter bank
   canonical correlation analysis for implementing a high-speed SSVEP-based
   brain-computer interface", J. Neural Eng., 12: 046008, 2015.
.. [3] X. Chen, Y. Wang, M. Nakanishi, X. Gao, T. -P. Jung, S. Gao,
   "High-speed spelling with a noninvasive brain-computer interface",
   Proc. Int. Natl. Acad. Sci. U. S. A, 112(44): E6058-6067, 2015.

This code is based on the Matlab implementation from
https://github.com/mnakanishi/TRCA-SSVEP

"""
# Author: Giuseppe Ferraro <giuseppe.ferraro@isae-supaero.fr>
import os
import time

import numpy as np
import scipy.io
from meegkit.trca import TRCA
from meegkit.utils.trca import itr, normfit, round_half_up

t = time.time()

###############################################################################
# Parameters
# -----------------------------------------------------------------------------
len_gaze_s = 0.5  # data length for target identification [s]
len_delay_s = 0.13  # visual latency being considered in the analysis [s]
n_bands = 5  # number of sub-bands in filter bank analysis
is_ensemble = True  # True = ensemble TRCA method; False = TRCA method
alpha_ci = 0.05   # 100*(1-alpha_ci): confidence interval for accuracy
sfreq = 250  # sampling rate [Hz]
len_shift_s = 0.5  # duration for gaze shifting [s]
list_freqs = np.concatenate(
    [[x + 8 for x in range(8)],
     [x + 8.2 for x in range(8)],
     [x + 8.4 for x in range(8)],
     [x + 8.6 for x in range(8)],
     [x + 8.8 for x in range(8)]])  # list of stimulus frequencies
n_targets = len(list_freqs)  # The number of stimuli

# Preparing useful variables (DONT'T need to modify)
len_gaze_smpl = round_half_up(len_gaze_s * sfreq)  # data length [samples]
len_delay_smpl = round_half_up(len_delay_s * sfreq)  # visual latency [samples]
len_sel_s = len_gaze_s + len_shift_s  # selection time [s]
ci = 100 * (1 - alpha_ci)  # confidence interval

# Performing the TRCA-based SSVEP detection algorithm
print('Results of the ensemble TRCA-based method:\n')

###############################################################################
# Load data
# -----------------------------------------------------------------------------
path = os.path.join('..', 'tests', 'data', 'trcadata.mat')
mat = scipy.io.loadmat(path)
eeg = mat["eeg"]

n_trials = eeg.shape[0]
n_chans = eeg.shape[1]
n_samples = eeg.shape[2]
n_blocks = eeg.shape[3]

# Convert dummy Matlab format to (sample, channels, trials) and construct
# vector of labels
eeg = np.reshape(eeg.transpose([2, 1, 3, 0]),
                 (n_samples, n_chans, n_trials * n_blocks))
labels = np.array([x for x in range(n_targets)] * n_blocks)

crop_data = np.arange(len_delay_smpl, len_delay_smpl + len_gaze_smpl)
eeg = eeg[crop_data]

###############################################################################
# TRCA classification
# -----------------------------------------------------------------------------
# Estimate classification performance with a Leave-One-Block-Out
# cross-validation approach.

# We use the filterbank specification described in [2]_.
filterbank = [[(6, 90), (4, 100)],  # passband, stopband freqs [(Wp), (Ws)]
              [(14, 90), (10, 100)],
              [(22, 90), (16, 100)],
              [(30, 90), (24, 100)],
              [(38, 90), (32, 100)],
              [(46, 90), (40, 100)],
              [(54, 90), (48, 100)]]
trca = TRCA(sfreq, filterbank, is_ensemble)

accs = np.zeros(n_blocks)
itrs = np.zeros(n_blocks)
for i in range(n_blocks):

    # Training stage
    traindata = eeg.copy()

    # Select all folds except one for training
    traindata = np.concatenate(
        (traindata[..., :i * n_trials],
         traindata[..., (i + 1) * n_trials:]), 2)
    y_train = np.concatenate(
        (labels[:i * n_trials], labels[(i + 1) * n_trials:]), 0)

    # Construction of the spatial filter and the reference signals
    trca.fit(traindata, y_train)

    # Test stage
    testdata = eeg[..., i * n_trials:(i + 1) * n_trials]
    y_test = labels[i * n_trials:(i + 1) * n_trials]
    estimated = trca.predict(testdata)

    # Evaluation of the performance for this fold (accuracy and ITR)
    is_correct = estimated == y_test
    accs[i] = np.mean(is_correct) * 100
    itrs[i] = itr(n_targets, np.mean(is_correct), len_sel_s)
    print(f"Block {i}: accuracy = {accs[i]:.1f}, \tITR = {itrs[i]:.1f}")

# Mean accuracy and ITR computation
mu, _, muci, _ = normfit(accs, alpha_ci)
print()
print(f"Mean accuracy = {mu:.1f}%\t({ci:.0f}% CI: {muci[0]:.1f}-{muci[1]:.1f}%)")  # noqa

mu, _, muci, _ = normfit(itrs, alpha_ci)
print(f"Mean ITR = {mu:.1f}\t({ci:.0f}% CI: {muci[0]:.1f}-{muci[1]:.1f}%)")
if is_ensemble:
    ensemble = 'ensemble TRCA-based method'
else:
    ensemble = 'TRCA-based method'

print(f"\nElapsed time: {time.time()-t:.1f} seconds")
