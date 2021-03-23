"""
Task-related component analysis (TRCA)-based SSVEP detection
============================================================

Sample code for the task-related component analysis (TRCA)-based steady
-state visual evoked potential (SSVEP) detection method [1]_. The filter
bank analysis [2]_ can also be combined to the TRCA-based algorithm.

Uses meegkit.trca

References
----------
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
# Author: Giuseppe Ferraro <giuseppe.ferraro@isae.supaero.fr>
import os.path as op
import time

import numpy as np
import scipy.io
import scipy.stats as scp
from meegkit.trca import trca, train_trca, test_trca
from meegkit.utils.trca import itrFunc, normfit, round_half_up

t = time.time()

# Data length for target identification [s]
len_gaze_s = 0.5

# Visual latency being considered in the analysis [s]
len_delay_s = 0.13

# The number of sub-bands in filter bank analysis
num_fbs = 5

# True corresponds to the ensemble TRCA-based method and False to The TRCA-based method
is_ensemble = True

# 100*(1-alpha_ci): confidence intervals for accuracy estimation
alpha_ci = 0.05

## Fixed parameter (Modify according to the experimental setting)

# Sampling rate [Hz]
fs = 250

# Duration for gaze shifting [s]
len_shift_s = 0.5

# List of stimulus frequencies
list_freqs = np.concatenate(
    [[x+8 for x in range(8)],
     [x+8.2 for x in range(8)],
     [x+8.4 for x in range(8)],
     [x+8.6 for x in range(8)],
     [x+8.8 for x in range(8)]]
)

# The number of stimuli
num_targs = len(list_freqs)

## Preparing useful variables (DONT'T need to modify)

# Data length [samples]
len_gaze_smpl = round_half_up(len_gaze_s*fs)

# Visual latency [samples]
len_delay_smpl = round_half_up(len_delay_s*fs)

# Selection time [s]
len_sel_s = len_gaze_s + len_shift_s

# Confidence interval
ci = 100*(1-alpha_ci)

## Performing the TRCA-based SSVEP detection algorithm
print('Results of the ensemble TRCA-based method.\n')

# Preparing data
path = os.path.join('..', 'tests', 'data', 'trcadata.mat')
mat = scipy.io.loadmat(path)
eeg = mat["eeg"]

block_len = eeg.shape[0]
num_chans  = eeg.shape[1]
num_sample = eeg.shape[2]
num_blocks = eeg.shape[3]

# Convert dummy Matlab format to MNE format (trials, channels, sample) and construct vector of labels
eeg_tmp = np.zeros((block_len*num_blocks,num_chans,num_sample))
for blk in range(num_blocks):
    eeg_tmp[blk*block_len:(blk+1)*block_len]=eeg[:,:,:,blk]
eeg = eeg_tmp
del eeg_tmp
labels = np.array([x for x in range(0,num_targs)]*num_blocks, dtype=np.uint8)

segment_data = range(len_delay_smpl,len_delay_smpl+len_gaze_smpl)
eeg = eeg[:, :, segment_data]

accs = np.zeros(num_blocks)
itrs = np.zeros(num_blocks)

# Estimate classification performance
for loocv_i in range(num_blocks):

    # Training stage
    traindata = eeg.copy()

    # Select all the folds except one for the training
    traindata = np.concatenate((traindata[:loocv_i*block_len, :, :], traindata[(loocv_i+1)*block_len:, :, :]),0)
    y_train = np.concatenate((labels[:loocv_i*block_len], labels[(loocv_i+1)*block_len:]),0)

    # Construction of the spatial filter and the reference signals
    model = train_trca(traindata, y_train, fs, num_fbs)

    # Test stage
    testdata = eeg[loocv_i*block_len:(loocv_i+1)*block_len, :, :]
    y_test = labels[loocv_i*block_len:(loocv_i+1)*block_len]


    estimated = test_trca(testdata, model, is_ensemble)
    # Evaluation of the performance for this fold (accuracy and ITR)
    is_correct = [1  if int(estimated[i])==y_test[i] else 0 for i in range(len(y_test))]

    accs[loocv_i] = np.mean(is_correct)*100
    itrs[loocv_i] = itrFunc(num_targs, np.mean(is_correct), len_sel_s)
    print("Trial ",str(loocv_i), ": Accuracy = ", str(accs[loocv_i]), "ITR = ",str(itrs[loocv_i]))


# Mean accuracy and ITR computation
mu, _, muci, _ = normfit(accs, alpha_ci)
print("\nMean accuracy = ",mu,"## ",ci, "## CI: ",muci[0]," - ", muci[1]," %)\n")

mu, _, muci, _ = normfit(itrs, alpha_ci)
print("Mean ITR = ",mu,"## ",ci, "## CI: ",muci[0]," - ", muci[1]," %)\n")
if is_ensemble:
    ensemble='ensemble TRCA-based method,'
else:
    ensemble='not ensemble TRCA-based method,'
print("For", ensemble, "the elapsed time is", time.time()-t, "seconds\n")
