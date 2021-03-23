"""Task-Related Component Analysis (TRCA)."""
# Author: Giuseppe Ferraro <giuseppe.ferraro@isae.supaero.fr>
import numpy as np
import scipy.linalg as linalg

from .utils.trca import filterbank


def trca(eeg):
    """Task-related component analysis (TRCA).

    This script was written based on the reference paper [1]_.

    Parameters
    ----------
    eeg : array, shape=(n_trials, n_chans, n_samples)
        Training data.

    Returns
    -------
    W : array, shape=(n_chans,)
        Weight coefficients for electrodes which can be used as a spatial
        filter.

    References
    ----------
    .. [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
       "Enhancing detection of SSVEPs for a high-speed brain speller using
       task-related component analysis", IEEE Trans. Biomed. Eng,
       65(1):104-112, 2018.

    """
    num_chans = eeg.shape[1]
    num_smpls = eeg.shape[2]

    if(eeg.ndim == 3):
        num_trials = eeg.shape[0]

    elif(eeg.ndim == 2):  # For testdata
        num_trials = 1

    S = np.zeros((num_chans, num_chans))
    for trial_i in range(num_trials - 1):
        x1 = np.squeeze(eeg[trial_i, :, :])
        if x1.ndim > 1:
            # Mean centering for the selected trial
            x1 -= (np.mean(x1, 1) * np.ones((x1.shape[0], x1.shape[1])).T).T
        else:
            x1 -= np.mean(x1)

        # Select a second trial that is different
        for trial_j in range(trial_i + 1, num_trials):
            x2 = np.squeeze(eeg[trial_j, :, :])
            if x2.ndim > 1:
                # Mean centering for the selected trial
                x2 = x2 - (np.mean(x2, 1) *
                           np.ones((x2.shape[0], x2.shape[1])).T).T
            else:
                x2 = x2 - np.mean(x2)

            # Compute empirical covariance betwwen the two selected trials and
            # sum it
            S = S + np.dot(x1, x2.T) + np.dot(x2, x1.T)

    # Reshape to have all the data as a sequence
    UX = np.zeros((num_chans, num_smpls * num_trials))
    for trial in range(num_trials):
        UX[:, trial * num_smpls:(trial + 1) * num_smpls] = eeg[trial, :, :]

    # Mean centering
    UX = UX - (np.mean(UX, 1) * np.ones((UX.shape[0], UX.shape[1])).T).T
    # Compute empirical variance of all data (to be bounded)
    Q = np.dot(UX, UX.T)

    # Compute eigenvalues and vectors
    lambdas, W = linalg.eig(S, Q, left=True, right=False)
    # Select the eigenvector corresponding to the biggest eigenvalue
    W_best = W[:, np.argmax(lambdas)]

    return W_best


def train_trca(eeg, y_train, fs, num_fbs):
    """Training stage of the TRCA-based SSVEP detection [1].

    Parameters
    ----------
    eeg: np.array, shape (trials, channels, samples)
        Training data
    y_train: list or np.array, shape (trials)
        True label corresponding to each trial of the data
        array.
    fs: int
        Sampling frequency of the data.
    num_fb: int
        Number of sub-bands considered for the filterbank analysis

    Returns
    -------
    model: dict
        Fitted model containing:

        - traindata : array, shape=(n_trials, n_sub-bands, n_chans)
            Reference (training) data decomposed into sub-band components by
            the filter bank analysis.
        - y_train : array, shape=(n_trials)
            Labels associated with the train data.
        - W : array, shape=()
            Weight coefficients for electrodes which can be used as a spatial
            filter.
        - num_fbs : int
            Number of sub-bands
        - fs : float
            Sampling rate.
        - num_targs : int
            Number of targets

    """
    num_chans = eeg.shape[1]
    num_smpls = eeg.shape[2]

    num_class = np.unique(y_train)

    trains = np.zeros((len(num_class), num_fbs, num_chans, num_smpls))

    W = np.zeros((num_fbs, len(num_class), num_chans))

    for class_i in num_class:
        eeg_tmp = eeg[y_train == class_i]  # Select data with a specific label
        for fb_i in range(num_fbs):
            # Filter the signal with fb_i
            eeg_tmp = filterbank(eeg_tmp, fs, fb_i)
            if(eeg_tmp.ndim == 3):
                # Compute mean of the signal across the trials
                trains[class_i, fb_i, :, :] = np.mean(eeg_tmp, 0)
            else:
                trains[class_i, fb_i, :, :] = eeg_tmp
            # Find the spatial filter for the corresponding filtered signal and
            # label
            w_best = trca(eeg_tmp)
            W[fb_i, class_i, :] = w_best  # Store the spatial filter

    model = {'trains': trains,
             'W': W,
             'num_fbs': num_fbs,
             'fs': fs,
             'num_targs': num_class}
    return model


def test_trca(eeg, model, is_ensemble):
    """Test phase of the TRCA-based SSVEP detection.

    Parameters
    ----------
    eeg: array, shape=(n_trials, n_chans, n_samples)
        Test data.
    model: dict
        Fitted model to be used in testing phase.
    is_ensemble: bool
        Perform the ensemble TRCA analysis or not.

    Returns
    -------
    results: np.array, shape (trials)
        The target estimated by the method.

    """
    fb_coefs = [(x + 1)**(-1.25) + 0.25 for x in range(model["num_fbs"])
                ]  # Alpha coefficients for the fusion of filterbank analysis
    testdata_len = len(eeg)

    r = np.zeros((model["num_fbs"], len(model["num_targs"])))
    results = np.zeros((testdata_len), 'int')  # To store predictions

    for trial in range(testdata_len):
        test_tmp = eeg[trial, :, :]  # Pick a trial to be analysed
        for fb_i in range(model["num_fbs"]):

            # Filterbank on testdata
            testdata = filterbank(test_tmp, model["fs"], fb_i)

            for class_i in model["num_targs"]:
                # Retrieve reference signal for clss_i (shape: (# of channel, #
                # of sample))
                traindata = np.squeeze(model["trains"][class_i, fb_i, :, :])
                if is_ensemble:
                    # Shape of (# of channel, # of class)
                    w = np.squeeze(model["W"][fb_i, :, :]).T
                else:
                    # Shape of (# of channel)
                    w = np.squeeze(model["W"][fb_i, class_i, :])

                # Compute 2D correlation of spatially filtered test data with
                # ref
                r_tmp = np.corrcoef(
                    np.dot(
                        testdata.T, w).flatten(), np.dot(
                        traindata.T, w).flatten())
                r[fb_i, class_i] = r_tmp[0, 1]

        rho = np.dot(fb_coefs, r)  # Fusion for the filterbank analysis

        tau = np.argmax(rho)  # Retrieving the index of the max
        results[trial] = int(tau)

    return results
