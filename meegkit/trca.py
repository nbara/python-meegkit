"""Task-Related Component Analysis (TRCA)."""
# Author: Giuseppe Ferraro <giuseppe.ferraro@isae.supaero.fr>
import numpy as np
import scipy.linalg as linalg

from .utils.trca import filterbank
from .utils import theshapeof


def trca(eeg):
    """Task-related component analysis (TRCA).

    This script was written based on the reference paper [1]_.

    Parameters
    ----------
    eeg : array, shape=(n_samples, n_chans[, n_trials])
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
    n_samples, n_chans, n_trials = theshapeof(eeg)

    S = np.zeros((n_chans, n_chans))
    for trial_i in range(n_trials - 1):
        x1 = np.squeeze(eeg[..., trial_i])

        # Mean centering for the selected trial
        x1 -= np.mean(x1, 0)

        # Select a second trial that is different
        for trial_j in range(trial_i + 1, n_trials):
            x2 = np.squeeze(eeg[..., trial_j])

            # Mean centering for the selected trial
            x2 -= np.mean(x2, 0)

            # Compute empirical covariance between the two selected trials and
            # sum it
            S = S + x1.T @ x2 + x2.T @ x1

    # Reshape to have all the data as a sequence
    UX = np.zeros((n_chans, n_samples * n_trials))
    for trial in range(n_trials):
        UX[:, trial * n_samples:(trial + 1) * n_samples] = eeg[..., trial].T

    # Mean centering
    UX -= np.mean(UX, 1)[:, None]

    # Compute empirical variance of all data (to be bounded)
    Q = np.dot(UX, UX.T)

    # Compute eigenvalues and vectors
    lambdas, W = linalg.eig(S, Q, left=True, right=False)

    # Select the eigenvector corresponding to the biggest eigenvalue
    W_best = W[:, np.argmax(lambdas)]

    return W_best


def train_trca(eeg, y_train, fs, num_fbs):
    """Training stage of the TRCA-based SSVEP detection.

    Parameters
    ----------
    eeg : array, shape=(n_samples, n_chans[, n_trials])
        Training data
    y_train : array, shape=(trials,)
        True label corresponding to each trial of the data array.
    fs : int
        Sampling frequency of the data.
    num_fb : int
        Number of sub-bands considered for the filterbank analysis

    Returns
    -------
    model: dict
        Fitted model containing:

        - traindata : array, shape=(n_sub-bands, n_chans, n_trials)
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
    n_samples, n_chans, n_trials = theshapeof(eeg)
    n_classes = np.unique(y_train)

    trains = np.zeros((len(n_classes), num_fbs, n_samples, n_chans))

    W = np.zeros((num_fbs, len(n_classes), n_chans))

    for class_i in n_classes:
        # Select data with a specific label
        eeg_tmp = eeg[..., y_train == class_i]
        for fb_i in range(num_fbs):
            # Filter the signal with fb_i
            eeg_tmp = filterbank(eeg_tmp, fs, fb_i)
            if (eeg_tmp.ndim == 3):
                # Compute mean of the signal across the trials
                trains[class_i, fb_i] = np.mean(eeg_tmp, -1)
            else:
                trains[class_i, fb_i] = eeg_tmp
            # Find the spatial filter for the corresponding filtered signal and
            # label
            w_best = trca(eeg_tmp)
            W[fb_i, class_i, :] = w_best  # Store the spatial filter

    model = {'trains': trains,
             'W': W,
             'num_fbs': num_fbs,
             'fs': fs,
             'num_targs': n_classes}
    return model


def test_trca(eeg, model, is_ensemble):
    """Test phase of the TRCA-based SSVEP detection.

    Parameters
    ----------
    eeg: array, shape=(n_samples, n_chans[, n_trials])
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
    # Alpha coefficients for the fusion of filterbank analysis
    fb_coefs = [(x + 1)**(-1.25) + 0.25 for x in range(model["num_fbs"])]
    n_samples, n_chans, n_trials = theshapeof(eeg)

    r = np.zeros((model["num_fbs"], len(model["num_targs"])))
    results = np.zeros((n_trials), 'int')  # To store predictions

    for trial in range(n_trials):
        test_tmp = eeg[..., trial]  # Pick a trial to be analysed
        for fb_i in range(model["num_fbs"]):

            # Filterbank on testdata
            testdata = filterbank(test_tmp, model["fs"], fb_i)

            for class_i in model["num_targs"]:
                # Retrieve reference signal for clss_i (shape: (# of channel, #
                # of sample))
                traindata = np.squeeze(model["trains"][class_i, fb_i])
                if is_ensemble:
                    # Shape of (# of channel, # of class)
                    w = np.squeeze(model["W"][fb_i]).T
                else:
                    # Shape of (# of channel)
                    w = np.squeeze(model["W"][fb_i, class_i])

                # Compute 2D correlation of spatially filtered test data with
                # ref
                r_tmp = np.corrcoef((testdata @ w).flatten(),
                                    (traindata @ w).flatten())
                r[fb_i, class_i] = r_tmp[0, 1]

        rho = np.dot(fb_coefs, r)  # Fusion for the filterbank analysis

        tau = np.argmax(rho)  # Retrieving the index of the max
        results[trial] = int(tau)

    return results
