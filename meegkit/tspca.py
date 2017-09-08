import numpy as np
from scipy import linalg

from .utils import (demean, fold, multishift, normcol, pcarot, regcov, tscov,
                    tsxcov, unfold)


def tsr(data, ref, shifts=None, weights_data=None, weights_ref=None, keep=None,
        thresh=None):
    """Time-shift regression.

    The basic idea is to project the signal DATA on a basis formed by the
    orthogonalized time-shifted REF, and remove the projection. Supposing REF
    gives a good observation of the noise that contaminates DATA, the noise is
    removed. By allowing time shifts, the algorithm finds the optimal FIR
    filter to apply to REF so as to compensate for any convolutional mismatch
    between DATA and REF.

    INPUT
    data:         data to denoise (time * channels * trials)
    ref:          reference (time * channels * trials)
    shifts:       np.array of shifts to apply to ref (default: [0])
    weights_data: weights to apply to data (time * 1 * trials);
    weights_ref:  weights to apply to ref (time * 1 * trials);
    keep:         number of shifted-ref PCs to retain (default: all)
    thresh:       ignore shifted-ref PCs smaller than thresh (default: 10.^-12)

    OUTPUT
    denoised_data: denoised data
    idx:           data[idx] is aligned with denoised_data
    mean:          channel means (removed by tsr)
    weights:       weights applied by tsr
    """
    if shifts is None:
        shifts = np.array([0])
    if not weights_data:
        weights_data = np.array([])
    if not weights_ref:
        weights_ref = np.array([])
    if not keep:
        keep = np.array([])
    if not thresh:
        thresh = 10**-12

    # adjust to make shifts non-negative
    initial_samples = data.shape[0]

    offset1 = max(0, -min(shifts))
    idx = np.arange(offset1, data.shape[0])
    data = data[idx, :, :]

    if weights_data:
        weights_data = weights_data[idx, :, :]

    ref = ref[:ref.shape[0] - offset1, :, :]

    if weights_ref:
        weights_ref = weights_ref[0:-offset1, :, :]

    shifts += offset1  # shifts are now positive

    # adjust size of data np.array
    offset2 = max(0, max(shifts))

    idx = np.arange(data.shape[0]) - offset2
    idx = idx[idx >= 0]
    data = data[idx, :, :]

    if weights_data:
        weights_data = weights_data[idx, :, :]

    samples_data, channels_data, trials_data = data.shape
    # print "data.shape", data.shape
    samples_ref,  channels_ref,  trials_ref = ref.shape

    #    1/0

    # consolidate weights into single weight matrix
    weights = np.zeros((samples_data, 1, trials_ref))

    if not weights_data and not weights_ref:
        weights[np.arange(samples_data), :, :] = 1
    elif not weights_ref:
        weights[:, :, :] = weights_data[:, :, :]
    elif not weights_data:
        for trial in np.xrange(trials_data):
            wr = multishift(weights_ref[:, :, trial], shifts).min(1)
            weights[:, :, trial] = wr
    else:
        for trial in np.xrange(trials_data):
            wr = multishift(weights_ref[:, :, trial], shifts).min(1)
            wr = (wr, wx[0:wr.shape[0], :, trial]).min()
            weights[:, :, trial] = wr

    weights_data = weights
    weights_ref = np.zeros((samples_ref, 1, trials_ref))
    weights_ref[idx, :, :] = weights

    # remove weighted means
    data, mean1 = demean(data, weights_data)
    # print "pre demean", ref[0:10,0,0]
    ref = demean(ref, weights_ref)[0]

    # equalize power of ref channels, the equalize power of the ref PCs
    # print "pre", ref[0:10,0,0]
    ref = normcol(ref, weights_ref)
    # print "normcol", ref[0:10,0,0]
    ref = tspca(ref)[0]
    # print "tspca", ref[0:10,0,0]
    ref = normcol(ref, weights_ref)
    # print "normcol", ref[0:10,0,0]

    # covariances and cross-covariance with time-shifted refs
    cref, twcref = tscov(ref, shifts, weights_ref)
    cxref, twcxref = tsxcov(data, ref, shifts, weights_data)

    # regression matrix of x on time-shifted refs
    regression = regcov(cxref / twcxref, cref / twcref, keep, thresh)

    # TSPCA: clean x by removing regression on time-shifted refs
    denoised_data = np.zeros((samples_data, channels_data, trials_data))
    for trial in np.xrange(trials_data):
        z = np.dot(np.squeeze(multishift(ref[:, :, trial], shifts)),
                   regression)
        denoised_data[:, :, trial] = data[np.arange(z.shape[0]), :, trial] - z

    denoised_data, mean2 = demean(denoised_data, weights_data)

    # idx = r_[offset1:initial_samples-offset2]
    idx = np.arange(offset1, initial_samples - offset2)
    mean_total = mean1 + mean2
    weights = weights_ref

    return denoised_data, idx, mean_total, weights


def tspca(data, shifts=None, keep=None, threshold=None, weights=None):
    """Time-shift PCA.

    INPUT
    data:      data matrix
    shifts:    np.array of shifts to apply
    keep:      number of components shifted regressor PCs to keep (default: all)
    threshold: discard PCs with eigenvalues below this (default: 10 ** -6)
    weights:   ignore samples with absolute value above this

    OUTPUT
    principal_components: PCs
    idx: data[idx] maps to principal_components
    """
    if not shifts:
        shifts = np.array([0])
    if not keep:
        keep = np.array([])
    if not threshold:
        threshold = 10 ** -6
    if not weights:
        weights = np.array([])

    samples, channels, trials = data.shape

    # offset of z relative to data
    offset = np.max(0, -np.min(shifts, 0))
    shifts += offset
    idx = offset + (np.arange(samples) - np.max([shifts]))

    # remove mean
    data = unfold(data)
    data = demean(data, weights)[0]
    data = fold(data, samples)

    # covariance
    if not any(weights):
        c = tscov(data, shifts)[0]
    else:
        if np.sum(weights) == 0:
            raise ValueError('Weights are all zero.')
        c = tscov(data, shifts, weights)[0]

    # PCA matrix
    topcs, eigenvalues = pcarot(c)

    # truncate
    if keep:
        topcs = topcs[:, np.arange(keep)]
        eigenvalues = eigenvalues[np.arange(keep)]

    if threshold:
        ii = eigenvalues / eigenvalues[0] > threshold
        topcs = topcs[:, ii]
        eigenvalues = eigenvalues[ii]

    # apply PCA matrix to time-shifted data
    principal_components = np.zeros((idx.size, topcs.shape[1], trials))

    for trial in np.xrange(trials):
        principal_components[:, :, trial] = np.dot(
            np.squeeze(multishift(data[:, :, trial], shifts)),
            np.squeeze(topcs))

    return principal_components, idx
