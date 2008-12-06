from numpy import *
import scipy.linalg

#data = random.random((800,157,241))
#ref  = random.random((800,3,241))

def tsr(data, ref, shifts = None, weights_data = None, weights_ref = None, keep = None, thresh = None):
    """
    Time-shift regression.
    
    The basic idea is to project the signal DATA on a basis formed by the
    orthogonalized time-shifted REF, and remove the projection. Supposing REF
    gives a good observation of the noise that contaminates DATA, the noise is
    removed. By allowing time shifts, the algorithm finds the optimal FIR filter
    to apply to REF so as to compensate for any convolutional mismatch
    between DATA and REF.
    
    INPUT
    data:         data to denoise (time * channels * trials)
    ref:          reference (time * channels * trials)
    shifts:       array of shifts to apply to ref (default: [0])
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
    
    initial_samples = data.shape[0]
    
    if shifts == None:
        shifts       = array([0])
    if not weights_data:
        weights_data = array([])
    if not weights_ref:
        weights_ref  = array([])
    if not keep:
        keep         = array([])
    if not thresh:
        thresh       = 10**-12
        
    #adjust to make shifts non-negative
    
    offset1 = max(0, -min(shifts))
    idx = r_[offset1:data.shape[0]]
    data = data[idx, :, :]
    
    if weights_data: 
        weights_data = weights_data[idx, :, :]
    
    ref = ref[:ref.shape[0]-offset1, :, :]
    
    if weights_ref:
        weights_ref = weights_ref[0:-offset1, :, :]
    
    shifts += offset1
    
    # adjust size of data array
    offset2 = max(0, max(shifts))
    
    idx = r_[0:data.shape[0]-offset2]
    data = data[idx, :, :]
    
    if weights_data:
        weights_data = weights_data[idx, :, :]
    
    samples_data, channels_data, trials_data = data.shape
    samples_ref,  channels_ref,  trials_ref  = ref.shape
    
    # consolidate weights into single weight matrix
    weights = zeros((samples_data, 1, trials_ref))
    
    if not weights_data and not weights_ref:
        weights[arange(samples_data), :, :] = 1
    elif not weights_ref:
        weights[:, :, :] = weights_data[:, :, :]
    elif not weights_data:
        for trial in xrange(trials_data):
            wr = multishift(weights_ref[:, :, trial], shifts).min(1)
            weights[:, :, trial] = wr
    else:
        for trial in xrange(trials_data):
            wr = multishift(weights_ref[:, :, trial], shifts).min(1)
            wr = (wr, wx[0:wr.shape[0], :, trial]).min()
            weights[:, :, trial] = wr
    
    weights_data = weights
    weights_ref = zeros((samples_ref, 1, trials_ref))
    weights_ref[idx, :, :] = weights[idx, :, :]
    
    # remove weighted means
    data, mean1 = demean(data, weights_data)
    ref         = demean(ref, weights_ref)[0]
    
    # equalize power of ref channels, the equalize power of the ref PCs
    ref = normcol(ref, weights_ref)
    ref = tspca(ref)[0]
    ref = normcol(ref, weights_ref)
    
    #covariances and cross-covariance with time-shifted refs
    cref, twcref = tscov(ref, shifts, weights_ref)
    cxref, twcxref = tsxcov(data, ref, shifts, weights_data)
    
    # regression matrix of x on time-shifted refs
    r = regcov(cxref/twcxref, cref/twcref, keep, thresh)
    
    # TSPCA: clean x by removing regression on time-shifted refs
    denoised_data = zeros((samples_data, channels_data, trials_data))
    for trial in xrange(trials_data):
        z = dot(squeeze(multishift(ref[:, :, trial], shifts)), r)
        denoised_data[:, :, trial] = data[0:z.shape[0], :, trial] - z
    
    denoised_data, mean2 = demean(denoised_data, weights_data)
    
    idx = r_[offset1:initial_samples-offset2]
    mean_total = mean1 + mean2
    weights = weights_ref
    
    return denoised_data, idx, mean_total, weights


def tspca(data, shifts = None, keep = None, threshold = None, weights = None):
    """
    Time-shift PCA.
    
    INPUT
    data:      data matrix
    shifts:    array of shifts to apply
    keep:      number of components shifted regressor PCs to keep (default: all)
    threshold: discard PCs with eigenvalues below this (default: 10 ** -6)
    weights:   ignore samples with absolute value above this
    
    OUTPUT
    principal_components: PCs
    idx: data[idx] maps to principal_components
    """
    
    if not shifts:    
        shifts    = array([0])
    if not keep:      
        keep      = array([])
    if not threshold: 
        threshold = 10 ** -6
    if not weights:   
        weights   = array([])
    
    samples, channels, trials = data.shape
    
    # offset of z relative to data
    offset = max(0, -min(shifts, 0))
    shifts += offset
    idx = offset + (arange(samples) - max([shifts]))
    
    # remove mean
    data = unfold(data)
    data = demean(data, weights)[0]
    data = fold(data, samples)
    
    # covariance
    if not any(weights):
        c = tscov(data, shifts)[0]
    else:
        if sum(weights) == 0: raise ValueError, "Weights are all zero."
        c = tscov(data, shifts, weights)[0]
    
    # PCA matrix
    topcs, eigenvalues = pcarot(c)
    
    # truncate
    if keep:
        topcs = topcs[:, arange(keep)]
        eigenvalues = eigenvalues[arange(keep)]
    
    if threshold:
        ii = eigenvalues/eigenvalues[0] > threshold
        topcs = topcs[:, ii]
        eigenvalues = eigenvalues[ii]
    
    # apply PCA matrix to time-shifted data
    principal_components = zeros((idx.size, topcs.shape[1], trials))
    
    for trial in xrange(trials):
        principal_components[:, :, trial] = dot(squeeze(multishift(data[:, :, trial], shifts)), squeeze(topcs))
    
    return principal_components, idx


def multishift(data, shifts, amplitudes = None):
    """
    Apply multiple shifts to an array.
    
    INPUT
    data:       array to shift
    shifts:     array of shifts (must be nonnegative)
    amplitudes: array of amplitudes
    
    OUTPUT
    z: result
    """    
    if not amplitudes: 
        amplitudes = array([])
    
    if shifts.min() > 0: 
        raise ValueError, "Shifts should be non-negative."
    
    shifts = shifts.T
    shifts_length = shifts.size
    
    # array of shift indices
    
    if data.ndim == 3:
        time, channels, trials = data.shape # m, n, o
    elif data.ndim == 2:
        time, channels = data.shape
        trials = 1
    elif data.ndim == 1:
        time = data.shape
        channels, trials = 1, 1
    
    N = time - max(shifts)
    shiftarray = ((ones((N, shifts_length), int) * shifts).T + arange(N)).T
    
    z = zeros((N, channels * shifts_length, trials))
    
    if amplitudes:
        for trial in xrange(trials):
            for channel in xrange(channels):
                y = data[:, channel]
                z[:, (channel * shifts_length):(channel * shifts_length + shifts_length), trial] = (y[shiftarray].T * amplitudes).T
    else:
        for trial in xrange(trials):
            for channel in xrange(channels):
                y = data[:, channel]
                z[:, (channel * shifts_length):(channel * shifts_length + shifts_length), trial] = y[shiftarray]
                
    return z


def pcarot(cov, keep = None):
    """
    PCA rotation from covariance.
    
    INPUT
    cov:  covariance matrix
    keep: number of components to keep [default: all]
    
    OUTPUT
    topcs:       PCA rotation matrix
    eigenvalues: PCA eigenvalues
    """
    
    if not keep: 
        keep = cov.shape[0] # keep all components
        
    eigenvalues, eigenvector = linalg.eig(cov)
    
    idx = argsort(eigenvalues.real)[::-1] # reverse sort ev order
    eigenvalues = sort(eigenvalues.real)[::-1]
    
    topcs = eigenvector.real[:, idx]
    
    topcs = topcs[:, arange(keep)]
    eigenvalues = eigenvalues[arange(keep)]
    
    return topcs, eigenvalues


def tscov(data, shifts = None, weights = None):
    """
    Time shift covariance.
    
    This function calculates, for each pair [DATA[i], DATA[j]] of columns of 
    DATA, the cross-covariance matrix between the time-shifted versions of 
    DATA[i]. Shifts are taken from array SHIFTS. Weights are taken from WEIGHTS.
    
    DATA can be 1D, 2D or 3D.  WEIGHTS is 1D (if DATA is 1D or 2D) or 
    2D (if DATA is 3D).
    
    Output is a 2D matrix with dimensions (ncols(X)*nshifts)^2.
    This matrix is made up of a DATA.shape[1]^2 matrix of submatrices
    of dimensions nshifts^2.
    
    The weights are not shifted.
    
    INPUT
    data: data
    shifts: array of time shifts (must be non-negative)
    weights: weights
    
    OUTPUT
    covariance_matrix: covariance matrix
    total_weight: total weight (covariance_matrix/total_weight is normalized 
    covariance)
    """
    
    if shifts == None:  
        shifts  = array([0])
    if not any(weights): 
        weights = array([])
    
    if shifts.min() < 0: raise ValueError, "Shifts should be non-negative."
    
    shifts = shifts.T
    nshifts = shifts.size
    
    if data.ndim == 3:
        samples, channels, trials = data.shape
    else:
        samples, channels = data.shape
        trials = 1
    
    covariance_matrix = zeros((channels * nshifts, channels * nshifts))
    
    if any(weights):
        # weights
        if weights.shape[1] > 1: 
            raise ValueError, "Weights array should have a single column"
        
        for trial in xrange(trials):
            if data.ndim == 3:
                shifted_trial = multishift(data[:, :, trial], shifts)
            else:
                shifted_trial = multishift(data[:, trial], shifts)
            
            trial_weight = weights[arange(shifted_trial.shape[0]), :, trial]
            shifted_trial = (squeeze(shifted_trial).T * squeeze(trial_weight)).T
            covariance_matrix += dot(shifted_trial.T, shifted_trial)
        
        total_weight = sum(weights[:])
    else:
        # no weights
        for trial in xrange(trials):
            if data.ndim == 3:
                shifted_trial = squeeze(multishift(data[:, :, trial], shifts))
            else:
                shifted_trial = multishift(data[:, trial], shifts)
            
            covariance_matrix += dot(shifted_trial.T, shifted_trial)
        
        total_weight = shifted_trial.shape[0] * trials
    
    return covariance_matrix, total_weight


def fold(data, epochsize):
    '''fold'''
    return transpose(reshape(data, (epochsize, data.shape[0]/epochsize, data.shape[1])), (0, 2, 1))


def unfold(data):
    '''unfold'''
    try:
        samples, channels, trials = data.shape
    except:
        samples, channels = data.shape
        trials = 1
    
    if trials > 1:
        return reshape(transpose(data, (0, 2, 1)), (samples * trials, channels))
    else:
        return data


def demean(data, weights = None):
    """Remove weighted mean over columns."""
    
    if data.ndim == 3:
        samples, channels, trials = data.shape
    else:
        samples, channels = data.shape
    
    data = unfold(data)
    
    if any(weights):
        weights = unfold(weights)
        
        if weights.shape[0] != data.shape[0]:
            raise ValueError, "Data and weights arrays should have same number of rows and pages."
        
        if weights.shape[1] == 1 or weights.shape[1] == channels:
            the_mean = sum(data * weights) / sum(weights)
        else:
            raise ValueError, "Weight array should have either the same number of columns as data array, or 1 column."
        
        demeaned_data = data - the_mean
    else:
        the_mean = mean(data, 0)
        demeaned_data = data - the_mean
    
    demeaned_data = fold(demeaned_data, samples)
    
    return demeaned_data, the_mean


def normcol(data, weights = None):
    """
    Normalize each column so its weighted msq is 1.
    
    If DATA is 3D, pages are concatenated vertically before calculating the 
    norm.
    
    Weight should be either a column vector, or a matrix (2D or 3D) of same size
    as data.
    
    INPUT
    data: data to normalize
    weights: weight
    
    OUTPUT
    normalized_data: normalized data
    
    """
        
    if data.ndim == 3:
        samples, channels, trials = data.shape
        data = unfold(data)
        if not weights.any():
            # no weights
            normalized_data = fold(normcol(data), samples)
        else:
            if weights.shape[0] != samples: 
                raise ValueError, "Weight array should have same number of columns as data array."
            
            if weights.ndim == 2 and weights.shape[1] == 1: 
                weights = tile(weights, (1, samples, trials))
            
            if weights.shape != weights.shape: 
                raise ValueError, "Weight array should have be same shape as data array"
            
            weights = unfold(weights)
            
            normalized_data = fold(normcol(data, weights), samples)
    else:
        samples, channels = data.shape
        if not weights.any():
            normalized_data = data * ((sum(data ** 2) / samples) ** -0.5)
        else:
            if weights.shape[0] != data.shape[0]: 
                raise ValueError, "Weight array should have same number of columns as data array."
            
            if weights.ndim == 2 and weights.shape[1] == 1: 
                weights = tile(weights, (1, channels))
            
            if weights.shape != data.shape: 
                raise ValueError, "Weight array should have be same shape as data array"
            
            if weights.shape[1] == 1: 
                weights = tile(weights, (1, channels))
            
            normalized_data = data * (sum((data ** 2) * weights) / sum(weights)) ** -0.5
    
    return normalized_data


def regcov(cxy, cyy, keep=array([]), threshold=array([])):
    """regression matrix from cross covariance"""
    
    # PCA of regressor
    [topcs, eigenvalues] = pcarot(cyy)
    
    # discard negligible regressor PCs
    if keep:
        keep = max(keep, topcs.shape[1])
        topcs = topcs[:, 0:keep]
        eigenvalues = eigenvalues[0:keep]
    
    if threshold:
        idx = where(eigenvalues/max(eigenvalues) > threshold)
        topcs = topcs[:, idx]
        eigenvalues = eigenvalues[idx]
    
    # cross-covariance between data and regressor PCs
    cxy = cxy.T
    r = dot(topcs.T, cxy)
    
    # projection matrix from regressor PCs
    r = (r.T * 1/eigenvalues).T
    
    #projection matrix from regressors
    r = dot(squeeze(topcs), squeeze(r))
    
    return r


def tsxcov(x, y, shifts = None, w = array([])):
    """
    Calculate cross-covariance of X and time-shifted Y.
    
    This function calculates, for each pair of columns (Xi,Yj) of X and Y, the
    scalar products between Xi and time-shifted versions of Yj. 
    Shifts are taken from array SHIFTS. 
    
    The weights are applied to X.
    
    X can be 1D, 2D or 3D.  W is 1D (if X is 1D or 2D) or 2D (if X is 3D).
    
    Output is a 2D matrix with dimensions ncols(X)*(ncols(Y)*nshifts).
    
    INPUT
    x, y: data to cross correlate
    shifts: array of time shifts (must be non-negative)
    w: weights
    
    OUTPUT
    c: cross-covariance matrix
    tw: total weight
    """
    
    if shifts == None: 
        shifts = array([0])
        
    nshifts = shifts.size
    
    mx, nx, ox = x.shape
    my, ny, oy = y.shape
    c = zeros((nx, ny*nshifts))
    
    if any(w): 
        x = fold(unfold(x) * unfold(w), mx)
        
    # cross covariance
    for trial in xrange(ox):
        yy = squeeze(multishift(y[:, :, trial], shifts))
        xx = squeeze(x[0:yy.shape[0], :, trial])
        
        c += dot(xx.T, yy)
        
    if not any(w):
        tw = ox * ny * yy.shape[0]
    else:
        w = w[0:yy.shape[0], :, :]
        tw = sum(w[:])
        
    return c, tw


def tsregress(x, y, shifts = array([0]), keep = array([]), threshold = array([]), toobig1 = array([]), toobig2 = array([])):
    """docstring for tsregress"""
    
    # shifts must be non-negative
    mn = shifts.min()
    if mn < 0:
        shifts = shifts - mn
        x = x[-mn+1:, :, :]
        y = y[-mn+1:, :, :]
    
    nshifts = shifts.size
    
    # flag outliers in x and y
    if toobig1 or toobig2:
        xw = find_outliers(x, toobig1, toobig2)
        yw = find_outliers(y, toobig1, toobig2)
    else:
        xw = []
        yw = []
    
    if x.ndim == 3:
        [Mx, Nx, Ox] = x.shape
        [My, Ny, Oy] = y.shape
        x = unfold(x)
        y = unfold(y)
        [x, xmn] = demean(x, xw)
        [y, ymn] = demean(y, yw)
        x = fold(x, Mx)
        y = fold(y, My)
    else:
        [x, xmn] = demean(x, xw)
        [y, ymn] = demean(y, yw)
    
    # covariance of y
    [cyy, totalweight] = tscov(y, shifts.T, yw)
    cyy = cyy / totalweight
    
    # cross-covariance of x and y
    [cxy, totalweight] = tscov2(x, y, shifts.T, xw, yw)
    cxy = cxy / totalweight
    
    # regression matrix
    r = regcov(cxy, cyy, keep, threshold)
    
    # regression
    if x.ndim == 3:
        x = unfold(x)
        y = unfold(y)
        
        [m, n, o] = x.shape
        mm = m - max(shifts)
        z = zeros(x.shape)
        
        for k in xrange(nshifts):
            kk = shifts(k)
            idx1 = r_[kk+1:kk+mm]
            idx2 = k + r_[0:y.shape[1]] * nshifts
            z[0:mm, :] = z[0:mm, :] + y[idx1, :] * r[idx2, :]
        
        z = fold(z, Mx)
        z = z[0:-max(shifts), :, :]
    else:
        m, n = x.shape
        z = zeros((m-max(shifts), n))
        for k in xrange(nshifts):
            kk = shifts(k)
            idx1 = r_[kk+1:kk+z.shape[0]]
            idx2 = k + r_[0:y.shape[1]] * nshifts
            z = z + y[idx1, :] * r[idx2, :]
    
    offset = max(0, -mn)
    idx = r_[offset+1:offset+z.shape[0]]
    
    return [z, idx]


def sns1(x, nneighbors, skip):
    """docstring for sns1"""
    if x.ndim > 2:
        raise Exception("SNS1 works only with 2D matrices")
    
    m, n = x.shape
    
    if not nneighbors:
        nneighbors = n-1
    
    if not skip:
        skip = 0
    
    mn = mean(x)
    x = (x - mn) # remove mean
    N = sqrt(sum(x ** 2))
    NN = 1 / N
    NN[where(isnan(NN))] = 0
    x = (x * NN) # normalize
    
    y = zeros(x.shape)
    
    for k in xrange(n):
        
        c1 = x.T * x[:, k]                  #correlation with neighbors
        c1 = c1 / c1[k]
        c1[k] = 0                           # demote self
        [c1, idx] = sort(c1 ** 2, 0)[::-1]  # sort
        idx = idx[1+skip:nneighbors+skip]   # keep best
        
        # pca neighbors to orthogonalize them
        xx = x[:, idx]
        c2 = xx.T * xx
        [topcs, eigenvalues] = pcarot(c2)
        topcs = topcs * diag(1/sqrt(eigenvalues))
        
        y[:,k] = tsregress(x[:,k], xx * topcs)
        
        #if mod(k,1000) == 0:
            #[k 100 * sum(y[:,0:k] ** 2) / sum(x[:, 0:k] ** 2)]
    
    y = (y * N)
    
    return y


def find_outliers(x, toobig1, toobig2 = []):
    """docstring for find_outliers"""
    
    [m, n, o] = x.shape
    x = unfold(x)
    
    # remove mean
    x = demean(x)[0]
    
    # apply absolute threshold
    w = ones(x.shape)
    if toobig1:
        w[where(abs(x) > toobig1)] = 0
        x = demean(x, w)[0]
        
        w[where(abs(x) > toobig1)] = 0
        x = demean(x, w)[0]
        
        w[where(abs(x) > toobig1)] = 0
        x = demean(x, w)[0]
    else:
        w = ones(x.shape)
    
    # apply relative threshold
    if toobig2:
        X = wmean(x ** 2, w)
        X = tile(X, (x.shape[0], 1))
        idx = where(x**2 > (X * toobig2))
        w[idx] = 0
    
    w = fold(w, m)
    
    return w


def sns0(c, nneighbors, skip=0, wc=[]):
    """docstring for sns0"""
    
    if not any(wc):
        wc = c
    
    n = c.shape[0]
    
    if not nneighbors:
        nneighbors = n - 1
    
    r = zeros(c.shape)
    
    # normalize
    d = sqrt(1 / diag(c))
    c = c * d * d.T
    
    for k in xrange(n):
        
        c1 = c[:, k] # correlation of channel k with all other channels
        corr_sq = c1 ** 2
        idx = argsort(corr_sq, 0)[::-1] # sort by correlation, descending order
        c1 = c1[idx]
        idx = idx[skip+2:skip+1+nneighbors+1] # keep best
        
        # pca neighbors to orthogonalize them
        c2 = wc[idx, :][:, idx]
        [topcs, eigenvalues] = pcarot(c2)
        topcs = topcs * diag(1/sqrt(eigenvalues))
        
        # augment rotation matric to include this channel
        stack1 = hstack((1, zeros(topcs.shape[0])))
        stack2 = hstack((zeros((topcs.shape[0], 1)), topcs))
        topcs = vstack((stack1, stack2))
        
        # correlation matrix for rotated data
        #c3 = topcs.T * wc[c_[k,idx],c_[k,idx]] * topcs
        idx.shape = (idx.shape[0], 1)
        c3 = topcs.T * wc[vstack((k, idx)), vstack((k, idx))] * topcs
        
        # first row defines projection to clean component k
        c4 = dot(c3[0, 1:], topcs[1:, 1:].T)
        c4.shape = (c4.shape[0], 1)
        
        # insert new column into denoising matrix
        r[idx, k] = c4
    
    return r


def wmean(x, w=[], dim=0):
    """docstring for wmean"""
    
    if not w:
        y = mean(x, dim)
    else:
        if x.shape[0] != w.shape[0]:
            raise Exception("data and weight must have same nrows")
        if w.shape[1] == 1:
            w = tile(w, (1, x.shape(1)))
        if w.shape[1] != x.shape[1]:
            raise Exception("weight must have same ncols as data, or 1")
        y = sum(x * w, dim) / sum(w, dim)
    
    return y


def sns(data, nneighbors = 0, skip = 0, w = array([])):
    """docstring for sns"""
    if not nneighbors:
        nneighbors = x.shape[1]-1
    
    m, n, o = data.shape
    data = unfold(data)
    
    data, mn0 = demean(data)
    c, nc = tscov(data)
    
    if w:
        w = unfold(w)
        data, mn1 = demean(data, w)
        wc, nwc = tscov(data, [], w)
        r = sns0(c, nneighbors, skip, wc)
    else:
        mn1 = 0
        w = ones((n, o))
        r = sns0(c, nneighbors, skip, c)
    
    y = dot(squeeze(data), r)
    y = fold(y, m)
    
    mn = mn0 + mn1
    
    return y


def dss1(data, weights = None, keep1 = None, keep2 = None):
    """docstring for dss1"""
    
    if not any(weights): weights = array([])
    if not keep1: keep1 = array([])
    if not keep2: keep2 = 10.0 ** -12
    
    #m, n, o = data.shape
    data, data_mean = demean(data, weights) # remove weighted mean
    
    # weighted mean over trials (--> bias function for DSS)
    xx, ww = mean_over_trials(data, weights)
    ww = ww.min(1)
    
    # covariance of raw and biased data
    c0, nc0 = tscov(data, None, weights)
    c1, nc1 = tscov(xx, None, ww)
    c1 = linalg.lstsq(c1, o)[0]
    
    todss, fromdss, ratio, pwr = dss0(c0, c1, keep1, keep2)
    
    return todss, fromdss, ratio, pwr


def dss0(c1, c2, keep1, keep2):
    """docstring for dss0"""
    
    # SANITY CHECKS GO HERE
    
    # derive PCA and whitening matrix from unbiased covariance
    topcs1, evs1 = pcarot(c1)
    if keep1:
        topcs1 = topcs1[:, arange(keep1)]
        evs1 = evs1[arange(keep1)]
    
    if keep2:
        idx = where(evs1/max(evs1) > keep2)
        topcs1 = topcs[:, idx]
        evs1 = evs1[idx]
        
    # apply whitening and PCA matrices to the biased covariance
    # (== covariance of bias whitened data)
    N = diag(sqrt(1/evs1))
    c3 = N.T * topcs1.T * c2 * topcs1 * N
    
    # derive the dss matrix
    topcs2, evs2 = pcarot(c3)
    todss = topcs1 * N * topcs2
    fromdss = linalg.pinv(todss)
    
    # dss to data projection matrix
    cxy = c1 * todss # covariance between unbiased data and selected DSS component
    
    # estimate power per DSS component
    pwr = zeros((todss.shape[1], 1))
    
    for k in xrange(todss.shape[1]):
        to_component = todss[:, k] * fromdss[k, :]
        cc = to_component.T * c1 * to_component
        cc = diag(cc)
        pwr[k] = sum(cc**2)
    
    ratio = diag(todss.T * c2 * todss) / diag(todss.T * c1 * todss)
    
    return todss, fromdss, ratio, pwr


def mean_over_trials(x, w):
    """docstring for mean_over_trials"""
    
    m, n, o = x.shape
    
    if not any(w):
        y = mean(x, 2)
        tw = ones((m, n, 1)) * o
    else:
        mw, nw, ow = w.shape
        if mw != m: raise "!"
        if ow != o: raise "!"
        
        x = unfold(x)
        w = unfold(w)
        
        if nw == n:
            x = x * w
            x = fold(x, m)
            w = fold(w, m)
            y = sum(x, 3) / sum(w, 3)
        elif nw == 1:
            x = x * w
            x = fold(x, m)
            w = fold(w, m)
            y = sum(x, 3) * 1/sum(w, 3)
        
        tw = sum(w, 3)
    
    return y, tw
    
            
            
            




