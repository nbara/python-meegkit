from numpy import *
from numpy.random import permutation
import scipy as Sci
import scipy.linalg

data = random.random((200,50,100))
ref  = random.random((200,3,100))

def tsr(data, ref, shifts = array([0]), weights_data = array([]), weights_ref = array([]), keep = array([]), thresh = 10**-20):
    """docstring for tsr"""
    
    print "tsr", data.shape
    samples_data, channels_data, trials_data = data.shape
    samples_ref,  channels_ref,  trials_ref  = ref.shape
        
    #adjust to make shifts non-negative
    
    offset1 = max(0, -min(shifts))
    idx = r_[ offset1:samples_data ]
    data = data[idx, :, :]
    if weights_data:
        weights_data = weights_data[idx, :, :]
    ref = ref[:samples_ref-offset1, :, :]
    if weights_ref:
        weights_ref = weights_ref[0:-offset1, :, :]
    shifts += offset1
    
    # adjust size of data
    offset2 = max(0, (max(shifts)))
    idx = r_[0:samples_data-offset2]
    data = data[idx,:,:]
    if weights_data:
        weights_data = weights_data[idx,:,:]
    
    # consolidate weights into single weight matrix
    weights = zeros((samples_data, 1, trials_ref))
    if not weights_data and not weights_ref:
        weights[0:samples_data, :, :] = 1
    elif not weights_ref:
        weights[:, :, :] = weights_data[:, :, :]
    elif not weights_data:
        for trial in arange(trials_data):
            wr = multishift(weights_ref[:, :, trial], shifts).min(1)
            weights[:, :, trial] = wr
    else:
        for trial in arange(trials_data):
            wr = multishift(wref[:, :, trial], shifts).min(1)
            wr = min(wr, wx[0:wr.shape[0], :, trial])
            weights[:, :, trial] = wr
    
    weights_data = weights
    weights_ref = zeros((samples_ref, 1, trials_ref))
    weights_ref[idx, :, :] = weights
    weights_ref = weights_ref
    
    # remove weighted means
    data, mean1 = demean(data, weights_data)
    ref = demean(ref, weights_ref)[0]
    
    # equalize power of ref channels, the equalize power of the ref PCs
    print "tsr: data ", data.shape, "weights_ref", weights_ref.shape
    
    ref = normcol(ref, weights_ref)
    print "X01", ref.__class__
    ref = tspca(ref, array([0]), [], 10 ** -6)[0]
    ref = normcol(ref, weights_ref)
    
    #covariances and cros covariance with time-shifted refs
    [cref, twcref] = tscov(ref, shifts, weights_ref)
    [cxref, twcxref] = tsxcov(data, ref, shifts, weights_data)
    
    # regression matrix of x on time-shifted refs
    r = regcov(cxref/twcxref, cref/twcref, keep, thresh)
    
    # TSPCA: clean x by removing regression on time-shifted refs
    denoised_data = zeros((samples_data, channels_data, trials_data))
    for trial in arange(trials_data):
        z = multishift(ref[:, :, trial], shifts) * r
        denoised_data[:, :, trial] = data[0:z.shape[0], :, trial] - z
    
    denoised_data, mean2 = demean(denoised_data, weights_data)
    
    idx = r_[1+offset1:samples_data-offset2]
    mean = mean1 + mean2
    weights = weights_ref
    
    return denoised_data, idx, mean, weights

def tspca(data, shifts = array([0]), keep=array([]), threshold=array([]), weights=array([])):
    """TSPCA"""
    
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
        if sum(weights) == 0: raise Exception('weights are all zero')
        c = tscov(data, shifts, weights)[0]
        
    # PCA matrix
    topcs, eigenvalues = pcarot(c)
    
    # truncate
    if keep:
        topcs = topcs[:, arange(keep)]
        eigenvalues = eigenvalues[arange(keep)]
    
        
    if threshold:
        ii = squeeze(where(eigenvalues/eigenvalues[0] > threshold))
        print "eigenvalues", eigenvalues/eigenvalues[0]
        print "ii", ii
        topcs = topcs[:, ii]
        eigenvalues = eigenvalues[ii]
    
    # apply PCA matrix to time-shifted data 
    print "topcs.shape[1]", topcs.shape[1], topcs.shape
    z = zeros((idx.size, topcs.shape[1], trials))
    
    for trial in arange(trials):
        z[:, :, trial] = dot(squeeze(multishift(data[:, :, trial], shifts)), squeeze(topcs))
        
    return z, idx
        

def multishift(data, shifts, amplitudes = array([])):
    """apply multiple shifts to an array"""
    if min(shifts) > 0: raise Exception('shifts should be non-negative')
        
    shifts = shifts.T
    shifts_length = shifts.size
    
    # array of shift indices
    
    if data.ndim == 3:
        time, channels, trials = data.shape # m, n, o
    else:
        time, channels = data.shape
        trials = 1
    
    N = time - max(shifts)
    shiftarray = ((ones((N, shifts_length), int) * shifts).T + r_[ 0:N ]).T
    
    z = zeros((N, channels * shifts_length, trials))
    
    if amplitudes:
        for trial in arange(trials):
            for channel in arange(channels):
                y = data[:, channel]
                z[:, (channel * shifts_length):(channel * shifts_length + shifts_length), trial] = (y[shiftarray].T * amplitudes).T
    else:
        for trial in arange(trials):
            for channel in arange(channels):
                y = data[:, channel]
                z[:, (channel * shifts_length):(channel * shifts_length + shifts_length), trial] = y[shiftarray]
        
                
    return z
                    

def pcarot(cov, keep=array([])):
    """PCA rotation from covariance
    
    topcs: PCA rotation matrix
    eigenvalues: PCA eigenvalues
    
    cov: covariance matrix
    keep: number of components to keep [default: all]
    """
    
    if not keep: keep = cov.shape[0]
    
    eigenvalues, eigenvector = linalg.eig(cov)
        
    idx = argsort(eigenvalues.real)[::-1] # reverse sort ev order
    eigenvalues = sort(eigenvalues.real)[::-1]
    
    topcs = eigenvector.real[:, idx]
    
    eigenvalues = eigenvalues[arange(keep)]
    topcs = topcs[:, arange(keep)]    
    
    return topcs, eigenvalues


def tscov(data, shifts = array([0]), weights = []):
    """docstring for tscov"""
    if min(shifts) < 0:
        raise Exception('shifts should be non-negative')
        
    nshifts = shifts.size
    
    samples, channels, trials = data.shape
    covariance_matrix = zeros((channels * nshifts, channels * nshifts))
    #print covariance_matrix.shape
    
    if any(weights):
        # weights
        if weights.shape[1] > 1: raise Exception('w should have a single column')
            
        for trial in arange(trials):
            shifted_trial = multishift(data[:, :, trial], shifts)
            trial_weight = weights[arange(shifted_trial.shape[0]), :, trial]
            shifted_trial = (squeeze(shifted_trial).T * squeeze(trial_weight)).T
            covariance_matrix += dot(shifted_trial.T, shifted_trial)
        
        total_weight = sum(w[:])
    else:
        # no weights
        for trial in arange(trials):
            shifted_trial = squeeze(multishift(data[:, :, trial], shifts))
            covariance_matrix += dot(shifted_trial.T, shifted_trial)
            
        total_weight = shifted_trial.shape[0] * trials
        
    return covariance_matrix, total_weight


def fold(data, epochsize):
    """docstring for fold"""
    print "fold", data.__class__
    
    return transpose(reshape(data, (epochsize, data.shape[0]/epochsize, data.shape[1])), (0, 2, 1))


def unfold(data):
    """docstring for unfold"""    
    
    if data.ndim == 3:
        samples, channels, trials = data.shape
        return reshape(transpose(data, (0, 2, 1)), (samples * trials, channels))
    else:
        return data

def demean(data, weights = array([])):
    """docstring for demean"""
    if data.ndim == 3:
        samples, channels, trials = data.shape
    else:
        samples, channels = data.shape
        
    data = unfold(data)
    
    print weights.__class__
    if not any(weights):
        the_mean = mean(data, 0)
        demeaned_data = data - the_mean
    else:
        print "demean weights", weights.shape
        weights = unfold(weights)        
        
        if weights.shape[0] != data.shape[0]:
            raise Exception('data and weights should have same nrows & npages')
        
        if weights.shape[1] == 1:
            the_mean = sum(data * weights) / sum(weights)
        elif weights.shape[1] == channels:
            the_mean = sum(data * weights) / sum(weights)
        else:
            raise Exception('weights should have same number of cols as data, or else 1')
        
        demeaned_data = data - the_mean
    
    demeaned_data = fold(demeaned_data, samples)
    
    return demeaned_data, the_mean


def normcol(data, weights = array([])):
    """docstring for normcol"""
    #print "normcol", data.shape, weights.shape
    
    if data.ndim == 3:
        samples, channels, trials = data.shape
        data = unfold(data)
        if not any(weights):
            # no weights
            normalized_data = fold(normcol(data), samples)
        else:
            if weights.shape[0] != samples:
                raise Exception('weight matrix should have same ncols as data')
            if weights.ndim == 2 and weights.shape[1] == 1:
                weights = tile(weights, (1, samples, trials))
            if weights.shape != weights.shape:
                raise Exception('weight should have same size as data')
            weights = unfold(weights)
            normalized_data = fold(normcol(data, weights), samples)
    else:
        samples, channels = data.shape
        if not weights.any():
            normalized_data = data * ((sum(data ** 2) / samples) ** -0.5)
        else:
            print "normcol", weights.shape[0], data.shape[0]
            if weights.shape[0] != data.shape[0]:
                raise Exception('weight matrix should have same ncols as data')
            if weights.ndim == 2 and weights.shape[1] == 1:
                weights = tile(weights, (1, channels))
            if weights.shape != data.shape:
                raise Exception('weight should have same size as data')
            if weights.shape[1] == 1:
                weights = tile(weights, (1, channels))
            normalized_data = data * (sum((data ** 2) * weights) / sum(weights)) ** -0.5
        
    return normalized_data

def regcov(cxy,cyy,keep=array([]),threshold=array([])):
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
    r = topcs.T * cxy
    
    # projection matrix from regressor PCs
    r = (r * 1/eigenvalues.T)
    
    #projection matrix from regressors
    r = topcs * r
    
    return r

def tsregress(x, y, shifts = array([0]), keep = array([]), threshold = array([]), toobig1 = array([]), toobig2 = array([])):
    """docstring for tsregress"""
    
    # shifts must be non-negative
    mn = min(shifts)
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
        
        for k in arange(nshifts):
            kk = shifts(k)
            idx1 = r_[kk+1:kk+mm]
            idx2 = k + r_[0:y.shape[1]] * nshifts
            z[0:mm, :] = z[0:mm, :] + y[idx1, :] * r[idx2, :]
        
        z = fold(z,Mx)
        z = z[0:-max(shifts), :, :]
    else:
        [m,n] = x.shape
        z = zeros((m-max(shifts), n))
        for k in arange(nshifts):
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
    
    [m,n] = x.shape
    
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
    
    for k in arange(n):
        
        c1 = x.T * x[:, k]                  #correlation with neighbors
        c1 = c1 / c1[k]                     
        c1[k] = 0                           # demote self
        [c1, idx] = sort(c1 ** 2, 0)[::-1]  # sort
        idx = idx[1+skip:nneighbors+skip]   # keep best
        
        # pca neighbors to orthogonalize them
        xx = x[:,idx]
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
        x = demean(x,w)[0]
        
        w[where(abs(x) > toobig1)] = 0
        x = demean(x,w)[0]
        
        w[where(abs(x) > toobig1)] = 0
        x = demean(x,w)[0]
    else:
        w = ones(x.shape)
        
    # apply relative threshold
    if toobig2:
        X = wmean(x ** 2, w)
        X = tile(X, (x.shape[0], 1))
        idx = where(x**2 > (X * toobig2))
        w[idx] = 0
        
    w = fold(w,m)
    
    return w

def sns0(c, nneighbors, skip=0, wc=[]):
    """docstring for sns0"""
    
    if not wc:
        wc = c
    
    n = c.shape[0]
    
    if not nneighbors:
        nneighbors = n - 1
    
    r = zeros(c.size)
    
    # normalize
    d = sqrt(1 / diag(c))
    c = c * d * d.T
    
    for k in arange(n):
        
        c1 = c[:,k] # correlation of channel
        [c1, idx] = sort(c1 ** 2, 0)[::-1] # sort by correlation
        idx = idx[skip+2:skip+1+nneighbors] # keep best
        
        # pca neighbors to orthogonalize them
        c2 = wc[idx,idx]
        [topcs, eigenvalues] = pcarot(c2)
        topcs = topcs * diag(1/sqrt(eigenvalues))
        
        # augment rotation matric to include this channel
        topcs = array([[1, zeros(nneighbors)], [zeros((nneighbors, 1)), topcs]])
        
        # correlation matrix for rotated data
        c3 = topcs.T * wc[c_[k,idx],c_[k,idx]] * topcs
        
        # first row defines projection to clean component k
        c4 = c3[0,1:] * topcs[1:,1:].T
        
        # insert new column into denoising matric
        r[idx,k] = c4
        
    return c4

def tsxcov(x,y,shifts=0,w=[]):
    """docstring for tsxcov"""
    
    shifts = shifts[:]
    nshifts = shifts.size
    
    [mx,nx,ox] = x.size
    [my,ny,oy] = y.size
    c = zeros((nx, ny*nshifts))
    
    if w:
        x = fold(unfold(x) * unfold(w), mx)
        
    # cross covariance
    for k in arange(ox):
        yy = multishift(y[:,:,k], shifts)
        xx = x[1:yy.shape[0],:,k]
        c = c + xx.T * yy
    
    if not w:
        tw = ox * ny * yy.shape[0]
    else:
        w = w[1:yy.shape[0],:,:]
        tw = sum(w[:])
        
    return [c, tw]

def wmean(x,w=[],dim=0):
    """docstring for wmean"""
    
    if not w:
        y = mean(x,dim)
    else:
        if x.shape[0] != w.shape[0]:
            raise Exception("data and weight must have same nrows")
        if w.shape[1] == 1:
            w = tile(w,(1,x.shape(1)))
        if w.shape[1] != x.shape[1]:
            raise Exception("weight must have same ncols as data, or 1")
        y = sum(x * w, dim) / sum(w,dim)
    
    return y


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    