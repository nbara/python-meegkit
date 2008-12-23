from numpy import *
import scipy.linalg
from denoise import *

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
        idx = argsort(c1**2, 0)[::-1] # sort by correlation, descending order
        c1 = c1[idx]
        idx = idx[skip:skip+nneighbors] # keep best
        #print "c1", c1.shape
        #print "idx", idx
        
        # pca neighbors to orthogonalize them
        c2 = wc[idx, :][:, idx]
        [topcs, eigenvalues] = pcarot(c2)
        topcs = dot(topcs, diag(1/sqrt(eigenvalues)))
        #print "c2", c2.shape
        
        # augment rotation matrix to include this channel
        stack1 = hstack((1, zeros(topcs.shape[0])))
        stack2 = hstack((zeros((topcs.shape[0], 1)), topcs))
        topcs = vstack((stack1, stack2))
        
        # correlation matrix for rotated data
        #c3 = topcs.T * wc[hstack((k,idx)), hstack((k,idx))] * topcs
        c3 = dot(dot(topcs.T, wc[hstack((k, idx)), :][:, hstack((k, idx))]), topcs)
        #print "c3", c3.shape
        
        # first row defines projection to clean component k
        c4 = dot(c3[0, 1:], topcs[1:, 1:].T)
        c4.shape = (c4.shape[0], 1)
        #print "c4", c4
        
        # insert new column into denoising matrix
        r[idx, k] = squeeze(c4)
    
    return r


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


