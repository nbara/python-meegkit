"""LOF test."""
import os

import numpy as np
import pytest
import scipy.io as sio

from meegkit.lof import LOF

rng = np.random.default_rng(10)

# Data files
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) # data folder of MEEGKIT


@pytest.mark.parametrize(argnames="n_neighbors", argvalues=(8, 20, 40, 2048))
def test_lof(n_neighbors, show=False):
    mat = sio.loadmat(os.path.join(THIS_FOLDER, "data", "lofdata.mat"))
    X = mat["X"]
    lof = LOF(n_neighbors)
    bad_channel_indices = lof.predict(X)
    print(bad_channel_indices)

@pytest.mark.parametrize(argnames="metric",
                         argvalues=("euclidean", "nan_euclidean",
                                    "cosine", "cityblock", "manhattan"))
def test_lof2(metric, show=False):
    mat = sio.loadmat(os.path.join(THIS_FOLDER, "data", "lofdata.mat"))
    X = mat["X"]
    lof = LOF(20, metric)
    bad_channel_indices = lof.predict(X)
    print(bad_channel_indices)

if __name__ == "__main__":
    pytest.main([__file__])
    #test_lof(20, True)
    #test_lof(metric='euclidean')
