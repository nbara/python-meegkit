"""Utility functions."""
from .cca import nt_cca
from .covariances import tscov, tsxcov, cov_lags
from .denoise import (find_outlier_trials, find_outliers,
                      mean_over_trials, pca, regcov, tsregress, wmean, wpwr)
from .matrix import (fold, multishift, theshapeof, unfold, widen_mask,
                     unsqueeze, demean, normcol, shiftnd, shift, relshift)
from .stats import (bootstrap_ci, bootstrap_snr, cronbach, rms, robust_mean,
                    snr_spectrum)
from .viz import plot_montage
