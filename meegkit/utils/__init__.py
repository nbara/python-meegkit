"""Utility functions."""
from .base import mldivide, mrdivide
from .covariances import (block_covariance, convmtx, cov_lags,
                          nonlinear_eigenspace, pca, regcov, tscov, tsxcov)
from .denoise import (demean, find_outlier_samples, find_outlier_trials,
                      mean_over_trials, wpwr)
from .matrix import (fold, multishift, multismooth, normcol, relshift, shift,
                     shiftnd, theshapeof, unfold, unsqueeze, widen_mask)
from .sig import (AuditoryFilterbank, GammatoneFilterbank, erb2hz, erbspace,
                  gaussfilt, hilbert_envelope, hz2erb, slope_sum, smooth,
                  spectral_envelope, teager_kaiser)
from .stats import (bootstrap_ci, bootstrap_snr, cronbach, rms, robust_mean,
                    rolling_corr, snr_spectrum)
