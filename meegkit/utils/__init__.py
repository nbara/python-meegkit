from .cca import nt_cca
from .matrix import theshapeof, unsqueeze, fold, unfold
from .stats import (bootstrap_ci, bootstrap_snr, cronbach, rms, robust_mean,
                    snr_spectrum)
from .utils import (demean, find_outlier_trials, fold, mean_over_trials,
                    multishift, normcol, pca, regcov, rms, tscov, tsregress,
                    tsxcov, unfold, wmean, wpwr)
from .viz import plot_montage
