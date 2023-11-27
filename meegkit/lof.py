"""Local Outlier Factor (LOF)."""
# Authors: Velu Prabhakar Kumaravel <vkumaravel@fbk.eu>
# License: BSD-3-Clause

import logging

from sklearn.neighbors import LocalOutlierFactor


class LOF:
    """Local Outlier Factor.

    Local Outlier Factor (LOF) is an automatic, density-based outlier detection
    algorithm based on [1]_ and [2]_.

    Parameters
    ----------
    n_neighbours : int
        Number of neighbours defining the local neighbourhood.
    metric: str in {'euclidean', 'nan_euclidean', 'cosine',
                    'cityblock', 'manhattan'}
        Metric to use for distance computation. Default is “euclidean”
    threshold : float
        Threshold to define outliers. Theoretical threshold ranges anywhere
        between 1.0 and any integer. Default: 1.5

    Notes
    -----
    It is recommended to perform a CV (e.g., 10-fold) on training set to
    calibrate this parameter for the given M/EEG dataset.

    See [2]_ for details.

    References
    ----------
    .. [1] Breunig M, Kriegel HP, Ng RT, Sander J.
        2000. LOF: identifying density-based local outliers.
        SIGMOD Rec. 29, 2, 93-104. https://doi.org/10.1145/335191.335388
    .. [2] Kumaravel VP, Buiatti M, Parise E, Farella E.
        2022. Adaptable and Robust EEG Bad Channel Detection Using
        Local Outlier Factor (LOF). Sensors (Basel). 2022 Sep 27;22(19):7314.
        doi: 10.3390/s22197314. PMID: 36236413; PMCID: PMC9571252.

    """

    def __init__(self, n_neighbors=20,  metric="euclidean",
                 threshold=1.5, **kwargs):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.threshold = threshold

    def predict(self, X):
        """Detect bad channels using Local Outlier Factor algorithm.

        Parameters
        ----------
        X : array, shape=(n_channels, n_samples)
            The data X should have been high-pass filtered.

        Returns
        -------
        bad_channel_indices : Detected bad channel indices.

        """
        if X.ndim == 3:  # in case the input data is epoched
            logging.warning("Expected input data with shape "
                            "(n_channels, n_samples)")
            return []

        if self.n_neighbors >= X.shape[0]:
            logging.warning("Number of neighbours cannot be greater than the "
                            "number of channels")
            return []

        if self.threshold < 1.0:
            logging.warning("Invalid threshold. Try a positive integer >= 1.0")
            return []

        clf = LocalOutlierFactor(self.n_neighbors)
        logging.debug("[LOF] Predicting bad channels")
        clf.fit_predict(X)
        lof_scores = clf.negative_outlier_factor_
        bad_channel_indices = -lof_scores >= self.threshold

        return bad_channel_indices
