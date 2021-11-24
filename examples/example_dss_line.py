"""
dss_line() example (ZapLine)
============================

Find a spatial filter to get rid of line noise [1]_.

Uses meegkit.dss_line().

References
----------
.. [1] de Cheveigné, A. (2019). ZapLine: A simple and effective method to
    remove power line artifacts [Preprint]. https://doi.org/10.1101/782029

"""
import matplotlib.pyplot as plt
from meegkit import dss
from meegkit.utils.testing import create_line_data
from scipy import signal

###############################################################################
# Line noise removal
# =============================================================================

###############################################################################
# Remove line noise with dss_line()
# -----------------------------------------------------------------------------
# We first generate some noisy data to work with
sr = 250
fline = 50
nsamples = 10000
nchans = 10
data = create_line_data(n_samples=3 * nsamples, n_chans=nchans,
                        n_trials=1, fline=fline / sr, SNR=2)[0]
data = data[..., 0]  # only take first trial

# 2D case, n_outputs == 1
out, _ = dss.dss_line(data, fline, sr, nkeep=1)

###############################################################################
# Plot before/after
f, ax = plt.subplots(1, 2, sharey=True)
f, Pxx = signal.welch(data, sr, nperseg=1024, axis=0, return_onesided=True)
ax[0].semilogy(f, Pxx)
f, Pxx = signal.welch(out, sr, nperseg=1024, axis=0, return_onesided=True)
ax[1].semilogy(f, Pxx)
ax[0].set_xlabel('frequency [Hz]')
ax[1].set_xlabel('frequency [Hz]')
ax[0].set_ylabel('PSD [V**2/Hz]')
ax[0].set_title('before')
ax[1].set_title('after')
plt.show()


###############################################################################
# Remove line noise with dss_line_iter()
# -----------------------------------------------------------------------------
# We first load some noisy data to work with
