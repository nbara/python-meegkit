"""
Remove line noise with ZapLine
==============================

Find a spatial filter to get rid of line noise [1]_.

Uses meegkit.dss_line().

References
----------
.. [1] de Cheveigné, A. (2019). ZapLine: A simple and effective method to
    remove power line artifacts [Preprint]. https://doi.org/10.1101/782029

"""
# Authors: Maciej Szul <maciej.szul@isc.cnrs.fr>
#          Nicolas Barascud <nicolas.barascud@gmail.com>
import os

import matplotlib.pyplot as plt
import numpy as np
from meegkit import dss
from meegkit.utils import create_line_data, unfold
from scipy import signal

###############################################################################
# Line noise removal
# =============================================================================

###############################################################################
# Remove line noise with dss_line()
# -----------------------------------------------------------------------------
# We first generate some noisy data to work with
sfreq = 250
fline = 50
nsamples = 10000
nchans = 10
data = create_line_data(n_samples=3 * nsamples, n_chans=nchans,
                        n_trials=1, fline=fline / sfreq, SNR=2)[0]
data = data[..., 0]  # only take first trial

# Apply dss_line (ZapLine)
out, _ = dss.dss_line(data, fline, sfreq, nkeep=1)

###############################################################################
# Plot before/after
f, ax = plt.subplots(1, 2, sharey=True)
f, Pxx = signal.welch(data, sfreq, nperseg=500, axis=0, return_onesided=True)
ax[0].semilogy(f, Pxx)
f, Pxx = signal.welch(out, sfreq, nperseg=500, axis=0, return_onesided=True)
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
data = np.load(os.path.join('..', 'tests', 'data', 'dss_line_data.npy'))
fline = 50
sfreq = 200
print(data.shape)  # n_samples, n_chans, n_trials

# Apply dss_line(), removing only one component
out1, _ = dss.dss_line(data, fline, sfreq, nremove=1, nfft=400)

###############################################################################
# Now try dss_line_iter(). This applies dss_line() repeatedly until the
# artifact is gone
out2, iterations = dss.dss_line_iter(data, fline, sfreq, nfft=400)
print(f'Removed {iterations} components')

###############################################################################
# Plot results with dss_line() vs. dss_line_iter()
f, ax = plt.subplots(1, 2, sharey=True)
f, Pxx = signal.welch(unfold(out1), sfreq, nperseg=200, axis=0,
                      return_onesided=True)
ax[0].semilogy(f, Pxx, lw=.5)
f, Pxx = signal.welch(unfold(out2), sfreq, nperseg=200, axis=0,
                      return_onesided=True)
ax[1].semilogy(f, Pxx, lw=.5)
ax[0].set_xlabel('frequency [Hz]')
ax[1].set_xlabel('frequency [Hz]')
ax[0].set_ylabel('PSD [V**2/Hz]')
ax[0].set_title('dss_line')
ax[1].set_title('dss_line_iter')
plt.tight_layout()
plt.show()
