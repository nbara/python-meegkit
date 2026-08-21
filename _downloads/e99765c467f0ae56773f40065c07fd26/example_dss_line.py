"""
Remove line noise with ZapLine
==============================

Find a spatial filter to get rid of line noise [1]_.

Uses meegkit.dss_line().

The key diagnostic is whether the narrow spectral peak at the line frequency is
reduced while the surrounding spectrum remains broadly similar.

References
----------
.. [1] de Cheveigné, A. (2019). ZapLine: A simple and effective method to remove
   power line artifacts. NeuroImage, 116356.
   https://doi.org/10.1016/j.neuroimage.2019.116356

"""
# Authors: Maciej Szul <maciej.szul@isc.cnrs.fr>
#          Nicolas Barascud <nicolas.barascud@gmail.com>
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from meegkit import dss
from meegkit.utils import create_line_data, unfold

###############################################################################
# Line noise removal
# =============================================================================

###############################################################################
# Remove line noise with dss_line()
# -----------------------------------------------------------------------------
# We first generate some noisy data to work with. The seeded RNG is intentional:
# it keeps the rendered example visually stable across documentation rebuilds.
rng = np.random.default_rng(42)
sfreq = 250
fline = 50
nsamples = 10000
nchans = 10
data = create_line_data(n_samples=3 * nsamples, n_chans=nchans,
                        n_trials=1, fline=fline / sfreq, SNR=2, rng=rng)[0]
data = data[..., 0]  # only take first trial

# Apply dss_line (ZapLine)
out, _ = dss.dss_line(data, fline, sfreq, nkeep=1)
freq_before, psd_before = signal.welch(
   data, sfreq, nperseg=500, axis=0, return_onesided=True)
freq_after, psd_after = signal.welch(
   out, sfreq, nperseg=500, axis=0, return_onesided=True)
line_bin_before = np.argmin(np.abs(freq_before - fline))
line_bin_after = np.argmin(np.abs(freq_after - fline))
mean_before = float(np.mean(psd_before[line_bin_before]))
mean_after = float(np.mean(psd_after[line_bin_after]))

###############################################################################
# Plot before/after
f, ax = plt.subplots(1, 2, sharey=True)
ax[0].semilogy(freq_before, psd_before)
ax[1].semilogy(freq_after, psd_after)
ax[0].set_xlabel("frequency [Hz]")
ax[1].set_xlabel("frequency [Hz]")
ax[0].set_ylabel("PSD [V**2/Hz]")
ax[0].set_title("before")
ax[1].set_title("after")
print(f"Mean line-bin power before ZapLine: {mean_before:.6f}")
print(f"Mean line-bin power after ZapLine:  {mean_after:.6f}")
plt.show()


###############################################################################
# Remove line noise with dss_line_iter()
# -----------------------------------------------------------------------------
# We first load some noisy data to work with
data = np.load(os.path.join("..", "tests", "data", "dss_line_data.npy"))
fline = 50
sfreq = 200
print(data.shape)  # n_samples, n_chans, n_trials

# Apply dss_line(), removing only one component
out1, _ = dss.dss_line(data, fline, sfreq, nfft=400, nremove=1)

###############################################################################
# Now try dss_line_iter(). This applies dss_line() repeatedly until the
# artifact is gone
out2, iterations = dss.dss_line_iter(data, fline, sfreq, nfft=400, show=True)
print(f"Removed {iterations} components")

###############################################################################
# Plot results with dss_line() vs. dss_line_iter()
f, ax = plt.subplots(1, 2, sharey=True)
f, Pxx = signal.welch(unfold(out1), sfreq, nperseg=200, axis=0,
                      return_onesided=True)
ax[0].semilogy(f, Pxx, lw=.5)
f, Pxx = signal.welch(unfold(out2), sfreq, nperseg=200, axis=0,
                      return_onesided=True)
ax[1].semilogy(f, Pxx, lw=.5)
ax[0].set_xlabel("frequency [Hz]")
ax[1].set_xlabel("frequency [Hz]")
ax[0].set_ylabel("PSD [V**2/Hz]")
ax[0].set_title("dss_line")
ax[1].set_title("dss_line_iter")
plt.tight_layout()
print("Interpretation: iterative removal should help when the line artifact")
print("spans more than one dominant component.")
plt.show()
