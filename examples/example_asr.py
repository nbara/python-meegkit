"""
ASR example
===========

This example demonstrates a full ASR workflow on short EEG data:

1. Calibrate ASR on a mostly clean segment.
2. Apply ASR in 1-second windows.
3. Compare raw and cleaned traces and quantify amplitude reduction.

Uses meegkit.ASR().

References
----------
.. [1] Mullen, T., Kothe, C., Chi, Y., Ojeda, A., Kerth, T., Makeig, S.,
   Jung, T. P., & Cauwenberghs, G. (2015). Real-time neuroimaging and
   cognitive monitoring using wearable dry EEG. IEEE Transactions on
   Biomedical Engineering, 62(11), 2553-2567.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window

# THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
raw = np.load(os.path.join("..", "tests", "data", "eeg_raw.npy"))
sfreq = 250

###############################################################################
# Calibration and processing
# -----------------------------------------------------------------------------

# Train on a clean portion of data
asr = ASR(method="euclid")
train_idx = np.arange(0 * sfreq, 30 * sfreq, dtype=int)
_, sample_mask = asr.fit(raw[:, train_idx])

# Apply filter using sliding (non-overlapping) windows
X = sliding_window(raw, window=int(sfreq), step=int(sfreq))
Y = np.zeros_like(X)
for i in range(X.shape[1]):
    Y[:, i, :] = asr.transform(X[:, i, :])

raw = X.reshape(8, -1)  # reshape to (n_chans, n_times)
clean = Y.reshape(8, -1)

# A simple quality metric: root-mean-square attenuation per channel.
rms_before = np.sqrt(np.mean(raw ** 2, axis=1))
rms_after = np.sqrt(np.mean(clean ** 2, axis=1))
rms_ratio = rms_after / np.maximum(rms_before, np.finfo(float).eps)

###############################################################################
# Plot the results
# -----------------------------------------------------------------------------
#
# Data was trained on a 40s window from 5s to 45s onwards (gray filled area).
# The algorithm then removes portions of this data with high amplitude
# artifacts before running the calibration (hatched area = good).
#
# What to look for:
# - After ASR, sharp bursts should be attenuated in many channels.
# - The RMS ratio (after/before) should generally be below 1.

times = np.arange(raw.shape[-1]) / sfreq
f, ax = plt.subplots(8, sharex=True, figsize=(9, 6))
for i in range(8):
    ax[i].fill_between(train_idx / sfreq, 0, 1, color="grey", alpha=.3,
                       transform=ax[i].get_xaxis_transform(),
                       label="calibration window")
    ax[i].fill_between(train_idx / sfreq, 0, 1, where=sample_mask.flat,
                       transform=ax[i].get_xaxis_transform(),
                       facecolor="none", hatch="...", edgecolor="k",
                       label="selected window")
    ax[i].plot(times, raw[i], lw=.5, label="before ASR")
    ax[i].plot(times, clean[i], label="after ASR", lw=.5)
    ax[i].set_ylim([-50, 50])
    ax[i].set_ylabel(f"ch{i}")
    ax[i].set_yticks([])
ax[0].set_title("Raw and cleaned EEG traces")
ax[i].set_xlabel("Time (s)")
ax[0].legend(fontsize="small", bbox_to_anchor=(1.04, 1), borderaxespad=0)
plt.subplots_adjust(hspace=0, right=0.75)
plt.suptitle("Before/after ASR")

fig, axm = plt.subplots(1, 1, figsize=(7, 3))
axm.bar(np.arange(raw.shape[0]), rms_ratio)
axm.axhline(1.0, color="k", ls=":", lw=1)
axm.set_xlabel("Channel")
axm.set_ylabel("RMS ratio (after / before)")
axm.set_title("Channel-wise attenuation summary")
axm.set_xticks(np.arange(raw.shape[0]))
axm.grid(True, axis="y", ls=":", alpha=.4)
plt.tight_layout()

print(f"Median RMS ratio across channels: {np.median(rms_ratio):.3f}")
plt.show()
