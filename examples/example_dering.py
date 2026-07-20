"""
Ringing artifact reduction example
==================================

This example shows how to subtract an estimated filter impulse response to
reduce ringing artifacts around a sharp event.

The script reports a simple quantitative proxy: mean absolute amplitude in a
window around the artifact, before versus after correction.

Uses `meegkit.detrend.reduce_ringing()`.

References
----------
.. [1] See the method documentation for
   `meegkit.detrend.reduce_ringing` in meegkit.

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

from meegkit.detrend import reduce_ringing

rng = np.random.default_rng(9)

###############################################################################
# Detrending
# =============================================================================

###############################################################################
# Basic example with a linear trend
# -----------------------------------------------------------------------------
# Simulate the effect of filtering a signal containing a discontinuity, and try
# to remove the resulting ringing artifact by subtracting an estimate of the
# impulse response.

x = np.arange(1000) < 1
[b, a] = butter(6, 0.2)     # Butterworth filter design
x = lfilter(b, a, x) * 50    # Filter data using above filter
x = np.roll(x, 500)
x = x[:, None] + rng.standard_normal((1000, 2))
y = reduce_ringing(x, samples=np.array([500]))

roi = slice(450, 560)
mad_before = np.mean(np.abs(x[roi]))
mad_after = np.mean(np.abs(y[roi]))

plt.figure()
plt.plot(x + np.array([-10, 10]), "C0", label="before")
plt.plot(y + np.array([-10, 10]), "C1:", label="after")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Ringing reduction around the discontinuity")
plt.legend()
plt.tight_layout()

print(f"Mean absolute amplitude around event (before): {mad_before:.3f}")
print(f"Mean absolute amplitude around event (after):  {mad_after:.3f}")
print(f"Relative change (after/before): {mad_after / mad_before:.3f}")
plt.show()
