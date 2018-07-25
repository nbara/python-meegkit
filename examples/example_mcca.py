"""Example multiway canonical correlation analysis (mCCA) code.

Find a set of components which are shared between different datasets.

Uses cca.mmca().
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from meegkit import cca  # noqa:E402


# Create 3 uncorrelated data sets
x1 = np.random.randn(10000, 10)
x2 = np.random.randn(10000, 10)
x3 = np.random.randn(10000, 10)
x = np.hstack((x1, x2, x3))
C = np.dot(x.T, x)
print('Aggregated data covariance shape: {}'.format(C.shape))
[A, score, AA] = cca.mcca(C, 10)
z = x.dot(A)

f, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(A, aspect='auto')
axes[0].set_title('mCCA transform matrix')
axes[1].imshow(A.T.dot(C.dot(A)), aspect='auto')
axes[1].set_title('Covariance of\ntransformed data')
axes[2].imshow(x.T.dot((x.dot(A))), aspect='auto')
axes[2].set_title('Cross-correlation between\nraw & transformed data')
axes[2].set_xlabel('transformed')
axes[2].set_ylabel('raw')
plt.plot(np.mean(z ** 2, axis=0))
plt.show()

# Now Create 3 data sets with shared parts
x1 = np.random.randn(10000, 5)
x2 = np.random.randn(10000, 5)
x3 = np.random.randn(10000, 5)
x4 = np.random.randn(10000, 5)
x = np.hstack((x2, x1, x3, x1, x4, x1))
C = np.dot(x.T, x)
print('Aggregated data covariance shape: {}'.format(C.shape))
A, score, AA = cca.mcca(C, 10)

f, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(A, aspect='auto')
axes[0].set_title('mCCA transform matrix')
axes[1].imshow(A.T.dot(C.dot(A)), aspect='auto')
axes[1].set_title('Covariance of\ntransformed data')
axes[2].imshow(x.T.dot((x.dot(A))), aspect='auto')
axes[2].set_title('Cross-correlation between\nraw & transformed data')
axes[2].set_xlabel('transformed')
axes[2].set_ylabel('raw')
plt.show()

# Finally let's create 3 identical 10-channel data sets. Only 10 worthwhile
# components should be found, and the transformed dataset should perfectly
# explain all the variance (empty last two block-columns in the
# cross-correlation plot).
x1 = np.random.randn(10000, 10)
x = np.hstack((x1, x1, x1))
C = np.dot(x.T, x)
print('Aggregated data covariance shape: {}'.format(C.shape))
A, score, AA = cca.mcca(C, 10)

f, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(A, aspect='auto')
axes[0].set_title('mCCA transform matrix')
axes[1].imshow(A.T.dot(C.dot(A)), aspect='auto')
axes[1].set_title('Covariance of\ntransformed data')
axes[2].imshow(x.T.dot((x.dot(A))), aspect='auto')
axes[2].set_title('Cross-correlation between\nraw & transformed data')
axes[2].set_xlabel('transformed')
axes[2].set_ylabel('raw')
plt.show()
