# MEEGkit

[![unit-tests](https://github.com/nbara/python-meegkit/workflows/unit-tests/badge.svg)](https://github.com/nbara/python-meegkit/actions?workflow=unit-tests)
[![Travis](https://travis-ci.org/nbara/python-meegkit.svg?branch=master)](https://travis-ci.org/nbara/python-meegkit)
[![codecov](https://codecov.io/gh/nbara/python-meegkit/branch/master/graph/badge.svg)](https://codecov.io/gh/nbara/python-meegkit)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nbara/python-meegkit/master)

Denoising tools for M/EEG processing in Python.

> **Disclaimer:** The project mostly consists of development code, although
> some modules and functions are already working. Bugs and performance problems
> are to be expected, so use at your own risk. More tests and improvements will
> be added in the near future. Comments and suggestions are welcome.

Python 2.7 and 3.5+ should be supported.

This code can be tested directly from your browser using
[Binder](https://mybinder.org), by clicking on the binder badge above.

## Installation

This package can be installed easily using `pip+git`:

```bash
pip install git+https://github.com/nbara/python-meegkit.git
```

Or you can clone this repository and run the following command inside the
`python-meegkit` directory:

```bash
pip install .
```

*Note* : Use developer mode with the `-e` flag (`pip install -e .`) to be able
to modify the sources even after install.

## Python implementation of CCA, STAR, SNS, DSS, and robust detrending

This is mostly a translation of Matlab code from the NoiseTools toolbox by
Alain de Cheveigné:
http://audition.ens.fr/adc/NoiseTools/

Original python implementation by Pedro Alcocer:
https://github.com/pealco

Only CCA, SNS, DSS, STAR and robust detrending have been properly tested so
far. TSCPA may give inaccurate results due to insufficient testing (PR
welcome!)

### References

If you use this code, you should cite the relevant methods from the original
articles :

```text
de Cheveigné, A., & Arzounian, D. (2018). Robust detrending, rereferencing, outlier detection,
    and inpainting for multichannel data. NeuroImage, 172, 903-912.
de Cheveigne, A., Di Liberto, G. M., Arzounian, D., Wong, D., Hjortkjaer, J., Fuglsang, S. A.,
    & Parra, L. C. (2018). Multiway Canonical Correlation Analysis of Brain Signals. bioRxiv,
    344960.
de Cheveigné A (2016). Sparse Time Artifact Removal, Journal of Neuroscience Methods, 262, 14-20
de Cheveigné A, Arzounian D (2015). Scanning for oscillations, Journal of Neural Engineering, 12,
    066020.
de Cheveigné, A., Parra, L. (2014). Joint decorrelation: a flexible tool for multichannel data
    analysis, Neuroimage
de Cheveigné, A., Edeline, J.M., Gaucher, Q. Gourévitch, B. (2013). Component analysis reveals
    sharp tuning of the local field potential in the guinea pig auditory cortex. J. Neurophysiol.
    109, 261-272.
de Cheveigné, A. (2012). Quadratic component analysis. Neuroimage 59: 3838-3844.
de Cheveigné, A. (2010). Time-shift denoising source separation. Journal of Neuroscience Methods
    189: 113-120.
de Cheveigné, A. and Simon, J. Z. (2008). Denoising based on spatial filtering. Journal of
    Neuroscience Methods 171: 331-339.
de Cheveigné, A. and Simon, J. Z. (2008). Sensor Noise Suppression. Journal of Neuroscience
    Methods 168: 195-202.
de Cheveigné, A. and Simon, J. Z. (2007). Denoising based on Time-Shift PCA. Journal of
    Neuroscience Methods 165: 297-305.
```
