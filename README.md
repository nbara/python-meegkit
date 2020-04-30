# MEEGkit

[![unit-tests](https://github.com/nbara/python-meegkit/workflows/unit-tests/badge.svg)](https://github.com/nbara/python-meegkit/actions?workflow=unit-tests)
[![Travis](https://travis-ci.org/nbara/python-meegkit.svg?branch=master)](https://travis-ci.org/nbara/python-meegkit)
[![codecov](https://codecov.io/gh/nbara/python-meegkit/branch/master/graph/badge.svg)](https://codecov.io/gh/nbara/python-meegkit)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nbara/python-meegkit/master)

Denoising tools for M/EEG processing in Python 3.6+.

> **Disclaimer:** The project mostly consists of development code, although some modules and functions are already working. Bugs and performance problems are to be expected, so use at your own risk. More tests and improvements will be added in the future. Comments and suggestions are welcome.

## Documentation

Automatic documentation is [available online](https://nbara.github.io/python-meegkit/).

This code can also be tested directly from your browser using [Binder](https://mybinder.org), by clicking on the binder badge above.

## Installation

This package can be installed easily using `pip+git`:

```bash
pip install git+https://github.com/nbara/python-meegkit.git
```

Or you can clone this repository and run the following command inside the `python-meegkit` directory:

```bash
pip install .
```

*Note* : Use developer mode with the `-e` flag (`pip install -e .`) to be able to modify the sources even after install.

## Python implementation of CCA, STAR, SNS, DSS, and robust detrending

This is mostly a translation of Matlab code from the [NoiseTools toolbox](http://audition.ens.fr/adc/NoiseTools/) by Alain de Cheveigné. It builds on an initial python implementation by [Pedro Alcocer](https://github.com/pealco).

Only CCA, SNS, DSS, STAR and robust detrending have been properly tested so far. ZapLine and TSCPA may give inaccurate results due to insufficient testing (contributions welcome!)

### References

If you use this code, you should cite the relevant methods from the original articles:

```sql
[1] de Cheveigné, A. (2019). ZapLine: A simple and effective method to remove power line artifacts.
    NeuroImage, 116356. https://doi.org/10.1016/j.neuroimage.2019.116356
[2] de Cheveigné, A. et al. (2019). Multiway canonical correlation analysis of brain data.
    NeuroImage, 186, 728–740. https://doi.org/10.1016/j.neuroimage.2018.11.026
[3] de Cheveigné, A. et al. (2018). Decoding the auditory brain with canonical component analysis.
    NeuroImage, 172, 206–216. https://doi.org/10.1016/j.neuroimage.2018.01.033
[4] de Cheveigné, A. (2016). Sparse time artifact removal.
    Journal of Neuroscience Methods, 262, 14–20. https://doi.org/10.1016/j.jneumeth.2016.01.005
[5] de Cheveigné, A., & Parra, L. C. (2014). Joint decorrelation, a versatile tool for multichannel data
    analysis. NeuroImage, 98, 487–505. https://doi.org/10.1016/j.neuroimage.2014.05.068
[6] de Cheveigné, A. (2012). Quadratic component analysis.
    NeuroImage, 59(4), 3838–3844. https://doi.org/10.1016/j.neuroimage.2011.10.084
[7] de Cheveigné, A. (2010). Time-shift denoising source separation.
    Journal of Neuroscience Methods, 189(1), 113–120. https://doi.org/10.1016/j.jneumeth.2010.03.002
[8] de Cheveigné, A., & Simon, J. Z. (2008a). Denoising based on spatial filtering.
    Journal of Neuroscience Methods, 171(2), 331–339. https://doi.org/10.1016/j.jneumeth.2008.03.015
[9] de Cheveigné, A., & Simon, J. Z. (2008b). Sensor noise suppression.
    Journal of Neuroscience Methods, 168(1), 195–202. https://doi.org/10.1016/j.jneumeth.2007.09.012
[10] de Cheveigné, A., & Simon, J. Z. (2007). Denoising based on time-shift PCA.
     Journal of Neuroscience Methods, 165(2), 297–305. https://doi.org/10.1016/j.jneumeth.2007.06.003

```

## Python implementation of Artifact subspace reconstruction (ASR)

The base code is inspired from the original [EEGLAB inplementation](https://github.com/sccn/clean_rawdata) [1], while the riemannian variant [2] was adapted from the [rASR toolbox](https://github.com/s4rify/rASRMatlab) by Sarah Blum.

### References

If you use this code, you should cite the relevant methods from the original articles:

```sql
[1] Mullen, T. R., Kothe, C. A. E., Chi, Y. M., Ojeda, A., Kerth, T., Makeig, S., et al. (2015). Real-time
    neuroimaging and cognitive monitoring using wearable dry EEG. IEEE Trans. Bio-Med. Eng. 62, 2553–2567.
    https://doi.org/10.1109/TBME.2015.2481482
[2] Blum, S., Jacobsen, N., Bleichner, M. G., & Debener, S. (2019). A Riemannian modification of
    artifact subspace reconstruction for EEG artifact handling. Frontiers in human neuroscience,
    13, 141.

```
