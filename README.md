# MEEGkit

[![unit-tests](https://github.com/nbara/python-meegkit/workflows/unit-tests/badge.svg?style=flat)](https://github.com/nbara/python-meegkit/actions?workflow=unit-tests)
[![documentation](https://img.shields.io/travis/nbara/python-meegkit.svg?label=documentation&logo=travis)](https://www.travis-ci.com/github/nbara/python-meegkit)
[![codecov](https://codecov.io/gh/nbara/python-meegkit/branch/master/graph/badge.svg)](https://codecov.io/gh/nbara/python-meegkit)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nbara/python-meegkit/master)
[![twitter](https://img.shields.io/twitter/follow/lebababa?label=Twitter&style=flat&logo=Twitter)](https://twitter.com/intent/follow?screen_name=lebababa)

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

## References

### 1. CCA, STAR, SNS, DSS, ZapLine, and robust detrending

This is mostly a translation of Matlab code from the [NoiseTools toolbox](http://audition.ens.fr/adc/NoiseTools/) by Alain de Cheveigné. It builds on an initial python implementation by [Pedro Alcocer](https://github.com/pealco).

Only CCA, SNS, DSS, STAR, ZapLine and robust detrending have been properly tested so far. TSCPA may give inaccurate results due to insufficient testing (contributions welcome!)

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
[5] de Cheveigné, A., & Parra, L. C. (2014). Joint decorrelation, a versatile tool for multichannel
    data analysis. NeuroImage, 98, 487–505. https://doi.org/10.1016/j.neuroimage.2014.05.068
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

### 2. Artifact subspace reconstruction (ASR)

The base code is inspired from the original [EEGLAB inplementation](https://github.com/sccn/clean_rawdata) [1], while the riemannian variant [2] was adapted from the [rASR toolbox](https://github.com/s4rify/rASRMatlab) by Sarah Blum.

If you use this code, you should cite the relevant methods from the original articles:

```sql
[1] Mullen, T. R., Kothe, C. A. E., Chi, Y. M., Ojeda, A., Kerth, T., Makeig, S., et al. (2015).
    Real-time neuroimaging and cognitive monitoring using wearable dry EEG. IEEE Trans. Bio-Med.
    Eng. 62, 2553–2567. https://doi.org/10.1109/TBME.2015.2481482
[2] Blum, S., Jacobsen, N., Bleichner, M. G., & Debener, S. (2019). A Riemannian modification of
    artifact subspace reconstruction for EEG artifact handling. Frontiers in human neuroscience,
    13, 141.
```

### 3. Rhythmic entrainment source separation (RESS)

The code is based on [Matlab code from Mike X. Cohen](https://mikexcohen.com/data/) [1]

If you use this, you should cite the following article:

```sql
[1] Cohen, M. X., & Gulbinaite, R. (2017). Rhythmic entrainment source separation: Optimizing analyses
    of neural responses to rhythmic sensory stimulation. Neuroimage, 147, 43-56.
```

### 4. Task-Related Component Analysis (TRCA)

This code is based on the [Matlab implementation from Masaki Nakanishi](https://github.com/mnakanishi/TRCA-SSVEP), and was adapted to python by [Giuseppe Ferraro](mailto:giuseppe.ferraro@isae-supaero.fr)

If you use this, you should cite the following articles:

```sql
[1] M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung,
    "Enhancing detection of SSVEPs for a high-speed brain speller using
    task-related component analysis", IEEE Trans. Biomed. Eng, 65(1): 104-112,
    2018.
[2] X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao, "Filter bank
    canonical correlation analysis for implementing a high-speed SSVEP-based
    brain-computer interface", J. Neural Eng., 12: 046008, 2015.
[3] X. Chen, Y. Wang, M. Nakanishi, X. Gao, T. -P. Jung, S. Gao,
    "High-speed spelling with a noninvasive brain-computer interface",
    Proc. Int. Natl. Acad. Sci. U. S. A, 112(44): E6058-6067, 2015.
```
