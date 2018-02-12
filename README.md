# MEEGkit

[![Build Status](https://travis-ci.org/nbara/python-meegkit.svg?branch=master)](https://travis-ci.org/nbara/python-meegkit)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/nbara/python-meegkit/master)

Denoising tools for M/EEG processing in Python.

> **Disclaimer:** The project is in early stages of development, although some modules and functions are already working. Bugs and performance problems are to be expected, so use at your own risk. More tests and improvements will be added in the near future. Comments and suggestions are welcome.  

## Python implementation of STAR, SNS, and DSS denoising

This is mostly a translation of Matlab code from the NoiseTools toolbox by
Alain de Cheveigné:  
http://audition.ens.fr/adc/NoiseTools/

Original python implementation by Pedro Alcocer:  
https://github.com/pealco

Python 2.7 and 3.5+ should be supported. Only SNS, DSS and STAR have been
properly tested so far. TSCPA seems to run fine but gives inaccurate
results.

This code can be tested directly from your browser using
[Binder](https://mybinder.org), by clicking on the binder badge above.

## References 

If you use this code, you should cite the relevant methods from the original
articles : 

```text
de Cheveigné A (2016) Sparse Time Artifact Removal, Journal of Neuroscience 
  Methods, 262, 14-20, doi:10.1016/j.jneumeth.2016.01.005
de Cheveigné A, Arzounian D (2015) Scanning for oscillations, Journal of 
  Neural Engineering, 12, 066020, DOI: 10.1088/1741-2560/12/6/066020.
de Cheveigné, A., Parra, L. (2014), Joint decorrelation: a flexible tool for 
  multichannel data analysis, Neuroimage, DOI: 10.1016/j.neuroimage.2014.05.068 
de Cheveigné, A., Edeline, J.M., Gaucher, Q. Gourévitch, B. (2013). 
  "Component analysis reveals sharp tuning of the local field potential in 
  the guinea pig auditory cortex." J. Neurophysiol. 109, 261-272.
de Cheveigné, A. (2012). "Quadratic component analysis." Neuroimage 59: 
  3838-3844.
de Cheveigné, A. (2010). "Time-shift denoising source separation." Journal of 
  Neuroscience Methods 189: 113-120.
de Cheveigné, A. and Simon, J. Z. (2008). "Denoising based on spatial 
  filtering." Journal of Neuroscience Methods 171: 331-339.
de Cheveigné, A. and Simon, J. Z. (2008). "Sensor Noise Suppression." Journal 
  of Neuroscience Methods 168: 195-202.
de Cheveigné, A. and Simon, J. Z. (2007). "Denoising based on Time-Shift 
  PCA." Journal of Neuroscience Methods 165: 297-305.
```
