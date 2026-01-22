Purpose
-------
This package subtract wisps from JWST/NIRCam images using data-driven, multi-component wisp templates. 

[Wisps](https://jwst-docs.stsci.edu/known-issues-with-jwst-data/nircam-known-issues/nircam-scattered-light-artifacts#:~:text=Figure%204.%20Claws%20and%20wisps
) are scatterd light artifacts in JWST NIRCam images. They usually appear in the same location of NIRCam detectors with mild morphological variation between observations.  Wisps are a significant contamination for sources fainter than 25 AB mag. 

We construct detector- and filter-specific wisp templates using the non-negative matrix factorization (NMF) algorithm, based on extensive NIRCam data from JADES and other programs. This algorithm efficiently extracts wisp morphology and its principal modes of variations. The NMF-derived templates yield substantial improvement in wisp subtraction compared to existing single-template approaches

Implementation
------------

Wisp subtraction should be applied at [stage 2 of the JWST data reduction pipeline](https://jwst-docs.stsci.edu/jwst-science-calibration-pipeline/stages-of-jwst-data-processing#gsc.tab=0).  For a single NIRCam detector, the runtime is about 0.4 seconds per exposure on one CPU core of an Apple M4 Pro. The runtime is 2 seconds when performing joint fitting with the [$1/f$ noise](https://jwst-docs.stsci.edu/known-issues-with-jwst-data/1-f-noise#gsc.tab=0).

The wisp templates are available at [link]. The main interface is the `fit_wisp` function in `nmfwisp.py`, which returns the best fit wisp model and its uncertainty. The only required user input is a source mask, which can be constructed from, for example, long-wavelength NIRCam images.

The `developer` directory contains code used to build the wisp template library. 

Example
-------

```py
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

filter_name = 'F150W'
detector_name = 'nrcb4'
wisp_path = './library'

# Example File
filename = './data/jw01286001001_07201_00003_nrcb4_rate.fits'
maskfile = './data/jw01286001001_07201_00003_nrcb4_cal_bkgsub_tweak_smask-full.fits'

data = fits.open(filename)['SCI'].data
err  = fits.open(filename)['ERR'].data
mask = fits.open(maskfile)[0].data

# Fit Wisp
from nmfwisp import fit_wisp
wisp, wisp_e = fit_wisp(data, err, mask, wisp_path, detector_name, filter_name, correct_1f=False)
```

Visualization of the wisp subtraction result:
```py
data0 = np.nan_to_num(data, nan=0.0) # remove nan values
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
vmin, vmax = np.nanpercentile(data, 5), np.nanpercentile(data, 95)
ax[0].imshow(data0, origin='lower', vmin=vmin, vmax=vmax)
ax[0].set_title('Data')
ax[1].imshow(data0 - wisp, origin='lower', vmin=vmin, vmax=vmax)
ax[1].set_title('Data - WISP')
ax[2].imshow(wisp, origin='lower', vmin=0, vmax=np.nanpercentile(wisp, 99))
ax[2].set_title('WISP')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()
```

![](doc/example.png)

Wisp Morphology
------------

![Demonstration of wisp morphology in all affected NIRCam detectors in the F150W band](doc/all_wisps.jpg)