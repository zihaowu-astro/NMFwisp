# NMFwisp

NMFwisp subtracts wisps from JWST/NIRCam images using data-driven, detector- and filter-specific templates.

[Wisps](https://jwst-docs.stsci.edu/known-issues-with-jwst-data/nircam-known-issues/nircam-scattered-light-artifacts#:~:text=Figure%204.%20Claws%20and%20wisps) are scattered-light artifacts in JWST/NIRCam images. They usually appear in the same locations on NIRCam detectors with mild morphological variation between observations. Wisps are a significant source of contamination for objects fainter than 25 AB mag.

The template library is built using the Non-negative Matrix Factorization (NMF) algorithm, leveraging extensive NIRCam data from JADES and other programs. Compared with single-template approaches, the NMF-based method captures exposure-to-exposure wisp morpholigical variation, while retaining sensitivity to the low-surface-brightness structure of the wisps.

Wisp subtraction is intended for [Stage 2 of the JWST calibration pipeline](https://jwst-docs.stsci.edu/jwst-science-calibration-pipeline/stages-of-jwst-data-processing#gsc.tab=0). The main user interface is the `fit_wisp` funciton, which returns the best-fit wisp model and its uncertainty. Templates are distributed with the package in the `nmfwisp/templates` folder. The `developer/` directory contains scripts used to build the template library. The package size is 82 MB including the templates. Typical runtime on one CPU core (Apple M4 Pro) is ~0.4 s per exposure for standard fitting and ~2 s per exposure with iterative [1/f noise](https://jwst-docs.stsci.edu/known-issues-with-jwst-data/1-f-noise#gsc.tab=0) correction. 

## Installation

Install from PyPI:

```bash
pip install nmfwisp
```

Alternatively, development install from source:

```bash
git clone https://github.com/zihaowu-astro/NMFwisp.git
cd NMFwisp
pip install -e .
```

## Example

Example Data (Optional): to run the example below, download the sample dataset:

```bash
curl -L -O https://github.com/zihaowu-astro/NMFwisp/releases/download/v1.0/example-data.tar.gz
tar -xzf example-data.tar.gz
```

Direct download link: [example-data.tar.gz](https://github.com/zihaowu-astro/NMFwisp/releases/download/v1.0/example-data.tar.gz)



```python
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from nmfwisp import fit_wisp

filter_name = "F150W"
detector_name = "nrcb4"

filename = "./data/jw01286001001_07201_00003_nrcb4_rate.fits"
maskfile = "./data/jw01286001001_07201_00003_nrcb4_cal_bkgsub_tweak_smask-full.fits"

data = fits.open(filename)["SCI"].data
err = fits.open(filename)["ERR"].data
mask = fits.open(maskfile)[0].data

wisp, wisp_e = fit_wisp(
    data, err, mask,
    detector_name=detector_name,
    filter_name=filter_name,
    correct_1f=False,
)
```

Use a custom template path (optional):

```python
wisp, wisp_e = fit_wisp(
    data, err, mask,
    wisp_path="/path/to/templates",
    detector_name=detector_name,
    filter_name=filter_name,
    correct_1f=False,
)
```

Visualization:

```python
data0 = np.nan_to_num(data, nan=0.0)
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
vmin, vmax = np.nanpercentile(data, 5), np.nanpercentile(data, 95)
ax[0].imshow(data0, origin="lower", vmin=vmin, vmax=vmax)
ax[0].set_title("Data")
ax[1].imshow(data0 - wisp, origin="lower", vmin=vmin, vmax=vmax)
ax[1].set_title("Data - WISP")
ax[2].imshow(wisp, origin="lower", vmin=0, vmax=np.nanpercentile(wisp, 99))
ax[2].set_title("WISP")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()
```

![](doc/example.png)

## Wisp Morphology

![Demonstration of wisp morphology in all affected NIRCam detectors in the F150W band](doc/all_wisps.jpg)

## Citation

If you use this code, please cite [this paper](https://arxiv.org/abs/2601.15958):
```
@ARTICLE{2026arXiv260115958W,
       author = {{Wu}, Zihao and {Johnson}, Benjamin D. and {Eisenstein}, Daniel J. and {Cargile}, Phillip and {Hainline}, Kevin and {Hausen}, Ryan and {Rinaldi}, Pierluigi and {Robertson}, Brant E. and {Tacchella}, Sandro and {Williams}, Christina C. and {Willmer}, Christopher N.~A.},
        title = "{JWST Advanced Deep Extragalactic Survey (JADES) Data Release 5: Wisp Subtraction with the Non-negative Matrix Factorization Algorithm}",
      journal = {arXiv e-prints},
     keywords = {Instrumentation and Methods for Astrophysics, Astrophysics of Galaxies},
         year = 2026,
        month = jan,
          eid = {arXiv:2601.15958},
        pages = {arXiv:2601.15958},
          doi = {10.48550/arXiv.2601.15958},
archivePrefix = {arXiv},
       eprint = {2601.15958},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260115958W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
