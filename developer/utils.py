import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import warnings

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
#### configuration ####
SOURCE_BIT = 30
CRDS_FLATS_MAPPING = '../data/flat/jwst_nircam_flat_0025.rmap'
FLAT_PATH = '../data/flat/'
##########################


def make_mask(dq, err, ndilate=2, mask_jumps=False):
    source = np.bitwise_and(dq, 1<<SOURCE_BIT) == 1<<SOURCE_BIT
    if ndilate > 0:
        source = np.array([binary_dilation(s, iterations=ndilate) for s in source])

    valid = (np.isfinite(err) &  # useful err
             (err != 0) &        # useful err
             (~np.isnan(err)) &   # useful err
             ((dq & 1<<0) == 0)  # make sure DO_NOT_USE is not set.
            )
    if mask_jumps:
        valid = valid & (dq == 0)
    mask = (~valid) | source

    # remove images that have no source mask
    has_mask = source.sum(axis=(1,2)) > (0.2 * np.prod(source.shape[-2:]))
    no_sourcemask = ~has_mask
    if (has_mask.sum() < len(source)):
        print(f"{(~has_mask).sum()} frames have no source mask; masking entire frame stack")
        mask[no_sourcemask] = True

    return mask


def rectify_db(dbname, ndilate=2, mask_jumps=False):

    with fits.open(dbname) as pix1:
        pix1.info()
        sci = pix1["SCI"].data
        err = pix1["ERR"].data
        dq = pix1["DQ"].data
        meta = pix1["META"].data

    assert np.all(dq >= 0)
    assert np.all(err >= 0)

    mask = make_mask(dq, err, ndilate=ndilate, mask_jumps=mask_jumps)
    mask = mask | np.isnan(sci)
    return sci, err, dq, meta, mask


def load_stsci_wisp(wisp_path, downsample=4, ext=0):
    from astropy.io import fits
    from photutils.segmentation import detect_sources
    # --- load wisp mask ---
    init_wisp = fits.open(wisp_path)
    wisp = init_wisp[ext].data
    wisp = rebin(wisp, (downsample, downsample)).astype(float)
    wisp_mask = (init_wisp["MASK"].data == 1).astype(np.int32)
    wisp_mask = rebin(wisp_mask, (downsample, downsample), bitwise=True).astype(bool)
    
    # --- only keep the largest wisp region ---
    wisp_mask_seg = detect_sources(wisp_mask.astype(float), threshold=0.5, npixels=100)
    largest_obj_idx = np.argmax(wisp_mask_seg.areas) + 1
    wisp_mask = np.zeros(wisp_mask_seg.shape, dtype=bool)
    wisp_mask[wisp_mask_seg.data == largest_obj_idx] = True

    return wisp, wisp_mask


def stsci_flat_name(detector, filter):
    from astropy.time import Time
    from crds import rmap
    sflats = np.array(rmap.load_mapping(os.path.expandvars(CRDS_FLATS_MAPPING)).todict()["selections"])
    
    pupil = "CLEAR"
    if filter.upper() == "F150W2":
        filter = "F150W, F150W2"
        pupil = "F162M"

    sel = (sflats[:, 0] == detector.upper()) & (sflats[:, 1] == filter.upper()) & (sflats[:, 2] == pupil)
    times = Time(sflats[sel, 3])
    st_flatn = sflats[sel, 4][np.argmax(times.mjd)]
    return str(st_flatn)


def load_stsci_flat_from(flat_path, downsample=4):
    with fits.open(flat_path) as flat_hdul:
        flat = rebin(flat_hdul["SCI"].data, (downsample, downsample))
    return flat

def load_stsci_flat(detector, filter, downsample=4):
    flat_name = stsci_flat_name(detector, filter)
    flat_path = os.path.join(os.path.expandvars(FLAT_PATH), flat_name)
    print(f"Loading flat from {flat_path}")
    return load_stsci_flat_from(flat_path, downsample=downsample)


def show_stacked(data, mask=None, title=None, range=(5, 95), stack_func=np.nanmedian, filter_hotpixel=False):
    """
    Show median stacked image.
    """
    if mask is not None:
        data = data.copy()
        data[mask] = np.nan
    stacked = stack_func(data, axis=0)
    stacked[np.isnan(stacked)] = 0.
    if filter_hotpixel:
        from scipy.ndimage import median_filter
        stacked = median_filter(stacked, size=3)
    vmin, vmax = np.percentile(stacked, range)
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    im = ax.imshow(stacked, origin='lower', vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%")
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if title is not None:
        ax.set_title(title)
    plt.show()
    return stacked


def rebin(a, factor, bitwise=False):
    """
    Rebin a 2D array by a factor of `factor` along each axis.
    If `bitwise` is True, the reduction is done by bitwise OR.
    """
    factor = np.array(factor)
    if np.all(factor == 1):
        return a

    assert np.all(np.mod(factor, 1) == 0)
    factor = factor.astype(int)
    new_shape = (np.array(a.shape) / np.array(factor)).astype(int)

    # reshape
    view = a.reshape(new_shape[0],factor[0],new_shape[1],factor[1],)

    if bitwise:
        # bitwise or reduce over new axes
        new = np.bitwise_or.reduce(view, axis=1)
        new = np.bitwise_or.reduce(new, axis=2)
    else:
        # average reduce over new axes
        new = view.sum(1).sum(2)/factor[0]/factor[1]

    return new

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

def load_data(data_name):
    hdul = fits.open(data_name)
    cal = hdul['SCI'].data
    err = hdul['ERR'].data
    mask = hdul['MASK'].data
    bg = hdul['BG'].data[0]
    meta = hdul['META'].data
    return cal, err, bg, mask, meta

def load_wisp_region(wisp_region_name):
    hdul = fits.open(wisp_region_name)
    wisp_region = hdul['WISP_REGIONS'].data
    return wisp_region


def show_images(data, title=None, mask=None, vlim=(1, 99), vlim_value=None, cmap='viridis', show_colorbar=False):
    # Determine grid size based on number of outliers
    n_img = len(data)
    ny = int(np.sqrt(n_img))
    nx = int(np.ceil(n_img / ny))

    ysize = 4 if show_colorbar else 3.3
    
    fig, axs = plt.subplots(ny, nx, figsize=(nx * 3, ny * ysize))
    if n_img == 1:
        axs = np.array([axs])
    for i, ax in enumerate(axs.flatten()):
        if i < n_img:
            data_copy = data[i].copy()
            data_copy[np.isnan(data_copy)] = 0
        
            vmin, vmax = np.percentile(data_copy, vlim)
            if vlim_value is not None:
                vmin, vmax = vlim_value
            im = ax.imshow(data_copy, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
            if title is not None:
                ax.set_title(f'{title[i]}', fontsize=18)
            ax.axis('off')
            if mask is not None:
                ax.imshow(mask[i], origin='lower', alpha=0.5, cmap='Grays')
            if show_colorbar:
                cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig
    
def plot_comps(comps, percent_range=(1, 99.9), cmap='viridis'):
    n_comp = len(comps)
    titles = [f'Component {k}' for k in range(n_comp)]
    fig = show_images(comps, titles, vlim=percent_range, cmap=cmap)
    return fig

def plot_amps(amp, outliers=None):
    fig, ax = plt.subplots(figsize=(15, 5))
    n_comp = amp.shape[1]
    for i in range(n_comp):
        ax.plot(amp[:, i], '.', label=f'Component {i}')
    
    plt.legend()
    
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1])
    if outliers is not None:
        for i in range(n_comp):
            ax.plot(outliers, amp[outliers, i], 'o', color='r', alpha=0.5)
        # plt.scatter(outliers, [ylim[0]] * amp[:, i], color='r', alpha=0.5, label='Outliers')
        # for i in range(len(outliers)):
            # ax.vlines(outliers[i], *ylim, color='r', alpha=0.5)
    
    plt.show()
    return fig

def vectorize_and_select_data(data, err, mask, valid_region):
    selected_pixels = ~np.all(np.isnan(data), axis=0)  # remove pixels with all NaN
    wisp_pixels = np.any(valid_region, axis=0)
    selected_pixels = selected_pixels & wisp_pixels
    data = data[:, selected_pixels].reshape(data.shape[0], -1)
    err = err[:, selected_pixels].reshape(err.shape[0], -1)
    mask = mask[:, selected_pixels].astype(int)
    
    err[err == 0] = np.inf
    invVar = 1. / err**2

    data *= 1 - mask
    invVar *= 1 - mask

    nan_values = np.isnan(data)
    data[nan_values] = 0
    invVar[nan_values] = 0
    return data, invVar, selected_pixels

def pca(cal, err, mask, wisp_region):
    data, invVar, selected_pixels = vectorize_and_select_data(cal, err, mask, wisp_region)
    amps, comps = _pca(data, invVar, n_comp=9)
    comps = transform2D(comps, selected_pixels)
    return amps, comps

def _pca(X, invVar, n_comp=6):
    """
    Perform PCA on the data scaled by the inverse variance of each feature.
    
    Parameters:
    X : numpy.ndarray
        Input data matrix of shape (n_samples, n_pix).
    invVar : numpy.ndarray
        Inverse variance (1/σ²) for each feature, array of shape (n_sample, n_pix).
        
    Returns:
    amps : numpy.ndarray
        Amplitudes of the principal components, shape (n_sample, n_comp).
    components : numpy.ndarray
        Principal components (eigenvectors) sorted by explained variance, shape (n_comp, n_pix).
    """
    X_centered = X - np.mean(X, axis=0)
    X_scaled = X_centered * np.sqrt(invVar)
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)    
    amps = U * s

    signs = np.sign(amps[np.argmax(np.abs(amps), axis=0), range(amps.shape[1])])
    components = Vt * signs[:, None]
    amps *= signs

    return amps[:, :n_comp], components[:n_comp]

def transform2D(data, selected_pixels, fill_value=0):
    """data is either 1D or (n_exposure, n_pixel).
       allmasked: (ny, nx)"""
    ny, nx = selected_pixels.shape
    if data.ndim == 1:
        new_data = np.ones((ny, nx), dtype=data.dtype) * fill_value
        new_data[selected_pixels] = data
    else:
        new_shape = (data.shape[0], ny, nx)
        new_data = np.ones(new_shape, dtype=data.dtype) * fill_value
        new_data[:, selected_pixels] = data

    return new_data