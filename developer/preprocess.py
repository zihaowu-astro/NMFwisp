import numpy as np
from astropy.stats import sigma_clipped_stats, sigma_clip
from scipy.ndimage import median_filter, gaussian_filter, binary_dilation
from photutils.segmentation import detect_sources
from astropy.convolution import Gaussian2DKernel, convolve
import bottleneck as bn
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


def estimate_background(data, mask):
    """
    Estimate the background of a 3D data cube by sigma-clipped median.
    
    Parameters
    ----------
    data : 3D array
        The input data cube.
    wisp_mask : 2D boolean array
        A mask array for the WISP region.
        
    Returns
    -------
    bg : 1D array
        The estimated background at each frame.
    """
    bg = np.zeros(data.shape[0], dtype=np.float32)
    for i in range(data.shape[0]):
        valid_pixels = ~mask[i] & ~np.isnan(data[i])
        data_flat = data[i][valid_pixels]
        _, bg[i], _ = sigma_clipped_stats(data_flat, sigma=3, maxiters=5)
    bg[np.isnan(bg)] = 0.
    return bg

def calibrate_rates(rates, error, flat, mask, wisp_mask):
    bg = estimate_background(rates/flat[None, :, :], mask | wisp_mask[None, :, :],)
    cal, cal_e = rates / flat[None, :, :], error / flat[None, :, :]
    bsub = cal - bg[:, None, None]
    return bsub, cal_e, bg

def oversample_mask(mask, factor):
    """
    Oversample a mask by a factor of `factor` along each axis.
    """
    factor = np.array(factor)
    if np.any(factor <= 1):
        raise ValueError("Oversampling factors must be greater than 1.")
    oversampled = mask.repeat(factor[0], axis=0).repeat(factor[1], axis=1)
    return oversampled


def gaussian_interpolate(data, sigma=3, axes=None):
    """
    Interpolate NaN values in a 2D array using a Gaussian filter.

    Parameters
    ----------
    data : 2D array
        The input data array.
    sigma : float
        The standard deviation of the Gaussian filter.

    Returns
    -------
    interpolated : 2D array
        The interpolated data array.
    """
    mask = np.isnan(data)
    if axes is None:
        axes = np.arange(data.ndim)

    data[mask] = 0.
    smoothed = gaussian_filter(data, sigma=sigma, axes=axes)
    pix_num = np.ones_like(data)
    pix_num[mask] = 0.
    pix_num = gaussian_filter(pix_num, sigma=sigma, axes=axes)
    
    pix_num[np.isnan(pix_num)] = 0.5
    pix_num = np.clip(pix_num, 0.5, None)
    interpolated = smoothed/pix_num

    data[mask] = interpolated[mask]
    return data


def clip_strip_outlier(stripes, sigma=3):
    """
    Clip outliers in horizontal or vertical stripes.
    """
    _, _, std = sigma_clipped_stats(stripes, axis=(1, 2), sigma=sigma, maxiters=10,)
    mask_outlier = np.abs(stripes) > 10 * std[:, None, None]
    stripes[mask_outlier] = np.nan
    stripes = gaussian_interpolate(stripes, sigma=5, axes=(1, 2))
    return stripes


def estimate_1f_noise(bsub, mask, split_amplifier=False):
    """
    Correct 1/f noise in a 3D array.
    
    
    Parameters
    ----------
    bsub : 3D array    
    mask : 3D boolean array
        A boolean mask array with the same shape as `bsub`.
    split_amplifier : bool
        Whether the input data is split into 4 amplifiers.

    Returns
    -------
    hstripes : 3D array
        The horizontal stripes.
    vstripes : 3D array
        The vertical stripes.
    """
    bsub_temp = bsub.copy()
    bsub_temp[mask] = np.nan
    min_pix = 15

    if split_amplifier:
        hstripes = np.zeros_like(bsub_temp)
        idx = np.array([0, 1, 2, 3, 4]) * (bsub_temp.shape[-1] // 4)
        for i in range(4):
            sliced = bsub_temp[:, :, idx[i]:idx[i+1]]
            sliced = sigma_clip(sliced, axis=2, sigma=5, maxiters=5).data
            hstripes[:, :, idx[i]:idx[i+1]] = np.nanmedian(sliced, axis=2, keepdims=True)

            bool_invalid = np.sum(~np.isnan(sliced), axis=2) < min_pix
            bool_invalid = bool_invalid[:, :, None]
            hstripes[:, :, idx[i]:idx[i+1]][bool_invalid] = np.nan
    else:
        bsub_temp_clipped = sigma_clip(bsub_temp, axis=2, sigma=5, maxiters=5)
        hstripes = np.nanmean(bsub_temp_clipped, axis=2, keepdims=True)
        bool_invalid = np.sum(~np.isnan(bsub_temp), axis=2, keepdims=True) < min_pix
        hstripes[bool_invalid] = np.nan

    hstripes = clip_strip_outlier(hstripes)
    hstripes = hstripes.data

    vstripes = np.nanmedian(bsub_temp - hstripes, axis=1, keepdims=True)
    bool_invalid = np.sum(~np.isnan(bsub_temp), axis=1, keepdims=True) < min_pix
    vstripes[bool_invalid] = np.nan
    vstripes = clip_strip_outlier(vstripes)

    return hstripes, vstripes


def estimate_1f_noise_beta(bsub, mask, split_amplifier=False):
    bsub_temp = np.where(mask, np.nan, bsub)
    min_pix = 15

    if split_amplifier:
        hstripes = np.empty_like(bsub_temp)
        n_cols = bsub_temp.shape[-1]
        idx = np.linspace(0, n_cols, 5, dtype=int)
        
        for i in range(4):
            sliced = bsub_temp[:, :, idx[i]:idx[i+1]]
            med = bn.nanmedian(sliced, axis=2)[:, :, None]
            mad = bn.nanmedian(np.abs(sliced - med), axis=2)[:, :, None]
            outlier_mask = np.abs(sliced - med) > 5 * 1.4826 * mad
            
            sliced_clean = np.where(outlier_mask, np.nan, sliced)
            median_vals = bn.nanmedian(sliced_clean, axis=2)[:, :, None]
            valid_count = np.sum(~np.isnan(sliced), axis=2)[:, :, None]
            median_vals[valid_count < min_pix] = np.nan
            hstripes[:, :, idx[i]:idx[i+1]] = median_vals
    else:
        med = bn.nanmedian(bsub_temp, axis=2)[:, :, None]
        mad = bn.nanmedian(np.abs(bsub_temp - med), axis=2)[:, :, None]
        outlier_mask = np.abs(bsub_temp - med) > 5 * 1.4826 * mad
        
        bsub_clean = np.where(outlier_mask, np.nan, bsub_temp)
        hstripes = bn.nanmedian(bsub_clean, axis=2)[:, :, None]
        
        valid_count = np.sum(~np.isnan(bsub_temp), axis=2)[:, :, None]
        hstripes[valid_count < min_pix] = np.nan

    hstripes = clip_strip_outlier(hstripes).data

    diff = bsub_temp - hstripes
    vstripes = bn.nanmedian(diff, axis=1)[:, None, :]
    valid_count_v = np.sum(~np.isnan(bsub_temp), axis=1)[:, None, :]
    vstripes[valid_count_v < min_pix] = np.nan
    vstripes = clip_strip_outlier(vstripes).data

    return hstripes, vstripes

def correct_1f_noise(data, mask, wisp_mask=None, split_amplifier=False, beta_version=True):
    """
    Correct 1/f noise in a 3D array.
    
    Parameters
    ----------
    data : 3D array
        The input data cube.
    mask : 3D boolean array
        The mask array.
    wisp_mask : 2D boolean array
        The mask array for the WISP region.
    split_amplifier : bool
        Whether the input data is split into 4 amplifiers.

    Returns
    -------
    corrected : 3D array
        The 1/f noise corrected data cube.
    """
    if wisp_mask is not None:
        wisp_mask = binary_dilation(wisp_mask, iterations=50)
        if wisp_mask.shape[0] == 1:
            mask = mask | wisp_mask
        else:
            mask = mask | wisp_mask[None, :, :]
            
    if beta_version:
        hstripes, vstripes = estimate_1f_noise_beta(data, mask, split_amplifier=split_amplifier)
    else:
        hstripes, vstripes = estimate_1f_noise(data, mask, split_amplifier=split_amplifier)
    corrected = data - hstripes - vstripes
    return corrected


def mask_bad_pixels(data, err, mask, std_threshold=10):   
    """
    Mask bad pixels in a 3D data cube.

    Parameters
    ----------
    data : 3D array
        The input data cube.
    err : 3D array

    Returns
    -------
    data : 3D array
        The masked data cube.
    mask : 2D boolean array
        The mask array.
    """ 
    data = data.copy()
    data[mask] = np.nan

    # --- mask hot pixels ---
    data_med = np.nanmedian(data, axis=0)
    err_med  = np.nanmedian(err,  axis=0)
    smoothed = median_filter(data_med, size=7)
    smoothed_err = median_filter(err_med, size=15)  # larger size to help remove hot pixels
    mask_hot = np.abs(data - smoothed) / smoothed_err  > std_threshold 
    data[mask_hot] = np.nan

    # --- mask hot pixels in the mean stack again ---
    data_mean = np.nanmean(data, axis=0)
    n_exposure = np.clip(np.sum(~np.isnan(data), axis=0), 1, None)
    err_med  = np.median(smoothed_err / np.sqrt(n_exposure))
    mask_hot2 = np.abs(data_mean - data_med) / err_med  > std_threshold
    data[:, mask_hot2] = np.nan

    # --- remove cold pixels ---
    mask_cold = data_mean / err_med  < -15
    data[:, mask_cold] = np.nan

    mask = mask_hot | mask_hot2[None, :] | mask_cold
    
    return mask

def mask_edges(mask, edges):
    """
    Modify a mask by masking the edges.

    Parameters
    ----------
    mask : 3D boolean array
        The input mask array.
    edges : tuple
        The number of pixels to mask on each edge (left, right, top, bottom).

    Returns
    -------
    mask : 3D boolean array
        The modified mask array.
    """
    left, right, top, bottom = edges
    mask[:, :, :left] = True
    mask[:, :bottom, :] = True
    if right != 0:
        mask[:, :, -right:] = True
    if top != 0:
        mask[:, -top:, :] = True
    return mask



def stack(data, mode='mean'):
    """
    Stack using either the mean, median, or standard deviation (std), ignoring NaNs. 

    Parameters
    ----------
    data : 3D numpy.ndarray
        The input data to stack. The shape is assumed to be (N, H, W) where
        N is the number of frames/images, and H and W are the height and width
        of each frame.
    mode : {'mean', 'median', 'std'}, optional
        The method used to combine values along the stack:
          - 'mean': compute the mean of each pixel across frames.
          - 'median': compute the median of each pixel across frames.
          - 'std': compute the standard deviation of each pixel across frames.

    Returns
    -------
    stacked : 2D numpy.ndarray
        The stacked result with the same height and width as each frame in
        `data` (shape (H, W)). Any pixel that is NaN in all frames will be 0
        in the output.

    Raises
    ------
    ValueError
        If `mode` is not one of 'mean', 'median', or 'std'.
    """
    valid = ~np.all(np.isnan(data), axis=0)
    stacked = np.zeros_like(data[0], dtype=np.float32)

    if mode == 'mean':
        stacked[valid] = np.nanmean(data[:, valid], axis=0)
    elif mode == 'median':
        stacked[valid] = np.nanmedian(data[:, valid], axis=0)
    elif mode == 'std':
        stacked[valid] = np.nanstd(data[:, valid], axis=0)
    else:
        raise ValueError(f"Invalid mode {mode}, please choose from 'mean', 'median', 'std'.")

    return stacked

from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage import binary_dilation
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources

def interpolate(data, mask):
    """
    Interpolate missing data in the stacked image.
    """
    interp_kernel = Gaussian2DKernel(5)
    conved = convolve(data, interp_kernel)
    ones = np.ones_like(data)
    ones[mask] = 0
    ones_conv = convolve(ones, interp_kernel)
    ones_conv[ones_conv==0] = 1
    data[mask] = conved[mask] / ones_conv[mask]
    return data


def set_edges(data, edges, fill_value=0):
    if edges is None:
        return data
    data[:, :edges[0]] = fill_value
    if edges[1] > 0:
        data[:, -edges[1]:] = fill_value
    data[:edges[3]] = fill_value
    if edges[2] > 0:
        data[-edges[2]:] = fill_value
    return data

def make_segmentation(data, kernel_size=5, std_threshold=0.5, npixels=100, min_flux_ratio=0.1):
    """
    Detect sources in the data.
    args: smooth, threshold, npixels, min_flux_ratio, dilation, max_comp_num
    """
    mean, median, std = sigma_clipped_stats(data)
    kernel = Gaussian2DKernel(kernel_size)
    convolved_data = convolve(data, kernel, boundary='extend')
    segment_map = detect_sources(convolved_data - median, threshold=std * std_threshold, npixels=npixels)

    # sort labels by total flux
    max_label = segment_map.data.max()
    fluxes = np.bincount(segment_map.data.ravel(), weights=data.ravel(), minlength=max_label + 1)
    fluxes[0] = 0   # background
    sorted_labels = np.argsort(fluxes)[::-1]

    # select valid labels
    source_labels = sorted_labels[fluxes[sorted_labels] > min_flux_ratio * fluxes.max()]
    return segment_map, source_labels

def make_wisp_regions(segment_map, wisp_labels, dilation=1, 
                      source_labels=None, allow_overlap=False, source_dilation=10):
    shape = segment_map.data.shape
    wmask = np.zeros((len(wisp_labels), shape[0], shape[1]), dtype=bool)

    for i, label in enumerate(wisp_labels):
        wmask[i] = segment_map.data == label
        wmask[i] = binary_dilation(wmask[i], iterations=dilation)

        if (not allow_overlap) and (source_labels is not None):
            mask_other_sources = np.isin(segment_map.data, source_labels[source_labels != label])
            mask_other_sources = binary_dilation(mask_other_sources, iterations=source_dilation)
            wmask[i][mask_other_sources] = False
    return wmask

def build_wisp_regions(data, edges=None, max_comp_num=1, dilation=1,
                       kernel_size=5, threshold=1.0, npixels=100, min_flux_ratio=0.1, 
                       overlap=False, source_dilation=10):
    """
    Build WISP regions from the stacked image.

    Parameters
    ----------
    data : 2D numpy.ndarray
        The stacked image.
    edges : tuple or None
        The number of pixels to mask on each edge (left, right, top, bottom); if None do nothing.
    max_comp_num : int
        Maximum number of WISP regions (components) to keep.
    kernel_size : int
        Size of the Gaussian kernel for source detection.
    threshold : float
        Detection threshold factor.
    npixels : int
        Minimum number of connected pixels required.
    min_flux_ratio : float
        Minimum flux ratio for a region to be considered.
    dilation : int
        Number of iterations for dilating the regions.
    overlap : bool
        If True, allow overlapping regions.
    source_dilation : int
        Dilation iterations to exclude nearby sources.

    Returns
    -------
    wmask : list of 2D numpy.ndarray
        The WISP regions.
    segment_map : `photutils.segmentation.SourceCatalog`
        The segmentation map.
    """
    # Preprocess data
    data = data.copy()
    mask = np.isnan(data) | (data == 0)
    data = interpolate(data, mask)
    data = set_edges(data, edges)

    # Detect sources and select WISP regions
    segment_map, source_labels = make_segmentation(data, kernel_size, threshold, npixels, min_flux_ratio)
    wisp_labels = source_labels[:max_comp_num]
    wmask = make_wisp_regions(segment_map, wisp_labels, dilation, source_labels, overlap, source_dilation)

    # Apply edge cutoff to each region mask
    for i in range(len(wmask)):
        wmask[i] = set_edges(wmask[i], edges, fill_value=0)

    return wmask, segment_map

def show_wisp_regions(data, wmask, segmentmap, title='WISP Regions'):
    """
    Show the WISP regions on the data.
    """
    import matplotlib.pyplot as plt
    percentiles = np.percentile(data, [1, 99])
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(data, origin='lower', cmap='inferno', vmin=percentiles[0], vmax=percentiles[1])
    ax[1].imshow(segmentmap.data, origin='lower')
    for i in range(len(wmask)):
        ax[0].contour(wmask[i], levels=[0.5], colors='white')
        ax[1].contour(wmask[i], levels=[0.5], colors='white')
        # add number
        y, x = np.where(wmask[i])
        ax[1].text(x.mean(), y.mean(), str(i), color='tab:red', fontsize=12)

    plt.suptitle(title)
    plt.show()
    return fig, ax