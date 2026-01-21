from astropy.stats import sigma_clipped_stats
import numpy as np


def interpolate_wisp_nans(wisp, nan_mask, sigma=1):
    """
    Interpolate NaN values in the WISP data.
    wisp: 2D array with NaN values
    nan_mask: boolean mask of the same shape as wisp, where True indicates NaN values
    """
    from scipy.ndimage import gaussian_filter
    n_template = wisp.shape[0]
    wisp_interp = np.copy(wisp)
    for i in range(n_template):
        interpolated = gaussian_filter(wisp[i], sigma=sigma, order=0, mode='nearest')[nan_mask]
        weights = gaussian_filter(wisp[i] != 0, sigma=sigma, order=0, mode='nearest')[nan_mask]
        weights[weights < 0.1] = 1  # Avoid division by zero
        wisp_interp[i][nan_mask] = interpolated / weights

    return wisp_interp


def weighted_stack(data, invVar):
    """
    Weighted stack of data.
    data: 2D array, shape (n_exp, n_pix)
    invVar: 2D array, shape (n_exp, n_pix)
    """
    data_invVar = data * invVar
    data_sum = np.sum(data_invVar, axis=0)
    invVar_sum = np.sum(invVar, axis=0)
    data_stack = data_sum / (invVar_sum + 1e-10)
    return data_stack


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


def estimate_errors(residual, invVar, outliers, selected_pixels, nan_pixels):
    """
    Estimate the WISP bias and error.
    """
    from scipy.ndimage import median_filter, gaussian_filter
    non_outliers = np.delete(np.arange(residual.shape[0]), outliers)
    residual, invVar = residual[non_outliers], invVar[non_outliers]
    wisp_bias = weighted_stack(residual, invVar)
    noise_power = invVar**(-1)
    noise_power = np.nan_to_num(noise_power, nan=0.0)
    noise_power = np.clip(noise_power, 0, None)
    residual_power = residual**2
    wisp_noise_power = weighted_stack((residual_power - noise_power), invVar)
    median_noise_power = np.nanmedian(noise_power[noise_power>0])
    wisp_err_base = median_noise_power / np.sum(invVar > 0, axis=0)
    wisp_err = np.sqrt(np.clip(wisp_noise_power, wisp_err_base, None))
    
    wisp_bias = transform2D(wisp_bias, selected_pixels)
    wisp_err  = transform2D(wisp_err, selected_pixels)
    
    wisp_bias, wisp_err = interpolate_wisp_nans(np.array([wisp_bias, wisp_err]), nan_pixels)
    wisp_err = median_filter(wisp_err, size=3)
    wisp_err = gaussian_filter(wisp_err, sigma=1)

    return wisp_bias, wisp_err