import numpy as np
import time
from astropy.stats import sigma_clipped_stats

def nmf_basic(data, invVar, A, W, n_iter=100, verbose=False):
    """
    Non-negative matrix factorization
    A: shape (n_exp, n_comp)
    W: shape (n_comp, n_pix)
    """
    data_invVar = data * invVar
    if verbose:
        import tqdm
        for _ in tqdm.tqdm(range(n_iter)):
            W_factor = (A.T @ data_invVar) / (A.T @ (invVar * (A @ W)) + 1e-10)
            W_factor = np.clip(W_factor, 1e-1, 1e1)
            W *= W_factor
            A_factor = (data_invVar @ W.T) / ((invVar * (A @ W)) @ W.T + 1e-10)
            A_factor = np.clip(A_factor, 1e-1, 1e1)
            A *= A_factor
    else:
        for _ in range(n_iter):
            W_factor = (A.T @ data_invVar) / (A.T @ (invVar * (A @ W)) + 1e-10)
            W_factor = np.clip(W_factor, 1e-1, 1e1)
            W *= W_factor
            A_factor = (data_invVar @ W.T) / ((invVar * (A @ W)) @ W.T + 1e-10)
            A_factor = np.clip(A_factor, 1e-1, 1e1)
            A *= A_factor
    A_med = np.median(A, axis=0)
    H, A = W * A_med[:, None], A / A_med[None, :]
    return A, H
    

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


def _nmf_iterative(data, invVar, n_comp, n_iter=100, init_amp_ratio=0.1, verbose=False):
    """
    Iterative NMF.
    params:
    data: 2D array, shape (n_exp, n_pix)
    invVar: 2D array, shape (n_exp, n_pix)
    n_comp: int, number of components
    n_iter: int, number of iterations
    logger: logger object
    decay_rate: float, the initial amplitude of the new component is decay_rate * the maximum value of the first component.
    """
    n_exp, n_pix = data.shape
    stacked = weighted_stack(data, invVar)
    mean, std = np.mean(stacked), np.std(stacked)

    for i in range(n_comp):
        go = time.time()
        if verbose:
            print(f'Fitting component {i+1}/{n_comp} ... ', end='')
        if i == 0:
            A_init = np.random.rand(n_exp, 1) * 2
            W_init = stacked[None, :]
            A, W = nmf_basic(data, invVar, A_init, W_init, n_iter=n_iter, verbose=verbose)
        else:
            A *= (1-init_amp_ratio)
            A_init = np.random.rand(n_exp, 1) * 2
            W_init = mean + np.random.rand(1, n_pix) * std * init_amp_ratio
            A_init, W_init = nmf_basic(np.clip(data - A @ W, 0, None), invVar, A_init, W_init, n_iter=20)
            # W_init += np.random.rand(1, n_pix) * np.std(W_init) * 1e-3

            A_init = np.hstack((A, A_init))
            W_init = np.vstack((W, W_init))
            A, W = nmf_basic(data, invVar, A_init, W_init, n_iter=n_iter, verbose=verbose)

        A_med = np.median(A, axis=0)
        H, A = W * A_med[:, None], A / A_med[None, :]
            
    return A, H


def _solve_amps_nnls(data, invVar, W):
    """Solve NNLS for a single 1D exposure."""
    import scipy.optimize
    n_data = data.shape[0]
    A = np.zeros((n_data, W.shape[0]))
    for i in range(data.shape[0]):
        Xe = W * (invVar[None, i, :]) ** 0.5
        Ye = data[i] * (invVar[i, :]) ** 0.5
        A[i] = scipy.optimize.nnls(Xe.T, Ye)[0]
    return A

def nmf_iterative(data, invVar, n_comp, n_iter=100, init_amp_ratio=0.1, outliers=None, verbose=False):
    idx = np.arange(data.shape[0])
    if outliers is not None:
        idx = np.delete(idx, outliers)
    A, W = _nmf_iterative(data[idx], invVar[idx], n_comp, n_iter=n_iter, init_amp_ratio=init_amp_ratio, verbose=verbose)
    A_all = np.zeros((data.shape[0], n_comp))
    A_all[idx] = A
    if outliers is not None:
        A_all[outliers] = _solve_amps_nnls(data[outliers], invVar[outliers], W)
    return A_all, W

def nmf_iterative_correct1f(data, invVar, n_comp, n_iter=100, 
                            outliers=None, verbose=False, n_iter_1f=5):
    """data: 2D array, shape (n_exp, n_pix)"""
    from preprocess import correct_1f_noise

    n_exp, n_pix = data.shape
    data_corrected = np.zeros_like(data)
    stripes_all = np.zeros_like(data)
    
    for i in range(n_exp):
        stripes = correct_1f_noise(data[i].reshape(1, -1), invVar[i].reshape(1, -1))
        data_corrected[i] = data[i] - stripes
        stripes_all[i] = stripes

    A, W = nmf_iterative(data_corrected, invVar, n_comp, n_iter=n_iter, outliers=outliers, verbose=verbose)
    return A, W, stripes_all

def vectorize_data(data, err, mask, valid_region):
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


def identify_outliers(A, std_threshold=8):
    outliers = []
    for i in range(A.shape[1]):
        _, median, std = sigma_clipped_stats(A[:, i], maxiters=3)
        mask = np.abs(A[:, i] - median) > std_threshold * std
        outliers.append(np.where(mask)[0])
    if len(outliers) == 0:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(outliers)).astype(int)