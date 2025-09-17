import numpy as np
import numba
from .primitives import project_xyz, digitize_point

@numba.njit(parallel=True)
def projected_bounds(X: np.ndarray, directions: np.ndarray, reference_time: np.float64):
    n, d = X.shape
    num_dir = directions.shape[0]
    min_x_arr = np.full(num_dir, np.inf)
    max_x_arr = np.full(num_dir, -np.inf)
    min_y_arr = np.full(num_dir, np.inf)
    max_y_arr = np.full(num_dir, -np.inf)
    
    for dir_idx in numba.prange(num_dir): # for each direction
        if dir_idx < num_dir:
            direction = directions[dir_idx]
            for i in range(n): # for each point
                p = X[i]
                x_prime, y_prime = project_xyz(p[0], p[1], p[2], direction[0], direction[1], reference_time)
                min_x_arr[dir_idx] = min(min_x_arr[dir_idx], x_prime)
                max_x_arr[dir_idx] = max(max_x_arr[dir_idx], x_prime)
                min_y_arr[dir_idx] = min(min_y_arr[dir_idx], y_prime)
                max_y_arr[dir_idx] = max(max_y_arr[dir_idx], y_prime)

    min_x = np.min(min_x_arr)
    max_x = np.max(max_x_arr)
    min_y = np.min(min_y_arr)
    max_y = np.max(max_y_arr)
    
    return min_x, max_x, min_y, max_y


def hough_max(hough):
    idx = np.unravel_index(hough.argmax(), hough.shape)
    return idx, hough[idx]

@numba.njit(parallel=True)
def make_bins(X: np.ndarray, directions: np.ndarray, x_min, y_min, dx, dy, reference_time):
    n, d = X.shape
    num_dir = directions.shape[0]
    bins = np.full((n, num_dir, 2), -1)
    for dir_idx in numba.prange(num_dir): # for each direction
        if dir_idx < num_dir:
            direction = directions[dir_idx]
            for i in range(n): # for each point
                p = X[i]
                x_prime, y_prime = project_xyz(p[0], p[1], p[2], direction[0], direction[1], reference_time)
                x_idx, y_idx = digitize_point(x_prime, y_prime, x_min, y_min, dx, dy)
                bins[i, dir_idx, 0] = x_idx
                bins[i, dir_idx, 1] = y_idx
    return bins

@numba.njit(parallel=True)
def vote_points(hough: np.ndarray, X: np.ndarray, directions: np.ndarray, x_min, y_min, dx, dy, reference_time, coef):
    n, d = X.shape
    num_dir = directions.shape[0]
    for dir_idx in numba.prange(num_dir): # for each direction
        if dir_idx < num_dir:
            direction = directions[dir_idx]
            for i in range(n): # for each point
                p = X[i]
                x_prime, y_prime = project_xyz(p[0], p[1], p[2], direction[0], direction[1], reference_time)
                x_idx, y_idx = digitize_point(x_prime, y_prime, x_min, y_min, dx, dy)
                if 0 <= x_idx < hough.shape[1] and 0 <= y_idx < hough.shape[2]:
                    hough[dir_idx, x_idx, y_idx] += coef
    return hough

@numba.njit(parallel=True)
def vote_bins(hough: np.ndarray, bins: np.ndarray, coef):
    n, num_dir, _ = bins.shape
    for dir_idx in numba.prange(num_dir): # for each direction
        if dir_idx < num_dir:
            for i in range(n): # for each point
                x_idx = bins[i, dir_idx, 0]
                y_idx = bins[i, dir_idx, 1]
                if 0 <= x_idx < hough.shape[1] and 0 <= y_idx < hough.shape[2]:
                    hough[dir_idx, x_idx, y_idx] += coef
    return hough

@numba.njit(parallel=True)
def find_voters_bins(hough: np.ndarray, bins: np.ndarray, mask_dir, mask_x, mask_y):
    n, num_dir, _ = bins.shape
    
    num_threads = numba.get_num_threads()
    mask = np.full((num_threads, n), False)
    _mask = np.full(n, False)
    for dir_idx in numba.prange(num_dir): # for each direction
        thread_idx = dir_idx % num_threads
        if dir_idx < num_dir:
            for i in range(n): # for each point
                x_idx = bins[i, dir_idx, 0]
                y_idx = bins[i, dir_idx, 1]                
                if 0 <= x_idx < hough.shape[1] and 0 <= y_idx < hough.shape[2]:
                    if dir_idx == mask_dir and x_idx == mask_x and y_idx == mask_y:
                        mask[thread_idx, i] |= True
    for i in range(n):
        for dir_idx in range(num_dir):
            thread_idx = dir_idx % num_threads
            _mask[i] |= mask[thread_idx, i]
            
    return _mask

@numba.njit(parallel=True)
def find_voters_points(hough: np.ndarray, X: np.ndarray, directions: np.ndarray, x_min, y_min, dx, dy, reference_time, mask_dir, mask_x, mask_y):
    n, d = X.shape
    num_dir = directions.shape[0]
    num_threads = numba.get_num_threads()
    
    mask = np.full((num_threads, n), False)
    _mask = np.full(n, False)
    for dir_idx in numba.prange(num_dir): # for each direction
        thread_idx = dir_idx % num_threads
        if dir_idx < num_dir:
            direction = directions[dir_idx]
            for i in range(n): # for each point
                p = X[i]
                x_prime, y_prime = project_xyz(p[0], p[1], p[2], direction[0], direction[1], reference_time)
                x_idx, y_idx = digitize_point(x_prime, y_prime, x_min, y_min, dx, dy)
                if 0 <= x_idx < hough.shape[1] and 0 <= y_idx < hough.shape[2]:
                    if dir_idx == mask_dir and x_idx == mask_x and y_idx == mask_y:
                        mask[thread_idx, i] = True
    for i in range(n):
        for thread_idx in range(num_threads):
            _mask[i] |= mask[thread_idx, i]
            
    return _mask

