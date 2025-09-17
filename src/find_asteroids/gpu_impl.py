from numba import cuda
import numpy as np
from .primitives import project_xyz, digitize_point

@cuda.jit
def _hough_max(result, hough):
    """Find the maximum value in values and store in result[0]"""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    i = (bid * bdim) + tid
    
    if i >= hough.shape[0]:
        return
    
    y = -1
    z = -1
    v = -np.inf
    for j in range(hough.shape[1]):
        for k in range(hough.shape[2]):
            val = hough[i, j, k]
            if val > v:
                v = val
                y = j
                z = k
    
    result[i][0] = y
    result[i][1] = z
    result[i][2] = v

def hough_max(hough):
    result = np.zeros((hough.shape[0], 3), dtype=np.int32)
    _hough_max[512, hough.shape[0] // 512 + 1](result, hough)
    i = result[:, 2].argmax()
    j = result[i, 0]
    k = result[i, 1]
    v = result[i, 2]
    return (i, j, k), v

@cuda.jit
def _projected_bounds(X, directions, reference_time, min_x_arr, max_x_arr, min_y_arr, max_y_arr):
    dir_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    n, d = X.shape
    num_dir = directions.shape[0]
    
    if dir_idx >= num_dir:
        return

    direction = directions[dir_idx]
    
    # Initialize local min/max
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf

    # Iterate over all points
    for i in range(n):
        p = X[i]
        x_prime, y_prime = project_xyz(p[0], p[1], p[2], direction[0], direction[1], reference_time)
        min_x = min(min_x, x_prime)
        max_x = max(max_x, x_prime)
        min_y = min(min_y, y_prime)
        max_y = max(max_y, y_prime)

    # Store results in global memory
    min_x_arr[dir_idx] = min_x
    max_x_arr[dir_idx] = max_x
    min_y_arr[dir_idx] = min_y
    max_y_arr[dir_idx] = max_y
    
def projected_bounds(X: np.ndarray, directions: np.ndarray, reference_time: np.float64):
    n, d = X.shape
    num_dir = directions.shape[0]

    # Allocate device arrays
    d_X = cuda.to_device(X)
    d_directions = cuda.to_device(directions)
    d_min_x_arr = cuda.device_array(num_dir, dtype=np.float64)
    d_max_x_arr = cuda.device_array(num_dir, dtype=np.float64)
    d_min_y_arr = cuda.device_array(num_dir, dtype=np.float64)
    d_max_y_arr = cuda.device_array(num_dir, dtype=np.float64)

    # Define CUDA thread/block config
    threads_per_block = 128
    blocks_per_grid = (num_dir + threads_per_block - 1) // threads_per_block

    # Launch kernel
    print(blocks_per_grid, threads_per_block)
    _projected_bounds[blocks_per_grid, threads_per_block](
        d_X, d_directions, reference_time, d_min_x_arr, d_max_x_arr, d_min_y_arr, d_max_y_arr
    )

    # Copy results back to host
    min_x_arr = d_min_x_arr.copy_to_host()
    max_x_arr = d_max_x_arr.copy_to_host()
    min_y_arr = d_min_y_arr.copy_to_host()
    max_y_arr = d_max_y_arr.copy_to_host()

    # Final reduction on CPU
    min_x = np.min(min_x_arr)
    max_x = np.max(max_x_arr)
    min_y = np.min(min_y_arr)
    max_y = np.max(max_y_arr)

    return min_x, max_x, min_y, max_y

@cuda.jit
def _make_bins(bins, X, directions, x_min, y_min, dx, dy, reference_time):
    dir_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    point_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    num_dir, n = directions.shape[0], X.shape[0]

    if dir_idx >= num_dir or point_idx >= n:
        return  # Out-of-bounds check

    direction = directions[dir_idx]
    p = X[point_idx]
    x_prime, y_prime = project_xyz(p[0], p[1], p[2], direction[0], direction[1], reference_time)
    x_idx, y_idx = digitize_point(x_prime, y_prime, x_min, y_min, dx, dy)
    bins[point_idx, dir_idx, 0] = x_idx
    bins[point_idx, dir_idx, 1] = y_idx


def make_bins(X, directions, x_min, y_min, dx, dy, reference_time):
    n, d = X.shape
    num_dir = directions.shape[0]
    bins = np.full((n, num_dir, 2), -1)

    # Transfer data to GPU
    d_bins = cuda.to_device(bins)
    d_X = cuda.to_device(X)
    d_directions = cuda.to_device(directions)

    # Configure GPU threads and blocks
    threads_per_block = (16, 16)  # Tunable parameters
    blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                       (n + threads_per_block[1] - 1) // threads_per_block[1])
    
    print(blocks_per_grid, threads_per_block)
    _make_bins[blocks_per_grid, threads_per_block](
        d_bins, d_X, d_directions, x_min, y_min, dx, dy, reference_time
    )

    # Copy back the updated Hough space
    return d_bins.copy_to_host()


@cuda.jit
def _vote_points(hough, X, directions, x_min, y_min, dx, dy, reference_time, coef):
    """
    GPU version of vote_points using CUDA parallelism.
    Each thread processes one (dir_idx, point_idx) pair.
    """
    dir_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    point_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    num_dir, n = directions.shape[0], X.shape[0]

    if dir_idx >= num_dir or point_idx >= n:
        return  # Out-of-bounds check

    direction = directions[dir_idx]
    p = X[point_idx]
    x_prime, y_prime = project_xyz(p[0], p[1], p[2], direction[0], direction[1], reference_time)
    x_idx, y_idx = digitize_point(x_prime, y_prime, x_min, y_min, dx, dy)

    if 0 <= x_idx < hough.shape[1] and 0 <= y_idx < hough.shape[2]:
        cuda.atomic.add(hough, (dir_idx, x_idx, y_idx), coef)  # Atomic update to avoid race conditions

@cuda.jit
def _vote_points_mask(hough, mask, X, directions, x_min, y_min, dx, dy, reference_time, coef):
    """
    GPU version of vote_points using CUDA parallelism.
    Each thread processes one (dir_idx, point_idx) pair.
    """
    dir_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    point_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    include = False
    for i in range(mask.shape[0]):
        if mask[i, point_idx]:
            include = True
    
    if not include:
        return

    num_dir, n = directions.shape[0], X.shape[0]

    if dir_idx >= num_dir or point_idx >= n:
        return  # Out-of-bounds check

    direction = directions[dir_idx]
    p = X[point_idx]
    x_prime, y_prime = project_xyz(p[0], p[1], p[2], direction[0], direction[1], reference_time)
    x_idx, y_idx = digitize_point(x_prime, y_prime, x_min, y_min, dx, dy)

    if 0 <= x_idx < hough.shape[1] and 0 <= y_idx < hough.shape[2]:
        cuda.atomic.add(hough, (dir_idx, x_idx, y_idx), coef)  # Atomic update to avoid race conditions


def vote_points(hough: np.ndarray, X: np.ndarray, directions: np.ndarray, x_min, y_min, dx, dy, reference_time, coef: np.float64 = 1):
    """
    Host function to launch the GPU kernel.
    """
    n, d = X.shape
    num_dir = directions.shape[0]

    # Transfer data to GPU
    d_hough = cuda.to_device(hough)
    d_X = cuda.to_device(X)
    d_directions = cuda.to_device(directions)

    # Configure GPU threads and blocks
    threads_per_block = (16, 16)  # Tunable parameters
    blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                       (n + threads_per_block[1] - 1) // threads_per_block[1])

    print(blocks_per_grid, threads_per_block)
    _vote_points[blocks_per_grid, threads_per_block](
        d_hough, d_X, d_directions, x_min, y_min, dx, dy, reference_time, coef
    )

    # Copy back the updated Hough space
    return d_hough.copy_to_host()

@cuda.jit
def _vote_bins(hough, bins, coef):
    """
    GPU version of vote_points using CUDA parallelism.
    Each thread processes one (dir_idx, point_idx) pair.
    """
    dir_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    point_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    n, num_dir, _ = bins.shape

    if dir_idx >= num_dir or point_idx >= n:
        return  # Out-of-bounds check

    x_idx = bins[point_idx, dir_idx, 0]
    y_idx = bins[point_idx, dir_idx, 1]

    if 0 <= x_idx < hough.shape[1] and 0 <= y_idx < hough.shape[2]:
        cuda.atomic.add(hough, (dir_idx, x_idx, y_idx), coef)  # Atomic update to avoid race conditions


def vote_bins(hough: np.ndarray, bins: np.ndarray, coef: np.float64 = 1):
    """
    Host function to launch the GPU kernel.
    """
    n, num_dir, _ = bins.shape

    # Transfer data to GPU
    d_hough = cuda.to_device(hough)
    d_bins = cuda.to_device(bins)

    # Configure GPU threads and blocks
    threads_per_block = (16, 16)  # Tunable parameters
    blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                       (n + threads_per_block[1] - 1) // threads_per_block[1])

    print(blocks_per_grid, threads_per_block)
    _vote_bins[blocks_per_grid, threads_per_block](
        d_hough, d_bins, coef
    )

    # Copy back the updated Hough space
    return d_hough.copy_to_host()

@cuda.jit
def _find_voters_bins(bins, mask, mask_dir, mask_x, mask_y):
    dir_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if dir_idx != mask_dir:
        return
    
    point_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    n, num_dir, _ = bins.shape

    if dir_idx >= num_dir or point_idx >= n:
        return  # Out-of-bounds check

    x_idx = bins[point_idx, dir_idx, 0]
    y_idx = bins[point_idx, dir_idx, 1]

    if x_idx == mask_x and y_idx == mask_y:
        mask[dir_idx, point_idx] = True

def find_voters_bins(hough: np.ndarray, bins: np.ndarray, mask_dir, mask_x, mask_y):
    n, num_dir, _ = bins.shape

    mask = np.full((num_dir, n), False)

    # Transfer data to GPU
    d_bins = cuda.to_device(bins)
    d_mask = cuda.to_device(mask)

    # Configure GPU threads and blocks
    threads_per_block = (16, 16)  # Tunable parameters
    blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                       (n + threads_per_block[1] - 1) // threads_per_block[1])

    print(blocks_per_grid, threads_per_block)
    _find_voters_bins[blocks_per_grid, threads_per_block](
        d_bins, d_mask, mask_dir, mask_x, mask_y
    )

    return np.logical_or.reduce(d_mask.copy_to_host(), axis=0)

@cuda.jit
def _find_voters_points(X, directions, mask, x_min, y_min, dx, dy, reference_time, mask_dir, mask_x, mask_y):
    dir_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if dir_idx != mask_dir:
        return
    
    point_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    n, _ = X.shape
    num_dir = directions.shape[0]

    if dir_idx >= num_dir or point_idx >= n:
        return  # Out-of-bounds check

    direction = directions[dir_idx]
    p = X[point_idx]
    x_prime, y_prime = project_xyz(p[0], p[1], p[2], direction[0], direction[1], reference_time)
    x_idx, y_idx = digitize_point(x_prime, y_prime, x_min, y_min, dx, dy)

    if x_idx == mask_x and y_idx == mask_y:
        mask[dir_idx, point_idx] = True


def find_voters_points(hough, X, directions, x_min, y_min, dx, dy, reference_time, mask_dir, mask_x, mask_y):
    n, d = X.shape
    num_dir = directions.shape[0]
    mask = np.full((num_dir, n), False)
    # Transfer data to GPU
    d_X = cuda.to_device(X)
    d_directions = cuda.to_device(directions)
    d_mask = cuda.to_device(mask)

    # Configure GPU threads and blocks
    threads_per_block = (16, 16)  # Tunable parameters
    blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                       (n + threads_per_block[1] - 1) // threads_per_block[1])

    print(blocks_per_grid, threads_per_block)
    _find_voters_points[blocks_per_grid, threads_per_block](
        d_X, d_directions, d_mask, x_min, y_min, dx, dy, reference_time, mask_dir, mask_x, mask_y
    )

    return np.logical_or.reduce(d_mask.copy_to_host(), axis=0)


