import numpy as np
import numba

@numba.njit()
def project_xyz(x: np.float64, y: np.float64, t: np.float64, vx: np.float64, vy: np.float64, reference_time: np.float64):
    x_prime = x - vx * (t - reference_time)
    y_prime = y - vy * (t - reference_time)
    return x_prime, y_prime

@numba.njit()
def digitize_point(x, y, min_x, min_y, dx, dy):
    return int((x - min_x)/dx), int((y - min_y)/dy)
