import numpy as np
import astropy.units as u

class SearchDirections(object):
    def __init__(self, velocity_range, angle_range, dx, dt, b=None):
        """
        specify two ranges and units
        """
        if b is not None:
            self._b = b
            _v = ((b**2).sum(axis=1)**0.5)
            _phi = np.arctan2(b[:, 1], b[:, 0]) # this provides values in [-pi, pi]
            self.velocity_range = [_v.min(), _v.max()] * b.unit
            self.angle_range = [_phi.min(), _phi.max()] * u.rad
        else:
            self.velocity_range = velocity_range
            # self.angle_range = angle_range
            # fit into [-pi, pi]
            self.angle_range = [
                (angle_range[0] + np.pi*u.rad) % (2 * np.pi * u.rad) - np.pi * u.rad,
                (angle_range[1] + np.pi*u.rad) % (2 * np.pi * u.rad) - np.pi * u.rad
            ]
        
        self.dx = dx
        self.dt = dt

    @property
    def v_min(self):
        return min(self.velocity_range)
    
    @property
    def v_max(self):
        return max(self.velocity_range)
        
    @property
    def phi_min(self):
        return min(self.angle_range)
    
    @property
    def phi_max(self):
        return max(self.angle_range)

        
    @property
    def b(self):
        if not hasattr(self, "_b"):
            _min = (-self.v_max * self.dt/self.dx).to(u.dimensionless_unscaled)
            _max = (self.v_max * self.dt/self.dx).to(u.dimensionless_unscaled)
            _v_x = np.arange(_min, _max + 1, 1)
            _v_y = np.arange(_min, _max + 1, 1)
            v_x_m, v_y_m = np.meshgrid(_v_x, _v_y)
            b = (np.vstack([v_x_m.flatten(), v_y_m.flatten()]).T * (self.dx/self.dt)).to(self.v_max.unit)
            _v = ((b**2).sum(axis=1)**0.5)
            _phi = np.arctan2(b[:, 1], b[:, 0]) # this provides values in [-pi, pi]
            
            v_mask = (_v >= self.v_min) & (_v <= self.v_max)
            # how do I slice a region of [-pi, pi] that resepcts [phi_min, phi_max]?
            if (self.angle_range[1] < self.angle_range[0]) and (self.angle_range[0] < 180*u.deg):
                phi_mask = (_phi >= self.angle_range[0]) | (_phi <= self.angle_range[1]) 
            else:
                phi_mask = (_phi >= self.angle_range[0]) & (_phi <= self.angle_range[1])

            b_m = b.reshape(len(_v_x), len(_v_y), 2)
            adjacent_dx_width = (b_m - np.roll(np.roll(b_m, 1, axis=0), 1, axis=1))[1:, 1:] * self.dt/self.dx
            assert(np.allclose(adjacent_dx_width[~np.isnan(adjacent_dx_width)], 1)) # all adjacent velocities are within 1 dx over timespan of dt
            self._b = b[v_mask & phi_mask]
        return self._b

