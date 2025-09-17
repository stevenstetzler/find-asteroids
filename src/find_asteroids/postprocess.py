import logging
import numpy as np
from mcd_regression import MCDRegression
import sys
import astropy.table

logging.basicConfig()
log = logging.getLogger(__name__)

def refine(points):
    x, y = points[:, 2][:, None], points[:, :2]
    log.info(f"refining cluster with {len(points)} points")
    if len(points) < 2:
        log.warn("cluster has too few points to fit a line")
        return None
    try:
        try:
            mcdr = MCDRegression()
            mcdr.fit(x, y)
        except AssertionError as e:
            log.warn("regression on points in cluster failed %s", e)
            return None
    except Exception as e:
        log.exception(e)
        return None
    
    regression_error = (np.diag(mcdr.sigma_e)**2).sum()
    log.debug(f"regression error {regression_error}")
    outliers = mcdr.outliers_r
    inliers = ~outliers
    log.info(f"refined cluster has {inliers.sum()} inliers")

    return mcdr

def gather(mcdr : MCDRegression, ra, dec, time, threshold):
    """
    gather all points within some threshold of a line
    """
    y_pred = mcdr.predict(time[:, None])
    residuals = (
        np.array([ra, dec]).T - 
        y_pred
    )
    distance = (residuals**2).sum(axis=1)**0.5
    mask = distance < threshold
    log.info("gathered %s points", mask.sum())
    return mask