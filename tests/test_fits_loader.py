"""Tests for the image-based (FITS) reference implementation.

Covers:
- load_fits_images (loader.py)
- search() with coef_index parameter (search.py)
- search_image() (search.py)
"""

import numpy as np
import pytest
import astropy.units as u
import astropy.table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_fits(tmp_path, filename, ra_center, dec_center, time_mjd, nx=10, ny=10, seed=0):
    """Create a minimal FITS image with a TAN WCS and an MJD-OBS keyword."""
    from astropy.io import fits
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    wcs.wcs.cdelt = [-0.001, 0.001]           # 3.6 arcsec per pixel
    wcs.wcs.crval = [ra_center, dec_center]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]

    header = wcs.to_header()
    header["MJD-OBS"] = time_mjd

    rng = np.random.default_rng(seed)
    data = rng.uniform(0, 100, size=(ny, nx)).astype(np.float32)

    hdu = fits.PrimaryHDU(data=data, header=header)
    path = tmp_path / filename
    hdu.writeto(str(path), overwrite=True)
    return path


# ---------------------------------------------------------------------------
# Tests for load_fits_images
# ---------------------------------------------------------------------------

def test_load_fits_images_single(tmp_path):
    """A single FITS file should produce nx*ny rows with correct columns."""
    from find_asteroids.loader import load_fits_images
    from astropy.io import fits
    from astropy.wcs import WCS

    nx, ny = 5, 4
    path = _make_test_fits(tmp_path, "img.fits", ra_center=10.0, dec_center=20.0,
                           time_mjd=59000.0, nx=nx, ny=ny)

    catalog = load_fits_images([path])

    assert len(catalog) == nx * ny
    assert set(catalog.colnames) >= {"ra", "dec", "time", "flux"}
    assert catalog["ra"].unit == u.deg
    assert catalog["dec"].unit == u.deg
    assert catalog["time"].unit == u.day
    # All rows should have the same MJD
    assert np.allclose(catalog["time"].value, 59000.0)
    # Flux values should match the image data (flat-ordered)
    with fits.open(str(path)) as hdul:
        expected_flux = hdul[0].data.ravel().astype(float)
    np.testing.assert_allclose(catalog["flux"], expected_flux)


def test_load_fits_images_multiple(tmp_path):
    """Multiple FITS files should be concatenated into a single catalog."""
    from find_asteroids.loader import load_fits_images

    nx, ny = 4, 3
    paths = [
        _make_test_fits(tmp_path, "img0.fits", 10.0, 20.0, 59000.0, nx=nx, ny=ny, seed=1),
        _make_test_fits(tmp_path, "img1.fits", 10.0, 20.0, 59001.0, nx=nx, ny=ny, seed=2),
    ]

    catalog = load_fits_images(paths)

    assert len(catalog) == 2 * nx * ny
    times = np.unique(catalog["time"].value)
    assert set(times) == {59000.0, 59001.0}


def test_load_fits_images_date_obs(tmp_path):
    """load_fits_images should accept DATE-OBS instead of MJD-OBS."""
    from find_asteroids.loader import load_fits_images
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.time

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [5.5, 5.5]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crval = [15.0, 25.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    # Use DATE-OBS instead of MJD-OBS
    header["DATE-OBS"] = "2020-01-01T00:00:00.0"

    rng = np.random.default_rng(42)
    data = rng.uniform(0, 10, (10, 10)).astype(np.float32)
    hdu = fits.PrimaryHDU(data=data, header=header)
    path = tmp_path / "date_obs.fits"
    hdu.writeto(str(path), overwrite=True)

    catalog = load_fits_images([path])

    expected_mjd = astropy.time.Time("2020-01-01T00:00:00.0").mjd
    np.testing.assert_allclose(catalog["time"].value, expected_mjd)


def test_load_fits_images_no_time_raises(tmp_path):
    """A FITS file without time info should raise ValueError."""
    from find_asteroids.loader import load_fits_images
    from astropy.io import fits
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [5.5, 5.5]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crval = [15.0, 25.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    # Intentionally omit time keywords

    data = np.ones((10, 10), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data, header=header)
    path = tmp_path / "no_time.fits"
    hdu.writeto(str(path), overwrite=True)

    with pytest.raises(ValueError, match="No time information"):
        load_fits_images([path])


# ---------------------------------------------------------------------------
# Tests for search() with coef_index
# ---------------------------------------------------------------------------

def test_search_with_coef_index(tmp_path):
    """search() should accept coef_index and weight votes by X[:, coef_index]."""
    from find_asteroids.search import search
    from find_asteroids.directions import SearchDirections

    catalog = astropy.table.Table.read("docs/notebooks/catalog.ecsv")
    flux = np.ones(len(catalog))
    X = np.array([catalog["ra"], catalog["dec"], catalog["time"], flux]).T

    dx = 10 * u.arcsec
    dt = (X[:, 2].max() - X[:, 2].min()) * u.day
    directions = SearchDirections(
        [0.1 * u.deg / u.day, 0.2 * u.deg / u.day],
        [0 * u.deg, 180 * u.deg],
        dx,
        dt,
    )

    results, results_points = search(
        X, directions, dx, X[:, 2].min(), num_results=3, coef_index=3
    )

    assert results.shape == (3, 4)
    assert len(results_points) == 3
    for pts in results_points:
        assert pts.shape[1] == 4  # X has 4 columns


def test_search_with_coef_index_precompute(tmp_path):
    """search() with coef_index should also work in precompute mode."""
    from find_asteroids.search import search
    from find_asteroids.directions import SearchDirections

    catalog = astropy.table.Table.read("docs/notebooks/catalog.ecsv")
    flux = np.ones(len(catalog))
    X = np.array([catalog["ra"], catalog["dec"], catalog["time"], flux]).T

    dx = 10 * u.arcsec
    dt = (X[:, 2].max() - X[:, 2].min()) * u.day
    directions = SearchDirections(
        [0.1 * u.deg / u.day, 0.2 * u.deg / u.day],
        [0 * u.deg, 180 * u.deg],
        dx,
        dt,
    )

    results, results_points = search(
        X, directions, dx, X[:, 2].min(), num_results=3, precompute=True, coef_index=3
    )

    assert results.shape == (3, 4)
    assert len(results_points) == 3


# ---------------------------------------------------------------------------
# Tests for search_image()
# ---------------------------------------------------------------------------

def test_search_image(tmp_path):
    """search_image() should load FITS files and return search results."""
    from find_asteroids.search import search_image

    # Create a pair of small FITS images with widely-separated times so that
    # SearchDirections generates at least one trial velocity.
    paths = [
        _make_test_fits(tmp_path, "s0.fits", 10.0, 20.0, 59000.0, nx=8, ny=8, seed=3),
        _make_test_fits(tmp_path, "s1.fits", 10.0, 20.0, 59010.0, nx=8, ny=8, seed=4),
    ]

    results, results_points = search_image(
        paths,
        velocity_range=[0.01 * u.deg / u.day, 0.05 * u.deg / u.day],
        angle_range=[0 * u.deg, 360 * u.deg],
        num_results=2,
    )

    assert results.shape == (2, 4)
    assert len(results_points) == 2
    # Each returned point set should contain 4-column rows [ra, dec, time, flux]
    for pts in results_points:
        assert pts.ndim == 2
        assert pts.shape[1] == 4
