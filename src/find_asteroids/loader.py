import numpy as np
import astropy.table
import astropy.units as u
import astropy.time


def load_fits_images(fits_files):
    """Load a list of FITS image files into a detection catalog.

    Each pixel in each image becomes a row in the resulting catalog.  The
    on-sky position of every pixel center is computed using the WCS stored in
    the FITS header, and the observation time is read from the standard FITS
    keywords ``MJD-OBS`` (preferred) or ``DATE-OBS``.

    Parameters
    ----------
    fits_files : list of str or pathlib.Path
        Paths to FITS image files.  Each file must contain a valid 2-D WCS
        and at least one of the time keywords described above.

    Returns
    -------
    astropy.table.Table
        Detection catalog with the following columns:

        ``ra``
            Right ascension of each pixel center (deg).
        ``dec``
            Declination of each pixel center (deg).
        ``time``
            Observation time in Modified Julian Date (MJD, days).
        ``flux``
            Pixel flux value (dimensionless; units depend on the image).

    Raises
    ------
    ValueError
        If a FITS file contains no image data or no recognizable time keyword.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    all_ra = []
    all_dec = []
    all_time = []
    all_flux = []

    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            # Find the first HDU that contains image data.
            ext = 0
            while ext < len(hdul) and hdul[ext].data is None:
                ext += 1
            if ext >= len(hdul):
                raise ValueError(f"No image data found in {fits_file}")

            header = hdul[ext].header
            data = hdul[ext].data

            # Build a 2-D WCS from the header.
            wcs = WCS(header, naxis=2)

            # Extract observation time as MJD (days).
            if "MJD-OBS" in header:
                time_val = float(header["MJD-OBS"])
            elif "DATE-OBS" in header:
                t = astropy.time.Time(header["DATE-OBS"])
                time_val = t.mjd
            else:
                raise ValueError(
                    f"No time information (MJD-OBS or DATE-OBS) found in {fits_file}"
                )

            # Collapse any extra dimensions (e.g. spectral or Stokes axes).
            while data.ndim > 2:
                data = data[0]

            ny, nx = data.shape

            # Pixel-centre coordinates (0-indexed, following the FITS/astropy
            # convention used by pixel_to_world).
            x_pix, y_pix = np.meshgrid(
                np.arange(nx, dtype=float), np.arange(ny, dtype=float)
            )

            # Convert pixel centres to sky coordinates.
            sky = wcs.pixel_to_world(x_pix.ravel(), y_pix.ravel())

            all_ra.append(sky.ra.deg)
            all_dec.append(sky.dec.deg)
            all_time.append(np.full(nx * ny, time_val))
            all_flux.append(data.ravel().astype(float))

    catalog = astropy.table.Table(
        [
            astropy.table.Column(np.concatenate(all_ra), name="ra", unit=u.deg),
            astropy.table.Column(np.concatenate(all_dec), name="dec", unit=u.deg),
            astropy.table.Column(
                np.concatenate(all_time), name="time", unit=u.day
            ),
            astropy.table.Column(np.concatenate(all_flux), name="flux"),
        ]
    )

    return catalog
