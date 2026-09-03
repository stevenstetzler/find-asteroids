def test_search():
    from find_asteroids.search import search
    from find_asteroids.directions import SearchDirections
    import astropy.table
    import astropy.units as u
    import numpy as np
    catalog = astropy.table.Table.read("docs/notebooks/catalog.ecsv")
    X = np.array([catalog['ra'], catalog['dec'], catalog['time']]).T
    dx = 10 * u.arcsec
    dt = (X[:, 2].max() - X[:, 2].min())*u.day
    directions = SearchDirections([0.1 * u.deg/u.day, 0.2 * u.deg/u.day], [0 * u.deg, 180 * u.deg], dx, dt)
    search(X, directions, dx, X[:, 2].min())


def _run_main(argv):
    import sys
    from find_asteroids.search import main
    old_argv = sys.argv
    sys.argv = ["find-asteroids"] + argv
    try:
        main()
    finally:
        sys.argv = old_argv


def test_normalize_time_assumes_utc_and_warns(caplog):
    """A plain Quantity/Column 'time' column (no attached scale) is assumed
    to be MJD, UTC -- and this assumption must be logged, since it can't be
    verified from the data itself."""
    from find_asteroids.search import normalize_time
    from astropy.time import Time
    import astropy.table
    import numpy as np

    col = astropy.table.Column([58576.216, 58576.401], unit='d', name='time')
    with caplog.at_level("WARNING"):
        t = normalize_time(col)

    assert isinstance(t, Time)
    assert t.scale == 'tai'
    expected = Time(col.data, format='mjd', scale='utc').tai
    assert np.allclose(t.mjd, expected.mjd)
    assert any("assum" in rec.message.lower() for rec in caplog.records)


def test_normalize_time_uses_existing_scale_without_warning(caplog):
    """A real Time-typed 'time' column's own scale is used directly, with
    no assumption made and no warning logged."""
    from find_asteroids.search import normalize_time
    from astropy.time import Time
    import numpy as np

    t_in = Time([58576.216, 58576.401], format='mjd', scale='tt')
    with caplog.at_level("WARNING"):
        t = normalize_time(t_in)

    assert t.scale == 'tai'
    assert np.allclose(t.mjd, t_in.tai.mjd)
    assert not any("assum" in rec.message.lower() for rec in caplog.records)


def test_main_wraps_ra_and_scales_time(tmp_path):
    """Regression test for the units/scale normalization as a whole: after
    a real search run, every output table's ra is in [0, 360) degrees, and
    every time value round-trips as an astropy Time in the TAI scale."""
    import astropy.table
    from astropy.time import Time

    results_dir = tmp_path / "results"
    _run_main([
        "--catalog", "docs/notebooks/catalog.ecsv",
        "--psfs", "docs/notebooks/psfs.ecsv",
        "--velocity", "0.1", "0.5",
        "--angle", "0", "359.99",
        "--dx", "10",
        "--num-results", "3",
        "--results-dir", str(results_dir),
    ])

    for name in ["gathered", "points", "tracklet"]:
        t = astropy.table.Table.read(results_dir / "0" / f"{name}.ecsv")
        ra_cols = [c for c in t.colnames if c == "ra" or c.startswith("ra_")]
        for c in ra_cols:
            assert (t[c] >= 0).all() and (t[c] < 360).all(), f"{name}.{c} not wrapped to [0, 360)"

        time_cols = [c for c in t.colnames if c == "time" or c in ("tref", "tmin", "tmax")]
        for c in time_cols:
            assert isinstance(t[c], Time), f"{name}.{c} did not round-trip as an astropy Time"
            assert t[c].scale == 'tai', f"{name}.{c} is not in the TAI scale"


def test_main_catalog_in_radians_matches_degrees(tmp_path):
    """Regression test for the original bug: a catalog with ra/dec given in
    radians (as deep_asteroids' real catalogs are) must produce the same
    search result as the same physical catalog given in degrees. Before the
    unit-normalization fix, X was built directly from whatever raw numbers
    the columns held, so a radians catalog was silently misinterpreted as
    degrees.
    """
    import astropy.table
    import astropy.units as u
    import numpy as np

    catalog_deg = astropy.table.Table.read("docs/notebooks/catalog.ecsv")
    catalog_rad = catalog_deg.copy()
    catalog_rad['ra'] = catalog_rad['ra'].to(u.rad)
    catalog_rad['dec'] = catalog_rad['dec'].to(u.rad)
    catalog_rad_path = tmp_path / "catalog_rad.ecsv"
    catalog_rad.write(catalog_rad_path)

    common = [
        "--psfs", "docs/notebooks/psfs.ecsv",
        "--velocity", "0.1", "0.5",
        "--angle", "0", "359.99",
        "--dx", "10",
        "--num-results", "3",
    ]
    np.random.seed(0)
    _run_main(["--catalog", "docs/notebooks/catalog.ecsv", "--results-dir", str(tmp_path / "results_deg")] + common)
    np.random.seed(0)
    _run_main(["--catalog", str(catalog_rad_path), "--results-dir", str(tmp_path / "results_rad")] + common)

    t_deg = astropy.table.Table.read(tmp_path / "results_deg" / "0" / "tracklet.ecsv")
    t_rad = astropy.table.Table.read(tmp_path / "results_rad" / "0" / "tracklet.ecsv")

    assert abs(t_deg["ra_0"][0] - t_rad["ra_0"][0]) < 1e-6
    assert abs(t_deg["dec_0"][0] - t_rad["dec_0"][0]) < 1e-6

