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

def test_run_search():
    from find_asteroids.search import run_search
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as tmpdir:
        run_search(
            "docs/notebooks/catalog.ecsv", 
            "docs/notebooks/psfs.ecsv",
            [0.1, 0.5],
            [0, 359.99],
            10,
            10,
            Path(tmpdir) / "results"
        )

def test_run_search_mlflow():
    from find_asteroids.search import run_search_mlflow
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as tmpdir:
        run_search_mlflow(
            "experiment",
            "docs/notebooks/catalog.ecsv", 
            "docs/notebooks/psfs.ecsv",
            [0.1, 0.5],
            [0, 359.99],
            10,
            10,
            Path(tmpdir) / "results"
        )
