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


def test_compile_results_astropy():
    from find_asteroids.search import run_search
    from find_asteroids.results import compile_results_astropy
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        run_search(
            "docs/notebooks/catalog.ecsv",
            "docs/notebooks/psfs.ecsv",
            [0.1, 0.5],
            [0, 359.99],
            10,
            10,
            results_dir,
        )
        compiled = dict(compile_results_astropy(results_dir))
        assert set(compiled.keys()) == {"gathered", "result", "points", "tracklet"}
        assert len(compiled["result"]) == 10


def test_compile_results_db():
    from find_asteroids.search import run_search
    from find_asteroids.results import compile_results_db
    from find_asteroids.models import Result, Search, Gathered
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    with TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        run_search(
            "docs/notebooks/catalog.ecsv",
            "docs/notebooks/psfs.ecsv",
            [0.1, 0.5],
            [0, 359.99],
            10,
            10,
            results_dir,
        )
        db_uri = f"sqlite:///{tmpdir}/results.db"
        compile_results_db(db_uri, results_dir, run_id="test-run-1")

        engine = create_engine(db_uri)
        with Session(engine) as session:
            searches = session.query(Search).all()
            assert len(searches) == 1
            assert searches[0].run_id == "test-run-1"

            results = session.query(Result).all()
            assert len(results) == 10
            assert {r.search.run_id for r in results} == {"test-run-1"}

            gathered = session.query(Gathered).all()
            assert len(gathered) > 0
            assert all(g.result.search.run_id == "test-run-1" for g in gathered)


def test_compile_results_db_stores_normalized_units(tmp_path):
    """The database's ra/time columns must hold find-asteroids' canonical
    convention (degrees in [0, 360), MJD in the TAI scale) regardless of
    what units/scale the input catalog used -- run_search() writes Time-
    typed columns to the intermediate result tables, and Time isn't a type
    a Float DB column can bind directly, so compile_results_db must convert
    it explicitly rather than pass it through."""
    from find_asteroids.search import run_search
    from find_asteroids.results import compile_results_db
    from find_asteroids.models import Gathered
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    import astropy.table
    import astropy.units as u

    # a catalog in radians, like deep_asteroids' real catalogs
    catalog = astropy.table.Table.read("docs/notebooks/catalog.ecsv")
    catalog['ra'] = catalog['ra'].to(u.rad)
    catalog['dec'] = catalog['dec'].to(u.rad)
    catalog_path = tmp_path / "catalog_rad.ecsv"
    catalog.write(catalog_path)

    results_dir = tmp_path / "results"
    run_search(catalog_path, "docs/notebooks/psfs.ecsv", [0.1, 0.5], [0, 359.99], 10, 10, results_dir)

    # cross-check against what run_search actually wrote to disk
    expected = astropy.table.Table.read(results_dir / "0" / "gathered.ecsv")

    db_uri = f"sqlite:///{tmp_path}/results.db"
    compile_results_db(db_uri, results_dir, run_id="units-test")

    engine = create_engine(db_uri)
    with Session(engine) as session:
        gathered = session.query(Gathered).filter_by(result_id=1).order_by(Gathered.id).all()
        assert len(gathered) == len(expected)
        for row, exp in zip(gathered, expected):
            assert 0 <= row.ra < 360
            exp_ra = getattr(exp['ra'], 'value', exp['ra'])
            assert abs(row.ra - exp_ra) < 1e-9
            assert abs(row.time - exp['time'].tai.mjd) < 1e-9


def test_compile_results_db_catalog_with_id_column():
    """Regression test: a detection catalog with its own 'id' column (e.g.
    a per-image detection id, as deep_asteroids' catalogs have) must not
    collide with Gathered/Points' own primary key column, also named 'id'.
    """
    from find_asteroids.search import run_search
    from find_asteroids.results import compile_results_db
    from find_asteroids.models import Gathered
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    import astropy.table

    catalog = astropy.table.Table.read("docs/notebooks/catalog.ecsv")
    # a non-unique 'id' column, like a per-image detection id that restarts
    # for each exposure -- this is what triggered the collision.
    catalog["id"] = [i % 50 for i in range(len(catalog))]
    catalog["detector"] = 5
    catalog["visit"] = [845580 + i % 5 for i in range(len(catalog))]

    with TemporaryDirectory() as tmpdir:
        catalog_path = Path(tmpdir) / "catalog.ecsv"
        catalog.write(catalog_path)

        results_dir = Path(tmpdir) / "results"
        run_search(
            catalog_path,
            "docs/notebooks/psfs.ecsv",
            [0.1, 0.5],
            [0, 359.99],
            10,
            10,
            results_dir,
        )
        db_uri = f"sqlite:///{tmpdir}/results.db"
        compile_results_db(db_uri, results_dir, run_id="test-run-1")  # must not raise IntegrityError

        engine = create_engine(db_uri)
        with Session(engine) as session:
            gathered = session.query(Gathered).all()
            assert len(gathered) > 0
            # the DB's own primary keys stay unique/sequential...
            assert sorted(g.id for g in gathered) == list(range(1, len(gathered) + 1))
            # ...while the source catalog's 'id'/'detector'/'visit' survive in
            # `extra`, stored (and read back) as a plain dict -- not a
            # JSON-encoded string requiring a second json.loads() to unpack.
            extra = gathered[0].extra
            assert isinstance(extra, dict)
            assert {"id", "detector", "visit"} <= extra.keys()


def test_main_results_dir_defaults_to_tempdir_with_results_db_uri():
    """--results-dir should default to a (cleaned-up) temporary directory
    when --results-db-uri is given, since it's pure intermediate scratch on
    the way into the database."""
    import sys
    from find_asteroids.search import main
    from find_asteroids.models import Result, Search
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    with TemporaryDirectory() as tmpdir:
        db_uri = f"sqlite:///{tmpdir}/results.db"
        argv = [
            "find-asteroids",
            "--catalog", "docs/notebooks/catalog.ecsv",
            "--psfs", "docs/notebooks/psfs.ecsv",
            "--velocity", "0.1", "0.5",
            "--angle", "0", "359.99",
            "--dx", "10",
            "--num-results", "5",
            "--run-id", "test-run-tempdir",
            "--results-db-uri", db_uri,
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            main()
        finally:
            sys.argv = old_argv

        engine = create_engine(db_uri)
        with Session(engine) as session:
            results = session.query(Result).all()
            assert len(results) == 5
            assert {r.search.run_id for r in results} == {"test-run-tempdir"}


def test_compile_results_db_params():
    """A single Search row is created for a run_id, populated from `params`
    (keys not matching a Search column are ignored), and every Result row
    from that run links to it."""
    from find_asteroids.search import run_search
    from find_asteroids.results import compile_results_db
    from find_asteroids.models import Result, Search
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    with TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        run_search(
            "docs/notebooks/catalog.ecsv",
            "docs/notebooks/psfs.ecsv",
            [0.1, 0.5],
            [0, 359.99],
            10,
            3,
            results_dir,
        )
        db_uri = f"sqlite:///{tmpdir}/results.db"
        params = {
            "velocity_0": 0.1, "velocity_1": 0.5,
            "angle_0": 0, "angle_1": 359.99,
            "dx": 10,
            "catalog": "docs/notebooks/catalog.ecsv",
            "not_a_search_column": "ignored",
        }
        compile_results_db(db_uri, results_dir, run_id="test-run-params", params=params)

        engine = create_engine(db_uri)
        with Session(engine) as session:
            searches = session.query(Search).all()
            assert len(searches) == 1
            search = searches[0]
            assert search.run_id == "test-run-params"
            assert search.velocity_0 == 0.1 and search.velocity_1 == 0.5
            assert search.angle_0 == 0 and search.angle_1 == 359.99
            assert search.dx == 10
            assert search.catalog == "docs/notebooks/catalog.ecsv"

            results = session.query(Result).all()
            assert len(results) == 3
            assert all(r.search_id == search.id for r in results)


def test_compile_results_db_reuses_search_for_same_run_id():
    """Calling compile_results_db twice for the same run_id must not create
    a second Search row -- the existing one is reused as-is."""
    from find_asteroids.search import run_search
    from find_asteroids.results import compile_results_db
    from find_asteroids.models import Search
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    with TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        run_search("docs/notebooks/catalog.ecsv", "docs/notebooks/psfs.ecsv", [0.1, 0.5], [0, 359.99], 10, 3, results_dir)

        db_uri = f"sqlite:///{tmpdir}/results.db"
        compile_results_db(db_uri, results_dir, run_id="repeat-run", params={"dx": 10})
        compile_results_db(db_uri, results_dir, run_id="repeat-run", params={"dx": 999})  # ignored on the repeat call

        engine = create_engine(db_uri)
        with Session(engine) as session:
            searches = session.query(Search).filter_by(run_id="repeat-run").all()
            assert len(searches) == 1
            assert searches[0].dx == 10  # from the first call, not overwritten by the second


def test_params_for_db():
    """params_for_db()'s three special cases: catalog gets a host-qualified
    resolved path, list-valued args are broken into <name>_<i> fields, and
    results_dir is recorded as None when it's a (soon to be cleaned up)
    temporary directory."""
    from find_asteroids.search import params_for_db
    from pathlib import Path
    import socket

    args = {
        "catalog": Path("docs/notebooks/catalog.ecsv"),
        "psfs": Path("docs/notebooks/psfs.ecsv"),
        "velocity": [0.1, 0.5],
        "angle": [0, 359.99],
        "dx": 10.0,
        "results_dir": Path("/tmp/find-asteroids-abc123/results"),
    }

    params = params_for_db(args, results_dir_is_tempdir=True)
    assert params["catalog"] == f"{socket.getfqdn()}:{Path('docs/notebooks/catalog.ecsv').resolve()}"
    assert params["psfs"] == "docs/notebooks/psfs.ecsv"
    assert "velocity" not in params
    assert params["velocity_0"] == 0.1 and params["velocity_1"] == 0.5
    assert "angle" not in params
    assert params["angle_0"] == 0 and params["angle_1"] == 359.99
    assert params["dx"] == 10.0
    assert params["results_dir"] is None  # tempdir, gets cleaned up -- meaningless to keep

    params = params_for_db(args, results_dir_is_tempdir=False)
    assert params["results_dir"] == "/tmp/find-asteroids-abc123/results"  # user-chosen, kept


def test_main_stores_params(tmp_path):
    """main()'s CLI wiring end to end: params_for_db() is actually applied
    to a real run's Search row."""
    from find_asteroids.models import Result, Search
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from pathlib import Path
    import socket

    db_uri = f"sqlite:///{tmp_path}/results.db"
    _run_main([
        "--catalog", "docs/notebooks/catalog.ecsv",
        "--psfs", "docs/notebooks/psfs.ecsv",
        "--velocity", "0.1", "0.5",
        "--angle", "0", "359.99",
        "--dx", "10",
        "--num-results", "3",
        "--run-id", "test-run-params-cli",
        "--results-db-uri", db_uri,
    ])

    engine = create_engine(db_uri)
    with Session(engine) as session:
        search = session.query(Search).filter_by(run_id="test-run-params-cli").one()
        assert search.catalog == f"{socket.getfqdn()}:{Path('docs/notebooks/catalog.ecsv').resolve()}"
        assert search.psfs == "docs/notebooks/psfs.ecsv"
        assert search.velocity_0 == 0.1 and search.velocity_1 == 0.5
        assert search.angle_0 == 0.0 and search.angle_1 == 359.99
        assert search.dx == 10
        # --results-dir wasn't passed, so this run used (and cleaned up) a tempdir
        assert search.results_dir is None

        results = session.query(Result).all()
        assert all(r.search_id == search.id for r in results)
