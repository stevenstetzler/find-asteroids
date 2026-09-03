from .directions import SearchDirections
from .postprocess import refine, gather

import numpy as np
from numba import cuda
import astropy.table

import astropy.units as u
from astropy.time import Time
from pathlib import Path
import logging
import socket

logging.basicConfig()
log = logging.getLogger(__name__)

# CLI args whose value is a fixed-length list -- broken out into individual
# <name>_<i> fields in params_for_db() rather than stored as a list, so
# they're queryable as plain scalar columns/JSON fields instead of an array.
LIST_PARAMS = ("velocity", "angle")


def params_for_db(args: dict, results_dir_is_tempdir: bool) -> dict:
    """Build a JSON-serializable params dict (see compile_results_db) from
    main()'s parsed CLI args (as a plain dict, e.g. vars(parsed_args)).

    - 'catalog' is recorded as "<hostname>:<resolved absolute path>", so
      it's unambiguous about which machine/filesystem it came from.
    - list-valued args (velocity, angle) are broken out into `<name>_0`,
      `<name>_1`, ... fields instead of stored as a list.
    - 'results_dir' is recorded as None when it's a temporary directory
      (results_dir_is_tempdir=True) -- it gets cleaned up right after this
      run, so its path doesn't mean anything to anyone reading the DB later.
    - any other Path value is stored as a plain string (Path isn't
      JSON-serializable); everything else is passed through as-is.
    """
    params = {}
    for k, v in args.items():
        if k == "catalog":
            params[k] = f"{socket.getfqdn()}:{v.resolve()}"
        elif k == "results_dir":
            params[k] = None if results_dir_is_tempdir else str(v)
        elif k in LIST_PARAMS:
            for i, x in enumerate(v):
                params[f"{k}_{i}"] = x
        elif isinstance(v, Path):
            params[k] = str(v)
        else:
            params[k] = v
    return params


def normalize_time(time_col):
    """Normalize a catalog's 'time' column to an astropy Time in the TAI
    scale.

    If `time_col` is already an astropy Time (e.g. the caller wrote
    `catalog['time'] = Time(mjd_values, format='mjd', scale='utc')`), its
    scale is used directly. Otherwise -- a plain Quantity/Column in units
    of time, with no scale attached -- it's assumed to be MJD in the UTC
    scale (astropy.time.Time's own default scale for format='mjd'), and a
    warning is logged. There's no way to recover the correct scale from a
    bare number; pass a Time-typed 'time' column to avoid this assumption.
    """
    if isinstance(time_col, Time):
        t = time_col
    else:
        log.warning(
            "catalog['time'] is not an astropy Time column, so its time scale "
            "is unknown; assuming MJD, UTC scale. Pass a Time-typed 'time' "
            "column (e.g. Time(values, format='mjd', scale='utc')) to avoid "
            "this assumption."
        )
        t = Time(time_col.to(u.day).value, format='mjd', scale='utc')
    return t.tai

def search_gpu(X, directions, dx, reference_time, num_results=10):
    from .gpu_impl import projected_bounds, _vote_points, _vote_points_mask, _find_voters_points, _hough_max
    n = X.shape[0]
    
    x_min, x_max, y_min, y_max = projected_bounds(X, directions.b, reference_time)
    
    num_dir = directions.b.shape[0]
    _dx = dx.to(u.deg).value
    _dy = _dx
    num_x = int((x_max - x_min) / _dx  + 1)
    num_y = int((y_max - y_min) / _dy  + 1)

    log.info("creating hough space with shape (%d, %d, %d)", num_dir, num_x, num_y)
    hough = np.zeros((num_dir, num_x, num_y), dtype=np.int32)

    num_dir, num_x, num_y = hough.shape
    
    mask = np.full((num_dir, n), False)
    
    d_hough = cuda.to_device(hough)
    d_mask = cuda.to_device(mask)
    d_X = cuda.to_device(X)
    d_directions = cuda.to_device(directions.b)
    d_max = cuda.to_device(np.zeros((hough.shape[0], 3), dtype=np.int32))
    d_results = cuda.to_device(np.zeros((num_results, 4), dtype=np.int32))

    def _vote(coef=1):
        # Configure GPU threads and blocks
        threads_per_block = (16, 16)  # Tunable parameters
        blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                           (n + threads_per_block[1] - 1) // threads_per_block[1])

        # Launch the CUDA kernel
        _vote_points[blocks_per_grid, threads_per_block](
            d_hough, d_X, d_directions, x_min, y_min, _dx, _dy, reference_time, coef
        )

    def _vote_mask(coef=1):
        # Configure GPU threads and blocks
        threads_per_block = (16, 16)  # Tunable parameters
        blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                           (n + threads_per_block[1] - 1) // threads_per_block[1])

        # Launch the CUDA kernel
        _vote_points_mask[blocks_per_grid, threads_per_block](
            d_hough, d_mask, d_X, d_directions, x_min, y_min, _dx, _dy, reference_time, coef
        )
    
    def _max():
        _hough_max[256, num_dir // 256 + 1](d_max, d_hough)
        
    def _find(mask_dir, mask_x, mask_y):
        threads_per_block = (16, 16)  # Tunable parameters
        blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                           (n + threads_per_block[1] - 1) // threads_per_block[1])

        _find_voters_points[blocks_per_grid, threads_per_block](
            d_X, d_directions, d_mask, x_min, y_min, _dx, _dy, reference_time, mask_dir, mask_x, mask_y
        )

    _vote(coef=1)
    # results = [] # cuda device array
    for n_i in range(num_results):
        _max()
        i = -1
        v = -np.inf
        for _ in range(len(d_max)):
            if d_max[_, 2] > v:
                v = d_max[_, 2]
                i = _
        j = d_max[i, 0]
        k = d_max[i, 1]
        d_results[n_i, 0] = i
        d_results[n_i, 1] = j
        d_results[n_i, 2] = k
        d_results[n_i, 3] = v
        _find(i, j, k)
        print("cluster has value", v, "at", (i, j, k))
        _vote_mask(coef=-1)
    
    return d_results.copy_to_host()

def search(X, directions, dx, reference_time, num_results=10, precompute=False, gpu=False):
    if gpu:
        from .gpu_impl import projected_bounds, hough_max, make_bins, vote_points, vote_bins, find_voters_points, find_voters_bins
    else:
        from .cpu_impl import projected_bounds, hough_max, make_bins, vote_points, vote_bins, find_voters_points, find_voters_bins

    def find_clusters_points(X, hough, directions, x_min, y_min, dx, dy, reference_time, n=10):
        results = np.full((n, 4), -1)
        results_points = []
        include = np.full(X.shape[0], True)
        for i in range(n):
            idx, val = hough_max(hough)
            print("cluster has value", val, "at", idx)
            voters = find_voters_points(
                hough, X, directions.b, x_min, y_min, dx, dy, reference_time, *idx
            )
            mask = include & voters
            hough = vote_points(
                hough, X[mask], directions.b, x_min, y_min, dx, dy, reference_time, -1
            )
            include &= ~voters # exclude voters
            print(include.sum(), "/", X.shape[0], "points remain")
            results[i, 0] = idx[0]
            results[i, 1] = idx[1]
            results[i, 2] = idx[2]
            results[i, 3] = val
            results_points.append(X[mask])
            
        return results, results_points

    def find_clusters_bins(X, bins, hough, n=10):
        results = np.full((n, 4), -1)
        results_points = []
        include = np.full(bins.shape[0], True)
        for i in range(n):
            idx, val = hough_max(hough)
            print("cluster has value", val, "at", idx)
            voters = find_voters_bins(
                hough, bins, *idx
            )
            mask = include & voters
            hough = vote_bins(
                hough, bins[mask], -1
            )
            include &= ~voters # exclude voters
            print(include.sum(), "/", X.shape[0], "points remain")
            results[i, 0] = idx[0]
            results[i, 1] = idx[1]
            results[i, 2] = idx[2]
            results[i, 3] = val
            results_points.append(X[mask])
            
        return results, results_points

    n = X.shape[0]
    
    x_min, x_max, y_min, y_max = projected_bounds(X, directions.b, reference_time)
    
    num_dir = directions.b.shape[0]
    _dx = dx.to(u.deg).value
    _dy = _dx
    num_x = int((x_max - x_min) / _dx  + 1)
    num_y = int((y_max - y_min) / _dy  + 1)

    log.info("creating hough space with shape (%d, %d, %d)", num_dir, num_x, num_y)
    hough = np.zeros((num_dir, num_x, num_y), dtype=np.uint32)    
    
    if precompute:
        bins = make_bins(X, directions.b, x_min, y_min, _dx, _dy, reference_time)
        hough = vote_bins(hough, bins, 1)
        results, results_points = find_clusters_bins(X, bins, hough, n=num_results)
    else:
        hough = vote_points(hough, X, directions.b, x_min, y_min, _dx, _dy, reference_time, 1)
        results, results_points = find_clusters_points(X, hough, directions, x_min, y_min, _dx, _dy, reference_time, n=num_results)
    
    return results, results_points

def run_search(catalog, psfs, velocity, angle, dx, num_results, results_dir, precompute=False, gpu=False, gpu_kernels=False, device=-1, output_format='ecsv', refine_iterations=1):
    """Search `catalog` for moving objects and write results to `results_dir`.

    This is find-asteroids' whole public contract: read a detection catalog
    (an astropy-readable table with at least 'ra', 'dec', 'time' columns,
    with units), write `results_dir/<i>/{result,tracklet,points,gathered}.<output_format>`
    for each of the `num_results` candidates found. Nothing about *how* this
    function is invoked, tracked, or scheduled is find-asteroids' concern --
    see results.py for compiling the output into a single table or database,
    and bring your own orchestration/experiment-tracking layer if you want
    one; it only needs to call this function (or the CLI below) and read the
    directory it writes.
    """
    if gpu and device > -1:
        cuda.select_device(device)

    if psfs:
        psfs = astropy.table.Table.read(psfs)['psf']
        log.info("seeing [min, median, max]: [%f, %f, %f] %s", np.min(psfs), np.median(psfs), np.max(psfs), psfs.unit)
        psf_scaling = np.median(psfs) * psfs.unit
    else:
        psf_scaling = 1 * u.arcsec

    dx = dx * psf_scaling
    log.info(f"using dx = {dx}")
    catalog = astropy.table.Table.read(catalog)

    # Normalize to find-asteroids' internal working convention: ra/dec in
    # degrees, time as an astropy Time in the TAI scale
    catalog['ra'] = catalog['ra'].to(u.deg)
    catalog['dec'] = catalog['dec'].to(u.deg)
    catalog['time'] = normalize_time(catalog['time'])

    X = np.array([catalog['ra'].value, catalog['dec'].value, catalog['time'].mjd]).T
    reference_epoch = X[:, 2].min() * u.day
    dt = (X[:, 2].max() - X[:, 2].min()) * u.day
    vmin = velocity[0] * u.deg/u.day
    vmax = velocity[1] * u.deg/u.day
    phimin = angle[0] * u.deg
    phimax = angle[1] * u.deg

    directions = SearchDirections([vmin, vmax], [phimin, phimax], dx, dt)
    log.info("searching %d directions", len(directions.b))

    results_dir.mkdir(parents=True, exist_ok=False)

    if gpu_kernels:
        results, results_points = search_gpu(X, directions, dx, reference_epoch.value, num_results=num_results)
    else:
        results, results_points = search(X, directions, dx, reference_epoch.value, num_results=num_results, precompute=precompute, gpu=gpu)

    for i, (result, points) in enumerate(zip(results, results_points)):
        # refine
        try:
            _points = points
            for j in range(refine_iterations):
                mcdr = refine(_points)
                gathered = gather(mcdr, X[:, 0], X[:, 1], X[:, 2], dx.to(u.deg).value)
                _points = catalog[gathered]
                _points = np.array([_points['ra'].value, _points['dec'].value, _points['time'].mjd]).T
        except Exception as e:
            log.error(str(e))
            continue

        try:
            n1 = gather(mcdr, X[:, 0], X[:, 1], X[:, 2], 1 * psf_scaling.to(u.deg).value).sum()
            n2 = gather(mcdr, X[:, 0], X[:, 1], X[:, 2], 2 * psf_scaling.to(u.deg).value).sum()
            n5 = gather(mcdr, X[:, 0], X[:, 1], X[:, 2], 5 * psf_scaling.to(u.deg).value).sum()
            n10 = gather(mcdr, X[:, 0], X[:, 1], X[:, 2], 10 * psf_scaling.to(u.deg).value).sum()
        except Exception as e:
            log.error(str(e))
            n1 = -1
            n2 = -1
            n5 = -1
            n10 = -1

        d = results_dir / str(i)
        d.mkdir(parents=True, exist_ok=True)
        astropy.table.Table(
            [
                {
                    "x": result[0],
                    "y": result[1],
                    "direction": result[2],
                    "n": result[3],
                    "n1": n1,
                    "n2": n2,
                    "n5": n5,
                    "n10": n10,
                }
            ]
        ).write(d / f"result.{output_format}")

        reference_sky_pos = mcdr.predict(np.atleast_2d([reference_epoch.value]))
        astropy.table.Table(
            [
                {
                    "vra": mcdr.beta[0, 0] * u.deg/u.day,
                    "vdec": mcdr.beta[0, 1] * u.deg/u.day,
                    "ra_0": (mcdr.alpha[0] % 360) * u.deg,
                    "dec_0": mcdr.alpha[1] * u.deg,
                    "ra_ref": (reference_sky_pos[0][0] % 360) * u.deg,
                    "dec_ref": reference_sky_pos[0][1] * u.deg,
                    "tref": Time(reference_epoch.value, format='mjd', scale='tai'),
                    "tmin": Time(points[:, 2].min(), format='mjd', scale='tai'),
                    "tmax": Time(points[:, 2].max(), format='mjd', scale='tai'),
                    "sigma_vra": mcdr.sigma_e[0, 0]**0.5 * u.deg/u.day,
                    "sigma_vdec": mcdr.sigma_e[1, 1]**0.5 * u.deg/u.day,
                    "sigma_vravdec": mcdr.sigma_e[0, 1] * (u.deg/u.day)**2,
                    "sigma_vdecvra": mcdr.sigma_e[1, 0] * (u.deg/u.day)**2,
                    "sigma_t": mcdr.sigma_xx[0, 0] * u.day,
                }
            ]
        ).write(d / f"tracklet.{output_format}")

        t = astropy.table.Table(
            [
                {
                    "ra": (p[0] % 360) * u.deg,
                    "dec": p[1] * u.deg,
                    "time": Time(p[2], format='mjd', scale='tai'),
                }
                for p in points
            ]
        )
        t.sort("time")
        t.write(d / f"points.{output_format}")

        t = catalog[gathered]
        t['ra'] = t['ra'] % (360 * u.deg)  # wrap only at write time, see note above
        t.sort("time")
        t.write(d / f"gathered.{output_format}")


def main():
    import argparse
    import uuid
    import tempfile
    parser = argparse.ArgumentParser(prog="find-asteroids", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--catalog", required=True, type=Path, help="The detection catalog to search. An astropy-readable table containing at least 'ra', 'dec', and 'time' columns (with units).")
    parser.add_argument("--psfs", required=False, default=None, type=Path, help="An astropy-readable table containing a 'psf' column (with units) that specifies the PSF-widths of the images from which the detection catalog is derived. If not provided, a value of 1 arcsec is assumed.")
    parser.add_argument("--velocity", required=True, nargs=2, type=float, help="The velocity range over which to search, in units of deg/day.")
    parser.add_argument("--angle", required=True, nargs=2, type=float, help="The on-sky angles over which to search, in units of deg.")
    parser.add_argument("--dx", required=True, type=float, help="Search bin-width, in units of the PSF-width.")
    parser.add_argument("--num-results", required=True, type=int, help="Number of results to produce.")
    parser.add_argument("--results-dir", type=Path, required=False, default=None, help="The directory into which to write results. Required unless --results-db-uri is given, in which case it defaults to a temporary directory that's removed after compiling into the database.")
    parser.add_argument("--precompute", action='store_true', help="Precompute projected positions of detections for all trial velocities (uses more memory, but may be faster).")
    parser.add_argument("--gpu", action='store_true', help="Run the core-search components of the algorithm on GPU.")
    parser.add_argument("--gpu-kernels", action='store_true', help="Run the entirety of the search algorithm on the GPU.")
    parser.add_argument("--device", type=int, required=False, default=-1, help="The GPU device number to use.")
    parser.add_argument("--output-format", type=str, default='ecsv', help="The astropy.table supported format for writing results.")
    parser.add_argument("--refine-iterations", type=int, default=1, help="The number of times to refine a candidate result.")
    parser.add_argument("--compile-results", action='store_true', help="Compile results from individual result directories into single files. Implied if --results-db-uri is given.")
    parser.add_argument("--run-id", type=str, default=None, help="Opaque identifier for this run, recorded on each compiled result row (e.g. an id from whatever orchestration/experiment-tracking system invoked this run). find-asteroids does not interpret it. Defaults to a random UUID if not provided.")
    parser.add_argument("--results-db-uri", type=str, default=None, help="If provided, compile results into this SQLAlchemy database URI instead of writing single compiled table files.")

    args = parser.parse_args()

    do_compile = args.compile_results or bool(args.results_db_uri)
    delattr(args, "compile_results")
    results_db_uri = args.results_db_uri
    delattr(args, "results_db_uri")
    run_id = args.run_id or str(uuid.uuid4())
    delattr(args, "run_id")

    if args.results_dir is None and not results_db_uri:
        parser.error("--results-dir is required unless --results-db-uri is given")

    # With --results-db-uri and no explicit --results-dir, the per-result
    # directory tree is pure intermediate scratch on the way into the
    # database -- write it to a temporary directory and clean it up
    # afterward rather than leaving it on disk for no reason.
    tmpdir = None
    if args.results_dir is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="find-asteroids-")
        args.results_dir = Path(tmpdir.name) / "results"

    try:
        run_search(**vars(args))

        if do_compile:
            if results_db_uri:
                from .results import compile_results_db
                params = params_for_db(vars(args), results_dir_is_tempdir=(tmpdir is not None))
                compile_results_db(results_db_uri, args.results_dir, run_id=run_id, params=params, output_format=args.output_format)
            else:
                from .results import compile_results_astropy
                for name, tbl in compile_results_astropy(args.results_dir, output_format=args.output_format):
                    log.info("writing compiled results to %s", args.results_dir / f"{name}.{args.output_format}")
                    tbl.write(args.results_dir / f"{name}.{args.output_format}")
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


if __name__ == "__main__":
    main()
