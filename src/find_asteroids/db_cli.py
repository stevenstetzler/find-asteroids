from .models import *

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
    
def db_cli():
    import argparse
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("--log-level", default="INFO", help="The logging level.")
    global_parser.add_argument("--db", type=str, help="The database to interact with.")
    global_parser.add_argument("--echo", action="store_true", help="Echo SQL commands.")

    parser = argparse.ArgumentParser(
        prog="find-asteroids-db",
        parents=[global_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )    

    subparsers = parser.add_subparsers(dest="command", required=True)
    
    insert_detections_parser = subparsers.add_parser("insert-detections")
    insert_detections_parser.add_argument("catalog", type=Path, help="The catalog to ingest, readable with astropy tables.")
    insert_detections_parser.add_argument("--name", type=str, default='catalog', help="The name of the catalog.")
    insert_detections_parser.set_defaults(func=insert_detections)

    insert_psfs_parser = subparsers.add_parser("insert-psfs")
    insert_psfs_parser.add_argument("psfs", type=Path, help="An astropy-readable table containing a 'psf' column (with units) that specifies the PSF-widths of the images from which the detection catalog is derived.")
    insert_psfs_parser.add_argument("--name", required=False, default='psfs', type=str, help="The name of the PSF collection. If not provided, uses the default collection name.")
    insert_psfs_parser.set_defaults(func=insert_psfs)

    create_collection_parser = subparsers.add_parser("create-collection")
    create_collection_parser.add_argument("--name", type=str, default="collection", help="The name of the collection.")
    create_collection_parser.add_argument("--catalogs", type=str, nargs="+", help="The catalogs to insert into the collection.")
    create_collection_parser.add_argument("--psfs", type=str, nargs="+", default=[], help="The psfs catalogs to insert into the collection.")
    create_collection_parser.set_defaults(func=create_collection)

    insert_search_parameters_parser = subparsers.add_parser("insert-search-parameters")
    insert_search_parameters_parser.add_argument("--velocity", required=True, nargs=2, type=float, help="The velocity range over which to search, in units of deg/day.")
    insert_search_parameters_parser.add_argument("--angle", required=True, nargs=2, type=float, help="The on-sky angles over which to search, in units of deg.")
    insert_search_parameters_parser.add_argument("--dx", required=True, type=float, help="Search bin-width, in units of the PSF-width.")
    insert_search_parameters_parser.add_argument("--refine-iterations", type=int, default=1, help="The number of times to refine a candidate result.")
    insert_search_parameters_parser.add_argument("--name", required=True, type=str, help="The name of the search parameter set")
    insert_search_parameters_parser.set_defaults(func=insert_search_parameters)

    create_db_parser = subparsers.add_parser("create-db")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--name", required=True, type=str, help="A label for the search.")
    search_parser.add_argument("--collection", required=True, type=str, help="The collection to search.")
    search_parser.add_argument("--search-parameters", required=True, type=str, help="The search parameter set to use.")
    search_parser.add_argument("--num-results", required=True, type=int, help="Number of results to produce.")
    search_parser.add_argument("--precompute", action='store_true', help="Precompute projected positions of detections for all trial velocities (uses more memory, but may be faster).")
    search_parser.add_argument("--gpu", action='store_true', help="Run the core-search components of the algorithm on GPU.")
    search_parser.add_argument("--gpu-kernels", action='store_true', help="Run the entirety of the search algorithm on the GPU.")
    search_parser.add_argument("--device", type=int, required=False, default=-1, help="The GPU device number to use.")
    search_parser.set_defaults(func=search)

    global_args, _ = global_parser.parse_known_args()
    log.setLevel(global_args.log_level.upper())
    echo = global_args.echo
    db = global_args.db
    engine = create_engine(db, echo=echo)

    args, _ = parser.parse_known_args()
    vargs = vars(args)
    for c in ['log_level', 'db', 'echo']:
        if c in vargs:
            vargs.pop(c)
    cmd = vargs.pop("command")
    
    if cmd == "create-db":

        Base.metadata.create_all(engine)
    else:
        func = vargs.pop("func")
        with Session(engine) as session:
            func(session, **vargs)    

def insert_detections(session, catalog, name='catalog', echo=False):
    import astropy.table
    t = astropy.table.Table.read(catalog)
    detections = []
    catalog = Catalog(name=name)
    for r in t:
        detections.append(Detection(ra=r['ra'], dec=r['dec'], time=r['time'], catalog=catalog))
            
    session.add(catalog)
    session.commit()

def insert_psfs(session, psfs, name='psfs'):
    import astropy.table
    import astropy.units as u
    widths = astropy.table.Table.read(psfs)['psf'].to(u.arcsec).value
    psfs = PSFs(name=name)
    session.add_all(list(map(lambda x : PSFWidth(width=x, psfs=psfs), widths)))
    session.commit()

def create_collection(session, catalogs, psfs, name='collection'):
    collection = Collection(name=name)
    for catalog in catalogs:
        c = session.query(Catalog).filter(Catalog.name == catalog).first()
        collection.catalogs.append(c)
    for psf in psfs:
        p = session.query(PSFs).filter(PSFs.name == psf).first()
        collection.psfs.append(p)
    session.add(collection)
    session.commit()
        
def insert_search_parameters(session, velocity, angle, dx, name, refine_iterations=1):
    search_parameters = SearchParameters(
        name=name,
        velocity_1=velocity[0],
        velocity_2=velocity[1],
        angle_1=angle[0],
        angle_2=angle[1],
        dx=dx,
        refine_iterations=refine_iterations,
    )                
    session.add(search_parameters)
    session.commit()

def search(session, name, collection, search_parameters, num_results, precompute=False, gpu=False, gpu_kernels=False, device=-1):
    import numpy as np
    from numba import cuda
    import astropy.units as u
    from .directions import SearchDirections
    from .search import search, search_gpu
    from .postprocess import refine, gather
    
    if gpu and device > -1:
        cuda.select_device(device)

    parameters = session.query(SearchParameters).filter(SearchParameters.name == search_parameters).first()
    if parameters is None:
        raise Exception(f"search parameters '{search_parameters}' not found.")
    collection = session.query(Collection).filter(Collection.name == collection).first()
    if collection is None:
        raise Exception(f"collection '{collection}' not found.")
    
    _search = session.query(Search).filter(Search.name == name).first()
    if not _search:
        _search = Search(
            name=name,
            search_parameters=parameters,
            collection=collection,
        )
        new_search = True
    else:
        log.info(f"continuing search '{name}' which has {len(_search.results)} results generated")
        new_search = False
    
    _catalog = []
    for catalog in collection.catalogs:
        for detection in catalog.detections:
            _catalog.append([detection.id, detection.ra, detection.dec, detection.time])
    _catalog = np.array(_catalog)
    X = _catalog[:, 1:4]

    psfs = []
    for psf in collection.psfs:
        for psf_width in psf.widths:
            psfs.append(psf_width.width)
    if len(psfs) > 0:
        psf_scaling = np.median(psfs) * u.arcsec
    else:
        psf_scaling = 1 * u.arcsec
    dx = parameters.dx * psf_scaling
    log.info(f"using dx = {dx}")

    reference_epoch = X[:, 2].min() * u.day
    dt = (X[:, 2].max() - X[:, 2].min()) * u.day
    directions = SearchDirections(
        [parameters.velocity_1 * u.deg/u.day, parameters.velocity_2 * u.deg/u.day], 
        [parameters.angle_1 * u.deg, parameters.angle_2 * u.deg], 
        dx, 
        dt
    )
    log.info("searching %d directions", len(directions.b))

    if gpu_kernels:
        results, results_points = search_gpu(X, directions, dx, reference_epoch.value, num_results=num_results)
    else:
        results, results_points = search(X, directions, dx, reference_epoch.value, num_results=num_results, precompute=precompute, gpu=gpu)
    
    for i, (result, points) in enumerate(zip(results, results_points)):
        if session.query(Result).filter(Result.id == i).first():
            log.info(f"skipping previously generated result {i}")
            continue
        # refine
        try:
            _points = points
            for j in range(parameters.refine_iterations):
                mcdr = refine(_points)
                gathered = gather(mcdr, X[:, 0], X[:, 1], X[:, 2], 1/3600)
                _points = _catalog[gathered, 1:4]
        except Exception as e:
            log.error(str(e))
            continue
        
        r = Result(id=i, x=result[0], y=result[1], direction=result[2], n=result[3])
        reference_sky_pos = mcdr.predict(np.atleast_2d([reference_epoch.value]))
        t = Tracklet(
            vra=mcdr.beta[0, 0],
            vdec=mcdr.beta[0, 1],
            ra0=mcdr.alpha[0],
            dec0=mcdr.alpha[1],
            raRef=reference_sky_pos[0][0],
            decRef=reference_sky_pos[0][1],
            timeRef=reference_epoch.value,
            timeMin=_points[:, 2].min(),
            timeMax=_points[:, 2].max(),
            sigma_vra=mcdr.sigma_e[0, 0]**0.5,
            sigma_vdec=mcdr.sigma_e[1, 1]**0.5,
            sigma_vravdec=mcdr.sigma_e[0, 1],
            sigma_vdecvra=mcdr.sigma_e[1, 0],
            sigma_t=mcdr.sigma_xx[0, 0],
            result=r
        )
        _search.results.append(r)
        _search.tracklets.append(t)
    if new_search:
        session.add(_search)
    session.commit()

