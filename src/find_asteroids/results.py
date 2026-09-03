from pathlib import Path
import logging
from astropy.time import Time
log = logging.getLogger(__name__)


def read_results(results_dir: Path, name: str, output_format='ecsv'):
    """Yield one dict per row of `results_dir/<result_num>/<name>.<output_format>`,
    for every result_num subdirectory (see run_search), tagged with its
    `result_num`."""
    import astropy.table
    if type(results_dir) is str:
        results_dir = Path(results_dir)
    for p in sorted(results_dir.glob("*/" + f"{name}.{output_format}"), key=lambda x: int(x.parent.name)):
        log.info(f"reading {p}")
        result_num = p.parent.name
        if output_format == 'fits':
            t = astropy.table.Table.read(p, memmap=True)
            for row in t:
                d = dict(row)
                d['result_num'] = int(result_num)
                yield d
        elif output_format in ['csv', 'ascii']:
            import astropy.io
            tbls = astropy.io.ascii.read(
                p,
                format=output_format,
                guess=False,
                fast_reader={
                    'chunk_size': 100 * 1000000,
                    'chunk_generator': True
                }
            )
            for i, tbl in enumerate(tbls):
                for row in tbl:
                    d = dict(row)
                    d['result_num'] = int(result_num)
                    yield d
        else:
            tbl = astropy.table.Table.read(p)
            for row in tbl:
                d = dict(row)
                d['result_num'] = int(result_num)
                yield d


def compile_results_astropy(results_dir, output_format='ecsv'):
    """Compile a run's results_dir into one vstacked astropy Table per
    result kind: 'gathered', 'result', 'points', 'tracklet'."""
    import astropy.table
    results_dir = Path(results_dir)
    for name in ['gathered', 'result', 'points', 'tracklet']:
        yield (name, astropy.table.vstack(list(read_results(results_dir, name, output_format=output_format))))


def compile_results_db(results_db_uri, results_dir, run_id, params=None, output_format='ecsv', echo=False):
    """Insert a run's results_dir into the database at `results_db_uri`,
    linking every Result row to a Search row for the given (caller-
    supplied, opaque) `run_id`.

    If a Search row for `run_id` doesn't exist yet, one is created from
    `params` -- a dict of the parameters that produced this run (e.g.
    velocity_0/velocity_1/angle_0/angle_1/dx/catalog/..., see
    params_for_db() in search.py); keys that aren't a column on Search are
    ignored. If a Search row for `run_id` already exists (e.g. this
    function is called more than once for the same run), it's reused as-is
    -- `params` is not applied to it on a repeat call.

    Columns from the compiled tables that aren't part of the Result/
    Gathered/Points/Tracklet models (see models.py) -- e.g. extra metadata
    carried through from the input detection catalog -- are preserved in
    each row's `extra` JSON column rather than dropped.
    """
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.orm import Session
    from .models import Base, Search
    import importlib

    results_dir = Path(results_dir)
    engine = create_engine(results_db_uri, echo=echo)
    # create the database from models if it does not exist
    if not inspect(engine).has_table('result'):
        Base.metadata.create_all(engine)

    with Session(engine) as session:
        search = session.query(Search).filter_by(run_id=run_id).first()
        if search is None:
            search_columns = {c.name for c in Search.__table__.columns} - {'id', 'run_id'}
            search = Search(run_id=run_id, **{k: v for k, v in (params or {}).items() if k in search_columns})
            session.add(search)
            session.flush()  # assign search.id before Result rows reference it

        results = {}  # result_num -> Result object, for linking child rows
        for name in ['result', 'gathered', 'points', 'tracklet']:
            cls = importlib.import_module('find_asteroids.models').__dict__[name.capitalize()]
            # get the column names from the model, excluding the primary key:
            # it's always DB-assigned, never taken from source data -- input
            # catalogs commonly have their own unrelated 'id' column (e.g. a
            # per-image detection id), which would otherwise collide with
            # this table's own row identity instead of falling through to
            # `extra` where it belongs.
            model_columns = {c.name for c in cls.__table__.columns} - {'id'}

            def do_add(row):
                # filter the row to only include columns that are in the model
                filtered_row = {k: v for k, v in row.items() if k in model_columns}
                for k, v in filtered_row.items():
                    if isinstance(v, Time):
                        # store the DB's canonical convention (MJD, TAI)
                        # regardless of what scale/format the value already
                        # carried -- run_search() already normalizes to
                        # this, but enforce it here too rather than trust
                        # every possible caller to have done so.
                        filtered_row[k] = v.tai.mjd
                    elif hasattr(v, 'item'):  # convert numpy types to python types
                        filtered_row[k] = v.item()
                # Assign the plain dict directly -- `extra` is a JSON column,
                # so the column type handles serialization itself. (Do not
                # json.dumps() this first: that would serialize it a second
                # time, storing a JSON-encoded *string* instead of an object,
                # which then needs an extra json.loads() to unpack.)
                filtered_row['extra'] = {str(k): str(v) for k, v in row.items() if k not in model_columns}

                obj = cls(**filtered_row)
                session.add(obj)
                return obj

            for row in read_results(results_dir, name, output_format=output_format):
                if name == 'result':
                    row['search_id'] = search.id
                    results[row['result_num']] = do_add(row)
                else:  # link to the owning result
                    row['result_id'] = results[row.pop('result_num')].id
                    do_add(row)
            if name == 'result':
                # Assign auto-increment ids for the rows just added, since
                # they're read back (via .id, above) to link the child
                # tables -- plain attribute access doesn't trigger an
                # autoflush, so without this every *_id ends up NULL.
                session.flush()
        session.commit()
