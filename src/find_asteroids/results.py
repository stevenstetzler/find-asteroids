from pathlib import Path
import logging
from astropy.time import Time
log = logging.getLogger(__name__)


def _json_safe(v):
    """Coerce numpy scalars to native Python types (.item()) and Time
    values to MJD/TAI, so they serialize correctly instead of stringifying."""
    if isinstance(v, Time):
        v = v.tai.mjd
    if hasattr(v, 'item'):
        return v.item()
    return v


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
    """Insert a run's results_dir into `results_db_uri`, linking every
    Result row to a Search row for `run_id`.

    A Search row for `run_id` is created from `params` (unrecognized keys
    ignored) if one doesn't already exist, otherwise the existing row is
    reused as-is. Columns not in a model are preserved per-row in `extra`.
    """
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.orm import Session
    from .models import Base, Search
    import importlib

    results_dir = Path(results_dir)
    engine = create_engine(results_db_uri, echo=echo)
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
            # exclude 'id': it's DB-assigned, but input catalogs commonly
            # have their own unrelated 'id' column that would otherwise
            # collide with it instead of falling through to `extra`.
            model_columns = {c.name for c in cls.__table__.columns} - {'id'}

            def do_add(row):
                filtered_row = {k: _json_safe(v) for k, v in row.items() if k in model_columns}
                # assign a dict, not json.dumps(dict): the column type
                # serializes it; dumping first would double-encode it.
                filtered_row['extra'] = {str(k): _json_safe(v) for k, v in row.items() if k not in model_columns}
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
                session.flush()  # assign ids before 'gathered'/etc. reference them via .id
        session.commit()
