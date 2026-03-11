from pathlib import Path
import logging

log = logging.getLogger(__name__)

def read_results(results_dir : Path, name : str, output_format='ecsv'):
    import astropy.table
    if type(results_dir) is str:
        results_dir = Path(results_dir)
    for p in sorted(results_dir.glob("*/"+f"{name}.{output_format}"), key=lambda x: int(x.parent.name)):
        log.info("reading %s", p)
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

def read_results_mlflow(experiment, name, tracking_uri=None, output_format='ecsv', run_id=None):
    import mlflow
    from mlflow.tracking import MlflowClient
    import json
    import tempfile
    from hashlib import md5
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient()
    
    # search experiment by name
    exp = client.get_experiment_by_name(experiment)
    experiment_ids = [exp.experiment_id]

    for exp_id in experiment_ids:
        runs = client.search_runs(experiment_ids=[exp_id], filter_string="", max_results=5000)
        for run in runs:
            run_info = run.info
            if run_id is not None and run_info.run_id != run_id:
                continue
            results = client.list_artifacts(run_info.run_id, path="results")
            # get run tags as json
            run_tags = {key: value for key, value in run.data.tags.items() if not key.startswith('mlflow.')}
            run_params = {key: value for key, value in run.data.params.items()}
            run_hash_data = json.dumps(
                {
                    'tags': json.dumps(run_tags, sort_keys=True), 
                    'params': json.dumps(
                        {
                            k: v 
                            for k, v in run_params.items() 
                            if k not in ['results_dir', 'precompute', 'gpu', 'device', 'output_format', 'gpu_kernels'] # these shouldn't affect results
                        }, 
                        sort_keys=True
                    )
                },
                sort_keys=True
            )
            run_hash = md5(run_hash_data.encode()).hexdigest()
            # make temporary directory and download artifacts; compile results in temporary directory and read the compiled result
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdir = Path(tmpdirname)
                for result in results:
                    # print(f"Downloading artifacts from {result.path} to {tmpdirname}")
                    client.download_artifacts(run_info.run_id, result.path, tmpdirname)
                for r in read_results(tmpdir, name, output_format=output_format):
                    r['run_id'] = run_info.run_id
                    r['run_tags'] = run_tags
                    r['run_params'] = run_params
                    r['run_hash'] = run_hash
                    yield r

def compile_results_astropy(arg, output_format='ecsv', reader='local', run_id=None):
    import astropy.table
    if reader == 'local':
        arg = Path(arg)

    for name in ['gathered', 'result', 'points', 'tracklet']:
        if reader == 'mlflow':
            yield (name, astropy.table.vstack(list(read_results_mlflow(arg, name, output_format=output_format, run_id=run_id))))
        elif reader == 'local':
            yield (name, astropy.table.vstack(list(read_results(arg, name, output_format=output_format))))
        else:
            raise ValueError(f"Unknown reader: {reader}")
        
def compile_results_db(results_db_uri, arg, reader='local', output_format='ecsv', echo=False, run_id=None):
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.orm import Session
    from .models import Base, Run, Experiment, Result
    import json
    import importlib
    engine = create_engine(results_db_uri, echo=echo)
    # create the database from models if it does not exist
    if not inspect(engine).has_table('result'):
        Base.metadata.create_all(engine)

    with Session(engine) as session:
        runs = {}
        results = {} # dictionary of Result objects
        for name in ['result', 'gathered', 'points', 'tracklet']:
            def do_add(row):
                cls = importlib.import_module('find_asteroids.models').__dict__[name.capitalize()]
                # get the column names from the model
                model_columns = {c.name for c in cls.__table__.columns}
                # filter the row to only include columns that are in the model
                filtered_row = {k: v for k, v in row.items() if k in model_columns}
                for k, v in filtered_row.items():
                    if hasattr(v, 'item'): # convert numpy types to python types
                        filtered_row[k] = v.item()
                extra = json.dumps({str(k): str(v) for k, v in row.items() if k not in model_columns})
                filtered_row['extra'] = extra

                obj = cls(**filtered_row)
                session.add(obj)
                return obj

            if reader == 'local':
                arg = Path(arg)
                for row in read_results(arg, name, output_format=output_format):
                    if name == 'result':
                        results[row['result_num']] = do_add(row)
                    else: # link results
                        row['result_id'] = results[row.pop('result_num')].id
                        do_add(row)
            elif reader == 'mlflow':
                experiment = session.query(Experiment).filter_by(name=arg).first()
                if experiment is None:
                    experiment = Experiment(name=arg)
                    session.add(experiment)
                    session.commit()
                
                for row in read_results_mlflow(arg, name, output_format=output_format, run_id=run_id):
                    run = runs.get(row['run_id'], None)
                    if run is None:
                        run = session.query(Run).filter_by(run_id=row['run_id']).first()
                        if run is None:
                            run = Run(
                                run_id=row['run_id'], 
                                tags=row.get('run_tags', {}),
                                params=row.get('run_params'),
                                hash=str(row.get('run_hash')),
                                experiment_id=experiment.id
                            )
                            session.add(run)
                            session.commit()
                    runs[row['run_id']] = run

                    row.pop('run_id')
                    row.pop('run_tags', None)
                    row.pop('run_params')
                    row.pop('run_hash')
                    row['run_id'] = run.id

                    if name == 'result':
                        results[(row['run_id'], row['result_num'])] = do_add(row)
                    else: # link results
                        key = (row.pop('run_id'), row.pop('result_num'))
                        row['result_id'] = results[key].id
                        do_add(row)
            else:
                raise ValueError(f"Unknown reader: {reader}")
            session.commit()
