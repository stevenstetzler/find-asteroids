"""
MLFlow Search Documentation
============================

This script demonstrates how to use MLFlow to track asteroid search
experiments with find-asteroids, and how to list experiments and
print compiled results.

Prerequisites
-------------
Install the pipeline extras to get MLFlow support:

    pip install "find-asteroids[pipeline]"

Usage
-----
Run this script from the docs/ directory:

    cd docs/
    python mlflow_search.py

The script will:
  1. Run a find-asteroids search under an MLFlow experiment.
  2. List all experiments in the local MLFlow tracking store.
  3. List the runs recorded under the experiment, printing their
     parameters and tags.
  4. Compile the results across all runs and print a summary of
     the result table.
"""

import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from find_asteroids.search import run_search_mlflow
from find_asteroids.results import compile_results_astropy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Data files bundled with this repository (relative to docs/).
CATALOG = Path(__file__).parent / "notebooks" / "catalog.ecsv"
PSFS    = Path(__file__).parent / "notebooks" / "psfs.ecsv"

# MLFlow experiment name.  Change this to group related runs together.
EXPERIMENT_NAME = "asteroid-search-example"

# Search parameters
VELOCITY  = [0.1, 0.5]   # deg / day  [min, max]
ANGLE     = [0, 359.99]  # deg        [min, max]
DX        = 10            # bin-width in PSF units (i.e. 10 × median PSF width)
NUM_RESULTS = 10


# ---------------------------------------------------------------------------
# 1. Run a search and record it with MLFlow
# ---------------------------------------------------------------------------

def run_experiment(results_dir: Path) -> str:
    """
    Run a find-asteroids search inside an MLFlow run and return the run ID.

    Parameters
    ----------
    results_dir : Path
        Directory where per-result files will be written.  The directory
        must not already exist; it is created by the search.

    Returns
    -------
    str
        The MLFlow run ID of the newly-created run.
    """
    print(f"\n=== Running search under experiment '{EXPERIMENT_NAME}' ===")
    run_id = run_search_mlflow(
        EXPERIMENT_NAME,
        str(CATALOG),
        str(PSFS),
        VELOCITY,
        ANGLE,
        DX,
        NUM_RESULTS,
        results_dir,
        tags=[("dataset", "example-catalog")],
    )
    print(f"Finished.  MLFlow run ID: {run_id}")
    return run_id


# ---------------------------------------------------------------------------
# 2. List all experiments
# ---------------------------------------------------------------------------

def print_experiments() -> None:
    """Print every experiment registered in the current MLFlow tracking store."""
    client = MlflowClient()
    experiments = client.search_experiments()

    print("\n=== Experiments ===")
    if not experiments:
        print("  (no experiments found)")
        return

    for exp in experiments:
        print(
            f"  id={exp.experiment_id!r:>4}  "
            f"name={exp.name!r}  "
            f"artifact_location={exp.artifact_location!r}"
        )


# ---------------------------------------------------------------------------
# 3. List runs inside a specific experiment
# ---------------------------------------------------------------------------

def print_runs(experiment_name: str) -> None:
    """
    Print all runs recorded under *experiment_name*, including their
    parameters and custom tags.

    Parameters
    ----------
    experiment_name : str
        The name of the MLFlow experiment to inspect.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"\nExperiment '{experiment_name}' not found.")
        return

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="",
        max_results=5000,
    )

    print(f"\n=== Runs in experiment '{experiment_name}' ({len(runs)} total) ===")
    for run in runs:
        info = run.info
        # Custom tags (omit internal mlflow.* keys)
        user_tags = {
            k: v
            for k, v in run.data.tags.items()
            if not k.startswith("mlflow.")
        }
        print(
            f"\n  run_id   : {info.run_id}\n"
            f"  status   : {info.status}\n"
            f"  params   : {run.data.params}\n"
            f"  tags     : {user_tags}"
        )


# ---------------------------------------------------------------------------
# 4. Compile results across all runs and print a summary
# ---------------------------------------------------------------------------

def print_compiled_results(experiment_name: str) -> None:
    """
    Compile result tables from all runs in *experiment_name* and print
    a summary of each table.

    The four tables compiled are:

    * **result**   – Hough-space peak (x, y, direction, vote counts *n*,
                     and gathered counts at 1/2/5/10 PSF widths).
    * **tracklet** – Refined on-sky trajectory (velocities, positions,
                     uncertainties).
    * **points**   – Catalog detections that voted for each result.
    * **gathered** – Original catalog entries matched to each result.

    Parameters
    ----------
    experiment_name : str
        The name of the MLFlow experiment to compile results from.
    """
    print(f"\n=== Compiled results for experiment '{experiment_name}' ===")

    for name, table in compile_results_astropy(
        experiment_name, reader="mlflow", output_format="ecsv"
    ):
        print(f"\n  -- Table: '{name}' --")
        print(f"     rows    : {len(table)}")
        print(f"     columns : {table.colnames}")
        table.pprint(max_lines=5, max_width=100)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use a temporary directory so re-running the script doesn't fail with
    # "directory already exists".  Replace with a fixed path if you want to
    # keep the results on disk.
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"

        run_experiment(results_dir)

    print_experiments()
    print_runs(EXPERIMENT_NAME)
    print_compiled_results(EXPERIMENT_NAME)
