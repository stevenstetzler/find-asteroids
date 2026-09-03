from sqlalchemy import (
    Column, Integer, Float, String, Boolean, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Search(Base):
    """The parameters of one find-asteroids search run (see run_search()
    and the CLI in search.py). One row per run_id -- a run's parameters
    are stored once here and referenced by every one of its Result rows
    via `search_id`, rather than duplicated per result.

    `run_id` is an opaque identifier supplied by whatever invoked the
    search (an MLflow run id, a Snakemake job id, a plain UUID, ...).
    find-asteroids doesn't know or care what produced it, only that rows
    sharing a run_id came from the same `run_search()` invocation. Whoever
    owns run_id is still responsible for tracking anything about the run
    *besides* its search parameters (tags, code version, ...) -- that's
    not this database's job.

    Columns are a plain, hand-written mirror of run_search()'s parameters
    (see params_for_db() in search.py) -- not derived from the CLI
    automatically, so adding/renaming/removing a CLI argument means
    updating this model (and a migration) too.
    """
    __tablename__ = 'search'
    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, unique=True, index=True)

    catalog = Column(String)
    psfs = Column(String, nullable=True)
    velocity_0 = Column(Float)
    velocity_1 = Column(Float)
    angle_0 = Column(Float)
    angle_1 = Column(Float)
    dx = Column(Float)
    num_results = Column(Integer)
    results_dir = Column(String, nullable=True)  # None if a temporary directory was used, see params_for_db()
    precompute = Column(Boolean)
    gpu = Column(Boolean)
    gpu_kernels = Column(Boolean)
    device = Column(Integer)
    output_format = Column(String)
    refine_iterations = Column(Integer)

    results = relationship("Result", back_populates="search")


class Result(Base):
    """One candidate result from a single find-asteroids search run."""
    __tablename__ = 'result'
    id = Column(Integer, primary_key=True)
    search_id = Column(Integer, ForeignKey('search.id'), nullable=False, index=True)
    result_num = Column(Integer)  # index within the run; matches results_dir/<result_num>/

    x = Column(Integer)
    y = Column(Integer)
    direction = Column(Integer)
    n = Column(Integer)
    n1 = Column(Integer)
    n2 = Column(Integer)
    n5 = Column(Integer)
    n10 = Column(Integer)

    extra = Column(JSON)
    search = relationship("Search", back_populates="results")
    points_entries = relationship("Points", back_populates="result")
    gathered_entries = relationship("Gathered", back_populates="result")
    tracklet_entries = relationship("Tracklet", back_populates="result")


class Gathered(Base):
    __tablename__ = 'gathered'
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('result.id'), index=True)

    ra = Column(Float)
    dec = Column(Float)
    time = Column(Float)

    extra = Column(JSON)
    result = relationship("Result", back_populates="gathered_entries")


class Points(Base):
    __tablename__ = 'points'
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('result.id'), index=True)

    ra = Column(Float)
    dec = Column(Float)
    time = Column(Float)

    extra = Column(JSON)
    result = relationship("Result", back_populates="points_entries")


class Tracklet(Base):
    __tablename__ = 'tracklet'
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('result.id'), index=True)

    vra = Column(Float, nullable=False)
    vdec = Column(Float, nullable=False)
    ra_0 = Column(Float, nullable=False)
    dec_0 = Column(Float, nullable=False)
    tref = Column(Float, nullable=False)
    ra_ref = Column(Float, nullable=True)
    dec_ref = Column(Float, nullable=True)
    tmin = Column(Float, nullable=True)
    tmax = Column(Float, nullable=True)
    sigma_vra = Column(Float, nullable=True)
    sigma_vdec = Column(Float, nullable=True)
    sigma_vravdec = Column(Float, nullable=True)
    sigma_vdecvra = Column(Float, nullable=True)
    sigma_t = Column(Float, nullable=True)

    extra = Column(JSON)
    result = relationship("Result", back_populates="tracklet_entries")
