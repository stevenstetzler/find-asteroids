from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Result(Base):
    """One candidate result from a single find-asteroids search run.

    `run_id` is an opaque identifier supplied by whatever invoked the
    search (an MLflow run id, a Snakemake job id, a plain UUID, ...).
    find-asteroids doesn't know or care what produced it, only that rows
    sharing a run_id came from the same `run_search()` invocation. Whoever
    owns run_id is responsible for tracking anything about the run itself
    (tags, code version, ...) -- that's not this database's job. `params`
    is the one exception: a caller-supplied, JSON-serializable dict of the
    search parameters (velocity/angle/dx/catalog/...) that produced this
    run, stored as-is on every Result row from that run. It's a convenience
    for querying without needing an external run-tracker online, not a
    substitute for one -- there's no run-level table here, so it's
    duplicated per result rather than normalized.
    """
    __tablename__ = 'result'
    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    result_num = Column(Integer)  # index within the run; matches results_dir/<result_num>/

    x = Column(Integer)
    y = Column(Integer)
    direction = Column(Integer)
    n = Column(Integer)
    n1 = Column(Integer)
    n2 = Column(Integer)
    n5 = Column(Integer)
    n10 = Column(Integer)

    params = Column(JSON)
    extra = Column(JSON)
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
