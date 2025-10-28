from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiment'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    runs = relationship("Run", back_populates="experiment")

class Run(Base):
    __tablename__ = 'run'
    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True)
    experiment_id = Column(Integer, ForeignKey('experiment.id'))
    tags = Column(JSON)
    params = Column(JSON)
    hash = Column(String)

    result_entries = relationship("Result", back_populates="run")
    experiment = relationship("Experiment", back_populates="runs")

class Result(Base):
    __tablename__ = 'result'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('run.id'))

    x = Column(Integer)
    y = Column(Integer)
    direction = Column(Integer)
    n = Column(Integer)
    n1 = Column(Integer)
    n2 = Column(Integer)
    n5 = Column(Integer)
    n10 = Column(Integer)
    result_num = Column(Integer)

    extra = Column(JSON)
    run = relationship("Run", back_populates="result_entries")
    points_entries = relationship("Points", back_populates="result")
    gathered_entries = relationship("Gathered", back_populates="result")
    tracklet_entries = relationship("Tracklet", back_populates="result")

class Gathered(Base):
    __tablename__ = 'gathered'
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('result.id'))

    ra = Column(Float)
    dec = Column(Float)
    time = Column(Float)
    
    extra = Column(JSON)
    result = relationship("Result", back_populates="gathered_entries")

class Points(Base):
    __tablename__ = 'points'
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('result.id'))

    ra = Column(Float)
    dec = Column(Float)
    time = Column(Float)

    extra = Column(JSON)

    result = relationship("Result", back_populates="points_entries")

class Tracklet(Base):
    __tablename__ = 'tracklet'
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('result.id'))

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
