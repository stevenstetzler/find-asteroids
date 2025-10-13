from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey, Table, Index, MetaData
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

detection_result = Table(
    "detection_result",
    Base.metadata,
    Column("detection_id", ForeignKey("detection.id"), primary_key=True),
    Column("result_id", ForeignKey("result.id"), primary_key=True),
    # optional: if you don't want composite PK above, use a unique constraint instead:
    # UniqueConstraint("detection_id", "result_id", name="uq_detection_result"),
    Index("ix_detection_result__result_id", "result_id"),
)

catalog_collection = Table(
    "catalog_collection",
    Base.metadata,
    Column("catalog_id", ForeignKey("catalog.id"), primary_key=True),
    Column("collection_id", ForeignKey("collection.id"), primary_key=True),
)

psfs_collection = Table(
    "psfs_collection",
    Base.metadata,
    Column("psfs_id", ForeignKey("psfs.id"), primary_key=True),
    Column("collection_id", ForeignKey("collection.id"), primary_key=True),
)

class Catalog(Base):
    __tablename__ = "catalog"

    id = Column(Integer, primary_key=True, autoincrement=True)    
    name = Column(String, nullable=False, unique=True)

    detections = relationship(
        "Detection", 
        back_populates="catalog", 
        cascade="all, delete-orphan"
    )

class Detection(Base):
    __tablename__ = "detection"

    id = Column(Integer, primary_key=True, autoincrement=True)
    catalog_id = Column(Integer, ForeignKey("catalog.id"), nullable=True)

    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    time = Column(Float, nullable=False)

    catalog = relationship("Catalog", back_populates="detections")

    results = relationship(
        "Result",
        secondary=detection_result,
        back_populates="detections",
    )

class PSFWidth(Base):
    __tablename__ = "psf_width"

    id = Column(Integer, primary_key=True, autoincrement=True)
    psfs_id = Column(Integer, ForeignKey("psfs.id"), nullable=True)

    width = Column(Float, nullable=False)

    psfs = relationship("PSFs", back_populates="widths")

class PSFs(Base):
    __tablename__ = "psfs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)

    widths = relationship("PSFWidth", back_populates="psfs", cascade="all, delete-orphan")

class Collection(Base):
    __tablename__ = "collection"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    
    searches = relationship("Search", back_populates="collection")

    catalogs = relationship(
        "Catalog",
        secondary=catalog_collection,
        lazy="selectin",
    )
    psfs = relationship(
        "PSFs",
        secondary=psfs_collection,
        lazy="selectin",
    )

class SearchParameters(Base):
    __tablename__ = "search_parameters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    velocity_1 = Column(Float)
    velocity_2 = Column(Float)
    angle_1 = Column(Float)
    angle_2 = Column(Float)
    dx = Column(Float)
    refine_iterations = Column(Integer)
    name = Column(String, unique=True)

    searches = relationship("Search", back_populates="search_parameters", cascade="all, delete-orphan")

class Search(Base):
    __tablename__ = "search"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_parameters_id = Column(Integer, ForeignKey("search_parameters.id"), nullable=False)
    collection_id = Column(Integer, ForeignKey("collection.id"), nullable=False)
    
    name = Column(String, nullable=False, unique=True)

    search_parameters = relationship("SearchParameters", back_populates="searches")
    collection = relationship("Collection", back_populates="searches")

    results = relationship("Result", back_populates="search", cascade="all, delete-orphan")
    tracklets = relationship("Tracklet", back_populates="search", cascade="all, delete-orphan")

class Result(Base):
    __tablename__ = "result"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey("search.id"), nullable=True)

    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    direction = Column(Integer, nullable=False)
    n = Column(Integer, nullable=False)

    detections = relationship(
        "Detection",
        secondary=detection_result,
        back_populates="results",
    )

    search = relationship("Search", back_populates="results")
    tracklets = relationship("Tracklet", back_populates="result", cascade="all, delete-orphan")

class Tracklet(Base):
    __tablename__ = "tracklet"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, ForeignKey("result.id"), nullable=True)
    search_id = Column(Integer, ForeignKey("search.id"), nullable=True)

    vra = Column(Float, nullable=False)
    vdec = Column(Float, nullable=False)
    ra0 = Column(Float, nullable=False)
    dec0 = Column(Float, nullable=False)
    timeRef = Column(Float, nullable=False)
    raRef = Column(Float, nullable=True)
    decRef = Column(Float, nullable=True)
    timeMin = Column(Float, nullable=True)
    timeMax = Column(Float, nullable=True)
    sigma_vra = Column(Float, nullable=True)
    sigma_vdec = Column(Float, nullable=True)
    sigma_vravdec = Column(Float, nullable=True)
    sigma_vdecvra = Column(Float, nullable=True)
    sigma_t = Column(Float, nullable=True)

    search = relationship("Search", back_populates="tracklets")
    result = relationship("Result", back_populates="tracklets")

