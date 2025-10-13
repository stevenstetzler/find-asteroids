def test_catalog():
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from find_asteroids.models import Base, Catalog


    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "temp.db")
        engine = create_engine(db, echo=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            catalog = Catalog(name="catalog")
            session.add(catalog)
            session.commit()
            assert(len(list(session.query(Catalog).filter(Catalog.name == "catalog"))) == 1)

def test_detection():
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from find_asteroids.models import Base, Detection

    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "temp.db")
        engine = create_engine(db, echo=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            detection = Detection(ra=1.0, dec=1.0, time=1.0)
            session.add(detection)
            session.commit()
            assert(len(list(session.query(Detection).filter(Detection.id == detection.id))) == 1)

def test_search_parameters():
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from find_asteroids.models import Base, SearchParameters

    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "temp.db")
        engine = create_engine(db, echo=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            search_parameters = SearchParameters(
                velocity_1=0.1,
                velocity_2=0.5,
                angle_1=0.,
                angle_2=359.99,
                dx=10,
                refine_iterations=1,
                name="asteroids"
            )
            session.add(search_parameters)
            session.commit()
            assert(len(list(session.query(SearchParameters).filter(SearchParameters.id == search_parameters.id))) == 1)

def test_psf_width():
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from find_asteroids.models import Base, PSFWidth

    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "temp.db")
        engine = create_engine(db, echo=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            psf_width = PSFWidth(
                width=1,
            )
            session.add(psf_width)
            session.commit()
            assert(len(list(session.query(PSFWidth).filter(PSFWidth.id == psf_width.id))) == 1)

def test_psfs():
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from find_asteroids.models import Base, PSFWidth, PSFs

    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "temp.db")
        engine = create_engine(db, echo=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            psfs = PSFs(
                name="psfs"
            )
            psf_width = PSFWidth(
                width=1, psfs=psfs
            )
            session.add(psfs)
            session.commit()
            assert(len(list(session.query(PSFs).filter(PSFs.id == psfs.id))) == 1)
            assert(session.query(PSFs).filter(PSFs.id == psfs.id).first().widths[0].id == psf_width.id)


def test_collection():
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from find_asteroids.models import Base, Collection, PSFs, Catalog

    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "temp.db")
        engine = create_engine(db, echo=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            collection = Collection(
                name="collection"
            )
            catalog = Catalog(
                name="catalog"
            )
            psfs = PSFs(
                name="psfs"
            )
            collection.catalogs.append(catalog)
            collection.psfs.append(psfs)
            session.add(collection)
            session.commit()
            assert(len(list(session.query(Collection).filter(Collection.id == collection.id))) == 1)
            assert(session.query(Collection).filter(Collection.id == collection.id).first().psfs[0].id == psfs.id)
            assert(session.query(Collection).filter(Collection.id == collection.id).first().catalogs[0].id == catalog.id)

def test_search():
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from find_asteroids.models import Base, Search, SearchParameters, Collection
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "temp.db")
        engine = create_engine(db, echo=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            search_parameters = SearchParameters(
                velocity_1=0.1,
                velocity_2=0.5,
                angle_1=0.,
                angle_2=359.99,
                dx=10,
                refine_iterations=1,
                name="asteroids"
            )
            collection = Collection(
                name="collection"
            )
            search = Search(
                collection=collection,
                search_parameters=search_parameters,
                name="search",
            )
            session.add(search)
            session.commit()
            assert(len(list(session.query(Search).filter(Search.id == search.id))) == 1)
            assert(session.query(Search).filter(Search.id == search.id).first().collection.id == collection.id)
            assert(session.query(Search).filter(Search.id == search.id).first().search_parameters.id == search_parameters.id)


def test_result():
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from find_asteroids.models import Base, Result

    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "temp.db")
        engine = create_engine(db, echo=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            result = Result(
                x=1,
                y=1,
                direction=1,
                n=1
            )
            session.add(result)
            session.commit()
            assert(len(list(session.query(Result).filter(Result.id == result.id))) == 1)
            
def test_tracklet():
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from find_asteroids.models import Base, Tracklet

    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "temp.db")
        engine = create_engine(db, echo=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            tracklet = Tracklet(
                vra=1,
                vdec=1,
                ra0=1,
                dec0=1,
                raRef=1,
                decRef=1,
                timeRef=1,
                sigma_vra=1,
                sigma_vdec=1,
                sigma_vravdec=1,
                sigma_t=1,
            )
            session.add(tracklet)
            session.commit()
            assert(len(list(session.query(Tracklet).filter(Tracklet.id == tracklet.id))) == 1)
