def test_cli():
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        p = subprocess.Popen(
            [
                "find-asteroids",
                "--catalog", "docs/notebooks/catalog.ecsv",
                "--psfs", "docs/notebooks/psfs.ecsv",
                "--dx", "10",
                "--velocity", "0.1", "0.5",
                "--angle", "0", "359.99",
                "--results-dir", str(Path(tmpdir) / "results"),
                "--num-results", "1"
            ]
        )
        stdout, stderr = p.communicate()
        assert(p.returncode == 0)

def test_db_cli():
    import subprocess
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        db = "sqlite:///" + str(Path(tmpdir) / "results.db")
        p = subprocess.Popen(
            [
                "find-asteroids-db",
                "--db", db,
                "create-db",
            ]
        )
        stdout, stderr = p.communicate()
        assert(p.returncode == 0)
        
        p = subprocess.Popen(
            [
                "find-asteroids-db",
                "--db", db,
                "insert-detections",
                "docs/notebooks/catalog.ecsv",
                "--name", "catalog"
            ]
        )
        stdout, stderr = p.communicate()
        assert(p.returncode == 0)

        p = subprocess.Popen(
            [
                "find-asteroids-db",
                "--db", db,
                "insert-psfs",
                "docs/notebooks/psfs.ecsv",
                "--name", "psfs",
            ]
        )
        stdout, stderr = p.communicate()
        assert(p.returncode == 0)

        p = subprocess.Popen(
            [
                "find-asteroids-db",
                "--db", db,
                "insert-search-parameters",
                "--name", "asteroids",
                "--velocity", "0.1", "0.5",
                "--angle", "0", "359.99",
                "--dx", "10",
                "--refine-iterations", "3",
            ]
        )
        stdout, stderr = p.communicate()
        assert(p.returncode == 0)

        p = subprocess.Popen(
            [
                "find-asteroids-db",
                "--db", db,
                "create-collection",
                "--name", "collection",
                "--catalogs", "catalog",
                "--psfs", "psfs",
            ]
        )
        stdout, stderr = p.communicate()
        assert(p.returncode == 0)

        p = subprocess.Popen(
            [
                "find-asteroids-db",
                "--db", db,
                "search",
                "--name", "search",
                "--collection", "collection",
                "--search-parameters", "asteroids",
                "--num-results", "1",
            ]
        )
        stdout, stderr = p.communicate()
        assert(p.returncode == 0)


        
        
        
