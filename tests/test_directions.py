def test_directions():
    from find_asteroids.directions import SearchDirections
    import astropy.units as u
    dx = 10 * u.arcsec
    dt = 4 * u.hour
    directions = SearchDirections([0.1 * u.deg/u.day, 0.2 * u.deg/u.day], [0 * u.deg, 180 * u.deg], dx, dt)