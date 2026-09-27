def test_directions():
    from find_asteroids.directions import SearchDirections
    import astropy.units as u
    dx = 10 * u.arcsec
    dt = 4 * u.hour
    directions = SearchDirections([0.1 * u.deg/u.day, 0.2 * u.deg/u.day], [0 * u.deg, 180 * u.deg], dx, dt)
    assert len(directions.b) > 0
    directions.raise_if_empty()  # no-op: doesn't raise when the lattice is non-empty


def test_directions_b_is_empty_for_a_too_short_time_baseline():
    """A short enough `dt`, relative to `dx` and the requested velocity
    window, makes the discrete velocity lattice (spacing dx/dt) too coarse
    for any point to survive the [v_min, v_max] filter -- a legitimate,
    reproducible outcome (not a bug in the lattice/filter logic itself),
    but one that crashes search()/search_gpu() downstream if not guarded
    against (see raise_if_empty())."""
    from find_asteroids.directions import SearchDirections
    import astropy.units as u
    dx = 10 * u.arcsec
    dt = (1 * u.minute).to(u.day)
    directions = SearchDirections([0.1 * u.deg/u.day, 0.5 * u.deg/u.day], [0 * u.deg, 359.99 * u.deg], dx, dt)
    assert len(directions.b) == 0


def test_raise_if_empty_raises_with_a_clear_message():
    from find_asteroids.directions import SearchDirections
    import astropy.units as u
    import pytest

    dx = 10 * u.arcsec
    dt = (1 * u.minute).to(u.day)
    directions = SearchDirections([0.1 * u.deg/u.day, 0.5 * u.deg/u.day], [0 * u.deg, 359.99 * u.deg], dx, dt)
    with pytest.raises(ValueError, match="no achievable search directions"):
        directions.raise_if_empty()