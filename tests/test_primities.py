def test_project_xyz():
    from find_asteroids.primitives import project_xyz
    x, y = 1, 1
    vx, vy = 1, 1
    t, tref = 1, 0
    x_prime, y_prime = project_xyz(x, y, t, vx, vy, tref)
    assert(x_prime == 0 and y_prime == 0)

def test_project_digitize_point():
    from find_asteroids.primitives import digitize_point
    x_bin, y_bin = digitize_point(0.5, 0.5, 0, 0, 1, 1)
    assert(x_bin == 0 and y_bin == 0)
