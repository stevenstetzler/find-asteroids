def test_refine():
    from find_asteroids.postprocess import refine
    import numpy as np
    X = np.array(
        [
            [0, 0, 0],
            [1, -1, 1],
            [2, -2, 2],
            [3, -3, 4],
        ]
    )
    mcdr = refine(X)
    r = np.polyfit(X[:, 2], X[:, :2], deg=1)
    assert(np.allclose(r, np.vstack([mcdr.beta, [mcdr.alpha]])))

def test_gather():
    from find_asteroids.postprocess import refine, gather
    import numpy as np
    X = np.array(
        [
            [0, 0, 0],
            [1, -1, 1],
            [2, -2, 2],
            [3, -3, 4],
        ]
    )
    mask = gather(refine(X), X[:, 0], X[:, 1], X[:, 2], 1/3600)
