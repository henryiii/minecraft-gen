from minecraft_gen import voronoi, voronoi_map
import numpy as np


def test_voronoi_map():
    size = 16
    gen = np.random.default_rng(42)
    points = gen.integers(0, size, (8, 2))
    vor = voronoi(points, size)
    vor_map = voronoi_map(vor, size)

    assert vor_map.shape == (size, size)
    assert np.all(vor_map >= 0)
    assert vor_map[0, 0] == 3
