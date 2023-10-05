# Last names: S
import numpy as np
from minecraft_gen import quantize


def test_quantize():
    # Simple single element test
    assert quantize(np.array([0.0]), 1) == 0
    assert quantize(np.array([10.0]), 3) == 2
    assert quantize(np.array([-10.0]), 3) == 0

    # Simple two element test
    assert (quantize(np.array([-1, 10.0]), 5) == np.array([0, 4])).all()
    assert (quantize(np.array([-1, 10.0]), 3) == np.array([0, 2])).all()

    # Simple three element test
    assert (quantize(np.array([-1, -1, 10.0]), 2) == np.array([0, 0, 1])).all()
    assert (quantize(np.array([-1, -1, 10.0]), 3) == np.array([0, 0, 2])).all()
    assert (quantize(np.array([-1, 0, 10.0]), 4) == np.array([0, 2, 3])).all()


# Test for output with known data
def test_quantize_random():
    gen = np.random.default_rng(42)
    data = gen.uniform(-1, 2, size=10)
    assert (quantize(data, 2) == np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1])).all()
    assert (quantize(data, 3) == np.array([2, 1, 2, 2, 0, 2, 2, 2, 0, 2])).all()
