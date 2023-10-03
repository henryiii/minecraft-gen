# Last names: L, M

from minecraft_gen import gradient
import numpy as np

def test_gradient():
    gen = np.random.default_rng(42)
    dummy_im_smooth = gen.random(100)[np.newaxis]
    grad = gradient(dummy_im_smooth)

    assert not np.isnan(grad).any()
    assert not np.isinf(grad).any()
