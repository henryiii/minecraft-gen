# Last names: P

from minecraft_gen import histeq
import numpy as np

def test_histeq():
  img_test = np.ones((3,3))
  hist_test = histeq(img_test, 1)
  assert not np.isnan(hist_test).any()
  assert not np.isinf(hist_test).any()
