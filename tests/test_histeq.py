# Last names: P

from minecraft_gen import histeq
import numpy as np

def test_histeq():
  img_test = np.ones((3,3))
  alpha = 1
  
  hist_test = histeq(img_test, alpha)
  
  #copied code from histeq to ensure output is consistent with previous version
  img_cdf, bin_centers = exposure.cumulative_distribution(img_test)
  img_eq = np.interp(img_test, bin_centers, img_cdf)
  img_eq = np.interp(img_eq, (0, 1), (-1, 1))
  hist_compare = alpha * img_eq + (1 - alpha) * img
  
  assert hist_test.shape == (3, 3)
  assert not np.isnan(hist_test).any()
  assert not np.isinf(hist_test).any()
  assert hist_test == hist_compare
