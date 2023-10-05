# Last names: Q, R
import numpy as np
import pytest

from minecraft_gen import get_normal_light


def test_get_normal_light():
    height_map = np.array([[0.0, 0.5], [0.8, 1.0]])
    expected_result = np.array([[0.65027821, 0.65267539], [0.60773718, 0.61224008]])

    result = get_normal_light(height_map)

    assert result.shape == expected_result.shape
    tolerance = 1e-6
    assert result == pytest.approx(expected_result, rel=tolerance, abs=tolerance)
