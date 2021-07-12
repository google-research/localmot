# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for horizon_util."""

from . import horizon_util

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np


class HorizonUtilTest(parameterized.TestCase):

  def test_int_frame_horizons(self):
    radius = np.array([0.1, 0.3, 1, 3])
    actual = horizon_util.int_frame_horizon(radius, num_frames=10, time_scale=5)
    expected = [0, 1, 5, 9]
    np.testing.assert_equal(actual, expected)
    self.assertTrue(np.issubdtype(actual.dtype, np.integer))

  def test_int_frame_horizons_infinite(self):
    num_frames = 10
    radius = np.array([-np.inf, 0, np.inf])
    actual = horizon_util.int_frame_horizon(radius, num_frames, time_scale=1)
    expected = [0, 0, num_frames - 1]
    np.testing.assert_equal(actual, expected)
    self.assertTrue(np.issubdtype(actual.dtype, np.integer))

  def test_cover(self):
    actual = horizon_util.horizon_range(100)
    expected = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])
    np.testing.assert_equal(actual, expected)

  def test_cover_coeffs_13(self):
    actual = horizon_util.horizon_range(100, coeffs=(1, 3))
    expected = np.array([0.3, 1, 3, 10, 30, 100])
    np.testing.assert_allclose(actual, expected)  # 0.3 = 3/10 not in float

  def test_cover_base2(self):
    actual = horizon_util.horizon_range(100, base=2, coeffs=(1,))
    expected = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    np.testing.assert_equal(actual, expected)

  def test_cover_fraction(self):
    n = 100
    time_scale = n  # use fraction
    expected = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    actual = horizon_util.horizon_range(n, time_scale)
    np.testing.assert_allclose(actual, expected)  # 0.2 = 1/5 not in float

  def test_cover_seconds(self):
    n = 100
    time_scale = 30  # fps
    # 1/30 = 0.0333, 100/30 = 3.33
    expected = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    actual = horizon_util.horizon_range(n, time_scale)
    np.testing.assert_allclose(actual, expected)  # 0.2 = 1/5 not in float

  def test_cover_properties(self):
    n = 1729
    time_scale = np.pi
    values = horizon_util.horizon_range(n, time_scale)
    frames = horizon_util.int_frame_horizon(values, n, time_scale)
    self.assertEqual(frames[0], 0)
    self.assertGreater(frames[1], 0)
    self.assertLess(frames[-2], n - 1)
    self.assertEqual(frames[-1], n - 1)

  @parameterized.parameters(
      dict(min_val=1, max_val=10, coeffs=(1,), base=10,
           expected=[1, 10]),
      dict(min_val=1e-6, max_val=1, coeffs=(1,), base=10,
           expected=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]),
      dict(min_val=0.1, max_val=10, coeffs=(1, 2, 5), base=10,
           expected=[0.1, 0.2, 0.5, 1, 2, 5, 10]),
      dict(min_val=0.2, max_val=5, coeffs=(1, 2, 5), base=10,
           expected=[0.2, 0.5, 1, 2, 5]),
      dict(min_val=0.3, max_val=3, coeffs=(1, 3), base=10,
           expected=[0.3, 1, 3]),
      dict(min_val=0.2, max_val=5, coeffs=(1, 3), base=10,
           expected=[0.3, 1, 3]),
      dict(min_val=0.2 + 1e-7, max_val=5, coeffs=(1, 2, 5), base=10,
           expected=[0.2, 0.5, 1, 2, 5]),
      dict(min_val=0.2, max_val=5 - 1e-7, coeffs=(1, 2, 5), base=10,
           expected=[0.2, 0.5, 1, 2, 5]),
  )
  def test_nice_logspace(self, min_val, max_val, coeffs, base, expected):
    actual = horizon_util.nice_logspace(min_val, max_val, coeffs, base)
    np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
  absltest.main()
