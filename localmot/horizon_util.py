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

"""Tools for working with temporal horizons in different units.

This module is designed to express logarithmic temporal scales in various units.
It is most useful when a dataset contains sequences of different lengths or
frame-rates.

It is based around the concept of a scale factor such that
`time_in_frames = time_in_units * scale`.

The units supported out of the box are "frames", "seconds" and "fraction".
Other units can be used by setting the scale manually.

Example usage:
```
num_frames = 1000
fps = 30
# Obtain the scale factor for this sequence.
scale = horizon_util.units_to_time_scale('seconds', num_frames, fps)
# Obtain the horizon range in seconds that covers the sequence.
horizons = horizon_util.horizon_range(num_frames, scale)
# Obtain the equivalent integer horizons in frames.
frame_horizons = horizon_util.int_frame_horizon(horizons, num_frames, scale)
```
"""

__all__ = [
    'units_to_time_scale',
    'int_frame_horizon',
    'horizon_range',
    'nice_logspace',
]

import numpy as np


def units_to_time_scale(units, num_frames, fps=None):
  if units == 'frames':
    return 1
  elif units == 'seconds':
    assert fps is not None
    return fps
  elif units == 'fraction':
    return num_frames
  else:
    raise ValueError('unknown units', units)


def int_frame_horizon(horizons, num_frames, time_scale=1):
  """Returns integer frame horizon in [0, num_frames).

  Args:
    horizons: Horizons in desired time units.
    num_frames: Sequence length in frames.
    time_scale: Scale to convert from units to frames.
  """
  horizons = np.asarray(horizons)
  assert num_frames == int(num_frames)
  frames = np.clip(floor_prec(horizons * time_scale), 0, int(num_frames) - 1)
  return frames.astype(int)


def horizon_range(num_frames, time_scale=1, coeffs=(1, 2, 5), base=10):
  """Returns nice_logspace horizons that cover the sequence using scaled units.

  Args:
    num_frames: Sequence length in frames.
    time_scale: Scale to convert from horizon units to frames.
    coeffs: See nice_logspace.
    base: See nice_logspace.

  Ensures that int_frame_horizon() spans [0, num_frames - 1].
  """
  # Find range of frames that spans all horizons.
  coeffs = np.atleast_1d(np.asarray(coeffs, dtype=np.float64))
  min_horizon = 1 / time_scale
  max_horizon = num_frames / time_scale
  # Generate larger logspace than required.
  # This ensures that frame horizons of 0 and (num_frames - 1) are present.
  horizons = nice_logspace(min_horizon / base, max_horizon * base, coeffs, base)
  # Obtain corresponding integer number of frames.
  frames = int_frame_horizon(horizons, num_frames, time_scale)
  middle, = np.nonzero((0 < frames) & (frames < num_frames - 1))
  # Include one element either side.
  idx_first = middle[0] - 1
  idx_last = middle[-1] + 1
  assert 0 <= idx_first
  assert idx_last < len(horizons)
  return horizons[idx_first:(idx_last + 1)]


def nice_logspace(min_val, max_val, coeffs=(1, 2, 5), base=10, rtol=1e-6):
  """Returns logspace in form (c * base**k) with c in coeffs and k integer.

  Uses np.isclose() with rtol to allow for round-off error in limits.

  Args:
    min_val: Finite positive scalar.
    max_val: Finite positive scalar.
    coeffs: Integer coefficients in [1, base).
    base: Multiplicative step size.
    rtol: Relative tolerance for comparison to min and max val.
  """
  # Find range of frames that spans all horizons.
  assert 0 < min_val
  assert max_val < np.inf
  coeffs = np.atleast_1d(np.asarray(coeffs, dtype=np.float64))
  assert np.all(0 < coeffs)
  assert np.all(coeffs < base)
  assert np.all(np.diff(coeffs) > 0)
  assert np.all(coeffs == np.rint(coeffs))
  min_pow = floor_prec(np.log(min_val) / np.log(base)) - 1
  max_pow = ceil_prec(np.log(max_val) / np.log(base)) + 1
  powers = np.arange(min_pow, max_pow + 1)
  coeffs = coeffs[np.newaxis, :]
  powers = powers[:, np.newaxis]
  values = np.ravel(coeffs * base**powers)
  assert np.all(np.diff(values) > 0), 'values not increasing'
  # mask = (min_val <= values) & (values <= max_val)
  mask = (less_close(min_val, values, rtol=rtol, atol=0) &
          less_close(values, max_val, rtol=rtol, atol=0))
  return values[mask]


def less_close(a, b, rtol=1e-6, atol=0):
  return np.less(a, b) | np.isclose(a, b, rtol=rtol, atol=atol)


def floor_prec(x, decimals=6):
  # Round to fixed number of decimals before floor().
  # If x is close to an integer, round() will return that integer.
  # (Integers up to certain size are exactly represented in floating point.)
  # This prevents floor(0.9999999999) => 0.
  return np.floor(np.around(x, decimals))


def ceil_prec(x, decimals=6):
  # Like floor_prec.
  return np.ceil(np.around(x, decimals))
