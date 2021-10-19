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

"""Miscellaneous tools for tracking metrics."""

import numpy as np
import scipy.optimize


def iou_xywh(box_a, box_b):
  """Returns IOU between boxes in the form [x, y, w, h]."""
  assert np.ndim(box_a) == 2
  assert np.ndim(box_b) == 2
  box_a = box_a[:, np.newaxis]
  box_b = box_b[np.newaxis, :]
  min_a, size_a = box_a[..., :2], box_a[..., 2:]
  min_b, size_b = box_b[..., :2], box_b[..., 2:]
  max_a = min_a + size_a
  max_b = min_b + size_b
  # I = intersect(A, B)
  min_i = np.maximum(min_a, min_b)
  max_i = np.minimum(max_a, max_b)
  size_i = np.maximum(0., max_i - min_i)
  vol_a = np.prod(size_a, axis=-1)
  vol_b = np.prod(size_b, axis=-1)
  vol_i = np.prod(size_i, axis=-1)
  vol_u = vol_a + vol_b - vol_i
  return np.true_divide(vol_i, vol_u)


def match_detections(overlap_matrix, similarity_matrix):
  """Computes matching between detections in a single frame.

  Uses similarity to disambiguate between non-unique solutions.

  Args:
    overlap_matrix: Thresholded similarity matrix.
    similarity_matrix: Matrix with values in [0, 1] or nan.

  Returns:
    Integer array of pairs with shape [num_matches, 2].
  """
  assert np.all(~(similarity_matrix < 0))
  assert np.all(~(similarity_matrix > 1))
  m, n = overlap_matrix.shape
  eps = 1 / (1 + m + n)
  # Maximize number of overlaps and then total similarity.
  weight_matrix = (overlap_matrix + eps * similarity_matrix * overlap_matrix)
  argmax = solve_assignment(-weight_matrix, exclude_zero=True)
  # Check that solution only includes matches with sufficient similarity.
  assert np.all(overlap_matrix[argmax[:, 0], argmax[:, 1]] > 0), (
      'zeros not excluded')
  # Check that similarity did not overpower number of matches.
  # This should be guaranteed by the size of `eps`.
  sum_weights = weight_matrix[argmax[:, 0], argmax[:, 1]].sum()
  sum_overlap = overlap_matrix[argmax[:, 0], argmax[:, 1]].sum()
  assert 0 <= (sum_weights - sum_overlap) < 1, (
      'similarity should not overpower count')
  return argmax


def solve_assignment(weights, exclude_zero=False):
  """Finds matching that maximizes sum of edge weights.

  Args:
    weights: 2D array of edge weights.
    exclude_zero: Exclude pairs with zero weight from result.

  Returns:
    Integer array of pairs with shape [num_matches, 2].
  """
  rs, cs = scipy.optimize.linear_sum_assignment(weights)
  # Ensure that shape is correct if empty.
  if not rs.size:
    return np.empty([0, 2], dtype=int)
  pairs = np.stack([rs, cs], axis=-1)
  if exclude_zero:
    is_nonzero = ~(weights[pairs[:, 0], pairs[:, 1]] == 0)
    pairs = pairs[is_nonzero]
  return pairs
