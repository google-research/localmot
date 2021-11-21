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

"""Functions to compute local metrics.

To compute the metrics for a set of sequences, the statistics are first computed
for each sequence, these are then summed and normalized.

Sequences are specified by (gt_id_subset, pr_id_subset, similarity):
  gt_id_subset: List of integer arrays of groundtruth tracks in each frame.
  pr_id_subset: List of integer arrays of predicted tracks in each frame.
  similarity: List of 2D float arrays that give similarity between tracks.
The value `similarity[t][i, j]` gives the overlap of tracks `gt_id_subset[t][i]`
and `pr_id_subset[t][j]` in frame `t`.

IDs must be integers but do not need to be consecutive from zero.
"""

__all__ = [
    'StatsEvaluator',
    'local_stats',
    'normalize',
    'normalize_diagnostics',
]

import collections
import itertools

from . import horizon_util
from . import util

import numpy as np
import pandas as pd

NAMES = {
    'ada': 'DetF1',
    'ata': 'ATA',
    'ata_assoc': 'ATA / DetF1',
    'atr': 'Track Recall',
    'atp': 'Track Prec',
    'mota': 'MOTA',
    'idf1': 'IDF1',
    'idf1_assoc': 'IDF1 / DetF1',
    'recall': 'Det Recall',
    'precision': 'Det Prec',
    'det_f1': 'Det F1',
    'num_switches': 'Switches',
}

FIELDS_STATS = [
    'gt_num_tracks',
    'pr_num_tracks',
    'gt_num_is_present',
    'pr_num_is_present',
    'num_frames',
    'track_tp',
    'idtp',
]

FIELDS_METRICS = [
    'ata', 'atr', 'atp',
    'idf1', 'idr', 'idp',
]


def strict_stats(
    num_frames, gt_id_subset, pr_id_subset, similarity,
    similarity_threshold=0.5,
    with_diagnostics=True):
  """Returns pd.Series of stats for strict metrics (infinite horizon)."""
  evaluator = StatsEvaluator(
      num_frames, gt_id_subset, pr_id_subset, similarity,
      similarity_threshold=similarity_threshold,
      with_diagnostics=with_diagnostics)
  return evaluator.strict()


def local_stats(
    num_frames, gt_id_subset, pr_id_subset, similarity, horizons,
    time_scale=1,
    similarity_threshold=0.5,
    with_diagnostics=True):
  """Returns pd.DataFrame indexed by horizons in given time-scale."""
  frame_horizons = horizon_util.int_frame_horizon(
      horizons, num_frames, time_scale)
  stats = local_stats_at_int_horizons(
      num_frames, gt_id_subset, pr_id_subset, similarity, frame_horizons,
      similarity_threshold=similarity_threshold,
      with_diagnostics=with_diagnostics)
  return stats.loc[frame_horizons].set_index(pd.Index(horizons, name='horizon'))


def local_stats_at_int_horizons(
    num_frames, gt_id_subset, pr_id_subset, similarity, horizons,
    similarity_threshold=0.5,
    with_diagnostics=True):
  """Returns pd.DataFrame indexed by unique integer horizons."""
  horizons = np.atleast_1d(np.asarray(horizons))
  assert np.all(horizons >= 0)
  assert np.all(horizons < np.inf)
  assert np.all(horizons == horizons.astype(int))
  horizons = horizons.astype(int)
  horizons = np.unique(horizons)
  evaluator = StatsEvaluator(
      num_frames, gt_id_subset, pr_id_subset, similarity,
      similarity_threshold=similarity_threshold,
      with_diagnostics=with_diagnostics)
  stats = [evaluator.local(h) for h in horizons]
  return pd.DataFrame(stats, index=pd.Index(horizons, name='horizon'))


class StatsEvaluator:
  """Computes stats for local metrics efficiently at arbitrary horizons."""

  def __init__(self, num_frames, gt_id_subset, pr_id_subset, similarity,
               similarity_threshold=0.5, with_diagnostics=True):
    self.num_frames = num_frames
    self.with_diagnostics = with_diagnostics

    # Obtain indicators for presence, overlap, match.
    (gt_is_present, pr_is_present,
     overlap_pairs, overlap_occurs,
     match_pairs, match_occurs) = _indicator_arrays(
         num_frames, gt_id_subset, pr_id_subset, similarity,
         similarity_threshold=similarity_threshold,
         with_diagnostics=self.with_diagnostics)

    # Pre-compute cumulative sums of indicators for use by all horizons.
    cumsum_gt_is_present = _cumsum_with_zero(gt_is_present, axis=-1)
    cumsum_pr_is_present = _cumsum_with_zero(pr_is_present, axis=-1)
    cumsum_overlap_occurs = _cumsum_with_zero(overlap_occurs, axis=-1)
    if self.with_diagnostics:
      cumsum_match_occurs = _cumsum_with_zero(match_occurs, axis=-1)

    self.gt_is_present = gt_is_present
    self.pr_is_present = pr_is_present
    self.cumsum_gt_is_present = cumsum_gt_is_present
    self.cumsum_pr_is_present = cumsum_pr_is_present
    self.overlap_pairs = overlap_pairs
    self.overlap_occurs = overlap_occurs
    self.cumsum_overlap_occurs = cumsum_overlap_occurs
    if self.with_diagnostics:
      self.match_pairs = match_pairs
      self.match_occurs = match_occurs
      self.cumsum_match_occurs = cumsum_match_occurs

  def strict(self):
    """Computes stats for strict."""
    # Compute metrics for full sequence: take sum over time axis.
    gt_num_is_present = self.gt_is_present.sum(axis=-1)
    pr_num_is_present = self.pr_is_present.sum(axis=-1)
    overlap_num_occurs = self.overlap_occurs.sum(axis=-1)
    stats = _stats_from_overlap_counts(
        num_frames=self.num_frames,
        gt_num_is_present=gt_num_is_present,
        pr_num_is_present=pr_num_is_present,
        overlap_pairs=self.overlap_pairs,
        overlap_num_occurs=overlap_num_occurs)

    if self.with_diagnostics:
      # Compute diagnostics for full sequence: take sum over time axis.
      match_num_occurs = self.match_occurs.sum(axis=-1)
      stats.update(_diagnostic_stats_from_match_counts(
          gt_num_is_present=gt_num_is_present,
          pr_num_is_present=pr_num_is_present,
          match_pairs=self.match_pairs,
          match_num_occurs=match_num_occurs))

    stats = pd.Series(stats)
    return stats

  def local(self, horizon):
    """Returns stats for local metrics at finite horizon."""
    interval_stats = [None for _ in range(self.num_frames)]
    for t in range(self.num_frames):
      a = max(t - horizon, 0)
      b = min(t + horizon + 1, self.num_frames)

      # Identify subset of tracks that are present in the interval.
      gt_num_is_present = (self.cumsum_gt_is_present[:, b] -
                           self.cumsum_gt_is_present[:, a])
      pr_num_is_present = (self.cumsum_pr_is_present[:, b] -
                           self.cumsum_pr_is_present[:, a])
      gt_mask = (gt_num_is_present > 0)
      pr_mask = (pr_num_is_present > 0)
      # Indices required for re-indexing.
      gt_subset, = gt_mask.nonzero()
      pr_subset, = pr_mask.nonzero()

      overlap_mask = np.logical_and(gt_mask[self.overlap_pairs[:, 0]],
                                    pr_mask[self.overlap_pairs[:, 1]])
      # Compute metrics for this interval using overlap counts.
      curr_interval_stats = _stats_from_overlap_counts(
          num_frames=(b - a),
          gt_num_is_present=gt_num_is_present[gt_mask],
          pr_num_is_present=pr_num_is_present[pr_mask],
          overlap_pairs=_reindex_pairs(gt_subset, pr_subset,
                                       self.overlap_pairs[overlap_mask]),
          overlap_num_occurs=(self.cumsum_overlap_occurs[overlap_mask, b] -
                              self.cumsum_overlap_occurs[overlap_mask, a]))

      if self.with_diagnostics:
        match_mask = np.logical_and(gt_mask[self.match_pairs[:, 0]],
                                    pr_mask[self.match_pairs[:, 1]])
        # Compute diagnostics for this interval using per-frame matching.
        curr_interval_stats.update(_diagnostic_stats_from_match_counts(
            gt_num_is_present=gt_num_is_present[gt_mask],
            pr_num_is_present=pr_num_is_present[pr_mask],
            match_pairs=_reindex_pairs(gt_subset, pr_subset,
                                       self.match_pairs[match_mask]),
            match_num_occurs=(self.cumsum_match_occurs[match_mask, b] -
                              self.cumsum_match_occurs[match_mask, a])))
      interval_stats[t] = curr_interval_stats

    # Take mean over all windows.
    interval_stats = pd.DataFrame(interval_stats)
    stats = interval_stats.sum(axis=0) / self.num_frames
    return stats


def _indicator_dicts(num_frames, gt_id_subset, pr_id_subset, similarity,
                     similarity_threshold=0.5, with_diagnostics=True):
  """Returns dicts of indicator time-series."""
  gt_ids = sorted(set.union(*map(set, gt_id_subset)))
  pr_ids = sorted(set.union(*map(set, pr_id_subset)))
  # Construct indicator time-series for presence, overlap, per-frame match.
  gt_is_present = {gt_id: np.zeros(num_frames, dtype=bool) for gt_id in gt_ids}
  pr_is_present = {pr_id: np.zeros(num_frames, dtype=bool) for pr_id in pr_ids}
  overlap_occurs = collections.defaultdict(
      lambda: np.zeros(num_frames, dtype=bool))
  match_occurs = collections.defaultdict(
      lambda: np.zeros(num_frames, dtype=bool))
  for t in range(num_frames):
    gt_curr_ids = gt_id_subset[t]
    pr_curr_ids = pr_id_subset[t]
    # Require that IDs do not appear twice in the same frame.
    _assert_all_different(gt_curr_ids)
    _assert_all_different(pr_curr_ids)
    for gt_id in gt_curr_ids:
      gt_is_present[gt_id][t] = 1
    for pr_id in pr_curr_ids:
      pr_is_present[pr_id][t] = 1
    overlap_matrix = (similarity[t] >= similarity_threshold)
    rs, cs = overlap_matrix.nonzero()
    for r, c in zip(rs, cs):
      gt_id, pr_id = gt_curr_ids[r], pr_curr_ids[c]
      overlap_occurs[gt_id, pr_id][t] = 1
    if with_diagnostics:
      # Solve for independent, per-frame correspondence.
      argmax = util.match_detections(overlap_matrix, similarity[t])
      for r, c in argmax:
        gt_id, pr_id = gt_curr_ids[r], pr_curr_ids[c]
        match_occurs[gt_id, pr_id][t] = 1
  # Use dict instead of defaultdict to raise KeyError for empty pairs.
  overlap_occurs = dict(overlap_occurs)
  match_occurs = dict(match_occurs)
  return gt_is_present, pr_is_present, overlap_occurs, match_occurs


def _indicator_arrays(num_frames, gt_id_subset, pr_id_subset, similarity,
                      similarity_threshold=0.5,
                      with_diagnostics=True):
  """Returns arrays containing indicator time-series."""
  gt_is_present, pr_is_present, overlap_occurs, match_occurs = (
      _indicator_dicts(num_frames, gt_id_subset, pr_id_subset, similarity,
                       similarity_threshold=similarity_threshold,
                       with_diagnostics=with_diagnostics))

  gt_ids = sorted(gt_is_present.keys())
  pr_ids = sorted(pr_is_present.keys())
  overlap_pairs = sorted(overlap_occurs.keys())
  match_pairs = sorted(match_occurs.keys())

  gt_is_present = _stack_maybe_empty(
      [gt_is_present[gt_id] for gt_id in gt_ids],
      out=np.empty([len(gt_ids), num_frames], dtype=bool))
  pr_is_present = _stack_maybe_empty(
      [pr_is_present[pr_id] for pr_id in pr_ids],
      out=np.empty([len(pr_ids), num_frames], dtype=bool))
  overlap_occurs = _stack_maybe_empty(
      [overlap_occurs[pair] for pair in overlap_pairs],
      out=np.empty([len(overlap_pairs), num_frames], dtype=bool))
  match_occurs = _stack_maybe_empty(
      [match_occurs[pair] for pair in match_pairs],
      out=np.empty([len(match_pairs), num_frames], dtype=bool))
  # Replace IDs with zero-based integers.
  overlap_pairs = _reindex_pairs(gt_ids, pr_ids, overlap_pairs)
  match_pairs = _reindex_pairs(gt_ids, pr_ids, match_pairs)
  return (gt_is_present, pr_is_present,
          overlap_pairs, overlap_occurs,
          match_pairs, match_occurs)


def _stats_from_overlap_counts(
    num_frames,
    gt_num_is_present,
    pr_num_is_present,
    overlap_pairs,
    overlap_num_occurs):
  """Obtains statistics for IDF1 and ATA given number of frames that overlap.

  Args:
    num_frames: Integer.
    gt_num_is_present: Integer array of shape [num_gt].
    pr_num_is_present: Integer array of shape [num_pr].
    overlap_pairs: Integer array of (gt, pr) pairs with shape [num_pairs, 2].
      Indices should be in [0, num_gt) and [0, num_pr) respectively.
    overlap_num_occurs: Integer array of shape [num_pairs].
      Number of frames where the pair of tracks satisfy overlap criterion.

  Returns:
    Dict that maps field name to value.
  """
  sums = {}
  num_gt, = gt_num_is_present.shape
  num_pr, = pr_num_is_present.shape
  # Ensure counts are floats for division.
  gt_num_is_present = gt_num_is_present.astype(np.float64)
  pr_num_is_present = pr_num_is_present.astype(np.float64)
  overlap_num_occurs = overlap_num_occurs.astype(np.float64)

  # Find correspondence for ATA.
  overlap_gt, overlap_pr = overlap_pairs[:, 0], overlap_pairs[:, 1]
  overlap_union = (gt_num_is_present[overlap_gt] + pr_num_is_present[overlap_pr]
                   - overlap_num_occurs)
  overlap_pair_track_tp = overlap_num_occurs / overlap_union
  track_tp_matrix = _make_dense(
      [num_gt, num_pr], (overlap_gt, overlap_pr), overlap_pair_track_tp)
  argmax = util.solve_assignment(-track_tp_matrix, exclude_zero=True)
  sums['track_tp'] = track_tp_matrix[argmax[:, 0], argmax[:, 1]].sum()

  # Find correspondence for IDF1.
  num_overlap_matrix = _make_dense(
      [num_gt, num_pr], (overlap_gt, overlap_pr), overlap_num_occurs)
  argmax = util.solve_assignment(-num_overlap_matrix, exclude_zero=True)
  sums['idtp'] = num_overlap_matrix[argmax[:, 0], argmax[:, 1]].sum()

  sums['num_frames'] = num_frames
  sums['gt_num_tracks'] = num_gt
  sums['pr_num_tracks'] = num_pr
  sums['gt_num_is_present'] = np.sum(gt_num_is_present)
  sums['pr_num_is_present'] = np.sum(pr_num_is_present)
  return sums


def _diagnostic_stats_from_match_counts(
    gt_num_is_present,
    pr_num_is_present,
    match_pairs,
    match_num_occurs):
  """Obtains stats for diagnostics from independent per-frame correspondence.

  Args:
    gt_num_is_present: Integer array of shape [num_gt].
    pr_num_is_present: Integer array of shape [num_pr].
    match_pairs: Integer array of (gt, pr) pairs with shape [num_pairs, 2].
      Indices should be in [0, num_gt) and [0, num_pr) respectively.
    match_num_occurs: Integer array of shape [num_pairs].
      Number of frames where the pair of tracks are matched.

  Returns:
    Dict that maps field name to value.
  """
  sums = {}
  num_gt = len(gt_num_is_present)
  num_pr = len(pr_num_is_present)

  # Ensure all counts are floats for division.
  gt_num_is_present = gt_num_is_present.astype(np.float64)
  pr_num_is_present = pr_num_is_present.astype(np.float64)
  match_num_occurs = match_num_occurs.astype(np.float64)

  match_gt, match_pr = match_pairs[:, 0], match_pairs[:, 1]
  match_union = (gt_num_is_present[match_gt] + pr_num_is_present[match_pr]
                 - match_num_occurs)

  sums['det_tp'] = np.sum(match_num_occurs)
  # Find optimal track correspondence using match instead of overlap.
  match_pair_approx_track_tp = match_num_occurs / match_union
  approx_track_tp_matrix = _make_dense(
      [num_gt, num_pr], (match_gt, match_pr), match_pair_approx_track_tp)
  opt_pairs = util.solve_assignment(-approx_track_tp_matrix, exclude_zero=True)
  opt_gt, opt_pr = opt_pairs[:, 0], opt_pairs[:, 1]
  sums['track_tp_approx'] = approx_track_tp_matrix[opt_gt, opt_pr].sum()

  # Measure fraction of gt/pr track instead of fraction of union.
  num_match_matrix = _make_dense(
      [num_gt, num_pr], (match_gt, match_pairs[:, 1]), match_num_occurs)
  gt_sum_match = num_match_matrix.sum(axis=1)
  pr_sum_match = num_match_matrix.sum(axis=0)
  gt_max_match = num_match_matrix.max(axis=1, initial=0)
  pr_max_match = num_match_matrix.max(axis=0, initial=0)
  sums['gt_frac_det'] = np.sum(gt_sum_match / gt_num_is_present)
  sums['pr_frac_det'] = np.sum(pr_sum_match / pr_num_is_present)
  sums['gt_frac_max'] = np.sum(gt_max_match / gt_num_is_present)
  sums['pr_frac_max'] = np.sum(pr_max_match / pr_num_is_present)
  opt_num_match = num_match_matrix[opt_gt, opt_pr]
  sums['gt_frac_opt'] = np.sum(opt_num_match / gt_num_is_present[opt_gt])
  sums['pr_frac_opt'] = np.sum(opt_num_match / pr_num_is_present[opt_pr])

  sums['track_fn_cover'] = num_gt - sums['gt_frac_opt']
  sums['track_fp_cover'] = num_pr - sums['pr_frac_opt']
  sums['track_fn_cover_det'] = num_gt - sums['gt_frac_det']
  sums['track_fp_cover_det'] = num_pr - sums['pr_frac_det']
  sums['track_fn_cover_max'] = sums['gt_frac_det'] - sums['gt_frac_max']
  sums['track_fp_cover_max'] = sums['pr_frac_det'] - sums['pr_frac_max']
  sums['track_fn_cover_opt'] = sums['gt_frac_max'] - sums['gt_frac_opt']
  sums['track_fp_cover_opt'] = sums['pr_frac_max'] - sums['pr_frac_opt']

  sums['track_fn_union'] = sums['gt_frac_opt'] - sums['track_tp_approx']
  sums['track_fp_union'] = sums['pr_frac_opt'] - sums['track_tp_approx']
  opt_union = (gt_num_is_present[opt_gt] + pr_num_is_present[opt_pr]
               - opt_num_match)
  sums['track_fn_union_det'] = np.sum(
      (opt_num_match / gt_num_is_present[opt_gt]) *
      ((pr_num_is_present[opt_pr] - pr_sum_match[opt_pr]) / opt_union))
  sums['track_fn_union_ass'] = np.sum(
      (opt_num_match / gt_num_is_present[opt_gt]) *
      ((pr_sum_match[opt_pr] - opt_num_match) / opt_union))
  sums['track_fp_union_det'] = np.sum(
      (opt_num_match / pr_num_is_present[opt_pr]) *
      ((gt_num_is_present[opt_gt] - gt_sum_match[opt_gt]) / opt_union))
  sums['track_fp_union_ass'] = np.sum(
      (opt_num_match / pr_num_is_present[opt_pr]) *
      ((gt_sum_match[opt_gt] - opt_num_match) / opt_union))

  return sums


def normalize(stats):
  """Returns pd.DataFrame or pd.Series of metrics obtained from stats.

  Includes diagnostic metrics if present in stats.

  Args:
    stats: pd.DataFrame or pd.Series.
  """
  squeeze = False
  if isinstance(stats, pd.Series):
    # Create trivial DataFrame with single row.
    stats = pd.DataFrame.from_records([stats])
    squeeze = True
  assert isinstance(stats, pd.DataFrame)

  metrics = pd.DataFrame(index=stats.index)
  metrics['ata'] = stats['track_tp'] / (0.5 * (
      stats['gt_num_tracks'] + stats['pr_num_tracks']))
  metrics['atr'] = stats['track_tp'] / stats['gt_num_tracks']
  metrics['atp'] = stats['track_tp'] / stats['pr_num_tracks']

  metrics['idf1'] = stats['idtp'] / (0.5 * (stats['gt_num_is_present'] +
                                            stats['pr_num_is_present']))
  metrics['idr'] = stats['idtp'] / stats['gt_num_is_present']
  metrics['idp'] = stats['idtp'] / stats['pr_num_is_present']

  # Compute normalized diagnostics if present.
  if 'track_tp_approx' in stats:
    metrics = metrics.join(normalize_diagnostics(stats))

  if squeeze:
    metrics = metrics.squeeze(axis=0)
  return metrics


def normalize_diagnostics(stats):
  """Returns pd.DataFrame or pd.Series of diagnostic metrics from stats."""
  squeeze = False
  if isinstance(stats, pd.Series):
    # Create trivial DataFrame with single row.
    stats = pd.DataFrame.from_records([stats])
    squeeze = True
  assert isinstance(stats, pd.DataFrame)

  metrics = pd.DataFrame(index=stats.index)
  metrics['ata_approx'] = stats['track_tp_approx'] / (0.5 * (
      stats['gt_num_tracks'] + stats['pr_num_tracks']))
  metrics['atr_approx'] = stats['track_tp_approx'] / stats['gt_num_tracks']
  metrics['atp_approx'] = stats['track_tp_approx'] / stats['pr_num_tracks']

  error = pd.DataFrame(index=stats.index)
  error['det_fn'] = stats['track_fn_cover_det'] + stats['track_fp_union_det']
  error['det_fp'] = stats['track_fp_cover_det'] + stats['track_fn_union_det']
  error['ass_split'] = (stats['track_fn_cover_max'] +
                        stats['track_fp_cover_opt'] +
                        stats['track_fp_union_ass'])
  error['ass_merge'] = (stats['track_fp_cover_max'] +
                        stats['track_fn_cover_opt'] +
                        stats['track_fn_union_ass'])
  error = error.div(stats['gt_num_tracks'] + stats['pr_num_tracks'], axis=0)
  metrics = metrics.join(error.add_prefix('ata_error_'))

  error_gt = stats[[
      'track_fn_cover',
      'track_fn_cover_det',
      'track_fn_cover_max',
      'track_fn_cover_opt',
      'track_fn_union',
      'track_fn_union_det',
      'track_fn_union_ass',
  ]].div(stats['gt_num_tracks'], axis=0)
  error_gt.columns = error_gt.columns.str.replace('^track_fn_', '', regex=True)
  # Group by cause of error.
  cause_gt = pd.DataFrame({
      'det_fn': error_gt['cover_det'],
      'det_fp': error_gt['union_det'],
      'ass_split': error_gt['cover_max'],
      'ass_merge': error_gt['cover_opt'] + error_gt['union_ass'],
  })
  metrics = metrics.join(error_gt.add_prefix('atr_error_'))
  metrics = metrics.join(cause_gt.add_prefix('atr_error_'))

  error_pr = stats[[
      'track_fp_cover',
      'track_fp_cover_det',
      'track_fp_cover_max',
      'track_fp_cover_opt',
      'track_fp_union',
      'track_fp_union_det',
      'track_fp_union_ass',
  ]].div(stats['pr_num_tracks'], axis=0)
  error_pr.columns = error_pr.columns.str.replace('^track_fp_', '', regex=True)
  # Group by cause of error. (Swap FN/FP and split/merge.)
  cause_pr = pd.DataFrame({
      'det_fp': error_pr['cover_det'],
      'det_fn': error_pr['union_det'],
      'ass_merge': error_pr['cover_max'],
      'ass_split': error_pr['cover_opt'] + error_pr['union_ass'],
  })
  metrics = metrics.join(error_pr.add_prefix('atp_error_'))
  metrics = metrics.join(cause_pr.add_prefix('atp_error_'))

  if squeeze:
    metrics = metrics.squeeze(axis=0)
  return metrics


def _reindex_pairs(gt_ids, pr_ids, pairs):
  """Re-indexes subsets of integers as consecutive integers from 0."""
  if not len(pairs):  # pylint: disable=g-explicit-length-test
    return np.empty([0, 2], dtype=int)
  gt_map = dict(zip(gt_ids, itertools.count()))
  pr_map = dict(zip(pr_ids, itertools.count()))
  # Will raise KeyError if id was not present.
  return np.array([(gt_map[gt_id], pr_map[pr_id]) for gt_id, pr_id in pairs],
                  dtype=int)


def _cumsum_with_zero(x, axis):
  """Like np.cumsum() but adds a zero at the start."""
  x = np.asarray(x)
  zero_shape = list(x.shape)
  zero_shape[axis] = 1
  s = np.cumsum(x, axis=axis)
  return np.concatenate([np.zeros(zero_shape, s.dtype), s], axis=axis)


def _make_dense(shape, keys, values, default=0, dtype=None):
  """Creates a dense matrix from (i, j) keys and scalar values."""
  if dtype is None:
    values = np.asarray(values)
    dtype = values.dtype
  x = np.full(shape, default, dtype=dtype)
  x[keys] = values
  return x


def _assert_all_different(xs):
  if len(xs) != len(set(xs)):
    raise ValueError('elements are not all different', xs)


def _stack_maybe_empty(elems, axis=0, out=None):
  """Like np.stack() but permits elems to be empty if out is empty."""
  if elems:
    return np.stack(elems, axis=axis, out=out)
  else:
    assert np.size(out) == 0, 'output is not empty'
    return out
