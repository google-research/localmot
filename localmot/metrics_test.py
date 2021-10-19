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

"""Tests for metrics."""

from . import metrics as localmot_metrics

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd


def testcase_no_confusion():
  num_timesteps = 5
  num_gt_ids = 2
  num_tracker_ids = 2

  # No overlap between pairs (0, 0) and (1, 1).
  similarity = np.zeros([num_timesteps, num_gt_ids, num_tracker_ids])
  similarity[:, 0, 1] = [0, 0, 0, 1, 1]
  similarity[:, 1, 0] = [1, 1, 0, 0, 0]
  gt_present = np.zeros([num_timesteps, num_gt_ids])
  gt_present[:, 0] = [1, 1, 1, 1, 1]
  gt_present[:, 1] = [1, 1, 1, 0, 0]
  tracker_present = np.zeros([num_timesteps, num_tracker_ids])
  tracker_present[:, 0] = [1, 1, 1, 1, 0]
  tracker_present[:, 1] = [1, 1, 1, 1, 1]

  expected = {
      0: {
          'num_frames': 1,
          'gt_num_tracks': 8 / 5,
          'pr_num_tracks': 9 / 5,
          'gt_num_is_present': 8 / 5,
          'pr_num_is_present': 9 / 5,
          'track_tp': 4 / 5,
          'ata': 4 / (0.5 * (8 + 9)),
          'idtp': 4 / 5,
          'idf1': 4 / (0.5 * (8 + 9)),
      },
      1: {
          'num_frames': (2 + 3 + 3 + 3 + 2) / 5,
          'gt_num_tracks': (2 + 2 + 2 + 2 + 1) / 5,
          'pr_num_tracks': 2,
          'gt_num_is_present': (4 + 6 + 5 + 4 + 2) / 5,
          'pr_num_is_present': (4 + 6 + 6 + 5 + 3) / 5,
          'track_tp': ((1) + (2 / 3) + (1 / 3 + 1 / 3) + (2 / 3) + (1)) / 5,
          'ata': (((1) + (2 / 3) + (1 / 3 + 1 / 3) + (2 / 3) + (1)) /
                  (0.5 * ((2 + 2 + 2 + 2 + 1) +
                          (2 + 2 + 2 + 2 + 2)))),
          'idtp': (2 + 2 + 2 + 2 + 2) / 5,
          'idf1': ((2 + 2 + 2 + 2 + 2) /
                   (0.5 * ((4 + 6 + 5 + 4 + 2) + (4 + 6 + 6 + 5 + 3)))),
      },
      4: {
          'num_frames': 5,
          'gt_num_tracks': 2,
          'pr_num_tracks': 2,
          'gt_num_is_present': 8,
          'pr_num_is_present': 9,
          'track_tp': 2 / 5 + 2 / 4,
          'ata': (2 / 5 + 2 / 4) / 2,
          'idtp': 4,
          'idf1': 4 / (0.5 * (8 + 9)),
      },
  }

  data = _from_dense(
      num_timesteps=num_timesteps,
      num_gt_ids=num_gt_ids,
      num_tracker_ids=num_tracker_ids,
      gt_present=gt_present,
      tracker_present=tracker_present,
      similarity=similarity,
  )
  return data, expected


def testcase_with_confusion():
  num_timesteps = 5
  num_gt_ids = 2
  num_tracker_ids = 2

  similarity = np.zeros([num_timesteps, num_gt_ids, num_tracker_ids])
  similarity[:, 0, 1] = [0, 0, 0, 1, 1]
  similarity[:, 1, 0] = [1, 1, 0, 0, 0]
  # Add some overlap between (0, 0) and (1, 1).
  similarity[:, 0, 0] = [0, 0, 1, 0, 0]
  similarity[:, 1, 1] = [0, 1, 0, 0, 0]
  gt_present = np.zeros([num_timesteps, num_gt_ids])
  gt_present[:, 0] = [1, 1, 1, 1, 1]
  gt_present[:, 1] = [1, 1, 1, 0, 0]
  tracker_present = np.zeros([num_timesteps, num_tracker_ids])
  tracker_present[:, 0] = [1, 1, 1, 1, 0]
  tracker_present[:, 1] = [1, 1, 1, 1, 1]

  expected = {
      0: {
          'num_frames': 1,
          'gt_num_tracks': 8 / 5,
          'pr_num_tracks': 9 / 5,
          'gt_num_is_present': 8 / 5,
          'pr_num_is_present': 9 / 5,
          'track_tp': 5 / 5,
          'ata': 5 / (0.5 * (8 + 9)),
          'idtp': 5 / 5,
          'idf1': 5 / (0.5 * (8 + 9)),
      },
      4: {
          'num_frames': 5,
          'gt_num_tracks': 2,
          'pr_num_tracks': 2,
          'gt_num_is_present': 8,
          'pr_num_is_present': 9,
          'track_tp': 2 / 5 + 2 / 4,
          'ata': (2 / 5 + 2 / 4) / 2,
          'idtp': 4,
          'idf1': 4 / (0.5 * (8 + 9)),
      },
  }

  data = _from_dense(
      num_timesteps=num_timesteps,
      num_gt_ids=num_gt_ids,
      num_tracker_ids=num_tracker_ids,
      gt_present=gt_present,
      tracker_present=tracker_present,
      similarity=similarity,
  )
  return data, expected


def testcase_split_tracks():
  num_timesteps = 5
  num_gt_ids = 2
  num_tracker_ids = 5

  similarity = np.zeros([num_timesteps, num_gt_ids, num_tracker_ids])
  # Split ground-truth 0 between tracks 0, 3.
  similarity[:, 0, 0] = [1, 1, 0, 0, 0]
  similarity[:, 0, 3] = [0, 0, 0, 1, 1]
  # Split ground-truth 1 between tracks 1, 2, 4.
  similarity[:, 1, 1] = [0, 0, 1, 1, 0]
  similarity[:, 1, 2] = [0, 0, 0, 0, 1]
  similarity[:, 1, 4] = [1, 1, 0, 0, 0]
  gt_present = np.zeros([num_timesteps, num_gt_ids])
  gt_present[:, 0] = [1, 1, 0, 1, 1]
  gt_present[:, 1] = [1, 1, 1, 1, 1]
  tracker_present = np.zeros([num_timesteps, num_tracker_ids])
  tracker_present[:, 0] = [1, 1, 0, 0, 0]
  tracker_present[:, 1] = [0, 0, 1, 1, 1]
  tracker_present[:, 2] = [0, 0, 0, 0, 1]
  tracker_present[:, 3] = [0, 0, 1, 1, 1]
  tracker_present[:, 4] = [1, 1, 0, 0, 0]

  expected = {
      0: {
          'num_frames': 1,
          'gt_num_tracks': 9 / 5,
          'pr_num_tracks': 11 / 5,
          'gt_num_is_present': 9 / 5,
          'pr_num_is_present': 11 / 5,
          'track_tp': 9 / 5,
          'ata': 9 / (0.5 * (9 + 11)),
          'idtp': 9 / 5,
          'idf1': 9 / (0.5 * (9 + 11)),
      },
      4: {
          'num_frames': 5,
          'gt_num_tracks': 2,
          'pr_num_tracks': 5,
          'gt_num_is_present': 9,
          'pr_num_is_present': 11,
          # For gt 0:
          # (0, 0): 2 / 4
          # (0, 3): 2 / 5
          # For gt 1:
          # (1, 1): 2 / 5
          # (1, 2): 1 / 5
          # (1, 4): 2 / 5
          'track_tp': 2 / 4 + 2 / 5,
          'ata': (2 / 4 + 2 / 5) / (0.5 * (2 + 5)),
          # For gt 0:
          # (0, 0): 2
          # (0, 3): 2
          # For gt 1:
          # (1, 1): 2
          # (1, 2): 1
          # (1, 4): 2
          'idtp': 4,
          'idf1': 4 / (0.5 * (9 + 11)),
      },
  }

  data = _from_dense(
      num_timesteps=num_timesteps,
      num_gt_ids=num_gt_ids,
      num_tracker_ids=num_tracker_ids,
      gt_present=gt_present,
      tracker_present=tracker_present,
      similarity=similarity,
  )
  return data, expected


def _from_dense(num_timesteps, num_gt_ids, num_tracker_ids, gt_present,
                tracker_present, similarity):
  gt_subset = [np.flatnonzero(gt_present[t, :]) for t in range(num_timesteps)]
  tracker_subset = [
      np.flatnonzero(tracker_present[t, :]) for t in range(num_timesteps)
  ]
  similarity_subset = [
      similarity[t][gt_subset[t], :][:, tracker_subset[t]]
      for t in range(num_timesteps)
  ]
  data = {
      'num_timesteps': num_timesteps,
      'num_gt_ids': num_gt_ids,
      'num_tracker_ids': num_tracker_ids,
      'num_gt_dets': np.sum(gt_present),
      'num_tracker_dets': np.sum(tracker_present),
      'gt_ids': gt_subset,
      'tracker_ids': tracker_subset,
      'similarity_scores': similarity_subset,
  }
  return data


TESTCASE_BY_NAME = {
    'no_confusion': testcase_no_confusion(),
    'with_confusion': testcase_with_confusion(),
    'split_tracks': testcase_split_tracks(),
}


class MetricsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('no_confusion',),
      ('with_confusion',),
      ('split_tracks',))
  def test_metrics_integer_horizons(self, sequence_name):
    data, expected = TESTCASE_BY_NAME[sequence_name]
    horizons = list(expected.keys())
    local_stats = localmot_metrics.local_stats(
        num_frames=data['num_timesteps'],
        gt_id_subset=data['gt_ids'],
        pr_id_subset=data['tracker_ids'],
        similarity=data['similarity_scores'],
        horizons=horizons)
    normalized = localmot_metrics.normalize(local_stats)
    result = pd.concat([normalized, local_stats], axis=1)
    for r in horizons:
      for key, value in expected[r].items():
        self.assertAlmostEqual(result.loc[r, key], value,
                               msg=f'different value for {key} at horizon {r}')

  @parameterized.parameters(
      ('no_confusion',),
      ('with_confusion',),
      ('split_tracks',))
  def test_metrics_large_horizon_equals_strict(self, sequence_name):
    data, _ = TESTCASE_BY_NAME[sequence_name]
    evaluator = localmot_metrics.StatsEvaluator(
        num_frames=data['num_timesteps'],
        gt_id_subset=data['gt_ids'],
        pr_id_subset=data['tracker_ids'],
        similarity=data['similarity_scores'])
    local_stats = evaluator.local(data['num_timesteps'] - 1)
    strict_stats = evaluator.strict()
    pd.testing.assert_series_equal(local_stats, strict_stats, check_names=False)

  @parameterized.product(
      sequence_name=['no_confusion', 'with_confusion'],
      with_diagnostics=[True, False])
  def test_fields(self, sequence_name, with_diagnostics):
    data, _ = TESTCASE_BY_NAME[sequence_name]
    stats = localmot_metrics.local_stats(
        num_frames=data['num_timesteps'],
        gt_id_subset=data['gt_ids'],
        pr_id_subset=data['tracker_ids'],
        similarity=data['similarity_scores'],
        horizons=[1, 2, 5],
        with_diagnostics=with_diagnostics)
    self.assertContainsSubset(localmot_metrics.FIELDS_STATS, stats.columns)
    result = localmot_metrics.normalize(stats)
    self.assertContainsSubset(localmot_metrics.FIELDS_METRICS, result.columns)

  @parameterized.parameters(
      ('no_confusion',),
      ('with_confusion',),
      ('split_tracks',))
  def test_metrics_inf_horizon(self, sequence_name):
    data, _ = TESTCASE_BY_NAME[sequence_name]
    max_horizon = data['num_timesteps'] - 1
    local_stats = localmot_metrics.local_stats(
        num_frames=data['num_timesteps'],
        gt_id_subset=data['gt_ids'],
        pr_id_subset=data['tracker_ids'],
        similarity=data['similarity_scores'],
        horizons=[max_horizon, np.inf])
    pd.testing.assert_series_equal(local_stats.loc[np.inf],
                                   local_stats.loc[max_horizon],
                                   check_names=False)

  @parameterized.product(
      sequence_name=['no_confusion', 'with_confusion', 'split_tracks'],
      horizon=[0, 1, 3, np.inf],
  )
  def test_decomposition_stats(self, sequence_name, horizon):
    data, _ = TESTCASE_BY_NAME[sequence_name]
    stats = localmot_metrics.local_stats(
        num_frames=data['num_timesteps'],
        gt_id_subset=data['gt_ids'],
        pr_id_subset=data['tracker_ids'],
        similarity=data['similarity_scores'],
        horizons=[horizon],
        with_diagnostics=True)

    self.assertTrue(np.all(stats >= 0))
    self.assertTrue(np.all(stats['track_tp_approx'] <= stats['track_tp']))

    self.assertTrue(np.all(stats['track_tp_approx'] <= stats['gt_frac_opt']))
    self.assertTrue(np.all(stats['track_tp_approx'] <= stats['pr_frac_opt']))
    self.assertTrue(np.all(stats['gt_frac_opt'] <= stats['gt_frac_max']))
    self.assertTrue(np.all(stats['pr_frac_opt'] <= stats['pr_frac_max']))
    self.assertTrue(np.all(stats['gt_frac_max'] <= stats['gt_frac_det']))
    self.assertTrue(np.all(stats['pr_frac_max'] <= stats['pr_frac_det']))

    np.testing.assert_allclose(
        stats['gt_num_tracks'], (stats['track_tp_approx'] +
                                 stats['track_fn_cover'] +
                                 stats['track_fn_union']))
    np.testing.assert_allclose(
        stats['pr_num_tracks'], (stats['track_tp_approx'] +
                                 stats['track_fp_cover'] +
                                 stats['track_fp_union']))
    np.testing.assert_allclose(
        stats['track_fn_cover'], (stats['track_fn_cover_det'] +
                                  stats['track_fn_cover_ass']))
    np.testing.assert_allclose(
        stats['track_fp_cover'], (stats['track_fp_cover_det'] +
                                  stats['track_fp_cover_ass']))
    np.testing.assert_allclose(
        stats['track_fn_cover_ass'], (stats['track_fn_cover_ass_indep'] +
                                      stats['track_fn_cover_ass_joint']))
    np.testing.assert_allclose(
        stats['track_fp_cover_ass'], (stats['track_fp_cover_ass_indep'] +
                                      stats['track_fp_cover_ass_joint']))
    np.testing.assert_allclose(
        stats['track_fn_union'], (stats['track_fn_union_det'] +
                                  stats['track_fn_union_ass']))
    np.testing.assert_allclose(
        stats['track_fp_union'], (stats['track_fp_union_det'] +
                                  stats['track_fp_union_ass']))

  @parameterized.product(
      sequence_name=['no_confusion', 'with_confusion', 'split_tracks'],
      horizon=[0, 1, 3, np.inf],
  )
  def test_decomposition_ata(self, sequence_name, horizon):
    data, _ = TESTCASE_BY_NAME[sequence_name]
    stats = localmot_metrics.local_stats(
        num_frames=data['num_timesteps'],
        gt_id_subset=data['gt_ids'],
        pr_id_subset=data['tracker_ids'],
        similarity=data['similarity_scores'],
        horizons=[horizon],
        with_diagnostics=True)
    metrics = localmot_metrics.normalize(stats)

    self.assertTrue(np.all(metrics >= 0))
    self.assertTrue(np.all(metrics <= 1))

    # Decomposition of ATR.
    np.testing.assert_allclose(
        1 - metrics['atr_approx'], (metrics['atr_error_cover'] +
                                    metrics['atr_error_union']))
    np.testing.assert_allclose(
        metrics['atr_error_cover'], (metrics['atr_error_cover_det'] +
                                     metrics['atr_error_cover_ass']))
    np.testing.assert_allclose(
        metrics['atr_error_cover_ass'], (metrics['atr_error_cover_ass_indep'] +
                                         metrics['atr_error_cover_ass_joint']))
    np.testing.assert_allclose(
        metrics['atr_error_union'], (metrics['atr_error_union_det'] +
                                     metrics['atr_error_union_ass']))
    np.testing.assert_allclose(
        1 - metrics['atr_approx'], (metrics['atr_error_det_fn'] +
                                    metrics['atr_error_det_fp'] +
                                    metrics['atr_error_ass_split'] +
                                    metrics['atr_error_ass_merge']))

    # Decomposition of ATP.
    np.testing.assert_allclose(
        1 - metrics['atp_approx'], (metrics['atp_error_cover'] +
                                    metrics['atp_error_union']))
    np.testing.assert_allclose(
        metrics['atp_error_cover'], (metrics['atp_error_cover_det'] +
                                     metrics['atp_error_cover_ass']))
    np.testing.assert_allclose(
        metrics['atp_error_cover_ass'], (metrics['atp_error_cover_ass_indep'] +
                                         metrics['atp_error_cover_ass_joint']))
    np.testing.assert_allclose(
        metrics['atp_error_union'], (metrics['atp_error_union_det'] +
                                     metrics['atp_error_union_ass']))
    np.testing.assert_allclose(
        1 - metrics['atp_approx'], (metrics['atp_error_det_fn'] +
                                    metrics['atp_error_det_fp'] +
                                    metrics['atp_error_ass_split'] +
                                    metrics['atp_error_ass_merge']))

    # Decomposition of ATA.
    np.testing.assert_allclose(
        1 - metrics['ata_approx'], (metrics['ata_error_det_fn'] +
                                    metrics['ata_error_det_fp'] +
                                    metrics['ata_error_ass_split'] +
                                    metrics['ata_error_ass_merge']))

  @parameterized.product(
      sequence_name=['no_confusion', 'with_confusion', 'split_tracks'],
      horizon=[0, 1, 3, np.inf],
  )
  def test_decomposition_idf1(self, sequence_name, horizon):
    data, _ = TESTCASE_BY_NAME[sequence_name]
    stats = localmot_metrics.local_stats(
        num_frames=data['num_timesteps'],
        gt_id_subset=data['gt_ids'],
        pr_id_subset=data['tracker_ids'],
        similarity=data['similarity_scores'],
        horizons=[horizon],
        with_diagnostics=True)
    metrics = localmot_metrics.normalize(stats)

    self.assertTrue(np.all(metrics >= 0))
    self.assertTrue(np.all(metrics <= 1))

    # Decomposition of IDR.
    self.assertTrue(np.all(metrics['idr_error_det_fp'] == 0))
    np.testing.assert_allclose(
        1 - metrics['idr_approx'], (metrics['idr_error_det_fn'] +
                                    metrics['idr_error_ass_split'] +
                                    metrics['idr_error_ass_merge']))
    # Decomposition of IDP.
    self.assertTrue(np.all(metrics['idp_error_det_fn'] == 0))
    np.testing.assert_allclose(
        1 - metrics['idp_approx'], (metrics['idp_error_det_fp'] +
                                    metrics['idp_error_ass_split'] +
                                    metrics['idp_error_ass_merge']))
    # Decomposition of IDF1.
    np.testing.assert_allclose(
        1 - metrics['idf1_approx'], (metrics['idf1_error_det_fn'] +
                                     metrics['idf1_error_det_fp'] +
                                     metrics['idf1_error_ass_split'] +
                                     metrics['idf1_error_ass_merge']))

  @parameterized.parameters(
      ('no_confusion',),
      ('with_confusion',),
      ('split_tracks',))
  def test_normalize_pd_series(self, sequence_name):
    data, expected = TESTCASE_BY_NAME[sequence_name]
    horizons = list(expected.keys())
    stats = localmot_metrics.local_stats(
        num_frames=data['num_timesteps'],
        gt_id_subset=data['gt_ids'],
        pr_id_subset=data['tracker_ids'],
        similarity=data['similarity_scores'],
        horizons=horizons,
        with_diagnostics=True)
    dataframe = localmot_metrics.normalize(stats)
    for r in horizons:
      series = localmot_metrics.normalize(stats.loc[r])
      self.assertIsInstance(series, pd.Series)
      pd.testing.assert_series_equal(
          series, dataframe.loc[r], check_names=False)


if __name__ == '__main__':
  absltest.main()
