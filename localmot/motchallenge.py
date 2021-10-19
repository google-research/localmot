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

"""Tools for working with motchallenge data.

Tabular data format is described at: https://motchallenge.net/instructions/
"""

import collections
import configparser

from . import util

from absl import logging
import numpy as np

FRAME_ID_COLUMN = 0
TRACK_ID_COLUMN = 1
BBOX_COLUMNS = slice(2, 6)
CONFIDENCE_COLUMN = 6
CATEGORY_COLUMN = 7
VISIBILITY_COLUMN = 8

POSITIVE_CATEGORY = 1
# https://github.com/dendorferpatrick/MOTChallengeEvalKit/blob/45098695e9c57999793181b658bebce380df6686/matlab_devkit/utils/preprocessResult.m#L67-L73
CHALLENGE_TO_IGNORE_CATEGORIES = {
    'MOT16': [2, 7, 8, 12],
    'MOT17': [2, 7, 8, 12],
    'MOT20': [2, 6, 7, 8, 12],
}


def to_similarity(num_frames, gt_data, pr_data):
  """Converts tabular data from CSV file to sequences of IDs and similarity.

  Args:
    num_frames: Positive integer giving sequence length.
    gt_data: Tabular data in motchallenge format.
    pr_data: Tabular data in motchallenge format.

  Returns:
    gt_ids: List of ID arrays such that gt_ids[t] gives tracks in frame t.
    pr_ids: List of ID arrays such that pr_ids[t] gives tracks in frame t.
    similarity: List of 2D arrays giving similarity between gt_ids and pr_ids.
  """
  gt_data_in_frame = split_into_frames(num_frames, gt_data)
  pr_data_in_frame = split_into_frames(num_frames, pr_data)
  gt_id_subset = [None for _ in range(num_frames)]
  pr_id_subset = [None for _ in range(num_frames)]
  similarity = [None for _ in range(num_frames)]
  for t in range(num_frames):
    gt_id_subset[t] = gt_data_in_frame[t][:, TRACK_ID_COLUMN]
    pr_id_subset[t] = pr_data_in_frame[t][:, TRACK_ID_COLUMN]
    similarity[t] = util.iou_xywh(gt_data_in_frame[t][:, BBOX_COLUMNS],
                                  pr_data_in_frame[t][:, BBOX_COLUMNS])
  return gt_id_subset, pr_id_subset, similarity


def load_gt(fname):
  """Loads ground-truth tracks in tabular format."""
  return np.loadtxt(fname, delimiter=',', dtype=np.float64, ndmin=2)


def load_pr(fname):
  """Loads predicted tracks in tabular format."""
  try:
    data = np.loadtxt(fname, delimiter=',', dtype=np.float64, ndmin=2)
  except (ValueError, IndexError):
    # Try using whitespace delim (default).
    data = np.loadtxt(fname, delimiter=None, dtype=np.float64, ndmin=2)
  # If category is not -1, then filter by pedestrian.
  _, num_cols = data.shape
  if CATEGORY_COLUMN < num_cols and not np.all(data[:, CATEGORY_COLUMN] == -1):
    data = data[data[:, CATEGORY_COLUMN] == POSITIVE_CATEGORY, :]
  return data


def load_seqinfo(fname):
  """Loads seqinfo.ini file."""
  config = configparser.ConfigParser()
  config.read(fname)
  return dict(config['Sequence'])


def preprocess(num_frames, gt_data, pr_data, iou_threshold, ignore_categories,
               vis_threshold):
  """Performs preprocessing for pedestrian tracking in motchallenge."""
  ignore_categories = ignore_categories or []
  # Remove all classes that are neither in keep or ignore sets.
  keep_or_ignore_categories = np.unique(
      [POSITIVE_CATEGORY] + list(ignore_categories))
  gt_mask = np.isin(gt_data[:, CATEGORY_COLUMN], keep_or_ignore_categories)
  logging.info('remove irrelevant categories: annotations %d -> %d',
               len(gt_mask), gt_mask.sum())
  gt_data = gt_data[gt_mask, :]
  # Remove ignore classes and non-visible boxes.
  gt_data, pr_data = remove_ignored(num_frames, gt_data, pr_data,
                                    iou_threshold=iou_threshold,
                                    ignore_categories=ignore_categories,
                                    vis_threshold=vis_threshold)
  assert np.all(gt_data[:, CATEGORY_COLUMN] == POSITIVE_CATEGORY), (
      'expect only categories to keep')
  assert np.all(gt_data[:, CONFIDENCE_COLUMN] == 1), (
      'expect all remaining annotations have confidence one')
  return gt_data, pr_data


def remove_ignored(num_frames, gt_data, pr_data, iou_threshold,
                   ignore_categories, vis_threshold):
  """Eliminates objects that belong to ignore classes or have low visibility."""
  ignore_categories = np.unique(ignore_categories)
  gt_data_in_frame = split_into_frames(num_frames, gt_data)
  pr_data_in_frame = split_into_frames(num_frames, pr_data)
  for t in range(num_frames):
    gt_curr = gt_data_in_frame[t]
    pr_curr = pr_data_in_frame[t]
    # Find matching within frame.
    iou_matrix = util.iou_xywh(gt_curr[:, BBOX_COLUMNS],
                               pr_curr[:, BBOX_COLUMNS])
    overlap_matrix = (iou_matrix >= iou_threshold)
    match_pairs = util.match_detections(overlap_matrix, iou_matrix)
    # Remove annotations in any ignored category.
    gt_exclude = np.zeros(len(gt_curr), dtype=bool)
    if np.size(ignore_categories):
      gt_exclude |= np.isin(gt_curr[:, CATEGORY_COLUMN], ignore_categories)
    if vis_threshold is not None:
      gt_exclude |= ~(gt_curr[:, VISIBILITY_COLUMN] >= vis_threshold)
    # Exclude any predictions that match to excluded annotations.
    pr_exclude = np.zeros(len(pr_curr), dtype=bool)
    pr_exclude[match_pairs[:, 1]] = gt_exclude[match_pairs[:, 0]]
    gt_data_in_frame[t] = gt_curr[~gt_exclude]
    pr_data_in_frame[t] = pr_curr[~pr_exclude]
  gt_num_before = len(gt_data)
  pr_num_before = len(pr_data)
  gt_data = np.concatenate(gt_data_in_frame, axis=0)
  pr_data = np.concatenate(pr_data_in_frame, axis=0)
  gt_num_after = len(gt_data)
  pr_num_after = len(pr_data)
  logging.info(
      'remove ignore instances: predictions %d -> %d, annotations %d -> %d',
      pr_num_before, pr_num_after, gt_num_before, gt_num_after)
  return gt_data, pr_data


def split_into_frames(num_frames, data):
  """Returns list of row-subsets of data array."""
  inds = collections.defaultdict(list)
  for i, frame_id in enumerate(data[:, FRAME_ID_COLUMN]):
    inds[frame_id].append(i)
  return [data[inds[t + 1], :] for t in range(num_frames)]
