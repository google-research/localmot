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

r"""Computes metrics and generates plots for data in motchallenge format.

Example usage:
```
python -m localmot.scripts.eval_motchallenge \
  --gt_dir=$GT_DIR --pr_dir=$PR_DIR --out_dir=$OUT_DIR \
  --challenge=MOT17 \
  --nodiagnostics \
  --plot \
  --skip_on_fail
```

The results for each sequence are cached to disk to avoid re-computing.
Cached results are written to $OUT_DIR/cache/.

Each tracker is assigned a style index to ensure that colors are consistent
across plots. These are cached in $OUT_DIR/style.json.

Different output directories should be used per dataset.
"""

import collections
import itertools
import json
import os

from absl import app
from absl import flags
from absl import logging
import localmot
from localmot import motchallenge
from localmot import plots
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

flags.DEFINE_string('gt_dir', '', 'Dir that contains ground-truth files.')
flags.DEFINE_string('pr_dir', '', 'Dir that contains predictions.')
flags.DEFINE_string('out_dir', '', 'Dir to write results.')
flags.DEFINE_string(
    'units', 'frames',
    'Units for horizon. May be frames, seconds or fraction.')
flags.DEFINE_list(
    'trackers', [],
    'If empty, then trackers are discovered from dir structure.')
flags.DEFINE_string(
    'sequence_file', None,
    'If empty, then sequences are discovered from dir structure.')
flags.DEFINE_float('iou_threshold', 0.5, 'IOU threshold for match.')
flags.DEFINE_integer(
    'vis_threshold', None,
    'Minimum ground-truth visibility. No threshold if not specified.')
flags.DEFINE_string(
    'challenge', None,
    'Which challenge? Used to determine which classes to ignore.')
flags.DEFINE_boolean('diagnostics', True, 'Whether to include diagnostics.')
flags.DEFINE_boolean('plot', False, 'Whether to generate figures.')
flags.DEFINE_boolean(
    'skip_on_fail', False,
    'Fail silently and continue when an error is encountered.')
flags.DEFINE_boolean('ignore_cache', False, 'Do not load results from cache.')

FLAGS = flags.FLAGS

COLORS = list(cm.get_cmap('Dark2').colors) + list(cm.get_cmap('tab10').colors)
# Ensure that this cycle has no common factors with color cycle.
# Using a list of length 7 or 11 should be relatively safe.
MARKERS = ['o', 's', 'd', 'v', '^', 'p', '>', '<', 'h', 'D', '*']

LOGSPACE_KWARGS = {'coeffs': (1, 2, 5), 'base': 10}

Metadata = collections.namedtuple('Metadata', ['num_frames', 'fps'])


def main(_):
  np.seterr(all='raise')

  seqs = get_sequence_list()
  logging.info('sequences (%d): %s', len(seqs), seqs)
  trackers = get_tracker_list()
  logging.info('trackers (%d): %s', len(trackers), trackers)

  # Load metadata for sequences (length and frame-rate).
  metadata = {seq: load_sequence_metadata(seq) for seq in seqs}
  # Get covering horizons for each sequence.
  seq_horizons = {}
  for seq in seqs:
    num_frames, fps = metadata[seq]
    time_scale = localmot.horizon_util.units_to_time_scale(
        FLAGS.units, num_frames, fps)
    seq_horizons[seq] = localmot.horizon_util.horizon_range(
        num_frames, time_scale, **LOGSPACE_KWARGS)
  # Obtain horizons that span all sequences.
  min_horizon = min(min(seq_horizons[seq]) for seq in seqs)
  max_horizon = max(max(seq_horizons[seq]) for seq in seqs)
  horizons = localmot.horizon_util.nice_logspace(
      min_horizon, max_horizon, **LOGSPACE_KWARGS)

  local_metrics = collections.OrderedDict()
  local_metrics_per_seq = collections.OrderedDict()
  for tracker in trackers:
    logging.info('process tracker %s', tracker)
    # Compute stats for each sequence.
    stats = collections.OrderedDict()
    try:
      for seq in seqs:
        stats[seq] = evaluate_sequence(tracker, seq)
    except Exception as ex:  # pylint: disable=broad-except
      if FLAGS.skip_on_fail:
        logging.warning('skip tracker %s due to error: %s', tracker, ex)
        continue
      else:
        raise

    # Lookup by frame horizon to extend stats to standard horizons.
    extended = collections.OrderedDict()
    for seq in seqs:
      num_frames, fps = metadata[seq]
      time_scale = localmot.horizon_util.units_to_time_scale(
          FLAGS.units, num_frames, fps)
      frame_horizons = localmot.horizon_util.int_frame_horizon(
          horizons, num_frames, time_scale)
      # Lookup frame horizon and set index to time horizon.
      extended[seq] = (stats[seq].loc[frame_horizons]
                       .set_index(pd.Index(horizons, name='horizon')))
    # Take sum over all sequences.
    totals = sum(extended[seq] for seq in seqs)
    print(totals)
    # Normalize to obtain metrics.
    local_metrics[tracker] = localmot.metrics.normalize(totals)

    # Do the same for individual sequences and create one large DataFrame.
    local_metrics_per_seq[tracker] = pd.concat(collections.OrderedDict(zip(
        extended.keys(), map(localmot.metrics.normalize, extended.values()),
    )), names=['sequence'])
    print(local_metrics[tracker])

  local_metrics = pd.concat(local_metrics, names=['tracker'])
  local_metrics_per_seq = pd.concat(local_metrics_per_seq, names=['tracker'])

  local_metrics.to_csv(os.path.join(FLAGS.out_dir, 'metrics_overall.csv'))
  local_metrics_per_seq.to_csv(os.path.join(FLAGS.out_dir, 'metrics.csv'))

  if FLAGS.plot:
    plot_comparisons(local_metrics)


def get_sequence_list():
  """Returns list of sequences using flags."""
  if FLAGS.sequence_file:
    # Take list of sequences from sequence file if provided.
    with open(FLAGS.sequence_file) as f:
      seqs = [seq for seq in map(str.strip, f.readlines()) if seq]
  else:
    # Otherwise discover from directory.
    seqs = [seq for seq in os.listdir(FLAGS.gt_dir)
            if os.path.exists(os.path.join(FLAGS.gt_dir, seq, 'gt', 'gt.txt'))]
    if not seqs:
      raise ValueError('no sequences found in dir', FLAGS.gt_dir)
  return sorted(set(seqs))


def get_tracker_list():
  """Returns list of trackers using flags."""
  if FLAGS.trackers:
    return FLAGS.trackers
  trackers = [tracker for tracker in os.listdir(FLAGS.pr_dir)
              if os.path.isdir(os.path.join(FLAGS.pr_dir, tracker))]
  if not trackers:
    raise ValueError('no trackers found in dir', FLAGS.pr_dir)
  return sorted(trackers)


def get_ignore_categories():
  if FLAGS.challenge:
    return motchallenge.CHALLENGE_TO_IGNORE_CATEGORIES[FLAGS.challenge]
  logging.warning('no challenge specified; no categories marked as ignore')
  return []


def load_sequence_metadata(seq):
  seqinfo_file = os.path.join(FLAGS.gt_dir, seq, 'seqinfo.ini')
  seqinfo = motchallenge.load_seqinfo(seqinfo_file)
  return Metadata(num_frames=int(seqinfo['seqlength']),
                  fps=float(seqinfo['framerate']))


def evaluate_sequence(tracker, seq):
  """Obtains local stats for tracker on sequence."""
  config_str = 'units_{}_diagnostics_{}'.format(
      FLAGS.units, str(FLAGS.diagnostics).lower())
  stats_file = os.path.join(
      FLAGS.out_dir, 'cache', config_str, tracker, seq + '.txt')
  # Try to read results from file if it exists.
  if not FLAGS.ignore_cache and os.path.exists(stats_file):
    logging.info('load cached results: %s', stats_file)
    try:
      stats = pd.read_csv(stats_file, index_col=0)
    except Exception as ex:  # pylint: disable=broad-except
      logging.warning('could not load cached results: %s', ex)
    else:
      return stats
  logging.info('compute stats: %s, %s', tracker, seq)

  num_frames, fps = load_sequence_metadata(seq)
  gt_data = motchallenge.load_gt(
      os.path.join(FLAGS.gt_dir, seq, 'gt', 'gt.txt'))
  pr_data = motchallenge.load_pr(
      os.path.join(FLAGS.pr_dir, tracker, seq + '.txt'))
  gt_data, pr_data = motchallenge.preprocess(
      num_frames, gt_data, pr_data,
      iou_threshold=FLAGS.iou_threshold,
      ignore_categories=get_ignore_categories(),
      vis_threshold=FLAGS.vis_threshold)
  gt_id_subset, pr_id_subset, similarity = motchallenge.to_similarity(
      num_frames, gt_data, pr_data)

  time_scale = localmot.horizon_util.units_to_time_scale(
      FLAGS.units, num_frames, fps)
  horizons = localmot.horizon_util.horizon_range(
      num_frames, time_scale, **LOGSPACE_KWARGS)
  logging.info('horizon in units "%s": %s', FLAGS.units, list(horizons))
  frame_horizons = localmot.horizon_util.int_frame_horizon(
      horizons, num_frames, time_scale)
  logging.info('integer frame horizons: %s', list(frame_horizons))

  stats = localmot.metrics.local_stats_at_int_horizons(
      num_frames, gt_id_subset, pr_id_subset, similarity, frame_horizons,
      similarity_threshold=FLAGS.iou_threshold,
      with_diagnostics=FLAGS.diagnostics)

  logging.info('write stats to file: %s', stats_file)
  os.makedirs(os.path.dirname(stats_file), exist_ok=True)
  stats.to_csv(stats_file)
  return stats


def plot_comparisons(metrics):
  """Generates plots and saves them in the output dir.

  Args:
    metrics: pd.DataFrame with index (tracker, horizon).
  """
  trackers = metrics.index.unique('tracker')

  # Obtain detection and strict metrics from min and max horizons.
  # (Since the horizons range from r < 1 to r >= sequence length.)
  min_horizon = metrics.index.get_level_values('horizon').min()
  max_horizon = metrics.index.get_level_values('horizon').max()
  det_metrics = metrics.xs(min_horizon, level='horizon')
  strict_metrics = metrics.xs(max_horizon, level='horizon')

  # Fix style for all trackers using strict metrics.
  tracker_order = strict_metrics['ata'].sort_values(ascending=False).index
  tracker_styles = get_styles(
      tracker_order, style_file=os.path.join(FLAGS.out_dir, 'style.json'))
  assert max(tracker_styles.values()) < np.lcm(len(COLORS), len(MARKERS)), (
      'not enough colors and markers')
  tracker_colors = {tracker: COLORS[tracker_styles[tracker] % len(COLORS)]
                    for tracker in trackers}
  tracker_markers = {tracker: MARKERS[tracker_styles[tracker] % len(MARKERS)]
                     for tracker in trackers}

  for field in ['ata', 'idf1']:
    plt.figure()
    ax = plt.gca()
    _leave_space_for_legend(ax)
    handles = plots.plot_ratio_scatter(
        strict_metrics[field], det_metrics[field],
        colors=tracker_colors, markers=tracker_markers)
    ax.legend(handles=handles[:20], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('DetF1')
    plt.ylabel(localmot.metrics.NAMES.get(field, field) + ' / DetF1')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    basename = f'ratio_scatter_{field}'
    plt.savefig(os.path.join(FLAGS.out_dir, basename + '.pdf'))
    plt.close()

  for field in ['ata', 'idf1']:
    plt.figure()
    ax = plt.gca()
    _leave_space_for_legend(ax)
    handles = plots.plot_horizon(
        metrics[field], top_k=8, colors=tracker_colors,
        markers=tracker_markers)
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(f'Horizon ({FLAGS.units})')
    plt.ylabel(localmot.metrics.NAMES.get(field, field))
    basename = f'{field}_vs_horizon_{FLAGS.units}'
    plt.savefig(os.path.join(FLAGS.out_dir, basename + '.pdf'))
    plt.close()

  if FLAGS.diagnostics:
    plt.figure(figsize=(10, 10))
    plots.plot_decompose_ata_scatter(
        strict_metrics, level_set_power=-1, colors=tracker_colors,
        markers=tracker_markers, aspect=3, with_inset=True)
    basename = 'scatter_decompose_ata'
    plt.savefig(os.path.join(FLAGS.out_dir, basename + '.pdf'))
    plt.close()

    for field in ['ata']:
      plt.figure()
      plots.plot_decompose_ata_field(
          strict_metrics.sort_values(field, ascending=False), field,
          colors=tracker_colors, markers=tracker_markers,
          legend_kwargs=dict(loc='lower right'))
      plt.ylabel(localmot.metrics.NAMES[field])
      plt.tight_layout()
      basename = f'decompose_{field}'
      plt.savefig(os.path.join(FLAGS.out_dir, basename + '.pdf'))
      plt.close()

    subdir = os.path.join(FLAGS.out_dir, 'detail')
    os.makedirs(subdir, exist_ok=True)
    for tracker in trackers:
      for field in ['ata', 'atr', 'atp']:
        plt.figure()
        plots.plot_decompose_horizon(metrics.loc[tracker], field)
        plt.xlabel(f'Horizon ({FLAGS.units})')
        plt.ylabel(localmot.metrics.NAMES[field])
        plt.title(tracker)
        basename = f'decompose_{field}_vs_horizon_{FLAGS.units}_{tracker}'
        plt.savefig(os.path.join(subdir, basename + '.pdf'))
        plt.close()


def get_styles(tracker_order, style_file=None):
  """Tracker style is zero-based index."""
  tracker_to_style = {}
  if style_file:
    if os.path.exists(style_file):
      with open(style_file) as f:
        tracker_to_style = json.load(f)

  if all(tracker in tracker_to_style for tracker in tracker_order):
    # Already have styles for every tracker.
    return tracker_to_style

  taken = set(tracker_to_style.values())
  available = filter(lambda s: s not in taken, itertools.count())
  for tracker in tracker_order:
    if tracker not in tracker_to_style:
      tracker_to_style[tracker] = next(available)

  if style_file:
    os.makedirs(os.path.dirname(style_file), exist_ok=True)
    with open(style_file, 'w') as f:
      json.dump(tracker_to_style, f)

  return tracker_to_style


def _leave_space_for_legend(ax, box_width_frac=0.7):
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * box_width_frac, box.height])


if __name__ == '__main__':
  app.run(main)
