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

"""Tools for producing plots like in the paper."""

__all__ = [
    'plot_horizon',
    'plot_decompose_ata_scatter',
    'plot_decompose_ata_field',
    'plot_decompose_horizon',
    'plot_ratio_scatter',
    'plot_level_sets',
]

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np


def plot_horizon(
    metric,
    trackers=None,
    mode='value',
    colors=None,
    markers=None,
    name_map=None,
    top_k=None,
    marker_freq=3,
    ax=None):
  """Plots local metric versus temporal horizon.

  Args:
    metric: pd.Series with index levels (tracker, horizon).
    trackers: Optional subset of trackers to plot.
    mode: One of 'value', 'rank', 'relative'.
    colors: Dict that maps tracker to color.
    markers: Dict that maps tracker to marker.
    name_map: Dict that maps tracker to "nice" name.
    top_k: Plot this many best trackers (best at any horizon).
    marker_freq: Period at which to place markers.
    ax: Axes on which to plot.

  Returns:
    List of line handles for legend.
  """
  ax = _set_or_get_axis(ax)
  trackers = trackers or metric.index.unique('tracker')
  name_map = name_map or {}

  top_k = top_k or len(trackers)
  # Obtain (tracker, horizon) table.
  values = metric.unstack()
  horizons = values.columns
  # Sort trackers by metric at largest horizon.
  values = values.sort_values(horizons[-1], ascending=False)
  # Obtain rank of trackers at each horizon.
  ranks = values.rank(method='min', ascending=False)

  tracker_mask = (ranks <= top_k).any(axis=1)
  if mode == 'rank':
    excluded = ranks.loc[~tracker_mask]
    values = ranks.loc[tracker_mask]
  else:
    excluded = values.loc[~tracker_mask]
    values = values.loc[tracker_mask]
    if mode == 'relative':
      excluded = excluded / values.max(axis=0)
      values = values / values.max(axis=0)
  trackers = values.index
  order = dict(zip(trackers, range(len(trackers))))

  if excluded.size:
    plt.plot(horizons, excluded.T, color='0.8')

  # Draw in reverse order to put best on top.
  tracker_to_handle = {}
  for tracker in reversed(trackers):
    color = colors[tracker] if colors is not None else None
    marker = markers[tracker] if markers is not None else None
    label = name_map.get(tracker, tracker)
    handle, = plt.plot(
        horizons, values.loc[tracker],
        fillstyle='none', color=color, marker=marker,
        markevery=slice(order[tracker] % marker_freq, None, marker_freq),
        label=label)
    tracker_to_handle[tracker] = handle

  # Re-draw markers to ensure they are above lines!
  for tracker in reversed(trackers):
    color = colors[tracker] if colors is not None else None
    marker = markers[tracker] if markers is not None else None
    plt.plot(
        horizons, values.loc[tracker],
        fillstyle='none', color=color, marker=marker,
        markevery=slice(order[tracker] % marker_freq, None, marker_freq),
        label=None, linestyle='none')

  plt.xscale('log')
  ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

  if mode == 'rank':
    plt.ylim(top_k + 0.5, 0.5)
    plt.yticks(np.arange(top_k) + 1)
  else:
    plt.ylim(values.min().min() - 0.02, values.max().max() + 0.02)

  handles = [tracker_to_handle[tracker] for tracker in trackers]
  return handles


def plot_decompose_ata_scatter(
    metrics,
    level_set_power=None,
    colors=None,
    markers=None,
    aspect=1,
    with_inset=False,
    fig=None):
  """Plots precision-recall scatter for ATA and decompositions of each.

  Args:
    metrics: pd.DataFrame with index levels (tracker,).
    level_set_power: Power in generalized mean for level sets.
    colors: Dict that maps tracker to color.
    markers: Dict that maps tracker to marker.
    aspect: Ratio of decomposition to scatter plot size.
    with_inset: Draw inset in bottom left quadrant?
    fig: Figure in which to plot.
  """
  fig = fig or plt.gcf()
  (ax_pr, ax_2d), (ax_spare, ax_gt) = fig.subplots(
      2, 2, sharex='col', sharey='row',
      gridspec_kw={'width_ratios': (aspect, 1), 'height_ratios': (1, aspect)})
  fig.delaxes(ax_spare)

  ax_pr.set_ylabel('Average Track Precision')
  ax_gt.set_xlabel('Average Track Recall')
  plt.tight_layout()
  ax_2d.set_zorder(0)
  ax_pr.set_zorder(1)
  ax_gt.set_zorder(1)

  x_field, y_field = 'atr', 'atp'
  metrics = metrics.sort_values('ata', ascending=False)
  rank = metrics['ata'].rank(method='min', ascending=False)
  trackers = list(metrics.index)

  def _plot_scatter(ax):
    for tracker in trackers:
      row = metrics.loc[tracker]
      x = row[x_field]
      y = row[y_field]
      color = colors[tracker] if colors is not None else None
      marker = markers[tracker] if markers is not None else None
      label = '{:.3f} ({:g}) {:s}'.format(row['ata'], rank[tracker], tracker)
      ax.plot(x, y, color=color, zorder=3, marker=marker, label=label)
    # Disable auto-scaling before plotting level sets.
    if level_set_power is not None:
      plot_level_sets(ax, level_set_power, num_levels=40, num_points=1000,
                      color='0.8', zorder=1, linewidth=1)

  _plot_scatter(ax_2d)
  ax_2d.set_xlim(0, 1)
  ax_2d.set_ylim(0, 1)

  if with_inset:
    clearance = 0.6
    plot_size = (aspect - 0.1) - clearance
    ax_inset = ax_2d.inset_axes([-(clearance + plot_size),
                                 -(clearance + plot_size),
                                 plot_size,
                                 plot_size])
    x_min, x_max = metrics[x_field].min(), metrics[x_field].max()
    y_min, y_max = metrics[y_field].min(), metrics[y_field].max()
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    region_size = np.clip(max(x_max - x_min, y_max - y_min) + 2 * 0.05, None, 1)
    x_center = np.clip(x_center, region_size / 2, 1 - region_size / 2)
    y_center = np.clip(y_center, region_size / 2, 1 - region_size / 2)
    ax_inset.set_xlim(x_center - region_size / 2, x_center + region_size / 2)
    ax_inset.set_ylim(y_center - region_size / 2, y_center + region_size / 2)
    _plot_scatter(ax_inset)
    ax_2d.indicate_inset_zoom(ax_inset)

  # Precision plot (upper left).
  plt.sca(ax_pr)
  ax_pr.xaxis.set_tick_params(which='both', labelbottom=True)
  plot_decompose_ata_field(
      metrics.sort_values('atp'), field='atp',
      horizontal=False, show_markers=True, colors=colors, markers=markers,
      legend_kwargs=dict(
          framealpha=1, loc='upper left', bbox_to_anchor=(-0.15 / aspect, -0.3),
          title='Precision error (predicted tracks)'))
  plt.ylabel('ATP')

  # Recall plot (lower right).
  plt.sca(ax_gt)
  ax_gt.yaxis.set_tick_params(which='both', labelbottom=True)
  plot_decompose_ata_field(
      metrics.sort_values('atr'), field='atr',
      horizontal=True, show_markers=True, colors=colors, markers=markers,
      legend_kwargs=dict(
          framealpha=1, loc='lower right', bbox_to_anchor=(-0.3, -0.1 / aspect),
          title='Recall error (ground-truth tracks)'))
  plt.xlabel('ATR')


def plot_decompose_ata_field(
    metrics, field,
    colors=None,
    markers=None,
    horizontal=False,
    show_markers=False,
    legend_kwargs=None,
    ax=None):
  """Generates decomposition bar plot for ATA, ATR or ATP.

  Sub-routine of plot_decompose_ata_scatter.

  Args:
    metrics: pd.DataFrame with index (tracker).
    field: Field of which to plot decomposition, one of ata, atr, atp.
    colors: Dict that maps tracker to color.
    markers: Dict that maps tracker to marker.
    horizontal: Use barh instead of bar?
    show_markers: Show markers?
    legend_kwargs: Optional args to legend().
    ax: Axes on which to plot.
  """
  ax = _set_or_get_axis(ax)
  legend_kwargs = legend_kwargs or {}

  field_to_components = {
      'ata': [
          'ata_error_det_fn',
          'ata_error_det_fp',
          'ata_error_ass_split',
          'ata_error_ass_merge',
          'ata_approx',
      ],
      'atr': [
          'atr_error_cover_det',
          'atr_error_cover_ass_indep',
          'atr_error_cover_ass_joint',
          'atr_error_union_ass',
          'atr_error_union_det',
          'atr_approx',
      ],
      'atp': [
          'atp_error_cover_det',
          'atp_error_cover_ass_indep',
          'atp_error_cover_ass_joint',
          'atp_error_union_ass',
          'atp_error_union_det',
          'atp_approx',
      ],
  }

  field_names = {
      # ATA.
      'ata_approx': None,
      'ata_error_ass_merge': 'Merge',
      'ata_error_ass_split': 'Split',
      'ata_error_det_fp': 'Det FP',
      'ata_error_det_fn': 'Det FN',
      # Track recall.
      'atr_approx': None,
      'atr_error_cover_det': 'FN det',
      'atr_error_cover_ass_indep': 'Split (contains multiple)',
      'atr_error_cover_ass_joint': 'Merge (best match unavailable)',
      'atr_error_union': 'union (FP, Merge)',
      'atr_error_union_det': 'FP det (in partner, not in track)',
      'atr_error_union_ass': 'Merge (in partner, not in track)',
      # Track precision.
      'atp_approx': None,
      'atp_error_cover_det': 'FP det',
      'atp_error_cover_ass_indep': 'Merge (contains multiple)',
      'atp_error_cover_ass_joint': 'Split (best match unavailable)',
      'atp_error_union': 'union (FN, split)',
      'atp_error_union_det': 'FN det (in partner, not in track)',
      'atp_error_union_ass': 'Split (in partner, not in track)',
  }

  cmap = cm.get_cmap('Pastel1')
  good = [cmap(2)]
  bad = [cmap(0), cmap(4), cmap(1), cmap(3), cmap(8)]
  field_colors = {
      # ATA.
      'ata_approx': good[0],
      'ata_error_det_fn': bad[0],
      'ata_error_det_fp': bad[1],
      'ata_error_ass_split': bad[2],
      'ata_error_ass_merge': bad[3],
      # Track recall.
      'atr_approx': good[0],
      'atr_error_cover_det': bad[0],  # Det FN
      'atr_error_cover_ass_indep': bad[2],  # Assoc split
      'atr_error_cover_ass_joint': bad[3],  # Assoc merge
      'atr_error_union': bad[4],
      'atr_error_union_det': bad[1],  # Det FP
      'atr_error_union_ass': bad[3],  # Assoc merge
      # Track precision.
      'atp_approx': good[0],
      'atp_error_cover_det': bad[1],  # Det FP
      'atp_error_cover_ass_indep': bad[3],  # Assoc merge
      'atp_error_cover_ass_joint': bad[2],  # Assoc split
      'atp_error_union': bad[4],
      'atp_error_union_det': bad[0],  # Det FN
      'atp_error_union_ass': bad[2],  # Assoc split
  }

  components = field_to_components[field]
  num_trackers = len(metrics.index)
  bar_fn = ax.barh if horizontal else ax.bar
  legend_handles = []
  start = np.zeros(num_trackers)
  for component in reversed(components):
    hatch = '////' if 'union_' in component else ''
    label = field_names.get(component, component)
    h = bar_fn(np.arange(num_trackers), metrics[component], 0.8, start,
               color=field_colors.get(component),
               label=label,
               hatch=hatch,
               tick_label=metrics.get('display_name', metrics.index),
               linewidth=1,
               edgecolor='0.95')
    if label:
      # Avoid adding empty labels to legend as this can raise a warning.
      legend_handles.append(h)
    start += metrics[component]

  if show_markers:
    for i, tracker in enumerate(metrics.index):
      row = metrics.loc[tracker]
      value = row[field]
      color = colors[tracker] if colors is not None else None
      marker = markers[tracker] if markers is not None else None
      # fillstyle = _get_marker_fillstyle(row['is_private'])
      if not horizontal:
        x, y = i, value
      else:
        y, x = i, value
      ax.plot(x, y, color=color, marker=marker, zorder=4)

  if not horizontal:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')
    ax.set_xlim(-1, len(metrics.index))
  else:
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
    ax.set_ylim(-1, len(metrics.index))
  ax.legend(handles=list(reversed(legend_handles)), **legend_kwargs)


def plot_decompose_horizon(metrics, field='ata', ax=None):
  """Plots error decomposition versus temporal horizon for a single tracker.

  Args:
    metrics: pd.DataFrame with index horizon.
    field: Column to plot. One of ata, atr, atp.
    ax: Axes on which to plot.
  """
  ax = _set_or_get_axis(ax)
  horizons = metrics.index.unique('horizon')
  subfields = [
      f'{field}_approx',
      f'{field}_error_ass_merge',
      f'{field}_error_ass_split',
      f'{field}_error_det_fp',
      f'{field}_error_det_fn',
  ]
  names = {
      f'{field}_approx': 'Correct (approx)',
      f'{field}_error_det_fn': 'Error: det, false neg',
      f'{field}_error_det_fp': 'Error: det, false pos',
      f'{field}_error_ass_split': 'Error: assoc, split',
      f'{field}_error_ass_merge': 'Error: assoc, merge',
  }

  cmap = cm.get_cmap('Pastel1')
  good = [cmap(2)]
  bad = [cmap(0), cmap(4), cmap(1), cmap(3), cmap(8)]
  colors = {
      f'{field}_approx': good[0],
      f'{field}_error_det_fn': bad[0],
      f'{field}_error_det_fp': bad[1],
      f'{field}_error_ass_split': bad[2],
      f'{field}_error_ass_merge': bad[3],
  }

  subfield_handles = ax.stackplot(
      horizons, metrics[subfields].transpose(),
      colors=[colors[subfield] for subfield in subfields])
  correct_handle, = ax.plot(
      horizons, metrics[field], color='black', label='Correct',
      linestyle='dashed', linewidth=2)
  for handle, pattern in zip(subfield_handles, ['', '-', '|', '\\', '/']):
    if pattern:
      handle.set_hatch(pattern * 2)
      handle.set_edgecolor('white')
  plt.xscale('log')
  ax.set_xlim(min(horizons), max(horizons))
  ax.set_ylim(0, 1)
  ax.legend(
      list(reversed(subfield_handles)) + [correct_handle],
      list(reversed([names[subfield] for subfield in subfields])) + ['Correct'],
      loc='lower left')


def plot_ratio_scatter(
    metric, det_metric,
    colors=None,
    markers=None,
    ax=None):
  """Plots error decomposition versus temporal horizon for a single tracker.

  Args:
    metric: pd.Series with index tracker.
    det_metric: pd.Series with index tracker.
    colors: Dict that maps tracker to color.
    markers: Dict that maps tracker to marker.
    ax: Axes on which to plot.
  """
  ax = _set_or_get_axis(ax)
  order = metric.sort_values(ascending=False).index
  metric = metric[order]
  det_metric = det_metric[order]
  ratio = metric / det_metric
  rank = metric.rank(method='min', ascending=False)

  handles = []
  for tracker in order:
    color = colors[tracker] if colors is not None else None
    marker = markers[tracker] if markers is not None else None
    label = '{:.3f} ({:g}) {:s}'.format(
        metric[tracker], rank[tracker], tracker)
    h, = ax.plot(
        det_metric[tracker], ratio[tracker], color=color, zorder=3,
        marker=marker, label=label)
    handles.append(h)

  # Plot level sets of geometric mean (product).
  plot_level_sets(ax, p=0, num_levels=40, num_points=1000,
                  color='0.8', zorder=1, linewidth=1)

  return handles


def plot_level_sets(ax, p, num_levels=10, num_points=100, **plot_kwargs):
  """Plots levels sets of p-mean in [0, 1]^2."""
  x = np.arange(1, num_points + 1) / num_points
  alpha = np.arange(1, num_levels) / num_levels
  x = x[:, None]
  alpha = alpha[None, :]
  if p == 0:
    # x y = alpha**2
    y = alpha ** 2 / x
  elif p == np.inf:
    raise NotImplementedError
  elif p == -np.inf:
    raise NotImplementedError
  else:
    # There may be some (alpha, x) where y does not exist. Set to nan.
    yp = 2 * alpha**p - x**p
    with np.errstate(divide='ignore', invalid='ignore'):
      y = np.where(yp > 0, yp**(1 / p), np.nan)
  # Draw level sets separately on either side of y = x.
  # This avoids issues with coarse resolution at small values of x.
  # Ensure equality is exact when alpha = x so that the two lines coincide.
  y = np.where(alpha == x, x, y)
  # Note that y <= x is equivalent to alpha <= x.
  x_masked = np.where(alpha <= x, x, np.nan)
  ax.plot(x_masked, y, **plot_kwargs)
  ax.plot(y, x_masked, **plot_kwargs)


def _set_or_get_axis(ax):
  if ax is None:
    ax = plt.gca()
  else:
    plt.sca(ax)
  return ax
