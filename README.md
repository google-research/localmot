# Local Metrics for Multi-Object Tracking

<p align="center">Jack Valmadre, Alex Bewley, Jonathan Huang, Chen Sun, Cristian Sminchisescu, Cordelia Schmid</p>

<p align="center"><strong>Google Research</strong></p>

Paper: https://arxiv.org/abs/2104.02631


## Introduction

<img src='figures/demo.png' width="500" align="right"/>
<p align="justify">Temporally local metrics for Multi-Object Tracking. These
metrics are obtained by restricting existing metrics based on track matching to
a finite temporal horizon, and provide new insight into the ability of trackers
to maintain identity over time. Moreover, the horizon parameter offers a novel,
meaningful mechanism by which to define the relative importance of detection and
association, a common dilemma in applications where imperfect association is
tolerable. It is shown that the historical Average Tracking Accuracy (ATA)
metric exhibits superior sensitivity to association, enabling its proposed local
variant, ALTA, to capture a wide range of characteristics. In particular, ALTA
is better equipped to identify advances in association independent of detection.
The paper further presents an error decomposition for ATA that reveals the
impact of four distinct error types and is equally applicable to ALTA. The
diagnostic capabilities of ALTA are demonstrated on the MOT 2017 and Waymo Open
Dataset benchmarks.</p>


## Demo

We provide a script to evaluate predictions in
[MOT Challenge format](https://motchallenge.net/instructions/).

First set up the code:

```bash
git clone https://github.com/google-research/localmot
```

Next download the ground-truth data:

```bash
mkdir -p data/gt
( cd data/gt && \
  curl -L -O -C - https://motchallenge.net/data/MOT17Labels.zip && \
  unzip -od MOT17 MOT17Labels.zip )
```

and the predictions of several trackers:

```bash
mkdir -p data/pr/MOT17
( cd data/pr/MOT17 && \
  curl -L -o SORT17.zip -C - "https://motchallenge.net/download_results.php?shakey=d80cceb92629e8236c60129417830bd2fdec8025&name=SORT17&chl=10" && \
  curl -L -o Tracktor++v2.zip -C - "https://motchallenge.net/download_results.php?shakey=b555a1f532c3d161f836fadc8d283fa2100f05c8&name=Tracktor++v2&chl=10" && \
  curl -L -o MAT.zip -C - "https://motchallenge.net/download_results.php?shakey=3af4ae73bef6ece5564ef10931cf49773631b7eb&name=MAT&chl=10" && \
  curl -L -o Fair.zip -C - "https://www.motchallenge.net/download_results.php?shakey=4a2cf604010e9994a49883db083d44ad63e8765a&name=Fair&chl=10" && \
  unzip -o -d SORT17 SORT17.zip && \
  unzip -o -d Tracktor++v2 Tracktor++v2.zip && \
  unzip -o -d MAT MAT.zip && \
  unzip -o -d Fair Fair.zip )
```

To run the evaluation:

```bash
python -m localmot.scripts.eval_motchallenge \
  --gt_dir=data/gt/MOT17/train \
  --pr_dir=data/pr/MOT17 \
  --out_dir=out/MOT17/train \
  --challenge=MOT17 \
  --units=frames \
  --diagnostics \
  --plot
```

Note that the demo script caches the results for each sequence.
Use `--ignore_cache` to ignore and overwrite the cached results.
Separate caches are used for each temporal unit and with/without diagnostics.


## Library usage

```python
import localmot
import numpy as np
import pandas as pd
```

Example using fixed horizons (frames):

```python
horizons = [0, 10, 100, np.inf]

stats = {}
for key, seq in sequences:
  stats[key] = localmot.metrics.local_stats(
      seq.num_frames, seq.gt_id_subset, seq.pr_id_subset, seq.similarity,
      horizons=horizons)
total_stats = sum(stats.values())
metrics = localmot.metrics.normalize(total_stats)
```

Example using fixed horizons (seconds):

```python
horizons = [0, 1, 3, 10, np.inf]
units = 'seconds'

stats = {}
for key, seq in sequences:
  time_scale = localmot.horizon_util.units_to_time_scale(
      units, seq.num_frames, seq.fps)
  stats[key] = localmot.metrics.local_stats(
      seq.num_frames, seq.gt_id_subset, seq.pr_id_subset, seq.similarity,
      horizons=horizons, time_scale=time_scale)
total_stats = sum(stats.values())
metrics = localmot.metrics.normalize(total_stats)
```

Example using full horizon range where all sequences have same length and fps:

```python
num_frames = 1000
fps = 10
units = 'seconds'

time_scale = localmot.horizon_util.units_to_time_scale(units, num_frames, fps)
horizons = localmot.horizon_util.horizon_range(num_frames, time_scale)

stats = {}
for key, seq in sequences:
  stats[key] = localmot.metrics.local_stats(
      num_frames, seq.gt_id_subset, seq.pr_id_subset, seq.similarity,
      horizons=horizons, time_scale=time_scale)
total_stats = sum(stats.values())
metrics = localmot.metrics.normalize(total_stats)
```

Example using full horizon range with heterogeneous sequences:

```python
units = 'seconds'

stats = {}
min_horizon = np.inf
max_horizon = 0
for key, seq in sequences:
  time_scale = localmot.horizon_util.units_to_time_scale(
      units, seq.num_frames, seq.fps)
  horizons = localmot.horizon_util.horizon_range(seq.num_frames, time_scale)
  frame_horizons = localmot.horizon_util.int_frame_horizon(
      horizons, seq.num_frames, time_scale)
  stats[key] = localmot.metrics.local_stats_at_int_horizons(
      seq.num_frames, seq.gt_id_subset, seq.pr_id_subset, seq.similarity,
      horizons=frame_horizons)
  min_horizon = min(min_horizon, min(horizons))
  max_horizon = max(max_horizon, max(horizons))

horizons = localmot.horizon_util.nice_logspace(min_horizon, max_horizon)

extended_stats = {}
for key, seq in sequences:
  time_scale = localmot.horizon_util.units_to_time_scale(
      units, seq.num_frames, seq.fps)
  frame_horizons = localmot.horizon_util.int_frame_horizon(
      horizons, seq.num_frames, time_scale)
  extended_stats[key] = (stats[key].loc[frame_horizons]
                         .set_index(pd.Index(horizons, name='horizon')))

total_stats = sum(extended_stats.values())
metrics = localmot.metrics.normalize(total_stats)
```


## Citation

If you use these local metrics or code for your publication, please cite the
following paper:

```
@article{valmadre2021local,
  title={Local Metrics for Multi-Object Tracking},
  author={Valmadre, Jack and Bewley, Alex and Huang, Jonathan and Sun, Chen and Sminchisescu, Cristian and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2104.02631},
  year={2021}
}
```


## Contact

For bugs or questions about the code, feel free to open an issue on github.
For further queries, send an email to <localmot-dev@google.com>.
