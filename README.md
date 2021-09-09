This is not an officially supported Google product.

# Local Metrics for Multi-Object Tracking

Jack Valmadre, Alex Bewley, Jonathan Huang, Chen Sun, Cristian Sminchisescu, Cordelia Schmid

**Google Research**

Paper: https://arxiv.org/abs/2104.02631


## Introduction

Metrics for Multi-Object Tracking (MOT) can be divided into _strict metrics_, which enforce a fixed, one-to-one correspondence between ground-truth and predicted tracks, and _non-strict metrics_, which award some credit for tracks that are correct in a subset of frames.
IDF1 and ATA are examples of strict metrics, whereas MOTA and HOTA are examples of non-strict metrics.
The type of metric which is appropriate is determined by the priorities of the application.
While strict metrics are relatively uncontroversial, the design of a non-strict metric usually involves two disputable decisions: (i) how to quantify association error and (ii) how to combine detection and association metrics.

**Local metrics** are obtained by applying an existing strict metric locally in a sliding window.
Local metrics represent an alternative way to define a non-strict metric, where the degree of strictness (that is, the balance between detection and association) is controlled via the temporal horizon of the local window.
This replaces the two open questions above with the need to choose a temporal horizon.
Moreover, it provides a single family of metrics that can be used for both strict and non-strict evaluation.
Varying the horizon parameter enables analysis of association error with respect to temporal distance.

One historical weakness of metrics based on one-to-one track correspondence is their lack of transparency with respect to error type.
That is, it can be unclear whether a reduction in overall tracking error is due to improved detection or association (or both).
To address this, we develop a **decomposition** of overall tracking error into four components: over- and under-detection (FN det, FP det) and over- and under-association (merge, split).
The error decomposition is equally applicable to local metrics.


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

Example using fixed horizons (in frames):

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

Example using fixed horizons (in seconds):

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

Example using the spanning range of horizons where all sequences have same length and fps:

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

Example using the spanning range of horizons with heterogeneous sequences:

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

If you use these local metrics or code for your publication, please cite the following paper:

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
