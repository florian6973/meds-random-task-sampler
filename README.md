# MEDS Random Task Sampler

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Model-independent generation of query-based task rows from MEDS datasets.

The package provides two separate workflows, one command each:

| Command                    | Samples                                                                      |
| -------------------------- | ---------------------------------------------------------------------------- |
| `meds-sample-random-tasks` | random `(code, duration)` specifications paired with random patient contexts |
| `meds-generate-task-grid`  | explicit `code x duration` grids at sampled patient prediction times         |

These names describe how rows are sampled, not how a downstream model must use them. For example, either output
could be used for training, validation, benchmarking, probing, or analysis.

Both workflows follow
[`payalchandak/EveryQuery@9bd85a1`](https://github.com/payalchandak/EveryQuery/commit/9bd85a1d2c68000aa9362731c7612007d262ac56).
The package owns the shared task schema, code-source resolution, future-occurrence labeling, death and censoring
semantics, deterministic seeds, and atomic output writes. It does not depend on a model framework. Configuration
is [Hydra](https://hydra.cc), confined to the command layer: the sampling core takes plain Python values and
never sees a Hydra or OmegaConf object.

## Random task samples

```bash
meds-sample-random-tasks \
	data_dir=/path/to/MEDS \
	out_dir=/path/to/random_tasks \
	split=train \
	sampling.num_queries=1024 \
	sampling.min_prediction_times_per_subject=50
```

Output is partitioned under `random_tasks/{split}/*.parquet`; restartable intermediate artifacts use the sibling
`random_tasks_artifacts/{split}/` directory. Machine-readable summary statistics are written to
`random_tasks_artifacts/{split}/_summary.json`.

`sampling.query_codes` defaults to `${data_dir}`, which resolves the full vocabulary from
`{data_dir}/metadata/codes.parquet`. Point it at an explicit list, a `codes.parquet`, or a YAML file to sample
from a narrower universe. `sampling=smoke` swaps in a tiny draw for checking a pipeline end to end.

## Dense task grids

```bash
meds-generate-task-grid \
	data_dir=/path/to/MEDS \
	out_dir=/path/to/task_grid \
	split=held_out \
	'grid.query_codes=[CODE_A,CODE_B]' \
	'grid.durations=[30,90,180,365,731]'
```

Grid rows are written to `task_grid/{split}/{shard}.parquet`. Optional unique prediction times use the sibling
`task_grid_unique/` root and per-shard summaries use `task_grid_summary/`. Nullable/censored labels are retained
by default; use `grid=everyquery_eval` (or `grid.censored_rows=drop`) to reproduce current EveryQuery evaluation
output.

Every shard of the split is built in sorted order. To fan the work out across jobs instead, name one shard per
invocation with `input_shard=0`, or sweep them in one command:

```bash
meds-generate-task-grid --multirun input_shard=0,1,2 data_dir=/path/to/MEDS out_dir=/path/to/task_grid
```

## Zero-shot export: future-only ACES configs

Zero-shot (trajectory-generating) models are scored by generating a future from each
`(subject_id, prediction_time)` and labeling that generated future with the task definition, usually with
`ZSACES_label` from [MEDS_trajectory_evaluation](https://github.com/mmcdermott/MEDS_trajectory_evaluation). Set
`aces_dir` on either command to also write one future-only ACES config per distinct sampled task:

```bash
meds-sample-random-tasks data_dir=/path/to/MEDS out_dir=/path/to/random_tasks aces_dir=/path/to/zs_tasks
```

```text
zs_tasks/{split}/
|-- tasks.parquet             # task_id, query, duration_days, window, criteria_fp
|-- prediction_times.parquet  # random workflow only: contexts to generate trajectories from
`-- criteria/{task_id}.yaml   # pass as `task.criteria_fp` to ZSACES_label
```

The dense workflow exports its full `query_codes x durations` axes (identically from every shard job); its
generation contexts are the existing `{out_dir}_unique/` rows. `task_id` is a stable hash of
`(query, float32 duration_days)`; join each task's zero-shot labels to the sampled rows on
`(subject_id, prediction_time)` plus the manifest's `(query, duration_days)`.

**Reproducibility.** All randomness lives in the sampler and is fixed by `seed`. The export draws nothing: it
is a deterministic function of the sampled rows (or grid axes), so the same seed gives byte-identical configs.
There is no need to run ACES over the real data to rebuild the cohort; the sampled rows already carry the
prediction times and the ground-truth labels. Prediction times also do not need to line up with a special
trigger event: they are already event timestamps, and the zero-shot labeler anchors on each trajectory's
`prediction_time`, so the configs use a nominal `_ANY_EVENT` trigger.

**Window.** Each config encodes exactly the sampler's `(t, t + D]`. ZSACES requires the label window to sit
below the prediction-time node, so the horizon is split into an empty open gap `(t, t + 1us)` and a closed
target `[t + 1us, t + D]`. Together these cover the same microsecond timestamps. `D` comes from the same Polars
expression the labeling kernel uses (`window_length_expr`), including its float32 arithmetic. The tests check
that the labeler, run on each subject's *real* future, reproduces the sampler's label on every uncensored row.
That test runs when `MEDS_trajectory_evaluation` is installed.

**Caveats.** The sampler labels a window that runs past the real record as `null` (censored), and it treats
death inside the window as full observation (so the label is false unless the code occurred first). ACES
windows cannot express either rule. For a generated trajectory, determinability is the zero-shot labeler's
call, so generate past `t + D` or handle trajectories that stop early (e.g. at `MEDS_DEATH`) explicitly.

## Configuration

Every setting lives in a YAML file under
[`src/meds_random_task_sampler/configs/`](src/meds_random_task_sampler/configs/) and can be overridden on the
command line. `--help` prints the fully composed config and the available config groups; `--cfg job` prints just
the config a run would use, without running it.

| Key           | Meaning                                                                        |
| ------------- | ------------------------------------------------------------------------------ |
| `data_dir`    | MEDS dataset root (required)                                                   |
| `out_dir`     | final-output root (required); sibling roots derive their names from it         |
| `split`       | which MEDS split to read                                                       |
| `seed`        | seeds every draw; the query and context axes reproduce independently           |
| `overwrite`   | redo work whose output already exists instead of skipping it                   |
| `log_dir`     | where Hydra writes its run log and resolved-config snapshot; never a data root |
| `input_shard` | dense grid only: build one named shard instead of all of them                  |
| `aces_dir`    | optional: export zero-shot ACES task configs to `${aces_dir}/${split}/`        |
| `sampling.*`  | the `RandomTaskSamplerConfig` fields — options: `default`, `smoke`             |
| `grid.*`      | the `TaskGridGeneratorConfig` fields — options: `default`, `everyquery_eval`   |

Each config group holds exactly the fields of its dataclass, so a key that drifts from the dataclass fails at the
command boundary rather than deep in a stage. To keep site-specific defaults outside the package, copy the config
directory and compose against it with `--config-dir /path/to/my_configs`.

## Python API

The commands are a thin shell over the public API, which takes ordinary Python values:

```python
from meds_random_task_sampler import RandomTaskSamplerConfig, sample_random_tasks

result = sample_random_tasks(
    data_dir="/path/to/MEDS",
    output_dir="/path/to/random_tasks",
    split="train",
    config=RandomTaskSamplerConfig(
        num_queries=1024,
        num_contexts_per_query=1,
        min_prediction_times_per_subject=50,
        query_codes="/path/to/MEDS",
    ),
)
```

`generate_task_grid` / `generate_task_grids` and `TaskGridGeneratorConfig` are the dense-grid equivalents.

See [DESIGN.md](DESIGN.md) for the behavioral contract and planned EveryQuery adapter boundary.

## Development

```bash
uv sync --group dev
uv run pytest -v
uv run pre-commit run --all-files
```

This repository retains the
[`McDermottHealthAI/MHAL-template`](https://github.com/McDermottHealthAI/MHAL-template) project structure.
