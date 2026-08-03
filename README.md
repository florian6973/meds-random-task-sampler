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
