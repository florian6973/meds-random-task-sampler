# MEDS Task Generation

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Generate deterministic, model-independent collections of `(code, horizon)` prediction tasks and labels
from a MEDS dataset.

> [!WARNING]
> This package is an early prototype. Its schemas and CLI may change before the first tagged release.

## Quick start

This repository follows the
[`McDermottHealthAI/MHAL-template`](https://github.com/McDermottHealthAI/MHAL-template) conventions and
uses `uv`:

```bash
uv sync --group dev
uv run meds-task-generation generate \
	--data-dir /path/to/MEDS \
	--config collection.yaml \
	--output-dir generated_collection
```

The generated collection reuses the subject assignments in
`metadata/subject_splits.parquet`; it never creates replacement patient splits. Output includes a resolved
`manifest.yaml`, task labels, prediction-time indexes, `summary.parquet`, and `summary.json`.

See [DESIGN.md](DESIGN.md) for scope, schema rationale, and planned MEDS-DEV integration.

## Example configuration

```yaml
schema_version: 1
metadata:
  name: demo
  description: Small code-occurrence collection.
seed: 1
subjects:
  splits: [train, tuning, held_out]
  subsample_fraction: 1.0
prediction_times:
  strategy: random_event_time
  count_per_subject: 1
  minimum_prior_events: 5
tasks:
  type: code_occurrence
  query_codes:
    source: explicit
    values: [DIAGNOSIS//A, DIAGNOSIS//B]
  horizons_days: [7, 30]
labeling:
  window:
    start_inclusive: false
    end_inclusive: true
  censoring:
    policy: preserve
output:
  partition_by: [split]
```

## Commands

```bash
meds-task-generation generate --data-dir MEDS --config collection.yaml --output-dir collection
meds-task-generation validate --collection-dir collection
meds-task-generation summarize --collection-dir collection
```

## Public MIMIC-IV demo

Build the demo with MEDS-DEV, then generate and validate the included six-task collection:

```bash
meds-dev-dataset dataset=MIMIC-IV demo=True output_dir=/tmp/mimic-iv-demo
meds-task-generation generate \
	--data-dir /tmp/mimic-iv-demo \
	--config examples/mimic_demo.yaml \
	--output-dir /tmp/mimic-task-collection
meds-task-generation validate --collection-dir /tmp/mimic-task-collection
```

The example uses two diagnosis codes and horizons of 7, 30, and 365 days. It is intentionally small enough
for a public integration test; dataset and generated artifacts remain outside the repository.

## Development

```bash
uv sync --group dev
uv run pytest -v
uv run pre-commit run --all-files
```

No clinical or synthetic datasets are committed to this repository. Tests construct temporary MEDS fixtures
at runtime.
