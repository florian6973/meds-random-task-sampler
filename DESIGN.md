# MEDS Random Task Sampler: EQ-Derived Design

**Status:** Initial random-sample and dense-grid implementation

**Repository:** `meds-random-task-sampler`

**Python module:** `meds_random_task_sampler`

**Behavioral reference:** `payalchandak/EveryQuery@9bd85a1d2c68000aa9362731c7612007d262ac56`
(2026-07-18)

**Reference inspected:** 2026-07-22

## 1. Decision

The first implementation will extract both of EveryQuery's current task-generation workflows into a
model-independent MEDS package:

1. sparse random tasks for model training; and
2. dense fixed-task grids for model evaluation.

EveryQuery is the behavioral reference for both workflows, including their distinct sampling rules,
random-number behavior, task-row schema, prediction-time eligibility, sharding constraints, death handling,
censoring precedence, restart behavior, and output layout.

The removed prototype is not an implementation base. In particular, the new package will not begin with an
all-shards-in-memory evaluation grid and incrementally turn it into a training sampler. It will begin with the
five-stage EQ training architecture.

The initial success criterion is behavioral equivalence with the pinned EQ commit for both workflows on frozen
compatibility fixtures. Generalization comes after equivalence.

## 2. Why this package should exist

EveryQuery currently owns infrastructure that is useful beyond the EveryQuery model:

- resolving a MEDS code universe;
- constructing eligible patient prediction-time maps;
- sampling random `(code, duration)` query specifications;
- sampling random patient contexts;
- pairing queries and contexts deterministically;
- labeling future code occurrence with censoring and terminal-event semantics;
- partitioning work by MEDS shard;
- atomically writing restartable sampled-task datasets;
- constructing dense evaluation grids over fixed `(code, duration)` tasks;
- filtering censored evaluation rows and optionally writing unique prediction times.

None of these operations requires an EveryQuery neural network, PyTorch, Lightning, or a learned query
embedding. Moving them to a small package lets another model consume the same random training distribution
without importing EveryQuery's model stack. The sampling core needs no Hydra either: Hydra enters only through
the package's own command-line layer (section 8), which converts a composed config into the plain dataclasses
below and is the single module allowed to import it.

EveryQuery should retain thin adapters so existing `EQ_generate_training_tasks` and
`EQ_generate_evaluation_tasks` commands continue to work.

## 3. Source of truth and provenance

The reference implementation is:

- `src/every_query/generate_tasks/sample_tasks.py`;
- `src/every_query/generate_tasks/sample_evaluation_tasks.py`;
- `src/every_query/generate_tasks/redesign-spec.md`;
- `src/every_query/generate_tasks/configs/sample_training_tasks_config.yaml`;
- `src/every_query/generate_tasks/configs/sample_evaluation_tasks_config.yaml`;
- `src/every_query/data/schema.py`;
- `src/every_query/utils/seeds.py`;
- `tests/sampler/` and the sampler integration tests.

The reference commit is recorded above so later EQ changes do not silently change this design's meaning.
Before extraction begins, the relevant EveryQuery tests will be copied as behavioral specifications with
appropriate attribution under the compatible MIT license. Implementation code should be rewritten behind the
package API rather than copied wholesale without an ownership review.

Any deliberate divergence from the reference must be documented in a compatibility ledger with:

1. the EQ behavior;
2. the package behavior;
3. the rationale;
4. an explicit test;
5. the expected effect on previously generated rows.

## 4. Package workflows

Version 0.1 will provide two explicit public workflows that share infrastructure but produce different task
shapes.

### 4.1 Random task samples

```text
MEDS event shards
    + query-code universe
    + query distribution
    + patient-context distribution
    -> sparse labeled random task rows
```

One output row asks:

> For `subject_id` at `prediction_time`, does `query` occur in
> `(prediction_time, prediction_time + duration_days]`?

The initial sampler supports:

- a single MEDS code per query;
- continuous positive durations in days;
- uniform code sampling;
- uniform or log-uniform duration sampling;
- random subjects sampled with replacement;
- random eligible prediction times sampled with replacement through their subject draws;
- nullable Boolean occurrence labels;
- one MEDS split per invocation;
- shard-parallel labeling on one node;
- deterministic restartable output.

### 4.2 Dense task grids

```text
one MEDS event shard
    + fixed query-code universe
    + fixed integer duration horizons
    + per-subject prediction-time sampling
    -> dense labeled evaluation rows
    -> censored rows removed
    -> optional unique prediction-time rows
```

For every sampled `(subject_id, prediction_time)`, evaluation constructs the full Cartesian product of caller-
specified query codes and duration horizons. This produces the dense shape needed to calculate metrics for every
`(query, duration_days)` task over a split.

To match EQ, evaluation:

- operates on one named input shard per invocation, allowing external parallelism across shards;
- samples up to `prediction_times_per_subject` candidate times without replacement for each subject;
- defines eligibility using the count of prior events, not the training sampler's distinct-prediction-time map;
- optionally applies deterministic subject subsampling;
- uses fixed caller-provided integer duration horizons rather than sampled continuous durations;
- explicitly keeps or drops rows whose `boolean_value` is null after labeling; and
- optionally writes deduplicated `(subject_id, prediction_time)` rows for trajectory generation.

`random_sample` and `dense_grid` deliberately remain separate entry points and typed configurations. These names
describe sampling shape rather than prescribing downstream use as training or evaluation. A mode flag on one
ambiguous API would hide meaningful differences in sampling unit, eligibility, sharding, censor handling, and
output cardinality.

## 5. Non-goals for version 0.1

The package will not initially:

- train or load EveryQuery;
- depend on PyTorch, Lightning, Transformers, or Weights & Biases;
- own EveryQuery's Hydra configuration, its key names, or its adapters — the package ships its own two-command
    CLI and config groups instead (section 8);
- tensorize MEDS data;
- create patient train/tuning/test assignments;
- calculate AUROC or other metrics;
- sample EveryQuery's validation tracking pairs;
- define ACES predicates or composite clinical outcomes;
- weight codes by empirical frequency;
- support distributed multi-node labeling;
- promise stability for artifacts explicitly marked internal.

The package generates EQ-compatible evaluation labels; it does not own model inference or metric calculation.

## 6. Ownership boundary

### 6.1 Package-owned

The package owns:

- core dataclasses and validation;
- `TaskQuerySchema` and its empty-frame/schema-alignment helpers;
- stable seed derivation;
- query-code resolution;
- random query sampling;
- eligible prediction-time map construction;
- random patient-context sampling;
- query/context pairing;
- per-subject evaluation-time sampling;
- dense evaluation-grid construction;
- evaluation censor filtering and unique-time output;
- shard partitioning;
- occurrence labeling;
- death truncation and censoring semantics;
- atomic writes and cache validation;
- row-count and schema validation;
- provenance and summary metadata;
- the `meds-sample-random-tasks` and `meds-generate-task-grid` commands, their Hydra config groups, and the
    conversion from a composed config into the core dataclasses.

### 6.2 EveryQuery-owned

EveryQuery retains:

- the `EQ_generate_training_tasks` and `EQ_generate_evaluation_tasks` commands;
- Hydra configuration and command-line overrides;
- translation from `DictConfig` into package dataclasses;
- temporary re-exports of moved sampler and schema symbols during migration;
- downstream `EveryQueryPytorchDataset` behavior;
- token/vocabulary consistency with the trained model;
- model training, prediction, tracking callbacks, and metrics.

The adapter may re-export package types temporarily to avoid breaking internal imports during migration.

### 6.3 MEDS-DEV-owned

MEDS-DEV may invoke an EveryQuery model recipe that internally requests random pretraining tasks. Random
training rows are not a benchmark definition and should not become a MEDS-DEV task stage. Patient splits remain
the MEDS dataset's existing splits.

## 7. Public API

The core must not accept Hydra or OmegaConf objects: `meds_random_task_sampler.cli` is the only module that
imports either, and it hands the core plain dataclasses, lists, and paths. Random sampling and dense-grid
generation have separate top-level APIs:

```python
from pathlib import Path

from meds_random_task_sampler import (
    TaskGridGeneratorConfig,
    QueryDistribution,
    RandomTaskSamplerConfig,
    generate_task_grid,
    sample_random_tasks,
)

config = RandomTaskSamplerConfig(
    num_queries=1024,
    num_contexts_per_query=1,
    min_prediction_times_per_subject=50,
    query_distribution=QueryDistribution(
        query_codes=("A", "B"),
        min_duration_days=1.0,
        max_duration_days=731.0,
        duration_distribution="log-uniform",
    ),
    seed=1,
    max_workers=None,
)

result = sample_random_tasks(
    data_dir=Path("/path/to/MEDS"),
    output_dir=Path("/path/to/training_tasks"),
    split="train",
    config=config,
    overwrite=False,
)

grid_config = TaskGridGeneratorConfig(
    prediction_times_per_subject=1,
    min_context_per_subject=50,
    query_codes=("A", "B"),
    durations=(30, 90, 180, 365, 731),
    subject_subsample_fraction=None,
    write_unique_prediction_times=True,
    seed=1,
)

grid_result = generate_task_grid(
    data_dir=Path("/path/to/MEDS"),
    output_dir=Path("/path/to/task_grid"),
    split="held_out",
    input_shard="0",
    config=grid_config,
    overwrite=False,
)
```

Each result is a small immutable run-result object containing output paths and counts, not the full generated
dataset in memory.

Pure stage functions remain public enough for testing and advanced orchestration:

```python
build_prediction_time_map(...)
sample_queries(...)
sample_patient_contexts(...)
build_partitioned_index(...)
label_index_partition(...)
label_partitions(...)
sample_prediction_times_per_subject(...)
build_task_grid(...)
label_task_grid_shard(...)
```

File-system orchestration should depend on these pure or narrowly stateful primitives, rather than combining
all behavior inside the CLI.

## 8. Configuration model

Core configuration uses frozen dataclasses or an equivalently strict typed model. It must reject Boolean values
where integers are expected and reject non-finite numeric bounds.

Those dataclasses are also the package's command-line surface. `src/meds_random_task_sampler/configs/` holds one
Hydra root config per workflow — `random_sample.yaml` and `dense_grid.yaml` — each with a defaults list selecting
one option from its workflow's config group. A group holds exactly the fields of its dataclass, so composition is
a keyword splat and a key that drifts from the dataclass fails at the boundary, naming itself. Path roots, split,
seed, and `overwrite` stay at the top level because both workflows share them.

Hydra is confined to `cli.py`: it composes the config, resolves interpolations, converts every container to a
builtin, and calls the same public API a Python caller would. Nothing below that module can tell it was invoked
from a command line.

### 8.1 Training configuration

`meds-sample-random-tasks`, composed from `random_sample.yaml` and the `sampling` group:

```yaml
defaults:
  - sampling: default # or `smoke`
  - _self_

data_dir: ??? # MEDS root; also where `metadata/codes.parquet` is read from
out_dir: ??? # rows land in `${out_dir}/${split}/{shard}.parquet`
split: train
seed: 1
overwrite: false
log_dir: ${oc.select:out_dir,outputs}_logs # Hydra's own run log; never a data root

sampling:
  num_queries: 1024
  num_contexts_per_query: 1
  min_prediction_times_per_subject: 50
  query_codes: ${data_dir}
  min_duration: 1.0
  max_duration: 731.0
  duration_distribution: log-uniform
  max_workers:
```

The package resolves `query_codes` before sampling. It accepts an ordinary Python sequence, a YAML or Parquet
path, or a MEDS metadata root containing `metadata/codes.parquet`; it never accepts Hydra `ListConfig` directly.
Explicit sequences preserve order while removing duplicates. File-derived codes are deduplicated and sorted.

The first EveryQuery training adapter maps its existing Hydra keys onto this model. Renaming EQ's CLI keys is
not part of the first integration PR. The semantic mapping is nearly one-to-one:

| EveryQuery Hydra key               | Package key                                 | Translation               |
| ---------------------------------- | ------------------------------------------- | ------------------------- |
| `num_queries`                      | `sampling.num_queries`                      | none                      |
| `num_contexts_per_query`           | `sampling.num_contexts_per_query`           | none                      |
| `min_prediction_times_per_subject` | `sampling.min_prediction_times_per_subject` | none                      |
| `query_codes`                      | `sampling.query_codes`                      | convert `ListConfig` only |
| `min_duration`                     | `sampling.min_duration`                     | float conversion          |
| `max_duration`                     | `sampling.max_duration`                     | float conversion          |
| `duration_distribution`            | `sampling.duration_distribution`            | none                      |
| `data_dir`                         | `data_dir`                                  | path conversion           |
| `out_dir`                          | `out_dir`                                   | path conversion           |
| `split`                            | `split`                                     | none                      |
| `seed`                             | `seed`                                      | none                      |
| `max_workers`                      | `sampling.max_workers`                      | none                      |
| `overwrite`                        | `overwrite`                                 | none                      |

Only the grouping differs, and only for the keys that describe the sampler rather than the run: an EQ adapter
copies values across without renaming or reinterpreting any of them.

### 8.2 Dense-grid configuration

`meds-generate-task-grid`, composed from `dense_grid.yaml` and the `grid` group:

```yaml
defaults:
  - grid: default # or `everyquery_eval`, which drops censored rows
  - _self_

data_dir: ???
out_dir: ??? # `${out_dir}_unique/` and `${out_dir}_summary/` are siblings of this root
split: held_out
input_shard: # null builds every shard of the split; name one to fan the work out across jobs
seed: 1
overwrite: false
log_dir: ${oc.select:out_dir,outputs}_logs

grid:
  prediction_times_per_subject: 1
  min_context_per_subject: 50
  query_codes: ${data_dir}
  durations: [30, 90, 180, 365, 731]
  subject_subsample_fraction:
  write_unique_prediction_times: true
  censored_rows: keep
```

The evaluation adapter is also nearly one-to-one. `input_shard` stays an explicit key because EQ parallelizes
evaluation through a Hydra shard sweep, unlike the training driver, which fans out labeling internally; leaving
it null asks a single process to build every shard in sorted order.

| EveryQuery Hydra key            | Package key                          | Translation               |
| ------------------------------- | ------------------------------------ | ------------------------- |
| `prediction_times_per_subject`  | `grid.prediction_times_per_subject`  | none                      |
| `min_context_per_subject`       | `grid.min_context_per_subject`       | none                      |
| `query_codes`                   | `grid.query_codes`                   | convert `ListConfig` only |
| `durations`                     | `grid.durations`                     | tuple conversion          |
| `subject_subsample_fraction`    | `grid.subject_subsample_fraction`    | none                      |
| `write_unique_prediction_times` | `grid.write_unique_prediction_times` | none                      |
| `data_dir`, `out_dir`, `split`  | `data_dir`, `out_dir`, `split`       | path conversion           |
| `input_shard`                   | `input_shard`                        | string conversion         |
| `seed`                          | `seed`                               | none                      |
| `overwrite`                     | `overwrite`                          | none                      |

Both public configurations reject a zero work budget. Training requires `num_queries > 0` and
`num_contexts_per_query > 0`; evaluation requires `prediction_times_per_subject > 0`, at least one resolved code,
and at least one duration. Lower-level pure helpers may return typed empty frames for empty data, but requesting
zero work through a top-level generator is a configuration error.

## 9. Canonical row schema

For zero-copy EveryQuery compatibility, version 0.1 uses the existing EQ task-query field names:

| Column            | Type                 | Meaning                                  |
| ----------------- | -------------------- | ---------------------------------------- |
| `subject_id`      | MEDS subject ID type | Patient identifier                       |
| `prediction_time` | `datetime[us]`       | Inclusive history cutoff                 |
| `query`           | UTF-8 string         | MEDS code queried                        |
| `duration_days`   | `float32`            | Continuous positive future horizon       |
| `boolean_value`   | nullable Boolean     | `true`, `false`, or `null` when censored |

The package owns `TaskQuerySchema`, including its nullable `boolean_value`, Arrow alignment contract, and
empty-frame helper. EveryQuery imports and temporarily re-exports that class so existing downstream imports can
migrate without a flag day. This moves the producer-consumer row contract together with the producer.

There is no `task_id` in random training output. Continuous durations make most sampled query specifications
unique, and training consumes rows rather than a stable benchmark task registry.

There is no separate `is_censored` column in the compatibility output: `boolean_value == null` is the EQ
contract. A future enriched output may expose censoring explicitly, but not by changing the default EQ-compatible
schema.

## 10. Exact sampling contract

### 10.1 Seed derivation

Derived seeds use EQ's cross-process-stable BLAKE2b algorithm and 31-bit mask. The two initial streams are:

```text
derive_seed(seed, "queries")
derive_seed(seed, "contexts")
```

Python's salted `hash()` must never influence sampling.

### 10.2 Query draw

For `Q = num_queries`:

1. Resolve and deterministically order the code universe.
2. Draw all `Q` uniform code indices first.
3. Draw all `Q` durations second from the configured distribution.
4. Zip them into `Q` `QuerySpec(code, duration_days)` values.

Log-uniform durations are sampled by drawing uniformly in log space and exponentiating. Durations remain
unrounded Python floats until the partitioned index write, where they align to `float32`.

### 10.3 Patient-context draw

Let `N = num_queries * num_contexts_per_query`.

1. Sort the eligible subject-count table by `subject_id`.
2. Draw all `N` subject row indices uniformly with replacement.
3. Gather each sampled subject's `n_prediction_times`.
4. Draw all `N` prediction-time indices with array bounds:

```text
low  = min_prediction_times_per_subject
high = n_prediction_times[sampled_subject]
```

The upper bound is exclusive, so the last prediction time is eligible. Subjects are eligible only when
`n_prediction_times > min_prediction_times_per_subject`.

The draw order is part of the compatibility contract. Changing vectorization or interleaving subject/time draws
may change every sampled row for the same seed.

### 10.4 Pairing

Each sampled query is repeated `num_contexts_per_query` consecutive times and zipped with the `N` contexts.
Duplicates are valid and must not be removed.

## 11. Generation pipelines

### 11.1 Random training pipeline

### Stage 0: prediction-time map

Scan `data/{split}/*.parquet`, reading temporal identity columns and the code required for death truncation.

For each subject:

1. truncate events strictly after the earliest `MEDS_DEATH` event;
2. deduplicate `(subject_id, time)`;
3. sort distinct times ascending;
4. assign a gapless zero-based `prediction_time_index`;
5. calculate `n_prediction_times`;
6. retain subjects with `n_prediction_times > minimum`.

A subject appearing in multiple shards is a hard error. Version 0.1 will preserve this EQ invariant rather than
silently repartitioning data.

Outputs:

```text
artifacts/{split}/_prediction_times/{shard}.parquet
artifacts/{split}/_prediction_time_counts.parquet
artifacts/{split}/_prediction_times_meta.json
```

### Stage 1: random queries

Sample `num_queries` `(query, duration_days)` specifications using the query RNG stream.

### Stage 2: random patient contexts

Sample `N` `(subject_id, shard, prediction_time_index)` rows using the context RNG stream.

### Stage 3: resolved partitioned index

Repeat queries, zip them with contexts, resolve `prediction_time_index -> prediction_time` against one shard map
at a time, and write:

```text
artifacts/{split}/_index/{shard}.parquet
```

The index columns are:

```text
subject_id, prediction_time, query, duration_days
```

Any unresolved context or join-key dtype mismatch is a hard error.

### Stage 4: shard-parallel labeling

Spawn workers rather than forking after Polars has initialized its thread pool. Each worker receives paths and a
shard identifier, reads its own data and index, labels the rows, validates the schema, and writes its final shard
atomically.

The union of final shards must contain exactly `N` rows.

### 11.2 Dense-grid pipeline

Evaluation reuses query-code resolution, the labeling kernel, `TaskQuerySchema`, schema alignment, and atomic
writes, but does not reuse the training sampler's Stage 0-3 orchestration.

For one `input_shard`:

1. Read and sort its events by `(subject_id, time)`.
2. Optionally retain subjects through EQ's deterministic hash-threshold subsampling rule.
3. Identify candidate event times where the subject has accumulated at least
    `min_context_per_subject` prior events, deduplicate `(subject_id, prediction_time)`, and sample up to
    `prediction_times_per_subject` candidates without replacement per subject. EQ derives this seed as
    `derive_seed(seed, "prediction_times", split, input_shard)`; subject subsampling independently uses
    `derive_seed(seed, "subject_subsample", split, input_shard)`.
4. Cross-join every sampled prediction time with the full resolved `query_codes x durations` grid.
5. Label the grid with the shared future-occurrence kernel and align it to `TaskQuerySchema`.
6. Apply the explicit `censored_rows` policy: retain null labels for general-purpose grids or remove them for
    EQ-compatible evaluation.
7. Atomically write `{output_dir}/{split}/{input_shard}.parquet` and, when requested, the deduplicated prediction
    times to the sibling `{output_dir.name}_unique/{split}/{input_shard}.parquet` root.

The expected pre-censor row count is:

```text
n_sampled_prediction_times * n_query_codes * n_durations
```

With `censored_rows="drop"`, the final count may be lower only because censored rows are removed. Dense-grid
generation preserves EQ's current integer-duration validation even though the shared canonical schema stores
`duration_days` as `float32`.

## 12. Label semantics

Let:

```text
window_end = prediction_time + duration_days
```

The outcome window is open on the left and closed on the right:

```text
(prediction_time, window_end]
```

An event exactly at `prediction_time` is history and cannot satisfy the query. An event exactly at `window_end`
does satisfy it.

Events strictly after the earliest `MEDS_DEATH` event are ignored. The death event itself remains available as a
query and as a prediction time.

Observation is complete when either:

```text
window_end <= max_observed_time
```

or:

```text
death_time <= window_end
```

Labels are resolved in this order:

1. If observation is incomplete, `boolean_value = null`.
2. Otherwise, if a matching event occurs in the window, `boolean_value = true`.
3. Otherwise, `boolean_value = false`.

Therefore censoring beats an occurrence observed in only the partial window. Death is a fully observed terminal
state, so a non-occurrence through death is false rather than censored.

These semantics intentionally match EQ commit `9bd85a1` and must be pinned with boundary tests.

## 13. Artifact and restart contract

### 13.1 Training artifacts

Final rows and intermediates use disjoint sibling roots:

```text
training_tasks/
`-- {split}/
    `-- {shard}.parquet

training_tasks_artifacts/
`-- {split}/
    |-- _prediction_time_counts.parquet
    |-- _prediction_times_meta.json
    |-- _summary.json
    |-- _prediction_times/
    |   `-- {shard}.parquet
    |-- _index/
    |   `-- {shard}.parquet
    `-- _labeled/
        `-- {shard}.json
```

All durable writes use a unique sibling temporary file followed by `os.replace`. A present final file is therefore
complete. Hidden temporary files are cleaned on retry.

Stage 4 may skip an existing final shard only when its sidecar fingerprint matches the logical content of the
current index partition. Stale output shards and sidecars not present in the current index are pruned.

Stage 0 cache reuse validates a cheap input manifest containing every relative event-shard path, byte size, and
modification time, plus the split, eligibility threshold, death-truncation version, sampler schema version, and
any authoritative MEDS dataset identifier available in metadata. Any mismatch invalidates and rebuilds the
cache. This detects normal file replacement without hashing every shard; it is not an adversarial content
integrity guarantee.

### 13.2 Dense-grid artifacts

```text
task_grid/{split}/{input_shard}.parquet
task_grid_unique/{split}/{input_shard}.parquet  # optional
task_grid_summary/{split}/{input_shard}.json
```

Evaluation has no training prediction-time cache or random-query index. Each independently invoked shard is
written atomically and may be skipped when its existing output is complete and `overwrite` is false, matching
EQ's shard-sweep workflow.

## 14. Determinism contract

For identical logical MEDS input, configuration, package version, and seed:

- query specifications are identical;
- patient contexts are identical;
- paired index rows are identical;
- labels are identical;
- worker completion order does not affect output content;
- restart does not change completed rows;
- Python process hash randomization does not affect output.

Compatibility with the pinned EQ commit additionally requires exact sampled values, not merely equivalent
distributions.

Byte-identical Parquet output is not required because writer metadata and row-group choices may differ. Logical
row equality after deterministic sorting is required.

## 15. EveryQuery adapters

EveryQuery's existing Hydra entry point remains stable:

```text
EQ_generate_training_tasks
```

Its `run(cfg: DictConfig)` becomes approximately:

```python
def run(cfg):
    package_config = RandomTaskSamplerConfig.from_values(
        num_queries=cfg.num_queries,
        num_contexts_per_query=cfg.num_contexts_per_query,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        query_codes=to_plain_query_code_source(cfg.query_codes),
        min_duration_days=cfg.min_duration,
        max_duration_days=cfg.max_duration,
        duration_distribution=cfg.duration_distribution,
        seed=cfg.seed,
        max_workers=cfg.max_workers,
    )
    sample_random_tasks(
        data_dir=cfg.data_dir,
        output_dir=cfg.out_dir,
        split=cfg.split,
        config=package_config,
        overwrite=cfg.overwrite,
    )
```

The first EQ PR should preserve:

- command name;
- Hydra keys and defaults;
- output and artifact paths;
- task-row schema;
- deterministic samples for frozen fixtures;
- logging information relied upon operationally;
- downstream dataset behavior.

Only after that PR is accepted should EQ-specific compatibility wrappers be deprecated.

`EQ_generate_evaluation_tasks` follows the same boundary: its Hydra wrapper converts OmegaConf containers to
ordinary Python values and calls `generate_task_grid` with `output_dir=Path(cfg.out_dir) / "eval"` and
`censored_rows="drop"`. The package's sibling-root convention then produces EQ's existing `eval_unique/` root.
Its command name, shard-sweep interface, defaults, and deterministic results remain unchanged.

## 16. Test migration and equivalence gate

The package test suite will be built from EQ's behavioral tests, organized by stage.

### Stage 0

- distinct-time indexing and gapless ranks;
- eligibility boundary (`n > minimum`);
- death truncation;
- subject-shard conflict rejection;
- stable sorted subject index;
- cache validation and invalidation.

### Stage 1

- exact deterministic samples for frozen seeds;
- independent code and duration draw order;
- uniform and log-uniform bounds and distribution shape;
- unrounded float durations;
- code-universe validation.

### Stage 2

- exact deterministic contexts for frozen seeds;
- sampling with replacement;
- subject draw before time draw;
- last-time reachability;
- join-key dtype preservation;
- invalid eligibility detection.

### Stage 3

- exact query repetition order;
- per-shard resolution;
- no dropped unresolved rows;
- dtype mismatch errors;
- stale index replacement.

### Stage 4

- strict lower and inclusive upper bounds;
- query-code isolation;
- three-valued labels;
- censoring precedence;
- earliest-death truncation;
- death as complete observation;
- unknown-subject rejection;
- atomic write and stale-temp cleanup;
- sidecar fingerprint skip/relabel behavior;
- worker-count invariance.

### Evaluation

- prior-event eligibility boundaries, including tied event times;
- up-to-`K` without-replacement prediction-time sampling per subject;
- determinism in `(seed, split, input_shard)`;
- deterministic subject-subsample boundaries;
- exact dense Cartesian grid ordering and row count;
- required non-empty code and duration axes and integer-duration validation;
- shared death/censor labeling semantics;
- removal of null labels;
- optional unique prediction-time output;
- atomic overwrite and idempotent skip behavior.

### End to end

For a frozen multi-shard MEDS fixture, compare both package workflows with their pinned EQ counterparts:

```text
EveryQuery@9bd85a1 sampler
meds-random-task-sampler
```

Compare sorted logical training rows and artifacts, then compare evaluation and unique-time rows for every
fixture shard. The EveryQuery integration PR is blocked until both equivalence tests pass.

## 17. Implementation sequence

1. Add core schema constants, dataclasses, seed derivation, and query-code resolution.
2. Implement and test Stage 1 query sampling with exact EQ fixtures.
3. Implement and test Stage 2 context sampling with exact EQ fixtures.
4. Implement the pure Stage 4 labeling kernel and its full death/censoring boundary suite.
5. Implement Stage 0 prediction-time maps and cache contract.
6. Implement Stage 3 pairing and per-shard index resolution.
7. Implement atomic I/O, Stage 4 workers, restartability, summaries, and validation.
8. Add the `meds-sample-random-tasks` command and the `sampling` config group.
9. Implement evaluation-time sampling, subject subsampling, and dense-grid construction.
10. Add evaluation orchestration, censor filtering, unique-time output, and the `meds-generate-task-grid`
    command with its `grid` config group.
11. Run exact training and evaluation equivalence against the pinned EQ checkout.
12. Tag a package release or pin an immutable package commit.
13. Open the thin-adapter EveryQuery PR.

Pure sampling and labeling functions come before orchestration so the compatibility contract is testable without
large file-system fixtures.

## 18. Design decisions and remaining choices

### 18.1 Resolved: Python 3.11+

The package supports Python 3.11 and newer so EveryQuery can adopt it without raising EQ's minimum Python
version. MHAL-template tooling must be tested under both 3.11 and 3.12 before release.

### 18.2 Resolved: move `TaskQuerySchema`

`TaskQuerySchema`, schema alignment, and the typed empty-frame helper move into this package. EveryQuery may
re-export them temporarily. This prevents the generic producer from depending on an EQ-owned consumer schema and
gives other models one authoritative row contract.

### 18.3 Resolved: package-owned code resolution

Current EQ accepts three shapes for `query_codes`:

1. an explicit Hydra/Python list;
2. a YAML or Parquet file;
3. a metadata-root directory, resolved as `{root}/metadata/codes.parquet`.

For explicit lists, EQ preserves user order while removing duplicates. For Parquet/metadata input, it takes the
unique codes and sorts them so the same RNG code index always maps to the same code. This ordering is part of
deterministic compatibility.

The package resolves all three forms. It owns YAML, Parquet, and MEDS-root resolution and deterministic ordering.
Both the EQ adapters and this package's own CLI convert `ListConfig` objects to ordinary lists and otherwise
pass paths through; the resolver itself never sees an OmegaConf container. Tests pin explicit-list order
preservation, file-based sorting, duplicate removal, empty input, non-string values, and missing `code` columns.

### 18.4 Resolved: cheap file-manifest cache validation

Current EQ's Stage 0 sidecar records the split, minimum prediction-time threshold, death-truncation version,
written shards, subject count, and prediction-time row count. This catches configuration/schema evolution but
does not prove that the underlying Parquet contents are unchanged. Replacing a shard at the same path can make a
cache look valid.

A version 0.1 cache records each relative shard path, byte size, and modification time, plus any authoritative
MEDS dataset identifier. Cache metadata also includes the sampler schema version and every Stage 0 semantic
input. A manifest mismatch rebuilds the cache. Copying or touching files may conservatively invalidate it, and a
same-path/same-size/same-mtime content mutation is outside this contract. This policy affects reuse only, not
sampled values from a fresh run.

### 18.5 Resolved: zero top-level budgets are errors

Current EQ is internally inconsistent:

- `QueryDistribution.sample(0)` returns an empty list;
- `sample_patient_contexts(..., n=0)` returns a typed empty frame;
- the full Stage 3 pipeline asserts that queries and contexts are non-empty.

Public configuration fails fast when a requested work axis has zero size. Training rejects zero
`num_queries` or `num_contexts_per_query`; evaluation rejects zero `prediction_times_per_subject`, no codes, or
no durations. This matches the effective EQ training behavior and avoids silently successful empty jobs. Pure
lower-level helpers may still support typed empty data inputs where that makes composition and tests cleaner.

### 18.6 Resolved: dense-grid generation is a version 0.1 package workflow

The package includes purpose-neutral dense-grid generation alongside random sampling. It uses fixed caller-
provided durations and per-subject prediction-time sampling, explicitly keeps or drops null labels, and optionally
emits the deduplicated `(subject_id, prediction_time)` trajectory index. EveryQuery selects `censored_rows="drop"`
through its adapter; other consumers may retain nullable labels with the package default.

### 18.7 Deferred: package publication mechanism

PyPI publication versus an immutable Git dependency is intentionally deferred. The package API and EQ
equivalence suite must exist before selecting a release mechanism.

## 19. Definition of done for the package extraction

The extraction is complete when:

- the package reproduces pinned EQ training samples on the equivalence fixture;
- the package reproduces pinned EQ evaluation and unique-time outputs on the equivalence fixture;
- its full sampler test suite passes independently of EveryQuery;
- EveryQuery's existing sampler, dataset, training, and CLI tests pass through the adapter;
- EveryQuery no longer contains a second implementation of generic sampling and labeling;
- the dependency is pinned to an immutable released version;
- documentation identifies the compatibility reference and any deliberate divergences;
- no model-framework dependency enters this package.

At that point, EveryQuery uses this package fully for random training-task generation and fixed-grid evaluation-
task generation while retaining ownership of model training, inference, metrics, and its Hydra-facing user
experience.
