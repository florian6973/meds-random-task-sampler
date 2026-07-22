# MEDS Task Generation: Design

**Status:** Initial design draft  
**Package:** `meds-task-generation`  
**Python module:** `meds_task_generation`

## 1. Summary

`meds-task-generation` generates reproducible collections of prediction tasks and labels from a
MEDS dataset. Its initial task language is the single-code, fixed-horizon question:

> Given a subject's history through prediction time `t`, does code `c` occur in the interval
> `(t, t + h]`?

The package generalizes the evaluation-task generation logic currently used by EveryQuery without
depending on EveryQuery or any other model. MEDS-DEV can use the generated collection as a benchmark
artifact, while model packages can consume it for zero-shot prediction, probing, fine-tuning, or
joint multitask learning.

The package owns task definition, prediction-time sampling, labeling, censoring, manifests, and
schema validation. It does not own model preprocessing, model training, prediction, metric
calculation, or result packaging.

## 2. Repository foundation

The repository will be created from
[`McDermottHealthAI/MHAL-template`](https://github.com/McDermottHealthAI/MHAL-template), rather than
assembled as an ad hoc Python project. In particular, it will retain the template's conventions:

- `uv` for dependency and environment management;
- a `src/meds_task_generation/` package layout;
- `pytest` and doctests, with doctests treated as first-class tests;
- Ruff formatting and linting using the Google Python style;
- pre-commit hooks, secret scanning, and the existing GitHub Actions structure;
- `setuptools-scm` versioning from Git tags;
- `AGENTS.md` and `CONTRIBUTORS.md` as contributor guidance;
- no clinical or synthetic dataset artifacts committed to the repository.

The initial implementation should customize, not remove, those template files. The package name,
repository URLs, authors, description, and Ruff first-party import configuration must all be updated
from the template placeholders.

## 3. Motivation

MEDS-DEV currently treats a benchmark task as a single binary prediction problem. This works well
for task-specific models, but it does not natively represent one model trained or evaluated over a
collection of tasks.

Several model families benefit from a task-collection abstraction:

- query-conditioned models such as EveryQuery;
- multi-outcome survival models such as MOTOR;
- frozen encoders evaluated with many linear probes;
- jointly trained multitask models;
- tabular in-context models such as TabPFN;
- autoregressive models whose generated futures can be resolved against many outcomes.

The benchmark task collection must be independent of all of these models so that they are evaluated
against identical task definitions, patients, prediction times, labels, and censoring decisions.

## 4. Goals

The first release should:

1. Generate a fixed collection of `(query_code, horizon_days)` tasks.
2. Sample prediction times deterministically from MEDS event data.
3. Generate labels on the `train`, `tuning`, and `held_out` patient splits.
4. Represent censoring explicitly and consistently.
5. Produce a self-describing, versioned manifest.
6. Support explicit in-distribution and held-out-query-code partitions.
7. Produce deterministic output independent of worker count and shard scheduling.
8. Expose a CLI for benchmark orchestration and a Python API reusable by model packages.
9. Remain independent of EveryQuery, MEDS-DEV, PyTorch, and any particular model framework.
10. Make a singleton task collection a valid special case.

## 5. Non-goals

The initial package will not:

- preprocess MEDS data into model-specific tensors;
- train or load models;
- implement EveryQuery's random pretraining-query sampler as a MEDS-DEV benchmark stage;
- emit model predictions;
- calculate AUROC, AUPRC, calibration, or survival metrics;
- package or publish MEDS-DEV results;
- express arbitrary ACES logic, composite outcomes, readmission, or numeric-value predicates;
- define a general clinical task language in the first release;
- silently approximate an unsupported task.

Reusable low-level sampling and labeling functions may later be called by a model's internal
training recipe. Such use does not make model-specific pretraining artifacts part of the MEDS-DEV
benchmark contract.

## 6. Ownership boundaries

```text
MEDS dataset
    |
    v
meds-task-generation
    |-- task collection manifest
    |-- prediction-time index
    `-- labels for train/tuning/held_out
              |
              v
model package (EveryQuery, MOTOR, probe, TabPFN, ...)
    |-- model-specific preprocessing
    |-- optional pretraining/adaptation
    `-- predictions keyed by task_id
              |
              v
meds-evaluation
    |-- per-task metrics
    `-- collection aggregates
              |
              v
MEDS-DEV orchestration and result packaging
```

### 6.1 MEDS-DEV-owned behavior

MEDS-DEV selects the registered task-collection configuration, invokes task generation, passes the
result to a model recipe, invokes evaluation, and packages results.

### 6.2 Model-owned behavior

A model owns all internal training decisions. For example, an EveryQuery model may use
`meds-task-generation` Python primitives to generate millions of scattered pretraining queries, but
that operation occurs inside the model's training command and is not a direct MEDS-DEV task stage.

### 6.3 Evaluation-owned behavior

Metric computation belongs in `meds-evaluation`. This package may validate prediction keys against a
task manifest, but it will not score predictions.

## 7. Two distinct label products

It is important not to conflate benchmark labels with model-specific pretraining samples.

### 7.1 Fixed benchmark collection

The benchmark collection contains a stable set of task definitions across the canonical MEDS patient
splits. It does **not** assign subjects to new splits. The authoritative assignment remains
`{data_dir}/metadata/subject_splits.parquet`; task generation joins labels to that mapping and may partition
outputs by its existing `split` values for convenience:

```text
collection/
|-- manifest.yaml
`-- labels/
    |-- train/
    |   `-- *.parquet
    |-- tuning/
    |   `-- *.parquet
    `-- held_out/
        `-- *.parquet
```

The `train` labels allow probes and supervised models to adapt to the benchmark tasks. The `tuning`
and `held_out` labels support model selection and final evaluation. A zero-shot model may ignore
the training labels.

This differs from current ACES/MEDS-DEV storage, where one ACES task is extracted across all data shards and
its MEDS label rows are keyed by `subject_id` and `prediction_time`; the subject's split is inferred from the
MEDS metadata. A collection may physically partition its larger output by split, but that is a projection of
the existing mapping, not a second split definition.

### 7.2 Model-specific training samples

Model-specific samples may use different task distributions, horizons, prediction-time policies,
and row shapes. They are not benchmark task definitions and do not determine benchmark identity.

## 8. Authored configuration, resolved manifest, and task identity

`collection.yaml` and `manifest.yaml` serve different purposes:

- `collection.yaml` is the human-authored, portable generation request. It may use compact Cartesian-product
  declarations and references to code-list files or MEDS metadata.
- `manifest.yaml` is the immutable, fully resolved output. It lists every concrete task and records the
  dataset fingerprint, resolved codes, derived task IDs, generation parameters, and observed provenance.

An illustrative authored `collection.yaml` is:

```yaml
schema_version: 1

metadata:
  name: mimic-demo-code-occurrence-v1
  description: Fixed code-occurrence collection for integration testing.

seed: 1

subjects:
  splits: [train, tuning, held_out]
  subsample_fraction: 1.0

prediction_times:
  strategy: random_event_time
  count_per_subject: 1
  minimum_prior_events: 50

tasks:
  type: code_occurrence
  query_codes:
    source: explicit
    values:
      - ICD10CM//I10
      - ICD10CM//E11
  horizons_days: [7, 30, 365]
  groups:
    in_distribution:
      query_codes: [ICD10CM//I10]
    held_out_query:
      query_codes: [ICD10CM//E11]

labeling:
  window:
    start_inclusive: false
    end_inclusive: true
  censoring:
    policy: preserve

output:
  partition_by: [split]
```

The `groups` field attaches benchmark-protocol metadata to tasks; it does not itself train a model or remove
codes from patient histories. A model recipe is responsible for honoring a held-out-query protocol.

Alternative code sources may include an external YAML list or a MEDS `metadata/codes.parquet` file, but all
such sources are expanded into concrete values in `manifest.yaml`. The first release should support either a
Cartesian-product declaration or an explicit `tasks: [...]` list, not arbitrary templating.

Every task must have a stable `task_id`. It must be derived from a canonical task definition, not
from list position or row order.

An initial canonical task payload is:

```yaml
type: code_occurrence
query_code: ICD10CM//I10
horizon_days: 30
window:
  start_inclusive: false
  end_inclusive: true
```

`task_id` should combine a human-readable slug with a digest of canonical serialized content. The
digest prevents collisions when slugs are truncated or normalized.

An illustrative resolved `manifest.yaml` is:

```yaml
schema_version: 1
generator:
  package: meds-task-generation
  version: 0.1.0
seed: 1

dataset:
  fingerprint: "..."

prediction_times:
  count_per_subject: 1
  minimum_prior_events: 50
  sampling: without_replacement

tasks:
  - task_id: icd10cm-i10--30d--a1b2c3d4
    type: code_occurrence
    query_code: ICD10CM//I10
    horizon_days: 30
    query_partition: in_distribution

  - task_id: icd10cm-e11--30d--d4c3b2a1
    type: code_occurrence
    query_code: ICD10CM//E11
    horizon_days: 30
    query_partition: held_out

source_splits:
  path: metadata/subject_splits.parquet
  included: [train, tuning, held_out]
  fingerprint: "..."
```

The manifest written to disk must contain fully resolved code lists and task definitions. It must
not rely on external YAML aliases, unresolved Hydra interpolation, or mutable metadata files.

## 9. Query-code partitions

Patient splits and query-code partitions are independent axes. Patient splits are read from MEDS and never
resampled by this package:

| Axis | Training | Validation | Test |
|---|---|---|---|
| Patients | `train` | `tuning` | `held_out` |
| ID query codes | available | available | evaluated |
| Held-out query codes | excluded from model query training | optionally monitored | evaluated |

Holding out a code means excluding it from a model's query-target training distribution. It does
not remove the code from patient histories or from the MEDS vocabulary.

The package must validate explicit policies, including:

- `allow_overlap`: no disjointness requirement;
- `held_out_from_training`: held-out evaluation codes must not appear in the training-query list;
- `fully_disjoint`: all named query partitions must be pairwise disjoint.

The resolved manifest must record both the policy and the observed intersections. A violated policy
must fail before label generation.

## 10. Prediction-time sampling

The initial sampler will select up to `K` distinct prediction times per subject. A candidate time
must satisfy a declared minimum-history condition.

Required properties:

- deterministic in the global seed, dataset split, subject ID, and candidate timestamp;
- sampling without replacement within a subject;
- invariant to input shard order, worker count, and parallel scheduling;
- stable when unrelated subjects are added or removed, where practical;
- explicit behavior when a subject has fewer than `K` eligible times;
- a separate deterministic subject-subsampling policy for demo and low-cost runs.

Hash-ranking candidate times per subject is preferred over a stateful global random-number stream
because it is easier to make invariant to execution order.

## 11. Label semantics and censoring

For prediction time `t`, query code `c`, and horizon `h`, the initial task asks whether `c` occurs in
the interval `(t, t + h]`.

The label has three logical states:

- `true`: the event occurs within the window;
- `false`: the full window is observed and the event does not occur;
- censored: follow-up ends before the window closes and no qualifying event was observed first.

Unlike a bounded ACES task whose cohort definition can exclude cases without an observable target window,
arbitrary `(code, horizon)` tasks encounter right censoring whenever a subject's observable record ends
before `t + h`. Treating those rows as negatives would be incorrect. Avoiding censoring by requiring complete
follow-up through the largest horizon is possible, but changes the cohort, leaks future record length into
eligibility, and can remove many subjects.

The generator should therefore detect censoring even if the selected output policy later drops censored rows.
The supported policies should be:

- `preserve` (default): keep censored rows with a nullable label;
- `drop`: detect and count censored rows, then omit them from model-facing labels;
- `require_full_followup`: define eligibility using complete follow-up through the collection's maximum
  horizon; intended only for explicitly chosen sensitivity analyses.

A proposed schema for the default `preserve` policy is:

```text
subject_id: int64
prediction_time: timestamp[us]
task_id: string
boolean_value: bool, nullable
is_censored: bool
query_code: string
horizon_days: float32
```

`boolean_value = null` and `is_censored = true` represent censoring. The explicit flag makes filtering and
validation less error-prone, while the nullable value is a compact three-state representation. Because the
current MEDS label schema and ACES output are ordinary observed labels, the collection extension must be
reconciled with those schemas before implementation. If compatibility requires a strict MEDS label export,
that export can contain only observed rows while the canonical collection table retains censoring.

Event occurrence before loss to follow-up takes precedence over censoring. Death and other terminal
events require a declared dataset/task policy rather than an implicit special case.

## 12. Output layout

The proposed output layout is:

```text
output_dir/
|-- manifest.yaml
|-- summary.parquet
|-- summary.json
|-- metadata/
|   |-- tasks.parquet
|   `-- generation.json
|-- prediction_times/
|   |-- train/*.parquet
|   |-- tuning/*.parquet
|   `-- held_out/*.parquet
`-- labels/
    |-- train/*.parquet
    |-- tuning/*.parquet
    `-- held_out/*.parquet
```

The prediction-time index is stored independently because several model families can perform
expensive task-agnostic inference once per `(subject_id, prediction_time)` and resolve many tasks
afterward.

Final output directories must not contain transient artifacts. Temporary and cached intermediates
belong in a separate sibling directory or a caller-provided cache directory.

### 12.1 Summary statistics

`summary.parquet` is the canonical machine-readable summary, with one row per `(split, task_id)` and at least:

```text
split
task_id
n_rows
n_subjects
n_prediction_times
n_observed
n_censored
n_positive
n_negative
prevalence_observed
```

It should also contain aggregate rows or be accompanied by `summary.json` containing overall counts,
generation duration, warnings, and the number of degenerate tasks. Statistics are descriptive properties of
generated labels, not model-evaluation metrics. They must be computed from final outputs and validated against
them, rather than maintained as independent counters that can drift.

## 13. Public interfaces

### 13.1 CLI

The first CLI should be a single end-to-end command:

```bash
meds-task-generation generate \
  data_dir="$MEDS_DATASET" \
  config=collection.yaml \
  output_dir="$TASK_COLLECTION"
```

Useful inspection commands may follow:

```bash
meds-task-generation validate collection_dir="$TASK_COLLECTION"
meds-task-generation summarize collection_dir="$TASK_COLLECTION"
```

The initial CLI should avoid exposing pipeline-internal stages as separate user-facing commands.
Internal stages can remain testable Python functions.

### 13.2 Python API

The public API should center on typed, model-independent objects:

```python
manifest = TaskCollectionManifest.from_yaml("collection.yaml")
resolved = resolve_collection(manifest, dataset_dir)
generate_collection(resolved, dataset_dir, output_dir)
validate_collection(output_dir)
```

Lower-level functions for code resolution, prediction-time sampling, grid construction, and
labeling should be public only when they have stable schemas and deterministic contracts.

## 14. Proposed source layout

Following the MHAL template:

```text
meds-task-generation/
|-- .github/
|-- AGENTS.md
|-- CONTRIBUTORS.md
|-- DESIGN.md
|-- README.md
|-- pyproject.toml
|-- src/
|   `-- meds_task_generation/
|       |-- __init__.py
|       |-- __main__.py
|       |-- cli.py
|       |-- schemas.py
|       |-- manifest.py
|       |-- codes.py
|       |-- prediction_times.py
|       |-- labeling.py
|       |-- censoring.py
|       |-- generation.py
|       `-- validation.py
`-- tests/
```

## 15. MEDS-DEV integration

The first MEDS-DEV integration should treat a task collection as a generated, cacheable benchmark
artifact:

```text
demo_dataset
    -> task_collection
        -> model run
            -> collection predictions
                -> evaluation
```

Existing single-task models can consume a collection by iterating over its tasks. Collection-aware
models can consume the entire manifest and label dataset in one run.

A singleton collection should be exportable to the current MEDS-DEV single-task layout, allowing
backward-compatibility tests without immediately changing all model integrations.

## 16. Model integration examples

### 16.1 EveryQuery

- Uses the fixed task collection for evaluation.
- May reuse labeling primitives inside its own pretraining command.
- Owns random pretraining-query generation, tensorization, training, and query-conditioned prediction.

### 16.2 MOTOR

- Can query compatible native survival heads for `(code, horizon)` tasks.
- Can use collection training labels for frozen probes or fine-tuning.
- Requires a future survival-label extension for faithful time-to-event adaptation.

### 16.3 TabPFN

- Reuses dataset-level tabularization across tasks.
- Builds label-conditioned context or fitted state per task.
- Can process the collection in one orchestrated run even though adaptation remains task-specific.

### 16.4 Autoregressive models

- Use `prediction_times/` for one task-agnostic generation pass.
- Resolve all collection tasks against the same generated futures.

## 17. Reproducibility and provenance

The generated artifact must record:

- package and schema versions;
- normalized configuration;
- global and derived seeds;
- dataset fingerprint and relevant metadata fingerprint;
- resolved code universe and code partitions;
- exact task definitions and IDs;
- patient-time sampling parameters;
- label window boundary semantics;
- censoring policy;
- input and output row counts by split and task;
- creation timestamp and, when available, source revision.

Changing any field that can change task identity or labels must change the collection fingerprint.

## 18. Testing strategy

The initial test suite should include:

1. Doctests for public schemas and small pure functions.
2. Unit tests for task canonicalization and stable IDs.
3. Property tests for deterministic prediction-time sampling.
4. Property tests for interval-boundary and censoring semantics.
5. Tests that partition policies reject forbidden code overlap.
6. Tests that worker count and shard order do not change output.
7. Tests that a singleton collection exports to the standard MEDS label shape.
8. An end-to-end synthetic MEDS test with known labels.
9. A MIMIC-IV demo integration test, marked slow and requiring no committed data.
10. Differential tests against EveryQuery's evaluation-task generator during migration.

The differential tests may be removed after the behavior is independently specified and the
migration is complete; the package must not permanently define correctness as “whatever EveryQuery
currently does.”

## 19. Initial milestones

### Milestone 0: repository bootstrap

- Instantiate the MHAL template.
- Rename the package and update project metadata.
- Add this design document and an initial README.
- Confirm template tests, formatting, and pre-commit checks pass unchanged.

### Milestone 1: schemas and manifest

- Define `TaskSpec`, `TaskCollectionManifest`, and label schemas.
- Define canonical serialization, stable task IDs, and collection fingerprints.
- Implement code-list resolution and overlap policies.

### Milestone 2: deterministic task generation

- Implement prediction-time sampling.
- Implement dense task-grid construction.
- Implement event-window labeling and explicit censoring.
- Write partitioned outputs and provenance metadata.

### Milestone 3: validation and demo

- Add collection validation and summary commands.
- Add synthetic end-to-end tests.
- Run a small fixed collection on the MIMIC-IV demo.

### Milestone 4: ecosystem adapters

- Add an EveryQuery evaluation adapter.
- Add singleton export for current MEDS-DEV.
- Prototype collection-aware MEDS-DEV orchestration.
- Coordinate collection prediction and aggregation support with `meds-evaluation`.

## 20. Open questions

1. Can the canonical collection schema extend the MEDS label schema with nullable `boolean_value` and
   `is_censored`, or should it store canonical collection labels separately and export observed-only MEDS
   labels for compatibility?
2. What is the canonical dataset fingerprint for a sharded MEDS dataset?
3. Should prediction times be shared across all tasks, or may a task request its own eligibility
   criteria?
4. Should fixed task collections list tasks explicitly, or may a manifest contain a resolved
   Cartesian-product declaration?
5. Which code-normalization rules, if any, are safe across MEDS datasets?
6. How should terminal events such as death affect censoring for unrelated outcomes?
7. Should task IDs include dataset-specific code mappings, or should those live in a separate
   realization identifier?
8. What collection-prediction schema should `meds-evaluation` adopt?
9. How should aggregate metrics treat tasks with no positives, no negatives, or only censored rows?
10. Which behaviors should be identical to EveryQuery for migration compatibility, and which should
    deliberately change before the first stable release?

## 21. Initial decisions

The following decisions are considered part of the initial project direction:

- The repository and package are named `meds-task-generation` and `meds_task_generation`.
- The repository is bootstrapped from MHAL-template.
- The package is model-independent.
- MEDS-DEV directly invokes fixed benchmark collection generation, not model-specific pretraining
  sampling.
- Benchmark task definitions are stable across patient splits.
- Existing patient split assignments are read from MEDS `metadata/subject_splits.parquet`; the package never
  creates an alternative train/tuning/held-out assignment.
- Patient splits and query-code partitions are separate concepts.
- The first task language is single-code occurrence within a fixed future horizon.
- Prediction times are stored separately from the dense task-label table.
- Censored examples are preserved at generation time.
- The generator writes canonical per-task/per-split summary statistics.
- Metric computation remains outside this package.
