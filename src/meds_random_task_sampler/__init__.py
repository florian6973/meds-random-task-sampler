"""Model-independent EveryQuery-compatible MEDS task generation."""

import os

# Polars initializes its global thread pool at import time. Set the default before
# importing any package submodule, since dense_grid and random_sample import Polars.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from meds_random_task_sampler.aces_export import (
    AcesExportResult,
    aces_task_config,
    build_task_manifest,
    export_task_rows_to_aces,
    task_id,
    write_aces_task_configs,
)
from meds_random_task_sampler.dense_grid import (
    TaskGridGeneratorConfig,
    build_task_grid,
    discover_shards,
    generate_task_grid,
    generate_task_grids,
    sample_prediction_times_per_subject,
    subsample_subject_ids,
)
from meds_random_task_sampler.random_sample import (
    GenerationResult,
    QueryDistribution,
    QuerySpec,
    RandomTaskSamplerConfig,
    evaluate_index_df,
    read_query_codes,
    sample_patient_contexts,
    sample_random_tasks,
)
from meds_random_task_sampler.schema import TaskQuerySchema, empty_task_query_df, window_length_expr
from meds_random_task_sampler.seeds import derive_seed

__all__ = [
    "AcesExportResult",
    "GenerationResult",
    "QueryDistribution",
    "QuerySpec",
    "RandomTaskSamplerConfig",
    "TaskGridGeneratorConfig",
    "TaskQuerySchema",
    "aces_task_config",
    "build_task_grid",
    "build_task_manifest",
    "derive_seed",
    "discover_shards",
    "empty_task_query_df",
    "evaluate_index_df",
    "export_task_rows_to_aces",
    "generate_task_grid",
    "generate_task_grids",
    "read_query_codes",
    "sample_patient_contexts",
    "sample_prediction_times_per_subject",
    "sample_random_tasks",
    "subsample_subject_ids",
    "task_id",
    "window_length_expr",
    "write_aces_task_configs",
]
