"""Model-independent EveryQuery-compatible MEDS task generation."""

import os

# Polars initializes its global thread pool at import time. Set the default before
# importing any package submodule, since dense_grid and random_sample import Polars.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

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
from meds_random_task_sampler.schema import TaskQuerySchema, empty_task_query_df
from meds_random_task_sampler.seeds import derive_seed

__all__ = [
    "GenerationResult",
    "QueryDistribution",
    "QuerySpec",
    "RandomTaskSamplerConfig",
    "TaskGridGeneratorConfig",
    "TaskQuerySchema",
    "build_task_grid",
    "derive_seed",
    "discover_shards",
    "empty_task_query_df",
    "evaluate_index_df",
    "generate_task_grid",
    "generate_task_grids",
    "read_query_codes",
    "sample_patient_contexts",
    "sample_prediction_times_per_subject",
    "sample_random_tasks",
    "subsample_subject_ids",
]
