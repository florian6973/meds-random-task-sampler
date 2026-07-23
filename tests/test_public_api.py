"""Public configuration, resolution, and end-to-end generation tests."""

import json
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import polars as pl
import pytest

from meds_random_task_sampler import (
    RandomTaskSamplerConfig,
    TaskGridGeneratorConfig,
    TaskQuerySchema,
    discover_shards,
    generate_task_grid,
    generate_task_grids,
    read_query_codes,
    sample_random_tasks,
)


def _write_dataset(root: Path, split: str = "train") -> None:
    """Write a small, single-shard temporal MEDS dataset."""
    start = datetime(2020, 1, 1)
    rows = []
    for subject_id in (1, 2):
        for day in range(8):
            rows.append(
                {
                    "subject_id": subject_id,
                    "time": start + timedelta(days=day),
                    "code": "TARGET" if day in (4, 7) else "HISTORY",
                }
            )
    fp = root / "data" / split / "0.parquet"
    fp.parent.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(fp)


def _add_second_shard(root: Path, split: str) -> None:
    """Add a second shard with a disjoint subject."""
    source = pl.read_parquet(root / "data" / split / "0.parquet")
    source.with_columns(pl.lit(3).alias("subject_id")).write_parquet(root / "data" / split / "1.parquet")


def test_query_code_resolution_owns_all_sources(tmp_path: Path) -> None:
    """Lists preserve order; YAML and MEDS metadata roots resolve inside the package."""
    assert read_query_codes(["B", "A", "B"]) == ["B", "A"]

    yaml_fp = tmp_path / "codes.yaml"
    yaml_fp.write_text("codes: [B, A, B]\n")
    assert read_query_codes(yaml_fp) == ["B", "A"]

    metadata_fp = tmp_path / "cohort" / "metadata" / "codes.parquet"
    metadata_fp.parent.mkdir(parents=True)
    pl.DataFrame({"code": ["B", "A", "B"]}).write_parquet(metadata_fp)
    assert read_query_codes(metadata_fp.parents[1]) == ["A", "B"]


@pytest.mark.parametrize("field", ["num_queries", "num_contexts_per_query"])
def test_random_config_rejects_zero_budget(field: str) -> None:
    """Top-level random-sample work axes must be positive."""
    values = {"num_queries": 1, "num_contexts_per_query": 1}
    values[field] = 0
    with pytest.raises(ValueError, match=field):
        RandomTaskSamplerConfig(
            **values,
            min_prediction_times_per_subject=1,
            query_codes=["TARGET"],
        )


def test_grid_config_rejects_zero_budget() -> None:
    """Top-level grid work axes must be positive and non-empty."""
    with pytest.raises(ValueError, match="prediction_times_per_subject"):
        TaskGridGeneratorConfig(0, 1, ["TARGET"], [30])
    with pytest.raises(ValueError, match="durations"):
        TaskGridGeneratorConfig(1, 1, ["TARGET"], [])


def test_random_generation_end_to_end(tmp_path: Path) -> None:
    """The random API produces exactly the requested sampled row budget."""
    data_dir = tmp_path / "meds"
    _write_dataset(data_dir)
    config = RandomTaskSamplerConfig(
        num_queries=3,
        num_contexts_per_query=2,
        min_prediction_times_per_subject=1,
        query_codes=["TARGET"],
        min_duration=1,
        max_duration=2,
        duration_distribution="uniform",
        max_workers=1,
    )
    result = sample_random_tasks(data_dir, tmp_path / "tasks", "train", config)

    labels = pl.read_parquet(result.output_dir / "0.parquet")
    assert result.rows == labels.height == 6
    assert labels.columns == [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.boolean_value_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ]
    summary = json.loads((result.artifacts_dir / "_summary.json").read_text())
    assert summary["sampling_strategy"] == "random"
    assert summary["package_version"] == version("meds-random-task-sampler")
    assert summary["rows"] == 6
    assert summary["labels_null"] + summary["labels_false"] + summary["labels_true"] == 6


def test_dense_grid_generation_end_to_end(tmp_path: Path) -> None:
    """The grid API builds fixed tasks and can drop censored rows explicitly."""
    data_dir = tmp_path / "meds"
    _write_dataset(data_dir, "held_out")
    config = TaskGridGeneratorConfig(
        prediction_times_per_subject=2,
        min_context_per_subject=1,
        query_codes=["TARGET", "HISTORY"],
        durations=[1, 2],
        censored_rows="drop",
    )
    result = generate_task_grid(data_dir, tmp_path / "tasks", "held_out", "0", config)

    labels = pl.read_parquet(result.output_dir / "0.parquet")
    unique = pl.read_parquet(tmp_path / "tasks_unique" / "held_out" / "0.parquet")
    assert labels.height == result.rows
    assert labels[TaskQuerySchema.boolean_value_name].null_count() == 0
    assert labels.select("query", "duration_days").unique().height == 4
    assert unique.height <= 4
    summary = json.loads((result.artifacts_dir / "0.json").read_text())
    assert summary["sampling_strategy"] == "dense_grid"
    assert summary["package_version"] == version("meds-random-task-sampler")
    assert summary["rows"] == labels.height
    assert summary["labels_null"] == 0


def test_dense_grid_can_keep_censored_rows(tmp_path: Path) -> None:
    """Keeping nullable labels is the purpose-neutral grid default."""
    data_dir = tmp_path / "meds"
    _write_dataset(data_dir, "held_out")
    config = TaskGridGeneratorConfig(
        prediction_times_per_subject=2,
        min_context_per_subject=1,
        query_codes=["NEVER"],
        durations=[365],
        write_unique_prediction_times=False,
    )
    result = generate_task_grid(data_dir, tmp_path / "grid", "held_out", "0", config)
    labels = pl.read_parquet(result.output_dir / "0.parquet")
    assert labels.height == 4
    assert labels[TaskQuerySchema.boolean_value_name].null_count() == 4


def test_dense_grid_discovers_and_generates_every_shard(tmp_path: Path) -> None:
    """Dataset-level generation matches an exhaustive sorted shard sweep."""
    data_dir = tmp_path / "meds"
    _write_dataset(data_dir, "held_out")
    _add_second_shard(data_dir, "held_out")
    config = TaskGridGeneratorConfig(2, 1, ["TARGET"], [1], censored_rows="drop")

    assert discover_shards(data_dir, "held_out") == ["0", "1"]
    result = generate_task_grids(data_dir, tmp_path / "grid", "held_out", config)

    outputs = sorted(path.stem for path in result.output_dir.glob("*.parquet"))
    assert outputs == ["0", "1"]
    assert result.shards == 2
    assert result.rows == sum(
        pl.read_parquet(result.output_dir / f"{shard}.parquet").height for shard in outputs
    )


def test_dense_grid_discovery_rejects_empty_split(tmp_path: Path) -> None:
    """Dataset-level generation cannot silently succeed without input shards."""
    split_dir = tmp_path / "meds" / "data" / "held_out"
    split_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No shards found"):
        generate_task_grids(
            tmp_path / "meds",
            tmp_path / "grid",
            "held_out",
            TaskGridGeneratorConfig(1, 1, ["TARGET"], [1]),
        )


def test_dense_grid_discovery_warns_about_orphan_outputs(tmp_path: Path, caplog) -> None:
    """Outputs with no discovered input shard are retained but reported."""
    data_dir = tmp_path / "meds"
    _write_dataset(data_dir, "held_out")
    orphan = tmp_path / "grid" / "held_out" / "orphan.parquet"
    orphan.parent.mkdir(parents=True)
    pl.DataFrame({"sentinel": [1]}).write_parquet(orphan)

    generate_task_grids(
        data_dir,
        tmp_path / "grid",
        "held_out",
        TaskGridGeneratorConfig(1, 1, ["TARGET"], [1]),
    )

    assert orphan.exists()
    assert "match no discovered shard" in caplog.text
