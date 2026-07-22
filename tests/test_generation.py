"""End-to-end tests for deterministic task collection generation."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import yaml

from meds_random_task_sampler import generate_collection, load_collection_config, validate_collection


def _write_dataset(root: Path) -> None:
    data_dir = root / "data"
    metadata_dir = root / "metadata"
    data_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    start = datetime(2020, 1, 1)
    rows = []
    splits = []
    for subject_id, split in ((1, "train"), (2, "tuning"), (3, "held_out")):
        splits.append({"subject_id": subject_id, "split": split})
        for day in range(8):
            code = "A" if day == 6 and subject_id != 2 else "HISTORY"
            rows.append({"subject_id": subject_id, "time": start + timedelta(days=day), "code": code})
    pl.DataFrame(rows).write_parquet(data_dir / "0.parquet")
    pl.DataFrame(splits).write_parquet(metadata_dir / "subject_splits.parquet")
    pl.DataFrame({"code": ["A", "B", "HISTORY"]}).write_parquet(metadata_dir / "codes.parquet")


def _write_config(path: Path) -> None:
    config = {
        "schema_version": 1,
        "metadata": {"name": "test"},
        "seed": 4,
        "subjects": {"splits": ["train", "tuning", "held_out"]},
        "prediction_times": {"count_per_subject": 1, "minimum_prior_events": 2},
        "tasks": {
            "type": "code_occurrence",
            "query_codes": {"source": "explicit", "values": ["A", "B"]},
            "horizons_days": [2, 10],
        },
        "labeling": {
            "window": {"start_inclusive": False, "end_inclusive": True},
            "censoring": {"policy": "preserve"},
        },
    }
    path.write_text(yaml.safe_dump(config))


def test_generate_collection_is_deterministic(tmp_path: Path) -> None:
    """Generation is reproducible and records censoring statistics."""
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    config_path = tmp_path / "collection.yaml"
    _write_config(config_path)
    config = load_collection_config(config_path)
    first = generate_collection(config, dataset, tmp_path / "first")
    second = generate_collection(config, dataset, tmp_path / "second")
    assert validate_collection(first) == {"n_tasks": 4, "n_rows": 12, "n_splits": 3}
    assert pl.read_parquet(first / "summary.parquet").equals(pl.read_parquet(second / "summary.parquet"))
    for split in config.splits:
        first_labels = pl.read_parquet(first / "labels" / split / "0.parquet")
        second_labels = pl.read_parquet(second / "labels" / split / "0.parquet")
        assert first_labels.equals(second_labels)
    summary = pl.read_parquet(first / "summary.parquet")
    assert summary["n_censored"].sum() > 0


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    """An existing output is protected unless overwrite is explicit."""
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    config_path = tmp_path / "collection.yaml"
    _write_config(config_path)
    config = load_collection_config(config_path)
    output = generate_collection(config, dataset, tmp_path / "collection")
    try:
        generate_collection(config, dataset, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("generation should refuse to overwrite an existing collection")


def test_conflicting_source_splits_are_rejected(tmp_path: Path) -> None:
    """A subject cannot leak across source splits."""
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    splits_path = dataset / "metadata" / "subject_splits.parquet"
    splits = pl.read_parquet(splits_path)
    pl.concat(
        [splits, pl.DataFrame({"subject_id": [1], "split": ["held_out"]})],
        how="vertical",
    ).write_parquet(splits_path)
    config_path = tmp_path / "collection.yaml"
    _write_config(config_path)

    try:
        generate_collection(load_collection_config(config_path), dataset, tmp_path / "collection")
    except ValueError as error:
        assert "multiple splits" in str(error)
    else:
        raise AssertionError("generation should reject subjects assigned to multiple splits")
