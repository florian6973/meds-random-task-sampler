"""Validate generated random-task collection artifacts."""

from pathlib import Path

import polars as pl
import yaml

REQUIRED_LABEL_COLUMNS = {
    "subject_id",
    "prediction_time",
    "task_id",
    "boolean_value",
    "is_censored",
    "query_code",
    "horizon_days",
}


def validate_collection(collection_dir: str | Path) -> dict[str, int]:
    """Validate required files, schemas, task references, and summary counts."""
    collection_dir = Path(collection_dir)
    manifest_path = collection_dir / "manifest.yaml"
    summary_path = collection_dir / "summary.parquet"
    tasks_path = collection_dir / "metadata" / "tasks.parquet"
    for path in (manifest_path, summary_path, tasks_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing collection artifact: {path}")
    manifest = yaml.safe_load(manifest_path.read_text())
    splits = manifest["source_splits"]["included"]
    tasks = pl.read_parquet(tasks_path)
    task_ids = set(tasks["task_id"])
    total_rows = 0
    for split in splits:
        files = sorted((collection_dir / "labels" / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no label files found for split {split}")
        labels = pl.read_parquet(files)
        missing_columns = REQUIRED_LABEL_COLUMNS - set(labels.columns)
        if missing_columns:
            raise ValueError(f"labels for {split} lack columns: {sorted(missing_columns)}")
        unknown_tasks = set(labels["task_id"].unique()) - task_ids
        if unknown_tasks:
            raise ValueError(f"labels for {split} reference unknown tasks: {sorted(unknown_tasks)}")
        invalid_censor = labels.filter(pl.col("is_censored") & pl.col("boolean_value").is_not_null())
        if invalid_censor.height:
            raise ValueError(f"labels for {split} contain censored rows with non-null labels")
        total_rows += labels.height
    summary = pl.read_parquet(summary_path)
    if int(summary["n_rows"].sum()) != total_rows:
        raise ValueError("summary row count does not match label files")
    return {"n_tasks": tasks.height, "n_rows": total_rows, "n_splits": len(splits)}
