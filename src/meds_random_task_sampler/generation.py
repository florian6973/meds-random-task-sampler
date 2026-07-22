"""End-to-end task-collection generation."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import yaml

if TYPE_CHECKING:
    from meds_random_task_sampler.manifest import CollectionConfig

LABEL_COLUMNS = [
    "subject_id",
    "prediction_time",
    "task_id",
    "boolean_value",
    "is_censored",
    "query_code",
    "horizon_days",
]


def _hash_fraction(*parts: object) -> float:
    value = "|".join(map(str, parts)).encode()
    integer = int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")
    return integer / 2**64


def _hash_rank_key(seed: int, split: str, subject_id: int, timestamp: datetime) -> bytes:
    value = f"{seed}|{split}|{subject_id}|{timestamp.isoformat()}".encode()
    return hashlib.blake2b(value, digest_size=16).digest()


def _dataset_files(data_dir: Path) -> list[Path]:
    files = sorted((data_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no MEDS data parquet files found under {data_dir / 'data'}")
    return files


def _fingerprint(data_dir: Path, data_files: list[Path]) -> str:
    digest = hashlib.sha256()
    candidates = [
        data_dir / "metadata" / "subject_splits.parquet",
        data_dir / "metadata" / "codes.parquet",
        data_dir / "metadata" / "dataset.json",
        *data_files,
    ]
    for path in candidates:
        if not path.is_file():
            continue
        digest.update(path.relative_to(data_dir).as_posix().encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _load_events(data_files: list[Path]) -> pl.DataFrame:
    return (
        pl.scan_parquet(data_files)
        .select("subject_id", "time", "code")
        .filter(pl.col("time").is_not_null() & pl.col("code").is_not_null())
        .collect()
        .with_columns(pl.col("subject_id").cast(pl.Int64), pl.col("code").cast(pl.String))
        .sort("subject_id", "time")
    )


def _sample_prediction_times(
    events: pl.DataFrame,
    splits: pl.DataFrame,
    cfg: CollectionConfig,
) -> pl.DataFrame:
    events = events.join(splits, on="subject_id", how="inner").filter(pl.col("split").is_in(cfg.splits))
    if cfg.subject_subsample_fraction < 1:
        subjects = (
            events.select("subject_id", "split")
            .unique()
            .filter(
                pl.struct("subject_id", "split").map_elements(
                    lambda row: _hash_fraction(cfg.seed, row["split"], row["subject_id"])
                    < cfg.subject_subsample_fraction,
                    return_dtype=pl.Boolean,
                )
            )
        )
        events = events.join(subjects, on=["subject_id", "split"], how="inner")

    candidates = (
        events.group_by("subject_id", "split", "time")
        .len(name="events_at_time")
        .sort("subject_id", "time")
        .with_columns(
            (pl.col("events_at_time").cum_sum().over("subject_id") - pl.col("events_at_time")).alias(
                "prior_events"
            )
        )
        .filter(pl.col("prior_events") >= cfg.minimum_prior_events)
    )
    if cfg.prediction_times_per_subject == 0 or candidates.is_empty():
        return pl.DataFrame(
            schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us"), "split": pl.String}
        )

    rows: list[dict[str, object]] = []
    for group in candidates.partition_by("subject_id", "split", as_dict=False):
        subject_id = int(group["subject_id"][0])
        split = str(group["split"][0])
        times = sorted(
            group["time"].to_list(),
            key=lambda timestamp: _hash_rank_key(cfg.seed, split, subject_id, timestamp),
        )[: cfg.prediction_times_per_subject]
        rows.extend(
            {"subject_id": subject_id, "prediction_time": timestamp, "split": split} for timestamp in times
        )
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))
        .sort("split", "subject_id", "prediction_time")
    )


def _label_split(
    events: pl.DataFrame,
    prediction_times: pl.DataFrame,
    cfg: CollectionConfig,
) -> pl.DataFrame:
    if prediction_times.is_empty():
        return pl.DataFrame(
            schema={
                "subject_id": pl.Int64,
                "prediction_time": pl.Datetime("us"),
                "task_id": pl.String,
                "boolean_value": pl.Boolean,
                "is_censored": pl.Boolean,
                "query_code": pl.String,
                "horizon_days": pl.Int32,
            }
        )
    split_subjects = prediction_times.select("subject_id").unique()
    split_events = events.join(split_subjects, on="subject_id", how="inner")
    last_times = split_events.group_by("subject_id").agg(pl.col("time").max().alias("record_end"))
    tasks = pl.DataFrame(
        {
            "task_id": [task.task_id for task in cfg.tasks],
            "query_code": [task.query_code for task in cfg.tasks],
            "horizon_days": [task.horizon_days for task in cfg.tasks],
        },
        schema={"task_id": pl.String, "query_code": pl.String, "horizon_days": pl.Int32},
    )
    index = (
        prediction_times.drop("split")
        .join(tasks, how="cross")
        .with_columns(
            (
                pl.col("prediction_time")
                + pl.duration(microseconds=pl.col("horizon_days").cast(pl.Int64) * 86_400_000_000)
            ).alias("horizon_end")
        )
        .join(last_times, on="subject_id", how="left")
    )
    future_events = split_events.select(
        "subject_id", pl.col("code").alias("query_code"), pl.col("time").alias("event_time")
    ).sort("event_time")
    index = index.sort("prediction_time")
    labeled = index.join_asof(
        future_events,
        left_on="prediction_time",
        right_on="event_time",
        by=["subject_id", "query_code"],
        strategy="forward",
        allow_exact_matches=False,
        check_sortedness=False,
    ).with_columns(
        (pl.col("event_time").is_not_null() & (pl.col("event_time") <= pl.col("horizon_end"))).alias(
            "_occurs"
        )
    )
    labeled = labeled.with_columns(
        (~pl.col("_occurs") & (pl.col("record_end") < pl.col("horizon_end"))).alias("is_censored")
    ).with_columns(
        pl.when(pl.col("_occurs"))
        .then(pl.lit(True))
        .when(~pl.col("is_censored"))
        .then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("boolean_value")
    )
    if cfg.censoring_policy == "drop":
        labeled = labeled.filter(~pl.col("is_censored"))
    return labeled.select(LABEL_COLUMNS).sort("task_id", "subject_id", "prediction_time")


def _package_version() -> str:
    try:
        return version("meds-random-task-sampler")
    except PackageNotFoundError:
        return "0+unknown"


def _write_summary(
    all_labels: pl.DataFrame,
    tasks: pl.DataFrame,
    splits: tuple[str, ...],
    output_dir: Path,
) -> None:
    base = pl.DataFrame({"split": list(splits)}).join(tasks.select("task_id"), how="cross")
    if all_labels.is_empty():
        summary = base.with_columns(
            *[
                pl.lit(0, dtype=pl.UInt32).alias(name)
                for name in (
                    "n_rows",
                    "n_subjects",
                    "n_prediction_times",
                    "n_observed",
                    "n_censored",
                    "n_positive",
                    "n_negative",
                )
            ],
            pl.lit(None, dtype=pl.Float64).alias("prevalence_observed"),
        )
    else:
        counts = (
            all_labels.group_by("split", "task_id")
            .agg(
                pl.len().alias("n_rows"),
                pl.col("subject_id").n_unique().alias("n_subjects"),
                pl.struct("subject_id", "prediction_time").n_unique().alias("n_prediction_times"),
                pl.col("boolean_value").is_not_null().sum().alias("n_observed"),
                pl.col("is_censored").sum().alias("n_censored"),
                (pl.col("boolean_value") == True).sum().alias("n_positive"),  # noqa: E712
                (pl.col("boolean_value") == False).sum().alias("n_negative"),  # noqa: E712
            )
            .with_columns(
                pl.when(pl.col("n_observed") > 0)
                .then(pl.col("n_positive") / pl.col("n_observed"))
                .otherwise(None)
                .alias("prevalence_observed")
            )
        )
        summary = base.join(counts, on=["split", "task_id"], how="left").with_columns(
            pl.exclude("split", "task_id", "prevalence_observed").fill_null(0)
        )
    summary.sort("split", "task_id").write_parquet(output_dir / "summary.parquet")
    payload = {
        "n_tasks": tasks.height,
        "n_rows": int(summary["n_rows"].sum()),
        "n_observed": int(summary["n_observed"].sum()),
        "n_censored": int(summary["n_censored"].sum()),
        "n_degenerate_tasks": int(summary.filter(pl.col("n_positive") == 0)["task_id"].n_unique()),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def generate_collection(
    cfg: CollectionConfig,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Generate a complete task collection.

    Args:
        cfg: Validated collection configuration.
        data_dir: MEDS dataset root.
        output_dir: Destination directory.
        overwrite: Replace an existing destination when true.

    Returns:
        The generated collection directory.
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    splits_path = data_dir / "metadata" / "subject_splits.parquet"
    if not splits_path.is_file():
        raise FileNotFoundError(f"missing MEDS subject splits: {splits_path}")
    data_files = _dataset_files(data_dir)
    events = _load_events(data_files)
    splits = pl.read_parquet(splits_path, columns=["subject_id", "split"]).with_columns(
        pl.col("subject_id").cast(pl.Int64), pl.col("split").cast(pl.String)
    )
    conflicting_subjects = (
        splits.group_by("subject_id")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
    )
    if conflicting_subjects.height:
        examples = conflicting_subjects["subject_id"].head(10).to_list()
        raise ValueError(f"MEDS metadata assigns subjects to multiple splits: {examples}")
    splits = splits.unique("subject_id", keep="first")
    available_splits = set(splits["split"].unique())
    missing = set(cfg.splits) - available_splits
    if missing:
        raise ValueError(f"requested splits are absent from MEDS metadata: {sorted(missing)}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    prediction_times = _sample_prediction_times(events, splits, cfg)
    if cfg.censoring_policy == "require_full_followup" and not prediction_times.is_empty():
        max_horizon = max(task.horizon_days for task in cfg.tasks)
        record_end = events.group_by("subject_id").agg(pl.col("time").max().alias("record_end"))
        prediction_times = (
            prediction_times.join(record_end, on="subject_id", how="left")
            .filter(pl.col("record_end") >= pl.col("prediction_time") + pl.duration(days=max_horizon))
            .drop("record_end")
        )

    tasks_df = pl.DataFrame(
        {
            "task_id": [task.task_id for task in cfg.tasks],
            "type": ["code_occurrence"] * len(cfg.tasks),
            "query_code": [task.query_code for task in cfg.tasks],
            "horizon_days": [task.horizon_days for task in cfg.tasks],
            "groups": [list(task.groups) for task in cfg.tasks],
        }
    )
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir()
    tasks_df.write_parquet(metadata_dir / "tasks.parquet")

    labeled_parts: list[pl.DataFrame] = []
    for split in cfg.splits:
        split_predictions = prediction_times.filter(pl.col("split") == split)
        pred_dir = output_dir / "prediction_times" / split
        label_dir = output_dir / "labels" / split
        pred_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        split_predictions.drop("split").write_parquet(pred_dir / "0.parquet")
        labels = _label_split(events, split_predictions, cfg)
        labels.write_parquet(label_dir / "0.parquet")
        labeled_parts.append(labels.with_columns(pl.lit(split).alias("split")))

    all_labels = pl.concat(labeled_parts, how="vertical") if labeled_parts else pl.DataFrame()
    _write_summary(all_labels, tasks_df, cfg.splits, output_dir)
    manifest = {
        "schema_version": 1,
        "generator": {"package": "meds-random-task-sampler", "version": _package_version()},
        "created_at": datetime.now(UTC).isoformat(),
        "collection": {"name": cfg.name, "description": cfg.description},
        "seed": cfg.seed,
        "dataset": {"path": str(data_dir), "fingerprint": _fingerprint(data_dir, data_files)},
        "source_splits": {
            "path": "metadata/subject_splits.parquet",
            "included": list(cfg.splits),
        },
        "prediction_times": {
            "count_per_subject": cfg.prediction_times_per_subject,
            "minimum_prior_events": cfg.minimum_prior_events,
            "sampling": "deterministic_hash_without_replacement",
            "subject_subsample_fraction": cfg.subject_subsample_fraction,
        },
        "labeling": {
            "interval": "(prediction_time, prediction_time + horizon_days]",
            "censoring_policy": cfg.censoring_policy,
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "type": "code_occurrence",
                "query_code": task.query_code,
                "horizon_days": task.horizon_days,
                "groups": list(task.groups),
            }
            for task in cfg.tasks
        ],
    }
    (output_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    generation = {
        "input_config": cfg.raw,
        "data_files": [str(path.relative_to(data_dir)) for path in data_files],
    }
    (metadata_dir / "generation.json").write_text(json.dumps(generation, indent=2, sort_keys=True) + "\n")
    return output_dir
