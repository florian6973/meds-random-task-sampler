"""Future-only ACES export: determinism, manifest/row correspondence, and label equivalence.

The equivalence test is the contract that matters: scoring each subject's *real* future with the
exported config via ``MEDS_trajectory_evaluation`` (the labeler zero-shot models use) must reproduce
the sampler's own label on every uncensored row, including events exactly on the window boundaries.
It is skipped when that optional package is not installed.
"""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

from meds_random_task_sampler import (
    RandomTaskSamplerConfig,
    TaskQuerySchema,
    aces_task_config,
    build_task_manifest,
    evaluate_index_df,
    export_task_rows_to_aces,
    sample_random_tasks,
    task_id,
    write_aces_task_configs,
)
from meds_random_task_sampler.cli import generate_task_grid_main, sample_random_tasks_main
from meds_random_task_sampler.schema import window_length_expr

US = timedelta(microseconds=1)


def _write_dataset(root: Path, split: str = "train") -> Path:
    start = datetime(2020, 1, 1)
    rows = [
        {"subject_id": s, "time": start + timedelta(days=day, hours=s), "code": code}
        for s in (1, 2, 3)
        for day in range(40)
        for code in (["TARGET"] if day % 5 == s else ["HISTORY"])
    ]
    fp = root / "data" / split / "0.parquet"
    fp.parent.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(fp)
    metadata_fp = root / "metadata" / "codes.parquet"
    metadata_fp.parent.mkdir(parents=True)
    pl.DataFrame({"code": ["TARGET", "HISTORY"]}).write_parquet(metadata_fp)
    return root


def _sample(tmp_path: Path, seed: int = 3) -> Path:
    data_dir = _write_dataset(tmp_path / "meds")
    out = tmp_path / "tasks"
    sample_random_tasks(
        data_dir=data_dir,
        output_dir=out,
        split="train",
        config=RandomTaskSamplerConfig(
            num_queries=12,
            num_contexts_per_query=3,
            min_prediction_times_per_subject=2,
            query_codes=["TARGET", "HISTORY"],
            min_duration=0.5,
            max_duration=20.0,
            seed=seed,
            max_workers=1,
        ),
    )
    return out


def test_every_sampled_row_maps_to_exactly_one_exported_config(tmp_path):
    out = _sample(tmp_path)
    res = export_task_rows_to_aces(out, "train", tmp_path / "aces")

    rows = pl.read_parquet(out / "train" / "*.parquet")
    manifest = pl.read_parquet(res.manifest_fp)
    joined = rows.join(manifest, on=["query", "duration_days"], how="left")

    assert joined["task_id"].null_count() == 0
    assert manifest.height == rows.select("query", "duration_days").unique().height == res.n_tasks
    for criteria_fp in manifest["criteria_fp"]:
        assert (res.export_dir / criteria_fp).is_file()

    pts = pl.read_parquet(res.prediction_times_fp)
    assert pts.height == rows.select("subject_id", "prediction_time").unique().height


def test_export_is_reproducible_byte_for_byte(tmp_path):
    out = _sample(tmp_path)
    a = export_task_rows_to_aces(out, "train", tmp_path / "a")
    b = export_task_rows_to_aces(out, "train", tmp_path / "b")
    for fp in sorted((a.export_dir / "criteria").iterdir()):
        assert fp.read_bytes() == (b.export_dir / "criteria" / fp.name).read_bytes()
    assert pl.read_parquet(a.manifest_fp).equals(pl.read_parquet(b.manifest_fp))


def test_window_in_config_matches_the_labeling_kernel(tmp_path):
    out = _sample(tmp_path)
    rows = pl.read_parquet(out / "train" / "*.parquet")
    kernel_windows = rows.select("query", "duration_days", window_length_expr().alias("w")).unique()
    manifest = build_task_manifest(rows).join(kernel_windows, on=["query", "duration_days"])
    assert (manifest["window"] == manifest["w"]).all()
    # Sampled durations are fractional, so this exercises the sub-second (float32) path.
    assert (manifest["window"].dt.total_microseconds() % 1_000_000 != 0).any()


def test_task_ids_are_stable_across_float_widths():
    assert task_id("A", 1.2345678) == task_id("A", pl.Series([1.2345678], dtype=pl.Float32)[0])
    assert build_task_manifest([("A", 1.2345678)])["task_id"][0] == task_id("A", 1.2345678)


def test_empty_export_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no tasks"):
        write_aces_task_configs([], tmp_path)
    with pytest.raises(FileNotFoundError):
        export_task_rows_to_aces(tmp_path / "missing", "train", tmp_path / "aces")


def test_random_sample_cli_exports_when_aces_dir_is_set(tmp_path):
    data_dir = _write_dataset(tmp_path / "meds")
    from hydra import compose, initialize_config_dir

    from meds_random_task_sampler.cli import config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir())):
        cfg = compose(
            config_name="random_sample",
            overrides=[
                f"data_dir={data_dir}",
                f"out_dir={tmp_path / 'out'}",
                f"aces_dir={tmp_path / 'zs'}",
                "sampling=smoke",
            ],
        )
    sample_random_tasks_main.__wrapped__(cfg)
    assert (tmp_path / "zs" / "train" / "tasks.parquet").is_file()
    assert (tmp_path / "zs" / "train" / "prediction_times.parquet").is_file()


def test_dense_grid_cli_exports_the_full_grid_axes(tmp_path):
    data_dir = _write_dataset(tmp_path / "meds", split="held_out")
    from hydra import compose, initialize_config_dir

    from meds_random_task_sampler.cli import config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir())):
        cfg = compose(
            config_name="dense_grid",
            overrides=[
                f"data_dir={data_dir}",
                f"out_dir={tmp_path / 'out'}",
                f"aces_dir={tmp_path / 'zs'}",
                "grid.min_context_per_subject=2",
                "grid.durations=[7,30]",
                "grid.censored_rows=drop",
            ],
        )
    generate_task_grid_main.__wrapped__(cfg)
    manifest = pl.read_parquet(tmp_path / "zs" / "held_out" / "tasks.parquet")
    # 2 codes x 2 durations, even though `drop` may have removed every censored 30-day row.
    assert manifest.height == 4
    assert set(manifest["duration_days"].to_list()) == {7.0, 30.0}


def test_no_export_by_default(tmp_path):
    data_dir = _write_dataset(tmp_path / "meds")
    from hydra import compose, initialize_config_dir

    from meds_random_task_sampler.cli import config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir())):
        cfg = compose(
            config_name="random_sample",
            overrides=[f"data_dir={data_dir}", f"out_dir={tmp_path / 'out'}", "sampling=smoke"],
        )
    assert cfg.aces_dir is None
    sample_random_tasks_main.__wrapped__(cfg)
    assert not any(p.name == "criteria" for p in tmp_path.rglob("*"))


# ---------------------------------------------------------------------------
# Label equivalence with the zero-shot labeler
# ---------------------------------------------------------------------------


def _zero_shot_label(trajectories: pl.DataFrame, criteria_fp: Path) -> pl.DataFrame:
    pytest.importorskip("MEDS_trajectory_evaluation")
    import tempfile as _tempfile
    from functools import partial
    from unittest import mock

    from MEDS_trajectory_evaluation.ACES_config_evaluation.label import label_trajectories
    from MEDS_trajectory_evaluation.ACES_config_evaluation.task_config import resolve_zero_shot_task_cfg
    from omegaconf import OmegaConf

    zs_cfg = resolve_zero_shot_task_cfg(
        OmegaConf.create({"criteria_fp": str(criteria_fp), "predicates_fp": None}),
        OmegaConf.create(
            {
                "remove_all_criteria": False,
                "collapse_temporal_gap_windows": False,
                "remove_post_label_windows": False,
            }
        ),
    )
    # MEDS_trajectory_evaluation reopens a still-open NamedTemporaryFile, which Windows forbids.
    with mock.patch.object(
        _tempfile, "NamedTemporaryFile", partial(_tempfile.NamedTemporaryFile, delete=False)
    ):
        return label_trajectories(trajectories, zs_cfg)


def _future_as_trajectory(events: pl.DataFrame, contexts: pl.DataFrame) -> pl.DataFrame:
    """Present each context's real future (events strictly after prediction_time) as a trajectory."""
    return (
        contexts.select("subject_id", "prediction_time")
        .unique()
        .join(events, on="subject_id")
        .filter(pl.col("time") > pl.col("prediction_time"))
        .select(
            "subject_id",
            "time",
            "code",
            pl.lit(None, dtype=pl.Float32).alias("numeric_value"),
            "prediction_time",
        )
        .sort("subject_id", "prediction_time", "time")
    )


@pytest.mark.parametrize("duration_days", [30.0, 1.2345678])
def test_zero_shot_labeler_reproduces_sampler_labels_on_real_futures(tmp_path, duration_days):
    pytest.importorskip("MEDS_trajectory_evaluation")
    t = datetime(2020, 1, 1)
    window = pl.select(window_length_expr(pl.lit(duration_days, dtype=pl.Float32))).item()
    # One subject per boundary case; each record runs well past the window so nothing is censored.
    offsets = {
        1: timedelta(0),  # at prediction time: history, not future
        2: US,  # first representable future instant: inside
        3: window,  # inclusive right edge: inside
        4: window + US,  # just past the edge: outside
        5: window / 2,  # interior
    }
    events = pl.DataFrame(
        [
            row
            for s, off in offsets.items()
            for row in (
                {"subject_id": s, "time": t - timedelta(days=1), "code": "HISTORY"},
                {"subject_id": s, "time": t, "code": "HISTORY"},
                {"subject_id": s, "time": t + off, "code": "TARGET"},
                {"subject_id": s, "time": t + window + timedelta(days=5), "code": "HISTORY"},
            )
        ]
    )
    contexts = pl.DataFrame(
        {
            "subject_id": list(offsets),
            "prediction_time": [t] * len(offsets),
            "query": ["TARGET"] * len(offsets),
            "duration_days": [duration_days] * len(offsets),
        },
        schema_overrides={"duration_days": pl.Float32},
    )

    truth = evaluate_index_df(contexts, events)
    assert truth["boolean_value"].to_list() == [False, True, True, False, True]

    res = write_aces_task_configs(contexts, tmp_path / "aces")
    criteria_fp = res.export_dir / pl.read_parquet(res.manifest_fp)["criteria_fp"][0]
    zs = _zero_shot_label(_future_as_trajectory(events, contexts), criteria_fp)

    merged = truth.join(zs, on=["subject_id", "prediction_time"], how="left").sort("subject_id")
    assert merged["valid"].all()
    assert merged["label"].to_list() == merged["boolean_value"].to_list()


def test_zero_shot_labeler_reproduces_sampled_task_labels(tmp_path):
    """End to end over a real random sample: every uncensored row agrees with ZSACES on its config."""
    pytest.importorskip("MEDS_trajectory_evaluation")
    out = _sample(tmp_path)
    res = export_task_rows_to_aces(out, "train", tmp_path / "aces")
    events = pl.read_parquet(tmp_path / "meds" / "data" / "train" / "*.parquet")
    rows = pl.read_parquet(out / "train" / "*.parquet").filter(
        pl.col(TaskQuerySchema.boolean_value_name).is_not_null()
    )
    manifest = pl.read_parquet(res.manifest_fp)
    checked = 0
    for task in manifest.iter_rows(named=True):
        task_rows = rows.filter(
            (pl.col("query") == task["query"]) & (pl.col("duration_days") == task["duration_days"])
        )
        if task_rows.height == 0:
            continue
        cfg = yaml.safe_load((res.export_dir / task["criteria_fp"]).read_text())
        assert cfg == aces_task_config(task["query"], int(task["window"].total_seconds() * 1e6 + 0.5))
        zs = _zero_shot_label(_future_as_trajectory(events, task_rows), res.export_dir / task["criteria_fp"])
        merged = task_rows.join(zs, on=["subject_id", "prediction_time"], how="left")
        assert merged["label"].to_list() == merged["boolean_value"].to_list(), task
        checked += task_rows.height
    assert checked > 0
