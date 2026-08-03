"""Hydra entry-point tests: config composition, group presets, and both console scripts."""

import subprocess
import sys
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from meds_random_task_sampler import RandomTaskSamplerConfig, TaskGridGeneratorConfig
from meds_random_task_sampler.cli import (
    config_dir,
    generate_task_grid_main,
    required_roots,
    sample_random_tasks_main,
    workflow_kwargs,
)

WORKFLOWS = [
    ("random_sample", "sampling", RandomTaskSamplerConfig),
    ("dense_grid", "grid", TaskGridGeneratorConfig),
]


def _write_dataset(root: Path, split: str) -> Path:
    """Write a two-shard MEDS dataset plus the code metadata `query_codes` defaults to."""
    start = datetime(2020, 1, 1)
    rows = [
        {
            "subject_id": subject_id,
            "time": start + timedelta(days=day),
            "code": "TARGET" if day % 4 == 0 else "HISTORY",
        }
        for subject_id in (1, 2, 3)
        for day in range(12)
    ]
    for shard, subjects in enumerate([(1, 2), (3,)]):
        fp = root / "data" / split / f"{shard}.parquet"
        fp.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([row for row in rows if row["subject_id"] in subjects]).write_parquet(fp)

    metadata_fp = root / "metadata" / "codes.parquet"
    metadata_fp.parent.mkdir(parents=True)
    pl.DataFrame({"code": ["TARGET", "HISTORY"]}).write_parquet(metadata_fp)
    return root


def _compose(config_name: str, overrides: list[str]) -> DictConfig:
    """Compose a shipped root config exactly as the console script would."""
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir())):
        return compose(config_name=config_name, overrides=overrides)


def _roots(tmp_path: Path, split: str) -> list[str]:
    """Build the two mandatory path overrides over a fresh dataset."""
    data_dir = _write_dataset(tmp_path / "meds", split)
    return [f"data_dir={data_dir}", f"out_dir={tmp_path / 'out'}"]


@pytest.mark.parametrize(("config_name", "group", "config_cls"), WORKFLOWS)
def test_shipped_group_covers_every_dataclass_field(config_name, group, config_cls, tmp_path):
    cfg = _compose(config_name, [f"data_dir={tmp_path}", f"out_dir={tmp_path / 'out'}"])
    kwargs = workflow_kwargs(cfg, group, config_cls)

    # Every field is set from YAML rather than left to an implicit dataclass default, so the config
    # files are the whole interface: what `--help` prints is what the run uses.
    assert set(kwargs) == {field.name for field in fields(config_cls)}
    config_cls(**kwargs)


@pytest.mark.parametrize(
    ("config_name", "group", "option", "config_cls"),
    [
        ("random_sample", "sampling", "default", RandomTaskSamplerConfig),
        ("random_sample", "sampling", "smoke", RandomTaskSamplerConfig),
        ("dense_grid", "grid", "default", TaskGridGeneratorConfig),
        ("dense_grid", "grid", "everyquery_eval", TaskGridGeneratorConfig),
    ],
)
def test_every_group_option_builds_a_valid_config(config_name, group, option, config_cls, tmp_path):
    cfg = _compose(config_name, [f"{group}={option}", f"data_dir={tmp_path}", f"out_dir={tmp_path}/o"])
    config_cls(**workflow_kwargs(cfg, group, config_cls))


def test_group_presets_differ_from_their_defaults(tmp_path):
    roots = [f"data_dir={tmp_path}", f"out_dir={tmp_path}/o"]
    default_sampling = _compose("random_sample", roots).sampling
    smoke_sampling = _compose("random_sample", [*roots, "sampling=smoke"]).sampling
    assert smoke_sampling.num_queries < default_sampling.num_queries
    # Inherited from `sampling/default.yaml` rather than restated by the preset.
    assert smoke_sampling.duration_distribution == default_sampling.duration_distribution == "log-uniform"

    assert _compose("dense_grid", roots).grid.censored_rows == "keep"
    assert _compose("dense_grid", [*roots, "grid=everyquery_eval"]).grid.censored_rows == "drop"


def test_query_codes_default_to_the_dataset_vocabulary(tmp_path):
    cfg = _compose("random_sample", _roots(tmp_path, "train"))
    assert workflow_kwargs(cfg, "sampling", RandomTaskSamplerConfig)["query_codes"] == cfg.data_dir


def test_list_overrides_reach_the_core_as_plain_lists(tmp_path):
    overrides = ["grid.durations=[7,30]", "grid.query_codes=[ICD//A,ICD//B]"]
    cfg = _compose("dense_grid", [f"data_dir={tmp_path}", f"out_dir={tmp_path}/o", *overrides])
    kwargs = workflow_kwargs(cfg, "grid", TaskGridGeneratorConfig)

    # The core rejects OmegaConf containers (DESIGN.md section 7), so the CLI must hand it builtins.
    assert isinstance(kwargs["durations"], list)
    assert kwargs["durations"] == [7, 30]
    assert isinstance(kwargs["query_codes"], list)
    assert kwargs["query_codes"] == ["ICD//A", "ICD//B"]


def test_unset_roots_are_reported_together():
    with pytest.raises(ValueError, match="data_dir and out_dir are required"):
        required_roots(_compose("random_sample", []))


def test_random_sample_main_writes_the_requested_row_budget(tmp_path):
    overrides = [*_roots(tmp_path, "train"), "sampling=smoke", "sampling.num_contexts_per_query=2"]
    # Hydra's decorator runs the task function directly when handed a config, skipping argv parsing.
    sample_random_tasks_main(_compose("random_sample", overrides))

    shards = sorted((tmp_path / "out" / "train").glob("*.parquet"))
    assert sum(pl.read_parquet(shard).height for shard in shards) == 16


def test_dense_grid_main_builds_every_shard_by_default(tmp_path):
    overrides = [
        *_roots(tmp_path, "held_out"),
        "grid.min_context_per_subject=1",
        "grid.durations=[2,30]",
    ]
    generate_task_grid_main(_compose("dense_grid", overrides))

    assert sorted(path.stem for path in (tmp_path / "out" / "held_out").glob("*.parquet")) == ["0", "1"]
    assert (tmp_path / "out_unique" / "held_out" / "0.parquet").exists()


def test_dense_grid_main_builds_one_named_shard(tmp_path):
    overrides = [
        *_roots(tmp_path, "held_out"),
        "grid.min_context_per_subject=1",
        "grid.durations=[2]",
        "input_shard=1",  # an int on the command line; shards are named by string
    ]
    generate_task_grid_main(_compose("dense_grid", overrides))

    assert sorted(path.stem for path in (tmp_path / "out" / "held_out").glob("*.parquet")) == ["1"]


def test_console_script_runs_a_full_pipeline(tmp_path):
    """The installed entry point composes its packaged configs from a real argv."""
    script = Path(sys.executable).parent / "meds-sample-random-tasks"
    assert script.exists(), f"console script missing from {script.parent}; run `uv sync`"

    data_dir = _write_dataset(tmp_path / "meds", "train")
    result = subprocess.run(
        [str(script), f"data_dir={data_dir}", f"out_dir={tmp_path / 'tasks'}", "sampling=smoke"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert sum(pl.read_parquet(p).height for p in (tmp_path / "tasks" / "train").glob("*.parquet")) == 8
    # Hydra's own outputs stay out of both data roots, which hold nothing but parquet.
    assert (tmp_path / "tasks_logs" / "random_sample" / ".hydra" / "config.yaml").exists()
    assert not any(path.is_dir() for path in (tmp_path / "tasks").iterdir() if path.name != "train")
