"""Hydra command-line entry points for both sampling workflows.

This module is the only place in the package that imports Hydra or OmegaConf.  Everything it calls
takes ordinary Python values (DESIGN.md section 7): the CLI's whole job is to compose a config,
convert it to plain containers, and then call the same public API a Python caller would.

Two commands, one per workflow, each with its own root config in :func:`config_dir`:

- ``meds-sample-random-tasks`` (``random_sample.yaml``) ->
  :func:`~meds_random_task_sampler.random_sample.sample_random_tasks`
- ``meds-generate-task-grid`` (``dense_grid.yaml``) ->
  :func:`~meds_random_task_sampler.dense_grid.generate_task_grid` for one shard, or
  :func:`~meds_random_task_sampler.dense_grid.generate_task_grids` for every shard of a split.

Each workflow's config group (``sampling`` / ``grid``) holds exactly the fields of its dataclass, so
composition is a keyword splat and a missing or misspelled key is caught here rather than deep in a
stage.  ``seed`` is the one field kept at the top level, since both workflows share it.
"""

import logging
from dataclasses import MISSING, fields
from importlib.resources import files
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from meds_random_task_sampler.dense_grid import (
    TaskGridGeneratorConfig,
    generate_task_grid,
    generate_task_grids,
)
from meds_random_task_sampler.random_sample import (
    GenerationResult,
    RandomTaskSamplerConfig,
    sample_random_tasks,
)

logger = logging.getLogger(__name__)

CONFIG_DIRNAME = "configs"


def config_dir() -> Path:
    """Return the packaged Hydra config directory.

    Both entry points search it by absolute path, so the configs stay plain data files rather than an
    importable module.  Copy it — or point ``--config-dir`` at a directory of your own — to keep
    site-specific defaults outside the package while still composing against the shipped groups.

    Examples:
        >>> sorted(p.name for p in config_dir().glob("*.yaml"))
        ['dense_grid.yaml', 'random_sample.yaml']
        >>> sorted(p.name for p in config_dir().glob("*/*.yaml"))
        ['default.yaml', 'default.yaml', 'everyquery_eval.yaml', 'smoke.yaml']
    """
    return Path(str(files("meds_random_task_sampler").joinpath(CONFIG_DIRNAME)))


def workflow_kwargs(cfg: DictConfig, group: str, config_cls: type) -> dict[str, Any]:
    """Convert one config group plus the shared ``seed`` into keyword arguments for ``config_cls``.

    Interpolations are resolved and OmegaConf containers become plain ``list``/``dict`` objects, so
    nothing below this module ever sees a ``ListConfig`` (which
    :func:`~meds_random_task_sampler.random_sample.read_query_codes` would reject).

    Args:
        cfg: The composed root config.
        group: Name of the config group holding the workflow's settings.
        config_cls: The frozen dataclass the group mirrors.

    Returns:
        Keyword arguments for ``config_cls``.

    Raises:
        ValueError: If the group holds a key that is not a field of ``config_cls``, or omits one that
            has no default.  Either way the YAML has drifted from the dataclass, and failing here
            names the offending key instead of surfacing a bare ``TypeError`` from the constructor.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Demo:
        ...     n: int
        ...     seed: int = 1
        >>> cfg = OmegaConf.create({"seed": 7, "demo": {"n": "${seed}"}})
        >>> workflow_kwargs(cfg, "demo", Demo)
        {'n': 7, 'seed': 7}

        Drift in either direction is reported against the group it came from:

        >>> workflow_kwargs(OmegaConf.create({"seed": 7, "demo": {"n": 1, "extra": True}}), "demo", Demo)
        Traceback (most recent call last):
            ...
        ValueError: unknown demo key(s): extra. Demo accepts: n
        >>> workflow_kwargs(OmegaConf.create({"seed": 7, "demo": {}}), "demo", Demo)
        Traceback (most recent call last):
            ...
        ValueError: missing required demo key(s): n
    """
    kwargs = OmegaConf.to_object(cfg[group])
    accepted = {field.name for field in fields(config_cls)} - {"seed"}
    required = {
        field.name
        for field in fields(config_cls)
        if field.default is MISSING and field.default_factory is MISSING
    } - {"seed"}

    if unknown := sorted(set(kwargs) - accepted):
        raise ValueError(
            f"unknown {group} key(s): {', '.join(unknown)}. "
            f"{config_cls.__name__} accepts: {', '.join(sorted(accepted))}"
        )
    if missing := sorted(required - set(kwargs)):
        raise ValueError(f"missing required {group} key(s): {', '.join(missing)}")
    return {**kwargs, "seed": cfg.seed}


def required_roots(cfg: DictConfig) -> tuple[str, str]:
    """Return the mandatory ``(data_dir, out_dir)`` pair, naming both if either is unset.

    Reading them up front also keeps the error direct: ``sampling.query_codes`` defaults to
    ``${data_dir}``, so leaving ``data_dir`` unset would otherwise first surface as a failed
    interpolation inside an unrelated group.

    Examples:
        >>> required_roots(OmegaConf.create({"data_dir": "/meds", "out_dir": "/tasks"}))
        ('/meds', '/tasks')
        >>> required_roots(OmegaConf.create({"data_dir": "???", "out_dir": "???"}))
        Traceback (most recent call last):
            ...
        ValueError: data_dir and out_dir are required: pass data_dir=/path/to/MEDS
        out_dir=/path/to/output on the command line.
    """
    unset = [key for key in ("data_dir", "out_dir") if OmegaConf.is_missing(cfg, key)]
    if unset:
        raise ValueError(
            f"{' and '.join(unset)} {'is' if len(unset) == 1 else 'are'} required: pass "
            "data_dir=/path/to/MEDS out_dir=/path/to/output on the command line."
        )
    return cfg.data_dir, cfg.out_dir


def _log_result(result: GenerationResult, workflow: str) -> None:
    """Report where a finished run put its rows."""
    logger.info(
        "%s: wrote %s row(s) across %s shard(s) to %s (artifacts: %s).",
        workflow,
        f"{result.rows:,}",
        f"{result.shards:,}",
        result.output_dir,
        result.artifacts_dir,
    )


@hydra.main(version_base="1.3", config_path=str(config_dir()), config_name="random_sample")
def sample_random_tasks_main(cfg: DictConfig) -> None:
    """Run the random query-task sampler (``meds-sample-random-tasks``)."""
    data_dir, out_dir = required_roots(cfg)
    config = RandomTaskSamplerConfig(**workflow_kwargs(cfg, "sampling", RandomTaskSamplerConfig))
    result = sample_random_tasks(
        data_dir=data_dir,
        output_dir=out_dir,
        split=cfg.split,
        config=config,
        overwrite=cfg.overwrite,
    )
    _log_result(result, "random_sample")


@hydra.main(version_base="1.3", config_path=str(config_dir()), config_name="dense_grid")
def generate_task_grid_main(cfg: DictConfig) -> None:
    """Build dense task grids (``meds-generate-task-grid``).

    ``input_shard=null`` (the default) builds every shard of the split in sorted order; naming one
    builds that shard alone, which is what a per-shard job array or a ``--multirun`` sweep wants.
    """
    data_dir, out_dir = required_roots(cfg)
    config = TaskGridGeneratorConfig(**workflow_kwargs(cfg, "grid", TaskGridGeneratorConfig))
    shared = {"data_dir": data_dir, "output_dir": out_dir, "split": cfg.split, "config": config}
    if cfg.input_shard is None:
        result = generate_task_grids(**shared, overwrite=cfg.overwrite)
    else:
        result = generate_task_grid(**shared, input_shard=str(cfg.input_shard), overwrite=cfg.overwrite)
    _log_result(result, "dense_grid")
