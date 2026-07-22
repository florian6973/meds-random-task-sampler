"""Configuration parsing and stable task identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskSpec:
    """A concrete code-occurrence task."""

    task_id: str
    query_code: str
    horizon_days: int
    groups: tuple[str, ...] = ()

    @property
    def canonical_payload(self) -> dict[str, Any]:
        """Return the task definition that determines semantic identity."""
        return {
            "type": "code_occurrence",
            "query_code": self.query_code,
            "horizon_days": self.horizon_days,
            "window": {"start_inclusive": False, "end_inclusive": True},
        }


@dataclass(frozen=True)
class CollectionConfig:
    """Validated user-authored collection configuration."""

    raw: dict[str, Any]
    name: str
    description: str
    seed: int
    splits: tuple[str, ...]
    subject_subsample_fraction: float
    prediction_times_per_subject: int
    minimum_prior_events: int
    tasks: tuple[TaskSpec, ...]
    censoring_policy: str


def _task_id(query_code: str, horizon_days: int) -> str:
    payload = {
        "type": "code_occurrence",
        "query_code": query_code,
        "horizon_days": horizon_days,
        "window": {"start_inclusive": False, "end_inclusive": True},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:10]
    readable = "".join(char.lower() if char.isalnum() else "-" for char in query_code).strip("-")
    readable = "-".join(filter(None, readable.split("-")))[:48] or "code"
    return f"{readable}--{horizon_days}d--{digest}"


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def load_collection_config(path: str | Path) -> CollectionConfig:
    """Load and validate a collection YAML file.

    Args:
        path: YAML configuration path.

    Returns:
        Resolved configuration with one :class:`TaskSpec` per code/horizon pair.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    raw = _require_mapping(raw, "configuration")
    if raw.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    metadata = _require_mapping(raw.get("metadata", {}), "metadata")
    subjects = _require_mapping(raw.get("subjects", {}), "subjects")
    prediction_times = _require_mapping(raw.get("prediction_times", {}), "prediction_times")
    tasks_cfg = _require_mapping(raw.get("tasks", {}), "tasks")
    if tasks_cfg.get("type") != "code_occurrence":
        raise ValueError("tasks.type must be 'code_occurrence'")
    codes_cfg = _require_mapping(tasks_cfg.get("query_codes", {}), "tasks.query_codes")
    if codes_cfg.get("source") != "explicit":
        raise ValueError("the first release supports only tasks.query_codes.source=explicit")
    codes = codes_cfg.get("values")
    horizons = tasks_cfg.get("horizons_days")
    if not isinstance(codes, list) or not codes or not all(isinstance(code, str) and code for code in codes):
        raise ValueError("tasks.query_codes.values must be a non-empty list of strings")
    if not isinstance(horizons, list) or not horizons:
        raise ValueError("tasks.horizons_days must be a non-empty list")
    valid_horizons = all(
        isinstance(horizon, int) and not isinstance(horizon, bool) and horizon > 0 for horizon in horizons
    )
    if not valid_horizons:
        raise ValueError("every horizon must be a positive integer number of days")

    unique_codes = list(dict.fromkeys(codes))
    unique_horizons = list(dict.fromkeys(horizons))
    groups_cfg = _require_mapping(tasks_cfg.get("groups", {}), "tasks.groups")
    groups_by_code: dict[str, list[str]] = {code: [] for code in unique_codes}
    for group_name, group_value in groups_cfg.items():
        group = _require_mapping(group_value, f"tasks.groups.{group_name}")
        group_codes = group.get("query_codes", [])
        if not isinstance(group_codes, list) or not all(code in groups_by_code for code in group_codes):
            raise ValueError(f"tasks.groups.{group_name}.query_codes contains an unknown code")
        for code in group_codes:
            groups_by_code[code].append(str(group_name))

    task_specs = tuple(
        TaskSpec(
            task_id=_task_id(code, horizon),
            query_code=code,
            horizon_days=horizon,
            groups=tuple(groups_by_code[code]),
        )
        for code in unique_codes
        for horizon in unique_horizons
    )
    splits = subjects.get("splits", ["train", "tuning", "held_out"])
    if not isinstance(splits, list) or not splits or not all(isinstance(split, str) for split in splits):
        raise ValueError("subjects.splits must be a non-empty list of strings")
    fraction = float(subjects.get("subsample_fraction", 1.0))
    if not 0 < fraction <= 1:
        raise ValueError("subjects.subsample_fraction must be in (0, 1]")
    count = prediction_times.get("count_per_subject", 1)
    minimum = prediction_times.get("minimum_prior_events", 50)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("prediction_times.count_per_subject must be a non-negative integer")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
        raise ValueError("prediction_times.minimum_prior_events must be a non-negative integer")
    labeling = _require_mapping(raw.get("labeling", {}), "labeling")
    window = _require_mapping(labeling.get("window", {}), "labeling.window")
    if window.get("start_inclusive", False) is not False or window.get("end_inclusive", True) is not True:
        raise ValueError("the first release supports only the interval (prediction_time, horizon_end]")
    censoring = _require_mapping(labeling.get("censoring", {}), "labeling.censoring")
    policy = censoring.get("policy", "preserve")
    if policy not in {"preserve", "drop", "require_full_followup"}:
        raise ValueError("censoring policy must be preserve, drop, or require_full_followup")

    return CollectionConfig(
        raw=raw,
        name=str(metadata.get("name", path.stem)),
        description=str(metadata.get("description", "")),
        seed=int(raw.get("seed", 1)),
        splits=tuple(dict.fromkeys(splits)),
        subject_subsample_fraction=fraction,
        prediction_times_per_subject=count,
        minimum_prior_events=minimum,
        tasks=task_specs,
        censoring_policy=policy,
    )
