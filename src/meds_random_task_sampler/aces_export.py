"""Export sampled ``(query, duration_days)`` tasks as future-only ACES task configs.

Zero-shot (trajectory-generating) models are not trained on task labels; they generate a future
trajectory from each ``(subject_id, prediction_time)`` and then *label that trajectory* with the task
definition, typically via ``ZSACES_label`` from
`MEDS_trajectory_evaluation <https://github.com/mmcdermott/MEDS_trajectory_evaluation>`_.  That
labeler takes one ACES config per task.  This module writes one such config per distinct sampled
task, plus a manifest that maps each config back to the ``(query, duration_days)`` columns of the
sampler's task rows.

**Randomness stays in the sampler.**  The export is a pure, deterministic function of the sampled
rows (random workflow) or of the grid axes (dense workflow): it draws nothing.  The sampler's rows
are the ground truth — prediction times, task identity, and labels — and the configs only tell a
zero-shot labeler how to score a *generated* future against the same task.  Nobody has to run ACES
over real data to reproduce a cohort, so the prediction times need not coincide with any particular
trigger predicate: the sampler's prediction times are already event timestamps, and the zero-shot
labeler anchors each config on the trajectory's own ``prediction_time`` column.

**Window shape.**  The sampler's task asks whether ``query`` occurs in
``(prediction_time, prediction_time + duration]``.  ``ZSACES_label`` requires the label window to hang
*below* the prediction-time node in the ACES window tree, so the window cannot start at the trigger
itself.  Each config therefore splits the horizon at one microsecond — MEDS's time resolution::

    input   (-inf, t]            index_timestamp: end   -> the prediction time
    gap     (t, t + 1us)         open on both sides     -> contains no representable timestamp
    target  [t + 1us, t + D]     closed on both sides   -> label: presence of ``query``

Together, ``gap`` and ``target`` contain exactly the microsecond timestamps of ``(t, t + D]``, so the
ACES window is the sampler's window, not an approximation of it.  ``D`` is computed with the same
Polars expression the labeling kernel uses
(:func:`~meds_random_task_sampler.schema.window_length_expr`), so the float32 ``duration_days`` stored in
the rows maps to the identical microsecond count in both places.

**Semantic caveats** (documented, not encoded — ACES windows cannot express them):

- the sampler labels a window that runs past the end of a subject's real record as ``null``
  (censored); a zero-shot labeler instead sees the generated trajectory and decides determinability
  itself;
- the sampler treats ``MEDS_DEATH`` inside the window as complete observation (label ``False`` unless
  ``query`` occurred first); a generated trajectory that stops at death is labeled by the zero-shot
  tool's own rules.
"""

import hashlib
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from meds import LabelSchema

from meds_random_task_sampler.random_sample import _atomic_write_parquet, _unique_tmp_path
from meds_random_task_sampler.schema import TaskQuerySchema, window_length_expr

logger = logging.getLogger(__name__)

#: Bumped whenever the exported config shape or task-id derivation changes.
ACES_EXPORT_VERSION = 1

#: Name of the ACES predicate that matches the queried code.
QUERY_PREDICATE = "query"

TASK_ID = "task_id"
WINDOW = "window"
CRITERIA_FP = "criteria_fp"

_MICROSECONDS_PER_SECOND = 1_000_000


def task_id(query: str, duration_days: float) -> str:
    """Return the stable identifier of one ``(query, duration_days)`` task.

    The id hashes the code and the *float32* duration — the precision task rows store — so a task
    read back from Parquet maps to the same id as the one that was sampled.  It is stable across
    processes, platforms, and runs, and safe as a file name whatever characters the code contains.

    Examples:
        >>> task_id("ICD//A01", 30.0)
        'task_528bb4f20b7af8b8'
        >>> task_id("ICD//A01", 30.0) == task_id("ICD//A01", np.float32(30.0))
        True
        >>> task_id("ICD//A01", 30.0) != task_id("ICD//A01", 90.0)
        True
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(query).encode("utf-8"))
    h.update(b"\x1f")
    h.update(np.float32(duration_days).tobytes())
    return f"task_{h.hexdigest()}"


def _format_seconds(microseconds: int) -> str:
    """Render an exact microsecond count as a duration string ACES (``pytimeparse``) parses exactly.

    Examples:
        >>> _format_seconds(2_592_000_000_000)
        '2592000 seconds'
        >>> _format_seconds(106_666_656_250)
        '106666.65625 seconds'
        >>> _format_seconds(1)
        '0.000001 seconds'
    """
    if microseconds <= 0:
        raise ValueError(f"window length must be positive; got {microseconds} microseconds")
    whole, frac = divmod(int(microseconds), _MICROSECONDS_PER_SECOND)
    if frac == 0:
        return f"{whole} seconds"
    return f"{whole}.{frac:06d}".rstrip("0") + " seconds"


def aces_task_config(query: str, window_microseconds: int) -> dict:
    """Build the future-only ACES config for one task.

    Args:
        query: The MEDS code whose occurrence is the label.
        window_microseconds: The exact window length ``D``; the window is ``(t, t + D]``.

    Returns:
        A plain dict, ready to dump as an ACES task-criteria YAML file.

    Examples:
        >>> cfg = aces_task_config("ICD//A01", 30 * 86_400 * 1_000_000)
        >>> print(yaml.safe_dump(cfg, sort_keys=False), end="")
        predicates:
          query:
            code: ICD//A01
        trigger: _ANY_EVENT
        windows:
          input:
            start: null
            end: trigger
            start_inclusive: true
            end_inclusive: true
            index_timestamp: end
          gap:
            start: input.end
            end: start + 0.000001 seconds
            start_inclusive: false
            end_inclusive: false
          target:
            start: gap.end
            end: start + 2591999.999999 seconds
            start_inclusive: true
            end_inclusive: true
            label: query

        A one-microsecond window is the shortest the split can express:

        >>> aces_task_config("A", 1)["windows"]["target"]["end"]
        Traceback (most recent call last):
            ...
        ValueError: window length must exceed 1 microsecond to be exported to ACES; got 1
    """
    if window_microseconds <= 1:
        raise ValueError(
            f"window length must exceed 1 microsecond to be exported to ACES; got {window_microseconds}"
        )
    return {
        "predicates": {QUERY_PREDICATE: {"code": str(query)}},
        "trigger": "_ANY_EVENT",
        "windows": {
            "input": {
                "start": None,
                "end": "trigger",
                "start_inclusive": True,
                "end_inclusive": True,
                "index_timestamp": "end",
            },
            "gap": {
                "start": "input.end",
                "end": f"start + {_format_seconds(1)}",
                "start_inclusive": False,
                "end_inclusive": False,
            },
            "target": {
                "start": "gap.end",
                "end": f"start + {_format_seconds(window_microseconds - 1)}",
                "start_inclusive": True,
                "end_inclusive": True,
                "label": QUERY_PREDICATE,
            },
        },
    }


def build_task_manifest(tasks: pl.DataFrame | Iterable[tuple[str, float]]) -> pl.DataFrame:
    """Deduplicate ``(query, duration_days)`` pairs into a sorted, id'd task manifest.

    Args:
        tasks: Either a frame with ``query`` and ``duration_days`` columns (e.g. sampled task rows —
            extra columns are ignored) or an iterable of ``(query, duration_days)`` pairs.

    Returns:
        One row per distinct task, sorted by ``(query, duration_days)``, with columns ``task_id``,
        ``query``, ``duration_days`` (``float32``), ``window`` (``duration[us]``), and ``criteria_fp``
        (the config's path relative to the export root).

    Examples:
        >>> m = build_task_manifest([("B", 90), ("A", 30), ("B", 90.0)])
        >>> m.select("query", "duration_days", "window", "criteria_fp").to_dicts()
        [{'query': 'A', 'duration_days': 30.0, 'window': datetime.timedelta(days=30),
          'criteria_fp': 'criteria/task_....yaml'},
         {'query': 'B', 'duration_days': 90.0, 'window': datetime.timedelta(days=90),
          'criteria_fp': 'criteria/task_....yaml'}]
        >>> m["task_id"].to_list() == [task_id("A", 30), task_id("B", 90)]
        True
        >>> build_task_manifest([]).columns
        ['task_id', 'query', 'duration_days', 'window', 'criteria_fp']
    """
    q, d = TaskQuerySchema.query_name, TaskQuerySchema.duration_days_name
    if isinstance(tasks, pl.DataFrame):
        df = tasks.select(q, d)
    else:
        pairs = list(tasks)
        df = pl.DataFrame(
            {q: [str(p[0]) for p in pairs], d: [float(p[1]) for p in pairs]},
            schema={q: pl.Utf8, d: pl.Float64},
        )
    df = (
        df.with_columns(pl.col(q).cast(pl.Utf8), pl.col(d).cast(pl.Float32))
        .unique()
        .sort(q, d)
        .with_columns(window_length_expr(d).alias(WINDOW))
    )
    ids = [task_id(code, dur) for code, dur in zip(df[q], df[d], strict=True)]
    return df.select(
        pl.Series(TASK_ID, ids, dtype=pl.Utf8),
        q,
        d,
        WINDOW,
        pl.Series(CRITERIA_FP, [f"criteria/{i}.yaml" for i in ids], dtype=pl.Utf8),
    )


@dataclass(frozen=True)
class AcesExportResult:
    """Where an export put its files and how many tasks it covers."""

    export_dir: Path
    manifest_fp: Path
    n_tasks: int
    prediction_times_fp: Path | None = None


def _atomic_write_text(text: str, fp: Path) -> None:
    tmp = _unique_tmp_path(fp)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, fp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_aces_task_configs(
    tasks: pl.DataFrame | Iterable[tuple[str, float]],
    export_dir: str | Path,
    prediction_times: pl.DataFrame | None = None,
) -> AcesExportResult:
    """Write one future-only ACES config per distinct task, plus a manifest.

    Layout::

        {export_dir}/
        |-- tasks.parquet              # task_id, query, duration_days, window, criteria_fp
        |-- prediction_times.parquet   # optional: distinct (subject_id, prediction_time)
        `-- criteria/
            `-- {task_id}.yaml         # pass as `task.criteria_fp` to ZSACES_label

    Every file is written atomically, and contents are a deterministic function of the inputs, so
    concurrent or repeated exports of the same tasks (e.g. one per dense-grid shard job) converge on
    identical files.  Configs left over from a different task set are pruned, keeping ``criteria/``
    in one-to-one correspondence with the manifest.

    To score a zero-shot model, join each task's ``ZSACES_label`` output (keyed by ``subject_id`` and
    ``prediction_time``) to the sampler's rows through ``tasks.parquet`` on ``(query, duration_days)``.

    Args:
        tasks: Task rows or ``(query, duration_days)`` pairs; see :func:`build_task_manifest`.
        export_dir: Root to write into.
        prediction_times: Optional frame whose distinct ``(subject_id, prediction_time)`` pairs are the
            contexts a zero-shot model must generate trajectories from.

    Returns:
        An :class:`AcesExportResult`.

    Raises:
        ValueError: If there are no tasks, or a task's window is too short to export.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     res = write_aces_task_configs([("A", 30), ("B", 1.5)], Path(tmp) / "aces")
        ...     print(res.n_tasks, sorted(p.name for p in (res.export_dir / "criteria").iterdir()) ==
        ...           sorted(f"{i}.yaml" for i in pl.read_parquet(res.manifest_fp)["task_id"]))
        ...     _ = write_aces_task_configs([("A", 30)], res.export_dir)  # prunes B's config
        ...     print(len(list((res.export_dir / "criteria").iterdir())))
        2 True
        1
    """
    manifest = build_task_manifest(tasks)
    if manifest.height == 0:
        raise ValueError("no tasks to export")

    export_dir = Path(export_dir)
    criteria_dir = export_dir / "criteria"
    criteria_dir.mkdir(parents=True, exist_ok=True)

    windows_us = manifest[WINDOW].dt.total_microseconds().to_list()
    queries = manifest[TaskQuerySchema.query_name]
    for tid, query, window_us in zip(manifest[TASK_ID], queries, windows_us, strict=True):
        text = yaml.safe_dump(aces_task_config(query, int(window_us)), sort_keys=False)
        _atomic_write_text(text, criteria_dir / f"{tid}.yaml")

    keep = {f"{tid}.yaml" for tid in manifest[TASK_ID]}
    for stale in criteria_dir.glob("task_*.yaml"):
        if stale.name not in keep:
            stale.unlink(missing_ok=True)

    manifest_fp = export_dir / "tasks.parquet"
    _atomic_write_parquet(manifest, manifest_fp)

    pt_fp = None
    if prediction_times is not None:
        pt_fp = export_dir / "prediction_times.parquet"
        unique_times = (
            prediction_times.select(LabelSchema.subject_id_name, LabelSchema.prediction_time_name)
            .unique()
            .sort(LabelSchema.subject_id_name, LabelSchema.prediction_time_name)
        )
        _atomic_write_parquet(unique_times, pt_fp)

    logger.info("Exported %d future-only ACES task config(s) to %s", manifest.height, criteria_dir)
    return AcesExportResult(
        export_dir=export_dir, manifest_fp=manifest_fp, n_tasks=manifest.height, prediction_times_fp=pt_fp
    )


def export_task_rows_to_aces(tasks_dir: str | Path, split: str, export_dir: str | Path) -> AcesExportResult:
    """Export every distinct task found in a finished run's ``{tasks_dir}/{split}/*.parquet`` rows.

    Works for either workflow's output.  Also writes the distinct prediction times, so a random
    sample — which has no ``_unique`` root of its own — gets the trajectory-generation index too.

    Raises:
        FileNotFoundError: If the split has no task-row shards.
    """
    files = sorted((Path(tasks_dir) / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no task-row shards under {Path(tasks_dir) / split}")
    cols = [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ]
    rows = pl.concat([pl.read_parquet(fp, columns=cols) for fp in files], how="vertical_relaxed")
    return write_aces_task_configs(rows, export_dir, prediction_times=rows)
