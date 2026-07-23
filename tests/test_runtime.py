"""Runtime initialization tests."""

import os
import subprocess
import sys


def test_package_sets_polars_thread_limit_before_import() -> None:
    """Importing the package initializes Polars with one thread by default."""
    env = os.environ.copy()
    env.pop("POLARS_MAX_THREADS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import meds_random_task_sampler; "
                "import polars as pl; "
                "print(os.environ['POLARS_MAX_THREADS']); "
                "print(pl.thread_pool_size())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.splitlines() == ["1", "1"]
