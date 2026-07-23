"""Package provenance recorded alongside generated task artifacts."""

from importlib.metadata import version

PACKAGE_NAME = "meds-random-task-sampler"


def package_version() -> str:
    """Return the installed sampler package version."""
    return version(PACKAGE_NAME)
