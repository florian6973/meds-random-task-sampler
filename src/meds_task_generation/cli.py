"""Command-line interface for task collection generation and inspection."""

import argparse
import json
from pathlib import Path

import polars as pl

from meds_task_generation.generation import generate_collection
from meds_task_generation.manifest import load_collection_config
from meds_task_generation.validation import validate_collection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meds-task-generation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Generate a task collection")
    generate.add_argument("--data-dir", required=True, type=Path)
    generate.add_argument("--config", required=True, type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--overwrite", action="store_true")
    validate = subparsers.add_parser("validate", help="Validate a generated collection")
    validate.add_argument("--collection-dir", required=True, type=Path)
    summarize = subparsers.add_parser("summarize", help="Print generated collection statistics")
    summarize.add_argument("--collection-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the command-line interface."""
    args = _parser().parse_args(argv)
    if args.command == "generate":
        config = load_collection_config(args.config)
        output = generate_collection(config, args.data_dir, args.output_dir, overwrite=args.overwrite)
        print(json.dumps(validate_collection(output), sort_keys=True))
    elif args.command == "validate":
        print(json.dumps(validate_collection(args.collection_dir), sort_keys=True))
    else:
        print(pl.read_parquet(args.collection_dir / "summary.parquet"))
