"""CLI: count rows whose label column normalizes to ``parse_error`` in a Parquet file."""
# python -m src.quanlitative_evaluation.exploretory <parquet file path> # default: --column guard_label
# python -m src.quanlitative_evaluation.exploretory path/to/file.parquet --column other_col

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

TARGET = "parse_error"


def count_parse_errors(series: pd.Series) -> int:
    s = series.astype(str).str.strip().str.lower()
    return int((s == TARGET).sum())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            f"Count rows in a Parquet file where the given column equals "
            f"{TARGET!r} (case-insensitive, stripped)."
        )
    )
    p.add_argument(
        "parquet",
        type=Path,
        help="Path to the Parquet file.",
    )
    p.add_argument(
        "--column",
        default="guard_label",
        help="Column to inspect (default: guard_label).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path: Path = args.parquet
    col: str = args.column

    if not path.exists():
        print(f"error: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"error: failed to read Parquet: {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if col not in df.columns:
        print(
            f"error: column {col!r} not found. Available: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    n = len(df)
    pe = count_parse_errors(df[col])

    print(path.name)
    print(f"  column: {col}")
    print(f"  rows: {n}")
    print(f"  {TARGET}: {pe}")
    if n:
        print(f"  {TARGET} (%): {100.0 * pe / n:.2f}")


if __name__ == "__main__":
    main()
