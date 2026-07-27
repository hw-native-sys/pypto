# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Command-line interface for generating IR pass trace reports."""

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .diff import build_trace
from .discovery import discover_snapshots
from .html import render_html
from .model import IRTraceError


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value}")
    return parsed


def _write_atomic(output: Path, content: str) -> None:
    output_parent = output.parent
    if not output_parent.is_dir():
        raise IRTraceError(f"output directory does not exist: {output_parent}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        temporary.replace(output)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise IRTraceError(f"failed to write {output}: {error}") from error


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pypto-ir-trace")
    parser.add_argument("passes_dump", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("ir_trace.html"))
    parser.add_argument("--context", type=_non_negative_int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a self-contained HTML report from a pass dump directory."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1

    try:
        snapshots = discover_snapshots(args.passes_dump)
        traces = build_trace(snapshots, context=args.context)
        report = render_html(traces, source_name=args.passes_dump.name)
        _write_atomic(args.output, report)
    except IRTraceError as error:
        print(f"pypto-ir-trace: error: {error}", file=sys.stderr)
        return 1
    return 0
