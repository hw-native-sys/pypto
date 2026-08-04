# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Read and compare PTOAS InsertSync JSONL summaries by function.

One PTO file can contain several InCore functions. PTOAS writes one summary
object per function, so comparing JSONL rows by position silently attributes
one function's synchronization to another when traversal order changes. This
module keys every record by its explicit ``function`` field and rejects
duplicates or mismatched function sets.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_MAPPING_METRICS = ("pipe_pair_groups", "pipe_pair_event_ids")
_IGNORED_NUMERIC_FIELDS = {"schema_version"}


def load_sync_summaries(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a PTOAS sync-summary JSONL file keyed by function name.

    Args:
        path: JSONL summary emitted by PTOAS.

    Returns:
        Summary objects keyed by their explicit function names.

    Raises:
        ValueError: A non-empty line is not a JSON object, has no valid
            ``function`` field, or repeats a function name.
    """
    source = Path(path)
    summaries: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{source}:{line_number}: expected a JSON object")
        function = record.get("function")
        if not isinstance(function, str) or not function:
            raise ValueError(f"{source}:{line_number}: missing non-empty 'function'")
        if function in summaries:
            raise ValueError(f"{source}:{line_number}: duplicate function '{function}'")
        summaries[function] = record
    if not summaries:
        raise ValueError(f"{source}: no sync-summary records")
    return summaries


def _numeric_deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, int | float]:
    deltas: dict[str, int | float] = {}
    for key in sorted(set(baseline) | set(candidate)):
        if key in _IGNORED_NUMERIC_FIELDS or key in _MAPPING_METRICS:
            continue
        before = baseline.get(key)
        after = candidate.get(key)
        if (
            isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
        ):
            deltas[key] = after - before
    return deltas


def _mapping_delta(baseline: Any, candidate: Any) -> dict[str, int | float]:
    before = baseline if isinstance(baseline, Mapping) else {}
    after = candidate if isinstance(candidate, Mapping) else {}
    delta: dict[str, int | float] = {}
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key, 0)
        after_value = after.get(key, 0)
        if (
            not isinstance(before_value, (int, float))
            or isinstance(before_value, bool)
            or not isinstance(after_value, (int, float))
            or isinstance(after_value, bool)
        ):
            raise ValueError(f"sync metric '{key}' is not numeric")
        delta[key] = after_value - before_value
    return delta


def diff_sync_summaries(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return candidate-minus-baseline deltas keyed by function.

    The two inputs must contain exactly the same function names. Numeric scalar
    metrics and the two pipe-pair maps emitted by PTOAS are compared
    independently for each function.

    Args:
        baseline: Baseline summary objects keyed by function.
        candidate: Candidate summary objects keyed by function.

    Returns:
        Numeric and pipe-pair deltas for each function.

    Raises:
        ValueError: Function sets differ or a pipe-pair metric is non-numeric.
    """
    baseline_functions = set(baseline)
    candidate_functions = set(candidate)
    if baseline_functions != candidate_functions:
        missing = sorted(baseline_functions - candidate_functions)
        added = sorted(candidate_functions - baseline_functions)
        raise ValueError(f"sync-summary function mismatch: missing={missing}, added={added}")

    result: dict[str, dict[str, Any]] = {}
    for function in sorted(baseline):
        before = baseline[function]
        after = candidate[function]
        function_delta: dict[str, Any] = {"metrics": _numeric_deltas(before, after)}
        for key in _MAPPING_METRICS:
            function_delta[key] = _mapping_delta(before.get(key), after.get(key))
        result[function] = function_delta
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ptoas_sync_summary",
        description="Compare PTOAS InsertSync JSONL summaries by function.",
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Write JSON to this path instead of stdout.")
    args = parser.parse_args(argv)

    try:
        result = diff_sync_summaries(
            load_sync_summaries(args.baseline),
            load_sync_summaries(args.candidate),
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
