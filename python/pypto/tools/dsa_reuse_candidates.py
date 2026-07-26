# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Inspect raw PyPTO DSA reuse candidates before promotion-policy filtering."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

_RECORDS_KEY = "recognized_reuse_candidate_records_v4"


@dataclass(frozen=True)
class ReuseCandidateRecord:
    first_buffer: int
    second_buffer: int
    prior_buffer: int
    next_buffer: int
    prior_route: str
    next_route: str
    dependence: Literal["write_after_read", "write_after_write"]
    ordered_by_logical_dag: bool
    hazard: Literal["cross_resource", "same_resource"]
    dag_path: tuple[str, ...]
    fields: tuple[str, ...]


def _split_arrow(value: str, separator: str, field: str) -> tuple[str, str]:
    parts = value.split(separator)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid {field}: '{value}'")
    return parts[0], parts[1]


def parse_candidate_record(record: str) -> ReuseCandidateRecord:
    """Parse and validate one raw reuse-candidate record.

    Args:
        record: One item from ``recognized_reuse_candidate_records_v4``.

    Returns:
        Parsed candidate with validated ordering evidence.

    Raises:
        ValueError: The record is malformed or its ordering flag and witness
            disagree.
    """
    fields = tuple(record.split(","))
    if len(fields) < 17:
        raise ValueError(f"candidate record has only {len(fields)} fields")
    try:
        first_buffer = int(fields[0])
        second_buffer = int(fields[1])
        prior_text, next_text = _split_arrow(fields[2], "->", "buffer handoff")
        prior_buffer = int(prior_text)
        next_buffer = int(next_text)
    except ValueError as error:
        raise ValueError(f"invalid buffer identifiers in '{record}'") from error

    prior_route, next_route = _split_arrow(fields[3], "=>", "route handoff")
    dependence_text = fields[5]
    if dependence_text not in {"write_after_read", "write_after_write"}:
        raise ValueError(f"invalid dependence '{dependence_text}'")
    dependence = cast(Literal["write_after_read", "write_after_write"], dependence_text)
    if fields[6] not in {"logical_order", "no_logical_order"}:
        raise ValueError(f"invalid logical-order field '{fields[6]}'")
    ordered = fields[6] == "logical_order"

    keyed: dict[str, str] = {}
    for field in fields[13:]:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        keyed[key] = value
    hazard_text = keyed.get("hazard")
    if hazard_text not in {"cross_resource", "same_resource"}:
        raise ValueError(f"invalid or missing hazard in '{record}'")
    hazard = cast(Literal["cross_resource", "same_resource"], hazard_text)
    path_text = keyed.get("dag_path")
    if path_text is None:
        raise ValueError(f"missing dag_path in '{record}'")
    dag_path = () if path_text == "none" else tuple(path_text.split(">"))
    if ordered and len(dag_path) < 2:
        raise ValueError("logical_order candidate requires a dependency path")
    if not ordered and dag_path:
        raise ValueError("no_logical_order candidate must use dag_path=none")

    return ReuseCandidateRecord(
        first_buffer=first_buffer,
        second_buffer=second_buffer,
        prior_buffer=prior_buffer,
        next_buffer=next_buffer,
        prior_route=prior_route,
        next_route=next_route,
        dependence=dependence,
        ordered_by_logical_dag=ordered,
        hazard=hazard,
        dag_path=dag_path,
        fields=fields,
    )


def load_candidate_records(path: str | Path) -> list[ReuseCandidateRecord]:
    """Load raw candidate records from an exported ``.dsa.json`` document.

    Args:
        path: Exported DSA problem document.

    Returns:
        Raw candidates before promotion-policy filtering.

    Raises:
        ValueError: The document, metadata, records, or candidate count is
            invalid.
    """
    source = Path(path)
    try:
        document: Any = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON: {error.msg}") from error
    if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
        raise ValueError(f"{source}: missing metadata object")
    metadata = document["metadata"]
    encoded = metadata.get(_RECORDS_KEY)
    if not isinstance(encoded, str):
        raise ValueError(f"{source}: missing metadata.{_RECORDS_KEY}")
    records = [] if not encoded else [parse_candidate_record(record) for record in encoded.split(";")]
    expected_text = metadata.get("recognized_reuse_candidates")
    if isinstance(expected_text, str):
        try:
            expected = int(expected_text)
        except ValueError as error:
            raise ValueError(f"{source}: invalid recognized_reuse_candidates '{expected_text}'") from error
        if expected != len(records):
            raise ValueError(
                f"{source}: candidate count mismatch: metadata={expected}, records={len(records)}"
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsa_reuse_candidates",
        description="Print raw, pre-promotion PyPTO DSA reuse candidates as JSON.",
    )
    parser.add_argument("problem", type=Path)
    parser.add_argument("--hazard", choices=("all", "cross_resource", "same_resource"), default="all")
    parser.add_argument("--ordering", choices=("all", "ordered", "unordered"), default="all")
    args = parser.parse_args(argv)

    try:
        records = load_candidate_records(args.problem)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    selected = [
        record
        for record in records
        if args.hazard in ("all", record.hazard)
        and (
            args.ordering == "all"
            or (args.ordering == "ordered" and record.ordered_by_logical_dag)
            or (args.ordering == "unordered" and not record.ordered_by_logical_dag)
        )
    ]
    json.dump([asdict(record) for record in selected], sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
