# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiler evidence for device callees with no provisioned task stack."""

from pathlib import Path


def validate_stack_usage(report: Path, functions: frozenset[str]) -> None:
    """Require static zero-stack evidence for every retained device function.

    The persistent runtime has no per-task compiler stack reservation contract.
    Accepting a positive frame size, or summing frames without proving the call
    graph, would invent a budget. Reject both until that contract exists. The
    report must be generated beside the object in the same private compilation;
    a source declaration or a report from a different build is not evidence.
    """
    if not report.is_file():
        raise ValueError("Device image: compiler produced no stack-usage report")
    records: dict[str, tuple[str, str]] = {}
    for line in report.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or ":" not in fields[0]:
            raise ValueError(f"Device image: malformed stack-usage record {line!r}")
        # CCEC emits source-path:mangled-symbol, so colons in paths are safe.
        name = fields[0].rsplit(":", 1)[1]
        if not name or name in records or not fields[1].isascii() or not fields[1].isdecimal():
            raise ValueError(f"Device image: invalid or duplicate stack-usage record {line!r}")
        records[name] = (fields[1], fields[2])
    missing = functions - records.keys()
    if missing:
        raise ValueError(f"Device image: missing stack-usage evidence for {sorted(missing)}")
    for name in functions:
        size, kind = records[name]
        if int(size) != 0 or kind != "static":
            raise ValueError(
                f"Device image: {name!r} needs {size} bytes of {kind!r} stack; "
                "only static zero-stack functions are supported"
            )
