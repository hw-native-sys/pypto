# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared preprocessing for C++ emitted by PTOAS."""

import re
from bisect import bisect_left

_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_PTOAS_UB_POINTER_ALIAS_RE = re.compile(
    r"^\s*__(?:ubuf|cbuf)__\s+.+?\*\s*(?P<alias>[A-Za-z_]\w*)\s*="
    r"\s*(?P<wrapper>[A-Za-z_]\w*)\.data\(\);\s*$"
)
_PTOAS_GM_POINTER_ALIAS_RE = re.compile(
    r"^\s*__gm__\s+.+?\*\s*(?P<alias>[A-Za-z_]\w*)\s*="
    r"\s*\(__gm__\s+.+?\*\)\s*(?P<wrapper>[A-Za-z_]\w*);\s*$"
)
_PTOAS_MGATHER_CALL_RE = re.compile(
    r"(?P<prefix>\bMGATHER(?:<[^;()]+>)?\()"
    r"(?P<dst>[A-Za-z_]\w*)\s*,\s*"
    r"(?P<table>[A-Za-z_]\w*)\s*,\s*"
    r"(?P<idx>[A-Za-z_]\w*)"
    r"(?:\s*,\s*(?P<scratch>[A-Za-z_]\w*))?"
    r"(?P<suffix>\);)"
)


def _restore_mgather_wrapper_operands(content: str) -> str:
    """Undo PTOAS' legacy pointer lowering for the MGATHER wrapper ABI.

    PTOAS through v0.53 lowers partition-view MGATHER operands to raw UB/GM
    pointers even though the current PTO-ISA intrinsic accepts Tile and
    GlobalTensor wrappers. Rewrite the three-argument Vec/Mat-row and
    four-argument Mat-elem forms when every alias is uniquely used.
    """
    if "MGATHER" not in content:
        return content

    lines = content.splitlines(keepends=True)
    aliases: dict[str, list[tuple[str, int]]] = {}
    identifier_occurrences: dict[str, list[int]] = {}
    for line_index, line in enumerate(lines):
        for identifier in _IDENTIFIER_RE.findall(line):
            identifier_occurrences.setdefault(identifier, []).append(line_index)
        for pattern in (_PTOAS_UB_POINTER_ALIAS_RE, _PTOAS_GM_POINTER_ALIAS_RE):
            if match := pattern.match(line):
                aliases.setdefault(match.group("alias"), []).append((match.group("wrapper"), line_index))
                break
    alias_definition_lines = {
        alias: [line_index for _, line_index in definitions] for alias, definitions in aliases.items()
    }

    def find_unique_definition(alias: str, call_line_index: int) -> tuple[str, int] | None:
        definitions = aliases.get(alias, [])
        definition_lines = alias_definition_lines.get(alias, [])
        definition_position = bisect_left(definition_lines, call_line_index) - 1
        if definition_position < 0:
            return None

        definition = definitions[definition_position]
        scope_end = (
            definition_lines[definition_position + 1]
            if definition_position + 1 < len(definition_lines)
            else len(lines)
        )
        occurrence_lines = identifier_occurrences.get(alias, [])
        occurrence_start = bisect_left(occurrence_lines, definition[1])
        occurrence_end = bisect_left(occurrence_lines, scope_end)
        if occurrence_end - occurrence_start != 2:
            return None
        return definition

    declaration_lines_to_drop: set[int] = set()
    for line_index, line in enumerate(lines):
        match = _PTOAS_MGATHER_CALL_RE.search(line)
        if match is None:
            continue

        required_names = [match.group("dst"), match.group("table"), match.group("idx")]
        required_definitions = [find_unique_definition(argument, line_index) for argument in required_names]
        if any(definition is None for definition in required_definitions):
            continue

        definitions = [definition for definition in required_definitions if definition is not None]
        wrapper_names = [definition[0] for definition in definitions]
        if scratch_name := match.group("scratch"):
            scratch_definition = find_unique_definition(scratch_name, line_index)
            if scratch_definition is not None:
                definitions.append(scratch_definition)
                wrapper_names.append(scratch_definition[0])
            elif scratch_name in aliases:
                continue
            else:
                wrapper_names.append(scratch_name)

        replacement = f"{match.group('prefix')}{', '.join(wrapper_names)}{match.group('suffix')}"
        lines[line_index] = f"{line[: match.start()]}{replacement}{line[match.end() :]}"
        declaration_lines_to_drop.update(definition[1] for definition in definitions)

    return "".join(
        line for line_index, line in enumerate(lines) if line_index not in declaration_lines_to_drop
    )


def preprocess_ptoas_output(content: str) -> str:
    """Prepare PTOAS output for embedding in PyPTO kernel wrappers."""
    lines = content.splitlines(keepends=True)
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#include") and (
            "pto-inst" in stripped or "cstdint" in stripped or "tensor.h" in stripped
        ):
            continue
        if stripped == "using namespace pto;":
            continue
        if stripped.startswith("set_ffts_base_addr("):
            continue
        filtered.append(line)

    result = _restore_mgather_wrapper_operands("".join(filtered))
    # The current PTOAS emitter spells the 128-byte, chip-resident descriptor
    # ``Tensor``.  Simpler renamed that runtime ABI type to ``ChipTensor``
    # (simpler#1681), then split it in two (simpler#1974): the 72-byte
    # ``ChipTensor`` is now only the argument as it arrives at the boundary,
    # while the 128-byte descriptor a kernel actually reads is its runtime's
    # own type, spelled ``TaskTensor`` by the per-runtime ``tensor.h`` shim.
    # Rewrite the exact identifier before embedding the body in PyPTO's
    # wrapper; names such as ``GlobalTensor`` are intentionally unaffected by
    # the word boundaries.
    result = re.sub(r"\bTensor\b", "TaskTensor", result)
    result = re.sub(
        r'(?:extern\s*"C"\s*)?(?:__global__\s+)?AICORE\s+void',
        "static __aicore__ void",
        result,
    )
    return re.sub(r"\bAICORE\b", "__aicore__", result)
