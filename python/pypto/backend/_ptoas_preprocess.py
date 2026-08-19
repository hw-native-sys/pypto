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
_PTOAS_FILLPAD_MODE_CALL_RE = re.compile(
    r"\bTFILLPAD\s*<\s*pto::TFillPadMode::(?P<mode>Expand|InPlace)\s*>\s*\("
)
_PTOAS_FILLPAD_MODE_INTRINSIC = {
    "Expand": "TFILLPAD_EXPAND(",
    "InPlace": "TFILLPAD_INPLACE(",
}
# A2/A3 TCI_b32_normal (validCol < 64) ends in count-mode without restoring
# mask_norm.  The next TCVT tail calls SetContinuousMask(n) which interprets
# the leftover count as a huge bitmask and UB-OOBs.  Match only the dst tile's
# static row/col shape on the first Tile<...> inside TCI<...>.
_TCI_NARROW_DST_COLS_RE = re.compile(r"TCI<[^;]*?Tile<[^,]+,\s*\w+,\s*1,\s*(?P<cols>\d+),")
_TCI_MASK_RESTORE_THRESHOLD = 64
# PTO-ISA f51c92f's tmp-path TCI_IMPL dispatches b16/b32 with a *runtime* if,
# so the b16 arm's TMULS_IMPL(dst, dst, -1) is instantiated for every element
# type -- and TMULS static-asserts on unsigned. Unsigned destinations therefore
# cannot take the tmp overload; reroute them to the scalar no-tmp overload,
# which compiles for unsigned and computes both orders correctly.
_TCI_UNSIGNED_TMP_CALL_RE = re.compile(
    r"TCI<(?P<dst_tile>Tile<[^;]*?uint(?:16|32)_t[^;]*?>),\s*"
    r"(?P<tmp_tile>Tile<[^;]*?>),\s*"
    r"(?P<elem>uint(?:16|32)_t),\s*(?P<descending>[01])>"
    r"\((?P<args>[^;]*?)\);"
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


def _restore_tci_mask_norm_after_narrow_arange(content: str) -> str:
    """Restore vector mask mode after narrow A2/A3 ``TCI`` (``tile.ci``) calls.

    PTO-ISA ``TCI_b32_normal`` leaves the vector unit in count-mode when
    ``validCol < 64``.  PTOAS cannot emit ``pto.ub.set_mask_norm`` (the op is
    marked illegal during lowering), so repair the generated C++ here.
    """
    if "TCI<" not in content:
        return content

    lines = content.splitlines(keepends=True)
    out: list[str] = []
    for index, line in enumerate(lines):
        out.append(line)
        match = _TCI_NARROW_DST_COLS_RE.search(line)
        if match is None:
            continue
        if int(match.group("cols")) >= _TCI_MASK_RESTORE_THRESHOLD:
            continue
        next_line = lines[index + 1].lstrip() if index + 1 < len(lines) else ""
        if next_line.startswith("set_mask_norm();"):
            continue
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f"{indent}set_mask_norm();\n")
        out.append(f"{indent}set_vector_mask(-1, -1);\n")
    return "".join(out)


def _restore_fillpad_mode_intrinsics(content: str) -> str:
    """Adapt PTOAS v0.58's TFillPadMode spelling to the pinned PTO-ISA API.

    TODO: remove this rewrite once the pinned PTO-ISA version is updated to
    expose ``TFillPadMode``. Until then PTOAS v0.58 emits
    ``TFILLPAD<pto::TFillPadMode::Expand|InPlace>`` while the current ISA only
    has ``TFILLPAD_EXPAND`` / ``TFILLPAD_INPLACE`` and no ``TFillPadMode`` enum.
    Keep the dialect input accepted by PTOAS and repair only these generated
    intrinsic names before embedding the C++ body.
    """
    if "TFillPadMode" not in content:
        return content
    return _PTOAS_FILLPAD_MODE_CALL_RE.sub(
        lambda match: _PTOAS_FILLPAD_MODE_INTRINSIC[match.group("mode")], content
    )


def _reroute_unsigned_tci_tmp_calls(content: str) -> str:
    """Reroute unsigned tmp-path TCI calls to the scalar no-tmp overload.

    TODO: remove this rewrite once the pinned PTO-ISA version includes the
    a2a3 TCI tmp fix (descending as 2S-(S+i) on a signed view). Until then the
    tmp overload does not compile for uint16/uint32 destinations; the scalar
    overload has no such instantiation and produces the same sequence.
    """
    if "TCI<" not in content:
        return content

    def _drop_tmp_arg(match: re.Match[str]) -> str:
        args = [arg.strip() for arg in match.group("args").split(",")]
        if len(args) < 3:  # not the (dst, start, tmp) form; leave untouched
            return match.group(0)
        return (
            f"TCI<{match.group('dst_tile')}, {match.group('elem')}, {match.group('descending')}>"
            f"({args[0]}, {args[1]});"
        )

    return _TCI_UNSIGNED_TMP_CALL_RE.sub(_drop_tmp_arg, content)


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
    # ``Tensor``, while Simpler renamed that runtime ABI type to
    # ``ChipTensor`` without a compatibility alias (simpler#1681).  Rewrite the
    # exact identifier before embedding the body in PyPTO's wrapper; names such
    # as ``GlobalTensor`` are intentionally unaffected by the word boundaries.
    result = re.sub(r"\bTensor\b", "ChipTensor", result)
    result = _restore_fillpad_mode_intrinsics(result)
    result = _restore_tci_mask_norm_after_narrow_arange(result)
    result = _reroute_unsigned_tci_tmp_calls(result)
    result = re.sub(
        r'(?:extern\s*"C"\s*)?(?:__global__\s+)?AICORE\s+void',
        "static __aicore__ void",
        result,
    )
    return re.sub(r"\bAICORE\b", "__aicore__", result)
