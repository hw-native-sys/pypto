# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Apply per-transfer ISA hints to marked PTOAS C++ output.

PTOAS sees ordinary tile operations, preserving memory planning and sync
analysis. Its existing emitc.verbatim support carries the markers. This
finalization belongs to PyPTO, including the debug rebuild path.
"""

import re

from pypto.pypto_core.ir import LoadL2Hint, StoreL2Hint

_MARKER_PREFIX = "// __pypto_l2_hint_"
_BEGIN = re.compile(r"// __pypto_l2_hint_begin (load|store) ([A-Za-z_]\w*)")
_END = "// __pypto_l2_hint_end"
# Skip comments and quoted literals before recognizing intrinsic identifiers.
_CPP_TOKEN = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'
    r"'(?:\\.|[^'\\])*'|[A-Za-z_]\w*|[^\s]"
)
_TRANSFERS = {"TLOAD", "TSTORE", "TSTORE_FP"}


def _apply_transfer_hint(body: str, direction: str, hint: str) -> str:
    tokens = [token for token in _CPP_TOKEN.finditer(body) if not token[0].startswith(("//", "/*", '"', "'"))]
    calls = [
        i for i, token in enumerate(tokens[:-1]) if token[0] in _TRANSFERS and tokens[i + 1][0] in ("(", "<")
    ]
    expected = {"TLOAD"} if direction == "load" else {"TSTORE", "TSTORE_FP"}
    if len(calls) != 1 or tokens[calls[0]][0] not in expected:
        raise RuntimeError(
            f"L2 hint {direction}.{hint}: expected exactly one matching ISA transfer "
            f"between markers, found {[tokens[i][0] for i in calls]}"
        )

    i = calls[0]
    callee = tokens[i]
    template = tokens[i + 1]
    enum = "TLoadL2Hint" if direction == "load" else "TStoreL2Hint"
    argument = f"pto::{enum}::{hint}"
    name = "TLOAD" if direction == "load" else "TSTORE"
    if template[0] == "(":
        return f"{body[: callee.start()]}{name}<{argument}>{body[callee.end() :]}"

    depth = 0
    for j in range(i + 1, len(tokens)):
        token = tokens[j]
        if token[0] == "<":
            depth += 1
        elif token[0] == ">":
            depth -= 1
            if depth == 0:
                if j + 1 >= len(tokens) or tokens[j + 1][0] != "(":
                    break
                existing = body[template.end() : token.start()]
                if "L2Hint" in existing:
                    raise RuntimeError(f"L2 hint {direction}.{hint}: ISA call already has a hint")
                comma = ", " if existing.strip() else ""
                return (
                    f"{body[: callee.start()]}{name}{body[callee.end() : template.end()]}"
                    f"{argument}{comma}{existing}{body[token.start() :]}"
                )
        elif token[0] in (";", "{", "}"):
            break
    raise RuntimeError(f"L2 hint {direction}.{hint}: malformed ISA template argument list")


def apply_l2_hints(content: str) -> str:
    """Consume markers and prepend the selected PTO-ISA template parameter.

    Unmarked C++ is unchanged. An unsupported assembler output shape is an
    error, never an unhinted fallback. No ordering correspondence between
    separate IR operations and C++ calls is assumed.
    """
    if _MARKER_PREFIX not in content:
        return content

    output: list[str] = []
    region: list[str] = []
    active: tuple[str, str] | None = None
    for line in content.splitlines(keepends=True):
        # EmitC appends a statement terminator to verbatim ops in some
        # structured-control-flow regions, even when the text is a comment.
        marker = line.strip().removesuffix(";")
        if not marker.startswith(_MARKER_PREFIX):
            (region if active is not None else output).append(line)
            continue
        if marker == _END:
            if active is None:
                raise RuntimeError("L2 hint: end marker without a begin marker")
            output.append(_apply_transfer_hint("".join(region), *active))
            active = None
            region.clear()
            continue
        match = _BEGIN.fullmatch(marker)
        if match is None:
            raise RuntimeError(f"L2 hint: malformed marker {marker!r}")
        if active is not None:
            raise RuntimeError("L2 hint: nested begin markers")
        direction, hint = match.groups()
        enum = LoadL2Hint if direction == "load" else StoreL2Hint
        if hint not in enum.__members__:
            raise RuntimeError(f"L2 hint: invalid {direction} hint {hint!r}")
        active = (direction, hint)
    if active is not None:
        raise RuntimeError(f"L2 hint {active[0]}.{active[1]}: missing end marker")
    return "".join(output)
