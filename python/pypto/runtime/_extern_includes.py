# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conservative include scanning for relocatable extern source packages."""

import re
from collections.abc import Iterator


class UnsupportedArtifactInput(ValueError):
    """A valid compiler input needs ordinary private compilation, not packaging."""


_TOKENS = re.compile(
    r'R"([^\s()\\]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\])*"|'
    r"'(?:\\.|[^'\\])*'|/\*.*?\*/|//[^\n]*",
    re.DOTALL,
)
_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(\w+)\b([^\n]*)", re.MULTILINE)
_LITERAL = re.compile(r'(?:"([^"\n]+)"|<([^>\n]+)>)\s*')


def _condition(expression: str) -> bool | None:
    """Recognize only literal constants; unknown conditions may take either branch."""
    expression = expression.strip()
    while expression.startswith("(") and expression.endswith(")"):
        expression = expression[1:-1].strip()
    if expression in ("0", "1"):
        return expression == "1"
    return None


def literal_includes(data: bytes) -> Iterator[tuple[str, str]]:
    """Yield possible literal includes, ignoring comments and provably dead branches.

    This is deliberately not a C preprocessor. Unknown conditions are scanned
    conservatively, and active macro includes request a private-build fallback.
    Decoding is only for scanning; the packager retains the original bytes.
    """
    text = re.sub(r"\\\r?\n", "", data.decode("utf-8", errors="replace"))

    def mask(match: re.Match[str]) -> str:
        token = match.group()
        if token.startswith(("/*", "//", 'R"')):
            return "".join("\n" if char == "\n" else " " for char in token)
        return token

    text = _TOKENS.sub(mask, text)
    # Each frame stores parent reachability and whether an untaken branch is possible.
    stack: list[tuple[bool, bool, bool]] = []
    active = True
    for match in _DIRECTIVE.finditer(text):
        directive, expression = match.groups()
        expression = expression.strip()
        if directive in ("if", "ifdef", "ifndef"):
            condition = _condition(expression) if directive == "if" else None
            stack.append((active, condition is not True, False))
            active = active and condition is not False
        elif directive in ("elif", "else"):
            if not stack or stack[-1][2]:
                raise ValueError("Malformed extern conditional directive")
            parent, remaining, _ = stack[-1]
            condition = _condition(expression) if directive == "elif" else True
            active = parent and remaining and condition is not False
            stack[-1] = (parent, remaining and condition is not True, directive == "else")
        elif directive == "endif":
            if not stack:
                raise ValueError("Unmatched extern #endif")
            active = stack.pop()[0]
        elif directive == "include" and active:
            literal = _LITERAL.fullmatch(expression)
            if literal is None:
                raise UnsupportedArtifactInput(f"Artifact extern includes must be literal: {expression}")
            quoted, angle = literal.groups()
            yield ('"', quoted) if quoted is not None else ("<", angle)
    if stack:
        raise ValueError("Unterminated extern conditional directive")
