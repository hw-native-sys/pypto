# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that codegen and backend emitters classify their checks correctly.

``src/codegen`` and ``src/backend`` run strictly *after* verification. An invariant that fails
there is a compiler bug, not user input, so it belongs to ``INTERNAL_CHECK`` /
``INTERNAL_CHECK_SPAN`` (``pypto::InternalError``) rather than ``CHECK`` / ``CHECK_SPAN``
(``pypto::ValueError``). See ``.claude/rules/error-checking.md`` and
``docs/en/dev/02-error-handling.md``.

These sites cannot be reached from Python by construction -- arity and operand types are settled
by the op registry's deduce-type functions at IR-construction time, and ``Call`` / ``ForStmt`` have
no Python bindings -- so no runtime test can hold the classification in place. This lint is the
guard.

Two rules, both scoped to ``src/codegen`` and ``src/backend``:

Rule A -- a ``CHECK`` whose message says "Internal error". The macro throws ``ValueError`` while
the message tells the user it is a compiler bug; one of the two is wrong.

Rule B -- a ``CHECK`` on a call's argument count. Arity is fixed by the op definition and enforced
before codegen runs, so a mismatch is a broken pass or a broken op definition.

The scanner walks each file once, tracking comments and string literals, and extracts one whole
macro statement at a time (balanced predicate parens, then to the terminating ``;``). A naive
fixed-line-window scan instead reports adjacent ``INTERNAL_CHECK_SPAN`` lines that legitimately
say "Internal error" -- ``pto_ops_crosscore.cpp`` and ``scope_outline_utils.h`` both have that
shape.

C++ raw string literals get their own masking pass. Emitter code builds target syntax out of
literals such as ``R"(", dtype="opaque", count=)"`` (``distributed_codegen.cpp``), whose body
carries unpaired quotes. Scanning those as ordinary literals desynchronizes the masker, which
both hides real checks after the literal and can report text inside one as code.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Trees that run after verification. Everything here is post-verification by construction.
SCANNED_DIRS = ("src/codegen", "src/backend")

SOURCE_SUFFIXES = frozenset({".cpp", ".cc", ".h", ".hpp"})

# Sites that are genuinely user-facing despite matching a rule. Each entry is
# "<repo-relative path>:<line>" and needs a comment saying why the user can actually reach it.
ALLOWLIST: frozenset[str] = frozenset()

# ``CHECK(`` / ``CHECK_SPAN(`` but never ``INTERNAL_CHECK(`` / ``INTERNAL_CHECK_SPAN(``.
_MACRO_RE = re.compile(r"(?<![A-Za-z0-9_])(CHECK|CHECK_SPAN)\s*\(")

# ``op->args_.size()`` / ``call->args_.empty()`` and friends.
_ARITY_RE = re.compile(r"\bargs_\s*\.\s*(?:size|empty)\s*\(")

# ``const size_t arity = op->args_.size();`` -- a local alias for the argument count.
_ARITY_ALIAS_RE = re.compile(
    r"\b(?:const\s+)?(?:size_t|auto|int|int64_t|std::size_t)\s+(\w+)\s*=\s*[^;]*"
    r"\bargs_\s*\.\s*(?:size|empty)\s*\(\s*\)"
)

_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")

_INTERNAL_MARKER = "internal error"


def get_git_tracked_sources(root_dir: Path) -> list[Path]:
    """Get the git-tracked C++ sources under the scanned emitter trees."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", *SCANNED_DIRS],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to get git tracked files: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found", file=sys.stderr)
        sys.exit(1)

    files = []
    for line in result.stdout.splitlines():
        path = root_dir / line
        if Path(line).suffix in SOURCE_SUFFIXES and path.is_file():
            files.append(path)
    return sorted(files)


def _blank_line_comment(text: str, out: list[str], i: int, n: int) -> int:
    """Blank a ``//`` comment through end of line; return the offset just past it."""
    while i < n and text[i] != "\n":
        out[i] = " "
        i += 1
    return i


def _blank_block_comment(text: str, out: list[str], i: int, n: int) -> int:
    """Blank a ``/* ... */`` comment, keeping newlines; return the offset just past it."""
    while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
        if text[i] != "\n":
            out[i] = " "
        i += 1
    for _ in range(2):
        if i < n:
            out[i] = " "
            i += 1
    return i


# C++ raw-string prefixes, longest first so `u8R` wins over `R`.
_RAW_PREFIXES = ("u8R", "LR", "uR", "UR", "R")

# A raw-string delimiter is at most 16 characters and excludes whitespace,
# parentheses and backslash.
_RAW_DELIM_MAX = 16


def _raw_prefix_start(text: str, quote: int) -> int:
    """Offset where a raw-string prefix ending at *quote* begins, or -1 for none."""
    for prefix in _RAW_PREFIXES:
        start = quote - len(prefix)
        if start < 0 or text[start:quote] != prefix:
            continue
        before = text[start - 1] if start > 0 else ""
        if not (before.isalnum() or before == "_"):
            return start
    return -1


def _blank_raw_string(text: str, out: list[str], i: int, n: int) -> tuple[int, int, int] | None:
    """Blank a raw string literal ``R"delim( ... )delim"`` opening at *i*.

    Returns ``(offset past the literal, content start, content end)``, or None when the
    literal is malformed, so the caller can fall back to ordinary quote handling.
    """
    j = i + 1
    while j < n and text[j] != "(" and (j - i - 1) < _RAW_DELIM_MAX:
        if text[j].isspace() or text[j] in ")\\":
            return None
        j += 1
    if j >= n or text[j] != "(":
        return None
    terminator = ")" + text[i + 1 : j] + '"'
    body = j + 1
    end = text.find(terminator, body)
    stop = n if end == -1 else end + len(terminator)
    end = n if end == -1 else end
    for k in range(i + 1, min(stop, n)):
        if text[k] != "\n":
            out[k] = " "
    return stop, body, min(end, n)


def _blank_quoted(text: str, out: list[str], i: int, n: int, quote: str) -> tuple[int, int]:
    """Blank the contents of a ``quote``-delimited literal starting at its opening quote.

    Returns ``(offset just past the literal, offset of its first content char)``.
    """
    i += 1
    start = i
    while i < n and text[i] != quote:
        if text[i] == "\\":
            out[i] = " "
            i += 1
        if i < n:
            if text[i] != "\n":
                out[i] = " "
            i += 1
    return i + 1, start


def _blank_comments_and_strings(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Blank out comments and string-literal *contents*, keeping every offset stable.

    Returns the masked text plus the ``(start, end)`` span of each string literal's contents, so
    the caller can still read message text without tripping over ``(``, ``)`` or ``;`` inside it.
    """
    out = list(text)
    strings: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            i = _blank_line_comment(text, out, i, n)
        elif ch == "/" and nxt == "*":
            i = _blank_block_comment(text, out, i, n)
        elif ch == '"':
            raw = None
            if _raw_prefix_start(text, i) >= 0:
                raw = _blank_raw_string(text, out, i, n)
            if raw is not None:
                i, body_start, body_end = raw
                strings.append((body_start, body_end))
            else:
                i, start = _blank_quoted(text, out, i, n, '"')
                strings.append((start, min(i - 1, n)))
        elif ch == "'":
            i, _ = _blank_quoted(text, out, i, n, "'")
        else:
            i += 1
    return "".join(out), strings


def _statement_end(masked: str, close_paren: int) -> int:
    """Offset of the ``;`` terminating the macro statement, scanning from the predicate's ``)``."""
    depth = 0
    for i in range(close_paren + 1, len(masked)):
        ch = masked[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth <= 0:
            return i
    return len(masked)


def _matching_paren(masked: str, open_paren: int) -> int:
    """Offset of the ``)`` closing the ``(`` at *open_paren*, or -1 when unbalanced."""
    depth = 0
    for i in range(open_paren, len(masked)):
        if masked[i] == "(":
            depth += 1
        elif masked[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def find_violations(path: Path) -> list[tuple[int, str, str]]:
    """Return ``(line, rule, detail)`` for every mis-classified check in *path*."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error: Failed to read {path}: {e}", file=sys.stderr)
        sys.exit(1)

    masked, strings = _blank_comments_and_strings(text)
    aliases = {m.group(1) for m in _ARITY_ALIAS_RE.finditer(masked)}

    violations: list[tuple[int, str, str]] = []
    for match in _MACRO_RE.finditer(masked):
        open_paren = match.end() - 1
        close_paren = _matching_paren(masked, open_paren)
        if close_paren < 0:
            continue
        end = _statement_end(masked, close_paren)
        line = text.count("\n", 0, match.start()) + 1
        macro = match.group(1)

        predicate = masked[open_paren + 1 : close_paren]
        message = " ".join(text[s:e] for s, e in strings if s >= close_paren and e <= end).lower()

        if _INTERNAL_MARKER in message:
            violations.append((line, "A", f'{macro} message says "Internal error"'))
            continue

        idents = set(_IDENT_RE.findall(predicate))
        if _ARITY_RE.search(predicate) or (idents & aliases):
            violations.append((line, "B", f"{macro} on a call's argument count"))

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that codegen/backend emitters use INTERNAL_CHECK for post-verification invariants."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to the repo containing this script)",
    )
    args = parser.parse_args()
    root_dir = args.root.resolve()

    total = 0
    for path in get_git_tracked_sources(root_dir):
        rel = path.relative_to(root_dir).as_posix()
        for line, rule, detail in find_violations(path):
            if f"{rel}:{line}" in ALLOWLIST:
                continue
            print(f"{rel}:{line}: [rule {rule}] {detail}")
            total += 1

    if total:
        print(
            f"\nFound {total} mis-classified check(s) in {' and '.join(SCANNED_DIRS)}.\n"
            "These trees run after verification, so a failed invariant there is a compiler bug:\n"
            "    CHECK(cond)                       -> INTERNAL_CHECK(cond)\n"
            "    CHECK(cond)      // op in scope   -> INTERNAL_CHECK_SPAN(cond, op->span_)\n"
            "    CHECK_SPAN(cond, span)            -> INTERNAL_CHECK_SPAN(cond, span)\n"
            "Prefer the _SPAN form whenever an IR node is in scope -- it attaches the IR source\n"
            "location. Keep the plain form when the predicate is itself the null guard for the\n"
            "pointer whose ->span_ you would read (see .claude/rules/error-checking.md).\n"
            "If a site is genuinely reachable by user input -- an unsupported dtype, a bad kwarg,\n"
            f"a documented limitation -- add its `<path>:<line>` to ALLOWLIST in {Path(__file__).name}\n"
            "with a comment saying how the user reaches it.",
            file=sys.stderr,
        )
        return 1

    print(f"All checks in {' and '.join(SCANNED_DIRS)} are classified correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
