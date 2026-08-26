# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that every ``IRProperty`` enumerator is declared in all four of its layers.

``enum class IRProperty`` is spelled out four times: the enum itself, the ``IRPropertyToString``
switch, the hand-written nanobind ``.value(...)`` chain, and the ``.pyi`` stub. Nothing in the build
links them -- the switch has a ``default:`` arm, nanobind never sees the stub, and pyright checks
code against the stub rather than the stub against the binding. So an enumerator added to the header
alone compiles, links, and imports; the omission only surfaces when a set that happens to contain
that property is enumerated from Python, where ``IRPropertySet.to_list()`` raises
``ValueError: <n> is not a valid IRProperty``. ``str(set)`` still renders the name correctly, so the
set looks healthy right up to that point.

This check enforces, against the header as the authoritative list:

1. *Coverage* -- each of the other three layers declares each of the header's enumerators exactly
   once, and nothing else. ``kCount`` is a sentinel, not a property, and is excluded everywhere.
2. *Order* -- each layer lists them in the header's declaration order. Order carries no runtime
   meaning (nanobind binds each ``.value`` to its C++ value, and stub members hold no value at all),
   but a layer that drifts out of order cannot be diffed against the header by position, which is
   what let a single missing entry hide among 40 present ones.
3. *Name agreement* -- ``IRPropertyToString`` returns each enumerator's own spelling, and every
   ``.value("Foo", IRProperty::Foo)`` pairs the Python name with the enumerator of the same name.
   A set's ``str()`` and its ``to_list()`` must agree on what a property is called.

Each layer is also sanity-checked for pattern drift: a declaration written in a form the regex here
does not match would be skipped in silence, which is the one failure this lint must not have, so an
unparsed entry is reported rather than ignored. C++ comments are blanked before scanning, so a
commented-out declaration cannot satisfy the check on behalf of the code it describes.

Usage:

    python tests/lint/check_ir_property_parity.py
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from _cpp_text import strip_cpp_comments

# The four declaration sites, relative to the repo root.
ENUM_DECL = "include/pypto/ir/transforms/ir_property.h"
TO_STRING_DECL = "src/ir/transforms/ir_property.cpp"
BINDING_DECL = "python/bindings/modules/passes.cpp"
STUB_DECL = "python/pypto/pypto_core/passes.pyi"

# Sentinel terminating the enum; it is not a property and is bound nowhere.
SENTINEL = "kCount"

# `enum class IRProperty : uint64_t { ... };` in ir_property.h.
ENUM_BLOCK_RE = re.compile(r"enum class IRProperty\s*:\s*\w+\s*\{(.*?)\n\};", re.DOTALL)
# One enumerator per line, with an optional explicit value: `SSAForm = 0,` / `TypeChecked,`.
ENUM_MEMBER_RE = re.compile(r"^\s*(\w+)\s*(?:=\s*[^,]+)?,?\s*$", re.MULTILINE)

# `std::string IRPropertyToString(IRProperty prop) { ... }` in ir_property.cpp.
TO_STRING_BLOCK_RE = re.compile(r"std::string IRPropertyToString\([^)]*\)\s*\{(.*?)\n\}", re.DOTALL)
TO_STRING_CASE_RE = re.compile(r'case IRProperty::(\w+):\s*return\s*"([^"]*)";')
TO_STRING_CASE_LABEL_RE = re.compile(r"case IRProperty::(\w+):")

# The `nb::enum_<IRProperty>(...)....value(...)...;` chain in passes.cpp. The chain contains no blank
# line, so the first one ends it -- more robust than looking for `;`, which also occurs inside the
# per-value docstrings.
BINDING_BLOCK_RE = re.compile(r"nb::enum_<IRProperty>\(.*?\n\n", re.DOTALL)
BINDING_VALUE_RE = re.compile(r'\.value\("(\w+)",\s*IRProperty::(\w+)')
BINDING_VALUE_ANY_RE = re.compile(r"\.value\(")

# `class IRProperty(Enum): ... Foo = ...` in passes.pyi.
STUB_BLOCK_RE = re.compile(r"^class IRProperty\(Enum\):\n(.*?)(?=^\S)", re.DOTALL | re.MULTILINE)
STUB_MEMBER_RE = re.compile(r"^    (\w+) = \.\.\.$", re.MULTILINE)
# Any indented binding in the class body, however spelled -- used to catch a member the strict
# pattern would skip. The docstring line is not an assignment and so does not match.
STUB_MEMBER_ANY_RE = re.compile(r"^    (\w+)\s*[:=]", re.MULTILINE)


def read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def read_cpp(root: Path, rel: str) -> str:
    """Read a C++ source with its comments blanked out.

    A declaration inside a comment is prose about the code, not the code. Counting one would let a
    commented-out `.value(...)` or `case` satisfy this check while the runtime enum stays short --
    the exact silent-success this lint exists to prevent.
    """
    return strip_cpp_comments(read(root, rel))


def parse_enum(root: Path) -> tuple[list[str], list[str]]:
    """Return (enumerators in declaration order, errors) from the header."""
    match = ENUM_BLOCK_RE.search(read_cpp(root, ENUM_DECL))
    if match is None:
        return [], [f"{ENUM_DECL}: could not locate `enum class IRProperty : <type> {{ ... }};`"]
    names = [m.group(1) for m in ENUM_MEMBER_RE.finditer(match.group(1))]
    errors: list[str] = []
    if SENTINEL not in names:
        errors.append(
            f"{ENUM_DECL}: `{SENTINEL}` sentinel not found -- the enum body did not parse as expected"
        )
    return [n for n in names if n != SENTINEL], errors


def parse_to_string(root: Path) -> tuple[list[str], list[str]]:
    """Return (enumerators handled by IRPropertyToString, errors), checking each returned spelling."""
    match = TO_STRING_BLOCK_RE.search(read_cpp(root, TO_STRING_DECL))
    if match is None:
        return [], [f"{TO_STRING_DECL}: could not locate the body of `IRPropertyToString`"]
    body = match.group(1)
    errors: list[str] = []
    names: list[str] = []
    for m in TO_STRING_CASE_RE.finditer(body):
        names.append(m.group(1))
        if m.group(1) != m.group(2):
            errors.append(
                f'{TO_STRING_DECL}: `case IRProperty::{m.group(1)}` returns "{m.group(2)}" -- '
                f"the string must be the enumerator's own name, since it is what `str(IRPropertySet)` prints"
            )
    for m in TO_STRING_CASE_LABEL_RE.finditer(body):
        if m.group(1) not in names:
            errors.append(
                f"{TO_STRING_DECL}: `case IRProperty::{m.group(1)}` is not followed by a plain "
                f'`return "...";`, so its spelling goes unchecked'
            )
    return names, errors


def parse_binding(root: Path) -> tuple[list[str], list[str]]:
    """Return (enumerators bound by nanobind, errors), checking each Python-visible name."""
    match = BINDING_BLOCK_RE.search(read_cpp(root, BINDING_DECL))
    if match is None:
        return [], [f"{BINDING_DECL}: could not locate the `nb::enum_<IRProperty>(...)` chain"]
    block = match.group(0)
    errors: list[str] = []
    names: list[str] = []
    for m in BINDING_VALUE_RE.finditer(block):
        names.append(m.group(2))
        if m.group(1) != m.group(2):
            errors.append(
                f'{BINDING_DECL}: `.value("{m.group(1)}", IRProperty::{m.group(2)}, ...)` exposes the '
                f"enumerator under a different name than `IRPropertyToString` prints"
            )
    bound = len(BINDING_VALUE_ANY_RE.findall(block))
    if bound != len(names):
        errors.append(
            f"{BINDING_DECL}: the chain has {bound} `.value(` call(s) but only {len(names)} parse as "
            f'`.value("Name", IRProperty::Name, ...)`, so some go unpoliced -- write them in that form'
        )
    return names, errors


def parse_stub(root: Path) -> tuple[list[str], list[str]]:
    """Return (enumerators declared in the .pyi stub, errors)."""
    match = STUB_BLOCK_RE.search(read(root, STUB_DECL))
    if match is None:
        return [], [f"{STUB_DECL}: could not locate `class IRProperty(Enum):`"]
    body = match.group(1)
    names = [m.group(1) for m in STUB_MEMBER_RE.finditer(body)]
    errors = [
        f"{STUB_DECL}: `{m.group(1)}` is declared in a form this check does not match, so it would go "
        f"unpoliced -- write it as `{m.group(1)} = ...`"
        for m in STUB_MEMBER_ANY_RE.finditer(body)
        if m.group(1) not in names
    ]
    return names, errors


def compare(layer: str, expected: list[str], actual: list[str], how: str) -> list[str]:
    """Report enumerators the layer is missing, ones it invents or repeats, and any order drift."""
    errors = [
        f"{layer}: `IRProperty::{name}` is declared in {ENUM_DECL} but not here -- add it as {how}"
        for name in expected
        if name not in actual
    ]
    errors += [
        f"{layer}: `{name}` is declared here but is not an `IRProperty` enumerator in {ENUM_DECL}"
        for name in actual
        if name not in expected
    ]
    errors += [
        f"{layer}: `{name}` is declared {count} times -- declare each enumerator exactly once"
        for name, count in sorted(Counter(actual).items())
        if count > 1
    ]
    # Past the checks above, `actual` holds each expected name exactly once and nothing else, so it
    # is a permutation of `expected` -- equal in length, and unequal only at some shared index.
    if not errors and actual != expected:
        first = next(i for i, (a, e) in enumerate(zip(actual, expected)) if a != e)
        errors.append(
            f"{layer}: declaration order diverges from {ENUM_DECL} at position {first} "
            f"(`{actual[first]}` here, `{expected[first]}` in the enum) -- keep the layers in the enum's "
            f"order so they stay diffable by position"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2], help="Repository root"
    )
    args = parser.parse_args()
    root: Path = args.root

    enum, errors = parse_enum(root)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    for layer, (names, layer_errors), how in (
        (TO_STRING_DECL, parse_to_string(root), '`case IRProperty::<Name>: return "<Name>";`'),
        (BINDING_DECL, parse_binding(root), '`.value("<Name>", IRProperty::<Name>, "<doc>")`'),
        (STUB_DECL, parse_stub(root), "`<Name> = ...`"),
    ):
        errors.extend(layer_errors)
        errors.extend(compare(layer, enum, names, how))

    if errors:
        print(
            f"Every IRProperty enumerator must be declared in all four layers.\n"
            f"An enumerator missing from {BINDING_DECL} or {STUB_DECL} still compiles and still prints "
            f"correctly from `str(IRPropertySet)`; it fails only when a set containing it reaches "
            f"`IRPropertySet.to_list()`.\n",
            file=sys.stderr,
        )
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"\n{len(errors)} problem(s).", file=sys.stderr)
        return 1

    print(f"OK: {len(enum)} IRProperty enumerator(s) declared consistently across all four layers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
