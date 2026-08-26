# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that the three ``IRPropertySet`` getters are spelled the same everywhere.

``GetVerifiedProperties()``, ``GetStructuralProperties()`` and ``GetDefaultVerifyProperties()`` each
have a C++ initializer and three prose copies of it: the ``Returns {...}`` clause on the declaration,
and a summary row in the English and Chinese verifier docs. Nothing in the build links them, and no
test reads either copy, so a property added to the initializer alone compiles and passes CI while
every list a developer actually reads stays one entry short.

That is not hypothetical: ``DistTensorCtxMaterialized`` was absent from all three copies of
``GetVerifiedProperties()``, which named 17 properties for a set that returns 18. The omitted entry
was the distributed-only property -- the least likely of the 18 to appear in a reader's own test, and
so the least likely to be noticed. The lists are consulted precisely when someone needs to know what
``PassPipeline`` auto-verifies (e.g. whether a new pass must preserve a property), which is when
being wrong by one costs the most.

This check enforces, against the initializers in ``ir_property.cpp`` as the authoritative lists:

1. *Coverage* -- each copy names each initializer's properties exactly once, and nothing else.
2. *Order* -- each copy lists them in the initializer's order, so a copy can be diffed against the
   C++ by position rather than by set membership.
3. *Completeness* -- every getter with an initializer has a ``Returns {...}`` clause and a row in
   both docs, and no site states members for a getter the C++ no longer defines, so neither adding a
   set nor removing one can leave a copy behind in one language and not the other.
4. *Uniqueness* -- no site states a getter's members twice. Two copies can disagree, and only one of
   them would be compared against the C++.

Each site is also sanity-checked for pattern drift: a copy written in a form the regexes here do not
match would be skipped in silence, which is the one failure this lint must not have, so an unparsed
site is reported rather than ignored.

Usage:

    python tests/lint/check_property_set_doc_parity.py
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# The initializers (authoritative) and the three sites that restate them, relative to the repo root.
IMPL_DECL = "src/ir/transforms/ir_property.cpp"
HEADER_DECL = "include/pypto/ir/transforms/ir_property.h"
DOC_DECLS = ("docs/en/dev/passes/99-verifier.md", "docs/zh/dev/passes/99-verifier.md")

# `const IRPropertySet& GetFooProperties() { static const IRPropertySet props{...}; return props; }`
IMPL_RE = re.compile(
    r"const IRPropertySet& (\w+)\(\)\s*\{\s*"
    r"(?:/[/*].*?\n\s*)*"  # optional leading comment lines inside the body
    r"static const IRPropertySet \w+\{(.*?)\};",
    re.DOTALL,
)
# Any definition of a getter returning the set, used to catch one the strict pattern would skip.
IMPL_ANY_RE = re.compile(r"const IRPropertySet& (\w+)\(\)\s*\{")

# A doxygen block immediately followed by `const IRPropertySet& GetFooProperties();` in the header.
HEADER_RE = re.compile(r"/\*\*(.*?)\*/\s*const IRPropertySet& (\w+)\(\);", re.DOTALL)
# Any declaration of such a getter, used to catch one with no doxygen block at all.
HEADER_ANY_RE = re.compile(r"^const IRPropertySet& (\w+)\(\);", re.MULTILINE)

# `| `GetFooProperties()` | `{A, B, C}` | description |` in the docs' summary table. Requiring the
# `{...}` cell is specific enough on its own -- it matches the three property-set rows and nothing
# else in either doc -- so this is deliberately NOT narrowed to the getters the C++ defines: a row
# left behind for a deleted getter has to keep matching in order to be reported as stale.
DOC_ROW_RE = re.compile(r"^\|\s*`(\w+)\(\)`\s*\|\s*`\{([^}]*)\}`\s*\|", re.MULTILINE)
# Any row keyed on such a getter, used to catch one whose set cell is written some other way. Both
# doc files are full of tables keyed on a `method()` cell, so this one IS matched only against the
# getters the C++ defines -- everything else is an unrelated API table, however it is written.
DOC_ROW_ANY_RE = re.compile(r"^\|\s*`(\w+)\(\)`\s*\|", re.MULTILINE)

# The `{...}` payload of a `Returns {...}` clause, which wraps across ` * `-prefixed comment lines.
RETURNS_RE = re.compile(r"Returns\s*\{([^}]*)\}")


def read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def split_names(payload: str) -> list[str]:
    """Split a `{A, B, C}` payload into property names, folding any line wrapping."""
    flat = re.sub(r"\s*\n\s*\*?\s*", " ", payload)
    return [name.strip() for name in flat.split(",") if name.strip()]


def parse_impl(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Return (getter -> properties in initializer order, errors) from the C++ implementation."""
    text = read(root, IMPL_DECL)
    sets = {m.group(1): re.findall(r"IRProperty::(\w+)", m.group(2)) for m in IMPL_RE.finditer(text)}
    errors = [
        f"{IMPL_DECL}: `{m.group(1)}()` does not parse as `static const IRPropertySet <name>{{...}};`, "
        f"so the lists that restate it go unpoliced"
        for m in IMPL_ANY_RE.finditer(text)
        if m.group(1) not in sets
    ]
    if not sets:
        errors.append(f"{IMPL_DECL}: found no `const IRPropertySet& <name>()` definitions at all")
    return sets, errors


def parse_header(root: Path, known: set[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Return (getter -> properties named by its `Returns {...}` clause, errors) from the header."""
    text = read(root, HEADER_DECL)
    parsed: dict[str, list[str]] = {}
    errors: list[str] = []
    for m in HEADER_RE.finditer(text):
        if m.group(2) in parsed:
            errors.append(
                f"{HEADER_DECL}: `{m.group(2)}()` is declared more than once with a `Returns {{...}}` "
                f"clause -- the copies can disagree and only one is compared, so keep exactly one"
            )
            continue
        returns = RETURNS_RE.search(m.group(1))
        if returns is None:
            errors.append(
                f"{HEADER_DECL}: the doc comment on `{m.group(2)}()` has no `Returns {{...}}` clause "
                f"naming the set's members"
            )
            continue
        parsed[m.group(2)] = split_names(returns.group(1))
    errors += [
        f"{HEADER_DECL}: `{m.group(1)}()` is declared with no preceding doxygen block, so its members "
        f"are documented nowhere"
        for m in HEADER_ANY_RE.finditer(text)
        if m.group(1) in known and m.group(1) not in parsed and not any(m.group(1) in e for e in errors)
    ]
    return parsed, errors


def parse_doc(root: Path, rel: str, known: set[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Return (getter -> properties named by its summary row, errors) from one verifier doc.

    Rows are collected without reference to `known` so that a row surviving a getter's deletion is
    still returned, and so reaches the caller's "documented but has no initializer" check. Only the
    malformed-row sweep is scoped to `known`, because its pattern alone cannot tell a property-set
    row from the several unrelated API tables these docs key on a `method()` cell.
    """
    text = read(root, rel)
    parsed: dict[str, list[str]] = {}
    errors: list[str] = []
    for m in DOC_ROW_RE.finditer(text):
        if m.group(1) in parsed:
            errors.append(
                f"{rel}: `{m.group(1)}()` has more than one summary row -- the rows can disagree and "
                f"only one is compared, so keep exactly one"
            )
            continue
        parsed[m.group(1)] = split_names(m.group(2))
    errors += [
        f"{rel}: the row for `{m.group(1)}()` does not spell its members as a `` `{{A, B, C}}` `` cell, "
        f"so it goes unpoliced"
        for m in DOC_ROW_ANY_RE.finditer(text)
        if m.group(1) in known and m.group(1) not in parsed
    ]
    return parsed, errors


def compare(site: str, getter: str, expected: list[str], actual: list[str]) -> list[str]:
    """Report members the copy is missing, ones it invents or repeats, and any order drift."""
    errors = [
        f"{site}: `{getter}()` returns `{name}` but the list here omits it "
        f"({len(actual)} named, {len(expected)} returned)"
        for name in expected
        if name not in actual
    ]
    errors += [
        f"{site}: `{getter}()` is listed here as containing `{name}`, which is not in the "
        f"{IMPL_DECL} initializer"
        for name in actual
        if name not in expected
    ]
    errors += [
        f"{site}: `{getter}()` lists `{name}` {count} times -- name each member exactly once"
        for name, count in sorted(Counter(actual).items())
        if count > 1
    ]
    # Past the checks above, `actual` holds each expected name exactly once and nothing else, so it
    # is a permutation of `expected` -- equal in length, and unequal only at some shared index.
    if not errors and actual != expected:
        first = next(i for i, (a, e) in enumerate(zip(actual, expected)) if a != e)
        errors.append(
            f"{site}: `{getter}()` diverges from the {IMPL_DECL} initializer at position {first} "
            f"(`{actual[first]}` here, `{expected[first]}` in C++) -- keep the order so the lists stay "
            f"diffable by position"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2], help="Repository root"
    )
    args = parser.parse_args()
    root: Path = args.root

    impl, errors = parse_impl(root)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    known = set(impl)
    sites = [(HEADER_DECL, parse_header(root, known))]
    sites += [(rel, parse_doc(root, rel, known)) for rel in DOC_DECLS]
    for site, (parsed, site_errors) in sites:
        errors.extend(site_errors)
        for getter, expected in impl.items():
            if getter not in parsed:
                errors.append(
                    f"{site}: `{getter}()` is defined in {IMPL_DECL} but its members are not listed "
                    f"here -- add them, in the initializer's order"
                )
                continue
            errors.extend(compare(site, getter, expected, parsed[getter]))
        errors.extend(
            f"{site}: `{getter}()` is documented here but has no initializer in {IMPL_DECL}"
            for getter in parsed
            if getter not in impl
        )

    if errors:
        print(
            f"Every IRPropertySet getter must name the same members in {IMPL_DECL}, in its doc comment, "
            f"and in both verifier docs.\n"
            f"A property added to the initializer alone compiles and passes CI, so the drift shows up "
            f"only as a list a developer reads and trusts while it is short by one.\n",
            file=sys.stderr,
        )
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"\n{len(errors)} problem(s).", file=sys.stderr)
        return 1

    total = sum(len(v) for v in impl.values())
    print(
        f"OK: {len(impl)} IRPropertySet getter(s), {total} membership(s), listed consistently across "
        f"the initializer, the header comment, and both verifier docs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
