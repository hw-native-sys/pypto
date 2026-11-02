# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that a DSL op wrapper does not drop the explanation its IR twin carries.

Every operator is written twice in Python: ``pypto/ir/op/<ns>_ops.py`` takes raw ``ir.Expr`` with an
explicit ``span`` and returns an ``ir.Call``; ``pypto/language/op/<ns>_ops.py`` takes ``Tile`` /
``Tensor``, auto-captures the span, and returns a wrapped object. Only the second one is published:
``docs/en/user/api/{tile,tensor}.md`` render ``pypto.language.tile`` and ``pypto.language.tensor``
through mkdocstrings, so the DSL docstring *is* the user manual and the IR docstring is read by
compiler developers.

The failure mode this catches is that the two drift in the wrong direction. The add-op workflow asks
for a docstring on the IR wrapper but describes the DSL wrapper as ``unwrap -> call -> wrap``, so a
new op tends to get its shape rules, broadcasting semantics and dtype restrictions written on the
*internal* copy and only a one-line summary on the *published* one. Users then read the lossy
paraphrase while the real explanation sits in a module they never import.

The rule enforced here is structural, so it has no fuzzy similarity threshold and no false positives
from legitimate rewording:

    If the IR wrapper's docstring has a body -- a prose paragraph after the summary, or a section
    other than Args/Arguments/Returns/Yields (Note, Example, Warning, ...) -- then the DSL wrapper's
    docstring must have a body too.

One relocation is recognized rather than reported: when every IR body paragraph opens with a
parameter name (``core_type`` targets the operation to one lane ...), it is a note *about that
parameter*, and the published layer's convention is to carry it in the parameter's own ``Args``
entry. The check accepts that placement instead of demanding a redundant paragraph above ``Args``.

It deliberately does *not* compare the two bodies. The DSL copy is expected to differ: it drops the
``span`` argument and codegen internals, names ``pl.tile.foo`` where the IR copy names
``tile.foo``, and is frequently longer and better than its twin. What it must not do is say nothing
at all where the internal copy explains something.

Note that the fix is always a **source literal**. Assigning ``__doc__`` at import time from the
wrapped function does not work: ``docs/requirements.txt`` records that mkdocstrings reads
``python/pypto`` through griffe, which "parses the source statically -- nothing is imported", so a
runtime-assigned ``__doc__`` renders as an empty description on exactly the page being fixed.

Scope and deliberate limits:

* Only module-level functions defined in *both* layers of the same namespace are paired. Definitions
  in private `_<namespace>_*.py` implementation modules are included after a namespace is split.
  A helper that exists on one side only, or a name re-exported from another namespace, is not paired.
* A DSL wrapper with *no* docstring at all is reported, not skipped: it is the extreme of the same
  defect, since the published page then renders an empty description. Nothing else catches it --
  ``[tool.ruff.lint] select`` is ``PL, I, E, W, F, UP``, so pydocstyle (``D``) is not enabled. No op
  is in that state today; the check is here so the first one cannot land quietly.
* A pair whose *IR* docstring is missing is skipped -- there is then nothing to carry across, and
  "every internal wrapper is documented" is a different check.
* Overloads (``@typing.overload`` stubs) are ignored; only the implementation carries the docstring.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

# Namespaces written at both layers, as <ir module>/<language module> basenames.
NAMESPACES = ("tile", "tensor", "system", "array", "prefetch")

# Google-style section headers, matched only at the docstring's own indentation level.
SECTION_RE = re.compile(
    r"^(Args|Arguments|Returns|Yields|Raises|Note|Notes|Example|Examples|Warning|Warnings"
    r"|Attributes|See Also):\s*$"
)

# Sections that describe the signature rather than the operator. Content living only here does not
# count as a body: an op whose IR docstring has nothing but Args/Returns has nothing to drop.
SIGNATURE_SECTIONS = frozenset({"Args", "Arguments", "Returns", "Yields"})

# Pairs where the IR body is genuinely carried by the DSL docstring, just not as a body paragraph.
# Each entry maps "<namespace>.<op>" to (why, marker): `marker` is a substring the DSL docstring must
# still contain for the exemption to hold, so removing the explanation the exemption was granted for
# re-arms the check instead of passing silently. Add an entry only when the DSL copy really does
# state the same fact somewhere else -- not to silence an op whose explanation was dropped.
ALLOWLIST = {
    "system.import_peer_buffer": (
        "IR body states the INT32 result type; the DSL copy states it in Returns:.",
        "``pl.Scalar[pl.INT32]``",
    ),
    "system.reserve_buffer": (
        "IR body states the INT32 result type; the DSL copy states it in Returns:.",
        "``pl.Scalar[pl.INT32]``",
    ),
}


def split_docstring(doc: str) -> tuple[str, str, dict[str, str]]:
    """Split a cleaned Google-style docstring into (summary, prose, sections)."""
    lines = doc.split("\n")
    end_of_summary = 0
    while end_of_summary < len(lines) and lines[end_of_summary].strip():
        end_of_summary += 1
    summary = "\n".join(lines[:end_of_summary])

    prose: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    buf: list[str] = []
    for line in lines[end_of_summary:]:
        match = SECTION_RE.match(line.strip())
        # A header only opens a section at column 0 of the dedented docstring; an indented
        # "Note:" belongs to the enclosing block (an Args entry, say) and is body text.
        if match and not line[:1].isspace():
            if current is None:
                prose = buf
            else:
                sections[current] = buf
            current, buf = match.group(1), []
        else:
            buf.append(line)
    if current is None:
        prose = buf
    else:
        sections[current] = buf

    return summary, "\n".join(prose), {k: "\n".join(v) for k, v in sections.items()}


def has_body(doc: str) -> bool:
    """True when the docstring explains something beyond its summary and its signature sections."""
    _, prose, sections = split_docstring(doc)
    if prose.strip():
        return True
    return any(name not in SIGNATURE_SECTIONS and body.strip() for name, body in sections.items())


# A body paragraph that opens with a parameter name -- ``core_type`` targets the operation
# to one lane ... -- is a note *about that parameter*, and the DSL layer's convention is to
# carry it in the parameter's own Args entry rather than as a separate paragraph. Recognize
# that relocation so the check does not demand a redundant second copy above Args.
LEADING_PARAM_RE = re.compile(r"^\s*(?:``)?([a-z_][a-z0-9_]*)(?:``)?\b")


def parameter_names(node: FunctionNode) -> set[str]:
    """Every name the function binds as a parameter, positional or keyword-only."""
    a = node.args
    names = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    for extra in (a.vararg, a.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return names


def documented_parameters(doc: str) -> set[str]:
    """Parameter names that the docstring's Args section gives an entry to."""
    _, _, sections = split_docstring(doc)
    args = next((sections[name] for name in ("Args", "Arguments") if name in sections), "")
    documented = set()
    for line in args.split("\n"):
        # An entry starts at the section's own indent level; continuation lines are deeper.
        entry = re.match(r"^ {4}([a-z_][a-z0-9_]*)\s*(?:\([^)]*\))?\s*:", line)
        if entry:
            documented.add(entry.group(1))
    return documented


def body_is_parameter_notes(ir_doc: str, ir_node: FunctionNode, dsl_doc: str) -> bool:
    """True when every IR body paragraph is a note on a parameter the DSL documents in Args."""
    _, prose, sections = split_docstring(ir_doc)
    if any(name not in SIGNATURE_SECTIONS and body.strip() for name, body in sections.items()):
        return False  # a Note/Example section is not a parameter note
    paragraphs = [p for p in re.split(r"\n\s*\n", prose.strip()) if p.strip()]
    if not paragraphs:
        return False
    params = parameter_names(ir_node)
    dsl_documented = documented_parameters(dsl_doc)
    for paragraph in paragraphs:
        match = LEADING_PARAM_RE.match(paragraph)
        if not match or match.group(1) not in params:
            return False
        if match.group(1) not in dsl_documented:
            return False
    return True


def public_functions(path: Path) -> dict[str, tuple[FunctionNode, str | None]]:
    """Map public module-level function name -> (node, docstring or None), skipping overload stubs."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[FunctionNode, str | None]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        if any(
            (isinstance(d, ast.Name) and d.id == "overload")
            or (isinstance(d, ast.Attribute) and d.attr == "overload")
            for d in node.decorator_list
        ):
            continue
        out[node.name] = (node, ast.get_docstring(node, clean=True))
    return out


def namespace_functions(path: Path) -> dict[str, tuple[FunctionNode, str | None, Path]]:
    """Collect a namespace's definitions, retaining their source paths for diagnostics."""
    namespace = path.stem.removesuffix("_ops")
    # The public module may only re-export operators after a split. Scan its
    # private implementation siblings as well so moving a wrapper cannot
    # silently remove it from the IR/DSL documentation parity check.
    paths = [path, *sorted(path.parent.glob(f"_{namespace}_*.py"))]
    return {
        name: (node, doc, source)
        for source in paths
        for name, (node, doc) in public_functions(source).items()
    }


def find_violations(root_dir: Path) -> list[tuple[str, str, int, str, str]]:
    """Return (namespace, op, dsl lineno, dsl relative path, what went wrong) per violation."""
    violations = []
    for namespace in NAMESPACES:
        ir_path = root_dir / "python" / "pypto" / "ir" / "op" / f"{namespace}_ops.py"
        dsl_path = root_dir / "python" / "pypto" / "language" / "op" / f"{namespace}_ops.py"
        if not (ir_path.is_file() and dsl_path.is_file()):
            continue
        ir_funcs = namespace_functions(ir_path)
        dsl_funcs = namespace_functions(dsl_path)
        for name in sorted(ir_funcs.keys() & dsl_funcs.keys()):
            ir_node, ir_doc, _ = ir_funcs[name]
            dsl_node, dsl_doc, source = dsl_funcs[name]
            rel = source.relative_to(root_dir).as_posix()
            if ir_doc is None:
                continue  # nothing to carry across

            # The extreme of the same defect: the published wrapper has no description at all,
            # so the API page renders empty. Nothing else catches this -- pydocstyle is not
            # among the ruff rules this repo selects.
            if dsl_doc is None:
                violations.append((namespace, name, dsl_node.lineno, rel, "has no docstring at all"))
                continue

            exemption = ALLOWLIST.get(f"{namespace}.{name}")
            if exemption is not None:
                marker = exemption[1]
                if marker in dsl_doc:
                    continue
                violations.append(
                    (
                        namespace,
                        name,
                        dsl_node.lineno,
                        rel,
                        f"is allowlisted, but its docstring no longer contains {marker}",
                    )
                )
                continue

            if not has_body(ir_doc) or has_body(dsl_doc):
                continue
            if body_is_parameter_notes(ir_doc, ir_node, dsl_doc):
                continue
            violations.append(
                (namespace, name, dsl_node.lineno, rel, "drops the explanation its IR twin carries")
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n")[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (default: inferred from this file's location)",
    )
    args = parser.parse_args()
    root_dir = args.root.resolve()

    violations = find_violations(root_dir)
    for namespace, name, lineno, rel, problem in violations:
        print(f"{rel}:{lineno}: pl.{namespace}.{name} {problem}")

    if violations:
        print(
            f"\nFound {len(violations)} op wrapper(s) whose published docstring is worse than its twin.\n"
            "The DSL wrapper in python/pypto/language/op/ is what users read -- it is rendered into\n"
            "docs/en/user/api/ -- while python/pypto/ir/op/ is internal. When the internal copy\n"
            "explains a shape rule, a broadcasting rule or a dtype restriction, the published copy\n"
            "must explain it too.\n\n"
            "Write the explanation as a source literal in the DSL docstring, adapted to the DSL\n"
            "surface (no `span`, no codegen internals, `pl.tile.foo` rather than `tile.foo`).\n"
            "Do NOT assign __doc__ from the IR wrapper at import time: mkdocstrings reads the\n"
            "source statically through griffe, so a runtime-assigned __doc__ renders empty.\n\n"
            "If the DSL copy genuinely states the same fact elsewhere (in Returns:, say), add\n"
            f'"<namespace>.<op>" to ALLOWLIST in {Path(__file__).name} as (reason, marker), where the'
            " marker is a substring the DSL docstring must keep for the exemption to hold.",
            file=sys.stderr,
        )
        return 1

    print("Every DSL op wrapper keeps the explanation its IR twin carries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
