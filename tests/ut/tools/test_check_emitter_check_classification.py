# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the emitter check-classification lint.

The lint guards an invariant no runtime test can reach: ``src/codegen`` and ``src/backend`` run
after verification, so their argument-count and "Internal error" checks must be ``INTERNAL_CHECK*``
rather than ``CHECK``. These tests pin the scanner's behaviour on the C++ shapes that a naive
line-window or regex implementation gets wrong.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_lint() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "lint" / "check_emitter_check_classification.py"
    spec = importlib.util.spec_from_file_location("pypto_check_emitter_check_classification", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lint = _load_lint()


def _rules(tmp_path: Path, source: str) -> list[str]:
    """Run the scanner over *source* and return the rule letter of each violation."""
    path = tmp_path / "emitter.cpp"
    path.write_text(source + "\n")
    return [rule for _, rule, _ in lint.find_violations(path)]


@pytest.mark.parametrize(
    "source, expected",
    [
        # Rule B -- a CHECK on a call's argument count.
        ('CHECK(op->args_.size() == 2) << "x";', ["B"]),
        ('CHECK(op->args_.empty()) << "x";', ["B"]),
        ('CHECK_SPAN(op->args_.size() == 2, op->span_) << "x";', ["B"]),
        # An arity value bound to a local alias is still an arity check.
        ('const size_t arity = op->args_.size();\nCHECK(arity == 2) << "x";', ["B"]),
        # Rule A -- the macro throws ValueError while the message claims a compiler bug.
        ('CHECK(foo) << "Internal error: bad";', ["A"]),
        # The INTERNAL_ forms are the fix, never the violation.
        ('INTERNAL_CHECK(op->args_.size() == 2) << "x";', []),
        ('INTERNAL_CHECK_SPAN(op->args_.size() == 2, op->span_) << "x";', []),
        ('INTERNAL_CHECK(foo) << "Internal error: bad";', []),
        # A genuine user-facing check is left alone.
        ('CHECK(dtype.IsFloat()) << "not supported on this backend; use FP32";', []),
    ],
)
def test_rule_classification(tmp_path: Path, source: str, expected: list[str]) -> None:
    assert _rules(tmp_path, source) == expected


def test_adjacent_internal_check_does_not_bleed_into_a_clean_check(tmp_path: Path) -> None:
    """The scanner must read one whole statement, not a fixed window of lines.

    ``pto_ops_crosscore.cpp`` and ``scope_outline_utils.h`` both put an ``INTERNAL_CHECK_SPAN``
    whose message legitimately says "Internal error" directly below a ``CHECK``. A line-window
    implementation reports the clean ``CHECK`` for its neighbour's message.
    """
    source = 'CHECK(foo) << "fine";\nINTERNAL_CHECK_SPAN(bar, op->span_)\n    << "Internal error: nope";'
    assert _rules(tmp_path, source) == []


@pytest.mark.parametrize(
    "source",
    [
        '// Internal error: prose about a CHECK(op->args_.size() == 1)\nCHECK(foo) << "fine";',
        '/* CHECK(op->args_.size() == 1) << "Internal error"; */\nCHECK(foo) << "fine";',
        'const char* s = "CHECK(op->args_.size() == 1)";\nCHECK(foo) << "fine";',
    ],
)
def test_comments_and_string_literals_are_not_scanned(tmp_path: Path, source: str) -> None:
    assert _rules(tmp_path, source) == []


@pytest.mark.parametrize(
    "source",
    [
        # A quote inside a char literal must not open a string and swallow the rest of the file.
        'char c = \'"\';\nCHECK(op->args_.size() == 1) << "x";',
        # An escaped quote must not close the message early.
        'CHECK(foo) << "a \\" b";\nCHECK(op->args_.size() == 3) << "y";',
    ],
)
def test_quoting_edge_cases_do_not_desync_the_scanner(tmp_path: Path, source: str) -> None:
    assert _rules(tmp_path, source) == ["B"]


@pytest.mark.parametrize(
    "source, expected",
    [
        # The reported case: emitter code builds target syntax from raw literals whose body
        # carries unpaired quotes. Scanning one as an ordinary literal desynchronizes the
        # masker and silently hides every check after it.
        ('os << R"(", dtype="opaque", count=)" << n;\nCHECK(op->args_.size() == 1) << "x";', ["B"]),
        # The mirror failure: arity text inside a raw literal is data, not code.
        ('const char* s = R"(CHECK(op->args_.size() == 1))";', []),
        # A custom delimiter must terminate on its own )delim" and nothing shorter.
        (
            'auto s = R"d(" CHECK(op->args_.size()==2) )d";\nCHECK(op->args_.size() == 3) << "y";',
            ["B"],
        ),
        # Encoding-prefixed raw literals are raw too.
        (
            'auto a = u8R"(")"; auto b = uR"(")";\n'
            'auto c = UR"(")"; auto d = LR"(")";\n'
            'CHECK(op->args_.size() == 1) << "z";',
            ["B"],
        ),
        # An identifier merely ending in R does not open a raw literal.
        ('MYR"abc";\nCHECK(op->args_.size() == 1) << "w";', ["B"]),
        # A raw-string message still carries its text to rule A.
        ('CHECK(foo) << R"(Internal error: bad)";', ["A"]),
        # A malformed literal must not hang or crash the scan.
        ('auto s = R"(never closed\nCHECK(op->args_.size() == 1);', []),
    ],
)
def test_raw_string_literals_are_masked(tmp_path: Path, source: str, expected: list[str]) -> None:
    """C++ raw literals may contain unpaired quotes; the masker must skip their bodies whole."""
    assert _rules(tmp_path, source) == expected


def test_repository_is_clean() -> None:
    """The emitter trees must stay classified correctly; this is the gate itself."""
    root = Path(__file__).resolve().parents[3]
    offenders = [
        f"{path.relative_to(root).as_posix()}:{line}: [rule {rule}] {detail}"
        for path in lint.get_git_tracked_sources(root)
        for line, rule, detail in lint.find_violations(path)
        if f"{path.relative_to(root).as_posix()}:{line}" not in lint.ALLOWLIST
    ]
    assert offenders == [], "Mis-classified checks in codegen/backend:\n" + "\n".join(offenders)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
