# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MLIR source locations in generated .pto.

PTO codegen suffixes every emitted operation with ``loc("file":line:col)`` taken
from the IR ``Span``, so a ptoas verifier rejection names the user's source line
instead of a line in the generated artifact (which, under ``@pl.jit``, the user
never sees).

Two properties matter and are both easy to regress:

1. **Structural lines take no location.** MLIR's trailing ``loc(...)`` is legal
   only at the end of a complete operation, so region braces and block labels
   must not carry one -- otherwise the module stops parsing.
2. **The location names the statement, not the ``def`` line.** Passes that
   synthesize tile ops stamp the enclosing *function*'s span on the ``Call`` they
   build while preserving the ``AssignStmt``'s own span, so codegen must prefer
   the statement span unless the Call's span is genuinely nested inside it.
"""

import importlib.util
import os
import re
import sys

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.backend.pto_backend import emit_source_loc_default
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

_THIS_FILE = os.path.abspath(__file__)

with open(_THIS_FILE) as _source:
    _SOURCE_LINES = _source.read().splitlines()

# `loc("<path>":<line>:<col>)` at end of line; the path is greedy-free so an
# escaped quote inside it would fail the match rather than silently pass.
_LOC_RE = re.compile(r' loc\("((?:[^"\\]|\\.)*)":(\d+):(\d+)\)$')


def _generate(program_cls, *, emit_source_loc: bool = True) -> str:
    """Run the Default pipeline and PTO codegen over the first function."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
    funcs = list(optimized.functions.values())
    assert funcs, "Program has no functions"
    single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
    return codegen.PTOCodegen().generate(single, emit_source_loc=emit_source_loc)


def _is_structural(line: str) -> bool:
    """True for region braces and block labels -- lines that are not operations."""
    stripped = line.strip()
    return stripped.endswith("{") or stripped in ("}", "} else {", "} do {") or stripped.startswith("^bb")


def _is_operation(line: str) -> bool:
    """True for emitted MLIR operations that should carry a location.

    Excludes the constants section: ``arith.constant`` values are deduplicated
    across every use, so no single span is the right one for them.
    """
    stripped = line.strip()
    if not stripped or _is_structural(stripped) or stripped == "return":
        return False
    if "arith.constant" in stripped:
        return False
    return "pto." in stripped or "arith." in stripped


@pl.program
class ElementwiseProg:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_elementwise(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        out: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        ta: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        tb: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        ts: pl.Tile[[128, 128], pl.FP32] = pl.add(ta, tb)
        te: pl.Tile[[128, 128], pl.FP32] = pl.exp(ts)
        updated: pl.Tensor[[128, 128], pl.FP32] = pl.store(te, [0, 0], out)
        return updated


# Source lines of the statements above, resolved once so the assertions stay
# correct if the classes move within this file.
_ADD_LINE = next(i for i, ln in enumerate(_SOURCE_LINES, 1) if "pl.add(ta, tb)" in ln)
_EXP_LINE = next(i for i, ln in enumerate(_SOURCE_LINES, 1) if "pl.exp(ts)" in ln)


@pl.program
class MultiLineProg:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_multiline(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        out: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        ta: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        tb: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        # MULTILINE_ADD_START
        ts: pl.Tile[[128, 128], pl.FP32] = pl.add(
            ta,
            tb,
        )
        # MULTILINE_ADD_END
        return pl.store(ts, [0, 0], out)


# Inclusive source range of the multi-line `pl.add` statement above, resolved
# from the markers so the assertion survives edits elsewhere in this file.
_MULTILINE_ADD_RANGE = (
    next(i for i, ln in enumerate(_SOURCE_LINES, 1) if "# MULTILINE_ADD_START" in ln) + 1,
    next(i for i, ln in enumerate(_SOURCE_LINES, 1) if "# MULTILINE_ADD_END" in ln) - 1,
)


@pl.program
class ControlFlowProg:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_loop(
        self,
        a: pl.Tensor[[256, 128], pl.FP32],
        out: pl.Tensor[[256, 128], pl.FP32],
    ) -> pl.Tensor[[256, 128], pl.FP32]:
        acc: pl.Tensor[[256, 128], pl.FP32] = out
        for i in pl.range(2):
            ta: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [i * 128, 0], [128, 128])
            te: pl.Tile[[128, 128], pl.FP32] = pl.exp(ta)
            acc = pl.store(te, [i * 128, 0], acc)
        return acc


class TestSourceLocEmission:
    """Every emitted operation carries a location naming this test file."""

    def test_every_operation_carries_a_loc(self):
        mlir = _generate(ElementwiseProg)
        ops = [ln for ln in mlir.splitlines() if _is_operation(ln)]
        assert ops, f"no operations found in:\n{mlir}"
        missing = [ln.strip() for ln in ops if not _LOC_RE.search(ln)]
        assert not missing, f"operations without loc(): {missing}"

    def test_loc_names_this_source_file(self):
        mlir = _generate(ElementwiseProg)
        for line in mlir.splitlines():
            match = _LOC_RE.search(line)
            if match is None:
                continue
            assert match.group(1) == _THIS_FILE, f"unexpected loc file in: {line.strip()}"
            assert int(match.group(2)) > 0
            assert int(match.group(3)) > 0

    def test_loc_names_the_statement_not_the_def_line(self):
        """Regression guard for the statement-span fallback.

        ``ConvertTensorToTileOps`` rebuilds these tile ops carrying the enclosing
        function's span. Reading ``Call::span_`` alone would report the ``def``
        line for every one of them; codegen must fall back to the statement span.
        """
        mlir = _generate(ElementwiseProg)

        def loc_line_of(op_name: str) -> int:
            line = next(ln for ln in mlir.splitlines() if op_name in ln)
            match = _LOC_RE.search(line)
            assert match is not None, f"{op_name} carries no loc: {line.strip()}"
            return int(match.group(2))

        assert loc_line_of("pto.tadd") == _ADD_LINE
        assert loc_line_of("pto.texp") == _EXP_LINE
        # Distinct statements must not collapse onto one line.
        assert loc_line_of("pto.tadd") != loc_line_of("pto.texp")

    def test_structural_lines_carry_no_loc(self):
        """A brace or block label with a trailing loc(...) is a parse error."""
        mlir = _generate(ControlFlowProg)
        structural = [ln for ln in mlir.splitlines() if _is_structural(ln)]
        assert any("scf.for" in ln for ln in structural), f"no scf.for region in:\n{mlir}"
        offenders = [ln.strip() for ln in structural if "loc(" in ln]
        assert not offenders, f"structural lines must not carry loc(): {offenders}"

    def test_constants_carry_no_loc(self):
        """arith.constant is deduplicated across uses, so no span fits it."""
        mlir = _generate(ElementwiseProg)
        constants = [ln for ln in mlir.splitlines() if "arith.constant" in ln]
        assert constants, f"no constants section in:\n{mlir}"
        assert not [ln.strip() for ln in constants if "loc(" in ln]


class TestSourceLocDisabled:
    """emit_source_loc=False must reproduce the pre-location output exactly."""

    @pytest.mark.parametrize("program_cls", [ElementwiseProg, ControlFlowProg])
    def test_disabled_output_is_the_loc_stripped_output(self, program_cls):
        with_loc = _generate(program_cls, emit_source_loc=True)
        without_loc = _generate(program_cls, emit_source_loc=False)
        assert "loc(" not in without_loc
        stripped = "\n".join(_LOC_RE.sub("", ln) for ln in with_loc.splitlines())
        assert stripped == without_loc.rstrip("\n")


class TestUnknownSpan:
    """A node with no usable span emits no location rather than a bogus one."""

    def test_unknown_span_emits_no_loc(self):
        span = ir.Span.unknown()
        assert not span.is_valid() or not span.filename

    def test_synthesized_ops_do_not_invent_a_location(self):
        """Ops whose span is unknown are simply left unannotated."""
        mlir = _generate(ElementwiseProg)
        # A location is never emitted with an empty filename or a non-positive line.
        assert 'loc("":' not in mlir
        assert not re.search(r"loc\(\"[^\"]*\":-?0*:", mlir)
        assert ":-1:" not in mlir


class TestMultiLineStatements:
    """A statement spanning several lines still resolves to one location."""

    def test_multi_line_call_reports_the_statement_start_line(self):
        """Locations stay within the statement that produced them.

        A Call span is trusted only when it is nested inside the enclosing
        statement's span — both ends, not just the start. A multi-line statement
        is where those ends genuinely differ, so every operation it lowers to must
        report a line inside the statement's own range.
        """
        mlir = _generate(MultiLineProg)
        add_line = next(ln for ln in mlir.splitlines() if "pto.tadd" in ln)
        match = _LOC_RE.search(add_line)
        assert match is not None, f"pto.tadd carries no loc: {add_line.strip()}"
        reported = int(match.group(2))
        start, end = _MULTILINE_ADD_RANGE
        assert start <= reported <= end, (
            f"pto.tadd reports line {reported}, outside its statement range {start}-{end}"
        )


class TestPathEscaping:
    """A source path is an arbitrary OS string; unescaped bytes break parsing."""

    def test_special_characters_in_source_path_are_escaped(self, tmp_path):
        """A quote, a backslash and a control character must all survive escaping.

        MLIR rejects an unescaped quote or control character inside a string
        literal, so any of them would make the whole module unparseable. POSIX
        allows every byte except `/` and NUL in a path, so all three are reachable.
        """
        try:
            src_dir = tmp_path / 'q"d\\ir\ttab'
            src_dir.mkdir()
        except OSError:  # filesystem rejects these bytes in names (e.g. Windows)
            pytest.skip("filesystem does not allow quotes or control characters in path names")

        module_path = src_dir / "kern.py"
        module_path.write_text(
            "import pypto.language as pl\n"
            "\n"
            "@pl.program\n"
            "class P:\n"
            "    @pl.function(type=pl.FunctionType.InCore)\n"
            "    def k(self, a: pl.Tensor[[128, 128], pl.FP32],\n"
            "          o: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
            "        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])\n"
            "        e: pl.Tile[[128, 128], pl.FP32] = pl.exp(t)\n"
            "        u: pl.Tensor[[128, 128], pl.FP32] = pl.store(e, [0, 0], o)\n"
            "        return u\n"
        )

        spec = importlib.util.spec_from_file_location("pypto_loc_escape_case", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            mlir = _generate(module.P)
        finally:
            sys.modules.pop(spec.name, None)

        texp_line = next(ln for ln in mlir.splitlines() if "pto.texp" in ln)
        assert '\\"d' in texp_line, f"quote left unescaped in: {texp_line.strip()!r}"
        assert "\\\\ir" in texp_line, f"backslash left unescaped in: {texp_line.strip()!r}"
        assert "\\t" in texp_line, f"tab left unescaped in: {texp_line.strip()!r}"
        # No raw control character reached the literal — that alone would make the
        # module unparseable regardless of the visible escapes above.
        assert "\t" not in texp_line, f"raw tab emitted in: {texp_line.strip()!r}"
        # The escaped form still parses back out as a single location, and
        # unescaping it recovers the exact path on disk.
        match = _LOC_RE.search(texp_line)
        assert match is not None
        unescaped = match.group(1).replace('\\"', '"').replace("\\t", "\t").replace("\\\\", "\\")
        assert unescaped == str(module_path)


class TestEnvironmentDefault:
    """PYPTO_EMIT_PTO_LOC is the no-rebuild kill switch for an ptoas that chokes."""

    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            (None, True),
            ("1", True),
            ("true", True),
            ("0", False),
            ("false", False),
            ("False", False),
            ("off", False),
            ("no", False),
        ],
    )
    def test_env_var_controls_the_default(self, monkeypatch, env_value, expected):
        if env_value is None:
            monkeypatch.delenv("PYPTO_EMIT_PTO_LOC", raising=False)
        else:
            monkeypatch.setenv("PYPTO_EMIT_PTO_LOC", env_value)
        assert emit_source_loc_default() is expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
