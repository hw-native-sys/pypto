# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen behaviour of the declared GM cache-access policy (pypto #2534).

PTOAS has no L2-bypass path yet
(https://github.com/hw-native-sys/PTOAS/issues/1356), so a ``CachePolicy.BYPASS``
declaration is carried all the way to codegen but changes nothing it emits. That
makes the codegen contract two-sided, and both sides are asserted here:

1. **The generated MLIR is byte-identical** with and without the declaration.
   Anything else would mean the "compiles as an ordinary cached access" promise
   is already broken.
2. **The user is told.** The declaration is diagnosed once per tensor per
   kernel — not once per emitted load — and the message points at the PTOAS
   issue so the reader can see when the request will start to mean something.

The warning travels the C++ ``LOG_WARN`` channel, which writes to ``std::cerr``
from native code, so it is read with pytest's ``capfd`` (file-descriptor level)
rather than ``capsys`` — the same mechanism ``tests/ut/core/test_logging.py``
and the MemoryReuse fallback diagnostics use.
"""

import pypto.language as pl
import pytest
from _pto_loc_common import strip_loc
from pypto import LogLevel, backend, codegen, ir, set_log_level
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager

# The exact link the message must carry: it is the only thing in the warning
# that tells a reader when BYPASS stops being a no-op.
PTOAS_ISSUE_URL = "https://github.com/hw-native-sys/PTOAS/issues/1356"
# Diagnostic tag, used to pick this warning out of unrelated pipeline output.
BYPASS_WARNING_TAG = "[CacheBypassUnsupported]"

M, K, N = 256, 128, 256
ROWS, COLS = 32, 128


@pytest.fixture(autouse=True)
def _setup_backend_and_log_level():
    """Pin the backend and make WARN-level output visible for every test here.

    ``conftest.py`` restores the process-global log level after each test, so
    raising it to WARN here is contained; setting it explicitly keeps the
    stderr assertions independent of ``PYPTO_LOG_LEVEL`` and of test order.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    set_log_level(LogLevel.WARN)
    yield
    backend.reset_for_testing()


# ---------------------------------------------------------------------------
# Programs
#
# `DeclaredBypass` and `PlainMatmul` are the SAME kernel; the declaration line
# is the only difference between them, which is what makes the byte-identity
# comparison meaningful.
# ---------------------------------------------------------------------------


@pl.program
class DeclaredBypass:
    """Scope-level declaration on one of the two matmul operands."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
            c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out


@pl.program
class PlainMatmul:
    """The same kernel with no declaration — the byte-identity reference."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out


@pl.program
class TwoDeclaredTensors:
    """Both operands declared: the diagnostic is per tensor, so two warnings."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm2"):
            pl.set_cache_policy(a, pl.CachePolicy.BYPASS)
            pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
            c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out


@pl.program
class TwoBypassingLoadsOfOneTensor:
    """Two `pl.load(..., cache=BYPASS)` reads of ONE tensor — still one warning."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[ROWS, COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        top: pl.Tile[[16, COLS], pl.FP32] = pl.load(x, [0, 0], [16, COLS], cache=pl.CachePolicy.BYPASS)
        bottom: pl.Tile[[16, COLS], pl.FP32] = pl.load(x, [16, 0], [16, COLS], cache=pl.CachePolicy.BYPASS)
        out_0: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(top, [0, 0], out)
        out_1: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(bottom, [16, 0], out_0)
        return out_1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _incore_mlir(program_cls, *, emit_source_loc: bool = False) -> str:
    """Run the Default pipeline and emit PTO MLIR for the single in-core kernel.

    ``PTOCodegen.generate`` only accepts in-core functions, so the Orchestration
    parent left behind by ``pl.at`` outlining is dropped first.

    Args:
        program_cls: The ``@pl.program`` class to compile.
        emit_source_loc: Whether to suffix each operation with its
            ``loc("file":line:col)``. Off by default: the declaration line shifts
            every following statement down by one, so the locations legitimately
            differ between two otherwise identical kernels and would mask a
            byte-for-byte comparison of what is actually emitted.

    Returns:
        The generated MLIR text.
    """
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    assert len(incore) == 1, f"expected one in-core function, got {[f.name for f in incore]}"
    single = ir.Program([incore[0]], incore[0].name, optimized.span)
    result = codegen.PTOCodegen().generate(single, emit_source_loc=emit_source_loc)
    return result if isinstance(result, str) else "".join(result.values())


def _bypass_warnings(capfd) -> list[str]:
    """Drain captured stderr and return only the cache-bypass warning lines."""
    err = capfd.readouterr().err
    return [line for line in err.splitlines() if BYPASS_WARNING_TAG in line]


def _tload_lines(mlir: str) -> list[str]:
    """Every emitted ``pto.tload`` — the operation a declaration would change."""
    return [line.strip() for line in mlir.splitlines() if "pto.tload" in line]


# ---------------------------------------------------------------------------
# (a) The declaration must not change a single byte of the generated MLIR
# ---------------------------------------------------------------------------


def test_declaration_leaves_generated_mlir_byte_identical():
    """`pl.set_cache_policy(b, BYPASS)` compiles to exactly today's MLIR.

    The policy reaches codegen through the ``cache`` kwarg on ``tile.load``, and
    codegen deliberately consumes it without emitting anything: no extra
    operation, no extra attribute, no reordering. Comparing the whole module
    text — not just the tload lines — is what makes "emit nothing else" testable.
    """
    with_decl = _incore_mlir(DeclaredBypass)
    without_decl = _incore_mlir(PlainMatmul)

    # Guard against a vacuous pass: there must be real loads to have changed.
    assert _tload_lines(without_decl), f"reference kernel emitted no pto.tload:\n{without_decl}"
    assert with_decl == without_decl, (
        "CachePolicy.BYPASS must not change generated code while PTOAS lacks a "
        f"bypass path ({PTOAS_ISSUE_URL})"
    )


def test_declaration_leaves_generated_mlir_identical_with_source_locations():
    """The same, through the default emit path that also writes `loc(...)`.

    ``emit_source_loc=True`` is what production codegen uses, so it is worth
    exercising; the locations themselves must differ, because the declaration
    is a real source line that shifts the statements after it. Everything else
    — every operation, in order — must still match.
    """
    with_decl = _incore_mlir(DeclaredBypass, emit_source_loc=True)
    without_decl = _incore_mlir(PlainMatmul, emit_source_loc=True)

    assert "loc(" in with_decl, f"expected source locations in the emitted MLIR:\n{with_decl}"
    stripped_with = [strip_loc(line) for line in with_decl.splitlines()]
    stripped_without = [strip_loc(line) for line in without_decl.splitlines()]
    assert stripped_with == stripped_without


# ---------------------------------------------------------------------------
# (b) The warning: once per tensor, and it names the PTOAS issue
# ---------------------------------------------------------------------------


def test_undeclared_kernel_emits_no_bypass_warning(capfd):
    """No declaration, no warning: the diagnostic tracks the request, not loads."""
    _incore_mlir(PlainMatmul)
    assert _bypass_warnings(capfd) == []


def test_declared_bypass_warns_once_and_links_the_ptoas_issue(capfd):
    """One declared tensor produces exactly one warning naming it and the issue.

    The link is asserted verbatim: it is the message's only forward reference,
    and it is what tells the reader the request is recorded rather than ignored.
    """
    _incore_mlir(DeclaredBypass)
    warnings = _bypass_warnings(capfd)

    assert len(warnings) == 1, f"expected exactly one bypass warning, got {warnings}"
    message = warnings[0]
    assert "tensor 'b'" in message, f"warning must name the declared tensor: {message}"
    assert "CachePolicy.BYPASS" in message, message
    assert PTOAS_ISSUE_URL in message, f"warning must link the PTOAS issue: {message}"


def test_each_declared_tensor_warns_exactly_once(capfd):
    """Two declared tensors get one warning each — the state is keyed by tensor."""
    _incore_mlir(TwoDeclaredTensors)
    warnings = _bypass_warnings(capfd)

    assert len(warnings) == 2, f"expected one warning per declared tensor, got {warnings}"
    assert len([w for w in warnings if "tensor 'a'" in w]) == 1, warnings
    assert len([w for w in warnings if "tensor 'b'" in w]) == 1, warnings
    assert all(PTOAS_ISSUE_URL in w for w in warnings), warnings


def test_repeated_bypassing_loads_of_one_tensor_warn_once(capfd):
    """Many loads, one declaration, one warning.

    The policy is read by every load of the tensor and a load inside an unrolled
    loop is emitted many times over, so the naive placement would produce one
    line per emitted ``pto.tload``. This kernel reads `x` twice with
    ``cache=BYPASS``; the MLIR is asserted to really carry both loads, so a
    single warning proves de-duplication rather than a missing load.
    """
    mlir = _incore_mlir(TwoBypassingLoadsOfOneTensor)
    warnings = _bypass_warnings(capfd)

    assert len(_tload_lines(mlir)) == 2, f"expected two emitted loads:\n{mlir}"
    assert len(warnings) == 1, f"warning must be once per tensor, not once per load: {warnings}"
    assert "tensor 'x'" in warnings[0], warnings[0]
    assert PTOAS_ISSUE_URL in warnings[0], warnings[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
