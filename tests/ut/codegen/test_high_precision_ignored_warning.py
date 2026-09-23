# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``high_precision=True`` is reported when the target backend will drop it.

The request survives every layer below the DSL: PyPTO writes ``precisionType``
into the PTO IR and PTOAS keeps it as a ``HIGH_PRECISION`` template argument.
Only PTO-ISA decides whether to read it, and Ascend910B's ``TDIV_IMPL`` /
``TLOG_IMPL`` / ``TREM_IMPL`` / ``TFMOD_IMPL`` accept the template parameter
without ever branching on it — the op computes the default result and nothing
downstream says so. Ascend950 selects a separate compensated algorithm.

So the attribute alone proves nothing about what runs, and these tests assert
the two halves of that separately: the attribute is still emitted on both
backends (PTOAS's input must not change), and the warning appears on exactly
the backend that ignores it.

``tile.rsqrt`` is deliberately absent. It opts into high precision by taking a
scratch tile rather than by carrying ``precisionType``, and that form is
implemented on Ascend910B too, so it is honoured and must stay silent. The DSL
rejects ``pl.rsqrt(tile, high_precision=True)`` outright, which is what keeps it
away from this path.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import LogLevel, backend, codegen, ir, set_log_level
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager

WARNING_TAG = "[HighPrecisionIgnored]"

N = 64
CHUNKS = 4


@pytest.fixture(autouse=True)
def _reset_backend_and_log_level():
    """Make WARN-level output visible and leave no backend pinned behind.

    ``conftest.py`` restores the process-global log level after each test, so
    raising it to WARN here is contained; setting it explicitly keeps the
    stderr assertions independent of ``PYPTO_LOG_LEVEL`` and of test order.
    Each test pins its own backend, because the backend is what is under test.
    """
    backend.reset_for_testing()
    set_log_level(LogLevel.WARN)
    yield
    backend.reset_for_testing()


# ---------------------------------------------------------------------------
# Programs — one per op that can carry `precisionType`
# ---------------------------------------------------------------------------


@pl.program
class HighPrecisionDiv:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[N, N], pl.FP32],
        b: pl.Tensor[[N, N], pl.FP32],
        output: pl.Out[pl.Tensor[[N, N], pl.FP32]],
    ) -> pl.Tensor[[N, N], pl.FP32]:
        ta: pl.Tile[[N, N], pl.FP32] = pl.load(a, [0, 0], [N, N], target_memory=pl.MemorySpace.Vec)
        tb: pl.Tile[[N, N], pl.FP32] = pl.load(b, [0, 0], [N, N], target_memory=pl.MemorySpace.Vec)
        tc: pl.Tile[[N, N], pl.FP32] = pl.div(ta, tb, high_precision=True)
        return pl.store(tc, [0, 0], output)


@pl.program
class DefaultPrecisionDiv:
    """`HighPrecisionDiv` without the flag — the silent control."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[N, N], pl.FP32],
        b: pl.Tensor[[N, N], pl.FP32],
        output: pl.Out[pl.Tensor[[N, N], pl.FP32]],
    ) -> pl.Tensor[[N, N], pl.FP32]:
        ta: pl.Tile[[N, N], pl.FP32] = pl.load(a, [0, 0], [N, N], target_memory=pl.MemorySpace.Vec)
        tb: pl.Tile[[N, N], pl.FP32] = pl.load(b, [0, 0], [N, N], target_memory=pl.MemorySpace.Vec)
        tc: pl.Tile[[N, N], pl.FP32] = pl.div(ta, tb)
        return pl.store(tc, [0, 0], output)


@pl.program
class HighPrecisionLog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[N, N], pl.FP32],
        output: pl.Out[pl.Tensor[[N, N], pl.FP32]],
    ) -> pl.Tensor[[N, N], pl.FP32]:
        ta: pl.Tile[[N, N], pl.FP32] = pl.load(a, [0, 0], [N, N], target_memory=pl.MemorySpace.Vec)
        tc: pl.Tile[[N, N], pl.FP32] = pl.log(ta, high_precision=True)
        return pl.store(tc, [0, 0], output)


@pl.program
class HighPrecisionRecip:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[N, N], pl.FP32],
        output: pl.Out[pl.Tensor[[N, N], pl.FP32]],
    ) -> pl.Tensor[[N, N], pl.FP32]:
        ta: pl.Tile[[N, N], pl.FP32] = pl.load(a, [0, 0], [N, N], target_memory=pl.MemorySpace.Vec)
        tc: pl.Tile[[N, N], pl.FP32] = pl.recip(ta, high_precision=True)
        return pl.store(tc, [0, 0], output)


@pl.program
class HighPrecisionUnrolledDiv:
    """One written `pl.div`, replicated by `pl.unroll` into several emitted ops.

    Every copy carries the same span, which is what the per-site deduplication
    has to collapse.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[N * CHUNKS, N], pl.FP32],
        b: pl.Tensor[[N * CHUNKS, N], pl.FP32],
        output: pl.Out[pl.Tensor[[N * CHUNKS, N], pl.FP32]],
    ) -> pl.Tensor[[N * CHUNKS, N], pl.FP32]:
        for i in pl.unroll(CHUNKS):
            ta: pl.Tile[[N, N], pl.FP32] = pl.load(a, [i * N, 0], [N, N], target_memory=pl.MemorySpace.Vec)
            tb: pl.Tile[[N, N], pl.FP32] = pl.load(b, [i * N, 0], [N, N], target_memory=pl.MemorySpace.Vec)
            tc: pl.Tile[[N, N], pl.FP32] = pl.div(ta, tb, high_precision=True)
            out = pl.store(tc, [i * N, 0], output)
        return out


@pl.program
class TwoWrittenTileDivSites:
    """Two `pl.div` statements on adjacent lines, already at tile level."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[N, N], pl.FP32],
        b: pl.Tensor[[N, N], pl.FP32],
        output: pl.Out[pl.Tensor[[N, N], pl.FP32]],
    ) -> pl.Tensor[[N, N], pl.FP32]:
        ta: pl.Tile[[N, N], pl.FP32] = pl.load(a, [0, 0], [N, N], target_memory=pl.MemorySpace.Vec)
        tb: pl.Tile[[N, N], pl.FP32] = pl.load(b, [0, 0], [N, N], target_memory=pl.MemorySpace.Vec)
        first: pl.Tile[[N, N], pl.FP32] = pl.div(ta, tb, high_precision=True)
        second: pl.Tile[[N, N], pl.FP32] = pl.div(first, tb, high_precision=True)
        return pl.store(second, [0, 0], output)


@pl.program
class TwoWrittenTensorDivSites:
    """The same pair written on tensors, so `ConvertTensorToTileOps` rebuilds
    both calls before codegen ever sees them."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[N, N], pl.FP32],
        b: pl.Tensor[[N, N], pl.FP32],
    ) -> pl.Tensor[[N, N], pl.FP32]:
        first: pl.Tensor[[N, N], pl.FP32] = pl.div(a, b, high_precision=True)
        second: pl.Tensor[[N, N], pl.FP32] = pl.div(first, b, high_precision=True)
        return second


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _incore_mlir(program_cls: ir.Program, backend_type: BackendType) -> str:
    """Lower and emit the single in-core function for one backend."""
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    assert len(incore) == 1, f"expected one in-core function, got {[f.name for f in incore]}"
    single = ir.Program([incore[0]], incore[0].name, optimized.span)
    result = codegen.PTOCodegen().generate(single)
    return result if isinstance(result, str) else "".join(result.values())


def _warnings(capfd) -> list[str]:
    """Drain captured stderr and return the high-precision warning lines.

    The warning travels the C++ ``LOG_WARN`` channel, which writes to
    ``std::cerr`` from native code, so it is read with pytest's ``capfd``
    (file-descriptor level) rather than ``capsys``.
    """
    err = capfd.readouterr().err
    return [line for line in err.splitlines() if WARNING_TAG in line]


def _source_location(warning_line: str) -> str:
    """The `file:line:col` a warning points at, with the directory stripped."""
    match = re.search(r"at (\S+):(\d+):(\d+)\s*$", warning_line)
    assert match, f"warning carries no source location: {warning_line}"
    return f"{Path(match.group(1)).name}:{match.group(2)}:{match.group(3)}"


# ---------------------------------------------------------------------------
# (a) The backend that drops the request warns; the one that honours it does not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("program_cls", "op_name"),
    [
        (HighPrecisionDiv, "tile.div"),
        (HighPrecisionLog, "tile.log"),
        (HighPrecisionRecip, "tile.recip"),
    ],
)
def test_ascend910b_warns_naming_the_op_and_the_arch(program_cls, op_name, capfd):
    _incore_mlir(program_cls, BackendType.Ascend910B)
    warnings = _warnings(capfd)

    assert len(warnings) == 1, f"expected exactly one warning, got {warnings}"
    assert f"{op_name}(high_precision=True)" in warnings[0], warnings[0]
    assert "a2a3" in warnings[0], warnings[0]


@pytest.mark.parametrize("program_cls", [HighPrecisionDiv, HighPrecisionLog, HighPrecisionRecip])
def test_ascend950_stays_silent_because_it_honours_the_request(program_cls, capfd):
    _incore_mlir(program_cls, BackendType.Ascend950)
    assert _warnings(capfd) == []


def test_no_warning_without_the_flag(capfd):
    """The warning tracks the request, not the operator."""
    _incore_mlir(DefaultPrecisionDiv, BackendType.Ascend910B)
    assert _warnings(capfd) == []


# ---------------------------------------------------------------------------
# (b) Reported once per written site, not once per emitted op
# ---------------------------------------------------------------------------


def test_unrolled_loop_warns_once_for_its_single_written_site(capfd):
    mlir = _incore_mlir(HighPrecisionUnrolledDiv, BackendType.Ascend910B)
    emitted = [line for line in mlir.splitlines() if "pto.tdiv" in line]

    assert len(emitted) == CHUNKS, f"expected the loop body to be replicated:\n{mlir}"
    assert len(_warnings(capfd)) == 1, "one written `pl.div` must produce one warning"


@pytest.mark.parametrize("program_cls", [TwoWrittenTileDivSites, TwoWrittenTensorDivSites])
def test_two_written_sites_keep_their_own_locations(program_cls, capfd):
    """Deduplication must key on the written statement, not on the operator.

    Both the dedup key and the reported location come from the call's own
    `span_`. `ConvertTensorToTileOps` rebuilds a tensor-level `pl.div` into a
    `tile.div`, and it attributes each synthesized op to the source that
    motivated it rather than to the enclosing `def` (only structural nodes take
    the function span). If that ever regressed to a function-level span, both
    sites here would collapse onto one key and report the `def` line, so this
    pins the two apart on both the tile and the rebuilt-tensor path.
    """
    _incore_mlir(program_cls, BackendType.Ascend910B)
    warnings = _warnings(capfd)

    assert len(warnings) == 2, f"expected one warning per written site, got {warnings}"
    locations = {_source_location(line) for line in warnings}
    assert len(locations) == 2, f"both sites reported the same location: {locations}"
    assert all(location.startswith(f"{Path(__file__).name}:") for location in locations), (
        f"warnings must name this source file, got {locations}"
    )


# ---------------------------------------------------------------------------
# (c) The warning does not change what PTOAS is given
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend_type", "arch"), [(BackendType.Ascend910B, "a2a3"), (BackendType.Ascend950, "a5")]
)
def test_attribute_is_emitted_on_both_backends(backend_type, arch):
    """Warning about the drop must not suppress the attribute.

    PyPTO reports what PTO-ISA will do with the request; it does not decide the
    request on PTOAS's behalf. The emitted MLIR stays identical either way, so a
    `.pto` captured on one backend keeps meaning the same thing on the other.
    """
    mlir = _incore_mlir(HighPrecisionDiv, backend_type)

    assert f'pto.target_arch = "{arch}"' in mlir, mlir
    assert "precisionType = #pto<div_precision high_precision>" in mlir, mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
