# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen behaviour of the declared GM cache-access policy (pypto #2534, #2680).

PTOAS >= v0.64 carries a streaming GM read as a ``cache_policy`` attribute on
``pto.tload`` plus an optional byte ``offset`` it adds to that one load's source
address. On a2a3 the uncached mapping of a page is reached by address alias, so
the attribute alone declares the policy and moves nothing: the offset is what
makes the load bypass. That distance is a per-device value only the driver
knows, so it is threaded in as the synthetic ``%__pypto_l2_cache_offset``
parameter the kernel wrapper fills from ``get_l2_cache_offset(args)``.

The contract asserted here has four sides:

1. **A declared read carries the attribute** — once per emitted load, since the
   hint belongs to the instruction rather than to the tensor (an unrolled loop
   emits the same load many times, and each one must carry it).
2. **An undeclared read carries nothing.** ``CachePolicy.DEFAULT`` emits no
   attribute at all, so a kernel that states no policy keeps the PTO form it had
   before the feature existed. That is also what makes the emitted dict the
   *only* difference between the two otherwise identical kernels below.
3. **The per-access surface still wins.** An explicit
   ``cache=CachePolicy.DEFAULT`` inside a bypassing scope re-caches that one
   read, which is observable in the emitted MLIR for the first time — while
   BYPASS was a no-op, the documented precedence could only be checked on the IR
   (``tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py``).
4. **The device offset reaches every declared load, from one parameter.** The
   value cannot change during a dispatch, so it is read once at kernel entry and
   passed down; a kernel with no declared read gains no parameter. An
   architecture that does not map GM twice — a5 — emits the attribute and no
   offset, and the declaration has no effect there.

The assembler's own acceptance of the attribute is not asserted here: every UT
runs with ``skip_ptoas=True``, so these tests stop at the emitted MLIR text (the
same contract as the MX ``layout`` attribute in ``test_mx_ops_codegen.py``).
"""

import pypto.language as pl
import pytest
from pypto import LogLevel, backend, codegen, ir, set_log_level
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager

# The exact attribute PTOAS >= v0.64 consumes on `pto.tload`.
BYPASS_ATTR = "cache_policy = #pto.load_cache_policy<l2_bypass>"
# The operand that carries the device's no-cache alias distance to that load,
# and the synthetic parameter it is bound to.
OFFSET_PARAM = "%__pypto_l2_cache_offset"
OFFSET_OPERAND = f"offset = {OFFSET_PARAM} : i64"
# The diagnostic that stood in for the attribute while PTOAS had no bypass path.
# It must never be emitted again — the request is now honoured, not reported.
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
# is the only difference between them, which is what makes the "the attribute is
# the whole difference" comparison meaningful.
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
    """The same kernel with no declaration — the comparison reference."""

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
    """Both operands declared: both synthesised loads carry the attribute."""

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
    """Two `pl.load(..., cache=BYPASS)` reads of ONE tensor — two hinted loads."""

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


@pl.program
class ReCachedSingleLoad:
    """A declared parameter with one access explicitly opted back into the cache.

    The declaration is written the way pass 9 leaves it — ``cache_policy`` on the
    outlined kernel, ``(param index, policy)`` — because the override has to be
    stated at the access, and a hand-written InCore kernel is where an access is
    spelled out. See the pass-level pair in
    ``tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py``.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[ROWS, COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        # (param index, policy-as-int); 1 is CachePolicy.BYPASS. The parser takes
        # integer literals here only, which is also how pass 9 writes the attr.
        pl.func_attr({"cache_policy": [(0, 1)]})
        t: pl.Tile[[16, COLS], pl.FP32] = pl.load(x, [0, 0], [16, COLS], cache=pl.CachePolicy.DEFAULT)
        return pl.store(t, [0, 0], out)


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
            comparison of what is actually emitted.

    Returns:
        The generated MLIR text.
    """
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    assert len(incore) == 1, f"expected one in-core function, got {[f.name for f in incore]}"
    single = ir.Program([incore[0]], incore[0].name, optimized.span)
    result = codegen.PTOCodegen().generate(single, emit_source_loc=emit_source_loc)
    return result if isinstance(result, str) else "".join(result.values())


def _tload_lines(mlir: str) -> list[str]:
    """Every emitted ``pto.tload`` — the operation the declaration changes."""
    return [line.strip() for line in mlir.splitlines() if "pto.tload" in line]


def _hinted_tloads(mlir: str) -> list[str]:
    """The emitted loads that carry the L2-bypass attribute."""
    return [line for line in _tload_lines(mlir) if BYPASS_ATTR in line]


def _source_of(tload_line: str) -> str:
    """The declared tensor a ``pto.tload`` reads, taken from its partition view.

    ``pto.tload ins(%b__ssa_v0_pview : ...)`` → ``b``. The SSA name is built from
    the source tensor's name, which is what makes the operand identifiable at all.
    """
    ins = tload_line.split("ins(%", 1)[1]
    return ins.split("__", 1)[0]


def _warnings(capfd) -> list[str]:
    """Drain captured stderr and return any cache-bypass warning lines."""
    err = capfd.readouterr().err
    return [line for line in err.splitlines() if BYPASS_WARNING_TAG in line]


# ---------------------------------------------------------------------------
# (a) A declared read carries the attribute, and it is the whole difference
# ---------------------------------------------------------------------------


def test_declared_tensor_load_carries_the_bypass_attribute():
    """Only the declared operand's load is hinted; the other is untouched."""
    mlir = _incore_mlir(DeclaredBypass)
    loads = _tload_lines(mlir)

    assert len(loads) == 2, f"expected one load per matmul operand:\n{mlir}"
    hinted = {_source_of(line) for line in loads if BYPASS_ATTR in line}
    plain = {_source_of(line) for line in loads if BYPASS_ATTR not in line}
    assert hinted == {"b"}, f"only the declared tensor may be hinted, got {hinted}"
    assert plain == {"a"}, f"the undeclared tensor must stay cached, got {plain}"


def test_the_declaration_adds_only_the_attribute_offset_and_parameter():
    """`pl.set_cache_policy(b, BYPASS)` adds an annotated load and nothing else.

    No extra operation, no extra view, no reordering: stripping the attribute,
    the offset operand and the parameter that feeds it from the declared
    kernel's MLIR must reproduce the undeclared kernel's, line for line. That is
    what keeps the declaration a property of the load rather than a codegen mode.
    """
    with_decl = _incore_mlir(DeclaredBypass)
    without_decl = _incore_mlir(PlainMatmul)

    assert BYPASS_ATTR in with_decl, f"declared kernel emitted no bypass hint:\n{with_decl}"
    assert BYPASS_ATTR not in without_decl, f"undeclared kernel must carry no hint:\n{without_decl}"
    stripped = with_decl.replace(" {" + BYPASS_ATTR + "}" + " " + OFFSET_OPERAND, "")
    stripped = stripped.replace(f", {OFFSET_PARAM}: i64", "")
    assert stripped == without_decl


def test_each_declared_tensor_gets_its_own_hinted_load():
    """Two declared operands produce two hinted loads."""
    mlir = _incore_mlir(TwoDeclaredTensors)
    hinted = {_source_of(line) for line in _hinted_tloads(mlir)}
    assert hinted == {"a", "b"}, f"expected both operands hinted, got {hinted}:\n{mlir}"


def test_every_emitted_load_of_a_declared_tensor_is_hinted():
    """Many loads, one declaration, one hint each.

    The hint belongs to the instruction, not to the tensor: the old diagnostic
    was deliberately once-per-tensor, but an L2 hint that lands on only the
    first of two loads would leave the second one allocating in L2. This kernel
    reads `x` twice with ``cache=BYPASS``; both emitted loads must carry it.
    """
    mlir = _incore_mlir(TwoBypassingLoadsOfOneTensor)
    loads = _tload_lines(mlir)

    assert len(loads) == 2, f"expected two emitted loads:\n{mlir}"
    assert len(_hinted_tloads(mlir)) == 2, f"every emitted load must carry the hint:\n{mlir}"


# ---------------------------------------------------------------------------
# (b) The device offset reaches every declared load through one parameter
# ---------------------------------------------------------------------------


def _signature_line(mlir: str) -> str:
    """The emitted ``func.func`` signature — where synthetic params are appended."""
    return next(line.strip() for line in mlir.splitlines() if line.strip().startswith("func.func @"))


def test_declared_load_takes_the_device_offset_as_its_tload_operand():
    """The attribute declares the policy; the offset is what moves the address.

    PTOAS applies the offset only to a load that also declared ``l2_bypass``, so
    the two travel together on the same ``pto.tload`` — an attribute without an
    offset compiles to an ordinary cached load, which would make the declaration
    silently inert.
    """
    mlir = _incore_mlir(DeclaredBypass)
    hinted = _hinted_tloads(mlir)

    assert len(hinted) == 1, f"expected exactly one hinted load:\n{mlir}"
    assert OFFSET_OPERAND in hinted[0], f"hinted load carries no device offset:\n{hinted[0]}"
    assert f"{OFFSET_PARAM}: i64" in _signature_line(mlir), _signature_line(mlir)


def test_one_parameter_serves_every_declared_load():
    """Two hinted loads, one parameter: the value cannot change within a dispatch.

    The wrapper reads ``get_l2_cache_offset(args)`` once at kernel entry, so a
    second parameter would mean a second GM read of a constant.
    """
    mlir = _incore_mlir(TwoBypassingLoadsOfOneTensor)
    hinted = _hinted_tloads(mlir)

    assert len(hinted) == 2, f"expected two hinted loads:\n{mlir}"
    assert all(OFFSET_OPERAND in line for line in hinted), f"every hinted load takes it:\n{mlir}"
    assert _signature_line(mlir).count(f"{OFFSET_PARAM}: i64") == 1, _signature_line(mlir)


def test_a5_declares_the_policy_without_an_address_offset():
    """A5 does not map GM twice, so there is no alias for an offset to reach.

    Its runtime exposes no ``get_l2_cache_offset`` either, so emitting the
    parameter would produce a kernel wrapper that cannot compile. PTOAS v0.64
    lowers the bare attribute to an ordinary load, so the declaration is
    accepted there and does nothing — which is a property of the target, not a
    missing operand.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    mlir = _incore_mlir(TwoBypassingLoadsOfOneTensor)

    assert _hinted_tloads(mlir), f"a5 must still carry the declaration:\n{mlir}"
    assert OFFSET_PARAM not in mlir, f"a5 must take no address offset:\n{mlir}"


# ---------------------------------------------------------------------------
# (c) An undeclared read — and an explicitly re-cached one — carry nothing
# ---------------------------------------------------------------------------


def test_undeclared_kernel_emits_no_cache_attribute():
    """`CachePolicy.DEFAULT` emits nothing, so unrelated kernels are unchanged."""
    mlir = _incore_mlir(PlainMatmul)

    assert _tload_lines(mlir), f"reference kernel emitted no pto.tload:\n{mlir}"
    assert "cache_policy" not in mlir, f"an undeclared kernel must emit no cache attribute:\n{mlir}"


def test_explicit_default_load_beats_the_declaration_and_emits_no_attribute():
    """An explicit ``cache=CachePolicy.DEFAULT`` re-caches that one read.

    The documented precedence — the per-access kwarg wins over the scope
    declaration, in both directions — becomes observable in codegen only now
    that BYPASS emits something.
    """
    mlir = _incore_mlir(ReCachedSingleLoad)

    assert _tload_lines(mlir), f"kernel emitted no pto.tload:\n{mlir}"
    assert "cache_policy" not in mlir, f"the re-cached access must carry no hint:\n{mlir}"


# ---------------------------------------------------------------------------
# (d) The stand-in diagnostic is gone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("program_cls", [PlainMatmul, DeclaredBypass, TwoBypassingLoadsOfOneTensor])
def test_no_cache_bypass_warning_is_emitted(program_cls, capfd):
    """Nothing warns any more: the request is honoured rather than reported.

    The warning travelled the C++ ``LOG_WARN`` channel, which writes to
    ``std::cerr`` from native code, so it is read with pytest's ``capfd``
    (file-descriptor level) rather than ``capsys`` — the same mechanism
    ``tests/ut/core/test_logging.py`` uses.
    """
    _incore_mlir(program_cls)
    assert _warnings(capfd) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
