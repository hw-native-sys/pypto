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

Most tests stop at emitted MLIR. The L2-hint integration cases also run the
installed stock PTOAS and PyPTO finalization, checking the actual ISA calls
under both memory planners and inside structured control flow.
"""

import pypto.language as pl
import pytest
from pypto import LogLevel, backend, codegen, ir, passes, set_log_level
from pypto.backend import BackendType, pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
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


_LOAD_HINTS = {
    pl.CachePolicy.DEFAULT: None,
    pl.A5CachePolicy.NORMAL_FIRST_VICTIM: "NormalFirstVictim",
    pl.A5CachePolicy.NORMAL_LAST_VICTIM: "NormalLastVictim",
    pl.A5CachePolicy.NORMAL_PERSISTENT: "NormalPersistent",
    pl.A5CachePolicy.NOT_ALLOC_KEEP: "NotAllocKeep",
    pl.A5CachePolicy.NOT_ALLOC_CLEAN: "NotAllocClean",
    pl.A5CachePolicy.NOT_ALLOC_DROP: "NotAllocDrop",
}
_STORE_HINTS = {
    pl.CachePolicy.DEFAULT: None,
    pl.A5CachePolicy.NORMAL_FIRST_VICTIM: "NormalFirstVictim",
    pl.A5CachePolicy.NORMAL_LAST_VICTIM: "NormalLastVictim",
    pl.A5CachePolicy.NORMAL_PERSISTENT: "NormalPersistent",
    pl.A5CachePolicy.NOT_ALLOC_CLEAN: "NotAllocClean",
}
_LOAD_POLICIES = list(_LOAD_HINTS)
_STORE_POLICIES = list(_STORE_HINTS)


@pytest.fixture(autouse=True)
def _setup_backend_and_log_level():
    """Pin the backend and make WARN-level output visible for every test here.

    ``conftest.py`` restores the process-global log level after each test, so
    raising it to WARN here is contained; setting it explicitly keeps the
    stderr assertions independent of ``PYPTO_LOG_LEVEL`` and of test order.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
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
    # Compare the policy within Tile IR: bypass keeps a kernel off the Buffer path.
    with passes.PassContext([], enable_buffer_ir=False):
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


def test_a5_rejects_a2a3_runtime_bypass():
    """A2/A3 address aliases cannot request an A5 instruction mode."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    with pytest.raises(ValueError, match="use A5CachePolicy on A5"):
        _incore_mlir(TwoBypassingLoadsOfOneTensor)


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


@pytest.mark.parametrize("load_hint", _LOAD_POLICIES)
def test_explicit_load_hint_is_not_silently_dropped(load_hint):
    @pl.program
    class HintedLoad:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            t = pl.load(x, [0, 0], [32, 32], cache=load_hint)
            out = pl.store(t, [0, 0], out)
            return out

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    mlir = _incore_mlir(HintedLoad)
    if _LOAD_HINTS[load_hint] is None:
        assert "__pypto_l2_hint_begin load" not in mlir
    else:
        assert f"__pypto_l2_hint_begin load {_LOAD_HINTS[load_hint]}" in mlir


@pytest.mark.parametrize("store_hint", _STORE_POLICIES)
def test_explicit_store_hint_is_not_silently_dropped(store_hint):
    @pl.program
    class HintedStore:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            t = pl.load(x, [0, 0], [32, 32])
            out = pl.store(t, [0, 0], out, cache=store_hint)
            return out

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    mlir = _incore_mlir(HintedStore)
    if _STORE_HINTS[store_hint] is None:
        assert "__pypto_l2_hint_begin store" not in mlir
    else:
        assert f"__pypto_l2_hint_begin store {_STORE_HINTS[store_hint]}" in mlir


def test_explicit_hint_overrides_scope_cache_declaration():
    @pl.program
    class HintedScope:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            pl.func_attr({"cache_policy": [(0, 1)]})
            t = pl.load(x, [0, 0], [32, 32], cache=pl.A5CachePolicy.NORMAL_FIRST_VICTIM)
            return pl.store(t, [0, 0], out)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    mlir = _incore_mlir(HintedScope)
    assert "__pypto_l2_hint_begin load NormalFirstVictim" in mlir
    assert BYPASS_ATTR not in mlir
    assert OFFSET_PARAM not in mlir


@pytest.mark.parametrize("policy", _LOAD_POLICIES)
def test_extended_scope_cache_policy_roundtrips_and_reaches_a5(policy):
    program = pl.parse_program(f"""
@pl.program
class ScopedPolicy:
    @pl.function
    def main(self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
             ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="scoped"):
            pl.set_cache_policy(x, pl.{type(policy).__name__}.{policy.name})
            value = pl.load(x, [0, 0], [32, 32])
            out = pl.store(value, [0, 0], out)
        return out
""")
    restored = ir.deserialize(ir.serialize(program))
    assert ir.python_print(restored) == ir.python_print(program)
    enum_name = "A5CachePolicy" if type(policy) is pl.A5CachePolicy else "CachePolicy"
    assert f"pl.set_cache_policy(x, pl.{enum_name}.{policy.name})" in ir.python_print(restored)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    mlir = _incore_mlir(restored)
    expected = _LOAD_HINTS[policy]
    if expected is None:
        assert "__pypto_l2_hint_begin load" not in mlir
    else:
        assert f"__pypto_l2_hint_begin load {expected}" in mlir
    assert OFFSET_PARAM not in mlir
    assert BYPASS_ATTR not in mlir


@pytest.mark.parametrize("policy", _LOAD_POLICIES)
def test_scope_nonallocating_policies_reject_written_tensors(policy):
    program = pl.parse_program(f"""
@pl.program
class ReadWritePolicy:
    @pl.function
    def main(self, x: pl.Tensor[[32, 32], pl.FP32]) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="read_write"):
            pl.set_cache_policy(x, pl.{type(policy).__name__}.{policy.name})
            value = pl.load(x, [0, 0], [32, 32])
            x = pl.store(value, [0, 0], x)
        return x
""")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    nonallocating = (
        pl.CachePolicy.BYPASS,
        pl.A5CachePolicy.NOT_ALLOC_KEEP,
        pl.A5CachePolicy.NOT_ALLOC_CLEAN,
        pl.A5CachePolicy.NOT_ALLOC_DROP,
    )
    if policy in nonallocating:
        with pytest.raises(ValueError, match="is not allowed on a tensor this scope writes"):
            _incore_mlir(program)
    else:
        assert "pto.tload" in _incore_mlir(program)


@pytest.mark.parametrize(
    "policy",
    [
        pl.A5CachePolicy.NORMAL_FIRST_VICTIM,
        pl.A5CachePolicy.NORMAL_LAST_VICTIM,
        pl.A5CachePolicy.NORMAL_PERSISTENT,
    ],
)
@pytest.mark.parametrize("explicit_default", [False, True])
def test_scope_cache_policy_survives_assemble_and_store_rebindings(policy, explicit_default):
    override = ", cache=pl.CachePolicy.DEFAULT" if explicit_default else ""
    program = pl.parse_program(f"""
@pl.program
class ScopedRebind:
    @pl.function
    def main(self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
             ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_cache_policy(out, pl.{type(policy).__name__}.{policy.name})
            value = pl.add(x, x)
            out = pl.assemble(out, value, [0, 0])
            first = pl.load(out, [0, 0], [32, 32])
            out = pl.store(pl.add(first, first), [0, 0], out)
            alias = out
            second = pl.load(alias, [0, 0], [32, 32]{override})
            out = pl.store(second, [0, 0], alias)
        return out
""")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    mlir = _incore_mlir(program)
    assert mlir.count(f"__pypto_l2_hint_begin load {_LOAD_HINTS[policy]}") == (1 if explicit_default else 2)
    assert "__pypto_l2_hint_begin store" not in mlir


@pytest.mark.parametrize(
    "policy",
    [
        pl.A5CachePolicy.NORMAL_FIRST_VICTIM,
        pl.A5CachePolicy.NORMAL_LAST_VICTIM,
        pl.A5CachePolicy.NORMAL_PERSISTENT,
    ],
)
@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("read_kind", ["explicit", "explicit_default", "synthesized"])
def test_scope_cache_policy_survives_loop_results(policy, loop_kind, read_kind):
    loop = (
        "for i, (carried,) in pl.range(steps, init_values=(out,)):"
        if loop_kind == "for"
        else "for i, carried in pl.while_(init_values=(0, out)):\n                pl.cond(i < steps)"
    )
    yielded = "out = pl.yield_(updated)" if loop_kind == "for" else "_i, out = pl.yield_(i + 1, updated)"
    read = {
        "explicit": "pl.load(out, [0, 0], [32, 32])",
        "explicit_default": "pl.load(out, [0, 0], [32, 32], cache=pl.CachePolicy.DEFAULT)",
        "synthesized": "pl.add(out, x)",
    }[read_kind]
    write = (
        "pl.assemble(out, second, [0, 0])" if read_kind == "synthesized" else "pl.store(second, [0, 0], out)"
    )
    program = pl.parse_program(f"""
@pl.program
class ScopedLoopRebind:
    @pl.function
    def main(self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
             steps: pl.Scalar[pl.INT32]) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_cache_policy(out, pl.{type(policy).__name__}.{policy.name})
            out = pl.assemble(out, pl.add(x, x), [0, 0])
            {loop}
                updated = pl.assemble(carried, pl.add(carried, x), [0, 0])
                {yielded}
            second = {read}
            out = {write}
        return out
""")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    mlir = _incore_mlir(program)
    expected = 1 if read_kind == "explicit_default" else 2
    assert mlir.count(f"__pypto_l2_hint_begin load {_LOAD_HINTS[policy]}") == expected
    assert "__pypto_l2_hint_begin store" not in mlir


@pytest.mark.parametrize("hint", [p for p in _LOAD_POLICIES if type(p) is pl.A5CachePolicy])
def test_a2a3_rejects_all_explicit_load_hints(hint):
    @pl.program
    class HintedLoad:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            t = pl.load(x, [0, 0], [32, 32], cache=hint)
            return pl.store(t, [0, 0], out)

    message = "is only supported on A5"
    with pytest.raises(ValueError, match=message):
        _incore_mlir(HintedLoad)


@pytest.mark.parametrize("hint", [p for p in _STORE_POLICIES if type(p) is pl.A5CachePolicy])
def test_a2a3_rejects_all_explicit_store_hints(hint):
    @pl.program
    class HintedStore:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            t = pl.load(x, [0, 0], [32, 32])
            return pl.store(t, [0, 0], out, cache=hint)

    with pytest.raises(ValueError, match="is only supported on A5"):
        _incore_mlir(HintedStore)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("load_hint", _LOAD_POLICIES)
@pytest.mark.parametrize("store_hint", _STORE_POLICIES)
def test_l2_hints_reach_isa_with_stock_ptoas(tmp_path, planner, load_hint, store_hint):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")

    @pl.program
    class HintedTransfers:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            t = pl.load(x, [0, 0], [32, 32], cache=load_hint)
            return pl.store(t, [0, 0], out, cache=store_hint)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(HintedTransfers)
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    cpp = pto_backend._compile_pto_module(mlir, "hinted_transfers", str(tmp_path), planner)
    if _LOAD_HINTS[load_hint] is None:
        assert "TLOAD(" in cpp
    else:
        assert f"TLOAD<pto::TLoadL2Hint::{_LOAD_HINTS[load_hint]}>" in cpp
    if _STORE_HINTS[store_hint] is None:
        assert "TSTORE(" in cpp
    else:
        assert f"TSTORE<pto::TStoreL2Hint::{_STORE_HINTS[store_hint]}>" in cpp
    assert "get_l2_cache_offset" not in cpp
    assert OFFSET_PARAM not in mlir
    assert BYPASS_ATTR not in mlir
    assert "__pypto_l2_hint" not in cpp
    assert (tmp_path / "ptoas" / "hinted_transfers.cpp").read_text() == cpp


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize(
    "load_hint", [pl.A5CachePolicy.NORMAL_FIRST_VICTIM, pl.A5CachePolicy.NORMAL_LAST_VICTIM]
)
def test_l2_hints_stay_with_transfers_in_control_flow(tmp_path, planner, load_hint):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")

    @pl.program
    class HintedControlFlow:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[64, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            flag: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[64, 32], pl.FP32]:
            for i, (out_iter,) in pl.range(2, init_values=(out,)):
                offset = i * 32
                if flag == 0:
                    tile = pl.load(x, [offset, 0], [32, 32], cache=load_hint)
                else:
                    tile = pl.load(x, [offset, 0], [32, 32])
                updated = pl.store(tile, [offset, 0], out_iter, cache=pl.A5CachePolicy.NOT_ALLOC_CLEAN)
                result = pl.yield_(updated)
            return result

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(HintedControlFlow)
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    cpp = pto_backend._compile_pto_module(mlir, "hinted_control_flow", str(tmp_path), planner)
    assert cpp.count(f"TLOAD<pto::TLoadL2Hint::{_LOAD_HINTS[load_hint]}>") == 1
    assert cpp.count("TLOAD(") == 1
    assert cpp.count("TSTORE<pto::TStoreL2Hint::NotAllocClean>") == 1
    assert "__pypto_l2_hint" not in cpp


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("with_relu", [False, True])
@pytest.mark.parametrize("policy", _STORE_POLICIES)
def test_store_l2_hint_preserves_epilogue_with_stock_ptoas(tmp_path, planner, with_relu, policy):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")

    @pl.program
    class HintedWriteback:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.INT8],
            y: pl.Tensor[[32, 64], pl.INT8],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
        ) -> pl.Tensor[[16, 32], pl.FP16]:
            lhs = pl.load(x, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
            rhs = pl.load(y, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(
                pl.tile.move(lhs, target_memory=pl.Mem.Left),
                pl.tile.move(pl.tile.transpose_view(rhs), target_memory=pl.Mem.Right),
            )
            return pl.store(
                acc,
                [0, 0],
                out,
                atomic=pl.AtomicType.Add,
                pre_quant=1.0 / 1024,
                pre_relu=with_relu,
                cache=policy,
            )

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(HintedWriteback)
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    cpp = pto_backend._compile_pto_module(mlir, "hinted_writeback", str(tmp_path), planner)
    store = next(line for line in cpp.splitlines() if "TSTORE<" in line)
    if type(policy) is pl.A5CachePolicy:
        assert f"TSTORE<pto::TStoreL2Hint::{_STORE_HINTS[policy]}," in store
    else:
        assert "TStoreL2Hint" not in store
    assert "AtomicType::AtomicAdd" in store
    assert ("ReluPreMode::NormalRelu" in store) == with_relu
    assert store.rsplit(">(", 1)[1].count(",") == 2  # Destination, accumulator, packed scale.
    assert "981467136" in cpp  # FP32 bits of 1 / 1024.
    assert "__pypto_l2_hint" not in cpp


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_store_l2_hint_preserves_phase_in_pto_ir(planner):
    """Keep phase and hint through lowering; finalizer tests cover ISA arguments.

    Current PTOAS A5 GEMV shape verification rejects PyPTO's one-row vector
    with a padded 16-row accumulator, so this case stops at the emitted IR.
    """

    @pl.program
    class HintedPhase:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 128], pl.FP32],
            y: pl.Tensor[[128, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            lhs = pl.load(x, [0, 0], [1, 128], target_memory=pl.Mem.Mat)
            rhs = pl.load(y, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
            acc = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
            return pl.store(
                acc, [0, 0], out, st_phase=pl.STPhase.Final, cache=pl.A5CachePolicy.NORMAL_FIRST_VICTIM
            )

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(HintedPhase)
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    store = next(line for line in mlir.splitlines() if "pto.tstore" in line)
    assert "stPhase = #pto<st_phase final>" in store
    assert "__pypto_l2_hint_begin store NormalFirstVictim" in mlir


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_stock_ptoas_phase_store_preserves_l2_hint(tmp_path, planner):
    """Use a legal A5 matrix producer to check the real phase-bearing ISA call."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    lhs_type = (
        "!pto.tile_buf<loc=left, dtype=f16, rows=16, cols=32, v_row=16, v_col=32, "
        "blayout=row_major, slayout=row_major, fractal=512, pad=0>"
    )
    rhs_type = (
        "!pto.tile_buf<loc=right, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, "
        "blayout=row_major, slayout=col_major, fractal=512, pad=0>"
    )
    acc_type = (
        "!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32, v_row=16, v_col=32, "
        "blayout=col_major, slayout=row_major, fractal=1024, pad=0>"
    )
    address = "addr = %addr " if planner == passes.MemoryPlanner.PYPTO else ""
    mlir = f"""module attributes {{pto.target_arch = "a5"}} {{
  func.func @phase_store(%out: !pto.ptr<f32>) attributes {{pto.kernel_kind = #pto.kernel_kind<cube>}} {{
    %addr = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %view = pto.make_tensor_view %out, shape = [%c16, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %dst = pto.partition_view %view, offsets = [%c0, %c0], sizes = [%c16, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x32xf32>
    %lhs = pto.alloc_tile {address}: {lhs_type}
    %rhs = pto.alloc_tile {address}: {rhs_type}
    %acc = pto.alloc_tile {address}: {acc_type}
    pto.tmatmul ins(%lhs, %rhs : {lhs_type}, {rhs_type}) outs(%acc : {acc_type})
      {{accPhase = #pto<acc_phase final>}}
    emitc.verbatim "// __pypto_l2_hint_begin store NormalPersistent"
    pto.tstore ins(%acc : {acc_type}) outs(%dst : !pto.partition_tensor_view<16x32xf32>)
      {{stPhase = #pto<st_phase final>}}
    emitc.verbatim "// __pypto_l2_hint_end"
    return
  }}
}}
"""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    cpp = pto_backend._compile_pto_module(mlir, "phase_store", str(tmp_path), planner)
    assert "AccPhase::Final" in cpp
    assert "TSTORE<pto::TStoreL2Hint::NormalPersistent, STPhase::Final>" in cpp
    assert "__pypto_l2_hint" not in cpp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
