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
   offset, and reports that the declaration falls back to a cached read.

Tests marked ``ptoas_compile`` also check the real assembler output and its
wrapper: the runtime supplies the numeric offset, the callee receives a symbolic
parameter, and only the declared load's descriptor is rebased in bytes. CI runs
these tests with ``--require-ptoas`` so missing compiler coverage cannot skip.
"""

import re
from pathlib import Path

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
# A missing target offset path must be visible while preserving cached reads.
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
class MixedCacheLoads:
    """Read the same source window with both policies and consume both tiles."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[ROWS, COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        # The source descriptor can be shared, but applying BYPASS to it in
        # place would incorrectly change the following DEFAULT read as well.
        bypassed = pl.load(x, [0, 0], [ROWS, COLS], cache=pl.CachePolicy.BYPASS)
        cached = pl.load(x, [0, 0], [ROWS, COLS], cache=pl.CachePolicy.DEFAULT)
        total = pl.add(bypassed, cached)
        return pl.store(total, [0, 0], out)


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


def _incore_function(program_cls) -> ir.Function:
    """Return the sole optimized kernel, excluding its orchestration parent."""
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    assert len(incore) == 1, f"expected one in-core function, got {[f.name for f in incore]}"
    return incore[0]


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
    func = _incore_function(program_cls)
    single = ir.Program([func], func.name, func.span)
    result = codegen.PTOCodegen().generate(single, emit_source_loc=emit_source_loc)
    return result if isinstance(result, str) else "".join(result.values())


def _compile_pto(mlir: str, output_dir: Path) -> str:
    """Compile with the active target and PyPTO's checked PTOAS invocation."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    return pto_backend._compile_pto_module(mlir, "cache_policy", str(output_dir))


def _assert_offset_loads(
    cpp: str, func_name: str, bypass_count: int, cached_count: int, *, same_source: bool = False
) -> None:
    """Follow the runtime parameter through byte arithmetic and copied descriptors.

    PTOAS assigns its own C++ variable names. Capture their dataflow instead of
    pinning the test to a particular assembler's temporary numbering.
    """
    signature = re.search(rf"\bvoid\s+{re.escape(func_name)}\(([^)]*)\)", cpp)
    assert signature is not None, cpp
    offset = re.search(r"\b(?:u?int64_t)\s+(\w+)\s*$", signature[1])
    assert offset is not None, signature[0]
    rebased = re.findall(
        rf"__gm__\s+\w+\s*\*\s*(\w+)\s*=\s*[^;]*"
        rf"\(\s*(\w+)\s*\+\s*{re.escape(offset[1])}\s*\)\s*;",
        cpp,
    )
    assert len(rebased) == bypass_count, cpp
    load_sources = re.findall(r"\bTLOAD\(\w+,\s*(\w+)\);", cpp)
    assert len(load_sources) == bypass_count + cached_count, cpp
    bypass_sources = []
    original_sources = []
    for pointer, byte_pointer in rebased:
        assign = re.search(rf"\bTASSIGN\((\w+),\s*{re.escape(pointer)}\);", cpp)
        assert assign is not None, cpp
        alias = assign[1]
        copy = re.search(rf"\b{re.escape(alias)}\s*=\s*(\w+)\s*;", cpp)
        assert copy is not None and copy[1] != alias, cpp
        # The byte pointer must originate from this load's source descriptor,
        # not merely have the correct uint8_t type: rebasing another tensor's
        # address would preserve the policy syntax while reading wrong data.
        byte_definition = re.search(
            rf"__gm__\s+uint8_t\s*\*\s*{re.escape(byte_pointer)}\s*=\s*"
            r"reinterpret_cast<\s*__gm__\s+uint8_t\s*\*>\(([^;]+)\);",
            cpp,
        )
        assert byte_definition is not None, cpp
        tensor_data = rf"PTOAS__GLOBAL_TENSOR_DATA\(\s*{re.escape(copy[1])}\s*\)"
        byte_source = byte_definition[1].strip()
        if re.fullmatch(r"\w+", byte_source):
            # Some assembler versions materialize the typed data pointer
            # before casting it to a byte pointer; follow that extra SSA edge.
            assert re.search(
                rf"__gm__\s+\w+\s*\*\s*{re.escape(byte_source)}\s*=\s*{tensor_data}\s*;",
                cpp,
            ), cpp
        else:
            assert re.fullmatch(tensor_data, byte_source), cpp
        bypass_sources.append(alias)
        original_sources.append(copy[1])
    assert all(load_sources.count(alias) == 1 for alias in bypass_sources), cpp
    cached_sources = [source for source in load_sources if source not in bypass_sources]
    assert len(cached_sources) == cached_count, cpp
    if same_source:
        # MixedCacheLoads uses the identical source window twice. The cached
        # load must retain exactly the descriptor copied for the bypass read.
        assert cached_sources == original_sources, cpp
    assert "TLoadL2Hint" not in cpp, cpp


def _assert_no_gm_byte_rebase(cpp: str) -> None:
    """The FP32 control kernels must not enter the GM byte-offset path."""
    # TASSIGN also binds ordinary on-chip tile buffers to UB/L1 addresses.
    # Those are expected here; the BYPASS-specific operation is casting a GM
    # data pointer to bytes before adding the runtime alias distance.
    assert not re.search(r"reinterpret_cast<\s*__gm__\s+uint8_t\s*\*>", cpp), cpp


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


def test_a5_declares_the_policy_without_an_address_offset(capfd):
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
    warnings = _warnings(capfd)
    assert len(warnings) == 1, warnings
    assert "kernel" in warnings[0] and "a5" in warnings[0], warnings


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
# (d) Supported A2/A3 kernels need no fallback diagnostic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("program_cls", [PlainMatmul, DeclaredBypass, TwoBypassingLoadsOfOneTensor])
def test_no_cache_bypass_warning_is_emitted(program_cls, capfd):
    """The supported offset path and ordinary DEFAULT reads need no warning.

    The warning travelled the C++ ``LOG_WARN`` channel, which writes to
    ``std::cerr`` from native code, so it is read with pytest's ``capfd``
    (file-descriptor level) rather than ``capsys`` — the same mechanism
    ``tests/ut/core/test_logging.py`` uses.
    """
    _incore_mlir(program_cls)
    assert _warnings(capfd) == []


# ---------------------------------------------------------------------------
# (e) Runtime wrapper -> symbolic parameter -> actual PTOAS load lowering
# ---------------------------------------------------------------------------


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "program_cls,bypass_count,cached_count",
    [(DeclaredBypass, 1, 1), (TwoBypassingLoadsOfOneTensor, 2, 0), (MixedCacheLoads, 1, 1)],
)
@pytest.mark.parametrize(
    "runtime", [passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, passes.RuntimeKind.HOST_BUILD_GRAPH]
)
def test_runtime_offset_reaches_only_bypassing_loads(
    tmp_path, program_cls, bypass_count, cached_count, runtime
):
    """Check the complete bridge rather than only the MLIR attribute (#3116)."""
    # Both supported runtimes transport the same device-owned byte offset.
    # Select through PassContext, matching normal compilation legalization.
    with passes.PassContext([], runtime=runtime):
        func = _incore_function(program_cls)
        mlir = codegen.PTOCodegen().generate(ir.Program([func], func.name, func.span), emit_source_loc=False)
        assert isinstance(mlir, str)
        cpp = _compile_pto(mlir, tmp_path)
        wrapper = pto_backend._generate_kernel_wrapper(func, cpp)
    assert _signature_line(mlir).count(f"{OFFSET_PARAM}: i64") == 1, mlir
    assert all(OFFSET_OPERAND in line for line in _hinted_tloads(mlir)), mlir
    _assert_offset_loads(
        cpp, func.name, bypass_count, cached_count, same_source=program_cls is MixedCacheLoads
    )

    # The concrete value belongs to runtime, so its accessor appears in the
    # wrapper exactly once, rather than in the bare PTOAS-generated callee.
    assert "get_l2_cache_offset" not in cpp, cpp
    assert wrapper.count("get_l2_cache_offset(args)") == 1, wrapper
    assert re.search(
        r"\buint64_t\s+__pypto_l2_cache_offset\s*=\s*get_l2_cache_offset\(args\)\s*;", wrapper
    ), wrapper
    call = re.search(rf"^\s*{re.escape(func.name)}\(([^;]*)\);", wrapper, re.MULTILINE)
    assert call is not None, wrapper
    assert call[1].split(", ")[-1] == "__pypto_l2_cache_offset", call[0]


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("target", [BackendType.Ascend910, BackendType.Ascend950])
def test_ptoas_without_offset_preserves_safe_cached_loads(tmp_path, target):
    """A bare declaration must keep the assembler's ordinary-load fallback."""
    backend.reset_for_testing()
    backend.set_backend_type(target)
    mlir = _incore_mlir(MixedCacheLoads)
    # On A2/A3 remove the symbolic bridge deliberately to test PTOAS's safe
    # fallback; on A5 this is the target's naturally emitted contract.
    no_offset = mlir.replace(f" {OFFSET_OPERAND}", "").replace(f", {OFFSET_PARAM}: i64", "")
    plain = no_offset.replace(f" {{{BYPASS_ATTR}}}", "")
    assert BYPASS_ATTR in no_offset and BYPASS_ATTR not in plain, mlir
    fallback_cpp = _compile_pto(no_offset, tmp_path / "fallback")
    plain_cpp = _compile_pto(plain, tmp_path / "default")
    assert fallback_cpp == plain_cpp
    assert "TLOAD(" in fallback_cpp and "TLoadL2Hint" not in fallback_cpp, fallback_cpp
    _assert_no_gm_byte_rebase(fallback_cpp)


@pytest.mark.ptoas_compile
def test_default_kernel_keeps_cached_loads_and_wrapper_abi(tmp_path):
    """A DEFAULT kernel gains neither pointer rebasing nor runtime arguments."""
    # This test covers the ordinary Tile load path; Buffer selection is not
    # relevant to the hidden argument's absence.
    with passes.PassContext([], enable_buffer_ir=False):
        func = _incore_function(PlainMatmul)
        mlir = codegen.PTOCodegen().generate(ir.Program([func], func.name, func.span), emit_source_loc=False)
    assert isinstance(mlir, str)
    assert OFFSET_PARAM not in mlir and BYPASS_ATTR not in mlir, mlir
    cpp = _compile_pto(mlir, tmp_path)
    assert len(re.findall(r"\bTLOAD\(", cpp)) == 2, cpp
    _assert_no_gm_byte_rebase(cpp)
    wrapper = pto_backend._generate_kernel_wrapper(func, cpp)
    assert "get_l2_cache_offset" not in wrapper and "__pypto_l2_cache_offset" not in wrapper, wrapper
    call = re.search(rf"^\s*{re.escape(func.name)}\(([^;]*)\);", wrapper, re.MULTILINE)
    assert call is not None and len(call[1].split(", ")) == len(func.params), wrapper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
