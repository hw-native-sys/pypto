# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen tests for the ``prefetch.*`` async GM->L2 prefetch op family."""

import re
import shutil
import subprocess

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType, pto_backend
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

_ROWS = pl.dynamic("ROWS")


def _generate_mlir(program_cls) -> str:
    """Run PassManager and PTOCodegen on the given program, return MLIR string."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
    funcs = list(optimized.functions.values())
    assert funcs, "Program has no functions"
    single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
    return codegen.PTOCodegen().generate(single)


@pl.program
class PrefetchProgram:
    """Prefetch a 1D GM row into L2, then copy a slice of it out."""

    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self,
        x: pl.Tensor[[1, 4096], pl.FP32],
        out: pl.Tensor[[1, 128], pl.FP32],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        ctx = pl.prefetch.make_context()
        evt = pl.prefetch.async_prefetch(x, ctx)
        session = pl.prefetch.session(ctx)
        pl.prefetch.wait(evt, session)
        tile = pl.load(x, [0, 0], [1, 128])
        return pl.store(tile, [0, 0], out)


@pl.program
class ZeroParamPrefetchProgram:
    """Create a prefetch context in a kernel with no user parameters."""

    @pl.function(type=pl.FunctionType.InCore)
    def main(self):
        pl.prefetch.make_context()


@pl.program
class OrderedSyntheticArgsProgram:
    """Combine dynamic shape, SDMA workspace, and SPMD synthetic arguments."""

    @pl.function(type=pl.FunctionType.InCore)
    def ordered(
        self,
        prefetch_src: pl.Tensor[[1, 4096], pl.FP32],
        x: pl.Tensor[[_ROWS, 128], pl.FP32],
        out: pl.Tensor[[_ROWS, 128], pl.FP32],
    ) -> pl.Tensor[[_ROWS, 128], pl.FP32]:
        ctx = pl.prefetch.make_context()
        evt = pl.prefetch.async_prefetch(prefetch_src, ctx)
        session = pl.prefetch.session(ctx)
        pl.prefetch.wait(evt, session)
        row = pl.tile.get_block_idx()
        return pl.store(pl.load(x, [row, 0], [1, 128]), [row, 0], out)


class TestPrefetchPTOCodegen:
    """Each prefetch op lowers to its PTOAS counterpart with the right operand types."""

    def test_context_reserves_scratch_and_uses_runtime_channel(self):
        mlir = _generate_mlir(PrefetchProgram)
        assert "pto.reserve_buffer" in mlir
        assert "size = 256" in mlir
        assert "%__pypto_sdma_channel: i32" in mlir
        assert "func.call @pypto_make_prefetch_context" in mlir

    def test_make_context_lowers_from_hidden_i8_pointer(self):
        """The adapter receives workspace, worker channel, and reserved scratch."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(r"func\.func @main\([^)]*!pto\.ptr<i8>[^)]*\)", mlir), mlir
        assert re.search(
            r"func.call @pypto_make_prefetch_context\(%\w+, %__pypto_sdma_channel, %\w+\)",
            mlir,
        ), mlir

    def test_hidden_sdma_pointer_is_first_parameter_without_user_args(self):
        """A hidden-only signature starts directly with ``%arg0``, not a comma."""
        mlir = _generate_mlir(ZeroParamPrefetchProgram)
        assert "func.func @main(%arg0: !pto.ptr<i8>, %__pypto_sdma_channel: i32)" in mlir, mlir

    def test_synthetic_argument_order_matches_wrapper(self):
        """Dynamic dims precede SDMA, which precedes SPMD in both call layers."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            OrderedSyntheticArgsProgram
        )
        func = optimized.get_function("ordered")
        assert func is not None

        mlir = codegen.PTOCodegen().generate(ir.Program([func], func.name, optimized.span))
        signature_line = next(line.strip() for line in mlir.splitlines() if "func.func @ordered(" in line)
        assert re.search(
            r"func\.func @ordered\("
            r"%arg0: !pto\.ptr<f32>, %arg1: !pto\.ptr<f32>, %arg2: !pto\.ptr<f32>, "
            r"%arg3: index, %arg4: !pto\.ptr<i8>, %__pypto_sdma_channel: i32, "
            r"%__pypto_spmd_block_idx: i32, %__pypto_spmd_block_num: i32\)",
            signature_line,
        ), signature_line

        wrapper = pto_backend._generate_kernel_wrapper(
            func, 'extern "C" __global__ AICORE void test_func() {}\n'
        )
        call_line = next(line.strip() for line in wrapper.splitlines() if line.strip().startswith("ordered("))
        assert re.search(
            r"ordered\(prefetch_src\w*, x\w*, out\w*, ROWS, "
            r"__pypto_sdma_workspace, __pypto_sdma_channel, "
            r"__pypto_spmd_block_idx, __pypto_spmd_block_num\);",
            call_line,
        ), call_line

    def test_async_prefetch_lowers_with_partition_view(self):
        """``tprefetch_async`` takes a whole-tensor partition view plus the context."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(
            r"= pto\.tprefetch_async\(%\w+, %\w+ : "
            r"!pto\.partition_tensor_view<1x4096xf32>, !pto\.prefetch_async_context\) "
            r"-> !pto\.async_event",
            mlir,
        ), mlir

    def test_session_uses_projection_assembly_form(self):
        """``get_prefetch_async_session`` is a bare projection — no parenthesised operand list."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(
            r"= pto\.get_prefetch_async_session %\w+ : "
            r"!pto\.prefetch_async_context -> !pto\.async_session",
            mlir,
        ), mlir

    def test_wait_lowers_to_paced_prefetch_adapter(self):
        """``wait`` pairs the event and session and yields an ``i1``."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(
            r"= func\.call @pypto_wait_prefetch\(%\w+, %\w+\) : \("
            r"!pto\.async_event, !pto\.async_session\) -> i1",
            mlir,
        ), mlir

    def test_handle_ssa_values_are_defined_before_use(self):
        """Each handle operand resolves to an SSA name defined earlier in the function.

        Regression guard: the handle types carry no buffer, so an emitter that
        invented a fresh temp instead of defining the assignment's bound LHS name
        would produce operands referencing undefined SSA values.
        """
        mlir = _generate_mlir(PrefetchProgram)
        defined: set[str] = set()
        for line in mlir.splitlines():
            stripped = line.strip()
            operands = re.findall(r"%\w+", stripped)
            if " = " in stripped:
                lhs = stripped.split(" = ", 1)[0].strip()
                operands = re.findall(r"%\w+", stripped.split(" = ", 1)[1])
            else:
                lhs = None
            if "pto.tprefetch_async" in stripped or "func.call @pypto_wait_prefetch" in stripped:
                for operand in operands:
                    assert operand in defined, f"{operand} used before definition in: {stripped}"
            if lhs is not None:
                defined.add(lhs)


@pl.program
class MixedPrefetchProgram:
    """Prefetch alongside a cube op, so ExpandMixedKernel splits AIC/AIV."""

    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self,
        a: pl.Tensor[[1, 128], pl.FP16],
        b: pl.Tensor[[128, 128], pl.FP16],
        out: pl.Tensor[[1, 128], pl.FP32],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        ctx = pl.prefetch.make_context()
        evt = pl.prefetch.async_prefetch(a, ctx)
        session = pl.prefetch.session(ctx)
        pl.prefetch.wait(evt, session)
        tile_a = pl.load(a, [0, 0], [1, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        return pl.store(pl.tile.matmul(tile_a, tile_b), [0, 0], out)


class TestPrefetchCoreAffinity:
    """The prefetch family is AIV-only and must not leak onto the cube lane."""

    def test_prefetch_stays_off_the_cube_lane(self):
        """In a mixed kernel, prefetch ops land only in the AIV function.

        ``TPREFETCH_ASYNC`` drives its SDMA tmpBuf from a Vec(UB) scratch tile
        inside ``PrefetchAsyncContext`` (pto-isa static_asserts
        ``ScratchTile::Loc == TileType::Vec``), and UB lives on the vector core.
        These ops carry no tile operand, so without an explicit VECTOR core
        affinity they classify as SHARED and ExpandMixedKernel duplicates them
        onto the cube lane — which has no UB, and which would also run the
        side-effecting prefetch a second time.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(MixedPrefetchProgram)

        lowered_prefetch_ops = (
            "func.call @pypto_make_prefetch_context",
            "pto.tprefetch_async",
            "pto.get_prefetch_async_session",
            "func.call @pypto_wait_prefetch",
        )
        per_core: dict[str, int] = {}
        for func in optimized.functions.values():
            if func.func_type.name not in ("AIC", "AIV"):
                continue
            mlir = codegen.PTOCodegen().generate(ir.Program([func], func.name, optimized.span))
            per_core[func.func_type.name] = sum(mlir.count(op_name) for op_name in lowered_prefetch_ops)

        assert "AIC" in per_core and "AIV" in per_core, f"expected a mixed AIC/AIV split, got {per_core}"
        assert per_core["AIC"] == 0, f"prefetch leaked onto the cube lane: {per_core}"
        assert per_core["AIV"] > 0, f"prefetch missing from the vector lane: {per_core}"


@pl.program
class PrefetchArtifactProgram:
    """End-to-end backend artifact with a runtime-injected SDMA workspace."""

    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self,
        x: pl.Tensor[[1, 4096], pl.FP32],
        out: pl.Tensor[[1, 128], pl.FP32],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        ctx = pl.prefetch.make_context()
        evt = pl.prefetch.async_prefetch(x, ctx)
        session = pl.prefetch.session(ctx)
        pl.prefetch.wait(evt, session)
        return pl.store(pl.load(x, [0, 0], [1, 128]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrate(self, x: pl.Tensor[[1, 4096], pl.FP32]) -> pl.Tensor[[1, 128], pl.FP32]:
        out = pl.create_tensor([1, 128], dtype=pl.FP32)
        return self.main(x, out)


@pl.program
class NonPrefetchArtifactProgram:
    """Control backend artifact that does not require SDMA runtime state."""

    @pl.function(type=pl.FunctionType.InCore)
    def plain(
        self,
        x: pl.Tensor[[1, 128], pl.FP32],
        out: pl.Tensor[[1, 128], pl.FP32],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        return pl.store(pl.load(x, [0, 0], [1, 128]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrate(self, x: pl.Tensor[[1, 128], pl.FP32]) -> pl.Tensor[[1, 128], pl.FP32]:
        out = pl.create_tensor([1, 128], dtype=pl.FP32)
        return self.plain(x, out)


class TestPrefetchBackendArtifact:
    """The backend injects SDMA state without exposing a user argument."""

    @staticmethod
    def _generate(program_cls, output_dir, monkeypatch) -> dict[str, str]:
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
        monkeypatch.setattr(
            pto_backend,
            "_compile_pto_module",
            lambda _pto_code, _module_name, _output_dir, _memory_planner=None: (
                'extern "C" __global__ AICORE void test_func() {}\n'
            ),
        )
        return pto_backend.generate(optimized, str(output_dir), skip_ptoas=False)

    def test_prefetch_artifact_injects_hidden_sdma_workspace(self, tmp_path, monkeypatch):
        result = self._generate(PrefetchArtifactProgram, tmp_path / "prefetch", monkeypatch)

        wrapper = result["kernels/aiv/main.cpp"]
        assert "get_dma_workspace(args, DMA_WORKSPACE_SDMA)" in wrapper
        assert re.search(r"main\(x\w*, out\w*, __pypto_sdma_workspace", wrapper), wrapper
        assert '"enable_sdma": True' in result["kernel_config.py"]
        assert "workspace" not in result["orchestration/orchestrate.cpp"]
        assert '"signature": [_D.IN, _D.OUT]' in result["kernel_config.py"]

    def test_non_prefetch_artifact_does_not_enable_sdma(self, tmp_path, monkeypatch):
        result = self._generate(NonPrefetchArtifactProgram, tmp_path / "plain", monkeypatch)

        wrapper = result["kernels/aiv/plain.cpp"]
        assert "get_dma_workspace" not in wrapper
        assert '#include "intrinsic.h"' not in wrapper
        assert '"enable_sdma"' not in result["kernel_config.py"]


@pl.program
class LiveTilePrefetchProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, x: pl.Tensor[[1, 1024], pl.FP32], out: pl.Tensor[[1, 1024], pl.FP32]):
        value = pl.load(x, [0, 0], [1, 1024])
        ctx = pl.prefetch.make_context()
        event = pl.prefetch.async_prefetch(x, ctx)
        pl.prefetch.wait(event, pl.prefetch.session(ctx))
        return pl.store(pl.add(value, 1.0), [0, 0], out)


@pl.program
class ReservedMultiContextProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, x: pl.Tensor[[1, 1024], pl.FP32], out: pl.Tensor[[1, 1024], pl.FP32]):
        pipe_buf = pl.reserve_buffer(name="user_scratch", size=4096, base=1024)
        pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=pipe_buf)
        value = pl.load(x, [0, 0], [1, 1024])
        ctx1 = pl.prefetch.make_context()
        ctx2 = pl.prefetch.make_context()
        event1 = pl.prefetch.async_prefetch(x, ctx1)
        pl.prefetch.wait(event1, pl.prefetch.session(ctx1))
        event2 = pl.prefetch.async_prefetch(x, ctx2)
        pl.prefetch.wait(event2, pl.prefetch.session(ctx2))
        return pl.store(pl.add(value, 1.0), [0, 0], out)


@pytest.mark.parametrize(
    "program,scratch_bases,tile_floor",
    [(LiveTilePrefetchProgram, [0], 256), (ReservedMultiContextProgram, [5120, 5376], 5632)],
)
def test_prefetch_scratch_never_aliases_live_tiles_or_user_reservations(program, scratch_bases, tile_floor):
    mlir = _generate_mlir(program)
    reservations = [
        line for line in mlir.splitlines() if "pto.reserve_buffer" in line and "size = 256" in line
    ]
    matches = [re.search(r"base = (\d+)", line) for line in reservations]
    assert all(match is not None for match in matches)
    assert [int(match.group(1)) for match in matches if match is not None] == scratch_bases
    constants = {name: int(value) for name, value in re.findall(r"(%\w+) = arith.constant (\d+) : i64", mlir)}
    addresses = re.findall(r"pto.alloc_tile addr = (%\w+)[^\n]*loc=vec", mlir)
    assert addresses
    assert all(constants[address] >= tile_floor for address in addresses), mlir
    assert mlir.index("pto.tload") < mlir.index("func.call @pypto_make_prefetch_context")
    assert mlir.index("func.call @pypto_wait_prefetch") < mlir.index("pto.tadds")


def test_ptoas_planner_reserves_prefetch_scratch(tmp_path):
    from pypto import passes  # noqa: PLC0415
    from pypto.backend._ptoas_locate import find_ptoas_binary  # noqa: PLC0415
    from pypto.backend.pto_backend import _compile_pto_module  # noqa: PLC0415

    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(LiveTilePrefetchProgram)
        mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=False)
    assert "size = 256, location = #pto.address_space<vec>, auto = true" in mlir
    cpp = _compile_pto_module(mlir, "live_prefetch", str(tmp_path), passes.MemoryPlanner.PTOAS)
    constants = {}
    for name, value in re.findall(r"(?:u?int(?:32|64)_t) (\w+) = (?:\(\w+\) )*(\w+);", cpp):
        if value.isdecimal():
            constants[name] = int(value)
        elif value in constants:
            constants[name] = constants[value]
    addresses = [constants[value] for value in re.findall(r"TASSIGN\([^,]+, (\w+)\)", cpp)]
    assert addresses, cpp
    scratch_match = re.search(r"= pypto_make_prefetch_context\(\w+, \w+, (\w+)\)", cpp)
    assert scratch_match is not None, cpp
    scratch_arg = scratch_match.group(1)
    scratch_base = constants[scratch_arg]
    # PTOAS may put the reservation after ordinary tiles, unlike PyPTO's floor.
    assert all(base + 4096 <= scratch_base or base >= scratch_base + 256 for base in addresses), cpp


def test_prefetch_wait_adapter_paces_probes_and_bounds_failure(tmp_path):
    """Exercise the emitted wait with a deterministic clock and completion probe."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    adapter = pto_backend._PREFETCH_ADAPTER
    wait = adapter[adapter.index('extern "C" inline __aicore__ bool pypto_wait_prefetch') :]
    source = tmp_path / "wait_contract.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#define __aicore__
constexpr uint64_t PLATFORM_PROF_SYS_CNT_FREQ = 50000000;
uint64_t clock_ticks = 0;
unsigned probe_count = 0;
uint64_t get_sys_cnt() { return clock_ticks++; }
namespace pto::comm {
struct AsyncSession {
    bool valid = true;
    mutable unsigned probes = 0;
    mutable uint64_t last_probe = 0;
};
struct AsyncEvent {
    unsigned ready_after;
    bool Test(const AsyncSession& session) const {
        if (!session.valid) return false;
        if (session.probes != 0) assert(clock_ticks - session.last_probe >= 50);
        session.last_probe = clock_ticks;
        ++session.probes;
        ++probe_count;
        return ready_after != 0 && session.probes >= ready_after;
    }
    bool Wait(const AsyncSession&) const { assert(false); return false; }
};
}
"""
        + wait
        + r"""
int main() {
    pto::comm::AsyncSession ready;
    assert(pypto_wait_prefetch({1}, ready));
    assert(probe_count == 1 && clock_ticks < 50);
    clock_ticks = 0;
    probe_count = 0;
    pto::comm::AsyncSession delayed;
    assert(pypto_wait_prefetch({3}, delayed));
    assert(probe_count == 3 && clock_ticks >= 100);
    clock_ticks = 0;
    probe_count = 0;
    pto::comm::AsyncSession absent;
    assert(!pypto_wait_prefetch({0}, absent));
    assert(clock_ticks >= 1000000 && clock_ticks < 1000100);
    assert(probe_count < 20001);
    clock_ticks = 0;
    pto::comm::AsyncSession invalid;
    invalid.valid = false;
    assert(!pypto_wait_prefetch({1}, invalid));
    assert(clock_ticks == 0);
}
"""
    )
    binary = tmp_path / "wait_contract"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
