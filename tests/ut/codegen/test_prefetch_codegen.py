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

    def test_make_context_lowers_from_hidden_i8_pointer(self):
        """A synthetic INT8 pointer feeds ``pto.make_prefetch_async_context``."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(r"func\.func @main\([^)]*!pto\.ptr<i8>[^)]*\)", mlir), mlir
        assert re.search(
            r"pto\.make_prefetch_async_context\(%\w+ : !pto\.ptr<i8>\)",
            mlir,
        ), mlir

    def test_hidden_sdma_pointer_is_first_parameter_without_user_args(self):
        """A hidden-only signature starts directly with ``%arg0``, not a comma."""
        mlir = _generate_mlir(ZeroParamPrefetchProgram)
        assert "func.func @main(%arg0: !pto.ptr<i8>)" in mlir, mlir

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
            r"%arg3: index, %arg4: !pto\.ptr<i8>, "
            r"%__pypto_spmd_block_idx: i32, %__pypto_spmd_block_num: i32\)",
            signature_line,
        ), signature_line

        wrapper = pto_backend._generate_kernel_wrapper(
            func, 'extern "C" __global__ AICORE void test_func() {}\n'
        )
        call_line = next(line.strip() for line in wrapper.splitlines() if line.strip().startswith("ordered("))
        assert re.search(
            r"ordered\(prefetch_src\w*, x\w*, out\w*, ROWS, "
            r"__pypto_sdma_workspace, __pypto_spmd_block_idx, __pypto_spmd_block_num\);",
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

    def test_wait_lowers_to_comm_wait_async_event(self):
        """``wait`` pairs the event and session and yields an ``i1``."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(
            r"= pto\.comm\.wait_async_event\(%\w+, %\w+ : "
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
            if "pto.tprefetch_async" in stripped or "pto.comm.wait_async_event" in stripped:
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

        per_core: dict[str, int] = {}
        for func in optimized.functions.values():
            if func.func_type.name not in ("AIC", "AIV"):
                continue
            mlir = codegen.PTOCodegen().generate(ir.Program([func], func.name, optimized.span))
            per_core[func.func_type.name] = mlir.count("pto.tprefetch_async") + mlir.count(
                "pto.make_prefetch_async_context"
            )

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
