# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mat prefetch values rotate descriptors while keeping both physical banks live."""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen
from pypto.runtime import ExecutionMode, RunConfig

_MAT_TO_MAT = re.compile(r"pto\.tmov\s+ins\([^)]*loc=mat[^)]*\)\s*outs\([^)]*loc=mat[^)]*\)", re.DOTALL)
_PANEL = "pl.Tile[[64, 64], pl.FP16, pl.Mem.Mat]"
_RESULT = "pl.Tile[[64, 16], pl.FP32, pl.Mem.Acc]"


def _rotation_program(
    blocks: int,
    *,
    preserve_seed: bool = False,
    stalled: bool = False,
    valid_rows: int | None = None,
) -> ir.Program:
    saved = f"saved: {_PANEL} = current" if preserve_seed else ""
    observe = (
        f"""
        saved_left: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
            saved, target_memory=pl.Mem.Left)
        saved_result: {_RESULT} = pl.tile.matmul(saved_left, right)
        output = pl.tile.store(saved_result, [0, 0], output)
"""
        if preserve_seed
        else ""
    )
    prefetch = f"block + 1 < {blocks}" + (" and block % 3 != 1" if stalled else "")
    valid = f", valid_shape=[{valid_rows}, 64]" if valid_rows is not None else ""
    return pl.parse_program(
        f"""
@pl.program
class Rotation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[512, 64], pl.FP16],
               b: pl.Tensor[[64, 16], pl.FP16],
               output: pl.Out[pl.Tensor[[512, 16], pl.FP32]]) -> pl.Tensor[[512, 16], pl.FP32]:
        current: {_PANEL} = pl.tile.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat{valid})
        {saved}
        weight: pl.Tile[[64, 16], pl.FP16, pl.Mem.Mat] = pl.tile.load(
            b, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
        right: pl.Tile[[64, 16], pl.FP16, pl.Mem.Right] = pl.tile.move(
            weight, target_memory=pl.Mem.Right)
        for block in pl.range({blocks}):
            if {prefetch}:
                following: {_PANEL} = pl.tile.load(
                    a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat{valid})
            else:
                following: {_PANEL} = current
            left: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
                current, target_memory=pl.Mem.Left)
            result: {_RESULT} = pl.tile.matmul(left, right)
            output = pl.tile.store(result, [block * 64, 0], output)
            current = following
{observe}
        return output
"""
    )


def _codegen(
    program: ir.Program,
    enable_buffer_ir: bool,
    planner: passes.MemoryPlanner = passes.MemoryPlanner.PYPTO,
) -> str:
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    function = next(function for function in optimized.functions.values() if function.name == "kernel")
    return codegen.PTOCodegen().generate(ir.Program([function], "kernel", optimized.span))


def _panel_addresses(text: str, rows: int = 64, cols: int = 64) -> set[int]:
    constants = {
        name: int(value)
        for name, value in re.findall(r"(%\w+) = arith\.constant (\d+) : (?:i64|index)", text)
    }
    addresses = set()
    for line in text.splitlines():
        if "pto.alloc_tile" not in line or "loc=mat" not in line:
            continue
        if f"rows={rows}," not in line or f"cols={cols}," not in line:
            continue
        address = re.search(r"addr\s*=\s*(%\w+)", line)
        assert address is not None, line
        assert address[1] in constants, f"The physical bank address must be constant: {line}"
        addresses.add(constants[address[1]])
    return addresses


def _assert_disjoint_panels(text: str, expected: int = 2, rows: int = 64, cols: int = 64) -> None:
    addresses = sorted(_panel_addresses(text, rows, cols))
    assert len(addresses) == expected, text
    assert all(right - left >= rows * cols * 2 for left, right in zip(addresses, addresses[1:])), text


def _native_ir(text: str, tmp_path: Path, arch: str = "a2") -> str:
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source = tmp_path / "rotation.pto"
    output = tmp_path / "rotation.sync.pto"
    source.write_text(text)
    _run_ptoas(
        str(source),
        str(output),
        ["--pto-level=level3", f"--pto-arch={arch}", "--enable-insert-sync", "--emit-pto-ir"],
    )
    return output.read_text()


def _assert_native_rotation_sync(text: str) -> None:
    # A successful native compile alone cannot catch an overwrite race. Check
    # both load readiness and the consumer-to-next-prefetch loop-back credit.
    loop = text.split("scf.for", 1)[1]
    war_wait = re.search(r"pto\.wait_flag\[<PIPE_MTE1>,\s*<PIPE_MTE2>", loop)
    war_set = re.search(r"pto\.set_flag\[<PIPE_MTE1>,\s*<PIPE_MTE2>", loop)
    raw_set = re.search(r"pto\.set_flag\[<PIPE_MTE2>,\s*<PIPE_MTE1>", loop)
    raw_wait = re.search(r"pto\.wait_flag\[<PIPE_MTE2>,\s*<PIPE_MTE1>", loop)
    assert war_wait and war_set and raw_set and raw_wait, text
    assert war_wait.start() < loop.index("pto.tload"), text
    assert loop.index("pto.tmov") < war_set.start(), text
    assert raw_set.start() < raw_wait.start() < loop.index("pto.tmov"), text


@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
@pytest.mark.parametrize("blocks", [0, 1, 3, 4, 7, 8])
def test_current_following_rotation_keeps_distinct_mat_banks(blocks, enable_buffer_ir):
    """The final alias keeps the current bank; an odd count must not change its meaning."""
    text = _codegen(_rotation_program(blocks), enable_buffer_ir)
    assert not _MAT_TO_MAT.search(text), text
    if blocks > 1:
        _assert_disjoint_panels(text)
        assert re.search(r"arith\.select[^\n]*!pto\.tile_buf<[^\n]*loc=mat", text), text
        # Prefetch loads are observable writes even after the value carry becomes
        # a scalar selector. DCE must not remove the load that fills the next bank.
        assert text.count("pto.tload") >= 3, text  # Seed panel, weight, and next panel.


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("arch", ["a2", "a3"])
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
@pytest.mark.parametrize("blocks", [3, 4])
def test_native_current_following_rotation_preserves_loop_sync(tmp_path, blocks, enable_buffer_ir, arch):
    text = _codegen(_rotation_program(blocks), enable_buffer_ir)
    native = _native_ir(text, tmp_path, arch)
    assert not _MAT_TO_MAT.search(native), native
    _assert_native_rotation_sync(native)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_nonterminal_else_keeps_the_actual_current_bank(tmp_path, enable_buffer_ir):
    """Conditional stalls make the current bank depend on history, not loop parity."""
    text = _codegen(_rotation_program(8, stalled=True), enable_buffer_ir)
    _assert_disjoint_panels(text)
    assert not _MAT_TO_MAT.search(text), text
    _assert_native_rotation_sync(_native_ir(text, tmp_path))


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_dsa_rp_current_following_rotation_preserves_storage_and_sync(tmp_path, enable_buffer_ir):
    text = _codegen(_rotation_program(3), enable_buffer_ir, passes.MemoryPlanner.DSA_RP)
    _assert_disjoint_panels(text)
    assert not _MAT_TO_MAT.search(text), text
    _assert_native_rotation_sync(_native_ir(text, tmp_path))


@pl.program
class Bf16Rotation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        panels: pl.Tensor[[96, 64], pl.BF16],
        weight: pl.Tensor[[64, 16], pl.BF16],
        output: pl.Out[pl.Tensor[[96, 16], pl.FP32]],
    ) -> pl.Tensor[[96, 16], pl.FP32]:
        active = pl.load(panels, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
        rhs = pl.load(weight, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
        right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
        for step in pl.range(3):
            if step + 1 < 3:
                pending = pl.load(panels, [(step + 1) * 32, 0], [32, 64], target_memory=pl.Mem.Mat)
            else:
                pending = active
            left = pl.tile.move(active, target_memory=pl.Mem.Left)
            result = pl.tile.matmul(left, right)
            output = pl.store(result, [step * 32, 0], output)
            active = pending
        return output


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_rotation_handles_other_panel_geometry_dtype_and_variable_names(tmp_path, planner, enable_buffer_ir):
    text = _codegen(Bf16Rotation, enable_buffer_ir, planner)
    assert "dtype=bf16" in text, text
    assert not _MAT_TO_MAT.search(text), text
    _assert_disjoint_panels(text, rows=32, cols=64)
    _assert_native_rotation_sync(_native_ir(text, tmp_path))


@pytest.mark.ptoas_compile
def test_matching_partial_valid_shape_is_preserved_on_every_bank(tmp_path):
    text = _codegen(_rotation_program(3, valid_rows=32), enable_buffer_ir=True)
    _assert_disjoint_panels(text)
    constants = {
        name: int(value)
        for name, value in re.findall(r"(%\w+) = arith\.constant (\d+) : (?:i64|index)", text)
    }
    banks = [
        line
        for line in text.splitlines()
        if "pto.alloc_tile" in line and "loc=mat" in line and "rows=64," in line and "cols=64," in line
    ]
    assert banks, text
    for bank in banks:
        valid_row = re.search(r"valid_row\s*=\s*(%\w+)", bank)
        assert valid_row is not None and constants[valid_row[1]] == 32, bank
    _assert_native_rotation_sync(_native_ir(text, tmp_path))


@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_seed_observed_after_rotation_has_protected_storage(enable_buffer_ir):
    """A saved seed remains immutable while the two scratch banks alternate."""
    text = _codegen(_rotation_program(7, preserve_seed=True), enable_buffer_ir)
    assert not _MAT_TO_MAT.search(text), text
    _assert_disjoint_panels(text, expected=3)


@pl.program
class RepeatedInitializer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[512, 64], pl.FP16],
        b: pl.Tensor[[64, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
    ) -> pl.Tensor[[512, 16], pl.FP32]:
        seed = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
        right = pl.tile.move(weight, target_memory=pl.Mem.Right)
        for repeat in pl.range(2):
            current = seed
            for block in pl.range(4):
                if block + 1 < 4:
                    following = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
                else:
                    following = current
                left = pl.tile.move(current, target_memory=pl.Mem.Left)
                result = pl.tile.matmul(left, right)
                output = pl.store(result, [repeat * 256 + block * 64, 0], output)
                current = following
        return output


@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_one_lexical_seed_use_can_initialize_multiple_rotation_executions(enable_buffer_ir):
    """The outer loop reads the same initializer again after scratch banks were written."""
    text = _codegen(RepeatedInitializer, enable_buffer_ir)
    assert not _MAT_TO_MAT.search(text), text
    _assert_disjoint_panels(text, expected=3)


def _dual_rotation_program() -> ir.Program:
    return pl.parse_program(
        f"""
@pl.program
class DualRotation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[512, 64], pl.FP16],
               c: pl.Tensor[[512, 64], pl.FP16], b: pl.Tensor[[64, 16], pl.FP16],
               output: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
               output2: pl.Out[pl.Tensor[[512, 16], pl.FP32]]) -> tuple[
                   pl.Tensor[[512, 16], pl.FP32], pl.Tensor[[512, 16], pl.FP32]]:
        current: {_PANEL} = pl.tile.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        current2: {_PANEL} = pl.tile.load(c, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        weight: pl.Tile[[64, 16], pl.FP16, pl.Mem.Mat] = pl.tile.load(
            b, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
        right: pl.Tile[[64, 16], pl.FP16, pl.Mem.Right] = pl.tile.move(
            weight, target_memory=pl.Mem.Right)
        for block in pl.range(7):
            if block + 1 < 7:
                following: {_PANEL} = pl.tile.load(
                    a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
                following2: {_PANEL} = pl.tile.load(
                    c, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
            else:
                following: {_PANEL} = current
                following2: {_PANEL} = current2
            left: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
                current, target_memory=pl.Mem.Left)
            result: {_RESULT} = pl.tile.matmul(left, right)
            output = pl.tile.store(result, [block * 64, 0], output)
            left2: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
                current2, target_memory=pl.Mem.Left)
            result2: {_RESULT} = pl.tile.matmul(left2, right)
            output2 = pl.tile.store(result2, [block * 64, 0], output2)
            current = following
            current2 = following2
        return output, output2
"""
    )


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_two_mat_streams_rotate_without_sharing_their_banks(tmp_path, enable_buffer_ir):
    text = _codegen(_dual_rotation_program(), enable_buffer_ir)
    assert not _MAT_TO_MAT.search(text), text
    _assert_disjoint_panels(text, expected=4)
    assert text.count("pto.tload") >= 5, text
    _assert_native_rotation_sync(_native_ir(text, tmp_path))


def _declared_slot_program() -> ir.Program:
    slots = pl.MemRef(slots=2)

    @pl.program
    class DeclaredPrefetch:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[512, 64], pl.FP16],
            b: pl.Tensor[[64, 16], pl.FP16],
            output: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
        ) -> pl.Tensor[[512, 16], pl.FP32]:
            _seed: pl.Tile[[64, 64], pl.FP16, slots[0], pl.Mem.Mat] = pl.load(
                a, [0, 0], [64, 64], target_memory=pl.Mem.Mat
            )
            weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
            right = pl.tile.move(weight, target_memory=pl.Mem.Right)
            for block in pl.range(7):
                if block + 1 < 7:
                    _next: pl.Tile[[64, 64], pl.FP16, slots[(block + 1) % 2], pl.Mem.Mat] = pl.load(
                        a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat
                    )
                current: pl.Tile[[64, 64], pl.FP16, slots[block % 2], pl.Mem.Mat] = pl.tile.create(
                    [64, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat
                )
                left = pl.tile.move(current, target_memory=pl.Mem.Left)
                result = pl.tile.matmul(left, right)
                output = pl.tile.store(result, [block * 64, 0], output)
            return output

    return DeclaredPrefetch


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_declared_dynamic_slots_keep_native_alias_provenance(tmp_path, enable_buffer_ir):
    """Slot addressing must retain all bank roots for native cross-pipe analysis."""
    text = _codegen(_declared_slot_program(), enable_buffer_ir)
    _assert_disjoint_panels(text)
    assert re.search(r"arith\.select[^\n]*!pto\.tile_buf<[^\n]*loc=mat", text), text
    assert text.count("pto.tload") >= 3, text
    _assert_native_rotation_sync(_native_ir(text, tmp_path))


@pl.jit.incore
def _issue_rotate(
    a: pl.Tensor[[512, 64], pl.FP16],
    b: pl.Tensor[[64, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
):
    current = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(8):
        if block + 1 < 8:
            following = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        pl.store(result, [block * 64, 0], out)
        current = following
    return out


@pl.jit
def _issue_entry(
    a: pl.Tensor[[512, 64], pl.FP16],
    b: pl.Tensor[[64, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = _issue_rotate(a, b, out)
    return out


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_issue_3039_original_jit_spmd_compiles_without_device_execution(tmp_path, enable_buffer_ir):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    options = RunConfig(
        backend_type=BackendType.Ascend910,
        execution_mode=ExecutionMode.ONBOARD,
        save_kernels_dir=str(tmp_path / "kernel"),
        dump_passes=True,
    )
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=enable_buffer_ir):
        artifact = _issue_entry.compile(config=options)
    emitted = list(Path(artifact.output_dir).rglob("*.pto"))
    assert emitted, artifact.output_dir
    for source in emitted:
        assert not _MAT_TO_MAT.search(source.read_text()), source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
