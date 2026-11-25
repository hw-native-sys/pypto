# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Automatic A5 L1-to-UB transfer through the default mixed-kernel pipeline."""

import re
import subprocess
import sys

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _program(cols=64, valid_rows=32, setup=()):
    setup_lines = "".join(f"        {line}\n" for line in setup)
    return pl.parse_program(f"""
@pl.program
class Transfer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, data: pl.Tensor[[32, {cols}], pl.INT16], out: pl.Out[pl.Tensor[[32, {cols}], pl.INT16]]
    ) -> pl.Tensor[[32, {cols}], pl.INT16]:
{setup_lines}        l1 = pl.tile.load(
            data, [0, 0], [32, {cols}], [{valid_rows}, {cols}], target_memory=pl.Mem.Mat
        )
        ub = pl.tile.move(l1, pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
        return pl.tile.store(ub, [0, 0], out)
""")


def _lower(program, planner=passes.MemoryPlanner.PYPTO):
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    try:
        with passes.PassContext([], memory_planner=planner):
            lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
            members = [func for func in lowered.functions.values() if ir.is_incore_type(func.func_type)]
            return lowered, codegen.PTOCodegen().generate(
                ir.Program(members, "dma", lowered.span),
                emit_tile_addr=planner == passes.MemoryPlanner.PYPTO,
            )
    finally:
        backend.reset_for_testing()


def _compile(tmp_path, pto, planner, sync):
    compiler = find_ptoas_binary()
    if not compiler:
        pytest.skip("real PTOAS is required")
    check_ptoas_version(compiler)
    path, output = tmp_path / "dma.pto", tmp_path / "dma.cpp"
    path.write_text(pto)
    level = "level3" if planner == passes.MemoryPlanner.PYPTO else "level2"
    result = subprocess.run(
        [
            compiler,
            "--pto-arch=a5",
            f"--pto-level={level}",
            sync,
            str(path),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return output.read_text()


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("cols", [64, 1024])
def test_l1_to_ub_lowering_compiles(tmp_path, planner, cols):
    lowered, pto = _lower(_program(cols=cols), planner)
    ir.assert_structural_equal(lowered, pl.parse_program(lowered.as_python()))
    assert "pto.tdma_l1_to_ub" in pto
    assert "pto.tdma_ub_copy" in pto
    assert "emitc." not in pto
    cpp = _compile(tmp_path, pto, planner, "--enable-insert-sync")
    assert cpp.count("copy_cbuf_to_ubuf(") == 1
    assert "copy_ubuf_to_ubuf(" in cpp
    assert "set_intra_block(PIPE_MTE1" in cpp
    assert "wait_intra_block(PIPE_V" in cpp


def _reuse_program(case):
    def transfer(row, label):
        return [
            f"l1{label} = pl.tile.load(data, [{row}, 0], [32, 64], target_memory=pl.Mem.Mat)",
            f"ub{label} = pl.tile.move(l1{label}, pl.Mem.Vec, "
            "blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)",
            *([f"value{label} = pl.tile.add(ub{label}, ub{label})"] if case == "loop_add" else []),
            f"out = pl.tile.store({'value' if case == 'loop_add' else 'ub'}{label}, [{row}, 0], out)",
        ]

    if case == "consecutive":
        body = ["        " + line for line in transfer(0, "0") + transfer(32, "1")]
    else:
        body = ["        for i in pl.range(2):", *("            " + line for line in transfer("i * 32", ""))]
    return pl.parse_program(
        """
@pl.program
class Transfer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, data: pl.Tensor[[64, 64], pl.INT16], out: pl.Out[pl.Tensor[[64, 64], pl.INT16]]
    ) -> pl.Tensor[[64, 64], pl.INT16]:
"""
        + "\n".join([*body, "        return out", ""])
    )


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("sync", ["--enable-insert-sync", "--enable-bufid_sync"])
@pytest.mark.parametrize("case", ["loop", "loop_add", "consecutive"])
def test_receive_orders_war_before_ub_overwrite(tmp_path, sync, case):
    _, pto = _lower(_reuse_program(case))
    vec_addresses = re.findall(r"pto.alloc_tile addr = (%\w+).*?tile_buf<loc=vec", pto)
    assert len(vec_addresses) == (1 if case == "loop" else 2)
    assert len(set(vec_addresses)) == 1
    cpp = _compile(tmp_path, pto, passes.MemoryPlanner.PYPTO, sync)
    aiv = cpp[cpp.index("kernel_aiv(") :]
    copies = [m.start() for m in re.finditer(r"\bcopy_ubuf_to_ubuf\(", aiv)]
    stores = [m.start() for m in re.finditer(r"\bTSTORE\(", aiv)]
    assert len(copies) == len(stores) == (2 if case == "consecutive" else 1)
    assert copies[0] < stores[0]
    if case == "loop_add":
        assert copies[0] < aiv.index("TADD(") < stores[0]
    if sync == "--enable-insert-sync":
        release = r"set_flag\(PIPE_MTE3, PIPE_V, (EVENT_ID\d+)\)"
        acquire = r"wait_flag\(PIPE_MTE3, PIPE_V, (EVENT_ID\d+)\)"
    else:
        release = r"rls_buf\(PIPE_MTE3, (\d+), \d+\)"
        acquire = r"get_buf\(PIPE_V, (\d+), \d+\)"
    if case == "consecutive":
        before_overwrite = aiv[stores[0] : copies[1]]
        assert any(
            event.group(1) in re.findall(acquire, before_overwrite[event.end() :])
            for event in re.finditer(release, before_overwrite)
        )
    else:
        loop = aiv.index("for (")
        assert set(re.findall(acquire, aiv[loop : copies[0]])) & set(re.findall(release, aiv[stores[0] :]))


def _pipe_program(explicit_base):
    pipe_attrs = "" if explicit_base is None else f", flag_base={explicit_base}"
    init = []
    for pipe in range(2):
        attrs = pipe_attrs if pipe == 1 else ""
        init += [
            f'aic_buffer_{pipe} = pl.system.reserve_buffer(name="aic{pipe}", size=8192)',
            f'aiv_buffer_{pipe} = pl.system.reserve_buffer(name="aiv{pipe}", size=8192)',
            f'peer_aic_{pipe} = pl.system.import_peer_buffer(name="aic{pipe}")',
            f'peer_aiv_{pipe} = pl.system.import_peer_buffer(name="aiv{pipe}")',
            f"pl.system.aic_initialize_pipe(peer_aiv_{pipe}, aic_buffer_{pipe}, "
            f"dir_mask=3, slot_size=2048, id={pipe}{attrs})",
            f"pl.system.aiv_initialize_pipe(aiv_buffer_{pipe}, peer_aic_{pipe}, "
            f"dir_mask=3, slot_size=2048, id={pipe}{attrs})",
        ]
    return _program(setup=init)


@pytest.mark.parametrize(("explicit_base", "events"), [(None, (13, 12)), (10, (9, 8))])
def test_dma_events_count_each_peer_pipe_once_and_skip_explicit_range(explicit_base, events):
    lowered, pto = _lower(_pipe_program(explicit_base))
    ir.assert_structural_equal(lowered, pl.parse_program(lowered.as_python()))
    ready, released = events
    assert f"ready = {ready} : i32, released = {released} : i32" in pto
    assert f"pto.wait_intra_block <PIPE_V>, {ready}" in pto
    if explicit_base is not None:
        assert pto.count(f"flag_base = {explicit_base}") == 2


def test_dma_rejects_unscoped_move_with_split_aiv():
    program = pl.parse_program("""
@pl.program
class Transfer:
    @pl.function
    def kernel(
        self, data: pl.Tensor[[64, 64], pl.INT16], out: pl.Out[pl.Tensor[[64, 64], pl.INT16]]
    ) -> pl.Tensor[[64, 64], pl.INT16]:
        with pl.at(level=pl.Level.CORE_GROUP):
            l1 = pl.tile.load(data, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
            ub = pl.tile.move(l1, pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
            out = pl.tile.store(ub, [0, 0], out)
            another = pl.tile.load(data, [32, 0], [32, 64], target_memory=pl.Mem.Mat)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                shard = pl.aiv_shard(another)
                out = pl.tile.store(shard, [32 + lane * 16, 0], out)
        return out
""")
    with pytest.raises(ValueError, match="tile.move has no defined lane in a dual-AIV mixed kernel"):
        _lower(program)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"cols": 24}, "32-byte-aligned physical row"),
        ({"valid_rows": 16}, "requires full valid extents"),
    ],
)
def test_l1_to_ub_rejects_incompatible_geometry(options, message):
    with pytest.raises(ValueError, match=message):
        _lower(_program(**options))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
