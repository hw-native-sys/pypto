# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the HTML memory map tool."""

import json
import re
from pathlib import Path

import pytest
from pypto import backend, ir
from pypto.tools import memory_map

# A minimal pass dump: one AIC function with two Mat tiles that reuse the same
# base at disjoint lifetimes, one Acc phi chain (alias merge), one Acc view, and
# an Orchestration function that must be skipped.
DUMP = """# pypto.program: _jit_demo
import pypto.language as pl


@pl.program
class _jit_demo:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def demo_aic(
        x__ssa_v0: pl.Tensor[[16, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        mem_mat_1: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_acc_2: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 8192)
        a__tile: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_mat_1, pl.const(0, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(x__ssa_v0)
        acc__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_2, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create(a__tile)
        acc__phi: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_2, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(acc__tile)
        acc_half__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_acc_2, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.slice(acc__phi)
        b__tile: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_mat_1, pl.const(0, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(acc_half__tile)
        return b__tile

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def main(y: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
        return y
"""

# Two Mat bases both still at offset 0 — what every dump before AllocateMemoryAddr looks like.
UNALLOCATED_DUMP = """# pypto.program: _jit_demo
import pypto.language as pl


@pl.program
class _jit_demo:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def demo_aic(x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
        mem_mat_1: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_mat_2: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        a__tile: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_mat_1, pl.const(0, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(x)
        b__tile: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_mat_2, pl.const(0, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.add(a__tile, a__tile)
        return b__tile
"""


@pytest.fixture
def case(tmp_path: Path) -> Path:
    """Lay out a build_output-like case directory and return its root."""
    dump_dir = tmp_path / "passes_dump"
    dump_dir.mkdir()
    (dump_dir / "32_after_AllocateMemoryAddr.py").write_text(DUMP)
    return tmp_path


def _add_ptoas(case: Path, arch: str) -> None:
    """Drop a stub .pto carrying the target arch the detector reads."""
    ptoas = case / "ptoas"
    ptoas.mkdir(exist_ok=True)
    (ptoas / "demo.pto").write_text(f'module attributes {{pto.target_arch = "{arch}"}} {{\n}}\n')


def _dump_of(case: Path) -> Path:
    return case / "passes_dump" / "32_after_AllocateMemoryAddr.py"


def _boxes_by_name(functions: list[memory_map.FunctionMap]) -> dict[str, memory_map.Box]:
    return {b.name: b for f in functions for b in f.boxes}


def test_parse_dump_skips_non_compute_functions(case: Path):
    functions = memory_map.parse_dump(_dump_of(case))
    assert [f.name for f in functions] == ["demo_aic"]
    assert functions[0].ftype == "AIC"


def test_parse_dump_captures_source_slice(case: Path):
    function = memory_map.parse_dump(_dump_of(case))[0]
    lines = DUMP.splitlines()
    # src_start points at the decorator, and the slice is 1-indexed by line.
    assert lines[function.src_start - 1].lstrip().startswith("@pl.function")
    assert function.source[0] == lines[function.src_start - 1]
    assert len(function.source) == function.src_end - function.src_start + 1


def test_tile_metadata_is_extracted(case: Path):
    box = _boxes_by_name(memory_map.parse_dump(_dump_of(case)))["a__tile"]
    assert (box.space, box.base, box.offset, box.size) == ("Mat", "mem_mat_1", 0, 4096)
    assert box.shape == [16, 128]
    assert box.dtype == "BF16"
    assert box.op == "tile.load"


def test_ddr_tensors_are_excluded(case: Path):
    # The x__ssa_v0 parameter carries a DDR MemRef; only tile memory is mapped.
    assert "x__ssa_v0" not in _boxes_by_name(memory_map.parse_dump(_dump_of(case)))


def test_touching_aliases_merge_into_one_box(case: Path):
    boxes = _boxes_by_name(memory_map.parse_dump(_dump_of(case)))
    assert "acc__phi" not in boxes  # folded into acc__tile
    assert boxes["acc__tile"].aliases == ["acc__phi"]


def test_disjoint_lifetimes_on_one_slot_stay_separate(case: Path):
    boxes = _boxes_by_name(memory_map.parse_dump(_dump_of(case)))
    # a__tile and b__tile share mem_mat_1 [0, 4096) but never overlap: reuse.
    assert boxes["a__tile"].base == boxes["b__tile"].base
    assert boxes["a__tile"].end < boxes["b__tile"].start
    assert not boxes["a__tile"].conflict and not boxes["b__tile"].conflict


def test_same_base_subrange_is_a_view_not_a_conflict(case: Path):
    boxes = _boxes_by_name(memory_map.parse_dump(_dump_of(case)))
    assert boxes["acc_half__tile"].view
    assert not boxes["acc_half__tile"].conflict
    assert not boxes["acc__tile"].view


def test_cross_base_overlap_is_flagged_as_conflict():
    def tile(name: str, base: str, offset: int, start: int, end: int) -> memory_map.Tile:
        return memory_map.Tile(
            name=name,
            space="Vec",
            base=base,
            offset=offset,
            size=1024,
            shape=[8, 32],
            dtype="FP32",
            op="tile.create",
            start=start,
            end=end,
        )

    boxes = {
        b.name: b
        for b in memory_map.build_boxes(
            [
                tile("lhs", "mem_vec_1", 0, 10, 20),
                tile("rhs", "mem_vec_2", 512, 15, 25),
            ]
        )
    }
    assert boxes["lhs"].conflict and boxes["rhs"].conflict
    assert not boxes["lhs"].view and not boxes["rhs"].view


def test_backend_limits_come_from_the_backend_interface():
    instance = backend.get_backend_instance(backend.BackendType.Ascend910B)
    limits = memory_map.backend_limits("Ascend910B")
    assert limits["Vec"] == instance.get_mem_size(ir.MemorySpace.Vec)
    assert limits["Mat"] == instance.get_mem_size(ir.MemorySpace.Mat)
    assert "DDR" not in limits  # off-chip; never mapped
    with pytest.raises(ValueError, match="unknown backend"):
        memory_map.backend_limits("Ascend000")


def test_backend_limits_cover_every_space_the_soc_describes():
    # Walking the SoC rather than a fixed space list is what keeps a backend
    # that grew a space mapped: Ascend950 has Bias, Ascend910B does not.
    for name in memory_map.backend_names():
        instance = memory_map.backend_instance(name)
        expected = {
            mem.mem_type.name: mem.mem_size
            for die in instance.soc.die_counts
            for cluster in die.cluster_counts
            for core in cluster.core_counts
            for mem in core.mems
            if mem.mem_type != ir.MemorySpace.DDR
        }
        assert memory_map.backend_limits(name) == expected

    assert "Bias" in memory_map.backend_limits("Ascend950")
    assert "Bias" not in memory_map.backend_limits("Ascend910B")
    # Every space a backend can report has a panel slot, so none falls back to
    # its own high-water mark and renders permanently full.
    for name in memory_map.backend_names():
        assert set(memory_map.backend_limits(name)) <= set(memory_map.SPACE_ORDER)


def test_backend_is_detected_from_the_case_target_arch(case: Path):
    _add_ptoas(case, "a5")
    choice = memory_map.resolve_backend(_dump_of(case))
    assert (choice.name, choice.arch, choice.detected) == ("Ascend950", "a5", True)


def test_backend_falls_back_when_no_ptoas_output_exists(case: Path):
    choice = memory_map.resolve_backend(_dump_of(case))
    assert choice.name == memory_map.DEFAULT_BACKEND
    assert choice.detected is False


def test_explicit_backend_overrides_detection(case: Path):
    _add_ptoas(case, "a5")
    choice = memory_map.resolve_backend(_dump_of(case), "Ascend910B")
    assert (choice.name, choice.detected) == ("Ascend910B", True)
    with pytest.raises(ValueError, match="unknown backend"):
        memory_map.resolve_backend(_dump_of(case), "Ascend000")


def test_render_scales_panels_by_the_backend_capacity(case: Path):
    payload = _payload(memory_map.render(_dump_of(case), "Ascend910B"))
    assert payload["backend"] == {"name": "Ascend910B", "arch": "a2a3", "detected": True}

    spaces = {s["space"]: s for s in payload["functions"][0]["spaces"]}
    expected = memory_map.backend_limits("Ascend910B")
    assert spaces["Mat"]["limit"] == expected["Mat"]
    assert spaces["Mat"]["hwm"] == 4096
    assert spaces["Acc"]["bases"] == 1


def test_render_emits_self_contained_page(case: Path):
    page = memory_map.render(_dump_of(case), None)
    assert memory_map._DATA_PLACEHOLDER not in page
    assert "</script>" not in page.split("<script>")[1].split("</script>")[0]
    assert "http://" not in page and "https://" not in page
    assert _payload(page)["pass_name"] == "AllocateMemoryAddr"


def test_render_rejects_a_dump_without_tile_memrefs(tmp_path: Path):
    dump = tmp_path / "00_frontend.py"
    dump.write_text("import pypto.language as pl\n")
    with pytest.raises(ValueError, match="AllocateMemoryAddr"):
        memory_map.render(dump, None)


def test_render_rejects_a_dump_whose_addresses_are_unassigned(tmp_path: Path):
    # Before AllocateMemoryAddr every MemRef sits at offset 0; drawing that
    # would collapse all bases onto address 0 and flag them all as conflicts.
    dump = tmp_path / "30_after_MemoryReuse.py"
    dump.write_text(UNALLOCATED_DUMP)
    with pytest.raises(ValueError, match="offset 0"):
        memory_map.render(dump, None)


def test_one_base_per_space_at_offset_zero_is_a_real_layout():
    def tile(name: str, space: str, base: str) -> memory_map.Tile:
        return memory_map.Tile(
            name=name,
            space=space,
            base=base,
            offset=0,
            size=1024,
            shape=[8, 32],
            dtype="FP32",
            op="tile.create",
            start=1,
            end=2,
        )

    single = memory_map.FunctionMap(
        name="tiny",
        ftype="AIC",
        src_start=1,
        src_end=2,
        source=[],
        boxes=memory_map.build_boxes([tile("a", "Vec", "mem_vec_1"), tile("b", "Mat", "mem_mat_1")]),
    )
    assert not memory_map.is_unallocated(single)


def test_resolve_dump_accepts_file_case_dir_and_passes_dir(case: Path):
    expected = _dump_of(case)
    assert memory_map.resolve_dump(expected, "AllocateMemoryAddr") == expected
    assert memory_map.resolve_dump(case, "AllocateMemoryAddr") == expected
    assert memory_map.resolve_dump(case / "passes_dump", "AllocateMemoryAddr") == expected
    with pytest.raises(FileNotFoundError, match="MemoryReuse"):
        memory_map.resolve_dump(case, "MemoryReuse")


def test_main_writes_html_next_to_the_dump(case: Path, capsys: pytest.CaptureFixture[str]):
    assert memory_map.main([str(case)]) == 0
    output = case / "passes_dump" / "32_after_AllocateMemoryAddr.memory_map.html"
    assert output.is_file()
    assert "Memory Map" in output.read_text()
    assert str(output) in capsys.readouterr().out


def test_main_reports_a_missing_dump(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert memory_map.main([str(tmp_path)]) == 1
    assert "no '*_after_AllocateMemoryAddr.py' dump" in capsys.readouterr().err


def _payload(page: str) -> dict:
    """Pull the injected JSON back out of a rendered page."""
    match = re.search(r"const DATA = (.*?);\n", page, re.S)
    assert match is not None, "rendered page has no DATA payload"
    return json.loads(match.group(1).replace("<\\/", "</"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
