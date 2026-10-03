# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end PTOAS contracts; detailed scheduling and alias proofs live in UTs."""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto.runtime import RunConfig

from .software_pipeline_cases import make_epilogue, make_mixed, make_nested


def _compile(kernel, output: Path, software: bool = True) -> dict[str, tuple[str, str]]:
    kernel.compile(
        config=RunConfig(
            platform="a2a3",
            enable_software_pipeline=software,
            codegen_only=True,
            save_kernels=True,
            save_kernels_dir=str(output),
        )
    )
    sources = list((output / "ptoas").glob("*.pto"))
    assert sources, "Compilation must reach PTOAS"
    assert all(path.with_suffix(".cpp").is_file() for path in sources)
    return {path.stem: (path.read_text(), path.with_suffix(".cpp").read_text()) for path in sources}


def _single(kernel, output: Path, software: bool = True) -> tuple[str, str]:
    sources = _compile(kernel, output, software)
    assert len(sources) == 1
    return next(iter(sources.values()))


def _vec_add(stage: int, trips: int):
    @pl.jit
    def kernel(x: pl.Tensor[[trips, 1024], pl.FP32], y: pl.Out[pl.Tensor[[trips, 1024], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            for i in pl.pipeline(trips, stage=stage):
                a = x[i : i + 1, :]
                y[i : i + 1, :] = pl.add(a, 1.0)
        return y

    return kernel


@pytest.mark.parametrize(
    ("stage", "trips", "software"),
    [
        (3, 65, False),
        (2, 1, True),
        (3, 1, True),
        (3, 2, True),
        (3, 3, True),
        (3, 65, True),
        (4, 3, True),
        (4, 65, True),
    ],
)
def test_jit_pipeline_compiles_with_ptoas(tmp_path, stage, trips, software):
    """Default, fallback, exact preload, first steady iteration and wrapping."""
    pto, cpp = _single(_vec_add(stage, trips), tmp_path, software)
    expect_slots = software and trips >= stage - 1
    assert ("pto.alloc_multi_tile" in pto) == expect_slots
    if expect_slots:
        assert "pto.alloc_multi_tile addr =" in pto
        assert f"count={stage}" in pto
    assert all(op in pto for op in ("pto.tload", "pto.tadds", "pto.tstore"))
    if software and trips > stage:
        assert "arith.remui" in pto
        steady = re.search(r"for \(int64_t .*?\n  \}", cpp, re.DOTALL)
        assert steady and "pipe_barrier(PIPE_ALL)" not in steady[0]


@pytest.mark.parametrize("stage", [2, 3, 4])
def test_binary_chain_reuses_an_input_with_standard_ptoas(tmp_path, stage):
    @pl.jit
    def kernel(
        x: pl.Tensor[[65, 1024], pl.FP32],
        z: pl.Tensor[[65, 1024], pl.FP32],
        y: pl.Out[pl.Tensor[[65, 1024], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            for i in pl.pipeline(65, stage=stage):
                a = x[i : i + 1, :]
                b = z[i : i + 1, :]
                y[i : i + 1, :] = pl.add(pl.add(a, b), 1.0)
        return y

    pto, cpp = _single(kernel, tmp_path)
    assert pto.count("pto.alloc_multi_tile") == 2
    steady = re.search(r"for \(int64_t .*?\n  \}", cpp, re.DOTALL)
    assert steady
    assert "TADD(" in steady[0] and "TADDS(" in steady[0]


@pytest.mark.parametrize("trips", [1, 3, 65])
def test_score_epilogue_keeps_typed_reuse_and_workspace(tmp_path, trips):
    pto, _ = _single(make_epilogue(trips), tmp_path)
    assert pto.count("pto.alloc_multi_tile") == (0 if trips < 2 else 2)
    if trips > 3:
        assert "arith.remui" in pto and "pto.treshape" in pto
        assert "v_row=64, v_col=64" in pto
    assert "pto.trowsum" in pto


@pytest.mark.parametrize(("stage", "fifo_slots"), [(2, 3), (3, 3), (4, 3), (3, 2)])
def test_mixed_pipeline_uses_standard_fifo_and_local_slots(tmp_path, stage, fifo_slots):
    sources = _compile(make_mixed(stage, fifo_slots), tmp_path)
    pto, _ = next(value for name, value in sources.items() if "mixed" in name)
    assert "pto.talloc_to_aiv" in pto and "pto.tpush_to_aiv" in pto
    aiv = pto.split("pto.kernel_kind<vector>", 1)[1]
    assert "pto.alloc_multi_tile" not in aiv
    assert "pto.alloc_tile addr =" in aiv and "pto.subview" in aiv
    assert "arith.remui" in aiv and aiv.count("scf.for") == 1
    assert aiv.count("pto.tpop_from_aic") == min(stage, fifo_slots)
    assert "gm_slot_tensor =" in aiv and "pto.reserve_buffer" not in aiv


def test_flag_preserves_the_dependent_task(tmp_path):
    enabled = _compile(make_mixed(), tmp_path / "enabled")
    disabled = _compile(make_mixed(), tmp_path / "disabled", False)
    mixed = next(name for name in enabled if "mixed" in name)
    dependent = next(name for name in enabled if "dependent" in name)
    assert "gm_slot_tensor =" in enabled[mixed][0]
    assert "gm_slot_tensor =" not in disabled[mixed][0]
    assert enabled[mixed][1] != disabled[mixed][1]
    assert enabled[dependent][1] == disabled[dependent][1]


@pytest.mark.parametrize("stage", [2, 3, 4])
def test_runtime_bound_pipeline_keeps_one_guarded_loop(tmp_path, stage):
    @pl.jit
    def kernel(
        x: pl.Tensor[[128, 64], pl.FP32],
        n: pl.Scalar[pl.INDEX],
        y: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            for i in pl.pipeline(2, n, 2, stage=stage):
                a = pl.load(x, [i, 0], [1, 64])
                y = pl.store(pl.add(a, 1.0), [i, 0], y)
        return y

    pto, _ = _single(kernel, tmp_path)
    assert "pto.alloc_multi_tile" not in pto
    assert "pto.alloc_tile addr =" in pto
    assert pto.count("scf.for") == 1 and "scf.if" in pto


def test_runtime_score_bound_avoids_unreachable_fallback_storage(tmp_path):
    pto, _ = _single(make_epilogue(dynamic=True), tmp_path)
    assert "pto.alloc_multi_tile" not in pto and "pto.alloc_tile addr =" in pto
    assert pto.count("scf.for") == 1
    assert "pto.verified_" not in pto and "pto.trowsum" in pto


@pytest.mark.parametrize(
    ("outer_stage", "children", "inner_stage", "leaves", "leaf_stage"),
    [(3, 5, 2, 0, 4), (3, 2, 3, 2, 4), (2, 1, 4, 0, 4)],
)
def test_nested_versions_reach_ptoas_as_parent_buffer_views(
    tmp_path,
    outer_stage,
    children,
    inner_stage,
    leaves,
    leaf_stage,
):
    pto, _ = _single(
        make_nested(
            children=children,
            outer_stage=outer_stage,
            inner_stage=inner_stage,
            leaves=leaves,
            leaf_stage=leaf_stage,
        ),
        tmp_path,
    )
    assert "pto.alloc_multi_tile" not in pto
    assert "pto.alloc_tile addr =" in pto and "arith.remui" in pto
    assert "pto.subview" in pto
    assert pto.count("scf.for") == 1
    assert "pto.verified_nested_pipeline" not in pto
    assert pto.count("pto.tstore ") == children * max(leaves, 1)


def test_nested_reduction_cannot_alias_input_or_workspace(tmp_path):
    @pl.jit
    def kernel(x: pl.Tensor[[256, 16], pl.FP32], y: pl.Out[pl.Tensor[[256, 1], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(2, stage=2):
                    row = g * 64 + p * 32
                    value = pl.load(x, [row, 0], [32, 16])
                    scratch = pl.create_tile([32, 16], dtype=pl.FP32)
                    result = pl.row_sum(value, scratch)
                    pl.store(result, [row, 0], y)
        return y

    pto, _ = _single(kernel, tmp_path)
    assert "pto.alloc_multi_tile" not in pto and "pto.alloc_tile addr =" in pto
    assert pto.count("pto.trowsum") == 2


def test_nested_partial_result_view_keeps_compatible_storage(tmp_path):
    @pl.jit
    def kernel(x: pl.Tensor[[128, 64], pl.FP32], y: pl.Out[pl.Tensor[[128, 32], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            for g in pl.pipeline(4, stage=3):
                for p in pl.pipeline(2, stage=2):
                    row = g * 32 + p * 16
                    value = pl.load(x, [row, 0], [16, 64])
                    result = pl.add(value, 1.0)
                    pl.store(pl.slice(result, [16, 32], [0, 0]), [row, 0], y)
        return y

    pto, _ = _single(kernel, tmp_path)
    assert "arith.remui" in pto and "pto.subview" in pto
    assert pto.count("pto.tadds") == 2


def test_nested_conditional_ancestors_preserve_guarded_accesses(tmp_path):
    pto, _ = _single(
        make_nested(
            children=2,
            inner_stage=2,
            leaves=2,
            leaf_stage=2,
            guarded=True,
        ),
        tmp_path,
    )
    assert "pto.alloc_multi_tile" not in pto and "pto.alloc_tile addr =" in pto
    assert pto.count("pto.tload ") == pto.count("pto.tstore ") == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
