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
from pypto import passes
from pypto.backend import BackendType, pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import ExecutionMode, RunConfig

from tests.software_pipeline_cases import make_epilogue

pytestmark = [
    pytest.mark.ptoas_compile,
    pytest.mark.skipif(find_ptoas_binary() is None, reason="PTOAS is not available"),
]


def _read_single(output: Path) -> tuple[str, str]:
    """Read the PTO and native C++ artifacts of exactly one compiled kernel."""
    sources = list((output / "ptoas").glob("*.pto"))
    assert len(sources) == 1, "Compilation must produce exactly one in-core kernel"
    source = sources[0]
    native = source.with_suffix(".cpp")
    assert native.is_file(), "Compilation must reach PTOAS"
    return source.read_text(), native.read_text()


def _single(kernel, output: Path, software: bool = True) -> tuple[str, str]:
    """Compile one kernel through PTOAS and return its PTO and native C++."""
    # UTs already install a verification context. Keep compilation controls in
    # a nested context, retaining the default Buffer IR path when the option is off.
    with passes.PassContext([], enable_buffer_ir=not software, enable_software_pipeline=software):
        kernel.compile(
            config=RunConfig(
                backend_type=BackendType.Ascend910,
                execution_mode=ExecutionMode.ONBOARD,
                codegen_only=True,
                save_kernels=True,
                save_kernels_dir=str(output),
            )
        )
    return _read_single(output)


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
    expect_region = software and trips == stage - 1
    expect_views = software and trips > stage - 1
    assert ("pto.alloc_multi_tile" in pto) == expect_region
    assert ("pto.subview" in pto) == expect_views
    if expect_region:
        assert "pto.alloc_multi_tile addr =" in pto
        assert f"count={stage}" in pto
    if expect_views:
        assert "pto.alloc_tile addr =" in pto and "pto.multi_tile_get" not in pto
    assert all(op in pto for op in ("pto.tload", "pto.tadds", "pto.tstore"))
    if software and trips > stage:
        assert "arith.remui" in pto
        steady = re.search(r"for \(int64_t .*?\n  \}", cpp, re.DOTALL)
        # Stock PTOAS may conservatively synchronize dynamic parent subviews.
        assert steady and "TLOAD(" in steady[0] and "TADDS(" in steady[0]


@pytest.mark.parametrize("trips", [2, 65])
def test_separate_lower_and_codegen_preserve_slot_storage(tmp_path, trips):
    """Lowered IR must carry its slot contract, including schedules without a steady loop."""
    kernel = _vec_add(3, trips)
    with passes.PassContext([], enable_buffer_ir=False, enable_software_pipeline=True):
        lowered = kernel.lower(
            config=RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD)
        )
    pto_backend.generate(lowered, str(tmp_path))
    pto, _ = _read_single(tmp_path)
    assert ("pto.alloc_multi_tile" in pto) == (trips == 2)
    assert ("pto.subview" in pto) == (trips == 65)


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
    assert "pto.alloc_multi_tile" not in pto and "pto.multi_tile_get" not in pto
    assert pto.count("pto.alloc_tile addr =") == 2 and "pto.subview" in pto
    steady = re.search(r"for \(int64_t .*?\n  \}", cpp, re.DOTALL)
    assert steady
    assert "TADD(" in steady[0] and "TADDS(" in steady[0]


@pytest.mark.parametrize("trips", [1, 3, 65])
def test_score_epilogue_keeps_typed_reuse_and_workspace(tmp_path, trips):
    pto, _ = _single(make_epilogue(trips), tmp_path)
    assert "pto.alloc_multi_tile" not in pto
    assert ("pto.subview" in pto) == (trips >= 3)
    if trips > 3:
        assert "arith.remui" in pto and "pto.treshape" in pto
        assert "v_row=64, v_col=64" in pto
    assert "pto.trowsum" in pto


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
