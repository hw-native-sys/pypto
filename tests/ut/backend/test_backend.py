# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Backend, SoC construction, and serialization."""

import pytest
from pypto import DataType, backend, ir, passes
from pypto.backend import (
    Cluster,
    Core,
    Die,
    Mem,
    SoC,
)


class TestSoCConstruction:
    """Tests for direct SoC construction."""

    def test_build_simple_core(self):
        """Test building a simple core with direct construction."""
        core = Core(
            ir.CoreType.CUBE,
            [
                Mem(ir.MemorySpace.Left, 512 * 1024, 64),
                Mem(ir.MemorySpace.Right, 512 * 1024, 64),
            ],
        )

        assert core.core_type == ir.CoreType.CUBE
        assert len(core.mems) == 2
        assert core.mems[0].mem_size == 512 * 1024

    def test_build_cluster_with_cores(self):
        """Test building cluster with multiple cores."""
        core = Core(ir.CoreType.VECTOR, [Mem(ir.MemorySpace.Mat, 1024 * 1024, 128)])

        cluster = Cluster(core, 4)  # 4 identical cores

        assert cluster.total_core_count() == 4

    def test_build_complete_soc_nested(self):
        """Test building complete SoC with nested construction."""
        core = Core(ir.CoreType.CUBE, [Mem(ir.MemorySpace.Left, 256 * 1024, 64)])
        cluster = Cluster(core, 2)
        die = Die(cluster, 4)
        soc = SoC(die, 1)

        assert soc.total_die_count() == 1
        assert soc.total_cluster_count() == 4
        assert soc.total_core_count() == 8

    def test_build_multi_die_soc(self):
        """Test building SoC with multiple dies."""
        # Build a die first
        core = Core(ir.CoreType.VECTOR, [Mem(ir.MemorySpace.Vec, 512 * 1024, 64)])
        cluster = Cluster(core, 2)
        die = Die(cluster, 2)

        # Create SoC with 2 dies
        soc = SoC(die, 2)

        assert soc.total_die_count() == 2
        assert soc.total_cluster_count() == 4
        assert soc.total_core_count() == 8


class TestSoCStructure:
    """Tests for SoC structure and hierarchy."""

    def test_mem_properties(self):
        """Test Mem component properties."""
        mem = Mem(ir.MemorySpace.Acc, 1024 * 1024, 256)

        assert mem.mem_type == ir.MemorySpace.Acc
        assert mem.mem_size == 1024 * 1024
        assert mem.alignment == 256

    def test_core_properties(self):
        """Test Core properties."""
        mems = [Mem(ir.MemorySpace.Left, 512 * 1024, 64), Mem(ir.MemorySpace.Right, 512 * 1024, 64)]
        core = Core(ir.CoreType.CUBE, mems)

        assert core.core_type == ir.CoreType.CUBE
        assert len(core.mems) == 2

    def test_cluster_convenience_constructor(self):
        """Test Cluster convenience constructor."""
        core = Core(ir.CoreType.CUBE, [Mem(ir.MemorySpace.Left, 256 * 1024, 64)])
        cluster = Cluster(core, 4)

        assert cluster.total_core_count() == 4

    def test_die_convenience_constructor(self):
        """Test Die convenience constructor."""
        core = Core(ir.CoreType.CUBE, [Mem(ir.MemorySpace.Left, 256 * 1024, 64)])
        cluster = Cluster(core, 2)
        die = Die(cluster, 3)

        assert die.total_cluster_count() == 3
        assert die.total_core_count() == 6

    def test_soc_convenience_constructor(self):
        """Test SoC convenience constructor."""
        core = Core(ir.CoreType.CUBE, [Mem(ir.MemorySpace.Left, 256 * 1024, 64)])
        cluster = Cluster(core, 2)
        die = Die(cluster, 3)
        soc = SoC(die, 2)

        assert soc.total_die_count() == 2
        assert soc.total_cluster_count() == 6
        assert soc.total_core_count() == 12


@pytest.mark.parametrize("target_type", [backend.BackendType.Ascend910, backend.BackendType.Ascend950])
@pytest.mark.parametrize(
    "configured_type", [None, backend.BackendType.Ascend910, backend.BackendType.Ascend950]
)
def test_select_scratch_query_uses_owning_backend(target_type, configured_type):
    """Registered rules must not accidentally consult the process-global target."""
    backend.reset_for_testing()
    try:
        if configured_type is not None:
            backend.set_backend_type(configured_type)
        target = backend.get_backend_instance(target_type)
        for columns in (16, 128, 256):
            plan = target.plan_op_scratch("tile.sels", DataType.FP16, columns)
            assert plan is not None and plan.status == backend.OpScratchStatus.Required
            assert plan.spec is not None
            expected = (
                (DataType.FP16, 1, columns)
                if target_type == backend.BackendType.Ascend910
                else (DataType.UINT8, 1, 32)
            )
            assert (plan.spec.dtype, plan.spec.rows, plan.spec.cols) == expected
        assert backend.is_backend_configured() == (configured_type is not None)
        if configured_type is not None:
            assert backend.get_backend_type() == configured_type
            assert passes.PassContext([]).get_backend() is backend.get_backend_instance(configured_type)
    finally:
        backend.reset_for_testing()


def test_scratch_query_distinguishes_missing_rule_unknown_and_unsupported():
    target = backend.get_backend_instance(backend.BackendType.Ascend910)
    # A registered operation without migrated rules is not a no-scratch verdict.
    assert target.plan_op_scratch("tile.add") is None
    with pytest.raises(ValueError, match="not registered"):
        target.plan_op_scratch("tile.not_registered")
    with pytest.raises(ValueError, match="not registered"):
        target.plan_op_scratch("tile.matmul_mx")

    for dtype, columns, expected in (
        (None, None, backend.OpScratchStatus.Unknown),
        (DataType.FP16, None, backend.OpScratchStatus.Unknown),
        (DataType.INT8, 128, backend.OpScratchStatus.Unsupported),
        (DataType.FP16, 0, backend.OpScratchStatus.Unsupported),
    ):
        plan = target.plan_op_scratch("tile.sels", dtype, columns)
        assert plan is not None and plan.status == expected
        assert plan.spec is None and plan.reason

    # A5's fixed recipe needs no column count once the source type is known.
    a5 = backend.get_backend_instance(backend.BackendType.Ascend950)
    plan = a5.plan_op_scratch("tile.sels", DataType.FP16)
    assert plan is not None and plan.status == backend.OpScratchStatus.Required


@pytest.mark.parametrize("target_type", [backend.BackendType.Ascend910, backend.BackendType.Ascend950])
@pytest.mark.parametrize(
    "dtype", [DataType.INT8, DataType.UINT8, DataType.FP16, DataType.FP32, DataType.BF16]
)
def test_select_scratch_candidate_dtype_gate(target_type, dtype):
    """BF16 must use TSEL; byte values have neither form on A2/A3."""
    target = backend.get_backend_instance(target_type)
    byte_supported = target_type == backend.BackendType.Ascend950 or dtype not in (
        DataType.INT8,
        DataType.UINT8,
    )
    for name, supported in (
        ("tile.sel", byte_supported),
        ("tile.sels", byte_supported and dtype != DataType.BF16),
    ):
        plan = target.plan_op_scratch(name, dtype, 128)
        assert plan is not None
        assert plan.status == (
            backend.OpScratchStatus.Required if supported else backend.OpScratchStatus.Unsupported
        )
        assert (plan.spec is not None) == supported


def test_select_scratch_geometry_is_a_read_only_snapshot():
    target = backend.get_backend_instance(backend.BackendType.Ascend910)
    plan = target.plan_op_scratch("tile.sel")
    assert plan is not None and plan.spec is not None
    assert (plan.spec.dtype, plan.spec.rows, plan.spec.cols) == (DataType.UINT32, 1, 16)
    with pytest.raises(AttributeError):
        setattr(plan.spec, "cols", 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
