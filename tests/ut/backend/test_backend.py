# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Backend, SoC construction, and serialization."""

from pypto import ir
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
                Mem(ir.MemorySpace.L0A, 512 * 1024, 64),
                Mem(ir.MemorySpace.L0B, 512 * 1024, 64),
            ],
        )

        assert core.core_type == ir.CoreType.CUBE
        assert len(core.mems) == 2
        assert core.mems[0].mem_size == 512 * 1024

    def test_build_cluster_with_cores(self):
        """Test building cluster with multiple cores."""
        core = Core(ir.CoreType.VECTOR, [Mem(ir.MemorySpace.L1, 1024 * 1024, 128)])

        cluster = Cluster(core, 4)  # 4 identical cores

        assert cluster.total_core_count() == 4

    def test_build_complete_soc_nested(self):
        """Test building complete SoC with nested construction."""
        core = Core(ir.CoreType.CUBE, [Mem(ir.MemorySpace.L0A, 256 * 1024, 64)])
        cluster = Cluster(core, 2)
        die = Die(cluster, 4)
        soc = SoC(die, 1)

        assert soc.total_die_count() == 1
        assert soc.total_cluster_count() == 4
        assert soc.total_core_count() == 8

    def test_build_multi_die_soc(self):
        """Test building SoC with multiple dies."""
        # Build a die first
        core = Core(ir.CoreType.VECTOR, [Mem(ir.MemorySpace.UB, 512 * 1024, 64)])
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
        mem = Mem(ir.MemorySpace.L0C, 1024 * 1024, 256)

        assert mem.mem_type == ir.MemorySpace.L0C
        assert mem.mem_size == 1024 * 1024
        assert mem.alignment == 256

    def test_core_properties(self):
        """Test Core properties."""
        mems = [Mem(ir.MemorySpace.L0A, 512 * 1024, 64), Mem(ir.MemorySpace.L0B, 512 * 1024, 64)]
        core = Core(ir.CoreType.CUBE, mems)

        assert core.core_type == ir.CoreType.CUBE
        assert len(core.mems) == 2

    def test_cluster_convenience_constructor(self):
        """Test Cluster convenience constructor."""
        core = Core(ir.CoreType.CUBE, [Mem(ir.MemorySpace.L0A, 256 * 1024, 64)])
        cluster = Cluster(core, 4)

        assert cluster.total_core_count() == 4

    def test_die_convenience_constructor(self):
        """Test Die convenience constructor."""
        core = Core(ir.CoreType.CUBE, [Mem(ir.MemorySpace.L0A, 256 * 1024, 64)])
        cluster = Cluster(core, 2)
        die = Die(cluster, 3)

        assert die.total_cluster_count() == 3
        assert die.total_core_count() == 6

    def test_soc_convenience_constructor(self):
        """Test SoC convenience constructor."""
        core = Core(ir.CoreType.CUBE, [Mem(ir.MemorySpace.L0A, 256 * 1024, 64)])
        cluster = Cluster(core, 2)
        die = Die(cluster, 3)
        soc = SoC(die, 2)

        assert soc.total_die_count() == 2
        assert soc.total_cluster_count() == 6
        assert soc.total_core_count() == 12
