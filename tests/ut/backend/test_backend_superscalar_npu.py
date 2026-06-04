# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the SuperscalarNPU backend (DDR + TREG register file)."""

import pytest
from pypto.backend import BackendSuperscalarNPU, BackendType

from pypto import ir


class TestConstruction:
    """Backend construction and basic properties."""

    def test_singleton(self):
        assert BackendSuperscalarNPU.instance() is BackendSuperscalarNPU.instance()

    def test_type_name(self):
        assert BackendSuperscalarNPU.instance().get_type_name() == "SuperscalarNPU"

    def test_backend_type_enum_exists(self):
        assert hasattr(BackendType, "SuperscalarNPU")


class TestSoC:
    """SoC memory model: DDR + a single 1MB TREG register file."""

    def test_single_core(self):
        soc = BackendSuperscalarNPU.instance().soc
        assert soc.total_die_count() == 1
        assert soc.total_cluster_count() == 1
        assert soc.total_core_count() == 1

    def test_treg_size_is_one_megabyte(self):
        # 256 fixed 4KB blocks = 1 MiB.
        assert BackendSuperscalarNPU.instance().get_mem_size(ir.MemorySpace.TREG) == 256 * 4096

    def test_ddr_to_treg_path(self):
        be = BackendSuperscalarNPU.instance()
        assert be.find_mem_path(ir.MemorySpace.DDR, ir.MemorySpace.TREG) == [
            ir.MemorySpace.DDR,
            ir.MemorySpace.TREG,
        ]

    def test_no_ascend_spaces(self):
        # SuperscalarNPU has neither vector nor matrix buffers.
        be = BackendSuperscalarNPU.instance()
        assert be.get_mem_size(ir.MemorySpace.Vec) == 0
        assert be.get_mem_size(ir.MemorySpace.Mat) == 0


class TestHandler:
    """BackendHandler defaults for SuperscalarNPU."""

    def test_default_on_chip_space_is_treg(self):
        handler = BackendSuperscalarNPU.instance().get_handler()
        assert handler.get_default_on_chip_memory_space() == ir.MemorySpace.TREG

    def test_no_cube_or_cross_core_workarounds(self):
        handler = BackendSuperscalarNPU.instance().get_handler()
        assert handler.requires_gm_pipe_buffer() is False
        assert handler.requires_vto_c_fractal_adapt() is False
        assert handler.get_l0a_capacity_bytes() == 0
        assert handler.get_l0c_capacity_bytes() == 0


class TestMemorySpaceEnum:
    """The TREG memory space is exposed and round-trips."""

    def test_treg_value_and_alias(self):
        assert ir.MemorySpace.TREG is ir.Mem.TREG
        assert ir.MemorySpace.TREG.name == "TREG"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
