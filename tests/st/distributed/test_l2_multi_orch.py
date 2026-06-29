# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Multi-Orchestration L2-only program: codegen lands each orch in its own sub-dir.

The program defines two independent L2 ``Orchestration`` functions
(``chip_orch_add`` / ``chip_orch_sub``) with no L3 HOST wrapper. The
top-level :func:`pypto.backend.pto_backend.generate` dispatcher detects
``orch_count > 1`` and falls through to :func:`_generate_multi_chip`,
which emits each orch as a self-contained sub-build under
``next_levels/{orch_name}/`` — the same layout the chip tier of L3+
programs uses, just without ``host_orch.py`` / ``sub_workers/``.

This file currently covers the codegen path; execution dispatch lives in
``CompiledProgram`` and is exercised separately once the multi-orch call
API lands.
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir


@pl.program
class TwoL2AddSubProgram:
    """Two L2 Orchestration functions sharing the same I/O signature."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        ta = pl.load(a, [0, 0], [128, 128])
        tb = pl.load(b, [0, 0], [128, 128])
        tf = pl.add(ta, tb)
        return pl.store(tf, [0, 0], f)

    @pl.function(type=pl.FunctionType.InCore)
    def tile_sub(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        ta = pl.load(a, [0, 0], [128, 128])
        tb = pl.load(b, [0, 0], [128, 128])
        tf = pl.sub(ta, tb)
        return pl.store(tf, [0, 0], f)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        return self.tile_add(a, b, f)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch_sub(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        return self.tile_sub(a, b, f)


class TestL2MultiOrch:
    def test_codegen_emits_per_orch_subdirs(self, test_config, tmp_path):
        """Each L2 Orchestration produces its own ``next_levels/<name>/`` sub-build."""
        ir.compile(
            TwoL2AddSubProgram,
            output_dir=str(tmp_path),
            platform=test_config.platform,
            skip_ptoas=True,
        )

        add_dir = tmp_path / "next_levels" / "chip_orch_add"
        sub_dir = tmp_path / "next_levels" / "chip_orch_sub"

        # Each orch gets its own orchestration C++ and the kernels it transitively calls.
        assert (add_dir / "orchestration" / "chip_orch_add.cpp").is_file(), (
            f"chip_orch_add orchestration source missing under {add_dir}"
        )
        assert (sub_dir / "orchestration" / "chip_orch_sub.cpp").is_file(), (
            f"chip_orch_sub orchestration source missing under {sub_dir}"
        )

        # Kernels are filtered per orch (chip_orch_add only sees tile_add; chip_orch_sub only tile_sub).
        add_kernels = {p.name for p in (add_dir / "kernels").rglob("*.pto")}
        sub_kernels = {p.name for p in (sub_dir / "kernels").rglob("*.pto")}
        assert add_kernels == {"tile_add.pto"}, f"chip_orch_add kernels = {add_kernels}"
        assert sub_kernels == {"tile_sub.pto"}, f"chip_orch_sub kernels = {sub_kernels}"

        # No flat-layout orchestration at the root — those would be the old single-chip
        # path's output and indicate the multi-orch dispatcher didn't fire.
        assert not (tmp_path / "orchestration").exists(), (
            "root orchestration/ directory should not exist for multi-orch programs"
        )

    def test_dispatch_surface(self, test_config, tmp_path):
        """``CompiledProgram`` exposes each L2 orch via subscript and attribute access.

        Verifies the dispatch wiring without invoking the runtime: the
        sub-callables are bound to their sub-build directories and carry
        the right per-orch parameter metadata. Plain ``compiled(...)``
        must refuse to guess which orch to run.
        """
        compiled = ir.compile(
            TwoL2AddSubProgram,
            output_dir=str(tmp_path),
            platform=test_config.platform,
            skip_ptoas=True,
        )

        # 1) Inspection surface
        assert compiled.orchestration_names == ["chip_orch_add", "chip_orch_sub"]

        # 2) Subscript and attribute lookup return distinct sub-callables
        add_via_sub = compiled["chip_orch_add"]
        add_via_attr = compiled.chip_orch_add
        sub_via_sub = compiled["chip_orch_sub"]
        assert add_via_sub.name == "chip_orch_add"
        assert add_via_attr.name == "chip_orch_add"
        assert sub_via_sub.name == "chip_orch_sub"
        assert add_via_sub.output_dir == tmp_path / "next_levels" / "chip_orch_add"
        assert sub_via_sub.output_dir == tmp_path / "next_levels" / "chip_orch_sub"

        # 3) Per-orch metadata is correct (both orchs share the same signature here)
        assert add_via_sub.param_names == ["a", "b", "f"]
        assert sub_via_sub.param_names == ["a", "b", "f"]

        # 4) Plain compiled(...) refuses — user must select an orch explicitly
        a = torch.zeros((128, 128), dtype=torch.float32)
        with pytest.raises(TypeError, match="select one explicitly"):
            compiled(a, a, a)

        # 5) Unknown orch name raises KeyError listing what *is* available
        with pytest.raises(KeyError, match=r"chip_orch_add.*chip_orch_sub"):
            compiled["chip_orch_missing"]

        # 6) Unknown attribute that doesn't shadow an orch falls through to the
        #    standard AttributeError, not a silent KeyError from __getitem__
        with pytest.raises(AttributeError, match="no attribute 'definitely_not_an_orch'"):
            compiled.definitely_not_an_orch  # noqa: B018 — intentional attribute access

    def test_execute_one_orch_on_device(self, test_config, device_ids, tmp_path):
        """On-device smoke test: a sub-callable routes to its sub-build and runs.

        Compiles the multi-orch program with ptoas (no ``skip_ptoas``),
        then invokes one sub-orch through subscript dispatch. Verifies
        (a) ``execute_compiled`` accepts a ``next_levels/<name>/``
        directory and (b) the kernel produces correct output on real
        hardware.

        Skipped on hosts without a device (``--device`` unset) and under
        ``--codegen-only``, since both bypass real dispatch.

        Process isolation note: a ``Worker(level=2)`` device session
        currently leaks runtime state that hangs / segfaults the next
        ``Worker(level=3)`` init in the same Python process. CI keeps
        this file in its own ``pytest`` invocation
        (see ``.github/workflows/ci.yml`` "Test multi-orch L2") so the
        leak never crosses into ``test_l3_distributed``. Do not add an
        L3 test to this file or an L2 device test to the main
        distributed step without revisiting that split.
        """
        if not device_ids:
            pytest.skip("multi-orch on-device test needs at least one device")
        if test_config.codegen_only:
            pytest.skip("--codegen-only disables device execution")

        compiled = ir.compile(
            TwoL2AddSubProgram,
            output_dir=str(tmp_path),
            platform=test_config.platform,
        )

        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        f_add = torch.zeros((128, 128), dtype=torch.float32)

        add_via_sub = compiled["chip_orch_add"]
        add_via_sub(a, b, f_add, config=test_config)

        torch.testing.assert_close(f_add, torch.full((128, 128), 5.0, dtype=torch.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
