# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType, pto_backend
from pypto.backend._ptoas_config import PTOAS_BUFFER_ID_SYNC_MARKER
from pypto.runtime.debug import pto_rebuild


def _member(**attrs) -> ir.Function:
    span = ir.Span.unknown()
    return ir.Function("peer", [], [], ir.SeqStmts([], span), span, attrs=attrs)


def test_mixed_group_can_delegate_memory_planning_to_ptoas():
    members = [_member(ptoas_plan_memory=True), _member(ptoas_plan_memory=True)]

    assert (
        pto_backend._group_memory_planner(members, passes.MemoryPlanner.PYPTO) == passes.MemoryPlanner.PTOAS
    )


def test_mixed_group_rejects_inconsistent_memory_planning_options():
    members = [_member(ptoas_plan_memory=True), _member(ptoas_plan_memory=False)]

    with pytest.raises(ValueError, match="Grouped kernels must agree on ptoas_plan_memory"):
        pto_backend._group_memory_planner(members, passes.MemoryPlanner.PYPTO)


def test_local_ptoas_planning_legalizes_mixed_fp4_cast_before_outlining(tmp_path):
    @pl.program
    class MixedCast:
        @pl.function(type=pl.FunctionType.InCore)
        def project(
            self,
            src: pl.Tensor[[32, 128], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[32, 256], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[32, 256], pl.FP8E4M3FN]:
            peer = pl.import_peer_buffer(name="c2v")
            pl.aic_initialize_pipe(peer, dir_mask=1, slot_size=512, ptoas_plan_memory=True)
            local = pl.reserve_buffer(name="c2v", size=1024)
            pl.aiv_initialize_pipe(local, dir_mask=1, slot_size=512, ptoas_plan_memory=True)
            packed = pl.load(src, [0, 0], [32, 128])
            decoded = pl.cast(packed, pl.FP8E4M3FN)
            return pl.store(decoded, [0, 0], out)

    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        ir.compile(
            MixedCast,
            output_dir=str(tmp_path),
            backend_type=BackendType.Ascend950,
            skip_ptoas=True,
            dump_passes=False,
        )
    (pto_file,) = (tmp_path / "kernels").glob("*.pto")
    pto = pto_file.read_text()
    assert pto.count("pto.tcvt") == 3
    assert "pto.alloc_tile addr" not in pto


def test_mixed_group_keeps_ptoas_options_when_rebuilding(tmp_path, monkeypatch):
    members = [_member(ptoas_plan_memory=True, ptoas_buffer_id_sync=True) for _ in range(2)]
    result_files = {}
    pto_code = 'module attributes {pto.target_arch = "a5"} {\n  func.func @aic()\n}\n'

    pto_backend._emit_group_output(
        result_files,
        "mixed",
        members,
        pto_code,
        output_dir=str(tmp_path),
        skip_ptoas=True,
    )

    emitted = result_files["kernels/mixed.pto"]
    assert emitted.splitlines()[1] == PTOAS_BUFFER_ID_SYNC_MARKER
    expected_flags = ["--enable-bufid_sync", "--pto-level=level2"]
    assert pto_rebuild._ptoas_flags(emitted) == expected_flags

    def fake_run_ptoas(pto_path, output_path, ptoas_flags=None):
        assert Path(pto_path).read_text() == emitted
        assert ptoas_flags == expected_flags
        Path(output_path).write_text("generated cpp")

    monkeypatch.setattr(pto_backend, "_run_ptoas", fake_run_ptoas)
    handler = SimpleNamespace(get_extra_ptoas_flags=lambda: [])
    monkeypatch.setattr(pto_backend._backend_core, "get_handler", lambda: handler)
    planner = pto_backend._group_memory_planner(members, passes.MemoryPlanner.PYPTO)
    assert pto_backend._compile_pto_module(emitted, "mixed", str(tmp_path), planner) == "generated cpp"


def test_mixed_group_rejects_inconsistent_buffer_id_sync_options():
    members = [_member(ptoas_buffer_id_sync=True), _member(ptoas_buffer_id_sync=False)]

    with pytest.raises(ValueError, match="Grouped kernels must agree on ptoas_buffer_id_sync"):
        pto_backend._emit_group_output(
            {},
            "mixed",
            members,
            'module attributes {pto.target_arch = "a5"} {\n}\n',
            output_dir="unused",
            skip_ptoas=True,
        )


def test_buffer_id_sync_marker_requires_a5_pto():
    members = [_member(ptoas_buffer_id_sync=True)]

    with pytest.raises(ValueError, match="ptoas_buffer_id_sync requires A5 PTO code generation"):
        pto_backend._emit_group_output(
            {},
            "mixed",
            members,
            'module attributes {pto.target_arch = "a2a3"} {\n}\n',
            output_dir="unused",
            skip_ptoas=True,
        )
