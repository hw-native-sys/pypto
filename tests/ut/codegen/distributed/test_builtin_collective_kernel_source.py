# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""The HOST and CHIP rails must render one shared builtin collective kernel.

``pld.tensor.all_to_all_v`` has two managed lowering rails. Both render their
AIV kernel from the *same* template
(``pypto/runtime/builtins/collectives/all_to_all_v/templates/kernel.cpp.in``):

- HOST/L3 — ``LowerHostTensorCollectives`` emits ``builtin.tensor.all_to_all_v``
  and the distributed codegen renders the template into
  ``next_levels/builtin.tensor.all_to_all_v__fp32/``.
- CHIP/L2 — ``LowerL2TensorCollectives`` synthesizes an AIV function carrying
  ``builtin_template_dir`` / ``builtin_template_vars``, and the PTO backend
  renders the same template into the chip sub-build.

``dtype_cpp`` is the only substitution either rail makes and both give it the
same value, so the rendered sources must be **byte-identical**. That is the
property under test: it is what makes "one transport implementation, two rails"
true rather than aspirational. A change that reintroduces a per-rail difference
in the kernel body — an extra template variable, a rail-specific ``#ifdef`` —
fails here, instead of silently forking the transport into two implementations
that can drift apart on the wire.

This is a compile-only check: no device is dispatched and ptoas is skipped, so
it belongs here rather than in the hardware ST. (It lived in the ST briefly;
compiling a second distributed program inside a test that then dispatches its
own destabilised the run and cost device force-resets.)
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir
from pypto.ir import DistributedConfig
from pypto.pypto_core import passes


@pytest.fixture(autouse=True)
def _property_verification_only():
    """Property verification without the print -> parse roundtrip check.

    Same reason ``test_lower_host_tensor_collectives.py`` needs it: these
    programs run ``MaterializeCommDomainScopes``, and neither the
    ``CommDomainScopeStmt`` it synthesizes nor the ``WindowBuffer``
    back-references it stamps on ``DistributedTensorType`` has a DSL surface,
    so whole-program structural equality cannot survive a re-parse. What this
    file asserts — the rendered kernel text — is unaffected.
    """
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


SIZE = 64
NR = 2
MAX_RECV = 4
TOTAL = NR * MAX_RECV

_VARIANT = "builtin.tensor.all_to_all_v__fp32"
_L2_KERNEL = "__builtin_all_to_all_v__fp32.cpp"


def _build_chip_rail_program():
    """CHIP/L2 rail: the collective written in a CHIP orchestration body.

    Identical to the HOST program below except for *where* the collective is
    written — same five windows, same host allocation, same comm domain. That
    one difference is the whole point of the comparison.
    """

    @pl.program
    class ChipRail:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=1)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            stage_buf = pld.alloc_window_buffer(TOTAL * SIZE * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(TOTAL * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=pl.FP32)
                data = pld.window(data_buf, [TOTAL, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [NR, 1], dtype=pl.INT32)
                counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
                recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
                self.chip_pipeline(stage, data, sig, counts, recv, device=r)

    return ChipRail


def _build_host_rail_program():
    """HOST/L3 rail: the collective written in the host orchestrator.

    Mirrors the CHIP program: same five windows, same one dispatch per rank.
    The dispatched orchestration here does *not* hold the collective — that is
    the single difference under test — but it must still exist, because
    ``MaterializeCommDomainScopes`` infers a window's comm domain from the
    dispatches that consume it.
    """

    @pl.program
    class HostRail:
        @pl.function(type=pl.FunctionType.InCore)
        def touch_step(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NR, 1], pl.INT32]:
            """Touch every window, so each one has an inferable comm domain."""
            row = pl.load(stage, [0, 0], [1, SIZE])
            data = pl.store(row, [0, 0], data)
            # Read-back-and-write rather than a constant: the point is only to
            # give each window a consuming dispatch, and a scalar constant would
            # need a dtype-matched Scalar the DSL does not mint from a literal.
            for d in pl.range(NR):
                pl.write(counts, [d, 0], pl.read(counts, [d, 0]))
                pl.write(recv, [d, 0], pl.read(recv, [d, 0]))
                pl.write(signal, [d, 0], pl.read(signal, [d, 0]))
            return counts

        @pl.function(type=pl.FunctionType.Orchestration)
        def touch_orch(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NR, 1], pl.INT32]:
            return self.touch_step(stage, data, signal, counts, recv)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            stage_buf = pld.alloc_window_buffer(TOTAL * SIZE * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(TOTAL * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())

            stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf, [TOTAL, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [NR, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)

            for r in pl.range(pld.world_size()):
                r_stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=pl.FP32)
                r_data = pld.window(data_buf, [TOTAL, SIZE], dtype=pl.FP32)
                r_sig = pld.window(signal_buf, [NR, 1], dtype=pl.INT32)
                r_counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
                r_recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
                self.touch_orch(r_stage, r_data, r_sig, r_counts, r_recv, device=r)

            pld.tensor.all_to_all_v(stage, data, sig, counts, recv)

    return HostRail


def _compile(program, tmp_path, name):
    return ir.compile(
        program,
        output_dir=str(tmp_path / name),
        platform="a2a3",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
    )


def _sole_file(directory, pattern):
    matches = sorted(directory.glob(pattern))
    assert len(matches) == 1, f"expected exactly one {pattern} under {directory}, got {matches}"
    return matches[0]


def test_both_rails_render_a_byte_identical_builtin_kernel(tmp_path):
    """The two rails' rendered kernel sources must match exactly."""
    chip = _compile(_build_chip_rail_program(), tmp_path, "chip")
    host = _compile(_build_host_rail_program(), tmp_path, "host")

    chip_kernel = chip.output_dir / "next_levels" / "chip_pipeline" / "kernels" / "aiv" / _L2_KERNEL
    assert chip_kernel.is_file(), f"expected the rendered CHIP kernel at {chip_kernel}"
    host_kernel = _sole_file(host.output_dir / "next_levels" / _VARIANT / "kernels" / "aiv", "*.cpp")

    assert host_kernel.read_text() == chip_kernel.read_text(), (
        "HOST and CHIP rails must render the same builtin kernel source; a difference means "
        "the two rails no longer share one transport implementation"
    )


def test_chip_rail_emits_no_builtin_chip_dispatch(tmp_path):
    """The CHIP rail keeps the collective inside the caller's own pipeline.

    A ``next_levels/builtin.tensor.*`` directory would mean the HOST fan-out
    rail ran, i.e. the collective became a nested L2 orchestration task instead
    of one AIV task of ``chip_pipeline``.
    """
    chip = _compile(_build_chip_rail_program(), tmp_path, "chip")

    next_levels = sorted(p.name for p in (chip.output_dir / "next_levels").iterdir())
    assert next_levels == ["chip_pipeline"], next_levels


def test_host_rail_still_emits_its_builtin_chip_dispatch(tmp_path):
    """The HOST rail is unchanged — the fan-out dispatch is still its shape.

    Pins the contrast the CHIP rail is defined against, so a regression that
    silently routed HOST calls through the CHIP rail would not pass unnoticed.
    """
    host = _compile(_build_host_rail_program(), tmp_path, "host")

    next_levels = sorted(p.name for p in (host.output_dir / "next_levels").iterdir())
    assert _VARIANT in next_levels, next_levels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
