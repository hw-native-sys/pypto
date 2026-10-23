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

import re

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir
from pypto.ir import DistributedConfig
from pypto.pypto_core import passes


@pytest.fixture(autouse=True)
def _property_verification_only():
    """Property verification without the print -> parse roundtrip check.

    These tests assert rendered kernel artifacts. Complete distributed IR text
    roundtrips are checked with the default instrument in the materialization
    and host-collective transform tests.
    """
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


SIZE = 64
NR = 2
MAX_RECV = 4
TOTAL = NR * MAX_RECV


def _dtype_of(name: str):
    return {"fp32": pl.FP32, "int8": pl.INT8}[name]


def _variant_of(name: str) -> str:
    return f"builtin.tensor.all_to_all_v__{name}"


def _l2_kernel_of(name: str) -> str:
    return f"__builtin_all_to_all_v__{name}.cpp"


def _build_chip_rail_program(dtype_name: str = "fp32", core_num: int = 1, signal_stride: int = 1):
    """CHIP/L2 rail: the collective written in a CHIP orchestration body.

    Identical to the HOST program below except for *where* the collective is
    written — same five windows, same host allocation, same comm domain. That
    one difference is the whole point of the comparison.

    ``core_num`` / ``signal_stride`` choose the requested block limit and the
    signal's per-block lane count ``S`` (the signal is ``[NR, S]``).
    """
    dtype = _dtype_of(dtype_name)
    payload_bytes = TOTAL * SIZE * dtype.get_byte()
    i32_bytes = NR * pl.INT32.get_byte()
    signal_bytes = NR * signal_stride * pl.INT32.get_byte()

    @pl.program
    class ChipRail:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, signal_stride], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], dtype]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=core_num)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            stage_buf = pld.alloc_window_buffer(payload_bytes)
            data_buf = pld.alloc_window_buffer(payload_bytes)
            signal_buf = pld.alloc_window_buffer(signal_bytes)
            counts_buf = pld.alloc_window_buffer(i32_bytes)
            recv_buf = pld.alloc_window_buffer(i32_bytes)

            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=dtype)
                data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
                sig = pld.window(signal_buf, [NR, signal_stride], dtype=pl.INT32)
                counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
                recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
                self.chip_pipeline(stage, data, sig, counts, recv, device=r)

    return ChipRail


def _build_host_rail_program(dtype_name: str = "fp32"):
    """HOST/L3 rail: the collective written in the host orchestrator.

    Mirrors the CHIP program: same five windows, same one dispatch per rank.
    The dispatched orchestration here does *not* hold the collective — that is
    the single difference under test — but it must still exist, because
    ``MaterializeCommDomainScopes`` infers a window's comm domain from the
    dispatches that consume it.
    """
    dtype = _dtype_of(dtype_name)
    payload_bytes = TOTAL * SIZE * dtype.get_byte()
    i32_bytes = NR * pl.INT32.get_byte()

    @pl.program
    class HostRail:
        @pl.function(type=pl.FunctionType.InCore)
        def touch_step(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
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
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NR, 1], pl.INT32]:
            return self.touch_step(stage, data, signal, counts, recv)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            stage_buf = pld.alloc_window_buffer(payload_bytes)
            data_buf = pld.alloc_window_buffer(payload_bytes)
            signal_buf = pld.alloc_window_buffer(i32_bytes)
            counts_buf = pld.alloc_window_buffer(i32_bytes)
            recv_buf = pld.alloc_window_buffer(i32_bytes)

            stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=dtype)
            data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
            sig = pld.window(signal_buf, [NR, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)

            for r in pl.range(pld.world_size()):
                r_stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=dtype)
                r_data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
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


@pytest.mark.parametrize("dtype_name", ["fp32", "int8"])
def test_both_rails_render_a_byte_identical_builtin_kernel(tmp_path, dtype_name):
    """The two rails' rendered kernel sources must match exactly."""
    chip = _compile(_build_chip_rail_program(dtype_name), tmp_path, f"chip_{dtype_name}")
    host = _compile(_build_host_rail_program(dtype_name), tmp_path, f"host_{dtype_name}")

    chip_kernel = (
        chip.output_dir / "next_levels" / "chip_pipeline" / "kernels" / "aiv" / _l2_kernel_of(dtype_name)
    )
    assert chip_kernel.is_file(), f"expected the rendered CHIP kernel at {chip_kernel}"
    host_kernel = _sole_file(
        host.output_dir / "next_levels" / _variant_of(dtype_name) / "kernels" / "aiv", "*.cpp"
    )

    assert host_kernel.read_text() == chip_kernel.read_text(), (
        "HOST and CHIP rails must render the same builtin kernel source; a difference means "
        "the two rails no longer share one transport implementation"
    )


def _chip_orchestration_sources(output_dir) -> str:
    """Every emitted CHIP-orchestration C++ source of ``chip_pipeline``, concatenated."""
    sources = sorted((output_dir / "next_levels" / "chip_pipeline").rglob("*.cpp"))
    orchestration = [s for s in sources if "orchestration" in s.parts]
    assert orchestration, f"expected an emitted orchestration source under {output_dir}"
    return "\n".join(s.read_text() for s in orchestration)


def test_chip_rail_multi_block_request_emits_the_launch_spec(tmp_path):
    """``core_num=2`` over two ranks launches a two-block, sync-start gang.

    The pass folds ``B = CalAllToAllVBlocks(2, 2) = 2`` and the orchestration
    codegen renders it onto the kernel's task, next to ``require_sync_start``.
    """
    chip = _compile(_build_chip_rail_program(core_num=2, signal_stride=2), tmp_path, "chip_multi")
    source = _chip_orchestration_sources(chip.output_dir)
    assert re.search(r"\.launch_spec\.\w+\(2\);", source), source
    assert ".launch_spec.set_require_sync_start(true);" in source, source


def test_chip_rail_single_block_request_emits_no_launch_spec(tmp_path):
    """``core_num=1`` is unchanged: a plain single-block task, no gang launch."""
    chip = _compile(_build_chip_rail_program(core_num=1), tmp_path, "chip_single")
    assert ".launch_spec." not in _chip_orchestration_sources(chip.output_dir)


def test_multi_block_chip_rail_still_renders_the_shared_kernel(tmp_path):
    """A multi-block request must not fork the kernel from the HOST rail's.

    The launch spec is a call attr, not a kernel parameter, so the rendered
    source stays byte-identical — the tripwire for a future "just pass it as an
    argument" change that would shift the CommContext off ``args[5]``.
    """
    chip = _compile(_build_chip_rail_program(core_num=2, signal_stride=2), tmp_path, "chip_multi_kernel")
    host = _compile(_build_host_rail_program(), tmp_path, "host_multi_kernel")
    chip_kernel = (
        chip.output_dir / "next_levels" / "chip_pipeline" / "kernels" / "aiv" / _l2_kernel_of("fp32")
    )
    host_kernel = _sole_file(
        host.output_dir / "next_levels" / _variant_of("fp32") / "kernels" / "aiv", "*.cpp"
    )
    assert chip_kernel.is_file(), f"expected the rendered CHIP kernel at {chip_kernel}"
    assert host_kernel.read_text() == chip_kernel.read_text()


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
    assert _variant_of("fp32") in next_levels, next_levels


_OPERANDS = ("input", "target", "signal", "send_counts", "recv_counts")


@pytest.mark.parametrize("core_num, signal_stride", [(1, 1), (2, 2)])
def test_chip_rail_checks_operand_contiguity_before_the_submit(tmp_path, core_num, signal_stride):
    """RFC #2521 O1 criterion 7: a strided runtime operand never reaches the kernel.

    The builtin kernel addresses every operand as dense row-major from its view
    origin, and the type deducer can only reject views it can see at compile
    time. The orchestration therefore checks each operand's runtime
    ``is_contiguous`` flag ahead of the collective's submit, latching
    ``SIMPLER_ERROR_INVALID_ARGS`` (which also turns the submit into a no-op).
    The checks must sit *before* the submit, cover all five operands, and be
    chained so only the first violation is reported; the launch spec must not
    change where they go.
    """
    chip = _compile(
        _build_chip_rail_program(core_num=core_num, signal_stride=signal_stride),
        tmp_path,
        f"chip_contig_{core_num}",
    )
    source = _chip_orchestration_sources(chip.output_dir)

    checks = re.findall(r"(if|\} else if) \(!(\w+)\.is_contiguous\) \{", source)
    assert [keyword for keyword, _ in checks] == ["if"] + ["} else if"] * 4, source
    for operand in _OPERANDS:
        assert (
            f'rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "__builtin_all_to_all_v__fp32: {operand} '
            'must be a contiguous row-major view");'
        ) in source, source
    assert source.index(".is_contiguous") < source.index("rt_submit_aiv_task"), source


def test_ordinary_chip_tasks_get_no_contiguity_check(tmp_path):
    """Only the builtin kernel opts in; an ordinary InCore task is emitted as before.

    The HOST rail's ``touch_orch`` dispatches a plain DSL kernel over the same
    five windows. Its orchestration must carry no operand checks — the attribute,
    not the operand types, is what decides.
    """
    host = _compile(_build_host_rail_program(), tmp_path, "host_no_contig")
    sources = sorted((host.output_dir / "next_levels" / "touch_orch").rglob("*.cpp"))
    orchestration = [s for s in sources if "orchestration" in s.parts]
    assert orchestration, f"expected touch_orch's orchestration source under {host.output_dir}"
    for path in orchestration:
        assert "is_contiguous" not in path.read_text(), path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
