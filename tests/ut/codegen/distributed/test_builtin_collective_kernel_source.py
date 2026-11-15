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
import shutil
import subprocess
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import codegen, ir
from pypto._artifact_contract import ArtifactExecutionMode
from pypto.backend import pto_backend
from pypto.backend.pto_backend import (
    _render_builtin_kernel_sources,
    _render_builtin_template,
    _resolve_builtin_template_dir,
)
from pypto.ir import DistributedConfig
from pypto.ir.compile import _compile_to_files
from pypto.pypto_core import passes
from pypto.pypto_core.backend import BackendType


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


def _build_indexed_rail_program(*, host: bool, dtype_name: str):
    dtype = _dtype_of(dtype_name)
    input_bytes = 13 * SIZE * dtype.get_byte()
    target_bytes = TOTAL * SIZE * dtype.get_byte()

    @pl.program
    class IndexedRail:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            inp: pld.DistributedTensor[[13, SIZE], dtype],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, NR], pl.INT32]],
            counts: pld.DistributedTensor[[NR, 1], pl.INT32],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            index: pld.DistributedTensor[[NR, MAX_RECV], pl.INT32],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], dtype]:
            return pld.tensor.indexed_all_to_all_v(inp, data, signal, counts, recv, index, core_num=NR)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            inp_buf = pld.alloc_window_buffer(input_bytes)
            data_buf = pld.alloc_window_buffer(target_bytes)
            sig_buf = pld.alloc_window_buffer(NR * NR * 4)
            counts_buf = pld.alloc_window_buffer(NR * 4)
            recv_buf = pld.alloc_window_buffer(NR * 4)
            index_buf = pld.alloc_window_buffer(NR * MAX_RECV * 4)
            for r in pl.range(pld.world_size()):
                inp = pld.window(inp_buf, [13, SIZE], dtype=dtype)
                data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
                sig = pld.window(sig_buf, [NR, NR], dtype=pl.INT32)
                counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
                recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
                index = pld.window(index_buf, [NR, MAX_RECV], dtype=pl.INT32)
                self.chip_pipeline(inp, data, sig, counts, recv, index, device=r)

    if not host:
        return IndexedRail
    # Keep the same windows/domain registration while moving the exchange to HOST.
    source = ir.python_print(IndexedRail, format=False)
    source, replacements = re.subn(
        r"return pld\.tensor\.indexed_all_to_all_v\(inp, data, signal, counts, recv, index, core_num=2\)",
        "return data",
        source,
    )
    assert replacements == 1, source
    # This registration dispatch only reads its windows after moving the exchange.
    source = re.sub(r"pl\.InOut\[(pld\.DistributedTensor\[\[[^]]+\], pl\.\w+\])\]", r"\1", source)
    # HOST windows must be defined outside the registration loop to stay in scope.
    lines = source.splitlines()
    windows = [
        line[4:] for line in lines if line.startswith("            ") and " = pld.tensor.window(" in line
    ]
    lines = [
        line for line in lines if not (line.startswith("            ") and " = pld.tensor.window(" in line)
    ]
    loop = next(i for i, line in enumerate(lines) if line.startswith("        for r in "))
    lines[loop:loop] = windows
    source = "\n".join(lines)
    # The HOST call belongs inside host_orch, after its dispatch loop.
    source = source.rstrip() + (
        "\n        pld.tensor.indexed_all_to_all_v(inp, data, sig, counts, recv, index, core_num=2)\n"
    )
    return pl.parse_program(source)


@pytest.mark.parametrize("dtype_name", ["fp32", "int8"])
def test_indexed_l2_compiles_six_operand_kernel(tmp_path, dtype_name):
    result = ir.compile(
        _build_indexed_rail_program(host=False, dtype_name=dtype_name),
        output_dir=str(tmp_path / dtype_name),
        platform="a5",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
    )
    kernel = (
        result.output_dir
        / "next_levels"
        / "chip_pipeline"
        / "kernels"
        / "aiv"
        / f"__builtin_indexed_all_to_all_v__{dtype_name}.cpp"
    )
    assert kernel.is_file()


@pytest.mark.parametrize("dtype_name", ["fp32", "int8"])
def test_indexed_rejects_host_call(tmp_path, dtype_name):
    with pytest.raises(ValueError, match="L2 orchestration.*HOST is unsupported"):
        ir.compile(
            _build_indexed_rail_program(host=True, dtype_name=dtype_name),
            output_dir=str(tmp_path / dtype_name),
            platform="a5",
            skip_ptoas=True,
            distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
        )


_OPERANDS = ("input", "target", "signal", "send_counts", "recv_counts")


def _assert_operand_guards(source: str, kernel: str, operands: tuple[str, ...]) -> None:
    """One chained ``is_contiguous`` check per operand, in order, ahead of the submit."""
    keywords = re.findall(r"(if|\} else if) \(!\w+\.is_contiguous\) \{", source)
    assert keywords == ["if"] + ["} else if"] * (len(operands) - 1), source
    for operand in operands:
        fatal = f"{kernel}: {operand} must be a contiguous row-major view"
        assert f'rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "{fatal}");' in source, source
    assert source.rindex(".is_contiguous") < source.index("rt_submit_aiv_task"), source


def test_chip_rail_checks_operand_contiguity_before_the_submit(tmp_path):
    """RFC #2521 O1 criterion 7: the runtime backstop. The orchestration checks each
    operand's ``is_contiguous`` flag ahead of the collective's submit, latching
    ``SIMPLER_ERROR_INVALID_ARGS`` on the first violation (which also turns the submit
    into a no-op). The compile-time proof lives in ``LowerL2TensorCollectives``."""
    chip = _compile(_build_chip_rail_program(), tmp_path, "chip_contig")
    _assert_operand_guards(
        _chip_orchestration_sources(chip.output_dir), "__builtin_all_to_all_v__fp32", _OPERANDS
    )


def test_indexed_rail_checks_all_six_operands_before_the_submit(tmp_path):
    """The indexed kernel reads `index` as dense row-major too, so it gets a sixth check."""
    result = ir.compile(
        _build_indexed_rail_program(host=False, dtype_name="fp32"),
        output_dir=str(tmp_path / "indexed_contig"),
        platform="a5",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
    )
    _assert_operand_guards(
        _chip_orchestration_sources(result.output_dir),
        "__builtin_indexed_all_to_all_v__fp32",
        (*_OPERANDS, "index"),
    )


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


def test_indexed_rail_requests_the_urma_workspace(tmp_path):
    """The domain must provision the URMA descriptor the kernel reads.

    After the runtime's per-engine workspace split, a domain that does not list
    ``urma`` gets a zero descriptor, and the indexed kernel aborts on the
    zero-workspace guard instead of falling back to another engine. The check is
    on the emitted host orchestration because the request is an allocation-time
    argument, not something the kernel can repair.
    """
    result = ir.compile(
        _build_indexed_rail_program(host=False, dtype_name="fp32"),
        output_dir=str(tmp_path / "indexed_workspace"),
        platform="a5",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
    )
    host_orch = (result.output_dir / "orchestration" / "host_orch.py").read_text()
    assert 'workspace_engines=("urma",)' in host_orch, host_orch


def test_indexed_use_requests_urma_for_all_domains(tmp_path):
    """Program-wide URMA detection provisions every materialized domain."""
    dtype = _dtype_of("fp32")
    dtype_bytes = dtype.get_byte()
    i32_bytes = pl.INT32.get_byte()
    input_bytes = 13 * SIZE * dtype_bytes
    target_bytes = TOTAL * SIZE * dtype_bytes
    signal_bytes = NR * NR * i32_bytes
    counts_bytes = NR * i32_bytes
    index_bytes = NR * MAX_RECV * i32_bytes
    plain_bytes = SIZE * pl.FP32.get_byte()

    @pl.program
    class TwoDomains:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_indexed(
            self,
            inp: pld.DistributedTensor[[13, SIZE], dtype],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, NR], pl.INT32]],
            counts: pld.DistributedTensor[[NR, 1], pl.INT32],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            index: pld.DistributedTensor[[NR, MAX_RECV], pl.INT32],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], dtype]:
            return pld.tensor.indexed_all_to_all_v(inp, data, signal, counts, recv, index, core_num=NR)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_plain(
            self, a: pld.DistributedTensor[[SIZE], pl.FP32]
        ) -> pld.DistributedTensor[[SIZE], pl.FP32]:
            return a

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(input_bytes)
            data_buf = pld.alloc_window_buffer(target_bytes)
            signal_buf = pld.alloc_window_buffer(signal_bytes)
            counts_buf = pld.alloc_window_buffer(counts_bytes)
            recv_buf = pld.alloc_window_buffer(counts_bytes)
            index_buf = pld.alloc_window_buffer(index_bytes)
            plain_buf = pld.alloc_window_buffer(plain_bytes)

            inp = pld.window(input_buf, [13, SIZE], dtype=dtype)
            data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
            sig = pld.window(signal_buf, [NR, NR], dtype=pl.INT32)
            counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
            index = pld.window(index_buf, [NR, MAX_RECV], dtype=pl.INT32)
            self.chip_indexed(inp, data, sig, counts, recv, index, device=0)

            plain = pld.window(plain_buf, [SIZE], dtype=pl.FP32)
            self.chip_plain(plain, device=1)
            return 0

    result = ir.compile(
        TwoDomains,
        output_dir=str(tmp_path / "two_domains"),
        platform="a5",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    )
    host_orch = (result.output_dir / "orchestration" / "host_orch.py").read_text()

    assert host_orch.count('workspace_engines=("urma",)') == 2, host_orch
    d0 = host_orch.split('name="comm_d0"', 1)[1].split("as __comm_d0:", 1)[0]
    d1 = host_orch.split('name="comm_d1"', 1)[1].split("as __comm_d1:", 1)[0]
    assert 'workspace_engines=("urma",)' in d0, host_orch
    assert 'workspace_engines=("urma",)' in d1, host_orch


def test_shared_dispatch_provisions_all_domains_conservatively(tmp_path):
    """Program-wide URMA detection provisions every materialized domain."""
    dtype = _dtype_of("fp32")
    dtype_bytes = dtype.get_byte()
    i32_bytes = pl.INT32.get_byte()
    input_bytes = 13 * SIZE * dtype_bytes
    target_bytes = TOTAL * SIZE * dtype_bytes
    signal_bytes = NR * NR * i32_bytes
    counts_bytes = NR * i32_bytes
    index_bytes = NR * MAX_RECV * i32_bytes
    plain_bytes = SIZE * pl.FP32.get_byte()

    @pl.program
    class SharedDispatch:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_plain(
            self, a: pld.DistributedTensor[[SIZE], pl.FP32]
        ) -> pld.DistributedTensor[[SIZE], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_mixed(
            self,
            plain: pld.DistributedTensor[[SIZE], pl.FP32],
            inp: pld.DistributedTensor[[13, SIZE], dtype],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, NR], pl.INT32]],
            counts: pld.DistributedTensor[[NR, 1], pl.INT32],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            index: pld.DistributedTensor[[NR, MAX_RECV], pl.INT32],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], dtype]:
            # ``plain`` links this dispatch to the sibling domain but carries no
            # part of the indexed exchange.
            return pld.tensor.indexed_all_to_all_v(inp, data, signal, counts, recv, index, core_num=NR)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(input_bytes)
            data_buf = pld.alloc_window_buffer(target_bytes)
            signal_buf = pld.alloc_window_buffer(signal_bytes)
            counts_buf = pld.alloc_window_buffer(counts_bytes)
            recv_buf = pld.alloc_window_buffer(counts_bytes)
            index_buf = pld.alloc_window_buffer(index_bytes)
            plain_buf = pld.alloc_window_buffer(plain_bytes)

            inp = pld.window(input_buf, [13, SIZE], dtype=dtype)
            data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
            sig = pld.window(signal_buf, [NR, NR], dtype=pl.INT32)
            counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
            index = pld.window(index_buf, [NR, MAX_RECV], dtype=pl.INT32)
            plain = pld.window(plain_buf, [SIZE], dtype=pl.FP32)

            # Establishes the plain window's device coverage ({0, 1} after the
            # mixed dispatch below) independently of the indexed domain ({0}).
            self.chip_plain(plain, device=1)
            # One dispatch, two domains' windows; only ``inp``..``index`` reach
            # the indexed collective.
            self.chip_mixed(plain, inp, data, sig, counts, recv, index, device=0)

    result = ir.compile(
        SharedDispatch,
        output_dir=str(tmp_path / "shared_dispatch"),
        platform="a5",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    )
    host_orch = (result.output_dir / "orchestration" / "host_orch.py").read_text()

    # The dispatch really is shared: one TaskArgs carries the sibling domain's
    # ``plain`` window alongside the indexed domain's operands, plus one
    # ordering token per domain. Both domains receive the program's URMA requirement.
    mixed = host_orch.split('_submit_chip(orch, callables["chip_mixed"]', 1)[0]
    mixed = mixed[mixed.rindex("= TaskArgs()") :]
    assert '__comm_d1[0].buffers["plain_buf' in mixed, host_orch
    assert '__comm_d0[0].buffers["input_buf' in mixed, host_orch
    assert "__comm_d1_ord" in mixed and "__comm_d0_ord" in mixed, host_orch

    assert host_orch.count('workspace_engines=("urma",)') == 2, host_orch
    d0 = host_orch.split('name="comm_d0"', 1)[1].split("as __comm_d0:", 1)[0]
    d1 = host_orch.split('name="comm_d1"', 1)[1].split("as __comm_d1:", 1)[0]
    assert 'workspace_engines=("urma",)' in d0, host_orch
    assert 'workspace_engines=("urma",)' in d1, host_orch


def test_indexed_collective_fed_by_if_result_requests_urma(tmp_path):
    """A collective nested behind control flow still contributes program engines."""
    dtype = _dtype_of("fp32")
    dtype_bytes = dtype.get_byte()
    i32_bytes = pl.INT32.get_byte()
    input_bytes = 13 * SIZE * dtype_bytes
    target_bytes = TOTAL * SIZE * dtype_bytes
    signal_bytes = NR * NR * i32_bytes
    counts_bytes = NR * i32_bytes
    index_bytes = NR * MAX_RECV * i32_bytes

    @pl.program
    class IfResult:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            choose: pl.Scalar[pl.INT32],
            inp: pld.DistributedTensor[[13, SIZE], dtype],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, NR], pl.INT32]],
            counts: pld.DistributedTensor[[NR, 1], pl.INT32],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            index: pld.DistributedTensor[[NR, MAX_RECV], pl.INT32],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], dtype]:
            # Route operands through the merge before invoking the collective.
            if choose > 0:
                c_inp, c_data = inp, data
                c_signal, c_counts = signal, counts
                c_recv, c_index = recv, index
            else:
                c_inp, c_data = inp, data
                c_signal, c_counts = signal, counts
                c_recv, c_index = recv, index
            return pld.tensor.indexed_all_to_all_v(
                c_inp, c_data, c_signal, c_counts, c_recv, c_index, core_num=NR
            )

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(input_bytes)
            data_buf = pld.alloc_window_buffer(target_bytes)
            signal_buf = pld.alloc_window_buffer(signal_bytes)
            counts_buf = pld.alloc_window_buffer(counts_bytes)
            recv_buf = pld.alloc_window_buffer(counts_bytes)
            index_buf = pld.alloc_window_buffer(index_bytes)

            inp = pld.window(input_buf, [13, SIZE], dtype=dtype)
            data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
            sig = pld.window(signal_buf, [NR, NR], dtype=pl.INT32)
            counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
            index = pld.window(index_buf, [NR, MAX_RECV], dtype=pl.INT32)

            self.chip_pipeline(1, inp, data, sig, counts, recv, index, device=0)

    result = ir.compile(
        IfResult,
        output_dir=str(tmp_path / "if_result"),
        platform="a5",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0),
    )
    host_orch = (result.output_dir / "orchestration" / "host_orch.py").read_text()
    assert 'workspace_engines=("urma",)' in host_orch, host_orch


def test_local_operand_domain_also_receives_program_engines(tmp_path):
    """Program-wide URMA detection provisions every materialized domain."""
    dtype = _dtype_of("fp32")
    dtype_bytes = dtype.get_byte()
    i32_bytes = pl.INT32.get_byte()
    input_bytes = 13 * SIZE * dtype_bytes
    target_bytes = TOTAL * SIZE * dtype_bytes
    signal_bytes = NR * NR * i32_bytes
    counts_bytes = NR * i32_bytes
    index_bytes = NR * MAX_RECV * i32_bytes

    @pl.program
    class LocalOperandDomain:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_indexed(
            self,
            inp: pld.DistributedTensor[[13, SIZE], dtype],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, NR], pl.INT32]],
            counts: pld.DistributedTensor[[NR, 1], pl.INT32],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            index: pld.DistributedTensor[[NR, MAX_RECV], pl.INT32],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], dtype]:
            return pld.tensor.indexed_all_to_all_v(inp, data, signal, counts, recv, index, core_num=NR)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_index_owner(
            self, index: pld.DistributedTensor[[NR, MAX_RECV], pl.INT32]
        ) -> pld.DistributedTensor[[NR, MAX_RECV], pl.INT32]:
            return index

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(input_bytes)
            data_buf = pld.alloc_window_buffer(target_bytes)
            signal_buf = pld.alloc_window_buffer(signal_bytes)
            counts_buf = pld.alloc_window_buffer(counts_bytes)
            recv_buf = pld.alloc_window_buffer(counts_bytes)
            index_buf = pld.alloc_window_buffer(index_bytes)

            inp = pld.window(input_buf, [13, SIZE], dtype=dtype)
            data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
            sig = pld.window(signal_buf, [NR, NR], dtype=pl.INT32)
            counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
            index = pld.window(index_buf, [NR, MAX_RECV], dtype=pl.INT32)

            self.chip_indexed(inp, data, sig, counts, recv, index, device=0)
            # Also dispatched on device 1, so the index window merges to a
            # different comm domain from the collective's context domain.
            self.chip_index_owner(index, device=1)

    result = ir.compile(
        LocalOperandDomain,
        output_dir=str(tmp_path / "local_operand_domain"),
        platform="a5",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    )
    host_orch = (result.output_dir / "orchestration" / "host_orch.py").read_text()

    assert host_orch.count('workspace_engines=("urma",)') == 2, host_orch
    d0 = host_orch.split('name="comm_d0"', 1)[1].split("as __comm_d0:", 1)[0]
    d1 = host_orch.split('name="comm_d1"', 1)[1].split("as __comm_d1:", 1)[0]
    assert 'workspace_engines=("urma",)' in d0, host_orch
    assert 'workspace_engines=("urma",)' in d1, host_orch


@pytest.mark.parametrize("engines", [[], ["sdma"], ["urma"], ["sdma", "urma"]])
def test_codegen_reads_explicit_domain_engines(engines):
    """Codegen emits IR requirements even when the body contains no collective."""
    span = ir.Span.unknown()
    buffer = ir.WindowBuffer(ir.Var("buf", ir.PtrType(), span), ir.ConstInt(64, pl.INT64, span))
    body = ir.CommDomainScopeStmt(
        [0], [buffer], "comm_d0", body=ir.SeqStmts([], span), span=span, engines=engines
    )
    host = ir.Function("host_orch", [], [], body, span, level=ir.Level.HOST, role=ir.Role.Orchestrator)
    program = ir.Program([host], "explicit_engines", span)
    source = codegen.DistributedCodegen().generate(program)
    if engines:
        expected = "workspace_engines=(" + "".join(f'"{engine}",' for engine in engines) + "),"
        assert expected in source
    else:
        assert "workspace_engines" not in source


def test_chip_rail_requests_no_async_workspace(tmp_path):
    """A domain whose body consumes no async engine provisions none."""
    chip = _compile(_build_chip_rail_program(), tmp_path, "chip_no_workspace")
    host_orch = (chip.output_dir / "orchestration" / "host_orch.py").read_text()
    assert "allocate_domain" in host_orch, host_orch
    assert "workspace_engines" not in host_orch, host_orch


@pytest.mark.parametrize("skip_ptoas", [True, False])
def test_async_transfer_requests_the_urma_workspace(tmp_path, monkeypatch, skip_ptoas):
    """End-to-end A9: an async transfer reaches the host domain's engine list.

    The engine scan must recognise ``put_async`` on the chip function, the pass
    must record ``urma`` on every materialized domain under the program-wide
    policy, and codegen must render it as ``workspace_engines``. Nothing else in
    the program asks for URMA, so a missing match would emit a domain with no
    engine list and the runtime would hand the helper a zero workspace.
    """
    f32_bytes = pl.FP32.get_byte()
    window_bytes = SIZE * f32_bytes
    # Exercise both backend emission branches without requiring PTOAS. The
    # generated MLIR and wrapper ABI are checked independently below.
    emitted_mlir = []

    def capture_ptoas(pto_code, *args, **kwargs):
        emitted_mlir.append(pto_code)
        return "// PTOAS output omitted in this host-codegen unit test\n"

    monkeypatch.setattr(pto_backend, "_compile_pto_module", capture_ptoas)

    @pl.program
    class AsyncProgram:
        # The transfer is issued by the task that owns the destination, so the
        # op lives in the InCore body — exactly where the VECTOR placement puts
        # it. An Orchestration body would only dispatch this kernel.
        @pl.function(type=pl.FunctionType.InCore)
        def async_kernel(
            self,
            dst: pld.DistributedTensor[[SIZE], pl.FP32],
            src: pld.DistributedTensor[[SIZE], pl.FP32],
        ) -> pld.DistributedTensor[[SIZE], pl.FP32]:
            pld.tensor.put_async(dst, 1, src)
            return dst

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_async(
            self,
            dst: pld.DistributedTensor[[SIZE], pl.FP32],
            src: pld.DistributedTensor[[SIZE], pl.FP32],
        ) -> pld.DistributedTensor[[SIZE], pl.FP32]:
            return self.async_kernel(dst, src)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            dst_buf = pld.alloc_window_buffer(window_bytes)
            src_buf = pld.alloc_window_buffer(window_bytes)
            dst = pld.window(dst_buf, [SIZE], dtype=pl.FP32)
            src = pld.window(src_buf, [SIZE], dtype=pl.FP32)
            self.chip_async(dst, src, device=0)
            return 0

    result = ir.compile(
        AsyncProgram,
        output_dir=str(tmp_path / "async_workspace"),
        platform="a5",
        skip_ptoas=skip_ptoas,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
    )
    host_orch = (result.output_dir / "orchestration" / "host_orch.py").read_text()

    assert 'workspace_engines=("urma",)' in host_orch, host_orch
    # No synchronous fallback may be introduced by the lowering.
    assert "DmaEngine" not in host_orch, host_orch
    if skip_ptoas:
        emitted_mlir = [p.read_text() for p in result.output_dir.rglob("*.pto")]
    else:
        wrappers = [p.read_text() for p in result.output_dir.rglob("*.cpp") if "kernels" in p.parts]
        assert any("pypto_urma_put_async_f32" in wrapper and ", args);" in wrapper for wrapper in wrappers)
    assert any("%__pypto_deferred_raw_args: !pto.ptr<i64>" in source for source in emitted_mlir)


def _ccu_allgather_source_program(*, host=False, ranks: int = 2, dispatch_ranks: int | None = None):
    device_count = ranks if dispatch_ranks is None else dispatch_ranks

    @pl.program
    class Chip:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            data: pld.DistributedTensor[[1, 64], pl.BF16],
            target: pl.InOut[pld.DistributedTensor[[ranks, 64], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[ranks, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[ranks, 64], pl.BF16]:
            target = pld.tensor.allgather(data, target, signal, engine="ccu")
            target = pld.tensor.allgather(data, target, signal, engine="ccu")
            return target

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(128)
            target_buf = pld.alloc_window_buffer(ranks * 128)
            signal_buf = pld.alloc_window_buffer(ranks * 4)
            for r in pl.range(device_count):
                data = pld.window(input_buf, [1, 64], dtype=pl.BF16)
                target = pld.window(target_buf, [ranks, 64], dtype=pl.BF16)
                signal = pld.window(signal_buf, [ranks, 1], dtype=pl.INT32)
                self.chip_pipeline(data, target, signal, device=r)

    if host:
        return Chip
    return ir.Program(
        [f for f in Chip.functions.values() if f.name == "chip_pipeline"], "Chip", ir.Span.unknown()
    )


@pytest.mark.parametrize("ranks", [2, 4, 8, 64])
@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize(
    "runtime", [passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, passes.RuntimeKind.HOST_BUILD_GRAPH]
)
def test_ccu_allgather_full_codegen(tmp_path, host, runtime, ranks):
    output = tmp_path / "allgather"
    program = _ccu_allgather_source_program(host=host, ranks=ranks)
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)], runtime=runtime
    ):
        ir.compile(
            program,
            output_dir=str(output),
            backend_type=BackendType.Ascend950,
            skip_ptoas=True,
            dump_passes=False,
            distributed_config=DistributedConfig(device_ids=list(range(ranks)), num_sub_workers=0)
            if host
            else None,
        )
    config_paths = list(output.rglob("kernel_config.py"))
    assert len(config_paths) == 1
    config = config_paths[0].read_text()
    assert "CCU_MODULES =" in config
    assert "'ccu_id': 0" in config
    assert f"'parameter_size': {8 * (9 + 3 * ranks)}" in config
    assert "'parameter_alignment': 64" in config
    assert "request_count" not in config
    assert "CCU_RUNTIME_SUPPORTED = False" in config
    root = config_paths[0].parent
    wrapper = _sole_file(root / "kernels" / "aiv", "*allgather*.cpp").read_text()
    assert "get_async_ctx(args)" in wrapper
    assert "CommContext *>(args[4])" in wrapper
    assert "submit_ccu_mission(ctx, comm, ccu_id, call_args)" in wrapper
    assert "{Arg::Kind::Window, rank, input_offset, bytes}" in wrapper
    assert "{Arg::Kind::Window, peer, target_offset, nranks * bytes}" in wrapper
    assert "{Arg::Kind::Scalar, 0, peer, 0}" in wrapper
    module = (root / "ccu/module_0.cpp").read_text()
    assert "CCU_WHILE(service != 0)" in module
    assert "SIMPLER_CCU_TASK_ARGUMENT_COUNT" in module
    assert "stable_input.addr = words[0]" in module and "stable_input.token = words[1]" in module
    assert "ccu::LoopGroup main_group" in module
    assert "ccu::LoopGroup remainder_group" in module
    assert (root / "ccu/allgather_schedule.hpp").is_file()
    assert f"constexpr uint32_t kRanks = {ranks}" in module
    assert "stable_output[rank].addr += words[3]" in module
    assert "ccu::LocalCopy(local_[slot][rank], source_[slot], length_[slot], event, bit)" in module
    assert "ccu::Write(binding_->channels[peer], remote_[slot][rank], source_[slot]," in module
    assert "CCU_IF(words[11 + 3 * slot] == peer)" in module
    assert "{{" not in wrapper + module
    orch = _sole_file(root / "orchestration", "*.cpp").read_text()
    assert orch.count("rt_submit_aiv_task") == 2
    if host:
        host_source = (output / "orchestration/host_orch.py").read_text()
        assert 'workspace_engines=("ccu",)' in host_source
        assert host_source.count(".add_scalar(") == 3, host_source
        assert host_source.count(".get_or_register_ccu(") == ranks
        assert host_source.rindex(".get_or_register_ccu(") < host_source.index("_submit_chip(")
        for rank in range(ranks):
            assert f"ccu_id=0, worker={rank})" in host_source
        assert host_source.count("ccu_bindings=[") == 1
        assert host_source.rindex(".get_or_register_ccu(") < host_source.index("for r")


@pytest.mark.parametrize("dispatch_ranks", [3, 5])
def test_ccu_allgather_host_rank_count_mismatch(tmp_path, dispatch_ranks):
    # Both undersized and enlarged domains would otherwise write wrong rows or
    # leave a native mission waiting for a rank that never participates.
    program = _ccu_allgather_source_program(host=True, ranks=4, dispatch_ranks=dispatch_ranks)
    with pytest.raises(ValueError, match="Host domain size must match target rank count 4"):
        ir.compile(
            program,
            output_dir=str(tmp_path),
            backend_type=BackendType.Ascend950,
            skip_ptoas=True,
        )


@pytest.mark.parametrize("ranks", [2, 3, 4, 8, 17, 64])
@pytest.mark.parametrize("native", [False, True], ids=["aiv-metadata", "native-hbm"])
def test_ccu_allgather_generated_abi_and_data_path(tmp_path, native, ranks):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required for the generated CCU contract test")
    templates = _resolve_builtin_template_dir(":pypto.runtime.builtins.ccu.allgather") / "templates"
    if native:
        template = "module.cpp.in"
        output = "ccu/generated_module.inc"
        variables = {"entry_symbol": "pypto_ccu_allgather_test"}
        headers = (
            "hcomm/ccu/ccu_primitives.hpp",
            "hcomm/ccu/ccu_res.h",
            "platform_comm/ccu/ccu_native_abi.h",
        )
        fixture = "ccu_allgather_native_test.cpp"
    else:
        template = "kernel.cpp.in"
        output = "kernels/aiv/generated_kernel.inc"
        variables = {"template_package": "ccu.allgather", "ccu_id": "7", "ctx_index": "4"}
        headers = ("pto/pto-inst.hpp", "async_kernel_api.h", "tensor.h")
        fixture = "ccu_allgather_wrapper_test.cpp"
    variables["nranks_cpp"] = str(ranks)
    (tmp_path / output).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "ccu").mkdir(exist_ok=True)
    (tmp_path / "ccu/allgather_schedule.hpp").write_text(
        _render_builtin_template(templates / "schedule.hpp.in", {})
    )
    (tmp_path / output).write_text(_render_builtin_template(templates / template, variables))
    # CPU primitives execute the actual emitted source. They do not emulate
    # HCOMM instruction synthesis, device scheduling or cross-rank progress.
    for name in headers:
        header = tmp_path / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text("")
    source = Path(__file__).with_name(fixture)
    binary = tmp_path / "contract_test"
    built = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wno-attributes",
            f"-DTEST_RANKS={ranks}",
            "-DSIMPLER_CCU_DOMAIN_ABI_VERSION=2",
            "-I",
            str(tmp_path),
            "-I",
            str(tmp_path / "ccu"),
            "-I",
            str(tmp_path / "kernels/aiv"),
            str(source),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_ccu_entry_ids_follow_first_use_and_are_submit_aware():
    # Synthetic specialization tags exercise two entries without claiming a
    # second hardware algorithm. Identity, not a function-name substring, selects CCU.
    span = ir.Span.unknown()
    kernels = []
    for name, variant in (("first", "b"), ("second", "a"), ("unused", "c")):
        definition = ir.Function(
            name + "_ccu",
            [],
            [],
            ir.ReturnStmt([], span),
            span,
            type=ir.FunctionType.CCU,
            attrs={
                "builtin_template_dir": ":pypto.runtime.builtins.ccu.allgather",
                "builtin_template_vars": f"fixture_variant={variant},nranks_cpp=2",
            },
        )
        params = [ir.Var(f"p{i}", ir.ScalarType(pl.UINT32), span) for i in range(4)]
        params += [ir.Var(f"ctx{i}", ir.CommCtxType.get(), span) for i in range(3)]
        request = ir.Call(ir.GlobalVar(definition.name), [], {}, {}, ir.TupleType([]), span)
        wrapper = ir.Function(
            name,
            params,
            [],
            ir.EvalStmt(request, span),
            span,
            type=ir.FunctionType.AIV,
            attrs={"builtin_template_dir": ":pypto.runtime.builtins.ccu.allgather"},
        )
        kernels.extend([definition, wrapper])
    # Ordinary AIV builtins and CCU wrappers use the same artifact contract.
    # An unreferenced ordinary definition must not produce an artifact either.
    kernels.extend(
        ir.Function(
            name,
            [],
            [],
            ir.ReturnStmt([], span),
            span,
            type=ir.FunctionType.AIV,
            attrs={
                "builtin_template_dir": ":pypto.runtime.builtins.collectives.all_to_all_v",
                "builtin_template_vars": "dtype_cpp=float",
            },
        )
        for name in ("ordinary", "unused_ordinary")
    )
    calls: list[ir.Stmt] = [
        ir.EvalStmt(ir.Call(ir.GlobalVar("ordinary"), [], ir.TupleType([]), span), span),
        ir.EvalStmt(
            ir.Call(ir.GlobalVar("first"), [ir.ConstInt(0, pl.UINT32, span)] * 7, ir.TupleType([]), span),
            span,
        ),
        ir.EvalStmt(
            ir.Submit(
                ir.GlobalVar("second"), [ir.ConstInt(0, pl.UINT32, span)] * 7, [], ir.TupleType([]), span
            ),
            span,
        ),
    ]
    chip = ir.Function("chip", [], [], ir.SeqStmts(calls, span), span, type=ir.FunctionType.Orchestration)
    # Another entry calls the otherwise-unused implementation. Its mission
    # must not leak into chip's descriptors, even in an unsplit Program.
    other = ir.Function(
        "other",
        [],
        [],
        ir.EvalStmt(
            ir.Call(ir.GlobalVar("unused"), [ir.ConstInt(0, pl.UINT32, span)] * 7, ir.TupleType([]), span),
            span,
        ),
        span,
        type=ir.FunctionType.Orchestration,
    )
    program = ir.Program([chip, other, *kernels], "ids", span)
    program = passes.derive_call_directions()(program)
    program = passes.materialize_runtime_scopes()(program)
    chip = next(f for f in program.functions.values() if f.name == "chip")
    result = codegen.generate_orchestration(program, chip)
    entries = result.ccu_modules
    assert [(e.ccu_id, e.wrappers) for e in entries] == [(0, ["first"]), (1, ["second"])]
    assert all(e.parameter_size == 8 * (9 + 3 * 2) and e.parameter_alignment == 64 for e in entries)
    specs = result.builtin_sources
    assert len({spec.output_path for spec in specs}) == len(specs) == 6
    sources = _render_builtin_kernel_sources(specs)
    assert "kernels/aiv/ordinary.cpp" in sources
    assert "kernels/aiv/unused_ordinary.cpp" not in sources
    assert "submit_ccu_mission(ctx, comm, ccu_id, call_args)" in sources["kernels/aiv/first.cpp"]
    assert "submit_ccu_mission(ctx, comm, ccu_id, call_args)" in sources["kernels/aiv/second.cpp"]
    assert "constexpr uint32_t ccu_id = 0;" in sources["kernels/aiv/first.cpp"]
    assert "constexpr uint32_t ccu_id = 1;" in sources["kernels/aiv/second.cpp"]
    assert "kernels/aiv/unused.cpp" not in sources
    assert set(p for p in sources if p.startswith("ccu/")) == {
        "ccu/module_0.cpp",
        "ccu/module_1.cpp",
        "ccu/allgather_schedule.hpp",
    }
    reordered = ir.Program([*reversed(kernels), chip], "ids", span)
    reordered_result = codegen.generate_orchestration(reordered, chip)
    assert sources == _render_builtin_kernel_sources(reordered_result.builtin_sources)


def test_ccu_multi_callable_entries_and_shape_deduplication(tmp_path):
    @pl.program
    class Multi:
        @pl.function(type=pl.FunctionType.Orchestration)
        def first(
            self,
            data: pld.DistributedTensor[[1, 64], pl.BF16],
            target: pl.InOut[pld.DistributedTensor[[2, 64], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ):
            target = pld.tensor.allgather(data, target, signal, engine="ccu")
            return target

        @pl.function(type=pl.FunctionType.Orchestration)
        def second(
            self,
            data: pld.DistributedTensor[[1, 64], pl.BF16],
            target: pl.InOut[pld.DistributedTensor[[2, 64], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
            data2: pld.DistributedTensor[[1, 32], pl.BF16],
            target2: pl.InOut[pld.DistributedTensor[[2, 32], pl.BF16]],
            signal2: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ):
            target = pld.tensor.allgather(data, target, signal, engine="ccu")
            target2 = pld.tensor.allgather(data2, target2, signal2, engine="ccu")
            return target, target2

    _compile_to_files(
        Multi,
        contract=ArtifactExecutionMode.PROGRAM,
        output_dir=str(tmp_path),
        backend_type=BackendType.Ascend950,
        skip_ptoas=True,
    )
    # Native lifetime must not change with the number of static call sites.
    assert (tmp_path / "next_levels/first/ccu/module_0.cpp").read_text() == (
        tmp_path / "next_levels/second/ccu/module_0.cpp"
    ).read_text()
    for name, count in (("first", 1), ("second", 2)):
        root = tmp_path / "next_levels" / name
        config = (root / "kernel_config.py").read_text()
        assert "'ccu_id': 0" in config and "'ccu_id': 1" not in config
        assert "request_count" not in config
        assert len(list((root / "ccu").glob("*.cpp"))) == 1
        assert "CCU_WHILE(service != 0)" in (root / "ccu/module_0.cpp").read_text()
        wrappers = list((root / "kernels/aiv").glob("*.cpp"))
        assert len(wrappers) == count
        assert all("submit_ccu_mission(ctx, comm, ccu_id, call_args)" in p.read_text() for p in wrappers)
        source = _sole_file(root / "orchestration", "*.cpp").read_text()
        tokens = re.findall(r"\.add_inout\((_ccu_order_\w*)\)", source)
        assert len(tokens) == count and len(set(tokens)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
