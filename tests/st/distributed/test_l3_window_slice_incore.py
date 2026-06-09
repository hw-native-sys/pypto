# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: in-core ``pld.DistributedTensor`` (window) slice / slice-assign.

Regression for issue #1694. #1685/#1672 made ``tensor.slice`` / ``assemble`` /
slice-assign accept a window at the type level, but ``ConvertTensorToTileOps``
still rejected a window slice that appears **inside an InCore scope**
(``pl.at(CORE_GROUP)`` / ``FunctionType.InCore``), crashing with::

    pypto.InternalError: tensor.slice conversion: unexpected input type:
        DistributedTensorType

Inside an InCore scope a window is simply this rank's local GM, so its slice /
slice-assign must lower like a plain ``Tensor`` slice (``tile.load`` /
``tile.store``) — letting a window be staged with the assemble idiom instead of
falling back to ``pl.load`` / ``pl.store``.

Two directions are exercised end-to-end (single-rank-local work, replicated
across the ring so the existing 2-device harness drives it):

* **window-as-source** (dsv4 ``dispatch_ep`` idiom): copy column 0 of a window
  into a plain output via ``out[:, :] = win[:, 0:1]``.
* **window-as-target** (dsv4 ``combine_ep`` staging idiom): stage a local input
  into a window slice via ``win[:, :] = inp[:, :]``, then read it back.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

N = 16  # rows
W = 8  # window width (padded columns)
COL = 0  # column extracted by the window-as-source read


def _build_window_source_slice_program():
    """window-as-source: ``out[:, :] = win[:, COL:COL+1]`` inside an InCore scope.

    The window is staged from the local input via the proven ``pl.load`` /
    ``pl.store`` path; the op under test is the in-core column slice-assign that
    reads the window (the exact ``dispatch_ep`` idiom that crashed before the
    fix).
    """

    @pl.program
    class WindowSourceSlice:
        @pl.function(type=pl.FunctionType.InCore)
        def col_step(
            self,
            inp: pl.Tensor[[N, W], pl.FP32],
            out: pl.Out[pl.Tensor[[N, 1], pl.FP32]],
            win: pld.DistributedTensor[[N, W], pl.FP32],
        ) -> pl.Tensor[[N, 1], pl.FP32]:
            # Stage the local input into this rank's own window (load/store path).
            staged = pl.load(inp, [0, 0], [N, W])
            win = pl.store(staged, [0, 0], win)
            # Issue #1694: read column COL of the local window via an in-core
            # slice-assign. ``win[:, COL:COL+1]`` is a window ``tensor.slice``;
            # the LHS is a plain-tensor slice-assign (``tensor.assemble``).
            out[0:N, COL : COL + 1] = win[0:N, COL : COL + 1]
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[N, W], pl.FP32],
            out: pl.Out[pl.Tensor[[N, 1], pl.FP32]],
            win: pld.DistributedTensor[[N, W], pl.FP32],
        ) -> pl.Tensor[[N, 1], pl.FP32]:
            return self.col_step(inp, out, win)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, N, W], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, N, 1], pl.FP32]],
        ) -> pl.Tensor[[2, N, 1], pl.FP32]:
            win_buf = pld.alloc_window_buffer(N * W * 4)  # N x W x FP32

            for r in pl.range(pld.world_size()):
                win = pld.window(win_buf, [N, W], dtype=pl.FP32)
                self.chip_orch(inputs[r], outputs[r], win, device=r)
            return outputs

    return WindowSourceSlice


def _build_window_target_slice_program():
    """window-as-target: ``win[:, :] = inp[:, :]`` inside an InCore scope.

    The op under test is the slice-assign whose *target* is the window
    (``tensor.assemble`` with a DistributedTensorType target). The staged window
    is then read back with the proven ``pl.load`` / ``pl.store`` path to surface
    a golden output.
    """

    @pl.program
    class WindowTargetSlice:
        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[N, W], pl.FP32],
            out: pl.Out[pl.Tensor[[N, W], pl.FP32]],
            win: pld.DistributedTensor[[N, W], pl.FP32],
        ) -> pl.Tensor[[N, W], pl.FP32]:
            # Issue #1694: stage the local input into the window via a
            # slice-assign whose target is the window (no pl.store fallback).
            win[0:N, 0:W] = inp[0:N, 0:W]
            # Read the staged window back through the proven load/store path.
            recv = pl.load(win, [0, 0], [N, W])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[N, W], pl.FP32],
            out: pl.Out[pl.Tensor[[N, W], pl.FP32]],
            win: pld.DistributedTensor[[N, W], pl.FP32],
        ) -> pl.Tensor[[N, W], pl.FP32]:
            return self.stage_step(inp, out, win)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, N, W], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, N, W], pl.FP32]],
        ) -> pl.Tensor[[2, N, W], pl.FP32]:
            win_buf = pld.alloc_window_buffer(N * W * 4)

            for r in pl.range(pld.world_size()):
                win = pld.window(win_buf, [N, W], dtype=pl.FP32)
                self.chip_orch(inputs[r], outputs[r], win, device=r)
            return outputs

    return WindowTargetSlice


def _build_window_compute_reduce_program():
    """combine_ep reduce: ``out = sh + cast(window)`` inside an InCore scope.

    Issue #1694 follow-on — compute directly on a window-derived operand. The
    window (FP16) is staged from a local input, then ``pl.cast`` + ``pl.add``
    consume the window slice at tensor level (no ``pl.load`` / ``pl.store``
    fallback), matching the single-card combine kernel.
    """

    @pl.program
    class WindowComputeReduce:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            sh: pl.Tensor[[N, W], pl.FP32],
            routed_in: pl.Tensor[[N, W], pl.FP16],
            out: pl.Out[pl.Tensor[[N, W], pl.FP32]],
            win: pld.DistributedTensor[[N, W], pl.FP16],
        ) -> pl.Tensor[[N, W], pl.FP32]:
            # Stage the FP16 routed input into this rank's own window.
            staged = pl.load(routed_in, [0, 0], [N, W])
            win = pl.store(staged, [0, 0], win)
            # Issue #1694 follow-on: cast + add directly on the window slice.
            y_slice = pl.tensor.slice(win, [N, W], [0, 0])
            y_f32 = pl.cast(y_slice, pl.FP32)
            acc = pl.add(sh, y_f32)
            out[0:N, 0:W] = acc
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            sh: pl.Tensor[[N, W], pl.FP32],
            routed_in: pl.Tensor[[N, W], pl.FP16],
            out: pl.Out[pl.Tensor[[N, W], pl.FP32]],
            win: pld.DistributedTensor[[N, W], pl.FP16],
        ) -> pl.Tensor[[N, W], pl.FP32]:
            return self.reduce_step(sh, routed_in, out, win)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            sh_in: pl.Tensor[[2, N, W], pl.FP32],
            routed_in: pl.Tensor[[2, N, W], pl.FP16],
            outputs: pl.Out[pl.Tensor[[2, N, W], pl.FP32]],
        ) -> pl.Tensor[[2, N, W], pl.FP32]:
            win_buf = pld.alloc_window_buffer(N * W * 2)  # N x W x FP16

            for r in pl.range(pld.world_size()):
                win = pld.window(win_buf, [N, W], dtype=pl.FP16)
                self.chip_orch(sh_in[r], routed_in[r], outputs[r], win, device=r)
            return outputs

    return WindowComputeReduce


def _ramp_inputs() -> torch.Tensor:
    """Two distinct [N, W] ramps, one per rank, so a wrong column/order shows up."""
    base = torch.arange(N * W, dtype=torch.float32).reshape(1, N, W)
    return torch.cat([base, base + 1000.0], dim=0)


class TestL3WindowSourceSlice:
    """In-core window-as-source slice read (dsv4 dispatch_ep idiom)."""

    def test_window_column_slice_read(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"window slice st needs 2 devices, got {device_ids}")

        program = _build_window_source_slice_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        inputs = _ramp_inputs()
        outputs = torch.zeros((2, N, 1), dtype=torch.float32)

        compiled(inputs, outputs)

        # Each rank extracts column COL of its own input.
        expected = inputs[:, :, COL : COL + 1].clone()
        assert torch.allclose(outputs, expected), (
            f"window column slice mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


class TestL3WindowTargetSlice:
    """In-core window-as-target slice-assign staging (dsv4 combine_ep idiom)."""

    def test_local_stage_into_window_slice(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"window slice st needs 2 devices, got {device_ids}")

        program = _build_window_target_slice_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        inputs = _ramp_inputs()
        outputs = torch.zeros((2, N, W), dtype=torch.float32)

        compiled(inputs, outputs)

        # Window staging is an identity copy: out[r] == inp[r].
        assert torch.allclose(outputs, inputs), (
            f"window stage mismatch: max diff = {(outputs - inputs).abs().max().item()}"
        )


class TestL3WindowComputeReduce:
    """In-core compute on a window operand (dsv4 combine_ep reduce)."""

    def test_cast_add_on_window(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"window compute st needs 2 devices, got {device_ids}")

        program = _build_window_compute_reduce_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        sh = _ramp_inputs()
        routed = (_ramp_inputs() * 0.5).to(torch.float16)
        outputs = torch.zeros((2, N, W), dtype=torch.float32)

        compiled(sh, routed, outputs)

        # out[r] == sh[r] + cast(routed[r], fp32); FP16 staging needs a tolerance.
        expected = sh + routed.to(torch.float32)
        assert torch.allclose(outputs, expected, rtol=1e-2, atol=1e-2), (
            f"window compute reduce mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
