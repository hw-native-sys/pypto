# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end ``RunTiming`` surfacing across every L2 dispatch entry point (issue #1679).

The unit-level plumbing test (``tests/ut/runtime/test_run_timing_plumbing.py``)
mocks the runtime and only proves the ``RunTiming`` *object* is forwarded layer
by layer. This system test runs a real kernel on device/simulator and asserts
the *measured* timing actually surfaces on each user-reachable L2 call path:

* ``JITFunction.__call__``      — ``kernel(*args, config=...)`` → ``kernel.last_run_timing``
* ``CompiledProgram.__call__``  — one-shot ``Worker`` path → ``compiled.last_run_timing``
* ``CompiledProgram.__call__``  — active-``ChipWorker`` reuse path (``_run_chip``)
* ``execute_compiled``          — direct call returns the ``RunTiming``
* ``execute_on_device``         — direct call, one-shot **and** ChipWorker-reuse branch

L3 distributed paths live in ``tests/st/distributed/test_l3_run_timing.py``: a
level-2 ``Worker`` session leaks runtime state that breaks a later level-3
``Worker`` init in the same process, so the two levels must never share a
``pytest`` process (mirrors the split documented in ``test_l2_multi_orch.py``).

Timing semantics asserted here (L2 single-task, default ``PTO2_PROFILING`` build):

* ``host_wall_us > 0``      — a real host-side dispatch always takes nonzero time.
* ``device_wall_us > 0``    — on-NPU orchestrator wall, available from the swimlane
  shared-region path whenever ``PTO2_PROFILING`` is compiled in (the default).
* ``host_wall_us >= device_wall_us`` — the host wall wraps the device wall.

L3 differs: ``device_wall_us`` is ``0`` there (per-task device cycles are not
aggregated in the ring scheduler) — that contrast is asserted in the L3 file.
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.runtime import ChipWorker, RunConfig, execute_compiled
from pypto.runtime.device_runner import (
    build_orch_args_from_inputs,
    compile_and_assemble,
    execute_on_device,
)
from pypto.runtime.task_interface import RunTiming

_M = 128


# A @pl.jit entry — drives the JITFunction.__call__ path (A).
@pl.jit
def add_kernel(
    a: pl.Tensor[[_M, _M], pl.FP32],
    b: pl.Tensor[[_M, _M], pl.FP32],
    c: pl.Out[pl.Tensor[[_M, _M], pl.FP32]],
):
    with pl.incore():
        ta = pl.load(a, [0, 0], [_M, _M])
        tb = pl.load(b, [0, 0], [_M, _M])
        pl.store(pl.add(ta, tb), [0, 0], c)
    return c


# A @pl.program with a single orchestration — drives the CompiledProgram /
# execute_compiled / execute_on_device paths (B, C, E, F). ``ir.compile`` both
# returns the ``CompiledProgram`` and writes a work_dir that ``execute_compiled``
# / ``compile_and_assemble`` can re-consume.
@pl.program
class AddProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[_M, _M], pl.FP32],
        b: pl.Tensor[[_M, _M], pl.FP32],
        c: pl.Out[pl.Tensor[[_M, _M], pl.FP32]],
    ) -> pl.Tensor[[_M, _M], pl.FP32]:
        ta = pl.load(a, [0, 0], [_M, _M])
        tb = pl.load(b, [0, 0], [_M, _M])
        return pl.store(pl.add(ta, tb), [0, 0], c)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch_add(
        self,
        a: pl.Tensor[[_M, _M], pl.FP32],
        b: pl.Tensor[[_M, _M], pl.FP32],
        c: pl.Out[pl.Tensor[[_M, _M], pl.FP32]],
    ) -> pl.Tensor[[_M, _M], pl.FP32]:
        return self.tile_add(a, b, c)


def _assert_real_dispatch(timing) -> None:
    """Universal invariant: every real L2 dispatch yields a usable RunTiming."""
    assert isinstance(timing, RunTiming), (
        f"expected a RunTiming from a real dispatch, got {type(timing).__name__}: {timing!r}"
    )
    assert timing.host_wall_us > 0.0, f"host_wall_us must be > 0, got {timing.host_wall_us}"
    assert timing.host_wall_ns > 0, f"host_wall_ns must be > 0, got {timing.host_wall_ns}"


def _report(label: str, timing) -> None:
    """Print the measured wall times so they surface in the CI log (``-s``)."""
    if isinstance(timing, RunTiming):
        print(
            f"[run-timing] {label}: host_wall_us={timing.host_wall_us:.3f} "
            f"device_wall_us={timing.device_wall_us:.3f}"
        )
    else:
        print(f"[run-timing] {label}: no RunTiming ({timing!r})")


def _assert_l2_semantics(timing, label: str) -> None:
    """L2 single-task semantics: device wall present and wrapped by host wall."""
    _assert_real_dispatch(timing)
    _report(label, timing)
    # device_wall is sourced from the swimlane shared-region and is always
    # available on the default PTO2_PROFILING build (see the runtime ST
    # ``examples/workers/l2/vector_add/test_run_timing.py``).
    assert timing.device_wall_us > 0.0, (
        f"device_wall_us must be > 0 on the default PTO2_PROFILING build, "
        f"got {timing.device_wall_us}"
    )
    assert timing.host_wall_us >= timing.device_wall_us, (
        f"host_wall_us ({timing.host_wall_us}) must wrap device_wall_us ({timing.device_wall_us})"
    )


def _inputs():
    a = torch.full((_M, _M), 2.0, dtype=torch.float32)
    b = torch.full((_M, _M), 3.0, dtype=torch.float32)
    c = torch.zeros((_M, _M), dtype=torch.float32)
    return a, b, c


_EXPECTED = torch.full((_M, _M), 5.0, dtype=torch.float32)


def _compile_add(test_config, tmp_path):
    """Compile ``AddProgram`` once; return ``(compiled, work_dir)``.

    ``work_dir`` is returned as the ``pathlib.Path`` ``tmp_path`` (not a
    ``str``): ``compile_and_assemble`` builds paths via ``work_dir / "..."``
    and only accepts a ``Path``, while ``execute_compiled`` coerces either.
    """
    compiled = ir.compile(
        AddProgram,
        output_dir=str(tmp_path),
        platform=test_config.platform,
    )
    return compiled, tmp_path


class TestL2RunTimingSurface:
    """``RunTiming`` surfaces on every user-reachable L2 dispatch entry point."""

    def test_jit_function_call_surfaces_timing(self, test_config):
        """(A) ``@pl.jit`` ``kernel(*args)`` populates ``kernel.last_run_timing``."""
        add_kernel._cache.clear()
        add_kernel.last_run_timing = None

        a, b, c = _inputs()
        add_kernel(a, b, c, config=test_config)

        torch.testing.assert_close(c, _EXPECTED, rtol=1e-5, atol=1e-5)
        _assert_l2_semantics(add_kernel.last_run_timing, "JITFunction.__call__")

    def test_compiled_program_call_one_shot_surfaces_timing(self, test_config, tmp_path):
        """(B) ``compiled(*args)`` with no active worker → one-shot ``Worker`` path."""
        compiled, _ = _compile_add(test_config, tmp_path)

        a, b, c = _inputs()
        compiled(a, b, c, config=test_config)

        torch.testing.assert_close(c, _EXPECTED, rtol=1e-5, atol=1e-5)
        _assert_l2_semantics(compiled.last_run_timing, "CompiledProgram.__call__ (one-shot)")

    def test_compiled_program_call_reuse_path_surfaces_timing(self, test_config, tmp_path):
        """(C) ``compiled(*args)`` inside ``with ChipWorker`` → reuse (``_run_chip``).

        Constructing the ``ChipWorker`` with ``runtime=compiled.runtime_name``
        makes ``execute_on_device``'s ``ChipWorker.current(...)`` lookup match,
        so the dispatch genuinely takes the reuse branch rather than silently
        falling back to a one-shot ``Worker``.
        """
        compiled, _ = _compile_add(test_config, tmp_path)

        a, b, c = _inputs()
        worker_cfg = RunConfig(platform=test_config.platform, device_id=test_config.device_id)
        with ChipWorker(config=worker_cfg, runtime=compiled.runtime_name):
            compiled(a, b, c, config=test_config)

        torch.testing.assert_close(c, _EXPECTED, rtol=1e-5, atol=1e-5)
        _assert_l2_semantics(compiled.last_run_timing, "CompiledProgram.__call__ (reuse)")

    def test_execute_compiled_returns_timing(self, test_config, tmp_path):
        """(E) ``execute_compiled`` returns the ``RunTiming`` to its caller."""
        _, work_dir = _compile_add(test_config, tmp_path)

        a, b, c = _inputs()
        timing = execute_compiled(
            work_dir,
            [a, b, c],
            platform=test_config.platform,
            device_id=test_config.device_id,
        )

        torch.testing.assert_close(c, _EXPECTED, rtol=1e-5, atol=1e-5)
        _assert_l2_semantics(timing, "execute_compiled")

    def test_execute_on_device_one_shot_returns_timing(self, test_config, tmp_path):
        """(F-1) ``execute_on_device`` with no active worker → one-shot ``Worker``."""
        _, work_dir = _compile_add(test_config, tmp_path)
        chip_callable, runtime_name, runtime_cfg = compile_and_assemble(
            work_dir, test_config.platform, pto_isa_commit=test_config.pto_isa_commit
        )

        a, b, c = _inputs()
        orch_args, _, _, outputs = build_orch_args_from_inputs(
            [("a", a), ("b", b), ("c", c)], {"c"}
        )
        timing = execute_on_device(
            chip_callable,
            orch_args,
            test_config.platform,
            runtime_name,
            test_config.device_id,
            block_dim=runtime_cfg.get("block_dim"),
            aicpu_thread_num=runtime_cfg.get("aicpu_thread_num"),
        )

        torch.testing.assert_close(outputs["c"], _EXPECTED, rtol=1e-5, atol=1e-5)
        _assert_l2_semantics(timing, "execute_on_device (one-shot)")

    def test_execute_on_device_reuse_path_returns_timing(self, test_config, tmp_path):
        """(F-2) ``execute_on_device`` inside ``with ChipWorker`` → reuse (``_run_chip``)."""
        _, work_dir = _compile_add(test_config, tmp_path)
        chip_callable, runtime_name, runtime_cfg = compile_and_assemble(
            work_dir, test_config.platform, pto_isa_commit=test_config.pto_isa_commit
        )

        a, b, c = _inputs()
        orch_args, _, _, outputs = build_orch_args_from_inputs(
            [("a", a), ("b", b), ("c", c)], {"c"}
        )
        worker_cfg = RunConfig(platform=test_config.platform, device_id=test_config.device_id)
        with ChipWorker(config=worker_cfg, runtime=runtime_name):
            timing = execute_on_device(
                chip_callable,
                orch_args,
                test_config.platform,
                runtime_name,
                test_config.device_id,
                block_dim=runtime_cfg.get("block_dim"),
                aicpu_thread_num=runtime_cfg.get("aicpu_thread_num"),
            )

        torch.testing.assert_close(outputs["c"], _EXPECTED, rtol=1e-5, atol=1e-5)
        _assert_l2_semantics(timing, "execute_on_device (reuse)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
