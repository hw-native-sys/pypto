# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The ``[SplitPipeHazard]`` guard fires on a split V2C push and nowhere else.

Defect C (device-verified 2026-10): a data-parallel ``pl.split`` region whose
duplicated AIV lanes push a *computed* tile to the cube (``tile.tpush_to_aic``
carrying a split code) deadlocks the device (scheduler timeout, lane poisoned)
or silently corrupts a few tiles at dense scale on the current PTOAS/pto-isa
split-pipe protocol; a FLOAT32 ``pl.split`` region that never pushes to the cube
is unaffected. The failing lane/consumer synchronization lives inside the closed
TPUSH/TPOP ``TileSplitAxis`` pipe protocol, so ``ExpandMixedKernel`` cannot
repair it — it warns at the boundary emission point and every affected kernel
drops ``pl.split``.

These tests pin the guard:

* the hazardous program warns exactly once, and the message carries the remedy;
* the same body without the split region stays silent;
* the guard is diagnostics only — the warning does not change what PTOAS is
  given (the split-coded push is still emitted).

They are device-free: ``JITFunction.compile(..., config=RunConfig(...))`` lowers
the programs through PTO codegen without dispatching them to the runtime. An
autouse fixture fails if runtime execution is attempted. The guard lives in the
lowering pass, so the warning is observable without a device. It travels the C++
``LOG_WARN`` channel, which writes to ``std::cerr`` from native code, so it is
read with pytest's ``capfd`` (file-descriptor level) rather than ``capsys``.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false

import re
import shutil

import pypto.language as pl
import pytest
import torch
from pypto import LogLevel, backend, set_log_level
from pypto.runtime import RunConfig

WARNING_TAG = "[SplitPipeHazard]"
N = 128


@pytest.fixture(autouse=True)
def _forbid_runtime_dispatch(monkeypatch):
    """Fail before runtime dispatch if a diagnostics test tries to execute."""

    def fail_runtime_dispatch(*args, **kwargs):
        pytest.fail("Split-pipe hazard diagnostics must compile without runtime execution")

    monkeypatch.setattr("pypto.runtime.runner._execute_compiled", fail_runtime_dispatch)


@pytest.fixture(autouse=True)
def _warn_level_and_clean_backend():
    """Make WARN-level output visible and leave no backend pinned behind.

    ``conftest.py`` restores the process-global log level after each test, so
    raising it here is contained; setting it explicitly keeps the stderr
    assertions independent of ``PYPTO_LOG_LEVEL`` and of test order.
    """
    backend.reset_for_testing()
    set_log_level(LogLevel.WARN)
    yield
    backend.reset_for_testing()


@pl.jit
def split_v2c_push(
    a: pl.Tensor[[N, N], pl.FP8E4M3FN],
    b: pl.Tensor[[N, N], pl.FP32],
    c: pl.Out[pl.Tensor[[N, N], pl.FP32]],
):
    """Data-parallel split region whose cast result the cube consumes.

    ``pl.cast`` runs on the vector side, so its result crosses to the cube as an
    op-driven V2C boundary — the split-coded push the guard is about.
    """
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="split_v2c_push_tile",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        at = pl.load(a, [0, 0], [N, N])
        bt = pl.load(b, [0, 0], [N, N])
        bt2 = pl.cast(bt, pl.FP8E4M3FN)
        acc = pl.matmul(at, bt2)
        c = pl.store(acc, [0, 0], c)
    return c


@pl.jit
def plain_v2c_push(
    a: pl.Tensor[[N, N], pl.FP8E4M3FN],
    b: pl.Tensor[[N, N], pl.FP32],
    c: pl.Out[pl.Tensor[[N, N], pl.FP32]],
):
    """The identical body without the split region — the silent control."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="plain_v2c_push_tile"):
        at = pl.load(a, [0, 0], [N, N])
        bt = pl.load(b, [0, 0], [N, N])
        bt2 = pl.cast(bt, pl.FP8E4M3FN)
        acc = pl.matmul(at, bt2)
        c = pl.store(acc, [0, 0], c)
    return c


def _compile(fn, out_dir, *tensors) -> str:
    """Lower *fn* through PTO codegen on a5 and return the emitted ``.pto`` text."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cfg = RunConfig(
        platform="a5",
        save_kernels=True,
        save_kernels_dir=str(out_dir),
    )
    if tensors:
        fn.compile(*tensors, config=cfg)
    else:
        fn.compile(
            torch.zeros(N, N, dtype=torch.float8_e4m3fn),
            torch.zeros(N, N, dtype=torch.float32),
            torch.zeros(N, N, dtype=torch.float32),
            config=cfg,
        )
    ptos = sorted(out_dir.rglob("*.pto"))
    assert ptos, f"codegen emitted no .pto under {out_dir}"
    return "\n".join(p.read_text() for p in ptos)


def _warnings(capfd) -> list[str]:
    """Drain captured stderr and return the split-pipe hazard lines."""
    err = capfd.readouterr().err
    return [line for line in err.splitlines() if WARNING_TAG in line]


def test_split_v2c_push_warns_once_and_names_the_remedy(capfd, tmp_path):
    pto = _compile(split_v2c_push, tmp_path / "split")

    warnings = _warnings(capfd)
    assert len(warnings) == 1, f"expected exactly one hazard warning, got {warnings}"
    assert "drop pl.split" in warnings[0].lower(), warnings[0]
    # The guard is diagnostics only: the split-coded push must still be emitted,
    # so the warning cannot be mistaken for a lowering that silently changed the
    # program PTOAS sees. The split coding and the push must be the SAME
    # instruction (PR#2987 review: separate substring assertions could pass
    # with an unsplit push plus an unrelated split elsewhere).
    assert re.search(r"pto\.tpush_to_aic\([^)]*\)\s*\{split = 1\}", pto), pto[:2000]


def test_the_same_body_without_a_split_region_stays_silent(capfd, tmp_path):
    pto = _compile(plain_v2c_push, tmp_path / "plain")

    assert _warnings(capfd) == []
    assert "pto.tpush_to_aic" in pto, pto[:2000]


@pl.jit
def split_without_v2c_push(
    b: pl.Tensor[[N, N], pl.FP32],
    c: pl.Out[pl.Tensor[[N, N], pl.FP32]],
):
    """Split region whose body stays on the vector side — the silent control.

    The guard is about split regions that PUSH a computed tile to the cube;
    a split whose body never crosses cores (load + store, no cube consumer)
    has no V2C push to hazard and must stay silent (PR#2987 review nitpick).
    """
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="split_without_v2c_push_tile",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        bt = pl.load(b, [0, 0], [N, N])
        c = pl.store(bt, [0, 0], c)
    return c


def test_split_region_without_a_v2c_push_stays_silent(capfd, tmp_path):
    pto = _compile(
        split_without_v2c_push,
        tmp_path / "split-no-push",
        torch.zeros(N, N, dtype=torch.float32),
        torch.zeros(N, N, dtype=torch.float32),
    )

    assert _warnings(capfd) == []
    assert "pto.tpush_to_aic" not in pto, pto[:2000]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
