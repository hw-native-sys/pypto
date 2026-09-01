# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared pytest behavior for distributed system tests."""

import hashlib
import itertools
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def disable_runtime_execution_in_codegen_only(request, monkeypatch) -> None:
    """Let distributed tests compile, then skip at every public execution edge."""
    if not request.config.getoption("--codegen-only"):
        return

    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415
    from pypto.runtime.distributed_runner import DistributedWorker  # noqa: PLC0415

    def skip_execution(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        pytest.skip("--codegen-only disables distributed runtime execution")

    monkeypatch.setattr("pypto.runtime.runner.execute_compiled", skip_execution)
    monkeypatch.setattr("pypto.runtime.distributed_runner.execute_distributed", skip_execution)
    monkeypatch.setattr(DistributedCompiledProgram, "prepare", skip_execution)
    monkeypatch.setattr(DistributedWorker, "__init__", skip_execution)


def _artifact_dir(store: Path, nodeid: str, call_index: int) -> Path:
    """Return this test's slot for its *call_index*-th ``ir.compile``.

    Keyed by test id rather than by content, which is sound only because both
    passes run from one checkout inside a single CI step: the sources cannot
    change between them, so a slot's artifact is by construction the one this
    test would have compiled. It is NOT a general compile cache — reusing a
    store across commits would serve stale binaries (the generated-kernel binary
    cache is keyed by position, not content). CI creates the store fresh per job
    and deletes it after; keep it that way.

    The digest keeps the path short and filesystem-safe; parametrised node ids
    carry ``[...]``, ``/`` and ``::``.
    """
    return store / f"{hashlib.sha256(nodeid.encode()).hexdigest()[:16]}_{call_index}"


@pytest.fixture(autouse=True)
def _precompiled_artifacts(request, monkeypatch) -> None:
    """Route ``ir.compile`` through the ``--precompile-dir`` artifact store.

    Splits a distributed run into the half that needs no card and the half that
    does. ``--precompile-only`` (pass 1) compiles each program into its slot and
    builds the chip binaries there via
    :meth:`DistributedCompiledProgram.build_binaries`, then skips before any
    device work — so it can fan out under ``-n`` on a card-free host. A later run
    naming the same store (pass 2) rebinds each slot with
    :meth:`DistributedCompiledProgram.from_dir` and never recompiles, leaving the
    borrowed cards to do only device work.

    The artifact does not depend on the cards it will run on: ``ir.compile``
    consumes ``distributed_config`` only after codegen, so pass 2 hands its real
    ``DistributedConfig`` (the actually-borrowed ``device_ids``) to ``from_dir``.

    Degrades to a plain compile whenever a slot is missing — a test whose compile
    count or node id differs between the passes still runs, just without the
    saving. Non-distributed programs are left alone: they write no
    ``distributed_meta.json``, so no slot ever matches them.
    """
    store_opt = request.config.getoption("--precompile-dir")
    if not store_opt:
        return

    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

    store = Path(store_opt)
    precompile_only = request.config.getoption("--precompile-only")
    nodeid = request.node.nodeid
    call_index = itertools.count()
    real_compile = ir.compile

    def compile_via_store(program: Any, **kwargs: Any) -> Any:
        if kwargs.get("output_dir") is not None:
            # The caller picked its own build directory and reads artifacts back
            # out of it (``test_l3_manual`` hands that path to a manual L2/L3
            # flow). Redirecting it into a slot would break the test, so this
            # call keeps its directory and takes no slot — a decision both passes
            # reach identically, which is what keeps their slot indices aligned.
            return real_compile(program, **kwargs)
        slot = _artifact_dir(store, nodeid, next(call_index))
        if not precompile_only and (slot / "distributed_meta.json").exists():
            return DistributedCompiledProgram.from_dir(
                slot,
                platform=kwargs.get("platform"),
                distributed_config=kwargs.get("distributed_config"),
            )
        compiled = real_compile(program, output_dir=str(slot), **kwargs)
        if precompile_only and isinstance(compiled, DistributedCompiledProgram):
            compiled.build_binaries()
        return compiled

    monkeypatch.setattr(ir, "compile", compile_via_store)


@pytest.fixture(autouse=True)
def _skip_device_work_when_precompiling(request, monkeypatch) -> None:
    """Under ``--precompile-only``, stop each test at its first device edge.

    Mirrors :func:`disable_runtime_execution_in_codegen_only`, but deliberately
    later: the compile *and* the chip-binary build must both happen (that is the
    point of the pass), so only the execution edges are cut.

    ``simpler.worker.Worker`` is cut too, which is what makes the pass safe to
    run off the cards without knowing which tests are precompilable. A test that
    drives the runtime by hand rather than through ``prepare`` /
    ``execute_distributed`` (``test_l3_manual``) would otherwise reach a real
    device here; instead it skips, warms nothing, and compiles normally in pass 2.
    """
    if not request.config.getoption("--precompile-only"):
        return

    import simpler.worker  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415
    from pypto.runtime.distributed_runner import DistributedWorker  # noqa: PLC0415

    def skip_execution(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        pytest.skip("--precompile-only stops before device execution")

    monkeypatch.setattr("pypto.runtime.runner.execute_compiled", skip_execution)
    monkeypatch.setattr("pypto.runtime.distributed_runner.execute_distributed", skip_execution)
    monkeypatch.setattr(DistributedCompiledProgram, "prepare", skip_execution)
    monkeypatch.setattr(DistributedWorker, "__init__", skip_execution)
    monkeypatch.setattr(simpler.worker.Worker, "__init__", skip_execution)
