# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synchronization policy, cache isolation and persisted PTO replay contracts."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType, pto_backend
from pypto.backend._sync_strategy import read_sync_metadata, write_sync_metadata
from pypto.compile_profiling import StageRecord
from pypto.ir.compile import _run_pass_pipeline
from pypto.jit.cache import make_cache_key
from pypto.jit.decorator import _resolve_compile_request, _resolve_sync_strategy
from pypto.runtime import CompileOptions, ExecutionMode, RunConfig
from pypto.runtime.debug import pto_rebuild


@pl.program
class VectorCopy:
    @pl.function
    def main(
        self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            x = pl.load(a, [0, 0], [32, 32])
            y = pl.add(x, 1.0)
            pl.store(y, [0, 0], b)
        return b


@pytest.mark.parametrize("strategy", list(passes.AutoSyncStrategy.__members__.values()))
@pytest.mark.parametrize("planner", list(passes.MemoryPlanner.__members__.values()))
def test_flags_replace_event_and_preserve_level(monkeypatch, strategy, planner):
    handler = SimpleNamespace(get_extra_ptoas_flags=lambda: ["--pto-arch", "a5"])
    monkeypatch.setattr(pto_backend._backend_core, "get_handler", lambda: handler)
    flags = pto_backend._get_ptoas_flags(planner, sync_strategy=strategy)
    expected = "--enable-bufid_sync" if strategy == passes.AutoSyncStrategy.BUFID else "--enable-insert-sync"
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    assert flags == [expected, f"--pto-level={level}", "--pto-arch", "a5"]


def test_default_flags_remain_event(monkeypatch):
    handler = SimpleNamespace(get_extra_ptoas_flags=lambda: ["--pto-arch", "a3"])
    monkeypatch.setattr(pto_backend._backend_core, "get_handler", lambda: handler)
    assert pto_backend._get_ptoas_flags() == [
        "--enable-insert-sync",
        "--pto-level=level3",
        "--pto-arch",
        "a3",
    ]


@pytest.mark.parametrize(
    "extra", ["--enable-insert-sync", "--enable-unified-sync", "--enable-bufid_sync=false"]
)
def test_handler_cannot_add_competing_sync_flags(monkeypatch, extra):
    handler = SimpleNamespace(get_extra_ptoas_flags=lambda: ["--pto-arch", "a5", extra])
    monkeypatch.setattr(pto_backend._backend_core, "get_handler", lambda: handler)
    with pytest.raises(ValueError, match="mutually exclusive"):
        pto_backend._get_ptoas_flags(sync_strategy=passes.AutoSyncStrategy.BUFID)


@pytest.mark.parametrize("invalid", ["bufid", "unified", "typo", 1, None])
def test_invalid_strategy_rejected(monkeypatch, invalid):
    with pytest.raises(ValueError, match="sync_strategy must be AutoSyncStrategy"):
        pto_backend._get_ptoas_flags(sync_strategy=invalid)


def test_context_round_trip_and_cache_resolution():
    assert passes.PassContext([]).get_sync_strategy() == passes.AutoSyncStrategy.EVENT
    with passes.PassContext([], sync_strategy=passes.AutoSyncStrategy.BUFID):
        assert _resolve_sync_strategy() == passes.AutoSyncStrategy.BUFID
        with passes.PassContext([]):
            assert _resolve_sync_strategy() == passes.AutoSyncStrategy.EVENT
        assert _resolve_sync_strategy() == passes.AutoSyncStrategy.BUFID


def test_sync_option_preserves_existing_context_options():
    # Appending sync selection must preserve the mainline software-pipeline
    # and hardware-loop reporting arguments in the C++/Python constructor ABI.
    ctx = passes.PassContext(
        [],
        enable_buffer_ir=False,
        enable_software_pipeline=True,
        report_hardware_loops=True,
        sync_strategy=passes.AutoSyncStrategy.BUFID,
    )
    assert ctx.get_enable_software_pipeline()
    assert ctx.get_report_hardware_loops()
    assert ctx.get_sync_strategy() == passes.AutoSyncStrategy.BUFID


@pytest.mark.parametrize("options", [CompileOptions, RunConfig])
def test_compile_options_forward_only_explicit_strategy(options):
    default = options()
    kwargs = default.as_compile_kwargs() if isinstance(default, CompileOptions) else default.compile_kwargs()
    assert "sync_strategy" not in kwargs
    configured = options(sync_strategy=passes.AutoSyncStrategy.BUFID)
    kwargs = (
        configured.as_compile_kwargs()
        if isinstance(configured, CompileOptions)
        else configured.compile_kwargs()
    )
    assert kwargs["sync_strategy"] == passes.AutoSyncStrategy.BUFID


def test_jit_rejects_explicit_context_conflict_before_cache_lookup():
    with pytest.raises(RuntimeError, match="sync_strategy while a PassContext is already active"):
        _resolve_compile_request(RunConfig(sync_strategy=passes.AutoSyncStrategy.BUFID))


@pytest.mark.parametrize("via_config", [False, True])
def test_jit_cache_does_not_reuse_other_policy(monkeypatch, via_config):
    # An explicit artifact directory requests rebuilding instead of cache reuse.
    monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)

    @pl.jit
    def kernel(x: pl.Tensor[[32], pl.FP32]):
        with pl.at(level=pl.Level.CORE_GROUP):
            y = pl.add(x, 1.0)
        return y

    builds = []

    def compile_kernel(*args, **kwargs):
        result = object()
        builds.append(result)
        return result

    monkeypatch.setattr(kernel, "_compile", compile_kernel)
    config = RunConfig(backend_type=BackendType.Ascend950, dump_passes=False)
    with passes.PassContext([], sync_strategy=passes.AutoSyncStrategy.EVENT):
        event = kernel.compile(config=config)
        assert kernel.compile(config=config) is event
    if via_config:
        # The unit-test harness supplies a context; test the explicit field's
        # key resolution separately from that harness with a default-free scope.
        monkeypatch.setattr(passes.PassContext, "current", lambda: None)
        bufid = kernel.compile(
            config=RunConfig(
                backend_type=BackendType.Ascend950,
                sync_strategy=passes.AutoSyncStrategy.BUFID,
                dump_passes=False,
            )
        )
    else:
        with passes.PassContext([], sync_strategy=passes.AutoSyncStrategy.BUFID):
            bufid = kernel.compile(config=config)
            assert kernel.compile(config=config) is bufid
    assert bufid is not event
    assert len(builds) == 2


def test_policy_splits_cache_identity():
    args: dict[str, Any] = dict(
        source_hash="source", param_names=[], tensor_shapes={}, tensor_dtypes={}, dynamic_dims=set()
    )
    event = make_cache_key(**args)
    bufid = make_cache_key(**args, sync_strategy=passes.AutoSyncStrategy.BUFID)
    assert event != bufid
    assert dict(event[5] or ())["sync_strategy"] == "event"
    assert dict(bufid[5] or ())["sync_strategy"] == "bufid"


def test_bufid_rejected_on_a2a3(tmp_path):
    with passes.PassContext([], sync_strategy=passes.AutoSyncStrategy.BUFID):
        with pytest.raises(ValueError, match="BUFID requires Ascend950"):
            ir.compile(
                VectorCopy, backend_type=BackendType.Ascend910B, skip_ptoas=True, output_dir=str(tmp_path)
            )


@pytest.mark.parametrize("strategy", list(passes.AutoSyncStrategy.__members__.values()))
def test_explicit_compile_option_without_an_active_context(tmp_path, monkeypatch, strategy):
    # Hide the harness's implicit context to exercise the public keyword path.
    monkeypatch.setattr(passes.PassContext, "current", lambda: None)
    seen = []

    def run(pto_path, output_path, ptoas_flags):
        seen.append(ptoas_flags)
        Path(output_path).write_text("// generated")

    monkeypatch.setattr(pto_backend, "_run_ptoas", run)
    ir.compile(
        VectorCopy,
        backend_type=BackendType.Ascend950,
        output_dir=str(tmp_path),
        sync_strategy=strategy,
        dump_passes=False,
    )
    expected = "--enable-bufid_sync" if strategy == passes.AutoSyncStrategy.BUFID else "--enable-insert-sync"
    assert seen and all(flags[0] == expected for flags in seen)


@pytest.mark.parametrize("dump,profiling", [(False, False), (True, False), (False, True), (True, True)])
def test_compile_carries_policy_through_nested_contexts_and_replay(tmp_path, monkeypatch, dump, profiling):
    commands = []
    observed = []

    def observe(_pass, _program):
        ctx = passes.PassContext.current()
        assert ctx is not None
        observed.append(ctx.get_sync_strategy())

    def run_ptoas(pto_path, cpp_path, ptoas_flags):
        commands.append(ptoas_flags)
        Path(cpp_path).write_text("// generated kernel")

    monkeypatch.setattr(pto_backend, "_run_ptoas", run_ptoas)
    instrument = passes.CallbackInstrument(before_pass=observe)
    with passes.PassContext([instrument], sync_strategy=passes.AutoSyncStrategy.BUFID):
        ir.compile(
            VectorCopy,
            backend_type=BackendType.Ascend950,
            output_dir=str(tmp_path),
            dump_passes=dump,
            profiling=profiling,
        )
    assert observed and set(observed) == {passes.AutoSyncStrategy.BUFID}
    assert commands and all(flags[0] == "--enable-bufid_sync" for flags in commands)
    for pto_path in (tmp_path / "ptoas").glob("*.pto"):
        flags = read_sync_metadata(pto_path, ["--enable-insert-sync"])
        assert flags == commands[0]
        assert json.loads(pto_path.with_suffix(".json").read_text())["sync_strategy"] == "bufid"


def test_direct_backend_codegen_inherits_context_policy(tmp_path, monkeypatch):
    seen = []

    def run(pto_path, output_path, ptoas_flags):
        seen.append(ptoas_flags)
        Path(output_path).write_text("// generated")

    monkeypatch.setattr(pto_backend, "_run_ptoas", run)
    with passes.PassContext([], sync_strategy=passes.AutoSyncStrategy.BUFID):
        pipeline = _run_pass_pipeline(VectorCopy, operation="compile", backend_type=BackendType.Ascend950)
        pto_backend.generate(pipeline.transformed_program, str(tmp_path))
    assert seen and all(flags[0] == "--enable-bufid_sync" for flags in seen)


@pytest.mark.parametrize("strategy", list(passes.AutoSyncStrategy.__members__.values()))
def test_a5_policy_with_real_ptoas(tmp_path, strategy):
    """Exercise DSL lowering, PTO emission and the real compiler when available."""
    if pto_backend._find_ptoas_binary() is None:
        pytest.skip("PTOAS is not installed")
    with passes.PassContext([], sync_strategy=strategy):
        ir.compile(
            VectorCopy,
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            output_dir=str(tmp_path),
            dump_passes=False,
        )
    pto_paths = list((tmp_path / "ptoas").glob("*.pto"))
    assert pto_paths
    for pto_path in pto_paths:
        assert pto_path.with_suffix(".cpp").stat().st_size > 0
        metadata = json.loads(pto_path.with_suffix(".json").read_text())
        assert metadata["sync_strategy"] == strategy.name.lower()


@pytest.mark.parametrize("grouped", [False, True])
def test_grouped_and_single_unit_persist_the_same_policy(tmp_path, monkeypatch, grouped):
    handler = SimpleNamespace(get_extra_ptoas_flags=lambda: ["--pto-arch", "a5"])
    monkeypatch.setattr(pto_backend._backend_core, "get_handler", lambda: handler)
    monkeypatch.setattr(pto_backend, "_generate_kernel_wrapper", lambda *args, **kwargs: "wrapped")
    monkeypatch.setattr(pto_backend, "_get_kernel_output_path", lambda f, _: f"kernels/{f.name}.cpp")
    monkeypatch.setattr(pto_backend, "_uses_spmd_block_ops", lambda _: False)
    seen = []

    def run(pto_path, output_path, ptoas_flags):
        seen.append(ptoas_flags)
        Path(output_path).write_text("generated")

    monkeypatch.setattr(pto_backend, "_run_ptoas", run)
    files = {}
    funcs: Any = [SimpleNamespace(name="first"), SimpleNamespace(name="second")]
    if grouped:
        pto_backend._emit_group_output(
            files,
            "group",
            funcs,
            "module {}",
            str(tmp_path),
            False,
            sync_strategy=passes.AutoSyncStrategy.BUFID,
        )
        unit = "group"
    else:
        pto_backend._emit_single_function_output(
            files,
            funcs[0],
            "module {}",
            str(tmp_path),
            False,
            sync_strategy=passes.AutoSyncStrategy.BUFID,
        )
        unit = "first"
    assert seen == [["--enable-bufid_sync", "--pto-level=level3", "--pto-arch", "a5"]]
    assert read_sync_metadata(tmp_path / "ptoas" / f"{unit}.pto", []) == seen[0]


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_ptoas_workers_receive_policy_without_context(tmp_path, monkeypatch, workers):
    seen = []
    monkeypatch.setattr(pto_backend, "_get_max_workers", lambda: workers)

    def emit(unit, output_dir, skip, memory_planner, dump_ptoas_passes, sync_strategy):
        seen.append(sync_strategy)
        return pto_backend._EmitResult(
            name=unit.name, files={}, ptoas_record=StageRecord(name="ptoas", start=0.0, end=1.0)
        )

    monkeypatch.setattr(pto_backend, "_emit_unit", emit)
    units: Any = [
        SimpleNamespace(name=name, stage_record=StageRecord(name=name, start=0.0)) for name in ["a", "b"]
    ]
    pto_backend._run_ptoas_phase(
        units, str(tmp_path), False, None, {}, [], sync_strategy=passes.AutoSyncStrategy.BUFID
    )
    assert seen == [passes.AutoSyncStrategy.BUFID, passes.AutoSyncStrategy.BUFID]


@pytest.mark.parametrize("layout", ["", "next_levels/rank0"])
def test_replay_runs_with_persisted_flags(tmp_path, monkeypatch, layout):
    pto_path = tmp_path / layout / "ptoas" / "unit.pto"
    pto_path.parent.mkdir(parents=True)
    pto_path.write_text("module {}")
    flags = ["--enable-bufid_sync", "--pto-level=level3", "--pto-arch", "a5"]
    write_sync_metadata(pto_path, passes.AutoSyncStrategy.BUFID, flags)
    captured = []
    monkeypatch.setattr(pto_rebuild, "_check_ptoas_version", lambda _: None)

    def run(cmd, **kwargs):
        captured.append(cmd)
        assert kwargs["timeout"] == 60
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pto_rebuild.subprocess, "run", run)
    pto_rebuild._run_ptoas("ptoas", pto_path, pto_path.with_suffix(".cpp"))
    assert captured == [["ptoas", str(pto_path), "-o", str(pto_path.with_suffix(".cpp")), *flags]]


@pytest.mark.parametrize("corrupt", [False, True])
def test_metadata_is_validated_and_invalidates_replay_stamp(tmp_path, monkeypatch, corrupt):
    pto = tmp_path / "ptoas" / "unit.pto"
    pto.parent.mkdir()
    pto.write_text("module {}")
    cpp = pto.with_suffix(".cpp")
    cpp.write_text("old cpp")
    flags = ["--enable-bufid_sync", "--pto-level=level2", "--pto-arch", "a5"]
    write_sync_metadata(pto, passes.AutoSyncStrategy.BUFID, flags)
    if corrupt:
        pto.with_suffix(".json").write_text('{"sync_strategy":"invalid"}')
    os.utime(pto, (1, 1))
    os.utime(cpp, (2, 2))
    os.utime(pto.with_suffix(".json"), (3, 3))
    calls = []

    def run(binary, pto_path, out_cpp):
        calls.append(pto_path)
        out_cpp.write_text("// new body")

    monkeypatch.setattr(pto_rebuild, "_run_ptoas", run)
    if corrupt:
        with pytest.raises(ValueError, match="Invalid PTOAS compile metadata"):
            pto_rebuild._rebuild_one_dir(tmp_path, tmp_path, "ptoas")
        assert not calls
    else:
        pto_rebuild._rebuild_one_dir(tmp_path, tmp_path, "ptoas")
        assert calls == [pto]


@pytest.mark.parametrize("metadata", [None, {}, {"sync_strategy": "event"}])
def test_legacy_metadata_defaults_to_event(tmp_path, metadata):
    path = tmp_path / "unit.pto"
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata))
    legacy = ["--enable-insert-sync", "--pto-level=level2"]
    assert read_sync_metadata(path, legacy) == legacy


@pytest.mark.parametrize(
    "metadata",
    [
        "{",
        "null",
        "[]",
        '{"sync_strategy": null}',
        '{"sync_strategy": "unified"}',
        '{"sync_strategy": "bufid"}',
        '{"ptoas_flags": "bad"}',
        '{"sync_strategy": "event", "ptoas_flags": ["--enable-bufid_sync"]}',
    ],
)
def test_corrupt_metadata_never_silently_uses_event(tmp_path, metadata):
    path = tmp_path / "unit.pto"
    path.with_suffix(".json").write_text(metadata)
    with pytest.raises(ValueError, match="Invalid PTOAS compile metadata"):
        read_sync_metadata(path, ["--enable-insert-sync", "--pto-level=level2"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
