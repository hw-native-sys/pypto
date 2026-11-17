# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU checks for independent-sequence reset and opt-in diagnostic snapshots."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    cls = importlib.import_module("examples.models.qwen3_tp.runner").Runner
    obj = cls.__new__(cls)
    obj.trace_layers = True
    obj.trace = {}
    return obj


def test_snapshot_owns_last_valid_row(runner):
    value = torch.arange(12).reshape(3, 4)
    runner.snapshot(2, "output", value, 1)
    value.zero_()
    torch.testing.assert_close(runner.trace["2.output"], torch.tensor([4, 5, 6, 7]))


def test_disabled_snapshot_does_not_read_tensor(runner):
    runner.trace_layers = False
    runner.snapshot(0, "output", torch.empty(0), 9)
    assert runner.trace == {}


def test_reset_clears_sequence_but_retains_programs(runner):
    runner.k_cache = torch.ones(4, 8)
    runner.v_cache = torch.ones(4, 8)
    runner.hidden = torch.ones(2, 8)
    runner.position = 4
    runner.trace = {"old": torch.ones(1)}
    runner.programs = {"compiled": object()}
    programs = runner.programs
    runner.reset()
    assert runner.position == 0 and runner.trace == {}
    assert runner.programs is programs
    for value in (runner.k_cache, runner.v_cache, runner.hidden):
        assert torch.count_nonzero(value) == 0


def test_benchmark_uses_slowest_rank_and_separates_phases(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    summarize = importlib.import_module("examples.models.qwen3_tp.benchmark").summarize
    rows = [
        {"phase": "prefill", "rank_ms": [10, 20], "rank_collective_ms": [2, 4]},
        {"phase": "decode", "rank_ms": [3, 4], "rank_collective_ms": [1, 2]},
        {"phase": "decode", "rank_ms": [8, 6], "rank_collective_ms": [2, 3]},
    ]
    result = summarize(rows)
    assert result["prefill"]["median_ms"] == 20
    assert result["decode"]["mean_ms"] == 6
    assert result["decode"]["median_ms"] == 6
    assert result["decode"]["tokens_per_second"] == pytest.approx(1000 / 6)
    assert result["decode"]["mean_rank_collective_ms"] == 2


def test_profile_excludes_warmup_and_preserves_nested_span_meaning(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    analyze = importlib.import_module("examples.models.qwen3_tp.profile").analyze
    path = tmp_path / "trace.log"
    path.write_text(
        "\n".join(
            f"[mono_ns={stamp}][TIMING] [STRACE] v=1 pid=7 tid=7 name={name} ts=0 dur={duration}"
            for stamp, name, duration in (
                (50, "chip.run", 9000000),
                (120, "chip.run", 3000000),
                (125, "chip.run.runner_run.device_wall", 1000000),
                (210, "chip.run", 9000000),
            )
        )
    )
    data: dict[str, Any] = {
        "rows": [
            {
                "phase": "decode",
                "rank_ms": [10],
                "rank_collective_ms": [0],
                "rank_profiles": [
                    {"pid": 7, "start_ns": 100, "end_ns": 200, "dispatch": {"kernel": {"calls": 1, "ms": 5}}}
                ],
            }
        ]
    }
    result = analyze(data, path)["decode"]
    assert result["outside_dispatch_ms"] == 5
    assert result["python_dispatch_overhead_ms"] == 2
    assert result["native_outside_device_window_ms"] == 2
    assert result["native"]["chip.run"]["calls"] == 1
    data["rows"][0]["rank_profiles"][0]["dispatch"]["kernel"]["calls"] = 2
    with pytest.raises(ValueError, match="count"):
        analyze(data, path)


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_dispatch_plan_reuses_and_invalidates_by_identity(runner, monkeypatch, enabled):
    module = importlib.import_module("examples.models.qwen3_tp.runner")
    built, calls = [], []

    def make_plan(compiled):
        plan = SimpleNamespace(compiled=compiled)
        built.append(plan)
        return plan

    monkeypatch.setattr(module, "DispatchPlan", make_plan)
    monkeypatch.setattr(module, "run_kernel", lambda *a, **kw: calls.append((a, kw)))
    runner.cache_dispatch_plan = enabled
    runner.dispatch_plans = {}
    runner.chip = object()
    original = object()
    runner.programs = {"kernel": original}
    runner.dispatch("kernel", 11)
    runner.dispatch("kernel", 22)
    assert calls[0][0][-1] == 11 and calls[1][0][-1] == 22
    assert len(built) == int(enabled)
    runner.programs["kernel"] = object()
    runner.dispatch("kernel", 33)
    assert len(built) == 2 * int(enabled)
    if enabled:
        assert calls[0][1]["plan"] is calls[1][1]["plan"]
        assert calls[2][1]["plan"] is not calls[1][1]["plan"]
    else:
        assert all(not kwargs for _, kwargs in calls)


def test_setup_profile_keeps_residual_and_collective_overlap(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    analyze = importlib.import_module("examples.models.qwen3_tp.profile").analyze_setup
    data: dict[str, Any] = {
        "profile_setup": True,
        "rows": [
            {
                "phase": "decode",
                "rank_ms": [10],
                "rank_collective_ms": [4],
                "rank_profiles": [
                    {
                        "dispatch": {"kernel": {"calls": 2, "ms": 8}},
                        "setup_ms": {"native_dispatch": 5, "buffer_register": 1},
                    }
                ],
            }
        ],
    }
    row = analyze(data)["decode"]
    assert row["unattributed_dispatch_ms"] == 2
    assert row["outside_dispatch_ms"] == 2
    assert row["collective_ms"] == 4
    assert row["dispatch_calls"] == 2
    data["rows"][0]["rank_profiles"][0]["setup_ms"]["native_dispatch"] = 20
    with pytest.raises(ValueError, match="exceed"):
        analyze(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
