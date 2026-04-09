# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the PipelineProfiler."""

import json
import os
import time

import pypto.pipeline_profiling as _pipeline_profiling_mod
import pytest
from pypto.pipeline_profiling import PipelineProfiler, StageRecord, get_active_profiler


class TestStageRecord:
    """StageRecord dataclass tests."""

    def test_duration(self):
        record = StageRecord(name="test", start=1.0, end=2.5)
        assert record.duration == pytest.approx(1.5)

    def test_to_dict(self):
        child = StageRecord(name="child", start=1.0, end=1.5)
        parent = StageRecord(name="parent", start=0.5, end=2.0, children=[child])
        d = parent.to_dict()
        assert d["name"] == "parent"
        assert d["seconds"] == pytest.approx(1.5)
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "child"


class TestPipelineProfiler:
    """PipelineProfiler core functionality."""

    def test_context_manager_sets_current(self):
        assert PipelineProfiler.current() is None
        with PipelineProfiler() as prof:
            assert PipelineProfiler.current() is prof
        assert PipelineProfiler.current() is None

    def test_nested_context(self):
        with PipelineProfiler() as outer:
            assert PipelineProfiler.current() is outer
            with PipelineProfiler() as inner:
                assert PipelineProfiler.current() is inner
            assert PipelineProfiler.current() is outer
        assert PipelineProfiler.current() is None

    def test_single_stage(self):
        with PipelineProfiler() as prof:
            with prof.stage("compile"):
                time.sleep(0.01)
        stages = prof.to_dict()["stages"]
        assert len(stages) == 1
        assert stages[0]["name"] == "compile"
        assert stages[0]["seconds"] > 0

    def test_nested_stages(self):
        with PipelineProfiler() as prof:
            with prof.stage("compile"):
                with prof.stage("passes"):
                    time.sleep(0.005)
                with prof.stage("codegen"):
                    time.sleep(0.005)

        d = prof.to_dict()
        assert len(d["stages"]) == 1
        compile_stage = d["stages"][0]
        assert compile_stage["name"] == "compile"
        assert len(compile_stage["children"]) == 2
        assert compile_stage["children"][0]["name"] == "passes"
        assert compile_stage["children"][1]["name"] == "codegen"

    def test_multiple_root_stages(self):
        with PipelineProfiler() as prof:
            with prof.stage("parse"):
                pass
            with prof.stage("passes"):
                pass
            with prof.stage("codegen"):
                pass

        stages = prof.to_dict()["stages"]
        assert len(stages) == 3
        assert [s["name"] for s in stages] == ["parse", "passes", "codegen"]

    def test_total_seconds(self):
        with PipelineProfiler() as prof:
            time.sleep(0.01)
        assert prof.total_seconds > 0

    def test_begin_end_stage(self):
        with PipelineProfiler() as prof:
            prof._begin_stage("pass_1")
            time.sleep(0.005)
            prof._end_stage()
            prof._begin_stage("pass_2")
            time.sleep(0.005)
            prof._end_stage()

        stages = prof.to_dict()["stages"]
        assert len(stages) == 2
        assert stages[0]["name"] == "pass_1"
        assert stages[1]["name"] == "pass_2"
        for s in stages:
            assert s["seconds"] > 0


class TestSummary:
    """Human-readable summary output."""

    def test_summary_contains_header(self):
        with PipelineProfiler() as prof:
            with prof.stage("passes"):
                pass
        text = prof.summary()
        assert "PyPTO Pipeline Profile" in text
        assert "Total:" in text
        assert "passes" in text

    def test_summary_shows_percentage(self):
        with PipelineProfiler() as prof:
            with prof.stage("passes"):
                time.sleep(0.01)
        text = prof.summary()
        assert "%" in text


class TestJsonOutput:
    """JSON serialisation."""

    def test_to_json_string(self):
        with PipelineProfiler() as prof:
            with prof.stage("test"):
                pass
        text = prof.to_json()
        data = json.loads(text)
        assert "total_seconds" in data
        assert "stages" in data

    def test_to_json_file(self, tmp_path):
        path = str(tmp_path / "profile.json")
        with PipelineProfiler() as prof:
            with prof.stage("test"):
                pass
        prof.to_json(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["stages"][0]["name"] == "test"


class TestWriteReport:
    """Report file output."""

    def test_write_report_creates_files(self, tmp_path):
        report_dir = str(tmp_path / "report")
        with PipelineProfiler() as prof:
            with prof.stage("passes"):
                pass
            with prof.stage("codegen"):
                pass
        prof.write_report(report_dir)

        txt_path = os.path.join(report_dir, "pipeline_profile.txt")
        json_path = os.path.join(report_dir, "pipeline_profile.json")
        assert os.path.exists(txt_path)
        assert os.path.exists(json_path)

        with open(txt_path) as f:
            assert "passes" in f.read()
        with open(json_path) as f:
            data = json.load(f)
            assert len(data["stages"]) == 2


class TestGetActiveProfiler:
    """get_active_profiler() helper."""

    def test_returns_none_by_default(self):
        assert get_active_profiler() is None

    def test_returns_current_profiler(self):
        with PipelineProfiler() as prof:
            assert get_active_profiler() is prof

    def test_env_var_creates_profiler(self, monkeypatch):
        monkeypatch.setenv("PYPTO_PIPELINE_PROFILING", "1")
        _pipeline_profiling_mod._env_profiler = None
        try:
            result = get_active_profiler()
            assert result is not None
            assert isinstance(result, PipelineProfiler)
        finally:
            PipelineProfiler._local.current = None
            _pipeline_profiling_mod._env_profiler = None
