# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Caller-side direct session staging and bounded control protocol."""

import json
from types import SimpleNamespace

import pytest
from pypto.runtime import _cluster_direct_launch as launcher
from pypto.runtime import _cluster_direct_session as session_module
from pypto.runtime._cluster_direct_session import DirectSession


def test_direct_launcher_preserves_relative_launcher_arguments(tmp_path, monkeypatch):
    from pathlib import Path  # noqa: PLC0415

    import simpler.mpi_direct_supervisor as supervisor  # noqa: PLC0415
    from simpler.mpi_direct_bootstrap import encode_startup  # noqa: PLC0415

    monkeypatch.chdir(tmp_path)
    settings = {
        "topology_path": str(tmp_path / "topology.json"),
        "controller": "portable_package.controller:main",
        "mpirun_path": "/usr/local/bin/mpirun",
        "python_executable": "/runtime-only-python",
        "startup_by_rank": {"0": {}, "1": {}},
    }
    startup = tmp_path / "startup.json"
    startup.write_bytes(encode_startup(settings))
    seen = []

    def run(topology, controller, **kwargs):
        seen.append((Path.cwd(), topology, controller, kwargs["startup_by_rank"]))
        return 0

    monkeypatch.setattr(supervisor, "run_supervisor", run)
    assert launcher.main(str(startup)) == 0
    assert seen == [(tmp_path, settings["topology_path"], settings["controller"], {0: {}, 1: {}})]
    assert json.loads((tmp_path / "cleanup.json").read_text()) == {}


@pytest.mark.parametrize("arguments", [[], ["--hostfile", "relative-hosts"]])
def test_direct_rank_working_directory_does_not_require_caller_checkout(arguments, monkeypatch):
    monkeypatch.setattr(session_module.socket, "gethostbyname", lambda host: "127.0.0.1")
    topology = {"controller_host": "node", "executor_ranks": [{"host": "node"}], "launcher_args": arguments}
    result = session_module._numeric_topology(topology)
    assert result["launcher_args"] == arguments + ["-wdir", "/"]
    assert topology["launcher_args"] == arguments


def test_direct_rank_working_directory_preserves_explicit_override(monkeypatch):
    monkeypatch.setattr(session_module.socket, "gethostbyname", lambda host: "127.0.0.1")
    topology = {
        "controller_host": "node",
        "executor_ranks": [{"host": "node"}],
        "launcher_args": ["-wdir", "/shared"],
    }
    assert session_module._numeric_topology(topology)["launcher_args"] == ["-wdir", "/shared"]


def test_direct_relative_executables_resolve_before_rank_directory_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert session_module._executable_path("./runtime-python") == str(tmp_path / "runtime-python")
    assert session_module._executable_path("./bin/mpirun") == str(tmp_path / "bin/mpirun")
    assert session_module._executable_path("python3") == "python3"


@pytest.fixture
def cleanup_session(tmp_path):
    session = object.__new__(DirectSession)
    session.directory = tmp_path
    session._reports = {}
    session._topology = SimpleNamespace(
        world_size=3, executor_for_rank=lambda rank: SimpleNamespace(worker_id=rank - 1)
    )
    reports = {
        str(rank): {
            "rank": rank,
            "worker_id": None if rank == 0 else rank - 1,
            "role": "L4" if rank == 0 else "L3",
            "sequence": 2,
            "command": "not_observed" if rank == 0 else "observed",
            "cleanup": "succeeded",
            "failures": [],
            "job_id": "test-job",
            "session_id": 42,
        }
        for rank in range(3)
    }
    return session, reports


def test_direct_cleanup_requires_final_reports_from_every_rank(cleanup_session):
    session, reports = cleanup_session
    filename = session.directory / "cleanup.json"
    with pytest.raises(RuntimeError, match="missing"):
        session._verify_reports()
    missing = dict(reports)
    del missing["2"]
    filename.write_text(json.dumps(missing))
    with pytest.raises(RuntimeError, match="every"):
        session._verify_reports()
    filename.write_text(json.dumps(reports))
    session._verify_reports()
    assert session.reports == reports
    session.reports["0"]["cleanup"] = "corrupted"
    assert session.reports == reports


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence", 1),
        ("command", "not_observed"),
        ("cleanup", "unknown"),
        ("failures", ["L3 drain failed"]),
        ("job_id", "different-job"),
        ("session_id", 43),
        ("worker_id", 99),
        ("role", "L4"),
    ],
)
def test_direct_cleanup_rejects_uncertain_or_unrelated_reports(cleanup_session, field, value):
    session, reports = cleanup_session
    reports["2"][field] = value
    (session.directory / "cleanup.json").write_text(json.dumps(reports))
    with pytest.raises(RuntimeError, match="rank 2"):
        session._verify_reports()
    assert session.reports == {}


def test_direct_failed_run_releases_files_after_confirmed_rank_drain(cleanup_session):
    session, reports = cleanup_session
    (session.directory / "cleanup.json").write_text(json.dumps(reports))
    cleaned = []
    session._temporary = SimpleNamespace(cleanup=lambda: cleaned.append(True))
    session._process = SimpleNamespace(poll=lambda: 1, wait=lambda timeout: 1)
    session._stream = None
    session._failed = True
    session._closed = False
    session._close_sent = False
    session._log = None
    session._timeout = 1
    # The request already raised the workload error. Successful rank cleanup
    # plus confirmed process exit permits the caller's storage journal to drain.
    session.close()
    assert session._closed and cleaned == [True]
    session.close()
    assert cleaned == [True]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
