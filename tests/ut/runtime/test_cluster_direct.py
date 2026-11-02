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
import socket
import struct
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from pypto.runtime import _cluster_channel as channel
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


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan"), -0.0])
def test_direct_staging_preserves_float_scalar_bits(tmp_path, value):
    with channel.PackedValues([value], [], tmp_path) as packed:
        # JSON is strict: special IEEE values must have an explicit wire encoding.
        request = json.loads(json.dumps(packed.request(), allow_nan=False))
        with channel.map_values(request, tmp_path) as values:
            assert struct.pack("!d", values[0]) == struct.pack("!d", value)


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


def test_direct_staging_preserves_storage_aliases_offsets_and_write_ranges(tmp_path):
    backing = torch.arange(16, dtype=torch.float32)
    first, output = backing[4:12], backing[8:16]
    with channel.PackedValues([first, output, 7, True], [1], tmp_path) as packed:
        request = packed.request()
        assert len(request["storages"]) == 1
        with channel.map_values(request, tmp_path) as values:
            assert values[0].untyped_storage().data_ptr() == values[1].untyped_storage().data_ptr()
            assert values[0].storage_offset() == 4
            assert values[1].storage_offset() == 8
            assert values[2:] == [7, True]
            assert torch.equal(values[0], first)
            values[0][:4].fill_(-999)
            values[1].add_(100)
        assert torch.equal(backing, torch.arange(16, dtype=torch.float32))
        packed.publish()
        assert torch.equal(backing[:8], torch.arange(8, dtype=torch.float32))
        assert torch.equal(backing[8:], torch.arange(8, 16, dtype=torch.float32) + 100)
    assert not list(tmp_path.glob("storage-*"))


def test_direct_failed_invocation_does_not_publish_partial_outputs(tmp_path):
    output = torch.zeros(8)
    with channel.PackedValues([output], [0], tmp_path) as packed:
        with channel.map_values(packed.request(), tmp_path) as values:
            values[0].fill_(123)
        # No success acknowledgement: the caller must not invoke publish().
    assert torch.equal(output, torch.zeros(8))


def test_direct_staging_rejects_noncontiguous_views_before_launch(tmp_path):
    with pytest.raises(ValueError, match="contiguous CPU"):
        channel.PackedValues([torch.zeros(4, 4).t()], [0], tmp_path)
    assert not list(tmp_path.glob("storage-*"))


def test_direct_mapping_rejects_escaped_paths_and_out_of_bounds_views(tmp_path):
    with channel.PackedValues([torch.ones(4)], [0], tmp_path) as packed:
        request = packed.request()
        request["storages"][0]["name"] = "../outside"
        with pytest.raises(ValueError, match="storage"):
            with channel.map_values(request, tmp_path):
                pass
        request = packed.request()
        request["values"][0]["offset"] = 10000
        with pytest.raises(ValueError, match="view"):
            with channel.map_values(request, tmp_path):
                pass


def test_direct_record_protocol_is_bounded_and_preserves_plain_values():
    left, right = socket.socketpair()
    try:
        channel.send_record(left, {"op": "run", "program": 2, "values": [1, 2.5, True, None]})
        assert channel.recv_record(right) == {"op": "run", "program": 2, "values": [1, 2.5, True, None]}
        left.sendall(struct.pack("!I", channel.MAX_RECORD_BYTES + 1))
        with pytest.raises(ValueError, match="limit"):
            channel.recv_record(right)
    finally:
        left.close()
        right.close()


def test_direct_controller_channel_rejects_wrong_session_authentication():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    key = bytes(range(32))
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            server = pool.submit(channel.accept_session, listener, key, 2.0)
            with pytest.raises(ValueError, match="auth"):
                channel.connect_session(listener.getsockname(), b"x" * 32, 2.0)
            with pytest.raises(ValueError, match="auth"):
                server.result(timeout=3)
    finally:
        listener.close()


def test_direct_authenticated_channel_does_not_apply_startup_timeout_to_runs():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            server = pool.submit(channel.accept_session, listener, b"key", 2.0)
            client = channel.connect_session(listener.getsockname(), b"key", 2.0)
            accepted = server.result(timeout=3)
            try:
                assert client.gettimeout() is None
                assert accepted.gettimeout() is None
            finally:
                client.close()
                accepted.close()
    finally:
        listener.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
