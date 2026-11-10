# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Caller ownership of a persistent rank-0 direct controller and its launcher."""

import copy
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ._cluster_channel import accept_session, recv_record, send_record

_UNCLOSED_SESSIONS: set[Any] = set()


def _numeric_topology(topology: dict[str, Any]) -> dict[str, Any]:
    """Resolve launcher names once, without changing logical worker ordering."""
    result = copy.deepcopy(topology)
    result["controller_host"] = socket.gethostbyname(result["controller_host"])
    for spec in result["executor_ranks"]:
        spec["host"] = socket.gethostbyname(spec["host"])
    arguments = list(result.get("launcher_args", ()))
    if not any(arg.partition("=")[0] in ("-wdir", "--wdir", "-wd", "--wd") for arg in arguments):
        # MPI must not require the caller's PyPTO checkout on runtime-only
        # executors. Keep launcher cwd unchanged for relative launcher files.
        arguments.extend(("-wdir", "/"))
    result["launcher_args"] = arguments
    return result


def _executable_path(value: str) -> str:
    """Resolve relative executable paths before MPI changes the rank cwd."""
    return os.path.abspath(value) if os.path.dirname(value) and not os.path.isabs(value) else value


class DirectSession:
    """Keep rank 0 and all executor ranks alive across synchronous caller runs.

    The private local connection carries invocation metadata only. Generated
    L4 tasks and remote domain operations are dispatched by native direct MPI.
    """

    def __init__(self, topology: dict[str, Any], startup_by_rank: dict[int, Any], target: str, config: Any):
        from simpler.mpi_direct_bootstrap import encode_startup  # noqa: PLC0415
        from simpler.mpi_direct_topology import MpiDirectTopology  # noqa: PLC0415

        self._temporary = tempfile.TemporaryDirectory(prefix="pypto-direct-")
        self.directory = Path(self._temporary.name)
        os.chmod(self.directory, 0o700)
        self.data_dir = self.directory / "data"
        self.data_dir.mkdir(mode=0o700)
        self._timeout = float(topology["startup_timeout_s"])
        self._process: subprocess.Popen[Any] | None = None
        self._stream: socket.socket | None = None
        self._closed = False
        self._failed = False
        self._close_sent = False
        self._log = None
        self._topology = MpiDirectTopology.from_dict(_numeric_topology(topology))
        self._reports: dict[str, Any] = {}
        _UNCLOSED_SESSIONS.add(self)
        listener = socket.socket()
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            key = secrets.token_bytes(32)
            startup = copy.deepcopy(startup_by_rank)
            self.data_dir = Path(startup[0].get("parameters", {}).get("data_dir", self.data_dir))
            startup[0].setdefault("parameters", {}).update(
                address=list(listener.getsockname()), authkey=key.hex(), data_dir=str(self.data_dir)
            )
            topology_path = self.directory / "topology.json"
            topology_record = asdict(self._topology)
            topology_record["executor_ranks"] = topology_record.pop("executors")
            topology_path.write_text(json.dumps(topology_record))
            settings = {
                "topology_path": str(topology_path),
                "controller": target,
                "mpirun_path": _executable_path(config.mpirun_path),
                "python_executable": _executable_path(config.python_executable),
                "startup_by_rank": {str(rank): record for rank, record in startup.items()},
            }
            filename = self.directory / "startup.json"
            descriptor = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(encode_startup(settings))
            self._log = (self.directory / "launcher.log").open("wb")
            self._process = subprocess.Popen(
                [sys.executable, "-m", "pypto.runtime._cluster_direct_launch", str(filename)],
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._stream = accept_session(listener, key, self._timeout)
            self._stream.settimeout(self._timeout)
            ready = recv_record(self._stream)
            if ready != {
                "type": "ready",
                "transport": "direct",
                "executor_count": len(self._topology.executors),
            }:
                raise RuntimeError("Direct controller returned an invalid readiness contract")
            self._stream.settimeout(None)
        except BaseException as error:
            self._failed = True
            try:
                self.close()
            except BaseException as cleanup_error:
                # Preserve the process, private files and controller journal for
                # a caller's explicit retry; GC cannot erase uncertain ownership.
                error.direct_session = self
                if hasattr(error, "add_note"):
                    error.add_note(f"Direct session cleanup remains unconfirmed: {cleanup_error}")
            raise
        finally:
            listener.close()

    @property
    def reports(self) -> dict[str, Any]:
        """Return the immutable-in-practice snapshot of final per-rank reports."""
        return copy.deepcopy(self._reports)

    def request(self, record: dict[str, Any]) -> dict[str, Any]:
        """Send one invocation without admitting work after uncertain failure."""
        if self._closed or self._failed or self._stream is None:
            raise RuntimeError("Direct cluster session is closed or failed")
        try:
            send_record(self._stream, record)
            result = recv_record(self._stream)
            if result.get("type") == "error":
                self._failed = True
                raise RuntimeError(
                    f"Direct controller {result.get('error_type', 'error')}: {result.get('message', '')}"
                )
            if result != {"type": "done"}:
                raise RuntimeError("Direct controller returned an invalid completion contract")
            return result
        except BaseException:
            self._failed = True
            raise

    def _verify_reports(self) -> None:
        filename = self.directory / "cleanup.json"
        if not filename.is_file():
            raise RuntimeError(f"Direct cleanup reports are missing; retained journal: {self.directory}")
        reports = json.loads(filename.read_text())
        if set(reports) != {str(rank) for rank in range(self._topology.world_size)}:
            raise RuntimeError("Direct cleanup reports do not cover every controller/executor rank")
        first = reports["0"]
        for rank, report in reports.items():
            index = int(rank)
            expected_worker = None if index == 0 else self._topology.executor_for_rank(index).worker_id
            if (
                report.get("rank") != index
                or report.get("role") != ("L4" if index == 0 else "L3")
                or report.get("worker_id") != expected_worker
                or not report.get("job_id")
                or report.get("job_id") != first.get("job_id")
                or not report.get("session_id")
                or report.get("session_id") != first.get("session_id")
                or report.get("sequence") != 2
                or report.get("command") != ("not_observed" if index == 0 else "observed")
                or report.get("cleanup") != "succeeded"
                or report.get("failures")
            ):
                raise RuntimeError(f"Direct rank {rank} did not confirm successful resource release")
        self._reports = reports

    def close(self) -> None:
        """Confirm both per-rank resource cleanup and supervisor termination."""
        if self._closed:
            return
        if self._process is None:
            self._temporary.cleanup()
            self._closed = True
            _UNCLOSED_SESSIONS.discard(self)
            return
        if self._stream is not None and not self._failed and not self._close_sent:
            self._close_sent = True
            self._stream.settimeout(self._timeout)
            send_record(self._stream, {"op": "close"})
            if recv_record(self._stream) != {"type": "closed"}:
                self._failed = True
                raise RuntimeError("Direct controller did not confirm close")
        if self._failed and self._process.poll() is None:
            # The supervisor's termination handler owns the MPI process tree.
            self._process.terminate()
        try:
            result = self._process.wait(timeout=self._timeout)
        except subprocess.TimeoutExpired as error:
            self._failed = True
            raise RuntimeError(f"Direct launcher teardown remains unconfirmed: {self.directory}") from error
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._log is not None:
            self._log.close()
        self._verify_reports()
        self._closed = True
        self._temporary.cleanup()
        _UNCLOSED_SESSIONS.discard(self)
        if result != 0 and not self._failed:
            raise RuntimeError(f"Direct launcher exited {result} after confirmed resource cleanup")
