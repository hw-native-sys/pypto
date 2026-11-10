# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Prepared CLUSTER_0 execution exclusively through static direct MPI."""

import json
import math
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch

from pypto.ir.compiled_program import _coerce_args
from pypto.jit._artifact_manifest import BuildKind

from . import _cluster_controller
from ._artifact_sources import build_cluster_session_bundle
from ._cluster_channel import PackedValues
from ._cluster_direct_session import DirectSession as _DirectSession
from ._prebuilt import chip_directories
from .distributed_runner import _make_call_config

_UNCONFIRMED_WORKERS: set[Any] = set()
_InvocationIO = _cluster_controller._InvocationIO


def Worker(**kwargs: Any) -> Any:
    """Load the optional runtime only when preparing an execution session."""
    from simpler.worker import Worker as RuntimeWorker  # noqa: PLC0415

    return RuntimeWorker(**kwargs)


def _assemble_cluster_chips(compiled: Any) -> dict[str, tuple[Any, str, dict[str, Any]]]:
    """Compile the recursively declared CHIP closure exclusively on the parent."""
    artifact_runtime = vars(compiled).get("_artifact_runtime")
    if artifact_runtime is not None:
        return artifact_runtime.load()
    from .device_runner import _compile_and_assemble  # noqa: PLC0415

    return {
        path: _compile_and_assemble(directory, compiled.platform)
        for path, directory in chip_directories(compiled.output_dir, BuildKind.DISTRIBUTED).items()
    }


def _entry_bindings(program, chips, staging):
    chip_bindings = {
        path: (staging.register(chip).digest.hex(), tuple(config.get("host_tensor_indices", ())))
        for path, (chip, _, config) in chips.items()
    }
    return {
        entry["logical_path"]: {
            "module": entry["module"],
            "world_size": entry["local_device_count"],
            "chips": {name: chip_bindings[path] for name, path in entry["chip_paths"].items()},
            "sub_workers": {
                name: path.removesuffix(".py").replace("/", ".")
                for name, path in entry["sub_worker_paths"].items()
            },
            "domain_bindings": [],
        }
        for entry in program.l3_entries["entries"]
    }


def _inner_python_registrations(prefix, programs, bindings):
    from simpler.callable_identity import (  # noqa: PLC0415
        build_python_import_descriptor,
        compute_callable_hashid,
    )

    imports = {
        (f"{prefix}.programs.p{index}.{module}", name)
        for index, entries in enumerate(bindings)
        for binding in entries.values()
        for name, module in binding["sub_workers"].items()
    }
    if any(entry["return_copy_callable"] for program in programs for entry in program.l3_entries["entries"]):
        imports.add((f"{prefix}.orchestration._distributed_support", "_copy_return_tensor"))
    return [
        {
            "hashid": compute_callable_hashid(build_python_import_descriptor(module, name)),
            "kind": "PYTHON_IMPORT",
            "target_registry": "INNER_L3_WORKER",
            "target": f"{module}:{name}",
        }
        for module, name in sorted(imports)
    ]


def _call_config_record(config: Any) -> dict[str, Any]:
    """Retain task diagnostics and every ring override across the local bridge."""
    fields = (
        "aicpu_thread_num",
        "enable_chip_swimlane",
        "enable_dump_args",
        "enable_pmu",
        "enable_dep_gen",
        "enable_scope_stats",
        "output_prefix",
    )
    result = {field: getattr(config, field) for field in fields}
    result["runtime_env"] = {
        field: list(getattr(config.runtime_env, field))
        for field in ("ring_task_window", "ring_heap", "ring_dep_pool")
    }
    return result


def _validate_programs(programs: list[Any], compiled: Any, cc: Any) -> None:
    for program in programs:
        if program.entry_kind != "cluster":
            raise ValueError("ClusterWorker requires CLUSTER_0 compiled programs")
        program._require_artifact_ready()
        if program.cluster_config != cc or program.platform != compiled.platform:
            raise ValueError("Cluster programs require compatible topology/platform/runtime/profile config")
        for record in program.l3_entries["entries"]:
            required = program.output_dir / record["logical_path"] / "sub_workers/__required__.json"
            if required.exists():
                if json.loads(required.read_text(encoding="utf-8")):
                    raise NotImplementedError("Cluster runtime-bound SUB callbacks are unsupported")


class ClusterWorker:
    """Caller-side prepared session whose L4 worker lives in MPI rank zero."""

    def __init__(
        self,
        programs: list[Any],
        config: Any = None,
        *,
        startup_timeout_s: float | None = None,
    ) -> None:
        from simpler.mpi_direct_topology import MpiDirectExecutorSpec  # noqa: PLC0415
        from simpler.startup_artifacts import StartupArtifactSpec, read_startup_artifacts  # noqa: PLC0415

        programs = list({id(program): program for program in programs}.values())
        if not programs:
            raise ValueError("ClusterWorker requires at least one compiled program")
        compiled = programs[0]
        cc = compiled.cluster_config
        _validate_programs(programs, compiled, cc)
        timeout = cc.startup_timeout_s if startup_timeout_s is None else startup_timeout_s
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Cluster startup_timeout_s must be positive and finite")
        chip_sets = [_assemble_cluster_chips(program) for program in programs]
        if any(runtime != cc.runtime for chips in chip_sets for _, runtime, _ in chips.values()):
            raise ValueError("Every cluster CHIP dependency must use ClusterConfig.runtime")
        self._compiled = compiled
        self._config = config
        self._states = {id(program): (index, program) for index, program in enumerate(programs)}
        self._lock = threading.RLock()
        self._closed = False
        self._closing = False
        self._session = None
        self._active_io: PackedValues | None = None
        self._temporary = tempfile.TemporaryDirectory(prefix="pypto-cluster-caller-")
        self._data_dir = Path(self._temporary.name) / "data"
        self._data_dir.mkdir(mode=0o700)
        staging = None
        try:
            enable_sdma = any(
                bool(value[2].get("enable_sdma", False)) for chips in chip_sets for value in chips.values()
            )
            staging = Worker(
                level=4,
                platform=compiled.platform,
                runtime=cc.runtime,
                startup_timeout_s=timeout,
                enable_sdma=enable_sdma,
            )
            bindings = [
                _entry_bindings(program, chips, staging)
                for program, chips in zip(programs, chip_sets, strict=True)
            ]
            digest, payload = build_cluster_session_bundle(programs, bindings)
            prefix = "_pypto_cluster_" + digest
            archive = Path(self._temporary.name) / "sources.tar.gz"
            archive.write_bytes(payload)
            artifacts = read_startup_artifacts([StartupArtifactSpec(archive, digest, prefix).to_manifest()])
            execution = []
            for index, program in enumerate(programs):
                program_prefix = prefix + f".programs.p{index}"
                infos, _, _ = program._get_metadata()
                execution.append(
                    {
                        "entry_name": program.entry_name,
                        "module_prefix": program_prefix,
                        "entries": program.l3_entries["entries"],
                        "param_names": [info.name for info in infos],
                    }
                )
            minimum = sum(
                record["minimum_sub_workers"] for program in execution for record in program["entries"]
            )
            executors = [
                {
                    "rank": index + 1,
                    "worker_id": index,
                    "host": node.host,
                    "platform": compiled.platform,
                    "runtime": cc.runtime,
                    "device_ids": list(node.devices),
                    "num_sub_workers": max(cc.num_sub_workers, minimum),
                    "comm_profile": cc.comm_profile,
                    "global_device_ranks": list(cc.global_device_ranks()[index]),
                    "enable_sdma": enable_sdma,
                }
                for index, node in enumerate(cc.nodes)
            ]
            topology = {
                "controller_rank": 0,
                "controller_host": cc.nodes[0].host,
                "executor_ranks": executors,
                "startup_timeout_s": timeout,
                "session_timeout_s": timeout,
                "heartbeat_interval_s": min(1.0, timeout / 4),
                "max_pending_frame_bytes": 128 * 1024 * 1024,
                "launcher_args": list(cc.mpirun_args),
            }
            python_registrations = _inner_python_registrations(prefix, programs, bindings)
            session = {
                "platform": compiled.platform,
                "runtime": cc.runtime,
                "startup_timeout_s": timeout,
                "enable_sdma": enable_sdma,
                "node_sizes": cc.node_device_counts(),
                "programs": execution,
            }
            records = {}
            for spec in executors:
                typed = MpiDirectExecutorSpec.from_dict(spec)
                records[spec["rank"]] = {
                    "startup_artifacts": artifacts,
                    "inner_l3_worker": staging._inner_registry_entries_for_spec(typed) + python_registrations,
                }
            records[0] = {
                "startup_artifacts": artifacts,
                "inner_l3_worker": [
                    entry for entry in records[1]["inner_l3_worker"] if entry["kind"] == "CHIP_CALLABLE"
                ],
                "parameters": {"session": session, "data_dir": str(self._data_dir)},
            }
            self._session = _DirectSession(
                topology, records, prefix + ".orchestration._cluster_controller:main", cc
            )
        except BaseException as error:
            if self._session is None:
                self._session = getattr(error, "direct_session", None)
            try:
                self.close()
            except BaseException:
                _UNCONFIRMED_WORKERS.add(self)
            raise
        finally:
            if staging is not None:
                staging.close()

    def run(self, compiled: Any, *args: Any, config: Any = None) -> Any:
        """Publish writable caller storage only after a successful direct run."""
        item = self._states.get(id(compiled))
        if item is None or item[1] is not compiled:
            raise ValueError("ClusterWorker.run requires its prepared compiled program")
        infos, outputs, returns = compiled._get_metadata()
        values, return_style = _coerce_args(args, infos, outputs, returns, caller_name="ClusterWorker")
        for info, value in zip(infos, values, strict=True):
            if info.shape is None:
                continue
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not value.is_contiguous():
                raise ValueError(f"Cluster parameter {info.name!r} requires a contiguous CPU tensor")
            expected_dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(
                str(info.dtype), str(info.dtype)
            )
            if str(value.dtype).removeprefix("torch.") != expected_dtype:
                raise ValueError(f"Cluster parameter {info.name!r} has wrong dtype: {value.dtype}")
            if len(value.shape) != len(info.shape) or any(
                expected >= 0 and actual != expected
                for actual, expected in zip(value.shape, info.shape, strict=True)
            ):
                raise ValueError(f"Cluster parameter {info.name!r} has wrong shape: {tuple(value.shape)}")
        writable = [index for index, info in enumerate(infos) if info.direction.name in ("Out", "InOut")]
        call_config = _call_config_record(
            _make_call_config(
                compiled.cluster_config,
                config if config is not None else self._config,
                dfx_base=compiled.output_dir / "dfx_outputs",
            )
        )
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("ClusterWorker is closed")
            if self._active_io is not None:
                raise RuntimeError("Previous cluster invocation requires a successful close before reuse")
            packed = PackedValues(values, writable, self._data_dir)
            self._active_io = packed
            try:
                request = packed.request()
                request.update(op="run", program=item[0], call_config=call_config)
                self._session.request(request)
                packed.publish()
                packed.close()
                self._active_io = None
            except BaseException:
                self.close()
                raise
        if return_style:
            result = [values[index] for index in outputs]
            return result[0] if len(result) == 1 else tuple(result)
        return None

    def __call__(self, *args: Any, config: Any = None) -> Any:
        return self.run(self._compiled, *args, config=config)

    def submit(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ClusterWorker supports synchronous run only")

    def close(self) -> None:
        """Release I/O only after rank cleanup and launcher termination proof."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            try:
                if self._session is not None:
                    self._session.close()
                if self._active_io is not None:
                    self._active_io.close()
                    self._active_io = None
                self._states.clear()
                self._temporary.cleanup()
                self._closed = True
                _UNCONFIRMED_WORKERS.discard(self)
            except BaseException:
                _UNCONFIRMED_WORKERS.add(self)
                raise

    def __enter__(self) -> "ClusterWorker":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
