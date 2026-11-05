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
from collections.abc import Mapping
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
from .distributed_runner import _make_call_config, _resolve_persistent_window_reset

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


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _session_domains(programs: list[Any], bindings: Any) -> list[list[dict[str, Any]]]:
    """Snapshot explicit domain identities and compare canonical physical layouts."""
    from pypto.ir.distributed_compiled_program import SharedDomainBinding  # noqa: PLC0415

    selected = {}
    keys: dict[str, int] = {}
    for binding in tuple(bindings):
        if not isinstance(binding, SharedDomainBinding):
            raise TypeError("shared_domains requires SharedDomainBinding values")
        index = next((i for i, program in enumerate(programs) if binding.program is program), None)
        if index is None:
            raise ValueError("Shared domain program is not prepared")
        if type(binding.domain) is not int or not 0 <= binding.domain < len(programs[index].comm_domains):
            raise ValueError("Shared domain ordinal is unavailable; regenerate legacy artifacts")
        if not isinstance(binding.key, str) or not binding.key:
            raise ValueError("Shared domain key must be nonempty")
        identity = (index, binding.domain)
        if identity in selected:
            raise ValueError("Duplicate shared domain binding")
        if not isinstance(binding.slots, Mapping):
            raise TypeError("Shared domain slots must be a mapping")
        slots = dict(binding.slots)
        if (
            any(type(slot) is not int for slot in slots)
            or set(slots) != set(range(len(programs[index].comm_domains[binding.domain]["slots"])))
            or any(not isinstance(name, str) or not name for name in slots.values())
            or len(set(slots.values())) != len(slots)
        ):
            raise ValueError("Shared domain slots require a complete one-to-one canonical slot map")
        keys.setdefault(binding.key, len(keys))
        selected[identity] = (binding.key, slots)
    signatures = {}
    result = []
    for index, program in enumerate(programs):
        cc = program.cluster_config
        global_ranks = cc.global_device_ranks()
        plans = []
        for domain in _plain(program.comm_domains):
            identity = (index, domain["ordinal"])
            sharing = selected.get(identity)
            slot_map = (
                sharing[1]
                if sharing
                else {slot["ordinal"]: slot["generated_name"] for slot in domain["slots"]}
            )
            canonical = []
            for slot in domain["slots"]:
                views = sorted(
                    set(
                        (
                            window["dtype"],
                            tuple(window["shape"]),
                            tuple(window["strides"]),
                            window["byte_offset"],
                        )
                        for window in domain["entry_windows"]
                        if window["slot_ordinal"] == slot["ordinal"]
                    )
                )
                canonical.append(
                    (
                        slot_map[slot["ordinal"]],
                        slot["logical_nbytes"],
                        slot["allocated_nbytes"],
                        tuple(views),
                    )
                )
            if sharing:
                canonical.sort()
            members = [
                (node, local, global_ranks[node][local])
                for node in domain["node_indices"]
                for local in range(len(cc.nodes[node].devices))
            ]
            if any(slot[1] is None for slot in canonical):
                raise ValueError("Persistent cluster domain requires static slot sizes")
            signature = (tuple(members), tuple(canonical))
            pool_key = ("shared", sharing[0]) if sharing else ("isolated", index, domain["ordinal"])
            if pool_key in signatures and signatures[pool_key] != signature:
                raise ValueError("Shared domain has incompatible member/window/dtype/view layout")
            signatures[pool_key] = signature
            runtime_name = (
                f"pypto:shared:{keys[sharing[0]]}" if sharing else f"pypto:p{index}:{domain['ordinal']}"
            )
            plans.append(
                {
                    "domain": domain,
                    "pool_key": pool_key,
                    "name": runtime_name,
                    "slot_map": slot_map,
                    "canonical": canonical,
                    "members": members,
                }
            )
        result.append(plans)
    return result


def _node_domain_bindings(plans: list[dict[str, Any]], logical_path: str) -> list[dict[str, Any]]:
    bindings = []
    for plan in plans:
        windows = [
            window
            for window in plan["domain"]["entry_windows"]
            if window["entry_logical_path"] == logical_path
        ]
        if windows:
            bindings.append(
                {
                    "name": plan["name"],
                    "members": [[rank, local, i] for i, (_, local, rank) in enumerate(plan["members"])],
                    "window_size": sum(slot[2] for slot in plan["canonical"]),
                    "slots": [[slot[0], slot[2]] for slot in plan["canonical"]],
                    "windows": [
                        [
                            window["ctx_scalar_index"],
                            window["logical_window_name"],
                            plan["slot_map"][window["slot_ordinal"]],
                        ]
                        for window in windows
                    ],
                }
            )
    merged = {}
    for binding in bindings:
        if binding["name"] in merged:
            merged[binding["name"]]["windows"].extend(binding["windows"])
        else:
            merged[binding["name"]] = binding
    return list(merged.values())


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
        persistent: bool = False,
        reset_persistent_windows: bool | None = None,
        shared_domains: Any = (),
    ) -> None:
        from simpler.mpi_direct_topology import MpiDirectExecutorSpec  # noqa: PLC0415
        from simpler.startup_artifacts import StartupArtifactSpec, read_startup_artifacts  # noqa: PLC0415
        from simpler.startup_preparation import NodeStartupHook  # noqa: PLC0415

        programs = list({id(program): program for program in programs}.values())
        if not programs:
            raise ValueError("ClusterWorker requires at least one compiled program")
        compiled = programs[0]
        cc = compiled.cluster_config
        _validate_programs(programs, compiled, cc)
        if shared_domains and not persistent:
            raise ValueError("shared_domains requires persistent=True")
        plans = _session_domains(programs, shared_domains) if persistent else [[] for _ in programs]
        timeout = cc.startup_timeout_s if startup_timeout_s is None else startup_timeout_s
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Cluster startup_timeout_s must be positive and finite")
        reset = _resolve_persistent_window_reset(persistent, reset_persistent_windows)
        chip_sets = [_assemble_cluster_chips(program) for program in programs]
        if any(runtime != cc.runtime for chips in chip_sets for _, runtime, _ in chips.values()):
            raise ValueError("Every cluster CHIP dependency must use ClusterConfig.runtime")
        digest, payload = build_cluster_session_bundle(programs)
        prefix = "_pypto_cluster_" + digest
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
            archive = Path(self._temporary.name) / "sources.tar.gz"
            archive.write_bytes(payload)
            artifacts = read_startup_artifacts([StartupArtifactSpec(archive, digest, prefix).to_manifest()])
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
            prepared = []
            execution = []
            for index, (program, chips) in enumerate(zip(programs, chip_sets, strict=True)):
                program_prefix = prefix + f".programs.p{index}"
                prepared.append(
                    {
                        "module_prefix": program_prefix,
                        "entries": program.l3_entries["entries"],
                        "chip_digests": {
                            path: staging.register(chip).digest.hex() for path, (chip, _, _) in chips.items()
                        },
                        "chip_configs": {path: value[2] for path, value in chips.items()},
                        "domain_bindings": {
                            record["logical_path"]: _node_domain_bindings(
                                plans[index], record["logical_path"]
                            )
                            for record in program.l3_entries["entries"]
                        },
                    }
                )
                infos, _, _ = program._get_metadata()
                execution.append(
                    {
                        "entry_name": program.entry_name,
                        "module_prefix": program_prefix,
                        "entries": program.l3_entries["entries"],
                        "param_names": [info.name for info in infos],
                        "plans": [
                            {
                                key: _plain(plan[key])
                                for key in ("domain", "pool_key", "name", "canonical", "members")
                            }
                            for plan in plans[index]
                        ],
                    }
                )
            minimum = sum(
                record["minimum_sub_workers"] for program in prepared for record in program["entries"]
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
            hook = NodeStartupHook(
                prefix + ".orchestration.cluster_bootstrap:prepare_node", {"programs": prepared}
            ).to_manifest()
            session = {
                "platform": compiled.platform,
                "runtime": cc.runtime,
                "startup_timeout_s": timeout,
                "enable_sdma": enable_sdma,
                "persistent": bool(persistent),
                "reset": reset,
                "node_sizes": cc.node_device_counts(),
                "programs": execution,
            }
            records = {}
            for spec in executors:
                typed = MpiDirectExecutorSpec.from_dict(spec)
                records[spec["rank"]] = {
                    "startup_artifacts": artifacts,
                    "startup_hook": hook,
                    "inner_l3_worker": staging._inner_registry_entries_for_spec(typed),
                }
            records[0] = {
                "startup_artifacts": artifacts,
                "inner_l3_worker": records[1]["inner_l3_worker"],
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
