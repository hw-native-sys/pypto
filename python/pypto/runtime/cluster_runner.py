# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synchronous CLUSTER_0 execution with parent compilation and rank-local I/O."""

import ctypes
import importlib.util
import json
import math
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pypto.ir.compiled_program import _coerce_args
from pypto.jit._artifact_manifest import BuildKind

from ._artifact_sources import build_cluster_session_bundle, build_cluster_source_bundle
from ._prebuilt import chip_directories
from .distributed_runner import _make_call_config, _resolve_persistent_window_reset, _RetainedDomainLease


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


def _intervals(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(values):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        elif start < end:
            result.append((start, end))
    return result


@dataclass
class _Storage:
    """One owned storage range; strong references forbid address reuse mid-run."""

    storage: Any
    tensor: torch.Tensor
    worker: int
    handle: Any
    writes: list[tuple[int, int]] = field(default_factory=list)


class _InvocationIO:
    """Ordinary contiguous tensors staged through rank-local remote allocations.

    Offsets refer to the Torch storage base, preserving overlapping aliases on
    one rank. Cross-rank write/read or write/write overlaps require explicit
    forwarding and are rejected, including when both MPI ranks share a host.
    """

    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.storages: dict[tuple[int, int, int], _Storage] = {}
        self.accesses: list[tuple[int, int, int, bool]] = []

    def prepare(self, tensors: dict[str, Any], nodes: tuple[int, ...]) -> None:
        """Stage owned backings before entering the graph's first submission.

        Ordinary routing can depend on generated Python control flow. Stage the
        complete static tensor inventory on every node so graph construction
        performs descriptor lookup only, respecting Simpler's mailbox ordering.
        """
        for tensor in tensors.values():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError("Cluster ordinary I/O requires contiguous CPU tensors")
            storage = tensor.untyped_storage()
            base, size = storage.data_ptr(), storage.nbytes()
            if size <= 0:
                raise ValueError("Cluster ordinary I/O requires nonempty tensor storage")
            for node in nodes:
                key = (node, base, size)
                if key in self.storages:
                    continue
                handle = self.worker.remote_malloc(worker=node, nbytes=size)
                owned = _Storage(storage, tensor, node, handle)
                self.storages[key] = owned
                self.worker.remote_copy_to(handle, base, size)

    def argument(self, rank: int, tensor: torch.Tensor, tag: Any) -> Any:
        from simpler.task_interface import DataType, RemoteTensorRef, TensorArgType  # noqa: PLC0415

        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("Cluster ordinary I/O requires contiguous CPU torch.Tensor arguments")
        storage = tensor.untyped_storage()
        base, size = storage.data_ptr(), storage.nbytes()
        start = tensor.data_ptr() - base
        nbytes = tensor.numel() * tensor.element_size()
        end = start + nbytes
        if nbytes <= 0:
            raise ValueError("Cluster ordinary I/O requires nonempty tensor views")
        if start < 0 or end > size:
            raise ValueError("Cluster tensor view lies outside its owned storage range")
        reading = tag in (TensorArgType.INPUT, TensorArgType.INOUT)
        writing = tag in (TensorArgType.OUTPUT, TensorArgType.OUTPUT_EXISTING, TensorArgType.INOUT)
        if not reading and not writing:
            raise ValueError(f"Unsupported cluster tensor dependency tag {tag!r}")
        absolute_start, absolute_end = base + start, base + end
        for previous_rank, previous_start, previous_end, previous_write in self.accesses:
            if (
                previous_rank != rank
                and (writing or previous_write)
                and max(absolute_start, previous_start) < min(absolute_end, previous_end)
            ):
                raise NotImplementedError(
                    "Cross-node ordinary tensor forwarding requires explicit transfer and fences; "
                    "use independent outputs until global-domain execution is available"
                )
        self.accesses.append((rank, absolute_start, absolute_end, writing))
        key = (rank, base, size)
        owned = self.storages.get(key)
        if owned is None:
            raise NotImplementedError(
                "Generated cluster root introduced tensor storage after I/O preparation; "
                "all ordinary tensor storage must be declared in _alloc_intermediates"
            )
        if writing:
            owned.writes = _intervals([*owned.writes, (start, end)])
        dtype_name = str(tensor.dtype).removeprefix("torch.").upper()
        dtype_name = {"BFLOAT16": "BFLOAT16", "FLOAT": "FLOAT32", "DOUBLE": "FLOAT64"}.get(
            dtype_name, dtype_name
        )
        dtype = getattr(DataType, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported cluster tensor dtype {tensor.dtype}")
        return RemoteTensorRef(
            owned.handle, offset=start, shape=tuple(tensor.shape), dtype=dtype, nbytes=nbytes
        )

    def publish(self) -> None:
        """Copy all ranks' completed outputs before releasing any allocation."""
        for owned in self.storages.values():
            for start, end in owned.writes:
                self.worker.remote_copy_from(
                    owned.handle,
                    owned.storage.data_ptr() + start,
                    end - start,
                    offset=start,
                )

    def release(self) -> None:
        # Keep every allocation journaled if a free fails; whole-tree close
        # remains the reclamation boundary and still owns every strong ref.
        for owned in self.storages.values():
            self.worker.remote_free(owned.handle)
        self.clear()

    def clear(self) -> None:
        self.storages.clear()
        self.accesses.clear()


def _io_arg(binding: tuple[_InvocationIO, int], tensor: torch.Tensor, tag: Any) -> Any:
    return binding[0].argument(binding[1], tensor, tag)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _session_domains(programs: list[Any], bindings: Any, persistent: bool) -> list[list[dict[str, Any]]]:
    """Snapshot explicit domain identities and compare canonical physical layouts."""
    from pypto.ir.distributed_compiled_program import SharedDomainBinding  # noqa: PLC0415

    if bindings and not persistent:
        raise ValueError("shared_domains requires persistent=True")
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
            static = all(slot[1] is not None for slot in canonical)
            if sharing and not static:
                raise ValueError("Shared domain layout requires static slot sizes")
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
                    "static": static,
                }
            )
        result.append(plans)
    return result


def _node_domain_bindings(plans: list[dict[str, Any]], logical_path: str) -> list[dict[str, Any]]:
    bindings = []
    for plan in plans:
        if not plan["static"]:
            continue
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


class ClusterWorker:
    """Prepared CLUSTER_0 session for synchronous Tensor/Scalar and global-window runs.

    All CHIP compilation happens on the parent. The installed rank source
    package imports only Simpler/Torch and is prepared before the inner fork.
    Sessions accept ordinary CPU tensors allocated after preparation.
    """

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
        from simpler.startup_artifacts import StartupArtifactSpec  # noqa: PLC0415
        from simpler.startup_preparation import NodeStartupHook  # noqa: PLC0415
        from simpler.worker import MpiL3GroupSpec, RemoteCallable  # noqa: PLC0415

        programs = list({id(program): program for program in programs}.values())
        compiled = programs[0]
        cc = compiled.cluster_config
        for program in programs:
            if program.entry_kind != "cluster":
                raise ValueError("ClusterWorker requires CLUSTER_0 compiled programs")
            program._require_artifact_ready()
            if program.cluster_config != cc or program.platform != compiled.platform:
                raise ValueError(
                    "Cluster programs require compatible topology/platform/runtime/profile config"
                )
            for record in program.l3_entries["entries"]:
                required = program.output_dir / record["logical_path"] / "sub_workers/__required__.json"
                if required.exists() and json.loads(required.read_text(encoding="utf-8")):
                    raise NotImplementedError("Cluster runtime-bound SUB callbacks are unsupported")
        self._persistent = bool(persistent)
        self._reset = _resolve_persistent_window_reset(persistent, reset_persistent_windows)
        plans = _session_domains(programs, shared_domains, persistent)
        aggregate = len(programs) > 1 or persistent
        timeout = cc.startup_timeout_s if startup_timeout_s is None else startup_timeout_s
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Cluster startup_timeout_s must be positive and finite")
        chip_sets = [_assemble_cluster_chips(program) for program in programs]
        if any(runtime != cc.runtime for chips in chip_sets for _, runtime, _ in chips.values()):
            raise ValueError("Every cluster CHIP dependency must use ClusterConfig.runtime")
        digest, payload = (
            build_cluster_session_bundle([program.output_dir for program in programs])
            if aggregate
            else build_cluster_source_bundle(compiled.output_dir)
        )
        self._compiled = compiled
        self._closed = False
        self._closing = False
        self._lock = threading.RLock()
        self._active_io: _InvocationIO | None = None
        self._keep: list[Any] = []
        self._temporary = tempfile.TemporaryDirectory(prefix="pypto-cluster-")
        self._worker = None
        self._domains: dict[tuple[Any, ...], Any] = {}
        self._states: dict[int, dict[str, Any]] = {}
        try:
            archive = Path(self._temporary.name) / "sources.tar.gz"
            archive.write_bytes(payload)
            prefix = "_pypto_cluster_" + digest
            self._worker = Worker(
                level=4,
                platform=compiled.platform,
                runtime=cc.runtime,
                startup_timeout_s=timeout,
                enable_sdma=any(
                    bool(value[2].get("enable_sdma", False))
                    for chips in chip_sets
                    for value in chips.values()
                ),
            )
            prepared = []
            for index, (program, chips) in enumerate(zip(programs, chip_sets, strict=True)):
                prepared.append(
                    {
                        "module_prefix": prefix + (f".programs.p{index}" if aggregate else ""),
                        "entries": program.l3_entries["entries"],
                        "chip_digests": {
                            path: self._worker.register(chip).digest.hex()
                            for path, (chip, _, _) in chips.items()
                        },
                        "chip_configs": {path: value[2] for path, value in chips.items()},
                    }
                )
                if aggregate:
                    prepared[-1]["domain_bindings"] = {
                        record["logical_path"]: _node_domain_bindings(plans[index], record["logical_path"])
                        for record in prepared[-1]["entries"]
                    }
            parameters = (
                {"module_prefix": prefix, "programs": prepared, "version": 1} if aggregate else prepared[0]
            )
            records = [record for program in prepared for record in program["entries"]]
            spec = MpiL3GroupSpec(
                hosts=tuple(node.host for node in cc.nodes),
                platform=compiled.platform,
                device_ids_by_rank=tuple(tuple(node.devices) for node in cc.nodes),
                runtime=cc.runtime,
                num_sub_workers_by_rank=(
                    max(cc.num_sub_workers, sum(record["minimum_sub_workers"] for record in records)),
                )
                * len(cc.nodes),
                comm_profile=cc.comm_profile,
                global_device_ranks_by_rank=tuple(cc.global_device_ranks()),
                mpirun_path=cc.mpirun_path,
                mpirun_args=tuple(cc.mpirun_args),
                python_executable=cc.python_executable,
                startup_artifacts=(StartupArtifactSpec(archive, digest, prefix),),
                startup_hooks_by_rank=tuple(
                    NodeStartupHook(prefix + ".orchestration.cluster_bootstrap:prepare_node", parameters)
                    for _ in cc.nodes
                ),
            )
            self._nodes = self._worker.add_mpirun_worker_group(spec)
            for program, specification, domain_plans in zip(programs, prepared, plans, strict=True):
                callables = {
                    record["entry_name"]: [
                        self._worker.register(
                            RemoteCallable(
                                specification["module_prefix"]
                                + "."
                                + record["module"]
                                + ":"
                                + record["qualname"]
                            ),
                            workers=list(self._nodes),
                        )
                    ]
                    * len(self._nodes)
                    for record in specification["entries"]
                }
                path = program.output_dir / "orchestration/cluster_orch.py"
                module_spec = importlib.util.spec_from_file_location(
                    specification["module_prefix"] + "_parent", path
                )
                if module_spec is None or module_spec.loader is None:
                    raise RuntimeError(f"Cannot load generated cluster root {path}")
                module = importlib.util.module_from_spec(module_spec)
                module._io_arg = _io_arg
                # Never write bytecode or modify a published artifact inventory.
                exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
                self._states[id(program)] = {
                    "program": program,
                    "entry": getattr(module, program.entry_name),
                    "allocate": getattr(module, "_alloc_intermediates", None),
                    "callables": callables,
                    "plans": domain_plans,
                    "provider": aggregate,
                }
            self._worker.init()
        except BaseException:
            self.close()
            raise

    def _domain_provider(self, state: dict[str, Any], orch: Any) -> Any:
        """Prepare current root scopes before submission, then validate native requests."""
        from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

        local = {}
        by_name = {plan["domain"]["generated_name"]: plan for plan in state["plans"]}

        def signature(kwargs):
            return (
                tuple(tuple(member) for member in kwargs["members"]),
                int(kwargs["window_size"]),
                tuple(
                    (buffer.name, buffer.dtype, int(buffer.count), int(buffer.nbytes))
                    for buffer in kwargs["buffers"]
                ),
            )

        def acquire(plan, kwargs):
            key = plan["pool_key"]
            spec = signature(kwargs)
            existing = self._domains.get(key) if self._persistent else local.get(key)
            if existing is not None:
                previous, handle = existing
                if previous != spec or handle.released or handle.freed:
                    raise ValueError("Retained global domain changed native specification or lifetime")
                return handle, True
            handle = orch.allocate_global_domain(**kwargs)
            expected = [
                (node, local_worker, rank, i)
                for i, (node, local_worker, rank) in enumerate(
                    (self._nodes[node], local_worker, rank) for node, local_worker, rank in plan["members"]
                )
            ]
            actual = [
                (member.node_worker_id, member.local_worker_id, member.global_device_rank, member.domain_rank)
                for member in handle.members
            ]
            if actual != expected or handle.mapping_size < kwargs["window_size"]:
                # R1 still owns the new handle in its run journal on this path.
                raise ValueError("Allocated global domain differs from requested ordered members/window")
            (self._domains if self._persistent else local)[key] = (spec, handle)
            return handle, False

        requests = {}
        reset = set()
        preexisting = set(self._domains)
        for plan in state["plans"]:
            if not plan["static"]:
                continue
            kwargs = {
                "name": plan["name"],
                "members": [(self._nodes[node], local_worker) for node, local_worker, _ in plan["members"]],
                "window_size": sum(slot[2] for slot in plan["canonical"]),
                "buffers": [
                    CommBufferSpec(name=name, dtype="opaque", count=size, nbytes=allocated)
                    for name, size, allocated, _ in plan["canonical"]
                ],
                "retain_after_run": self._persistent,
            }
            handle, reused = acquire(plan, kwargs)
            requests[plan["domain"]["generated_name"]] = handle
            if reused and self._reset and plan["pool_key"] in preexisting and plan["pool_key"] not in reset:
                zero = bytes(min(handle.mapping_size, 8 * 1024 * 1024))
                for rank in range(len(handle.members)):
                    for offset in range(0, handle.mapping_size, len(zero)):
                        orch.copy_to_global_domain(
                            handle, rank, zero[: min(len(zero), handle.mapping_size - offset)], offset=offset
                        )
                reset.add(plan["pool_key"])

        def provide(**kwargs):
            name = kwargs["name"]
            if name not in by_name:
                raise ValueError(f"Native global domain request {name!r} is absent from cold metadata")
            plan = by_name[name]
            if not plan["static"]:
                raise ValueError("Persistent cluster domain requires a static cold slot layout")
            expected = (
                tuple((self._nodes[node], local_worker) for node, local_worker, _ in plan["members"]),
                sum(slot["allocated_nbytes"] for slot in plan["domain"]["slots"]),
                tuple(
                    (slot["generated_name"], "opaque", slot["logical_nbytes"], slot["allocated_nbytes"])
                    for slot in plan["domain"]["slots"]
                ),
            )
            if (
                signature(kwargs) != expected
                or bool(kwargs.get("retain_after_run", False)) != self._persistent
            ):
                raise ValueError(f"Native global domain {name!r} request differs from cold specification")
            handle = requests[name]
            return _RetainedDomainLease(handle) if self._persistent else handle

        return provide

    def run(self, compiled: Any, *args: Any, config: Any = None) -> Any:
        """Execute the generated parent root and publish results after its fence."""
        if id(compiled) not in self._states or self._states[id(compiled)]["program"] is not compiled:
            raise ValueError("ClusterWorker.run requires its prepared compiled program")
        state = self._states[id(compiled)]
        infos, outputs, returns = compiled._get_metadata()
        values, return_style = _coerce_args(args, infos, outputs, returns, caller_name="ClusterWorker")
        for info, value in zip(infos, values, strict=True):
            if info.shape is not None:
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or not value.is_contiguous()
                ):
                    raise ValueError(f"Cluster parameter {info.name!r} requires a contiguous CPU tensor")
                if str(value.dtype).removeprefix("torch.") != {
                    "fp16": "float16",
                    "bf16": "bfloat16",
                    "fp32": "float32",
                }.get(str(info.dtype), str(info.dtype)):
                    raise ValueError(f"Cluster parameter {info.name!r} has wrong dtype: {value.dtype}")
                if len(value.shape) != len(info.shape) or any(
                    expected >= 0 and actual != expected
                    for actual, expected in zip(value.shape, info.shape, strict=True)
                ):
                    raise ValueError(f"Cluster parameter {info.name!r} has wrong shape: {tuple(value.shape)}")
        tensors = {
            info.name: value.value if isinstance(value, ctypes._SimpleCData) else value
            for info, value in zip(infos, values, strict=True)
        }
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("ClusterWorker is closed")
            if self._active_io is not None:
                raise RuntimeError("Previous cluster invocation requires a successful close before reuse")
            io = _InvocationIO(self._worker)
            self._active_io = io
            cc = compiled.cluster_config

            def orchestration(orch: Any, unused: Any, unused_config: Any) -> None:
                provider = self._domain_provider(state, orch) if state["provider"] else None
                state["entry"](
                    orch,
                    unused,
                    call_config,
                    tensors=tensors,
                    callables=state["callables"],
                    nodes=list(self._nodes),
                    node_sizes=cc.node_device_counts(),
                    node_bindings=[(io, node) for node in self._nodes],
                    _keep=self._keep,
                    _retain=self._persistent,
                    **({"_domain_provider": provider} if provider is not None else {}),
                )

            released = False
            try:
                if state["allocate"] is not None:
                    state["allocate"](tensors, nodes=list(self._nodes))
                io.prepare(tensors, self._nodes)
                call_config = _make_call_config(cc, config, dfx_base=compiled.output_dir / "dfx_outputs")
                self._worker.run(orchestration, config=call_config)
                io.publish()
                io.release()
                released = True
            except BaseException:
                # A native failure may leave accepted work unsettled. Closing
                # the hierarchy drains/reaps it before releasing metadata; no
                # partial output is copied after a failed DAG.
                self.close()
                raise
            finally:
                if self._closed or released:
                    # An unsuccessful tree close leaves acceptance uncertain;
                    # its submitted argument frames remain owned until retry.
                    self._keep.clear()
                    self._active_io = None
        if return_style:
            result = [values[index] for index in outputs]
            return result[0] if len(result) == 1 else tuple(result)
        return None

    def __call__(self, *args: Any, config: Any = None) -> Any:
        return self.run(self._compiled, *args, config=config)

    def submit(self, *args: Any, **kwargs: Any) -> Any:
        """Reject asynchronous admission before it can allocate or submit work."""
        raise NotImplementedError("ClusterWorker supports synchronous run only")

    def close(self) -> None:
        """Drain the complete worker tree before releasing invocation storage."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            first_error = None
            if self._active_io is None:
                # Successful synchronous runs already supplied the drain proof.
                # Keep failed handles reachable for R1's close journal to retry.
                for _, handle in reversed(tuple(self._domains.values())):
                    if not handle.freed:
                        try:
                            handle.release()
                        except BaseException as error:
                            if first_error is None:
                                first_error = error
            if self._worker is not None:
                self._worker.close()
            if self._active_io is not None:
                self._active_io.clear()
                self._active_io = None
            self._keep.clear()
            self._domains.clear()
            self._states.clear()
            self._temporary.cleanup()
            self._closed = True
            if first_error is not None:
                raise first_error

    def __enter__(self) -> "ClusterWorker":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
