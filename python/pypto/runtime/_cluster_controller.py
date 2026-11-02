# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Portable execution engine installed in the generated runtime source package."""

import importlib
import threading
from dataclasses import dataclass, field
from typing import Any

import torch


def _freeze(value: Any) -> Any:
    return tuple(_freeze(item) for item in value) if isinstance(value, (list, tuple)) else value


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


class _RetainedDomainLease:
    """Context-manager view that keeps its physical CommDomain alive."""

    def __init__(self, handle: Any) -> None:
        """Wrap a retained CommDomain handle without taking release ownership."""
        self._handle = handle

    def __enter__(self) -> Any:
        """Return the retained CommDomain handle to generated orchestration."""
        return self._handle

    def __exit__(self, *_exc: Any) -> bool:
        """Leave the generated scope without releasing the retained handle."""
        return False


class ClusterController:
    """Persistent runtime-only root owned by the static direct controller rank."""

    def __init__(self, context: Any, spec: dict[str, Any]) -> None:
        from simpler.worker import RemoteCallable  # noqa: PLC0415

        self._spec = spec
        self._persistent = bool(spec["persistent"])
        self._reset = bool(spec["reset"])
        self._closed = False
        self._closing = False
        self._lock = threading.RLock()
        self._active_io = None
        self._keep = []
        self._domains = {}
        self._states = {}
        self._nodes = tuple(executor.worker_id for executor in context.topology.executors)
        self._worker = None
        try:
            self._worker = context.create_worker(
                platform=spec["platform"],
                runtime=spec["runtime"],
                startup_timeout_s=spec["startup_timeout_s"],
                num_sub_workers=0,
                enable_sdma=spec["enable_sdma"],
            )
            for index, program in enumerate(spec["programs"]):
                callables = {
                    record["entry_name"]: [
                        self._worker.register(
                            RemoteCallable(
                                program["module_prefix"] + "." + record["module"] + ":" + record["qualname"]
                            ),
                            workers=list(self._nodes),
                        )
                    ]
                    * len(self._nodes)
                    for record in program["entries"]
                }
                module = importlib.import_module(program["module_prefix"] + ".orchestration.cluster_orch")
                module._io_arg = _io_arg
                plans = [dict(plan, pool_key=_freeze(plan["pool_key"])) for plan in program["plans"]]
                self._states[index] = {
                    "entry": getattr(module, program["entry_name"]),
                    "allocate": getattr(module, "_alloc_intermediates", None),
                    "callables": callables,
                    "plans": plans,
                    "provider": program["provider"],
                    "param_names": program["param_names"],
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

    def run(self, program_index: int, values: list[Any], call_config: dict[str, Any]) -> None:
        """Publish ordinary outputs only after the direct worker's run fence."""
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("ClusterController is closed or requires close")
            if self._active_io is not None:
                raise RuntimeError("Previous cluster invocation requires a successful close before reuse")
            if type(program_index) is not int or program_index not in self._states:
                raise ValueError("Unknown prepared cluster program")
            state = self._states[program_index]
            tensors = dict(zip(state["param_names"], values, strict=True))
            native_config = CallConfig()
            for key, value in call_config.items():
                if key == "runtime_env":
                    for field_name, field_value in value.items():
                        setattr(native_config.runtime_env, field_name, field_value)
                else:
                    setattr(native_config, key, value)
            io = _InvocationIO(self._worker)
            self._active_io = io

            def orchestration(orch: Any, unused: Any, unused_config: Any) -> None:
                provider = self._domain_provider(state, orch) if state["provider"] else None
                state["entry"](
                    orch,
                    unused,
                    native_config,
                    tensors=tensors,
                    callables=state["callables"],
                    nodes=list(self._nodes),
                    node_sizes=self._spec["node_sizes"],
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
                self._worker.run(orchestration, config=native_config)
                io.publish()
                io.release()
                released = True
            except BaseException:
                self.close()
                raise
            finally:
                if self._closed or released:
                    self._keep.clear()
                    self._active_io = None

    def close(self) -> None:
        """Keep all journals reachable until the direct worker confirms drain."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            first_error = None
            if self._active_io is None:
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
            self._closed = True
            if first_error is not None:
                raise first_error


def main(context: Any) -> None:
    """Serve the authenticated local parent for the lifetime of one MPI world."""
    from ._cluster_channel import connect_session, map_values, recv_record, send_record  # noqa: PLC0415

    parameters = context.startup_parameters
    sock = connect_session(
        parameters["address"], parameters["authkey"], parameters["session"]["startup_timeout_s"]
    )
    controller = None
    try:
        controller = ClusterController(context, parameters["session"])
        send_record(sock, {"type": "ready", "transport": "direct", "executor_count": len(controller._nodes)})
        while True:
            request = recv_record(sock)
            if request["op"] == "close":
                controller.close()
                send_record(sock, {"type": "closed"})
                return
            if request["op"] != "run":
                raise ValueError("Unknown cluster session operation")
            try:
                with map_values(request, parameters["data_dir"]) as values:
                    try:
                        controller.run(request["program"], values, request["call_config"])
                    finally:
                        del values
                send_record(sock, {"type": "done"})
            except BaseException as error:
                send_record(
                    sock, {"type": "error", "error_type": type(error).__name__, "message": str(error)}
                )
                raise
    finally:
        try:
            if controller is not None:
                controller.close()
        finally:
            sock.close()
