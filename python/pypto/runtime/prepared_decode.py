# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cache fixed decode rank bindings without caching submission authorization."""

from __future__ import annotations

import ast
import importlib
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns
from typing import Any, cast

import torch

from .device_tensor import DeviceTensor, StackedDeviceTensor

# Exact orchestration sources audited for static domain specs, rank-only graph
# construction, no scalar public args, and no cross-rank TaskHandle references.
HOST_SOURCES: set[str] = {
    "e4b07c15b7b6007f070bd1a9f03986436bc116ebb75695a708a703e7fbfdb380",  # wave13-p3a-head8-aligned-r3 one
    "b5b110960fee9fdea16908d1d7d3d810b27f5efc0a515ef43290e10d9d64ddfd",  # wave13-p3a-head8-aligned-r3 two
    "fcd9000ce64441a472ee4b06c1e60c5466c58787fc662d25f51461bb457102ef",  # wave13-p3b-drafter-tp8-r1 one
    "9fcbd4ace0c430420cf731405df386c434646d5029ca45064107b6e7d2e09028",  # wave13-p3b-drafter-tp8-r1 two
}

_OwnerRecord = tuple[tuple[int, int], Any]
_CacheEntry = tuple[tuple[Any, ...], tuple[_OwnerRecord, ...], list[Any], tuple[tuple[str, Any], ...]]
_Selection = tuple[Any, _CacheEntry | None]


def _domain_scope(
    orch: Any,
    registered: Mapping[str, Any],
    prefix: str,
    name: str,
    factory: Callable[[], Any],
    lease_cls: Callable[[Any], Any],
) -> Any:
    namespaced = prefix + name
    previous = registered.get(name)
    if previous is None:
        return factory()
    handle = orch.get_domain(namespaced)
    if handle is not previous[1] or not handle.retain_after_run:
        raise RuntimeError("Prepared decode requires a retained domain")
    return lease_cls(handle)


def _host_key(value: torch.Tensor) -> tuple[Any, ...]:
    storage_cdata: Any = getattr(value.untyped_storage(), "_cdata")
    is_shared = bool(getattr(value, "is_shared")())
    return (
        id(value),
        value.data_ptr(),
        storage_cdata,
        tuple(value.shape),
        value.stride(),
        value.dtype,
        is_shared,
    )


def _binding_key(value: Any) -> Any:
    if type(value) is DeviceTensor or type(value) is StackedDeviceTensor:
        return id(value)
    if type(value) is torch.Tensor:
        return _host_key(value)
    if type(value) is list:
        items = cast(list[Any], value)
        if all(type(item) is torch.Tensor for item in items):
            return tuple(_host_key(item) for item in items)
    raise TypeError("Unsupported prepared decode binding")


def _owner_records(args: Sequence[Any]) -> Iterator[_OwnerRecord]:
    for value in args:
        if type(value) is StackedDeviceTensor:
            for shard, rank in zip(value.shards, value.worker_ids, strict=True):
                yield (rank, shard.data_ptr - shard.byte_offset), shard.buffer
        elif type(value) is DeviceTensor:
            owner_rank = int(getattr(value.buffer, "owner_worker_id"))
            yield (owner_rank, value.data_ptr - value.byte_offset), value.buffer


class PrepareRanks(ast.NodeTransformer):
    def visit_For(self, node: ast.For) -> ast.AST:
        # The pinned entry contains only independent rank loops. The caller
        # executes each entire loop once on a cache miss to create snapshots.
        visited = self.generic_visit(node)
        if not isinstance(visited, ast.stmt):
            raise TypeError("Prepared decode rank loop must remain a statement")
        body: list[ast.stmt] = [visited]
        return ast.copy_location(
            ast.If(
                test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="_cache_hit", ctx=ast.Load())),
                body=body,
                orelse=[],
            ),
            node,
        )

    def visit_With(self, node: ast.With) -> ast.AST:
        (item,) = node.items
        call = item.context_expr
        if not isinstance(call, ast.Call):
            raise ValueError("Prepared decode requires a domain context call")
        name = next(ast.literal_eval(k.value) for k in call.keywords if k.arg == "name")
        if any(isinstance(n, ast.Name) and n.id == "tensors" for n in ast.walk(call)):
            raise ValueError("Dynamic domain definitions need original admission")
        factory_expr = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=call,
        )
        item.context_expr = ast.Call(
            func=ast.Name(id="_domain_scope", ctx=ast.Load()),
            args=[
                ast.Name(id="orch", ctx=ast.Load()),
                ast.Name(id="_registered_domains", ctx=ast.Load()),
                ast.Name(id="_domain_prefix", ctx=ast.Load()),
                ast.Constant(name),
                factory_expr,
                ast.Name(id="_retained_lease", ctx=ast.Load()),
            ],
            keywords=[],
        )
        body: list[ast.stmt] = []
        for stmt in node.body:
            visited = self.visit(stmt)
            if not isinstance(visited, ast.stmt):
                raise TypeError("Prepared decode body item must remain a statement")
            body.append(visited)
        node.body = body
        return node


def load(
    entry: Callable[..., Any],
    path: object,
    lease_cls: Callable[[Any], Any] | None = None,
) -> PreparedDecode | None:
    if not isinstance(path, Path) or not path.is_file():
        return None
    source = path.read_bytes()
    if sha256(source).hexdigest() not in HOST_SOURCES:
        return None
    try:
        prepared_local = importlib.import_module("simpler.prepared_local")
    except ImportError:
        return None
    task_cls: Any = getattr(prepared_local, "PreparedLocalChipTask", None)
    if task_cls is None:
        return None
    entry_name = str(getattr(entry, "__name__"))
    entry_globals: dict[str, Any] = getattr(entry, "__globals__")
    tree = ast.parse(source, filename=str(path))
    fn = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == entry_name)
    kwonly_names = (
        "_cache_hit",
        "_domain_prefix",
        "_registered_domains",
        "_retained_lease",
        "_submit_chip",
    )
    fn.args.kwonlyargs.extend(ast.arg(arg=name) for name in kwonly_names)
    fn.args.kw_defaults.extend([None] * len(kwonly_names))
    PrepareRanks().visit(fn)
    namespace: dict[str, Any] = {**entry_globals, "_domain_scope": _domain_scope}
    module_body: list[ast.stmt] = [fn]
    module = ast.fix_missing_locations(ast.Module(body=module_body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return PreparedDecode(namespace[entry_name], task_cls=task_cls, lease_cls=lease_cls)


class PreparedDecode:
    def __init__(
        self,
        prepare: Callable[..., Any],
        *,
        task_cls: Callable[..., Any],
        lease_cls: Callable[[Any], Any] | None = None,
    ) -> None:
        self.prepare = prepare
        self._task_cls = task_cls
        self._lease_cls = lease_cls
        self.entries: OrderedDict[Any, _CacheEntry] = OrderedDict()
        self.trace: list[dict[str, Any]] | None = None
        self.hits = 0
        self.misses = 0

    def select(
        self,
        worker: Any,
        state: Mapping[str, Any],
        tensors: Mapping[str, Any],
        config: Any,
        args: Sequence[Any],
        slot: int,
    ) -> _Selection | None:
        if (
            config is None
            or config.output_prefix
            or not worker._persistent
            or worker._reset_persistent_windows
        ):
            return None
        try:
            key = (slot, tuple(_binding_key(v) for v in tensors.values()))
        except TypeError:
            return None
        entry = self.entries.get(key)
        if entry is None or any(
            worker._device_buffers.get(owner_key) is not owner for owner_key, owner in entry[1]
        ):
            # Go through the original public validators on any miss or revoke.
            for info, arg in zip(state["param_infos"], args, strict=True):
                worker._validate_prepared_dispatch_arg(info, arg)
            entry = None
        return key, entry

    def submit(
        self,
        selection: _Selection,
        orch: Any,
        config: Any,
        state: Mapping[str, Any],
        *,
        tensors: dict[str, Any],
        keepalive: list[Any],
        provider: Any,
        registered_domains: dict[str, Any],
    ) -> None:
        key, entry = selection
        tasks: list[Any] = [] if entry is None else entry[2]
        domains: Mapping[str, Any] = orch._worker._live_domains
        program = str(state["persistent_id"])
        world = int(state["device_nums"])
        prefix = f"{program}:"
        current = tuple((name, handle) for name, handle in domains.items() if name.startswith(prefix))
        if entry is not None and current != entry[3]:
            tasks = []
            entry = None

        def record(rec_orch: Any, handle: Any, rec_args: Any, _config: Any, rank: int) -> None:
            tasks.append(self._task_cls(rec_orch, handle, rec_args, rank))

        roots = tuple(tensors.values()) if entry is None else entry[0]
        self.prepare(
            orch,
            None,
            config,
            tensors=tensors,
            callables=state["chip_cids"],
            sub_ids=state["sub_ids"],
            _keep=keepalive,
            world_size=world,
            _domain_provider=provider,
            _cache_hit=entry is not None,
            _domain_prefix=prefix,
            _registered_domains=registered_domains,
            _retained_lease=self._lease_cls,
            _submit_chip=record,
        )
        if entry is None:
            if [task._rank for task in tasks] != list(range(world)):
                raise ValueError("Prepared decode requires one task for every rank in order")
            self.misses += 1
            if len(self.entries) == 4:
                self.entries.popitem(last=False)
            self.entries[key] = (
                roots,
                tuple(_owner_records(roots)),
                tasks,
                tuple((name, handle) for name, handle in domains.items() if name.startswith(prefix)),
            )
        else:
            self.hits += 1
            self.entries.move_to_end(key)
        keepalive.extend(tasks)
        if self.trace is None:
            for task in tasks:
                task.submit(orch, config)
        else:
            stamps: list[tuple[int, int]] = []
            for task in tasks:
                begin = perf_counter_ns()
                task.submit(orch, config)
                stamps.append((begin, perf_counter_ns()))
            self.trace.append({"hit": entry is not None, "rank_submit_ns": stamps})
