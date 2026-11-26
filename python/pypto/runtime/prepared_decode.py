# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cache fixed decode rank bindings without caching submission authorization."""

import ast
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns

import torch

from .device_tensor import DeviceTensor, StackedDeviceTensor

# Exact orchestration sources audited for static domain specs, rank-only graph
# construction, no scalar public args, and no cross-rank TaskHandle references.
HOST_SOURCES = {
    "e4b07c15b7b6007f070bd1a9f03986436bc116ebb75695a708a703e7fbfdb380",  # wave13-p3a-head8-aligned-r3 one
    "b5b110960fee9fdea16908d1d7d3d810b27f5efc0a515ef43290e10d9d64ddfd",  # wave13-p3a-head8-aligned-r3 two
    "fcd9000ce64441a472ee4b06c1e60c5466c58787fc662d25f51461bb457102ef",  # wave13-p3b-drafter-tp8-r1 one
    "9fcbd4ace0c430420cf731405df386c434646d5029ca45064107b6e7d2e09028",  # wave13-p3b-drafter-tp8-r1 two
}


def _domain_scope(orch, registered, prefix, name, factory):
    namespaced = prefix + name
    previous = registered.get(name)
    if previous is None:
        return factory()
    from .distributed_runner import _RetainedDomainLease

    handle = orch.get_domain(namespaced)
    if handle is not previous[1] or not handle.retain_after_run:
        raise RuntimeError("Prepared decode requires a retained domain")
    return _RetainedDomainLease(handle)


def _host_key(value):
    return (id(value), value.data_ptr(), value.untyped_storage()._cdata, tuple(value.shape),
            value.stride(), value.dtype, value.is_shared())


def _binding_key(value):
    if type(value) in (DeviceTensor, StackedDeviceTensor):
        return id(value)
    if type(value) is torch.Tensor:
        return _host_key(value)
    if type(value) is list and all(type(item) is torch.Tensor for item in value):
        return tuple(map(_host_key, value))
    raise TypeError("Unsupported prepared decode binding")


def _owner_records(args):
    for value in args:
        if type(value) is StackedDeviceTensor:
            for shard, rank in zip(value.shards, value.worker_ids, strict=True):
                yield (rank, shard.data_ptr - shard.byte_offset), shard.buffer
        elif type(value) is DeviceTensor:
            yield (value.buffer.owner_worker_id, value.data_ptr - value.byte_offset), value.buffer


class PrepareRanks(ast.NodeTransformer):
    def visit_For(self, node):
        # The pinned entry contains only independent rank loops. The caller
        # executes each entire loop once on a cache miss to create snapshots.
        return ast.copy_location(ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="_cache_hit", ctx=ast.Load())),
            body=[self.generic_visit(node)], orelse=[],
        ), node)

    def visit_With(self, node):
        (item,) = node.items
        call = item.context_expr
        name = next(ast.literal_eval(k.value) for k in call.keywords if k.arg == "name")
        if any(isinstance(n, ast.Name) and n.id == "tensors" for n in ast.walk(call)):
            raise ValueError("Dynamic domain definitions need original admission")
        item.context_expr = ast.Call(
            func=ast.Name(id="_domain_scope", ctx=ast.Load()),
            args=[ast.Name(id="orch", ctx=ast.Load()), ast.Name(id="_registered_domains", ctx=ast.Load()),
                  ast.Name(id="_domain_prefix", ctx=ast.Load()), ast.Constant(name),
                  ast.parse("lambda: None", mode="eval").body], keywords=[],
        )
        item.context_expr.args[-1].body = call
        node.body = [self.visit(stmt) for stmt in node.body]
        return node


def load(entry, path):
    if not isinstance(path, Path) or not path.is_file():
        return None
    source = path.read_bytes()
    if not isinstance(source, (bytes, bytearray)) or sha256(source).hexdigest() not in HOST_SOURCES:
        return None
    try:
        from simpler.prepared_local import PreparedLocalChipTask
    except ImportError:
        return None
    tree = ast.parse(source, filename=str(path))
    fn = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == entry.__name__)
    fn.args.kwonlyargs.extend(
        ast.arg(arg=name) for name in ("_cache_hit", "_domain_prefix", "_registered_domains", "_submit_chip")
    )
    fn.args.kw_defaults.extend((None, None, None, None))
    PrepareRanks().visit(fn)
    namespace = {**entry.__globals__, "_domain_scope": _domain_scope}
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return PreparedDecode(namespace[entry.__name__], task_cls=PreparedLocalChipTask)


class PreparedDecode:
    def __init__(self, prepare, *, task_cls):
        self.prepare = prepare
        self._task_cls = task_cls
        self.entries = OrderedDict()
        self.trace = None
        self.hits = 0
        self.misses = 0

    def select(self, worker, state, tensors, config, args, slot):
        if config is None or config.output_prefix or not worker._persistent or worker._reset_persistent_windows:
            return None
        try:
            key = (slot, tuple(map(_binding_key, tensors.values())))
        except TypeError:
            return None
        entry = self.entries.get(key)
        if entry is None or any(worker._device_buffers.get(key) is not owner for key, owner in entry[1]):
            # Go through the original public validators on any miss or revoke.
            for info, arg in zip(state["param_infos"], args, strict=True):
                worker._validate_prepared_dispatch_arg(info, arg)
            entry = None
        return key, entry

    def submit(self, selection, orch, config, *, tensors, callables, sub_ids, keepalive, world, provider, program, registered_domains):
        key, entry = selection
        tasks = [] if entry is None else entry[2]
        domains = orch._worker._live_domains
        prefix = f"{program}:"
        current = tuple((name, handle) for name, handle in domains.items() if name.startswith(prefix))
        if entry is not None and current != entry[3]:
            tasks = []
            entry = None

        def record(orch, handle, args, _config, rank):
            tasks.append(self._task_cls(orch, handle, args, rank))

        roots = tuple(tensors.values()) if entry is None else entry[0]
        self.prepare(orch, None, config, tensors=tensors, callables=callables, sub_ids=sub_ids,
                     _keep=keepalive, world_size=world, _domain_provider=provider, _cache_hit=entry is not None,
                     _domain_prefix=prefix, _registered_domains=registered_domains, _submit_chip=record)
        if entry is None:
            if [task._rank for task in tasks] != list(range(world)):
                raise ValueError("Prepared decode requires one task for every rank in order")
            self.misses += 1
            if len(self.entries) == 4:
                self.entries.popitem(last=False)
            self.entries[key] = (
                roots, tuple(_owner_records(roots)), tasks,
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
            stamps = []
            for task in tasks:
                begin = perf_counter_ns()
                task.submit(orch, config)
                stamps.append((begin, perf_counter_ns()))
            self.trace.append({"hit": entry is not None, "rank_submit_ns": stamps})
