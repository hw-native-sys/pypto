# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime-only node preparation, copied under the installed artifact namespace.

The parent owns compilation and cold metadata validation. This module binds
native generated entries to local runtime handles before the inner worker forks.
It deliberately has no dependency on the compiler or the PyPTO package.
"""

import importlib
from typing import Any

import torch

from ._distributed_support import _ChipCallables, _copy_return_tensor, _EmbeddedEntryState, _EmbeddedNodeState


def _tensor(dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    name = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(dtype, dtype)
    return torch.zeros(shape, dtype=getattr(torch, name)).share_memory_()


class _NodeBindings:
    """Own prefork storage until all runtime children have drained and closed."""

    def __init__(self) -> None:
        self.entries: list[tuple[Any, _EmbeddedEntryState]] = []
        self.node_state: _EmbeddedNodeState | None = None

    def task_completed(self) -> None:
        """Called by the generic command loop after a successful DAG fence."""
        for _, state in self.entries:
            state.complete_invocation()

    def close(self) -> None:
        """The caller closes its inner worker before releasing these references."""
        for entry, state in self.entries:
            state.close()
            if getattr(entry, "_pypto_embedded_state", None) is state:
                del entry._pypto_embedded_state
        self.entries.clear()
        if self.node_state is not None:
            self.node_state.close()
            self.node_state = None


def prepare_node(context: Any) -> _NodeBindings:
    """Bind all installed entries and one canonical prefork token pool."""
    entries = []
    for program in context.parameters["programs"]:
        prefix = program["module_prefix"]
        for record in program["entries"]:
            module = importlib.import_module(prefix + "." + record["module"])
            entry = getattr(module, record["qualname"])
            entries.append((record, entry, entry._pypto_embedded_layout, program))

    bindings = _NodeBindings()
    try:
        capacity = sum(
            row[1]
            for _, _, layout, _ in entries
            for row in layout["prefork_requirements"]
            if row[0] == "inherited_domain_tokens"
        )
        if capacity:
            count = entries[0][0]["local_device_count"]
            bindings.node_state = _EmbeddedNodeState(
                local_worker_ids=tuple(range(count)),
                prepared_domain_tokens=tuple(
                    tuple(_tensor("int32", (1,)) for _ in range(count)) for _ in range(capacity)
                ),
            )
        for record, entry, layout, parameters in entries:
            count = record["local_device_count"]
            prefix = parameters["module_prefix"]
            digests, configs = parameters["chip_digests"], parameters["chip_configs"]
            prepared: dict[str, Any] = {}
            for row in layout["prefork_requirements"]:
                if row[0] == "tensor_storage":
                    prepared[row[1]] = _tensor(row[2], row[3])
                elif row[0] == "local_domain_tokens":
                    prepared[row[1]] = tuple(_tensor("int32", (1,)) for _ in range(count))
            callables = _ChipCallables()
            for name, path in record["chip_paths"].items():
                callables.add(name, context.chip_handles[digests[path]], configs[path])
            sub_ids = {}
            for name, path in record["sub_worker_paths"].items():
                target = importlib.import_module(prefix + "." + path[:-3].replace("/", "."))
                sub_ids[name] = context.inner_worker.register(getattr(target, name))
            copying = (
                context.inner_worker.register(_copy_return_tensor) if record["return_copy_callable"] else None
            )
            state = _EmbeddedEntryState(
                layout,
                callables=callables,
                sub_ids=sub_ids,
                world_size=count,
                prepared_tensors=prepared,
                domain_provider=None,
                return_copy_callable=copying,
                node_state=bindings.node_state,
                domain_bindings=parameters["domain_bindings"][record["logical_path"]],
            )
            entry._pypto_embedded_state = state
            bindings.entries.append((entry, state))
    except BaseException:
        bindings.close()
        raise
    return bindings
