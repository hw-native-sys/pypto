# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Borrowed-context sessions using the public Simpler Worker lifecycle.

Configuration and capability checks live beside the session that consumes them.
Process admission, registration caching and pending ownership belong to context.py;
Simpler owns execution, callable slots and native run resources.
"""

import importlib
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pypto.runtime.runtime_pin import check_runtime_pin

from .arguments import SourcePool
from .owner import _OwnerThread


@dataclass(frozen=True)
class KernelConfig:
    """Immutable process configuration; no callable or per-call stream is stored."""

    platform: str
    runtime: str
    device_id: int
    aicpu_thread_num: int = 0
    enable_chip_swimlane: int | bool = 0
    enable_dep_gen: bool = False
    output_dir: str | Path | None = None
    # Ring arena sizing is fixed at initialization and reused by later calls.
    ring_task_window: int | Sequence[int] | None = None
    ring_heap: int | Sequence[int] | None = None
    ring_dep_pool: int | Sequence[int] | None = None

    workspace_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        from pypto.runtime.runner import _normalize_swimlane_level  # noqa: PLC0415

        if self.platform != "a2a3" or self.runtime not in ("tensormap_and_ringbuffer", "host_build_graph"):
            raise ValueError(
                f"Kernel initialization supports a2a3 TMR/HBG, got {self.platform}/{self.runtime}"
            )
        object.__setattr__(
            self,
            "enable_chip_swimlane",
            _normalize_swimlane_level(self.enable_chip_swimlane, "enable_chip_swimlane"),
        )
        if type(self.enable_dep_gen) is not bool:
            raise TypeError(f"enable_dep_gen must be bool, got {self.enable_dep_gen!r}")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        if (self.enable_chip_swimlane or self.enable_dep_gen) and self.output_dir is None:
            raise ValueError("Kernel DFX requires output_dir when swimlane or dep_gen is enabled")
        if type(self.device_id) is not int or self.device_id < 0:
            raise ValueError(f"Expected a nonnegative kernel device id, got {self.device_id!r}")
        if type(self.aicpu_thread_num) is not int or self.aicpu_thread_num not in (0, 2, 3, 4, 5):
            raise ValueError(f"Expected auto (0) or 2..5 AICPU threads, got {self.aicpu_thread_num!r}")
        if self.workspace_budget_bytes is not None and (
            type(self.workspace_budget_bytes) is not int or self.workspace_budget_bytes < 1024
        ):
            raise ValueError("workspace_budget_bytes must be an integer of at least 1024 bytes")
        for field in ("ring_task_window", "ring_heap", "ring_dep_pool"):
            value = getattr(self, field)
            if value is None:
                continue
            if type(value) is int:
                value = (value,) * 4
            else:
                value = tuple(value)
                if len(value) != 4:
                    raise ValueError(
                        f"{field} must be a scalar or exactly 4 per-ring values, got {len(value)}"
                    )
            if any(type(v) is not int or v <= 0 for v in value):
                raise ValueError(f"{field} entries must be positive ints, got {value}")
            object.__setattr__(self, field, value)


def _require_method(owner: Any, name: str, label: str) -> None:
    if not callable(getattr(owner, name, None)):
        raise NotImplementedError(f"Torch kernel runtime requires {label}.{name}")


def require_runtime() -> Any:
    """Preserve the shared pin guard, then check the installed public interface.

    This checks interface availability, not device execution. Native identity is
    checked by check_runtime_pin under its existing missing/skip rules.
    """
    check_runtime_pin()
    simpler = importlib.import_module("simpler")
    interface = importlib.import_module("simpler.task_interface")
    workers = importlib.import_module("simpler.worker")
    worker = getattr(simpler, "Worker", None)
    for name in ("init", "register", "unregister", "submit", "release_buffer", "close", "flush_diagnostics"):
        _require_method(worker, name, "Worker")
    for method, keyword in (("init", "borrowed_context_generation"), ("submit", "caller_stream")):
        if keyword not in inspect.signature(getattr(worker, method)).parameters:
            raise NotImplementedError(f"Torch kernel runtime requires Worker.{method}(..., {keyword}=...)")
    if not hasattr(worker, "device_location") or not hasattr(getattr(workers, "RunHandle", None), "done"):
        raise NotImplementedError("Torch kernel runtime requires Worker.device_location and RunHandle.done")
    for owner, label, methods in (
        (getattr(workers, "RunHandle", None), "RunHandle", ("wait",)),
        (getattr(interface, "Buffer", None), "Buffer", ("wrap", "close")),
        (getattr(interface, "TaskArgs", None), "TaskArgs", ("add_tensor", "add_scalar")),
        (getattr(interface, "CallConfig", None), "CallConfig", ("validate",)),
    ):
        for method in methods:
            _require_method(owner, method, label)
    if not callable(getattr(interface, "Tensor", None)):
        raise NotImplementedError("Torch kernel runtime requires Tensor views")
    return SimpleNamespace(Worker=worker, CallConfig=interface.CallConfig)


def make_call_config(config: KernelConfig, factory: Any) -> Any:
    """Validate the shared runtime config before native initialization claims a mode."""
    cfg = factory()
    cfg.aicpu_thread_num = config.aicpu_thread_num
    cfg.enable_chip_swimlane = config.enable_chip_swimlane
    cfg.enable_dep_gen = config.enable_dep_gen
    for field in ("ring_task_window", "ring_heap", "ring_dep_pool"):
        value = getattr(config, field)
        if value is not None:
            setattr(cfg.runtime_env, field, list(value))
    if config.output_dir is not None:
        cfg.output_prefix = str(config.output_dir)
    cfg.validate()
    return cfg


class _BorrowedWorkerSession:
    """Bind lifecycle operations to Torch's context on the owner thread.

    The wrapped object is a regular public simpler.Worker. This session only
    coordinates its borrowed context and lifetime with the framework.
    """

    def __init__(self, config: KernelConfig, state: Any) -> None:
        from pypto.torch.shutdown import install_shutdown  # noqa: PLC0415

        self.sources = SourcePool(lambda buffer: self.worker.release_buffer(buffer))
        self._api = require_runtime()
        self._native = install_shutdown(state)
        self._context = self._native.borrow_context(config.device_id)
        self._device_id = config.device_id
        self._generation = state.generation
        self.worker: Any = None
        self._owner = _OwnerThread()

    def check_context(self) -> None:
        if self._native.borrow_context(self._device_id) != self._context:
            raise RuntimeError("Kernel Worker belongs to a different borrowed context; use a fresh process")

    def on_owner_thread(self) -> bool:
        import threading  # noqa: PLC0415

        return threading.current_thread() is self._owner.thread

    def init(self, config: KernelConfig) -> None:
        self._owner.call(lambda: self._init(config))

    def _init(self, config: KernelConfig) -> None:
        self._native.bind_context(self._context)
        cfg = make_call_config(config, self._api.CallConfig)
        options = {}
        if config.workspace_budget_bytes is not None:
            options["workspace_budget_bytes"] = config.workspace_budget_bytes
        self.worker = self._api.Worker(
            level=2,
            platform=config.platform,
            runtime=config.runtime,
            device_id=config.device_id,
            enable_sdma=False,
            collect_across_runs=bool(config.enable_chip_swimlane or config.enable_dep_gen),
            **options,
        )
        self.worker.init(prewarm_config=cfg, borrowed_context_generation=self._generation)

    def prepare(self, callable_: Any) -> Any:
        def register() -> Any:
            self._native.bind_context(self._context)
            return self.worker.register(callable_)

        return self._owner.call(register)

    @property
    def device_location(self) -> Any:
        return self.worker.device_location

    def wait(self, ticket: Any) -> None:
        def retire() -> None:
            self._native.bind_context(self._context)
            ticket.wait()

        self._owner.call(retire)

    def close(self) -> None:
        def finalize() -> None:
            self._native.bind_context(self._context)
            self.sources.close()
            if self.worker is not None:
                self.worker.close()

        self._owner.call(finalize)
        self._owner.stop()


__all__ = ["_BorrowedWorkerSession"]
