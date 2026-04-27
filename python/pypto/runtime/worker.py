# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""User-facing :class:`Worker` that amortizes init/close across multiple runs.

Inside a ``with Worker(...) as _:`` block, calls to ``CompiledProgram(...)``
(and :func:`pypto.runtime.run`) reuse the active Worker instead of creating a
fresh one. Outside such a block, behavior is unchanged from one-shot
construction in :func:`pypto.runtime.device_runner.execute_on_device`.

Example::

    from pypto.runtime import Worker, RunConfig

    with Worker(config=RunConfig(platform="a2a3")):
        out1 = Add(*tensors1)   # uses active Worker
        out2 = Mul(*tensors2)   # reuses same Worker
    # close() runs once on exit
"""

from __future__ import annotations

import contextvars
from typing import Any

from .runner import RunConfig

# ``simpler`` is loaded lazily on first ``Worker(...)`` instantiation, matching
# the pattern used by ``device_runner.py`` (imported via lazy ``from .device_runner
# import ...`` inside function bodies). Eager loading would make ``simpler`` a
# hard import-time dependency of ``pypto.runtime`` and break unit-test
# environments that do not install simpler.
_SimplerWorker: type | None = None


def _get_simpler_worker_cls() -> type:
    global _SimplerWorker  # noqa: PLW0603 - module-level cache that tests patch directly
    if _SimplerWorker is None:
        from .task_interface import (  # noqa: PLC0415
            Worker as _W,  # pyright: ignore[reportAttributeAccessIssue]
        )

        _SimplerWorker = _W
    assert _SimplerWorker is not None
    return _SimplerWorker


# Stack of active workers (most-recent last). ContextVar gives correct
# scoping under nested ``with`` blocks and ``asyncio`` tasks.
_ACTIVE_WORKERS: contextvars.ContextVar[tuple[Worker, ...]] = contextvars.ContextVar(
    "_pypto_active_workers", default=()
)

# Default runtime name — matches ``compile_and_assemble``'s fallback in
# ``device_runner.py`` and the most common user-program runtime.
_DEFAULT_RUNTIME = "host_build_graph"


class Worker:
    """Reusable execution Worker bound to one ``(level, platform, device_id, runtime)``.

    A ``Worker`` constructed with ``level=2`` auto-initializes device state in
    ``__init__`` so that an immediate ``with worker:`` block can dispatch runs
    without further setup. Construction without entering a ``with`` block also
    works — call ``close()`` manually when done, or re-enter via ``with`` later.

    Inside a ``with`` block, ``CompiledProgram.__call__`` and
    :func:`pypto.runtime.run` find this Worker via a ``ContextVar`` and reuse
    its initialized device context instead of creating a fresh Worker per call.
    Reuse only happens when all four binding fields match — otherwise the
    caller falls through to the one-shot path.

    Args:
        config: Run configuration providing ``platform`` and ``device_id``.
            Defaults to :class:`RunConfig` defaults.
        level: Hierarchy level. Only ``2`` (single-chip) is currently
            supported at the pypto user-API layer.
        runtime: Runtime implementation name. Must match the runtime the
            program is compiled against; otherwise reuse silently falls
            through to the one-shot path. Defaults to ``"host_build_graph"``.
        auto_init: If ``True``, call :meth:`init` from ``__init__``. Default
            is ``True`` for ``level=2`` and ``False`` otherwise (level 3+
            requires ``register()`` calls before ``init()``, but is not
            currently supported anyway).
    """

    def __init__(
        self,
        config: RunConfig | None = None,
        *,
        level: int = 2,
        runtime: str = _DEFAULT_RUNTIME,
        auto_init: bool | None = None,
    ) -> None:
        if level != 2:
            raise ValueError(
                f"pypto.runtime.Worker currently only supports level=2; got level={level}. "
                f"L3 (multi-chip / DistWorker) is not yet exposed at the pypto user-API layer."
            )

        self._config = config or RunConfig()
        self._level = level
        self._runtime = runtime
        self._token: contextvars.Token | None = None

        self._impl = _get_simpler_worker_cls()(
            level=level,
            device_id=self._config.device_id,
            platform=self._config.platform,
            runtime=runtime,
        )
        self._initialized = False

        if auto_init is None:
            auto_init = level == 2
        if auto_init:
            self.init()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Initialize device state. Idempotent — a second call is a no-op."""
        if self._initialized:
            return
        self._impl.init()
        self._initialized = True

    def close(self) -> None:
        """Release device state. Idempotent. The Worker may be re-``init()``'d."""
        if not self._initialized:
            return
        self._impl.close()
        self._initialized = False

    # ------------------------------------------------------------------
    # Binding accessors
    # ------------------------------------------------------------------

    @property
    def level(self) -> int:
        return self._level

    @property
    def platform(self) -> str:
        return self._config.platform

    @property
    def device_id(self) -> int:
        return self._config.device_id

    @property
    def runtime(self) -> str:
        return self._runtime

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def _binding(self) -> tuple[int, str, int, str]:
        return (self._level, self._config.platform, self._config.device_id, self._runtime)

    # ------------------------------------------------------------------
    # Active-Worker discovery (mirrors PassContext.Current pattern)
    # ------------------------------------------------------------------

    @classmethod
    def current(cls, *, level: int, platform: str, device_id: int, runtime: str) -> Worker | None:
        """Return the topmost active Worker matching the binding, or ``None``.

        Used by :func:`pypto.runtime.device_runner.execute_on_device` to
        decide whether to reuse a user-published Worker or fall through to
        constructing a fresh one-shot Worker.
        """
        target = (level, platform, device_id, runtime)
        for w in reversed(_ACTIVE_WORKERS.get()):
            if w._binding == target:
                return w
        return None

    # ------------------------------------------------------------------
    # Internal hook for the runner reuse path
    # ------------------------------------------------------------------

    def _run_chip(self, chip_callable: Any, orch_args: Any, cfg: Any) -> None:
        if not self._initialized:
            raise RuntimeError("Worker is not initialized; call init() or use `with worker:`")
        self._impl.run(chip_callable, orch_args, cfg)

    # ------------------------------------------------------------------
    # Context manager — publishes ``self`` on the active stack
    # ------------------------------------------------------------------

    def __enter__(self) -> Worker:
        stack = _ACTIVE_WORKERS.get()
        if any(w._binding == self._binding for w in stack):
            level, platform, device_id, runtime = self._binding
            raise ValueError(
                f"A Worker for (level={level}, platform={platform!r}, "
                f"device_id={device_id}, runtime={runtime!r}) is already "
                f"active in an enclosing scope. Reuse the outer Worker instead of nesting "
                f"a second one with identical binding."
            )
        if not self._initialized:
            self.init()
        self._token = _ACTIVE_WORKERS.set(stack + (self,))
        return self

    def __exit__(self, *_exc: Any) -> None:
        assert self._token is not None
        _ACTIVE_WORKERS.reset(self._token)
        self._token = None
        self.close()
