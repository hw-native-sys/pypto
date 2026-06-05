# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L2 :class:`ChipWorker` — the single-chip concrete runtime handle.

Inside a ``with ChipWorker(...) as _:`` block, calls to ``CompiledProgram(...)``
(and :func:`pypto.runtime.run`) reuse the active worker instead of creating a
fresh one. Outside such a block, behavior is unchanged from one-shot
construction in :func:`pypto.runtime.device_runner.execute_on_device`.

For explicit dispatch (no ``ContextVar`` discovery), call
:meth:`ChipWorker.run` directly, or pre-register with :meth:`ChipWorker.register`
and call the returned :class:`RegistrationHandle`.

Example — implicit reuse::

    from pypto.runtime import ChipWorker, RunConfig

    with ChipWorker(config=RunConfig(platform="a2a3")):
        out1 = Add(*tensors1)   # uses active ChipWorker
        out2 = Mul(*tensors2)   # reuses same ChipWorker
    # close() runs once on exit

Example — explicit dispatch::

    w = ChipWorker(config=RunConfig(platform="a2a3"))
    try:
        out = w.run(compiled_add, a, b)
        h = w.register(compiled_mul)        # pre-register hot path
        for _ in range(1000):
            h(a, b, out)
    finally:
        w.close()
"""

from __future__ import annotations

import contextvars
import weakref
from typing import TYPE_CHECKING, Any

from .runner import RunConfig
from .runtime_base import Worker

if TYPE_CHECKING:
    from pypto.ir.compiled_program import CallArg, CompiledProgram

# ``simpler`` is loaded lazily on first ``ChipWorker(...)`` instantiation,
# matching the pattern used by ``device_runner.py`` (imported via lazy
# ``from .device_runner import ...`` inside function bodies). Eager loading
# would make ``simpler`` a hard import-time dependency of ``pypto.runtime`` and
# break unit-test environments that do not install simpler.
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


# Stack of active ChipWorkers (most-recent last). ContextVar gives correct
# scoping under nested ``with`` blocks and ``asyncio`` tasks.
_ACTIVE_WORKERS: contextvars.ContextVar[tuple[ChipWorker, ...]] = contextvars.ContextVar(
    "_pypto_active_workers", default=()
)

# Default runtime name — matches ``compile_and_assemble``'s fallback in
# ``device_runner.py`` and the most common user-program runtime.
_DEFAULT_RUNTIME = "host_build_graph"


class ChipWorker(Worker):
    """L2 single-chip execution handle, bound to one ``(platform, device_id, runtime)``.

    A ``ChipWorker`` auto-initializes device state in ``__init__`` so that an
    immediate ``with chipworker:`` block can dispatch runs without further
    setup. Construction without entering a ``with`` block also works — call
    :meth:`close` manually when done, or re-enter via ``with`` later.

    Inside a ``with`` block, ``CompiledProgram.__call__`` and
    :func:`pypto.runtime.run` find this worker via a ``ContextVar`` and reuse
    its initialized device context instead of creating a fresh worker per call.
    Reuse only happens when all four binding fields match — otherwise the
    caller falls through to the one-shot path.

    .. note::
       Distinct from ``simpler.worker.ChipWorker`` (the C++ L2 backend handle
       that this class wraps internally via ``self._impl``). pypto users
       interact only with this class; the simpler C++ name is not re-exported
       through ``pypto.runtime.task_interface``.

    Args:
        config: Run configuration providing ``platform`` and ``device_id``.
            Defaults to :class:`RunConfig` defaults.
        level: Hierarchy level. Only ``2`` (single-chip) is currently
            supported on ``ChipWorker``; pass ``level=2`` explicitly or rely
            on the default. L3+ goes through
            :class:`~pypto.runtime.distributed_runner.DistributedWorker`.
        runtime: Runtime implementation name. Must match the runtime the
            program is compiled against; otherwise reuse silently falls
            through to the one-shot path. Defaults to ``"host_build_graph"``.
        auto_init: If ``True``, call :meth:`init` from ``__init__``. Default
            is ``True``.
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
                f"ChipWorker only supports level=2; got level={level}. "
                f"L3+ runtimes go through DistributedWorker."
            )

        super().__init__()  # initialize Worker ABC state (_owned_tensors)

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
        # Maps id(chip_callable) -> cid returned by simpler Worker.register().
        # Simpler's L2 ABI now requires every ChipCallable to be registered
        # before dispatch (see runtime PR #710); we cache per-callable cids
        # so repeated runs of the same compiled program inside one
        # `with ChipWorker:` block re-use the same registration.
        self._cid_cache: dict[int, int] = {}
        # Live RegistrationHandles, so close() can mark them closed
        # synchronously. Weak refs so handles GC'd before close() don't
        # keep the dict alive forever.
        self._handles: weakref.WeakSet[RegistrationHandle] = weakref.WeakSet()

        if auto_init is None:
            auto_init = True
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
        """Release device state. Idempotent. The ChipWorker may be re-``init()``'d."""
        if not self._initialized:
            return
        # Auto-free any DeviceTensors the caller forgot. Run BEFORE we drop
        # cid registrations / tear down the impl so the underlying free path
        # is still live.
        self._close_owned_tensors()
        # Drop per-cid host-side state before tearing down the device so
        # the underlying ChipWorker.finalize() doesn't observe stale
        # registrations on a re-init().
        for cid in self._cid_cache.values():
            self._impl.unregister_callable(cid)
        self._cid_cache.clear()
        # Mark every still-alive RegistrationHandle as closed so subsequent
        # handle(...) calls raise instead of silently dispatching to a
        # released cid.
        for handle in list(self._handles):
            handle._mark_closed()
        self._handles.clear()
        self._impl.close()
        self._initialized = False

    # ------------------------------------------------------------------
    # Device memory primitives (forwarded to the underlying chip worker)
    #
    # All methods require an active init(). ``worker_id`` is kept as a keyword
    # for forward compatibility with L3, even though ChipWorker currently only
    # supports level=2 with worker_id=0.
    # ------------------------------------------------------------------

    def _require_initialized(self, op: str) -> None:
        if not self._initialized:
            raise RuntimeError(
                f"ChipWorker.{op}() requires an initialized ChipWorker. "
                f"Use `with chipworker:` or call `chipworker.init()` first."
            )

    def _require_ready(self, op: str) -> None:
        # Worker ABC hook: device-memory ops need an initialized ChipWorker.
        self._require_initialized(op)

    def malloc(self, nbytes: int, *, worker_id: int = 0) -> int:
        """Allocate ``nbytes`` of device memory; returns an opaque pointer.

        The returned pointer lives in *worker_id*'s address space.  Pair every
        ``malloc()`` with a matching :meth:`free` before this ChipWorker is
        closed, otherwise the device memory is leaked.
        """
        self._require_initialized("malloc")
        if not isinstance(nbytes, int) or nbytes <= 0:
            raise ValueError(f"nbytes must be a positive int, got {nbytes!r}")
        return self._impl.malloc(nbytes, worker_id)

    def free(self, ptr: int, *, worker_id: int = 0) -> None:
        """Release a pointer previously returned by :meth:`malloc`."""
        self._require_initialized("free")
        self._impl.free(ptr, worker_id)

    def copy_to(
        self,
        dst_dev_ptr: int,
        src_host_ptr: int,
        nbytes: int,
        *,
        worker_id: int = 0,
    ) -> None:
        """H2D copy: ``nbytes`` bytes from host *src_host_ptr* to device *dst_dev_ptr*.

        *src_host_ptr* is typically obtained from ``host_tensor.data_ptr()``;
        the caller is responsible for keeping the host tensor alive until
        this call returns.
        """
        self._require_initialized("copy_to")
        self._impl.copy_to(dst_dev_ptr, src_host_ptr, nbytes, worker_id)

    def copy_from(
        self,
        dst_host_ptr: int,
        src_dev_ptr: int,
        nbytes: int,
        *,
        worker_id: int = 0,
    ) -> None:
        """D2H copy: ``nbytes`` bytes from device *src_dev_ptr* back to host *dst_host_ptr*."""
        self._require_initialized("copy_from")
        self._impl.copy_from(dst_host_ptr, src_dev_ptr, nbytes, worker_id)

    # ``alloc_tensor`` / ``free_tensor`` are inherited from Worker (ABC).
    # L2 uses the default ``_prepare_init`` (a defensive contiguous CPU copy);
    # only ``_require_ready`` is overridden above to require an initialized
    # ChipWorker.

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
    # Diagnostic counters — direct passthrough to simpler.
    # ------------------------------------------------------------------

    @property
    def aicpu_dlopen_count(self) -> int:
        """Distinct cids the AICPU has dlopened for.

        Useful in tests to verify that ``register`` + repeated ``run`` of the
        same callable does NOT retrigger the AICPU dlopen.
        """
        return self._impl.aicpu_dlopen_count

    @property
    def host_dlopen_count(self) -> int:
        """Host-side orch SO dlopens (host_build_graph variant)."""
        return self._impl.host_dlopen_count

    # ------------------------------------------------------------------
    # Active-Worker discovery (mirrors PassContext.Current pattern)
    # ------------------------------------------------------------------

    @classmethod
    def current(cls, *, level: int, platform: str, device_id: int, runtime: str) -> ChipWorker | None:
        """Return the topmost active ChipWorker matching the binding, or ``None``.

        Used by :func:`pypto.runtime.device_runner.execute_on_device` to
        decide whether to reuse a user-published ChipWorker or fall through
        to constructing a fresh one-shot worker.
        """
        target = (level, platform, device_id, runtime)
        for w in reversed(_ACTIVE_WORKERS.get()):
            if w._binding == target:
                return w
        return None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _check_binding(self, compiled: CompiledProgram) -> None:
        """Raise ValueError on platform / runtime_name mismatch.

        ``compiled.runtime_name`` triggers ``compile_and_assemble`` lazily,
        which is acceptable because any subsequent dispatch needs it anyway.
        """
        if compiled.platform != self.platform:
            raise ValueError(
                f"CompiledProgram.platform={compiled.platform!r} does not match "
                f"ChipWorker.platform={self.platform!r}. Compile for the matching "
                f"platform, or construct ChipWorker(config=RunConfig(platform={compiled.platform!r}))."
            )
        if compiled.runtime_name != self._runtime:
            raise ValueError(
                f"CompiledProgram requires runtime={compiled.runtime_name!r} but "
                f"ChipWorker was constructed with runtime={self._runtime!r}. "
                f"Construct ChipWorker(..., runtime={compiled.runtime_name!r})."
            )

    def run(
        self,
        compiled: CompiledProgram,
        *args: CallArg,
        config: RunConfig | None = None,
    ) -> Any:
        """Dispatch *compiled* on this ChipWorker explicitly.

        Equivalent to ``compiled(*args, config=config)`` running under
        ``with chipworker:``, but the binding (platform / runtime_name) is
        checked against this ChipWorker up-front rather than relying on
        ``ContextVar`` discovery. Falls through to the same internal
        ``_run_chip`` path, so the cid cache is shared with the implicit path.

        Returns the same shape as ``compiled(...)``: ``None`` for in-place
        calls, a single ``torch.Tensor`` for one-output return-style calls,
        or a tuple of tensors otherwise.

        Raises:
            ValueError: ``compiled.platform`` != ``self.platform`` or
                ``compiled.runtime_name`` != ``self.runtime``.
            RuntimeError: ChipWorker not initialized.
        """
        self._require_initialized("run")
        self._check_binding(compiled)

        # Import lazily to avoid a cycle: compiled_program imports from
        # pypto.runtime.runner which imports worker for ChipWorker.current.
        from pathlib import Path  # noqa: PLC0415

        rc = config if config is not None else RunConfig()

        dfx_dir: Path | None = None
        if rc.any_dfx_enabled():
            dfx_dir = Path(compiled.output_dir) / "dfx_outputs"
            dfx_dir.mkdir(parents=True, exist_ok=True)

        orch_args, coerced, return_style = compiled.build_orch_args(*args)
        cfg = compiled.build_call_config(rc, dfx_dir=dfx_dir)
        self._run_chip(compiled.chip_callable, orch_args, cfg)

        if dfx_dir is not None:
            from .runner import _collect_dfx_artifacts, _DfxOpts  # noqa: PLC0415

            _collect_dfx_artifacts(dfx_dir, self.platform, _DfxOpts.from_run_config(rc))

        if not return_style:
            return None
        outputs = [coerced[i] for i in compiled.output_indices]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def register(self, compiled: CompiledProgram) -> RegistrationHandle:
        """Pre-register *compiled* on this ChipWorker. Returns a callable handle.

        Eager registration: triggers ``compile_and_assemble`` on *compiled*
        and ``simpler.Worker.register`` immediately, so configuration errors
        surface here rather than at first dispatch. The handle reuses the
        ChipWorker's existing cid cache (multiple ``register`` calls for the
        same *compiled* return aliases of the same cid).

        Raises:
            ValueError: Binding mismatch (see :meth:`run`).
            RuntimeError: ChipWorker not initialized.
        """
        self._require_initialized("register")
        self._check_binding(compiled)
        cc = compiled.chip_callable  # triggers compile_and_assemble lazily
        key = id(cc)
        cid = self._cid_cache.get(key)
        if cid is None:
            cid = self._impl.register(cc)
            self._cid_cache[key] = cid
        handle = RegistrationHandle(self, compiled, cid)
        self._handles.add(handle)
        return handle

    # ------------------------------------------------------------------
    # Internal hook for the runner reuse path
    # ------------------------------------------------------------------

    def _run_chip(self, chip_callable: Any, orch_args: Any, cfg: Any) -> Any:
        """Dispatch *chip_callable* and return the simpler ``RunTiming``.

        Returns the ``RunTiming`` produced by the underlying simpler
        ``Worker.run`` (host + device wall). :meth:`run` ignores it — it
        returns tensor outputs instead — but :func:`execute_on_device`
        surfaces it on the ChipWorker-reuse path.
        """
        if not self._initialized:
            raise RuntimeError("ChipWorker is not initialized; call init() or use `with chipworker:`")
        key = id(chip_callable)
        cid = self._cid_cache.get(key)
        if cid is None:
            cid = self._impl.register(chip_callable)
            self._cid_cache[key] = cid
        return self._impl.run(cid, orch_args, cfg)

    # ------------------------------------------------------------------
    # Context manager — publishes ``self`` on the active stack
    # ------------------------------------------------------------------

    def __enter__(self) -> ChipWorker:
        stack = _ACTIVE_WORKERS.get()
        if any(w._binding == self._binding for w in stack):
            level, platform, device_id, runtime = self._binding
            raise ValueError(
                f"A ChipWorker for (level={level}, platform={platform!r}, "
                f"device_id={device_id}, runtime={runtime!r}) is already "
                f"active in an enclosing scope. Reuse the outer ChipWorker instead of nesting "
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


class RegistrationHandle:
    """Bound dispatcher for one ``(Worker, compiled)`` pair.

    Returned by :meth:`Worker.register` (implemented on both
    :class:`ChipWorker` and
    :class:`~pypto.runtime.distributed_runner.DistributedWorker`).

    Three usage styles, all equivalent in steady state:

    Direct call (most common)::

        h = worker.register(compiled)
        h(a, b)
        h.unregister()

    Context manager (auto-release on scope exit)::

        with worker.register(compiled) as h:
            h(a, b)

    Manual control over cid lifetime (rare; cf. benchmarks)::

        h = worker.register(compiled)
        for _ in range(1000):
            h(a, b)
        # h.unregister() optional — Worker.close() releases everything

    Calling a handle after :meth:`unregister` or after the parent
    ``Worker.close()`` raises ``RuntimeError``.

    **cid reuse semantics (L2):** Multiple :meth:`ChipWorker.register` calls
    for the same ``compiled.chip_callable`` return aliases of the same
    underlying cid. :meth:`unregister` only marks the handle closed; it does
    NOT call ``simpler.unregister_callable``. Real cid release happens once,
    in :meth:`Worker.close`. ``cid`` is informational only.

    **L3 note:** ``DistributedWorker`` doesn't expose a per-callable cid the
    way ChipWorker does (its chip / sub registrations are baked at prepare()
    time). For L3 handles, ``cid`` is ``0``; the dispatch path still routes
    through the orchestrator via ``DistributedWorker.run``.
    """

    __test__ = False  # Not a pytest test class

    def __init__(
        self,
        worker: Worker,
        compiled: Any,
        cid: int,
    ) -> None:
        # Strong ref to the worker so the handle stays usable across the
        # parent Worker's scope; the worker tracks the handle weakly so this
        # strong ref doesn't outlive close().
        self._worker = worker
        self._compiled = compiled
        self._cid = cid
        self._closed = False

    @property
    def cid(self) -> int:
        return self._cid

    @property
    def compiled(self) -> Any:
        return self._compiled

    @property
    def closed(self) -> bool:
        return self._closed

    def __call__(self, *args: Any, config: RunConfig | None = None) -> Any:
        """Dispatch the bound compiled program on the bound worker.

        Delegates to :meth:`Worker.run`. Same return contract.
        """
        if self._closed:
            raise RuntimeError(
                "RegistrationHandle has been unregistered (or its parent Worker was closed). "
                "Re-register via worker.register(compiled) to get a fresh handle."
            )
        return self._worker.run(self._compiled, *args, config=config)

    def unregister(self) -> None:
        """Mark this handle closed. Idempotent.

        Does NOT call ``simpler.unregister_callable`` — other handle aliases
        for the same cid would silently break. The real reverse-registration
        happens once, in :meth:`Worker.close`.
        """
        self._closed = True

    def _mark_closed(self) -> None:
        """Internal: called by Worker.close() to invalidate the handle."""
        self._closed = True

    def __enter__(self) -> RegistrationHandle:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.unregister()
