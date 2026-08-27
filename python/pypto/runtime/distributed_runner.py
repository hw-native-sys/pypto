# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute L3 distributed programs via simpler Worker(level=3)."""

from __future__ import annotations

import ctypes
import importlib.util
import inspect
import json
import logging
import math
import sys
import threading
import time
import types
import warnings
import weakref
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np  # pyright: ignore[reportMissingImports]
import torch

from .device_tensor import DeviceTensor, StackedDeviceTensor
from .runtime_base import Worker

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig

    from .runner import RunConfig
    from .worker import RegistrationHandle


# ---------------------------------------------------------------------------
# simpler Tensor → torch.Tensor conversion
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, tuple[type, torch.dtype]] = {
    "FLOAT32": (ctypes.c_float, torch.float32),
    "FLOAT16": (ctypes.c_uint8, torch.float16),
    "BFLOAT16": (ctypes.c_uint8, torch.bfloat16),
    "INT8": (ctypes.c_int8, torch.int8),
    "INT16": (ctypes.c_int16, torch.int16),
    "INT32": (ctypes.c_int32, torch.int32),
    "INT64": (ctypes.c_int64, torch.int64),
    "UINT8": (ctypes.c_uint8, torch.uint8),
}


def _resolve_persistent_window_reset(persistent: bool, reset_persistent_windows: bool | None) -> bool:
    """Resolve the retained-window reset policy.

    Args:
        persistent: Whether CommDomains are retained across dispatches.
        reset_persistent_windows: Explicit reset override. ``None`` enables
            reset only when persistent execution is enabled.

    Returns:
        Whether retained windows should be reset before reuse.

    Raises:
        ValueError: If reset is explicitly enabled without persistent execution.
    """
    if reset_persistent_windows is None:
        return persistent
    if reset_persistent_windows and not persistent:
        raise ValueError("DistributedWorker reset_persistent_windows=True requires persistent=True")
    return reset_persistent_windows


@dataclass
class _DispatchFrame:
    """One of two reusable, pre-fork dispatch metadata frames."""

    slot_id: int
    in_use: bool = False
    tensors: dict[str, Any] = field(default_factory=dict)
    keepalive: list[Any] = field(default_factory=list)
    cleanup: list[Callable[[], None]] = field(default_factory=list)
    handle: DistributedRunHandle | None = None


class DistributedRunHandle:
    """Completion handle returned by :meth:`DistributedWorker.submit`.

    The handle keeps its worker, immutable dispatch configuration, argument
    references, generated task arguments, and one bounded metadata frame alive
    until terminal completion. Waiting is idempotent and every waiter observes
    the same cached outcome.
    """

    def __init__(
        self,
        worker: DistributedWorker,
        native_handle: Any | None,
        frame: _DispatchFrame,
        dispatch_id: int,
        postprocess: Callable[[], None] | None = None,
    ) -> None:
        self._worker: DistributedWorker | None = worker
        self._native_handle = native_handle
        self._frame: _DispatchFrame | None = frame
        self._dispatch_id = dispatch_id
        self._postprocess = postprocess
        self._cv = threading.Condition()
        self._wait_in_progress = False
        self._terminal = False
        self._error: BaseException | None = None

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        value = float(timeout)
        if value < 0 or not math.isfinite(value):
            raise ValueError("DistributedRunHandle timeout must be a non-negative finite number of seconds")
        return time.monotonic() + value

    @classmethod
    def _completed(
        cls,
        worker: DistributedWorker,
        error: BaseException | None = None,
    ) -> DistributedRunHandle:
        handle = cls.__new__(cls)
        handle._worker = worker
        handle._native_handle = None
        handle._frame = None
        handle._dispatch_id = 0
        handle._postprocess = None
        handle._cv = threading.Condition()
        handle._wait_in_progress = False
        handle._terminal = True
        handle._error = error
        return handle

    @property
    def done(self) -> bool:
        """Whether the dispatch and its result publication are terminal."""
        with self._cv:
            if self._terminal:
                return True
            if self._wait_in_progress:
                return False
        try:
            self.result(timeout=0.0)
        except TimeoutError:
            return False
        except BaseException:  # noqa: BLE001 - a failed dispatch is terminal
            return True
        return True

    def result(self, timeout: float | None = None) -> None:
        """Wait for completion and raise the cached dispatch error, if any.

        Args:
            timeout: Maximum wait in seconds. ``None`` waits without a deadline.

        Raises:
            TimeoutError: The dispatch did not complete before ``timeout``.
            ValueError: ``timeout`` is negative or non-finite.
            BaseException: The cached native or result-publication failure.
        """
        deadline = self._deadline(timeout)
        with self._cv:
            while not self._terminal and self._wait_in_progress:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("DistributedRunHandle.result() timed out")
                self._cv.wait(timeout=remaining)
            if self._terminal:
                if self._error is not None:
                    raise self._error
                return
            self._wait_in_progress = True

        error: BaseException | None = None
        try:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            native_handle = self._native_handle
            if native_handle is None:
                raise RuntimeError("DistributedRunHandle lost its native handle before completion")
            native_handle.result(timeout=remaining)
        except TimeoutError:
            with self._cv:
                self._wait_in_progress = False
                self._cv.notify_all()
            raise
        except BaseException as exc:  # noqa: BLE001 - cached for every waiter
            error = exc

        worker = self._worker
        frame = self._frame
        if worker is not None and frame is not None:
            try:
                worker._run_dispatch_cleanup(frame.cleanup)
            except BaseException as exc:  # noqa: BLE001 - cleanup is part of terminal publication
                if error is None:
                    error = exc
                elif error.__context__ is None:
                    error.__context__ = exc
        if error is None and self._postprocess is not None:
            try:
                self._postprocess()
            except BaseException as exc:  # noqa: BLE001 - terminal post-processing outcome
                error = exc

        with self._cv:
            self._error = error
            self._terminal = True
            self._wait_in_progress = False
            self._native_handle = None
            self._postprocess = None
            self._frame = None
            self._worker = None
            self._cv.notify_all()
        if worker is not None and frame is not None:
            worker._retire_dispatch_handle(self, frame, error)
        if error is not None:
            raise error

    def wait(self, timeout: float | None = None) -> None:
        """Wait for completion as an alias for :meth:`result`.

        Args:
            timeout: Maximum wait in seconds. ``None`` waits without a deadline.
        """
        self.result(timeout)


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


def _tensor_from_continuous(ct) -> torch.Tensor:
    """Convert a mapped wire arg (or legacy chip tensor) to torch, zero-copy.

    Current simpler Python SubWorkers receive ``MappedArg`` objects and expose
    the view origin through ``arg.buffer``. The explicit ``data`` pointer path
    is retained narrowly for older/direct chip-tensor callers.

    For dtypes that ``torch.from_numpy`` cannot accept directly (FP16/BF16),
    we view the buffer as raw bytes (uint8) and reinterpret with
    ``torch.Tensor.view(dtype)`` — a zero-copy bit-cast that preserves the
    shared-memory aliasing required for ``Out``/``InOut`` parameters.
    """
    # ``str(ct.dtype)`` yields ``"DataType.FLOAT32"``; strip the enum prefix
    # to match the bare type names used as keys in ``_DTYPE_MAP``.
    dtype_str = str(ct.dtype)
    dtype_key = dtype_str.rsplit(".", 1)[-1]
    if dtype_key.isdecimal():
        from simpler.task_interface import DataType  # pyright: ignore[reportMissingImports]  # noqa: PLC0415

        dtype_key = DataType(int(dtype_key)).name
    try:
        c_type, torch_dtype = _DTYPE_MAP[dtype_key]
    except KeyError as exc:
        raise TypeError(
            f"Unsupported simpler Tensor dtype: {dtype_str!r}. "
            f"Add an explicit mapping in _DTYPE_MAP. "
            f"Known dtypes: {sorted(_DTYPE_MAP)}"
        ) from exc

    n_elements = 1
    for s in ct.shapes:
        n_elements *= s

    # Compute the buffer length in units of c_type, then in elements of torch_dtype.
    element_bytes = ctypes.sizeof(c_type)
    torch_bytes = torch.tensor([], dtype=torch_dtype).element_size()
    n_c_elements = n_elements * torch_bytes // element_bytes

    if hasattr(ct, "buffer"):
        strides = tuple(int(s) for s in ct.strides)
        shapes = tuple(int(s) for s in ct.shapes)
        span = 1 + sum((shape - 1) * stride for shape, stride in zip(shapes, strides, strict=True))
        t = torch.frombuffer(ct.buffer, dtype=torch_dtype, count=span)
        return torch.as_strided(t, size=shapes, stride=strides)
    if not hasattr(ct, "data"):
        raise TypeError(
            "Expected a simpler MappedArg with .buffer or a legacy chip tensor with .data, "
            f"got {type(ct).__name__}"
        )

    arr = np.ctypeslib.as_array(ctypes.cast(ct.data, ctypes.POINTER(c_type)), shape=(n_c_elements,))
    t = torch.from_numpy(arr)
    if t.dtype != torch_dtype:
        # view(dtype) reinterprets the bytes without copying — preserves shared memory.
        t = t.view(torch_dtype)
    return t.reshape(ct.shapes)


def _load_generated_module(path: Path) -> Any:
    """Dynamically load a generated Python module from *path*."""
    module_name = f"_pypto_generated.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generated module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # Generated modules live only in ``sys.modules`` — there is no
    # ``_pypto_generated`` package on disk to re-import them by name. The
    # runtime cloudpickles every registered callable to derive its hashid
    # descriptor (runtime #891); without this, cloudpickle would serialize
    # functions from this module *by reference* and fail to re-import
    # ``_pypto_generated.<stem>`` (PicklingError). Force by-value pickling so
    # the function code travels inside the payload.
    #
    # Best-effort: cloudpickle is a ``simpler`` (runtime) dependency, absent in
    # lean codegen-only / unit-test environments. When it is missing the
    # callable-registration path that needs by-value pickling cannot run
    # either, so there is nothing to protect — skip the registration. The
    # import is local so plain ``import pypto`` never requires cloudpickle.
    try:
        import cloudpickle  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        cloudpickle.register_pickle_by_value(module)
    except ImportError:
        pass
    return module


# ---------------------------------------------------------------------------
# Setup steps shared by the one-shot ``execute_distributed`` path and the
# reusable ``DistributedWorker`` handle. Keeping them as free functions lets
# both paths run identical, expensive setup (compile_and_assemble, module load,
# Worker construction + registration) without duplicating it.
# ---------------------------------------------------------------------------


def _assemble_chip_callables(
    compiled: DistributedCompiledProgram,
) -> tuple[dict[str, Any], str, bool]:
    """Build a ChipCallable for each chip-level task under ``next_levels/{name}/``.

    Driven entirely by the on-disk layout — each ``next_levels/{name}/`` that
    contains a ``kernel_config.py`` is a complete single-chip sub-build that
    :func:`compile_and_assemble` consumes directly. This requires no live IR, so
    it works identically for a freshly-compiled program and one reconstructed via
    :meth:`DistributedCompiledProgram.from_dir` (the ``runtime_dir`` replay path).
    """
    chip_callables: dict[str, Any] = {}
    runtime_name: str | None = None
    enable_sdma = False
    next_levels_dir = compiled.output_dir / "next_levels"
    if next_levels_dir.is_dir():
        for chip_dir in sorted(next_levels_dir.iterdir()):
            if not (chip_dir / "kernel_config.py").exists():
                continue
            # Imported lazily — and only once there is a real chip to build — so
            # the "no chip-level tasks" error path below stays usable without the
            # heavy device_runner → simpler toolchain import.
            from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

            chip_callable, chip_runtime, chip_runtime_config = compile_and_assemble(
                chip_dir, compiled.platform
            )
            chip_callables[chip_dir.name] = chip_callable
            enable_sdma = enable_sdma or bool(chip_runtime_config.get("enable_sdma", False))
            if runtime_name is None:
                runtime_name = chip_runtime
            elif chip_runtime != runtime_name:
                raise RuntimeError(
                    f"Inconsistent runtime across next_levels/ sub-builds in {next_levels_dir}: "
                    f"{runtime_name!r} (earlier chip) vs {chip_runtime!r} (chip {chip_dir.name!r}). "
                    f"All chip-level tasks in one distributed build must share a single runtime."
                )

    if not chip_callables:
        raise RuntimeError(
            f"No chip-level tasks found in {next_levels_dir} (expected one or more "
            f"next_levels/<name>/ sub-builds each containing a kernel_config.py)."
        )
    # Non-empty chip_callables guarantees the loop set runtime_name at least once.
    assert runtime_name is not None
    return chip_callables, runtime_name, enable_sdma


# Sentinel attribute that DistributedCodegen sets on the generated host
# orchestrator function (``<name>._pypto_distributed_entry = True``). The entry
# is resolved by this marker rather than by function name, so renaming the
# ``@pl.jit.host`` orchestrator does not break dispatch (issue #1678). Keep in
# sync with ``EmitEntryMarker`` in src/codegen/distributed/distributed_codegen.cpp.
_ENTRY_MARKER = "_pypto_distributed_entry"


def _load_orch_entry(output_dir: Path) -> tuple[Any, Any]:
    """Load the generated ``host_orch.py`` and return ``(entry_fn, alloc_fn)``.

    The dispatch entry is the unique module-level function carrying the
    ``_pypto_distributed_entry`` marker emitted by codegen — resolution never
    depends on the function's Python name (issue #1678).

    ``alloc_fn`` is the optional ``_alloc_intermediates(tensors)`` that
    pre-allocates HOST-level scratch tensors (``None`` when absent).
    """
    orch_path = output_dir / "orchestration" / "host_orch.py"
    if not orch_path.exists():
        raise FileNotFoundError(
            f"Generated orchestration not found at {orch_path}. Did the codegen produce distributed output?"
        )
    orch_module = _load_generated_module(orch_path)

    entry_candidates = [
        obj
        for name in dir(orch_module)
        if isinstance((obj := getattr(orch_module, name)), types.FunctionType)
        and getattr(obj, "__module__", None) == orch_module.__name__
        and getattr(obj, _ENTRY_MARKER, False)
    ]
    if len(entry_candidates) != 1:
        found = [fn.__name__ for fn in entry_candidates]
        raise RuntimeError(
            f"Expected exactly one entry function marked with `{_ENTRY_MARKER}` in "
            f"{orch_path}, found {len(entry_candidates)}: {found}. The generated "
            f"orchestration module is malformed — regenerate via distributed codegen."
        )
    entry_fn = entry_candidates[0]

    alloc_fn = getattr(orch_module, "_alloc_intermediates", None)
    return entry_fn, alloc_fn


def _call_alloc_intermediates(alloc_fn: Any, tensors: dict[str, Any], world_size: int) -> None:
    """Invoke the generated ``_alloc_intermediates``, passing ``world_size`` only if it
    accepts it.

    Codegen emits ``_alloc_intermediates(tensors, world_size=1)`` so it can size the
    per-rank comm ordering tokens, but the one-argument form must keep working: a
    ``build_output/`` directory produced by an older pypto is replayable via
    :meth:`DistributedCompiledProgram.from_dir`, and callers may inject their own
    allocator. Probing the signature (rather than catching :class:`TypeError` around the
    call) keeps a ``TypeError`` raised *inside* the allocator from being mistaken for a
    signature mismatch and silently retried.
    """
    try:
        params = inspect.signature(alloc_fn).parameters
    except (TypeError, ValueError):  # builtin / C callable with no introspectable signature
        params = {}
    takes_world_size = "world_size" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if takes_world_size:
        alloc_fn(tensors, world_size=world_size)
    else:
        alloc_fn(tensors)


def _load_sub_worker_fns(output_dir: Path) -> dict[str, Any]:
    """Load SubWorker callables from ``sub_workers/*.py`` (keyed by file stem)."""
    sub_worker_fns: dict[str, Any] = {}
    sub_workers_dir = output_dir / "sub_workers"
    if sub_workers_dir.exists():
        for py_file in sorted(sub_workers_dir.glob("*.py")):
            mod = _load_generated_module(py_file)
            fn_name = py_file.stem
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                sub_worker_fns[fn_name] = fn
    return sub_worker_fns


def _load_required_callbacks(output_dir: Path) -> set[str]:
    """Names of abstract SubWorkers that MUST be bound via ``callbacks={...}``.

    Read from the ``sub_workers/__required__.json`` manifest emitted by codegen
    for ``...``-body SubWorkers. Missing manifest ⇒ no required callbacks.
    """
    manifest = output_dir / "sub_workers" / "__required__.json"
    if not manifest.exists():
        return set()
    return set(json.loads(manifest.read_text()))


def _construct_worker(
    dc: DistributedConfig,
    platform: str,
    runtime_name: str,
    num_sub: int,
    enable_sdma: bool = False,
    startup_timeout_s: float | None = None,
) -> Any:
    """Construct a simpler ``Worker(level=3)`` from the distributed config."""
    if startup_timeout_s is not None and (not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0):
        raise ValueError(
            "DistributedWorker startup_timeout_s must be a positive finite number of seconds, "
            f"got {startup_timeout_s!r}"
        )

    from simpler.worker import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        Worker,
    )

    worker_config: dict[str, Any] = {
        "level": 3,
        "device_ids": dc.device_ids,
        "num_sub_workers": num_sub,
        "platform": platform,
        "runtime": runtime_name,
        "enable_sdma": enable_sdma,
    }
    if startup_timeout_s is not None:
        worker_config["startup_timeout_s"] = startup_timeout_s
    return Worker(**worker_config)


def _close_local_worker(w: Any) -> None:
    """Close a locally owned Worker, retrying incomplete Simpler cleanup once.

    Simpler keeps failed cleanup-journal entries for the next ``close()`` call.
    Local workers are not returned to the user, so give transient cleanup one
    bounded retry here while still surfacing a persistent failure.
    """
    try:
        w.close()
    except BaseException as first_error:  # noqa: BLE001 - cleanup can preserve control-flow failures
        try:
            w.close()
        except BaseException:  # noqa: BLE001 - surface the retry outcome
            raise
        if not isinstance(first_error, Exception):
            # Cleanup completed, but KeyboardInterrupt/SystemExit must remain
            # visible when there is no earlier operation failure to preserve.
            raise


def _close_local_worker_after_error(w: Any, operation: str) -> None:
    """Best-effort cleanup without replacing an active operation failure."""
    try:
        _close_local_worker(w)
    except BaseException:  # noqa: BLE001 - preserve the primary failure below
        logger.exception(
            "%s failed; Worker cleanup was interrupted or still failed after one retry",
            operation,
        )


def _register_callables(
    w: Any, sub_worker_fns: dict[str, Any], chip_callables: dict[str, Any]
) -> tuple[dict[str, int], dict[str, int]]:
    """Register SubWorker + Chip callables before ``w.init()``.

    Both must happen before ``w.init()`` so the L3 fork inherits the registry
    via COW (runtime PR #710); the emitted host_orch then dispatches via cids —
    ``orch.submit_sub(sub_ids[name], …)`` / ``orch.submit_next_level(callables[name], …)``.
    """
    # ``w.register`` returns an opaque ``CallableHandle`` (runtime #891); typed
    # ``Any`` here and threaded straight back into ``submit_sub`` /
    # ``submit_next_level``, which accept the handle.
    sub_ids: dict[str, Any] = {name: w.register(fn) for name, fn in sub_worker_fns.items()}
    chip_cids: dict[str, Any] = {name: w.register(cc) for name, cc in chip_callables.items()}
    return sub_ids, chip_cids


def _check_callback_arity(name: str, fn: Callable[..., Any]) -> None:
    """Validate that a user callback can be invoked as ``fn(args)``.

    SubWorker callables receive a single ``MappedArgs`` positional argument. A
    callback that cannot accept exactly one positional arg is almost certainly
    the wrong function — reject it with a clear error instead of failing deep
    inside dispatch with an opaque ``TypeError``.
    """
    if not callable(fn):
        raise TypeError(f"callback for SubWorker '{name}' is not callable: {fn!r}")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return  # builtins / C callables expose no signature — skip the check
    try:
        sig.bind(object())  # one positional arg, like the runtime's fn(args)
    except TypeError as exc:
        raise TypeError(
            f"callback for SubWorker '{name}' must accept a single positional "
            f"argument fn(args: MappedArgs); got signature {sig}."
        ) from exc


def _coalesce_callbacks(
    callbacks: dict[str, Callable[..., Any]] | None,
    sub_worker_overrides: dict[str, Callable[..., Any]] | None,
) -> dict[str, Callable[..., Any]] | None:
    """Merge the deprecated ``sub_worker_overrides`` alias into ``callbacks``.

    ``callbacks`` takes precedence on name collisions. Returns ``None`` when both
    are empty so downstream ``or {}`` handling stays simple.
    """
    if sub_worker_overrides is None:
        return callbacks
    warnings.warn(
        "sub_worker_overrides is deprecated; use callbacks= instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return {**sub_worker_overrides, **(callbacks or {})}


def _bind_sub_workers(
    loaded: dict[str, Any],
    callbacks: dict[str, Callable[..., Any]] | None,
    required: set[str],
) -> dict[str, Any]:
    """Bind user callbacks onto the codegen-loaded SubWorker set (by name).

    Each callback replaces the generated module for an existing SubWorker.

    - Unknown names are rejected: binding a name the program does not declare
      would register an unused callable while the orchestrator kept calling the
      generated module (a silent no-op, usually a typo).
    - Abstract SubWorkers (``...`` body, listed in *required*) MUST be bound —
      their generated module only raises. A missing binding is reported here, at
      prepare time, rather than at dispatch.
    """
    callbacks = callbacks or {}
    unknown = sorted(set(callbacks) - set(loaded))
    if unknown:
        raise ValueError(
            f"callbacks names {unknown} are not sub-workers of this program. "
            f"Available sub-workers: {sorted(loaded)}."
        )
    missing = sorted(required - set(callbacks))
    if missing:
        raise ValueError(
            f"SubWorkers {missing} are runtime-bound callbacks (declared with a "
            f"`...` body) and must be supplied via callbacks={{...}}."
        )
    for name, fn in callbacks.items():
        _check_callback_arity(name, fn)
    return {**loaded, **callbacks}


def _make_call_config(
    dc: DistributedConfig,
    run_config: RunConfig | None = None,
    *,
    dfx_base: Path | None = None,
    co_enable_swimlane_dep_gen: bool = True,
) -> Any:
    """Build a simpler ``CallConfig`` from the distributed config.

    The ``aicpu_thread_num`` baseline always comes from the
    program's :class:`DistributedConfig`. When *run_config* is given, its
    per-task ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
    ``ring_dep_pool``, each a scalar or a per-ring list of 4 ints) are overlaid
    on top, so a single L3 dispatch can size the
    runtime's ring buffers without mutating the prepared program's shared
    config. ``None`` (the default) leaves the baseline untouched and the runtime
    applies its own compile-time default.

    DFX diagnostics (``enable_dump_args`` / ``enable_pmu`` / ``enable_dep_gen``
    / ``enable_scope_stats`` / ``enable_chip_swimlane``) are likewise read from
    *run_config* and written to the shared ``config`` the host_orch chip dispatch
    forwards to every ``orch.submit_next_level``; their artifacts land under
    *dfx_base* (``<output_dir>/dfx_outputs``). By default,
    ``enable_chip_swimlane`` co-enables dep_gen so a single dispatch still has the
    task graph needed by the converter. Onboard L3 callers use a two-pass
    graph/timing protocol and set *co_enable_swimlane_dep_gen* false while
    building the clean timing pass.

    Args:
        dc: The program's distributed configuration (baseline).
        run_config: Optional per-dispatch :class:`RunConfig` whose ``ring_*`` and
            DFX overrides are applied. ``None`` means no override.
        dfx_base: Directory under which DFX artifacts are written
            (``<output_dir>/dfx_outputs``). Required whenever *run_config*
            enables a DFX flag; created if missing.
        co_enable_swimlane_dep_gen: Whether swimlane implicitly enables
            dep_gen. The onboard graph pass and simulator single-pass use the
            default; the onboard clean timing pass disables it.

    Returns:
        A fresh simpler ``CallConfig``.

    Raises:
        ValueError: a DFX flag is enabled but *dfx_base* is ``None``.
    """
    from simpler.task_interface import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        CallConfig,  # pyright: ignore[reportAttributeAccessIssue]
    )

    call_config = CallConfig()
    call_config.aicpu_thread_num = dc.aicpu_thread_num
    if run_config is not None:
        from .runner import _apply_ring_overrides, _DfxOpts  # noqa: PLC0415

        _apply_ring_overrides(call_config, run_config)

        dfx = _DfxOpts.from_run_config(run_config)
        if dfx.any():
            if dfx_base is None:
                raise ValueError("_make_call_config: dfx_base is required when a DFX flag is enabled on L3")
            dfx_base.mkdir(parents=True, exist_ok=True)
            call_config.enable_dump_args = dfx.enable_dump_args
            call_config.enable_pmu = dfx.enable_pmu
            # Swimlane needs ``deps.json`` so the converter can resolve task
            # arrows / kernel names. Onboard one-shot and prepared paths run a
            # clean two-pass (pass 1 dep_gen → deps.json, pass 2 swimlane → clean
            # records) and set ``co_enable_swimlane_dep_gen=False`` on the timing
            # pass so dep_gen does not perturb it. Simulator and direct
            # single-pass builders keep the default co-enable behavior.
            # ``enable_chip_swimlane`` is a collection level (0-4), so the
            # ``or``/``and`` chain can yield an int; the
            # ``CallConfig.enable_dep_gen`` pybind setter only accepts ``bool``.
            # Wrap in ``bool(...)`` to avoid a TypeError.
            call_config.enable_dep_gen = bool(
                dfx.enable_dep_gen or (co_enable_swimlane_dep_gen and dfx.enable_chip_swimlane)
            )
            call_config.enable_scope_stats = dfx.enable_scope_stats
            call_config.enable_chip_swimlane = dfx.enable_chip_swimlane
            # Base dir shared by every chip; ``_submit_chip`` namespaces it per
            # dispatch (``<dfx_base>/rank{worker}/d{k}``) so per-dispatch
            # artifacts (pmu.csv, deps.json, chip_swimlane_records.json, ...) don't
            # overwrite each other — even when one card runs multiple dispatches.
            call_config.output_prefix = str(dfx_base)
    return call_config


def _run_l3_swimlane_two_pass(
    dc: DistributedConfig,
    config: RunConfig,
    dfx_base: Path,
    run_pass: Callable[[Any], None],
) -> None:
    """Capture the L3 task graph, then run a dep-gen-free timing pass.

    ``run_pass`` owns the execution lifecycle: the one-shot path creates a
    fresh Worker for each call, while a prepared ``DistributedWorker`` reuses
    its existing Worker and waits on a submitted run handle. Both paths
    reset their per-card dispatch counters, so matching graph/timing dispatches
    land in the same ``rank{r}/d{k}`` directory.

    Both calls execute the program. As with the existing one-shot L3 protocol,
    mutable arguments are not snapshotted or restored between passes.
    """
    import dataclasses  # noqa: PLC0415

    from .bench import (  # noqa: PLC0415
        _L3_SWIMLANE_GRAPH_BEGIN,
        _L3_SWIMLANE_GRAPH_END,
        _L3_SWIMLANE_TIMING_BEGIN,
        _L3_SWIMLANE_TIMING_END,
    )

    print(
        "[swimlane] L3 swimlane enabled -> running the dispatch twice "
        "(dep_gen perturbs timing, so the graph and the timing are captured separately):"
    )
    print("[swimlane] run 1/2: capturing the per-dispatch task graph (deps.json); its timing is discarded.")
    deps_cfg = dataclasses.replace(
        config,
        enable_chip_swimlane=0,
        enable_dep_gen=True,
        enable_pmu=0,
        enable_scope_stats=False,
        enable_dump_args=0,
    )
    print(_L3_SWIMLANE_GRAPH_BEGIN, file=sys.stderr, flush=True)
    run_pass(_make_call_config(dc, deps_cfg, dfx_base=dfx_base))
    print(_L3_SWIMLANE_GRAPH_END, file=sys.stderr, flush=True)

    print("[swimlane] run 2/2: measuring clean per-task timing (these are the reported numbers).")
    # ``benchmark`` and pypto-lib's resident benchmark capture fd 2 around the
    # prepared worker. Bracketing the blocking timing run lets their shared
    # parser retain its child-process STRACE records while discarding graph-pass
    # records, even when each pass has a data-dependent dispatch count.
    print(_L3_SWIMLANE_TIMING_BEGIN, file=sys.stderr, flush=True)
    timing_cfg = dataclasses.replace(config, enable_dep_gen=False)
    run_pass(_make_call_config(dc, timing_cfg, dfx_base=dfx_base, co_enable_swimlane_dep_gen=False))
    print(_L3_SWIMLANE_TIMING_END, file=sys.stderr, flush=True)


# A dispatch's DFX artifacts live at ``<dfx_base>/<rank label>/d{k}``. The
# producer (``_submit_chip``) and the consumers (``_clear_dfx_dispatch_dirs``,
# ``_collect_l3_swimlane``) must agree on that scheme, and drift between them is
# *silent* — a glob that no longer matches simply clears/converts nothing rather
# than raising. The globs below are what the two consumers share; the label
# builder names the producer's half of the contract in one place.
_RANK_DIR_GLOB = "rank*"
_DISPATCH_DIR_GLOB = "d[0-9]*"

# Written by ``_submit_chip`` into each dispatch dir and read back by
# ``_collect_l3_swimlane``. ``rank{w}/d{k}`` records *where* a dispatch ran but not
# *what* it ran, and a kernel's ``func_id`` only means something within one
# ``next_levels/<program>`` — so naming a dispatch's tasks needs this marker
# (issue #2169).
_DISPATCH_PROGRAM_FILE = "dispatch_program.json"


def _dfx_rank_label(worker: int) -> str:
    """Directory name namespacing one card's DFX artifacts.

    Every dispatch names a concrete chip by the time it reaches here (see
    :func:`_resolve_chip_worker`), so the label is always the chip's own rank
    and matches :data:`_RANK_DIR_GLOB`.
    """
    return f"rank{worker}"


def _resolve_chip_worker(orch: Any, worker: int | None) -> int:
    """Pick the chip a dispatch runs on.

    A ``device=``-pinned dispatch arrives with its rank and is returned as is.
    A comm-less dispatch arrives as ``None``: it expresses no affinity, but the
    runtime requires an exact target — simpler #1436 made ``worker`` a required,
    non-negative NEXT_LEVEL id and removed the "unconstrained" mode where the
    scheduler picked an idle worker itself. Those dispatches are handed out
    round-robin over the program's chips in submit order, so a host_orch with
    one comm-less dispatch per chip still spreads across them (and does so
    deterministically, unlike the old idle-pool pick).

    The chip count is stamped on ``orch`` by ``_dispatch.orch_fn``; without it
    (a caller that bypassed ``orch_fn``) every comm-less dispatch falls back to
    chip 0, which always exists.
    """
    if worker is not None:
        return worker
    chip_count = max(1, int(getattr(orch, "_pypto_chip_count", 1)))
    seq = getattr(orch, "_pypto_commless_seq", None)
    if seq is None:
        seq = 0
    orch._pypto_commless_seq = seq + 1
    return seq % chip_count


def _reset_dfx_dispatch_state(orch: Any, chip_cids: dict[str, Any]) -> None:
    """Reset the per-run dispatch state :func:`_submit_chip` reads off ``orch``.

    ``_dfx_dispatch_idx`` numbers each card's dispatches ``d0, d1, ...`` fresh per
    run, so the swimlane two-pass files one dispatch under one directory.

    ``_dfx_chip_names`` reverses the registered ``callables`` mapping so
    ``_submit_chip`` can name the L2 program behind an otherwise opaque
    ``CallableHandle``. Keyed by ``id()``: the handle is not required to be
    hashable, and ``chip_cids`` holds every one alive for the whole run, so the
    ids are stable and unique.
    """
    orch._dfx_dispatch_idx = {}
    orch._dfx_chip_names = {id(cid): name for name, cid in chip_cids.items()}


def _record_dispatch_program(orch: Any, callable_id: Any, disp_dir: Path) -> None:
    """Record which L2 program a dispatch runs, for the swimlane post-pass.

    A kernel's ``func_id`` is a per-L2-program namespace — every
    ``next_levels/<program>/kernel_config.py`` numbers its kernels from 0 — so
    labelling a dispatch's records requires knowing the program that produced
    them. This wrapper is the only place that sees both the dispatch directory
    and the callable being dispatched, so it stamps the pairing on disk
    (:data:`_DISPATCH_PROGRAM_FILE`) for :func:`_collect_l3_swimlane` to read
    back. Going through the filesystem keeps the two halves independent of where
    the runtime places the L3 orchestrator, and the marker is rewritten
    identically by both swimlane passes.

    Best-effort: a marker that cannot be written costs kernel labels, never the
    run. Silent when the caller bypassed :func:`_reset_dfx_dispatch_state` and
    left no name table on ``orch``.
    """
    name = getattr(orch, "_dfx_chip_names", {}).get(id(callable_id))
    if name is None:
        return
    try:
        disp_dir.mkdir(parents=True, exist_ok=True)
        (disp_dir / _DISPATCH_PROGRAM_FILE).write_text(json.dumps({"program": name}), encoding="utf-8")
    except OSError as e:
        print(
            f"Could not record the dispatch program for {disp_dir} ({type(e).__name__}: {e}); "
            "its swimlane falls back to anonymous task labels unless the build has a single "
            "L2 program"
        )


def _submit_chip(orch: Any, callable_id: Any, task_args: Any, config: Any, worker: int | None) -> Any:
    """``orch.submit_next_level`` with per-dispatch DFX ``output_prefix`` isolation.

    The runtime path helpers root every diagnostic artifact at a fixed filename
    under ``output_prefix`` (``<prefix>/pmu.csv`` etc.), so any two dispatches
    sharing one prefix clobber each other. Namespacing by card alone is not
    enough: one card may receive several dispatches in a single host_orch run
    (pipeline stages, expert kernels, or genuinely different chip programs all
    pinned to the same ``device``), and each re-init+finalize of the runtime's
    per-run collector rewrites the fixed-name file. So this wrapper appends
    ``/rank{worker}/d{k}`` — card *and* the card's k-th dispatch — for the
    duration of the submit, then restores the shared ``config``. The restore is
    safe because ``submit_next_level`` copies the ``CallConfig`` into the task
    slot synchronously (orchestrator ``s.config = config``) before it returns,
    so it never races the already-queued task.

    ``k`` comes from a per-card counter on ``orch`` reset at the top of every
    run (see :func:`_reset_dfx_dispatch_state`), so the numbering is
    deterministic and matches across the swimlane two-pass. The dispatched
    program is stamped into the same directory by
    :func:`_record_dispatch_program`, so the offline post-pass can label the
    records with the right program's kernel names.

    Every dispatch is namespaced ``rank{worker}/d{k}`` by the chip it runs on.
    When DFX is off (``output_prefix`` unset) the call is forwarded unchanged.

    The codegen routes every chip dispatch through this wrapper — a rank-pinned
    dispatch passes its rank, a comm-less one passes ``None`` and is resolved by
    :func:`_resolve_chip_worker`. Resolution happens before the DFX namespacing
    so the artifacts land under the chip that actually ran the dispatch.
    """
    worker = _resolve_chip_worker(orch, worker)
    base = config.output_prefix
    if not base:
        return orch.submit_next_level(callable_id, task_args, config, worker=worker)
    idx_map = getattr(orch, "_dfx_dispatch_idx", None)
    if idx_map is None:
        # Defensive: a caller that bypassed ``orch_fn`` (no reset) still gets
        # per-card isolation, just without a guaranteed two-pass match.
        idx_map = orch._dfx_dispatch_idx = {}
    rank_label = _dfx_rank_label(worker)
    k = idx_map.get(rank_label, 0)
    idx_map[rank_label] = k + 1
    config.output_prefix = f"{base}/{rank_label}/d{k}"
    _record_dispatch_program(orch, callable_id, Path(config.output_prefix))
    try:
        return orch.submit_next_level(callable_id, task_args, config, worker=worker)
    finally:
        config.output_prefix = base


def _clear_dfx_dispatch_dirs(dfx_base: Path) -> None:
    """Remove stale ``rank*/d{k}`` dispatch dirs before a fresh DFX run.

    The per-card dispatch counter resets to ``d0`` at the start of every run, so
    a prepared :class:`DistributedWorker` reusing one ``output_dir`` across
    dispatches would otherwise leave higher-numbered ``d{k}`` dirs from an
    earlier, larger run on disk. ``_collect_l3_swimlane`` globs ``d[0-9]*``, so
    those stale dirs would be re-converted as if they belonged to the current
    run. Clearing them once, before the first dispatch of a DFX run, scopes the
    artifacts (and their post-processing) to exactly this run. Called only when
    DFX is enabled; best-effort (a removal failure must not abort the dispatch).
    """
    if not dfx_base.is_dir():
        return
    import shutil  # noqa: PLC0415

    for rank_dir in dfx_base.glob(_RANK_DIR_GLOB):
        if not rank_dir.is_dir():
            continue
        for disp_dir in rank_dir.glob(_DISPATCH_DIR_GLOB):
            if disp_dir.is_dir():
                shutil.rmtree(disp_dir, ignore_errors=True)


def _read_dispatch_program(disp_dir: Path) -> str | None:
    """Name of the L2 program that ran this dispatch, or ``None`` if unrecorded.

    Reads back the marker :func:`_record_dispatch_program` wrote. A missing or
    malformed marker is not an error — it only means the labels for this dispatch
    cannot be resolved (see :func:`_collect_l3_swimlane`).
    """
    marker = disp_dir / _DISPATCH_PROGRAM_FILE
    if not marker.exists():
        return None
    try:
        program = json.loads(marker.read_text(encoding="utf-8"))["program"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return str(program)


def _write_dispatch_name_map(disp_dir: Path, chip_dir: Path, cache: dict[str, dict[str, str]]) -> Path | None:
    """Write *disp_dir*'s ``name_map.json`` from *chip_dir*'s ``kernel_config.py``.

    The map is the one the L2 path synthesises (:func:`~pypto.runtime.runner._write_name_map`),
    scoped to a single program: ``func_id`` numbering restarts per
    ``next_levels/<program>``, so merging several programs' tables would silently
    relabel one program's tasks with another's names (issue #2169).

    *cache* memoises the per-program table across the dispatches that share a
    program, so each ``kernel_config.py`` is exec'd once per run.

    Returns the written path, or ``None`` when no table could be resolved (the
    converter then falls back to anonymous ``task(rXtY)`` labels).
    """
    program = chip_dir.name
    if program not in cache:
        kc = chip_dir / "kernel_config.py"
        table: dict[str, str] = {}
        if kc.exists():
            try:
                from simpler_setup.tools.swimlane_converter import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
                    load_kernel_config,
                )

                table = load_kernel_config(str(kc))
            except Exception as e:  # noqa: BLE001 - best-effort label resolution, never fatal
                print(
                    f"Skipping L3 swimlane name_map for {program} ({type(e).__name__}: {e}); "
                    "its labels fall back to defaults"
                )
        cache[program] = table
    table = cache[program]
    if not table:
        return None
    name_map_path = disp_dir / "name_map.json"
    name_map_path.write_text(
        json.dumps({"level": 2, "orchestrator_name": None, "callable_id_to_name": table}, indent=2),
        encoding="utf-8",
    )
    return name_map_path


def _collect_l3_swimlane(output_dir: Path, platform: str) -> None:
    """Convert each dispatch's swimlane records into a ``merged_swimlane_*.json``.

    The runtime writes ``rank{r}/d{k}/deps.json`` in the graph pass and
    ``rank{r}/d{k}/chip_swimlane_records.json`` in the clean timing pass
    (``_submit_chip`` namespaces the directory by card *and* the card's k-th
    dispatch, and both passes reset that counter). Globbing ``rank*`` — rather
    than iterating a rank count — picks up
    whichever cards actually ran, so a comm-less / single-card L3 program (which never
    creates ``rank{0..n}``) still has its records converted. This best-effort
    post-pass runs the offline ``swimlane_converter`` once per dispatch dir. Each
    dispatch's records are single-chip, so the L2 converter applies unchanged —
    and a card that ran several (possibly different) programs keeps one swimlane
    per dispatch instead of overwriting down to the last.

    Kernel names are resolved **per dispatch**, from the ``kernel_config.py`` of
    the ``next_levels/<program>`` that :func:`_record_dispatch_program` stamped on
    that dispatch. ``func_id`` numbering restarts in every program, so a table
    merged across programs would relabel one program's tasks with another's names
    — silently and plausibly (issue #2169). A dispatch whose program cannot be
    resolved is therefore converted with anonymous labels rather than guessed
    ones, and says so.

    Onboard-only: the simulator emits records but not the task metadata the
    converter joins against, so conversion is skipped there (mirrors the L2
    ``_collect_dfx_artifacts`` swimlane branch). Any failure is logged, never
    raised — the raw records remain for manual conversion.
    """
    if platform.endswith("sim"):
        print(
            "Skipping L3 swimlane conversion on simulator: merged_swimlane_*.json "
            "is only generated for onboard runs (raw chip_swimlane_records.json kept)."
        )
        return

    from .runner import _CHIP_SWIMLANE_RECORDS_NAME, _generate_swimlane  # noqa: PLC0415

    # ``glob("*/")`` directory filtering is only reliable on 3.11+; filter
    # explicitly so this works on the 3.10 baseline too.
    # A ``next_levels/<name>/`` is an L2 program exactly when it carries a
    # ``kernel_config.py`` — the same test ``_assemble_chip_callables`` applies,
    # so both halves count the same programs and a stray subdir cannot make the
    # single-program fallback below look ambiguous.
    chip_dirs = {
        d.name: d
        for d in sorted((output_dir / "next_levels").glob("*"))
        if d.is_dir() and (d / "kernel_config.py").exists()
    }
    # program name -> its ``func_id`` table; filled on first use by
    # ``_write_dispatch_name_map`` so each config is loaded once per run.
    name_map_cache: dict[str, dict[str, str]] = {}

    dfx_base = output_dir / "dfx_outputs"
    # See the docstring for why we glob rather than iterate a rank count.
    # 3.10-safe dir filter (``glob`` directory filtering is only reliable on 3.11+).
    rank_dirs = sorted(d for d in dfx_base.glob(_RANK_DIR_GLOB) if d.is_dir())
    for rank_dir in rank_dirs:
        # One card may have run several dispatches: ``<rank>/d0``, ``d1``, ...
        # Match only ``d`` + digits (the names ``_submit_chip`` emits) so an
        # unrelated diagnostic dir under rank_dir is never picked up.
        dispatch_dirs = sorted(d for d in rank_dir.glob(_DISPATCH_DIR_GLOB) if d.is_dir())
        for disp_dir in dispatch_dirs:
            records = disp_dir / _CHIP_SWIMLANE_RECORDS_NAME
            if not records.exists():
                continue
            # Best-effort, as documented: a write/convert failure for one
            # dispatch must not turn a successful run into a post-processing
            # crash. The raw records remain on disk for manual conversion.
            try:
                # A dispatch must be rendered from a map this run wrote, so drop
                # any left by an earlier one before deciding what to write. When
                # no map is passed, the converter falls back to a sibling
                # ``name_map*.json``, and a stale one quietly resurrects the
                # mislabelling below. Doing it here — rather than per branch —
                # makes that hold whatever the converter's own precedence
                # between ``--func-names``, ``-k`` and the sibling turns out to be.
                for stale in disp_dir.glob("name_map*.json"):
                    stale.unlink(missing_ok=True)
                program = _read_dispatch_program(disp_dir)
                if program is None and len(chip_dirs) == 1:
                    # One L2 program in the build: no ambiguity to resolve, so an
                    # unmarked dispatch (e.g. artifacts from an older run) can
                    # still be named correctly.
                    program = next(iter(chip_dirs))
                if program is None or program not in chip_dirs:
                    print(
                        f"No L2 program recorded for {rank_dir.name}/{disp_dir.name}; converting with "
                        "anonymous task labels. For real kernel names, re-run: python -m "
                        "simpler_setup.tools.swimlane_converter "
                        f"{records} -k {output_dir / 'next_levels'}/<program>/kernel_config.py"
                    )
                    # No ``kernel_config.py`` here, so ``_generate_swimlane``
                    # omits ``-k`` rather than pointing the converter at another
                    # program's table.
                    work_dir: Path = output_dir
                    name_map_path: Path | None = None
                else:
                    # ``work_dir`` feeds the converter's ``-k`` fallback and the
                    # ``name_map`` passed as ``func_names`` takes precedence —
                    # both must name the program that ran this dispatch.
                    work_dir = chip_dirs[program]
                    name_map_path = _write_dispatch_name_map(disp_dir, work_dir, name_map_cache)
                _generate_swimlane(work_dir, disp_dir, records, func_names=name_map_path)
            except Exception as e:  # noqa: BLE001 - best-effort post-pass, never fatal
                print(
                    f"Skipping L3 swimlane conversion for {disp_dir.name} of {rank_dir.name} "
                    f"({type(e).__name__}: {e}); raw records kept"
                )


def _is_simpler_tensor(arg: Any) -> bool:
    """True if *arg* is a simpler ``Tensor``.

    Returns ``False`` (rather than raising) when simpler is unavailable, so the
    DeviceTensor-only path stays importable without the runtime package.
    """
    try:
        from .task_interface import (  # noqa: PLC0415
            Tensor,  # pyright: ignore[reportAttributeAccessIssue]
        )
    except ImportError:
        return False
    return isinstance(arg, Tensor)


def _make_dispatch_orchestration(
    entry_fn: Any,
    tensors: dict[str, Any],
    chip_cids: dict[str, Any],
    sub_ids: dict[str, Any],
    call_config: Any,
    device_nums: int,
    keepalive: list[Any],
) -> Callable[..., None]:
    """Build one orchestration closure over a handle-owned metadata frame."""
    # ``world_size`` is the only worker-level scalar the entry needs; codegen
    # binds ``pld.system.world_size()`` to this kwarg uniformly across comm
    # and comm-less paths.

    def orch_fn(orch, _unused_args, _unused_cfg):
        # Reset the per-card DFX dispatch counter at the start of every run so
        # ``_submit_chip`` numbers a card's dispatches ``d0, d1, ...`` fresh each
        # pass. Two-pass swimlane reissues the same dispatch order, so pass 1
        # (deps.json) and pass 2 (records) land the same dispatch in the same
        # ``rank{w}/d{k}`` dir — letting the converter join them. The same call
        # publishes the callable -> L2 program names ``_submit_chip`` stamps into
        # each dispatch dir.
        _reset_dfx_dispatch_state(orch, chip_cids)
        # Comm-less dispatches carry no rank, so ``_resolve_chip_worker`` hands
        # them out round-robin over the program's chips; both the count and the
        # sequence live on ``orch`` so the wrapper stays a pure function of the
        # dispatch. Resetting the sequence per run keeps placement identical
        # across the swimlane two-pass, exactly like the DFX counter above.
        orch._pypto_chip_count = device_nums
        orch._pypto_commless_seq = 0
        entry_fn(
            orch,
            _unused_args,
            call_config,
            tensors=tensors,
            callables=chip_cids,
            sub_ids=sub_ids,
            _keep=keepalive,
            world_size=device_nums,
        )

    return orch_fn


def _submit_dispatch(
    w: Any,
    entry_fn: Any,
    tensors: dict[str, Any],
    chip_cids: dict[str, Any],
    sub_ids: dict[str, Any],
    call_config: Any,
    device_nums: int,
    keepalive: list[Any],
) -> Any:
    """Submit one orchestration closure and return Simpler's run handle."""
    orch_fn = _make_dispatch_orchestration(
        entry_fn,
        tensors,
        chip_cids,
        sub_ids,
        call_config,
        device_nums,
        keepalive,
    )
    return w.submit(orch_fn)


def _dispatch(
    w: Any,
    entry_fn: Any,
    tensors: dict[str, Any],
    chip_cids: dict[str, Any],
    sub_ids: dict[str, Any],
    call_config: Any,
    device_nums: int,
) -> None:
    """Blocking compatibility composition of submit plus result."""
    keepalive: list[Any] = []
    native_handle = _submit_dispatch(
        w,
        entry_fn,
        tensors,
        chip_cids,
        sub_ids,
        call_config,
        device_nums,
        keepalive,
    )
    native_handle.result()


def execute_distributed(
    compiled: DistributedCompiledProgram,
    coerced_args: Sequence[torch.Tensor | DeviceTensor | StackedDeviceTensor],
    config: RunConfig | None = None,
) -> None:
    """Execute a distributed compiled program once via simpler Worker(level=3).

    One-shot path: runs the full setup, dispatches once, then tears the Worker
    down. Supports host ``torch.Tensor`` inputs (placed in shared memory before
    the fork). Device-resident arguments require a retained owner ``Buffer``
    from this exact runtime Worker, which a one-shot call cannot expose; use
    :meth:`DistributedCompiledProgram.prepare` → :class:`DistributedWorker`.

    Args:
        compiled: The DistributedCompiledProgram instance.
        coerced_args: Coerced host ``torch.Tensor`` arguments. A
            :class:`~pypto.runtime.DeviceTensor` or
            :class:`~pypto.runtime.StackedDeviceTensor` is rejected on this
            one-shot path.
        config: Optional per-dispatch :class:`RunConfig`. Its per-task
            ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
            ``ring_dep_pool``, each a scalar or a per-ring list of 4 ints) size
            this dispatch's runtime ring buffers, and its
            runtime-diagnostic DFX flags (``enable_dump_args`` / ``enable_pmu``
            / ``enable_dep_gen`` / ``enable_scope_stats`` / ``enable_chip_swimlane``)
            are written per dispatch under
            ``<output_dir>/dfx_outputs/rank{r}/d{k}/`` (``d{k}`` is the card's
            k-th dispatch, so multiple — even different — chip programs on one
            card keep separate artifacts). Onboard, ``enable_chip_swimlane`` runs a
            clean two-pass dispatch (pass 1 dep_gen → ``deps.json``, pass 2
            swimlane → records with unperturbed timing) and additionally produces
            ``merged_swimlane_*.json`` per dispatch. The remaining compile-side
            fields are not consumed on the dispatch path. ``None`` defers every
            ring field to the runtime and leaves DFX off.

    Returns:
        ``None``. Device results are written back into the host tensors in
        place; per-run timing is read from the runtime's ``[STRACE]`` log
        markers (simpler PR #1177), not returned here.
    """
    host_args: list[torch.Tensor] = []
    for arg in coerced_args:
        if not isinstance(arg, torch.Tensor):
            raise TypeError(
                "One-shot distributed execution cannot accept DeviceTensor or StackedDeviceTensor: "
                "their Buffer/provenance must belong to the same prepared DistributedWorker. "
                "Use `with compiled.prepare() as worker:`, allocate with "
                "`worker.alloc_tensor()` / `worker.alloc_stacked_tensor()`, then call `worker.run(...)`."
            )
        host_args.append(arg)

    dc = compiled._distributed_config
    output_dir = compiled.output_dir

    chip_callables, runtime_name, enable_sdma = _assemble_chip_callables(compiled)
    entry_fn, alloc_fn = _load_orch_entry(output_dir)

    # Build tensor mapping from parameter names. One-shot inputs are host
    # torch.Tensor objects and must be in shared memory before the fork.
    param_infos, _, _ = compiled._get_metadata()
    tensors: dict[str, torch.Tensor | DeviceTensor | StackedDeviceTensor] = {}
    for info, arg in zip(param_infos, host_args, strict=True):
        if not arg.is_shared():
            arg.share_memory_()
        tensors[info.name] = arg

    # Pre-fork: allocate HOST-level intermediate tensors so the POSIX
    # shared-memory mappings exist before w.init() forks child processes.
    # ``world_size`` sizes the per-rank comm ordering tokens, which must exist
    # pre-fork like every other host-side intermediate.
    if alloc_fn is not None:
        _call_alloc_intermediates(alloc_fn, tensors, len(dc.device_ids))

    sub_worker_fns = _load_sub_worker_fns(output_dir)
    # The one-shot path cannot supply callbacks; if the program declares any
    # runtime-bound (`...`-body) SubWorker, fail early with a clear message
    # pointing at prepare(callbacks={...}).
    sub_worker_fns = _bind_sub_workers(sub_worker_fns, None, _load_required_callbacks(output_dir))

    num_sub = max(dc.num_sub_workers, len(sub_worker_fns))

    def _run_once(call_config: Any) -> None:
        """One full worker lifecycle (construct → register → init → dispatch → close).

        Each call forks fresh chip workers and closes them, so the per-pass DFX
        collectors — which live in the forked children, not this host process —
        get clean SVM state every pass. That is why the L3 two-pass below does
        not need the subprocess the in-process L2 path uses to dodge the
        ``halHostRegister`` cap (rc 8).

        Construct/register/init run inside the try so a failure in any setup step
        still closes the worker and unlinks the rootinfo temp file.
        """
        w = None
        try:
            w = _construct_worker(
                dc,
                compiled.platform,
                runtime_name,
                num_sub,
                enable_sdma=enable_sdma,
            )
            sub_ids, chip_cids = _register_callables(w, sub_worker_fns, chip_callables)
            # Prewarm with this dispatch's own config so the single run below hits
            # the prebuilt runtime-arena cache instead of paying the ~800ms cold
            # build inside the timed dispatch. No-op without a prebuilt arena.
            w.init(prewarm_config=call_config)
            _dispatch(w, entry_fn, tensors, chip_cids, sub_ids, call_config, len(dc.device_ids))
        except BaseException:  # noqa: BLE001 - cleanup must also run for interruption
            if w is not None:
                _close_local_worker_after_error(w, "one-shot distributed execution")
            raise
        else:
            if w is not None:
                _close_local_worker(w)

    dfx_base = output_dir / "dfx_outputs"
    swimlane = config is not None and config.enable_chip_swimlane > 0

    # Scope DFX artifacts to this run: drop any stale ``rank*/d{k}`` dirs from an
    # earlier (possibly larger) run before the first dispatch writes new ones.
    if config is not None:
        from .runner import _DfxOpts  # noqa: PLC0415

        if _DfxOpts.from_run_config(config).any():
            _clear_dfx_dispatch_dirs(dfx_base)

    if config is not None and config.enable_chip_swimlane > 0 and not compiled.platform.endswith("sim"):
        # Two-pass for clean timing, mirroring the L2 swimlane workflow: dep_gen
        # collection perturbs timing, so the per-dispatch task graph and the kept
        # timing come from separate dispatches.
        _run_l3_swimlane_two_pass(dc, config, dfx_base, _run_once)
    else:
        _run_once(_make_call_config(dc, config, dfx_base=dfx_base))

    # Offline post-pass (reads the per-dispatch deps.json + records on disk).
    if swimlane:
        _collect_l3_swimlane(output_dir, compiled.platform)


def execute_distributed_compiled(
    output_dir: str | Path,
    args: Sequence[torch.Tensor | DeviceTensor | StackedDeviceTensor | ctypes._SimpleCData],
    config: RunConfig | None = None,
    *,
    platform: str | None = None,
    distributed_config: DistributedConfig | None = None,
) -> (
    torch.Tensor
    | DeviceTensor
    | StackedDeviceTensor
    | tuple[torch.Tensor | DeviceTensor | StackedDeviceTensor, ...]
    | None
):
    """Reconstruct a distributed program from ``output_dir`` and run it once.

    The distributed counterpart of :func:`pypto.runtime.execute_compiled`: it
    reconstructs a :class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`
    from an already-compiled build directory (via
    :meth:`DistributedCompiledProgram.from_dir`) and dispatches it once —
    **without** re-running the pypto compile. This is the entry point the
    ``runtime_dir`` replay workflow uses for L3 programs (point it at a
    ``build_output/`` with hand-edited ``.pto``/``.cpp`` and re-run on device).

    Args:
        output_dir: A build directory produced by a prior ``ir.compile`` of a
            distributed (L3+) program (must contain ``distributed_meta.json``).
        args: Host ``torch.Tensor`` arguments matching the orchestrator's
            parameter order (in-place, or input-only for a return-style
            program). Resident tensors require a prepared
            :class:`DistributedWorker` and are rejected here.
        config: Optional per-dispatch :class:`RunConfig`, forwarded to
            ``__call__``. Its per-task ring-sizing overrides size this dispatch's
            runtime ring buffers, and its runtime-diagnostic DFX flags
            (``enable_dump_args`` / ``enable_pmu`` / ``enable_dep_gen`` /
            ``enable_scope_stats`` / ``enable_chip_swimlane``) are written per
            dispatch under ``<output_dir>/dfx_outputs/rank{r}/d{k}/``. Other
            compile-side fields are not consumed on the dispatch path.
        platform: Override the persisted platform (e.g. ``a2a3sim`` → ``a2a3``).
        distributed_config: Override the persisted run config (e.g. a different
            set of ``device_ids``).

    Returns:
        The call result: allocated output tensor(s) for a return-style program,
        otherwise ``None`` (outputs written in place into the passed arguments).
    """
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

    compiled = DistributedCompiledProgram.from_dir(
        output_dir, platform=platform, distributed_config=distributed_config
    )
    return compiled(*args, config=config)


class DistributedWorker(Worker):
    """L3 distributed execution handle: prepare once, dispatch many.

    Holds an initialized simpler ``Worker(level=3)`` plus all setup artifacts
    (chip callables, host_orch entry, sub-worker fns, comm bootstrap) so the
    expensive setup — ``compile_and_assemble``, generated-module loading, Worker
    construction + registration + ``init()`` (fork) — happens exactly once.

    Mirrors the L2 ``with ChipWorker(...)`` reuse block: it exposes device-memory
    helpers (:meth:`malloc`, :meth:`copy_to`, :meth:`copy_from`, :meth:`free`,
    :meth:`alloc_tensor`) so callers can build worker-resident
    :class:`~pypto.runtime.DeviceTensor` buffers that survive across dispatches,
    then call ``rt(*device_args)`` or ``rt.run(compiled, *device_args)``
    repeatedly.

    Per-call IO buffers (inputs **and** outputs) are shared-memory host
    ``torch.Tensor`` objects allocated **before** :meth:`prepare` and reused in
    place across dispatches — the forked chip worker reads/writes them through
    the inherited shared mapping, and outputs are read straight back from the
    tensor (no ``copy_from``). Explicit :meth:`alloc_tensor` /
    :meth:`alloc_stacked_tensor` uploads and copy-backs stage through
    runtime-owned POSIX-shm Buffers, so their CPU-contiguous host endpoints may
    be ordinary tensors created after ``prepare``.

    ``inherited_host_tensors`` keeps such a host range alive for the worker's
    lifetime *and* is a caller guarantee about it: listing a tensor asserts that
    its backing is visible across processes — a ``MAP_SHARED`` mapping, whether
    torch's own shared memory or an external file mapping — and that the mapping
    stays valid for as long as the worker lives. Listed ranges are then named in
    place instead of staged, which is what removes the per-copy ``memmove``.
    Passing a ``MAP_PRIVATE`` backing is **unsupported**: copy-on-write leaves the
    child reading its pre-fork snapshot, so uploads may carry stale or incorrect
    data. PyPTO cannot verify the guarantee — ``torch.is_shared()`` returns
    ``False`` for valid external ``MAP_SHARED`` mappings — so it warns once at
    prepare time for the tensors it cannot confirm and proceeds. Listing a tensor
    does not make it a valid direct dispatch argument.

    ``callbacks`` binds a caller-supplied callable to a SubWorker by name — e.g.
    a real sampling closure. Abstract SubWorkers (declared with a ``...`` body)
    are runtime-bound callback points and MUST be supplied here; a missing
    binding raises ``ValueError`` at prepare time. A callback may also replace a
    concrete SubWorker's generated body. Each name must be a sub-worker the
    program declares; an unknown name raises ``ValueError``.
    (``sub_worker_overrides`` is a deprecated alias for ``callbacks``.)

    **Multi-program dispatch.** Pass a sequence of compatible
    :class:`DistributedCompiledProgram` objects (or use
    ``compiled.prepare(extra_compiled=[...])``) to prepare several HOST programs
    on one L3 worker. Each program's chip callables, sub-worker functions,
    orchestration entry, base tensors, and parameter metadata are registered
    independently and selected at dispatch via ``rt.run(compiled, *args)``. This
    is what serving needs: prefill and decode are separate JIT HOST programs that
    must share one worker lifecycle and one worker-resident
    :class:`DeviceTensor` KV cache. Programs must agree on platform, runtime, and
    device ids; a mismatch raises ``ValueError``. The ``rt(*args)`` shortcut is
    only for single-program workers — in multi-program mode it raises
    ``TypeError`` since the target program is ambiguous.

    Obtain via :meth:`DistributedCompiledProgram.prepare`. Use as a context
    manager (recommended) or call :meth:`close` when done::

        host_x = torch.zeros(seq, 4096, dtype=torch.float16).share_memory_()
        host_out = torch.zeros(seq, 4096, dtype=torch.float16).share_memory_()
        host_w = load_weight().share_memory_()      # before prepare()
        with compiled.prepare() as rt:
            weight = rt.alloc_tensor(host_w.shape, host_w.dtype, init=host_w)
            for step in steps:
                host_x.copy_(next_input(step))      # update in place
                rt(host_x, weight, host_out)        # host shm IO + resident weight
                consume(host_out)                   # read directly
            rt.free_tensor(weight)
    """

    __test__ = False

    def __init__(
        self,
        compiled: DistributedCompiledProgram | Sequence[DistributedCompiledProgram],
        config: RunConfig | None = None,
        *,
        persistent: bool = False,
        reset_persistent_windows: bool | None = None,
        callbacks: dict[str, Callable[..., Any]] | None = None,
        sub_worker_overrides: dict[str, Callable[..., Any]] | None = None,
        inherited_host_tensors: Sequence[torch.Tensor] | None = None,
        startup_timeout_s: float | None = None,
    ) -> None:
        super().__init__()  # initialize Worker ABC state (_owned_tensors)
        # Simpler owns allocations as Buffer objects. PyPTO continues exposing
        # raw pointers from its public memory API and resolves them through this
        # table at every control-plane operation. The retained handle is also
        # attached to DeviceTensor for address-free wire TaskArgs.
        self._device_buffers: dict[tuple[int, int], Any] = {}
        callbacks = _coalesce_callbacks(callbacks, sub_worker_overrides)
        reset_persistent_windows = _resolve_persistent_window_reset(persistent, reset_persistent_windows)
        inherited = tuple(inherited_host_tensors) if inherited_host_tensors is not None else ()
        for tensor in inherited:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "DistributedWorker inherited_host_tensors entries must be torch.Tensor objects, "
                    f"got {type(tensor).__name__}."
                )
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(
                    "DistributedWorker inherited_host_tensors must be contiguous CPU tensors; "
                    f"got device={tensor.device} shape={tuple(tensor.shape)}."
                )
        self._inherited_host_tensors = inherited
        # `copy_to` / `copy_from` stage through a simpler-owned shm Buffer so an ordinary
        # post-fork tensor is a legal endpoint. That relaxation costs one full copy of the
        # payload, which a fork-inherited range visible across processes does not need:
        # parent and child see the same pages, so it can be named in place. Record the
        # extents here, where the tensors are still in hand, since an address alone says
        # nothing about its backing.
        #
        # Listing a tensor is the caller's guarantee that the backing is cross-process
        # visible (see the class docstring). It is not inferred, because no portable check
        # exists: `torch.is_shared()` answers "is this storage a torch shared-memory
        # allocation", which is a different question. A read-only MAP_SHARED file mapping
        # built with `mmap` + `from_numpy` is genuinely shared at the OS level and still
        # reports False, so inference would reject exactly the case that benefits most,
        # while reading /proc/self/maps would tie this file to Linux and the simulator also
        # runs on macOS.
        self._inherited_host_spans: tuple[tuple[int, int], ...] = tuple(
            (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            )
            for tensor in inherited
        )
        # `is_shared()` is kept as a one-way signal: True confirms a torch-managed shared
        # backing, False is inconclusive. Warn once rather than per tensor, and never
        # reject or silently fall back to staging — falling back would hide the cost this
        # facility exists to remove, and rejecting would refuse the valid external mapping.
        unverifiable = sum(1 for tensor in inherited if not tensor.is_shared())
        if unverifiable:
            warnings.warn(
                f"DistributedWorker: {unverifiable} of {len(inherited)} inherited_host_tensors "
                "cannot be verified as cross-process visible (torch.is_shared() is False, which "
                "is inconclusive for external MAP_SHARED mappings). Naming them relies on the "
                "caller's guarantee; a MAP_PRIVATE backing yields stale or incorrect data.",
                RuntimeWarning,
                stacklevel=3,
            )
        self._buffer_owner_id: bytes | None = None
        self._buffer_id_seq = 0
        # One identity per host *range*, not per copy. Both properties this buys are load
        # bearing: a consumer's ImportRegistry only drops an entry when the owner releases the
        # Buffer, which a named copy never does, so minting per copy would leave one permanent
        # ImportedBuffer per copy in every chip child — unbounded for a per-step D2H read-back.
        # And re-copying the same range must reuse its identity, because one identity may name
        # only one backing: `materialize` refuses a second descriptor for an identity it has
        # already handed out.
        self._named_identities: dict[tuple[int, int, bool], tuple[bytes, int]] = {}
        # Minting is not atomic (`+=` is load/add/store, and the lazy owner mint is
        # check-then-act) while `alloc_stacked_tensor` runs one thread per chip through this
        # path, so the cache and the counter are guarded together.
        self._named_identity_mu = threading.Lock()
        self._persistent = bool(persistent)
        self._reset_persistent_windows = reset_persistent_windows
        self._persistent_error: BaseException | None = None
        self._persistent_error_reported = False
        self._persistent_domains_by_program: dict[str, dict[str, tuple[tuple[Any, ...], Any]]] = {}
        self._dispatch_submit_mu = threading.Lock()
        self._dispatch_cv = threading.Condition()
        self._dispatch_frames = [_DispatchFrame(slot_id) for slot_id in range(2)]
        self._active_dispatch_handles: set[DistributedRunHandle] = set()
        self._next_dispatch_id = 1
        self._accepting_dispatches = False
        self._closing = False

        programs = list(compiled) if isinstance(compiled, Sequence) else [compiled]
        if not programs:
            raise ValueError("DistributedWorker requires at least one compiled program")

        primary = programs[0]
        self.dc = primary._distributed_config
        self._compiled = primary  # primary program: dispatched by ``rt(*args)``
        # In multi-program mode ``rt(*args)`` is ambiguous (which program?), so it
        # is disabled — callers must pick explicitly via ``rt.run(compiled, ...)``.
        self._multi_program = len(programs) > 1
        # Per-program dispatch state keyed by the program object (not id(prog)):
        # the dict keeps every prepared program alive for the worker's lifetime,
        # so there is no id()-reuse hazard from a GC'd program. ``run(compiled,
        # ...)`` looks the selected program up here; ``__call__`` uses ``_compiled``.
        self._states: dict[DistributedCompiledProgram, dict[str, Any]] = {}

        # Wrap setup so a failure at any step still releases the worker and the
        # comm rootinfo temp file. ``self.close()`` can't be used here — it reads
        # ``self._closed``, which isn't set until setup completes — so cleanup is
        # inlined and guarded against the partially-constructed state.
        self._w: Any = None
        try:
            # Phase 1 (pre-fork): load + validate every program's artifacts and
            # allocate its HOST-level scratch tensors. All shared-memory mappings
            # must exist before ``init()`` forks so the children inherit them.
            runtime_name: str | None = None
            num_sub = 0
            enable_sdma = False
            # (program, chip_callables, sub_worker_fns) deferred to phase 2 so all
            # registrations happen on one already-constructed worker.
            loaded: list[tuple[DistributedCompiledProgram, dict[str, Any], dict[str, Any]]] = []
            # A callback applies to whichever prepared programs declare that
            # sub-worker (e.g. a shared sampler used by both prefill and decode);
            # programs with different sub-worker sets are fine. We track which
            # callback names were consumed so a typo that matches no program is
            # still reported (see the post-loop check), while each program's own
            # required-callback manifest is enforced per program.
            callbacks = callbacks or {}
            consumed: set[str] = set()
            for program_index, prog in enumerate(programs):
                self._check_compatible(prog, primary)
                chip_callables, prog_runtime, prog_enable_sdma = _assemble_chip_callables(prog)
                runtime_name = self._unify_runtime(runtime_name, prog_runtime)
                enable_sdma = enable_sdma or prog_enable_sdma
                entry_fn, alloc_fn = _load_orch_entry(prog.output_dir)
                loaded_subs = _load_sub_worker_fns(prog.output_dir)
                prog_callbacks = {name: fn for name, fn in callbacks.items() if name in loaded_subs}
                consumed |= set(prog_callbacks)
                sub_worker_fns = _bind_sub_workers(
                    loaded_subs, prog_callbacks, _load_required_callbacks(prog.output_dir)
                )
                num_sub = max(num_sub, prog._distributed_config.num_sub_workers, len(sub_worker_fns))
                base_tensor_frames: list[dict[str, Any]] = []
                for _frame in self._dispatch_frames:
                    base_tensors: dict[str, Any] = {}
                    if alloc_fn is not None:
                        _call_alloc_intermediates(
                            alloc_fn, base_tensors, len(prog._distributed_config.device_ids)
                        )
                    base_tensor_frames.append(base_tensors)
                self._states[prog] = {
                    "entry_fn": entry_fn,
                    "base_tensor_frames": tuple(base_tensor_frames),
                    "call_config": _make_call_config(prog._distributed_config),
                    "param_infos": tuple(prog._get_metadata()[0]),
                    "device_nums": len(prog._distributed_config.device_ids),
                    "persistent_id": f"p{program_index}",
                }
                if self._persistent and "_domain_provider" not in inspect.signature(entry_fn).parameters:
                    raise ValueError(
                        "persistent distributed execution requires regenerated host orchestration "
                        "with the internal _domain_provider hook"
                    )
                loaded.append((prog, chip_callables, sub_worker_fns))

            unconsumed = sorted(set(callbacks) - consumed)
            if unconsumed:
                raise ValueError(f"callbacks names {unconsumed} are not sub-workers of any prepared program.")

            if runtime_name is None:  # unreachable: programs is non-empty
                raise RuntimeError("failed to resolve distributed runtime")

            # Phase 2: one worker for all programs. Register every program's
            # callables before ``init()`` so the L3 fork inherits the whole
            # registry via COW; each program keeps its own cids in its state.
            self._w = _construct_worker(
                self.dc,
                primary.platform,
                runtime_name,
                num_sub,
                enable_sdma=enable_sdma,
                startup_timeout_s=startup_timeout_s,
            )
            from .tensor_arg import bind_tensor_arg_owner  # noqa: PLC0415

            bind_tensor_arg_owner(self._w, self)
            self._validate_persistent_runtime_hooks()
            for prog, chip_callables, sub_worker_fns in loaded:
                sub_ids, chip_cids = _register_callables(self._w, sub_worker_fns, chip_callables)
                self._states[prog]["sub_ids"] = sub_ids
                self._states[prog]["chip_cids"] = chip_cids

            # Prewarm the prebuilt runtime-arena cache so the first run() hits it
            # instead of paying the ~800ms cold build. The cache is single-slot per
            # worker: exactly one ring sizing is prewarmed — ``config``'s when given
            # (built exactly as ``run()`` builds it, so the sizing keys match), else
            # the primary program's baseline. No-op without a prebuilt arena.
            prewarm_cc = self._states[primary]["call_config"]
            if config is not None:
                prewarm_cc = _make_call_config(
                    primary._distributed_config, config, dfx_base=primary.output_dir / "dfx_outputs"
                )
            self._w.init(prewarm_config=prewarm_cc)

            # ``Worker.init()`` eagerly starts the chip/sub-worker hierarchy, so
            # the device-memory API is ready before the first dispatch without a
            # separate call into Simpler's private startup implementation.
        except BaseException:  # noqa: BLE001 - partially built Workers still require cleanup
            if self._w is not None:
                _close_local_worker_after_error(self._w, "DistributedWorker construction")
            raise

        self._closed = False
        self._close_complete = False
        self._accepting_dispatches = True
        # Live RegistrationHandles so close() can mark them closed. WeakSet
        # so handles that drop out of scope first don't pin DistributedWorker.
        self._handles: weakref.WeakSet[Any] = weakref.WeakSet()

    @staticmethod
    def _check_compatible(prog: DistributedCompiledProgram, primary: DistributedCompiledProgram) -> None:
        """Reject programs that cannot share one L3 worker with *primary*."""
        if prog.platform != primary.platform:
            raise ValueError(
                "DistributedWorker multi-program mode requires the same platform: "
                f"{primary.platform!r} != {prog.platform!r}"
            )
        primary_ids = list(primary._distributed_config.device_ids)
        if list(prog._distributed_config.device_ids) != primary_ids:
            raise ValueError(
                "DistributedWorker multi-program mode requires the same device_ids: "
                f"{primary_ids} != {list(prog._distributed_config.device_ids)}"
            )

    @staticmethod
    def _unify_runtime(runtime_name: str | None, prog_runtime: str) -> str:
        """Return the shared runtime name, rejecting a per-program mismatch."""
        if runtime_name is None:
            return prog_runtime
        if runtime_name != prog_runtime:
            raise ValueError(
                "DistributedWorker multi-program mode requires the same runtime: "
                f"{runtime_name!r} != {prog_runtime!r}"
            )
        return runtime_name

    def _acquire_dispatch_frame(self) -> tuple[_DispatchFrame, int]:
        """Reserve one of two metadata frames, draining the oldest on pressure."""
        while True:
            with self._dispatch_cv:
                if not self._accepting_dispatches:
                    raise RuntimeError("DistributedWorker.submit() called while the worker is closing")
                frame = next((candidate for candidate in self._dispatch_frames if not candidate.in_use), None)
                if frame is not None:
                    if frame.cleanup:
                        raise RuntimeError(
                            "DistributedWorker dispatch frame retained stale cleanup callbacks"
                        )
                    frame.in_use = True
                    dispatch_id = self._next_dispatch_id
                    self._next_dispatch_id += 1
                    return frame, dispatch_id
                if not self._active_dispatch_handles:
                    raise RuntimeError(
                        "DistributedWorker dispatch frames are occupied without owning handles"
                    )
                oldest = min(self._active_dispatch_handles, key=lambda handle: handle._dispatch_id)
            try:
                oldest.result()
            except BaseException:  # noqa: BLE001 - the handle owner still observes its cached outcome
                pass

    @staticmethod
    def _run_dispatch_cleanup(cleanup: list[Callable[[], None]]) -> None:
        """Run and consume terminal callbacks, preserving the first failure."""
        first_error: BaseException | None = None
        while cleanup:
            callback = cleanup.pop()
            try:
                callback()
            except BaseException as exc:  # noqa: BLE001 - every cleanup must still be attempted
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _release_unpublished_dispatch_frame(self, frame: _DispatchFrame) -> None:
        """Return a frame whose submission failed before handle publication."""
        try:
            self._run_dispatch_cleanup(frame.cleanup)
        finally:
            with self._dispatch_cv:
                frame.tensors.clear()
                frame.keepalive.clear()
                frame.handle = None
                frame.in_use = False
                self._dispatch_cv.notify_all()

    def _discard_unsubmitted_dispatch_handle(
        self,
        handle: DistributedRunHandle,
        frame: _DispatchFrame,
    ) -> None:
        """Discard a provisional handle after submission failed before acceptance."""
        with self._dispatch_cv:
            self._active_dispatch_handles.discard(handle)
            if frame.handle is handle:
                frame.handle = None
        self._release_unpublished_dispatch_frame(frame)

    def _accepted_native_handles(self) -> set[Any] | None:
        """Snapshot Simpler's accepted set for interruption recovery."""
        handles = getattr(self._w, "_accepted_run_handles", None)
        lifecycle_cv = getattr(self._w, "_hierarchical_start_cv", None)
        if not isinstance(handles, set) or lifecycle_cv is None:
            return None
        with lifecycle_cv:
            return set(handles)

    def _recover_accepted_native_handle(self, before: set[Any] | None) -> Any | None:
        """Recover the sole handle accepted while one serialized submit ran."""
        if before is None:
            return None
        after = self._accepted_native_handles()
        if after is None:
            return None
        accepted = after - before
        if len(accepted) != 1:
            return None
        return next(iter(accepted))

    def _retire_dispatch_handle(
        self,
        handle: DistributedRunHandle,
        frame: _DispatchFrame,
        error: BaseException | None,
    ) -> None:
        """Release one terminal handle's frame and publish persistent failure."""
        if error is not None and self._persistent and self._persistent_error is None:
            self._persistent_error = error
            self._persistent_error_reported = True
        with self._dispatch_cv:
            self._active_dispatch_handles.discard(handle)
            if frame.handle is handle:
                frame.tensors.clear()
                frame.keepalive.clear()
                frame.cleanup.clear()
                frame.handle = None
                frame.in_use = False
            self._dispatch_cv.notify_all()

    @staticmethod
    def _remember_close_error(
        first_error: BaseException | None,
        error: BaseException,
    ) -> BaseException:
        """Retain the first close failure and chain one teardown failure."""
        if first_error is None:
            return error
        if first_error.__context__ is None:
            first_error.__context__ = error
        return first_error

    @staticmethod
    def _persistent_domain_spec(kwargs: dict[str, Any]) -> tuple[Any, ...]:
        """Build a stable identity tuple for one generated CommDomain request."""
        buffers = tuple(
            (buffer.name, buffer.dtype, int(buffer.count), int(buffer.nbytes))
            for buffer in kwargs.get("buffers", ())
        )
        return (
            tuple(int(worker) for worker in kwargs["workers"]),
            int(kwargs["window_size"]),
            buffers,
        )

    def _validate_persistent_runtime_hooks(self) -> None:
        """Fail before worker initialization when Simpler cannot retain domains."""
        if not self._persistent:
            return
        live_domains = getattr(self._w, "_live_domains", None)
        missing = []
        if not isinstance(live_domains, dict):
            missing.append("_live_domains")
        if not hasattr(self._w, "_building_run_resources"):
            missing.append("_building_run_resources")
        if missing:
            raise RuntimeError(
                "persistent distributed execution requires Simpler's private retention hooks: "
                + ", ".join(missing)
            )

    def _reset_persistent_domains(
        self,
        orch: Any,
        domains: dict[str, tuple[tuple[Any, ...], Any]],
        reset_buffers: dict[int, Any],
    ) -> None:
        """Restore every named buffer in retained domains to its initial zero state.

        The host Buffers are created before ``Worker.run`` and released after it;
        this callback only issues copies while Simpler holds its submit lock.
        """
        for domain_name, (spec, handle) in domains.items():
            _workers, window_nbytes, buffer_specs = spec
            for worker_id in handle.workers:
                actual_window_size = int(handle[worker_id].actual_window_size)
                if actual_window_size != int(window_nbytes):
                    raise RuntimeError(
                        f"persistent CommDomain {domain_name!r} changed window size on worker {worker_id}: "
                        f"{actual_window_size} != {window_nbytes}"
                    )
            for buffer_name, _dtype, _count, buffer_nbytes in buffer_specs:
                nbytes = int(buffer_nbytes)
                zero_buffer = reset_buffers[nbytes]
                for worker_id in handle.workers:
                    dst_buffer = handle[worker_id].buffers[buffer_name]
                    if int(dst_buffer.nbytes) != nbytes:
                        raise RuntimeError(
                            f"persistent CommDomain buffer {buffer_name!r} changed size: "
                            f"{dst_buffer.nbytes} != {nbytes}"
                        )
                    orch.copy_to(dst_buffer, zero_buffer)

    def _prepare_persistent_reset_buffers(
        self,
        domains: dict[str, tuple[tuple[Any, ...], Any]],
        reset_buffers: dict[int, Any],
    ) -> None:
        """Create zero-filled host Buffers, shared by every named buffer of the same size."""
        for domain_name, (spec, _handle) in domains.items():
            _workers, window_nbytes, buffer_specs = spec
            named_nbytes = sum(int(buffer_spec[3]) for buffer_spec in buffer_specs)
            if named_nbytes != int(window_nbytes):
                raise RuntimeError(
                    f"persistent CommDomain {domain_name!r} reset requires named buffers "
                    "to cover its window: "
                    f"named bytes {named_nbytes} != window size {window_nbytes}"
                )
            for _name, _dtype, _count, buffer_nbytes in buffer_specs:
                nbytes = int(buffer_nbytes)
                if nbytes in reset_buffers:
                    continue
                zero_buffer = self._w.create_buffer(nbytes)
                reset_buffers[nbytes] = zero_buffer
                ctypes.memset(int(zero_buffer.base), 0, nbytes)

    def _release_persistent_reset_buffers(self, reset_buffers: dict[int, Any]) -> None:
        """Release request-local reset buffers after the run callback unlocks."""
        first_error: BaseException | None = None
        for buffer in reset_buffers.values():
            try:
                self._w.release_buffer(buffer)
            except BaseException as exc:  # noqa: BLE001 - best-effort release of every staging buffer
                if first_error is None:
                    first_error = exc
        reset_buffers.clear()
        if first_error is not None:
            raise first_error

    def _detach_persistent_domain(self, handle: Any) -> None:
        """Transfer one CommDomain from the current run to Worker ownership.

        Simpler records a newly allocated domain in both the current
        ``_RunResources.live_domains`` journal and ``Worker._live_domains``.
        Remove only the run-local claim: the global entry must remain reachable
        so ``Worker.close()`` can reclaim it if request finalization is
        interrupted before the run is retired.
        """
        live_domains = getattr(self._w, "_live_domains", None)
        resources = getattr(self._w, "_building_run_resources", None)
        run_live_domains = getattr(resources, "live_domains", None)
        domain_lock = getattr(resources, "domain_lock", None)
        if (
            not isinstance(live_domains, dict)
            or not isinstance(run_live_domains, dict)
            or domain_lock is None
        ):
            raise RuntimeError(
                "persistent distributed execution requires Simpler's active per-run CommDomain journal"
            )
        with domain_lock:
            if bool(getattr(resources, "retired", False)):
                raise RuntimeError("persistent CommDomain cannot be retained from an already-retired run")
            if live_domains.get(handle.name) is not handle or run_live_domains.get(handle.name) is not handle:
                raise RuntimeError(
                    "persistent distributed execution could not transfer the CommDomain's run-local ownership"
                )
            del run_live_domains[handle.name]

    def _release_persistent_domains(
        self,
        domains_by_program: dict[str, dict[str, tuple[tuple[Any, ...], Any]]],
    ) -> None:
        """Release retained domains after the last request run-fence."""
        handles = [
            handle
            for program_domains in reversed(tuple(domains_by_program.values()))
            for _spec, handle in reversed(tuple(program_domains.values()))
        ]
        for handle in handles:
            handle.release()
        not_freed = [handle for handle in handles if not bool(getattr(handle, "freed", False))]
        if not_freed:
            names = ", ".join(repr(handle.name) for handle in not_freed)
            raise RuntimeError(f"persistent CommDomain release did not free: {names}")

    def _submit_persistent(
        self,
        state: dict[str, Any],
        tensors: dict[str, Any],
        call_config: Any,
        keepalive: list[Any],
        cleanup: list[Callable[[], None]],
    ) -> Any:
        """Submit one persistent request directly through Simpler."""
        self._raise_persistent_error()
        domains_by_program = self._persistent_domains_by_program
        program_id = str(state["persistent_id"])
        program_domains = domains_by_program.get(program_id)
        reset_buffers: dict[int, Any] = {}

        def release_reset_buffers() -> None:
            self._release_persistent_reset_buffers(reset_buffers)

        # Buffer creation is direct control and must happen before Simpler
        # enters the serialized graph callback. Its release is registered on
        # the PyPTO dispatch frame and therefore runs only after the async
        # native handle reaches a terminal state (or submission is rejected).
        cleanup.append(release_reset_buffers)
        if program_domains and self._reset_persistent_windows:
            self._prepare_persistent_reset_buffers(program_domains, reset_buffers)

        def run_request(
            orch: Any,
            _args: Any,
            _config: Any,
        ) -> None:
            def domain_provider(**kwargs: Any) -> _RetainedDomainLease:
                generated_name = str(kwargs["name"])
                program_domains = domains_by_program.setdefault(program_id, {})
                spec = self._persistent_domain_spec(kwargs)
                existing = program_domains.get(generated_name)
                if existing is None:
                    runtime_kwargs = dict(kwargs)
                    runtime_kwargs["name"] = f"{program_id}:{generated_name}"
                    handle = orch.allocate_domain(**runtime_kwargs)
                    self._detach_persistent_domain(handle)
                    program_domains[generated_name] = (spec, handle)
                else:
                    prior_spec, handle = existing
                    if spec != prior_spec:
                        raise ValueError(
                            f"persistent CommDomain {generated_name!r} changed specification "
                            f"for program {program_id}"
                        )
                return _RetainedDomainLease(handle)

            program_domains = domains_by_program.get(program_id)
            if program_domains and self._reset_persistent_windows:
                self._reset_persistent_domains(orch, program_domains, reset_buffers)
            _reset_dfx_dispatch_state(orch, state["chip_cids"])
            state["entry_fn"](
                orch,
                None,
                call_config,
                tensors=tensors,
                callables=state["chip_cids"],
                sub_ids=state["sub_ids"],
                _keep=keepalive,
                world_size=state["device_nums"],
                _domain_provider=domain_provider,
            )

        try:
            return self._w.submit(run_request)
        except Exception as exc:
            self._persistent_error = exc
            self._persistent_error_reported = True
            raise

    def _raise_persistent_error(self) -> None:
        """Raise the terminal persistent failure and mark it as delivered."""
        if self._persistent_error is not None:
            self._persistent_error_reported = True
            raise self._persistent_error

    def _submit_prepared_native(
        self,
        state: dict[str, Any],
        tensors: dict[str, Any],
        call_config: Any,
        keepalive: list[Any],
        cleanup: list[Callable[[], None]],
    ) -> Any:
        """Submit through either the ordinary or persistent prepared path."""
        if self._persistent:
            return self._submit_persistent(state, tensors, call_config, keepalive, cleanup)
        return _submit_dispatch(
            self._w,
            state["entry_fn"],
            tensors,
            state["chip_cids"],
            state["sub_ids"],
            call_config,
            state["device_nums"],
            keepalive,
        )

    def _submit_native_dispatch(
        self,
        handle: DistributedRunHandle,
        frame: _DispatchFrame,
        state: dict[str, Any],
        tensors: dict[str, Any],
        call_config: Any,
    ) -> None:
        """Install or recover the native handle for one published frame."""
        accepted_before = self._accepted_native_handles()
        native_handle: Any | None = None
        try:
            native_handle = self._submit_prepared_native(
                state,
                tensors,
                call_config,
                frame.keepalive,
                frame.cleanup,
            )
        except BaseException as exc:
            if native_handle is None:
                native_handle = self._recover_accepted_native_handle(accepted_before)
            if native_handle is None:
                if self._persistent and self._persistent_error is None:
                    self._persistent_error = exc
                    self._persistent_error_reported = True
                self._discard_unsubmitted_dispatch_handle(handle, frame)
            else:
                handle._native_handle = native_handle
            raise
        handle._native_handle = native_handle

    def _dispatch_prepared(
        self,
        state: dict[str, Any],
        tensors: dict[str, Any],
        call_config: Any,
        keepalive: list[Any] | None = None,
    ) -> None:
        """Blocking compatibility composition used by diagnostic two-pass runs."""
        cleanup: list[Callable[[], None]] = []
        error: BaseException | None = None
        try:
            native_handle = self._submit_prepared_native(
                state,
                tensors,
                call_config,
                [] if keepalive is None else keepalive,
                cleanup,
            )
            native_handle.result()
        except BaseException as exc:  # noqa: BLE001 - preserve the dispatch failure across cleanup
            error = exc
        try:
            self._run_dispatch_cleanup(cleanup)
        except BaseException as exc:  # noqa: BLE001 - cleanup remains mandatory on failure
            if error is None:
                error = exc
            elif error.__context__ is None:
                error.__context__ = exc
        if error is not None:
            raise error

    # ------------------------------------------------------------------
    # Device memory primitives
    #
    # Simpler's public Worker methods own lifecycle admission and serialize
    # device control against close() and in-flight hierarchical runs. Keep all
    # memory traffic on that public surface instead of bypassing its lease via
    # the private Orchestrator facade.
    # ------------------------------------------------------------------

    def malloc(self, nbytes: int, *, worker_id: int = 0) -> int:
        """Allocate ``nbytes`` on chip *worker_id*; returns a device pointer."""
        self._require_open("malloc")
        if not isinstance(nbytes, int) or nbytes <= 0:
            raise ValueError(f"nbytes must be a positive int, got {nbytes!r}")
        from simpler.task_interface import DataType  # pyright: ignore[reportMissingImports]  # noqa: PLC0415

        handle = self._w.alloc_child_tensor(worker_id, (nbytes,), DataType.UINT8)
        try:
            ptr = int(handle.base)
        except (AttributeError, TypeError, ValueError) as e:
            with suppress(Exception):
                self._w.free(handle)
            raise TypeError("simpler Worker.alloc_child_tensor() must return a Buffer") from e
        if ptr <= 0:
            with suppress(Exception):
                self._w.free(handle)
            raise ValueError(f"simpler Worker.alloc_child_tensor() returned invalid Buffer base {ptr!r}")
        self._device_buffers[(worker_id, ptr)] = handle
        return ptr

    def _device_buffer(self, ptr: int, worker_id: int, op: str) -> Any:
        try:
            return self._device_buffers[(worker_id, ptr)]
        except KeyError as e:
            raise ValueError(
                f"DistributedWorker.{op}() requires the allocation base returned by "
                f"this worker's malloc(..., worker_id={worker_id}); got 0x{ptr:x}. "
                "PyPTO cannot safely reconstruct an owner Buffer for an interior pointer."
            ) from e

    def _buffer_for_ptr(self, ptr: int, *, worker_id: int = 0) -> Any:
        return self._device_buffer(ptr, worker_id, "alloc_tensor")

    def free(self, ptr: int, *, worker_id: int = 0) -> None:
        """Release a pointer previously returned by :meth:`malloc`."""
        self._require_open("free")
        handle = self._device_buffer(ptr, worker_id, "free")
        self._w.free(handle)
        del self._device_buffers[(worker_id, ptr)]

    def committed_device_memory(self, worker_id: int = 0) -> int:
        """Total device HBM (bytes) committed by chip *worker_id*'s ``MemoryAllocator``
        (tensors + pooled arenas + runtime buffers). Routes through the underlying
        simpler ``Worker(level=3)`` facade, which forwards a
        ``CTRL_COMMITTED_DEVICE_MEMORY`` query to the forked chip child. Sum across
        ``worker_id``s for a multi-chip total."""
        self._require_open("committed_device_memory")
        if self._w is None:
            return 0
        return int(self._w.committed_device_memory(worker_id))

    def device_memory_info(self, worker_id: int = 0) -> tuple[int, int]:
        """Free and total device HBM (bytes) as the driver sees it, for the device
        chip *worker_id* runs on. Routes through the underlying simpler
        ``Worker(level=3)`` facade, which forwards the query to the forked chip child.

        This is a device-wide snapshot and answers a different question from
        :meth:`committed_device_memory`, which reports only what this worker's own
        ``MemoryAllocator`` has committed. Anything else on the card -- another
        process, another worker, the driver itself -- moves ``free_bytes`` without
        moving that committed total.

        Unlike :meth:`committed_device_memory`, a failed query is never softened
        into a zero here: a caller sizing a KV cache from a fabricated ``(0, 0)``
        would silently under-allocate, so the underlying error propagates instead.
        Simulator backends synthesize no device-wide memory and raise
        ``NotImplementedError``.

        Returns:
            ``(free_bytes, total_bytes)``, both as Python ints.
        """
        self._require_open("device_memory_info")
        info = self._w.device_memory_info(worker_id)
        return int(info.free_bytes), int(info.total_bytes)

    def _buffer_identity_for(self, host_ptr: int, nbytes: int, *, writing: bool) -> tuple[bytes, int]:
        """Return the stable ``(owner_instance_id, buffer_id)`` naming this host range.

        Keyed on the range rather than counted per call, so a range copied N times keeps one
        identity: the consumer's ``ImportRegistry`` refuses a second descriptor for an identity
        it has already materialized, and it only drops an entry when the owner releases the
        Buffer — which the named path never does, since it hands out the caller's own mapping.
        Per-copy identities would therefore both collide on a re-copy and grow a child's
        registry without bound. Distinct sub-ranges of one registered tensor still get distinct
        identities, which is what a sharded upload needs.

        The direction is part of the key because it decides the descriptor's ``access``: a
        source is named ``READ`` and a destination ``READWRITE``. One identity may name only
        one backing, so a range copied in both directions under a single identity would have
        ``materialize`` reject the second descriptor for the changed access. Two identities per
        range is still bounded — the property that matters is that it does not grow per copy.

        Held under a lock because ``alloc_stacked_tensor`` drives this from one thread per chip.
        """
        from simpler.buffer import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            mint_owner_instance_id,
        )

        key = (int(host_ptr), int(nbytes), bool(writing))
        with self._named_identity_mu:
            cached = self._named_identities.get(key)
            if cached is not None:
                return cached
            owner = self._buffer_owner_id
            if owner is None:
                owner = mint_owner_instance_id()
                self._buffer_owner_id = owner
            self._buffer_id_seq += 1
            identity = (owner, self._buffer_id_seq)
            self._named_identities[key] = identity
            return identity

    def _named_host_buffer(self, host_ptr: int, nbytes: int, *, writing: bool = False) -> Any:
        """Name a fork-inherited MAP_SHARED host range in place, or ``None`` to stage it.

        Only memory registered through ``inherited_host_tensors`` can be named at all: it
        predates the fork, so every child holds it at the same virtual address and the copy
        needs no staging buffer and no host-side memcpy. A tensor allocated after
        ``prepare`` has no mapping in the child, so the caller stages it.

        **The list is the guarantee.** Listing a tensor in ``inherited_host_tensors``
        asserts that its backing is visible across processes and stays valid for the
        worker's lifetime, so any range inside a listed span is named in place. Nothing is
        inferred and nothing is rejected: ``torch.is_shared()`` is a one-way signal, so
        deciding from it would refuse a read-only ``MAP_SHARED`` file mapping — which
        reports ``False`` while being perfectly safe to name — and reading
        ``/proc/self/maps`` would tie this to Linux while the simulator also runs on macOS.
        A ``MAP_PRIVATE`` backing is unsupported here rather than handled: the child keeps
        reading its pre-fork snapshot, so the caller would upload stale bytes.

        Writability is left to the MMU rather than tracked here, because visibility and
        writability are separate properties and one caller guarantee only covers the first.
        A range mapped ``MAP_SHARED`` from a read-only fd is genuinely shared and still
        faults on write, so a read-back into it fails immediately instead of corrupting
        anything. A named source is granted ``READ`` and only a destination ``READWRITE``,
        so the ABI is never told a consumer may write memory it may not.

        Each range is wrapped on its own rather than offset into a whole-tensor Buffer,
        because a Buffer carries no offset: a shard's address is interior to its stacked
        tensor, so per-range wrapping is what keeps one shard's copy from moving the whole
        stack.
        """
        from simpler.buffer import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            AccessMode,
            BackendKind,
            wrap_fork_inherited,
        )

        end = host_ptr + nbytes
        for start, stop in self._inherited_host_spans:
            if host_ptr < start or end > stop:
                continue
            owner, buffer_id = self._buffer_identity_for(host_ptr, nbytes, writing=writing)
            return wrap_fork_inherited(
                host_ptr,
                nbytes,
                owner,
                buffer_id,
                # A source is read by the child and nothing more, so say so: naming a read-only
                # mapping READWRITE would tell the ABI the consumer may write memory that faults
                # on a write, which is the same class of misdeclaration this whole option exists
                # to remove. FORK_SHM accepts any access, so READ is expressible here.
                access=AccessMode.READWRITE if writing else AccessMode.READ,
                backend_kind=BackendKind.FORK_SHM,
            )
        return None

    def copy_to(self, dst_dev_ptr: int, src_host_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        """H2D copy: ``nbytes`` from host *src_host_ptr* to device *dst_dev_ptr*."""
        self._require_open("copy_to")
        dst = self._device_buffer(dst_dev_ptr, worker_id, "copy_to")
        if not isinstance(nbytes, int) or nbytes <= 0:
            raise ValueError(f"nbytes must be a positive int, got {nbytes!r}")
        src = self._named_host_buffer(int(src_host_ptr), int(nbytes))
        if src is not None:
            self._w.copy_to(dst, src)
            return
        host = self._w.create_buffer(nbytes)
        try:
            ctypes.memmove(int(host.base), src_host_ptr, nbytes)
            self._w.copy_to(dst, host)
        finally:
            self._w.release_buffer(host)

    def copy_from(self, dst_host_ptr: int, src_dev_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        """D2H copy: ``nbytes`` from device *src_dev_ptr* back to host *dst_host_ptr*."""
        self._require_open("copy_from")
        src = self._device_buffer(src_dev_ptr, worker_id, "copy_from")
        if not isinstance(nbytes, int) or nbytes <= 0:
            raise ValueError(f"nbytes must be a positive int, got {nbytes!r}")
        dst = self._named_host_buffer(int(dst_host_ptr), int(nbytes), writing=True)
        if dst is not None:
            self._w.copy_from(dst, src)
            return
        host = self._w.create_buffer(nbytes)
        try:
            self._w.copy_from(host, src)
            ctypes.memmove(dst_host_ptr, int(host.base), nbytes)
        finally:
            self._w.release_buffer(host)

    # ``alloc_tensor`` / ``free_tensor`` are inherited from Worker ABC. The
    # readiness guard differs from L2; explicit upload validation accepts an
    # ordinary CPU-contiguous tensor because copy_to stages through POSIX shm.

    def _require_ready(self, op: str) -> None:
        # Worker ABC hook: device-memory ops are valid until close().
        self._require_open(op)

    @staticmethod
    def _require_copy_host_tensor(tensor: torch.Tensor, api: str) -> None:
        """Require a CPU-contiguous endpoint for an explicit staged copy.

        ``copy_to`` / ``copy_from`` stage through a simpler-owned POSIX-shm
        Buffer, so the user's tensor no longer needs to predate the worker fork
        or use ``share_memory_()``. Dispatch arguments are different: they still
        cross the wire directly and retain their pre-fork shared-memory rule.
        """
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(
                f"{api} requires a CPU, contiguous tensor; got "
                f"device={tensor.device}, contiguous={tensor.is_contiguous()}"
            )

    def _prepare_init(self, init: torch.Tensor) -> torch.Tensor:
        # Explicit copies stage through a simpler-owned POSIX-shm Buffer.
        self._require_copy_host_tensor(init, "DistributedWorker.alloc_tensor(init=...)")
        return init

    def alloc_stacked_tensor(
        self,
        host: torch.Tensor,
        *,
        worker_ids: Sequence[int] | None = None,
    ) -> StackedDeviceTensor:
        """Upload each leading-dim shard of *host* to a worker once; reuse it.

        The leading dimension of *host* is the stack/shard dimension: shard ``i``
        (``host[i]``, shape ``host.shape[1:]``) is uploaded to worker
        ``worker_ids[i]`` and stays resident for the worker's lifetime. Pass the
        returned :class:`~pypto.runtime.StackedDeviceTensor` in place of *host*
        for a leading-dim-sharded program parameter (a ``[B, *tail]`` tensor the
        orchestrator slices per rank: ``for r in range(world_size):
        child(x[r], device=...)``). The generated ``host_orch`` indexes ``x[i]``
        to shard ``i``'s :class:`~pypto.runtime.DeviceTensor`, so the runtime
        skips the per-dispatch H2D upload through its retained Buffer — the stack is
        uploaded once here and reused across every ``rt(...)`` dispatch.

        Args:
            host: A CPU, contiguous ``[B, *tail]`` tensor. Explicit upload
                stages through a runtime-owned shared Buffer, so it may be an
                ordinary tensor created after :meth:`prepare`.
            worker_ids: ``worker_ids[i]`` is the worker that holds shard ``i``
                and whose task consumes ``x[i]``; it MUST equal the worker the
                program submits ``x[i]``'s dispatch to (its ``device=``
                expression). Entries must be distinct and within
                ``[0, world_size)``. Defaults to ``range(B)`` — the canonical
                ``for r in range(world_size): child(x[r], device=r)`` program. A
                permuted/subset placement (``device=perm[r]`` / ``device=2*r``)
                needs the matching ``worker_ids``.

        Returns:
            A :class:`~pypto.runtime.StackedDeviceTensor`; its shards are tracked
            by this worker and auto-freed on :meth:`close` if not released earlier
            via :meth:`free_stacked_tensor`.
        """
        self._require_open("alloc_stacked_tensor")
        if not isinstance(host, torch.Tensor):
            raise TypeError(
                f"alloc_stacked_tensor(host=...) expects a torch.Tensor, got {type(host).__name__}"
            )
        # Ahead of every upload on purpose: `alloc_tensor` re-checks the host in
        # `_prepare_init`, but only after its own device malloc, so N concurrent shards
        # would each commit device memory before any of them surfaced this ValueError.
        self._require_copy_host_tensor(host, "alloc_stacked_tensor(host=...)")
        if host.ndim < 2:
            raise ValueError(
                f"alloc_stacked_tensor needs a [B, *tail] tensor (rank >= 2), got shape {tuple(host.shape)}"
            )
        b = int(host.shape[0])
        if b < 1:
            raise ValueError(
                f"alloc_stacked_tensor needs at least one shard in the leading dim, "
                f"got shape {tuple(host.shape)}"
            )
        world = len(self.dc.device_ids)
        ids = list(range(b)) if worker_ids is None else [int(w) for w in worker_ids]
        if len(ids) != b:
            raise ValueError(f"worker_ids has {len(ids)} entries; host leading dim is {b}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"worker_ids must be distinct (one shard per worker), got {ids}")
        for w in ids:
            if not 0 <= w < world:
                raise ValueError(f"worker id {w} out of range [0, {world}) (world_size from device_ids)")

        shards: list[DeviceTensor] = []
        try:
            # Upload the shards concurrently: each one targets a different chip
            # worker, so the H2D transfers overlap instead of running back-to-back
            # at single-chip bandwidth (the device-memory bindings release the GIL,
            # and Simpler serializes provenance-guarded device ops per worker).
            def _upload_shard(i: int, w: int) -> DeviceTensor:
                return self.alloc_tensor(
                    tuple(host.shape[1:]),
                    host.dtype,
                    init=host[i].contiguous(),
                    worker_id=w,
                )

            futures: dict[Future[DeviceTensor], int] = {}
            errors: list[BaseException] = []
            with ThreadPoolExecutor(max_workers=len(ids)) as pool:
                for i, w in enumerate(ids):
                    try:
                        futures[pool.submit(_upload_shard, i, w)] = i
                    except BaseException as exc:
                        # Stop submitting, but still drain what is already running.
                        errors.append(exc)
                        break

            # Leaving the pool joined every submitted task, so this only collects
            # results. Every future is drained even once the group has failed —
            # dropping one would strand a live device buffer with no owner.
            uploaded: dict[int, DeviceTensor] = {}
            for future, index in futures.items():
                try:
                    uploaded[index] = future.result()
                except BaseException as exc:
                    errors.append(exc)

            if errors:
                # A concurrent group can fail anywhere, so the successes are not a
                # prefix of `ids`: roll back by index, against the owning worker.
                for index, shard in uploaded.items():
                    self.free_tensor(shard, worker_id=ids[index])
                raise errors[0]
            shards.extend(uploaded[index] for index in range(len(ids)))
        except Exception:
            # Roll back any shards already uploaded so a mid-loop failure
            # never leaks device memory.
            for shard, w in zip(shards, ids, strict=False):
                self.free_tensor(shard, worker_id=w)
            raise
        return StackedDeviceTensor(shards, tuple(host.shape), tuple(ids))

    def free_stacked_tensor(self, stacked: StackedDeviceTensor) -> None:
        """Release every shard of *stacked* against its owning worker. Idempotent."""
        for shard, w in zip(stacked.shards, stacked.worker_ids, strict=True):
            self.free_tensor(shard, worker_id=w)

    def release_inherited_host_tensor_refs(self) -> None:
        """Release compatibility lifetime references supplied at construction.

        Explicit uploads now stage through POSIX shm and remain valid while the
        caller itself retains the source tensor. Resident :class:`DeviceTensor`
        and :class:`StackedDeviceTensor` allocations are unaffected.
        """
        self._require_open("release_inherited_host_tensor_refs")
        self._inherited_host_tensors = ()
        # No later copy may name these ranges: the parent has dropped its references, so a
        # copy after this point stages rather than wrapping memory nobody vouches for.
        self._inherited_host_spans = ()
        with self._named_identity_mu:
            self._named_identities.clear()

    def copy_stacked_from(self, stacked: StackedDeviceTensor, host: torch.Tensor) -> None:
        """Read every shard of *stacked* back to *host* (D2H) — the read-back
        symmetric to :meth:`alloc_stacked_tensor`.

        Because a :class:`~pypto.runtime.StackedDeviceTensor` skips the
        per-dispatch D2H copy, callers that want the shards' current device
        contents (e.g. a resident KV cache at the end of an L3 step) must read
        them back explicitly. Shard ``i`` is copied from its owning worker
        ``stacked.worker_ids[i]`` into ``host[i]``.

        Args:
            stacked: The resident stacked tensor to read back.
            host: A CPU, contiguous ``[B, *tail]`` tensor whose shape and dtype
                match ``stacked.full_shape`` / ``stacked.dtype``. It may be an
                ordinary tensor allocated after :meth:`prepare`; D2H stages
                through a runtime-owned shared Buffer before copying into it.
        """
        self._require_open("copy_stacked_from")
        if not isinstance(stacked, StackedDeviceTensor):
            raise TypeError(
                f"copy_stacked_from(stacked=...) expects a StackedDeviceTensor, got {type(stacked).__name__}"
            )
        if not isinstance(host, torch.Tensor):
            raise TypeError(f"copy_stacked_from(host=...) expects a torch.Tensor, got {type(host).__name__}")
        if tuple(host.shape) != stacked.full_shape:
            raise ValueError(
                f"host shape {tuple(host.shape)} does not match stacked full_shape {stacked.full_shape}"
            )
        if host.dtype != stacked.dtype:
            raise ValueError(f"host dtype {host.dtype} does not match stacked dtype {stacked.dtype}")
        self._require_copy_host_tensor(host, "copy_stacked_from(host=...)")
        for i, (shard, w) in enumerate(zip(stacked.shards, stacked.worker_ids, strict=True)):
            # The public raw-pointer API targets the parent view; copy_from
            # handles the child-visible POSIX-shm staging internally.
            self.copy_from(host[i].data_ptr(), shard.data_ptr, shard.nbytes, worker_id=w)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, config: RunConfig | None = None) -> None:
        """Dispatch the primary compiled program, reusing all setup.

        Pass one argument per program parameter (in-place). Each argument is
        either:

        - a **shared-memory** host ``torch.Tensor`` (call ``.share_memory_()``
          and allocate it **before** :meth:`prepare`, then reuse the same buffer
          across dispatches, updating its contents in place). The forked chip
          worker reads/writes it through the inherited shared mapping; read
          outputs back directly from the tensor — no ``copy_from`` needed.
        - a worker-resident :class:`~pypto.runtime.DeviceTensor` (e.g. a static
          weight from :meth:`alloc_tensor`) or a simpler ``Tensor``.

        A non-shared ``torch.Tensor`` is rejected: a buffer allocated after the
        fork is invisible to the chip worker.

        Available only for single-program workers. When several programs were
        prepared together (multi-program), the target is ambiguous, so this
        raises ``TypeError`` — dispatch explicitly via ``rt.run(compiled, ...)``.

        ``config`` is an optional per-dispatch :class:`RunConfig`: its per-task
        ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
        ``ring_dep_pool``, each a scalar or a per-ring list of 4 ints) size this
        dispatch's runtime ring buffers without
        touching the prepared program's shared config, so consecutive dispatches
        can use different ring sizes. Its runtime DFX fields are also applied per
        dispatch. On onboard L3, ``enable_chip_swimlane`` executes the workload
        twice on the same prepared worker: first with dep-gen only, then with
        swimlane enabled and dep-gen disabled. Mutable host/resident arguments
        are not restored between those profiling passes and can therefore be
        updated twice. ``None`` snapshots the program's baseline for this
        dispatch.
        """
        if self._multi_program:
            raise TypeError(
                "rt(*args) is ambiguous on a multi-program DistributedWorker; "
                "dispatch explicitly with rt.run(compiled, *args)."
            )
        return self._run_compiled(self._compiled, *args, config=config)

    def _validate_prepared_dispatch_arg(self, info: Any, arg: Any) -> None:
        """Validate one non-scalar argument at the public prepared boundary."""
        from pypto.ir.compiled_program import (  # noqa: PLC0415
            _validate_device_tensor,
            _validate_stacked_tensor,
        )

        if isinstance(arg, StackedDeviceTensor):
            for shard_index, (shard, worker_id) in enumerate(zip(arg.shards, arg.worker_ids, strict=True)):
                self._require_owned_resident_tensor(
                    shard,
                    f"Parameter {info.name!r} shard {shard_index}",
                    worker_id=worker_id,
                )
            _validate_stacked_tensor(arg, info)
        elif isinstance(arg, DeviceTensor):
            self._require_owned_resident_tensor(arg, f"Parameter {info.name!r}")
            _validate_device_tensor(arg, info)
        elif isinstance(arg, torch.Tensor):
            if not arg.is_shared():
                raise TypeError(
                    f"Parameter {info.name!r}: a host torch.Tensor passed to a DistributedWorker "
                    "must be shared memory allocated BEFORE prepare() (call .share_memory_() and "
                    "reuse the same buffer across dispatches), so the forked chip worker can see it."
                )
        elif not _is_simpler_tensor(arg):
            raise TypeError(
                f"DistributedWorker parameter {info.name!r} got {type(arg).__name__}; expected a "
                f"shared-memory torch.Tensor, a DeviceTensor allocated by this worker, a "
                f"StackedDeviceTensor, or a simpler Tensor."
            )

    def _submit_compiled(
        self, compiled: DistributedCompiledProgram, *args: Any, config: RunConfig | None = None
    ) -> DistributedRunHandle:
        """Submit *compiled* on the shared Worker via one bounded frame.

        ``config`` is an optional per-dispatch :class:`RunConfig` whose per-task
        ring sizing and runtime DFX fields apply to this dispatch. When given, a
        fresh ``CallConfig`` is built from the program's ``aicpu_thread_num``
        baseline, leaving the prepared shared config untouched. ``None`` also
        builds a fresh baseline snapshot so an in-flight dispatch never shares
        mutable configuration with its successor. Onboard L3 swimlane capture
        runs a dep-gen graph pass followed by a dep-gen-disabled timing pass on
        the same prepared Worker; mutable arguments are not restored between
        the two executions.
        """
        self._require_open("submit")
        state = self._states.get(compiled)
        if state is None:
            raise ValueError(
                "DistributedWorker.submit/run requires a DistributedCompiledProgram "
                "registered when this worker was constructed."
            )

        param_infos = state["param_infos"]
        n_params = len(param_infos)
        if len(args) != n_params:
            raise TypeError(
                f"DistributedWorker expects {n_params} arguments (in-place, one per parameter), "
                f"got {len(args)}. Parameters: {[p.name for p in param_infos]}"
            )

        frame, dispatch_id = self._acquire_dispatch_frame()
        try:
            call_config, dfx_base, two_pass_swimlane = self._prepare_dispatch_config(
                compiled,
                config,
            )

            tensors = frame.tensors
            tensors.clear()
            tensors.update(state["base_tensor_frames"][frame.slot_id])
            frame.keepalive.clear()
            frame.keepalive.extend((compiled, call_config, config, *args))
            for info, arg in zip(param_infos, args, strict=True):
                if info.shape is None:
                    # Scalar parameter (e.g. seq_len): forwarded as-is to the entry.
                    tensors[info.name] = arg
                    continue
                self._validate_prepared_dispatch_arg(info, arg)
                tensors[info.name] = arg

            if two_pass_swimlane:
                assert config is not None
                assert dfx_base is not None
                _run_l3_swimlane_two_pass(
                    compiled._distributed_config,
                    config,
                    dfx_base,
                    lambda pass_config: self._dispatch_prepared(
                        state,
                        tensors,
                        pass_config,
                        frame.keepalive,
                    ),
                )
                _collect_l3_swimlane(compiled.output_dir, compiled.platform)
                self._release_unpublished_dispatch_frame(frame)
                return DistributedRunHandle._completed(self)

            postprocess: Callable[[], None] | None = None
            if config is not None and config.enable_chip_swimlane > 0:

                def collect_swimlane() -> None:
                    _collect_l3_swimlane(compiled.output_dir, compiled.platform)

                postprocess = collect_swimlane
            handle = DistributedRunHandle(self, None, frame, dispatch_id, postprocess)
            try:
                frame.handle = handle
                with self._dispatch_cv:
                    self._active_dispatch_handles.add(handle)
            except BaseException:
                self._discard_unsubmitted_dispatch_handle(handle, frame)
                raise

            self._submit_native_dispatch(handle, frame, state, tensors, call_config)
            return handle
        except BaseException:
            if frame.handle is None and frame.in_use:
                self._release_unpublished_dispatch_frame(frame)
            raise

    def _prepare_dispatch_config(
        self,
        compiled: DistributedCompiledProgram,
        config: RunConfig | None,
    ) -> tuple[Any, Path | None, bool]:
        """Snapshot one dispatch's runtime config after capacity admission."""
        if config is None:
            return _make_call_config(compiled._distributed_config), None, False

        dfx_base = compiled.output_dir / "dfx_outputs"
        two_pass_swimlane = config.enable_chip_swimlane > 0 and not compiled.platform.endswith("sim")
        call_config = None
        if not two_pass_swimlane:
            call_config = _make_call_config(compiled._distributed_config, config, dfx_base=dfx_base)

        # This worker reuses one output_dir across dispatches, so stale
        # ``rank*/d{k}`` dirs from an earlier, larger run must be cleared before
        # this run rewrites ``d0, d1, ...``.
        from .runner import _DfxOpts  # noqa: PLC0415

        if _DfxOpts.from_run_config(config).any():
            # Every dispatch writes below the same output directory. Finish
            # earlier work before clearing or repopulating those paths.
            self._drain_dispatch_handles()
            _clear_dfx_dispatch_dirs(dfx_base)
        return call_config, dfx_base, two_pass_swimlane

    def _run_compiled(
        self, compiled: DistributedCompiledProgram, *args: Any, config: RunConfig | None = None
    ) -> None:
        """Blocking compatibility wrapper around :meth:`_submit_compiled`."""
        self.submit(compiled, *args, config=config).result()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _require_open(self, op: str) -> None:
        if self._closed:
            raise RuntimeError(f"DistributedWorker.{op}() called after close()")

    def _drain_dispatch_handles(self) -> BaseException | None:
        """Finalize every published dispatch in FIFO order."""
        first_error: BaseException | None = None
        while True:
            with self._dispatch_cv:
                if not self._active_dispatch_handles:
                    return first_error
                handle = min(self._active_dispatch_handles, key=lambda item: item._dispatch_id)
            try:
                handle.result()
            except BaseException as exc:  # noqa: BLE001 - close continues bounded cleanup
                if first_error is None:
                    first_error = exc

    def close(self) -> None:
        """Release runtime resources, retrying incomplete Worker cleanup."""
        # Serialize the admission transition with submit() and prevent two
        # callers from running teardown concurrently.
        with self._dispatch_submit_mu:
            if self._close_complete or self._closing:
                return
            self._closing = True
            first_attempt = not self._closed
            if first_attempt:
                with self._dispatch_cv:
                    self._accepting_dispatches = False
                    self._dispatch_cv.notify_all()

        try:
            first_error: BaseException | None = None
            if first_attempt:
                if self._persistent_error is not None and not self._persistent_error_reported:
                    first_error = self._remember_close_error(first_error, self._persistent_error)
                    self._persistent_error_reported = True

                drain_error = self._drain_dispatch_handles()
                if drain_error is not None:
                    first_error = self._remember_close_error(first_error, drain_error)

                # A failed native run may have abandoned its per-run finalizer. In
                # that case keep retained domains globally reachable for the
                # underlying Worker's whole-tree cleanup.
                if self._persistent and self._persistent_error is None:
                    try:
                        self._release_persistent_domains(self._persistent_domains_by_program)
                    except BaseException as exc:  # noqa: BLE001 - preserve primary error and continue
                        first_error = self._remember_close_error(first_error, exc)

                # DeviceTensor frees use the still-live simpler control path.
                try:
                    self._close_owned_tensors()
                except BaseException as exc:  # noqa: BLE001 - underlying worker still must close
                    first_error = self._remember_close_error(first_error, exc)

                self._closed = True
                for handle in list(self._handles):
                    handle._mark_closed()
                self._handles.clear()
            try:
                self._w.close()
            except BaseException as exc:  # noqa: BLE001 - report after local teardown
                first_error = self._remember_close_error(first_error, exc)
            else:
                self._close_complete = True
            finally:
                self._inherited_host_tensors = ()
                self._inherited_host_spans = ()
                self._named_identities.clear()
                self._persistent_domains_by_program.clear()
                if self._close_complete:
                    self._device_buffers.clear()
            if first_error is not None:
                raise first_error
        finally:
            with self._dispatch_submit_mu:
                self._closing = False

    def __enter__(self) -> DistributedWorker:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Explicit dispatch — mirror ChipWorker's run / register surface so
    # library code can use one method name across L2 / L3.
    # ------------------------------------------------------------------

    def submit(
        self,
        compiled: DistributedCompiledProgram,
        *args: Any,
        config: RunConfig | None = None,
    ) -> DistributedRunHandle:
        """Submit *compiled* and return before device completion when supported.

        The returned handle owns one of two bounded metadata frames plus all
        argument and configuration lifetimes. A third submission waits for the
        oldest handle before reusing a frame. Diagnostic two-pass swimlane
        capture remains a synchronous fallback and returns a completed handle.

        Args:
            compiled: A program registered when this worker was constructed.
            *args: In-place program arguments. Mutable arguments must not be
                reused or modified until the returned handle completes.
            config: Optional per-dispatch runtime configuration.

        Returns:
            A handle that owns the dispatch lifetime and cached outcome.
        """
        with self._dispatch_submit_mu:
            return self._submit_compiled(compiled, *args, config=config)

    def run(
        self,
        compiled: DistributedCompiledProgram,
        *args: Any,
        config: RunConfig | None = None,
    ) -> None:
        """Dispatch *compiled* on this DistributedWorker.

        Provided for symmetry with :meth:`ChipWorker.run`, so library code can
        write ``rt.run(compiled, *args)`` and accept either runtime kind. For a
        multi-program worker this selects which prepared program to dispatch.

        *compiled* must be one of the :class:`DistributedCompiledProgram` objects
        this worker was constructed from; passing an unregistered one raises
        ``ValueError``.

        ``config`` is an optional per-dispatch :class:`RunConfig`; its per-task
        ring sizing and runtime DFX fields apply without touching the prepared
        program's shared config. In a multi-program worker each program can
        therefore use its own ring sizes and diagnostics. On onboard L3,
        ``enable_chip_swimlane`` executes a dep-gen graph pass followed by a
        dep-gen-disabled timing pass; mutable arguments are not restored between
        them. ``None`` snapshots the program's baseline for this dispatch.
        """
        return self.submit(compiled, *args, config=config).result()

    def register(self, compiled: DistributedCompiledProgram) -> RegistrationHandle:
        """Pre-register *compiled* on this DistributedWorker.

        Returns a :class:`~pypto.runtime.RegistrationHandle` whose
        ``__call__`` delegates to :meth:`run`. The cid alias-safety rules
        described on :class:`RegistrationHandle` apply.

        Unlike L2, the underlying simpler registration already happened
        during :meth:`__init__` (it must, for COW propagation to forked
        chip children). This method just packages the existing setup as a
        callable handle, exposing ``cid=0`` as a placeholder.

        Raises:
            RuntimeError: This DistributedWorker has been closed.
            ValueError: *compiled* is not one of the programs this worker was
                constructed from.
        """
        self._require_open("register")
        if compiled not in self._states:
            raise ValueError(
                "DistributedWorker.register(compiled) requires a DistributedCompiledProgram "
                "registered when this worker was constructed."
            )
        # Avoid a hard cycle: distributed_runner imports from worker only
        # for RegistrationHandle; worker never imports from distributed_runner.
        from .worker import RegistrationHandle  # noqa: PLC0415

        # L3 doesn't have a per-callable cid; expose 0 as a placeholder.
        # __call__ delegates to self.run() which is the existing dispatch path.
        handle = RegistrationHandle(self, compiled, cid=0)
        self._handles.add(handle)
        return handle
