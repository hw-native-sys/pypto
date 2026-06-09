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
import sys
import types
import warnings
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np  # pyright: ignore[reportMissingImports]
import torch

from .device_tensor import DeviceTensor
from .runtime_base import Worker

if TYPE_CHECKING:
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig

    from .worker import RegistrationHandle


# ---------------------------------------------------------------------------
# ContinuousTensor → torch.Tensor conversion
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


def _tensor_from_continuous(ct) -> torch.Tensor:
    """Convert a simpler ContinuousTensor to a torch.Tensor (zero-copy).

    The returned tensor shares the same memory as the ContinuousTensor
    (via shared memory), so modifications are visible across processes.

    For dtypes that ``torch.from_numpy`` cannot accept directly (FP16/BF16),
    we view the buffer as raw bytes (uint8) and reinterpret with
    ``torch.Tensor.view(dtype)`` — a zero-copy bit-cast that preserves the
    shared-memory aliasing required for ``Out``/``InOut`` parameters.
    """
    # ``str(ct.dtype)`` yields ``"DataType.FLOAT32"``; strip the enum prefix
    # to match the bare type names used as keys in ``_DTYPE_MAP``.
    dtype_str = str(ct.dtype)
    dtype_key = dtype_str.rsplit(".", 1)[-1]
    try:
        c_type, torch_dtype = _DTYPE_MAP[dtype_key]
    except KeyError as exc:
        raise TypeError(
            f"Unsupported ContinuousTensor dtype: {dtype_str!r}. "
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

    arr = np.ctypeslib.as_array(
        ctypes.cast(ct.data, ctypes.POINTER(c_type)),
        shape=(n_c_elements,),
    )
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
    return module


# ---------------------------------------------------------------------------
# Setup steps shared by the one-shot ``execute_distributed`` path and the
# reusable ``DistributedWorker`` handle. Keeping them as free functions lets
# both paths run identical, expensive setup (compile_and_assemble, module load,
# Worker construction + registration) without duplicating it.
# ---------------------------------------------------------------------------


def _assemble_chip_callables(compiled: DistributedCompiledProgram) -> tuple[dict[str, Any], str]:
    """Build a ChipCallable for each chip-level task under ``next_levels/{name}/``."""
    from pypto.pypto_core.ir import FunctionType  # noqa: PLC0415
    from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

    chip_callables: dict[str, Any] = {}
    runtime_name = "tensormap_and_ringbuffer"
    next_levels_dir = compiled.output_dir / "next_levels"
    for func in compiled._program.functions.values():
        if func.func_type == FunctionType.Orchestration:
            chip_dir = next_levels_dir / func.name
            if chip_dir.exists():
                chip_callable, runtime_name, _ = compile_and_assemble(chip_dir, compiled.platform)
                chip_callables[func.name] = chip_callable

    if not chip_callables:
        raise RuntimeError(f"No chip-level tasks found in {next_levels_dir}")
    return chip_callables, runtime_name


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
) -> Any:
    """Construct a simpler ``Worker(level=3)`` from the distributed config."""
    from simpler.worker import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        Worker,
    )

    return Worker(
        level=3,
        device_ids=dc.device_ids,
        num_sub_workers=num_sub,
        platform=platform,
        runtime=runtime_name,
    )


def _register_callables(
    w: Any, sub_worker_fns: dict[str, Any], chip_callables: dict[str, Any]
) -> tuple[dict[str, int], dict[str, int]]:
    """Register SubWorker + Chip callables before ``w.init()``.

    Both must happen before ``w.init()`` so the L3 fork inherits the registry
    via COW (runtime PR #710); the emitted host_orch then dispatches via cids —
    ``orch.submit_sub(sub_ids[name], …)`` / ``orch.submit_next_level(callables[name], …)``.
    """
    sub_ids: dict[str, int] = {name: w.register(fn) for name, fn in sub_worker_fns.items()}
    chip_cids: dict[str, int] = {name: w.register(cc) for name, cc in chip_callables.items()}
    return sub_ids, chip_cids


def _check_callback_arity(name: str, fn: Callable[..., Any]) -> None:
    """Validate that a user callback can be invoked as ``fn(args)``.

    SubWorker callables receive a single ``TaskArgs`` positional argument. A
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
            f"argument fn(args: TaskArgs); got signature {sig}."
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


def _make_call_config(dc: DistributedConfig) -> Any:
    """Build a simpler ``CallConfig`` from the distributed config."""
    from simpler.task_interface import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        CallConfig,
    )

    call_config = CallConfig()
    if dc.block_dim is not None:
        call_config.block_dim = dc.block_dim
    call_config.aicpu_thread_num = dc.aicpu_thread_num
    return call_config


def _is_continuous_tensor(arg: Any) -> bool:
    """True if *arg* is a simpler ``ContinuousTensor``.

    Returns ``False`` (rather than raising) when simpler is unavailable, so the
    DeviceTensor-only path stays importable without the runtime package.
    """
    try:
        from .task_interface import (  # noqa: PLC0415
            ContinuousTensor,  # pyright: ignore[reportAttributeAccessIssue]
        )
    except ImportError:
        return False
    return isinstance(arg, ContinuousTensor)


def _dispatch(
    w: Any,
    entry_fn: Any,
    tensors: dict[str, Any],
    chip_cids: dict[str, int],
    sub_ids: dict[str, int],
    call_config: Any,
    device_nums: int,
) -> None:
    """Build the orchestration closure and run it once on ``w``."""
    # Fresh _keep per dispatch: it pins per-call TaskArgs alive for the run.
    _keep: list[Any] = []

    # ``world_size`` is the only worker-level scalar the entry needs; codegen
    # binds ``pld.system.world_size()`` to this kwarg uniformly across comm
    # and comm-less paths.

    def orch_fn(orch, _unused_args, _unused_cfg):
        entry_fn(
            orch,
            _unused_args,
            call_config,
            tensors=tensors,
            callables=chip_cids,
            sub_ids=sub_ids,
            _keep=_keep,
            world_size=device_nums,
        )

    w.run(orch_fn)


def execute_distributed(
    compiled: DistributedCompiledProgram,
    coerced_args: list[torch.Tensor | DeviceTensor],
    config: Any = None,
) -> None:
    """Execute a distributed compiled program once via simpler Worker(level=3).

    One-shot path: runs the full setup, dispatches once, then tears the Worker
    down. Supports host ``torch.Tensor`` inputs (placed in shared memory before
    the fork). For repeated dispatch with device-resident inputs, prefer
    :meth:`DistributedCompiledProgram.prepare` → :class:`DistributedWorker`.

    Args:
        compiled: The DistributedCompiledProgram instance.
        coerced_args: Coerced arguments — host ``torch.Tensor`` or
            worker-resident :class:`~pypto.runtime.DeviceTensor`.
        config: Optional run configuration (unused for now).
    """
    dc = compiled._distributed_config
    output_dir = compiled.output_dir

    chip_callables, runtime_name = _assemble_chip_callables(compiled)
    entry_fn, alloc_fn = _load_orch_entry(output_dir)

    # Build tensor mapping from parameter names. Host torch.Tensor inputs must
    # be in shared memory before the fork; DeviceTensor inputs are device
    # pointers forwarded at submit time and need no pre-fork shared memory.
    param_infos, _, _ = compiled._get_metadata()
    tensors: dict[str, torch.Tensor | DeviceTensor] = {}
    for info, arg in zip(param_infos, coerced_args, strict=True):
        if isinstance(arg, DeviceTensor):
            tensors[info.name] = arg
            continue
        if not arg.is_shared():
            arg.share_memory_()
        tensors[info.name] = arg

    # Pre-fork: allocate HOST-level intermediate tensors so the POSIX
    # shared-memory mappings exist before w.init() forks child processes.
    if alloc_fn is not None:
        alloc_fn(tensors)

    sub_worker_fns = _load_sub_worker_fns(output_dir)
    # The one-shot path cannot supply callbacks; if the program declares any
    # runtime-bound (`...`-body) SubWorker, fail early with a clear message
    # pointing at prepare(callbacks={...}).
    sub_worker_fns = _bind_sub_workers(sub_worker_fns, None, _load_required_callbacks(output_dir))

    num_sub = max(dc.num_sub_workers, len(sub_worker_fns))

    # Construct/register/init inside the try so a failure in any setup step still
    # closes the worker and unlinks the rootinfo temp file — none of these leak.
    w = None
    try:
        w = _construct_worker(dc, compiled.platform, runtime_name, num_sub)
        sub_ids, chip_cids = _register_callables(w, sub_worker_fns, chip_callables)
        w.init()
        _dispatch(w, entry_fn, tensors, chip_cids, sub_ids, _make_call_config(dc), len(dc.device_ids))
    finally:
        if w is not None:
            w.close()


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
    tensor (no ``copy_from``). Large static weights are uploaded once to a
    worker-resident :class:`~pypto.runtime.DeviceTensor` via :meth:`alloc_tensor`
    (its ``init`` source must likewise be a pre-``prepare`` shared tensor) and
    mixed in. This mirrors the runtime's ``child_memory`` example.

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
        config: Any = None,
        *,
        callbacks: dict[str, Callable[..., Any]] | None = None,
        sub_worker_overrides: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        super().__init__()  # initialize Worker ABC state (_owned_tensors)
        del config  # reserved for future per-runtime overrides
        callbacks = _coalesce_callbacks(callbacks, sub_worker_overrides)

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
            for prog in programs:
                self._check_compatible(prog, primary)
                chip_callables, prog_runtime = _assemble_chip_callables(prog)
                runtime_name = self._unify_runtime(runtime_name, prog_runtime)
                entry_fn, alloc_fn = _load_orch_entry(prog.output_dir)
                loaded_subs = _load_sub_worker_fns(prog.output_dir)
                prog_callbacks = {name: fn for name, fn in callbacks.items() if name in loaded_subs}
                consumed |= set(prog_callbacks)
                sub_worker_fns = _bind_sub_workers(
                    loaded_subs, prog_callbacks, _load_required_callbacks(prog.output_dir)
                )
                num_sub = max(num_sub, prog._distributed_config.num_sub_workers, len(sub_worker_fns))
                base_tensors: dict[str, Any] = {}
                if alloc_fn is not None:
                    alloc_fn(base_tensors)
                self._states[prog] = {
                    "entry_fn": entry_fn,
                    "base_tensors": base_tensors,
                    "call_config": _make_call_config(prog._distributed_config),
                    "param_infos": tuple(prog._get_metadata()[0]),
                    "device_nums": len(prog._distributed_config.device_ids),
                }
                loaded.append((prog, chip_callables, sub_worker_fns))

            unconsumed = sorted(set(callbacks) - consumed)
            if unconsumed:
                raise ValueError(f"callbacks names {unconsumed} are not sub-workers of any prepared program.")

            if runtime_name is None:  # unreachable: programs is non-empty
                raise RuntimeError("failed to resolve distributed runtime")

            # Phase 2: one worker for all programs. Register every program's
            # callables before ``init()`` so the L3 fork inherits the whole
            # registry via COW; each program keeps its own cids in its state.
            self._w = _construct_worker(self.dc, primary.platform, runtime_name, num_sub)
            for prog, chip_callables, sub_worker_fns in loaded:
                sub_ids, chip_cids = _register_callables(self._w, sub_worker_fns, chip_callables)
                self._states[prog]["sub_ids"] = sub_ids
                self._states[prog]["chip_cids"] = chip_cids

            self._w.init()

            # Fork the chip/sub workers now (rather than lazily on the first
            # ``run()``) so the device-memory API — ``malloc`` / ``copy_to`` /
            # ``alloc_tensor`` — is usable before the first dispatch: those route
            # through the orchestrator, which only exists after the hierarchy is
            # started. ``_start_hierarchical`` is idempotent and is the same fork
            # the first ``run()`` would trigger; the comm path already runs it from
            # ``init()``. Intermediates are allocated above (pre-fork) so forked
            # children inherit their shared-memory mappings.
            self._w._start_hierarchical()
        except Exception:
            if self._w is not None:
                try:
                    self._w.close()
                except Exception:
                    pass
            raise

        self._closed = False
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

    # ------------------------------------------------------------------
    # Device memory primitives
    #
    # Routed through the simpler Orchestrator facade (``Worker._orch``) rather
    # than ``Worker.malloc`` etc.: the level>=3 branch of those wrappers calls
    # ``self._orch._impl.<op>(...)``, but the orchestrator's C++ handle lives on
    # ``_o`` (no ``_impl``), so ``Worker.malloc`` raises ``AttributeError``. The
    # facade methods (``malloc(worker_id, size)`` etc.) are the working path the
    # generated host_orch and runtime examples use. ``_orch`` exists because
    # __init__ starts the hierarchy eagerly.
    # ------------------------------------------------------------------

    def _orch(self) -> Any:
        orch = getattr(self._w, "_orch", None)
        if orch is None:
            raise RuntimeError(
                "DistributedWorker worker has no active orchestrator; the chip hierarchy was not started."
            )
        return orch

    def malloc(self, nbytes: int, *, worker_id: int = 0) -> int:
        """Allocate ``nbytes`` on chip *worker_id*; returns a device pointer."""
        self._require_open("malloc")
        return int(self._orch().malloc(worker_id, nbytes))

    def free(self, ptr: int, *, worker_id: int = 0) -> None:
        """Release a pointer previously returned by :meth:`malloc`."""
        self._require_open("free")
        self._orch().free(worker_id, ptr)

    def copy_to(self, dst_dev_ptr: int, src_host_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        """H2D copy: ``nbytes`` from host *src_host_ptr* to device *dst_dev_ptr*."""
        self._require_open("copy_to")
        self._orch().copy_to(worker_id, dst_dev_ptr, src_host_ptr, nbytes)

    def copy_from(self, dst_host_ptr: int, src_dev_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        """D2H copy: ``nbytes`` from device *src_dev_ptr* back to host *dst_host_ptr*."""
        self._require_open("copy_from")
        self._orch().copy_from(worker_id, dst_host_ptr, src_dev_ptr, nbytes)

    # ``alloc_tensor`` / ``free_tensor`` are inherited from Worker ABC.
    # Only the two behaviours that genuinely differ from L2 are overridden below:
    # the readiness guard (open vs. closed) and the host-init upload policy (the
    # upload runs in a forked chip worker, so no defensive copy is possible).

    _WORKER_KIND = "chip worker"

    def _require_ready(self, op: str) -> None:
        # Worker ABC hook: device-memory ops are valid until close().
        self._require_open(op)

    def _prepare_init(self, init: torch.Tensor) -> torch.Tensor:
        # Worker ABC hook: the upload (``copy_to``) runs **inside the
        # forked chip worker**, so ``init`` must be a CPU, contiguous,
        # shared-memory tensor allocated **before**
        # :meth:`DistributedCompiledProgram.prepare` (call ``.share_memory_()``).
        # Unlike L2 we cannot make a defensive ``.cpu().contiguous()`` copy: that
        # copy would live only in the parent and be invisible to the child.
        if not (init.is_shared() and init.is_contiguous() and init.device.type == "cpu"):
            raise ValueError(
                "DistributedWorker.alloc_tensor(init=...) requires a CPU, contiguous, "
                "shared-memory tensor allocated BEFORE prepare() (call .share_memory_()). "
                "The upload runs in the forked chip worker, which can only read host "
                "memory it inherited at fork."
            )
        return init

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, config: Any = None) -> None:
        """Dispatch one run on the primary compiled program, reusing all setup.

        Pass one argument per program parameter (in-place). Each argument is
        either:

        - a **shared-memory** host ``torch.Tensor`` (call ``.share_memory_()``
          and allocate it **before** :meth:`prepare`, then reuse the same buffer
          across dispatches, updating its contents in place). The forked chip
          worker reads/writes it through the inherited shared mapping; read
          outputs back directly from the tensor — no ``copy_from`` needed.
        - a worker-resident :class:`~pypto.runtime.DeviceTensor` (e.g. a static
          weight from :meth:`alloc_tensor`) or a simpler ``ContinuousTensor``.

        A non-shared ``torch.Tensor`` is rejected: a buffer allocated after the
        fork is invisible to the chip worker.

        Available only for single-program workers. When several programs were
        prepared together (multi-program), the target is ambiguous, so this
        raises ``TypeError`` — dispatch explicitly via ``rt.run(compiled, ...)``.
        """
        if self._multi_program:
            raise TypeError(
                "rt(*args) is ambiguous on a multi-program DistributedWorker; "
                "dispatch explicitly with rt.run(compiled, *args)."
            )
        return self._run_compiled(self._compiled, *args, config=config)

    def _run_compiled(self, compiled: DistributedCompiledProgram, *args: Any, config: Any = None) -> None:
        """Dispatch *compiled* on the shared Worker via its per-program state."""
        del config  # reserved for future per-call overrides
        self._require_open("run")
        from pypto.ir.compiled_program import _validate_device_tensor  # noqa: PLC0415

        state = self._states.get(compiled)
        if state is None:
            raise ValueError(
                "DistributedWorker.run(compiled, ...) requires a DistributedCompiledProgram "
                "registered when this worker was constructed."
            )

        param_infos = state["param_infos"]
        n_params = len(param_infos)
        if len(args) != n_params:
            raise TypeError(
                f"DistributedWorker expects {n_params} arguments (in-place, one per parameter), "
                f"got {len(args)}. Parameters: {[p.name for p in param_infos]}"
            )

        tensors: dict[str, Any] = dict(state["base_tensors"])
        for info, arg in zip(param_infos, args, strict=True):
            if info.shape is None:
                # Scalar parameter (e.g. seq_len): forwarded as-is to the entry.
                tensors[info.name] = arg
                continue
            if isinstance(arg, DeviceTensor):
                _validate_device_tensor(arg, info)
            elif isinstance(arg, torch.Tensor):
                if not arg.is_shared():
                    raise TypeError(
                        f"Parameter {info.name!r}: a host torch.Tensor passed to a DistributedWorker "
                        f"must be shared memory allocated BEFORE prepare() (call .share_memory_() and "
                        f"reuse the same buffer across dispatches), so the forked chip worker can see "
                        f"it. Got a non-shared tensor."
                    )
            elif not _is_continuous_tensor(arg):
                raise TypeError(
                    f"DistributedWorker parameter {info.name!r} got {type(arg).__name__}; expected a "
                    f"shared-memory torch.Tensor, a worker-resident DeviceTensor, or a ContinuousTensor."
                )
            tensors[info.name] = arg

        _dispatch(
            self._w,
            state["entry_fn"],
            tensors,
            state["chip_cids"],
            state["sub_ids"],
            state["call_config"],
            state["device_nums"],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _require_open(self, op: str) -> None:
        if self._closed:
            raise RuntimeError(f"DistributedWorker.{op}() called after close()")

    def close(self) -> None:
        """Release the Worker and comm rootinfo file. Idempotent."""
        if self._closed:
            return
        # Auto-free any DeviceTensors the caller forgot. Run BEFORE we set
        # ``_closed`` so the per-op ``_require_open`` guard inside ``free``
        # still admits these calls, and BEFORE we tear down the underlying
        # worker so the free path is still live.
        self._close_owned_tensors()
        self._closed = True
        # Mark every still-alive RegistrationHandle as closed so subsequent
        # handle(...) calls raise instead of dispatching to a torn-down runtime.
        for handle in list(self._handles):
            handle._mark_closed()
        self._handles.clear()
        self._w.close()

    def __enter__(self) -> DistributedWorker:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Explicit dispatch — mirror ChipWorker's run / register surface so
    # library code can use one method name across L2 / L3.
    # ------------------------------------------------------------------

    def run(
        self,
        compiled: DistributedCompiledProgram,
        *args: Any,
        config: Any = None,
    ) -> None:
        """Dispatch *compiled* on this DistributedWorker.

        Provided for symmetry with :meth:`ChipWorker.run`, so library code can
        write ``rt.run(compiled, *args)`` and accept either runtime kind. For a
        multi-program worker this selects which prepared program to dispatch.

        *compiled* must be one of the :class:`DistributedCompiledProgram` objects
        this worker was constructed from; passing an unregistered one raises
        ``ValueError``.
        """
        return self._run_compiled(compiled, *args, config=config)

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
