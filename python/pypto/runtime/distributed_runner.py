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

from .device_tensor import DeviceTensor, StackedDeviceTensor
from .runtime_base import Worker

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


def _tensor_from_continuous(ct) -> torch.Tensor:
    """Convert a simpler ``Tensor`` to a torch.Tensor (zero-copy).

    The returned tensor shares the same memory as the simpler ``Tensor``
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


def _assemble_chip_callables(compiled: DistributedCompiledProgram) -> tuple[dict[str, Any], str]:
    """Build a ChipCallable for each chip-level task under ``next_levels/{name}/``.

    Driven entirely by the on-disk layout — each ``next_levels/{name}/`` that
    contains a ``kernel_config.py`` is a complete single-chip sub-build that
    :func:`compile_and_assemble` consumes directly. This requires no live IR, so
    it works identically for a freshly-compiled program and one reconstructed via
    :meth:`DistributedCompiledProgram.from_dir` (the ``runtime_dir`` replay path).
    """
    chip_callables: dict[str, Any] = {}
    runtime_name: str | None = None
    next_levels_dir = compiled.output_dir / "next_levels"
    if next_levels_dir.is_dir():
        for chip_dir in sorted(next_levels_dir.iterdir()):
            if not (chip_dir / "kernel_config.py").exists():
                continue
            # Imported lazily — and only once there is a real chip to build — so
            # the "no chip-level tasks" error path below stays usable without the
            # heavy device_runner → simpler toolchain import.
            from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

            chip_callable, chip_runtime, _ = compile_and_assemble(chip_dir, compiled.platform)
            chip_callables[chip_dir.name] = chip_callable
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
    # ``w.register`` returns an opaque ``CallableHandle`` (runtime #891); typed
    # ``Any`` here and threaded straight back into ``submit_sub`` /
    # ``submit_next_level``, which accept the handle.
    sub_ids: dict[str, Any] = {name: w.register(fn) for name, fn in sub_worker_fns.items()}
    chip_cids: dict[str, Any] = {name: w.register(cc) for name, cc in chip_callables.items()}
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


def _make_call_config(
    dc: DistributedConfig,
    run_config: RunConfig | None = None,
    *,
    dfx_base: Path | None = None,
    co_enable_swimlane_dep_gen: bool = True,
) -> Any:
    """Build a simpler ``CallConfig`` from the distributed config.

    The ``block_dim`` / ``aicpu_thread_num`` baseline always comes from the
    program's :class:`DistributedConfig`. When *run_config* is given, its
    per-task ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
    ``ring_dep_pool``, each a scalar or a per-ring list of 4 ints) are overlaid
    on top, so a single L3 dispatch can size the
    runtime's ring buffers without mutating the prepared program's shared
    config. ``None`` (the default) leaves the baseline untouched and the runtime
    applies its own ``PTO2_RING_*`` env var / compile-time fallback.

    DFX diagnostics (``enable_dump_args`` / ``enable_pmu`` / ``enable_dep_gen``
    / ``enable_scope_stats`` / ``enable_l2_swimlane``) are likewise read from
    *run_config* and written to the shared ``config`` the host_orch chip dispatch
    forwards to every ``orch.submit_next_level``; their artifacts land under
    *dfx_base* (``<output_dir>/dfx_outputs``). ``enable_l2_swimlane`` co-enables
    dep_gen so the converter can resolve task arrows / kernel names (see the
    inline note on the single-pass timing trade-off vs the L2 two-pass).

    Args:
        dc: The program's distributed configuration (baseline).
        run_config: Optional per-dispatch :class:`RunConfig` whose ``ring_*`` and
            DFX overrides are applied. ``None`` means no override.
        dfx_base: Directory under which DFX artifacts are written
            (``<output_dir>/dfx_outputs``). Required whenever *run_config*
            enables a DFX flag; created if missing.

    Returns:
        A fresh simpler ``CallConfig``.

    Raises:
        ValueError: a DFX flag is enabled but *dfx_base* is ``None``.
    """
    from simpler.task_interface import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        CallConfig,
    )

    call_config = CallConfig()
    if dc.block_dim is not None:
        call_config.block_dim = dc.block_dim
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
            # arrows / kernel names. The one-shot path runs a clean two-pass
            # (pass 1 dep_gen → deps.json, pass 2 swimlane → clean records) and
            # sets ``co_enable_swimlane_dep_gen=False`` on the timing pass so
            # dep_gen does not perturb it. Everywhere else (the timing-pass-less
            # single-pass: prepared worker, or sim where conversion is skipped)
            # co-enable dep_gen so swimlane still has a graph in one dispatch.
            # ``enable_l2_swimlane`` is an int (0/1/2), so the ``or``/``and`` chain
            # can yield an int; the ``CallConfig.enable_dep_gen`` pybind setter
            # only accepts ``bool``. Wrap in ``bool(...)`` to avoid a TypeError.
            call_config.enable_dep_gen = bool(
                dfx.enable_dep_gen or (co_enable_swimlane_dep_gen and dfx.enable_l2_swimlane)
            )
            call_config.enable_scope_stats = dfx.enable_scope_stats
            call_config.enable_l2_swimlane = dfx.enable_l2_swimlane
            # Base dir shared by every chip; ``_submit_chip`` namespaces it per
            # dispatch (``<dfx_base>/rank{worker}/d{k}``) so per-dispatch
            # artifacts (pmu.csv, deps.json, l2_swimlane_records.json, ...) don't
            # overwrite each other — even when one card runs multiple dispatches.
            call_config.output_prefix = str(dfx_base)
    return call_config


def _submit_chip(orch: Any, callable_id: Any, task_args: Any, config: Any, worker: int) -> Any:
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
    run (see ``_dispatch.orch_fn``), so the numbering is deterministic and
    matches across the swimlane two-pass.

    When DFX is off (``output_prefix`` unset) or the dispatch is unconstrained
    (``worker < 0``) the call is forwarded unchanged.

    The codegen emits this for every rank-pinned chip dispatch; the comm-less
    single-dispatch path keeps the bare ``orch.submit_next_level(...)`` call.
    """
    base = config.output_prefix
    if not base or worker < 0:
        return orch.submit_next_level(callable_id, task_args, config, worker=worker)
    idx_map = getattr(orch, "_dfx_dispatch_idx", None)
    if idx_map is None:
        # Defensive: a caller that bypassed ``orch_fn`` (no reset) still gets
        # per-card isolation, just without a guaranteed two-pass match.
        idx_map = orch._dfx_dispatch_idx = {}
    k = idx_map.get(worker, 0)
    idx_map[worker] = k + 1
    config.output_prefix = f"{base}/rank{worker}/d{k}"
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

    for rank_dir in dfx_base.glob("rank*"):
        if not rank_dir.is_dir():
            continue
        for disp_dir in rank_dir.glob("d[0-9]*"):
            if disp_dir.is_dir():
                shutil.rmtree(disp_dir, ignore_errors=True)


def _collect_l3_swimlane(output_dir: Path, n_ranks: int, platform: str) -> None:
    """Convert each dispatch's swimlane records into a ``merged_swimlane_*.json``.

    The runtime writes ``rank{r}/d{k}/l2_swimlane_records.json`` +
    ``rank{r}/d{k}/deps.json`` per dispatch (``_submit_chip`` namespaced the dir
    by card *and* the card's k-th dispatch; dep_gen is co-enabled with
    swimlane). This best-effort post-pass runs the offline ``swimlane_converter``
    once per dispatch dir, resolving kernel names from a merged map of every
    chip callable's ``kernel_config.py`` (``next_levels/*/``). Each dispatch's
    records are single-chip, so the L2 converter applies unchanged — and a card
    that ran several (possibly different) programs keeps one swimlane per
    dispatch instead of overwriting down to the last.

    Onboard-only: the simulator emits records but not the task metadata the
    converter joins against, so conversion is skipped there (mirrors the L2
    ``_collect_dfx_artifacts`` swimlane branch). Any failure is logged, never
    raised — the raw records remain for manual conversion.
    """
    if platform.endswith("sim"):
        print(
            "Skipping L3 swimlane conversion on simulator: merged_swimlane_*.json "
            "is only generated for onboard runs (raw l2_swimlane_records.json kept)."
        )
        return

    from .runner import _generate_swimlane  # noqa: PLC0415

    # ``glob("*/")`` directory filtering is only reliable on 3.11+; filter
    # explicitly so this works on the 3.10 baseline too.
    chip_dirs = sorted(d for d in (output_dir / "next_levels").glob("*") if d.is_dir())
    merged: dict = {}
    try:
        from simpler_setup.tools.swimlane_converter import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            load_kernel_config,
        )

        for chip_dir in chip_dirs:
            kc = chip_dir / "kernel_config.py"
            if kc.exists():
                merged.update(load_kernel_config(str(kc)))
    except Exception as e:  # noqa: BLE001 - best-effort label resolution, never fatal
        print(f"Skipping L3 swimlane name_map ({type(e).__name__}: {e}); labels fall back to defaults")

    dfx_base = output_dir / "dfx_outputs"
    for r in range(n_ranks):
        rank_dir = dfx_base / f"rank{r}"
        if not rank_dir.is_dir():
            continue
        # One card may have run several dispatches: ``rank{r}/d0``, ``d1``, ...
        # Match only ``d`` + digits (the names ``_submit_chip`` emits) so an
        # unrelated diagnostic dir under rank_dir is never picked up. 3.10-safe
        # dir filter (``glob`` directory filtering is only reliable on 3.11+).
        dispatch_dirs = sorted(d for d in rank_dir.glob("d[0-9]*") if d.is_dir())
        for disp_dir in dispatch_dirs:
            records = disp_dir / "l2_swimlane_records.json"
            if not records.exists():
                continue
            # Best-effort, as documented: a write/convert failure for one
            # dispatch must not turn a successful run into a post-processing
            # crash. The raw records remain on disk for manual conversion.
            try:
                name_map_path: Path | None = None
                if merged:
                    name_map_path = disp_dir / "name_map.json"
                    name_map_path.write_text(
                        json.dumps(
                            {"level": 2, "orchestrator_name": None, "callable_id_to_name": merged},
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                # ``work_dir`` only feeds the converter's ``-k`` fallback; the
                # merged ``name_map`` passed as ``func_names`` takes precedence.
                work_dir = chip_dirs[0] if chip_dirs else output_dir
                _generate_swimlane(work_dir, disp_dir, records, func_names=name_map_path)
            except Exception as e:  # noqa: BLE001 - best-effort post-pass, never fatal
                print(
                    f"Skipping L3 swimlane conversion for {disp_dir.name} of rank {r} "
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


def _dispatch(
    w: Any,
    entry_fn: Any,
    tensors: dict[str, Any],
    chip_cids: dict[str, Any],
    sub_ids: dict[str, Any],
    call_config: Any,
    device_nums: int,
) -> None:
    """Build the orchestration closure and run it once on ``w``.

    The simpler ``Worker.run`` returns ``None`` (per-run timing is read from
    the runtime's ``[STRACE]`` log markers, simpler PR #1177).
    """
    # Fresh _keep per dispatch: it pins per-call TaskArgs alive for the run.
    _keep: list[Any] = []

    # ``world_size`` is the only worker-level scalar the entry needs; codegen
    # binds ``pld.system.world_size()`` to this kwarg uniformly across comm
    # and comm-less paths.

    def orch_fn(orch, _unused_args, _unused_cfg):
        # Reset the per-card DFX dispatch counter at the start of every run so
        # ``_submit_chip`` numbers a card's dispatches ``d0, d1, ...`` fresh each
        # pass. Two-pass swimlane reissues the same dispatch order, so pass 1
        # (deps.json) and pass 2 (records) land the same dispatch in the same
        # ``rank{w}/d{k}`` dir — letting the converter join them.
        orch._dfx_dispatch_idx = {}
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
    coerced_args: list[torch.Tensor | DeviceTensor | StackedDeviceTensor],
    config: RunConfig | None = None,
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
        config: Optional per-dispatch :class:`RunConfig`. Its per-task
            ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
            ``ring_dep_pool``, each a scalar or a per-ring list of 4 ints) size
            this dispatch's runtime ring buffers, and its
            runtime-diagnostic DFX flags (``enable_dump_args`` / ``enable_pmu``
            / ``enable_dep_gen`` / ``enable_scope_stats`` / ``enable_l2_swimlane``)
            are written per dispatch under
            ``<output_dir>/dfx_outputs/rank{r}/d{k}/`` (``d{k}`` is the card's
            k-th dispatch, so multiple — even different — chip programs on one
            card keep separate artifacts). Onboard, ``enable_l2_swimlane`` runs a
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
    dc = compiled._distributed_config
    output_dir = compiled.output_dir

    chip_callables, runtime_name = _assemble_chip_callables(compiled)
    entry_fn, alloc_fn = _load_orch_entry(output_dir)

    # Build tensor mapping from parameter names. Host torch.Tensor inputs must
    # be in shared memory before the fork; DeviceTensor inputs are device
    # pointers forwarded at submit time and need no pre-fork shared memory.
    param_infos, _, _ = compiled._get_metadata()
    tensors: dict[str, torch.Tensor | DeviceTensor | StackedDeviceTensor] = {}
    for info, arg in zip(param_infos, coerced_args, strict=True):
        # Worker-resident inputs (a DeviceTensor, or a StackedDeviceTensor whose
        # per-rank shards are each DeviceTensors) are device pointers forwarded
        # at submit time — no pre-fork shared memory needed.
        if isinstance(arg, (DeviceTensor, StackedDeviceTensor)):
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
            w = _construct_worker(dc, compiled.platform, runtime_name, num_sub)
            sub_ids, chip_cids = _register_callables(w, sub_worker_fns, chip_callables)
            w.init()
            _dispatch(w, entry_fn, tensors, chip_cids, sub_ids, call_config, len(dc.device_ids))
        finally:
            if w is not None:
                w.close()

    dfx_base = output_dir / "dfx_outputs"
    swimlane = config is not None and config.enable_l2_swimlane

    # Scope DFX artifacts to this run: drop any stale ``rank*/d{k}`` dirs from an
    # earlier (possibly larger) run before the first dispatch writes new ones.
    if config is not None:
        from .runner import _DfxOpts  # noqa: PLC0415

        if _DfxOpts.from_run_config(config).any():
            _clear_dfx_dispatch_dirs(dfx_base)

    if config is not None and config.enable_l2_swimlane and not compiled.platform.endswith("sim"):
        # Two-pass for clean timing, mirroring the L2 swimlane workflow: dep_gen
        # collection perturbs timing, so the per-dispatch task graph and the kept
        # timing come from separate dispatches.
        import dataclasses  # noqa: PLC0415

        print(
            "[swimlane] L3 swimlane enabled -> running the dispatch twice "
            "(dep_gen perturbs timing, so the graph and the timing are captured separately):"
        )
        print(
            "[swimlane] run 1/2: capturing the per-dispatch task graph (deps.json); its timing is discarded."
        )
        deps_cfg = dataclasses.replace(
            config,
            enable_l2_swimlane=False,
            enable_dep_gen=True,
            enable_pmu=0,
            enable_scope_stats=False,
            enable_dump_args=0,
        )
        _run_once(_make_call_config(dc, deps_cfg, dfx_base=dfx_base))

        print("[swimlane] run 2/2: measuring clean per-task timing (these are the reported numbers).")
        timing_cfg = dataclasses.replace(config, enable_dep_gen=False)
        _run_once(_make_call_config(dc, timing_cfg, dfx_base=dfx_base, co_enable_swimlane_dep_gen=False))
    else:
        _run_once(_make_call_config(dc, config, dfx_base=dfx_base))

    # Offline post-pass (reads the per-dispatch deps.json + records on disk).
    if swimlane:
        _collect_l3_swimlane(output_dir, len(dc.device_ids), compiled.platform)


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
        args: Positional arguments — host ``torch.Tensor`` or worker-resident
            :class:`~pypto.runtime.DeviceTensor` — matching the orchestrator's
            parameter order (in-place, or input-only for a return-style program).
        config: Optional per-dispatch :class:`RunConfig`, forwarded to
            ``__call__``. Its per-task ring-sizing overrides size this dispatch's
            runtime ring buffers, and its runtime-diagnostic DFX flags
            (``enable_dump_args`` / ``enable_pmu`` / ``enable_dep_gen`` /
            ``enable_scope_stats`` / ``enable_l2_swimlane``) are written per
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
    tensor (no ``copy_from``). Large static weights are uploaded once to a
    worker-resident :class:`~pypto.runtime.DeviceTensor` via :meth:`alloc_tensor`
    (its ``init`` source must likewise be a pre-``prepare`` shared tensor) and
    mixed in. This mirrors the runtime's ``child_memory`` example.

    Immutable input tensors may instead be supplied through
    ``inherited_host_tensors``. They are retained before the chip workers fork
    and subsequently read through their inherited copy-on-write mappings. This
    path is input-only: outputs and in-place tensors still require shared memory
    so child writes remain visible to the parent.

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
        inherited_host_tensors: Sequence[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()  # initialize Worker ABC state (_owned_tensors)
        del config  # reserved for future per-runtime overrides
        callbacks = _coalesce_callbacks(callbacks, sub_worker_overrides)
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
        self._inherited_host_tensor_ids = {id(tensor) for tensor in inherited}

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

    def _require_ready(self, op: str) -> None:
        # Worker ABC hook: device-memory ops are valid until close().
        self._require_open(op)

    @staticmethod
    def _require_forked_host_buffer(tensor: torch.Tensor, api: str, access: str) -> None:
        """Validate *tensor* is a host buffer the forked chip worker can ``access``.

        Every H2D/D2H copy runs **inside the forked chip worker**, which can only
        touch host memory it inherited at fork. So *tensor* must be a CPU,
        contiguous, **shared-memory** tensor allocated **before**
        :meth:`DistributedCompiledProgram.prepare` (call ``.share_memory_()``); a
        buffer allocated after ``prepare()`` — or a non-shared one — is invisible
        to the child.

        Args:
            tensor: The host buffer to validate.
            api: The calling API signature, woven into the error message
                (e.g. ``"copy_stacked_from(host=...)"``).
            access: The child's access verb — ``"read"`` for uploads, ``"write"``
                for read-backs.

        Raises:
            ValueError: If *tensor* is not CPU, contiguous, and shared-memory.
        """
        if not (tensor.is_shared() and tensor.is_contiguous() and tensor.device.type == "cpu"):
            raise ValueError(
                f"{api} requires a CPU, contiguous, shared-memory tensor allocated "
                f"BEFORE prepare() (call .share_memory_()). The copy runs in the forked "
                f"chip worker, which can only {access} host memory it inherited at fork."
            )

    def _prepare_init(self, init: torch.Tensor) -> torch.Tensor:
        # Worker ABC hook: the upload (``copy_to``) runs **inside the forked chip
        # worker**, so ``init`` must be a CPU, contiguous, shared-memory tensor
        # allocated **before** prepare() (see _require_forked_host_buffer). Unlike
        # L2 we cannot make a defensive ``.cpu().contiguous()`` copy: that copy
        # would live only in the parent and be invisible to the child.
        self._require_forked_host_buffer(init, "DistributedWorker.alloc_tensor(init=...)", "read")
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
        skips the per-dispatch H2D upload (``child_memory``) — the stack is
        uploaded once here and reused across every ``rt(...)`` dispatch.

        Args:
            host: A CPU, contiguous, **shared-memory** ``[B, *tail]`` tensor
                allocated BEFORE :meth:`~DistributedCompiledProgram.prepare`
                (call ``.share_memory_()``); the upload runs in the forked chip
                worker, which can only read host memory inherited at fork.
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
            for i, w in enumerate(ids):
                shards.append(
                    self.alloc_tensor(
                        tuple(host.shape[1:]),
                        host.dtype,
                        init=host[i].contiguous(),
                        worker_id=w,
                    )
                )
        except Exception:
            # Roll back any shards already uploaded so a mid-loop failure
            # (e.g. a non-shared host) never leaks device memory.
            for shard, w in zip(shards, ids, strict=False):
                self.free_tensor(shard, worker_id=w)
            raise
        return StackedDeviceTensor(shards, tuple(host.shape), tuple(ids))

    def free_stacked_tensor(self, stacked: StackedDeviceTensor) -> None:
        """Release every shard of *stacked* against its owning worker. Idempotent."""
        for shard, w in zip(stacked.shards, stacked.worker_ids, strict=True):
            self.free_tensor(shard, worker_id=w)

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
            host: A CPU, contiguous, **shared-memory** ``[B, *tail]`` tensor
                allocated BEFORE :meth:`~DistributedCompiledProgram.prepare`
                (call ``.share_memory_()``), whose shape and dtype match
                ``stacked.full_shape`` / ``stacked.dtype``. Filled in place
                (``host[i]`` receives shard ``i``). The D2H copy runs in the
                forked chip worker, which can only write to host memory it
                inherited at fork — a buffer allocated after ``prepare()`` (or a
                non-shared one) would leave *host* untouched.
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
        self._require_forked_host_buffer(host, "copy_stacked_from(host=...)", "write")
        for i, (shard, w) in enumerate(zip(stacked.shards, stacked.worker_ids, strict=True)):
            # host is contiguous + shared, so host[i] is a contiguous view at the
            # right offset into the same shm segment the child inherited at fork;
            # host[i].data_ptr() is therefore the correct cross-process D2H dst.
            self.copy_from(host[i].data_ptr(), shard.data_ptr, shard.nbytes, worker_id=w)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, config: RunConfig | None = None) -> None:
        """Dispatch one run on the primary compiled program, reusing all setup.

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
        can use different ring sizes. ``None`` reuses the program's baseline.
        """
        if self._multi_program:
            raise TypeError(
                "rt(*args) is ambiguous on a multi-program DistributedWorker; "
                "dispatch explicitly with rt.run(compiled, *args)."
            )
        return self._run_compiled(self._compiled, *args, config=config)

    def _run_compiled(
        self, compiled: DistributedCompiledProgram, *args: Any, config: RunConfig | None = None
    ) -> None:
        """Dispatch *compiled* on the shared Worker via its per-program state.

        ``config`` is an optional per-dispatch :class:`RunConfig` whose per-task
        ring-sizing overrides size this dispatch's runtime ring buffers. When
        given, a fresh ``CallConfig`` is built for this dispatch only (from the
        program's ``block_dim`` / ``aicpu_thread_num`` baseline plus the ring
        overrides), leaving the prepared, shared ``call_config`` untouched. When
        ``None``, the prepared baseline is reused with zero extra allocation.
        """
        self._require_open("run")
        from pypto.ir.compiled_program import (  # noqa: PLC0415
            _validate_device_tensor,
            _validate_stacked_tensor,
        )

        state = self._states.get(compiled)
        if state is None:
            raise ValueError(
                "DistributedWorker.run(compiled, ...) requires a DistributedCompiledProgram "
                "registered when this worker was constructed."
            )

        # Per-task ring sizing: a per-dispatch RunConfig yields a fresh
        # CallConfig for this call only (the prepared, shared one is never
        # mutated). With no RunConfig the prepared baseline is reused as-is.
        call_config = state["call_config"]
        if config is not None:
            dfx_base = compiled.output_dir / "dfx_outputs"
            call_config = _make_call_config(compiled._distributed_config, config, dfx_base=dfx_base)
            # This worker reuses one output_dir across dispatches, so stale
            # ``rank*/d{k}`` dirs from an earlier, larger run must be cleared
            # before this run rewrites ``d0, d1, ...`` (see _clear_dfx_dispatch_dirs).
            from .runner import _DfxOpts  # noqa: PLC0415

            if _DfxOpts.from_run_config(config).any():
                _clear_dfx_dispatch_dirs(dfx_base)

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
            if isinstance(arg, StackedDeviceTensor):
                _validate_stacked_tensor(arg, info)
            elif isinstance(arg, DeviceTensor):
                _validate_device_tensor(arg, info)
            elif isinstance(arg, torch.Tensor):
                if not arg.is_shared():
                    if id(arg) not in self._inherited_host_tensor_ids:
                        raise TypeError(
                            f"Parameter {info.name!r}: a non-shared host tensor must be registered "
                            "through inherited_host_tensors before the chip workers fork."
                        )
                    if info.direction.name != "In":
                        raise TypeError(
                            f"Parameter {info.name!r}: inherited host tensors are input-only; "
                            f"direction {info.direction.name} requires shared memory."
                        )
            elif not _is_simpler_tensor(arg):
                raise TypeError(
                    f"DistributedWorker parameter {info.name!r} got {type(arg).__name__}; expected a "
                    f"shared-memory torch.Tensor, a worker-resident DeviceTensor, a "
                    f"StackedDeviceTensor, or a simpler Tensor."
                )
            tensors[info.name] = arg

        _dispatch(
            self._w,
            state["entry_fn"],
            tensors,
            state["chip_cids"],
            state["sub_ids"],
            call_config,
            state["device_nums"],
        )

        # Offline post-pass (reads the per-dispatch records on disk; no worker needed).
        # Note: unlike the one-shot ``execute_distributed`` path, the prepared
        # worker reuses its forked chip children across dispatches, so it cannot
        # re-fork between a deps pass and a timing pass without tripping the
        # per-child ``halHostRegister`` cap (rc 8). It therefore runs swimlane
        # single-pass (dep_gen co-enabled), so the on-disk records include
        # dep_gen collection overhead. Use ``execute_distributed`` (one-shot) for
        # clean two-pass swimlane timing.
        if config is not None and config.enable_l2_swimlane:
            _collect_l3_swimlane(compiled.output_dir, state["device_nums"], compiled.platform)

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
        self._inherited_host_tensors = ()
        self._inherited_host_tensor_ids.clear()

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
        ring-sizing overrides size this dispatch's runtime ring buffers without
        touching the prepared program's shared config. In a multi-program worker
        each program can therefore be dispatched with its own ring sizes (e.g.
        a larger ``ring_task_window`` for prefill than for decode). ``None``
        reuses the program's baseline.
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
