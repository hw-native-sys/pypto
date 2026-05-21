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
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np  # pyright: ignore[reportMissingImports]
import torch

from pypto.ir.comm_manifest import COMM_MANIFEST_FILENAME, COMM_MANIFEST_VERSION

from .device_memory import DeviceMemoryHandle
from .device_tensor import DeviceTensor

if TYPE_CHECKING:
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig


# Placeholder dtype for ChipBufferSpec: WindowBuffer carries no dtype (matching
# MemRef), and simpler does not consume this field at runtime.
_OPAQUE_DTYPE = "opaque"


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
# Manifest → ChipBootstrapConfig translation (runtime side).
#
# The compile-time half (Program → manifest dict, written to
# ``output_dir/orchestration/<COMM_MANIFEST_FILENAME>``) lives in
# ``pypto.ir.comm_manifest``. The runtime only consumes the file.
# ---------------------------------------------------------------------------


def _build_chip_bootstrap_configs_from_manifest(
    manifest: dict[str, Any] | None,
    dc: DistributedConfig,
    rootinfo_path: str,
) -> list[Any] | None:
    """Materialise a manifest dict into a list of ``ChipBootstrapConfig``.

    Returns ``None`` for ``manifest is None`` (no CommGroup).

    The CommGroup's ``devices`` is a list of physical device ids — **an empty
    list means "all devices"** (every entry of ``dc.device_ids``). Covered
    ranks receive comm + buffers; the rest stay comm-less so simpler's
    per-device cfg list stays 1:1 with ``device_ids``.
    """
    if manifest is None:
        return None

    from simpler.task_interface import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        ChipBootstrapConfig,
        ChipBufferSpec,
        ChipCommBootstrapConfig,
    )

    version = manifest.get("version")
    if version != COMM_MANIFEST_VERSION:
        raise RuntimeError(
            f"comm manifest version mismatch: file is {version!r}, runtime expects {COMM_MANIFEST_VERSION}"
        )

    comm_groups = manifest["comm_groups"]
    if len(comm_groups) != 1:
        raise RuntimeError(
            f"comm manifest must declare exactly one CommGroup (got {len(comm_groups)}); "
            "if you need multiple groups, file a follow-up — currently unsupported."
        )
    group = comm_groups[0]

    n_devices = len(dc.device_ids)
    devices_field = list(group.get("devices", []))
    if not devices_field:
        # Empty list = all devices.
        covered_ranks = list(range(n_devices))
    else:
        covered_ranks = sorted(int(d) for d in devices_field)
        if any(d < 0 or d >= n_devices for d in covered_ranks):
            raise RuntimeError(
                f"CommGroup devices {covered_ranks!r} contains entries outside "
                f"DistributedConfig.device_ids range [0, {n_devices})"
            )
    nranks_value = len(covered_ranks)

    specs: list[Any] = []
    for slot in group["slots"]:
        nbytes = int(slot["nbytes"])
        specs.append(
            ChipBufferSpec(
                name=slot["name"],
                dtype=_OPAQUE_DTYPE,
                count=nbytes,
                nbytes=nbytes,
                load_from_host=bool(slot.get("load_from_host", False)),
                store_to_host=bool(slot.get("store_to_host", False)),
            )
        )

    window_size = sum(spec.nbytes for spec in specs)

    covered_set = set(covered_ranks)
    cfgs: list[Any] = []
    rank_in_group = 0
    for r in range(n_devices):
        if r in covered_set:
            cfgs.append(
                ChipBootstrapConfig(
                    comm=ChipCommBootstrapConfig(
                        rank=rank_in_group,
                        nranks=nranks_value,
                        rootinfo_path=rootinfo_path,
                        window_size=window_size,
                    ),
                    buffers=specs,
                )
            )
            rank_in_group += 1
        else:
            # Devices outside the group: skip HCCL bring-up entirely.
            cfgs.append(ChipBootstrapConfig())
    return cfgs


# ---------------------------------------------------------------------------
# Setup steps shared by the one-shot ``execute_distributed`` path and the
# reusable ``DistributedRuntime`` handle. Keeping them as free functions lets
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


def _load_orch_entry(output_dir: Path) -> tuple[Any, Any]:
    """Load the generated ``host_orch.py`` and return ``(entry_fn, alloc_fn)``.

    ``alloc_fn`` is the optional ``_alloc_intermediates(tensors)`` that
    pre-allocates HOST-level scratch tensors (``None`` when absent).
    """
    orch_path = output_dir / "orchestration" / "host_orch.py"
    if not orch_path.exists():
        raise FileNotFoundError(
            f"Generated orchestration not found at {orch_path}. Did the codegen produce distributed output?"
        )
    orch_module = _load_generated_module(orch_path)

    entry_fn = None
    for attr_name in ("entry", "host_orch"):
        entry_fn = getattr(orch_module, attr_name, None)
        if entry_fn is not None:
            break
    if entry_fn is None:
        for name in dir(orch_module):
            obj = getattr(orch_module, name)
            if callable(obj) and not name.startswith("_"):
                entry_fn = obj
                break
    if entry_fn is None:
        raise RuntimeError(f"No entry function found in {orch_path}")

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


def _build_chip_bootstrap(output_dir: Path, dc: DistributedConfig) -> tuple[list[Any] | None, str | None]:
    """Build per-chip comm bootstrap configs from the AOT comm manifest.

    Comm-less programs (no CommGroup declared) have no manifest and return
    ``(None, None)``; the runner stays on the comm-less Worker path.
    """
    manifest_path = output_dir / "orchestration" / COMM_MANIFEST_FILENAME
    manifest: dict[str, Any] | None = None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError:
        return None, None

    # ``mkstemp`` returns a unique, unpredictable path and atomically creates
    # the file (mode 0o600) — safer than a PID-derived name, which is racy under
    # PID recycling. The fd is closed immediately because HCCL only needs the
    # path: comm bring-up overwrites the file on rank-0 and reads it elsewhere.
    rootinfo_fd, rootinfo_path = tempfile.mkstemp(prefix="pypto_distributed_rootinfo_", suffix=".bin")
    os.close(rootinfo_fd)
    chip_bootstrap_configs = _build_chip_bootstrap_configs_from_manifest(manifest, dc, rootinfo_path)
    return chip_bootstrap_configs, rootinfo_path


def _construct_worker(
    dc: DistributedConfig,
    platform: str,
    runtime_name: str,
    chip_bootstrap_configs: list[Any] | None,
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
        chip_bootstrap_configs=chip_bootstrap_configs,
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


def _merge_sub_worker_overrides(
    loaded: dict[str, Any], overrides: dict[str, Callable[..., Any]] | None
) -> dict[str, Any]:
    """Merge user sub-worker overrides onto the codegen-loaded set (by name).

    Each override replaces the generated placeholder for an existing sub-worker.
    Overriding a name the program does not declare is rejected: it would register
    an unused callable while the generated orchestrator kept calling the
    placeholder (a silent no-op the caller almost never intends — usually a typo).
    """
    if not overrides:
        return loaded
    unknown = sorted(set(overrides) - set(loaded))
    if unknown:
        raise ValueError(
            f"sub_worker_overrides names {unknown} are not sub-workers of this "
            f"program. Available sub-workers: {sorted(loaded)}."
        )
    return {**loaded, **overrides}


def _make_call_config(dc: DistributedConfig) -> Any:
    """Build a simpler ``CallConfig`` from the distributed config."""
    from simpler.task_interface import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        CallConfig,
    )

    call_config = CallConfig()
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
) -> None:
    """Build the orchestration closure and run it once on ``w``."""
    # Fresh _keep per dispatch: it pins per-call TaskArgs alive for the run.
    _keep: list[Any] = []

    # Codegen always emits ``contexts`` in the entry signature; comm-less programs
    # ignore it (``pld.system.world_size`` lowers to ``len(contexts)``, never used
    # without comm). Runtime PR #817 removed the static ``Worker.chip_contexts``
    # bootstrap in favour of dynamic ``orch.allocate_domain`` (driven from inside
    # the orch body), so there is no worker-level context list to pass — comm-less
    # programs get an empty list, matching the prior comm-less semantics.
    contexts: list[Any] = []

    def orch_fn(orch, _unused_args, _unused_cfg):
        entry_fn(
            orch,
            _unused_args,
            call_config,
            tensors=tensors,
            callables=chip_cids,
            sub_ids=sub_ids,
            _keep=_keep,
            contexts=contexts,
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
    :meth:`DistributedCompiledProgram.prepare` → :class:`DistributedRuntime`.

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
    chip_bootstrap_configs, rootinfo_path = _build_chip_bootstrap(output_dir, dc)

    num_sub = max(dc.num_sub_workers, len(sub_worker_fns))
    w = _construct_worker(dc, compiled.platform, runtime_name, chip_bootstrap_configs, num_sub)
    sub_ids, chip_cids = _register_callables(w, sub_worker_fns, chip_callables)
    w.init()

    try:
        _dispatch(w, entry_fn, tensors, chip_cids, sub_ids, _make_call_config(dc))
    finally:
        w.close()
        if rootinfo_path is not None:
            try:
                os.unlink(rootinfo_path)
            except FileNotFoundError:
                pass


class DistributedRuntime(DeviceMemoryHandle):
    """Reusable L3 execution handle: prepare once, dispatch many.

    Holds an initialized simpler ``Worker(level=3)`` plus all setup artifacts
    (chip callables, host_orch entry, sub-worker fns, comm bootstrap) so the
    expensive setup — ``compile_and_assemble``, generated-module loading, Worker
    construction + registration + ``init()`` (fork) — happens exactly once.

    Mirrors the L2 ``with Worker(...)`` reuse block: it exposes device-memory
    helpers (:meth:`malloc`, :meth:`copy_to`, :meth:`copy_from`, :meth:`free`,
    :meth:`alloc_tensor`) so callers can build worker-resident
    :class:`~pypto.runtime.DeviceTensor` buffers that survive across dispatches,
    then call ``rt(*device_args)`` repeatedly.

    Per-call IO buffers (inputs **and** outputs) are shared-memory host
    ``torch.Tensor`` objects allocated **before** :meth:`prepare` and reused in
    place across dispatches — the forked chip worker reads/writes them through
    the inherited shared mapping, and outputs are read straight back from the
    tensor (no ``copy_from``). Large static weights are uploaded once to a
    worker-resident :class:`~pypto.runtime.DeviceTensor` via :meth:`alloc_tensor`
    (its ``init`` source must likewise be a pre-``prepare`` shared tensor) and
    mixed in. This mirrors the runtime's ``child_memory`` example.

    ``sub_worker_overrides`` replaces a generated sub-worker placeholder (matched
    by name) with a caller-supplied callable — e.g. a real sampling closure in
    place of the codegen stub. Each name must be a sub-worker the program
    declares; an unknown name raises ``ValueError``.

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
        compiled: DistributedCompiledProgram,
        config: Any = None,
        *,
        sub_worker_overrides: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        del config  # reserved for future per-runtime overrides
        dc = compiled._distributed_config
        self._chip_callables, runtime_name = _assemble_chip_callables(compiled)
        self._entry_fn, alloc_fn = _load_orch_entry(compiled.output_dir)
        sub_worker_fns = _load_sub_worker_fns(compiled.output_dir)
        sub_worker_fns = _merge_sub_worker_overrides(sub_worker_fns, sub_worker_overrides)
        chip_bootstrap_configs, self._rootinfo_path = _build_chip_bootstrap(compiled.output_dir, dc)

        num_sub = max(dc.num_sub_workers, len(sub_worker_fns))
        self._w = _construct_worker(dc, compiled.platform, runtime_name, chip_bootstrap_configs, num_sub)
        self._sub_ids, self._chip_cids = _register_callables(self._w, sub_worker_fns, self._chip_callables)

        # Allocate HOST-level intermediate scratch tensors ONCE, before init()
        # forks. They are reused (by name) across every dispatch; per-call
        # inputs are merged on top in __call__.
        self._base_tensors: dict[str, Any] = {}
        if alloc_fn is not None:
            alloc_fn(self._base_tensors)

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

        self._call_config = _make_call_config(dc)
        # Cache param metadata once: the dispatch contract is "setup once, run
        # many", so re-extracting it on every __call__ would be wasted work.
        self._param_infos, _, _ = compiled._get_metadata()
        self._closed = False

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
                "DistributedRuntime worker has no active orchestrator; the chip hierarchy was not started."
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

    # ``alloc_tensor`` / ``free_tensor`` are inherited from DeviceMemoryHandle.
    # Only the two behaviours that genuinely differ from L2 are overridden below:
    # the readiness guard (open vs. closed) and the host-init upload policy (the
    # upload runs in a forked chip worker, so no defensive copy is possible).

    _WORKER_KIND = "chip worker"

    def _require_ready(self, op: str) -> None:
        # DeviceMemoryHandle hook: device-memory ops are valid until close().
        self._require_open(op)

    def _prepare_init(self, init: torch.Tensor) -> torch.Tensor:
        # DeviceMemoryHandle hook: the upload (``copy_to``) runs **inside the
        # forked chip worker**, so ``init`` must be a CPU, contiguous,
        # shared-memory tensor allocated **before**
        # :meth:`DistributedCompiledProgram.prepare` (call ``.share_memory_()``).
        # Unlike L2 we cannot make a defensive ``.cpu().contiguous()`` copy: that
        # copy would live only in the parent and be invisible to the child.
        if not (init.is_shared() and init.is_contiguous() and init.device.type == "cpu"):
            raise ValueError(
                "DistributedRuntime.alloc_tensor(init=...) requires a CPU, contiguous, "
                "shared-memory tensor allocated BEFORE prepare() (call .share_memory_()). "
                "The upload runs in the forked chip worker, which can only read host "
                "memory it inherited at fork."
            )
        return init

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, config: Any = None) -> None:
        """Dispatch one run on the held Worker, reusing all setup.

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
        """
        del config  # reserved for future per-call overrides
        self._require_open("__call__")
        from pypto.ir.compiled_program import _validate_device_tensor  # noqa: PLC0415

        param_infos = self._param_infos
        n_params = len(param_infos)
        if len(args) != n_params:
            raise TypeError(
                f"DistributedRuntime expects {n_params} arguments (in-place, one per parameter), "
                f"got {len(args)}. Parameters: {[p.name for p in param_infos]}"
            )

        tensors: dict[str, Any] = dict(self._base_tensors)
        for info, arg in zip(param_infos, args, strict=True):
            if isinstance(arg, DeviceTensor):
                _validate_device_tensor(arg, info)
            elif isinstance(arg, torch.Tensor):
                if not arg.is_shared():
                    raise TypeError(
                        f"Parameter {info.name!r}: a host torch.Tensor passed to a DistributedRuntime "
                        f"must be shared memory allocated BEFORE prepare() (call .share_memory_() and "
                        f"reuse the same buffer across dispatches), so the forked chip worker can see "
                        f"it. Got a non-shared tensor."
                    )
            elif not _is_continuous_tensor(arg):
                raise TypeError(
                    f"DistributedRuntime parameter {info.name!r} got {type(arg).__name__}; expected a "
                    f"shared-memory torch.Tensor, a worker-resident DeviceTensor, or a ContinuousTensor."
                )
            tensors[info.name] = arg

        _dispatch(self._w, self._entry_fn, tensors, self._chip_cids, self._sub_ids, self._call_config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _require_open(self, op: str) -> None:
        if self._closed:
            raise RuntimeError(f"DistributedRuntime.{op}() called after close()")

    def close(self) -> None:
        """Release the Worker and comm rootinfo file. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._w.close()
        finally:
            if self._rootinfo_path is not None:
                try:
                    os.unlink(self._rootinfo_path)
                except FileNotFoundError:
                    pass

    def __enter__(self) -> DistributedRuntime:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
