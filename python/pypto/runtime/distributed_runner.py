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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np  # pyright: ignore[reportMissingImports]
import torch

from pypto.ir.comm_manifest import COMM_MANIFEST_FILENAME, COMM_MANIFEST_VERSION

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


def execute_distributed(  # noqa: PLR0912
    compiled: DistributedCompiledProgram,
    coerced_args: list[torch.Tensor],
    config: Any = None,
) -> None:
    """Execute a distributed compiled program via simpler Worker(level=3).

    Args:
        compiled: The DistributedCompiledProgram instance.
        coerced_args: List of coerced torch.Tensor arguments.
        config: Optional run configuration (unused for now).
    """
    from simpler.task_interface import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        CallConfig,
    )
    from simpler.worker import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        Worker,
    )

    from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

    dc = compiled._distributed_config
    output_dir = compiled.output_dir

    # 1. Build ChipCallable for each chip-level task (under next_levels/{name}/)
    from pypto.pypto_core.ir import FunctionType  # noqa: PLC0415

    chip_callables: dict[str, Any] = {}
    runtime_name = "tensormap_and_ringbuffer"
    next_levels_dir = output_dir / "next_levels"
    for func in compiled._program.functions.values():
        if func.func_type == FunctionType.Orchestration:
            chip_dir = next_levels_dir / func.name
            if chip_dir.exists():
                chip_callable, runtime_name = compile_and_assemble(chip_dir, compiled.platform)
                chip_callables[func.name] = chip_callable

    if not chip_callables:
        raise RuntimeError(f"No chip-level tasks found in {next_levels_dir}")

    # 2. Load the generated Python orchestration module
    orch_path = output_dir / "orchestration" / "host_orch.py"
    if not orch_path.exists():
        raise FileNotFoundError(
            f"Generated orchestration not found at {orch_path}. Did the codegen produce distributed output?"
        )
    orch_module = _load_generated_module(orch_path)

    # Find the entry function in the generated module
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

    # 3. Build tensor mapping from parameter names
    param_infos, _, _ = compiled._get_metadata()
    tensors: dict[str, torch.Tensor] = {}
    for info, arg in zip(param_infos, coerced_args, strict=True):
        if not arg.is_shared():
            arg.share_memory_()
        tensors[info.name] = arg

    # 3b. Pre-fork: allocate HOST-level intermediate tensors so the POSIX
    # shared-memory mappings exist before w.init() forks subworker /
    # chip-worker child processes. Mappings created after fork are not
    # visible to inherited children.
    alloc_fn = getattr(orch_module, "_alloc_intermediates", None)
    if alloc_fn is not None:
        alloc_fn(tensors)

    # 4. Load SubWorker callables from sub_workers/*.py files
    sub_worker_fns: dict[str, Any] = {}
    sub_workers_dir = output_dir / "sub_workers"
    if sub_workers_dir.exists():
        for py_file in sorted(sub_workers_dir.glob("*.py")):
            mod = _load_generated_module(py_file)
            fn_name = py_file.stem
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                sub_worker_fns[fn_name] = fn

    # 5. Build per-chip bootstrap configs from the AOT comm manifest emitted at
    #    compile time (output_dir/orchestration/comm_manifest.json). Comm-less
    #    programs (no CommGroup declared) skip the manifest entirely and the
    #    runner stays on the existing comm-less Worker path.
    manifest_path = output_dir / "orchestration" / COMM_MANIFEST_FILENAME
    manifest: dict[str, Any] | None = None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError:
        pass

    chip_bootstrap_configs: list[Any] | None = None
    rootinfo_path: str | None = None
    if manifest is not None:
        # ``mkstemp`` returns a unique, unpredictable path and atomically creates
        # the file (mode 0o600) — safer than a PID-derived name, which is racy
        # under PID recycling. The fd is closed immediately because HCCL only
        # needs the path: comm bring-up overwrites the file on rank-0 and
        # reads it on the other ranks.
        rootinfo_fd, rootinfo_path = tempfile.mkstemp(prefix="pypto_distributed_rootinfo_", suffix=".bin")
        os.close(rootinfo_fd)
        chip_bootstrap_configs = _build_chip_bootstrap_configs_from_manifest(manifest, dc, rootinfo_path)

    num_sub = max(dc.num_sub_workers, len(sub_worker_fns))
    w = Worker(
        level=3,
        device_ids=dc.device_ids,
        num_sub_workers=num_sub,
        platform=compiled.platform,
        runtime=runtime_name,
        chip_bootstrap_configs=chip_bootstrap_configs,
    )

    # 6. Register SubWorker callables
    sub_ids: dict[str, int] = {}
    for name, fn in sub_worker_fns.items():
        sub_ids[name] = w.register(fn)

    w.init()

    # 7. Build the orchestration closure and execute
    _keep: list[Any] = []

    # Codegen always emits ``contexts`` in the entry signature; for comm-less
    # programs ``w.chip_contexts`` is an empty list and the body simply ignores
    # it, so the runner uses a single uniform call shape.
    def orch_fn(orch, _unused_args, _unused_cfg):
        entry_fn(
            orch,
            _unused_args,
            call_config,
            tensors=tensors,
            callables=chip_callables,
            sub_ids=sub_ids,
            _keep=_keep,
            contexts=w.chip_contexts,
        )

    call_config = CallConfig()
    call_config.block_dim = dc.block_dim
    call_config.aicpu_thread_num = dc.aicpu_thread_num

    try:
        w.run(orch_fn)
    finally:
        w.close()
        if rootinfo_path is not None:
            try:
                os.unlink(rootinfo_path)
            except FileNotFoundError:
                pass
