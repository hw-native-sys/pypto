# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-time CommGroup manifest support (v2 schema).

Lifts ``program.comm_groups`` into a JSON-safe dict suitable for serialisation
under ``output_dir/orchestration/<COMM_MANIFEST_FILENAME>``. The runtime
re-enters the output directory and reads the file via
``pypto.runtime.distributed_runner._build_chip_bootstrap_configs_from_manifest``
without needing the live ``Program`` object.

Two halves of the AOT pipeline:

  compile-time:  Program → ``lift_comm_manifest`` → dict (JSON-safe)
  on disk:       output_dir/orchestration/<COMM_MANIFEST_FILENAME>
  runtime:       dict   →  _build_chip_bootstrap_configs_from_manifest(...)
                                                            → list[ChipBootstrapConfig]

v2 schema (CommGroups are pass-inferred, not user-declared):

* ``devices``: list of physical device ids covered by the group; **empty list
  = all devices** (resolved by the driver against
  ``DistributedConfig.device_ids``).
* ``slots``: list of allocation specs, each a 1:1 image of
  ``simpler.task_interface.ChipBufferSpec``. ``load_from_host`` /
  ``store_to_host`` are bool flags here — the specific host tensor binding
  lives on the alloc op, not on this allocation spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ConstInt, Program

# Manifest filename under output_dir/orchestration/. Keep in sync with the
# runtime loader in pypto.runtime.distributed_runner.
COMM_MANIFEST_FILENAME = "comm_manifest.json"
COMM_MANIFEST_VERSION = 2


# Maps PyPTO DataType to the dtype-string convention simpler's ChipBufferSpec
# expects (numpy/torch style, e.g. "float32"). DataType.to_string() returns
# "fp32" / "bfloat16" which simpler does not understand; this table is the
# explicit single source of truth.
_SIMPLER_DTYPE_STR: dict[DataType, str] = {
    DataType.FP32: "float32",
    DataType.FP16: "float16",
    DataType.BF16: "bfloat16",
    DataType.INT8: "int8",
    DataType.INT16: "int16",
    DataType.INT32: "int32",
    DataType.INT64: "int64",
    DataType.UINT8: "uint8",
}


def _simpler_dtype_str(dtype: DataType) -> str:
    try:
        return _SIMPLER_DTYPE_STR[dtype]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported WindowBuffer dtype {dtype!r} for ChipBufferSpec; "
            f"add an entry to _SIMPLER_DTYPE_STR. "
            f"Known dtypes: {sorted(d.to_string() for d in _SIMPLER_DTYPE_STR)}"
        ) from exc


def lift_comm_manifest(program: Program) -> dict[str, Any] | None:
    """Lift ``program.comm_groups`` into a JSON-safe dict for AOT serialization.

    Returns ``None`` when the program declares no CommGroup — callers should
    skip emitting / loading the manifest entirely (preserving the comm-less
    path used by multi_chip_dispatch and parallel_reduce).

    Current v2 supports a single CommGroup with literal ``int`` per-slot
    ``size``. Symbolic sizes raise ``RuntimeError`` at compile time — that's
    a better failure point than runtime, since the program author can fix it
    without redeploying.
    """
    comm_groups = list(program.comm_groups)
    if not comm_groups:
        return None
    if len(comm_groups) > 1:
        raise RuntimeError(
            f"distributed_runner currently supports at most one CommGroup per program, got {len(comm_groups)}"
        )

    group = comm_groups[0]

    slots_data: list[dict[str, Any]] = []
    for slot in group.slots:
        size_const = slot.size if isinstance(slot.size, ConstInt) else None
        if size_const is None:
            raise RuntimeError(
                f"dynamic WindowBuffer size is not supported yet "
                f"(slot {slot.name!r}); declare size as a literal int"
            )
        slots_data.append(
            {
                "name": slot.name,
                "dtype": _simpler_dtype_str(slot.dtype),
                "size": int(size_const.value),
                "bits_per_element": slot.dtype.get_bit(),
                "load_from_host": bool(slot.load_from_host),
                "store_to_host": bool(slot.store_to_host),
            }
        )

    return {
        "version": COMM_MANIFEST_VERSION,
        "comm_groups": [
            {
                # Empty list = all devices (resolved by the driver against
                # DistributedConfig.device_ids).
                "devices": [int(d) for d in group.devices],
                "slots": slots_data,
            }
        ],
    }


def emit_comm_manifest(program: Program, output_dir: Path | str) -> Path | None:
    """Lift ``program.comm_groups`` and write the JSON manifest to disk.

    Writes ``output_dir/orchestration/<COMM_MANIFEST_FILENAME>``. Returns the
    written path, or ``None`` when the program declares no CommGroup (no file
    is created — comm-less programs are unaffected).

    This is the compile-time emission point. The runner re-enters
    ``output_dir`` and reads the file via
    ``pypto.runtime.distributed_runner._build_chip_bootstrap_configs_from_manifest``,
    so a ``CompiledProgram`` instance can be reconstructed from the output
    directory alone — no live ``Program`` object required.
    """
    manifest = lift_comm_manifest(program)
    if manifest is None:
        return None
    orch_dir = Path(output_dir) / "orchestration"
    orch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = orch_dir / COMM_MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest_path


__all__ = [
    "COMM_MANIFEST_FILENAME",
    "COMM_MANIFEST_VERSION",
    "lift_comm_manifest",
    "emit_comm_manifest",
]
