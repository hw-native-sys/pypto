# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Subprocess entry point that captures a dep_gen ``deps.json`` for swimlane.

When ``enable_l2_swimlane`` is requested on an onboard platform, the swimlane
converter needs a task graph that only a dep_gen run produces. dep_gen and
swimlane cannot share one in-process run: the runtime's per-run finalize does
not reliably reclaim the SVM host-register mappings the DFX collectors allocate,
so a second DFX run in the same process hits the registration cap
(``halHostRegister`` rc 8). Running the dep_gen pass in a **separate process**
sidesteps that — the OS reclaims all device/SVM state on exit, leaving the
in-process swimlane pass a clean slate.

This module is invoked by :mod:`pypto.runtime.runner` as
``python -m pypto.runtime._dep_gen_capture <spec.json>``. The spec is a small
JSON file (see :func:`pypto.runtime.runner` for the writer) describing how to
reconstruct the orchestration arguments:

* ``mode="golden"`` — regenerate the exact inputs from ``golden.py`` (the test
  harness path; inputs are deterministic, so the captured graph is faithful).
* ``mode="argspec"`` — the compiled-program path. The task graph can be routed
  by tensor *values* (e.g. paged-attention ``block_tables`` / ``seq_lens``), so
  host tensors are saved verbatim and reloaded with real data and scalars are
  preserved exactly; only device-resident tensors (unreachable from a fresh
  child) fall back to zero-filled tensors of the recorded shape.
"""

import ctypes
import json
import sys
from pathlib import Path

import torch


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a ``"float32"``-style name to a ``torch.dtype``."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype name: {name!r}")
    return dtype


def _build_argspec_orch_args(args_spec: list[dict]):
    """Rebuild orch args from a recorded spec (see ``runner._build_args_spec``).

    Host tensors are reloaded from disk with real data (so data-as-control
    inputs route the same graph); device-resident tensors are rebuilt as zeros;
    scalars are reconstructed exactly.

    Returns ``(orch_args, keepalive)``. ``orch_args`` holds only raw host
    pointers, so *keepalive* (the backing tensors/scalars) MUST outlive the
    ``execute_on_device`` call or the H2D copy reads freed memory.
    """
    from .runner import _coerced_to_orch_args  # noqa: PLC0415

    coerced: list = []
    for entry in args_spec:
        kind = entry["kind"]
        if kind == "tensor_file":
            # Saved contiguous in runner._build_args_spec, so no re-contiguous needed.
            coerced.append(torch.load(entry["path"]))
        elif kind == "tensor_zeros":
            coerced.append(torch.zeros(entry["shape"], dtype=_torch_dtype(entry["dtype"])))
        elif kind == "scalar":
            ctype = getattr(ctypes, entry["ctype"])
            coerced.append(ctype(entry["value"]))
        else:
            raise ValueError(f"Unknown arg spec kind: {kind!r}")
    return _coerced_to_orch_args(coerced), coerced


def _build_golden_orch_args(golden_path: Path):
    """Regenerate orch args from ``golden.py``'s ``generate_inputs`` (faithful).

    Returns ``(orch_args, keepalive)`` — see :func:`_build_argspec_orch_args`
    for why the backing tensors must be kept alive past the device run.
    """
    from .device_runner import build_orch_args_from_inputs  # noqa: PLC0415
    from .runner import _load_golden_module  # noqa: PLC0415

    golden_module = _load_golden_module(golden_path, module_name="_golden_dep_gen")

    result = golden_module.generate_inputs({"name": "Default"})
    output_names = set(getattr(golden_module, "__outputs__", []))
    orch_args, all_tensors, inputs, outputs = build_orch_args_from_inputs(result, output_names)
    return orch_args, (all_tensors, inputs, outputs)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m pypto.runtime._dep_gen_capture <spec.json>", file=sys.stderr)
        return 2

    spec = json.loads(Path(argv[0]).read_text(encoding="utf-8"))

    print("[swimlane:dep_gen subprocess] capturing the task dependency graph for the swimlane; "
          "the following compile/run output is for deps.json only (timing is discarded).")

    from .device_runner import compile_and_assemble, execute_on_device  # noqa: PLC0415

    work_dir = Path(spec["work_dir"])
    platform = spec["platform"]
    device_id = int(spec["device_id"])
    dfx_dir = Path(spec["dfx_dir"])
    pto_isa_commit = spec.get("pto_isa_commit")
    level = int(spec.get("level", 2))

    chip_callable, runtime_name, runtime_config = compile_and_assemble(work_dir, platform, pto_isa_commit)

    if spec["mode"] == "golden":
        orch_args, _keepalive = _build_golden_orch_args(Path(spec["golden_path"]))
    elif spec["mode"] == "argspec":
        orch_args, _keepalive = _build_argspec_orch_args(spec["args"])
    else:
        raise ValueError(f"Unknown spec mode: {spec['mode']!r}")
    # ``orch_args`` stores raw host pointers; ``_keepalive`` (the backing
    # tensors) must stay referenced until execute_on_device's H2D copy completes.

    # Caller-supplied block_dim / aicpu_thread_num take precedence; fall back to
    # the values baked into kernel_config.py so the captured graph matches the
    # in-process run's scheduling.
    block_dim = spec.get("block_dim")
    if block_dim is None:
        block_dim = runtime_config.get("block_dim")
    aicpu_thread_num = spec.get("aicpu_thread_num")
    if aicpu_thread_num is None:
        aicpu_thread_num = runtime_config.get("aicpu_thread_num")

    dfx_dir.mkdir(parents=True, exist_ok=True)
    execute_on_device(
        chip_callable,
        orch_args,
        platform,
        runtime_name,
        device_id,
        level=level,
        block_dim=block_dim,
        aicpu_thread_num=aicpu_thread_num,
        output_prefix=str(dfx_dir),
        enable_dep_gen=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
