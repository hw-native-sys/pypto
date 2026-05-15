# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Re-execute an existing ``build_output/<jit_dir>/`` directory.

Debug-only entry point for the "I edited a kernel cpp by hand, now re-run
with DFX (PMU / swimlane / dump_tensor / dep_gen) enabled" workflow.

Reuses :func:`pypto.runtime.runner.execute_compiled`, so the device-side
execution path is identical to the normal :func:`pypto.runtime.run` flow.
The added value is:

1. A friendlier signature for the replay use case
   (``replay(work_dir, *tensors, config=...)``) — no IR / ``@pl.program``
   needed.
2. Pre-flight invalidation of cached kernel/orchestration binaries so a
   hand-edited cpp is actually picked up on the next call. Without this,
   ``compile_and_assemble`` would silently load a stale ``.so``/``.bin``
   built from the previous version of the cpp.

CLI::

    python -m pypto.runtime.debug.replay build_output/<jit_dir>/ \\
        --pmu 2 --swimlane

Python::

    from pypto.runtime.debug import replay
    from pypto.runtime import RunConfig
    replay(
        "build_output/_jit_xxx/",
        a, b, c,
        config=RunConfig(platform="a2a3sim", enable_pmu=2, enable_l2_swimlane=True),
    )
"""

from __future__ import annotations

import argparse
import importlib.util
from ctypes import _SimpleCData
from pathlib import Path

import torch

from pypto.runtime.device_tensor import DeviceTensor
from pypto.runtime.runner import RunConfig, _DfxOpts, execute_compiled

__all__ = ["replay", "invalidate_binary_cache"]


def invalidate_binary_cache(work_dir: Path | str) -> None:
    """Remove cached kernel/orchestration binaries under *work_dir*.

    Both ``<work_dir>/cache/*.bin`` (the pre-build cache written by
    ``prebuild_binaries``) and the sibling ``.so`` / ``.o`` files next to
    each cpp are deleted. CPP sources are untouched, so the next
    ``compile_and_assemble`` rebuilds from source and picks up hand-edits.

    Safe to call when nothing is cached — silently no-ops on missing
    files / directories.
    """
    work_dir = Path(work_dir)
    cache_dir = work_dir / "cache"
    if cache_dir.is_dir():
        for f in cache_dir.glob("*.bin"):
            f.unlink()
    for sub in ("kernels", "orchestration"):
        root = work_dir / sub
        if not root.is_dir():
            continue
        for ext in ("*.so", "*.o"):
            for f in root.rglob(ext):
                f.unlink()


def replay(
    work_dir: Path | str,
    *tensors: torch.Tensor | DeviceTensor | _SimpleCData,
    config: RunConfig | None = None,
    recompile: bool = True,
) -> None:
    """Re-execute an existing ``build_output/<jit_dir>/`` with new tensors.

    Args:
        work_dir: A ``build_output/<jit_dir>/`` produced by a prior
            ``ir.compile`` / ``run`` call. Must contain ``kernel_config.py``,
            ``orchestration/`` and ``kernels/``.
        *tensors: Positional ``torch.Tensor`` (host), :class:`DeviceTensor`,
            or ctypes scalar arguments matching the orchestration entry's
            parameter order. Outputs are written in-place into the
            corresponding host tensors.
        config: Run configuration (platform, device_id, DFX flags, ...).
            Defaults to ``RunConfig()``.
        recompile: When ``True`` (default), invalidate cached kernel /
            orchestration binaries via :func:`invalidate_binary_cache` so
            hand-edited cpps are picked up. Set to ``False`` to reuse
            cached binaries (faster re-runs when no cpp has been modified).

    Raises:
        FileNotFoundError: If *work_dir* does not contain ``kernel_config.py``.
    """
    config = config or RunConfig()
    work_dir = Path(work_dir)
    if not (work_dir / "kernel_config.py").exists():
        raise FileNotFoundError(
            f"replay(): {work_dir} is not a build_output directory "
            f"(missing kernel_config.py)"
        )
    if recompile:
        invalidate_binary_cache(work_dir)
    execute_compiled(
        work_dir,
        list(tensors),
        platform=config.platform,
        device_id=config.device_id,
        pto_isa_commit=config.pto_isa_commit,
        dfx=_DfxOpts.from_run_config(config),
    )


def _load_inputs_from_golden(work_dir: Path) -> list[torch.Tensor]:
    """Load ordered input tensors from ``<work_dir>/golden.py``.

    Mirrors the loading pattern in ``runner._execute_on_device`` so the CLI
    can run without the user writing a Python harness for simple cases.
    Insertion order of ``generate_inputs``'s returned dict is the
    orchestration parameter order.
    """
    golden_path = work_dir / "golden.py"
    if not golden_path.exists():
        raise FileNotFoundError(
            f"replay CLI requires {golden_path} to load inputs; "
            f"call replay() from Python directly to pass tensors yourself."
        )
    spec = importlib.util.spec_from_file_location("_replay_golden", str(golden_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.generate_inputs({"name": "Default"})
    return list(result.values())


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pypto.runtime.debug.replay",
        description=(
            "Re-execute an existing build_output/<jit_dir>/ directory with "
            "DFX flags. Loads inputs via the directory's golden.py."
        ),
    )
    parser.add_argument("work_dir", type=Path, help="Path to build_output/<jit_dir>/")
    parser.add_argument("--platform", default="a2a3sim", help="Target execution platform")
    parser.add_argument("--device-id", type=int, default=0, help="Hardware device index")
    parser.add_argument("--pmu", type=int, default=0, metavar="LEVEL", help="PMU level")
    parser.add_argument("--swimlane", action="store_true", help="Enable L2 swimlane capture")
    parser.add_argument("--dump-tensor", action="store_true", help="Enable per-task tensor dump")
    parser.add_argument("--dep-gen", action="store_true", help="Enable dep_gen profiling")
    parser.add_argument(
        "--no-recompile",
        action="store_true",
        help="Reuse cached binaries (faster, but ignores cpp edits)",
    )
    args = parser.parse_args(argv)

    config = RunConfig(
        platform=args.platform,
        device_id=args.device_id,
        enable_pmu=args.pmu,
        enable_l2_swimlane=args.swimlane,
        enable_dump_tensor=args.dump_tensor,
        enable_dep_gen=args.dep_gen,
    )
    tensors = _load_inputs_from_golden(args.work_dir)
    replay(args.work_dir, *tensors, config=config, recompile=not args.no_recompile)
    print(f"Replay finished. DFX artefacts (if any) under {args.work_dir / 'dfx_outputs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
