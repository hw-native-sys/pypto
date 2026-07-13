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
with DFX (PMU / swimlane / args_dump / dep_gen / scope_stats) enabled" workflow.

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
        --pmu 2 --swimlane --log-level debug

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
from collections.abc import Callable
from ctypes import _SimpleCData
from pathlib import Path
from types import ModuleType

import torch

from pypto.runtime.debug.pto_rebuild import rebuild_kernel_cpp_from_pto
from pypto.runtime.device_tensor import DeviceTensor
from pypto.runtime.runner import RunConfig, _DfxOpts, execute_compiled

__all__ = ["replay", "invalidate_binary_cache"]


def _invalidate_one_dir(base: Path) -> int:
    """Remove cached binaries directly under *base*; return the count removed.

    Deletes ``<base>/cache/*.bin`` (the pre-build cache written by
    ``prebuild_binaries``) and the sibling ``.so`` / ``.o`` files under
    ``<base>/kernels`` and ``<base>/orchestration``. CPP sources are untouched.
    """
    removed = 0
    cache_dir = base / "cache"
    if cache_dir.is_dir():
        for f in cache_dir.glob("*.bin"):
            f.unlink()
            removed += 1
    for sub in ("kernels", "orchestration"):
        root = base / sub
        if not root.is_dir():
            continue
        for ext in ("*.so", "*.o"):
            for f in root.rglob(ext):
                f.unlink()
                removed += 1
    return removed


def invalidate_binary_cache(work_dir: Path | str) -> None:
    """Remove cached kernel/orchestration binaries under *work_dir*.

    Both ``cache/*.bin`` (the pre-build cache written by ``prebuild_binaries``)
    and the sibling ``.so`` / ``.o`` files next to each cpp are deleted. CPP
    sources are untouched, so the next ``compile_and_assemble`` rebuilds from
    source and picks up hand-edits.

    Handles both layouts: a single-chip / L2 build keeps ``cache/`` +
    ``kernels/`` + ``orchestration/`` at the root; an L3 distributed build also
    has one complete sub-build per rank under ``next_levels/{rank}/``, each with
    its own cache + binaries — those are invalidated too.

    Safe to call when nothing is cached — silently no-ops on missing
    files / directories.

    Prints the number of files removed so users running ``debug/run.py``
    can see the ``cpp -> .o`` rebuild path was taken.
    """
    work_dir = Path(work_dir)
    removed = _invalidate_one_dir(work_dir)
    next_levels = work_dir / "next_levels"
    if next_levels.is_dir():
        for rank_dir in sorted(next_levels.iterdir()):
            if rank_dir.is_dir():
                removed += _invalidate_one_dir(rank_dir)
    if removed:
        print(f"[cpp->.so] invalidated {removed} cached binary file(s); cpp will rebuild")
    else:
        print("[cpp->.so] no cached binaries to invalidate")


def replay(
    work_dir: Path | str,
    *tensors: torch.Tensor | DeviceTensor | _SimpleCData,
    config: RunConfig | None = None,
    recompile: bool = True,
    rebuild_from_pto: bool = True,
    validate: bool = False,
) -> None:
    """Re-execute an existing ``build_output/<jit_dir>/`` with new tensors.

    Args:
        work_dir: A ``build_output/<jit_dir>/`` produced by a prior
            ``ir.compile`` / ``run`` call. For a single-chip / L2 build this
            contains ``kernel_config.py``, ``orchestration/`` and ``kernels/``.
            An L3 distributed build (``orchestration/host_orch.py`` +
            ``next_levels/{rank}/`` + ``distributed_meta.json``) is detected
            automatically and dispatched via ``execute_distributed_compiled``.
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
        rebuild_from_pto: When ``True`` (default), before cache
            invalidation, scan ``ptoas/*.pto`` and rerun ``ptoas`` for any
            file newer than its sibling ``ptoas/<unit>.cpp``; the new body
            is spliced into the matching ``kernels/<core>/<func>.cpp``. Set
            to ``False`` to ignore ``.pto`` edits entirely. Independent of
            ``recompile``: the cpp itself is still picked up by the cpp →
            ``.so`` path even when ``rebuild_from_pto`` is off.
        validate: When ``True``, after execution compare each output tensor
            (identified via ``golden.py::__outputs__``) against the value
            produced by ``golden.py::compute_golden`` using ``torch.allclose``
            with the tolerances declared in ``golden.py``. The number of
            ``*tensors`` must match the length of
            ``golden.py::generate_inputs`` so positional names line up.
            Raises ``AssertionError`` on mismatch, ``FileNotFoundError`` if
            the directory has no ``golden.py``.

    Raises:
        FileNotFoundError: If *work_dir* contains neither ``kernel_config.py``
            (single-chip) nor ``orchestration/host_orch.py`` (L3), or
            ``golden.py`` is missing when ``validate=True``.
        ValueError: If ``validate=True`` and the number of *tensors* does not
            match the orchestration parameter count from ``golden.py``.
        AssertionError: If ``validate=True`` and any output tensor disagrees
            with the golden reference within the declared tolerances.
    """
    config = config or RunConfig()
    work_dir = Path(work_dir)
    # L3 distributed builds have no top-level ``kernel_config.py`` (per-rank
    # configs live under ``next_levels/{rank}/``); they are identified by the
    # generated HOST orchestrator and dispatched via ``execute_distributed_compiled``.
    is_l3 = (
        not (work_dir / "kernel_config.py").exists()
        and (work_dir / "orchestration" / "host_orch.py").exists()
    )
    if not is_l3 and not (work_dir / "kernel_config.py").exists():
        raise FileNotFoundError(
            f"replay(): {work_dir} is not a build_output directory "
            f"(missing kernel_config.py and orchestration/host_orch.py)"
        )

    named_tensors: list[tuple[str, torch.Tensor]] | None = None
    golden_module = None
    if validate:
        golden_module = _load_golden_module(work_dir)
        named_defaults = list(golden_module.generate_inputs({"name": "Default"}))
        if len(tensors) != len(named_defaults):
            raise ValueError(
                f"replay(validate=True): expected {len(named_defaults)} tensors "
                f"(orchestration parameter count from {work_dir}/golden.py), "
                f"got {len(tensors)}"
            )
        named_tensors = []
        for (name, _), t in zip(named_defaults, tensors, strict=True):
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"replay(validate=True): parameter {name!r} must be a torch.Tensor, "
                    f"got {type(t).__name__}"
                )
            named_tensors.append((name, t))

    if rebuild_from_pto:
        rebuild_kernel_cpp_from_pto(work_dir)
    else:
        print("[pto->cpp] skipped (rebuild_from_pto=False)")

    if recompile:
        invalidate_binary_cache(work_dir)
    else:
        print("[cpp->.so] reusing cached binaries (recompile=False)")

    print("[execute] running on device...")
    if is_l3:
        # L3: reconstruct the distributed program from the build dir and dispatch
        # via simpler Worker(level=3). DFX flags on ``config`` are not yet plumbed
        # through the distributed dispatch path (tracked separately).
        from pypto.runtime.distributed_runner import (  # noqa: PLC0415
            execute_distributed_compiled,
        )

        execute_distributed_compiled(work_dir, list(tensors), config=config, platform=config.platform)
    else:
        execute_compiled(
            work_dir,
            list(tensors),
            platform=config.platform,
            device_id=config.device_id,
            pto_isa_commit=config.pto_isa_commit,
            dfx=_DfxOpts.from_run_config(config),
        )

    if named_tensors is not None:
        assert golden_module is not None
        _validate_against_golden_module(golden_module, named_tensors)


def _load_golden_module(work_dir: Path) -> ModuleType:
    """Import and return ``<work_dir>/golden.py`` as a module object.

    Raises ``FileNotFoundError`` when ``golden.py`` is absent. The module is
    loaded under a fixed name (``_replay_golden``); callers should reuse the
    returned object instead of re-importing the file.
    """
    golden_path = work_dir / "golden.py"
    if not golden_path.exists():
        raise FileNotFoundError(f"{golden_path} not found; cannot derive named inputs / outputs.")
    spec = importlib.util.spec_from_file_location("_replay_golden", str(golden_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_named_inputs_from_golden(
    work_dir: Path,
) -> list[tuple[str, torch.Tensor]]:
    """Load ``[(name, value), ...]`` from ``<work_dir>/golden.py``.

    The list order matches the orchestration entry parameter order. Both
    inputs and outputs are present — outputs are zero-initialised tensors
    that orchestration writes back into in place.
    """
    module = _load_golden_module(work_dir)
    return list(module.generate_inputs({"name": "Default"}))


def _validate_against_golden_module(
    module: ModuleType,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    """Compute expected outputs via a pre-loaded golden module and compare.

    Actual outputs (already written in place by orchestration) are matched
    by name against the ``__outputs__`` list. Expected outputs are produced
    by cloning the actual output tensors (so dtype/shape match) and letting
    ``compute_golden`` populate them from the user inputs. Comparison uses
    :func:`torch.testing.assert_close` with the tolerances declared on the
    ``golden.py`` module (``RTOL`` / ``ATOL``, defaulting to ``1e-5``).
    """
    output_names = set(getattr(module, "__outputs__", []))
    if not output_names:
        return  # nothing declared as output — skip silently

    inputs = {n: v for n, v in named_tensors if n not in output_names}
    actual_outputs = {n: v for n, v in named_tensors if n in output_names}
    expected = {n: v.clone() for n, v in actual_outputs.items()}
    module.compute_golden({**inputs, **expected}, {})

    rtol = getattr(module, "RTOL", 1e-5)
    atol = getattr(module, "ATOL", 1e-5)
    for name, actual in actual_outputs.items():
        try:
            torch.testing.assert_close(actual.cpu(), expected[name].cpu(), rtol=rtol, atol=atol)
        except AssertionError as e:
            raise AssertionError(
                f"Output '{name}' does not match golden (rtol={rtol}, atol={atol}):\n{e}"
            ) from e


def _main(
    argv: list[str] | None = None,
    *,
    inline_inputs: Callable[[], list] | None = None,
    user_compare: Callable[..., None] | None = None,
    default_platform: str = "a2a3sim",
) -> int:
    """Shared CLI entry for both ``python -m pypto.runtime.debug.replay`` and
    the auto-generated ``debug/run.py`` shim.

    Args:
        argv: Argument vector (positional ``work_dir`` first).
        inline_inputs: Auto-runner hook. When provided, signals that the
            caller is the JIT-emitted ``debug/run.py`` and supplies tensors
            for the no-golden / ``--no-validate`` paths. Switches the
            ``--validate`` default from False (standalone, opt-in) to True
            (auto-runner, opt-out).
        user_compare: Auto-runner hook called with ``*tensors`` after a
            non-validating replay finishes — the JIT path's hand-edited
            comparison stub.
        default_platform: Default for ``--platform``. The auto-runner bakes
            the compile-time platform here so users get the right target
            without re-typing it.
    """
    parser = argparse.ArgumentParser(
        prog="python -m pypto.runtime.debug.replay",
        description=(
            "Re-execute an existing build_output/<jit_dir>/ directory with "
            "DFX flags. Loads inputs via the directory's golden.py."
        ),
    )
    parser.add_argument("work_dir", type=Path, help="Path to build_output/<jit_dir>/")
    parser.add_argument("--platform", default=default_platform, help="Target execution platform")
    parser.add_argument("--device-id", type=int, default=0, help="Hardware device index")
    parser.add_argument("--pmu", type=int, default=0, metavar="LEVEL", help="PMU level")
    parser.add_argument("--swimlane", action="store_true", help="Enable L2 swimlane capture")
    parser.add_argument(
        "--dump-args",
        nargs="?",
        type=int,
        const=1,
        default=0,
        metavar="LEVEL",
        help="Per-task argument dump level: bare flag = 1 (partial, dump_tag-marked), "
        "2 = full (every task), absent = 0 (off)",
    )
    parser.add_argument("--dep-gen", action="store_true", help="Enable dep_gen profiling")
    parser.add_argument("--scope-stats", action="store_true", help="Enable scope_stats profiling")
    parser.add_argument(
        "--no-recompile",
        action="store_true",
        help="Reuse cached binaries (faster, but ignores cpp edits)",
    )
    parser.add_argument(
        "--no-rebuild-from-pto",
        action="store_true",
        help="Skip .pto -> cpp rebuild even when ptoas/*.pto is newer than cpp",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        metavar="LEVEL",
        help=(
            "PyPTO runtime log level (debug, v0..v9, info, warn, error, null). "
            "Equivalent to setting PYPTO_RUNTIME_LOG=<level> in the environment. "
            "Pass --log-sync-pypto to also push the band to PyPTO's C++ logger."
        ),
    )
    parser.add_argument(
        "--log-sync-pypto",
        action="store_true",
        help="When used with --log-level, mirror the level onto PyPTO's C++ logger.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=inline_inputs is not None,
        help=(
            "Compare outputs against golden.py::compute_golden using "
            "torch.allclose. Defaults to off for the standalone CLI "
            "(opt-in) and on for the auto-emitted debug/run.py "
            "(opt-out with --no-validate)."
        ),
    )
    args = parser.parse_args(argv)

    if args.log_level is not None:
        from pypto.runtime.log_config import configure_log  # noqa: PLC0415 — keep import lazy

        configure_log(args.log_level, sync_pypto=args.log_sync_pypto)

    golden_exists = (args.work_dir / "golden.py").exists()
    if inline_inputs is not None and not (args.validate and golden_exists):
        tensors = list(inline_inputs())
        do_validate = False
    else:
        tensors = [v for _, v in _load_named_inputs_from_golden(args.work_dir)]
        do_validate = args.validate

    config = RunConfig(
        platform=args.platform,
        device_id=args.device_id,
        enable_pmu=args.pmu,
        enable_l2_swimlane=args.swimlane,
        enable_dump_args=args.dump_args,
        enable_dep_gen=args.dep_gen,
        enable_scope_stats=args.scope_stats,
    )
    replay(
        args.work_dir,
        *tensors,
        config=config,
        recompile=not args.no_recompile,
        rebuild_from_pto=not args.no_rebuild_from_pto,
        validate=do_validate,
    )
    print(f"Replay finished. DFX artefacts (if any) under {args.work_dir / 'dfx_outputs'}")
    if do_validate:
        print("Golden validation: PASSED")
    elif user_compare is not None:
        user_compare(*tensors)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
