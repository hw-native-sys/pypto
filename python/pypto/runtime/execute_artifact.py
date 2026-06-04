# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Rebuild a compiled artifact directory from its on-disk cache and run it on one device.

Thin CLI used by the ST harness ``--execute-via-task-submit`` execution mode. The
compile pool writes each test case's ``.o``/``.so`` and ``golden.py`` into a
shared ``work_dir``; this entry point — launched as a subprocess on a borrowed
NPU via ``task-submit`` — reloads that directory, rebuilds the ``ChipCallable``
from the cached binaries (no device-side recompile), runs ``golden.py`` on the
given device, and validates against the pre-computed golden.

It reuses :func:`pypto.runtime.device_runner.compile_and_assemble` (cache hit →
no recompile) and :func:`pypto.runtime.runner._execute_on_device` (load inputs,
run, ``allclose``) verbatim, so the device-side path is identical to the harness's
in-process path.

The last stdout line is a machine-parseable marker::

    __PYPTO_EXEC__ result=PASS device=3
    __PYPTO_EXEC__ result=FAIL device=3

The parent harness trusts this marker as the source of truth for pass/fail and
for the real borrowed device id (only the subprocess knows ``$TASK_DEVICE``),
independent of whether the ``task-submit`` wrapper passes the inner exit code
through.

CLI::

    python -m pypto.runtime.execute_artifact \\
        --work-dir build_output/st_artifacts/<case>/ --platform a2a3 --device-id 3
"""

import argparse
import traceback
from pathlib import Path

from pypto.runtime.runner import _DfxOpts, _execute_on_device

# Result marker emitted on the final stdout line. Shared with the harness parser
# (``tests/st/harness/core/test_runner.py``) via import so the two never drift.
_EXEC_MARKER = "__PYPTO_EXEC__"


def execute_artifact_dir(
    work_dir: Path,
    platform: str,
    device_id: int,
    *,
    pto_isa_commit: str | None = None,
    dfx: _DfxOpts = _DfxOpts(),
) -> None:
    """Rebuild *work_dir* from cache and execute + validate it on *device_id*.

    Args:
        work_dir: Compiled artifact directory (``kernels/``, ``orchestration/``,
            ``kernel_config.py``, ``golden.py``, ``data/``) produced by the
            compile pool on a shared filesystem.
        platform: Target execution platform (e.g. ``a2a3``).
        device_id: Hardware device index to run on.
        pto_isa_commit: Pin the pto-isa clone to this commit, if set.
        dfx: Runtime DFX toggles.
    """
    # Imported lazily — device_runner eagerly pulls in the optional ``simpler``
    # runtime package, which keeps this module importable (e.g. in unit tests)
    # without it.  Mirrors runner._execute_on_device / test_runner.
    from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

    chip_callable, runtime_name, _ = compile_and_assemble(  # cache hit → no recompile, no card
        work_dir, platform, pto_isa_commit=pto_isa_commit
    )
    _execute_on_device(
        work_dir,
        work_dir / "golden.py",
        chip_callable,
        runtime_name,
        platform,
        device_id,
        dfx=dfx,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. DFX flag names match the conftest options and the
    :class:`_DfxOpts` field names so ``_dfx_to_cli`` round-trips losslessly."""
    parser = argparse.ArgumentParser(
        prog="python -m pypto.runtime.execute_artifact",
        description=(
            "Rebuild a compiled build_output/<case>/ directory from its on-disk "
            ".o/.so cache and execute it on one device, validating against golden.py."
        ),
    )
    parser.add_argument("--work-dir", type=Path, required=True, help="Compiled artifact directory")
    parser.add_argument("--platform", required=True, help="Target execution platform")
    parser.add_argument("--device-id", type=int, required=True, help="Hardware device index")
    parser.add_argument(
        "--pto-isa-commit", default=None, help="Pin the pto-isa clone to this commit (hash or tag)"
    )
    # DFX toggles — two carry an integer payload (value, not store_true).
    parser.add_argument("--enable-l2-swimlane", action="store_true", help="Enable L2 swimlane capture")
    parser.add_argument(
        "--dump-tensor", type=int, default=0, metavar="LEVEL", help="Per-task tensor dump level (0/1/2)"
    )
    parser.add_argument("--enable-pmu", type=int, default=0, metavar="LEVEL", help="PMU event type (0=off)")
    parser.add_argument("--enable-dep-gen", action="store_true", help="Enable dep_gen profiling")
    parser.add_argument("--enable-scope-stats", action="store_true", help="Enable scope_stats profiling")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the artifact, and emit the result marker.

    Returns ``0`` when execution + validation succeed, ``1`` on any exception
    (the traceback is printed to stderr). The final stdout line is always the
    ``__PYPTO_EXEC__`` marker carrying the result and device id.
    """
    args = _build_parser().parse_args(argv)
    dfx = _DfxOpts(
        enable_l2_swimlane=args.enable_l2_swimlane,
        enable_dump_tensor=args.dump_tensor,
        enable_pmu=args.enable_pmu,
        enable_dep_gen=args.enable_dep_gen,
        enable_scope_stats=args.enable_scope_stats,
    )
    try:
        execute_artifact_dir(
            args.work_dir,
            args.platform,
            args.device_id,
            pto_isa_commit=args.pto_isa_commit,
            dfx=dfx,
        )
    except Exception:
        traceback.print_exc()
        print(f"{_EXEC_MARKER} result=FAIL device={args.device_id}", flush=True)
        return 1
    print(f"{_EXEC_MARKER} result=PASS device={args.device_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
