# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute a pre-compiled ``work_dir`` artifact on one device.

Thin CLI that re-binds an already-compiled build directory (``.o``/``.so``
written next to each kernel by a prior ``compile_and_assemble``) and runs its
``golden.py`` on a single device, validating against the pre-computed golden.

It is the device-side half of the "compile on CPU, borrow a card per case via
``task-submit``" CI flow: the test harness compiles every case on a card-free
CPU pool, then for each case spawns::

    task-submit --device auto --run \\
      'python -m pypto.runtime.execute_artifact --work-dir <wd> \\
         --platform <p> --device-id $TASK_DEVICE ...'

Because the binaries are already on disk, ``compile_and_assemble`` hits the
``.o``/``.so`` cache and rebuilds the ``ChipCallable`` without recompiling or
touching a card — the only card window is the device run + ``allclose``.

The same CLI doubles as a manual reproduction entry point for a failed case::

    python -m pypto.runtime.execute_artifact --work-dir build_output/<case> \\
        --platform a2a3 --device-id 0

Exit contract (the harness relies on it; ``task-submit`` propagates it verbatim):

- Success: prints ``PYPTO_EXEC_RESULT=PASS device=<N>`` to stdout, exits ``0``.
- Failure: prints the traceback to stderr, then ``PYPTO_EXEC_RESULT=FAIL``,
  exits ``1``.
"""

import argparse
import json
import traceback
from pathlib import Path

from pypto.runtime.runner import _DfxOpts, _execute_on_device

__all__ = ["execute_artifact_dir", "execute_batch_manifest", "main"]

# Sentinel line parsed by the harness (``_execute_via_task_submit``) to tell a
# genuine case failure apart from an infra kill (``task-submit`` --max-time /
# --timeout, missing binary). Keep in sync with the harness-side parser.
_RESULT_PREFIX = "PYPTO_EXEC_RESULT"


def execute_artifact_dir(
    work_dir: Path,
    platform: str,
    device_id: int,
    *,
    pto_isa_commit: str | None = None,
    dfx: _DfxOpts = _DfxOpts(),
    validate: bool = True,
) -> None:
    """Rebuild the compiled artifact in *work_dir* and run it on *device_id*.

    ``compile_and_assemble`` reuses the cached ``.o``/``.so`` next to each
    kernel, so this neither recompiles nor needs a card until the device run.
    The actual device outputs are always persisted under ``data/actual/`` so a
    caller can validate them separately with the test's real tolerance.

    Args:
        work_dir: A build directory produced by ``compile_program`` +
            ``compile_and_assemble`` (contains ``kernel_config.py``,
            ``kernels/``, ``orchestration/``, ``golden.py``, ``data/``).
        platform: Target execution platform (e.g. ``"a2a3"``).
        device_id: Hardware device index to run on.
        pto_isa_commit: If set, pin the pto-isa clone to this commit.
        dfx: Runtime DFX toggles; artefacts land under ``work_dir/dfx_outputs``.
        validate: When ``True`` (manual repro default), compare outputs against
            the golden in-process using ``golden.py``'s tolerances. When
            ``False`` (the harness's split path), only run the device and
            persist outputs — the harness validates them later with the
            per-test tolerance, so this run is tolerance-independent.

    Raises:
        Exception: Any compile/load/device/validation error is propagated to
            the caller (``main`` turns it into exit code ``1``).
    """
    from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

    chip_callable, runtime_name, _ = compile_and_assemble(work_dir, platform, pto_isa_commit=pto_isa_commit)
    _execute_on_device(
        work_dir,
        work_dir / "golden.py",
        chip_callable,
        runtime_name,
        platform,
        device_id,
        dfx=dfx,
        validate=validate,
        actual_out_dir=work_dir / "data" / "actual",
    )


def execute_batch_manifest(
    manifest_path: Path,
    device_id: int,
    *,
    pto_isa_commit: str | None = None,
    dfx: _DfxOpts = _DfxOpts(),
    validate: bool = False,
) -> bool:
    """Run a batch of artifacts in ONE process, reusing the device session.

    *manifest_path* is a JSON list of ``{"work_dir": str, "platform": str}``.
    The whole batch runs inside a single ``ChipWorker`` context so the torch /
    pypto import AND the NPU device init are paid once for the batch (not once
    per artifact) — the fix for the per-artifact cold-start cost.  Artifacts
    that share the batch's ``(platform, runtime)`` reuse the worker; a differing
    one falls back to a fresh one-shot worker inside ``_execute_on_device``.

    Each artifact runs under its own ``try`` so one failure doesn't abort the
    rest, and emits a per-artifact marker the harness parses::

        PYPTO_EXEC_RESULT=PASS work_dir=<wd> device=<N>
        PYPTO_EXEC_RESULT=FAIL work_dir=<wd>

    Returns ``True`` iff every artifact in the batch succeeded.  (A hard process
    crash leaves later artifacts without a marker; the harness treats a missing
    marker as a failure.)
    """
    from pypto.runtime import ChipWorker, RunConfig  # noqa: PLC0415
    from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not entries:
        return True

    def _run_one(work_dir: Path, platform: str) -> bool:
        chip_callable, runtime_name, _ = compile_and_assemble(
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
            validate=validate,
            actual_out_dir=work_dir / "data" / "actual",
        )
        return True

    # Open one ChipWorker for the batch, bound to the first artifact's runtime
    # (uniform within a backend group in practice). compile_and_assemble is a
    # cache hit, so peeking the first runtime is cheap.
    first = entries[0]
    _, first_runtime, _ = compile_and_assemble(
        Path(first["work_dir"]), first["platform"], pto_isa_commit=pto_isa_commit
    )
    all_ok = True
    with ChipWorker(
        config=RunConfig(platform=first["platform"], device_id=device_id),
        runtime=first_runtime,
    ):
        for entry in entries:
            work_dir = Path(entry["work_dir"])
            try:
                _run_one(work_dir, entry["platform"])
                print(f"{_RESULT_PREFIX}=PASS work_dir={work_dir} device={device_id}", flush=True)
            except Exception:
                # Print the traceback to stdout (not stderr) so it sits right
                # before this artifact's FAIL marker — the harness attributes the
                # preceding stdout lines to this case for an inline error report.
                print(traceback.format_exc(), flush=True)
                print(f"{_RESULT_PREFIX}=FAIL work_dir={work_dir}", flush=True)
                all_ok = False
    return all_ok


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pypto.runtime.execute_artifact",
        description=(
            "Execute a pre-compiled build directory on one device, validating "
            "against its golden.py. Reuses cached .o/.so (no recompile)."
        ),
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="Path to the compiled build directory (single)"
    )
    parser.add_argument("--platform", default=None, help="Target execution platform (single mode)")
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        default=None,
        help="Path to a JSON list of {work_dir, platform} — run all in ONE process (one device "
        "init for the whole batch). Mutually exclusive with --work-dir.",
    )
    parser.add_argument("--device-id", type=int, required=True, help="Hardware device index")
    parser.add_argument("--pto-isa-commit", default=None, help="Pin pto-isa to this commit")
    # DFX toggles — names mirror tests/st/conftest.py so the harness round-trip
    # (_dfx_to_cli) is symmetric.
    parser.add_argument("--enable-l2-swimlane", action="store_true", help="Capture L2 swimlane records")
    parser.add_argument(
        "--dump-tensor",
        type=int,
        default=0,
        metavar="LEVEL",
        help="Per-task tensor dump level (0=off, 1=partial, 2=full)",
    )
    parser.add_argument(
        "--enable-pmu", type=int, default=0, metavar="EVENT", help="AICore PMU event type (0=off)"
    )
    parser.add_argument("--enable-dep-gen", action="store_true", help="Capture PTO2 dependency edges")
    parser.add_argument("--enable-scope-stats", action="store_true", help="Capture per-scope ring-fill stats")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Only run the device and persist outputs under data/actual/ — skip the in-process "
        "allclose. The harness uses this so the device run is tolerance-independent and can be "
        "submitted eagerly; it validates the persisted outputs later with the per-test tolerance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: run one artifact, print the result marker, return exit code.

    Returns ``0`` on success (after printing ``PYPTO_EXEC_RESULT=PASS
    device=<N>``) and ``1`` on any failure (after printing the traceback to
    stderr and ``PYPTO_EXEC_RESULT=FAIL``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    dfx = _DfxOpts(
        enable_l2_swimlane=args.enable_l2_swimlane,
        enable_dump_tensor=args.dump_tensor,
        enable_pmu=args.enable_pmu,
        enable_dep_gen=args.enable_dep_gen,
        enable_scope_stats=args.enable_scope_stats,
    )

    # Batch mode: per-artifact markers are emitted by execute_batch_manifest;
    # the harness parses those, so don't print a single-run marker here.
    if args.batch_manifest is not None:
        try:
            all_ok = execute_batch_manifest(
                args.batch_manifest,
                args.device_id,
                pto_isa_commit=args.pto_isa_commit,
                dfx=dfx,
                validate=not args.no_validate,
            )
        except Exception:
            # A batch-level failure (e.g. opening the ChipWorker / device init):
            # no per-artifact markers — the harness treats every artifact in the
            # batch as failed.
            traceback.print_exc()
            print(f"{_RESULT_PREFIX}=FAIL", flush=True)
            return 1
        return 0 if all_ok else 1

    if args.work_dir is None or args.platform is None:
        parser.error("--work-dir and --platform are required unless --batch-manifest is given")
    try:
        execute_artifact_dir(
            args.work_dir,
            args.platform,
            args.device_id,
            pto_isa_commit=args.pto_isa_commit,
            dfx=dfx,
            validate=not args.no_validate,
        )
    except Exception:
        traceback.print_exc()
        print(f"{_RESULT_PREFIX}=FAIL", flush=True)
        return 1
    print(f"{_RESULT_PREFIX}=PASS device={args.device_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
