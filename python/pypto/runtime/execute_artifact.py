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
- Device / validation failure: prints the traceback to stderr, then
  ``PYPTO_EXEC_RESULT=FAIL``, exits ``1``.
- Setup / reconstruction failure (stale/missing cache, bad ``work_dir``): prints
  the traceback to stderr, then ``PYPTO_EXEC_RESULT=INFRA``, exits ``1`` — kept
  distinct from ``FAIL`` so an infra miss isn't misread as a numerical regression.
"""

import argparse
import json
import traceback
import warnings
from pathlib import Path
from typing import Any

from pypto.runtime.runner import (
    _SWIMLANE_CLI_HELP,
    _SWIMLANE_FULL_LEVEL,
    _SWIMLANE_MAX_LEVEL,
    _DfxOpts,
    _execute_on_device,
)

__all__ = ["ArtifactSetupError", "execute_artifact_dir", "execute_batch_manifest", "main"]

# Sentinel line parsed by the harness (``_execute_via_task_submit``) to tell a
# genuine case failure apart from an infra kill (``task-submit`` --max-time /
# --timeout, missing binary). Keep in sync with the harness-side parser.
_RESULT_PREFIX = "PYPTO_EXEC_RESULT"


class ArtifactSetupError(Exception):
    """A pre-run artifact reconstruction / setup failure.

    Raised when ``compile_and_assemble`` cannot rebuild the cached artifact
    (missing / stale ``.o``/``.so``, bad ``work_dir``, corrupt inputs) — i.e.
    *before* any device execution. The CLI reports this as an infra problem
    (``PYPTO_EXEC_RESULT=INFRA``) rather than a device test failure
    (``=FAIL``), so a cache/setup miss isn't misread as a numerical regression.
    """


def execute_artifact_dir(
    work_dir: Path,
    platform: str,
    device_id: int,
    *,
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
        dfx: Runtime DFX toggles; artefacts land under ``work_dir/dfx_outputs``.
        validate: When ``True`` (manual repro default), compare outputs against
            the golden in-process using ``golden.py``'s tolerances. When
            ``False`` (the harness's split path), only run the device and
            persist outputs — the harness validates them later with the
            per-test tolerance, so this run is tolerance-independent.

    Raises:
        ArtifactSetupError: ``compile_and_assemble`` could not rebuild the
            cached artifact (infra, not a test failure). ``main`` maps it to
            ``PYPTO_EXEC_RESULT=INFRA``.
        Exception: Any device / validation error is propagated to the caller
            (``main`` turns it into ``PYPTO_EXEC_RESULT=FAIL`` + exit ``1``).
    """
    chip_callable, runtime_name, enable_sdma = _reconstruct_artifact(work_dir, platform)
    _run_on_device(
        work_dir,
        platform,
        device_id,
        chip_callable,
        runtime_name,
        enable_sdma,
        dfx=dfx,
        validate=validate,
    )


def _reconstruct_artifact(work_dir: Path, platform: str) -> tuple[Any, str, bool]:
    """Rebuild the cached artifact, reclassifying any failure as infra.

    ``compile_and_assemble`` reuses the cached ``.o``/``.so`` next to each kernel
    (a cache hit — no recompile, no card). A failure here is a reconstruction /
    setup problem, so it is wrapped as :class:`ArtifactSetupError` (the CLI reports
    ``PYPTO_EXEC_RESULT=INFRA``) rather than a device test failure.
    """
    from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

    try:
        chip_callable, runtime_name, runtime_config = compile_and_assemble(work_dir, platform)
    except Exception as exc:  # noqa: BLE001 — reclassified as infra, re-raised below
        raise ArtifactSetupError(f"artifact reconstruction failed for {work_dir}: {exc}") from exc
    enable_sdma = bool(runtime_config.get("enable_sdma", False))
    return chip_callable, runtime_name, enable_sdma


def _run_on_device(
    work_dir: Path,
    platform: str,
    device_id: int,
    chip_callable: Any,
    runtime_name: str,
    enable_sdma: bool,
    *,
    dfx: _DfxOpts,
    validate: bool,
) -> None:
    """Device-run half shared by the single (:func:`execute_artifact_dir`) and
    batch (:func:`execute_batch_manifest`) paths.

    Always persists the actual device outputs under ``data/actual/`` (tolerance
    independent) so a caller can validate them separately; runs the in-process
    ``allclose`` only when *validate* is set.
    """
    _execute_on_device(
        work_dir,
        work_dir / "golden.py",
        chip_callable,
        runtime_name,
        platform,
        device_id,
        enable_sdma=enable_sdma,
        dfx=dfx,
        validate=validate,
        actual_out_dir=work_dir / "data" / "actual",
    )


def execute_batch_manifest(
    manifest_path: Path,
    device_id: int,
    *,
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
    Before opening the shared worker, the batch reconstructs every artifact so
    it can provision SDMA when any usable artifact on the shared worker's
    binding requires it.  Ordinary artifacts can run on that SDMA-enabled
    worker, while an all-ordinary batch keeps SDMA disabled.

    Each artifact runs under its own ``try`` so one failure doesn't abort the
    rest, and emits a per-artifact marker the harness parses::

        PYPTO_EXEC_RESULT=PASS  work_dir=<wd> device=<N>   # ran + (deferred) validate
        PYPTO_EXEC_RESULT=FAIL  work_dir=<wd>              # device / validation failure
        PYPTO_EXEC_RESULT=INFRA work_dir=<wd>              # reconstruction/setup failure

    Every entry — including one whose artifact cannot be rebound — gets its own
    marker, so a single bad artifact never sinks the whole batch (a marker-less
    batch-level crash would make the harness fail *all* entries).

    Returns ``True`` iff every artifact in the batch succeeded.  (A hard process
    crash leaves later artifacts without a marker; the harness treats a missing
    marker as a failure.)
    """
    from pypto.runtime import ChipWorker, RunConfig  # noqa: PLC0415

    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not entries:
        return True

    all_ok = True
    # The worker caches registrations by ``id(chip_callable)``; if a callable is
    # GC'd mid-batch its id can be recycled by the next one and the worker would
    # dispatch the *stale* registration. Hold a strong ref to every callable for
    # the batch lifetime to keep ids distinct.
    live_callables: list[Any] = []

    def _rebind(work_dir: Path, platform: str) -> tuple[Any, str, bool]:
        # _reconstruct_artifact already wraps a compile failure as ArtifactSetupError.
        chip_callable, runtime_name, enable_sdma = _reconstruct_artifact(work_dir, platform)
        live_callables.append(chip_callable)
        return chip_callable, runtime_name, enable_sdma

    def _run_one(work_dir: Path, platform: str) -> None:
        chip_callable, runtime_name, enable_sdma = _rebind(work_dir, platform)
        _run_on_device(
            work_dir,
            platform,
            device_id,
            chip_callable,
            runtime_name,
            enable_sdma,
            dfx=dfx,
            validate=validate,
        )

    def _run_and_mark(entry: dict) -> None:
        nonlocal all_ok
        work_dir = Path(entry["work_dir"])
        try:
            _run_one(work_dir, entry["platform"])
            print(f"{_RESULT_PREFIX}=PASS work_dir={work_dir} device={device_id}", flush=True)
        except ArtifactSetupError:
            # Rebuild failed before the device run — infra, not a test failure.
            print(traceback.format_exc(), flush=True)
            print(f"{_RESULT_PREFIX}=INFRA work_dir={work_dir}", flush=True)
            all_ok = False
        except Exception:
            # Print the traceback to stdout (not stderr) so it sits right before
            # this artifact's FAIL marker — the harness attributes the preceding
            # stdout lines to this case for an inline error report.
            print(traceback.format_exc(), flush=True)
            print(f"{_RESULT_PREFIX}=FAIL work_dir={work_dir}", flush=True)
            all_ok = False

    # Onboard swimlane runs a dep-gen subprocess that must own the card/SVM state
    # for the duration of dep capture; a batch ChipWorker held open on the same
    # device_id would collide with it. Fall back to per-artifact one-shot
    # execution (each _execute_on_device opens + closes its own worker) so the
    # dep-gen child owns the device first, preserving the two-pass design.
    first_platform = str(entries[0]["platform"])
    if dfx.enable_chip_swimlane and not first_platform.endswith("sim"):
        for entry in entries:
            _run_and_mark(entry)
        return all_ok

    # Reconstruct the whole batch before opening the shared worker so its SDMA
    # capability does not depend on manifest order. Keep setup failures in the
    # prepared list and emit their INFRA markers in manifest order below.
    prepared: list[tuple[dict, tuple[Any, str, bool] | None, str | None]] = []
    for entry in entries:
        try:
            rebound = _rebind(Path(entry["work_dir"]), entry["platform"])
        except Exception:
            prepared.append((entry, None, traceback.format_exc()))
        else:
            prepared.append((entry, rebound, None))

    def _mark_setup_failure(entry: dict, setup_traceback: str | None) -> None:
        nonlocal all_ok
        assert setup_traceback is not None
        print(setup_traceback, flush=True)
        print(f"{_RESULT_PREFIX}=INFRA work_dir={Path(entry['work_dir'])}", flush=True)
        all_ok = False

    first_usable_idx = next(
        (i for i, (_, rebound, _) in enumerate(prepared) if rebound is not None),
        None,
    )
    if first_usable_idx is None:
        for entry, _, setup_traceback in prepared:
            _mark_setup_failure(entry, setup_traceback)
        return False

    # Preserve the historical leading-failure behavior: publish each leading
    # INFRA marker before attempting to open the worker for the remaining batch.
    for entry, _, setup_traceback in prepared[:first_usable_idx]:
        _mark_setup_failure(entry, setup_traceback)

    first_entry, first_rebound, _ = prepared[first_usable_idx]
    assert first_rebound is not None
    _, first_runtime, _ = first_rebound
    shared_platform = str(first_entry["platform"])
    shared_enable_sdma = any(
        rebound[2]
        for entry, rebound, _ in prepared
        if rebound is not None and str(entry["platform"]) == shared_platform and rebound[1] == first_runtime
    )

    with ChipWorker(
        config=RunConfig(platform=shared_platform, device_id=device_id),
        runtime=first_runtime,
        enable_sdma=shared_enable_sdma,
    ):
        for entry, rebound, setup_traceback in prepared[first_usable_idx:]:
            work_dir = Path(entry["work_dir"])
            if rebound is None:
                _mark_setup_failure(entry, setup_traceback)
                continue

            chip_callable, runtime_name, enable_sdma = rebound
            try:
                _run_on_device(
                    work_dir,
                    entry["platform"],
                    device_id,
                    chip_callable,
                    runtime_name,
                    enable_sdma,
                    dfx=dfx,
                    validate=validate,
                )
                print(f"{_RESULT_PREFIX}=PASS work_dir={work_dir} device={device_id}", flush=True)
            except Exception:
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
    # DFX toggles — names mirror tests/st/conftest.py so the harness round-trip
    # (_dfx_to_cli) is symmetric.
    parser.add_argument(
        "--enable-chip-swimlane",
        nargs="?",
        const=_SWIMLANE_FULL_LEVEL,
        default=0,
        type=int,
        choices=range(_SWIMLANE_MAX_LEVEL + 1),
        metavar="LEVEL",
        help=_SWIMLANE_CLI_HELP,
    )
    # Deprecated spelling. Separate dest so ``main`` can tell it was used and
    # warn; ``default=None`` distinguishes "absent" from an explicit ``0``.
    parser.add_argument(
        "--enable-l2-swimlane",
        dest="enable_l2_swimlane_deprecated",
        nargs="?",
        const=_SWIMLANE_FULL_LEVEL,
        default=None,
        type=int,
        choices=range(_SWIMLANE_MAX_LEVEL + 1),
        metavar="LEVEL",
        help="Deprecated alias for --enable-chip-swimlane.",
    )
    parser.add_argument(
        "--dump-args",
        type=int,
        default=0,
        metavar="LEVEL",
        help="Per-task argument dump level (0=off, 1=partial, 2=full)",
    )
    parser.add_argument(
        "--enable-pmu", type=int, default=0, metavar="EVENT", help="AICore PMU event type (0=off)"
    )
    parser.add_argument("--enable-dep-gen", action="store_true", help="Capture simpler dependency edges")
    parser.add_argument("--enable-scope-stats", action="store_true", help="Capture per-scope ring-fill stats")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Only run the device and persist outputs under data/actual/ — skip the in-process "
        "allclose. The harness uses this so the device run is tolerance-independent and can be "
        "submitted eagerly; it validates the persisted outputs later with the per-test tolerance.",
    )
    return parser


def _resolve_swimlane_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Fold the deprecated ``--enable-l2-swimlane`` into the canonical level.

    Returns the requested collection level, warning when the old flag supplied
    it and erroring when both flags disagree.
    """
    deprecated = args.enable_l2_swimlane_deprecated
    if deprecated is None:
        return args.enable_chip_swimlane
    warnings.warn(
        "--enable-l2-swimlane is deprecated; use --enable-chip-swimlane.",
        DeprecationWarning,
        stacklevel=2,
    )
    if args.enable_chip_swimlane and args.enable_chip_swimlane != deprecated:
        parser.error(
            f"--enable-chip-swimlane {args.enable_chip_swimlane} conflicts with the "
            f"deprecated --enable-l2-swimlane {deprecated}; pass only the former."
        )
    return deprecated


def main(argv: list[str] | None = None) -> int:
    """CLI entry: run one artifact, print the result marker, return exit code.

    Returns ``0`` on success (after printing ``PYPTO_EXEC_RESULT=PASS
    device=<N>``) and ``1`` on any failure (after printing the traceback to
    stderr and ``PYPTO_EXEC_RESULT=FAIL``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    dfx = _DfxOpts(
        enable_chip_swimlane=_resolve_swimlane_args(parser, args),
        enable_dump_args=args.dump_args,
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
                dfx=dfx,
                validate=not args.no_validate,
            )
        except Exception:
            # A batch-level failure (e.g. opening the ChipWorker / device init) is
            # infra, not a test failure: no per-artifact markers — the harness
            # treats every artifact in the batch as failed.
            traceback.print_exc()
            print(f"{_RESULT_PREFIX}=INFRA", flush=True)
            return 1
        return 0 if all_ok else 1

    if args.work_dir is None or args.platform is None:
        parser.error("--work-dir and --platform are required unless --batch-manifest is given")
    try:
        execute_artifact_dir(
            args.work_dir,
            args.platform,
            args.device_id,
            dfx=dfx,
            validate=not args.no_validate,
        )
    except ArtifactSetupError:
        # Reconstruction/setup failed before the device run — infra, not a test
        # failure. Emit INFRA (not FAIL) so the harness doesn't count a stale /
        # missing cache or bad --work-dir as a device regression.
        traceback.print_exc()
        print(f"{_RESULT_PREFIX}=INFRA", flush=True)
        return 1
    except Exception:
        traceback.print_exc()
        print(f"{_RESULT_PREFIX}=FAIL", flush=True)
        return 1
    print(f"{_RESULT_PREFIX}=PASS device={args.device_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
