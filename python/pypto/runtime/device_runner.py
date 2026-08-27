# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device compilation, execution, and golden validation pipeline.

This module replaces Simpler's ``CodeRunner`` by providing PyPTO-internal
implementations of:

- :func:`compile_and_assemble`: Compile kernels + orchestration C++ → binaries,
  assemble into ``ChipCallable``, locate runtime binaries.
- :func:`execute_on_device`: Run a ``ChipCallable`` on device via ``ChipWorker``.
- :func:`validate_golden`: Compare actual outputs against golden reference.

These functions keep orchestration in PyPTO while relying on the installed
runtime packages for two integration surfaces:

- ``simpler`` provides the ``_task_interface`` nanobind C++ module.
- ``simpler_setup`` provides the kernel compiler plus packaged runtime sources,
  binaries, and ``pto_isa.pin`` for non-source installs, and owns the only
  PTO-ISA resolver. In a source checkout, those assets come from the
  ``runtime/`` git submodule instead.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from pypto._external_source import kernel_binary_cache_path
from pypto.pypto_core.passes import RuntimeKind, runtime_kind_to_name

from ._binary_cache import (
    BinaryCacheContext,
    binary_context_lock,
    prepare_binary_context,
    record_binary_context,
)
from .elf_parser import elf_build_id_64, extract_text_section
from .kernel_compiler import KernelCompiler
from .pto_isa import ensure_pto_isa_root
from .task_interface import (
    CallConfig,  # pyright: ignore[reportAttributeAccessIssue]
    ChipCallable,  # pyright: ignore[reportAttributeAccessIssue]
    CoreCallable,  # pyright: ignore[reportAttributeAccessIssue]
    Worker,  # pyright: ignore[reportAttributeAccessIssue]
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binary cache helpers
# ---------------------------------------------------------------------------


def _save_binary(data: bytes, path: Path) -> None:
    """Save compiled binary bytes to *path* atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        os.write(fd, data)
        os.close(fd)
        fd = -1
        os.replace(tmp_name, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _load_binary(path: Path) -> bytes | None:
    """Load compiled binary bytes from *path*. Returns ``None`` on miss."""
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except Exception:
        return None


def _kernel_cache_file(
    cache_dir: Path,
    kernel: dict,
    platform: str,
    pto_isa_root: str,
    runtime_name: str,
    compiler: KernelCompiler,
) -> Path:
    """Return a collision-free cache path for one compiled kernel binary."""
    include_dirs = []
    if kernel.get("external", False):
        include_dirs = [
            *compiler.get_incore_include_dirs(),
            *compiler.get_kernel_include_dirs(runtime_name),
            *(kernel.get("extra_include_dirs") or ()),
        ]
    return kernel_binary_cache_path(
        cache_dir,
        source=kernel["source"],
        core_type=kernel["core_type"],
        func_id=kernel.get("func_id", "anon"),
        platform=platform,
        external=bool(kernel.get("external", False)),
        pto_isa_root=pto_isa_root,
        runtime_name=runtime_name,
        include_dirs=include_dirs,
    )


def _clean_git_revision(repo_root: Path) -> tuple[bool, str | None]:
    """Return ``(is_checkout, HEAD)`` when *repo_root* is a clean checkout.

    ``is_checkout`` remains true for a dirty or unreadable checkout. Callers use
    that distinction to avoid falling back to packaged revision metadata that
    does not describe the source files which will actually be compiled.
    """
    if not (repo_root / ".git").exists():
        return False, None
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True, None

    revision = revision_result.stdout.strip()
    if (
        revision_result.returncode != 0
        or status_result.returncode != 0
        or not revision
        or status_result.stdout.strip()
    ):
        return True, None
    return True, revision


def _runtime_revision(compiler: KernelCompiler) -> str | None:
    """Return the runtime source/build revision used by *compiler*."""
    project_root = getattr(compiler, "project_root", None)
    if project_root is not None:
        is_checkout, revision = _clean_git_revision(Path(project_root))
        if is_checkout:
            return revision

    # Wheels carry the runtime sources under simpler_setup/_assets without a
    # .git directory. The runtime binding embeds the source commit at build time
    # so the wheel still has a stable compatibility identity.
    try:
        task_interface = import_module("_task_interface")
    except ImportError:
        return None
    revision = getattr(task_interface, "__build_commit__", "")
    return revision.strip() if isinstance(revision, str) and revision.strip() else None


def _current_binary_context(
    compiler: KernelCompiler,
    *,
    platform: str,
    runtime_name: str,
    pto_isa_root: str,
) -> BinaryCacheContext | None:
    """Build the compatibility identity for generated binary reuse."""
    runtime_revision = _runtime_revision(compiler)
    _, pto_isa_revision = _clean_git_revision(Path(pto_isa_root))
    if runtime_revision is None or pto_isa_revision is None:
        return None
    return BinaryCacheContext(
        platform=platform,
        runtime_name=runtime_name,
        runtime_revision=runtime_revision,
        pto_isa_revision=pto_isa_revision,
    )


# ---------------------------------------------------------------------------
# PTO-ISA management
# ---------------------------------------------------------------------------

# Re-exported from ``.pto_isa`` so this module keeps its historical entry point
# and so tests can monkey-patch the name used by ``_compile_and_assemble_locked``
# below. The implementation lives there because resolving the pin must not
# require the Simpler-backed ``task_interface`` extension this module imports.


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


@contextmanager
def _temporary_env(env_updates: dict[str, str]):
    """Temporarily apply env vars for the duration of the context."""
    old = {k: os.environ.get(k) for k in env_updates}
    for k, v in env_updates.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, prev in old.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


# ---------------------------------------------------------------------------
# Shared compilation functions
# ---------------------------------------------------------------------------


def compile_single_kernel(
    kernel: dict,
    compiler: KernelCompiler,
    platform: str,
    pto_isa_root: str,
    runtime_name: str,
    cache_dir: Path | None = None,
) -> tuple[bytes, bytes]:
    """Compile a single incore kernel with binary caching.

    Generated sources use a cached ``.o``/``.so`` alongside the artifact
    source. External sources never use sidecars: their final binary cache key
    includes source/include contents, core type, platform, and function id.
    For hardware platforms, extracts the ``.text`` section to produce the
    final kernel binary.

    When *cache_dir* is provided, the final (possibly stripped) binary is
    additionally written under that directory using the function/core identity
    and, for external kernels, the content fingerprint. This is the pre-build
    cache that :func:`compile_and_assemble` checks before calling this function.

    Args:
        kernel: Kernel descriptor dict with keys ``"source"``, ``"core_type"``,
            and optionally ``"signature"``, ``"func_id"``, ``"external"``,
            ``"extra_include_dirs"``.
        compiler: Configured :class:`KernelCompiler` instance.
        platform: Target execution platform.
        pto_isa_root: Resolved PTO-ISA root directory.
        runtime_name: Runtime name (e.g. ``"tensormap_and_ringbuffer"``).  Passed to
            :meth:`KernelCompiler.compile_incore` for include-dir resolution.
        cache_dir: Optional directory to write the final kernel binary for
            pre-build caching.

    Returns:
        ``(raw_binary, kernel_binary)`` where *raw_binary* is the compiled
        ``.o``/``.so`` and *kernel_binary* is the final binary (possibly
        ``.text``-extracted) ready for ``CoreCallable.build()``.
    """
    source = Path(kernel["source"])
    core_type = kernel["core_type"]

    ext = ".so" if platform.endswith("sim") else ".o"
    is_external = bool(kernel.get("external", False))
    output_file = None if is_external else source.with_suffix(ext)

    raw = None if output_file is None else _load_binary(output_file)
    if raw is None:
        raw = compiler.compile_incore(
            kernel["source"],
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            runtime_name=runtime_name,
            extra_include_dirs=kernel.get("extra_include_dirs"),
        )
        if output_file is not None:
            _save_binary(raw, output_file)

    kernel_bin = raw if platform.endswith("sim") else extract_text_section(raw)

    if cache_dir is not None:
        cache_file = _kernel_cache_file(cache_dir, kernel, platform, pto_isa_root, runtime_name, compiler)
        _save_binary(kernel_bin, cache_file)

    return raw, kernel_bin


def compile_single_orchestration(
    source: str | Path,
    compiler: KernelCompiler,
    runtime_name: str,
    cache_dir: Path | None = None,
) -> bytes:
    """Compile orchestration source to a shared library with binary caching.

    Checks for a cached ``.so`` alongside the source file. On miss, compiles
    via *compiler* and saves the result.

    When *cache_dir* is provided, the binary is additionally written to
    ``cache_dir/orch_{stem}.bin`` for the pre-build cache.

    Args:
        source: Path to the orchestration C++ source file.
        compiler: Configured :class:`KernelCompiler` instance.
        runtime_name: Runtime name (e.g. ``"tensormap_and_ringbuffer"``).
        cache_dir: Optional directory to write the binary for pre-build caching.

    Returns:
        Orchestration ``.so`` binary bytes.
    """
    source_path = Path(source)
    output_file = source_path.with_suffix(".so")

    raw = _load_binary(output_file)
    if raw is None:
        raw = compiler.compile_orchestration(runtime_name, str(source))
        _save_binary(raw, output_file)

    if cache_dir is not None:
        cache_file = cache_dir / f"orch_{source_path.stem}.bin"
        _save_binary(raw, cache_file)

    return raw


# ---------------------------------------------------------------------------
# compile_and_assemble
# ---------------------------------------------------------------------------

# ``hid`` (ELF Build-ID 64 of an orchestration ``.so``, lowercase hex) → that
# orchestration's display name (see ``callable_display_name``). The runtime
# identifies a callable in its ``[STRACE]`` timing markers by this hash alone (it
# never emits a name), so recording the pairing at assemble time is what lets
# ``pypto.runtime.benchmark`` label a measured dispatch readably. Process-wide and
# append-only: ``hid`` is content-derived, so two entries can only collide when
# the ``.so`` bytes are identical — in which case the name is identical too.
_CALLABLE_NAMES: dict[str, str] = {}


def callable_display_name(orchestration: dict[str, Any]) -> str:
    """A human-readable, per-program name for an ``ORCHESTRATION`` manifest entry.

    The manifest's ``function_name`` is the fixed AICPU entry symbol the runtime
    dlsym's — ``aicpu_orchestration_entry`` for *every* program — so it cannot
    tell two callables apart. The generated source file is named after the
    orchestration itself (``orchestration/prefill_fwd.cpp``, and for an L3 build
    its ``next_levels/<name>/`` directory matches), so its stem is the
    distinguishing name. Falls back to ``function_name`` if ``source`` is absent.
    """
    source = orchestration.get("source")
    return Path(source).stem if source else str(orchestration.get("function_name", ""))


def register_callable_identity(orch_so: bytes, name: str) -> str:
    """Record ``hid → name`` for an orchestration ``.so`` and return the hid.

    *orch_so* must be the exact buffer handed to the runtime (the same bytes it
    hashes in ``record_device_orch_callable``), so the computed hid matches the
    ``hid=`` field of that callable's ``[STRACE]`` markers.

    Args:
        orch_so: Complete orchestration shared-object bytes.
        name: Display name for the callable — see :func:`callable_display_name`.

    Returns:
        The callable's hid — :func:`~pypto.runtime.elf_parser.elf_build_id_64`
        formatted as lowercase hex, matching the marker wire format.
    """
    hid = f"{elf_build_id_64(orch_so):x}"
    _CALLABLE_NAMES.setdefault(hid, name)
    return hid


def callable_name(hid: str) -> str | None:
    """The orchestration's display name for *hid*, or ``None`` if unknown.

    Returns ``None`` when the callable was not assembled in this process, or on a
    ``*sim`` platform: the sim host seeds the marker hid with the runtime's
    ``callable_id`` rather than the ELF Build-ID, so marker hids do not match the
    hashes recorded here.
    """
    return _CALLABLE_NAMES.get(hid.lower())


def _missing_kernel_config_error(work_dir: Path) -> FileNotFoundError:
    """Explain why a build output has no runtime manifest."""
    config_path = work_dir / "kernel_config.py"
    kernels_dir = work_dir / "kernels"
    raw_pto_count = sum(1 for _ in kernels_dir.rglob("*.pto"))
    if raw_pto_count == 0:
        return FileNotFoundError(
            f"Cannot execute PyPTO artifact '{work_dir}': required '{config_path}' is missing.\n"
            f"No raw .pto kernel files were found under '{kernels_dir}' either. This directory "
            "cannot be loaded as a single-chip PyPTO artifact; it may be incomplete or use a "
            "different build layout.\n"
            "Recompile the program and inspect any earlier codegen error."
        )

    ptoas_root = os.environ.get("PTOAS_ROOT")
    configuration_step: str | None = None
    if ptoas_root:
        ptoas_path = Path(ptoas_root) / "ptoas"
        if ptoas_path.is_file() and os.access(ptoas_path, os.X_OK):
            environment_status = (
                f"ptoas is now available at '{ptoas_path}', but this artifact was generated "
                "without it and must be recompiled."
            )
        else:
            environment_status = (
                f"PTOAS_ROOT is set to '{ptoas_root}', but the expected executable "
                f"'{ptoas_path}' does not exist or is not executable."
            )
            configuration_step = (
                "Correct or remove the invalid PTOAS_ROOT setting:\n"
                "       export PTOAS_ROOT=/path/to/ptoas-bin\n"
                "     Or use a ptoas executable from PATH instead:\n"
                "       unset PTOAS_ROOT\n"
                "       export PATH=/path/to/ptoas-bin:$PATH\n"
                "     On a managed PyPTO development machine, reset the invalid override first:\n"
                "       unset PTOAS_ROOT\n"
                '       eval "$(pypto-setup --export)"'
            )
    else:
        ptoas_path = shutil.which("ptoas")
        if ptoas_path:
            environment_status = (
                f"ptoas is now available on PATH at '{ptoas_path}', but this artifact was "
                "generated without it and must be recompiled."
            )
        else:
            environment_status = "PTOAS_ROOT is not set and 'ptoas' was not found on PATH."
            configuration_step = (
                "Configure ptoas with one of these options:\n"
                "       export PTOAS_ROOT=/path/to/ptoas-bin\n"
                "     Or:\n"
                "       export PATH=/path/to/ptoas-bin:$PATH\n"
                "     On a managed PyPTO development machine, run:\n"
                '       eval "$(pypto-setup --export)"'
            )

    recovery_steps = []
    if configuration_step is not None:
        recovery_steps.append(configuration_step)
    recovery_steps.extend(
        [
            "Restart the Python process and rerun the program so the artifact is recompiled.",
            "If calling ir.compile() directly, use skip_ptoas=False.",
        ]
    )
    recovery_text = "\n".join(f"  {index}. {step}" for index, step in enumerate(recovery_steps, start=1))

    return FileNotFoundError(
        f"Cannot execute PyPTO artifact '{work_dir}': required '{config_path}' was not generated.\n"
        "Reason:\n"
        f"  Found {raw_pto_count} raw .pto kernel file(s) under '{kernels_dir}'.\n"
        "  This is a compile-only artifact produced with skip_ptoas=True. That mode intentionally "
        "omits kernel_config.py and cannot be executed. @pl.jit selects it automatically when "
        "ptoas is unavailable.\n"
        "PTOAS check:\n"
        f"  {environment_status}\n"
        f"How to fix:\n{recovery_text}"
    )


def compile_and_assemble(
    work_dir: Path,
    platform: str,
) -> tuple[ChipCallable, str, dict[str, Any]]:
    """Compile and assemble one chip artifact under a work-directory lock.

    Serializing the complete cache transaction prevents concurrent callers
    from pairing a context stamp with binaries produced for another runtime or
    platform.
    """
    if not (work_dir / "kernel_config.py").exists():
        raise _missing_kernel_config_error(work_dir)
    with binary_context_lock(work_dir):
        return _compile_and_assemble_locked(work_dir, platform)


def _compile_and_assemble_locked(
    work_dir: Path,
    platform: str,
) -> tuple[ChipCallable, str, dict[str, Any]]:
    """Compile kernels + orchestration from *work_dir*, assemble ``ChipCallable``.

    Reads ``kernel_config.py`` from *work_dir* to discover kernel sources,
    orchestration source, and runtime configuration.

    Args:
        work_dir: Root output directory containing ``kernels/``, ``orchestration/``,
            and ``kernel_config.py`` (produced by :func:`compile_program`).
        platform: Target execution platform.

    Returns:
        ``(chip_callable, runtime_name, runtime_config)`` — the assembled
        callable, the runtime name (e.g. ``"tensormap_and_ringbuffer"``),
        and the full ``RUNTIME_CONFIG`` dict loaded from
        ``kernel_config.py``. Callers can read defaults such as
        ``aicpu_thread_num`` from ``runtime_config`` — keys are only
        present when the producer of the artifact opted to bake them in.

    Raises:
        FileNotFoundError: If ``kernel_config.py`` is missing. Compile-only
            artifacts include the detected ptoas configuration and recovery
            steps in the error.

    Notes:
        The caller must hold :func:`binary_context_lock` for *work_dir*.
    """
    # Load kernel_config.py
    config_path = work_dir / "kernel_config.py"
    if not config_path.exists():
        raise _missing_kernel_config_error(work_dir)

    spec = importlib.util.spec_from_file_location("_kernel_config", str(config_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load kernel_config.py from {config_path}")
    kernel_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_config)

    kernels = kernel_config.KERNELS
    orchestration = kernel_config.ORCHESTRATION
    runtime_config = getattr(kernel_config, "RUNTIME_CONFIG", {})
    # Default to the runtime that ``pto_backend`` bakes into every generated
    # ``kernel_config.py``; only legacy / hand-written configs omit the key.
    runtime_name = runtime_config.get("runtime", runtime_kind_to_name(RuntimeKind.TENSORMAP_AND_RINGBUFFER))

    # Resolve the pinned PTO-ISA checkout (raises with an actionable message
    # when the pin cannot be honoured).
    pto_isa_root = ensure_pto_isa_root()

    # Create compiler
    compiler = KernelCompiler(platform=platform)

    # Generated binaries include runtime and PTO-ISA headers. A runtime bump can
    # therefore make both cache/*.bin and source-adjacent .so/.o files ABI
    # incompatible even though their paths did not change. Validate the whole
    # sub-build before the first cache lookup; legacy artifacts have no stamp and
    # are rebuilt once.
    binary_context = _current_binary_context(
        compiler,
        platform=platform,
        runtime_name=runtime_name,
        pto_isa_root=pto_isa_root,
    )
    invalidated = prepare_binary_context(work_dir, binary_context)
    if invalidated:
        if binary_context is None:
            logger.warning(
                "Could not determine the current Simpler/PTO-ISA revision; invalidated %d cached "
                "PyPTO binary file(s) under %s and will rebuild from generated C++",
                invalidated,
                work_dir,
            )
        else:
            logger.info(
                "Cached PyPTO binaries under %s have no matching build context for "
                "runtime %s@%s, platform %s, PTO-ISA %s; invalidated %d file(s) and "
                "will rebuild from generated C++",
                work_dir,
                runtime_name,
                binary_context.runtime_revision[:12],
                platform,
                binary_context.pto_isa_revision[:12],
                invalidated,
            )

    # --- Parallel compilation ---

    def _compile_one_kernel(kernel: dict) -> tuple[int, CoreCallable]:
        func_id = kernel["func_id"]

        # Check cache/ for pre-stripped binary (written by prebuild_binaries)
        prebuild_cache = work_dir / "cache"
        cache_file = _kernel_cache_file(
            prebuild_cache,
            kernel,
            platform,
            pto_isa_root,
            runtime_name,
            compiler,
        )
        cached_bin = _load_binary(cache_file)
        if cached_bin is not None:
            sig = kernel.get("signature", [])
            return (func_id, CoreCallable.build(signature=sig, binary=cached_bin))

        # Compile via shared function and populate the content-addressed cache.
        _, kernel_bin = compile_single_kernel(
            kernel,
            compiler,
            platform,
            pto_isa_root,
            runtime_name,
            cache_dir=prebuild_cache,
        )

        sig = kernel.get("signature", [])
        return (func_id, CoreCallable.build(signature=sig, binary=kernel_bin))

    def _compile_orchestration() -> bytes:
        source = Path(orchestration["source"])

        # Check cache/ for pre-built binary (written by prebuild_binaries)
        prebuild_cache = work_dir / "cache"
        cache_file = prebuild_cache / f"orch_{source.stem}.bin"
        cached_bin = _load_binary(cache_file)
        if cached_bin is not None:
            return cached_bin

        # Compile via shared function; skip secondary prebuild cache write
        return compile_single_orchestration(orchestration["source"], compiler, runtime_name)

    max_workers = min(64, 1 + len(kernels))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_orch = executor.submit(_compile_orchestration)
        fut_kernels = [executor.submit(_compile_one_kernel, k) for k in kernels]

        orch_so_binary = fut_orch.result()
        kernel_binaries = [f.result() for f in fut_kernels]

    # Assemble ChipCallable
    orch_sig = orchestration.get("signature", [])
    func_name = orchestration["function_name"]
    chip_callable = ChipCallable.build(
        signature=orch_sig,
        func_name=func_name,
        binary=orch_so_binary,
        children=kernel_binaries,
    )
    # ``orch_so_binary`` is the buffer the runtime hashes into the ``hid=`` of this
    # callable's [STRACE] markers; pair it with the per-program display name (NOT
    # ``func_name``, the shared AICPU entry symbol) so benchmark timing can be
    # attributed to a readable orchestration rather than an opaque hash.
    register_callable_identity(orch_so_binary, callable_display_name(orchestration))

    # A stamp denotes a complete, successfully assembled binary set. Do not
    # write it earlier: a failed kernel/orchestration compile must leave the
    # partial cache untrusted on the next attempt.
    record_binary_context(work_dir, binary_context)

    return chip_callable, runtime_name, runtime_config


# ---------------------------------------------------------------------------
# execute_on_device
# ---------------------------------------------------------------------------


def execute_on_device(  # noqa: PLR0913
    chip_callable: ChipCallable,
    orch_args: list[Any],
    platform: str,
    runtime_name: str,
    device_id: int,
    *,
    level: int = 2,
    aicpu_thread_num: int | None = None,
    enable_sdma: bool = False,
    output_prefix: str | None = None,
    enable_chip_swimlane: int | bool = 0,
    enable_dump_args: int = 0,
    enable_pmu: int = 0,
    enable_dep_gen: bool = False,
    enable_scope_stats: bool = False,
    runtime_env: dict[str, str] | None = None,
) -> None:
    """Execute *chip_callable* on device via Simpler's unified ``Worker``.

    If a :class:`pypto.runtime.ChipWorker` is currently active (the call site is
    inside a ``with ChipWorker(...):`` block) and matches the
    ``(level, platform, device_id, runtime_name)`` binding, that ChipWorker is
    reused — its already-initialized device context dispatches the run
    without re-running ``init`` / ``close``. Otherwise a fresh one-shot
    simpler Worker is constructed exactly as before.

    Args:
        chip_callable: Assembled callable (orchestration + kernels).
        orch_args: Ordered host tensors, worker-owned tensors, and scalar
            arguments. They are converted to the runtime's address-free
            ``TaskArgs`` only after the owning Worker is known.
        platform: Target execution platform (e.g. ``"a2a3sim"``).
        runtime_name: Runtime implementation name (e.g. ``"tensormap_and_ringbuffer"``).
        device_id: NPU device index.
        level: Hierarchy level. Only ``2`` (single-chip) is currently
            supported; passing any other value raises ``ValueError``. The
            parameter exists so callers can plumb level through ahead of L3
            user-API support.
        aicpu_thread_num: Number of AICPU threads. ``None`` leaves the
            field unset and uses the simpler runtime default.
        enable_sdma: Whether the worker must provision the SDMA workspace
            required by prefetch artifacts. Defaults to ``False`` for legacy,
            hand-built, and non-prefetch callables.
        output_prefix: Directory under which the runtime writes diagnostic
            artifacts (``chip_swimlane_records.json`` / ``args_dump/`` /
            ``pmu.csv`` / ``deps.json`` / ``scope_stats/``). Required
            whenever any ``enable_*`` DFX flag is set — Simpler's
            ``CallConfig::validate()`` would otherwise reject the call.
            Passing it with all flags off creates no artefacts.
        enable_chip_swimlane: Chip swimlane collection **level** for the
            per-task perf records (``chip_swimlane_records.json``). ``0`` off;
            ``1`` AICore timing; ``2`` plus AICPU dispatch / finish; ``3`` plus
            scheduler phases; ``4`` plus orchestrator phases. ``True`` requests
            the full level (``4``), matching the runtime harness's bare
            ``--enable-chip-swimlane``.
        enable_dump_args: Per-task argument dump level into
            ``<output_prefix>/args_dump/``. ``0`` off; ``1`` partial
            (only ``pl.dump_tag`` / ``dumps=`` marked tensors); ``2`` full
            (every task). Mirrors ``--dump-args``.
        enable_pmu: AICore PMU event type. ``0`` disables; ``>0`` selects
            an event type (``2`` = PIPE_UTILIZATION, ``4`` = MEMORY).
            Mirrors ``--enable-pmu N``.
        enable_dep_gen: Capture simpler dependency edges (``deps.json``).
            Mirrors ``--enable-dep-gen``.
        enable_scope_stats: Capture per-scope ring-fill peaks
            (``scope_stats/scope_stats.jsonl``). Mirrors
            ``--enable-scope-stats``.
        runtime_env: Optional per-example environment variable overrides.
            Applied around the device ``run`` call. When an active
            :class:`pypto.runtime.ChipWorker` is reused, ``init()`` has already
            executed before this call, so env vars that influence device
            initialization will not take effect on the reuse path — pass
            those at ``ChipWorker(...)`` construction instead.

    Returns:
        ``None``. The dispatch writes device results back into the host
        tensors in *orch_args* in place; per-run timing is no longer
        returned — read it from the runtime's ``[STRACE]`` log markers
        (simpler PR #1177) or the chip swimlane records instead.

    Raises:
        ValueError: If ``level != 2`` (L3 not yet exposed), or any DFX flag
            is enabled without a corresponding ``output_prefix``.
    """
    if level != 2:
        raise ValueError(
            f"execute_on_device currently only supports level=2; got level={level}. "
            f"L3 execution is not yet exposed at the pypto user-API layer."
        )

    from .runner import _normalize_swimlane_level  # noqa: PLC0415

    # Validate the level here too, so both entry points reject an out-of-range
    # request identically instead of letting the CallConfig setter clamp it.
    enable_chip_swimlane = _normalize_swimlane_level(enable_chip_swimlane, "enable_chip_swimlane")

    any_dfx = (
        enable_chip_swimlane > 0
        or enable_dump_args > 0
        or enable_pmu > 0
        or enable_dep_gen
        or enable_scope_stats
    )
    if any_dfx and not output_prefix:
        raise ValueError(
            "execute_on_device: output_prefix is required when any DFX flag "
            "(enable_chip_swimlane / enable_dump_args / enable_pmu / enable_dep_gen / "
            "enable_scope_stats) is enabled — runtime CallConfig::validate() would "
            "otherwise reject the call."
        )

    from .worker import ChipWorker as _PyptoWorker  # noqa: PLC0415

    cfg = CallConfig()
    if aicpu_thread_num is not None:
        cfg.aicpu_thread_num = aicpu_thread_num
    # CallConfig nanobind setters: ``enable_chip_swimlane`` accepts a bool or
    # level, while ``enable_dep_gen`` takes `bool`; ``enable_pmu`` is a raw
    # ``int32_t`` (0 disabled, >0 event type); ``enable_dump_args`` is a dump
    # level (0 off, 1 partial, 2 full) whose setter also accepts a bool
    # (True→1 partial, False→0).
    cfg.enable_chip_swimlane = enable_chip_swimlane
    cfg.enable_dump_args = enable_dump_args
    cfg.enable_pmu = enable_pmu
    cfg.enable_dep_gen = enable_dep_gen
    cfg.enable_scope_stats = enable_scope_stats
    if output_prefix:
        cfg.output_prefix = output_prefix

    env = runtime_env or {}
    active = _PyptoWorker.current(
        level=level,
        platform=platform,
        device_id=device_id,
        runtime=runtime_name,
        require_sdma=enable_sdma,
    )
    with _temporary_env(env):
        if active is not None:
            from .runner import _coerced_to_orch_args  # noqa: PLC0415

            wire_args = _coerced_to_orch_args(orch_args, active._impl)
            active._run_chip(chip_callable, wire_args, cfg)
            return
        worker = Worker(
            level=level,
            device_id=device_id,
            platform=platform,
            runtime=runtime_name,
            enable_sdma=enable_sdma,
        )
        # Prewarm with this dispatch's own config so the single run below hits the
        # prebuilt runtime-arena cache instead of paying the ~800ms cold build
        # inside the timed dispatch. No-op without a prebuilt arena.
        worker.init(prewarm_config=cfg)
        try:
            from .runner import _coerced_to_orch_args  # noqa: PLC0415

            wire_args = _coerced_to_orch_args(orch_args, worker)
            # Simpler's L2 ABI now dispatches by callable id (see runtime PR #710);
            # register the callable, run it, then close — close() runs finalize()
            # so explicit unregister is unnecessary here.
            cid = worker.register(chip_callable)
            worker.run(cid, wire_args, cfg)
        finally:
            worker.close()


# ---------------------------------------------------------------------------
# Golden validation
# ---------------------------------------------------------------------------


def validate_golden(
    outputs: dict[str, torch.Tensor],
    golden: dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Compare actual outputs against golden reference using ``torch.allclose``.

    Positions where the golden holds ``NaN`` are treated as *don't-care* and are
    excluded from the comparison. Tests use this to mark output regions that the
    kernel leaves undefined by contract — e.g. the area outside a tile's
    ``valid_shape``, or an oversized scratch buffer's unused tail. (The runtime
    no longer zero-fills pure-output buffers, so such regions hold pooled-allocator
    garbage rather than 0.) A golden with no ``NaN`` compares every element, so this
    is fully backward-compatible.

    Raises:
        AssertionError: If any output tensor does not match within tolerances.
    """
    for name, actual_tensor in outputs.items():
        actual = actual_tensor.cpu()
        expected = golden[name].cpu()
        logger.info(f"Comparing {name}: shape={actual.shape}, dtype={actual.dtype}")

        care_mask = ~torch.isnan(expected)
        # An element passes if it is close OR the golden marked it don't-care (NaN).
        close_mask = torch.isclose(actual, expected, rtol=rtol, atol=atol) | ~care_mask

        if not bool(close_mask.all()):
            mismatch_indices = torch.where(~close_mask.flatten())[0]
            flat_actual = actual.flatten()
            flat_expected = expected.flatten()
            n_show = min(20, mismatch_indices.numel())
            idx = mismatch_indices[:n_show]
            lines = [
                f"    [{i.item()}] actual={flat_actual[i].item()}, expected={flat_expected[i].item()}"
                for i in idx
            ]
            n_dont_care = int((~care_mask).sum().item())
            skipped = f" ({n_dont_care} don't-care skipped)" if n_dont_care else ""
            raise AssertionError(
                f"Output '{name}' does not match golden.\n"
                f"Mismatched elements: {mismatch_indices.numel()}/{actual.numel()}{skipped}\n"
                f"rtol={rtol}, atol={atol}\n"
                f"First {n_show} mismatches:\n" + "\n".join(lines)
            )

        n_compared = int(care_mask.sum().item())
        matched = int((close_mask & care_mask).sum().item())
        logger.info(f"  {name}: PASS ({matched}/{n_compared} compared elements matched)")


# ---------------------------------------------------------------------------
# Tensor argument construction
# ---------------------------------------------------------------------------

# Return type for build_orch_args_from_inputs. The first element deliberately
# remains unmaterialized until execute_on_device has selected its owning Worker.
_OrchArgsTuple = tuple[list[Any], dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]


def _collect_orch_args(
    items: list[tuple[str, torch.Tensor | ctypes._SimpleCData]],
    is_output: Callable[[str], bool],
) -> _OrchArgsTuple:
    """Normalize ordered ``(name, value)`` pairs for worker-owned packing.

    Args:
        items: Ordered ``(name, value)`` pairs.  Each value is either a
            ``torch.Tensor`` or a ``ctypes._SimpleCData`` scalar.
        is_output: Predicate that returns ``True`` if the named tensor is an
            output to be validated.

    Returns:
        ``(orch_args, all_tensors, inputs, outputs)``. ``orch_args`` is an
        ordered Python list; :func:`execute_on_device` turns it into
        address-free ``TaskArgs`` after selecting the Worker.
    """
    orch_args: list[Any] = []
    all_tensors: dict[str, Any] = {}
    inputs: dict[str, torch.Tensor] = {}
    outputs: dict[str, torch.Tensor] = {}

    for name, val in items:
        if isinstance(val, torch.Tensor):
            val = val.cpu().contiguous()
            orch_args.append(val)
            all_tensors[name] = val
            if is_output(name):
                outputs[name] = val
            else:
                inputs[name] = val
        elif isinstance(val, ctypes._SimpleCData):
            orch_args.append(val)
            all_tensors[name] = val.value

    return orch_args, all_tensors, inputs, outputs


def build_orch_args_from_inputs(
    inputs_result: list[tuple[str, Any]],
    output_names: set[str],
) -> _OrchArgsTuple:
    """Normalize pre-generated ``(name, value)`` tuples for device dispatch.

    This variant is used by the test harness path where inputs come from
    ``golden.py``'s ``generate_inputs()`` function rather than ``TensorSpec``.

    Args:
        inputs_result: List of ``(name, value)`` tuples where each value is
            either a ``torch.Tensor`` or a ``ctypes._SimpleCData`` scalar.
        output_names: Set of tensor names that are outputs.

    Returns:
        ``(orch_args, all_tensors, inputs, outputs)``.
    """
    return _collect_orch_args(
        inputs_result,
        lambda name: name in output_names or name.startswith("out"),
    )
