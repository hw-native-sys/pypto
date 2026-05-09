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
- :func:`ensure_pto_isa_root`: Manage PTO-ISA repository (clone/checkout).

These functions eliminate all Python-level imports from Simpler. The only
Simpler dependency remaining is:

- ``pip install simpler`` → provides the ``_task_interface`` nanobind C++ module.
- The ``runtime/`` git submodule at the repository root provides C++ headers and
  pre-built runtime binaries.
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
from pathlib import Path
from typing import Any

import torch

from .elf_parser import extract_text_section
from .kernel_compiler import KernelCompiler
from .task_interface import (
    CallConfig,  # pyright: ignore[reportAttributeAccessIssue]
    ChipCallable,  # pyright: ignore[reportAttributeAccessIssue]
    ChipStorageTaskArgs,  # pyright: ignore[reportAttributeAccessIssue]
    CoreCallable,  # pyright: ignore[reportAttributeAccessIssue]
    Worker,  # pyright: ignore[reportAttributeAccessIssue]
    make_tensor_arg,  # pyright: ignore[reportAttributeAccessIssue]
    scalar_to_uint64,  # pyright: ignore[reportAttributeAccessIssue]
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


# ---------------------------------------------------------------------------
# PTO-ISA management
# ---------------------------------------------------------------------------

_PTO_ISA_HTTPS = "https://github.com/hw-native-sys/pto-isa.git"
_PTO_ISA_SSH = "git@github.com:hw-native-sys/pto-isa.git"
_PTO_ISA_HTTPS_FALLBACK = "https://gitcode.com/luohuan40/pto-isa.git"
_PTO_ISA_SSH_FALLBACK = "git@gitcode.com:luohuan40/pto-isa.git"
_PTO_ISA_PRIMARY_CLONE_TIMEOUT = 60
_PTO_ISA_FALLBACK_CLONE_TIMEOUT = 300


def _get_pto_isa_clone_path() -> Path:
    """Return the default path where PTO-ISA is cloned."""
    return Path(__file__).parent.parent.parent.parent / "build_output" / "_deps" / "pto-isa"


def _clone_pto_isa(clone_path: Path, primary_url: str, fallback_url: str) -> bool:
    """Clone pto-isa, trying *primary_url* first with a short timeout.

    Returns:
        ``True`` if the clone succeeded (based on return code), ``False`` otherwise.
    """
    clone_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["git", "clone", primary_url, str(clone_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=_PTO_ISA_PRIMARY_CLONE_TIMEOUT,
        )
        if result.returncode == 0:
            return True
    except subprocess.TimeoutExpired:
        logger.warning(
            f"Cloning pto-isa from {primary_url} timed out after "
            f"{_PTO_ISA_PRIMARY_CLONE_TIMEOUT}s; falling back to {fallback_url}"
        )
    except Exception as e:
        logger.warning(f"Cloning pto-isa from {primary_url} failed: {e}; falling back to {fallback_url}")

    # Clean up any partial clone before retrying with the fallback.
    if clone_path.exists():
        shutil.rmtree(clone_path, ignore_errors=True)
    try:
        result = subprocess.run(
            ["git", "clone", fallback_url, str(clone_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=_PTO_ISA_FALLBACK_CLONE_TIMEOUT,
        )
        if result.returncode != 0:
            logger.warning(f"Failed to clone pto-isa:\n{result.stderr}")
            return False
    except Exception as e:
        logger.warning(f"Failed to clone pto-isa: {e}")
        return False
    return True


def ensure_pto_isa_root(commit: str | None = None, clone_protocol: str = "https") -> str | None:
    """Ensure ``PTO_ISA_ROOT`` is available, either from env or by cloning.

    Args:
        commit: If provided, checkout this specific commit.
        clone_protocol: ``"https"`` or ``"ssh"``.

    Returns:
        PTO-ISA root path if successful, ``None`` otherwise.
    """
    existing_root = os.environ.get("PTO_ISA_ROOT")
    if existing_root:
        if commit:
            _checkout_pto_isa_commit(Path(existing_root), commit)
        return existing_root

    clone_path = _get_pto_isa_clone_path()
    include_dir = clone_path / "include"

    if not (clone_path.exists() and include_dir.exists() and include_dir.is_dir()):
        if clone_protocol == "https":
            primary_url, fallback_url = _PTO_ISA_HTTPS, _PTO_ISA_HTTPS_FALLBACK
        else:
            primary_url, fallback_url = _PTO_ISA_SSH, _PTO_ISA_SSH_FALLBACK
        if not _clone_pto_isa(clone_path, primary_url, fallback_url):
            return None
        if commit:
            _checkout_pto_isa_commit(clone_path, commit)
    elif commit:
        _checkout_pto_isa_commit(clone_path, commit)
    else:
        _update_pto_isa_to_latest(clone_path)

    if not include_dir.exists():
        return None

    resolved = str(clone_path.resolve())
    os.environ["PTO_ISA_ROOT"] = resolved
    return resolved


def _checkout_pto_isa_commit(clone_path: Path, commit: str) -> None:
    """Checkout the specified commit if the existing clone is at a different revision."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(clone_path),
            timeout=5,
        )
        current = result.stdout.strip() if result.returncode == 0 else ""
        if current and not commit.startswith(current) and not current.startswith(commit):
            subprocess.run(
                ["git", "fetch", "origin"],
                capture_output=True,
                text=True,
                cwd=str(clone_path),
                timeout=30,
                check=True,
            )
            subprocess.run(
                ["git", "checkout", commit],
                capture_output=True,
                text=True,
                cwd=str(clone_path),
                timeout=30,
                check=True,
            )
    except Exception as e:
        logger.warning(f"Failed to checkout pto-isa commit {commit}: {e}")


def _update_pto_isa_to_latest(clone_path: Path) -> None:
    """Fetch and reset existing clone to the remote default branch."""
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            capture_output=True,
            text=True,
            cwd=str(clone_path),
            timeout=30,
            check=True,
        )
        subprocess.run(
            ["git", "reset", "--hard", "origin/HEAD"],
            capture_output=True,
            text=True,
            cwd=str(clone_path),
            timeout=30,
            check=True,
        )
    except Exception as e:
        logger.warning(f"Failed to update pto-isa to latest: {e}")


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

    Checks for a cached ``.o``/``.so`` alongside the source file. On miss,
    compiles via *compiler* and saves the result. For hardware platforms,
    extracts the ``.text`` section to produce the final kernel binary.

    When *cache_dir* is provided, the final (possibly stripped) binary is
    additionally written to ``cache_dir/incore_{core_type}_{stem}.bin``.
    This is the pre-build cache that :func:`compile_and_assemble` checks
    before calling this function.

    Args:
        kernel: Kernel descriptor dict with keys ``"source"``, ``"core_type"``,
            and optionally ``"signature"``, ``"func_id"``.
        compiler: Configured :class:`KernelCompiler` instance.
        platform: Target execution platform.
        pto_isa_root: Resolved PTO-ISA root directory.
        runtime_name: Runtime name (e.g. ``"host_build_graph"``).  Passed to
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
    output_file = source.with_suffix(ext)

    raw = _load_binary(output_file)
    if raw is None:
        raw = compiler.compile_incore(
            kernel["source"],
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            runtime_name=runtime_name,
        )
        _save_binary(raw, output_file)

    kernel_bin = raw if platform.endswith("sim") else extract_text_section(raw)

    if cache_dir is not None:
        cache_file = cache_dir / f"incore_{core_type}_{source.stem}.bin"
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
        runtime_name: Runtime name (e.g. ``"host_build_graph"``).
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


def compile_and_assemble(
    work_dir: Path,
    platform: str,
    pto_isa_commit: str | None = None,
) -> tuple[ChipCallable, str]:
    """Compile kernels + orchestration from *work_dir*, assemble ``ChipCallable``.

    Reads ``kernel_config.py`` from *work_dir* to discover kernel sources,
    orchestration source, and runtime configuration.

    Args:
        work_dir: Root output directory containing ``kernels/``, ``orchestration/``,
            and ``kernel_config.py`` (produced by :func:`compile_program`).
        platform: Target execution platform.
        pto_isa_commit: If set, pin the pto-isa clone to this commit.

    Returns:
        ``(chip_callable, runtime_name)`` — the assembled callable and the
        runtime name (e.g. ``"tensormap_and_ringbuffer"``).
    """
    # Load kernel_config.py
    config_path = work_dir / "kernel_config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"kernel_config.py not found in {work_dir}")

    spec = importlib.util.spec_from_file_location("_kernel_config", str(config_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load kernel_config.py from {config_path}")
    kernel_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_config)

    kernels = kernel_config.KERNELS
    orchestration = kernel_config.ORCHESTRATION
    runtime_config = getattr(kernel_config, "RUNTIME_CONFIG", {})
    runtime_name = runtime_config.get("runtime", "host_build_graph")

    # Ensure PTO-ISA root
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    if pto_isa_root is None:
        raise OSError(
            "PTO_ISA_ROOT could not be resolved.\n"
            "Please set it to the PTO-ISA root directory, e.g.:\n"
            "  export PTO_ISA_ROOT=/path/to/pto-isa"
        )

    # Create compiler
    compiler = KernelCompiler(platform=platform)

    # --- Parallel compilation ---

    def _compile_one_kernel(kernel: dict) -> tuple[int, CoreCallable]:
        func_id = kernel["func_id"]
        source = Path(kernel["source"])
        core_type = kernel["core_type"]

        # Check cache/ for pre-stripped binary (written by prebuild_binaries)
        prebuild_cache = work_dir / "cache"
        cache_file = prebuild_cache / f"incore_{core_type}_{source.stem}.bin"
        cached_bin = _load_binary(cache_file)
        if cached_bin is not None:
            sig = kernel.get("signature", [])
            return (func_id, CoreCallable.build(signature=sig, binary=cached_bin))

        # Compile via shared function; skip secondary prebuild cache write
        _, kernel_bin = compile_single_kernel(kernel, compiler, platform, pto_isa_root, runtime_name)

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
    chip_callable = ChipCallable.build(
        signature=orch_sig,
        func_name=orchestration["function_name"],
        binary=orch_so_binary,
        children=kernel_binaries,
    )

    return chip_callable, runtime_name


# ---------------------------------------------------------------------------
# execute_on_device
# ---------------------------------------------------------------------------


def execute_on_device(
    chip_callable: ChipCallable,
    orch_args: ChipStorageTaskArgs,
    platform: str,
    runtime_name: str,
    device_id: int,
    *,
    level: int = 2,
    block_dim: int = 24,
    aicpu_thread_num: int = 4,
    output_prefix: str | None = None,
    runtime_env: dict[str, str] | None = None,
) -> None:
    """Execute *chip_callable* on device via Simpler's unified ``Worker``.

    If a :class:`pypto.runtime.Worker` is currently active (the call site is
    inside a ``with Worker(...):`` block) and matches the
    ``(level, platform, device_id, runtime_name)`` binding, that Worker is
    reused — its already-initialized device context dispatches the run
    without re-running ``init`` / ``close``. Otherwise a fresh one-shot
    Worker is constructed exactly as before.

    Args:
        chip_callable: Assembled callable (orchestration + kernels).
        orch_args: Tensor/scalar arguments.
        platform: Target execution platform (e.g. ``"a2a3sim"``).
        runtime_name: Runtime implementation name (e.g. ``"tensormap_and_ringbuffer"``).
        device_id: NPU device index.
        level: Hierarchy level. Only ``2`` (single-chip) is currently
            supported; passing any other value raises ``ValueError``. The
            parameter exists so callers can plumb level through ahead of L3
            user-API support.
        block_dim: Block dimension for execution.
        aicpu_thread_num: Number of AICPU threads.
        output_prefix: Directory under which the runtime writes diagnostic
            artifacts (``l2_perf_records.json`` etc.). Passing a non-empty
            value also enables L2 swimlane profiling; pass ``None`` to run
            without profiling. Simpler's ``CallConfig::validate()`` rejects
            an empty prefix whenever any diagnostic flag is enabled, so the
            two are kept linked at this entry point.
        runtime_env: Optional per-example environment variable overrides.
            Applied around the device ``run`` call. When an active
            :class:`pypto.runtime.Worker` is reused, ``init()`` has already
            executed before this call, so env vars that influence device
            initialization will not take effect on the reuse path — pass
            those at ``Worker(...)`` construction instead.
    """
    if level != 2:
        raise ValueError(
            f"execute_on_device currently only supports level=2; got level={level}. "
            f"L3 execution is not yet exposed at the pypto user-API layer."
        )

    enable_profiling = bool(output_prefix)

    from .worker import Worker as _PyptoWorker  # noqa: PLC0415

    cfg = CallConfig()
    cfg.block_dim = block_dim
    cfg.aicpu_thread_num = aicpu_thread_num
    cfg.enable_l2_swimlane = enable_profiling
    if enable_profiling:
        cfg.output_prefix = output_prefix

    env = runtime_env or {}
    active = _PyptoWorker.current(level=level, platform=platform, device_id=device_id, runtime=runtime_name)
    with _temporary_env(env):
        if active is not None:
            active._run_chip(chip_callable, orch_args, cfg)
            return
        worker = Worker(level=level, device_id=device_id, platform=platform, runtime=runtime_name)
        worker.init()
        worker.run(chip_callable, orch_args, cfg)
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

    Raises:
        AssertionError: If any output tensor does not match within tolerances.
    """
    for name, actual_tensor in outputs.items():
        actual = actual_tensor.cpu()
        expected = golden[name].cpu()
        logger.info(f"Comparing {name}: shape={actual.shape}, dtype={actual.dtype}")

        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            close_mask = torch.isclose(actual, expected, rtol=rtol, atol=atol)
            mismatch_indices = torch.where(~close_mask.flatten())[0]
            flat_actual = actual.flatten()
            flat_expected = expected.flatten()
            n_show = min(20, mismatch_indices.numel())
            idx = mismatch_indices[:n_show]
            lines = [
                f"    [{i.item()}] actual={flat_actual[i].item()}, expected={flat_expected[i].item()}"
                for i in idx
            ]
            raise AssertionError(
                f"Output '{name}' does not match golden.\n"
                f"Mismatched elements: {mismatch_indices.numel()}/{actual.numel()}\n"
                f"rtol={rtol}, atol={atol}\n"
                f"First {n_show} mismatches:\n" + "\n".join(lines)
            )

        matched = torch.isclose(actual, expected, rtol=rtol, atol=atol).sum().item()
        logger.info(f"  {name}: PASS ({matched}/{actual.numel()} elements matched)")


# ---------------------------------------------------------------------------
# Tensor argument construction
# ---------------------------------------------------------------------------

# Return type for build_orch_args_from_inputs.
_OrchArgsTuple = tuple[ChipStorageTaskArgs, dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]


def _collect_orch_args(
    items: list[tuple[str, torch.Tensor | ctypes._SimpleCData]],
    is_output: Callable[[str], bool],
) -> _OrchArgsTuple:
    """Shared logic for building ``ChipStorageTaskArgs`` from ``(name, value)`` pairs.

    Args:
        items: Ordered ``(name, value)`` pairs.  Each value is either a
            ``torch.Tensor`` or a ``ctypes._SimpleCData`` scalar.
        is_output: Predicate that returns ``True`` if the named tensor is an
            output to be validated.

    Returns:
        ``(orch_args, all_tensors, inputs, outputs)``.
    """
    orch_args = ChipStorageTaskArgs()
    all_tensors: dict[str, Any] = {}
    inputs: dict[str, torch.Tensor] = {}
    outputs: dict[str, torch.Tensor] = {}

    for name, val in items:
        if isinstance(val, torch.Tensor):
            val = val.cpu().contiguous()
            orch_args.add_tensor(make_tensor_arg(val))
            all_tensors[name] = val
            if is_output(name):
                outputs[name] = val
            else:
                inputs[name] = val
        elif isinstance(val, ctypes._SimpleCData):
            orch_args.add_scalar(scalar_to_uint64(val))
            all_tensors[name] = val.value

    return orch_args, all_tensors, inputs, outputs


def build_orch_args_from_inputs(
    inputs_result: list[tuple[str, Any]],
    output_names: set[str],
) -> _OrchArgsTuple:
    """Build ``ChipStorageTaskArgs`` from pre-generated ``(name, value)`` tuples.

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
