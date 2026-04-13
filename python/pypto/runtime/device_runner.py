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
- ``SIMPLER_ROOT`` → provides C++ headers and pre-built runtime binaries.
"""

from __future__ import annotations

import ctypes
import fcntl
import importlib.util
import logging
import os
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .elf_parser import extract_text_section
from .kernel_compiler import KernelCompiler
from .task_interface import (
    ChipCallable,  # pyright: ignore[reportAttributeAccessIssue]
    ChipCallConfig,  # pyright: ignore[reportAttributeAccessIssue]
    ChipStorageTaskArgs,  # pyright: ignore[reportAttributeAccessIssue]
    ChipWorker,
    CoreCallable,  # pyright: ignore[reportAttributeAccessIssue]
    make_tensor_arg,
    scalar_to_uint64,
)
from .tensor_spec import ScalarSpec, TensorSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RuntimeBinaries
# ---------------------------------------------------------------------------


@dataclass
class RuntimeBinaries:
    """Paths to compiled runtime binaries (host, aicpu, aicore)."""

    host_path: Path
    aicpu_path: Path
    aicore_path: Path
    sim_context_path: Path | None = None


# ---------------------------------------------------------------------------
# Binary cache helpers
# ---------------------------------------------------------------------------

_BINARY_RUNTIME_CACHE = (
    Path(__file__).parent.parent.parent.parent / "build_output" / "binary_cache" / "runtimes"
)


def _save_binary(data: bytes, path: Path) -> None:
    """Save compiled binary bytes to *path* atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _load_binary(path: Path) -> bytes | None:
    """Load compiled binary bytes from *path*. Returns ``None`` on miss."""
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except Exception:
        return None


def _get_simpler_stamp() -> str:
    """Return Simpler's current git commit (short hash) as a cache-key stamp."""
    simpler_root = os.environ.get("SIMPLER_ROOT", "")
    if not simpler_root:
        return "unknown"
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=simpler_root,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

_PLATFORM_MAP = {
    "a2a3": ("a2a3", "onboard"),
    "a2a3sim": ("a2a3", "sim"),
    "a5": ("a5", "onboard"),
    "a5sim": ("a5", "sim"),
}


def _parse_platform(platform: str) -> tuple[str, str]:
    """Parse platform string into ``(arch, variant)``."""
    if platform not in _PLATFORM_MAP:
        raise ValueError(f"Unknown platform: {platform!r}. Expected one of {list(_PLATFORM_MAP)}")
    return _PLATFORM_MAP[platform]


# ---------------------------------------------------------------------------
# PTO-ISA management
# ---------------------------------------------------------------------------

_PTO_ISA_HTTPS = "https://github.com/PTO-ISA/pto-isa.git"
_PTO_ISA_SSH = "git@github.com:PTO-ISA/pto-isa.git"


def _get_pto_isa_clone_path() -> Path:
    """Return the default path where PTO-ISA is cloned."""
    return Path(__file__).parent.parent.parent.parent / "_deps" / "pto-isa"


def ensure_pto_isa_root(commit: str | None = None, clone_protocol: str = "https") -> str | None:
    """Ensure ``PTO_ISA_ROOT`` is available, either from env or by cloning.

    Uses a file lock to prevent parallel processes from racing on the clone.

    Args:
        commit: If provided, checkout this specific commit.
        clone_protocol: ``"https"`` or ``"ssh"``.

    Returns:
        PTO-ISA root path if successful, ``None`` otherwise.
    """
    existing_root = os.environ.get("PTO_ISA_ROOT")
    if existing_root:
        return existing_root

    clone_path = _get_pto_isa_clone_path()
    lock_path = clone_path.parent / ".pto-isa.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return _ensure_pto_isa_root_locked(clone_path, commit=commit, clone_protocol=clone_protocol)


def _ensure_pto_isa_root_locked(
    clone_path: Path, commit: str | None = None, clone_protocol: str = "https"
) -> str | None:
    """Inner logic for :func:`ensure_pto_isa_root`, called while holding the file lock."""
    include_dir = clone_path / "include"

    if not (clone_path.exists() and include_dir.exists() and include_dir.is_dir()):
        # Need to clone
        repo_url = _PTO_ISA_HTTPS if clone_protocol == "https" else _PTO_ISA_SSH
        clone_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["git", "clone", repo_url, str(clone_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                # Another process may have succeeded
                if not include_dir.exists():
                    logger.warning(f"Failed to clone pto-isa:\n{result.stderr}")
                    return None
            if commit:
                subprocess.run(
                    ["git", "checkout", commit],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=str(clone_path),
                    timeout=30,
                )
        except (subprocess.TimeoutExpired, Exception) as e:
            if not include_dir.exists():
                logger.warning(f"Failed to clone pto-isa: {e}")
                return None
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
                timeout=120,
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
            timeout=120,
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


def _kernel_config_runtime_env(kernel_config_module, kernels_dir: Path) -> dict[str, str]:
    """Extract optional per-example environment variables from kernel_config.py."""
    runtime_env = getattr(kernel_config_module, "RUNTIME_ENV", None)
    if not isinstance(runtime_env, dict):
        return {}

    out: dict[str, str] = {}
    for k, v in runtime_env.items():
        if not isinstance(k, str):
            continue
        s = str(v)
        is_path_like = k.endswith("_DIR") or k.endswith("_PATH")
        if is_path_like and s:
            p = Path(s)
            if not p.is_absolute():
                s = str((kernels_dir / p).resolve())
        out[k] = s
    return out


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
) -> tuple[ChipCallable, RuntimeBinaries]:
    """Compile kernels + orchestration from *work_dir*, assemble ``ChipCallable``.

    Reads ``kernel_config.py`` from *work_dir* to discover kernel sources,
    orchestration source, and runtime configuration.

    Binary caching is integrated: cached binaries are served from
    ``work_dir/cache/`` and ``build_output/binary_cache/runtimes/``.

    Args:
        work_dir: Root output directory containing ``kernels/``, ``orchestration/``,
            and ``kernel_config.py`` (produced by :func:`compile_program`).
        platform: Target execution platform.
        pto_isa_commit: If set, pin the pto-isa clone to this commit.

    Returns:
        ``(chip_callable, runtime_binaries)`` ready for execution.
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

    max_workers = 1 + len(kernels)
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

    # Locate runtime binaries
    binaries = _find_runtime_binaries(platform, runtime_name)

    return chip_callable, binaries


# ---------------------------------------------------------------------------
# Runtime binary lookup
# ---------------------------------------------------------------------------


def _find_runtime_binaries(platform: str, runtime_name: str) -> RuntimeBinaries:
    """Find pre-built runtime binaries from ``SIMPLER_ROOT/build/lib/``.

    Also checks the persistent binary cache under ``build_output/binary_cache/runtimes/``.
    """
    stamp = _get_simpler_stamp()
    arch, variant = _parse_platform(platform)

    # Check persistent cache first
    cache_dir = _BINARY_RUNTIME_CACHE / stamp
    host_cache = cache_dir / f"{runtime_name}_{platform}_host.bin"
    aicpu_cache = cache_dir / f"{runtime_name}_{platform}_aicpu.bin"
    aicore_cache = cache_dir / f"{runtime_name}_{platform}_aicore.bin"

    if host_cache.exists() and aicpu_cache.exists() and aicore_cache.exists():
        sim_ctx = _resolve_sim_context(arch, variant)
        return RuntimeBinaries(
            host_path=host_cache,
            aicpu_path=aicpu_cache,
            aicore_path=aicore_cache,
            sim_context_path=sim_ctx,
        )

    # Look up from SIMPLER_ROOT/build/lib/
    simpler_root = Path(os.environ["SIMPLER_ROOT"])
    lib_dir = simpler_root / "build" / "lib" / arch / variant / runtime_name

    # Binary names match RuntimeCompiler's target config
    host_name = "libhost_runtime.so"
    aicpu_name = "libaicpu_kernel.so"
    aicore_name = "aicore_kernel.o" if variant == "onboard" else "libaicore_kernel.so"

    host_path = lib_dir / host_name
    aicpu_path = lib_dir / aicpu_name
    aicore_path = lib_dir / aicore_name

    missing = [str(p) for p in (host_path, aicpu_path, aicore_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pre-built runtime binaries not found for '{runtime_name}' "
            f"(platform={platform}):\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nRun 'cd $SIMPLER_ROOT && pip install .' to build them."
        )

    # Cache for future use
    _save_binary(host_path.read_bytes(), host_cache)
    _save_binary(aicpu_path.read_bytes(), aicpu_cache)
    _save_binary(aicore_path.read_bytes(), aicore_cache)

    sim_ctx = _resolve_sim_context(arch, variant)
    return RuntimeBinaries(
        host_path=host_path,
        aicpu_path=aicpu_path,
        aicore_path=aicore_path,
        sim_context_path=sim_ctx,
    )


def _resolve_sim_context(arch: str, variant: str) -> Path | None:
    """Return path to ``libcpu_sim_context.so`` for sim platforms, ``None`` for onboard."""
    if variant != "sim":
        return None
    simpler_root = Path(os.environ.get("SIMPLER_ROOT", ""))
    if not simpler_root:
        return None
    return simpler_root / "build" / "lib" / arch / variant / "libcpu_sim_context.so"


# ---------------------------------------------------------------------------
# execute_on_device
# ---------------------------------------------------------------------------


def execute_on_device(
    chip_callable: ChipCallable,
    orch_args: ChipStorageTaskArgs,
    runtime_binaries: RuntimeBinaries,
    device_id: int,
    *,
    block_dim: int = 24,
    aicpu_thread_num: int = 3,
    enable_profiling: bool = False,
    runtime_env: dict[str, str] | None = None,
) -> None:
    """Execute *chip_callable* on device via ``ChipWorker``.

    Args:
        chip_callable: Assembled callable (orchestration + kernels).
        orch_args: Tensor/scalar arguments.
        runtime_binaries: Paths to host/aicpu/aicore runtime binaries.
        device_id: NPU device index.
        block_dim: Block dimension for execution.
        aicpu_thread_num: Number of AICPU threads.
        enable_profiling: Enable runtime profiling.
        runtime_env: Optional per-example environment variable overrides.
    """
    worker = ChipWorker()
    worker.init(
        str(runtime_binaries.host_path),
        str(runtime_binaries.aicpu_path),
        str(runtime_binaries.aicore_path),
        sim_context_lib_path=str(runtime_binaries.sim_context_path)
        if runtime_binaries.sim_context_path
        else "",
    )
    worker.set_device(device_id)

    config = ChipCallConfig()
    config.block_dim = block_dim
    config.aicpu_thread_num = aicpu_thread_num
    if enable_profiling:
        config.enable_profiling = True

    env = runtime_env or {}
    with _temporary_env(env):
        worker.run(chip_callable, orch_args, config)

    worker.reset_device()
    worker.finalize()


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

        if actual.numel() > 0:
            flat_actual = actual.flatten()
            flat_expected = expected.flatten()
            n_show = min(10, flat_actual.numel())
            logger.debug(f"  First {n_show} actual:   {flat_actual[:n_show].tolist()}")
            logger.debug(f"  First {n_show} expected: {flat_expected[:n_show].tolist()}")

        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            close_mask = torch.isclose(actual, expected, rtol=rtol, atol=atol)
            mismatches = (~close_mask).sum().item()
            total = actual.numel()
            raise AssertionError(
                f"Output '{name}' does not match golden.\n"
                f"Mismatched elements: {mismatches}/{total}\n"
                f"rtol={rtol}, atol={atol}"
            )

        matched = torch.isclose(actual, expected, rtol=rtol, atol=atol).sum().item()
        logger.info(f"  {name}: PASS ({matched}/{actual.numel()} elements matched)")


# ---------------------------------------------------------------------------
# Tensor argument construction
# ---------------------------------------------------------------------------

# Return type shared by build_orch_args / build_orch_args_from_inputs.
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


def build_orch_args(
    tensor_specs: list[TensorSpec],
    scalar_specs: list[ScalarSpec] | None = None,
) -> _OrchArgsTuple:
    """Build ``ChipStorageTaskArgs`` from tensor and scalar specs.

    Creates tensors from *tensor_specs*, adds them to a ``ChipStorageTaskArgs``,
    then appends any scalar arguments from *scalar_specs*.

    Args:
        tensor_specs: List of ``TensorSpec`` objects.
        scalar_specs: Optional list of ``ScalarSpec`` objects for scalar TaskArg
            parameters.

    Returns:
        ``(orch_args, all_tensors, inputs, outputs)`` where:
        - *orch_args*: Ready for ``worker.run()``.
        - *all_tensors*: All named tensors.
        - *inputs*: Non-output tensors.
        - *outputs*: Output tensors.
    """
    output_names = {spec.name for spec in tensor_specs if spec.is_output}

    items: list[tuple[str, torch.Tensor | ctypes._SimpleCData]] = [
        (spec.name, spec.create_tensor()) for spec in tensor_specs
    ]
    if scalar_specs:
        for scalar_spec in scalar_specs:
            ctype_cls = getattr(ctypes, f"c_{scalar_spec.ctype}")
            items.append((scalar_spec.name, ctype_cls(scalar_spec.value)))

    return _collect_orch_args(items, lambda name: name in output_names)


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
        ``(orch_args, all_tensors, inputs, outputs)`` — same layout as
        :func:`build_orch_args`.
    """
    return _collect_orch_args(
        inputs_result,
        lambda name: name in output_names or name.startswith("out"),
    )
