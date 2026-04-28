# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiler for PTO kernels and orchestration functions.

Public entry points:
- :meth:`KernelCompiler.compile_incore`: Compile a kernel source file for AICore/AIVector
- :meth:`KernelCompiler.compile_orchestration`: Compile an orchestration function

Toolchain selection is determined by platform:
- Hardware (a2a3/a5): CCEC for kernels, aarch64-g++ for orchestration
- Simulation (a2a3sim/a5sim): g++-15 for kernels, g++ for orchestration

C++ header paths (runtime headers, task_interface, platform includes) are resolved
from the ``simpler`` git submodule at the repository root.
"""

import importlib.util
import logging
import os
import subprocess
import sys
import tempfile

from . import env_manager
from .toolchain import (
    Aarch64GxxToolchain,
    CCECToolchain,
    Gxx15Toolchain,
    GxxToolchain,
    ToolchainType,
)

logger = logging.getLogger(__name__)


class KernelCompiler:
    """Compiler for PTO kernels and orchestration functions.

    Args:
        platform: Target platform (``"a2a3"``, ``"a2a3sim"``, ``"a5"``, or ``"a5sim"``).

    Raises:
        ValueError: If platform is unknown.
        EnvironmentError: If ``ASCEND_HOME_PATH`` is not set for hardware platforms.
        FileNotFoundError: If required compiler not found.
    """

    def __init__(self, platform: str = "a2a3"):
        self.platform = platform
        self.runtime_root = env_manager.get_simpler_root()

        # Map platform to architecture directory
        if platform in ("a2a3", "a2a3sim"):
            self.arch = "a2a3"
        elif platform in ("a5", "a5sim"):
            self.arch = "a5"
        else:
            raise ValueError(f"Unknown platform: {platform}")
        self.platform_dir = self.runtime_root / "src" / self.arch / "platform"

        # Create toolchain objects based on platform
        if platform in ("a2a3", "a5"):
            env_manager.ensure("ASCEND_HOME_PATH")
            self.ccec = CCECToolchain(platform)
            self.aarch64 = Aarch64GxxToolchain()
            self.host_gxx = GxxToolchain()
        else:
            self.ccec = None
            self.aarch64 = None
            self.host_gxx = GxxToolchain()

        self.gxx15 = Gxx15Toolchain()

    def get_platform_include_dirs(self) -> list[str]:
        """Get platform-specific include directories for orchestration compilation."""
        return [
            str(self.platform_dir / "include"),
        ]

    def get_orchestration_include_dirs(self, runtime_name: str) -> list[str]:
        """Get all include directories needed for orchestration compilation.

        Combines the runtime-specific directory with platform include directories.

        Args:
            runtime_name: Name of the runtime (e.g., ``"host_build_graph"``).

        Returns:
            List of include directory paths.
        """
        runtime_dir = str(self.runtime_root / "src" / self.arch / "runtime" / runtime_name / "runtime")
        common_dir = str(self.runtime_root / "src" / "common" / "task_interface")
        return [runtime_dir, common_dir] + self.get_platform_include_dirs()

    def get_kernel_include_dirs(self, runtime_name: str) -> list[str]:
        """Get include directories needed for incore kernel compilation.

        Reads ``build_config.py`` from the runtime directory to discover
        ``aicore`` include paths. Falls back to ``runtime/`` if no config
        exists. Always appends ``common/task_interface``.

        Args:
            runtime_name: Name of the runtime (e.g., ``"host_build_graph"``).

        Returns:
            List of absolute include directory paths.
        """
        runtime_base_dir = self.runtime_root / "src" / self.arch / "runtime" / runtime_name
        include_dirs: list[str] = []

        build_config_path = runtime_base_dir / "build_config.py"
        if build_config_path.is_file():
            spec = importlib.util.spec_from_file_location("build_config", str(build_config_path))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                aicore_cfg = mod.BUILD_CONFIG.get("aicore", {})
                for p in aicore_cfg.get("include_dirs", []):
                    include_dirs.append(str(runtime_base_dir / p))
        else:
            include_dirs.append(str(runtime_base_dir / "runtime"))
        include_dirs.append(str(self.runtime_root / "src" / "common" / "task_interface"))

        return include_dirs

    def _get_orchestration_config(self, runtime_name: str) -> tuple[list[str], list[str]]:
        """Load the optional "orchestration" section from a runtime's build_config.py.

        Returns:
            ``(include_dirs, source_files)`` — both as absolute paths, or ``([], [])``.
        """
        config_path = self.runtime_root / "src" / self.arch / "runtime" / runtime_name / "build_config.py"
        if not config_path.is_file():
            return [], []

        spec = importlib.util.spec_from_file_location("build_config", str(config_path))
        if spec is None or spec.loader is None:
            return [], []
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        build_config = getattr(mod, "BUILD_CONFIG", {})

        orch_cfg = build_config.get("orchestration")
        if orch_cfg is None:
            return [], []

        config_dir = config_path.parent

        include_dirs = [str((config_dir / p).resolve()) for p in orch_cfg.get("include_dirs", [])]

        source_files = []
        for src_dir_rel in orch_cfg.get("source_dirs", []):
            src_dir = (config_dir / src_dir_rel).resolve()
            if src_dir.is_dir():
                for f in sorted(src_dir.iterdir()):
                    if f.suffix in (".cpp", ".c") and f.is_file():
                        source_files.append(str(f))

        return include_dirs, source_files

    def _run_subprocess(
        self, cmd: list[str], label: str, error_hint: str = "Compiler not found"
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command with standardized logging and error handling."""
        logger.debug(f"[{label}] Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.stdout and logger.isEnabledFor(10):
                logger.debug(f"[{label}] stdout:\n{result.stdout}")
            if result.stderr and logger.isEnabledFor(10):
                logger.debug(f"[{label}] stderr:\n{result.stderr}")

            if result.returncode != 0:
                logger.error(f"[{label}] Compilation failed: {result.stderr}")
                raise RuntimeError(
                    f"{label} compilation failed with exit code {result.returncode}:\n{result.stderr}"
                )

            return result

        except FileNotFoundError:
            raise RuntimeError(error_hint)

    def _compile_to_bytes(
        self,
        cmd: list[str],
        output_path: str,
        label: str,
        error_hint: str = "Compiler not found",
        delete_output: bool = True,
    ) -> bytes:
        """Run compilation command, read output file, clean up, return bytes."""
        self._run_subprocess(cmd, label, error_hint)

        if not os.path.isfile(output_path):
            raise RuntimeError(f"Compilation succeeded but output file not found: {output_path}")

        with open(output_path, "rb") as f:
            binary_data = f.read()

        if delete_output:
            os.remove(output_path)
        logger.info(f"[{label}] Compilation {output_path} successful: {len(binary_data)} bytes")
        return binary_data

    def _get_toolchain(self, toolchain_map: dict) -> ToolchainType:
        """Get toolchain for the current platform."""
        if self.platform not in toolchain_map:
            raise ValueError(f"No toolchain for platform: {self.platform}")
        return toolchain_map[self.platform]

    @staticmethod
    def _make_temp_path(prefix: str, suffix: str, build_dir: str | None = None) -> str:
        """Create a unique temporary file path via mkstemp."""
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=build_dir or tempfile.gettempdir())
        os.close(fd)
        return path

    def compile_incore(
        self,
        source_path: str,
        core_type: str = "aiv",
        pto_isa_root: str | None = None,
        runtime_name: str | None = None,
        extra_include_dirs: list[str] | None = None,
        build_dir: str | None = None,
    ) -> bytes:
        """Compile a kernel source file.

        Dispatches based on platform:
        - Hardware (a2a3/a5): Uses ccec compiler (requires pto_isa_root)
        - Simulation (a2a3sim/a5sim): Uses g++-15

        Args:
            source_path: Path to kernel source file (.cpp).
            core_type: Core type: ``"aic"`` (cube) or ``"aiv"`` (vector).
            pto_isa_root: Path to PTO-ISA root directory.
            runtime_name: Name of the runtime (e.g., ``"host_build_graph"``).
                When provided, runtime include directories are resolved via
                :meth:`get_kernel_include_dirs` and prepended to includes.
            extra_include_dirs: Additional include directories.
            build_dir: Optional build directory for output files.

        Returns:
            Binary contents of the compiled .o file.
        """
        all_include_dirs: list[str] = []
        if runtime_name is not None:
            all_include_dirs.extend(self.get_kernel_include_dirs(runtime_name))
        if extra_include_dirs:
            all_include_dirs.extend(extra_include_dirs)
        incore_toolchain = self._get_toolchain(
            {
                "a2a3": ToolchainType.CCEC,
                "a2a3sim": ToolchainType.HOST_GXX_15,
                "a5": ToolchainType.CCEC,
                "a5sim": ToolchainType.HOST_GXX_15,
            },
        )

        if incore_toolchain == ToolchainType.HOST_GXX_15:
            return self._compile_incore_sim(
                source_path,
                core_type=core_type,
                pto_isa_root=pto_isa_root,
                extra_include_dirs=all_include_dirs or None,
                build_dir=build_dir,
            )

        assert self.ccec is not None, "ccec toolchain is only available for hardware platforms"
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if pto_isa_root is None:
            raise ValueError("pto_isa_root is required for incore compilation")

        pto_include = os.path.join(pto_isa_root, "include")
        pto_pto_include = os.path.join(pto_isa_root, "include", "pto")

        output_path = self._make_temp_path(
            prefix=f"{os.path.basename(source_path)}.incore_", suffix=".o", build_dir=build_dir
        )

        cmd = [self.ccec.cxx_path] + self.ccec.get_compile_flags(core_type=core_type)
        cmd.extend([f"-I{pto_include}", f"-I{pto_pto_include}"])

        if all_include_dirs:
            for inc_dir in all_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        cmd.extend(["-o", output_path, source_path])

        core_type_name = "AIV" if core_type == "aiv" else "AIC"
        logger.info(f"[Incore] Compiling ({core_type_name}): {source_path}")

        return self._compile_to_bytes(
            cmd,
            output_path,
            "Incore",
            error_hint=f"ccec compiler not found at {self.ccec.cxx_path}",
            delete_output=build_dir is None,
        )

    def compile_orchestration(
        self,
        runtime_name: str,
        source_path: str,
        extra_include_dirs: list[str] | None = None,
        build_dir: str | None = None,
    ) -> bytes:
        """Compile an orchestration function for the given runtime.

        Args:
            runtime_name: Name of the runtime (e.g., ``"tensormap_and_ringbuffer"``).
            source_path: Path to orchestration source file (.cpp).
            extra_include_dirs: Additional include directories.
            build_dir: Optional build directory for output files.

        Returns:
            Binary contents of the compiled orchestration .so file.
        """
        include_dirs = self.get_orchestration_include_dirs(runtime_name)
        if extra_include_dirs:
            include_dirs = include_dirs + list(extra_include_dirs)

        orch_includes, orch_sources = self._get_orchestration_config(runtime_name)
        if orch_includes:
            include_dirs = include_dirs + orch_includes

        toolchain_type = self._get_toolchain(
            {
                "a2a3": ToolchainType.AARCH64_GXX,
                "a2a3sim": ToolchainType.HOST_GXX,
                "a5": ToolchainType.AARCH64_GXX,
                "a5sim": ToolchainType.HOST_GXX,
            },
        )
        toolchain: GxxToolchain | Aarch64GxxToolchain
        if toolchain_type == ToolchainType.AARCH64_GXX:
            assert self.aarch64 is not None, "aarch64 toolchain is only available for hardware platforms"
            toolchain = self.aarch64
        else:
            toolchain = self.host_gxx

        return self._compile_orchestration_shared_lib(
            source_path,
            toolchain,
            extra_include_dirs=include_dirs,
            extra_sources=orch_sources or None,
            build_dir=build_dir,
        )

    def _compile_orchestration_shared_lib(
        self,
        source_path: str,
        toolchain: GxxToolchain | Aarch64GxxToolchain,
        extra_include_dirs: list[str] | None = None,
        extra_sources: list[str] | None = None,
        build_dir: str | None = None,
    ) -> bytes:
        """Compile an orchestration function to a shared library (.so)."""
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        output_path = self._make_temp_path(
            prefix=f"{os.path.basename(source_path)}.orch_", suffix=".so", build_dir=build_dir
        )

        cmd = [toolchain.cxx_path] + toolchain.get_compile_flags()

        # Force a deterministic ELF GNU Build-ID so simpler's DeviceRunner orch-SO
        # upload cache (keyed on .note.gnu.build-id) stays stable across GCC/ld
        # versions. The aarch64 cross-toolchain always produces Linux ELF and
        # ships GNU ld, so it supports --build-id even on a macOS host. Only
        # skip when host g++ runs on macOS (Mach-O target, Apple ld).
        if isinstance(toolchain, Aarch64GxxToolchain) or sys.platform != "darwin":
            cmd.append("-Wl,--build-id=sha1")

        if extra_sources:
            for src in extra_sources:
                src = os.path.abspath(src)
                if os.path.isfile(src):
                    cmd.append(src)
                    logger.debug(f"  Including extra source: {os.path.basename(src)}")

        if sys.platform == "darwin":
            cmd.append("-undefined")
            cmd.append("dynamic_lookup")

        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        cmd.extend(["-o", output_path, source_path])

        logger.info(f"[Orchestration] Compiling: {source_path}")

        return self._compile_to_bytes(
            cmd,
            output_path,
            "Orchestration",
            error_hint=f"{toolchain.cxx_path} not found. Please install it.",
            delete_output=build_dir is None,
        )

    def _compile_incore_sim(
        self,
        source_path: str,
        *,
        core_type: str,
        pto_isa_root: str | None = None,
        extra_include_dirs: list[str] | None = None,
        build_dir: str | None = None,
    ) -> bytes:
        """Compile a simulation kernel to .so/.dylib using g++-15."""
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        ext = ".dylib" if sys.platform == "darwin" else ".so"
        output_path = self._make_temp_path(
            prefix=f"{os.path.basename(source_path)}.sim_", suffix=ext, build_dir=build_dir
        )

        cmd = [self.gxx15.cxx_path] + self.gxx15.get_compile_flags(core_type=core_type)

        if pto_isa_root:
            pto_include = os.path.join(pto_isa_root, "include")
            pto_pto_include = os.path.join(pto_isa_root, "include", "pto")
            cmd.extend([f"-I{pto_include}", f"-I{pto_pto_include}"])

        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        cmd.extend(["-o", output_path, source_path])

        logger.info(f"[SimKernel] Compiling: {source_path}")

        return self._compile_to_bytes(
            cmd,
            output_path,
            "SimKernel",
            error_hint=f"{self.gxx15.cxx_path} not found. Please install g++-15.",
            delete_output=build_dir is None,
        )
