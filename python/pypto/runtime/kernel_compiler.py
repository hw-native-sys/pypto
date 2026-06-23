# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Thin pypto extension of ``simpler_setup.KernelCompiler``.

Adds the two pypto-specific pieces simpler_setup does not yet ship:

- :meth:`KernelCompiler.get_kernel_include_dirs` — reads ``aicore.include_dirs``
  from the runtime's ``build_config.py`` for incore compilation.
- :meth:`KernelCompiler.compile_incore` with ``runtime_name=...`` — when given,
  prepends those include dirs.

Everything else (toolchain selection, ``project_root`` resolution,
orchestration compilation, sim/ccec dispatch) is inherited unchanged from
``simpler_setup.KernelCompiler``. Once these two additions land in
simpler_setup (issue #1064), this file can be deleted and callers can
``from simpler_setup import KernelCompiler`` directly.
"""

import importlib.util

from simpler_setup import KernelCompiler as _SimplerKernelCompiler  # pyright: ignore[reportMissingImports]


class KernelCompiler(_SimplerKernelCompiler):
    """``simpler_setup.KernelCompiler`` + pypto's runtime-aware incore includes."""

    def _arch(self) -> str:
        """Map the configured platform to its runtime architecture directory."""
        if self.platform in ("a2a3", "a2a3sim"):
            return "a2a3"
        if self.platform in ("a5", "a5sim"):
            return "a5"
        raise ValueError(f"Unknown platform: {self.platform}")

    def get_kernel_include_dirs(self, runtime_name: str) -> list[str]:
        """Get include directories needed for incore kernel compilation.

        Reads ``build_config.py`` from the runtime directory to discover
        ``aicore`` include paths. Falls back to ``runtime/`` if no config
        exists. Always appends ``common/task_interface``.

        Args:
            runtime_name: Name of the runtime (e.g., ``"tensormap_and_ringbuffer"``).

        Returns:
            List of absolute include directory paths.
        """
        runtime_base_dir = self.project_root / "src" / self._arch() / "runtime" / runtime_name
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
        include_dirs.append(str(self.project_root / "src" / "common"))
        include_dirs.append(str(self.project_root / "src" / "common" / "task_interface"))

        return include_dirs

    def compile_incore(
        self,
        source_path: str,
        core_type: str = "aiv",
        pto_isa_root: str | None = None,
        runtime_name: str | None = None,
        extra_include_dirs: list[str] | None = None,
        build_dir: str | None = None,
    ) -> bytes:
        """Compile a kernel source file, resolving runtime includes when given.

        Identical to ``simpler_setup.KernelCompiler.compile_incore`` except for
        the extra ``runtime_name``: when provided, the runtime's kernel include
        directories are resolved via :meth:`get_kernel_include_dirs` and
        prepended to ``extra_include_dirs``.

        Args:
            source_path: Path to kernel source file (.cpp).
            core_type: Core type: ``"aic"`` (cube) or ``"aiv"`` (vector).
            pto_isa_root: Path to PTO-ISA root directory.
            runtime_name: Name of the runtime (e.g., ``"tensormap_and_ringbuffer"``).
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
        return super().compile_incore(
            source_path,
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=all_include_dirs or None,
            build_dir=build_dir,
        )
