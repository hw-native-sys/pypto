# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO-owned kernel builds using metadata from the installed Simpler SDK.

The current SDK exposes discovery on its KernelCompiler object. We consume
only its toolchain/include/source queries; no SDK compile method runs here.
Commands, temporary outputs, linking and returned bytes belong to PyPTO.
"""

import hashlib
import importlib.util
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

from pypto._environment import toolchain_probe_env
from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary

from . import _device_image, _stack_usage, _vpto

# `check_runtime_pin()` has to run before the simpler imports it guards, so the imports
# below are deliberately not at the top of the file (E402 is ignored in pyproject.toml).
from .runtime_pin import check_runtime_pin

# The same guard `task_interface` applies, for the same reason: simpler_setup's compile
# paths and toolchain tables move with the revision too. Cached, so this second call is free.
check_runtime_pin()

# Simpler is an optional build dependency, absent from compiler-only type-check environments.
from simpler_setup import KernelCompiler as _SimplerCompilerSDK
from simpler_setup.compile_paths import compiler_visible_path
from simpler_setup.toolchain import GxxToolchain

logger = logging.getLogger(__name__)


class KernelCompiler:
    """Build program binaries without initializing or submitting to a Worker."""

    _sanitizers = ""

    def __init__(self, platform: str = "a2a3"):
        """Load SDK metadata and the per-process compiler timeout in seconds.

        ``PYPTO_COMPILER_TIMEOUT`` defaults to 900 seconds and must be positive
        and finite. Invalid configuration raises ``ValueError`` before SDK setup.
        """
        timeout = os.environ.get("PYPTO_COMPILER_TIMEOUT", "900")
        try:
            self._timeout_s = float(timeout)
        except ValueError as exc:
            raise ValueError(
                f"PYPTO_COMPILER_TIMEOUT must be positive finite seconds, got {timeout!r}"
            ) from exc
        if not math.isfinite(self._timeout_s) or self._timeout_s <= 0:
            raise ValueError(f"PYPTO_COMPILER_TIMEOUT must be positive finite seconds, got {timeout!r}")
        self.platform = platform
        self.sdk = _SimplerCompilerSDK(platform)
        self._sanitizers = self._sanitizers or getattr(self.sdk, "_sanitizers", "")
        if self._sanitizers:
            self.sdk.host_gxx = GxxToolchain(prefer_g15=True)
            self.sdk._sanitizers = self._sanitizers
            if self.sdk.ccu is not None:
                self.sdk.ccu.host_toolchain = self.sdk.host_gxx
        self.project_root = self.sdk.project_root

    def get_incore_include_dirs(self) -> list[str]:
        """Return the SDK's shared kernel headers."""
        return self.sdk.get_incore_include_dirs()

    def get_orchestration_cache_inputs(self, runtime_name: str) -> tuple[list[str], list[str]]:
        """Return the exact SDK headers and helper sources consumed by the build."""
        return self.sdk.get_orchestration_cache_inputs(runtime_name)

    def _orchestration_toolchain(self, runtime_name: str) -> Any:
        """Select the SDK toolchain for the runtime's orchestration target."""
        return self.sdk._orchestration_toolchain(runtime_name)

    def _sanitizer_flags(self, toolchain: Any) -> list[str]:
        """Return sanitizer options only for builds using a host toolchain."""
        if not self._sanitizers or not toolchain.is_host:
            return []
        return [f"-fsanitize={self._sanitizers}", "-fno-omit-frame-pointer", "-O1"]

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

    @staticmethod
    def _source_path(source_path: str) -> Path:
        """Resolve a build input and reject missing source files."""
        source = Path(source_path).absolute()
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        return source

    @staticmethod
    def _include_flags(include_dirs: list[str]) -> list[str]:
        """Translate include directories to paths visible to the SDK compiler."""
        return [f"-I{compiler_visible_path(Path(path).absolute())}" for path in include_dirs]

    def _run(self, cmd: list[str], output: Path, label: str, *, env: dict[str, str] | None = None) -> bytes:
        """Run a bounded compiler process and return its nonempty binary output."""
        logger.debug(f"[{label}] Command: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{label}: compiler timed out after {self._timeout_s:g} seconds") from exc
        except OSError as exc:
            raise RuntimeError(f"{label}: cannot run compiler {cmd[0]!r}: {exc}") from exc
        if result.returncode:
            raise RuntimeError(
                f"{label} compilation failed with exit code {result.returncode}:\n{result.stderr}"
            )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"{label}: compiler produced no binary at {output}")
        return output.read_bytes()

    def compile_incore(
        self,
        source_path: str,
        core_type: str = "aiv",
        pto_isa_root: str | None = None,
        runtime_name: str | None = None,
        extra_include_dirs: list[str] | None = None,
        build_dir: str | None = None,
        *,
        vpto_sources: list[str] | tuple[str, ...] | None = None,
        text_only: bool = False,
        _vpto_snapshot: tuple[str, ...] | None = None,
    ) -> bytes:
        """Compile a simulator SO or linked device ELF in a private build directory.

        SDK toolchains supply fixed target flags. PyPTO owns the invocation,
        output validation and cleanup, including failed compiles and links.
        ``build_dir`` optionally selects the parent for temporary build files.
        ``vpto_sources`` supplies self-contained A5 AIV device functions. They
        are compiled to device objects and linked before .text extraction;
        Host fatobjs and C++ Bridge dependencies are not supported here.
        ``text_only=True`` admits only A5 compiler-produced, static zero-stack
        callees. It discards unreachable sections and validates the final ELF
        and compiler stack report before returning bytes. It does not establish
        resource ownership, submission ordering or device completion semantics.
        """
        source = self._source_path(source_path)
        helpers = _vpto.source_paths(vpto_sources)
        if text_only and (self.platform != "a5" or helpers):
            raise ValueError("text_only requires platform='a5' without separately compiled VPTO helpers")
        if helpers:
            _vpto.validate_target(self.platform, core_type)
            if _vpto_snapshot is None:
                _vpto_snapshot = _vpto.snapshot_sources(helpers)
        if core_type not in ("aic", "aiv"):
            raise ValueError(f"Unknown core_type: {core_type!r}; expected 'aic' or 'aiv'")
        simulation = self.platform.endswith("sim")
        if not simulation and pto_isa_root is None:
            raise ValueError("pto_isa_root is required for incore compilation")
        toolchain = self.sdk.gxx15 if simulation else self.sdk.ccec
        assert toolchain is not None, f"SDK did not provide an incore toolchain for {self.platform}"
        if simulation:
            # The generic GCC flags omit the simulator family and the loader's
            # required metadata. Consume the SDK's complete command description
            # while retaining PyPTO ownership of the actual compiler process.
            compile_flags = cast("list[str]", self.sdk.incore_compile_cache_token(core_type)["flags"])
            if pto_isa_root is None:
                from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

                pto_isa_root = ensure_pto_isa_root()
        else:
            compile_flags = toolchain.get_compile_flags(core_type=core_type)
        includes: list[str] = []
        if pto_isa_root is not None:
            includes.extend([str(Path(pto_isa_root) / "include"), str(Path(pto_isa_root) / "include/pto")])
        includes.extend(self.get_incore_include_dirs())
        if runtime_name is not None:
            includes.extend(self.get_kernel_include_dirs(runtime_name))
        includes.extend(extra_include_dirs or [])
        with tempfile.TemporaryDirectory(prefix="pypto-incore-", dir=build_dir) as directory:
            output = Path(directory).absolute() / ("kernel.so" if simulation else "kernel.o")
            cmd = [
                toolchain.cxx_path,
                *compile_flags,
                *(_vpto.CALLER_FLAGS if helpers or text_only else ()),
                # CCEC otherwise keeps a frame-pointer save around VF calls even
                # without spills. Still require compiler evidence of zero stack.
                *(("-fdata-sections", "-fstack-usage", "-fomit-frame-pointer") if text_only else ()),
                *self._sanitizer_flags(toolchain),
                *self._include_flags(includes),
                "-o",
                str(output),
                str(compiler_visible_path(source)),
            ]
            binary = self._run(cmd, output, "Incore")
            if simulation:
                return binary
            assert self.sdk.ccec is not None, "Device linking requires the CCEC toolchain"
            linked = output.with_suffix(".elf")
            helper_objects = (
                self._compile_vpto_helpers(helpers, Path(directory), _vpto_snapshot) if helpers else []
            )
            link_flags: list[str] = []
            if helpers or text_only:
                order = Path(directory) / "entry.order"
                order.write_text("\n".join(_vpto.LINK_SYMBOL_ORDER) + "\n")
                link_flags.append(f"--symbol-ordering-file={order}")
            if text_only:
                # CANN headers may define unused allocated data (g_tilingKey).
                # GC removes only unreachable sections. Any surviving data
                # remains an error: the persistent loader retains only .text.
                link_flags.append("--gc-sections")
            binary = self._run(
                [
                    self.sdk.ccec.linker_path,
                    "-e",
                    "kernel_entry",
                    *link_flags,
                    "-o",
                    str(linked),
                    str(output),
                    *(str(path) for path in helper_objects),
                ],
                linked,
                "Incore-link",
            )
            _device_image.validate_device_entry(binary)
            if helpers:
                _vpto.validate_device_image(binary)
            if text_only:
                functions = _device_image.validate_device_image(binary, entry_symbol="kernel_entry")
                _stack_usage.validate_stack_usage(output.with_suffix(".su"), functions)
            return binary

    def compile_ccu(self, source_path: str) -> bytes:
        """Build a native Host ELF using Simpler's complete CCU build metadata."""
        if self.platform != "a5":
            raise ValueError(f"Native CCU modules require a5, got {self.platform!r}")
        source = self._source_path(source_path)
        inputs = self.sdk.ccu_compile_cache_token()
        with tempfile.TemporaryDirectory(prefix="pypto-ccu-") as directory:
            root = Path(directory)
            (root / "ccu_manifest_identity.h").write_text(
                inputs["manifest_identity_header"], encoding="ascii"
            )
            output = root / "module.so"
            cmd = [
                inputs["cxx_path"],
                *inputs["compile_flags"],
                *self._include_flags([str(root), *inputs["include_dirs"]]),
                str(compiler_visible_path(source)),
                *inputs["link_flags"],
                *inputs["library_paths"],
                "-o",
                str(output),
            ]
            return self._run(cmd, output, "CCU")

    def _vpto_toolchain(self) -> tuple[str, str]:
        """Use the assembler selected by PyPTO and the SDK's matching Bisheng."""
        ptoas = find_ptoas_binary()
        if ptoas is None:
            raise FileNotFoundError("VPTO helpers require ptoas; set PTOAS_ROOT or add ptoas to PATH")
        # Discovery is relative to the caller's cwd; compiler subprocesses run
        # from the SDK project root. Preserve the selected launcher across both.
        ptoas = os.path.abspath(ptoas)
        check_ptoas_version(ptoas)
        assert self.sdk.ccec is not None
        bisheng = str(Path(self.sdk.ccec.cxx_path).with_name("bisheng"))
        return ptoas, bisheng

    def vpto_cache_identity(
        self, sources: tuple[str, ...], core_type: str, *, snapshot: tuple[str, ...] | None = None
    ) -> str:
        """Hash helper contents, lowering options, and the selected compiler versions.

        As with the JIT's PTOAS identity, published compiler versions are
        assumed immutable. Rebuilding a tool under an unchanged version is
        outside this version-based identity contract.
        """
        _vpto.validate_target(self.platform, core_type)
        snapshots = snapshot if snapshot is not None else _vpto.snapshot_sources(sources)
        effective_flags = tuple(_vpto.ptoas_flags_for_source(text) for text in snapshots)
        ptoas, bisheng = self._vpto_toolchain()
        result = subprocess.run(
            [bisheng, "--version"],
            env=toolchain_probe_env(),
            capture_output=True,
            text=True,
            check=True,
            timeout=self._timeout_s,
        )
        assert self.sdk.ccec is not None
        # Hash the exact snapshots later written to compiler inputs. Helpers are
        # self-contained PTO modules, with no C/C++ include closure.
        digest = hashlib.sha256()
        metadata = (
            # Cache identities include the native SIMT object/alias contract.
            "reg-auto-sync-v1",
            repr(effective_flags),
            "native-simt-object-abi-v1",
            repr(_vpto.SIMT_OBJECT_FLAGS),
            "bisheng-memory-attrs-v1",
            check_ptoas_version(ptoas),
            result.stdout,
            result.stderr,
            repr(_vpto.PTOAS_FLAGS),
            repr(_vpto.DEVICE_FLAGS),
            repr(_vpto.CALLER_FLAGS),
            repr(_vpto.LINK_SYMBOL_ORDER),
            repr(self.sdk.ccec.get_compile_flags(core_type=core_type)),
        )
        for item in metadata:
            payload = item.encode()
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
        for source, text in zip(sources, snapshots, strict=True):
            for payload in (Path(source).name.encode(), text.encode()):
                digest.update(len(payload).to_bytes(8, "little"))
                digest.update(payload)
        return digest.hexdigest()

    def _compile_vpto_helpers(
        self, sources: tuple[str, ...], directory: Path, snapshot: tuple[str, ...] | None = None
    ) -> list[Path]:
        """Keep all helper outputs in the caller's bounded, temporary build."""
        ptoas, bisheng = self._vpto_toolchain()
        objects: list[Path] = []
        snapshots = snapshot if snapshot is not None else _vpto.snapshot_sources(sources)
        for index, (source, source_text) in enumerate(zip(sources, snapshots, strict=True)):
            flags = _vpto.ptoas_flags_for_source(source_text)
            snapshot_path = directory / f"vpto_{index}.pto"
            snapshot_path.write_text(source_text)
            llvm = directory / f"vpto_{index}.ll"
            obj = directory / f"vpto_{index}.o"
            self._run([ptoas, str(snapshot_path), *flags, "-o", str(llvm)], llvm, "VPTO lowering")
            llvm_text = llvm.read_text()
            # LLVM text export does not link PTOAS's generated Bridge wrappers.
            # Reject all non-intrinsic declarations instead of producing an
            # object that only fails later or depends on hidden runtime support.
            declarations = re.findall(r'^declare\b[^\n]*?@(?:"([^"]+)"|([^\s(]+))', llvm_text, re.MULTILINE)
            unsupported = [
                quoted or plain for quoted, plain in declarations if not (quoted or plain).startswith("llvm.")
            ]
            if unsupported:
                raise ValueError(
                    f"VPTO helper {source} is not self-contained: {', '.join(unsupported)}. "
                    "Bridge operations and external helper dependencies are not supported."
                )
            if re.search(r"^define\s+[^@\n]*\bsimt_entry\b", llvm_text, re.MULTILINE) and re.search(
                r"\basm\s", llvm_text
            ):
                # Preserve the LLVM admission checks above, then let PTOAS own
                # native ABI lowering and VF_SIMT size validation/repair. A
                # relocatable link adds logical-name aliases without rewriting
                # instruction bytes. The final linked image is still validated.
                native = directory / f"vpto_{index}.native.o"
                object_flags = [flag for flag in flags if flag != "--emit-vpto-llvm-ir"]
                native_data = self._run(
                    [ptoas, str(snapshot_path), *object_flags, *_vpto.SIMT_OBJECT_FLAGS, "-o", str(native)],
                    native,
                    "SIMT native object compilation",
                    env={**os.environ, "ASCEND_HOME_PATH": str(Path(bisheng).parent.parent)},
                )
                aliases = _vpto.simt_object_aliases(llvm_text, native_data)
                assert self.sdk.ccec is not None
                self._run(
                    [self.sdk.ccec.linker_path, "-r", *aliases, str(native), "-o", str(obj)],
                    obj,
                    "SIMT logical export linking",
                )
                objects.append(obj)
                continue
            llvm.write_text(_vpto.prepare_llvm_for_bisheng(llvm_text))
            # PTOAS owns VF dependency analysis, including ordering through SSA
            # values. Do not re-infer synchronization from emitted LLVM text.
            command = [bisheng, *_vpto.DEVICE_FLAGS, "-c", "-x", "ir", str(llvm), "-o", str(obj)]
            self._run(command, obj, "VPTO device compilation")
            objects.append(obj)
        return objects

    def compile_orchestration(
        self,
        runtime_name: str,
        source_path: str,
        extra_include_dirs: list[str] | None = None,
        build_dir: str | None = None,
    ) -> bytes:
        """Compile orchestration and the SDK's helper sources into one shared library."""
        source = self._source_path(source_path)
        toolchain = self._orchestration_toolchain(runtime_name)
        includes, sources = self.get_orchestration_cache_inputs(runtime_name)
        # Every declared helper is required: a missing SDK source must fail the
        # build instead of silently yielding a library with unresolved helpers.
        helpers = [self._source_path(path) for path in sources]
        link_flags = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-Wl,--build-id=sha1"]
        if toolchain.is_host:
            link_flags.append("-pthread")
        with tempfile.TemporaryDirectory(prefix="pypto-orchestration-", dir=build_dir) as directory:
            output = Path(directory).absolute() / "orchestration.so"
            cmd = [
                toolchain.cxx_path,
                *toolchain.get_compile_flags(),
                *self._sanitizer_flags(toolchain),
                *link_flags,
                *(str(compiler_visible_path(path)) for path in helpers),
                *self._include_flags([*includes, *(extra_include_dirs or [])]),
                "-o",
                str(output),
                str(compiler_visible_path(source)),
            ]
            return self._run(cmd, output, "Orchestration")
