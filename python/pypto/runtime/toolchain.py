# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiler toolchain definitions for kernel and orchestration compilation.

Defines toolchain wrappers for:
- CCEC: Ascend AICore compiler (hardware)
- g++-15: Host simulation kernels
- g++: Host orchestration shared libraries
- aarch64-g++: Cross-compilation for device orchestration
"""

from __future__ import annotations

import os
import subprocess
from enum import IntEnum

from . import env_manager


# Must match compile_strategy.h
class ToolchainType(IntEnum):
    """Toolchain types matching the C enum in compile_strategy.h."""

    CCEC = 0  # ccec (Ascend AICore compiler)
    HOST_GXX_15 = 1  # g++-15 (host, simulation kernels)
    HOST_GXX = 2  # g++ (host, orchestration .so)
    AARCH64_GXX = 3  # aarch64-target-linux-gnu-g++ (cross-compile)


def _is_gcc(cxx_path: str) -> bool:
    """Return True if *cxx_path* is a real GCC (not clang masquerading as g++)."""
    try:
        out = subprocess.run(
            [cxx_path, "--version"], check=False, capture_output=True, text=True, timeout=5
        ).stdout.lower()
        return "clang" not in out
    except (OSError, subprocess.SubprocessError):
        return False


class Toolchain:
    """Base class for all compile toolchains.

    A Toolchain represents a compiler identity: which compiler binary to use,
    what flags to pass, and what CMake -D arguments to generate.

    The Ascend SDK path is managed by env_manager. Call
    env_manager.ensure("ASCEND_HOME_PATH") before creating toolchains that
    need the Ascend SDK (CCECToolchain, Aarch64GxxToolchain, GxxToolchain
    with Ascend includes).
    """

    cxx_path: str

    def __init__(self):
        self.ascend_home_path = env_manager.get("ASCEND_HOME_PATH")

    def get_compile_flags(self, **kwargs) -> list[str]:
        """Return base compiler flags for direct invocation."""
        raise NotImplementedError

    def get_cmake_args(self) -> list[str]:
        """Return compiler-specific CMake -D arguments."""
        raise NotImplementedError


class CCECToolchain(Toolchain):
    """Ascend ccec compiler for AICore kernels."""

    def __init__(self, platform: str = "a2a3"):
        super().__init__()
        self.platform = platform

        if self.ascend_home_path is None:
            raise RuntimeError("ASCEND_HOME_PATH is required for CCEC toolchain")

        self.cxx_path = os.path.join(self.ascend_home_path, "bin", "ccec")
        self.linker_path = os.path.join(self.ascend_home_path, "bin", "ld.lld")

        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(f"ccec compiler not found: {self.cxx_path}")
        if not os.path.isfile(self.linker_path):
            raise FileNotFoundError(f"ccec linker not found: {self.linker_path}")

    def get_compile_flags(self, core_type: str = "aiv", **kwargs) -> list[str]:
        # A5 uses dav-c310 architecture, A2A3 uses dav-c220
        if self.platform in ("a5", "a5sim"):
            arch = "dav-c310-vec" if core_type == "aiv" else "dav-c310-cube"
        elif self.platform in ("a2a3", "a2a3sim"):
            arch = "dav-c220-vec" if core_type == "aiv" else "dav-c220-cube"
        else:
            raise ValueError(f"Unknown platform: {self.platform}. Supported: a2a3, a2a3sim, a5, a5sim")

        return [
            "-c",
            "-O3",
            "-g",
            "-x",
            "cce",
            "-Wall",
            "-std=c++17",
            "--cce-aicore-only",
            f"--cce-aicore-arch={arch}",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=false",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            "-DMEMORY_BASE",
        ]

    def get_cmake_args(self) -> list[str]:
        return [
            f"-DBISHENG_CC={self.cxx_path}",
            f"-DBISHENG_LD={self.linker_path}",
        ]


class Gxx15Toolchain(Toolchain):
    """g++-15 compiler for simulation kernels."""

    def __init__(self):
        super().__init__()
        self.cxx_path = "g++-15"

    def get_compile_flags(self, core_type: str = "", **kwargs) -> list[str]:
        flags = [
            "-shared",
            "-O2",
            "-fPIC",
            "-std=c++23",
            "-fpermissive",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            "-D__CPU_SIM",
            "-DPTO_CPU_MAX_THREADS=1",
            "-DNDEBUG",
        ]
        # g++ does not define __DAV_VEC__/__DAV_CUBE__ like ccec does,
        # so we must add them explicitly based on core_type.
        if core_type == "aiv":
            flags.append("-D__DAV_VEC__")
        elif core_type == "aic":
            flags.append("-D__DAV_CUBE__")
        return flags

    def get_cmake_args(self) -> list[str]:
        cc = os.environ.get("CC", "gcc")
        cxx = os.environ.get("CXX", "g++")
        return [
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        ]


class GxxToolchain(Toolchain):
    """g++ compiler for host compilation."""

    def __init__(self):
        super().__init__()
        self.cxx_path = "g++"
        self._gcc = _is_gcc(self.cxx_path)

    def get_compile_flags(self, **kwargs) -> list[str]:
        flags = ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]
        # -fno-gnu-unique: prevent STB_GNU_UNIQUE binding so dlclose actually
        # unloads the SO.  GCC-only; clang does not produce STB_GNU_UNIQUE.
        if self._gcc:
            flags.append("-fno-gnu-unique")
        return flags

    def get_cmake_args(self) -> list[str]:
        cc = os.environ.get("CC", "gcc")
        cxx = os.environ.get("CXX", "g++")
        args = [
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        ]
        if self.ascend_home_path:
            args.append(f"-DASCEND_HOME_PATH={self.ascend_home_path}")
        return args


class Aarch64GxxToolchain(Toolchain):
    """aarch64 cross-compiler for device code."""

    def __init__(self):
        super().__init__()

        if self.ascend_home_path is None:
            raise RuntimeError("ASCEND_HOME_PATH is required for aarch64 toolchain")

        self.cxx_path = os.path.join(
            self.ascend_home_path,
            "tools",
            "hcc",
            "bin",
            "aarch64-target-linux-gnu-g++",
        )
        self.cc_path = os.path.join(
            self.ascend_home_path,
            "tools",
            "hcc",
            "bin",
            "aarch64-target-linux-gnu-gcc",
        )
        if not os.path.isfile(self.cc_path):
            raise FileNotFoundError(f"aarch64 C compiler not found: {self.cc_path}")
        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(f"aarch64 C++ compiler not found: {self.cxx_path}")
        self._gcc = _is_gcc(self.cxx_path)

    def get_compile_flags(self, **kwargs) -> list[str]:
        flags = ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]
        # -fno-gnu-unique: prevent STB_GNU_UNIQUE binding so dlclose actually unloads the SO.
        if self._gcc:
            flags.append("-fno-gnu-unique")
        return flags

    def get_cmake_args(self) -> list[str]:
        return [
            f"-DCMAKE_C_COMPILER={self.cc_path}",
            f"-DCMAKE_CXX_COMPILER={self.cxx_path}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
        ]
