# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Consumed surface of ``simpler_setup.kernel_compiler``."""

from pathlib import Path

from .toolchain import GxxToolchain, Toolchain

class KernelCompiler:
    def __init__(self, platform: str = ...) -> None: ...

    platform: str
    project_root: Path
    platform_dir: Path
    ccec: Toolchain | None
    aarch64: Toolchain | None
    host_gxx: GxxToolchain
    gxx15: Toolchain

    def get_incore_include_dirs(self) -> list[str]: ...
    def get_orchestration_cache_inputs(self, runtime_name: str) -> tuple[list[str], list[str]]: ...
    def _orchestration_toolchain(self, runtime_name: str) -> Toolchain: ...

__all__ = ["KernelCompiler"]
