# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Consumed surface of ``simpler_setup.toolchain``."""

from typing import TypedDict

class _CcuCacheInputs(TypedDict):
    cxx_path: str
    compile_flags: tuple[str, ...]
    include_dirs: tuple[str, ...]
    link_flags: tuple[str, ...]
    library_paths: tuple[str, ...]
    host_target: str
    pto_isa_commit: str
    manifest_identity_header: str
    dependency_digests: tuple[tuple[str, str], ...]

class Toolchain:
    is_host: bool
    cxx_path: str
    linker_path: str

    def get_compile_flags(self, **kwargs: object) -> list[str]: ...
    def get_cmake_args(self) -> list[str]: ...

class GxxToolchain(Toolchain):
    def __init__(self, prefer_g15: bool = ...) -> None: ...

class CCUToolchain(Toolchain):
    host_toolchain: GxxToolchain

__all__ = ["GxxToolchain", "Toolchain"]
