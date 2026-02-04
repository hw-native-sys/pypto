# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type stubs for backend module."""

from typing import Dict, List

from pypto import ir

class Mem:
    """Memory component."""

    def __init__(self, mem_type: ir.MemorySpace, mem_size: int, alignment: int) -> None: ...
    @property
    def mem_type(self) -> ir.MemorySpace: ...
    @property
    def mem_size(self) -> int: ...
    @property
    def alignment(self) -> int: ...

class Core:
    """Processing core."""

    def __init__(self, core_type: ir.CoreType, mems: List[Mem]) -> None: ...
    @property
    def core_type(self) -> ir.CoreType: ...
    @property
    def mems(self) -> List[Mem]: ...

class Cluster:
    """Cluster of cores."""

    @property
    def core_counts(self) -> Dict[Core, int]: ...
    def total_core_count(self) -> int: ...

class Die:
    """Die containing clusters."""

    @property
    def cluster_counts(self) -> Dict[Cluster, int]: ...
    def total_cluster_count(self) -> int: ...
    def total_core_count(self) -> int: ...

class SoC:
    """System on Chip."""

    @property
    def die_counts(self) -> Dict[Die, int]: ...
    def total_die_count(self) -> int: ...
    def total_cluster_count(self) -> int: ...
    def total_core_count(self) -> int: ...

class Backend:
    """Abstract backend base class."""

    def export_to_file(self, path: str) -> None: ...
    @staticmethod
    def import_from_file(path: str) -> "Backend": ...
    def find_mem_path(self, from_mem: ir.MemorySpace, to_mem: ir.MemorySpace) -> List[ir.MemorySpace]: ...
    def get_mem_size(self, mem_type: ir.MemorySpace) -> int: ...
    @property
    def soc(self) -> SoC: ...

class Backend910B(Backend):
    """910B backend implementation."""

    def __init__(self) -> None: ...
