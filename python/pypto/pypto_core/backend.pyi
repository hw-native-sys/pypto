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

class BackendType:
    """Backend type for passes and codegen (CCE or PTO)."""

    CCE: "BackendType"
    PTO: "BackendType"

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

    def get_type_name(self) -> str: ...
    def export_to_file(self, path: str) -> None: ...
    @staticmethod
    def import_from_file(path: str) -> "Backend": ...
    def find_mem_path(self, from_mem: ir.MemorySpace, to_mem: ir.MemorySpace) -> List[ir.MemorySpace]: ...
    def get_mem_size(self, mem_type: ir.MemorySpace) -> int: ...
    @property
    def soc(self) -> SoC: ...

class Backend910B_CCE(Backend):
    """910B CCE backend implementation (singleton)."""

    @staticmethod
    def instance() -> "Backend910B_CCE":
        """Get singleton instance of 910B CCE backend."""
        ...

class Backend910B_PTO(Backend):
    """910B PTO backend implementation (singleton)."""

    @staticmethod
    def instance() -> "Backend910B_PTO":
        """Get singleton instance of 910B PTO backend."""
        ...

def set_backend_type(backend_type: BackendType) -> None:
    """
    Set the global backend type.

    Must be called before any backend operations. Can be called multiple times
    with the same type (idempotent), but will raise an error if attempting to
    change to a different type.

    Args:
        backend_type: The backend type to use (CCE or PTO)

    Raises:
        ValueError: If attempting to change an already-set backend type
    """
    ...

def get_backend_type() -> BackendType:
    """
    Get the configured backend type.

    Returns:
        The configured backend type

    Raises:
        ValueError: If backend type has not been configured
    """
    ...

def is_backend_configured() -> bool:
    """
    Check if backend type has been configured.

    Returns:
        True if set_backend_type() has been called, False otherwise
    """
    ...

def reset_for_testing() -> None:
    """
    Reset backend configuration (for testing only).

    WARNING: This function should ONLY be used in tests to reset the
    backend configuration between test cases. Do NOT use in production code.
    """
    ...
