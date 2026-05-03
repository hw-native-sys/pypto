# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type stubs for backend module."""

from pypto import ir

class BackendType:
    """Backend type for passes and codegen."""

    Ascend910B: BackendType
    Ascend950: BackendType

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

    def __init__(self, core_type: ir.CoreType, mems: list[Mem]) -> None: ...
    @property
    def core_type(self) -> ir.CoreType: ...
    @property
    def mems(self) -> list[Mem]: ...

class Cluster:
    """Cluster of cores."""

    @property
    def core_counts(self) -> dict[Core, int]: ...
    def total_core_count(self) -> int: ...

class Die:
    """Die containing clusters."""

    @property
    def cluster_counts(self) -> dict[Cluster, int]: ...
    def total_cluster_count(self) -> int: ...
    def total_core_count(self) -> int: ...

class SoC:
    """System on Chip."""

    @property
    def die_counts(self) -> dict[Die, int]: ...
    def total_die_count(self) -> int: ...
    def total_cluster_count(self) -> int: ...
    def total_core_count(self) -> int: ...

class BackendHandler:
    """Per-backend behaviour dispatch interface.

    Encapsulates every behavioural difference between backends. Passes,
    codegen, and the Python runtime consume this interface rather than
    branching on :class:`BackendType` directly.
    """

    def get_pto_target_arch(self) -> str: ...
    def get_launch_spec_core_count_method(self) -> str: ...
    def get_default_sim_platform(self) -> str: ...
    def get_extra_ptoas_flags(self) -> list[str]: ...
    def requires_gm_pipe_buffer(self) -> bool: ...
    def requires_split_load_tpop_workaround(self) -> bool: ...
    def requires_vto_c_fractal_adapt(self) -> bool: ...
    def requires_runtime_subblock_bridge(self) -> bool: ...
    def requires_no_split_dual_aiv_dispatch(self) -> bool: ...
    def get_gm_access_granularity_bytes(self) -> int: ...
    def get_l2_cache_line_bytes(self) -> int: ...
    def get_recommended_innermost_dim_bytes(self) -> int: ...
    def get_l0a_capacity_bytes(self) -> int: ...
    def get_l0b_capacity_bytes(self) -> int: ...
    def get_l0c_capacity_bytes(self) -> int: ...
    def get_l0_fractal_alignment(self) -> int: ...
    def get_min_l0_tile_dim(self) -> int: ...

class Backend:
    """Abstract backend base class."""

    def get_type_name(self) -> str: ...
    def get_handler(self) -> BackendHandler: ...
    def export_to_file(self, path: str) -> None: ...
    @staticmethod
    def import_from_file(path: str) -> Backend: ...
    def find_mem_path(self, from_mem: ir.MemorySpace, to_mem: ir.MemorySpace) -> list[ir.MemorySpace]: ...
    def get_mem_size(self, mem_type: ir.MemorySpace) -> int: ...
    @property
    def soc(self) -> SoC: ...

class Backend910B(Backend):
    """910B backend implementation (singleton)."""

    @staticmethod
    def instance() -> Backend910B:
        """Get singleton instance of 910B backend."""
        ...

class Backend950(Backend):
    """950 PTO backend implementation (singleton)."""

    @staticmethod
    def instance() -> Backend950:
        """Get singleton instance of 950 backend."""
        ...

def set_backend_type(backend_type: BackendType) -> None:
    """
    Set the global backend type.

    Must be called before any backend operations. Can be called multiple times
    with the same type (idempotent), but will raise an error if attempting to
    change to a different type.

    Args:
        backend_type: The backend type to use

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

def get_backend_instance(backend_type: BackendType) -> Backend:
    """
    Get the singleton :class:`Backend` instance for a specific
    :class:`BackendType` regardless of the global configuration.

    This is useful for callers that already know which backend they want
    (for example, :class:`RunOptions.backend_type` may differ from the
    globally-configured type when running multiple backends in sequence).
    """
    ...

def get_handler() -> BackendHandler:
    """
    Get the :class:`BackendHandler` for the currently configured backend.

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
