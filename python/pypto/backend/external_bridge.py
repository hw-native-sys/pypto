# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Internal bridge boundary; concrete runtime adapters belong to task 34.4.

A bridge must check binding-dependent constraints (alignment, strides, offset,
aliasing and launch occupancy) on every execution, including artifact hits.
Validation of a PreparedKernel alone never authorizes execution.
"""

from dataclasses import dataclass
from typing import Protocol

from pypto.jit.external.contracts import Access, CompileTarget, Coverage, PreparedKernel, SourceArtifact
from pypto.jit.external.validation import ContractError, validate_prepared


@dataclass(frozen=True)
class RuntimeTarget:
    runtime: str
    version: str
    compile_target: CompileTarget
    supported_abis: frozenset[tuple[str, int]]
    capabilities: frozenset[str]


@dataclass(frozen=True)
class ExternalTask:
    """Bridge-owned source plus provenance; no per-invocation addresses.

    Task 34.4 supplies runtime-specific entry generation and binding checks.
    The source bundle uses the existing external-kernel entry convention.
    """

    source: SourceArtifact
    prepared: PreparedKernel
    target: RuntimeTarget
    bridge_version: str


class RuntimeBridge(Protocol):
    def lower(self, prepared: PreparedKernel, target: RuntimeTarget) -> ExternalTask:
        """Validate target/ABI capabilities and produce runtime-compatible source."""
        ...


def validate_target(prepared: PreparedKernel, target: RuntimeTarget) -> None:
    """Reject incompatible ABI/launch requirements before generating a wrapper."""
    validate_prepared(prepared)
    if target.compile_target != prepared.request.target:
        raise ContractError("bridge: compile target differs from the prepared target")
    abi = prepared.native_abi
    if (abi.name, abi.version) not in target.supported_abis:
        raise ContractError(f"bridge: unsupported native ABI {abi.name!r} version {abi.version}")
    execution = prepared.execution
    required = set(prepared.contract.required_capabilities)
    required.update(f"core:{core.value}" for core in execution.cores)
    required.add(f"aiv_lanes:{execution.aiv_lanes}")
    required.add(f"completion:{execution.completion.value}")
    for name in ("full_residency", "sync_start", "logical_launch_context"):
        if getattr(execution, name):
            required.add(name)
    if prepared.contract.public_returns:
        required.add("parameter_returns")
    if prepared.contract.writebacks:
        required.add("tensor_writeback")
    if any(
        effect.access in (Access.WRITE, Access.READ_WRITE)
        and effect.coverage in (Coverage.PARTIAL, Coverage.UNKNOWN)
        for effect in prepared.contract.effects
    ):
        required.add("partial_write")
    if prepared.contract.alias_permissions:
        required.add("aliasing")
    missing = required - target.capabilities
    if missing:
        raise ContractError(f"bridge: unsupported execution/semantic capabilities {sorted(missing)}")
