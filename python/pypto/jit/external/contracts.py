# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Versioned, immutable hand-off between providers, JIT, and runtime bridges.

Indices always refer to logical parameter order. A provider reports semantics;
a bridge interprets the native ABI. Neither owns JIT AST state or runtime data.
These are internal interfaces, not a stable third-party registration API.
"""

from dataclasses import dataclass
from enum import Enum

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout

SCHEMA_VERSION = 1
Dimension = int | str
Constant = bool | int | float | str


class Access(Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class Coverage(Enum):
    NONE = "none"
    FULL = "full"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class Core(Enum):
    AIC = "aic"
    AIV = "aiv"


class Completion(Enum):
    # Completion includes visibility of writes, not just issuing instructions.
    ENTRY_RETURN = "entry_return_visible"


class ABIValue(Enum):
    SCALAR = "scalar_bits"
    DATA_POINTER = "tensor_data_pointer"
    DESCRIPTOR = "tensor_descriptor"
    DIMENSION = "tensor_dimension"
    STRIDE = "tensor_stride"
    OFFSET = "tensor_offset"


class Unit(Enum):
    ELEMENTS = "elements"
    BYTES = "bytes"


@dataclass(frozen=True)
class TensorSpec:
    """Logical tensor constraints; symbols are request-local dimension names.

    Strides and offset use elements. An empty stride tuple means unconstrained,
    never implicitly contiguous. None offset means a runtime-bound view offset.
    """

    shape: tuple[Dimension, ...]
    dtype: DataType
    strides: tuple[Dimension, ...] = ()
    offset: Dimension | None = None
    layout: TensorLayout = TensorLayout.ND
    alignment: int = 1  # Minimum data-pointer alignment, in bytes.


@dataclass(frozen=True)
class ScalarSpec:
    dtype: DataType


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type: TensorSpec | ScalarSpec


@dataclass(frozen=True)
class CompileTarget:
    architecture: str
    compiler: str
    compiler_version: str


@dataclass(frozen=True)
class KernelRequest:
    parameters: tuple[ParameterSpec, ...]
    target: CompileTarget
    constants: tuple[tuple[str, Constant], ...] = ()
    options: tuple[tuple[str, Constant], ...] = ()
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class TensorEffect:
    """Coverage is relative to the argument view, not its allocation root.

    Partial/unknown writes must preserve untouched bytes. They retain incoming
    dependencies even if the device code never reads the bytes it overwrites.
    """

    parameter: int
    access: Access
    coverage: Coverage = Coverage.NONE
    preserves_unwritten: bool = True


@dataclass(frozen=True)
class ReturnBinding:
    """One public tensor return aliases this complete logical argument view."""

    parameter: int


@dataclass(frozen=True)
class WritebackBinding:
    """The new logical version of a written argument uses that argument's storage."""

    parameter: int


@dataclass(frozen=True)
class AliasPermission:
    """Provider guarantees safe overlap for this pair, including full aliasing.

    Unlisted distinct argument pairs require disjoint storage when either is
    written. The bridge must prove/check this on every invocation, including
    cache hits. Scheduling alone cannot establish kernel-internal alias safety.
    """

    first: int
    second: int


@dataclass(frozen=True)
class KernelContract:
    parameters: tuple[ParameterSpec, ...]
    effects: tuple[TensorEffect, ...]
    public_returns: tuple[ReturnBinding, ...] = ()
    writebacks: tuple[WritebackBinding, ...] = ()
    alias_permissions: tuple[AliasPermission, ...] = ()
    required_capabilities: frozenset[str] = frozenset()
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class ABIParameter:
    """One physical argument in native order; repeated logical indices are legal.

    SCALAR copies bits without numeric conversion. DESCRIPTOR layout is defined
    by NativeABI's name/version. Metadata units are explicit; DATA_POINTER is
    the allocation base, with OFFSET applied exactly once by the bridge.
    """

    parameter: int
    value: ABIValue
    size_bytes: int
    alignment: int
    axis: int | None = None
    unit: Unit = Unit.ELEMENTS


@dataclass(frozen=True)
class NativeABI:
    name: str
    version: int
    parameters: tuple[ABIParameter, ...]
    byte_order: str = "little"


@dataclass(frozen=True)
class ExecutionRequirements:
    """Core members and completion guarantees for a single schedulable task.

    aiv_lanes is zero for AIC-only, one for standalone AIV, and one or two for
    mixed. Full residency requires a synchronized start. Logical launch context
    must be supplied by the bridge rather than inferred from physical core IDs.
    """

    cores: tuple[Core, ...]
    aiv_lanes: int
    completion: Completion
    full_residency: bool = False
    sync_start: bool = False
    logical_launch_context: bool = False


@dataclass(frozen=True)
class SourceFile:
    """Owned immutable payload, independent of provider directories or cache leases."""

    path: str  # Bundle-relative path, including required headers.
    sha256: str
    content: bytes


@dataclass(frozen=True)
class SourceArtifact:
    """Self-contained source bundle retained for the lifetime of this record.

    Providers snapshot all source/header bytes before returning. Bridges can
    materialize these bytes without a provider directory or artifact-store
    lookup. Persistent publication still belongs to the existing artifact store.
    """

    files: tuple[SourceFile, ...]
    entry: str
    symbol: str
    include_dirs: tuple[str, ...] = ()  # Ordered bundle-relative search paths.


@dataclass(frozen=True)
class PreparationIdentity:
    provider: str
    provider_version: str
    adapter_version: str
    kernel_digest: str  # Source plus referenced constants/dependencies.
    complete: bool  # False forbids reusing provider preparation by this identity.


@dataclass(frozen=True)
class PreparedKernel:
    request: KernelRequest
    contract: KernelContract
    artifact: SourceArtifact
    native_abi: NativeABI
    execution: ExecutionRequirements
    identity: PreparationIdentity
