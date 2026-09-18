# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static kernel descriptors for the pinned simpler ChipStorageTaskArgs ABI.

These records contain no device state. They do not expose or call runtime C
symbols; the Worker/native adapter owns that boundary in later integration.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# This is a PyPTO descriptor revision, not a version exported by simpler.
KERNEL_DESCRIPTOR_SCHEMA = 1
SIMPLER_KERNEL_REVISION = "cbafd5247109c8b5998fb7dd7a141db357d15461"
TENSOR_DTYPE_TAGS = {
    "fp32": 0,
    "fp16": 1,
    "int32": 2,
    "int16": 3,
    "int8": 4,
    "uint8": 5,
    "bfloat16": 6,
    "int64": 7,
    "uint64": 8,
    "uint16": 9,
    "uint32": 10,
    "bool": 11,
    "fp8e4m3fn": 12,
    "fp8e8m0": 13,
    "fp4": 14,
}
# The slot holds zero-extended object bytes, not int(float_value).
SCALAR_FORMATS = {
    "fp32": "f",
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "bool": "?",
    "index": "q",
}
TENSOR_DIRECTION_TAGS = {"In": 1, "Out": 2, "InOut": 3}
MAX_KERNEL_TENSORS = 256
MAX_KERNEL_SCALARS = 128
MAX_KERNEL_RANK = 5
# (platform, runtime) pairs whose native launch is validated for framework
# eager and graph calls. Binary descriptors may name other targets.
EAGER_KERNEL_TARGETS = frozenset({("a2a3", "tensormap_and_ringbuffer")})


@dataclass(frozen=True)
class KernelParameter:
    """A logical parameter; pool and pool index are derived from signature order."""

    name: str
    dtype: str
    direction: str
    shape: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Kernel parameter requires a nonempty name, got {self.name!r}")
        if self.direction not in ("In", "Out", "InOut"):
            raise ValueError(f"Invalid kernel direction for {self.name!r}: {self.direction!r}")
        if self.shape is None:
            if self.direction != "In" or self.dtype not in SCALAR_FORMATS:
                raise ValueError(f"Unsupported kernel scalar {self.name!r}: {self.dtype}/{self.direction}")
        else:
            if self.dtype not in TENSOR_DTYPE_TAGS:
                raise ValueError(f"Unsupported kernel tensor dtype for {self.name!r}: {self.dtype!r}")
            shape = tuple(self.shape)
            if not 1 <= len(shape) <= MAX_KERNEL_RANK or any(
                type(dim) is not int or not (dim == -1 or 0 < dim < 2**32) for dim in shape
            ):
                raise ValueError(f"Kernel tensor {self.name!r} requires rank 1..5 and positive u32/-1 dims")
            object.__setattr__(self, "shape", shape)


def validate_kernel_signature(
    parameters: Sequence[KernelParameter], return_aliases: Sequence[int]
) -> tuple[tuple[KernelParameter, ...], tuple[int, ...]]:
    """Check the target-independent pools and aliases shared by every kernel target."""
    params = tuple(parameters)
    if any(not isinstance(param, KernelParameter) for param in params):
        raise ValueError("Kernel ABI requires KernelParameter entries")
    if len({param.name for param in params}) != len(params):
        raise ValueError("Kernel parameter names must be unique")
    tensor_count = sum(param.shape is not None for param in params)
    if tensor_count > MAX_KERNEL_TENSORS or len(params) - tensor_count > MAX_KERNEL_SCALARS:
        raise ValueError("Kernel ABI exceeds simpler tensor/scalar argument capacity")
    aliases = tuple(return_aliases)
    for index in aliases:
        if type(index) is not int or not 0 <= index < len(params):
            raise ValueError(f"Invalid kernel return alias index {index!r}")
        if params[index].shape is None:
            raise ValueError(f"Kernel return alias {index} must name an external tensor")
    return params, aliases


@dataclass(frozen=True)
class KernelABI:
    """Compile-time identity and signature for one orchestration.

    Runtime/platform identify a binary target, not a claim that its kernel
    execution or capture is implemented. Return aliases are logical parameter
    indices, including repeated aliases. Only external tensors can be returned by this contract.
    """

    platform: str
    runtime: str
    parameters: tuple[KernelParameter, ...]
    return_aliases: tuple[int, ...] = ()
    simpler_revision: str = SIMPLER_KERNEL_REVISION

    def __post_init__(self) -> None:
        if self.simpler_revision != SIMPLER_KERNEL_REVISION:
            raise ValueError(f"Unsupported simpler kernel ABI revision {self.simpler_revision!r}; recompile")
        if self.platform not in ("a2a3", "a5"):
            raise ValueError(f"Unsupported kernel ABI platform {self.platform!r}")
        if self.runtime not in ("host_build_graph", "tensormap_and_ringbuffer"):
            raise ValueError(f"Unsupported kernel ABI runtime {self.runtime!r}")
        params, aliases = validate_kernel_signature(self.parameters, self.return_aliases)
        object.__setattr__(self, "parameters", params)
        object.__setattr__(self, "return_aliases", aliases)

    def binary_tag(self) -> bytes:
        """Identify this descriptor inside the generated orchestration binary."""
        encoded = json.dumps(self.record(), sort_keys=True, separators=(",", ":")).encode()
        return f"pypto-kernel-abi-v1:{hashlib.sha256(encoded).hexdigest()}\0".encode("ascii")

    def record(self) -> dict[str, Any]:
        """Return the canonical descriptor, including independently indexed pools."""
        params = []
        tensor_index = scalar_index = 0
        for param in self.parameters:
            entry: dict[str, Any] = {
                "name": param.name,
                "dtype": param.dtype,
                "direction": param.direction,
                "shape": list(param.shape) if param.shape is not None else None,
            }
            if param.shape is None:
                entry.update(kind="scalar", scalar_index=scalar_index, encoding=SCALAR_FORMATS[param.dtype])
                scalar_index += 1
            else:
                entry.update(
                    kind="tensor",
                    tensor_index=tensor_index,
                    dtype_tag=TENSOR_DTYPE_TAGS[param.dtype],
                    direction_tag=TENSOR_DIRECTION_TAGS[param.direction],
                )
                tensor_index += 1
            params.append(entry)
        return {
            "schema": KERNEL_DESCRIPTOR_SCHEMA,
            "simpler_revision": self.simpler_revision,
            "argument_abi": "ChipStorageTaskArgs",
            "tensor_layout": "positive_element_strides",
            "scalar_storage": "little_endian_zero_extended_u64",
            "platform": self.platform,
            "runtime": self.runtime,
            "params": params,
            "return_aliases": list(self.return_aliases),
        }

    @classmethod
    def from_record(cls, value: object) -> "KernelABI":
        """Reject missing, noncanonical or incompatible descriptors before loading."""
        if not isinstance(value, dict):
            raise ValueError("Kernel ABI must be an explicit descriptor object; recompile")
        try:
            params = tuple(
                KernelParameter(p["name"], p["dtype"], p["direction"], p["shape"]) for p in value["params"]
            )
            result = cls(
                value["platform"],
                value["runtime"],
                params,
                value["return_aliases"],
                value["simpler_revision"],
            )
            # JSON equality alone treats True as 1 and 1.0 as 1. Require exact
            # canonical JSON types as well as the derived tags/pool indices.
            if not _same_record(value, result.record()):
                raise ValueError("Kernel ABI descriptor differs from the supported canonical contract")
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid kernel ABI descriptor: {exc}; recompile") from exc

    def require_compatible(self, expected: "KernelABI") -> None:
        """Reject an ABI/signature mismatch without trying another execution mode."""
        if self != expected:
            raise ValueError("Kernel artifact ABI/signature does not match the requested contract; recompile")


def _same_record(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same_record(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_record(a, b) for a, b in zip(actual, expected))
    return actual == expected
