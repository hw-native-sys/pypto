# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiler-to-loader contract for selectable distributed weight storage."""

import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path

import torch

from pypto._function_attrs import AUTO_LAYOUT_HOST_INPUTS_ATTR
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Function, ParamDirection, Program, Role, TensorType, level_to_linqu_level

from ._artifact_metadata import _is_obj_list, _is_str_dict
from .param_info import (
    _extract_func_param_infos,
    _is_int_list,
    _matches_torch_dtype,
    _param_info_to_dict,
    _ParamInfo,
    _reorder_nd_nz,
    _to_torch_dtype,
    block_nz_shape,
)

_HOST_PATH = "orchestration/host_orch.py"
_MARKER = b"# pypto_distributed_input_layouts: "


@dataclass(frozen=True)
class DistributedInputLayout:
    """Physical layout of an offered read-only HOST parameter.

    A rank-2 input is a complete matrix for a single-device program. In a
    higher-rank input, the first axis selects a rank and an optional second
    axis selects a layer. The final two axes specify the rows and columns.
    HOST transport preserves the logical shape. The physical shape describes
    the contiguous storage under the device kernel's ND or NZ indexing.
    """

    index: int
    name: str
    dtype: DataType
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    layout: str

    @property
    def _is_single_matrix(self) -> bool:
        return len(self.logical_shape) == 2

    def _record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "dtype": str(self.dtype),
            "logical_shape": list(self.logical_shape),
            "physical_shape": list(self.physical_shape),
            "layout": self.layout,
        }

    def _validate_tensor(self, tensor: torch.Tensor) -> None:
        if (
            tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.shape != self.logical_shape
            or not _matches_torch_dtype(self.dtype, tensor.dtype)
        ):
            raise ValueError(
                f"Distributed input {self.name!r} expects a dense CPU tensor with "
                f"dtype {self.dtype} and logical shape {self.logical_shape}; got "
                f"{tensor.device} {tensor.layout} {tensor.dtype} {tuple(tensor.shape)}"
            )

    def _materialize(self, value: torch.Tensor, source_layout: str) -> torch.Tensor:
        self._validate_tensor(value)
        if source_layout not in ("ND", "NZ"):
            raise ValueError(
                f"Distributed input {self.name!r} has unsupported source layout {source_layout!r}"
            )
        if source_layout == "NZ":
            # Packed bytes must already have canonical order and no padding.
            if not value.is_contiguous():
                raise ValueError(f"Distributed packed input {self.name!r} must be contiguous")
            block_nz_shape(self.logical_shape, self.dtype)
        return _reorder_nd_nz(value.detach(), source_layout, self.layout)


@dataclass(frozen=True, eq=False)
class PreparedDistributedInput:
    """Snapshot of one immutable weight version, prepared for a distributed input slot.

    Obtain a snapshot using ``compiled.prepare_input(...)``. To change the
    weight, prepare a new snapshot from its logical tensor or its saved packed
    representation. Keep a snapshot alive for as long as it is used in dispatches.
    """

    _layout: DistributedInputLayout
    _tensor: torch.Tensor

    def _for_layout(self, layout: DistributedInputLayout) -> torch.Tensor:
        if self._layout != layout:
            raise ValueError(f"Prepared distributed input does not match the layout of {layout.name!r}")
        layout._validate_tensor(self._tensor)
        return self._tensor


def _host_entries(program: Program) -> list[Function]:
    return [
        f
        for f in program.functions.values()
        if f.level is not None and level_to_linqu_level(f.level) >= 3 and f.role == Role.Orchestrator
    ]


def _make_layout(index: int, name: str, param: _ParamInfo, layout: str) -> DistributedInputLayout:
    shape = param.shape
    if (
        param.direction != ParamDirection.In
        or param.layout != "ND"
        or shape is None
        or len(shape) not in (2, 3, 4)
        or any(dim <= 0 for dim in shape)
        or _to_torch_dtype(param.dtype) is None
        or param.dtype.get_bit() % 8
        or prod(shape) * (param.dtype.get_bit() // 8) > sys.maxsize
    ):
        raise ValueError(
            f"Distributed input layout for {name!r} requires a static dense read-only "
            f"[rows, cols] or [rank, (layer,) rows, cols] tensor; got {param}"
        )
    if layout not in ("ND", "NZ"):
        raise ValueError(f"Distributed input layout for {name!r} is unsupported ({layout!r})")
    logical = tuple(shape)
    physical = logical if layout == "ND" else tuple(block_nz_shape(logical, param.dtype))
    return DistributedInputLayout(index, name, param.dtype, logical, physical, layout)


def _validate_layout_topology(layouts: Mapping[int, DistributedInputLayout], world_size: int) -> None:
    for layout in layouts.values():
        if layout._is_single_matrix:
            if world_size != 1:
                raise ValueError(
                    f"Rank-2 distributed input {layout.name!r} requires exactly one device, got {world_size}"
                )
        elif layout.logical_shape[0] != world_size:
            raise ValueError(
                f"Distributed input layout for {layout.name!r} requires its rank extent "
                f"({layout.logical_shape[0]}) to match the configured device count ({world_size})"
            )


def _validate_input_layouts(
    source: Program, lowered: Program, names: Sequence[str], world_size: int | None
) -> dict[int, DistributedInputLayout]:
    """Resolve public names to slots and check selection against the logical HOST ABI."""
    original_entries = _host_entries(source)
    lowered_entries = _host_entries(lowered)
    if any(AUTO_LAYOUT_HOST_INPUTS_ATTR in f.attrs for f in original_entries):
        raise ValueError("Distributed input layout requires source IR without a preexisting layout decision")
    if not names:
        if any(AUTO_LAYOUT_HOST_INPUTS_ATTR in f.attrs for f in lowered_entries):
            raise ValueError("Distributed input layout selected without a loader policy")
        return {}
    if world_size is None or world_size <= 0:
        raise ValueError("Distributed input layout requires a known positive compiler rank count")
    if len(original_entries) != 1 or len(lowered_entries) != 1:
        raise ValueError("Distributed input layout requires one explicit HOST entry")
    original = original_entries[0]
    host = lowered_entries[0]
    if host.name != original.name:
        raise ValueError("Distributed input layout requires the lowered HOST entry to preserve its identity")
    original_params, _, _ = _extract_func_param_infos(original)
    lowered_params, _, _ = _extract_func_param_infos(host)
    if len(original_params) != len(lowered_params):
        raise ValueError("Distributed input layout must preserve the complete HOST transport signature")
    for before, after in zip(original_params, lowered_params, strict=True):
        # Lowering can discover writes to an ordinary input, e.g.
        # a KV cache updated by a nested kernel. Its physical ABI stays fixed.
        inferred_write = (
            before.name not in names
            and before.direction == ParamDirection.In
            and after.direction in (ParamDirection.Out, ParamDirection.InOut)
        )
        if (before.dtype, before.shape, before.layout) != (after.dtype, after.shape, after.layout) or (
            before.direction != after.direction and not inferred_write
        ):
            raise ValueError(
                f"Distributed input layout must preserve the HOST transport signature for {before.name!r}"
            )
    by_name = {param.name: i for i, param in enumerate(original_params)}
    if missing := set(names) - by_name.keys():
        raise ValueError(f"Distributed input layout policy has unknown HOST inputs {sorted(missing)}")
    expected = {by_name[name] for name in names}
    raw = host.attrs.get(AUTO_LAYOUT_HOST_INPUTS_ATTR)
    if not isinstance(raw, str) or not re.fullmatch(r"v1(?:;(?:0|[1-9][0-9]*)=(?:ND|NZ|UNSUPPORTED))+", raw):
        raise ValueError("Distributed input layout requires a valid native decision for each offered slot")
    pairs = [(int(index), layout) for index, layout in (entry.split("=") for entry in raw.split(";")[1:])]
    if [index for index, _ in pairs] != sorted(expected):
        raise ValueError("Distributed input layout decisions do not match the offered HOST slots")
    result: dict[int, DistributedInputLayout] = {}
    for index, target in pairs:
        param = original_params[index]
        if not isinstance(original.params[index].type, TensorType):
            raise ValueError(f"Distributed input layout policy expected a dense tensor for {param.name!r}")
        result[index] = _make_layout(index, param.name, lowered_params[index], target)
    _validate_layout_topology(result, world_size)
    return result


def _layout_records(layouts: Mapping[int, DistributedInputLayout]) -> list[dict[str, object]]:
    return [layout._record() for layout in layouts.values()]


def _read_input_layouts(records: object, params: Sequence[_ParamInfo]) -> dict[int, DistributedInputLayout]:
    if not _is_obj_list(records):
        raise ValueError("input_layouts must be an array")
    layouts: dict[int, DistributedInputLayout] = {}
    names: set[str] = set()
    for entry in records:
        if not _is_str_dict(entry):
            raise ValueError("input_layouts entries must be objects")
        index = entry.get("index")
        name = entry.get("name")
        target = entry.get("layout")
        if (
            type(index) is not int
            or not 0 <= index < len(params)
            or index in layouts
            or not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(target, str)
        ):
            raise ValueError(f"Invalid distributed input layout identity {entry!r}")
        for key in ("logical_shape", "physical_shape"):
            shape = entry.get(key)
            if not _is_int_list(shape):
                raise ValueError(f"Distributed input layout for {name!r} has invalid {key}")
        layout = _make_layout(index, name, params[index], target)
        if layout._record() != entry:
            raise ValueError(f"Distributed input layout for {name!r} does not match its HOST signature")
        names.add(name)
        layouts[index] = layout
    if list(layouts) != sorted(layouts):
        raise ValueError("input_layouts must be ordered by HOST slot")
    return layouts


def _artifact_marker(
    params: Sequence[_ParamInfo], layouts: Mapping[int, DistributedInputLayout], world_size: int
) -> bytes:
    signature = {
        "params": [_param_info_to_dict(info) for info in params],
        "inputs": _layout_records(layouts),
        "world_size": world_size,
    }
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return _MARKER + hashlib.sha256(canonical).hexdigest().encode() + b"\n"


def _check_artifact_marker(
    output_dir: Path,
    params: Sequence[_ParamInfo],
    layouts: Mapping[int, DistributedInputLayout],
    world_size: int | None,
) -> None:
    if world_size is not None:
        _validate_layout_topology(layouts, world_size)
    try:
        with (output_dir / _HOST_PATH).open("rb") as host:
            marker = host.readline()
    except FileNotFoundError:
        marker = b""
    if layouts:
        valid = world_size is not None and marker == _artifact_marker(params, layouts, world_size)
    else:
        valid = not marker.startswith(_MARKER)
    if not valid:
        raise ValueError("Distributed input layout metadata does not match the generated HOST entry")


def _invalidate_layout_entry(output_dir: Path, *, selected: bool) -> bool:
    """Mark a reused packing artifact unusable until codegen publishes its replacement."""
    host_path = output_dir / _HOST_PATH
    if not host_path.is_file():
        return selected
    source = host_path.read_bytes()
    if not selected and not source.startswith(_MARKER):
        return False
    host_path.write_bytes(_MARKER + b"incomplete compilation\n" + source)
    return True


def _bind_layout_entry(
    files: dict[str, str],
    output_dir: Path,
    lowered: Program,
    layouts: Mapping[int, DistributedInputLayout],
    *,
    world_size: int | None,
    invalidated: bool,
) -> None:
    if layouts:
        assert world_size is not None
        params, _, _ = _extract_func_param_infos(_host_entries(lowered)[0])
        files[_HOST_PATH] = _artifact_marker(params, layouts, world_size).decode() + files[_HOST_PATH]
    if invalidated:
        # Compile-only or removed chip tasks may leave a configuration from a
        # previous build that references executable code in the other layout.
        for config in output_dir.glob("next_levels/*/kernel_config.py"):
            if config.relative_to(output_dir).as_posix() not in files:
                config.unlink()


__all__ = [
    "DistributedInputLayout",
    "PreparedDistributedInput",
    "_bind_layout_entry",
    "_check_artifact_marker",
    "_host_entries",
    "_invalidate_layout_entry",
    "_layout_records",
    "_read_input_layouts",
    "_validate_input_layouts",
    "_validate_layout_topology",
]
