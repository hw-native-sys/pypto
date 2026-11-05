# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parent-side recursive entry records; cold readers never execute generated Python."""

import ast
import json
import re
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from pypto.pypto_core import codegen, ir

_LAYOUT_FIELDS = {
    "version",
    "tensor_slots",
    "scalar_slots",
    "window_slots",
    "implicit_outputs",
    "prefork_requirements",
}
_ENTRY_FIELDS = {
    "entry_name",
    "logical_path",
    "module",
    "qualname",
    "chip_paths",
    "sub_worker_paths",
    "layout",
    "view_geometry",
    "host_tensor_indices",
    "minimum_sub_workers",
    "local_device_count",
    "return_copy_callable",
}
_SCALARS = {
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "index",
    "fp32",
    "fp64",
    "comm_ctx",
}
_TENSORS = _SCALARS - {"index", "comm_ctx"} | {"fp16", "bf16"}
_COPY = {"module": "orchestration._distributed_support", "qualname": "_copy_return_tensor"}


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(f"l3_entries {detail}")


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z_]\w*", value, flags=re.ASCII))


def logical_path(value: Any) -> str:
    """Check normalized portable paths before they select files or registrations."""
    _require(
        isinstance(value, str) and bool(value) and "\\" not in value and "\0" not in value,
        f"invalid logical path {value!r}",
    )
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and str(path) == value
        and all(part not in {".", ".."} for part in value.split("/")),
        f"invalid logical path {value!r}",
    )
    return value


def reachable_functions(program: ir.Program, entry: ir.Function) -> list[ir.Function]:
    """Follow Call and Submit once per function, keeping the complete owned IR."""
    from pypto.backend.pto_backend import _CallCollector  # noqa: PLC0415

    found: dict[str, ir.Function] = {}
    pending = [entry]
    while pending:
        function = pending.pop()
        if function.name in found:
            continue
        found[function.name] = function
        calls = _CallCollector()
        calls.visit_stmt(function.body)
        for name in calls.callee_names:
            callee = program.get_function(name)
            if callee is not None and name not in found:
                pending.append(callee)
    return [found[name] for name in sorted(found)]


def host_entries(program: ir.Program) -> list[ir.Function]:
    from .distributed_compiled_program import _actual_orchestration_entry  # noqa: PLC0415

    root = _actual_orchestration_entry(program)
    _require(root is not None and root.level == ir.Level.CLUSTER_0, "requires a CLUSTER_0 entry")
    return [
        function
        for function in reachable_functions(program, root)
        if function.level == ir.Level.HOST and function.role == ir.Role.Orchestrator
    ]


def source_layout(source: str, entry_name: str) -> dict[str, Any]:
    """Read only the native immutable literal, without importing or executing source."""
    values = [
        statement.value
        for statement in ast.parse(source).body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_pypto_embedded_layout"
            and isinstance(target.value, ast.Name)
            and target.value.id == entry_name
            for target in statement.targets
        )
    ]
    _require(
        len(values) == 1 and isinstance(values[0], ast.Call) and len(values[0].args) == 1,
        f"entry {entry_name!r} lacks one native layout literal",
    )
    return json.loads(json.dumps(ast.literal_eval(values[0].args[0])))


def _geometry(function: ir.Function, layout: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for window in layout["window_slots"]:
        formal = window[0]
        tensor = function.params[formal].type
        _require(isinstance(tensor, ir.DistributedTensorType), "window formal must be DistributedTensor")
        view = tensor.tensor_view
        strides = [stride.value for stride in view.stride] if view is not None and view.stride else []
        if not strides:
            stride = 1
            for extent in reversed(window[7]):
                strides.insert(0, stride)
                stride *= extent
        # Native embedded descriptors use offsets relative to the owned WB slot,
        # even when a later allocation pass assigned the param a separate base.
        offset = tensor.memref.byte_offset_.value if tensor.memref is not None else 0
        rows.append({"formal_index": formal, "strides": strides, "byte_offset": offset})
    return {"version": 1, "window_slots": rows}


def make_entry_records(program: ir.Program, local_device_count: int) -> dict[str, Any]:
    """Capture native ABI and independent geometry while full parent IR is available."""
    records = []
    for function in host_entries(program):
        name = function.name
        _require(_name(name), f"invalid entry name {name!r}")
        prefix = f"next_levels/{name}"
        generator = codegen.DistributedCodegen()
        source = generator._generate_embedded(program, entry_name=name)
        layout = source_layout(source, name)
        closure = reachable_functions(program, function)
        chips = {
            f.name: f"{prefix}/next_levels/{f.name}"
            for f in closure
            if f.level == ir.Level.CHIP and f.role == ir.Role.Orchestrator
        }
        # Builtin registrations are scoped to this exact HOST generation, not
        # the preceding root or another entry sharing the generator type.
        for spec in generator.get_builtin_next_level_specs():
            chips[spec.variant] = f"{prefix}/next_levels/{spec.variant}"
        subs = {
            f.name: f"{prefix}/sub_workers/{f.name}.py"
            for f in closure
            if f.level == ir.Level.HOST and f.role == ir.Role.SubWorker
        }
        copying = bool(layout["implicit_outputs"])
        records.append(
            {
                "entry_name": name,
                "logical_path": prefix,
                "module": f"next_levels.{name}.orchestration.host_orch",
                "qualname": name,
                "chip_paths": dict(sorted(chips.items())),
                "sub_worker_paths": dict(sorted(subs.items())),
                "layout": layout,
                "view_geometry": _geometry(function, layout),
                "host_tensor_indices": list(range(len(layout["tensor_slots"]))),
                "minimum_sub_workers": len(subs) + int(copying),
                "local_device_count": local_device_count,
                "return_copy_callable": dict(_COPY) if copying else None,
            }
        )
    result = {"version": 1, "entries": records}
    validate_entry_records(result, local_device_count)
    return result


def _shape(value: Any, *, dynamic: bool = False) -> bool:
    if not isinstance(value, list):
        return False
    for dimension in value:
        if _integer(dimension):
            continue
        if not dynamic or not isinstance(dimension, list):
            return False
        if len(dimension) == 2 and dimension[0] == "var" and _name(dimension[1]):
            continue
        if (
            len(dimension) == 3
            and dimension[0] == "mul"
            and isinstance(dimension[1], list)
            and len(dimension[1]) == 2
            and dimension[1][0] == "var"
            and _name(dimension[1][1])
            and _integer(dimension[2], 1)
        ):
            continue
        return False
    return True


def _rows(layout: dict[str, Any], field: str, length: int) -> list[list[Any]]:
    rows = layout[field]
    _require(
        isinstance(rows, list) and all(isinstance(row, list) and len(row) == length for row in rows),
        f"invalid layout {field} rows",
    )
    return rows


def _validate_requirements(layout: dict[str, Any], local_count: int) -> None:
    requirements = layout["prefork_requirements"]
    _require(isinstance(requirements, list), "layout prefork_requirements must be a list")
    names: set[str] = set()
    contexts = {row[2] for row in layout["scalar_slots"] if row[3] == "comm_ctx"}
    for row in requirements:
        _require(isinstance(row, list) and bool(row), "invalid layout prefork requirement")
        kind = row[0]
        if kind == "tensor_storage":
            _require(
                len(row) == 4
                and _name(row[1])
                and row[1] not in names
                and row[2] in _TENSORS
                and _shape(row[3]),
                "invalid prefork tensor_storage",
            )
            names.add(row[1])
        elif kind == "return_copy_subworker":
            _require(len(row) == 2 and type(row[1]) is int and row[1] == 1, "invalid return copy capacity")
        elif kind == "inherited_domain_tokens":
            _require(
                len(row) == 2 and _integer(row[1], 1) and row[1] == len(contexts),
                "invalid inherited domain token capacity",
            )
        elif kind in {"domain_dispatch", "local_domain_tokens"}:
            _require(len(row) == 3, "invalid domain dispatch requirement")
            _require(
                (type(row[1]) is int and row[1] in contexts)
                if kind == "domain_dispatch"
                else isinstance(row[1], str) and bool(row[1]),
                "invalid domain token identity",
            )
            workers = row[2]
            _require(
                workers is None
                or isinstance(workers, list)
                and len(workers) == len(set(workers))
                and all(_integer(worker) and worker < local_count for worker in workers),
                "domain token workers exceed local topology",
            )
        elif kind == "same_domain":
            _require(
                len(row) == 2
                and isinstance(row[1], list)
                and all(type(index) is int and index in contexts for index in row[1]),
                "invalid same_domain context group",
            )
        else:
            raise ValueError(f"l3_entries unknown layout prefork requirement {kind!r}")
    _require(
        len(requirements) == len({json.dumps(row, sort_keys=True) for row in requirements}),
        "duplicate layout prefork requirements",
    )
    _require(
        bool(layout["implicit_outputs"]) == (["return_copy_subworker", 1] in requirements),
        "layout return copy requirement disagrees with implicit outputs",
    )


def _validate_layout(layout: Any, geometry: Any, local_count: int) -> None:
    _require(isinstance(layout, dict) and set(layout) == _LAYOUT_FIELDS, "invalid layout fields")
    _require(type(layout["version"]) is int and layout["version"] == 1, "unsupported layout version")
    tensors = _rows(layout, "tensor_slots", 6)
    scalars = _rows(layout, "scalar_slots", 4)
    windows = _rows(layout, "window_slots", 8)
    outputs = _rows(layout, "implicit_outputs", 4)
    positions: set[int] = set()
    names: set[str] = set()
    for rows, field in ((tensors, "tensor"), (scalars, "scalar"), (windows, "window")):
        for row in rows:
            _require(
                _integer(row[0]) and row[0] not in positions and _name(row[1]) and row[1] not in names,
                f"duplicate/invalid layout {field} formal position or name",
            )
            positions.add(row[0])
            names.add(row[1])
        if field != "window":
            _require(
                all(type(row[2]) is int and row[2] == index for index, row in enumerate(rows)),
                f"layout {field} wire indices must be independently dense",
            )
    _require(positions == set(range(len(positions))), "layout formal positions are not complete")
    for row in tensors:
        _require(
            row[3] in _TENSORS and row[4] in {"In", "Out", "InOut"} and _shape(row[5], dynamic=True),
            "invalid layout tensor dtype/direction/shape",
        )
    for row in scalars:
        _require(row[3] in _SCALARS, "invalid layout scalar dtype")
    contexts = {row[0]: row for row in scalars if row[3] == "comm_ctx"}
    for row in windows:
        _require(
            type(row[2]) is int
            and row[2] in contexts
            and type(row[3]) is int
            and contexts[row[2]][2] == row[3]
            and _name(row[4])
            and row[5] in _TENSORS
            and row[6] in {"In", "Out", "InOut"}
            and _shape(row[7]),
            "invalid layout window/context pairing or shape",
        )
    # C2b's final N contexts pair in order to N DT formals, after the original
    # scalar/CommCtx prefix. No implicit DT tensor wire slot is permitted.
    if windows:
        suffix = sorted(positions)[-len(windows) :]
        _require([row[2] for row in windows] == suffix, "layout window contexts must be the final DT suffix")
    for index, row in enumerate(outputs):
        _require(
            _integer(row[0])
            and type(row[1]) is int
            and row[1] == len(tensors) + index
            and row[2] in _TENSORS
            and _shape(row[3]),
            "invalid layout implicit output",
        )
    _require(len({row[0] for row in outputs}) == len(outputs), "duplicate layout implicit return position")
    _validate_requirements(layout, local_count)
    _require(
        isinstance(geometry, dict)
        and set(geometry) == {"version", "window_slots"}
        and type(geometry["version"]) is int
        and geometry["version"] == 1,
        "invalid view geometry fields/version",
    )
    views = geometry["window_slots"]
    _require(isinstance(views, list) and len(views) == len(windows), "incomplete view geometry")
    for view, window in zip(views, windows, strict=True):
        _require(
            isinstance(view, dict)
            and set(view) == {"formal_index", "strides", "byte_offset"}
            and type(view["formal_index"]) is int
            and view["formal_index"] == window[0]
            and isinstance(view["strides"], list)
            and len(view["strides"]) == len(window[7])
            and all(_integer(stride) for stride in view["strides"])
            and _integer(view["byte_offset"]),
            "invalid view geometry formal/stride/byte offset",
        )


def validate_entry_records(value: Any, local_device_count: int) -> dict[str, Any]:
    """Validate the entire cold contract before paths, code or native handles are consumed."""
    _require(isinstance(value, dict) and set(value) == {"version", "entries"}, "invalid closure fields")
    _require(type(value["version"]) is int and value["version"] == 1, "unsupported closure version")
    _require(isinstance(value["entries"], list), "entries must be a list")
    entries = value["entries"]
    seen: set[str] = set()
    for entry in entries:
        _require(isinstance(entry, dict) and set(entry) == _ENTRY_FIELDS, "invalid entry fields")
        name = entry["entry_name"]
        _require(_name(name) and name not in seen, "invalid/duplicate entry name")
        seen.add(name)
        prefix = logical_path(entry["logical_path"])
        _require(prefix == f"next_levels/{name}", "entry logical path does not match its name")
        _require(
            entry["module"] == f"next_levels.{name}.orchestration.host_orch" and entry["qualname"] == name,
            "entry module/qualname does not match its owner",
        )
        _require(
            type(entry["local_device_count"]) is int and entry["local_device_count"] == local_device_count,
            "entry local topology differs from cluster_config",
        )
        for field, segment, suffix in (
            ("chip_paths", "next_levels", ""),
            ("sub_worker_paths", "sub_workers", ".py"),
        ):
            mapping = entry[field]
            _require(isinstance(mapping, dict), f"entry {field} must be a mapping")
            for child, path in mapping.items():
                _require(
                    _name(child) and logical_path(path) == f"{prefix}/{segment}/{child}{suffix}",
                    f"entry {field} path is outside its owner",
                )
        _validate_layout(entry["layout"], entry["view_geometry"], local_device_count)
        tensor_count = len(entry["layout"]["tensor_slots"])
        _require(
            entry["host_tensor_indices"] == list(range(tensor_count))
            and all(type(index) is int for index in entry["host_tensor_indices"]),
            "entry host tensor indices do not match dense layout",
        )
        copying = bool(entry["layout"]["implicit_outputs"])
        _require(
            entry["return_copy_callable"] == (_COPY if copying else None),
            "entry return copy callable disagrees with its layout",
        )
        _require(
            type(entry["minimum_sub_workers"]) is int
            and entry["minimum_sub_workers"] == len(entry["sub_worker_paths"]) + int(copying),
            "entry minimum SubWorkers disagrees with prefork requirements",
        )
    _require([entry["entry_name"] for entry in entries] == sorted(seen), "entry order must be canonical")
    return value


def freeze_domains(value: Any) -> Any:
    """Return immutable descriptors without retaining caller-owned containers."""
    if isinstance(value, dict):
        return MappingProxyType({key: freeze_domains(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_domains(item) for item in value)
    return value


def make_domain_records(program: ir.Program, entries: dict[str, Any], node_count: int) -> dict[str, Any]:
    """Persist root-domain provenance from native post-pass IR, never source bodies."""
    from .distributed_compiled_program import _actual_orchestration_entry  # noqa: PLC0415

    scopes = []

    class Collector(ir.IRVisitor):
        def visit_stmt(self, op):
            if isinstance(op, ir.CommDomainScopeStmt):
                scopes.append(op)
            super().visit_stmt(op)

    root = _actual_orchestration_entry(program)
    Collector().visit_stmt(root.body)
    domains = []
    for ordinal, scope in enumerate(scopes):
        slots = []
        for index, slot in enumerate(scope.slots):
            size = slot.size.value if isinstance(slot.size, ir.ConstInt) else None
            slots.append(
                {
                    "ordinal": index,
                    "generated_name": re.sub(r"[^A-Za-z0-9_]", "_", slot.name_hint),
                    "logical_nbytes": size,
                    "allocated_nbytes": ((size + 63) // 64) * 64 if size is not None else None,
                }
            )
        by_name = {slot["generated_name"]: slot["ordinal"] for slot in slots}
        windows = []
        for entry in entries["entries"]:
            for window, geometry in zip(
                entry["layout"]["window_slots"], entry["view_geometry"]["window_slots"], strict=True
            ):
                if window[4] in by_name:
                    windows.append(
                        {
                            "entry_logical_path": entry["logical_path"],
                            "ctx_scalar_index": window[3],
                            "logical_window_name": window[4],
                            "slot_ordinal": by_name[window[4]],
                            "dtype": window[5],
                            "shape": window[7],
                            "strides": geometry["strides"],
                            "byte_offset": geometry["byte_offset"],
                        }
                    )
        domains.append(
            {
                "ordinal": ordinal,
                "generated_name": scope.name_hint,
                "node_indices": list(scope.devices) or list(range(node_count)),
                "slots": slots,
                "entry_windows": windows,
            }
        )
    return validate_domain_records({"version": 1, "domains": domains}, entries, node_count)


def validate_domain_records(value: Any, entries: dict[str, Any], node_count: int) -> dict[str, Any]:
    """Check complete cold slot/view identities before session selection or startup."""

    def require(condition, detail):
        if not condition:
            raise ValueError("cluster_domains " + detail)

    require(isinstance(value, dict) and set(value) == {"version", "domains"}, "invalid fields")
    require(type(value["version"]) is int and value["version"] == 1, "unsupported version")
    require(isinstance(value["domains"], list), "domains must be a list")
    native_windows = {}
    for entry in entries["entries"]:
        for window, geometry in zip(
            entry["layout"]["window_slots"], entry["view_geometry"]["window_slots"], strict=True
        ):
            native_windows[(entry["logical_path"], window[3], window[4])] = (
                window[5],
                window[7],
                geometry["strides"],
                geometry["byte_offset"],
            )
    seen_domains, seen_slots, covered_windows = set(), set(), set()
    for ordinal, domain in enumerate(value["domains"]):
        require(
            isinstance(domain, dict)
            and set(domain) == {"ordinal", "generated_name", "node_indices", "slots", "entry_windows"},
            "invalid domain fields",
        )
        require(
            type(domain["ordinal"]) is int and domain["ordinal"] == ordinal, "domain ordinal is not dense"
        )
        name = domain["generated_name"]
        require(_name(name) and name not in seen_domains, "invalid or duplicate domain name")
        seen_domains.add(name)
        nodes = domain["node_indices"]
        require(
            isinstance(nodes, list)
            and bool(nodes)
            and len(set(nodes)) == len(nodes)
            and all(type(node) is int and 0 <= node < node_count for node in nodes),
            "invalid member order",
        )
        slots = domain["slots"]
        require(isinstance(slots, list) and bool(slots), "domain needs slots")
        for index, slot in enumerate(slots):
            require(
                isinstance(slot, dict)
                and set(slot) == {"ordinal", "generated_name", "logical_nbytes", "allocated_nbytes"},
                "invalid slot fields",
            )
            require(type(slot["ordinal"]) is int and slot["ordinal"] == index, "slot ordinal is not dense")
            require(
                _name(slot["generated_name"]) and slot["generated_name"] not in seen_slots,
                "invalid or duplicate slot name",
            )
            seen_slots.add(slot["generated_name"])
            size, allocated = slot["logical_nbytes"], slot["allocated_nbytes"]
            require(
                (size is None and allocated is None)
                or (
                    type(size) is int
                    and size > 0
                    and type(allocated) is int
                    and allocated == ((size + 63) // 64) * 64
                ),
                "invalid slot size/alignment",
            )
        windows = domain["entry_windows"]
        require(isinstance(windows, list), "entry_windows must be a list")
        seen_windows = set()
        for window in windows:
            require(
                isinstance(window, dict)
                and set(window)
                == {
                    "entry_logical_path",
                    "ctx_scalar_index",
                    "logical_window_name",
                    "slot_ordinal",
                    "dtype",
                    "shape",
                    "strides",
                    "byte_offset",
                },
                "invalid view fields",
            )
            index = window["slot_ordinal"]
            require(type(index) is int and 0 <= index < len(slots), "invalid view slot reference")
            key = (window["entry_logical_path"], window["ctx_scalar_index"], window["logical_window_name"])
            require(
                type(window["ctx_scalar_index"]) is int and key in native_windows and key not in seen_windows,
                "invalid or duplicate entry view",
            )
            require(window["logical_window_name"] == slots[index]["generated_name"], "view slot name differs")
            require(
                (window["dtype"], window["shape"], window["strides"], window["byte_offset"])
                == native_windows[key],
                "view geometry differs from native entry",
            )
            size = slots[index]["logical_nbytes"]
            itemsize = {"bool": 1, "fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8}.get(window["dtype"])
            if itemsize is None:
                itemsize = int(re.sub(r"\D", "", window["dtype"])) // 8
            extent = window["byte_offset"] + itemsize * (
                1
                + sum(
                    (dimension - 1) * stride
                    for dimension, stride in zip(window["shape"], window["strides"], strict=True)
                )
            )
            require(size is None or extent <= size, "view exceeds its logical slot")
            seen_windows.add(key)
        slot_names = {slot["generated_name"] for slot in slots}
        expected_windows = {key for key in native_windows if key[2] in slot_names}
        require(seen_windows == expected_windows, "incomplete native window view coverage")
        require(not (seen_windows & covered_windows), "duplicate native window view ownership")
        covered_windows.update(seen_windows)
    require(covered_windows == set(native_windows), "incomplete global domain window view coverage")
    return value


def restore_domain_records(raw: dict[str, Any], entries: Any, node_count: int) -> dict[str, Any] | None:
    if "cluster_domains" not in raw:
        return None
    if entries is None:
        raise ValueError("cluster_domains requires l3_entries")
    return validate_domain_records(raw["cluster_domains"], entries, node_count)
