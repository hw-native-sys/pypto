# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-free warmup and read-only artifact inventory commands."""

import argparse
import importlib
import inspect
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pypto import CacheConfig, cache_stats
from pypto.runtime import RunConfig

from .decorator import JITFunction, _torch_dtype_to_pypto


def _object(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}")
    return value


def _path(value: Any, base: Path) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"Expected a nonempty path string, got {value!r}")
    return base / value


def _cache(value: Any, base: Path) -> CacheConfig:
    data = dict(
        _object(value, {"enabled", "root", "readonly", "extra_source_paths", "extra_fingerprint"}, "cache")
    )
    if data.get("root") is not None:
        data["root"] = _path(data["root"], base)
    if "extra_source_paths" in data:
        if type(data["extra_source_paths"]) is not list:
            raise ValueError("cache.extra_source_paths must be a JSON array")
        data["extra_source_paths"] = tuple(_path(p, base) for p in data["extra_source_paths"])
    return CacheConfig(**data)


def _run_config(value: Any, cache: CacheConfig | None, base: Path) -> RunConfig:
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
    from pypto.ir.pass_manager import OptimizationStrategy  # noqa: PLC0415
    from pypto.pypto_core.passes import MemoryPlanner  # noqa: PLC0415

    # JSON has no Python enums/callbacks. Expose serializable build controls,
    # not arbitrary runtime objects or execution-only settings.
    data = dict(
        _object(
            value,
            {
                "platform",
                "strategy",
                "memory_planner",
                "distributed_config",
                "dump_passes",
                "dump_ptoas_passes",
                "save_kernels",
                "save_kernels_dir",
            },
            "run_config",
        )
    )
    for name in ("dump_passes", "dump_ptoas_passes", "save_kernels"):
        if name in data and type(data[name]) is not bool:
            raise ValueError(f"run_config.{name} must be a boolean")
    for name, enum in (("strategy", OptimizationStrategy), ("memory_planner", MemoryPlanner)):
        if name in data:
            if type(data[name]) is not str or data[name] not in enum.__members__:
                raise ValueError(f"Invalid run_config.{name}: {data[name]!r}")
            data[name] = enum.__members__[data[name]]
    if "save_kernels_dir" in data:
        data["save_kernels_dir"] = str(_path(data["save_kernels_dir"], base))
    if "distributed_config" in data:
        dist = _object(
            data["distributed_config"],
            {"device_ids", "num_sub_workers", "runtime", "aicpu_thread_num"},
            "distributed_config",
        )
        if "device_ids" in dist and (
            type(dist["device_ids"]) is not list
            or any(type(x) is not int or x < 0 for x in dist["device_ids"])
        ):
            raise ValueError("distributed_config.device_ids must be an array of nonnegative integers")
        data["distributed_config"] = DistributedConfig(**dist)
    return RunConfig(cache_config=cache, **data)


def _arguments(kernel: JITFunction, request: dict[str, Any]) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    from pypto.language import Tensor  # noqa: PLC0415

    from .decorator import _resolve_annotation  # noqa: PLC0415
    from .specializer import func_name_lookup  # noqa: PLC0415

    names = set(kernel.param_names)
    scalars = _object(request.get("scalars", {}), names, "scalars")
    tensors = _object(request.get("tensors", {}), names, "tensors")
    if set(scalars) & set(tensors):
        raise ValueError("An argument cannot appear in both tensors and scalars")
    args: dict[str, Any] = dict(scalars)
    signature = inspect.signature(kernel._func)
    annotations = {
        name: _resolve_annotation(p.annotation, func_name_lookup(kernel._func))
        for name, p in signature.parameters.items()
    }
    for name, scalar in scalars.items():
        if annotations[name] is Tensor or isinstance(annotations[name], Tensor):
            raise ValueError(f"Tensor parameter {name} cannot be supplied through scalars")
        if type(scalar) not in (bool, int, float) or (type(scalar) is float and not math.isfinite(scalar)):
            raise ValueError(f"scalars.{name} must be a finite JSON scalar number or boolean")
    for name, value in tensors.items():
        if annotations[name] is not Tensor and not isinstance(annotations[name], Tensor):
            raise ValueError(f"Parameter {name} is not annotated as a tensor")
        data = _object(value, {"shape", "dtype"}, f"tensors.{name}")
        shape, dtype = data.get("shape"), data.get("dtype")
        if type(shape) is not list or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f"tensors.{name}.shape must contain positive integer extents")
        dtypes = {
            "FP16": torch.float16,
            "BF16": torch.bfloat16,
            "FP32": torch.float32,
            "INT8": torch.int8,
            "INT16": torch.int16,
            "INT32": torch.int32,
            "INT64": torch.int64,
            "BOOL": torch.bool,
        }
        if type(dtype) is not str or dtype not in dtypes:
            raise ValueError(f"Unsupported tensors.{name}.dtype: {dtype!r}; expected {sorted(dtypes)}")
        annotation = _resolve_annotation(
            signature.parameters[name].annotation, func_name_lookup(kernel._func)
        )
        annotated_shape = getattr(annotation, "shape", None)
        annotated_dtype = getattr(annotation, "dtype", None)
        if annotated_shape is not None and (
            len(shape) != len(annotated_shape)
            or any(type(dim) is int and dim != actual for dim, actual in zip(annotated_shape, shape))
        ):
            raise ValueError(f"Tensor metadata for {name} conflicts with its annotated shape")
        if annotated_dtype is not None and annotated_dtype != _torch_dtype_to_pypto(dtypes[dtype]):
            raise ValueError(f"Tensor metadata for {name} conflicts with its annotated dtype")
        args[name] = torch.empty(shape, dtype=dtypes[dtype], device="meta")
    # The shared binder validates parameter kinds, required arguments, scalar
    # defaults and dynamic dimensions without compilation or tensor allocation.
    kernel._resolve_specialization((), args, allow_signature_mode=True)
    return args


def warm(module_name: str, configuration: Path) -> dict[str, Any]:
    """Validate all requested metadata, then prepare each explicitly named kernel."""
    path = configuration.resolve(strict=True)
    data = _object(
        json.loads(path.read_text()), {"schema_version", "cache", "requests"}, "warmup configuration"
    )
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("warmup configuration requires schema_version 1")
    requests = data.get("requests")
    if type(requests) is not list or not requests:
        raise ValueError("warmup configuration requires a nonempty requests array")
    cache = _cache(data["cache"], path.parent) if "cache" in data else None
    module = importlib.import_module(module_name)
    prepared = []
    for raw in requests:
        request = _object(raw, {"kernel", "run_config", "tensors", "scalars"}, "request")
        name = request.get("kernel")
        if type(name) is not str or not name.isidentifier():
            raise ValueError(f"kernel must name a module-level JIT function, got {name!r}")
        kernel = vars(module).get(name)
        if not isinstance(kernel, JITFunction):
            raise ValueError(f"{module_name}.{name} is not a JIT function")
        config = _run_config(request.get("run_config", {}), cache, path.parent)
        prepared.append((name, kernel, config, _arguments(kernel, request)))
    results = []
    for name, kernel, config, arguments in prepared:
        compiled = kernel.warmup(**arguments, config=config)
        runtime = vars(compiled).get("_artifact_runtime")
        shared = runtime is not None and runtime.handle.spec.state.value == "ready"
        results.append(
            {
                "kernel": name,
                "storage": "shared" if shared else "private",
                "output_dir": str(compiled.output_dir),
            }
        )
    return {"requests": results, "statistics": asdict(cache_stats())}


def stat(root: Path) -> dict[str, Any]:
    """Scan only JSON manifests and sizes; never import cached executable code."""
    entries = []
    if root.exists():
        for path in sorted(root.glob("artifacts/*/*/*/*/artifact_manifest.json")):
            if path.is_symlink() or any(p.is_symlink() for p in path.parents):
                continue
            try:
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("Manifest exceeds size limit")
                manifest = json.loads(path.read_text())
                files = manifest["files"]
                size = sum(entry["size"] for entry in files)
                entries.append({"manifest": str(path), "state": path.parent.name, "payload_bytes": size})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                entries.append({"manifest": str(path), "error": str(exc)})
    return {
        "root": str(root.resolve()),
        "entries": entries,
        "payload_bytes": sum(entry.get("payload_bytes", 0) for entry in entries),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; preparation failures produce a nonzero exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    warm_parser = commands.add_parser("warm")
    warm_parser.add_argument("--module", required=True)
    warm_parser.add_argument("--config", type=Path, required=True)
    stat_parser = commands.add_parser("stat")
    stat_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = warm(args.module, args.config) if args.command == "warm" else stat(args.root)
    except Exception as exc:
        print(f"pypto.jit {args.command}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
