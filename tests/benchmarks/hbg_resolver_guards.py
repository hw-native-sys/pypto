# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure guarded representation reuse inside the unchanged eager resolver.

This is a single-caller, warmed object-cache-hit diagnostic, NOT a product cache.
Binding, annotation evaluation, dependency/source checks, compile policy, cache
lookup, stats and actual submission all remain active. Only pure static metadata
construction and make_cache_key can reuse their previous plain-value result.
Specialization, argument dictionaries, key input dictionaries/sets, and contract
extension are STILL constructed. A compile sentinel rejects misses rather than
pretending to implement a safe mutable-TensorMeta compilation fallback.

Run --check-only for CPU correctness and guard microbenchmarks; otherwise supply
--device and --output with TASK_QUEUE_ENABLE set in a fresh process.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from pypto.ir.pass_manager import OptimizationStrategy
from pypto.jit import _arguments, cache, decorator
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout
from pypto.pypto_core.passes import MemoryPlanner, RuntimeKind

_ATOMS = (
    type(None),
    str,
    int,
    bool,
    DataType,
    TensorLayout,
    OptimizationStrategy,
    MemoryPlanner,
    RuntimeKind,
)
_KEY_FIELDS = frozenset(
    "source_hash param_names tensor_shapes tensor_dtypes dynamic_dims platform strategy "
    "distributed_config analyze_auto_scopes_for_deps emit_source_loc memory_planner "
    "enable_pypto_l0c_double_buffer tensor_layouts dep_layouts runtime enable_buffer_ir".split()
)


def _plain(value: Any) -> bool:
    """Reject user objects before equality can invoke their custom protocols."""
    if type(value) in _ATOMS:
        return True
    if type(value) in (tuple, list, set):
        return all(_plain(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain(item) for key, item in value.items())
    return False


def _copy_plain(value: Any) -> Any:
    if type(value) is dict:
        return {key: _copy_plain(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy_plain(item) for item in value]
    if type(value) is tuple:
        return tuple(_copy_plain(item) for item in value)
    if type(value) is set:
        return set(value)
    return value


class KeyReuse:
    """One pre-contract-extension key, guarded by all currently supplied fields."""

    def __init__(self) -> None:
        self.original = cache.make_cache_key
        self.previous: dict[str, Any] | None = None
        self.semantics = None
        self.result = None

    def _eligible(self, kwargs: dict[str, Any]) -> bool:
        return kwargs.keys() == _KEY_FIELDS and kwargs["distributed_config"] is None and _plain(kwargs)

    def __call__(self, **kwargs: Any) -> Any:
        if not self._eligible(kwargs):
            return self.original(**kwargs)
        if self.semantics == cache.SCALAR_SEMANTICS and kwargs == self.previous:
            return self.result
        result = self.original(**kwargs)
        self.previous = _copy_plain(kwargs)
        self.semantics = cache.SCALAR_SEMANTICS
        self.result = result
        return result


class ResolverKeyReuse(KeyReuse):
    """Trust shape/index normalization ONLY at the real resolver call site.

    TensorMeta extents have already passed int(); dimension indices come from
    enumerate(). Inspect unvalidated config fields and annotation enum results
    shallowly, without recursively validating the same internal shape tree.
    Nonempty dep layouts and custom values conservatively use the original key.
    This still receives every freshly constructed key-input dict/set.
    """

    _OPTIONS = (
        ("source_hash", (str,)),
        ("platform", (str, type(None))),
        ("strategy", (OptimizationStrategy, type(None))),
        ("distributed_config", (type(None),)),
        ("analyze_auto_scopes_for_deps", (bool,)),
        ("emit_source_loc", (bool,)),
        ("memory_planner", (MemoryPlanner, type(None))),
        ("enable_pypto_l0c_double_buffer", (bool,)),
        ("enable_buffer_ir", (bool,)),
        ("runtime", (RuntimeKind,)),
    )

    def _eligible(self, kwargs: dict[str, Any]) -> bool:
        if kwargs.keys() != _KEY_FIELDS or type(cache.SCALAR_SEMANTICS) is not int:
            return False
        if kwargs["dep_layouts"] or any(type(kwargs[name]) not in kinds for name, kinds in self._OPTIONS):
            return False
        return (
            all(type(name) is str for name in kwargs["param_names"])
            and all(
                type(name) is str and type(dtype) is DataType
                for name, dtype in kwargs["tensor_dtypes"].items()
            )
            and all(
                layout is None or type(layout) is TensorLayout for layout in kwargs["tensor_layouts"].values()
            )
        )


class MetadataReuse:
    """One owned mutable result per argument slot; never shared across slots."""

    def __init__(self) -> None:
        self.original = _arguments._build_tensor_meta
        self.slots: list[Any] = []
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def __call__(self, extents: Any, dtype: Any, dyn_dims: Any = None, layout: Any = None) -> Any:
        slot = self.index
        self.index += 1
        if (
            slot >= 32
            or type(extents) not in (list, tuple)
            or any(type(dim) is not int for dim in extents)
            or type(dtype) is not DataType
            or (layout is not None and type(layout) is not TensorLayout)
            or (dyn_dims is not None and (type(dyn_dims) is not dict or dyn_dims))
        ):
            return self.original(extents, dtype, dyn_dims, layout)
        shape = tuple(extents)
        if slot < len(self.slots):
            previous = self.slots[slot]
            if previous is not None and (
                previous.shape == shape and previous.dtype is dtype and previous.layout is layout
            ):
                return previous
        result = self.original(extents, dtype, dyn_dims, layout)
        while len(self.slots) <= slot:
            self.slots.append(None)
        self.slots[slot] = result
        return result


def _example_key() -> dict[str, Any]:
    return dict(
        source_hash="source",
        param_names=["x", "y", "out"],
        tensor_shapes={name: (32, 128) for name in ("x", "y", "out")},
        tensor_dtypes={name: DataType.FP32 for name in ("x", "y", "out")},
        tensor_layouts={name: None for name in ("x", "y", "out")},
        dynamic_dims=set(),
        platform="a2a3",
        strategy=OptimizationStrategy.Default,
        distributed_config=None,
        analyze_auto_scopes_for_deps=False,
        emit_source_loc=True,
        memory_planner=MemoryPlanner.PYPTO,
        enable_pypto_l0c_double_buffer=False,
        dep_layouts=(),
        runtime=RuntimeKind.HOST_BUILD_GRAPH,
        enable_buffer_ir=True,
    )


def check_reuse() -> dict[str, Any]:
    """Compare changed current facts against the original pure constructors."""
    values = _example_key()
    reuse = KeyReuse()
    first = reuse(**values)
    assert reuse(**_copy_plain(values)) is first
    resolved_reuse = ResolverKeyReuse()
    resolved_first = resolved_reuse(**values)
    assert resolved_reuse(**_copy_plain(values)) is resolved_first
    changes = {
        "source_hash": "changed",
        "param_names": ["y", "x", "out"],
        "tensor_shapes": {**values["tensor_shapes"], "x": (64, 128)},
        "tensor_dtypes": {**values["tensor_dtypes"], "x": DataType.FP16},
        "tensor_layouts": {**values["tensor_layouts"], "x": TensorLayout.ND},
        "dynamic_dims": {("x", 0)},
        "platform": "a2a3sim",
        "strategy": None,
        "analyze_auto_scopes_for_deps": True,
        "emit_source_loc": False,
        "memory_planner": MemoryPlanner.PTOAS,
        "enable_pypto_l0c_double_buffer": True,
        "dep_layouts": (("helper", "x", "ND"),),
        "runtime": RuntimeKind.TENSORMAP_AND_RINGBUFFER,
        "enable_buffer_ir": False,
    }
    for name, value in changes.items():
        changed: dict[str, Any] = {**_copy_plain(values), name: value}
        expected = cache.make_cache_key(**changed)
        actual = reuse(**changed)
        assert actual == expected, name
        assert resolved_reuse(**changed) == expected, name
        assert resolved_reuse(**values) == first
        assert actual != first, name
        assert reuse(**values) == first
    values["tensor_shapes"]["x"] = (16, 128)
    assert reuse(**values) == cache.make_cache_key(**values)
    assert not _plain(object())
    assert not resolved_reuse._eligible({**values, "strategy": object()})
    assert not resolved_reuse._eligible({**values, "analyze_auto_scopes_for_deps": 1})
    assert not resolved_reuse._eligible({**values, "dep_layouts": (("helper", "x", "ND"),)})

    class CustomPlanner:
        calls = 0

        def __str__(self) -> str:
            self.calls += 1
            return "diagnostic_custom_planner"

    planner = CustomPlanner()
    fallback_calls: list[None] = []

    def original_key(*args: Any, **kwargs: Any) -> Any:
        fallback_calls.append(None)
        return cache.make_cache_key(*args, **kwargs)

    resolved_reuse.original = original_key
    for unsupported in (
        {**values, "memory_planner": planner},
        {**values, "analyze_auto_scopes_for_deps": 1},
        {**values, "dep_layouts": (("helper", "x", "ND"),)},
    ):
        before = len(fallback_calls)
        assert resolved_reuse(**unsupported) == resolved_reuse(**unsupported)
        assert len(fallback_calls) == before + 2
    assert planner.calls == 2  # Original __str__ runs once per call, never cached.
    metadata = MetadataReuse()
    metadata.reset()
    one = metadata([32, 128], DataType.FP32)
    two = metadata([32, 128], DataType.FP32)
    assert one == two and one is not two
    metadata.reset()
    assert metadata([32, 128], DataType.FP32) is one
    one.layout = TensorLayout.ND
    metadata.reset()
    restored = metadata([32, 128], DataType.FP32)
    assert restored is not one and restored.layout is None
    for extents, dtype, layout in (
        ([64, 128], DataType.FP32, None),
        ([32, 128], DataType.FP16, None),
        ([32, 128], DataType.FP32, TensorLayout.ND),
    ):
        metadata.reset()
        assert metadata(extents, dtype, layout=layout) == metadata.original(extents, dtype, layout=layout)
    from types import SimpleNamespace  # noqa: PLC0415

    import torch  # noqa: PLC0415

    invalid_fp4 = (
        SimpleNamespace(dtype=torch.uint8, shape=()),
        SimpleNamespace(dtype=torch.uint8, shape=(1, 0)),
        SimpleNamespace(dtype=torch.float32, shape=(32, 128)),
    )
    original_build = _arguments._build_tensor_meta
    errors: list[list[str]] = []
    try:
        for build in (original_build, metadata):
            _arguments._build_tensor_meta = build
            observed = []
            for tensor in invalid_fp4:
                try:
                    _arguments.extract_tensor_meta(tensor, expected_dtype=DataType.FP4E1M2X2)
                except TypeError as error:
                    observed.append(str(error))
                else:
                    raise AssertionError("Packed FP4 validation unexpectedly accepted invalid input")
            errors.append(observed)
    finally:
        _arguments._build_tensor_meta = original_build
    assert errors[0] == errors[1]
    return {
        "key_mutations_checked": list(changes),
        "metadata_slot_and_mutation_checks": "passed",
        "resolver_key_custom_value_fallback_checks": "passed",
        "packed_fp4_errors": errors[0],
    }


def microbench(iterations: int = 10000) -> dict[str, Any]:
    values = _example_key()
    key_reuse, resolved_reuse, metadata = KeyReuse(), ResolverKeyReuse(), MetadataReuse()
    key_reuse(**values)
    resolved_reuse(**values)
    metadata([32, 128], DataType.FP32)

    def metadata_hit() -> Any:
        metadata.reset()
        return metadata([32, 128], DataType.FP32)

    names = ["x", "y", "out"]
    specialization_fields = (
        names,
        {name: object() for name in names},
        {name: metadata.original([32, 128], DataType.FP32) for name in names},
        {},
        {},
        {},
    )
    functions = {
        "empty_call_control": lambda: None,
        "specialization_container_only": lambda: decorator._Specialization(*specialization_fields),
        "key_original": lambda: cache.make_cache_key(**values),
        "key_guard_and_hit": lambda: key_reuse(**values),
        "resolver_key_guard_and_hit": lambda: resolved_reuse(**values),
        "metadata_original": lambda: metadata.original([32, 128], DataType.FP32),
        "metadata_guard_reset_and_hit": metadata_hit,
    }
    samples = {name: [] for name in functions}
    for repeat in range(8):
        order = list(functions)
        if repeat % 2:
            order.reverse()
        for name in order:
            call = functions[name]
            start = time.thread_time_ns()
            for _ in range(iterations):
                call()
            samples[name].append((time.thread_time_ns() - start) / iterations / 1000)
    return {
        name: {"median_cpu_us": statistics.median(rows), "samples_cpu_us": rows}
        for name, rows in samples.items()
    }


def measure_guarded(probe: Any, output: Path, samples: int, batches: int) -> None:
    from hbg_eager_latency import _DiagnosticPatches, add_kernel, measure, npu  # noqa: PLC0415

    patches = _DiagnosticPatches()
    prepare = probe.prepare
    finish = probe.finish
    bind = add_kernel._bind_args

    def compile_miss(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Guarded representation diagnostic encountered a compile miss; aborting")

    def prepare_variant(stage: str) -> None:
        patches.restore()
        prepare("aclnn" if stage == "aclnn" else "jit")
        (probe.aclnn_out if stage == "aclnn" else probe.out).fill_(float("nan"))
        npu.synchronize()
        if stage in ("baseline", "aclnn"):
            return
        patches.set(add_kernel, "_compile", compile_miss)
        if stage in ("metadata", "combined", "combined-resolver-key"):
            metadata = MetadataReuse()

            def bind_current(*args: Any, **kwargs: Any) -> Any:
                metadata.reset()
                return bind(*args, **kwargs)

            patches.set(add_kernel, "_bind_args", bind_current)
            patches.set(_arguments, "_build_tensor_meta", metadata)
        if stage in ("key", "combined"):
            patches.set(decorator, "make_cache_key", KeyReuse())
        if stage in ("resolver-key", "combined-resolver-key"):
            patches.set(decorator, "make_cache_key", ResolverKeyReuse())

    def finish_variant() -> None:
        try:
            finish()
        finally:
            # Include release of retained metadata/key records in lifecycle CPU.
            patches.restore()

    probe.prepare = prepare_variant
    probe.finish = finish_variant
    try:
        # This changes only the spelling of an equivalent annotation. Observe
        # the real resolver evaluating it equally often in baseline and reuse.
        annotations = add_kernel._func.__annotations__
        original_annotation = annotations["x"]
        namespace = add_kernel._func.__globals__
        probe_name = "_hbg_guard_annotation_probe"
        assert probe_name not in namespace
        annotation_calls: list[None] = []

        def annotation_probe() -> Any:
            annotation_calls.append(None)
            return namespace["pl"].Tensor

        namespace[probe_name] = annotation_probe
        counts: dict[str, int] = {}
        try:
            annotations["x"] = f"{probe_name}()"
            for stage in ("baseline", "combined", "combined-resolver-key"):
                probe.finish()
                prepare_variant(stage)
                annotation_calls.clear()
                key_factory = decorator.make_cache_key
                observed_keys = []

                def observe_key(*args: Any, **kwargs: Any) -> Any:
                    key = key_factory(*args, **kwargs)
                    observed_keys.append(key)
                    return key

                decorator.make_cache_key = observe_key
                try:
                    for _ in range(3):
                        probe.call()
                        npu.synchronize()
                        probe.check()
                finally:
                    decorator.make_cache_key = key_factory
                if stage != "baseline":
                    assert isinstance(key_factory, KeyReuse)
                    assert key_factory.result is not None
                    assert all(key is key_factory.result for key in observed_keys)
                counts[stage] = len(annotation_calls)
                assert counts[stage] > 0
            assert counts["baseline"] == counts["combined"] == counts["combined-resolver-key"]
            try:
                add_kernel(*(tensor.reshape(64, 64) for tensor in probe.inputs))
            except AssertionError as error:
                assert str(error) == (
                    "Guarded representation diagnostic encountered a compile miss; aborting"
                )
            else:
                raise AssertionError("Changed shape did not reach the compile-miss sentinel")
            (output / "live-guards.json").write_text(
                json.dumps(
                    {
                        "string_annotation_evaluations": counts,
                        "key_identity_hits": ["combined", "combined-resolver-key"],
                        "shape_change": "compile-miss sentinel reached; not a supported compilation fallback",
                    },
                    indent=2,
                )
            )
        finally:
            annotations["x"] = original_annotation
            del namespace[probe_name]
            try:
                probe.finish()
            finally:
                patches.restore()
        measure(
            probe,
            output,
            samples,
            batches,
            ("baseline", "metadata", "key", "combined", "resolver-key", "combined-resolver-key", "aclnn"),
        )
    finally:
        try:
            probe.finish()
        finally:
            patches.restore()
            probe.prepare = prepare
            probe.finish = finish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()
    if not args.check_only and args.device is None:
        parser.error("hardware measurement requires --device")
    if not args.check_only and decorator.capture_cache_config(None).enabled:
        parser.error("object-cache diagnostic requires persistent caching disabled (PYPTO_CACHE=0)")
    if os.environ.get("LD_PRELOAD") or os.environ.get("PYPTO_LAUNCH_PROBE_OUTPUT"):
        parser.error("use a fresh process without instrumentation")
    args.output.mkdir(parents=True, exist_ok=True)
    checks = check_reuse()
    (args.output / "checks.json").write_text(json.dumps(checks, indent=2))
    (args.output / "guard-cpu.json").write_text(json.dumps(microbench(), indent=2))
    (args.output / "scope.json").write_text(
        json.dumps(
            {
                "queue": os.environ.get("TASK_QUEUE_ENABLE"),
                "persistent_cache_enabled": decorator.capture_cache_config(None).enabled,
                "guards": "original binding, annotation, source, dependency, options and cache checks",
                "metadata": "static-integer _build_tensor_meta only; independent slots, at most 32",
                "key": "compare all current plain-value fields; one key before contract extension",
                "resolver_key": "trust internal shape/index normalization; shallow option/name/enum checks",
                "still_constructed": [
                    "_Specialization",
                    "arguments",
                    "tensor_meta mapping",
                    "per-function maps",
                    "key input dictionaries",
                    "dynamic dimension set",
                    "contract extension",
                ],
                "compile_miss": "explicit failure sentinel, not a supported compilation fallback",
                "concurrency": "single-caller fixture only; not a product cache",
                "interpretation": "cost and net benefit of two factories; does not bound the entire resolver",
            },
            indent=2,
        )
    )
    if args.check_only:
        return
    from hbg_eager_latency import FixedProbe, configure_logging, npu, torch  # noqa: PLC0415

    configure_logging("error")
    torch.set_num_threads(1)
    npu.set_device(args.device)
    npu.set_compile_mode(jit_compile=False)
    probe = FixedProbe(args.device)
    try:
        measure_guarded(probe, args.output, args.samples, args.batches)
    finally:
        try:
            probe.finish()
        finally:
            probe.state.close()


if __name__ == "__main__":
    main()
