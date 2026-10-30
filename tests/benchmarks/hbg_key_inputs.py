# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Diagnose key input assembly without changing product sources.

Only exact static TensorMeta records use direct tensor infos. Unsupported records
execute the original expression, including its evaluation order and miss path.
All source, dependency, annotation, option, contract and cache guards remain live.
AST transforms assert their source shape; baseline and clone expose tooling cost.
Single-caller diagnostic only; not a proposed product cache.
"""

import argparse
import ast
import copy
import inspect
import json
import os
import textwrap
import types
from pathlib import Path
from typing import Any, cast

from hbg_resolver_guards import ResolverKeyReuse
from pypto.jit import cache, decorator
from pypto.jit.specializer import DynDim, TensorMeta
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout

_STATIC_SHAPE = TensorMeta.static_shape
_DYNAMIC_INDICES = TensorMeta.dynamic_dim_indices
_META_FIELDS = {"shape", "dtype", "layout"}


class DirectInputs:
    """One immutable infos tuple; no tensors or mutable metadata are retained."""

    def __init__(self, reuse: bool) -> None:
        self.reuse = reuse
        self.previous: tuple[Any, ...] | None = None

    def __call__(self, names: Any, metadata: Any) -> tuple[Any, ...] | None:
        if (
            type(names) is not list
            or type(metadata) is not dict
            or TensorMeta.static_shape is not _STATIC_SHAPE
            or TensorMeta.dynamic_dim_indices is not _DYNAMIC_INDICES
            or any(type(name) is not str for name in names)
        ):
            return None
        for name, meta in metadata.items():
            if type(name) is not str or type(meta) is not TensorMeta:
                return None
            if (
                meta.__dict__.keys() != _META_FIELDS
                or type(meta.shape) is not tuple
                or any(type(dim) is not int for dim in meta.shape)
                or type(meta.dtype) is not DataType
                or (meta.layout is not None and type(meta.layout) is not TensorLayout)
            ):
                return None
        previous = self.previous
        if self.reuse and previous is not None:
            index = 0
            for name in names:
                if name not in metadata:
                    continue
                meta = metadata[name]
                if index >= len(previous):
                    break
                info = previous[index]
                if (
                    info.name != name
                    or info.shape != meta.shape
                    or info.dtype is not meta.dtype
                    or info.layout is not meta.layout
                ):
                    break
                index += 1
            else:
                if index == len(previous):
                    return previous
        result = tuple(
            cache.TensorCacheInfo(name, metadata[name].shape, metadata[name].dtype, metadata[name].layout)
            for name in names
            if name in metadata
        )
        self.previous = result
        return result


def compile_function(node: ast.FunctionDef, namespace: dict[str, Any]) -> Any:
    local: dict[str, Any] = {}
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, "<hbg-key-input-diagnostic>", "exec"), namespace, local)
    return local[node.name]


def direct_key_function() -> Any:
    """Preserve original compile-option canonicalization and key construction."""
    node = ast.parse(textwrap.dedent(inspect.getsource(cache.make_cache_key))).body[0]
    assert isinstance(node, ast.FunctionDef)
    # Docstring, tensor_infos declaration, tensor loop, then unchanged options.
    assert isinstance(node.body[1], ast.AnnAssign) and isinstance(node.body[2], ast.For)
    assert ast.unparse(node.body[1].target) == "tensor_infos"
    assert ast.unparse(node.body[2].iter) == "param_names"
    del node.body[1:3]
    removed = {"param_names", "tensor_shapes", "tensor_dtypes", "tensor_layouts", "dynamic_dims"}
    required = len(node.args.args) - len(node.args.defaults)
    pairs = list(zip(node.args.args, [None] * required + node.args.defaults, strict=True))
    pairs = [(arg, default) for arg, default in pairs if arg.arg not in removed]
    pairs.insert(1, (ast.arg(arg="tensor_infos"), None))
    node.args.args = [arg for arg, _ in pairs]
    node.args.defaults = [default for _, default in pairs if default is not None]
    node.name = "_diagnostic_direct_key"
    return compile_function(node, cast(dict[str, Any], vars(cache)))


_DIRECT_KEY_FUNCTION = direct_key_function()


class DirectKey:
    """Optional reuse of immutable direct-key inputs after every dynamic read."""

    def __init__(self, reuse: bool) -> None:
        self.original = _DIRECT_KEY_FUNCTION
        self.reuse = reuse
        self.previous: dict[str, Any] | None = None
        self.semantics = None
        self.result = None

    def __call__(self, **kwargs: Any) -> Any:
        eligible = self.reuse and type(cache.SCALAR_SEMANTICS) is int
        if eligible:
            eligible = not kwargs["dep_layouts"] and all(
                type(kwargs[name]) in kinds for name, kinds in ResolverKeyReuse._OPTIONS
            )
        if eligible and self.semantics == cache.SCALAR_SEMANTICS and kwargs == self.previous:
            return self.result
        result = self.original(**kwargs)
        if eligible:
            self.previous, self.result, self.semantics = kwargs, result, cache.SCALAR_SEMANTICS
        return result


def key_statements(call: ast.Call) -> list[ast.stmt]:
    """Read source/names once, then select direct construction or exact fallback."""
    assert [kw.arg for kw in call.keywords][:5] == [
        "source_hash",
        "param_names",
        "tensor_shapes",
        "tensor_dtypes",
        "tensor_layouts",
    ]
    expressions = {kw.arg: kw.value for kw in call.keywords}
    expected = {
        "tensor_shapes": "{n: m.static_shape() for n, m in specialization.tensor_meta.items()}",
        "tensor_dtypes": "{n: m.dtype for n, m in specialization.tensor_meta.items()}",
        "tensor_layouts": "{n: m.layout for n, m in specialization.tensor_meta.items()}",
        "dynamic_dims": (
            "{(n, i) for n, m in specialization.tensor_meta.items() for i in m.dynamic_dim_indices()}"
        ),
    }
    # unparse formatting differs across supported Python versions.
    for name, expression in expected.items():
        assert ast.dump(expressions[name]) == ast.dump(ast.parse(expression, mode="eval").body), name
    original = copy.deepcopy(call)
    original.keywords[0].value = ast.Name(id="_diag_source", ctx=ast.Load())
    original.keywords[1].value = ast.Name(id="_diag_names", ctx=ast.Load())
    direct = copy.deepcopy(original)
    direct.func = ast.Name(id="_hbg_direct_key", ctx=ast.Load())
    removed = {"param_names", "tensor_shapes", "tensor_dtypes", "tensor_layouts", "dynamic_dims"}
    direct.keywords = [kw for kw in direct.keywords if kw.arg not in removed]
    direct.keywords.insert(
        1, ast.keyword(arg="tensor_infos", value=ast.Name(id="_diag_infos", ctx=ast.Load()))
    )
    statements = ast.parse(
        "_diag_source = None\n_diag_names = None\n"
        "_diag_infos = _hbg_direct_inputs(_diag_names, specialization.tensor_meta)\n"
        "if _diag_infos is None:\n    key = None\nelse:\n    key = None\n"
    ).body
    assert isinstance(statements[0], ast.Assign)
    assert isinstance(statements[1], ast.Assign)
    assert isinstance(statements[3], ast.If)
    assert isinstance(statements[3].body[0], ast.Assign)
    assert isinstance(statements[3].orelse[0], ast.Assign)
    statements[0].value = copy.deepcopy(expressions["source_hash"])
    statements[1].value = copy.deepcopy(expressions["param_names"])
    statements[3].body[0].value = original
    statements[3].orelse[0].value = direct
    return statements


def resolver_function(transform: bool) -> Any:
    node = ast.parse(textwrap.dedent(inspect.getsource(decorator.JITFunction._resolve_compiled))).body[0]
    assert isinstance(node, ast.FunctionDef)
    matches = [
        i
        for i, statement in enumerate(node.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value.func) == "make_cache_key"
    ]
    assert len(matches) == 1
    if transform:
        index = matches[0]
        statement = node.body[index]
        assert isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
        node.body[index : index + 1] = key_statements(statement.value)
    node.name = "_diagnostic_resolver"
    # Original shared globals and capture_namespaces decorator are retained.
    return compile_function(node, cast(dict[str, Any], vars(decorator)))


def check_inputs() -> dict[str, Any]:
    from hbg_resolver_guards import _example_key  # noqa: PLC0415

    facts = _example_key()
    names = facts["param_names"]
    metadata = {name: TensorMeta((32, 128), DataType.FP32) for name in names}
    inputs, direct = DirectInputs(True), DirectKey(True)

    def key() -> Any:
        infos = inputs(names, metadata)
        assert infos is not None
        kwargs = {
            k: v
            for k, v in facts.items()
            if k not in {"param_names", "tensor_shapes", "tensor_dtypes", "tensor_layouts", "dynamic_dims"}
        }
        return direct(tensor_infos=infos, **kwargs)

    first = key()
    assert first == cache.make_cache_key(**facts) and key() is first
    for field, value in (("shape", (64, 128)), ("dtype", DataType.FP16), ("layout", TensorLayout.ND)):
        old = getattr(metadata["x"], field)
        setattr(metadata["x"], field, value)
        expected = dict(facts)
        expected[dict(shape="tensor_shapes", dtype="tensor_dtypes", layout="tensor_layouts")[field]] = {
            name: getattr(meta, field) for name, meta in metadata.items()
        }
        assert key() == cache.make_cache_key(**expected) and key() != first
        setattr(metadata["x"], field, old)
    for field, value in (("source_hash", "new"), ("emit_source_loc", False), ("enable_buffer_ir", False)):
        old = facts[field]
        facts[field] = value
        assert key() == cache.make_cache_key(**facts) and key() != first
        facts[field] = old
    metadata["x"].shape = (DynDim("M", "M", 32), 128)
    assert inputs(names, metadata) is None
    metadata["x"].shape = (32, 128)
    metadata["x"].static_shape = lambda: (32, 128)
    assert inputs(names, metadata) is None
    del metadata["x"].static_shape
    original = TensorMeta.static_shape
    try:
        TensorMeta.static_shape = lambda self: (32, 128)
        assert inputs(names, metadata) is None
    finally:
        TensorMeta.static_shape = original
    return {"static_keys_and_mutations": "passed", "dynamic_and_method_override_fallback": "passed"}


def check_expression_order() -> dict[str, Any]:
    """Run the exact expression with dynamic/custom metadata and failures."""
    from types import SimpleNamespace  # noqa: PLC0415

    from hbg_resolver_guards import _example_key  # noqa: PLC0415

    source = ast.parse(textwrap.dedent(inspect.getsource(decorator.JITFunction._resolve_compiled)))
    call = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "make_cache_key"
    )
    facts = _example_key()
    results = []
    for transformed in (False, True):
        events: list[str] = []
        fail = [""]

        def event(name: str, result: Any) -> Any:
            events.append(name)
            if fail[0] == name:
                raise ValueError(name)
            return result

        class CustomMeta:
            dtype = DataType.FP32
            layout = None

            def static_shape(self) -> tuple[int, ...]:
                return event("shape", (32, 128))

            def dynamic_dim_indices(self) -> set[int]:
                return event("dynamic", set())

        namespace = dict(vars(decorator))
        namespace.update(
            _hbg_direct_inputs=DirectInputs(True),
            _hbg_direct_key=DirectKey(True),
            _resolve_memory_planner=lambda _: event("planner", facts["memory_planner"]),
            _resolve_enable_pypto_l0c_double_buffer=lambda: event("dbc", False),
            _resolve_enable_buffer_ir=lambda: event("buffer", True),
            _request_source_hash=lambda src, _: src,
        )
        node = ast.parse(
            "def expression(self, specialization, compile_kwargs, runtime, "
            "target_platform, constexpr_bindings): pass"
        ).body[0]
        assert isinstance(node, ast.FunctionDef)
        node.body = (
            key_statements(call)
            if transformed
            else [ast.Assign(targets=[ast.Name(id="key", ctx=ast.Store())], value=copy.deepcopy(call))]
        )
        node.body.append(ast.Return(value=ast.Name(id="key", ctx=ast.Load())))
        expression = compile_function(node, namespace)
        owner = SimpleNamespace(
            _get_source_hash=lambda: event("source", "source"),
            _constexpr_identity_records=lambda _: event("constexpr", ()),
            _dep_declared_layouts=lambda: event("dep", ()),
        )
        opts = {
            name: facts[name]
            for name in ("strategy", "analyze_auto_scopes_for_deps", "emit_source_loc", "memory_planner")
        }
        observed = []
        for meta in (
            TensorMeta((32, 128), DataType.FP32),
            TensorMeta((DynDim("M", "M", 32), 128), DataType.FP32),
            CustomMeta(),
        ):
            specialization = SimpleNamespace(param_names=["x"], tensor_meta={"x": meta})
            for failure in ("", "shape", "dep", "dynamic", "planner"):
                events.clear()
                fail[0] = failure
                try:
                    result = expression(owner, specialization, opts, facts["runtime"], "a2a3", {})
                    outcome = ("key", result)
                except ValueError as error:
                    outcome = ("error", str(error))
                observed.append((list(events), outcome))
        results.append(observed)
    assert results[0] == results[1]
    return {"expression_fallback_order_and_errors": "15 scenarios identical"}


def live_checks(probe: Any, prepare: Any, finish: Any, output: Path) -> None:
    """Verify real warm hits, annotation reads, and a true compiled shape miss."""
    from hbg_eager_latency import add_kernel, npu, torch  # noqa: PLC0415

    annotations = add_kernel._func.__annotations__
    namespace = add_kernel._func.__globals__
    old = annotations["x"]
    name = "_hbg_key_input_annotation"
    assert name not in namespace
    calls: list[None] = []

    def annotation() -> Any:
        calls.append(None)
        return namespace["pl"].Tensor

    namespace[name] = annotation
    annotations["x"] = f"{name}()"
    counts = {}
    try:
        for stage in ("baseline", "clone", "direct", "infos", "key"):
            finish()
            prepare(stage)
            calls.clear()
            for _ in range(3):
                probe.call()
                npu.synchronize()
                probe.check()
            counts[stage] = len(calls)
            if stage == "key":
                factory = getattr(decorator, "_hbg_direct_key")
                assert factory.result is not None
                original = factory.original

                def unexpected_build(**kwargs: Any) -> Any:
                    raise AssertionError("Expected a warm direct key hit")

                factory.original = unexpected_build
                try:
                    probe.call()
                    npu.synchronize()
                    probe.check()
                finally:
                    factory.original = original
        assert len(set(counts.values())) == 1 and counts["baseline"] > 0
        # Keep the real compile/cache insertion path. Reshaping creates a new
        # specialization; no compile sentinel and no cache dictionary edits.
        # Registration of a new artifact requires finalized native runs, not
        # just device synchronization. Keep the diagnostic resolver installed.
        probe.state.drain()
        before = len(add_kernel._cache)
        shaped = tuple(tensor.reshape(64, 64) for tensor in probe.inputs)
        assert add_kernel(*shaped) is shaped[2]
        npu.synchronize()
        torch.testing.assert_close(shaped[2].cpu(), probe.expected.reshape(64, 64), rtol=0, atol=0)
        assert len(add_kernel._cache) == before + 1
        finish()
        prepare("baseline")
        assert add_kernel(*shaped) is shaped[2]
        npu.synchronize()
        assert len(add_kernel._cache) == before + 1
        torch.testing.assert_close(shaped[2].cpu(), probe.expected.reshape(64, 64), rtol=0, atol=0)
        (output / "live-checks.json").write_text(
            json.dumps(
                {
                    "annotation_evaluations": counts,
                    "warm_key_hit": "original constructor replaced by raising sentinel for one warm call",
                    "shape_miss": "real compilation; original resolver then hits the same artifact",
                },
                indent=2,
            )
        )
    finally:
        annotations["x"] = old
        del namespace[name]
        finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checks.json").write_text(json.dumps(check_inputs() | check_expression_order(), indent=2))
    clone, transformed = resolver_function(False), resolver_function(True)
    if args.check_only:
        return
    if args.device is None or decorator.capture_cache_config(None).enabled:
        parser.error("specify --device and disable persistent cache with PYPTO_CACHE=0")
    if os.environ.get("LD_PRELOAD") or os.environ.get("PYPTO_LAUNCH_PROBE_OUTPUT"):
        parser.error("formal timing requires an uninstrumented process")
    from hbg_eager_latency import (  # noqa: PLC0415
        FixedProbe,
        _DiagnosticPatches,
        add_kernel,
        configure_logging,
        measure,
        npu,
        torch,
    )

    configure_logging("error")
    torch.set_num_threads(1)
    npu.set_device(args.device)
    npu.set_compile_mode(jit_compile=False)
    probe = FixedProbe(args.device)
    for name in ("_hbg_direct_inputs", "_hbg_direct_key"):
        assert not hasattr(decorator, name)
        setattr(decorator, name, None)
    patches = _DiagnosticPatches()
    prepare, finish = probe.prepare, probe.finish
    stages = ("baseline", "clone", "direct", "infos", "key", "aclnn")

    def prepare_variant(stage: str) -> None:
        patches.restore()
        prepare("aclnn" if stage == "aclnn" else "jit")
        (probe.aclnn_out if stage == "aclnn" else probe.out).fill_(float("nan"))
        npu.synchronize()
        if stage in ("baseline", "aclnn"):
            return
        if stage != "clone":
            patches.set(decorator, "_hbg_direct_inputs", DirectInputs(stage in ("infos", "key")))
            patches.set(decorator, "_hbg_direct_key", DirectKey(stage == "key"))
        patches.set(
            add_kernel,
            "_resolve_compiled",
            types.MethodType(clone if stage == "clone" else transformed, add_kernel),
        )

    def finish_variant() -> None:
        try:
            finish()
        finally:
            patches.restore()

    probe.prepare, probe.finish = prepare_variant, finish_variant
    try:
        live_checks(probe, prepare_variant, finish_variant, args.output)
        measure(probe, args.output, args.samples, args.batches, stages)
    finally:
        try:
            probe.finish()
        finally:
            probe.prepare, probe.finish = prepare, finish
            try:
                probe.state.close()
            finally:
                delattr(decorator, "_hbg_direct_inputs")
                delattr(decorator, "_hbg_direct_key")


if __name__ == "__main__":
    main()
