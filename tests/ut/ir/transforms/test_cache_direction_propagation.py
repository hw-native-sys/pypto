# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Storage reads must survive interprocedural write-direction propagation."""

import ast
import json

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.backend.pto_backend import _generate_config_file
from pypto.ir._artifact_metadata import ArtifactMetadata
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.op import tile as tile_ops
from pypto.ir.param_info import _extract_func_param_infos


def _cache_program(*, reader_first=False, use_submit=False, **options):
    """A current-row update must not discard another historical cache row."""
    span = ir.Span.unknown()
    dtype = options.get("dtype", DataType.FP32)
    dynamic = options.get("dynamic", False)
    rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), span) if dynamic else 2
    extent = rows * 32 if dynamic else 64
    tensor_type = ir.TensorType([extent, 1], dtype)
    cache_type = ir.TensorType([rows, 32, 1, 1], dtype) if options.get("reshape") else tensor_type

    def leaf(name, read, direction):
        source = ir.Var(name + "_source", tensor_type, span)
        destination = ir.Var(name + "_destination", tensor_type, span)
        if read and options.get("gather"):
            create = tile_ops.create([1, 1], dtype, target_memory=ir.MemorySpace.Vec)
            acc = ir.Var(name + "_acc", create.type, span)
            load = tile_ops.gather_row(acc, source, [0, 0], [32, 0], [1, 1])
            prefix = [ir.AssignStmt(acc, create, span)]
        else:
            # A declared Out scratch is fully produced before its later read.
            # Plain-In cache cases still update only half, so their untouched
            # historical rows must survive the host-to-device transfer.
            full_output = (
                not read
                and direction == ir.ParamDirection.Out
                and options.get("cache_direction") == ir.ParamDirection.Out
            )
            load = tile_ops.load(source, [32 if read else 0, 0], [64 if full_output else 32, 1])
            prefix = []
        tile = ir.Var(name + "_tile", load.type, span)
        body = ir.SeqStmts(
            [
                *prefix,
                ir.AssignStmt(tile, load, span),
                ir.EvalStmt(tile_ops.store(tile, [0, 0], destination), span),
                ir.ReturnStmt([], span),
            ],
            span,
        )
        return ir.Function(
            name,
            [(source, ir.ParamDirection.In), (destination, direction)],
            [],
            body,
            span,
            type=options.get("leaf_type", ir.FunctionType.InCore),
        )

    writer = leaf("writer", False, options.get("writer_direction", ir.ParamDirection.Out))
    reader = leaf("reader", True, ir.ParamDirection.Out)
    x = ir.Var("x", tensor_type, span)
    cache = ir.Var("cache", cache_type, span)
    out = ir.Var("out", tensor_type, span)

    def dispatch(name, args):
        call = (
            ir.Submit(
                ir.GlobalVar(name),
                args,
                [],
                {},
                {},
                ir.TupleType([ir.ScalarType(DataType.TASK_ID)]),
                span,
            )
            if use_submit
            else ir.Call(ir.GlobalVar(name), args, ir.UnknownType(), span)
        )
        return ir.EvalStmt(call, span)

    functions = [writer, reader]
    cache_arg = cache
    prefix = []
    if options.get("reshape"):
        view = tensor_ops.reshape(cache, [extent, 1])
        cache_arg = ir.Var("cache_flat", view.type, span)
        prefix.append(ir.AssignStmt(cache_arg, view, span))
    reader_name = "reader"
    if wrapper_type := options.get("wrapper_type"):
        a = ir.Var("wrapped_cache", tensor_type, span)
        b = ir.Var("wrapped_out", tensor_type, span)
        wrapper = ir.Function(
            "z_reader_wrapper",
            [(a, ir.ParamDirection.In), (b, ir.ParamDirection.Out)],
            [],
            ir.SeqStmts([dispatch("reader", [a, b]), ir.ReturnStmt([], span)], span),
            span,
            type=wrapper_type,
        )
        functions.append(wrapper)
        reader_name = wrapper.name
    writer_name = "writer"
    if options.get("writer_wrapper"):
        a = ir.Var("wrapper_source", tensor_type, span)
        b = ir.Var("wrapper_destination", tensor_type, span)
        wrapper = ir.Function(
            "z_writer_wrapper",
            [a, b],
            [],
            ir.SeqStmts([dispatch("writer", [a, b]), ir.ReturnStmt([], span)], span),
            span,
            type=ir.FunctionType.Opaque,
        )
        functions.append(wrapper)
        writer_name = wrapper.name
    statements = []
    if options.get("include_writer", True):
        statements.append(dispatch(writer_name, [x, cache_arg]))
    if options.get("include_reader", True):
        statements.append(dispatch(reader_name, [cache_arg, out]))
    if reader_first:
        statements.reverse()
    if options.get("descriptor_only"):
        statements.append(ir.EvalStmt(tensor_ops.dim(cache_arg, 0), span))
    statements: list[ir.Stmt] = [*prefix, *statements, ir.ReturnStmt([], span)]
    main = ir.Function(
        "main",
        [
            (x, ir.ParamDirection.In),
            (cache, options.get("cache_direction", ir.ParamDirection.In)),
            (out, ir.ParamDirection.Out),
        ],
        [],
        ir.SeqStmts(statements, span),
        span,
        type=ir.FunctionType.Orchestration,
    )
    return ir.Program([main, *functions], "cache_history", span)


@pytest.mark.parametrize("reader_first", [False, True])
@pytest.mark.parametrize("use_submit", [False, True])
def test_partial_writer_and_separate_reader_preserve_cache_input(reader_first, use_submit, tmp_path):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    before = _cache_program(reader_first=reader_first, use_submit=use_submit)
    with passes.PassContext([]):
        after = passes.convert_tensor_to_tile_ops()(before)
    (tmp_path / "before.py").write_text(ir.python_print(before))
    (tmp_path / "after.py").write_text(ir.python_print(after))
    main = after.get_function("main")
    assert main is not None
    directions = {p.name_hint: d for p, d in zip(main.params, main.param_directions)}
    assert directions == {
        "x": ir.ParamDirection.In,
        "cache": ir.ParamDirection.InOut,
        "out": ir.ParamDirection.Out,
    }, directions


def _lower(program):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([]):
        return passes.convert_tensor_to_tile_ops()(program)


def _directions(program):
    main = program.get_function("main")
    assert main is not None
    return {p.name_hint: d for p, d in zip(main.params, main.param_directions)}


@pytest.mark.parametrize("dtype", [DataType.FP32, DataType.BF16, DataType.INT8])
@pytest.mark.parametrize("reader_first", [False, True])
@pytest.mark.parametrize("use_submit", [False, True])
def test_dynamic_reshape_gather_history(dtype, reader_first, use_submit):
    """The gather DPS result aliases its tile destination, never the GM source."""
    before = _cache_program(
        reshape=True,
        dynamic=True,
        gather=True,
        dtype=dtype,
        reader_first=reader_first,
        use_submit=use_submit,
    )
    after = _lower(before)
    assert _directions(after) == {
        "x": ir.ParamDirection.In,
        "cache": ir.ParamDirection.InOut,
        "out": ir.ParamDirection.Out,
    }
    # Keep the original shaped view and the actual gather source, not just an
    # ABI label that could pass after dropping the historical reader entirely.
    text = ir.python_print(after)
    assert "reshape(" in text and "tile.gather_row(" in text, text
    assert "self.reader" in text and "cache_flat" in text, text


@pytest.mark.parametrize("wrapper_type", [ir.FunctionType.Group, ir.FunctionType.Spmd])
@pytest.mark.parametrize("use_submit", [False, True])
def test_reader_through_later_named_wrapper(wrapper_type, use_submit):
    after = _lower(
        _cache_program(
            wrapper_type=wrapper_type,
            use_submit=use_submit,
            reshape=True,
            dynamic=True,
        )
    )
    assert _directions(after)["cache"] == ir.ParamDirection.InOut
    wrapper = after.get_function("z_reader_wrapper")
    assert wrapper is not None
    assert wrapper.param_directions == [ir.ParamDirection.In, ir.ParamDirection.Out]


@pytest.mark.parametrize("use_submit", [False, True])
@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"cache_direction": ir.ParamDirection.Out}, ir.ParamDirection.Out),
        (
            {"cache_direction": ir.ParamDirection.Out, "writer_direction": ir.ParamDirection.InOut},
            ir.ParamDirection.InOut,
        ),
        ({"cache_direction": ir.ParamDirection.InOut}, ir.ParamDirection.InOut),
        ({"include_reader": False}, ir.ParamDirection.Out),
        ({"include_reader": False, "descriptor_only": True, "reshape": True}, ir.ParamDirection.Out),
        ({"include_writer": False}, ir.ParamDirection.In),
        ({"include_reader": False, "writer_wrapper": True}, ir.ParamDirection.Out),
    ],
)
def test_declared_contracts_and_no_false_reads(options, expected, use_submit):
    """Do not turn fresh Out scratch or an unresolved pure writer into an input."""
    after = _lower(_cache_program(use_submit=use_submit, **options))
    assert _directions(after) == {
        "x": ir.ParamDirection.In,
        "cache": expected,
        "out": ir.ParamDirection.Out,
    }


def test_lowered_metadata_and_native_signature_preserve_history(tmp_path):
    before = _cache_program(leaf_type=ir.FunctionType.AIV)
    after = _lower(before)
    logical_main = before.get_function("main")
    lowered_main = after.get_function("main")
    assert logical_main is not None and lowered_main is not None
    logical, _, _ = _extract_func_param_infos(logical_main)
    lowered, _, _ = _extract_func_param_infos(lowered_main)
    metadata = ArtifactMetadata(
        tuple(logical),
        tuple(lowered),
        0,
        "a2a3",
        BackendType.Ascend910,
    )
    metadata.write(tmp_path)
    # Run real orchestration codegen and the native-config serializer. Comparing
    # two independently generated ABI views is insufficient if both say Out, so
    # the expected history-read contract is checked separately below.
    with passes.PassContext([]):
        ready = passes.normalize_return_order()(after)
        ready = passes.derive_call_directions()(ready)
        ready = passes.materialize_runtime_scopes()(ready)
        ready = passes.classify_iter_arg_carry()(ready)
    ready_main = ready.get_function("main")
    assert ready_main is not None
    result = codegen.generate_orchestration(ready, ready_main)
    text = _generate_config_file(
        orch_func_name="main",
        kernel_registrations=list(result.kernel_registrations),
        orchestration_signature=list(result.orchestration_signature),
    )
    tree = ast.parse(text)
    orch = next(
        item.value
        for item in tree.body
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ORCHESTRATION" for target in item.targets)
    )
    assert isinstance(orch, ast.Dict)
    signature = next(
        value
        for key, value in zip(orch.keys, orch.values)
        if isinstance(key, ast.Constant) and key.value == "signature"
    )
    assert isinstance(signature, ast.List)
    native = []
    for item in signature.elts:
        assert isinstance(item, ast.Attribute)
        native.append(item.attr)
    persisted = json.loads((tmp_path / "compiled_meta.json").read_text())
    mapping = {"In": "IN", "Out": "OUT", "InOut": "INOUT"}
    assert native == [mapping[item["direction"]] for item in persisted["lowered_params"]]
    assert native == ["IN", "INOUT", "OUT"]
    assert persisted["params"][1]["direction"] == "In"
    assert persisted["lowered_params"][1]["direction"] == "InOut"


@pytest.mark.parametrize("trip_count", [0, 1, 2])
def test_repeated_descriptor_swap_preserves_post_loop_read(trip_count):
    @pl.program
    class Before:
        @pl.function
        def main(self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]):
            for i, (left, right) in pl.range(trip_count, init_values=(a, b)):
                first, second = pl.yield_(right, left)
            value = pl.tile.load(first, [32], [32], [32], target_memory=pl.Mem.Vec)
            _updated = pl.tile.store(value, [0], a)

    @pl.program
    class Expected:
        @pl.function
        def main(self, a: pl.InOut[pl.Tensor[[64], pl.FP32]], b: pl.Tensor[[64], pl.FP32]):
            for i, (left, right) in pl.range(trip_count, init_values=(a, b)):
                first, second = pl.yield_(right, left)
            value = pl.tile.load(first, [32], [32], [32], target_memory=pl.Mem.Vec)
            _updated = pl.tile.store(value, [0], a)

    after = _lower(Before)
    if trip_count == 1:
        assert _directions(after)["a"] == ir.ParamDirection.Out
    else:
        ir.assert_structural_equal(after, Expected)


@pytest.mark.parametrize("trip_count", [1, 2])
def test_fresh_loop_seed_still_transfers_later_iteration_reads(trip_count):
    @pl.program
    class Before:
        @pl.function
        def main(self, seed: pl.Tensor[[64], pl.FP32], cache: pl.Tensor[[64], pl.FP32]):
            fresh = pl.tensor.add(seed, seed)
            for i, (carry,) in pl.range(trip_count, init_values=(fresh,)):
                value = pl.tile.load(carry, [32], [32], [32], target_memory=pl.Mem.Vec)
                updated = pl.tile.store(value, [0], cache)
                _result = pl.yield_(updated)

    @pl.program
    class Expected:
        @pl.function
        def main(self, seed: pl.Tensor[[64], pl.FP32], cache: pl.InOut[pl.Tensor[[64], pl.FP32]]):
            fresh = pl.tensor.add(seed, seed)
            for i, (carry,) in pl.range(trip_count, init_values=(fresh,)):
                value = pl.tile.load(carry, [32], [32], [32], target_memory=pl.Mem.Vec)
                updated = pl.tile.store(value, [0], cache)
                _result = pl.yield_(updated)

    after = _lower(Before)
    if trip_count == 1:
        assert _directions(after)["cache"] == ir.ParamDirection.Out
    else:
        ir.assert_structural_equal(after, Expected)


def test_slice_write_reaches_parent_storage():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def write(self, x: pl.Tensor[[32], pl.FP32], out: pl.Out[pl.Tensor[[32], pl.FP32]]):
            value = pl.tile.load(x, [0], [32], [32], target_memory=pl.Mem.Vec)
            _updated = pl.tile.store(value, [0], out)
            return ()

        @pl.function
        def main(self, x: pl.Tensor[[32], pl.FP32], cache: pl.Tensor[[64], pl.FP32]):
            view = pl.tensor.slice(cache, [32], [32])
            self.write(x, view)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def write(self, x: pl.Tensor[[32], pl.FP32], out: pl.Out[pl.Tensor[[32], pl.FP32]]):
            value = pl.tile.load(x, [0], [32], [32], target_memory=pl.Mem.Vec)
            _updated = pl.tile.store(value, [0], out)
            return ()

        @pl.function
        def main(self, x: pl.Tensor[[32], pl.FP32], cache: pl.InOut[pl.Tensor[[64], pl.FP32]]):
            view = pl.tensor.slice(cache, [32], [32])
            self.write(x, view)

    ir.assert_structural_equal(_lower(Before), Expected)


def test_fresh_callee_return_does_not_read_its_out_scratch():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def produce(
            self, seed: pl.Tensor[[64], pl.FP32], scratch: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> pl.Tensor[[64], pl.FP32]:
            fresh = pl.tensor.add(seed, seed)
            return fresh

        @pl.function
        def main(self, seed: pl.Tensor[[64], pl.FP32], scratch: pl.Tensor[[64], pl.FP32]):
            with pl.graph("layer"):
                fresh = self.produce(seed, scratch)
                _value = pl.tile.load(fresh, [0], [32], [32], target_memory=pl.Mem.Vec)

    with passes.PassContext([]):
        after = passes.outline_graph_scopes()(Before)
    layer = after.get_function("layer")
    assert layer is not None
    assert layer.param_directions == [ir.ParamDirection.In, ir.ParamDirection.Out]


def test_same_typed_outputs_follow_proven_callee_return():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def produce(
            self, a: pl.Out[pl.Tensor[[64], pl.FP32]], b: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> pl.Tensor[[64], pl.FP32]:
            return b

        @pl.function
        def main(self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]):
            result = self.produce(a, b)
            _value = pl.tile.load(result, [32], [32], [32], target_memory=pl.Mem.Vec)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def produce(
            self, a: pl.Out[pl.Tensor[[64], pl.FP32]], b: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> pl.Tensor[[64], pl.FP32]:
            return b

        @pl.function
        def main(self, a: pl.Out[pl.Tensor[[64], pl.FP32]], b: pl.InOut[pl.Tensor[[64], pl.FP32]]):
            result = self.produce(a, b)
            _value = pl.tile.load(result, [32], [32], [32], target_memory=pl.Mem.Vec)

    ir.assert_structural_equal(_lower(Before), Expected)


def test_partial_cache_writer_and_reader_structural_contract():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def write(self, x: pl.Tensor[[64], pl.FP32], cache: pl.Out[pl.Tensor[[64], pl.FP32]]):
            tile = pl.tile.load(x, [0], [32], [32], target_memory=pl.Mem.Vec)
            _updated = pl.tile.store(tile, [0], cache)
            return ()

        @pl.function(type=pl.FunctionType.AIV)
        def read(self, cache: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
            tile = pl.tile.load(cache, [32], [32], [32], target_memory=pl.Mem.Vec)
            _updated = pl.tile.store(tile, [0], out)
            return ()

        @pl.function
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            cache: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ):
            self.write(x, cache)
            self.read(cache, out)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def write(self, x: pl.Tensor[[64], pl.FP32], cache: pl.Out[pl.Tensor[[64], pl.FP32]]):
            tile = pl.tile.load(x, [0], [32], [32], target_memory=pl.Mem.Vec)
            _updated = pl.tile.store(tile, [0], cache)
            return ()

        @pl.function(type=pl.FunctionType.AIV)
        def read(self, cache: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
            tile = pl.tile.load(cache, [32], [32], [32], target_memory=pl.Mem.Vec)
            _updated = pl.tile.store(tile, [0], out)
            return ()

        @pl.function
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            cache: pl.InOut[pl.Tensor[[64], pl.FP32]],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ):
            self.write(x, cache)
            self.read(cache, out)

    ir.assert_structural_equal(_lower(Before), Expected)


@pytest.mark.parametrize("trip_count", [1, 2, 3])
def test_callee_loop_return_preserves_repeated_storage_roots(trip_count):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Graph)
        def swap(
            self, a: pl.Out[pl.Tensor[[64], pl.FP32]], b: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> pl.Tensor[[64], pl.FP32]:
            for i, (left, right) in pl.range(trip_count, init_values=(a, b)):
                first, _second = pl.yield_(right, left)
            return first

        @pl.function
        def main(self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]):
            result = self.swap(a, b)
            _value = pl.tile.load(result, [32], [32], [32], target_memory=pl.Mem.Vec)

    after = _lower(Before)
    assert _directions(after)["a" if trip_count == 2 else "b"] == ir.ParamDirection.InOut


def test_graph_carry_seed_expression_preserves_input_contents():
    @pl.program
    class Before:
        @pl.function
        def main(self, cache: pl.Tensor[[2, 32], pl.FP32]):
            with pl.graph("layer"):
                for i, (carry,) in pl.range(1, init_values=(pl.tensor.reshape(cache, [64]),)):
                    value = pl.tensor.read(carry, [32])
                    pl.tensor.write(cache, [0, 0], value)
                    _result = pl.yield_(carry)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.Graph, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def layer(self, cache: pl.InOut[pl.Tensor[[2, 32], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            for i, (carry,) in pl.range(1, init_values=(pl.tensor.reshape(cache, [64]),)):
                value = pl.tensor.read(carry, [32])
                pl.tensor.write(cache, [0, 0], value)
                _result = pl.yield_(carry)
            return _result

        @pl.function
        def main(self, cache: pl.Tensor[[2, 32], pl.FP32]):
            _result = self.layer(cache)

    with passes.PassContext([]):
        after = passes.outline_graph_scopes()(Before)
    ir.assert_structural_equal(after, Expected)


def test_loop_yield_tuple_projection_preserves_input_contents():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def update_pair(
            self, cache: pl.Out[pl.Tensor[[64], pl.FP32]], other: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            pl.tensor.write(cache, [0], pl.const(1.0, pl.FP32))
            return cache, other

        @pl.function
        def main(self, cache: pl.Tensor[[64], pl.FP32], other: pl.Tensor[[64], pl.FP32]):
            for i, (carry,) in pl.range(1, init_values=(cache,)):
                pair = self.update_pair(cache, other)
                result = pl.yield_(pair[0])
            value = pl.tensor.read(result, [32])
            pl.tensor.write(other, [0], value)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def update_pair(
            self, cache: pl.Out[pl.Tensor[[64], pl.FP32]], other: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            pl.tensor.write(cache, [0], pl.const(1.0, pl.FP32))
            return cache, other

        @pl.function
        def main(self, cache: pl.InOut[pl.Tensor[[64], pl.FP32]], other: pl.Out[pl.Tensor[[64], pl.FP32]]):
            for i, (carry,) in pl.range(1, init_values=(cache,)):
                pair = self.update_pair(cache, other)
                result = pl.yield_(pair[0])
            value = pl.tensor.read(result, [32])
            pl.tensor.write(other, [0], value)

    ir.assert_structural_equal(_lower(Before), Expected)


def test_tuple_valued_return_preserves_element_storage():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def update_pair(
            self, a: pl.Out[pl.Tensor[[64], pl.FP32]], b: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            pl.tensor.write(a, [0], pl.const(1.0, pl.FP32))
            pl.tensor.write(b, [0], pl.const(2.0, pl.FP32))
            return a, b

        @pl.function(type=pl.FunctionType.Group)
        def forward(
            self, a: pl.Out[pl.Tensor[[64], pl.FP32]], b: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            packed = self.update_pair(a, b)
            return packed

        @pl.function
        def main(self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]):
            packed = self.forward(a, b)
            selected = packed[1]
            value = pl.tensor.read(selected, [32])
            pl.tensor.write(a, [0], value)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def update_pair(
            self, a: pl.Out[pl.Tensor[[64], pl.FP32]], b: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            pl.tensor.write(a, [0], pl.const(1.0, pl.FP32))
            pl.tensor.write(b, [0], pl.const(2.0, pl.FP32))
            return a, b

        @pl.function(type=pl.FunctionType.Group)
        def forward(
            self, a: pl.Out[pl.Tensor[[64], pl.FP32]], b: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            packed = self.update_pair(a, b)
            return packed

        @pl.function
        def main(self, a: pl.Out[pl.Tensor[[64], pl.FP32]], b: pl.InOut[pl.Tensor[[64], pl.FP32]]):
            packed = self.forward(a, b)
            selected = packed[1]
            value = pl.tensor.read(selected, [32])
            pl.tensor.write(a, [0], value)

    ir.assert_structural_equal(_lower(Before), Expected)


def test_shared_callee_transition_uses_each_branch_argument_binding():
    tensor = "pl.Tensor[[64], pl.FP32]"
    params = (
        f"condition: pl.Scalar[pl.BOOL], a: pl.Out[{tensor}], b: pl.Out[{tensor}], scratch: pl.Out[{tensor}]"
    )
    source = f"""import pypto.language as pl

@pl.program
class BranchCalls:
    @pl.function(type=pl.FunctionType.Graph)
    def layer0(self, {params}) -> {tensor}:
        for i, (left, right) in pl.range(1, init_values=(a, b)):
            first, second = pl.yield_(right, left)
        return first
"""
    for level in range(1, 7):
        source += f"""
    @pl.function(type=pl.FunctionType.Graph)
    def layer{level}(self, {params}) -> {tensor}:
        if condition:
            left = self.layer{level - 1}(condition, a, b, scratch)
            selected = pl.yield_(left)
        else:
            right = self.layer{level - 1}(condition, b, a, scratch)
            selected = pl.yield_(right)
        return selected
"""
    source += f"""
    @pl.function
    def main(self, condition: pl.Scalar[pl.BOOL], a: {tensor}, b: {tensor}, scratch: {tensor}):
        result = self.layer6(condition, a, b, scratch)
        value = pl.tensor.read(result, [32])
        pl.tensor.write(a, [0], value)
"""
    directions = _directions(_lower(pl.parse(source)))
    assert directions["a"] == ir.ParamDirection.InOut
    assert directions["b"] == ir.ParamDirection.InOut
    assert directions["scratch"] == ir.ParamDirection.Out


def test_graph_branch_view_yields_preserve_backing_storage():
    @pl.program
    class Before:
        @pl.function
        def main(self, condition: pl.Scalar[pl.BOOL], cache: pl.Tensor[[2, 32], pl.FP32]):
            with pl.graph("layer"):
                if condition:
                    selected = pl.yield_(pl.tensor.reshape(cache, [64]))
                else:
                    selected = pl.yield_(pl.tensor.reshape(cache, [64]))
                value = pl.tensor.read(selected, [32])
                pl.tensor.write(cache, [0, 0], value)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.Graph, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def layer(
            self, condition: pl.Scalar[pl.BOOL], cache: pl.InOut[pl.Tensor[[2, 32], pl.FP32]]
        ) -> pl.Tensor[[64], pl.FP32]:
            if condition:
                selected = pl.yield_(pl.tensor.reshape(cache, [64]))
            else:
                selected = pl.yield_(pl.tensor.reshape(cache, [64]))
            value = pl.tensor.read(selected, [32])
            pl.tensor.write(cache, [0, 0], value)
            return selected

        @pl.function
        def main(self, condition: pl.Scalar[pl.BOOL], cache: pl.Tensor[[2, 32], pl.FP32]):
            _selected = self.layer(condition, cache)

    with passes.PassContext([]):
        after = passes.outline_graph_scopes()(Before)
    ir.assert_structural_equal(after, Expected)


def test_graph_inline_view_read_preserves_input_contents():
    @pl.program
    class Before:
        @pl.function
        def main(self, cache: pl.Tensor[[2, 32], pl.FP32]):
            with pl.graph("layer"):
                value = pl.tensor.read(pl.tensor.reshape(cache, [64]), [32])
                pl.tensor.write(cache, [0, 0], value)

    with passes.PassContext([]):
        after = passes.outline_graph_scopes()(Before)
    layer = after.get_function("layer")
    assert layer is not None
    assert layer.param_directions == [ir.ParamDirection.InOut]


def test_graph_branch_tuple_yields_preserve_element_storage():
    source = """
import pypto.language as pl

@pl.program
class Before:
    @pl.function
    def main(self, condition: pl.Scalar[pl.BOOL], cache: pl.Tensor[[2, 32], pl.FP32]):
        with pl.graph("layer"):
            if condition:
                packed = pl.yield_((pl.tensor.reshape(cache, [64]), pl.tensor.reshape(cache, [64])))
            else:
                packed = pl.yield_((pl.tensor.reshape(cache, [64]), pl.tensor.reshape(cache, [64])))
            selected = packed[0]
            value = pl.tensor.read(selected, [32])
            pl.tensor.write(cache, [0, 0], value)
"""
    before = pl.parse(source)
    assert isinstance(before, ir.Program)
    with passes.PassContext([]):
        after = passes.outline_graph_scopes()(before)
    layer = after.get_function("layer")
    assert layer is not None
    directions = {
        param.name_hint: direction for param, direction in zip(layer.params, layer.param_directions)
    }
    assert directions["cache"] == ir.ParamDirection.InOut


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
