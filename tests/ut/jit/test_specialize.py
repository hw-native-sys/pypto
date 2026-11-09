# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``JITFunction.specialize`` and the parameter-direction accessors.

``specialize()`` is the pre-pass half of ``lower()``: it returns the parsed
program before any pass has run, which is what a consumer driving the pass
pipeline itself needs — ``ir.compile(program, output_dir=...)`` runs passes and
code generation together, so handing it ``lower()``'s output would run the
pipeline twice.

The system-test harness is that consumer: it builds every case's IR through
this method, whichever surface authored the kernel.
"""

from typing import Any

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit import _tensor_meta
from pypto.jit.decorator import _discover_dep_bindings, jit
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.pypto_core import DataType, ir, passes
from pypto.pypto_core.ir import MemorySpace
from pypto.runtime.runner import RunConfig

M = 16
N = 16


@jit.incore
def _abs_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    tile_a = pl.load(a, [0, 0], [M, N])
    return pl.store(pl.tile.abs(tile_a), [0, 0], out)


@jit
def _abs_entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = _abs_kernel(a, out)
    return out


@jit.incore
def _shaped_kernel(a: pl.Tensor[[M, N], pl.FP32], out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
    tile_a = pl.load(a, [0, 0], [M, N])
    return pl.store(pl.tile.abs(tile_a), [0, 0], out)


@jit
def _shaped_entry(a: pl.Tensor[[M, N], pl.FP32], out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
    out = _shaped_kernel(a, out)
    return out


@jit
def _mixed_directions(
    x: pl.Tensor, acc: pl.InOut[pl.Tensor], y: pl.Tensor, z: pl.Out[pl.Tensor]
):  # pragma: no cover - never specialized, only its signature is read
    return acc


@pl.program
class _AbsRef:
    """Hand-written equivalent of ``_abs_entry``."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [M, N])
        out = pl.store(pl.tile.abs(tile_a), [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        out = self.kernel(a, out)
        return out


# ---------------------------------------------------------------------------
# Free names a body inherits from its module or an enclosing function
# ---------------------------------------------------------------------------

_CAST_MODE = "trunc"
_CAST_DTYPE = pl.INT8
_LOAD_MEM = pl.Mem.Vec
_TILE_SHAPE = [1, N]


@jit.incore
def _free_name_kernel(
    a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
) -> pl.Tensor[[1, N], pl.INT8]:
    tile_a = pl.load(a, [0, 0], _TILE_SHAPE, target_memory=_LOAD_MEM)
    quantized = pl.cast(tile_a, _CAST_DTYPE, mode=_CAST_MODE)
    return pl.store(quantized, [0, 0], out)


@jit
def _free_name_entry(
    a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
) -> pl.Tensor[[1, N], pl.INT8]:
    return _free_name_kernel(a, out)


# Deliberately typed ``Any``: this stands for any value the renderer cannot write
# as source, and the point of the test is what the *specializer* does with it.
_OPAQUE: Any = object()


@jit.incore
def _opaque_name_kernel(
    a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
) -> pl.Tensor[[1, N], pl.INT8]:  # pragma: no cover - specialization is expected to fail
    tile_a = pl.load(a, [0, 0], [1, N], target_memory=_OPAQUE)
    quantized = pl.cast(tile_a, pl.INT8, mode="trunc")
    return pl.store(quantized, [0, 0], out)


@jit
def _opaque_name_entry(
    a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
) -> pl.Tensor[[1, N], pl.INT8]:  # pragma: no cover - specialization is expected to fail
    return _opaque_name_kernel(a, out)


def _make_parameterized_entry(mode: str, dtype):
    """A kernel factory — the shape every parameterized test suite wants to write."""

    @jit.incore
    def kernel(
        a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
    ) -> pl.Tensor[[1, N], pl.INT8]:
        tile_a = pl.load(a, [0, 0], [1, N])
        quantized = pl.cast(tile_a, dtype, mode=mode)
        return pl.store(quantized, [0, 0], out)

    @jit
    def entry(
        a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
    ) -> pl.Tensor[[1, N], pl.INT8]:
        return kernel(a, out)

    return entry


def _find_call(program: ir.Program, op_name: str) -> ir.Call:
    """The single call to ``op_name`` in ``program``."""
    found: list[ir.Call] = []
    target = ir.get_op(op_name).name

    class _Collector(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == target:
                found.append(op)
            super().visit_call(op)

    _Collector().visit_program(program)
    assert len(found) == 1, f"expected exactly one {op_name}, got {len(found)}"
    return found[0]


class TestFreeNameResolution:
    """A body may name a value defined in its module or an enclosing function.

    The generated ``@pl.program`` source is parsed in a namespace holding only
    ``pl`` and ``pld``, so each such name has to be replaced by source text that
    evaluates back to the same value. Before that covered strings, dtypes, enums
    and sequences, only ``int``/``float``/``bool`` folded and everything else
    failed with "Cannot resolve expression" / "Undefined variable" — which made a
    parameterized kernel factory impossible to write.
    """

    def test_module_constants_of_every_renderable_kind_resolve(self):
        """One kernel naming a str, a DataType, an enum and a list constant."""
        program = _free_name_entry.specialize()

        cast_call = _find_call(program, "tile.cast")
        assert cast_call.kwargs["mode"] == 5, "the str constant reached the op as trunc"
        assert cast_call.kwargs["target_type"] == DataType.INT8, "the DataType constant survived"

        load_call = _find_call(program, "tile.load")
        assert load_call.kwargs["target_memory"] == MemorySpace.Vec, "the enum constant survived"
        loaded = load_call.type
        assert isinstance(loaded, ir.TileType)
        shape = [dim.value for dim in loaded.shape if isinstance(dim, ir.ConstInt)]
        assert shape == [1, N], "the list constant became the load shape"

    def test_an_unrenderable_constant_still_fails_loudly(self):
        """A value with no source form must keep reporting the name, not fold to something wrong."""
        with pytest.raises(ParserTypeError, match="Cannot resolve expression '_OPAQUE'"):
            _opaque_name_entry.specialize()

    @pytest.mark.parametrize(
        "mode, dtype, expected_mode, expected_dtype",
        [
            ("trunc", pl.INT8, 5, DataType.INT8),
            ("round", pl.INT16, 2, DataType.INT16),
        ],
    )
    def test_a_factory_can_parameterize_its_kernel(self, mode, dtype, expected_mode, expected_dtype):
        """Values captured from the factory's frame resolve like module ones."""
        program = _make_parameterized_entry(mode, dtype).specialize()
        cast_call = _find_call(program, "tile.cast")
        assert cast_call.kwargs["mode"] == expected_mode
        assert cast_call.kwargs["target_type"] == expected_dtype


_GLOBAL_REDUCE_OP = pld.ReduceOp.Sum


@pl.jit
def _initialize_allreduce(data: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        data = pl.store(pl.tile.full([1, 256], pl.FP32, 1.0), [0, 0], data)
    return data


def _global_allreduce():
    data_buf = pld.alloc_window_buffer(256 * 4)
    signal_buf = pld.alloc_window_buffer(2 * 4)
    for rank in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
        _initialize_allreduce(data, device=rank)
    data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
    signal = pld.window(signal_buf, [2], dtype=pl.INT32)
    data = pld.tensor.allreduce(data, signal, op=_GLOBAL_REDUCE_OP)


def _make_allreduce(reduce_op):
    @pl.jit.host
    def host():
        data_buf = pld.alloc_window_buffer(256 * 4)
        signal_buf = pld.alloc_window_buffer(2 * 4)
        for rank in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
            _initialize_allreduce(data, device=rank)
        data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
        signal = pld.window(signal_buf, [2], dtype=pl.INT32)
        data = pld.tensor.allreduce(data, signal, op=reduce_op)

    def set_op(value):
        nonlocal reduce_op
        reduce_op = value

    return host, set_op


@pl.jit.host
def _literal_allreduce_sum():
    data_buf = pld.alloc_window_buffer(256 * 4)
    signal_buf = pld.alloc_window_buffer(2 * 4)
    for rank in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
        _initialize_allreduce(data, device=rank)
    data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
    signal = pld.window(signal_buf, [2], dtype=pl.INT32)
    data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)


@pl.jit.host
def _literal_allreduce_max():
    data_buf = pld.alloc_window_buffer(256 * 4)
    signal_buf = pld.alloc_window_buffer(2 * 4)
    for rank in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
        _initialize_allreduce(data, device=rank)
    data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
    signal = pld.window(signal_buf, [2], dtype=pl.INT32)
    data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Max)


@pl.jit.host
def _literal_allreduce_min():
    data_buf = pld.alloc_window_buffer(256 * 4)
    signal_buf = pld.alloc_window_buffer(2 * 4)
    for rank in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
        _initialize_allreduce(data, device=rank)
    data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
    signal = pld.window(signal_buf, [2], dtype=pl.INT32)
    data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Min)


@pl.jit.host
def _literal_allreduce_prod():
    data_buf = pld.alloc_window_buffer(256 * 4)
    signal_buf = pld.alloc_window_buffer(2 * 4)
    for rank in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
        _initialize_allreduce(data, device=rank)
    data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
    signal = pld.window(signal_buf, [2], dtype=pl.INT32)
    data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Prod)


class TestDistributedEnumConstants:
    @pytest.fixture
    def pass_verification_instruments(self):
        # Distributed tensor dumps with absent memref/tensor_view cannot reparse,
        # including the literal controls. Keep full compilation and before/after
        # IR property checks; enum folding is checked independently above.
        return [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]

    @pytest.fixture(params=["closure", "global"])
    def allreduce(self, request, monkeypatch):
        if request.param == "closure":
            return _make_allreduce(pld.ReduceOp.Sum)

        def set_op(value):
            monkeypatch.setitem(_global_allreduce.__globals__, "_GLOBAL_REDUCE_OP", value)

        return pl.jit.host(_global_allreduce), set_op

    @pytest.fixture
    def config(self):
        return RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=[0, 1]))

    @pytest.mark.parametrize(
        "reduce_op, literal",
        [
            (pld.ReduceOp.Sum, _literal_allreduce_sum),
            (pld.ReduceOp.Max, _literal_allreduce_max),
            (pld.ReduceOp.Min, _literal_allreduce_min),
            (pld.ReduceOp.Prod, _literal_allreduce_prod),
        ],
    )
    def test_alias_matches_literal_through_compile(self, allreduce, config, reduce_op, literal):
        kernel, set_op = allreduce
        set_op(reduce_op)
        specialized = kernel.specialize()
        assert _find_call(specialized, "pld.tensor.allreduce").kwargs["op"] == int(reduce_op)
        control = literal.compile(config=config)
        compiled = kernel.compile(config=config)
        assert compiled.program is not None and control.program is not None
        actual_op = _find_call(compiled.program, "builtin.tensor.allreduce").kwargs["op"]
        literal_op = _find_call(control.program, "builtin.tensor.allreduce").kwargs["op"]
        assert actual_op == literal_op == int(reduce_op)

    def test_rebinding_enum_invalidates_cache(self, allreduce, config):
        kernel, set_op = allreduce
        hashes = set()
        compiled_by_op = {}
        for reduce_op in pld.ReduceOp:
            set_op(reduce_op)
            hashes.add(kernel._get_source_hash())
            compiled = kernel.compile(config=config)
            assert compiled.program is not None
            assert _find_call(compiled.program, "builtin.tensor.allreduce").kwargs["op"] == int(reduce_op)
            assert kernel.compile(config=config) is compiled
            compiled_by_op[reduce_op] = compiled
        assert len(hashes) == len(compiled_by_op) == 4
        assert len({id(compiled) for compiled in compiled_by_op.values()}) == 4
        set_op(pld.ReduceOp.Sum)
        assert kernel.compile(config=config) is compiled_by_op[pld.ReduceOp.Sum]


class TestSpecialize:
    """``specialize()`` returns a usable pre-pass program."""

    def test_returns_pre_pass_program(self):
        """The entry and its dep are both present, untransformed."""
        program = _abs_entry.specialize(torch.randn(M, N), torch.zeros(M, N))
        assert isinstance(program, ir.Program)
        assert len(list(program.functions)) == 2, "entry + its @pl.jit.incore dep"

    def test_matches_hand_written_program_after_passes(self):
        """The specialized program lowers to the same IR as the reference.

        Asserted after the pass pipeline, not before: the specializer renames
        SSA-rebound locals (``out`` becomes ``out_v1``), so the two programs are
        only textually different beforehand. Running both through the same
        strategy is the equivalence that matters — it is what the harness
        compiles.

        The JIT functions are declared here, rather than at module level,
        because function names are part of the IR: they must match the
        reference's ``kernel`` / ``orchestrator`` for the comparison to be
        about structure rather than about naming.
        """

        @jit.incore
        def kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            tile_a = pl.load(a, [0, 0], [M, N])
            return pl.store(pl.tile.abs(tile_a), [0, 0], out)

        @jit
        def orchestrator(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = kernel(a, out)
            return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        got = pm.run_passes(orchestrator.specialize(torch.randn(M, N), torch.zeros(M, N)))
        ir.assert_structural_equal(got, pm.run_passes(_AbsRef))

    def test_agrees_with_lower(self):
        """Running passes over ``specialize()`` reproduces ``lower()`` exactly."""
        a, out = torch.randn(M, N), torch.zeros(M, N)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        ir.assert_structural_equal(pm.run_passes(_abs_entry.specialize(a, out)), _abs_entry.lower(a, out))

    def test_does_not_populate_the_compiled_cache(self):
        """No compilation happens, so the L1 cache stays empty."""
        _abs_entry._cache.clear()
        _abs_entry.specialize(torch.randn(M, N), torch.zeros(M, N))
        assert len(_abs_entry._cache) == 0

    def test_signature_mode_needs_full_shapes(self):
        """A bare ``pl.Tensor`` cannot be specialized without a sample."""
        with pytest.raises(TypeError, match="bare 'pl.Tensor' annotation with no shape"):
            _abs_entry.specialize()

    def test_signature_mode_works_with_shaped_annotations(self):
        """Fully-shaped annotations specialize with no sample tensors at all."""
        program = _shaped_entry.specialize()
        assert isinstance(program, ir.Program)
        assert len(list(program.functions)) == 2

    def test_rejects_config(self):
        """``config=`` is refused rather than silently discarded.

        No pass runs here, so nothing would read a ``RunConfig``. Consuming it
        quietly would let a caller believe a strategy or diagnostics setting
        shaped the returned IR.
        """
        with pytest.raises(TypeError, match="specialize\\(\\) does not accept config="):
            _abs_entry.specialize(torch.randn(M, N), torch.zeros(M, N), config=RunConfig())

    def test_rejects_config_in_signature_mode(self):
        """The same refusal applies with no sample arguments at all."""
        with pytest.raises(TypeError, match="specialize\\(\\) does not accept config="):
            _shaped_entry.specialize(config=RunConfig(strategy=OptimizationStrategy.Default))


class TestParamAccessors:
    """``param_names`` / ``output_param_names`` describe the signature."""

    def test_param_names_in_declaration_order(self):
        assert _abs_entry.param_names == ("a", "out")
        assert _mixed_directions.param_names == ("x", "acc", "y", "z")

    def test_output_param_names_covers_out_and_inout(self):
        """Both directions are outputs, reported in declaration order."""
        assert _mixed_directions.output_param_names == ("acc", "z")

    def test_output_param_names_excludes_pure_inputs(self):
        assert _abs_entry.output_param_names == ("out",)
        assert _abs_kernel.output_param_names == ("out",)


def test_implicit_first_out_is_saved_before_rebinding_and_ignores_inout():
    from pypto.debug import torch_codegen  # noqa: PLC0415

    @jit
    def entry(
        state: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        other: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ):
        pypto_implicit_out = state  # noqa: F841
        pypto_implicit_out1 = other  # noqa: F841
        out = other  # noqa: F841

    values = [torch.full((M, N), value) for value in (1.0, 2.0, 3.0)]
    before = entry.specialize(*values)
    after = passes.normalize_return_order()(passes.convert_to_ssa()(before))
    function = after.get_function("entry")
    assert function is not None
    assert ir.returned_param_indices(function, after) == [1]
    namespace = {}
    exec(torch_codegen(after), namespace)  # noqa: S102
    torch.testing.assert_close(namespace["entry"](*values), values[1])


@pytest.mark.parametrize("with_docstring", [False, True])
def test_implicit_first_out_preserves_func_attr_prologue(with_docstring):
    @jit
    def entry(out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
        """The implicit result capture must follow the complete attribute prologue."""
        pl.func_attr({"split_aic": True})
        pl.func_attr({"split_aiv": True})
        saved = out  # noqa: F841

    # Exercise both legal prologues through the public specialization surface.
    if not with_docstring:
        # A second source function avoids mutating a JIT object's source snapshot.
        @jit
        def entry_without_docstring(out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
            pl.func_attr({"split_aic": True})
            pl.func_attr({"split_aiv": True})
            saved = out  # noqa: F841

        entry = entry_without_docstring
    program = entry.specialize(torch.zeros((M, N)))
    function = program.get_function(entry.__name__)
    assert function is not None
    assert ir.returned_param_indices(function, program) == [0]


def test_no_return_entry_without_out_does_not_invent_an_inout_result():
    @jit
    def entry(state: pl.InOut[pl.Tensor[[M, N], pl.FP32]]):
        saved = state  # noqa: F841

    program = entry.specialize(torch.ones((M, N)))
    function = program.get_function("entry")
    assert function is not None
    assert not function.return_types
    assert ir.returned_param_indices(function, program) == []


@pytest.mark.parametrize(
    "first_result",
    [
        "name",
        "jit_call",
        "builtin_call",
        "tuple_projection",
        "scoped_tuple_projection",
        "assigned_tuple_projection",
        "rebound_tuple_element",
        "tuple_alias",
        "tuple_unpack",
        "branch_tuple_projection",
        "branch_tuple_join",
        "tuple_element_expression",
        "tuple_dep_call",
        "named_tuple_dep_call",
        "inline_tuple_dep_projection",
        "literal_tuple_projection",
        "forwarded_tuple_dep_call",
        "loop_tuple_projection",
    ],
)
def test_host_specializes_downstream_chip_in_explicit_return_order(first_result):
    @pl.jit.incore
    def kernel(out: pl.Out[pl.Tensor]):
        return out

    @pl.jit
    def chip_produce(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        return second, first

    @pl.jit
    def chip_forward(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        return kernel(second), first

    @pl.jit
    def chip_builtin(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        return pl.tensor.assemble(second, second, [0]), first

    @pl.jit
    def chip_projection(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = (second, first)
        return pair[0], first

    @pl.jit(auto_scope=False)
    def chip_scoped_projection(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        with pl.manual_scope():
            pair = (second, first)
            return pair[0], first

    @pl.jit
    def chip_assigned_projection(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = (second, first)
        value = pair[0]
        return value, first

    @pl.jit
    def chip_rebound_tuple_element(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = (second, first)
        second = first  # noqa: F841
        return pair[0], first

    @pl.jit
    def chip_tuple_alias(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = (second, first)
        alias = pair
        pair = (first, second)  # noqa: F841
        return alias[0], first

    @pl.jit
    def chip_tuple_unpack(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = (second, first)
        value, other = pair
        return value, other

    @pl.jit
    def chip_branch_projection(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = (second, first)
        if pl.tensor.dim(first, 0) > 1:
            value = pair[0]
        else:
            value = second
        return value, first

    @pl.jit
    def chip_branch_tuple_join(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        if pl.tensor.dim(first, 0) > 1:
            pair = (second, first)
        else:
            alias = second
            pair = (alias, first)
        return pair[0], first

    @pl.jit
    def chip_tuple_element_expression(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = (second, first)
        forwarded = (pair[0], first)
        return forwarded[0], first

    @pl.jit.graph
    def make_pair(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        return second, first

    @pl.jit
    def chip_tuple_dep_call(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = make_pair(first, second)
        return pair[0], first

    @pl.jit.graph
    def make_named_pair(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        result = (second, first)
        return result

    @pl.jit
    def chip_named_tuple_dep_call(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = make_named_pair(first, second)
        return pair[0], first

    @pl.jit
    def chip_inline_tuple_dep_projection(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        return make_named_pair(first, second)[0], first

    @pl.jit
    def chip_literal_tuple_projection(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        return (second, first)[0], first

    @pl.jit.graph
    def forward_pair(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        return make_named_pair(first, second)

    @pl.jit
    def chip_forwarded_tuple_dep_call(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = forward_pair(first, second)
        return pair[0], first

    @pl.jit
    def chip_loop_tuple_projection(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = (second, first)
        for i in pl.range(3):
            pair = (second, first)
        value = pair[0]
        return value, first

    produce = {
        "name": chip_produce,
        "jit_call": chip_forward,
        "builtin_call": chip_builtin,
        "tuple_projection": chip_projection,
        "scoped_tuple_projection": chip_scoped_projection,
        "assigned_tuple_projection": chip_assigned_projection,
        "rebound_tuple_element": chip_rebound_tuple_element,
        "tuple_alias": chip_tuple_alias,
        "tuple_unpack": chip_tuple_unpack,
        "branch_tuple_projection": chip_branch_projection,
        "branch_tuple_join": chip_branch_tuple_join,
        "tuple_element_expression": chip_tuple_element_expression,
        "tuple_dep_call": chip_tuple_dep_call,
        "named_tuple_dep_call": chip_named_tuple_dep_call,
        "inline_tuple_dep_projection": chip_inline_tuple_dep_projection,
        "literal_tuple_projection": chip_literal_tuple_projection,
        "forwarded_tuple_dep_call": chip_forwarded_tuple_dep_call,
        "loop_tuple_projection": chip_loop_tuple_projection,
    }[first_result]

    @pl.jit
    def chip_consume(x: pl.Tensor):
        return x

    @pl.jit.host
    def entry(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        left, right = produce(first, second)
        return chip_consume(left)

    program = entry.specialize(torch.empty(64), torch.empty(32))
    consume = program.get_function("chip_consume")
    assert consume is not None
    assert isinstance(consume.params[0].type, ir.TensorType)
    extent = consume.params[0].type.shape[0]
    assert isinstance(extent, ir.ConstInt)
    assert extent.value == 32


def test_host_specializes_consumer_from_saved_chip_tuple():
    @pl.jit
    def produce(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        return second, first

    @pl.jit
    def consume(x: pl.Tensor):
        return x

    @pl.jit.host
    def entry(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
        pair = produce(first, second)
        left = pair[0]
        return consume(left)

    program = entry.specialize(torch.empty(64), torch.empty(32))
    consumer = program.get_function("consume")
    assert consumer is not None
    assert isinstance(consumer.params[0].type, ir.TensorType)
    extent = consumer.params[0].type.shape[0]
    assert isinstance(extent, ir.ConstInt)
    assert extent.value == 32


def test_host_reuses_dependency_return_summary_for_equal_inputs(monkeypatch):
    @pl.jit
    def chip(out: pl.Out[pl.Tensor], width: pl.constexpr):
        return out

    @pl.jit.host
    def entry(a: pl.Out[pl.Tensor], b: pl.Out[pl.Tensor]):
        first = chip(a, 2)
        repeat = chip(a, 2)
        different_shape = chip(b, 2)
        different_constexpr = chip(a, 3)
        return first, repeat, different_shape, different_constexpr

    original = _tensor_meta.extract_local_tensor_metas
    descents = 0

    def count(func, *args, **kwargs):
        nonlocal descents
        if func is chip._func:
            descents += 1
        return original(func, *args, **kwargs)

    monkeypatch.setattr(_tensor_meta, "extract_local_tensor_metas", count)
    metas = original(
        entry._func,
        seed_meta={
            "a": _tensor_meta.TensorMeta((64,), DataType.FP32),
            "b": _tensor_meta.TensorMeta((32,), DataType.FP32),
        },
        caller_func_type="host",
        discover_deps=_discover_dep_bindings,
    )
    assert [metas[name].shape for name in ("first", "repeat", "different_shape", "different_constexpr")] == [
        (64,),
        (64,),
        (32,),
        (64,),
    ]
    assert descents == 3


def test_host_shares_return_summary_across_independent_chip_callers(monkeypatch):
    @pl.jit.graph
    def shared(x: pl.Tensor):
        value = x
        return value

    @pl.jit
    def chip_left(out: pl.Out[pl.Tensor]):
        return shared(out)

    @pl.jit
    def chip_right(out: pl.Out[pl.Tensor]):
        return shared(out)

    @pl.jit.host
    def entry(out: pl.Out[pl.Tensor]):
        left = chip_left(out)
        right = chip_right(out)
        return left, right

    original = _tensor_meta.extract_local_tensor_metas
    descents = 0

    def count(func, *args, **kwargs):
        nonlocal descents
        if func is shared._func:
            descents += 1
        return original(func, *args, **kwargs)

    monkeypatch.setattr(_tensor_meta, "extract_local_tensor_metas", count)
    metas = original(
        entry._func,
        seed_meta={"out": _tensor_meta.TensorMeta((64,), DataType.FP32)},
        caller_func_type="host",
        discover_deps=_discover_dep_bindings,
    )
    assert metas["left"].shape == metas["right"].shape == (64,)
    assert descents == 1


def test_host_infers_out_metadata_for_chip_returning_kernel_call():
    @pl.jit.incore
    def kernel(out: pl.Out[pl.Tensor]):
        return out

    @pl.jit
    def chip_produce(out: pl.Out[pl.Tensor]):
        return kernel(out)

    @pl.jit
    def chip_consume(x: pl.Tensor):
        return x

    @pl.jit.host
    def entry(out: pl.Out[pl.Tensor]):
        result = chip_produce(out)
        return chip_consume(result)

    program = entry.specialize(torch.empty(32))
    consume = program.get_function("chip_consume")
    assert consume is not None
    assert isinstance(consume.params[0].type, ir.TensorType)
    extent = consume.params[0].type.shape[0]
    assert isinstance(extent, ir.ConstInt)
    assert extent.value == 32


@pytest.mark.parametrize("op", ["assemble", "store", "write"])
def test_host_preserves_metadata_for_qualified_chip_writeback(op):
    @pl.jit
    def chip_store(src: pl.Tensor, out: pl.Out[pl.Tensor]):
        tile = pl.tile.load(src, [0, 0], [1, 32], [1, 32])
        out = pl.tile.store(tile, [0, 0], out)
        return out

    @pl.jit
    def chip_assemble(src: pl.Tensor, out: pl.Out[pl.Tensor]):
        out = pl.tensor.assemble(out, src, [0, 0])
        return out

    @pl.jit
    def chip_write(src: pl.Tensor, out: pl.Out[pl.Tensor]):
        out = pl.tensor.write(out, [0, 0], pl.tensor.read(src, [0, 0]))
        return out

    produce = {"assemble": chip_assemble, "store": chip_store, "write": chip_write}[op]

    @pl.jit
    def chip_consume(x: pl.Tensor):
        return x

    @pl.jit.host
    def entry(src: pl.Tensor, out: pl.Out[pl.Tensor]):
        result = produce(src, out)
        return chip_consume(result)

    program = entry.specialize(torch.empty((1, 32)), torch.empty((1, 64)))
    consume = program.get_function("chip_consume")
    assert consume is not None
    assert isinstance(consume.params[0].type, ir.TensorType)
    extent = consume.params[0].type.shape[-1]
    assert isinstance(extent, ir.ConstInt)
    assert extent.value == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
