# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for python/pypto/jit/specializer.py — AST transformation correctness."""

import ast
import math
import textwrap
import types
import typing
import warnings
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import backend
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import PartialCodegenError
from pypto.ir import python_print
from pypto.jit import decorator
from pypto.jit._specialize_body import BodyTransformer
from pypto.jit._specialize_values import layout_str, render_free_value
from pypto.jit.specializer import (
    DynDim,
    SpecializeContext,
    Specializer,
    TensorMeta,
    _build_tensor_annotation,
    _classify_params,
    _collect_annotation_dynamic_dims,
    _collect_dynamic_dims,
    _collect_dynvar_names,
    _infer_return_type,
    _unwrap_direction_annotation,
    build_specialize_context,
    specialize,
    tile_annotation_fingerprints,
)
from pypto.language.parser.ast_parser import ASTParser
from pypto.language.parser.diagnostics import UndefinedVariableError
from pypto.pypto_core import DataType, ir, passes
from pypto.runtime.runner import ExecutionMode, RunConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_func(src: str) -> ast.FunctionDef:
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))


def _make_ctx(
    func_name: str = "kernel",
    source: str = "",
    func_type: str = "orchestration",
    param_names: list[str] | None = None,
    tensor_meta: dict[str, TensorMeta] | None = None,
    scalar_dtypes: dict | None = None,
    dep_names: list[str] | None = None,
    auto_scope: bool = True,
) -> SpecializeContext:
    # Dynamic dims now live as DynDim entries inside tensor_meta.shape — tests
    # construct them directly when they need dynamic behavior.
    return SpecializeContext(
        func_name=func_name,
        source=source,
        func_type=func_type,
        level=None,
        param_names=param_names or [],
        tensor_meta=tensor_meta or {},
        scalar_dtypes=scalar_dtypes or {},
        dep_names=dep_names or [],
        auto_scope=auto_scope,
    )


# ---------------------------------------------------------------------------
# _collect_dynamic_dims
# ---------------------------------------------------------------------------


class TestCollectDynamicDims:
    def test_single_bind_dynamic(self):
        src = textwrap.dedent("""
            def f(a, b):
                M = pl.dynamic("M")
                a.bind_dynamic(0, M)
        """)
        func_def = _parse_func(src)
        dims = _collect_dynamic_dims(func_def, {"a", "b"})
        assert ("a", 0) in dims

    def test_multiple_bind_dynamic(self):
        src = textwrap.dedent("""
            def f(a, c):
                M = pl.dynamic("M")
                a.bind_dynamic(0, M)
                c.bind_dynamic(0, M)
        """)
        func_def = _parse_func(src)
        dims = _collect_dynamic_dims(func_def, {"a", "c"})
        assert ("a", 0) in dims
        assert ("c", 0) in dims

    def test_no_bind_dynamic(self):
        src = textwrap.dedent("""
            def f(a):
                x = a.shape[0]
        """)
        func_def = _parse_func(src)
        dims = _collect_dynamic_dims(func_def, {"a"})
        assert len(dims) == 0

    def test_non_param_ignored(self):
        src = textwrap.dedent("""
            def f(a):
                tmp.bind_dynamic(0, M)
        """)
        func_def = _parse_func(src)
        dims = _collect_dynamic_dims(func_def, {"a"})
        assert len(dims) == 0


# ---------------------------------------------------------------------------
# _collect_annotation_dynamic_dims
# ---------------------------------------------------------------------------

# Module-level dynvars + functions so inspect.signature sees real annotations.
_M = pl.dynamic("M")
_BATCH = pl.dynamic("USER_BATCH")


def _ann_dyn_kernel(
    a: pl.Tensor[[_M, 128], pl.FP32],
    seq: pl.Tensor[[_M], pl.INT32],
    weight: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[_M, 256], pl.FP32]],
):
    return out


def _ann_static_kernel(a: pl.Tensor[[64, 128], pl.FP32], c: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
    return c


class TestCollectAnnotationDynamicDims:
    def test_dynvar_in_annotation_detected(self):
        """A pl.dynamic() var used directly in the annotation marks the dim dynamic."""
        names = {"a", "seq", "weight", "out"}
        dims, bindings, literals = _collect_annotation_dynamic_dims(_ann_dyn_kernel, names)
        assert dims == {("a", 0), ("seq", 0), ("out", 0)}
        assert bindings == {"a__0": "M", "seq__0": "M", "out__0": "M"}
        assert literals == {"M": "M"}

    def test_static_annotation_has_no_dynamic_dims(self):
        dims, bindings, literals = _collect_annotation_dynamic_dims(_ann_static_kernel, {"a", "c"})
        assert dims == set()
        assert bindings == {}
        assert literals == {}

    def test_out_wrapped_annotation_detected(self):
        """pl.Out[...] annotations unwrap to the inner Tensor and are scanned."""
        dims, _, _ = _collect_annotation_dynamic_dims(_ann_dyn_kernel, {"out"})
        assert ("out", 0) in dims

    def test_literal_matches_dynvar_name(self):
        """The emitted literal is DynVar.name, so generated pl.dynamic() round-trips."""

        def kernel(x: pl.Tensor[[_BATCH, 32], pl.FP32]):
            return x

        _, bindings, literals = _collect_annotation_dynamic_dims(kernel, {"x"})
        assert bindings == {"x__0": "USER_BATCH"}
        assert literals == {"USER_BATCH": "USER_BATCH"}


# ---------------------------------------------------------------------------
# _collect_dynvar_names
# ---------------------------------------------------------------------------


class TestCollectDynvarNames:
    def test_single(self):
        src = textwrap.dedent("""
            def f(a):
                M = pl.dynamic("M")
        """)
        func_def = _parse_func(src)
        names = _collect_dynvar_names(func_def)
        assert "M" in names

    def test_multiple(self):
        src = textwrap.dedent("""
            def f(a):
                M = pl.dynamic("M")
                N = pl.dynamic("N")
        """)
        func_def = _parse_func(src)
        names = _collect_dynvar_names(func_def)
        assert {"M", "N"} <= names.keys()

    def test_no_dynvar(self):
        src = textwrap.dedent("""
            def f(a):
                x = 1
        """)
        func_def = _parse_func(src)
        names = _collect_dynvar_names(func_def)
        assert len(names) == 0

    def test_literal_preserved_when_var_differs(self):
        """rows = pl.dynamic("M") should map rows → "M", not rows → "rows"."""
        src = textwrap.dedent("""
            def f(a):
                rows = pl.dynamic("M")
        """)
        func_def = _parse_func(src)
        names = _collect_dynvar_names(func_def)
        assert names["rows"] == "M"

    def test_literal_fallback_when_no_string_arg(self):
        """pl.dynamic() with no string arg should fall back to variable name."""
        src = textwrap.dedent("""
            def f(a):
                M = pl.dynamic()
        """)
        func_def = _parse_func(src)
        names = _collect_dynvar_names(func_def)
        assert names["M"] == "M"


# ---------------------------------------------------------------------------
# _classify_params
# ---------------------------------------------------------------------------


class TestClassifyParams:
    def test_tensor_params(self):
        src = textwrap.dedent("""
            def f(a: pl.Tensor, b: pl.Tensor):
                pass
        """)
        func_def = _parse_func(src)
        out_params, _, tensor_params, _, distributed_params = _classify_params(func_def)
        assert "a" in tensor_params
        assert "b" in tensor_params
        assert len(out_params) == 0
        assert not distributed_params

    def test_out_param(self):
        src = textwrap.dedent("""
            def f(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                pass
        """)
        func_def = _parse_func(src)
        out_params, _, tensor_params, _, distributed_params = _classify_params(func_def)
        assert "c" in out_params
        assert "c" in tensor_params
        assert "a" not in out_params
        assert not distributed_params

    def test_out_param_subscripted_tensor(self):
        """Out[pl.Tensor[[64], pl.FP32]] should still be recognised as Out+tensor."""
        src = textwrap.dedent("""
            def f(c: pl.Out[pl.Tensor[[64], pl.FP32]]):
                pass
        """)
        func_def = _parse_func(src)
        out_params, _, tensor_params, _, distributed_params = _classify_params(func_def)
        assert "c" in out_params
        assert "c" in tensor_params
        assert not distributed_params

    def test_scalar_bare_dtype(self):
        src = textwrap.dedent("""
            def f(BLOCK_M: pl.INDEX, alpha: pl.FP32):
                pass
        """)
        func_def = _parse_func(src)
        _, _, _, scalar_strs, _ = _classify_params(func_def)
        assert "BLOCK_M" in scalar_strs
        assert "alpha" in scalar_strs

    def test_distributed_tensor_subscripted(self):
        """pld.DistributedTensor[[...], dtype] is classified as tensor + distributed."""
        src = textwrap.dedent("""
            def f(a: pl.Tensor[[64], pl.FP32],
                  b: pld.DistributedTensor[[256], pl.INT8]):
                pass
        """)
        func_def = _parse_func(src)
        _, _, tensor_params, _, distributed_params = _classify_params(func_def)
        assert "a" in tensor_params
        assert "b" in tensor_params
        assert "a" not in distributed_params
        assert "b" in distributed_params

    def test_distributed_tensor_bare(self):
        """Bare pld.DistributedTensor (no subscript) is classified as distributed."""
        src = textwrap.dedent("""
            def f(b: pld.DistributedTensor):
                pass
        """)
        func_def = _parse_func(src)
        _, _, tensor_params, _, distributed_params = _classify_params(func_def)
        assert "b" in tensor_params
        assert "b" in distributed_params

    def test_distributed_tensor_bare_name(self):
        """Bare DistributedTensor name (without pld. prefix) is also recognised."""
        src = textwrap.dedent("""
            def f(b: DistributedTensor):
                pass
        """)
        func_def = _parse_func(src)
        _, _, tensor_params, _, distributed_params = _classify_params(func_def)
        assert "b" in tensor_params
        assert "b" in distributed_params

    def test_out_wrapped_distributed_tensor(self):
        """Out[pld.DistributedTensor[...]] is Out + tensor + distributed."""
        src = textwrap.dedent("""
            def f(c: pl.Out[pld.DistributedTensor[[64], pl.FP32]]):
                pass
        """)
        func_def = _parse_func(src)
        out_params, _, tensor_params, _, distributed_params = _classify_params(func_def)
        assert "c" in out_params
        assert "c" in tensor_params
        assert "c" in distributed_params

    def test_inout_param(self):
        src = textwrap.dedent("""
            def f(a: pl.Tensor, c: pl.InOut[pl.Tensor]):
                pass
        """)
        func_def = _parse_func(src)
        out_params, inout_params, tensor_params, _, distributed_params = _classify_params(func_def)
        assert "c" in inout_params
        assert "c" in tensor_params
        assert "c" not in out_params
        assert not distributed_params

    def test_inout_param_subscripted_tensor(self):
        """InOut[pl.Tensor[[64], pl.FP32]] is recognised as InOut + tensor."""
        src = textwrap.dedent("""
            def f(c: pl.InOut[pl.Tensor[[64], pl.FP32]]):
                pass
        """)
        func_def = _parse_func(src)
        _out, inout_params, tensor_params, _, distributed_params = _classify_params(func_def)
        assert "c" in inout_params
        assert "c" in tensor_params
        assert not distributed_params


# ---------------------------------------------------------------------------
# BodyTransformer
# ---------------------------------------------------------------------------


class TestBodyTransformer:
    def _transform(self, src: str, tensor_meta: dict, dep_names=None):
        src = textwrap.dedent(src)
        tree = ast.parse(src)
        func_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        # DynDim info now lives inside tensor_meta.shape — tests inject it there
        # directly when needed (see test_shape_unpack_expanded_dynamic / etc.).
        transformer = BodyTransformer(
            tensor_meta=tensor_meta,
            dep_names=dep_names or set(),
        )
        new_body = transformer.transform_body(func_def.body)
        new_func = ast.FunctionDef(
            name="f",
            args=func_def.args,
            body=new_body or [ast.Pass()],
            decorator_list=[],
            returns=None,
            lineno=1,
            col_offset=0,
        )
        ast.fix_missing_locations(new_func)
        return ast.unparse(new_func)

    @pytest.mark.parametrize("capture", [False, True])
    def test_graph_local_task_id_shadow_preserves_folded_shape(self, capture):
        producer = (
            "with pl.at(level=pl.Level.CORE_GROUP) as width:\n"
            "                            pl.store(pl.load(c, [0, 0], [128, 64]), [0, 0], c)"
            if capture
            else "width = pl.system.task_dummy(deps=[])"
        )
        source = textwrap.dedent(f"""
            def f(c: pl.InOut[pl.Tensor]):
                width = c.shape[1]
                with pl.manual_scope():
                    with pl.graph("recording"):
                        {producer}
                        pl.system.task_dummy(deps=[width])
                return pl.create_tensor([128, width], dtype=pl.FP32)
        """)
        ctx = _make_ctx(
            "f", source, param_names=["c"], tensor_meta={"c": TensorMeta((128, 64), DataType.FP32)}
        )
        function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
        assert function is not None
        assert isinstance(function.body, ir.SeqStmts)
        returned = function.body.stmts[-1]
        assert isinstance(returned, ir.ReturnStmt)
        (value,) = returned.value
        result = value.type
        assert isinstance(result, ir.TensorType)
        assert isinstance(result.shape[1], ir.ConstInt) and result.shape[1].value == 64

    def test_graph_rebind_prefers_task_id_alias_over_old_folded_shape(self):
        source = textwrap.dedent("""
            def f(c: pl.Tensor):
                with pl.manual_scope():
                    seed = pl.system.task_dummy(deps=[])
                    for i in pl.range(1):
                        ready = c.shape[1]
                        ready = seed
                    with pl.graph("recording") as done:
                        ready = pl.system.task_dummy(deps=[])
                        pl.system.task_dummy(deps=[ready])
                    pl.system.task_dummy(deps=[ready, done])
        """)
        ctx = _make_ctx(
            "f", source, param_names=["c"], tensor_meta={"c": TensorMeta((128, 64), DataType.FP32)}
        )
        rendered = specialize("Before", [ctx])
        assert "ready_v2 = pl.system.task_dummy(deps=[])" in rendered
        assert pl.parse_program(rendered).get_function("f") is not None

    def test_graph_plain_task_id_does_not_replace_outer_alias(self):
        source = """
            def f():
                with pl.manual_scope():
                    for i in pl.range(1):
                        seed = pl.system.task_dummy(deps=[])
                        latest = pl.system.task_dummy(deps=[])
                        ready = seed
                        ready = latest
                    with pl.graph("recording") as done:
                        ready = pl.system.task_dummy(deps=[])
                    pl.system.task_dummy(deps=[ready, done])
        """
        rendered = specialize("Before", [_make_ctx("f", textwrap.dedent(source))])
        assert "ready_v2 = pl.system.task_dummy(deps=[])" in rendered
        assert pl.parse_program(rendered).get_function("f") is not None

    @pytest.mark.parametrize(
        "rhs",
        ["recorded", "pl.create_tensor([64, 64], dtype=pl.FP32)", "pl.tensor.reshape(recorded, [32, 128])"],
    )
    def test_graph_host_tensor_assignments_can_change_shape(self, rhs):
        source = textwrap.dedent(f"""
            def f(first: pl.Tensor, recorded: pl.InOut[pl.Tensor]):
                for i in pl.range(1):
                    out = first
                with pl.graph("recording"):
                    out = {rhs}
                    with pl.at(level=pl.Level.CORE_GROUP):
                        pl.store(pl.load(recorded, [0, 0], [64, 64]), [0, 0], recorded)
                return recorded
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["first", "recorded"],
            tensor_meta={
                "first": TensorMeta((128, 128), DataType.FP32),
                "recorded": TensorMeta((64, 64), DataType.FP32),
            },
        )
        rendered = specialize("Before", [ctx])
        assert "out_v1 =" in rendered
        assert pl.parse_program(rendered).get_function("f") is not None

    def test_bind_dynamic_removed(self):
        src = """
            def f(a):
                a.bind_dynamic(0, M)
                x = 1
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((128, 128), DataType.FP32)})
        assert "bind_dynamic" not in out

    def test_pl_dynamic_assignment_removed(self):
        src = """
            def f(a):
                M = pl.dynamic("M")
                x = 1
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((128, 128), DataType.FP32)})
        assert "pl.dynamic" not in out
        assert "x = 1" in out

    def test_shape_attribute_replaced(self):
        src = """
            def f(a):
                s = a.shape
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((128, 64), DataType.FP32)})
        assert "(128, 64)" in out

    def test_shape_subscript_replaced(self):
        src = """
            def f(a):
                k = a.shape[1]
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((256, 128), DataType.FP32)})
        # Static dim: assignment suppressed, k inlined at use sites
        assert "k = 128" not in out
        assert "a.shape" not in out

    def test_shape_unpack_expanded(self):
        src = """
            def f(a):
                M, N = a.shape
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((128, 64), DataType.FP32)})
        # Static dims are inlined — no M/N assignments, unpacking removed
        assert "M = 128" not in out
        assert "N = 64" not in out
        assert "a.shape" not in out

    def test_dtype_replaced(self):
        src = """
            def f(a):
                d = a.dtype
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((8, 8), DataType.FP32)})
        assert "pl.FP32" in out
        assert "a.dtype" not in out

    def test_dep_call_rewritten(self):
        src = """
            def f(a):
                c = sub_kernel(a)
        """
        out = self._transform(
            src,
            tensor_meta={"a": TensorMeta((8,), DataType.FP32)},
            dep_names={"sub_kernel"},
        )
        assert "self.sub_kernel" in out

    def test_non_dep_call_not_rewritten(self):
        src = """
            def f(a):
                c = pl.add(a, a)
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((8,), DataType.FP32)})
        assert "self.pl.add" not in out
        assert "pl.add" in out

    def test_scalar_param_is_not_substituted_in_body(self):
        """A scalar parameter is a runtime value: the body keeps the symbolic name.

        Folding it would bake one call site's value into the artifact and split
        the specialization cache by value (issue #2751).
        """
        src = """
            def f(a: pl.Tensor, block_m: pl.INDEX):
                tile = pl.load(a, [0], [block_m])
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((256,), DataType.FP32)})
        assert "pl.load(a, [0], [block_m])" in out

    def _transform_with_globals(self, src: str, tensor_meta: dict, py_globals: dict):
        src = textwrap.dedent(src)
        tree = ast.parse(src)
        func_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        transformer = BodyTransformer(
            tensor_meta=tensor_meta,
            dep_names=set(),
            py_globals=py_globals,
        )
        new_body = transformer.transform_body(func_def.body)
        new_func = ast.FunctionDef(
            name="f",
            args=func_def.args,
            body=new_body or [ast.Pass()],
            decorator_list=[],
            returns=None,
            lineno=1,
            col_offset=0,
        )
        ast.fix_missing_locations(new_func)
        return ast.unparse(new_func)

    def test_local_annotation_shape_constant_inlined(self):
        """A module-level int used in a body-level annotation (``t: pl.Tile[[1,
        W_PAD], pl.FP32]``) must be inlined to its value. Otherwise the un-inlined
        name leaks into the generated source and the parser rejects it with
        "Unknown shape variable: W_PAD"."""
        src = """
            def f(a: pl.Tensor):
                t: pl.Tile[[1, W_PAD], pl.FP32] = pl.load(a, [0, 0], [1, W_PAD])
        """
        out = self._transform_with_globals(
            src,
            tensor_meta={"a": TensorMeta((1, 96), DataType.FP32)},
            py_globals={"W_PAD": 96},
        )
        # The constant is inlined in BOTH the annotation and the value expression.
        assert "W_PAD" not in out
        assert "pl.Tile[[1, 96], pl.FP32]" in out

    def test_string_constant_is_inlined_as_a_literal(self):
        """A free name holding a str must render as a quoted literal.

        Operator kwargs are commonly strings (``mode="trunc"``,
        ``saturation_mode="on"``). Left un-inlined the name reaches the parser as
        an undefined identifier, so a kernel factory could not be parameterized
        by mode at all.
        """
        src = """
            def f(a: pl.Tensor):
                t = pl.load(a, [0, 0], [1, 64])
                q = pl.cast(t, pl.INT8, mode=MODE)
        """
        out = self._transform_with_globals(
            src,
            tensor_meta={"a": TensorMeta((1, 64), DataType.FP16)},
            py_globals={"MODE": "trunc"},
        )
        assert "mode='trunc'" in out
        assert "MODE" not in out

    def test_dtype_and_enum_constants_render_as_pl_paths(self):
        """``DataType`` / enum values render as the ``pl.`` spelling that evaluates back to them."""
        src = """
            def f(a: pl.Tensor):
                t = pl.load(a, [0, 0], [1, 64], target_memory=MEM)
                q = pl.cast(t, DTYPE, mode='trunc')
        """
        out = self._transform_with_globals(
            src,
            tensor_meta={"a": TensorMeta((1, 64), DataType.FP16)},
            py_globals={"DTYPE": pl.INT8, "MEM": pl.Mem.Vec},
        )
        assert "pl.INT8" in out
        assert "pl.Mem.Vec" in out
        assert "DTYPE" not in out and "MEM" not in out

    def test_sequence_constant_renders_elementwise(self):
        """A shape held in a module constant is a list of ints, not a scalar."""
        src = """
            def f(a: pl.Tensor):
                t = pl.load(a, [0, 0], SHAPE)
        """
        out = self._transform_with_globals(
            src,
            tensor_meta={"a": TensorMeta((1, 64), DataType.FP16)},
            py_globals={"SHAPE": [1, 64]},
        )
        assert "pl.load(a, [0, 0], [1, 64])" in out

    def test_unrenderable_value_leaves_the_name_alone(self):
        """A value with no source form must not be folded — the parser's error is the diagnostic."""
        src = """
            def f(a: pl.Tensor):
                t = pl.load(a, [0, 0], [1, 64], target_memory=WIDGET)
        """
        out = self._transform_with_globals(
            src,
            tensor_meta={"a": TensorMeta((1, 64), DataType.FP16)},
            py_globals={"WIDGET": object()},
        )
        assert "WIDGET" in out

    def test_a_local_still_shadows_a_free_name(self):
        """Folding is skipped for a name the body assigns — the local wins, as in Python."""
        src = """
            def f(a: pl.Tensor):
                MODE = pl.load(a, [0, 0], [1, 64])
                q = pl.cast(MODE, pl.INT8, mode='trunc')
        """
        out = self._transform_with_globals(
            src,
            tensor_meta={"a": TensorMeta((1, 64), DataType.FP16)},
            py_globals={"MODE": "trunc"},
        )
        assert "pl.cast(MODE, pl.INT8, mode='trunc')" in out


class TestRenderFreeValue:
    """``render_free_value`` — the single answer to "can this value be written as source?"."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (7, "7"),
            (1.5, "1.5"),
            (True, "True"),
            (None, "None"),
            ("trunc", "'trunc'"),
            (pl.INT8, "pl.INT8"),
            (pl.FP32, "pl.FP32"),
            (pl.Mem.Vec, "pl.Mem.Vec"),
            (pl.PadValue.zero, "pl.PadValue.zero"),
            (pl.NZ, "pl.TensorLayout.NZ"),
            ([1, 64], "[1, 64]"),
            ((1, 64), "(1, 64)"),
            ([[1, 2], [3, 4]], "[[1, 2], [3, 4]]"),
            ([pl.INT8, pl.FP32], "[pl.INT8, pl.FP32]"),
            # Non-finite floats: ast.unparse writes evaluable expressions, not the
            # bare names inf / nan, so a fill or padding value bound to one folds
            # like any other float.
            (float("inf"), "1e309"),
            (float("-inf"), "-1e309"),
        ],
    )
    def test_renders_to_evaluable_source(self, value, expected):
        rendered = render_free_value(value)
        assert rendered is not None
        assert ast.unparse(rendered) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (pl.INT8, pl.INT8),
            (pl.Mem.Vec, pl.Mem.Vec),
            (pl.NZ, pl.NZ),
            ([1, 64], [1, 64]),
            (float("inf"), float("inf")),
            (float("-inf"), float("-inf")),
        ],
    )
    def test_rendered_source_evaluates_back_to_the_value(self, value, expected):
        """The rendered text is only useful if ``pl`` can evaluate it back."""
        rendered = render_free_value(value)
        assert rendered is not None
        assert eval(ast.unparse(rendered), {"pl": pl}) == expected  # noqa: S307

    @pytest.mark.parametrize(
        "value",
        [object(), pl.Tensor, pl.load, [1, object()], {"a": 1}],
        ids=["opaque", "class", "function", "list-with-opaque", "dict"],
    )
    def test_declines_values_with_no_source_form(self, value):
        """Declining is the safe answer: the name survives and the parser reports it."""
        assert render_free_value(value) is None

    @pytest.mark.parametrize("enum_type", [pld.ReduceOp, pld.NotifyOp, pld.WaitCmp, pld.AtomicType])
    def test_distributed_enum_members_round_trip(self, enum_type):
        # AtomicType is shared with pl; its existing canonical spelling stays stable.
        prefix = "pl" if enum_type is pld.AtomicType else "pld"
        for member in enum_type:
            rendered = render_free_value(member)
            assert rendered is not None
            source = ast.unparse(rendered)
            assert source == f"{prefix}.{enum_type.__name__}.{member.name}"
            assert eval(source, {"pl": pl, "pld": pld}) is member  # noqa: S307

    def test_nan_renders_as_a_nan_producing_expression(self):
        """NaN has no literal, but ast.unparse still writes something evaluable."""
        rendered = render_free_value(float("nan"))
        assert rendered is not None
        assert math.isnan(eval(ast.unparse(rendered), {"__builtins__": {}}))  # noqa: S307


# ---------------------------------------------------------------------------
# Specializer (source generation)
# ---------------------------------------------------------------------------


class TestSpecializer:
    def _specialize_simple(
        self,
        func_source: str,
        param_names: list[str],
        tensor_meta: dict[str, TensorMeta],
        func_type: str = "orchestration",
        dep_names: list[str] | None = None,
        auto_scope: bool = True,
    ) -> str:
        ctx = _make_ctx(
            func_name="kernel",
            source=textwrap.dedent(func_source),
            func_type=func_type,
            param_names=param_names,
            tensor_meta=tensor_meta,
            dep_names=dep_names or [],
            auto_scope=auto_scope,
        )
        return specialize("_TestClass", [ctx])

    def test_folded_distributed_enum_adds_import_without_distributed_params(self):
        ctx = _make_ctx(
            func_name="kernel",
            source="def kernel():\n    pld.tensor.allreduce(data, signal, op=REDUCE_OP)\n",
            func_type="host",
            param_names=[],
            tensor_meta={},
        )
        ctx.py_globals = {"REDUCE_OP": pld.ReduceOp.Max}
        source = specialize("_TestClass", [ctx])
        assert "import pypto.language.distributed as pld" in source
        assert "op=pld.ReduceOp.Max" in source
        assert "REDUCE_OP" not in source

    def test_generated_has_pl_program(self):
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((128, 128), DataType.FP32),
                "c": TensorMeta((128, 128), DataType.FP32),
            },
        )
        assert "@pl.program" in out
        assert "class _TestClass" in out

    def test_generated_has_pl_function(self):
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((128, 128), DataType.FP32),
                "c": TensorMeta((128, 128), DataType.FP32),
            },
        )
        assert "@pl.function" in out

    def test_tensor_annotation_filled(self):
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((64, 32), DataType.FP16),
                "c": TensorMeta((64, 32), DataType.FP16),
            },
        )
        assert "pl.Tensor[[64, 32], pl.FP16]" in out

    def test_out_annotation_filled(self):
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((64, 32), DataType.FP16),
                "c": TensorMeta((64, 32), DataType.FP16),
            },
        )
        assert "pl.Out[pl.Tensor[[64, 32], pl.FP16]]" in out

    def test_inout_annotation_filled(self):
        """pl.InOut[...] on an orchestration entry round-trips into the
        generated @pl.function source. Regression: the InOut wrapper used to be
        dropped in _classify_params, so the param was emitted bare and downstream
        codegen raised 'Parameter <name> missing type annotation'."""
        src = """
            def kernel(a: pl.Tensor, c: pl.InOut[pl.Tensor]):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((64, 32), DataType.FP16),
                "c": TensorMeta((64, 32), DataType.FP16),
            },
        )
        assert "pl.InOut[pl.Tensor[[64, 32], pl.FP16]]" in out

    def test_dynvar_emitted_at_module_level(self):
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                M = pl.dynamic("M")
                a.bind_dynamic(0, M)
                c.bind_dynamic(0, M)
                return c
        """
        # DynDim entries inside tensor_meta.shape replace the previous
        # parallel dynamic_dims / dynvar_bindings tables.
        m_dim = DynDim(name="M", literal="M", static_bound=256)
        ctx = _make_ctx(
            func_name="kernel",
            source=textwrap.dedent(src),
            func_type="orchestration",
            param_names=["a", "c"],
            tensor_meta={
                "a": TensorMeta((m_dim, 128), DataType.FP32),
                "c": TensorMeta((m_dim, 128), DataType.FP32),
            },
        )
        out = specialize("_Dyn", [ctx])
        # DynVar declaration should appear before the class
        class_pos = out.index("class _Dyn")
        dv_pos = out.index('pl.dynamic("M")')
        assert dv_pos < class_pos

    def test_bind_dynamic_removed_from_body(self):
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                M = pl.dynamic("M")
                a.bind_dynamic(0, M)
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((128, 128), DataType.FP32),
                "c": TensorMeta((128, 128), DataType.FP32),
            },
        )
        assert "bind_dynamic" not in out

    def test_shape_constants_inlined(self):
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                M, N = a.shape
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((32, 16), DataType.FP32),
                "c": TensorMeta((32, 16), DataType.FP32),
            },
        )
        # Static dims are inlined — M, N assignments suppressed
        assert "M = 32" not in out
        assert "N = 16" not in out
        assert "a.shape" not in out

    def test_import_line_present(self):
        src = """
            def kernel(a: pl.Tensor):
                return a
        """
        out = self._specialize_simple(
            src,
            ["a"],
            {"a": TensorMeta((8,), DataType.FP32)},
        )
        assert "import pypto.language as pl" in out
        # pld import is only emitted when a DistributedTensor param is present.
        assert "import pypto.language.distributed" not in out

    def test_distributed_tensor_annotation_filled(self):
        src = """
            def kernel(a: pld.DistributedTensor):
                return a
        """
        out = self._specialize_simple(
            src,
            ["a"],
            {"a": TensorMeta((256,), DataType.INT8)},
        )
        assert "pld.DistributedTensor[[256], pl.INT8]" in out
        assert "import pypto.language.distributed as pld" in out

    def test_distributed_tensor_subscripted_annotation_specialized(self):
        """Subscripted pld.DistributedTensor annotations round-trip with concrete
        shape/dtype from TensorMeta — same plumbing as pl.Tensor, only the head
        type changes."""
        src = """
            def kernel(recv_x: pld.DistributedTensor[[256], pl.INT8]):
                return recv_x
        """
        out = self._specialize_simple(
            src,
            ["recv_x"],
            {"recv_x": TensorMeta((256,), DataType.INT8)},
        )
        assert "recv_x: pld.DistributedTensor[[256], pl.INT8]" in out
        # The plain Tensor head must not leak through when the annotation is
        # specifically DistributedTensor.
        assert "recv_x: pl.Tensor" not in out

    def test_mixed_tensor_and_distributed_tensor(self):
        """A function mixing pl.Tensor and pld.DistributedTensor params emits
        both head types and triggers the pld import."""
        src = """
            def kernel(indices: pl.Tensor, recv_x: pld.DistributedTensor):
                return indices
        """
        out = self._specialize_simple(
            src,
            ["indices", "recv_x"],
            {
                "indices": TensorMeta((128,), DataType.INT32),
                "recv_x": TensorMeta((256,), DataType.INT8),
            },
        )
        assert "indices: pl.Tensor[[128], pl.INT32]" in out
        assert "recv_x: pld.DistributedTensor[[256], pl.INT8]" in out
        assert "import pypto.language.distributed as pld" in out

    def test_distributed_tensor_return_annotation(self):
        """Return-type inference must preserve the distributed head — a function
        returning a pld.DistributedTensor must annotate the return as
        ``pld.DistributedTensor[...]``, not ``pl.Tensor[...]`` (the two kinds
        have distinct IR ObjectKind, so a leaked head type would be a real
        type-system bug)."""
        src = """
            def kernel(recv_x: pld.DistributedTensor):
                return recv_x
        """
        out = self._specialize_simple(
            src,
            ["recv_x"],
            {"recv_x": TensorMeta((256,), DataType.INT8)},
        )
        assert "-> pld.DistributedTensor[[256], pl.INT8]" in out
        assert "-> pl.Tensor" not in out

    def test_distributed_tensor_tuple_return_annotation(self):
        """Multi-return: the tuple element keeps the distributed head when the
        returned name is a pld.DistributedTensor param."""
        src = """
            def kernel(a: pl.Tensor, recv_x: pld.DistributedTensor):
                return a, recv_x
        """
        out = self._specialize_simple(
            src,
            ["a", "recv_x"],
            {
                "a": TensorMeta((128,), DataType.FP32),
                "recv_x": TensorMeta((256,), DataType.INT8),
            },
        )
        assert "-> tuple[pl.Tensor[[128], pl.FP32], pld.DistributedTensor[[256], pl.INT8]]" in out

    def test_incore_decorator_generated(self):
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((16, 16), DataType.FP32),
                "c": TensorMeta((16, 16), DataType.FP32),
            },
            func_type="incore",
        )
        assert "pl.FunctionType.InCore" in out

    def test_jit_host_emits_host_orchestrator_decorator(self):
        """@pl.jit.host generates a HOST Orchestrator @pl.function decorator —
        matching the canonical spelling in tests/st/distributed/test_l3_allreduce.py.
        ``type=`` is omitted (defaults to FunctionType.Opaque, which is correct
        for a HOST orchestrator)."""
        # Use ``def kernel(...)`` so _specialize_simple's hard-coded
        # func_name="kernel" matches; the function-body shape is what we're
        # testing, not the name.
        src = """
            def kernel(inputs: pld.DistributedTensor,
                       outputs: pl.Out[pld.DistributedTensor]):
                return outputs
        """
        out = self._specialize_simple(
            src,
            ["inputs", "outputs"],
            {
                "inputs": TensorMeta((2, 1, 256), DataType.FP32),
                "outputs": TensorMeta((2, 1, 256), DataType.FP32),
            },
            func_type="host",
        )
        assert "@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)" in out
        # A host entry must NOT emit a plain Orchestration decorator — that
        # would lose the HOST level and the role= attribute, leaving the
        # generated source indistinguishable from a chip orchestrator.
        assert "@pl.function(type=pl.FunctionType.Orchestration)" not in out

    def test_jit_host_with_orchestration_dep(self):
        """A host orchestrator calling a chip orchestration dep emits both
        decorators in the generated source. The host body's
        ``chip_orch(args, device=r)`` call is rewritten to
        ``self.chip_orch(args, device=r)``; ``device=`` is preserved as a
        keyword for the parser to recognise as the orchestration-dispatch
        kwarg."""
        host_src = textwrap.dedent("""
            def host_orch(inputs: pl.Tensor, outputs: pl.Out[pl.Tensor]):
                for r in pl.range(2):
                    chip_orch(inputs, outputs, device=r)
                return outputs
        """)
        chip_src = textwrap.dedent("""
            def chip_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                return c
        """)
        chip_ctx = _make_ctx(
            func_name="chip_orch",
            source=chip_src,
            func_type="orchestration",
            param_names=["a", "c"],
            tensor_meta={
                "a": TensorMeta((1, 256), DataType.FP32),
                "c": TensorMeta((1, 256), DataType.FP32),
            },
        )
        host_ctx = _make_ctx(
            func_name="host_orch",
            source=host_src,
            func_type="host",
            param_names=["inputs", "outputs"],
            tensor_meta={
                "inputs": TensorMeta((2, 1, 256), DataType.FP32),
                "outputs": TensorMeta((2, 1, 256), DataType.FP32),
            },
            dep_names=["chip_orch"],
        )
        out = specialize("_HostProgram", [chip_ctx, host_ctx])
        assert "@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)" in out
        assert "@pl.function(type=pl.FunctionType.Orchestration)" in out
        assert "self.chip_orch(inputs, outputs, device=r)" in out

    def test_orchestration_entry_emits_auto_scope_false(self):
        """auto_scope=False on an Orchestration entry emits
        ``@pl.function(type=..., auto_scope=False)`` so the body can place
        runtime scopes by hand."""
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((128, 128), DataType.FP32),
                "c": TensorMeta((128, 128), DataType.FP32),
            },
            auto_scope=False,
        )
        assert "@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)" in out

    def test_orchestration_entry_default_omits_auto_scope(self):
        """The default (auto_scope=True) emits no auto_scope= kwarg."""
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((128, 128), DataType.FP32),
                "c": TensorMeta((128, 128), DataType.FP32),
            },
        )
        assert "auto_scope" not in out

    def test_host_entry_emits_auto_scope_false(self):
        """auto_scope=False on a HOST orchestrator augments the HOST
        Orchestrator decorator rather than replacing it."""
        src = """
            def kernel(inputs: pld.DistributedTensor,
                       outputs: pl.Out[pld.DistributedTensor]):
                return outputs
        """
        out = self._specialize_simple(
            src,
            ["inputs", "outputs"],
            {
                "inputs": TensorMeta((2, 1, 256), DataType.FP32),
                "outputs": TensorMeta((2, 1, 256), DataType.FP32),
            },
            func_type="host",
            auto_scope=False,
        )
        assert "@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator, auto_scope=False)" in out

    def test_inline_dep_emits_auto_scope_false(self):
        """auto_scope=False on an inline sub-function emits
        ``@pl.function(type=pl.FunctionType.Inline, auto_scope=False)`` so its
        body can place runtime scopes by hand (#1733)."""
        src = """
            def kernel(a: pl.Tensor, c: pl.Tensor):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((128, 128), DataType.FP32),
                "c": TensorMeta((128, 128), DataType.FP32),
            },
            func_type="inline",
            auto_scope=False,
        )
        assert "@pl.function(type=pl.FunctionType.Inline, auto_scope=False)" in out

    def test_inline_dep_default_omits_auto_scope(self):
        """The default (auto_scope=True) inline decorator emits no auto_scope= kwarg."""
        src = """
            def kernel(a: pl.Tensor, c: pl.Tensor):
                return c
        """
        out = self._specialize_simple(
            src,
            ["a", "c"],
            {
                "a": TensorMeta((128, 128), DataType.FP32),
                "c": TensorMeta((128, 128), DataType.FP32),
            },
            func_type="inline",
        )
        assert "auto_scope" not in out


# ---------------------------------------------------------------------------
# Integration: specialize → parseable by pl.parse()
# ---------------------------------------------------------------------------


class TestSpecializerIntegration:
    """Verify that specializer output can be parsed by pl.parse()."""

    def test_single_func_parseable(self):
        src = textwrap.dedent("""
            def add_kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
                with pl.at(level=pl.Level.CORE_GROUP):
                    M, N = a.shape
                    tile_a = pl.load(a, [0, 0], [M, N])
                    tile_b = pl.load(b, [0, 0], [M, N])
                    tile_c = pl.add(tile_a, tile_b)
                    pl.store(tile_c, [0, 0], c)
                return c
        """)

        ctx = _make_ctx(
            func_name="add_kernel",
            source=src,
            func_type="orchestration",
            param_names=["a", "b", "c"],
            tensor_meta={
                "a": TensorMeta((128, 128), DataType.FP32),
                "b": TensorMeta((128, 128), DataType.FP32),
                "c": TensorMeta((128, 128), DataType.FP32),
            },
        )
        generated = specialize("_JitAddKernel", [ctx])
        prog = pl.parse(generated)

        assert isinstance(prog, ir.Program)

    def test_single_func_structural_equal_to_equivalent_program(self):
        """Generated JIT program should be structurally equal to a hand-written @pl.program."""

        # Hand-written equivalent
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def add_kernel(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile_a = pl.load(a, [0, 0], [128, 128])
                    tile_b = pl.load(b, [0, 0], [128, 128])
                    tile_c = pl.add(tile_a, tile_b)
                    pl.store(tile_c, [0, 0], c)
                return c

        src = textwrap.dedent("""
            def add_kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
                with pl.at(level=pl.Level.CORE_GROUP):
                    M, N = a.shape
                    tile_a = pl.load(a, [0, 0], [M, N])
                    tile_b = pl.load(b, [0, 0], [M, N])
                    tile_c = pl.add(tile_a, tile_b)
                    pl.store(tile_c, [0, 0], c)
                return c
        """)

        ctx = _make_ctx(
            func_name="add_kernel",
            source=src,
            func_type="orchestration",
            param_names=["a", "b", "c"],
            tensor_meta={
                "a": TensorMeta((128, 128), DataType.FP32),
                "b": TensorMeta((128, 128), DataType.FP32),
                "c": TensorMeta((128, 128), DataType.FP32),
            },
        )
        generated_src = specialize("add_kernel", [ctx])
        got = pl.parse(generated_src)

        ir.assert_structural_equal(got, Expected)

    def test_inline_dep_with_hand_scope_parseable(self):
        """An auto_scope=False entry calling an auto_scope=False inline dep whose
        body hand-places ``with pl.scope()`` specializes to parseable source —
        the parser accepts hand AUTO scopes in both functions (#1733)."""
        entry_src = textwrap.dedent("""
            def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                with pl.scope():
                    c = inline_fn(a, c)
                return c
        """)
        inline_src = textwrap.dedent("""
            def inline_fn(a: pl.Tensor, c: pl.Tensor):
                with pl.scope():
                    c = pl.add(a, c)
                return c
        """)
        meta = {
            "a": TensorMeta((128, 128), DataType.FP32),
            "c": TensorMeta((128, 128), DataType.FP32),
        }
        inline_ctx = _make_ctx(
            func_name="inline_fn",
            source=inline_src,
            func_type="inline",
            param_names=["a", "c"],
            tensor_meta=meta,
            auto_scope=False,
        )
        entry_ctx = _make_ctx(
            func_name="entry",
            source=entry_src,
            func_type="orchestration",
            param_names=["a", "c"],
            tensor_meta=meta,
            dep_names=["inline_fn"],
            auto_scope=False,
        )
        out = specialize("_InlineHandScope", [inline_ctx, entry_ctx])
        assert "@pl.function(type=pl.FunctionType.Inline, auto_scope=False)" in out
        assert "@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)" in out
        prog = pl.parse(out)
        assert isinstance(prog, ir.Program)


# ---------------------------------------------------------------------------
# TestVariableRebinding
# ---------------------------------------------------------------------------


class TestVariableRebinding:
    """Tests for alpha-renaming of variable rebindings in BodyTransformer."""

    @staticmethod
    def _verify_ssa(program: ir.Program) -> ir.Program:
        converted = passes.convert_to_ssa()(program)
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.SSAForm)
        passes.run_verifier(properties)(converted)
        return converted

    def _transform(self, src: str, tensor_meta: dict | None = None) -> str:
        src = textwrap.dedent(src)
        tree = ast.parse(src)
        func_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        param_names = [arg.arg for arg in func_def.args.args]
        all_defined = {
            node.id
            for node in ast.walk(func_def)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        transformer = BodyTransformer(
            tensor_meta=tensor_meta or {},
            dep_names=set(),
            param_names=param_names,
            initial_used_names=all_defined,
        )
        new_body = transformer.transform_body(func_def.body)
        new_func = ast.FunctionDef(
            name="f",
            args=func_def.args,
            body=new_body or [ast.Pass()],
            decorator_list=[],
            returns=None,
            lineno=1,
            col_offset=0,
        )
        ast.fix_missing_locations(new_func)
        return ast.unparse(new_func)

    def test_single_assignment_unchanged(self):
        src = """
            def f(a):
                x = pl.load(a)
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((64,), DataType.FP32)})
        assert "x =" in out
        assert "x_v1" not in out

    def test_chained_assignment_keeps_rhs_snapshot(self):
        """Chains evaluate once and do not read stale inlined or renamed targets."""
        src = """
            def f(a):
                total = 32
                total = half = total = observe(total // 2)
                saved = total + half
                total = total + 1
                total = half = observe(a.shape[1])
                width = a.shape[1]
                with scope():
                    width = other = observe(width // 2)
                return saved, total, half, width, other
        """
        transformed = self._transform(src, tensor_meta={"a": TensorMeta((1, 128), DataType.FP32)})
        calls = []

        def observe(value):
            calls.append(value)
            return value

        namespace: dict = {"observe": observe, "scope": nullcontext}
        exec(transformed, namespace)
        assert namespace["f"](None) == (32, 128, 128, 64, 64)
        assert calls == [16, 128, 64]
        assert all(
            len(node.targets) == 1
            for node in ast.walk(ast.parse(transformed))
            if isinstance(node, ast.Assign)
        )

    def test_rebind_generates_fresh_name(self):
        src = """
            def f(a):
                x = a
                x = pl.load(x)
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((64,), DataType.FP32)})
        assert "x =" in out
        assert "x_v1 =" in out

    @pytest.mark.parametrize(
        "body, expected",
        [
            ("x = 1; x = x + 1; x, y = (10, 20); return x, y", (10, 20)),
            ("x = 1; x = x + 1; x, y = (10, 20); x = x + 5; return x, y", (15, 20)),
            ("x = 1; x = x + 1; x, x = (10, 20); return x", 20),
            ("x = 1; x = x + 1; y = 3; y = y + 1; x, y = y, x; return x, y", (4, 2)),
            ("x = 1; x = x + 1; x, (y, z) = (10, (20, 30)); return x, y, z", (10, 20, 30)),
            ("x_v2 = 99; x = 1; x = x + 1; x, y = (10, 20); return x, x_v2", (10, 99)),
        ],
        ids=["read", "subsequent-rebind", "repeated-target", "swap", "nested", "alias-collision"],
    )
    def test_tuple_rebind_preserves_python_values(self, body, expected):
        """Unpacking must replace an active alias after evaluating the whole RHS."""
        transformed = self._transform("def f():\n    " + body)
        namespace = {}
        exec(transformed, namespace)
        assert namespace["f"]() == expected

    def test_tuple_first_assignment_is_tracked(self):
        """A later type-changing rebind needs a fresh name after tuple initialization."""
        transformed = self._transform("""
            def f():
                x, y = (1, 2)
                x = 3.0
                return x, y
        """)
        assert "x_v1 = 3.0" in transformed
        namespace = {}
        exec(transformed, namespace)
        assert namespace["f"]() == (3.0, 2)

    def test_tuple_rebind_evaluates_rhs_once(self):
        """A tuple-producing call sees the old binding and executes only once."""
        transformed = self._transform("""
            def f():
                x = 1
                x = x + 1
                x, y = pair(x)
                return x, y
        """)
        calls = []

        def pair(value):
            calls.append(value)
            return value + 10, value + 20

        namespace = {"pair": pair}
        exec(transformed, namespace)
        assert namespace["f"]() == (12, 22)
        assert calls == [2]

    @pytest.mark.parametrize("count", [0, 3])
    def test_tuple_rebind_in_loop_updates_incoming_alias(self, count):
        """Loop writes must keep updating the incoming binding on every iteration."""
        transformed = self._transform("""
            def f(count):
                x = 1
                x = x + 1
                y = 0
                for i in range(count):
                    x, y = (x + 1, y + 2)
                return x, y
        """)
        namespace = {}
        exec(transformed, namespace)
        assert namespace["f"](count) == (2 + count, 2 * count)

    @pytest.mark.parametrize("flag", [False, True])
    def test_tuple_rebind_in_branch_updates_incoming_alias(self, flag):
        """Branch writes cannot introduce a version undefined on the other path."""
        transformed = self._transform("""
            def f(flag):
                x = 1
                x = x + 1
                if flag:
                    x, y = (10, 20)
                else:
                    x, y = (30, 40)
                return x, y
        """)
        namespace = {}
        exec(transformed, namespace)
        assert namespace["f"](flag) == ((10, 20) if flag else (30, 40))

    def test_tuple_rebind_invalidates_inlined_shapes(self):
        """An unpack target must shadow an earlier folded shape constant."""
        transformed = self._transform(
            """
                def f(a):
                    rows, cols = a.shape
                    with scope():
                        rows, cols = (10, 20)
                        observed = (rows, cols)
                    return observed
            """,
            tensor_meta={"a": TensorMeta((64, 32), DataType.FP32)},
        )
        namespace = {"scope": nullcontext}
        exec(transformed, namespace)
        assert namespace["f"](None) == (10, 20)

    @pytest.mark.parametrize("count", [0, 3])
    @pytest.mark.parametrize("rebound", [False, True])
    def test_review_tuple_while_updates_condition(self, count, rebound):
        """Bound the probe so a stale loop-condition binding fails without hanging."""
        initialization = "x = 0; x = x + 0; y = 0" if rebound else "x, y = (0, 0)"
        transformed = self._transform(f"""
            def f(count):
                {initialization}
                while x < count and tick():
                    x, y = (x + 1, y + 2)
                return x, y
        """)
        ticks = []

        def tick():
            ticks.append(None)
            assert len(ticks) <= count, "while condition kept reading the incoming value"
            return True

        namespace = {"tick": tick}
        exec(transformed, namespace)
        assert namespace["f"](count) == (count, count * 2)
        assert len(ticks) == count

    @pytest.mark.parametrize("active", [False, True])
    @pytest.mark.parametrize("scope", ["if flag", "for i in range(count)", "while count > 0 and tick()"])
    @pytest.mark.parametrize("unpack_shape", [False, True])
    def test_review_conditional_tuple_retains_shape_predecessor(self, active, scope, unpack_shape):
        """Untaken branches and zero-trip loops still need the suppressed shape initialization."""
        initialization = "rows, cols = a.shape" if unpack_shape else "rows = a.shape[0]; cols = a.shape[1]"
        transformed = self._transform(
            f"""
                def f(a, flag, count):
                    {initialization}
                    {scope}:
                        rows, cols = (rows // 2, cols // 2)
                        count = 0
                    return rows, cols
            """,
            tensor_meta={"a": TensorMeta((64, 32), DataType.FP32)},
        )
        ticks = []

        def tick():
            ticks.append(None)
            assert len(ticks) <= 1, "while condition kept reading the incoming count"
            return True

        namespace = {"tick": tick}
        exec(transformed, namespace)
        assert namespace["f"](None, active, int(active)) == ((32, 16) if active else (64, 32))

    @pytest.mark.parametrize("flag", [False, True])
    def test_review_value_preserving_tuple_keeps_shape(self, flag):
        """Even writing back the same shape must preserve the false path's initial value."""
        transformed = self._transform(
            """
                def f(a, flag):
                    rows, cols = a.shape
                    if flag:
                        rows, cols = (rows, cols)
                    return rows, cols
            """,
            tensor_meta={"a": TensorMeta((64, 32), DataType.FP32)},
        )
        namespace = {}
        exec(transformed, namespace)
        assert namespace["f"](None, flag) == (64, 32)

    @pytest.mark.parametrize(
        "scope",
        [
            "for i in pl.range(0):\n    rows = 2\n",
            "for i in pl.range(-1):\n    rows = 2\n",
            "for i in pl.unroll(0):\n    rows = 2\n",
            "if flag > 0:\n    for i in pl.range(0):\n        rows = 2\n",
        ],
    )
    def test_empty_loop_retains_literal_shape_bound(self, scope):
        """An unreachable write cannot turn a later static unroll into a runtime bound."""
        source = (
            "def f(a: pl.Tensor, flag: pl.Scalar[pl.INT32]):\n"
            "    rows = a.shape[0]\n"
            + textwrap.indent(scope, "    ")
            + "    for j in pl.unroll(rows):\n        copied = a\n    return a\n"
        )
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INT32},
        )
        specialized = specialize("EmptyLoopShape", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    def test_empty_loop_local_does_not_constrain_later_tensor_binding(self):
        """A dead scalar binding cannot fix the type of a later tensor local."""
        source = textwrap.dedent("""
            def f(a: pl.Tensor):
                for i in pl.range(0):
                    scratch = 0
                scratch = a
                return scratch
        """)
        ctx = _make_ctx("f", source, param_names=["a"], tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)})
        self._verify_ssa(pl.parse_program(specialize("EmptyLoopLocal", [ctx])))

    @pytest.mark.parametrize("iterator", ["range", "unroll"])
    def test_empty_loop_induction_shadows_tensor_alias_and_restores_it(self, iterator):
        """A retained empty body sees its scalar index and preserves the outer tensor."""
        source = textwrap.dedent(f"""
            def f(i: pl.Tensor):
                i, spare = (i, i)
                for i in pl.{iterator}(0):
                    scratch = pl.create_tensor([i], dtype=pl.FP32)
                return i
        """)
        ctx = _make_ctx("f", source, param_names=["i"], tensor_meta={"i": TensorMeta((4, 8), DataType.FP32)})
        specialized = specialize("EmptyInduction", [ctx])
        assert "pl.create_tensor([i], dtype=pl.FP32)" in specialized
        program = self._verify_ssa(pl.parse_program(specialized))
        function = program.get_function("f")
        assert function is not None and len(function.return_types) == 1
        ir.assert_structural_equal(function.return_types[0], ir.TensorType([4, 8], DataType.FP32))
        assert isinstance(function.body, ir.SeqStmts)
        returned = function.body.stmts[-1]
        assert isinstance(returned, ir.ReturnStmt) and isinstance(returned.value[0], ir.Var)
        assert returned.value[0].name_hint.startswith("i_v1")

    def test_empty_explicit_carry_preserves_conditional_shape_predecessor(self):
        """A zero-trip carry still publishes a result on the taken branch."""
        source = textwrap.dedent("""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INT32]):
                rows = a.shape[0]
                if flag > 0:
                    for i, (carry,) in pl.range(0, init_values=(2,)):
                        rows, = pl.yield_(carry + 1)
                return rows
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INT32},
        )
        program = pl.parse_program(specialize("EmptyCarryShapePredecessor", [ctx]))
        function = program.get_function("f")
        assert function is not None and isinstance(function.body, ir.SeqStmts)
        predecessor, conditional, _ = function.body.stmts
        assert isinstance(predecessor, ir.AssignStmt)
        assert isinstance(predecessor.value, ir.ConstInt) and predecessor.value.value == 4
        assert isinstance(conditional, ir.IfStmt)
        body = conditional.then_body
        loop = body.stmts[0] if isinstance(body, ir.SeqStmts) else body
        assert isinstance(loop, ir.ForStmt)
        assert isinstance(loop.stop, ir.ConstInt) and loop.stop.value == 0
        assert len(loop.iter_args) == 1
        initial = loop.iter_args[0].initValue
        assert isinstance(initial, ir.ConstInt) and initial.value == 2

    @pytest.mark.parametrize("terminator", ["return out", "break", "continue"])
    def test_nonjoining_branch_retains_fallthrough_tensor_metadata(self, terminator):
        """Only a branch reaching the following shape consumer supplies its descriptor."""
        body = (
            "if flag > 0:\n    x = a\n    x, spare = (a, a)\n"
            f"    {terminator}\n"
            "else:\n    x = b\n    x, spare = (b, b)\n"
            "for j in pl.unroll(x.shape[0]):\n    copied = x\n"
        )
        if terminator != "return out":
            body = "for iteration in pl.range(1):\n" + textwrap.indent(body, "    ")
        source = (
            "def f(a: pl.Tensor, b: pl.Tensor, out: pl.Tensor, flag: pl.Scalar[pl.INT32]):\n"
            + textwrap.indent(body, "    ")
            + "    return out\n"
        )
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "b", "out", "flag"],
            tensor_meta={
                "a": TensorMeta((4, 8), DataType.FP32),
                "b": TensorMeta((8, 16), DataType.BF16),
                "out": TensorMeta((4, 8), DataType.FP32),
            },
            scalar_dtypes={"flag": DataType.INT32},
        )
        specialized = specialize("JoiningDescriptor", [ctx])
        assert "pl.unroll(8)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize(
        ("break_body", "fallthrough", "expected"),
        [
            ("x = a\nx, spare = (a, a)\n", "b", (4, 8)),
            ("x = a\nwith pl.scope():\n    x = unknown(b)\n", "a", (8, 4)),
        ],
        ids=["known-descriptor", "unknown-scoped-write"],
    )
    def test_break_branch_descriptor_is_rejoined_at_owning_loop_exit(self, break_body, fallthrough, expected):
        """A known fallthrough descriptor cannot conceal a different break predecessor."""
        transformed = self._transform(
            "def f(a, b, flag):\n    for i in pl.range(1):\n        if flag:\n"
            + textwrap.indent(break_body, "            ")
            + "            break\n        else:\n"
            + f"            x = {fallthrough}\n            x, spare = ({fallthrough}, {fallthrough})\n"
            + "    return x.shape[0]\n",
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32), "b": TensorMeta((8, 16), DataType.BF16)},
        )
        namespace = {
            "pl": SimpleNamespace(
                range=range, scope=nullcontext, tensor=SimpleNamespace(dim=lambda value, i: value.shape[i])
            ),
            "unknown": lambda value: value,
        }
        exec(transformed, namespace)
        a, b = SimpleNamespace(shape=(4, 8)), SimpleNamespace(shape=(8, 16))
        assert namespace["f"](a, b, True) == expected[0]
        assert namespace["f"](a, b, False) == expected[1]

    @pytest.mark.parametrize("scope", ["with pl.scope()", "if 1 < 2", "for i in pl.range(1)"])
    def test_review_outer_rebind_after_nested_tuple_changes_type(self, scope):
        """A nested tuple's binding depth cannot pin later outer writes to one scalar type."""
        source = textwrap.dedent(f"""
            def f():
                {scope}:
                    x, y = (1, 2)
                x = 3
                x = 4.0
                return x
        """)
        program = pl.parse_program(specialize("OuterRebind", [_make_ctx("f", source, auto_scope=False)]))
        function = program.get_function("f")
        assert function is not None and isinstance(function.body, ir.SeqStmts)
        returned = function.body.stmts[-1]
        assert isinstance(returned, ir.ReturnStmt)
        assert isinstance(returned.value[0].type, ir.ScalarType)
        assert returned.value[0].type.dtype == DataType.FP32
        passes.convert_to_ssa()(program)

    @pytest.mark.parametrize("iterator", ["range", "unroll"])
    @pytest.mark.parametrize("output", ["x", "result"])
    def test_final_review_explicit_range_carry_shadows_outer_alias(self, iterator, output):
        """Every iteration reads the carried value, while distinct outputs restore x."""
        source = textwrap.dedent(f"""
            def f():
                x, spare = (0, 0)
                x, spare = (1, 0)
                for i, (x,) in pl.{iterator}(3, init_values=(x,)):
                    {output}, = pl.yield_(x + 1)
                return x, {output}
        """)
        carry = []

        def explicit_range(count, *, init_values):
            carry[:] = init_values
            for i in range(count):
                yield i, tuple(carry)

        def yield_value(value):
            carry[:] = [value]
            return (value,)

        namespace = {"pl": SimpleNamespace(**{iterator: explicit_range}, yield_=yield_value)}
        transformed = self._transform(source)
        # A Python loop updates its yield-LHS eagerly, unlike the DSL's
        # post-loop result. Check that the DSL yield reads its header carry.
        assert "pl.yield_(x + 1)" in transformed
        exec(transformed, namespace)
        assert namespace["f"]() == ((4, 4) if output == "x" else (1, 4))
        self._verify_ssa(pl.parse_program(specialize("ExplicitRangeCarry", [_make_ctx("f", source)])))

    @pytest.mark.parametrize("assignment", ["x, y = (VALUE, VALUE + 1)", "x = VALUE; y = VALUE + 1"])
    def test_final_review_definite_branch_kills_partial_binding(self, assignment):
        """The first partial value is dead; either later arm initializes both results."""
        then_body = assignment.replace("VALUE", "3")
        else_body = assignment.replace("VALUE", "5")
        source = textwrap.dedent(f"""
            def f(a, b):
                if a > 0:
                    x, y = (1, 2)
                if b > 0:
                    {then_body}
                else:
                    {else_body}
                return x, y
        """)
        namespace = {}
        exec(self._transform(source), namespace)
        for a in [0, 1]:
            for b in [0, 1]:
                assert namespace["f"](a, b) == ((3, 4) if b else (5, 6))
        source = source.replace("def f(a, b):", "def f(a: pl.Scalar[pl.INDEX], b: pl.Scalar[pl.INDEX]):")
        ctx = _make_ctx(
            "f", source, param_names=["a", "b"], scalar_dtypes={"a": DataType.INDEX, "b": DataType.INDEX}
        )
        self._verify_ssa(pl.parse_program(specialize("KilledPartial", [ctx])))

    @pytest.mark.parametrize("control", ["for i in pl.unroll(1)", "while count > 0"])
    def test_final_review_loop_local_survives_sibling_conditional(self, control):
        """The untaken sibling branch retains a value initialized by the preceding loop."""
        source = textwrap.dedent(f"""
            def f(flag):
                count = 1
                {control}:
                    x, y = (1, 2)
                    count = count - 1
                if flag > 0:
                    x, y = (3, 4)
                return x, y
        """)
        namespace = {"pl": SimpleNamespace(unroll=range)}
        exec(self._transform(source), namespace)
        for flag in [0, 1]:
            assert namespace["f"](flag) == ((3, 4) if flag else (1, 2))
        source = source.replace("def f(flag):", "def f(flag: pl.Scalar[pl.INDEX]):")
        ctx = _make_ctx("f", source, param_names=["flag"], scalar_dtypes={"flag": DataType.INDEX})
        # Loop-local live-outs without an incoming value remain pre-SSA here;
        # native execution checks the alias, and parsing checks DSL acceptance.
        pl.parse_program(specialize("LoopLocalExport", [ctx]))

    @pytest.mark.parametrize("control", ["while flag > 0", "for j in pl.range(flag)"])
    @pytest.mark.parametrize("initial_alias", [False, True])
    def test_sibling_zero_trip_loop_retains_active_tuple_alias(self, control, initial_alias):
        """A later loop retains the preceding tuple value when it takes zero trips."""
        initial = "x = 1\n                    x, y = (2, 3)" if initial_alias else "x, y = (1, 2)"
        source = textwrap.dedent(f"""
            def f(flag):
                for i in pl.unroll(1):
                    {initial}
                {control}:
                    x = 3
                    x, y = (4, 5)
                    flag = flag - flag
                return x
        """)
        namespace = {"pl": SimpleNamespace(unroll=range, range=range)}
        exec(self._transform(source), namespace)
        assert namespace["f"](0) == (2 if initial_alias else 1)
        assert namespace["f"](1) == 4
        source = source.replace("def f(flag):", "def f(flag: pl.Scalar[pl.INT32]):")
        ctx = _make_ctx("f", source, param_names=["flag"], scalar_dtypes={"flag": DataType.INT32})
        # The initial live-out is defined by an unrolled body, so native Python
        # execution checks the value and parsing checks the emitted DSL scope.
        pl.parse_program(specialize("SiblingLoopAlias", [ctx]))

    @pytest.mark.parametrize("control", ["if flag > 0", "for i in pl.range(flag)"])
    def test_implicit_first_out_survives_unknown_scoped_rebinding(self, control):
        """A no-return entry keeps its original output despite an unknown scoped RHS."""
        source = textwrap.dedent(f"""
            def f(out: pl.Out[pl.Tensor], flag: pl.Scalar[pl.INT32]):
                {control}:
                    out = pl.create_tensor([4, 8], dtype=pl.FP32)
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["out", "flag"],
            tensor_meta={"out": TensorMeta((4, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INT32},
        )
        program = self._verify_ssa(pl.parse_program(specialize("ImplicitOutput", [ctx])))
        function = program.get_function("f")
        assert function is not None and len(function.return_types) == 1
        assert isinstance(function.body, ir.SeqStmts)
        saved = next(
            stmt
            for stmt in function.body.stmts
            if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("pypto_implicit_out")
        )
        assert isinstance(saved.value, ir.Var)
        assert saved.value.unique_id == function.params[0].unique_id
        returned = function.body.stmts[-1]
        assert isinstance(returned, ir.ReturnStmt)
        assert isinstance(returned.value[0], ir.Var)
        assert returned.value[0].unique_id == saved.var.unique_id

    @pytest.mark.parametrize(
        "assignment,scoped,expected_rows,expected_dtype",
        [
            ("x, spare = (x, x)", False, 4, "pl.FP32"),
            ("x = x; x, spare = (x, x)", False, 4, "pl.FP32"),
            ("x, y = (y, x)", False, 8, "pl.BF16"),
            ("x, spare = (x, x)", True, 4, "pl.FP32"),
        ],
        ids=["duplicate", "existing-alias", "swap", "scoped-copy"],
    )
    def test_final_review_tuple_operand_metadata_snapshot(
        self, assignment, scoped, expected_rows, expected_dtype
    ):
        """Known operands keep shape/dtype; a swap snapshots both before either target changes."""
        binding = "with pl.scope():\n    " + assignment if scoped else assignment
        source = (
            "def f(x: pl.Tensor, y: pl.Tensor):\n"
            + textwrap.indent(binding, "    ")
            + "\n    rows, cols = x.shape\n    old_rows, old_cols = y.shape\n"
            "    total = 0\n    for i in pl.unroll(rows):\n        total = total + 1\n"
            "    return pl.create_tensor([total, cols], dtype=x.dtype)\n"
        )
        meta = {"x": TensorMeta((4, 16), DataType.FP32), "y": TensorMeta((8, 32), DataType.BF16)}
        specialized = specialize(
            "TupleMetadata",
            [_make_ctx("f", source, param_names=["x", "y"], tensor_meta=meta, auto_scope=not scoped)],
        )
        assert f"pl.unroll({expected_rows})" in specialized
        assert f"dtype={expected_dtype}" in specialized
        assert f"pl.unroll({expected_rows})" in self._transform(
            source.replace("pl.unroll(rows)", "pl.unroll(x.shape[0])"), tensor_meta=meta
        )
        # The swap must preserve the original x metadata for its second RHS
        # operand even after the first target acquires y's metadata.
        expected_old_rows = 4 if expected_rows == 8 else 8
        transformed = self._transform(source, tensor_meta=meta)
        assert "old_rows" not in transformed
        assert f"pl.unroll({expected_old_rows})" in self._transform(
            source.replace("pl.unroll(rows)", "pl.unroll(old_rows)"), tensor_meta=meta
        )
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize("control", ["if flag > 0", "for iteration in pl.range(flag)"])
    @pytest.mark.parametrize(
        "nested",
        ["if False:\n    rows = 2", "if True:\n    pass\nelse:\n    rows = 2", "if 1 > 2:\n    rows = 2"],
        ids=["false", "true-else", "literal-comparison"],
    )
    def test_final_review_dead_nested_shape_write_keeps_unroll_constant(self, control, nested):
        """Pruned writes must not materialize a runtime alias for a static bound."""
        source = (
            "def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX]):\n    rows = a.shape[0]\n"
            f"    {control}:\n"
            + textwrap.indent(nested, "        ")
            + "\n    total = 0\n    for i in pl.unroll(rows):\n        total = total + 1\n    return total\n"
        )
        meta = {"a": TensorMeta((4, 16), DataType.FP32)}
        ctx = _make_ctx(
            "f", source, param_names=["a", "flag"], tensor_meta=meta, scalar_dtypes={"flag": DataType.INDEX}
        )
        specialized = specialize("DeadShapeWrite", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("[tids[k] for k in range(4)]", [10, 11, 12, 13]),
            ("[tids[k] for k in range(4) if k != 1]", [10, 12, 13]),
            ("{tids[k] for k in range(4)}", {10, 11, 12, 13}),
            ("{k: tids[k] for k in range(4)}", {0: 10, 1: 11, 2: 12, 3: 13}),
            ("tuple(tids[k] for k in range(4))", (10, 11, 12, 13)),
            ("[(k, j) for k in range(k) for j in range(k + 1)]", [(0, 0), (1, 0), (1, 1)]),
            ("[(k, j) for k, j in [(2, 3), (4, 5)]]", [(2, 3), (4, 5)]),
            ("[[k for k in range(k)] for k in range(3)]", [[], [0], [0, 1]]),
        ],
        ids=["list", "filter", "set", "dict", "generator", "multiple", "unpacking", "nested"],
    )
    def test_ai_review_comprehension_shadows_tuple_alias(self, expression, expected):
        """Comprehension binders shadow outer aliases and restore them after evaluation."""
        transformed = self._transform(f"""
            def f(tids):
                k, spare = (0, 0)
                k, spare = (2, 0)
                result = {expression}
                return result, k
        """)
        namespace = {}
        exec(transformed, namespace)
        assert namespace["f"]([10, 11, 12, 13]) == (expected, 2)

    @pytest.mark.parametrize("filtered", [False, True])
    def test_ai_review_dependency_comprehension_keeps_each_producer(self, filtered):
        """The parser must substitute the binder into every producer slot, including filters."""
        condition = " if k != 1" if filtered else ""
        transformed = self._transform(f"""
            def f(tids):
                k, spare = (0, 0)
                k, spare = (1, 0)
                consume(deps=[tids[k] for k in range(4){condition}])
        """)
        comp = next(node for node in ast.walk(ast.parse(transformed)) if isinstance(node, ast.ListComp))
        parser = ASTParser("<dependency regression>", [])
        entries = parser.dispatch.unroll_deps_comprehension(comp, "submit", ir.Span.unknown())
        assert [ast.unparse(entry) for entry in entries] == [
            f"tids[{index}]" for index in range(4) if not filtered or index != 1
        ]

    @pytest.mark.parametrize("binding", ["shape", "tensor"])
    def test_ai_review_comprehension_shadows_metadata(self, binding):
        """A binder cannot inherit a same-named tensor's metadata or a folded dimension."""
        setup = "k = a.shape[0]" if binding == "shape" else "k = a"
        expression = "[k for k in range(3)]" if binding == "shape" else "[k.shape[0] for k in items]"
        source = f"""
            def f(a, items):
                {setup}
                result = {expression}
                return result, k
        """
        transformed = self._transform(source, tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)})
        namespace = {
            "pl": SimpleNamespace(tensor=SimpleNamespace(dim=lambda value, index: value.shape[index]))
        }
        exec(transformed, namespace)
        tensor = SimpleNamespace(shape=(4, 8))
        result, outer = namespace["f"](tensor, [SimpleNamespace(shape=(7,)), SimpleNamespace(shape=(9,))])
        assert result == ([0, 1, 2] if binding == "shape" else [7, 9])
        assert outer == (4 if binding == "shape" else tensor)

    @pytest.mark.parametrize("iterator", ["range", "unroll", "while_"])
    @pytest.mark.parametrize("count", [0, 3])
    @pytest.mark.parametrize("transparent", [False, True])
    def test_ai_review_explicit_loop_result_survives_sibling_branch(self, iterator, count, transparent):
        """Owned yield results have an initialized value even when the loop takes zero trips."""
        header = (
            f"for (carry,) in pl.while_(init_values=(0,)):\n    pl.cond(carry < {count})"
            if iterator == "while_"
            else f"for i, (carry,) in pl.{iterator}({count}, init_values=(0,)):"
        )
        loop_body = "out, = pl.yield_(carry + 1)"
        if transparent:
            loop_body = "with pl.scope():\n    " + loop_body
        source = (
            "def f(flag: pl.Scalar[pl.INDEX]):\n"
            + textwrap.indent(header, "    ")
            + "\n"
            + textwrap.indent(loop_body, "        ")
            + "\n    if flag > 0:\n        out, spare = (9, 0)\n    return out\n"
        )
        transformed = self._transform(source)
        assert "out_v1" not in transformed
        ctx = _make_ctx(
            "f",
            source,
            param_names=["flag"],
            scalar_dtypes={"flag": DataType.INDEX},
            auto_scope=not transparent,
        )
        self._verify_ssa(pl.parse_program(specialize("ExplicitLoopExport", [ctx])))

    @pytest.mark.parametrize("constant_kind", ["constexpr", "free"])
    @pytest.mark.parametrize("control", ["if flag > 0", "for iteration in pl.range(flag)"])
    @pytest.mark.parametrize("condition", ["enabled", "not enabled", "enabled == True"])
    def test_ai_review_nested_static_dead_write_keeps_shape(self, constant_kind, control, condition):
        """Pre-scanning nested writes must fold external constants before materializing shapes."""
        enabled = condition == "not enabled"
        signature = ", enabled: pl.constexpr" if constant_kind == "constexpr" else ""
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX]{signature}):
                rows = a.shape[0]
                {control}:
                    if {condition}:
                        rows = 2
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return total
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INDEX},
        )
        if constant_kind == "constexpr":
            ctx.param_names.append("enabled")
            ctx.constexpr_values = {"enabled": str(enabled)}
        else:
            ctx.py_globals = {"enabled": enabled}
        specialized = specialize("NestedStaticWrite", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize("constant_kind", ["constexpr", "free"])
    def test_ai_review_nested_static_condition_respects_loop_shadow(self, constant_kind):
        """A same-named loop binder is runtime data, even if the outer constant is false."""
        source = textwrap.dedent("""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX], enabled: pl.constexpr):
                rows = a.shape[0]
                if flag > 0:
                    for enabled in pl.range(3):
                        if enabled > 0:
                            rows = 2
                return rows
        """)
        if constant_kind == "free":
            source = source.replace(", enabled: pl.constexpr", "")
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INDEX},
        )
        if constant_kind == "constexpr":
            ctx.param_names.append("enabled")
            ctx.constexpr_values = {"enabled": "False"}
        else:
            ctx.py_globals = {"enabled": False}
        if constant_kind == "constexpr":
            with pytest.raises(ValueError, match="compile-time constant and cannot be assigned"):
                specialize("ShadowedStaticWrite", [ctx])
            return
        specialized = specialize("ShadowedStaticWrite", [ctx])
        assert "rows_v1 = 4" in specialized
        assert "rows_v1 = 2" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize(
        "assignment,expected_rows,expected_dtype",
        [
            ("x, spare = [x, x]", 4, "pl.FP32"),
            ("x = x; x, spare = [x, x]", 4, "pl.FP32"),
            ("x, y = [y, x]", 8, "pl.BF16"),
        ],
        ids=["duplicate", "existing-alias", "swap"],
    )
    def test_ai_review_list_operand_metadata_snapshot(self, assignment, expected_rows, expected_dtype):
        """List-valued unpacking has the same operand snapshot contract as tuple-valued unpacking."""
        source = textwrap.dedent(f"""
            def f(x: pl.Tensor, y: pl.Tensor):
                {assignment}
                rows, cols = x.shape
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return pl.create_tensor([total, cols], dtype=x.dtype)
        """)
        meta = {"x": TensorMeta((4, 16), DataType.FP32), "y": TensorMeta((8, 32), DataType.BF16)}
        ctx = _make_ctx("f", source, param_names=["x", "y"], tensor_meta=meta)
        specialized = specialize("ListMetadata", [ctx])
        assert f"pl.unroll({expected_rows})" in specialized
        assert f"dtype={expected_dtype}" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    def test_ai_review_nested_list_metadata_snapshot(self):
        """Nested unpacking is handled by the transformer; the parser accepts flat targets only."""
        transformed = self._transform(
            """
                def f(x, y):
                    x, (y, spare) = [y, [x, x]]
                    return x.shape[0], y.shape[0], spare.shape[0]
            """,
            tensor_meta={"x": TensorMeta((4, 16), DataType.FP32), "y": TensorMeta((8, 32), DataType.BF16)},
        )
        returned = next(node for node in ast.walk(ast.parse(transformed)) if isinstance(node, ast.Return))
        assert isinstance(returned.value, ast.Tuple)
        assert [ast.literal_eval(element) for element in returned.value.elts] == [8, 4, 4]

    @pytest.mark.parametrize("control", ["if flag > 0", "for i in pl.range(flag)"])
    def test_ai_review_list_shape_identity_keeps_unroll_constant(self, control):
        """List operands that write back the same folded dimensions need no runtime predecessor."""
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX]):
                rows, cols = a.shape
                {control}:
                    rows, cols = [rows, cols]
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return total
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INDEX},
        )
        specialized = specialize("ListShapeIdentity", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize("control", ["if flag > 0", "for i in pl.range(flag)"])
    def test_followup_dynamic_shape_unpack_reuses_incoming_tuple(self, control):
        """An untaken shape write preserves both initialized dimensions."""
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX]):
                rows, cols = (7, 8)
                {control}:
                    rows, cols = a.shape
                return rows, cols
        """)
        meta = {"a": TensorMeta((DynDim("M", "M", 32), 8), DataType.FP32)}
        transformed = self._transform(
            source.replace("a: pl.Tensor, flag: pl.Scalar[pl.INDEX]", "a, flag"), tensor_meta=meta
        )
        namespace = {"pl": SimpleNamespace(range=range, tensor=SimpleNamespace(dim=lambda a, i: a.shape[i]))}
        exec(transformed, namespace)
        for flag in [0, 1]:
            assert namespace["f"](SimpleNamespace(shape=(5, 8)), flag) == ((5, 8) if flag else (7, 8))
        ctx = _make_ctx(
            "f", source, param_names=["a", "flag"], tensor_meta=meta, scalar_dtypes={"flag": DataType.INDEX}
        )
        self._verify_ssa(pl.parse_program(specialize("DynamicShapeWrite", [ctx])))

    @pytest.mark.parametrize("wrapper", ["True", "enabled", "1 < 2"])
    def test_followup_pruned_branch_exports_initialized_loop(self, wrapper):
        """Flattening a selected branch must also publish the loop's owned results."""
        source = textwrap.dedent(f"""
            def f(flag: pl.Scalar[pl.INDEX], enabled: pl.constexpr):
                if {wrapper}:
                    for i, (carry,) in pl.range(0, init_values=(0,)):
                        out, = pl.yield_(carry + 1)
                if flag > 0:
                    out, spare = (9, 0)
                return out
        """)
        ctx = _make_ctx("f", source, param_names=["flag", "enabled"], scalar_dtypes={"flag": DataType.INDEX})
        ctx.constexpr_values = {"enabled": "True"}
        specialized = specialize("PrunedLoopExport", [ctx])
        assert "out_v1" not in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize("count", [0, 3])
    def test_followup_explicit_loop_is_definite_branch_write(self, count):
        """Explicit loop outputs initialize a complete branch even on zero trips."""
        source = textwrap.dedent(f"""
            def f(early: pl.Scalar[pl.INDEX], flag: pl.Scalar[pl.INDEX]):
                if early > 0:
                    out, spare = (1, 0)
                if flag > 0:
                    for i, (carry,) in pl.range({count}, init_values=(0,)):
                        out, = pl.yield_(carry + 1)
                else:
                    out = 9
                return out
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["early", "flag"],
            scalar_dtypes={"early": DataType.INDEX, "flag": DataType.INDEX},
        )
        self._verify_ssa(pl.parse_program(specialize("DefiniteLoopExport", [ctx])))

    @pytest.mark.parametrize("condition", ["a.shape[0] > 8", "a.shape[1] < 4"])
    def test_followup_shape_condition_prunes_dead_write(self, condition):
        """Write analysis must apply the same safe descriptor folding as the emitted body."""
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX]):
                rows = a.shape[0]
                if flag > 0:
                    if {condition}:
                        rows = 2
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return total
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INDEX},
        )
        specialized = specialize("ShapeCondition", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    def test_followup_comprehension_store_does_not_materialize_outer_shape(self):
        """Dependency-comprehension binders cannot invalidate an enclosing folded dimension."""
        source = textwrap.dedent("""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX]):
                rows = a.shape[0]
                ready = pl.system.task_invalid()
                if flag > 0:
                    pl.system.task_dummy(deps=[ready for rows in range(3)])
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return total
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INDEX},
        )
        specialized = specialize("ComprehensionStores", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize("count", [0, 3])
    @pytest.mark.parametrize("output", ["x", "result"])
    def test_followup_explicit_tensor_carry_keeps_identity_metadata(self, count, output):
        """Identity carries preserve init/yield metadata for all published output spellings."""
        source = textwrap.dedent(f"""
            def f(x: pl.Tensor):
                for i, (x,) in pl.range({count}, init_values=(x,)):
                    {output}, = pl.yield_(x)
                return pl.create_tensor([1, {output}.shape[1]], dtype={output}.dtype)
        """)
        ctx = _make_ctx("f", source, param_names=["x"], tensor_meta={"x": TensorMeta((4, 8), DataType.FP32)})
        specialized = specialize("TensorCarryMetadata", [ctx])
        assert "dtype=pl.FP32" in specialized
        assert "pl.create_tensor([1, 8]" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    def test_followup_changing_tensor_carry_does_not_fold_initial_shape(self):
        """Later iterations can see a different descriptor than the initial tensor."""
        transformed = self._transform(
            """
                def f(x, other):
                    for i, (x,) in pl.range(3, init_values=(x,)):
                        consume(x.shape[0])
                        result, = pl.yield_(other)
                    return result
            """,
            tensor_meta={"x": TensorMeta((4, 8), DataType.FP32), "other": TensorMeta((16, 8), DataType.FP32)},
        )
        assert "consume(4)" not in transformed
        assert "pl.tensor.dim(x, 0)" in transformed

    def test_followup_shape_condition_respects_rebound_tensor(self):
        """An incoming descriptor cannot prune a test after its source was overwritten."""
        source = """
            def f(a, other, flag):
                rows = a.shape[0]
                if flag > 0:
                    a = other
                    if a.shape[0] > 8:
                        rows = 2
                return rows
        """
        transformed = self._transform(
            source,
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32), "other": TensorMeta((16, 8), DataType.FP32)},
        )
        namespace = {
            "pl": SimpleNamespace(tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]))
        }
        exec(transformed, namespace)
        for flag in [0, 1]:
            assert namespace["f"](SimpleNamespace(shape=(4, 8)), SimpleNamespace(shape=(16, 8)), flag) == (
                2 if flag else 4
            )

    @pytest.mark.parametrize("rhs", ["(y, x)", "[y, x]"])
    def test_latest_review_return_signature_uses_emitted_tensor_meta(self, rhs):
        """A rebound return has its operand's descriptor, rather than its old parameter's ABI."""
        source = textwrap.dedent(f"""
            def f(x: pl.Tensor, y: pl.Tensor):
                x, spare = {rhs}
                return x
        """)
        meta = {"x": TensorMeta((4, 8), DataType.FP32), "y": TensorMeta((8, 16), DataType.BF16)}
        ctx = _make_ctx("f", source, param_names=["x", "y"], tensor_meta=meta)
        specialized = specialize("ReboundReturn", [ctx])
        assert "-> pl.Tensor[[8, 16], pl.BF16]" in specialized
        program = pl.parse_program(specialized)
        self._verify_ssa(program)

    @pytest.mark.parametrize("control", ["if flag > 0", "for i in pl.range(flag)"])
    @pytest.mark.parametrize("restore", ["rows = a.shape[0]", "rows, cols = a.shape"])
    def test_latest_review_direct_shape_restore_retains_literal_bound(self, control, restore):
        """Only values escaping the region require a runtime shape predecessor."""
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX]):
                rows = a.shape[0]
                {control}:
                    rows = 2
                    {restore}
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return total
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INDEX},
        )
        specialized = specialize("DirectShapeRestore", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize("count", [0, 3])
    def test_latest_review_equal_descriptor_carry_keeps_static_meta(self, count):
        """Different tensor names can have the same invariant descriptor across a backedge."""
        source = textwrap.dedent(f"""
            def f(x: pl.Tensor, other: pl.Tensor):
                for i, (x,) in pl.range({count}, init_values=(x,)):
                    x, = pl.yield_(other)
                rows, cols = x.shape
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return pl.create_tensor([total, cols], dtype=x.dtype)
        """)
        meta = {name: TensorMeta((4, 8), DataType.FP32) for name in ["x", "other"]}
        ctx = _make_ctx("f", source, param_names=["x", "other"], tensor_meta=meta)
        specialized = specialize("EqualCarryDescriptor", [ctx])
        assert "pl.unroll(4)" in specialized
        assert "dtype=pl.FP32" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    def test_latest_review_restore_after_break_keeps_escaping_value(self):
        """An unreachable restore cannot erase the value carried by an earlier loop exit."""
        transformed = self._transform(
            """
                def f(a, count):
                    rows = a.shape[0]
                    for i in pl.range(count):
                        rows = 2
                        if i > 0:
                            break
                        rows = a.shape[0]
                    total = 0
                    for j in pl.unroll(rows):
                        total = total + 1
                    return total
            """,
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)},
        )
        namespace = {"pl": SimpleNamespace(range=range, unroll=range)}
        exec(transformed, namespace)
        for count in [0, 1, 3]:
            assert namespace["f"](None, count) == (2 if count == 3 else 4)

    @pytest.mark.parametrize("parameter", ["pl.Tensor", "pl.Out[pl.Tensor]"])
    def test_review_unknown_tuple_result_does_not_inherit_parameter_abi(self, parameter):
        """An unannotated result must be deduced from its body at dependency calls."""
        meta = {"x": TensorMeta((4, 8), DataType.FP32)}
        helper = _make_ctx(
            "helper",
            f"def helper(x: {parameter}):\n"
            "    x, spare = (pl.create_tensor([8, 16], dtype=pl.BF16), x)\n"
            "    return x\n",
            func_type="incore",
            param_names=["x"],
            tensor_meta=meta,
        )
        caller = _make_ctx(
            "f",
            "def f(x: pl.Tensor):\n    x = helper(x)\n    return x\n",
            param_names=["x"],
            tensor_meta=meta,
            dep_names=["helper"],
        )
        source = specialize("UnknownTupleResult", [helper, caller])
        program = pl.parse_program(source)
        function = program.get_function("f")
        assert function is not None
        assert isinstance(function.body, ir.SeqStmts)
        returned = function.body.stmts[-1]
        assert isinstance(returned, ir.ReturnStmt)
        result = returned.value[0].type
        expected = ir.TensorType([8, 16], DataType.BF16)
        ir.assert_structural_equal(result, expected)
        for name in ["helper", "f"]:
            declared = program.get_function(name)
            assert declared is not None and len(declared.return_types) == 1
            ir.assert_structural_equal(declared.return_types[0], expected)
        self._verify_ssa(program)

    @pytest.mark.parametrize(
        "binding",
        [
            "x, y = y, x",
            "x, y = [y, x]",
            "x, y = helper(x, True, y)",
            "x, y = helper(y=y, enabled=True, x=x)",
        ],
    )
    def test_review_tuple_swap_propagates_tensor_kinds(self, binding):
        """Simultaneous rebinding carries both operands' descriptor and object kind."""
        meta = {"x": TensorMeta((4, 8), DataType.FP32), "y": TensorMeta((8, 16), DataType.BF16)}
        helper = _make_ctx(
            "helper",
            "def helper(x: pld.DistributedTensor, enabled: pl.constexpr, y: pl.Tensor):\n"
            "    if enabled:\n        return y, x\n    else:\n        return x, y\n",
            func_type="incore",
            param_names=["x", "enabled", "y"],
            tensor_meta=meta,
        )
        helper.constexpr_values = {"enabled": "True"}
        ctx = _make_ctx(
            "f",
            f"def f(x: pld.DistributedTensor, y: pl.Tensor):\n    {binding}\n    return x, y\n",
            param_names=["x", "y"],
            tensor_meta=meta,
            dep_names=["helper"],
        )
        source = specialize("TupleTensorKinds", [helper, ctx])
        assert "-> tuple[pl.Tensor[[8, 16], pl.BF16], pld.DistributedTensor[[4, 8], pl.FP32]]" in source
        program = pl.parse_program(source)
        function = program.get_function("f")
        assert function is not None
        plain, distributed = function.return_types
        assert isinstance(plain, ir.TensorType) and not isinstance(plain, ir.DistributedTensorType)
        assert isinstance(distributed, ir.DistributedTensorType)
        self._verify_ssa(program)

    @pytest.mark.parametrize("condition", ["False", "1 > 2", "enabled"])
    def test_review_dead_successor_does_not_merge_branch_temporary(self, condition):
        """A read removed by static pruning cannot require a cross-type branch export."""
        ctx = _make_ctx(
            "f",
            textwrap.dedent(f"""
                def f(flag: pl.Scalar[pl.INT32], enabled: pl.constexpr):
                    if flag > 0:
                        scratch = 0
                        scratch, spare = (2.0, 0)
                    else:
                        scratch, spare = (4, 0)
                    if {condition}:
                        unused = scratch
                    return flag
            """),
            param_names=["flag", "enabled"],
            scalar_dtypes={"flag": DataType.INT32},
        )
        ctx.constexpr_values = {"enabled": "False"}
        self._verify_ssa(pl.parse_program(specialize("DeadSuccessor", [ctx])))

    @pytest.mark.parametrize("later_write", ["", "a = b"])
    def test_dead_descriptor_successor_does_not_merge_branch_temporary(self, later_write):
        """Descriptor pruning precedes liveness, including a test before a later write."""
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, b: pl.Tensor, flag: pl.Scalar[pl.INT32]):
                if flag > 0:
                    scratch = 0
                    scratch, spare = (2.0, 0)
                else:
                    scratch, spare = (4, 0)
                if a.shape[0] > 8:
                    unused = scratch
                {later_write}
                return flag
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "b", "flag"],
            tensor_meta={"a": TensorMeta((4, 8), DataType.FP32), "b": TensorMeta((16, 8), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INT32},
        )
        self._verify_ssa(pl.parse_program(specialize("DeadDescriptorSuccessor", [ctx])))

    def test_liveness_descriptor_preview_respects_prior_writes(self):
        """A descriptor changed before a test cannot use the parameter's original shape."""
        source = textwrap.dedent("""
            def f(a, b, flag):
                result = 0
                if flag > 0:
                    scratch = 0
                    scratch, spare = (2, 0)
                else:
                    scratch, spare = (4, 0)
                a = b
                if a.shape[0] > 8:
                    result = scratch
                return result
        """)
        meta = {"a": TensorMeta((4, 8), DataType.FP32), "b": TensorMeta((16, 8), DataType.FP32)}
        namespace = {}
        exec(self._transform(source, tensor_meta=meta), namespace)
        assert namespace["f"](SimpleNamespace(shape=(4, 8)), SimpleNamespace(shape=(16, 8)), 1) == 2
        assert namespace["f"](SimpleNamespace(shape=(4, 8)), SimpleNamespace(shape=(16, 8)), 0) == 4

    @pytest.mark.parametrize("initial_rows", [4, 16])
    @pytest.mark.parametrize("control", ["", "for i in pl.range(2):", "while flag > 0:"])
    def test_liveness_preview_tracks_copied_descriptors_and_invariant_loops(self, initial_rows, control):
        """A known RHS descriptor controls pruning after copies and on unchanged backedges."""
        successor = "if a.shape[0] > 8:\n    unused = scratch\n"
        if control:
            successor = control + "\n" + textwrap.indent(successor + "flag = flag - flag\n", "    ")
        source = (
            "def f(a: pl.Tensor, b: pl.Tensor, flag: pl.Scalar[pl.INT32]):\n"
            "    if flag > 0:\n        scratch = 0\n        scratch = 2.0\n"
            "    else:\n        scratch = 4\n"
            "    a = b\n" + textwrap.indent(successor, "    ") + "    return flag\n"
        )
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "b", "flag"],
            tensor_meta={
                "a": TensorMeta((initial_rows, 8), DataType.FP32),
                "b": TensorMeta((4, 8), DataType.FP32),
            },
            scalar_dtypes={"flag": DataType.INT32},
        )
        self._verify_ssa(pl.parse_program(specialize("KnownDescriptorLiveness", [ctx])))

    @pytest.mark.parametrize("submitted", [False, True])
    def test_liveness_preview_preserves_dependency_variant_identity(self, submitted):
        """Two constexpr variants of one helper must retain distinct returned operands."""
        meta = {"a": TensorMeta((16, 8), DataType.FP32), "b": TensorMeta((4, 8), DataType.FP32)}
        helpers = []
        for index, enabled in enumerate([True, False]):
            helper = _make_ctx(
                f"choose_v{index}",
                "def choose(a: pl.Tensor, b: pl.Tensor, enabled: pl.constexpr):\n"
                "    if enabled:\n        return a\n    else:\n        return b\n",
                func_type="incore",
                param_names=["a", "b", "enabled"],
                tensor_meta=meta,
            )
            helper.source_func_name = "choose"
            helper.constexpr_values = {"enabled": str(enabled)}
            helpers.append(helper)
        calls = (
            "    first, first_done = pl.submit(choose, a, b, True)\n"
            "    second, done = pl.submit(choose, a, b, False)\n"
            if submitted
            else "    first = choose(a, b, True)\n    second = choose(a, b, False)\n"
        )
        ctx = _make_ctx(
            "f",
            "def f(a: pl.Tensor, b: pl.Tensor, flag: pl.Scalar[pl.INT32]):\n"
            "    if flag > 0:\n        scratch = 0\n        scratch = 2.0\n"
            "    else:\n        scratch = 4\n"
            + calls
            + "    if second.shape[0] > 8:\n        unused = scratch\n    return flag\n",
            param_names=["a", "b", "flag"],
            tensor_meta=meta,
            scalar_dtypes={"flag": DataType.INT32},
            dep_names=["choose"],
        )
        ctx.dep_func_names = {"choose": "choose_v0"}
        ctx.dep_call_variants = {("choose", index): f"choose_v{index}" for index in range(2)}
        self._verify_ssa(pl.parse_program(specialize("VariantDescriptorLiveness", [*helpers, ctx])))

    @pytest.mark.parametrize(
        "binding",
        [
            "x, done = pl.submit(forward, x, True)",
            "x, done = pl.spmd_submit(forward, enabled=True, value=x, core_num=1)",
        ],
    )
    def test_submit_identity_result_retains_tensor_metadata(self, binding):
        """Submit result packing preserves identity metadata and constexpr argument binding."""
        meta = {"value": TensorMeta((4, 8), DataType.FP32)}
        helper = _make_ctx(
            "forward",
            "def forward(value: pl.Tensor, enabled: pl.constexpr):\n    return value\n",
            func_type="incore",
            param_names=["value", "enabled"],
            tensor_meta=meta,
        )
        helper.constexpr_values = {"enabled": "True"}
        ctx = _make_ctx(
            "f",
            "def f(x: pl.Tensor):\n"
            f"    {binding}\n"
            "    for i in pl.unroll(x.shape[0]):\n"
            "        copied = pl.create_tensor([4, 8], dtype=x.dtype)\n"
            "    return x\n",
            param_names=["x"],
            tensor_meta={"x": meta["value"]},
            dep_names=["forward"],
        )
        source = specialize("SubmitIdentity", [helper, ctx])
        assert "pl.unroll(4)" in source
        assert "dtype=pl.FP32" in source
        self._verify_ssa(pl.parse_program(source))

    def test_submit_rewritten_multi_result_preserves_operand_snapshot(self):
        """A rewritten callee retains nested Submit packing and simultaneous metadata."""
        meta = {"x": TensorMeta((4, 8), DataType.FP32), "y": TensorMeta((8, 16), DataType.BF16)}
        helper = _make_ctx(
            "swap",
            "def swap(x: pl.Tensor, y: pl.Tensor):\n    return y, x\n",
            func_type="incore",
            param_names=["x", "y"],
            tensor_meta=meta,
        )
        ctx = _make_ctx(
            "f",
            "def f(x: pl.Tensor, y: pl.Tensor):\n"
            "    (x, y), done = pl.submit(self.swap, x, y)\n"
            "    for i in pl.unroll(x.shape[0]):\n        first = x\n"
            "    for j in pl.unroll(y.shape[0]):\n        second = y\n"
            "    return x, y\n",
            param_names=["x", "y"],
            tensor_meta=meta,
            dep_names=["swap"],
        )
        source = specialize("SubmitSwap", [helper, ctx])
        assert "pl.unroll(8)" in source and "pl.unroll(4)" in source
        program = self._verify_ssa(pl.parse_program(source))
        function = program.get_function("f")
        assert function is not None and len(function.return_types) == 2
        ir.assert_structural_equal(function.return_types[0], ir.TensorType([8, 16], DataType.BF16))
        ir.assert_structural_equal(function.return_types[1], ir.TensorType([4, 8], DataType.FP32))

    @pytest.mark.parametrize("control", ["if flag > 0", "for i in pl.range(flag)"])
    @pytest.mark.parametrize("identity_first", [False, True])
    @pytest.mark.parametrize(
        "identity", ["a = a\n", "a: pl.Tensor[[4, 8], pl.FP32] = a\n", "a, spare = (same, 0)\n"]
    )
    def test_review_descriptor_shadowing_follows_source_order(self, control, identity_first, identity):
        """An identity write cannot shadow a descriptor test before or after it."""
        test = "if a.shape[0] > 8 or rows > 8:\n    rows = 2\n"
        body = identity + test if identity_first else test + identity
        source = (
            "def f(a: pl.Tensor, same: pl.Tensor, flag: pl.Scalar[pl.INT32]):\n"
            "    rows = a.shape[0]\n"
            f"    {control}:\n"
            + textwrap.indent(body, "        ")
            + "    total = 0\n    for j in pl.unroll(rows):\n        total = total + 1\n"
            "    return total\n"
        )
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "same", "flag"],
            tensor_meta={name: TensorMeta((4, 8), DataType.FP32) for name in ["a", "same"]},
            scalar_dtypes={"flag": DataType.INT32},
        )
        specialized = specialize("OrderedDescriptorShadows", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize(
        "body,count,expected",
        [
            ("rows = 2\nif rows > 8:\n    cols = 2", 1, 8),
            ("rows = 9\nif rows > 8:\n    cols = 2", 1, 2),
            ("rows = count\nif rows > 8:\n    cols = 2", 10, 2),
            (
                "if rows == 4:\n    cols = 2\nif i > 0:\n    rows = 8\nelse:\n    rows = 2",
                1,
                2,
            ),
            ("for rows in pl.range(10):\n    if rows > 8:\n        cols = 2", 1, 2),
        ],
        ids=[
            "dead-emitted-arm",
            "crossing-threshold",
            "unknown-write",
            "non-monotone-comparison",
            "scoped-induction",
        ],
    )
    def test_literal_shape_bounds_keep_possible_writes_live(self, body, count, expected):
        """Unknown values, equality, and shadowed induction variables cannot prune live writes."""
        source = (
            "def f(a, count):\n    rows, cols = a.shape\n    for i in pl.range(count):\n"
            + textwrap.indent(body, "        ")
            + "\n    return cols\n"
        )
        transformed = self._transform(source, tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)})
        namespace = {"pl": SimpleNamespace(range=range)}
        exec(transformed, namespace)
        assert namespace["f"](None, count) == expected

    @pytest.mark.parametrize("incoming_alias", [False, True])
    @pytest.mark.parametrize("transparent_scope", [False, True])
    def test_review_tuple_yield_uses_common_branch_names(self, incoming_alias, transparent_scope):
        """Branch-local aliases are operands, not distinct IfStmt output contracts."""
        initialization = "x = 0; x = x + 1; y = 0; y = y + 1" if incoming_alias else "pass"
        then_body = "x = 2\ny = 3\nx = x + 1\ny = y + 1\nx, y = pl.yield_(x, y)"
        else_body = "x = 4\ny = 5\nx = x + 1\ny = y + 1\nx, y = pl.yield_(x, y)"
        if transparent_scope:
            then_body = "with pl.scope():\n" + textwrap.indent(then_body, "    ")
            else_body = "with pl.scope():\n" + textwrap.indent(else_body, "    ")
        source = (
            "def f(flag: pl.Scalar[pl.INT32]):\n"
            + textwrap.indent(initialization, "    ")
            + "\n    if flag > 0:\n"
            + textwrap.indent(then_body, "        ")
            + "\n    else:\n"
            + textwrap.indent(else_body, "        ")
            + "\n    return x, y\n"
        )
        ctx = _make_ctx(
            "f",
            source,
            param_names=["flag"],
            scalar_dtypes={"flag": DataType.INT32},
            auto_scope=not transparent_scope,
        )
        program = pl.parse_program(specialize("TupleYield", [ctx]))
        function = program.get_function("f")
        assert function is not None and isinstance(function.body, ir.SeqStmts)
        branch = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.IfStmt))
        assert len(branch.return_vars) == 2
        # SSA conversion verifies the return-variable/yield arity and references.
        passes.convert_to_ssa()(program)

    def test_graph_tuple_task_id_shadow_preserves_folded_shape(self):
        """The Graph must restore the folded predecessor after a tuple TaskId write."""
        source = textwrap.dedent("""
            def f(c: pl.InOut[pl.Tensor]):
                width = c.shape[1]
                with pl.manual_scope():
                    with pl.graph("recording"):
                        local_ready = pl.system.task_dummy(deps=[])
                        width, unused = (local_ready, 0)
                        pl.system.task_dummy(deps=[width])
                return pl.create_tensor([128, width], dtype=pl.FP32)
        """)
        ctx = _make_ctx(
            "f", source, param_names=["c"], tensor_meta={"c": TensorMeta((128, 64), DataType.FP32)}
        )
        function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
        assert function is not None and isinstance(function.body, ir.SeqStmts)
        returned = function.body.stmts[-1]
        assert isinstance(returned, ir.ReturnStmt)
        (value,) = returned.value
        result = value.type
        assert isinstance(result, ir.TensorType)
        assert isinstance(result.shape[1], ir.ConstInt) and result.shape[1].value == 64

    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize("unpack", [False, True])
    def test_followup_constexpr_shape_write_keeps_unroll_bound(self, enabled, unpack):
        """A selected shape write must remain a literal accepted by pl.unroll."""
        initialization = "rows, cols = a.shape" if unpack else "rows = a.shape[0]"
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, enabled: pl.constexpr):
                {initialization}
                if enabled:
                    rows, cols = a.shape
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return total
        """)
        ctx = _make_ctx(
            "f", source, param_names=["a", "enabled"], tensor_meta={"a": TensorMeta((4, 8), DataType.FP32)}
        )
        ctx.constexpr_values = {"enabled": str(enabled)}
        program = pl.parse_program(specialize("ConstexprShape", [ctx]))
        self._verify_ssa(program)

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize(
        "then_body,else_body,expected_true,expected_false",
        [
            ("x = 1; x, y = (2, 3)", "x, y = (4, 5)", (2, 3), (4, 5)),
            (
                "x, y = (1, 2); x, y = (x + 10, y + 20)",
                "x, y = (3, 4); x, y = (x + 30, y + 40)",
                (11, 22),
                (33, 44),
            ),
            ("x, y = (1, 2); x, y = (3.0, 4)", "x, y = (5, 6); x, y = (7.0, 8)", (3.0, 4), (7.0, 8)),
        ],
        ids=["one-rebind", "both-rebind", "type-changing"],
    )
    def test_followup_tuple_branch_exports(self, flag, then_body, else_body, expected_true, expected_false):
        """The chosen arm must export its last tuple result, not its first assignment."""
        source = (
            f"def f(flag):\n    if flag > 0:\n        {then_body}\n"
            f"    else:\n        {else_body}\n    return x, y\n"
        )
        namespace = {}
        exec(self._transform(source), namespace)
        assert namespace["f"](flag) == (expected_true if flag else expected_false)
        source = source.replace("def f(flag):", "def f(flag: pl.Scalar[pl.INT32]):")
        ctx = _make_ctx("f", source, param_names=["flag"], scalar_dtypes={"flag": DataType.INT32})
        self._verify_ssa(pl.parse_program(specialize("BranchExports", [ctx])))

    @pytest.mark.parametrize("flag", [False, True])
    def test_followup_dead_branch_temporaries_do_not_merge_types(self, flag):
        """A later overwrite kills scratch; only x and y need a branch export."""
        source = textwrap.dedent("""
            def f(flag):
                if flag > 0:
                    scratch = 0
                    scratch, tmp = (2.0, 0)
                    x = 1
                    x, y = (2, 3)
                else:
                    scratch = 0
                    scratch, tmp = (4, 0)
                    x, y = (4, 5)
                scratch = 0
                x = x + 10
                return x, y, scratch
        """)
        namespace = {}
        exec(self._transform(source), namespace)
        assert namespace["f"](flag) == ((12, 3, 0) if flag else (14, 5, 0))
        source = source.replace("def f(flag):", "def f(flag: pl.Scalar[pl.INT32]):")
        ctx = _make_ctx("f", source, param_names=["flag"], scalar_dtypes={"flag": DataType.INT32})
        self._verify_ssa(pl.parse_program(specialize("DeadTemporaries", [ctx])))

    def test_followup_split_aiv_keeps_enclosing_yield_names(self):
        """split_aiv owns a transparent scope rather than a new yield contract."""
        source = textwrap.dedent("""
            def f(flag: pl.Scalar[pl.INT32]):
                if flag > 0:
                    x = 1
                    y = 2
                    x = x + 1
                    y = y + 1
                    for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        x, y = pl.yield_(x, y)
                else:
                    x = 3
                    y = 4
                    x = x + 1
                    y = y + 1
                    for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        x, y = pl.yield_(x, y)
                return x, y
        """)
        ctx = _make_ctx(
            "f", source, func_type="incore", param_names=["flag"], scalar_dtypes={"flag": DataType.INT32}
        )
        program = pl.parse_program(specialize("SplitYield", [ctx]))
        function = program.get_function("f")
        assert function is not None and isinstance(function.body, ir.SeqStmts)
        branch = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.IfStmt))
        assert len(branch.return_vars) == 2
        self._verify_ssa(program)

    @pytest.mark.parametrize("control", ["if flag > 0", "for iteration in pl.range(flag)"])
    def test_followup_graph_conditional_shape_has_predecessor(self, control):
        """A Graph consumer needs an incoming scalar on the untaken/zero-trip path."""
        source = textwrap.dedent(f"""
            def f(a: pl.InOut[pl.Tensor], flag: pl.Scalar[pl.INT32]):
                rows = a.shape[0]
                with pl.manual_scope():
                    with pl.graph("recording"):
                        {control}:
                            rows, tmp = (rows, 0)
                        for i in pl.range(rows):
                            with pl.at(level=pl.Level.CORE_GROUP):
                                pl.store(pl.load(a, [i, 0], [1, 4]), [i, 0], a)
                return a
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 4), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INT32},
        )
        program = pl.parse_program(specialize("GraphShape", [ctx]))
        passes.outline_graph_scopes()(self._verify_ssa(program))

    @pytest.mark.parametrize("control", ["if flag > 0", "for iteration in pl.range(flag)"])
    def test_followup_graph_task_id_shadow_keeps_shape_predecessor(self, control):
        """A private TaskId shadow must not mutate an integer runtime predecessor."""
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INT32]):
                rows = a.shape[0]
                with pl.manual_scope():
                    with pl.graph("recording"):
                        {control}:
                            ready = pl.system.task_dummy(deps=[])
                            rows, tmp = (ready, 0)
                            pl.system.task_dummy(deps=[rows])
                return pl.create_tensor([rows, 4], dtype=pl.FP32)
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 4), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INT32},
        )
        program = pl.parse_program(specialize("TaskShape", [ctx]))
        function = program.get_function("f")
        assert function is not None and isinstance(function.body, ir.SeqStmts)
        returned = function.body.stmts[-1]
        assert isinstance(returned, ir.ReturnStmt)
        shape = returned.value[0].type
        assert isinstance(shape, ir.TensorType)
        assert isinstance(shape.shape[0], ir.ConstInt) and shape.shape[0].value == 4
        self._verify_ssa(program)

    @pytest.mark.parametrize("control", ["if flag > 0", "for iteration in pl.range(flag)", "while flag > 0"])
    def test_followup_runtime_identity_shape_write_keeps_unroll_bound(self, control):
        """Writing the identical static shape does not require a runtime scalar."""
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INT32]):
                rows, cols = a.shape
                {control}:
                    rows, cols = a.shape
                    flag = pl.cast(0, pl.INT32)
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return total
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 4), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INT32},
        )
        self._verify_ssa(pl.parse_program(specialize("IdentityShape", [ctx])))

    @pytest.mark.parametrize("flag", [False, True])
    def test_followup_rebound_shape_source_keeps_incoming_dimensions(self, flag):
        """An entry-shape snapshot cannot prove a write after the source is rebound."""
        source = textwrap.dedent("""
            def f(a, flag):
                rows, cols = a.shape
                if flag > 0:
                    a = pl.reshape(a, [4, 4])
                    rows, cols = a.shape
                return rows, cols
        """)
        meta = {"a": TensorMeta((4, 4), DataType.FP32)}
        namespace = {"pl": SimpleNamespace(reshape=lambda value, shape: value)}
        exec(self._transform(source, tensor_meta=meta), namespace)
        assert namespace["f"](SimpleNamespace(shape=(4, 4)), flag) == (4, 4)
        source = source.replace("def f(a, flag):", "def f(a: pl.Tensor, flag: pl.Scalar[pl.INT32]):")
        ctx = _make_ctx(
            "f", source, param_names=["a", "flag"], tensor_meta=meta, scalar_dtypes={"flag": DataType.INT32}
        )
        self._verify_ssa(pl.parse_program(specialize("ReboundShape", [ctx])))

    @pytest.mark.parametrize("first", [False, True])
    @pytest.mark.parametrize("second", [False, True])
    def test_next_review_sequential_if_promotes_common_binding(self, first, second):
        """A second conditional updates the value joined by the first one."""
        source = textwrap.dedent("""
            def f(first, second):
                if first > 0:
                    x, y = (2, 3)
                else:
                    x, y = (4, 5)
                if second > 0:
                    x, y = (6, 7)
                return x, y
        """)
        namespace = {}
        exec(self._transform(source), namespace)
        assert namespace["f"](first, second) == ((6, 7) if second else ((2, 3) if first else (4, 5)))
        source = source.replace(
            "def f(first, second):", "def f(first: pl.Scalar[pl.INT32], second: pl.Scalar[pl.INT32]):"
        )
        ctx = _make_ctx(
            "f",
            source,
            param_names=["first", "second"],
            scalar_dtypes={"first": DataType.INT32, "second": DataType.INT32},
        )
        self._verify_ssa(pl.parse_program(specialize("SequentialBranches", [ctx])))

    @pytest.mark.parametrize("unpack", [False, True])
    @pytest.mark.parametrize("annotated", [False, True])
    @pytest.mark.parametrize("control", ["if flag > 0", "for iteration in pl.range(flag)", "while flag > 0"])
    def test_next_review_dead_shape_write_keeps_restored_unroll_bound(self, unpack, annotated, control):
        """An unconditional shape restore kills the earlier conditional write."""
        shape = "rows, cols = a.shape" if unpack else "rows = a.shape[0]"
        write = "rows: pl.Scalar[pl.INDEX] = 2" if annotated else "rows = 2"
        source = textwrap.dedent(f"""
            def f(a, flag):
                {shape}
                total = 0
                {control}:
                    {write}
                    total = rows
                    flag = flag - flag
                {shape}
                for i in pl.unroll(rows):
                    total = total + 1
                return total
        """)
        meta = {"a": TensorMeta((4, 4), DataType.FP32)}
        namespace = {"pl": SimpleNamespace(range=range, unroll=range)}
        transformed = self._transform(source, tensor_meta=meta)
        exec(transformed, namespace)
        for flag in [0, 1]:
            assert namespace["f"](SimpleNamespace(shape=(4, 4)), flag) == (6 if flag else 4)
        source = source.replace("def f(a, flag):", "def f(a: pl.Tensor, flag: pl.Scalar[pl.INT32]):")
        ctx = _make_ctx(
            "f", source, param_names=["a", "flag"], tensor_meta=meta, scalar_dtypes={"flag": DataType.INT32}
        )
        specialized = specialize("RestoredShape", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("terminator", ["break", "continue"])
    @pytest.mark.parametrize("nested", [False, True])
    def test_next_review_branch_exports_before_loop_exit(self, flag, terminator, nested):
        """A loop exit exports its current tuple, including before a later write."""
        exit_body = (
            f"if flag > 0:\n                {terminator}\n            x, y = (6, 7)" if nested else terminator
        )
        source = (
            "def f(flag):\n    for iteration in pl.range(1):\n        if flag > 0:\n"
            "            x = 1\n            x, y = (2, 3)\n"
            f"            {exit_body}\n        else:\n            x, y = (4, 5)\n"
            "        x, y = (8, 9)\n    return x, y\n"
        )
        namespace = {"pl": SimpleNamespace(range=range)}
        exec(self._transform(source), namespace)
        assert namespace["f"](flag) == ((2, 3) if flag else (8, 9))
        source = source.replace("def f(flag):", "def f(flag: pl.Scalar[pl.INT32]):")
        ctx = _make_ctx("f", source, param_names=["flag"], scalar_dtypes={"flag": DataType.INT32})
        # Native execution checks the exit value. Raw-program SSA separately
        # rejects loop locals escaping without an incoming initialization.
        assert isinstance(pl.parse_program(specialize("LoopExitExports", [ctx])), ir.Program)

    def test_loop_exit_liveness_ignores_dead_cross_type_temporary(self):
        """Exit exports retain live values without merging a dead cross-type local."""
        source = textwrap.dedent("""
            def f(flag):
                x, y = (0, 0)
                for iteration in pl.range(1):
                    if flag > 0:
                        scratch = 0
                        scratch = 2.0
                        x, y = (2, 3)
                        break
                    else:
                        scratch = 4
                        x, y = (4, 5)
                    x, y = (6, 7)
                return x, y
        """)
        namespace = {"pl": SimpleNamespace(range=range)}
        exec(self._transform(source), namespace)
        assert namespace["f"](0) == (6, 7)
        assert namespace["f"](1) == (2, 3)
        source = source.replace("def f(flag):", "def f(flag: pl.Scalar[pl.INT32]):")
        ctx = _make_ctx("f", source, param_names=["flag"], scalar_dtypes={"flag": DataType.INT32})
        self._verify_ssa(pl.parse_program(specialize("LoopExitDeadTemporary", [ctx])))

    @pytest.mark.parametrize("control", ["if flag > 0", "for iteration in pl.range(flag)", "while flag > 0"])
    def test_next_review_graph_identity_tuple_keeps_caller_shape(self, control):
        """An identity tuple keeps a caller-only shape consumer compile-time constant."""
        source = textwrap.dedent(f"""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INT32]):
                rows = a.shape[0]
                with pl.manual_scope():
                    with pl.graph("recording"):
                        pl.system.task_dummy(deps=[])
                        {control}:
                            rows, unused = (rows, 0)
                            flag = flag - flag
                return pl.create_tensor([rows, 4], dtype=pl.FP32)
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 4), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INT32},
        )
        program = pl.parse_program(specialize("CallerShape", [ctx]))
        function = program.get_function("f")
        assert function is not None and isinstance(function.body, ir.SeqStmts)
        returned = function.body.stmts[-1]
        assert isinstance(returned, ir.ReturnStmt)
        result = returned.value[0].type
        assert isinstance(result, ir.TensorType)
        assert isinstance(result.shape[0], ir.ConstInt) and result.shape[0].value == 4
        passes.outline_graph_scopes()(self._verify_ssa(program))

    @pytest.mark.parametrize("scope", ["manual_scope", "scope"])
    @pytest.mark.parametrize("flag", [False, True])
    def test_comment_sibling_scope_exports_tuple_binding(self, scope, flag):
        """A later conditional retains a tuple bound in an earlier with scope."""
        source = textwrap.dedent(f"""
            def f(flag):
                with pl.{scope}():
                    x, y = (1, 2)
                if flag > 0:
                    x, y = (3, 4)
                return x, y
        """)
        namespace = {"pl": SimpleNamespace(manual_scope=nullcontext, scope=nullcontext)}
        exec(self._transform(source), namespace)
        assert namespace["f"](flag) == ((3, 4) if flag else (1, 2))
        source = source.replace("def f(flag):", "def f(flag: pl.Scalar[pl.INT32]):")
        ctx = _make_ctx(
            "f", source, param_names=["flag"], scalar_dtypes={"flag": DataType.INT32}, auto_scope=False
        )
        self._verify_ssa(pl.parse_program(specialize("SiblingScopes", [ctx])))

    @pytest.mark.parametrize("increment", [0, 1])
    @pytest.mark.parametrize("flag", [False, True])
    def test_comment_shape_identity_observes_prior_operand_write(self, increment, flag):
        """An operand's old shape fold cannot prove an identity after a scalar write."""
        source = textwrap.dedent(f"""
            def f(a, flag):
                rows, cols = a.shape
                if flag > 0:
                    cols = cols + {increment}
                    rows, unused = (cols, 0)
                return rows
        """)
        meta = {"a": TensorMeta((4, 4), DataType.FP32)}
        namespace = {}
        exec(self._transform(source, tensor_meta=meta), namespace)
        assert namespace["f"](SimpleNamespace(shape=(4, 4)), flag) == 4 + (increment if flag else 0)
        source = source.replace("def f(a, flag):", "def f(a: pl.Tensor, flag: pl.Scalar[pl.INT32]):")
        ctx = _make_ctx(
            "f", source, param_names=["a", "flag"], tensor_meta=meta, scalar_dtypes={"flag": DataType.INT32}
        )
        self._verify_ssa(pl.parse_program(specialize("WrittenOperand", [ctx])))

    def test_comment_shape_identity_ignores_later_operand_write(self):
        """A later scalar write cannot invalidate an earlier, genuinely identical tuple."""
        source = textwrap.dedent("""
            def f(a: pl.Tensor, flag: pl.Scalar[pl.INT32]):
                rows, cols = a.shape
                if flag > 0:
                    rows, unused = (cols, 0)
                    cols = cols + 1
                total = 0
                for i in pl.unroll(rows):
                    total = total + 1
                return total
        """)
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag"],
            tensor_meta={"a": TensorMeta((4, 4), DataType.FP32)},
            scalar_dtypes={"flag": DataType.INT32},
        )
        specialized = specialize("LaterWrite", [ctx])
        assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    @pytest.mark.parametrize("count", [0, 2])
    def test_comment_shape_identity_observes_prior_loop_iterations(self, count):
        """A write later in the body changes the operand seen by the next iteration."""
        source = textwrap.dedent("""
            def f(a, count):
                rows, cols = a.shape
                for iteration in pl.range(count):
                    rows, unused = (cols, 0)
                    cols = cols + 1
                return rows
        """)
        meta = {"a": TensorMeta((4, 4), DataType.FP32)}
        namespace = {"pl": SimpleNamespace(range=range)}
        exec(self._transform(source, tensor_meta=meta), namespace)
        assert namespace["f"](SimpleNamespace(shape=(4, 4)), count) == (4 if count == 0 else 5)
        source = source.replace("def f(a, count):", "def f(a: pl.Tensor, count: pl.Scalar[pl.INDEX]):")
        ctx = _make_ctx(
            "f", source, param_names=["a", "count"], tensor_meta=meta, scalar_dtypes={"count": DataType.INDEX}
        )
        self._verify_ssa(pl.parse_program(specialize("CarriedOperand", [ctx])))

    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_comment_for_target_rebinds_tuple_alias(self, count):
        """Body reads follow the scoped target; the caller alias is restored."""
        source = textwrap.dedent("""
            def f(count):
                i, y = (1, 2)
                i, y = (3, 4)
                total = 0
                for i in pl.range(count):
                    total = total + i
                return i, total
        """)
        namespace = {"pl": SimpleNamespace(range=range)}
        exec(self._transform(source), namespace)
        assert namespace["f"](count) == (3, sum(range(count)))
        source = source.replace("def f(count):", "def f(count: pl.Scalar[pl.INDEX]):")
        ctx = _make_ctx("f", source, param_names=["count"], scalar_dtypes={"count": DataType.INDEX})
        self._verify_ssa(pl.parse_program(specialize("TupleCounter", [ctx])))

    def test_comment_nested_loop_target_restores_parent_identity(self):
        """A scoped induction assignment preserves the caller's operand fold."""
        source = textwrap.dedent("""
            def f(a, flag, count):
                rows, cols = a.shape
                if flag > 0:
                    for cols in pl.range(count):
                        pass
                    rows, unused = (cols, 0)
                return rows
        """)
        meta = {"a": TensorMeta((4, 4), DataType.FP32)}
        namespace = {"pl": SimpleNamespace(range=range)}
        exec(self._transform(source, tensor_meta=meta), namespace)
        for flag in [0, 1]:
            for count in [0, 1]:
                assert namespace["f"](SimpleNamespace(shape=(4, 4)), flag, count) == 4
        source = source.replace(
            "def f(a, flag, count):",
            "def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX], count: pl.Scalar[pl.INDEX]):",
        )
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag", "count"],
            tensor_meta=meta,
            scalar_dtypes={"flag": DataType.INDEX, "count": DataType.INDEX},
        )
        self._verify_ssa(pl.parse_program(specialize("NestedTarget", [ctx])))

    def test_comment_for_target_body_reassignment_does_not_change_iterator(self):
        """Body reassignment leaves iteration and the caller binding intact."""
        source = textwrap.dedent("""
            def f():
                i, y = (1, 2)
                i, y = (3, 4)
                total = 0
                for i in pl.range(2):
                    i = 10
                    total = total + i
                return i, total
        """)
        namespace = {"pl": SimpleNamespace(range=range)}
        exec(self._transform(source), namespace)
        assert namespace["f"]() == (3, 20)
        self._verify_ssa(pl.parse_program(specialize("AssignedCounter", [_make_ctx("f", source)])))

    @pytest.mark.parametrize("restored", [False, True])
    @pytest.mark.parametrize("control", ["for iteration in pl.range(count)", "while count > 0"])
    def test_comment_nested_shape_write_uses_child_liveness(self, control, restored):
        """An inner write killed by a body-local restore is dead at loop exit."""
        restore = "rows = a.shape[0]" if restored else "pass"
        source = textwrap.dedent(f"""
            def f(a, flag, count):
                rows = a.shape[0]
                {control}:
                    if flag > 0:
                        rows = 2
                    {restore}
                    count = count - 1
                return rows
        """)
        meta = {"a": TensorMeta((4, 4), DataType.FP32)}
        namespace = {"pl": SimpleNamespace(range=range)}
        exec(self._transform(source, tensor_meta=meta), namespace)
        for count in [0, 2]:
            for flag in [0, 1]:
                expected = 2 if count and flag and not restored else 4
                assert namespace["f"](SimpleNamespace(shape=(4, 4)), flag, count) == expected
        if restored:
            source = source.replace(
                "    return rows\n",
                "    total = 0\n    for i in pl.unroll(rows):\n        total = total + 1\n    return total\n",
            )
        source = source.replace(
            "def f(a, flag, count):",
            "def f(a: pl.Tensor, flag: pl.Scalar[pl.INDEX], count: pl.Scalar[pl.INDEX]):",
        )
        ctx = _make_ctx(
            "f",
            source,
            param_names=["a", "flag", "count"],
            tensor_meta=meta,
            scalar_dtypes={"flag": DataType.INDEX, "count": DataType.INDEX},
        )
        specialized = specialize("NestedRestore", [ctx])
        if restored:
            assert "pl.unroll(4)" in specialized
        self._verify_ssa(pl.parse_program(specialized))

    def test_comment_metadata_preview_runs_liveness_before_dependency(self):
        """A restored unary-plus shape must retain metadata for a JIT dependency."""

        @pl.jit.inline
        def keep(a: pl.Tensor):
            return a

        @pl.jit
        def entry(src: pl.Tensor[[1, 128], pl.FP32], flag: pl.Scalar[pl.INDEX]):
            if TYPE_CHECKING:
                assert src.shape is not None
            rows = src.shape[1]
            if flag > 0:
                rows = 2
            rows = src.shape[1]
            reshaped = pl.reshape(src, [+rows])
            return keep(reshaped)

        program = entry.specialize()
        helper = next(function for function in program.functions.values() if function.name.startswith("keep"))
        assert isinstance(helper.params[0].type, ir.TensorType)
        assert isinstance(helper.params[0].type.shape[0], ir.ConstInt)
        assert helper.params[0].type.shape[0].value == 128
        self._verify_ssa(program)

    @pytest.mark.parametrize(
        "use_ptoas",
        [False, pytest.param(True, marks=pytest.mark.ptoas_compile)],
        ids=["diagnostic", "executable"],
    )
    def test_register_tuple_rebind_reaches_store(self, tmp_path, monkeypatch, use_ptoas):
        """The store must use the intlv projection, rather than the earlier add."""
        if use_ptoas:
            if find_ptoas_binary() is None:
                pytest.skip("PTOAS is unavailable")
        else:
            monkeypatch.setattr(decorator, "_ptoas_available", lambda: False)

        @pl.jit.aiv
        def register_rebind(source: pl.Tensor[[1, 64], pl.FP32], output: pl.Out[pl.Tensor[[1, 64], pl.FP32]]):
            src = pl.load(source, [0, 0], [1, 64], target_memory=pl.MemorySpace.Vec)
            dst = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
            src = pl.reshape(src, [64])
            dst = pl.reshape(dst, [64])
            with pl.reg_scope():
                mask = pl.reg.full_mask(lanes=64)
                value = pl.reg.load(src, lanes=64)
                value = pl.reg.add(value, value)
                value, unused = pl.reg.intlv(value, value, mask)
                dst = pl.reg.store(dst, value=value)
            dst = pl.reshape(dst, [1, 64])
            pl.store(dst, [0, 0], output)
            return output

        @pl.jit
        def entry(source: pl.Tensor[[1, 64], pl.FP32], output: pl.Out[pl.Tensor[[1, 64], pl.FP32]]):
            return register_rebind(source, output)

        assignments = []

        class Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                assignments.append(op)
                super().visit_assign_stmt(op)

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend950)
        try:
            Collector().visit_program(entry.specialize())
            projection = next(
                op for op in assignments if isinstance(op.value, ir.TupleGetItemExpr) and op.value.index == 0
            )
            store = next(
                op.value
                for op in assignments
                if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("reg.store").name
            )
            assert store.args[2] is projection.var
            entry.compile(
                config=RunConfig(
                    backend_type=BackendType.Ascend950,
                    execution_mode=ExecutionMode.ONBOARD,
                    save_kernels_dir=str(tmp_path),
                )
            )
            assert list(tmp_path.rglob("*.pto"))
        finally:
            backend.reset_for_testing()

    def test_rebind_rhs_references_prior_alias(self):
        src = """
            def f(a):
                x = a
                x = pl.load(x)
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((64,), DataType.FP32)})
        # The RHS of x_v1 = ... should reference the original x, not x_v1
        assert "x_v1 = pl.load(x)" in out

    def test_rebind_multiple_times(self):
        src = """
            def f():
                x = a
                x = b
                x = c
        """
        out = self._transform(src)
        assert "x =" in out
        assert "x_v1 =" in out
        assert "x_v2 =" in out

    def test_later_reads_see_latest_alias(self):
        src = """
            def f(a):
                x = a
                x = pl.mul(x, 2.0)
                y = x
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((64,), DataType.FP32)})
        # y should be assigned the latest alias x_v1
        assert "y = x_v1" in out

    def test_ann_assign_rebind(self):
        """Annotated assignment rebinding is alpha-renamed."""
        src = """
            def f():
                x = a
                x: int = b
        """
        out = self._transform(src)
        assert "x =" in out
        assert "x_v1:" in out

    def test_rebind_supersedes_shape_inline(self):
        """After a shape-inlined variable is rebound, reads see the new alias."""
        src = """
            def f(a):
                M, N = a.shape
                M = some_call(M)
                y = M
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((64, 32), DataType.FP32)})
        # M is inlined (static), then M = some_call(...) becomes M_v1 = some_call(64)
        # y = M should resolve to y = M_v1
        assert "y = M_v1" in out

    def test_param_rebind_generates_alias(self):
        """Assigning to a function parameter generates a renamed alias."""
        src = """
            def f(x):
                x = some_call(x)
                y = x
        """
        out = self._transform(src)
        # x is a param, so re-assignment generates x_v1; reads of x after rebinding → x_v1
        assert "x_v1 = some_call(x)" in out
        assert "y = x_v1" in out

    def test_rebound_tensor_shape_reads_current_value(self):
        """Dimension arithmetic after a parameter rebind uses its new tensor."""
        src = """
            def f(src, other):
                before = src.shape[1]
                saved = src
                with scope():
                    src = identity(src)
                    src = saved
                    restored = src.shape
                with scope():
                    src = other
                    nested = src.shape[1]
                src = other
                width = src.shape[1]
                half = width // 2
                rows, cols, tail = src.shape
                whole = src.shape
                dtype = src.dtype
                src: object = src
                annotated = src.shape
                total = src.shape[0] * width
                return before, nested, width, half, total, rows, cols, tail, whole, dtype, annotated, restored
        """
        transformed = self._transform(
            src,
            tensor_meta={
                "src": TensorMeta((1, 128), DataType.FP32),
                "other": TensorMeta((4, 64, 2), DataType.FP16),
            },
        )
        namespace = {
            "pl": SimpleNamespace(
                tensor=SimpleNamespace(dim=lambda tensor, axis: tensor[axis]), FP16=DataType.FP16
            ),
            "scope": nullcontext,
            "identity": lambda tensor: tensor,
        }
        exec(transformed, namespace)
        assert namespace["f"]((1, 128), (4, 64, 2)) == (
            128,
            64,
            64,
            32,
            256,
            4,
            64,
            2,
            (4, 64, 2),
            DataType.FP16,
            (4, 64, 2),
            (1, 128),
        )

    def test_alias_skips_collision(self):
        """Generated alias skips names already defined by the user."""
        src = """
            def f():
                x = a
                x_v1 = b
                x = c
        """
        out = self._transform(src)
        # x_v1 is taken by user, so rebinding of x should skip to x_v2
        assert "x_v2 =" in out
        assert "x_v1 = b" in out

    @pytest.mark.parametrize(
        "context",
        [
            'pl.graph("layer", deps=[ready])',
            "pl.at(level=pl.Level.CORE_GROUP, deps=[ready])",
            "pl.spmd(1, deps=[ready])",
        ],
    )
    def test_scope_capture_uses_the_existing_emitted_binding(self, context):
        out = self._transform(f"""
            def f(seed, latest):
                ready = seed
                ready = latest
                with {context} as ready:
                    work()
                consume(deps=[ready])
        """)
        function = ast.parse(out).body[0]
        assert isinstance(function, ast.FunctionDef)
        assigned, scope, consumer = function.body[1:]
        assert isinstance(assigned, ast.Assign) and isinstance(assigned.targets[0], ast.Name)
        assert isinstance(scope, ast.With) and isinstance(scope.items[0].optional_vars, ast.Name)
        expected = assigned.targets[0].id
        assert scope.items[0].optional_vars.id == expected, out
        call = scope.items[0].context_expr
        assert isinstance(call, ast.Call)
        deps = next(kw.value for kw in call.keywords if kw.arg == "deps")
        assert isinstance(deps, ast.List) and isinstance(deps.elts[0], ast.Name)
        assert deps.elts[0].id == expected, out
        assert isinstance(consumer, ast.Expr) and isinstance(consumer.value, ast.Call)
        deps = next(kw.value for kw in consumer.value.keywords if kw.arg == "deps")
        assert isinstance(deps, ast.List) and isinstance(deps.elts[0], ast.Name)
        assert deps.elts[0].id == expected, out

    def test_if_branch_rebind_not_renamed(self):
        """Assignments inside if branches are not alpha-renamed.

        The Parser handles if-branch variables via leak_vars=True (variables
        are visible after the if), so renaming here would produce an alias
        that only conditionally exists, breaking code after the if.
        """
        src = """
            def f():
                t = a
                if cond:
                    t = b
                y = t
        """
        out = self._transform(src)
        # t inside the if must NOT be renamed — it should remain 't = b'
        assert "t_v1" not in out
        assert "y = t" in out

    def test_if_else_branch_rebind_not_renamed(self):
        """Assignments in both if/else branches are not alpha-renamed."""
        src = """
            def f():
                t = a
                if cond:
                    t = b
                else:
                    t = c
                y = t
        """
        out = self._transform(src)
        assert "t_v1" not in out
        assert "y = t" in out

    def test_parallel_for_rebind_with_bridge(self):
        """Two parallel for loops assigning the same variable get a bridge assignment."""
        src = """
            def f():
                for i in range(n):
                    x = a
                for j in range(m):
                    x = some_op(x)
                y = x
        """
        out = self._transform(src)
        # Bridge assignment inserted before the second loop
        assert "x_v1 = x" in out
        # Both LHS and RHS inside the second loop use x_v1
        assert "x_v1 = some_op(x_v1)" in out
        # Final read uses the latest alias
        assert "y = x_v1" in out

    def test_for_induction_shadows_scalar_alias_and_restores_it(self):
        out = self._transform("""
            def f(seed):
                i = seed
                i = i + 1
                for i in range(i):
                    before_inner(i)
                    for i in range(i):
                        inner(i)
                    after_inner(i)
                after_outer(i)
        """)
        assert "for i in range(i_v1):" in out
        assert "for i in range(i):" in out
        assert "before_inner(i)" in out
        assert "inner(i)" in out
        assert "after_inner(i)" in out
        assert "after_outer(i_v1)" in out

    def test_for_induction_shadows_folded_shape_and_restores_it(self):
        out = self._transform(
            """
            def f(src):
                i = src.shape[0]
                for i in range(i):
                    consume(i)
                after(i)
            """,
            tensor_meta={"src": TensorMeta(shape=(8, 32), dtype=DataType.FP32)},
        )
        assert "for i in range(8):" in out
        assert "consume(i)" in out
        assert "after(8)" in out

    @pytest.mark.parametrize("setup", ["", "x = 99; x = x + 1", "x = src.shape[0]"])
    def test_explicit_while_publishes_yield_binding(self, setup):
        source = f"""
            def f(src):
                {setup}
                for (x,) in pl.while_(init_values=(0,)):
                    pl.cond(x < 3)
                    x = pl.yield_(x + 1)
                consume(x)
        """
        out = self._transform(source, tensor_meta={"src": TensorMeta((8, 32), DataType.FP32)})
        assert "pl.cond(x < 3)" in out
        assert "x = pl.yield_(x + 1)" in out
        assert "consume(x)" in out

    def test_explicit_while_distinct_yield_restores_folded_outer_binding(self):
        out = self._transform(
            """
            def f(src):
                x = src.shape[0]
                for (x,) in pl.while_(init_values=(0,)):
                    pl.cond(x < 3)
                    result = pl.yield_(x + 1)
                consume(result, x)
            """,
            tensor_meta={"src": TensorMeta((8, 32), DataType.FP32)},
        )
        assert "pl.cond(x < 3)" in out
        assert "result = pl.yield_(x + 1)" in out
        assert "consume(result, 8)" in out

    def test_conditional_yields_do_not_publish_enclosing_while_carry(self):
        out = self._transform("""
            def f():
                x = 99
                x = x + 1
                for (x,) in pl.while_(init_values=(0,)):
                    pl.cond(x < 3)
                    if x < 1:
                        x = pl.yield_(x + 1)
                    else:
                        x = pl.yield_(x + 2)
                    result = pl.yield_(x)
                consume(x, result)
        """)
        # Branch outputs update the carry inside the loop; only result escapes.
        # Restoring the original spelling x here would read 99 instead of 100.
        assert "consume(x_v1, result)" in out

    @pytest.mark.parametrize(
        ("wrapper", "transparent"),
        [
            ("if condition:", False),
            ("for i in pl.range(2):", False),
            ("while condition:", False),
            ("with pl.at(pl.Level.CORE_GROUP):", False),
            ("with pl.scope():", True),
            ("with pl.manual_scope():", True),
            ("for i in pl.split_aiv(2):", True),
        ],
    )
    def test_loop_yield_ownership_boundaries(self, wrapper, transparent):
        source = f"{wrapper}\n    nested = pl.yield_(1)\nresult = pl.yield_(2)"
        names = BodyTransformer._loop_yield_targets(ast.parse(source).body)
        assert names == ({"nested", "result"} if transparent else {"result"})

    def test_nested_loop_yield_does_not_publish_outer_induction(self):
        out = self._transform("""
            def f(seed):
                x = seed
                x = x + 1
                for x in pl.range(2):
                    for (x,) in pl.while_(init_values=(0,)):
                        pl.cond(x < 3)
                        x = pl.yield_(x + 1)
                consume(x)
        """)
        assert "consume(x_v1)" in out

    def test_parallel_for_simple_rebind(self):
        """Two sibling for loops, each a *fresh* (non-loop-carried) local of the
        same name, are left untouched.

        ``x`` is written before it is read in each loop body, so it is a fresh
        per-loop local — not a carried value.  The renamer must NOT bridge it:
        a bridge ``x_v1 = x`` would read loop ``i``'s local outside loop ``i``.
        Both loops keep ``x`` and ``y`` reads it directly; the parser scopes
        the sibling bodies, exactly as for a hand-written ``@pl.program``.
        """
        src = """
            def f():
                for i in range(n):
                    x = a
                for j in range(m):
                    x = b
                y = x
        """
        out = self._transform(src)
        assert "x_v1" not in out  # fresh local — neither bridged nor renamed
        assert "y = x" in out  # reads the leaked `x` directly, no bridge

    def test_parallel_for_carried_rebind(self):
        """A genuine carry across sibling loops still gets a bridge.

        Loop ``j`` reads ``acc`` (the value loop ``i`` left) before writing it
        — even though the write ``acc = tmp`` is staged through a temporary and
        is not self-referential.  ``acc`` is upward-exposed, so the renamer must
        bridge it (``acc_v1 = acc``) to thread the carried value across the
        loop boundary.
        """
        src = """
            def f():
                for i in range(n):
                    acc = seed
                for j in range(m):
                    tmp = some_op(acc)
                    acc = tmp
                y = acc
        """
        out = self._transform(src)
        assert "acc_v1 = acc" in out  # carried value bridged across the boundary
        assert "y = acc_v1" in out

    def test_sibling_loops_reuse_inner_local_no_bridge(self):
        """Regression: a loop-local reused across sibling loops is not bridged.

        Mirrors the qwen3 decode attention kernel — per-stage SPMD loops each
        declare a fresh inner index of the same name.  Before the fix the
        renamer emitted a bridge ``row_v1 = row`` that read ``row`` outside its
        defining loop, which ``ConvertToSSA`` rejects with "used outside its
        defining scope".
        """
        src = """
            def f(a, out):
                for gi in pl.spmd(4):
                    for sb in pl.range(2):
                        row = sb * 8
                        out = pl.assemble(out, a, [row, 0])
                for gi in pl.spmd(4):
                    for sb in pl.range(2):
                        row = sb * 8 + 16
                        out = pl.assemble(out, a, [row, 0])
                return out
        """
        out = self._transform(src)
        assert "row_v1" not in out  # fresh inner local — no bridge, no rename

    def test_if_else_branch_rebind(self):
        """The same name assigned in both branches of an if/else is left alone.

        then/else are mutually exclusive sibling scopes — the else-branch can
        never read the then-branch's local, so its assignment is a fresh
        binding, not a rebind.  Aliasing the else-branch to ``vlen_v1`` would
        leave ``vlen_v1`` undefined whenever the then-branch runs.
        """
        src = """
            def f(is_last, a, b):
                if is_last:
                    vlen = a
                else:
                    vlen = b
                use(vlen)
        """
        out = self._transform(src)
        assert "vlen_v1" not in out  # no cross-branch alias
        assert "use(vlen)" in out

    def test_if_branch_local_intra_rebind_kept(self):
        """A branch-local rebound *within* one branch still gets versioned.

        ``x`` is assigned twice inside the same then-branch — a genuine
        straight-line rebind — so the second write is aliased to ``x_v1`` and
        the in-branch read follows it.  Only *cross-branch* reuse is exempt.
        The read is kept inside the branch so ``x`` is defined on every path
        that reaches it.
        """
        src = """
            def f(c):
                if c:
                    x = 1
                    x = 2
                    use(x)
        """
        out = self._transform(src)
        assert "x_v1 = 2" in out  # intra-branch rebind still versioned
        assert "use(x_v1)" in out  # in-branch read follows the rename

    def test_single_for_loop_carried_unchanged(self):
        """A single for loop with loop-carried variable is NOT renamed."""
        src = """
            def f():
                x = init_val
                for i in range(n):
                    x = some_op(x)
                y = x
        """
        out = self._transform(src)
        assert "x_v1" not in out
        assert "y = x" in out


class TestInferReturnType:
    """Coverage for `_infer_return_type` — the JIT specializer's annotation inferer."""

    @staticmethod
    def _meta(shape=(32, 32), dtype=DataType.FP32) -> TensorMeta:
        return TensorMeta(shape=shape, dtype=dtype)

    def test_tuple_return_emits_tuple_annotation(self):
        """`return out0, out1` -> `tuple[T_out0, T_out1]` (issue #1304)."""
        func_def = _parse_func(
            """
            def f(a, out0, out1):
                return out0, out1
            """
        )
        meta = {"a": self._meta(), "out0": self._meta(), "out1": self._meta()}
        ann = _infer_return_type(func_def, meta, ["out0", "out1"])
        assert ann == ("tuple[pl.Tensor[[32, 32], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]")

    def test_tuple_return_with_unknown_element_drops_annotation(self):
        """Tuple returns with non-tensor elements bail to no annotation."""
        func_def = _parse_func(
            """
            def f(a, out):
                return out, a_local
            """
        )
        meta = {"a": self._meta(), "out": self._meta()}
        # `a_local` isn't in tensor_meta — can't infer it.
        assert _infer_return_type(func_def, meta, ["out"]) is None

    @pytest.mark.parametrize("tuple_annotation", ["Tuple", "typing.Tuple"])
    def test_tuple_return_with_typing_annotation_preserves_scalar_element(self, tuple_annotation):
        """Explicit Tuple annotations provide metadata for non-tensor elements."""
        func_def = _parse_func(
            f"""
            def f(out) -> {tuple_annotation}[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
                return out, tid
            """
        )
        meta = {"out": self._meta()}
        ann = _infer_return_type(func_def, meta, ["out"])
        assert ann == "tuple[pl.Tensor[[32, 32], pl.FP32], pl.Scalar[pl.TASK_ID]]"

    def test_explicit_tuple_element_folds_module_constants(self):
        """A local return element uses the explicit annotation; module int
        constants in its shape are folded to literals so the emitted source
        carries concrete extents rather than unresolved names (would otherwise
        parse-fail with ``NameError: name 'BATCH' is not defined``)."""
        func_def = _parse_func(
            """
            def f(out) -> tuple[pl.Tensor[[BATCH, VOCAB], pl.FP32], pl.Tensor]:
                return logits, out
            """
        )
        meta = {"out": self._meta()}
        # `logits` (local) -> folded explicit annotation; `out` -> from meta.
        ann = _infer_return_type(func_def, meta, ["out"], py_globals={"BATCH": 16, "VOCAB": 128})
        assert ann == "tuple[pl.Tensor[[16, 128], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]"

    def test_bare_explicit_tuple_element_drops_annotation(self):
        """A bare (shapeless) explicit element for an un-inferrable local drops the
        whole annotation — the pre-#2016 behavior — instead of emitting an
        incomplete ``pl.Tensor`` the parser rejects (``Incomplete type
        annotation``). Regression for the qwen3 decode host wrapper."""
        func_def = _parse_func(
            """
            def f(out) -> tuple[pl.Tensor, pl.Tensor]:
                return logits, out
            """
        )
        meta = {"out": self._meta()}
        # `logits` is un-inferrable and its explicit element is a bare `pl.Tensor`.
        assert _infer_return_type(func_def, meta, ["out"], py_globals={}) is None

    def test_bare_element_alongside_task_id_keeps_annotation(self):
        """A bare tensor element must NOT drop a sibling TASK_ID's annotation.

        The parser cannot re-derive a TASK_ID from a value (that is why #2016
        added the explicit-annotation copy). So when a bare, un-inferrable
        element sits next to a subscripted one that needs the annotation, keep
        the annotation — the bare element then surfaces as a clear
        ``Incomplete type annotation`` instead of silently stripping the TASK_ID."""
        func_def = _parse_func(
            """
            def f(out) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
                return local_tensor, tid
            """
        )
        meta = {"out": self._meta()}
        # `local_tensor` bare + un-inferrable, `tid` needs its Scalar annotation.
        ann = _infer_return_type(func_def, meta, ["out"])
        assert ann == "tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]"

    def test_explicit_tuple_element_without_globals_stays_symbolic(self):
        """No ``py_globals`` (or an unknown name) leaves the annotation as-is —
        folding only substitutes names bound to module int/float/bool constants."""
        func_def = _parse_func(
            """
            def f(out) -> tuple[pl.Tensor[[BATCH, 128], pl.FP32], pl.Tensor]:
                return logits, out
            """
        )
        meta = {"out": self._meta()}
        ann = _infer_return_type(func_def, meta, ["out"])
        assert ann == "tuple[pl.Tensor[[BATCH, 128], pl.FP32], pl.Tensor[[32, 32], pl.FP32]]"

    @pytest.mark.parametrize("return_expr", ["tid", "0", "tid + 1"])
    def test_scalar_return_annotation_preserved_for_any_expression(self, return_expr):
        """Explicit scalar returns do not require a TensorMeta-backed name."""
        func_def = _parse_func(
            f"""
            def f() -> pl.Scalar[pl.TASK_ID]:
                return {return_expr}
            """
        )
        assert _infer_return_type(func_def, {}, []) == "pl.Scalar[pl.TASK_ID]"

    def test_return_call_drops_annotation(self):
        """`return f(...)` can't be inferred — caller may have multi-return."""
        func_def = _parse_func(
            """
            def f(a, out):
                return inline_call(a, out)
            """
        )
        meta = {"a": self._meta(), "out": self._meta()}
        assert _infer_return_type(func_def, meta, ["out"]) is None

    def test_bare_tensor_param_return_inferred(self):
        """Bare `pl.Tensor` returns (no `pl.Out`) — issue #1304 deprecation case."""
        func_def = _parse_func(
            """
            def f(a, out):
                return out
            """
        )
        meta = {"a": self._meta(), "out": self._meta()}
        # No out_params at all (bare pl.Tensor params).
        ann = _infer_return_type(func_def, meta, [])
        assert ann == "pl.Tensor[[32, 32], pl.FP32]"


class TestSpecializerExternReturnTypes:
    """Extern declarations keep complete types without changing ordinary DSL inference."""

    @staticmethod
    def _specialize(
        annotation: str, body: str = "...", func_type: str = "extern", one_output: bool = False
    ) -> str:
        params = "y: pl.Out[pl.Tensor]" if one_output else "y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]"
        ctx = _make_ctx(
            source=f"def kernel({params}){annotation}:\n    {body}",
            func_type=func_type,
            param_names=["y"] if one_output else ["y", "z"],
            tensor_meta={
                "y": TensorMeta((16, 16), DataType.INT8),
                "z": TensorMeta((16, 1), DataType.FP32),
            },
        )
        ctx.external_core_type = "aiv"
        ctx.external_aiv_source = "kernel.cpp"
        ctx.py_globals = {"Tuple": vars(typing)["Tuple"], "typing": typing, "pl": pl}
        return Specializer("Generated", [ctx]).specialize()

    @pytest.mark.parametrize("tuple_name", ["tuple", "Tuple", "typing.Tuple"])
    def test_complete_tuple_preserves_output_order(self, tuple_name):
        source = self._specialize(
            f" -> {tuple_name}[pl.Tensor[[16, 16], pl.INT8], pl.Tensor[[16, 1], pl.FP32]]"
        )
        assert "-> tuple[pl.Tensor[[16, 16], pl.INT8], pl.Tensor[[16, 1], pl.FP32]]:" in source

    def test_reordered_outputs_are_rejected(self):
        # A signature-only declaration has no return-to-parameter mapping. The
        # consumer metadata and runtime aliases must agree on parameter order.
        with pytest.raises(ValueError, match="result 0 must match output parameter 'y'"):
            self._specialize(" -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 16], pl.INT8]]")

    def test_partial_output_return_is_rejected(self):
        with pytest.raises(ValueError, match="1 results for 2 Out/InOut"):
            self._specialize(" -> pl.Tensor[[16, 1], pl.FP32]")

    @pytest.mark.parametrize("shape, dtype", [("[32, 16]", "INT8"), ("[16, 16]", "FP32")])
    def test_incompatible_single_output_type_is_rejected(self, shape, dtype):
        with pytest.raises(ValueError, match="result 0 must match output parameter 'y'"):
            self._specialize(f" -> pl.Tensor[{shape}, pl.{dtype}]", one_output=True)

    def test_nested_tuple_value_is_not_flattened(self):
        with pytest.raises(ValueError, match="single tuple-valued result"):
            self._specialize(" -> tuple[pl.Tuple[pl.Tensor[[16, 16], pl.INT8]]]")

    @pytest.mark.parametrize("constructor", [False, True])
    def test_dynamic_inout_and_out_keep_parameter_order(self, constructor):
        ctx = _make_ctx(
            source="""
                def kernel(y: pl.InOut[pl.Tensor], z: pl.Out[pl.Tensor]) -> tuple[
                    pl.Tensor[[ROWS, 16], pl.INT8], pl.Tensor[[ROWS, 1], pl.FP32]
                ]: ...
            """,
            func_type="extern",
            param_names=["y", "z"],
            tensor_meta={
                "y": TensorMeta((DynDim("ROWS", "rows", 16), 16), DataType.INT8),
                "z": TensorMeta((DynDim("ROWS", "rows", 16), 1), DataType.FP32),
            },
        )
        ctx.external_core_type = "aiv"
        ctx.external_aiv_source = "kernel.cpp"
        if constructor:
            ctx.source = ctx.source.replace(
                "pl.Tensor[[ROWS, 16], pl.INT8]", "pl.Tensor([ROWS, 16], pl.INT8)"
            )
            ctx.source = ctx.source.replace("pl.Tensor[[ROWS, 1], pl.FP32]", "pl.Tensor([ROWS, 1], pl.FP32)")
        # The emitted dynamic label must not fold to the unrelated global.
        ctx.py_globals = {"ROWS": pl.dynamic("rows"), "rows": 128}
        source = Specializer("Generated", [ctx]).specialize()
        assert "-> tuple[pl.Tensor[[rows, 16], pl.INT8], pl.Tensor[[rows, 1], pl.FP32]]:" in source

    def test_single_tuple_value_is_not_flattened_into_results(self):
        with pytest.raises(ValueError, match="single tuple-valued result"):
            self._specialize(" -> pl.Tuple[pl.Tensor[[16, 16], pl.INT8], pl.Tensor[[16, 1], pl.FP32]]")

    @pytest.mark.parametrize("elements", ["", "pl.Tensor([16, 16], pl.INT8), pl.Tensor([16, 1], pl.FP32)"])
    def test_legacy_tuple_constructor_is_not_inferred_as_tensor(self, elements):
        with pytest.raises(ValueError, match="single tuple-valued result"):
            self._specialize(f" -> 'pl.Tuple([{elements}])'")

    @pytest.mark.parametrize("annotation", ["", " -> pl.Tensor", " -> typing.Tuple"])
    @pytest.mark.parametrize("one_output", [False, True])
    def test_legacy_output_inference(self, annotation, one_output):
        source = self._specialize(annotation, one_output=one_output)
        if one_output:
            assert "-> pl.Tensor[[16, 16], pl.INT8]:" in source
        else:
            # The public legacy value remains y. The declaration now also
            # returns z internally so callers can materialize its writeback.
            assert "-> tuple[pl.Tensor[[16, 16], pl.INT8], pl.Tensor[[16, 1], pl.FP32]]:" in source
            assert "'external_return_params': [y, z]" in source

    @pytest.mark.parametrize(
        "annotation",
        [
            " -> tuple[pl.Tensor, pl.Tensor]",
            " -> tuple[pl.Tensor[[16, 16], pl.INT8], pl.Tensor]",
        ],
    )
    def test_incomplete_tuple_still_uses_return_statement(self, annotation):
        source = self._specialize(annotation, body="return y, z")
        assert "-> tuple[pl.Tensor[[16, 16], pl.INT8], pl.Tensor[[16, 1], pl.FP32]]:" in source

    def test_ordinary_dsl_return_still_uses_runtime_metadata(self):
        # An ordinary DSL function specializes its body and return from the
        # actual call metadata, even when a static annotation says otherwise.
        source = self._specialize(" -> pl.Tensor[[32, 32], pl.FP32]", body="return y", func_type="incore")
        assert "-> pl.Tensor[[16, 16], pl.INT8]:" in source


class TestSpecializerInlineDeprecation:
    """`@pl.jit.inline` should emit bare `pl.Tensor[...]` and warn on `pl.Out`."""

    def _spec(self, ctx_args: dict) -> str:
        ctx = _make_ctx(**ctx_args)
        return Specializer("Generated", [ctx]).specialize()

    def test_inline_strips_pl_out_and_warns(self):
        """Inline helpers with pl.Out -> bare pl.Tensor + DeprecationWarning."""
        src = """
            def util(a: pl.Tensor, out: pl.Out[pl.Tensor]):
                return out
        """
        meta = {
            "a": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
        }
        with pytest.warns(DeprecationWarning, match="Direction annotations are deprecated"):
            out = self._spec(
                {
                    "func_name": "util",
                    "source": src,
                    "func_type": "inline",
                    "param_names": ["a", "out"],
                    "tensor_meta": meta,
                }
            )
        # Generated source must use bare pl.Tensor[...] for `out`, not pl.Out[...].
        assert "out: pl.Tensor[[32, 32], pl.FP32]" in out
        assert "pl.Out[pl.Tensor[[32, 32], pl.FP32]]" not in out

    def test_inline_with_bare_tensor_no_warning(self):
        """Bare `pl.Tensor` inline params don't trigger the deprecation warning."""
        src = """
            def util(a: pl.Tensor, out: pl.Tensor):
                return out
        """
        meta = {
            "a": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._spec(
                {
                    "func_name": "util",
                    "source": src,
                    "func_type": "inline",
                    "param_names": ["a", "out"],
                    "tensor_meta": meta,
                }
            )
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_orchestration_keeps_pl_out(self):
        """Non-inline functions keep `pl.Out[...]` annotations untouched."""
        src = """
            def util(a: pl.Tensor, out: pl.Out[pl.Tensor]):
                return out
        """
        meta = {
            "a": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
        }
        out = self._spec(
            {
                "func_name": "util",
                "source": src,
                "func_type": "orchestration",
                "param_names": ["a", "out"],
                "tensor_meta": meta,
            }
        )
        assert "pl.Out[pl.Tensor[[32, 32], pl.FP32]]" in out


class TestSpecializerSourceMap:
    """specialize() builds a generated→original line map (issue #1612).

    The map covers each function's signature line plus its body statements. The
    signature entry is what lets an annotation-level parse error (e.g. a
    rejected layout) report against the user's file instead of the generated
    source, so these tests assert on the two kinds separately.
    """

    #: Original file line of the ``def`` in each fixture below: the source
    #: starts at the decorator (``orig_start_line``) with ``def`` one line down.
    _DEF_LINE = 101

    def _meta(self) -> TensorMeta:
        return TensorMeta(shape=(32, 32), dtype=DataType.FP32)

    def test_maps_statements_to_original_file_lines(self):
        # Source as inspect.getsourcelines would return it: first line is the
        # decorator, so body statements sit at dedented lines 3, 4, 5.
        src = textwrap.dedent(
            """
            @pl.jit
            def kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
                t = pl.load(a, [0, 0], [32, 32])
                pl.store(t, [0, 0], out)
                return out
            """
        ).strip("\n")
        ctx = _make_ctx(
            func_name="kernel",
            source=src,
            param_names=["a", "out"],
            tensor_meta={"a": self._meta(), "out": self._meta()},
        )
        # Pretend the kernel's first source line lives at line 100 of a real file.
        ctx.orig_file = "/real/kernel.py"
        ctx.orig_start_line = 100

        spec = Specializer("Gen", [ctx])
        spec.specialize()
        source_map = spec.source_map

        # Exact multiset: the signature line at column 0 (so annotation errors
        # land in the user's file) plus dedented body lines 3,4,5 +
        # (orig_start_line - 1), each indented one level inside the function.
        assert sorted(source_map.values()) == [
            ("/real/kernel.py", self._DEF_LINE, 0),
            ("/real/kernel.py", 102, 4),
            ("/real/kernel.py", 103, 4),
            ("/real/kernel.py", 104, 4),
        ]

    def test_no_real_file_yields_empty_map(self):
        """Without on-disk source (orig_file=None), no mapping is produced."""
        src = textwrap.dedent(
            """
            def kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
                return out
            """
        ).strip("\n")
        ctx = _make_ctx(
            func_name="kernel",
            source=src,
            param_names=["a", "out"],
            tensor_meta={"a": self._meta(), "out": self._meta()},
        )
        # orig_file defaults to None.
        spec = Specializer("Gen", [ctx])
        spec.specialize()
        assert spec.source_map == {}

    def test_synthesized_statements_are_not_mapped(self):
        """Synthesized statements must be skipped, not mis-mapped to the def line.

        ``ast.fix_missing_locations`` backfills a synthesized statement's missing
        lineno with the function's ``lineno=1``, which (without the pre-fix
        capture) would map it to the kernel's decorator line. Here the whole body
        deletes (``K = a.shape[1]`` is inlined), collapsing to a synthesized
        ``pass`` that must produce no mapping at all. See issue #1612.
        """
        src = textwrap.dedent(
            """
            @pl.jit
            def kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
                K = a.shape[1]
            """
        ).strip("\n")
        ctx = _make_ctx(
            func_name="kernel",
            source=src,
            param_names=["a", "out"],
            tensor_meta={"a": self._meta(), "out": self._meta()},
        )
        ctx.orig_file = "/real/kernel.py"
        ctx.orig_start_line = 100

        spec = Specializer("Gen", [ctx])
        spec.specialize()
        # The body collapses to a synthesized `pass`; it must not be mapped, and
        # in particular must not point at the decorator line (100). Only the
        # signature line remains.
        assert list(spec.source_map.values()) == [("/real/kernel.py", self._DEF_LINE, 0)]


# ---------------------------------------------------------------------------
# Array-typed parameter annotations (e.g. pl.Array[N, pl.TASK_ID])
# ---------------------------------------------------------------------------


class TestArrayParamAnnotation:
    """A param that is neither a tensor nor a scalar — a fixed-size on-core
    ``pl.Array[N, dtype]`` (the TaskId-array dependency-threading idiom) — must
    round-trip through specialization with its annotation rendered.

    Before the fix, ``_build_params`` emitted such params *bare* (just the name).
    The generated ``@pl.function`` source then failed PyPTO parsing with
    ``ParserTypeError: Parameter '<name>' missing type annotation``.
    """

    def _specialize_with_globals(
        self,
        func_source: str,
        param_names: list[str],
        tensor_meta: dict[str, TensorMeta],
        py_globals: dict,
        func_type: str = "orchestration",
    ) -> str:
        ctx = SpecializeContext(
            func_name="kernel",
            source=textwrap.dedent(func_source),
            func_type=func_type,
            level=None,
            param_names=param_names,
            tensor_meta=tensor_meta,
            scalar_dtypes={},
            py_globals=py_globals,
        )
        return specialize("_TestClass", [ctx])

    # --- _build_params plumbing (unit) --------------------------------------

    def test_build_params_renders_extra_ann(self):
        """extra_anns entries are emitted as ``name: <annotation>``."""
        ctx = _make_ctx(
            param_names=["a", "tids"],
            tensor_meta={"a": TensorMeta((4, 4), DataType.FP32)},
        )
        spec = Specializer("_T", [ctx])
        params = spec._build_params(
            all_param_names=["a", "tids"],
            out_params=[],
            inout_params=[],
            tensor_params=["a"],
            scalar_dtype_strs={},
            distributed_params=set(),
            ctx=ctx,
            extra_anns={"tids": "pl.Array[4, pl.TASK_ID]"},
        )
        assert "tids: pl.Array[4, pl.TASK_ID]" in params

    def test_build_params_without_extra_ann_stays_bare(self):
        """A non-tensor, non-scalar param with no extra_anns is emitted bare —
        documents the pre-fix behaviour that broke PyPTO parsing."""
        ctx = _make_ctx(
            param_names=["a", "tids"],
            tensor_meta={"a": TensorMeta((4, 4), DataType.FP32)},
        )
        spec = Specializer("_T", [ctx])
        params = spec._build_params(
            all_param_names=["a", "tids"],
            out_params=[],
            inout_params=[],
            tensor_params=["a"],
            scalar_dtype_strs={},
            distributed_params=set(),
            ctx=ctx,
            extra_anns=None,
        )
        assert "tids" in params  # bare, unannotated

    # --- end-to-end rendering through specialize() --------------------------

    def test_task_id_array_param_rendered(self):
        """``tids: pl.Array[N, pl.TASK_ID]`` renders with N folded to its value."""
        N = 4
        src = """
            def kernel(a: pl.Tensor, tids: pl.Array[N, pl.TASK_ID], out: pl.Out[pl.Tensor]):
                return out
        """
        out = self._specialize_with_globals(
            src,
            ["a", "tids", "out"],
            {
                "a": TensorMeta((16, 16), DataType.FP32),
                "out": TensorMeta((16, 16), DataType.FP32),
            },
            py_globals={"pl": pl, "N": N},
        )
        assert "tids: pl.Array[4, pl.TASK_ID]" in out

    def test_array_param_literal_extent(self):
        """A literal extent (no closure constant) renders unchanged."""
        src = """
            def kernel(a: pl.Tensor, tids: pl.Array[3, pl.TASK_ID], out: pl.Out[pl.Tensor]):
                return out
        """
        out = self._specialize_with_globals(
            src,
            ["a", "tids", "out"],
            {
                "a": TensorMeta((8, 8), DataType.FP32),
                "out": TensorMeta((8, 8), DataType.FP32),
            },
            py_globals={"pl": pl},
        )
        assert "tids: pl.Array[3, pl.TASK_ID]" in out

    def test_array_param_non_task_id_dtype(self):
        """The rendering generalises to any Array dtype, not just TASK_ID."""
        src = """
            def kernel(a: pl.Tensor, route: pl.Array[8, pl.INT32], out: pl.Out[pl.Tensor]):
                return out
        """
        out = self._specialize_with_globals(
            src,
            ["a", "route", "out"],
            {
                "a": TensorMeta((8, 8), DataType.FP32),
                "out": TensorMeta((8, 8), DataType.FP32),
            },
            py_globals={"pl": pl},
        )
        assert "route: pl.Array[8, pl.INT32]" in out

    def test_array_param_bf16_dtype_renders_pl_constant(self):
        """Regression: a dtype whose enum ``str`` diverges from its DSL constant
        (``BF16`` → enum str ``bfloat16``) must render as the real ``pl.BF16``,
        not ``pl.BFLOAT16`` (which does not exist on ``pl`` and would make the
        generated source fail at import)."""
        src = """
            def kernel(a: pl.Tensor, buf: pl.Array[4, pl.BF16], out: pl.Out[pl.Tensor]):
                return out
        """
        out = self._specialize_with_globals(
            src,
            ["a", "buf", "out"],
            {
                "a": TensorMeta((8, 8), DataType.FP32),
                "out": TensorMeta((8, 8), DataType.FP32),
            },
            py_globals={"pl": pl},
        )
        assert "buf: pl.Array[4, pl.BF16]" in out
        assert "BFLOAT16" not in out

    @pytest.mark.parametrize("shadowed", [False, True])
    def test_array_with_module_alias_and_dtype_constant(self, shadowed):
        """Original aliases work whether canonical ``pl`` is absent or shadowed."""
        src = """
            def kernel(buf: p.Array[4, DTYPE]):
                return buf
        """
        namespace: dict[str, Any] = {"p": pl, "DTYPE": pl.BF16}
        if shadowed:
            namespace["pl"] = object()
        out = self._specialize_with_globals(src, ["buf"], {}, py_globals=namespace)
        assert "buf: pl.Array[4, pl.BF16]" in out

    def test_annotation_evaluation_respects_original_pl_binding(self):
        """Canonical rendering must not make an invalid original annotation valid."""
        src = """
            def kernel(buf: pl.Array[4, DTYPE]):
                return buf
        """
        out = self._specialize_with_globals(
            src,
            ["buf"],
            {},
            py_globals={"p": pl, "pl": object(), "DTYPE": pl.BF16},
        )
        assert "def kernel(self, buf):" in out

    def test_generated_source_is_parseable(self):
        """The generated source is syntactically valid Python and carries no
        bare (unannotated) parameter — the concrete regression guard."""
        N = 5
        src = """
            def kernel(a: pl.Tensor, tids: pl.Array[N, pl.TASK_ID], out: pl.Out[pl.Tensor]):
                return out
        """
        out = self._specialize_with_globals(
            src,
            ["a", "tids", "out"],
            {
                "a": TensorMeta((16, 16), DataType.FP32),
                "out": TensorMeta((16, 16), DataType.FP32),
            },
            py_globals={"pl": pl, "N": N},
        )
        ast.parse(out)  # must be valid Python
        # The param must be annotated, never emitted bare as ", tids,".
        assert "tids: pl.Array[5, pl.TASK_ID]" in out
        assert ", tids," not in out and ", tids)" not in out

    def test_unevaluable_annotation_stays_bare_without_crash(self):
        """An annotation that cannot be evaluated (unknown name, not in globals)
        is best-effort skipped — the specializer must not raise."""
        src = """
            def kernel(a: pl.Tensor, weird: SomeUndefinedType, out: pl.Out[pl.Tensor]):
                return out
        """
        # No exception expected; `weird` simply stays bare.
        out = self._specialize_with_globals(
            src,
            ["a", "weird", "out"],
            {
                "a": TensorMeta((8, 8), DataType.FP32),
                "out": TensorMeta((8, 8), DataType.FP32),
            },
            py_globals={"pl": pl},
        )
        assert "pl.Array" not in out  # not misclassified as an Array

    def test_annotation_eval_excludes_builtins(self):
        """The annotation-eval namespace strips Python builtins: an annotation
        that depends on one (here ``len``) cannot evaluate, so it is silently
        skipped (stays bare) rather than executed at specialization time."""
        src = """
            def kernel(a: pl.Tensor, x: pl.Array[len([0, 0, 0, 0]), pl.TASK_ID], out: pl.Out[pl.Tensor]):
                return out
        """
        out = self._specialize_with_globals(
            src,
            ["a", "x", "out"],
            {
                "a": TensorMeta((8, 8), DataType.FP32),
                "out": TensorMeta((8, 8), DataType.FP32),
            },
            py_globals={"pl": pl},
        )
        # `len(...)` is unavailable without builtins -> eval fails -> no Array
        # annotation is emitted for `x` (best-effort, no execution, no crash).
        assert "x: pl.Array" not in out


def _annotation_function(
    func: Any, annotations: dict[str, Any], annotation_globals: dict[str, Any] | None = None
) -> types.FunctionType:
    """Clone annotation bindings without mutating the defining module's globals."""
    clone = types.FunctionType(
        func.__code__, {**func.__globals__, **(annotation_globals or {})}, func.__name__
    )
    clone.__annotations__ = annotations
    return clone


def _inline_context(
    func: Any,
    *,
    tensor_meta: dict[str, TensorMeta] | None = None,
    constexpr_values: dict[str, str] | None = None,
) -> SpecializeContext:
    """Build an inline annotation context without unrelated caller metadata."""
    return build_specialize_context(
        func,
        func.__name__,
        "inline",
        None,
        tensor_meta or {},
        {},
        [],
        auto_scope=False,
        constexpr_values=constexpr_values,
    )


def _specialized_function(ctx: SpecializeContext, source: str | None = None) -> ir.Function:
    """Parse generated annotation source and retrieve its typed function."""
    program = pl.parse(source if source is not None else specialize("_TestClass", [ctx]))
    assert isinstance(program, ir.Program)
    func = program.get_function(ctx.func_name)
    assert func is not None
    return func


class TestInlineTileParamAnnotations:
    """Explicit Tile parameters must survive JIT signature specialization."""

    @pytest.mark.parametrize(
        "label",
        [
            "pl",
            "pld",
            "class",
            "for",
            "__debug__",
            "\uff50\uff4c",
            "\U0001d699\U0001d695",
            "\uff50\uff4c\uff44",
            "\uff43\uff4c\uff41\uff53\uff53",
            "\uff2d",
            "\U0001d474",
        ],
    )
    @pytest.mark.parametrize("postponed", [False, True])
    def test_unsafe_dynamic_labels_rejected_before_source_emission(self, label, postponed):
        """Canonical Tile symbols cannot overwrite DSL imports or invalidate Python syntax."""
        dimension = pl.dynamic(label)

        def keep_tile(tile: pl.Tile[[dimension, 64], pl.FP32]):
            return tile

        helper = keep_tile
        if postponed:
            helper = _annotation_function(
                keep_tile, {"tile": "pl.Tile[[dimension, 64], pl.FP32]"}, {"dimension": dimension}
            )
        with pytest.raises(ValueError, match=rf"dynamic symbol.*'{label}'.*different"):
            ctx = _inline_context(helper)
            specialize("_TestClass", [ctx])

    @pytest.mark.parametrize("label", ["class", "for", "__debug__"])
    @pytest.mark.parametrize("metadata", [False, True], ids=["shape", "metadata"])
    def test_parameterized_annotation_fingerprint_rejects_keyword_label(self, label, metadata):
        """Cache rendering diagnoses an invalid free label before parsing canonical source."""
        dimension = pl.dynamic(label)

        def keep_tile(tile, ROWS):
            return tile

        component = (
            "[[ROWS, 64], pl.FP32, pl.TileView(valid_shape=[dimension, 64])]"
            if metadata
            else "[[ROWS, dimension], pl.FP32]"
        )
        helper = _annotation_function(
            keep_tile, {"tile": f"pl.Tile{component}", "ROWS": "pl.constexpr"}, {"dimension": dimension}
        )
        with pytest.raises(ValueError, match=rf"dynamic symbol.*'{label}'.*different"):
            tile_annotation_fingerprints(helper)

    @pytest.mark.parametrize(
        "label",
        ["match", "case", "type", "_pypto_tile_param_0", "__name__", "__builtins__", "\u884c", "\u03b1"],
    )
    def test_safe_dynamic_labels_still_parse(self, label):
        """Soft keywords and internal-looking user symbols remain legal declarations."""
        dimension = pl.dynamic(label)

        def keep_tile(tile: pl.Tile[[dimension, 64], pl.FP32]):
            return tile

        ctx = _inline_context(keep_tile)
        source = specialize("_TestClass", [ctx])
        assert f'{label} = pl.dynamic("{label}")' in source
        helper = _specialized_function(ctx, source)
        tile_type = helper.params[0].type
        assert isinstance(tile_type, ir.TileType)
        assert isinstance(tile_type.shape[0], ir.Var)
        assert tile_type.shape[0].name_hint == label

    @pytest.mark.parametrize(
        "label",
        [
            "pl",
            "pld",
            "class",
            "__debug__",
            "\uff50\uff4c",
            "\uff43\uff4c\uff41\uff53\uff53",
            "match",
            "\u884c",
        ],
    )
    def test_dynamic_label_validation_through_jit_compile(self, label):
        """The public compile path rejects unsafe labels and compiles a legal soft keyword."""
        dimension = pl.dynamic(label)

        def compile_entry():
            @pl.jit.inline(auto_scope=False)
            def keep_tile(tile: pl.Tile[[dimension, 64], pl.FP32]):
                return tile

            @pl.jit.incore
            def copy_incore(src: pl.Tensor[[8, 64], pl.FP32], dst: pl.Out[pl.Tensor[[8, 64], pl.FP32]]):
                tile = pl.load(src, [0, 0], [8, 64])
                kept = keep_tile(tile)
                return pl.store(kept, [0, 0], dst)

            @pl.jit
            def entry(src: pl.Tensor[[8, 64], pl.FP32], dst: pl.Out[pl.Tensor[[8, 64], pl.FP32]]):
                return copy_incore(src, dst)

            return entry.compile(
                config=RunConfig(
                    backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.SIM, codegen_only=True
                )
            )

        if label not in {"match", "\u884c"}:
            with pytest.raises(ValueError, match=rf"dynamic symbol.*'{label}'.*different"):
                compile_entry()
            return
        assert compile_entry() is not None

    @pytest.mark.parametrize("label", ["not-valid", "bad label"])
    def test_invalid_native_annotation_symbol_has_clear_diagnostic(self, label):
        """Prebuilt IR metadata cannot bypass the generated-identifier validation."""
        dimension = pl.Scalar(expr=ir.Var(label, ir.ScalarType(pl.INDEX), ir.Span.unknown()))

        def keep_tile(tile: pl.Tile[[8, 64], pl.FP32, pl.TileView(valid_shape=[dimension, 64])]):
            return tile

        with pytest.raises(ValueError, match=rf"dynamic symbol.*'{label}'.*different"):
            _inline_context(keep_tile)

    @pytest.mark.parametrize("literal", ["pl", "__debug__"])
    def test_tensor_alias_with_reserved_literal_keeps_its_safe_binding(self, literal):
        """A safe emitted Tensor alias is distinct from the DynVar's internal literal."""
        ctx = _make_ctx(
            "entry",
            "def entry(src: pl.Tensor):\n    return src",
            param_names=["src"],
            tensor_meta={"src": TensorMeta((DynDim("ROWS", literal, 8), 64), DataType.FP32)},
            auto_scope=False,
        )
        source = specialize("_TestClass", [ctx])
        assert f'ROWS = pl.dynamic("{literal}")' in source
        program = pl.parse(source)
        assert isinstance(program, ir.Program)

    @pytest.mark.parametrize("postponed", [False, True])
    @pytest.mark.parametrize("symbol_name", ["tile", "anchor"])
    def test_generic_tile_symbol_can_match_nonscalar_parameter(self, postponed, symbol_name):
        """Shape globals remain needed even when an earlier parameter has the same name."""
        ROWS = pl.dynamic(symbol_name)

        def keep_tile(
            anchor: pl.Tensor[[8, 64], pl.FP32],
            tile: pl.Tile[[ROWS, 64], pl.FP32],
            other: pl.Tile[[ROWS, 64], pl.FP32],
        ):
            return tile

        helper = keep_tile
        if postponed:
            helper = _annotation_function(
                keep_tile,
                {
                    "anchor": "pl.Tensor[[8, 64], pl.FP32]",
                    "tile": "pl.Tile[[ROWS, 64], pl.FP32]",
                    "other": "pl.Tile[[ROWS, 64], pl.FP32]",
                },
                {"ROWS": ROWS},
            )
        ctx = _inline_context(helper, tensor_meta={"anchor": TensorMeta((8, 64), DataType.FP32)})
        source = specialize("_TestClass", [ctx])
        assert f'{symbol_name} = pl.dynamic("{symbol_name}")' in source
        func = _specialized_function(ctx, source)
        for param in func.params[1:]:
            tile_type = param.type
            assert isinstance(tile_type, ir.TileType)
            assert isinstance(tile_type.shape[0], ir.Var)
            assert isinstance(tile_type.shape[0].type, ir.ScalarType)
            assert tile_type.shape[0].name_hint == symbol_name

    @pytest.mark.parametrize("postponed", [False, True])
    def test_scalar_dimension_matching_parameter_stays_local(self, postponed):
        """A real Scalar supplies the dimension and must not be shadowed by a new global."""
        tile = pl.dynamic("tile")

        def keep_tile(tile: pl.Scalar[pl.INDEX], value: pl.Tile[[tile, 64], pl.FP32]):
            return value

        helper = keep_tile
        if postponed:
            helper = _annotation_function(
                keep_tile,
                {"tile": "pl.Scalar[pl.INDEX]", "value": "pl.Tile[[tile, 64], pl.FP32]"},
                {"tile": tile},
            )
        ctx = _inline_context(helper)
        source = specialize("_TestClass", [ctx])
        assert 'tile = pl.dynamic("tile")' not in source
        func = _specialized_function(ctx, source)
        tile_type = func.params[1].type
        assert isinstance(tile_type, ir.TileType)
        assert ir.structural_equal(tile_type.shape[0], func.params[0])

    @pytest.mark.parametrize("rows", [8, 16])
    @pytest.mark.parametrize("mixed_axes", [False, True])
    def test_free_dynvar_name_is_not_a_constexpr_reference(self, rows, mixed_axes):
        """Canonical free-symbol spelling must not turn into a helper parameter use."""
        M = pl.dynamic("ROWS")
        ROWS = 99
        if mixed_axes:

            def keep_tile(tile: pl.Tile[[M, ROWS], pl.FP32], ROWS: pl.constexpr):
                return tile

        else:

            def keep_tile(tile: pl.Tile[[M, 64], pl.FP32], ROWS: pl.constexpr):
                return tile

        ctx = _inline_context(keep_tile, constexpr_values={"ROWS": str(rows)})
        func = _specialized_function(ctx)
        tile_type = func.params[0].type
        assert isinstance(tile_type, ir.TileType)
        assert isinstance(tile_type.shape[0], ir.Var)
        assert tile_type.shape[0].name_hint == "ROWS"
        assert isinstance(tile_type.shape[1], ir.ConstInt)
        assert tile_type.shape[1].value == (rows if mixed_axes else 64)

    def test_parameter_marker_avoids_folded_free_symbol_names(self):
        """A free alias may use the marker prefix even when eager math folds it away."""
        M = pl.dynamic("_pypto_tile_param_0")
        ROWS = 0

        def keep_tile(tile: pl.Tile[[ROWS * M, 64], pl.FP32], ROWS: pl.constexpr):
            return tile

        # This hygiene case requires a source-evaluable free operand. A
        # factory-only operand erased inside a product is diagnosed separately.
        helper = _annotation_function(keep_tile, keep_tile.__annotations__, {"M": M})
        ctx = _inline_context(helper, constexpr_values={"ROWS": "8"})
        func = _specialized_function(ctx)
        span = ir.Span.unknown()
        ir.assert_structural_equal(
            func.params[0].type,
            ir.TileType(
                [ir.ConstInt(8, DataType.INDEX, span) * M.unwrap(), ir.ConstInt(64, DataType.INDEX, span)],
                DataType.FP32,
            ),
            enable_auto_mapping=True,
        )

    @pytest.mark.parametrize("postponed", [False, True])
    def test_distinct_free_symbol_matching_scalar_name_is_diagnosed(self, postponed):
        """Separate free and local dimensions cannot silently share one emitted name."""
        M = pl.dynamic("tile")
        tile = 99

        def keep_tile(tile: pl.Scalar[pl.INDEX], value: pl.Tile[[M, tile], pl.FP32]):
            return value

        helper = keep_tile
        if postponed:
            helper = _annotation_function(
                keep_tile, {"tile": "pl.Scalar[pl.INDEX]", "value": "pl.Tile[[M, tile], pl.FP32]"}, {"M": M}
            )
        with pytest.raises(ValueError, match="Free dynamic symbol 'tile'.*Scalar parameter"):
            ctx = _inline_context(helper)
            specialize("_TestClass", [ctx])

    @pytest.mark.parametrize("postponed", [False, True])
    def test_scalar_metadata_override_keeps_true_local_binding(self, postponed):
        """A raw ambient view is replaced before checking the genuine Scalar dimension."""
        count = pl.dynamic("count")

        def keep_tile(
            count: pl.Scalar[pl.INDEX],
            tile: pl.Tile[
                [8, 64], pl.FP32, pl.TileView(valid_shape=[count, 64], start_offset=count.unwrap())
            ],
        ):
            return tile

        helper = keep_tile
        if postponed:
            helper = _annotation_function(
                keep_tile,
                {
                    "count": "pl.Scalar[pl.INDEX]",
                    "tile": "pl.Tile[[8, 64], pl.FP32, "
                    "pl.TileView(valid_shape=[count, 64], start_offset=count.unwrap())]",
                },
            )
        ctx = _inline_context(helper)
        source = specialize("_TestClass", [ctx])
        assert 'count = pl.dynamic("count")' not in source
        func = _specialized_function(ctx, source)
        tile_type = func.params[1].type
        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert ir.structural_equal(tile_type.tile_view.valid_shape[0], func.params[0])
        assert tile_type.tile_view.start_offset is not None
        assert ir.structural_equal(tile_type.tile_view.start_offset, func.params[0])

    def test_direction_wrapper_unwrap_uses_object_identity(self):
        source = "Direction[pl.Tile[[8, 64], pl.FP32]]"
        for wrapper in (pl.Out, pl.InOut):
            node = _unwrap_direction_annotation(ast.parse(source, mode="eval").body, {"Direction": wrapper})
            assert ast.unparse(node) == "pl.Tile[[8, 64], pl.FP32]"
        # An unrelated object named InOut has no DSL passthrough contract.
        source = "InOut[pl.Tile[[8, 64], pl.FP32]]"
        node = _unwrap_direction_annotation(ast.parse(source, mode="eval").body, {"InOut": object()})
        assert ast.unparse(node) == source

    @pytest.mark.parametrize("dtype", ["FP32", "BF16"])
    @pytest.mark.parametrize("func_type", ["inline", "incore"])
    def test_explicit_tile_annotation_preserved(self, dtype, func_type):
        ctx = _make_ctx(
            "keep_tile",
            f"""
                def keep_tile(tile: pl.Tile[[8, 64], pl.{dtype}, pl.MemorySpace.Vec]) -> pl.Tile:
                    return tile
            """,
            func_type=func_type,
            param_names=["tile"],
            auto_scope=False,
        )
        ctx.py_globals = {"pl": pl}

        func = _specialized_function(ctx)
        assert func.func_type == (ir.FunctionType.Inline if func_type == "inline" else ir.FunctionType.InCore)
        assert len(func.params) == 1
        ir.assert_structural_equal(
            func.params[0].type,
            ir.TileType([8, 64], getattr(DataType, dtype), memory_space=ir.MemorySpace.Vec),
        )

    @pytest.mark.parametrize(
        "tile_name,view_name,memref_name,space_name",
        [
            ("p.Tile", "p.TileView", "p.MemRef", "p.Mem.Vec"),
            ("Tile", "View", "Ref", "MS.Vec"),
        ],
    )
    def test_aliases_render_canonically_without_losing_metadata(
        self, tile_name, view_name, memref_name, space_name
    ):
        """Aliases disappear from generated source while the full Tile contract remains."""
        ctx = _make_ctx(
            "keep_tile",
            f"""
                def keep_tile(tile: {tile_name}[[8, 64], DTYPE,
                              {memref_name}("tile_buffer", 64, 2048), {space_name},
                              {view_name}(valid_shape=[3, 64], stride=[64, 1], start_offset=0,
                                          pad=p.PadValue.zero, compact=p.CompactMode.normal)]):
                    return tile
            """,
            func_type="inline",
            param_names=["tile"],
            auto_scope=False,
        )
        ctx.py_globals = {
            "p": pl,
            "pl": object(),
            "Tile": pl.Tile,
            "View": pl.TileView,
            "Ref": pl.MemRef,
            "MS": pl.Mem,
            "DTYPE": pl.BF16,
        }

        source = specialize("_TestClass", [ctx])
        assert "tile: pl.Tile[[8, 64], pl.BF16," in source
        assert "pl.TileView(" in source and "pl.MemRef(" in source
        func = _specialized_function(ctx, source)
        expected_type = ir.TileType(
            [8, 64],
            DataType.BF16,
            memref=ir.MemRef("tile_buffer", 64, 2048),
            tile_view=ir.TileView(
                valid_shape=[3, 64],
                stride=[64, 1],
                start_offset=ir.ConstInt(0, DataType.INDEX, ir.Span.unknown()),
                pad=ir.PadValue.zero,
                compact=ir.CompactMode.normal,
            ),
            memory_space=ir.MemorySpace.Vec,
        )
        ir.assert_structural_equal(func.params[0].type, expected_type, enable_auto_mapping=True)

    @pytest.mark.parametrize("shadowed", [False, True])
    def test_eager_factory_annotation_keeps_original_local_value(self, monkeypatch, shadowed):
        """An annotation-only local cannot be recovered from globals or closure cells."""

        def factory(tile_rows):
            dtype = pl.BF16
            allocation = pl.MemRef("tile_buffer", 64, 2048)
            view = pl.TileView(valid_shape=[3, 64], stride=[64, 1], start_offset=0, pad=pl.PadValue.zero)

            def keep_tile(tile: pl.Tile[[tile_rows, 64], dtype, allocation, pl.Mem.Vec, view]):
                return tile

            return keep_tile

        helper = factory(8)
        assert not helper.__code__.co_freevars
        if shadowed:
            # A same-named module global must not replace the local value that
            # Python already used when it evaluated the eager annotation.
            monkeypatch.setitem(helper.__globals__, "tile_rows", 99)
        ctx = _inline_context(helper)
        func = _specialized_function(ctx)
        original = helper.__annotations__["tile"]
        expected = ir.TileType(
            original.shape,
            original.dtype,
            memref=original.memref,
            tile_view=original.tile_view,
            memory_space=original.memory_space,
        )
        ir.assert_structural_equal(func.params[0].type, expected, enable_auto_mapping=True)

    def test_eager_generic_tile_normalizes_static_axes(self):
        """An evaluated [M, 64] mixes a DynVar and an int before IR normalization."""
        M = pl.dynamic("M")

        def keep_tile(tile: pl.Tile[[M, 64], pl.FP32, pl.Mem.Vec]):
            return tile

        def entry(src: pl.Tensor[[M, 64], pl.FP32]):
            return src

        helper_ctx = _inline_context(keep_tile)
        entry_ctx = build_specialize_context(
            entry,
            "entry",
            "orchestration",
            None,
            {"src": TensorMeta((DynDim("M", "M", 8), 64), DataType.FP32)},
            {},
            [],
        )
        source = specialize("_TestClass", [helper_ctx, entry_ctx])
        assert 'M = pl.dynamic("M")' in source
        helper = _specialized_function(helper_ctx, source)
        tile_type = helper.params[0].type
        assert isinstance(tile_type, ir.TileType)
        assert isinstance(tile_type.shape[0], ir.Var)
        assert tile_type.shape[0].name_hint == "M"
        assert isinstance(tile_type.shape[1], ir.ConstInt)
        assert tile_type.shape[1].value == 64

    @pytest.mark.parametrize(
        "annotation",
        [
            "pl.Tile[[ROWS, 64], pl.FP32, pl.Mem.Vec]",
            "pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, "
            "pl.TileView(valid_shape=[ROWS, 64], stride=[ROWS * 64, 1], start_offset=0)]",
        ],
        ids=["physical-shape", "tile-view"],
    )
    def test_source_tile_only_dynvars_use_canonical_names(self, annotation):
        """Tile-only symbols need declarations and may use a different Python alias."""
        ctx = _make_ctx(
            "keep_tile",
            f"""
                def keep_tile(tile: {annotation}):
                    return tile
            """,
            func_type="inline",
            param_names=["tile"],
            auto_scope=False,
        )
        ctx.py_globals = {"pl": pl, "ROWS": pl.dynamic("TILE_ROWS")}

        source = specialize("_TestClass", [ctx])
        assert 'TILE_ROWS = pl.dynamic("TILE_ROWS")' in source
        helper = _specialized_function(ctx, source)
        tile_type = helper.params[0].type
        assert isinstance(tile_type, ir.TileType)
        assert "TILE_ROWS" in ir.python_print_type(tile_type)

    def test_eager_factory_tile_view_declares_its_symbols(self):
        """Dynamic eager metadata retains its factory-only symbol and canonical name."""

        def factory():
            rows = pl.dynamic("TILE_VALID_ROWS")
            view = pl.TileView(valid_shape=[rows, 64], stride=[rows * 64, 1], start_offset=rows.unwrap())

            def keep_tile(tile: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, view]):
                return tile

            return keep_tile

        helper = factory()
        assert not helper.__code__.co_freevars
        ctx = _inline_context(helper)
        source = specialize("_TestClass", [ctx])
        assert 'TILE_VALID_ROWS = pl.dynamic("TILE_VALID_ROWS")' in source
        func = _specialized_function(ctx, source)
        assert "TILE_VALID_ROWS" in ir.python_print_type(func.params[0].type)

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_source_tile_aggregate_and_metadata_constants_are_canonical(self, dynamic):
        """Evaluated generic shapes and metadata cannot remain undefined aliases."""
        dimension = pl.dynamic("TILE_ROWS") if dynamic else 8
        allocation = pl.MemRef("tile_buffer", 64, 4096)
        view = pl.TileView(valid_shape=[3, 64], stride=[64, 1], start_offset=0, pad=pl.PadValue.zero)
        ctx = _make_ctx(
            "keep_tile",
            """
                def keep_tile(tile: pl.Tile[SHAPE, DTYPE, ALLOCATION, SPACE, VIEW]):
                    return tile
            """,
            func_type="inline",
            param_names=["tile"],
            auto_scope=False,
        )
        ctx.py_globals = {
            "pl": pl,
            "SHAPE": [dimension, 64],
            "DTYPE": pl.BF16,
            "ALLOCATION": allocation,
            "SPACE": pl.Mem.Vec,
            "VIEW": view,
        }

        source = specialize("_TestClass", [ctx])
        if dynamic:
            assert 'TILE_ROWS = pl.dynamic("TILE_ROWS")' in source
        assert not any(name in source for name in ("SHAPE", "DTYPE", "SPACE", "ALLOCATION", "VIEW"))
        helper = _specialized_function(ctx, source)
        shape = (
            [dimension.unwrap(), ir.ConstInt(64, DataType.INDEX, ir.Span.unknown())]
            if isinstance(dimension, pl.Scalar)
            else [dimension, 64]
        )
        ir.assert_structural_equal(
            helper.params[0].type,
            ir.TileType(
                shape,
                DataType.BF16,
                memref=allocation,
                tile_view=view,
                memory_space=ir.MemorySpace.Vec,
            ),
            enable_auto_mapping=True,
        )

    @pytest.mark.parametrize("ambient", [False, True])
    def test_scalar_tile_dimension_stays_in_function_scope(self, ambient):
        """A runtime Scalar dimension is a formal, not a frozen or module-level symbol."""
        ctx = _make_ctx(
            "keep_tile",
            """
                def keep_tile(ROWS: pl.Scalar[pl.INDEX], tile: pl.Tile[[ROWS, 64], pl.FP32]):
                    return tile
            """,
            func_type="inline",
            param_names=["ROWS", "tile"],
            auto_scope=False,
        )
        ctx.py_globals = {"pl": pl}
        if ambient:
            ctx.py_globals["ROWS"] = 8

        source = specialize("_TestClass", [ctx])
        assert 'ROWS = pl.dynamic("ROWS")' not in source
        assert "tile: pl.Tile[[ROWS, 64], pl.FP32]" in source
        helper = _specialized_function(ctx, source)
        tile_type = helper.params[1].type
        assert isinstance(tile_type, ir.TileType)
        assert ir.structural_equal(tile_type.shape[0], helper.params[0])

    @pytest.mark.parametrize("rows", [8, 16])
    def test_eager_annotation_does_not_freeze_helper_constexpr(self, rows):
        """Eager evaluation happens before calls and cannot select a constexpr variant."""
        ROWS = 99

        def keep_tile(tile: pl.Tile[[ROWS, 64], pl.FP32, pl.Mem.Vec], ROWS: pl.constexpr):
            return tile

        ctx = _inline_context(keep_tile, constexpr_values={"ROWS": str(rows)})
        func = _specialized_function(ctx)
        ir.assert_structural_equal(
            func.params[0].type,
            ir.TileType([rows, 64], DataType.FP32, memory_space=ir.MemorySpace.Vec),
        )

    @pytest.mark.parametrize("rows", [8, 16])
    @pytest.mark.parametrize("valid_rows", [3, 8])
    @pytest.mark.parametrize("direction", ["bare", "Out", "InOut"])
    def test_eager_factory_constants_coexist_with_constexpr_axes(self, rows, valid_rows, direction):
        """An erased factory column and metadata survive a per-call row binding."""
        ROWS = 8

        def factory(cols):
            dtype = pl.BF16
            allocation = pl.MemRef("tile_buffer", 64, 2048)
            # A full ambient view must retain eight valid rows when the
            # per-call physical shape later grows to sixteen.
            view = pl.TileView(valid_shape=[valid_rows, cols], stride=[cols, 1], start_offset=0)

            if direction == "Out":

                def keep_tile(
                    tile: pl.Out[pl.Tile[[ROWS, cols], dtype, allocation, pl.Mem.Vec, view]],
                    ROWS: pl.constexpr,
                ):
                    return tile

            elif direction == "InOut":

                def keep_tile(
                    tile: pl.InOut[pl.Tile[[ROWS, cols], dtype, allocation, pl.Mem.Vec, view]],
                    ROWS: pl.constexpr,
                ):
                    return tile

            else:

                def keep_tile(
                    tile: pl.Tile[[ROWS, cols], dtype, allocation, pl.Mem.Vec, view], ROWS: pl.constexpr
                ):
                    return tile

            return keep_tile

        helper = factory(64)
        assert not helper.__code__.co_freevars
        ctx = _inline_context(helper, constexpr_values={"ROWS": str(rows)})
        func = _specialized_function(ctx)
        original = helper.__annotations__["tile"]
        ir.assert_structural_equal(
            func.params[0].type,
            ir.TileType(
                [rows, 64],
                DataType.BF16,
                memref=original.memref,
                tile_view=original.tile_view,
                memory_space=original.memory_space,
            ),
            enable_auto_mapping=True,
        )

    @pytest.mark.parametrize("space", [pl.Mem.Vec, pl.Mem.Mat])
    @pytest.mark.parametrize("position", ["first", "last"])
    def test_eager_factory_shape_keeps_constexpr_memory_space(self, space, position):
        """Metadata source order cannot lose an erased factory shape or duplicate SPACE."""
        SPACE = pl.Mem.Vec

        def factory(rows):
            allocation = pl.MemRef("buffer", 0, 4096)
            view = pl.TileView(valid_shape=[3, 64])
            if position == "first":

                def keep_tile(
                    tile: pl.Tile[[rows, 64], pl.FP32, SPACE, allocation, view],
                    SPACE: pl.constexpr,
                ):
                    return tile

            else:

                def keep_tile(
                    tile: pl.Tile[[rows, 64], pl.FP32, view, allocation, SPACE],
                    SPACE: pl.constexpr,
                ):
                    return tile

            return keep_tile

        helper = factory(8)
        assert not helper.__code__.co_freevars
        ctx = _inline_context(helper, constexpr_values={"SPACE": f"pl.Mem.{space.name}"})
        func = _specialized_function(ctx)
        original = helper.__annotations__["tile"]
        ir.assert_structural_equal(
            func.params[0].type,
            ir.TileType(
                [8, 64],
                DataType.FP32,
                memref=original.memref,
                tile_view=original.tile_view,
                memory_space=space,
            ),
            enable_auto_mapping=True,
        )

    @pytest.mark.parametrize("symbol_name", ["NEW_VALID_COLS", "tile"])
    def test_metadata_override_declares_rebound_dynvar_alias(self, symbol_name):
        """The active override can contain symbols absent from the original eager value."""
        VALID = 3
        dimension = pl.dynamic("OLD_VALID_COLS")

        def keep_tile(
            tile: pl.Tile[[8, 64], pl.FP32, pl.TileView(valid_shape=[VALID, dimension])], VALID: pl.constexpr
        ):
            return tile

        raw = _annotation_function(keep_tile, keep_tile.__annotations__, {"dimension": dimension})
        raw.__globals__["dimension"] = pl.dynamic(symbol_name)
        ctx = _inline_context(raw, constexpr_values={"VALID": "4"})
        source = specialize("_TestClass", [ctx])
        assert f'{symbol_name} = pl.dynamic("{symbol_name}")' in source
        func = _specialized_function(ctx, source)
        tile_type = func.params[0].type
        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert isinstance(tile_type.tile_view.valid_shape[0], ir.ConstInt)
        assert tile_type.tile_view.valid_shape[0].value == 4
        assert isinstance(tile_type.tile_view.valid_shape[1], ir.Var)
        assert tile_type.tile_view.valid_shape[1].name_hint == symbol_name

    def test_metadata_override_rejects_kind_change(self):
        """A different kind cannot leave the original eager metadata silently attached."""
        USE_VIEW = False

        def factory(rows):
            def keep_tile(
                tile: pl.Tile[
                    [rows, 64], pl.FP32, pl.TileView(valid_shape=[3, 64]) if USE_VIEW else pl.Mem.Vec
                ],
                USE_VIEW: pl.constexpr,
            ):
                return tile

            return keep_tile

        ctx = _inline_context(factory(8), constexpr_values={"USE_VIEW": "True"})
        with pytest.raises(ValueError, match="must retain its original MemorySpace kind, got TileView"):
            specialize("_TestClass", [ctx])

    @pytest.mark.parametrize("metadata_name_collision", [False, True])
    def test_eager_mixed_axis_rejects_unrecoverable_factory_operand(self, metadata_name_collision):
        """The eager sum cannot reveal a lost operand after ROWS is rebound."""
        ROWS = 99

        def factory(padding):
            # An unrelated metadata DynVar with the erased operand's name
            # cannot stand in for that numeric factory value.
            valid_rows = pl.dynamic("padding") if metadata_name_collision else 3
            view = pl.TileView(valid_shape=[valid_rows, 64])

            def keep_tile(tile: pl.Tile[[ROWS + padding, 64], pl.FP32, view], ROWS: pl.constexpr):
                return tile

            return keep_tile

        helper = factory(3)
        assert not helper.__code__.co_freevars
        with pytest.raises(ValueError, match="Cannot recover.*padding.*ROWS \\+ padding"):
            _inline_context(helper, constexpr_values={"ROWS": "8"})

    def test_eager_factory_axis_keeps_runtime_scalar_in_local_scope(self):
        """Replacing an ambient generic axis must not declare its obsolete symbol."""
        ROWS = pl.dynamic("AMBIENT_ROWS")

        def factory(cols):
            def keep_tile(ROWS: pl.Scalar[pl.INDEX], tile: pl.Tile[[ROWS, cols], pl.FP32]):
                return tile

            return keep_tile

        helper = factory(64)
        assert not helper.__code__.co_freevars
        ctx = _inline_context(helper)
        source = specialize("_TestClass", [ctx])
        assert "AMBIENT_ROWS" not in source
        assert 'ROWS = pl.dynamic("ROWS")' not in source
        func = _specialized_function(ctx, source)
        tile_type = func.params[1].type
        assert isinstance(tile_type, ir.TileType)
        assert ir.structural_equal(tile_type.shape[0], func.params[0])
        assert isinstance(tile_type.shape[1], ir.ConstInt)
        assert tile_type.shape[1].value == 64

    @pytest.mark.parametrize(
        "rows,alias,shadowed",
        [(4, "pl", False), (4, "pl", True), (8, "pl", False), (8, "pl", True), (4, "p", True)],
    )
    def test_constexpr_folded_before_annotation_evaluation(self, rows, alias, shadowed):
        """Call bindings override ambient values and decode separately from original DSL aliases."""
        ctx = _make_ctx(
            "keep_tile",
            f"""
                def keep_tile(tile: {alias}.Tile[[ROWS * 2, 64], DTYPE, SPACE],
                              ROWS: {alias}.constexpr, DTYPE: {alias}.constexpr, SPACE: {alias}.constexpr):
                    return tile
            """,
            func_type="inline",
            param_names=["tile", "ROWS", "DTYPE", "SPACE"],
            auto_scope=False,
        )
        ctx.py_globals = {alias: pl}
        if shadowed:
            ctx.py_globals.update(ROWS=99, DTYPE=pl.FP32, SPACE=pl.Mem.Acc)
        if alias == "p":
            ctx.py_globals["pl"] = object()
        ctx.constexpr_values = {"ROWS": str(rows), "DTYPE": "pl.BF16", "SPACE": "pl.Mem.Vec"}

        source = specialize("_TestClass", [ctx])
        func = _specialized_function(ctx, source)
        assert len(func.params) == 1
        ir.assert_structural_equal(
            func.params[0].type,
            ir.TileType([rows * 2, 64], DataType.BF16, memory_space=ir.MemorySpace.Vec),
        )
        assert "ROWS" not in source and "DTYPE" not in source and "SPACE" not in source


class TestTensorLayoutAnnotation:
    """The annotation's layout slot must survive specialization.

    ``TensorMeta`` used to carry only shape and dtype, so ``pl.Tensor[[...],
    dtype, pl.NZ]`` regenerated as a plain two-slot annotation — the layout was
    dropped without a word and the parameter silently became ND.
    """

    def test_layout_str_renders_dsl_constant(self):
        assert layout_str(ir.TensorLayout.NZ) == "pl.NZ"
        assert layout_str(ir.TensorLayout.MX_A_ZZ) == "pl.MX_A_ZZ"

    def test_layout_str_omits_nd(self):
        """ND is what an absent slot already means — emitting it is noise."""
        assert layout_str(ir.TensorLayout.ND) is None

    def test_annotation_carries_layout(self):
        meta = TensorMeta((64, 128), DataType.FP16, layout=ir.TensorLayout.NZ)
        assert _build_tensor_annotation(meta, is_out=False) == "pl.Tensor[[64, 128], pl.FP16, pl.NZ]"

    def test_annotation_without_layout_unchanged(self):
        """The two-slot form must stay byte-identical — no trailing slot."""
        meta = TensorMeta((64, 128), DataType.FP16)
        assert _build_tensor_annotation(meta, is_out=False) == "pl.Tensor[[64, 128], pl.FP16]"

    def test_annotation_layout_survives_out_wrapper(self):
        meta = TensorMeta((64, 128), DataType.FP16, layout=ir.TensorLayout.NZ)
        assert _build_tensor_annotation(meta, is_out=True) == "pl.Out[pl.Tensor[[64, 128], pl.FP16, pl.NZ]]"

    def test_annotation_layout_with_dynamic_dim(self):
        """A DynDim shape and a layout coexist — both slots render."""
        meta = TensorMeta(
            (DynDim(name="M", literal="M", static_bound=64), 128),
            DataType.FP16,
            layout=ir.TensorLayout.NZ,
        )
        assert _build_tensor_annotation(meta, is_out=False) == "pl.Tensor[[M, 128], pl.FP16, pl.NZ]"

    def test_specialized_source_keeps_layout(self):
        """End to end: the generated @pl.function source carries the slot."""
        ctx = _make_ctx(
            source=textwrap.dedent(
                """
                def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                    return c
                """
            ),
            param_names=["a", "c"],
            tensor_meta={
                "a": TensorMeta((64, 128), DataType.FP16, layout=ir.TensorLayout.NZ),
                "c": TensorMeta((64, 128), DataType.FP16),
            },
        )
        out = specialize("_T", [ctx])
        assert "a: pl.Tensor[[64, 128], pl.FP16, pl.NZ]" in out
        # The un-annotated param keeps the plain two-slot form.
        assert "c: pl.Out[pl.Tensor[[64, 128], pl.FP16]]" in out


class TestRenamedGeneratedFunction:
    """``func_name`` names the generated method; ``source_func_name`` the ``def``.

    The two diverge when the JIT layer uniquifies a dep name (two distinct deps
    sharing a ``__name__``). Everything that reads the *original* source must
    still find the ``def`` under the Python name.
    """

    @staticmethod
    def _renamed_ctx():
        ctx = _make_ctx(
            func_name="helper__2",
            func_type="incore",
            source=textwrap.dedent(
                """
                def helper(src: pl.Tensor, dst: pl.Out[pl.Tensor]):
                    return dst
                """
            ),
            param_names=["src", "dst"],
            tensor_meta={
                "src": TensorMeta((64, 64), DataType.FP32),
                "dst": TensorMeta((64, 64), DataType.FP32),
            },
        )
        ctx.source_func_name = "helper"
        return ctx

    def test_source_def_name_falls_back_to_func_name(self):
        """A context that never renamed anything keeps one name for both roles."""
        ctx = _make_ctx(func_name="kernel")
        assert ctx.source_func_name is None
        assert ctx.source_def_name == "kernel"

    def test_generated_method_uses_the_renamed_name(self):
        out = specialize("_T", [self._renamed_ctx()])
        assert "def helper__2(self, src" in out
        assert "def helper(self" not in out

    def test_inline_deprecation_names_the_python_function(self):
        """The user's own name is what they can act on, not the generated one."""
        ctx = self._renamed_ctx()
        ctx.func_type = "inline"
        ctx.source = textwrap.dedent(
            """
            def helper(src: pl.Tensor, dst: pl.Out[pl.Tensor]):
                return dst
            """
        )
        with pytest.warns(DeprecationWarning, match="helper' uses pl.Out"):
            specialize("_T", [ctx])


class TestInlineTupleTileAnnotations:
    def test_tensor_leaf_keeps_explicit_full_view_after_constexpr_shape_changes(self):
        ROWS = 8
        view = pl.TensorView(layout=pl.TensorLayout.ND, valid_shape=[8, 64])

        def keep(
            pair: pl.Tuple[pl.Tensor[[ROWS, 64], pl.FP32, view], pl.Tile[[8, 64], pl.FP32]],
            ROWS: pl.constexpr,
        ):
            return pair[1]

        ctx = _inline_context(keep, constexpr_values={"ROWS": "16"})
        function = _specialized_function(ctx)
        pair = function.params[0].type
        assert isinstance(pair, ir.TupleType)
        tensor = pair.types[0]
        assert isinstance(tensor, ir.TensorType)
        assert isinstance(tensor.shape[0], ir.ConstInt) and tensor.shape[0].value == 16
        assert tensor.tensor_view is not None
        valid = tensor.tensor_view.valid_shape[0]
        assert isinstance(valid, ir.ConstInt) and valid.value == 8

    @pytest.mark.parametrize("spelling", ["dtype", "subscript", "constructor"])
    def test_mixed_scalar_leaf_obeys_its_own_constexpr_dtype(self, spelling):
        DTYPE = pl.FP32

        if spelling == "subscript":

            def keep(pair: pl.Tuple[pl.Tile[[8, 64], pl.FP32], pl.Scalar[DTYPE]], DTYPE: pl.constexpr):
                return pair[0]

        elif spelling == "dtype":

            def keep(
                pair: pl.Tuple[pl.Tile[[8, 64], pl.FP32], DTYPE],  # pyright: ignore[reportInvalidTypeForm]
                DTYPE: pl.constexpr,
            ):
                return pair[0]

        else:

            def keep(
                pair: pl.Tuple[pl.Tile[[8, 64], pl.FP32], pl.Scalar(DTYPE)],  # pyright: ignore[reportInvalidTypeForm]
                DTYPE: pl.constexpr,
            ):
                return pair[0]

        ctx = _inline_context(keep, constexpr_values={"DTYPE": "pl.INDEX"})
        function = _specialized_function(ctx)
        pair = function.params[0].type
        assert isinstance(pair, ir.TupleType)
        scalar = pair.types[1]
        assert isinstance(scalar, ir.ScalarType) and scalar.dtype == pl.INDEX

    def test_parameterized_postponed_tuple_fingerprint_tracks_nested_view(self):
        def keep(pair, ROWS):
            return pair[1]

        helper = _annotation_function(
            keep,
            {
                "pair": "pl.Tuple[pl.Scalar[pl.INDEX], pl.Tile[[ROWS, 64], pl.FP32, VIEW]]",
                "ROWS": "pl.constexpr",
            },
            {"VIEW": pl.TileView(valid_shape=[3, 64])},
        )
        before = tile_annotation_fingerprints(helper)
        helper.__globals__["VIEW"] = pl.TileView(valid_shape=[2, 64])
        assert tile_annotation_fingerprints(helper) != before

    @pytest.mark.parametrize("postponed", [False, True])
    def test_nested_tuple_preserves_mixed_types_and_dynamic_metadata(self, postponed):
        rows = pl.dynamic("TUPLE_TILE_ROWS")
        valid = pl.dynamic("TUPLE_VALID_ROWS")
        view = pl.TileView(valid_shape=[valid, 64], stride=[64, 1], start_offset=valid.unwrap())

        def keep(
            pair: pl.Tuple[
                pl.Scalar[pl.INDEX],
                pl.Tuple[pl.Tensor[[8, 64], pl.FP32], pl.Tile[[rows, 64], pl.FP32, pl.Mem.Vec, view]],
            ],
        ):
            return pair[1][1]

        helper = keep
        if postponed:
            helper = _annotation_function(
                keep,
                {
                    "pair": "TupleAlias[pl.Scalar[pl.INDEX], TupleAlias[pl.Tensor[[8, 64], pl.FP32], "
                    "pl.Tile[[rows, 64], pl.FP32, pl.Mem.Vec, view]]]"
                },
                {"rows": rows, "view": view, "TupleAlias": pl.Tuple},
            )
        ctx = _inline_context(helper)
        source = specialize("_TestClass", [ctx])
        assert 'TUPLE_TILE_ROWS = pl.dynamic("TUPLE_TILE_ROWS")' in source
        assert 'TUPLE_VALID_ROWS = pl.dynamic("TUPLE_VALID_ROWS")' in source
        function = _specialized_function(ctx, source)
        pair = function.params[0].type
        assert isinstance(pair, ir.TupleType)
        assert isinstance(pair.types[0], ir.ScalarType)
        nested = pair.types[1]
        assert isinstance(nested, ir.TupleType)
        assert isinstance(nested.types[0], ir.TensorType)
        tile = nested.types[1]
        assert isinstance(tile, ir.TileType)
        assert tile.tile_view is not None
        assert isinstance(tile.tile_view.valid_shape[0], ir.Var)
        assert tile.tile_view.valid_shape[0].name_hint == "TUPLE_VALID_ROWS"

    def test_eager_tuple_factory_keeps_independent_values_and_distinct_formals(self):
        FIRST, SECOND = 8, 16
        SPACE = pl.Mem.Vec

        def factory(cols):
            view = pl.TileView(valid_shape=[3, cols], stride=[cols, 1], start_offset=0)

            def keep(
                pair: pl.Tuple[
                    pl.Tile[[FIRST, cols], pl.FP32, SPACE, view],
                    pl.Tuple[pl.Tile[[SECOND, cols], pl.FP32, view, SPACE], pl.Scalar[pl.INDEX]],
                ],
                FIRST: pl.constexpr,
                SECOND: pl.constexpr,
                SPACE: pl.constexpr,
            ):
                return pair[1][0]

            return keep

        helper = factory(64)
        ctx = _inline_context(helper, constexpr_values={"FIRST": "16", "SECOND": "8", "SPACE": "pl.Mem.Vec"})
        function = _specialized_function(ctx)
        pair = function.params[0].type
        assert isinstance(pair, ir.TupleType)
        nested = pair.types[1]
        assert isinstance(nested, ir.TupleType)
        for tile, rows in ((pair.types[0], 16), (nested.types[0], 8)):
            assert isinstance(tile, ir.TileType)
            assert isinstance(tile.shape[0], ir.ConstInt) and tile.shape[0].value == rows
            assert isinstance(tile.shape[1], ir.ConstInt) and tile.shape[1].value == 64
            assert tile.tile_view is not None
            valid = tile.tile_view.valid_shape[0]
            assert isinstance(valid, ir.ConstInt) and valid.value == 3
            assert tile.memory_space == ir.MemorySpace.Vec
        assert tile_annotation_fingerprints(helper) != tile_annotation_fingerprints(factory(128))

    def test_postponed_tuple_alias_fingerprint_tracks_nested_contract_rebinding(self):
        def keep(pair):
            return pair[1]

        helper = _annotation_function(
            keep,
            {"pair": "CONTRACT"},
            {"CONTRACT": pl.Tuple[pl.Scalar[pl.INDEX], pl.Tile[[8, 64], pl.FP32]]},
        )
        before = tile_annotation_fingerprints(helper)
        helper.__globals__["CONTRACT"] = pl.Tuple[pl.Scalar[pl.INDEX], pl.Tile[[16, 64], pl.FP32]]
        assert tile_annotation_fingerprints(helper) != before

    def test_incomplete_tuple_tile_leaf_has_clear_error(self):
        def keep(pair: pl.Tuple[pl.Scalar[pl.INDEX], pl.Tile]):
            return pair[1]

        with pytest.raises(ValueError, match="Tuple annotation.*incomplete"):
            _inline_context(keep)

    def test_unsupported_tensor_subclass_in_tuple_is_not_rendered_as_plain_tensor(self):
        def keep(pair: pl.Tuple[pld.DistributedTensor[[8, 64], pl.FP32], pl.Tile[[8, 64], pl.FP32]]):
            return pair[1]

        with pytest.raises(ValueError, match="Tuple annotation.*unsupported type.*path \\(0,\\)"):
            _inline_context(keep)

    def test_tuple_tile_annotation_is_preserved_for_non_inline_function(self):
        ctx = _make_ctx(
            "keep",
            "def keep(pair: pl.Tuple[pl.Tile[[8, 64], pl.FP32]]):\n    return pair[0]",
            func_type="incore",
            param_names=["pair"],
            auto_scope=False,
        )
        ctx.py_globals = {"pl": pl}
        func = _specialized_function(ctx)
        assert func.func_type == ir.FunctionType.InCore
        ir.assert_structural_equal(func.params[0].type, ir.TupleType([ir.TileType([8, 64], DataType.FP32)]))


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_constexpr_branch_discards_incompatible_dead_tile_store(mode):
    """The inactive FP8 publication branch cannot store 512 bytes into FP4 output."""

    width = (256, 1024, 512)[mode]
    ctx = _make_ctx(
        "publish",
        f"""
        def publish(out: pl.Tensor[[8, {width}], pl.UINT8], MODE: pl.constexpr):
            if MODE == 2:
                wide = pl.tile.full([1, 512], dtype=pl.UINT8, value=0)
                pl.store(wide, [0, 0], out)
            elif MODE == 1:
                extra_wide = pl.tile.full([1, 1024], dtype=pl.UINT8, value=0)
                pl.store(extra_wide, [0, 0], out)
            else:
                narrow = pl.tile.full([1, 256], dtype=pl.UINT8, value=0)
                pl.store(narrow, [0, 0], out)
            return out
        """,
        func_type="inline",
        param_names=["out", "MODE"],
        tensor_meta={"out": TensorMeta((8, width), DataType.UINT8)},
        auto_scope=False,
    )
    ctx.py_globals = {"pl": pl}
    ctx.constexpr_values = {"MODE": str(mode)}
    source = specialize("_TestClass", [ctx])
    tree = ast.parse(source)
    # Constants must prune before type deduction, not only in later IR passes.
    assert not any(isinstance(node, ast.If) for node in ast.walk(tree))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert names & {"wide", "extra_wide", "narrow"} == {("narrow", "extra_wide", "wide")[mode]}
    _specialized_function(ctx, source)


def test_runtime_condition_remains_an_ir_branch():
    tree = ast.parse("if count == 2:\n    x = 1\nelse:\n    x = 3\n")
    transformed = BodyTransformer({}, set(), param_names=["count"]).visit(tree)
    assert any(isinstance(node, ast.If) for node in ast.walk(transformed))


@pytest.mark.parametrize(
    ("mode", "condition", "taken"),
    [
        (False, "MODE", False),
        (True, "MODE", True),
        (-1, "MODE", True),
        (-0.5, "MODE == -0.5", True),
        (-1, "+1 == MODE", False),
    ],
)
@pytest.mark.parametrize("target", ["x", "x: int"])
def test_constexpr_branch_rebinding_survives_specialization(mode, condition, taken, target):
    source = f"def select():\n    x = 1\n    if {condition}:\n        {target} = 2\n    return x\n"
    tree = BodyTransformer({}, set(), constexpr_values={"MODE": str(mode)}).visit(ast.parse(source))
    assert not any(isinstance(node, ast.If) for node in ast.walk(tree))
    namespace: dict[str, Any] = {}
    # Specialization emits source through unparse. Its intermediate rewritten
    # AST can retain child locations beyond newly synthesized parent ranges;
    # Python 3.11 rejects those ranges if compiled directly.
    exec(compile(ast.unparse(tree), "<specialized>", "exec"), namespace)
    assert namespace["select"]() == (2 if taken else 1)


@pytest.mark.parametrize("iterations", [0, 4])
def test_jit_while_graph_capture_becomes_taskid_carry(iterations):
    """Native SSA carries completion; full codegen rejects orchestration while."""

    @pl.jit
    def chain(c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]):
        with pl.manual_scope():
            with pl.at(level=pl.Level.CORE_GROUP) as ready:
                tile = pl.load(c, [0, 0], [128, 128])
                pl.store(pl.add(tile, 1.0), [0, 0], c)
            i = 0
            while i < iterations:
                with pl.graph("layer", deps=[ready]) as ready:
                    with pl.at(level=pl.Level.CORE_GROUP):
                        tile = pl.load(c, [0, 0], [128, 128])
                        pl.store(pl.add(tile, 1.0), [0, 0], c)
                i = i + 1
            with pl.at(level=pl.Level.CORE_GROUP, deps=[ready]):
                tile = pl.load(c, [0, 0], [128, 128])
                pl.store(pl.add(tile, 1.0), [0, 0], c)
        return c

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        program = passes.outline_incore_scopes()(
            passes.outline_graph_scopes()(passes.convert_to_ssa()(chain.specialize()))
        )
    loops = []
    submits = []

    class Collect(ir.IRVisitor):
        def visit_while_stmt(self, op):
            loops.append(op)
            super().visit_while_stmt(op)

        def visit_submit(self, op):
            submits.append(op)
            super().visit_submit(op)

    main = program.get_function("chain")
    assert main is not None
    Collect().visit_function(main)
    assert len(loops) == 1
    loop = loops[0]
    task_carries = [
        (i, arg)
        for i, arg in enumerate(loop.iter_args)
        if isinstance(arg.type, ir.ScalarType) and arg.type.dtype == DataType.TASK_ID
    ]
    assert len(task_carries) == 1
    index, carry = task_carries[0]
    graph = next(f for f in program.functions.values() if f.func_type == ir.FunctionType.Graph)
    launch = next(s for s in submits if isinstance(s.op, ir.GlobalVar) and s.op.name == graph.name)
    assert launch.deps[0].unique_id == carry.unique_id
    assert submits[-1].deps[0].unique_id == loop.return_vars[index].unique_id
    # Native SSA supports the carry, but the orchestration backend has no
    # while emitter. Full compilation must reject it instead of dropping it.
    with (
        passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH),
        pytest.raises(PartialCodegenError, match="While loops are not supported in orchestration"),
    ):
        chain.compile(config=RunConfig(backend_type=BackendType.Ascend910))


@pytest.mark.parametrize("setup_iterations,graph_iterations", [(0, 4), (1, 0), (1, 4)])
def test_jit_independent_capture_compiles_across_empty_sibling_loops(setup_iterations, graph_iterations):
    """A fresh nonempty capture must not require a previous loop to execute."""

    @pl.jit
    def chain(c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]):
        with pl.manual_scope():
            with pl.at(level=pl.Level.CORE_GROUP) as seed:
                pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
            with pl.at(level=pl.Level.CORE_GROUP, deps=[seed]) as latest:
                pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
            for setup in pl.range(setup_iterations):
                ready = seed
                ready = latest
            for invocation in pl.range(graph_iterations):
                # Intentional: an empty loop must preserve the preceding TaskId binding.
                with pl.graph("layer") as ready:  # noqa: F811
                    with pl.at(level=pl.Level.CORE_GROUP):
                        pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
            with pl.at(level=pl.Level.CORE_GROUP, deps=[ready]):
                pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
        return c

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = chain.compile(config=RunConfig(backend_type=BackendType.Ascend910))
    assert compiled.program is not None


@pytest.mark.parametrize("enabled", [False, True])
def test_jit_conditional_capture_preserves_sibling_loop_carry(enabled):
    """Conditionally captured completions carry across the surrounding loop."""

    @pl.jit
    def chain(c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]):
        with pl.manual_scope():
            with pl.at(level=pl.Level.CORE_GROUP) as seed:
                pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
            for setup in pl.range(1):
                if enabled:
                    ready = seed
                else:
                    ready = seed
            for invocation in pl.range(4):
                if enabled:
                    with pl.graph("layer", deps=[ready]) as ready:
                        with pl.at(level=pl.Level.CORE_GROUP):
                            pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
            with pl.at(level=pl.Level.CORE_GROUP, deps=[ready]):
                pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
        return c

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        program = passes.outline_incore_scopes()(
            passes.outline_graph_scopes()(passes.convert_to_ssa()(chain.specialize()))
        )
        compiled = chain.compile(config=RunConfig(backend_type=BackendType.Ascend910))
    assert compiled.program is not None
    if enabled:
        loops = []
        launches = []

        class Collect(ir.IRVisitor):
            def visit_for_stmt(self, op):
                loops.append(op)
                super().visit_for_stmt(op)

            def visit_submit(self, op):
                if isinstance(op.op, ir.GlobalVar) and op.op.name == "layer":
                    launches.append(op)
                super().visit_submit(op)

        main = program.get_function("chain")
        assert main is not None
        Collect().visit_function(main)
        assert len(loops) == 2 and len(launches) == 1
        carries = [
            arg
            for arg in loops[1].iter_args
            if isinstance(arg.type, ir.ScalarType) and arg.type.dtype == DataType.TASK_ID
        ]
        assert len(carries) == 1
        assert launches[0].deps[0].unique_id == carries[0].unique_id


@pytest.mark.parametrize("enabled", [False, True])
def test_jit_branch_merge_preserves_last_capture_binding(enabled):
    """A branch-local rebind must export latest, rather than its first capture."""

    @pl.jit
    def chain(c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]):
        with pl.manual_scope():
            with pl.at(level=pl.Level.CORE_GROUP) as seed:
                pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
            if enabled:
                with pl.at(level=pl.Level.CORE_GROUP, deps=[seed]) as ready:
                    pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
                with pl.at(level=pl.Level.CORE_GROUP, deps=[ready]) as latest:
                    pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
                ready = latest
            else:
                ready = seed
            with pl.at(level=pl.Level.CORE_GROUP, deps=[ready]):
                pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
        return c

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        program = passes.convert_to_ssa()(chain.specialize())
        compiled = chain.compile(config=RunConfig(backend_type=BackendType.Ascend910))
    assert compiled.program is not None
    branches = []
    scopes = []
    aliases = {}
    yields = []

    class Collect(ir.IRVisitor):
        def visit_if_stmt(self, op):
            branches.append(op)
            super().visit_if_stmt(op)

        def visit_in_core_scope_stmt(self, op):
            if "task_id_var" in op.attrs or "manual_dep_edges" in op.attrs:
                scopes.append(op)
            super().visit_in_core_scope_stmt(op)

        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Var):
                aliases[op.var.unique_id] = op.value
            super().visit_assign_stmt(op)

        def visit_yield_stmt(self, op):
            yields.append(op)
            super().visit_yield_stmt(op)

    main = program.get_function("chain")
    assert main is not None
    Collect().visit_function(main)
    assert len(branches) == 1
    branch = branches[0]
    assert scopes, python_print(program)
    consumer_dep = scopes[-1].attrs["manual_dep_edges"][0]
    index = next(i for i, value in enumerate(branch.return_vars) if value.unique_id == consumer_dep.unique_id)
    assert len(scopes) == 4
    expected_origins = (
        (branch.then_body, scopes[2].attrs["task_id_var"]),
        (branch.else_body, scopes[0].attrs["task_id_var"]),
    )
    # Resolve scalar copies at both exits: true exports latest, false exports seed.
    for body, origin in expected_origins:
        assert body is not None
        yields.clear()
        Collect().visit_stmt(body)
        value = yields[-1].value[index]
        while value.unique_id in aliases:
            value = aliases[value.unique_id]
        assert value.unique_id == origin.unique_id


def test_jit_recorded_local_capture_name_can_be_reused():
    """An independent capture must not read a previous Definition's local ID.

    Its completion orders a consumer within each iteration; it needs no entry
    TaskId from the previous Definition or a value exported from an empty loop.
    """

    @pl.jit
    def chain(
        first: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ):
        with pl.manual_scope():
            with pl.graph("first"):
                with pl.at(level=pl.Level.CORE_GROUP) as ready:
                    pl.store(pl.load(first, [0, 0], [128, 128]), [0, 0], first)
            for invocation in pl.range(4):
                with pl.graph("second") as ready:
                    with pl.at(level=pl.Level.CORE_GROUP):
                        pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
                with pl.at(level=pl.Level.CORE_GROUP, deps=[ready]):
                    pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
        return c

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = chain.compile(config=RunConfig(backend_type=BackendType.Ascend910))
    assert compiled.program is not None


def test_jit_plain_recorded_local_assignment_preserves_caller_task_id():
    @pl.jit
    def chain(c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]):
        with pl.manual_scope():
            seed = pl.system.task_dummy(deps=[])
            latest = pl.system.task_dummy(deps=[seed])
            for _ in pl.range(1):
                ready = seed
                ready = latest
            with pl.graph("recording") as done:
                inner = pl.system.task_dummy(deps=[])
                ready = inner
                with pl.at(level=pl.Level.CORE_GROUP, deps=[ready]):
                    c = pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
            with pl.at(level=pl.Level.CORE_GROUP, deps=[ready, done]):
                c = pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
        return c

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = chain.compile(config=RunConfig(backend_type=BackendType.Ascend910))
    assert compiled.program is not None


def test_jit_graph_preserves_independent_kernel_tile_rebinding():
    @pl.jit
    def chain(
        first: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        recorded: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
    ):
        for _ in pl.range(1):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(first, [0, 0], [128, 128])
                pl.store(tile, [0, 0], first)
        with pl.graph("recording"):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(recorded, [0, 0], [64, 64])
                pl.store(tile, [0, 0], recorded)
        return recorded

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = chain.compile(config=RunConfig(backend_type=BackendType.Ascend910))
    assert compiled.program is not None


def test_jit_graph_host_alias_keeps_independent_kernel_tensor_rebinding():
    @pl.jit
    def chain(
        first: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        recorded: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
    ):
        for _ in pl.range(1):
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.store(pl.load(first, [0, 0], [128, 128]), [0, 0], first)
        with pl.graph("recording"):
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.store(pl.load(recorded, [0, 0], [64, 64]), [0, 0], recorded)
            out = recorded
        return out

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = chain.compile(config=RunConfig(backend_type=BackendType.Ascend910))
    assert compiled.program is not None


@pytest.mark.parametrize("write_first", [False, True])
def test_jit_graph_tensor_rebinding_with_same_name_task_capture_is_exported(write_first):
    capture = (
        "with pl.at(level=pl.Level.CORE_GROUP) as selected:\n"
        "    pl.store(pl.load(b, [0, 0], [128, 128]), [0, 0], b)"
    )
    lines = ["selected = b", capture] if write_first else [capture, "selected = b"]
    source = textwrap.dedent("""
        def chain(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                selected = pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
            with pl.graph("recording"):
                CAPTURE
            return selected
    """).replace("        CAPTURE", textwrap.indent("\n".join(lines), "        "))
    meta = TensorMeta((128, 128), DataType.FP32)
    ctx = _make_ctx("chain", source, param_names=["a", "b"], tensor_meta={"a": meta, "b": meta})

    function = pl.parse_program(specialize("Before", [ctx])).get_function("chain")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    graph = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.GraphScopeStmt))
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    (selected,) = (
        var
        for var in ir.collect_def_vars(graph.body)
        if var.name_hint == "selected_v1" and isinstance(var.type, ir.TensorType)
    )
    assert value.same_as(selected)


@pytest.mark.parametrize("repeat_capture", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_jit_graph_capture_keeps_tensor_written_in_device_body(repeat_capture, overwrite):
    second_capture = (
        "        with pl.at(level=pl.Level.CORE_GROUP, deps=[selected]) as selected:\n"
        "            pl.store(pl.load(b, [0, 0], [128, 128]), [0, 0], b)"
        if repeat_capture
        else ""
    )
    if overwrite:
        second_capture += "\n        selected = c"
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor], c: pl.InOut[pl.Tensor]):
            for i in pl.range(1):
                with pl.at(level=pl.Level.CORE_GROUP):
                    selected = pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
            with pl.graph("recording"):
                with pl.at(level=pl.Level.CORE_GROUP) as selected:
                    selected = pl.store(pl.load(b, [0, 0], [128, 128]), [0, 0], b)
            SECOND_CAPTURE
            return selected
    """).replace("    SECOND_CAPTURE", second_capture)
    meta = TensorMeta((128, 128), DataType.FP32)
    ctx = _make_ctx("f", source, param_names=["a", "b", "c"], tensor_meta={"a": meta, "b": meta, "c": meta})
    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    graph = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.GraphScopeStmt))
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    assert isinstance(graph.body, ir.SeqStmts)
    device_scope = next(stmt for stmt in graph.body.stmts if isinstance(stmt, ir.InCoreScopeStmt))
    (selected,) = (
        var
        for var in ir.collect_def_vars(device_scope.body)
        if var.name_hint == "selected_v1" and isinstance(var.type, ir.TensorType)
    )
    assert value.same_as(selected)
    if overwrite:
        assignment = graph.body.stmts[-1]
        assert isinstance(assignment, ir.AssignStmt)
        assert assignment.var.same_as(selected)
        assert assignment.value.same_as(function.params[2])


def test_jit_graph_capture_shape_uses_tensor_written_in_device_body():
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
            for i in pl.range(1):
                with pl.at(level=pl.Level.CORE_GROUP):
                    selected = a
                    pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
            with pl.graph("recording"):
                with pl.at(level=pl.Level.CORE_GROUP) as selected:
                    selected = b
                    pl.store(pl.load(b, [0, 0], [64, 128]), [0, 0], b)
            for i in pl.range(selected.shape[0]):
                with pl.at(level=pl.Level.CORE_GROUP):
                    pl.store(pl.load(selected, [i, 0], [1, 128]), [i, 0], selected)
            return selected
    """)
    ctx = _make_ctx(
        "f",
        source,
        param_names=["a", "b"],
        tensor_meta={"a": TensorMeta((128, 128), DataType.FP32), "b": TensorMeta((64, 128), DataType.FP32)},
    )
    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    _, loop = (stmt for stmt in function.body.stmts if isinstance(stmt, ir.ForStmt))
    assert isinstance(loop.stop, ir.ConstInt) and loop.stop.value == 64


def test_jit_graph_capture_matches_restored_tensor_alias():
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP):
                    selected = pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
                with pl.graph("first"):
                    selected = pl.system.task_dummy(deps=[])
                    with pl.at(level=pl.Level.CORE_GROUP, deps=[selected]):
                        pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
                with pl.graph("second"):
                    with pl.at(level=pl.Level.CORE_GROUP) as selected:
                        selected = pl.store(pl.load(b, [0, 0], [128, 128]), [0, 0], b)
                    with pl.at(level=pl.Level.CORE_GROUP, deps=[selected]):
                        pl.store(pl.load(b, [0, 0], [128, 128]), [0, 0], b)
            return selected
    """)
    meta = TensorMeta((128, 128), DataType.FP32)
    ctx = _make_ctx("f", source, param_names=["a", "b"], tensor_meta={"a": meta, "b": meta})
    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    runtime = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.RuntimeScopeStmt))
    assert isinstance(runtime.body, ir.SeqStmts)
    _, graph = (stmt for stmt in runtime.body.stmts if isinstance(stmt, ir.GraphScopeStmt))
    assert isinstance(graph.body, ir.SeqStmts)
    producer, consumer = (stmt for stmt in graph.body.stmts if isinstance(stmt, ir.InCoreScopeStmt))
    (dependency,) = consumer.attrs["manual_dep_edges"]
    assert dependency.same_as(producer.attrs["task_id_var"])
    (selected,) = (var for var in ir.collect_def_vars(producer.body) if isinstance(var.type, ir.TensorType))
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    assert value.same_as(selected)


@pytest.mark.parametrize("nested", [False, True])
def test_jit_graph_capture_does_not_reuse_incoming_tensor_alias_for_assignment(nested):
    capture = textwrap.dedent("""
        with pl.at(level=pl.Level.CORE_GROUP) as selected:
            pl.store(pl.load(b, [0, 0], [128, 128]), [0, 0], b)
    """).strip("\n")
    if nested:
        capture = "for i in pl.range(1):\n" + textwrap.indent(capture, "    ")
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
            selected = a
            selected = b
            with pl.graph("recording"):
                CAPTURE
                selected = a
            return selected
    """).replace("        CAPTURE", textwrap.indent(capture, "        "))
    meta = TensorMeta((128, 128), DataType.FP32)
    ctx = _make_ctx("f", source, param_names=["a", "b"], tensor_meta={"a": meta, "b": meta})

    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    graph = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.GraphScopeStmt))
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    (selected,) = (var for var in ir.collect_def_vars(graph.body) if isinstance(var.type, ir.TensorType))
    assert value.same_as(selected)


@pytest.mark.parametrize("capture_kind", ["dispatch", "graph"])
@pytest.mark.parametrize("write_scope", ["device", "host", "loop"])
def test_jit_dispatch_capture_then_tensor_assignment(capture_kind, write_scope):
    capture = textwrap.dedent("""
        with pl.at(level=pl.Level.CORE_GROUP) as selected:
            pl.store(pl.load(b, [0, 0], [128, 128]), [0, 0], b)
    """).strip("\n")
    if capture_kind == "graph":
        capture = 'with pl.graph("recording") as selected:\n' + textwrap.indent(
            capture.replace(" as selected", ""), "    "
        )
    second_dispatch = textwrap.dedent("""
        with pl.at(level=pl.Level.CORE_GROUP, deps=[selected]):
            pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
    """).strip("\n")
    if write_scope == "device":
        second_dispatch = second_dispatch.replace("    pl.store", "    selected = pl.store")
    else:
        write = "selected = a" if write_scope == "host" else "for i in pl.range(1):\n    selected = a"
        second_dispatch += "\n" + write
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
            selected = a
            selected = b
            with pl.manual_scope():
                CAPTURE
                SECOND_DISPATCH
            return selected
    """)
    source = source.replace("        CAPTURE", textwrap.indent(capture, "        ")).replace(
        "        SECOND_DISPATCH", textwrap.indent(second_dispatch, "        ")
    )
    meta = TensorMeta((128, 128), DataType.FP32)
    ctx = _make_ctx("f", source, param_names=["a", "b"], tensor_meta={"a": meta, "b": meta})

    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    (original,) = (
        stmt.var
        for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint == "selected_v1"
    )
    assert value.same_as(original)


@pytest.mark.parametrize("capture_kind", ["dispatch", "graph"])
def test_jit_dispatch_capture_then_scalar_assignment(capture_kind):
    capture = textwrap.dedent("""
        with pl.at(level=pl.Level.CORE_GROUP) as index:
            pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
    """).strip("\n")
    if capture_kind == "graph":
        capture = 'with pl.graph("recording") as index:\n' + textwrap.indent(
            capture.replace(" as index", ""), "    "
        )
    source = textwrap.dedent("""
        def f(c: pl.InOut[pl.Tensor]):
            index = 1
            index = 2
            with pl.manual_scope():
                CAPTURE
                pl.system.task_dummy(deps=[index])
                index = 3
            return index
    """).replace("        CAPTURE", textwrap.indent(capture, "        "))
    ctx = _make_ctx("f", source, param_names=["c"], tensor_meta={"c": TensorMeta((128, 128), DataType.FP32)})

    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    (original,) = (
        stmt.var
        for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint == "index_v1"
    )
    assert value.same_as(original)


@pytest.mark.parametrize("selected", [True, False])
@pytest.mark.parametrize("inactive_capture", [False, True])
@pytest.mark.parametrize("condition", ["enabled", "mode == 0", "not (mode > 0 or mode < 0)"])
def test_jit_constexpr_selected_branch_preserves_folded_shape(selected, inactive_capture, condition):
    fold = "rows = a.shape[0]"
    inactive = (
        "with pl.at(level=pl.Level.CORE_GROUP) as rows:\n"
        "    pl.store(pl.load(a, [0, 0], [64, 128]), [0, 0], a)"
        if inactive_capture
        else "pass"
    )
    body, orelse = (fold, inactive) if selected else (inactive, fold)
    source = (
        textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], enabled: pl.constexpr, mode: pl.constexpr):
            if CONDITION:
                BODY
            else:
                ORELSE
            return pl.create_tensor([rows, 128], dtype=pl.FP32)
    """)
        .replace("        BODY", textwrap.indent(body, "        "))
        .replace("        ORELSE", textwrap.indent(orelse, "        "))
    )
    source = source.replace("CONDITION", condition)
    ctx = _make_ctx("f", source, param_names=["a"], tensor_meta={"a": TensorMeta((64, 128), DataType.FP32)})
    ctx.constexpr_values = {"enabled": str(selected), "mode": "0" if selected else "1"}
    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    body = function.body
    returned = body.stmts[-1] if isinstance(body, ir.SeqStmts) else body
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    assert isinstance(value.type, ir.TensorType)
    assert isinstance(value.type.shape[0], ir.ConstInt) and value.type.shape[0].value == 64


@pytest.mark.parametrize("enabled", [False, True])
def test_jit_graph_capture_preserves_later_shape_fold(enabled):
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor], enabled: pl.constexpr):
            rows = a.shape[0]
            with pl.graph("recording"):
                if enabled:
                    with pl.at(level=pl.Level.CORE_GROUP) as rows:
                        pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
                rows = b.shape[0]
                with pl.at(level=pl.Level.CORE_GROUP):
                    pl.store(pl.load(b, [0, 0], [64, 128]), [0, 0], b)
            for i in pl.range(rows):
                with pl.at(level=pl.Level.CORE_GROUP):
                    pl.store(pl.load(b, [i, 0], [1, 128]), [i, 0], b)
            return b
    """)
    ctx = _make_ctx(
        "f",
        source,
        param_names=["a", "b"],
        tensor_meta={"a": TensorMeta((128, 128), DataType.FP32), "b": TensorMeta((64, 128), DataType.FP32)},
    )
    ctx.constexpr_values = {"enabled": str(enabled)}
    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    loop = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.ForStmt))
    assert isinstance(loop.stop, ir.ConstInt) and loop.stop.value == 64


def test_jit_scalar_after_graph_shape_capture_keeps_loop_carry():
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
            with pl.graph("recording"):
                row = a.shape[0]
                with pl.at(level=pl.Level.CORE_GROUP) as row:
                    pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
                row = b.shape[0]
                row = 0
                for i in pl.range(2):
                    row = row + 1
                if row == 2:
                    row = row + 1
            return a
    """)
    ctx = _make_ctx(
        "f",
        source,
        param_names=["a", "b"],
        tensor_meta={"a": TensorMeta((128, 128), DataType.FP32), "b": TensorMeta((64, 128), DataType.FP32)},
    )
    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    graph = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.GraphScopeStmt))
    assert isinstance(graph.body, ir.SeqStmts)
    initial = next(
        stmt
        for stmt in graph.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.ConstInt)
    )
    loop = next(stmt for stmt in graph.body.stmts if isinstance(stmt, ir.ForStmt))
    (updated,) = ir.collect_def_vars(loop.body)
    assert updated.same_as(initial.var)


def test_jit_graph_capture_does_not_reuse_sibling_branch_shape():
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], flag: pl.Scalar[pl.BOOL]):
            if flag:
                rows = a.shape[0]
            else:
                with pl.graph("recording"):
                    with pl.at(level=pl.Level.CORE_GROUP) as rows:
                        pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
            return a
    """)
    ctx = _make_ctx(
        "f", source, param_names=["a", "flag"], tensor_meta={"a": TensorMeta((128, 128), DataType.FP32)}
    )
    assert pl.parse_program(specialize("Before", [ctx])).get_function("f") is not None


@pytest.mark.parametrize("capture_kind", ["dispatch", "graph"])
def test_jit_dispatch_capture_then_folded_scalar_assignment(capture_kind):
    capture = textwrap.dedent("""
        with pl.at(level=pl.Level.CORE_GROUP) as rows:
            pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
    """).strip("\n")
    if capture_kind == "graph":
        capture = 'with pl.graph("recording") as rows:\n' + textwrap.indent(
            capture.replace(" as rows", ""), "    "
        )
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
            rows = a.shape[0]
            with pl.manual_scope():
                CAPTURE
                pl.system.task_dummy(deps=[rows])
                rows = b.shape[0]
            return rows
    """).replace("        CAPTURE", textwrap.indent(capture, "        "))
    ctx = _make_ctx(
        "f",
        source,
        param_names=["a", "b"],
        tensor_meta={"a": TensorMeta((128, 128), DataType.FP32), "b": TensorMeta((64, 128), DataType.FP32)},
    )

    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    assert isinstance(value.type, ir.ScalarType)
    assert value.type.dtype == DataType.INDEX


def test_jit_dispatch_capture_then_assignment_after_graph_restores_constant():
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor], b: pl.InOut[pl.Tensor]):
            rows = a.shape[0]
            with pl.manual_scope():
                with pl.graph("recording"):
                    rows = pl.system.task_dummy(deps=[])
                    with pl.at(level=pl.Level.CORE_GROUP, deps=[rows]):
                        pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
                with pl.at(level=pl.Level.CORE_GROUP) as rows:
                    pl.store(pl.load(a, [0, 0], [128, 128]), [0, 0], a)
                with pl.at(level=pl.Level.CORE_GROUP, deps=[rows]):
                    pl.store(pl.load(b, [0, 0], [64, 128]), [0, 0], b)
                rows = b.shape[0]
            return rows
    """)
    ctx = _make_ctx(
        "f",
        source,
        param_names=["a", "b"],
        tensor_meta={"a": TensorMeta((128, 128), DataType.FP32), "b": TensorMeta((64, 128), DataType.FP32)},
    )
    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    runtime = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.RuntimeScopeStmt))
    assert isinstance(runtime.body, ir.SeqStmts)
    assignment = runtime.body.stmts[-1]
    returned = function.body.stmts[-1]
    assert isinstance(assignment, ir.AssignStmt)
    assert isinstance(assignment.value, ir.ConstInt) and assignment.value.value == 64
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    assert value.same_as(assignment.var)


@pytest.mark.parametrize("recorded", [False, True])
def test_jit_dispatch_capture_then_task_id_array_assignment(recorded):
    body = textwrap.dedent("""
        with pl.at(level=pl.Level.CORE_GROUP) as ids:
            pl.store(pl.load(a, [0, 0], [64, 128]), [0, 0], a)
        with pl.at(level=pl.Level.CORE_GROUP, deps=[ids]):
            pl.store(pl.load(a, [0, 0], [64, 128]), [0, 0], a)
        ids = pl.array.create(1, pl.TASK_ID)
        with pl.at(level=pl.Level.CORE_GROUP, deps=[ids]):
            pl.store(pl.load(a, [0, 0], [64, 128]), [0, 0], a)
    """).strip("\n")
    if recorded:
        body = 'with pl.graph("recording") as done:\n' + textwrap.indent(body, "    ")
        body += "\nwith pl.at(level=pl.Level.CORE_GROUP, deps=[ids, done]):\n"
        body += "    pl.store(pl.load(a, [0, 0], [64, 128]), [0, 0], a)"
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor]):
            ids = pl.array.create(1, pl.TASK_ID)
            ids = pl.array.create(1, pl.TASK_ID)
            with pl.manual_scope():
                BODY
            return a
    """).replace("        BODY", textwrap.indent(body, "        "))
    ctx = _make_ctx("f", source, param_names=["a"], tensor_meta={"a": TensorMeta((64, 128), DataType.FP32)})
    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    (original,) = (
        stmt.var
        for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint == "ids_v1"
    )
    runtime = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.RuntimeScopeStmt))
    assert isinstance(runtime.body, ir.SeqStmts)
    scope_body = runtime.body
    if recorded:
        graph = next(stmt for stmt in runtime.body.stmts if isinstance(stmt, ir.GraphScopeStmt))
        assert isinstance(graph.body, ir.SeqStmts)
        scope_body = graph.body
        consumer = runtime.body.stmts[-1]
        assert isinstance(consumer, ir.InCoreScopeStmt)
        outer_ids, done = consumer.attrs["manual_dep_edges"]
        assert outer_ids.same_as(original)
        assert done.same_as(graph.attrs["task_id_var"])
    (recreated,) = (
        stmt.var
        for stmt in scope_body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.ArrayType)
    )
    assert recreated.same_as(original) is not recorded
    consumer = scope_body.stmts[-1]
    assert isinstance(consumer, ir.InCoreScopeStmt)
    (dep,) = consumer.attrs["manual_dep_edges"]
    assert dep.same_as(recreated)


def test_jit_dispatch_capture_then_rebinding_keeps_parameter_dim_immutable():
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor]):
            rows = 0
            rows = pl.tensor.dim(a, 0)
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP) as rows:
                    pl.store(pl.load(a, [0, 0], [1, 128]), [0, 0], a)
                rows = 1
            return rows
    """)
    ctx = _make_ctx(
        "f",
        source,
        param_names=["a"],
        tensor_meta={"a": TensorMeta((DynDim("M", "M", 128), 128), DataType.FP32)},
    )

    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    (value,) = returned.value
    (parameter,) = function.params
    assert isinstance(parameter.type, ir.TensorType)
    assert isinstance(value, ir.Var)
    assert value.name_hint == "rows_v1"
    assert not value.same_as(parameter.type.shape[0])


@pytest.mark.parametrize("repeat_capture", [False, True])
def test_jit_graph_capture_preserves_outer_task_id_over_older_scalar(repeat_capture):
    capture = textwrap.dedent("""
        with pl.at(level=pl.Level.CORE_GROUP) as ready:
            pl.store(pl.load(a, [0, 0], [64, 128]), [0, 0], a)
    """).strip("\n")
    captures = [capture, capture] if repeat_capture else [capture]
    source = textwrap.dedent("""
        def f(a: pl.InOut[pl.Tensor]):
            ready = 0
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP) as ready:
                    pl.store(pl.load(a, [0, 0], [64, 128]), [0, 0], a)
                with pl.graph("recording") as done:
                    CAPTURES
                with pl.at(level=pl.Level.CORE_GROUP, deps=[ready, done]):
                    pl.store(pl.load(a, [0, 0], [64, 128]), [0, 0], a)
                ready = 1
            return a
    """).replace("            CAPTURES", textwrap.indent("\n".join(captures), "            "))
    ctx = _make_ctx("f", source, param_names=["a"], tensor_meta={"a": TensorMeta((64, 128), DataType.FP32)})
    function = pl.parse_program(specialize("Before", [ctx])).get_function("f")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    runtime = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.RuntimeScopeStmt))
    assert isinstance(runtime.body, ir.SeqStmts)
    producer, consumer = (stmt for stmt in runtime.body.stmts if isinstance(stmt, ir.InCoreScopeStmt))
    graph = next(stmt for stmt in runtime.body.stmts if isinstance(stmt, ir.GraphScopeStmt))
    ready, done = consumer.attrs["manual_dep_edges"]
    assert ready.same_as(producer.attrs["task_id_var"])
    assert done.same_as(graph.attrs["task_id_var"])

    initial = function.body.stmts[0]
    reassignment = runtime.body.stmts[-1]
    assert isinstance(initial, ir.AssignStmt)
    assert isinstance(reassignment, ir.AssignStmt)
    assert reassignment.var.same_as(initial.var)


def test_jit_explicit_recorded_local_dependency_remains_rejected():
    """Fresh capture names must not erase an actual cross-recording TaskId read."""

    @pl.jit
    def chain(
        first: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ):
        with pl.manual_scope():
            with pl.graph("first"):
                with pl.at(level=pl.Level.CORE_GROUP) as ready:
                    pl.store(pl.load(first, [0, 0], [128, 128]), [0, 0], first)
            with pl.graph("second", deps=[ready]) as done:
                with pl.at(level=pl.Level.CORE_GROUP):
                    pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
            with pl.at(level=pl.Level.CORE_GROUP, deps=[done]):
                pl.store(pl.load(c, [0, 0], [128, 128]), [0, 0], c)
        return c

    with (
        passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH),
        pytest.raises(UndefinedVariableError, match="Undefined variable 'ready'"),
    ):
        chain.compile(config=RunConfig(backend_type=BackendType.Ascend910))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
