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
import textwrap
import warnings

import pypto.language as pl
import pytest
from pypto.jit.specializer import (
    SpecializeContext,
    Specializer,
    TensorMeta,
    _BodyTransformer,
    _classify_params,
    _collect_dynamic_dims,
    _collect_dynvar_names,
    _infer_return_type,
    specialize,
)
from pypto.pypto_core import DataType, ir

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
    scalar_values: dict | None = None,
    scalar_dtypes: dict | None = None,
    dynamic_dims: set | None = None,
    dep_names: list[str] | None = None,
) -> SpecializeContext:
    return SpecializeContext(
        func_name=func_name,
        source=source,
        func_type=func_type,
        level=None,
        param_names=param_names or [],
        tensor_meta=tensor_meta or {},
        scalar_values=scalar_values or {},
        scalar_dtypes=scalar_dtypes or {},
        dynamic_dims=dynamic_dims or set(),
        dep_names=dep_names or [],
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
        out_params, tensor_params, scalar_strs = _classify_params(func_def)
        assert "a" in tensor_params
        assert "b" in tensor_params
        assert len(out_params) == 0

    def test_out_param(self):
        src = textwrap.dedent("""
            def f(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                pass
        """)
        func_def = _parse_func(src)
        out_params, tensor_params, scalar_strs = _classify_params(func_def)
        assert "c" in out_params
        assert "c" in tensor_params
        assert "a" not in out_params

    def test_out_param_subscripted_tensor(self):
        """Out[pl.Tensor[[64], pl.FP32]] should still be recognised as Out+tensor."""
        src = textwrap.dedent("""
            def f(c: pl.Out[pl.Tensor[[64], pl.FP32]]):
                pass
        """)
        func_def = _parse_func(src)
        out_params, tensor_params, _ = _classify_params(func_def)
        assert "c" in out_params
        assert "c" in tensor_params

    def test_scalar_bare_dtype(self):
        src = textwrap.dedent("""
            def f(BLOCK_M: pl.INDEX, alpha: pl.FP32):
                pass
        """)
        func_def = _parse_func(src)
        _, _, scalar_strs = _classify_params(func_def)
        assert "BLOCK_M" in scalar_strs
        assert "alpha" in scalar_strs


# ---------------------------------------------------------------------------
# _BodyTransformer
# ---------------------------------------------------------------------------


class TestBodyTransformer:
    def _transform(self, src: str, tensor_meta: dict, dynamic_dims=None, dep_names=None):
        src = textwrap.dedent(src)
        tree = ast.parse(src)
        func_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        transformer = _BodyTransformer(
            tensor_meta=tensor_meta,
            scalar_values={},
            dynamic_dims=dynamic_dims or set(),
            dynvar_python_names={},
            dep_names=dep_names or set(),
            dynvar_var_names=set(),
        )
        new_body = []
        for stmt in func_def.body:
            result = transformer.visit(stmt)
            if result is None:
                continue
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
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

    def _transform_with_scalars(self, src: str, tensor_meta: dict, scalar_values: dict):
        src = textwrap.dedent(src)
        tree = ast.parse(src)
        func_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        transformer = _BodyTransformer(
            tensor_meta=tensor_meta,
            scalar_values=scalar_values,
            dynamic_dims=set(),
            dynvar_python_names={},
            dep_names=set(),
            dynvar_var_names=set(),
        )
        new_body = []
        for stmt in func_def.body:
            result = transformer.visit(stmt)
            if result is None:
                continue
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
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

    def test_scalar_param_substituted_in_body(self):
        """Scalar param references in the body should be replaced by their concrete value."""
        src = """
            def f(a: pl.Tensor, BLOCK_M: pl.INDEX):
                tile = pl.load(a, [0], [BLOCK_M])
        """
        out = self._transform_with_scalars(
            src,
            tensor_meta={"a": TensorMeta((256,), DataType.FP32)},
            scalar_values={"BLOCK_M": 64},
        )
        assert "64" in out
        # Substitution happens in the body; parameter name stays in the signature
        assert "pl.load(a, [0], [64])" in out


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
        dynamic_dims: set | None = None,
        dep_names: list[str] | None = None,
    ) -> str:
        ctx = _make_ctx(
            func_name="kernel",
            source=textwrap.dedent(func_source),
            func_type=func_type,
            param_names=param_names,
            tensor_meta=tensor_meta,
            dynamic_dims=dynamic_dims or set(),
            dep_names=dep_names or [],
        )
        return specialize("_TestClass", [ctx], {})

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

    def test_dynvar_emitted_at_module_level(self):
        src = """
            def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
                M = pl.dynamic("M")
                a.bind_dynamic(0, M)
                c.bind_dynamic(0, M)
                return c
        """
        ctx = _make_ctx(
            func_name="kernel",
            source=textwrap.dedent(src),
            func_type="orchestration",
            param_names=["a", "c"],
            tensor_meta={
                "a": TensorMeta((256, 128), DataType.FP32),
                "c": TensorMeta((256, 128), DataType.FP32),
            },
            dynamic_dims={("a", 0), ("c", 0)},
        )
        dynvar_bindings = {"a__0": "M", "c__0": "M"}
        out = specialize("_Dyn", [ctx], dynvar_bindings)
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
        generated = specialize("_JitAddKernel", [ctx], {})
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
        generated_src = specialize("add_kernel", [ctx], {})
        got = pl.parse(generated_src)

        ir.assert_structural_equal(got, Expected)


# ---------------------------------------------------------------------------
# TestVariableRebinding
# ---------------------------------------------------------------------------


class TestVariableRebinding:
    """Tests for alpha-renaming of variable rebindings in _BodyTransformer."""

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
        transformer = _BodyTransformer(
            tensor_meta=tensor_meta or {},
            scalar_values={},
            dynamic_dims=set(),
            dynvar_python_names={},
            dep_names=set(),
            dynvar_var_names=set(),
            param_names=param_names,
            initial_used_names=all_defined,
        )
        new_body = []
        for stmt in func_def.body:
            result = transformer.visit(stmt)
            if result is None:
                continue
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
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

    def test_rebind_generates_fresh_name(self):
        src = """
            def f(a):
                x = a
                x = pl.load(x)
        """
        out = self._transform(src, tensor_meta={"a": TensorMeta((64,), DataType.FP32)})
        assert "x =" in out
        assert "x_v1 =" in out

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

    def test_parallel_for_simple_rebind(self):
        """Two parallel for loops with simple (non-loop-carried) rebinding."""
        src = """
            def f():
                for i in range(n):
                    x = a
                for j in range(m):
                    x = b
                y = x
        """
        out = self._transform(src)
        assert "x_v1 = x" in out
        assert "y = x_v1" in out

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
        ann = _infer_return_type(func_def, meta, set(), {}, ["out0", "out1"])
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
        assert _infer_return_type(func_def, meta, set(), {}, ["out"]) is None

    def test_return_call_drops_annotation(self):
        """`return f(...)` can't be inferred — caller may have multi-return."""
        func_def = _parse_func(
            """
            def f(a, out):
                return inline_call(a, out)
            """
        )
        meta = {"a": self._meta(), "out": self._meta()}
        assert _infer_return_type(func_def, meta, set(), {}, ["out"]) is None

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
        ann = _infer_return_type(func_def, meta, set(), {}, [])
        assert ann == "pl.Tensor[[32, 32], pl.FP32]"


class TestSpecializerInlineDeprecation:
    """`@pl.jit.inline` should emit bare `pl.Tensor[...]` and warn on `pl.Out`."""

    def _spec(self, ctx_args: dict) -> str:
        ctx = _make_ctx(**ctx_args)
        return Specializer("Generated", [ctx], {}).specialize()

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
        with pytest.warns(DeprecationWarning, match="pl.Out annotations are deprecated"):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
