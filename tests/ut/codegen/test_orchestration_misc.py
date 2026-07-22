# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Assorted orchestration-codegen tests (small focused classes)."""

import re

import pypto.language as pl
import pytest
from _orchestration_codegen_common import (
    SELF_ALIAS_RE,
    _generate_orch_code,
)
from pypto import backend, codegen
from pypto.backend import BackendType
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import DataType, ir


class TestTaskIsValidCodegen:
    """``system.task_is_valid`` lowers to ``<expr>.is_valid()`` in C++.

    The op guards each per-slot fill of a manual_scope array-carry TaskId
    into the ``set_dependencies`` stack array.
    Codegen is hand-tested here on minimal IR rather than waiting for the
    end-to-end pass, so the emitter contract is pinned independently of the
    pass implementation.
    """

    def test_task_is_valid_emits_dot_is_valid(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ib = IRBuilder()
        with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
            tid = orch_f.param("tid", ir.ScalarType(DataType.TASK_ID))
            orch_f.return_type(ir.ScalarType(DataType.BOOL))
            # b = task_is_valid(tid)
            check = ir.create_op_call("system.task_is_valid", [tid], {}, ir.Span.unknown())
            b = ib.let("b", check)
            ib.return_stmt(b)
        orch_func = orch_f.get_result()
        program = ir.Program([orch_func], "test_task_is_valid", ir.Span.unknown())

        code = codegen.generate_orchestration(program, orch_func).code
        assert "bool b = tid.is_valid();" in code, code


class TestTupleLineagePointerKeying:
    """Tuple return-alias lineage must be keyed by Var identity, not name_hint.

    Regression for issue #1463: after inlining + OutWindowExternalizer, two
    distinct tuple-producing assignments can share a ``name_hint`` (e.g. several
    rebuilt ``ret__tmp_v0`` MakeTuples). When the orchestration codegen keyed its
    tuple lineage maps by ``name_hint``, the colliding tuples' TupleGetItem
    consumers were cross-wired: the emit names of one tuple's elements were
    propagated onto the other tuple's consumers. In the DeepSeek-V4 KV compressor
    this made the ``kv_state`` / ``score_state`` return aliases reuse the
    externalized ``kv_cache`` / ``kv`` window reshape names, so the generated
    orchestration C++ declared those names twice (``Tensor X = ...`` then
    ``const Tensor& X = ...``) and failed to compile with ``conflicting
    declaration``.
    """

    def test_same_name_tuple_vars_do_not_cross_wire_return_aliases(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        t2d = ir.TensorType([16, 16], pl.FP32)
        tflat = ir.TensorType([256, 1], pl.FP32)
        span = ir.Span.unknown()

        ib = IRBuilder()
        with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
            o0 = orch_f.param("o0", t2d)
            o1 = orch_f.param("o1", t2d)
            orch_f.return_type(ir.TupleType([t2d, t2d]))

            # Two tuple-producing MakeTuple assignments that deliberately share
            # the SAME name_hint "ret" (distinct Var objects) — exactly what the
            # OutWindowExternalizer rebuild produces after inlining. Each tuple
            # wraps a distinct reshape local (rsh0 / rsh1); its TupleGetItem
            # consumer is then reshaped again, so the consumer's lineage must
            # resolve to its OWN tuple's element. Old name_hint keying collapsed
            # both "ret" tuples onto one key, so the first tuple's consumer (a0)
            # lost its lineage and was emitted as the undeclared ``a0.reshape``.
            rsh0 = ib.let("rsh0", tensor_ops.reshape(o0, [256, 1]))
            ret_a = ib.let("ret", ib.make_tuple([rsh0]))
            a0 = ib.let("a0", ir.TupleGetItemExpr(ret_a, 0, span), type=tflat)
            rsh1 = ib.let("rsh1", tensor_ops.reshape(o1, [256, 1]))
            ret_b = ib.let("ret", ib.make_tuple([rsh1]))
            b0 = ib.let("b0", ir.TupleGetItemExpr(ret_b, 0, span), type=tflat)
            fa = ib.let("fa", tensor_ops.reshape(a0, [16, 16]))
            fb = ib.let("fb", tensor_ops.reshape(b0, [16, 16]))
            ib.return_stmt(ib.make_tuple([fa, fb]))

        orch_func = orch_f.get_result()
        program = ir.Program([orch_func], "test_tuple_pointer_keying", span)
        code = codegen.generate_orchestration(program, orch_func).code

        # No declared name may appear twice (the conflicting-declaration bug).
        declared = re.findall(
            r"^\s*(?:const\s+Tensor&|Tensor|PTO2TaskId|auto)\s+([A-Za-z_]\w*)\s*=",
            code,
            flags=re.MULTILINE,
        )
        dups = sorted({n for n in declared if declared.count(n) > 1})
        assert not dups, f"duplicate declarations {dups} in:\n{code}"

        # Each consumer must reshape from its OWN tuple's element, not a stale /
        # undeclared getitem name. Before the fix, ``fa`` read undeclared ``a0``.
        assert "Tensor fa = rsh0.reshape" in code, code
        assert "Tensor fb = rsh1.reshape" in code, code
        assert "a0.reshape" not in code and "b0.reshape" not in code, code


class TestUnregisteredOpError:
    """Test that unregistered/misplaced ops in Orchestration functions raise errors."""

    def test_unregistered_tensor_op_raises_error(self):
        """Unregistered tensor op (tensor.full) in Orchestration must raise RuntimeError."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ib = IRBuilder()
        with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
            orch_f.param("x", ir.TensorType([16, 16], pl.FP32))
            orch_f.return_type(ir.TensorType([16, 16], pl.FP32))
            filled = ib.let("filled", tensor_ops.full([16, 16], pl.FP32, 0.0))
            ib.return_stmt(filled)
        orch_func = orch_f.get_result()

        program = ir.Program([orch_func], "test_prog", ir.Span.unknown())

        with pytest.raises(RuntimeError, match="Misplaced tensor op.*tensor.full"):
            codegen.generate_orchestration(program, orch_func)

    def test_reinterpret_view_has_explicit_orchestration_error(self):
        """Runtime Tensor views cannot change dtype; direct users get an actionable error."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ib = IRBuilder()
        with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
            x = orch_f.param("x", ir.TensorType([8, 16], pl.FP32))
            viewed = ib.let("viewed", tensor_ops.reinterpret_view(x, pl.INT16))
            orch_f.return_type(viewed.type)
            ib.return_stmt(viewed)
        orch_func = orch_f.get_result()
        program = ir.Program([orch_func], "ReinterpretViewOrchestration", ir.Span.unknown())

        with pytest.raises(ValueError, match="not supported in Orchestration functions.*InCore"):
            codegen.generate_orchestration(program, orch_func)


class TestLocalAllocWAWPromotion:
    """Test that locally allocated tensors get add_inout instead of add_output.

    Issue #1022: when a tensor is pre-allocated via alloc_tensors and then
    passed as Out to multiple InCore tasks in separate loops, the codegen
    must use add_inout (not add_output) to establish WAW dependencies.

    The promotion is now performed by the ``DeriveCallDirections`` IR pass,
    which writes ``ArgDirection::InOut`` into ``Call.attrs['arg_directions']`` for
    locally allocated buffers (replacing the legacy ``CallSiteDirectionResolver``
    analysis that lived in orchestration codegen).
    """

    def test_alloc_tensor_two_loops_gets_inout(self):
        """Two loops writing to the same alloc_tensors buffer must use add_inout."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TwoLoopAllocProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def task_init(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                buf: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], buf)
                return out

            @pl.function(type=pl.FunctionType.AIV)
            def task_compute(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                buf: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], buf)
                return out

            @pl.function(type=pl.FunctionType.AIV)
            def task_read(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                buf: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                for _i in pl.range(4):
                    buf = self.task_init(x, buf)
                for _i in pl.range(4):
                    buf = self.task_compute(x, buf)
                out = self.task_read(buf, out)
                return out

        code = _generate_orch_code(TwoLoopAllocProgram)

        assert "add_inout(buf)" in code, (
            "Locally allocated tensor 'buf' passed as Out must generate "
            "add_inout (not add_output) to establish WAW dependencies. "
            f"Generated code:\n{code}"
        )
        assert "add_output(buf)" not in code, (
            f"Locally allocated tensor 'buf' must NOT use add_output. Generated code:\n{code}"
        )

    def test_external_tensor_keeps_add_output(self):
        """Function parameter tensors with Out direction keep add_output."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ExternalOutProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out = self.kernel(a, out)
                return out

        code = _generate_orch_code(ExternalOutProgram)

        assert "add_output(ext_out)" in code, (
            f"External (parameter) tensor should keep add_output. Generated code:\n{code}"
        )

    def test_parallel_loop_local_buf_keeps_add_output(self):
        """Issue #1086: a single ``pl.parallel`` writer of a local buffer must
        emit ``add_output`` (not ``add_inout``).

        Promoting Out → InOut here injects a spurious WAW dependency that
        forces the runtime to serialize otherwise independent iterations of
        the parallel loop, causing the regression observed in Qwen3 decode.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SingleParallelProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def task(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                buf: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                for _i in pl.parallel(4):
                    buf = self.task(a, buf)
                out = self.task(buf, out)
                return out

        code = _generate_orch_code(SingleParallelProgram)

        assert "add_output(buf)" in code, (
            f"Local buf written from a single pl.parallel loop must use add_output, "
            f"not add_inout (issue #1086). Generated code:\n{code}"
        )
        assert "add_inout(buf)" not in code, (
            f"Local buf must not be promoted to add_inout when only a single "
            f"pl.parallel loop writes it. Generated code:\n{code}"
        )

    def test_two_parallel_loops_promote_only_second(self):
        """Two consecutive ``pl.parallel`` loops writing the same local buffer.

        The first loop is the only writer-unit at its scope and stays
        ``add_output``; the second loop hits R-prior so it is promoted to
        ``add_inout`` to keep the cross-loop WAW dependency.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TwoParallelProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def task(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                buf: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                for _i in pl.parallel(4):
                    buf = self.task(a, buf)
                for _j in pl.parallel(4):
                    buf = self.task(a, buf)
                out = self.task(buf, out)
                return out

        code = _generate_orch_code(TwoParallelProgram)

        # Both add_output (first loop, R-prior not yet active) and add_inout
        # (second loop, R-prior fires) must be present for the same `buf`.
        assert "add_output(buf)" in code, (
            f"First pl.parallel writer of buf should remain add_output. Generated code:\n{code}"
        )
        assert "add_inout(buf)" in code, (
            f"Second pl.parallel writer of buf should be promoted to add_inout via R-prior. "
            f"Generated code:\n{code}"
        )


class TestArgDirectionsCodegen:
    """Verify that orchestration codegen prefers Call.attrs['arg_directions'] when present.

    These tests exercise the new ArgDirection-driven path in BuildTaskParams:
    every recognised ArgDirection enum value is mapped to the matching runtime
    method (add_input / add_output / add_inout / add_no_dep / add_scalar) and
    the value emitted at the call site reflects the per-argument direction
    written by the DeriveCallDirections pass — independently of the callee's
    ParamDirection.
    """

    @staticmethod
    def _generate_orch_direct(program) -> str:
        """Bypass ``_ensure_arg_directions`` so explicit overrides survive."""
        for func in program.functions.values():
            if func.func_type == ir.FunctionType.Orchestration:
                return codegen.generate_orchestration(program, func).code
        raise ValueError("No orchestration function found in program")

    def _build_program_with_arg_directions(self, arg_dirs):
        """Build a tiny Orchestration program where the call site has explicit arg_directions.

        The callee declares ``Out`` for the second parameter, and the orchestration
        body pre-allocates the tensor with ``tensor.create``. We then patch the
        call expression with the requested ``arg_directions`` so that codegen
        consumes them directly (bypassing the legacy ParamDirection mapping).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class ArgDirProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                buf: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                buf = self.kernel(a, buf)
                out = self.kernel(buf, out)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        program = pm.run_passes(ArgDirProgram)

        rewritten = self._rewrite_kernel_calls(program, arg_dirs)
        return rewritten

    @staticmethod
    def _rewrite_kernel_calls(program, arg_dirs):
        """Replace every ``self.kernel(...)`` Call with a copy carrying the given arg_directions."""

        class _RewriteKernel(ir.IRMutator):
            def visit_call(self, op: ir.Call) -> ir.Expr:
                expr = super().visit_call(op)
                call = expr if isinstance(expr, ir.Call) else op
                if call.op.name != "kernel" or len(call.args) != len(arg_dirs):
                    return expr
                attrs = {"arg_directions": list(arg_dirs)}
                return ir.Call(call.op, list(call.args), dict(call.kwargs), attrs, call.type, call.span)

        return _RewriteKernel().visit_program(program)

    def test_arg_direction_inout_emits_add_inout(self):
        program = self._build_program_with_arg_directions([ir.ArgDirection.Input, ir.ArgDirection.InOut])
        code = self._generate_orch_direct(program)
        assert "add_inout(buf)" in code, (
            f"ArgDirection::InOut on the second argument must produce add_inout(...). Generated code:\n{code}"
        )

    def test_arg_direction_output_existing_emits_add_output(self):
        program = self._build_program_with_arg_directions(
            [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]
        )
        code = self._generate_orch_direct(program)
        assert "add_output(" in code, (
            f"ArgDirection::OutputExisting must produce add_output(...). Generated code:\n{code}"
        )
        assert "add_inout(" not in code or code.count("add_inout(") < code.count("add_output("), (
            f"Expected add_output to dominate over add_inout. Generated code:\n{code}"
        )

    def test_arg_direction_no_dep_emits_add_no_dep(self):
        program = self._build_program_with_arg_directions([ir.ArgDirection.Input, ir.ArgDirection.NoDep])
        code = self._generate_orch_direct(program)
        assert "add_no_dep(" in code, (
            f"ArgDirection::NoDep must produce add_no_dep(...). Generated code:\n{code}"
        )

    def test_arg_direction_input_emits_add_input(self):
        program = self._build_program_with_arg_directions([ir.ArgDirection.Input, ir.ArgDirection.Input])
        code = self._generate_orch_direct(program)
        assert "add_input(" in code, (
            f"ArgDirection::Input must produce add_input(...). Generated code:\n{code}"
        )
        assert "add_output(" not in code and "add_inout(" not in code, (
            "When all tensor args are ArgDirection::Input the codegen must not emit add_output/add_inout. "
            f"Generated code:\n{code}"
        )


class TestNoOpAliasSkip:
    """Regression coverage for issue #1281 sub-problem 2.

    When VarLineageCollector collapses several Vars onto the same param-rooted
    emit name (because they all alias the same buffer), a chained Var-RHS
    AssignStmt like ``u = t`` reaches the catch-all emit branch with both LHS
    and RHS resolving to the same C++ identifier. Pre-fix, the codegen emitted
    ``auto X = X;`` literally, which gcc rejects with
    ``use of 'X' before deduction of 'auto'``.

    The trigger is: an Orchestration entry that

      1. Calls a kernel with an Out/InOut param,
      2. Passes the entry's own pl.Out param as the actual arg, and
      3. Binds the call result to one local AND aliases it to a second local
         before returning.

    Forms A and B below are control cases that already worked; form C is the
    one that used to emit the self-alias.
    """

    def test_form_a_direct_return_no_alias(self):
        """Control: `return self.kern(...)` — single return path, never tripped the bug."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class FormA:
            @pl.function(type=pl.FunctionType.AIV)
            def kern(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def entry(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                q_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                return self.kern(x, q_out)

        code = _generate_orch_code(FormA)
        assert not SELF_ALIAS_RE.search(code), f"unexpected `auto X = X;` in form A. Code:\n{code}"

    def test_form_b_single_bind_no_alias(self):
        """Control: `t = self.kern(...); return t` — single AssignStmt, handled by
        GenerateSingleReturnAlias's existing ``alias_name != out_arg`` guard."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class FormB:
            @pl.function(type=pl.FunctionType.AIV)
            def kern(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def entry(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                q_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tensor[[16, 16], pl.FP32] = self.kern(x, q_out)
                return t

        code = _generate_orch_code(FormB)
        assert not SELF_ALIAS_RE.search(code), f"unexpected `auto X = X;` in form B. Code:\n{code}"

    def test_form_c_chained_alias_drops_no_op(self):
        """`t = self.kern(...); u = t; return u` — the bug trigger.

        Pre-fix: emits `auto q_out = q_out;` for the `u = t` AssignStmt because
        VarLineageCollector collapses both `t` and `u` onto the entry's `q_out`
        param emit name. Post-fix: the catch-all Var-RHS branch detects
        LHS-name == RHS-name and drops the AssignStmt entirely.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class FormC:
            @pl.function(type=pl.FunctionType.AIV)
            def kern(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
                ret: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def entry(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                q_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tensor[[16, 16], pl.FP32] = self.kern(x, q_out)
                u: pl.Tensor[[16, 16], pl.FP32] = t
                return u

        code = _generate_orch_code(FormC)
        assert not SELF_ALIAS_RE.search(code), (
            f"`auto X = X;` regression — issue #1281 sub-problem 2 is back. Code:\n{code}"
        )
        # Sanity: the task submission must still be present; the fix only
        # drops the no-op alias, not the actual kernel call.
        assert "rt_submit_aiv_task" in code, f"task submission missing from form C output. Code:\n{code}"


class TestTupleReturnNoDepAliasing:
    """``GenerateTupleReturnAliases`` must classify output slots by the
    callee's ``ParamDirection`` — same convention as the submit path. If it
    classified by call-site ``ArgDirection`` instead, a ``pl.no_dep(out)``
    on a tuple-return non-submit Call would drop the alias for that slot
    (``NoDep`` is excluded from the writer set), and downstream uses of the
    tuple-element SSA var would emit undeclared ``__rv_*`` symbols.
    """

    def test_no_dep_on_tuple_out_param_preserves_alias(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class TupleNoDepProgram:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel_pair(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                out_s: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                s: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                d: pl.Tile[[16, 16], pl.FP32] = pl.sub(a_tile, b_tile)
                rs: pl.Tensor[[16, 16], pl.FP32] = pl.store(s, [0, 0], out_s)
                rd: pl.Tensor[[16, 16], pl.FP32] = pl.store(d, [0, 0], out_d)
                return rs, rd

            @pl.function(type=pl.FunctionType.AIV)
            def kernel_consume(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
                result: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
                out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                result: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                y: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                # ``pl.no_dep(y)`` rewrites the y slot's ArgDirection to NoDep
                # on a multi-output (tuple-returning) non-submit call. The
                # codegen must still treat y as a writer — otherwise the
                # downstream `kernel_consume(x, y, ...)` would reference an
                # undeclared y.
                x, y = self.kernel_pair(a, b, x, pl.no_dep(y))
                result = self.kernel_consume(x, y, result)
                return result

        # Same trick as ``test_flatten_call_expr_pass.TestFlattenPreservesAttrs``
        # / ``TestOutlineNoDepArgs``: ``derive_call_directions`` produces a
        # Call whose ``attrs[arg_directions]`` includes a NoDep slot; the
        # printer does not surface that attr, so the default
        # RoundtripInstrument check fails. Use VerificationInstrument only.
        from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

        ctx = _core_passes.PassContext(
            [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
        )
        with ctx:
            code = _generate_orch_code(TupleNoDepProgram)

        # The y slot must be marked NoDep on the kernel_pair call.
        assert "add_no_dep(" in code, (
            f"expected add_no_dep(...) on the NoDep y slot of kernel_pair; generated code:\n{code}"
        )
        # Both x and y must have aliases bound (otherwise the consume call
        # below would reference undeclared symbols). The aliasing path uses
        # ``orch_args.tensor(i).ref()`` on the args array.
        assert code.count(".ref()") >= 2, (
            "expected at least two orch_args.tensor(i).ref() bindings (one per "
            f"tuple element); generated code:\n{code}"
        )
        # Both tasks must submit.
        assert code.count("rt_submit_aiv_task") == 2, code


class TestTupleReturnNameHintCollision:
    """Tuple metadata must track tuple Vars by identity, not name_hint."""

    def test_same_name_hint_tuple_calls_keep_distinct_elements(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        span = ir.Span.unknown()
        tensor_type = ir.TensorType([16, 16], DataType.FP32)
        tuple_type = ir.TupleType([tensor_type, tensor_type])

        input_a = ir.Var("input_a", tensor_type, span)
        input_b = ir.Var("input_b", tensor_type, span)
        # Distinct Out tensors per call, so the two tuple results' element->call
        # attachment is observable in the consumer's inputs under emit-name
        # remap (the alias-decl form this test predates is no longer emitted).
        out_a1 = ir.Var("out_a1", tensor_type, span)
        out_a2 = ir.Var("out_a2", tensor_type, span)
        out_b1 = ir.Var("out_b1", tensor_type, span)
        out_b2 = ir.Var("out_b2", tensor_type, span)
        final_out = ir.Var("final_out", tensor_type, span)
        kernel_a_input = ir.Var("input_a", tensor_type, span)
        kernel_a_first = ir.Var("first_out", tensor_type, span)
        kernel_a_second = ir.Var("second_out", tensor_type, span)
        kernel_b_input = ir.Var("input_b", tensor_type, span)
        kernel_b_first = ir.Var("first_out", tensor_type, span)
        kernel_b_second = ir.Var("second_out", tensor_type, span)

        kernel_a_body = ir.SeqStmts([ir.ReturnStmt([kernel_a_first, kernel_a_second], span)], span)
        kernel_a = ir.Function(
            "kernel_a",
            [
                (kernel_a_input, ir.ParamDirection.In),
                (kernel_a_first, ir.ParamDirection.Out),
                (kernel_a_second, ir.ParamDirection.Out),
            ],
            [tensor_type, tensor_type],
            kernel_a_body,
            span,
            ir.FunctionType.AIV,
        )

        kernel_b_body = ir.SeqStmts([ir.ReturnStmt([kernel_b_first, kernel_b_second], span)], span)
        kernel_b = ir.Function(
            "kernel_b",
            [
                (kernel_b_input, ir.ParamDirection.In),
                (kernel_b_first, ir.ParamDirection.Out),
                (kernel_b_second, ir.ParamDirection.Out),
            ],
            [tensor_type, tensor_type],
            kernel_b_body,
            span,
            ir.FunctionType.AIV,
        )

        consume_a = ir.Var("consume_a", tensor_type, span)
        consume_b = ir.Var("consume_b", tensor_type, span)
        consume_c = ir.Var("consume_c", tensor_type, span)
        consume_d = ir.Var("consume_d", tensor_type, span)
        consume_out = ir.Var("consume_out", tensor_type, span)
        consume_body = ir.SeqStmts([ir.ReturnStmt([consume_out], span)], span)
        kernel_consume = ir.Function(
            "kernel_consume",
            [
                (consume_a, ir.ParamDirection.In),
                (consume_b, ir.ParamDirection.In),
                (consume_c, ir.ParamDirection.In),
                (consume_d, ir.ParamDirection.In),
                (consume_out, ir.ParamDirection.Out),
            ],
            [tensor_type],
            consume_body,
            span,
            ir.FunctionType.AIV,
        )

        tmp_first = ir.Var("ret__tmp_v0", tuple_type, span)
        tmp_second = ir.Var("ret__tmp_v0", tuple_type, span)
        first_a = ir.Var("first_a", tensor_type, span)
        second_a = ir.Var("second_a", tensor_type, span)
        first_b = ir.Var("first_b", tensor_type, span)
        second_b = ir.Var("second_b", tensor_type, span)
        consume_result = ir.Var("consume_result", tensor_type, span)

        call_a = ir.Call(
            ir.GlobalVar("kernel_a"),
            [input_a, out_a1, out_a2],
            {},
            {
                "arg_directions": [
                    ir.ArgDirection.Input,
                    ir.ArgDirection.OutputExisting,
                    ir.ArgDirection.OutputExisting,
                ]
            },
            tuple_type,
            span,
        )
        call_b = ir.Call(
            ir.GlobalVar("kernel_b"),
            [input_b, out_b1, out_b2],
            {},
            {
                "arg_directions": [
                    ir.ArgDirection.Input,
                    ir.ArgDirection.OutputExisting,
                    ir.ArgDirection.OutputExisting,
                ]
            },
            tuple_type,
            span,
        )
        call_consume = ir.Call(
            ir.GlobalVar("kernel_consume"),
            [first_a, second_a, first_b, second_b, final_out],
            {},
            {
                "arg_directions": [
                    ir.ArgDirection.Input,
                    ir.ArgDirection.Input,
                    ir.ArgDirection.Input,
                    ir.ArgDirection.Input,
                    ir.ArgDirection.OutputExisting,
                ]
            },
            tensor_type,
            span,
        )

        orch_body = ir.SeqStmts(
            [
                ir.AssignStmt(tmp_first, call_a, span),
                ir.AssignStmt(tmp_second, call_b, span),
                ir.AssignStmt(first_a, ir.TupleGetItemExpr(tmp_first, 0, span), span),
                ir.AssignStmt(second_a, ir.TupleGetItemExpr(tmp_first, 1, span), span),
                ir.AssignStmt(first_b, ir.TupleGetItemExpr(tmp_second, 0, span), span),
                ir.AssignStmt(second_b, ir.TupleGetItemExpr(tmp_second, 1, span), span),
                ir.AssignStmt(consume_result, call_consume, span),
                ir.ReturnStmt([consume_result], span),
            ],
            span,
        )
        orch = ir.Function(
            "orch",
            [
                (input_a, ir.ParamDirection.In),
                (input_b, ir.ParamDirection.In),
                (out_a1, ir.ParamDirection.Out),
                (out_a2, ir.ParamDirection.Out),
                (out_b1, ir.ParamDirection.Out),
                (out_b2, ir.ParamDirection.Out),
                (final_out, ir.ParamDirection.Out),
            ],
            [tensor_type],
            orch_body,
            span,
            ir.FunctionType.Orchestration,
        )
        program = ir.Program(
            [kernel_a, kernel_b, kernel_consume, orch],
            "TupleNameHintCollisionProgram",
            span,
        )

        code = codegen.generate_orchestration(program, orch).code

        # tmp_first and tmp_second share the name_hint "ret__tmp_v0"; the tuple
        # metadata must still attach each call's elements to that call. Each
        # element is the in-place Out arg of its call, so it remaps to that arg
        # (no ``const Tensor& first_a = ...`` alias is minted). The consumer
        # reading first_a/second_a/first_b/second_b therefore reads call_a's
        # outs then call_b's outs, in order — not cross-contaminated.
        i_a1 = code.index("// Task 2: kernel_consume")
        consume = code[i_a1:]
        a1 = consume.index("add_input(ext_out_a1)")
        a2 = consume.index("add_input(ext_out_a2)")
        b1 = consume.index("add_input(ext_out_b1)")
        b2 = consume.index("add_input(ext_out_b2)")
        assert a1 < a2 < b1 < b2, code
        # No per-element const-ref alias survives the remap.
        for name in ("first_a", "second_a", "first_b", "second_b"):
            assert f"const Tensor& {name} " not in code, code

        declared_names = re.findall(
            r"^\s*(?:const\s+Tensor&|Tensor)\s+([A-Za-z_]\w*)\s*=",
            code,
            flags=re.MULTILINE,
        )
        duplicate_declarations = {name for name in declared_names if declared_names.count(name) > 1}
        assert not duplicate_declarations, (
            f"generated C++ redeclared tensor names {sorted(duplicate_declarations)}:\n{code}"
        )


class TestScalarCarryPhiCodegen:
    """Regression tests for scalar loop carries in orchestration codegen."""

    def test_scalar_carry_phi_not_emitted_as_tensor(self):
        """Regression for #1580: Scalar loop carry must not be aliased as const Tensor&.

        When a Scalar variable is defined before a pl.parallel loop and then
        reused (reassigned) inside it, alongside Tensor carries, ConvertToSSA
        promotes the scalar into the parallel-loop carry tuple.  The orchestration
        codegen must emit the Scalar carry phi as ``int64_t = 0`` (untraced scalar
        default), NOT as ``const Tensor& = <carry_var>`` (type mismatch that causes
        a C++ compile error).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, TILE = 16, 4

        @pl.program
        class ScalarCarryProg:
            @pl.function(type=pl.FunctionType.AIV)
            def scope_b_kernel(
                self,
                x: pl.Tensor[[N, N], pl.FP32],
                out_b: pl.Out[pl.Tensor[[N, N], pl.FP32]],
                out_c: pl.Out[pl.Tensor[[N, N], pl.FP32]],
            ) -> tuple[pl.Tensor[[N, N], pl.FP32], pl.Tensor[[N, N], pl.FP32]]:
                t: pl.Tile[[N, N], pl.FP32] = pl.load(x, [0, 0], [N, N])
                out_b = pl.store(t, [0, 0], out_b)
                out_c = pl.store(t, [0, 0], out_c)
                return out_b, out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[N, N], pl.FP32],
                out_b: pl.Out[pl.Tensor[[N, N], pl.FP32]],
                out_c: pl.Out[pl.Tensor[[N, N], pl.FP32]],
                row_start: pl.Scalar[pl.INDEX],
            ) -> tuple[pl.Tensor[[N, N], pl.FP32], pl.Tensor[[N, N], pl.FP32]]:
                # global_c_idx is assigned from a scalar param before the
                # parallel loop — ConvertToSSA sees it in 'before' and adds it
                # as a carry when it is reassigned inside the loop body.
                global_c_idx = row_start

                # The parallel loop carries global_c_idx (Scalar) mixed with
                # Tensor carries out_b, out_c.  Before the fix, the Scalar carry
                # phi was emitted as ``const Tensor&`` causing a C++ compile error.
                for batch_idx in pl.parallel(0, N // TILE):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scope_b"):
                        for inner in pl.range(TILE):
                            global_c_idx = batch_idx + inner  # noqa: F841
                            out_b, out_c = self.scope_b_kernel(x, out_b, out_c)

                return out_b, out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(ScalarCarryProg)
        code = _generate_orch_code(transformed)

        # The Scalar carry phi must be emitted as int64_t = 0 (untraced scalar
        # default), never as const Tensor& = <carry> (type mismatch / #1580).
        assert "int64_t global_c_idx__rv" in code, (
            "global_c_idx carry phi should be emitted as int64_t, not const Tensor&\n" + code
        )
        assert "const Tensor& global_c_idx" not in code, (
            "global_c_idx must not be aliased as const Tensor& (scalar/tensor type mismatch)\n" + code
        )

        # out_b and out_c Tensor carries must each alias to their own carry.
        # The scrambled (shifted-by-one) bindings must NOT appear.
        for line in code.splitlines():
            stripped = line.strip()
            if "=" not in stripped:
                continue
            lhs, _, rhs = stripped.partition("=")
            # out_c phi must not be initialized from out_b's carry value
            if "out_c" in lhs and "out_b" in rhs and "out_c" not in rhs:
                raise AssertionError(f"Wrong phi: out_c assigned from out_b value (scrambled):\n  {stripped}")
            # out_b phi must not be initialized from out_c's carry value
            if "out_b" in lhs and "out_c" in rhs and "out_b" not in rhs:
                raise AssertionError(f"Wrong phi: out_b assigned from out_c value (scrambled):\n  {stripped}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
