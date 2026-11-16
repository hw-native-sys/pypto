# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU coverage for explicit CCU registration and per-dispatch domain binding."""

import ast

import pypto.language as pl
import pytest
from pypto import Error, ir
from pypto.ir import DistributedConfig
from pypto.pypto_core import passes
from pypto.pypto_core.backend import BackendType


def overlapping_domains(
    *, bad_signal=False, missing_rank=False, cyclic_order=False, submit=False, two_domains=False
):
    functions = []
    groups = {0: ("a", "z"), 1: ("a", "b", "z"), 2: ("b", "z")}
    if two_domains:
        groups = {0: ("z",), 1: ("b", "z"), 2: ("b", "z")}
    if cyclic_order:
        groups = {0: ("a", "z"), 1: ("b", "a", "z"), 2: ("z", "b")}
    for rank, names in groups.items():
        params = []
        calls = []
        for name in names:
            size = 3 if name == "z" else 2
            params += [
                f"{name}_src: pld.DistributedTensor[[1, 64], pl.BF16]",
                f"{name}_dst: pl.InOut[pld.DistributedTensor[[{size}, 64], pl.BF16]]",
                f"{name}_sig: pl.InOut[pld.DistributedTensor[[{size}, 1], pl.INT32]]",
            ]
            signal = "b_sig" if bad_signal and rank == 1 and name == "a" else f"{name}_sig"
            if missing_rank and rank == 2 and name == "z":
                continue
            calls.append(f"{name}_dst = pld.tensor.allgather({name}_src, {name}_dst, {signal}, engine='ccu')")
        functions.append(
            "    @pl.function(type=pl.FunctionType.Orchestration)\n"
            f"    def chip{rank}(self, " + ", ".join(params) + "):\n        " + "\n        ".join(calls)
        )
    host = []
    for name in sorted({name for names in groups.values() for name in names}):
        size = 3 if name == "z" else 2
        for role, rows, width, dtype, itemsize in (
            ("src", 1, 64, "BF16", 2),
            ("dst", size, 64, "BF16", 2),
            ("sig", size, 1, "INT32", 4),
        ):
            host += [
                f"{name}_{role}_buf = pld.alloc_window_buffer({rows * width * itemsize})",
            ]
    for rank, names in groups.items():
        for name in names:
            size = 3 if name == "z" else 2
            for role, rows, width, dtype in (
                ("src", 1, 64, "BF16"),
                ("dst", size, 64, "BF16"),
                ("sig", size, 1, "INT32"),
            ):
                host.append(
                    f"{name}_{role}_{rank} = pld.window({name}_{role}_buf, "
                    f"[{rows}, {width}], dtype=pl.{dtype})"
                )
        args = [f"{name}_{role}_{rank}" for name in names for role in ("src", "dst", "sig")]
        if submit:
            host.append(f"task{rank} = pl.submit(self.chip{rank}, " + ", ".join(args) + f", device={rank})")
        else:
            host.append(f"self.chip{rank}(" + ", ".join(args) + f", device={rank})")
    functions.append(
        "    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)\n"
        "    def host(self):\n        " + "\n        ".join(host)
    )
    return pl.parse_program("@pl.program\nclass Overlap:\n" + "\n\n".join(functions))


def _compile(program, output):
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        ir.compile(
            program,
            output_dir=str(output),
            backend_type=BackendType.Ascend950,
            skip_ptoas=True,
            dump_passes=True,
            distributed_config=DistributedConfig(device_ids=[0, 1, 2], num_sub_workers=0),
        )


def _assert_local_bindings(source):
    """Evaluate only rank/handle routing in generated HOST code, without runtime."""
    submitted = []

    def value(expr, env):
        if isinstance(expr, ast.Constant):
            return expr.value
        if isinstance(expr, ast.Name):
            return env.get(expr.id, ast.dump(expr))
        return ast.dump(expr)

    def check_call(call, env):
        if not (
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_submit_chip"
        ):
            return
        bindings = next(kw.value for kw in call.keywords if kw.arg == "ccu_bindings")
        assert isinstance(bindings, ast.List)
        for handle in bindings.elts:
            assert isinstance(handle, ast.Name)
            registration = env[handle.id]
            assert registration == (value(call.args[4], env), value(call.args[1], env))
        submitted.append(call)

    def execute(stmts, env):
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.With)):
                execute(stmt.body, env)
            elif isinstance(stmt, ast.For):
                assert isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name)
                assert stmt.iter.func.id == "range" and isinstance(stmt.target, ast.Name)
                bounds = [value(arg, env) for arg in stmt.iter.args]
                integer_bounds = []
                for bound in bounds:
                    assert isinstance(bound, int)
                    integer_bounds.append(bound)
                for rank in range(*integer_bounds):
                    env[stmt.target.id] = rank
                    execute(stmt.body, env)
            elif isinstance(stmt, ast.If):
                condition = stmt.test
                assert isinstance(condition, ast.Compare) and isinstance(condition.ops[0], ast.Eq)
                branch = (
                    stmt.body
                    if value(condition.left, env) == value(condition.comparators[0], env)
                    else stmt.orelse
                )
                execute(branch, env)
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                expr = stmt.value
                if (
                    isinstance(expr, ast.Call)
                    and isinstance(expr.func, ast.Attribute)
                    and expr.func.attr == "get_or_register_ccu"
                ):
                    worker = next(kw.value for kw in expr.keywords if kw.arg == "worker")
                    env[stmt.targets[0].id] = (value(worker, env), value(expr.args[0], env))
                elif isinstance(expr, (ast.Name, ast.Constant)):
                    env[stmt.targets[0].id] = value(expr, env)
                check_call(expr, env)
            elif isinstance(stmt, ast.Expr):
                check_call(stmt.value, env)

    execute(ast.parse(source).body, {})
    assert submitted


@pytest.mark.parametrize("submit", [False, True])
def test_overlapping_domains_codegen(tmp_path, submit):
    _compile(overlapping_domains(submit=submit), tmp_path)
    host = (tmp_path / "orchestration/host_orch.py").read_text()
    ast.parse(host)
    assert host.count(".get_or_register_ccu(") == 7
    assert host.count("ccu_bindings=[") == 3
    assert "workers=[0, 1]" in host
    assert "workers=[1, 2]" in host
    assert "workers=[0, 1, 2]" in host
    assert "ccu_bindings=[_ccu_registration_0, _ccu_registration_1]" in host
    assert "ccu_bindings=[_ccu_registration_2, _ccu_registration_3, _ccu_registration_4]" in host
    assert "ccu_bindings=[_ccu_registration_5, _ccu_registration_6]" in host
    assert host.rindex(".get_or_register_ccu(") < host.index("_submit_chip(")
    _assert_local_bindings(host)
    for chip in ("chip0", "chip1", "chip2"):
        root = tmp_path / "next_levels" / chip
        modules = list((root / "ccu").glob("module_*.cpp"))
        assert len(modules) == 2
        config = (root / "kernel_config.py").read_text()
        assert "CCU_DOMAIN_ABI_VERSION = 2" in config
        assert "CCU_RUNTIME_SUPPORTED = False" in config
        for wrapper in (root / "kernels/aiv").glob("*.cpp"):
            source = wrapper.read_text()
            assert "CommContext *>(args[4])" in source
            assert "get_chip_ccu_execution" not in source
            assert "binding slot" not in source
            assert "submit_ccu_mission(ctx, comm, ccu_id, call_args)" in source
            assert "PeerRank" not in source


def test_same_callable_binds_different_domains(tmp_path):
    program = pl.parse_program("""
@pl.program
class Shared:
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, src: pld.DistributedTensor[[1, 64], pl.BF16],
             dst: pl.InOut[pld.DistributedTensor[[2, 64], pl.BF16]],
             sig: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]]
    ) -> pld.DistributedTensor[[2, 64], pl.BF16]:
        dst = pld.tensor.allgather(src, dst, sig, engine="ccu")
        return dst

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        a_src_buf = pld.alloc_window_buffer(128)
        a_dst_buf = pld.alloc_window_buffer(256)
        a_sig_buf = pld.alloc_window_buffer(8)
        b_src_buf = pld.alloc_window_buffer(128)
        b_dst_buf = pld.alloc_window_buffer(256)
        b_sig_buf = pld.alloc_window_buffer(8)
        for r in pl.range(2):
            a_src = pld.window(a_src_buf, [1, 64], dtype=pl.BF16)
            a_dst = pld.window(a_dst_buf, [2, 64], dtype=pl.BF16)
            a_sig = pld.window(a_sig_buf, [2, 1], dtype=pl.INT32)
            self.chip(a_src, a_dst, a_sig, device=r)
        for r in pl.range(1, 3):
            b_src = pld.window(b_src_buf, [1, 64], dtype=pl.BF16)
            b_dst = pld.window(b_dst_buf, [2, 64], dtype=pl.BF16)
            b_sig = pld.window(b_sig_buf, [2, 1], dtype=pl.INT32)
            self.chip(b_src, b_dst, b_sig, device=r)
""")
    _compile(program, tmp_path)
    host = (tmp_path / "orchestration/host_orch.py").read_text()
    assert host.count('.get_or_register_ccu(callables["chip"], ccu_id=0, worker=') == 4
    _assert_local_bindings(host)
    assert host.count("ccu_bindings=[") == 2
    assert host.rindex(".get_or_register_ccu(") < host.index("_submit_chip(")
    assert len(list((tmp_path / "next_levels/chip/ccu").glob("module_*.cpp"))) == 1


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"bad_signal": True}, "same communication domain"),
        ({"missing_rank": True}, "participation must cover"),
        ({"cyclic_order": True}, "cross-domain wait cycle"),
    ],
)
def test_domain_binding_diagnostics(tmp_path, options, message):
    with pytest.raises((ValueError, RuntimeError, Error), match=message):
        _compile(overlapping_domains(**options), tmp_path)


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_materialized_registration_roundtrip_and_validation(ascend_backend):
    program = passes.lower_l2_tensor_collectives()(passes.convert_to_ssa()(overlapping_domains()))
    program = passes.materialize_comm_domain_scopes()(program)
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))
    text = ir.python_print(program)
    ir.assert_structural_equal(program, pl.parse_program(text))
    ir.assert_structural_equal(program, passes.materialize_comm_domain_scopes()(program))
    assert "pl._dump.CcuRegistration" in text
    assert text.count('"pld.system.register_ccu"') == 7
    assert "ccu_shared_context" not in text
    assert "ccu_uses" not in text
    assert "ccu_slot" not in text

    class BreakSymbol(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("pld.system.register_ccu").name:
                return ir.Call(op.op, op.args, {"callee": "missing"}, op.attrs, op.type, op.span)
            return super().visit_call(op)

    with pytest.raises((ValueError, RuntimeError, Error), match="CCU (function|callee)"):
        passes.materialize_comm_domain_scopes()(BreakSymbol().visit_program(program))


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_reject_mismatched_registration_binding(ascend_backend):
    program = passes.lower_l2_tensor_collectives()(passes.convert_to_ssa()(overlapping_domains()))
    program = passes.materialize_comm_domain_scopes()(program)

    class BreakBinding(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == "chip1":
                attrs = dict(op.attrs)
                bindings = list(attrs["ccu_bindings"])
                bindings[1] = bindings[0]
                attrs["ccu_bindings"] = bindings
                return ir.Call(op.op, op.args, op.kwargs, attrs, op.type, op.span)
            return super().visit_call(op)

    with pytest.raises(
        (ValueError, RuntimeError, Error), match="match the required callable, domain and definition"
    ):
        passes.materialize_comm_domain_scopes()(BreakBinding().visit_program(program))


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("error", ["missing_device", "outside_domain", "wrong_device", "other_rank_handle"])
def test_reject_invalid_registration_device(ascend_backend, error):
    program = passes.materialize_comm_domain_scopes()(
        passes.lower_l2_tensor_collectives()(passes.convert_to_ssa()(overlapping_domains()))
    )

    class BreakDevice(ir.IRMutator):
        def __init__(self):
            super().__init__()
            self.rank0_handle = None

        def visit_call(self, op):
            if op.op.name == ir.get_op("pld.system.register_ccu").name and error != "other_rank_handle":
                attrs = dict(op.attrs)
                if attrs["device"].value == 0:
                    if error == "missing_device":
                        del attrs["device"]
                    else:
                        attrs["device"] = ir.ConstInt(
                            9 if error == "outside_domain" else 1, pl.INDEX, op.span
                        )
                    return ir.Call(op.op, op.args, op.kwargs, attrs, op.type, op.span)
            if op.op.name == "chip0":
                self.rank0_handle = op.attrs["ccu_bindings"][0]
            if op.op.name == "chip1" and error == "other_rank_handle":
                attrs = dict(op.attrs)
                bindings = list(attrs["ccu_bindings"])
                bindings[0] = self.rank0_handle
                attrs["ccu_bindings"] = bindings
                return ir.Call(op.op, op.args, op.kwargs, attrs, op.type, op.span)
            return super().visit_call(op)

    message = {
        "missing_device": "explicit device",
        "outside_domain": "outside the registration domain",
        "wrong_device": "same device",
        "other_rank_handle": "same device",
    }[error]
    with pytest.raises((ValueError, RuntimeError, Error), match=message):
        passes.materialize_comm_domain_scopes()(BreakDevice().visit_program(program))


def test_registration_resolves_callable_local_definition_ids(tmp_path):
    _compile(overlapping_domains(two_domains=True), tmp_path)
    host = (tmp_path / "orchestration/host_orch.py").read_text()
    _assert_local_bindings(host)
    assert host.count(".get_or_register_ccu(") == 5
    assert '.get_or_register_ccu(callables["chip0"], ccu_id=0, worker=0)' in host
    for rank in (1, 2):
        assert f'.get_or_register_ccu(callables["chip{rank}"], ccu_id=1, worker={rank})' in host
    root = tmp_path / "next_levels"
    assert "kRanks = 3" in (root / "chip0/ccu/module_0.cpp").read_text()
    assert "kRanks = 3" in (root / "chip1/ccu/module_1.cpp").read_text()
    assert "kRanks = 3" in (root / "chip2/ccu/module_1.cpp").read_text()
    for chip, expected_id in (("chip0", 0), ("chip1", 1), ("chip2", 1)):
        wrapper = next((root / chip / "kernels/aiv").glob("*bf16_p3*.cpp"))
        assert f"constexpr uint32_t ccu_id = {expected_id};" in wrapper.read_text()


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_distinct_rank_local_definition_symbols(tmp_path, ascend_backend):
    program = passes.lower_l2_tensor_collectives()(passes.convert_to_ssa()(overlapping_domains()))
    functions = {func.name: func for func in program.functions.values()}
    definition = next(
        func
        for func in functions.values()
        if func.func_type == ir.FunctionType.CCU and "nranks_cpp=3" in func.attrs["builtin_template_vars"]
    )
    wrapper = next(
        func
        for func in functions.values()
        if func.func_type == ir.FunctionType.AIV and "bf16_p3" in func.name
    )
    span = definition.span
    params = [ir.Var(param.name_hint, param.type, span) for param in definition.params]
    local_definition = ir.Function(
        "rank2_native",
        list(zip(params, definition.param_directions, strict=True)),
        definition.return_types,
        ir.ReturnStmt([params[1]], span),
        span,
        type=ir.FunctionType.CCU,
        attrs={
            **definition.attrs,
            "builtin_template_vars": ",".join(reversed(definition.attrs["builtin_template_vars"].split(","))),
        },
    )
    params = [ir.Var(param.name_hint, param.type, span) for param in wrapper.params]
    result = ir.Var("result", params[1].type, span)
    call = ir.Call(ir.GlobalVar(local_definition.name), params[:2], {}, {}, result.type, span)
    local_wrapper = ir.Function(
        "rank2_wrapper",
        list(zip(params, wrapper.param_directions, strict=True)),
        wrapper.return_types,
        ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([params[1]], span)], span),
        span,
        type=ir.FunctionType.AIV,
        attrs=wrapper.attrs,
    )

    class ReplaceWrapper(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == wrapper.name:
                return ir.Call(
                    ir.GlobalVar(local_wrapper.name), op.args, op.kwargs, op.attrs, op.type, op.span
                )
            return super().visit_call(op)

    functions["chip2"] = ReplaceWrapper().visit_function(functions["chip2"])
    program = ir.Program([*functions.values(), local_definition, local_wrapper], program.name, program.span)
    _compile(program, tmp_path)
    host = (tmp_path / "orchestration/host_orch.py").read_text()
    assert host.count(".get_or_register_ccu(") == 7
    _assert_local_bindings(host)


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_binding_sequence_has_no_positional_identity(ascend_backend):
    program = passes.materialize_comm_domain_scopes()(
        passes.lower_l2_tensor_collectives()(passes.convert_to_ssa()(overlapping_domains()))
    )

    class Reorder(ir.IRMutator):
        def visit_call(self, op):
            if "ccu_bindings" in op.attrs:
                attrs = dict(op.attrs)
                refs = list(reversed(attrs["ccu_bindings"]))
                attrs["ccu_bindings"] = refs + refs[:1]
                return ir.Call(op.op, op.args, op.kwargs, attrs, op.type, op.span)
            return super().visit_call(op)

    reordered = Reorder().visit_program(program)
    ir.assert_structural_equal(reordered, passes.materialize_comm_domain_scopes()(reordered))


def _rank_loop_program(*, submit=False, two_callables=False, repeat_callable=False):
    dispatch = (
        "dst, task = pl.submit(self.chip, src, dst, sig, device=r)"
        if submit
        else "dst = self.chip(src, dst, sig, device=r)"
    )
    other = (
        "\n            sig = pld.window(sig_buf, [2, 1], dtype=pl.INT32)"
        "\n            dst = pld.window(dst_buf, [2, 64], dtype=pl.BF16)"
        "\n            self.other(src, dst, sig, device=r)"
        if two_callables
        else ""
    )
    source = f"""
@pl.program
class RankLoop:
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, src: pld.DistributedTensor[[1, 64], pl.BF16],
             dst: pl.InOut[pld.DistributedTensor[[2, 64], pl.BF16]],
             sig: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]]
    ) -> pld.DistributedTensor[[2, 64], pl.BF16]:
        dst = pld.tensor.allgather(src, dst, sig, engine="ccu")
        return dst

    @pl.function(type=pl.FunctionType.Orchestration)
    def other(self, src: pld.DistributedTensor[[1, 64], pl.BF16],
              dst: pl.InOut[pld.DistributedTensor[[2, 64], pl.BF16]],
              sig: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]]
    ) -> pld.DistributedTensor[[2, 64], pl.BF16]:
        dst = pld.tensor.allgather(src, dst, sig, engine="ccu")
        return dst

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        src_buf = pld.alloc_window_buffer(128)
        dst_buf = pld.alloc_window_buffer(256)
        sig_buf = pld.alloc_window_buffer(8)
        for r in pl.range(2):
            src = pld.window(src_buf, [1, 64], dtype=pl.BF16)
            dst = pld.window(dst_buf, [2, 64], dtype=pl.BF16)
            sig = pld.window(sig_buf, [2, 1], dtype=pl.INT32)
            {dispatch}{other}
"""
    if repeat_callable:
        source = source.replace("self.other(", "self.chip(")
    return pl.parse_program(source)


@pytest.mark.parametrize("submit", [False, True])
def test_rank_loop_is_preserved_and_registrations_precede_tasks(tmp_path, submit):
    _compile(_rank_loop_program(submit=submit), tmp_path)
    source = (tmp_path / "orchestration/host_orch.py").read_text()
    tree = ast.parse(source)
    assert sum(isinstance(node, ast.For) for node in ast.walk(tree)) == 1
    assert source.count(".get_or_register_ccu(") == 2
    assert source.rindex(".get_or_register_ccu(") < source.index("for r")
    _assert_local_bindings(source)


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("shared", [False, True])
def test_each_dispatch_uses_its_callable_local_definition_ids(tmp_path, ascend_backend, shared):
    # Independent submissions may bind different definitions to ID 0, or the
    # same definition to different IDs. No registration crosses the calls.
    program = passes.lower_l2_tensor_collectives()(
        passes.normalize_return_order()(passes.convert_to_ssa()(_rank_loop_program(two_callables=True)))
    )
    functions = {func.name: func for func in program.functions.values()}
    definition = next(f for f in functions.values() if f.func_type == ir.FunctionType.CCU)
    wrapper = next(f for f in functions.values() if f.func_type == ir.FunctionType.AIV)
    span = program.span
    attrs = dict(definition.attrs)
    attrs["builtin_template_vars"] = "dtype_cpp=bfloat16_t,fixture_variant=a,nranks_cpp=2"
    native = ir.Function(
        "other_native",
        list(zip(definition.params, definition.param_directions, strict=True)),
        definition.return_types,
        definition.body,
        span,
        type=ir.FunctionType.CCU,
        attrs=attrs,
    )
    request = ir.Call(ir.GlobalVar(native.name), wrapper.params[:2], {}, {}, wrapper.params[1].type, span)
    local_wrapper = ir.Function(
        "other_wrapper",
        list(zip(wrapper.params, wrapper.param_directions, strict=True)),
        wrapper.return_types,
        ir.SeqStmts([ir.EvalStmt(request, span), ir.ReturnStmt([wrapper.params[1]], span)], span),
        span,
        type=ir.FunctionType.AIV,
        attrs=wrapper.attrs,
    )

    class ChangeCall(ir.IRMutator):
        def visit_assign_stmt(self, op):
            call = op.value
            if isinstance(call, ir.Call) and call.op.name == wrapper.name:
                other = ir.Call(
                    ir.GlobalVar(local_wrapper.name), call.args, call.kwargs, call.attrs, call.type, call.span
                )
                if shared:
                    prior = ir.Var("prior_target", call.type, op.span)
                    args = list(call.args)
                    args[1] = prior
                    original = ir.Call(call.op, args, call.kwargs, call.attrs, call.type, call.span)
                    return ir.SeqStmts(
                        [ir.AssignStmt(prior, other, op.span), ir.AssignStmt(op.var, original, op.span)],
                        op.span,
                    )
                return ir.AssignStmt(op.var, other, op.span)
            return super().visit_assign_stmt(op)

    functions["other"] = ChangeCall().visit_function(functions["other"])
    program = ir.Program([*functions.values(), native, local_wrapper], program.name, span)
    _compile(program, tmp_path)
    host = (tmp_path / "orchestration/host_orch.py").read_text()
    _assert_local_bindings(host)
    assert host.count(".get_or_register_ccu(") == (6 if shared else 4)
    assert host.rindex(".get_or_register_ccu(") < host.index("_submit_chip(")
    for rank in range(2):
        assert f'callables["chip"], ccu_id=0, worker={rank})' in host
        assert f'callables["other"], ccu_id=0, worker={rank})' in host
        if shared:
            assert f'callables["other"], ccu_id=1, worker={rank})' in host
    if shared:
        # The very same native symbol has ID 0 in chip and ID 1 in other.
        for name, expected in (("chip", 0), ("other", 1)):
            source = (tmp_path / f"next_levels/{name}/kernels/aiv/{wrapper.name}.cpp").read_text()
            assert f"constexpr uint32_t ccu_id = {expected};" in source


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_registration_cannot_be_interleaved_with_rank_submissions(ascend_backend):
    program = passes.materialize_comm_domain_scopes()(
        passes.lower_l2_tensor_collectives()(
            passes.normalize_return_order()(passes.convert_to_ssa()(_rank_loop_program()))
        )
    )

    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))
    ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program)))
    ir.assert_structural_equal(program, passes.materialize_comm_domain_scopes()(program))

    class MoveRegistration(ir.IRMutator):
        def __init__(self):
            super().__init__()
            self.registration = None

        def visit_assign_stmt(self, op):
            if (
                self.registration is None
                and isinstance(op.value, ir.Call)
                and op.value.op.name == ir.get_op("pld.system.register_ccu").name
            ):
                self.registration = op
                return ir.EvalStmt(ir.ConstInt(0, pl.INT32, op.span), op.span)
            return super().visit_assign_stmt(op)

        def visit_for_stmt(self, op):
            assert self.registration is not None
            assert isinstance(op.body, ir.SeqStmts)
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                op.iter_args,
                ir.SeqStmts([self.registration, *op.body.stmts], op.span),
                op.return_vars,
                op.span,
            )

    with pytest.raises((ValueError, RuntimeError, Error), match="precede the first task submission"):
        passes.materialize_comm_domain_scopes()(MoveRegistration().visit_program(program))


@pytest.mark.parametrize("same_callable", [False, True])
def test_each_call_gets_separate_registrations(tmp_path, same_callable):
    _compile(_rank_loop_program(two_callables=True, repeat_callable=same_callable), tmp_path)
    source = (tmp_path / "orchestration/host_orch.py").read_text()
    assert source.count(".get_or_register_ccu(") == 4
    assert source.count("ccu_bindings=[") == 2
    assert source.rindex(".get_or_register_ccu(") < source.index("_submit_chip(")
    _assert_local_bindings(source)
    for chip in ("chip",) if same_callable else ("chip", "other"):
        wrapper = next((tmp_path / "next_levels" / chip / "kernels/aiv").glob("*bf16_p2*.cpp"))
        assert "constexpr uint32_t ccu_id = 0;" in wrapper.read_text()


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_registration_cannot_name_another_consuming_callable(ascend_backend):
    program = passes.lower_l2_tensor_collectives()(
        passes.normalize_return_order()(passes.convert_to_ssa()(_rank_loop_program(two_callables=True)))
    )
    program = passes.materialize_comm_domain_scopes()(program)

    class ChangeCallable(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("pld.system.register_ccu").name and op.kwargs["chip_callee"] == "chip":
                kwargs = dict(op.kwargs)
                kwargs["chip_callee"] = "other"
                return ir.Call(op.op, op.args, kwargs, op.attrs, op.type, op.span)
            return super().visit_call(op)

    # Same native symbol, domain and rank cannot justify taking another call's
    # registration: its callable-local ID may differ after code generation.
    with pytest.raises((ValueError, RuntimeError, Error), match="match the required callable"):
        passes.materialize_comm_domain_scopes()(ChangeCallable().visit_program(program))
