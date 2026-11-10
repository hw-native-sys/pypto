# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Closed Graph-boundary metadata may choose topology; runtime/data may not."""

import pypto.language as pl
import pytest
from pypto import backend, passes
from pypto import ir as public_ir
from pypto.pypto_core import codegen, ir
from pypto.pypto_core import passes as core_passes


def _program(case):
    implicit = case == "posthoist"
    guarded = case == "guarded_division"
    tail = "" if implicit else ", width: pl.Scalar[pl.INDEX]"
    condition = "pl.tensor.dim(profile, 0) > 8" if implicit else "width > 8"
    calls = "            self.touch(out)\n"
    if guarded:
        calls = "            quotient = 128 // (width - 8)\n" + calls
    if case == "guarded_view":
        calls = "            slab = pl.tensor.slice(profile, [16, 16], [128 // (width - 8), 0])\n" + calls
    if case == "overflow":
        calls = "            for i in pl.range(1100):\n                self.touch(out)\n"
    fallback = calls
    if case == "one_arm":
        fallback = "            pass\n"
    if case == "guarded_reshape":
        condition = "width == 16"
        calls = "            slab = pl.tensor.view(profile, [8, 32])\n" + calls
    source = f"""
N = pl.dynamic("TOPOLOGY_ROWS")
K = pl.dynamic("EXTERNAL_ROWS")
@pl.program
class P:
    @pl.function(type=pl.FunctionType.AIV)
    def touch(self, out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        t: pl.Tile[[16, 16], pl.FP32] = pl.load(out, [0, 0], [16, 16])
        return pl.store(t, [0, 0], out)

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self, out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], profile: pl.Tensor[[N, 16], pl.FP32],
        data: pl.Tensor[[1], pl.INT32]{tail},
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        if {condition}:
{calls}        else:
{fallback}        return out
"""
    if case == "no_caller":
        return pl.parse_program(source)
    argument = {
        "metadata": "pl.tensor.dim(profile, 0)",
        "guarded_division": "pl.tensor.dim(profile, 0)",
        "guarded_view": "pl.tensor.dim(profile, 0)",
        "posthoist": "",
        "local_tensor": "pl.tensor.dim(local, 0)",
        "num_tokens": "runtime",
        "device_data": "pl.cast(pl.tensor.read(data, [0]), pl.INDEX)",
        "external_tensor": "pl.tensor.dim(external, 0)",
        "foreign_at_site": "pl.tensor.dim(external, 0)",
        "mixed": "pl.tensor.dim(profile, 0)",
        "all_metadata": "pl.tensor.dim(profile, 0)",
        "loop_index": "i",
        "overflow": "pl.tensor.dim(profile, 0)",
        "one_arm": "pl.tensor.dim(profile, 0)",
        "guarded_reshape": "pl.tensor.dim(profile, 0)",
    }[case]
    prelude = (
        "        local = pl.create_tensor([runtime, 16], dtype=pl.FP32)\n" if case == "local_tensor" else ""
    )
    profile_arg = "local" if case == "local_tensor" else "profile"
    invocation = f"self.layer(out, {profile_arg}, data" + (f", {argument})" if not implicit else ")")
    body = prelude + "        " + invocation + "\n"
    if case == "loop_index":
        body = "        for i in pl.range(2):\n            " + invocation + "\n"
    if case == "foreign_at_site":
        # The second Graph pins external. Its descriptor must not bless the
        # first Graph, which reads external's shape but forwards profile.
        graph_start = source.index("    @pl.function(type=pl.FunctionType.Graph)")
        source += source[graph_start:].replace("def layer(", "def other_layer(", 1)
        body += "        self.other_layer(out, external, data, pl.tensor.dim(external, 0))\n"
    if case == "all_metadata":
        # The scalar values differ. GraphConfig keys the generated topology
        # on the actual scalar slots read by the Graph body.
        body += (
            "        other_width = pl.tensor.dim(profile, 0) * 2\n"
            "        self.layer(out, profile, data, other_width)\n"
        )
    source += f"""
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], profile: pl.Tensor[[N, 16], pl.FP32],
        data: pl.Tensor[[1], pl.INT32], runtime: pl.Scalar[pl.INDEX],
        external: pl.Tensor[[K, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
{body}        return out
"""
    if case == "mixed":
        source += """
    @pl.function(type=pl.FunctionType.Orchestration)
    def other(
        self, out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], profile: pl.Tensor[[N, 16], pl.FP32],
        data: pl.Tensor[[1], pl.INT32], num_tokens: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        self.layer(out, profile, data, num_tokens)
        return out
"""
    return pl.parse_program(source)


def _lower(program):
    return passes.normalize_return_order()(passes.outline_incore_scopes()(passes.convert_to_ssa()(program)))


def _diagnostics(program):
    properties = core_passes.IRPropertySet()
    properties.insert(core_passes.IRProperty.GraphBoundaryLegalized)
    return [
        d
        for d in core_passes.PropertyVerifierRegistry.verify(properties, program)
        if d.rule_name == "GraphBoundaryLegalized"
    ]


@pytest.mark.parametrize("case", ["metadata", "posthoist", "local_tensor", "all_metadata"])
def test_metadata_topology_pass_and_independent_verifier(case):
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        normalized = _lower(_program(case))
        legalized = passes.legalize_graph_boundary()(normalized)
        ir.assert_structural_equal(legalized, normalized)
        assert _diagnostics(legalized) == []
        if case == "posthoist":
            # The parser already folded dim(profile, 0) to the descriptor symbol.
            graph = legalized.get_function("layer")
            assert graph is not None
            assert not any(isinstance(p.type, ir.ScalarType) for p in graph.params)

        if case == "all_metadata":
            # Different metadata expressions at two sites are safe because
            # host topology reads are keyed through GraphConfig.
            backend.reset_for_testing()
            backend.set_backend_type(backend.BackendType.Ascend950)
            lowered = passes.derive_call_directions()(legalized)
            lowered = passes.materialize_runtime_scopes()(lowered)
            lowered = passes.classify_iter_arg_carry()(lowered)
            main = lowered.get_function("main")
            assert main is not None
            cpp = codegen.generate_orchestration(lowered, main).code
            assert "from_u64<int64_t>(_pypto_graph_config.scalar_0)" in cpp
            assert cpp.count(".scalar_0 = to_u64(") == 2


@pytest.mark.parametrize(
    "case",
    [
        "num_tokens",
        "device_data",
        "external_tensor",
        "foreign_at_site",
        "no_caller",
        "mixed",
        "loop_index",
        "overflow",
    ],
)
def test_nonclosed_topology_pass_and_independent_verifier_reject(case):
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        normalized = _lower(_program(case))
        # Verify the untrusted current IR directly, without running legalization.
        assert any("conditional" in d.message or "limit" in d.message for d in _diagnostics(normalized))
        with pytest.raises(ValueError, match="conditional|limit"):
            passes.legalize_graph_boundary()(normalized)


@pytest.mark.parametrize("shadow", ["same_var", "same_name", "local_definition", "commctx_name"])
def test_shape_annotation_cannot_launder_a_scalar_or_shadow_into_metadata(shadow):
    # Raw IR preserves identity differences which frontend name resolution erases.

    s = public_ir.Span.unknown()
    index = public_ir.ScalarType(public_ir.DataType.INDEX)
    symbol = public_ir.Var("ROWS", index, s)
    scalar_type = public_ir.CommCtxType.get() if shadow == "commctx_name" else index
    scalar = symbol if shadow == "same_var" else public_ir.Var("ROWS", scalar_type, s)
    profile_type = public_ir.TensorType([symbol, 16], public_ir.DataType.FP32)
    out_type = public_ir.TensorType([16, 16], public_ir.DataType.FP32)
    out = public_ir.Var("out", out_type, s)
    profile = public_ir.Var("profile", profile_type, s)
    width = public_ir.Var("width", index, s)
    child_out = public_ir.Var("child_out", out_type, s)
    child = public_ir.Function(
        "touch",
        [(child_out, public_ir.ParamDirection.InOut)],
        [out_type],
        public_ir.ReturnStmt([child_out], s),
        s,
        public_ir.FunctionType.AIV,
    )

    def task():
        value = public_ir.Var("touch_result", out_type, s)
        return public_ir.AssignStmt(
            value, public_ir.Call(public_ir.GlobalVar("touch"), [out], out_type, s), s
        )

    condition = public_ir.Gt(
        width, public_ir.ConstInt(8, public_ir.DataType.INDEX, s), public_ir.DataType.BOOL, s
    )
    graph = public_ir.Function(
        "layer",
        [(out, public_ir.ParamDirection.InOut), profile, width],
        [out_type],
        public_ir.SeqStmts(
            [public_ir.IfStmt(condition, task(), task(), [], s), public_ir.ReturnStmt([out], s)], s
        ),
        s,
        public_ir.FunctionType.Graph,
    )
    runtime = public_ir.Var("runtime", index, s)
    params = [(out, public_ir.ParamDirection.InOut), profile]
    statements = []
    if shadow == "local_definition":
        params.append(runtime)
        statements.append(public_ir.AssignStmt(scalar, runtime, s))
    else:
        params.append(scalar)
    result = public_ir.Var("result", out_type, s)
    # The actual width is the shape symbol even when a distinct scalar owns its emitted name.
    statements.extend(
        [
            public_ir.AssignStmt(
                result, public_ir.Call(public_ir.GlobalVar("layer"), [out, profile, symbol], out_type, s), s
            ),
            public_ir.ReturnStmt([result], s),
        ]
    )
    caller = public_ir.Function(
        "main", params, [out_type], public_ir.SeqStmts(statements, s), s, public_ir.FunctionType.Orchestration
    )
    program = public_ir.Program([child, graph, caller], "shadowed_shape_symbol", s)
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        assert any("conditional" in d.message for d in _diagnostics(program))
        with pytest.raises(ValueError, match="conditional"):
            passes.legalize_graph_boundary()(program)


def test_explicit_dim_call_and_posthoist_scalar_are_independently_reproved():
    # Keep a real dim Call in late IR: frontend folding to a free dimension
    # symbol would not exercise Step A's added scalar parameter and callsite.

    s = public_ir.Span.unknown()
    dtype = public_ir.DataType
    index = public_ir.ScalarType(dtype.INDEX)
    n = public_ir.Var("ROWS", index, s)
    tensor = public_ir.TensorType([16, 16], dtype.FP32)
    profile_type = public_ir.TensorType([n, 16], dtype.FP32)
    out = public_ir.Var("out", tensor, s)
    profile = public_ir.Var("profile", profile_type, s)
    width = public_ir.Var("computed_width", index, s)
    child_out = public_ir.Var("child_out", tensor, s)
    child_width = public_ir.Var("child_width", index, s)
    child = public_ir.Function(
        "touch",
        [(child_out, public_ir.ParamDirection.InOut), child_width],
        [tensor],
        public_ir.ReturnStmt([child_out], s),
        s,
        public_ir.FunctionType.AIV,
    )

    def task(name):
        result = public_ir.Var(name, tensor, s)
        return public_ir.AssignStmt(
            result, public_ir.Call(public_ir.GlobalVar("touch"), [out, width], tensor, s), s
        )

    dim = public_ir.Call(
        public_ir.Op("tensor.dim"), [profile, public_ir.ConstInt(0, dtype.INDEX, s)], index, s
    )
    condition = public_ir.Gt(width, public_ir.ConstInt(8, dtype.INDEX, s), dtype.BOOL, s)
    graph = public_ir.Function(
        "layer",
        [(out, public_ir.ParamDirection.InOut), profile],
        [tensor],
        public_ir.SeqStmts(
            [
                public_ir.AssignStmt(width, dim, s),
                public_ir.IfStmt(condition, task("then_result"), task("else_result"), [], s),
                public_ir.ReturnStmt([out], s),
            ],
            s,
        ),
        s,
        public_ir.FunctionType.Graph,
    )
    result = public_ir.Var("result", tensor, s)
    caller = public_ir.Function(
        "main",
        [(out, public_ir.ParamDirection.InOut), profile],
        [tensor],
        public_ir.SeqStmts(
            [
                public_ir.AssignStmt(
                    result, public_ir.Call(public_ir.GlobalVar("layer"), [out, profile], tensor, s), s
                ),
                public_ir.ReturnStmt([result], s),
            ],
            s,
        ),
        s,
        public_ir.FunctionType.Orchestration,
    )
    program = public_ir.Program([child, graph, caller], "metadata_hoist", s)
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        result = passes.legalize_graph_boundary()(program)
        ir.assert_structural_equal(result, program)
        assert _diagnostics(result) == []
        # Independently test the post-hoist ABI, without requiring the pass to
        # hoist an expression it may safely retain inside the region.
        post_graph = public_ir.Function(
            "layer",
            [(out, public_ir.ParamDirection.InOut), profile, width],
            [tensor],
            public_ir.SeqStmts(
                [
                    public_ir.IfStmt(condition, task("post_then"), task("post_else"), [], s),
                    public_ir.ReturnStmt([out], s),
                ],
                s,
            ),
            s,
            public_ir.FunctionType.Graph,
        )
        post_caller = public_ir.Function(
            "main",
            [(out, public_ir.ParamDirection.InOut), profile],
            [tensor],
            public_ir.SeqStmts(
                [
                    public_ir.AssignStmt(
                        result_var := public_ir.Var("post_result", tensor, s),
                        public_ir.Call(public_ir.GlobalVar("layer"), [out, profile, dim], tensor, s),
                        s,
                    ),
                    public_ir.ReturnStmt([result_var], s),
                ],
                s,
            ),
            s,
            public_ir.FunctionType.Orchestration,
        )
        post_program = public_ir.Program([child, post_graph, post_caller], "metadata_posthoist", s)
        assert isinstance(dim, public_ir.Call)
        assert _diagnostics(post_program) == []
        legalized = passes.legalize_graph_boundary()(post_program)
        ir.assert_structural_equal(legalized, post_program)
        graph = legalized.get_function("layer")
        assert graph is not None
        assert any(isinstance(p.type, public_ir.ScalarType) for p in graph.params)
        assert _diagnostics(legalized) == []


@pytest.mark.parametrize("case", ["guarded_division", "guarded_view"])
def test_graph_hoisting_does_not_speculate_guarded_integer_division(case):
    with (
        passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH),
        pytest.raises(ValueError, match="guarded integer division"),
    ):
        passes.legalize_graph_boundary()(_lower(_program(case)))


def test_graph_rejects_a_recording_path_with_no_nodes():
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        normalized = _lower(_program("one_arm"))
        assert any("launches no tasks" in d.message for d in _diagnostics(normalized))
        with pytest.raises(ValueError, match="launches no tasks"):
            passes.legalize_graph_boundary()(normalized)


def test_graph_keeps_shaped_view_inside_its_guard():
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        normalized = _lower(_program("guarded_reshape"))
        legalized = passes.legalize_graph_boundary()(normalized)
        ir.assert_structural_equal(legalized, normalized)
        assert _diagnostics(legalized) == []
        graph = legalized.get_function("layer")
        assert graph is not None
        assert "tensor.view" in str(graph)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
