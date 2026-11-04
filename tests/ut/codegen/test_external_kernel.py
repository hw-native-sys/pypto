# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for integrating hand-written external C++ kernels.

An external kernel is declared with ``@pl.function(type=AIC/AIV,
external_source=...)`` and a bare ``...`` body: the orchestration calls it like
any InCore kernel, but the compiler skips PyPTO codegen for its body and copies
the referenced ``.cpp`` into ``kernels/<ct>/<name>.cpp`` for the runtime to
compile. A mixed AIC+AIV kernel is expressed as a ``pl.FunctionType.Group`` of
one AIC + one AIV external member, reusing the Group -> MixedKernels path.
"""

from pathlib import Path

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.compile import compile as ir_compile
from pypto.language.parser.diagnostics import ParserSyntaxError, ParserTypeError

_KERNEL_SRC = '#include <cstdint>\nextern "C" void kernel_entry(int64_t* args) { (void)args; }\n'


def _write_kernel(tmp_path: Path) -> Path:
    src = tmp_path / "ext_kernel.cpp"
    src.write_text(_KERNEL_SRC)
    return src


def _build_mixed_program(cpp: Path):
    """AIC + AIV external members combined via a Group, driven by orchestration."""

    @pl.program
    class ExternalMixed:
        @pl.function(type=pl.FunctionType.AIC, external_source=cpp)
        def K_AIC(
            self,
            a: pl.Tensor[[128, 128], pl.FP16],
            out: pl.Out[pl.Tensor[[128, 128], pl.FP16]],
        ) -> pl.Tensor[[128, 128], pl.FP16]: ...

        @pl.function(type=pl.FunctionType.AIV, external_source=cpp)
        def K_AIV(
            self,
            a: pl.Tensor[[128, 128], pl.FP16],
            out: pl.Out[pl.Tensor[[128, 128], pl.FP16]],
        ) -> pl.Tensor[[128, 128], pl.FP16]: ...

        @pl.function(type=pl.FunctionType.Group)
        def K(
            self,
            a: pl.Tensor[[128, 128], pl.FP16],
            out: pl.Out[pl.Tensor[[128, 128], pl.FP16]],
        ) -> pl.Tensor[[128, 128], pl.FP16]:
            r = self.K_AIC(a, out)
            self.K_AIV(a, out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.FP16],
            out: pl.Out[pl.Tensor[[128, 128], pl.FP16]],
        ) -> pl.Tensor[[128, 128], pl.FP16]:
            out = self.K(a, out)
            return out

    return ExternalMixed


def test_external_source_attr_and_empty_body(tmp_path):
    """The external kernel parses to an AIC/AIV function carrying external_source."""
    cpp = _write_kernel(tmp_path)
    program = _build_mixed_program(cpp)
    funcs = {f.name: f for f in program.functions.values()}

    aic = funcs["K_AIC"]
    aiv = funcs["K_AIV"]
    assert aic.func_type == pl.FunctionType.AIC
    assert aiv.func_type == pl.FunctionType.AIV
    assert dict(aic.attrs)["external_source"] == str(cpp.resolve())
    assert dict(aiv.attrs)["external_source"] == str(cpp.resolve())
    # DSL functions carry no external_source attr.
    assert "external_source" not in dict(funcs["K"].attrs)
    assert "external_source" not in dict(funcs["main"].attrs)


def test_external_mixed_kernel_codegen(tmp_path):
    """End-to-end: manifest + orchestration submit + copied kernel sources."""
    cpp = _write_kernel(tmp_path)
    program = _build_mixed_program(cpp)

    out_dir = tmp_path / "gen"
    ir_compile(program, skip_ptoas=False, platform="a2a3", output_dir=str(out_dir))

    # 1. External kernels are referenced in place — not copied or regenerated —
    #    so the entry .cpp keeps its sibling files (relative includes resolve).
    assert not (out_dir / "kernels" / "aic" / "K_AIC.cpp").exists()
    assert not (out_dir / "kernels" / "aiv" / "K_AIV.cpp").exists()

    # 2. The manifest lists both kernels with consecutive func_ids, the right
    #    core types, and sources pointing at the original hand-written .cpp.
    config = (out_dir / "kernel_config.py").read_text()
    assert '"func_id": 0, "name": "K_AIC"' in config
    assert '"func_id": 1, "name": "K_AIV"' in config
    assert '"core_type": "aic"' in config
    assert '"core_type": "aiv"' in config
    assert config.count('"external": True') == 2
    # Both members reference the same original source at its absolute path.
    assert config.count(repr(str(cpp))) == 2

    # 3. Orchestration dispatches the pair as a single MixedKernels submit
    #    (aic func_id 0, aiv func_id 1) with the declared arg directions.
    orch = (out_dir / "orchestration" / "main.cpp").read_text()
    assert "MixedKernels mixed_0 = {0, 1, INVALID_KERNEL_ID}" in orch
    assert "rt_submit_task(mixed_0" in orch
    assert "add_input(ext_a)" in orch
    assert "add_output(ext_out)" in orch


def test_all_external_graph_emits_manifest_when_ptoas_is_skipped(tmp_path):
    cpp = _write_kernel(tmp_path)
    program = _build_mixed_program(cpp)
    out_dir = tmp_path / "skip_ptoas"

    ir_compile(program, skip_ptoas=True, platform="a2a3", output_dir=str(out_dir))

    config = (out_dir / "kernel_config.py").read_text()
    assert config.count('"external": True') == 2
    assert not (out_dir / "kernels").exists()


def test_external_source_requires_aic_or_aiv(tmp_path):
    """external_source on a non-AIC/AIV function is rejected with a clear error."""
    cpp = _write_kernel(tmp_path)
    with pytest.raises(ParserTypeError, match=r"external_source is only valid on FunctionType\.AIC"):

        @pl.program
        class BadType:
            @pl.function(type=pl.FunctionType.InCore, external_source=cpp)
            def K(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]: ...

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                out = self.K(a, out)
                return out


def test_external_kernel_requires_empty_body(tmp_path):
    """An external kernel with a non-``...`` body is rejected."""
    cpp = _write_kernel(tmp_path)
    with pytest.raises(ParserSyntaxError, match="must have an empty"):

        @pl.program
        class BadBody:
            @pl.function(type=pl.FunctionType.AIV, external_source=cpp)
            def K(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                out = pl.store(pl.load(a, [0, 0], [16, 16]), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                out = self.K(a, out)
                return out


def test_external_source_not_found(tmp_path):
    """A non-existent external_source path fails fast at declaration time."""
    missing = tmp_path / "does_not_exist.cpp"
    with pytest.raises(ValueError, match="file not found"):

        @pl.program
        class Missing:
            @pl.function(type=pl.FunctionType.AIV, external_source=missing)
            def K(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]: ...

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                out = self.K(a, out)
                return out


def _external_contract_program(
    cpp: Path, indices: list[int] | None, *, result_count: int = 2, invalid: str | None = None
) -> ir.Program:
    """Construct raw IR to test the external declaration boundary independently of JIT."""
    span = ir.Span.unknown()
    tensor = ir.TensorType([1, 32], DataType.FP32)
    params = [ir.Var(name, tensor, span) for name in ("first", "second", "third")]
    attrs: dict[str, object] = {"external_source": str(cpp)}
    if indices is not None:
        returned = [params[i] for i in indices]
        if invalid == "foreign":
            returned[0] = ir.Var("foreign", tensor, span)
        attrs["external_return_params"] = returned
    types = [tensor] * result_count
    if invalid == "dtype":
        types[0] = ir.TensorType([1, 32], DataType.FP16)
    if invalid == "shape":
        types[0] = ir.TensorType([1, 64], DataType.FP32)
    function = ir.Function(
        "producer",
        [(p, ir.ParamDirection.Out) for p in params],
        types,
        ir.SeqStmts([], span),
        span,
        type=ir.FunctionType.AIV,
        attrs=attrs,
    )
    return ir.Program([function], "ExternalContract", span)


@pytest.mark.parametrize("indices", [[1, 2], [2, 0], [1, 1], []])
def test_external_contract_survives_ssa_and_round_trip(tmp_path, indices):
    """Parameter references must follow SSA renaming and survive persisted pass dumps."""
    before = _external_contract_program(_write_kernel(tmp_path), indices, result_count=len(indices))
    after = passes.convert_to_ssa()(before)
    restored = ir.deserialize(ir.serialize(after))
    reparsed = pl.parse(str(after))
    for program in (after, restored, reparsed):
        assert isinstance(program, ir.Program)
        ir.assert_structural_equal(after, program)
        func = program.get_function("producer")
        assert func is not None
        returned = func.attrs["external_return_params"]
        assert len(returned) == len(indices)
        assert all(value is func.params[index] for value, index in zip(returned, indices))
        passes.normalize_return_order()(program)


@pytest.mark.parametrize(
    ("indices", "invalid", "message"),
    [
        (None, None, "returned parameter contract has 3 entries"),
        ([1], None, "returned parameter contract has 1 entries"),
        ([1, 2], "foreign", "must reference its own parameter"),
        ([1, 2], "dtype", "same tensor dtype and shape"),
        ([1, 2], "shape", "different shape"),
    ],
)
def test_external_contract_rejects_ambiguous_or_invalid_returns(tmp_path, indices, invalid, message):
    program = _external_contract_program(_write_kernel(tmp_path), indices, invalid=invalid)
    if invalid != "foreign":
        with pytest.raises(ValueError, match=message):
            passes.normalize_return_order()(program)
    # The property verifier must validate declarations instead of exempting all
    # external functions just because they have no executable ReturnStmt.
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.ReturnParamsExplicit)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, program)
    assert diagnostics
    assert any(message in diagnostic.message for diagnostic in diagnostics)


def test_external_signature_only_full_output_contract(tmp_path):
    """Raw signature-only IR retains the ordered-all-outputs convention."""
    program = _external_contract_program(_write_kernel(tmp_path), None, result_count=3)
    passes.normalize_return_order()(program)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.ReturnParamsExplicit)
    assert not passes.PropertyVerifierRegistry.verify(properties, program)


@pytest.mark.parametrize("contract", [0, "output", [0], [True]])
def test_external_contract_requires_parameter_references(tmp_path, contract):
    span = ir.Span.unknown()
    tensor = ir.TensorType([1, 32], DataType.FP32)
    output = ir.Var("output", tensor, span)
    with pytest.raises(TypeError, match="parameter Vars|expected Var"):
        ir.Function(
            "producer",
            [(output, ir.ParamDirection.Out)],
            [tensor],
            ir.SeqStmts([], span),
            span,
            type=ir.FunctionType.AIV,
            attrs={"external_source": str(_write_kernel(tmp_path)), "external_return_params": contract},
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
