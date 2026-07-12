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
from pypto.ir.compile import compile as ir_compile

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
    with pytest.raises(Exception, match="external_source is only valid on FunctionType.AIC"):

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
    with pytest.raises(Exception, match="must have an empty"):

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
