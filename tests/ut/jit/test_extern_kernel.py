# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for @pl.jit.extern — integrating hand-written C++ kernels via JIT.

An external kernel is a signature-only ``@pl.jit.extern`` stub backed by a
hand-written ``.cpp``. The specializer renders it into a header-only
``@pl.function(external_source=...)`` declaration; a ``core_type="mixed"``
kernel expands to an AIC member + AIV member + Group wrapper so the entry's
call lowers to a single MixedKernels submit.
"""

import importlib.util
import json
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto._external_source import EXTERNAL_INCLUDE_DIRS_ATTR
from pypto.backend.pto_backend import generate
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit.specializer import Specializer
from pypto.pypto_core import codegen, ir

_KERNEL_SRC = '#include <cstdint>\nextern "C" void kernel_entry(int64_t* args) { (void)args; }\n'


def _write_kernel(tmp_path: Path, name: str = "ext.cpp") -> Path:
    src = tmp_path / name
    src.write_text(_KERNEL_SRC)
    return src


def _specialize(entry, *args):
    """Run the JIT front-half (bind -> contexts -> specialize) and return source."""
    pn, _, tmeta, sv, sd, pfd = entry._bind_args(args, {})
    contexts = entry._build_contexts(tmeta, sv, sd, pfd)
    return Specializer(f"_jit_{entry.__name__}", contexts).specialize()


def _generate_external_config(entry, args, tmp_path: Path):
    """Specialize an all-external JIT graph and return its manifest and IR."""
    program = pl.parse(_specialize(entry, *args))
    assert isinstance(program, ir.Program)
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    files = generate(after, str(tmp_path / "codegen"), skip_ptoas=True)
    return files["kernel_config.py"], after


def test_mixed_extern_expands_to_group(tmp_path):
    cpp = _write_kernel(tmp_path)

    @pl.jit.extern(core_type="mixed", aic_source=cpp, aiv_source=cpp)
    def pa(
        a: pl.Tensor[[128, 128], pl.FP16],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP16]],
    ) -> pl.Tensor[[128, 128], pl.FP16]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = pa(a, out)
        return out

    a = torch.zeros(128, 128, dtype=torch.float16)
    out = torch.zeros(128, 128, dtype=torch.float16)
    src = _specialize(entry, a, out)

    # AIC member + AIV member + Group wrapper, entry dispatches the group.
    assert "type=pl.FunctionType.AIC, external_source=" in src
    assert "type=pl.FunctionType.AIV, external_source=" in src
    assert "type=pl.FunctionType.Group" in src
    assert "def pa_aic(self," in src
    assert "def pa_aiv(self," in src
    assert "self.pa_aic(a, out)" in src
    assert "self.pa_aiv(a, out)" in src
    assert "self.pa(a, out)" in src

    # The generated program parses and survives the full pass pipeline.
    program = pl.parse(src)
    assert isinstance(program, ir.Program)
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    names = {f.name for f in after.functions.values()}
    assert {"entry", "pa", "pa_aic", "pa_aiv"} <= names


def test_mixed_extern_can_dispatch_both_aiv_lanes(tmp_path):
    cpp = _write_kernel(tmp_path)

    @pl.jit.extern(
        core_type="mixed",
        aic_source=cpp,
        aiv_source=cpp,
        dual_aiv_dispatch=True,
    )
    def pa(
        a: pl.Tensor[[128, 128], pl.FP16],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP16]],
    ) -> pl.Tensor[[128, 128], pl.FP16]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = pa(a, out)
        return out

    a = torch.zeros(128, 128, dtype=torch.float16)
    out = torch.zeros(128, 128, dtype=torch.float16)
    src = _specialize(entry, a, out)
    assert 'attrs={"dual_aiv_dispatch": True}' in src

    program = pl.parse(src)
    assert isinstance(program, ir.Program)
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    aiv = after.get_function("pa_aiv")
    assert aiv is not None
    assert aiv.attrs.get("dual_aiv_dispatch") is True
    orch = after.get_function("entry")
    assert orch is not None
    generated = codegen.generate_orchestration(after, orch).code
    assert "MixedKernels mixed_0 = {0, 1, 1}" in generated


@pytest.mark.parametrize("core_type", ["aic", "aiv"])
def test_dual_aiv_dispatch_requires_mixed_extern(tmp_path, core_type):
    cpp = _write_kernel(tmp_path)
    with pytest.raises(ValueError, match="requires core_type='mixed'"):

        @pl.jit.extern(core_type=core_type, source=cpp, dual_aiv_dispatch=True)
        def k(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...


def test_single_core_extern(tmp_path):
    cpp = _write_kernel(tmp_path)

    @pl.jit.extern(core_type="aiv", source=cpp)
    def relu(
        a: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = relu(a, out)
        return out

    a = torch.zeros(64, 64, dtype=torch.float32)
    out = torch.zeros(64, 64, dtype=torch.float32)
    src = _specialize(entry, a, out)

    # One AIV declaration, no Group / member split.
    assert "type=pl.FunctionType.AIV, external_source=" in src
    assert "def relu(self," in src
    assert "FunctionType.Group" not in src
    assert "relu_aic" not in src
    assert "self.relu(a, out)" in src


def test_single_core_extern_include_dirs_reach_kernel_descriptor(tmp_path):
    cpp = _write_kernel(tmp_path)
    include_dir = tmp_path / "include"
    include_dir.mkdir()

    @pl.jit.extern(core_type="aiv", source=cpp, include_dirs=[include_dir])
    def relu(
        a: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = relu(a, out)
        return out

    a = torch.zeros(64, 64, dtype=torch.float32)
    out = torch.zeros(64, 64, dtype=torch.float32)
    config, program = _generate_external_config(entry, (a, out), tmp_path)

    relu_ir = program.get_function("relu")
    assert relu_ir is not None
    assert json.loads(relu_ir.attrs[EXTERNAL_INCLUDE_DIRS_ATTR]) == [str(include_dir.resolve())]
    assert f'"extra_include_dirs": {[str(include_dir.resolve())]!r}' in config


def test_mixed_extern_include_dirs_reach_both_kernel_descriptors(tmp_path):
    cpp = _write_kernel(tmp_path)
    first_include = tmp_path / "include_first"
    second_include = tmp_path / "include_second"
    first_include.mkdir()
    second_include.mkdir()
    expected = [str(first_include.resolve()), str(second_include.resolve())]

    @pl.jit.extern(
        core_type="mixed",
        aic_source=cpp,
        aiv_source=cpp,
        include_dirs=[first_include, second_include],
    )
    def pa(
        a: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = pa(a, out)
        return out

    a = torch.zeros(64, 64, dtype=torch.float32)
    out = torch.zeros(64, 64, dtype=torch.float32)
    config, program = _generate_external_config(entry, (a, out), tmp_path)

    for name in ("pa_aic", "pa_aiv"):
        member = program.get_function(name)
        assert member is not None
        assert json.loads(member.attrs[EXTERNAL_INCLUDE_DIRS_ATTR]) == expected
    assert config.count(f'"extra_include_dirs": {expected!r}') == 2


def test_extern_relative_include_dirs_resolve_from_stub_file(tmp_path):
    kernels = tmp_path / "kernels"
    includes = tmp_path / "headers"
    kernels.mkdir()
    includes.mkdir()
    (kernels / "ext.cpp").write_text(_KERNEL_SRC)
    module_path = tmp_path / "external_stub.py"
    module_path.write_text(
        "import pypto.language as pl\n"
        "@pl.jit.extern(core_type='aiv', source='kernels/ext.cpp', include_dirs=['headers'])\n"
        "def ext(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...\n"
    )
    spec = importlib.util.spec_from_file_location("_pypto_test_external_stub", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ext._external_aiv_source == str((kernels / "ext.cpp").resolve())
    assert module.ext._external_include_dirs == (str(includes.resolve()),)


def test_source_hash_tracks_cpp_edits(tmp_path):
    """Editing the .cpp changes the JIT cache key (the Python stub never does)."""
    cpp = _write_kernel(tmp_path)

    @pl.jit.extern(core_type="aiv", source=cpp)
    def k(
        a: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = k(a, out)
        return out

    h1 = entry._get_source_hash()
    cpp.write_text(_KERNEL_SRC + "\n// changed\n")
    entry._source_hash = None  # force recompute (normally a fresh interpreter)
    h2 = entry._get_source_hash()
    assert h1 != h2


def test_source_hash_tracks_external_launch_metadata(tmp_path):
    cpp = _write_kernel(tmp_path)

    def make_entry(dual_aiv_dispatch):
        @pl.jit.extern(
            core_type="mixed",
            aic_source=cpp,
            aiv_source=cpp,
            dual_aiv_dispatch=dual_aiv_dispatch,
        )
        def pa(
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]: ...

        @pl.jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            out = pa(a, out)
            return out

        return entry

    assert make_entry(False)._get_source_hash() != make_entry(True)._get_source_hash()


def test_source_hash_tracks_include_dir_metadata_and_headers(tmp_path):
    first_include = tmp_path / "first_include"
    second_include = tmp_path / "second_include"
    first_include.mkdir()
    second_include.mkdir()
    first_header = first_include / "config.h"
    first_header.write_text("constexpr int VALUE = 1;\n")
    (second_include / "config.h").write_text("constexpr int VALUE = 1;\n")
    cpp = _write_kernel(tmp_path)
    cpp.write_text("#include <config.h>\n" + _KERNEL_SRC)

    def make_entry(include_dir):
        @pl.jit.extern(core_type="aiv", source=cpp, include_dirs=[include_dir])
        def k(
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]: ...

        @pl.jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            out = k(a, out)
            return out

        return entry

    first_entry = make_entry(first_include)
    initial_hash = first_entry._get_source_hash()
    assert initial_hash != make_entry(second_include)._get_source_hash()

    first_header.write_text("constexpr int VALUE = 2;\n")
    assert initial_hash != first_entry._get_source_hash()


def test_source_hash_tracks_quoted_include_edits(tmp_path):
    include = tmp_path / "impl.cce"
    include.write_text("constexpr int VALUE = 1;\n")
    cpp = _write_kernel(tmp_path)
    cpp.write_text('#include "impl.cce"\n' + _KERNEL_SRC)

    @pl.jit.extern(core_type="aiv", source=cpp)
    def k(
        a: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = k(a, out)
        return out

    h1 = entry._get_source_hash()
    include.write_text("constexpr int VALUE = 2;\n")
    h2 = entry._get_source_hash()
    assert h1 != h2


def test_extern_bad_core_type(tmp_path):
    cpp = _write_kernel(tmp_path)
    with pytest.raises(ValueError, match="core_type must be"):

        @pl.jit.extern(core_type="cube", source=cpp)
        def k(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...


def test_extern_mixed_requires_both_sources(tmp_path):
    cpp = _write_kernel(tmp_path)
    with pytest.raises(ValueError, match="requires both aic_source"):

        @pl.jit.extern(core_type="mixed", aic_source=cpp)
        def k(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...


def test_extern_source_not_found(tmp_path):
    missing = tmp_path / "nope.cpp"
    with pytest.raises(ValueError, match="source file not found"):

        @pl.jit.extern(core_type="aiv", source=missing)
        def k(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...


def test_extern_include_dirs_rejects_scalar_path(tmp_path):
    cpp = _write_kernel(tmp_path)
    with pytest.raises(TypeError, match="must be a sequence of paths"):

        @pl.jit.extern(core_type="aiv", source=cpp, include_dirs=str(tmp_path))
        def k(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...


def test_extern_include_dirs_rejects_missing_directory(tmp_path):
    cpp = _write_kernel(tmp_path)
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="include directory not found"):

        @pl.jit.extern(core_type="aiv", source=cpp, include_dirs=[missing])
        def k(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
