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
from pypto import CacheConfig
from pypto._external_source import EXTERNAL_INCLUDE_DIRS_ATTR
from pypto.backend import BackendType
from pypto.backend.pto_backend import generate
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit import JITFunction
from pypto.jit.foreign import ExternalKernelSpec
from pypto.jit.specializer import Specializer
from pypto.pypto_core import codegen, ir
from pypto.runtime import ExecutionMode, RunConfig

_KERNEL_SRC = '#include <cstdint>\nextern "C" void kernel_entry(int64_t* args) { (void)args; }\n'


def _write_kernel(tmp_path: Path, name: str = "ext.cpp") -> Path:
    src = tmp_path / name
    src.write_text(_KERNEL_SRC)
    return src


def _specialize(entry, *args):
    """Run the JIT front-half (bind -> contexts -> specialize) and return source."""
    pn, _, tmeta, sd, cx, pfd = entry._bind_args(args, {})
    contexts = entry._build_contexts(tmeta, sd, cx, pfd)
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


class _ForeignKernel:
    """不带 PyPTO 注解，且不强制依赖第三方包的提供方。"""

    def __init__(self, func, source, access, *, core="aiv"):
        self.jit = JITFunction(
            func,
            func_type="extern",
            external_builder=lambda ctx: ExternalKernelSpec(
                source=str(source),
                core_type=core,
                parameter_access=access,
                dual_aiv_dispatch=core == "mixed",
            ),
        )

    def __pypto_jit__(self):
        return self.jit


@pytest.mark.parametrize("core", ["aiv", "aic", "mixed"])
def test_foreign_call_infers_outputs_and_emits_task(tmp_path, core):
    def foreign_body(x, out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"x": "read", "out": "write"}, core=core)

    @pl.jit
    def entry(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        leaf(x, out=out)
        return out

    args = (torch.zeros(1, 32), torch.empty(1, 32))
    src = _specialize(entry, *args)
    assert "out_v1 = self.foreign_body(x, out)" in src
    assert "return out_v1" in src
    config, after = _generate_external_config(entry, args, tmp_path)
    assert "foreign_body" in config
    orch = after.get_function("entry")
    assert orch is not None
    generated = codegen.generate_orchestration(after, orch).code
    assert "add_output(" in generated
    if core == "mixed":
        assert "MixedKernels mixed_0 = {0, 1, 1}" in generated


def test_foreign_inout_call_is_loop_carried(tmp_path):
    def foreign_body(out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "read_write"})

    @pl.jit
    def entry(out: pl.InOut[pl.Tensor]) -> pl.Tensor:
        for _ in pl.range(0, 3):
            leaf(out)
        return out

    args = (torch.zeros(1, 32),)
    src = _specialize(entry, *args)
    assert "out = self.foreign_body(out)" in src
    _, after = _generate_external_config(entry, args, tmp_path)
    orch = after.get_function("entry")
    assert orch is not None
    generated = codegen.generate_orchestration(after, orch).code
    assert "add_inout(" in generated


def test_foreign_multiple_outputs_are_rebound(tmp_path):
    def foreign_body(x, y):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"x": "write", "y": "write"})

    @pl.jit
    def entry(x: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]) -> tuple[pl.Tensor, pl.Tensor]:
        leaf(x, y)
        leaf(x, y)
        return x, y

    args = (torch.empty(1, 32), torch.empty(1, 32))
    src = _specialize(entry, *args)
    assert "x_v1, y_v1 = self.foreign_body(x, y)" in src
    assert "x_v2, y_v2 = self.foreign_body(x_v1, y_v1)" in src
    _generate_external_config(entry, args, tmp_path)


def test_foreign_scoped_writeback_uses_active_alias(tmp_path):
    def foreign_body(out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "read_write"})

    @pl.jit
    def entry(out: pl.InOut[pl.Tensor], enabled: pl.Scalar[pl.INT32]) -> pl.Tensor:
        leaf(out)
        for _ in pl.range(0, 3):
            leaf(out)
        if enabled > 0:
            leaf(out)
        return out

    args = (torch.zeros(1, 32), 1)
    source = _specialize(entry, *args)
    assert source.count("out_v1 = self.foreign_body(out_v1)") == 2
    assert "return out_v1" in source
    _generate_external_config(entry, args, tmp_path)


def test_foreign_void_inline_writebacks_propagate_transitively(tmp_path):
    def foreign_body(out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "read_write"})

    @pl.jit.inline(auto_scope=False)
    def inner(out: pl.Tensor):
        leaf(out)

    @pl.jit.inline(auto_scope=False)
    def outer(out: pl.Tensor):
        inner(out)

    @pl.jit
    def entry(out: pl.InOut[pl.Tensor]) -> pl.Tensor:
        outer(out)
        leaf(out)
        return out

    args = (torch.zeros(1, 32),)
    source = _specialize(entry, *args)
    assert "out_v1 = self.outer(out)" in source
    assert "out_v2 = self.foreign_body(out_v1)" in source
    _generate_external_config(entry, args, tmp_path)


def test_foreign_void_inline_loop_and_branch_writebacks(tmp_path):
    def foreign_body(out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "read_write"})

    @pl.jit.inline(auto_scope=False)
    def helper(out: pl.Tensor, enabled: pl.Scalar[pl.INT32]):
        for _ in pl.range(0, 3):
            if enabled > 0:
                leaf(out)

    @pl.jit
    def entry(out: pl.InOut[pl.Tensor], enabled: pl.Scalar[pl.INT32]) -> pl.Tensor:
        leaf(out)
        for _ in pl.range(0, 2):
            helper(out, enabled)
        leaf(out)
        return out

    args = (torch.zeros(1, 32), 1)
    source = _specialize(entry, *args)
    assert "out_v1 = self.helper(out_v1, enabled)" in source
    assert "out_v2 = self.foreign_body(out_v1)" in source
    _generate_external_config(entry, args, tmp_path)


def test_foreign_explicit_inline_return_keeps_value_semantics(tmp_path):
    def foreign_body(out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "read_write"})

    @pl.jit.inline(auto_scope=False)
    def helper(out: pl.Tensor) -> pl.Tensor:
        leaf(out)
        return out

    @pl.jit
    def entry(out: pl.InOut[pl.Tensor]) -> pl.Tensor:
        out = helper(out)
        leaf(out)
        return out

    args = (torch.zeros(1, 32),)
    source = _specialize(entry, *args)
    assert "out_v1 = self.helper(out)" in source
    assert "out_v2 = self.foreign_body(out_v1)" in source
    _generate_external_config(entry, args, tmp_path)


def test_foreign_void_inline_multiple_outputs_keep_parameter_order(tmp_path):
    def foreign_body(x, y):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"x": "write", "y": "write"})

    @pl.jit.inline(auto_scope=False)
    def helper(x: pl.Tensor, y: pl.Tensor) -> None:
        leaf(y=y, x=x)

    @pl.jit
    def entry(x: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]) -> tuple[pl.Tensor, pl.Tensor]:
        helper(y=y, x=x)
        leaf(x, y)
        return x, y

    args = (torch.empty(1, 32), torch.empty(1, 32))
    source = _specialize(entry, *args)
    assert "x_v1, y_v1 = self.helper(x, y)" in source
    assert "x_v2, y_v2 = self.foreign_body(x_v1, y_v1)" in source
    _generate_external_config(entry, args, tmp_path)


def test_foreign_void_inline_parameter_rebind_requires_explicit_return(tmp_path):
    def foreign_body(out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "read_write"})

    @pl.jit.inline(auto_scope=False)
    def helper(out: pl.Tensor):
        leaf(out)
        out = pl.create_tensor([1, 32], pl.FP32)

    @pl.jit
    def entry(out: pl.InOut[pl.Tensor]):
        helper(out)

    with pytest.raises(TypeError, match="helper.*rebinds.*out.*return those tensors explicitly"):
        _specialize(entry, torch.empty(1, 32))


def test_foreign_untyped_scalar_reports_parameter_before_rendering(tmp_path):
    def foreign_body(out, scale):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "write"})

    @pl.jit
    def entry(out: pl.Out[pl.Tensor]):
        leaf(out, 1.0)

    with pytest.raises(ValueError, match="foreign_body.*cannot infer.*scale"):
        _specialize(entry, torch.empty(1, 32))


def test_foreign_cache_tracks_generated_cpp(tmp_path, monkeypatch):
    def foreign_body(out):
        pass

    cpp = _write_kernel(tmp_path)
    leaf = _ForeignKernel(foreign_body, cpp, {"out": "write"})

    @pl.jit
    def entry(out: pl.Out[pl.Tensor]) -> pl.Tensor:
        leaf(out)
        return out

    builds = []

    def compile_stub(*args, **kwargs):
        artifact = object()
        builds.append(artifact)
        return artifact

    monkeypatch.setattr(entry, "_compile", compile_stub)
    config = RunConfig(
        backend_type=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        cache_config=CacheConfig(enabled=False),
    )
    out = torch.empty(1, 32)
    first = entry.compile(out, config=config)
    assert entry.compile(out, config=config) is first
    cpp.write_text(_KERNEL_SRC + "\n// changed producer implementation\n")
    assert entry.compile(out, config=config) is not first
    assert len(builds) == 2


def test_foreign_scalar_argument_inherits_caller_dtype(tmp_path):
    def foreign_body(out, scale):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "write"})

    @pl.jit
    def entry(out: pl.Out[pl.Tensor], scale: pl.Scalar[pl.FP32]) -> pl.Tensor:
        leaf(scale=scale, out=out)
        return out

    source = _specialize(entry, torch.empty(1, 32), 2.5)
    assert "scale: pl.Scalar[pl.FP32]" in source
    _generate_external_config(entry, (torch.empty(1, 32), 2.5), tmp_path)


def test_foreign_mx_scale_storage_view_preserves_dtype_and_nd_layout(tmp_path):
    def foreign_body(scale, out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"scale": "read", "out": "write"})

    @pl.jit
    def entry(scale: pl.Tensor[[8, 16], pl.FP8E8M0, pl.MX_B_NN], out: pl.Out[pl.Tensor]):
        backing = pl.tensor.view(scale, [1, 128], layout=pl.ND)
        leaf(backing, out)
        return out

    args = (torch.empty(8, 16, dtype=torch.float8_e8m0fnu), torch.empty(1, 128))
    source = _specialize(entry, *args)
    assert "scale: pl.Tensor[[1, 128], pl.FP8E8M0]" in source
    _generate_external_config(entry, args, tmp_path)


def test_foreign_scalar_argument_inherits_loop_index_dtype(tmp_path):
    def foreign_body(out, page):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "write"})

    @pl.jit
    def entry(out: pl.Out[pl.Tensor]) -> pl.Tensor:
        for page in pl.parallel(2):
            leaf(out, page)
        return out

    source = _specialize(entry, torch.empty(1, 32))
    assert "page: pl.Scalar[pl.INDEX]" in source
    _generate_external_config(entry, (torch.empty(1, 32),), tmp_path)


def test_foreign_rebound_loop_index_does_not_keep_index_dtype(tmp_path):
    def foreign_body(out, page):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "write"})

    @pl.jit
    def entry(out: pl.Out[pl.Tensor], scale: pl.Scalar[pl.FP32]) -> pl.Tensor:
        for page in pl.parallel(2):
            page = scale
            leaf(out, page)
        return out

    with pytest.raises(ValueError, match="page"):
        _specialize(entry, torch.empty(1, 32), 0.5)


def test_foreign_different_call_shapes_are_specialized(tmp_path):
    def foreign_body(out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "write"})

    @pl.jit
    def entry(x: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
        leaf(x)
        leaf(y)

    args = (torch.empty(1, 32), torch.empty(1, 64))
    _, _, tensor_meta, scalars, constexpr, dynamic = entry._bind_args(args, {})
    contexts = entry._build_contexts(tensor_meta, scalars, constexpr, dynamic)
    foreign = [ctx for ctx in contexts if ctx.external_parameter_access is not None]
    assert {ctx.tensor_meta["out"].shape for ctx in foreign} == {(1, 32), (1, 64)}
    assert len({ctx.func_name for ctx in foreign}) == 2
    assert pl.parse(Specializer("_jit_foreign_shapes", contexts).specialize()) is not None


def test_foreign_void_call_cannot_be_used_as_value(tmp_path):
    def foreign_body(out):
        pass

    leaf = _ForeignKernel(foreign_body, _write_kernel(tmp_path), {"out": "write"})

    @pl.jit
    def entry(out: pl.Out[pl.Tensor]):
        out = leaf(out)
        return out

    with pytest.raises(TypeError, match="standalone call statement"):
        entry.specialize(torch.empty(1, 32))


def test_foreign_hook_rejects_invalid_result():
    class BadHook:
        def __pypto_jit__(self):
            return None

    leaf = BadHook()

    @pl.jit
    def entry(x: pl.Tensor):
        leaf(x)

    with pytest.raises(TypeError, match="must return a JITFunction"):
        entry.specialize(torch.zeros(1, 32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
