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
import re
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto._external_source import EXTERNAL_INCLUDE_DIRS_ATTR
from pypto.backend import BackendType
from pypto.backend.pto_backend import generate
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit.specializer import Specializer
from pypto.pypto_core import codegen, ir
from pypto.runtime import RunConfig

_KERNEL_SRC = '#include <cstdint>\nextern "C" void kernel_entry(int64_t* args) { (void)args; }\n'
_PAIR_ROWS = 16
_PAIR_COLS = 16
_PAIR_DTYPE = pl.INT8
_TensorAlias = pl.Tensor
_TupleValueAlias = pl.Tuple
_OutputsAlias = tuple[pl.Tensor[[16], pl.INT8], pl.Tensor[[1], pl.FP32]]
_IncompleteOutputsAlias = tuple[pl.Tensor, pl.Tensor]
_DistributedOutputsAlias = tuple[pld.DistributedTensor[[16], pl.INT8], pld.DistributedTensor[[1], pl.FP32]]
_RETURN_ROWS = 128


def _write_kernel(tmp_path: Path, name: str = "ext.cpp") -> Path:
    src = tmp_path / name
    src.write_text(_KERNEL_SRC)
    return src


def _load_jit_module(path: Path):
    """Load a generated JIT fixture with a real source file for specialization."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.mark.parametrize("core_type", ["aic", "aiv", "mixed"])
def test_signature_only_extern_tuple_outputs_feed_later_kernels(tmp_path, core_type):
    """Two declared results must survive specialization without a Python return (#3249)."""
    cpp = _write_kernel(tmp_path)

    @pl.jit.extern(core_type=core_type, source=cpp, aic_source=cpp, aiv_source=cpp)
    def pair(
        a: pl.Tensor,
        y: pl.Out[pl.Tensor],
        z: pl.Out[pl.Tensor],
    ) -> tuple[pl.Tensor[[_PAIR_ROWS, _PAIR_COLS], _PAIR_DTYPE], pl.Tensor[[_PAIR_ROWS, 1], pl.FP32]]: ...

    @pl.jit.extern(core_type="aiv", source=cpp)
    def consume_y(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...

    @pl.jit.extern(core_type="aiv", source=cpp)
    def consume_z(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...

    @pl.jit
    def entry(
        a: pl.Tensor,
        y: pl.Out[pl.Tensor],
        z: pl.Out[pl.Tensor],
        y_result: pl.Out[pl.Tensor],
        z_result: pl.Out[pl.Tensor],
    ):
        quantized, scales = pair(a, y, z)
        y_result = consume_y(quantized, y_result)
        z_result = consume_z(scales, z_result)
        return y_result, z_result

    # Different shapes and dtypes expose loss, duplication, or reordering of a
    # tuple element. Module constants in the declared types must also be folded.
    a = torch.empty(_PAIR_ROWS, _PAIR_COLS, dtype=torch.float32)
    y = torch.empty(_PAIR_ROWS, _PAIR_COLS, dtype=torch.int8)
    z = torch.empty(_PAIR_ROWS, 1, dtype=torch.float32)
    config, program = _generate_external_config(
        entry, (a, y, z, torch.empty_like(y), torch.empty_like(z)), tmp_path
    )
    assert config

    expected = [ir.TensorType([16, 16], pl.INT8), ir.TensorType([16, 1], pl.FP32)]
    names = ["pair", "entry"]
    if core_type == "mixed":
        names += ["pair_aic", "pair_aiv"]
    for name in names:
        func = program.get_function(name)
        assert func is not None
        assert len(func.return_types) == 2
        for actual, declared in zip(func.return_types, expected, strict=True):
            ir.assert_structural_equal(actual, declared)

    # The consumer input types verify propagation through freshly unpacked
    # names, rather than only reusing metadata on the original Out arguments.
    for name, declared in zip(["consume_y", "consume_z"], expected, strict=True):
        func = program.get_function(name)
        assert func is not None
        actual = func.params[0].type
        assert isinstance(actual, ir.TensorType)
        # The default pipeline adds storage metadata (MemRef) to parameters.
        # This regression checks the logical type propagated through unpacking.
        assert actual.dtype == declared.dtype
        assert len(actual.shape) == len(declared.shape)
        for actual_dim, declared_dim in zip(actual.shape, declared.shape, strict=True):
            ir.assert_structural_equal(actual_dim, declared_dim)


def test_extern_postponed_tensor_alias(tmp_path):
    cpp = _write_kernel(tmp_path)

    @pl.jit.extern(core_type="aiv", source=cpp)
    def pair(
        y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]
    ) -> "tuple[_TensorAlias[[16], pl.INT8], _TensorAlias[[1], pl.FP32]]": ...

    @pl.jit
    def entry(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]):
        y, z = pair(y, z)
        return y, z

    _, program = _generate_external_config(
        entry, (torch.empty(16, dtype=torch.int8), torch.empty(1)), tmp_path
    )
    pair_ir = program.get_function("pair")
    assert pair_ir is not None
    assert len(pair_ir.return_types) == 2
    for typ, dtype in zip(pair_ir.return_types, [pl.INT8, pl.FP32], strict=True):
        assert isinstance(typ, ir.TensorType)
        assert typ.dtype == dtype


@pytest.mark.parametrize("postponed", [False, True])
def test_extern_complete_tuple_alias_preserves_multiple_results(tmp_path, postponed):
    cpp = _write_kernel(tmp_path)
    if postponed:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def pair(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]) -> "_OutputsAlias": ...
    else:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def pair(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]) -> _OutputsAlias: ...

    @pl.jit
    def entry(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]):
        y, z = pair(y, z)
        return y, z

    _, program = _generate_external_config(
        entry, (torch.empty(16, dtype=torch.int8), torch.empty(1)), tmp_path
    )
    pair_ir = program.get_function("pair")
    assert pair_ir is not None
    assert len(pair_ir.return_types) == 2
    ir.assert_structural_equal(pair_ir.return_types[0], ir.TensorType([16], pl.INT8))
    ir.assert_structural_equal(pair_ir.return_types[1], ir.TensorType([1], pl.FP32))


@pytest.mark.parametrize("postponed", [False, True])
def test_extern_distributed_tuple_alias_preserves_result_kinds(tmp_path, postponed):
    cpp = _write_kernel(tmp_path)
    if postponed:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def pair(
            y: pl.Out[pld.DistributedTensor], z: pl.Out[pld.DistributedTensor]
        ) -> "_DistributedOutputsAlias": ...
    else:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def pair(
            y: pl.Out[pld.DistributedTensor], z: pl.Out[pld.DistributedTensor]
        ) -> _DistributedOutputsAlias: ...

    @pl.jit
    def entry(y: pl.Out[pld.DistributedTensor], z: pl.Out[pld.DistributedTensor]):
        y, z = pair(y, z)
        return y, z

    _, program = _generate_external_config(
        entry, (torch.empty(16, dtype=torch.int8), torch.empty(1)), tmp_path
    )
    pair_ir = program.get_function("pair")
    assert pair_ir is not None
    assert len(pair_ir.return_types) == 2
    assert all(isinstance(typ, ir.DistributedTensorType) for typ in pair_ir.return_types)


@pytest.mark.parametrize("postponed", [False, True])
@pytest.mark.parametrize("tuple_value", [False, True])
def test_extern_constexpr_constructor_return(tmp_path, postponed, tuple_value):
    _write_kernel(tmp_path)
    # The DSL parser accepts legacy constructor notation that Python's static
    # type grammar does not. Load a real source module to exercise inspect too.
    module_path = tmp_path / "constructor_stub.py"
    annotation = "tuple[pl.Tensor([_RETURN_ROWS], pl.INT8), pl.Tensor([1], pl.FP32)]"
    if tuple_value:
        annotation = "pl.Tuple([pl.Tensor([_RETURN_ROWS], pl.INT8), pl.Tensor([1], pl.FP32)])"
    if postponed:
        annotation = repr(annotation)
    module_path.write_text(
        "import pypto.language as pl\n"
        "_RETURN_ROWS = 128\n"
        "@pl.jit.extern(core_type='aiv', source='ext.cpp')\n"
        "def pair(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor], _RETURN_ROWS: pl.constexpr)"
        f" -> {annotation}: ...\n"
        "@pl.jit\n"
        "def entry(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]):\n"
        "    y, z = pair(y, z, 16)\n"
        "    return y, z\n"
    )
    spec = importlib.util.spec_from_file_location("_extern_constructor_stub", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = (torch.empty(16, dtype=torch.int8), torch.empty(1))
    if tuple_value:
        with pytest.raises(ValueError, match="single tuple-valued result"):
            _specialize(module.entry, *args)
        return
    _, program = _generate_external_config(module.entry, args, tmp_path)
    pair_ir = program.get_function("pair")
    assert pair_ir is not None
    assert len(pair_ir.return_types) == 2
    ir.assert_structural_equal(pair_ir.return_types[0], ir.TensorType([16], pl.INT8))
    ir.assert_structural_equal(pair_ir.return_types[1], ir.TensorType([1], pl.FP32))


def test_extern_factory_return_snapshot_and_cache_identity(tmp_path):
    cpp = _write_kernel(tmp_path)

    def factory(rows):
        # rows exists only in the evaluated annotation, not in a closure cell.
        @pl.jit.extern(core_type="aiv", source=cpp)
        def kernel(out: pl.Out[pl.Tensor]) -> pl.Tensor[[rows], pl.FP32]: ...

        @pl.jit
        def entry(out: pl.Out[pl.Tensor]):
            out = kernel(out)
            return out

        return kernel, entry

    kernel16, entry16 = factory(16)
    kernel32, entry32 = factory(32)
    kernel16_again, _ = factory(16)
    assert "rows" not in kernel16._func.__code__.co_freevars
    assert kernel16._get_source_hash() != kernel32._get_source_hash()
    assert kernel16._persistent_source_digest() != kernel32._persistent_source_digest()
    assert kernel16._get_source_hash() == kernel16_again._get_source_hash()
    assert kernel16._persistent_source_digest() == kernel16_again._persistent_source_digest()
    for rows, entry in ((16, entry16), (32, entry32)):
        _, program = _generate_external_config(entry, (torch.empty(rows),), tmp_path / str(rows))
        kernel = program.get_function("kernel")
        assert kernel is not None
        ir.assert_structural_equal(kernel.return_types[0], ir.TensorType([rows], pl.FP32))


@pytest.mark.parametrize("postponed", [False, True])
def test_extern_constexpr_return_overrides_ambient_value(tmp_path, postponed):
    cpp = _write_kernel(tmp_path)

    def factory(columns):
        if postponed:

            @pl.jit.extern(core_type="aiv", source=cpp)
            def kernel(
                out: pl.Out[pl.Tensor], _RETURN_ROWS: pl.constexpr
            ) -> "pl.Tensor[[_RETURN_ROWS, 3], pl.FP32]": ...
        else:

            @pl.jit.extern(core_type="aiv", source=cpp)
            def kernel(
                out: pl.Out[pl.Tensor], _RETURN_ROWS: pl.constexpr
            ) -> pl.Tensor[[_RETURN_ROWS, columns], pl.FP32]: ...

        return kernel

    kernel = factory(3)

    @pl.jit
    def entry(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]):
        y = kernel(y, 16)
        z = kernel(z, 32)
        return y, z

    # Neither variant may reuse the eager ambient _RETURN_ROWS=128. In the
    # eager case, the independent factory-only column count must also survive.
    _, program = _generate_external_config(entry, (torch.empty(16, 3), torch.empty(32, 3)), tmp_path)
    kernels = [fn for fn in program.functions.values() if "external_source" in fn.attrs]
    assert len(kernels) == 2
    shapes = set()
    for fn in kernels:
        typ = fn.return_types[0]
        assert isinstance(typ, ir.TensorType)
        shape = []
        for dim in typ.shape:
            assert isinstance(dim, ir.ConstInt)
            shape.append(dim.value)
        shapes.add(tuple(shape))
    assert shapes == {(16, 3), (32, 3)}


@pytest.mark.parametrize("alias", [False, True])
def test_extern_tuple_value_is_rejected_without_changing_its_contract(tmp_path, alias):
    cpp = _write_kernel(tmp_path)
    if alias:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def pair(
            y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]
        ) -> "_TupleValueAlias[pl.Tensor[[16], pl.INT8], pl.Tensor[[1], pl.FP32]]": ...
    else:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def pair(
            y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]
        ) -> pl.Tuple[pl.Tensor[[16], pl.INT8], pl.Tensor[[1], pl.FP32]]: ...

    @pl.jit
    def entry(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]):
        y, z = pair(y, z)
        return y, z

    with pytest.raises(ValueError, match="single tuple-valued result"):
        _specialize(entry, torch.empty(16, dtype=torch.int8), torch.empty(1))


@pytest.mark.parametrize("tuple_name", ["tuple", "typing.Tuple", "pl.Tuple"])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("postponed", [False, True])
@pytest.mark.parametrize(
    "core_type, output", [("aiv", True), ("aiv", False), ("aic", False), ("mixed", False)]
)
def test_extern_empty_tuple_contract(tmp_path, tuple_name, alias, postponed, core_type, output):
    _write_kernel(tmp_path)
    annotation = "Empty" if alias else f"{tuple_name}[()]"
    if postponed:
        annotation = repr(annotation)
    parameter = "pl.Out[pl.Tensor]" if output else "pl.Tensor"
    sources = "aic_source='ext.cpp', aiv_source='ext.cpp'" if core_type == "mixed" else "source='ext.cpp'"
    module_path = tmp_path / "empty_tuple_stub.py"
    module_path.write_text(
        "import typing\n"
        "import pypto.language as pl\n"
        f"Empty = {tuple_name}[()]\n"
        f"@pl.jit.extern(core_type={core_type!r}, {sources})\n"
        f"def kernel(x: {parameter}) -> {annotation}: ...\n"
        "@pl.jit\n"
        f"def entry(x: {parameter}):\n"
        "    kernel(x)\n"
    )
    spec = importlib.util.spec_from_file_location("_extern_empty_tuple_stub", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = (torch.empty(16),)
    if tuple_name == "pl.Tuple" or output:
        message = "single tuple-valued result" if tuple_name == "pl.Tuple" else "0 results for 1 Out/InOut"
        with pytest.raises(ValueError, match=message):
            _specialize(module.entry, *args)
    else:
        config, program = _generate_external_config(module.entry, args, tmp_path)
        names = ["kernel", "kernel_aic", "kernel_aiv"] if core_type == "mixed" else ["kernel"]
        for name in names:
            function = program.get_function(name)
            assert function is not None
            assert len(function.return_types) == 0
        assert "ext.cpp" in config


@pytest.mark.parametrize("core_type", ["aic", "aiv", "mixed"])
@pytest.mark.parametrize("bare_return", [False, True])
def test_legacy_multi_output_extern_used_for_side_effects(tmp_path, core_type, bare_return):
    cpp = _write_kernel(tmp_path)
    decorator = (
        pl.jit.extern(core_type="mixed", aic_source=cpp, aiv_source=cpp)
        if core_type == "mixed"
        else pl.jit.extern(core_type=core_type, source=cpp)
    )
    if bare_return:

        @decorator
        def fill(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]) -> pl.Tensor: ...
    else:

        @decorator
        def fill(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]): ...

    @pl.jit
    def entry(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]):
        fill(y, z)
        return y, z

    # Legacy kernels such as qkv_quant write several buffers and callers do
    # not capture their inferred return. Full-contract checks must not turn
    # the historical first-Out inference into an explicit partial contract.
    config, program = _generate_external_config(
        entry, (torch.empty(16, dtype=torch.int8), torch.empty(1)), tmp_path
    )
    assert "fill" in config
    assert program.get_function("entry") is not None


@pytest.mark.parametrize("postponed", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_incomplete_tuple_alias_keeps_legacy_side_effect_calls(tmp_path, monkeypatch, postponed, partial):
    cpp = _write_kernel(tmp_path)
    if partial:
        monkeypatch.setitem(globals(), "_IncompleteOutputsAlias", tuple[pl.Tensor[[16], pl.INT8], pl.Tensor])
    if postponed:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def fill(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]) -> "_IncompleteOutputsAlias": ...
    else:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def fill(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]) -> _IncompleteOutputsAlias: ...

    @pl.jit
    def entry(y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]):
        fill(y, z)
        return y, z

    # Fingerprinting and specialization must both decline the incomplete
    # contract, just as an inline tuple[pl.Tensor, pl.Tensor] would.
    assert fill._get_source_hash()
    assert fill._persistent_source_digest()
    config, _ = _generate_external_config(
        entry, (torch.empty(16, dtype=torch.int8), torch.empty(1)), tmp_path
    )
    assert "fill" in config


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


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("core_type", ["aiv", "aic", "mixed"])
@pytest.mark.parametrize("dispatch", ["direct", "spmd", "submit"])
@pytest.mark.parametrize("complete_annotation", [False, True])
@pytest.mark.parametrize(
    ("returned", "consumed"),
    [
        (("first", "second", "third"), ("second", "third")),
        (("first", "second", "third"), ("first", "third")),
        (("first", "second", "third"), ("first", "second")),
        (("first", "second", "third"), ("first", "second", "third")),
        (("first", "second", "third"), ("third",)),
        (("second", "third"), ("second", "third")),
        (("third", "first", "second"), ("first", "second", "third")),
    ],
    ids=["unused-leading", "unused-middle", "unused-trailing", "all-live", "one-live", "subset", "reordered"],
)
def test_extern_return_buffer_identities(
    tmp_path, ascend_backend, core_type, dispatch, returned, consumed, complete_annotation
):
    """Unused results must not shift physical buffers at a later task (#3251).

    Generate a real JIT module so each matrix entry exercises specialization,
    outlining, the complete pass pipeline, and orchestration emission. All
    output types are equal: only buffer identity can detect this regression.
    """
    cpp = _write_kernel(tmp_path)
    source_arg = (
        f"aic_source={str(cpp)!r}, aiv_source={str(cpp)!r}"
        if core_type == "mixed"
        else f"source={str(cpp)!r}"
    )
    # Complete annotations must validate against the explicit return aliases,
    # not force the signature-only all-output convention onto this declaration.
    tensor_type = "pl.Tensor[[1, 32], pl.FP32]" if complete_annotation else "pl.Tensor"
    result_type = tensor_type if len(returned) == 1 else f"tuple[{', '.join(tensor_type for _ in returned)}]"
    producer_call = f"{', '.join(returned)} = producer(first, second, third)"
    consumer_call = f"output = consumer({', '.join(consumed)}, output)"
    if dispatch == "direct":
        calls = f"    {producer_call}\n    {consumer_call}"
    else:
        capture = " as producer_tid" if dispatch == "submit" else ""
        deps = ", deps=[producer_tid]" if dispatch == "submit" else ""
        calls = (
            f'    with pl.spmd(1, name_hint="produce"){capture}:\n'
            f"        {producer_call}\n"
            f'    with pl.spmd(1, name_hint="consume"{deps}):\n'
            f"        {consumer_call}"
        )
    module_path = tmp_path / "extern_case.py"
    module_path.write_text(
        f"""import pypto.language as pl

@pl.jit.extern(core_type={core_type!r}, {source_arg})
def producer(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor], third: pl.Out[pl.Tensor]) -> {result_type}:
    return {", ".join(returned)}

@pl.jit.extern(core_type="aiv", source={str(cpp)!r})
def consumer(
    {", ".join(f"x{i}: pl.Tensor" for i in range(len(consumed)))},
    output: pl.Out[pl.Tensor],
) -> pl.Tensor:
    return output

@pl.jit(auto_scope=False)
def entry(output: pl.Out[pl.Tensor[[1, 32], pl.FP32]]):
    first = pl.create_tensor([1, 32], dtype=pl.FP32)
    second = pl.create_tensor([1, 32], dtype=pl.FP32)
    third = pl.create_tensor([1, 32], dtype=pl.FP32)
{calls}
    return output
"""
    )
    module = _load_jit_module(module_path)
    generated = tmp_path / "generated"
    module.entry.compile(config=RunConfig(backend_type=ascend_backend, save_kernels_dir=str(generated)))
    orchestration = (generated / "orchestration" / "entry.cpp").read_text()
    actual = re.findall(r"params_t\d+\.add_input\((\w+)\);", orchestration)
    assert actual == list(consumed), orchestration
    if dispatch == "submit":
        assert ".task_id()" in orchestration
        assert ".set_dependencies(" in orchestration


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("complete_annotation", [False, True])
@pytest.mark.parametrize(
    ("shape", "dtype", "layout"),
    [
        ([1, 32], pl.FP32, pl.ND),
        ([16, 32], pl.FP32, pl.NZ),
        ([32, 4], pl.FP8E8M0, pl.MX_A_ZZ),
        ([4, 32], pl.FP8E8M0, pl.MX_B_NN),
    ],
    ids=["nd", "nz", "mx-a", "mx-b"],
)
def test_extern_single_readonly_return_keeps_input_identity(
    tmp_path, ascend_backend, shape, dtype, layout, complete_annotation
):
    """An explicit return names its parameter even when the kernel has no Out slots."""
    cpp = _write_kernel(tmp_path)
    if complete_annotation:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def forward(value: pl.Tensor) -> pl.Tensor[shape, dtype, layout]:
            return value
    else:

        @pl.jit.extern(core_type="aiv", source=cpp)
        def forward(value: pl.Tensor) -> pl.Tensor:
            return value

    @pl.jit.extern(core_type="aiv", source=cpp)
    def consumer(value: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
        return output

    @pl.jit(auto_scope=False)
    def entry(value: pl.Tensor[shape, dtype, layout], output: pl.Out[pl.Tensor[[1, 32], pl.FP32]]):
        alias = forward(value)
        output = consumer(alias, output)
        return output

    generated = tmp_path / "generated"
    # Layout passes replace parameter Vars; the extern's return reference must
    # follow those replacements all the way through orchestration emission.
    entry.compile(config=RunConfig(backend_type=ascend_backend, save_kernels_dir=str(generated)))
    code = (generated / "orchestration" / "entry.cpp").read_text()
    assert "params_t1.add_input(ext_value);" in code


@pytest.mark.parametrize(
    ("annotation", "error"),
    [
        ("tuple[pl.Tensor[[1], pl.FP32], pl.Tensor[[16], pl.INT8]]", None),
        ("tuple[pl.Tensor[[16], pl.INT8], pl.Tensor[[1], pl.FP32]]", "parameter 'second'"),
        ("pl.Tensor[[1], pl.FP32]", "1 results for 2 explicitly returned"),
    ],
)
def test_extern_explicit_return_annotation_matches_named_parameters(tmp_path, annotation, error):
    """Validate heterogeneous aliases in return order after combining #3249 and #3251."""
    cpp = _write_kernel(tmp_path)
    module_path = tmp_path / "annotated_aliases.py"
    module_path.write_text(
        f"""import pypto.language as pl
@pl.jit.extern(core_type="aiv", source={str(cpp)!r})
def pair(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]) -> {annotation}:
    return second, first
@pl.jit
def entry(first: pl.Out[pl.Tensor], second: pl.Out[pl.Tensor]):
    second, first = pair(first, second)
    return first, second
"""
    )
    entry = _load_jit_module(module_path).entry
    args = (torch.empty(16, dtype=torch.int8), torch.empty(1))
    if error is not None:
        with pytest.raises(ValueError, match=error):
            _specialize(entry, *args)
    else:
        entry.compile(*args, config=RunConfig(save_kernels_dir=str(tmp_path / "generated")))


@pytest.mark.parametrize(
    "body",
    ["return pl.add(output, output)", "alias = output\n    return alias", "if True:\n        return output"],
)
def test_extern_rejects_computed_return_identity(tmp_path, body):
    """An external stub must not silently discard computation that selects its result."""
    cpp = _write_kernel(tmp_path)
    module_path = tmp_path / "invalid_extern.py"
    module_path.write_text(
        f"""import pypto.language as pl
@pl.jit.extern(core_type="aiv", source={str(cpp)!r})
def invalid(output: pl.Out[pl.Tensor]) -> pl.Tensor:
    {body}
@pl.jit
def entry(output: pl.Out[pl.Tensor]):
    output = invalid(output)
    return output
"""
    )
    module = _load_jit_module(module_path)
    with pytest.raises(ValueError, match="can only return tensor parameters|single return statement"):
        _specialize(module.entry, torch.empty(1, 32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
