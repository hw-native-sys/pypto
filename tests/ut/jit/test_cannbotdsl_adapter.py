# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""覆盖可选 CANNBotDSL 适配器的职责归属和 E8M0 ABI。

提供方测试核从 TensorSpec 或调用者元数据推导参数契约，
因此静态收集测试时无需安装可选包。
"""

import importlib
import json
import os
import sys
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto.backend import BackendType
from pypto.jit import cannbotdsl as adapter
from pypto.jit.decorator import _discover_dep_bindings
from pypto.jit.specializer import DynDim, SpecializeContext, Specializer, TensorMeta
from pypto.runtime import RunConfig


def test_import_without_cannbotdsl(monkeypatch):
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "cannbotdsl", None)
        importlib.reload(adapter)


def test_e8m0_aliases():
    cb = pytest.importorskip("cannbotdsl")
    for dtype in (pl.FP8E8M0, "FP8E8M0", "FP8_E8M0"):
        assert adapter._dtype(dtype) == cb.dtypes.float8_e8m0


@pytest.mark.skipif(
    not os.environ.get("ASCEND_HOME_PATH"), reason="CANN include discovery requires its environment"
)
def test_e8m0_export_preserves_byte_storage_and_effects(tmp_path, monkeypatch):
    cb = pytest.importorskip("cannbotdsl")

    @cb.kernel
    def copy_scales(source, output):
        # 以字节形式复制缩放位模式；设备入口的 ABI 类型仍为 E8M0。
        staging = cb.Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, 32)))
        cb.mem_copy(staging, source.view(dtype=cb.dtypes.uint8))
        cb.mem_copy(output.view(dtype=cb.dtypes.uint8), staging)

    meta = TensorMeta(shape=(1, 32), dtype=pl.FP8E8M0)
    context = SpecializeContext(
        func_name="copy_scales",
        source="",
        func_type="extern",
        level=None,
        param_names=["source", "output"],
        tensor_meta={"source": meta, "output": meta},
        scalar_dtypes={},
        constexpr_values={},
    )
    monkeypatch.setenv("CANNBOTDSL_PYPTO_CACHE_DIR", str(tmp_path))
    spec = adapter._build(copy_scales, context)
    assert spec.parameter_access == {"source": "read", "output": "write"}
    assert spec.core_type == "aiv"
    source = Path(spec.source)
    manifest = json.loads(source.with_name("manifest.json").read_text())
    assert [p["dtype"] for p in manifest["parameters"]] == ["f8e8m0", "f8e8m0"]
    assert manifest["export_backend"] == "source_bundle"
    assert [p["logical_index"] for p in manifest["device_abi"]["parameters"]] == [0, 1]
    assert "PyPTODeviceArg<0, decltype(kernel_copy_scales)>" in source.read_text()
    assert "fp8_e8m0_t" in source.with_name("device_source.hpp").read_text()


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_discovery_without_provider_hook(tmp_path, monkeypatch, ascend_backend):
    cb = pytest.importorskip("cannbotdsl")

    @cb.kernel
    def copy_row(source, output):
        staging = cb.Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, 32)))
        cb.mem_copy(staging, source)
        cb.mem_copy(output, staging)

    monkeypatch.delattr(type(copy_row), "__pypto_jit__", raising=False)
    monkeypatch.setenv("CANNBOTDSL_PYPTO_CACHE_DIR", str(tmp_path))

    @pl.jit
    def entry(source: pl.Tensor, output: pl.Out[pl.Tensor]):
        copy_row(source, output)
        return output

    first = _discover_dep_bindings(entry._func)[0].dep
    assert _discover_dep_bindings(entry._func)[0].dep is first
    assert not hasattr(copy_row, "_pypto_jit")
    compiled = entry.compile(
        torch.zeros(1, 32), torch.zeros(1, 32), config=RunConfig(backend_type=BackendType.Ascend950)
    )
    assert compiled is not None


def test_configured_kernel_rejected_after_binding():
    cb = pytest.importorskip("cannbotdsl")

    @cb.kernel
    def leaf(source):
        pass

    assert adapter.try_bind_kernel(leaf) is adapter.try_bind_kernel(leaf)
    leaf[2]
    with pytest.raises(ValueError, match="unconfigured"):
        adapter.try_bind_kernel(leaf)


def test_source_bundle_uses_native_scalar_order_and_dynamic_fields(tmp_path, monkeypatch):
    cb = pytest.importorskip("cannbotdsl")

    @cb.kernel
    def scaled(alpha, source, output, count):
        for row in cb.range(count):
            for col in cb.range(32):
                output[row, col] = source[row, col] * alpha

    meta = TensorMeta(
        shape=(DynDim(name="rows", literal="rows", static_bound=8), 32), dtype=pl.FP32, layout=None
    )
    context = SpecializeContext(
        func_name="scaled",
        source="",
        func_type="extern",
        level=None,
        param_names=["alpha", "source", "output", "count"],
        tensor_meta={"source": meta, "output": meta},
        scalar_dtypes={"alpha": pl.FP32, "count": pl.INDEX},
        constexpr_values={},
    )
    monkeypatch.setenv("CANNBOTDSL_PYPTO_CACHE_DIR", str(tmp_path))
    spec = adapter._build(scaled, context)
    path = Path(spec.source)
    manifest = json.loads(path.with_name("manifest.json").read_text())
    assert [p["logical_index"] for p in manifest["device_abi"]["parameters"]] == [1, 2, 3, 0]
    assert spec.parameter_access == {"source": "read", "output": "write"}
    wrapper = path.read_text()
    assert "kernel_scaled(arg1, arg2, arg3, arg0);" in wrapper
    assert "arg1.shape[0] = t1->shapes[0];" in wrapper
    assert "arg1.stride[0] = t1->strides[0];" in wrapper
    assert "reinterpret_cast<__gm__ Arg0*>(&args[0])" in wrapper


def test_optional_provider_is_not_imported_for_other_callables(monkeypatch):
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "cannbotdsl.lang.kernel", None)
        assert adapter.try_bind_kernel(lambda: None) is None


@pytest.mark.parametrize("query", ["get_block_idx", "get_block_num"])
def test_launch_queries_use_logical_context_before_constant_folding(query, tmp_path):
    cb = pytest.importorskip("cannbotdsl")

    source_bundle = importlib.import_module("pypto.jit._cannbotdsl_source")

    @cb.kernel
    def block_index(output):
        output[0] = cb.get_block_idx()

    @cb.kernel
    def block_count(output):
        output[0] = cb.get_block_num()

    kernel = {"get_block_idx": block_index, "get_block_num": block_count}[query]
    exported = source_bundle.export_source_bundle(kernel, [cb.TensorSpec((1,), cb.dtypes.int64)])
    assert exported.manifest["parameters"][1] == {
        "logical_index": 1,
        "name": query,
        "kind": "context",
    }
    # 导出时的占位 block_dim=1 不能把运行时核数折叠掉。
    wrapper = adapter.write_wrapper(exported, tmp_path).read_text()
    assert f"Arg1 arg1 = {query}(args);" in wrapper
    assert f"{query}()" not in exported.source


@pytest.mark.parametrize("core", ["aiv", "aic", "mixed"])
def test_global_syncall_declares_external_launch_requirement(core):
    cb = pytest.importorskip("cannbotdsl")
    source_bundle = importlib.import_module("pypto.jit._cannbotdsl_source")
    cube_fill_l1_zero = importlib.import_module("cannbotdsl.ops.memcpy").cube_fill_l1_zero
    cube = core in ("aic", "mixed")
    vector = core in ("aiv", "mixed")

    @cb.kernel
    def barrier(source, output):
        if cube:
            l1 = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float32, cb.make_layout((1, 8)))
            cube_fill_l1_zero(l1)
        if vector:
            data = cb.Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, 32)))
            cb.mem_copy(data, source)
            cb.mem_copy(output, data)
        cb.global_sync_all()

    spec = cb.TensorSpec((1, 32), cb.dtypes.float32)
    exported = source_bundle.export_source_bundle(barrier, [spec, spec])
    assert exported.manifest["core_type"] == core
    assert exported.manifest["hard_syncall"] == {"mixed": "mix", "aiv": "aiv_only", "aic": "aic_only"}[core]
    assert "ffts_cross_core_sync" in exported.source


def test_subblock_binding_preserves_static_partition_shapes(tmp_path):
    cb = pytest.importorskip("cannbotdsl")
    source_bundle = importlib.import_module("pypto.jit._cannbotdsl_source")
    cube_fill_l1_zero = importlib.import_module("cannbotdsl.ops.memcpy").cube_fill_l1_zero

    @cb.kernel
    def split_copy(source, output):
        l1 = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float32, cb.make_layout((1, 8)))
        cube_fill_l1_zero(l1)
        lane = cb.get_subblock_id()
        data = cb.Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, 32)))
        cb.mem_copy(data, cb.tile_slice(source, (1, 32), (0, lane)))
        cb.mem_copy(cb.tile_slice(output, (1, 32), (lane, 0)), data)

    # 提前把 lane 替换为无范围的 Scalar，会把每条通道的静态 32 列误推为动态列。
    exported = source_bundle.export_source_bundle(
        split_copy,
        [cb.TensorSpec((1, 64), cb.dtypes.float32), cb.TensorSpec((2, 32), cb.dtypes.float32)],
    )
    assert exported.manifest["core_type"] == "mixed"
    assert exported.manifest["parameters"][-1]["name"] == "get_subblock_id"
    assert "get_subblockid()" not in exported.source
    assert "get_sub_block_id(args)" in adapter.write_wrapper(exported, tmp_path).read_text()


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize(
    "second_shape,second_dtype",
    [((1, 64), torch.float32), ((1, 32), torch.bfloat16), ((1, 64), torch.bfloat16)],
)
def test_nested_foreign_calls_preserve_tensor_specializations(
    tmp_path, monkeypatch, ascend_backend, second_shape, second_dtype
):
    cb = pytest.importorskip("cannbotdsl")

    monkeypatch.setenv("CANNBOTDSL_PYPTO_CACHE_DIR", str(tmp_path))

    @cb.kernel
    def copy_tile(source, output):
        staging = cb.Buffer(cb.MemLoc.UB, source.dtype, cb.make_layout(source.shape))
        cb.mem_copy(staging, source)
        cb.mem_copy(output, staging)

    @pl.jit.inline(auto_scope=False)
    def dispatch(source: pl.Tensor, output: pl.Tensor):
        with pl.spmd(1):
            copy_tile(source, output)

    @pl.jit.inline(auto_scope=False)
    def project(source: pl.Tensor, output: pl.Tensor):
        dispatch(source, output)

    @pl.jit
    def entry(a: pl.Tensor, b: pl.Tensor, x: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]):
        project(a, x)
        project(b, y)
        project(a, z)
        return x, y, z

    a = torch.zeros(1, 32)
    b = torch.zeros(second_shape, dtype=second_dtype)
    _, _, tensor_meta, scalar_dtypes, constexpr, dynamic = entry._bind_args(
        (a, b, torch.zeros_like(a), torch.zeros_like(b), torch.zeros_like(a)), {}
    )
    contexts = entry._build_contexts(tensor_meta, scalar_dtypes, constexpr, dynamic)
    foreign = [ctx for ctx in contexts if ctx.external_parameter_access is not None]
    observed = {(ctx.tensor_meta["source"].shape, ctx.tensor_meta["source"].dtype) for ctx in foreign}
    expected = {((1, 32), pl.FP32), (second_shape, pl.FP32 if second_dtype == torch.float32 else pl.BF16)}
    assert observed == expected
    sources = {ctx.external_aiv_source for ctx in foreign}
    assert len(sources) == 2
    for ctx in foreign:
        assert ctx.tensor_meta["source"] == ctx.tensor_meta["output"]
        manifest = json.loads(Path(ctx.external_aiv_source).with_name("manifest.json").read_text())
        dtype = "f32" if ctx.tensor_meta["output"].dtype == pl.FP32 else "bf16"
        assert [parameter["dtype"] for parameter in manifest["parameters"]] == [dtype, dtype]
    program = pl.parse(Specializer("_jit_nested_foreign", contexts).specialize())
    assert program is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
