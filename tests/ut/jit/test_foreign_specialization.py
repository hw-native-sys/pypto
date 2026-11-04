# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""校验签名向依赖外部核的调用者之下的普通辅助函数传播。"""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.jit import JITFunction
from pypto.jit.foreign import ExternalKernelSpec
from pypto.jit.specializer import Specializer


@pytest.mark.parametrize(
    "second_shape,second_dtype",
    [((1, 64), torch.float32), ((1, 32), torch.bfloat16), ((1, 64), torch.bfloat16)],
)
def test_ordinary_descendants_inherit_foreign_caller_signature(tmp_path, second_shape, second_dtype):
    cpp = tmp_path / "reader.cpp"
    cpp.write_text('extern "C" void kernel_entry(long long* args) {}\n')

    def native(source):
        pass

    leaf = JITFunction(
        native,
        func_type="extern",
        external_builder=lambda ctx: ExternalKernelSpec(
            source=str(cpp), core_type="aiv", parameter_access={"source": "read"}
        ),
    )

    @pl.jit.inline(auto_scope=False)
    def ordinary_inner(source: pl.Tensor):
        width = pl.tensor.dim(source, 1)
        result = pl.slice(source, [1, width], [0, 0])
        return result

    @pl.jit.inline(auto_scope=False)
    def ordinary(source: pl.Tensor):
        result = ordinary_inner(source)
        return result

    @pl.jit.inline(auto_scope=False)
    def mixed(source: pl.Tensor):
        leaf(source)
        result = ordinary(source)
        return result

    @pl.jit
    def entry(a: pl.Tensor, b: pl.Tensor):
        first = mixed(a)
        second = mixed(b)
        repeated = mixed(a)
        return first, second, repeated

    args = (torch.empty(1, 32), torch.empty(second_shape, dtype=second_dtype))
    _, _, tensors, scalars, constants, dynamic = entry._bind_args(args, {})
    contexts = entry._build_contexts(tensors, scalars, constants, dynamic)
    expected = {
        ((1, 32), pl.FP32),
        (second_shape, pl.FP32 if second_dtype == torch.float32 else pl.BF16),
    }
    for original_name in ("ordinary_inner", "ordinary", "mixed", "native"):
        variants = [ctx for ctx in contexts if ctx.source_func_name == original_name]
        assert len(variants) == 2
        assert {
            (ctx.tensor_meta["source"].shape, ctx.tensor_meta["source"].dtype) for ctx in variants
        } == expected

    program = pl.parse(Specializer("_jit_mixed_foreign", contexts).specialize())
    assert isinstance(program, ir.Program)
    for ctx in contexts:
        if ctx.source_func_name != "ordinary_inner":
            continue
        function = program.get_function(ctx.func_name)
        assert function is not None
        rendered = function.as_python()
        assert f"[1, {ctx.tensor_meta['source'].shape[1]}]" in rendered


def test_ordinary_only_graph_retains_constexpr_plan():
    @pl.jit.inline
    def ordinary(source: pl.Tensor):
        result = pl.add(source, 1.0)
        return result

    @pl.jit
    def entry(source: pl.Tensor):
        result = ordinary(source)
        return result

    _, _, tensors, scalars, constants, dynamic = entry._bind_args((torch.empty(1, 32),), {})
    original = entry._resolve_constexpr_bindings(constants)
    foreign_aware = entry._resolve_foreign_variants(tensors, scalars, constants, dynamic)
    assert foreign_aware.order == original.order
    assert foreign_aware.binding == original.binding
    assert foreign_aware.call_targets == original.call_targets
