# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile the URMA async transfer adapter against the A5 runtime headers."""

import os

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.backend.pto_backend import _URMA_TRANSFER_INCLUDE, _generate_urma_transfer_adapter


@pytest.mark.ptoas_compile
def test_urma_adapter_compiles_with_a5_runtime(tmp_path):
    pytest.importorskip("simpler_setup")
    pto_isa_root = os.environ.get("PTO_ISA_ROOT")
    if not pto_isa_root or not os.path.isdir(pto_isa_root):
        pytest.skip("PTO_ISA_ROOT is not available")
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP32],
            src: pld.DistributedTensor[[16, 64], pl.FP32],
            peer: pl.Scalar[pl.INT32],
        ):
            _out = pld.tensor.put_async(dst, peer=peer, src=src, engine=pld.CommEngine.URMA)
            _back = pld.tensor.get_async(dst, peer=peer, src=src, engine=pld.CommEngine.URMA)

    func = P.get_function("kernel")
    assert func is not None
    adapter = _generate_urma_transfer_adapter(func)
    assert "pypto_urma_put_async_f32" in adapter
    assert "pypto_urma_get_async_f32" in adapter
    assert "pypto_urma_transfer_impl<float, true>" in adapter
    assert "pypto_urma_transfer_impl<float, false>" in adapter

    source = tmp_path / "urma_adapter.cpp"
    source.write_text(
        "#include <pto/pto-inst.hpp>\n"
        + _URMA_TRANSFER_INCLUDE
        + adapter
        + 'extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {\n'
        + "  pypto_urma_put_async_f32(args, args, nullptr, nullptr, 0, 0, 0, 0, 0, 0);\n"
        + "  pypto_urma_get_async_f32(args, args, nullptr, nullptr, 0, 0, 0, 0, 0, 0);\n"
        + "}\n"
    )
    binary = KernelCompiler("a5").compile_incore(
        str(source),
        runtime_name="tensormap_and_ringbuffer",
        pto_isa_root=pto_isa_root,
    )
    assert binary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
