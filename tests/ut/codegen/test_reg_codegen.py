# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Register lowering tests assert typed VMI, external storage and lexical scope."""

import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile

SPAN = ir.Span("register_kernel.py", 10, 1, 10, 60)


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def integer(n):
    return ir.ConstInt(n, DataType.INDEX, SPAN)


def build_kernel(operation="add", dtype=DataType.FP32, offset: int | ir.Expr = 0, masked=False):
    ib = IRBuilder()
    view = ir.TileView([1, 128], [128, 1], integer(0))
    tile_type = ir.TileType([1, 128], dtype, tile_view=view, memory_space=ir.MemorySpace.Vec)
    with ib.function("register_kernel", span=SPAN, type=ir.FunctionType.AIV) as f:
        fill = 1.0 if dtype in (DataType.FP16, DataType.BF16, DataType.FP32) else 1
        src = ib.let("src", tile.full([1, 128], dtype=dtype, value=fill), type=tile_type)
        dst = ib.let("dst", tile.full([1, 128], dtype=dtype, value=0), type=tile_type)
        with ib.reg_scope(SPAN):
            x = ib.emit_reg(
                "x",
                "reg.load",
                [src, (integer(offset) if isinstance(offset, int) else offset)],
                {"lanes": 64},
                SPAN,
            )
            m = ib.emit_reg("mask", "reg.active_mask", [integer(32)], {"lanes": 64}, SPAN)
            if operation == "convert":
                y = ib.emit_reg("y", "reg.convert", [x], {"dtype": DataType.FP16, "saturate": False}, SPAN)
            elif operation == "compare":
                y = ib.emit_reg("y", "reg.compare", [x, x, m], {"predicate": "lt"}, SPAN)
                y = ib.emit_reg("selected", "reg.select", [y, x, x], span=SPAN)
            elif operation.startswith("mask_"):
                args = [m] if operation == "mask_not" else [m, m]
                mask = ib.emit_reg("logic", "reg." + operation, args, span=SPAN)
                y = ib.emit_reg("selected", "reg.select", [mask, x, x], span=SPAN)
            else:
                y = ib.emit_reg("y", "reg." + operation, [x, x, m] if masked else [x, x], span=SPAN)
            # This result must be bound to dst's outer tile handle. No vstore
            # result or vecscope-local tile handle is legal outside the scope.
            out = ib.emit_reg_store("out", dst, y, mask=m if masked else None, span=SPAN)
        ib.let("consumer", tile.add(out, src))
    program = ir.Program([f.get_result()], "RegisterKernel", SPAN)
    with passes.PassContext([]):
        return passes.init_mem_ref()(program)


def generate(program):
    return codegen.PTOCodegen().generate(program, emit_tile_addr=False, emit_source_loc=False)


def test_typed_load_store_and_external_tile_alias():
    mlir = generate(build_kernel())
    before, region = mlir.split("pto.vecscope {", 1)
    body, after = region.split("\n  }", 1)
    assert before.count("pto.tile_buf_addr") == 2
    assert "pto.tile_buf_addr" not in body
    assert "!pto.vmi.vreg<64xf32>" in body
    assert "!pto.ptr<f32, ub>" in body
    store = next(line.strip() for line in body.splitlines() if "pto.vmi.vstore" in line)
    assert store.startswith("pto.vmi.vstore ") and "->" not in store
    assert "pto.tadd" in after
    assert "pto.alloc_tile" not in body
    assert "scf.yield" not in body


@pytest.mark.parametrize(
    "operation,opcode",
    [
        ("minimum", "vmin"),
        ("maximum", "vmax"),
        ("compare", "vcmp"),
        ("mask_and", "vand"),
        ("mask_or", "vor"),
        ("mask_xor", "vxor"),
        ("mask_not", "vnot"),
    ],
)
def test_schema_opcode_and_typed_operands(operation, opcode):
    mlir = generate(build_kernel(operation))
    assert "pto.vmi." + opcode + " " in mlir
    assert "!pto.vmi.mask<64xpred>" in mlir
    if operation == "compare":
        assert 'cmp = "lt"' in mlir
        assert "pto.vmi.vsel" in mlir


@pytest.mark.parametrize("dtype,spelling", [(DataType.INT32, "si32"), (DataType.UINT32, "ui32")])
def test_register_signedness_is_explicit(dtype, spelling):
    mlir = generate(build_kernel(dtype=dtype))
    assert f"!pto.vmi.vreg<64x{spelling}>" in mlir


def test_dynamic_offset_is_rejected_with_source_span():
    # Schema effects may be unknown, but codegen must never silently emit an
    # unproven pointer. Offset is legal IR until address proof is required.
    offset = ir.Var("offset", ir.ScalarType(DataType.INDEX), SPAN)
    with pytest.raises(ValueError, match="bounds|offset"):
        generate(build_kernel(offset=offset))


def test_non_a5_codegen_rejects_registers():
    program = build_kernel()
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    with pytest.raises(ValueError, match="A5|register|RegScope"):
        generate(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
