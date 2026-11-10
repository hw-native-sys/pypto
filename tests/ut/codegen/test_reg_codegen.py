# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Register lowering tests assert typed VMI, external storage and lexical scope."""

import subprocess

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile

SPAN = ir.Span("register_kernel.py", 10, 1, 10, 60)


@pytest.mark.ptoas_compile
def test_dynamic_active_mask_compiles_without_vector_comparison(tmp_path):
    ib = IRBuilder()
    with ib.function("dynamic_mask", span=SPAN, type=ir.FunctionType.AIV) as f:
        count = f.param("count", ir.ScalarType(DataType.INDEX), SPAN)
        dst = ib.let("dst", tile.full([64], dtype=DataType.FP32, value=0))
        with ib.reg_scope(SPAN):
            mask = ib.let("mask", pl.reg.active_mask(count, lanes=64).unwrap())
            value = ib.emit_reg(
                "value",
                "reg.broadcast",
                [ir.ConstFloat(1, DataType.FP32, SPAN)],
                {"lanes": 64, "dtype": DataType.FP32},
                SPAN,
            )
            ib.emit_reg_store("out", dst, value, mask=mask, span=SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "DynamicMask", SPAN))
    source = generate(program)
    assert "pto.vmi.create_mask %arg0 : index" in source
    assert "pto.vmi.vcmp" not in source
    compile_review_pto(source, tmp_path)


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def integer(n):
    return ir.ConstInt(n, DataType.INDEX, SPAN)


def build_kernel(
    operation="add", dtype=DataType.FP32, offset: int | ir.Expr = 0, masked=False, dynamic_valid=False
):
    ib = IRBuilder()
    with ib.function("register_kernel", span=SPAN, type=ir.FunctionType.AIV) as f:
        if isinstance(offset, ir.Var):
            offset = f.param(offset.name_hint, offset.type, SPAN)
        valid_col = f.param("valid_col", ir.ScalarType(DataType.INDEX), SPAN) if dynamic_valid else 128
        view = ir.TileView([1, valid_col], [128, 1], integer(0))
        tile_type = ir.TileType([1, 128], dtype, tile_view=view, memory_space=ir.MemorySpace.Vec)
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
            out = ib.emit_reg_store("out", dst, y, offset=offset, mask=m if masked else None, span=SPAN)
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


def test_register_module_selects_vpto():
    assert 'pto.backend = "vpto"' in generate(build_kernel())


def test_ordinary_module_keeps_emitc_default():
    @pl.function(type=pl.FunctionType.AIV)
    def ordinary():
        _tile = pl.tile.full([1, 128], pl.FP32, 0.0)

    program = ir.Program([ordinary], "Ordinary", SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(program)
    assert "pto.backend" not in generate(program)


def frontend_tile_kernel():
    @pl.function(type=pl.FunctionType.AIV)
    def register_kernel(
        src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
        dst: pl.InOut[pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec]],
    ):
        with pl.reg_scope():
            x = pl.reg.load(src, lanes=64)
            y = pl.reg.add(x, x)
            out = pl.reg.store(dst, value=y)
        _unused = pl.tile.add(out, out)

    return ir.Program([register_kernel], "RegisterKernel", SPAN)


def test_frontend_register_scope_survives_flattening():
    with passes.PassContext([]):
        lowered = passes.flatten_tile_nd_to_2d()(frontend_tile_kernel())
        lowered = passes.init_mem_ref()(lowered)
    mlir = codegen.PTOCodegen().generate(lowered, emit_source_loc=False)
    assert "!pto.ptr<f32, ub>" in mlir.splitlines()[1]
    assert "pto.castptr %arg0 : !pto.ptr<f32, ub> -> i64" in mlir


@pytest.mark.ptoas_compile
def test_frontend_register_kernel_compiles_with_vpto(tmp_path):
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    # Exercise the same Python example that previously produced a GM pointer
    # consumed as a tile_buf. Level3 preserves caller-owned UB addresses.
    with passes.PassContext([]):
        lowered = passes.flatten_tile_nd_to_2d()(frontend_tile_kernel())
        lowered = passes.init_mem_ref()(lowered)
        lowered = passes.allocate_memory_addr()(lowered)
    source = tmp_path / "register_kernel.pto"
    source.write_text(codegen.PTOCodegen().generate(lowered, emit_source_loc=False))
    output = tmp_path / "register_kernel.vpto"
    result = subprocess.run(
        [compiler, str(source), "--pto-arch=a5", "--pto-level=level3", "--emit-vpto", "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "pto.vecscope" in output.read_text()


@pytest.mark.ptoas_compile
def test_frontend_fp32_8x256_muladd_register_loop_compiles_with_vpto(tmp_path):
    @pl.function(type=pl.FunctionType.AIV)
    def multiply_add(
        a: pl.Tile[
            [8, 256],
            pl.FP32,
            pl.MemorySpace.Vec,
            pl.TileView(valid_shape=[8, 256], stride=[256, 1], start_offset=0),
        ],
        b: pl.Tile[
            [8, 256],
            pl.FP32,
            pl.MemorySpace.Vec,
            pl.TileView(valid_shape=[8, 256], stride=[256, 1], start_offset=0),
        ],
        c: pl.Tile[
            [8, 256],
            pl.FP32,
            pl.MemorySpace.Vec,
            pl.TileView(valid_shape=[8, 256], stride=[256, 1], start_offset=0),
        ],
        dst: pl.Out[
            pl.Tile[
                [8, 256],
                pl.FP32,
                pl.MemorySpace.Vec,
                pl.TileView(valid_shape=[8, 256], stride=[256, 1], start_offset=0),
            ]
        ],
    ):
        with pl.reg_scope():
            # One 256-byte FP32 register covers 64 elements. The 32 iterations
            # cover all 8*256 elements; offsets are in elements, not bytes.
            for block in pl.range(32):
                offset = block * 64
                x = pl.reg.load(a, offset=offset, lanes=64)
                y = pl.reg.load(b, offset=offset, lanes=64)
                z = pl.reg.load(c, offset=offset, lanes=64)
                product = pl.reg.mul(x, y)
                result = pl.reg.add(product, z)
                _out = pl.reg.store(dst, offset=offset, value=result)

    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    with passes.PassContext([]):
        program = ir.Program([multiply_add], "RegisterMulAdd", SPAN)
        program = passes.flatten_tile_nd_to_2d()(program)
        program = passes.init_mem_ref()(program)
        program = passes.allocate_memory_addr()(program)
    source = codegen.PTOCodegen().generate(program, emit_source_loc=False)
    body = source.split("pto.vecscope {", 1)[1]
    assert "scf.for" in body
    assert "arith.muli" in body
    assert body.count("pto.vmi.vload ") == 3
    assert body.count("pto.vmi.vstore ") == 1
    assert "pto.vmi.vmul " in body and "pto.vmi.vadd " in body
    assert "!pto.vmi.vreg<64xf32>" in body
    assert "rows=8, cols=256" in source
    path = tmp_path / "reg_muladd_fp32_8x256.pto"
    output = tmp_path / "reg_muladd_fp32_8x256.vpto"
    path.write_text(source)
    compiled = subprocess.run(
        [compiler, str(path), "--pto-arch=a5", "--pto-level=level3", "--emit-vpto", "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    assert "pto.vecscope" in output.read_text()


@pytest.mark.ptoas_compile
def test_review_reg_access_adds_tile_view_origin(tmp_path):
    ib = IRBuilder()
    with ib.function("view_origin", span=SPAN, type=ir.FunctionType.AIV) as f:
        src = ib.let(
            "src",
            tile.full([128], dtype=DataType.FP32, value=1.0),
            type=ir.TileType(
                [128],
                DataType.FP32,
                tile_view=ir.TileView([128], [1], ir.ConstInt(32, DataType.INT64, SPAN)),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        dst = ib.let("dst", tile.create([128], dtype=DataType.FP32, target_memory=ir.MemorySpace.Vec))
        with ib.reg_scope(SPAN):
            x = ib.emit_reg("x", "reg.load", [src, integer(0)], {"lanes": 64}, SPAN)
            ib.emit_reg_store("out", dst, x, span=SPAN)
    with passes.PassContext([]):
        source = generate(passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN)))
    assert "arith.addi" in source
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
def test_review_parameter_plain_alias_keeps_caller_backing(tmp_path):
    ib = IRBuilder()
    with ib.function("parameter_alias", span=SPAN, type=ir.FunctionType.AIV) as f:
        src_type = ir.TileType([128], DataType.FP32, memory_space=ir.MemorySpace.Vec)
        src = f.param("src", src_type, SPAN)
        alias = ib.let("alias", src, type=src_type)
        with ib.reg_scope(SPAN):
            ib.emit_reg("x", "reg.load", [alias, integer(0)], {"lanes": 64}, SPAN)
    with passes.PassContext([]):
        program = ir.Program([f.get_result()], "Review", SPAN)
        program = passes.allocate_memory_addr()(passes.init_mem_ref()(program))
    source = codegen.PTOCodegen().generate(program, emit_source_loc=False)
    assert "pto.tile_buf_addr %src_tile" in source
    assert "pto.tile_buf_addr %alias" not in source
    compile_review_pto(source, tmp_path, level="level3")


@pytest.mark.ptoas_compile
def test_review_branch_accepts_equivalent_external_tile_alias(tmp_path):
    """Branch results compare canonical external handles, not Var identity."""
    ib = IRBuilder()
    with ib.function("branch_alias", span=SPAN, type=ir.FunctionType.AIV) as f:
        condition = f.param("condition", ir.ScalarType(DataType.BOOL), SPAN)
        src_type = ir.TileType([128], DataType.FP32, memory_space=ir.MemorySpace.Vec)
        src = f.param("src", src_type, SPAN)
        alias = ib.let("alias", src, type=src_type)
        with ib.reg_scope(SPAN):
            value = ib.emit_reg(
                "value",
                "reg.broadcast",
                [ir.ConstFloat(1.0, DataType.FP32, SPAN)],
                {"lanes": 64, "dtype": DataType.FP32},
                SPAN,
            )
            with ib.if_stmt(condition, SPAN) as branch:
                branch.return_var("out", src_type, SPAN)
                then_out = ib.emit_reg_store("then_out", src, value, span=SPAN)
                ib.emit(ir.YieldStmt([then_out], SPAN))
                branch.else_(SPAN)
                else_out = ib.emit_reg_store("else_out", alias, value, span=SPAN)
                ib.emit(ir.YieldStmt([else_out], SPAN))
            out = branch.output(0)
            ib.emit_reg("loaded", "reg.load", [out, integer(0)], {"lanes": 64}, SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN))
        program = passes.allocate_memory_addr()(program)
    source = codegen.PTOCodegen().generate(program, emit_source_loc=False)
    assert source.count("pto.tile_buf_addr %src") == 1
    assert "pto.tile_buf_addr %alias" not in source
    compile_review_pto(source, tmp_path, level="level3")


@pytest.mark.parametrize("separate_calls", [False, True])
def test_review_tile_alias_provenance_resets_between_functions(separate_calls):
    """A Var reused in a later IR must not inherit an earlier alias binding."""
    tile_type = ir.TileType([128], DataType.FP32, memory_space=ir.MemorySpace.Vec)
    src = ir.Var("old_source", tile_type, SPAN)
    alias = ir.Var("alias", tile_type, SPAN)
    first = ir.Function("first", [src], [], ir.AssignStmt(alias, src, SPAN), SPAN, type=ir.FunctionType.AIV)
    with passes.PassContext([]):
        first_program = passes.init_mem_ref()(ir.Program([first], "First", SPAN))
    first = first_program.get_function("first")
    assert first is not None
    assert isinstance(first.body, ir.SeqStmts)
    assignment = first.body.stmts[-1]
    assert isinstance(assignment, ir.AssignStmt)
    alias = assignment.var
    load = ir.create_op_call("reg.load", [alias, integer(0)], {"lanes": 64}, SPAN)
    loaded = ir.Var("loaded", load.type, SPAN)
    second = ir.Function(
        "second",
        [alias],
        [],
        ir.RegScopeStmt("reg", ir.AssignStmt(loaded, load, SPAN), SPAN),
        SPAN,
        type=ir.FunctionType.AIV,
    )
    # Reusing the same node deterministically exercises the stale raw-pointer
    # key, without relying on allocator address reuse after an IR is destroyed.
    emitter = codegen.PTOCodegen()
    program = ir.Program([second], "Second", SPAN)
    fresh = codegen.PTOCodegen().generate(program, emit_source_loc=False)
    if separate_calls:
        emitter.generate(first_program, emit_source_loc=False)
        reused = emitter.generate(program, emit_source_loc=False)
        assert reused == fresh
    else:
        reused = emitter.generate(ir.Program([first, second], "Both", SPAN), emit_source_loc=False)
        second_body = reused.split("func.func @second", 1)[1]
        assert "old_source" not in second_body
        assert "pto.tile_buf_addr %alias_tile" in second_body


def test_index_gen_bf16_rejected_by_builder_before_codegen():
    ib = IRBuilder()
    with pytest.raises(ValueError, match="index_gen.*BF16|BF16.*VCI"):
        with ib.function("bf16_index_gen", span=SPAN, type=ir.FunctionType.AIV):
            with ib.reg_scope(SPAN):
                ib.emit_reg(
                    "indices",
                    "reg.index_gen",
                    [ir.ConstFloat(0.0, DataType.BF16, SPAN)],
                    {"lanes": 64, "dtype": DataType.BF16, "order": "ascending"},
                    SPAN,
                )


@pytest.mark.ptoas_compile
def test_review_parameter_view_keeps_caller_backing(tmp_path):
    ib = IRBuilder()
    with ib.function("parameter_view", span=SPAN, type=ir.FunctionType.AIV) as f:
        src = f.param(
            "src",
            ir.TileType([128], DataType.FP32, memory_space=ir.MemorySpace.Vec),
            SPAN,
        )
        view = ib.let(
            "view",
            tile.reshape(src, [1, 128]),
            type=ir.TileType(
                [1, 128],
                DataType.FP32,
                tile_view=ir.TileView([1, 128], [128, 1], integer(0)),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        with ib.reg_scope(SPAN):
            ib.emit_reg("x", "reg.load", [view, integer(0)], {"lanes": 64}, SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN))
        program = passes.allocate_memory_addr()(program)
    source = codegen.PTOCodegen().generate(program, emit_source_loc=False)
    assert source.count("pto.alloc_tile") == 1
    assert "pto.alloc_tile addr = %src_addr" in source
    compile_review_pto(source, tmp_path, level="level3")


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_tensor_frontend_default_pipeline(tmp_path, planner):
    # Tensor parameters are the public GM ABI; Tile parameters above exercise
    # the lower-level caller-owned UB ABI. Both must reach valid codegen.
    @pl.function(type=pl.FunctionType.AIV)
    def register_tensor_kernel(
        src: pl.Tensor[[1, 128], pl.FP32],
        dst: pl.InOut[pl.Tensor[[1, 128], pl.FP32]],
    ):
        tile_src: pl.Tile[
            [1, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[1, 128], stride=[128, 1])
        ] = pl.tile.load(src, [0, 0], [1, 128], [1, 128], target_memory=pl.MemorySpace.Vec)
        tile_dst: pl.Tile[
            [1, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[1, 128], stride=[128, 1])
        ] = pl.tile.load(dst, [0, 0], [1, 128], [1, 128], target_memory=pl.MemorySpace.Vec)
        with pl.reg_scope():
            x = pl.reg.load(tile_src, lanes=64)
            y = pl.reg.add(x, x)
            out = pl.reg.store(tile_dst, value=y)
        _stored = pl.tile.store(out, [0, 0], dst)

    with passes.PassContext([], memory_planner=planner):
        ir.compile(
            ir.Program([register_tensor_kernel], "RegisterTensor", SPAN),
            output_dir=str(tmp_path),
            backend_type=backend.BackendType.Ascend950,
            skip_ptoas=True,
        )
    source = tmp_path / "kernels/aiv/register_tensor_kernel.pto"
    assert 'pto.backend = "vpto"' in source.read_text()
    compiler = find_ptoas_binary()
    if compiler is not None:
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        result = subprocess.run(
            [compiler, str(source), "--pto-arch=a5", f"--pto-level={level}", "--emit-vpto"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_vpto_object_artifact_is_not_wrapped_as_cpp(tmp_path, monkeypatch):
    binary = b"\x7fELF\x00\xff"

    def fake_ptoas(pto_path, output_path, ptoas_flags=None):
        assert output_path.endswith(".o")
        with open(output_path, "wb") as output:
            output.write(binary)

    monkeypatch.setattr(pto_backend, "_run_ptoas", fake_ptoas)
    # Explicit diagnostic VPTO modules retain their object-output route. The
    # execution driver now generates an EmitC caller plus separate helpers.
    program = build_kernel()
    function = program.get_function("register_kernel")
    assert function is not None
    files = {}
    pto_backend._emit_single_function_output(
        files,
        function,
        generate(program),
        str(tmp_path),
        skip_ptoas=False,
    )
    assert (tmp_path / "kernels/aiv/register_kernel.o").read_bytes() == binary
    assert not (tmp_path / "kernels/aiv/register_kernel.cpp").exists()
    assert '"artifact_kind": "fatobj"' in files["kernels/aiv/register_kernel.o.json"]


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "operation,opcode",
    [
        ("add", "vadd"),
        ("sub", "vsub"),
        ("mul", "vmul"),
        ("div", "vdiv"),
        ("minimum", "vmin"),
        ("maximum", "vmax"),
        ("compare", "vcmp"),
        ("mask_and", "vand"),
        ("mask_or", "vor"),
        ("mask_xor", "vxor"),
        ("mask_not", "vnot"),
    ],
)
def test_schema_opcode_and_typed_operands(operation, opcode, tmp_path):
    mlir = generate(build_kernel(operation))
    assert "pto.vmi." + opcode + " " in mlir
    assert "!pto.vmi.mask<64xpred>" in mlir
    if operation == "compare":
        assert 'cmp = "lt"' in mlir
        assert "pto.vmi.vsel" in mlir
    compile_review_pto(mlir, tmp_path)


@pytest.mark.parametrize("dtype,spelling", [(DataType.INT32, "si32"), (DataType.UINT32, "ui32")])
def test_register_signedness_is_explicit(dtype, spelling):
    mlir = generate(build_kernel(dtype=dtype))
    assert f"!pto.vmi.vreg<64x{spelling}>" in mlir


@pytest.mark.ptoas_compile
def test_dynamic_offset_passes_through_to_vpto(tmp_path):
    offset = ir.Var("offset", ir.ScalarType(DataType.INDEX), SPAN)
    source = generate(build_kernel(offset=offset))
    assert "pto.vmi.vload" in source
    assert "arith.addi %arg0," in source
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
def test_dynamic_valid_shape_passes_through_to_vpto(tmp_path):
    source = generate(build_kernel(dynamic_valid=True))
    assert "valid_col = %arg0" in source
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "dtype",
    [
        DataType.FP16,
        DataType.BF16,
        DataType.FP32,
        DataType.INT8,
        DataType.UINT8,
        DataType.INT16,
        DataType.UINT16,
        DataType.INT32,
        DataType.UINT32,
    ],
)
@pytest.mark.parametrize("active", [0, 32, 64])
def test_vf_storage_generators_and_masks_compile(dtype, active, tmp_path):
    """Exercise each documented storage dtype and empty/partial/full predication."""
    ib = IRBuilder()
    with ib.function("generators", span=SPAN, type=ir.FunctionType.AIV) as f:
        src = ib.let("src", tile.full([128], dtype=dtype, value=1))
        dst = ib.let("dst", tile.full([128], dtype=dtype, value=0))
        with ib.reg_scope(SPAN):
            x = ib.emit_reg("x", "reg.load", [src, integer(0)], {"lanes": 64}, SPAN)
            literal = ir.ConstFloat(1, dtype, SPAN) if dtype.is_float() else ir.ConstInt(1, dtype, SPAN)
            value = ib.emit_reg("value", "reg.broadcast", [literal], {"lanes": 64, "dtype": dtype}, SPAN)
            if dtype != DataType.BF16:
                zero = ir.ConstFloat(0, dtype, SPAN) if dtype.is_float() else ir.ConstInt(0, dtype, SPAN)
                value = ib.emit_reg("indices", "reg.index_gen", [zero], {"lanes": 64, "dtype": dtype}, SPAN)
            full = ib.emit_reg("full", "reg.full_mask", [], {"lanes": 64}, SPAN)
            prefix = ib.emit_reg("prefix", "reg.active_mask", [integer(active)], {"lanes": 64}, SPAN)
            predicate = ib.emit_reg("predicate", "reg.mask_and", [full, prefix], span=SPAN)
            result = ib.emit_reg("sum", "reg.add", [x, value, predicate], span=SPAN)
            ib.emit_reg_store("out", dst, result, mask=predicate, span=SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "Generators", SPAN))
    source = generate(program)
    assert "pto.vmi.vbr" in source and "pto.vmi.pset" in source
    assert "pto.vmi.vload" in source and "pto.vmi.vstore" in source
    assert 'pmode = "merge"' in source
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("predicate", ["eq", "ne", "lt", "le", "gt", "ge"])
@pytest.mark.parametrize("scalar", [False, True])
def test_vf_comparison_predicates_compile(predicate, scalar, tmp_path):
    ib = IRBuilder()
    with ib.function("comparison", span=SPAN, type=ir.FunctionType.AIV) as f:
        dst = ib.let("dst", tile.full([128], dtype=DataType.FP32, value=0))
        with ib.reg_scope(SPAN):
            x = ib.emit_reg(
                "x",
                "reg.broadcast",
                [ir.ConstFloat(1, DataType.FP32, SPAN)],
                {"lanes": 64, "dtype": DataType.FP32},
                SPAN,
            )
            seed = ib.emit_reg("seed", "reg.full_mask", [], {"lanes": 64}, SPAN)
            rhs = ir.ConstFloat(0, DataType.FP32, SPAN) if scalar else x
            condition = ib.emit_reg(
                "condition",
                "reg.compares" if scalar else "reg.compare",
                [x, rhs, seed],
                {"predicate": predicate},
                SPAN,
            )
            value = ib.emit_reg("value", "reg.select", [condition, x, x], span=SPAN)
            ib.emit_reg_store("out", dst, value, span=SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "Comparison", SPAN))
    source = generate(program)
    assert f'cmp = "{predicate}"' in source
    assert "pto.vmi.vsel " in source
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "dtype,index_dtype,lanes",
    [
        (DataType.INT8, DataType.UINT8, 64),
        (DataType.INT8, DataType.UINT8, 128),
        (DataType.INT8, DataType.UINT8, 256),
        (DataType.FP16, DataType.UINT16, 64),
        (DataType.FP16, DataType.UINT16, 128),
        (DataType.FP32, DataType.UINT32, 64),
    ],
)
def test_vf_gather_lane_widths_compile(dtype, index_dtype, lanes, tmp_path):
    ib = IRBuilder()
    with ib.function("gather", span=SPAN, type=ir.FunctionType.AIV) as f:
        dst = ib.let("dst", tile.full([256], dtype=dtype, value=0))
        with ib.reg_scope(SPAN):
            value = ib.emit_reg("value", "reg.load", [dst, integer(0)], {"lanes": lanes}, SPAN)
            indices = ib.emit_reg(
                "indices",
                "reg.index_gen",
                [ir.ConstInt(0, index_dtype, SPAN)],
                {"lanes": lanes, "dtype": index_dtype},
                SPAN,
            )
            result = ib.emit_reg("result", "reg.gather_lanes", [value, indices], span=SPAN)
            ib.emit_reg_store("out", dst, result, span=SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "Gather", SPAN))
    source = generate(program)
    assert "pto.vmi.vselr" in source
    compile_review_pto(source, tmp_path)


def test_vf_fp32_to_fp16_rejects_backend_unsupported_h_rounding():
    x = vector(DataType.FP32)
    with pytest.raises(ValueError, match="Unsupported rounding"):
        pl.reg.convert(x, dtype=DataType.FP16, rounding="H", saturate=False)


def test_non_a5_codegen_rejects_registers():
    program = build_kernel()
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    with pytest.raises(ValueError, match="A5|register|RegScope"):
        generate(program)


def compile_review_pto(source, tmp_path, level="level2"):
    """Verify generated VMI with the real assembler, not just string matching."""
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    path = tmp_path / "review.pto"
    path.write_text(source)
    result = subprocess.run(
        [compiler, str(path), "--pto-arch=a5", f"--pto-level={level}", "--emit-vpto"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("operation", ["reduce_sum", "reduce_max", "reduce_min"])
@pytest.mark.parametrize(
    "dtype", [DataType.FP16, DataType.FP32, DataType.INT16, DataType.UINT16, DataType.INT32, DataType.UINT32]
)
def test_reduction_register_broadcast(operation, dtype, tmp_path):
    ib = IRBuilder()
    with ib.function("reduction_broadcast", span=SPAN, type=ir.FunctionType.AIV) as f:
        dst = ib.let("dst", tile.create([64], dtype=dtype, target_memory=ir.MemorySpace.Vec))
        with ib.reg_scope(SPAN):
            x = ib.emit_reg(
                "x",
                "reg.broadcast",
                [ir.ConstFloat(1.0, dtype, SPAN) if dtype.is_float() else ir.ConstInt(1, dtype, SPAN)],
                {"lanes": 64, "dtype": dtype},
                SPAN,
            )
            m = ib.emit_reg("m", "reg.full_mask", [], {"lanes": 64}, SPAN)
            reduced = ib.emit_reg(
                "reduced",
                f"reg.{operation}",
                [x, m],
                {"reassoc": True} if operation == "reduce_sum" else {},
                SPAN,
            )
            result = ib.emit_reg("result", "reg.broadcast", [reduced], {"lanes": 64, "dtype": dtype}, SPAN)
            ib.emit_reg_store("out", dst, result, span=SPAN)
    with passes.PassContext([]):
        source = generate(passes.init_mem_ref()(ir.Program([f.get_result()], "Broadcast", SPAN)))
    broadcast = next(line for line in source.splitlines() if "pto.vmi.vbrc %reduced" in line)
    assert "!pto.vmi.vreg<1x" in broadcast and "-> !pto.vmi.vreg<64x" in broadcast
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
def test_review_masked_store_preserves_inactive_lanes(tmp_path):
    source = generate(build_kernel(masked=True))
    store = next(line for line in source.splitlines() if "pto.vmi.vstore" in line)
    assert 'pmode = "merge"' in store
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "src_dtype,dst_dtype,attrs",
    [
        (DataType.FP16, DataType.FP32, {}),
        (DataType.BF16, DataType.FP32, {}),
        *[
            (DataType.FP32, DataType.BF16, {"saturate": sat, "rounding": rounding})
            for rounding in ("R", "A", "Z")
            for sat in (False, True)
        ],
        (DataType.INT32, DataType.FP32, {}),
        (DataType.INT8, DataType.FP16, {}),
        (DataType.INT8, DataType.INT16, {}),
        (DataType.INT16, DataType.INT32, {}),
        (DataType.UINT8, DataType.UINT16, {}),
        (DataType.UINT16, DataType.UINT32, {}),
        (DataType.INT32, DataType.INT8, {"saturate": False}),
        *[
            (DataType.FP32, DataType.FP16, {"saturate": sat, "rounding": rounding})
            for rounding in ("R", "A", "Z")
            for sat in (False, True)
        ],
        *[
            (DataType.FP32, DataType.INT32, {"saturate": sat, "rounding": rounding})
            for rounding in ("R", "A", "F", "C", "Z")
            for sat in (False, True)
        ],
    ],
)
def test_review_conversion_attribute_presence(src_dtype, dst_dtype, attrs, tmp_path):
    ib = IRBuilder()
    with ib.function("convert", span=SPAN, type=ir.FunctionType.AIV) as f:
        dst = ib.let("dst", tile.create([128], dtype=dst_dtype, target_memory=ir.MemorySpace.Vec))
        with ib.reg_scope(SPAN):
            scalar = (
                ir.ConstFloat(1.0, src_dtype, SPAN)
                if src_dtype in (DataType.FP16, DataType.BF16, DataType.FP32)
                else ir.ConstInt(1, src_dtype, SPAN)
            )
            x = ib.emit_reg("x", "reg.broadcast", [scalar], {"lanes": 64, "dtype": src_dtype}, SPAN)
            y = ib.emit_reg("y", "reg.convert", [x], {"dtype": dst_dtype, **attrs}, SPAN)
            ib.emit_reg_store("out", dst, y, span=SPAN)
    with passes.PassContext([]):
        source = generate(passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN)))
    cvt = next(line for line in source.splitlines() if "pto.vmi.vcvt" in line)
    assert ("rounding" in cvt) == ("rounding" in attrs)
    assert ("saturate" in cvt) == ("saturate" in attrs)
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("dtype", [DataType.INT8, DataType.INT16, DataType.INT32])
def test_review_signed_broadcast(dtype, tmp_path):
    ib = IRBuilder()
    with ib.function("broadcast", span=SPAN, type=ir.FunctionType.AIV) as f:
        dst = ib.let("dst", tile.create([128], dtype=dtype, target_memory=ir.MemorySpace.Vec))
        with ib.reg_scope(SPAN):
            x = ib.emit_reg(
                "x", "reg.broadcast", [ir.ConstInt(-1, dtype, SPAN)], {"lanes": 64, "dtype": dtype}, SPAN
            )
            # Same-type conversion folds to a plain Reg alias assignment.
            alias = ib.emit_reg("alias", "reg.convert", [x], {"dtype": dtype}, SPAN)
            ib.emit_reg_store("out", dst, alias, span=SPAN)
    with passes.PassContext([]):
        source = generate(passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN)))
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("kind", ["for", "if"])
@pytest.mark.parametrize("mask_carry", [False, True])
def test_review_register_control_flow_ssa(kind, mask_carry, tmp_path):
    ib = IRBuilder()
    with ib.function("control", span=SPAN, type=ir.FunctionType.AIV) as f:
        condition = f.param("condition", ir.ScalarType(DataType.BOOL), SPAN)
        dst = ib.let("dst", tile.create([128], dtype=DataType.FP32, target_memory=ir.MemorySpace.Vec))
        with ib.reg_scope(SPAN):
            if mask_carry:
                initial = ib.emit_reg("mask", "reg.full_mask", [], {"lanes": 64}, SPAN)
            else:
                initial = ib.emit_reg(
                    "x",
                    "reg.broadcast",
                    [ir.ConstFloat(1, DataType.FP32, SPAN)],
                    {"lanes": 64, "dtype": DataType.FP32},
                    SPAN,
                )
            op = "reg.mask_not" if mask_carry else "reg.add"
            if kind == "if":
                with ib.if_stmt(condition, SPAN) as branch:
                    branch.return_var("result", initial.type, SPAN)
                    value = ib.emit_reg(
                        "next", op, [initial] if mask_carry else [initial, initial], span=SPAN
                    )
                    ib.emit(ir.YieldStmt([value], SPAN))
                    branch.else_(SPAN)
                    ib.emit(ir.YieldStmt([initial], SPAN))
                result = branch.output(0)
            else:
                with ib.for_loop(
                    ib.var("i", ir.ScalarType(DataType.INDEX), SPAN), 0, 3, 1, span=SPAN
                ) as loop:
                    carry = loop.iter_arg("carry", initial, span=SPAN)
                    result = loop.return_var("result", span=SPAN)
                    value = ib.emit_reg("next", op, [carry] if mask_carry else [carry, carry], span=SPAN)
                    ib.emit(ir.YieldStmt([value], SPAN))
            # Consume the carried mask through select: arbitrary-mask stores
            # additionally require PTOAS to prove the UB pointer's alignment.
            if mask_carry:
                x = ib.emit_reg(
                    "x",
                    "reg.broadcast",
                    [ir.ConstFloat(1, DataType.FP32, SPAN)],
                    {"lanes": 64, "dtype": DataType.FP32},
                    SPAN,
                )
                result = ib.emit_reg("selected", "reg.select", [result, x, x], span=SPAN)
            ib.emit_reg_store("out", dst, result, span=SPAN)
    with passes.PassContext([]):
        initialized = passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN))
        allocated = passes.allocate_memory_addr()(initialized)
        source = codegen.PTOCodegen().generate(allocated, emit_source_loc=False)
    control = next(line for line in source.splitlines() if f"scf.{kind}" in line)
    expected = "!pto.vmi.mask<64xpred>" if mask_carry else "!pto.vmi.vreg<64xf32>"
    assert expected in control
    assert "scf.yield" in source
    # The consumer must use the control-flow result, not the initial value.
    result_ssa = control.strip().split(" = scf.", 1)[0]
    opcode = "pto.vmi.vsel" if mask_carry else "pto.vmi.vstore"
    consumer = next(line for line in source.splitlines() if opcode in line)
    assert f" {result_ssa}," in consumer
    compile_review_pto(source, tmp_path, level="level3")


@pytest.mark.ptoas_compile
def test_review_partial_valid_tile_parameter(tmp_path):
    ib = IRBuilder()
    with ib.function("partial", span=SPAN, type=ir.FunctionType.AIV) as f:
        src = f.param(
            "src",
            ir.TileType(
                [1, 128],
                DataType.FP32,
                tile_view=ir.TileView([1, 64], [128, 1], integer(0)),
                memory_space=ir.MemorySpace.Vec,
            ),
            SPAN,
        )
        dst = ib.let("dst", tile.create([128], dtype=DataType.FP32, target_memory=ir.MemorySpace.Vec))
        with ib.reg_scope(SPAN):
            x = ib.emit_reg("x", "reg.load", [src, integer(0)], {"lanes": 64}, SPAN)
            ib.emit_reg_store("out", dst, x, span=SPAN)
    with passes.PassContext([]):
        initialized = passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN))
        # Parameter handles require the level3 UB pointer ABI.
        allocated = passes.allocate_memory_addr()(initialized)
    source = codegen.PTOCodegen().generate(allocated, emit_source_loc=False)
    assert "pto.vmi.vload" in source
    compile_review_pto(source, tmp_path, level="level3")


@pytest.mark.ptoas_compile
def test_review_address_annotation_uses_actual_view_ssa_type(tmp_path):
    ib = IRBuilder()
    with ib.function("view_address", span=SPAN, type=ir.FunctionType.AIV) as f:
        base = ib.let("base", tile.full([2, 128], dtype=DataType.FP32, value=1.0))
        view = ib.let(
            "view",
            tile.slice(base, [1, 128], [0, 0]),
            type=ir.TileType(
                [1, 128],
                DataType.FP32,
                tile_view=ir.TileView([1, 128], [128, 1], integer(0)),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        dst = ib.let("dst", tile.create([128], dtype=DataType.FP32, target_memory=ir.MemorySpace.Vec))
        with ib.reg_scope(SPAN):
            x = ib.emit_reg("x", "reg.load", [view, integer(0)], {"lanes": 64}, SPAN)
            ib.emit_reg_store("out", dst, x, span=SPAN)
    with passes.PassContext([]):
        source = generate(passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN)))
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
def test_review_loop_returned_view_address(tmp_path):
    ib = IRBuilder()
    with ib.function("loop_view", span=SPAN, type=ir.FunctionType.AIV) as f:
        base = ib.let("base", tile.full([2, 128], dtype=DataType.FP32, value=1.0))
        view = ib.let(
            "view",
            tile.slice(base, [1, 128], [0, 0]),
            type=ir.TileType(
                [1, 128],
                DataType.FP32,
                tile_view=ir.TileView([1, 128], [128, 1], integer(0)),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        with ib.reg_scope(SPAN):
            x = ib.emit_reg("x", "reg.load", [view, integer(0)], {"lanes": 64}, SPAN)
            with ib.for_loop(ib.var("i", ir.ScalarType(DataType.INDEX), SPAN), 0, 3, 1, span=SPAN) as loop:
                carry = loop.iter_arg("carry", view, span=SPAN)
                result = loop.return_var("result", span=SPAN)
                updated = ib.emit_reg_store("updated", carry, x, span=SPAN)
                ib.emit(ir.YieldStmt([updated], SPAN))
            # The loop result has the slice type, while root fallback has [2,128].
            y = ib.emit_reg("y", "reg.load", [result, integer(0)], {"lanes": 64}, SPAN)
            ib.emit_reg_store("out", view, y, span=SPAN)
    with passes.PassContext([]):
        initialized = passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN))
        source = generate(initialized)
    compile_review_pto(source, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("addressed", [False, True])
@pytest.mark.parametrize("nested_loop", [False, True])
def test_review_branch_returned_view_address(tmp_path, addressed, nested_loop):
    ib = IRBuilder()
    with ib.function("branch_view", span=SPAN, type=ir.FunctionType.AIV) as f:
        condition = f.param("condition", ir.ScalarType(DataType.BOOL), SPAN)
        base = ib.let("base", tile.full([2, 128], dtype=DataType.FP32, value=1.0))
        view = ib.let(
            "view",
            tile.slice(base, [1, 128], [0, 0]),
            type=ir.TileType(
                [1, 128],
                DataType.FP32,
                tile_view=ir.TileView([1, 128], [128, 1], integer(0)),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        with ib.reg_scope(SPAN):
            x = ib.emit_reg("x", "reg.load", [view, integer(0)], {"lanes": 64}, SPAN)
            with ib.if_stmt(condition, SPAN) as branch:
                branch.return_var("result", view.type, SPAN)
                updated = ib.emit_reg_store("then_updated", view, x, span=SPAN)
                if nested_loop:
                    with ib.for_loop(
                        ib.var("i", ir.ScalarType(DataType.INDEX), SPAN), 0, 3, 1, span=SPAN
                    ) as loop:
                        carry = loop.iter_arg("carry", updated, span=SPAN)
                        updated = loop.return_var("loop_result", span=SPAN)
                        stored = ib.emit_reg_store("loop_updated", carry, x, span=SPAN)
                        ib.emit(ir.YieldStmt([stored], SPAN))
                ib.emit(ir.YieldStmt([updated], SPAN))
                branch.else_(SPAN)
                updated = ib.emit_reg_store("else_updated", view, x, span=SPAN)
                ib.emit(ir.YieldStmt([updated], SPAN))
            # Hoisting precedes if-result binding. Resolve the shared external
            # slice, not its differently shaped [2,128] allocation fallback.
            y = ib.emit_reg("y", "reg.load", [branch.output(0), integer(0)], {"lanes": 64}, SPAN)
            ib.emit_reg_store("out", view, y, span=SPAN)
    with passes.PassContext([]):
        initialized = passes.init_mem_ref()(ir.Program([f.get_result()], "Review", SPAN))
        if addressed:
            initialized = passes.allocate_memory_addr()(initialized)
        source = codegen.PTOCodegen().generate(initialized, emit_tile_addr=addressed, emit_source_loc=False)
    # Every store updates the shared external view; no phi-buffer copy belongs
    # inside the register scope, under either memory-planning mode.
    assert "pto.tmov" not in source.split("pto.vecscope {", 1)[1]
    compile_review_pto(source, tmp_path, level="level3" if addressed else "level2")


FAMILY_SPAN = ir.Span.unknown()
UNARY = ["abs", "neg", "relu", "exp", "log", "sqrt", "bitwise_not"]
BINARY = ["bitwise_and", "bitwise_or", "bitwise_xor", "shift_left", "shift_right"]
SCALAR = ["adds", "muls", "maximums", "minimums", "shift_lefts", "shift_rights"]
REDUCE = ["reduce_sum", "reduce_max", "reduce_min"]
OPS = UNARY + BINARY + SCALAR + REDUCE + ["compares", "gather_lanes"]
OPCODES = dict(
    zip(
        OPS,
        [
            "vabs",
            "vneg",
            "vrelu",
            "vexp",
            "vln",
            "vsqrt",
            "vnot",
            "vand",
            "vor",
            "vxor",
            "vshl",
            "vshr",
            "vadds",
            "vmuls",
            "vmaxs",
            "vmins",
            "vshls",
            "vshrs",
            "vcadd",
            "vcmax",
            "vcmin",
            "vcmps",
            "vselr",
        ],
        strict=True,
    )
)


def vector(dtype=DataType.FP32, lanes=64):
    return ir.Var("x", ir.RegType(lanes, dtype, ir.RegKind.VecReg), FAMILY_SPAN)


def mask(lanes=64):
    return ir.Var("m", ir.RegType(lanes, DataType.BOOL, ir.RegKind.MaskReg), FAMILY_SPAN)


def invoke(op, x, m, masked=True):
    fn = getattr(pl.reg, op)
    if op in UNARY:
        return fn(x, m if masked else None).unwrap()
    if op in BINARY:
        return fn(x, x, m if masked else None).unwrap()
    if op in SCALAR:
        return fn(x, 1, m).unwrap()
    if op in REDUCE:
        return fn(x, m, **({"reassoc": True} if op == "reduce_sum" else {})).unwrap()
    if op == "compares":
        return fn(x, 1, m, predicate="ge").unwrap()
    idx_dtype = {1: DataType.UINT8, 2: DataType.UINT16, 4: DataType.UINT32}[x.type.dtype.get_byte()]
    idx = pl.reg.index_gen(0, lanes=64, dtype=idx_dtype)
    return fn(x, idx).unwrap()


def kernel(op, dtype, masked):
    ib = IRBuilder()
    with ib.function("extended", type=ir.FunctionType.AIV, span=FAMILY_SPAN) as f:
        ty = ir.TileType(
            [1, 128],
            dtype,
            tile_view=ir.TileView([1, 128], [128, 1], ir.ConstInt(0, DataType.INDEX, FAMILY_SPAN)),
            memory_space=ir.MemorySpace.Vec,
        )
        src = ib.let("src", tile.full([1, 128], dtype=dtype, value=1), type=ty)
        dst = ib.let("dst", tile.full([1, 128], dtype=dtype, value=0), type=ty)
        with ib.reg_scope(FAMILY_SPAN):
            x = ib.emit_reg(
                "x",
                "reg.load",
                [src, ir.ConstInt(0, DataType.INDEX, FAMILY_SPAN)],
                {"lanes": 64},
                FAMILY_SPAN,
            )
            m = ib.emit_reg(
                "m",
                "reg.active_mask",
                [ir.ConstInt(32, DataType.INDEX, FAMILY_SPAN)],
                {"lanes": 64},
                FAMILY_SPAN,
            )
            if op == "gather_lanes":
                idx = ib.emit_reg(
                    "idx",
                    "reg.index_gen",
                    [ir.ConstInt(0, DataType.UINT32, FAMILY_SPAN)],
                    {"lanes": 64, "dtype": DataType.UINT32},
                    FAMILY_SPAN,
                )
                call = pl.reg.gather_lanes(x, idx).unwrap()
            else:
                call = invoke(op, x, m, masked)
            y = ib.let("y", call)
            if op == "compares":
                y = ib.emit_reg("selected", "reg.select", [y, x, x], span=FAMILY_SPAN)
            ib.emit_reg_store("out", dst, y, span=FAMILY_SPAN)
    with passes.PassContext([]):
        return passes.init_mem_ref()(ir.Program([f.get_result()], "Extended", FAMILY_SPAN))


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("masked", [False, True])
def test_basic_family_compiles(op, masked, tmp_path):
    dtype = (
        DataType.INT32
        if op in BINARY or op in ("bitwise_not", "shift_lefts", "shift_rights")
        else DataType.FP32
    )
    mlir = codegen.PTOCodegen().generate(
        kernel(op, dtype, masked), emit_tile_addr=False, emit_source_loc=False
    )
    assert f"pto.vmi.{OPCODES[op]} " in mlir
    if op in REDUCE:
        assert "-> !pto.vmi.vreg<1xf32>" in mlir
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    source = tmp_path / "extended.pto"
    source.write_text(mlir)
    result = subprocess.run(
        [compiler, str(source), "--pto-arch=a5", "--emit-vpto", "-o", str(tmp_path / "out.vpto")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"{mlir}\n{result.stderr}"


@pytest.mark.parametrize(
    "op", ["adds", "muls", "maximums", "minimums", "compares", "shift_lefts", "shift_rights"]
)
@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.UINT32])
@pytest.mark.ptoas_compile
def test_integer_scalar_compiles(op, dtype, tmp_path):
    mlir = codegen.PTOCodegen().generate(kernel(op, dtype, True), emit_tile_addr=False, emit_source_loc=False)
    source = tmp_path / "integer.pto"
    source.write_text(mlir)
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    result = subprocess.run(
        [compiler, str(source), "--pto-arch=a5", "--emit-vpto", "-o", str(tmp_path / "out.vpto")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"{mlir}\n{result.stderr}"


@pytest.mark.parametrize(
    "op,dtype",
    [
        ("abs", DataType.UINT32),
        ("neg", DataType.BF16),
        ("relu", DataType.INT16),
        ("exp", DataType.INT32),
        ("log", DataType.INT32),
        ("sqrt", DataType.INT32),
        ("bitwise_not", DataType.FP32),
        ("shift_left", DataType.FP32),
        ("muls", DataType.INT8),
        ("reduce_sum", DataType.INT8),
        ("reduce_max", DataType.BF16),
        ("compares", DataType.INT16),
    ],
)
def test_unsupported_dtype_rejected(op, dtype):
    with pytest.raises((ValueError, TypeError), match="dtype|type|support"):
        invoke(op, vector(dtype), mask())


def test_float_sum_requires_explicit_reassociation():
    with pytest.raises(ValueError, match="reassoc"):
        pl.reg.reduce_sum(vector(), mask())


@pytest.mark.parametrize("op", UNARY + BINARY + SCALAR + REDUCE + ["compares"])
def test_mask_lane_mismatch_rejected(op):
    dtype = DataType.INT32 if "bitwise" in op or "shift" in op else DataType.FP32
    with pytest.raises(ValueError, match="[Mm]ask|lane"):
        invoke(op, vector(dtype), mask(32))


def test_scalar_ssa_is_not_implicitly_cast():
    with pytest.raises(ValueError, match="[Ss]calar|dtype"):
        pl.reg.adds(vector(), ir.Var("s", ir.ScalarType(DataType.INT32), FAMILY_SPAN), mask())


def test_scalar_shift_requires_i16_ssa():
    with pytest.raises(ValueError, match="[Ss]calar|i16|dtype"):
        pl.reg.shift_lefts(
            vector(DataType.INT32), ir.Var("s", ir.ScalarType(DataType.INT32), FAMILY_SPAN), mask()
        )


@pytest.mark.parametrize("lanes,dtype", [(32, DataType.UINT32), (64, DataType.UINT16)])
def test_gather_rejects_unsupported_lanes_or_index_width(lanes, dtype):
    with pytest.raises(ValueError, match="lane|width"):
        pl.reg.gather_lanes(vector(lanes=lanes), vector(dtype, lanes))


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize(
    "direction,policy",
    [("load", policy) for policy in pl.CachePolicy if int(policy) >= 2]
    + [("load", pl.CachePolicy.BYPASS)]
    + [
        ("store", policy)
        for policy in pl.CachePolicy
        if int(policy) >= 2 and policy not in (pl.CachePolicy.NOT_ALLOC_KEEP, pl.CachePolicy.NOT_ALLOC_DROP)
    ],
)
@pytest.mark.ptoas_compile
def test_register_unit_native_verifier_rejects_a5_cache_modes(tmp_path, planner, direction, policy):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is unavailable")

    load_cache = f", cache=pl.{type(policy).__name__}.{policy.name}" if direction == "load" else ""
    store_cache = f", cache=pl.CachePolicy.{policy.name}" if direction == "store" else ""
    program = pl.parse_program(f"""
@pl.program
class RegisterCache:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[[1, 128], pl.FP32],
               dst: pl.InOut[pl.Tensor[[1, 128], pl.FP32]]):
        tile_src: pl.Tile[
            [1, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 128], stride=[128, 1])
        ] = pl.load(
            src, [0, 0], [1, 128], target_memory=pl.Mem.Vec{load_cache})
        tile_dst: pl.Tile[
            [1, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 128], stride=[128, 1])
        ] = pl.load(
            dst, [0, 0], [1, 128], target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            value = pl.reg.load(tile_src, lanes=64)
            doubled = pl.reg.add(value, value)
            out = pl.reg.store(tile_dst, value=doubled)
        stored = pl.store(out, [0, 0], dst{store_cache})
""")
    with passes.PassContext([], memory_planner=planner):
        optimized = ir.PassManager(ir.OptimizationStrategy.Default).run_passes(program)
    source = codegen.PTOCodegen().generate(
        optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
    )
    assert 'pto.backend = "vpto"' in source.partition("\n")[0]
    assert f"l2_hint = #pto.{direction}_l2_hint<" in source
    # Validation belongs to PTOAS, including standalone binary compilation.
    with pytest.raises(RuntimeError, match="l2_hint requires the EmitC backend"):
        pto_backend._compile_pto_module(source, "register_cache", str(tmp_path), planner)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("rounding", ["R", "A", "Z"])
@pytest.mark.parametrize("saturate", [False, True])
def test_fp32_to_fp8_register_conversion_preserves_compact_bytes(tmp_path, rounding, saturate):
    """Window publication needs 64 contiguous FP8 bytes after FP32 conversion."""
    ib = IRBuilder()
    with ib.function("fp8_publish", span=SPAN, type=ir.FunctionType.AIV) as f:
        src = ib.let("src", tile.full([64], dtype=DataType.FP32, value=1.0))
        dst = ib.let("dst", tile.create([64], dtype=DataType.UINT8, target_memory=ir.MemorySpace.Vec))
        with ib.reg_scope(SPAN):
            value = ib.emit_reg("value", "reg.load", [src, integer(0)], {"lanes": 64}, SPAN)
            fp8 = ib.emit_reg(
                "fp8",
                "reg.convert",
                [value],
                {"dtype": DataType.FP8E4M3FN, "rounding": rounding, "saturate": saturate},
                SPAN,
            )
            packed = ib.emit_reg("packed", "reg.interpret_cast", [fp8], {"dtype": DataType.UINT8}, SPAN)
            ib.emit_reg_store("written", dst, packed, span=SPAN)
    with passes.PassContext([]):
        source = generate(passes.init_mem_ref()(ir.Program([f.get_result()], "FP8Publish", SPAN)))
    assert "!pto.vmi.vreg<64xf8E4M3FN>" in source
    compile_review_pto(source, tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
