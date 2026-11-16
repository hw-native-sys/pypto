// Hand-written stage-2: a pto.tmatmul where the lhs tile lives in `vec`
// (Ub) instead of `left`. Triggers SafetyKind::OpConstraint.
module attributes {pto.target_arch = "a5"} {
  func.func @bad_matmul_placement(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    // lhs — WRONG: placed in `vec` instead of `left`.
    %0 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xf16, strided<[32, 16]>, #pto.address_space<vec>>
    %1 = pto.bind_tile %0, %c16, %c128 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xf16, strided<[32, 16]>, #pto.address_space<vec>> -> memref<16x128xf16, strided<[32, 16], offset: ?>, #pto.address_space<vec>>
    // rhs — correct (right).
    %2 = pto.pointer_cast(%c1024_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<col_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xf16, strided<[256, 32]>, #pto.address_space<right>>
    %3 = pto.bind_tile %2, %c128, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<col_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xf16, strided<[256, 32]>, #pto.address_space<right>> -> memref<128x256xf16, strided<[256, 32], offset: ?>, #pto.address_space<right>>
    // acc — correct.
    %4 = pto.pointer_cast(%c2048_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=1024, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x256xf32, strided<[16, 16]>, #pto.address_space<acc>>
    %5 = pto.bind_tile %4, %c16, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=1024, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x256xf32, strided<[16, 16]>, #pto.address_space<acc>> -> memref<16x256xf32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>
    pto.tmatmul ins(%1, %3 : memref<16x128xf16, strided<[32, 16], offset: ?>, #pto.address_space<vec>>, memref<128x256xf16, strided<[256, 32], offset: ?>, #pto.address_space<right>>) outs(%5 : memref<16x256xf32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>)
    return
  }
}
