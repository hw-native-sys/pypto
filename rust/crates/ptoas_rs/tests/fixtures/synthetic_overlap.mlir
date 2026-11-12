// Hand-written stage-2: two tiles in UB with different shapes occupying
// overlapping byte ranges. Triggers SafetyKind::Aliasing.
module attributes {pto.target_arch = "a5"} {
  func.func @overlap(%arg0: memref<?xf32>) {
    %c0_i64 = arith.constant 0 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %0 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>>
    %1 = pto.bind_tile %0, %c1, %c1024 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>> -> memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>
    // Overlaps %1's range [0, 4096) — different dtype so it's unintentional.
    %2 = pto.pointer_cast(%c2048_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x512xf16, strided<[512, 1]>, #pto.address_space<vec>>
    %3 = pto.bind_tile %2, %c1, %c512 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x512xf16, strided<[512, 1]>, #pto.address_space<vec>> -> memref<1x512xf16, strided<[512, 1], offset: ?>, #pto.address_space<vec>>
    return
  }
}
