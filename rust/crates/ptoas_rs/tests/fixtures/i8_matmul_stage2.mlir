module attributes {pto.target_arch = "a5"} {
  func.func @kv_proj_i8_dequant_f16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xui64>, %arg3: memref<?xf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1536 = arith.constant 1536 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%c16, %c1536], strides: [%c1536, %c1] {layout = #pto.layout<nd>} : memref<?xi8> to memref<?x?xi8, strided<[?, ?], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%c1536, %c256], strides: [%c256, %c1] {layout = #pto.layout<nd>} : memref<?xi8> to memref<?x?xi8, strided<[?, ?], offset: ?>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%c1, %c256], strides: [%c256, %c1] {layout = #pto.layout<nd>} : memref<?xui64> to memref<?x?xui64, strided<[?, ?], offset: ?>>
    %0 = pto.pointer_cast(%c2048_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xi8, strided<[32, 16]>, #pto.address_space<mat>>
    %1 = pto.bind_tile %0, %c16, %c128 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xi8, strided<[32, 16]>, #pto.address_space<mat>> -> memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<mat>>
    %2 = pto.pointer_cast(%c4096_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xi8, strided<[32, 128]>, #pto.address_space<mat>>
    %3 = pto.bind_tile %2, %c128, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xi8, strided<[32, 128]>, #pto.address_space<mat>> -> memref<128x256xi8, strided<[32, 128], offset: ?>, #pto.address_space<mat>>
    %4 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xi8, strided<[32, 16]>, #pto.address_space<left>>
    %5 = pto.bind_tile %4, %c16, %c128 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xi8, strided<[32, 16]>, #pto.address_space<left>> -> memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<left>>
    %6 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<col_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xi8, strided<[256, 32]>, #pto.address_space<right>>
    %7 = pto.bind_tile %6, %c128, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<col_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xi8, strided<[256, 32]>, #pto.address_space<right>> -> memref<128x256xi8, strided<[256, 32], offset: ?>, #pto.address_space<right>>
    %8 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=1024, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x256xi32, strided<[16, 16]>, #pto.address_space<acc>>
    %9 = pto.bind_tile %8, %c16, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=1024, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x256xi32, strided<[16, 16]>, #pto.address_space<acc>> -> memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>
    %10 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=32, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x256xui64, strided<[256, 1]>, #pto.address_space<mat>>
    %11 = pto.bind_tile %10, %c1, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=32, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x256xui64, strided<[256, 1]>, #pto.address_space<mat>> -> memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<mat>>
    %12 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=32, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x256xui64, strided<[256, 1]>, #pto.address_space<scaling>>
    %13 = pto.bind_tile %12, %c1, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=32, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x256xui64, strided<[256, 1]>, #pto.address_space<scaling>> -> memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<scaling>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%c16, %c256], strides: [%c256, %c1] {layout = #pto.layout<nd>} : memref<?xf16> to memref<?x?xf16, strided<[?, ?], offset: ?>>
    %subview = memref.subview %reinterpret_cast_1[0, 0] [1, 256] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xui64, strided<[?, ?], offset: ?>> to memref<1x256xui64, strided<[?, ?], offset: ?>>
    pto.tload ins(%subview : memref<1x256xui64, strided<[?, ?], offset: ?>>) outs(%11 : memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<mat>>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
    pto.tmov ins(%11 : memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<mat>>) outs(%13 : memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<scaling>>)
    scf.for %arg4 = %c0 to %c12 step %c1 {
      %14 = arith.muli %arg4, %c128 : index
      %subview_4 = memref.subview %reinterpret_cast[0, %14] [16, 128] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xi8, strided<[?, ?], offset: ?>> to memref<16x128xi8, strided<[?, ?], offset: ?>>
      pto.tload ins(%subview_4 : memref<16x128xi8, strided<[?, ?], offset: ?>>) outs(%1 : memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<mat>>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
      pto.tmov ins(%1 : memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<mat>>) outs(%5 : memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<left>>)
      %subview_5 = memref.subview %reinterpret_cast_0[%14, 0] [128, 256] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xi8, strided<[?, ?], offset: ?>> to memref<128x256xi8, strided<[?, ?], offset: ?>>
      pto.tload ins(%subview_5 : memref<128x256xi8, strided<[?, ?], offset: ?>>) outs(%3 : memref<128x256xi8, strided<[32, 128], offset: ?>, #pto.address_space<mat>>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
      pto.tmov ins(%3 : memref<128x256xi8, strided<[32, 128], offset: ?>, #pto.address_space<mat>>) outs(%7 : memref<128x256xi8, strided<[256, 32], offset: ?>, #pto.address_space<right>>)
      %15 = arith.cmpi eq, %arg4, %c0 : index
      scf.if %15 {
        pto.tmatmul ins(%5, %7 : memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<left>>, memref<128x256xi8, strided<[256, 32], offset: ?>, #pto.address_space<right>>) outs(%9 : memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>)
      } else {
        pto.tmatmul.acc ins(%9, %5, %7 : memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>, memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<left>>, memref<128x256xi8, strided<[256, 32], offset: ?>, #pto.address_space<right>>) outs(%9 : memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>)
      }
    }
    %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [16, 256] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<16x256xf16, strided<[?, ?], offset: ?>>
    pto.tstore_fp ins(%9, %13 : memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>, memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<scaling>>) outs(%subview_3 : memref<16x256xf16, strided<[?, ?], offset: ?>>)
    return
  }
  func.func @gate_up_i8_dequant_f16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xui64>, %arg3: memref<?xf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c34816_i64 = arith.constant 34816 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index
    %c70 = arith.constant 70 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1536 = arith.constant 1536 : index
    %c17920 = arith.constant 17920 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%c16, %c1536], strides: [%c1536, %c1] {layout = #pto.layout<nd>} : memref<?xi8> to memref<?x?xi8, strided<[?, ?], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%c1536, %c17920], strides: [%c17920, %c1] {layout = #pto.layout<nd>} : memref<?xi8> to memref<?x?xi8, strided<[?, ?], offset: ?>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%c1, %c17920], strides: [%c17920, %c1] {layout = #pto.layout<nd>} : memref<?xui64> to memref<?x?xui64, strided<[?, ?], offset: ?>>
    %0 = pto.pointer_cast(%c34816_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xi8, strided<[32, 16]>, #pto.address_space<mat>>
    %1 = pto.bind_tile %0, %c16, %c128 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xi8, strided<[32, 16]>, #pto.address_space<mat>> -> memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<mat>>
    %2 = pto.pointer_cast(%c2048_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xi8, strided<[32, 128]>, #pto.address_space<mat>>
    %3 = pto.bind_tile %2, %c128, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xi8, strided<[32, 128]>, #pto.address_space<mat>> -> memref<128x256xi8, strided<[32, 128], offset: ?>, #pto.address_space<mat>>
    %4 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xi8, strided<[32, 16]>, #pto.address_space<left>>
    %5 = pto.bind_tile %4, %c16, %c128 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x128xi8, strided<[32, 16]>, #pto.address_space<left>> -> memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<left>>
    %6 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<col_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xi8, strided<[256, 32]>, #pto.address_space<right>>
    %7 = pto.bind_tile %6, %c128, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<col_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<128x256xi8, strided<[256, 32]>, #pto.address_space<right>> -> memref<128x256xi8, strided<[256, 32], offset: ?>, #pto.address_space<right>>
    %8 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=1024, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x256xi32, strided<[16, 16]>, #pto.address_space<acc>>
    %9 = pto.bind_tile %8, %c16, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=1024, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<16x256xi32, strided<[16, 16]>, #pto.address_space<acc>> -> memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>
    %10 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=32, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x256xui64, strided<[256, 1]>, #pto.address_space<mat>>
    %11 = pto.bind_tile %10, %c1, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=32, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x256xui64, strided<[256, 1]>, #pto.address_space<mat>> -> memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<mat>>
    %12 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=32, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x256xui64, strided<[256, 1]>, #pto.address_space<scaling>>
    %13 = pto.bind_tile %12, %c1, %c256 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=32, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x256xui64, strided<[256, 1]>, #pto.address_space<scaling>> -> memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<scaling>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%c16, %c17920], strides: [%c17920, %c1] {layout = #pto.layout<nd>} : memref<?xf16> to memref<?x?xf16, strided<[?, ?], offset: ?>>
    %14 = pto.get_block_idx
    %15 = pto.get_block_num
    %16 = arith.index_cast %14 : i64 to index
    %17 = arith.index_cast %15 : i64 to index
    scf.for %arg4 = %16 to %c70 step %17 {
      %18 = arith.muli %arg4, %c256 : index
      scf.for %arg5 = %c0 to %c12 step %c1 {
        %19 = arith.muli %arg5, %c128 : index
        %subview_4 = memref.subview %reinterpret_cast[0, %19] [16, 128] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xi8, strided<[?, ?], offset: ?>> to memref<16x128xi8, strided<[?, ?], offset: ?>>
        pto.tload ins(%subview_4 : memref<16x128xi8, strided<[?, ?], offset: ?>>) outs(%1 : memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<mat>>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
        pto.tmov ins(%1 : memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<mat>>) outs(%5 : memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<left>>)
        %subview_5 = memref.subview %reinterpret_cast_0[%19, %18] [128, 256] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xi8, strided<[?, ?], offset: ?>> to memref<128x256xi8, strided<[?, ?], offset: ?>>
        pto.tload ins(%subview_5 : memref<128x256xi8, strided<[?, ?], offset: ?>>) outs(%3 : memref<128x256xi8, strided<[32, 128], offset: ?>, #pto.address_space<mat>>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
        pto.tmov ins(%3 : memref<128x256xi8, strided<[32, 128], offset: ?>, #pto.address_space<mat>>) outs(%7 : memref<128x256xi8, strided<[256, 32], offset: ?>, #pto.address_space<right>>)
        %20 = arith.cmpi eq, %arg5, %c0 : index
        scf.if %20 {
          pto.tmatmul ins(%5, %7 : memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<left>>, memref<128x256xi8, strided<[256, 32], offset: ?>, #pto.address_space<right>>) outs(%9 : memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>)
        } else {
          pto.tmatmul.acc ins(%9, %5, %7 : memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>, memref<16x128xi8, strided<[32, 16], offset: ?>, #pto.address_space<left>>, memref<128x256xi8, strided<[256, 32], offset: ?>, #pto.address_space<right>>) outs(%9 : memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>)
        }
      }
      %subview = memref.subview %reinterpret_cast_2[0, %18] [16, 256] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<16x256xf16, strided<[?, ?], offset: ?>>
      %subview_3 = memref.subview %reinterpret_cast_1[0, %18] [1, 256] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xui64, strided<[?, ?], offset: ?>> to memref<1x256xui64, strided<[?, ?], offset: ?>>
      pto.tload ins(%subview_3 : memref<1x256xui64, strided<[?, ?], offset: ?>>) outs(%11 : memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<mat>>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
      pto.tmov ins(%11 : memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<mat>>) outs(%13 : memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<scaling>>)
      pto.tstore_fp ins(%9, %13 : memref<16x256xi32, strided<[16, 16], offset: ?>, #pto.address_space<acc>>, memref<1x256xui64, strided<[256, 1], offset: ?>, #pto.address_space<scaling>>) outs(%subview : memref<16x256xf16, strided<[?, ?], offset: ?>>)
    }
    return
  }
}
