module attributes {pto.target_arch = "a5"} {
  func.func @tile_softmax(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %c16448_i64 = arith.constant 16448 : i64
    %c16416_i64 = arith.constant 16416 : i64
    %c12320_i64 = arith.constant 12320 : i64
    %c8224_i64 = arith.constant 8224 : i64
    %c4128_i64 = arith.constant 4128 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%c1, %c1024], strides: [%c1024, %c1] {layout = #pto.layout<nd>} : memref<?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %subview = memref.subview %reinterpret_cast[0, 0] [1, 1024] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<1x1024xf32, strided<[?, ?], offset: ?>>
    %0 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>>
    %1 = pto.bind_tile %0, %c1, %c1024 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>> -> memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>
    pto.tload ins(%subview : memref<1x1024xf32, strided<[?, ?], offset: ?>>) outs(%1 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
    %2 = pto.pointer_cast(%c4096_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<8x1xf32, strided<[1, 8]>, #pto.address_space<vec>>
    %3 = pto.bind_tile %2, %c1, %c1 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<8x1xf32, strided<[1, 8]>, #pto.address_space<vec>> -> memref<8x1xf32, strided<[1, 8], offset: ?>, #pto.address_space<vec>>
    %4 = pto.pointer_cast(%c4128_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>>
    %5 = pto.bind_tile %4, %c1, %c1024 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>> -> memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>
    %6 = pto.pointer_cast(%c8224_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>>
    %7 = pto.bind_tile %6, %c1, %c1024 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>> -> memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>
    %8 = pto.pointer_cast(%c12320_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>>
    %9 = pto.bind_tile %8, %c1, %c1024 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>> -> memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>
    %10 = pto.pointer_cast(%c16416_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<8x1xf32, strided<[1, 8]>, #pto.address_space<vec>>
    %11 = pto.bind_tile %10, %c1, %c1 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<8x1xf32, strided<[1, 8]>, #pto.address_space<vec>> -> memref<8x1xf32, strided<[1, 8], offset: ?>, #pto.address_space<vec>>
    %12 = pto.pointer_cast(%c16448_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>>
    %13 = pto.bind_tile %12, %c1, %c1024 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x1024xf32, strided<[1024, 1]>, #pto.address_space<vec>> -> memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>
    pto.trowmax ins(%1, %5 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>, memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>) outs(%3 : memref<8x1xf32, strided<[1, 8], offset: ?>, #pto.address_space<vec>>)
    pto.trowexpandsub ins(%1, %3 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>, memref<8x1xf32, strided<[1, 8], offset: ?>, #pto.address_space<vec>>) outs(%7 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>)
    pto.texp ins(%7 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>) outs(%9 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>)
    pto.trowsum ins(%9, %5 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>, memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>) outs(%11 : memref<8x1xf32, strided<[1, 8], offset: ?>, #pto.address_space<vec>>)
    pto.trowexpanddiv ins(%9, %11 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>, memref<8x1xf32, strided<[1, 8], offset: ?>, #pto.address_space<vec>>) outs(%13 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>)
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%c1, %c1024], strides: [%c1024, %c1] {layout = #pto.layout<nd>} : memref<?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %subview_1 = memref.subview %reinterpret_cast_0[0, 0] [1, 1024] [1, 1] {layout = #pto.layout<nd>} : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<1x1024xf32, strided<[?, ?], offset: ?>>
    pto.tstore ins(%13 : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #pto.address_space<vec>>) outs(%subview_1 : memref<1x1024xf32, strided<[?, ?], offset: ?>>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
    return
  }
}
