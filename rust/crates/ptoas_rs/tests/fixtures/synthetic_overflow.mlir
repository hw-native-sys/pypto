// Hand-written stage-2: a Left tile whose high-water mark (offset + size)
// exceeds the 64 KiB Left cap on 910B2/CANN 8.5. Triggers
// SafetyKind::Capacity.
module attributes {pto.target_arch = "a5"} {
  func.func @overflow(%arg0: memref<?xi8>) {
    %c0_i64 = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c65537 = arith.constant 65537 : index
    %0 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x65537xi8, strided<[65537, 1]>, #pto.address_space<left>>
    %1 = pto.bind_tile %0, %c1, %c65537 {config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>, compact=#pto.compact_mode<null>>} : memref<1x65537xi8, strided<[65537, 1]>, #pto.address_space<left>> -> memref<1x65537xi8, strided<[65537, 1], offset: ?>, #pto.address_space<left>>
    return
  }
}
