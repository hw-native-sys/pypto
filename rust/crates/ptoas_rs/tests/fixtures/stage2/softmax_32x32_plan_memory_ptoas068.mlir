// -----// IR Dump After PlanMemoryModern (pto-plan-memory) //----- //
module attributes {pto.backend = "emitc", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a3"} {
  func.func @op_softmax(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>) {
    %c25088_i64 = arith.constant 25088 : i64
    %c8448_i64 = arith.constant 8448 : i64
    %c33280_i64 = arith.constant 33280 : i64
    %c24832_i64 = arith.constant 24832 : i64
    %c20736_i64 = arith.constant 20736 : i64
    %c16640_i64 = arith.constant 16640 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = pto.make_tensor_view %arg0, shape = [%c32, %c32], strides = [%c32, %c1] {layout = #pto.layout<nd>, pto.inferred_layout = true} : !pto.tensor_view<?x?xf32>
    %1 = pto.partition_view %0, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32>
    %2 = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<vec, 32x32xf32>
    pto.tload ins(%1 : !pto.partition_tensor_view<32x32xf32>) outs(%2 : !pto.tile_buf<vec, 32x32xf32>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
    %3 = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<vec, 32x1xf32, blayout=col_major>
    %4 = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<vec, 32x32xf32>
    %5 = pto.alloc_tile addr = %c16640_i64 : !pto.tile_buf<vec, 32x32xf32>
    %6 = pto.alloc_tile addr = %c20736_i64 : !pto.tile_buf<vec, 32x32xf32>
    %7 = pto.alloc_tile addr = %c24832_i64 : !pto.tile_buf<vec, 32x1xf32, blayout=col_major>
    %8 = pto.alloc_tile addr = %c33280_i64 : !pto.tile_buf<vec, 32x32xf32>
    pto.trowmax ins(%2, %4 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) outs(%3 : !pto.tile_buf<vec, 32x1xf32, blayout=col_major>)
    %9 = pto.alloc_tile addr = %c8448_i64 : !pto.tile_buf<vec, 1x2048xf32>
    pto.trowexpandsub ins(%2, %3, %9 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x1xf32, blayout=col_major>, !pto.tile_buf<vec, 1x2048xf32>) outs(%5 : !pto.tile_buf<vec, 32x32xf32>)
    pto.texp ins(%5 : !pto.tile_buf<vec, 32x32xf32>) outs(%6 : !pto.tile_buf<vec, 32x32xf32>)
    pto.trowsum ins(%6, %4 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) outs(%7 : !pto.tile_buf<vec, 32x1xf32, blayout=col_major>)
    %10 = pto.alloc_tile addr = %c25088_i64 : !pto.tile_buf<vec, 1x2048xf32>
    pto.trowexpanddiv ins(%6, %7, %10 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x1xf32, blayout=col_major>, !pto.tile_buf<vec, 1x2048xf32>) outs(%8 : !pto.tile_buf<vec, 32x32xf32>) {precisionType = #pto<div_precision default>}
    %11 = pto.make_tensor_view %arg1, shape = [%c32, %c32], strides = [%c32, %c1] {layout = #pto.layout<nd>, pto.inferred_layout = true} : !pto.tensor_view<?x?xf32>
    %12 = pto.partition_view %11, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32>
    pto.tstore ins(%8 : !pto.tile_buf<vec, 32x32xf32>) outs(%12 : !pto.partition_tensor_view<32x32xf32>) {layout = #pto.layout<nd>, pto.inferred_layout = true}
    return
  }
}
