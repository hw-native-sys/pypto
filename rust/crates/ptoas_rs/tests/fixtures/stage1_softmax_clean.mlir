// Stage-1 PTO MLIR — clean softmax slice.
// Parent tensor view is 1×1024; partition reads at offset [0, 0]
// with size [1, 1024]. Fully in-bounds.
// Expected: pto-diff --stage1 reports 0 findings.
module {
  func.func @tile_softmax(%arg601: !pto.ptr<f32>, %arg602: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %pto0 = pto.make_tensor_view %arg601, shape = [%c1, %c1024], strides = [%c1024, %c1] : !pto.tensor_view<?x?xf32>
    %pto1 = pto.partition_view %pto0, offsets = [%c0, %c0], sizes = [%c1, %c1024] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1024xf32>
    return
  }
}
