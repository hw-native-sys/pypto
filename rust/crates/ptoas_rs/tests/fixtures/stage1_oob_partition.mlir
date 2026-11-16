// Stage-1 PTO MLIR — manually-constructed OOB partition_view.
// Parent tensor view is 1×1024; partition reads at offset [2, 166] with size
// [1, 1024]. Both row offset (2 >= 1 parent rows) and col extent (166+1024
// = 1190 > 1024 parent cols) exceed parent bounds.
// This is the byte-for-byte reproduction of pre-fix mlir_to_pto GEP bug
// 06d8d30b's runtime SSA — see project_neurips_oob_oracle_validated.md.
// Expected: pto-diff --stage1 reports 1 OobPartition violation on @bad
// whose message mentions both axes (end [3,1190] vs parent [1,1024]).
module {
  func.func @bad(%arg0: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c166 = arith.constant 166 : index
    %c1024 = arith.constant 1024 : index
    %pto0 = pto.make_tensor_view %arg0, shape = [%c1, %c1024], strides = [%c1024, %c1] : !pto.tensor_view<?x?xf32>
    %pto1 = pto.partition_view %pto0, offsets = [%c2, %c166], sizes = [%c1, %c1024] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1024xf32>
    return
  }
}
