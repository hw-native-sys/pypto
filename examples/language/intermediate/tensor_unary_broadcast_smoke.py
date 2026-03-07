import pypto.language as pl
from pypto import ir
from pypto.backend import BackendType


@pl.program
class TensorUnaryBroadcastSmoke:
    @pl.function(type=pl.FunctionType.Opaque)
    def kernel(
        self,
        x: pl.Tensor[[16, 128], pl.FP32],
        row: pl.Tensor[[16, 1], pl.FP32],
        col: pl.Tensor[[1, 128], pl.FP32],
        out: pl.Tensor[[16, 128], pl.FP32],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        with pl.auto_incore():
            tmp = pl.rsqrt(pl.add(x, 1.0))
            tmp = pl.row_expand_mul(tmp, row)
            tmp = pl.col_expand_mul(tmp, col)
            tmp = pl.minimum(pl.relu(pl.neg(tmp)), tmp)
            out = pl.assemble(out, tmp, [0, 0])
        return out


if __name__ == "__main__":
    output_dir = ir.compile(
        TensorUnaryBroadcastSmoke,
        output_dir="build/tensor_unary_broadcast_smoke",
        dump_passes=False,
        backend_type=BackendType.PTO,
        skip_ptoas=True,
    )
    print(output_dir)
