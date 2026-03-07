import pypto.language as pl
from pypto import ir
from pypto.backend import BackendType


@pl.program
class TensorOpCoverageSmoke:
    @pl.function(type=pl.FunctionType.Opaque)
    def fp_ops(
        self,
        x: pl.Tensor[[16, 128], pl.FP32],
        row: pl.Tensor[[16, 1], pl.FP32],
        col: pl.Tensor[[1, 128], pl.FP32],
        y: pl.Tensor[[16, 128], pl.FP32],
        z: pl.Tensor[[16, 128], pl.FP32],
        out: pl.Tensor[[16, 128], pl.FP32],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        with pl.auto_incore():
            tmp = pl.adds(x, 1.0)
            tmp = pl.subs(tmp, 2.0)
            tmp = pl.muls(tmp, 3.0)
            tmp = pl.divs(tmp, 4.0)
            tmp = pl.rems(tmp, 5.0)
            tmp = pl.maxs(tmp, 0.0)
            tmp = pl.mins(tmp, 6.0)
            tmp = pl.lrelu(tmp, 0.1)
            tmp = pl.expands(tmp, 2.0)
            tmp = pl.row_expand_mul(tmp, row)
            tmp = pl.col_expand_mul(tmp, col)
            tmp = pl.addc(tmp, y, z)
            tmp = pl.subc(tmp, y, z)
            tmp = pl.addsc(tmp, 1.0, y)
            tmp = pl.subsc(tmp, 1.0, y)
            tmp = pl.prelu(tmp, y, z)
            tmp = pl.minimum(pl.relu(pl.neg(tmp)), tmp)
            red = pl.sum(tmp)
            mx = pl.max(tmp)
            mn = pl.min(tmp)
            rmn = pl.row_min(tmp)
            tmp = pl.add(pl.add(red, mx), pl.add(mn, rmn))
            out = pl.assemble(out, tmp, [0, 0])
        return out

    @pl.function(type=pl.FunctionType.Opaque)
    def int_ops(
        self,
        x: pl.Tensor[[16, 128], pl.INT32],
        y: pl.Tensor[[16, 128], pl.INT32],
        z: pl.Tensor[[16, 128], pl.INT32],
        out: pl.Tensor[[16, 128], pl.INT32],
    ) -> pl.Tensor[[16, 128], pl.INT32]:
        with pl.auto_incore():
            tmp = pl.and_(x, y)
            tmp = pl.ands(tmp, 7)
            tmp = pl.or_(tmp, y)
            tmp = pl.ors(tmp, 3)
            tmp = pl.xor(tmp, y, z)
            tmp = pl.xors(tmp, 1, z)
            tmp = pl.shl(tmp, y)
            tmp = pl.shls(tmp, 1)
            tmp = pl.shr(tmp, y)
            tmp = pl.shrs(tmp, 1)
            tmp = pl.not_(tmp)
            mask = pl.cmp(tmp, y, cmp_type=0)
            mask = pl.cmps(mask, 0, cmp_type=1)
            tmp = pl.sel(mask, tmp, y)
            tmp = pl.sels(tmp, y, 0)
            out = pl.assemble(out, tmp, [0, 0])
        return out

    @pl.function(type=pl.FunctionType.Opaque)
    def matmul_ops(
        self,
        acc: pl.Tensor[[16, 64], pl.FP32],
        lhs: pl.Tensor[[16, 32], pl.FP16],
        rhs: pl.Tensor[[32, 64], pl.FP16],
        bias: pl.Tensor[[1, 64], pl.FP32],
        out: pl.Tensor[[16, 64], pl.FP32],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        with pl.auto_incore():
            tmp = pl.matmul_bias(lhs, rhs, bias)
            tmp = pl.matmul_acc(acc, lhs, rhs)
            vec = pl.view(lhs, [1, 32], [0, 0])
            vacc = pl.view(acc, [1, 64], [0, 0])
            vout = pl.gemv(vec, rhs)
            vout = pl.gemv_acc(vacc, vec, rhs)
            vout = pl.gemv_bias(vec, rhs, bias)
            tmp = pl.assemble(tmp, vout, [0, 0])
            out = pl.assemble(out, tmp, [0, 0])
        return out


if __name__ == "__main__":
    output_dir = ir.compile(
        TensorOpCoverageSmoke,
        output_dir="build/tensor_op_coverage_smoke",
        dump_passes=False,
        backend_type=BackendType.PTO,
        skip_ptoas=True,
    )
    print(output_dir)
