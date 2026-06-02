# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression test: pipelined 3D batch matmul must not emit Mat->Mat ``pto.tmov``.

A 3D batch matmul (operands sliced from a tensor with a leading dim of 1)
written inside a ``pl.pipeline`` K-reduction loop with a ``create_tensor``
accumulator and an ``if kb == 0`` first-iteration branch.

``FlattenTileNdTo2D`` lowers the rank-3 ``tile.batch_matmul`` by unrolling the
batch dimension; with a 1-sized batch the per-page extraction emits a no-op
full-shape, offset-0 ``tile.slice`` whose result is ``Mem.Mat``. Without
canonicalization that dead slice lowers to an unsupported
``loc=mat -> loc=mat`` ``pto.tmov``. The ``CanonicalizeTileSlice`` pass folds
every Mat-resident ``tile.slice`` into its ``tile.extract`` consumer, so the
generated ``.pto`` contains zero Mat->Mat ``pto.tmov``.
"""

import re
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import pypto.language as pl  # noqa: E402
from pypto.runtime import RunConfig  # noqa: E402

# K reduced in chunks of KC; the leading dim is the batch dimension.
T, K, N, KC = 32, 4096, 128, 512

# Deterministic dump location so the per-pass IR / .pto is easy to inspect.
DUMP_DIR = Path(__file__).resolve().parents[4] / "build_output" / "batch_matmul_pipeline_repro"

# A pto.tmov whose input and output tile_buf are both loc=mat (the illegal
# Mat->Mat move this test guards against). The op spans two lines in the .pto.
_MAT_TO_MAT_TMOV = re.compile(r"pto\.tmov\s+ins\([^)]*loc=mat[^)]*\)\s*outs\([^)]*loc=mat[^)]*\)", re.DOTALL)


@pl.jit
def batch_matmul_pipeline_repro(
    x: pl.Tensor[[1, T, K], pl.INT8],
    w: pl.Tensor[[1, N, K], pl.INT8],
    out: pl.Out[pl.Tensor[[1, T, N], pl.INT32]],
):
    """3D batch matmul, leading dim == 1, in a pipelined K-loop."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="repro_mm"):
        acc = pl.create_tensor([1, T, N], dtype=pl.INT32)  # -> pl.Mem.Acc
        for kb in pl.pipeline(0, K // KC, stage=2):
            k0 = kb * KC
            x_k = x[:, :, k0 : k0 + KC]  # [1, T, KC]  3D batch slice
            w_k = w[:, :, k0 : k0 + KC]  # [1, N, KC]
            if kb == 0:
                acc = pl.matmul(x_k, w_k, b_trans=True, out_dtype=pl.INT32)
            else:
                acc = pl.matmul_acc(acc, x_k, w_k, b_trans=True)
    out[:, :, :] = acc
    return out


def test_no_mat_to_mat_tmov():
    """The 3D-batch pipelined matmul must not lower a Mat ``tile.slice`` to a
    ``loc=mat -> loc=mat`` ``pto.tmov``."""
    batch_matmul_pipeline_repro._cache.clear()
    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)

    x = torch.zeros((1, T, K), dtype=torch.int8)
    w = torch.zeros((1, N, K), dtype=torch.int8)
    out = torch.zeros((1, T, N), dtype=torch.int32)

    cfg = RunConfig(
        platform="a2a3",
        codegen_only=True,
        dump_passes=True,
        save_kernels=True,
        save_kernels_dir=str(DUMP_DIR),
    )
    # End-to-end compilation may still fail on an unrelated downstream codegen
    # bug (transposed-weight DN->NZ TLOAD — tracked separately); the Mat->Mat
    # ``pto.tmov`` check below is on the generated .pto, which codegen emits
    # before that.
    compile_error: Exception | None = None
    try:
        batch_matmul_pipeline_repro(x, w, out, config=cfg)
    except Exception as e:  # noqa: BLE001 - see comment above
        compile_error = e

    # The generated .pto lands in different subdirs depending on whether ptoas
    # is available: ``ptoas/<name>.pto`` (the ptoas input) when it runs, or the
    # raw MLIR ``kernels/<core>/<name>.pto`` under ``--codegen-only`` / no ptoas.
    # Both carry the same pre-ptoas ``pto.tmov`` ops, so scan every emitted .pto.
    ptos = sorted(DUMP_DIR.rglob("*.pto"))
    assert ptos, (
        f"codegen did not emit any .pto under {DUMP_DIR}; "
        f"compile raised before .pto materialized: {compile_error!r}"
    )
    mat_to_mat = [m for p in ptos for m in _MAT_TO_MAT_TMOV.findall(p.read_text())]
    assert not mat_to_mat, (
        f"{len(mat_to_mat)} unsupported loc=mat -> loc=mat pto.tmov in {[str(p) for p in ptos]}; "
        f"the Mat tile.slice was not canonicalized into tile.extract"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
