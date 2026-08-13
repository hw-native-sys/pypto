# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the AtomicAddDtypeValid property verifier.

A bf16 atomic-add into global memory lowers to pto-isa ``SetAtomicAdd<bfloat16_t>``
-> ``set_atomic_bf16``, which the A2/A3 (Ascend910B) store pipe honours and the
A5 (Ascend950) one does not. The distributed paths are the same mechanism, not
parallel ones — pto-isa's comm ``TPut`` lands each chunk with
``TSTORE_IMPL<..., AtomicAdd>``, and remote_store emits a ``pto.tstore``
directly — so one predicate (``BackendHandler::SupportsBf16AtomicAdd``) governs
every atomic site: ``tile.store``, ``tensor.assemble``, ``pld.tensor.put``,
``pld.tile.put``, ``pld.tensor.remote_store`` and ``pld.tile.remote_store``.

The property is in ``GetStructuralProperties()``, so ``PassPipeline`` verifies it
at pipeline input — the error carries the user's own source span rather than a
line in a generated ``.pto``.
"""

import pypto
import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType, backend
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import ir as _ir
from pypto.pypto_core import passes


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    backend.reset_for_testing()


def _verify(prog, backend_type):
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.AtomicAddDtypeValid)
    return passes.PropertyVerifierRegistry.verify(props, prog)


def _store_program(dtype, atomic):
    """``pl.store(..., atomic=...)`` — the tile.store site."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[16, 16], dtype], out: pl.Tensor[[16, 16], dtype]):
            t = pl.load(x, [0, 0], [16, 16])
            pl.store(t, [0, 0], out, atomic=atomic)

    return Prog


def _assemble_program(dtype, atomic):
    """``pl.assemble(..., atomic=...)`` — the tensor.assemble site."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[16, 16], dtype], out: pl.Tensor[[16, 16], dtype]):
            y = pl.add(x, x)
            out = pl.assemble(out, y, [0, 0], atomic=atomic)
            return out

    return Prog


def _put_program(dtype, atomic):
    """``pld.tensor.put(..., atomic=...)`` — the remote-put site."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], dtype],
            src: pld.DistributedTensor[[16, 64], dtype],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.put(dst, peer=peer, src=src, atomic=atomic)

    return Prog


def _remote_store_program(dtype, atomic):
    """``pld.tensor.remote_store(..., atomic=...)`` — the computed-value push site."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], dtype],
            target: pld.DistributedTensor[[16, 64], dtype],
            peer: pl.Scalar[pl.INT32],
        ):
            y = pl.add(x, x)
            pld.tensor.remote_store(y, target, peer, [0, 0], atomic=atomic)

    return Prog


_BUILDERS = [
    pytest.param(_store_program, pl.AtomicType.Add, id="tile.store"),
    pytest.param(_assemble_program, pl.AtomicType.Add, id="tensor.assemble"),
    pytest.param(_put_program, pld.AtomicType.Add, id="pld.tensor.put"),
    pytest.param(_remote_store_program, pld.AtomicType.Add, id="pld.tensor.remote_store"),
]


@pytest.mark.parametrize("build,atomic", _BUILDERS)
def test_bf16_atomic_add_rejected_on_ascend950(build, atomic):
    """A5's store pipe cannot combine bf16 — every atomic site is flagged."""
    diags = _verify(build(pl.BF16, atomic), BackendType.Ascend950)
    assert len(diags) == 1
    assert diags[0].rule_name == "AtomicAddDtypeValid"
    assert "bf16" in diags[0].message
    assert "a5" in diags[0].message
    assert "Ascend910B" in diags[0].message


@pytest.mark.parametrize("build,atomic", _BUILDERS)
def test_bf16_atomic_add_accepted_on_ascend910b(build, atomic):
    """The same programs are legal on A2/A3 (pto-isa set_atomic_bf16)."""
    assert _verify(build(pl.BF16, atomic), BackendType.Ascend910B) == []


@pytest.mark.parametrize("build,atomic", _BUILDERS)
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
def test_fp32_atomic_add_accepted_everywhere(build, atomic, backend_type):
    """Only bf16 varies by backend; fp32 atomic-add is legal on both profiles."""
    assert _verify(build(pl.FP32, atomic), backend_type) == []


@pytest.mark.parametrize(
    "build,none_atomic",
    [
        pytest.param(_store_program, pl.AtomicType.None_, id="tile.store"),
        pytest.param(_assemble_program, pl.AtomicType.None_, id="tensor.assemble"),
        pytest.param(_put_program, pld.AtomicType.None_, id="pld.tensor.put"),
        pytest.param(_remote_store_program, pld.AtomicType.None_, id="pld.tensor.remote_store"),
    ],
)
def test_plain_bf16_write_accepted_on_ascend950(build, none_atomic):
    """The check is keyed on the atomic kwarg: a plain bf16 write is unaffected."""
    assert _verify(build(pl.BF16, none_atomic), BackendType.Ascend950) == []


def test_pipeline_input_verification_rejects_bf16_atomic_put():
    """End-to-end: the property is structural, so PassPipeline verifies it at
    ``pipeline_input`` — before any lowering, with the user's own span."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    prog = _put_program(pl.BF16, pld.AtomicType.Add)
    with pytest.raises(pypto.Error, match="AtomicAddDtypeValid"):
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(prog)


def test_pipeline_accepts_bf16_atomic_put_on_ascend910b():
    """Regression guard against an over-strict check: the identical program
    compiles on the profile whose store pipe honours set_atomic_bf16."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    prog = _put_program(pl.BF16, pld.AtomicType.Add)
    PassManager.get_strategy(OptimizationStrategy.Default).run_passes(prog)


# --- Tile-level (post-conversion) atomic sites ------------------------------
#
# ``pld.tile.put`` and ``pld.tile.remote_store`` are internal forms that
# ConvertTensorToTileOps synthesizes; the DSL cannot author them, so the
# builders above cannot reach them and the verifier's dispatch for the two
# would otherwise go unexercised. Build the IR directly instead, as
# test_manual_deps_on_submit_only.py does, and run the property verifier on it.


# Operator names routed through the registry getter: a typo raises at import
# rather than silently skipping a branch or parameterizing over a
# non-existent op (see .claude/rules/operator-identity-checks.md).
_TILE_PUT = _ir.get_op("pld.tile.put").name
_TILE_REMOTE_STORE = _ir.get_op("pld.tile.remote_store").name


def _dist_tensor_var(name, shape, dtype, span):
    """A DistributedTensor-typed Var, as a window-bound parameter binds."""
    shape_exprs = [_ir.ConstInt(v, DataType.INT64, span) for v in shape]
    return _ir.Var(name, _ir.DistributedTensorType(shape_exprs, dtype), span)


def _tile_var(name, shape, dtype, span):
    shape_exprs = [_ir.ConstInt(v, DataType.INT64, span) for v in shape]
    return _ir.Var(name, _ir.TileType(shape_exprs, dtype), span)


def _tile_level_program(op_name, dtype, atomic):
    """One InCore function whose body is a single tile-level atomic call.

    ``pld.tile.put(dst, peer, src, stage, *, atomic)`` keys on args[0];
    ``pld.tile.remote_store(src_tile, target, peer, offsets, *, atomic)`` keys
    on args[1]. Both destinations are the bf16 DistributedTensor here.
    """
    span = _ir.Span.unknown()
    peer = _ir.Var("peer", _ir.ScalarType(DataType.INT32), span)
    dist = _dist_tensor_var("target", [16, 64], dtype, span)
    kwargs = {"atomic": int(atomic)}

    if op_name == _TILE_PUT:
        src = _dist_tensor_var("src", [16, 64], dtype, span)
        stage = _tile_var("stage", [16, 64], dtype, span)
        call = _ir.create_op_call(op_name, [dist, peer, src, stage], kwargs, span)
        params = [dist, peer, src, stage]
    else:
        src_tile = _tile_var("src_tile", [16, 64], dtype, span)
        offsets = _ir.MakeTuple([_ir.ConstInt(0, DataType.INDEX, span)] * 2, span)
        call = _ir.create_op_call(op_name, [src_tile, dist, peer, offsets], kwargs, span)
        params = [src_tile, dist, peer]

    body = _ir.SeqStmts([_ir.EvalStmt(call, span)], span)
    func = _ir.Function(
        "kernel",
        [(p, _ir.ParamDirection.In) for p in params],
        [],
        body,
        span,
        _ir.FunctionType.InCore,
    )
    return _ir.Program([func], "kernel", span)


_TILE_LEVEL_OPS = [_TILE_PUT, _TILE_REMOTE_STORE]


@pytest.mark.parametrize("op_name", _TILE_LEVEL_OPS)
def test_tile_level_bf16_atomic_add_rejected_on_ascend950(op_name):
    """The post-conversion forms are dispatched too, not just their tensor twins."""
    diags = _verify(_tile_level_program(op_name, DataType.BF16, 1), BackendType.Ascend950)
    assert len(diags) == 1
    assert diags[0].rule_name == "AtomicAddDtypeValid"
    assert "bf16" in diags[0].message


@pytest.mark.parametrize("op_name", _TILE_LEVEL_OPS)
def test_tile_level_bf16_atomic_add_accepted_on_ascend910b(op_name):
    assert _verify(_tile_level_program(op_name, DataType.BF16, 1), BackendType.Ascend910B) == []


@pytest.mark.parametrize("op_name", _TILE_LEVEL_OPS)
def test_tile_level_plain_bf16_write_accepted_on_ascend950(op_name):
    """Keyed on the atomic kwarg here as well."""
    assert _verify(_tile_level_program(op_name, DataType.BF16, 0), BackendType.Ascend950) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
