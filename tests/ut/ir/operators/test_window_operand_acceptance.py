# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every ``tensor.*`` operator states whether it accepts a window operand.

Inside an InCore / chip-orchestration scope a ``pld.DistributedTensor`` window slice *is* this
rank's local GM, so an ordinary tensor op should read and write it like any GM tensor. Whether it
can is decided by one line in its type deducer: ``AsTensorTypeLike`` matches a window,
the exact-kind ``As<TensorType>`` does not (see ``.claude/rules/ir-kind-traits.md``).

That line is easy to get wrong, and wrong in a way nothing else notices — the op simply refuses a
legal program, or silently skips the operand and fails later in a pass or emitter. This file makes
the decision explicit and enforced:

* Every registered ``tensor.*`` operator is classified into exactly one bucket below. Adding an
  operator without classifying it fails ``test_every_registered_tensor_op_is_classified``.
* Every classified operator with a GM operand is probed with a window in that slot, and its bucket
  is checked against what the deducer actually does.

``WINDOW_GAP`` is the honest part: those operators *should* accept a window but do not yet. The
test asserts they still reject, so fixing one fails here and prompts moving it to ``WINDOW_OK``.
A fix is not complete at the deducer, though — see the note on that constant.
"""

import re
import subprocess
from functools import partial
from pathlib import Path

import pytest
from pypto import DataType as DT
from pypto import ir

SPAN = ir.Span.unknown()
I32 = DT.INT32
t = ir.op.tensor


def _dims(shape):
    return [ir.ConstInt(d, I32, SPAN) for d in shape]


def W(name="win", shape=(16, 32), dtype=DT.FP32):
    """A window operand: a Var whose type is DistributedTensorType."""
    return ir.Var(name, ir.DistributedTensorType(_dims(list(shape)), dtype), SPAN)


def T(name="t", shape=(16, 32), dtype=DT.FP32):
    return ir.Var(name, ir.TensorType(_dims(list(shape)), dtype), SPAN)


def S(name="s", dtype=DT.FP32):
    return ir.Var(name, ir.ScalarType(dtype), SPAN)


# --------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------


def _ops(*names: str) -> set[str]:
    """Validate each operator-name literal through the registry getter.

    ``ir.get_op`` raises on an unregistered name, so a typo in the classification below fails at
    import instead of silently creating a bucket entry that matches nothing
    (``.claude/rules/operator-identity-checks.md``).
    """
    return {ir.get_op(n).name for n in names}


#: Operators with no GM tensor operand at all -- there is no slot a window could occupy.
NO_GM_OPERAND = _ops(
    "tensor.alloc",
    "tensor.ci",
    "tensor.create",
    "tensor.create_l1",
    "tensor.full",
    "tensor.get_block_idx",
    "tensor.get_block_num",
    "tensor.get_subblock_idx",
    "tensor.random",
)

#: Operators that reject a window on purpose. Each entry names the reason.
WINDOW_REJECT_BY_DESIGN = {
    # A window's storage is a comm-window slice; reinterpreting its dtype has no lowering
    # contract yet, and the deducer says so in its own message.
    ir.get_op(
        "tensor.reinterpret_view"
    ).name: "no zero-copy dtype reinterpretation contract for window storage",
    # AIV/AIC split boundary ops are cube/vector-lane transfers, not GM reads; a window operand
    # is out of scope and rejected upstream (docs/en/dev/passes/11-convert_tensor_to_tile_ops.md).
    ir.get_op("tensor.aiv_shard").name: "cross-lane split boundary, not a GM read",
    ir.get_op("tensor.aic_gather").name: "cross-lane split boundary, not a GM read",
}

#: Operators whose deducer accepts a window today. Keep this list growing, never shrinking.
WINDOW_OK = _ops(
    # element-wise / unary / scalar-rhs (relaxed by issue #1694)
    "tensor.abs",
    "tensor.add",
    "tensor.adds",
    "tensor.and",
    "tensor.ands",
    "tensor.cast",
    "tensor.cmp",
    "tensor.cos",
    "tensor.div",
    "tensor.divs",
    "tensor.exp",
    "tensor.fmod",
    "tensor.fmods",
    "tensor.log",
    "tensor.maximum",
    "tensor.minimum",
    "tensor.mul",
    "tensor.muls",
    "tensor.neg",
    "tensor.not",
    "tensor.or",
    "tensor.ors",
    "tensor.part_add",
    "tensor.part_max",
    "tensor.part_min",
    "tensor.part_mul",
    "tensor.recip",
    "tensor.rsqrt",
    "tensor.shl",
    "tensor.shls",
    "tensor.shr",
    "tensor.shrs",
    "tensor.sin",
    "tensor.sqrt",
    "tensor.sub",
    "tensor.subs",
    "tensor.xor",
    "tensor.xors",
    # views and scalar access -- these propagate the window kind onto their result
    "tensor.assemble",
    "tensor.read",
    "tensor.slice",
    "tensor.view",
    "tensor.write",
    # reductions
    "tensor.col_argmax",
    "tensor.col_argmin",
    "tensor.col_max",
    "tensor.col_min",
    "tensor.col_prod",
    "tensor.col_sum",
    "tensor.row_argmax",
    "tensor.row_argmin",
    "tensor.row_max",
    "tensor.row_min",
    "tensor.row_prod",
    "tensor.row_sum",
    # cube
    "tensor.matmul",
    "tensor.matmul_acc",
)

#: Operators that *should* accept a window (they read or write plain GM) but do not yet.
#:
#: Relaxing the deducer is only the first of two gates: the operand must also survive its
#: lowering site. ``tensor.fillpad``, ``tensor.fillpad_expand``, ``tensor.expand_clone``,
#: ``tensor.gather`` and ``tensor.paged_gather`` carry the same exact-kind cast inside their
#: rules in ``src/ir/transforms/op_conversion_registry.cpp``, so relaxing the deducer alone
#: turns a clear ``CHECK`` into an ``INTERNAL_UNREACHABLE`` deeper in the pipeline. Fix both,
#: add a lowering test, then move the name into ``WINDOW_OK``.
WINDOW_GAP = _ops(
    "tensor.col_expand",
    "tensor.col_expand_add",
    "tensor.col_expand_div",
    "tensor.col_expand_expdif",
    "tensor.col_expand_max",
    "tensor.col_expand_min",
    "tensor.col_expand_mul",
    "tensor.col_expand_sub",
    "tensor.concat",
    "tensor.dim",
    "tensor.expand_clone",
    "tensor.expands",
    "tensor.fillpad",
    "tensor.fillpad_expand",
    "tensor.gather",
    "tensor.gather_compare",
    "tensor.gather_mask",
    "tensor.gather_row",
    "tensor.mrgsort_format1",
    "tensor.mrgsort_format2",
    "tensor.paged_gather",
    "tensor.reshape",
    "tensor.row_expand",
    "tensor.row_expand_add",
    "tensor.row_expand_div",
    "tensor.row_expand_expdif",
    "tensor.row_expand_max",
    "tensor.row_expand_min",
    "tensor.row_expand_mul",
    "tensor.row_expand_sub",
    "tensor.scatter",
    "tensor.scatter_mask",
    "tensor.scatter_update",
    "tensor.set_validshape",
    "tensor.sort32",
    "tensor.transpose",
)


# --------------------------------------------------------------------------------------------
# Probes: build each op's call with a window in its GM slot
# --------------------------------------------------------------------------------------------

#: Float-dtype families. The bitwise / shift ops below need an integer operand instead.
_UNARY = ["abs", "cos", "exp", "log", "neg", "recip", "rsqrt", "sin", "sqrt"]
_BINARY = [
    "add",
    "cmp",
    "div",
    "fmod",
    "maximum",
    "minimum",
    "mul",
    "part_add",
    "part_max",
    "part_min",
    "part_mul",
    "sub",
]
_SCALAR_RHS = ["adds", "divs", "expands", "fmods", "muls", "subs"]
_INT_UNARY: list[str] = []  # tensor.not needs a 16-bit int; probed explicitly below
_INT_BINARY = ["and_", "or_", "shl", "shr", "xor"]
_INT_SCALAR_RHS = ["ands", "ors", "shls", "shrs", "xors"]
_ROW_REDUCE = ["row_argmax", "row_argmin", "row_max", "row_min", "row_prod", "row_sum"]
_COL_REDUCE = ["col_argmax", "col_argmin", "col_max", "col_min", "col_prod", "col_sum"]
_ROW_EXPAND = [
    "row_expand_add",
    "row_expand_div",
    "row_expand_expdif",
    "row_expand_max",
    "row_expand_min",
    "row_expand_mul",
    "row_expand_sub",
]
_COL_EXPAND = [
    "col_expand_add",
    "col_expand_div",
    "col_expand_expdif",
    "col_expand_max",
    "col_expand_min",
    "col_expand_mul",
    "col_expand_sub",
]

#: op name -> zero-arg callable putting a window in the operand slot under test.
PROBES: dict[str, object] = {}


def _probe(fn, *arg_makers):
    """Bind an op builder to zero-arg operand factories, evaluated when the probe runs."""
    return lambda: fn(*(make() for make in arg_makers))


_INT_W = partial(W, dtype=I32)
_INT_T = partial(T, dtype=I32)
_INT_S = partial(S, dtype=I32)
_ROW_CARRIER = partial(T, shape=(16, 1))
_COL_CARRIER = partial(T, shape=(1, 32))

for _n in _UNARY:
    PROBES[ir.get_op(f"tensor.{_n}").name] = _probe(getattr(t, _n), W)
for _n in _BINARY:
    PROBES[ir.get_op(f"tensor.{_n}").name] = _probe(getattr(t, _n), W, T)
for _n in _SCALAR_RHS:
    PROBES[ir.get_op(f"tensor.{_n}").name] = _probe(getattr(t, _n), W, S)
for _n in _INT_BINARY:
    PROBES[ir.get_op(f"tensor.{_n.rstrip('_')}").name] = _probe(getattr(t, _n), _INT_W, _INT_T)
for _n in _INT_SCALAR_RHS:
    PROBES[ir.get_op(f"tensor.{_n}").name] = _probe(getattr(t, _n), _INT_W, _INT_S)
for _n in _ROW_REDUCE + _COL_REDUCE:
    PROBES[ir.get_op(f"tensor.{_n}").name] = _probe(getattr(t, _n), W)
for _n in _ROW_EXPAND:
    PROBES[ir.get_op(f"tensor.{_n}").name] = _probe(getattr(t, _n), W, _ROW_CARRIER)
for _n in _COL_EXPAND:
    PROBES[ir.get_op(f"tensor.{_n}").name] = _probe(getattr(t, _n), W, _COL_CARRIER)

PROBES.update(
    {
        ir.get_op("tensor.cast").name: lambda: t.cast(W(), DT.FP16),
        # tensor.not accepts only 16-bit integer operands.
        ir.get_op("tensor.not").name: lambda: t.not_(W(dtype=DT.INT16)),
        ir.get_op("tensor.matmul").name: lambda: t.matmul(W(dtype=DT.BF16), T(shape=(32, 16), dtype=DT.BF16)),
        ir.get_op("tensor.matmul_acc").name: lambda: t.matmul_acc(
            T(shape=(16, 16)), W(dtype=DT.BF16), T(shape=(32, 16), dtype=DT.BF16)
        ),
        ir.get_op("tensor.slice").name: lambda: t.slice(W(), [8, 32], [0, 0]),
        ir.get_op("tensor.assemble").name: lambda: t.assemble(W(), T(shape=(8, 32)), [0, 0]),
        ir.get_op("tensor.view").name: lambda: t.view(W(), [16, 32]),
        ir.get_op("tensor.read").name: lambda: t.read(
            W(), [ir.ConstInt(0, I32, SPAN), ir.ConstInt(0, I32, SPAN)]
        ),
        ir.get_op("tensor.write").name: lambda: t.write(
            W(), [ir.ConstInt(0, I32, SPAN), ir.ConstInt(0, I32, SPAN)], S()
        ),
        ir.get_op("tensor.reshape").name: lambda: t.reshape(W(), [512]),
        ir.get_op("tensor.transpose").name: lambda: t.transpose(W(), 0, 1),
        ir.get_op("tensor.set_validshape").name: lambda: t.set_validshape(W(), 8, 32),
        ir.get_op("tensor.concat").name: lambda: t.concat(W(), T()),
        ir.get_op("tensor.reinterpret_view").name: lambda: t.reinterpret_view(W(), dtype=DT.FP16),
        ir.get_op("tensor.fillpad").name: lambda: t.fillpad(W()),
        ir.get_op("tensor.fillpad_expand").name: lambda: t.fillpad_expand(W(), [16, 64]),
        ir.get_op("tensor.dim").name: lambda: t.dim(W(), 0),
        ir.get_op("tensor.expand_clone").name: lambda: t.expand_clone(W(), T(shape=(1, 32))),
        ir.get_op("tensor.row_expand").name: lambda: t.row_expand(T(), W(shape=(16, 1))),
        ir.get_op("tensor.col_expand").name: lambda: t.col_expand(T(), W(shape=(1, 32))),
        ir.get_op("tensor.gather").name: lambda: t.gather(W(), 1, T(dtype=I32)),
        ir.get_op("tensor.gather_mask").name: lambda: t.gather_mask(W(), 1),
        ir.get_op("tensor.gather_row").name: lambda: t.gather_row(
            T(shape=(16, 32)), W(shape=(64, 32)), [0, 0], [0, 0], [16, 32]
        ),
        ir.get_op("tensor.gather_compare").name: lambda: t.gather_compare(
            W(), S(), cmp_mode="lt", out_cols=32
        ),
        ir.get_op("tensor.scatter").name: lambda: t.scatter(W(), 1, T(dtype=I32), T()),
        ir.get_op("tensor.scatter_mask").name: lambda: t.scatter_mask(W(), T(), 1),
        ir.get_op("tensor.scatter_update").name: lambda: t.scatter_update(
            W(), 0, T(shape=(16, 1), dtype=I32), T()
        ),
        ir.get_op("tensor.sort32").name: lambda: t.sort32(W(), T(dtype=I32)),
        ir.get_op("tensor.paged_gather").name: lambda: t.paged_gather(
            W(shape=(64, 32), dtype=DT.FP16),
            T(shape=(16, 1), dtype=I32),
            T(shape=(1, 16), dtype=I32),
            16,
            16,
            16,
        ),
        ir.get_op("tensor.mrgsort_format1").name: lambda: t.mrgsort_format1(W(), 16),
        ir.get_op("tensor.mrgsort_format2").name: lambda: t.mrgsort_format2(W(), T()),
    }
)

#: Classified operators with no ``ir.op.tensor`` entry point to probe through. Keep this set
#: minimal and justified: an operator listed here is classified but not behaviourally checked.
NO_PYTHON_ENTRY = _ops(
    # Reachable only through the pl.split_aiv region, which supplies the split mode; there is no
    # standalone tensor-level builder to call with a window.
    "tensor.aiv_shard",
    "tensor.aic_gather",
)

_REJECT_MESSAGE = re.compile(r"DistributedTensorType")


def _rejects_window(name: str) -> tuple[bool, str]:
    """Run the probe. Returns (rejected, detail). Raises if the probe itself is malformed."""
    probe = PROBES.get(name)
    assert probe is not None, (
        f"{name} is classified but has no probe recipe in PROBES -- add one so its window "
        f"behaviour is actually checked"
    )
    try:
        call = probe()
    except (ValueError, TypeError) as exc:
        first = str(exc).splitlines()[0]
        assert _REJECT_MESSAGE.search(first), (
            f"{name}: probe recipe is malformed -- it failed for a reason unrelated to the "
            f"window operand: {first}"
        )
        return True, first
    return False, type(call.type).__name__


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------


def _registered_tensor_ops() -> set[str]:
    """Every ``tensor.*`` name registered in C++, read from the sources.

    The registry exposes no enumeration binding, so the names are read from the
    ``REGISTER_OP("tensor.…")`` sites instead. That is the same list the runtime builds, and it
    means a newly added operator is picked up here the moment it is registered.
    """
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    pattern = re.compile(r'REGISTER_OP\("(tensor\.[a-z0-9_]+)"\)')
    names: set[str] = set()
    for path in (root / "src" / "ir" / "op").rglob("*.cpp"):
        names.update(pattern.findall(path.read_text(encoding="utf-8")))
    assert names, 'found no REGISTER_OP("tensor.…") sites -- the scan path is wrong'
    return names


def test_every_registered_tensor_op_is_classified():
    """A new ``tensor.*`` operator must declare whether it accepts a window operand."""
    classified = NO_GM_OPERAND | set(WINDOW_REJECT_BY_DESIGN) | WINDOW_OK | WINDOW_GAP
    registered = _registered_tensor_ops()

    unclassified = sorted(registered - classified)
    assert not unclassified, (
        "these tensor operators are registered but not classified for window operands: "
        f"{unclassified}. Decide whether each accepts a pld.DistributedTensor in its GM slot "
        "(AsTensorTypeLike) or not (As<TensorType>), then add it to WINDOW_OK, WINDOW_GAP, "
        "WINDOW_REJECT_BY_DESIGN or NO_GM_OPERAND in this file."
    )

    stale = sorted(classified - registered)
    assert not stale, f"these classified operators are no longer registered: {stale}"


def test_buckets_are_disjoint():
    buckets = {
        "NO_GM_OPERAND": NO_GM_OPERAND,
        "WINDOW_REJECT_BY_DESIGN": set(WINDOW_REJECT_BY_DESIGN),
        "WINDOW_OK": WINDOW_OK,
        "WINDOW_GAP": WINDOW_GAP,
    }
    names = list(buckets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sorted(buckets[a] & buckets[b])
            assert not overlap, f"{a} and {b} both claim {overlap}"


@pytest.mark.parametrize("op_name", sorted(WINDOW_OK))
def test_window_ok_ops_accept_a_window_operand(op_name):
    """These deducers must keep accepting a window: it is this rank's local GM."""
    if op_name in NO_PYTHON_ENTRY:
        pytest.skip(f"{op_name} has no ir.op.tensor entry point to probe")
    rejected, detail = _rejects_window(op_name)
    assert not rejected, (
        f"{op_name} regressed: it now rejects a window operand ({detail}). Its deducer must use "
        f"AsTensorTypeLike, not the exact-kind As<TensorType> -- see .claude/rules/ir-kind-traits.md"
    )


@pytest.mark.parametrize("op_name", sorted(WINDOW_GAP))
def test_window_gap_ops_still_reject(op_name):
    """Inventory of known gaps -- shrink this list, never grow it.

    If this fails because the operator now *accepts* a window, that is the intended fix: verify
    its lowering path too (the conversion rule in ``op_conversion_registry.cpp`` and the entry
    load in ``ConvertTensorToTileOps``), add a lowering test, and move the name to WINDOW_OK.
    """
    if op_name in NO_PYTHON_ENTRY:
        pytest.skip(f"{op_name} has no ir.op.tensor entry point to probe")
    rejected, detail = _rejects_window(op_name)
    assert rejected, (
        f"{op_name} now accepts a window operand (result: {detail}). Move it from WINDOW_GAP to "
        "WINDOW_OK, and make sure its lowering path accepts one too."
    )


@pytest.mark.parametrize("op_name", sorted(WINDOW_REJECT_BY_DESIGN))
def test_by_design_rejections_stay_rejections(op_name):
    if op_name in NO_PYTHON_ENTRY:
        pytest.skip(f"{op_name} has no ir.op.tensor entry point to probe")
    rejected, detail = _rejects_window(op_name)
    assert rejected, (
        f"{op_name} is documented as rejecting a window "
        f"({WINDOW_REJECT_BY_DESIGN[op_name]}) but accepted one (result: {detail})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
