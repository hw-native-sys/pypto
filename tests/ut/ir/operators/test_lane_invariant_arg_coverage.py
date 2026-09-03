# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep ``LaneInvariantArg`` declarations aligned with what type deduction can decide.

``LowerAutoVectorSplit`` decides whether an operand may stay full width under a halved result by
re-deducing the halved call (``HalvedCallStaysTypeConsistent``). That answer needs no metadata and
cannot go stale, so a registry declaration is only load-bearing for the operands the operator's own
``f_deduce_type`` never reads.

This module measures which operands those are, the same way the pass does: widen one tile operand on
one axis and ask the deducer again.

* deduction throws, or the deduced result changes -> the deducer **pins** that operand's extent, and
  the pass's type-consistency gate already decides it;
* deduction is unchanged -> the deducer is **blind** to it, and only a declaration can say whether
  full width is in contract.

Two things are then checked:

1. Every declared ``LaneInvariantArg.Scratch`` position names a blind operand. A declaration on a
   pinned operand is inert and actively misleading — it reads as "full width is fine here" while the
   operator's own deducer rejects that width. ``tile.rsqrt`` carried exactly such a declaration.
2. The blind positions match ``EXPECTED_BLIND_ARGS`` below. Adding an operator or loosening a
   deducer moves a position into that set, and the diff forces the question the model turns on: is
   this operand hardware scratch (declare it) or per-lane data (leave it undeclared, so the pass
   requires it to be sharded)?
"""

import itertools
import re
from pathlib import Path

import pytest
from pypto.pypto_core import DataType, ir

SPAN = ir.Span.unknown()

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Probe shapes. Rectangular on purpose: a square tile cannot tell "the deducer reads dim 0" from
# "the deducer reads dim 1". The degenerate forms let the search satisfy operators whose second
# operand is a row or column vector (the row_expand / col_expand families).
BASE_SHAPE = [64, 32]
_SHAPE_CANDIDATES = ([64, 32], [64, 1], [1, 32])

# Kwargs an operator requires before its deducer will produce any result at all. Values are chosen to
# be neutral: they must not themselves depend on a tile operand's extent, or the probe would read the
# kwarg's influence as the operand's.
_KWARGS: dict[str, dict] = {
    "tile.cast": {"target_type": DataType.INT32, "mode": 0},
    "tile.cmp": {"cmp_type": 0},
    "tile.gather_compare": {"cmp_mode": 0, "out_cols": 32},
    "tile.mrgsort_format2": {"exhausted": False},
    "tile.scatter_mask": {"mask_pattern": 1},
    "tile.tmov_x2zz": {"group_axis": 1},
    "tile.tquant_mx_raw": {"group_axis": 1},
}

_DTYPES = [
    DataType.FP32,
    DataType.FP16,
    DataType.INT32,
    DataType.UINT32,
    DataType.INT16,
    DataType.UINT16,
    DataType.INT8,
    DataType.UINT8,
]

# Operators the uniform-dtype, per-argument-shape search cannot build a baseline call for. Each is
# excluded explicitly rather than being silently reported as having no blind operands, and the set is
# asserted below so it cannot grow unnoticed.
UNSEEDED_OPS = {
    # Its third argument is a TensorType (a GM destination), and `idx` must be INT32 beside a
    # non-INT32 data tile — a dtype PAIR the uniform search does not express.
    ir.get_op("tile.mscatter").name,
    # Deduction relates the two tiles to each other (`dst.shape[1] == src.shape[1] * mask_pattern`),
    # which no assignment of independent probe shapes satisfies. That relation is itself the deducer
    # reading both operands, so both would be pinned.
    ir.get_op("tile.scatter_mask").name,
    # Raw UINT8 src/tmp plus a `dst_rows`/`dst_cols` pair that must agree with the source's static
    # valid shape.
    ir.get_op("tile.tmov_x2zz").name,
}

# (operator, argument index) pairs whose extent the operator's own type deduction never reads, so
# `HalvedCallStaysTypeConsistent` cannot decide them and the registry declaration is what the split
# pass consults. Keep sorted.
EXPECTED_BLIND_ARGS = {
    # --- declared Scratch: hardware workspace, legal at full width beside a halved input ---
    ("tile.cast", 1),
    ("tile.col_sum", 1),
    ("tile.gather", 2),
    ("tile.gather_compare", 2),
    ("tile.prelu", 2),
    ("tile.rem", 2),
    ("tile.rems", 2),
    ("tile.row_expand_add", 2),
    ("tile.row_max", 1),
    ("tile.row_min", 1),
    ("tile.row_prod", 1),
    ("tile.row_sum", 1),
    ("tile.sel", 3),
    ("tile.sels", 2),
    ("tile.sort32", 2),
    ("tile.xor", 2),
    ("tile.xors", 2),
    # --- declared IndexAddressedSource: a lane-shared table read at absolute indices ---
    ("tile.gather", 0),
    ("tile.gatherb", 0),
    # --- undeclared: per-lane data, so the pass requires it to be sharded ---
    # (tile.col_argmax / col_argmin arg 1 used to sit here; gh#2615 gave their
    # deducer the row forms' exact-shape check, so type consistency decides them
    # now and no declaration is reachable for either.)
    ("tile.col_expand", 1),
    ("tile.col_expand_add", 1),
    ("tile.col_expand_div", 1),
    ("tile.col_expand_expdif", 1),
    ("tile.col_expand_max", 1),
    ("tile.col_expand_min", 1),
    ("tile.col_expand_mul", 1),
    ("tile.col_expand_sub", 1),
    ("tile.gather_compare", 0),
    ("tile.mrgsort_format2", 0),
    ("tile.mrgsort_format2", 1),
    ("tile.mrgsort_format2", 2),
    ("tile.mrgsort_format2", 3),
    ("tile.scatter_update", 1),
    ("tile.scatter_update", 2),
    ("tile.sel", 0),
    ("tile.sels", 0),
    ("tile.sort32", 1),
}


def _registered_tile_ops():
    """Every ``tile.*`` operator, read from the registrations themselves.

    Scanning the sources rather than listing names keeps the coverage check growing with the
    codebase: a new tile operator enters the probe the moment it is registered.
    """
    names = set()
    for source in sorted((_REPO_ROOT / "src/ir/op").rglob("*.cpp")):
        for name in re.findall(r'REGISTER_OP\("(tile\.[^"]+)"\)', source.read_text()):
            names.add(name)
    return sorted(names)


def _tile(name, shape, dtype):
    return ir.Var(name, ir.TileType(shape, dtype, memory_space=ir.MemorySpace.Vec), SPAN)


def _scalar(name, dtype):
    return ir.Var(name, ir.ScalarType(dtype), SPAN)


def _deduce(op_name, args):
    return ir.create_op_call(op_name, args, _KWARGS.get(op_name, {}), SPAN).type


def _result_shape(result_type):
    try:
        return [str(d) for d in result_type.shape]
    except Exception:  # noqa: BLE001 - a non-shaped result is still a distinguishable answer
        return None


def _vector_tile_args(op_name):
    """Positional indices of this operator's Vec-constrained tile arguments."""
    spec = ir.get_op_memory_spec(op_name)
    if spec is None:
        return []
    constraints = spec.get("input_constraints") or []
    vec = str(ir.MemorySpace.Vec)
    return [i for i, allowed in enumerate(constraints) if allowed and all(str(a) == vec for a in allowed)]


def _find_baseline(op_name, n_args, tile_idx):
    """First (args, result) the deducer accepts.

    Searches one dtype across all tiles (mixed-dtype operators are rare and are listed in
    ``UNSEEDED_OPS``) crossed with a per-argument shape from ``_SHAPE_CANDIDATES``, which is what
    lets the row/col-expand families — whose second operand is a vector — reach a baseline.
    """
    ordered = sorted(tile_idx)
    for dtype in _DTYPES:
        for shapes in itertools.product(_SHAPE_CANDIDATES, repeat=len(ordered)):
            per_arg = dict(zip(ordered, shapes, strict=True))
            for scalar_dtype in _DTYPES:
                args = [
                    _tile(f"a{i}", per_arg[i], dtype) if i in per_arg else _scalar(f"s{i}", scalar_dtype)
                    for i in range(n_args)
                ]
                try:
                    return args, _deduce(op_name, args)
                except Exception:  # noqa: BLE001 - searching for any accepted combination
                    continue
    return None, None


def _deducer_pins(op_name, args, baseline, idx, axis):
    """Whether widening ``args[idx]`` on ``axis`` changes the deducer's answer."""
    shape = [int(str(d)) for d in args[idx].type.shape]
    shape[axis] *= 2
    probe = list(args)
    probe[idx] = _tile("probe", shape, args[idx].type.dtype)
    try:
        widened = _deduce(op_name, probe)
    except Exception:  # noqa: BLE001 - a refusal is the deducer reading the operand
        return True
    return _result_shape(widened) != _result_shape(baseline)


def _measure_blind_args():
    """{(op, arg_index)} the deducer never reads, plus the ops no baseline could be built for."""
    blind = set()
    unseeded = set()
    for op_name in _registered_tile_ops():
        tile_idx = _vector_tile_args(op_name)
        if len(tile_idx) < 2:
            continue
        n_args = ir.get_op_argument_count(op_name)
        args, baseline = _find_baseline(op_name, n_args, set(tile_idx))
        if args is None:
            unseeded.add(op_name)
            continue
        for idx in tile_idx:
            if idx >= len(args):
                continue
            if not any(_deducer_pins(op_name, args, baseline, idx, axis) for axis in (0, 1)):
                blind.add((op_name, idx))
    return blind, unseeded


@pytest.fixture(scope="module")
def measured():
    return _measure_blind_args()


def test_probe_covers_every_multi_tile_vector_operator(measured):
    """The inventory below is only meaningful if the probe reached the operators it claims to."""
    _, unseeded = measured
    assert unseeded == UNSEEDED_OPS, (
        "operators the deduction probe could not build a baseline for changed; a new entry means an "
        "operator dropped out of the coverage check and needs either a probe seed or a documented "
        f"reason.\n  now unseeded but expected seeded: {sorted(unseeded - UNSEEDED_OPS)}\n"
        f"  now seeded but expected unseeded: {sorted(UNSEEDED_OPS - unseeded)}"
    )


def test_blind_operand_inventory_is_unchanged(measured):
    """Every operand type deduction cannot decide is accounted for.

    A position entering this set is one the split pass can no longer decide for itself: declare it
    ``set_lane_invariant_arg(i)`` when it is hardware workspace, or leave it undeclared when it
    carries per-lane data and must be sharded.
    """
    blind, _ = measured
    assert blind == EXPECTED_BLIND_ARGS, (
        "the set of operands type deduction is blind to changed.\n"
        f"  newly blind (classify these): {sorted(blind - EXPECTED_BLIND_ARGS)}\n"
        f"  no longer blind (the deducer now decides; drop any Scratch declaration): "
        f"{sorted(EXPECTED_BLIND_ARGS - blind)}"
    )


def test_declared_scratch_names_an_operand_deduction_cannot_decide(measured):
    """A ``Scratch`` declaration on a pinned operand is inert and misleading.

    The split pass consults the declaration only where its type-consistency gate is blind, so
    declaring a scratch whose shape the deducer pins claims "full width is fine here" while the
    operator itself rejects that width. Delete such a declaration rather than leaving it to be read
    as a contract.
    """
    blind, unseeded = measured
    inert = []
    for op_name in _registered_tile_ops():
        if op_name in unseeded:
            continue
        for idx in range(ir.get_op_argument_count(op_name)):
            if ir.get_op_lane_invariant_arg(op_name, idx) != ir.LaneInvariantArg.Scratch:
                continue
            if (op_name, idx) not in blind:
                inert.append((op_name, idx))
    assert not inert, (
        "these arguments declare LaneInvariantArg.Scratch, but their operator's own f_deduce_type "
        f"pins the extent, so the declaration can never be reached: {inert}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
