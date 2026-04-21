# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Emit executable PyTorch code from PyPTO IR for debugging and numerical verification."""

import keyword
import re
from collections.abc import Callable
from typing import Any

from pypto import DataType
from pypto import ir as _ir

# ---------------------------------------------------------------------------
# DataType -> torch dtype string
# ---------------------------------------------------------------------------
_DTYPE_MAP: dict[str, str] = {
    "fp16": "torch.float16",
    "fp32": "torch.float32",
    "fp64": "torch.float64",
    "bfloat16": "torch.bfloat16",
    "int8": "torch.int8",
    "int16": "torch.int16",
    "int32": "torch.int32",
    "int64": "torch.int64",
    "uint8": "torch.uint8",
    "uint16": "torch.int32",  # torch has no uint16; upcast
    "uint32": "torch.int64",  # torch has no uint32; upcast
    "uint64": "torch.int64",  # torch has no uint64; best-effort
    "bool": "torch.bool",
    "index": "torch.int64",
}


def _torch_dtype(dt: DataType) -> str:
    return _DTYPE_MAP.get(str(dt), "torch.float32")


# ---------------------------------------------------------------------------
# Comparison type int -> Python operator
# ---------------------------------------------------------------------------
_CMP_OPS: dict[int, str] = {
    0: "==",  # EQ
    1: "!=",  # NE
    2: "<",  # LT
    3: "<=",  # LE
    4: ">",  # GT
    5: ">=",  # GE
}


def _sanitize_name_hint(hint: str) -> str:
    """Convert an IR name hint into a valid Python identifier base."""
    base = hint or "v"
    base = re.sub(r"[^a-zA-Z0-9_]", "_", base)
    base = re.sub(r"__+", "_", base).strip("_") or "v"
    if base[0].isdigit():
        base = f"v_{base}"
    if keyword.iskeyword(base):
        base = f"{base}_v"
    return base


def _make_unique_names(hints: list[str]) -> list[str]:
    """Make sanitized Python identifiers unique while preserving order."""
    unique_names: list[str] = []
    counts: dict[str, int] = {}
    for hint in hints:
        base = _sanitize_name_hint(hint)
        count = counts.get(base, 0)
        if count == 0:
            unique_names.append(base)
            counts[base] = 1
        else:
            unique_names.append(f"{base}_{count}")
            counts[base] = count + 1
    return unique_names


# ---------------------------------------------------------------------------
# Preamble inserted at top of every generated script
# ---------------------------------------------------------------------------
_PREAMBLE = """\
import torch
from collections import deque

_pipes = {'to_aiv': deque(), 'to_aic': deque()}

def _coerce_shape(shape):
    return tuple(int(s) for s in shape)

def _pad_scalar(tensor, pad_mode):
    if pad_mode == "zero":
        return 0
    if tensor.dtype.is_floating_point:
        finfo = torch.finfo(tensor.dtype)
        return finfo.min if pad_mode == "min" else finfo.max
    if tensor.dtype == torch.bool:
        return False if pad_mode == "min" else True
    iinfo = torch.iinfo(tensor.dtype)
    return iinfo.min if pad_mode == "min" else iinfo.max

def _mask_valid_region(tensor, shapes, valid_shapes):
    shapes_t = _coerce_shape(shapes)
    valid_t = _coerce_shape(valid_shapes) if valid_shapes is not None else None
    if valid_t is not None:
        if valid_t != shapes_t:
            masked = tensor.new_zeros(shapes_t)
            valid_slices = tuple(slice(0, s) for s in valid_t)
            masked[valid_slices] = tensor[valid_slices]
            tensor = masked
        tensor._pypto_valid_shape = valid_t
        tensor._pypto_full_shape = shapes_t
    return tensor

def _tile_load(tensor, offsets, shapes, valid_shapes=None):
    offsets_t = _coerce_shape(offsets)
    shapes_t = _coerce_shape(shapes)
    slices = tuple(slice(o, o + s) for o, s in zip(offsets_t, shapes_t))
    tile = tensor[slices].clone()
    actual_shape = tuple(tile.shape)
    # Pad to requested shape if source is smaller (boundary case)
    if actual_shape != shapes_t:
        padded = tile.new_zeros(shapes_t)
        pad_slices = tuple(slice(0, s) for s in actual_shape)
        padded[pad_slices] = tile
        tile = padded
    # Use provided valid_shapes or fall back to the physical boundary; cap by actual data bounds.
    v_shape = _coerce_shape(valid_shapes) if valid_shapes is not None else actual_shape
    v_shape = tuple(min(v, a) for v, a in zip(v_shape, actual_shape))
    return _mask_valid_region(tile, shapes_t, v_shape)

def _tile_store(tile, offsets, output_tensor):
    offsets_t = _coerce_shape(offsets)
    valid_shape = getattr(tile, "_pypto_valid_shape", tuple(tile.shape))
    slices = tuple(slice(o, o + s) for o, s in zip(offsets_t, valid_shape))
    valid_slices = tuple(slice(0, s) for s in valid_shape)
    output_tensor[slices] = tile[valid_slices]
    return output_tensor

def _tensor_slice(tensor, offsets, shapes, valid_shapes=None):
    offsets_t = _coerce_shape(offsets)
    shapes_t = _coerce_shape(shapes)
    slices = tuple(slice(o, o + s) for o, s in zip(offsets_t, shapes_t))
    sliced = tensor[slices]
    # Out-of-bounds tensor.slice in kernels should still materialize the
    # requested shape for downstream matmul/fillpad paths.
    if tuple(sliced.shape) != shapes_t:
        padded = sliced.new_zeros(shapes_t)
        pad_slices = tuple(slice(0, s) for s in sliced.shape)
        padded[pad_slices] = sliced
        sliced = padded
    if valid_shapes is not None:
        sliced._pypto_valid_shape = _coerce_shape(valid_shapes)
        sliced._pypto_full_shape = shapes_t
    return sliced

def _fillpad(tensor, pad_mode="zero"):
    valid_shape = getattr(tensor, "_pypto_valid_shape", None)
    full_shape = getattr(tensor, "_pypto_full_shape", tuple(tensor.shape))
    full_shape = _coerce_shape(full_shape)
    if tuple(tensor.shape) != full_shape:
        padded = tensor.new_zeros(full_shape)
        pad_slices = tuple(slice(0, s) for s in tensor.shape)
        padded[pad_slices] = tensor
        tensor = padded
    if valid_shape is None:
        return tensor
    valid_shape = _coerce_shape(valid_shape)
    if valid_shape == full_shape:
        return tensor
    fill_value = _pad_scalar(tensor, pad_mode)
    padded = tensor.new_full(full_shape, fill_value)
    valid_slices = tuple(slice(0, s) for s in valid_shape)
    padded[valid_slices] = tensor[valid_slices]
    return padded

def _write_and_return(container, index, value):
    container[index] = value
    return container

def _assemble(target, source, offsets):
    offsets_t = _coerce_shape(offsets)
    valid_shape = getattr(source, "_pypto_valid_shape", tuple(source.shape))
    slices = tuple(slice(o, o + s) for o, s in zip(offsets_t, valid_shape))
    valid_slices = tuple(slice(0, s) for s in valid_shape)
    target[slices] = source[valid_slices]
    return target

def _split_for_aiv_consumer(tensor, split_mode):
    if split_mode == 0:
        return [tensor.clone()]
    elif split_mode == 1:  # UpDown: split along rows (dim -2)
        mid = tensor.shape[-2] // 2
        return [tensor[..., :mid, :].clone(), tensor[..., mid:, :].clone()]
    elif split_mode == 2:  # LeftRight: split along cols (dim -1)
        mid = tensor.shape[-1] // 2
        return [tensor[..., :mid].clone(), tensor[..., mid:].clone()]
    return [tensor.clone()]

def _tpush_to_aiv(pipe, tensor, split_mode):
    for chunk in _split_for_aiv_consumer(tensor, split_mode):
        pipe.append(chunk)
    return tensor

def _tpush_to_aic(pipe, tensor, _split_mode):
    # Outside the cooperative scheduler we do not model two AIV subblocks
    # running in parallel, so the C2V push always carries the full tile and
    # the split kwarg is informational.  Scheduler-mode emission uses the
    # ``_g`` variants which do honor split.
    pipe.append(tensor.clone())
    return tensor

def _tpop_from_aic(pipe, split_mode):
    if split_mode == 0:
        return pipe.popleft()
    # split>0: ``_tpush_to_aiv`` queued two halves on the same deque; pop
    # both and reassemble the full tile so legacy single-AIV-subblock code
    # paths see the same end-to-end value as the scheduled path.
    first = pipe.popleft()
    second = pipe.popleft()
    if split_mode == 1:  # UpDown -> rows
        return torch.cat([first, second], dim=-2)
    if split_mode == 2:  # LeftRight -> cols
        return torch.cat([first, second], dim=-1)
    return first

def _tpop_from_aiv(pipe, _split_mode):
    return pipe.popleft()

# Per-subblock pipes (only used when split>0).  In real hardware a split tile
# is delivered to two AIV subblocks (sb0 / sb1) which run the same kernel on
# their own half; their outputs are reassembled at the AIC pop side.  These
# extra deques carry the per-subblock chunks so each AIV invocation in the
# scheduler picks up its own half, and AIC's tpop_from_aiv can ``cat`` the two
# halves back together.
_pipes_sb = {
    'to_aiv_sb0': deque(), 'to_aiv_sb1': deque(),
    'to_aic_sb0': deque(), 'to_aic_sb1': deque(),
}

# Current subblock id (0 or 1).  ``_run_scheduler`` writes this slot before
# resuming each task so the generator-mode pipe helpers and
# tile.get_subblock_idx codegen can read it via runtime context.
_current_sb = [0]

def _reset_pipes():
    _pipes['to_aiv'].clear()
    _pipes['to_aic'].clear()
    for q in _pipes_sb.values():
        q.clear()
    _current_sb[0] = 0

# --- Cooperative scheduler for cross-core simulation -----------------------
# Used when a Group function couples AIC and AIV functions whose tpush/tpop
# operations cannot be modeled by simply running one side and then the other
# (e.g. bidirectional V<->C, producer/consumer feedback loops, or any pipe
# op carrying split>0 which needs two AIV subblocks running in parallel).
# Each AIC or AIV function is emitted as a Python generator that ``yield``s a
# ``_WaitPop`` / ``_WaitPush`` request at every pipe synchronization point.
# ``_run_scheduler`` advances each generator until it blocks, then switches
# to the next, mirroring the cooperative interleaving of the real cores.

class _WaitPop:
    __slots__ = ("pipe",)
    def __init__(self, pipe):
        self.pipe = pipe

class _WaitPush:
    __slots__ = ("pipe", "item")
    def __init__(self, pipe, item):
        self.pipe = pipe
        self.item = item

def _tpush_to_aiv_g(pipe, tensor, split_mode):
    # split_mode == 0: single full-tile chunk on the unified to_aiv pipe.
    # split_mode > 0:  split the tile and route halves to per-subblock pipes
    #                  so each AIV subblock picks up its own portion.
    if split_mode == 0:
        yield _WaitPush(pipe, tensor.clone())
        return tensor
    chunks = _split_for_aiv_consumer(tensor, split_mode)
    yield _WaitPush(_pipes_sb['to_aiv_sb0'], chunks[0])
    yield _WaitPush(_pipes_sb['to_aiv_sb1'], chunks[1])
    return tensor

def _tpush_to_aic_g(pipe, tensor, split_mode):
    # split_mode == 0: single chunk on the unified to_aic pipe.
    # split_mode > 0:  push each AIV subblock's contribution onto its own pipe;
    #                  AIC's tpop_from_aiv reassembles them.
    if split_mode == 0:
        yield _WaitPush(pipe, tensor.clone())
        return tensor
    sb = _current_sb[0]
    yield _WaitPush(_pipes_sb[f'to_aic_sb{sb}'], tensor.clone())
    return tensor

def _tpop_from_aic_g(pipe, split_mode):
    if split_mode == 0:
        return (yield _WaitPop(pipe))
    sb = _current_sb[0]
    return (yield _WaitPop(_pipes_sb[f'to_aiv_sb{sb}']))

def _tpop_from_aiv_g(pipe, split_mode):
    if split_mode == 0:
        return (yield _WaitPop(pipe))
    sb0 = yield _WaitPop(_pipes_sb['to_aic_sb0'])
    sb1 = yield _WaitPop(_pipes_sb['to_aic_sb1'])
    if split_mode == 1:  # UpDown -> rows
        return torch.cat([sb0, sb1], dim=-2)
    if split_mode == 2:  # LeftRight -> cols
        return torch.cat([sb0, sb1], dim=-1)
    return sb0

def _run_scheduler(tasks):
    # Cooperative round-robin scheduler over generator-style AIC/AIV bodies.
    # ``tasks`` is a list of ``(name, generator, subblock_id)`` tuples.  Each
    # generator yields _WaitPop / _WaitPush requests at pipe sync points.
    # Pipes are unbounded deques so _WaitPush always succeeds; the
    # interesting suspend point is _WaitPop on an empty pipe.  The scheduler
    # keeps cycling until every generator returns; if a full pass makes no
    # progress and any task is still alive, it raises a deadlock error with
    # the pending request kinds.  Before resuming each task we set
    # ``_current_sb[0]`` so the split-aware pipe helpers and
    # tile.get_subblock_idx see the right id.
    states = []
    for name, gen, sb in tasks:
        _current_sb[0] = sb
        try:
            req = next(gen)
            states.append([name, gen, req, False, sb])
        except StopIteration:
            states.append([name, gen, None, True, sb])
    while True:
        progressed = False
        all_done = True
        for st in states:
            if st[3]:
                continue
            all_done = False
            req = st[2]
            advance_value = None
            advance = False
            if isinstance(req, _WaitPush):
                req.pipe.append(req.item)
                advance = True
            elif isinstance(req, _WaitPop):
                if len(req.pipe) > 0:
                    advance_value = req.pipe.popleft()
                    advance = True
            else:
                # Defensive: unknown yield value treated as cooperative yield.
                advance = True
            if advance:
                progressed = True
                _current_sb[0] = st[4]
                try:
                    st[2] = st[1].send(advance_value)
                except StopIteration:
                    st[3] = True
        if all_done:
            return
        if not progressed:
            blocked = [(s[0], type(s[2]).__name__) for s in states if not s[3]]
            raise RuntimeError(
                "Cross-core simulation deadlock; tasks blocked: " + repr(blocked)
            )
"""

# ---------------------------------------------------------------------------
# Op dispatch table: op_name -> Callable[[list[str], dict], str]
#
# Each handler receives (args: list[str], kwargs: dict[str, Any]) and returns
# a Python expression string.
# ---------------------------------------------------------------------------

OpHandler = Callable[[list[str], dict[str, Any]], str]


def _binop(op: str) -> OpHandler:
    """Create handler for a binary infix operator."""
    return lambda a, _kw: f"({a[0]} {op} {a[1]})"


def _torch_fn(name: str, nargs: int = 1) -> OpHandler:
    """Create handler for torch.<name>(arg0, ..., argN-1)."""

    def _handler(a: list[str], _kw: dict[str, Any]) -> str:
        return f"torch.{name}({', '.join(a[:nargs])})"

    return _handler


def _identity() -> OpHandler:
    return lambda a, _kw: a[0]


def _expand_as_target() -> OpHandler:
    # row_expand/col_expand in IR deduce promoted dtype from both operands.
    # Materialize expanded view to avoid aliasing issues with zero-stride expands.
    return lambda a, _kw: (
        f"{a[1]}.expand_as({a[0]}).clone().to(torch.promote_types({a[0]}.dtype, {a[1]}.dtype))"
    )


def _noop(comment: str = "") -> OpHandler:
    return lambda _a, _kw: f"None  # {comment}" if comment else "None"


def _handle_tensor_matmul(a: list[str], kw: dict[str, Any]) -> str:
    lhs, rhs = a[0], a[1]
    if kw.get("a_trans"):
        lhs = f"{lhs}.mT"
    if kw.get("b_trans"):
        rhs = f"{rhs}.mT"
    expr = f"torch.matmul({lhs}, {rhs})"
    out_dtype = kw.get("out_dtype")
    if isinstance(out_dtype, DataType):
        expr = f"{expr}.to({_torch_dtype(out_dtype)})"
    return expr


def _handle_tensor_matmul_acc(a: list[str], kw: dict[str, Any]) -> str:
    acc, lhs, rhs = a[0], a[1], a[2]
    if kw.get("a_trans"):
        lhs = f"{lhs}.mT"
    if kw.get("b_trans"):
        rhs = f"{rhs}.mT"
    expr = f"({acc} + torch.matmul({lhs}, {rhs}))"
    out_dtype = kw.get("out_dtype")
    if isinstance(out_dtype, DataType):
        expr = f"{expr}.to({_torch_dtype(out_dtype)})"
    return expr


def _handle_cast(a: list[str], kw: dict[str, Any]) -> str:
    dt = kw.get("target_type")
    dtype_str = _torch_dtype(dt) if isinstance(dt, DataType) else "torch.float32"
    return f"{a[0]}.to({dtype_str})"


def _kw_dtype(kw: dict[str, Any]) -> str:
    """Extract dtype from kwargs and convert to torch dtype string."""
    dt = kw.get("dtype")
    return _torch_dtype(dt) if isinstance(dt, DataType) else "torch.float32"


def _handle_tile_load(a: list[str], kw: dict[str, Any]) -> str:
    # args: [tensor, offsets_tuple, shapes_tuple, valid_shapes_tuple]
    expr = f"_tile_load({a[0]}, {a[1]}, {a[2]}, {a[3]})"
    if kw.get("transpose"):
        expr += ".mT"
    return expr


def _handle_tile_store(a: list[str], _kw: dict[str, Any]) -> str:
    # args: [tile, offsets_tuple, output_tensor] or [tile, offsets_tuple, output_tensor, shapes]
    return f"_tile_store({a[0]}, {a[1]}, {a[2]})"


def _handle_create(a: list[str], kw: dict[str, Any]) -> str:
    return f"torch.zeros({a[0]}, dtype={_kw_dtype(kw)})"


def _handle_full(a: list[str], kw: dict[str, Any]) -> str:
    return f"torch.full({a[0]}, {a[1]}, dtype={_kw_dtype(kw)})"


def _handle_cmp(a: list[str], kw: dict[str, Any]) -> str:
    op_str = _CMP_OPS.get(kw.get("cmp_type", 0), "==")
    return f"({a[0]} {op_str} {a[1]})"


def _handle_reduction(torch_fn: str) -> OpHandler:
    def _handler(a: list[str], kw: dict[str, Any]) -> str:
        axis = kw.get("axis")
        keepdim = kw.get("keepdim", False)
        if axis is not None:
            return f"{a[0]}.{torch_fn}(dim={axis}, keepdim={keepdim})"
        return f"{a[0]}.{torch_fn}()"

    return _handler


def _handle_slice(a: list[str], _kw: dict[str, Any]) -> str:
    # args: [tensor, shapes, offsets] or [tensor, shapes, offsets, valid_shapes]
    if len(a) >= 4:
        return f"_tensor_slice({a[0]}, {a[2]}, {a[1]}, {a[3]})"
    return f"_tensor_slice({a[0]}, {a[2]}, {a[1]})"


def _pad_mode_literal(kw: dict[str, Any]) -> str:
    pad_value = kw.get("pad_value")
    if pad_value is None:
        return '"zero"'
    s = getattr(pad_value, "name", str(pad_value)).lower()
    if "min" in s:
        return '"min"'
    if "max" in s:
        return '"max"'
    return '"zero"'


def _handle_fillpad(a: list[str], kw: dict[str, Any]) -> str:
    return f"_fillpad({a[0]}, {_pad_mode_literal(kw)})"


# Build the dispatch table
_OP_MAP: dict[str, OpHandler] = {}

# Cross-core scheduler-mode overrides: same op names, but emissions wrap into
# ``(yield from _..._g(...))`` so the enclosing AIC/AIV function becomes a
# Python generator that yields at every pipe sync point.  Only populated for
# the four cross-core ops; everything else falls back to ``_OP_MAP``.
_OP_MAP_SCHED: dict[str, OpHandler] = {
    "tile.tpush_to_aiv": (
        lambda a, kw: f"(yield from _tpush_to_aiv_g(_pipes['to_aiv'], {a[0]}, {kw.get('split', 0)}))"
    ),
    "tile.tpush_to_aic": (
        lambda a, kw: f"(yield from _tpush_to_aic_g(_pipes['to_aic'], {a[0]}, {kw.get('split', 0)}))"
    ),
    "tile.tpop_from_aic": (
        lambda _a, kw: f"(yield from _tpop_from_aic_g(_pipes['to_aiv'], {kw.get('split', 0)}))"
    ),
    "tile.tpop_from_aiv": (
        lambda _a, kw: f"(yield from _tpop_from_aiv_g(_pipes['to_aic'], {kw.get('split', 0)}))"
    ),
}


def _register_ops() -> None:
    m = _OP_MAP

    # --- Tensor element-wise binary ---
    for prefix in ("tensor", "tile"):
        m[f"{prefix}.add"] = _torch_fn("add", 2)
        m[f"{prefix}.sub"] = _torch_fn("sub", 2)
        m[f"{prefix}.mul"] = _torch_fn("mul", 2)
        m[f"{prefix}.div"] = _torch_fn("div", 2)
        m[f"{prefix}.maximum"] = _torch_fn("maximum", 2)
        m[f"{prefix}.minimum"] = _torch_fn("minimum", 2)

        # scalar variants: same math, torch broadcasting handles it
        m[f"{prefix}.adds"] = _binop("+")
        m[f"{prefix}.subs"] = _binop("-")
        m[f"{prefix}.muls"] = _binop("*")
        m[f"{prefix}.divs"] = _binop("/")
        m[f"{prefix}.maxs"] = _torch_fn("maximum", 2)
        m[f"{prefix}.mins"] = _torch_fn("minimum", 2)
        m[f"{prefix}.rems"] = _binop("%")

        # unary
        m[f"{prefix}.neg"] = _torch_fn("neg")
        m[f"{prefix}.exp"] = _torch_fn("exp")
        m[f"{prefix}.sqrt"] = _torch_fn("sqrt")
        # rsqrt in tile form may carry an optional tmp_tile arg for the high-precision
        # path; torch.rsqrt takes only the input, so ignore any extra operands.
        m[f"{prefix}.rsqrt"] = lambda a, _kw: f"torch.rsqrt({a[0]})"
        m[f"{prefix}.recip"] = _torch_fn("reciprocal")
        m[f"{prefix}.abs"] = _torch_fn("abs")

        # cast
        m[f"{prefix}.cast"] = _handle_cast

        # row reductions (take a tmp_tile arg in tile, ignore it)
        m[f"{prefix}.row_sum"] = lambda a, _kw: f"{a[0]}.sum(dim=-1, keepdim=True)"
        m[f"{prefix}.row_max"] = lambda a, _kw: f"{a[0]}.amax(dim=-1, keepdim=True)"
        m[f"{prefix}.row_min"] = lambda a, _kw: f"{a[0]}.amin(dim=-1, keepdim=True)"

        # reshape / transpose / slice / concat
        m[f"{prefix}.reshape"] = lambda a, _kw: f"{a[0]}.reshape({a[1]})"
        m[f"{prefix}.transpose"] = lambda a, _kw: f"{a[0]}.transpose({a[1]}, {a[2]})"
        m[f"{prefix}.concat"] = lambda a, _kw: f"torch.cat([{a[0]}, {a[1]}], dim=-1)"

        # fillpad
        m[f"{prefix}.fillpad"] = _handle_fillpad

        # assemble -> write source into target at offset
        m[f"{prefix}.assemble"] = lambda a, _kw: f"_assemble({a[0]}, {a[1]}, {a[2]})"

        # scatter_update
        m[f"{prefix}.scatter_update"] = lambda a, kw: f"{a[0]}.scatter_(-2, {a[1]}.expand_as({a[2]}), {a[2]})"

        # broadcast ops - torch broadcasting handles these naturally
        m[f"{prefix}.row_expand_add"] = _binop("+")
        m[f"{prefix}.row_expand_sub"] = _binop("-")
        m[f"{prefix}.row_expand_mul"] = _binop("*")
        m[f"{prefix}.row_expand_div"] = _binop("/")
        m[f"{prefix}.col_expand_mul"] = _binop("*")
        m[f"{prefix}.col_expand_sub"] = _binop("-")
        m[f"{prefix}.col_expand_div"] = _binop("/")
        m[f"{prefix}.col_expand"] = _expand_as_target()
        m[f"{prefix}.row_expand"] = _expand_as_target()
        m[f"{prefix}.expands"] = lambda a, _kw: f"torch.full_like({a[0]}, {a[1]})"

    # --- Tensor-only ops ---
    m["tensor.matmul"] = _handle_tensor_matmul
    m["tensor.matmul_acc"] = _handle_tensor_matmul_acc
    m["tensor.dim"] = lambda a, _kw: f"{a[0]}.shape[{a[1]}]"
    m["tensor.create"] = _handle_create
    m["tensor.full"] = _handle_full
    m["tensor.slice"] = _handle_slice
    m["tensor.read"] = lambda a, _kw: f"{a[0]}[{a[1]}]"
    m["tensor.write"] = lambda a, _kw: f"_write_and_return({a[0]}, {a[1]}, {a[2]})"

    # --- Tile-only ops ---
    m["tile.load"] = _handle_tile_load
    m["tile.store"] = _handle_tile_store
    m["tile.create"] = _handle_create
    m["tile.full"] = _handle_full
    m["tile.alloc"] = _handle_create
    m["tile.move"] = _identity()
    m["tile.slice"] = _handle_slice
    m["tile.read"] = lambda a, _kw: f"{a[0]}[{a[1]}]"
    m["tile.write"] = lambda a, _kw: f"_write_and_return({a[0]}, {a[1]}, {a[2]})"
    m["tile.get_block_idx"] = lambda _a, _kw: "0"
    # tile.get_subblock_idx returns the active AIV subblock id at runtime so
    # split-aware kernels can compute per-subblock offsets / slices.  Outside
    # a scheduled Group ``_current_sb[0]`` stays 0, matching the legacy
    # single-subblock behavior for unidirectional / split=0 callers.
    m["tile.get_subblock_idx"] = lambda _a, _kw: "_current_sb[0]"

    # tile log / relu
    m["tile.log"] = _torch_fn("log")
    m["tile.relu"] = _torch_fn("relu")
    m["tile.rem"] = _binop("%")

    # tile bitwise
    m["tile.and"] = _torch_fn("bitwise_and", 2)
    m["tile.or"] = _torch_fn("bitwise_or", 2)
    m["tile.not"] = _torch_fn("bitwise_not")
    m["tile.shl"] = _binop("<<")
    m["tile.shr"] = _binop(">>")
    m["tile.ands"] = _torch_fn("bitwise_and", 2)
    m["tile.ors"] = _torch_fn("bitwise_or", 2)
    m["tile.shls"] = _binop("<<")
    m["tile.shrs"] = _binop(">>")

    # tile cmp
    m["tile.cmp"] = _handle_cmp
    m["tile.cmps"] = _handle_cmp

    # tile matmul variants — .float() to match hardware FP32 accumulation output
    m["tile.matmul"] = lambda a, _kw: f"torch.matmul({a[0]}, {a[1]}).float()"
    m["tile.batch_matmul"] = lambda a, _kw: f"torch.matmul({a[0]}, {a[1]}).float()"
    m["tile.matmul_acc"] = lambda a, _kw: f"({a[0]} + torch.matmul({a[1]}, {a[2]}).float())"
    m["tile.matmul_bias"] = lambda a, _kw: f"(torch.matmul({a[0]}, {a[1]}).float() + {a[2]})"
    m["tile.gemv"] = lambda a, _kw: f"torch.matmul({a[0]}, {a[1]}).float()"
    m["tile.gemv_acc"] = lambda a, _kw: f"({a[0]} + torch.matmul({a[1]}, {a[2]}).float())"
    m["tile.gemv_bias"] = lambda a, _kw: f"(torch.matmul({a[0]}, {a[1]}).float() + {a[2]})"

    # tile reductions with axis kwarg
    m["tile.sum"] = _handle_reduction("sum")
    m["tile.max"] = _handle_reduction("amax")
    m["tile.min"] = _handle_reduction("amin")

    # tile ternary ops (third arg is workspace/tmp, ignore it)
    m["tile.xor"] = lambda a, _kw: f"torch.bitwise_xor({a[0]}, {a[1]})"
    m["tile.xors"] = lambda a, _kw: f"torch.bitwise_xor({a[0]}, {a[1]})"
    m["tile.prelu"] = lambda a, _kw: f"torch.where({a[0]} > 0, {a[0]}, {a[0]} * {a[1]})"

    # tile selection
    m["tile.sel"] = lambda a, _kw: f"torch.where({a[0]}, {a[1]}, {a[2]})"
    m["tile.sels"] = lambda a, _kw: f"torch.where({a[0]}, {a[1]}, {a[2]})"
    m["tile.lrelu"] = lambda a, _kw: f"torch.where({a[0]} > 0, {a[0]}, {a[0]} * {a[1]})"

    # tile ternary add/sub with carry
    m["tile.addc"] = lambda a, _kw: f"({a[0]} + {a[1]} + {a[2]})"
    m["tile.subc"] = lambda a, _kw: f"({a[0]} - {a[1]} - {a[2]})"
    m["tile.addsc"] = lambda a, _kw: f"({a[0]} + {a[1]} + {a[2]})"
    m["tile.subsc"] = lambda a, _kw: f"({a[0]} - {a[1]} - {a[2]})"

    # --- Cross-core pipe ops ---
    m["tile.tpush_to_aiv"] = lambda a, kw: f"_tpush_to_aiv(_pipes['to_aiv'], {a[0]}, {kw.get('split', 0)})"
    m["tile.tpush_to_aic"] = lambda a, kw: f"_tpush_to_aic(_pipes['to_aic'], {a[0]}, {kw.get('split', 0)})"
    m["tile.tpop_from_aic"] = lambda _a, kw: f"_tpop_from_aic(_pipes['to_aiv'], {kw.get('split', 0)})"
    m["tile.tpop_from_aiv"] = lambda _a, kw: f"_tpop_from_aiv(_pipes['to_aic'], {kw.get('split', 0)})"

    # --- System ops (no-ops) ---
    for op_name in (
        "system.sync_src",
        "system.sync_dst",
        "system.bar_v",
        "system.bar_m",
        "system.bar_all",
        "system.aic_initialize_pipe",
        "system.aiv_initialize_pipe",
        "system.reserve_buffer",
        "system.import_peer_buffer",
        "system.tfree_to_aic",
        "system.tfree_to_aiv",
    ):
        m[op_name] = _noop(op_name.split(".")[-1])


_register_ops()


# ---------------------------------------------------------------------------
# Helpers for cross-core program simulation
# ---------------------------------------------------------------------------


def _extract_group_member_names(group_func: _ir.Function) -> list[str]:
    """Extract AIC/AIV member function names from a Group function's body."""
    names: list[str] = []
    stmts = _ir.flatten_to_stmts(group_func.body)
    for stmt in stmts:
        call = None
        if isinstance(stmt, _ir.EvalStmt):
            call = stmt.expr
        elif isinstance(stmt, _ir.AssignStmt):
            call = stmt.value
        if isinstance(call, _ir.Call) and isinstance(call.op, _ir.GlobalVar):
            names.append(call.op.name)
    return names


def _generate_entry_point(program: _ir.Program) -> str:
    """Generate a ``run()`` entry-point wrapper for a Program.

    Prefers an Orchestration function, falls back to the first Opaque
    function, then Group, and returns an empty string if none exists.
    """
    entry_func = None
    for func in program.functions.values():
        if func.func_type == _ir.FunctionType.Orchestration:
            entry_func = func
            break
    if entry_func is None:
        for func in program.functions.values():
            if func.func_type == _ir.FunctionType.Opaque:
                entry_func = func
                break
    if entry_func is None:
        for func in program.functions.values():
            if func.func_type == _ir.FunctionType.Group:
                entry_func = func
                break
    if entry_func is None:
        return ""
    # If the entry function itself is named ``run``, skip emitting a wrapper
    # to avoid producing ``def run(...): return run(...)`` (infinite recursion).
    if entry_func.name == "run":
        return ""
    param_names = _make_unique_names([p.name_hint for p in entry_func.params])
    return (
        f"# Entry point\n"
        f"def run({', '.join(param_names)}):\n"
        f"    return {entry_func.name}({', '.join(param_names)})\n"
    )


# ---------------------------------------------------------------------------
# Binary / unary IR expression -> Python operator string
# ---------------------------------------------------------------------------
_BINARY_OP_STR: dict[type, str] = {
    _ir.Add: "+",
    _ir.Sub: "-",
    _ir.Mul: "*",
    _ir.FloorDiv: "//",
    _ir.FloorMod: "%",
    _ir.FloatDiv: "/",
    _ir.Min: "min",
    _ir.Max: "max",
    _ir.Pow: "**",
    _ir.Eq: "==",
    _ir.Ne: "!=",
    _ir.Lt: "<",
    _ir.Le: "<=",
    _ir.Gt: ">",
    _ir.Ge: ">=",
    _ir.And: "and",
    _ir.Or: "or",
    _ir.Xor: "^",
    _ir.BitAnd: "&",
    _ir.BitOr: "|",
    _ir.BitXor: "^",
    _ir.BitShiftLeft: "<<",
    _ir.BitShiftRight: ">>",
}


# ---------------------------------------------------------------------------
# TorchCodegen - IRVisitor subclass
# ---------------------------------------------------------------------------


class TorchCodegen(_ir.IRVisitor):
    """Emit executable PyTorch code from PyPTO IR."""

    def __init__(self, *, check_shapes: bool = False) -> None:
        super().__init__()
        self._lines: list[str] = []
        self._indent: int = 0
        self._expr_result: str = ""
        self._var_names: dict[int, str] = {}  # id(Var) -> unique name
        self._stable_hints: dict[str, str] = {}  # hint -> name for params/aliases only
        self._var_refs: list[_ir.Var] = []  # prevent GC of Var wrappers
        self._name_counter: dict[str, int] = {}
        self._yield_targets: list[str] = []  # names to assign on yield
        self._check_shapes: bool = check_shapes
        # Cross-core scheduler mode: function names that should be emitted as
        # generators using ``_OP_MAP_SCHED`` for tpush/tpop ops.  Populated by
        # ``visit_program`` after pattern detection.
        self._sched_funcs: set[str] = set()
        # AIV member functions that must be scheduled twice (one task per
        # AIV subblock id) because they participate in split>0 transfers.
        self._sched_aiv_dup: set[str] = set()
        self._current_func_name: str = ""

    # -- helpers --

    def _emit(self, line: str) -> None:
        self._lines.append("    " * self._indent + line)

    def _unique_name(self, hint: str) -> str:
        base = _sanitize_name_hint(hint)
        count = self._name_counter.get(base, 0)
        if count == 0:
            self._name_counter[base] = 1
            return base
        self._name_counter[base] = count + 1
        return f"{base}_{count}"

    def _name_of(self, var: _ir.Var) -> str:
        vid = id(var)
        if vid not in self._var_names:
            hint = var.name_hint
            # Nanobind may create fresh Python wrappers for the same C++
            # Var, giving each a different id().  Fall back to stable hints
            # (function params and loop aliases) where the mapping is
            # unambiguous.  Do NOT fall back for local SSA vars — different
            # Vars can share a name_hint and must get unique names.
            if hint in self._stable_hints:
                name = self._stable_hints[hint]
            else:
                name = self._unique_name(hint)
            self._var_names[vid] = name
            self._var_refs.append(var)  # prevent GC id reuse
        return self._var_names[vid]

    def _visit_expr_str(self, expr: _ir.Expr) -> str:
        # Force nested Call nodes through our Python visit_call implementation.
        # Some C++ visitor dispatch paths can bypass visit_call for nested calls
        # and leave only the last-visited argument string in _expr_result.
        if isinstance(expr, _ir.Call):
            self.visit_call(expr)
        else:
            self.visit_expr(expr)
        return self._expr_result

    def _has_body_content(self, stmt: _ir.Stmt) -> bool:
        """Check if a statement body produces any lines."""
        if isinstance(stmt, _ir.SeqStmts):
            return len(stmt.stmts) > 0
        return True

    def _emit_iter_arg_inits(self, iter_args: list[_ir.IterArg]) -> list[str]:
        """Emit init assignments for SSA iter_args and return their names."""
        names: list[str] = []
        for ia in iter_args:
            name = self._name_of(ia)
            names.append(name)
            init_val = self._visit_expr_str(ia.initValue)
            self._emit(f"{name} = {init_val}")
        return names

    def _alias_return_vars(self, return_vars: list[_ir.Var], names: list[str]) -> None:
        """Map return_vars to the same names as iter_args after a loop."""
        for rv, name in zip(return_vars, names):
            self._var_names[id(rv)] = name
            self._stable_hints[rv.name_hint] = name
            self._var_refs.append(rv)

    # -- top-level --

    def _reset_var_scope(self) -> None:
        """Reset per-function variable naming state.

        Each function in a Program gets its own naming scope so that
        identically-named IR variables in different functions do not
        collide.
        """
        self._var_names.clear()
        self._stable_hints.clear()
        self._var_refs.clear()
        self._name_counter.clear()

    def visit_program(self, program: _ir.Program) -> None:
        # Classify functions by type for dependency-ordered emission
        aic_aiv_funcs: list[_ir.Function] = []
        group_funcs: list[_ir.Function] = []
        orch_funcs: list[_ir.Function] = []
        other_funcs: list[_ir.Function] = []

        for _gv, func in program.functions.items():
            ft = func.func_type
            if ft in (_ir.FunctionType.AIC, _ir.FunctionType.AIV):
                aic_aiv_funcs.append(func)
            elif ft == _ir.FunctionType.Group:
                group_funcs.append(func)
            elif ft == _ir.FunctionType.Orchestration:
                orch_funcs.append(func)
            else:
                other_funcs.append(func)

        # Cross-core scheduler detection: any Group whose AIC/AIV members
        # together contain bidirectional pipe access (tpush_to_aiv AND
        # tpush_to_aic), or any single-direction member that does both
        # tpush and tpop on the same side, OR uses split>0 on any pipe op,
        # must be emitted with the cooperative scheduler so that pipe sync
        # points are honored and split tiles fan out to two AIV subblocks.
        funcs_by_name = {f.name: f for f in program.functions.values()}
        scheduled_groups = self._detect_scheduled_groups(program, group_funcs)
        for grp in scheduled_groups:
            for member_name in _extract_group_member_names(grp):
                self._sched_funcs.add(member_name)
                member = funcs_by_name.get(member_name)
                if member is None:
                    continue
                # AIV members that touch the pipe with split>0 must run twice
                # under the scheduler (once per AIV subblock 0/1) so both
                # halves of a split tile are produced/consumed.
                if member.func_type == _ir.FunctionType.AIV and self._function_uses_split(member):
                    self._sched_aiv_dup.add(member_name)

        # Emit in dependency order: AIC/AIV leaves → Opaque/InCore → Group → Orchestration
        for func in aic_aiv_funcs:
            self._reset_var_scope()
            self.visit_function(func)
        for func in other_funcs:
            self._reset_var_scope()
            self.visit_function(func)
        for func in group_funcs:
            self._reset_var_scope()
            if func in scheduled_groups:
                self._visit_scheduled_group_function(func)
            else:
                self._visit_group_function(func, program)
        for func in orch_funcs:
            self._reset_var_scope()
            self.visit_function(func)

    _PIPE_OP_NAMES = (
        "tile.tpush_to_aiv",
        "tile.tpush_to_aic",
        "tile.tpop_from_aic",
        "tile.tpop_from_aiv",
    )

    @classmethod
    def _walk_pipe_calls(cls, func: _ir.Function):
        """Yield every cross-core pipe ``Call`` node found anywhere in ``func``."""
        targets = set(cls._PIPE_OP_NAMES)

        def walk(node):
            if isinstance(node, _ir.Call):
                name = node.op.name if hasattr(node.op, "name") else None
                if name in targets:
                    yield node
                for a in node.args:
                    yield from walk(a)
                for v in (node.kwargs or {}).values():
                    yield from walk(v)
                return
            for attr in ("body", "expr", "value", "args", "kwargs", "stmts"):
                if not hasattr(node, attr):
                    continue
                child = getattr(node, attr)
                if isinstance(child, dict):
                    for v in child.values():
                        yield from walk(v)
                elif isinstance(child, (list, tuple)):
                    for v in child:
                        yield from walk(v)
                elif child is not None:
                    yield from walk(child)

        yield from walk(func.body)

    @classmethod
    def _scan_pipe_ops(cls, func: _ir.Function) -> set[str]:
        """Return the set of cross-core pipe op names found anywhere in ``func``."""
        return {call.op.name for call in cls._walk_pipe_calls(func)}

    @classmethod
    def _function_uses_split(cls, func: _ir.Function) -> bool:
        """Return True if any pipe op in ``func`` carries split>0."""
        for call in cls._walk_pipe_calls(func):
            kw = call.kwargs or {}
            split = kw.get("split", 0)
            if isinstance(split, int) and split > 0:
                return True
        return False

    def _detect_scheduled_groups(
        self, program: _ir.Program, group_funcs: list[_ir.Function]
    ) -> set[_ir.Function]:
        """Identify Group functions that need the cooperative scheduler.

        Trigger conditions (any one is sufficient):
          * Bidirectional: members collectively contain both ``tpush_to_aiv``
            and ``tpush_to_aic``.
          * Same-side feedback: a single member contains both a ``tpush`` and
            a ``tpop`` on the *same* pipe direction (would deadlock self).
          * Split>0 on any pipe op: split tiles need both AIV subblocks to
            run, which only the scheduler models correctly.

        Single-direction Groups with simple linear push/pop and split=0
        continue to use the legacy sequential emission path (zero behavior
        change).
        """
        funcs_by_name = {f.name: f for f in program.functions.values()}
        scheduled: set[_ir.Function] = set()
        for grp in group_funcs:
            members = [funcs_by_name[n] for n in _extract_group_member_names(grp) if n in funcs_by_name]
            if not members:
                continue
            all_ops: set[str] = set()
            same_side_feedback = False
            uses_split = False
            for m in members:
                m_ops = self._scan_pipe_ops(m)
                all_ops |= m_ops
                # Same-side feedback within one member function:
                if "tile.tpush_to_aiv" in m_ops and "tile.tpop_from_aic" in m_ops:
                    same_side_feedback = True
                if "tile.tpush_to_aic" in m_ops and "tile.tpop_from_aiv" in m_ops:
                    same_side_feedback = True
                if self._function_uses_split(m):
                    uses_split = True
            bidirectional = "tile.tpush_to_aiv" in all_ops and "tile.tpush_to_aic" in all_ops
            if bidirectional or same_side_feedback or uses_split:
                scheduled.add(grp)
        return scheduled

    def _visit_scheduled_group_function(self, func: _ir.Function) -> None:
        """Emit a Group function in cooperative-scheduler form.

        Each AIC/AIV member call inside the Group body is converted into a
        ``(name, generator)`` entry handed to ``_run_scheduler``.  All other
        statements (asserts, tensor wrap-up, etc.) are re-emitted verbatim
        through the normal visitor so non-call setup still runs in order.
        """
        params = [self._name_of(p) for p in func.params]
        self._register_param_hints(func.params)
        self._emit(f"def {func.name}({', '.join(params)}):")
        self._indent += 1
        member_names = _extract_group_member_names(func)
        if member_names:
            self._emit(f"# Group (scheduled): {', '.join(member_names)}")
        if self._check_shapes:
            for p in func.params:
                self._emit_shape_dtype_check(self._name_of(p), p.type, shape=False)
        self._emit("_reset_pipes()")

        # Walk the (flattened) Group body and split call statements out into
        # the scheduler tasks list.  Anything else is emitted in place.
        stmts = _ir.flatten_to_stmts(func.body)
        task_lines: list[str] = []
        for stmt in stmts:
            call = None
            if isinstance(stmt, _ir.EvalStmt):
                call = stmt.expr
            elif isinstance(stmt, _ir.AssignStmt):
                call = stmt.value
            if (
                isinstance(call, _ir.Call)
                and isinstance(call.op, _ir.GlobalVar)
                and call.op.name in self._sched_funcs
            ):
                arg_strs = [self._visit_expr_str(a) for a in call.args]
                fname = call.op.name
                joined = ", ".join(arg_strs)
                if fname in self._sched_aiv_dup:
                    # Schedule the AIV body twice, once per subblock id.
                    for sb in (0, 1):
                        task_lines.append(f'        ("{fname}#{sb}", {fname}({joined}), {sb}),')
                else:
                    task_lines.append(f'        ("{fname}", {fname}({joined}), 0),')
            else:
                # Fall back to default per-statement emission for anything
                # that is not an AIC/AIV member call (rare in well-formed
                # Group bodies but kept for safety).
                self.visit_stmt(stmt)
        if task_lines:
            self._emit("_run_scheduler([")
            for line in task_lines:
                self._lines.append(line)
            self._emit("])")
        else:
            self._emit("pass")
        self._indent -= 1
        self._emit("")

    def _register_param_hints(self, params) -> None:
        """Register param ``name_hint`` -> emitted name in ``_stable_hints``.

        ``name_hint`` collisions are dropped (not registered) so that fresh
        nanobind wrappers for ambiguously-named params fall back to the
        counter-based uniquing path in ``_unique_name`` instead of silently
        resolving to the wrong param.
        """
        seen: dict[str, str] = {}
        ambiguous: set[str] = set()
        for p in params:
            hint = p.name_hint
            name = self._var_names[id(p)]
            if hint in ambiguous:
                continue
            if hint in seen and seen[hint] != name:
                ambiguous.add(hint)
                self._stable_hints.pop(hint, None)
                continue
            seen[hint] = name
            self._stable_hints[hint] = name

    def _visit_group_function(self, func: _ir.Function, _program: _ir.Program) -> None:
        """Generate a Group function that calls its AIC+AIV members sequentially."""
        params = [self._name_of(p) for p in func.params]
        # Register param hints as stable so nanobind wrapper GC doesn't
        # break references to these vars inside the function body.
        self._register_param_hints(func.params)
        self._emit(f"def {func.name}({', '.join(params)}):")
        self._indent += 1

        member_names = _extract_group_member_names(func)
        if member_names:
            self._emit(f"# Group: {', '.join(member_names)}")

        if self._check_shapes:
            for p in func.params:
                self._emit_shape_dtype_check(self._name_of(p), p.type, shape=False)

        n_before = len(self._lines)
        self.visit_stmt(func.body)
        if len(self._lines) == n_before:
            self._emit("pass")
        self._indent -= 1
        self._emit("")

    def visit_function(self, func: _ir.Function) -> None:
        params = [self._name_of(p) for p in func.params]
        # Register param hints as stable so nanobind wrapper GC doesn't
        # break references to these vars inside the function body.
        self._register_param_hints(func.params)
        self._emit(f"def {func.name}({', '.join(params)}):")
        self._indent += 1
        prev_name = self._current_func_name
        self._current_func_name = func.name
        if self._check_shapes:
            for p in func.params:
                # InCore kernel params may receive partial data (boundary tiles),
                # so only check dtype — not shape — for all function params.
                self._emit_shape_dtype_check(self._name_of(p), p.type, shape=False)
        n_before = len(self._lines)
        self.visit_stmt(func.body)
        if len(self._lines) == n_before:
            self._emit("pass")
        self._indent -= 1
        self._current_func_name = prev_name
        self._emit("")

    # -- expression visitors --

    def visit_var(self, op: _ir.Var) -> None:
        self._expr_result = self._name_of(op)

    def visit_iter_arg(self, op: _ir.IterArg) -> None:
        self._expr_result = self._name_of(op)

    def visit_mem_ref(self, op: _ir.MemRef) -> None:
        self._expr_result = self._name_of(op)

    def visit_const_int(self, op: _ir.ConstInt) -> None:
        self._expr_result = str(op.value)

    def visit_const_float(self, op: _ir.ConstFloat) -> None:
        self._expr_result = repr(op.value)

    def visit_const_bool(self, op: _ir.ConstBool) -> None:
        self._expr_result = "True" if op.value else "False"

    def visit_make_tuple(self, op: _ir.MakeTuple) -> None:
        elems = [self._visit_expr_str(e) for e in op.elements]
        self._expr_result = f"({', '.join(elems)},)" if len(elems) == 1 else f"({', '.join(elems)})"

    def visit_tuple_get_item_expr(self, op: _ir.TupleGetItemExpr) -> None:
        tup = self._visit_expr_str(op.tuple)
        self._expr_result = f"{tup}[{op.index}]"

    def visit_binary_expr(self, op: _ir.BinaryExpr) -> None:
        left = self._visit_expr_str(op.left)
        right = self._visit_expr_str(op.right)
        op_str = _BINARY_OP_STR.get(type(op), "+")
        if op_str in ("min", "max"):
            self._expr_result = f"{op_str}({left}, {right})"
        else:
            self._expr_result = f"({left} {op_str} {right})"

    def visit_unary_expr(self, op: _ir.UnaryExpr) -> None:
        operand = self._visit_expr_str(op.operand)
        if isinstance(op, _ir.Neg):
            self._expr_result = f"(-{operand})"
        elif isinstance(op, _ir.Not):
            self._expr_result = f"(not {operand})"
        elif isinstance(op, _ir.BitNot):
            self._expr_result = f"(~{operand})"
        elif isinstance(op, _ir.Abs):
            self._expr_result = f"abs({operand})"
        elif isinstance(op, _ir.Cast):
            self._expr_result = (
                f"{operand}.to({_torch_dtype(op.dtype)})" if hasattr(op, "dtype") else f"int({operand})"
            )
        else:
            self._expr_result = operand

    def visit_call(self, op: _ir.Call) -> None:
        op_name = op.op.name
        if self._current_func_name in self._sched_funcs and op_name in _OP_MAP_SCHED:
            handler = _OP_MAP_SCHED[op_name]
        else:
            handler = _OP_MAP.get(op_name)

        # Evaluate arguments
        arg_strs = [self._visit_expr_str(a) for a in op.args]
        kw = dict(op.kwargs) if op.kwargs else {}

        if handler is not None:
            self._expr_result = handler(arg_strs, kw)
        elif isinstance(op.op, _ir.GlobalVar):
            # Cross-function call
            self._expr_result = f"{op_name}({', '.join(arg_strs)})"
        else:
            raise ValueError(
                f"Unsupported op '{op_name}' in torch_codegen. "
                f"Register a handler in _OP_MAP or use a GlobalVar for cross-function calls."
            )

    # -- statement visitors --

    def _emit_shape_dtype_check(self, var_name: str, var_type: _ir.Type, *, shape: bool = True) -> None:
        """Emit runtime assertions for tensor/tile shape and dtype.

        Args:
            var_name: The Python variable name to check.
            var_type: The IR type annotation.
            shape: If True, also check shape (not just dtype).  Function
                parameters may receive partial tiles so shape checks are
                skipped for them.
        """
        if not isinstance(var_type, (_ir.TensorType, _ir.TileType)):
            return

        ir_shape = var_type.shape
        dtype = var_type.dtype
        torch_dt = _torch_dtype(dtype)

        self._emit(
            f"assert isinstance({var_name}, torch.Tensor), "
            f'f"Expected {var_name} to be a Tensor, got {{type({var_name}).__name__}}"'
        )
        if shape:
            # Check if all dimensions are ConstInt.  Non-ConstInt dimensions
            # (including Vars from pl.dynamic()) cause us to fall back to an
            # ndim-only check plus per-static-dim assertions.
            all_static = all(isinstance(d, _ir.ConstInt) for d in ir_shape)
            if all_static:
                dim_strs = [self._visit_expr_str(d) for d in ir_shape]
                shape_expr = f"({', '.join(dim_strs)},)" if len(dim_strs) == 1 else f"({', '.join(dim_strs)})"
                self._emit(
                    f"assert {var_name}.shape == {shape_expr}, "
                    f'f"Shape mismatch for {var_name}: expected {shape_expr}, got {{{var_name}.shape}}"'
                )
            else:
                # At least one dynamic dim — only check rank and static dims
                ndim = len(ir_shape)
                self._emit(
                    f"assert {var_name}.ndim == {ndim}, "
                    f'f"Rank mismatch for {var_name}: expected {ndim}D, got {{{var_name}.ndim}}D"'
                )
                for i, d in enumerate(ir_shape):
                    if isinstance(d, _ir.ConstInt):
                        self._emit(
                            f"assert {var_name}.shape[{i}] == {d.value}, "
                            f'f"Dim {i} mismatch for {var_name}: expected {d.value}, '
                            f'got {{{var_name}.shape[{i}]}}"'
                        )
        self._emit(
            f"assert {var_name}.dtype == {torch_dt}, "
            f'f"Dtype mismatch for {var_name}: expected {torch_dt}, got {{{var_name}.dtype}}"'
        )

    def visit_assign_stmt(self, op: _ir.AssignStmt) -> None:
        name = self._name_of(op.var)
        val = self._visit_expr_str(op.value)
        self._emit(f"{name} = {val}")
        if self._check_shapes:
            self._emit_shape_dtype_check(name, op.var.type)

    def visit_eval_stmt(self, op: _ir.EvalStmt) -> None:
        val = self._visit_expr_str(op.expr)
        self._emit(val)

    def visit_return_stmt(self, op: _ir.ReturnStmt) -> None:
        if op.value:
            vals = [self._visit_expr_str(v) for v in op.value]
            if len(vals) == 1:
                self._emit(f"return {vals[0]}")
            else:
                self._emit(f"return {', '.join(vals)}")
        else:
            self._emit("return")

    def visit_seq_stmts(self, op: _ir.SeqStmts) -> None:
        for s in op.stmts:
            self.visit_stmt(s)

    def visit_scope_stmt(self, op: _ir.ScopeStmt) -> None:
        # Scopes are transparent - just emit the body
        self.visit_stmt(op.body)

    def visit_break_stmt(self, _op: _ir.BreakStmt) -> None:
        self._emit("break")

    def visit_continue_stmt(self, _op: _ir.ContinueStmt) -> None:
        self._emit("continue")

    def visit_yield_stmt(self, op: _ir.YieldStmt) -> None:
        if self._yield_targets and op.value:
            for target, val_expr in zip(self._yield_targets, op.value):
                val = self._visit_expr_str(val_expr)
                self._emit(f"{target} = {val}")

    def visit_for_stmt(self, op: _ir.ForStmt) -> None:
        loop_var = self._name_of(op.loop_var)
        start = self._visit_expr_str(op.start)
        stop = self._visit_expr_str(op.stop)
        step = self._visit_expr_str(op.step)

        iter_arg_names = self._emit_iter_arg_inits(op.iter_args)

        old_targets = self._yield_targets
        self._yield_targets = iter_arg_names

        self._emit(f"for {loop_var} in range({start}, {stop}, {step}):")
        self._indent += 1
        self.visit_stmt(op.body)
        if not op.iter_args and not self._has_body_content(op.body):
            self._emit("pass")
        self._indent -= 1

        self._yield_targets = old_targets
        self._alias_return_vars(op.return_vars, iter_arg_names)

    def visit_while_stmt(self, op: _ir.WhileStmt) -> None:
        iter_arg_names = self._emit_iter_arg_inits(op.iter_args)

        old_targets = self._yield_targets
        self._yield_targets = iter_arg_names

        cond = self._visit_expr_str(op.condition)
        self._emit(f"while {cond}:")
        self._indent += 1
        self.visit_stmt(op.body)
        self._indent -= 1

        self._yield_targets = old_targets
        self._alias_return_vars(op.return_vars, iter_arg_names)

    def visit_if_stmt(self, op: _ir.IfStmt) -> None:
        cond = self._visit_expr_str(op.condition)

        return_var_names = [self._name_of(rv) for rv in op.return_vars]

        old_targets = self._yield_targets
        self._yield_targets = return_var_names

        self._emit(f"if {cond}:")
        self._indent += 1
        self.visit_stmt(op.then_body)
        if not self._has_body_content(op.then_body):
            self._emit("pass")
        self._indent -= 1

        if op.else_body is not None:
            self._emit("else:")
            self._indent += 1
            self.visit_stmt(op.else_body)
            if not self._has_body_content(op.else_body):
                self._emit("pass")
            self._indent -= 1

        self._yield_targets = old_targets

    def get_output(self) -> str:
        return "\n".join(self._lines)


# The C++ IRVisitor dispatches to specific visit_add, visit_mul, etc. rather
# than the generic visit_binary_expr / visit_unary_expr.  Generate thin
# delegates so the codegen in those generic methods is actually reached.
for _method_name in (
    "visit_add",
    "visit_sub",
    "visit_mul",
    "visit_floor_div",
    "visit_floor_mod",
    "visit_float_div",
    "visit_min",
    "visit_max",
    "visit_pow",
    "visit_eq",
    "visit_ne",
    "visit_lt",
    "visit_le",
    "visit_gt",
    "visit_ge",
    "visit_and",
    "visit_or",
    "visit_xor",
    "visit_bit_and",
    "visit_bit_or",
    "visit_bit_xor",
    "visit_bit_shift_left",
    "visit_bit_shift_right",
):
    setattr(TorchCodegen, _method_name, TorchCodegen.visit_binary_expr)

for _method_name in ("visit_neg", "visit_not", "visit_bit_not", "visit_abs", "visit_cast"):
    setattr(TorchCodegen, _method_name, TorchCodegen.visit_unary_expr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def torch_codegen(node: _ir.Program | _ir.Function, *, check_shapes: bool = False) -> str:
    """Emit executable PyTorch code from a PyPTO IR Program or Function.

    The generated code can be exec()'d with torch available to numerically
    verify IR semantics at any pipeline stage.

    Args:
        node: A Program or Function IR node
        check_shapes: If True, emit runtime assertions to verify that every
            tensor/tile variable's shape and dtype match the IR type annotations.

    Returns:
        String of executable Python/PyTorch code
    """
    cg = TorchCodegen(check_shapes=check_shapes)
    lines = [_PREAMBLE]

    if isinstance(node, _ir.Program):
        cg.visit_program(node)
        lines.append(cg.get_output())
        entry = _generate_entry_point(node)
        if entry:
            lines.append(entry)
    elif isinstance(node, _ir.Function):
        cg.visit_function(node)
        lines.append(cg.get_output())
    else:
        raise TypeError(f"torch_codegen expects Program or Function, got {type(node).__name__}")

    return "\n".join(lines)
