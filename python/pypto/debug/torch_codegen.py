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

# ---------------------------------------------------------------------------
# Preamble inserted at top of every generated script
# ---------------------------------------------------------------------------
_PREAMBLE = """\
import torch
from collections import deque

_pipes = {'to_aiv': deque(), 'to_aic': deque()}

def _mask_valid_region(tensor, shapes, valid_shapes):
    if valid_shapes is not None and tuple(valid_shapes) != tuple(shapes):
        masked = tensor.new_zeros(shapes)
        valid_slices = tuple(slice(0, s) for s in valid_shapes)
        masked[valid_slices] = tensor[valid_slices]
        return masked
    return tensor

def _tile_load(tensor, offsets, shapes, valid_shapes=None):
    slices = tuple(slice(o, o + s) for o, s in zip(offsets, shapes))
    tile = tensor[slices].clone()
    # Pad to requested shape if source is smaller (boundary case)
    if tile.shape != tuple(shapes):
        padded = tile.new_zeros(shapes)
        pad_slices = tuple(slice(0, s) for s in tile.shape)
        padded[pad_slices] = tile
        tile = padded
    return _mask_valid_region(tile, shapes, valid_shapes)

def _tile_store(tile, offsets, output_tensor):
    slices = tuple(slice(o, o + s) for o, s in zip(offsets, tile.shape))
    output_tensor[slices] = tile
    return output_tensor

def _tensor_slice(tensor, offsets, shapes):
    slices = tuple(slice(o, o + s) for o, s in zip(offsets, shapes))
    return tensor[slices]

def _write_and_return(container, index, value):
    container[index] = value
    return container

def _assemble(target, source, offsets):
    slices = tuple(slice(o, o + s) for o, s in zip(offsets, source.shape))
    target[slices] = source
    return target
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
    return f"torch.matmul({lhs}, {rhs})"


def _handle_tensor_matmul_acc(a: list[str], kw: dict[str, Any]) -> str:
    acc, lhs, rhs = a[0], a[1], a[2]
    if kw.get("a_trans"):
        lhs = f"{lhs}.mT"
    if kw.get("b_trans"):
        rhs = f"{rhs}.mT"
    return f"({acc} + torch.matmul({lhs}, {rhs}))"


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
    # Slice returns a view — valid_shapes is metadata only and must not
    # trigger masking, otherwise in-place writes won't propagate back.
    return f"_tensor_slice({a[0]}, {a[2]}, {a[1]})"


# Build the dispatch table
_OP_MAP: dict[str, OpHandler] = {}


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

        # fillpad -> identity
        m[f"{prefix}.fillpad"] = _identity()

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
    m["tile.tpush_to_aiv"] = lambda a, _kw: f"_pipes['to_aiv'].append({a[0]}.clone())"
    m["tile.tpush_to_aic"] = lambda a, _kw: f"_pipes['to_aic'].append({a[0]}.clone())"
    m["tile.tpop_from_aic"] = lambda _a, _kw: "_pipes['to_aic'].popleft()"
    m["tile.tpop_from_aiv"] = lambda _a, _kw: "_pipes['to_aiv'].popleft()"

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
        self._name_counter: dict[str, int] = {}
        self._yield_targets: list[str] = []  # names to assign on yield
        self._check_shapes: bool = check_shapes

    # -- helpers --

    def _emit(self, line: str) -> None:
        self._lines.append("    " * self._indent + line)

    def _unique_name(self, hint: str) -> str:
        base = hint or "v"
        # Sanitize: replace non-identifier chars with underscore
        base = re.sub(r"[^a-zA-Z0-9_]", "_", base)
        # Collapse consecutive underscores
        base = re.sub(r"__+", "_", base).strip("_") or "v"
        # Ensure doesn't start with digit
        if base[0].isdigit():
            base = f"v_{base}"
        # Avoid Python keywords
        if keyword.iskeyword(base):
            base = f"{base}_v"
        count = self._name_counter.get(base, 0)
        if count == 0:
            self._name_counter[base] = 1
            return base
        self._name_counter[base] = count + 1
        return f"{base}_{count}"

    def _name_of(self, var: _ir.Var) -> str:
        vid = id(var)
        if vid not in self._var_names:
            self._var_names[vid] = self._unique_name(var.name_hint)
        return self._var_names[vid]

    def _visit_expr_str(self, expr: _ir.Expr) -> str:
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

    # -- top-level --

    def visit_program(self, program: _ir.Program) -> None:
        for _gv, func in program.functions.items():
            self.visit_function(func)

    def visit_function(self, func: _ir.Function) -> None:
        params = [self._name_of(p) for p in func.params]
        self._emit(f"def {func.name}({', '.join(params)}):")
        self._indent += 1
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
    elif isinstance(node, _ir.Function):
        cg.visit_function(node)
    else:
        raise TypeError(f"torch_codegen expects Program or Function, got {type(node).__name__}")

    lines.append(cg.get_output())
    return "\n".join(lines)
