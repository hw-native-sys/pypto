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
from pathlib import Path
from typing import Any

import torch

import pypto.language as pl
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

_SSA_SUFFIX_RE = re.compile(r"__ssa_v\d+$")

# ---------------------------------------------------------------------------
# Preamble inserted at top of every generated script
# ---------------------------------------------------------------------------
_PREAMBLE = """\
import torch
import threading
import time
import traceback
from collections import deque

_PIPE_WAIT_TIMEOUT_SEC = 10.0
_MIXED_KERNEL_TIMEOUT_SEC = 30.0

_GROUP_META = {}
_thread_local = threading.local()
_runtime_cancel_event = None


def _set_subblock_idx(idx):
    _thread_local.subblock_idx = int(idx)


def _get_subblock_idx():
    return int(getattr(_thread_local, "subblock_idx", 0))


def _set_runtime_cancel_event(event):
    global _runtime_cancel_event
    _runtime_cancel_event = event


def _get_runtime_cancel_event():
    return _runtime_cancel_event


def _copy_region_attrs(src, dst):
    valid_shape = getattr(src, "_pypto_valid_shape", None)
    full_shape = getattr(src, "_pypto_full_shape", None)
    if valid_shape is not None:
        dst._pypto_valid_shape = tuple(int(s) for s in valid_shape)
    if full_shape is not None:
        dst._pypto_full_shape = tuple(int(s) for s in full_shape)
    return dst


class _CrossCoreRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.reset()

    def reset(self, no_split_dual_aiv_dispatch=False):
        with self._cv:
            self._no_split_dual_aiv_dispatch = bool(no_split_dual_aiv_dispatch)
            self._to_aiv = deque()
            self._to_aiv_split = {1: {0: deque(), 1: deque()}, 2: {0: deque(), 1: deque()}}
            self._to_aiv_dual_nosplit = {0: deque(), 1: deque()}
            self._to_aic = deque()
            self._to_aic_split = {1: {0: deque(), 1: deque()}, 2: {0: deque(), 1: deque()}}
            self._to_aic_dual_nosplit = {0: deque(), 1: deque()}
            self._cv.notify_all()

    def snapshot(self):
        with self._lock:
            return self._snapshot_locked()

    def notify_all(self):
        with self._cv:
            self._cv.notify_all()

    def _snapshot_locked(self):
        return {
            "no_split_dual_aiv_dispatch": self._no_split_dual_aiv_dispatch,
            "to_aiv": len(self._to_aiv),
            "to_aiv_dual_nosplit": {
                0: len(self._to_aiv_dual_nosplit[0]),
                1: len(self._to_aiv_dual_nosplit[1]),
            },
            "to_aiv_split": {
                1: {0: len(self._to_aiv_split[1][0]), 1: len(self._to_aiv_split[1][1])},
                2: {0: len(self._to_aiv_split[2][0]), 1: len(self._to_aiv_split[2][1])},
            },
            "to_aic": len(self._to_aic),
            "to_aic_dual_nosplit": {
                0: len(self._to_aic_dual_nosplit[0]),
                1: len(self._to_aic_dual_nosplit[1]),
            },
            "to_aic_split": {
                1: {0: len(self._to_aic_split[1][0]), 1: len(self._to_aic_split[1][1])},
                2: {0: len(self._to_aic_split[2][0]), 1: len(self._to_aic_split[2][1])},
            },
        }

    def _wait_for_locked(self, predicate, op_name, split, lane):
        deadline = time.monotonic() + _PIPE_WAIT_TIMEOUT_SEC
        while not predicate():
            cancel_event = _get_runtime_cancel_event()
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError(
                    f"{op_name} cancelled (split={split}, lane={lane}); "
                    f"pipe_state={self._snapshot_locked()}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"{op_name} timeout (split={split}, lane={lane}); "
                    f"pipe_state={self._snapshot_locked()}"
                )
            self._cv.wait(timeout=min(0.05, remaining))

    @staticmethod
    def _split_tile(tile, split):
        dim = 0 if split == 1 else 1
        size = int(tile.shape[dim])
        if size % 2 != 0:
            raise ValueError(f"Split mode {split} requires even dimension size, got {size}")
        half = size // 2
        if dim == 0:
            part0 = tile[:half, ...].clone()
            part1 = tile[half:, ...].clone()
        else:
            part0 = tile[:, :half, ...].clone()
            part1 = tile[:, half:, ...].clone()

        full_shape = getattr(tile, "_pypto_full_shape", None)
        if full_shape is not None:
            full0 = list(int(s) for s in full_shape)
            full1 = list(int(s) for s in full_shape)
            full0[dim] = min(full0[dim], half)
            full1[dim] = max(full1[dim] - half, 0)
            part0._pypto_full_shape = tuple(full0)
            part1._pypto_full_shape = tuple(full1)

        valid_shape = getattr(tile, "_pypto_valid_shape", None)
        if valid_shape is not None:
            valid0 = list(int(s) for s in valid_shape)
            valid1 = list(int(s) for s in valid_shape)
            valid0[dim] = min(valid0[dim], half)
            valid1[dim] = max(valid1[dim] - half, 0)
            part0._pypto_valid_shape = tuple(valid0)
            part1._pypto_valid_shape = tuple(valid1)

        return part0, part1

    @staticmethod
    def _merge_tile(part0, part1, split):
        dim = 0 if split == 1 else 1
        merged = torch.cat([part0, part1], dim=dim)

        full0 = getattr(part0, "_pypto_full_shape", tuple(part0.shape))
        full1 = getattr(part1, "_pypto_full_shape", tuple(part1.shape))
        if len(full0) == len(full1):
            merged_full = list(int(s) for s in full0)
            merged_full[dim] = int(full0[dim]) + int(full1[dim])
            merged._pypto_full_shape = tuple(merged_full)

        valid0 = getattr(part0, "_pypto_valid_shape", tuple(part0.shape))
        valid1 = getattr(part1, "_pypto_valid_shape", tuple(part1.shape))
        if len(valid0) == len(valid1):
            merged_valid = list(int(s) for s in valid0)
            merged_valid[dim] = int(valid0[dim]) + int(valid1[dim])
            full_shape = getattr(merged, "_pypto_full_shape", None)
            if full_shape is not None:
                merged_valid = [min(v, int(f)) for v, f in zip(merged_valid, full_shape)]
            merged._pypto_valid_shape = tuple(merged_valid)

        return merged

    def push_to_aiv(self, tile, split):
        split = int(split)
        with self._cv:
            if split == 0:
                if self._no_split_dual_aiv_dispatch:
                    self._to_aiv_dual_nosplit[0].append(_copy_region_attrs(tile, tile.clone()))
                    self._to_aiv_dual_nosplit[1].append(_copy_region_attrs(tile, tile.clone()))
                else:
                    self._to_aiv.append(_copy_region_attrs(tile, tile.clone()))
            elif split in (1, 2):
                lane0, lane1 = self._split_tile(tile, split)
                self._to_aiv_split[split][0].append(lane0)
                self._to_aiv_split[split][1].append(lane1)
            else:
                raise ValueError(f"Unsupported split mode for push_to_aiv: {split}")
            self._cv.notify_all()

    def pop_from_aic(self, split):
        split = int(split)
        lane = _get_subblock_idx()
        with self._cv:
            if split == 0:
                if self._no_split_dual_aiv_dispatch:
                    if lane not in (0, 1):
                        raise ValueError(
                            f"No-split dual-dispatch tpop_from_aic requires lane in {{0,1}}, got {lane}"
                        )
                    queue = self._to_aiv_dual_nosplit[lane]
                    self._wait_for_locked(lambda: len(queue) > 0, "tpop_from_aic", split, lane)
                    return queue.popleft()
                self._wait_for_locked(lambda: len(self._to_aiv) > 0, "tpop_from_aic", split, lane)
                return self._to_aiv.popleft()
            if split in (1, 2):
                if lane not in (0, 1):
                    raise ValueError(f"Split tpop_from_aic requires lane in {{0,1}}, got {lane}")
                queue = self._to_aiv_split[split][lane]
                self._wait_for_locked(lambda: len(queue) > 0, "tpop_from_aic", split, lane)
                return queue.popleft()
            raise ValueError(f"Unsupported split mode for pop_from_aic: {split}")

    def push_to_aic(self, tile, split):
        split = int(split)
        lane = _get_subblock_idx()
        with self._cv:
            if split == 0:
                if self._no_split_dual_aiv_dispatch:
                    if lane not in (0, 1):
                        raise ValueError(
                            f"No-split dual-dispatch tpush_to_aic requires lane in {{0,1}}, got {lane}"
                        )
                    self._to_aic_dual_nosplit[lane].append(_copy_region_attrs(tile, tile.clone()))
                else:
                    self._to_aic.append(_copy_region_attrs(tile, tile.clone()))
            elif split in (1, 2):
                if lane not in (0, 1):
                    raise ValueError(f"Split tpush_to_aic requires lane in {{0,1}}, got {lane}")
                self._to_aic_split[split][lane].append(_copy_region_attrs(tile, tile.clone()))
            else:
                raise ValueError(f"Unsupported split mode for push_to_aic: {split}")
            self._cv.notify_all()

    def pop_from_aiv(self, split):
        split = int(split)
        lane = _get_subblock_idx()
        with self._cv:
            if split == 0:
                if self._no_split_dual_aiv_dispatch:
                    lane0_q = self._to_aic_dual_nosplit[0]
                    lane1_q = self._to_aic_dual_nosplit[1]
                    self._wait_for_locked(
                        lambda: len(lane0_q) > 0 and len(lane1_q) > 0,
                        "tpop_from_aiv",
                        split,
                        lane,
                    )
                    lane0_tile = lane0_q.popleft()
                    lane1_q.popleft()
                    return lane0_tile
                self._wait_for_locked(lambda: len(self._to_aic) > 0, "tpop_from_aiv", split, lane)
                return self._to_aic.popleft()
            if split in (1, 2):
                lane0_q = self._to_aic_split[split][0]
                lane1_q = self._to_aic_split[split][1]
                self._wait_for_locked(
                    lambda: len(lane0_q) > 0 and len(lane1_q) > 0,
                    "tpop_from_aiv",
                    split,
                    lane,
                )
                part0 = lane0_q.popleft()
                part1 = lane1_q.popleft()
                return self._merge_tile(part0, part1, split)
            raise ValueError(f"Unsupported split mode for pop_from_aiv: {split}")


_cross_core_rt = _CrossCoreRuntime()


def _run_mixed_kernels(group_name, meta, *args):
    aic_name = meta["aic"]
    aiv_name = meta["aiv"]
    split = int(meta.get("split", 0))
    dual_aiv_dispatch = bool(meta.get("dual_aiv_dispatch", False))
    num_aiv_lanes = 2 if split in (1, 2) or dual_aiv_dispatch else 1

    _cross_core_rt.reset(no_split_dual_aiv_dispatch=(split == 0 and dual_aiv_dispatch))
    aic_fn = globals().get(aic_name)
    aiv_fn = globals().get(aiv_name)
    if aic_fn is None or aiv_fn is None:
        raise RuntimeError(
            f"Mixed-kernel function lookup failed for {group_name}: "
            f"aic={aic_name!r}, aiv={aiv_name!r}"
        )

    run_cancel = threading.Event()
    _set_runtime_cancel_event(run_cancel)
    failures = []
    failure_lock = threading.Lock()
    results = {}
    results_lock = threading.Lock()

    def _runner(func, func_name, role, lane):
        try:
            _set_subblock_idx(0 if lane is None else lane)
            value = func(*args)
            with results_lock:
                results[(role, lane)] = value
        except Exception:
            if run_cancel.is_set():
                return
            with failure_lock:
                failures.append((func_name, lane, traceback.format_exc()))
        finally:
            _set_subblock_idx(0)

    try:
        threads = [
            threading.Thread(
                target=_runner,
                args=(aic_fn, aic_name, "aic", None),
                name=f"{group_name}-aic",
            )
        ]
        for lane in range(num_aiv_lanes):
            threads.append(
                threading.Thread(
                    target=_runner,
                    args=(aiv_fn, aiv_name, "aiv", lane),
                    name=f"{group_name}-aiv-lane{lane}",
                )
            )

        for t in threads:
            t.start()

        deadline = time.monotonic() + _MIXED_KERNEL_TIMEOUT_SEC
        for t in threads:
            remaining = max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)

        alive = [t.name for t in threads if t.is_alive()]
        timed_out = len(alive) > 0

        if timed_out or failures:
            run_cancel.set()
            _cross_core_rt.notify_all()
            for t in threads:
                if t.is_alive():
                    t.join(timeout=0.2)

        if timed_out:
            alive_after_cancel = [t.name for t in threads if t.is_alive()]
            if alive_after_cancel:
                alive = alive_after_cancel
            raise RuntimeError(
                f"Mixed-kernel execution timeout for {group_name}; "
                f"alive_threads={alive}; pipe_state={_cross_core_rt.snapshot()}"
            )
        if failures:
            fn, lane, tb = failures[0]
            raise RuntimeError(
                f"Mixed-kernel execution failed in {fn} (lane={lane}) for {group_name}\\n{tb}"
            )

        if ("aiv", 0) in results:
            return results[("aiv", 0)]
        non_none_keys = [f"{role}:{lane}" for (role, lane), value in results.items() if value is not None]
        if non_none_keys:
            raise RuntimeError(
                f"Mixed-kernel return contract violation for {group_name}; "
                f"expected aiv lane 0, got non-None returns from {non_none_keys}"
            )
        return None
    finally:
        _set_runtime_cancel_event(None)
        _set_subblock_idx(0)


def _run_group_call(group_name, *args):
    meta = _GROUP_META.get(group_name)
    if meta is None:
        return globals()[group_name](*args)
    return _run_mixed_kernels(group_name, meta, *args)

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


def _split_mode_to_int(split_mode: Any) -> int:
    if split_mode is None:
        return 0
    if isinstance(split_mode, int):
        return int(split_mode)
    value = getattr(split_mode, "value", None)
    if value is not None:
        return int(value)
    try:
        return int(split_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid split mode: {split_mode!r}") from exc


def _split_kwarg(kw: dict[str, Any]) -> int:
    return _split_mode_to_int(kw.get("split", 0))


def _handle_tpush_to_aiv(a: list[str], kw: dict[str, Any]) -> str:
    return f"_cross_core_rt.push_to_aiv({a[0]}, {_split_kwarg(kw)})"


def _handle_tpush_to_aic(a: list[str], kw: dict[str, Any]) -> str:
    return f"_cross_core_rt.push_to_aic({a[0]}, {_split_kwarg(kw)})"


def _handle_tpop_from_aic(_a: list[str], kw: dict[str, Any]) -> str:
    return f"_cross_core_rt.pop_from_aic({_split_kwarg(kw)})"


def _handle_tpop_from_aiv(_a: list[str], kw: dict[str, Any]) -> str:
    return f"_cross_core_rt.pop_from_aiv({_split_kwarg(kw)})"


_CROSS_CORE_SPLIT_OPS = {
    "tile.tpush_to_aiv",
    "tile.tpush_to_aic",
    "tile.tpop_from_aic",
    "tile.tpop_from_aiv",
}


def _collect_cross_core_split_from_expr(expr: _ir.Expr, out: set[int]) -> None:
    if isinstance(expr, _ir.Call):
        op_name = expr.op.name
        if op_name in _CROSS_CORE_SPLIT_OPS:
            kw = dict(expr.kwargs) if expr.kwargs else {}
            out.add(_split_kwarg(kw))
        for arg in expr.args:
            _collect_cross_core_split_from_expr(arg, out)
        return

    if isinstance(expr, _ir.MakeTuple):
        for e in expr.elements:
            _collect_cross_core_split_from_expr(e, out)
        return

    if isinstance(expr, _ir.TupleGetItemExpr):
        _collect_cross_core_split_from_expr(expr.tuple, out)
        return

    if isinstance(expr, _ir.BinaryExpr):
        _collect_cross_core_split_from_expr(expr.left, out)
        _collect_cross_core_split_from_expr(expr.right, out)
        return

    if isinstance(expr, _ir.UnaryExpr):
        _collect_cross_core_split_from_expr(expr.operand, out)
        return


def _collect_cross_core_split_from_stmt(stmt: _ir.Stmt, out: set[int]) -> None:
    if isinstance(stmt, _ir.SeqStmts):
        for s in stmt.stmts:
            _collect_cross_core_split_from_stmt(s, out)
        return

    if isinstance(stmt, (_ir.AssignStmt, _ir.EvalStmt)):
        expr = stmt.value if isinstance(stmt, _ir.AssignStmt) else stmt.expr
        _collect_cross_core_split_from_expr(expr, out)
        return

    if isinstance(stmt, (_ir.ReturnStmt, _ir.YieldStmt)):
        values = stmt.value if stmt.value is not None else []
        for v in values:
            _collect_cross_core_split_from_expr(v, out)
        return

    if isinstance(stmt, (_ir.ForStmt, _ir.WhileStmt)):
        if isinstance(stmt, _ir.ForStmt):
            _collect_cross_core_split_from_expr(stmt.start, out)
            _collect_cross_core_split_from_expr(stmt.stop, out)
            _collect_cross_core_split_from_expr(stmt.step, out)
        else:
            _collect_cross_core_split_from_expr(stmt.condition, out)

        iter_args = getattr(stmt, "iter_args", [])
        for ia in iter_args:
            if ia.initValue is not None:
                _collect_cross_core_split_from_expr(ia.initValue, out)

        _collect_cross_core_split_from_stmt(stmt.body, out)
        return

    if isinstance(stmt, _ir.IfStmt):
        _collect_cross_core_split_from_expr(stmt.condition, out)
        _collect_cross_core_split_from_stmt(stmt.then_body, out)
        if stmt.else_body is not None:
            _collect_cross_core_split_from_stmt(stmt.else_body, out)
        return

    if isinstance(stmt, _ir.ScopeStmt):
        _collect_cross_core_split_from_stmt(stmt.body, out)
        return


def _collect_cross_core_splits(func: _ir.Function) -> set[int]:
    splits: set[int] = set()
    if func.body is not None:
        _collect_cross_core_split_from_stmt(func.body, splits)
    return splits


def _resolve_group_split_from_cross_core_ops(
    group_name: str,
    aic_func: _ir.Function,
    aiv_func: _ir.Function,
) -> int | None:
    split_values = _collect_cross_core_splits(aic_func) | _collect_cross_core_splits(aiv_func)
    if not split_values:
        return None
    if len(split_values) > 1:
        raise ValueError(f"Conflicting cross-core split kwargs in group {group_name}: {sorted(split_values)}")
    return next(iter(split_values))


def _build_group_meta(program: _ir.Program) -> dict[str, dict[str, Any]]:
    funcs_by_name = {func.name: func for func in program.functions.values()}
    group_meta: dict[str, dict[str, Any]] = {}

    for func in program.functions.values():
        if func.func_type != _ir.FunctionType.Group:
            continue

        aic_name = f"{func.name}_aic"
        aiv_name = f"{func.name}_aiv"
        if aic_name not in funcs_by_name or aiv_name not in funcs_by_name:
            continue

        aic_func = funcs_by_name[aic_name]
        aiv_func = funcs_by_name[aiv_name]

        split_from_ops = _resolve_group_split_from_cross_core_ops(func.name, aic_func, aiv_func)
        if split_from_ops is not None:
            split = split_from_ops
        else:
            split = _split_mode_to_int(func.split)
            if split == 0:
                split = _split_mode_to_int(aiv_func.split)
        dual_aiv_dispatch = bool(getattr(aiv_func, "attrs", {}).get("dual_aiv_dispatch", False))
        group_meta[func.name] = {
            "aic": aic_name,
            "aiv": aiv_name,
            "split": split,
            "dual_aiv_dispatch": dual_aiv_dispatch,
        }

    return group_meta


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
        m[f"{prefix}.maximums"] = _torch_fn("maximum", 2)
        m[f"{prefix}.minimums"] = _torch_fn("minimum", 2)
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
    m["tile.batch_matmul_acc"] = lambda a, _kw: f"({a[0]} + torch.matmul({a[1]}, {a[2]}).float())"
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
    m["tile.tpush_to_aiv"] = _handle_tpush_to_aiv
    m["tile.tpush_to_aic"] = _handle_tpush_to_aic
    m["tile.tpop_from_aic"] = _handle_tpop_from_aic
    m["tile.tpop_from_aiv"] = _handle_tpop_from_aiv
    m["tile.get_subblock_idx"] = lambda _a, _kw: "_get_subblock_idx()"

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

    def __init__(
        self, *, check_shapes: bool = False, group_meta: dict[str, dict[str, Any]] | None = None
    ) -> None:
        super().__init__()
        self._lines: list[str] = []
        self._indent: int = 0
        self._expr_result: str = ""
        # Fast-path name cache keyed by Python wrapper id.
        self._var_names_by_id: dict[int, str] = {}
        # Stable name cache keyed by semantic variable identity.
        self._var_names_by_key: dict[tuple[Any, ...], str] = {}
        # Keep wrapper refs alive during one function emission to avoid Python
        # reusing object ids for short-lived nanobind wrappers.
        self._seen_var_refs: list[_ir.Var] = []
        self._name_counter: dict[str, int] = {}
        self._yield_targets: list[str] = []  # names to assign on yield
        self._check_shapes: bool = check_shapes
        self._group_meta: dict[str, dict[str, Any]] = group_meta or {}

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
        key = self._var_semantic_key(var)
        vid = id(var)
        if vid in self._var_names_by_id:
            return self._var_names_by_id[vid]

        if key in self._var_names_by_key:
            name = self._var_names_by_key[key]
        else:
            name = self._unique_name(var.name_hint)
            self._var_names_by_key[key] = name

        self._var_names_by_id[vid] = name
        self._seen_var_refs.append(var)
        return name

    @staticmethod
    def _var_semantic_key(var: _ir.Var) -> tuple[Any, ...]:
        """Build a stable key for nanobind-backed IR vars.

        IR callbacks may wrap the same underlying C++ Var with different Python
        objects, so ``id(var)`` alone is not stable across visits. We key by
        semantic fields that remain stable across wrappers.
        """
        span = getattr(var, "span", None)
        span_key: tuple[Any, ...] | None = None
        if span is not None and getattr(span, "is_valid", False):
            span_key = (
                getattr(span, "filename", ""),
                int(getattr(span, "begin_line", 0)),
                int(getattr(span, "begin_column", 0)),
                int(getattr(span, "end_line", 0)),
                int(getattr(span, "end_column", 0)),
            )

        var_type = getattr(var, "type", None)
        return (
            type(var).__name__,
            getattr(var, "name_hint", ""),
            span_key,
            str(var_type) if var_type is not None else "",
        )

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
            self._var_names_by_id[id(rv)] = name
            self._var_names_by_key[self._var_semantic_key(rv)] = name
            self._seen_var_refs.append(rv)

    # -- top-level --

    def visit_program(self, program: _ir.Program) -> None:
        for _gv, func in program.functions.items():
            self.visit_function(func)

    def visit_function(self, func: _ir.Function) -> None:
        # Keep names function-local. IR may reuse object ids across functions;
        # sharing maps at program scope can emit stale names.
        self._var_names_by_id = {}
        self._var_names_by_key = {}
        self._seen_var_refs = []
        self._name_counter = {}
        self._yield_targets = []

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
            if op_name in self._group_meta:
                args_str = ", ".join(arg_strs)
                if args_str:
                    self._expr_result = f"_run_group_call('{op_name}', {args_str})"
                else:
                    self._expr_result = f"_run_group_call('{op_name}')"
            else:
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


def _select_entry_function(node: _ir.Program | _ir.Function) -> _ir.Function:
    """Choose the callable entry function for runtime validation."""
    if isinstance(node, _ir.Function):
        return node

    funcs = list(node.functions.values())
    if not funcs:
        raise ValueError(f"Program {node.name!r} has no functions")
    if len(funcs) == 1:
        return funcs[0]

    for func in funcs:
        if func.func_type == _ir.FunctionType.Orchestration:
            return func
    raise ValueError(f"Program {node.name!r} has {len(funcs)} functions but no Orchestration entry")


def _resolve_tensor_from_dict(param_name: str, tensors: dict[str, Any]) -> Any:
    """Resolve a parameter name against tensor dict keys with SSA fallback."""
    if param_name in tensors:
        return tensors[param_name]

    base_name = _SSA_SUFFIX_RE.sub("", param_name)
    if base_name in tensors:
        return tensors[base_name]

    raise KeyError(
        f"Cannot resolve tensor for parameter {param_name!r}. Available keys: {sorted(tensors.keys())}"
    )


def _validate_tensor_dict_arg(name: str, value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be dict[str, Tensor], got {type(value).__name__}")

    for key, tensor in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} key must be str, got {type(key).__name__}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}[{key!r}] must be torch.Tensor, got {type(tensor).__name__}")
    return value


def _validate_non_negative_float_arg(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative float, got {value!r}")
    casted = float(value)
    if casted < 0:
        raise ValueError(f"{name} must be a non-negative float, got {value!r}")
    return casted


def _collect_pass_ir_files(ir_dir: str) -> list[Path]:
    ir_path = Path(ir_dir)
    if not ir_path.exists():
        raise FileNotFoundError(f"IR path does not exist: {ir_dir}")

    if ir_path.is_file():
        if ir_path.suffix != ".py":
            raise ValueError(f"IR file must be a .py file, got: {ir_path}")
        return [ir_path]

    ir_files = sorted(p for p in ir_path.iterdir() if p.is_file() and p.suffix == ".py")
    if not ir_files:
        raise ValueError(f"No .py IR files found under: {ir_dir}")
    return ir_files


def _build_codegen_entry(
    ir_file: Path,
) -> tuple[_ir.Function, Callable[..., Any]]:
    parsed = pl.loads(str(ir_file))
    if not isinstance(parsed, (_ir.Program, _ir.Function)):
        raise ValueError(
            f"Parsed object from {ir_file} must be Program/Function, got {type(parsed).__name__}"
        )

    entry_func = _select_entry_function(parsed)
    code = torch_codegen(parsed, check_shapes=True)
    namespace: dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102

    entry = namespace.get(entry_func.name)
    if not callable(entry):
        raise ValueError(f"Generated code from {ir_file} has no callable entry {entry_func.name!r}")
    return entry_func, entry


def _compare_expected_tensors(
    ir_file: Path,
    run_tensors: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> None:
    pass_name = ir_file.stem
    print("=" * 20, f"{pass_name}", "=" * 20)
    for key, exp_tensor in expected.items():
        actual = run_tensors.get(key)
        if actual is None:
            raise ValueError(
                f"{ir_file}: expected key {key!r} not found in tensors after execution; "
                f"available keys: {sorted(run_tensors.keys())}"
            )
        if not isinstance(actual, torch.Tensor):
            raise TypeError(
                f"{ir_file}: tensors[{key!r}] must be torch.Tensor after execution, "
                f"got {type(actual).__name__}"
            )
        ok = torch.allclose(actual, exp_tensor, rtol=rtol, atol=atol)
        diff = float((actual - exp_tensor).abs().max().item())
        print(f"validate tensor: {key!r}, max_abs_diff: {diff:.6e}, pass: {ok}")
        if not ok:
            raise AssertionError(
                f"{ir_file}: tensor mismatch for key {key!r}, "
                f"max abs diff {diff:.6e} exceeds tolerance "
                f"(rtol={rtol:.3e}, atol={atol:.3e})"
            )


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
    group_meta: dict[str, dict[str, Any]] = {}
    if isinstance(node, _ir.Program):
        group_meta = _build_group_meta(node)

    cg = TorchCodegen(check_shapes=check_shapes, group_meta=group_meta)
    lines = [_PREAMBLE]
    if group_meta:
        lines.append(f"_GROUP_META.update({group_meta!r})")
        lines.append("")

    if isinstance(node, _ir.Program):
        cg.visit_program(node)
    elif isinstance(node, _ir.Function):
        cg.visit_function(node)
    else:
        raise TypeError(f"torch_codegen expects Program or Function, got {type(node).__name__}")

    lines.append(cg.get_output())
    return "\n".join(lines)


def validate_pass_ir_codegen_results(
    ir_dir: str,
    tensors: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    rtol: float = 5e-2,
    atol: float = 5e-2,
) -> None:
    """Validate torch_codegen correctness on each pass IR file in a directory.

    Args:
        ir_dir: Directory that contains IR Python files (for example,
            PassManager dump files like ``00_frontend.py``, ``01_after_xxx.py``).
            A single ``.py`` file path is also accepted.
        tensors: Input tensors for executing generated functions, keyed by
            function parameter name.
        expected: Expected tensors keyed by tensor name.
        rtol: Relative tolerance for ``torch.allclose``.
        atol: Absolute tolerance for ``torch.allclose``.

    Prints:
        Per-pass validation summary with max absolute differences.

    Raises:
        FileNotFoundError: If ``ir_dir`` does not exist.
        ValueError: If no IR files found, parse result invalid, output missing,
            or tolerance arguments are invalid.
        TypeError: If ``tensors``/``expected`` are not
            ``dict[str, torch.Tensor]``.
        RuntimeError: If generated code execution fails.
        AssertionError: If numeric comparison fails.
    """
    validated_tensors = _validate_tensor_dict_arg("tensors", tensors)
    validated_expected = _validate_tensor_dict_arg("expected", expected)
    validated_rtol = _validate_non_negative_float_arg("rtol", rtol)
    validated_atol = _validate_non_negative_float_arg("atol", atol)
    ir_files = _collect_pass_ir_files(ir_dir)

    for ir_file in ir_files:
        entry_func, entry = _build_codegen_entry(ir_file)
        run_tensors = {k: v.clone() for k, v in validated_tensors.items()}
        args = [_resolve_tensor_from_dict(param.name_hint, run_tensors) for param in entry_func.params]
        entry(*args)
        _compare_expected_tensors(
            ir_file,
            run_tensors,
            validated_expected,
            rtol=validated_rtol,
            atol=validated_atol,
        )
