# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Portable helpers shared by parent orchestration and embedded HOST entries.

This source can live in an artifact's relative package. It must never import
PyPTO, compiler modules or a parent runner; optional Simpler imports stay lazy.
"""

import ctypes
import json
import math
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import torch

# Shared with the parent DFX reader; one filename describes each L2 program.
_DISPATCH_PROGRAM_FILE = "dispatch_program.json"


def _tensor_from_continuous(arg: Any) -> torch.Tensor:
    """View an actual mapped SUB argument (or legacy pointer) without copying.

    MappedArg.buffer already begins at its descriptor's byte offset. Keeping
    this converter in the canonical source lets generated HOST SubWorkers run
    in an installed package without importing the parent compiler/runtime.
    """
    dtype_name = str(arg.dtype).rsplit(".", 1)[-1]
    if dtype_name.isdecimal():
        from simpler.task_interface import DataType  # noqa: PLC0415

        dtype_name = DataType(int(dtype_name)).name
    dtypes = {
        "FLOAT32": torch.float32,
        "FLOAT16": torch.float16,
        "BFLOAT16": torch.bfloat16,
        "INT8": torch.int8,
        "INT16": torch.int16,
        "INT32": torch.int32,
        "INT64": torch.int64,
        "UINT8": torch.uint8,
        # Simpler FP4 descriptors count packed x2 byte carriers.
        "FP4E2M1": torch.uint8,
    }
    if dtype_name not in dtypes:
        raise TypeError(f"Unsupported simpler Tensor dtype: {arg.dtype!r}; known dtypes: {sorted(dtypes)}")
    dtype = dtypes[dtype_name]
    shapes = tuple(int(extent) for extent in arg.shapes)
    if hasattr(arg, "buffer"):
        strides = tuple(int(stride) for stride in arg.strides)
        span = 1 + sum((extent - 1) * stride for extent, stride in zip(shapes, strides, strict=True))
        return torch.as_strided(torch.frombuffer(arg.buffer, dtype=dtype, count=span), shapes, strides)
    if not hasattr(arg, "data"):
        raise TypeError(
            f"Expected a simpler MappedArg with .buffer or legacy tensor with .data, got {type(arg).__name__}"
        )
    count = math.prod(shapes)
    size = count * torch.tensor([], dtype=dtype).element_size()
    # Legacy callers supply either integer addresses or ctypes pointers;
    # normalize both forms before from_address, which accepts integers only.
    address = ctypes.cast(arg.data, ctypes.c_void_p).value
    if address is None:
        raise TypeError("Legacy tensor .data must contain a non-null pointer")
    backing = (ctypes.c_ubyte * size).from_address(address)
    return torch.frombuffer(backing, dtype=dtype, count=count).reshape(shapes)


class _ChipCallables(dict[str, Any]):
    """Chip callables (or registered handles) with their tensor transfer contracts."""

    def __init__(self) -> None:
        super().__init__()
        self.host_tensor_indices: dict[str, frozenset[int]] = {}

    def add(self, name: str, value: Any, config: dict[str, Any]) -> None:
        self[name] = value
        self.host_tensor_indices[name] = frozenset(config.get("host_tensor_indices", ()))


def make_tensor_arg(worker: Any, tensor: Any) -> Any:
    """Keep received descriptors intact; convert only local prefork tensors."""
    from simpler.buffer import Tensor  # noqa: PLC0415

    if isinstance(tensor, Tensor):
        return tensor
    from simpler_setup.torch_interop import make_tensor_arg as convert  # noqa: PLC0415

    return convert(worker, tensor)


def add_chip_tensor_arg(
    args: Any,
    worker: Any,
    tensor: Any,
    tag: Any,
    callables: dict[str, Any],
    name: str,
    index: int,
    *,
    tensor_converter: Callable[[Any, Any], Any] | None = None,
) -> None:
    """Pack a chip argument using the callee's emitted host-control contract."""
    convert = tensor_converter or make_tensor_arg
    host_slots = callables.host_tensor_indices.get(name, ()) if isinstance(callables, _ChipCallables) else ()
    if index in host_slots:
        from simpler.buffer import TensorTransfer  # noqa: PLC0415

        is_host = isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
        if not is_host:
            # Received controls already carry their actual backing provenance.
            # Inspect the descriptor's address space; never replace it with a
            # fabricated fork-inherited backing or treat DEVICE as host memory.
            try:
                from simpler.buffer import AddressSpace, Tensor  # noqa: PLC0415
            except ImportError:
                is_host = False
            else:
                is_host = isinstance(tensor, Tensor) and tensor.buffer.address_space == AddressSpace.HOST
        if not is_host:
            raise ValueError(
                f"Chip {name!r} host control tensor slot {index} requires a CPU tensor or a HOST Tensor"
            )
        args.add_tensor(convert(worker, tensor), tag, transfer=TensorTransfer.NONE)
    else:
        args.add_tensor(convert(worker, tensor), tag)


def _dfx_rank_label(worker: int) -> str:
    """Directory name namespacing one card's DFX artifacts.

    Every dispatch names a concrete chip by the time it reaches here (see
    :func:`_resolve_chip_worker`), so the label is always the chip's own rank
    and matches :data:`_RANK_DIR_GLOB`.
    """
    return f"rank{worker}"


def _resolve_chip_worker(orch: Any, worker: int | None) -> int:
    """Pick the chip a dispatch runs on.

    A ``device=``-pinned dispatch arrives with its rank and is returned as is.
    A comm-less dispatch arrives as ``None``: it expresses no affinity, but the
    runtime requires an exact target — simpler #1436 made ``worker`` a required,
    non-negative NEXT_LEVEL id and removed the "unconstrained" mode where the
    scheduler picked an idle worker itself. Those dispatches are handed out
    round-robin over the program's chips in submit order, so a host_orch with
    one comm-less dispatch per chip still spreads across them (and does so
    deterministically, unlike the old idle-pool pick).

    The chip count is stamped on ``orch`` by ``_dispatch.orch_fn``; without it
    (a caller that bypassed ``orch_fn``) every comm-less dispatch falls back to
    chip 0, which always exists.
    """
    if worker is not None:
        return worker
    chip_count = max(1, int(getattr(orch, "_pypto_chip_count", 1)))
    seq = getattr(orch, "_pypto_commless_seq", None)
    if seq is None:
        seq = 0
    orch._pypto_commless_seq = seq + 1
    return seq % chip_count


def _reset_dfx_dispatch_state(
    orch: Any,
    chip_cids: dict[str, Any],
    *,
    dispatch_idx: dict[str, int] | None = None,
) -> None:
    """Install the dispatch state :func:`_submit_chip` reads off ``orch``.

    ``_dfx_dispatch_idx`` numbers each card's dispatches ``d0, d1, ...`` and must
    mirror the ChipWorker child's ``local_capture_index``, which spans the
    *process*. Callers on a long-lived child (prepared and persistent workers)
    therefore pass the worker-owned dict here so the numbering continues across
    runs; the one-shot path, which forks a fresh child per pass, passes nothing
    and gets a fresh counter. Reset when the child resets.

    ``_dfx_chip_names`` reverses the registered ``callables`` mapping so
    ``_submit_chip`` can name the L2 program behind an otherwise opaque
    ``CallableHandle``. Keyed by ``id()``: the handle is not required to be
    hashable, and ``chip_cids`` holds every one alive for the whole run, so the
    ids are stable and unique.
    """
    orch._dfx_dispatch_idx = {} if dispatch_idx is None else dispatch_idx
    orch._dfx_chip_names = {id(cid): name for name, cid in chip_cids.items()}


def _record_dispatch_program(orch: Any, callable_id: Any, disp_dir: Path) -> None:
    """Record which L2 program a dispatch runs, for the swimlane post-pass.

    A kernel's ``func_id`` is a per-L2-program namespace — every
    ``next_levels/<program>/kernel_config.py`` numbers its kernels from 0 — so
    labelling a dispatch's records requires knowing the program that produced
    them. This wrapper is the only place that sees both the dispatch directory
    and the callable being dispatched, so it stamps the pairing on disk
    (:data:`_DISPATCH_PROGRAM_FILE`) for :func:`_collect_l3_swimlane` to read
    back. Going through the filesystem keeps the two halves independent of where
    the runtime places the L3 orchestrator, and the marker is rewritten
    identically by both swimlane passes.

    Best-effort: a marker that cannot be written costs kernel labels, never the
    run. Silent when the caller bypassed :func:`_reset_dfx_dispatch_state` and
    left no name table on ``orch``.
    """
    name = getattr(orch, "_dfx_chip_names", {}).get(id(callable_id))
    if name is None:
        return
    try:
        disp_dir.mkdir(parents=True, exist_ok=True)
        (disp_dir / _DISPATCH_PROGRAM_FILE).write_text(json.dumps({"program": name}), encoding="utf-8")
    except OSError as e:
        print(
            f"Could not record the dispatch program for {disp_dir} ({type(e).__name__}: {e}); "
            "its swimlane falls back to anonymous task labels unless the build has a single "
            "L2 program"
        )


# Called from generated host_orch.py code (pyright cannot see those references).
def _submit_chip(  # pyright: ignore[reportUnusedFunction]
    orch: Any, callable_id: Any, task_args: Any, config: Any, worker: int | None
) -> Any:
    """``orch.submit_next_level`` that stamps each dispatch's DFX directory.

    The runtime path helpers root every diagnostic artifact at a fixed filename
    under ``output_prefix`` (``<prefix>/pmu.csv`` etc.), so any two dispatches
    sharing one prefix clobber each other. Namespacing by card alone is not
    enough: one card may receive several dispatches in a single host_orch run
    (pipeline stages, expert kernels, or genuinely different chip programs all
    pinned to the same ``device``), and each re-init+finalize of the runtime's
    per-run collector rewrites the fixed-name file.

    The ChipWorker child applies that ``rank{worker}/d{k}`` split itself, so
    this wrapper only derives the same directory — for the marker below and the
    offline post-pass — and forwards ``config`` untouched.

    ``k`` comes from a per-card counter on ``orch`` that mirrors the child's own
    ``local_capture_index``. Both are process-scoped: a prepared or persistent
    worker carries the counter across runs, while a one-shot worker — which
    forks a fresh child for each pass — starts it over. The marker
    :func:`_record_dispatch_program` stamps therefore lands in the same
    ``rank{w}/d{k}`` the child files that capture under, on the graph pass of a
    swimlane two-pass; :func:`_collect_l3_swimlane` still resolves it through the
    capture pairing.

    When DFX is off (``output_prefix`` unset) the call is forwarded unchanged.

    The codegen routes every chip dispatch through this wrapper — a rank-pinned
    dispatch passes its rank, a comm-less one passes ``None`` and is resolved by
    :func:`_resolve_chip_worker`. Resolution happens before the DFX namespacing
    so the artifacts land under the chip that actually ran the dispatch.
    """
    worker = _resolve_chip_worker(orch, worker)
    base = config.output_prefix
    if not base:
        return orch.submit_next_level(callable_id, task_args, config, worker=worker)
    idx_map = getattr(orch, "_dfx_dispatch_idx", None)
    if idx_map is None:
        # Defensive: a caller that bypassed ``orch_fn`` (no reset) still gets
        # per-card isolation, just without a guaranteed two-pass match.
        idx_map = orch._dfx_dispatch_idx = cast("dict[str, int]", {})
    rank_label = _dfx_rank_label(worker)
    k = idx_map.get(rank_label, 0)
    _record_dispatch_program(orch, callable_id, Path(f"{base}/{rank_label}/d{k}"))
    handle = orch.submit_next_level(callable_id, task_args, config, worker=worker)
    # Advance only once the submit is accepted. A synchronous rejection creates
    # no capture, so counting it would leave every later marker one directory
    # ahead of the child's own numbering -- self-correcting when the counter was
    # reset per run, permanent now that a prepared worker carries it across
    # runs. The marker already at ``d{k}`` is overwritten by the next submit.
    idx_map[rank_label] = k + 1
    return handle


# ctypes retains the sender's explicit scalar width. In particular, Simpler
# zero-extends c_int8(-1) to 0xff rather than sign-extending it to 64 bits, and
# accepts an ordinary Python float as FP32. A double must travel as c_double.
_SCALAR_CTYPES: dict[str, Any] = {
    "bool": ctypes.c_bool,
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "index": ctypes.c_int64,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "fp32": ctypes.c_float,
    "fp64": ctypes.c_double,
}


def _encode_scalar(value: int | float | bool, dtype: str) -> Any:
    """Preserve an explicit scalar dtype through Simpler's uint64 encoder.

    FP64 is supported by this neutral wire helper; the current PyPTO type
    surface does not expose an FP64 formal.
    """
    constructor = _SCALAR_CTYPES.get(dtype)
    if constructor is None:
        raise ValueError(f"Unsupported embedded scalar dtype {dtype!r}")
    return constructor(value)


def _decode_scalar(raw: int, dtype: str) -> int | float | bool:
    """Decode a real uint64 slot using its declared width and representation."""
    if not isinstance(raw, int) or raw < 0 or raw >= 1 << 64:
        raise ValueError(f"Expected a uint64 scalar slot, got {raw!r}")
    if dtype == "fp32":
        return struct.unpack("<f", struct.pack("<I", raw & 0xFFFFFFFF))[0]
    if dtype == "fp64":
        return struct.unpack("<d", struct.pack("<Q", raw))[0]
    constructor = _SCALAR_CTYPES.get(dtype)
    if constructor is None:
        raise ValueError(f"Unsupported embedded scalar dtype {dtype!r}")
    # ctypes performs width truncation and sign recovery for integer types;
    # c_bool additionally returns an actual bool rather than an integer flag.
    return constructor(raw).value


class _EmbeddedFrame:
    """Bind one L3 invocation and reuse a token for each physical domain and worker."""

    def __init__(self, tensors: dict[str, Any], orch: Any) -> None:
        self.tensors = tensors
        self.orch = orch
        self.implicit_outputs: dict[int, Any] = {}
        self.return_copy_callable: Any = None
        self.tokens: dict[tuple[Any, ...], Any] = {}
        self.local_workers: dict[str, tuple[int, ...]] = {}
        self.window_slots: dict[tuple[int, int, str], str] = {}

    def window_tensor(self, domain: Any, worker: int, logical_name: str, **view: Any) -> Any:
        """Translate a slot identity, then use the actual runtime Buffer object."""
        physical = logical_name
        if _ordering_domain_key(domain)[0] == "global":
            physical = self.window_slots.get(
                (domain.domain_id, domain.generation, logical_name), logical_name
            )
        return _domain_context(domain, worker).buffers[physical].tensor(**view)

    @staticmethod
    def require_same_domain(domains: tuple[Any, ...]) -> None:
        if len({_ordering_domain_key(domain) for domain in domains}) > 1:
            raise ValueError("Builtin window arguments require the same physical domain")

    def bind_local_domain(self, domain: Any, key: str, window_names: tuple[str, ...]) -> None:
        workers = self.local_workers[key]
        for worker in workers:
            context = _domain_context(domain, worker)
            for name in window_names:
                if name not in context.buffers:
                    raise ValueError(f"Local domain is missing named window {name!r} for worker {worker}")

    def ordering_tokens(self, domains: tuple[Any, ...], worker: int) -> tuple[Any, ...]:
        tokens, seen = [], set()
        for domain in domains:
            key = (*_ordering_domain_key(domain), worker)
            if key in seen:
                continue
            seen.add(key)
            _domain_context(domain, worker)
            if key not in self.tokens:
                self.tokens[key] = _alloc_invocation_tensor(self.orch, "int32", (1,))
            tokens.append(self.tokens[key])
        return tuple(tokens)


def _ordering_domain_key(domain: Any) -> tuple[Any, ...]:
    # Owning local handles also carry domain IDs in the unified runtime API.
    if hasattr(domain, "allocation_id"):
        return ("local", domain.allocation_id)
    if hasattr(domain, "domain_id"):
        return ("global", domain.domain_id, domain.generation)
    return ("local", domain.allocation_id)


def _domain_context(domain: Any, worker: int) -> Any:
    if _ordering_domain_key(domain)[0] == "global" and not domain.committed:
        raise ValueError(f"Global domain {domain.domain_id} is not committed")
    try:
        context = domain[worker]
    except KeyError as error:
        raise ValueError(f"Domain is missing local worker context {worker}") from error
    if (
        not isinstance(context.device_ctx, int)
        or context.device_ctx <= 0
        or not isinstance(context.domain_size, int)
        or context.domain_size <= 0
        or not 0 <= context.domain_rank < context.domain_size
    ):
        raise ValueError(f"Domain worker {worker} has an invalid device context or domain size/rank")
    return context


def _tensor_dtype_code(dtype: Any) -> int:
    """MappedArgs use integer codes; native DataType enumerators expose .value."""
    code = getattr(dtype, "value", dtype)
    if not isinstance(code, int) or isinstance(code, bool):
        raise ValueError(f"Unsupported tensor dtype {dtype!r}")
    return code


def _descriptor_itemsize(dtype: Any) -> int:
    from simpler.task_interface import DataType  # noqa: PLC0415

    code = _tensor_dtype_code(dtype)
    for names, size in (
        (("BOOL", "INT8", "UINT8"), 1),
        (("INT16", "UINT16", "FLOAT16", "BFLOAT16"), 2),
        (("INT32", "UINT32", "FLOAT32"), 4),
        (("INT64", "UINT64", "FLOAT64"), 8),
    ):
        if any(getattr(getattr(DataType, name, None), "value", None) == code for name in names):
            return size
    raise ValueError(f"Unsupported descriptor view dtype {dtype!r}")


def _slice_tensor(
    value: Any,
    shape: tuple[int, ...],
    offsets: tuple[int, ...],
    drop_dims: tuple[int, ...],
    valid_shape: tuple[int, ...] | None,
) -> Any:
    """Compose a logical slice over an existing tensor's physical descriptor."""
    is_torch = isinstance(value, torch.Tensor)
    source_shape = tuple(value.shape if is_torch else value.shapes)
    shape, offsets, drop_dims = tuple(shape), tuple(offsets), tuple(drop_dims)
    if (
        len(shape) != len(source_shape)
        or len(offsets) != len(shape)
        or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (*shape, *offsets))
        or any(offset + extent > bound for offset, extent, bound in zip(offsets, shape, source_shape))
        or len(set(drop_dims)) != len(drop_dims)
        or any(axis < 0 or axis >= len(shape) or shape[axis] != 1 for axis in drop_dims)
    ):
        raise ValueError(
            "Embedded descriptor slice requires valid in-bounds extents, offsets and unit dropped axes"
        )
    if valid_shape and tuple(valid_shape) != shape:
        raise ValueError("Embedded descriptor slice with a partial valid shape is not supported")
    if is_torch:
        return value[
            tuple(o if i in drop_dims else slice(o, o + s) for i, (o, s) in enumerate(zip(offsets, shape)))
        ]
    from simpler.buffer import Tensor  # noqa: PLC0415

    if not isinstance(value, Tensor):
        raise ValueError("Embedded descriptor slice requires a native Tensor or CPU Torch tensor")
    strides = tuple(value.strides)
    byte_offset = value.byte_offset + sum(o * s for o, s in zip(offsets, strides)) * _descriptor_itemsize(
        value.dtype
    )
    return Tensor(
        value.buffer,
        shapes=tuple(s for i, s in enumerate(shape) if i not in drop_dims),
        dtype=value.dtype,
        strides=tuple(s for i, s in enumerate(strides) if i not in drop_dims),
        byte_offset=byte_offset,
    )


def _reshape_tensor(value: Any, shape: tuple[int, ...]) -> Any:
    """Create only a contiguous, product-preserving view; never copy storage."""
    shape = tuple(shape)
    is_torch = isinstance(value, torch.Tensor)
    source_shape = tuple(value.shape if is_torch else value.shapes)
    strides = tuple(value.stride() if is_torch else value.strides)
    if not shape or any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
        raise ValueError("Embedded descriptor shape view requires nonnegative integer dimensions")
    if math.prod(shape) != math.prod(source_shape):
        raise ValueError("Embedded descriptor shape view must preserve element count")
    expected = 1
    for extent, stride in reversed(tuple(zip(source_shape, strides))):
        if extent > 1 and stride != expected:
            raise ValueError("Embedded descriptor shape view requires contiguous element strides")
        expected *= extent
    if is_torch:
        return value.view(shape)
    from simpler.buffer import Tensor  # noqa: PLC0415

    if not isinstance(value, Tensor):
        raise ValueError("Embedded descriptor shape view requires a native Tensor or CPU Torch tensor")
    return Tensor(value.buffer, shapes=shape, dtype=value.dtype, byte_offset=value.byte_offset)


def _formal_dimension(dimension: Any) -> tuple[str | None, int]:
    """Decode the small persisted shape grammar without eval or IR imports."""
    if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0:
        return None, dimension
    if (
        isinstance(dimension, tuple)
        and len(dimension) == 2
        and dimension[0] == "var"
        and isinstance(dimension[1], str)
        and dimension[1]
    ):
        return dimension[1], 1
    if (
        isinstance(dimension, tuple)
        and len(dimension) == 3
        and dimension[0] == "mul"
        and isinstance(dimension[1], tuple)
        and len(dimension[1]) == 2
        and dimension[1][0] == "var"
        and isinstance(dimension[1][1], str)
        and dimension[1][1]
        and isinstance(dimension[2], int)
        and not isinstance(dimension[2], bool)
        and dimension[2] > 0
    ):
        return dimension[1][1], dimension[2]
    raise ValueError(f"Unsupported embedded formal shape expression {dimension!r}")


def _validate_formal_shapes(layout: Mapping[str, Any], tensors: Mapping[str, Any]) -> None:
    dimensions: dict[str, int] = {}
    for _, name, _, dtype, _, shape in layout["tensor_slots"]:
        value = tensors[name]
        if isinstance(value, torch.Tensor):
            torch_name = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(dtype, dtype)
            if value.device.type != "cpu" or value.dtype != getattr(torch, torch_name, None):
                raise ValueError(f"Embedded formal {name!r} requires CPU {dtype} tensor geometry")
            actual_shape = tuple(value.shape)
        else:
            from simpler.buffer import Tensor  # noqa: PLC0415
            from simpler.task_interface import DataType  # noqa: PLC0415

            enum = {"fp16": "FLOAT16", "bf16": "BFLOAT16", "fp32": "FLOAT32"}.get(dtype, dtype.upper())
            if not isinstance(value, Tensor) or _tensor_dtype_code(value.dtype) != getattr(
                getattr(DataType, enum, None), "value", None
            ):
                raise ValueError(f"Embedded formal {name!r} requires a native {dtype} tensor descriptor")
            actual_shape = tuple(value.shapes)
        if len(shape) != len(actual_shape):
            raise ValueError(
                f"Embedded formal {name!r} expected shape rank {len(shape)}, got {len(actual_shape)}"
            )
        for expression, actual in zip(shape, actual_shape):
            variable, factor = _formal_dimension(expression)
            if variable is None:
                if actual != factor:
                    raise ValueError(
                        f"Embedded formal {name!r} expected shape dimension {factor}, got {actual}"
                    )
            else:
                extent, remainder = divmod(actual, factor)
                if remainder or (variable in dimensions and dimensions[variable] != extent):
                    raise ValueError(
                        f"Embedded formal {name!r} has inconsistent shape dimension {variable!r}"
                    )
                dimensions[variable] = extent


def _copy_return_tensor(args: Any) -> None:
    """Registered CPU SUB: copy mapped return views after their producers drain.

    ImportRegistry already materialized these MappedArgs. Their buffers start
    at byte_offset, so no mapping, pointer wrapping or offset adjustment belongs
    here. TensorArgType.INPUT/OUTPUT_EXISTING supplies the canonical DAG edges.
    """
    from simpler.buffer import AddressSpace, BufferCapability, MappedArgs  # noqa: PLC0415
    from simpler.task_interface import DataType  # noqa: PLC0415

    if not isinstance(args, MappedArgs) or len(args) != 2 or args.scalar_count():
        raise ValueError("Return copy requires two mapped tensors and no scalars")
    source, destination = args
    source_dtype = _tensor_dtype_code(source.dtype)
    if source_dtype != _tensor_dtype_code(destination.dtype):
        raise ValueError("Return copy source and destination dtype must match")
    if tuple(source.shapes) != tuple(destination.shapes):
        raise ValueError("Return copy source and destination shape must match")
    names = {
        "BOOL": ("bool", 1),
        "INT8": ("int8", 1),
        "INT16": ("int16", 2),
        "INT32": ("int32", 4),
        "INT64": ("int64", 8),
        "UINT8": ("uint8", 1),
        "UINT16": ("uint16", 2),
        "UINT32": ("uint32", 4),
        "UINT64": ("uint64", 8),
        "FLOAT16": ("float16", 2),
        "BFLOAT16": ("bfloat16", 2),
        "FLOAT32": ("float32", 4),
        "FLOAT64": ("float64", 8),
    }
    dtype_info = next(
        (
            (getattr(torch, name), size)
            for enum, (name, size) in names.items()
            if getattr(getattr(DataType, enum, None), "value", None) == source_dtype and hasattr(torch, name)
        ),
        None,
    )
    if dtype_info is None:
        raise ValueError(f"Unsupported return-copy dtype {source.dtype!r}")
    dtype, itemsize = dtype_info

    def view(arg: Any, need: Any, operation: str) -> Any:
        if arg.imported.address_space != AddressSpace.HOST:
            raise ValueError("Return copy requires HOST mapped buffers")
        if need not in arg.imported.capabilities:
            raise ValueError(f"Return copy requires direct {operation} capability")
        shape, strides = tuple(arg.shapes), tuple(arg.strides)
        if len(shape) != len(strides) or any(dim < 0 for dim in shape) or any(s < 0 for s in strides):
            raise ValueError("Return copy requires matching nonnegative shape and stride ranks")
        if arg.byte_offset < 0 or arg.byte_offset % itemsize:
            raise ValueError("Return copy byte offset must be nonnegative and dtype aligned")
        # Destination overlap has ambiguous write semantics. Admit ordinary
        # contiguous/strided views, rejecting zero/overlapping write strides.
        if operation == "store":
            span = 1
            for stride, dim in sorted(zip(strides, shape)):
                if dim > 1:
                    if stride < span:
                        raise ValueError("Return copy destination strides overlap")
                    span += (dim - 1) * stride
        extent = 0 if 0 in shape else 1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides))
        buffer = arg.buffer
        if extent * itemsize > len(buffer):
            raise ValueError("Return copy view exceeds its mapped buffer bounds")
        if extent == 0:
            return None
        # A valid backing may end with a partial dtype element (e.g. 35 bytes
        # for an FP32 strided view). Explicit count avoids frombuffer's default
        # whole-buffer divisibility requirement without exposing that tail.
        return torch.frombuffer(buffer, dtype=dtype, count=len(buffer) // itemsize).as_strided(shape, strides)

    source_view = view(source, BufferCapability.DIRECT_LOAD, "load")
    destination_view = view(destination, BufferCapability.DIRECT_STORE, "store")
    if source_view is not None:
        destination_view.copy_(source_view)


def _alloc_invocation_tensor(orch: Any, dtype: str, shape: tuple[int, ...]) -> Any:
    from simpler.task_interface import DataType  # noqa: PLC0415

    enum = {"fp16": "FLOAT16", "bf16": "BFLOAT16", "fp32": "FLOAT32", "fp64": "FLOAT64"}
    native_dtype = getattr(DataType, enum.get(dtype, dtype.upper()))
    buffer = orch._worker.alloc_shared_tensor(shape, native_dtype)
    return buffer.tensor(shapes=shape, dtype=native_dtype)


def _validate_physical_domain(view: Any, binding: Mapping[str, Any]) -> None:
    members = tuple(
        (member.global_device_rank, member.local_worker_id, member.domain_rank) for member in view.members
    )
    if members != tuple(map(tuple, binding["members"])) or view.mapping_size < binding["window_size"]:
        raise ValueError("Global domain ordered members or mapping size differ from session layout")
    for worker in view.contexts:
        context = _domain_context(view, worker)
        if (
            context.actual_window_size != view.mapping_size
            or context.domain_size != len(members)
            or type(context.local_window_base) is not int
            or context.local_window_base <= 0
            or context.local_window_base % 64
            or members[context.domain_rank][1] != worker
        ):
            raise ValueError("Global domain physical window size/alignment/rank differs from session layout")
        offset = 0
        for name, size in binding["slots"]:
            if name not in context.buffers:
                raise ValueError(f"Global domain is missing canonical slot {name!r}")
            buffer = context.buffers[name]
            if int(buffer.base) != context.local_window_base + offset or int(buffer.nbytes) != size:
                raise ValueError("Global domain physical slot offset/size differs from canonical layout")
            offset += size


def _resolve_embedded_domains(
    layout: Mapping[str, Any],
    args: Any,
    tensors: dict[str, Any],
    orch: Any,
    workers: Mapping[int, tuple[int, ...]],
    groups: tuple[tuple[int, ...], ...],
    bindings: list[dict[str, Any]],
    slot_map: dict[tuple[int, int, str], str],
) -> None:
    by_id, by_scalar = {}, {}
    for _, name, index, dtype in layout["scalar_slots"]:
        if dtype != "comm_ctx":
            tensors[name] = _decode_scalar(args.scalar(index), dtype)
            continue
        domain_id = args.scalar(index)
        if not isinstance(domain_id, int) or isinstance(domain_id, bool) or not 0 < domain_id < 1 << 64:
            raise ValueError("Received CommCtx must be a nonzero uint64 global domain ID")
        if domain_id not in by_id:
            view = orch.get_global_domain(domain_id)
            if (
                view.domain_id != domain_id
                or not isinstance(view.generation, int)
                or isinstance(view.generation, bool)
                or view.generation <= 0
                or not view.committed
            ):
                raise ValueError(f"Global domain {domain_id} is not a committed matching generation")
            by_id[domain_id] = view
        view = by_id[domain_id]
        if bindings:
            binding = next((binding for binding in bindings if binding["name"] == view.name), None)
            if binding is None:
                raise ValueError("Received global domain is absent from the prepared slot bindings")
            _validate_physical_domain(view, binding)
            for ctx_index, logical, physical in binding["windows"]:
                if ctx_index == index:
                    key = (view.domain_id, view.generation, logical)
                    if key in slot_map and slot_map[key] != physical:
                        raise ValueError("Aliased CommCtx formals have conflicting canonical slots")
                    slot_map[key] = physical
        for worker in workers.get(index, ()):
            _domain_context(view, worker)
        tensors[name] = view
        by_scalar[index] = view
    for group in groups:
        _EmbeddedFrame.require_same_domain(tuple(by_scalar[index] for index in group))
    for _, _, _, ctx_index, window, *_ in layout["window_slots"]:
        view = by_scalar[ctx_index]
        physical = slot_map.get((view.domain_id, view.generation, window), window)
        for worker in workers.get(ctx_index, ()):
            context = _domain_context(view, worker)
            if physical not in context.buffers:
                raise ValueError(
                    f"Global domain is missing named window {window!r} for local worker {worker}"
                )


def _bind_embedded_invocation(
    entry: Any, orch: Any, args: Any
) -> tuple[_EmbeddedFrame, _ChipCallables, dict[str, Any], int]:
    """Decode a generated entry's ABI and bind its pre-registered callables."""
    from simpler.callable_identity import (  # noqa: PLC0415
        build_python_import_descriptor,
        compute_callable_hashid,
    )
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    layout = entry._pypto_embedded_layout
    bindings = entry._pypto_embedded_bindings
    world_size = bindings["world_size"]
    outputs = layout["implicit_outputs"]
    for pool, expected in (
        ("tensor", len(layout["tensor_slots"]) + len(outputs)),
        ("scalar", len(layout["scalar_slots"])),
    ):
        actual = getattr(args, f"{pool}_count")()
        if actual != expected:
            raise ValueError(f"Embedded {pool} pool expected {expected} slots, got {actual}")

    tensors = {name: args.tensor(index) for _, name, index, *_ in layout["tensor_slots"]}
    _validate_formal_shapes(layout, tensors)
    frame = _EmbeddedFrame(tensors, orch)
    frame.implicit_outputs = {position: args.tensor(index) for position, index, *_ in outputs}
    workers, groups = {}, []
    for kind, *values in layout["invocation_requirements"]:
        if kind == "domain_dispatch":
            index, selected = values
            workers[index] = tuple(range(world_size)) if selected is None else selected
        elif kind == "same_domain":
            groups.append(values[0])
        elif kind == "local_domain_tokens":
            name, selected = values
            frame.local_workers[name] = selected or tuple(range(world_size))
    _resolve_embedded_domains(
        layout, args, tensors, orch, workers, tuple(groups), bindings["domain_bindings"], frame.window_slots
    )
    for kind, *values in layout["invocation_requirements"]:
        if kind == "tensor_storage":
            name, dtype, shape = values
            tensors[name] = _alloc_invocation_tensor(orch, dtype, shape)

    def python_handle(module, name):
        return get_inner_handle(compute_callable_hashid(build_python_import_descriptor(module, name)))

    callables = _ChipCallables()
    for name, (digest, host_indices) in bindings["chips"].items():
        callables.add(name, get_inner_handle(digest), {"host_tensor_indices": host_indices})
    prefix = entry.__module__.removesuffix("." + bindings["module"])
    sub_ids = {
        name: python_handle(f"{prefix}.{module}", name) for name, module in bindings["sub_workers"].items()
    }
    if outputs:
        frame.return_copy_callable = python_handle(
            _copy_return_tensor.__module__, _copy_return_tensor.__qualname__
        )
    orch._pypto_chip_count = world_size
    orch._pypto_commless_seq = 0
    _reset_dfx_dispatch_state(orch, callables)
    return frame, callables, sub_ids, world_size
