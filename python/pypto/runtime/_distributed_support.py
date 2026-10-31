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
from types import MappingProxyType
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
    backing = (ctypes.c_ubyte * size).from_address(arg.data)
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
    from simpler.task_interface import Tensor  # noqa: PLC0415

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
                from simpler.buffer import AddressSpace  # noqa: PLC0415
                from simpler.task_interface import Tensor  # noqa: PLC0415
            except ImportError:
                is_host = False
            else:
                is_host = isinstance(tensor, Tensor) and tensor.buffer.address_space == AddressSpace.HOST
        if not is_host:
            raise ValueError(f"Chip {name!r} host control tensor slot {index} requires a CPU tensor")
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


def _reset_dfx_dispatch_state(orch: Any, chip_cids: dict[str, Any]) -> None:
    """Reset the per-run dispatch state :func:`_submit_chip` reads off ``orch``.

    ``_dfx_dispatch_idx`` numbers each card's dispatches ``d0, d1, ...`` fresh per
    run, so the swimlane two-pass files one dispatch under one directory.

    ``_dfx_chip_names`` reverses the registered ``callables`` mapping so
    ``_submit_chip`` can name the L2 program behind an otherwise opaque
    ``CallableHandle``. Keyed by ``id()``: the handle is not required to be
    hashable, and ``chip_cids`` holds every one alive for the whole run, so the
    ids are stable and unique.
    """
    orch._dfx_dispatch_idx = {}
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

    ``k`` comes from a per-card counter on ``orch`` reset at the top of every
    run (see :func:`_reset_dfx_dispatch_state`), so it is this run's dispatch
    ordinal. The child's own counter spans the whole process, so the two
    numberings agree only on a worker's first run: the marker
    :func:`_record_dispatch_program` stamps here therefore lands on the first
    run's capture directory — the graph pass of a swimlane two-pass — and
    :func:`_collect_l3_swimlane` reads it back through the capture pairing
    rather than expecting it beside the records.

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
    idx_map[rank_label] = k + 1
    _record_dispatch_program(orch, callable_id, Path(f"{base}/{rank_label}/d{k}"))
    return orch.submit_next_level(callable_id, task_args, config, worker=worker)


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
    """One invocation's bindings and references, retained through DAG drain."""

    def __init__(self, tensors: dict[str, Any], args: Any) -> None:
        self.tensors = tensors
        self.implicit_outputs: dict[int, Any] = {}
        self.keepalive: list[Any] = [args]
        self.domain_tokens: dict[tuple[int, int], dict[int, Any]] = {}
        self.local_tokens: dict[int, Mapping[int, Any]] = {}
        self.local_specs: dict[str, tuple[tuple[int, ...], Mapping[int, Any]]] = {}
        self.window_slots: Mapping[tuple[int, int, str], str] = {}

    def window_tensor(self, domain: Any, worker: int, logical_name: str, **view: Any) -> Any:
        """Translate a slot identity, then use the actual runtime Buffer object."""
        physical = logical_name
        if hasattr(domain, "domain_id"):
            physical = self.window_slots.get(
                (domain.domain_id, domain.generation, logical_name), logical_name
            )
        return _domain_context(domain, worker).buffers[physical].tensor(**view)

    @staticmethod
    def require_same_domain(domains: tuple[Any, ...]) -> None:
        if len({_ordering_domain_key(domain) for domain in domains}) > 1:
            raise ValueError("Builtin window arguments require the same physical domain")

    def bind_local_domain(self, domain: Any, key: str, window_names: tuple[str, ...]) -> None:
        workers, tokens = self.local_specs[key]
        for worker in workers:
            context = _domain_context(domain, worker)
            for name in window_names:
                if name not in context.buffers:
                    raise ValueError(f"Local domain is missing named window {name!r} for worker {worker}")
        self.local_tokens[domain.allocation_id] = tokens

    def ordering_tokens(self, domains: tuple[Any, ...], worker: int) -> tuple[Any, ...]:
        tokens, seen = [], set()
        for domain in domains:
            key = _ordering_domain_key(domain)
            if key in seen:
                continue
            seen.add(key)
            _domain_context(domain, worker)
            if key[0] == "global":
                tokens.append(self.domain_tokens[(key[1], key[2])][worker])
            else:
                tokens.append(self.local_tokens[key[1]][worker])
        return tuple(tokens)


def _ordering_domain_key(domain: Any) -> tuple[Any, ...]:
    if hasattr(domain, "domain_id"):
        return ("global", domain.domain_id, domain.generation)
    return ("local", domain.allocation_id)


def _domain_context(domain: Any, worker: int) -> Any:
    if hasattr(domain, "domain_id") and not domain.committed:
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


class _EmbeddedNodeState:
    """Bounded shared token leases; storage is supplied before the inner fork.

    Worker IDs identify participating local workers, independently of global
    domain ranks. Completed leases remain stable across entry calls while the
    imported view stays committed. Invalidated, drained views can release their
    slots on the next acquisition; explicit retirement remains available after
    drain. No method allocates tensors.
    """

    def __init__(
        self, *, local_worker_ids: tuple[int, ...], prepared_domain_tokens: tuple[tuple[Any, ...], ...]
    ) -> None:
        if (
            not isinstance(local_worker_ids, tuple)
            or not local_worker_ids
            or any(
                not isinstance(worker, int) or isinstance(worker, bool) or worker < 0
                for worker in local_worker_ids
            )
            or len(set(local_worker_ids)) != len(local_worker_ids)
        ):
            raise ValueError("Node token storage requires unique nonnegative local worker IDs")
        if not isinstance(prepared_domain_tokens, tuple):
            raise ValueError("Prepared domain tokens must be immutable bundles")
        seen: set[int] = set()
        for bundle in prepared_domain_tokens:
            if not isinstance(bundle, tuple) or len(bundle) != len(local_worker_ids):
                raise ValueError("Each prepared domain token bundle must cover the local worker IDs")
            for token in bundle:
                _validate_prepared_tensor(token, "domain ordering token", "int32", (1,))
                if not isinstance(token, torch.Tensor):
                    raise ValueError("Prepared domain ordering tokens require shared CPU Torch storage")
                storage = token.untyped_storage().data_ptr()
                if storage in seen:
                    raise ValueError("Domain ordering tokens must have separate physical storage")
                seen.add(storage)
        self.local_worker_ids = local_worker_ids
        self._tokens = prepared_domain_tokens
        self._bindings: dict[tuple[int, int], tuple[int, int]] = {}
        self._domain_views: dict[tuple[int, int], Any] = {}
        self._closed = False

    def acquire(self, domains: Mapping[tuple[int, int], Any]) -> dict[tuple[int, int], dict[int, Any]]:
        if self._closed:
            raise RuntimeError("Embedded node state is closed")
        # R1 invalidates the same imported view before destructive domain
        # release. Keep a strong reference until that boundary AND completion
        # of every entry lease; task completion alone cannot break the shared
        # ordering chain of a still-live physical domain.
        retired = {
            key
            for key, (_, active) in self._bindings.items()
            if not active and not self._domain_views[key].committed
        }
        bindings = {key: binding for key, binding in self._bindings.items() if key not in retired}
        missing = [key for key in domains if key not in bindings]
        occupied = {slot for slot, _ in bindings.values()}
        free = [slot for slot in range(len(self._tokens)) if slot not in occupied]
        if len(missing) > len(free):
            raise ValueError(
                "Insufficient prefork domain token capacity; retire completed leases before reuse"
            )
        for domain_id, generation in domains:
            if any(
                existing_id == domain_id and existing_generation != generation and active
                for (existing_id, existing_generation), (_, active) in bindings.items()
            ):
                raise RuntimeError("A prior generation still has active invocation leases")
        # All validation/capacity checks precede the commit. Calls with aliases
        # acquire one canonical key once; no failed preflight changes this map.
        for key, slot in zip(missing, free):
            bindings[key] = (slot, 0)
        result = {}
        for key in domains:
            slot, active = bindings[key]
            bindings[key] = (slot, active + 1)
            result[key] = dict(zip(self.local_worker_ids, self._tokens[slot]))
        views = {key: view for key, view in self._domain_views.items() if key not in retired}
        views.update((key, domains[key]) for key in missing)
        self._bindings = bindings
        self._domain_views = views
        return result

    def release(self, keys: tuple[tuple[int, int], ...]) -> None:
        for key in keys:
            slot, active = self._bindings[key]
            if active <= 0:
                raise RuntimeError("Embedded domain invocation lease was already completed")
            self._bindings[key] = (slot, active - 1)

    def retire_domain(self, domain_id: int, generation: int) -> None:
        key = (domain_id, generation)
        binding = self._bindings.get(key)
        if binding is None:
            return
        if binding[1]:
            raise RuntimeError(
                "Domain retirement requires all active invocation frames to complete and drain"
            )
        del self._bindings[key]
        self._domain_views.pop(key)

    def close(self) -> None:
        if any(active for _, active in self._bindings.values()):
            raise RuntimeError("Node close requires active invocation frames to complete and drain")
        self._bindings.clear()
        self._domain_views.clear()
        self._tokens = ()
        self._closed = True


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
    from simpler.task_interface import Tensor  # noqa: PLC0415

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
    from simpler.task_interface import Tensor  # noqa: PLC0415

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
            from simpler.task_interface import DataType, Tensor  # noqa: PLC0415

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


def _validate_prepared_tensor(value: Any, name: str, dtype: str, shape: tuple[int, ...]) -> None:
    """Check prefork storage without allocating or reminting its provenance."""
    torch_name = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(dtype, dtype)
    if isinstance(value, torch.Tensor):
        if (
            value.device.type != "cpu"
            or not value.is_shared()
            or value.dtype != getattr(torch, torch_name, None)
            or tuple(value.shape) != shape
        ):
            raise ValueError(
                f"Prepared tensor {name!r} requires shared CPU {dtype} storage with shape {shape}"
            )
        return
    from simpler.buffer import AddressSpace  # noqa: PLC0415
    from simpler.task_interface import DataType, Tensor  # noqa: PLC0415

    enum = {"fp16": "FLOAT16", "bf16": "BFLOAT16", "fp32": "FLOAT32"}.get(dtype, dtype.upper())
    if (
        not isinstance(value, Tensor)
        or value.buffer.address_space != AddressSpace.HOST
        or tuple(value.shapes) != shape
        or _tensor_dtype_code(value.dtype) != getattr(getattr(DataType, enum, None), "value", None)
    ):
        raise ValueError(f"Prepared tensor {name!r} requires a HOST {dtype} descriptor with shape {shape}")


def _validate_return_preparation(
    layout: Mapping[str, Any],
    prepared_tensors: Mapping[str, Any],
    return_copy_callable: Any,
) -> None:
    """Validate persisted return layout and prefork requirements without allocation."""
    outputs = layout["implicit_outputs"]
    positions: set[int] = set()
    for row in outputs:
        if not isinstance(row, tuple) or len(row) != 4:
            raise ValueError("Embedded implicit output must declare return position, slot, dtype and shape")
        position, index, dtype, shape = row
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            or position in positions
            or not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(dtype, str)
            or not isinstance(shape, tuple)
            or any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape)
        ):
            raise ValueError(
                "Embedded implicit output has an invalid position, slot, dtype or constant shape"
            )
        positions.add(position)
    output_indices = [row[1] for row in outputs]
    if output_indices != list(range(len(layout["tensor_slots"]), len(layout["tensor_slots"]) + len(outputs))):
        raise ValueError("Embedded implicit output indices must follow the dense tensor formals")
    requirements = layout["prefork_requirements"]
    if bool(outputs) != (("return_copy_subworker", 1) in requirements):
        raise ValueError("Embedded implicit outputs require one registered return-copy SubWorker")
    if outputs and return_copy_callable is None:
        raise ValueError("Embedded implicit outputs require return_copy_callable before worker init")
    for requirement in requirements:
        if not isinstance(requirement, tuple):
            raise ValueError("Embedded prefork requirement must be an immutable tuple")
        if requirement == ("return_copy_subworker", 1):
            continue
        if requirement and requirement[0] in {
            "inherited_domain_tokens",
            "domain_dispatch",
            "same_domain",
            "local_domain_tokens",
        }:
            continue
        if len(requirement) != 4 or requirement[0] != "tensor_storage":
            raise ValueError(f"Unsupported embedded prefork requirement {requirement!r}")
        _, name, dtype, shape = requirement
        if not isinstance(name, str) or not isinstance(dtype, str) or not isinstance(shape, tuple):
            raise ValueError("Prepared tensor requirement must declare name, dtype and shape")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise ValueError(f"Prepared tensor {name!r} requires a constant nonnegative shape")
        if name not in prepared_tensors:
            raise ValueError(f"Missing prefork prepared tensor {name!r}")
        _validate_prepared_tensor(prepared_tensors[name], name, dtype, shape)


def _local_token_spec(row: tuple[Any, ...], prepared: Mapping[str, Any], world_size: int) -> tuple[Any, Any]:
    if len(row) != 3 or not isinstance(row[1], str) or not isinstance(row[2], tuple):
        raise ValueError("Local domain token requirement must declare its key and workers")
    workers = row[2] or tuple(range(world_size))
    if any(
        not isinstance(worker, int) or isinstance(worker, bool) or not 0 <= worker < world_size
        for worker in workers
    ):
        raise ValueError("Local domain token worker lies outside the node topology")
    tokens = prepared.get(row[1])
    if not isinstance(tokens, (tuple, list)) or len(tokens) < world_size:
        raise ValueError(f"Missing prefork local domain token capacity for {row[1]!r}")
    for token in tokens:
        _validate_prepared_tensor(token, row[1], "int32", (1,))
        if not isinstance(token, torch.Tensor):
            raise ValueError("Local ordering tokens require shared CPU Torch storage")
    return workers, dict(enumerate(tokens))


def _validate_window_slots(layout: Mapping[str, Any], contexts: Mapping[int, tuple[Any, ...]]) -> None:
    windows = layout["window_slots"]
    for row in windows:
        if not isinstance(row, tuple) or len(row) != 8:
            raise ValueError("Window slot must declare formal, paired ctx, named buffer, dtype and shape")
        position, name, ctx_position, ctx_index, window, dtype, direction, shape = row
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            or not all(isinstance(value, str) for value in (name, window, dtype))
            or direction not in ("In", "Out", "InOut")
            or not isinstance(shape, tuple)
            or any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape)
            or ctx_index not in contexts
            or contexts[ctx_index][0] != ctx_position
        ):
            raise ValueError("Invalid embedded window/ctx binding record")
    if windows:
        formals = [
            row[0] for field in ("tensor_slots", "scalar_slots", "window_slots") for row in layout[field]
        ]
        if sorted(formals) != list(range(len(formals))):
            raise ValueError("Embedded formals must preserve every original position")
        if [row[2] for row in windows] != list(range(len(formals) - len(windows), len(formals))):
            raise ValueError("Window ctx pairs must use only the final materialized suffix")


def _domain_requirements(
    layout: Mapping[str, Any],
    prepared: Mapping[str, Any],
    world_size: int,
    node: _EmbeddedNodeState | None,
) -> tuple[dict[int, tuple[int, ...]], tuple[tuple[int, ...], ...], dict[str, Any]]:
    contexts = {row[2]: row for row in layout["scalar_slots"] if row[3] == "comm_ctx"}
    if contexts and not isinstance(node, _EmbeddedNodeState):
        raise ValueError("Received CommCtx formals require bound prefork node token state")
    _validate_window_slots(layout, contexts)
    workers, groups, locals_ = {}, [], {}
    upper_bounds = []
    for row in layout["prefork_requirements"]:
        kind = row[0]
        if kind == "inherited_domain_tokens":
            upper_bounds.append(row[1] if len(row) == 2 else None)
        elif kind == "domain_dispatch":
            if (
                len(row) != 3
                or row[1] not in contexts
                or row[2] is not None
                and not isinstance(row[2], tuple)
            ):
                raise ValueError("Domain dispatch must declare a received ctx scalar and local workers")
            selected = tuple(range(world_size)) if row[2] is None else row[2]
            if any(
                not isinstance(worker, int)
                or isinstance(worker, bool)
                or not 0 <= worker < world_size
                or node is None
                or worker not in node.local_worker_ids
                for worker in selected
            ):
                raise ValueError("Prefork domain tokens do not cover a required local worker")
            workers[row[1]] = selected
        elif kind == "same_domain":
            if (
                len(row) != 2
                or not isinstance(row[1], tuple)
                or any(index not in contexts for index in row[1])
            ):
                raise ValueError("Physical domain consensus must name received ctx scalar slots")
            groups.append(row[1])
        elif kind == "local_domain_tokens":
            spec = _local_token_spec(row, prepared, world_size)
            locals_[row[1]] = spec
    if upper_bounds != ([len(contexts)] if contexts else []):
        raise ValueError("Inherited token upper bound must preserve every received CommCtx formal")
    # Shared ordering must never accidentally reuse one physical backing for
    # separate local/global domains or worker lanes. This only inspects Torch
    # storage identity; it does not wrap or dereference an address.
    seen = (
        {token.untyped_storage().data_ptr() for bundle in node._tokens for token in bundle} if node else set()
    )
    for _, tokens in locals_.values():
        for token in tokens.values():
            storage = token.untyped_storage().data_ptr()
            if storage in seen:
                raise ValueError("Local and inherited ordering tokens require separate physical storage")
            seen.add(storage)
    return workers, tuple(groups), locals_


def _domain_bindings(value: Any, layout: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Snapshot bounded native binding records without requiring the compiler."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("Domain slot bindings must be a sequence")
    result, names = [], set()
    windows = {(row[3], row[4]) for row in layout["window_slots"]}
    for binding in value:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"name", "members", "window_size", "slots", "windows"}
            or not isinstance(binding["name"], str)
            or not binding["name"]
            or binding["name"] in names
        ):
            raise ValueError("Invalid domain slot binding fields/name")
        slots, members, selected = binding["slots"], binding["members"], binding["windows"]
        if (
            not isinstance(slots, (list, tuple))
            or not 0 < len(slots) <= 64
            or any(
                not isinstance(slot, (list, tuple))
                or len(slot) != 2
                or not isinstance(slot[0], str)
                or not slot[0]
                or type(slot[1]) is not int
                or slot[1] <= 0
                or slot[1] % 64
                for slot in slots
            )
            or len({slot[0] for slot in slots}) != len(slots)
            or type(binding["window_size"]) is not int
            or sum(slot[1] for slot in slots) != binding["window_size"]
        ):
            raise ValueError("Invalid canonical slot sizes/alignment/window")
        if (
            not isinstance(members, (list, tuple))
            or not 0 < len(members) <= 64
            or any(
                not isinstance(member, (list, tuple))
                or len(member) != 3
                or any(type(item) is not int or item < 0 for item in member)
                or member[2] != rank
                for rank, member in enumerate(members)
            )
            or len({member[0] for member in members}) != len(members)
        ):
            raise ValueError("Invalid ordered domain member layout")
        if (
            not isinstance(selected, (list, tuple))
            or not selected
            or any(
                not isinstance(row, (list, tuple))
                or len(row) != 3
                or type(row[0]) is not int
                or (row[0], row[1]) not in windows
                or row[2] not in {slot[0] for slot in slots}
                for row in selected
            )
            or len({(row[0], row[1]) for row in selected}) != len(selected)
        ):
            raise ValueError("Invalid logical-to-canonical window bindings")
        names.add(binding["name"])
        result.append(
            MappingProxyType(
                {
                    "name": binding["name"],
                    "window_size": binding["window_size"],
                    "members": tuple(tuple(row) for row in members),
                    "slots": tuple(tuple(row) for row in slots),
                    "windows": tuple(tuple(row) for row in selected),
                }
            )
        )
    return tuple(result)


def _validate_physical_domain(view: Any, binding: Mapping[str, Any]) -> None:
    members = tuple(
        (member.global_device_rank, member.local_worker_id, member.domain_rank) for member in view.members
    )
    if members != binding["members"] or view.mapping_size < binding["window_size"]:
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
    bindings: tuple[Mapping[str, Any], ...] = (),
    slot_map: dict[tuple[int, int, str], str] | None = None,
) -> dict[tuple[int, int], Any]:
    by_id, by_scalar, domains = {}, {}, {}
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
                if ctx_index == index and slot_map is not None:
                    key = (view.domain_id, view.generation, logical)
                    if key in slot_map and slot_map[key] != physical:
                        raise ValueError("Aliased CommCtx formals have conflicting canonical slots")
                    slot_map[key] = physical
        for worker in workers.get(index, ()):
            _domain_context(view, worker)
        tensors[name] = view
        by_scalar[index] = view
        domains[(view.domain_id, view.generation)] = view
    for group in groups:
        _EmbeddedFrame.require_same_domain(tuple(by_scalar[index] for index in group))
    for _, _, _, ctx_index, window, *_ in layout["window_slots"]:
        view = by_scalar[ctx_index]
        physical = (slot_map or {}).get((view.domain_id, view.generation, window), window)
        for worker in workers.get(ctx_index, ()):
            context = _domain_context(view, worker)
            if physical not in context.buffers:
                raise ValueError(
                    f"Global domain is missing named window {window!r} for local worker {worker}"
                )
    return domains


class _EmbeddedEntryState:
    """Bound dense-wire entry state with explicit invocation completion.

    Startup supplies registered handles and any prepared tensor storage before
    the inner worker forks. A frame survives the generated entry's return:
    complete_invocation must be called only after the corresponding Worker.run
    has drained, and close only after worker shutdown has drained pending work.
    No method allocates Torch storage or shared-memory ordering tokens.
    """

    def __init__(
        self,
        layout: Mapping[str, Any],
        *,
        callables: dict[str, Any],
        sub_ids: dict[str, Any],
        world_size: int,
        prepared_tensors: Mapping[str, Any],
        domain_provider: Any,
        return_copy_callable: Any = None,
        node_state: _EmbeddedNodeState | None = None,
        domain_bindings: Any = (),
    ) -> None:
        if layout.get("version") != 1:
            raise ValueError(f"Expected embedded layout version 1, got {layout.get('version')!r}")
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
            raise ValueError(f"Expected a positive local device count, got {world_size!r}")
        for field in (
            "tensor_slots",
            "scalar_slots",
            "window_slots",
            "implicit_outputs",
            "prefork_requirements",
        ):
            if not isinstance(layout.get(field), tuple):
                raise ValueError(f"Embedded layout {field!r} must be an immutable tuple")
        for field in ("tensor_slots", "scalar_slots"):
            indices = [row[2] for row in layout[field]]
            if indices != list(range(len(indices))):
                raise ValueError(f"Embedded {field} indices must be dense from zero, got {indices}")
        for row in layout["tensor_slots"]:
            if len(row) != 6 or not isinstance(row[5], tuple):
                raise ValueError("Embedded tensor formal must declare an immutable shape tuple")
            for dimension in row[5]:
                _formal_dimension(dimension)
        _validate_return_preparation(layout, prepared_tensors, return_copy_callable)
        self._domain_workers, self._domain_groups, self._local_specs = _domain_requirements(
            layout,
            prepared_tensors,
            world_size,
            node_state,
        )
        self.node_state = node_state
        self._domain_bindings = _domain_bindings(domain_bindings, layout)
        self.layout = MappingProxyType(dict(layout))
        self.callables = callables
        self.sub_ids = sub_ids
        self.world_size = world_size
        self.domain_provider = domain_provider
        self.return_copy_callable = return_copy_callable
        self._prepared_tensors = dict(prepared_tensors)
        self._frame: _EmbeddedFrame | None = None
        self._closed = False

    def begin_invocation(self, orch: Any, args: Any, config: Any) -> _EmbeddedFrame:
        """Bind dense pools without retiring an incomplete invocation."""
        if self._closed:
            raise RuntimeError("Embedded entry state is closed")
        if self._frame is not None:
            raise RuntimeError("Previous embedded invocation requires completion before reuse")
        for pool, field in (("tensor", "tensor_slots"), ("scalar", "scalar_slots")):
            expected = len(self.layout[field])
            if pool == "tensor":
                expected += len(self.layout["implicit_outputs"])
            actual = getattr(args, f"{pool}_count")()
            if actual != expected:
                raise ValueError(f"Embedded {pool} pool expected {expected} slots, got {actual}")
        tensors = dict(self._prepared_tensors)
        for _, name, index, *_ in self.layout["tensor_slots"]:
            tensors[name] = args.tensor(index)
        _validate_formal_shapes(self.layout, tensors)
        slot_map = {}
        domains = _resolve_embedded_domains(
            self.layout,
            args,
            tensors,
            orch,
            self._domain_workers,
            self._domain_groups,
            self._domain_bindings,
            slot_map,
        )
        frame = _EmbeddedFrame(tensors, args)
        frame.window_slots = MappingProxyType(slot_map)
        frame.local_specs = self._local_specs
        frame.implicit_outputs = {
            position: args.tensor(index) for position, index, *_ in self.layout["implicit_outputs"]
        }
        orch._pypto_chip_count = self.world_size
        orch._pypto_commless_seq = 0
        _reset_dfx_dispatch_state(orch, self.callables)
        # Publish only after every decode/view/context/slot check has passed.
        # Node acquisition preflights all requested capacity before mutation.
        if self.node_state is not None:
            frame.domain_tokens = self.node_state.acquire(domains)
        self._frame = frame
        return frame

    def complete_invocation(self) -> None:
        """Release invocation references after its runtime DAG has drained."""
        if self._frame is not None:
            if self.node_state is not None:
                self.node_state.release(tuple(self._frame.domain_tokens))
            self._frame.domain_tokens.clear()
            self._frame.local_tokens.clear()
            self._frame.keepalive.clear()
            self._frame = None

    def close(self) -> None:
        """Retire state after the bound worker has drained and closed."""
        self.complete_invocation()
        self._prepared_tensors.clear()
        self._closed = True
