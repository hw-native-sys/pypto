# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vendored boundary stub for the ``_task_interface`` nanobind extension.

Generated with ``nanobind.stubgen`` against the built ``_task_interface``
extension (build commit recorded in ``__build_commit__`` below), then
hand-parameterized where the C++ signature carries no type arguments but the
ABI is known: shapes and strides are int sequences.

This stub types the boundary pypto consumes while the runtime package itself
is untyped. Delete it when the runtime ships its own annotations/stubs.
"""

"""
Nanobind bindings for task_interface (DataType, Buffer/Tensor wire ABI, ChipTensor, TaskArgs variants)
"""

import enum
from collections.abc import Sequence

__build_commit__: str = "6e383fc57c12a8bd4dce045289b8b4f5d55a15f1"

class DataType(enum.Enum):
    FLOAT32 = 0

    FLOAT16 = 1

    INT32 = 2

    INT16 = 3

    INT8 = 4

    UINT8 = 5

    BFLOAT16 = 6

    INT64 = 7

    UINT64 = 8

    UINT16 = 9

    UINT32 = 10

    FP8E4M3FN = 12

    FP8E8M0 = 13

    FP4E2M1 = 14

def get_element_size(dtype: DataType) -> int:
    """Return the byte size of a single element of the given DataType."""

def get_dtype_name(dtype: DataType) -> str:
    """Return the string name of a DataType."""

MAX_TENSOR_DIMS: int = 5

MAX_REGISTERED_CALLABLE_IDS: int = 64

RUNTIME_ENV_RING_COUNT: int = 4

HOST_STRACE_ENABLED: bool = True

CHIP_TENSOR_STRIDE_BYTES: int = 72

CHIP_TENSOR_ADDRESS_SPACE_OFFSET: int = 69

OWNER_INSTANCE_ID_BYTES: int = 8

class AddressSpace(enum.IntEnum):
    HOST = 0

    DEVICE = 1

class AccessMode(enum.IntEnum):
    READ = 0

    WRITE = 1

    READWRITE = 2

class BackendKind(enum.IntEnum):
    FORK_SHM = 0

    POSIX_SHM = 1

    VMM_WINDOW = 2

    REMOTE_SIDECAR = 3

    DEVICE_MALLOC = 4

    FORK_COW = 5

class CanonicalIdentity:
    def __init__(self, owner_instance_id: bytes, buffer_id: int, generation: int = 1) -> None: ...
    @property
    def owner_instance_id(self) -> bytes:
        """
        The opaque per-incarnation nonce (bytewise-compared, no integer meaning).
        """

    @property
    def buffer_id(self) -> int: ...
    @property
    def generation(self) -> int: ...
    def __eq__(self, arg: CanonicalIdentity, /) -> bool: ...
    def __ne__(self, arg: CanonicalIdentity, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class BufferDescriptor:
    def __init__(
        self,
        identity: CanonicalIdentity,
        address_space: AddressSpace,
        access: AccessMode,
        backend_kind: BackendKind,
        nbytes: int,
        body: bytes = ...,
        owner_worker_path_id: int = 0,
    ) -> None: ...
    @property
    def identity(self) -> CanonicalIdentity: ...
    @property
    def address_space(self) -> AddressSpace: ...
    @property
    def access(self) -> AccessMode: ...
    @property
    def backend_kind(self) -> BackendKind: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def owner_worker_path_id(self) -> int: ...
    @property
    def body(self) -> bytes:
        """
        The per-backend materialization payload (shm name, base VA, ...), body_len bytes.
        """

    def __eq__(self, arg: BufferDescriptor, /) -> bool: ...
    def __ne__(self, arg: BufferDescriptor, /) -> bool: ...
    def tensor(
        self, shapes: object, dtype: object, strides: object | None = None, byte_offset: int = 0
    ) -> Tensor:
        """
        A Tensor viewing this backing. `strides` default to contiguous (row-major) element strides.
        """

    def __repr__(self) -> str: ...

class Tensor:
    def __init__(
        self,
        buffer: BufferDescriptor,
        byte_offset: int,
        shapes: Sequence[int],
        strides: Sequence[int],
        dtype: object,
    ) -> None: ...
    @property
    def buffer(self) -> BufferDescriptor: ...
    @property
    def byte_offset(self) -> int: ...
    @property
    def ndims(self) -> int: ...
    @property
    def shapes(self) -> tuple[int, ...]: ...
    @property
    def strides(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> int:
        """The dtype's int wire value (a DataType enumerator's .value)."""

    def __eq__(self, arg: Tensor, /) -> bool: ...
    def __repr__(self) -> str: ...

class ChipTensor:
    def __init__(self) -> None: ...
    @staticmethod
    def make(data: int, shapes: tuple[int, ...], dtype: DataType, child_memory: bool = False) -> ChipTensor:
        """
        Create a contiguous ChipTensor over pre-allocated memory. Set child_memory=True when data is a device pointer allocated by the child process (skips H2D copy in init_runtime_impl).
        """

    @property
    def data(self) -> int: ...
    @data.setter
    def data(self, arg: int, /) -> None: ...
    @property
    def shapes(self) -> tuple[int, ...]: ...
    @shapes.setter
    def shapes(self, arg: tuple[int, ...], /) -> None: ...
    @property
    def ndims(self) -> int: ...
    @property
    def dtype(self) -> DataType: ...
    @dtype.setter
    def dtype(self, arg: DataType, /) -> None: ...
    @property
    def child_memory(self) -> bool: ...
    @child_memory.setter
    def child_memory(self, arg: bool, /) -> None: ...
    @property
    def strides(self) -> tuple[int, ...]: ...
    @property
    def start_offset(self) -> int: ...
    @property
    def is_contiguous(self) -> bool: ...
    def nbytes(self) -> int:
        """Compute total bytes (product of shapes * element_size)."""

    def __repr__(self) -> str: ...

class ChipStorageTaskArgs:
    def __init__(self) -> None: ...
    def add_tensor(self, t: ChipTensor) -> None:
        """Add a ChipTensor. Must be called before any add_scalar()."""

    def add_scalar(self, s: object) -> None:
        """
        Add a scalar -- Python int, float, bool, or a ctypes scalar (c_int8..c_uint64, c_float, c_double, c_bool; a pointer or character type, and any scalar not in host byte order, is refused). Bit-encoded exactly like scalar_to_uint64(), which matches C++ to_u64(): a ctypes scalar zero-extends from its own width, and a native float narrows to IEEE-754 single precision -- a finite value out of that range raises rather than becoming an infinity, so use ctypes.c_double for full precision. After this, add_tensor() is no longer allowed.
        """

    def tensor(self, i: int) -> ChipTensor:
        """Return the ChipTensor at index i."""

    def scalar(self, i: int) -> int:
        """Return the scalar at index i."""

    def tensor_count(self) -> int: ...
    def scalar_count(self) -> int: ...
    def clear(self) -> None: ...
    def __len__(self) -> int:
        """Return total number of arguments (tensors + scalars)."""

    def __ptr__(self) -> int:
        """Return the memory address of the underlying C++ object."""

    @staticmethod
    def sizeof() -> int:
        """Return sizeof(ChipStorageTaskArgs) in bytes."""

class TensorArgType(enum.Enum):
    INPUT = 0

    OUTPUT = 1

    INOUT = 2

    OUTPUT_EXISTING = 3

    NO_DEP = 4

class TaskHandle:
    pass

class TaskArgs:
    def __init__(self) -> None: ...
    def add_tensor(self, t: Tensor, tag: TensorArgType = TensorArgType.INPUT) -> None:
        """
        Add a Tensor arg (the self-describing wire view built by Buffer.tensor) with an optional TensorArgType tag (default INPUT).
        """

    def add_scalar(self, s: object) -> None:
        """
        Add a scalar -- Python int, float, bool, or a ctypes scalar (c_int8..c_uint64, c_float, c_double, c_bool; a pointer or character type, and any scalar not in host byte order, is refused). Bit-encoded exactly like scalar_to_uint64(), which matches C++ to_u64(): a ctypes scalar zero-extends from its own width, and a native float narrows to IEEE-754 single precision -- a finite value out of that range raises rather than becoming an infinity, so use ctypes.c_double for full precision. After this, add_tensor() is no longer allowed.
        """

    def add_dep(self, *args) -> None:
        """Add dependencies that retain each producer until this task completes."""

    def add_dep_wait(self, *args) -> None:
        """
        Add one or more ordering-only dependencies returned by an Orchestrator submit.
        """

    def tensor(self, i: int) -> Tensor:
        """Return the Tensor at index i."""

    def scalar(self, i: int) -> int:
        """Return the scalar at index i."""

    def tag(self, i: int) -> TensorArgType:
        """Return the TensorArgType tag for the tensor at index i."""

    def set_tag(self, i: int, tag: TensorArgType) -> None:
        """Set the TensorArgType tag for the tensor at index i."""

    def tensor_count(self) -> int: ...
    def scalar_count(self) -> int: ...
    def identities(self) -> list:
        """Every tensor arg's buffer identity, in argument order."""

    def has_device_backed_tensor(self) -> bool:
        """
        Whether any arg names memory behind a chip boundary, which a dispatch must authorize.
        """

    def clear(self) -> None: ...
    def __len__(self) -> int:
        """Return total number of arguments (tensors + scalars)."""

PROV_NOT_LIVE: int = 0

PROV_DESCRIPTOR_MISMATCH: int = 1

class ProvenanceTable:
    def __init__(self) -> None: ...
    def insert(self, descriptor: BufferDescriptor, owner_worker_id: int) -> None:
        """Register one allocation, keyed by its descriptor's identity."""

    def erase(self, identity: CanonicalIdentity) -> None:
        """Revoke one allocation; absent identities are ignored."""

    def clear(self) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, arg: CanonicalIdentity, /) -> bool: ...
    def check_dispatch(self, args: TaskArgs, target_worker_id: int) -> object:
        """
        None when every device arg is live on target_worker_id, else (arg_index, reason).
        """

class ArgDirection(enum.Enum):
    SCALAR = 0

    IN = 1

    OUT = 2

    INOUT = 3

def arg_direction_name(direction: ArgDirection) -> str:
    """Return the string name of an ArgDirection."""

class CoreCallable:
    @staticmethod
    def build(signature: Sequence[ArgDirection], binary: bytes) -> CoreCallable:
        """
        Build a CoreCallable from a signature list and binary bytes. The dump maps signature entry i to payload slot i positionally.
        """

    def sig(self, i: int) -> ArgDirection:
        """Return the ArgDirection at signature index i."""

    @property
    def sig_count(self) -> int:
        """Number of signature entries."""

    @property
    def binary_size(self) -> int:
        """Size of the binary payload in bytes."""

    def buffer_ptr(self) -> int:
        """Return the memory address of the underlying buffer."""

    def buffer_size(self) -> int:
        """Return the total size of the underlying buffer in bytes."""

    def __repr__(self) -> str: ...

class ChipCallable:
    @staticmethod
    def build(
        signature: Sequence[ArgDirection],
        func_name: str,
        binary: bytes,
        children: Sequence[tuple[int, CoreCallable]],
        config_name: str = "",
    ) -> ChipCallable:
        """
        Build a ChipCallable from signature, func_name, binary, and list of (func_id, CoreCallable) children.
        """

    @staticmethod
    def from_bytes(raw: bytes) -> ChipCallable:
        """
        Reconstruct a ChipCallable from the contiguous bytes that buffer_ptr() points to (size buffer_size()). Inverse of the serialisation used to ship a ChipCallable across the L4 cascade IPC channel.
        """

    def sig(self, i: int) -> ArgDirection:
        """Return the ArgDirection at signature index i."""

    @property
    def sig_count(self) -> int:
        """Number of signature entries."""

    @property
    def binary_size(self) -> int:
        """Size of the binary payload in bytes."""

    @property
    def func_name(self) -> str:
        """The orchestration function name."""

    @property
    def config_name(self) -> str:
        """The optional orchestration config function name."""

    @property
    def scalar_count(self) -> int:
        """Number of SCALAR entries in the orchestration signature."""

    @property
    def child_count(self) -> int:
        """Number of child callables."""

    def child_func_id(self, i: int) -> int:
        """Return the func_id for child at index i."""

    def child(self, i: int) -> CoreCallable:
        """Return the CoreCallable child at index i."""

    def child_offset(self, i: int) -> int:
        """
        Return the byte offset of child i within storage (must be multiple of 64).
        """

    def buffer_ptr(self) -> int:
        """Return the memory address of the underlying buffer."""

    def buffer_size(self) -> int:
        """Return the total size of the underlying buffer in bytes."""

    def __repr__(self) -> str: ...

class RuntimeEnv:
    def __init__(self) -> None: ...
    @property
    def ring_task_window(self) -> list[int]: ...
    @ring_task_window.setter
    def ring_task_window(self, arg: object, /) -> None: ...
    @property
    def ring_heap(self) -> list[int]: ...
    @ring_heap.setter
    def ring_heap(self, arg: object, /) -> None: ...
    @property
    def ring_dep_pool(self) -> list[int]: ...
    @ring_dep_pool.setter
    def ring_dep_pool(self, arg: object, /) -> None: ...
    def __repr__(self) -> str: ...

class CallConfig:
    def __init__(self) -> None: ...
    @property
    def aicpu_thread_num(self) -> int: ...
    @aicpu_thread_num.setter
    def aicpu_thread_num(self, arg: int, /) -> None: ...
    @property
    def runtime_env(self) -> RuntimeEnv: ...
    @runtime_env.setter
    def runtime_env(self, arg: RuntimeEnv, /) -> None: ...
    @property
    def enable_chip_swimlane(self) -> int: ...
    @enable_chip_swimlane.setter
    def enable_chip_swimlane(self, arg: object, /) -> None: ...
    @property
    def enable_dump_args(self) -> int: ...
    @enable_dump_args.setter
    def enable_dump_args(self, arg: object, /) -> None: ...
    @property
    def enable_pmu(self) -> int: ...
    @enable_pmu.setter
    def enable_pmu(self, arg: int, /) -> None: ...
    def validate(self) -> None: ...
    @property
    def enable_dep_gen(self) -> bool: ...
    @enable_dep_gen.setter
    def enable_dep_gen(self, arg: bool, /) -> None: ...
    @property
    def enable_scope_stats(self) -> bool: ...
    @enable_scope_stats.setter
    def enable_scope_stats(self, arg: bool, /) -> None: ...
    @property
    def capture_clock_anchors(self) -> bool: ...
    @capture_clock_anchors.setter
    def capture_clock_anchors(self, arg: bool, /) -> None: ...
    @property
    def output_prefix(self) -> str: ...
    @output_prefix.setter
    def output_prefix(self, arg: str, /) -> None: ...
    def __repr__(self) -> str: ...

DEFAULT_LOG_THRESHOLD: int = 25

class DeviceMemoryInfo:
    @property
    def free_bytes(self) -> int: ...
    @property
    def total_bytes(self) -> int: ...
    def __iter__(self) -> object: ...
    def __repr__(self) -> str: ...

def scalar_to_uint64(value: object) -> int:
    """
    Bit-encode a Python int, float, bool, or ctypes scalar into the uint64 a scalar slot stores, matching C++ to_u64() bit for bit. A ctypes scalar is read at its own width and zero-extended, so ctypes.c_int8(-1) is 0xFF rather than a sign-extended 0xFF..FF; the admitted ctypes types are c_int8..c_uint64, c_float, c_double and c_bool in host byte order, and a pointer type, a character type, or a byte-order-qualified variant such as c_uint32.__ctype_be__ is refused. A native Python float narrows to IEEE-754 single precision and zero-extends, and a finite value out of single-precision range raises rather than becoming an infinity; pass ctypes.c_double for full precision, or another ctypes scalar for exact width.
    """

def materialize_task_args(args: TaskArgs, resolved: dict) -> ChipStorageTaskArgs:
    """
    Materialize a TaskArgs held in this process into the runtime.so-ABI ChipStorageTaskArgs POD — the sole path to that POD, whether the args are an L2 leaf's own or a chip child's read back from its mailbox with read_args_from_blob. Each tensor's embedded buffer identity is resolved via `resolved` {CanonicalIdentity: (local_base, address_space)}; addr = base + byte_offset. The caller pre-populates `resolved` by materializing each embedded descriptor on first receipt. Strided views (transpose / permute / step-slice) materialize to strided ChipTensors. Rejects an unknown identity and a non-dtype-aligned byte_offset.
    """

def read_args_from_blob(blob_ptr: int, capacity: int) -> TaskArgs:
    """
    Reconstruct a TaskArgs from the length-prefixed blob at blob_ptr. `capacity` bounds how far the reader may walk and belongs to the caller's mapping — the mailbox frame's args region, or the length of a buffer the caller owns. Every element is gated by validate_tensor on the way out. Tags are not preserved (the wire format strips them).
    """

class WorkerType(enum.Enum):
    NEXT_LEVEL = 0

    SUB = 1

class ControlResult:
    @property
    def worker_type(self) -> str: ...
    @property
    def worker_id(self) -> int: ...
    @property
    def ok(self) -> bool: ...
    @property
    def error_message(self) -> str: ...

class TaskState(enum.Enum):
    FREE = 0

    BUILDING = 7

    PENDING = 1

    READY = 2

    RUNNING = 3

    COMPLETED = 4

    FAILED = 5

    CONSUMED = 6

DEFAULT_HEAP_RING_SIZE: int = 1073741824

MAILBOX_SIZE: int = 196608

MAILBOX_FRAME_SIZE: int = 65536

MAILBOX_OFF_ERROR_MSG: int = 65280

MAILBOX_ERROR_MSG_SIZE: int = 256

MAILBOX_STATE_VALUES: dict = ...

MAILBOX_PREPARATION_DISPOSITION_VALUES: dict = ...

PTO_PIPELINE_MAX_DEPTH: int = 2

MAX_RING_DEPTH: int = 4

MAX_SCOPE_DEPTH: int = 64
