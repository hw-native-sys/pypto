# Kernel-mode integration foundations

The internal torch adapter describes borrowed NPU arguments for a future kernel
executor. It does not add a public kernel execution entry point. Existing JIT,
compiled-program and Worker calls keep their current behavior.

## Call metadata and ownership

[`CallSignature`](../../../../python/pypto/torch/interop.py) copies the shared
`ir.param_info.ParamInfo` signature once. `describe_call(args)` uses
`bind_complete_args`, so every Out/InOut parameter must be supplied before any
framework-context query. Return aliases are tensor indices in the complete
parameter list; an alias cannot refer to a scalar or an absent argument.

Each call produces a new immutable `CallFrame`:

| Field | Meaning |
| ----- | ------- |
| `tensors` | Tensor arguments in signature order, with original `param_index`, direction, dtype, shape, strides, format, logical pointer and storage bounds. |
| `scalars` | Current typed primitive values in signature order, with original `param_index`; mutable ctypes inputs are copied. |
| `device_index` | The current NPU device, checked against every tensor. |
| `stream` | This call's current torch_npu stream object, retained by the frame. |
| `return_tensors` | The exact caller objects selected by validated return aliases. |

Tensor and scalar ordering does not define a native ABI layout. Native argument
encoding will consume these values together with a validated kernel descriptor.
Addresses, scalar values and streams are per-call state, not compilation keys or
persistent metadata.

The frame retains each tensor and its storage object. `alias_result()` returns
`None`, one existing tensor, or a tuple of existing tensors. It never allocates
business outputs. Python ownership alone does not protect asynchronous device
use after the frame is released; native queue ownership and allocator stream
recording are separate launch-layer work. Callers must not resize or invalidate
borrowed storage while a frame is in use.

## Validation

- Accept real NPU torch tensors on one device. CPU, Meta/Fake tensors and
  Worker-owned objects are rejected; the adapter never selects program mode.
- Check dtype and rank against `ParamInfo`. Static carrier dimensions must
  match; `-1` dimensions use the current extent. Packed FP4 uses the carrier
  shape already provided by `ParamInfo`, without expanding it again.
- Require a contiguous strided view in base NCHW (0) or ND (2) format. Preserve
  nonzero storage offsets and logical data pointers. Other formats, transposed
  views, unresolved conjugate/negative views and invalid storage bounds fail
  explicitly, without copying or format conversion.
- Accept exact tensor-view aliases and disjoint views of shared storage.
  Overlapping read-only views are allowed. Partially overlapping views involving
  Out/InOut require a richer alias contract and are rejected.
- Copy Python or correctly typed ctypes scalars using the program scalar type
  map. Integer values must fit their declared type; floats are not silently
  truncated into integers. This does not change JIT runtime-scalar or constant
  classification.
- Reject gradient-bearing tensors when autograd is enabled. `torch.no_grad()`
  inference may borrow model parameters; no backward implementation is supplied.

After argument validation, the adapter obtains the current device and stream on
**every call**. It requires the current device to match the tensor device and
checks the stream device. It does not switch devices, cache the first stream,
read `npu_stream`, synchronize, or enqueue commands. These context queries can
initialize torch_npu's own framework context; they do not create or initialize a
Simpler/PyPTO Worker.

## Optional dependencies and scope

`import pypto.torch` exports no execution API. Importing it or its `interop`
module does not request torch_npu, Simpler or a native launch extension.
`torch` remains a normal PyPTO dependency. A real call description loads
`torch_npu` on demand and reports a targeted error if it is unavailable.

This foundation covers metadata validation and Python call-frame ownership.
It does not claim kernel launch, taskQueue ordering, allocator safety, eager
numerical execution or ACLGraph support. Those need the native adapter and
runtime integration before a public entry-point switch.

## Verification

`tests/ut/torch/test_interop.py` uses real CPU storage with an emulated NPU device
label and stubs only framework format/context queries. It tests view offsets,
aliases, independent scalar/stream snapshots, ownership, invalid inputs and
imports with optional runtime dependencies forbidden.

`tests/st/runtime/kernel/test_torch_interop.py` checks actual NPU tensors,
non-default streams, offset views and noncontiguous-input rejection. It skips
when a real NPU is unavailable or the selected platform is a simulator. These
are metadata tests, not PyPTO kernel-execution tests.
