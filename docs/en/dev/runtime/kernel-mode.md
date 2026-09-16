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
- Empty views access no elements, so their nonnegative storage offsets may
  exceed storage capacity. The accessed-range upper bound applies only to
  nonempty views.
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

## Internal schema and Fake/Meta helpers

`pypto.torch.registration.RegistrationSignature` copies the same `ParamInfo` carrier
shapes and return-slot indices. Its `schema(name)` method marks Out/InOut tensor
arguments writable and connects each tensor return to the corresponding input
alias set. All arguments remain required; no output allocation is inferred.
Scalar inputs map to dispatcher `SymInt`, `float` or `bool`. `SymInt` accepts
ordinary integers and preserves symbolic integers through dispatch, including
symbols without concrete hints. Scalar validation retains dtype range checks
without converting symbols to Python integers. Read-only input
identity returns, scalar outputs, scalar-only operators, invalid names and
aliases, and UINT64 scalars are rejected:
the dispatcher's signed integer type cannot represent the full UINT64 range.
Return aliases must name Out/InOut tensors: the dispatcher schema checker
rejects returning a read-only input object directly.

`fake(*args)` accepts FakeTensor or Meta tensors, validates dtype, rank, static
shape, contiguity, device consistency and inference-only use, then returns the
exact declared input objects. Dynamic dimensions and symbolic integer scalars
remain symbolic. Returning an input preserves its strides, storage offset and
alias identity, including empty slices. This helper never reads storage or data
pointers, queries NPU formats/device/stream, allocates business outputs or invokes
a Worker. Physical NPU format and overlapping-storage checks stay in the real
call adapter because abstract tensors do not establish those facts.

`define(library, name)` installs only the schema and fake kernel into a
caller-owned `torch.library.Library`. The caller must keep that library alive
and owns its registration lifetime. Repeated definitions, including a different
signature with the same name, raise PyTorch's duplicate-definition error;
existing definitions are not replaced. Importing or reloading this module does
not register an operator. `pypto.torch` exports no new public registration API,
and PyPTO installs no real device kernel in this foundation. If the installed
PyTorch lacks `torch.library.register_fake`, definition fails before installing
a schema.

The tests use temporary namespaces and CPU fixture implementations. On PyTorch
2.6, mutation-only schemas with no dispatcher return pass all
[`torch.library.opcheck`](https://docs.pytorch.org/docs/2.6/library.html#torch.library.opcheck)
checks and `torch.compile(backend="aot_eager", fullgraph=True, dynamic=True)`;
the test wrapper returns the caller's output tensor after the operator call.
Registered `torch.ops` tests verify that backed and unbacked integer symbols
reach the fake kernel unchanged without equality guards. A shape-derived scalar
test also verifies that different input sizes reuse one compiled graph.
Schemas with aliased returns are checked for schema correctness and Fake/Meta
behavior separately. These checks do not establish functionalization or compiled
execution of aliased-return operators. Actual kernel registration, device
execution, autograd and the final compiler integration remain later work.

## Optional dependencies and scope

`import pypto.torch` exports no execution API. Importing it or its `interop`
or `registration` module does not request torch_npu, Simpler or a native launch
extension.
`torch` remains a normal PyPTO dependency. A real call description loads
`torch_npu` on demand and reports a targeted error if it is unavailable.

This foundation covers metadata validation, Python call-frame ownership, and
internal schema/Fake helpers.
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

`tests/ut/torch/test_registration.py` covers schema mutation/alias contracts,
Fake/Meta and symbolic inputs, isolated imports, duplicate definitions, and
test-only dispatcher/compiler integration without a real kernel executor.
