# Kernel-mode integration foundations

The public JIT eager entry borrows NPU arguments and submits through the optional
native torch_npu adapter. Explicit `.compile()` produces program objects.

## Public JIT eager entry

On this integration branch, `op(x, scale, out)` runs in kernel mode. The caller
supplies real NPU tensors and every Out/InOut argument; there is no decorator
mode argument or explicit kernel compilation step. On each call PyPTO validates
arguments, snapshots typed scalar values and the current stream, and rejects
graph capture before compilation or Worker initialization. The first valid call
compiles a kernel artifact, initializes the process Worker and prepares the
callable. Later matching calls reuse the artifact and registration. Different
operators share that Worker. Changing runtime scalar values or streams does not
recompile; changing a constexpr can select a different artifact.

```python
# op is a @pl.jit entry; the caller selected the current NPU device.
x = torch.ones((16, 16), device="npu")
out = torch.empty_like(x)
op(x, 2.0, out)
```

This entry currently supports A2/A3 with `tensormap_and_ringbuffer`, including
non-default streams and taskQueue enabled or disabled. Omitting `config` selects
that target and the current torch NPU device. An explicit `RunConfig` must match
the target and current device. Program-only diagnostics, ring overrides,
distributed configuration, CPU/Meta/Fake tensors and Worker-owned handles are
rejected; they do not select another execution path. Native launch requires
rank 1–5 and positive uint32 extents/strides. A5, HBG execution, ACLGraph,
and public torch.ops registration remain later work. Automatic eager cleanup is
verified with torch_npu 2.6.0.post2 as described below; other framework versions
are rejected before native kernel initialization until their teardown contract
is validated. This remains integration-branch functionality.

Host/simulator and distributed execution use explicit program compilation:

```python
program = op.compile(host_x, 2.0, host_out, config=program_config)
program(host_x, 3.0, host_out, config=program_config)
```

Compilation alone never claims a process execution mode. Executing a program
and executing a kernel require separate processes. Program and kernel artifacts
have separate cache identities. Formal program calls, including restored objects
and orchestration children, require all Out/InOut tensors and return `None`;
they do not allocate omitted outputs. Low-level explicit Worker APIs retain
their existing memory-management behavior. Kernel calls return `None` or exactly
the original tensor objects selected by the IR return aliases, without allocating
outputs or running a warmup invocation.

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
encoding consumes these values together with a validated kernel descriptor.
Addresses, scalar values and streams are per-call state, not compilation keys or
persistent metadata.

The frame retains each tensor and its storage object. `alias_result()` returns
`None`, one existing tensor, or a tuple of existing tensors. It never allocates
business outputs. Python ownership alone does not protect asynchronous device
use after the frame is released; native queue ownership and allocator stream
recording are supplied by the launch adapter described below. Callers must not resize or invalidate
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
PyTorch lacks `torch.library.register_fake`, the helper falls back to
`torch.library.impl_abstract` (available in PyTorch 2.2–2.3), preserving the
caller-owned library lifetime. If neither API is available, definition fails
before installing a schema. This optional helper requires one of these APIs;
the fallback does not add support for PyTorch 2.0–2.1 or change the package-wide
minimum dependency version.

The tests use temporary namespaces and CPU fixture implementations. On PyTorch
2.6, mutation-only schemas with no dispatcher return pass all
[`torch.library.opcheck`](https://docs.pytorch.org/docs/2.6/library.html#torch.library.opcheck)
checks and `torch.compile(backend="aot_eager", fullgraph=True, dynamic=True)`;
the test wrapper returns the caller's output tensor after the operator call.
Registered `torch.ops` tests verify that backed and unbacked integer symbols
reach the fake kernel unchanged without equality guards. A shape-derived scalar
test also verifies that different input sizes reuse one compiled graph.
API-selection tests emulate the older registration entry point on PyTorch 2.6
and verify Fake/Meta dispatch, duplicate rejection and library cleanup; they
do not establish end-to-end compiler compatibility on older PyTorch releases.
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

The internal launch path requires the optional native adapter. Public torch.ops registration remains separate work. Eager submission
rejects graph capture; it does not provide an ACLGraph lifetime contract.

## Process kernel Worker and registration

The internal `runtime.kernel.context.get_process_kernel_state()` owns one lazy
kernel Worker for the process. All operators share that manager; changing the
operator, scalar values or caller stream does not create another Worker.
`KernelConfig` fixes platform, runtime, device and AICPU thread count. Other
context resources currently use Simpler defaults. An incompatible configuration
is rejected instead of opening another Worker.

The integration SDK is pinned to
`b5a0ea0c941576e4e9c409b7be5130a607c4f9dc`. Its supported Python surface is
`simpler.task_interface.ChipWorker.kernel_init`, `kernel_prepare_callable` and
`finalize`; the proposed L2 `Worker(execution_mode="kernel")` API is not present.
PyPTO's private adapter uses these existing methods. Init and prepare take no
caller stream; Simpler mints the native context generation and callable IDs.
The calling thread must already hold the framework's current device. The native
lifecycle thread borrows that ACL context without creating or resetting a device. Init uses
installed runtime binaries and checks capability; it does not compile an
operator or allocate business outputs. HBG kernel initialization is unsupported
at this pin even though HBG binary compilation works.

The manager transitions through UNINITIALIZED, INITIALIZING, READY, FAILED,
CLOSING and CLOSED. Concurrent initialization shares one result. An init failure
retains its error and any partial Worker for cleanup, and never silently creates
a replacement. A PID check runs before taking inherited locks; a forked child
cannot use an initialized manager or its registrations. Forking before any
kernel initialization replaces only the unused Python state. Use a fresh spawned
process after native initialization.

`ensure_callable(artifact, config)` loads the artifact and hashes its full
serialized ChipCallable, tensor signature, target/runtime and PyPTO ABI
descriptor using Simpler's existing descriptor helper. It does not use an ELF
display hash, file path or Python object address as registration identity.
Concurrent requests for the same identity share one prepare result or error;
distinct identities register separately on the same Worker. A failed prepare
publishes no registration and can be retried. Prepare never warms up the operator
by executing it. Each `KernelRegistration` retains the callable, artifact and
manager, and validates its PID, manager generation and registry membership.
Native handles and registrations never enter the disk cache.

PyPTO program initialization paths (explicit chip/distributed Workers and
one-shot runners) claim program mode before native initialization; kernel init
claims kernel mode. The claim is process-wide and remains after failure or
close, so switching modes requires a separate process. Simpler supplies native
per-context mode checks and duplicate kernel-context rejection. Direct use of
third-party Simpler objects bypasses PyPTO's process gate; it is not a supported
way to combine program and kernel execution in one process.

## Automatic eager shutdown

PyPTO installs a versioned integration at torch_npu's existing shutdown boundary
before native Worker initialization. In torch_npu 2.6.0.post2,
[`_npu_shutdown`](https://github.com/Ascend/pytorch/blob/eef1d5ae62b9118ae78bf2d7084e6fba1b13058f/torch_npu/__init__.py)
calls `_C._npu_shutdown_synchronize()`, destroys process groups, and then calls
`_C._npu_shutdown(success)`. PyPTO wraps both native entry attributes: its one-shot
cleanup runs before the first synchronization, and the teardown wrapper also
covers an explicit teardown that bypasses that step. The original functions and
arguments are preserved. No additional `atexit(worker.close)` registration is
used, and importing PyPTO or compiling a program installs no hook.

The native framework
[teardown](https://github.com/Ascend/pytorch/blob/eef1d5ae62b9118ae78bf2d7084e6fba1b13058f/torch_npu/csrc/InitNpuBindings.cpp)
clears allocators and calls `NpuSysCtrl::Finalize`, which destroys events, streams
and devices. Kernel cleanup must finish before those operations. A native
`GetInitFlag()` check detects an already-finalized framework without initializing
it. Private shutdown entry points and this ordering are version-sensitive;
PyPTO currently admits only torch_npu 2.6.0.post2 for native kernel initialization.

Each native kernel Worker owns one persistent daemon lifecycle thread. Its
constructor/init, callable preparation and finalize execute on that thread with
the borrowed framework context, satisfying Simpler's init-owner-thread rule even
when the application's first caller was a short-lived background thread. Hot
launches stay on the existing native torch queue path. Successful close stops
and joins the lifecycle thread; a failed close keeps it alive for cleanup retry.
The thread is daemon because Python joins non-daemon threads before framework
exit hooks run; ordinary shutdown explicitly joins it before framework teardown.

The internal manager's `close()` can be requested from another thread. It first
stops admission, waits for initialization and in-flight preparation/admission,
drains accepted eager tickets, and finalizes on the native owner thread. Concurrent
close requests share the completed close, and repeated framework notifications
do not retry or double-finalize. Reentrant close from initialization/preparation
is rejected. Success clears registrations, invalidates handles and cannot
reinitialize. An unused manager creates no Worker and performs no native close.
Operator garbage collection does not close the process Worker.

Failed drain/finalize keeps ownership and registration records. If cleanup cannot
complete, or the framework is already torn down, automatic cleanup reports a
warning, stops admission, and anchors the manager with a deliberately unreleased
native Python reference. This prevents later interpreter/module clearing from
running Worker/event destructors against a destroyed ACL context. Framework
shutdown still proceeds; this is a failure fallback, not proof of safe resource
release. A proven successful cleanup that reports an earlier submission error
is reported separately. Forked children never close an inherited Worker. Abrupt
termination (`os._exit`, signals, interpreter crashes) provides no cleanup promise.

This contract covers eager work only. Graph capture remains rejected, so no
PyPTO graph can replay a callable after this boundary. PR-08 must establish graph
stop/release ordering before enabling capture; device synchronization alone
cannot prevent later external replay. There is no public close/shutdown ritual.

## Internal torch queue submission

`pypto.torch.launch.enqueue(registration, args)` takes a registration from the
process manager and complete arguments in logical signature order. It validates
the registration and describes a fresh frame, then returns the declared output
aliases after host admission. Return does not mean device completion. It never
compiles, prepares a callable, creates a Worker or allocates business outputs.

The optional `pypto._torch_npu` module constructs `ChipStorageTaskArgs` using the
pinned SDK headers. Tensor and scalar pools are independent: a mixed signature
`(x, scale, out)` produces two tensor entries and one scalar slot. The assembled
kernel ChipCallable includes `IN, OUT, SCALAR`, including in restored binary
manifests. Generated program tensor-direction metadata remains unchanged.
Scalars copy zero-extended object bytes, preserving float bits and signed integer
width. Tensor addresses point at the logical view, including storage offsets.
The current native launch requires rank 1..5, positive u32 extents/strides and
base-format tensors; empty views remain valid metadata but cannot launch yet.

`OpCommand::RunOpApiV2` places the native callback in the framework's current
stream queue. The callback invokes the public C++ `ChipWorker::kernel_launch`;
it executes no Python, JIT or prepare. The same callback runs inline when the
framework disables taskQueue. The adapter captures native Tensor and Storage
owners, a copied argument POD, callable ID and current stream. It does not cache
the first call's stream or read a queue-draining Python `npu_stream` property.
Capture-state inspection uses the borrowed stream without draining the queue.

The process manager retains every native ticket **before** enqueue, serializes
host admission and refuses stale registrations. Native storage owners cover
the delayed host callback. Allocator `recordStream` is applied once per unique
storage before submission, including aliased inputs/outputs. A completion event
recorded after Simpler's caller-stream join covers device use. Later calls reap
completed tickets without draining the host queue; internal `state.drain()` or
`close()` waits for outstanding work. A retained last ticket is
released at that drain/close boundary.

Submission and asynchronous callback errors propagate through the framework and
ticket wait. Failed or partially enqueued tickets retain their Worker, argument
and storage owners: a failed stream wait does not establish that Simpler's
internal streams are quiescent. Internal close drains the host callback and, only on failure,
requires a full device synchronization to establish internal-stream quiescence.
It then finalizes and releases owners, while re-raising the original submission
error. Failed quiescence or teardown retains all owners for a later close attempt.
There is no implicit reinitialization. Automatic framework shutdown remains
follow-up work; the internal manager supplies the drain/cleanup boundary.

### Building the optional adapter

The ordinary build defaults `PYPTO_BUILD_TORCH_NPU=OFF` and does not discover or
link torch_npu. Importing `pypto.torch.launch` remains safe without the extension;
a real enqueue reports how to enable it when absent.

```bash
# Use the same Python environment/compiler ABI for PyPTO and Simpler.
source .claude/skills/testing/load-env.sh
# Source the installed CANN set_env.sh to set ASCEND_HOME_PATH.
cmake -S . -B build -DPYPTO_BUILD_TORCH_NPU=ON
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
```

The adapter is compiled against the installed torch/torch_npu headers and
libraries and CANN from `ASCEND_HOME_PATH`. The supported build requires the
C++11 libstdc++ ABI and compatible nanobind builds. The adapter embeds and checks
the exact Simpler revision. Simpler's Python module hides its C++ symbols, so
the adapter compiles the pinned SDK Worker implementation directly and uses
nanobind's registered `ChipWorker` type. It neither redefines the runtime ABI
nor extracts private context pointers; it creates no additional Worker. Rebuild
both modules together after changing the SDK, framework or compiler ABI.

The hardware test's deterministic host-queue gate is a separate, non-installed
module, enabled only with `-DPYPTO_BUILD_TORCH_NPU_TESTS=ON`. Add
`build/torch_npu_tests` to `PYTHONPATH` when running that test. No fault-injection
or queue-blocking entry is shipped in the production adapter.

## Verification

`tests/ut/torch/test_interop.py` uses real CPU storage with an emulated NPU device
label and stubs only framework format/context queries. It tests view offsets,
aliases, independent scalar/stream snapshots, ownership, invalid inputs and
imports with optional runtime dependencies forbidden.

`tests/st/runtime/kernel/test_torch_interop.py` checks actual NPU tensors,
non-default streams, offset views and noncontiguous-input rejection. It skips
when a real NPU is unavailable or the selected platform is a simulator. These
are metadata tests, not PyPTO kernel-execution tests.

`tests/ut/runtime/test_kernel_context.py` covers shared initialization and
registration, configuration conflicts, shared prepare failures, stale/forked
handles, ownership, owner-thread shutdown and retry. The isolated cases in
`tests/st/runtime/kernel/test_kernel_context.py` exercise real A2/A3 TRB
init/prepare/close with two DSL callables, duplicate native kernel-context
rejection and HBG capability refusal. They require the pinned runtime binaries
and a reserved NPU. They do not launch a PyPTO kernel or validate capture.

`tests/ut/torch/test_launch.py` checks dispatch, scalar encoding, aliases and
missing/incompatible extension errors. Manager tests cover retention, reaping,
failed admission and close ordering. `tests/st/runtime/kernel/test_torch_launch.py`
uses a generated scalar-bearing DSL callable with taskQueue enabled/disabled,
non-default streams, framework A → PyPTO → B ordering, offset views, changing
scalars, GC/allocation pressure, owner-thread close, a deliberately blocked host
callback and native error injection. These require a reserved A2/A3 NPU and
both locally built adapters. They do not claim A5 or ACLGraph acceptance.

`tests/ut/torch/test_registration.py` covers schema mutation/alias contracts,
Fake/Meta and symbolic inputs, isolated imports, duplicate definitions, and
test-only dispatcher/compiler integration without a real kernel executor.

`tests/ut/jit/test_kernel_eager.py` checks public entry routing, preflight rejection, scalar snapshots and cache separation. `tests/st/runtime/kernel/test_jit_eager.py` verifies real repeated InOut updates, constexpr variants, one shared Worker and isolated explicit program execution.

`tests/ut/runtime/test_kernel_shutdown.py` checks lifecycle affinity, initialization
races, idempotent close, framework ordering, failure retention, version refusal and
fork handling. `tests/st/runtime/kernel/test_kernel_shutdown.py` uses ordinary
Python subprocess exit (not multiprocessing's `os._exit`) with no user close or
drain. It covers taskQueue on/off, a departed first-caller thread, two operators,
delayed host callbacks, partial initialization, repeated notifications and a
failed finalize retained through framework teardown.
