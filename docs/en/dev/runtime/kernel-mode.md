# Kernel-mode integration foundations

The internal torch adapter describes borrowed NPU arguments for a future kernel
executor. It does not add a public kernel execution entry point. Existing JIT,
compiled-program and Worker calls share the parameter contract described below.

## Shared signature and L2 ABI

`ParamInfo` is the immutable parameter snapshot shared by program execution,
Torch call validation, registration metadata and assembly. Shape lists become
tuples; JSON retains arrays. Shared helpers validate unique names and external
return aliases. Aliases index the complete signature (including scalar slots)
and may repeat. L2 pool capacities are checked separately at the single-chip
boundary, so distributed metadata is not restricted by those limits.

There is no separate kernel ABI descriptor or SDK field. Compilation identity
uses the existing `ArtifactKey` and `ToolchainIdentity`, including the requested
PROGRAM or KERNEL contract. Parameter dtype/layout serialization remains shared.
IR construction and passes own parameter legality; call adapters validate scalar
ranges and borrowed storage at their boundaries.

## Shared compilation product

The internal `CompiledArtifact` owns one single-chip directory, platform,
`ArtifactMetadata` and lazy callable loading. `CompiledProgram` composes it and
retains live IR, path compatibility, program return-style allocation and the
program runner. Parameter names, output allocation slots and return count are
computed from shared metadata, not stored again in the wrapper. Target accessors
use the artifact's effective platform, including explicit reload overrides.
Multi-orchestration children use the same product; the parent retains entry
selection. Distributed execution retains its existing wrapper. The shared
compiler writes each sidecar once; wrapping its result does not rewrite it.
Legacy direct construction from live IR still creates a sidecar for compatibility.

Public `ir.compile()` and internal `_compile_artifact(contract=KERNEL)` run the
same passes and codegen through `_compile_to_files()`, which returns lowered IR
and build metadata without an execution wrapper. `_compile_artifact()` accepts
one single-chip entry and always returns `CompiledArtifact`. Public `compile()`
owns single-program, multi-entry and distributed wrapping, including the legacy
fallback for programs without a canonical lowered signature.
Public `compile()` explicitly requests PROGRAM; KERNEL
is currently requested directly by internal compilation tests. Torch submission
has not yet been connected to that path. There is no public `compile_kernel()`.
The `contract` selects PROGRAM/KERNEL calling rules; `execution_mode` independently
selects SIM/ONBOARD for the target platform. Kernel validation requires one chip
and entry, caller-supplied Out/InOut buffers, and tensor returns
that alias external parameters. The existing C++ return-lineage analysis derives
the ordered aliases from the original IR, including duplicates. This contract
checks the external entry signature, without a separate internal write/alias scan.
Internal submits may omit runtime-allocated Out parameters. Post-lowering
checks validate the external parameter ABI: count, order, names, directions,
dtype and storage. SSA version suffixes are removed from external parameter names.
HBG host tensor data reads/writes are rejected; device submit
predicates retain their existing semantics. Program compilation keeps its
allocation and host-control behavior.

Metadata distinguishes `params` (logical caller signature) from `lowered_params`
(assembly signature); both use `ParamInfo`. PROGRAM argument binding and output
allocation use the original, pre-lowering `params`, including after reloading
an artifact and when selecting a multi-orchestration child. `lowered_params`
remains available for assembly ABI validation. Torch validates logical shapes.
NZ lowering is checked against the unpadded blocked representation in element
units with unchanged dtype, preserving byte size. Dynamic NZ supports the logical
`[B, R, C]` batch dimension. A blocked shape is never treated as a Torch shape.

`compiled_meta.json` schema 3 persists both signatures, return count, aliases,
platform/backend and verified execution capabilities. Producers declare only the
requested capability, never implicit dual-mode support. Schema 1/2 program
metadata can reload as PROGRAM only; it does not establish a kernel contract.
Missing kernel fields, invalid indices or mismatched capabilities require
recompilation. `from_dir()` reads metadata without loading native code.

`ArtifactRuntime` remains the sole cache promotion/loading engine. Single-chip
cache restoration first creates a `CompiledArtifact`, then wraps it for a PROGRAM
consumer. The internal KERNEL consumer receives the artifact directly. Manifest,
build locks, prebuilt validation and stage promotion are shared. Loading is
serialized, publishes success only after assembly, and does not initialize,
register with or execute a Worker. Cache promotion preserves source-path identity;
run output remains outside the immutable cache. Handles, streams, input tensors
and per-run resources never enter the common product.

Program submission builds address-free `TaskArgs`; Simpler materializes those
into `ChipStorageTaskArgs`. The future Torch adapter will also submit `TaskArgs`
through the public Worker interface, using `Buffer.wrap` and Simpler Tensor views
for borrowed device storage. Scalar values and types will use the public
`TaskArgs` typed scalar interface, leaving ABI encoding to Simpler.
`ScalarArgument` retains the validated value and dtype without encoding it.

Runtime capability checks run at explicit Torch kernel initialization. The lifecycle
layer supports Worker initialization and callable registration; JIT submission
and the public registration/DFX APIs remain separate migration steps.

## Process initialization and lifecycle

The internal `runtime/kernel` package groups code by ownership:

- `session.py`: configuration, runtime capability checks and `_BorrowedWorkerSession`,
  which binds lifecycle operations on a public `simpler.Worker` to the Torch context.
- `context.py`: process state, concurrent registration reuse and pending-ticket cleanup.
- `owner.py`: the lifecycle owner thread.
- `callable.py`: shared artifact identity and context-bound registration records.
- `pending.py`: callback result publication and per-call resource retention.

The session is a lifecycle adapter, not a separate execution engine. Simpler owns
callable registration and execution; the process manager owns admission and caches.

`pypto.torch.init()` explicitly borrows the current torch_npu device context. It
supports onboard `a2a3` with `tensormap_and_ringbuffer` or `host_build_graph`.
An explicit `device` must equal the framework's current device; initialization
never selects a device. Graph capture raises `NotImplementedError` before a
process mode is claimed. Shared `CallConfig.validate()` checks native ring-size
constraints during preflight. Invalid configuration, missing capabilities and stale
runtime identity also fail during preflight. Internal/runtime exceptions retain
their original types.

The configuration includes AICPU threads, swimlane/dep-gen settings, output
directory, ring task-window/heap/dependency-pool sizing and an optional
`workspace_budget_bytes`. Ring scalars broadcast to four scope-depth rings.
Diagnostics require an output directory; this step reserves the configuration,
while public DFX windows are a later step. The workspace budget follows the
public Worker's minimum of 1024 bytes and excludes borrowed tensor storage.

The runtime submodule follows upstream's Simpler pin
`6b6b88f2353cca12cfafe809750b50b59931b0d1`. The existing `check_runtime_pin`
policy remains authoritative for the installed native binding, including its
missing-identity and explicit-skip rules. A separate availability check requires
public borrowed init, caller-stream submit, Buffer/TaskArgs, RunHandle and
`flush_diagnostics` interfaces. It never consults the obsolete kernel supported
bit or private Worker slots. Interface checks and UT do not establish device
execution correctness.

One manager owns a Worker and a nonzero context-generation token per process.
Its states are UNINITIALIZED, INITIALIZING, READY, FAILED, CLOSING and CLOSED.
Identical configuration is idempotent; conflicts and recursive initialization
fail. Concurrent initialization shares one result. Native initialization failure
retains the Worker and the process mode for cleanup. Closing is terminal: it does
not permit switching to program execution. Program mode is claimed before
runner execution (including DFX dependency-capture subprocesses), and at direct
Worker initialization, not at import, compilation or metadata loading.
One-shot distributed execution claims program mode before changing host tensor
storage to shared memory or clearing prior dispatch diagnostics. A child
forked after kernel initialization must use a fresh spawned process; an untouched
manager is replaced once in the single-threaded at-fork child callback, including
its inherited locks, before concurrent first use.
The mode gate is also reset in an at-fork child callback when the parent was
unclaimed or in PROGRAM mode. Competing PROGRAM/KERNEL claims in that child
share one lock: one mode wins and the other raises before native Worker
initialization. An inherited KERNEL claim is never reset.

The lifecycle owner thread binds the borrowed RTS context before public
`Worker(level=2).init(prewarm_config=..., borrowed_context_generation=...)`,
`Worker.register(callable)` and `Worker.close()`. Simpler owns runtime binaries,
slots and native cleanup. Registration loads `CompiledArtifact` through its
existing shared loader, requires KERNEL capability and matching platform/runtime,
and retains its opaque `CallableHandle`. The existing `ArtifactKey` keys cached registrations; private products use object
identity because their directories can be overwritten. No new digest is defined.
Concurrent preparations share success or failure. Busy
registry and capacity errors propagate without drains, waits or eviction.

Pending tickets retain the registration and per-call owners before enqueue. A
future callback adapter publishes its `RunHandle` to the ticket without taking
manager locks. Ordinary admission only polls completed tickets. An unpublished
callback is incomplete; a true `RunHandle.done` still requires `wait()` to retire
native resources before borrowed sources are released. Explicit drain/shutdown
may synchronize the framework queue, then retire runs and release sources before
closing the Worker. A failed callback, retirement or source release retains the
ticket and Worker; cleanup does not assume device quiescence.

The optional `_torch_npu` extension supplies only lifecycle/context and NZ
storage queries in this step. Build it with `-DPYPTO_BUILD_TORCH_NPU=ON` using
matching Torch, torch_npu and CANN headers/libraries. The configure-time C++11
ABI check also runs when `PYTHONOPTIMIZE` is enabled. It links no Simpler C++
implementation or separate kernel ABI. The default core build does not require
these optional dependencies. Shutdown wraps the verified framework teardown
hooks (torch_npu `2.6.0.post2` and `2.10.0.post2`, allowing build suffixes), closing
PyPTO before framework teardown. Unsafe cleanup retains a native lifetime anchor
through process exit rather than running destructors after context destruction.
In a forked child, shutdown anchors the inherited state before framework teardown
without acquiring inherited locks or closing the parent's Worker.

Direct `@pl.jit` calls still enter program execution. Consequently, invoking
one after `pypto.torch.init()` is rejected by the process mode gate in this
intermediate migration stage. Initialization is independently testable; the
JIT-to-callback-to-`Worker.submit` connection belongs to the next step. No
public `compile_kernel`, shutdown, registration or DFX function is added here.

## Call metadata and ownership

[`CallSignature`](../../../../python/pypto/torch/interop.py) retains the shared immutable
`ir.param_info.ParamInfo` signature. `describe_call(args)` uses
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
encoding will consume these values together with validated artifact metadata.
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
- An explicit `layout="NZ"` and `platform="a2a3"` permit native format 29 for
  read-only FP16/BF16/INT8 inputs. Physical dimensions must match the complete,
  unpadded NZ matrix, with zero storage offset. The physical descriptor query is
  lazy and requires the optional native adapter; tests replace only that query.
  Signature validation compares the logical Torch shape directly with logical
  `ParamInfo`; the physical query separately checks blocking and byte capacity.
- Empty views access no elements, so their nonnegative storage offsets may
  exceed storage capacity. The accessed-range upper bound applies only to
  nonempty views.
- Accept exact byte-span aliases, including different dtype/shape views, and
  disjoint views of shared storage.
  Overlapping read-only views are allowed. Partially overlapping views involving
  Out/InOut require a richer alias contract and are rejected.
- Copy Python or correctly typed ctypes scalars using the program scalar type
  map. Integer values must fit their declared type; floats are not silently
  truncated into integers; finite floating-point overflow is rejected. This does not change JIT runtime-scalar or constant
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

`pypto.torch.registration.RegistrationSignature` retains the same immutable `ParamInfo` carrier
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

`import pypto.torch` exports `init`; it does not expose kernel submission. Importing it or its `interop`
or `registration` module does not request torch_npu, Simpler or a native launch
extension.
`torch` remains a normal PyPTO dependency. A real call description loads
`torch_npu` on demand and reports a targeted error if it is unavailable.

This foundation covers shared compilation, metadata and cache loading, Python
call-frame ownership, internal schema/Fake helpers, and process initialization.
It does not claim kernel launch, taskQueue ordering, allocator safety, eager
numerical execution or ACLGraph support. Those need the native adapter and
runtime integration before a public entry-point switch.

## Verification

The shared `run_without_optional_runtime` unit-test fixture runs a fresh Python
process with `torch_npu`, `simpler`, `simpler_setup`, `_task_interface` and
`pypto._torch_npu` blocked by an import finder. It detects both import statements and dynamic
`importlib.import_module()` calls, including attempts whose `ImportError` is
caught by the caller. Explicit enforcement checks remain active under
`PYTHONOPTIMIZE=1` and `2`; negative controls cover both settings and normal
execution. PyTorch backend autoload is disabled in that subprocess
to isolate PyPTO's behavior from installed framework plugins. The guard's own
negative tests ensure an attempted import cannot silently pass on a CPU runner
where the dependency is already absent.

These checks cover package import/reload, program configuration and registered
Fake/Meta dispatch. They run in the existing full unit-test CI job without
optional runtime dependencies or a new device job. They do not validate native
adapter builds or device execution; those require the integration branch.

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
