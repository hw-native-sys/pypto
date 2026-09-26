# Persistent JIT Cache

Persistent caching is opt-in. It reuses generated code and complete binaries
across processes, while each JIT function also retains live compiled objects.
Cached artifacts contain executable code: use a cache with trusted writers.

```python
from pathlib import Path

import pypto
from pypto.runtime import RunConfig
from my_kernels import decode

pypto.configure_cache(pypto.CacheConfig(
    enabled=True,
    root=Path("kernel-cache"),
    extra_source_paths=(Path("kernels"),),
    extra_fingerprint="model-config-v1",
))
prepared = decode.warmup(config=RunConfig(platform="a2a3"))
print(pypto.cache_stats())
```

`decode` has complete tensor annotations. Alternatively, pass sample tensors and
scalars using the ordinary `compile()` argument rules.
Warmup builds every required binary without initializing an NPU or executing a
kernel. The build host still needs the target compiler, SDK and host runtime.

## Request flow

1. Capture specialization, source namespaces, effective compiler options and
   cache policy. Diagnostics and explicit output requests compile afresh.
2. Establish full source, application-input and toolchain content identities.
   Unsupported or unreadable identity inputs report a bypass.
3. Look for a compatible live object. Persistent objects additionally match the
   captured cache root and read-only policy.
4. Validate GENERATED metadata and check its complete READY specification.
   Prefer READY; restore generated code if binaries have not been published.
5. On a miss, build privately under the per-key transaction, package extern
   inputs, recheck mutable sources, and publish immutable GENERATED output.
6. Execution or warmup completes the runtime's binary transaction and publishes
   READY. Later processes can load READY without running a compiler stage.

`compile()` does not execute or promise complete binaries. Ordinary execution
publishes missing binaries automatically; callers need no separate load/store.
`specialize()` and `lower()` always produce IR directly. A fresh compiled object
retains `.program`; a restored object has `.program is None`, including when
another process wins the build race. Disable persistence when `compile()` must
return IR. Disabling persistence selects compatible private objects separately.

The store deduplicates concurrent writable builds across cooperating processes.
Private fallback results are shared only among overlapping calls in one process;
invalid/unwritable storage and read-only misses do not provide cross-process
private-build deduplication. Compiler errors propagate and can be retried.
Unsupported extern packaging and changing application sources stay private.

## Program build ownership

`pypto.runtime.kernel_compiler.KernelCompiler` owns compiler invocations,
linking, temporary output directories and binary validation. It consumes
metadata from the installed Simpler SDK without inheriting its compiler or
calling its build methods. This works with the existing runtime pin; it does
not introduce kernel execution or change program call semantics.

| Previous inherited responsibility | Current owner |
| --------------------------------- | ------------- |
| SDK root, tool selection, target flags, runtime headers and helper sources | Simpler metadata queries, consumed by PyPTO |
| AICore compilation and `kernel_entry` linking | PyPTO `KernelCompiler.compile_incore` |
| Simulator kernel shared libraries | PyPTO `KernelCompiler.compile_incore` |
| Orchestration shared libraries, Build-ID and host threading flags | PyPTO `KernelCompiler.compile_orchestration` |
| Temporary output validation and cleanup on success/failure | PyPTO; optional `build_dir` selects the temporary parent, not retained intermediates |
| Callable assembly, binary publication and restoration | Existing PyPTO device runner, prebuilt loader and artifact store |

HBG orchestration uses the host compiler. TRB uses the host compiler for
simulation and the AArch64 compiler for onboard targets. Required SDK helper
sources must exist; a missing helper fails before invoking the compiler.
Compiler commands retain the SDK's relative path spelling and working directory.
All generated outputs reside in PyPTO-owned temporary or artifact directories.
Build and restoration do not initialize a Worker or execute business logic.

Each compiler or linker invocation has a 900-second timeout. Set
`PYPTO_COMPILER_TIMEOUT` to a positive, finite number of seconds before creating
the compiler to override it. This is a limit per invocation, not for the whole
program build. A timeout raises `RuntimeError` identifying the build stage;
temporary outputs are cleaned up and failed builds do not publish a cache stamp.

Mutable program output directories use binary-context schema 2. A successful
transaction records the context plus SHA-256 hashes of reusable binary files.
The next transaction preserves verified files and discards changed or unrecorded
files; missing files are rebuilt on demand. Old stamps require one rebuild.
The stamp is removed before assembly and published again only after success,
so failed or interrupted transactions cannot authorize partial output.

Persistent GENERATED entries still provide source identity only: promotion
builds their binaries privately. Inherited mutable binaries are never trusted
as READY evidence. Complete READY entries use their existing validated inventory
and restore without compiler invocation or cache writes, including read-only
restoration. Per-directory and per-key locks retain the existing concurrency
contract; no second cache store is introduced.

## Configuration

`CacheConfig` is immutable. Complete per-call `RunConfig.cache_config` objects
replace process defaults from `configure_cache()`, which replace environment
settings. Fields are never partially merged across those levels.

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `enabled` | `False` | Enable persistent lookup and publication. |
| `root` | `None` | Use `~/.cache/pypto/jit`; explicit relative paths resolve when the request is captured. |
| `readonly` | `False` | Prohibit writes, locks and bytecode under the cache root. Private builds and runtime output remain outside it. |
| `extra_source_paths` | `()` | Content-hash files, or recursively hash Python sources in directories, on every request. Missing inputs bypass reuse. |
| `extra_fingerprint` | `None` | Additional application revision/configuration token. It cannot replace missing toolchain evidence. |

Environment configuration uses `PYPTO_CACHE`, `PYPTO_CACHE_DIR`, and
`PYPTO_CACHE_READONLY`. Booleans accept exactly `0` or `1`; invalid values raise
`ValueError`. `configure_cache(None)` restores environment/default precedence.
An in-flight request keeps its captured policy. Configuration changes neither
clear statistics nor delete files.

```python
readonly = RunConfig(cache_config=pypto.CacheConfig(
    enabled=True,
    root=Path("kernel-cache"),
    readonly=True,
    extra_source_paths=(Path("kernels"),),
    extra_fingerprint="model-config-v1",
))
compiled = decode.compile(config=readonly)
with_ir = decode.compile(config=RunConfig(cache_config=pypto.CacheConfig(enabled=False)))
```

Repeat application identity inputs when changing storage policy. A read-only
miss can build privately, so successful warmup alone does not prove publication.
Cache policy is consumed by JIT before object selection; it is not forwarded
to compiler or per-launch options. Private output normally uses `build_output`
in the working directory. If that is inside the cache, an unpredictable 0700
temporary parent isolates both private builds and runtime output, without probing
`TMPDIR` for writability. The fallback parent may be created on a cache hit.
Runtime directories and private build trees remain alive with compiled objects;
there is no online cleanup or eviction API. Stop all consumers before offline
removal of cache entries or lock files.

## Toolchain support and cost

The default `PYPTO_CACHE_IDENTITY=build` policy identifies the effective tools
used for compilation. It compares those identities with the artifact's build
identities; it does not add an independent expected-version or pin audit.

| Component | Identity evidence |
| --------- | ----------------- |
| PyPTO | Package Python source and bundled codegen template contents, the actually imported native extension's full GNU ELF Build-ID, and Python version/ABI. |
| Runtime | A clean source checkout revision or installed build revision, actual native extension Build-ID, runtime Python sources, and available runtime/PTO-ISA build metadata. A dirty source checkout bypasses persistence. |
| PTO-ISA | The revision selected by the runtime's `pto_isa.pin`. Checkout acquisition and validation happen on compilation misses. |
| PTOAS | For a standard wheel launcher: metadata from its selected interpreter's package, its selected NumPy wheel record, native compiler Build-ID, interpreter startup `.pth` files and non-stdlib startup modules. Missing startup or wheel evidence and unsupported launchers bypass persistence. Standalone ELF builds use Build-ID. |
| Device and orchestration tools | Selected compiler paths/versions, the invoked executable's and executed GCC driver's Build-IDs, and GCC helper Build-IDs, plus CANN installation build version and linker Build-ID. Unrecognized compiler wrappers or missing CANN build versions bypass persistence. |

Native files without a usable Build-ID fall back to content hashing. Build IDs
are read from small ELF notes, not by reading the complete shared object. Wheel
PTOAS discovery does not import its compiler package. The selected interpreter's
startup search path and hook files also participate in the key. Missing
evidence bypasses persistence; it never creates an `UNKNOWN` cache key.
Existing runtime ABI and minimum PTOAS compatibility checks remain in place
on their normal paths.
PTOAS packages with an `_online` build directory bypass persistence because a
local extension rebuild can leave the reported version unchanged.

Python roots and native origins are resolved from actual imports. This supports
`pip install`, `pip install -e`, and `PYTHONPATH`, including source Python paired
with an editable installation's native extension. Distribution metadata alone
never identifies the imported PyPTO compiler.

This policy trusts published native build/version identifiers. It does not scan
system headers, CPython's standard library, or transitive dynamic libraries.
Source runtime checkouts must have no tracked or untracked Git changes. After
patching installed inputs without changing their published identity, set a new
`PYPTO_CACHE_EPOCH` value or clear the cache. Installed files must remain
immutable within a process; restart after
replacing them. Application extra sources are still refreshed on every request.
Effective selection inputs, including paths and environment overrides, remain
part of the key, so moving an installation can cause a miss.

`PYPTO_CACHE_IDENTITY=content` retains the previous Linux dependency inventory:
ELF dependency closures, compiler resources, implicit includes, link inputs and
Python/native runtime contents. It retains the existing PTOAS/CANN reported
version and verified PTO-ISA revision shortcuts; it is not a byte audit of every
vendor installation. Unsupported launchers, sanitizer builds, and unmodeled
implicit overrides such as `CPATH` or `LD_PRELOAD` bypass persistence. Both
policies report bypass reasons in `cache_stats().last_bypass_reason`. The
policies have separate identities and do not reuse one another's entries.

Artifact payloads still undergo complete manifest/content validation. Newly
packaged artifacts carry `kernel_config.json` so READY discovery does not execute
Python configuration. Restoration reuses the lookup's validated manifest and
constructs native callables without importing Worker/communication setup.
If the GENERATED slot is missing or damaged, lookup searches at most 32 READY
stage directories under the exact artifact key. Each candidate's own JSON must
derive its directory's spec digest and pass full manifest/payload validation;
multiple valid candidates are rejected.

Measure first hits in independent processes with:

```bash
PYTHONPATH=python python tests/benchmarks/jit_cache_latency.py \
  --cache-root /tmp/pypto-jit-benchmark --runs 5 --output build/jit-latency.json
```

Use an installed environment's Python and omit `PYTHONPATH` to test a wheel.
The first child populates the cache; every measured child must report a READY
hit with zero builds and bypasses. The interval includes first identity capture
and first warmup/callable restoration, excluding Python process startup, initial
imports, tensor creation, and device execution. Slow fallback probes, cold
filesystem pages, or large artifacts can exceed 100 ms; this is a measurement
target, not a latency guarantee. No cross-process timestamp-based identity memo
is used.

## Statistics and CLI

`cache_stats()` returns an immutable, thread-safe, process-local snapshot. It
never scans disk or resets counters. Counters include `requests`, `object_hits`,
`ready_hits`, `generated_hits`, `misses`, `invalid_entries`, `storage_errors`,
`generation_builds`, and `binary_builds`; time totals are `lookup_ns` and `build_ns`.
The following fields distinguish why persistence was not used:

| Field | Meaning |
| ----- | ------- |
| `disabled_requests` | Requests with persistence disabled, including private object hits. |
| `forced_rebuilds` | Diagnostic or explicit-output requests that force compilation, regardless of policy. |
| `bypasses` | Enabled requests that fall back because identity or packaging is unavailable or sources changed. |
| `last_bypass_reason` | Latest enabled-cache bypass diagnostic, or `None` before any bypass. |

Disabled requests and forced rebuilds do not increment `bypasses`. These counters
are not mutually exclusive: a disabled diagnostic request increments both
`disabled_requests` and `forced_rebuilds`. Initial lookups count once per request;
lock rechecks do not add requests. A miss that later encounters unsupported
packaging also records a bypass. Invalid/storage events are additional counters;
storage failures use typed results independently of diagnostic wording. Build
counts record actual stages, and timings exclude device execution. Compare numeric
fields between snapshots for intervals; `last_bypass_reason` is cumulative context.

```bash
python -m pypto.jit warm --module my_kernels --config warmup.json
python -m pypto.jit stat --root kernel-cache
```

```json
{
  "schema_version": 1,
  "cache": {"enabled": true, "root": "kernel-cache"},
  "requests": [
    {
      "kernel": "decode",
      "run_config": {"platform": "a2a3"},
      "tensors": {"x": {"shape": [1, 128], "dtype": "FP16"}},
      "scalars": {"block_size": 128}
    }
  ]
}
```

The CLI imports the named trusted module and resolves only explicitly named
module-level JIT functions. It validates the complete list before any build.
Paths are relative to the configuration file. Tensor metadata uses allocation-free
meta tensors; omit it when annotations fully determine tensor parameters. Dtypes
are `FP16`, `BF16`, `FP32`, `INT8`, `INT16`, `INT32`, `INT64`, and `BOOL`. Scalars
are finite JSON numbers or booleans; they complete the binding but no longer
select an artifact, since a scalar parameter is a runtime value. Omit them
unless the request also names sample tensors, which makes the binding positional.

Serializable `run_config` fields are `platform`, `strategy`, `memory_planner`,
`distributed_config`, `dump_passes`, `dump_ptoas_passes`, `save_kernels`, and
`save_kernels_dir`; enums use their Python member names. Distributed settings
accept `device_ids`, `num_sub_workers`, `runtime`, and `aicpu_thread_num`.
Warmup reports shared versus private preparation and exits nonzero on failure.
`stat` reads JSON manifests and declared payload sizes without importing cached
code, acquiring write locks or repairing entries.

See [artifact identities](08-artifact-identity.md) and the
[immutable store/runtime protocol](09-artifact-store.md) for internal contracts.
