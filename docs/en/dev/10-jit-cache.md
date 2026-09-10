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

`decode` has complete tensor annotations and scalar defaults. Alternatively,
pass sample tensors and scalars using the ordinary `compile()` argument rules.
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

The initial adapter supports Linux ELF GCC toolchains, the CANN BiSheng layout,
standalone ELF PTOAS, the packaged CPython PTOAS launcher grammar, and standard
pip/uv console scripts for PTOAS wheels with the NumPy dependency. Wheel
inventories include the selected virtualenv/interpreter, all installed package
resources, startup inputs, and native dependencies; import redirects are rejected. It hashes
installation content, compiler subprograms/resources, implicit include roots,
link inputs, Python/native runtime files and resolved ELF dependencies. Unknown
launchers, online-built PTOAS extensions, unsupported compiler layouts, sanitizer
builds and implicit dependency overrides such as `CPATH` or `LD_PRELOAD` bypass
persistence. The latest bypass reason is available in
`cache_stats().last_bypass_reason`, even when INFO logging is disabled. Every
bypass is also logged by `pypto.jit._persistent` at INFO.

Implicit linker scripts support absolute `INPUT`/`GROUP` dependencies, nested
`AS_NEEDED`, and `OUTPUT_FORMAT`/`OUTPUT_ARCH` declarations. Dependencies are
followed recursively with the selected sysroot. Relative inputs, `-l` names,
`SEARCH_DIR`, `INCLUDE`, and unknown syntax bypass persistence until the complete
linker search context can be modeled; ordinary private compilation still works.

Successful installation identities are memoized within each process by tool
selection and resolved component inventory. A working-directory change reruns
discovery because relative search roots can select different tools; components
with unchanged inventories reuse their existing content digests. Installed
files must remain immutable for the process lifetime; restart after replacing
them. Additional application sources are refreshed each request. Paths currently
participate in identity, so moving an installation may cause a miss.

Cold inventory reads are deliberately conservative and can be expensive. One
local CANN/PTOAS installation took approximately 12 seconds for its first content
inventory; that measurement is not a general performance claim. Each new
independent process currently pays this cost. A disk memo based only on path,
size, mtime and inode cannot prove unchanged contents and is not used. Statistics
include identity and validation time. Deployment without a verifiable local
toolchain is a separate protocol and is not enabled by this API.

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
are finite JSON numbers or booleans. The usual scalar defaults are honored.

Serializable `run_config` fields are `platform`, `strategy`, `memory_planner`,
`distributed_config`, `dump_passes`, `dump_ptoas_passes`, `save_kernels`, and
`save_kernels_dir`; enums use their Python member names. Distributed settings
accept `device_ids`, `num_sub_workers`, `runtime`, and `aicpu_thread_num`.
Warmup reports shared versus private preparation and exits nonzero on failure.
`stat` reads JSON manifests and declared payload sizes without importing cached
code, acquiring write locks or repairing entries.

See [artifact identities](08-artifact-identity.md) and the
[immutable store/runtime protocol](09-artifact-store.md) for internal contracts.
