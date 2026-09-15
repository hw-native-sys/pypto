# Replaying a Build

Re-running, hand-editing, and re-measuring an output directory — without going back to the
DSL.

## Concept

A `build_output/<jit_dir>/` is self-contained: the generated `.pto`, the kernel cpps, the
runtime config, and the parameter metadata are all there. Replay is the loop that closes
over it — edit a kernel by hand, rerun, measure — and it is the fastest way to answer
"would this change help" without a recompile.

The one thing that makes it work is **cache invalidation**. Without it, a hand-edited cpp is
silently served from a stale `.so` and you measure the old kernel while believing you
changed it.

## Quickstart

Every compile writes a re-runner, so there is one command to remember:

```bash
python build_output/<jit_dir>/debug/run.py
```

With a sibling `golden.py`, it loads inputs via `generate_inputs()` and validates against
`compute_golden`. On the JIT path there is no `golden.py`, so inputs are materialised from
the shape / dtype metadata embedded in the script — edit them freely.

## Mechanics

### The edit-and-rerun loop

| You edited | What runs |
| ---------- | --------- |
| `kernels/<core>/<func>.cpp` | `cpp → .so` |
| `ptoas/<unit>.pto` | `pto → cpp → .so` — ptoas reruns and the new body is spliced in |
| Both | `.pto` wins for the body region; your wrapper / header edits in the cpp are preserved |

The discriminator is mtime, evaluated per unit: a `.pto` newer than its sibling
`ptoas/<unit>.cpp` triggers the rerun. The splice replaces everything between the
`// --- ptoas-generated code ---` and `// --- Kernel entry point ---` sentinels.

Requires the `ptoas` binary on `PTOAS_ROOT` or `PATH`; silently no-ops otherwise. Disable
with `--no-rebuild-from-pto` or `PYPTO_REBUILD_FROM_PTO=0`.

> **Changing a kernel's signature in a `.pto` is out of scope.** The saved wrapper
> boilerplate will not match it; recompile from the DSL.

### The replay module

`debug/run.py` wraps it; the module is usable directly when you want arguments of your own:

```python
from pypto.runtime.debug import replay
from pypto.runtime import RunConfig

replay(
    "build_output/_jit_xxx/",
    a, b, c,
    config=RunConfig(platform="a2a3", enable_pmu=2, enable_chip_swimlane=4),
)
```

```bash
python -m pypto.runtime.debug.replay build_output/_jit_xxx/ \
    --pmu 2 --swimlane --dep-gen --scope-stats --log-level debug
```

**Every flag from [Runtime DFX](04-runtime-dfx.md) applies unchanged here** — replay is
the intended way to attach DFX to a build you have been hand-tuning.

| Option | Effect |
| ------ | ------ |
| `recompile=True` (default) / `--no-recompile` | Force-invalidate cached `.so` / `.bin` so hand edits are picked up |
| `validate=True` / `--validate` | Compare each output against `golden.py::compute_golden` at its declared `RTOL` / `ATOL`; raises `AssertionError` |
| `--log-level` | Same values as `PYPTO_RUNTIME_LOG` (`debug`, `info`, `timing`, `warn`, `error`, `null`) |
| `--log-sync-pypto` | Also push the level to PyPTO's C++ logger |

`--no-recompile` disables only the *forced* invalidation. Runtime and PTO-ISA compatibility
checks still run, and PyPTO **fails closed**: if either identity cannot be established, it
rebuilds rather than trusting the existing binaries.

### Measuring a replayed build

`benchmark()` needs a live `CompiledProgram` for the parameter metadata, which a directory
does not carry — so `ir.compile()` also writes a `compiled_meta.json` sidecar, and
`from_dir()` rebuilds a callable program from it with no pass re-run.

**`from_dir()` reloads metadata only — it does not rebuild sources.** Unlike `replay`, it
runs neither the `.pto` splice nor the cache invalidation, so an edit left to it alone is
silently ignored. Do both explicitly:

```python
from pypto.ir import CompiledProgram
from pypto.runtime import benchmark
from pypto.runtime.debug import invalidate_binary_cache, rebuild_kernel_cpp_from_pto

work_dir = "build_output/<jit_dir>/"
rebuild_kernel_cpp_from_pto(work_dir)   # only if you edited ptoas/*.pto
invalidate_binary_cache(work_dir)       # drop cached .o/.so so the edit is compiled

compiled = CompiledProgram.from_dir(work_dir, platform="a2a3")
compiled(a, b, c)                                     # correctness re-check
stats = benchmark(compiled, [a, b, c], rounds=100)    # and timing
```

`platform` / `backend_type` default to what was recorded at compile time and can be
overridden to replay elsewhere (`a2a3sim` → `a2a3`). `program` is `None` on the result — the
IR is not persisted — but `validate_ir()` still works, from `passes_dump/`.

### Distributed builds

An L3 build has no top-level `kernel_config.py` (per-rank configs live under
`next_levels/{rank}/`) and is driven by `orchestration/host_orch.py`. `replay` detects that
layout automatically and dispatches through a rebuilt `DistributedCompiledProgram`; the
`.pto` splice and cache invalidation recurse into every rank. The same two commands work
unchanged, and DFX artifacts land under `dfx_outputs/rank{r}/d{k}/`.

```python
from pypto.ir import DistributedCompiledProgram, DistributedConfig

DistributedCompiledProgram.from_dir("build_output/<jit_dir>/")(a, b, c)
```

### Running an artifact as its own process

`pypto.runtime.execute_artifact` runs a compiled directory in a subprocess and prints a
machine-readable result marker:

The marker's shape depends on which mode it ran in. A single directory reports only the
device, because the caller already knows which one it asked for:

```text
PYPTO_EXEC_RESULT=PASS device=<N>    # ran, and validated
PYPTO_EXEC_RESULT=FAIL               # device or validation failure
PYPTO_EXEC_RESULT=INFRA              # reconstruction / setup failure
```

A batch manifest runs several directories, so every line names the one it reports on:

```text
PYPTO_EXEC_RESULT=PASS  work_dir=<wd> device=<N>
PYPTO_EXEC_RESULT=FAIL  work_dir=<wd>
PYPTO_EXEC_RESULT=INFRA work_dir=<wd>
```

The third value is the point in both modes: it separates "the kernel is wrong" from "the
environment is broken", so a CI run does not record a device shortage as a regression.

## Edge Cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| **Edited the cpp, nothing changed** | Stale `.so` served from cache | Use `replay` (invalidates by default), not `from_dir()` alone |
| **Edited the `.pto`, nothing changed** | `ptoas` not on `PTOAS_ROOT` / `PATH`, so the splice no-opped | Point `PTOAS_ROOT` at a working ptoas |
| **No `debug/run.py`** | Emission is best-effort and skips programs without a clean orchestration entry; or `PYPTO_EMIT_DEBUG_RUNNER=0` | Use `python -m pypto.runtime.debug.replay <dir>` directly |
| **`ValueError` naming `compiled_meta.json`** | Sidecar hand-edited or truncated | Recompile; the message names the file and the recompile that regenerates it |
| **Replays as L2 but the build was L3** | A leftover top-level `kernel_config.py` from an earlier compile into the same directory | Compile into a fresh directory; every compile drops the markers its own kind does not write |
| **`validate=True` raises `FileNotFoundError`** | No `golden.py` (the JIT path writes none) | Use the `_user_compare(...)` hook in `debug/run.py` instead |

## See Also

- [Runtime DFX](04-runtime-dfx.md) — the flags to attach to a replay.
- [In-core trace](06-incore-trace.md) — profiling the kernel you just hand-edited.
- [Performance](../performance/index.md) — what to change, and how to confirm it worked.
- [Replaying an Existing `build_output`](../../dev/03-runtime-replay.md) — sidecar contracts, build-kind markers, and the reused-`output_dir` rules.
