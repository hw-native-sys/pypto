# Replaying an Existing `build_output`

Re-run, edit, and re-measure a previously compiled `build_output/<jit_dir>/`
without recompiling from the DSL. The diagnostic flags referenced throughout are
documented in [Runtime DFX Flags](03-runtime-dfx.md).

To re-run a previously compiled `build_output/<jit_dir>/` after editing
one or more kernel cpp files — typically to verify a hand-tuned change
under PMU / swimlane / args-dump — use the debug-only
[`pypto.runtime.debug.replay`](../../../python/pypto/runtime/debug/replay.py)
module. It reuses the same `execute_compiled` path as the normal
`pypto.runtime.run` flow, so DFX flags behave identically.

```python
from pypto.runtime.debug import replay
from pypto.runtime import RunConfig

replay(
    "build_output/_jit_xxx/",
    a, b, c,
    config=RunConfig(
        platform="a2a3sim",
        enable_pmu=2,
        enable_l2_swimlane=True,
    ),
)
```

CLI form (loads inputs from the directory's `golden.py`):

```bash
python -m pypto.runtime.debug.replay build_output/_jit_xxx/ \
    --pmu 2 --swimlane --log-level debug
```

`recompile=True` (default) force-invalidates cached `.so`/`.bin` artefacts so
hand-edited cpps are picked up. Pass `recompile=False` (or
`--no-recompile`) to disable only that forced invalidation. Runtime / PTO-ISA
compatibility checks still run and may invalidate and rebuild cached artefacts.
Reuse also requires resolvable runtime and PTO-ISA identities. Runtime source
checkouts must be clean; an installed runtime may instead use its embedded
build commit. PTO-ISA currently must be a clean Git checkout. If either
identity cannot be established, PyPTO fails closed and rebuilds instead of
trusting the existing binaries.
`--log-level` accepts the same values as `PYPTO_RUNTIME_LOG`
(`debug`, `info`, `timing`, `warn`, `error`, `null`); add
`--log-sync-pypto` to also push the band to PyPTO's C++ logger.

Pass `validate=True` (or `--validate`) to compare each output tensor
against the reference produced by `golden.py::compute_golden` using the
`RTOL`/`ATOL` tolerances declared in `golden.py`. Raises
`AssertionError` on mismatch. Requires the directory to contain a
`golden.py` (the default for `ir.compile`-produced artefacts).

## Editing `.pto` instead of cpp

`replay` (and the auto-emitted `debug/run.py`) checks `ptoas/*.pto`
mtimes before invalidating cpp binaries: any `.pto` newer than its
sibling `ptoas/<unit>.cpp` triggers a fresh `ptoas` run, and the new
preprocessed body is spliced between the `// --- ptoas-generated code
---` and `// --- Kernel entry point ---` sentinels in every matching
`kernels/<core>/<func>.cpp`. The cpp → `.so` rebuild then runs as
normal.

| You edited | What runs |
| ---------- | --------- |
| only `kernels/<core>/<func>.cpp` | `cpp → .so` (existing behaviour) |
| only `ptoas/<unit>.pto` | `pto → cpp → .so` (new — splice + recompile) |
| both | `pto` wins for the body region; your wrapper / header edits in the cpp are preserved |

Requires the `ptoas` binary on `PTOAS_ROOT` or `PATH`; silently no-ops
otherwise. Disable with `--no-rebuild-from-pto` or
`PYPTO_REBUILD_FROM_PTO=0`. Editing a `.pto` that changes the kernel
function signature is **out of scope** — the saved wrapper boilerplate
will not match, and a fresh `ir.compile()` is required.

## Auto-emitted `debug/run.py`

`ir.compile()` writes a self-contained re-runner at
`<output_dir>/debug/run.py` so the user only ever needs to remember one
command:

```bash
python build_output/<jit_dir>/debug/run.py
```

The script wraps the `replay` flow above:

- When a sibling `golden.py` is present, inputs come from
  `golden.generate_inputs()` and the run is validated against
  `compute_golden`.
- Otherwise (JIT path), inputs are materialised from the shape / dtype
  metadata embedded in the script. Edit them freely to experiment. The
  script also exposes a `_user_compare(<param_names>)` hook that runs
  after `replay` returns — write your own `assert torch.allclose(...)`
  there to validate kernel output against a hand-rolled reference.
- The same `.pto` rebuild flow described above applies: edit a `.pto`
  under `ptoas/`, rerun the script, and the splice happens
  transparently. Pass `--no-rebuild-from-pto` to skip.

Emission is **best-effort** — programs without a clean orchestration
entry skip the file silently and the rest of compilation succeeds.

Disable globally by setting `PYPTO_EMIT_DEBUG_RUNNER=0` (also accepts
`false` / `no`, case-insensitive). Useful for large test suites or
benchmark pipelines that compile many programs and don't need the
runner. When disabled, the underlying `pypto.runtime.debug.replay`
module / CLI is still usable directly against the output directory.

## Benchmarking a replayed single-chip build

`execute_compiled(work_dir, ...)` is directory-driven, but `benchmark()` needs a
live `CompiledProgram` for the orchestration param metadata — derived from the IR
`Program`, which a directory replay does not have. So `ir.compile()` also writes a
`compiled_meta.json` sidecar (param metadata + platform + backend) next to
`kernel_config.py`, and `CompiledProgram.from_dir()` rebuilds a fully callable
program from it — **no pypto recompile, no pass re-run**:

**`from_dir()` reloads metadata only — it does not rebuild sources.** Unlike
`replay`, it runs neither the `.pto` → cpp splice nor the binary-cache
invalidation, so an edit left to it alone can be silently ignored: a `.pto` edit
never reaches the cpp, and an edited cpp can still be served from a cached
`.o` / `.so`. Do both explicitly, exactly as `replay` does internally:

```python
from pypto.ir import CompiledProgram
from pypto.runtime import benchmark
from pypto.runtime.debug import invalidate_binary_cache, rebuild_kernel_cpp_from_pto

work_dir = "build_output/<jit_dir>/"
rebuild_kernel_cpp_from_pto(work_dir)  # only if you edited ptoas/*.pto
invalidate_binary_cache(work_dir)      # drop cached .o/.so so the edit is compiled

compiled = CompiledProgram.from_dir(work_dir, platform="a2a3")
compiled(a, b, c)                                    # correctness re-check
stats = benchmark(compiled, [a, b, c], rounds=100)   # and timing
```

`platform` / `backend_type` default to the values recorded at compile time and
can be overridden to replay elsewhere (e.g. `a2a3sim` → `a2a3`). Runtime artifacts
are rederived from `kernel_config.py`, and the reload rewrites neither the sidecar
nor a hand-edited `debug/run.py`. `program` is `None` on the result (the IR is not
persisted); `validate_ir()` still works from `passes_dump/`. A multi-orch parent
has no sidecar of its own — each `next_levels/<name>/` sub-build carries one, so
reload the sub-build you want. Distributed builds use
`DistributedCompiledProgram.from_dir` (below).

Compiling into a **reused `output_dir`** always leaves the sidecar describing
the program just compiled: it is rewritten atomically, and removed outright
whenever the new program has no signature to record for that directory (a
multi-orch parent, an unextractable orchestration, or a sub-build the IR carries
no matching function for). `ir.compile()` does not otherwise clear `output_dir`,
so this is what keeps `from_dir()` from handing out a stale parameter ABI.

Which of the two layouts a build *is* — one top-level program, or one sub-build
per orchestration — is decided by the codegen that just ran
(`pto_backend.multi_chip_orch_names`), never by scanning the directory. A
`next_levels/` left by an earlier multi-orch compile therefore does not make the
next single-orch build into the same directory look multi-orch: the new
top-level artifacts stay reachable through `compiled(...)` and `from_dir()`.
The leftover sub-builds are untouched by a single-chip codegen, so each keeps
its own artifacts *and* its own sidecar — stale as a pair, never mismatched, and
`CompiledProgram.from_dir(next_levels/<name>)` still replays that older build.

**Build-kind markers do not survive a compile of the other kind.** A directory
is read as L2 or L3 from a small set of files — top-level `kernel_config.py` and
`compiled_meta.json` for single-chip, `orchestration/host_orch.py` and
`distributed_meta.json` for distributed — and `replay()` picks the L3 path
exactly when there is no top-level `kernel_config.py`. A leftover from the other
kind therefore would not just age out, it would re-point the whole directory at
the older build (an L2 `kernel_config.py` makes a fresh L3 build replay as L2;
an L3 sidecar keeps `DistributedCompiledProgram.from_dir` loadable on top of L2
artifacts). Every fresh compile drops the markers its own kind does not write,
so a reused directory either resolves to the build that just ran or fails
loudly. Artifact trees are never deleted — only these markers.

## Replaying an L3 / distributed build

Distributed (L3) programs — a `@pl.jit.host` orchestrator compiled to a
`DistributedCompiledProgram` — support the same edit-`.pto`-and-rerun loop,
but their build directory has a different shape: there is **no top-level
`kernel_config.py`** (per-rank configs live under `next_levels/{rank}/`), the
host driver is `orchestration/host_orch.py`, and `ir.compile()` writes a
`distributed_meta.json` sidecar:

```text
build_output/<jit_dir>/
  distributed_meta.json          # param metadata + platform + DistributedConfig
  orchestration/host_orch.py     # L3 host driver
  next_levels/{rank}/            # one complete single-chip sub-build per rank
      kernels/{aic,aiv}/*.cpp
      ptoas/*.pto
      kernel_config.py
```

`replay` detects this layout automatically (no top-level `kernel_config.py`
but `orchestration/host_orch.py` present) and dispatches via simpler
`Worker(level=3)` instead of `execute_compiled`. The same CLI / `debug/run.py`
flow works unchanged:

```bash
python -m pypto.runtime.debug.replay build_output/<jit_dir>/
# or
python build_output/<jit_dir>/debug/run.py
```

The `.pto` → cpp splice and `.so` invalidation recurse into every
`next_levels/{rank}/`, so editing `next_levels/rank0/ptoas/<unit>.pto` (or the
kernel cpp directly) is picked up exactly as in the single-chip case.

Reconstruction works as in the single-chip case above, from
`distributed_meta.json` alone. Two entry points expose it directly:

```python
from pypto.runtime import execute_distributed_compiled
# one-shot (distributed counterpart of execute_compiled):
execute_distributed_compiled("build_output/<jit_dir>/", [a, b, c])

# reusable object (override the persisted platform / devices if needed):
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig
prog = DistributedCompiledProgram.from_dir(
    "build_output/<jit_dir>/",
    platform="a2a3",
    distributed_config=DistributedConfig(device_ids=[0, 1]),
)
prog(a, b, c)
```

Here the persisted param metadata is the HOST orchestrator's (post-SSA names
matching `host_orch.py`), and chip callables are rebuilt by walking
`next_levels/`; `distributed_config` joins `platform` as an overridable default.

`distributed_meta.json` carries the same contract as its single-chip
counterpart, through the same loader: it is written atomically, removed
whenever the program compiled into that directory has no signature to record,
and validated field by field on load — including the `distributed_config` block
— so a hand-edited or truncated sidecar fails with one `ValueError` naming the
file and the recompile that regenerates it.

L3 replay forwards runtime DFX fields from `RunConfig` through each chip
dispatch. Artifacts are written under
`dfx_outputs/rank{r}/d{k}/`; onboard swimlane uses the graph/timing two-pass
protocol described in [Runtime DFX Flags](03-runtime-dfx.md). The edit-and-rerun loop therefore supports both
correctness re-checks and L3 runtime diagnostics.
