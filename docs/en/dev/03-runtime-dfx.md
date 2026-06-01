# Runtime DFX (Design For X) Flags

PyPTO exposes Simpler's five runtime diagnostic sub-features as independent
toggles on [`RunConfig`](../../../python/pypto/runtime/runner.py). Each
toggle maps 1:1 to a field on Simpler's `CallConfig` and to the matching
pytest flag in `tests/st/conftest.py`, so the two surfaces stay aligned.

## Flag matrix

| `RunConfig` field | pytest flag | `CallConfig` member | Artefact under `dfx_outputs/` | Post-run converter |
| ----------------- | ----------- | ------------------- | ----------------------------- | ------------------ |
| `enable_l2_swimlane: bool` | `--enable-l2-swimlane` | `enable_l2_swimlane` | `l2_swimlane_records.json` | `swimlane_converter` → `merged_swimlane_*.json` |
| `enable_dump_tensor: bool` | `--dump-tensor` | `enable_dump_tensor` | `tensor_dump/{tensor_dump.json,bin}` | `dump_viewer` (manual) |
| `enable_pmu: int` | `--enable-pmu [N]` (bare = `2`) | `enable_pmu` (`0` off, `>0` event type) | `pmu.csv` | — |
| `enable_dep_gen: bool` | `--enable-dep-gen` | `enable_dep_gen` | `deps.json` | `deps_to_graph` (manual) |
| `enable_scope_stats: bool` | `--enable-scope-stats` | `enable_scope_stats` | `scope_stats/scope_stats.jsonl` | `scope_stats_plot` (manual) |

The five flags are **fully independent** and may be combined in any
subset. Enabling *any* of them auto-forces `RunConfig.save_kernels=True`
so the `<work_dir>/dfx_outputs/` directory survives the run.

## Output contract

The runtime writes every artefact under a single directory passed via
`CallConfig.output_prefix`. PyPTO sets that prefix to
`<work_dir>/dfx_outputs/` and the constituent subpaths are fixed per the
table above. Most artefacts are flat files directly under the prefix;
`scope_stats` is the exception — its collector writes a `scope_stats/`
subdir holding `scope_stats.jsonl`. Simpler's `CallConfig::validate()`
rejects the call if any
flag is enabled but `output_prefix` is empty; PyPTO mirrors that contract
on the Python side and raises `ValueError` from `execute_on_device`
*before* the C++ boundary so the failure traceback points at the
caller.

## Usage

### From Python (`RunConfig`)

```python
from pypto.runtime import run, RunConfig

run(
    MyProgram, a, b, c,
    config=RunConfig(
        platform="a2a3sim",
        enable_l2_swimlane=True,     # produces l2_swimlane_records.json
        enable_dep_gen=True,         # produces deps.json (render with deps_to_graph on demand)
        enable_pmu=4,                # PMU event = MEMORY
    ),
)
```

### From pytest

```bash
pytest tests/st/runtime/framework_and_models/test_perf_swimlane.py \
    --platform a2a3sim --enable-l2-swimlane

pytest tests/st/runtime/ \
    --platform a2a3sim --enable-l2-swimlane --enable-dep-gen
```

## Selective tensor dump

`enable_dump_tensor=True` writes every binding of every task to
`tensor_dump/`. On large workloads this can saturate the host-side dump
collector (~42 MB/s drain) and the AICPU will be killed by the STARS
op-execute timeout — large bindings such as a 1 GB KV-cache fill the
queue faster than it drains. Mark the *interesting* tensors to limit dump
to those tensors. Two surfaces, both backed by the runtime
`enable_dump_tensor_selective()` toggle + `Arg::dump(...)` API from
simpler#844. They mirror the two `deps=` surfaces exactly — a declarative
marker (`pl.dump_tag`, the dump analogue of auto-inferred deps) and an
explicit kwarg (`dumps=`, the dump analogue of `deps=`):

**Declarative (`pl.dump_tag(t)`)** — a statement that marks `t` so every
*subsequent* kernel dispatch consuming that exact value dumps it, whether the
dispatch lowers to a plain `ir.Call` (the typical `@pl.jit` / tensor-op path)
or an `ir.Submit`:

```python
@pl.function(type=pl.FunctionType.Orchestration)
def orch(self, q: pl.Tensor[...], k_cache: pl.Tensor[...], out: pl.Out[...]):
    pl.dump_tag(q)
    pl.dump_tag(out)
    out = self.qk_pv(q, k_cache, out)   # q and out dumped; k_cache filtered out
```

**Explicit kwarg (`dumps=[...]`)** — `pl.submit(...)` and `pl.at(...)` accept a
`dumps=[...]` kwarg (symmetric with `deps=[...]`) listing the tensors to dump
at that one task launch. Each entry must be a tensor argument of that submit /
a tensor captured by that scope:

```python
with pl.manual_scope():
    out, tid = pl.submit(self.qk_pv, q, k_cache, out, deps=[prev], dumps=[q, out])
    # codegen → params_t0.dump(ext_q, ext_out);
```

There is **no call-arg wrapper** — a plain `self.kernel(...)` call site offers
no `dumps=` surface; use `pl.dump_tag` to mark its inputs, or submit it with
`pl.submit(..., dumps=[...])`. Both surfaces feed the same `dump_vars` attr on
the consuming Call / `Submit`, tracked by **Var identity** — never by name. It
rides through SSA, inlining, and codegen the same way `manual_dep_edges` does,
so no fuzzy name matching and no false positives. When `enable_dump_tensor=False`
both surfaces are inert (the whole dump pipeline is dormant).

`pl.dump_tag` is also accepted inside an Inline helper
(`@pl.jit.inline` / `FunctionType.Inline`), and works for both kernel-call
styles:

- **Explicit `self.kernel(...)` dispatch** — the tag records `dump_vars`
  on the consuming Call; the `InlineFunctions` pass splices that call into the
  caller and substitutes the caller's arg for each inline parameter, so tags
  on inline parameters and inline body-local `pl.create_tensor(...)` results
  take effect at the inlined call sites.
- **`@pl.jit` / tensor-op style (`with pl.at(level=...)`, `c = a + 1.0`)** —
  here the kernel dispatch is *synthesised by the outline passes*, not written
  at parse time. The tag instead seeds the enclosing scope's `dump_vars` (which
  round-trips as `pl.at(..., dumps=[...])`); a tag applied at the inline
  call site rides the call's `dump_vars` and is transferred by
  `InlineFunctions` onto the scopes it splices in. The outliner then
  translates each captured scope dump Var into the synthesised dispatch's
  `dump_vars` by Var identity — the same scope-attr → Call-attr path
  `no_dep_args=` uses. A tag the scope never consumes as a kernel arg is
  silently dropped.

No tag migration is needed in either case; multi-level inlining is handled at
the pass's fixpoint.

### Limitations

| Marker location / target | Status |
| ------------------------ | ------ |
| `pl.dump_tag(t)` as a standalone statement in an Orchestration or Inline body | Supported (declarative marker; affects every subsequent consuming dispatch). |
| `dumps=[arg]` on `pl.submit(...)` | Supported — explicit submit-side surface (symmetric with `deps=`); each entry must be a positional arg of the submit. |
| `dumps=[t]` on `pl.at(...)` | Supported — explicit scope-side surface (symmetric with `deps=`); each entry must be a tensor captured by the scope body. |
| `dumps=` on a plain `self.kernel(...)` call | Not supported — raises `ParserTypeError`. A plain call is fire-and-forget; declare the target with `pl.dump_tag(t)` or submit it with `pl.submit(..., dumps=[...])`. |
| Tag consumed by an outline-synthesised dispatch (`@pl.jit` / `with pl.at(level=...)` / tensor-op style) | Supported — the tag rides a scope-level `dump_vars` carrier (`dumps=`) and the outliner maps it onto the synthesised dispatch arg. |
| `pl.dump_tag(t)` inside a `@pl.function(type=pl.FunctionType.InCore/AIC/AIV/Group)` body | Not supported — raises `ParserSyntaxError` at parse time. Dump filtering is applied by orchestration codegen at the kernel-call site; kernel-body functions have no corresponding call-site arg to attach the marker to. Place `pl.dump_tag` in the enclosing `Orchestration` (or `Inline`) function instead. |
| Synthetic outputs of `pl.submit(...)` (implicit `Out`) | Not supported — synth outputs have no call-site arg to wrap. |
| HOST-tier Python `SubWorker` tensors | Not supported — runtime exposes no equivalent `Arg::dump` hook. |
| Reassigning a tagged value (e.g. `q = self.foo(q)`) | The rebound result is a **new value**; a previous `pl.dump_tag(q)` does **not** carry over (tracked by Var identity, not name). Re-tag the rebound value if the kernel consumes it. |
| Tagging a value consumed only after a shape/dtype transform (`q2 = pl.reshape(q)`, `pl.cast`, an elementwise op, …) | The transform produces a **new Var**, so `pl.dump_tag(q)` does **not** cover `q2`. Same root cause as reassignment (Var identity, not name). Tag the value the kernel actually receives — e.g. `pl.dump_tag(q2)`. |
| Tagging a value read only through a dynamic, data-dependent offset (`q_flat[runtime_row : runtime_row + N, …]`) | Not supported — the indexed read lowers to a gather / dynamic-address load, not a static whole-tensor `Arg`. Orchestration codegen extracts no whole-Var from that arg slot (`AsVarLike` yields nothing to match by identity), so the tag never attaches. Stage the value through a buffer read with **static, compile-time-tiled** offsets and tag that buffer. |
| Tagging an orch-tier buffer filled by `y = pl.assemble(y, tile, offset)` | Not supported — an orch-level `pl.assemble` lowers to a pure name-alias (`emit_name_map_[lhs] = target`, `HandleTensorAssembleAssign`) and emits **no kernel dispatch**. The buffer never reaches a task as a whole-tensor `Arg`, so there is nothing for `Arg::dump` to mark (compounded by `assemble` rebinding the Var each iteration). Use a static in-place slice store `y[offset_slice] = tile` and tag `y`, or dump the producer kernels' output Args instead. |
| Tagging a tensor consumed only by orchestration-level scalar reads (`pl.read(block_table_flat, […])`) | Not supported — the tensor is read element-wise at orch/AICPU/HOST tier (e.g. to compute page offsets) and never enters a device kernel as a Tensor `Arg`. The MVP runtime selective-dump path covers per-task **device** Args only. Stage it into a tensor that a device kernel consumes as a whole Arg. |

## Rendering `deps.json` to HTML

`enable_dep_gen` only emits the raw `deps.json`; the HTML pan/zoom graph
is produced by a separate offline tool. The tool is **not** invoked
automatically — Graphviz layout on a multi-thousand-node graph can run
for many minutes and, when launched on the runner's hot path, has
caused outer schedulers (e.g. taskqueue daemons) to SIGKILL the entire
job tree. Render on demand instead:

```bash
# Default — Graphviz `dot` engine, hierarchical layout (<500 nodes).
python -m simpler_setup.tools.deps_to_graph <work_dir>/dfx_outputs/deps.json

# Large graphs — switch to the scalable force-directed engine.
python -m simpler_setup.tools.deps_to_graph <work_dir>/dfx_outputs/deps.json \
    --engine sfdp
```

The output is written next to the input as `deps_graph.html` (override
with `-o <path>`). Supported `--engine` values, mirroring Graphviz:
`dot | sfdp | fdp | neato | circo | twopi`. `dot` is the default and
gives the cleanest DAG-style layout up to ~500 nodes; for larger graphs
prefer `sfdp` (O(N log N) layout, scales to 10k+ nodes). The runner
prints this same hint at the end of every dep_gen-enabled run.

Requires Graphviz on `PATH` (`apt install graphviz` /
`brew install graphviz`). Open the resulting HTML in any browser —
drag to pan, wheel to zoom, `f` to fit, `r` to reset.

## Rendering `scope_stats.jsonl` to HTML

`enable_scope_stats` emits the raw `scope_stats/scope_stats.jsonl`
(line 1 is run metadata; each later line is one per-scope record). Turn
it into a single self-contained HTML report — one timeline per ring with
the heap / task_window / tensormap peaks — with the offline renderer:

```bash
python runtime/tools/scope_stats_plot.py \
    <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
```

The report is written next to the input as `scope_stats.html`. Like
`deps_to_graph`, it is **not** invoked automatically — the runner prints
this hint at the end of every scope-stats-enabled run.

## Implementation map

| Concern | File | Function / member |
| ------- | ---- | ----------------- |
| `RunConfig` field declarations | [runner.py](../../../python/pypto/runtime/runner.py) | `RunConfig` dataclass + `any_dfx_enabled()` |
| `CallConfig` plumbing | [device_runner.py](../../../python/pypto/runtime/device_runner.py) | `execute_on_device(..., enable_*, output_prefix)` |
| Pipeline bundle | [runner.py](../../../python/pypto/runtime/runner.py) | `_DfxOpts` dataclass + `_DfxOpts.from_run_config` |
| Per-flag post-run dispatch | [runner.py](../../../python/pypto/runtime/runner.py) | `_collect_dfx_artifacts` |
| pytest entry | [tests/st/conftest.py](../../../tests/st/conftest.py) | `pytest_addoption` |
| Harness pipeline ctx | [tests/st/harness/core/test_runner.py](../../../tests/st/harness/core/test_runner.py) | `start_pipeline(..., enable_*)` |

## Deprecated aliases

`RunConfig.runtime_profiling` and the pytest flag `--runtime-profiling`
were the original way to opt into L2 swimlane capture before the four
DFX features became independently controllable. They are kept as
aliases for `enable_l2_swimlane` / `--enable-l2-swimlane` so existing
scripts keep working; both paths emit a `DeprecationWarning` and will
be removed in a future release. Migrate to the new names.

## Replaying an existing build_output

To re-run a previously compiled `build_output/<jit_dir>/` after editing
one or more kernel cpp files — typically to verify a hand-tuned change
under PMU / swimlane / tensor-dump — use the debug-only
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

`recompile=True` (default) deletes cached `.so`/`.bin` artefacts so
hand-edited cpps are picked up. Pass `recompile=False` (or
`--no-recompile`) when no cpp changed and you want to skip the rebuild.
`--log-level` accepts the same values as `PYPTO_RUNTIME_LOG`
(`debug`, `v0..v9`, `info`, `warn`, `error`, `null`); add
`--log-sync-pypto` to also push the band to PyPTO's C++ logger.

Pass `validate=True` (or `--validate`) to compare each output tensor
against the reference produced by `golden.py::compute_golden` using the
`RTOL`/`ATOL` tolerances declared in `golden.py`. Raises
`AssertionError` on mismatch. Requires the directory to contain a
`golden.py` (the default for `ir.compile`-produced artefacts).

### Editing `.pto` instead of cpp

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

### Auto-emitted `debug/run.py`

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

## Related

- Simpler's runtime-side reference: `runtime/docs/dfx/{l2-swimlane,
  tensor-dump,pmu-profiling,dep_gen,scope-stats}.md`.
- Compile-time profiling (orthogonal, single PyPTO process):
  [01-compile-profiling.md](01-compile-profiling.md).
