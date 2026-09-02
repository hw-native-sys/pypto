# Runtime DFX (Design For X) Flags

PyPTO exposes Simpler's five runtime diagnostic sub-features as independent
toggles on [`RunConfig`](../../../python/pypto/runtime/runner.py). Each
toggle maps to a field on Simpler's `CallConfig` and to the matching pytest
flag in `tests/st/conftest.py`. Field names match Simpler's; the former
`enable_l2_swimlane` / `--enable-l2-swimlane` spellings still work and are
covered under [Deprecated aliases](#deprecated-aliases).

## Flag matrix

| `RunConfig` field | pytest flag | `CallConfig` member | Artefact under `dfx_outputs/` | Post-run converter |
| ----------------- | ----------- | ------------------- | ----------------------------- | ------------------ |
| `enable_chip_swimlane: int` | `--enable-chip-swimlane` (= `4`) / `--chip-swimlane-level N` | `enable_chip_swimlane` (`0` off .. `4` full) | `chip_swimlane_records.json` | `swimlane_converter` → `merged_swimlane_*.json` |
| `enable_dump_args: int` | `--dump-args [LEVEL]` (bare = `1`) | `enable_dump_args` (`0` off, `1` partial, `2` full) | `args_dump/{args_dump.json,bin}` | `dump_viewer` (manual) |
| `enable_pmu: int` | `--enable-pmu [N]` (bare = `2`) | `enable_pmu` (`0` off, `>0` event type) | `pmu.csv` | — |
| `enable_dep_gen: bool` | `--enable-dep-gen` | `enable_dep_gen` | `deps.json` | `deps_viewer` (manual) |
| `enable_scope_stats: bool` | `--enable-scope-stats` | `enable_scope_stats` | `scope_stats/scope_stats.jsonl` | `scope_stats_plot` (manual) |

The five flags are **fully independent** and may be combined in any
subset. Enabling *any* of them auto-forces `RunConfig.save_kernels=True`
so the `<work_dir>/dfx_outputs/` directory survives the run.

### Swimlane collection levels

`enable_chip_swimlane` is a **level**, not a toggle. Each level is a real guard
in the runtime collectors, so a lower level never stamps the data a higher one
does and no post-processing recovers it:

| Level | Adds | Unlocks |
| ----- | ---- | ------- |
| `0` / `False` | — | collection off |
| `1` | AICore per-task start / end + task record buffer | per-task lanes |
| `2` | + AICPU-stamped dispatch / finish | the `[dispatch, start]` pickup gap |
| `3` | + scheduler main-loop phase records | `simpler_setup.tools.sched_overhead_analysis`, the Toolkit plugin's Scheduler View |
| `4` / `True` | + orchestrator phase records | the Toolkit plugin's AICPU Orchestrator view |

`True` requests level `4` — the same thing the bare `--enable-chip-swimlane`
flag requests, in PyPTO and in the runtime harness alike. An out-of-range level
raises `ValueError` from `RunConfig`.

On the pytest surface the bare flag and the level are **two** options
(`--enable-chip-swimlane` and `--chip-swimlane-level N`) rather than one
optional-valued option. An optional-valued flag swallows the following token, so
`pytest --enable-chip-swimlane tests/st/runtime/` would fail with
`invalid int value: 'tests/st/runtime/'`. Splitting them keeps the bare flag
order-independent.

## Output contract

The runtime writes every artefact under a single directory passed via
`CallConfig.output_prefix`. PyPTO sets that prefix to
`<work_dir>/dfx_outputs/` and the constituent subpaths are fixed per the
table above. Most artefacts are flat files directly under the prefix;
`scope_stats` is the exception — its collector writes a `scope_stats/`
subdir holding `scope_stats.jsonl`. Simpler's `CallConfig::validate()`
rejects the call if any
flag is enabled but `output_prefix` is empty; PyPTO mirrors that contract
on the Python side and raises `ValueError` from `_execute_on_device`
*before* the C++ boundary so the failure traceback points at the
caller.

### L3 (distributed): one subdirectory per dispatch

A distributed run dispatches to several chips, and one chip may receive
several dispatches in a single host orchestration — every dispatch would
otherwise rewrite the same fixed-name files. So on the L3 path PyPTO
namespaces the prefix per dispatch:

```text
<work_dir>/dfx_outputs/
├── rank0/d0/          # rank 0, its 0th dispatch
│   └── dispatch_program.json   # which next_levels/<program> ran here
├── rank0/d1/          # rank 0, its 1st dispatch
└── rank1/d0/
```

`d{k}` counts that card's dispatches within the run, restarting at `d0`
each run. Every dispatch is filed under the chip that ran it: a `device=`
pinned dispatch under its own rank, and a comm-less one (no `device=`)
under the chip it was placed on — those are handed out round-robin over
the program's chips in submit order. Each leaf holds the flat artefacts
from the table above, so the L2 contract applies unchanged within one
dispatch directory.

The path records *where* a dispatch ran, not *what* it ran, so
`_submit_chip` also drops a `dispatch_program.json` naming the
`next_levels/<program>` behind it. Kernel names must be resolved through
it: `func_id` is a per-L2-program namespace — every program numbers its
kernels from 0 — so a name map merged across programs relabels one
program's tasks with another's names, silently and plausibly. A dispatch
whose program cannot be resolved is converted with anonymous labels
instead of guessed ones.

## Swimlane runs the workload twice (onboard)

The swimlane converter joins per-task timing against a task graph that **only
`deps.json` carries** — the device hot path no longer records per-task fanout,
so without a dep_gen capture the lanes degrade to anonymous `task(rXtY)` with no
dependency arrows. But dep_gen collection has high overhead that perturbs the
very timing the swimlane measures. The two captures therefore come from separate
runs (Simpler's documented "capture the graph once, time many times" workflow).

For **onboard L2**, enabling `enable_chip_swimlane` runs the kernel twice,
transparently:

1. **Graph pass** — dep_gen only, producing `deps.json`. Runs in a **separate
   subprocess** (`python -m pypto.runtime._dep_gen_capture`). This is required,
   not just tidy: the runtime's per-run finalize does not reliably reclaim the
   SVM host-register mappings the DFX collectors allocate, so a second DFX run
   in the *same* process hits the registration cap (`halHostRegister` rc 8). A
   child process fully reclaims that state on exit. The capture is best-effort —
   if the subprocess fails, a warning is logged and the timing pass still runs
   (lanes degrade to anonymous `task(rXtY)`).
2. **Timing pass** — swimlane (plus any other timing-sensitive DFX such as PMU /
   args-dump / scope-stats), dep_gen forced off, producing the clean
   `chip_swimlane_records.json` whose timing is reported. Runs in-process.

Both passes write into the same `dfx_outputs/`, so `swimlane_converter`
auto-joins the sibling `deps.json` with the records. Adding `--enable-dep-gen`
explicitly changes nothing about the passes (the graph pass already produced
`deps.json`); it only makes the run additionally print the `deps_viewer` render
hint. Simulator platforms (`*sim`) stay single-pass — swimlane conversion is
skipped there regardless.

Distributed L3 uses the same graph/timing split without the L2 capture
subprocess. The one-shot path creates a fresh Worker lifecycle for each pass.
A prepared `DistributedWorker` keeps its resident handles and forked hierarchy,
but enters two separate `Worker.run()` fences: dep-gen-only first, then
swimlane with dep_gen forced off. Per-card dispatch counters restart for both
passes, so graph and timing artefacts join in the same `rank{r}/d{k}` directory.
Both passes execute the program; mutable arguments are not restored between
them, matching the existing one-shot L3 replay semantics.

The L2 subprocess rebuilds the orchestration arguments two ways: from `golden.py`
when driven by the pytest harness (deterministic inputs → faithful graph), or
from a recorded spec when driven by the compiled-program API
(`compiled(...)`). The task graph can be routed by tensor *values*, not just
scalars (e.g. paged-attention `block_tables` / `seq_lens`), so the spec preserves
real data wherever it can cross the process boundary: host `torch.Tensor`s are
saved and reloaded verbatim, scalars are preserved exactly, and only
device-resident `DeviceTensor`s — unreachable from a fresh child — fall back to
zero-filled tensors of the recorded shape. The capture is therefore exact unless
a *device-resident* tensor routes the graph, in which case it is approximate.

## Usage

### From Python (`RunConfig`)

```python
from pypto import ir
from pypto.runtime import RunConfig

config = RunConfig(
    platform="a2a3sim",
    enable_chip_swimlane=4,      # full swimlane -> chip_swimlane_records.json
                                 # (True is the same level 4; use 1-3 for less)
    enable_dep_gen=True,         # produces deps.json (render with deps_viewer on demand)
    enable_pmu=4,                # PMU event = MEMORY
)
compiled = ir.compile(MyProgram, **config.compile_kwargs())
compiled(a, b, c, config=config)
```

### From pytest

```bash
# Bare flag = level 4 (full)
pytest tests/st/runtime/framework_and_models/test_perf_swimlane.py \
    --platform a2a3sim --enable-chip-swimlane

# AICore timing only — the cheapest capture
pytest tests/st/runtime/ \
    --platform a2a3sim --chip-swimlane-level 1 --enable-dep-gen
```

## Selective tensor dump

`enable_dump_args` is a **level** (`0`=off, `1`=partial, `2`=full;
`True`→`1`, `False`→`0`). Level `2` writes every binding of every task to
`args_dump/`. On large workloads that can saturate the host-side dump
collector (~42 MB/s drain) and the AICPU will be killed by the STARS
op-execute timeout — large bindings such as a 1 GB KV-cache fill the
queue faster than it drains. Run **partial** dump (level `1`) and mark the
*interesting* tensors to limit dump to those tensors. Two surfaces, both backed
by the runtime `Arg::dump(...)` API (simpler#844). Selective-vs-full is latched
host-side from the dump level, so no orch-body toggle is emitted (simpler#953).
They mirror the two `deps=` surfaces exactly — a declarative
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
rides through SSA, inlining, and codegen the same way `Submit::deps_` does,
so no fuzzy name matching and no false positives. The marks only take effect
under partial dump (`enable_dump_args == 1`); they are inert when dump is off
(`0`) and irrelevant under full dump (`2`), which captures every binding.

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
# Text summary (default) — grep-friendly, no Graphviz required.
python -m simpler_setup.tools.deps_viewer <work_dir>/dfx_outputs/deps.json

# HTML graph — Graphviz `dot` engine, hierarchical layout (<500 nodes).
python -m simpler_setup.tools.deps_viewer <work_dir>/dfx_outputs/deps.json \
    --format html

# Large graphs — switch to the scalable force-directed engine.
python -m simpler_setup.tools.deps_viewer <work_dir>/dfx_outputs/deps.json \
    --format html --engine sfdp
```

The output is written next to the input as `deps_viewer.txt` (text, the
default) or `deps_viewer.html` (`--format html`), override with
`-o <path>`. `--engine` applies to HTML only; supported values mirror
Graphviz: `dot | sfdp | fdp | neato | circo | twopi`. `dot` is the
default and gives the cleanest DAG-style layout up to ~500 nodes; for
larger graphs prefer `sfdp` (O(N log N) layout, scales to 10k+ nodes).
The runner prints this same hint at the end of every dep_gen-enabled run.

Requires Graphviz on `PATH` (`apt install graphviz` /
`brew install graphviz`). Open the resulting HTML in any browser —
drag to pan, wheel to zoom, `f` to fit, `r` to reset.

### Human-readable kernel names (`name_map_*.json`)

By default the swimlane / dependency-graph tools label tasks by numeric
id (`task(rXtY)` / `func_<id>(...)`). To recover real kernel names
(`matmul(rXtY)`), a name map must sit next to the records. Simpler's own
SceneTest harness writes this file; pypto does not use SceneTest, so when
`enable_chip_swimlane` or `enable_dep_gen` is set the runner synthesises
`<work_dir>/dfx_outputs/name_map_<case>.json` from the `func_id` / `name`
fields already in `kernel_config.py`. It is consumed automatically:
`swimlane_converter` is invoked with `--func-names <name_map>`, and
`deps_viewer` auto-discovers the sibling `name_map_*.json`. No manual
step is required.

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
`deps_viewer`, it is **not** invoked automatically — the runner prints
this hint at the end of every scope-stats-enabled run.

## Implementation map

| Concern | File | Function / member |
| ------- | ---- | ----------------- |
| `RunConfig` field declarations | [runner.py](../../../python/pypto/runtime/runner.py) | `RunConfig` dataclass + `any_dfx_enabled()` |
| `CallConfig` plumbing | [device_runner.py](../../../python/pypto/runtime/device_runner.py) | `_execute_on_device(..., enable_*, output_prefix)` |
| Pipeline bundle | [runner.py](../../../python/pypto/runtime/runner.py) | `_DfxOpts` dataclass + `_DfxOpts.from_run_config` |
| Per-flag post-run dispatch | [runner.py](../../../python/pypto/runtime/runner.py) | `_collect_dfx_artifacts` |
| Kernel-name map synthesis | [runner.py](../../../python/pypto/runtime/runner.py) | `_write_name_map` |
| L3 per-dispatch program marker | [distributed_runner.py](../../../python/pypto/runtime/distributed_runner.py) | `_record_dispatch_program` / `_read_dispatch_program` |
| L3 per-dispatch swimlane conversion | [distributed_runner.py](../../../python/pypto/runtime/distributed_runner.py) | `_collect_l3_swimlane` / `_write_dispatch_name_map` |
| pytest entry | [tests/st/conftest.py](../../../tests/st/conftest.py) | `pytest_addoption` |
| Harness pipeline ctx | [tests/st/harness/core/test_runner.py](../../../tests/st/harness/core/test_runner.py) | `start_pipeline(..., enable_*)` |

## Deprecated aliases

`RunConfig.enable_l2_swimlane` and the pytest flag `--enable-l2-swimlane`
are the former spellings of `enable_chip_swimlane` /
`--enable-chip-swimlane`. Simpler's Worker/Chip/Core naming migration
renamed the L2 layer to "chip" (`L2Swimlane*` -> `ChipSwimlane*`,
`l2_swimlane_records.json` -> `chip_swimlane_records.json`), and PyPTO now
follows that contract.

Both old spellings keep working and emit a `DeprecationWarning`; they will
be removed in a future release. Values and semantics are unchanged, so
migration is a rename:

```python
RunConfig(enable_l2_swimlane=True)    # deprecated
RunConfig(enable_chip_swimlane=4)     # same capture
```

Details worth knowing:

- `enable_l2_swimlane` is **not** a dataclass field — it is a constructor
  keyword plus a property. That keeps `dataclasses.replace(cfg,
  enable_chip_swimlane=N)` unambiguous; an alias field would be re-supplied
  by `replace()` from the old instance and could silently override the value
  you just passed.
- Reading `cfg.enable_l2_swimlane` is silent (it returns the canonical
  level). Passing the old constructor keyword, or assigning to the
  attribute, warns.
- Passing both spellings at once raises `ValueError`.

## Replaying an existing build_output

Re-running, editing, and re-measuring an existing `build_output/<jit_dir>/`
(including `debug/run.py`, the `.pto` splice, `benchmark()` against a directory
replay, and L3 builds) has its own page:
[Replaying an Existing `build_output`](03-runtime-replay.md). Every DFX flag
documented above applies unchanged on that path.

## Related

- Simpler's runtime-side reference: `runtime/docs/dfx/{chip-swimlane-profiling,
  args-dump,pmu-profiling,dep-gen,scope-stats}.md`.
- Compile-time profiling (orthogonal, single PyPTO process):
  [01-compile-profiling.md](01-compile-profiling.md).
