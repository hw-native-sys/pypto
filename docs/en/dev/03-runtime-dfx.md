# Runtime DFX (Design For X) Flags

PyPTO exposes Simpler's four runtime diagnostic sub-features as independent
toggles on [`RunConfig`](../../../python/pypto/runtime/runner.py). Each
toggle maps 1:1 to a field on Simpler's `CallConfig` and to the matching
flag in `runtime/conftest.py`, so the two surfaces stay aligned.

## Flag matrix

| `RunConfig` field | pytest flag | `CallConfig` member | Artefact under `dfx_outputs/` | Post-run converter |
| ----------------- | ----------- | ------------------- | ----------------------------- | ------------------ |
| `enable_l2_swimlane: bool` | `--enable-l2-swimlane` | `enable_l2_swimlane` | `l2_perf_records.json` | `swimlane_converter` → `merged_swimlane_*.json` |
| `enable_dump_tensor: bool` | `--dump-tensor` | `enable_dump_tensor` | `tensor_dump/{tensor_dump.json,bin}` | `dump_viewer` (manual) |
| `enable_pmu: int` | `--enable-pmu [N]` (bare = `2`) | `enable_pmu` (`0` off, `>0` event type) | `pmu.csv` | — |
| `enable_dep_gen: bool` | `--enable-dep-gen` | `enable_dep_gen` | `deps.json` | `deps_to_graph` → `deps_graph.html` |

The four flags are **fully independent** and may be combined in any
subset. Enabling *any* of them auto-forces `RunConfig.save_kernels=True`
so the `<work_dir>/dfx_outputs/` directory survives the run.

## Output contract

The runtime writes every artefact under a single directory passed via
`CallConfig.output_prefix`. PyPTO sets that prefix to
`<work_dir>/dfx_outputs/` and the constituent subpaths are fixed per the
table above. Simpler's `CallConfig::validate()` rejects the call if any
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
        enable_l2_swimlane=True,     # produces l2_perf_records.json
        enable_dep_gen=True,         # produces deps.json + deps_graph.html
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
  tensor-dump,pmu-profiling,dep_gen}.md`.
- Compile-time profiling (orthogonal, single PyPTO process):
  [01-compile-profiling.md](01-compile-profiling.md).
