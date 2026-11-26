# In-Core Simulator Profiling

Use the Ascend `msprof op simulator` workflow to inspect the instruction and
pipeline behavior of one generated PTOAS kernel on a single AI Core. It
produces cycle-level traces that can explain a low-utilization kernel after a
chip swimlane or PMU report has identified the task worth investigating.

This workflow is different from:

- `report/perf_hints.log`, which contains compile-time recommendations;
- the chip swimlane in [Performance Tuning](https://github.com/hw-native-sys/pypto-lib/blob/main/docs/debug-and-tune/performance-tuning.md), which
  shows the multi-kernel device schedule; and
- [CCE In-Core Profiling](https://github.com/hw-native-sys/pypto-lib/blob/main/docs/debug-and-tune/cce-incore-profiling.md),
  which instruments phases of a multi-core extern kernel on real inputs.

The repository currently exposes testcase generation and collection through
its agent profiling workflow rather than a stable public CLI. This page is the
canonical description of the inputs, evidence, caveats, and interpretation of
that workflow.

## Prerequisites

Start from a completed build under `build_output/<case>/`. The build must
contain generated PTOAS `.cpp` files and their sibling `.pto` files, either in
a top-level `ptoas/` directory or below `next_levels/**/ptoas/`. The `.pto`
tensor-view shapes and strides are used to size standalone GM buffers.

The host also needs:

- a CANN installation whose `bisheng` supports the Tile-Language option used
  by the generated kernels;
- `msprof op simulator` and its `msopprof` worker;
- a camodel library for the requested SoC; and
- `ptoas-bin`.

Choose the target explicitly:

| PyPTO target | AI Core architecture | Typical use |
| ------------ | -------------------- | ----------- |
| `a2a3` | `dav-c220` | Ascend A2/A3 |
| `a5` | `dav-c310` | Ascend A5 |

Do not silently provision a missing `msopprof` worker into a shared CANN
installation. That writes outside the repository and can mix toolchain
versions. Prefer installing the complete matching CANN package; any temporary
copy into an existing toolkit requires the toolkit owner's approval and must
include the worker's companion injection library.

## Select the scope before collecting

Use an existing build whenever possible. It preserves the exact generated
kernel already observed in the L2 or PMU evidence and avoids changing inputs
or compiler state.

First list the discovered function names without invoking the toolchain. Then
select one function unless the investigation genuinely needs every kernel.
Collecting all kernels is useful for an inventory, but it is slower and makes
it easier to overlook a failed or degenerate trace.

For each selected kernel, the workflow:

1. Reads the generated `.cpp` signature and sibling `.pto` tensor views.
2. Generates a standalone testcase with GM buffers, launch code, and input
   data.
3. Builds it for the chosen AI Core architecture.
4. Runs `msprof op simulator`.
5. Records collection status and trace artifacts without stopping unrelated
   kernels unless fail-fast was requested.

Mixed cube/vector kernels use a small dispatcher that launches the appropriate
AIC and AIV entry points. The standalone case is intentionally single-core; it
does not reproduce the full persistent or multi-core runtime schedule.

For a kernel with `pl.split_aiv`, the two vector cores may take different
branches. The normal runtime supplies a separate subblock ID to each AIV, but
the standalone launch passes the same host scalar arguments to both. When the
PTO IR identifies `__pypto_spmd_subblock_idx`, the profiling workflow rewrites
only the generated standalone kernel to read each core's physical subblock ID.
This is specific to the direct CCE simulator launch; it does not change the
source kernel or the normal runtime wrapper. The manifest records
`subblock_lane=get_subblockid()` when the rewrite was applied.

## Output layout

A collection is stored below the selected build directory:

```text
build_output/<case>/kernel_insight_all_funcs_<timestamp>/
├── manifest_export.csv
├── summary.txt
└── funcs/
    └── <kernel>/
        ├── collect/out/OPPROF_*/.../simulator/  # newer CANN
        └── export/.../simulator/                # older CANN export pass
```

Treat `manifest_export.csv` as authoritative. Each row records the status and
the actual `export_dir`, `trace_json`, and `visualize_data_bin` paths. Newer
CANN versions can emit final artifacts during collection, in which case the
recorded export directory is below `collect/out`; older versions use the
separate `export/` path. `summary.txt` is a convenient human-readable index.
Per-core directories contain instruction-execution and code-execution CSV
files in addition to raw traces.

Keep the complete collection under `build_output/`. It is generated evidence
and must not be committed.

## Clean and inspect a trace

The raw trace includes synchronization, cache-miss, and control-flow lanes.
PyPTO can convert it to a smaller, Perfetto-viewable pipe trace:

```bash
python -m pypto.tools.clean_sim_trace \
  <OPPROF_directory> \
  -o build_output/incore_<kernel>_<source>_<timestamp>
```

The cleaned directory contains:

```text
trace.clean.json
instr_metrics.json       # only when the source contains an API_INSTR block
raw_simulator/
```

Rename `trace.clean.json` to include the kernel name before comparing several
traces side by side. Record the source case, target, generated function, input
assumptions, and any patched scalar arguments in a nearby `summary.txt`.

Open the cleaned JSON in [Perfetto](https://ui.perfetto.dev/). When present,
use `instr_metrics.json` to summarize cycles by pipeline and to detect an
obviously empty workload before drawing conclusions. Its absence only means
the source trace had no `API_INSTR` block; consult the per-core CSV files and
the cleaned trace instead.

## Validate that the standalone workload is real

Generated inputs are synthetic. Integer tensors are commonly zero-filled and
dynamic scalar tail arguments may default to `1`. If a kernel derives a loop
bound, task count, valid length, work table, or tensor extent from those
values, the standalone case may execute only its scalar prologue and
synchronization path.

Treat any of these as a degenerate trace:

- total execution is implausibly short;
- `CUBE=0` for a kernel that should perform matmul;
- a mixed kernel has no vector work;
- a split-AIV kernel shows both vector cores taking the same branch when the
  source assigns different work to subblock IDs 0 and 1;
- the instruction CSV contains only scalar and synchronization operations; or
- the collection summary warns about data-dependent control.

Before trusting such a trace:

1. Identify every tensor and scalar that controls execution.
2. Derive hidden dynamic scalars from generated orchestration
   `add_scalar` calls or the kernel signature.
3. Choose representative values for the control tensors and scalars.
4. For a direct dynamic extent or stride, regenerate with
   `--dynamic-dim N`, where `N` is at least the largest scalar value.
   For a computed SSA extent or stride, use the full PTOAS generator.
5. Replace the control input binaries and wire scalar values without
   exceeding the generated bound.
6. Rebuild and recollect from the testcase's working directory.
7. Record the exact wired workload with the trace.

For split-AIV branches, inspect the instruction CSV or trace for each vector
core and confirm that the intended branch actually ran. Nonzero VECTOR cycles
alone do not establish this: both cores can spend time in the same wrong
branch. If the hardware subblock ID does not distinguish them in a particular
camodel, report that limitation and do not interpret the trace as covering the
missing branch.

Static `.pto` shapes size the full GM allocations. For a direct `%argN`
dynamic extent or stride, the bundled generator uses `--dynamic-dim`
(default `256`) as the allocation bound and emits a runtime guard. It rejects
computed SSA dimensions because it cannot bound them safely. Regenerate with
the intended bound instead of patching only the scalar or allocation; changing
only one can make the standalone case access beyond its generated buffer.

## Interpret pipeline evidence

Check the expected pipes before comparing totals:

- a cube kernel should show nonzero MTE1/MTE2, CUBE, and usually FIXPIPE work;
- a vector kernel should show MTE2, VECTOR, and MTE3 work;
- a mixed kernel should show both cube and vector activity; and
- long synchronization gaps require comparison with dependencies and the
  multi-core schedule, not only a single-core trace.

A simulator total is valid for the generated standalone case. It is not an
end-to-end latency prediction: host dispatch, other cores, cross-core
barriers, runtime dependencies, and production data can change the result.
Use repeated device measurements to confirm any optimization selected from a
simulator trace.

## Troubleshooting

| Symptom | Likely cause | Action |
| ------- | ------------ | ------ |
| `unknown type name '__biasbuf__'` or undeclared `aicore` | Selected CANN compiler is not Tile-Language capable | Use the CANN installation used for the device build |
| Header errors under an A5 include path while targeting A2/A3 | Architecture mismatch | Select the matching target and rebuild |
| `cannot find -lruntime_camodel` | Selected SoC has no camodel library | Choose an installed SoC variant that matches the device |
| Missing `msopprof` worker | Incomplete toolkit package | Install the matching operator-development tools; do not silently mutate a shared toolkit |
| Injection library cannot be preloaded, followed by `aclInit` failure | Worker and companion library are incomplete or mismatched | Install both from the same CANN package |
| Sibling `.pto` not found | Incomplete build source pair | Select a PTOAS directory containing both `.cpp` and `.pto` |
| Export reports no dump file | CANN version emitted traces during collection | Inspect the collection output before treating export as failed |
| Near-empty trace | Synthetic control inputs or scalar defaults | Wire a representative standalone workload and recollect |

## Reporting checklist

Report:

- source build and generated function;
- target, CANN/camodel selection, and whether any toolchain fallback was used;
- collection status and artifact directory;
- workload-defining tensor and scalar values;
- total and per-pipeline cycles;
- evidence that expected cube/vector work executed; and
- the device benchmark used to confirm the conclusion.
