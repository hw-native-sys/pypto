# Compile Profiling

PyPTO includes built-in compile profiling that records wall-clock timings at
each stage of the compilation pipeline — from frontend parsing through compiler
passes and code generation to on-device execution.

## Quick Start

### Option 1: Environment Variable

```bash
PYPTO_COMPILE_PROFILING=1 python3 my_program.py
```

### Option 2: `ir.compile()` Parameter

```python
output_dir = ir.compile(program, profiling=True)
# Results are written to output_dir/report/pipeline_profile.{txt,json}
```

### Option 3: Context Manager

```python
from pypto.compile_profiling import CompileProfiler

with CompileProfiler() as prof:
    @pl.program
    class MyProgram:
        ...
    ir.compile(MyProgram, ...)

print(prof.summary())
prof.to_json("profile.json")
```

### Option 4: `RunConfig`

```python
from pypto.runtime import run, RunConfig

result = run(
    program=MyProgram,
    tensor_specs=specs,
    golden=golden_fn,
    config=RunConfig(compile_profiling=True),
)
# result.profile contains the profiling data as a dict
```

## Output

### Human-Readable Summary (`pipeline_profile.txt`)

```text
PyPTO Compile Profile
======================
Total: 2.847s

  parse                            0.023s  ( 0.8%)
  passes                           1.204s  (42.3%)
    UnrollLoops                     0.012s  ( 0.4%)
    ConvertToSSA                    0.034s  ( 1.2%)
    ...
    AllocateMemoryAddr              0.156s  ( 5.5%)
  codegen                           0.418s  (14.7%)
    kernel_codegen:my_kernel        0.312s  (11.0%)
    orchestration_codegen           0.106s  ( 3.7%)
  device_execution                  1.202s  (42.2%)
```

### Structured JSON (`pipeline_profile.json`)

```json
{
  "total_seconds": 2.847,
  "stages": [
    {"name": "parse", "seconds": 0.023, "children": []},
    {"name": "passes", "seconds": 1.204, "children": [
      {"name": "UnrollLoops", "seconds": 0.012, "children": []},
      {"name": "ConvertToSSA", "seconds": 0.034, "children": []}
    ]},
    {"name": "codegen", "seconds": 0.418, "children": [
      {"name": "kernel_codegen:my_kernel", "seconds": 0.312, "children": []},
      {"name": "orchestration_codegen", "seconds": 0.106, "children": []}
    ]}
  ]
}
```

## Stage Hierarchy

The profiler records the following stages when using `runtime.run()`:

| Stage | Description |
| ----- | ----------- |
| `compile` | Full compilation (wraps `ir.compile()`) |
| `parse` | `@pl.program` decorator AST parsing |
| `passes` | Pass pipeline execution |
| Per-pass stages | Individual pass timings (e.g., `UnrollLoops`, `AllocateMemoryAddr`) |
| `codegen` | Code generation |
| `kernel_codegen:<name>` | Per-kernel PTO/ptoas codegen |
| `orchestration_codegen` | Orchestration C++ codegen |
| `golden_write` | Golden reference file generation |
| `device_execution` | On-device compilation and execution (Simpler) |

When using `ir.compile()` directly, only `passes` and `codegen` (with
sub-stages) are recorded.

## Programmatic API

```python
from pypto.compile_profiling import CompileProfiler, get_active_profiler

# Check if profiling is active (explicit or via env var)
prof = get_active_profiler()

# Use as context manager
with CompileProfiler() as prof:
    # Record custom stages
    with prof.stage("my_custom_stage"):
        do_something()

    # Access results
    data = prof.to_dict()      # dict
    text = prof.summary()      # human-readable string
    json_str = prof.to_json()  # JSON string
    prof.to_json("out.json")   # write to file
    prof.write_report("dir/")  # write both .txt and .json
```

## Overhead

When profiling is **not** enabled (the default), the overhead is a single
`CompileProfiler.current()` null-check per stage boundary — effectively zero.

When profiling **is** enabled, each stage records two `time.perf_counter()`
calls (sub-microsecond on modern hardware).

## Related

- **Runtime DFX** (`RunConfig.enable_l2_swimlane`, `enable_dump_tensor`,
  `enable_pmu`, `enable_dep_gen`) drives Simpler's per-task diagnostic
  artefacts — swimlane records, tensor I/O dumps, AICore PMU CSVs, and
  PTO2 dep_gen edges. The four flags are independent, share
  `<work_dir>/dfx_outputs/` as their output root, and are orthogonal to
  compile profiling. See [03-runtime-dfx.md](03-runtime-dfx.md).
