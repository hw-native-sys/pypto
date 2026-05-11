# Diagnostics: Warnings and Performance Hints

Unified advisory diagnostic channel for the pass pipeline. Surfaces warnings (likely user mistakes) and performance hints (advisory tuning suggestions) through a single registry, instrument, and output path.

## Overview

| Component | Purpose |
| --------- | ------- |
| **`Diagnostic` struct** | Carries severity (`Error` / `Warning` / `PerfHint`), `rule_name`, `error_code`, `hint_code`, message, span. |
| **`DiagnosticCheck` enum** | Identifies a specific check (e.g. `UnusedVariable`, `TileInnermostDimGranularity`). |
| **`DiagnosticCheckRegistry`** | Maps checks to verifier factories; each registration declares severity, phase, and hint code. |
| **`DiagnosticInstrument`** | `PassInstrument` that runs registered checks and dispatches output. |
| **`DiagnosticPhase`** | When a check fires: `PrePipeline`, `PostPass`, or `PostPipeline`. Per-check, declared at registration. |

Severity is independent of phase. A `Warning` may run at `PrePipeline`; a `PerfHint` may run at `PostPipeline` — declared per check.

## Severities

| Severity | When | Output | Suppression |
| -------- | ---- | ------ | ----------- |
| `Error` | IR is invalid | `VerificationError` thrown | Cannot be suppressed |
| `Warning` | Likely mistake or pass bug | `LOG_WARN` to stderr, in full | `disabled_diagnostics` set |
| `PerfHint` | Advisory tuning suggestion | `${ReportInstrument.output_dir}/perf_hints.log` if a report instrument is in the context (always true via `compile()` / `pl.jit`), with stderr getting a one-line `LOG_INFO` summary that points at the file. With no report instrument: each hint printed in full to stderr via `LOG_INFO`. | `disabled_diagnostics` set |

The release default for `PYPTO_LOG_LEVEL` is `INFO`, so the `[perf_hint] N hints …` summary (or, without a report instrument, the full `[perf_hint PH…] …` lines) reaches the console out of the box. Override with `PYPTO_LOG_LEVEL=warn` to mute perf hints on stderr. When a report instrument is present the per-hint detail still lands in `perf_hints.log` regardless of log level (the file output is independent); without one there is no file, so muting stderr drops perf hints entirely.

## How a check fires

```text
PassPipeline::Run(program)
 ├─ if phase != None: run PrePipeline checks  → EmitDiagnostics
 ├─ for each pass:
 │    ├─ run pass
 │    ├─ if phase != None: run PostPass checks → EmitDiagnostics
 ├─ if phase != None: run PostPipeline checks  → EmitDiagnostics
 └─ ctx.RunAfterPipeline(program)            (instrument hooks)
```

`EmitDiagnostics` prints Warnings in full via `LOG_WARN`. For PerfHints: when a `ReportInstrument` is in the active context it appends every hint to `perf_hints.log` and emits a single `LOG_INFO` summary line (`[perf_hint] N hints across M sites; see <path>`) to stderr; otherwise it prints each hint in full via `LOG_INFO`.

## Registering a new check

```cpp
// 1. Implement a PropertyVerifier subclass (src/ir/verifier/...)
class MyCheck : public PropertyVerifier {
 public:
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diags) override;
  std::string GetName() const override { return "MyCheck"; }
};
PropertyVerifierPtr CreateMyCheckVerifier() { return std::make_shared<MyCheck>(); }

// 2. Add an enum value (include/pypto/ir/verifier/diagnostic_check_registry.h)
enum class DiagnosticCheck : uint32_t { ..., MyCheck = N };

// 3. Register in DiagnosticCheckRegistry::DiagnosticCheckRegistry()
Register(DiagnosticCheck::MyCheck,
         DiagnosticSeverity::PerfHint,
         DiagnosticPhase::PostPipeline,
         "PHnnn",
         CreateMyCheckVerifier);
```

The registry stamps the registered severity and hint code onto every diagnostic the verifier emits.

## Performance hints (issue #1180)

Performance hints are best-effort advisory diagnostics that flag patterns likely to under-utilise the target hardware. They are on by default at `DiagnosticPhase::PostPipeline` and run after the IR is fully lowered.

### Per-backend thresholds

`BackendHandler` exposes:

| Method | Ascend910B | Ascend950 |
| ------ | ---------- | --------- |
| `GetGmAccessGranularityBytes()` | 512 | 128 |
| `GetL2CacheLineBytes()` | 512 | 512 |
| `GetRecommendedInnermostDimBytes()` | 512 | 128 |

Adding a new backend implements these alongside the existing virtuals; perf-hint checks consult them via `PassContext::Current()->GetBackendHandler()`.

### First check: `TileInnermostDimGranularity` (PH001)

Inspects every `tile.load` and `tile.store` op. When the innermost-dimension byte size (`shape[-1] * sizeof(dtype)`) is below `GetRecommendedInnermostDimBytes()`, emits one diagnostic per op pointing at the user-source span. With a `ReportInstrument` in the context the full set goes to `perf_hints.log` and the console only sees the summary line; without one, each hint prints to stderr.

Example: console summary plus `perf_hints.log` content (from `examples/kernels/08_assemble.py` on Ascend950):

```text
# stderr:
[perf_hint] 2 hints across 2 sites; see /tmp/build/perf_hints.log

# /tmp/build/perf_hints.log:
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 64B; recommended >= 128B for backend a5 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/kernels/08_assemble.py:60:4
[perf_hint PH001] TileInnermostDimGranularity: tile.store has innermost dim = 64B; recommended >= 128B for backend a5 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/kernels/08_assemble.py:60:4
```

## User-facing API

```python
# No report instrument: each perf hint prints in full on stderr at end of pipeline.
with passes.PassContext([]):
    pipeline.run(program)

# With a report instrument (the compile()/pl.jit default): full detail goes to
# perf_hints.log next to the other reports; stderr just gets the one-line summary.
with passes.PassContext([passes.ReportInstrument("/tmp/build")]):
    pipeline.run(program)
# → /tmp/build/perf_hints.log  (stderr: "[perf_hint] N hints across M sites; see …")

# Suppress a specific hint.
disabled = passes.DiagnosticCheckSet()
disabled.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
with passes.PassContext([], disabled_diagnostics=disabled):
    pipeline.run(program)

# Disable the whole channel.
with passes.PassContext([], diagnostic_phase=passes.DiagnosticPhase.NONE):
    pipeline.run(program)
```

`compile()` and `run()` accept the same parameters via `diagnostic_phase` and `disabled_diagnostics` kwargs.

## Environment variables

| Variable | Effect | Default |
| -------- | ------ | ------- |
| `PYPTO_LOG_LEVEL` | Threshold for stderr output (`debug`/`info`/`warn`/`error`/`fatal`/`event`/`none`) | `info` (release) / `debug` (non-release) |
| `PYPTO_WARNING_LEVEL` | Default `DiagnosticPhase` (`none`/`pre_pipeline`/`post_pass`/`post_pipeline`) | `pre_pipeline` |
| `PYPTO_VERIFY_LEVEL` | Verification level — orthogonal to diagnostics | `basic` |
