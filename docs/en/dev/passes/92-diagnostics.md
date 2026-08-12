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

Inspects every `tile.load` and `tile.store` op. When the innermost-dimension byte size (`shape[-1] * sizeof(dtype)`) is below `GetRecommendedInnermostDimBytes()`, emits a diagnostic pointing at the source span. The check applies to **every memory space, cube-private included** (issue #2309): `tile.load` / `tile.store` are the only ops inspected, and their non-tile side is a `TensorType` — always off-chip — so both are GM↔chip transfers whose GM side crosses L2 at the innermost-dim granularity no matter which space the tile lands in. Chip-internal movement (`tile.move` / `tile.extract`, e.g. Mat→L0) is genuinely cube-private and is not inspected at all. An earlier revision skipped `Mat`/`Left`/`Right`/`Acc` tiles on L2-locality grounds; because no chip-internal transfer ever reached the check, that skip could only suppress true positives — notably the GM→Mat weight load of a `b_trans` matmul, whose loaded window is the caller's `[N, K]` slice transposed (128 B rows against a 512 B recommendation, a measured 16–25 % penalty).

Each hint also reports the **transfer volume** — `moves <total>B as <rows> x <innermost>B rows` — where `rows` is the number of separate short bus transactions the transfer issues. Volume is what ranks a hint against its siblings: a `[1024, 64]` weight panel and a `[16, 64]` activation panel agree on span, op, dtype, innermost size and memory space, and differ only by 64× in traffic. The clause is omitted when an outer dim is symbolic. Hits are **deduplicated** by `(file, line, col, op, dtype, innermost_bytes, target_memory, transfer_bytes)` site — loop-unroll / per-fragment expansion that produces many *identical* tile transfers at the same source span collapses to one hint carrying an occurrence count, while distinct transfers sharing a span (different dtype/size/memory) each keep their own hint so the count is never misleading. Hits at an invalid/unknown span are emitted individually rather than collapsed together. The message echoes the `(dtype[innermost], target_memory)` tuple it evaluated so the byte figure can be reconciled against the IR. With a `ReportInstrument` in the context the full set goes to `perf_hints.log` and the console only sees the summary line; without one, each hint prints to stderr.

> **Source-span limitation:** the span is the post-pipeline IR-text location (`<string>:line:col`), not the originating DSL `pl.at` / slicing expression, and the controlling chunk constant is not named. Mapping back to user source and naming the inner-dim constant requires DSL source spans to be threaded through the parser/IR onto the tile op — that metadata is not yet carried on `TileType`, the `Call` op, or `Span` (issue #1305 asks 2/3).

Example: console summary plus `perf_hints.log` content (from `examples/intermediate/05_assemble.py` on Ascend910B). Note the first two lines — same op, span, dtype, innermost size and memory space, separated only by volume:

```text
# stderr:
[perf_hint] 4 hints across 4 sites; see /tmp/build/perf_hints.log

# /tmp/build/perf_hints.log:
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 64B (tile fp32[16], target_memory=Mat); moves 1024B as 16 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/intermediate/05_assemble.py:70:5
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 64B (tile fp32[16], target_memory=Mat); moves 2048B as 32 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/intermediate/05_assemble.py:70:5
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 128B (tile fp32[32], target_memory=Mat); moves 4096B as 32 x 128B rows; recommended >= 512B for backend a2a3 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/intermediate/05_assemble.py:70:5
[perf_hint PH001] TileInnermostDimGranularity: tile.store has innermost dim = 128B (tile fp32[32], target_memory=Vec); moves 4096B as 32 x 128B rows; recommended >= 512B for backend a2a3 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/intermediate/05_assemble.py:83:9
```

**Hint volume.** Because cube-space transfers are now in scope, kernels that stage many small GM→Mat panels see more PH001 lines than before. The hints are advisory and land in `perf_hints.log` (stderr gets one summary line); rank them by the `moves …` clause and silence the check wholesale with `disabled_diagnostics` if a kernel's tiling is deliberate.

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
