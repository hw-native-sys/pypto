---
name: incore-profiling
description: Profile PyPTO kernels in-core with the Ascend msprof op-simulator or compare standalone compact/loose kernels on a real NPU. Use for per-kernel timing, instruction traces, MindStudio Insight artifacts, or controlled DSA placement experiments.
---

# In-Core Kernel Profiling (msprof op-simulator)

Run cycle-accurate, single-AI-core profiling of every PTOAS kernel in a PyPTO
build via the Ascend `msprof op simulator`. For each kernel the tool generates a
standalone testcase, builds it, runs it on the op-simulator, and collects the
Insight trace artifacts.

This is **runtime** profiling — distinct from the compiler's
`report/perf_hints.log`, which records compile-time hints. Use that file for
"why is codegen suggesting X"; use this skill for "how does the kernel actually
execute".

## When to use

- The user wants per-kernel timing / instruction-level traces of a built case.
- The user asks for MindStudio Insight traces or `trace.json` artifacts.
- The user wants to compare kernel execution cost across changes.

## Prerequisites

- A built case containing sibling `.pto` and `.cpp` files in a top-level or
  nested `ptoas/` directory.
- A TL-capable CANN installation and readable `set_env.sh`; pass
  `--cann-set-env` when auto-discovery selects the wrong installation.
- The `msprof op simulator` worker (`msopprof`). The tool can copy a compatible
  local worker when it is missing; use `--msopprof` to select one explicitly.
- `ptoas-bin` installed. Use `--ptoas-root` only when the full validation
  harness is required.

Per-kernel build and collect directories are private (mode `0700`).

## Quick start

```bash
python .claude/skills/incore-profiling/incore_profile.py \
  --build-dir build_output/<case> --target a2a3
```

- `--target a2a3` for Ascend A2/A3 devices, `--target a5` for A5. This sets the
  compile arch (`dav-c220` / `dav-c310`) and constrains camodel-SoC selection.
- `--case <model.py>` instead of `--build-dir` builds the case first, then
  profiles it. Arguments after `--` are forwarded to the case script.
- `--list-funcs --build-dir <dir>` previews the kernels without running anything
  (needs no toolchain).
- `--func <name>` profiles a single kernel; repeatable.

## Standalone real-device comparison

Compare exact compact and loose PTOAS sources. For large models, prefer
deterministic standalone ABI inputs: full `enable_dump_args=2` capture can
overrun the DFX collector before it finalizes. Captured model inputs remain an
optional validation path for small workloads.

The input `kernel.cpp` must contain the synchronization that will execute on
device. Prefer the `.cpp` emitted by the normal PyPTO/PTOAS compile. If starting
from a pre-InsertSync level-3 `.pto`, regenerate the source with
`--enable-insert-sync`; in the same PTOAS invocation, write an InsertSync
summary and require it to match the topology being studied. Timing a source
generated from pre-sync PTO without this flag is invalid even when it happens
to run correctly.

For ablations, run `prepare_dsa_ablation.py --help` with `dsa_expert_placement_study_v1.json`.
It verifies fingerprints, hard geometry, exact moves, and placement legality,
then writes full replay-ready solution sets and an overlap/objective report. PyPTO
replay remains the authoritative C++ validator before codegen.

For the paper's first device comparison, freeze recognizer provenance and a
broad schema-v1 kernel inventory, then generate all three replayable
solutions in one pass:

```bash
python .claude/skills/incore-profiling/prepare_dsa_device_panel.py \
  --panel panel.json --dsa-bench <dsa-solver-build>/dsa-bench \
  --output-root <prepared-panel>
```

The default protocol includes every currently eligible A2/A3 kernel exported by
the selected PyPTO/PyPTO-Lib programs, without a buffer-count threshold, and
retains historical winners as forced attempts. The manifest is written before
any solver runs. Its three device arms are geometry FirstFit, Cypress
relaxation, and DSA-RP canonical greedy. Geometry canonical greedy and DSA-RP
local search remain model-study algorithms; earlier results show no useful
separation on the easier compiler corpus, so they do not consume device time in
the exploratory screen.

The input manifest has this shape (repeat `kernels` to the required stratum
counts):

```json
{
  "schema_version": 1,
  "selection_policy": "all_current_eligible_plus_historical_winners_v1",
  "recognizer": {
    "policy": "quadratic_unit_v0",
    "source_sha256": "<recognizer-source-sha256>"
  },
  "kernels": [{
    "tag": "stable-kernel-id",
    "program": "source-program-id",
    "kernel": "logical-function-id",
    "selection_class": "historical_winner",
    "problem": "relative/or/absolute/problem.dsa.json"
  }]
}
```

For a single buffer-pair experiment, construct the disjoint/overlapping pair
and its translated address control jointly:

```bash
python .claude/skills/incore-profiling/construct_dsa_pair_isolation.py \
  --problem <problem.dsa.json> --base-solution <base.dsa.solution.json> \
  --first-buffer <id> --second-buffer <id> --output-root <out>
```

The constructor preserves each target's complete overlap signature with every
unrelated buffer and emits `D0`, `O0`, `D1`, and `O1` only when the two disjoint
geometries and two overlapping geometries match exactly. Replay still provides
the authoritative compiler-side validation.

For repeatable campaigns, store launch metadata in one portable invocation
profile. It keeps exact scalar bounds and nonzero control inputs identical:

```json
{
  "schema_version": 1,
  "block_dim": 1,
  "input": {"kind": "synthetic", "seed": 19},
  "scalars": {"v4": 0, "v5": 0, "v6": 256},
  "pointer_fills": {"v1": 32},
  "outputs": ["v3"]
}
```

Preflight a baseline/candidate pair before using a device:

```bash
python .claude/skills/incore-profiling/preflight_standalone_comparison.py \
  --baseline-build <baseline-build> --candidate-build <candidate-build> \
  --function <kernel> --invocation-profile invocation.json \
  --ptoas-bin <instrumented-ptoas> --ptoas-root <PTOAS-checkout> \
  --output-root <preflight>
```

This resolves the target from persisted PTO, regenerates both C++ sources with
InsertSync, checks summaries, inputs, ABI, and launch metadata, and writes
`preflight.json`. Add `--build-npu --pto-isa-root <pto-isa>` on a host with
CANN to compile both standalone executables using two workers. Then run:

```bash
python .claude/skills/incore-profiling/standalone_compare.py \
  --compact-case <compact-case> --loose-case <loose-case> \
  --output <real-output-abi-name> --device-id 0 \
  --quartets 8 --warmup 10 --rounds 100 --output-root <results>
```

For three or more placements, use the balanced multi-arm driver. `--case` is
repeatable; with three variants it runs three cyclic orders and their reverses,
so every placement occurs twice in every process position:

```bash
python .claude/skills/incore-profiling/standalone_multi_compare.py \
  --case geometry_ff=<case> --case cypress=<case> --case dsa_rp_cg=<case> \
  --output <real-output-abi-name> --device-id 0 \
  --warmup 10 --rounds 100 --output-root <results>
```

The driver validates identical ABI, inputs, captured expectations, and output
hashes before timing. It writes all pairwise paired-block bootstrap intervals;
cross-device confirmation remains an experiment-level reporting decision.

After all kernels finish, aggregate the frozen panel per device. Repeat
`--report` for every kernel/device pair; never pool or average devices:

```bash
python .claude/skills/incore-profiling/summarize_dsa_device_panel.py \
  --report kernel_a@device4=<report.json> \
  --report kernel_a@device5=<report.json> \
  --output-root <panel-results>
```

The summary reports geometric-mean candidate/reference latency ratios and a
kernel bootstrap interval separately for each device. The per-kernel table is
retained so nulls and sign reversals remain visible.

Synthetic inputs are bounded and emitted in chunks. Integer pointers default to
zero; use `pointer_fills` or file inputs when kernels require indices,
sentinels, or positive extents. For pure NPU kernels, the generator also bounds
post-PTOAS `GlobalTensor` partition offsets across loops and SPMD blocks. It
allocates the complete physical span and rejects unresolved expressions or file
inputs shorter than that span. Every scalar must satisfy tensor-view bounds.

Mixed AIC/AIV inputs are launched as one co-scheduled group. They require
`--ptoas-root`; the generator reuses PTOAS's validation-harness wrapper, which
merges both bodies and their shared pipe objects into one global kernel. It
never launches an AIC or AIV half in isolation. PyPTO's synthetic block-index,
block-count, and AIV-lane parameters are removed from the host ABI and rebound
to direct-launch hardware builtins. In particular, the two vector lanes receive
distinct `get_subblockid()` values rather than one host-provided scalar.

For a small workload where exact model values matter, use
`--args-dump <args_dump.json> --func-id <id> --task-id <id>` instead of
`--synthetic-inputs`. Import rejects mixed AIC/AIV, incomplete, non-contiguous,
ambiguous, or truncated captures.

The driver verifies ABI, launch metadata, inputs, and captured outputs. It
restores inputs per launch, times with `aclrtEventElapsedTime`, runs serial ABBA
quartets, and writes `samples.tsv` plus `report.json`.

For broad DSA studies, use model compilation only to discover about 20
high-signal kernels or mixed groups. Rank by sync change, removed reuse pairs,
and useful work; exclude unchanged, trivial, and failed cases.

CANN, the camodel SoC, and the compile arch are auto-resolved from `--target`.
Override any of them with `--cann-set-env`, `--soc-version`, `--aicore-arch`.

## Output

Each run writes to `<build-dir>/kernel_insight_all_funcs_<timestamp>/`:

- `manifest_export.csv` and `summary.txt` — index and per-kernel status.
- `funcs/<kernel>/collect/out/OPPROF_*/simulator/` — Insight traces and
  per-core instruction data.

A final `EXPORTED N/M` line reports how many kernels succeeded.

### Clean and store the trace

```bash
python -m pypto.tools.clean_sim_trace \
  <build-dir>/kernel_insight_all_funcs_<ts>/funcs/<kernel>/collect/out/OPPROF_* -o <out>
```

Store the result under gitignored
`build_output/incore_<kernel>_<source>_<timestamp>/`, not `/tmp`. Preserve the
clean trace, instruction metrics, raw simulator data, and a provenance summary
that records inputs and scalars.

## Troubleshooting

- **`__biasbuf__` / `aicore` compile errors** — select a TL-capable CANN with
  `--cann-set-env`.
- **Build fails inside `pto/npu/a5/*.hpp`** — wrong target; pass `--target a2a3`
  for an A2/A3 device.
- **Missing `runtime_camodel`** — select the installed variant with
  `--soc-version`, commonly `Ascend910B1`.
- **Trace is ~0 cycles / `CUBE=0` / only SCALAR+sync instrs** — the kernel is
  data-dependent and its control input is invalid; use an invocation profile.
- **Missing CANN or `msopprof`** — pass the corresponding explicit path.
- **Missing sibling `.pto`** — use a complete `ptoas/` directory or the full
  validation generator through `--ptoas-root`.

## Caveats

Synthetic integer inputs can make data-dependent loops execute zero times. A
near-empty or `CUBE=0` trace is not performance evidence; replace control inputs
and scalar bounds with a real workload. NPU capture mode does this automatically.

## How it works

The generator derives ABI and physical spans from `.cpp`/`.pto`, emits the
harness, builds, runs `msprof op simulator`, and records each kernel
independently. Mixed NPU kernels use PTOAS's co-scheduled validation wrapper.
