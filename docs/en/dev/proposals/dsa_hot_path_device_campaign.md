# Device Task: Explain and Expand DSA-RP Hot-Path Effects

## Goal

Complete the original 20-kernel DSA-RP study, explain every reproducible
speedup or slowdown with controlled fixed placements, and test the
completion-frontier model prospectively on a larger kernel set.

The working conjecture is:

> Cross-buffer address reuse is costly when it introduces a late, frequently
> executed predecessor that extends the next consumer's asynchronous completion
> frontier.

Synchronization counts are diagnostic. Kernel-only device latency is the
performance result.

This task does not tune DSA algorithms, fit a global cycle weight, introduce
negative penalties, or change the non-negative pairwise DSA-RP problem.

## 1. Exact revisions

Fetch exact revisions over HTTPS. Stop rather than substituting a branch tip.

| Component | Repository | Revision |
| --- | --- | --- |
| PyPTO | `https://github.com/tonibohnlein/pypto.git` | `63413940e1db791556fd1830c255554cf930d7e9` |
| dsa-solver | `https://github.com/tonibohnlein/dsa-solver.git` | `553b9ce933711e8d78363475c81a9e1ca3b44466` |
| PyPTO-Lib | `https://github.com/hw-native-sys/pypto-lib.git` | `6e897cd99c28767b22e05f209da3e041f15c3dfc` |
| PTOAS | `https://github.com/tonibohnlein/PTOAS.git` | `007f2d637059d907a08faece045e6d3d82943d4b` |
| runtime | PyPTO submodule | `8cdb306cb9a81ad1a0561325021105c676a69c1e` |
| pto-isa | `runtime/pto_isa.pin` | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` |

Use one fresh artifact root:

```text
/opt/dsa-rp-hot-path-expansion
```

Record exact revisions and clean worktree status before and after.

## 2. Resource and hygiene rules

- Use at most two workers for builds, tests, compilation, and analysis.
- Set `PYPTO_CODEGEN_MAX_WORKERS=1`.
- Run at most one process per NPU and two device processes globally.
- Use fresh output directories for every endpoint.
- Do not edit exported DSA problems, raw candidate records, solutions, or PTO.
- Put drivers and generated results under the artifact root, not checkouts.
- Bound device commands with `timeout --kill-after=30s`.
- Record device health and processes before and after.
- Preserve failed endpoints with a terminal classification.

## 3. Build and host preflight

Build dsa-solver Release with testing and require all CTests to pass:

```bash
cmake -S dsa-solver -B dsa-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DDSA_ENABLE_MINIMALLOC_BASELINE=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/dsa-install"
cmake --build dsa-build --parallel 2
ctest --test-dir dsa-build -j2 --output-on-failure
cmake --install dsa-build
```

Install the pinned runtime and PyPTO in a fresh environment. Build PyPTO with
DSA enabled against that solver. Require:

```text
is_dsa_solver_available() == True
MemoryPlanner.DSA exists
DsaReusePenaltyRecognizer.QUADRATIC exists
PassContext round-trips DSA export, replay, recognizer, and reference fields
```

Run with at most two workers:

```bash
python -m pytest tests/ut/tools -n 2 --maxprocesses 2 -q
```

Build PTOAS at the exact revision and record its binary SHA-256. Require:

```text
--enable-insert-sync
--pto-insert-sync-summary
--pto-insert-sync-debug=3
```

## 4. Endpoint definitions

For every kernel, compile two normal solver endpoints:

```text
compact:
  memory_planner = DSA
  recognizer = DISABLED
  reference_placement = DEFAULT

rp:
  memory_planner = DSA
  recognizer = QUADRATIC
  reference_placement = DEFAULT
```

Both must fit capacity and pass independent solution validation. The DSA
problem's solver-relevant semantic fingerprint must match between endpoints.

Before any device run, prepare the pair with:

```bash
python .claude/skills/incore-profiling/preflight_standalone_comparison.py \
  --baseline-build <compact-build> \
  --candidate-build <rp-or-ablation-build> \
  --function <target-function> \
  --invocation-profile <kernel-invocation.json> \
  --ptoas-bin <exact-ptoas> \
  --ptoas-root <PTOAS-checkout> \
  --output-root <kernel-preflight> \
  --build-npu --pto-isa-root <pto-isa> \
  --soc-version Ascend910B2
```

Require `preflight.json` to show:

```text
post_insert_sync_compiled = true
npu_cases_built = true
identical ABI, block_dim, scalar values, and pointer inputs
at least one recommended output
```

The preflight resolves the target from persisted PTO. A missing saved `.cpp`
is not a skip: regenerate it from the persisted level-3 `.pto`.

## 5. Complete the original 20 kernels

Reconstruct the exact target list from the prior archive:

```text
/opt/pypto/dsa-rp-top20-sync.tar.gz
SHA-256 75c1d8395a5c74c5b9cd3e5ee2751cdf6b8e3a775061489d0d2063a450b394dd
```

Verify the hash before use. The archive's `results/targets.tsv` is the
authoritative 20-row selection; do not reconstruct the list from names in old
reports. Produce one canonical `results/top20-ledger.tsv`.

The prior report says 14 kernels were timed. Six were blocked:

| Kernel(s) | Count | Previous blocker | Required repair |
| --- | ---: | --- | --- |
| `kv_proj`, `gate_up_proj`, `mtp_projection_linear_aic` | 3 | invalid synthetic scalar/index bounds, device 507015 | source-proven invocation profiles with valid exact scalars and controls |
| Qwen3-32B `out_proj`, `rope_kv_cache` from two programs | 3 | standalone `.cpp` not persisted by the old harness | resolve and regenerate from persisted PTO |

Two additional reports, `rope_kv_cache` and
`rmsnorm_rope_cache_write`, exist but were omitted from the old consolidated
TSV. Recover them and remove the discrepancy.

For invocation profiles:

- every scalar ABI argument is explicit;
- scalar bounds cover the final tensor tile;
- nonzero index/control arrays use `pointer_fills` or exact files;
- values are justified from the source program or a real dispatch;
- both endpoints use the same profile;
- full-model compact and RP executions pass golden before standalone timing.

All 20 rows must end in one state:

```text
TIMED
NO_PLACEMENT_CHANGE
CORRECTNESS_BLOCKED
COMPILE_BLOCKED
INPUT_PROFILE_BLOCKED
```

Do not silently drop a row.

## 6. Reconfirm the existing effects

Time all valid top-20 endpoints with synchronized binaries on two quiet
devices. The effects requiring explanation are:

| Kernel | Prior observation | Current explanation |
| --- | ---: | --- |
| `mtp_projection_quant` | beneficial; corrected two-device result about 17.8% | mostly explained by a hot loop-carried load-destination handoff |
| `qk_norm` | about 27.5% faster in the broad run | unexplained |
| `mtp_projection_norm` | about 16.8% faster | unexplained |
| `rope_cs` | about 11.2% faster | unexplained |
| `rmsnorm_rope_cache_write` | about 4.7% faster | unexplained |
| `prefill_hca_c128_rmsnorm_rope` | about 9% slower | unexplained |
| `mtp_projection_rms` | about 19% faster in a later study | only a five-group bundle is known |
| `down_dual_proj` | about 0.5% faster | L1 handoffs implicated but not isolated |

Treat these numbers as previous observations, not expected values. Reclassify
them using the exact current pins and two-device protocol.

## 7. Explain every reproducible effect

Only continue with a kernel when compact versus RP has a reproducible material
effect on both devices. For each such kernel:

1. Compute the compact/RP physical-overlap symmetric difference.
2. Join changed pairs to raw v4 candidate records using logical buffer identity,
   range, route, loop, and access sites.
3. Map each pair into the function-aware synchronized PTOAS program.
4. Identify the final predecessor releasing the affected consumer.
5. Estimate dynamic execution count from loop structure; record unknown bounds
   as symbolic rather than inventing a number.
6. Rank candidate handoffs by completion lateness and dynamic frequency.
7. Freeze predictions before compiling ablations.

Construct:

- one-pair `on/off` endpoints for independently placeable candidates;
- leave-one-out endpoints when RP removes several overlaps;
- cumulative ladders in predicted priority order;
- a translated address control preserving the complete overlap graph; and
- a covered-frontier control where a later unavoidable release already exists.

Every endpoint must pass:

```text
exact intended overlap XOR
no unintended overlap changes
same capacity and hard constraints
PyPTO replay validation
address-only normalized pre-InsertSync PTO difference
function-aware synchronization attribution
standalone compile preflight
bit-identical outputs
```

An effect is `EXPLAINED` only when:

- the isolated or cumulative mechanism predicts its sign on both devices;
- the cumulative ladder accounts for at least 80% of the compact/RP median
  difference, or its confidence interval includes the full compact/RP effect;
- the address control reproduces topology and latency class; and
- no unmodeled topology change is required.

Otherwise record `PARTIALLY_EXPLAINED` or `UNEXPLAINED`. Do not force every
effect into the synchronization model. If synchronized topology and overlap
topology are unchanged but latency moves reproducibly, classify it as a
separate address-layout mechanism candidate.

## 8. Prospective expansion

Screen the full available PyPTO/PyPTO-Lib inventory. Target at least 40 new
kernel instances from at least 10 programs:

| Stratum | Target |
| --- | ---: |
| predicted exposed completion-frontier extensions | 24 |
| route-matched but already-covered/drained controls | 8 |
| low-frequency or distance-zero controls | 8 |

Minimum memory-space coverage:

```text
UB/Vec: at least 20
L1/Mat: at least 5
L0A/L0B/L0C combined: at least 5
```

If a space lacks constructible placements or valid standalone inputs, record
`COVERAGE_BLOCKED` with inventory counts. Do not manufacture an overlap or
change the kernel.

Candidate ranking uses:

```text
cross-buffer rather than self-recurrence
complete range and write evidence
overlap mechanically removable
consumer completion frontier extended
predecessor completion lateness
dynamic execution frequency
capacity headroom for a controlled alternative
```

Do not rank by reuse-pair count or synchronization-group count.

For each selected kernel, freeze before PTOAS:

```text
kernel and program
memory space
logical pair and byte subrange
WAR/WAW
source/destination route
consumer
predicted final predecessor
loop distance and dynamic frequency
predicted exposed/covered class
predicted latency direction
```

Use the same fixed-placement and address-control gates as Section 7.

## 9. Timing protocol

Use ACL event timing of the single standalone kernel or canonical mixed group.
Never time an isolated half of a mixed AIC/AIV group.

Initial screen on both devices:

```text
ABBA order
10 warmups per process
8 quartets
100 measured launches per block
3,200 samples per comparison per device
```

Escalate an effect or boundary case to 24 quartets. Use paired bootstrap
confidence intervals. Preserve raw `samples.tsv` and `report.json`.

Classify:

```text
BENEFICIAL: two-device paired CI excludes zero in the faster direction
HARMFUL: two-device paired CI excludes zero in the slower direction
EQUIVALENT: TOST passes at ±max(1%, 0.5 us)
INCONCLUSIVE: none of the above
```

Do not call a one-device observation confirmed.

## 10. Required outputs

Write:

```text
results/top20-ledger.tsv
results/kernel-selection.tsv
results/overlap-deltas.tsv
results/handoff-map.tsv
results/frozen-predictions.tsv
results/sync-attribution.tsv
results/timing.tsv
results/explanation-status.tsv
REPORT.md
HANDOFF.md
```

For each endpoint preserve:

- DSA problem and solution;
- semantic fingerprint;
- invocation profile;
- `preflight.json`;
- pre- and post-InsertSync PTO/C++;
- function-aware summary and debug trace;
- correctness hashes; and
- raw timing samples.

Archive only evidence and regeneration metadata. Exclude build trees and large
regenerable tensors. Record excluded file hashes and recipes.

## 11. Questions the report must answer

1. Did all original 20 kernels reach terminal status?
2. Were the six old infrastructure blockers removed?
3. Which previous effects reproduce on two devices?
4. Which reproduced effects are fully, partially, or not explained?
5. Does exposed completion-frontier extension predict sign prospectively?
6. Does completion lateness rank the magnitude within a consumer?
7. Does dynamic frequency separate material from free handoffs?
8. Which cases are free because an existing drain already covers the handoff?
9. Did any stable effect require an address-layout explanation?
10. Which mechanically recognizable pair families now justify a positive
    non-negative penalty?

The final verdict is one of:

```text
HOT_PATH_MODEL_SUPPORTED
HOT_PATH_MODEL_REFINED
HOT_PATH_MODEL_REFUTED
INFRASTRUCTURE_BLOCKED
PARTIAL
```
