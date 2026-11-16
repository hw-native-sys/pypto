# Gloo and symmetric-memory windows

> Closeout scope: performance coverage, final review/regression, and handoff only.
> Earlier next-step notes below are historical, not outstanding optimization tasks.
> See the updated-base model acceptance section for the current validation status.

Path B runs one process per rank on one machine. Gloo provides CPU coordination;
torch_npu symmetric memory provides peer-accessible windows. PyPTO chip kernels
perform TSTORE, TPUT and a multi-rank sum using `remote_load` plus `add`.
The entry point is `examples/distributed/08_gloo_shmem_collectives.py`.

## Window and context contract

`pypto.runtime.shmem_gloo` carves slots at 64-byte boundaries, matching distributed
codegen. The first slot, `_comm_ctx`, contains the 1088-byte runtime `CommContext`:

| Field | Byte offset |
| ----- | ----------- |
| rankId | 0 |
| rankNum | 4 |
| winSize | 8 |
| windowsIn[64] | 16 |
| windowsOut[64] | 528 |
| workspaces[3] | 1040 |

Each workspace is an `(address, size)` pair of uint64 values. All three are zero
for this path. Workspace-dependent SDMA/URMA/RDMA operations are unsupported.
The context lives in the symmetric heap, alongside the data, and must outlive
all kernels that receive its pointer.

`pld.import_window_buffer` (also `pld.tensor.import_window_buffer`) declares a
named slot using the same IR marker as `alloc_window_buffer`. Its assignment LHS
is the binding name; it does not embed a host address in IR. The alias alone does
not change allocation behavior: generated HOST code needs an explicit provider.
`RunConfig.domain_provider` forwards that provider to generated HOST entry points
and is excluded from compilation options. Persistent distributed workers reject
external providers because their domain-retention protocol owns those lifetimes.

`make_shmem_domain_provider` binds named slots through the current simpler
`ChipDomainContext.buffers` interface. Pass an initialized **direct L2** simpler
worker; its public `Buffer.wrap` / `device_location` API registers external
storage. Domain release revokes registrations without freeing the SHMEM heap.
Keep the domain alive until its synchronous dispatch drains. The provider checks
rank mapping, slot capacity, workspace requests, worker selection and process
identity. It cannot authorize SHMEM mappings in an L3 child.

## Running the two-rank example

Use matching PyPTO, its pinned simpler runtime, torch/torch_npu, CANN and SHMEM
installations, and two available A2/A3 devices. Source the SHMEM environment, then:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/distributed/08_gloo_shmem_collectives.py -p a2a3
```

On a shared machine, submit the launch through its device queue and set
`ASCEND_RT_VISIBLE_DEVICES` to the allocated device list before starting torchrun.
Local ranks then index those visible devices.

The example uses direct `ChipWorker` dispatch rather than generated HOST/L3
execution. It constructs the worker with `auto_init=False`, then calls
`init(borrowed_context_generation=1)` to borrow the already-current torch context.
The token identifies this process's single context generation; a reset/recreated
context requires a new token. Closing the worker does not reset torch's device.

Data operands use address-free `TaskArgs` backed by `Buffer.wrap`; materialized
CommContext pointers use uint64 scalar slots. Inputs and outputs remain NPU
tensors. Each launch synchronizes torch work first and passes the current caller
stream to `_run_chip`; this deliberately serial example does not claim framework
queue overlap. A successful synchronous run retires uses before registrations
are revoked. Gloo barriers separate cross-rank publication and consumption, and
prevent reuse of a slot while a peer still reads it. Golden values come from
actual rank inputs gathered over Gloo. Success prints
`GLOO_SHMEM_COLLECTIVES_PASS` on both ranks.

## Historical findings and limits

The source experiment is `nalinaly/pypto` at `f3b6e7f0`. Its debugging record
reported that a parent SHMEM GVA was not mapped in L3 fork children: TSTORE failed
with MTE `0x800000`. In-process L2 dispatch avoided that address-space mismatch.
Those historical results do not establish correctness on the current runtime.

The same record reported UB overflow when independently launching the builtin
allreduce in two processes. This example uses `remote_load + add` and does not
launch that builtin. NPUSHMEM barrier/put_signal support is not assumed. The collective example supports 2/4/8 ranks; the Qwen runner specializes its shards from `WORLD_SIZE`. No vLLM integration or multi-machine transport is included.

## Port validation status

M1 passed two-rank hardware acceptance on 2026-11-12, using PyPTO `88efefc33`
plus the local port, pinned simpler `355ce301`, CANN 9.3.0, torch 2.10.0+cpu,
torch_npu 2.10.0.post7, cann-shmem 1.6.0 (`backends/910`) and PTOAS 0.72.
The queued run on Ascend A3 devices 0 and 1 passed TSTORE, TPUT and allreduce
against live-input goldens on both ranks. Both printed
`GLOO_SHMEM_COLLECTIVES_PASS`; the job exited 0 and released both device locks.
SHMEM emitted peer-disconnect warnings during teardown after the assertions.
These results cover this two-rank configuration. M2 subsequently passed the
bounded full-model FP32-reference acceptance described below.

The original torch_npu 2.10.0 environment loaded the library but failed SHMEM
initialization. Setting `SHMEM_INSTANCE_PORT_RANGE` exposed implausible,
rank-dependent `instance_id` values rather than fixing it. An isolated
2.10.0.post7 environment passed allocation and rendezvous without that setting.
Do not infer compatibility from a successful library-loading check alone.

Hardware execution also exposed missing scalar entries in the exported
`ChipCallable` signature. The entry passed CommContext scalars while its
registration advertised tensors only, so borrowed execution refused preparation
with `-1004`. Codegen now emits tensor directions followed by scalar entries;
CoreCallable tensor-only signatures are unchanged. All five example regression
cases failed before this fix, then passed with the fix. The focused signature
and orchestration regression passed 50 tests; the expanded orchestration suite
then passed 359 tests (`build/pathb-orchestration-regression.log`). Earlier port regression passed
645 tests with four optional-runtime skips, and the 11 CPU-only window tests
passed with `--noconftest`. These CPU results alone do not establish hardware
correctness. Native bindings and runtime binaries were built in this worktree;
the runtime revision check and all five native kernel compilations passed.

Local logs: `build/pathb-example-onboard-signature.log` (hardware acceptance),
`build/pathb-shmem-probe-post7.log` (environment probe),
`build/pathb-signature-red.log` and `build/pathb-signature-green.log`
(regression before/after), and `build/pathb-example-native-compile.log`.
The local environment is `.venv-npu`; `build/pathb-shmem-env.sh` supplies the
library search paths after loading CANN. The launch used this worktree's
`python`, `runtime/python` and `runtime` on `PYTHONPATH`, and capped native
compilation at one worker per rank to respect the two-job machine budget.

## M2 standalone Qwen3 TP2

`examples/models/qwen3_tp/runner.py` uses real Qwen3-14B safetensors weights,
40 layers, batch one, up to 32 prefill tokens and a 128-position KV cache.
Embedding, RMSNorm, Q/K normalization, RoPE, attention, MLP and LM head are
PyPTO kernels. Each rank loads only its column/row matrix shards, with replicated
embedding, LM head and normalization weights. Python coordinates synchronous
L2 launches. Both o_proj and down_proj use FP32 SHMEM publication, a Gloo
barrier, remote-load/add, then another barrier before slot reuse.

The compute kernels are adapted from the Apache-2.0 reference at
`nalinaly/vllm-ascend` commit `e7cb166290dfcbf2f997aa67f01b323be643fe0e`;
the source attribution and original license are retained beside them. There is
no vLLM runtime dependency or device-side signaling protocol in this runner.

Launch with the repository root on `PYTHONPATH` in addition to the M1 paths,
through the hardware task queue and with the same native compilation limits:

```bash
torchrun --standalone --nproc_per_node=2 -m examples.models.qwen3_tp.runner \
    --model-dir /path/to/Qwen3-14B --steps 2 --output-dir build/qwen3-output
```

`QWEN3_TP_RANK_AGREEMENT_PASS` only establishes finite, matching rank logits.
It is not the M2 numerical acceptance marker. The independent CPU FP32 Hugging Face
check consumes saved logits from both prefill and decode:

```bash
PYPTO_QWEN3_MODEL=/path/to/Qwen3-14B \
PYPTO_QWEN3_OUTPUT=build/qwen3-output/outputs.pt \
python -m pytest --noconftest tests/st/distributed/test_qwen3_tp_reference.py -v
```

The reference checks greedy token agreement, relative RMS error below 3%, and
logit cosine similarity above 0.999. Safetensors and Transformers are optional
example/reference dependencies. The CPU lowering suite passed all 28 Qwen3
entries plus seven M1/shared-dispatch checks; nine weight-layout/configuration
tests passed (44 total, `build/pathb-m2-regression.log`).
CPU tests alone do not establish full-model hardware correctness.

The first full-model smoke job on A3 devices 2 and 3 completed all 40 layers
for prefill and one decode step with real weights. Both ranks produced
`" Paris."` for `The capital of France is`, passed finite-logit/rank-agreement
checks, and exited 0. See `build/pathb-m2-onboard/run.log`. The saved-output run on devices 0 and 1 also exited 0 and released both locks.
Independent CPU Hugging Face FP32 acceptance passed with Transformers 5.5.3:

| Stage | Relative RMS error | Cosine similarity | Greedy token (both) |
| ----- | ------------------ | ----------------- | ------------------- |
| Prefill | 0.005979 | 0.999992 | 12095 (`" Paris"`) |
| Decode | 0.011355 | 0.999941 | 13 (`.`) |

Evidence is in `build/pathb-m2-acceptance/run.log`, `outputs.pt`, and
`reference-fp32.log` (one reference test passed). Together with the 44 CPU
regression checks, this completes M2's first real-weight TP2 acceptance for this
prompt and one decode step. It does not establish long-context, general batch,
4/8-rank correctness or production performance; M3 collective generalization is described below.

**Numerical contract:** weights/activation storage are BF16, while the imported
kernels retain FP32 intermediates across normalization, RoPE, SwiGLU and
projection reductions. Acceptance therefore uses a high-precision FP32 eager
reference with the unchanged 3% relative-error and 0.999 cosine thresholds.
An initial eager **BF16** comparison produced relative errors of 2.26%/3.97%
for prefill/decode: decode failed the 3% threshold, despite matching greedy
tokens. That result is retained in `build/pathb-m2-acceptance/reference.log`.
This implementation does not promise operator-by-operator BF16 equivalence;
applications requiring that contract need separate rounding alignment.

## Layer diagnostics and independent sequences

The runner accepts `--prompts-file prompts.json` (a JSON array of prompt strings)
and resets hidden state and both KV caches between cases while retaining weights
and compiled programs. Each case saves `case_N/outputs.pt`. `--trace-layers`
requires `--output-dir` and captures the last valid token at eleven stages in all
40 layers on rank zero. Tracing synchronizes device copies and is for numerical
diagnosis, not performance measurement.

The CPU reference accepts either one artifact or a directory of cases.
`PYPTO_QWEN3_REFERENCE_DTYPE=bf16` selects eager BF16; the default remains FP32.
Both modes retain the same numerical assertions. `PYPTO_QWEN3_REPORT=report.json`
saves complete logit and per-stage metrics even when numerical assertions fail.
Local RMSNorm, RoPE and SwiGLU checks hold actual kernel inputs fixed and compare
FP32-fused and eager-BF16 rounding. These isolate local arithmetic from drift
already present in a layer's inputs; they do not change model execution.

## Extended numerical validation

The diagnostic matrix uses prompt lengths 1, 5, 16 and 32, each with one prefill
and seven consecutive decode steps. The original kernels completed all 32
steps on two ranks with matching finite logits, including cache positions past
32 and resets between independent sequences. Unmodified HF FP32 failed 1/32
steps: `Hello` prefill had 12.414% relative RMS error. HF BF16 failed 6/32 steps,
including three greedy-token mismatches. These failures are retained in
`build/pathb-m2-trace/{fp32,bf16}.{log,json}`; the earlier single-prompt acceptance
must not be generalized to this broader matrix.

For the original Paris decode BF16 comparison, layer-output error was already
0.883% at layer 0 and first exceeded 1% at layer 2 (1.606%); it reached 1.501%
at layer 39 before the final normalization/head produced 3.966% logit error.
This is accumulated drift rather than a single late-layer discontinuity.

Holding actual device inputs fixed exposed an additional numerical issue:
RMSNorm differed from the FP32-fused CPU calculation by 0.208% on average,
and Q/RoPE by 0.233%, while SwiGLU differed by only 0.000211%. The source used
`pl.rsqrt`'s default approximate A2/A3 `vrsqrt` path. The three normalizer sites
now request `high_precision=True`, selecting the scratch-backed sqrt/divide
implementation. Four lowered-entry checks require high-precision RSQRT; together
with three snapshot/reset tests they passed seven focused checks. The updated
two-rank device run completed all 32 steps with finite, matching rank outputs.
Mean fixed-input RMSNorm error fell to 0.000409%, and Q/RoPE to 0.000119%.
This verifies the local precision improvement, not full-model equivalence:
unmodified HF FP32 still failed 3/32 steps (one token mismatch), and HF BF16
failed 8/32 (two token mismatches). Results are retained in
`build/pathb-m2-rsqrt/{fp32,bf16}.{log,json}`.

`PYPTO_QWEN3_REFERENCE_DTYPE=fused` is a separate CPU diagnostic using full HF
matrices with PyPTO's storage-rounding boundaries. It is not an unmodified eager
reference and does not replace FP32/BF16 acceptance. CPU reference loading disables
Transformers' extra asynchronous loading pool to respect the two-worker budget.

The fused diagnostic passed RMS-error and cosine limits on all 32 updated steps,
but matched greedy tokens on only 31/32. In the 16-token case at step 2, device
logits selected `" human"` (19.210857) over the reference winner `" the"` (19.205627),
a gap of 0.005230. This remains a failed token check; no tolerance was relaxed.
Reports are `build/pathb-m2-rsqrt/{fused,fused_remaining}.{log,json}`.

The exact device midpoint test (`tests/st/distributed/test_bf16_rounding.py`)
confirmed that explicit `rint` matches Torch on all 32 signed BF16 midpoints;
default `round` differs on 16/32. This establishes a conversion-contract
mismatch, but does not by itself explain the remaining model token mismatch.
All nine model BF16 casts now explicitly request `rint`. All 28 chip-entry
compile checks pass, including a check that emitted BF16 conversions use RINT.
The updated device run completed all 32 steps with finite, matching rank outputs.
The fused CPU reference passed all 32 steps and greedy tokens: maximum relative
RMS error 1.9412%, minimum cosine similarity 0.999861. The previous fused token
mismatch is resolved. Unmodified HF FP32 matched all tokens but failed the
`Hello` prefill RMS check (9.7618%); its other 31 steps passed. Unmodified HF BF16 failed
6/32 steps, including three greedy-token mismatches; maximum relative RMS error
was 4.5996%. Reports are in `build/pathb-m2-rint-full/`. Broader eager-reference
acceptance remains open, and M3 transport validation is tracked below.

Reference decode is teacher-forced with the saved actual previous token, so each
comparison uses the same prefix. Token agreement is conditional on that prefix,
not a guarantee of identical independent free-running generation.

## M3 collective generalization

The agreed M2 contract is BF16 storage with FP32 fused arithmetic, validated by
all 32 fused-reference steps above; eager BF16 equivalence is not claimed.
The `08` example accepts 2/4/8 processes. TSTORE/TPUT validate the previous
ring rank; allreduce accumulates the local input and every other rank in FP32,
then casts to FP16 with nearest-even rounding. This is an O(N) read-based
correctness baseline, not a bandwidth-optimized ring algorithm.
Both publication and reuse Gloo barriers remain required. `--iterations 5`
changes each publication and checks every rank against independently gathered
CPU inputs. Compilation is serialized across ranks to respect the host budget.
Seven CPU checks and all 2/4/8-rank device runs passed (TSTORE, TPUT, five sums
on every rank). Each job exited zero and released its cards. Evidence:
`build/pathb-m3-codegen.log` and `build/pathb-m3-{2,4,8}rank.log`. The Qwen runner now supports 2/4/8-way sharding from `WORLD_SIZE`, records the
size in artifacts, and uses rank-ordered FP32 reduction. TP8 uses 128-wide K
tiles to cover its 640/2176-wide row shards without dropping a tail. All 28 chip
entries compile at each size; 16 weight/runner tests pass. TP4 and TP8 completed the
32-step device matrix with matching finite outputs on every rank. TP4 passed
all 32 fused-reference steps and tokens (max relative RMS error 2.2947%, minimum
cosine 0.999802). TP8 also passed all 32 steps and tokens (max relative RMS error
1.9481%, minimum cosine 0.999810). This completes the tested 2/4/8-rank model
scaling milestone under the agreed fused contract. Barrier removal,
fused collectives and stream overlap remain outstanding.
Evidence: `build/pathb-m3-tp{4,8}/`.

## Performance measurement

`examples/models/qwen3_tp/benchmark.py` measures one fixed prompt per queued
allocation. First prepare the artifacts on CPU, with the repository/runtime
paths and the supported PTOAS installation configured:

```bash
WORLD_SIZE=4 python -m examples.models.qwen3_tp.benchmark --prepare \
    --artifacts build/pathb-perf/tp4/artifacts
```

Then run the module through `torchrun --nproc_per_node=4` inside a four-card
TaskQueue job, passing `--artifacts`, `--model-dir`, `--prompt`, `--output`,
`--rounds 5 --steps 8`. Repeat as separate jobs for TP2 and TP8. Capture the
prepared CANN/SHMEM environment in the submission; run the architecture precheck
inside the allocation. Native artifact loading can remain lazy and is excluded
with one complete warm-up sequence. Do not mix model sizes in one allocation.

Each measured sequence resets the KV cache and uses the same prompt. A step
starts after device synchronization and a Gloo alignment barrier. Wall time
includes synchronous L2 launches, model collective barriers, and the final CPU
logits copy; it excludes rank comparisons, token selection, reporting, and the
next step's alignment barrier. Tracing and per-layer logging are disabled.
The slowest rank defines each step sample. Five prefill and 35 decode samples
are retained, with median, interpolated p95 and mean. The decode rate is the
reciprocal of mean forward latency for batch one, not serving throughput.
Inclusive allreduce time covers publish, two barriers and reduction dispatch;
it is not a network-only or device-only measurement. All ranks are checked for
finite, matching logits and warm/measured token repeatability outside timing.
These short shared-host measurements are a baseline, not a performance SLA.

Measured on Ascend A3, 2026-11-12, batch one, 32-token prompt, seven decode
steps per sequence, one warm-up plus five measured sequences:

| TP | Prefill median / p95 (ms) | Decode median / p95 (ms) | Forward tokens/s | Decode collective mean (ms) |
| -- | ------------------------- | ------------------------ | ---------------- | --------------------------- |
| 2 | 1045.5 / 1317.5 | 1038.1 / 1054.0 | 0.954 | 177.4 |
| 4 | 952.6 / 1248.8 | 943.4 / 959.7 | 1.045 | 210.4 |
| 8 | 946.7 / 1499.0 | 945.5 / 978.6 | 1.034 | 263.0 |

All runs passed rank agreement and token repeatability. TP4 improves median
decode latency by about 9% over TP2; TP8 shows no further benefit in this sample.
Inclusive collective time is 16.9% / 22.0% / 27.2% of summed per-rank decode
wall time. The remainder includes both compute and host dispatch; this
measurement does not separate them. Prefill p95 is based on only five samples
and shows substantial variation. These were sequential shared-host jobs on
cards 1–2, 0–3 and 0–7 respectively, not simultaneous controlled experiments.
Next profile compute versus host dispatch before selecting a fusion or stream
optimization; preserve the synchronization correctness baseline.
Raw per-rank samples: `build/pathb-perf/tp{2,4,8}/timings.json`; logs alongside.
`build/pathb-perf/summary.json` records task IDs, devices, source hashes and
repository revisions (working tree remains uncommitted).

## TP4 dispatch profile

A separate `--profile-dispatch --rounds 2 --steps 3` run uses the same 32-token
prompt, one excluded warm-up, and native TIMING spans. Profile logs go directly
to `native-rankN.log` files. The initial shared-pipe trace overflowed the native
logger queue and is rejected; it remains in `build/pathb-profile/tp4/`.
The valid rerun is `build/pathb-profile/tp4-direct/`. Analyze after releasing cards:

```bash
python -m examples.models.qwen3_tp.profile \
    --timings build/pathb-profile/tp4-direct/timings.json \
    --log build/pathb-profile/tp4-direct \
    --output build/pathb-profile/tp4-direct/breakdown.json
```

The analyzer matches PID and monotonic-time windows, excluding warm-up and
checks. Both native `chip.run` and device-window counts must equal the Python
dispatch count. All measured steps have 723 dispatches per rank, including
80 publications and 80 reductions. Decode averages below cover 16 rank samples
(four ranks × two measured sequences × two decode steps):

| Disjoint duration categories | Mean per rank/step (ms) |
| ---------------------------- | ----------------------- |
| Device execution windows, including device orchestration/scheduling | 211.2 |
| Native `chip.run` outside those windows | 221.3 |
| Dispatch helper outside native `chip.run` | 557.1 |
| Forward outside dispatch helpers | 78.4 |
| Total traced forward | 1068.0 |

The third row includes tensor/Buffer setup, Torch synchronization, config/ABI
binding and cleanup; it is not pure Python interpreter time. Host/device duration
subtraction is an attribution estimate, not a synchronized critical-path trace.
The fourth row includes about 69.8 ms of collective barriers and surrounding
Python. Inclusive collective time is 235.3 ms, overlapping the categories above;
do not add it again. Device-window time is not pure AI Core compute time.
Tracing raised forward latency relative to the 956.6 ms untraced TP4 mean, and
the profile uses fewer decode positions; do not replace the baseline with it.
First investigate repeated dispatch preparation and launch count (723 per step),
then consider compatible kernel fusion. Keep both collective barriers until a
replacement visibility/reuse protocol has its own correctness proof and tests.

## Reusable dispatch-plan experiment

`gloo_shmem_utils.DispatchPlan` retains the callable, tensor-direction metadata,
and default call configuration for one immutable compiled artifact. It holds no
tensor addresses or Buffer registrations. Each invocation still creates fresh
TaskArgs, scalar values and registrations, reads the current stream, synchronizes
Torch, dispatches synchronously, and closes registrations. A plan for a different
compiled artifact is rejected; recreate the plan after changing configuration.
The initial experiment used benchmark-only `--cache-dispatch-plan`; integration
into the model runner is validated below. The default remains uncached.

A fresh sequential TP4 comparison used the same 32-token prompt, five measured
sequences and seven decode steps per sequence, with native tracing disabled:

| Mode | Prefill median (ms) | Decode median / p95 (ms) | Forward tokens/s |
| ---- | ------------------- | ------------------------ | ---------------- |
| Fresh uncached baseline | 945.5 | 951.9 / 1001.0 | 1.041 |
| Reused dispatch plan | 833.8 | 834.3 / 864.5 | 1.185 |

Observed median decode latency fell 12.3%, prefill 11.8%. Both jobs used cards
0–3, exited zero, passed all rank/repeatability checks, and generated identical
tokens across modes. This is one shared-host pair, not a repeated crossover
study; independent HF numerical acceptance was not rerun for this metadata-only
experiment. Focused tests cover changed addresses/scalars, per-call cleanup and
artifact mismatch. Evidence: `build/pathb-cache/{baseline,cached}/timings.json`
and `build/pathb-cache/comparison.json`.
No barrier or launch was removed. After the integration validation below,
measure the remaining TaskArgs/Buffer setup separately before
considering persistent registrations with a larger lifetime-management contract.

## Model-runner dispatch-plan integration

The model runner now accepts `--cache-dispatch-plan`. It owns one plan per
compiled entry and replaces it when that entry's object identity changes.
Compute, publication and reduction dispatch use the same path. The benchmark
now measures this runner implementation rather than a separate cache wrapper.
The switch remains opt-in; omitting it or passing `--no-cache-dispatch-plan`
selects the uncached path.
`--artifacts` can reuse the CPU-prepared manifest, with TP size and kernel-source
hash checked before model/device setup. Lazy native loading stays in startup.
Runner tests cover reuse, changed scalar arguments, replaced programs and the
uncached path. All TP2/TP4/TP8 device runs and independent fused-reference
checks passed, each covering 32 steps and 32 matching greedy tokens. Maximum
relative RMS errors remain 1.9412% / 2.2947% / 1.9481%, respectively, identical
to the prior accepted reports. Evidence: `build/pathb-integrated/tp{2,4,8}/`.
All 96 saved logit vectors are also bitwise equal to the corresponding accepted
uncached TP2/TP4/TP8 runs. The integrated TP4 benchmark passed five measured
sequences: prefill median 867.8 ms; decode median/p95 845.8/861.7 ms; forward
rate 1.161 tokens/s. Against the fresh uncached 951.9 ms decode median, the
observed reduction is 11.1%. This sequential shared-host comparison confirms
that the gain survives integration, without claiming a controlled speedup.
Evidence: `build/pathb-integrated/perf-tp4/{timings.json,run.log}`.

## Cached dispatch setup profile

`--profile-setup --cache-dispatch-plan` adds non-overlapping timers to the shared
helper without native TIMING logging. Timer bookkeeping and helper teardown
outside those intervals remain explicitly unattributed. The same 32-token TP4
workload (one warm-up, five measured sequences, seven decode steps each) passed
all rank-agreement and token-repeatability checks. The analyzer runs off-device:

```bash
python -m examples.models.qwen3_tp.profile --setup-only \
    --timings build/pathb-setup/tp4/timings.json \
    --output build/pathb-setup/tp4/breakdown.json
```

Decode averages across 140 rank samples (723 dispatches per rank/step):

| Disjoint stage | Mean ms per rank/step |
| -------------- | --------------------- |
| Plan validation / TaskArgs initialization | 0.5 / 2.9 |
| Tensor inspection / Buffer registration | 21.6 / 110.3 |
| Tensor binding / scalar binding | 28.8 / 0.9 |
| Torch synchronization / current-stream lookup | 35.1 / 28.1 |
| Inclusive `_run_chip` invocation | 552.8 |
| Buffer release | 9.0 |
| Unattributed dispatch time | 11.7 |
| Forward outside dispatch | 78.2 |
| Total forward | 880.0 |

The invocation row includes Python binding, native runtime and device execution;
it is not the same boundary as the prior native `chip.run` span and is not pure
compute. Inclusive collective time (190.4 ms) overlaps the table. Instrumentation
and different shared-host conditions mean this run is diagnostic, not a new
controlled speedup result. Cards were 1, 3, 4, 5 in one scheduler-approved group.
TaskArgs initialization alone is only 0.3% of forward time; tensor binding adds
3.3%. Registration plus release is about 13.5%. First investigate reducing the
723 launches through compatible adjacent-kernel fusion, which could reduce both
invocation and registration overhead. Persistent Buffer registration is a
separate option requiring explicit allocation ownership, invalidation, failure
retirement and teardown tests; no such cache or barrier removal is introduced.

## SwiGLU + Down launch consolidation experiment

The runner and benchmark accept opt-in `--fuse-swiglu-down` (default off).
One chip entry calls the existing SwiGLU AIV task and then the Down AIC task.
The BF16 intermediate, RINT conversion, collective barriers and dispatch
ownership remain unchanged. The SwiGLU trace snapshot is captured after Down,
which only reads that intermediate. Both outputs remain caller-owned.

Generated orchestration declares `act` as the first task's InOut and the second
task's input, preserving the producer/consumer dependency. CPU lowering checks
pass for TP2/4/8. A TP4 setup-profile run confirms exactly **683 host dispatches
per rank/step**, including 40 fused calls, in all 16 measured rank samples.
The prior path has 723 dispatches: consolidation removes 40 calls (5.53%).

Device validation covers TP2/4/8, each with prompt lengths 1/5/16/32 and eight
steps per prompt. All ranks agree, all generated token IDs match the accepted
unfused runs, and **all 96 saved logits tensors are bitwise identical** to
`build/pathb-integrated/tp{2,4,8}/case_*/outputs.pt`. This preserves the previously
accepted BF16-storage/FP32-arithmetic contract; it is not a new comparison with
unmodified Hugging Face BF16 execution.

TP4 performance used the same allocated cards (1, 4, 5, 6), dispatch-plan caching,
one 32-token prompt, one warm-up sequence and five measured sequences of one
prefill plus seven decode steps. Baseline ran first, fused second. Native tracing
and setup timers were disabled for these two runs.

| Metric | Unfused | Fused |
| ------ | ------- | ----- |
| Prefill median / p95 (ms) | 868.5 / 1225.2 | 839.6 / 1032.3 |
| Decode median / p95 (ms) | 869.6 / 928.5 | 845.1 / 924.9 |
| Decode mean (ms) | 883.2 | 854.3 |
| Reciprocal mean decode rate (tokens/s) | 1.132 | 1.171 |

Observed decode median reduction is **2.82%** (24.5 ms). This is a sequential
shared-host measurement, not an isolated causal speedup guarantee. Collective
time also varied; the full change cannot be attributed solely to launch overhead.
The short instrumented run establishes dispatch counts separately from timing.

Evidence under `build/pathb-fusion/`:

- `tp{2,4,8}/correctness/case_*/outputs.pt` and `comparison.json`.
- `tp4/{baseline,fused}/timings.json`, `performance-comparison.json`,
  `performance.log`, and `count/{timings,breakdown}.json`.
- CPU-prepared artifacts in `tp{2,4,8}/artifacts`; native wrapper compilation
  and loading still occur during first use in the device job.
- The original TP4 job `task_20261113_200904_391595127625` was cancelled after
  the 600-second queue timeout without acquiring cards. Its successful retry
  was `task_20261113_203215_47228218169`, logged in `tp4/correctness-retry.log`.
  Subsequent submissions use `--timeout 0` for queue waiting.

## Full FFN launch consolidation

The runner and benchmark also accept `--fuse-ffn`, mutually exclusive with
`--fuse-swiglu-down`. Both default off. One chip entry schedules the existing
Residual/RMS, Gate/Up, SwiGLU and Down tasks. Their caller-owned intermediates,
BF16 rounding boundaries and both collective barriers are preserved. Diagnostic
snapshots are taken after the FFN entry; downstream tasks only read those buffers.

TP2/4/8 compilation checks validate all producer/consumer tensor dependencies,
RINT casts and high-precision RMSNorm. Device checks cover four prompt lengths
(1/5/16/32) and eight steps each. All 96 logits tensors are bitwise identical to
the accepted unfused outputs, with matching generated tokens and rank agreement.
Evidence: `build/pathb-ffn/tp{2,4,8}/{correctness,comparison.json}`.

TP4 measurement used cards 0–3, dispatch-plan caching, one 32-token prompt,
one warm-up and five measured sequences (prefill plus seven decode steps).
The three uninstrumented runs executed sequentially in the order shown:

| Metric | Unfused | SwiGLU + Down | Full FFN |
| ------ | ------- | ------------- | -------- |
| Prefill median / p95 (ms) | 832.0 / 1133.1 | 827.4 / 993.2 | 765.7 / 1064.8 |
| Decode median / p95 (ms) | 837.8 / 1166.1 | 822.4 / 845.5 | 769.6 / 804.0 |
| Decode mean (ms) | 915.4 | 828.9 | 773.1 |
| Reciprocal mean decode rate (tokens/s) | 1.092 | 1.206 | 1.293 |

Full FFN reduced observed decode median by **8.13%** versus unfused and **6.42%**
versus SwiGLU + Down. Generated tokens agree across all three runs. This remains
a sequential shared-host comparison: baseline tail latency and collective time
varied, so mean-rate changes are not isolated launch-overhead improvements.

A separate setup-profile run confirmed **603 dispatches per rank/step** in all
16 measured rank samples, including 40 full-FFN calls. This removes 120 calls
(16.60%) versus unfused, or 80 versus SwiGLU + Down. Four device tasks remain
inside each FFN entry; no arithmetic kernel or collective was removed.

Performance evidence: `build/pathb-ffn/tp4/perf/{baseline,swiglu_down,ffn}/timings.json`,
`perf/comparison.json`, `perf/count/{timings,breakdown}.json` and
`performance-retry.log`. The earlier job in `performance.log` was deliberately
stopped (exit 130) because it overlapped our TP8 validation; none of its timing
data is used. The successful retry started after TP8 completed and released cards.

Full FFN stays opt-in. Broader workload and repeated/interleaved timing checks
remain necessary before treating these results as a general performance claim.

## Order-balanced full-FFN measurement

TP4 A–B–B–A validation completed on cards 2–5, where A is unfused and B is
full FFN. Each block has one warm-up and five measured sequences, with the same
32-token prompt, seven decode steps and dispatch-plan caching.

| Block | A1 | B1 | B2 | A2 |
| ----- | -- | -- | -- | -- |
| Decode median (ms) | 844.6 | 774.3 | 765.9 | 850.6 |

Paired median reductions are 8.32% (A1/B1) and 9.95% (A2/B2).
Pooling 70 decode observations per variant gives medians of 848.8 versus
768.1 ms, a 9.50% reduction. All four token sequences match. These observations
support a repeatable benefit for this workload; pooled steps are correlated
within sequences and are not independent statistical replicates.

Evidence: `build/pathb-ffn-abba/tp4/{a1,b1,b2,a2}/timings.json`,
`comparison.json` and `run.log`. TP2/8 use separate device allocations, so
cross-TP absolute latencies must not be interpreted as a controlled scaling test.

TP2 and TP8 completed the same A–B–B–A protocol, sequentially with no overlap
between our measurement jobs. TP2 used cards 6/7; TP8 used 0/2/3/4/5/6/7/8.
Each block again has five measured prefill steps and 35 measured decode steps.
All rank-agreement, warm/measured token-repeatability and cross-block token
checks passed. Both jobs exited zero and released their devices.

| TP | A1 decode median | B1 | B2 | A2 | Paired reductions |
| -- | ---------------- | -- | -- | -- | ----------------- |
| 2 | 959.9 ms | 851.8 ms | 876.0 ms | 923.0 ms | 11.27% / 5.10% |
| 8 | 866.1 ms | 780.6 ms | 798.7 ms | 935.2 ms | 9.87% / 14.60% |

Pooling 70 decode steps per variant within each TP configuration:

| TP | Unfused median | Full FFN median | Observed reduction |
| -- | -------------- | --------------- | ------------------ |
| 2 | 941.2 ms | 866.5 ms | 7.94% |
| 4 | 848.8 ms | 768.1 ms | 9.50% |
| 8 | 903.8 ms | 790.0 ms | 12.59% |

All six paired comparisons favor fusion on this workload. The spread between
paired reductions and baseline drift remain visible; these measurements do not
establish a universal speedup or show that TP4 is intrinsically optimal.
Evidence: `build/pathb-ffn-abba/tp{2,8}/{a1,b1,b2,a2}/timings.json`,
per-TP `comparison.json` and `run.log`, plus consolidated `summary.json`.
Full FFN remains opt-in. Further optimization is outside the closeout scope.

## Closeout review and handoff

Scope is frozen at the existing opt-in dispatch cache and FFN launch fusion.
No further profiling, attention fusion, persistent-buffer cache or barrier
removal is planned as part of this closeout. No commit, push or PR was made.

**Review:** no new blocking correctness issue found within the supported
single-machine/direct-L2 workflow. Reviewed the scalar-pool ABI across C++,
bindings and stubs; provider forwarding and fork/L3/workspace guards; borrowed
context generation and close/reinitialize paths; registration ownership and
failure teardown; TP weight slicing; fixed-shape/cache bounds; FFN dependencies
and rounding points; benchmark measurement boundaries and token checks.

The review found a missing failure-path test and typing defects in new tests.
Only tests were changed during closeout: they now check teardown-before-buffer
release, no release after failed teardown, and explicitly narrow dynamic-import,
AST and mock-module types. No production/runtime/kernel behavior changed.

**Validation:**

- Worktree CMake build passed with two workers.
- Related regression suite: 964 passed, 268 warnings.
- Full model entry compilation: 30 passed each for TP2/4/8 (90 total).
- After test typing fixes: 90 affected tests passed, 76 warnings.
- Ruff check/format: passed for all 36 changed Python/stub files.
- Pyright with `--pythonpath .venv/bin/python`: zero errors/warnings.
- Documentation: 213 EN/ZH pairs and 213 navigation entries passed.
- `git diff --check`: passed. Deprecation warnings remain in test logs.

**Broader performance coverage:** TP4, cache enabled, A–B–B–A order, one
warm-up and three measured sequences per block. The one-token “Hello” prompt
with seven decode steps passed all rank/token checks. Decode block medians
A1/B1/B2/A2 were 845.7/740.7/782.8/850.8 ms; pooled medians were
850.4 → 756.2 ms (11.08% lower). The 32-token prompt with 31 decode steps
completed as `task_20261113_225819_369013932697` (exit 0, cards 2–5 released).
Its A1/B1/B2/A2 decode medians were 830.5/767.0/774.3/855.1 ms; both pairs
favored fusion (7.65% and 9.45%). Pooling 186 decode samples per variant gives
842.5 → 771.0 ms (8.48% lower). All four generated token sequences match.
All three closeout items are **complete**: broader performance coverage,
final review/regression, and the updated review package. No further optimization
is planned. Shared-host timing variability still limits generalization.
Evidence lives in `build/pathb-closeout/{short,longdecode}/`.

These extra benchmarks check finite logits, rank agreement and token
repeatability; they do not add a fresh independent CPU reference at every longer
decode position. The prior TP2/4/8 96-logit bitwise fusion equivalence and accepted
BF16-storage/FP32-arithmetic reference remain the numerical acceptance evidence.

**Review package:** `build/pathb-closeout/pathb-review.tar.gz` contains a complete
patch (including untracked new files), source snapshots, file hashes/status,
base/runtime revisions, reproduction commands, regression logs and performance
JSON evidence. It excludes weights, compiled binaries and large native traces.
The package manifest records the completed closeout checks. Apply the patch only to
a clean checkout at its recorded base, with the recorded runtime revision;
dependencies and CPU-prepared device artifacts must be supplied/rebuilt locally.

Supported boundaries remain Qwen3-14B, batch 1, TP2/4/8, prefill 1..32, cache 128,
one process per rank, Gloo control plus SHMEM and direct simpler L2. Unmodified
HF eager-BF16 equivalence, L3 SHMEM mappings, workspace-dependent communication
engines, serving throughput/SLA and cross-machine operation are not claimed.
The source's original kernel attribution/license is retained; no vLLM runtime
integration or simpler submodule modification is included.

## PR preparation on the updated main branch

The patch was reapplied without textual conflicts to PyPTO `7d2c5bf3889c`
on 2026-11-16. This base pins simpler `c1c234667d54` and requires PTOAS v0.74.
The original hardware results above were collected on PyPTO `88efefc33`
with simpler `355ce3015e3e`; they remain historical evidence, not a hardware
acceptance result for the updated dependency combination.

The updated base builds successfully. All six C++ test targets, 964 affected
regression tests in the freshly built native-runtime environment, and 11 SHMEM
ABI/window-layout checks pass. The TP4 and TP8 compiler-entry checks pass
(30 entries each). A fresh two-rank hardware smoke test passes TSTORE, TPUT,
and three changing-input allreduce iterations on both ranks, using CPU-prepared
kernels and queue-assigned devices.

These checks establish integration and two-rank communication coverage; the
full-model TP2/4/8 numerical and performance results above remain historical.
The feature scope remains single-machine direct L2 execution and the accepted
BF16-storage/FP32-arithmetic contract. No new performance work is included.

The BF16 midpoint system test uses the standard guarded compiled-call path
and the fixture-assigned device. Its 32-value rounding assertions pass on a
queue-assigned card; all nine card-lock regression checks also pass.

## Updated-base model acceptance

On 2026-11-16, full-model numerical acceptance passed on PyPTO `7d2c5bf3889c`,
simpler `c1c234667d54`, and PTOAS v0.74 with real Qwen3-14B weights. Each TP size
ran prompts of 1/5/16/32 tokens for eight steps each (one prefill and seven decode
steps). The independent CPU reference enforces the accepted BF16-storage/FP32-
arithmetic contract. Thresholds remain relative RMS error below 3%, cosine
above 0.999, and exact greedy-token agreement.

- TP2: 32/32 matching tokens; maximum relative RMS error 1.9412%; minimum cosine 0.999860644.
- TP4: 32/32 matching tokens; maximum relative RMS error 2.2947%; minimum cosine 0.999801934.
- TP8: 32/32 matching tokens; maximum relative RMS error 1.9481%; minimum cosine 0.999809921.

All six device runs passed rank agreement and released their cards. The default
unfused/cache-disabled configuration and the FFN-fused/cache-enabled
configuration produced 96 bitwise-identical logit pairs and matching tokens.
All 90 native kernel artifacts were prepared before device allocation.
No production code or acceptance threshold was changed for this validation.

Evidence: `build/pathb-acceptance-pr/summary.json`, `hardware-summary.json`,
`fusion-results.json`, and `tp{2,4,8}/reference.json`, plus per-run commands, logs,
and saved output tensors. This is numerical acceptance under the stated storage
contract, not equivalence to unmodified HF eager BF16. Performance was not
remeasured; earlier performance numbers remain historical.
