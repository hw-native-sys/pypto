# HBG eager Host measurement

`tests/benchmarks/hbg_csa_stability.py` measures HC pre → norm → CSA → HC post
with real single-layer weights and synthetic history. It requires an external
vllm-ascend CSA fixture, matching built native extensions, DeepSeek weights and
an NPU. These assets are not distributed with PyPTO; this is not a self-contained
benchmark that runs in a clean checkout.

## Prerequisites

Build PyPTO and its optional Torch NPU adapter in the checkout, with compatible
torch_npu/CANN. The external source must contain `tests/pypto_test/dsv4_csa_single_layer.py`,
`dsv4_csa_env.py` and their dependencies, plus the
`vllm_ascend.ops.pypto.deepseek_v4_flash_csa` package. `--native` names a directory
containing `src/vllm` and the built `src/vllm-ascend/vllm_ascend/vllm_ascend_C`
extension; `--checkpoint` names the weight directory.

Set `PYTHONNOUSERSITE=1`, `PTOAS_ROOT`, `PTO_ISA_ROOT`, `ASCEND_HOME_PATH` and verify
all import paths. Use the server scheduler and its current physical-card access
limits; `--device` is a logical index among visible devices. Use a fresh process
for each queue setting and repeat. Disable per-call TIMING/STRACE, preload probes
and graphs. Compile and bootstrap Identity during warmup; measure cold start separately.

## Running and scope

Supply the variables below; the output directory must not exist.

```bash
TASK_QUEUE_ENABLE=0 python tests/benchmarks/hbg_csa_stability.py \
  --source "$CSA_SOURCE" --native "$CSA_NATIVE" --checkpoint "$CSA_WEIGHTS" \
  --output "$RESULT" --device 0 --history 8192 --mode paired \
  --blocks 6 --samples 32 --warmup 5 --cpu-samples 8 --seed 1
```

`paired` alternates Native/PTO in one process, reversing order each block. Both
produce compact metadata inside timing; PTO adapter construction is included.
Reset, output poisoning and device synchronization are outside Host timing.
Consecutive calls within an arm retain previous leases; drain at arm boundaries.
Each implementation is checked against its own reference, not against the other
implementation's known zero-tolerance differences. Compilation is forbidden
during measurement of the fixed artifact.

`paired-diagnostic` separately records Native Torch dispatcher operator counts
and PyPTO spans. Instrumented timings are not performance evidence. Dispatcher
names do not establish which ACLNN API is used; inspect the matching extension
or a profiler trace before attributing costs to ACLNN.

The historical three-version `bench --suite current` runs one version per
process, with A/A controls. PTO reuses metadata while Native produces it, and
Native is measured first once. Its Native/PTO ratio is not an equivalent-work
performance acceptance result.

## Pinned versions and aggregation

`hbg_csa_version.py` requires a scikit-build editable environment. It remaps a
complete Python Git worktree before imports, retaining common built native
modules. Check native/runtime compatibility first; keep real Git metadata and
runtime pin verification. Do not copy shared libraries or substitute individual
functions. The manifest records source hashes. Run preflight first:

```bash
CSA_VERSION_ROOT="$VERSION_WORKTREE" CSA_VERSION_NAME=before \
  python tests/benchmarks/hbg_csa_version.py --output "$RESULT" --preflight
```

Replace `--preflight` with benchmark arguments for measurement. Historical labels
are `before=d5ee5fd`, `pre-unify=12460f9`, `current=6850dc0`. The baseline need not
contain the harness: execute the measurement checkout's harness against each
complete version worktree.

Evidence directories use names such as `h8192-q0-r1-before`, containing reports,
identity, manifests, Host/CPU samples, state checks and adjacent `.log` files.
Analyze without an NPU:

```bash
python tests/benchmarks/hbg_csa_aggregate.py --repo "$REPO" \
  --evidence "$EVIDENCE" --device "$RECORDED_PHYSICAL_DEVICE" \
  --repeats 5 --output "$SUMMARY"
```

Aggregation validates version hashes, runtime/native identity, state checks,
384 Host samples and lifecycle samples before pairing process statistics.
Take each stage median, average A/A, then take the block median. Noise is the
maximum A/A difference or process range. Use 50,000 process-level bootstrap
samples; require positive savings in every repeat and a CI lower bound above
noise. Report different cards separately.

The historical 136→11 ms result is an author-environment observation. External
fixtures and collected evidence are not published with this repository, so the
source alone does not independently verify that number. It is not acceptance
against latest main or ACLNN.
