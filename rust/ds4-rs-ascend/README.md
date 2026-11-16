# ds4-rs-ascend — serve DeepSeek V4.1 Flash on Ascend

A serving utility: memory profile, parallel split, kernel-set registry.

![ds4-rs-ascend demo](../demo/media/ds4-rs-ascend.gif)

*[Narrated MP4](../demo/media/ds4-rs-ascend.mp4) · 4:51 · [cue sheet](../demo/narration/ds4-rs-ascend.txt)*

## Build

```sh
cd ds4-rs-ascend && cargo build --release
```

A **separate Cargo workspace** from `rust/`, deliberately: those four tools have
no third-party dependencies at all and that is worth keeping checkable, while a
serving utility needs a SQLite registry, a signature verifier and a memory map.
Stable Rust, like the rest.

## Usage

```sh
ds4-rs-ascend mem                       # per-chip free HBM
ds4-rs-ascend plan   --mode latency     # the split it would choose, without serving
ds4-rs-ascend deploy --mode balanced    # plan, then serve
ds4-rs-ascend kernels list | verify     # the local kernel-set registry
ds4-rs-ascend profile --model ds4       # tune launch and tiling knobs on this box
ds4-rs-ascend help --advanced           # the expert knobs
```

`--mode` is the deployment **intent**, and the split follows from it rather than
being picked: `latency` for one user, `throughput` for aggregate tokens/sec,
`balanced` for a middle point, `max-context` for the longest context the box can
hold. TP, DP, EP and PP are not a free choice — two do nothing for a single
stream and two cost memory linearly, so the configuration follows from the
objective.

## What needs a card

**`mem` and `plan` need an Ascend driver on PATH.** The crate's own manifest
comment says the planning half works without the `ascend` cargo feature, and it
does — but the feature was never the only dependency: the memory profile shells
out to `npu-smi`, and `--mem-source` offers `npu-smi` or `acl` and nothing
synthetic. A capacity question still needs a box.

Works off a card: `version`, `help`, `help --advanced`, `kernels list`,
`kernels verify`.

## Two claims this tool does not make

- **The parallel split is a model.** Its own module header says nothing in it has
  run on multiple chips; the collective cost is an assumption, and so is the
  claim that *n* host links deliver *n* times one link's bandwidth. That is the
  first thing to measure.
- **A model name is not yet a choice.** `--model` is accepted and currently
  resolves to DeepSeek-V4-Flash whatever is passed — `--model banana` prints a
  complete plan. GLM-5.3 exists only as examples under `engine_core`
  (`glm53_*.rs`) and Kimi-K3 has no code path at all. Until the names are wired,
  a run is a DS4 run and must be labelled as one.

## What it can do today

`deploy` serves. A truncated model — four blocks of the merged MXFP4 set, 15.4 GB
— loads and decodes on eight 910B2 chips:

```
engine_ascend: routed-expert tables registered for 4 layers (Some(MXFP4))
decode: tokens=8 elapsed=70.385s ms_per_step=8798.1 tok_per_s=0.114
```

That is a **readiness** number, not a decode benchmark: four of forty-four blocks
cannot answer a prompt, and the figure reflects a contended board. Set
`DS4_TIMING=1` for per-token lines and `--max N` to bound the run; without either,
a slow run reports one word and no rate.

Three limits are worth knowing before planning a deployment:

- **The MoE expert tensors must be `Q2_K`, `Q4_K` or `MXFP4`.** `Q4_0` and `Q8_0`
  are rejected by role, so a quant outside that set fails at load.
- **Split GGUFs must be merged first.** The loader opens one file and resolves
  every tensor in it; a `-of-NNNNN` set fails on the first global tensor that
  lives in a later shard.
- **The whole model must fit in free HBM.** When it does not, the engine falls
  back to `DS4_SSD_STREAM` and streams weights per token, which on a 155 GB model
  behind 2.1 GB/s of disk does not reach a usable rate.

## Kernel sets are signature-verified

A kernel-set index is verified with Ed25519 before anything in it runs. The check
answers one question — *is this what was published* — and a test keeps it
present.

One debt this records rather than hides: dropping the FFI bridge to a vendored
third-party engine also removed a **correctness oracle**, and the Ascend side has
no equivalent yet. A CPU-reference comparison is owed.
