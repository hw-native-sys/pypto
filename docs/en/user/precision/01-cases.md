# Worked Cases

Five gaps, narrowed. Two of them turn out not to be bugs — which is the point of the
ordering.

> **Prerequisites:** [Narrowing down a gap](00-workflow.md).

## Case 1: bisecting to the pass

**Symptom.** A multi-stage kernel's output is wrong. Nothing errors.

**Step 1–3.** Tolerance matches the input precision. No entry in the acceptable-difference
table applies. Raising the verification level reports nothing.

**Step 4.** `pypto.debug.torch_codegen` turns the IR into executable torch, so the IR's
meaning runs on the host with no device involved. `CompiledProgram.validate_ir` does that
per pass, and the bisection is mechanical: the **first pass whose IR stops matching** is the
one that introduced the difference.

**Then read the two IRs.** `ir.compile(dump_passes=PassDumpLevel.EXPLICIT)` writes the dump
either side of that pass; the diff is the change in meaning.

**What this proves.** A semantic bug — the compiler changed what the program computes. If
instead every pass matches, this case is the wrong one and you want case 5.

## Case 2: the multi-hop cast that is not the culprit

**Symptom.** An A5 kernel with an `INT32→FP16` cast disagrees with the reference, and the
pass dump shows the cast expanded to `INT32→FP32→FP16`. The expansion is the obvious
suspect.

**Step 2 disqualifies it.** That chain is **bit-identical** to a direct conversion: FP16
saturates above 65504, every integer below that is exact in FP32, so the FP32 hop never
rounds and only the final hop does — exactly as a direct `INT32→FP16` would.
`LegalizeTileCast` expands what the ISA cannot emit in one instruction; expansion is not
approximation.

**The judgement to carry forward.** A chain differs only when an intermediate cannot
represent the in-range source values exactly. Check that property for *your* chain rather
than treating "it got expanded" as evidence. See
[LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md).

**Where to look instead.** Back to step 2 for another entry, or on to step 4.

## Case 3: padding contaminated a reduction

**Symptom.** Row maxima come back `0.0` for data that is entirely negative. The kernel is
correct on shapes that divide evenly into tiles.

**Located by.** Step 2, then reading the kernel: a `pl.load(..., valid_shape=[64, vlen])`
feeding `row_max`. The padding participates in the reduction, and zero beats every negative
value.

**Change.** `pl.fillpad(..., pl.PadValue.min)` before the reduction — pad with the identity
of the operation you are about to apply. `PadValue.zero` for `row_sum`, `max` for a min
reduction.

**Confirmed by.** The partial-tile case now matches. Note the full-tile case never failed,
which is why this survives casual testing — see
[Reduction and softmax](../tutorials/01-reduction-softmax.md).

## Case 4: split-K, working as designed

**Symptom.** The same input gives results differing in the last bits between runs.

**Step 2 disqualifies it.** Split-K accumulates partial products into one output with an
atomic add, and **the order across cores is not fixed**. Floating-point addition is not
associative, so the last bits may move.

**The decision, not the fix.** Nothing is broken, so there is nothing to repair — there is a
choice:

| Want | Do |
| ---- | -- |
| Bitwise reproducibility | K-blocking on one core — [Tiled matmul](../tutorials/02-matmul.md) |
| The parallelism | Keep split-K; set a tolerance that reflects it |

**Confirmed by.** Running twice. If the difference is bounded by the tolerance and does not
grow with input size, it is accumulation order, not a bug.

## Case 5: right IR, wrong data

**Symptom.** Step 4 shows the IR matching at **every** pass, and the device result is still
wrong.

**What that rules out.** Everything semantic. No amount of reading pass dumps will find it,
because the passes are not the problem.

**Step 5.** Mark the suspect tensors and compare actual values:

```python
pl.dump_tag(t)
cfg = RunConfig(platform="a2a3sim", enable_dump_args=1)
```

Read them with `python -m simpler_setup.tools.dump_viewer`, and walk forward from the inputs
until a tensor first disagrees with the host reference.

> **Fatal pitfall:** `enable_dump_args=2` dumps every task's inputs and outputs. On a large
> workload that saturates the host-side collector (~42 MB/s drain) and gets the AICPU killed
> by a STARS op-execute timeout. Use level `1` plus `pl.dump_tag` on the tensors you are
> chasing.

## What the five cases have in common

| Case | Resolved at | Was it a bug? |
| ---- | ----------- | ------------- |
| 1 | Step 4 | Yes — a pass |
| 2 | Step 2 | **No** — expansion is exact |
| 3 | Step 2 → kernel | Yes — in the kernel |
| 4 | Step 2 | **No** — by design |
| 5 | Step 5 | Yes — not semantic |

**Three of five were settled at step 2**, which costs minutes. Two of those were not bugs at
all. Starting at step 4 would have spent hours on cases 2 and 4 and found nothing, because
there was nothing to find.

## See Also

- [Narrowing down a gap](00-workflow.md) — the procedure.
- [Performance](../performance/index.md) — the same treatment for slow.
- [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) — cast chain exactness.
