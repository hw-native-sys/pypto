# Narrowing Down a Precision Gap

The result does not match the golden. This page is the order in which to suspect things —
not an API reference.

> **Prerequisites:** [Your first operator](../tutorials/00-elementwise.md) for the
> `allclose` comparison this page assumes you already have.

## The order

```text
Result does not match the golden
├─ 1. Is the golden itself right?         → write_golden; are rtol/atol sane?
├─ 2. Should there be a difference?       → the acceptable-difference table below
├─ 3. Did the compiler already warn?      → verification level, diagnostics
├─ 4. Which pass introduced it?           → torch codegen + validate_ir, bisect
└─ 5. Which tensor is wrong?              → dump_tag / dumps= + enable_dump_args
```

Steps 1 and 2 cost minutes and remove most reports. Steps 4 and 5 cost hours. Do not start
at 4.

## The tools

| Step | Layer | Entry point |
| ---- | ----- | ----------- |
| 1 | End to end | `pypto.runtime.write_golden` + `RunConfig(rtol=, atol=, golden_data_dir=)` |
| 3 | IR legality | `ir.compile(verification_level=...)` / `PYPTO_VERIFY_LEVEL` |
| 3 | Compile-time warnings | `diagnostic_phase` / `disabled_diagnostics` |
| 4 | IR semantics | `pypto.debug.torch_codegen` |
| 4 | Per-pass check | `CompiledProgram.validate_ir` |
| 4 | IR structure | `ir.compile(dump_passes=PassDumpLevel.EXPLICIT)` |
| 5 | Runtime data | `pl.dump_tag(t)` / `dumps=[t]` + `RunConfig(enable_dump_args=1\|2)` |

## Step 1: is the golden right?

The default tolerance is `rtol=1e-5`, which is **wrong for FP16 inputs** — those carry about
three decimal digits, so a correct FP16 matmul fails against an FP32 reference at `1e-5`.
Before investigating the kernel, check that the tolerance matches the *input* precision.

`write_golden` records a reference so later runs compare against a fixed artefact rather
than a recomputed one. That matters when the reference itself is nondeterministic.

## Step 2: should there be a difference?

Some differences are the correct behaviour of a correct compiler. Rule these out before
bisecting anything.

| Source | Difference | Notes |
| ------ | ---------- | ----- |
| Split-K / atomic add | Last bits, run to run | Accumulation order across cores is not fixed |
| FP16 / BF16 accumulation | Grows with reduction length | Accumulate in FP32 where you can |
| Reduction shape | Binary-tree vs sequential | Whether `col_sum` gets a `tmp_tile` changes the order |
| Backend differences | Instruction-level | The same op need not be bit-identical across backends |
| Multi-hop cast | **Usually none** — see below | `LegalizeTileCast` expands what the ISA cannot do in one step |

**The multi-hop cast is worth stating precisely, because it is easy to blame.** On A5,
`INT32→FP16` is expanded to `INT32→FP32→FP16`. That chain is **bit-identical** to the
reference it is standing in for — a hypothetical single-step `INT32→FP16` under the same
rounding mode and the same overflow behaviour. The argument has two halves:

- **`|x| ≤ 65504`** (largest finite FP16): such an `x` is well under `2^24`, so it is exact
  in FP32. The FP32 hop does not round, and the single rounding that occurs is the final
  hop — the same rounding the one-step reference would perform.
- **`65504 < |x| < 65520`**: these do *not* overflow. Under round-to-nearest they land on
  the largest finite FP16, `65504`, because `65520` is the midpoint between it and the next
  value the format would have had. Both forms round the same way.
- **`|x| ≥ 65520`**: both the chain and the one-step reference overflow to infinity. The
  FP32 hop *does* round for `|x| > 2^24`, but only among values already far outside FP16's
  range, so that rounding cannot change the outcome.

The middle case is the one worth remembering: FP16 "overflow" starts at `65520`, not at
`65504`.

A chain introduces a real difference only when an intermediate type cannot exactly represent
source values that *are* in the destination's range. Check
[LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) for the class your chain falls
into rather than assuming the hop is the culprit.

The claim is checkable, so this page checks it. The block below casts `INT32 → FP16`
through PyPTO and compares against torch's conversion of the same values, across the three
ranges the argument turns on — exactly representable, past FP16's limit, and past `2^24`
where the FP32 hop itself rounds:

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
```

<!-- doctest: run -->
```python
ROWS, COLS = 16, 128


@pl.jit
def to_fp16(x: pl.Tensor[[ROWS, COLS], pl.INT32], out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP16]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.cast(x, pl.FP16)
    return out


values = torch.zeros(ROWS, COLS, dtype=torch.int32)
values[0, :4] = torch.tensor([0, 1, 2048, 65504])            # exact in FP16
values[0, 4:8] = torch.tensor([65505, 65519, 65520, 70000])  # the boundary: 65519 -> 65504, 65520 -> inf
values[0, 8:12] = torch.tensor([1 << 20, 1 << 24, (1 << 24) + 1, 1 << 25])   # past 2^24
values[0, 12:16] = torch.tensor([-65504, -65519, -65520, -70000])            # the same boundary, negative

out = torch.zeros(ROWS, COLS, dtype=torch.float16)
to_fp16(values, out, config=CFG)

# Bit-for-bit against torch's conversion of the same integers.
expected = values.to(torch.float16)
assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), (
    f"differs at {(out != expected).nonzero()[:4].tolist()}"
)
```

That runs the backend this page's CI target uses, so it establishes the user-visible half of
the claim — the numbers you get match the reference. The A5-specific *expansion* is argued
above rather than executed here, since the chain only appears on that backend.

## Step 3: did the compiler already say something?

Raise the verification level and re-compile before doing any bisection — a malformed-IR
report names the pass for you:

```python
prog = ir.compile(program, verification_level=...)   # or PYPTO_VERIFY_LEVEL
```

## Step 4: which pass introduced it?

This is the expensive step, and the one worth doing properly.

`pypto.debug.torch_codegen` turns an IR `Program` or `Function` into executable torch, so
the IR's *semantics* can be run on the host and compared against your reference — no device
involved:

<!-- doctest: run -->
```python
from pypto.debug import torch_codegen


@pl.jit
def fused(a: pl.Tensor[[64, 128], pl.FP32], b: pl.Tensor[[64, 128], pl.FP32],
          out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.mul(pl.add(a, b), 2.0)
    return out


a = torch.randn(64, 128)
b = torch.randn(64, 128)

src = torch_codegen(fused.lower(a, b, torch.zeros(64, 128)))  # check_shapes=True to assert shapes
assert "def " in src and "torch" in src                       # it is executable python
```

The point is not the string: it is that the IR's meaning can be executed on the host and
compared against your reference, isolating "the IR is wrong" from "the device disagrees
with the IR".

`CompiledProgram.validate_ir` runs that comparison per pass. The bisection is then
mechanical: the first pass whose IR stops matching is the one that introduced the
difference. Dump the IR either side of it with
`ir.compile(dump_passes=PassDumpLevel.EXPLICIT)` and read the two.

> **Pass it your tolerance.** `validate_ir` defaults to `rtol=5e-2, atol=5e-2` — its own
> defaults, not the ones you settled on in step 1, and far looser than most references
> deserve. Left alone it reports a pass as matching whenever the regression is smaller than
> that, and the bisection then hands you a boundary that is not the real one:
>
> ```python
> compiled.validate_ir(..., rtol=RTOL, atol=ATOL)   # the tolerance from step 1
> ```

This locates a *semantic* change. It cannot see a difference that only appears on device —
for that, step 5.

## Step 5: which tensor is wrong?

When the IR is right at every pass but the device result is not, compare actual data:

```python
pl.dump_tag(t)       # mark the tensors you care about
cfg = RunConfig(platform="a2a3sim", enable_dump_args=1)
```

Level `1` dumps only tagged tensors; level `2` dumps every task's inputs and outputs. Read
them with `python -m simpler_setup.tools.dump_viewer`.

> **Fatal pitfall:** a full dump (`enable_dump_args=2`) on a large workload can saturate the
> host-side collector (~42 MB/s drain) and get the AICPU killed by a STARS op-execute
> timeout. Prefer level `1` plus `pl.dump_tag` on the specific tensors you are chasing.

### When the result changes between runs

Before dumping anything, re-run the same input a few times. A value that moves means
something is unordered — but two very different things can be, and they are cheap to tell
apart:

- **Last bits only, in a kernel that uses split-K or an atomic add.** That is accumulation
  order across cores. It is expected, step 2 above already covers it, and there is nothing
  to fix.
- **Anything larger** — whole regions wrong, values off by a lot, or a result that is
  sometimes right and sometimes not. That is a task-ordering bug, and no pass dump will
  explain it: the IR is entitled to look correct at every pass, because statement order does
  not constrain execution order.

For the second, the runtime infers RAW and WAW from buffer overlap, but **WAR is not
tracked**: a writer overwriting a buffer some other task may still be reading takes no
edge, because finding every in-flight reader would be a per-write walk on the hot path.
That anti-dependency is yours to declare. See [Dependencies](../performance/03-dependencies.md) for the full rule
and its cost.

```python
cfg = RunConfig(platform="a2a3sim", enable_dep_gen=True)   # writes deps.json
```

Read the graph and look for the edge you assumed was there. The fix is to name it —
`deps=[reader_tid]` on the writer, leaving the readers as plain inputs (`INPUT`, which is
what an unannotated parameter already is).

> **Do not fix it by promoting the reader to `pl.InOut`.** It does create the edge, since an
> `INOUT` registers as a writer and the overwrite then takes a WAW edge on it. It also
> serializes every *other* reader of that buffer against each other, because each one
> becomes the registered producer in turn. A tensor read concurrently by several tasks
> loses that concurrency entirely, to buy one anti-dependency.

Note that dumping perturbs this: `enable_dump_args` adds GM traffic and changes timing, so a
race can hide or move when you turn it on. Settle the ordering question from the graph
first, then go back to comparing values.

### When the value you want is not a tensor

Dumping only reaches things that *are* tensors. The value that would settle the question is
often an intermediate inside an InCore function — an accumulator before the final store, a
tile after one step of a fused chain — which never reaches GM and so cannot be tagged.

Give it somewhere to go: add a temporary `pl.Out` parameter to the kernel, store the
intermediate into it, and thread it out through the orchestration.

```python
@pl.jit.incore
def fused(x: pl.Tensor, out: pl.Out[pl.Tensor],
          probe: pl.Out[pl.Tensor]):        # temporary, for this investigation only
    acc = pl.add(pl.load(x, [0, 0], [64, 128]), 1.0)
    probe = pl.store(acc, [0, 0], probe)    # the intermediate, now inspectable
    out = pl.store(pl.exp(acc), [0, 0], out)
    return out, probe
```

It is a debugging edit, not a design: the extra parameter costs a GM round trip and changes
the dependency graph, so take it out once the question is answered. What it buys is a direct
comparison of the intermediate against your host reference — which usually converts "the
output is wrong" into "this step is wrong" in one run.

## Edge Cases

| Symptom | Likely cause | Step |
| ------- | ------------ | ---- |
| **Correct kernel fails `allclose`** | `rtol=1e-5` against FP16 inputs | 1 |
| **Differs run to run, same input** | Split-K atomic accumulation order, **or a missing WAR edge** | 2, then 5 |
| **Differs only for long reductions** | FP16/BF16 accumulator | 2 |
| **Blamed on a multi-hop cast** | Often bit-identical — check the class | 2 |
| **IR matches every pass, device does not** | Not a semantic bug | 5 |
| **Row maxima all `0.0`** | Padding participated in a reduction | See [Reduction and softmax](../tutorials/01-reduction-softmax.md) |

## See Also

- [Worked cases](01-cases.md) — this order applied end to end.
- [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) — when a cast chain is exact.
- [Reduction and softmax](../tutorials/01-reduction-softmax.md) — padding and reductions.
