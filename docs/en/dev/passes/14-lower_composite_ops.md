# LowerCompositeOps Pass

Decomposes composite tile ops into compositions of primitive arithmetic tile ops, so codegen never has to emit a high-level intrinsic. Today only `tile.sin` / `tile.cos` are handled; new composite ops add a lowering rule to the dispatch table inside the pass file without touching the dispatcher.

## Overview

`LowerCompositeOps` is a function-level pass that rewrites every `var = Call(...)` `AssignStmt` whose callee appears in the pass's lowering dispatch table. For `tile.sin` / `tile.cos`, the rule emits a fixed-shape primitive recipe: Cody-Waite range reduction (4-part π split) followed by a degree-9 odd Horner polynomial. The original target `Var` is preserved as the LHS of the final `AssignStmt`, so downstream uses keep the same name and identity.

The pass is **FP32-only**. Non-FP32 inputs are rejected at op-construction time by the shared `DeduceTileFP32OnlyType` deducer (see `src/ir/op/tile_ops/unary.cpp:94`), so the lowering pass itself only sees well-typed FP32 operands and never has to fail on dtype.

The pass is **structural no-op** on programs that contain no `tile.sin` / `tile.cos`: every other statement passes through `IRMutator::VisitStmt_`. The decomposition only emits primitive tile ops (`tile.muls`, `tile.adds`, `tile.add`, `tile.sub`, `tile.mul`, `tile.cast`), none of which the mutator rewrites — so the pass is also **idempotent**.

**Requires**: nothing.

**Produces**: nothing.

**Invalidates**: nothing.

The empty `PassProperties` contract (`kLowerCompositeOpsProperties` in `include/pypto/ir/transforms/pass_properties.h`) reflects that the lowering operates purely within the existing tile-op vocabulary — it neither establishes nor breaks any `IRProperty`.

## When It Runs

`LowerCompositeOps` is the **first entry of `tile_pto_passes`** in the `Default` pipeline (see `python/pypto/ir/pass_manager.py`), running immediately after `ConvertTensorToTileOps` (slot 12) and `OptimizeOrchTensors` (slot 13). At this point all tensor-level transcendental calls (`tensor.sin`, `tensor.cos`) have been rewritten to their tile equivalents (`tile.sin`, `tile.cos`) by the conversion registry, and the tile pipeline is about to start tile-shape canonicalisation. Lowering trig before `FlattenTileNdTo2D` keeps the decomposition independent of the 2D-flattening rules — every primitive op in the recipe has well-defined behaviour at any rank.

## Architecture

The pass is a single translation unit, `src/ir/transforms/lower_composite_ops_pass.cpp`:

```text
src/ir/transforms/lower_composite_ops_pass.cpp
  LoweringBuilder           — per-call scratchpad (Bind + primitive op builders)
  CompositeLoweringFn       — (args, span, builder) -> result expr
  Lower<Op>Rule             — one rule function per composite op (e.g. LowerSinRule, LowerCosRule)
  LookupCompositeRule       — file-local op-name → rule dispatch table (kRules)
  LowerCompositeOpsMutator  — walks the function, looks up a rule per Call
```

Adding a new composite op (all edits stay in `lower_composite_ops_pass.cpp`):

1. Write a `Lower<Op>Rule(args, span, builder)` function. It receives the visited arg expressions, a `Span`, and a `LoweringBuilder` whose `Bind` helper appends an `AssignStmt` for each intermediate temp.
2. Add a `{"<op>", &Lower<Op>Rule}` row to `kRules` inside `LookupCompositeRule`.

No edits to the mutator are needed. When the table grows past a handful of entries — or a rule wants its own translation unit — promote it back to a standalone registry under `src/ir/transforms/composite_ops/`.

## Algorithm (sin / cos rule)

`LowerSinCos` in `src/ir/transforms/lower_composite_ops_pass.cpp` is parameterised on `is_cos`. The mutator overrides `VisitStmt_(const AssignStmtPtr&)` (rather than `VisitCall`) because each trig op expands to ~33 statements and each statement needs a fresh temp `Var`. Working at the statement level lets the rule append directly to the surrounding sequence via the builder.

### Range Reduction (Cody-Waite, 4-part π split)

The goal is to express `x = k·π + t` (sin) or `x = k·π + π/2 + t` (cos) with `t ∈ [-π/2, π/2]` and `k` an integer. FP32 cannot represent π exactly, so a single `x - k·π_fp32` carries a relative error of ~1e-7 per multiplication, which range-reduction error inflates linearly with `|k|`. Cody-Waite splits π into a fast-rounding head plus three (here four) small corrections so the residual cancellation only loses bits at the finest scales:

```text
π ≈ PI_V2 + PI_C1 + PI_C2 + PI_C3 + PI_C4
```

`t` is computed as a chain of subtractions, each consuming one part:

```text
t0 = x  - k_f * PI_V2
t1 = t0 - k_f * PI_C1
t2 = t1 - k_f * PI_C2
t3 = t2 - k_f * PI_C3
t4 = t3 - k_f * PI_C4
```

For **sin**, `k_f = float(round(x · PI_INV))` using `tile.cast` mode `ROUND` (round-to-nearest, ties away from zero). For **cos**, the rounding is shifted by `0.5` so `k` represents the multiple of `π` whose midpoint lies near `x`:

```text
k_f = float(rint(x · PI_INV + 0.5))   ; mode RINT (round-half-to-even)
```

The cos path also adds `π/2` mid-reduction, split as `PI_HALF_HEAD + PI_HALF_TAIL` (Cody-Waite again). `PI_HALF_HEAD` is folded between `PI_C1` and `PI_C2`, `PI_HALF_TAIL` after `PI_C4`, so that each addition shares the magnitude scale of the surrounding subtractions and the catastrophic-cancellation regime is shared across all 5+2 corrections.

### Sign Computation

Once `k` is known as a float, the sign is computed without any conditional:

```text
sign = floor(k_f / 2) · 4 + k_f · (-2) + 1
     = (-1)^k
```

The identity `floor(k/2)·4 - 2·k + 1` evaluates to `+1` for even `k` and `-1` for odd `k`. To see this, write `k = 2m + r` with `r ∈ {0, 1}`:

```text
floor(k/2) = m
floor(k/2)·4 - 2·k + 1 = 4m - 2(2m + r) + 1 = 1 - 2r
```

which is `+1` when `r = 0` and `-1` when `r = 1`. The pass implements this in 6 ops:

```text
half_k     = k_f * 0.5
floor_hk_i = int32(floor(half_k))         ; tile.cast mode FLOOR
floor_hk_f = float(floor_hk_i)
floor_x4   = floor_hk_f * 4.0
neg2_k     = k_f * (-2.0)
sign_pre   = floor_x4 + neg2_k
sign       = sign_pre + 1.0
```

### Horner Polynomial

`sin(t)` for `t ∈ [-π/2, π/2]` is approximated by a degree-9 odd polynomial `t · P(t²)`, where:

```text
P(u) = (((R0·u + R1)·u + R2)·u + R3)·u + 1
```

The leading `1` constant in `P(u)` corresponds to the `t¹` coefficient of the Taylor series, and `R3 ≈ -1/6`, `R2 ≈ 1/120`, `R1 ≈ -1/5040`, `R0 ≈ 1/362880` correspond to the higher odd-power coefficients tuned for minimax accuracy on `[-π/2, π/2]`. Implementation:

```text
t2     = t * t
p_r0   = t2 * R0
p_r1   = p_r0 + R1
p_t2_r1= p_r1 * t2
p_r2   = p_t2_r1 + R2
p_t2_r2= p_r2 * t2
p_r3   = p_t2_r2 + R3
p_t2_r3= p_r3 * t2
p_one  = p_t2_r3 + 1.0
t_p    = t * p_one
out    = sign * t_p
```

The same polynomial is used for both sin and cos: the cos path differs only in the range reduction, so by the time `t` enters the polynomial it already lies in `[-π/2, π/2]` and the polynomial does not need separate coefficients.

### Sin vs Cos at a Glance

| Step | sin | cos |
| ---- | --- | --- |
| 1. k rounding | `round(x · 1/π)` (mode `ROUND`) | `rint(x · 1/π + 0.5)` (mode `RINT`) |
| 2. range reduction | `x - k·π` (4-part) | `x - k·π + π/2` (4-part + 2-part π/2) |
| 3. sign | `(-1)^k` | `(-1)^k` (same identity, different `k`) |
| 4. Horner | `t · P(t²)` | `t · P(t²)` (same polynomial) |
| 5. result | `sign · t · P(t²)` | `sign · t · P(t²)` |

## Constants

All constants are FP32 literals (the `k*` literals near the top of `src/ir/transforms/lower_composite_ops_pass.cpp`, matching the framework reference at `gitcode.com/cann/pypto:framework/src/interface/tileop/vector/unary.h`):

| Symbol | C++ literal | Role |
| ------ | ----------- | ---- |
| `PI_INV` | `0.31830988732818603515625f` | `1/π` (head) |
| `PI_V2` | `3.140625f` | π head (Cody-Waite part 1) |
| `PI_C1` | `0.0009670257568359375f` | π split-1 |
| `PI_C2` | `6.2771141529083251953125e-7f` | π split-2 |
| `PI_C3` | `1.21644916362129151821e-10f` | π split-3 |
| `PI_C4` | `-1.0290623200529979163e-13f` | π split-4 |
| `PI_HALF_HEAD` | `1.57079637050628662109375f` | π/2 head (cos only) |
| `PI_HALF_TAIL` | `-4.371139000189375e-8f` | π/2 tail (cos only) |
| `HALF` | `0.5f` | k-pre offset (cos), sign step |
| `M4` | `4.0f` | sign step |
| `NEG2` | `-2.0f` | sign step |
| `ONE` | `1.0f` | sign + Horner constant term |
| `R0` | `2.604926501e-6f` | Horner coeff (degree 9) |
| `R1` | `-1.980894471e-4f` | Horner coeff (degree 7) |
| `R2` | `8.333049340e-3f` | Horner coeff (degree 5) |
| `R3` | `-1.666665792e-1f` | Horner coeff (degree 3) |

`tile.cast` round modes (mirrors `src/ir/op/tile_ops/unary.cpp` registration):

| Symbol | Value | Meaning |
| ------ | ----- | ------- |
| `kCastModeNone` | `0` | no rounding (typically int → float) |
| `kCastModeRint` | `1` | round-half-to-even |
| `kCastModeRound` | `2` | round-half-away-from-zero |
| `kCastModeFloor` | `3` | round toward `-∞` |

## Numerical Properties

- **Absolute error**: ≤ ~1e-5 over `|x| ≤ 2π · 1024` (validated against NumPy by `tests/ut/ir/transforms/test_lower_composite_ops_numerical.py`). Inside one period, `max abs error` observed is ~1 ulp ≈ 1.19e-7.
- **Range-reduction breakdown**: beyond `|x| ≈ 2^17`, the FP32 representation of `x` itself loses fractional precision, so range-reduction error dominates regardless of how many π-correction terms are used. The 4-part Cody-Waite split chosen here is the standard CANN/PyPTO recipe and matches the reference implementation's behaviour on every tested `x` magnitude.
- **dtype**: FP32-only. FP16, BF16, and integer inputs are rejected at op-construction time (well before the pass runs) — see `tests/ut/ir/operators/test_tensor_ops.py` (tensor.sin/cos rejection) and `tests/ut/ir/operators/test_tile_ops.py` (tile.sin/cos rejection) for the rejection cases.
- **NaN/Inf**: NaN inputs propagate to NaN output (the polynomial preserves NaN). Inf inputs produce indeterminate values because the range-reduction `k = round(x/π)` step overflows; this matches the documented `|x| ≤ 2^17` validity range.

## Idempotency

Running `LowerCompositeOps` twice produces identical IR after the first run: the recipe emits only primitive ops (`tile.muls`, `tile.adds`, `tile.add`, `tile.sub`, `tile.mul`, `tile.cast`) and the mutator only rewrites `tile.sin` / `tile.cos` `Call`s, so the second invocation visits the body and changes nothing. This is verified by `test_sin_lowering_is_idempotent` and `test_cos_lowering_is_idempotent` in `tests/ut/ir/transforms/test_lower_composite_ops.py`.

## Implementation Notes

The mutator overrides `VisitStmt_(const AssignStmtPtr&)` rather than `VisitCall` because the decomposition splices ~33 statements per trig op into the surrounding sequence. Doing the splice from inside `VisitCall` would require returning multiple expressions, which `IRMutator` does not support; doing it from `VisitStmt_` lets `LowerSinCos` build a `vector<StmtPtr>` and return either a single bound `AssignStmt` or a fresh `SeqStmts`.

Each intermediate result is bound to a fresh `Var` named via `auto_name::BuildName` with the user's target name as the base. The mutator's `temp_counter_` is shared (by reference, through each `LoweringBuilder`) across all trig ops in a function so distinct ops do not collide on temp names.

The cast modes `RINT` (cos), `ROUND` (sin), `FLOOR` (sign), and `None` (int↔float) come from the tile-op registry's enum (`src/ir/op/tile_ops/unary.cpp`). Choosing the correct mode is load-bearing: `ROUND` for sin's `k` keeps `k` symmetric around zero so the Horner polynomial sees evenly distributed `t`; `RINT` for cos's `k` matches the `+0.5` shift and ensures even `k` corresponds to even multiples of `π/2`.

## Related

- **Issue**: [#1289 — Add FP32-only `tile.sin` / `tile.cos` and a lowering pass](https://github.com/hw-native-sys/pypto/issues/1289).
- **Reference implementation**: `gitcode.com/cann/pypto:framework/src/interface/tileop/vector/unary.h` — the upstream CANN/PyPTO recipe whose constants and op-sequence this pass mirrors verbatim.
- **Op deducer**: `DeduceTileFP32OnlyType` in `src/ir/op/tile_ops/unary.cpp:94` — enforces FP32-only at op-construction time.
- **Conversion registry**: `RegisterSimple("tensor.sin", "tile.sin")` and the cos counterpart in `src/ir/transforms/op_conversion_registry.cpp` — the upstream tensor-to-tile rewrite that produces the `tile.sin` / `tile.cos` calls this pass consumes.
- **Tests**: `tests/ut/ir/transforms/test_lower_composite_ops.py` (structural), `tests/ut/ir/transforms/test_lower_composite_ops_numerical.py` (NumPy-reference numerical).
