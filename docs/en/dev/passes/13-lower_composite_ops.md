# LowerCompositeOps Pass

Decomposes composite tile / distributed ops into compositions of primitive tile ops (`tile.muls`, `tile.adds`, `tile.add`, `tile.sub`, `tile.mul`, `tile.maximum`, `tile.minimum`, `tile.cast`) and distributed primitives, so codegen never has to emit a high-level intrinsic. Today the pass handles `tile.sin` / `tile.cos` (FP32 Cody-Waite + Horner) and `pld.tensor.*` distributed collectives (`allreduce` (mesh and ring), `allgather`, `reduce_scatter`, `broadcast`, `barrier`). Mesh and ring allreduce may also create a metadata-preserving `tensor.view` so tile load/remote/store operate on a 2D flattened target window. New composite ops add a lowering rule to the dispatch table inside the pass file without touching the dispatcher.

## Overview

`LowerCompositeOps` is a function-level pass that rewrites every `var = Call(...)` `AssignStmt` whose callee appears in the pass's lowering dispatch table. For `tile.sin` / `tile.cos`, the rule emits a fixed-shape primitive tile-op recipe using `tile.muls`, `tile.adds`, `tile.add`, `tile.sub`, `tile.mul`, and `tile.cast`: Cody-Waite range reduction (4-part π split) followed by a degree-9 odd Horner polynomial. For `pld.tensor.*` distributed collectives, the rules emit the cross-rank recipes documented below; `pld.tensor.allreduce` remains explicit-signal in InCore/composite lowering. The original target `Var` is preserved as the LHS of the final `AssignStmt`, so downstream uses keep the same name and identity.

Host-orchestrator `pld.tensor.allreduce` calls are skipped by this pass: `SynthesizeAllReduceSignals` first normalizes optional-signal host calls to the explicit-signal form, `MaterializeCommDomainScopes` places the data and signal windows into comm domains, and then `LowerHostTensorCollectives` lowers them to internal builtin dispatches.

The `tile.sin` / `tile.cos` rules are **FP32-only**. Non-FP32 trig inputs are rejected at op-construction time by the shared `DeduceTileFP32OnlyType` deducer (see `src/ir/op/tile_ops/unary.cpp:94`), so those rules only see well-typed FP32 operands. Distributed rules have their own dtype contracts; allreduce supports FP16 and FP32 as documented below.

The pass is **structural no-op** on programs that contain no registered composite call such as `tile.sin`, `tile.cos`, or `pld.tensor.*` distributed collectives: every other statement passes through `IRMutator::VisitStmt_`. The decomposition emits only primitive tile ops (`tile.muls`, `tile.adds`, `tile.add`, `tile.sub`, `tile.mul`, `tile.maximum`, `tile.minimum`, `tile.cast`) and distributed primitives, none of which the mutator rewrites — so the pass is also **idempotent**.

**Requires**: nothing.

**Produces**: nothing.

**Invalidates**: nothing.

The empty `PassProperties` contract (`kLowerCompositeOpsProperties` in `include/pypto/ir/transforms/pass_properties.h`) reflects that the lowering operates within the existing tile/distributed vocabulary plus metadata-only tensor views (`tensor.view`) used to expose canonical flattened windows; a partial-prefix ring view also carries a flattened `valid_shape`. The pass neither establishes nor breaks any `IRProperty`.

## When It Runs

`LowerCompositeOps` is the **first entry of `tile_pto_passes`** in the `Default` pipeline (see `python/pypto/ir/pass_manager.py`), running immediately after `ConvertTensorToTileOps` (slot 12) and `OptimizeOrchTensors` (slot 13). At this point all tensor-level transcendental calls (`tensor.sin`, `tensor.cos`) have been rewritten to their tile equivalents (`tile.sin`, `tile.cos`) by the conversion registry, and the tile pipeline is about to start tile-shape canonicalisation. Lowering trig before `FlattenTileNdTo2D` keeps the decomposition independent of the 2D-flattening rules — every primitive tile op in the recipe (`tile.muls`, `tile.adds`, `tile.add`, `tile.sub`, `tile.mul`, `tile.cast`) has well-defined behaviour at any rank.

## Architecture

The pass is a single translation unit, `src/ir/transforms/lower_composite_ops_pass.cpp`:

```text
src/ir/transforms/lower_composite_ops_pass.cpp
  LoweringBuilder           — per-call scratchpad (Bind + primitive tile-op builders:
                              tile.muls, tile.adds, tile.add, tile.sub, tile.mul,
                              tile.maximum, tile.minimum, tile.cast
                              + structured control-flow: EmitFor / EmitForReduce
                              / EmitIf / EmitIfExpr + NotEq scalar guard)
  CompositeLoweringFn       — (call, visited_args, builder) -> result expr
  Lower<Op>Rule             — one rule function per composite op (LowerSinRule,
                              LowerCosRule, LowerTensorAllReduceRule, ...)
  LookupCompositeRule       — file-local op-name → rule dispatch table (kRules)
  LowerCompositeOpsMutator  — walks the function, looks up a rule per Call
```

Adding a new composite op (all edits stay in `lower_composite_ops_pass.cpp`):

1. Write a `Lower<Op>Rule(call, args, builder)` function. It receives the original `CallPtr` (use `call->span_`, `call->kwargs_`, `call->op_->name_` as needed), the visited arg expressions (var-remap already applied), and a `LoweringBuilder` whose `Bind` helper appends an `AssignStmt` per intermediate temp. For rules that need control flow, use `builder.EmitFor` / `builder.EmitForReduce` / `builder.EmitIf` / `builder.EmitIfExpr` — each takes a body callback that receives a nested builder sharing the same temp counter, so emitted temps stay uniquely named regardless of nesting depth. `LowerTensorAllReduceRule` is the canonical example of a control-flow-bearing rule (ready barrier plus chunked remote_load+accumulate / barrier / store for mesh; `LowerTensorRingAllReduceRule` adds a chunked RS+AG ring schedule dispatched via a `mode` kwarg).
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

Running `LowerCompositeOps` twice produces identical IR after the first run: the recipes emit only primitives such as `tile.muls`, `tile.adds`, `tile.add`, `tile.sub`, `tile.mul`, `tile.maximum`, `tile.minimum`, and `tile.cast`, plus the distributed primitives listed below. The mutator only rewrites registered composite calls (`tile.sin`, `tile.cos`, `pld.tensor.*` distributed collectives, ...), so the second invocation visits the body and changes nothing. This is verified by the sin/cos and distributed-collective idempotency tests in `tests/ut/ir/transforms/test_lower_composite_ops.py`.

## `pld.tensor.*` distributed collectives

The pass also lowers the `pld.tensor.*` family of window-bound distributed collectives. Each collective is a single composite `Call` that expands into a notify / wait + data-movement recipe, plus a self-clearing epilogue. The data-movement primitive differs by op: `allgather` uses `pld.tile.put` (TPUT-based, auto-chunks through a VEC staging tile), `broadcast` relocates window data with `pld.tile.get` (GM→GM copy), while `allreduce` and `reduce_scatter` pull peer chunks into a UB tile with `pld.tile.remote_load`. Allreduce selects `tile.add`, `tile.maximum`, `tile.minimum`, or `tile.mul`; reduce-scatter currently accumulates with `tile.add`. All seven rules share the same **self-clearing credit-barrier protocol** (`LoweringBuilder::EmitBarrier` + `EmitEpilogueReset`) — see [Barrier-signal protocol](#barrier-signal-protocol) below — so a `signal` buffer is reusable across back-to-back calls, and even inside `for` / `while` / `if`.

### Barrier-signal protocol

Every call issues `N` barriers — `AtomicAdd(1)` into each peer's cell, then `Wait(>= g)` where `g` counts up **within that call only** (every fresh call restarts at 1, via `LoweringBuilder::EmitBarrier`'s call-local `barrier_count_`). After the body, an epilogue (`LoweringBuilder::EmitEpilogueReset`) subtracts the call's total credit `N` back out of every non-self cell with a single `AtomicAdd(-N)`. Because atomic adds and subtracts commute, the signal is provably all-zero again once every rank has finished its epilogue — there is no cross-call state to get wrong, so the *next* call on the same signal also starts at generation 1.

`N` may be a **runtime scalar** (`pld.system.notify`'s `value` only requires `ScalarType`), so a mesh allreduce's per-chunk credit count does not need to be a compile-time constant — unlike the barrier count itself, which is always a small compile-time sequence (`1`, `2`, ...) because each `Wait`'s `expected` must be resolvable without knowing how many times the surrounding (possibly dynamic) loop will execute at runtime.

`kGe` (not `kEq`) is the load-bearing choice for every wait predicate: a fast peer can advance a cell past the value a slow rank is looking for before that rank ever polls it, so equality would deadlock. For the same reason `kSet` must never be mixed with `kAtomicAdd` on the same cells — a set could clobber an already-advanced counter.

Mesh (`[NR, 1]`, one cell per rank) and ring (`[2*(NR-1), NR]`, one row per round) signals use incompatible cell addressing, so sharing one buffer between the two is rejected as a shape mismatch (`ValidateMeshSignalShape`) — the only restriction the protocol still imposes.

### `pld.tensor.allreduce`

The allreduce rule starts with a cross-rank ready barrier on shared `signal` cells (generation 1). It then processes a fully-valid target in UB-sized chunks. Every chunk performs a peer reduction, then barriers on its own call-local generation (`2`, `3`, ...) before storing its result — preventing a fast rank from overwriting bytes a slow rank has not remote-loaded yet. The epilogue then subtracts the call's total credit (`1 + chunk_count`, built as an IR expression since `chunk_count` may depend on a runtime extent) back out of every non-self row. The partial-valid single-rectangle path issues exactly two barriers (ready + post-reduce) and its epilogue subtracts `2`.

For a fully-valid packed target, mesh lowering creates a logical `[1, product(all dimensions)]` view and traverses it with physical tiles of at most 16 KiB. A statically known extent smaller than the budget shrinks the chunk to the smallest 32-byte-aligned physical width that covers it, so small allreduces do not reserve a full 16-KiB tile while remaining legal PTO tiles. The tail carries `valid_shape=[1, min(chunk, remaining)]` through both `tile.load` and `pld.tile.remote_load`, so the allocation stays static while the read/store extent is exact. If an ND target carries a partial `TensorView.valid_shape`, the pass preserves and reduces the representable `[rows, cols]` rectangle through the established single-rectangle path. Constant valid rectangles use their compact shape; symbolic valid extents fall back to the source's physical rectangle when that statically bounded rectangle fits within one 16-KiB chunk. Oversized partial rectangles, strided targets, DN partial views, and partial boxes that cannot be represented by the leading-dimension collapse are rejected explicitly.

Ring lowering uses one packed 2D view for its reduce-scatter and allgather phases. A fully valid target becomes `[1, SIZE]`; a contiguous partial prefix keeps physical shape `[1, product(target.shape)]` and carries logical `TensorView.valid_shape=[1, product(target.valid_shape)]`. FP32 retains balanced `floor(i * SIZE / NR)` segment boundaries. FP16 rounds each interior boundary up to 16 elements and caps it at `SIZE`; consequently every non-empty segment and every UB subchunk starts at a 32-byte-aligned address. A ragged FP16 remote load may read the aligned physical tail reserved by the communication domain, then `tile.set_validshape` restores the logical extent before reduction and store. This supports non-divisible inputs and `SIZE < NR` without inserting holes into the public tensor layout. Every subchunk of every round barriers on a call-local ready + read-complete generation pair; the epilogue then subtracts `2 * chunk_count` (uniform across rounds, since every round's subchunk loop shares the same bound) from every row of the `[2*(NR-1), NR]` signal.

Any symbolic target or partial-valid extent that survives lowering must be runtime-bound by a kernel scalar, loop variable, or physical tensor-shape parameter; a type-metadata-only symbol is rejected during PTO codegen. A fully dynamic physical target dimension is bound from that tensor parameter.

Signal buffers are reusable across back-to-back allreduce calls — including a mesh allreduce over a symbolic extent, whose credit total is simply a runtime-computed expression rather than a value the compiler must know. See [Barrier-signal protocol](#barrier-signal-protocol) above.

Mesh and ring lowering support FP16 and FP32 with `ReduceOp::kSum`, `kMax`,
`kMin`, and `kProd` for arbitrary positive element counts.

### `pld.tensor.allgather`

Signature: `allgather(local_data, target, signal)`. `local_data` is this rank's chunk (`Tensor` or `Tile` `[1, SIZE]`), `target` is a window-bound `DistributedTensor[NR, SIZE]` staging area that also serves as the result, and `signal` is the INT32 barrier. Push-based decomposition:

- ``tile.create([1, SIZE], dtype=..., target_memory=Vec)`` — allocate a VEC staging tile for ``pld.tile.put`` auto-chunking.  ``pld.tile.put`` reads directly from the ``local_data`` Tensor (or Tile) source — no explicit ``tile.load`` is emitted.
- Phase 1: for `peer` in `0..NR-1`, `pld.tile.put(target, peer, local_data, put_stage, [my_rank, 0], [0, 0], [1, SIZE])` — push this rank's chunk into every peer's window at row `my_rank`. Self-store (`peer == my_rank`) uses HCCL identity mapping. `pld.tile.put` auto-chunks when SIZE exceeds the staging-tile capacity
- Phase 2: barrier (generation 1) + epilogue (subtract 1 from every non-self cell)
- Return `target` — the window IS the gathered `[NR, SIZE]` result (window-as-result, `DistributedTensor`)

Compared to the original pull-based allgather (4-arg with a separate `out` tensor), this push-based variant drops the `out` parameter and the per-peer `pld.tile.get` gather loop. Total HBM drops from `(NR+1)×SIZE` to `NR×SIZE`, at the cost of the window remaining occupied until the caller consumes the result.

### `pld.tensor.reduce_scatter`

Decomposes into the same phase shape as `allreduce`'s rectangle path:

- Phase 2a/2b: ready barrier (generation 1)
- Phase 3: for each peer, `remote_load` chunk `r` from peer `p` and accumulate into a local scratch with `tile.add`
- Phase 3.5a/3.5b: post-reduce barrier (generation 2)
- Phase 4: `tile.store` the reduced chunk `r` back into `target[r, 0:SIZE]`
- Epilogue: subtract 2 from every non-self cell

`target` has shape `[NR, SIZE]`; each rank stages all `NR` chunks before the call. After the call, rank `r`'s row `[r, 0:SIZE]` holds the element-wise sum of chunk `r` across all ranks. The post-reduce barrier is required for the same WAR reason as `allreduce`.

Only `ReduceOp::kSum` is supported in the first version; the C++ deducer rejects `Max` / `Min` / `Prod`.

### `pld.tensor.broadcast`

Decomposes into a 3-phase recipe:

- Phase 2: barrier (generation 1) + epilogue (subtract 1 from every non-self cell)
- Phase 3: `tile.create` (VEC staging tile) + `pld.tile.get(target, peer=root, target, stage)` on every rank — each rank reads root's slice into its own `target`. For `peer == root` the HCCL identity mapping makes the get a local no-op, so root keeps its own data while non-root ranks receive root's.

`root` is a static `int` kwarg known at compile time.

### `pld.tensor.barrier`

Pure synchronization — no data movement. Decomposes into a barrier (generation 1) plus its epilogue (subtract 1 from every non-self cell).

The returned expression is the same `signal` tensor, enabling the rebind idiom `signal = pld.tensor.barrier(signal)`. Under the self-clearing protocol every call restarts at generation 1 regardless of how many times the signal has already been used, so the rebind no longer needs to chain onto any prior state.

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
