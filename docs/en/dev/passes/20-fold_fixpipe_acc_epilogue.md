# FoldFixpipeAccEpilogue

Folds a vector dequantization / ReLU epilogue into the cube's `Acc → GM`
writeback, so a scaled and activated matmul stays a pure-cube kernel.

## Why this is a pass and not a micro-optimization

Writing `store(maximum(cast(acc, FP32) * scale, 0))` costs far more than the
three vector instructions suggest. `InferTileMemorySpace` assigns the cast,
multiply and maximum to `Mem.Vec`, which splits one cube kernel into an AIC and
an AIV function joined by a cross-core round-trip:

```text
kernel_aic:  tile.matmul -> tpush_to_aiv -> ... -> tpop_from_aiv -> tile.store
kernel_aiv:  tpop_from_aic -> cast -> muls -> maximums -> cast -> tpush_to_aic
```

That pair costs a `tpush_to_aiv` / `tpop_from_aic` / `tpush_to_aic` /
`tpop_from_aiv` sequence, two `tfree`s, `aic_initialize_pipe`, and a GM slot
buffer for the C2V ring. At `[128, 128]` on a2a3 the ring's default depth plus
the vector tiles **overflow Vec space outright**, so the kernel does not compile
at all:

```text
ValueError: Vec buffer usage (196608 bytes) exceeds platform limit (188416 bytes).
The first 131072 bytes of that space are reserved by system.reserve_buffer — this
is the cross-core pipe ring.
```

The fix-pipe performs the multiply and the activation while draining L0C. Folded,
the same kernel is one AIC function with zero cross-core ops.

## Position

Between `CanonicalizeTileSlice` (19) and `InferTileMemorySpace` (21). Before the
latter is load-bearing: that pass is what assigns the vector chain to `Mem.Vec`
and spawns the cross-core transfer, so folding afterwards would be too late.
Running before it also means memory spaces are *not yet resolved*, which is why
the pass identifies the accumulator by **operator identity** (`tile.matmul`,
`tile.matmul_acc`, `tile.matmul_bias` produce Acc by construction) rather than by
`memory_space_`.

## What it recognises

Every link must be the **sole** consumer of the previous one and a top-level
assignment, so folding can never delete a value someone else still reads. All
links are optional except that at least one must be present:

```text
acc = tile.matmul(...)                    # Acc by construction
  [ t = tile.maximums(acc, 0.0) ]         # -> pre_relu=True   (form A)
  [ t = tile.cast(t, FP32) ]              # absorbed: FIXPIPE multiplies in FP32
  [ t = tile.muls(t, <const s>) ]         # -> pre_quant=s
  [ t = tile.maximums(t, 0.0) ]           # -> pre_relu=True   (form B)
  [ t = tile.cast(t, DST) ]               # the writeback's own conversion
  out = tile.store(t, offs, dst)          # rewritten to read `acc` directly
```

The chain statements are deleted and the store becomes
`tile.store(acc, offs, dst, pre_quant=s, pre_relu=r)`, keeping every kwarg the
original store carried (`atomic`, `st_phase`, …).

### The two ReLU positions are not equivalent

The hardware computes `clamp(ReLU(acc) * s)` — the activation is a *pre*-quant
stage that reads the raw accumulator, with the destination clamp last (measured
on a2a3 by `acc_to_gm_negative_scale`; see `99-verifier.md`).

| Form | Chain | Foldable |
| ---- | ----- | -------- |
| A | `muls(maximums(acc, 0), s)` | Always — this *is* the hardware's order |
| B | `maximums(muls(acc, s), 0)` | Only when `s >= 0` |

For `s >= 0` the two are the same function; for `s < 0` they disagree at every
element. B is the spelling users write (dequantize, then activate), so it is
folded under a sign guard rather than rejected outright.

## What it declines

A decline leaves the IR **byte-identical** — an optimization that cannot apply
must be a no-op, never a partial rewrite. Actionable declines emit a PerfHint.

| Situation | Code | Why |
| --------- | ---- | --- |
| Scale is not a compile-time constant | `PH-FE-001` | The packed word is built at compile time |
| Form B with `s < 0` | `PH-FE-002` | Not the hardware's order (above) |
| Backend has no scale-bearing `(acc, dst)` mode | `PH-FE-003` | `SupportsFixpipePreQuant`; folding would produce IR `AccToGmStoreValid` rejects |
| Cast uses the frontend default `mode="round"` | `PH-FE-004` | FIXPIPE rounds half-to-**even** (`RINT`); `ROUND` breaks ties away from zero, so folding would change results at ties. Pass `mode="rint"` to opt in |
| Cast requests an explicit `saturation_mode` | — | `pto.tstore` carries no `satmode` |
| Any intermediate read more than once | — | Folding deletes the chain |
| The store already carries an epilogue | — | The two would have to be composed |
| Target is `Mat` | — | Withheld by both handlers pending PTOAS#1570 |

The cast rules deliberately match `CastFoldableToFixpipeMat`
(`auto_tile_matmul_l0_pass.cpp`), the existing unscaled fold — the two compete
for the same IR and must agree about which casts the fix-pipe can reproduce.

## Scope

Only the `Acc → GM` (`pto.tstore`) writeback is folded. The `Acc → Mat`
(`pto.tinsert`) form is withheld by both backend handlers because ptoas mis-emits
the scale there ([PTOAS#1570](https://github.com/hw-native-sys/PTOAS/issues/1570),
mechanism in `99-verifier.md`); emitting it would produce IR
`FixpipeEpilogueValid` rejects. When ptoas is fixed, that arm is a handler
one-liner plus a second rewrite target here.

## Complexity

One pass to count uses, one to index consumers, one mutation walk — O(N log N)
with hash-map lookups, per `pass-complexity.md`.

## Tests

`tests/ut/ir/transforms/test_fold_fixpipe_acc_epilogue.py` — both folding forms,
and every decline asserted as `assert_structural_equal(After, Before)`.
