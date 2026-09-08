# LegalizeTileCast Pass

Expands `tile.cast` `(src, dst)` pairs that the active `pto.tcvt` profile cannot emit as a single instruction into the shortest chain of native casts.

## Overview

For each `var = tile.cast(...)`:

1. Ask the active `BackendHandler` for its native-conversion table via `GetTcvtAdjacency()`
   (transcribed from pto-isa `tcvt` Supported Conversions). The pass holds no per-architecture
   knowledge of its own, so a new backend ships its own table and needs no change here.
   With no backend configured the pass is a no-op.
2. Already native: leave unchanged (including FIXPIPE-foldable `FP32→BF16/FP16` with `mode=rint`).
3. Non-native: BFS for a shortest path; among equal-length paths prefer "same byte-width → float, then adjust width".

Typical A5 results: `INT32→FP16` → `INT32→FP32→FP16`; `FP16→BF16` → `FP16→FP32→BF16`.

Unreachable pairs hard-fail with src/dst/arch in the diagnostic.

**Requires / Produces / Invalidates**: none (empty `PassProperties`).

## Destination saturation

`pl.cast`, `pl.tensor.cast` and `pl.tile.cast` take a keyword-only
`saturation_mode`, spelled `"on"` / `"off"` or `1` / `0`:

```python
quantized = pl.cast(rounded_fp16, pl.INT8, mode="trunc", saturation_mode="on")
```

`"on"` clamps a rounded value that falls outside the destination range to that
range. `"off"` selects the target's non-saturating conversion, whose overflow
and non-finite behaviour is architecture-defined. A `Scalar` input rejects the
option.

**`"on"` is the default.** Clamping is the safer of the two to get by accident,
and on A2/A3 it is also the one the assembler converts natively rather than
emulating with a chunked vector sequence — so the default is both the safer and
the faster lowering. Pass `"off"` where wrapping is the kernel's contract.

The IR records only a *deviation* from that default: a cast that wants `"on"`
carries no `saturation_mode` kwarg, which is the same shape a pass-synthesized
cast has. That is what keeps a printed cast re-parsing to structurally equal IR —
stamping the default would make two forms differ with no semantic difference
between them. Codegen reads the default through, so the emitted `pto.tcvt` always
carries an explicit `satmode` either way and a reader of the MLIR never has to
know what the assembler would have picked.

The two modes agree only on values the destination can already represent, so this
default is a behavioural choice, not a no-op: a kernel that relied on wrapping
must now say `"off"`.

**Legalized chains: the request rides the final hop.** Saturation names the
*destination* range, and only the last hop reaches the destination dtype;
stamping an intermediate would clamp to a range the author never named.
Intermediates therefore keep exactly their previous behaviour — the original
rounding mode and nothing else.

Deferring costs nothing, because the BFS above already refuses any intermediate
that narrows relative to the destination: every value the destination *can*
represent reaches the final hop exactly, so `"on"` and `"off"` still agree there.
A value the destination cannot represent is out of range at both ends of the
chain — an intermediate float may overflow it to an infinity, but with its sign
intact, so it clamps to the same endpoint a hypothetical single-step conversion
would have picked. Non-finite inputs remain outside what either mode defines.

The same rule carries an explicit `pl.tile.cast(..., tmp=...)` scratch operand
onto the final hop, which is the narrowing one.

**A2/A3 scratch.** `InitMemRef` synthesises a scratch tile only for the
*non-saturating* narrowing `pto.tcvt`, whose PTOAS lowering emulates the
target's overflow behaviour with a chunked vector sequence. Saturating selects
the native conversion, which reads no scratch — so with `"on"` as the default,
only a cast that explicitly opted out allocates a tile at all. A caller-supplied
`tmp` is never dropped.

## Native casts vs legalized chains

`pl.cast` does not always compile to one instruction. Whether a given
`(src, dst)` pair is a single hardware `pto.tcvt` or a legalized chain depends
entirely on the target architecture, and the difference is visible in both
performance and numerics. The tables below list, per architecture, which pairs
are native and which this pass expands; `(n)` is the number of `tcvt`
instructions emitted.

**Cost.** A legalized chain issues one `tcvt` per hop over the whole tile, plus
an intermediate tile per hop (subject to the usual buffer reuse). A 3-hop chain
is therefore roughly three times the vector work of a native cast on the same
shape. In a vector-bound kernel that is a real cost — if a hot loop shows an
unexpected chain, consider whether an equivalent dtype path avoids it.

**Numerics.** A chain is bit-identical to a direct conversion when every
intermediate represents the source values *that fall inside the destination's
range* exactly — then only the final hop rounds. `INT32 -> FP32 -> FP16` is in
this class: FP16 saturates above 65504, and every integer below that is exact in
FP32, so the FP32 hop never rounds.

Where an intermediate rounds first, the chain double-rounds and can land one ULP
of the destination away from a directly rounded conversion:

| Chain | Why it double-rounds | Measured |
| ----- | -------------------- | -------- |
| `INT32 -> FP32 -> BF16` | BF16's range covers all of INT32, but FP32 holds only 24 significand bits, so the first hop rounds inputs above 2^24 | 3 in 200000 uniform INT32 differ by 1 BF16 ULP |
| `FP32 -> FP16 -> INT8` | FP16 rounds before the integer hop | boundary values only |

This is not a regression against a better option: the ISA offers no direct
conversion for these pairs at all, so the chain is the only available lowering.
It also matches the reference — `torch`'s own `int32 -> bfloat16` agrees with the
chain on all of 2000000 uniform INT32 samples, because torch bridges through
fp32 the same way.

**How to check what your kernel got.** Each hop appears as a `pto.tcvt` in the
generated MLIR:

```bash
grep -n 'pto.tcvt' build_output/<case>/ptoas/<kernel>.pto
```

### Ascend950 (a5)

| from | native (1 instruction) | legalized chain (n instructions) |
| ---- | ---------------------- | -------------------------------- |
| `bf16` | `fp16`, `fp32`, `fp4`, `int32` | `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `int16`(2), `int8`(2), `uint16`(2), `uint8`(2) |
| `fp16` | `fp32`, `hf8`, `int16`, `int32`, `int8`, `uint8` | `bf16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `uint16`(2), `fp4`(3) |
| `fp32` | `bf16`, `fp16`, `fp8e4m3`, `fp8e5m2`, `hf8`, `int16`, `int32`, `int64` | `fp4`(2), `int8`(2), `uint16`(2), `uint8`(2) |
| `fp4` | `bf16` | `fp8e4m3`(3), `fp8e5m2`(3), `hf8`(3), `int8`(3), `uint8`(3) |
| `fp8e4m3` | `fp32` | `bf16`(2), `fp16`(2), `fp8e5m2`(2), `hf8`(2), `int16`(2), `fp4`(3), `int8`(3), `uint16`(3), `uint8`(3) |
| `fp8e5m2` | `fp32` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `hf8`(2), `int16`(2), `fp4`(3), `int8`(3), `uint16`(3), `uint8`(3) |
| `hf8` | `fp32` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `int16`(2), `fp4`(3), `int8`(3), `uint16`(3), `uint8`(3) |
| `int16` | `fp16`, `fp32`, `int32`, `uint32`, `uint8` | `bf16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `int8`(2), `uint16`(2), `fp4`(3) |
| `int32` | `fp32`, `int16`, `int64`, `uint16`, `uint8` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `fp4`(3), `int8`(3) |
| `int64` | `fp32`, `int32` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `int16`(2), `uint16`(2), `uint8`(2), `fp4`(3), `int8`(3) |
| `int8` | `fp16`, `int16`, `int32` | `hf8`(2), `uint16`(2), `uint8`(2), `fp8e4m3`(3), `fp8e5m2`(3), `fp4`(4) |
| `uint32` | `int16`, `uint16`, `uint8` | `int8`(3) |
| `uint8` | `fp16`, `uint16` | `hf8`(2), `int8`(2), `fp8e4m3`(3), `fp8e5m2`(3), `fp4`(4) |

### Ascend910B (a2a3)

| from | native (1 instruction) | legalized chain (n instructions) |
| ---- | ---------------------- | -------------------------------- |
| `bf16` | `fp32`, `int32` | `fp16`(2), `int16`(2), `int4`(3), `int8`(3), `uint8`(3) |
| `fp16` | `fp32`, `int16`, `int32`, `int4`, `int8`, `uint8` | `bf16`(2) |
| `fp32` | `bf16`, `fp16`, `int16`, `int32`, `int64` | `int4`(2), `int8`(2), `uint8`(2) |
| `int16` | `fp16`, `fp32` | `bf16`(2), `int4`(2), `int8`(2), `uint8`(2) |
| `int32` | `fp16`, `fp32`, `int16`, `int64` | `bf16`(2), `int4`(2), `int8`(2), `uint8`(2) |
| `int4` | `fp16` | `int8`(2), `uint8`(2) |
| `int64` | `fp32`, `int32` | `bf16`(2), `fp16`(2), `int16`(2), `int4`(3), `int8`(3), `uint8`(3) |
| `int8` | `fp16` | `int4`(2), `uint8`(2) |
| `uint8` | `fp16` | `int4`(2), `int8`(2) |

Pairs absent from both columns are rejected: the pass reports src, dst and arch
rather than emitting a lossy chain.

## When it runs

Default pipeline:

```text
lower_composite_ops → flatten_tile_nd_to_2d → legalize_tile_cast → auto_tile_matmul_l0
```

## API

| C++ | Python |
| --- | ------ |
| `pass::LegalizeTileCast()` | `passes.legalize_tile_cast()` |
