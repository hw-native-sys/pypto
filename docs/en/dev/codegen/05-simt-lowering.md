# SIMT lowering and execution

The public `simt_scope`/JIT surface outlines into `FunctionType.SIMT` helpers
and `SimtLaunchStmt`, then lowers through the production A5 compiler and
persistent runtime. See the [canonical SIMT IR](../ir/11-simt.md).

## Supported executable subset

- A5 AIV callers, with constant three-dimensional block sizes specialized by
  `SpecializeSimtHelpers` (up to 2,048 threads).
- Borrowed GM Tensor and UB Vec Tile storage with supported non-packed numeric elements.
  Tiles require affine, unboxed layouts. Physical strides and Tile start offsets
  determine addressing; logical `valid_shape` never replaces the physical stride.
  Borrowed `tile.slice`/reshape view handles are rejected because their types
  may omit the parent's physical pitch. Materialize the window into independent
  storage with `tile.extract` before launching. View provenance follows aliases,
  conditional results, and loop carries, including loop backedges. Planner-inserted
  `tile.move` operations do not establish independent dense storage.
  In-place `tile.assemble` results inherit their destination's provenance, so
  updates to dense storage remain supported while updates to views remain rejected.
  Explicit strides on supported storage take precedence; dense DN Tensor fallback transposes only the trailing
  two axes and keeps batch axes outermost.
- FP16/BF16/FP32, signed/unsigned 8/16/32/64-bit integers, INDEX and BOOL scalar values; structured `if`, `for`, and `while`.
  Signed integer floor division/modulo retain floor semantics. FP32 inequality
  treats NaN as unequal, and numeric-to-BOOL conversion compares against zero.
- Thread/block/grid coordinates and extents, lane ID, and scalar loads/stores.
  GM cache policies pass through unchanged to native PTOAS operations.
- Active-lane votes/ballot, FP16/FP32 and 32/64-bit integer shuffle (width 16 or 32), and
  FP16/FP32 and 32-bit integer add/min/max redux. Integer min/max lower with explicit signed or unsigned
  comparison, inferred from unsigned dtypes or the explicit carrier flag. Shuffle accepts a source-window `bound`. See the [collective contract](../ir/11-simt.md#warp-collectives).

FP8 and packed storage, scalar exponentiation (`**`), floating-point floor division/modulo,
early return, break/continue, boxed Tile layouts and grouped kernel execution
currently produce explicit diagnostics. CPU simulation (`a5sim`) and other device
backends cannot execute these native helpers. As in canonical IR, indexing must
be in bounds and simultaneously written locations must not race.

## ABI and synchronization

Codegen retains reachable helpers when splitting the program into compilation
units. Helpers are never registered as independent runtime tasks. Each helper
artifact contains a native SIMT entry and an exported AIV launch wrapper; scalar
arguments are passed by value, Tensor arguments use GM pointers, and Tile
arguments use UB pointers. The specialization's thread/register limits become
`pto.simt_max_threads` and `pto.simt_max_regs`.

Runtime Tensor pointers already include the view origin. Tile pointers name the
allocation base, so the native body adds `TileView.start_offset` once. Dynamic
metadata captured by specialization remains a scalar ABI argument. Repeated
launches with the same specialized helper and caller types share an artifact.

The caller retains an annotated `simd` adapter with whole-storage read/readwrite
effects. A partial write remains readwrite so untouched values and prior
producers stay ordered. PTOAS InsertSync owns MTE/V dependencies and allocation
reuse synchronization. PyPTO adds no fixed barrier around a launch. Launch results
alias the writable handle selected by `MaterializeSimtOutputs`; original Tile
values remain distinct, while GM results retain their Tensor identity.

## Native compilation and artifacts

The existing VF artifact path also carries helpers marked `kind: simt` in the
manifest. It packages relative helper paths, lowers through PTOAS LLVM export,
compiles device objects with Bisheng, links `kernel_entry` first, validates the
ELF, and supplies its `.text` to the persistent runtime. Helper contents and
compiler identities participate in binary caching; relocating an artifact does
not require its original helper paths.

LLVM export bypasses PTOAS's object-level SIMT size repair. Intrinsic-only helpers
continue through LLVM export and Bisheng. SIMT helpers containing inline assembly
use PTOAS's native device-object emission. PTOAS validates or repairs `VF_SIMT`
sizes. PyPTO adds logical-name
aliases with a relocatable link for the emitted `.vector` or `_mix_aiv` ABI symbols.
The selected SDK's CANN installation is passed to PTOAS explicitly. Native object
flags participate in cache identity, and compilation or alias-link failures do
not fall back to unchecked LLVM compilation. No instruction bytes are patched by
PyPTO. Acceptance tests compare each linked `VF_SIMT` size with its callee symbol
size; this compilation support does not expose a public `keep/resume` API.

## Focused acceptance

`tests/st/runtime/vpto/test_simt_ir_onboard.py` now compiles public `@pl.jit.aiv`
functions containing `pl.simt_scope`, through `.compile()`, without replacing
generated sources. It replaces the earlier P3 canonical-IR-only authoring exception.

- GM copy covers Tensor subviews with nonzero origins, guards, inactive elements,
  and a downstream Tile load consuming the SIMT result.
- UB gather covers two-dimensional threads, INT32 indices, a vector producer,
  partial writes, and a later read of the original Tile.
- Both check cold compilation, warm cache loading, and directory relocation.
  Seven lengths and three input variants at each stage give 126 runtime cases.

The public JIT versions passed all 126 numerical checks on Ascend950DT_9581
with PTOAS v0.72, CANN 9.3.0 and pinned runtime
`391b2fbcd4dd053204a677662c39b6b7636e8043`. Build all artifacts in the current
checkout and reserve a device through the host queue before running:

```bash
PYPTO_RUN_VPTO_ONBOARD=1 PYPTO_VPTO_DEVICE_ID="$DEVICE_ID" \
PYPTO_VPTO_TIMEOUT=600 python -m pytest \
  tests/st/runtime/vpto/test_simt_ir_onboard.py --platform=a5 -k persistent -v
```

The `build` variants use `PYPTO_RUN_VPTO_VF_VALIDATION=1` and omit device execution.
On a host where pytest requires visible device nodes even for build-only tests,
invoke the bounded build worker directly with an existing output directory:

```bash
python tests/st/runtime/vpto/test_simt_ir_onboard.py --worker build/simt-ir-runtime gm
python tests/st/runtime/vpto/test_simt_ir_onboard.py --worker build/simt-ir-runtime ub
```

Warp acceptance is separate from the earlier GM/UB acceptance above.
`tests/st/runtime/vpto/test_simt_collectives_onboard.py` provides opt-in build
and device variants using the same environment gates. The unit JIT test also
exports every collective form through native PTOAS to LLVM without a device.

The collective acceptance passed all seven active-lane cases (0, 1, 15, 16,
17, 31 and 32) across two warps on Ascend950DT_9581 with PTOAS v0.72, CANN
9.3.0 and runtime `391b2fbcd4dd053204a677662c39b6b7636e8043`. This checks all
11 APIs with INT32/FP32 payloads and widths 16/32, including divergent
participation and untouched inactive outputs.

## Atomic and synchronization acceptance

The native mapping uses `pto.atomic_add/min/max/exch/cas`, `pto.syncthreads`,
`pto.threadfence`, and `pto.threadfence_block`. Integer min/max carry `signed` or `unsigned`.
Atoms use the same affine element offset calculation as scalar memory operations,
followed by `pto.addptr`; no untyped byte-address or allocation bypass is exposed.
All atomic targets are ReadWrite, even when the old scalar is unused. Synchronization
calls remain ordered effects; `no_arg_writes` does not declare them pure.

| Capability | Native compilation | A5 numerical coverage |
| ---------- | ------------------ | --------------------- |
| Five atomics × INT32/FP32 × GM/UB | All 20 combinations, consumed and discarded results | Contention, old values, failed/successful CAS, untouched elements |
| INT32 Add/Min/Max/CAS boundaries | GM and UB | Wraparound, signed extrema, conditional replacement |
| FP32 Add/Min/Max/CAS boundaries | GM and UB | NaNs, infinities, signed zero, bitwise CAS |
| `syncthreads` and both fences | LLVM workitem synchronization/fence intrinsics | Four-warp UB/GM publication and consumption; subsequent MTE output |

This matrix was validated on Ascend950DT_9581, PTOAS v0.72, CANN 9.3.0 and
runtime `391b2fbcd4dd053204a677662c39b6b7636e8043`. These device results cover naturally aligned four-byte atomic elements. Wider
and half/bfloat16 interfaces now have separate compile coverage; see the
[per-operation matrix](06-simt-interface-coverage.md). Packed values, selectable
ordering/scope and cross-AIV synchronization remain outside this frontend contract. See [API semantics and limits](../ir/11-simt.md#atomics-and-synchronization).
The publication test uses a rendezvous; it does not establish that a fence alone
is a synchronization protocol or promise system-wide visibility.

`tests/st/runtime/vpto/test_simt_atomics_onboard.py` provides the contention and
barrier cases (15 checks). `test_simt_atomic_edges_onboard.py` provides boundary
cases (9 checks). Both expose opt-in `build`/`persistent` pytest variants under
the same environment gates as the other SIMT tests, plus `--worker OUTPUT [DEVICE]`.
The unit JIT tests compile all four entries and export native LLVM without a device.
Structural regressions reject readonly/uninitialized atomic captures and nested
barriers or bypassing exits. No lock-free forward-progress guarantee is tested.

`tests/st/runtime/vpto/test_simt_atomic_regressions_onboard.py` additionally
checks all five atomics at rank-3 UB coordinate `(1, 2, 7)` in a `[2, 3, 32]`
Tile (physical element 167), untouched elements, and atomic evaluation counts
in arithmetic, branches and while conditions. Its three A5 checks complement
native LLVM assertions for the address and retained calls. Codegen rejects
index counts that disagree with the post-flatten storage rank.

The condition regressions include unselected atomic tuple elements before and after the selected element, and nested tuple projections. Each atomic runs twice for a one-iteration loop: once for the true condition and once for the final false condition.

JIT loop induction variables shadow prior scalar aliases and folded shape values within the loop body. Loop bounds use the incoming binding, and nested-loop exit restores the enclosing binding. The A5 regression reuses a while counter as a subsequent copy-loop index to verify this distinction.

`pl.while_` uses `for` syntax but binds only loop-carried values, with no induction variable. Assignment-form `pl.yield_` targets define the post-loop outputs, including zero-iteration results. Incoming aliases and folded shapes are shadowed inside the loop; an outer binding is restored only when its name is not an output of that loop. JIT and the parser share yield ownership rules: yields in nested conditionals or real loops belong to those constructs. Runtime scopes (`pl.scope` / `pl.manual_scope`) and `pl.split_aiv` are transparent wrappers. The A5 regression checks that conditional yields do not replace an outer scalar alias.

## Routing compaction acceptance

The `test_simt_compaction` case in
`tests/st/runtime/vpto/test_simt_collectives_onboard.py` exercises the public
[`04_simt_compaction.py`](../../../../examples/advanced/04_simt_compaction.py)
example. Ballot and shuffle-scan variants share an independent CPU oracle for
stable indices, counts and exclusive ranks. Seven flag patterns and nine valid
lengths (0/1/17/31/32/33/47/63/64) give 126 A5 checks. Cases include empty/full
selection, bit 31, independent warps, negative flags and untouched output tails.

The compaction and row-update examples compile only when invoked without
`--device`, so the CPU example sweep can validate them without hardware.
Pass `--device` with an allocated A5 device to execute and check the outputs;
the dedicated onboard acceptance below provides numerical coverage.

Prepare without a card, then replay under the local device queue:

```bash
python tests/st/runtime/vpto/test_simt_collectives_onboard.py \
  --compaction-worker build/simt-compaction
task-submit --device auto --max-time 600 --run \
  'python tests/st/runtime/vpto/test_simt_collectives_onboard.py --compaction-worker build/simt-compaction "$TASK_DEVICE"'
```

Both commands use the configured project environment. Set
`PYPTO_SIMT_COMPACTION_BENCH_ROUNDS=20` in the queued job to record three warmups
and 20 measured launches per method in `result.json`. Preparation records the
artifact directories in `artifacts.json`; rerun preparation after changing the
example. The replay uses `CompiledProgram.from_dir` with those artifacts.

On A5 with PTOAS v0.72 and CANN 9.3.0, the 2026-11-16 sample passed all 126
checks. Median device wall times were 63.03 µs (ballot) and 63.47 µs (scan), with
ranges 57.50–68.43 and 58.25–66.66 µs. These overlapping ranges establish no
performance advantage. Device wall includes orchestration, launch and GM work;
it neither isolates instruction latency nor compares against Tile/SIMD. The
mixed row-update acceptance below supplies the separate Tile/SIMD comparison.

## Mixed row-update acceptance

[`06_simt_row_update.py`](../../../../examples/advanced/06_simt_row_update.py)
compares Tile → SIMT → Tile with Tile broadcast multiply → masked SIMD → Tile.
Both process `[8, 64]` FP32 data and eight row coefficients, return the updated
Tile after a vector consumer, and preserve an original-Tile snapshot. The
coefficient row is 32-byte aligned. There are no manual synchronization calls.

`test_simt_integration` in `tests/st/runtime/vpto/test_simt_ir_onboard.py`
checks seven valid column counts (0/1/31/32/33/63/64) and three changing input
sets for each row variant. It also replays the existing GM-copy and UB-gather
kernels for seven lengths and three inputs each: **84 device checks** in all.
The oracle checks outputs, untouched tails, snapshots and unchanged row inputs.
GM copy retains nonzero-offset guards and a downstream Tile load; UB gather
retains two-dimensional addressing, a vector producer and an old-Tile read.

```bash
python tests/st/runtime/vpto/test_simt_ir_onboard.py \
  --integration-worker build/simt-integration
task-submit --device auto --max-time 600 --run \
  'PYPTO_SIMT_INTEGRATION_BENCH_ROUNDS=30 python tests/st/runtime/vpto/test_simt_ir_onboard.py --integration-worker build/simt-integration "$TASK_DEVICE"'
```

Run both commands in the configured project environment. Preparation builds
and loads all four programs without reserving a card; replay reconstructs them
from `integration-artifacts.json`. Rebuild after changing the example.
`integration-result.json` retains check counts and all timing samples.

The 2026-11-16 A5 sample (PTOAS v0.72, CANN 9.3.0) passed all 84 checks. With
three warmups and 30 measured launches per variant and length:

| Valid columns | Tile/SIMT median (µs) | Tile/SIMD median (µs) |
| ------------- | --------------------: | --------------------: |
| 0 | 61.58 | 62.73 |
| 33 | 62.95 | 61.55 |
| 64 | 61.13 | 62.32 |

The sample ranges overlap (SIMT 57.03–69.95 µs, SIMD 56.70–85.18 µs), so this
small regular workload establishes no speed advantage. Each invocation has one
runtime task and one SIMT/VF helper launch. Generated callers perform three GM
Tile loads and two stores: **8,224 logical bytes** (two 2,048-byte inputs, a
32-byte coefficient row, and two 2,048-byte outputs), plus a 2,048-byte UB seed
copy to preserve the original version. The SIMD path additionally materializes
the broadcast-multiply candidate. Logical byte counts are not measured bus
traffic; cache lines, alignment and UB accesses add further costs.

Even `valid=0` retains the Tile producers, consumer, snapshot and helper launch.
Its timing is a combined overhead control, not an isolated launch measurement.
Device wall includes orchestration and scheduling; subtracting these noisy
samples cannot separate launch, memory and arithmetic latency. Larger workloads
or in-core profiling are needed to attribute those costs independently.

## Scalar math and conversion validation

The explicit scalar APIs lower to native `pto.absf/minf/maxf`, math operations,
`pto.prmt`, `pto.mulhi` and `pto.ftoi/itof`. Conversion emits explicit signedness
and `round(r/a/f/c/z)`; `ftoi` always emits `sat`. The initial matrix is FP32
math and INT32 bit carriers. No toolchain version change is required beyond v0.72.

`tests/st/runtime/vpto/test_simt_math_onboard.py` exercises all 18 APIs through
public JIT, including both signedness modes, five rounding modes, saturation,
NaN/Inf, negative division/modulo, fused arithmetic and partial active blocks.
Its build/persistent variants use the same gates and queue requirements above.
Native LLVM export is covered by `test_public_math_native_compile` in the JIT
unit tests. Compilation and device execution are separate validation stages.

The LLVM-text device build removes unsupported LLVM memory-effect annotations,
matching PTOAS object emission, before invoking CANN Bisheng. It preserves
instructions, synchronization and other attributes.
Groups left empty by this cleanup and their references are removed together,
since LLVM rejects empty attribute groups.

## Pitched GM windows and launch-boundary acceptance

The public [strided-copy example](../../../../examples/advanced/07_simt_strided_copy.py)
passes three `[8, 256]` windows to an AIV helper: FP32 input with row pitch 288,
FP32 output with pitch 320, and INT32 thread tags with pitch 352. All have
nonzero row and column origins. `OptimizeOrchTensors` propagates these physical
strides to read and writable AIV parameters; SIMT specialization then retains
that metadata. Previously, the helper used logical pitch 256 for all three,
which addressed the wrong rows after the first row.

The acceptance case uses `thread_idx` and `block_dim` on all three axes, lane ID,
and a thread-strided loop. It covers `(n, 1, 1)` for
`n = 1, 256, 257, 512, 513, 1024, 1025, 2048`, plus `(8, 4, 8)`. Each variant
checks `valid = 0, 1, 255, 256, 257, 2047, 2048` with two input changes:
**126 exact device comparisons**. Checks include both complete output parents,
all inactive elements and surrounding guards, and an unchanged source.
Compilation also verifies each linked SIMT launch length.

The 2026-11-16 A5 run passed all 126 checks with PTOAS v0.74, CANN 9.3.0,
and pinned runtime `27fe21756a5230cc4cfe7e44fa360d07d782a111`. Prepare artifacts
without reserving a device, then execute them on a queued device:

```bash
python tests/st/runtime/vpto/test_simt_ir_onboard.py \
  --strided-worker build/simt-strided

task-submit --device auto --max-time 600 --run \
  'python tests/st/runtime/vpto/test_simt_ir_onboard.py --strided-worker build/simt-strided "$TASK_DEVICE"'
```

`strided-artifacts.json` records the prepared artifact directories;
`strided-result.json` records the per-variant counts, device, and runtime pin.
The opt-in pytest case is `test_simt_strided`, with the same build/persistent
gates as the earlier acceptance cases.

Generated AIV kernels require static physical strides for sliced arguments,
consistent across calls to each parameter. If pitches vary, pass the full
parent and use parent-relative indices. An external C++ AIV parameter with
static inner extents and no explicit tensor view can use runtime descriptor
strides. With dynamic inner extents, runtime strides require packed ND storage
on all incoming paths. Explicit parameter views enforce their storage contract.
Reshapes and shape-bearing ND views require packed storage on every incoming path.
Nested slices, aliases, tuple projections, dropped axes, and manual `Submit`
calls have pass-level regression coverage. Packed short vectors remain outside
this executable subset; this acceptance does not add packed storage or APIs.

## Interface coverage

See the [SIMT interface coverage matrix](06-simt-interface-coverage.md) for
first-batch additions, evidence, and remaining task 11.7 limitations.
