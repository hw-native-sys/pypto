# SIMT lowering and execution

The public `simt_scope`/JIT surface outlines into `FunctionType.SIMT` helpers
and `SimtLaunchStmt`, then lowers through the production A5 compiler and
persistent runtime. See the [canonical SIMT IR](../ir/11-simt.md).

## Supported executable subset

- A5 AIV callers, with constant three-dimensional block sizes specialized by
  `SpecializeSimtHelpers` (up to 2,048 threads).
- Borrowed GM Tensor and UB Vec Tile storage with FP32 or INT32 elements.
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
- FP32, INT32, INDEX and BOOL scalar values; structured `if`, `for`, and `while`.
  Signed integer floor division/modulo retain floor semantics. FP32 inequality
  treats NaN as unequal, and numeric-to-BOOL conversion compares against zero.
- Thread/block/grid coordinates and extents, lane ID, and scalar loads/stores.
  GM cache policies pass through unchanged to native PTOAS operations.
- Active-lane votes/ballot, INT32/FP32 shuffle (width 16 or 32), and
  INT32/FP32 add/min/max redux. Integer min/max lower with explicit signed
  comparison. See the [collective contract](../ir/11-simt.md#warp-collectives).

Unsigned arithmetic, other storage dtypes, scalar exponentiation (`**`), floating-point floor division/modulo,
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

LLVM export bypasses PTOAS's object-level SIMT size repair. The supported subset
uses intrinsics; a SIMT module containing inline assembly is rejected before
device compilation. The acceptance tests compare the linked `VF_SIMT` size with
the callee symbol size. No instruction bytes are patched by PyPTO.

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
`pto.threadfence`, and `pto.threadfence_block`. Integer min/max carry `signed`.
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
runtime `391b2fbcd4dd053204a677662c39b6b7636e8043`. Supported atomic locations
are naturally aligned four-byte elements; UINT32/INT64, half/bfloat16, packed
values, selectable ordering/scope and cross-AIV synchronization remain outside
this frontend contract. See [API semantics and limits](../ir/11-simt.md#atomics-and-synchronization).
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
