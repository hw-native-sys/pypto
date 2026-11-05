# SIMT lowering and execution

P3 lowers explicit `FunctionType.SIMT` helpers and `SimtLaunchStmt` through the
production A5 compiler and persistent runtime. The input is the
[canonical SIMT IR](../ir/11-simt.md); a higher-level `simt_scope`/JIT authoring
surface and automatic outlining are separate work.

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

`tests/st/runtime/vpto/test_simt_ir_onboard.py` compiles canonical IR directly,
without substituting generated sources. This compiler-facing test uses the IR
text form because the SIMT JIT authoring surface is not yet available. This is
an approved P3-specific exception to the ST JIT authoring policy, limited to this
compiler acceptance test. It validates canonical IR execution, not public JIT
support, and must migrate when SIMT JIT authoring/outlining is implemented.

- GM copy covers Tensor subviews with nonzero origins, guards, inactive elements,
  and a downstream Tile load consuming the SIMT result.
- UB gather covers two-dimensional threads, INT32 indices, a vector producer,
  partial writes, and a later read of the original Tile.
- Both check cold compilation, warm cache loading, and directory relocation.
  Seven lengths and three input variants at each stage give 126 runtime cases.

These cases passed on Ascend950DT with PTOAS v0.72, CANN 9.3.0 and pinned runtime
`31f9e72ee6832cd198b77df18c1c305e7fd83676`. Build all artifacts in the current
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
