# SIMT interface coverage

This matrix tracks task 11.7 against pinned PTOAS v0.74
(`be4c7a51f66ceadda6eb8a001d7bddccb3507700`). An interface declaration is
not evidence of device execution. General unsigned types are distinct from
INT32 values interpreted as unsigned by one operation.

## First batch: 32-bit interfaces

All APIs below are available in `pl.simt` and `pypto.ir.op.simt`.
They use registered SimtOp schemas; the existing generic binding exposes them
without adding C++ node classes or native binding signatures.

| PTOAS operation / parameter | PyPTO API | Types / storage | Status |
| --------------------------- | --------- | --------------- | ------ |
| `atomic_sub` | `atomic_sub` | INT32/FP32, GM/UB | Implemented; device acceptance pending |
| `atomic_and/or/xor` | Same names | INT32, GM/UB | Implemented; FP32 rejected |
| `atomic_min/max unsigned` | `atomic_min/max(..., unsigned=True)` | INT32 carrier, GM/UB | Implemented; FP32 unsigned rejected |
| `redux_mini/maxi unsigned` | `redux_min/max(..., unsigned=True)` | INT32 carrier | Implemented; FP32 unsigned rejected |
| Four `shuffle_*` operations, `bound` | `shuffle_*(..., width=32, bound=None)` | INT32/FP32, widths 16/32 | Implemented; default window unchanged |
| `get_lanemask_eq/le/lt/ge/gt` | `lanemask_eq/le/lt/ge/gt()` | INT32 carrier | Implemented; physical lane relations |
| `bitcast` | `bitcast(value, dtype)` | FP32/INT32, including identity | Implemented; preserves representation |

Atomics use the existing ReadWrite argument effects and Simt write channel.
They return the old value, retain discarded updates, preserve cache validation,
and participate in launch storage versioning. The source-window rules and
inactive-source restriction are in the [IR contract](../ir/11-simt.md).

Validation on 2026-11-19: a fresh worktree C++ build passed, followed by
**1,124 passing tests** across SIMT operators, atomic regressions, codegen,
public JIT, outlining and storage preparation. PTOAS v0.74 was required,
including LLVM export of both first-batch entries and all four shuffle modes
with INT32/FP32 payloads, widths 16/32 and both boundary bounds. Python lint,
implementation type checks and documentation parity/navigation checks passed.
No device execution is claimed.

Evidence is collected by:

- `tests/ut/ir/operators/test_simt_ops.py`: schemas, dtype/attribute errors,
  serialization, wrapper parity and atomic access effects.
- `tests/ut/jit/test_simt_scope.py::test_first_batch_simt_interfaces_native_compile`:
  complete public JIT lowering and pinned PTOAS LLVM export, including GM/UB,
  unsigned comparisons, window attributes and retained unused atomic results.
- `tests/st/runtime/vpto/test_simt_interface_gaps_onboard.py`: opt-in numerical
  fixture for unsigned high-bit boundaries, old values, bit patterns and windows.
  A successful compile does not count as a successful device run.

The current host's NPU probe fails with DCMI error -8005. Device evidence for
these additions remains pending, including atomic contention, divergent physical
lane masks, FP32 subtraction special values and all shuffle window combinations.

## Second batch: scalar types and explicit operations

The scalar storage ABI now accepts FP16/BF16/FP32 and signed/unsigned integers
of 8, 16, 32 and 64 bits. This does not make every instruction valid for every
type. Unsigned operands use bit-preserving signless carriers at native builtin
boundaries, with unsigned interpretation selected explicitly.

| Family | Supported non-packed types | Restriction |
| ------ | -------------------------- | ----------- |
| GM/UB load/store and scalar captures | FP16/BF16/FP32; signed/unsigned 8/16/32/64-bit integers | BOOL/INDEX are computation-only; no FP8 storage |
| `bitcast` | Same numeric subset | Source and destination must have equal bit width |
| `shuffle_*` | FP16/FP32; signed/unsigned 32/64-bit integers | No BF16; lane control remains INT32 |
| `redux_*` | FP16/FP32; signed/unsigned 32-bit integers | No BF16 or 64-bit reduction |
| Atomic add/sub/min/max/exch | FP16/BF16/FP32; signed/unsigned 32/64-bit integers | 64-bit atomics only in GM |
| Atomic and/or/xor | Signed/unsigned 32/64-bit integers | 64-bit atomics only in GM |
| Atomic CAS | FP32; signed/unsigned 32/64-bit integers | No FP16/BF16; 64-bit only in GM |
| `ftof` | Distinct pairs among FP16/BF16/FP32 | r/a/f/c/z; round-to-odd only FP32→FP16; optional saturation |
| `ftoi` | FP16/BF16/FP32→signed/unsigned INT32; FP32→signed/unsigned INT64 | r/a/f/c/z; mandatory saturation |
| `itof` | Signed/unsigned INT32→FP16/BF16/FP32; signed/unsigned INT64→FP32 | r/a/f/c/z |
| Existing explicit scalar math | FP32; `prmt`/`mulhi` use INT32 carriers | No blanket half-precision math expansion |
| `mul_i32toi64` | INT32 operands, INT64 result | Optional unsigned interpretation of INT32 carriers |
| `div`, `rem` | FP32 | Explicit precision and truncating remainder; ordinary `/` and `%` unchanged |
| `veccore_id` | INT32 result | Target vector core ID |
| `trap`, `assert_` | Void; assert requires BOOL | Runtime-dependent diagnostic, not a safe-access guard |

Extended tests live in `tests/ut/codegen/test_simt_codegen.py`, including native
LLVM export for storage, atomics/collectives and conversion pairs. The public JIT
acceptance fixture also covers FP16/BF16/INT64/UINT32 ABI, unsigned high bits,
explicit division/remainder and widened multiplication. Hardware
numerical acceptance remains pending; successful compilation is not numerical
validation. The worktree C++ build and focused SIMT regressions pass, including
the public JIT fixture and required PTOAS v0.74 native compilation. The device
probe still reports DCMI -8005, including outside the sandbox.

FP32 `rem` uses integer significand reduction because the A5 Bisheng backend
cannot select the LLVM `frem` emitted by `pto.remf`. This preserves subnormals
and the dividend sign, including zero, without a rounded floating-point quotient.
A zero divisor, infinite dividend or NaN operand returns NaN; finite/infinity
returns the dividend. NaN payloads are not specified.

The JIT `test_simt_interfaces_device_compile` regression compiles both the extended
fixture and a dynamic-divisor remainder kernel through the production PTOAS and
Bisheng helper path. Enable it with `PYPTO_RUN_VPTO_VF_VALIDATION=1`; no NPU is
needed. A host LLVM regression checks the emitted arithmetic against `fmod`, while
the onboard fixture checks bitwise results across signs and exponent boundaries.
The onboard remainder oracle uses exact IEEE-754 bit patterns, independent of
the host CPU's flush-to-zero mode. Positional and keyword `ftof` value operands
use typed expression parsing, including numeric attributes such as `math.inf`;
the destination dtype retains dtype resolution.
Host execution and device compilation do not substitute for onboard acceptance.

## Remaining scope

| Family | Current coverage | Remaining work / limitation |
| ------ | ---------------- | --------------------------- |
| Coordinates and extents, lane ID | Existing DSL/IR/native lowering | No duplicate implementation needed |
| `get_clock32/64` | Not exposed | Pinned PTOAS treats reads as pure and merges distinct samples; see below |
| Runtime trap/assert | DSL/IR/native lowering implemented | Device termination and host error propagation unverified |
| `keep/resume` | Not exposed | Requires paired launch representation, slot verification and per-thread lifetime enforcement |
| `addui_extended`, `mul_extended` | No direct tuple-result API | Existing arithmetic/bit operations and `mulhi` express portions; do not claim full type coverage |
| `ceildiv`, `select` | Can be expressed using arithmetic/control flow for supported scalars | Direct API absence is not itself missing computation |
| Packed values and short vectors | Deferred task 11.6 | Not a prerequisite for 11.7 |

## Toolchain boundary

PTOAS stays pinned to v0.74. A v0.77 upgrade needs both typed memory offsets
and regressions for vote/shuffle/redux active-lane effects under divergent control
flow; changing the version number alone is insufficient.

### Clock sampling blocker

PTOAS v0.74 declares `get_clock32/64` pure. A native probe with two `get_clock32`
reads, their subtraction and an observable GM store produces a literal zero
store in exported LLVM, with no clock read left. PyPTO retaining a call cannot
repair this downstream common-subexpression elimination. Clock APIs therefore
remain unexposed until the native effect contract and optimization regressions
are fixed; INT64 storage alone does not resolve this. Minimal entry body:

```mlir
%a = pto.get_clock32 : i32
%b = pto.get_clock32 : i32
%elapsed = arith.subi %b, %a : i32
%zero = arith.constant 0 : index
pto.stg %elapsed, %dst[%zero] : !pto.ptr<i32, gm>, i32
```

Compile inside a `pto.simt_entry` function with `--pto-arch=a5
--pto-backend=vpto --emit-vpto-llvm-ir`; the resulting store contains zero.

### Cross-entry state blocker

Native `keep/resume` compile individually, but their contract requires adjacent
SIMT entries in the same outer launch sequence. Executable PyPTO bundles emit a
separate module and outer wrapper for each helper. They currently carry memory
effects, not a typed per-thread state dependency between helper artifacts.
Exposing the instructions alone would leave launch pairing and state lifetime
unverified. A subsequent change must represent the pair in one launch sequence,
check matching dimensions and slot types, reject overlapping slots (64-bit
values need two words at an even slot), and enforce resume prologues / keep
epilogues. This state dependency must not be modeled as an ordinary memory fence.
