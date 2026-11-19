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

## Remaining scope

| Family | Current coverage | Remaining work / limitation |
| ------ | ---------------- | --------------------------- |
| Coordinates and extents, lane ID | Existing DSL/IR/native lowering | No duplicate implementation needed |
| `get_veccoreid`, `get_clock32/64` | Missing | Clock sampling must survive optimization; clock64 requires INT64 |
| FP32 arithmetic/math | Existing arithmetic, math and explicit conversions | Explicit high-precision `divf`, truncating `remf` |
| Non-packed FP16/BF16, INT64, unsigned types | Not generally executable | Per-operation storage/compute/collective/atomic matrix; UB i64 atomics remain forbidden |
| `ftof`, extra `ftoi/itof` pairs | FP32↔INT32 conversions already exist | Validate rounding/saturation for every new pair |
| `mul_i32toi64` | Missing | INT64 result and signed/unsigned operand interpretation |
| Runtime trap/assert | Missing | Target/runtime termination and host error reporting |
| `keep/resume` | Missing | Slots, prologue/epilogue, adjacent launches and per-thread state lifetime |
| `addui_extended`, `mul_extended` | No direct tuple-result API | Existing arithmetic/bit operations and `mulhi` express portions; do not claim full type coverage |
| `ceildiv`, `select` | Can be expressed using arithmetic/control flow for supported scalars | Direct API absence is not itself missing computation |
| Packed values and short vectors | Deferred task 11.6 | Not a prerequisite for 11.7 |

## Toolchain boundary

PTOAS stays pinned to v0.74. A v0.77 upgrade needs both typed memory offsets
and regressions for vote/shuffle/redux active-lane effects under divergent control
flow; changing the version number alone is insufficient.
