# A5 RegScope register spill policy

## Purpose

A5 `reg_scope` helpers must fail clearly when Bisheng needs VF stack storage for
register spills. The default policy is to pass
`--cce-disable-vf-stack-reserved-ubuf` to the device Bisheng invocation. This
sets the VF stack size to zero; a kernel that needs spill storage then fails at
compile time. A caller may explicitly opt into the existing behavior when a
spill is acceptable.

This policy applies only to A5 VPTO helpers produced from `reg_scope`. It does
not change A2/A3 compilation, non-RegScope kernels, or independent SIMT
helpers.

## Reproduced compiler behavior

The archived reproducer in
`build/register-spill-case-20261116` lowers `mask12.pto` with PTOAS v0.74 and
`--vpto-scheduler=off`, then invokes Bisheng with:

```text
--cce-long-scbz=true
-mllvm -cce-dyn-kernel-stack-size=true
-mllvm --cce-res-usage
```

Without the guard, Bisheng succeeds and reports:

```text
[BISHENG] SIMD VF Function properties for mask_spill_mix_aiv.vector.thread:
Total stack argument size: 0 bytes, Stack size: 672 bytes,
VReg number: 13, MaskReg(PReg) number: 7
```

The generated object contains `PSTI`/`PLDI` predicate spill and reload
instructions. With `--cce-disable-vf-stack-reserved-ubuf`, the same `.ll`
fails with driver exit code 1 (the internal frontend reports exit code 70):

```text
fatal error: error in backend: The total stack object size (704) exceeded vf stack size (0). The spilled objects are:
arguments size:  0
Aux-Scalars:     0
Predicates:      0
Vector Slots:    0
Total Spilled Byte Size: 704
bisheng: error: clang frontend command failed with exit code 70
```

The control kernel has zero stack size and succeeds with and without the guard.
The reproducer demonstrates that source mask count alone is not a pressure
metric: the failing case carries predicates through loop PHIs and combines them
with `pand`, which creates long live ranges.

## API and propagation

Expose `allow_reg_spill: bool = False` on the compile-side configuration. When
the target is A5 and a generated unit contains a `vf` helper from `reg_scope`,
append the Bisheng driver option only when `allow_reg_spill` is false:

```python
RunConfig(allow_reg_spill=False)  # default
CompileOptions(allow_reg_spill=True)  # explicit opt-out
```

The backend stamps each generated VF helper with a source comment,
`// pypto.reg_spill: deny` or `// pypto.reg_spill: allow`. SIMT helpers have no
such marker. Runtime helper compilation reads the marker and appends the guard
only on A5 for `deny`. This preserves policy through artifact relocation and
replay without requiring a runtime-only setting. Existing unmarked external
helper sources retain their original behavior. The helper manifest hashes the
complete stamped source.

The effective policy and flags must participate in the native build cache key.
The same flags must be used by fresh builds, replay, and debug rebuilds. A
failed guarded compile must not silently retry with spilling enabled.

## Diagnostic contract

Classify an error as `RegisterSpillError` only when the Bisheng output contains
the VF-stack guard phrase (`exceeded vf stack size (0)`), allowing whitespace
differences, and the guard is enabled. Preserve all other Bisheng failures as
ordinary compiler errors.

The user-facing summary should include the kernel, helper, source location when
available, compiler stage, and recovery action:

```text
RegisterSpillError [PYPTO-REG-SPILL]
A5 reg_scope compilation requires register spilling, but register spilling is
disabled by default.

Helper: <artifact>/vpto/<unit>/<helper>.pto
Stage: Bisheng device compilation

Bisheng diagnostic:
fatal error: error in backend: The total stack object size (704) exceeded vf stack size (0). The spilled objects are:

Full compiler diagnostics: <artifact>/vpto/<unit>/report/<helper>-spill.txt

Reduce simultaneously live vector/predicate values or split the reg_scope.
To allow spilling, set allow_reg_spill=True in RunConfig or CompileOptions.
```

The full command and complete compiler error (including any emitted version,
spill byte count and resource diagnostics) are saved beside the helper in
`report/<helper>-spill.txt`; the LLVM input is saved as `<helper>-spill.ll`.
A report write failure retains the full diagnostic in the exception.
The `Predicates: 0` field is preserved verbatim: its classification semantics
have not been established, and it must not be interpreted as a spill count. If no source span is present,
show the helper and generated `.pto` path without inventing a Python line.

## Verification scope

Regression coverage checks boolean validation, compile option propagation, JIT
and native cache separation, VF/SIMT source stamping, manifest hashes, guard
command construction, spill classification, persistent reports, and ordinary
compiler failures. Compile-only acceptance uses the archived loop-carried mask
case; no NPU execution is needed. Merely counting source masks does not prove
physical register interference.

The compile-only acceptance fixtures are
`tests/ut/backend/vpto_vf_validation/{mask_spill,no_spill}.pto`. To run the
real toolchain checks with PTOAS on PATH (or `PTOAS_ROOT` configured), set
`PYPTO_SPILL_TEST_BISHENG` to the Bisheng executable and run
`pytest tests/ut/runtime/test_kernel_compiler.py -k reg_spill_real_bisheng`.
These fixtures use a null UB base and must only be compiled, never executed.
