# Native SIMT runtime validation

The P0 probe validates a native A5 SIMT GM copy through PyPTO's existing VPTO
device compiler and persistent runtime. It is independent of general
[`SimtLaunchStmt` lowering](05-simt-lowering.md), now implemented for the
P3 executable subset.

## What the probe executes

`tests/ut/backend/vpto_vf_validation/simt_gm_copy.pto` exports an AIV function
that launches 128 SIMT threads. Active threads copy FP32 elements directly
between GM pointers; `valid` is an INT32 runtime argument. The exported AIV
function ends with a native `PIPE_ALL` drain, following PTOAS' SIMT GM-access
examples. This establishes the kernel-exit boundary for this isolated probe;
it does not implement automatic synchronization between SIMT and Tile operations.

The test generates Tensor descriptor/scalar unpacking through the production
kernel wrapper generator. Each pointer includes its Tensor subview origin
exactly once. A private generated source artifact substitutes the native copy
for a scheduled copy kernel and attaches it as `vpto_sources`. This is an ABI
fixture, not evidence that PyPTO SIMT IR already lowers to executable code.

The unchanged production path performs:

1. PTOAS VPTO LLVM export and external-symbol checking.
2. Bisheng compilation of the helper and the C++ wrapper to device objects.
3. Device linking with `kernel_entry` first, followed by ELF validation.
4. `.text` extraction, `CoreCallable` registration and persistent A5 execution.

The probe checks GM address spaces, thread/register annotations, direct
`LDG`/`STG` without UB staging, retention of the SIMT entry, and agreement
between the linked callee size and the `VF_SIMT` size operand. LLVM export
bypasses PTOAS' object-emission size patcher. This intrinsic-only fixture must
compile correctly without patching. Arbitrary SIMT bodies, especially inline
assembly that needs size repair, require a separate supported object-emission
contract before they can use this path. P3 rejects inline assembly explicitly.

## Focused acceptance

Build PyPTO and the pinned runtime in the current checkout. Configure the pinned
PTOAS v0.72 and an A5 CANN/Bisheng toolchain. Keep runtime pin checking enabled.
The test uses the same opt-in gates as the [VF tests](03-vpto-vf-integration.md).

The initial hardware acceptance passed on Ascend950DT with PTOAS v0.72,
CANN 9.3.0 and runtime revision `31f9e72ee6832cd198b77df18c1c305e7fd83676`.

Compiler/cache/relocation validation, without opening a device:

```bash
PYPTO_RUN_VPTO_VF_VALIDATION=1 python -m pytest \
  tests/st/runtime/vpto/test_simt_gm_copy_onboard.py --platform=a5 -k build -v
```

After reserving an A5 device through the host's device queue:

```bash
PYPTO_RUN_VPTO_ONBOARD=1 PYPTO_VPTO_DEVICE_ID="$DEVICE_ID" \
PYPTO_VPTO_TIMEOUT=600 python -m pytest \
  tests/st/runtime/vpto/test_simt_gm_copy_onboard.py --platform=a5 -k persistent -v
```

One persistent worker runs cold, warm-cache and relocated artifacts. Each stage
checks `valid = 0, 1, 31, 32, 33, 127, 128` with three input changes, a 32-element
Tensor subview offset, prefix/suffix guards and untouched inactive elements:
63 exact numerical comparisons in total. Warm loading and relocation must not
recompile the helper. The source directory is moved, so stale absolute source
paths cannot make relocation pass accidentally.

`PYPTO_VPTO_RUNTIME_OUTPUT` selects the evidence root (default
`build/simt-runtime`). Every child creates a fresh `simt-*` directory retaining
actual compiler commands, inputs, LLVM, objects, linked ELF and disassembly.
`result.json` distinguishes compiler-only validation from hardware execution;
`build-result.json` records the runtime revision and compilation count.
The pytest parent bounds the whole child process, including device teardown.

This milestone covers FP32 contiguous GM copy only. Strided/dynamic layouts,
UB gather, mixed-pipeline dependencies, multidimensional launches, lane
operations, atomics and general IR lowering need their own acceptance tests.
