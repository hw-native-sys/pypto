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
PTOAS v0.74 and an A5 CANN/Bisheng toolchain. Keep runtime pin checking enabled.
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

## Packed-value compiler capability gate

`tests/ut/backend/vpto_vf_validation/test_simt_packed.py` establishes the native
compiler boundary for task 11.6 before adding a public packed-value API. These
are compile-only fixtures: all lanes use the same address, and the UB entry
arguments are not a runnable host ABI. Do not execute these kernels on a device.

The matrix targets PTOAS v0.74, A5 (`dav-c310-vec`), and CANN 9.3.0 Bisheng
(Clang 15.0.5). It runs PTOAS LLVM export, PyPTO's production
`prepare_llvm_for_bisheng` compatibility step, and compilation with production
`DEVICE_FLAGS`. Raw and prepared LLVM are both retained. In particular,
LLVM 19 `memory(...)` annotations require the existing compatibility step.

| Operation | FP16×2 | BF16×2 | FP32×2 |
| --------- | ------ | ------ | ------ |
| GM and UB load/store, add, multiply | Compiles | Compiles | Compiles |
| `pto.absf` | Compiles | Compiles | Compiles |
| `pto.sqrt` | Compiles | PTOAS rejects type | PTOAS rejects type |
| `shuffle_idx/up/down/bfly` | Compiles | PTOAS rejects type | PTOAS rejects type |
| GM/UB atomic add/min/max/exch/cas, discard old value | Compiles | Compiles | Not probed |
| Same atomics, consume old value | PTOAS rejects use | PTOAS rejects use | Not probed |
| `vector.extract` + `vector.from_elements` | Dialect unavailable | Dialect unavailable | Dialect unavailable |

Conversions FP16×2 ↔ FP32×2 and BF16×2 ↔ FP32×2 compile with each native
rounding selector `r`, `z`, `a`, and `f`. This only establishes compilation;
numerical rounding behavior still needs device acceptance. Four-element
FP16/BF16/FP32 copies are rejected by GM `pto.ldg`, but compile with UB
`pto.load/store`. That difference does not establish four-element arithmetic
support or a single packed hardware instruction.

The 104 cases require 65 nonempty device objects with retained SIMT entries and
39 specific PTOAS rejections. A crash, timeout, wrong diagnostic, or newly
accepted negative case fails the gate. Positive cases also check packed-element
addressing in LLVM: offset 1 steps over a whole vector. UB loads carry natural
alignment (4 bytes for half pairs, 8 for FP32 pairs; 8/16 for four-element
vectors). These are emitted compiler assumptions, not runtime alignment checks.

To retain the full matrix and toolchain versions in a fresh evidence directory:

```bash
source .claude/skills/testing/load-env.sh
python tests/ut/backend/vpto_vf_validation/test_simt_packed.py \
  --probe build/simt-packed --jobs "$PYPTO_TEST_JOBS"
```

Alternatively, run the parametrized gate through pytest:

```bash
PYPTO_RUN_VPTO_VF_VALIDATION=1 python -m pytest \
  tests/ut/backend/vpto_vf_validation/test_simt_packed.py \
  -n "$PYPTO_TEST_JOBS" -v
```

Configure `PTOAS_ROOT` and `ASCEND_HOME_PATH` first. Ordinary unit-test runs skip
these native cases. This milestone does not add a PyPTO packed type or operations,
validate linking/launch sizes, or establish hardware correctness or performance.
Construction/extraction needs a supported lowering before exposing the public
API. Alignment validation, odd tails, subview offsets, numerical conversions,
atomic contention, and comparison with scalar code remain separate acceptance
work. Packed values must remain per-thread values, distinct from Tile storage
and the VMI `RegType`.
