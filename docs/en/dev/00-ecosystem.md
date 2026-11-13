# PTO Project Ecosystem

## Overview

The PTO (Parallel Tensor/Tile Operation) project is a multi-repo toolchain for programming AI accelerators. It spans the full stack from Python-level tensor programs down to hardware instruction execution.

This document describes **what each repo does**, **how they connect**, and **where the boundaries lie**.

## Repositories

All repositories live under [github.com/hw-native-sys](https://github.com/hw-native-sys).

| Repo | Role | URL |
| ---- | ---- | --- |
| **pypto** | Compiler framework | [hw-native-sys/pypto](https://github.com/hw-native-sys/pypto) |
| **pypto-lib** | Model zoo & real-world cases | [hw-native-sys/pypto-lib](https://github.com/hw-native-sys/pypto-lib) |
| **PTOAS** | PTO assembler & optimizer | [hw-native-sys/PTOAS](https://github.com/hw-native-sys/PTOAS) |
| **pto-isa** | Instruction set architecture | [hw-native-sys/pto-isa](https://github.com/hw-native-sys/pto-isa) |
| **simpler** | Task runtime | [hw-native-sys/simpler](https://github.com/hw-native-sys/simpler) |

## Compilation Pipeline

```text
                    ┌─────────────────────────────────────────────┐
                    │              pypto-lib                      │
                    │  Real-world models & primitive tensor funcs │
                    │  (uses pypto as its compiler framework)     │
                    └──────────────────┬──────────────────────────┘
                                       │ imports & compiles via
                    ┌──────────────────▼──────────────────────────┐
                    │                pypto                        │
                    │  Python DSL → IR → Passes → CodeGen         │
                    │                                             │
                    │  Produces:                                  │
                    │   • .pto files (InCore kernel → AICore)     │
                    │   • Orchestration C++ (scheduling → AICPU)  │
                    └───┬─────────────────────────────────────┬───┘
          .pto files    │                                     │  orchestration C++
        (InCore only)   │                                     │  (runs on AICPU)
                    ┌───▼────────────────────┐                │
                    │       PTOAS            │                │
                    │  Assembler & Optimizer │                │
                    │                        │                │
                    │  .pto MLIR → C++       │                │
                    │  (uses pto-isa hdrs)   │                │
                    └───┬────────────────────┘                │
                        │ kernel C++                          │
                        │  (includes pto-isa)                 │
                    ┌───▼────────────────────┐                │
                    │       pto-isa          │                │
                    │  ISA definition:       │                │
                    │  tile instruction hdrs │                │
                    └───┬────────────────────┘                │
                        │ compiled AICore binaries            │
                    ┌───▼─────────────────────────────────────▼───┐
                    │              simpler                        │
                    │  Runtime: task graph execution on device    │
                    │  Host ↔ AICPU ↔ AICore coordination         │
                    └─────────────────────────────────────────────┘
```

**Two codegen paths from pypto:**

- **InCore functions** (tile-level compute) → `.pto` → PTOAS → pto-isa → AICore binaries
- **Orchestration functions** (task scheduling) → C++ using simpler runtime API → compiled for AICPU

## Component Details

### pypto — Compiler Framework

The core compiler. Takes Python tensor programs and compiles them into device-executable code.

**Inputs:** Python programs written with `pypto.language` DSL (`@pl.program`, `@pl.function`)

**Outputs:**

- `.pto` files — PTO-ISA MLIR dialect, one per InCore kernel function (runs on AICore)
- Orchestration C++ — task scheduling code using simpler runtime API (runs on AICPU)

**Internal pipeline:**

```text
Python DSL → IR (immutable tree) → Pass Pipeline (20+ passes) → CodeGen
```

- **IR layer**: Multi-level representation — Tensor ops, Tile ops, and system ops coexist in the same IR
- **Pass pipeline**: Progressively lowers tensor-level IR to tile-level IR (unrolling, SSA conversion, tiling, memory allocation, etc.)
- **CodeGen**: Two backends — PTO codegen (InCore → `.pto` MLIR for AICore) and Orchestration codegen (→ C++ for AICPU)

**Key directories:**

| Path | Contents |
| ---- | -------- |
| `include/pypto/ir/` | C++ IR node definitions |
| `src/ir/transforms/` | Compiler passes |
| `src/codegen/` | PTO and Orchestration code generators |
| `python/pypto/language/` | Python DSL frontend |
| `python/pypto/ir/` | Pass manager, compile API |

### pypto-lib — Model Zoo & Primitives

A library of real-world models and primitive tensor functions built **on top of** pypto. Serves as:

1. **Model zoo** — end-to-end model examples (e.g., DeepSeek, FFN, LLaMA) that exercise the full compilation pipeline
2. **Primitive tensor functions** — reusable tensor-level building blocks (elementwise, reduction, matmul) that the compiler tiles and lowers to PTO-ISA

**Depends on:** pypto (imports `pypto.language`, compiles via `pypto.ir.compile`)

**Interface with pypto:** pypto-lib programs are standard pypto programs — they use the same `@pl.program`/`@pl.function` DSL and compile through the same pipeline. No special API exists between them; pypto-lib is a consumer of the pypto framework.

**Compiler CI coverage:** The `pypto-lib-compile` job compiles only the named
cases in [`tests/ci/pypto_lib_compile.json`](../../../tests/ci/pypto_lib_compile.json).
Coverage focuses on selected standalone operators, split by platform:

- **A5:** 10 entries under `models/deepseek_v4_1_flash/`: `compressor`,
  `sparse_attention`, `gate`, `expert_routed`, `expert_shared`, `hc`, `o_proj`,
  `engram`, `input_pack`, and `lm_head`.
- **A2/A3:** 10 entries under `models/deepseek_v4_flash_lowlat/`:
  `decode_compressor_ratio128`, `decode_compressor_ratio4`, `decode_sparse_attn_swa`,
  `expert_routed`, `gate`, `hc_post`, `hc_pre`, `o_proj_tp`, `qkv_proj_rope`, and
  `rmsnorm`; plus three under `models/qwen3_14b/`: `greedy_sample`,
  `test_paged_attention_pypto`, and `topk_select`. The Qwen attention case uses
  the runnable native PyPTO operator driver.

Layer and full-model entries, composed attention/MoE blocks, MTP, DSpark, vision,
and drafts are outside this list. New library files do not join automatically;
add selected operator entries or parameter variants explicitly as coverage needs
grow. Helpers are covered through their callers. Compilation needs no NPU.
The adapter forces `golden.run` into compile-only mode, records direct compile
calls, and requires PTOAS. Coverage includes frontend, passes, verification,
codegen, and PTOAS.
When an active argument parser declares `--compile-only`, the adapter supplies
that flag to that parser. Other parsers keep their original arguments, preserving
helper/component parsing while disabling post-compilation execution branches.
This does not build device binaries or check numerical results; `pypto-lib-model`
provides the separate device smoke tests.

To add operator coverage on either configured platform, append a case to the
JSON list; no workflow or runner edit is needed. Save the list as UTF-8.
For example, to compile a separate HC parameter variant:

```json
{
  "name": "models-deepseek_v4_1_flash-hc-prefill",
  "entry": "models/deepseek_v4_1_flash/hc.py",
  "platforms": ["a5"],
  "args": ["--batch", "1", "--sequence", "128"]
}
```

- `name`: globally unique case ID, using letters, digits, `_`, `-`, or `.`,
  starting with a letter or digit. `manifest.json` and `results.json` are reserved
  for output artifacts. Logs and build outputs are isolated by name.
- `entry`: exact Python script path relative to pypto-lib, under `examples/` or
  `models/`. Directories, globs, drafts, and paths outside the library are rejected.
- `platforms`: nonempty list of `a2a3` and/or `a5`, without duplicates. The runner
  supplies `-p`; do not put platform arguments in `args`.
- `args`: optional array of strings passed directly as arguments, without shell
  expansion. Omit it to keep the entry's defaults. An explicit `-d`/`--device`
  replaces inferred logical ranks. Compile-only enforcement remains active.
  The adapter enables each parser's declared `--compile-only` option at parse
  time, without injecting the flag into other components that do not accept it.

Give each parameter variant a different name, even when it uses the same entry.
The runner validates the entire configuration before compiling; duplicate names,
unknown fields/platforms, missing files, and an empty platform selection fail the
job. Use `--cases /path/to/cases.json` to select an alternative list locally.

The A2/A3 compile leg runs on a CANN-equipped `npu` host so external wrappers
can resolve SDK headers. It builds the A2/A3 and simulator runtime packages.
The A5 leg runs on a `cpu` host with the simulator runtime package. Neither
compile leg acquires or executes on an NPU.

To reproduce after activating a built PyPTO environment with the pinned PTOAS:

```bash
python tests/ci/compile_pypto_lib.py --lib /path/to/pypto-lib \
  --platform a5 --jobs 4 --output build/lib-compile-a5
```

Repeat with `--platform a2a3 --output build/lib-compile-a2a3` for the A2/A3 list.
Each entry has a 1200-second timeout (override with `--timeout`). Separately, each PTOAS kernel compilation defaults to 60 seconds.
Set `PYPTO_PTOAS_TIMEOUT_SECONDS` to a positive integer to override that deadline;
invalid values raise an error. The library compilation CI job sets it to `180`
for large kernels. Use `export PYPTO_PTOAS_TIMEOUT_SECONDS=180` to match CI locally.
The setting applies to both fresh compilation and PTO rebuilds during debug replay.
This does not change the per-entry `--timeout`.
Compilation failures, process failures, timeouts, and zero recorded
compilations fail the job. Malformed or unreadable receipts fail their entry
without aborting the remaining entries or losing the matrix summary.
When an entry dispatches cases by launching its own script with the current
Python interpreter, each child runs through the same compile-only adapter.
Child environment, working directory, and process isolation are preserved,
and the parent combines their compilation records. The compile-output allowlist
is currently empty, so the adapter does not inject or rewrite runtime directories.
`--runtime-dir` may identify a replay input or belong to a separate checkpoint
mode; merely declaring that option does not enable output-directory injection.
A child failure or missing, empty, or malformed child receipt fails the entry
even if the parent exits successfully. This interception applies to the current entry
script launched by the same Python interpreter.
On compilation failure, CI uploads per-case logs, results, and a manifest
containing the tested library revision, selected entries, named cases, and effective arguments.
Successful runs report results in the job log and summary without uploading artifacts.

### PTOAS — PTO Assembler & Optimizer

An MLIR-based assembler that consumes `.pto` files produced by pypto's codegen and produces optimized C++ kernel code.

**Inputs:** `.pto` files (PTO-ISA MLIR dialect)

**Outputs:** C++ source files that `#include` pto-isa headers

**Responsibilities:**

- Parse PTO-ISA MLIR dialect
- Apply PTO-level optimization passes (sync insertion, memory planning)
- Lower PTO MLIR to C++ code that calls pto-isa tile instructions

**Interface with pypto:** The `.pto` file is the contract. pypto's PTO codegen emits MLIR using the PTO dialect (ops like `pto.tload`, `pto.tmul`, `pto.alloc_tile`, etc.), and PTOAS parses that dialect. The two repos must agree on the PTO MLIR dialect definition.

### pto-isa — Instruction Set Architecture

Defines the tile-level instruction set for the target hardware. Provides C++ headers that declare the hardware tile instructions (load, store, compute, sync, etc.).

**Consumed by:**

- **PTOAS** — the C++ code PTOAS generates calls pto-isa instructions
- **simpler** — clones pto-isa headers at first build for runtime compilation

**Interface:** C++ header library defining the instruction API. Downstream consumers `#include` pto-isa headers; the hardware vendor provides the target-specific implementations that back these headers.

`runtime/pto_isa.pin` is the single source of truth for the revision, and
simpler owns the pin resolver (`simpler_setup.pto_isa.ensure_pto_isa_root`):
it manages one checkout under the runtime's `build/pto-isa`, reuses it only when
that tree is clean and already at the pin, re-clones it otherwise, and verifies
`HEAD` before returning. PyPTO delegates pin resolution to it rather than
resolving the pin itself — a second resolver could hand the kernel compiler an
off-pin tree, and the compiler skips its own revision re-check precisely because
the resolution guaranteed the tree it returned.

While the pinned headers need a temporary backport (currently the A2/A3
TREM/TREMS fix), `pypto.runtime.pto_isa.ensure_pto_isa_root()` derives a
detached worktree of that same verified pin, applies the reviewed patch, and
commits it locally; the patch is removed and the fallback disables itself as
soon as the pin carries the fix. To change the revision, update the runtime-side
pin.

`PTO_ISA_ROOT` is *exported* with the resolved path so downstream consumers that
build extern CCE kernels can find the ISA headers, but it is never *read*: an
ambient value is not the pin. Code that needs the headers should call
`pypto.runtime.pto_isa_include_dir()` instead of reading the variable.

### simpler — Task Runtime

Executes compiled programs on Ascend hardware. Manages the three-program execution model: Host, AICPU kernel, and AICore kernel.

**Inputs:**

- Compiled AICore kernel binaries (InCore path: pypto → PTOAS → pto-isa → device compiler)
- Compiled AICPU orchestration binary (Orchestration path: pypto → C++ with simpler runtime API → device compiler)

**Responsibilities:**

- Build and execute task dependency graphs
- Coordinate Host ↔ AICPU ↔ AICore execution
- Handle device memory, synchronization, and handshake protocols

**Interface with pypto:** The orchestration C++ code that pypto generates uses the simpler runtime API (`rt_submit_task`, `make_tensor_external`, etc.), which simpler implements. The runtime API is the contract between pypto's orchestration codegen and simpler.

#### The runtime pin

`runtime/` *is* the simpler submodule, so PyPTO's pin on it is the gitlink —
`git rev-parse HEAD:runtime`. There is no second pin file to drift, for the same reason
`toolchain/versions.env` declines to restate the pto-isa revision.

The installed side carries no version that tracks it: simpler's `pyproject.toml` says
`0.1.0` and always will. Its only revision identity is `_task_interface.__build_commit__`,
stamped by `runtime/python/bindings/CMakeLists.txt` at build time from `git rev-parse HEAD`.

`pypto.runtime.runtime_pin.check_runtime_pin()` compares the two. It runs at import of the
two PyPTO modules that import simpler at module scope — `pypto.runtime.task_interface` and
`pypto.runtime.kernel_compiler` — and a mismatch raises `RuntimePinMismatch` (an
`ImportError`) naming both revisions and the reinstall command.

Simpler carries its own equivalent guard
(`simpler.task_interface._assert_bindings_match_source_tree`), but that one compares the
extension against *its own* source tree and returns early when there is no `.git` beside
it. A wheel, or a `pip install ./runtime` that copies into `site-packages`, is therefore
never checked — and that install is where staleness is hardest to see, because a changed
struct layout makes fields read as 0 with no error.

Two comparisons, two severities:

| Comparison | Meaning | Result |
| ---------- | ------- | ------ |
| `__build_commit__` vs `git -C runtime rev-parse HEAD` | the installed simpler was not built from this tree | error |
| `git -C runtime rev-parse HEAD` vs `git rev-parse HEAD:runtime` | your runtime checkout is off PyPTO's pin | warning (`RuntimePinWarning`) |

The second is only a warning: working on a runtime branch is legitimate.

Everything else is a **skip**, never a failure — a wheel-installed PyPTO with no
`runtime/`, an uninitialized `runtime/` submodule (empty, so it has no revision of its
own), no git available, or a simpler built without git (empty stamp). Refusing to
run because the question is unanswerable would break every legitimate install. An
importable `_task_interface` with *no* `__build_commit__` attribute is not a skip: the
attribute only disappears on an extension compiled before the stamp existed, which is by
definition a different revision.

Check it by hand with `pypto-runtime-pin`, which prints both sides and exits non-zero on a
mismatch; `pypto-runtime-pin --fix` reinstalls `runtime/` and re-verifies in a fresh
interpreter. `PYPTO_SKIP_RUNTIME_PIN_CHECK=1` bypasses the check. The unit suite sets it in
`tests/ut/conftest.py` because it stubs simpler and must not be gated on the installed
revision; system tests run real kernels and are deliberately not exempted.

## Interface Summary

Each repo boundary has a well-defined interface:

```text
pypto-lib ──[ Python API: pypto.language / pypto.ir ]──► pypto
     pypto ──[ .pto files: PTO-ISA MLIR dialect     ]──► PTOAS
   pto-isa ──[ C++ #include: tile instruction hdrs  ]──► PTOAS
   pto-isa ──[ C++ #include: ISA headers            ]──► simpler
     pypto ──[ C++ API: simpler runtime API calls      ]──► simpler
```

| Border | Format | Who provides | Who consumes |
| ------ | ------ | ------------ | ------------ |
| pypto-lib → pypto | Python imports | pypto-lib | pypto compiler |
| pypto → PTOAS | `.pto` MLIR files | pypto PTO codegen | PTOAS parser |
| pto-isa → PTOAS | C++ `#include` | pto-isa headers | PTOAS codegen |
| pto-isa → simpler | C++ `#include` | pto-isa headers | simpler build |
| pypto → simpler | Orchestration C++ | pypto orchestration codegen | simpler runtime |

## Cross-Repo Development

When a change spans multiple repos, identify which interfaces are affected:

| Change | Repos involved | Interface affected |
| ------ | -------------- | ------------------ |
| New tile instruction | pto-isa + PTOAS + pypto | ISA headers, PTO MLIR dialect, pypto op/codegen |
| New tensor primitive | pypto-lib + pypto | Python DSL (if new ops needed) |
| New runtime feature | simpler + pypto | simpler runtime API, orchestration codegen |
| New PTO MLIR op | PTOAS + pypto | PTO MLIR dialect, pypto PTO codegen |
| New model example | pypto-lib only | None (consumer of existing APIs) |
