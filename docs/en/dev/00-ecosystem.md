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
- **Orchestration functions** (task scheduling) → C++ using PTO2 runtime API → compiled for AICPU

## Component Details

### pypto — Compiler Framework

The core compiler. Takes Python tensor programs and compiles them into device-executable code.

**Inputs:** Python programs written with `pypto.language` DSL (`@pl.program`, `@pl.function`)

**Outputs:**

- `.pto` files — PTO-ISA MLIR dialect, one per InCore kernel function (runs on AICore)
- Orchestration C++ — task scheduling code using PTO2 runtime API (runs on AICPU)

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

### simpler — Task Runtime

Executes compiled programs on Ascend hardware. Manages the three-program execution model: Host, AICPU kernel, and AICore kernel.

**Inputs:**

- Compiled AICore kernel binaries (InCore path: pypto → PTOAS → pto-isa → device compiler)
- Compiled AICPU orchestration binary (Orchestration path: pypto → C++ with PTO2 runtime API → device compiler)

**Responsibilities:**

- Build and execute task dependency graphs
- Coordinate Host ↔ AICPU ↔ AICore execution
- Handle device memory, synchronization, and handshake protocols

**Interface with pypto:** The orchestration C++ code that pypto generates uses the PTO2 runtime API (`rt_submit_task`, `make_tensor_external`, etc.), which simpler implements. The runtime API is the contract between pypto's orchestration codegen and simpler.

## Interface Summary

Each repo boundary has a well-defined interface:

```text
pypto-lib ──[ Python API: pypto.language / pypto.ir ]──► pypto
     pypto ──[ .pto files: PTO-ISA MLIR dialect     ]──► PTOAS
   pto-isa ──[ C++ #include: tile instruction hdrs  ]──► PTOAS
   pto-isa ──[ C++ #include: ISA headers            ]──► simpler
     pypto ──[ C++ API: PTO2 runtime API calls      ]──► simpler
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
| New runtime feature | simpler + pypto | PTO2 runtime API, orchestration codegen |
| New PTO MLIR op | PTOAS + pypto | PTO MLIR dialect, pypto PTO codegen |
| New model example | pypto-lib only | None (consumer of existing APIs) |
