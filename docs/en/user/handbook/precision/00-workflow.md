# Precision Localization: Workflow

**Symptom:** your kernel compiles and runs, but the output diverges from a
reference (a torch golden, a known-good build, or hand computation). This page is
the decision tree that routes you to the right localization tool.

## The Core Idea

PyPTO lowers your program through a pass pipeline. A wrong result was introduced
*somewhere* along that path — either in the program you wrote, or in one specific
pass that transformed it. Precision localization is about **bisecting where the
divergence first appears**, then reading only that stage.

Three tools cover the whole path:

| Tool | Answers | Chapter |
| ---- | ------- | ------- |
| Torch golden comparison | "Is the source program wrong, or a later stage?" | [01](01-torch-golden.md) |
| Pass-IR bisection | "Which pass first breaks the result?" | [02](02-pass-ir-bisection.md) |
| Selective tensor dump | "What is this exact tensor on-device?" | [03](03-selective-dump.md) |

**Two codegen paths, two roles here.** *torch codegen* re-expresses the IR as
runnable PyTorch and executes it on the host — a fast numerical **simulation**
that needs no device. **Steps 1 and 2 both run torch codegen** (on the pre-pass
IR and the post-pass IR respectively), so they localize divergences introduced in
your program or by a pass. *pto codegen* is the real device path
(`.pto` → AICore); **step 3 uses pto codegen on-device** to catch divergences that
appear only in the actual codegen / hardware execution, not in torch simulation.

## Decision Tree

```text
Output is wrong
│
├─ 1. TORCH codegen on your program IR, run against golden
│      (torch_codegen(program) → compare)                              → chapter 01
│      │
│      ├─ Already diverges?  → the bug is in your program / op usage.
│      │                        Fix the source; no pass is involved.
│      │
│      └─ Matches golden?    → a pass introduced the divergence. Go to 2.
│
├─ 2. TORCH codegen after PassManager(Default), run against golden
│      (torch_codegen(PassManager…run_passes(program)) → compare)      → chapter 01
│      │
│      ├─ Matches golden?    → divergence is below torch codegen — it
│      │                        appears only in pto codegen / on-device.
│      │                        Go to 3.
│      │
│      └─ Diverges?          → bisect the pass dump to find the first
│                               bad pass (each stage checked via TORCH    → chapter 02
│                               codegen). validate_pass_ir_codegen_results(...)
│                               or compiled.validate_ir(...)
│
└─ 3. PTO codegen on-device — need the real value of one suspect tensor? → chapter 03
       Tag it with pl.dump_tag, enable runtime-DFX selective dump,
       (L2 swimlane double-run for on-board compare), then diff.
```

## Step-by-Step

1. **Establish a golden.** Build inputs and a reference output (e.g. a torch
   function). See [chapter 01](01-torch-golden.md#build-inputs-and-golden-output)
   for a worked `build_tensors` / `golden` pair.

2. **Torch codegen directly on the program IR** and compare
   ([01 › mode 1](01-torch-golden.md#1-codegen-directly-from-program-ir)). If this
   already diverges, the problem is in your source program — stop here and fix it.

3. **Torch codegen after the default pass pipeline** and compare
   ([01 › mode 2](01-torch-golden.md#2-codegen-after-passmanagerdefault)). If mode 1
   was correct but this diverges, a pass is responsible.

4. **Bisect the pass dump** to find the *first* pass whose IR produces the wrong
   result — each dumped stage is checked with torch codegen
   ([chapter 02](02-pass-ir-bisection.md)). With a `CompiledProgram` this is one
   call:

   ```python
   compiled = ir.compile(MyProgram)          # dump_passes=True by default
   compiled.validate_ir(tensors, expected)   # per-pass numeric check (torch codegen)
   ```

   The report names the first diverging pass (e.g. `19_after_ExpandMixedKernel`);
   read that pass's developer doc under
   [`dev/passes/`](../../../dev/passes/00-pass_manager.md) to understand its
   transform.

5. **Confirm on-device with pto codegen** when the divergence is below torch
   codegen (correct in torch simulation, wrong on-board): capture the suspect
   tensor with [selective dump](03-selective-dump.md) and diff against golden. A
   common cause of exactly this "correct in torch, wrong on device" signature is a
   **lost WAR dependency** — see the gotcha below before digging further.

## Known Gotcha: Lost WAR Dependencies (loop-carried buffers)

One scheduling hazard produces **silent, intermittent** precision errors that
steps 1–2 will *not* catch: torch codegen runs sequentially, so the race never
reproduces in simulation — only on-device (step 3) does it surface.

**Cause.** In AUTO runtime scopes, the runtime scheduler currently omits
**WAR (write-after-read)** anti-dependencies for a buffer allocated once and
carried across loop iterations. For a buffer written then read each iteration:

- `writer(N)` (inout) → `reader(N)` (pure `Input`): RAW edge emitted ✓
- `writer(N)` → `writer(N+1)` (loop-carried): WAW edge emitted ✓
- `reader(N)` → `writer(N+1)`: **WAR edge missing ✗**

Because `reader(N)` produces no new version of the buffer, `writer(N+1)` depends
only on `writer(N)` — never on `reader(N)`. The scheduler may then run
`writer(N+1)` concurrently with `reader(N)`, overwriting the buffer mid-read → a
data race and corrupted output (observed: ~10–35% of values wrong, e.g. shared
gate/accumulator buffers racing across MLP bands).

**Symptom.** Correct in torch codegen (steps 1–2); wrong and *non-deterministic*
on-device.

**Fix (today): establish the WAR edge manually** with `pl.submit(..., deps=[...])`
so `reader(N)` completes before `writer(N+1)` overwrites the buffer:

```python
# reader(N) must finish before writer(N+1) overwrites `buf`
_, tid_read  = pl.submit(self.reader, buf, ...)
_, tid_write = pl.submit(self.writer, buf, ..., deps=[tid_read])
```

Automatic detection / guarding is tracked in **issue #2058**. Until it lands,
PyPTO neither warns nor auto-fixes this case in AUTO scopes — you must add the
edge yourself. See [Perf › Dependency & Dispatch](../perf/03-dependency-dispatch.md).

## See Also

- [Torch Codegen Debug](01-torch-golden.md) — the golden-comparison recipes
- [Pass-IR Bisection](02-pass-ir-bisection.md) — `--dump-passes` / `validate_ir`
- [Selective Tensor Dump](03-selective-dump.md) — `pl.dump_tag` + runtime-DFX
- Developer reference: [`dev/debug/00-torch_codegen.md`](../../../dev/debug/00-torch_codegen.md)
