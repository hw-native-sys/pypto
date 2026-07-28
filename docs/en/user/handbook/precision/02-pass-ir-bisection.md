# Precision Localization: Pass-IR Bisection

> **Status:** DRAFT skeleton. Find the *first* pass whose IR diverges from the
> expected result.

## Symptom

Codegen directly from program IR is correct, but codegen after
`PassManager(Default)` is wrong — so some pass introduced the divergence.

## Tools

- `--dump-passes` + `PassDumpLevel` — dump IR after each pass (including the
  explicit-layout dump level added on the current branch).
- `CompiledProgram.validate_ir` — validate each dumped pass IR against golden.

*TODO — enumerate the `PassDumpLevel` values and what each captures.*

## Steps

*TODO:*

1. Run with `--dump-passes` at the appropriate `PassDumpLevel`.
2. Codegen + validate each dumped stage.
3. Locate the first pass whose IR fails validation.
4. Read that pass's dev doc (`docs/en/dev/passes/NN-*.md`) to understand its
   transform.

## How to Read the Output

*TODO — describe the dump directory layout, file naming, and how to diff two
adjacent stages.*

## See Also

- [Torch Codegen Debug](01-torch-golden.md)
- Developer reference: [`dev/debug/00-torch_codegen.md`](../../../dev/debug/00-torch_codegen.md), [`dev/passes/00-pass_manager.md`](../../../dev/passes/00-pass_manager.md)
