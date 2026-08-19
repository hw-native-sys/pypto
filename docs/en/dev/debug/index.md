# Debug

Tools for narrowing down where a compiled program stops matching its reference.

| Page | What it covers |
| ---- | -------------- |
| [Torch Code Generation](00-torch_codegen.md) | Lowering PyPTO IR into an executable Python/PyTorch script for numerical validation |

## See Also

- [Torch Codegen Debug Guide](../../user/tools/01-torch-codegen.md) — the same tool from a user's perspective.
- [Error Handling](../02-error-handling.md) — exception types and IR source locations in failures.
- [Logging](../03-logging.md) — the two logging subsystems and how to raise their verbosity.
- [Runtime DFX Flags](../03-runtime-dfx.md) — runtime-side diagnostics, including selective tensor dumps.
- [IR Verifier](../passes/99-verifier.md) — catching illegal IR between passes.
