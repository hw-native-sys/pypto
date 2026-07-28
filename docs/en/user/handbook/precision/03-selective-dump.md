# Precision Localization: Selective Tensor Dump

> **Status:** DRAFT skeleton. Capture and compare a specific tensor on-device.

## Symptom

You suspect a single intermediate tensor is wrong and want its actual on-device
values, without dumping everything.

## Tools

- **Frontend tagging:** `pl.dump_tag(...)` — mark a tensor for selective dump.
- **Runtime-DFX selective dump** — the runtime writes only tagged tensors.
- **L2 swimlane double-run** — run twice to capture on-board values for
  comparison.

## Steps

_TODO:_

1. Tag the suspect tensor with `pl.dump_tag`.
2. Enable the runtime-DFX selective-dump flag.
3. Run (double-run for L2 swimlane) and collect the dump.
4. Compare against golden.

## How to Read the Output

_TODO — dump file format, naming, and comparison recipe._

## See Also

- Developer reference: [`dev/03-runtime-dfx.md`](../../../dev/03-runtime-dfx.md) (Selective tensor dump, L2 swimlane double-run)
- [DFX Features](../dfx/00-flag-matrix.md)
