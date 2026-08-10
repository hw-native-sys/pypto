# A5 Mixed Kernel: MX Scale Transport (V2C Failure → GM)

Chinese design note (full analysis): [zh version](../../../zh/dev/design/a5-mixed-mx-scale-transport.md).

**Decision:** keep FP8 **data** on V2C; **forbid** MX **scale** on V2C. Route scale through GM in the packed ZZ layout produced by `ExpandMxPackedQuant` (`store` + `tensor.view(MX_A_ZZ)` + `tile.load(Mat)`). [`ExpandMixedKernel`](../passes/23-expand_mixed_kernel.md) rejects `FP8E8M0` `tpush_to_aic`. Do **not** FP32-disguise or pad scale for V2C. ST tests use orchestration-level GM scale staging.
