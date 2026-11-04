# SpecializeSimtHelpers

Creates SIMT helper variants for the actual launch dimensions, storage metadata,
and compilation target. The helper remains a void device function; it does not
become a runtime task.

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::SpecializeSimtHelpers()` | `passes.specialize_simt_helpers()` | Program |

**Requires:** SSAForm.
**Produces:** SimtSpecialized.

## Specialization key and metadata

The key includes the original helper, all three block dimensions, canonical
parameter types/layouts, and target architecture. Equal thread products alone
are insufficient: `(64, 1, 1)` and `(8, 8, 1)` create different variants.
Concrete allocation identities and addresses are removed from helper types.
Runtime scalar values and loop iterations are not specialization keys.

Dynamic shape, stride, valid-shape, and offset expressions become explicit
scalar arguments when needed. Existing scalar parameters are reused when they
already supply the expression. Two callers with different runtime stride
variables can share one variant. Helper bodies must have no unresolved captures.
The compiler uses existing Tile/Tensor types and scalar parameters, without
introducing an additional pointer/view ABI.

Variants carry compiler-owned `simt_block_dims`, `simt_target`,
`simt_max_threads`, and `simt_max_regs` function attributes. Resources come from
the target handler's `infer_simt_resources` rules. Users do not specify register
or thread limits. `SimtSpecialized` verifies these attributes against every
launch. Serialization and Python printing preserve them.

Equivalent variants share a helper through the bounded structural-signature
index. Re-running the pass preserves existing variants and does not append
duplicate metadata arguments. Original helper definitions remain in the program.

## Ordering

The default pipeline runs this pass after `MaterializeSimtOutputs` and before
`InitMemRef`. Earlier `FlattenTileNdTo2D` rewrites both the caller's Tile types
and the helper's scalar indices. Its initial SIMT support accepts implicit
packed Tile layouts with positive static leading extents. GM Tensor ranks stay
unchanged. Later memory planning allocates only caller storage.

See [Scalar SIMT IR](../ir/11-simt.md) for the resource table and the boundary
between compiler preparation and PTOAS/device execution.
