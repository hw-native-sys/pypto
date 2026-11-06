# MaterializeSimtOutputs

Preserves Tile value semantics across partial SIMT writes. Every writable Tile
argument gets a dedicated `tile.move` copy before the launch. Its result uses
that copy's type and, after `InitMemRef`, the same allocation. Unwritten elements
therefore retain their input values, while old Tile handles remain readable.
Tensor arguments keep reference semantics: GM updates use the original storage
without staging through a Tile or creating a copy.

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeSimtOutputs()` | `passes.materialize_simt_outputs()` | Function |

**Requires:** SSAForm, TileMemoryInferred.
**Produces:** SimtOutputsMaterialized.
**Invalidates:** HasMemRefs, AllocatedMemoryAddr.

## Example

```python
# Before
updated = pl.simt.launch(self.update, old, block_dims=(64, 1, 1), result_param_indices=(0,))

# After (old is a Vec Tile)
seed = pl.tile.move(old, target_memory=pl.Mem.Vec, attrs={"simt_output_seed": True})
updated = pl.simt.launch(self.update, seed, block_dims=(64, 1, 1), result_param_indices=(0,))
```

`simt_output_seed` is compiler metadata. Each marked copy serves one launch;
running the pass again does not add another copy. The property verifier checks
that every writable Tile uses a dedicated initialized seed.

The pass rejects known overlapping arguments when at least one is writable.
It follows aliases and views and compares known MemRef intervals. Distinct
runtime GM arguments still require the caller to guarantee nonoverlap; this
analysis does not establish race freedom for arbitrary runtime addresses.
Interval sorting bounds the overlap checks by O(K log K) for K arguments.

## Pipeline integration

The default pipeline runs this pass after `MaterializeTensorStrides`, before
`SpecializeSimtHelpers` and `InitMemRef`. Borrowed helper parameters receive no
separate allocations. Memory planning treats each launch result as a definition
of the writable seed, and loop-input isolation preserves old Tile values even
when yield fixup copies an update back into a carry. PyPTO, DSA-RP and PTOAS
memory-planner paths use this protocol.

SIMT kernels currently stay in Functional IR; automatic Buffer IR selection
falls back for them. An explicitly Buffer-stage SIMT function is rejected.
See [Scalar SIMT IR](../ir/11-simt.md) for remaining execution limitations.
