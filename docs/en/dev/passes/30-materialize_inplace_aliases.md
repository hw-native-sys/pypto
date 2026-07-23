# MaterializeInplaceAliases Pass

Materializes safe operation-boundary input/output aliases for the PTOAS memory planner.

## Overview

PTOAS can pack handles whose lifetimes do not overlap, but an instruction's
input lifetime and output lifetime touch at the instruction itself. The PTO
module does not otherwise communicate whether that instruction permits
`dst == src`, so conservative PTOAS planning keeps both handles separate.

`MaterializeInplaceAliases` encodes only this operation-level fact. For a tile
assignment, it may retarget the output MemRef to a directly consumed input
when that input's effective last use is the output definition. PTOAS still
owns all independent lifetime packing, address selection, alignment, and
capacity validation.

The pass runs in the default pipeline after
[`MaterializeSemanticAliases`](29-materialize_semantic_aliases.md) and before
[`MemoryReuse`](31-memory_reuse.md). Its behavior is enabled only when the
active `PassContext` selects `MemoryPlanner.PTOAS`; under `MemoryPlanner.PYPTO`
it is a no-op.

## Safety rules

A candidate input must:

- have the same memory space, data type, physical allocation size, block
  layout, scatter layout, fractal size, and padding semantics as the output;
- die exactly where the output is defined, using loop- and phi-aware effective
  lifetime analysis (including post-loop return wrappers resolved through
  their physical allocation);
- belong to an operator registered as in-place safe;
- not be protected by `forbid_output_alias`;
- not cross a conflicting software-pipeline stage;
- not violate the Ascend910B split-AIV load/tpop hazard;
- not participate in a phi family.

Operations with `set_output_reuses_input(k)` are already handled by
`MaterializeSemanticAliases` / InitMemRef and are left unchanged. The pass
also consumes and strips `pipeline_membership` under PTOAS because the full
`MemoryReuse` pass is absent on that path.

The implementation reuses the lifetime, hazard, and registry no-alias
analysis in `memory_reuse_pass.cpp`. It builds indexed analysis once and
makes one ordered candidate decision per output; it does not perform global
buffer packing.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeInplaceAliases()` | `passes.materialize_inplace_aliases()` | Function-level |

```python
from pypto.pypto_core import passes

with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
    result = passes.materialize_inplace_aliases()(program)
```

## Tests

Coverage lives in:

- `tests/ut/ir/transforms/test_memory_reuse.py`
- `tests/ut/codegen/test_memory_planner_switch.py`
