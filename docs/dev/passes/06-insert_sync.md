# InsertSync Pass

Analyzes data dependencies and inserts synchronization operations for correct multi-pipeline execution.

## Overview

This pass is the most complex transformation pass in PyPTO. It analyzes data dependencies across hardware pipelines and inserts synchronization operations (sync_src, sync_dst, bar_v, bar_m) to ensure correct execution.

**Key responsibilities**:
- Analyze inter-pipeline data dependencies
- Insert sync_src/sync_dst for producer-consumer synchronization
- Insert barriers (bar_v, bar_m) for global synchronization
- Manage event IDs and pipeline masks

**When to use**: Run after InitMemRef and BasicMemoryReuse, before code generation. Required for correct multi-pipeline hardware execution.

## API

| C++ | Python | Level |
|-----|--------|-------|
| `pass::InsertSync()` | `passes.insert_sync()` | Function-level |

**Factory function**:
```cpp
Pass InsertSync();
```

**Python usage**:
```python
from pypto.pypto_core import passes

sync_pass = passes.insert_sync()
program_with_sync = sync_pass(program)
```

## Algorithm

1. **Pipeline Assignment**: Determine which pipeline each operation belongs to (using backend pipe info)
2. **Dependency Analysis**: Build dependency graph between operations across pipelines
3. **Sync Point Identification**: Identify producer-consumer pairs requiring synchronization
4. **Event ID Allocation**: Assign unique event IDs for sync operations
5. **Sync Insertion**:
   - Insert sync_src after producer (set_pipe = producer pipe)
   - Insert sync_dst before consumer (wait_pipe = producer pipe)
6. **Barrier Insertion**: Insert global barriers (bar_v, bar_m) where needed
7. **Optimization**: Merge redundant sync operations

**Synchronization patterns**:
- **Producer-consumer**: sync_src (producer) → sync_dst (consumer)
- **Global barrier**: bar_all / bar_v / bar_m
- **Pipeline-specific**: Use PipeType from backend

## Example

### Cross-Pipeline Dependency

**Before**:
```python
# Vector pipeline
tile_a = block.load(tensor, [0, 0], [64, 64])  # Pipe V

# Matrix pipeline
tile_b = block.matmul(tile_a, tile_c)  # Pipe M, depends on tile_a from Pipe V
```

**After**:
```python
# Vector pipeline
tile_a = block.load(tensor, [0, 0], [64, 64])  # Pipe V
system.sync_src(set_pipe=PipeType.V, wait_pipe=PipeType.M, event_id=0)

# Matrix pipeline
system.sync_dst(set_pipe=PipeType.M, wait_pipe=PipeType.V, event_id=0)
tile_b = block.matmul(tile_a, tile_c)  # Pipe M
```

### Multiple Dependencies

**Before**:
```python
tile_a = block.load(...)  # Pipe V
tile_b = block.load(...)  # Pipe V
tile_c = block.add(tile_a, tile_b)  # Pipe V
tile_d = block.matmul(tile_c, ...)  # Pipe M, depends on tile_c
```

**After**:
```python
tile_a = block.load(...)  # Pipe V
tile_b = block.load(...)  # Pipe V
tile_c = block.add(tile_a, tile_b)  # Pipe V
system.sync_src(set_pipe=PipeType.V, wait_pipe=PipeType.M, event_id=0)

system.sync_dst(set_pipe=PipeType.M, wait_pipe=PipeType.V, event_id=0)
tile_d = block.matmul(tile_c, ...)  # Pipe M
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`
```cpp
Pass InsertSync();
```

**Implementation**: `src/ir/transforms/insert_sync.cpp`
- Uses backend pipe information (via globally configured backend)
- Performs data flow analysis across pipelines
- Implements sync optimization algorithms
- Manages event ID allocation

**Backend integration**:
```cpp
#include "pypto/backend/common/backend.h"
// Uses Backend::GetPipeType() to determine operation pipelines
```

**Python binding**: `python/bindings/modules/passes.cpp`
```cpp
passes.def("insert_sync", &pass::InsertSync, "Insert synchronization operations");
```

**Tests**: `tests/ut/ir/transforms/test_insert_sync.py`
- Tests single cross-pipeline dependency
- Tests multiple dependencies
- Tests barrier insertion
- Tests event ID allocation
- Tests sync optimization

## Backend Dependency

This pass requires a configured backend to obtain pipeline information:

```python
from pypto import backend

# Set backend before running InsertSync
backend.set_backend(backend.Ascend910B())
program_with_sync = passes.insert_sync()(program)
```
