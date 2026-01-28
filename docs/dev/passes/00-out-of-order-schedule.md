# OutOfOrderSchedulerPass

## Overview

`OutOfOrderSchedulerPass` reschedules reorderable statements to optimize cross-pipe dependencies while keeping peak event pressure ≤ 8 per pipeline pair.

**Goal:** Under dependency constraints, reorder statements to minimize peak pressure of cross-pipe synchronization events.

## Core Concepts

### Pipeline Types
Different computational units: M (CUBE), V (VECTOR), S (SCALAR), MTE1/2/3 (transfers), FIX, ALL.

### Cross-Pipe Dependencies
When a statement on pipeline A depends on pipeline B (A ≠ B), synchronization via events is needed:
- Producer (A) issues `set_event`
- Consumer (B) waits on `wait_event`

### Live Events
Event is "live" from `set_event` to `wait_event`. Resource constraint: max 8 live events per pipeline pair.

### Reorderable Statements
Only computation can be reordered:
- `AssignStmt`, `EvalStmt`

Non-reorderable (barriers):
- Control flow: `For`, `If`, `While`
- Termination: `Return`, `Yield`

## Main Components

### MemRefCollector
Collects memory references from expressions to build dependency relationships. Analyzes reads/writes to detect:
- **RAW (Read-After-Write)**: reads must follow writes
- **WAW (Write-After-Write)**: writes must follow previous writes
- **WAR (Write-After-Read)**: writes must follow all reads

### GetStmtPipe
Extracts pipeline type of statement:
1. Use `Op::GetPipe()` if available
2. Fall back to `call.kwargs["pipe_type"]`
3. Default to `PipeType::S` (scalar)

Returns the pipeline where the statement executes.

### LiveCrossPipeEvents
Tracks cross-pipe event state during scheduling:
- `live_by_pair_`: Global live event count per pipeline pair
- `live_incoming_cross_by_pair_`: Incoming events per statement
- `peak_by_pair_`: Peak pressure statistics

**Key methods:**
- `PredictAfterScheduling(candidate)`: Predicts resource impact, returns whether scheduling is feasible
- `ReleaseIncomingBeforeExecute(stmt)`: Release wait-side events before statement execution
- `AllocateOutgoingAfterExecute(stmt)`: Allocate set-side events after statement execution

## Scheduling Algorithm

### Overall Flow
1. **Identify reorderable segments**: Group consecutive computation statements between barriers
2. **Build dependency graph**: MemRef-based hazard detection (RAW/WAW/WAR)
3. **Kahn topological sort**: Enhanced with resource constraints
4. **Multi-strategy scheduling**: Try multiple heuristics to find feasible schedule

### Building Dependency Graph

For each statement, collect read/write memory references:
- Track last writer for each memory location
- Track all readers since last write

Build edges:
- RAW: Add edge from last writer to current reader
- WAW: Add edge from last writer to current writer
- WAR: Add edges from all readers to current writer

Mark each edge as cross-pipe or same-pipe based on pipeline types.

### Kahn + Resource Constraints

Enhanced Kahn algorithm that respects event limits:

```
Initialize ready set with statements having indegree 0
While unscheduled statements exist:
  For each candidate in ready set:
    Predict resource impact if scheduled
    Skip if violates constraint (live events > 8)
    Score candidate using strategy

  Select best candidate
  Release incoming events (before execution)
  Mark as scheduled
  Allocate outgoing events (after execution)
  Update peak statistics

  Update ready set with new zero-indegree statements
```

### Candidate Selection Strategies

**Selection criteria** (in priority order):
1. **kMinMaxThenSumThenIndex** (default):
   - Primary: Minimize worst pipeline pair pressure (pred_max)
   - Secondary: Minimize total pressure (pred_sum)
   - Tertiary: By original index

2. **kMinSumThenMaxThenIndex**:
   - Primary: Minimize total pressure first
   - Avoids local greedy traps

3. **kMinMaxThenIndex**:
   - Only minimize worst pressure
   - Simpler, faster decisions

### Fallback Strategy

Try strategies in order:

1. **Strict mode** (enforce_limit=true):
   - Try each strategy
   - Enforce 8-event limit strictly
   - Return first successful schedule

2. **Relaxed mode** (enforce_limit=false):
   - If all strict strategies fail
   - Don't enforce limit, but minimize pressure
   - Generate best-effort topological order
   - Logs warning to user

## Invariants

### Resource Constraint
Each pipeline pair `(SRC, DST)` has at most 8 live events at any time. This is hardware-enforced and cannot be violated.

**Invariant verification:**
- `PredictAfterScheduling` checks this before scheduling
- `INTERNAL_CHECK(pred >= 0)` ensures release doesn't make count negative

### State Consistency
`live_by_pair_` and `live_incoming_cross_by_pair_` stay synchronized:
- Release operations never exceed current count
- Incoming and global counts match
- Peak statistics tracked accurately

### Topological Order
Output satisfies all dependencies (RAW/WAW/WAR). Guaranteed by Kahn algorithm: only schedules statements with indegree 0.

## Example

### Input Code

```python
A = compute_on_M(...)     # Pipeline M
B = compute_on_V(A)       # Pipeline V, depends on A (cross-pipe)
C = compute_on_M(...)     # Pipeline M
D = compute_on_V(C)       # Pipeline V, depends on C (cross-pipe)
E = compute_on_V(B, D)    # Pipeline V, depends on B and D
```

### Dependency Graph

```
A(M) → B(V)
       ↓
C(M) → D(V) → E(V)
```

Cross-pipe edges: A→B, C→D

### Original Schedule

**Order:** A → B → C → D → E

| Time | Execute | Live Events | (M→V) Count |
|------|---------|-------------|-------------|
| 1    | A       | {A→B}       | 1           |
| 2    | B       | {}          | 0           |
| 3    | C       | {C→D}       | 1           |
| 4    | D       | {}          | 0           |
| 5    | E       | {}          | 0           |

**Peak M→V events:** 1

### Optimized Schedule

**Order:** A → C → B → D → E

| Time | Execute | Live Events | (M→V) Count |
|------|---------|-------------|-------------|
| 1    | A       | {A→B}       | 1           |
| 2    | C       | {A→B, C→D}  | 2           |
| 3    | B       | {C→D}       | 1           |
| 4    | D       | {}          | 0           |
| 5    | E       | {}          | 0           |

**Peak M→V events:** 2

**Benefit:** Pipeline M operations batched together (A, C), then pipeline V operations (B, D, E). Reduces pipeline switches and improves instruction-level parallelism, even though peak event pressure slightly increases.

## Complexity

- **Time:** O(n²) graph building + O(n × |ready| × 3) Kahn scheduling = O(n²) worst case
- **Space:** O(n²) edges + O(pipeline pairs × n) live events

## Limitations

1. **Straight-line only:** Control flow blocks reordering
2. **Conservative:** MemRef-based analysis may be overly conservative
3. **Hardcoded limit:** `kMaxEventIds = 8` not configurable
4. **Best-effort fallback:** May not always satisfy constraints

## Debugging

Enable debug logs to track:
- Segment scheduling: "scheduled segment size=X, worst_peak=Y"
- Strategy recovery: "Recovered feasible schedule with strategy=Z"
- Relaxed fallback: "Cannot satisfy event limit, using best-effort"

Verify:
1. `GetStmtPipe` returns correct pipeline types
2. Dependency graph captures RAW/WAW/WAR correctly
3. Live event tracking matches expectations
