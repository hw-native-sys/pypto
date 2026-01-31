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
This pass runs on each `SeqStmts` node and may reorder its **direct children** under dependency constraints.

- **Compute-like (typical reorder candidates)**: `AssignStmt`, `EvalStmt`
- **Control-flow / terminator nodes (kept stable in relative order)**: `IfStmt`, `ForStmt`, `ReturnStmt`, `YieldStmt`

## Phase 1: Control Flow Node Support (CF-aware Analysis)

### Phase 1 Overview
**Phase 1** extends the scheduler to treat control flow nodes (IfStmt, ForStmt) as immovable black-box composite nodes in the dependency graph. This enables compute statements to be reordered across control flow boundaries when data dependencies allow.

**Key Innovation:** Instead of cutting statement streams into isolated segments separated by CF barriers, Phase 1 analyzes dependencies at the parent statement level (SeqStmts), allowing better reordering opportunities.

### Design Principle: Black-Box CF Nodes
- **Immovable:** Control flow nodes cannot change relative order (if A comes before B, A must stay before B)
- **Black-box:** Statement-level analysis uses `StmtEffect` to conservatively summarize CF node reads/writes
- **Permeable:** Compute statements can cross CF boundaries if data dependencies permit

### StmtEffect: Conservative Side-Effect Summary
Each statement (including CF nodes) is analyzed for side effects:

```cpp
struct StmtEffect {
  std::set<MemRefPtr> reads;                  // MemRefs read by statement
  std::set<MemRefPtr> writes;                 // MemRefs written by statement
  bool has_unknown_side_effect = false;       // Conservative flag
};
```

**Analysis rules by statement type:**
- **AssignStmt**: writes = var, reads = value's MemRefs
- **EvalStmt**: reads = expr's MemRefs, unknown_side_effect = true (conservative)
- **IfStmt**: Union of condition reads + both branch effects
- **ForStmt**: Union of bounds reads + body reads/writes (loop-carried)
- **SeqStmts/OpStmts**: Fold effects from all children
- **Return/Yield**: unknown_side_effect = true (terminators)

**Conservative union for branching:** When IfStmt or ForStmt can execute different code paths, we conservatively take the union of all possible effects.

### Scheduling with CF Nodes

**Dependency graph construction (CF-aware mode):**
1. Analyze all statements (compute + CF) using MemRef reads/writes
2. Build RAW/WAW/WAR edges using StmtEffect results
3. Unknown side effects create barriers (edges to all subsequent statements)

**Ordering constraints (Stability Chain):**
After dependency edges are built, add "CF stability chain":
- Identify all CF-like nodes in original order: `c0, c1, ..., ck`
- Add edges: `c0 → c1 → ... → ck`
- This preserves CF relative order while allowing compute to cross them

**Candidate selection (Strategy A):**
During Kahn scheduling, prioritize schedulable compute statements over CF nodes:
1. First pass: Schedule compute statements with best score
2. If none available: Fall back to CF nodes
3. This prevents CF nodes from "blocking" compute optimization

### Example: Cross-CF Reordering

**Input:**
```python
tile_a = load(input_a)      # Depends on input_a

if cond:                    # CF node (reads cond)
    tile_b = add(tile_a, tile_a)

tile_c = load(input_c)      # Independent of If (reads different input)
result = add(tile_c, tile_c) # Depends on tile_c
```

**Dependency Analysis:**
- tile_a depends on input_a (RAW)
- if depends on cond (reads cond expression)
- tile_c depends on input_c (RAW, independent of tile_a)
- result depends on tile_c (RAW)
- result does NOT depend on if statement (different MemRefs)

**Phase 1 Optimized Order:**
```python
tile_a = load(input_a)      # tile_a first (needed by if body)
tile_c = load(input_c)      # tile_c can cross if (no dependency)
result = add(tile_c, tile_c) # result follows tile_c

if cond:                    # If node preserved (CF stability chain)
    tile_b = add(tile_a, tile_a)
```

**Benefit:** tile_c load and result computation moved before if → better pipelining and cross-pipe synchronization.

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
- `live_by_pair_`: Global live event count per pipeline pair (counts unique active producers, not edges)
- `pending_successors_`: Per-producer map tracking unscheduled consumers per pipe pair
- `incoming_producers_`: Per-consumer list of (producer, pair) dependencies
- `peak_by_pair_`: Peak pressure statistics

**Key methods:**
- `PredictAfterScheduling(candidate)`: Predicts resource impact, returns whether scheduling is feasible
- `ReleaseIncomingBeforeExecute(stmt)`: Release wait-side events before statement execution
- `AllocateOutgoingAfterExecute(stmt)`: Allocate set-side events after statement execution

### Event Semantics: Broadcast Model

**Hardware reality**: Cross-pipe synchronization uses broadcast semantics:
- Producer issues **ONE** `set_event(id)` per unique (SRC, DST) pair
- Multiple consumers on the same DST pipe can **share** this event via `wait_event(id)`
- **Event_id slot is freed when the FIRST consumer is scheduled** (matching InsertSyncPass behavior: `sync_dst` is inserted only before the first consumer; after that the hardware event_id can be reused)

**Implementation**:
- `pending_successors_[producer][pair].remaining`: Counts unscheduled consumers (for correctness bookkeeping).
- `pending_successors_[producer][pair].event_live`: Tracks whether this (producer, pair) still occupies an event_id slot.
- `incoming_producers_[consumer]`: Tracks which (producer, pair) combinations this consumer depends on
- `live_by_pair_[pair]`: Counts unique active producers (NOT edges)

**Example**: If producer P on MTE2 has 3 consumers on V:
- Old (per-edge): `live_by_pair_[(MTE2,V)] += 3` ❌
- New (broadcast): `live_by_pair_[(MTE2,V)] += 1` ✓

The (producer, pair) event_id slot is freed when the **first** of these consumers is actually scheduled. Remaining consumers still keep the dependency relationship, but do not consume an event_id slot.

**Lifecycle**:
```text
P (MTE2) → C1, C2, C3 (all on V)

After P executes:  live_by_pair_[(MTE2,V)] = 1, pending_successors_[P][(MTE2,V)] = 3
After first scheduled consumer (e.g. C2): pending = 2, live = 0 (event_id slot freed)
After next consumer (e.g. C1): pending = 1, live = 0
After last consumer (e.g. C3): pending = 0 (bookkeeping cleanup), live = 0
```

### Consumer Role Tracking

This scheduler treats “first-consumer” as a **dynamic** concept:

- **releases_event (first scheduled consumer)**: If a ready candidate has at least one incoming (producer, pair) whose `event_live` is still true, then scheduling this candidate will free at least one event_id slot.
  - This matches the runtime insertion model: whichever consumer is scheduled first will be the one that carries the `sync_dst` wait, and thus frees the event_id slot for reuse.
  - Other consumers still keep dependency ordering (they must be scheduled after the producer), but they do not consume additional event_id slots.

## Scheduling Algorithm

### Overall Flow
1. **Visit each `SeqStmts`**: Collect and visit all direct children
2. **Build dependency graph (CF-aware)**: Conservative MemRef hazard detection (RAW/WAW/WAR) + unknown side-effect barriers
3. **Add CF stability chain**: Preserve relative order among CF/terminator nodes
4. **Kahn topological sort**: Enhanced with event_id resource constraints
5. **Multi-strategy scheduling**: Try multiple heuristics to find a feasible schedule (strict), then best-effort (relaxed)

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

```text
Initialize ready set with statements having indegree 0
While unscheduled statements exist:
  For each candidate in ready set:
    Predict resource impact if scheduled
    Skip if violates constraint (live events > 8)
    Score candidate using strategy
    Prefer candidates that release at least one live event_id slot (`releases_event`)

  Select best candidate
  Release incoming events (before execution)
  Mark as scheduled
  Allocate outgoing events (after execution)
  Update peak statistics

  Update ready set with new zero-indegree statements
```

**First-Consumer Priority Optimization**:

To minimize peak event pressure, the scheduler prioritizes first-consumers:
- Candidate comparison first prefers candidates that `releases_event == true`
- This schedules event-releasing consumers earlier, freeing event_id slots sooner
- Reduces the likelihood of exceeding the 8-event limit per pipeline pair
- Works in conjunction with Strategy A (compute over CF nodes)

**Example benefit**:
```text
Without priority:  tail_x → [consumer_1, consumer_2, ..., consumer_0] → event held until consumer_0
With priority:     tail_x → [consumer_0, consumer_1, consumer_2, ...] → event released immediately
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
Internal bookkeeping stays consistent:
- `live_by_pair_` never goes negative; predicted counts must be ≥ 0
- `pending_successors_` and `incoming_producers_` remain consistent (no double-release, no missing producer-pair state)
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

```text
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

1. **Phase 1 limitations:**
   - Control flow nodes treated as immovable black boxes (no inter-procedural analysis)
   - StmtEffect uses conservative union for branches (may create false dependencies)
   - No path-sensitive analysis (assumes all branches equally likely)
   - No loop-invariant code motion (LICM not implemented yet)
2. **Conservative:** MemRef-based analysis may be overly conservative
3. **Hardcoded limit:** `kMaxEventIds = 8` not configurable
4. **Best-effort fallback:** May not always satisfy constraints

## Future Work (Phase 2+)

**Path-sensitive analysis:**
- Analyze conditional branches to enable more aggressive reordering
- Differentiate between "must execute" vs "may execute" effects

**Loop-invariant code motion (LICM):**
- Move loop-invariant computations outside ForStmt bodies
- Requires proving expressions don't change across iterations

**Inter-procedural analysis:**
- Analyze nested CF bodies for finer-grained reordering opportunities
- Recursively schedule within If/For statement bodies

## Debugging

Enable debug logs to track:
- Segment scheduling: "scheduled segment size=X, worst_peak=Y"
- Strategy recovery: "Recovered feasible schedule with strategy=Z"
- Relaxed fallback: "Cannot satisfy event limit, using best-effort"

Verify:
1. `GetStmtPipe` returns correct pipeline types
2. Dependency graph captures RAW/WAW/WAR correctly
3. Live event tracking matches expectations
