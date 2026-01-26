# OutOfOrderSchedulerPass Implementation Details

## Overview

`OutOfOrderSchedulerPass` is an optimization pass in the PyPTO IR transformation layer that **reschedules reorderable statements to optimize cross-pipe dependencies**.

**Core Goal:** Under the premise of preserving program semantics (dependency relationships), reorder statements such that **the peak pressure of cross-pipe events does not exceed 8 per pipeline pair** (`kMaxEventIds = 8`).

---

## Core Concepts

### 1. Pipeline Types (PipeType)

Different computational units are abstracted as "pipelines":
- **M (CUBE)**: Matrix/cube computation unit
- **V (VECTOR)**: Vector computation unit
- **S (SCALAR)**: Scalar computation unit
- **MTE1/MTE2/MTE3**: Transfer pipelines
- **FIX**: Fixed pipeline
- **ALL**: All pipelines

Each statement is assigned to a pipeline based on its operation type.

### 2. Cross-Pipe Dependencies

A **cross-pipe dependency** occurs when a statement executes on pipeline A and its successor dependency statement executes on pipeline B (A ≠ B).

```
Statement1 (Pipeline M) → Statement2 (Pipeline V)  // Cross-pipe dependency
Statement2 (Pipeline V) → Statement3 (Pipeline V)  // Same-pipe dependency
```

Cross-pipe dependencies require synchronization via events:
- **Producer** (Statement1) executes on pipeline A and emits `set_event`
- **Consumer** (Statement2) waits on `wait_event` before executing on pipeline B

### 3. Live Events

An event is "live" from the moment `set_event` is issued until it is consumed by `wait_event`.

**Resource Constraint:** Each pipeline pair `(SRC, DST)` allows at most 8 simultaneously live events (`kMaxEventIds = 8`).

### 4. Reorderable Statements

Only "computation" statements can be reordered:
- `AssignStmt`: Assignment statement
- `EvalStmt`: Evaluation statement

**Non-reorderable statements** (act as scheduling barriers):
- Control flow: `For`, `If`, `While`
- Termination: `Return`, `Yield`
- Other complex statements

---

## Main Components

### 1. MemRefCollector

**Purpose:** Collects all memory references (MemRef) from expressions.

```cpp
class MemRefCollector : public IRVisitor {
 public:
  std::set<MemRefPtr> memrefs;

  void VisitExpr_(const VarPtr& var) override {
    if (auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(var->GetType())) {
      if (shaped_type->memref_.has_value()) {
        memrefs.insert(*shaped_type->memref_);
      }
    }
    IRVisitor::VisitExpr_(var);
  }
};
```

**Usage:** Analyzes read/write memory references in statements to build dependency relationships.

### 2. GetStmtPipe

**Purpose:** Extracts the pipeline type of a statement.

**Strategy:**
1. First use `Op::GetPipe()` to get the operation's pipeline type
2. If unavailable, get string-form pipeline type from `call.kwargs["pipe_type"]`
3. Default return `PipeType::S` (scalar pipeline)

```cpp
PipeType GetStmtPipe(const StmtPtr& stmt) {
  // Handle AssignStmt
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    if (auto call = std::dynamic_pointer_cast<const Call>(assign->value_)) {
      return get_call_pipe(call);
    }
  }
  // Handle EvalStmt
  else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    if (auto call = std::dynamic_pointer_cast<const Call>(eval->expr_)) {
      return get_call_pipe(call);
    }
  }
  return PipeType::S;
}
```

### 3. LiveCrossPipeEvents

**Purpose:** Tracks the live state of cross-pipe events and maintains resource constraints.

**Core Data Structures:**
```cpp
class LiveCrossPipeEvents {
 private:
  int max_events_;  // Maximum event limit (8)

  // Incoming cross-pipe events for each statement: live_incoming_cross_by_pair_[consumer][{SRC, DST}] = count
  std::vector<std::map<PipePair, int>> live_incoming_cross_by_pair_;

  // Global live cross-pipe events: live_by_pair_[{SRC, DST}] = count
  std::map<PipePair, int> live_by_pair_;

  // Peak statistics
  std::map<PipePair, int> peak_by_pair_;
};
```

**Key Methods:**

#### PredictAfterScheduling
Predicts whether scheduling a candidate statement would violate resource constraints.

**Logic:**
1. Calculate delta: event changes (release and allocate) if scheduling this statement
2. Apply delta to current state, check if any pipeline pair exceeds limit

#### ReleaseIncomingBeforeExecute
Releases incoming events (wait side) **before** statement execution.

```cpp
void ReleaseIncomingBeforeExecute(int consumer) {
  for (const auto& [pair, cnt] : live_incoming_cross_by_pair_[consumer]) {
    live_by_pair_[pair] -= cnt;  // Release events
  }
  live_incoming_cross_by_pair_[consumer].clear();
}
```

#### AllocateOutgoingAfterExecute
Allocates outgoing events (set side) **after** statement execution.

```cpp
void AllocateOutgoingAfterExecute(int producer, ...) {
  for (int v : succ_cross[producer]) {
    if (!scheduled[v]) {
      PipePair pair{pipes[producer], pipes[v]};
      live_by_pair_[pair] += 1;  // Allocate event
      live_incoming_cross_by_pair_[v][pair] += 1;
    }
  }
}
```

---

## Scheduling Algorithm

### Overall Flow

```
SeqStmts
  ↓
Identify reorderable segments
  ↓
Call ScheduleSegment for each segment
  ↓
Build dependency graph (MemRef-based hazard detection)
  ↓
Kahn topological sort + resource constraints
  ↓
Output reordered statements
```

### ScheduleSegment Details

#### 1. Build Dependency Graph

**Hazard detection based on memory references** (similar to InsertSyncPass):

```cpp
// RAW (Read-After-Write): read must be after write
for (const auto& r : reads) {
  for (auto const& [m, idx] : last_writer) {
    if (IsSameMem(r, m)) add_dep(idx, i);
  }
}

// WAW (Write-After-Write): write must be after previous write
for (const auto& w : writes) {
  for (auto const& [m, idx] : last_writer) {
    if (IsSameMem(w, m)) add_dep(idx, i);
  }
}

// WAR (Write-After-Read): write must be after all reads
for (const auto& w : writes) {
  for (auto const& [m, indices] : last_readers) {
    if (IsSameMem(w, m)) {
      for (int r_idx : indices) add_dep(r_idx, i);
    }
  }
}
```

**Dependency edge classification:**
```cpp
struct DepEdge {
  int producer_idx;
  int consumer_idx;
  bool cross_pipe;  // Whether cross-pipe
};
```

#### 2. Kahn Topological Sort

**Standard Kahn algorithm + resource constraints:**

```cpp
while (not all statements scheduled) {
  // 1. Select best candidate from ready set
  for (int cand : ready) {
    // Predict resource state after scheduling this candidate
    auto [ok, pred_max, pred_sum] =
        live_events.PredictAfterScheduling(cand, ...);

    if (!ok) continue;  // Violates resource constraint, skip

    // Select best candidate based on strategy
    if (Better(cur, best, strategy)) {
      best = cur;
    }
  }

  // 2. Schedule selected candidate
  live_events.ReleaseIncomingBeforeExecute(best.idx);  // Release incoming
  scheduled[best.idx] = true;
  order.push_back(best.idx);
  live_events.AllocateOutgoingAfterExecute(best.idx, ...);  // Allocate outgoing
  live_events.UpdatePeak();

  // 3. Update ready set
  for (int v : succ[best.idx]) {
    indeg[v]--;
    if (indeg[v] == 0) ready.insert(v);
  }
}
```

#### 3. Candidate Selection Strategy (PickStrategy)

**Three strategies to avoid greedy deadlock:**

1. **kMinMaxThenSumThenIndex (default)**
   - Prioritize minimizing `pred_max` (worst pipeline pair pressure)
   - Secondary: minimize `pred_sum` (total pressure)
   - Tertiary: by original index `idx`

2. **kMinSumThenMaxThenIndex**
   - Prioritize minimizing `pred_sum` (total pressure)
   - Secondary: minimize `pred_max`
   - Useful for avoiding local greedy traps

3. **kMinMaxThenIndex**
   - Only minimize `pred_max`
   - Simplified decision, reduces complexity

**Comparison function:**
```cpp
auto Better = [](const CandidateScore& a, const CandidateScore& b,
                 PickStrategy strategy) -> bool {
  switch (strategy) {
    case kMinMaxThenSumThenIndex:
      if (a.pred_max != b.pred_max) return a.pred_max < b.pred_max;
      if (a.pred_sum != b.pred_sum) return a.pred_sum < b.pred_sum;
      return a.idx < b.idx;
    // ...
  }
};
```

#### 4. Fallback Strategy

**Try in order:**

1. **Strict Mode (enforce_limit=true)**
   - Try three strategies in sequence, finding scheduling satisfying resource constraints
   - Return if found

2. **Relaxed Mode (enforce_limit=false)**
   - If all strict strategies fail, use relaxed mode
   - Don't enforce resource constraints, but try to minimize peak pressure
   - Generate a "best-effort" topological order

```cpp
// Try strict scheduling
for (auto s : strategies) {
  auto r = RunKahn(kMaxEventIds, /*enforce_limit=*/true, s);
  if (r.has_value()) {
    best_strict = std::move(r);
    break;
  }
}

// Fall back to relaxed scheduling
if (!best_strict.has_value()) {
  auto relaxed = RunKahn(kMaxEventIds, /*enforce_limit=*/false,
                         PickStrategy::kMinMaxThenSumThenIndex);
  LOG_WARN << "Cannot satisfy event limit, using best-effort schedule";
}
```

---

## OutOfOrderSchedulerMutator

**Purpose:** Traverses the IR tree and applies scheduling optimization to each `SeqStmts`.

```cpp
class OutOfOrderSchedulerMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // 1. Recursively visit child statements
    std::vector<StmtPtr> visited;
    for (const auto& s : op->stmts_)
      visited.push_back(VisitStmt(s));

    // 2. Segment by reorderability
    std::vector<StmtPtr> segment;
    for (const auto& s : visited) {
      if (IsReorderableStmt(s)) {
        segment.push_back(s);  // Accumulate reorderable statements
      } else {
        flush_segment();  // Barrier encountered, schedule and flush
        out.push_back(s);  // Barrier stays in place
      }
    }
    flush_segment();  // Handle last segment

    return std::make_shared<const SeqStmts>(out, op->span_);
  }
};
```

**Segment flush:**
```cpp
auto flush_segment = [&]() {
  if (segment.empty()) return;
  bool changed = false;
  auto scheduled = ScheduleSegment(segment, &changed);
  out.insert(out.end(), scheduled.begin(), scheduled.end());
  segment.clear();
};
```

---

## Key Invariants and Checks

### 1. Resource Constraint Invariant

**Invariant:** At any moment, each pipeline pair `(SRC, DST)` has at most `kMaxEventIds` live events.

**Check Points:**
- `PredictAfterScheduling`: Check before prediction
- `INTERNAL_CHECK(pred >= 0)`: Ensure release doesn't make count negative

### 2. State Consistency

**Invariant:** `live_by_pair_` and `live_incoming_cross_by_pair_` must stay synchronized.

**Check:**
```cpp
INTERNAL_CHECK(it->second >= cnt)
    << "Attempted to release more events than live for pipe pair";
```

### 3. Topological Order Guarantee

**Invariant:** Output scheduling must satisfy all dependencies (topological order).

**Guarantee Mechanism:** Kahn algorithm only schedules statements with `indegree == 0`.

---

## Example Scenario

### Input Code

```python
A = compute_on_M(...)  # Pipeline M
B = compute_on_V(A)    # Pipeline V, depends on A (cross-pipe)
C = compute_on_M(...)  # Pipeline M
D = compute_on_V(C)    # Pipeline V, depends on C (cross-pipe)
E = compute_on_V(B, D) # Pipeline V, depends on B and D
```

### Dependency Graph

```
A(M) → B(V)
       ↓
C(M) → D(V) → E(V)
```

**Cross-pipe edges:** A→B, C→D

### Original Schedule Live Events

| Time | Execute | Live Events | (M→V) Count |
|------|---------|-------------|-------------|
| 1    | A       | {A→B}       | 1           |
| 2    | B       | {C→D}       | 1           |
| 3    | C       | {C→D}       | 1           |
| 4    | D       | {}          | 0           |
| 5    | E       | {}          | 0           |

**Peak:** 1

### Optimized Schedule

```
A(M), C(M), B(V), D(V), E(V)
```

| Time | Execute | Live Events | (M→V) Count |
|------|---------|-------------|-------------|
| 1    | A       | {A→B}       | 1           |
| 2    | C       | {A→B, C→D}  | 2           |
| 3    | B       | {C→D}       | 1           |
| 4    | D       | {}          | 0           |
| 5    | E       | {}          | 0           |

**Peak:** 2

**Advantage:** Although peak increases slightly, reduced pipeline switches may improve overall throughput.

### Concrete Example

```bash
================================================================================
IR BEFORE OutOfOrderSchedulerPass:
================================================================================
@pl.function
def test_event_limit_fix(input_0: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_1: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_2: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_3: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_4: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_5: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_6: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_7: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_8: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_9: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_10: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_11: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], output: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)]):
    load_0: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_0, 0, 0, 64, 64)
    load_1: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_1, 0, 0, 64, 64)
    load_2: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_2, 0, 0, 64, 64)
    load_3: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_3, 0, 0, 64, 64)
    load_4: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_4, 0, 0, 64, 64)
    load_5: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_5, 0, 0, 64, 64)
    load_6: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_6, 0, 0, 64, 64)
    load_7: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_7, 0, 0, 64, 64)
    load_8: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_8, 0, 0, 64, 64)
    load_9: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_9, 0, 0, 64, 64)
    load_10: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_10, 0, 0, 64, 64)
    load_11: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_11, 0, 0, 64, 64)
    compute_0: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_0, load_1)
    compute_1: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_1, load_2)
    compute_2: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_2, load_3)
    compute_3: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_3, load_4)
    compute_4: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_4, load_5)
    compute_5: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_5, load_6)
    compute_6: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_6, load_7)
    compute_7: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_7, load_8)
    compute_8: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_8, load_9)
    compute_9: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_9, load_10)
    compute_10: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_10, load_11)
    agg_1: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(compute_0, compute_1)
    agg_2: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_1, compute_2)
    agg_3: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_2, compute_3)
    agg_4: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_3, compute_4)
    agg_5: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_4, compute_5)
    agg_6: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_5, compute_6)
    agg_7: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_6, compute_7)
    agg_8: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_7, compute_8)
    agg_9: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_8, compute_9)
    agg_10: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_9, compute_10)
    store_result: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.store(agg_10, 0, 0, 64, 64, output)
    return store_result

!!! InsertSyncPass FAILED without scheduler: Out of hardware event IDs (max 8). Deadlock or resource exhaustion.

C++ Traceback (most recent call last):
 File "./src/ir/op/block_ops/matmul.cpp", line 83
 File "./src/ir/transform/mutator.cpp", line 145
 File "./src/ir/transform/mutator.cpp", line 146
 File "./src/ir/transform/mutator.cpp", line 145
 File "./src/ir/transform/mutator.cpp", line 146
 File "./src/ir/transform/mutator.cpp", line 145
 File "./src/ir/transform/mutator.cpp", line 146
 File "./src/ir/transform/mutator.cpp", line 147
 File "./src/ir/transform/mutator.cpp", line 145
 File "./src/ir/transform/mutator.cpp", line 146
 File "./src/ir/transform/mutator.cpp", line 145
 File "./src/ir/transform/mutator.cpp", line 146
 File "./src/ir/transform/insert_sync_pass.cpp", line 294
 File "./src/ir/transform/mutator.cpp", line 34
 File "./include/pypto/ir/transform/base/functor.h", line 200
 File "./src/ir/transform/insert_sync_pass.cpp", line 230

2026-01-26 14:53:13.717 D | OutOfOrderSchedulerPass: scheduled segment size=34, worst_peak_live_events_per_pair=4

================================================================================
IR AFTER OutOfOrderSchedulerPass:
================================================================================
@pl.function
def test_event_limit_fix(input_0: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_1: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_2: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_3: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_4: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_5: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_6: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_7: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_8: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_9: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_10: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_11: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], output: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)]):
    load_0: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_0, 0, 0, 64, 64)
    load_11: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_11, 0, 0, 64, 64)
    load_1: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_1, 0, 0, 64, 64)
    compute_0: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_0, load_1)
    load_2: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_2, 0, 0, 64, 64)
    compute_1: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_1, load_2)
    agg_1: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(compute_0, compute_1)
    load_3: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_3, 0, 0, 64, 64)
    compute_2: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_2, load_3)
    agg_2: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_1, compute_2)
    load_4: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_4, 0, 0, 64, 64)
    compute_3: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_3, load_4)
    agg_3: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_2, compute_3)
    load_5: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_5, 0, 0, 64, 64)
    compute_4: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_4, load_5)
    agg_4: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_3, compute_4)
    load_6: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_6, 0, 0, 64, 64)
    compute_5: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_5, load_6)
    agg_5: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_4, compute_5)
    load_7: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_7, 0, 0, 64, 64)
    compute_6: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_6, load_7)
    agg_6: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_5, compute_6)
    load_8: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_8, 0, 0, 64, 64)
    compute_7: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_7, load_8)
    agg_7: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_6, compute_7)
    load_9: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_9, 0, 0, 64, 64)
    compute_8: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_8, load_9)
    agg_8: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_7, compute_8)
    load_10: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_10, 0, 0, 64, 64)
    compute_9: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_9, load_10)
    compute_10: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_10, load_11)
    agg_9: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_8, compute_9)
    agg_10: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_9, compute_10)
    store_result: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.store(agg_10, 0, 0, 64, 64, output)
    return store_result

================================================================================
IR AFTER OutOfOrderSchedulerPass + InsertSyncPass:
================================================================================
@pl.function
def test_event_limit_fix(input_0: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_1: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_2: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_3: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_4: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_5: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_6: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_7: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_8: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_9: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_10: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], input_11: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)], output: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.DDR, 0, 16384)]):
    load_0: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_0, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    load_11: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_11, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=1)
    load_1: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_1, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=2)
    compute_0: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_0, load_1)
    load_2: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_2, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    compute_1: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_1, load_2)
    system.bar_v()
    system.bar_v()
    agg_1: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(compute_0, compute_1)
    load_3: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_3, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    compute_2: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_2, load_3)
    system.bar_v()
    system.bar_v()
    agg_2: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_1, compute_2)
    load_4: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_4, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    compute_3: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_3, load_4)
    system.bar_v()
    system.bar_v()
    agg_3: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_2, compute_3)
    load_5: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_5, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    compute_4: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_4, load_5)
    system.bar_v()
    system.bar_v()
    agg_4: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_3, compute_4)
    load_6: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_6, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    compute_5: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_5, load_6)
    system.bar_v()
    system.bar_v()
    agg_5: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_4, compute_5)
    load_7: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_7, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    compute_6: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_6, load_7)
    system.bar_v()
    system.bar_v()
    agg_6: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_5, compute_6)
    load_8: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_8, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    compute_7: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_7, load_8)
    system.bar_v()
    system.bar_v()
    agg_7: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_6, compute_7)
    load_9: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_9, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    compute_8: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_8, load_9)
    system.bar_v()
    system.bar_v()
    agg_8: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_7, compute_8)
    load_10: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.load(input_10, 0, 0, 64, 64)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=0)
    system.sync_src(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=3)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=0)
    compute_9: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_9, load_10)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=2)
    system.sync_dst(, set_pipe=1, wait_pipe=4, event_id=1)
    compute_10: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(load_10, load_11)
    system.bar_v()
    system.bar_v()
    agg_9: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_8, compute_9)
    system.bar_v()
    system.bar_v()
    agg_10: pl.Tile[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.add(agg_9, compute_10)
    system.sync_src(, set_pipe=4, wait_pipe=2, event_id=0)
    system.sync_dst(, set_pipe=4, wait_pipe=2, event_id=0)
    store_result: pl.Tensor[[64, 64], pl.FP32, memref=pl.MemRef(pl.MemorySpace.UB, 0, 16384)] = block.store(agg_10, 0, 0, 64, 64, output)
    return store_result
PASSED
```

---

## Performance and Limitations

### Time Complexity

- **Build dependency graph:** O(n²) (n = number of statements)
- **Kahn scheduling:** O(n × |ready| × |strategies|)
  - Worst case: O(n² × 3)

### Space Complexity

- **Dependency edges:** O(n²)
- **Live events map:** O(number of pipeline pairs × n)

### Limitations

1. **Only schedule straight-line code:** Control flow acts as barriers
2. **Conservative dependency analysis:** Based on MemRef, may be overly conservative
3. **Hardcoded event limit:** `kMaxEventIds = 8`, not configurable
4. **Falls back to best-effort:** Cannot guarantee constraint satisfaction always

---

## Logging and Debugging

### Key Logs

```cpp
// Strategy recovery success
LOG_WARN << "Recovered feasible schedule with strategy="
         << StrategyName(s);

// Fall back to relaxed mode
LOG_WARN << "Cannot find schedule satisfying event limit <= "
         << kMaxEventIds << ", falling back to best-effort";

// Scheduling result
LOG_DEBUG << "Scheduled segment size=" << n
          << ", worst_peak_live_events_per_pair=" << worst_peak;
```

### Debugging Tips

1. **Check pipe allocation:** Verify `GetStmtPipe` returns correct pipeline type
2. **Verify dependency graph:** Print `edges` to check RAW/WAW/WAR correctness
3. **Trace live events:** Add logs to `ReleaseIncoming/AllocateOutgoing`
4. **Compare scheduling results:** Compare `order` with original order

---

## Summary

`OutOfOrderSchedulerPass` optimizes cross-pipe synchronization through these steps:

1. **Identify reorderable segments:** Group consecutive compute statements
2. **Build dependency graph:** RAW/WAW/WAR analysis based on memory references
3. **Kahn + resource constraint scheduling:** On top of topological order, select order minimizing cross-pipe event pressure
4. **Multi-strategy fault tolerance:** Try multiple heuristic strategies, fall back to best-effort if needed
