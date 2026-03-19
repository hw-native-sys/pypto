# Pass Complexity Requirements

## Core Principle

**All IR passes must have at most O(N log N) time complexity, where N is the size of the IR (number of nodes/statements).**

The log N factor is acceptable only when it comes from ordered map/set lookups or similar data structure operations. The dominant traversal must remain O(N).

## Allowed Complexity

| Pattern | Complexity | Allowed |
| ------- | ---------- | ------- |
| Single IR traversal with map lookups | O(N log N) | Yes |
| Single IR traversal, constant-time work per node | O(N) | Yes |
| Multiple independent traversals (fixed number) | O(N) | Yes |
| Nested full scans over the same/global IR node collection | O(N^2) | **No** |
| Repeated linear scans for lookups | O(N^2) | **No** |
| Fixed-point iteration without convergence bound | Unbounded | **No** |

## Why

Compiler passes run on every compilation. Quadratic passes become unusable on large programs — a 10x increase in IR size causes a 100x slowdown. The O(N log N) bound ensures passes scale predictably.

## How to Stay Within O(N log N)

### Use Maps for Lookups, Not Linear Scans

```cpp
// ❌ O(N²) — linear scan inside traversal
void BadPass::VisitStmt_(const AssignStmtPtr& op) {
  for (const auto& other : all_stmts_) {  // O(N) per visit
    if (other->var_ == op->var_) { ... }
  }
}

// ✅ O(N log N) — map lookup inside traversal
void GoodPass::VisitStmt_(const AssignStmtPtr& op) {
  auto it = var_to_stmt_.find(op->var_);  // O(log N) per visit
  if (it != var_to_stmt_.end()) { ... }
}
```

### Build Index First, Then Traverse

```cpp
// ✅ O(N log N) — build map O(N log N), then traverse O(N log N)
void GoodPass::Run(const ProgramPtr& prog) {
  // Phase 1: Build index — O(N log N)
  for (auto& stmt : prog->stmts_) {
    index_.emplace(stmt->key(), stmt);  // O(log N) per insert
  }
  // Phase 2: Transform — O(N log N)
  for (auto& stmt : prog->stmts_) {
    auto it = index_.find(stmt->dep());  // O(log N) per lookup
    INTERNAL_CHECK(it != index_.end()) << "missing dependency";
    Transform(stmt, it->second);
  }
}
```

### Avoid Nested Node Iteration

```python
# ❌ O(N²) — nested loops over IR nodes
for stmt in program.stmts:
    for other in program.stmts:  # Quadratic!
        if depends_on(stmt, other):
            ...

# ✅ O(N + E) — build dependency map, then single traversal
# where E is total dependency edges (bounded by O(N) in most IR passes)
dep_map: dict[str, list[Stmt]] = {}
for stmt in program.stmts:
    for dep in stmt.dependencies:
        dep_map.setdefault(dep, []).append(stmt)

for stmt in program.stmts:
    for dependent in dep_map.get(stmt.name, []):  # O(N + E) total across both loops
        ...
```

## Review Checklist

When writing or reviewing a pass:

- [ ] No nested full scans over the same/global IR node collection
- [ ] All lookups use indexed structures — map, set, unordered_map, or vector (not linear scans)
- [ ] Fixed-point loops have a proven convergence bound
- [ ] Overall complexity is O(N log N) or better

## Exceptions

If a pass genuinely requires super-linear work (e.g., graph algorithms with known higher bounds), document the complexity and justification in a comment at the top of the pass. Get explicit approval before merging.
