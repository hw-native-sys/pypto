# Auto Deps P3 Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `AutoDeriveTaskDependencies` so manual-scope dependency derivation handles finite root sets from control flow and falls back to runtime auto tracking when static dependency derivation cannot be encoded safely.

**Architecture:** Replace the pass-local single-root `StorageLocation` with a finite set of root/region alternatives. `IfStmt`, `ForStmt`, and `WhileStmt` return variables merge yield-derived alternatives instead of choosing a single root. If a manual scope contains a statically unresolved hazard that cannot be encoded as fixed TaskId deps, rewrite that entire `RuntimeScopeStmt` to `manual=false` so the runtime OverlapMap/TensorMap owns the scope consistently.

**Tech Stack:** C++17 IR transform in `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`, pytest transform tests in `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`, English and Chinese developer docs.

---

## File Structure

- Modify `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`
  - Add `StorageAlternative` and make `StorageLocation` hold `std::vector<StorageAlternative>`.
  - Add helpers to merge finite root sets, widen regions, slice all alternatives, and compare all alternatives for alias/overlap.
  - Update `IfStmt` lineage to merge different branch roots into a root set.
  - Update `ForStmt` and `WhileStmt` lineage to merge init and trailing-yield roots into a root set.
  - Update manual-scope dependency emission to expand root-set access alternatives and to rewrite a whole manual scope to auto scope when a required prior TaskId is missing.
- Modify `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`
  - Add tests for `IfStmt` different-root lineage.
  - Add tests for loop body-yield lineage.
  - Add tests for whole-scope fallback.
- Modify `docs/en/dev/passes/34-auto_derive_task_dependencies.md`
  - Document root-set lineage and fallback behavior.
- Modify `docs/zh-cn/dev/passes/34-auto_derive_task_dependencies.md`
  - Keep the Chinese pass doc aligned.
- Modify `docs/en/dev/proposals/auto_task_dependencies.md`
  - Mark P3 as the implementation target and clarify fallback semantics.
- Modify `docs/zh-cn/dev/proposals/auto_task_dependencies.md`
  - Keep the Chinese proposal aligned without changing unrelated content.

---

### Task 1: Add P3 Regression Tests

**Files:**

- Modify: `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`

- [ ] **Step 1: Add IfStmt different-root test**

Add this method after `test_if_yield_return_var_keeps_storage_lineage`:

```python
    def test_if_yield_different_roots_adds_edges_for_both_possible_producers(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32],
                right: pl.Tensor[[64], pl.FP32],
                cond: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced_left, left_tid = pl.submit(self.fill, left)
                    produced_right, right_tid = pl.submit(self.fill, right)
                    if cond:
                        selected = pl.yield_(produced_left)
                    else:
                        selected = pl.yield_(produced_right)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert [edge.name_hint for edge in edges] == ["left_tid", "right_tid"]
```

- [ ] **Step 2: Add loop body-yield root-set test**

Add this method near the IfStmt tests:

```python
    def test_loop_yield_different_root_adds_edges_for_init_and_yield_roots(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32],
                right: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced_left, left_tid = pl.submit(self.fill, left)
                    produced_right, right_tid = pl.submit(self.fill, right)
                    for _i, (selected_iter,) in pl.range(0, 4, init_values=(produced_left,)):
                        selected = pl.yield_(produced_right)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert [edge.name_hint for edge in edges] == ["left_tid", "right_tid"]
```

- [ ] **Step 3: Add whole-scope fallback test**

Add this helper and test near the bottom of the file:

```python
def _runtime_scopes(program: ir.Program) -> list[ir.RuntimeScopeStmt]:
    scopes: list[ir.RuntimeScopeStmt] = []

    class Collector(ir.IRVisitor):
        def visit_runtime_scope_stmt(self, op):
            scopes.append(op)
            super().visit_runtime_scope_stmt(op)

    Collector().visit_program(program)
    return scopes
```

```python
    def test_unencodable_manual_scope_hazard_falls_back_to_auto_scope(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced = self.fill(scratch)
                    out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False
        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []
```

- [ ] **Step 4: Run tests and confirm initial failures**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py -k "if_yield_different_roots or loop_yield_different_root or unencodable_manual_scope" -v
```

Expected before implementation: root-set tests fail with missing compiler edges, and fallback test fails because the pass still raises `ValueError` for the unbound producer TaskId.

---

### Task 2: Implement Finite Root-Set Storage Locations

**Files:**

- Modify: `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`

- [ ] **Step 1: Replace single-root storage shape**

Change:

```cpp
struct StorageLocation {
  const Var* root = nullptr;
  AccessRegion region;
};
```

to:

```cpp
struct StorageAlternative {
  const Var* root = nullptr;
  AccessRegion region;
};

struct StorageLocation {
  std::vector<StorageAlternative> alternatives;
};
```

Add helper constructors:

```cpp
StorageLocation UnknownLocation() { return StorageLocation{}; }

StorageLocation SingleLocation(const Var* root, AccessRegion region) {
  if (!root) return UnknownLocation();
  return StorageLocation{{StorageAlternative{root, std::move(region)}}};
}

bool HasLocation(const StorageLocation& location) { return !location.alternatives.empty(); }
```

- [ ] **Step 2: Add merge/slice helpers**

Add helpers below `SameRegion`:

```cpp
void AppendAlternativeUnique(std::vector<StorageAlternative>* alternatives, StorageAlternative candidate) {
  if (!alternatives || !candidate.root) return;
  for (auto& existing : *alternatives) {
    if (existing.root == candidate.root) {
      if (!SameRegion(existing.region, candidate.region)) existing.region = UnknownRegion();
      return;
    }
  }
  alternatives->push_back(std::move(candidate));
}

StorageLocation MergeLocations(const StorageLocation& lhs, const StorageLocation& rhs) {
  StorageLocation merged;
  for (const auto& alternative : lhs.alternatives) {
    AppendAlternativeUnique(&merged.alternatives, alternative);
  }
  for (const auto& alternative : rhs.alternatives) {
    AppendAlternativeUnique(&merged.alternatives, alternative);
  }
  return merged;
}

StorageLocation UnknownRegionsFor(const StorageLocation& location) {
  StorageLocation widened;
  widened.alternatives.reserve(location.alternatives.size());
  for (const auto& alternative : location.alternatives) {
    widened.alternatives.push_back(StorageAlternative{alternative.root, UnknownRegion()});
  }
  return widened;
}

StorageLocation SliceLocation(const StorageLocation& parent, const ExprPtr& shape_expr, const ExprPtr& offset_expr) {
  StorageLocation sliced;
  sliced.alternatives.reserve(parent.alternatives.size());
  for (const auto& alternative : parent.alternatives) {
    AppendAlternativeUnique(&sliced.alternatives,
                            StorageAlternative{alternative.root, SliceRegion(alternative.region, shape_expr, offset_expr)});
  }
  return sliced;
}
```

- [ ] **Step 3: Update root registration and call output collection**

Use `SingleLocation(param.get(), FullRegion())`, `SingleLocation(op->var_.get(), FullRegion())`, `SliceLocation(parent, ...)`, and `HasLocation(location)` in `StorageRootAnalysis`.

In `RegisterVarLocation`, iterate over every alternative when recording root MemRefs:

```cpp
for (const auto& alternative : location.alternatives) {
  if (alternative.root) root_memrefs_.try_emplace(alternative.root, memref);
}
```

- [ ] **Step 4: Run compile smoke**

Run:

```bash
python -m py_compile tests/ut/ir/transforms/test_auto_derive_task_dependencies.py
```

Expected: Python compile passes. C++ build may still fail until Task 3 finishes.

---

### Task 3: Implement Control-Flow Root-Set Lineage

**Files:**

- Modify: `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`

- [ ] **Step 1: Update `IfStmt` lineage**

Replace the same-root-only block in `VisitStmt_(const IfStmtPtr& op)` with:

```cpp
auto then_location = ResolveExpr(then_yield->value_[i]);
auto else_location = ResolveExpr(else_yield->value_[i]);
auto merged = MergeLocations(then_location, else_location);
if (HasLocation(merged)) {
  RegisterVarLocation(op->return_vars_[i], std::move(merged));
}
```

- [ ] **Step 2: Update loop and while lineage**

For each iter arg:

1. Register the iter arg from `initValue_`.
2. Visit the body.
3. Read the trailing `YieldStmt`.
4. Merge the init location and yield location into the return var with unknown regions.

Use this shape for both `ForStmt` and `WhileStmt`:

```cpp
std::vector<StorageLocation> init_locations;
init_locations.reserve(op->iter_args_.size());
for (const auto& iter_arg : op->iter_args_) {
  auto location = ResolveExpr(iter_arg->initValue_);
  init_locations.push_back(location);
  if (HasLocation(location)) RegisterVarLocation(iter_arg, location);
}

IRVisitor::VisitStmt_(op);

auto yield = GetTrailingYield(op->body_);
for (size_t i = 0; i < op->iter_args_.size() && i < op->return_vars_.size(); ++i) {
  auto location = init_locations[i];
  if (yield && i < yield->value_.size()) {
    location = MergeLocations(location, ResolveExpr(yield->value_[i]));
  }
  location = UnknownRegionsFor(location);
  if (HasLocation(location)) RegisterVarLocation(op->return_vars_[i], std::move(location));
}
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py -k "if_yield_different_roots or loop_yield_different_root" -v
```

Expected: both root-set tests pass if the native extension is available. If local native extension is unavailable, record the import/build blocker and continue with compile/lint verification.

---

### Task 4: Implement Whole-Scope Fallback

**Files:**

- Modify: `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`

- [ ] **Step 1: Expand access alternatives**

Update `SummarizeAccesses` so each storage alternative becomes one `StorageAccess`:

```cpp
auto location = storage_ ? storage_->ResolveExpr(call->args_[i]) : StorageLocation{};
if (!HasLocation(location)) continue;
...
for (const auto& alternative : location.alternatives) {
  out.push_back(StorageAccess{alternative, *kind, nullptr});
}
```

Change `StorageAccess::location` to `StorageAlternative`.

- [ ] **Step 2: Rewrite unencodable manual scope to auto**

Add a `bool fallback_current_scope_` per manual scope. In `VisitStmt_(RuntimeScopeStmtPtr)`:

```cpp
prior_stack_.emplace_back();
fallback_stack_.push_back(false);
auto new_body = IRMutator::VisitStmt_(op->body_);
bool fallback = fallback_stack_.back();
fallback_stack_.pop_back();
prior_stack_.pop_back();
if (fallback || new_body.get() != op->body_.get()) {
  return std::make_shared<const RuntimeScopeStmt>(fallback ? false : op->manual_, op->name_hint_, new_body,
                                                 op->span_, op->leading_comments_, op->attrs_);
}
return op;
```

In dependency emission, replace the `CHECK(prior.task_id_var)` with:

```cpp
if (!prior.task_id_var) {
  fallback_stack_.back() = true;
  return call;
}
```

This fallback exits dependency synthesis for the current call; after the body finishes, the whole scope becomes auto, and any compiler deps emitted earlier in the same traversal are discarded by rebuilding from the original body if needed.

- [ ] **Step 3: Ensure fallback discards partial compiler attrs**

If a later call triggers fallback after earlier calls were mutated, rerun the body traversal without auto-dep mutation or return a rebuilt `RuntimeScopeStmt(false, ..., op->body_, ...)` using the original body. The expected code shape is:

```cpp
if (fallback) {
  return std::make_shared<const RuntimeScopeStmt>(false, op->name_hint_, op->body_, op->span_,
                                                 op->leading_comments_, op->attrs_);
}
```

- [ ] **Step 4: Run focused fallback test**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py -k "unencodable_manual_scope" -v
```

Expected: scope `manual` becomes `False` and the consumer call has no compiler edges.

---

### Task 5: Update Documentation

**Files:**

- Modify: `docs/en/dev/passes/34-auto_derive_task_dependencies.md`
- Modify: `docs/zh-cn/dev/passes/34-auto_derive_task_dependencies.md`
- Modify: `docs/en/dev/proposals/auto_task_dependencies.md`
- Modify: `docs/zh-cn/dev/proposals/auto_task_dependencies.md`

- [ ] **Step 1: Update pass algorithm**

In the English pass doc, update algorithm bullets to state:

```markdown
2. Preserve storage lineage through `IfStmt.return_vars` by merging finite
   branch root sets. Different roots are retained as alternatives, not dropped.
3. Preserve loop and while return lineage by merging the initial carried value
   with the trailing body `pl.yield_()` value, then widening regions to unknown
   because the final iteration count is control-flow dependent.
```

Add:

```markdown
If a manual scope contains a hazard whose prior producer TaskId is not
statically available, the pass rewrites that whole `RuntimeScopeStmt` to
`manual=false`. This returns the entire region to runtime OverlapMap/TensorMap
tracking instead of mixing partial compiler deps with runtime state.
```

- [ ] **Step 2: Update proposal P3**

Add a short status note under `## P3`:

```markdown
Implementation target for this phase: finite root-set lineage for `IfStmt`,
loop, and while return variables, plus whole-scope fallback to runtime tracking
when a required dependency cannot be encoded as fixed TaskId deps.
```

- [ ] **Step 3: Keep Chinese docs aligned**

Mirror the English behavioral updates in the corresponding Chinese docs.

---

### Task 6: Verify

**Files:**

- All modified files.

- [ ] **Step 1: Check whitespace**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 2: Compile Python test file**

Run:

```bash
python -m py_compile tests/ut/ir/transforms/test_auto_derive_task_dependencies.py
```

Expected: no output.

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py -k "auto_derive_task_dependencies or if_yield_different_roots or loop_yield_different_root or unencodable_manual_scope" -v
```

Expected: pass when native `pypto_core` is built. If blocked by local native extension availability, record the exact error.

- [ ] **Step 4: Run formatting/lint where available**

Run:

```bash
pre-commit run --files src/ir/transforms/auto_derive_task_dependencies_pass.cpp tests/ut/ir/transforms/test_auto_derive_task_dependencies.py docs/en/dev/passes/34-auto_derive_task_dependencies.md docs/zh-cn/dev/passes/34-auto_derive_task_dependencies.md docs/en/dev/proposals/auto_task_dependencies.md docs/zh-cn/dev/proposals/auto_task_dependencies.md
```

Expected: pass, except known `check-english-only` false positives on `docs/zh-cn` may need to be skipped if it scans translated docs.
