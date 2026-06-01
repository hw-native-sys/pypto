# Auto Deps P1 Storage Lineage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `AutoDeriveTaskDependencies` P1 coverage for stable storage lineage through `IfStmt` yield results and MemRef-backed shaped values, without changing runtime contracts.

**Architecture:** Keep the pass internal and conservative. `StorageRootAnalysis` continues to produce storage identities for call arguments, but now records MemRef alias families and maps `IfStmt.return_vars_` when both branches yield the same storage root. Dependency emission still writes only `Call.attrs["compiler_manual_dep_edges"]`.

**Tech Stack:** C++17 IR pass in `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`, Python pytest regression coverage in `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`.

---

## File Structure

- Modify `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`
  - Add a regression test for `IfStmt.return_vars_` lineage from matching branch yields.
  - Add a regression test for MemRef-backed tensor aliases when two variables have different IR Vars but overlapping `MemRef` ranges.
- Modify `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`
  - Include `pypto/ir/memref.h`.
  - Extend `StorageRootAnalysis` with a MemRef alias-family map keyed by `MemRef::MayAlias`.
  - Teach `VisitStmt_(IfStmtPtr)` to derive return-var roots from branch trailing `YieldStmt` values only when both branch roots match.

---

## Task 1: Add P1 Regression Tests

**Files:**

- Modify: `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`

- [ ] **Step 1: Add failing tests**

Add these tests inside `class TestAutoDeriveTaskDependencies`:

```python
    def test_if_yield_return_var_keeps_storage_lineage(self):
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
                scratch: pl.Tensor[[64], pl.FP32],
                cond: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced, producer_tid = pl.submit(self.fill, scratch)
                    if cond:
                        selected = pl.yield_(produced)
                    else:
                        selected = pl.yield_(produced)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_memref_may_alias_adds_compiler_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                x: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)],
                right: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)]:
                with pl.manual_scope():
                    _produced, producer_tid = pl.submit(self.fill, left)
                    out, _ = pl.submit(self.consume, right)
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py::TestAutoDeriveTaskDependencies::test_if_yield_return_var_keeps_storage_lineage tests/ut/ir/transforms/test_auto_derive_task_dependencies.py::TestAutoDeriveTaskDependencies::test_memref_may_alias_adds_compiler_edge -q
```

Expected: both tests fail because no compiler edge is derived.

---

## Task 2: Implement If/Yield Return-Var Lineage

**Files:**

- Modify: `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`

- [ ] **Step 1: Add trailing yield helper and IfStmt visitor**

Before the existing `VisitStmt_(const ForStmtPtr& op)` in `StorageRootAnalysis`, add:

```cpp
  static YieldStmtPtr GetTrailingYield(const StmtPtr& stmt) {
    if (auto yield = As<YieldStmt>(stmt)) return yield;
    auto seq = As<SeqStmts>(stmt);
    if (!seq || seq->stmts_.empty()) return nullptr;
    return As<YieldStmt>(seq->stmts_.back());
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    if (!op || op->return_vars_.empty() || !op->else_body_.has_value()) return;

    auto then_yield = GetTrailingYield(op->then_body_);
    auto else_yield = GetTrailingYield(op->else_body_.value());
    if (!then_yield || !else_yield) return;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (i >= then_yield->value_.size() || i >= else_yield->value_.size()) break;
      const Var* then_root = ResolveExpr(then_yield->value_[i]);
      const Var* else_root = ResolveExpr(else_yield->value_[i]);
      if (!then_root || then_root != else_root) continue;
      roots_[op->return_vars_[i].get()] = then_root;
    }
  }
```

- [ ] **Step 2: Run If/yield test to verify GREEN**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py::TestAutoDeriveTaskDependencies::test_if_yield_return_var_keeps_storage_lineage -q
```

Expected: pass.

---

## Task 3: Implement MemRef Alias Families

**Files:**

- Modify: `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`

- [ ] **Step 1: Include MemRef header**

Add:

```cpp
#include "pypto/ir/memref.h"
```

- [ ] **Step 2: Add MemRef helpers and alias registration**

Inside `StorageRootAnalysis`, add helper methods:

```cpp
  void RegisterVarRoot(const VarPtr& var, const Var* root) {
    if (!var || !root) return;
    roots_[var.get()] = root;
    if (const auto memref = GetShapedMemRef(var->GetType())) {
      RegisterMemRefAlias(var.get(), memref);
    }
  }

  static MemRefPtr GetShapedMemRef(const TypePtr& type) {
    auto shaped = As<ShapedType>(type);
    if (!shaped || !shaped->memref_.has_value()) return nullptr;
    return shaped->memref_.value();
  }

  void RegisterMemRefAlias(const Var* var, const MemRefPtr& memref) {
    if (!var || !memref) return;
    for (const auto& [known_var, known_memref] : memrefs_) {
      if (!known_var || !known_memref || !MemRef::MayAlias(memref, known_memref)) continue;
      const Var* root = ResolveExpr(known_var);
      if (root) {
        roots_[var] = root;
        return;
      }
    }
    memrefs_[var] = memref;
  }
```

Add a new field:

```cpp
  std::unordered_map<const Var*, MemRefPtr> memrefs_;
```

Replace direct writes such as:

```cpp
roots_[param.get()] = param.get();
```

with:

```cpp
RegisterVarRoot(param, param.get());
```

Do the same for assignment, loop, while, if return vars, and call output roots.

- [ ] **Step 3: Run MemRef test to verify GREEN**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py::TestAutoDeriveTaskDependencies::test_memref_may_alias_adds_compiler_edge -q
```

Expected: pass.

---

## Task 4: Regression Sweep

**Files:**

- Test: `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`
- Test: `tests/ut/ir/transforms/test_pass_manager.py`
- Test: `tests/ut/codegen/test_orchestration_codegen.py`

- [ ] **Step 1: Run focused auto-deps tests**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py -q
```

Expected: all tests in that file pass.

- [ ] **Step 2: Run pipeline and lowering smoke tests**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_pass_manager.py tests/ut/codegen/test_orchestration_codegen.py -q
```

Expected: all selected tests pass.

---

## Self-Review

- Spec coverage: P1 If/yield lineage and MemRef alias coverage are covered by Tasks 1-3. Group/Spmd effective directions are already implemented in P0 via `ComputeGroupEffectiveDirections`; no production change is planned for that part in this slice.
- Placeholder scan: no placeholders remain.
- Type consistency: C++ snippets use existing `YieldStmtPtr`, `SeqStmts`, `ShapedType`, `MemRefPtr`, and `MemRef::MayAlias` APIs.
