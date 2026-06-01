# Auto Deps P2 Region-Aware Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce conservative manual-scope compiler dependency edges for statically disjoint `tensor.slice` windows while preserving conservative correctness for symbolic or unknown regions.

**Architecture:** Extend `AutoDeriveTaskDependencies` with a local `AccessRegion` carried alongside each storage root. The pass records rectangular constant slice boxes relative to the traced storage root, treats unknown/symbolic regions as full storage, and skips RAW/WAR/WAW edges only when two regions are proven disjoint.

**Tech Stack:** C++17 IR pass code, PyPTO Python DSL unit tests, pass documentation in `docs/en/dev/passes/` and `docs/zh-cn/dev/passes/`.

---

## Files

- Modify: `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`
  - Add `AccessRegion`.
  - Track region lineage for `tensor.slice`, direct aliases, `tensor.assemble`, tuple outputs, loop carries, and same-root `if/yield`.
  - Replace root-only hazard aliasing with `RegionsMayOverlap`.
- Modify: `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`
  - Update the old conservative disjoint-slice test to expect no compiler edge.
  - Add overlap and symbolic-offset coverage.
- Modify: `docs/en/dev/passes/34-auto_derive_task_dependencies.md`
  - Document region-aware static slice boxes and conservative symbolic fallback.
- Modify: `docs/zh-cn/dev/passes/34-auto_derive_task_dependencies.md`
  - Keep the Chinese pass doc aligned with the English behavior.

## Task 1: Region Tests

**Files:**

- Modify: `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`

- [ ] **Step 1: Rename the conservative disjoint test and change the expected result**

Change `test_disjoint_slices_are_conservatively_treated_as_overlap` to:

```python
def test_static_disjoint_slices_do_not_add_compiler_edge(self):
    ...
    assert _compiler_edges(consume_call) == []
```

This verifies the P2 target: two constant windows `[0, 32)` and `[32, 64)` over the same root are proven disjoint.

- [ ] **Step 2: Add an overlapping static slice test**

Add this test after the disjoint test:

```python
def test_static_overlapping_slices_add_compiler_edge(self):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def fill(
            self,
            out: pl.Out[pl.Tensor[[32], pl.FP32]],
        ) -> pl.Tensor[[32], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
            with pl.manual_scope():
                left = scratch[0:32]
                mid = scratch[16:48]
                _produced, producer_tid = pl.submit(self.fill, left)
                out, _ = pl.submit(self.consume, mid)
            return out

    out = _run_auto_deps(Prog)
    consume_call = _user_calls(out, "consume")[0]
    edges = _compiler_edges(consume_call)
    assert len(edges) == 1
    assert edges[0].name_hint == "producer_tid"
```

- [ ] **Step 3: Add a symbolic slice fallback test**

Add:

```python
def test_symbolic_slice_offset_stays_conservative(self):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def fill(
            self,
            out: pl.Out[pl.Tensor[[32], pl.FP32]],
        ) -> pl.Tensor[[32], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            scratch: pl.Tensor[[64], pl.FP32],
            offset: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[32], pl.FP32]:
            with pl.manual_scope():
                left = scratch[0:32]
                dynamic = pl.slice(scratch, [32], [offset])
                _produced, producer_tid = pl.submit(self.fill, left)
                out, _ = pl.submit(self.consume, dynamic)
            return out

    out = _run_auto_deps(Prog)
    consume_call = _user_calls(out, "consume")[0]
    edges = _compiler_edges(consume_call)
    assert len(edges) == 1
    assert edges[0].name_hint == "producer_tid"
```

- [ ] **Step 4: Run the focused tests and observe failure before implementation**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py -q
```

Expected before implementation: the renamed disjoint test fails because the current pass still adds an edge.

## Task 2: Region-Aware Storage Analysis

**Files:**

- Modify: `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`

- [ ] **Step 1: Add region data structures**

Add near `StorageAccess`:

```cpp
struct AccessRegion {
  bool known = false;
  std::vector<int64_t> offsets;
  std::vector<int64_t> shape;
};

struct StorageLocation {
  const Var* root = nullptr;
  AccessRegion region;
};
```

Change `StorageAccess` to store `StorageLocation location` instead of `const Var* root`.

- [ ] **Step 2: Add helpers for constant tuple parsing and region overlap**

Add helpers:

```cpp
std::optional<std::vector<int64_t>> ConstIntTupleValues(const ExprPtr& expr);
AccessRegion FullRegion();
AccessRegion SliceRegion(const AccessRegion& parent, const ExprPtr& shape, const ExprPtr& offset);
bool RegionsMayOverlap(const AccessRegion& lhs, const AccessRegion& rhs);
```

Rules:

- Non-`MakeTuple`, non-`ConstInt`, rank mismatch, and integer overflow return unknown/full.
- Unknown region overlaps everything.
- Known rank mismatch overlaps conservatively.
- Rectangles are disjoint if any dimension has non-overlapping half-open intervals.

- [ ] **Step 3: Change root analysis to location analysis**

Replace:

```cpp
std::unordered_map<const Var*, const Var*> roots_;
const Var* ResolveExpr(const ExprPtr& expr) const;
void RegisterVarRoot(const VarPtr& var, const Var* root);
```

with:

```cpp
std::unordered_map<const Var*, StorageLocation> locations_;
StorageLocation ResolveExpr(const ExprPtr& expr) const;
void RegisterVarLocation(const VarPtr& var, StorageLocation location);
```

Register tensor parameters and `tensor.create` with `{var.get(), FullRegion()}`. Direct aliases and tuple outputs copy the full `StorageLocation`; loop return variables and same-root `if/yield` joins widen to unknown when the yielded region is not statically identical.

- [ ] **Step 4: Track `tensor.slice` regions**

For `tensor.slice`, resolve the parent location, then register:

```cpp
RegisterVarLocation(op->var_, StorageLocation{parent.root, SliceRegion(parent.region, call->args_[1], call->args_[2])});
```

For `tensor.assemble`, keep the destination location from `args_[0]` conservatively.

- [ ] **Step 5: Apply region overlap in hazard detection**

Replace the current root-only check:

```cpp
if (access.root == nullptr || !storage_ || !storage_->MayAlias(access.root, prior.root)) continue;
```

with:

```cpp
if (!storage_ || !storage_->MayAlias(access.location.root, prior.location.root)) continue;
if (!RegionsMayOverlap(access.location.region, prior.location.region)) continue;
```

This keeps MemRef alias behavior intact and only removes edges for same-root or aliasing roots with proven disjoint constant regions.

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py -q
```

Expected: all auto-deps tests pass.

## Task 3: Documentation

**Files:**

- Modify: `docs/en/dev/passes/34-auto_derive_task_dependencies.md`
- Modify: `docs/zh-cn/dev/passes/34-auto_derive_task_dependencies.md`

- [ ] **Step 1: Update English algorithm bullets**

Document that the pass now tracks constant rectangular `tensor.slice` regions relative to storage roots and skips dependency edges for proven disjoint regions.

- [ ] **Step 2: Update Chinese doc**

Mirror the English behavior in Chinese. Preserve file path and pass ordering.

- [ ] **Step 3: Run doc sanity checks**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

## Task 4: Final Verification

- [ ] **Step 1: Format and lint Python tests**

Run:

```bash
ruff check tests/ut/ir/transforms/test_auto_derive_task_dependencies.py
ruff format --check tests/ut/ir/transforms/test_auto_derive_task_dependencies.py
```

- [ ] **Step 2: Run focused pytest**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_auto_derive_task_dependencies.py -q
```

- [ ] **Step 3: Report local limitations**

If pytest cannot import the native extension in this Windows checkout, report that clearly and leave the remote-validation task to cover runtime execution.
