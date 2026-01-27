---
name: feature-dev
description: Develop new PyPTO IR features including operators, passes, and transformations. Reads documentation first, implements changes following project patterns, and integrates testing. Use when adding IR operators, creating passes, implementing transformations, or developing new IR features.
---

# PyPTO Feature Development

Develop new IR features by reading docs first, then implementing with project patterns.

## Read Documentation First

Before coding, read the relevant doc:
- Adding operators → `docs/dev/03-operator_registration.md`
- Adding passes → `docs/dev/08-pass_manager.md`
- IR concepts → `docs/dev/00-ir_definition.md`
- IR builder → `docs/dev/06-ir_builder.md`

## Workflow

1. Read relevant documentation
2. Implement C++ (header + source)
3. Add Python bindings if needed
4. Create Python wrapper
5. Use `testing` skill to verify

## Adding Operators

**Operator Categories:**
- `TensorOp`: N-dimensional tensors (`src/ir/op/tensor_ops/`)
- `BlockOp`: Tile operations (`src/ir/op/block_ops/`)
- `SyncOp`: Synchronization (`src/ir/op/sync_ops/`)

**C++ Implementation** in `src/ir/op/[category]/`:

```cpp
TypePtr DeduceMyOpType(const std::vector<ExprPtr>& args,
                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "my_op requires 2 arguments";
  auto lhs = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto rhs = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

  auto result_dtype = PromoteDataTypes(lhs->dtype_, rhs->dtype_);
  auto broadcast_result = BroadcastShapes(lhs->shape_, rhs->shape_);
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

REGISTER_OP("tensor.my_op")
    .set_op_category("TensorOp")
    .set_description("Description")
    .add_argument("lhs", "Left tensor")
    .add_argument("rhs", "Right tensor")
    .f_deduce_type(DeduceMyOpType);
```

**Python Wrapper** in `python/pypto/ir/op/`:

```python
def my_op(lhs: Expr, rhs: Expr, flag: bool = False) -> Call:
    """Operation description."""
    kwargs = {"flag": flag} if flag else {}
    return _ir_core.create_op_call("tensor.my_op", [lhs, rhs], kwargs, Span.unknown())
```

## Adding Passes

**C++ Header** in `include/pypto/ir/transform/`:

```cpp
#pragma once
#include "pypto/ir/transform/base/pass.h"

class MyPass : public Pass {
 public:
  FunctionPtr Run(const FunctionPtr& func) override;
};
```

**C++ Implementation** in `src/ir/transform/`:

```cpp
FunctionPtr MyPass::Run(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MyPass cannot run on null function";
  // Transform using IRMutator methods
  return transformed_func;
}
```

**Python Bindings** in `python/bindings/modules/pass.cpp`:

```cpp
nb::class_<MyPass, Pass>(passes, "MyPass", "Description")
    .def(nb::init<>(), "Create MyPass");
```

**Register in PassManager** in `python/pypto/ir/pass_manager.py`:

```python
OptimizationStrategy.Custom2: [
    ("MyPass", lambda: passes.MyPass()),
]
```

## Key Utilities

**Type checking:**
```cpp
if (IsA<TensorType>(expr->GetType())) { /* ... */ }
if (auto tensor = As<TensorType>(expr->GetType())) { /* ... */ }
```

**Broadcasting and promotion:**
```cpp
auto result = BroadcastShapes(shape1, shape2);
auto dtype = PromoteDataTypes(dtype1, dtype2);
```

**Validation:**
```cpp
CHECK(args.size() == 2) << "Expected 2 args, got " << args.size();
auto tensor = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
CHECK(tensor) << "Expected TensorType, got " << args[0]->GetType()->TypeName();
```

## File Locations

| Component | Location |
|-----------|----------|
| Operators | `src/ir/op/[tensor\|block\|sync]_ops/` |
| Passes | `src/ir/transform/` |
| Python wrappers | `python/pypto/ir/op/` |
| Bindings | `python/bindings/modules/` |
| Tests | `tests/ut/ir/` |

## Key Patterns

**Operators:**
- Args for Expr (tensors/tiles)
- Kwargs for metadata (flags, dtypes, modes)
- Use `BroadcastShapes()` and `PromoteDataTypes()`

**Passes:**
- Extend `Pass`, implement `Run(FunctionPtr)`
- Use `IRMutator` for transformations
- Always return new IR nodes (immutable)

**Testing:**
Use `testing` skill after implementation.

**Building:**
Use `pypto-development` skill for CMake and environment setup.

## Quick Reference

| Task | Doc | Files |
|------|-----|-------|
| Add operator | `03-operator_registration.md` | `src/ir/op/[category]/` |
| Add pass | `08-pass_manager.md` | `src/ir/transform/` |
| Build IR | `06-ir_builder.md` | `python/pypto/ir/builder.py` |
