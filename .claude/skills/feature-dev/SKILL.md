---
name: feature-dev
description: Develop new PyPTO IR features including operators, passes, and transformations. Reads documentation first, implements changes following project patterns, and integrates testing. Use when adding IR operators, creating passes, implementing transformations, or developing new IR features.
---

# PyPTO Feature Development

Develop new IR features: read docs first, implement following project patterns.

## Read Documentation First

Before coding, read relevant docs:

- Operators → `docs/dev/05-operator_registration.md`
- Passes → `docs/dev/10-pass_manager.md`
- IR concepts → `docs/dev/00-ir_overview.md`
- IR builder → `docs/dev/08-ir_builder.md`

## Workflow

1. Read relevant documentation
2. Implement C++ (header + source)
3. Add Python bindings if needed
4. Create Python wrapper
5. Use `testing` skill to verify

## Adding Operators

**Categories**: TensorOp (`src/ir/op/tensor_ops/`), BlockOp (`src/ir/op/block_ops/`), SyncOp (`src/ir/op/sync_ops/`)

**C++ Implementation**:

```cpp
TypePtr DeduceMyOpType(const std::vector<ExprPtr>& args,
                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "my_op requires 2 arguments";
  auto result_dtype = PromoteDataTypes(lhs->dtype_, rhs->dtype_);
  auto broadcast_result = BroadcastShapes(lhs->shape_, rhs->shape_);
  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

REGISTER_OP("tensor.my_op")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left tensor")
    .add_argument("rhs", "Right tensor")
    .f_deduce_type(DeduceMyOpType);
```

**Python Wrapper** (`python/pypto/ir/op/`):

```python
def my_op(lhs: Expr, rhs: Expr, flag: bool = False) -> Call:
    """Operation description."""
    kwargs = {"flag": flag} if flag else {}
    return _ir_core.create_op_call("tensor.my_op", [lhs, rhs], kwargs, Span.unknown())
```

## Adding Passes

**C++ Header** (`include/pypto/ir/transform/`):

```cpp
#include "pypto/ir/transform/base/pass.h"

class MyPass : public Pass {
 public:
  FunctionPtr Run(const FunctionPtr& func) override;
};
```

**C++ Implementation** (`src/ir/transform/`):

```cpp
FunctionPtr MyPass::Run(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MyPass cannot run on null function";
  // Transform using IRMutator methods
  return transformed_func;
}
```

**Python Bindings** (`python/bindings/modules/pass.cpp`):

```cpp
nb::class_<MyPass, Pass>(passes, "MyPass", "Description")
    .def(nb::init<>(), "Create MyPass");
```

**Register** (`python/pypto/ir/pass_manager.py`):

```python
OptimizationStrategy.Custom2: [
    ("MyPass", lambda: passes.MyPass()),
]
```

## Key Utilities

**Type checking**: `IsA<TensorType>(...)`, `As<TensorType>(...)`
**Broadcasting**: `BroadcastShapes(shape1, shape2)`
**Promotion**: `PromoteDataTypes(dtype1, dtype2)`
**Validation**: `CHECK(args.size() == 2) << "Expected 2 args, got " << args.size();`

## File Locations

| Component | Location |
| --------- | -------- |
| Operators | `src/ir/op/[tensor\|block\|sync]_ops/` |
| Passes | `src/ir/transform/` |
| Python wrappers | `python/pypto/ir/op/` |
| Bindings | `python/bindings/modules/` |
| Tests | `tests/ut/ir/` |

## Key Patterns

**Operators**: Args for Expr, kwargs for metadata. Use `BroadcastShapes()` and `PromoteDataTypes()`.
**Passes**: Extend `Pass`, implement `Run(FunctionPtr)`. Use `IRMutator`. Return new nodes (immutable).
**Testing**: Use `testing` skill after implementation.

## Quick Reference

| Task | Doc | Files |
| ---- | --- | ----- |
| Add operator | `05-operator_registration.md` | `src/ir/op/[category]/` |
| Add pass | `10-pass_manager.md` | `src/ir/transform/` |
| Build IR | `08-ir_builder.md` | `python/pypto/ir/builder.py` |
