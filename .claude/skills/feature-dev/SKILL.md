---
name: feature-dev
description: Develop new PyPTO IR features including operators, passes, and transformations. Reads documentation first, implements changes following project patterns, and integrates testing. Use when adding IR operators, creating passes, implementing transformations, or developing new IR features.
---

# PyPTO Feature Development

Develop new IR features: read docs first, implement following project patterns.

## Read Documentation First

Before coding, read relevant docs:

- Operators → `docs/dev/05-operator_registration.md`
- Passes → `docs/dev/passes/00-pass_manager.md`
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

**C++ Factory** (`src/ir/transforms/my_pass.cpp`):

```cpp
#include "pypto/ir/transforms/passes.h"

namespace pypto::ir::pass {
Pass MyPass() {
  return CreateFunctionPass(
      [](const FunctionPtr& func) -> FunctionPtr {
        // Transform using IRMutator methods
        return transformed_func;
      },
      "MyPass",
      {.required = {IRProperty::SSAForm},
       .produced = {IRProperty::SomeProperty},
       .invalidated = {}});
}
}  // namespace pypto::ir::pass
```

**Declare** in `include/pypto/ir/transforms/passes.h`: `Pass MyPass();`

**Python Binding** (`python/bindings/modules/passes.cpp`):

```cpp
passes.def("my_pass", &pass::MyPass, "Create MyPass");
```

**Type Stub** (`python/pypto/pypto_core/passes.pyi`): `def my_pass() -> Pass: ...`

**Register** in `python/pypto/ir/pass_manager.py` if part of a strategy.

## Key Utilities

**Type checking**: `IsA<TensorType>(...)`, `As<TensorType>(...)`
**Broadcasting**: `BroadcastShapes(shape1, shape2)`
**Promotion**: `PromoteDataTypes(dtype1, dtype2)`
**Validation**: `CHECK(args.size() == 2) << "Expected 2 args, got " << args.size();`

## File Locations

| Component | Location |
| --------- | -------- |
| Operators | `src/ir/op/[tensor\|block\|sync]_ops/` |
| Passes | `src/ir/transforms/` |
| Python wrappers | `python/pypto/ir/op/` |
| Bindings | `python/bindings/modules/` |
| Tests | `tests/ut/ir/` |

## Key Patterns

**Operators**: Args for Expr, kwargs for metadata. Use `BroadcastShapes()` and `PromoteDataTypes()`.
**Passes**: Use `CreateFunctionPass`/`CreateProgramPass` with `PassProperties`. Return new nodes (immutable).
**Testing**: Use `testing` skill after implementation.

## Quick Reference

| Task | Doc | Files |
| ---- | --- | ----- |
| Add operator | `05-operator_registration.md` | `src/ir/op/[category]/` |
| Add pass | `passes/00-pass_manager.md` | `src/ir/transforms/` |
| Build IR | `08-ir_builder.md` | `python/pypto/ir/builder.py` |
