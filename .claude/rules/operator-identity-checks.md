# Operator Identity Checks

## Core Principle

**To test which operator a `Call`/`Submit` node carries, route the operator-name
literal through the registry getter — never compare a bare string literal, and
never rely on pointer identity.**

The registry getter (`OpRegistry::GetInstance().GetOp(name)` in C++,
`_ir_core.get_op(name)` in Python) **throws if `name` is not a registered
operator**, so a typo or a renamed operator fails loudly at the comparison site.
A raw `name_ == "tile.reshaep"` comparison silently evaluates to `false` instead
— the bug ships undetected.

```text
Need to know if `call` is operator "tile.reshape"?
├─ ❌ call->op_->name_ == "tile.reshape"   // typo => silent false
└─ ✅ IsOp(call, "tile.reshape")           // typo => ValueError at the call
```

**Match by name, not by pointer.** `Op` instances are constructed in several
places — registry singletons, the `.pto` deserializer (`deserializer.cpp`), and
the MemRef alloc builders (`memref_utils.h`) each build their own `Op` — so two
`Op`s sharing a name are the *same operator* yet *distinct pointers*. Name
identity is the invariant the IR maintains; pointer identity (`op_a == op_b` /
Python `is`) is **not**, and silently returns `false` on, e.g., a deserialized or
pass-synthesized op. The helpers below match by name and only use the getter to
validate the literal.

## C++ — use the `IsOp` helper

`IsOp` lives in `include/pypto/ir/op_registry.h`. It validates `op_name` via
`GetOp` (throws on a typo) and matches by canonical name:

```cpp
[[nodiscard]] inline bool IsOp(const OpPtr& op, const std::string& op_name);     // op may be null
[[nodiscard]] inline bool IsOp(const CallPtr& call, const std::string& op_name); // null-safe
[[nodiscard]] inline bool IsOp(const SubmitPtr& submit, const std::string& op_name);
```

| Before | After |
| ------ | ----- |
| `call->op_->name_ == "tile.reshape"` | `IsOp(call, "tile.reshape")` |
| `call->op_->name_ != "tile.store"` | `!IsOp(call, "tile.store")` |
| `submit->op_->name_ == "tile.matmul"` | `IsOp(submit, "tile.matmul")` |
| `opnode->name_ == "tile.store"` (have only the `OpPtr`) | `IsOp(opnode, "tile.store")` |
| `name == "tensor.create" \|\| name == "tensor.full"` | `IsOp(call, "tensor.create") \|\| IsOp(call, "tensor.full")` |

Keep any null-guard that protects a *later* dereference (e.g.
`if (!call || !call->op_) return;`) even though `IsOp` is itself null-safe.

File-local predicate helpers that only match names (e.g.
`IsCrossCoreSplitOp(const std::string&)`) should take a `const OpPtr&` and use
`IsOp` internally; update their in-file callers to pass `call->op_`.

**The literal must be a registered operator.** `IsOp` throws when `op_name` is
not registered. If a code path legitimately checks an *unregistered* name,
register the operator (preferred) rather than skipping it. Only fall back to a
raw `name_ == "..."` comparison if registration is genuinely impossible, with a
comment explaining why.

## Python — match `.name` through `get_op`

The getter raises on a typo; comparing `.name` keeps the check robust against
non-registry `Op` instances:

| Before | After |
| ------ | ----- |
| `expr.op.name == "array.get_element"` | `expr.op.name == _ir_core.get_op("array.get_element").name` |
| `expr.op.name != "array.get_element"` | `expr.op.name != _ir_core.get_op("array.get_element").name` |
| `value_expr.op.name in {"tile.load", "tile.create"}` | `value_expr.op.name in {_ir_core.get_op("tile.load").name, _ir_core.get_op("tile.create").name}` |

For a module-level set of operator-name literals, build it from `get_op(...).name`
so each literal is validated at import (a typo raises on load); the runtime check
stays a name-membership test:

```python
# ❌ raw names — a typo in any literal is silent
_SPMD_BLOCK_OPS = frozenset({"tile.get_block_idx", "tile.get_block_num"})
if isinstance(ir_op, _ir_core.Op) and ir_op.name in _SPMD_BLOCK_OPS: ...

# ✅ validated at import via get_op; matched by name at runtime
_SPMD_BLOCK_OPS = frozenset({_ir_core.get_op("tile.get_block_idx").name,
                             _ir_core.get_op("tile.get_block_num").name})
if isinstance(ir_op, _ir_core.Op) and ir_op.name in _SPMD_BLOCK_OPS: ...
```

## What is NOT an operator check (leave as-is)

These use the name as data, not as an operator literal — converting them is
wrong:

- **Registry / function lookups** keyed by name: `reg.IsRegistered(name)`,
  `GetEntry(name)`, `program_->GetFunction(name)`, `GetOpInfo(name)`,
  `LookupCompositeRule(name)`, `LookupFunction(name)`. No literal, no typo risk.
- **Namespace-prefix checks**: `name.find("tile.") == 0`, `IsBuiltinOp`,
  `name.rfind("dist.", 0) == 0` — these match a *family*, not one operator;
  there is no single op to `GetOp`.
- **Non-operator names**: function/kernel names, attribute/kwarg keys, dtype or
  layout strings, AST `node.name`, `Var.name`.
- **Construction**: literals fed to `GetOp(...)`, `create_op_call(...)`,
  `make_shared<Call>(...)`, or op registration.

## Checklist

When adding or reviewing a pass / codegen / DSL path that branches on which
operator a node carries:

- [ ] No bare `op_->name_ == "ns.op"` / `.op.name == "ns.op"` literal
      comparisons — route through `IsOp(...)` / `_ir_core.get_op(...).name`.
- [ ] Compound conditions and ternaries converted per-operand.
- [ ] Name-literal *sets* built from `get_op(...).name`; membership tested on `.name`.
- [ ] Matched by name, not pointer identity / Python `is`.
- [ ] Lookups, prefix checks, non-operator names, and construction left alone.
- [ ] Every literal names a *registered* operator (register it if not).
- [ ] Null-guards protecting later dereferences preserved.

## See Also

- `ir-kind-traits.md` — sibling-kind dispatch (`Var`/`IterArg`, `Call`/`Submit`)
- `pass-submit-awareness.md` — handle `Submit` wherever you handle `Call`
- `error-checking.md` — `GetOp`'s throw is a `pypto::ValueError`
