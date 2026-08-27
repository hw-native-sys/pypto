# Statements and Control Flow

Statement forms the parser accepts, and the SSA phi-node semantics behind
`pl.yield_`. See [Python IR Syntax Specification](00-python_syntax.md) for the
type system and expression grammar.

## Statements

### Assignment

```python
x: pl.INT64 = expr
y: pl.Tensor[[4], pl.FP32] = tensor_op(a)
```

### If Statement (SSA-style)

```python
# If with both branches
if condition:
    y1 = pl.yield_(value1)
else:
    y1 = pl.yield_(value2)

# Multiple return values (no inline type annotations)
if condition:
    y1, y2 = pl.yield_(value1, value2)
else:
    y1, y2 = pl.yield_(value3, value4)
```

**Key points:**

- `pl.yield_()` assigns to SSA phi nodes
- Variables defined in yield become accessible after if
- Both branches must yield the same variables
- Type annotations cannot be used inline with tuple unpacking

### For Loop (SSA-style with iter_args)

```python
# Simple loop (1-3 positional args, like Python's range())
for i in pl.range(stop):                    # start=0, step=1
for i in pl.range(start, stop):             # step=1
for i in pl.range(start, stop, step):       # explicit

# Loop with iter_args (loop-carried values)
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(n, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final = sum

# Parallel for loop (same 1-3 arg forms)
for i in pl.parallel(stop):
for i in pl.parallel(start, stop, step):
    body_statements
```

**Key points:** Loop-carried values use `pl.range()` or `pl.parallel()` with `init_values`, tuple unpacking `(sum,)` declares iter_args, `pl.yield_()` updates values for next iteration, after loop iter_args contain final values. `pl.parallel()` produces a `ForKind.Parallel` loop while `pl.range()` produces `ForKind.Sequential` (default).

### While Loop (SSA-style with iter_args)

```python
# Natural while: condition is the while-header expression
i: pl.Scalar[pl.INT64] = 0
while i < n:
    i = i + 1

# SSA form with init_values: header tuple = iter_args, first stmt is pl.cond().
# yield-LHS supplies the post-loop binding name (mirrors pl.range).
x_init: pl.Scalar[pl.INT64] = 0
for (x,) in pl.while_(init_values=(x_init,)):
    pl.cond(x < n)
    x_next = pl.yield_(x + 1)
# `x_next` is bound here (from the yield-LHS); `x` is loop-scoped only.

# Pre-SSA: no pl.yield_ at all; ConvertToSSA synthesizes it later.
for (x,) in pl.while_(init_values=(x_init,)):
    pl.cond(x < n)
    x = x + 1

# ❌ Bare pl.yield_(...) with non-empty init_values is rejected at parse time:
#    for (x,) in pl.while_(init_values=(x_init,)):
#        pl.cond(x < n)
#        pl.yield_(x + 1)             # ParserSyntaxError: requires assignment-form pl.yield_
```

**Key points:** `pl.while_(init_values=(...,))` reuses the `for ... in` header for SSA-style loops; the first body statement must be `pl.cond(<bool>)`. The post-loop binding name comes from the **yield-LHS** (`x_next` above), not the header tuple — header-tuple names are scoped to the loop body only. This convention is **uniform with `pl.range`**: assignment-form yield is required whenever `init_values` is non-empty AND the body contains a `pl.yield_(...)` call. Pre-SSA loops with no yield at all are still valid (last form above).

### Scope Context Managers

| Form | Scope Kind | Notes |
| ---- | ---------- | ----- |
| `pl.at(level=pl.Level.CORE_GROUP)` | `InCore` | Fixed-boundary outline at CORE_GROUP |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(MODE)])` | `InCore` | InCore + cross-core split hint |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=N)])` | `InCore` | InCore + cross-core pipe slot count |
| `pl.at(level=pl.Level.HOST)` *(or any non-`CORE_GROUP` level)* | `Hierarchy` | Distributed hierarchy scope |
| `pl.cluster()` | `Cluster` | Co-scheduled AIC+AIV group |
| `with pl.spmd(N)` / `for i in pl.spmd(N)` | `Spmd` (for-form wraps inner `InCore`) | SPMD multi-block dispatch — see [pl.spmd](#plspmd-multi-block-dispatch) |
| `pl.spmd(N, optimizations=[pl.split(MODE)])` | `Spmd(InCore(split=MODE))` | Split hint applies to the inner InCore (both forms) |
| `pl.spmd(N, optimizations=[pl.cross_core_slot(slot_num=N)])` | `Spmd(InCore(slot_num=N))` | Slot count applies to the inner InCore (both forms); combinable with `pl.split(MODE)` |
| `pl.scope(mode=pl.ScopeMode.MANUAL)` / `pl.manual_scope()` | `Runtime(manual=true)` | Orchestrator MANUAL scope — user manages task ordering. Allowed in either `auto_scope` mode (it is a dependency-semantics choice). See [Manual dependency primitives](02-manual_dependencies.md#manual-dependency-primitives) |
| `pl.scope()` | `Runtime(manual=false)` | Orchestrator AUTO scope (`SIMPLER_SCOPE()`). Hand-placing one requires `@pl.function(auto_scope=False)` (in the default `auto_scope=True` the compiler owns AUTO placement). See [MaterializeRuntimeScopes](../passes/46-materialize_runtime_scopes.md) |

See [Scopes and Placement](../../user/language/04-scopes.md) for examples.

#### `pl.spmd` multi-block dispatch

`pl.spmd(N)` dispatches a kernel across `N` blocks. Forms:

- `with pl.spmd(N): ...` — body is **either** a *dispatch* body calling a pre-defined InCore kernel (`SpmdScopeStmt(body=<stmts>)`, no inner InCore wrapper) **or** an *inline* block auto-outlined into a synthetic InCore region (like the for-form, minus the auto-bound index). Decided semantically, not by statement count: a body reading `pl.tile.get_block_idx()` is inline and gets wrapped; otherwise it is a dispatch body and stays unwrapped, however many statements it holds. A body that neither reads the index nor dispatches a `self.<kernel>(...)` call is rejected. Captures no producer TaskId.
  - An explicit `with pl.at(<CORE_GROUP level>, ...):` as the sole body statement *is* the InCore carrier: parsed as an ordinary nested scope, not wrapped a second time (positional or keyword `level`, with or without `as tid` / `name_hint=`). This is the form the printer emits for `Spmd(InCore(...))`, so it is what makes that IR round-trip. When the body provides a carrier, `optimizations=` must go on that `pl.at(...)` — putting it on the `pl.spmd(...)` line is rejected, whether or not the carrier also carries one.
  - A dispatch body may launch **one** kernel. It is lowered via `FindFirstInnerCall`, which stops at the first call, so a second dispatch would be silently dropped rather than launched; the parser rejects it instead. Hoisted temporaries and tuple projections are not dispatches and do not count.
- `for i in pl.spmd(N): ...` — loop variable binds the per-block index (`pl.tile.get_block_idx()`); the body is auto-outlined into a synthetic InCore region.
- `with pl.spmd(N, deps=[...]) as tid: ...` — **capture form**: mirrors `with pl.at(...) as tid:`. Same body shapes as the plain form above, and additionally captures the dispatch's grid-wide producer `pl.Scalar[pl.TASK_ID]` in `tid` (usable as a `deps=` edge, stored into a `pl.array.create(N, pl.TASK_ID)`, or crossed into `pl.manual_scope`). TaskId capture is orthogonal to the inline body — it is the only thing this form adds over the plain form. Lowers to an `ir.Submit` whose trailing tuple element is the grid TaskId; `core_num` / `sync_start` ride on that `Submit`'s own fields (the launch spec belongs to the launch site, not the outlined callee). See [Manual dependency primitives](02-manual_dependencies.md#manual-dependency-primitives).
- `out, tid = pl.spmd_submit(kernel, *args, core_num=N)` — **submit form**: dispatches the kernel across `N` blocks *and* captures the dispatch's producer `pl.Scalar[pl.TASK_ID]` (the `pl.submit` sibling for a pre-defined kernel). See [Manual dependency primitives](02-manual_dependencies.md#manual-dependency-primitives).

All three `pl.spmd(...)` scope forms also accept `allow_early_resolve=True` (a boolean literal; same early-dispatch opt-in as `pl.submit` / `pl.at`). It forces the dispatch to lower to an `ir.Submit` even without `as tid` and lowers to `Arg::set_allow_early_resolve(true)`. Rejected on a `pl.cluster()`-nested `pl.spmd` (such a scope is unwrapped into the Group function and never produces a Submit, so the hint would be lost).

Optional `optimizations=[...]`. The entries are orthogonal and may be combined
in one list (e.g. `[pl.split(MODE), pl.cross_core_slot(slot_num=4)]`):

| Entry | Form | Effect |
| ----- | ---- | ------ |
| `pl.split(MODE)` | both | Sets the inner InCore's `split_` field (cross-core transfer hint, consumed by `ExpandMixedKernel` / `MemoryReuse`). The with-form gains an inner `InCoreScopeStmt` wrapper around the call. |
| `pl.cross_core_slot(slot_num=N)` | both | Sets the inner InCore's `slot_num` attr — the slot count (ring depth) of the automatic cross-core pipe, consumed by `ExpandMixedKernel`. Sizes a data channel only; it does **not** partition work, so it coexists with `pl.split_aiv` regions where `pl.split(...)` does not. Omit to keep the default depth of 2 per active direction. |

> `pl.split(MODE, slot_num=N)` is a deprecated alias for the slot count and warns
> — see [ExpandMixedKernel](../passes/22-expand_mixed_kernel.md#overriding-the-slot-count-slot_num).

### Yield Statement

```python
yield            # No values
yield x          # Single value
yield x, y       # Multiple values
```

### Break and Continue

```python
break              # Exit innermost loop
continue           # Skip to next iteration
```

**Restrictions:** Only valid when the **innermost** enclosing loop is sequential (`pl.range`) or `while`. Not supported when the innermost loop is `pl.parallel()` or `pl.unroll()`. A `break` in an inner `pl.range` loop nested inside an outer `pl.parallel` loop is valid. **Note:** Codegen backend support for `break`/`continue` is tracked in [#448](https://github.com/hw-native-sys/pypto/issues/448).

### Compile-Time Debugging

`pl.static_print()` and `pl.static_assert()` are parse-time-only constructs for inspecting IR state and asserting conditions during parsing. They produce **no IR**.

```python
@pl.function
def func(x: pl.Tensor[[128, 64], pl.FP16]) -> pl.Tensor[[128, 64], pl.FP16]:
    pl.static_print("input:", x)          # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_print(f"input: {x}")        # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_assert(True)                # passes silently
    pl.static_assert(N > 32, "N too small")  # checks closure variable N at parse time
    return x
```

| Function | Purpose | On failure |
| -------- | ------- | ---------- |
| `pl.static_print(*args)` | Print variable types/values to stdout | Requires ≥1 argument |
| `pl.static_assert(cond, msg="")` | Assert compile-time condition | Raises `ParserError` |
| `pl.dump_tag(tensor)` | Mark a tensor for selective runtime tensor dump — declarative per-tensor marker (valid in Orchestration scope, or in an Inline helper that the orch inlines — see [Runtime DFX](../03-runtime-dfx.md#selective-tensor-dump)) | Raises `ParserSyntaxError` outside an Orchestration or Inline function, or for non-`Name` arguments |

**Key points:**

- All three are statement-only (cannot be used in expressions)
- `static_print` accepts variables, constants, string labels (printed as-is), and f-strings with plain `{expr}` placeholders (formatted as IR). Conversions (`!r`, `!s`, `!a`) and format specs (`:...`) are not supported.
- `static_assert` supports closure variable expressions (e.g. `N > 32`) and IR constants; message must be a string literal
- `dump_tag` takes one bare tensor variable name bound in the enclosing Orchestration (or Inline) scope; it is consumed at parse time and tracked by Var identity (not name) all the way to codegen. At an explicit `self.kernel(...)` site it records the tensor in the consuming Call's `dump_vars` on every subsequent consuming call; in the `@pl.jit` / `with pl.at(level=...)` style (where the dispatch is synthesised by the outline passes) it instead seeds the enclosing scope's `dump_vars` and the outliner maps it onto the synthesised dispatch arg (see [Runtime DFX](../03-runtime-dfx.md#selective-tensor-dump)). To list dump targets explicitly at a single task launch, use the `dumps=[...]` kwarg on `pl.submit(...)` / `pl.at(...)` (symmetric with `deps=`)
- Output appears even if parsing fails later — useful for debugging parse errors

### Statement Sequences

```python
stmt1            # Natural Python sequencing
stmt2
stmt3
```

## SSA-Style Control Flow

`pl.yield_()` creates SSA phi nodes for if/for statements:

```python
# If: phi node at merge point
if condition:
    y1 = pl.yield_(x + 1)
else:
    y1 = pl.yield_(x + 2)
# y1 = phi(x + 1, x + 2)

# For: loop-carried values via iter_args
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(10, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final: pl.INT64 = sum  # captures final value
```

## See Also

- [Python IR Syntax Specification](00-python_syntax.md) — types and expressions
- [Manual Dependency Primitives](02-manual_dependencies.md) — opting out of auto-dep tracking
- [Functions and Program Structure](03-functions.md) — function types and parameter directions
