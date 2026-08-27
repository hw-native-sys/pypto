# MaterializeValidShapeSymbols Pass

Turns each device-kernel `valid_shape` symbol that the kernel cannot bind into a
leading `Scalar[INDEX]` parameter, fed the caller's actual extent at every call
site.

## Overview

A `pl.dynamic()` symbol used in a parameter's `pl.TensorView(valid_shape=...)`
has no value inside a precompiled kernel:

```python
VALID = pl.dynamic("VALID")

@pl.function(type=pl.FunctionType.InCore)
def softmax_prepare(
    sij: pl.Tensor[[Q, BLK], pl.FP32,
                   pl.TensorView(valid_shape=[Q, VALID], layout=pl.TensorLayout.ND)],
    out: pl.Out[pl.Tensor[[Q, BLK], pl.FP32]],
): ...
```

`VALID` is neither a physical tensor dimension — those the kernel wrapper
recovers from the runtime tensor's `shapes[]` — nor a scalar parameter. The
runtime `ChipTensor` carries no valid extent (see `runtime/src/common/task_interface/tensor.h`),
so the value has to arrive as an argument. Before this pass existed, PTO codegen
logged `Variable VALID not found in MLIR mapping` and kept going, emitting an
operand-less `%0 = arith.minsi , %c128_index : index` that surfaced much later as
an opaque ptoas `error: expected SSA operand`.

The pass rewrites the program so the existing scalar-parameter path carries the
value end to end:

1. For every device kernel (`InCore` / `AIC` / `AIV` / `Spmd`), find the symbols
   read by a parameter's declared `valid_shape` that are not bound by any tensor
   parameter's physical shape and are not already scalar parameters.
2. Insert those symbols as `ParamDirection::In` parameters **at the front** of the
   signature. The symbol Var itself becomes the parameter — `DynVar.unwrap()`
   already builds it as a `Scalar[INDEX]` Var shared by every annotation naming
   it, so every occurrence binds at once with no type rewrite.
3. For every `Call` / `Submit` of such a kernel, read each symbol's value out of
   the actual argument's `valid_shape` at the declared position and prepend it.
4. When call-site `arg_directions` are already resolved, prepend matching
   `ArgDirection::Scalar` entries.

It runs last in the `Default` strategy: it only extends signatures and call
argument lists, and by that point both are final, so no later pass has to account
for the added parameter.

## Why the Parameters Go First

A symbol is read by the very parameter annotation that names it. The text form
declares parameters left to right and Python evaluates annotations in the
enclosing scope, so an appended parameter prints a signature that uses `VALID`
before declaring it — which does not re-parse:

```python
# does not re-parse: NameError at def time
def kernel(a: pl.Tensor[..., pl.TensorView(valid_shape=[M, VALID])],
           VALID: pl.Scalar[pl.INDEX]): ...
```

Leading placement fixes the ordering. Signature order is otherwise free: PTOParam
dispatches args as `[tensors..., scalars...]` regardless of it (see
`PTOCodegen::GenerateFunction`).

Two supporting rules keep the printed form round-tripping:

- The Python printer keeps the `pl.dynamic()` declaration for a **parameter** that
  is read inside a parameter's **valid_shape**, so the annotation resolves at def
  time. A body-local var in a valid_shape, and a parameter read as a *physical*
  dim, both stay undeclared (issue #854).
- The parser re-points that `DynVar` at the parameter as soon as the parameter is
  declared, so later annotations read the parameter's Var rather than a second,
  unbound Var of the same name.

## Binding Rule and Its Limits

A symbol is bound **positionally**: the declared slot must name the symbol on its
own, and the call site reads the actual argument's `valid_shape` at that same
position.

| Declared | Actual | Result |
| -------- | ------ | ------ |
| `valid_shape=[Q, VALID]` | `valid_shape=[16, valid_len]` | `VALID := valid_len` |
| `valid_shape=[Q, VALID * 2]` | `valid_shape=[16, n]` | rejected — cannot invert |
| `VALID` in two params, actuals disagree | — | rejected — one symbol, two extents |

Compound slots are rejected rather than inverted: a wrong valid extent silently
reads or writes the wrong region. The remedy is to name the symbol bare in some
parameter's `valid_shape`, or to pass the extent as a `pl.Scalar[pl.INDEX]`
parameter and use it in `pl.load(..., valid_shape=[...])`.

## Result

```mlir
func.func @softmax_prepare(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
                           %arg2: index, %arg3: index, %arg4: index) {
  %0 = arith.minsi %arg2, %c128_index : index      // %arg2 == VALID
  %t = pto.alloc_tile addr = %c0_i64 valid_row = %c16_index valid_col = %0 : ...
```

and orchestration passes the caller's extent at launch:

```cpp
params_t0.add_scalar(valid_len);
```

## Backstop

`PTOCodegen::GetVarName` fails with an actionable `ValueError` naming the symbol
and its originating parameter if any symbol still reaches codegen unbound — for
instance under a custom pass list that omits this pass. It never emits an empty
operand.

## See Also

- [44-materialize_dist_tensor_ctx.md](44-materialize_dist_tensor_ctx.md) — same
  signature-and-call-site shape, for `CommCtxType`
- [00-pass_manager.md](00-pass_manager.md) — pass ordering
