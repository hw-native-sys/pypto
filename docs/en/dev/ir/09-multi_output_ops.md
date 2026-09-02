# Multi-Output Operators

## Overview

Some hardware intrinsics produce more than one value. PTOAS `TGATHER` in its
compare form writes both the gathered indices and a per-row match count;
`pto.tgather` names them as two `outs(...)` operands. An operator wrapping such
an intrinsic is a **multi-output operator**, and PyPTO expresses its results as
a `TupleType`, never as destination arguments.

**The rule**: register the operator with its *inputs only*. Every output is an
element of the `TupleType` that `f_deduce_type` returns.

See [Operator System](05-operators.md) for the general registration API and
[Types and Examples](02-types.md#tupletype) for `TupleType` itself.

## Why destinations must not be arguments

Wrapping a destination-passing-style (DPS) intrinsic, the path of least
resistance is to mirror the hardware signature and declare the destinations as
arguments. That is wrong twice over:

```text
tile.gather_compare(src, kvalue, tmp, dst, cdst)   ← the leak
  → the caller must allocate dst and cdst
     but tile allocation belongs to InitMemRef, which owns every tile buffer
  → the op looks like a 5-input operator
     so direction inference reads dst/cdst as consumers, and the writes vanish
```

The second consequence is the dangerous one: an argument nobody classified
defaults to `ArgEffect::Read`, so no RAW edge is emitted against whoever reads
the destination, and the failure surfaces on device as stale data rather than at
compile time. See [Argument effects](05-operators.md#argument-effects).

Expressed as a `TupleType` instead, the results are ordinary SSA values:
`InitMemRef` allocates each element like any other tile, `MemoryReuse` treats
them as independent reuse candidates, and nothing about the hardware's DPS shape
reaches the user.

## Registration

Declare the arity with `set_output_arity(N)` and return an N-element `TupleType`
from `f_deduce_type`:

```cpp
// ❌ Wrong — DPS destinations leaked into the argument list
REGISTER_OP("tile.gather_compare")
    .add_argument("src", "...")
    .add_argument("kvalue", "...")
    .add_argument("tmp", "...")
    .add_argument("dst", "...")     // leak
    .add_argument("cdst", "...");   // leak

// ✅ Correct — inputs only; outputs carried by the deduced TupleType
REGISTER_OP("tile.gather_compare")
    .add_argument("src", "Source tile (FP16/FP32/INT16/INT32, 2D)")
    .add_argument("kvalue", "Scalar threshold")
    .add_argument("tmp", "Workspace tile (UINT8)")
    .set_output_arity(2)
    .set_arg_effect(0, ArgEffect::Read)
    .set_arg_effect(1, ArgEffect::Read)
    .set_arg_effect(2, ArgEffect::Write)
    .set_workspace_arg(2)
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return std::make_shared<TupleType>(std::vector<TypePtr>{
          DeduceDstType(args, kwargs),
          DeduceCdstType(args, kwargs),
      });
    });
```

| Method | Purpose |
| ------ | ------- |
| `set_output_arity(N)` | Declares N produced values. `N > 1` means the deduced result is a `TupleType` of exactly N elements |
| `set_workspace_arg(i)` | Declares argument `i` to be compiler-supplied scratch — written by the hardware, carrying no result anyone reads |

`set_output_memory(space)` applies to **every** `TileType` element inside the
`TupleType`. An operator whose outputs live in different memory spaces must set
`memory_space_` inside `f_deduce_type` instead of relying on that fallback.

### Workspace versus destination

A written argument is one of two things, and the registration must say which:

| Kind | Example | Declaration |
| ---- | ------- | ----------- |
| **Workspace** — hardware scratch, no result the caller reads | `tile.gather_compare`'s `tmp`, synthesized by `ConvertTensorToTileOps` | `set_arg_effect(i, ArgEffect::Write)` + `set_workspace_arg(i)` |
| **Destination** — a result the caller reads | `dst`, `cdst` | Not an argument at all — a `TupleType` element |

The distinction is not cosmetic. A workspace is allocated by the pass that
synthesizes it and is never read, so it needs no SSA result; a destination is a
value the program goes on to use.

## What the registry enforces

Two checks, both of which fail loudly rather than letting a leak reach device.

**At import** — `OpRegistry::ValidateMultiOutputOps()` walks every operator with
arity > 1 and rejects three shapes:

| Rejected | Why |
| -------- | --- |
| An argument with no declared effect | The default `Read` is indistinguishable from a destination in hiding |
| A written argument not declared a workspace | It is either scratch that must say so, or a destination that belongs in the `TupleType` |
| `set_workspace_arg(i)` naming no argument | An index past the end is a typo that silently protects nothing |
| A workspace argument the operator never writes | Scratch is hardware-written by definition. Declaring it `Read` — or reaching that through `no_arg_writes()` — is the same dropped dependency edge, wearing a marker that says otherwise |
| `set_output_reuses_input(N)` | With several results, "the output reuses input N" cannot say which one |

The check works from the argument list, so it catches a destination by the trace
it leaves — a write nobody declared scratch, or a slot nobody classified at all.
A destination declared `Read` and never marked a workspace leaves no such trace
and passes; what stops that one is the convention above, not the registry.

**At call creation** — `OpRegistry::Create` cross-checks the declared arity
against the deduced type, in both directions: a declared arity of N must deduce
an N-element `TupleType`, and a deduced `TupleType` must have been declared.
The second direction matters because codegen reads the arity from the registry;
a tuple result nobody declared would have no arity to resolve its elements by.

## How a multi-output call flows through the pipeline

```text
DSL wrapper          dst, cdst = pl.tile.gather_compare(src, kvalue, tmp, ...)
                     returns (Tile(TupleGetItemExpr(call, 0)),
                              Tile(TupleGetItemExpr(call, 1)))
        ↓
Parser desugaring    _tuple_tmp = tile.gather_compare(src, kvalue, tmp)
                     dst  = _tuple_tmp[0]
                     cdst = _tuple_tmp[1]
        ↓
InitMemRef           dst and cdst are ordinary TileType vars and each gets its
                     own MemRef; _tuple_tmp is TupleType and gets none
        ↓
MemoryReuse          the tuple temporary carries the call's no-alias inputs onto
                     every element; each element is an independent candidate
        ↓
PTO codegen          PrepareTupleOutputs(op) recovers the element vars from the
                     `<var> = _tuple_tmp[i]` bindings and allocates them
```

The DSL wrapper returns one `TupleGetItemExpr` per output over the *same* call;
`_unwrap_result` in `python/pypto/language/parser/_dsl_invoker.py` recognizes
that shape generically and hands the parser back the bare `Call` to rebind.

## Codegen

A multi-output op's `Call` does not carry its destinations in `args_` — the
parser put them in separate `AssignStmt`s — so an emitter looks them up:

```cpp
static std::string MakeGatherCompareCodegenPTO(const CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const auto outs = codegen.PrepareTupleOutputs(op);   // resolve + allocate

  std::ostringstream oss;
  oss << "pto.tgather ins(" << /* ... inputs ... */ ")"
      << " outs(" << outs[0].name << ", " << outs[1].name
      << " : " << outs[0].type_str << ", " << outs[1].type_str << ")";
  codegen.Emit(oss.str());
  return "";
}
```

`PrepareTupleOutputs` reads the arity from the registry, resolves each element
var, checks it carries the MemRef `InitMemRef` assigned, and emits its
`alloc_tile` **eagerly** — the intrinsic writes those buffers before the
`<var> = tuple[i]` `AssignStmt`s that would otherwise allocate them. The
emission is idempotent, so those statements then skip re-emitting.

The element bindings are indexed once per function on entry
(`fs_.tuple_element_index`), so resolving them is a map lookup rather than a
body rescan per call.

## Adding a multi-output operator

1. Register with inputs only; add `set_output_arity(N)`.
2. Classify **every** argument with `set_arg_effect`, or `no_arg_writes()` when
   the operator writes through none of them. The import-time check requires a
   verdict on each.
3. Mark any written argument with `set_workspace_arg(i)` — and if it is not
   scratch, it is a destination and does not belong in the argument list.
4. Return an N-element `TupleType` from `f_deduce_type`.
5. Write the DSL wrapper to return a tuple of `TupleGetItemExpr(call, i)` over
   one call — the parser's unpacking path recognizes that shape.
6. Write the codegen emitter using `PrepareTupleOutputs(op)`.
7. Test the registration contract (`tests/ut/ir/operators/test_op_registry.py`
   picks up any new `set_output_arity(N > 1)` automatically) and the lowering
   end to end.

## See Also

- [Operator System](05-operators.md) — the general registration API and argument effects
- [Types and Examples](02-types.md) — `TupleType` and the rest of the type system
- [Parameter Directions](08-param-directions.md) — how an undeclared write loses its dependency edge
- [InitMemRef](../passes/33-init_memref.md) — the pass that owns tile allocation
- [MemoryReuse](../passes/35-memory_reuse.md) — lifetime reuse across tuple elements
