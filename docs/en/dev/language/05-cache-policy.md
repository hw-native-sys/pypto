# GM Cache-Access Policy

A **declared** policy for how a kernel reads a tensor out of global memory
(GM). `CachePolicy.BYPASS` says "stream this tensor — do not spend cache on
it"; `CachePolicy.DEFAULT` is the ordinary cached access every read gets today.

The policy is a *contract the author states*, never a hint the compiler infers.
It is therefore written explicitly, at one of two granularities, and is carried
unchanged from the DSL to codegen.

> **Requires PTOAS >= v0.64** (`PTOAS_VERSION` in `toolchain/versions.env`), and
> **has an effect on a2a3 only**. A `BYPASS` declaration becomes a
> `cache_policy` attribute on `pto.tload` plus an `offset` operand carrying that
> device's no-cache alias distance, which the assembler adds to that one load's
> source address. See [What codegen emits](#what-codegen-emits) and
> [Architectures](#architectures).

## Two surfaces

| Surface | Granularity | Written as | Use when |
| ------- | ----------- | ---------- | -------- |
| `pl.set_cache_policy(t, policy)` | every read of `t` in the enclosing scope | a standalone statement at the top level of a `pl.at(...)` / `pl.spmd(...)` body | Tensor programming — the GM reads are implicit (`pl.matmul`, `pl.assemble`, slicing) and there is no load call to annotate |
| `pl.load(..., cache=policy)` | one read | a kwarg on the load | Tile programming — you already name the access |

`pl.slice` / `tensor.slice` deliberately take **no** `cache=` kwarg: a slice
computes an address descriptor, it moves no data. The policy belongs to the op
that actually issues the GM read.

### Scope declaration

```python
@pl.program
class Demo:
    @pl.function
    def main(self, a: pl.Tensor[[256, 128], pl.FP32], b: pl.Tensor[[128, 256], pl.FP32],
             out: pl.Out[pl.Tensor[[256, 256], pl.FP32]]) -> pl.Tensor[[256, 256], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            pl.set_cache_policy(b, pl.CachePolicy.BYPASS)     # every read of b streams
            c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out
```

### Per-access kwarg

```python
tile = pl.load(x, [0, 0], [32, 32], cache=pl.CachePolicy.BYPASS)
```

## The contract

`CachePolicy.BYPASS` asserts **two** things about the tensor:

| Claim | Who guarantees it | Does the compiler check it? |
| ----- | ----------------- | --------------------------- |
| The kernel streams these bytes — no reuse worth caching | Author | No (a performance claim) |
| Nothing writes these bytes while the kernel runs | Author | Partially — see below |

Mixing a cached write and a bypassing read of the same bytes is a **coherency
bug**: the two paths can see different data. The compiler cannot prove absence
of a concurrent writer across tasks, ranks, or the host, so coherency is the
author's contract. This is exactly why the policy is never a default and never
inferred by an optimisation pass.

The one part the compiler *can* check, it does: declaring `BYPASS` on a tensor
**the same scope writes** is rejected at outlining (see
[Errors](#errors)). The scope's own parameter direction already says whether it
writes, so a self-inflicted coherency bug is caught rather than trusted.

## Precedence

The effective policy of one GM read is the first of:

1. its own explicit `cache=` kwarg, else
2. the scope declaration for the parameter it reads, else
3. `CachePolicy.DEFAULT`.

Explicit wins **in both directions** — `cache=pl.CachePolicy.DEFAULT` opts a
single access back into the cache inside a bypassing scope:

```python
with pl.at(level=pl.Level.CORE_GROUP):
    pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
    hot = pl.load(b, [0, 0], [32, 32], cache=pl.CachePolicy.DEFAULT)  # cached anyway
    rest = pl.load(b, [32, 0], [32, 32])                              # BYPASS (declared)
```

A load already present in the body (hand-written, or produced by an earlier
pass) is stamped by the same rule as a load the compiler synthesises: the
declaration applies unless that load states its own `cache=`.

## Where the declaration may be written

`pl.set_cache_policy` attaches to the *enclosing scope*, and a `ScopeStmt`'s
attrs are fixed when the scope begins. The parser therefore pre-scans a scope
body's **top-level** statements and hoists the markers onto the scope before
parsing the body.

| Position | Accepted | Why |
| -------- | -------- | --- |
| Top level of `with pl.at(level=pl.Level.CORE_GROUP, ...):` | Yes | The scope becomes the device kernel whose params the declaration resolves against |
| Top level of `for i in pl.spmd(N):` / an inline `with pl.spmd(N):` body | Yes | Attaches to the InCore carrier the body is outlined into |
| Top level of `with pl.at(<non-CORE_GROUP level>):` (Hierarchy) | Yes, syntactically | Resolved onto that outlined function; nothing lowers it into loads today (see [Limits](#limits)) |
| Body of a *dispatch*-shaped `with pl.spmd(N):` (calls a pre-defined kernel) | No | The GM reads happen in the callee — declare it there |
| `pl.cluster(...)` / `pl.manual_scope()` / `pl.scope()` body | No | Those scopes co-schedule or choose dependency semantics; they issue no GM read |
| Nested in an `if` / `for` inside a scope | No | A conditionally-executed declaration is a promise the compiler cannot check |
| Function body, outside any scope | No | There is no scope to attach to |

Two more rules the parser enforces:

- **Tracked by Var identity, never by name.** The declaration names the binding
  live *at the scope*. Rebinding the name afterwards (`b = self.foo(b)`) yields
  a new value the declaration does not cover.
- **Bare name, tensor-typed, already bound.** Attribute / subscript / call
  expressions name no binding; a non-tensor binding has no GM read to govern; a
  tensor created *inside* the body is not captured by the scope.

Repeating a declaration for the same binding is redundant, not an error — the
first one is kept, so the attr stays a set of distinct tensors.

## Errors

| Message (abridged) | Raised by | Cause |
| ------------------ | --------- | ----- |
| `pl.set_cache_policy() must be a standalone statement directly inside a pl.at(...) / pl.spmd(...) scope body` | Parser (`ParserSyntaxError`) | Written outside a scope, or nested inside an `if` / `for` |
| `pl.set_cache_policy() has nothing to attach to on this <Kind> scope` | Parser (`ParserSyntaxError`) | Spmd dispatch body, `pl.cluster`, or a runtime scope |
| `pl.set_cache_policy() takes exactly two positional arguments (no keywords)` | Parser (`ParserSyntaxError`) | Wrong arity, or keyword form |
| `pl.set_cache_policy() first argument must be a bare variable name` | Parser (`ParserSyntaxError`) | `t.field`, `t[0]`, `f(t)` — no binding to track |
| `pl.set_cache_policy() argument '<n>' is not defined at this point` | Parser (`ParserSyntaxError`) | Name not bound where the scope starts |
| `pl.set_cache_policy() argument '<n>' is not a tensor` | Parser (`ParserTypeError`) | Only a GM tensor read has a cache policy |
| `pl.set_cache_policy(...) references tensor '<n>', which is not captured by the scope body` | `OutlineIncoreScopes` (`CHECK_SPAN` → `ValueError`) | The scope body neither reads nor writes the tensor, so it is not captured and no parameter carries the policy |
| `pl.set_cache_policy(<n>, CachePolicy.BYPASS) is not allowed on a tensor this scope writes (<dir>)` | `OutlineIncoreScopes` (`CHECK_SPAN` → `ValueError`) | A bypassing read of bytes the same kernel writes is a coherency bug |

The two outliner rejections are user errors, not compiler bugs — hence
`CHECK_SPAN`, which attaches the IR source location.

## Carrier chain

The declaration changes carrier three times on the way down. Each hop exists for
a reason; none of them is interchangeable with the others.

```text
pl.set_cache_policy(b, BYPASS)                 statement, consumed at parse
  -> ScopeStmt.attrs_["cache_policy_vars"]     parse .. pass 9   (Var identity)
  -> Function attr "cache_policy"              pass 9 .. pass 11 (param INDICES)
  -> tile.load kwarg "cache"                   pass 11 .. codegen
  -> codegen: `cache_policy` attribute on each emitted `pto.tload`
```

| Hop | Carrier | Payload type | Written by | Consumed by |
| --- | ------- | ------------ | ---------- | ----------- |
| 1 | `ScopeStmt.attrs_[kAttrCachePolicyVars]` | `vector<pair<VarPtr, int>>` | DSL parser | [`OutlineIncoreScopes`](../passes/09-outline_incore_scopes.md) (pass 9) |
| 2 | `Function.attrs_[kAttrCachePolicyParams]` | `vector<pair<int32_t, int>>`, sorted by index | pass 9 | [`ConvertTensorToTileOps`](../passes/11-convert_tensor_to_tile_ops.md) (pass 11) |
| 3 | `tile.load` kwarg `cache` | `int` (`ir::CachePolicy`) | pass 11 | PTO codegen |

Design notes that keep the chain honest:

- **Not a field on `TensorView`.** A plain kernel parameter has no
  `tensor_view_` at all, so stamping a policy there would force one into
  existence — dragging in the strict `TensorViewCanonical` verifier, and
  [`MaterializeTensorStrides`](../passes/33-materialize_tensor_strides.md)
  rebuilds the view through a positional constructor that would silently drop
  the field.
- **Param indices are valid only across passes 8..10.** Only
  `OutlineClusterScopes` sits between them, and it does not mutate an outlined
  InCore param list. Downstream passes *do*:
  [`InjectGMPipeBuffer`](../passes/25-inject_gm_pipe_buffer.md) and
  [`MaterializeDistTensorCtx`](../passes/47-materialize_dist_tensor_ctx.md)
  append, and
  [`MaterializeValidShapeSymbols`](../passes/52-materialize_valid_shape_symbols.md)
  *prepends*. That is why pass 11 erases the attr after converting it.
- **The kwarg is an `int`, not the enum.** It follows `tile.store`'s `atomic`
  kwarg, so the serializer, deserializer, `structural_hash` and
  `structural_equal` need no new enum arm. `pl.CachePolicy` is bound
  int-convertible (`nb::is_arithmetic`) for the same reason, so the DSL passes
  `int(cache)` straight through.
- **The `cache` kwarg survives the rest of the pipeline** for the same reason
  `target_memory` does — it rides the op's kwargs, which no later pass rewrites.

## Printing and round-trip

The scope attr prints as **marker statements**, not as a header kwarg (the way
`no_dep_args=` / `dumps=` do), because a statement is the surface the parser
accepts:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
    pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
    ...
```

| Property | Behaviour |
| -------- | --------- |
| Ordering | Position-normalising — markers always print first, however the author ordered them; the parser hoists them from anywhere in the body |
| Spmd inline forms | Printed from the nested InCore carrier, whose `pl.at(...)` header the Spmd printer inlines away |
| Otherwise-empty scope | A scope holding only a declaration prints the marker instead of `pass` |
| Function attr (`cache_policy`) | Prints as a list of `(index, policy)` tuples, so a pass dump taken between pass 9 and pass 11 — the only window where it exists — re-parses |

## What codegen emits

A `BYPASS` read becomes one attribute on the emitted load and one operand — the
distance from the tensor's address to the uncached alias of the same bytes:

```mlir
func.func @main(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
                %__pypto_l2_cache_offset: i64) {
  ...
  pto.tload ins(%b__ssa_v0_pview : !pto.partition_tensor_view<256x256xf32>)
            outs(%b__ssa_v0_mat  : !pto.tile_buf<loc=mat, ...>)
            {cache_policy = #pto.load_cache_policy<l2_bypass>}
            offset = %__pypto_l2_cache_offset : i64
```

The attribute alone declares the policy and moves no address: PTOAS applies the
offset only to a load that also declared `l2_bypass`, and a declaration without
one compiles to an ordinary cached load. Both together are what bypasses L2 —
which is why the offset, not the attribute, is what makes the feature real, and
why [only a2a3 has one](#architectures).

### Where the offset comes from

A2/A3 maps every GM page twice — once cached, once not — and a load issued
against the uncached alias does not allocate in L2. The distance between the two
mappings is a per-device value only the driver knows (`rtGetL2CacheOffset`), so
it cannot be a constant in the compiler: one box answers `0x80000000000` where a
pto-isa comment names `0x100000000000`.

simpler queries it once per Worker and carries it to every core's
`GlobalContext`, where an incore kernel reads it with
`get_l2_cache_offset(args)` (simpler PR #2323). The generated kernel wrapper
reads it **once at entry** — the value cannot change during a dispatch, so a
per-load read would go back through GM for a constant — and forwards it through
the synthetic `%__pypto_l2_cache_offset` parameter:

```cpp
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    // ... tensor unpacking ...
    uint64_t __pypto_l2_cache_offset = get_l2_cache_offset(args);
    main(a__ssa_v0, b__ssa_v0, out__ssa_v0, __pypto_l2_cache_offset);
}
```

PTOAS >= v0.64 turns the pair into address arithmetic on a *copy* of the source
descriptor, so other loads of the same tensor keep the cached address, and then
issues an ordinary `TLOAD`:

```diff
-  TLOAD(v45, v50);
+  __gm__ uint8_t* v51 = reinterpret_cast<__gm__ uint8_t*>(PTOAS__GLOBAL_TENSOR_DATA(v50));
+  __gm__ float* v52 = (__gm__ float*) (v51 + v5);
+  GlobalTensor<float, ...> v53(nullptr);
+  v53 = v50;
+  TASSIGN(v53, v52);
+  TLOAD(v45, v53);
```

Zero is a valid answer, not a failure: an a2a3 device that exposes no alias
reports zero, `addr + 0` is the ordinary address, and the declaration costs
bandwidth rather than correctness. The simulator is that case by construction.
That is a *device* answering zero — distinct from an architecture that has no
alias at all, which never gets an offset operand to begin with.

Five properties of the emit are worth stating, because each one is asserted in
`tests/ut/codegen/test_cache_policy_codegen.py` (the argument order, in
`tests/ut/codegen/test_prefetch_codegen.py`):

| Property | Why |
| -------- | --- |
| `CachePolicy.DEFAULT` emits **nothing** | A kernel that states no policy keeps the PTO form it had before this existed, so the attribute, the operand and the parameter are the only difference between two otherwise identical kernels |
| The attribute is emitted **per load**, not per tensor | It is a property of the instruction; a hint on only the first of two loads would leave the second one allocating in L2 (the superseded `[CacheBypassUnsupported]` diagnostic was deliberately once-per-tensor — the opposite granularity) |
| The offset parameter is appended **once per kernel** | One runtime value serves every bypassing load, read once at entry |
| It joins the MX `layout` in **one** attribute dict, after it, and the operand follows the dict | PTOAS takes all present attributes in a single dict; keeping `layout` first leaves an MX load that declares no policy byte-identical |
| a5 emits the attribute and **no** offset | There is no second mapping to address (see [Architectures](#architectures)), and no accessor to read a distance from |

### Architectures

The double mapping is an a2a3 property, and so is everything built on it:

| Target | What a `BYPASS` declaration does today |
| ------ | -------------------------------------- |
| a2a3 device | The load is issued against the uncached alias, `addr + get_l2_cache_offset(args)`. This is the case the feature exists for |
| a2a3 device with no alias | The driver reports `0`, `addr + 0` is the ordinary address, and the read is cached — correct, one optimisation short. The simulator is this case by construction |
| a5 | Codegen emits the attribute and no offset, and PTOAS v0.64 lowers a bare attribute to an ordinary `TLOAD`. The declaration is accepted and **does nothing** |

A5 is not a missing offset waiting to be supplied: it has no second mapping to
address. pto-isa carries an L2 hint on A5's `TLOAD` as an instruction operand
instead, which is where a future A5 path would go — it is not wired to
`cache_policy` in v0.64, so nothing in the emitted CCE distinguishes a declared
load from an undeclared one there.

### Older assemblers

The emit itself is unconditional — `offset` is a v0.64 addition, and an older
assembler would fail it at the `pto.tload` verifier. That never happens in
practice: before the first `.pto` is assembled, codegen runs `ptoas --version`
and rejects any assembler older than the `PTOAS_VERSION` this repo pins, with
an error that names both versions.

### Limits

- Only an **InCore** kernel's loads pick the declaration up:
  `ConvertTensorToTileOps` is what turns GM reads into `tile.load`, and it
  transforms InCore functions. Declare the policy on the scope that becomes the
  device kernel (a `CORE_GROUP` `pl.at`, or a `pl.spmd` inline body).
- The policy governs **reads**. There is no store-side counterpart; `BYPASS` on
  a written tensor is rejected rather than reinterpreted.

## Implementation map

| Layer | File |
| ----- | ---- |
| Enum, attr keys | `include/pypto/ir/expr.h` (`CachePolicy`, `kAttrCachePolicyVars`, `kAttrCachePolicyParams`) |
| Op registration | `src/ir/op/tile_ops/memory.cpp` (`tile.load` `.set_attr<int>("cache")`) |
| DSL | `python/pypto/language/op/tensor_ops.py` (`set_cache_policy`), `python/pypto/language/op/tile_ops.py` (`load(cache=...)`) |
| Parser | `python/pypto/language/parser/ast_parser.py` (marker hoisting + rejections) |
| Outlining | `src/ir/transforms/utils/scope_outline_utils.cpp` |
| Lowering | `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp` |
| Printer | `src/ir/transforms/python_printer.cpp` (`PrintScopeCachePolicyStmts`) |
| Codegen | `src/backend/common/pto_ops_memory.cpp` (`MakeTileLoadCodegenPTO`) |
| Synthetic param | `src/codegen/pto/pto_codegen.cpp` (`MemRefCollectorVisitor::UsesL2BypassLoad`, signature emission) |
| Kernel wrapper | `python/pypto/backend/pto_backend.py` (`_uses_l2_cache_offset`, `_generate_kernel_wrapper`) |
| Runtime accessor | `runtime/src/a2a3/runtime/*/common/intrinsic.h` (`get_l2_cache_offset`), `runtime/docs/l2-cache-bypass.md` |

## See Also

- [Statements and Control Flow](01-statements.md) — scope forms and the other
  parse-time markers (`pl.dump_tag`, `pl.static_assert`).
- [OutlineIncoreScopes](../passes/09-outline_incore_scopes.md) — hop 1 → hop 2.
- [ConvertTensorToTileOps](../passes/11-convert_tensor_to_tile_ops.md) — hop 2 → hop 3.
