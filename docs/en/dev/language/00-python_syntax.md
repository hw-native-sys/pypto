# Python IR Syntax Specification

## Overview

Python-style syntax for PyPTO IR:

- **Complete**: All information needed to reconstruct IR
- **Parseable**: Can be parsed back into IR (see [IR Parser](../ir/07-parser.md))
- **Pythonic**: Follows Python style, passes most linters
- **SSA-style**: Uses SSA with `pl.yield_()` and `pl.range()`

## Module Structure

```python
# pypto.program: program_name
import pypto.language as pl
```

For unnamed programs: `# pypto.program`

**Note:** Module prefix is configurable (default `pl`, legacy `ir`, custom allowed),
except `pld`, which is reserved for `pypto.language.distributed`.

The specification is split across four pages:

| Page | Covers |
| ---- | ------ |
| This page | Module structure, type system, expressions |
| [Statements and Control Flow](01-statements.md) | Assignment, if/for/while, scopes, yield, directives, SSA phi nodes |
| [Manual Dependency Primitives](02-manual_dependencies.md) | `pl.manual_scope`, `deps=`, dispatch predicates, array-carry fences |
| [Functions and Program Structure](03-functions.md) | Function types, parameter directions, cross-module reuse, printing |

## Type System

### Scalar Types

```python
x: pl.INT64
y: pl.FP32
z: pl.BOOL
```

Available types:

| Category | Types |
| -------- | ----- |
| **Integers** | `INT4`, `INT8`, `INT16`, `INT32`, `INT64` |
| **Unsigned** | `UINT4`, `UINT8`, `UINT16`, `UINT32`, `UINT64` |
| **Float** | `FP4`, `FP8`, `FP16`, `FP32` |
| **Brain Float** | `BF16` |
| **Hisilicon** | `HF4`, `HF8` |
| **Boolean** | `BOOL` |

### Tensor and Tile Types

```python
# Tensor (subscript notation)
a: pl.Tensor[[4, 8], pl.FP32]      # Fixed shape
b: pl.Tensor[[n, m], pl.INT64]     # Symbolic shape

# Tile (block in unified buffer)
t: pl.Tile[[16, 16], pl.FP16]
```

### Tensor Layout and View

The third subscript element is a layout or a `pl.TensorView`, each written inline or
held in a variable — bind it once to share one view across several parameters. A
layout is shorthand for a stride-less view, so every form yields a `TensorView`.
Same slot and spellings for `pl.DistributedTensor`.

```python
STRIDED = pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)

x: pl.Tensor[[32, 64], pl.FP32, pl.NZ]      # layout, inline
y: pl.Tensor[[32, 64], pl.FP32, STRIDED]    # view, by variable
```

Under `@pl.jit` only the **layout** spelling is supported. Specialization
regenerates each annotation from the shape/dtype/layout it recorded, and a
`pl.TensorView` has no slot in that record — passing one raises a `TypeError`
naming the parameter rather than dropping the stride. Declare such a kernel
with `@pl.function`, which resolves the annotation directly.

### Memory References (MemRef)

```python
# Create MemRef
addr_expr = pl.ConstInt(0x1000, pl.INT64, span)
memref = pl.MemRef(addr_expr, 1024, 0)

# Memory spaces: DDR, Vec, Mat, Left, Right, Acc
# Note: pl.Mem is a short alias for pl.MemorySpace

# Tensor with memref
tensor: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(addr_expr, 8192, 0)]

# Tiles keep memory space on the tile annotation, not inside MemRef
tile: pl.Tile[[16, 16], pl.FP16, pl.MemRef(addr_expr, 512, 0), pl.Mem.Left]
```

### Declared Allocations (one-argument MemRef)

A one-argument `pl.MemRef("name")` declares an allocation of your own, taking it out of
the compiler's opportunistic reuse. Tiles referencing it share it; nothing else is ever
packed in. Use it when the packer coalesces tiles you want to stay independent —
sharing storage adds a WAR dependency that serializes them.

It is the same IR node as the three-argument form; the arity says whether you are
describing an existing allocation or declaring one. Declaring gives only a name: the
size comes from the largest tile bound to it and the address from the allocator.

Declare it once, then reference it by variable. An unnamed declaration takes the name
of the variable it is bound to, so the name is written once:

```python
ping = pl.MemRef()
pong = pl.MemRef()

# Two tiles explicitly share one allocation; a third is kept private.
t0: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
t1: pl.Tile[[64, 64], pl.FP32, pong, pl.Mem.Vec] = pl.exp(t0)
t2: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.exp(t1)
```

Prefer that form: a misspelled reference is a Python `NameError`, whereas a misspelled
string in the inline `pl.MemRef("pign")` form silently declares a second allocation. The
inline form stays valid — it is what the IR printer emits, so a dumped program reparses
without a surrounding Python scope, and `pl.MemRef("other")` also names a declaration
explicitly when the variable name is not the one you want in the IR.

Since the variable supplies the name, variable and allocation must correspond one to one.
Reaching one declaration through two names (`alias = ping`) and two declarations claiming
one name are both **rejected** — either would silently merge or split an allocation.

Whether a MemRef is a declaration is recorded explicitly on the IR node
(`MemRef.is_pinned_`), not inferred from its size or from which pass is running.
`InitMemRef` consumes the declaration: from there on the allocation carries
`pinned=True` and the MemRef is an ordinary one, so re-parsing a post-allocation dump
cannot turn compiler allocations into declared ones.

#### Slots

Pass `slots=N` for N equally-sized slots of one allocation, then pick one by subscript.
The slots are contiguous and uniformly sized, so rotating through them is a ping-pong the
packer cannot collapse:

```python
l0c = pl.MemRef(slots=2)

ping: pl.Tile[[M, N], pl.FP32, l0c[0], pl.Mem.Acc] = pl.tile.matmul(q, b0)
pong: pl.Tile[[M, N], pl.FP32, l0c[1], pl.Mem.Acc] = pl.tile.matmul(q, b1)
```

**The index may be a runtime value**, so a rotation needs no unrolling:

```python
for i, (acc,) in pl.range(N, init_values=(out,)):
    a: pl.Tile[[M, N], pl.FP32, l0c[i % 2], pl.Mem.Acc] = pl.tile.matmul(q_l0, b_l0)
```

Under `@pl.jit`, name the declaration inline — `pl.MemRef("l0c", slots=2)[i % 2]` — rather
than binding it to a Python variable. `@pl.jit` re-parses a generated source in a fresh
module namespace, so a declaration held in a variable is not in scope there. The named form
is self-contained (and is what the IR printer emits), so it works in both.

`InitMemRef` sizes one slot to the largest tile bound to *any* slot — the slots are
uniform, so a per-slot size would make the stride inconsistent — and turns the index into
the byte offset `index * slot_size`. A constant index folds there and takes the ordinary
constant-address path; a runtime one survives as an expression that becomes the tile's
address at run time.

Co-liveness is checked **per slot**, not per allocation: two tiles on different slots are
meant to be live together, and only two tiles landing on the *same* slot can corrupt each
other. When the index is a runtime expression there is no static slot to attribute a tile
to, so the check is skipped — the rotation is yours to get right — while isolation from
every other allocation still holds.

##### Under the ptoas memory planner

`slots=N` is the one declaration form `memory_planner=PTOAS` accepts. ptoas has a matching
concept — a `pto.alloc_multi_tile` region of N slots it must keep in disjoint physical
segments — so codegen hands it the declaration whole: one region, and each use selects its
slot with `pto.multi_tile_get`. The slot **index** is what ptoas receives (not the byte
offset it resolves to), which is what lets it prove which accesses can share a slot and
give the rotation per-slot event ids, overlapping iteration *i*'s load with iteration
*i-1*'s compute.

A single-slot `pl.MemRef()` has no such counterpart and stays rejected under that planner:
ptoas would be free to pack the buffer you separated. So does a multi-slot declaration
whose slots ptoas cannot describe — differently shaped tiles across slots, a space other
than Vec / Mat / Acc, a runtime valid shape, or a slot carried out of an `if` or loop as a
phi. Those are errors naming the shape, not silent fallbacks, because a fallback would
undo the separation you declared.

The default PyPTO planner is unaffected: it bakes addresses, and at
`--pto-level=level3` ptoas does not fold its per-slot address fan-out, so the region form
would there lose the very slot analysis it exists for
([PTOAS#1106](https://github.com/hw-native-sys/PTOAS/issues/1106)).

A declared name lives in its own namespace — it never resolves to a Python variable that
happens to share it. The memory space **is** required (a `TileType` always pairs a
MemRef with a space), and all tiles bound to one allocation must agree on it. Tiles left
unannotated keep the default automatic reuse.

Declarations do not clone per pipeline stage, so one inside a `pl.pipeline(stage=2)`
body is **rejected** whenever that loop is lowered by replication: the cloned stages would
make the tile co-live with itself on one allocation. Declaring slots and asking the
compiler to multi-buffer are alternatives, not layers. To manage a level yourself, drive it
with `pl.range` and declare one allocation per slot; leave the levels you want the compiler
to manage unannotated.

Under `memory_planner=PTOAS` the compiler reaches for the *same* mechanism rather than a
different one: [`LowerPipelineToSlots`](../passes/28-lower_pipeline_to_slots.md) synthesizes
exactly the declaration above — `slots=F`, indexed `iv % F` — for every eligible top-level
`tile.load` of a `pl.pipeline` body, so one body rotates through the slots instead of being
replicated. A tile you bound yourself is left alone, and any loop that pass declines still
goes down the replication path (where the rejection above applies).

```python
l0b_ping, l0b_pong = pl.MemRef(), pl.MemRef()

# Outer level compiler-managed, inner level author-managed ping-pong.
for stack, (out_outer,) in pl.pipeline(STACKS, stage=2, init_values=(out,)):
    b_l1: pl.Tile[[K, N], pl.BF16, pl.Mem.Mat] = pl.load(b, [stack * K, 0], [K, N])
    for col, (out_inner,) in pl.range(0, N, 2 * STEP, init_values=[out_outer]):
        ping: pl.Tile[[K, STEP], pl.BF16, l0b_ping, pl.Mem.Right] = ...
        pong: pl.Tile[[K, STEP], pl.BF16, l0b_pong, pl.Mem.Right] = ...
```

See [InitMemRef](../passes/32-init_memref.md#declared-allocations) and
[MemoryReuse](../passes/34-memory_reuse.md#declared-allocations).

### Tile Views (TileView)

```python
# Create TileView
valid_shape = [pl.ConstInt(16, pl.INT64, span)] * 2
stride = [pl.ConstInt(1, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)]
start_offset = pl.ConstInt(0, pl.INT64, span)
tile_view = pl.TileView(valid_shape=valid_shape, stride=stride, start_offset=start_offset)

# Tile with memref and tile_view
tile: pl.Tile[
    [16, 16], pl.FP16,
    pl.MemRef(addr_expr, 512, 0), pl.Mem.Left,
    pl.TileView(valid_shape=..., stride=..., start_offset=...)
]
```

**Notes:**

- Omitting `pl.TileView(...)` does **not** mean "no TileView semantics". The DSL infers an implicit
  TileView from the tile shape and, when present, the tile memory space.
- In that implicit form, `valid_shape` defaults to the tile shape. Layout/fractal defaults are also
  inferred from the shape / memory-space combination.
- An explicit `pl.TileView()` (or one that only repeats those implicit defaults) is treated as
  semantically equivalent to the omitted form. Parser / printer roundtrips may canonicalize both
  forms to the same printed syntax.
- `compact=pl.CompactMode.normal` represents PTO's packed transfer format for a partial boxed tile.
  PyPTO infers it for partial `tile.extract` results in L0A/L0B, so kernels normally should not set it
  directly.

## Expressions

### Variables and Constants

```python
x                       # Variable reference
tensor_a                # Tensor variable
42                      # Integer literal — INDEX-typed
3.14                    # Float literal
pl.const(42, pl.INT64)  # Typed integer literal (any non-INDEX dtype)
```

A bare integer literal is always `INDEX`-typed. To carry any other integer
dtype (e.g. `INT64`), use `pl.const(value, dtype)` — this is also how the
printer renders such constants so printed IR round-trips through the parser.
Inside composite shape dimensions and pure-constant arithmetic (e.g.
`pl.const(32, pl.INDEX) + pl.const(32, pl.INDEX)`), the printer emits typed
leaves even for `INDEX` so the parser rebuilds the tree verbatim instead of
constant-folding it; simplification stays the Simplify pass's job.

**Closure variables:** Names not found in the DSL scope are resolved from the enclosing Python scope. Supported types: `int`, `float`, `bool`, `list`, `tuple`, and IR expressions.

```python
OFFSET = [0, 0]
TILE_SHAPE = [64, 64]

@pl.function
def func(t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
    a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, OFFSET, TILE_SHAPE)  # closure vars as positional args
    ...
```

**Enum op arguments:** Op wrappers have parameter slots that take a Python enum
rather than an IR expression — `DataType`, `MemorySpace`, `TensorLayout`,
`TileLayout`, `PadValue`, `ArgDirection`. These resolve identically whether
written positionally or by keyword, both as literal attributes and as closure
names, so the two lines below build the same call:

```python
p = pl.fillpad(t, pl.PadValue.min)              # positional
p = pl.fillpad(t, pad_value=pl.PadValue.min)    # keyword
```

The same holds for the numeric sugars an op accepts in such a slot: `pl.fillpad`
takes `0`, `0.0`, `math.inf`, and `-math.inf` in either position.

### Subscript Indexing

`Tensor` and `Tile` subscripts use numpy/torch-style semantics:

- A **scalar** index removes its dimension; a **slice** keeps it.
- Fewer indices than `rank` implies trailing `:` — `C[i]` on a 4D tensor is `C[i, :, :, :]`.
- Chained indexing composes — `C[i][j]` is two rank-reducing views.
- An **all-scalar, full-rank** index reads a scalar (`A[i, j]` on a 2D tensor → `tensor.read` / `tile.read`).

```python
C[i, j, k, l]   # all scalar, full rank   -> scalar
C[i, j]         # partial, all scalar      -> 64×64 view (dims 0,1 dropped)
C[i]            # partial                  -> 64×64×64 view (dim 0 dropped)
C[i][j]         # chained                  -> works (C[i] is 3D, then [j])
C[i:i+8, j]     # mixed slice + scalar     -> 8×64×64 view (dim 1 dropped)
C[i:i+8, :, :, :]  # all slices            -> 8×64×64×64 view
```

Restrictions (v1): no slice `step`, tile slice lower bounds must be static-foldable, no ellipsis / `None` / negative / advanced indexing. **Tiles are physically 2D**, so a tile result that would naturally be `< 2D` is auto-promoted to 2D (`[N]` → `[1, N]`) with a non-fatal warning — pass an explicit `pl.tile.reshape` if you want a different layout.

Mechanism: a non-trivial subscript lowers to `tensor.slice` / `tile.slice` with full-rank `shape`/`offset` plus a `drop_dims` list of the scalar-indexed axes (see the IR operator docs). The same rules apply on the assignment LHS — `C[i, j] = rhs` reshapes `rhs` back to the full-rank window before `tensor.assemble` (chained writes `C[i][j] = rhs` are not yet supported).

### Binary Operations

| Python Operator | PyPTO IR | Category |
| --------------- | -------- | -------- |
| `+` | Add | Arithmetic |
| `-` | Sub | Arithmetic |
| `*` | Mul | Arithmetic |
| `//` | FloorDiv | Arithmetic |
| `%` | FloorMod | Arithmetic |
| `/` | FloatDiv | Arithmetic |
| `**` | Pow | Arithmetic |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Eq, Ne, Lt, Le, Gt, Ge | Comparison |
| `and`, `or` | And, Or | Logical |
| `^` | Xor | Logical |
| `&` | BitAnd | Bitwise |
| `\|` | BitOr | Bitwise |
| `<<`, `>>` | BitShiftLeft, BitShiftRight | Bitwise |

**Note:** `and`/`or` are parsed from Python's `ast.BoolOp` syntax. Chained expressions like `a and b and c` are folded left-to-right into `And(And(a, b), c)`. Unlike Python, IR `And`/`Or` nodes evaluate both operands (no short-circuit semantics). The corresponding IR factory functions are `ir.and_(lhs, rhs)` and `ir.or_(lhs, rhs)`.

### Unary Operations and Functions

```python
-x              # Neg
~x              # BitNot
not x           # Not
abs(x)          # Abs
min(a, b)       # Min
max(a, b)       # Max
```

### Function/Op Calls

```python
# Explicit namespace
pl.tensor.add(a, b)                  # Tensor addition
pl.tile.load(t, [0, 0], [64, 64])      # Tile load

# Unified dispatch (auto-selects tensor/tile based on input type)
pl.add(a, b)                          # Tensor or Tile — dispatched automatically
pl.mul(tile, 2.0)                     # Tile + scalar → tile.muls
pl.exp(tile)                          # Tile → tile.exp

# Promoted ops (single-module ops accessible at pl.*)
pl.load(t, [0, 0], [64, 64])            # Promoted from block
pl.create_tensor([64], dtype=pl.FP32)       # Promoted from tensor

# System operations (synchronization primitives)
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.bar_v()                        # Vector barrier
pl.system.bar_m()                        # Matrix barrier
pl.system.bar_all()                      # Global barrier

# Cross-core operations (TPUSH/TPOP protocol)
pl.tpush_to_aic(tile0, split=0, id=0)        # Vector → Cube push on pipe 0
pl.tpush_to_aic(tile1, split=0, id=1)        # Vector → Cube push on pipe 1
tile0 = pl.tpop_from_aiv(split=0, id=0)      # Cube pops from Vector pipe 0
tile1 = pl.tpop_from_aiv(split=0, id=1)      # Cube pops from Vector pipe 1
pl.tfree_to_aiv(tile0, id=0)                 # Release slot to Vector pipe 0
pl.tfree_to_aiv(tile1, id=1)                 # Release slot to Vector pipe 1

# Cross-core pipe initialization and buffer management
buf = pl.reserve_buffer(name="slot_buf", size=4096, base=pl.AUTO)
peer = pl.import_peer_buffer(name="slot_buf", peer_func="other_func")
pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf, dir_mask=2, slot_size=512, id=0)
pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=512, id=0)
# Optional: pin the GM ring-buffer slot count (default 8 unidirectional / 4
# bidirectional) and, on a2/a3, the local slot count (must be <= slot_num).
# Size the reserved buffer yourself: a3 -> slot_size * local_slot_num,
# a5 -> slot_size * slot_num.
pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf, dir_mask=2, slot_size=512, slot_num=16, local_slot_num=4)
```

#### Cross-path arguments on unified ops

A unified `pl.<op>` accepts the union of both levels' arguments, so an argument
only the *other* dispatch path can honour is **rejected, never dropped** — a
silently discarded `b_trans` would compile wrong math. The same holds in
reverse: the scratch operand a Tensor input must omit (`pl.row_max(tensor)`) is
the one a Tile input must supply (`pl.row_max(tile, tmp_tile)`), because tile
buffer lifetimes are user-managed.

**Both directions raise `TypeError`** — these are wrong-arguments-for-this-overload
errors that Python itself raises for an unexpected keyword or a missing
required argument. Deeper validation reached through the wrapper (shape, dtype,
bounds — anything a C++ `CHECK` rejects) still raises `ValueError`, so code
guarding a whole call should catch both:

```python
pl.matmul(tile_a, tile_b, b_trans=True)   # TypeError — tile transpose is a view, not a flag
pl.rsqrt(tile, high_precision=True)       # TypeError — tile precision is selected by passing tmp
pl.div(tile, 2.0, high_precision=True)    # TypeError — high_precision needs a Tile rhs
pl.row_max(tile)                          # TypeError — Tile inputs require tmp_tile
pl.slice(tile, [64, 64], [64, 0])         # ValueError — window runs off the source tile
```

Inside a `@pl.function` body this distinction is invisible: the parser catches
both and re-raises `InvalidOperationError` with the source span.

## References

- [IR Overview](../ir/00-overview.md) - Core IR structures
- [IR Parser](../ir/07-parser.md) - Parsing Python syntax back to IR
- [Operator Registration](../ir/05-operators.md) - Op system and type inference
