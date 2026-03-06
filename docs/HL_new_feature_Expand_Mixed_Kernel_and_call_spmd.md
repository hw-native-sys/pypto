# ExpandMixedKernel Pass

Expands mixed InCore functions into properly separated AIC (Cube) and AIV (Vector) kernel functions with TPUSH/TPOP-based inter-core data communication.

## Overview

After the `OutlineIncoreScopes` pass, InCore functions may contain a mixture of orchestration-level operations (scalar reads, sub-tensor views, control flow) and compute-level operations (matmul, element-wise arithmetic, reductions). These are called **mixed InCore functions**. They can arise from:

1. **`pl.auto_incore()` scope** — the automatic chunked-loop splitting + interchange + outlining pipeline produces mixed functions where data access and compute coexist.
2. **Manually written mixed InCore functions** — users may directly write InCore functions that combine orchestration and compute logic.

The `ExpandMixedKernel` pass runs **after `OutlineIncoreScopes`** and transforms these mixed InCore functions into separate AIC and AIV kernel functions connected by `tpush_to_aiv` / `tpush_to_aic` / `tpop_from_aic` / `tpop_from_aiv` instructions.

## Motivation

Consider the InCore function produced by pa4.py after outlining:

```python
@pl.function(type=pl.FunctionType.InCore)
def paged_attention_incore_0(self, b_idx_0_out, block_table_0, context_lens_0,
                              key_cache_0, out_iter_1_outer_l1, q_idx_0_out,
                              query_0, value_cache_0) -> Tensor[[4096, 128], FP32]:
    for b_idx_0_in in pl.parallel(0, 8, 1, ...):
        for q_idx_0_in in pl.parallel(0, 2, 1, ...):
            # Orchestration-level: scalar read from global memory
            cur_seq_0 = pl.tensor.read(context_lens_0, [...])
            bn_this_batch_0 = (cur_seq_0 + 128 - 1) // 128

            # Orchestration-level: sub-tensor views
            qi_0 = pl.tensor.view(query_0, [16, 128], [cur_offset_0, 0])
            kj_0 = pl.tensor.view(key_cache_0, [128, 128], [kv_block_row_0, 0])
            vj_0 = pl.tensor.view(value_cache_0, [128, 128], [kv_block_row_0, 0])

            # Compute-level: matmul, softmax, online rescaling
            sij_0 = pl.tensor.matmul(qi_0, kj_0, b_trans=True)
            mi_0 = pl.tensor.row_max(scaled_0)
            ...
```

This function mixes tensor-level data access (`tensor.read`, `tensor.view`) with tensor-level compute (`tensor.matmul`, `tensor.row_max`, etc.). The backend codegen cannot handle this mixture directly — it expects either pure orchestration or pure compute. The `ExpandMixedKernel` pass splits this into an AIC kernel (Cube compute) and an AIV kernel (Vector data movement / element-wise ops), connected by ring-buffer-based `tpush`/`tpop` communication as defined in the [TPUSH/TPOP ISA specification](../../pto-isa/HL_ptoisa_newfeature20260306_TPUSH_TPOP.md).

## Context in the Pass Pipeline

```
... → SplitChunkedLoops → InterchangeChunkLoops → OutlineIncoreScopes → ExpandMixedKernel → ...
```

**Input**: Program with mixed InCore functions (produced by OutlineIncoreScopes or written manually).

**Output**: Program where each mixed InCore function has been expanded into a pair of AIC and AIV kernel functions, connected by `tpush`/`tpop` inter-core communication.

## Algorithm: Step-by-Step

### Step 1: Iterate Over InCore Functions

The pass walks through every InCore function in the program.

```
for each function F in program:
    if F.type != InCore:
        continue
    analyze(F)
```

### Step 2: Color Every IR Node (WHITE / GREEN / RED)

Each IR node in the function is assigned one of three colors:

| Color | Meaning | Examples |
|---|---|---|
| **WHITE** | Default. Not yet classified, or shared infrastructure (control flow, constants, loop structure) | `pl.parallel`, `pl.range`, literal constants, loop indices |
| **GREEN** | Operations and data structures belonging to **AIV** (Vector core) | `tensor.view`, `tensor.read`, `tensor.assemble`, MTE data movement, element-wise vector ops (`pl.sub`, `pl.mul`, `pl.div`, `pl.row_max`, `pl.row_sum`, `pl.exp`) |
| **RED** | Operations and data structures belonging to **AIC** (Cube core) | `tensor.matmul`, Cube-specific compute operations |

The coloring is propagated through data dependencies:
- If an operation is colored RED or GREEN, its output data structures inherit the same color.
- If a data structure is consumed only by RED operations, it is colored RED; only by GREEN operations, it is colored GREEN.
- Data structures consumed by both RED and GREEN operations are **cross-color boundaries** — these are the tiles that will require `tpush`/`tpop` communication.

```
function color_ir_nodes(F):
    for each node N in F:
        N.color = WHITE

    // Seed coloring based on operation type
    for each node N in F:
        if is_aic_operation(N):    // e.g., matmul
            N.color = RED
        elif is_aiv_operation(N):  // e.g., tensor.view, element-wise ops
            N.color = GREEN

    // Propagate colors through data flow
    propagate_colors(F)
```

### Step 3: Identify Mixed Kernels

A kernel is **mixed** if it contains nodes of all three colors (WHITE + GREEN + RED). If a kernel has only WHITE + GREEN or WHITE + RED, it is already a pure AIV or AIC kernel and does not need expansion.

```
function is_mixed_kernel(F):
    has_green = any(N.color == GREEN for N in F.nodes)
    has_red   = any(N.color == RED   for N in F.nodes)
    return has_green and has_red
```

Only mixed kernels proceed to the following steps.

### Step 4: Create `InCoreFunctionGroup` and Duplicate into AIC / AIV Kernels

#### New IR Concept: `InCoreFunctionGroup`

Before duplicating, the pass introduces a new IR hierarchy node: **`InCoreFunctionGroup`**. This node represents a **co-scheduled group of InCore functions** that must be executed together on a single core cluster (1 AIC + 2 AIV cores) with local affinity. The group encapsulates:

- The **AIC kernel** function (runs on the Cube core)
- The **AIV kernel** function (runs on both Vector cores, parameterized by `AIV_IDX`)
- Shared metadata: `DIR_MASK`, `SLOT_SIZE`, `CONSUMER_BUFFER_BASE`/`SIZE` constants, communication topology

```
// New IR node definition (conceptual):
class InCoreFunctionGroup : public IRNode {
    FunctionType type = FunctionType::InCoreFunctionGroup;

    Function* aic_kernel;          // the AIC (Cube) kernel function
    Function* aiv_kernel;          // the AIV (Vector) kernel function (parameterized by AIV_IDX)

    uint8_t   dir_mask;            // DIR_C2V, DIR_V2C, or both
    uint32_t  slot_size;           // tile size for ring buffer slots
    // ... additional group-level metadata
};
```

The `InCoreFunctionGroup` sits at the same level as `Function` in the IR module hierarchy:

```
Module
├── Function (type=Orchestration)            // orchestration function
├── Function (type=InCore)                   // non-mixed InCore function (unchanged)
├── InCoreFunctionGroup                      // NEW: expanded mixed kernel
│   ├── Function (type=InCore, core=AIC)     //   AIC kernel
│   └── Function (type=InCore, core=AIV)     //   AIV kernel
├── Function (type=InCore)                   // another non-mixed InCore function
└── ...
```

#### Why a New Hierarchy Level?

1. **Scheduling constraint**: The AIC and AIV kernels within a group **must** be co-scheduled on the same physical cluster. They share hardware flags and ring buffer resources that are local to the cluster. The IR must make this co-scheduling requirement explicit so that downstream passes (scheduling, resource allocation, code emission) can respect it.

2. **Shared DMA datapath**: Within a cluster, AIC and AIV cores have a special DMA datapath for cross-core data transfer (the TPUSH/TPOP ring buffer). This is not available across clusters. The group boundary defines the scope of this local affinity.

3. **Unified resource management**: The `CONSUMER_BUFFER_BASE`/`SIZE` constants, `GM_SLOT_BUFFER`, and flag assignments are shared resources scoped to the group. Placing them at the group level avoids duplication and ensures consistency.

4. **Backward compatibility**: Non-mixed InCore functions remain as standalone `Function` nodes — they are **not** wrapped in a group. All existing passes that iterate over `Function` nodes continue to work unchanged (see "IR Compatibility" section below).

#### Duplication

Within the new group, the mixed kernel's IR is duplicated into two kernel functions:

```
function expand_to_group(F):
    group = InCoreFunctionGroup()
    group.aic_kernel = deep_copy(F)    // will become the AIC (Cube) kernel
    group.aiv_kernel = deep_copy(F)    // will become the AIV (Vector) kernel

    group.aic_kernel.core_type = AIC
    group.aiv_kernel.core_type = AIV

    return group
```

Both copies initially contain the full IR with all colors. The subsequent steps will insert communication and prune irrelevant nodes.

#### IR Compatibility: Ensuring Existing Passes Remain Intact

Adding `InCoreFunctionGroup` as a new hierarchy level requires careful design so that existing passes — which expect to iterate over `Function` nodes — are not broken.

**Design principle**: The `InCoreFunctionGroup` is **transparent** to passes that do not know about it. The module provides two iteration interfaces:

```
// Interface 1: Flat iteration (backward-compatible)
// Yields all Function nodes, including those inside groups.
// Existing passes use this — they see the AIC and AIV kernels as regular InCore functions.
module.functions()  →  [orchestration, non_mixed_incore, aic_kernel, aiv_kernel, ...]

// Interface 2: Group-aware iteration (new)
// Yields top-level items: standalone Functions and InCoreFunctionGroups.
// New passes that need to understand co-scheduling use this.
module.top_level_items()  →  [orchestration, non_mixed_incore, group, ...]
```

**Compatibility guarantees:**

| Pass Category | Iteration Method | Behavior |
|---|---|---|
| **Existing function-level passes** (e.g., `AllocateMemoryAddr`, DCE, CSE) | `module.functions()` | See AIC and AIV kernels as independent InCore functions. Work correctly without modification. |
| **Scheduling / code emission passes** | `module.top_level_items()` | See `InCoreFunctionGroup` nodes and can enforce co-scheduling, cluster affinity, and shared resource allocation. |
| **Orchestration-level passes** | `module.functions()` | See orchestration functions that call the group. The call site in the orchestration IR references the `InCoreFunctionGroup` node, which the code emitter expands into the appropriate AIC + AIV kernel launches. |

**Function attributes**: Each function within a group carries an attribute indicating its role:

```
F_aic.attrs["core_type"] = "AIC"
F_aic.attrs["group"] = group_id         // back-reference to parent group

F_aiv.attrs["core_type"] = "AIV"
F_aiv.attrs["group"] = group_id
F_aiv.attrs["aiv_parameterized"] = True  // has AIV_IDX parameter
```

Existing passes that do not inspect these attributes are unaffected. Passes that need cluster-aware behavior (e.g., memory allocator for `CONSUMER_BUFFER_BASE` reservation) can check `attrs["group"]` to find the peer function.

**Call site in orchestration**: The orchestration function's call to the original mixed kernel is replaced with a call to the `InCoreFunctionGroup`:

```
// Before:
call @paged_attention_incore_0(...)          // single mixed kernel call

// After:
call_group @paged_attention_group_0(...)     // group call — expands to:
                                              //   @paged_attention_incore_0_aic(...)
                                              //   @paged_attention_incore_0_aiv(..., AIV_IDX=0)
                                              //   @paged_attention_incore_0_aiv(..., AIV_IDX=1)
```

The `call_group` IR node is a new call instruction that:
- References an `InCoreFunctionGroup` instead of a single `Function`.
- The code emitter expands it into the appropriate sequence of kernel launches.
- Existing passes that analyze call graphs see `call_group` as a call node with the group's constituent functions as callees.

**Verification**: A post-pass verifier checks:
1. Every `InCoreFunctionGroup` has exactly one AIC kernel and one AIV kernel.
2. Both kernels reference the same `group_id`.
3. The AIC kernel contains no GREEN-colored operations; the AIV kernel contains no RED-colored operations.
4. `tpush`/`tpop` operations in the AIC kernel have matching counterparts in the AIV kernel.
5. `CONSUMER_BUFFER_BASE`/`SIZE` constants are consistent between the two kernels.

### Step 5: Insert Pipe Initialization

Based on the [TPUSH/TPOP ISA specification](../../pto-isa/HL_ptoisa_newfeature20260306_TPUSH_TPOP.md), insert initialization calls at the top of each kernel:

- **In `F_aic`**: Insert `aic_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)` at the function entry.
- **In `F_aiv`**: Insert `aiv_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)` at the function entry.

The `DIR_MASK` is determined by analyzing cross-color data flow: if data flows from RED→GREEN, `DIR_C2V` is active; if GREEN→RED, `DIR_V2C` is active; if both, bidirectional.

Also insert `pl.reserve_buffer` / `pl.import_peer_buffer` declarations for A5 platform support (see TPUSH/TPOP spec, DSL Grammar section).

### Step 6: Insert TPUSH/TPOP at Cross-Color Boundaries

For each **cross-color tile** (a tile produced by one color and consumed by the other):

**Case A: Tile produced by AIC (RED), consumed by AIV (GREEN)**

Data flows Cube → Vector (C2V direction):

```
// In F_aic, after the RED producer node:
    tpush_to_aiv(tile, AIV_IDX)          // marked RED — stays in AIC kernel

// In F_aiv, at the corresponding consumption point:
    tile = tpop_from_aic(tile, AIV_IDX)  // marked GREEN — stays in AIV kernel
```

**Case B: Tile produced by AIV (GREEN), consumed by AIC (RED)**

Data flows Vector → Cube (V2C direction):

```
// In F_aiv, after the GREEN producer node:
    tpush_to_aic(tile, AIV_IDX)          // marked GREEN — stays in AIV kernel

// In F_aic, at the corresponding consumption point:
    tile = tpop_from_aiv(tile, AIV_IDX)  // marked RED — stays in AIC kernel
```

### Step 7: Prune by Color

Remove nodes that do not belong to each kernel:

- **In `F_aiv`**: Remove all **RED** nodes (AIC operations). Only WHITE and GREEN nodes remain, plus `tpop_from_aic` (GREEN) and `tpush_to_aic` (GREEN).
- **In `F_aic`**: Remove all **GREEN** nodes (AIV operations). Only WHITE and RED nodes remain, plus `tpush_to_aiv` (RED) and `tpop_from_aiv` (RED).

```
function prune_by_color(F_aic, F_aiv):
    // AIC kernel: remove GREEN
    for each node N in F_aic:
        if N.color == GREEN:
            remove(N)

    // AIV kernel: remove RED
    for each node N in F_aiv:
        if N.color == RED:
            remove(N)
```

### Step 8: Dead Code Elimination

After pruning, some WHITE nodes may become dead code — they no longer feed any live computation. Run a standard dead code elimination (DCE) pass on both kernels:

```
function eliminate_dead_code(F):
    repeat:
        for each node N in F:
            if N has no users and N is not a side-effecting operation:
                remove(N)
    until no changes
```

Side-effecting operations (e.g., `tpush_*`, `tpop_*`, `SET flag`, `WAIT flag`, `pl.assemble`) are never removed.

### Step 9: AIV Kernel — Split for Dual Vector Cores (AIV_IDX)

Each cluster has **two** buddy Vector cores (AIV_IDX = 0 and AIV_IDX = 1). The AIV kernel must be parameterized to run on both, each processing **half** of the workload.

**9a. Add `AIV_IDX` argument to `F_aiv`:**

```
// Before:
def F_aiv(self, ...) -> ...:

// After:
def F_aiv(self, ..., AIV_IDX: uint8_t) -> ...:
```

**9b. Split tensor variables by `AIV_IDX`:**

For each tensor variable `T` in the AIV kernel, replace it with a **half-sized** tensor `T_half` addressed by `AIV_IDX`. The split decision is governed by two principles in strict priority order:

1. **Correctness first**: The split must not change the mathematical result of any operation that consumes the tensor.
2. **Performance second**: Among correct splits, prefer the highest (outermost) axis to preserve lower-axis contiguous memory layout for efficient Vector core execution.

#### The Fundamental Correctness Constraint — Whole-Chain Analysis

**The compiler cannot determine whether an operation is splittable by examining that operation alone.** Instead, the **entire dependency chain** must be considered. Splitting a tensor at some operation means that subsequent operations on both AIV cores will only see their respective halves. If any downstream operation in the dependency chain is unsplittable and requires the full (un-split) tensor, the program would be **functionally incorrect** — there is no mechanism to gather data across two AIV cores at runtime.

**Example of incorrect per-operation analysis:**

```
// Suppose the compiler naively marks each op independently:
A = pl.tensor.view(input, [16, 128], ...)   // ← looks splittable (axis 0)
B = pl.sub(A, scalar)                        // ← looks splittable (element-wise)
C = pl.reshape(B, [1, 2048])                 // ← shape change, only axis has dim 1
D = pl.row_sum(C)                            // ← reduction on axis 1 → UNSPLITTABLE

// If the compiler splits A on axis 0, then B becomes [8, 128] per core.
// But C = reshape(B, [1, 2048]) would now try to reshape [8, 128] = 1024 elements
// into [1, 2048] — this is mathematically wrong. Even if reshape were reinterpreted,
// D = row_sum(C) reduces along axis 1 and needs ALL 2048 elements on one core.
// The two halves are on separate AIV cores with no way to recombine — INCORRECT.
```

The fundamental invariant is: **a tensor is only safely splittable if every operation in its downstream dependency chain — all the way to a communication boundary (tpush) or output (assemble) — is also safely splittable in a compatible fashion.** The two AIV cores have no cross-core communication; once data is split, it stays split until it exits the AIV kernel.

#### Operation Classification for Split Safety (Local Properties)

Each operation has **local** split properties — which axes are forbidden. These local properties are necessary but **not sufficient** for determining whether splitting actually occurs. The whole-chain algorithm (below) uses these properties as building blocks.

| Operation | Type | Reduction Axis | Locally Splittable Axes | Notes |
|---|---|---|---|---|
| `pl.sub(A, B, out)` | Element-wise | None | Any axis | Independent per-element |
| `pl.mul(A, B, out)` | Element-wise | None | Any axis | Independent per-element |
| `pl.div(A, B, out)` | Element-wise | None | Any axis | Independent per-element |
| `pl.exp(A, out)` | Element-wise | None | Any axis | Independent per-element |
| `pl.row_max(A, out)` | Row reduction | Last axis (innermost) | All axes **except** the last | Splitting the reduction axis produces incorrect partial results |
| `pl.row_sum(A, out)` | Row reduction | Last axis (innermost) | All axes **except** the last | Splitting the reduction axis produces incorrect partial sums |
| `tensor.view(...)` | Data access | None | Any axis | Pure address computation |
| `tensor.read(...)` | Data access | None | Any axis | Pure data load |
| `tensor.assemble(...)` | Data writeback | None | Any axis | Pure data store |
| `tpush_to_aic(...)` | Communication | None | Any axis | Communication boundary |
| `tpop_from_aic(...)` | Communication | None | Any axis | Communication boundary |

> **General rule**: For any reduction operation `reduce(A, axis=k, out)`, axis `k` is **locally forbidden** for splitting. Element-wise operations have no locally forbidden axes. Data movement operations have no locally forbidden axes.

#### Whole-Chain Splittability Algorithm — "Duplicated by Default"

The algorithm starts conservatively: **all operations are initially marked as `DUPLICATED`** (both AIV cores execute identically on full-size data). Then it iteratively discovers end-to-end dependency chains that can safely be converted to `SPLIT`.

This "duplicated by default, opt-in to split" design guarantees functional correctness for **any** input kernel — the worst case is that both cores redundantly compute everything (correct but no parallelism benefit), and the best case is that the compiler finds maximal splittable chains for full 2× throughput.

**Definitions:**

- **`DUPLICATED`**: Both AIV cores execute the operation on the same full-size data. The operation is not modified. This is always correct but wastes one core's compute.
- **`SPLIT`**: Each AIV core executes the operation on its half of the data (addressed by `AIV_IDX`). This is correct **only if** the entire chain from data source to data sink is uniformly splittable.
- **Dependency chain**: A maximal connected subgraph of operations linked by def-use edges within the AIV kernel. A chain is bounded by **sources** (function parameters, `tpop_from_aic`) and **sinks** (`tpush_to_aic`, `tensor.assemble`).
- **Compatible split**: All operations in a chain agree on a common split axis, and no operation has that axis as forbidden.

**Algorithm:**

```
function compute_split_chains(function F_aiv):
    // ─────────────────────────────────────────────────────
    // Phase 0: Build the dependency graph
    // ─────────────────────────────────────────────────────
    // Build def-use edges: for each variable, track which op produces it
    // and which ops consume it. Identify sources and sinks.
    G = build_def_use_graph(F_aiv)

    // ─────────────────────────────────────────────────────
    // Phase 1: Initialize all ops as DUPLICATED
    // ─────────────────────────────────────────────────────
    for each operation OP in F_aiv:
        OP.mode = DUPLICATED

    // ─────────────────────────────────────────────────────
    // Phase 2: Identify candidate chains
    // ─────────────────────────────────────────────────────
    // A "chain" is a maximal connected subgraph of tensor operations
    // bounded by sources (parameters, tpop_from_aic) and sinks
    // (tpush_to_aic, tensor.assemble to output).
    chains = extract_dependency_chains(G)

    // ─────────────────────────────────────────────────────
    // Phase 3: For each chain, check end-to-end splittability
    // ─────────────────────────────────────────────────────
    for each chain C in chains:
        // 3a. Collect the union of forbidden axes across ALL ops in the chain
        all_forbidden = {}
        for each operation OP in C:
            all_forbidden = all_forbidden ∪ local_forbidden_axes(OP)

        // 3b. Collect all tensor shapes in the chain and find a
        //     common split axis that works for every tensor
        candidate_axis = find_common_split_axis(C, all_forbidden)

        if candidate_axis == NONE:
            // No common axis can safely split the entire chain.
            // The whole chain stays DUPLICATED — correct by construction.
            continue

        // 3c. Verify compatibility: every tensor in the chain must have
        //     shape[candidate_axis] > 1 and divisible by 2
        if not all_tensors_splittable_on(C, candidate_axis):
            continue

        // 3d. All checks pass — convert the entire chain to SPLIT
        for each operation OP in C:
            OP.mode = SPLIT
            OP.split_axis = candidate_axis

    // ─────────────────────────────────────────────────────
    // Phase 4: Return the classification
    // ─────────────────────────────────────────────────────
    return {op.name: op.mode for op in F_aiv}
```

```
function find_common_split_axis(chain C, forbidden_axes):
    // Collect all tensor shapes in the chain
    shapes = {T.shape for each tensor T produced or consumed by ops in C
              where T is not a function parameter}

    // Try axes from highest (outermost) to lowest (innermost)
    max_ndim = max(len(s) for s in shapes)
    for axis in range(max_ndim):
        if axis in forbidden_axes:
            continue
        // Check that every tensor has this axis and it is splittable
        valid = true
        for each shape S in shapes:
            if axis >= len(S):
                valid = false; break       // tensor has fewer dims
            if S[axis] <= 1:
                valid = false; break       // dim too small to halve
            if S[axis] % 2 != 0:
                valid = false; break       // not evenly divisible
        if valid:
            return axis
    return NONE
```

**Key properties of this algorithm:**

1. **Correctness by construction**: Since all ops start as `DUPLICATED`, the default output is always correct. Chains are only promoted to `SPLIT` when end-to-end analysis confirms safety.
2. **No cross-core data dependency**: A chain is only split if every operation in it can work on its half of the data independently. Since the two AIV cores cannot communicate, this is the only safe strategy.
3. **Handles arbitrary kernels**: Even pathological cases (e.g., a single unsplittable op that consumes the output of a splittable chain) are handled correctly — the entire chain remains `DUPLICATED`.
4. **Incremental**: Chains are analyzed independently. Adding a new unsplittable operation to a kernel only affects the chain(s) that include it.

#### Why Per-Operation Analysis Is Insufficient

The earlier approach of checking each operation independently suffers from a critical flaw:

```
// Scenario: op A is locally splittable, but its output feeds into unsplittable op D

A = pl.sub(X, Y)           // element-wise → locally splittable on any axis
B = pl.reshape(A, [1, N])  // shape collapse → axis 0 has dim 1
C = pl.row_max(B)           // reduction on axis 1 → UNSPLITTABLE

// Per-operation analysis would split A (it looks safe locally),
// but then B and C receive half-sized data and produce wrong results.
// With whole-chain analysis, {A, B, C} form a single chain.
// The chain has no valid common split axis → entire chain stays DUPLICATED.
```

Even when an operation's output splits correctly, the downstream consumer may reshape, transpose, or reduce in a way that requires the full data. The only safe approach is to verify the entire producer→consumer chain agrees on a common split.

#### Handling Chains with Mixed Splittable/Unsplittable Segments

In some cases, a long dependency chain may contain a segment that is unsplittable in the middle, with splittable segments on either side. The chain extraction phase handles this by treating the unsplittable segment as a barrier that breaks the chain into separate sub-chains:

```
// Example:
A = view(input, [16, 128])       // ─┐
B = sub(A, scalar)                //  │ sub-chain 1: splittable on axis 0
C = tpush_to_aic(B)              // ─┘ (sink: communication boundary)

D = tpop_from_aic()              // ─┐
E = reshape(D, [1, 2048])        //  │ sub-chain 2: unsplittable (no valid axis)
F = row_sum(E)                   //  │
G = tpush_to_aic(F)              // ─┘

// Result: sub-chain 1 is SPLIT, sub-chain 2 is DUPLICATED.
// Correct, because D receives fresh data from tpop_from_aic (a source boundary)
// — it does not depend on the split output of sub-chain 1.
```

Communication boundaries (`tpush_to_aic`, `tpop_from_aic`) naturally serve as chain separators. Data that crosses the AIC↔AIV boundary is explicitly re-distributed, so the split/duplicate decision of one segment does not contaminate another.

#### Split Axis Selection (Within a Splittable Chain)

Once a chain is confirmed as end-to-end splittable, the axis selection follows the same priority as before:

```
function choose_split_axis_for_chain(chain C):
    all_forbidden = union of local_forbidden_axes(OP) for OP in C
    return find_common_split_axis(C, all_forbidden)
    // Prefers highest (outermost) axis for memory contiguity
```

#### Effect on AIC↔AIV Communication

The `SPLIT` vs `DUPLICATED` classification determines how data flows between the AIC and AIV kernels:

| Direction | SPLIT (chain is split) | DUPLICATED (chain is duplicated) |
|---|---|---|
| AIC → AIV (`tpush_to_aiv`) | Split tensor into halves, push half to each AIV | Push **full** tensor to both AIV cores |
| AIV → AIC (`tpop_from_aiv`) | Pop half from each AIV, reassemble | Pop from **both** AIV cores, use only `AIV_IDX=0` result |
| AIV `tpop_from_aic` | Receive half-size tensor | Receive **full-size** tensor |
| AIV `tpush_to_aic` | Send half-size with `AIV_IDX` | Send full-size with `AIV_IDX` |

#### Detailed Examples

**Example 1: Fully splittable chain**

All operations in the chain are element-wise or have compatible reduction axes. The whole chain is promoted to `SPLIT`.

```
// Chain: view → sub → exp → tpush_to_aic
// All element-wise, no forbidden axes.
// Common split axis = 0 (dim 16, divisible by 2) → SPLIT

// Before:
qi = view(query, [16, 128], [offset, 0])
centered = sub(qi, max_val)         // [16, 128]
exp_vals = exp(centered)            // [16, 128]
tpush_to_aic(exp_vals, AIV_IDX)

// After (SPLIT on axis 0):
qi = view(query, [8, 128], [offset + AIV_IDX * 8, 0])
centered = sub(qi, max_val)         // [8, 128]
exp_vals = exp(centered)            // [8, 128]
tpush_to_aic(exp_vals, AIV_IDX)
```

**Example 2: Chain containing row reduction — splittable on non-reduction axis**

```
// Chain: tpop_from_aic → row_max → sub → exp → tpush_to_aic
// row_max forbids axis 1. Common split axis = 0 (dim 16) → SPLIT

// Before:
sij = tpop_from_aic()              // [16, 128]
mi = row_max(sij)                  // [16, 1]
centered = sub(sij, mi)            // [16, 128]
exp_vals = exp(centered)           // [16, 128]
tpush_to_aic(exp_vals, AIV_IDX)

// After (SPLIT on axis 0):
sij = tpop_from_aic(AIV_IDX)       // [8, 128]
mi = row_max(sij)                  // [8, 1]
centered = sub(sij, mi)            // [8, 128]
exp_vals = exp(centered)           // [8, 128]
tpush_to_aic(exp_vals, AIV_IDX)
```

**Example 3: Chain with shape collapse — UNSPLITTABLE, stays DUPLICATED**

```
// Chain: tpop_from_aic → row_max → reshape([1, 16]) → row_max → sub → ...
// reshape produces [1, 16]: axis 0 has dim 1, axis 1 is forbidden by row_max.
// No valid common axis → entire chain stays DUPLICATED.

// Both AIV cores execute identically at full size:
sij = tpop_from_aic()              // [16, 128] — same on both cores
mi = row_max(sij)                  // [16, 1]
mi_flat = reshape(mi, [1, 16])     // [1, 16]
global_max = row_max(mi_flat)      // [1, 1]
centered = sub(sij, global_max)    // [16, 128]
// ... both cores produce identical results
```

**Example 4: Mixed kernel with two separate chains**

```
// Sub-chain A (splittable): view → tpush_to_aic
//   qi = view(query, [16, 128], ...) → split axis 0 → SPLIT
//
// Sub-chain B (unsplittable): tpop_from_aic → reshape → row_sum → tpush_to_aic
//   No valid axis → DUPLICATED
//
// These are independent chains (separated by AIC boundary).
// Sub-chain A is SPLIT, sub-chain B is DUPLICATED.
// Both decisions are correct and do not interfere with each other.
```

#### Summary of the "Duplicated by Default" Split Decision Flow

```
1. Build dependency graph for the AIV kernel
2. Initialize ALL operations as DUPLICATED
3. Extract dependency chains (bounded by sources and sinks)
4. For each chain:
   a. Collect union of forbidden axes across all ops in the chain
   b. Find a common split axis valid for all tensors in the chain
   c. If found: promote entire chain to SPLIT
   d. If not found: chain remains DUPLICATED (correct by default)
5. Apply the classification:
   - SPLIT ops: halve tensor shapes on the chosen axis, add AIV_IDX offsets
   - DUPLICATED ops: no modification, both cores compute identically
```

**9c. Update `tpush_to_aic` / `tpop_from_aic` in `F_aiv`:**

- Pass `AIV_IDX` as an argument to each `tpush_to_aic` and `tpop_from_aic` call.
- For **split** tensors: the tile size in each `tpush`/`tpop` is halved (each Vector core pushes/pops half the data).
- For **replicated** (unsplittable) tensors: the `tpush`/`tpop` retains the full tile size. Both AIV cores participate, each sending the same full data with its own `AIV_IDX`.

```
// Split case — both AIV cores participate with half data:
tpush_to_aic(half_tile, AIV_IDX)

// Replicated case — both AIV cores send identical full data:
tpush_to_aic(full_tile, AIV_IDX)
```

**9d. Update `F_aic` — double the TPUSH/TPOP for two AIV cores:**

Back in the AIC kernel, each `tpush_to_aiv`/`tpop_from_aiv` must account for the two Vector cores. The behavior depends on whether the corresponding tensor was **split** or **replicated** in the AIV kernel:

**Split case** — the AIC pushes/pops two halves, one for each Vector core:

```
// tpush_to_aiv (AIC → AIV), split tensor:
// Before:
tpush_to_aiv(full_tile, AIV_IDX)

// After:
half_tile_0 = first_half(full_tile)
half_tile_1 = second_half(full_tile)
tpush_to_aiv(half_tile_0, 0)    // push to Vector 0
tpush_to_aiv(half_tile_1, 1)    // push to Vector 1
```

```
// tpop_from_aiv (AIV → AIC), split tensor:
// Before:
tpop_from_aiv(full_tile, AIV_IDX)

// After:
tpop_from_aiv(half_tile_0, 0)    // pop from Vector 0
tpop_from_aiv(half_tile_1, 1)    // pop from Vector 1
full_tile = concat(half_tile_0, half_tile_1)
```

**Replicated case** — both AIV cores have the same full-size data:

```
// tpush_to_aiv (AIC → AIV), replicated tensor:
// Push the FULL tensor to BOTH Vector cores (no splitting)
tpush_to_aiv(full_tile, 0)      // full tile to Vector 0
tpush_to_aiv(full_tile, 1)      // full tile to Vector 1

// tpop_from_aiv (AIV → AIC), replicated tensor:
// Both AIV cores push identical full data; pop from both, use only one
full_tile = tpop_from_aiv(0)    // use result from Vector 0
__discard = tpop_from_aiv(1)    // must pop to unblock, discard result
```

### Step 10: Dump Output

After all steps, the program IR contains:
- For each original mixed InCore function: an **`InCoreFunctionGroup`** containing one AIC kernel and one AIV kernel (parameterized by `AIV_IDX`).
- Non-mixed InCore functions remain as standalone `Function` nodes (unchanged).
- The orchestration function uses `call_group` to invoke the group, which the code emitter expands into the AIC kernel call + two AIV kernel calls.

```
// Original IR (before ExpandMixedKernel):
Module:
├── orchestration()
│       for ...:
│           call @paged_attention_incore_0(...)     // mixed kernel
└── Function @paged_attention_incore_0 (type=InCore)  // mixed

// After ExpandMixedKernel:
Module:
├── orchestration()
│       for ...:
│           call_group @paged_attention_group_0(...)  // group call
│
├── InCoreFunctionGroup @paged_attention_group_0
│   ├── Function @paged_attention_incore_0_aic (type=InCore, core=AIC)
│   └── Function @paged_attention_incore_0_aiv (type=InCore, core=AIV, param=AIV_IDX)
│
└── Function @some_other_incore (type=InCore)       // non-mixed, unchanged
```

The `call_group` expands at code emission time to:

```
paged_attention_incore_0_aic(...)            // AIC kernel (Cube)
paged_attention_incore_0_aiv(..., AIV_IDX=0) // AIV kernel (Vector 0)
paged_attention_incore_0_aiv(..., AIV_IDX=1) // AIV kernel (Vector 1)
```

All three kernels are guaranteed to be co-scheduled on the same physical cluster (1 AIC + 2 AIV cores).

## Complete Algorithm Summary

```
ExpandMixedKernel(program):
    for each InCore function F in program:

        // Step 2: Color IR nodes
        color_ir_nodes(F)    // WHITE, GREEN, RED

        // Step 3: Check if mixed
        if not is_mixed_kernel(F):
            continue

        // Step 4: Create group and duplicate
        group = InCoreFunctionGroup()
        F_aic = deep_copy(F);  F_aic.core_type = AIC
        F_aiv = deep_copy(F);  F_aiv.core_type = AIV
        group.aic_kernel = F_aic
        group.aiv_kernel = F_aiv

        // Step 5: Insert pipe initialization
        insert_pipe_init(F_aic, "aic")
        insert_pipe_init(F_aiv, "aiv")

        // Step 6: Insert tpush/tpop at cross-color boundaries
        for each cross-color tile T:
            if T is RED→GREEN:
                insert tpush_to_aiv after producer in F_aic   (RED)
                insert tpop_from_aic at consumer in F_aiv      (GREEN)
            elif T is GREEN→RED:
                insert tpush_to_aic after producer in F_aiv    (GREEN)
                insert tpop_from_aiv at consumer in F_aic      (RED)

        // Step 7: Prune by color
        remove all GREEN nodes from F_aic
        remove all RED nodes from F_aiv

        // Step 8: Dead code elimination
        eliminate_dead_code(F_aic)
        eliminate_dead_code(F_aiv)

        // Step 9: AIV dual-core split (duplicated-by-default, whole-chain analysis)
        add AIV_IDX argument to F_aiv
        initialize ALL operations in F_aiv as DUPLICATED
        chains = extract_dependency_chains(F_aiv)
        for each chain C in chains:
            forbidden = union of local_forbidden_axes(OP) for OP in C
            axis = find_common_split_axis(C, forbidden)
            if axis != NONE:
                // Entire chain is safely splittable end-to-end
                mark all ops in C as SPLIT with split_axis = axis
                split all tensors in C on axis by AIV_IDX (half size)
                update tpush/tpop with AIV_IDX and half tiles
                in F_aic: double corresponding tpush/tpop for AIV_IDX=0,1
            // else: chain stays DUPLICATED — both cores compute identically
            //   in F_aic: tpush_to_aiv sends full data to both cores
            //   in F_aic: tpop_from_aiv pops from both, uses AIV_IDX=0 only

        // Step 10: Output
        replace F with group (InCoreFunctionGroup) in program
        replace call to F in orchestration with call_group to group
        // call_group expands to: F_aic once, F_aiv twice (AIV_IDX=0,1)
```

## Example: Before and After

### Before (Mixed InCore Function)

```python
@pl.incore
def mixed_kernel(query, key_cache, value_cache, context_lens, out):
    # GREEN: data loading (AIV)
    qi = pl.tensor.view(query, [16, 128], [offset, 0])
    kj = pl.tensor.view(key_cache, [128, 128], [kv_offset, 0])
    vj = pl.tensor.view(value_cache, [128, 128], [kv_offset, 0])

    # RED: compute (AIC)
    sij = pl.matmul(qi, kj, b_trans=True)       # Cube matmul

    # GREEN: post-processing (AIV)
    mi = pl.row_max(sij)
    pij = pl.exp(pl.sub(sij, mi))
    li = pl.row_sum(pij)

    # RED: compute (AIC)
    oi = pl.matmul(pij, vj)                      # Cube matmul

    # GREEN: write back (AIV)
    pl.assemble(out, [offset, 0], oi)
```

### After (Expanded into AIC + AIV Kernels)

**AIC Kernel (Cube):**

```python
@pl.incore
def mixed_kernel_aic(query, key_cache, value_cache, out, GM_SLOT_BUFFER, ...):
    aic_initialize_pipe(DIR_C2V | DIR_V2C, SLOT_SIZE, GM_SLOT_BUFFER, ...)

    # Receive qi from AIV (two halves)
    tpop_from_aiv(qi_half_0, 0)
    tpop_from_aiv(qi_half_1, 1)
    qi = concat(qi_half_0, qi_half_1)

    # Receive kj from AIV
    tpop_from_aiv(kj_half_0, 0)
    tpop_from_aiv(kj_half_1, 1)
    kj = concat(kj_half_0, kj_half_1)

    # RED: Cube matmul
    sij = pl.matmul(qi, kj, b_trans=True)

    # Push sij to AIV (two halves)
    tpush_to_aiv(first_half(sij), 0)
    tpush_to_aiv(second_half(sij), 1)

    # Receive pij from AIV
    tpop_from_aiv(pij_half_0, 0)
    tpop_from_aiv(pij_half_1, 1)
    pij = concat(pij_half_0, pij_half_1)

    # Receive vj from AIV
    tpop_from_aiv(vj_half_0, 0)
    tpop_from_aiv(vj_half_1, 1)
    vj = concat(vj_half_0, vj_half_1)

    # RED: Cube matmul
    oi = pl.matmul(pij, vj)

    # Push oi to AIV (two halves)
    tpush_to_aiv(first_half(oi), 0)
    tpush_to_aiv(second_half(oi), 1)
```

**AIV Kernel (Vector, parameterized by AIV_IDX):**

```python
@pl.incore
def mixed_kernel_aiv(query, key_cache, value_cache, out, GM_SLOT_BUFFER, AIV_IDX, ...):
    aiv_initialize_pipe(DIR_C2V | DIR_V2C, SLOT_SIZE, GM_SLOT_BUFFER, ...)

    # GREEN: data loading (half size, indexed by AIV_IDX)
    qi_half = pl.tensor.view(query, [8, 128], [offset + AIV_IDX * 8, 0])
    kj_half = pl.tensor.view(key_cache, [64, 128], [kv_offset + AIV_IDX * 64, 0])
    vj_half = pl.tensor.view(value_cache, [64, 128], [kv_offset + AIV_IDX * 64, 0])

    # Push qi, kj to AIC
    tpush_to_aic(qi_half, AIV_IDX)
    tpush_to_aic(kj_half, AIV_IDX)

    # Receive sij from AIC (half)
    tpop_from_aic(sij_half, AIV_IDX)

    # GREEN: post-processing (half size)
    mi_half = pl.row_max(sij_half)
    pij_half = pl.exp(pl.sub(sij_half, mi_half))
    li_half = pl.row_sum(pij_half)

    # Push pij, vj to AIC
    tpush_to_aic(pij_half, AIV_IDX)
    tpush_to_aic(vj_half, AIV_IDX)

    # Receive oi from AIC (half)
    tpop_from_aic(oi_half, AIV_IDX)

    ......

    # GREEN: write back (half size)
    pl.assemble(out, [offset + AIV_IDX * 8, 0], oi_half)
```

**Orchestration (using `call_group`):**

```python
def orchestration(...):
    for ...:
        call_group mixed_kernel_group(...)
        # Expands at code emission to:
        #   mixed_kernel_aic(...)
        #   mixed_kernel_aiv(..., AIV_IDX=0)
        #   mixed_kernel_aiv(..., AIV_IDX=1)
```

## SPMD Kernel Launch: `call_spmd_function` and `call_spmd_group`

### Motivation

In many workloads, the same InCore kernel (or function group) needs to be executed across **multiple core clusters in parallel** — each cluster processing a different data partition. This is an SPMD (Single Program, Multiple Data) execution pattern. Rather than requiring the programmer to manually issue multiple calls with explicit index arithmetic, the compiler provides SPMD call instructions that express this pattern directly.

A key motivation for introducing `call_spmd_function` and `call_spmd_group` is **interoperability with legacy kernels from other programming frontends**. Many existing kernel ecosystems — including **AscendC**, **TiLang**, **Triton**, **CuTile**, and **CUDA** — assume an SPMD execution model where kernels are written to be launched across multiple cores with an implicit `block_idx` / `thread_idx` style index. These legacy kernels (or mixed kernel function groups) may already be compiled and available as binary objects or IR modules.

By providing `call_spmd_function` and `call_spmd_group` as first-class call grammar in pypto:

1. **Code reuse**: A pypto orchestration function can directly call legacy SPMD kernels written in AscendC, TiLang, Triton, CuTile, or CUDA without rewriting them in pypto's DSL. The orchestration layer simply wraps the legacy kernel call in `pl.call_spmd(...)` with the appropriate `spmd_size`.

2. **Unified scheduling**: The **pypto-runtime** (which is intentionally kept simple) can uniformly launch and schedule both pypto-native kernels and legacy SPMD kernels using the same dispatch mechanism. The runtime does not need to distinguish between kernel origins — it just populates `spmd_idx` and `spmd_size` and dispatches to available clusters.

3. **Incremental migration**: Teams can incrementally migrate from legacy frontends to pypto. The orchestration layer can mix pypto-native `call` / `call_group` calls with `call_spmd_function` / `call_spmd_group` calls to legacy kernels in the same program.

4. **Function group interop**: Legacy mixed kernel function groups (e.g., an AscendC kernel pair that already implements AIC/AIV splitting) can be wrapped as an `InCoreFunctionGroup` and invoked via `call_spmd_group`, gaining pypto's orchestration and scheduling capabilities without modifying the kernel code.

```
Example: pypto orchestration calling a legacy AscendC SPMD kernel

    pypto orchestration function:
    ┌─────────────────────────────────────────────────┐
    │  // Native pypto kernel (non-SPMD)              │
    │  call @preprocess_kernel(input, temp)            │
    │                                                  │
    │  // Legacy AscendC kernel (SPMD, 8 clusters)    │
    │  call_spmd_function @ascendc_matmul(             │
    │      temp, weights, output, spmd_size=8)         │
    │                                                  │
    │  // Native pypto function group (SPMD)           │
    │  call_spmd_group @postprocess_group(             │
    │      output, final, spmd_size=4)                 │
    └─────────────────────────────────────────────────┘
```

### SPMD Parameters

Any InCore function or `InCoreFunctionGroup` can be promoted to an **SPMD kernel** by adding two implicit parameters:

| Parameter | Type | Description |
|---|---|---|
| `spmd_idx` | `uint32_t` | The index of this particular instance (0-based). Each parallel instance receives a unique value. |
| `spmd_size` | `uint32_t` | The total number of parallel instances being launched. |

These parameters are appended to the function's argument list. Inside the kernel, `spmd_idx` and `spmd_size` are used to partition the workload — each instance processes a slice of the data identified by its `spmd_idx`.

```python
@pl.incore
def my_kernel(..., spmd_idx: uint32_t, spmd_size: uint32_t):
    # Each instance processes a different batch slice
    batch_start = spmd_idx * batch_per_instance
    batch_end   = batch_start + batch_per_instance
    # ... process data[batch_start:batch_end] ...
```

### `call_spmd_function`

**Syntax** (in orchestration IR):

```
call_spmd_function @kernel_name(args..., spmd_size=N)
```

**Semantics**: Launches `N` parallel instances of the InCore function `@kernel_name`, where each instance `i` receives `spmd_idx=i` and `spmd_size=N`.

**Compiler expansion**: The compiler converts a single `call_spmd_function` into `N` sequential (or hardware-parallel) calls with incrementing `spmd_idx`:

```
// Before (SPMD call):
call_spmd_function @my_kernel(args..., spmd_size=4)

// After expansion:
call @my_kernel(args..., spmd_idx=0, spmd_size=4)   // cluster 0
call @my_kernel(args..., spmd_idx=1, spmd_size=4)   // cluster 1
call @my_kernel(args..., spmd_idx=2, spmd_size=4)   // cluster 2
call @my_kernel(args..., spmd_idx=3, spmd_size=4)   // cluster 3
```

Each call is dispatched to a different core cluster. The hardware scheduler maps `spmd_idx` to physical cluster IDs.

**DSL syntax** (pypto Python frontend):

```python
def orchestration(query, key_cache, value_cache, out, ...):
    # Launch the kernel on 4 clusters in parallel
    pl.call_spmd(my_kernel, args=(query, key_cache, value_cache, out),
                 spmd_size=4)
```

### `call_spmd_group`

**Syntax** (in orchestration IR):

```
call_spmd_group @group_name(args..., spmd_size=N)
```

**Semantics**: Launches `N` parallel instances of the `InCoreFunctionGroup` `@group_name`. Each instance `i` consists of the full group (1 AIC + 2 AIV kernels co-scheduled on one cluster), and each kernel within the group receives `spmd_idx=i` and `spmd_size=N`.

**Compiler expansion**: The compiler converts a single `call_spmd_group` into `N` `call_group` invocations:

```
// Before (SPMD group call):
call_spmd_group @paged_attention_group_0(args..., spmd_size=4)

// After expansion:
call_group @paged_attention_group_0(args..., spmd_idx=0, spmd_size=4)   // cluster 0
call_group @paged_attention_group_0(args..., spmd_idx=1, spmd_size=4)   // cluster 1
call_group @paged_attention_group_0(args..., spmd_idx=2, spmd_size=4)   // cluster 2
call_group @paged_attention_group_0(args..., spmd_idx=3, spmd_size=4)   // cluster 3
```

Each `call_group` further expands to AIC + AIV kernel launches (as described in Step 10):

```
// Full expansion of one call_group instance (cluster i):
@paged_attention_incore_0_aic(args..., spmd_idx=i, spmd_size=4)
@paged_attention_incore_0_aiv(args..., spmd_idx=i, spmd_size=4, AIV_IDX=0)
@paged_attention_incore_0_aiv(args..., spmd_idx=i, spmd_size=4, AIV_IDX=1)
```

**DSL syntax** (pypto Python frontend):

```python
def orchestration(query, key_cache, value_cache, out, ...):
    # Launch the function group on 4 clusters in parallel
    pl.call_spmd_group(paged_attention_group, 
                       args=(query, key_cache, value_cache, out),
                       spmd_size=4)
```

### IR Representation

**Important clarification**: The IR does **not** introduce a special type or node to distinguish SPMD functions/groups from regular functions/groups. An SPMD function is simply a regular `Function` (or `InCoreFunctionGroup`) whose **first two arguments** are `spmd_idx` and `spmd_size`:

```
// A regular InCore function:
func @my_kernel(query, key_cache, out) { ... }

// An SPMD InCore function — same IR type, distinguished only by argument convention:
func @my_spmd_kernel(spmd_idx: uint32, spmd_size: uint32, query, key_cache, out) { ... }
```

The same convention applies to `InCoreFunctionGroup` — an SPMD group is a regular group where both the AIC and AIV kernels have `spmd_idx` and `spmd_size` as their first two arguments.

**Argument convention**: `spmd_idx` and `spmd_size` must be the **first two arguments** in the function's argument list (in that order). This is a **calling convention**, not a type-level distinction. The implicit assumption is that all SPMD functions comply with the same **runtime ABI** for passing arguments from the orchestration function into the InCore SPMD functions — the orchestration layer populates `spmd_idx` and `spmd_size` at the fixed first two argument slots before dispatching the kernel.

```
// SPMD function argument layout (ABI convention):
arg[0] = spmd_idx    // uint32_t, filled by orchestration/scheduler
arg[1] = spmd_size   // uint32_t, filled by orchestration/scheduler
arg[2] = ...         // user-defined arguments (query, key_cache, etc.)
arg[3] = ...
...
```

For `InCoreFunctionGroup` under SPMD, both the AIC and AIV kernels within the group follow the same convention:

```
// SPMD function group — both kernels have spmd_idx/spmd_size as first two args:
InCoreFunctionGroup @my_group:
    func @my_group_aic(spmd_idx, spmd_size, query, key_cache, out, ...) { ... }
    func @my_group_aiv(spmd_idx, spmd_size, query, key_cache, out, ..., AIV_IDX) { ... }
```

The `call_spmd_function` and `call_spmd_group` IR instructions are new call nodes that reference regular functions/groups and additionally carry `spmd_size` as metadata:

```
// Call instruction hierarchy:
CallInstr                // call @function(args...)
CallGroupInstr           // call_group @group(args...)
CallSPMDInstr            // call_spmd_function @function(args..., spmd_size=N)
                         //   target is a regular Function with spmd_idx/spmd_size args
CallSPMDGroupInstr       // call_spmd_group @group(args..., spmd_size=N)
                         //   target is a regular InCoreFunctionGroup with spmd_idx/spmd_size args
```

| IR Node | Target | SPMD | Expansion |
|---|---|---|---|
| `CallInstr` | `Function` | No | Single kernel launch |
| `CallGroupInstr` | `InCoreFunctionGroup` | No | 1 AIC + 2 AIV launches (1 cluster) |
| `CallSPMDInstr` | `Function` (with `spmd_idx`, `spmd_size` as first 2 args) | Yes | N × single kernel launches |
| `CallSPMDGroupInstr` | `InCoreFunctionGroup` (with `spmd_idx`, `spmd_size` as first 2 args in each kernel) | Yes | N × (1 AIC + 2 AIV) launches |

> **Design rationale**: By not introducing a separate IR type for SPMD functions, we keep the IR simple and avoid bifurcating function-level passes. Any pass that processes `Function` nodes works on SPMD functions without modification — the `spmd_idx` and `spmd_size` arguments are just regular arguments from the IR's perspective. Only the `ExpandSPMDCalls` pass and the orchestration code emitter need to be aware of the SPMD calling convention.

### Workload Partitioning with `spmd_idx`

Inside the kernel, `spmd_idx` and `spmd_size` are used to partition the data. The standard pattern:

```python
@pl.incore
def paged_attention_aic(..., spmd_idx, spmd_size):
    # Total batch range: [0, BATCH_SIZE)
    # This instance processes: [spmd_idx * chunk, (spmd_idx+1) * chunk)
    chunk = BATCH_SIZE // spmd_size
    my_batch_start = spmd_idx * chunk

    for b_idx in pl.parallel(my_batch_start, my_batch_start + chunk, 1):
        # ... process batch element b_idx ...
```

For `InCoreFunctionGroup` under SPMD, both the AIC and AIV kernels within the same group instance receive the **same** `spmd_idx`, ensuring they process the same data partition and their `tpush`/`tpop` communication remains consistent.

### Interaction with `InCoreFunctionGroup`

The combination of SPMD and function groups creates a two-level parallelism:

```
call_spmd_group @group(spmd_size=4):

    Cluster 0 (spmd_idx=0):          Cluster 1 (spmd_idx=1):
    ┌──────────────────────┐          ┌──────────────────────┐
    │ AIC kernel (idx=0)   │          │ AIC kernel (idx=1)   │
    │ AIV kernel (idx=0,0) │          │ AIV kernel (idx=1,0) │
    │ AIV kernel (idx=0,1) │          │ AIV kernel (idx=1,1) │
    └──────────────────────┘          └──────────────────────┘

    Cluster 2 (spmd_idx=2):          Cluster 3 (spmd_idx=3):
    ┌──────────────────────┐          ┌──────────────────────┐
    │ AIC kernel (idx=2)   │          │ AIC kernel (idx=3)   │
    │ AIV kernel (idx=2,0) │          │ AIV kernel (idx=3,0) │
    │ AIV kernel (idx=2,1) │          │ AIV kernel (idx=3,1) │
    └──────────────────────┘          └──────────────────────┘

Level 1: SPMD across clusters (spmd_idx = 0..3)
Level 2: AIC/AIV co-scheduling within each cluster (tpush/tpop, AIV_IDX = 0/1)
```

- **Inter-cluster**: data is partitioned by `spmd_idx`. No cross-cluster communication (each cluster works independently on its partition).
- **Intra-cluster**: AIC and AIV kernels communicate via `tpush`/`tpop` ring buffers using hardware flags with local cluster affinity.

### Backward Compatibility

- `call_spmd_function` and `call_spmd_group` are **new** IR instructions. Existing `call` and `call_group` instructions are unchanged.
- Existing passes that analyze call instructions see SPMD calls as a new subtype. Passes that only handle `CallInstr` are unaffected — the SPMD variants are only introduced when the programmer explicitly uses `pl.call_spmd` / `pl.call_spmd_group`.
- The expansion from SPMD calls to individual calls happens in a dedicated **`ExpandSPMDCalls`** pass, which runs before code emission. After expansion, the IR contains only `call` and `call_group` instructions, which all downstream passes already understand.

### Call Instruction Summary

| DSL API | IR Instruction | Target | Expansion |
|---|---|---|---|
| `pl.call(func, args)` | `call @func(args)` | `Function` | 1 kernel launch |
| `pl.call_group(group, args)` | `call_group @group(args)` | `InCoreFunctionGroup` | 1 AIC + 2 AIV |
| `pl.call_spmd(func, args, spmd_size=N)` | `call_spmd_function @func(args, spmd_size=N)` | `Function` | N kernel launches |
| `pl.call_spmd_group(group, args, spmd_size=N)` | `call_spmd_group @group(args, spmd_size=N)` | `InCoreFunctionGroup` | N × (1 AIC + 2 AIV) |

## Related Documents

- [Enhanced TPUSH/TPOP ISA Design](../../pto-isa/HL_ptoisa_newfeature20260306_TPUSH_TPOP.md) — defines the `tpush_to_aiv`, `tpush_to_aic`, `tpop_from_aic`, `tpop_from_aiv` instructions, ring buffer protocol, `CONSUMER_BUFFER_BASE`/`CONSUMER_BUFFER_SIZE` constant symbols, `pl.reserve_buffer`/`pl.import_peer_buffer` DSL grammar, and initialization APIs.
