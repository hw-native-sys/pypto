# RFC: Pluggable DSA Memory-Planning Solver + PyPTO Adapter

## Status

**Draft — request for comments.** Author: Toni Boehnlein. 2026-07-08. Scopes the
*design* and *interface*; module layout and pass-order wiring are a follow-up.

## Summary

PyPTO plans local memory in two passes — `MemoryReuse` (coloring → buffers share a
`base_` Ptr) and `AllocateMemoryAddr` (bump → one max-sized slot per base group, no
region reclaim). No principled Dynamic Storage Allocation (DSA) formulation splits
*which buffers share* from *where each goes* this way, and the split is why a freed
region cannot be subdivided (issue #1908).

**Proposal:** a standalone, reusable **DSA solver** behind a field-standard interface
(buffers with lifetimes + sizes → offsets) plus a thin **PyPTO adapter**. The adapter
carries an **optional, capability-negotiated cost overlay** for NPU structure the
standard formulation drops — chiefly that *reuse isn't free*: overlapping two
lifetime-disjoint buffers manufactures a cross-pipe synchronization a pure peak
objective doesn't cost (today handled only by one hard-coded rule, #1949).
Off-the-shelf solvers consume the core; PyPTO's own (local-search) solver consumes
the overlay.

The hook is the principled solver. **As a byproduct it dissolves #1908** —
buffer-granularity packing subdivides freed regions natively. The fix ships on the
**v1** path and does **not** depend on the overlay/local-search research (**v2**).

## Motivation

`MemoryReuse`'s capacity-gated packing (#1949, now in main) separates pipelined
double-buffers to avoid a serializing false-WAR (#1475) — one hard-coded sync-aware
reuse decision. Two gaps remain: **(1)** that
awareness is one special case, not a general cost the placer optimizes → E2
generalizes it; **(2)** coloring is split from offset assignment, so a producer's
freed 64 KB slot can't be subdivided into a consumer's two lifetime-disjoint 32 KB
buffers → L0A overflow (#1908) → buffer-granularity planning closes it.

The standard DSA problem (min peak, no concurrent overlap) is a *projection* that
drops this overlay — capturing it without contaminating the portable core is the
design's real content.

**Strategic context.** ptoas has its own planner, and PTOAS #913 rewrites it toward a
func-level first-fit-by-lifetime planner (same algorithm class). The org keeps both
(pypto L3 / ptoas L2, `memory_planner` switch #1934) and consolidates later —
**coordinate with the #913 owners**; a pypto-owned L3 planner's niche shifts if the org
picks ptoas. Pursue anyway: (a) the reusable/benchmarkable solver is independent of
which planner ships; (b) the overlay models L3 reuse→sync info ptoas can't see; (c)
the fragmentation needs fixing on L3 now.

## Background

Core DSA — buffers each `(size, lifetime, align)` → offsets, min peak (or fit a cap),
no overlap for lifetime-overlapping buffers — recurs across systems; they differ in the
solver *and* which extensions they model (so the core is common, not identical).

| System | Problem object | Solver |
| ------ | -------------- | ------ |
| MiniMalloc (ASPLOS'23) | CSV `id,lower,upper,size` → `offset` | exact DFS over canonical solutions + pruning |
| TelaMalloc (ASPLOS'22) | same | heuristic search + ILP/CP-SAT |
| XLA | `BufferInterval{size,start,end,colocations}` → `Chunk` | `GlobalDecreasingSizeBestFitHeap` (heuristic, not optimal) + `NoFragmentationStatsHeap` (loose lower bound) |
| TVM USMP | `BufferInfo{size,align,conflicts[],pool_candidates[]}` → `PoolAllocation` | `greedy_by_size` / `greedy_by_conflicts` / `hill_climb` (local search) |
| ptoas (incl. #913) | free-region outline / `bufferLifeVec` per (core,pipe) | first-fit + region split/coalesce |

MiniMalloc's public CSV is single-pool/single-interval (no colocations/pins/multi-pool);
XLA's best-fit *heap* is a heuristic. **Benchmark:** MiniMalloc `challenging/` — A–K (11 instances, 154–454
buffers, 1 MiB), curated so greedy/best-fit *fail* → it exercises only the core and is
adversarial to best-fit.

## Goals / Non-goals

**Goals:** an IR-free `DsaSolver` library (reads the MiniMalloc CSV subset); a thin
adapter (IR → `DsaProblem`, `DsaResult` → MemRef addresses); an optional overlay; fix
the fragmentation as a byproduct (v1); evaluate vs MiniMalloc-optimal + the XLA lower bound.
**Non-goals:** not replacing ptoas's planner (L2); not solving sync/event-id allocation
(only *estimating* its cost); no new IR node.

## Design

### Core vs. overlay; constraint vs. cost

```text
core DSA (portable)                overlay (pypto-specific, optional)
buffers {intervals,size,align,     E1 multi-interval liveness  ← CONSTRAINT (changes feasibility)
pool}, colocations, separations,   E2 reuse → sync cost        ← COST (re-ranks feasible set)
pinned, pool_caps → offsets        E3 bank-conflict cost       ← COST
```

The overlay is **not homogeneous**: E2/E3 (costs) re-rank the feasible set, but E1
changes *what is feasible* — a hull-only solver can report infeasible on an instance
that *fits*, so "infeasible ⇒ genuine OOM" is only defined once E1 is supported.

### Interface

```cpp
struct Buffer { BufferId id; std::vector<Interval> intervals; uint64_t size, align; PoolId pool; };
                                     // intervals: E1 union; core solver uses the hull

struct ReusePenalty { BufferId a, b; uint64_t penalty; };   // reason: cross_pipe|cross_core|event_budget
struct BankGeometry  { PoolId pool; uint64_t bank_size; uint32_t num_banks; };
struct CostModel {                                          // sparse DATA, not opaque callbacks
  std::vector<ReusePenalty> reuse_penalties;                // only cross-pipe lifetime-disjoint pairs
  std::vector<BankGeometry> banks;
  struct { double sync, bank, peak_slack; } weights;
};

struct DsaProblem {
  std::vector<Buffer> buffers;
  std::vector<std::pair<BufferId,BufferId>> colocations;    // hard SAME offset (must-alias)
  std::vector<std::pair<BufferId,BufferId>> separations;    // hard KEEP-APART even if lifetime-disjoint
  std::vector<PinnedBuffer> pinned;                         // fixed offset + (interval | whole-program)
  std::map<PoolId,uint64_t> reserved_base, pool_caps;
  std::optional<CostModel> cost_model;                      // nullopt = core-only
};

enum class SolveStatus { kFeasible, kInfeasibleProven, kBestEffortNoFit, kTimeout, kUnsupported };
struct ObjectiveValue { uint64_t peak, sync, bank; std::map<PoolId,uint64_t> peak_by_pool;
                        std::optional<uint64_t> lower_bound; bool optimality_proven; };
struct DsaResult { SolveStatus status; std::optional<DsaSolution> solution;
                   ObjectiveValue objective; std::vector<std::string> diagnostics; };

struct SolverCapabilities { bool multi_interval, cost_model, colocations, separations, pinned, multi_pool; };
class DsaSolver { public:
  virtual SolverCapabilities capabilities() const = 0;
  virtual DsaResult solve(const DsaProblem&) = 0;           // never "just returns a solution"
};
```

### Objective — mode-split

- **Core / benchmark** (`cost_model == nullopt`): minimize peak subject to no-overlap.
- **Overlay** (`cost_model` set): peak is a **hard per-pool constraint** (`peak_by_pool ≤
  cap`); among fitting packings minimize `w_sync·Σsync + w_bank·Σbank`. Peak enters
  *only* as feasibility — below-cap peak on transient L0 scratch is dead space, so
  shaving it must never cost a barrier. `weights.peak_slack` is an optional weak
  regularizer for backends that value footprint (default 0).

### Overlay extensions

- **E1 (constraint) · multi-interval liveness.** Today `var_liveness` is a single
  `[def,last_use]` hull → v1 uses hulls (parity); multi-interval is a later refinement
  (ptoas has `bufferLifeVec`).
- **E2 (cost) · reuse → sync.** Same-pipe reuse ≈ free; cross-pipe costs an event-id
  (palette 8 intra / 16 cross-core) or a full barrier. `reuse_penalties` estimates what
  ptoas's `GraphSyncSolver` charges. **Caveat:** the hardware pipe isn't in PyPTO IR at
  pass 30/31 (only *core* AIC/AIV; the pipe is assigned in ptoas) → the adapter
  reconstructs it from op kind + memory space.
- **E3 (cost) · bank conflict** on `offset mod bank_size`, coupled to `align` (if
  commensurate, alignment fixes the residue — weights aren't independent).

Pipeline depth is **not** an overlay item — the adapter pre-expands a depth-N buffer
into N staggered clones + `separations`, reducing to the core.

### Reuse edges (formal)

Two buffers induce a reuse edge iff: same pool; address intervals intersect; live sets
disjoint; a hazard crosses the ordered boundary. Colocated pairs are excluded (a
must-alias overlaps deliberately). Sub-region packing: each later tenant induces its own
edge with the prior tenant. The solver maintains a **per-move adjacency delta**
(sweep/interval index) — `cost(placement)` is not recomputed from scratch.

### Pluggability & degradation (honest)

MiniMalloc / XLA-heap / TVM-USMP are **offline oracles** over CSV or **reimplementations**
behind our interface — not in-process plugins. Core solvers don't honor
`colocations`/`separations`/`pinned`/multi-pool natively → the adapter preprocesses
(merge colocated → super-buffer; carve pins/separations as keep-outs). `SolverCapabilities`
advertises each. A core-only path merges `intervals` to hulls, drops the overlay, and may
run a **bounded** iterative "solve → add separations → re-solve" (governed — adding
separations shrinks the feasible set and can loop back into the fragmentation). `solve()` returns a
`DsaResult`: `kInfeasibleProven` = real OOM; `kBestEffortNoFit` escalates, it does not
error.

### Adapter + pass order

| Field | Source |
| ----- | ------ |
| `Buffer{size,align,pool}` | each MemRef; `pool` = memory space |
| `intervals` (E1) | `MemoryReuse` `var_liveness` — single hull today |
| `colocations` | `MaterializeSemanticAliases` (pass 29) must-aliases |
| `separations` | 910B `load`+`tpop_from_aic` hazard, `forbid_output_alias`, pipeline clones |
| `pinned` / `reserved_base` | `reserve_buffer` / per-space `reserved_start` |
| `reuse_penalties`, `banks` | reconstructed pipe class + `PassContext::GetBackendHandler()` |

Write-back: offset → MemRef address; colocations → shared base. The WAR is realized by
**ptoas's address-interval sync** at L3 (confirmed: PyPTO's tile path emits no
`set_flag`/`wait_flag`) → disjoint sub-region tenants aren't serialized (a **v1
correctness gate**). The solver replaces `AllocateMemoryAddr`'s offset walk and subsumes
`MemoryReuse`'s opportunistic half; pass 29 must-aliases + the hazard guard stay inputs.

**v1 must include both** the buffer-granularity fragmentation fix **and** pipeline-clone
separation (the #1949 clones become `separations`, the capacity shed becomes separation
relaxation on overflow). Otherwise peak-only v1 collapses the clones and **serializes
pipelined kernels**, a regression invisible to a "peak ≤ bump" gate.

## Affected Files (coarse)

- `src/ir/transforms/dsa/` — **New** — IR-free library (types, best-fit-decreasing,
  exact DFS, local search, CSV subset I/O, independent validator).
- `src/ir/transforms/allocate_memory_addr_pass.cpp` — **Modify** — replace the bump with adapter + solver.
- `src/ir/transforms/memory_reuse_pass.cpp` — **Modify** — reduce to liveness + must-alias + separations + hazard guard.
- `include/pypto/ir/transforms/` (**New** adapter header); `tests/ut/ir/transforms/` (**New/Modify**).

## Testing Plan

- **Standalone:** A–K — exact DFS owns feasibility;
  report peak vs MiniMalloc-optimal (ground truth), and the loose `NoFragmentationStats`
  bound only where no optimum is known (don't report the NoFrag gap as achievable).
- **Independent validator** on every output (caps, align, pins, colocations, separations,
  no-overlap, objective recompute); **differential** small-instance tests vs the exact
  solver; **proven-infeasible** and **timeout** tests.
- **#1908 regression:** the A-stationary chained matmul now fits; retire the
  output-stationary workaround (`test_chained_matmul_uses_mat_scratch`).
- **No regression:** per-kernel peak ≤ bump **and** a pipeline-clones-stay-distinct assertion.
- **Overlay:** cheap (same-pipe) vs expensive (cross-pipe) collapse both fit → cost-aware
  solver picks cheap. **E1:** instance that fits only with multi-interval. Determinism +
  CSV round-trip.
- **Before v2 defaults:** a **model-vs-measured** study — `Σreuse_penalty` vs actual
  ptoas event-ids/barriers (a one-round ptoas feedback may beat a static heuristic).

## Security

Compile-time resource exhaustion only: exact solvers are offline/test-only; in-pass
solvers are time-boxed → `kTimeout` → deterministic fallback (best-fit / bump). No
secrets, no runtime I/O.

## Alternatives

- **A · group-granularity first-fit** (the issue's literal proposal): insufficient — the
  pre-grouped producer slot stays co-live with the double-buffer → 96 KB. The minimal fix
  is **buffer-granularity** first-fit-by-lifetime.
- **B · sub-region reuse in `MemoryReuse`:** a non-portable, shape-specific patch.
- **Extend the solver contract with sync:** breaks core-solver pluggability → optional
  overlay instead.
- **Two RFCs (fix vs. research):** rejected — the **v1-independent-of-v2** rollout gives
  the decoupling without splitting the document.

## Rollout

Level3-first; **v1 is independent of and can land ahead of v2.**
**v0** offline library + harness. **v1** the fragmentation fix + pipeline separation (flagged,
core only, gate = peak ≤ bump AND clones distinct; falls back to the bump). **v2** overlay
local search (gated on the model-vs-measured study and #913 coordination). Level2 untouched.

## Open Questions

- **Sync-cost fidelity (v2 crux):** is a pipe-pair + budget heuristic enough for
  `reuse_penalty`, or is a one-round ptoas feedback loop needed (re-couples the tools)?
- **Hard vs. soft** for scarce resources (event-ids, peak) — one general answer, not
  per-resource?
- **Pools genuinely fixed?** If a tile may live in L0A *or* Mat/L1, fixing `pool` at
  collection is the biggest modeling loss. If fixed (expected), the problem **decomposes
  into k independent per-pool solves** (E2/separations intra-pool) — state and exploit it.
- **Conditional non-conflict:** phi-family / branch-exclusive vars share despite
  overlapping hulls today; a pure interval-overlap solver drops this (raises peak on
  branch-heavy kernels). Encode as `colocations` + a `may_overlap` exemption list?

## References

Issues/PRs: #1908, #1949 (capacity-gated, merged), #1934 (planner switch); ptoas #913.
MiniMalloc (ASPLOS'23, `github.com/google/minimalloc`); TelaMalloc (ASPLOS'22); TVM USMP
RFC 0009; OpenXLA `heap_simulator`.
