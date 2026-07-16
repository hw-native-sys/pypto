# RFC: Pluggable DSA memory planner

## Status

Draft. Tracks issue #1980 and the fragmentation defect in #1908.

## Problem

PyPTO currently separates memory planning into `MemoryReuse`, which merges
values into allocation identities, and `AllocateMemoryAddr`, which assigns one
slot per identity. That split prevents ordinary freed-region subdivision.

For example, an early 64 KiB buffer followed by two co-live 32 KiB buffers
should need 64 KiB: the later buffers use the two halves of the expired region.
This is standard Dynamic Storage Allocation (DSA), not a PyPTO-specific variant.

## Standard DSA path

After semantic aliases are materialized, the adapter exports physical buffers
with fixed size, alignment, memory pool, and conservative half-open lifetime.
The solver chooses offsets. Buffers with overlapping lifetimes or an explicit
separation must have disjoint address ranges; all other partial spatial reuse is
legal. The objective is minimum peak, or equivalently fitting a fixed capacity.

Fixed PyPTO memory pools decompose into independent DSA problems. Capacity,
uniform alignment, reserved prefixes, collapsed aliases, and extra conflict
edges affect an instance but do not define a different packing problem.

The implementation pipeline is:

```text
InitMemRef
  -> MaterializeSemanticAliases
  -> collect unmerged physical buffers
  -> standalone DSA solver
  -> independent validation
  -> write offsets to MemRefs
```

The adapter exports one conservative physical-lifetime hull. It must not infer
holes by unioning SSA-member ranges: that previously corrupted DeepSeek-v4
loop-carried accumulators on device.

## Pipeline constraints from PR #1949

`pl.pipeline(stage=F)` creates clones that are sequential in scalar program
order but intended to overlap on asynchronous hardware units. Reusing one
address across concurrent stages introduces a false write-after-read dependency
and serializes the ping-pong pipeline. PR #1949 demonstrates this mechanism.

`pipeline_membership=(group,stage)` therefore survives to DSA collection. The
adapter first exports every distinct requested stage as a hard separation. This
is an ordinary extra conflict, not a whole-slot placement rule. It preserves
the requested pipeline depth while leaving all unrelated lifetime-disjoint
buffers available to standard DSA.

The solve policy is deliberately two-stage:

1. solve standard DSA with capacity and all pipeline-stage separations hard;
2. only when that search finds no fitting placement, remove the
   `pipeline_stage` reason, keep every other hard reason, add sparse
   `pipeline_serialization` reuse costs, and solve again.

The fallback emits `PH-DSA-001`. It means compilation succeeded by allowing
some intended pipeline copies to share physical ranges, so the generated
program may serialize. Because the current solver is heuristic, a zero-cost
fallback solution is revalidated against the strict problem and accepted
without a warning.

For pipeline-intent pairs \(P\), the strict problem adds:

```text
(i,j) in P  =>  address_range(i) does not intersect address_range(j)
```

The fallback removes only those extra edges and minimizes:

```text
lex(capacity_overflow, sum((i,j) in P) w_ij * reuse(i,j),
    total_peak, max_peak)
```

`reuse(i,j)` is one only when lifetime-disjoint buffers overlap in physical
address. Lifetime conflicts and all non-pipeline separations remain hard.
The fallback document retains the requested stage/residue mapping as
provenance; achieved depth is a placement measurement, not the
`effective_depth` field.

## Research refinements

A PyPTO-specific DSA refinement must change feasibility or the objective. The
following candidates have distinct evidence requirements.

### 1. Pipeline-overlap-aware placement

The implemented fallback minimizes sparse cross-stage overlap costs
lexicographically after capacity. Version 1 uses one unit per reused
stage-member pair. PR #1949 grounds the mechanism, but pair counting may
overweight groups with more members. Device A/B tests must compare it with a
group-level lost-depth objective before either becomes a production cost model.

### 2. PTOAS-synchronization-aware placement

Other address reuse may make PTOAS add an anti-dependency, event, wait, or
barrier. PyPTO does not know final hardware-pipe assignment at export time, so a
static `cross_pipe` guess is not sufficient. Candidate pair classes include
MTE-to-vector/cube reuse and reuse that cancels an earlier load motion. This
refinement requires PTOAS instrumentation or a bounded placement-to-PTOAS
feedback pass before weights are trusted.

### 3. Critical-path and event-budget-aware placement

Synchronization is not generally an additive pair cost. A reuse edge already
implied by dependencies can be free; several edges can form a new serial chain.
A stronger evaluator measures critical-path growth in the augmented dependency
graph. Event-identifier exhaustion is a discrete resource limit and may need a
hard bound rather than a weighted cost.

Bank costs, multi-interval liveness, flexible pool assignment, and piecewise
sizes remain hypotheses. They must not enter the required profile without an
export proof and controlled measurements.

## Interface

The standalone problem contains buffers, pools, colocations, separations,
reservations, optional fixed offsets, and a lexicographic objective. Solvers
advertise capabilities; unsupported constraints or objective terms return
`kUnsupported` and are never silently dropped. An independently named core
relaxation may remove features only for lower-bound benchmarking.

The strict solve minimizes peak under capacity. The explicit fallback uses:

```text
(capacity overflow, reuse/synchronization cost, total peak, max peak)
```

Raw components are always reported; bytes are not converted to cycles using an
arbitrary weight.

## Validation plan

- host regression for #1908: the 64 + 32 + 32 KiB shape has 64 KiB peak;
- independent checks for lifetime conflicts, separations, capacity, alignment,
  reservations, aliases, and writeback;
- device numerics for PyPTO and PyPTO-Lib, including DeepSeek and Qwen;
- pipeline tests that preserve required stage separation;
- controlled A/B placements with identical schedule and tiling, recording PTO,
  events/waits/barriers, retained depth, latency, and utilization; and
- held-out kernels when fitting any synchronization model.

The external solver dependency is temporary. Once a heuristic is selected and
validated, it can be ported into PyPTO and the dependency removed.

## References

Issues/PRs: #1908, #1934, #1949, #1980; PTOAS #913. Baselines: MiniMalloc,
TelaMalloc, TVM USMP, and OpenXLA heap simulation.
