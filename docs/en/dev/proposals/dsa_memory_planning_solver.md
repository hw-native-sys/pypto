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

`pipeline_membership=(group,stage)` therefore survives to DSA collection.
Distinct effective residues become hard separations. This is an ordinary extra
conflict, not a whole-slot placement rule. It preserves the chosen pipeline
depth while leaving all unrelated lifetime-disjoint buffers available to the
standard DSA solver.

## Research refinements

A PyPTO-specific DSA refinement must change feasibility or the objective. The
following candidates have distinct evidence requirements.

### Pipeline-overlap-aware placement

When capacity shedding maps several stages to one residue, assigning the same
address to different stages can serialize work. Research captures therefore
record sparse `pipeline_serialization` pairs. A fitting placement can minimize
the number or measured cost of such reuse edges before using peak as a
tie-break. PR #1949 grounds the mechanism; device A/B tests must calibrate the
cost.

### PTOAS-synchronization-aware placement

Other address reuse may make PTOAS add an anti-dependency, event, wait, or
barrier. PyPTO does not know final hardware-pipe assignment at export time, so a
static `cross_pipe` guess is not sufficient. This candidate requires PTOAS
instrumentation or a bounded placement-to-PTOAS feedback pass.

### Critical-path and event-budget-aware placement

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

For production, the objective is peak-only. Research documents may use:

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
