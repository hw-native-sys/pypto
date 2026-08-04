# Joint Scheduling and Local-Memory Planning

## Status

Deferred research direction. The active work keeps the existing topological
order fixed and studies placement-induced synchronization and pipeline
serialization first.

## Motivation

Buffer lifetimes are derived from a legal schedule. Moving a load, compute, or
store changes definition and last-use points and can change the best placement.
For example, these schedules satisfy the same data dependencies:

```text
overlap-oriented                 memory-oriented
load A                           load A
load B                           compute A
compute A                        load B
compute B                        compute B
```

The first exposes load/compute overlap but keeps `A` and `B` live together. The
second may reuse their address range but reduce overlap. Fixed-lifetime DSA sees
only the consequence of one choice; a joint optimizer can evaluate the trade-off.

A useful formulation is:

```text
minimize latency + synchronization cost + spill cost
subject to per-pool capacity, data dependencies, aliasing, and hardware rules
```

Peak memory can instead be a hard constraint. Variables may include a legal
topological order, pipeline depth and residue mapping, synchronization, and
buffer offsets.

## Current Boundary

PyPTO currently fixes the schedule before exporting DSA:

```text
SkewCrossCorePipeline
  -> LowerPipelineLoops
  -> CanonicalizeIOOrder
  -> InitMemRef
  -> MaterializeSemanticAliases
  -> lifetime analysis and DSA export
```

`CanonicalizeIOOrder` performs a dependency-preserving priority topological sort
inside same-core pipeline loops. `ComputeLifetimes` then derives intervals from
statement positions. The standalone solver chooses reuse and offsets for that
fixed schedule. `pipeline_groups`, alias classes, and separations preserve facts
that plain intervals do not express.

PTOAS normally consumes the frontend order. Its memory planner linearizes the
resulting MLIR and derives gen/kill points from MLIR liveness. PTOAS has a
block-local `OpScheduling` pass for accepted tile-fusion groups, but not a
general scheduler for every kernel. Automatic sync/event insertion and final
pipe behavior provide additional lower-level execution facts.

## Possible Ownership Models

Joint optimization is possible in either compiler, but each boundary exposes a
different search space.

| Owner | Can vary | Strongest knowledge | Main limitation |
| ----- | -------- | ------------------- | --------------- |
| PyPTO | Tiling, pipeline construction/depth, cross-core skew, operation order, reuse, offsets | Source semantics, loops, aliases, high-level alternatives | Must estimate final pipe, event, and instruction costs |
| PTOAS | Lowered instruction order, pipe synchronization, multi-buffer slots, reuse, offsets | Concrete PTO operations, pipes, events, backend legality | Cannot recover high-level choices erased by lowering |
| Cross-layer | PyPTO schedule alternatives plus PTOAS placement/backend cost | Semantic structure and backend truth | Needs a stable protocol and possibly iterative compilation |

One ownership invariant is mandatory: a component that changes the schedule
after lifetimes were derived must also recompute or revalidate the placement.
PyPTO cannot assign overlapping addresses and then allow PTOAS to reorder those
uses freely. A PyPTO-owned path must constrain later PTOAS transforms to preserve
the proved lifetime relation; a PTOAS-owned path must let PyPTO skip local-address
assignment so PTOAS can schedule and place together.

### PyPTO-owned optimization

PyPTO is the natural owner when the search changes loop transformations,
software-pipeline depth, tiling, or AIC/AIV structure. The joint problem would
be built before `CanonicalizeIOOrder` commits to one ordering. The selected
schedule would be materialized, lifetimes recomputed, placement validated, and
explicit addresses emitted through the DSA path.

This does not require changing PTOAS's planner, but needs a calibrated model for
pipe overlap, synchronization, and latency. PTOAS remains the legality and
device-validation boundary.

### PTOAS-owned optimization

PTOAS can co-optimize scheduling and memory for a fixed lowered kernel. It would
need a general legal operation scheduler coupled to `PlanMemory`, rather than
deriving liveness after one fixed order. This fits the PTOAS-owned planner mode,
where PyPTO does not assign local addresses.

This is attractive for instruction- and event-aware decisions. PTOAS cannot
reconsider tile shapes, pipeline construction, or cross-core decomposition
unless PyPTO preserves those choices as alternatives or metadata.

### Cross-layer optimization

The most complete design is staged or iterative:

1. PyPTO exports a dependency DAG, legal schedule alternatives, aliases,
   pipeline structure, pools, and sizes instead of only fixed intervals.
2. A joint solver proposes a schedule and placement, or PyPTO enumerates a
   small Pareto set.
3. PTOAS evaluates concrete pipe/event legality and cost.
4. Feedback rejects or re-scores the candidate; the final choice is validated
   independently and measured on device.

The portable benchmark can retain fixed-schedule profiles. A separate
PyPTO/PTOAS profile can record richer instances and backend feedback, so the
standalone benchmark need not acquire a build-time PTOAS dependency.

## Recommended Research Staging

1. Keep fixed-schedule DSA as the reproducible baseline.
2. Evaluate the strict-then-soft pipeline-intent policy and reuse-cost models
   against PTOAS output and device latency.
3. Only after those fixed-schedule models are predictive, export the
   pre-schedule dependency DAG and enough metadata to replay
   `CanonicalizeIOOrder` alternatives without changing solver behavior.
4. Add bounded schedule moves to `pypto-structured-search`: ready-node swaps,
   load/store motion, pipeline-depth changes, and placement repair.
5. Recompute lifetimes after each move and independently validate schedule and
   placement.
6. Compare predicted peak, synchronization, and latency with PTOAS output and
   device traces.
7. Then decide whether the production heuristic belongs in PyPTO, PTOAS, or is
   split between them.

The likely production split is hierarchical: PyPTO chooses high-level schedule
structure, PTOAS refines low-level instruction/event scheduling, and each memory
planner consumes the schedule at its abstraction level. A monolithic joint
solver is useful for research but is not required by the final architecture.

## Open Questions

- Which operations may move across pipeline stages or control-flow boundaries?
- Should capacity be a hard constraint with latency as the objective, or should
  the benchmark expose a Pareto frontier?
- How reliably can PyPTO predict PTOAS pipe/event information?
- Can schedule moves be evaluated incrementally without rebuilding all
  lifetimes and conflicts?
- Which high-level alternatives must survive lowering for PTOAS to use them?
