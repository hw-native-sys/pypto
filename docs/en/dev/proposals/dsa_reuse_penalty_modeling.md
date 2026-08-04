# DSA Reuse-Penalty Modeling

## Status

PyPTO's reuse-penalty recognizer is experimental and disabled by default. This
document separates three concerns that must not be conflated:

1. the stable DSA-RP optimization problem;
2. the recognizer and promotion policy currently implemented in PyPTO; and
3. the evidence used to decide which recognized candidates deserve a positive
   weight.

The current evidence supports keeping the optimization problem simple and
non-negative. It does not support enabling the current promotion policy in
production.

## Stable optimization problem

Each physical memory space is an independent standard DSA arena. Capacity and
correctness constraints are hard. A sparse soft edge
`e = (buffer_i, buffer_j, weight_e)` is active when the placed byte ranges of
the two lifetime-compatible buffers overlap:

```text
reuse_cost(p) =
    sum(weight_e for active overlapping edges e)
```

The planner finds a valid placement within capacity, minimizes
`reuse_cost`, and may minimize peak as a final tie-break. Weights are
non-negative. An absent edge or weight zero means that the compiler has no
evidence of performance damage; it does not mean that overlap is guaranteed
free.

The model deliberately contains only buffer pairs and weights. Recognition,
PTOAS synchronization behavior, and weight estimation are producer-side
questions. Pipeline-intent hard constraints and their explicit soft fallback
are orthogonal to the access-hazard recognizer discussed here.

## Implemented recognizer

`DsaReusePenaltyRecognizer.QUADRATIC` is the only enabled research mode. It is
off by default and favors coverage over compile time.

### Candidate generation

The recognizer:

1. normalizes semantic aliases into physical allocation identities;
2. collects execution-time reads and writes, including tuple results,
   mutating operations, base allocations, and known byte ranges;
3. maps each access to an abstract source/destination route and execution
   resource;
4. constructs terminal-access and initial-write frontiers per allocation;
5. scans every lifetime-compatible buffer pair in one address space; and
6. records distance-zero and distance-one WAR/WAW handoffs with route, range,
   loop, control-flow, and ordering provenance.

The raw records are exported in
`metadata.recognized_reuse_candidate_records_v4`. An SSA-reachable record has a
deterministic region/statement `dag_path`; an unordered record uses
`dag_path=none`. The command

```bash
python -m pypto.tools.dsa_reuse_candidates PROBLEM.dsa.json
```

parses these pre-policy records.

### Current v5 construction and weight policy

The current `cross_resource_completion_pair_v5` policy constructs one pair edge when at
least one record for the pair is:

- cross-resource;
- full-allocation and completely observed;
- not dependent on a conservative initial anchor;
- not a same-operation alias-contract question.

SSA-ordered and loop-carried records remain eligible. Device experiments
refuted SSA reachability as a completion proof and identified a costly
distance-one handoff. Same-resource, partial-view, and uncertain records remain
report-only. Multiple qualifying records for one buffer pair produce one edge.
`unit_v1` then assigns cost `1` to every constructed edge.

This produces an additive, non-negative `cross_pipe` model of potential
synchronization obligations. It does not yet decide whether an obligation
extends the consumer's effective completion frontier or is exposed on the
critical path; metadata records that limitation as
`reuse_penalty_completion_exposure_model=unmodeled_v1`. The recognizer remains
experimental and disabled by default.

The implementation uses same-resource issue order and SSA reachability while
constructing per-allocation access frontiers. Same-resource ordering is an
abstract completion-chain assumption. SSA reachability is exported only as
provenance for cross-resource candidates, not as a suppression rule.

## Evidence and rejected rules

Controlled placements hold the DSA problem and generated operations fixed and
change only selected physical overlaps. The accumulated results establish:

- Most legal reuse is synchronization-neutral.
- Synchronization-group count does not predict latency.
- Route class alone is insufficient: the same route can be harmful, neutral,
  or covered by another release.
- Loop frequency amplifies an exposed handoff, but cannot make a covered
  handoff costly.
- Several predecessors of one consumer behave approximately like the latest
  active predecessor, not like an additive count.
- No experiment has justified a negative optimization weight. Apparent
  synchronization removal has not produced a replicated latency benefit.

The exact ordered-pair study added two important counterexamples to the
previous v4 policy:

| Pair class | Result |
| --- | --- |
| SSA-ordered `V -> MTE2` WAR | overlap added one matching PTOAS handoff |
| unordered `M -> MTE1` WAR | overlap removed a redundant handoff, with no confirmed latency effect |
| four other matched pairs | synchronization unchanged |

Therefore v5:

- retains SSA `dag_path` as provenance rather than a suppression predicate;
- includes distance-one candidates rather than suppressing all loop-carried
  handoffs;
- still treats every constructed edge as an uncalibrated unit obligation.

The experiments also show that promoting every cross-resource obligation with
the same positive performance weight is too broad:

- A positive weight for every synchronization-changing pair is unjustified.
- The non-negative pair model itself remains viable: neutral or apparently
  beneficial pairs can simply receive no positive edge.
- Several whole-kernel RP placements are reproducibly faster, including UB and
  L1 cases, but the largest gains are not ranked by the number of inserted
  synchronization groups.
- Clean pair ablations can reproduce substantial latency changes while their
  changed synchronization summaries point in different directions. Candidate
  recognition is therefore ahead of mechanism attribution and weight
  calibration.
- A pair's effect can depend on the surrounding placement. Pair isolation must
  preserve capacity and account for every overlap relation changed by the
  construction.

## Current completion-frontier conjecture

Three forms of ordering must remain distinct:

| Layer | Question |
| --- | --- |
| Logical ordering | Can one PyPTO value-producing operation reach another? |
| Completion ordering | Has the prior asynchronous access released the reused bytes before the overwrite? |
| Exposed delay | Does extending that release point change initiation interval or kernel latency? |

For lifetime-compatible logical buffers `A != B`, let `u` be the last relevant
access to the reused subrange of `A`, and `v` the first overwrite of the same
subrange by `B`. Reuse creates a candidate physical WAR/WAW handoff `u -> v`.

The next recognizer policy should suppress the candidate only when a
**completion-carrying path** already orders `u` before `v`. Examples include an
explicit event/barrier/token or a target-guaranteed FIFO completion relation;
ordinary SSA reachability is insufficient. The v5 constructor therefore keeps
such candidates, while its unit weight remains an experimental upper-level
surrogate rather than a calibrated latency claim.

A positive edge is indicated only when `u` extends `v`'s existing
completion-release frontier. A qualitative cost conjecture is:

```text
dynamic frequency
    * max(0, completion(u) - latest unavoidable release of v)
```

This expression is a modeling guide, not a cycle estimator currently available
to PyPTO. The checked-in v5 unit model deliberately stops before this step:
it recognizes sparse pair obligations but does not label them production
performance costs. A future calibrated producer should use zero as the default
and assign a positive weight only to repeatedly demonstrated harmful
mechanisms. For several candidate predecessors of one consumer, retain the
dominant supported pair rather than summing duplicate evidence. OR groups,
hyperedges, negative weights, and global event-budget terms remain deferred.

## Remaining validation

The completion-frontier factorial has been run. It confirmed that several
active predecessors of one consumer may collapse to one release frontier, but
the isolated frontier extensions were latency-neutral because an existing drain
already quiesced the resource. It supports consumer-aware deduplication; it does
not justify a positive weight.

The next study should start from kernels with a replicated RP-versus-compact
latency difference and work backward:

1. identify candidate pairs changed by the endpoint placement;
2. construct exact-XOR single-pair and small factorial placements, preferably
   with a capacity-preserving address exchange;
3. freeze the predicted mechanism before compiling through PTOAS;
4. compare the complete synchronized instruction topology and predecessor
   identity, not only group counts;
5. validate all written outputs with real kernel inputs and scalars; and
6. measure kernel-only latency on two devices, escalating samples only when the
   initial confidence interval is informative.

The immediate targets are the confirmed UB and L1 kernels for which endpoint
speedups exist but pair-level mechanism attribution is incomplete. A mechanism
earns a positive weight only when it predicts the sign across fresh kernels and
placement backgrounds. The checked-in v5 unit model may be used to generate
algorithm-study instances, but production promotion remains unsupported until
that calibration exists.
