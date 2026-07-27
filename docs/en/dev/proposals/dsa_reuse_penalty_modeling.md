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

### Current v4 promotion and weight policy

The current `cross_resource_pair_v4` policy constructs one pair edge when at
least one record for the pair is:

- cross-resource;
- full-allocation and completely observed;
- not dependent on a conservative initial anchor;
- not a same-operation alias-contract question;
- distance zero; and
- not ordered by the recognizer's SSA dependency graph.

Same-resource, loop-carried, partial-view, uncertain, and SSA-ordered records
remain report-only. `unit_v1` then assigns cost `1` to every constructed edge.
This produces an additive, non-negative `cross_pipe` cost model.

The implementation currently uses same-resource issue order and SSA
reachability while constructing access frontiers. That is an experimental
approximation of completion ordering, not a hardware guarantee.

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

The exact ordered-pair study added two important counterexamples:

| Pair class | Result |
| --- | --- |
| SSA-ordered `V -> MTE2` WAR | overlap added one matching PTOAS handoff |
| unordered `M -> MTE1` WAR | overlap removed a redundant handoff, with no confirmed latency effect |
| four other matched pairs | synchronization unchanged |

Therefore:

- SSA `dag_path` is provenance, not a safe suppression predicate.
- Promoting every unordered cross-resource candidate is too broad.
- A positive weight for every synchronization-changing pair is unjustified.
- The non-negative pair model itself remains viable: neutral or apparently
  beneficial pairs can simply receive no positive edge.

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
ordinary SSA reachability is insufficient.

A positive edge is indicated only when `u` extends `v`'s existing
completion-release frontier. A qualitative cost conjecture is:

```text
dynamic frequency
    * max(0, completion(u) - latest unavoidable release of v)
```

This expression is a modeling guide, not a cycle estimator currently available
to PyPTO. Until it is validated, keep raw candidates, use zero as the default,
and promote only repeatedly demonstrated harmful mechanisms. For several
candidate predecessors of one consumer, retain the dominant supported pair
rather than summing duplicate evidence. OR groups, hyperedges, negative
weights, and global event-budget terms remain deferred.

## Next validation

The next fixed-placement experiment uses one consumer and a two-edge factorial:

```text
target overlap off/on
covering overlap off/on
```

It must compare:

- an uncovered target handoff that adds a release;
- the same handoff when an existing later completion release covers it; and
- a coalesced case in which overlap removes a redundant release.

Each geometry is repeated at two physical addresses. Predictions are frozen
before PTOAS:

```text
uncovered target -> synchronization addition
covered target   -> no additional synchronization
coalesced target -> synchronization removal
```

The experiment records final predecessor identity and kernel-only latency, not
only summary counts. All endpoints require exact overlap XOR, address-only
pre-InsertSync PTO differences, bit-identical outputs, real kernel inputs and
scalars, and two-device ABBA timing for structurally informative cases.

Promotion remains unsupported until the completion-frontier rule predicts
fresh cases across multiple kernels and memory spaces and separation removes a
replicated material latency cost without introducing another handoff.
