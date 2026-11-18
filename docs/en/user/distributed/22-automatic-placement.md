# Select L3 or L4 from an assigned rank map

An unchanged `@pl.jit.host` entry can compile for one or multiple hosts. A single
host selects L3. Multiple hosts cause the compiler to generate an L4 controller
and L3 relays, using the existing Direct MPI execution path. CHIP computation
functions remain unchanged.

## Configure assigned ranks

A rank is a global NPU rank for this invocation. List order defines global rank
order; each `RankPlacement` describes an already assigned host and physical NPU.
PyPTO does not discover hosts or allocate devices.

```python
from pypto.ir import RankConfig, RankPlacement
from pypto.runtime import RunConfig

ranks = RankConfig(
    world_size=4,
    placements=[
        RankPlacement("node-a", 0),  # global rank 0
        RankPlacement("node-a", 1),  # global rank 1
        RankPlacement("node-b", 0),  # global rank 2
        RankPlacement("node-b", 1),  # global rank 3
    ],
    python_executable="/opt/venv/bin/python",
)
compiled = lib_host_entry.compile(config=RunConfig(rank_config=ranks))
```

Use devices assigned by the scheduler; the numbers above illustrate the data
structure. Operator code uses logical ranks for `worker`. Use the same Python
executable path on all remote hosts. For a single-host placement the caller must
run on that host. Already parsed programs also accept
`ir.compile(program, rank_config=ranks)`. Do not combine `rank_config` with an
explicit distributed or cluster config.

## Compilation

1. Validate rank count, unique device ownership and host layout, then resolve the
   existing `DistributedConfig` or `ClusterConfig`.
2. Keep a local HOST entry unchanged. For multiple hosts, promote the HOST root
   into a CLUSTER controller, expand static dispatch loops, and insert reusable
   HOST relays keyed by CHIP callee and local worker.
3. Input slices retain global rank indices. Existing window materialization,
   communication-context forwarding and recursive artifact packaging handle the
   generated hierarchy.
4. The validated rank-map snapshot participates in JIT cache identity. Host,
   device or rank-order changes create separate artifacts. Existing metadata
   supports cold loading.

## Current support

The existing L4 ABI requires equal device counts per host and contiguous global
rank groups. Automatic placement requires statically resolved CHIP workers and
static loops without carried values, including nested epoch and rank loops.
Runtime-selected workers and explicit Submit dependencies require an explicit
hierarchy for now. Explicit L3/L4 entries and configurations retain their current
behavior.

## Verification

```bash
source .claude/skills/testing/load-env.sh
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    tests/ut/ir/test_rank_placement.py tests/ut/jit/test_auto_rank.py \
    -n "$PYPTO_TEST_JOBS" -q
```

These tests cover configuration, rank order, cache identity, compilation, shared
windows and cold loading. Compilation tests do not claim hardware execution
coverage.
