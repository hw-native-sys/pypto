# ISA macro synchronization resources

ISA macros can occupy numbered event/flag IDs internally. The backend provides
a sparse table of these signal claims, complementing the available pools and
legal IDs in the [hardware resource catalog](01-sync_resources.md). Only macros
with signal occupancy have entries; a macro that only executes a barrier needs
no entry.

This component provides declarations and queries. It does not bind actual IR
calls, allocate IDs, or insert synchronization. The existing automatic
synchronization path continues through PTOAS.

## Query a declaration

`Backend.find_isa_macro_resources(name)` looks up a qualified ISA macro name on
that backend. `Backend.get_isa_macro_names()` returns the sorted registered names.
Namespaces matter: `pto::TGATHER` and `pto::comm::TGATHER` are different macros.
Every returned declaration contains `name`, `target`, and nonempty `claims`.

```python
from pypto.pypto_core import backend, ir, sync

target = backend.get_backend_instance(backend.BackendType.Ascend950)
transfer = target.find_isa_macro_resources("pto::comm::TPUT")
assert transfer is not None
assert {(claim.pool.src, claim.pool.dst) for claim in transfer.claims} == {
    (ir.PipeType.MTE2, ir.PipeType.MTE3),
    (ir.PipeType.MTE3, ir.PipeType.MTE2),
}
assert all(claim.source == sync.IsaIdSource.Fixed for claim in transfer.claims)
assert all(claim.ids == [0, 1] for claim in transfer.claims)
```

The corresponding C++ query returns an owned optional value:

```cpp
const auto* target = pypto::backend::GetBackendInstance(pypto::backend::BackendType::Ascend950);
const auto transfer = target->FindIsaMacroResources("pto::comm::TPUT");
const auto names = target->GetIsaMacroNames();
```

An absent name returns `None` / `std::nullopt`, meaning no signal claim is
registered for it. This does not certify an unknown external implementation as
signal-free. Query results are independent snapshots; Python exposes read-only
objects and copies nested keys and collections. Queries use the explicit backend
without changing global backend selection.

## Claim fields

| Field | Meaning |
| ----- | ------- |
| `pool` | Resource family, core scope, directed pipe pair, and resource instance |
| `source` | `Fixed`, `Parameter`, or `CompilerAllocated` |
| `ids` | Fixed IDs, or candidate IDs for a compiler token |
| `parameter`, `offsets` | Parameter role and offsets added to its bound base ID |
| `lifetime` | `Invocation` or `Owner` |
| `owner` | Shared owner role whose actual identity the consumer must bind |
| `modes` | Applicable `Manual`, `Auto`, or `FlagTest` build modes |

`Fixed` records IDs already chosen by the ISA implementation. For example,
TPUT/TGET conservatively include IDs 0 and 1 in both MTE2/MTE3 directions because
one explicit staging tile can still use internal ping-pong lanes.

`Parameter` records a value such as `pipe.FlagID` or `session.syncId`. The bound
base plus every offset must fit the pool's legal ID domain. `Owner` keeps a shared
binding across calls: pipe ready/free flags must remain associated with that pipe
from initialization through its final drain. Returning from one push/pop does not
release them. `Invocation` covers the internal signal use of one call.

`CompilerAllocated` records that an ISA helper requests a token from its compiler.
Its candidate list does not mean all those IDs are occupied simultaneously, and
the declaration does not choose or control the returned token. Integration must
coordinate allocation with the implementation that emits that token. Candidates
follow hardware legality and can include IDs excluded by PyPTO's allocation policy.

`modes` separate ordinary manual code, `__PTO_AUTO__`, and `PTO_FLAG_TEST`;
`__PTO_AUTO__` takes precedence when both macros are defined. For example,
`PtoSetWaitFlag` uses fixed ID 0 in manual mode, requests a compiler token in
flag-test mode, and emits no flag in automatic mode.

## Backend integration boundary

Native rules in `src/backend/common/isa_macro_resources*.cpp` collect claims
across implementation variants. Their C++ comments explain hardware conditions
and shared-helper behavior. A later backend consumer must determine the applicable
implementation variant from the actual lowering and operands before binding its
claims, parameter values, and owner lifetimes. Mutually exclusive variants, such
as internal/external sessions or one-way/two-way pipes, cannot all be bound at
once. Within the applicable variant, retain a conservative union when a narrower
path cannot be proven. A declaration alone does not establish operand legality,
memory effects, or completion of the surrounding kernel.

The first query builds and validates the target table before publishing it.
Validation checks that declarations have signal claims, refer to valid pools,
use legal IDs and offsets, and carry valid modes and owner lifetimes.
Malformed native rules raise `pypto::InternalError`. Fixed ISA IDs remain
hardware-legal even when compiler policy excludes those IDs from new allocations.
