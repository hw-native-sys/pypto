# Backend

Per-architecture behaviour, kept out of the passes.

Passes never branch on `BackendType`. Everything architecture-specific — codegen
target, runtime API names, hazard workarounds, cross-core layout rules — is answered
by a `BackendHandler` obtained from the active `PassContext`. Adding an architecture
means adding a handler, not editing passes.

| Page | What it covers |
| ---- | -------------- |
| [BackendHandler: principled backend dispatch](00-backend_handler.md) | The virtual interface, how passes query it, and what adding a new backend requires |

## Registered operator inventory

Query a concrete backend to enumerate its actual operator registrations:

```python
from pypto import backend

target = backend.get_backend_instance(backend.BackendType.Ascend910B)
names = target.get_registered_op_names()
```

The returned list is sorted and independent of the registry. It includes the
target's exclusions and overrides, without changing the global backend selection.
Use this inventory when auditing lowering coverage; parsing registration source
can count commented examples or miss names registered through shared tables.
Registration alone does not guarantee support for every dtype, shape, or layout.
The list also includes legacy backend entries whose IR operators no longer exist;
use `ir.is_op_registered(name)` to distinguish those when auditing current IR.

The Buffer migration guard in
`tests/ut/backend/test_buffer_migration_inventory.py` compares both runtime
registries with an explicit audit of 168 live names in 22 families and eight
historical callbacks. It detects new or deleted names, duplicate classifications,
target changes, and historical callbacks that acquire an IR definition. Update
the test's audit ledger whenever the registered surface changes.

The ledger's `PLANNED` and `RESTRICTED` statuses describe declared migration
maturity. `RESTRICTED` means some forms are implemented; it does not promise
support for every dtype, layout, attribute, or target form. This guard makes
inventory omissions fail tests. Production conversion recipes and their
conversion, native compilation, and numerical tests establish implementation
coverage; the audit ledger does not drive lowering or enable the default switch.

The guard also queries `backend.get_buffer_elementwise_recipe_names()`, the
actual table used by logical conversion and native emission. Every reported
recipe must have a live backend operation and a `RESTRICTED` audit classification.
This keeps maturity declarations current without duplicating production
instruction mappings or claiming complete form coverage.

## See Also

- [Pass, PassContext, PassPipeline, and PassManager](../passes/00-pass_manager.md) — where the handler comes from.
- [PTO ISA reference](../../reference/index.md) — the hardware differences the handlers abstract over.
