# Temporary A2/A3 TREM/TREMS PTO-ISA patch

**Provenance:** `cann/pto-isa` commit `3d36366a0` — *"trem/trems(a2a3): never
write the input tiles"* (GitCode `!2081`), first released in pto-isa
`8fb67fa7e`.

**Why it is here:** PyPTO's runtime pins pto-isa `90a37965`, which predates the
fix. The pre-fix A2/A3 high-precision TREM/TREMS implementation used its source
tiles as output scratch, so INT32 sources were left in FP32 representation and
PyPTO's `test_remainder.py` failed. Moving the managed pin to `8fb67fa7e`
independently is tracked in simpler #3070; this backport keeps PyPTO unblocked
until that lands.

**How it is applied:** `pypto.runtime.pto_isa.ensure_pto_isa_root()` detects
that the pinned `include/pto/npu/a2a3/TRem.hpp` lacks the fix marker
(`read-only-src form`), creates a detached git worktree of the pin
next to the managed checkout, applies this patch, and commits it locally. The
managed checkout is never modified, and the local commit gives the patched tree
its own kernel-binary cache identity.

**Removal:** delete this directory and the resolver hook in
`pypto/runtime/pto_isa.py` once the managed `pto_isa.pin` already contains the
fix. The resolver detects the marker and returns the pristine pin, so the
fallback disables itself; the dead code should still be removed. Set
`PYPTO_DISABLE_PTO_ISA_TREM_PATCH=1` to bypass it for debugging.
