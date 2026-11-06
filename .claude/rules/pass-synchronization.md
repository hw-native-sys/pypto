# Synchronization in Passes

**PyPTO passes MUST NOT insert `bar`, `set_flag`, `wait_flag`, or equivalent
synchronization instructions, except for communication or for preserving
synchronization explicitly written by the frontend user.**

## Scope

This applies to all PyPTO IR passes and helpers invoked by them. It includes
`system.bar_v`, `system.bar_m`, `system.bar_all`, `system.sync_src`,
`system.sync_dst`, and equivalent operations under other names.

Ordinary intra-core pipeline synchronization belongs to downstream PTOAS.
Do not add barriers or set/wait flags to compensate for tiling, pipelining,
buffer reuse, instruction scheduling, or a backend synchronization defect.
Preserve correct dependencies and memory effects for downstream synchronization
analysis, and fix defects in the layer that owns them.

## Exceptions

- **Explicit frontend synchronization:** passes may preserve and forward
  synchronization instructions explicitly written by the user. Lowering or
  cloning them as part of a semantics-preserving transformation is allowed;
  this does not authorize adding independent compiler-generated synchronization.
- **Communication:** passes may insert synchronization required by a cross-core,
  cross-rank, or task communication protocol. This includes communication
  handshakes, notify/wait operations, memory-visibility fences, task dependency
  barriers, and `bar` or set/wait flags required by that protocol. Document the
  communication dependency and why the inserted synchronization is needed.

A pass handling communication does not gain permission to insert unrelated
intra-core synchronization. When reviewing an insertion, identify either its
explicit frontend origin or the communication protocol that requires it.
