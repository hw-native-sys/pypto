<!-- markdownlint-disable MD013 MD060 MD033 -->
# PTOAS Op Status Matrix

Rows are **all public and compatibility ops provided by the latest PTOAS**. The interface
baseline is Little-oil/PTOAS `main`
`d852dd2dba3e5bf7a69ce8324eb88afc336e8a33`: 189 public interfaces from the manual
plus 15 source-only compatibility/tile interfaces still present in `PTOOps.td`, for a
total of 204. Column statuses were checked against the **current PyPTO source** (last
updated 2026-08-14). When an op is added or changed, update only its corresponding row.

The matrix includes public/compatibility interfaces even when their PyPTO level is
`internal`. Separately, it excludes 32 additional `PTOOps.td` ops that exist only
for lowering/compiler plumbing, plus other dialects such as VPTO, VMI, and SIMT.

## Completion Criteria

**An op is complete only if it has same-name ST coverage.**

- `pypto frontend✅ + ST❌`: incomplete; the frontend or codegen path exists, but
  same-name hardware coverage is missing.
- `ST✅`: an active `tests/st/` test ultimately generates and executes the same-name
  `pto.*` op.
- `ST✅` indicates same-name execution evidence on hardware. Hardware verification
  still pending on other architectures remains documented in Notes.
- `ST—`: an internal primitive, compile-time helper, or superseded interface that is
  unsuitable for a standalone ST; covered by codegen or integration tests.
- A high-level test does not count as same-name coverage if it ultimately generates
  other PTO ops. For example, a collective decomposed into
  `tput/tget/tnotify/twait` does not count as ST coverage for
  `tbroadcast/tgather/tscatter/treduce`.

## Planner View Compilation Guard

The `system-tests-a5sim` CI job runs the `fillpad_inplace` planner/view regressions
with its pinned PTOAS installation, before installing the device runtime:

```bash
python -m pytest tests/ut/codegen/ --require-ptoas -m ptoas_compile -v
```

`--require-ptoas` fails test setup when PTOAS is missing, cannot run, or is older
than the supported version, so native compilation cannot silently skip for a
missing compiler. The tests cover PTOAS-owned memory planning and the PYPTO/DSA_RP
comparison paths on both backends. This guard checks MLIR and compilation to C++;
it does not replace device numerical tests or exhaustively validate the planner.

For an allocation-backed input, the padded result uses a dynamic-valid reshape
alias initialized to the full tile extent. This preserves subsequent
`set_validshape` updates, including runtime bounds and uses of the original padded
handle. This includes allocations declared at function entry for `if/else` results.
Slice and other non-mutable view inputs retain the static-valid lowering;
the existing restrictions on updating their validity still apply.

The shared `EnsureTileViewSsa` helper handles typed alias creation and explicit
runtime-validity initialization. Callers supply the initial extents in physical
tile coordinates; choosing the full extent remains part of fillpad's semantics.
An initialized mutable alias always receives a fresh handle, even if its type
matches the source: their current runtime extents can differ. Existing static
view callers keep their original behavior. Fillpad uses this helper only on the
PTOAS planner path; PYPTO and DSA_RP retain their address-backed lowering.

To extend this guard, add `@pytest.mark.ptoas_compile` to a regression test under
`tests/ut/codegen/`. The test must assert the relevant generated-code semantics
and actually invoke PTOAS compilation; the marker only selects tests. CI discovers
marked tests across this directory, so new operators or test files need no workflow
change. Keep `--require-ptoas` in CI to reject an unavailable compiler.

After installing the runtime, the same CI job also runs the existing
`fillpad_inplace` numerical tests with PTOAS memory planning on both simulators:

```bash
python -m pytest tests/st/runtime/ops/test_fillpad_inplace.py --memory-planner=ptoas --platform=a2a3sim,a5sim --forked -v
```

The six basic cases compare the full `[64, 64]` output against a reference: the first
48 rows retain the input, and the last 16 rows contain zero, positive infinity,
or negative infinity. Six more cases cover padding followed by static/runtime
validity updates and another fill through the original padded alias. Four additional
cases exercise both paths of an `if/else` result before padding and narrowing.
Two further cases repeat the same padding mode after narrowing, checking that
equal source/destination types still retain independent valid extents. This
complements the compilation guard with simulator numerical coverage; it does not
establish on-device correctness for every view
composition covered by the compilation tests.

## Cast Scratch Planning

On A2/A3, non-saturating FP32 -> INT16 and FP16 -> INT16/INT8/UINT8 casts
need scratch storage. PyPTO and DSA-RP materialize it in `InitMemRef` for PTOAS
level 3. With the PTOAS planner, level 2 owns that allocation: codegen supplies
static-valid source/destination reshape views while keeping the one-input
`pto.tcvt` form. These views alias the existing buffers and preserve the actual
valid shape, including partial tiles.
The cast rule registered with `f_plan_scratch` in `pto_ops_cast.cpp` owns scratch
selection, sizing, and dtype. InitMemRef and codegen query it through
`Backend::PlanOpScratch("tile.cast", request)`. InitMemRef materializes only a
`Required` specification; incomplete geometry returns `Unknown` and is diagnosed
at the call site, rather than being treated as no scratch.
An explicitly supplied `tmp` bypasses this synthesis rule and keeps its own type
and capacity. Rank-one sources with explicit scratch remain supported: codegen
maps their shape and static validity from `[N]` to `[1, N]`.

A2/A3 narrowing casts use disjoint source and destination storage, following
the shared allocation and codegen safety rule described in
[MemoryReuse](passes/37-memory_reuse.md). Static-valid views preserve each
operand's existing storage; they do not make the cast source and destination alias.

Only scratch-backed casts (or an explicit tmp on A2/A3) require static validity.
Native saturating casts, other native conversions, and A5 casts retain their
existing dynamic-validity behavior. Runtime validity on an A2/A3 scratch-backed
cast remains unsupported and is diagnosed before invoking PTOAS.

`tests/ut/codegen/test_ptoas_planner_views.py` checks all four conversion pairs,
both saturation modes, both backends, all three planners, and full/partial tiles;
it also invokes native PTOAS compilation when installed. CI runs
`tests/st/runtime/ops/test_cast_saturation.py` on the A2/A3 simulator to compare
exact numerical results across planners and verify that partial stores preserve
output elements outside the valid rectangle.

## Legend

- **Level**: PyPTO registration/generation level (tile / tensor / tile+tensor / comm /
  internal).
- **PTOAS API**: ✅ = the latest PTOAS `main` provides the canonical op.
- **pypto-tile / -tensor frontend**: ✅ = a corresponding public frontend exists;
  ❌ = not added; `—` = not applicable to internal or communication ops.
- **ST**: same-name ST status for non-communication ops; communication ops use the
  next column.
- **distributed ST**: same-name distributed ST status for communication ops;
  `—` for non-communication ops.
- **Notes**: records only current addition/coverage facts and direct blockers, not
  future implementation order.

| PTOAS op (pto.*) | pto-isa API | Level | PTOAS API | pypto-tile frontend | pypto-tensor frontend | ST | distributed ST | Notes |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Pointers / Views (13)** |  |  |  |  |  |  |  |  |
| pto.ptrtoint | — | internal | ✅ | — | — | — | — | compile-time/allocation helper; no standalone ST |
| pto.inttoptr | — | internal | ✅ | — | — | — | — | compile-time/allocation helper; no standalone ST |
| pto.addptr | — | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.castptr | — | internal | ✅ | — | — | — | — | compile-time/allocation helper; no standalone ST |
| pto.make_tensor_view | — | tensor | ✅ | ❌ | ✅ | ✅ | — | emitted by `tensor.view` |
| pto.get_tensor_view_dim | — | internal | ✅ | — | — | — | — | compile-time/allocation helper; no standalone ST |
| pto.get_tensor_view_stride | — | internal | ✅ | — | — | — | — | compile-time/allocation helper; no standalone ST |
| pto.partition_view | — | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.alloc_tile | — | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.alloc_multi_tile | — | internal | ✅ | — | — | — | — | compile-time/allocation helper; no standalone ST |
| pto.multi_tile_get | — | internal | ✅ | — | — | — | — | compile-time/allocation helper; no standalone ST |
| pto.subview | — | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.set_validshape | .SetValidShape | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **Buffer IDs (2)** |  |  |  |  |  |  |  |  |
| pto.get_buf | get_buf | internal | ✅ | — | — | — | — | compile-time/allocation helper; no standalone ST |
| pto.rls_buf | rls_buf | internal | ✅ | — | — | — | — | compile-time/allocation helper; no standalone ST |
| **DMA Data Movement (10)** |  |  |  |  |  |  |  |  |
| pto.tload | TLOAD | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tprefetch | TPREFETCH | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tprefetch_async | TPREFETCH_ASYNC | tile | ✅ | ❌ | ❌ | ✅ | — | Emitted by `pl.prefetch.*` (`make_context` / `async_prefetch` / `session` / `wait`); not a tile/tensor frontend. PTOAS op availability does not imply runtime SDMA workspace provision; artifact metadata enables it automatically, with ST coverage currently limited to onboard a2a3. Platforms without a provider fail during initialization rather than degrading to a no-op |
| pto.make_prefetch_async_context | pto::PrefetchAsyncContext | internal | ✅ | — | — | — | — | validated as part of async-prefetch integration |
| pto.get_prefetch_async_session | .session | internal | ✅ | — | — | — | — | validated as part of async-prefetch integration |
| pto.tstore | TSTORE | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.load | direct pointer load | tensor | ✅ | ❌ | ✅ | ✅ | — | emitted by `tensor.read`; replaces `pto.load_scalar` in PTOAS 0.64 |
| pto.store | direct pointer store | tensor | ✅ | ❌ | ✅ | ✅ | — | emitted by `tensor.write`; replaces `pto.store_scalar` in PTOAS 0.64 |
| pto.tmov | TMOV / TMOV_FP | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.ttrans | TTRANS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **Matrix Computation (12)** |  |  |  |  |  |  |  |  |
| pto.tmatmul | TMATMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmatmul.acc | TMATMUL_ACC | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmatmul.bias | TMATMUL_BIAS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tmatmul.mx | TMATMUL_MX | tile | ✅ | ✅ | ✅ | ✅ | — | A5 MXFP8 host-prequant frontend+codegen+ST; FP4 input is limited to FP4×FP8 with an explicit lhs FP4→FP8 cast; native FP4×FP4 is deferred; see [operators MX constraints](ir/05-operators.md#mx--ascend950-pto-isa-constraints) |
| pto.tmatmul.mx.acc | TMATMUL_MX (overload) | tile | ✅ | ✅ | ✅ | ✅ | — | A5 frontend+codegen+ST (`tile.matmul_mx_acc`) |
| pto.tmatmul.mx.bias | TMATMUL_MX (overload) | tile | ✅ | ✅ | ✅ | ❌ | — | NEW frontend+codegen (`tile.matmul_mx_bias`); ST pending |
| pto.tgemv | TGEMV | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 task-submit ST passed; A5 hardware validation is pending |
| pto.tgemv.acc | TGEMV_ACC | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 task-submit ST passed; A5 hardware validation is pending |
| pto.tgemv.bias | TGEMV_BIAS | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 task-submit ST passed; A5 hardware validation is pending |
| pto.tgemv.mx | TGEMV_MX | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tgemv.mx.acc | TGEMV_MX (overload) | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tgemv.mx.bias | TGEMV_MX (overload) | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| **Vector Arithmetic and Math (42)** |  |  |  |  |  |  |  |  |
| pto.tadd | TADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tsub | TSUB | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmul | TMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tdiv | TDIV | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | verified on A2/A3 hardware; A5 hardware verification pending |
| pto.tmax | TMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmin | TMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trem | TREM | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 exact-op hardware ST passed for FP32 and INT32 within the ISA domain `[-2^24, 2^24]`, including domain boundaries plus full/tail valid shapes; A5 hardware pending |
| pto.tpartadd | TPARTADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tpartmax | TPARTMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tpartmin | TPARTMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tpartargmax | TPARTARGMAX | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tpartargmin | TPARTARGMIN | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tpartmul | TPARTMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tprelu | TPRELU | tile | ✅ | ✅ | ❌ | ✅ | — | canonical 3-input path; verified on A2/A3 hardware, A5 hardware verification pending |
| pto.tadds | TADDS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tsubs | TSUBS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | verified on A2/A3 hardware; A5 hardware verification pending |
| pto.tmuls | TMULS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.taxpy | TAXPY | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tdivs | TDIVS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmaxs | TMAXS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tmins | TMINS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.trems | TREMS | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 exact-op hardware ST passed for FP32 and INT32 scalar forms within `[-2^24, 2^24]`, including both INT32 boundaries, negative values, and tail valid shapes; A5 hardware pending |
| pto.taddc | TADD + TADD | tile | ✅ | ✅ | ❌ | ✅ | — | verified on A2/A3 hardware, including overflow, full/tail valid shapes, and result/src0 alias and non-alias paths; A5 hardware verification pending |
| pto.tsubc | TSUB + TADD | tile | ✅ | ✅ | ❌ | ✅ | — | verified on A2/A3 hardware, including borrow, full/tail valid shapes, and result/src0 alias and non-alias paths; A5 hardware verification pending |
| pto.taddsc | TADDS + TADD | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 hardware passes full, column-tail, and combined-tail shapes with result/src0 alias and non-alias paths; PTO ISA v0.57's full-column row-tail fast path is a strict expected failure; A5 hardware verification pending |
| pto.tsubsc | TSUBS + TADD | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 hardware passes full, column-tail, and combined-tail shapes with result/src0 alias and non-alias paths; PTO ISA v0.57's full-column row-tail fast path is a strict expected failure; A5 hardware verification pending |
| pto.tabs | TABS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tneg | TNEG | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.texp | TEXP | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tlog | TLOG | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | verified on A2/A3 hardware; A5 hardware verification pending |
| pto.tsqrt | TSQRT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.ttri | TTRI | tile | ✅ | ✅ | ❌ | ✅ | — | frontend + exact codegen + same-name ST; verified on A2/A3 hardware (10/10); A5 hardware pending |
| pto.trsqrt | TRSQRT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trecip | TRECIP | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trelu | TRELU | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tlrelu | TLRELU | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.taddrelu | VADDRELU | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tfmod | TFMOD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 exact-op hardware ST passed for FP32 tile-tile forms with full and tail valid shapes; A5 hardware pending |
| pto.tfmods | TFMODS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 exact-op hardware ST passed for FP32 scalar forms, including negative values and tail valid shapes; A5 hardware pending |
| pto.tpow | TPOW | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tpows | TPOWS | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.trandom | TRANDOM | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | PTOAS source-only compatibility interface |
| **Reductions (13)** |  |  |  |  |  |  |  |  |
| pto.trowsum | TROWSUM | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowmax | TROWMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowargmax | TROWARGMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowmin | TROWMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | verified on A2/A3 hardware; A5 hardware verification pending |
| pto.trowargmin | TROWARGMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowprod | TROWPROD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.thistogram | THISTOGRAM | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tcolsum | TCOLSUM | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolmax | TCOLMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolargmax | TCOLARGMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolmin | TCOLMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolargmin | TCOLARGMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolprod | TCOLPROD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | PTOAS source-only compatibility interface |
| **Broadcasts (17)** |  |  |  |  |  |  |  |  |
| pto.trowexpand | TROWEXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpand | TCOLEXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandmul | TCOLEXPANDMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandadd | TCOLEXPANDADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpanddiv | TCOLEXPANDDIV | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandsub | TCOLEXPANDSUB | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandexpdif | TCOLEXPANDEXPDIF | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandmax | TCOLEXPANDMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandmin | TCOLEXPANDMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandmul | TROWEXPANDMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpanddiv | TROWEXPANDDIV | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandsub | TROWEXPANDSUB | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandadd | TROWEXPANDADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | verified on A2/A3 hardware; A5 hardware verification pending |
| pto.trowexpandexpdif | TROWEXPANDEXPDIF | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandmax | TROWEXPANDMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandmin | TROWEXPANDMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.texpands | TEXPANDS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **Comparison and Selection (4)** |  |  |  |  |  |  |  |  |
| pto.tcmp | TCMP | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcmps | TCMPS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tsel | TSEL | tile | ✅ | ✅ | ❌ | ✅ | — | also the target of `tile.select` when either branch is a scalar TSELS cannot take |
| pto.tsels | TSELS | tile | ✅ | ✅ | ❌ | ✅ | — | canonical 4-input path; verified on A2/A3 hardware, A5 hardware verification pending. The composite `tile.select` lowers here for `mask ? tile : scalar` — it has no same-name `pto.*` op of its own |
| **Bitwise Operations (11)** |  |  |  |  |  |  |  |  |
| pto.tand | TAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 hardware passes signed/unsigned 8/16-bit patterns across full, row-tail, column-tail, and combined-tail shapes; A5 hardware verification pending |
| pto.tor | TOR | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 hardware passes signed/unsigned 8/16-bit patterns across full, row-tail, column-tail, and combined-tail shapes; A5 hardware verification pending |
| pto.txor | TXOR | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 hardware passes signed/unsigned 8/16-bit patterns with explicit tmp across all four shape classes; IR UT covers alias rejection; A5 hardware verification pending |
| pto.tshl | TSHL | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | path exists; historical ISA/semantic issue requires revalidation against the current pin |
| pto.tshr | TSHR | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | path exists; historical ISA/semantic issue requires revalidation against the current pin |
| pto.tnot | TNOT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | covered by the pre-existing same-name `tile.not` ST |
| pto.tands | TANDS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 hardware passes signed/unsigned 8/16-bit tiles with immediate/SSA scalars across all four shape classes; A5 hardware verification pending |
| pto.tors | TORS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 hardware passes signed/unsigned 8/16-bit tiles with immediate/SSA scalars across all four shape classes; A5 hardware verification pending |
| pto.txors | TXORS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 hardware passes signed/unsigned 8/16-bit tiles with immediate/SSA scalars and explicit tmp across all four shape classes; IR UT covers alias rejection; A5 hardware verification pending |
| pto.tshls | TSHLS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | path exists; historical ISA/semantic issue requires revalidation against the current pin |
| pto.tshrs | TSHRS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | path exists; historical ISA/semantic issue requires revalidation against the current pin |
| **Data Rearrangement (15)** |  |  |  |  |  |  |  |  |
| pto.tconcat | TCONCAT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tconcatidx | TCONCAT (indexed) | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tgather | TGATHER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tgatherb | TGATHERB | tile | ✅ | ✅ | ❌ | ✅ | — | 32-byte block-offset frontend + exact codegen + same-name ST; verified on A2/A3 hardware (8/8); A5 hardware pending |
| pto.tscatter | TSCATTER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.mgather | MGATHER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | tensor flat-element Vec interface via `pl.gather(src, index=idx)` when src remains in GM; canonical tile Vec/Mat overloads with row/elem coalesce; expanded Vec/Mat matrix pending |
| pto.mscatter | MSCATTER | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.treshape | TRESHAPE | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tinsert | TINSERT | tile | ✅ | ❌ | ❌ | ✅ | — | emitted by `tile.assemble` / automatic matmul lowering |
| pto.textract | TEXTRACT | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tfillpad | TFILLPAD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tfillpad_expand | TFILLPAD_EXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tfillpad_inplace | TFILLPAD_INPLACE | tile | ✅ | ✅ | ❌ | ❌ | — | emits `pto.tfillpad`; PTOAS planning preserves the destination pad/full validity with a `pto.treshape` alias; allocation-backed results retain runtime validity updates |
| pto.textract_fp | TEXTRACT_FP / TEXTRACT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tinsert_fp | TINSERT_FP / TINSERT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| **Sorting (2)** |  |  |  |  |  |  |  |  |
| pto.tsort32 | TSORT32 | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmrgsort | TMRGSORT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **Type Conversion (1)** |  |  |  |  |  |  |  |  |
| pto.tcvt | TCVT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **Integer Sequences (1)** |  |  |  |  |  |  |  |  |
| pto.tci | TCI | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **Scalar Element Access (2)** |  |  |  |  |  |  |  |  |
| pto.tgetval | .GetValue | tile | ✅ | ✅ | ❌ | ✅ | — | emitted by `tile.read` |
| pto.tsetval | .SetValue | tile | ✅ | ✅ | ❌ | ✅ | — | emitted by `tile.write` |
| **MX Quantization (6)** |  |  |  |  |  |  |  |  |
| pto.tget_scale_addr | GetScaleAddr + TASSIGN | tile | ✅ | ✅ | ✅ | ❌ | — | NEW frontend+codegen; Mat→scale `tmov` emitted in source order, PTOAS `PTOA5NormalizeTMovPass` reorders bind-before-fill; see [operators MX constraints](ir/05-operators.md#mx--ascend950-ptoas-constraints) |
| pto.tmov.fp | TMOV_FP | tile | ✅ | ❌ | ❌ | ❌ | — | backend hook exists; IR/Python frontend and ST are missing |
| pto.tquant | TQUANT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| pto.tquant.mx | TQUANT (overload) | tile | ✅ | ✅ | ❌ | ✅ | — | A5 MXFP8-only standalone quantization frontend+codegen+hardware ST (`group_axis` A/B); MXFP4 quant deferred |
| pto.tstore_fp | TSTORE_FP | tile | ✅ | ❌ | ❌ | ❌ | — | the current backend emits `pto.tstore.fp` |
| pto.tdequant | TDEQUANT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING: lacks a complete frontend/codegen/ST path |
| **Synchronization (8)** |  |  |  |  |  |  |  |  |
| pto.barrier | pipe_barrier / dsb | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.barrier_sync | barrier lowering | internal | ✅ | — | — | — | — | internal synchronization/scheduling primitive; no standalone ST |
| pto.record_event | set_flag lowering | internal | ✅ | — | — | — | — | internal synchronization/scheduling primitive; no standalone ST |
| pto.wait_event | wait_flag lowering | internal | ✅ | — | — | — | — | internal synchronization/scheduling primitive; no standalone ST |
| pto.syncall | SYNCALL | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.sync.set | set_intra_block / FFTS | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.sync.wait | wait_intra_block / wait_flag_dev | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.tsync | TSYNC | internal | ✅ | — | — | — | — | internal synchronization/scheduling primitive; no standalone ST |
| **Core Virtualization (2)** |  |  |  |  |  |  |  |  |
| pto.section.cube | — | internal | ✅ | — | — | — | — | internal synchronization/scheduling primitive; no standalone ST |
| pto.section.vector | — | internal | ✅ | — | — | — | — | internal synchronization/scheduling primitive; no standalone ST |
| **Frontend Pipe (15)** |  |  |  |  |  |  |  |  |
| pto.reserve_buffer | — | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.import_reserved_buffer | — | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.aic_initialize_pipe | TPipe / internal init | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.aiv_initialize_pipe | TPipe / internal init | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.talloc_to_aiv | TALLOC | internal | ✅ | — | — | — | — | covered by pipe-lifecycle integration |
| pto.talloc_to_aic | TALLOC | internal | ✅ | — | — | — | — | covered by pipe-lifecycle integration |
| pto.tpush_to_aiv | TPUSH | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tpush_to_aic | TPUSH | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tpop_from_aic | TPOP | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tpop_from_aiv | TPOP | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tfree_from_aic | TFREE | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.tfree_from_aiv | TFREE | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.tpush | TPUSH | internal | ✅ | — | — | — | — | legacy; superseded by direction-specific interfaces |
| pto.tpop | TPOP | internal | ✅ | — | — | — | — | legacy; superseded by direction-specific interfaces |
| pto.tfree | TFREE | internal | ✅ | — | — | — | — | legacy; superseded by direction-specific interfaces |
| **Runtime Intrinsics (4)** |  |  |  |  |  |  |  |  |
| pto.get_block_idx | get_block_idx | tile+tensor | ✅ | ✅ | ✅ | — | — | codegen converts this to a wrapper parameter; no same-name PTO op is emitted |
| pto.get_subblock_idx | get_subblockid | tile+tensor | ✅ | ✅ | ✅ | — | — | codegen converts this to a wrapper parameter; no same-name PTO op is emitted |
| pto.get_block_num | get_block_num | tile+tensor | ✅ | ✅ | ✅ | — | — | codegen converts this to a wrapper parameter; no same-name PTO op is emitted |
| pto.get_subblock_num | get_subblockdim | internal | ✅ | — | — | — | — | codegen converts this to a wrapper parameter; no same-name PTO op is emitted |
| **Debugging (3)** |  |  |  |  |  |  |  |  |
| pto.tprint | TPRINT | tile | ✅ | ❌ | ❌ | ❌ | — | backend hook exists; IR/Python frontend and ST are missing |
| pto.print | cce::printf | internal | ✅ | — | — | — | — | internal/debugging helper; no standalone ST |
| pto.trap | trap | internal | ✅ | — | — | — | — | internal/debugging helper; no standalone ST |
| **Communication (14)** |  |  |  |  |  |  |  |  |
| pto.comm.build_async_session | pto::comm::BuildAsyncSession | comm | ✅ | — | — | — | ❌ | distributed interface lacks same-name ST coverage |
| pto.comm.tput_async | TPUT_ASYNC | comm | ✅ | — | — | — | ❌ | distributed interface lacks same-name ST coverage |
| pto.comm.tget_async | TGET_ASYNC | comm | ✅ | — | — | — | ❌ | distributed interface lacks same-name ST coverage |
| pto.comm.wait_async_event | AsyncEvent.Wait | comm | ✅ | — | — | — | ❌ | distributed interface lacks same-name ST coverage |
| pto.comm.test_async_event | AsyncEvent.Test | comm | ✅ | — | — | — | ❌ | distributed interface lacks same-name ST coverage |
| pto.comm.tput | TPUT | comm | ✅ | — | — | — | ✅ | test_l3_put / remote_store |
| pto.comm.tget | TGET | comm | ✅ | — | — | — | ✅ | test_l3_get |
| pto.comm.tnotify | TNOTIFY | comm | ✅ | — | — | — | ✅ | test_l3_notify_wait |
| pto.comm.twait | TWAIT | comm | ✅ | — | — | — | ✅ | test_l3_notify_wait |
| pto.comm.ttest | TTEST | comm | ✅ | — | — | — | ❌ | distributed interface lacks same-name ST coverage |
| pto.comm.tbroadcast | TBROADCAST | comm | ✅ | — | — | — | ❌ | high-level tests decompose this op; there is currently no same-name PTO ST |
| pto.comm.tgather | TGATHER | comm | ✅ | — | — | — | ❌ | high-level tests decompose this op; there is currently no same-name PTO ST |
| pto.comm.tscatter | TSCATTER | comm | ✅ | — | — | — | ❌ | high-level tests decompose this op; there is currently no same-name PTO ST |
| pto.comm.treduce | TREDUCE | comm | ✅ | — | — | — | ❌ | high-level tests decompose this op; there is currently no same-name PTO ST |
| **Stack-local Arrays / Structs (6)** |  |  |  |  |  |  |  |  |
| pto.declare_local_array | — | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.local_array_get | — | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.local_array_set | — | internal | ✅ | — | — | ✅ | — | emitted automatically by compiler/system lowering |
| pto.declare_struct | — | internal | ✅ | — | — | — | — | internal/debugging helper; no standalone ST |
| pto.struct_get | — | internal | ✅ | — | — | — | — | internal/debugging helper; no standalone ST |
| pto.struct_set | — | internal | ✅ | — | — | — | — | internal/debugging helper; no standalone ST |
| **Source Compatibility / Manual Mode (1)** |  |  |  |  |  |  |  |  |
| pto.tassign | TASSIGN | internal | ✅ | — | — | — | — | inactive backend hook; no standalone ST |

**Stats**: 204 public/compatibility PTOAS ops; 113 have a pypto tile frontend and 75 have a tensor frontend
(plus four non-tile/tensor `pl.prefetch.*` ops); 121 have same-name ST coverage
(117 regular STs and 4 distributed STs); 51 lack same-name ST coverage (41 regular and 10 distributed);
the remaining 32 ops are not suitable for standalone STs.
