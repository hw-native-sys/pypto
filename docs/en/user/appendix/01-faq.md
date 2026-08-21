# FAQ and Known Limitations

Questions that come up repeatedly, and the constraints behind them.

## Writing kernels

**Why did my kernel run and write nothing?**
`out = pl.add(a, b)` rebinds a Python name; it does not write the output parameter. Use
`out[:] = pl.add(a, b)`, which is sugar for `pl.assemble`. This is the single most common
first bug, and it fails silently — the run succeeds and the output stays zero. See
[Syntax](../language/06-syntax.md).

**Why is a plain `range` rejected?**
The DSL needs to know whether a loop is a compile-time unroll, a device-side loop, or a
parallel dispatch. Use `pl.range` / `pl.unroll` / `pl.parallel` / `pl.pipeline` /
`pl.spmd`. See [Control Flow](../language/02-control-flow.md).

**Why can't I collect TaskIds in a Python list?**
The body is traced, not executed, so `tids.append(tid)` is not a DSL operation. Use
`pl.array.create(n, pl.TASK_ID)` and index-assign. See
[Declaring an edge](../tasks/02-submit.md).

**Why does `pl.dynamic` fail on my `@pl.jit` entry?**
A dynamic axis belongs on an InCore kernel's annotation. Orchestration-level work on a
dynamic-shaped tensor reaches `InitMemRef`, which needs a constant dim. Keep the kernel
`@pl.jit.incore` and let the entry stay dynamic while delegating to it. See
[Types](../language/00-types.md).

## Compiling

**`TypeError: got an unexpected keyword argument` from `compile()`.**
`compile()`'s positional arguments are the kernel's own. Compile options travel in
`config=RunConfig(...)`. See [Compiling](../execution/00-compile.md).

**The worker rejects my compiled program.**
The artifact's `platform` must match the worker's. Compile for the platform you dispatch
on.

**`ptoas compilation failed:` with an empty message.**
That is a crash in the ptoas binary rather than a rejection of your IR. Point `PTOAS_ROOT`
at a working version.

## Running

**Why is `run()`'s time so much larger than the kernel?**
`RunResult.execution_time` is total wall clock — compile, golden generation and validation
included. For the device/host split use `pypto.runtime.benchmark`, whose `BenchmarkStats`
carries `device_wall_us` and `host_wall_us`. See [Host](../performance/06-host.md).

**Why does passing a `DeviceTensor` to a `@pl.jit` kernel fail?**
A `DeviceTensor` carries no shape or dtype for the specializer to read. Dispatch a
*compiled* program instead. See [Running](../execution/01-run.md).

**A capacity error names a ring I never configured.**
Ring choice follows scope depth — `min(scope_depth, 3)` — so a deeply nested kernel
concentrates work on ring 3 and a flat one puts everything on ring 0. Measure with
`enable_scope_stats` before resizing. See [Memory](../performance/05-memory.md).

## Numbers

**Split-K results differ between runs.**
Atomic accumulation order across cores is not fixed. Expect last-place differences; that is
correct behaviour, not a bug. See [Precision](../precision/00-workflow.md).

**A correct FP16 kernel fails `allclose`.**
The default `rtol=1e-5` is wrong for FP16 inputs, which carry about three decimal digits.
Match the tolerance to the input precision before investigating the kernel.

**Is the multi-hop cast lossy?**
No. `INT32 -> FP32 -> FP16` on A5 is bit-identical to a single-step conversion under the
same rounding and overflow behaviour — the page proves it with a runnable check. See
[Precision](../precision/00-workflow.md).

## Known limitations

| Limitation | Detail |
| ---------- | ------ |
| **Two co-live `MemRef` slots under PTOAS** | Rejected at codegen: ptoas guards only the first `multi_tile_get` of an iteration. One slot live per iteration is the shape to write |
| **Hard `syncall` needs full occupancy** | A partial launch deadlocks on device (507018); PyPTO rejects it at compile time. Use `mode=pl.SyncAllMode.SOFT` at partial occupancy |
| **Ring allreduce is not a one-argument change** | It needs an explicit `[2*(NR-1)+1, NR]` INT32 signal, a statically-known `src` shape, and `numel(src)` divisible by `NR`. See [Collectives](../distributed/01-collectives.md) |
| **`memory_planner=PTOAS` and the memory map** | Allocation passes are skipped, so pass dumps carry no offsets for the tool to draw |
| **Doc code blocks are backend-specific** | The manual's runnable blocks execute on `a2a3sim`; A5-only behaviour is argued rather than executed |

## See Also

- [Feature matrix](00-feature-matrix.md) — what each backend supports.
- [Tools](../tools/index.md) — the debugging surface behind most of these answers.
- [Getting started](../00-getting_started.md) — if you are hitting several of these at once.
