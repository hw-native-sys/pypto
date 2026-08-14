# Host

The one host-side cost that dominates everything else: copying data that did not need to
move.

> **Prerequisites:** none beyond running a kernel.

## Check this first, not last

`run()` does not hand you a host/device split — its `execution_time` is the whole wall
clock, compile and golden comparison included. `pypto.runtime.benchmark` does: its
`BenchmarkStats` carries `device_wall_us` and `host_wall_us` per round. When the host span
is the large one, nothing in pages [00](00-swimlane.md)–[05](05-memory.md) will move your
number — they all tune device-side work that is not the bottleneck.

The usual cause is not subtle. A kernel invoked in a loop with the same large weight
argument uploads that weight **every call**:

```text
per call:  H2D weights ──► compute ──► D2H results
           ▲──────────── the same bytes, every iteration
```

## Keeping resident data resident

`ChipWorker.alloc_tensor` allocates persistent device memory and returns a `DeviceTensor`
handle that a compiled program accepts wherever a `torch.Tensor` goes. The runtime treats
the buffer as already resident and skips both H2D and D2H for that argument.

```python
import torch
from pypto import ir
from pypto.runtime import ChipWorker, RunConfig

cfg = RunConfig(platform="a2a3")      # the artifact and the worker must agree
compiled = ir.compile(MyKernel, platform=cfg.platform)

with ChipWorker(config=cfg) as w:
    weight = w.alloc_tensor((1024, 4096), torch.float16, init=host_weight)  # uploaded once
    for batch in batches:
        out = torch.empty(batch.shape[0], 4096, dtype=torch.float16)
        compiled(batch, weight, out)                                        # not re-uploaded
    w.free_tensor(weight)
```

**Cost — three obligations you now own:**

- A `DeviceTensor` is **not** copied back automatically. If a kernel writes to one, read it
  back yourself with `w.copy_from(host_ptr, t.data_ptr, t.nbytes)` on the same worker.
- Free it with `w.free_tensor(t)` when you are done. The buffer is otherwise held for the
  worker's whole lifetime; `close()` does auto-free what you forgot, but that is a backstop,
  not the plan.
- Only the worker that allocated the buffer can use it.

**How to confirm:** `host_wall_us` shrinks and `device_wall_us` does not move. If device
time changed too, something else changed with it.

## What else is resident

The same reasoning covers anything that outlives one call:

| Data | Why it stays | Example |
| ---- | ------------ | ------- |
| Weights | Read every call, never written | The snippet above |
| KV cache | Written by one call, read by the next | `examples/runtime/multi_program_kv_cache.py` |
| Scratch / workspace | Never leaves the device at all | Allocate once, pass every call |

A KV cache is the case where residency is not just an optimisation — copying it back and
forth would dominate a decode step. `w.alloc_tensor(...)` holds one buffer across several
programs registered on the same worker.

## Registering once

The second host cost is registration. Compiling and registering a program on every
invocation pays the setup repeatedly; the register-once, dispatch-many pattern pays it
once:

```python
from pypto.runtime import benchmark

stats = benchmark(compiled, [a, b, c], rounds=100, warmup=3,
                  platform="a2a3", device_id=0)
print(stats.device_wall_us_median, stats.device_wall_us_min)
```

`benchmark` owns the loop: it registers the *compiled* program once, then dispatches
`rounds` cheap launches with no per-round register or load, and aggregates the `[STRACE]`
timing markers of each. That is also the right way to get a stable device number, because
it excludes the setup you are not trying to measure.

`examples/runtime/explicit_dispatch.py` shows the same structure for three real shapes —
an inference service, a training loop, and a register/dispatch overhead check.

## See also

- [Getting started § DeviceTensor](../00-getting_started.md#reusing-weights-on-the-worker-devicetensor)
  — the reference treatment, including the explicit-dispatch API.
