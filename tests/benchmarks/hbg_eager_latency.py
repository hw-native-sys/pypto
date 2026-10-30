# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Diagnostic bounds for one fixed HBG eager Add contract; not a production fast path.

Run each TASK_QUEUE_ENABLE setting in a fresh process. All prepared stages are
valid ONLY for these retained tensors, metadata, artifact, context and stream.
They deliberately omit guards, so their deltas are opportunity bounds, not
predictions for a transparent cache. Direct native submission bypasses the
Torch task queue and must be reported separately when that queue is enabled.

Stages progressively retain source leases, bypass resolution with one artifact,
reuse one frame, reuse TaskArgs/config, and reuse native materialized arguments.
Only the final direct stage bypasses the queue callback. Each delta includes all
work removed between those entry points, not just the method naming the stage.
Host samples synchronize outside the call interval; lifecycle batches include
setup, per-call synchronization, completion, real lease release and registry
revocation. They measure process CPU, including background threads.

Example: TASK_QUEUE_ENABLE=0 python tests/benchmarks/hbg_eager_latency.py
--device 0 --mode bench --output /tmp/hbg-bench-q0
For a separate phases process, set SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 and
optionally LD_PRELOAD to hbg_launch_probe.so with PYPTO_LAUNCH_PROBE_OUTPUT
naming a new CSV. Never enable the launch probe for formal bench measurements.
"""

import argparse
import cProfile
import importlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import pypto
import pypto.language as pl
import pypto.torch
import torch
from pypto.runtime.kernel.arguments import BorrowedArguments, _load_api
from pypto.runtime.kernel.context import get_process_kernel_state
from pypto.runtime.kernel.session import make_call_config
from pypto.torch import launch
from pypto.torch.interop import CallSignature

# Optional hardware packages are imported dynamically, as in the Torch adapter.
torch_npu = importlib.import_module("torch_npu")
npu = torch_npu.npu
configure_logging = importlib.import_module("simpler_setup.log_config").configure_logging


@pl.jit.incore
def add_core(x: pl.Tensor, y: pl.Tensor, out: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    a = pl.load(x, [0, 0], [rows, cols])
    b = pl.load(y, [0, 0], [rows, cols])
    pl.store(pl.add(a, b), [0, 0], out)
    return out


@pl.jit
def add_kernel(x: pl.Tensor, y: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = add_core(x, y, out)
    return out


STAGES = ("jit", "sources", "artifact", "frame", "arguments", "materialized", "direct", "aclnn")


class FixedProbe:
    """Own every borrowed object until queued callbacks and native runs finish."""

    def __init__(self, device: int) -> None:
        self.device = device
        self.x = torch.arange(4096, dtype=torch.float32, device=f"npu:{device}").reshape(32, 128)
        self.y = torch.full_like(self.x, 0.25)
        self.out, self.aclnn_out = torch.empty_like(self.x), torch.empty_like(self.x)
        self.inputs = (self.x, self.y, self.out)
        self.expected = self.x.cpu() + self.y.cpu()
        torch.add(self.x, self.y, out=self.aclnn_out)
        npu.synchronize()
        torch.testing.assert_close(self.aclnn_out.cpu(), self.expected, rtol=0, atol=0)
        pypto.torch.init(device=device, platform="a2a3", runtime="host_build_graph")
        self.state = get_process_kernel_state()
        self.config = self.state.require_config()
        self.native = launch._load_native()
        self.raw_stream = 0
        # enqueue borrows the exact callable object, not merely its bound self.
        self._capture_callback = self.capture_stream
        self._submit_callback = self.callback
        stream = npu.current_stream(device)
        self.context = self.native.borrow_context(device)
        # Also warms the adapter's once-per-process Python-callback bootstrap.
        self.native.enqueue([], stream.stream_id, device, self.context, self._capture_callback)
        npu.synchronize()
        assert self.raw_stream > 0
        observed = []
        resolver = add_kernel._resolve_compiled

        def capture(*args: Any, **kwargs: Any) -> Any:
            result = resolver(*args, **kwargs)
            observed.append(result[0])
            return result

        add_kernel._resolve_compiled = capture
        try:
            assert add_kernel(*self.inputs) is self.out
            npu.synchronize()
        finally:
            del add_kernel._resolve_compiled
        assert len(observed) == 1
        self.artifact = observed[0]
        metadata = self.artifact.metadata
        self.signature = CallSignature(
            metadata.parameters, return_aliases=metadata.return_aliases, platform="a2a3"
        )
        self.frame = self.signature.describe_call(self.inputs)
        self.session = self.state._worker
        self.source_api = _load_api()
        self.worker = self.session.worker
        self.chip = self.worker._chip_worker
        self.registration = self.state.ensure_callable(self.artifact, self.config)
        self.callable_id = self.worker._resolve_handle(
            self.registration.handle, expected_namespace="LOCAL_CHIP"
        ).slot_id
        self.call_config = make_call_config(self.config, self.source_api.CallConfig)
        self.arguments = None
        self.chip_args = None
        self.identities = ()
        self.last_run = None
        self.callback_error = None
        self.stage = "jit"
        self.tensors = list(self.inputs)
        self.finish()

    def capture_stream(self, stream: int) -> None:
        self.raw_stream = stream

    def retire(self) -> None:
        if self.callback_error is not None:
            raise self.callback_error
        if self.last_run is not None:
            self.last_run.wait()
            self.last_run = None

    def finish(self) -> None:
        npu.synchronize()
        self.retire()
        self.state.drain()
        if self.identities:
            self.worker._device_buffer_source.release(self.identities)
            self.identities = ()
        if self.arguments is not None:
            self.arguments.release()
            self.arguments = None
        self.chip_args = None
        # Close the current batch's lazy registry revocations as well as leases.
        with self.worker._child_prov_worker_lock(0), self.worker._child_prov_lock:
            empty = self.source_api.TaskArgs()
            assert self.worker._attach_source_buffers_locked(empty) == ()
            assert not self.worker._source_attachments
            assert not self.worker._device_buffer_source._records
        assert not self.session.sources._sources

    def prepare(self, stage: str) -> None:
        self.stage = stage
        if stage not in ("jit", "aclnn"):
            self.arguments = BorrowedArguments(self.frame, self.session.sources, self.session.device_location)
        if stage in ("materialized", "direct"):
            assert self.arguments is not None
            # Keep provenance/source reservations live through every native run.
            # No other submitter or allocation may mutate this diagnostic fixture.
            with self.worker._child_prov_worker_lock(0), self.worker._child_prov_lock:
                self.identities = self.worker._attach_source_buffers_locked(self.arguments.task_args)
            self.chip_args = self.worker._materialize_l2_args(self.arguments.task_args)

    def callback(self, stream: int) -> None:
        try:
            if self.stage == "arguments":
                assert self.arguments is not None
                self.last_run = self.worker.submit(
                    self.registration.handle, self.arguments.task_args, self.call_config, caller_stream=stream
                )
            else:
                self.last_run = self.chip._submit_chip_run_direct(
                    self.callable_id, self.chip_args, self.call_config, caller_stream=stream
                )
        except BaseException as error:
            self.callback_error = error
            raise

    def call(self) -> Any:
        if self.stage == "aclnn":
            return torch.add(self.x, self.y, out=self.aclnn_out)
        if self.stage in ("jit", "sources"):
            return add_kernel(*self.inputs)
        if self.stage == "artifact":
            launch.check_eager_call(self.config)
            return launch.invoke(self.artifact, self.signature.describe_call(self.inputs), self.config)
        if self.stage == "frame":
            return launch.invoke(self.artifact, self.frame, self.config)
        self.retire()
        if self.stage == "direct":
            self.callback(self.raw_stream)
        else:
            self.native.enqueue(
                self.tensors, self.frame.stream.stream_id, self.device, self.context, self._submit_callback
            )
        return self.out

    def check(self) -> None:
        npu.synchronize()
        self.retire()
        torch.testing.assert_close(self.out.cpu(), self.expected, rtol=0, atol=0)
        torch.testing.assert_close(self.aclnn_out.cpu(), self.expected, rtol=0, atol=0)


def measure(probe: FixedProbe, output: Path, samples: int, batches: int, stages: tuple[str, ...]) -> None:
    host, lifecycle = [], []
    for batch in range(batches):
        order = stages[batch % len(stages) :] + stages[: batch % len(stages)]
        if batch % 2:
            order = order[::-1]
        for stage in order:
            probe.finish()
            probe.prepare(stage)
            try:
                for _ in range(50):
                    probe.call()
                    npu.synchronize()
                for sample in range(samples):
                    npu.synchronize()
                    start = time.perf_counter_ns()
                    cpu = time.thread_time_ns()
                    probe.call()
                    cpu_end = time.thread_time_ns()
                    end = time.perf_counter_ns()
                    npu.synchronize()
                    host.append(
                        {
                            "batch": batch,
                            "stage": stage,
                            "sample": sample,
                            "host_us": (end - start) / 1000,
                            "caller_cpu_us": (cpu_end - cpu) / 1000,
                        }
                    )
                probe.check()
            finally:
                probe.finish()
            print("HOST", batch, stage, flush=True)
        (output / "host.json").write_text(json.dumps(host))
    for batch in range(12):
        order = stages if batch % 2 == 0 else stages[::-1]
        for stage in order:
            probe.finish()
            start = time.perf_counter_ns()
            cpu = time.process_time_ns()
            try:
                probe.prepare(stage)
                for _ in range(100):
                    probe.call()
                    npu.synchronize()
            finally:
                probe.finish()
            cpu_end = time.process_time_ns()
            end = time.perf_counter_ns()
            lifecycle.append(
                {
                    "batch": batch,
                    "stage": stage,
                    "cpu_us": (cpu_end - cpu) / 100000,
                    "wall_us": (end - start) / 100000,
                }
            )
    (output / "lifecycle.json").write_text(json.dumps(lifecycle))
    summary = {
        stage: {
            "host_us": statistics.median(x["host_us"] for x in host if x["stage"] == stage),
            "lifecycle_cpu_us": statistics.median(x["cpu_us"] for x in lifecycle if x["stage"] == stage),
        }
        for stage in stages
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def phases(probe: FixedProbe, output: Path, samples: int) -> None:
    # Delimit the measured interval in the same monotonic ns clock as host spans.
    # Logging and LD_PRELOAD probes are forbidden in the formal benchmark process.
    probe.prepare("direct")
    for _ in range(20):
        probe.call()
        npu.synchronize()
    intervals = []
    for sample in range(samples):
        for stage in ("direct", "aclnn") if sample % 2 == 0 else ("aclnn", "direct"):
            probe.stage = stage
            npu.synchronize()
            start = time.monotonic_ns()
            probe.call()
            returned = time.monotonic_ns()
            npu.synchronize()
            probe.retire()
            intervals.append(
                {
                    "sample": sample,
                    "stage": stage,
                    "start_ns": start,
                    "return_ns": returned,
                    "complete_ns": time.monotonic_ns(),
                }
            )
    probe.finish()
    (output / "intervals.json").write_text(json.dumps(intervals))


def profile_calls(probe: FixedProbe, output: Path, samples: int) -> None:
    """Profile the caller thread separately from uninstrumented timings.

    Use queue=0 to include the synchronous callback. With queue=1 this excludes
    background submission; it is never a whole-process lifecycle CPU measure.
    """
    for stage in ("jit", "artifact", "frame", "arguments"):
        probe.prepare(stage)
        try:
            for _ in range(20):
                probe.call()
                npu.synchronize()
            profile = cProfile.Profile()
            for _ in range(samples):
                npu.synchronize()
                profile.runcall(probe.call)
                npu.synchronize()
            profile.dump_stats(str(output / f"{stage}.prof"))
            probe.check()
        finally:
            probe.finish()


def measure_source_handoff(probe: FixedProbe, output: Path, samples: int, batches: int) -> None:
    """Isolate admission handoff with the same complete JIT and fresh frames."""
    pool = probe.session.sources
    hold = pool.hold_existing
    prepare = probe.prepare

    def prepare_variant(stage: str) -> None:
        pool.hold_existing = (lambda arguments: lambda: None) if stage == "baseline" else hold
        prepare("aclnn" if stage == "aclnn" else "jit")

    probe.prepare = prepare_variant
    try:
        measure(probe, output, samples, batches, ("baseline", "candidate", "aclnn"))
    finally:
        try:
            probe.finish()
        finally:
            probe.prepare = prepare
            pool.hold_existing = hold


def count_source_lifecycle(probe: FixedProbe, output: Path, samples: int) -> None:
    """Count separately from timings, including final lazy registry revocation."""
    pool = probe.session.sources
    worker = probe.worker
    acquire, release = pool.acquire, pool.release
    hold = pool.hold_existing
    record, drop = worker._record_device_alloc, worker._drop_device_alloc
    result = {}
    for variant in ("baseline", "candidate"):
        probe.finish()
        counts = dict.fromkeys(("acquire", "miss", "release", "last_release", "record", "drop"), 0)

        def count_acquire(argument: Any, location: Any, api: Any) -> Any:
            counts["acquire"] += 1
            key = (argument.metadata.storage_ptr, argument.metadata.storage_nbytes)
            counts["miss"] += int(key not in pool._sources)
            return acquire(argument, location, api)

        def count_release(key: Any) -> None:
            counts["release"] += 1
            counts["last_release"] += int(pool._sources[key].uses == 1)
            release(key)

        def count_record(*args: Any, **kwargs: Any) -> Any:
            counts["record"] += 1
            return record(*args, **kwargs)

        def count_drop(*args: Any, **kwargs: Any) -> Any:
            counts["drop"] += 1
            return drop(*args, **kwargs)

        pool.acquire, pool.release = count_acquire, count_release
        worker._record_device_alloc, worker._drop_device_alloc = count_record, count_drop
        pool.hold_existing = (lambda arguments: lambda: None) if variant == "baseline" else hold
        try:
            probe.prepare("jit")
            for _ in range(samples):
                assert probe.call() is probe.out
                npu.synchronize()
            probe.check()
            probe.finish()
            expected = 3 * samples if variant == "baseline" else 3
            assert counts["miss"] == counts["last_release"] == counts["record"] == counts["drop"] == expected
            result[variant] = counts.copy()
        finally:
            pool.acquire, pool.release, pool.hold_existing = acquire, release, hold
            worker._record_device_alloc, worker._drop_device_alloc = record, drop
            probe.finish()
    (output / "source-counts.json").write_text(json.dumps(result, indent=2))


class _DiagnosticPatches:
    """Restore process-local counterfactuals after every drained measurement block."""

    def __init__(self) -> None:
        self.saved: list[tuple[Any, str, bool, Any]] = []

    def set(self, owner: Any, name: str, value: Any) -> None:
        self.saved.append((owner, name, name in vars(owner), getattr(owner, name)))
        setattr(owner, name, value)

    def restore(self) -> None:
        while self.saved:
            owner, name, local, value = self.saved.pop()
            if local:
                setattr(owner, name, value)
            else:
                delattr(owner, name)


def measure_resolver_layers(probe: FixedProbe, output: Path, samples: int, batches: int) -> None:
    """Cumulative fixed-contract bounds; deliberately freeze dynamic invalidation.

    The complete JIT entry and unchanged Torch/Simpler submission still execute.
    Later layers include earlier freezes; their differences are not independent
    opportunities. Frozen values are valid only for this single-caller fixture.
    """
    from pypto.jit import decorator  # noqa: PLC0415

    groups = (
        ((add_kernel, "_resolve_specialization"),),
        ((add_kernel, "_get_dep_graph_state"),),
        tuple(
            (add_kernel, name)
            for name in (
                "_get_source_hash",
                "_dep_declared_layouts",
                "_resolve_constexpr_bindings",
                "_constexpr_identity_records",
            )
        ),
        tuple(
            (decorator, name)
            for name in (
                "_resolve_compile_request",
                "capture_cache_config",
                "_platform_string",
                "_resolve_runtime",
                "_eager_platform",
                "_parse_platform",
                "_eager_runtime",
                "_resolve_memory_planner",
                "_resolve_enable_pypto_l0c_double_buffer",
                "_resolve_enable_buffer_ir",
            )
        ),
        ((decorator, "make_cache_key"),),
        ((add_kernel, "_resolve_compiled"),),
    )
    stages = ("baseline", "specialization", "dependencies", "identity", "policy", "key", "resolved", "aclnn")
    patches = _DiagnosticPatches()
    captured: dict[str, Any] = {}

    def capture(name: str, original: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            # The resolver writes target/runtime into this mapping after return.
            captured[name] = (dict(result[0]), result[1]) if name == "_resolve_compile_request" else result
            return result

        return call

    try:
        for group in groups:
            for owner, name in group:
                patches.set(owner, name, capture(name, getattr(owner, name)))
        probe.prepare("jit")
        probe.call()
        probe.check()
    finally:
        try:
            probe.finish()
        finally:
            patches.restore()
    assert all(name in captured for group in groups for _, name in group)

    def frozen(name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            value = captured[name]
            return (dict(value[0]), value[1]) if name == "_resolve_compile_request" else value

        return call

    prepare = probe.prepare

    def prepare_variant(stage: str) -> None:
        patches.restore()
        prepare("aclnn" if stage == "aclnn" else "jit")
        (probe.aclnn_out if stage == "aclnn" else probe.out).fill_(float("nan"))
        npu.synchronize()
        if stage != "aclnn":
            for group in groups[: stages.index(stage)]:
                for owner, name in group:
                    patches.set(owner, name, frozen(name))

    (output / "layers.json").write_text(
        json.dumps(
            {
                stage: [name for group in groups[:i] for _, name in group]
                for i, stage in enumerate(stages[:-1])
            },
            indent=2,
        )
    )
    probe.prepare = prepare_variant
    try:
        measure(probe, output, samples, batches, stages)
    finally:
        try:
            probe.finish()
        finally:
            patches.restore()
            probe.prepare = prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "bench",
            "phases",
            "profile",
            "handoff",
            "counts",
            "resolver-layers",
        ),
        default="smoke",
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    args = parser.parse_args()
    if args.mode in (
        "bench",
        "handoff",
        "resolver-layers",
    ) and (os.environ.get("LD_PRELOAD") or os.environ.get("PYPTO_LAUNCH_PROBE_OUTPUT")):
        parser.error("formal bench requires a fresh process without LD_PRELOAD or launch probes")
    args.output.mkdir(parents=True, exist_ok=True)
    # The parent already loaded the interposer. Do not preload it into compiler
    # subprocesses, which must not open or append to the parent's evidence file.
    os.environ.pop("LD_PRELOAD", None)
    os.environ.pop("PYPTO_LAUNCH_PROBE_OUTPUT", None)
    native_log = None
    if args.mode == "phases":
        native_log = importlib.import_module("_task_interface")
        log_directory = args.output / "native-log"
        log_directory.mkdir(parents=True, exist_ok=True)
        assert native_log._set_host_log_directory(str(log_directory)) == str(log_directory)
    configure_logging("timing" if args.mode == "phases" else "error")
    torch.set_num_threads(1)
    npu.set_device(args.device)
    npu.set_compile_mode(jit_compile=False)
    probe = FixedProbe(args.device)
    try:
        for stage in STAGES:
            # Change contents without changing the fixed storage contract. Poison
            # output so a skipped/no-op launch cannot pass on the previous result.
            probe.x.add_(1)
            probe.expected.add_(1)
            if stage == "aclnn":
                probe.out.copy_(probe.x + probe.y)
            else:
                probe.out.fill_(float("nan"))
            probe.aclnn_out.copy_(probe.x + probe.y)
            npu.synchronize()
            probe.prepare(stage)
            assert probe.call() is (probe.aclnn_out if stage == "aclnn" else probe.out)
            successor = (probe.aclnn_out if stage == "aclnn" else probe.out) * 2
            probe.check()
            torch.testing.assert_close(successor.cpu(), probe.expected * 2, rtol=0, atol=0)
            probe.finish()
        identity = {
            "mode": args.mode,
            "pypto": pypto.__file__,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": args.device,
            "queue": os.environ.get("TASK_QUEUE_ENABLE"),
            "contract": "fixed storage/shape/dtype/offset/stream/scalars; no mutation or concurrency",
            "direct": "bypasses Torch task queue; not a queue-preserving optimization",
            "paired_control": "handoff restores selected old work, not an old install",
        }
        (args.output / "identity.json").write_text(json.dumps(identity, indent=2))
        if args.mode == "bench":
            measure(probe, args.output, args.samples, args.batches, tuple(args.stages))
        elif args.mode == "phases":
            phases(probe, args.output, args.samples)
        elif args.mode == "profile":
            profile_calls(probe, args.output, args.samples)
        elif args.mode in ("handoff", "resolver-layers"):
            {
                "handoff": measure_source_handoff,
                "resolver-layers": measure_resolver_layers,
            }[args.mode](probe, args.output, args.samples, args.batches)
        elif args.mode == "counts":
            count_source_lifecycle(probe, args.output, args.samples)
    finally:
        probe.finish()
        probe.state.close()
        if native_log is not None:
            assert native_log._flush_host_log(1000)
            drops = native_log._host_log_dropped_records_by_reason()
            (args.output / "log-drops.json").write_text(json.dumps(drops))
            assert native_log._host_log_dropped_records() == 0, drops
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
