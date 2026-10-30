# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure the real CSA fixture through this checkout's KERNEL eager entry.

The external fixture supplies formal layer weights and synthetic history. Its
Native numerical comparison remains separate from exact ablation equivalence.
No graph is captured. Each queue setting/seed must run in a fresh process.
"""

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import logging
import os
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def bind_csa(source: Path, native: Path) -> Any:
    """Load only the CSA fixture dependencies, preserving installed PyPTO/runtime."""
    sys.path[:0] = [str(source), str(source / "tests/pypto_test"), str(native / "src/vllm")]

    class NativeExtensionFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname == "vllm_ascend.vllm_ascend_C":
                return importlib.machinery.PathFinder.find_spec(
                    fullname, [str(native / "src/vllm-ascend/vllm_ascend")]
                )
            return None

    sys.meta_path.insert(0, NativeExtensionFinder())
    environment: Any = importlib.import_module("dsv4_csa_env")
    environment.activate = lambda: source
    os.environ.update(DYNAMIC_EPLB="false", EXPERT_MAP_RECORD="false")
    case = importlib.import_module("dsv4_csa_single_layer")
    package = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa"
    adapter = importlib.import_module(f"{package}.native_adapter")
    reduction = importlib.import_module(f"{package}.reduction")
    root = importlib.import_module(f"{package}.decode_csa").decode_csa_tp1_layer_hbg

    def register(cls: Any) -> Any:
        reduction.validate_reduction_mode()
        return cls(root)

    original_init = adapter.HBGNativeCSACall.__init__

    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, kernel=root, **kwargs)

    adapter.HBGCSAOperators.register = classmethod(register)
    adapter.HBGNativeCSACall.__init__ = initialize
    return case, root


def check_state(
    case: Any, fixture: Any, output: Any, topk: Any, reference: Any, exact: bool
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    observed = case.collect_state(fixture, output, topk())
    guards = case.guard_checks(fixture)
    assert all(check["status"] == "PASS" for check in guards.values()), guards
    comparisons = {
        name: case.compare_tensor(value, reference[name], 0, 0) for name, value in observed.items()
    }
    assert all(check.get("nonfinite") == 0 for check in comparisons.values()), comparisons
    if exact:
        assert all(check["status"] == "PASS" for check in comparisons.values()), comparisons
    hashes = {
        name: hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, value in observed.items()
    }
    return {"guards": guards, "comparison": comparisons, "hashes": hashes}


def measure_cpu(args: Any, stages: Any, reset: Any, run: Any, drain: Any, verify: Any) -> None:
    """Accumulate call/completion CPU and amortize one real release per group."""
    npu = importlib.import_module("torch_npu").npu
    rng = random.Random(args.seed + 1)
    lifecycle, orders = [], []
    for batch in range(args.blocks):
        order = list(stages)
        rng.shuffle(order)
        orders.append(order)
        for stage in order:
            drain()
            if experiment := getattr(args, "_experiment", None):
                experiment.install(stage)
            cpu_ns = 0
            for _ in range(args.cpu_samples):
                reset()
                begin = time.process_time_ns()
                run()
                # Preserve the preceding call's leases until the next call;
                # synchronization completes device work, not source ownership.
                npu.synchronize()
                cpu_ns += time.process_time_ns() - begin
            begin = time.process_time_ns()
            drain()
            cpu_ns += time.process_time_ns() - begin
            checked = verify()
            lifecycle.append(
                {
                    "batch": batch,
                    "stage": stage,
                    "calls": args.cpu_samples,
                    "cpu_us": cpu_ns / args.cpu_samples / 1000,
                    "checks": checked,
                }
            )
            print("CSA_CPU_PASS", batch, stage, flush=True)
        (args.output / "pto-lifecycle.json").write_text(json.dumps(lifecycle))
        (args.output / "pto-lifecycle-orders.json").write_text(json.dumps(orders))


def diagnose_sources(
    args: Any, stages: Any, root: Any, reset: Any, run: Any, drain: Any, verify: Any
) -> dict[str, Any]:
    """Count real source registration/release in a separate, untimed process."""
    from hbg_eager_latency import _DiagnosticPatches  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    state = get_process_kernel_state()
    session = state._worker
    rows = []
    for stage in stages:
        drain()
        counters: dict[str, int] = {}
        patches = _DiagnosticPatches()

        def instrument(owner: Any, name: str, label: str) -> None:
            original = getattr(owner, name)
            counters[label] = 0

            def counted(*call_args: Any, **kwargs: Any) -> Any:
                counters[label] += 1
                return original(*call_args, **kwargs)

            patches.set(owner, name, counted)

        try:
            for owner, name, label in (
                (root, "_resolve_compiled", "resolver_calls"),
                (root, "_compile", "compile_attempts"),
                (session.sources, "acquire", "source_acquire"),
                (session.sources, "release", "source_release"),
                (session.sources, "_release_buffer", "source_buffer_release"),
                (session.worker, "_record_device_alloc", "consumer_register"),
                (session.worker, "_drop_device_alloc", "consumer_unregister"),
            ):
                instrument(owner, name, label)
            for _ in range(args.samples):
                reset()
                run()
            drain()
        finally:
            patches.restore()
        assert counters["resolver_calls"] == args.samples and counters["compile_attempts"] == 0, counters
        assert counters["consumer_register"] == counters["consumer_unregister"], counters
        rows.append({"stage": stage, "calls": args.samples, "counts": counters, "checks": verify()})
    result = {"method": "untimed_diagnostic", "us_p50": None, "counts": rows}
    (args.output / "source-counts.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    # The legacy fixture divides the returned p50 values. This internal sentinel
    # propagates "not measured"; final report loading converts it to JSON null.
    return result | {"us_p50": float("nan")}


def install_measurement(case: Any, root: Any, args: Any) -> None:
    from hbg_eager_latency import npu  # noqa: PLC0415
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.runtime.kernel.arguments import _load_api  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    experiment = getattr(args, "_experiment", None)
    stages = experiment.stages if experiment is not None else ("current_a", "current_b")
    rng = random.Random(args.seed)
    calls = 0
    cold_compile = []
    original_compile = root._compile

    def time_compile(*call_args: Any, **kwargs: Any) -> Any:
        begin = time.perf_counter_ns()
        try:
            return original_compile(*call_args, **kwargs)
        finally:
            cold_compile.append({"wall_ms": (time.perf_counter_ns() - begin) / 1e6})
            (args.output / "cold-compile.json").write_text(json.dumps(cold_compile))

    root._compile = time_compile

    def measure(
        fixture: Any,
        run: Any,
        output: Any,
        topk: Any,
        reference: Any,
        *,
        iters: int,
        warmup: int,
        require_exact: bool,
        profile_dir: Any = None,
    ) -> dict[str, Any]:
        nonlocal calls
        assert (
            logging.getLogger("simpler").getEffectiveLevel()
            == importlib.import_module("simpler._log").get_current_config()
            == logging.ERROR
        )
        calls += 1
        assert calls in (1, 2), f"Unexpected timing callback {calls}"
        is_pto = calls == 2
        assert profile_dir is None, "Formal timing must not enable profiling"
        initial = {
            name: group["initial"].to(group["allocation"].device) for name, group in fixture["groups"].items()
        }
        state = get_process_kernel_state()
        artifact_ids = tuple(id(value) for value in root._cache.values())
        if is_pto:
            assert len(artifact_ids) == 1, f"Expected one compiled CSA artifact, got {artifact_ids}"
            for artifact in root._cache.values():
                artifact.execution_capabilities.require(ArtifactExecutionMode.KERNEL)

            def reject_compile(*unused_args: Any, **unused_kwargs: Any) -> Any:
                raise AssertionError("CSA recompiled during a fixed-artifact measurement")

            assert root._compile is time_compile
            root._compile = reject_compile

        def reset() -> None:
            for name, group in fixture["groups"].items():
                group["allocation"].copy_(initial[name])
            output.fill_(float("nan"))
            npu.synchronize()

        def drain() -> None:
            npu.synchronize()
            if is_pto:
                state.drain()
                session = state._worker
                worker = session.worker
                # Complete lazy registry revocations as well as source leases.
                with worker._child_prov_worker_lock(0), worker._child_prov_lock:
                    assert worker._attach_source_buffers_locked(_load_api().TaskArgs()) == ()
                    assert not worker._source_attachments
                    assert not worker._device_buffer_source._records
                assert not session.sources._sources

        expected_hashes = None
        rows, orders, checks = [], [], []
        selected = stages if is_pto else ("native",)

        def verify() -> dict[str, Any]:
            checked = check_state(case, fixture, output, topk, reference, require_exact)
            if expected_hashes is not None:
                assert checked["hashes"] == expected_hashes, "CSA lifecycle state differs from Host pass"
            return checked

        try:
            if is_pto and args.mode == "diagnostic":
                return diagnose_sources(args, stages, root, reset, run, drain, verify)
            for batch in range(args.blocks if is_pto else 1):
                order = list(selected)
                rng.shuffle(order)
                orders.append(order)
                for stage in order:
                    drain()
                    if is_pto and experiment is not None:
                        experiment.install(stage)
                    for _ in range(warmup):
                        reset()
                        run()
                        npu.synchronize()
                    for sample in range(iters):
                        reset()
                        begin = time.perf_counter_ns()
                        run()
                        end = time.perf_counter_ns()
                        npu.synchronize()
                        rows.append(
                            {
                                "batch": batch,
                                "stage": stage,
                                "sample": sample,
                                "host_us": (end - begin) / 1000,
                            }
                        )
                    drain()
                    checked = check_state(case, fixture, output, topk, reference, require_exact)
                    if expected_hashes is not None:
                        assert checked["hashes"] == expected_hashes, "CSA state differs across blocks or arms"
                    expected_hashes = checked["hashes"]
                    assert not is_pto or tuple(id(value) for value in root._cache.values()) == artifact_ids
                    checks.append({"batch": batch, "stage": stage, **checked})
                    print("CSA_HOST_PASS", batch, stage, flush=True)
                    (args.output / ("pto-host.json" if is_pto else "native-host.json")).write_text(
                        json.dumps(rows)
                    )
            if is_pto:
                measure_cpu(args, stages, reset, run, drain, verify)
        finally:
            try:
                drain()
            finally:
                if is_pto:
                    del root._compile
                    if experiment is not None:
                        experiment.restore()
        result = {
            "method": "eager_host_entry_to_return",
            "scope": (
                "Reset/synchronization excluded from Host; lifecycle includes completion and source release"
            ),
            "orders": orders,
            "checks": checks,
            "samples_us": [row["host_us"] for row in rows],
            "us_p50": statistics.median(
                row["host_us"] for row in rows if row["stage"] in ("current_a", "current_b", "native")
            ),
        }
        (args.output / ("pto-checks.json" if is_pto else "native-checks.json")).write_text(
            json.dumps(result, indent=2)
        )
        return result

    case.measure_graph_interval = measure


def install_paired_measurement(case: Any, root: Any, args: Any) -> None:
    """Retain both fixture callbacks, then alternate complete Native/PTO arms.

    The fixture normally restores its Native Top-K hook before measuring PTO.
    Retain that hook explicitly and install it outside each Native timed arm.
    Numerical equivalence is checked against each implementation's own reference;
    known Native/PTO zero-tolerance differences are not silently accepted as equal.
    """
    torch = importlib.import_module("torch")
    npu = importlib.import_module("torch_npu").npu
    get_process_kernel_state = importlib.import_module(
        "pypto.runtime.kernel.context"
    ).get_process_kernel_state

    contexts: dict[str, Any] = {}
    native_result: dict[str, Any] = {"us_p50": float("nan")}

    def capture(fixture: Any, run: Any, output: Any, topk: Any, reference: Any, **kwargs: Any) -> Any:
        name = "native" if not contexts else "pto"
        assert name not in contexts
        contexts[name] = (run, output, topk, reference, kwargs["require_exact"])
        if name == "native":
            contexts["native_hook"] = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
            return native_result
        state = get_process_kernel_state()
        assert len(root._cache) == 1
        artifact_ids = tuple(id(value) for value in root._cache.values())
        initial = {
            key: group["initial"].to(group["allocation"].device) for key, group in fixture["groups"].items()
        }
        rows: dict[str, list[Any]] = {"native": [], "pto": []}
        checks: dict[str, list[Any]] = {"native": [], "pto": []}
        original_hook = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer

        def reject_compile(*unused: Any, **unused_kwargs: Any) -> Any:
            raise AssertionError("Paired CSA recompiled after warmup")

        def reset(output: Any) -> None:
            for key, group in fixture["groups"].items():
                group["allocation"].copy_(initial[key])
            output.fill_(float("nan"))
            npu.synchronize()

        def run_arm(arm: str, batch: int) -> None:
            state.drain()
            run, output, topk, reference, exact = contexts[arm]
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = (
                contexts["native_hook"] if arm == "native" else original_hook
            )
            for sample in range(-args.warmup, args.samples):
                reset(output)
                begin = time.perf_counter_ns()
                run()
                elapsed = (time.perf_counter_ns() - begin) / 1000
                npu.synchronize()
                if sample >= 0:
                    rows[arm].append({"batch": batch, "stage": arm, "sample": sample, "host_us": elapsed})
            state.drain()
            checked = check_state(case, fixture, output, topk, reference, exact)
            if checks[arm]:
                assert checked["hashes"] == checks[arm][0]["hashes"]
            checks[arm].append(checked)
            assert tuple(id(value) for value in root._cache.values()) == artifact_ids
            for label, values in rows.items():
                (args.output / f"paired-{label}-host.json").write_text(json.dumps(values))
            (args.output / "paired-checks.json").write_text(json.dumps(checks))
            print("CSA_PAIRED_PASS", batch, arm, flush=True)

        root._compile = reject_compile
        try:
            # produce builds a new adapter per call, but specialization must hit.
            # The fixture has compiled it already; this warmup must also hit.
            reset(contexts["pto"][1])
            contexts["pto"][0]()
            npu.synchronize()
            state.drain()
            if args.mode == "paired":
                for batch in range(args.blocks):
                    order = ("native", "pto") if (batch + args.seed) % 2 else ("pto", "native")
                    for arm in order:
                        run_arm(arm, batch)
            else:
                Spans = importlib.import_module("hbg_pypto_spans").Spans
                TorchDispatchMode = importlib.import_module("torch.utils._python_dispatch").TorchDispatchMode

                counts: dict[str, int] = {}

                class CountOps(TorchDispatchMode):
                    def __torch_dispatch__(
                        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
                    ) -> Any:
                        key = str(func)
                        counts[key] = counts.get(key, 0) + 1
                        return func(*args, **(kwargs or {}))

                # Counts and spans are diagnostic only, never Host performance evidence.
                reset(contexts["native"][1])
                torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = contexts["native_hook"]
                with CountOps():
                    contexts["native"][0]()
                npu.synchronize()
                (args.output / "native-op-counts.json").write_text(json.dumps(counts, indent=2))
                torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_hook
                spans = Spans()
                spans.install(SimpleNamespace(session=state._worker, worker=state._worker.worker))
                try:
                    for sample in range(args.samples):
                        reset(contexts["pto"][1])
                        spans.sample = sample
                        contexts["pto"][0]()
                        npu.synchronize()
                        spans.sample = None
                    state.drain()
                finally:
                    spans.restore()
                (args.output / "pto-spans.json").write_text(json.dumps(spans.events))
                for arm in ("native", "pto"):
                    run, output, topk, reference, exact = contexts[arm]
                    reset(output)
                    torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = (
                        contexts["native_hook"] if arm == "native" else original_hook
                    )
                    run()
                    npu.synchronize()
                    checks[arm].append(check_state(case, fixture, output, topk, reference, exact))
                (args.output / "paired-checks.json").write_text(json.dumps(checks))
        finally:
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_hook
            del root._compile
            npu.synchronize()
            state.drain()
        result = {"method": "paired_eager_host", "metadata": "produce", "us_p50": float("nan")}
        if args.mode == "paired":
            native_result.update(result, us_p50=statistics.median(row["host_us"] for row in rows["native"]))
            result["us_p50"] = statistics.median(row["host_us"] for row in rows["pto"])
        else:
            native_result.update(method="untimed_diagnostic", us_p50=float("nan"))
        return result

    case.measure_graph_interval = capture


def write_identity(args: Any) -> None:
    modules = {}
    for name in (
        "pypto",
        "pypto.pypto_core",
        "pypto._torch_npu",
        "simpler",
        "_task_interface",
        "torch",
        "torch_npu",
        "vllm",
        "vllm_ascend",
    ):
        module = importlib.import_module(name)
        assert module.__file__ is not None, f"Module {name} has no source path"
        path = Path(module.__file__).resolve()
        modules[name] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    repo = Path(__file__).resolve().parents[2]
    source_repo = Path(os.environ.get("CSA_VERSION_ROOT", str(repo))).resolve()
    assert Path(modules["pypto"]["file"]) == source_repo / "python/pypto/__init__.py"
    assert Path(modules["simpler"]["file"]).is_relative_to(repo / "runtime")
    identity = {
        "modules": modules,
        "pid": os.getpid(),
        "command_line": sys.argv,
        "simpler_log_level": logging.getLogger("simpler").getEffectiveLevel(),
        "measurement": {
            name: getattr(args, name)
            for name in (
                "mode",
                "suite",
                "seed",
                "blocks",
                "samples",
                "warmup",
                "cpu_samples",
                "history",
                "metadata",
            )
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TASK_QUEUE_ENABLE",
                "PTOAS_ROOT",
                "ASCEND_HOME_PATH",
                "PYTHONNOUSERSITE",
                "PYPTO_CACHE",
                "ASCEND_RT_VISIBLE_DEVICES",
            )
        },
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=("current",), default="current")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("smoke", "bench", "diagnostic", "paired", "paired-diagnostic"), default="smoke"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--history", type=int, default=8192)
    parser.add_argument("--metadata", choices=("reuse", "produce"), default="reuse")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if min(args.blocks, args.samples, args.warmup, args.cpu_samples) < 1:
        parser.error("All measurement counts must be positive")
    if args.mode == "smoke":
        args.blocks, args.samples, args.warmup, args.cpu_samples = 1, 2, 1, 1
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_CACHE"] = "0"
    os.environ.update(
        PTO_CSA_VARIANT="performance",
        VLLM_ASCEND_ENABLE_NZ="2",
        VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0",
        HCCL_DETERMINISTIC="true",
    )
    case, root = bind_csa(args.source.resolve(), args.native.resolve())
    if args.mode.startswith("paired"):
        install_paired_measurement(case, root, args)
    else:
        install_measurement(case, root, args)
    # Worker.init snapshots this threshold into native logging. Configure it
    # before the fixture creates the Worker, not merely before timed calls.
    importlib.import_module("simpler_setup.log_config").configure_logging("error")
    write_identity(args)
    sys.argv = [
        str(args.source / "tests/pypto_test/dsv4_csa_single_layer.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--device",
        str(args.device),
        "--layer-index",
        "2",
        "--batch",
        "4",
        "--history",
        str(args.history),
        "--runtime",
        "host_build_graph",
        "--variant",
        "performance",
        "--weight-nz-mode",
        "2",
        "--atomic-add",
        "0",
        "--deterministic-level",
        "1",
        "--timing-iters",
        str(args.samples),
        "--timing-warmup",
        str(args.warmup),
        "--timing-metadata",
        "produce" if args.mode.startswith("paired") else args.metadata,
    ]
    try:
        case.main()
    finally:
        from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

        get_process_kernel_state().close()
    path = args.output / "report.json"
    report = json.loads(path.read_text(), parse_constant=lambda value: None)
    report["execution_config"]["path"] = "kernel_eager"
    if args.mode in ("diagnostic", "paired-diagnostic"):
        report["timing"].pop("native_over_pto_p50", None)
        report["timing"]["pto"].pop("us_p50", None)
        report["timing"]["native"].pop("us_p50", None)
    report["timing"].update(scope="CSA KERNEL eager Host timing", method="No graph capture or replay")
    path.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
