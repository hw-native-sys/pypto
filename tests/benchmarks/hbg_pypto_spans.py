# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Attribute warm eager calls with coarse Python wall spans, including queue callbacks.

Example: TASK_QUEUE_ENABLE=0 python tests/benchmarks/hbg_pypto_spans.py
--device 14 --output /tmp/hbg-spans-q0
Use a fresh process for each queue setting. Inclusive parent/child spans overlap
and MUST NOT be added. Background spans can occur after the Host return; their
sample attribution extends through the following synchronization. Unwrapped
before/after runs quantify instrumentation inflation, not optimization benefit.
"""

import argparse
import ast
import importlib
import inspect
import itertools
import json
import os
import statistics
import textwrap
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, cast

from hbg_eager_latency import FixedProbe, configure_logging, npu, pypto, torch, torch_npu
from pypto.jit import _arguments, _call_binding, decorator
from pypto.runtime.kernel.arguments import BorrowedArguments
from pypto.runtime.kernel.pending import PendingTicket
from pypto.torch import interop, launch


class Spans:
    """Record only measured samples; restore every patched attribute on exit."""

    def __init__(self) -> None:
        self.caller = threading.get_ident()
        self.sample: int | None = None
        self.events: list[dict[str, Any]] = []
        self._local = threading.local()
        self._ids = itertools.count()
        self._restore: list[tuple[Any, str, Any, bool]] = []

    def patch(self, owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)
        own_attribute = name in vars(owner)

        @wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            sample = self.sample
            if sample is None:
                return original(*args, **kwargs)
            stack = getattr(self._local, "stack", None)
            if stack is None:
                stack = self._local.stack = []
            event_id = next(self._ids)
            parent = stack[-1] if stack else None
            stack.append(event_id)
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                end = time.perf_counter_ns()
                stack.pop()
                thread = threading.get_ident()
                self.events.append(
                    {
                        "sample": sample,
                        "id": event_id,
                        "parent": parent,
                        "span": label,
                        "thread_id": thread,
                        "thread": "caller" if thread == self.caller else "background",
                        "start_ns": start,
                        "end_ns": end,
                        "wall_us": (end - start) / 1000,
                    }
                )

        self._restore.append((owner, name, original, own_attribute))
        setattr(owner, name, wrapped)

    def install_sections(self) -> None:
        """Timestamp coarse resolver sections without nested helper wrappers."""
        original = decorator.JITFunction._resolve_compiled
        node = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
        assert isinstance(node, ast.FunctionDef)
        checkpoints = {
            "specialization, run_config": "binding",
            "compile_kwargs, bypass_cache": "policy",
            "cache_config": "cache_policy",
            "ordered_args": "dispatch_setup",
            "constexpr_bindings": "constexpr",
            "key": "key_inputs",
        }
        body = []
        labels = []
        for statement in node.body:
            label = None
            if isinstance(statement, ast.Assign):
                label = checkpoints.get(ast.unparse(statement.targets[0]).strip("()"))
            elif isinstance(statement, ast.If) and ast.unparse(statement.test) == "_contract is not None":
                label = "contract"
            elif isinstance(statement, ast.With):
                label = "lookup"
            elif isinstance(statement, ast.Return):
                label = "returned"
            if label is not None:
                labels.append(label)
                body.append(ast.parse(f"_hbg_resolver_mark({label!r})").body[0])
            body.append(statement)
        assert labels == [*checkpoints.values(), "contract", "lookup", "returned"], labels
        node.body = body
        namespace = cast(dict[str, Any], vars(decorator))
        name = "_hbg_resolver_mark"
        assert name not in namespace
        self._restore.append((decorator, name, None, False))

        def mark(label: str) -> None:
            if self.sample is not None:
                self.events.append(
                    {"sample": self.sample, "section": label, "time_ns": time.perf_counter_ns()}
                )

        setattr(decorator, name, mark)
        local: dict[str, Any] = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                "<hbg-resolver-sections>",
                "exec",
            ),
            namespace,
            local,
        )
        self._restore.append((decorator.JITFunction, "_resolve_compiled", original, True))
        decorator.JITFunction._resolve_compiled = local[node.name]

    def install(self, probe: FixedProbe, resolver_only: bool = False) -> None:
        for name in (
            "_resolve_compiled",
            "_resolve_specialization",
            "_bind_args",
            "_get_source_hash",
            "_resolve_constexpr_bindings",
            "_dep_declared_layouts",
        ):
            self.patch(decorator.JITFunction, name, f"jit.{name}")
        for name in ("_resolve_compile_request", "_request_source_hash", "make_cache_key"):
            self.patch(decorator, name, f"jit.{name}")
        if resolver_only:
            for name in (
                "_param_names",
                "_get_dep_graph",
                "_get_dep_graph_state",
                "_constexpr_identity_records",
            ):
                self.patch(decorator.JITFunction, name, f"jit.{name}")
            for name in (
                "capture_cache_config",
                "_platform_string",
                "_resolve_runtime",
                "_eager_platform",
                "_parse_platform",
                "_eager_runtime",
                "record_stats",
                "_discover_dep_bindings",
            ):
                self.patch(decorator, name, f"jit.{name}")
            for name in ("param_layouts", "param_dtypes", "constexpr_params", "extract_tensor_meta"):
                self.patch(_arguments, name, f"binding.{name}")
            self.patch(_call_binding, "compute_per_func_dyndim_maps", "binding.dynamic_maps")
            return
        self.patch(interop.CallSignature, "describe_call", "frame.describe_call")
        self.patch(interop, "_load_torch_npu", "interop._load_torch_npu")
        for name in ("check_eager_call", "check_frame", "_load_native"):
            self.patch(launch, name, f"launch.{name}")
        for name in ("__init__", "release"):
            self.patch(BorrowedArguments, name, f"arguments.{name}")
        self.patch(PendingTicket, "done", "ticket.done")
        self.patch(probe.worker, "submit", "worker.submit")
        self.patch(probe.session, "check_context", "session.check_context")

    def restore(self) -> None:
        self.sample = None
        for owner, name, original, own_attribute in reversed(self._restore):
            if own_attribute:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self._restore.clear()


def measure(probe: FixedProbe, samples: int, spans: Spans | None = None) -> list[dict[str, Any]]:
    """Use identical Host clocks with and without wrappers; synchronize outside."""
    probe.prepare("jit")
    probe.x.add_(1)
    probe.expected.add_(1)
    probe.out.fill_(float("nan"))
    probe.aclnn_out.copy_(probe.x + probe.y)
    for _ in range(50):
        assert probe.call() is probe.out
        npu.synchronize()
    probe.check()
    records = []
    for sample in range(samples):
        npu.synchronize()
        if spans is not None:
            spans.sample = sample
        start = time.perf_counter_ns()
        try:
            result = probe.call()
            returned = time.perf_counter_ns()
            npu.synchronize()
            completed = time.perf_counter_ns()
        finally:
            if spans is not None:
                spans.sample = None
        assert result is probe.out
        records.append(
            {
                "sample": sample,
                "start_ns": start,
                "returned_ns": returned,
                "completed_ns": completed,
                "host_us": (returned - start) / 1000,
            }
        )
    probe.check()
    probe.finish()
    return records


def summarize(events: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    """Sum repeated instances within each sample, retaining zero-call samples."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        groups.setdefault(f"{event['thread']}:{event['span']}", []).append(event)
    result = {}
    for label, entries in groups.items():
        totals = [0.0] * samples
        counts = [0] * samples
        for event in entries:
            totals[event["sample"]] += event["wall_us"]
            counts[event["sample"]] += 1
        result[label] = {
            "inclusive_us_per_call": totals,
            "instances_per_call": counts,
            "median_inclusive_us": statistics.median(totals),
            "mean_inclusive_us": statistics.mean(totals),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--resolver-only", action="store_true")
    parser.add_argument("--resolver-sections", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples must be positive")
    if os.environ.get("LD_PRELOAD") or os.environ.get("PYPTO_LAUNCH_PROBE_OUTPUT"):
        parser.error("run without native launch probes or LD_PRELOAD")
    args.output.mkdir(parents=True, exist_ok=True)
    configure_logging("error")
    torch.set_num_threads(1)
    npu.set_device(args.device)
    npu.set_compile_mode(jit_compile=False)
    probe = FixedProbe(args.device)
    spans = Spans()
    try:
        before = measure(probe, args.samples)
        try:
            if args.resolver_sections:
                spans.install_sections()
            else:
                spans.install(probe, args.resolver_only)
            instrumented = measure(probe, args.samples, spans)
        finally:
            spans.restore()
        after = measure(probe, args.samples)
        medians = {
            label: statistics.median(row["host_us"] for row in rows)
            for label, rows in (("before", before), ("instrumented", instrumented), ("after", after))
        }
        summary = {
            "host_median_us": medians,
            "inflation_us": {
                label: medians["instrumented"] - medians[label] for label in ("before", "after")
            },
            "inflation_ratio": {
                label: medians["instrumented"] / medians[label] for label in ("before", "after")
            },
            "inclusive_spans": {} if args.resolver_sections else summarize(spans.events, args.samples),
            "warning": "Inclusive spans overlap; never add parent/child or caller/background totals to Host.",
        }
        identity = {
            "pypto": pypto.__file__,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "simpler": importlib.import_module("simpler").__file__,
            "device": args.device,
            "queue": os.environ.get("TASK_QUEUE_ENABLE"),
            "samples": args.samples,
            "resolver_only": args.resolver_only,
            "resolver_sections": args.resolver_sections,
            "caller_thread_id": spans.caller,
            "contract": "fixed retained storage and stream; fresh full JIT calls; single caller",
        }
        for name, payload in (
            ("identity", identity),
            ("host", {"before": before, "instrumented": instrumented, "after": after}),
            ("spans", spans.events),
            ("summary", summary),
        ):
            (args.output / f"{name}.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps({"host_median_us": medians, "inflation_us": summary["inflation_us"]}), flush=True)
    finally:
        spans.restore()
        probe.finish()
        probe.state.close()


if __name__ == "__main__":
    main()
