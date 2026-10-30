# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Controlled CSA cost experiments; no PyPTO product implementation changes.

Use CSA_COST_AUDIT=trace with --mode diagnostic, or =bench with --mode bench.
All arguments otherwise match hbg_csa_version.py, including pinned sources.
The experimental arms keep namespace reads, folding, guards and evaluation.
Callsite caching is a fixed-normal-stack experiment, not a promise to preserve
arbitrary resource-exhaustion behavior. Rendering-limit changes select a miss.
"""

import ast
import functools
import hashlib
import importlib
import inspect
import json
import os
import runpy
import sys
import textwrap
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def eligible(value: Any) -> bool:
    """Avoid subclasses, floats and digit-limit-sensitive large integers."""
    kind = type(value)
    return value is None or kind is bool or kind is str or (kind is int and value.bit_length() <= 64)


class Candidates:
    """Restore original functions at every arm boundary, never inside timing."""

    stages = ("current_a", "current_b", "immutable_text", "callsite_text", "combined")

    def __init__(self) -> None:
        self.specializer = importlib.import_module("pypto.jit.specializer")
        self.binding = importlib.import_module("pypto.jit._call_binding")
        self.decorator = importlib.import_module("pypto.jit.decorator")
        self.original_constant = self.specializer.constant_source
        self.saved: list[tuple[Any, str, Any, bool]] = []
        self.literal_text = functools.lru_cache(maxsize=512, typed=True)(self.original_constant)
        self.call_text = functools.lru_cache(maxsize=512)(self._call_text)
        node = ast.parse(textwrap.dedent(inspect.getsource(self.binding._fold_call_site_constant)))
        replaced = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "eval":
                assert ast.unparse(child.args[0]) == "ast.unparse(arg)"
                child.args[0] = ast.parse("_csa_cost_call_text(arg)", mode="eval").body
                replaced += 1
        assert replaced == 1, replaced
        local: dict[str, Any] = {}
        # Use live globals: the combined arm must see its own free-name reader.
        exec(
            compile(ast.fix_missing_locations(node), "<csa-cost-calltext>", "exec"), vars(self.binding), local
        )
        self.fold = local["_fold_call_site_constant"]

    @staticmethod
    def _call_text(arg: ast.expr, digit_limit: int, recursion_limit: int) -> str:
        # ASTs obey the parser's existing read-only contract. Include the global
        # integer-rendering limit so changing it cannot bypass an old exception.
        return ast.unparse(arg)

    def render(self, value: Any) -> str | None:
        return self.literal_text(value) if eligible(value) else self.original_constant(value)

    def free_name(self, name: str, namespace: Any) -> str | None:
        if name not in namespace:
            return None
        value = namespace[name]
        if value is self.specializer._UNBOUND:
            return None
        # Fallback consumes the already-read value; no second namespace lookup.
        return self.render(value)

    def call_source(self, arg: ast.expr) -> str:
        limit = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else 0
        return self.call_text(arg, limit, sys.getrecursionlimit())

    def set(self, owner: Any, name: str, value: Any) -> None:
        self.saved.append((owner, name, getattr(owner, name, None), name in vars(owner)))
        setattr(owner, name, value)

    def install(self, stage: str) -> None:
        self.restore()
        assert stage in self.stages, stage
        if stage in ("immutable_text", "combined"):
            for module in (self.specializer, self.binding, self.decorator):
                self.set(module, "free_name_source", self.free_name)
            self.set(self.specializer, "constant_source", self.render)
        if stage in ("callsite_text", "combined"):
            self.set(self.binding, "_csa_cost_call_text", self.call_source)
            self.set(self.binding, "_fold_call_site_constant", self.fold)

    def restore(self) -> None:
        for owner, name, value, own in reversed(self.saved):
            if own:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self.saved.clear()

    def info(self) -> dict[str, Any]:
        return {
            "literal_text": self.literal_text.cache_info()._asdict(),
            "callsite_text": self.call_text.cache_info()._asdict(),
        }


def trace_costs(
    original: Any, args: Any, stages: Any, root: Any, reset: Any, run: Any, drain: Any, verify: Any
) -> Any:
    """Instrument a separate process; never interpret these times as savings."""
    spans = importlib.import_module("hbg_pypto_spans").Spans()
    state = importlib.import_module("pypto.runtime.kernel.context").get_process_kernel_state()
    arguments = importlib.import_module("pypto.jit._arguments")
    binding = importlib.import_module("pypto.jit._call_binding")
    specializer = importlib.import_module("pypto.jit.specializer")
    interop = importlib.import_module("pypto.torch.interop")
    analysis = importlib.import_module("pypto.jit._ast_analysis")
    spans.install(SimpleNamespace(worker=state._worker.worker, session=state._worker))
    counters: Counter[str] = Counter()
    values: set[Any] = set()
    nodes: set[int] = set()
    before = analysis.body_local_names.cache_info()._asdict()
    for owner, names, prefix in (
        (
            arguments,
            (
                "param_layouts",
                "param_dtypes",
                "constexpr_params",
                "extract_tensor_meta",
                "_build_tensor_meta",
                "_annotation_namespace",
            ),
            "binding",
        ),
        (
            binding,
            (
                "compute_per_func_dyndim_maps",
                "expand_constexpr_variants",
                "resolve_call_site_constexpr",
                "_fold_call_site_constant",
            ),
            "constexpr",
        ),
        (specializer, ("tile_annotation_fingerprints", "_eager_tile_annotation_data"), "annotation"),
        (
            interop,
            ("_describe_tensor", "_validate_tensor", "_validate_aliases", "_native_storage_shape"),
            "frame",
        ),
        (state._worker.sources, ("acquire", "release", "_release_buffer"), "source"),
    ):
        for name in names:
            spans.patch(owner, name, f"{prefix}.{name}")

    def count_free(original_free: Any) -> Any:
        def call(name: str, namespace: Any) -> Any:
            if spans.sample is not None and name in namespace:
                value = namespace[name]
                counters[f"value_type:{type(value).__name__}"] += 1
                if eligible(value):
                    counters["eligible_free_name_reads"] += 1
                    values.add((type(value), value))
            return original_free(name, namespace)

        return call

    def count_fold(original_fold: Any) -> Any:
        def call(arg: Any, forwarded: Any, namespace: Any, locals_: Any) -> Any:
            if spans.sample is not None:
                counters[f"expression:{type(arg).__name__}"] += 1
                nodes.add(id(arg))
                if not isinstance(arg, ast.Name):
                    counters["namespace_entries_copied"] += len(namespace)
                    counters["forwarded_entries_parsed"] += len(forwarded)
            return original_fold(arg, forwarded, namespace, locals_)

        return call

    patches = Candidates()
    for module in (specializer, binding, patches.decorator):
        patches.set(module, "free_name_source", count_free(module.free_name_source))
    patches.set(binding, "_fold_call_site_constant", count_fold(binding._fold_call_site_constant))
    sample = 0

    def sampled_run() -> Any:
        nonlocal sample
        spans.sample = sample
        try:
            result = run()
            importlib.import_module("torch_npu").npu.synchronize()
            return result
        finally:
            spans.sample = None
            sample += 1

    try:
        result = original(args, stages, root, reset, sampled_run, drain, verify)
    finally:
        patches.restore()
        spans.restore()
    (args.output / "cost-spans.json").write_text(json.dumps(spans.events))
    (args.output / "cost-counts.json").write_text(
        json.dumps(
            {
                "calls": sample,
                "counts": counters,
                "unique_eligible_free_name_values": len(values),
                "unique_callsite_args": len(nodes),
                "body_local_before": before,
                "body_local_after": analysis.body_local_names.cache_info()._asdict(),
                "scope": "instrumented diagnostic only; per-leaf wrapper overhead is included",
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    mode = os.environ["CSA_COST_AUDIT"]
    assert mode in ("trace", "bench"), mode
    run_path = runpy.run_path
    candidates: list[Candidates] = []

    def execute(path: Any, *values: Any, **options: Any) -> Any:
        if Path(path).name != "hbg_csa_stability.py":
            return run_path(path, *values, **options)
        namespace = run_path(path, run_name="_csa_cost_target")
        original_install = namespace["install_measurement"]
        original_diagnose = namespace["diagnose_sources"]

        def install(case: Any, root: Any, args: Any) -> None:
            assert args.metadata == "produce", "Use the matched-work metadata scope"
            if mode == "bench":
                assert args.mode == "bench"
                candidate = Candidates()
                candidates.append(candidate)
                args._experiment = candidate
            else:
                assert args.mode == "diagnostic"
                namespace["main"].__globals__["diagnose_sources"] = lambda *args: trace_costs(
                    original_diagnose, *args
                )
            (args.output / "cost-audit-identity.json").write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "experiment_sha256": hashlib.sha256(
                            Path(inspect.getfile(Candidates)).read_bytes()
                        ).hexdigest(),
                        "stages": Candidates.stages if mode == "bench" else ("current_a", "current_b"),
                        "policy": "diagnostic only; preserve current reads/eval; fixed normal call stack",
                        "recursion_limit": sys.getrecursionlimit(),
                        "int_digit_limit": sys.get_int_max_str_digits()
                        if hasattr(sys, "get_int_max_str_digits")
                        else None,
                    },
                    indent=2,
                )
            )
            original_install(case, root, args)

        namespace["main"].__globals__["install_measurement"] = install
        return namespace["main"]()

    runpy.run_path = execute
    try:
        importlib.import_module("hbg_csa_version").main()
    finally:
        runpy.run_path = run_path
        for candidate in candidates:
            candidate.restore()
        if candidates:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            (output / "cost-cache-info.json").write_text(json.dumps(candidates[0].info(), indent=2))


if __name__ == "__main__":
    main()
