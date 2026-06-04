# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen regression for real pypto-lib window-dependency cases.

The fixtures in this directory are pinned snapshots of DeepSeek V4 indexer and
Qwen3 decode-layer programs.  The test compiles each entry and checks the
generated orchestration C++ so regressions in TensorMap-facing task args are
caught before running hardware swimlane / dep-gen validation.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
import types
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from pypto.runtime.runner import RunConfig

_CASE_DIR = Path(__file__).resolve().parent
_BUILD_ROOT = Path(__file__).resolve().parents[3] / "build_output" / "window_scope_real_cases"
_GOLDEN_MODULES = ("golden",)
_CASE_MODULES = (
    "config",
    "decode_indexer_compressor",
    "indexer_compressor",
    "rms_lm_head",
)


@dataclass(frozen=True)
class _TensorSpec:
    name: str
    shape: list[int]
    dtype: torch.dtype
    init_value: Any = None
    is_output: bool = False


@dataclass(frozen=True)
class _ScalarSpec:
    name: str
    dtype: torch.dtype
    value: Any


@dataclass(frozen=True)
class _Case:
    id: str
    path: Path
    support_dir: Path
    entry_name: str
    specs_builder: Callable[[Any], list[Any]]
    expectations: tuple[_Expectation, ...]


@dataclass(frozen=True)
class _Expectation:
    name: str
    check: Callable[[str], bool]
    message: str


def _deepseek_specs(module: Any) -> list[Any]:
    return module.build_tensor_specs()


def _qwen_specs(module: Any) -> list[Any]:
    return module.build_tensor_specs(batch=module.BATCH, use_max_seq=True)


def _full_task_arg_pattern(direction: str, root_pattern: str) -> re.Pattern[str]:
    return re.compile(rf"\bparams_t\d+\.add_{direction}\(\s*{root_pattern}\s*\)")


def _has_windowed_task_arg(code: str, direction: str, root_pattern: str) -> bool:
    window_vars = set(re.findall(rf"\b(auto\s+)?({root_pattern}\w*)\s*=\s*{root_pattern}\w*\.view\(", code))
    names = {name for _, name in window_vars}
    return any(
        re.search(rf"\bparams_t\d+\.add_{direction}\(\s*{re.escape(name)}\s*\)", code) for name in names
    )


def _no_full_arg(direction: str, root_pattern: str, label: str) -> _Expectation:
    return _Expectation(
        name=f"no_full_{direction}_{label}",
        check=lambda code: _full_task_arg_pattern(direction, root_pattern).search(code) is None,
        message=f"found full-tensor params.add_{direction}({label}); expected a window view",
    )


def _has_window_arg(direction: str, root_pattern: str, label: str) -> _Expectation:
    return _Expectation(
        name=f"has_window_{direction}_{label}",
        check=lambda code: _has_windowed_task_arg(code, direction, root_pattern),
        message=f"no params.add_{direction}(...) used a {label}*.view(...) window",
    )


def _contains(text: str, label: str) -> _Expectation:
    return _Expectation(
        name=f"contains_{label}",
        check=lambda code: text in code,
        message=f"missing expected generated code marker: {text}",
    )


def _not_contains(text: str, label: str) -> _Expectation:
    return _Expectation(
        name=f"not_contains_{label}",
        check=lambda code: text not in code,
        message=f"found unexpected generated code marker: {text}",
    )


_DEEPSEEK_SPMD_SHA = "1d1ced8520a77b6902183b2d7f2a612201799d8a"
_DEEPSEEK_LEGACY_SHA = "8ee6911f5f2c8737489a8d0858d17e6945ff948f"
_QWEN_SPMD_SHA = _DEEPSEEK_SPMD_SHA
_QWEN_LEGACY_SHA = "c45855eb7493d6dbd790d5cb33bc22fba23b98c5"

_CASES = (
    _Case(
        id=f"deepseek_v4_decode_indexer_spmd_{_DEEPSEEK_SPMD_SHA}",
        path=_CASE_DIR / f"deepseek_v4_decode_indexer_spmd_{_DEEPSEEK_SPMD_SHA}.py",
        support_dir=_CASE_DIR / "support" / "deepseek_v4" / _DEEPSEEK_SPMD_SHA,
        entry_name="indexer_test",
        specs_builder=_deepseek_specs,
        expectations=(
            _not_contains("score_spmd__windowed", "spmd_score_not_rewritten"),
            _not_contains("topk_spmd__windowed", "spmd_topk_not_rewritten"),
        ),
    ),
    _Case(
        id=f"deepseek_v4_indexer_legacy_{_DEEPSEEK_LEGACY_SHA}",
        path=_CASE_DIR / f"deepseek_v4_indexer_legacy_{_DEEPSEEK_LEGACY_SHA}.py",
        support_dir=_CASE_DIR / "support" / "deepseek_v4" / _DEEPSEEK_LEGACY_SHA,
        entry_name="indexer_test",
        specs_builder=_deepseek_specs,
        expectations=(
            _no_full_arg("input", r"score_flat_inline\d+(?:__rv_v\d+)?", "score_flat"),
            _has_window_arg("output", r"score_flat_inline\d+", "score_flat"),
            _has_window_arg("input", r"score_flat_inline\d+", "score_flat"),
        ),
    ),
    _Case(
        id=f"qwen3_14b_decode_layer_spmd_{_QWEN_SPMD_SHA}",
        path=_CASE_DIR / f"qwen3_14b_decode_layer_spmd_{_QWEN_SPMD_SHA}.py",
        support_dir=_CASE_DIR / "support" / "qwen3_14b" / _QWEN_SPMD_SHA,
        entry_name="test_decode_layer_no_lm_head",
        specs_builder=_qwen_specs,
        expectations=(),
    ),
    _Case(
        id=f"qwen3_14b_decode_layer_legacy_{_QWEN_LEGACY_SHA}",
        path=_CASE_DIR / f"qwen3_14b_decode_layer_legacy_{_QWEN_LEGACY_SHA}.py",
        support_dir=_CASE_DIR / "support" / "qwen3_14b" / _QWEN_LEGACY_SHA,
        entry_name="test_decode_layer_no_lm_head",
        specs_builder=_qwen_specs,
        expectations=(
            _no_full_arg("output", r"q_proj_inline\d+(?:__rv_v\d+)?", "q_proj"),
            _has_window_arg("output", r"q_proj_inline\d+", "q_proj"),
            _no_full_arg("output", r"k_proj_inline\d+(?:__rv_v\d+)?", "k_proj"),
            _has_window_arg("output", r"k_proj_inline\d+", "k_proj"),
            _no_full_arg("input", r"q_proj_inline\d+(?:__rv_v\d+)?", "q_proj"),
            _has_window_arg("input", r"q_proj_inline\d+", "q_proj"),
            _no_full_arg("input", r"k_proj_inline\d+(?:__rv_v\d+)?", "k_proj"),
            _has_window_arg("input", r"k_proj_inline\d+", "k_proj"),
        ),
    ),
)


def _install_minimal_golden() -> None:
    golden = types.ModuleType("golden")
    golden.TensorSpec = _TensorSpec
    golden.ScalarSpec = _ScalarSpec
    sys.modules["golden"] = golden


@contextmanager
def _isolated_import_path(paths: Iterable[Path]):
    saved_modules = {name: sys.modules.get(name) for name in (*_GOLDEN_MODULES, *_CASE_MODULES)}
    old_path = list(sys.path)
    try:
        for name in (*_GOLDEN_MODULES, *_CASE_MODULES):
            sys.modules.pop(name, None)
        _install_minimal_golden()
        for path in reversed([str(path) for path in paths]):
            sys.path.insert(0, path)
        yield
    finally:
        sys.path[:] = old_path
        for name in (*_GOLDEN_MODULES, *_CASE_MODULES):
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def _load_case_module(case: _Case) -> Any:
    module_name = f"_window_scope_real_case_{case.id}"
    spec = importlib.util.spec_from_file_location(module_name, case.path)
    assert spec is not None and spec.loader is not None, f"cannot load {case.path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _materialize_arg(spec: Any) -> Any:
    if isinstance(spec, _ScalarSpec):
        return spec.value
    init_value = getattr(spec, "init_value", None)
    shape = list(getattr(spec, "shape"))
    dtype = getattr(spec, "dtype")
    if init_value is None:
        return torch.zeros(shape, dtype=dtype)
    if isinstance(init_value, (int, float, bool)):
        return torch.full(shape, init_value, dtype=dtype)
    if isinstance(init_value, torch.Tensor):
        return init_value.to(dtype=dtype)
    if callable(init_value):
        value = init_value()
        return torch.as_tensor(value, dtype=dtype)
    raise TypeError(f"unsupported init_value {type(init_value)!r} for {getattr(spec, 'name', '<unnamed>')}")


def _orchestration_cpp(output_dir: Path) -> Path:
    candidates = sorted((output_dir / "orchestration").glob("*.cpp"))
    assert candidates, f"no orchestration C++ generated under {output_dir}"
    assert len(candidates) == 1, f"expected one orchestration C++, got {[str(path) for path in candidates]}"
    return candidates[0]


def _compile_case(case: _Case, platform: str, test_config: RunConfig) -> tuple[Path, str]:
    with _isolated_import_path((case.support_dir, _CASE_DIR)):
        module = _load_case_module(case)
        specs = case.specs_builder(module)
    entry = getattr(module, case.entry_name)
    args = [_materialize_arg(spec) for spec in specs]
    output_dir = _BUILD_ROOT / case.id
    shutil.rmtree(output_dir, ignore_errors=True)
    cfg = RunConfig(
        platform=platform,
        save_kernels=True,
        save_kernels_dir=str(output_dir),
        codegen_only=True,
        dump_passes=test_config.dump_passes,
        pto_isa_commit=test_config.pto_isa_commit,
    )
    entry._cache.clear()
    entry.compile(*args, config=cfg)
    cpp_path = _orchestration_cpp(output_dir)
    return cpp_path, cpp_path.read_text()


@pytest.mark.parametrize("case", _CASES, ids=[case.id for case in _CASES])
def test_real_model_window_dependencies_codegen(case: _Case, test_config: RunConfig):
    cpp_path, code = _compile_case(case, test_config.platform, test_config)
    failures = [expectation for expectation in case.expectations if not expectation.check(code)]
    assert not failures, f"{case.id} generated imprecise orchestration args in {cpp_path}:\n" + "\n".join(
        f"- {failure.name}: {failure.message}" for failure in failures
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
