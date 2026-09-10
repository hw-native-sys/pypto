# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Warmup CLI validates metadata before builds and inventories without execution."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pypto.language as pl
import pytest
from pypto.jit.__main__ import main, stat, warm


@pytest.fixture
def cli_kernel(monkeypatch):
    @pl.jit
    def kernel(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.store(pl.load(x, [0, 0], [16, 16]), [0, 0], out)
        return out

    module = ModuleType("cache_cli_fixture")
    vars(module)["kernel"] = kernel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_dir="private")

    monkeypatch.setattr(kernel, "warmup", prepare)
    return module.__name__, calls


def _write(tmp_path, requests, **extra):
    path = tmp_path / "warmup.json"
    path.write_text(json.dumps({"schema_version": 1, "requests": requests, **extra}))
    return path


def test_metadata_only_warmup_and_relative_paths(tmp_path, cli_kernel):
    module, calls = cli_kernel
    path = _write(
        tmp_path,
        [{"kernel": "kernel", "tensors": {n: {"shape": [16, 16], "dtype": "FP32"} for n in ("x", "out")}}],
        cache={"enabled": True, "root": "cache", "extra_source_paths": ["sources"]},
    )
    result = warm(module, path)
    assert result["requests"][0]["storage"] == "private"
    assert len(calls) == 1
    assert calls[0]["x"].device.type == "meta"
    assert calls[0]["config"].cache_config.root == tmp_path / "cache"
    assert calls[0]["config"].cache_config.extra_source_paths == (tmp_path / "sources",)


@pytest.mark.parametrize(
    "invalid_request",
    [
        {"kernel": "missing"},
        {"kernel": "kernel", "scalars": {"unknown": 1}},
        {"kernel": "kernel", "tensors": {"x": {"shape": [-1], "dtype": "FP32"}}},
        {"kernel": "kernel", "tensors": {"x": {"shape": [16, 16], "dtype": "unknown"}}},
        {"kernel": "kernel", "tensors": {"x": {"shape": [32, 16], "dtype": "FP32"}}},
        {"kernel": "kernel", "run_config": {"typo": 1}},
    ],
)
def test_validate_all_requests_before_any_build(tmp_path, cli_kernel, invalid_request):
    module, calls = cli_kernel
    path = _write(tmp_path, [{"kernel": "kernel"}, invalid_request])
    with pytest.raises((ValueError, TypeError)):
        warm(module, path)
    assert calls == []


def test_cli_failures_have_nonzero_status(tmp_path, capsys):
    assert main(["warm", "--module", "missing", "--config", str(tmp_path / "absent")]) == 1
    assert "pypto.jit warm:" in capsys.readouterr().err


def test_stat_never_executes_payload_or_creates_root(tmp_path):
    root = tmp_path / "absent"
    assert stat(root)["entries"] == []
    assert not root.exists()
    slot = root / "artifacts/env/key/spec/ready"
    slot.mkdir(parents=True)
    (slot / "kernel_config.py").write_text("raise AssertionError('must not execute')")
    (slot / "artifact_manifest.json").write_text(
        json.dumps({"files": [{"path": "kernel_config.py", "size": 42}]})
    )
    assert stat(root)["payload_bytes"] == 42
    assert not list(root.rglob("__pycache__"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
