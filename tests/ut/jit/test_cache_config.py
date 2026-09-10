# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public cache policy precedence and immutable statistics snapshots."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pypto
import pytest
from pypto._cache_config import capture_cache_config, record_stats
from pypto.runtime import RunConfig


@pytest.fixture(autouse=True)
def clean_policy(monkeypatch):
    pypto.configure_cache(None)
    for name in ("PYPTO_CACHE", "PYPTO_CACHE_DIR", "PYPTO_CACHE_READONLY"):
        monkeypatch.delenv(name, raising=False)
    yield
    pypto.configure_cache(None)


def test_default_and_complete_precedence(tmp_path, monkeypatch):
    assert not capture_cache_config(None).enabled
    monkeypatch.setenv("PYPTO_CACHE", "1")
    monkeypatch.setenv("PYPTO_CACHE_DIR", str(tmp_path / "environment"))
    assert capture_cache_config(None).enabled
    process = pypto.CacheConfig(enabled=True, root=tmp_path / "process", readonly=True)
    pypto.configure_cache(process)
    assert capture_cache_config(None) == process
    local = capture_cache_config(pypto.CacheConfig())
    assert not local.enabled and not local.readonly
    assert local.root != process.root
    pypto.configure_cache(None)
    assert capture_cache_config(None).root == tmp_path / "environment"


@pytest.mark.parametrize("name", ["PYPTO_CACHE", "PYPTO_CACHE_READONLY"])
@pytest.mark.parametrize("value", ["", "true", "false", "2", " 1"])
def test_environment_booleans_are_strict(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        capture_cache_config(None)
    # Explicit policy replaces malformed lower-precedence environment too.
    assert not capture_cache_config(pypto.CacheConfig()).enabled


def test_config_capture_is_immutable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pypto.configure_cache(pypto.CacheConfig(enabled=True, root=Path("relative")))
    captured = capture_cache_config(None)
    pypto.configure_cache(pypto.CacheConfig())
    assert captured.enabled and captured.root == tmp_path / "relative"
    with pytest.raises(FrozenInstanceError):
        setattr(captured, "enabled", False)
    assert RunConfig(cache_config=captured).cache_config is captured
    assert "cache_config" not in RunConfig(cache_config=captured).compile_kwargs()


def test_statistics_are_thread_safe_snapshots():
    before = pypto.cache_stats()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: record_stats(requests=1), range(100)))
    after = pypto.cache_stats()
    assert after.requests - before.requests == 100
    assert pypto.cache_stats() == after
    with pytest.raises(FrozenInstanceError):
        setattr(after, "requests", 0)


@pytest.mark.parametrize(
    "values", [{"enabled": 1}, {"root": "cache"}, {"extra_source_paths": []}, {"extra_fingerprint": 3}]
)
def test_invalid_explicit_config(values):
    with pytest.raises(TypeError):
        pypto.CacheConfig(**values)


def test_disabled_policy_does_not_probe_filesystem(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled dispatch probed the filesystem")

    monkeypatch.setattr(Path, "resolve", forbidden)
    assert not capture_cache_config(None).enabled
    assert not capture_cache_config(pypto.CacheConfig(root=Path("unused"))).enabled


def test_cache_policy_does_not_change_compiler_or_launcher_options():
    ordinary = RunConfig()
    cached = RunConfig(cache_config=pypto.CacheConfig(enabled=True))
    assert cached.compile_options() == ordinary.compile_options()
    assert cached.run_options() == ordinary.run_options()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
