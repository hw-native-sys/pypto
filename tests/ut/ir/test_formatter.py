# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The IR printer's format callback: memoization and the raw-print escape hatch."""

import threading
import time
from typing import Any

import pytest
from pypto.ir import formatter
from pypto.pypto_core import DataType, ir


class _FakeResult:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


@pytest.fixture()
def spawn_counter(monkeypatch: pytest.MonkeyPatch):
    """Count ruff subprocess spawns and fake their output (echo the stdin)."""
    calls: list[Any] = []

    def fake_run(*args: Any, **kwargs: Any) -> _FakeResult:
        calls.append(args)
        return _FakeResult(stdout=kwargs["input"] + "\n")

    monkeypatch.setattr(formatter.subprocess, "run", fake_run)
    formatter._FORMAT_CACHE.clear()
    return calls


def test_repeated_input_spawns_once(spawn_counter: list[Any]) -> None:
    """The printer emits the same type strings repeatedly; each distinct
    (code, line_length) must spawn ruff at most once."""
    assert formatter.ruff_format("code_a") == "code_a"
    assert formatter.ruff_format("code_a") == "code_a"
    assert formatter.ruff_format("code_a") == "code_a"
    assert len(spawn_counter) == 1


def test_line_length_splits_cache_entries(spawn_counter: list[Any]) -> None:
    formatter.ruff_format("code_a", line_length=200)
    formatter.ruff_format("code_a", line_length=100)
    assert len(spawn_counter) == 2


def test_failure_output_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed ruff run returns the input unchanged and must not poison the
    cache: the next call retries."""
    calls: list[Any] = []

    def failing_run(*args: Any, **kwargs: Any) -> _FakeResult:
        calls.append(args)
        return _FakeResult(stdout="", returncode=1)

    monkeypatch.setattr(formatter.subprocess, "run", failing_run)
    formatter._FORMAT_CACHE.clear()
    assert formatter.ruff_format("code_b") == "code_b"
    assert formatter.ruff_format("code_b") == "code_b"
    assert len(calls) == 2


def test_concurrent_same_key_spawns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent callers racing the same (code, line_length) must not both
    spawn: both miss the cache, both release the GIL inside the subprocess,
    and the in-flight map hands the key a single owner — the rest wait on
    the owner's result instead of racing it."""
    calls: list[Any] = []

    def slow_run(*args: Any, **kwargs: Any) -> _FakeResult:
        calls.append(args)
        time.sleep(0.05)  # hold the miss window open so the race actually happens
        return _FakeResult(stdout=kwargs["input"] + "\n")

    monkeypatch.setattr(formatter.subprocess, "run", slow_run)
    formatter._FORMAT_CACHE.clear()
    results: list[str] = []
    threads = [
        threading.Thread(target=lambda: results.append(formatter.ruff_format("code_c"))) for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["code_c"] * 8
    assert len(calls) == 1, "the ruff subprocess must run at most once per key"
    assert not formatter._IN_FLIGHT  # the owner's cleanup always runs


def test_formatting_callback_bypasses_ruff_under_raw_print(
    spawn_counter: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(formatter, "ruff_format", lambda *a, **k: pytest.fail("raw mode must not format"))
    with formatter.raw_print():
        assert formatter.formatting_callback("x = 1") == "x = 1"
    # and restores formatting on exit
    monkeypatch.setattr(formatter, "ruff_format", lambda code, **k: code + "!")
    assert formatter.formatting_callback("x = 1") == "x = 1!"


def test_str_ir_type_under_raw_print_spawns_nothing(spawn_counter: list[Any]) -> None:
    """The motivating path: keying code prints ``ir.Type`` values; under
    ``raw_print`` the C++ printer's callback returns its input verbatim, so
    no ruff process is spawned."""
    t = ir.TensorType([8, 2048], DataType.FP32)
    with formatter.raw_print():
        raw = str(t)
    assert raw  # non-empty: the raw printer still ran
    assert not spawn_counter


def test_raw_print_is_exception_safe() -> None:
    with pytest.raises(RuntimeError), formatter.raw_print():
        raise RuntimeError("boom")
    assert not formatter._RAW_PRINT.get()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
