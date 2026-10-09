# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Consumed surface of ``simpler_setup.tools.strace_timing``."""

from collections.abc import Iterable, Iterator

class Span:
    pid: int
    tid: int
    inv: int
    hid: str
    depth: int
    name: str
    ts: int
    dur: int
    attrs: str

    @property
    def is_device(self) -> bool: ...

class Invocation:
    """All spans emitted by one simpler_run call (one (pid, inv) group)."""

    pid: int
    inv: int
    hid: str
    spans: list[Span]

    def root(self) -> Span | None: ...
    def by_name(self) -> dict[str, Span]: ...

_ROUNDS_TABLE_NAMES: dict[str, str]

def parse_spans(lines: Iterable[str]) -> Iterator[Span]: ...
def invocation_spans(spans: Iterable[Span]) -> list[Span]: ...
def group_invocations(spans: Iterable[Span]) -> list[Invocation]: ...
def bucket_by_hid(invocations: Iterable[Invocation]) -> dict[str, list[Invocation]]: ...

__all__ = ["Invocation", "Span"]
