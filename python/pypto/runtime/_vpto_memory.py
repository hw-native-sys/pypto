# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Barrier frontiers and conservative interval histories for VF admission."""

from bisect import bisect_left
from dataclasses import dataclass
from typing import NoReturn


def reject(reason: str) -> NoReturn:
    raise ValueError(f"unsupported VF synchronization contract: {reason}")


class BarrierHistory:
    """Track issued operations and explicit ordering; neither pipe implies FIFO."""

    def __init__(self) -> None:
        self.before = {pipe: {"vld": -1, "vst": -1} for pipe in ("vld", "vst")}
        self.issued = {pipe: {"vld": -1, "vst": -1} for pipe in ("vld", "vst")}

    def barrier(self, kind: str) -> None:
        source, target = kind.split(".")
        for pipe in ("vld", "vst"):
            self.before[target][pipe] = max(self.before[target][pipe], self.issued[source][pipe])

    def record(self, target: str, position: int) -> None:
        for pipe in ("vld", "vst"):
            self.issued[target][pipe] = max(self.issued[target][pipe], self.before[target][pipe])
        self.issued[target][target] = position


class StaticMemoryHistory(BarrierHistory):
    """Verify fixed 256-byte accesses at 32-byte granularity in linear time."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: dict[int, int] = {}
        self.writes: dict[int, int] = {}

    def access(self, displacement: int, load: bool, position: int) -> None:
        blocks = range(displacement // 32, displacement // 32 + 8)
        target = "vld" if load else "vst"
        dependencies = [(self.writes, "vst")] if load else [(self.reads, "vld"), (self.writes, "vst")]
        for history, source in dependencies:
            if any(history.get(block, -1) > self.before[target][source] for block in blocks):
                reject(
                    f"missing {source}.{target} barrier/order for overlapping UB access "
                    f"at byte {displacement}"
                )
        history = self.reads if load else self.writes
        for block in blocks:
            history[block] = position
        self.record(target, position)


@dataclass(frozen=True)
class Access:
    """A conservative half-open byte footprint and a unique memory instruction ID.

    A moving access advances by exactly its 256-byte width. Its own iterations
    are disjoint, so only that same instruction's cross-iteration history may
    be ignored. Other accesses always use the whole union, including distant
    iteration conflicts and masked-off lanes.
    """

    begin: int
    end: int
    load: bool
    identity: int
    moving: bool = False


def _latest_two(*groups: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    latest: dict[int, int] = {}
    for group in groups:
        for position, identity in group:
            latest[identity] = max(position, latest.get(identity, -1))
    return tuple(sorted(((position, identity) for identity, position in latest.items()), reverse=True)[:2])


class _IntervalHistory:
    """Range-add history with top two distinct instruction IDs in O(log N).

    Coordinate compression depends on access sites, never on iteration count
    or allocation size. Two IDs suffice for a query that excludes at most one
    instruction. Lazy tags retain maxima, rather than enumerating byte blocks.
    """

    def __init__(self, coordinates: list[int]) -> None:
        self.coordinates = coordinates
        self.size = max(0, len(coordinates) - 1)
        self.values: list[tuple[tuple[int, int], ...]] = [()] * (4 * max(1, self.size))
        self.lazy: list[tuple[tuple[int, int], ...]] = [()] * len(self.values)

    def _apply(self, node: int, entries: tuple[tuple[int, int], ...]) -> None:
        self.values[node] = _latest_two(self.values[node], entries)
        self.lazy[node] = _latest_two(self.lazy[node], entries)

    def visit(
        self, begin: int, end: int, entry: tuple[int, int] | None = None
    ) -> tuple[tuple[int, int], ...]:
        left = bisect_left(self.coordinates, begin)
        right = bisect_left(self.coordinates, end)

        def walk(node: int, lo: int, hi: int) -> tuple[tuple[int, int], ...]:
            if right <= lo or hi <= left:
                return ()
            if left <= lo and hi <= right:
                if entry is not None:
                    self._apply(node, (entry,))
                return self.values[node]
            mid = (lo + hi) // 2
            self._apply(node * 2, self.lazy[node])
            self._apply(node * 2 + 1, self.lazy[node])
            self.lazy[node] = ()
            result = _latest_two(walk(node * 2, lo, mid), walk(node * 2 + 1, mid, hi))
            self.values[node] = _latest_two(self.values[node * 2], self.values[node * 2 + 1])
            return result

        return walk(1, 0, self.size) if self.size else ()


class LoopMemoryHistory(BarrierHistory):
    """Use whole-loop footprints to reject unproved inter-iteration ordering."""

    def __init__(self, accesses: list[Access]) -> None:
        super().__init__()
        coordinates = sorted({bound for access in accesses for bound in (access.begin, access.end)})
        self.reads = _IntervalHistory(coordinates)
        self.writes = _IntervalHistory(coordinates)

    def access(self, access: Access, position: int) -> None:
        target = "vld" if access.load else "vst"
        dependencies = [(self.writes, "vst")] if access.load else [(self.reads, "vld"), (self.writes, "vst")]
        for history, source in dependencies:
            entries = history.visit(access.begin, access.end)
            for prior, identity in entries:
                if access.moving and identity == access.identity:
                    continue
                if prior > self.before[target][source]:
                    reject(f"missing {source}.{target} barrier/order across hardware-loop footprints")
        history = self.reads if access.load else self.writes
        history.visit(access.begin, access.end, (position, access.identity))
        self.record(target, position)


def validate_loop_history(
    prefix: list[Access | str], body: list[Access | str], suffix: list[Access | str]
) -> None:
    """Check loop entry, stationary body ordering, the backedge, and loop exit.

    Whole-loop unions overapproximate every iteration's footprint. Each body
    access has an invariant identity and each barrier transfers an issued
    frontier. Repeating the summary twice checks the first entry and every
    cyclic edge: subsequent repetitions have the same relative frontiers and
    cannot add another dependency kind. Self-edges are exempt only for proven
    disjoint advancing accesses. This is O(N log N), independent of trip count.
    """
    trace = prefix + body + body + suffix
    history = LoopMemoryHistory([item for item in trace if isinstance(item, Access)])
    for position, item in enumerate(trace):
        if isinstance(item, str):
            history.barrier(item)
        else:
            history.access(item, position)
