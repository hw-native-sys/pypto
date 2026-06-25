# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Register-once, multi-round on-device benchmark (issue #1858).

Mirrors simpler's ``scene_test --rounds`` mode through pypto's public Worker:
register the compiled program once, dispatch ``rounds`` cheap launches via
:meth:`pypto.runtime.RegistrationHandle.run_timed`, and aggregate per-launch
``device_wall_us``. This avoids the one-shot ``execute_compiled`` /
``CompiledProgram.__call__`` path, which re-pays ``compile_and_assemble`` +
register/load every call (hundreds of ms of host overhead that swamps the
~1 ms device time).
"""

import statistics
from dataclasses import dataclass, field
from typing import Any

from .runner import RunConfig
from .worker import ChipWorker

__all__ = ["BenchmarkStats", "benchmark"]


@dataclass
class BenchmarkStats:
    """Aggregated per-launch timing from :func:`benchmark`.

    The min / median / mean / max / stdev helpers operate on
    ``device_wall_us`` — the on-NPU metric. ``host_wall_us`` samples are kept
    for context, but they include per-launch arg coercion + H2D and so are not
    the device metric.

    Attributes:
        device_wall_us: Per-measured-launch on-NPU orchestrator wall times
            (microseconds). Length is ``rounds`` (warmup launches excluded).
        host_wall_us: Per-measured-launch host wall times (microseconds).
        rounds: Number of measured launches.
        warmup: Number of leading launches discarded before measurement.
    """

    device_wall_us: list[float] = field(default_factory=list)
    host_wall_us: list[float] = field(default_factory=list)
    rounds: int = 0
    warmup: int = 0

    @property
    def device_us_min(self) -> float:
        return min(self.device_wall_us) if self.device_wall_us else 0.0

    @property
    def device_us_median(self) -> float:
        return statistics.median(self.device_wall_us) if self.device_wall_us else 0.0

    @property
    def device_us_mean(self) -> float:
        return statistics.fmean(self.device_wall_us) if self.device_wall_us else 0.0

    @property
    def device_us_max(self) -> float:
        return max(self.device_wall_us) if self.device_wall_us else 0.0

    @property
    def device_us_stdev(self) -> float:
        return statistics.stdev(self.device_wall_us) if len(self.device_wall_us) > 1 else 0.0

    # ``device_wall_us_*`` / ``samples`` are issue #1858-sketch-aligned aliases
    # of the ``device_us_*`` / ``device_wall_us`` accessors above.
    @property
    def samples(self) -> list[float]:
        """Alias for :attr:`device_wall_us` — the measured device-wall samples."""
        return self.device_wall_us

    @property
    def device_wall_us_min(self) -> float:
        return self.device_us_min

    @property
    def device_wall_us_median(self) -> float:
        return self.device_us_median

    @property
    def device_wall_us_mean(self) -> float:
        return self.device_us_mean

    @property
    def device_wall_us_max(self) -> float:
        return self.device_us_max

    @property
    def device_wall_us_stdev(self) -> float:
        return self.device_us_stdev

    @property
    def all_zero_device(self) -> bool:
        """``True`` if no real device wall was measured.

        Happens on a runtime built without ``PTO2_PROFILING`` (``device_wall_us``
        is ``0``, not absent) — benchmark callers should then fall back to
        ``host_wall_us`` or rebuild with profiling enabled.
        """
        return bool(self.device_wall_us) and not any(self.device_wall_us)

    def __str__(self) -> str:
        if not self.device_wall_us:
            return f"BenchmarkStats(rounds={self.rounds}: no samples)"
        if self.all_zero_device:
            return (
                f"BenchmarkStats(rounds={self.rounds}): device_wall_us all 0 — runtime "
                f"built without PTO2_PROFILING (use host_wall_us or rebuild with profiling)"
            )
        return (
            f"BenchmarkStats(rounds={self.rounds}, warmup={self.warmup}): "
            f"device_wall_us min={self.device_us_min:.1f} median={self.device_us_median:.1f} "
            f"mean={self.device_us_mean:.1f} max={self.device_us_max:.1f} "
            f"stdev={self.device_us_stdev:.1f}"
        )


def benchmark(
    compiled: Any,
    args: list[Any],
    *,
    rounds: int = 100,
    warmup: int = 3,
    platform: str | None = None,
    device_id: int | None = None,
    config: RunConfig | None = None,
) -> BenchmarkStats:
    """Register *compiled* once and dispatch *rounds* timed launches.

    Opens a single :class:`~pypto.runtime.ChipWorker`, registers *compiled*
    once, then loops :meth:`~pypto.runtime.RegistrationHandle.run_timed` so
    each launch only re-pays argument coercion + dispatch (not register/load).
    The on-NPU ``device_wall_us`` is measured between the orchestrator's
    ``orch_start`` / ``orch_end`` and is unaffected by the per-launch host-side
    arg building.

    Args:
        compiled: A single-orchestration
            :class:`~pypto.ir.CompiledProgram` from ``ir.compile`` /
            ``compile_program``. Multi-orch programs must pass
            ``compiled[<name>]``.
        args: Positional dispatch args, same as ``compiled(*args)``.
        rounds: Number of measured launches. Must be positive.
        warmup: Number of leading launches discarded before measurement
            (page-in / cache warm). Total launches = ``warmup + rounds``.
        platform: Target platform shorthand, e.g. ``"a2a3"``. Defaults to
            ``compiled.platform``. Mutually exclusive with *config*.
        device_id: NPU device index. Defaults to ``RunConfig``'s default.
            Mutually exclusive with *config*.
        config: Optional :class:`~pypto.runtime.RunConfig` for full control
            (``block_dim`` / ``aicpu_thread_num`` / ``pto_isa_commit``). Pass
            this *or* *platform*/*device_id*, not both.

    Returns:
        A :class:`BenchmarkStats` with the per-launch ``device_wall_us`` /
        ``host_wall_us`` samples and aggregate helpers.

    Raises:
        ValueError: ``rounds <= 0``, ``warmup < 0``, or *config* passed
            together with *platform* / *device_id*.

    Note:
        Only L2 single-chip runs carry a real ``device_wall_us``. On a runtime
        built without ``PTO2_PROFILING`` every sample is ``0`` — check
        :attr:`BenchmarkStats.all_zero_device`.
    """
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    if config is not None and (platform is not None or device_id is not None):
        raise ValueError(
            "benchmark(): pass either config=... or platform=/device_id=, not both"
        )

    if config is not None:
        rc = config
    else:
        rc_kwargs: dict[str, Any] = {"platform": platform or compiled.platform}
        if device_id is not None:
            rc_kwargs["device_id"] = device_id
        rc = RunConfig(**rc_kwargs)

    stats = BenchmarkStats(rounds=rounds, warmup=warmup)
    with ChipWorker(rc, runtime=compiled.runtime_name) as worker:
        handle = worker.register(compiled)  # register once; cid cached
        for _ in range(warmup):  # warm caches / page-in; results discarded
            handle.run_timed(*args, config=rc)
        for _ in range(rounds):  # measured launches
            _outputs, timing = handle.run_timed(*args, config=rc)
            stats.device_wall_us.append(float(timing.device_wall_us))
            stats.host_wall_us.append(float(timing.host_wall_us))
    return stats
