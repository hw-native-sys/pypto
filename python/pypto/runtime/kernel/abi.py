# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Adapter to the pinned Simpler ChipWorker's supported kernel lifecycle API."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypto._kernel_abi import KernelABI
from pypto.runtime._kernel_artifact import require_kernel_native


@dataclass(frozen=True)
class KernelConfig:
    """Immutable process configuration; no callable or per-call stream is stored."""

    platform: str
    runtime: str
    device_id: int
    aicpu_thread_num: int = 0
    enable_chip_swimlane: int | bool = 0
    enable_dep_gen: bool = False
    output_dir: str | Path | None = None
    # 每个 scope-depth ring 的运行时 arena 尺寸，None 表示沿用 Simpler 的编译期默认
    # （a2a3：ring_task_window=16384、ring_heap=256 MiB、ring_dep_pool=16384，共 4 个
    # ring）。kernel 模式没有 per-call 的 CallConfig——arena 在 init 时就由
    # ``prepare_kernel_runtime_impl`` 按 ``CallConfig.runtime_env`` 一次建好并冻结，
    # 所以这三项只能在这里给；给不了就只能吃掉 4×256 MiB 的默认 ring heap。
    # 标量广播到全部 4 个 ring，与 ``RunOptions`` 的同名字段语义一致。
    ring_task_window: int | tuple[int, ...] | None = None
    ring_heap: int | tuple[int, ...] | None = None
    ring_dep_pool: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        from pypto.runtime.runner import _normalize_swimlane_level  # noqa: PLC0415

        KernelABI(self.platform, self.runtime, ())
        object.__setattr__(
            self,
            "enable_chip_swimlane",
            _normalize_swimlane_level(self.enable_chip_swimlane, "enable_chip_swimlane"),
        )
        if type(self.enable_dep_gen) is not bool:
            raise TypeError(f"enable_dep_gen must be bool, got {self.enable_dep_gen!r}")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        if (self.enable_chip_swimlane or self.enable_dep_gen) and self.output_dir is None:
            raise ValueError("Kernel DFX requires output_dir when swimlane or dep_gen is enabled")
        if type(self.device_id) is not int or self.device_id < 0:
            raise ValueError(f"Expected a nonnegative kernel device id, got {self.device_id!r}")
        if type(self.aicpu_thread_num) is not int or self.aicpu_thread_num not in (0, 2, 3, 4, 5):
            raise ValueError(f"Expected auto (0) or 2..5 AICPU threads, got {self.aicpu_thread_num!r}")
        for field in ("ring_task_window", "ring_heap", "ring_dep_pool"):
            value = getattr(self, field)
            if value is None:
                continue
            if type(value) is int:
                value = (value,) * 4
            else:
                value = tuple(value)
                if len(value) != 4:
                    raise ValueError(f"{field} must be a scalar or exactly 4 per-ring values, got {len(value)}")
            if any(type(v) is not int or v <= 0 for v in value):
                raise ValueError(f"{field} entries must be positive ints, got {value}")
            object.__setattr__(self, field, value)


class _NativeWorker:
    def __init__(self, config: KernelConfig, state: Any):
        require_kernel_native(KernelABI(config.platform, config.runtime, ()))
        from pypto.torch.shutdown import install_shutdown  # noqa: PLC0415

        from .owner import _OwnerThread  # noqa: PLC0415

        self._native = install_shutdown(state)
        self._context = self._native.borrow_context(config.device_id)
        self.worker: Any = None
        self._owner = _OwnerThread()

    def init(self, config: KernelConfig) -> None:
        self._owner.call(lambda: self._init(config))

    def _init(self, config: KernelConfig) -> None:
        self._native.bind_context(self._context)
        interface = importlib.import_module("simpler.task_interface")
        runtime_builder = importlib.import_module("simpler_setup.runtime_builder")
        cfg = interface.CallConfig()
        cfg.aicpu_thread_num = config.aicpu_thread_num
        cfg.enable_chip_swimlane = config.enable_chip_swimlane
        cfg.enable_dep_gen = config.enable_dep_gen
        # kernel 模式的 arena 在 init 时一次建好并冻结（device_runner_base.cpp 的
        # init_kernel_context -> prepare_kernel_runtime_impl），之后 prepare_callable /
        # launch 都不再带 CallConfig，所以 ring 尺寸只有这一个入口。未设置的项保持 0，
        # 由 Simpler 落到自己的编译期默认。
        for field in ("ring_task_window", "ring_heap", "ring_dep_pool"):
            value = getattr(config, field)
            if value is not None:
                setattr(cfg.runtime_env, field, list(value))
        if config.output_dir is not None:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            cfg.output_prefix = str(config.output_dir)
        cfg.validate()
        bins = runtime_builder.RuntimeBuilder(platform=config.platform).get_binaries(
            config.runtime, build=False
        )
        self.worker = interface.ChipWorker()
        self.worker.kernel_init(config.device_id, bins, cfg)
        if not self.worker.kernel_mode_supported:
            raise RuntimeError(f"Simpler does not support kernel mode for {config.platform}/{config.runtime}")

    def prepare(self, callable_: Any) -> Any:
        return self._owner.call(lambda: self.worker.kernel_prepare_callable(callable_))

    def begin_dfx(self, caller_stream: int) -> None:
        self._owner.call(lambda: self.worker.kernel_begin_dfx(caller_stream))

    def end_dfx(self, caller_stream: int) -> None:
        self._owner.call(lambda: self.worker.kernel_end_dfx(caller_stream))

    @property
    def native_launch_target(self) -> Any:
        """Borrow the pinned SDK's registered C++ instance for a native type cast."""
        return self.worker._impl

    def close(self) -> None:
        def finalize() -> None:
            self._native.bind_context(self._context)
            if self.worker is not None:
                self.worker.finalize()

        self._owner.call(finalize)
        self._owner.stop()
