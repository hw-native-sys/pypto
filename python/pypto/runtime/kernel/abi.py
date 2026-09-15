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

    def __post_init__(self) -> None:
        KernelABI(self.platform, self.runtime, ())
        if type(self.device_id) is not int or self.device_id < 0:
            raise ValueError(f"Expected a nonnegative kernel device id, got {self.device_id!r}")
        if type(self.aicpu_thread_num) is not int or self.aicpu_thread_num not in (0, 2, 3, 4, 5):
            raise ValueError(f"Expected auto (0) or 2..5 AICPU threads, got {self.aicpu_thread_num!r}")


class _NativeWorker:
    def __init__(self, config: KernelConfig):
        require_kernel_native(KernelABI(config.platform, config.runtime, ()))
        interface = importlib.import_module("simpler.task_interface")
        self.worker = interface.ChipWorker()

    def init(self, config: KernelConfig) -> None:
        interface = importlib.import_module("simpler.task_interface")
        runtime_builder = importlib.import_module("simpler_setup.runtime_builder")
        cfg = interface.CallConfig()
        cfg.aicpu_thread_num = config.aicpu_thread_num
        cfg.validate()
        bins = runtime_builder.RuntimeBuilder(platform=config.platform).get_binaries(
            config.runtime, build=False
        )
        self.worker.kernel_init(config.device_id, bins, cfg)
        if not self.worker.kernel_mode_supported:
            raise RuntimeError(f"Simpler does not support kernel mode for {config.platform}/{config.runtime}")

    def prepare(self, callable_: Any) -> Any:
        return self.worker.kernel_prepare_callable(callable_)

    def close(self) -> None:
        self.worker.finalize()
