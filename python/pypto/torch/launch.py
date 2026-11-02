# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lazy native lifecycle queries. Callback submission is a later integration step."""

import importlib
from typing import Any


def _load_native() -> Any:
    try:
        native = importlib.import_module("pypto._torch_npu")
    except ModuleNotFoundError as exc:
        if exc.name != "pypto._torch_npu":
            raise
        raise RuntimeError(
            "Torch kernel initialization requires the optional pypto._torch_npu adapter; "
            "build with PYPTO_BUILD_TORCH_NPU=ON and matching torch/torch_npu/CANN"
        ) from exc
    for name in ("check_call", "borrow_context", "bind_context", "framework_alive", "retain_until_exit"):
        if not callable(getattr(native, name, None)):
            raise NotImplementedError(f"Torch lifecycle adapter requires {name}; rebuild the adapter")
    return native


__all__ = ["_load_native"]
