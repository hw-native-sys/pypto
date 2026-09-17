# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Versioned torch_npu teardown integration; no independent atexit callback."""

import os
import sys
import threading
import warnings
from typing import Any

from pypto.runtime.kernel.context import KernelState

from .interop import _load_torch_npu
from .launch import _load_native


def _warn_shutdown(message: str) -> None:
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    except Exception:
        # Warning filters (including -Werror) must not interrupt framework teardown.
        try:
            sys.stderr.write(f"RuntimeWarning: {message}\n")
        except Exception:
            pass  # Interpreter shutdown may already have closed stderr.


class _ShutdownHook:
    def __init__(self, state: Any, native: Any, synchronize: Any, finalize: Any) -> None:
        self.state = state
        self.native = native
        self.pid = os.getpid()
        self._lock = threading.RLock()
        self._attempted = False
        self._synchronize = synchronize
        self._finalize = finalize

    def _close(self) -> None:
        # Never acquire locks inherited from another process or finalize its Worker.
        if os.getpid() != self.pid:
            return
        with self._lock:
            if self._attempted:
                return
            self._attempted = True
            if self.state.state is KernelState.CLOSED:
                return
            try:
                if not self.native.framework_alive():
                    raise RuntimeError("torch_npu device context was already torn down")
                self.state.close()
            except BaseException as exc:
                if self.state.state is KernelState.CLOSED:
                    _warn_shutdown(
                        f"PyPTO kernel cleanup completed after an earlier submission error: {exc}",
                    )
                    return
                self.state.retain_shutdown_failure(exc)
                # A Python global is insufficient: module clearing could otherwise
                # destroy a native Worker or event after ACL teardown. This anchor
                # intentionally survives interpreter finalization on failure only.
                self.native.retain_until_exit(self.state)
                _warn_shutdown(
                    "PyPTO kernel shutdown did not complete safely; "
                    f"resources retained until process exit: {exc}",
                )

    def synchronize(self, *args: Any, **kwargs: Any) -> Any:
        self._close()
        return self._synchronize(*args, **kwargs)

    def finalize(self, *args: Any, **kwargs: Any) -> Any:
        # Also guard explicit native teardown that bypasses the normal sync step.
        self._close()
        return self._finalize(*args, **kwargs)


_install_lock = threading.Lock()


def require_supported_framework(framework: Any) -> None:
    """Reject torch_npu releases whose teardown contract has not been validated."""
    version = framework.__version__.split("+", 1)[0]
    if version != "2.6.0.post2":
        raise RuntimeError(
            f"Automatic kernel shutdown is verified for torch_npu 2.6.0.post2, got {version}; "
            "this framework's teardown contract must be validated before kernel initialization"
        )


def install_shutdown(state: Any) -> Any:
    """Install before native initialization, while the framework is still alive."""
    framework = _load_torch_npu()
    native = _load_native()
    require_supported_framework(framework)
    with _install_lock:
        existing = getattr(framework, "_pypto_kernel_shutdown", None)
        if existing is not None:
            if existing.state is not state or existing.pid != os.getpid():
                raise RuntimeError("torch_npu shutdown is already bound to another kernel process state")
            return native
        synchronize = getattr(framework._C, "_npu_shutdown_synchronize", None)
        finalize = getattr(framework._C, "_npu_shutdown", None)
        if not callable(synchronize) or not callable(finalize) or not native.framework_alive():
            raise RuntimeError("torch_npu kernel shutdown requires a live framework with sync/teardown hooks")
        hook = _ShutdownHook(state, native, synchronize, finalize)
        framework._pypto_kernel_shutdown = hook
        framework._C._npu_shutdown_synchronize = hook.synchronize
        framework._C._npu_shutdown = hook.finalize
    return native
