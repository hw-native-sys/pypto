# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Process-wide admission for mutually exclusive program and kernel initialization."""

import os
import threading

from pypto._artifact_contract import ArtifactExecutionMode


class _ModeGate:
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.mode: ArtifactExecutionMode | None = None
        self.lock = threading.Lock()

    def claim(self, mode: ArtifactExecutionMode) -> None:
        # Check before acquiring a possibly inherited, permanently locked mutex.
        if self.pid != os.getpid():
            raise RuntimeError("A forked process cannot inherit runtime mode state; use spawn")
        with self.lock:
            if self.mode is not None and self.mode is not mode:
                raise RuntimeError(
                    f"Cannot initialize {mode.value} mode: "
                    f"this process already claimed {self.mode.value} mode"
                )
            # Retain the claim even on failed initialization: native state may
            # be partially initialized. Closing does not permit mode switching.
            self.mode = mode


_gate = _ModeGate()


def _reset_mode_after_fork() -> None:
    # Reset exactly once while the child is single-threaded. Lazy reset in
    # claim() could clear another thread's claim and replace its active lock.
    # An inherited kernel claim remains terminal, even after failed init/close.
    if _gate.pid != os.getpid() and _gate.mode is not ArtifactExecutionMode.KERNEL:
        _gate.pid = os.getpid()
        _gate.mode = None
        _gate.lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_mode_after_fork)


def claim_program_mode() -> None:
    """Admit a program initializer before it touches native device state."""
    _gate.claim(ArtifactExecutionMode.PROGRAM)


def claim_kernel_mode() -> None:
    """Admit the unique kernel manager before its first native initialization."""
    _gate.claim(ArtifactExecutionMode.KERNEL)
