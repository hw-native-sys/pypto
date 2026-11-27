# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Model resource retention through Simpler; no framework graph interception."""

from collections.abc import Callable
from typing import Any


class CaptureInUseError(RuntimeError):
    """A caller-owned graph still has permission to replay."""


class CaptureResources:
    """One invocation's native resources, sharing its model's destroy notice.

    The session serializes admission and retirement. Model identities are only
    live keys: a destroyed model's address can be reused by a new recording.
    Neither this object nor the anchor table retains a Python NPUGraph.
    """

    def __init__(
        self,
        submission: Any,
        model: int,
        stream: int,
        anchors: dict[int, "CaptureResources"],
        synchronize_device: Callable[[], None],
    ) -> None:
        native = getattr(submission, "_native_submission", None)
        if native is None:
            raise NotImplementedError("Kernel capture requires a Simpler native submission")
        for method in (
            "_register_model_destroy_notice",
            "_model_destroyed",
            "_release_model_resources",
            "cancel",
        ):
            if not callable(getattr(native, method, None)):
                raise NotImplementedError(f"Kernel capture requires Simpler native submission.{method}")
        self.native = native
        self.model = model
        self.stream = stream
        self.anchors = anchors
        self.synchronize_device = synchronize_device
        self.retained = False
        # Native cancellation is a one-shot result. Python owner release can
        # fail independently and retry after the native token has retired.
        self._cancelled: bool | None = None

    def retain(self) -> None:
        anchor = self.anchors.get(self.model)
        if anchor is not None and anchor.native._model_destroyed():
            del self.anchors[self.model]
            anchor = None
        # The ticket already owns this object when preparation starts, including
        # failures after native allocation or destroy-notice registration.
        self.native._register_model_destroy_notice(anchor.native if anchor is not None else None)
        self.retained = True
        if anchor is None:
            self.anchors[self.model] = self

    def release(self) -> bool:
        if self._cancelled is not None:
            return self._cancelled
        if self.retained:
            if not self.native._model_destroyed():
                raise CaptureInUseError(
                    "Kernel graph resources are still live; finish capture and reset or destroy "
                    "the caller-owned NPUGraph before closing the Worker"
                )
            # Replay may use any stream. The native release only synchronizes
            # the recording stream, so the framework must drain all device use.
            self.synchronize_device()
        # Also cleans partial retention that never reached enqueue. Simpler
        # refuses release if a registered model notice has not arrived yet.
        cancelled: bool = self.native._release_model_resources()
        self._cancelled = cancelled
        return cancelled

    def forget(self) -> None:
        if self.anchors.get(self.model) is self:
            del self.anchors[self.model]
