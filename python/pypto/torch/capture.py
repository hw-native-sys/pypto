# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Versioned NPUGraph lifetime integration for graphs containing PyPTO calls."""

import threading
import weakref
from typing import Any

from .interop import _load_torch_npu
from .shutdown import install_shutdown


class _GraphLifecycle:
    def __init__(self, state: Any, framework: Any, native: Any) -> None:
        self.state = state
        self.framework = framework
        self.native = native
        self.lock = threading.RLock()
        self.pending: set[int] = set()
        self.graphs: weakref.WeakSet[Any] = weakref.WeakSet()
        self.stopped = False
        graph_type = framework.npu.NPUGraph
        self._capture_end = graph_type.capture_end
        self._replay = graph_type.replay
        self._reset = graph_type.reset

        def capture_end(graph: Any) -> Any:
            stream = framework.npu.current_stream()
            capture_id = native.check_call(stream.stream_id, stream.device.index)
            result = self._capture_end(graph)
            with self.lock:
                if capture_id in self.pending:
                    self.graphs.add(graph)
                    graph._pypto_kernel_lifecycle = self
                    self.pending.remove(capture_id)
            return result

        def replay(graph: Any) -> Any:
            if getattr(graph, "_pypto_kernel_lifecycle", None) is not self:
                return self._replay(graph)
            with self.lock:
                if self.stopped:
                    raise RuntimeError("Cannot replay a PyPTO graph after kernel shutdown")
                return self._replay(graph)

        def reset(graph: Any) -> Any:
            if getattr(graph, "_pypto_kernel_lifecycle", None) is not self:
                return self._reset(graph)
            with self.lock:
                if self.stopped:
                    raise RuntimeError("Cannot reset a PyPTO graph after kernel shutdown")
                # A graph may have been replayed on another stream. The graph
                # destroy callback alone is not proof of device completion.
                framework.npu.synchronize(state.config.device_id)
                result = self._reset(graph)
                self.graphs.discard(graph)
                del graph._pypto_kernel_lifecycle
                return result

        graph_type.capture_end = capture_end
        graph_type.replay = replay
        graph_type.reset = reset

    def notice(self, capture_id: int) -> None:
        with self.lock:
            if self.stopped:
                raise RuntimeError("Cannot capture a PyPTO call after kernel shutdown")
            self.pending.add(capture_id)

    def close(self) -> None:
        with self.lock:
            self.stopped = True
            if self.pending:
                raise RuntimeError("PyPTO graph capture has not ended; retain kernel resources")
            # Stop framework replay entry before draining queues, then destroy
            # only participating live graphs before any callable/Worker release.
            if self.state.config is not None:
                self.framework.npu.synchronize(self.state.config.device_id)
            for graph in list(self.graphs):
                self._reset(graph)
            self.graphs.clear()


_install_lock = threading.Lock()


def install_capture(state: Any) -> _GraphLifecycle:
    """Extend the verified framework shutdown hook on the first captured call."""
    native = install_shutdown(state)
    with _install_lock:
        if state._graph_lifecycle is None:
            state._graph_lifecycle = _GraphLifecycle(state, _load_torch_npu(), native)
        return state._graph_lifecycle
