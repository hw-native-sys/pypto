# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Participating graph lifetime and private lifecycle capture-mode contracts."""

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pypto.torch.capture import _GraphLifecycle


@pytest.fixture
def graphs():
    calls = []

    class Graph:
        def capture_end(self):
            calls.append("capture-end")

        def replay(self):
            calls.append("replay")

        def reset(self):
            calls.append("reset")

    state = SimpleNamespace(config=SimpleNamespace(device_id=0))
    framework = SimpleNamespace(
        npu=SimpleNamespace(
            NPUGraph=Graph,
            current_stream=lambda: SimpleNamespace(stream_id=7, device=SimpleNamespace(index=0)),
            synchronize=lambda device: calls.append(("synchronize", device)),
        )
    )
    native = SimpleNamespace(check_call=lambda stream, device: 42)
    lifecycle = _GraphLifecycle(state, framework, native)
    return Graph, lifecycle, calls


def test_close_stops_replay_drains_and_resets_only_participating_graphs(graphs):
    Graph, lifecycle, calls = graphs
    unrelated = Graph()
    unrelated.capture_end()
    graph = Graph()
    lifecycle.notice(42)
    graph.capture_end()
    graph.replay()
    lifecycle.close()
    assert calls == ["capture-end", "capture-end", "replay", ("synchronize", 0), "reset"]
    with pytest.raises(RuntimeError, match="after kernel shutdown"):
        graph.replay()
    unrelated.replay()
    assert calls[-1] == "replay"
    assert not lifecycle.graphs


def test_graph_reset_waits_before_releasing_and_avoids_second_reset(graphs):
    Graph, lifecycle, calls = graphs
    graph = Graph()
    lifecycle.notice(42)
    graph.capture_end()
    graph.reset()
    assert calls[-2:] == [("synchronize", 0), "reset"]
    lifecycle.close()
    assert calls.count("reset") == 1
    graph.replay()
    assert calls[-1] == "replay"


def test_graph_ownership_does_not_prevent_framework_gc(graphs):
    Graph, lifecycle, _ = graphs
    graph = Graph()
    lifecycle.notice(42)
    graph.capture_end()
    reference = weakref.ref(graph)
    del graph
    gc.collect()
    assert reference() is None and not lifecycle.graphs


def test_unfinished_capture_prevents_resource_release(graphs):
    _, lifecycle, calls = graphs
    lifecycle.notice(42)
    with pytest.raises(RuntimeError, match="capture has not ended"):
        lifecycle.close()
    assert not calls
    with pytest.raises(RuntimeError, match="after kernel shutdown"):
        lifecycle.notice(43)


def test_close_waits_for_replay_admission(graphs):
    Graph, lifecycle, calls = graphs
    entered, resume = threading.Event(), threading.Event()
    graph = Graph()
    lifecycle.notice(42)
    graph.capture_end()

    def replay(graph):
        entered.set()
        assert resume.wait(5)
        calls.append("replay-done")

    lifecycle._replay = replay
    with ThreadPoolExecutor(max_workers=2) as executor:
        execution = executor.submit(graph.replay)
        assert entered.wait(5)
        closing = executor.submit(lifecycle.close)
        assert not closing.done()
        resume.set()
        execution.result(timeout=5)
        closing.result(timeout=5)
    assert calls[-3:] == ["replay-done", ("synchronize", 0), "reset"]


@pytest.mark.parametrize("failure", ["drain", "reset"])
def test_failed_graph_cleanup_stops_replay_and_retains_graphs(graphs, failure):
    Graph, lifecycle, _ = graphs
    graph = Graph()
    lifecycle.notice(42)
    graph.capture_end()

    def fail(*args):
        raise RuntimeError(failure)

    if failure == "drain":
        lifecycle.framework.npu.synchronize = fail
    else:
        lifecycle._reset = fail
    with pytest.raises(RuntimeError, match=failure):
        lifecycle.close()
    assert graph in lifecycle.graphs
    with pytest.raises(RuntimeError, match="after kernel shutdown"):
        graph.replay()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
