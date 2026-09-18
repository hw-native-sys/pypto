# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Direct and registered JIT capture acceptance in isolated NPU processes."""

import ctypes
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _entrypoints(entry):
    import torch  # noqa: PLC0415
    from pypto.torch import register  # noqa: PLC0415

    from tests.st.runtime.kernel.test_jit_eager import accumulate, add_constant  # noqa: PLC0415

    direct = accumulate, add_constant
    if entry == "jit":
        return direct, direct
    register(accumulate, "pypto_capture_st::update")
    for value in (4, 5):
        register(add_constant, f"pypto_capture_st::add_{value}", constexpr={"value": value})

    def update(x, scalar, out):
        # Dispatcher schemas accept Python primitives, whereas direct JIT also
        # accepts typed ctypes values. Both paths snapshot the same FP32 value.
        value = scalar.value if isinstance(scalar, ctypes.c_float) else scalar
        return torch.ops.pypto_capture_st.update(x, value, out)

    def add(x, out, *, value=4):
        return getattr(torch.ops.pypto_capture_st, f"add_{value}")(x, out)

    registered = update, add
    if entry == "jit_to_ops":
        return direct, registered
    if entry == "ops_to_jit":
        return registered, direct
    if entry == "mixed":
        return direct, (update, direct[1])
    return registered, registered


def _configure_capture_cache(directory, case):
    from pypto import CacheConfig, configure_cache  # noqa: PLC0415

    if case == "build-dir":
        os.environ["PYPTO_PROG_BUILD_DIR"] = str(Path(directory) / "generated")
        for name in ("PYPTO_CACHE", "PYPTO_CACHE_DIR", "PYPTO_CACHE_READONLY"):
            os.environ.pop(name, None)
        configure_cache(None)
    else:
        configure_cache(CacheConfig(enabled=case == "persistent", root=Path(directory) / "cache"))


def _check_default_cache_reuse(counts, update, add, x, out, following, directory, restored):
    from pypto import cache_stats  # noqa: PLC0415
    from pypto.jit._artifact_manifest import MANIFEST_NAME  # noqa: PLC0415

    before_repeat = counts.copy()
    assert counts["compile"] == (0 if restored else 2)
    assert counts["prepare"] == 2, "each process must prepare its own callable"
    stats = cache_stats()
    assert stats.binary_builds == (0 if restored else 2)
    assert stats.ready_hits == (2 if restored else 0)
    update(x, 3.0, out)
    add(out, following, value=4)
    assert counts == before_repeat, "default-cache hits must not compile or prepare again"
    assert list((Path(directory) / "generated/.pypto-cache").rglob(MANIFEST_NAME))


def _run(device, directory, case, entry="jit", restored=False):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto.runtime import RunConfig  # noqa: PLC0415
    from pypto.runtime.kernel.abi import _NativeWorker  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init  # noqa: PLC0415

    from tests.st.runtime.kernel.test_jit_eager import accumulate  # noqa: PLC0415

    os.chdir(directory)
    torch_npu.npu.set_device(device)
    _configure_capture_cache(directory, case)
    # Only the internal artifact lookup takes compile-side RunConfig; kernel calls never do.
    build_config = RunConfig(platform="a2a3", device_id=device)
    x = torch.full((16, 16), 2.0, device=f"npu:{device}")
    out = torch.zeros_like(x)
    following = torch.empty_like(out)
    state = get_process_kernel_state()
    counts = dict(compile=0, init=0, prepare=0)
    compiler = importlib.import_module("pypto.ir.compile")

    def counted(name, original):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapped

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compiler, "_compile_impl", counted("compile", compiler._compile_impl))
        patch.setattr(_NativeWorker, "init", counted("init", _NativeWorker.init))
        patch.setattr(_NativeWorker, "prepare", counted("prepare", _NativeWorker.prepare))
        (warm_update, warm_add), (update, add) = _entrypoints(entry)
        assert counts == dict(compile=0, init=0, prepare=0)
        unbound = case in ("uninitialized", "init-in-capture")
        rejected = unbound or case in ("cold", "generated", "binary", "second-cold", "new-variant")
        if not unbound:
            init()
            assert counts == dict(compile=0, init=1, prepare=0)
        if case in ("generated", "binary"):
            artifact = accumulate._resolve_kernel_artifact((x, 3.0, out), dict(config=build_config))
            assert not state._registrations and artifact._loaded is None
            if case == "binary":
                artifact.load()
                assert not state._registrations
        elif not unbound and case != "cold":
            warm_update(x, 3.0, out)
            if case in (
                "multi",
                "graphs",
                "streams",
                "recreate",
                "gc",
                "owners",
                "shutdown",
                "new-variant",
                "persistent",
                "build-dir",
            ):
                warm_add(out, following, value=4)
            if case == "build-dir":
                _check_default_cache_reuse(
                    counts, warm_update, warm_add, x, out, following, directory, restored
                )
            torch_npu.npu.synchronize()
            out.zero_()
        warmed = counts.copy()
        _clear_compilation_caches()
        graph = torch_npu.npu.NPUGraph()
        scalar = ctypes.c_float(3.0)
        with torch_npu.npu.graph(graph):
            if rejected:
                if case in ("second-cold", "new-variant"):
                    # Join the warmed first operator's private stream to capture
                    # before checking that a cold second callable is refused.
                    assert update(x, scalar, out) is out
                # Catch inside capture so the framework can finish a valid graph.
                if case == "init-in-capture":
                    with pytest.raises(RuntimeError, match="outside graph capture"):
                        init()
                else:
                    refusal = r"call pypto\.torch\.init" if unbound else "requires warmup outside capture"
                    with pytest.raises(RuntimeError, match=refusal):
                        if case in ("second-cold", "new-variant"):
                            add(out, following, value=5 if case == "new-variant" else 4)
                        else:
                            update(x, scalar, out)
                following.copy_(out)
            else:
                assert update(x, scalar, out) is out
                if case != "single":
                    assert add(out, following, value=4) is following
        assert counts == warmed
        if rejected:
            assert (state._worker is None) == unbound
            graph.reset()
            return
        callables = 1 if case == "single" else 2
        assert warmed == dict(compile=0 if restored else callables, init=1, prepare=callables)
        scalar.value = 99.0
        tensors = [x, out, following]
        del x, out, following
        graphs = [graph]
        del graph
        graph = _replay_case(case, graphs, tensors, device, torch_npu, add)
        assert counts == warmed
        globals()["retained_graph"] = graph


def _clear_compilation_caches():
    """Prepared registrations must outlive every compiler-side cache."""
    from tests.st.runtime.kernel.test_jit_eager import accumulate, add_constant  # noqa: PLC0415

    for kernel in (accumulate, add_constant):
        kernel._cache.clear()
        kernel._kernel_contracts.clear()
        kernel._artifact_objects.clear()


def _replay(graph):
    from pypto.jit.decorator import JITFunction  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        raise AssertionError("Graph replay reentered Python JIT")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(JITFunction, "__call__", forbidden)
        graph.replay()


def _replay_case(case, graphs, tensors, device, torch_npu, add):
    import gc  # noqa: PLC0415
    import weakref  # noqa: PLC0415

    import torch  # noqa: PLC0415

    graph = graphs.pop()
    x, out, following = tensors
    tensors.clear()
    replay_stream = torch_npu.npu.Stream(device=device)
    expected = 0.0
    with torch_npu.npu.stream(replay_stream):
        for value in range(2, 7):
            x.fill_(value)
            _replay(graph)
            expected += value * 3.0
            torch.testing.assert_close(out.cpu(), torch.full((16, 16), expected))
            if case != "single":
                torch.testing.assert_close(following.cpu(), torch.full((16, 16), expected + 4))
    if case in ("graphs", "streams"):
        second = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(second):
            add(following, out, value=4)
        current = torch_npu.npu.current_stream()
        for _ in range(5):
            if case == "streams":
                replay_stream.wait_stream(current)
                with torch_npu.npu.stream(replay_stream):
                    _replay(graph)
                current.wait_stream(replay_stream)
            else:
                _replay(graph)
            _replay(second)
            expected += 6 * 3 + 8
        torch_npu.npu.synchronize()
        torch.testing.assert_close(out.cpu(), torch.full((16, 16), expected))
        globals()["second_graph"] = second
    elif case in ("recreate", "gc"):
        if case == "recreate":
            graph.reset()
        else:
            reference = weakref.ref(graph)
            del graph
            gc.collect()
            assert reference() is None
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph):
            add(out, following, value=4)
        _replay(graph)
        torch.testing.assert_close(following.cpu(), torch.full((16, 16), expected + 4))
    elif case == "owners":
        input_pointer = x.data_ptr()
        del x
        gc.collect()
        pressure = [torch.full_like(out, 999) for _ in range(32)]
        _replay(graph)
        expected += 6 * 3
        torch.testing.assert_close(out.cpu(), torch.full((16, 16), expected))
        torch.testing.assert_close(following.cpu(), torch.full((16, 16), expected + 4))
        assert all(tensor.data_ptr() != input_pointer for tensor in pressure)
    elif case == "shutdown":
        with torch_npu.npu.stream(replay_stream):
            for _ in range(20):
                _replay(graph)
        # Ordinary process exit must drain pending replay before Worker close.
    return graph


def _isolated(test_config, tmp_path, case, queue_enabled, entry, restored=False):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU")
    pytest.importorskip("torch_npu")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.st.runtime.kernel.test_capture import _run; "
            "import sys; _run(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == '1')",
            str(test_config.device_id),
            str(tmp_path),
            case,
            entry,
            str(int(restored)),
        ],
        env=dict(os.environ, TASK_QUEUE_ENABLE=str(queue_enabled)),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PyPTO kernel shutdown did not complete" not in result.stderr
    if case == "build-dir" and not restored:
        _isolated(test_config, tmp_path, case, queue_enabled, entry, restored=True)


@pytest.mark.parametrize("entry", ["jit", "torch_ops"])
@pytest.mark.parametrize(
    "case",
    [
        "uninitialized",
        "init-in-capture",
        "cold",
        "generated",
        "binary",
        "second-cold",
        "new-variant",
        "single",
        "multi",
        "graphs",
        "streams",
        "persistent",
        "build-dir",
        "owners",
        "recreate",
        "gc",
        "shutdown",
    ],
)
@pytest.mark.parametrize("queue_enabled", [0, 1])
def test_capture(test_config, tmp_path, case, queue_enabled, entry):
    _isolated(test_config, tmp_path, case, queue_enabled, entry)


@pytest.mark.parametrize("entry", ["jit_to_ops", "ops_to_jit", "mixed"])
@pytest.mark.parametrize("case", ["multi", "persistent", "build-dir"])
@pytest.mark.parametrize("queue_enabled", [0, 1])
def test_capture_entry_interop(test_config, tmp_path, entry, case, queue_enabled):
    _isolated(test_config, tmp_path, case, queue_enabled, entry)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
