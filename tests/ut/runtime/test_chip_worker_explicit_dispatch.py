# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the explicit dispatch surface on :class:`ChipWorker`.

Exercises ``ChipWorker.run / register`` and the :class:`RegistrationHandle`
returned by ``register`` without touching the device: ``_SimplerWorker`` is
patched so construction does no real work, and ``CompiledProgram`` extraction
helpers (``chip_callable / build_orch_args / build_call_config``) are stubbed
to return mocks the tests can inspect.
"""

from unittest.mock import MagicMock, patch

import pytest
from pypto.runtime import ChipWorker, RegistrationHandle, RunConfig


@pytest.fixture
def fake_simpler_worker():
    """Patch ``simpler.worker.Worker`` so ChipWorker construction does no I/O."""
    with patch("pypto.runtime.worker._SimplerWorker") as cls:
        instance = MagicMock()
        instance.register.side_effect = lambda cc: 100 + id(cc) % 100  # deterministic cid
        instance.aicpu_dlopen_count = 0
        instance.host_dlopen_count = 0
        cls.return_value = instance
        yield instance


def _fake_compiled(platform="a2a3sim", runtime="host_build_graph"):
    """Build a CompiledProgram mock matching ChipWorker(...) bindings."""
    cc = MagicMock(name="chip_callable")
    compiled = MagicMock(name="CompiledProgram")
    compiled.platform = platform
    compiled.runtime_name = runtime
    compiled.chip_callable = cc
    compiled.output_dir = "/tmp/fake_compiled"
    compiled.output_indices = [2]
    compiled.build_orch_args.return_value = ("orch_args", ["arg0", "arg1", "out_tensor"], False)
    compiled.build_call_config.return_value = MagicMock(name="CallConfig")
    return compiled


# ---------------------------------------------------------------------------
# Worker.run
# ---------------------------------------------------------------------------


def test_run_dispatches_via_run_chip(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    w.run(compiled, 1, 2, 3)
    # simpler.Worker.run called exactly once on this run.
    assert fake_simpler_worker.run.call_count == 1
    cid, orch_args, _cfg = fake_simpler_worker.run.call_args.args
    assert orch_args == "orch_args"
    # cid is the value returned by simpler register for this chip_callable.
    assert cid == fake_simpler_worker.register.return_value or isinstance(cid, int)
    w.close()


def test_run_reuses_cid_across_calls(fake_simpler_worker):
    """Two run() calls with the same compiled hit register once, run twice."""
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    w.run(compiled, 1, 2, 3)
    w.run(compiled, 1, 2, 3)
    assert fake_simpler_worker.register.call_count == 1
    assert fake_simpler_worker.run.call_count == 2
    w.close()


def test_run_rejects_platform_mismatch(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled(platform="a5sim")
    with pytest.raises(ValueError, match="platform"):
        w.run(compiled, 1, 2, 3)
    w.close()


def test_run_rejects_runtime_mismatch(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"), runtime="host_build_graph")
    compiled = _fake_compiled(runtime="tensormap_and_ringbuffer")
    with pytest.raises(ValueError, match="runtime"):
        w.run(compiled, 1, 2, 3)
    w.close()


def test_run_requires_init(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"), auto_init=False)
    compiled = _fake_compiled()
    with pytest.raises(RuntimeError, match="initialized ChipWorker"):
        w.run(compiled, 1, 2, 3)


def test_run_inplace_returns_none(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    # build_orch_args returns return_style=False — in-place call.
    result = w.run(compiled, 1, 2, 3)
    assert result is None
    w.close()


def test_run_return_style_packs_outputs(fake_simpler_worker):
    """When build_orch_args reports return_style, run() should pack outputs."""
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    out_tensor = MagicMock(name="out_tensor")
    compiled.build_orch_args.return_value = ("orch_args", [1, 2, out_tensor], True)
    compiled.output_indices = [2]
    ret = w.run(compiled, 1, 2)
    assert ret is out_tensor
    w.close()


# ---------------------------------------------------------------------------
# Worker.register + RegistrationHandle
# ---------------------------------------------------------------------------


def test_register_returns_handle(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    h = w.register(compiled)
    assert isinstance(h, RegistrationHandle)
    assert h.compiled is compiled
    assert h.closed is False
    w.close()


def test_register_eager_invokes_simpler_register(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    n0 = fake_simpler_worker.register.call_count
    w.register(compiled)
    assert fake_simpler_worker.register.call_count == n0 + 1
    w.close()


def test_register_rejects_binding_mismatch(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled(platform="a5sim")
    with pytest.raises(ValueError, match="platform"):
        w.register(compiled)
    w.close()


def test_handle_call_dispatches(fake_simpler_worker):
    """RegistrationHandle.__call__ delegates to Worker.run (same path)."""
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    h = w.register(compiled)
    fake_simpler_worker.run.reset_mock()
    h(1, 2, 3)
    assert fake_simpler_worker.run.call_count == 1
    w.close()


def test_handle_unregister_marks_closed(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    h = w.register(compiled)
    h.unregister()
    assert h.closed is True
    with pytest.raises(RuntimeError, match="unregistered"):
        h(1, 2, 3)
    w.close()


def test_handle_unregister_does_not_release_simpler_cid(fake_simpler_worker):
    """unregister() must NOT call simpler.unregister_callable — that's deferred to close()."""
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    h = w.register(compiled)
    n0 = fake_simpler_worker.unregister_callable.call_count
    h.unregister()
    assert fake_simpler_worker.unregister_callable.call_count == n0
    w.close()


def test_handle_context_manager(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    with w.register(compiled) as h:
        assert h.closed is False
    assert h.closed is True
    w.close()


def test_close_marks_alive_handles_closed(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    compiled = _fake_compiled()
    h = w.register(compiled)
    assert h.closed is False
    w.close()
    assert h.closed is True


def test_close_releases_all_cids(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    a = _fake_compiled()
    b = _fake_compiled()
    w.register(a)
    w.register(b)
    n0 = fake_simpler_worker.unregister_callable.call_count
    w.close()
    # Two distinct cids registered → two unregister_callable calls.
    assert fake_simpler_worker.unregister_callable.call_count - n0 == 2


# ---------------------------------------------------------------------------
# Diagnostic property passthrough
# ---------------------------------------------------------------------------


def test_aicpu_dlopen_count_passthrough(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    fake_simpler_worker.aicpu_dlopen_count = 7
    assert w.aicpu_dlopen_count == 7
    w.close()


def test_host_dlopen_count_passthrough(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    fake_simpler_worker.host_dlopen_count = 3
    assert w.host_dlopen_count == 3
    w.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
