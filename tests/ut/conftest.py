# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared fixtures for unit tests."""

import importlib
import os
import subprocess
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Final
from unittest.mock import MagicMock, patch

# The unit suite stubs simpler wherever it needs it — `simpler_setup` and
# `pypto.runtime.task_interface` are monkeypatched in sys.modules — so it never runs
# against the installed runtime, and must not be gated on that runtime matching
# `runtime/`. Set before the first `pypto` import below, so no import path can reach a
# guarded module ahead of it; a developer can re-enable the guard for a run with
# PYPTO_SKIP_RUNTIME_PIN_CHECK=0. The guard's own logic is covered directly in
# tests/ut/runtime/test_runtime_pin.py, and system tests (tests/st) run real kernels
# and are deliberately not exempted.
os.environ.setdefault("PYPTO_SKIP_RUNTIME_PIN_CHECK", "1")

import pytest
from pypto import LogLevel, get_log_level, set_log_level
from pypto import backend as _backend
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import _clear_thread_log_level, passes

# Snapshot the C++ log level at import — before any test has had a chance to
# mutate it — so every unit test starts from whatever PYPTO_LOG_LEVEL / the
# build-type default selected for this session. See `_reset_log_level`.
_INITIAL_LOG_LEVEL: Final[LogLevel] = get_log_level()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-ptoas",
        action="store_true",
        help="Fail before running tests if a supported PTOAS compiler is unavailable.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register compilation coverage and fail CI before optional checks can skip."""
    config.addinivalue_line(
        "markers",
        "ptoas_compile: invokes PTOAS compilation to check generated-code regressions",
    )
    if not config.getoption("--require-ptoas"):
        return
    binary = find_ptoas_binary()
    if binary is None:
        raise pytest.UsageError("--require-ptoas requires PTOAS; set PTOAS_ROOT to its installation.")
    try:
        check_ptoas_version(binary)
    except RuntimeError as exc:
        raise pytest.UsageError(str(exc)) from exc


@pytest.fixture
def run_without_optional_runtime() -> Callable[[str], subprocess.CompletedProcess[str]]:
    """Run source in a fresh process that rejects optional runtime import attempts.

    The import finder covers statements and importlib alike. Tracking attempts
    also makes a caught ImportError fail the check. PyTorch backend autoload is
    disabled to test PyPTO's imports independently of installed torch plugins.
    """

    def run(source: str) -> subprocess.CompletedProcess[str]:
        """Execute source with import enforcement and return captured diagnostics."""
        bootstrap = f"""
import importlib.abc
import sys

blocked = ('torch_npu', 'simpler', 'simpler_setup', '_task_interface', 'pypto._torch_npu')

def is_blocked(name):
    'Match an optional runtime root or one of its submodules.'
    return any(name == root or name.startswith(root + '.') for root in blocked)

if any(is_blocked(name) for name in sys.modules):
    raise AssertionError('Optional runtime already loaded')

class RejectOptionalRuntime(importlib.abc.MetaPathFinder):
    'Record and reject optional imports before any loader runs.'
    def __init__(self):
        'Keep evidence even when the caller catches an import failure.'
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        'Reject optional runtimes and defer all other imports.'
        if is_blocked(fullname):
            self.attempts.append(fullname)
            raise ModuleNotFoundError('Blocked optional runtime import: ' + fullname, name=fullname)
        return None

guard = RejectOptionalRuntime()
sys.meta_path.insert(0, guard)
try:
    exec(compile({source!r}, '<optional-runtime-check>', 'exec'))
finally:
    if guard.attempts:
        raise AssertionError('Optional runtime imports attempted: ' + repr(guard.attempts))
    if any(is_blocked(name) for name in sys.modules):
        raise AssertionError('Optional runtime was loaded')
"""
        return subprocess.run(
            [sys.executable, "-c", bootstrap],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=os.environ | {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        )

    return run


@pytest.fixture
def device_runner(monkeypatch):
    """Import ``device_runner`` without requiring the optional Simpler package."""
    import pypto.runtime as runtime_package  # noqa: PLC0415

    fake_kernel_compiler = SimpleNamespace(KernelCompiler=object)
    fake_task_interface = SimpleNamespace(
        CallConfig=object,
        ChipCallable=object,
        CoreCallable=object,
        Worker=object,
    )
    monkeypatch.setitem(sys.modules, "pypto.runtime.kernel_compiler", fake_kernel_compiler)
    monkeypatch.setitem(sys.modules, "pypto.runtime.task_interface", fake_task_interface)

    module_name = "pypto.runtime.device_runner"
    previous = sys.modules.pop(module_name, None)
    previous_attribute = getattr(runtime_package, "device_runner", None)
    try:
        module = importlib.import_module(module_name)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous
        if previous_attribute is not None:
            setattr(runtime_package, "device_runner", previous_attribute)
        elif hasattr(runtime_package, "device_runner"):
            delattr(runtime_package, "device_runner")


@pytest.fixture
def stub_device_runner():
    """Bind the assembly layer to mocks so a test can dispatch without a device.

    Library code reaches ``pypto.runtime.device_runner`` through function-local
    ``from ... import`` statements, because importing it eagerly would make the
    device-only ``simpler`` package a hard dependency of ``pypto.runtime``.
    Patching an attribute on the real module therefore requires importing it
    first, which the unit-test runners cannot do; a stub module in
    ``sys.modules`` lets those lazy imports bind to mocks on every platform.

    This fixture is the one place in the test suite that names the assembly
    layer's private functions, so renaming one stays a single edit here rather
    than a sweep over every test that stubs a dispatch.

    Yields the stub module. ``_compile_and_assemble`` returns a
    ``(chip_callable, runtime_name, runtime_config)`` triple and
    ``_execute_on_device`` returns ``None``; assign ``return_value`` /
    ``side_effect`` on either mock to refine that.
    """
    stub = ModuleType("pypto.runtime.device_runner")
    stub._compile_and_assemble = MagicMock(  # type: ignore[attr-defined]
        name="_compile_and_assemble",
        return_value=(MagicMock(name="chip_callable"), "tensormap_and_ringbuffer", {}),
    )
    stub._execute_on_device = MagicMock(name="_execute_on_device", return_value=None)  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"pypto.runtime.device_runner": stub}):
        yield stub


@pytest.fixture
def ascend_backend(request):
    """Configure an Ascend backend for the duration of a test, then reset.

    Use either as a plain fixture (defaults to ``Ascend910``) or via
    ``pytest.mark.parametrize("ascend_backend", [...], indirect=True)`` to
    cycle through multiple backends. Replaces the per-test
    ``backend.reset_for_testing()`` + ``backend.set_backend_type(...)`` pair
    that is otherwise duplicated across pass / codegen tests.
    """
    backend_type = getattr(request, "param", BackendType.Ascend910)
    _backend.reset_for_testing()
    _backend.set_backend_type(backend_type)
    try:
        yield backend_type
    finally:
        _backend.reset_for_testing()


@pytest.fixture
def default_pass_manager():
    """Return the default-strategy PassManager.

    Use this in tests that want to run the production pipeline without
    constructing the manager inline. Strategy-specific tests should keep
    building the manager themselves so the choice stays visible at the
    test site.
    """
    return PassManager.get_strategy(OptimizationStrategy.Default)


@pytest.fixture(autouse=True)
def _reset_backend_singleton():
    """Reset the process-global backend singleton around every unit test.

    The backend type can only be set once per process (``set_backend_type``
    rejects a change once set). Tests that call ``ir.compile`` / codegen without
    explicitly choosing a backend rely on a clean singleton, so a sibling test
    that left e.g. ``Ascend950`` set would make them fail with "Backend type
    already set" — an order-dependent flake under pytest-xdist. Resetting before
    and after each test makes every test start from a clean slate regardless of
    scheduling. Tests that set a backend (inline or via ``ascend_backend``) are
    unaffected: the reset runs first, then their own set wins.
    """
    _backend.reset_for_testing()
    try:
        yield
    finally:
        _backend.reset_for_testing()


@pytest.fixture
def initial_log_level():
    """The C++ log level every unit test starts from (see ``_reset_log_level``)."""
    return _INITIAL_LOG_LEVEL


@pytest.fixture(autouse=True)
def _reset_log_level():
    """Pin the process-global C++ log level around every unit test.

    ``LOG_WARN`` / ``LOG_INFO`` are gated on a process-global threshold, so a
    sibling test that leaves it at ERROR or NONE silently empties the stderr
    that diagnostic assertions read — e.g. MemoryReuse's "fell back to the
    legacy packing" warning, whose assertion then fails against ``''``. That
    made the failure depend on test order rather than on the code under test.
    Restoring before and after each test makes those assertions
    order-independent regardless of scheduling. Tests that choose a level
    themselves are unaffected: the reset runs first, then their own set wins.

    The thread-local override is cleared too — a test that installs one on the
    main thread and fails before clearing it would suppress the same output
    through a different door.
    """
    _clear_thread_log_level()
    set_log_level(_INITIAL_LOG_LEVEL)
    try:
        yield
    finally:
        _clear_thread_log_level()
        set_log_level(_INITIAL_LOG_LEVEL)


@pytest.fixture(autouse=True)
def _redirect_prog_build_dir(tmp_path, monkeypatch):
    """Redirect ir.compile() / @pl.jit artifacts into pytest's per-test tmp dir.

    Without an explicit ``output_dir``, ``ir.compile()`` writes generated
    kernels and pass dumps to ``build_output/<name>_<timestamp>_<random>`` relative to
    the working directory. Under pytest that accumulates stale directories in
    the repo / build tree. Pointing ``PYPTO_PROG_BUILD_DIR`` at a
    ``build_output`` dir inside pytest's per-test ``tmp_path`` keeps every
    test's artifacts isolated and auto-cleaned by pytest.
    """
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path / "build_output"))


@pytest.fixture
def pass_verification_instruments():
    """Create shared pass verification and roundtrip instruments.

    The behavior is controlled by the PYPTO_VERIFY_LEVEL environment variable:

    - ``roundtrip`` (default) — BEFORE_AND_AFTER property verification + print→parse
      roundtrip structural-equality and structural-hash checks after every pass.
    - ``basic`` — BEFORE_AND_AFTER property verification only (faster, no roundtrip).
    - ``none`` — no pass verification at all (fastest, for debugging only).
    """
    level_str = os.environ.get("PYPTO_VERIFY_LEVEL", "roundtrip").lower()

    instruments: list[passes.PassInstrument] = []

    if level_str != "none":
        instruments.append(passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER))

    if level_str == "roundtrip":
        from pypto.ir.instruments import make_roundtrip_instrument  # noqa: PLC0415

        instruments.append(make_roundtrip_instrument())

    return instruments


@pytest.fixture(autouse=True)
def pass_verification_context(pass_verification_instruments):
    """Enable the configured checks for all pass executions."""
    with passes.PassContext(pass_verification_instruments):
        yield


@pytest.fixture
def kernel_env(monkeypatch, tmp_path):
    """Isolate the process manager and mock only the Worker/device boundary."""
    from pypto._artifact_contract import ArtifactExecutionMode, ExecutionCapabilities  # noqa: PLC0415
    from pypto.ir._artifact_metadata import ArtifactMetadata  # noqa: PLC0415
    from pypto.ir.compiled_artifact import CompiledArtifact  # noqa: PLC0415
    from pypto.runtime import _execution_mode  # noqa: PLC0415
    from pypto.runtime.kernel import context  # noqa: PLC0415
    from pypto.runtime.kernel.session import KernelConfig  # noqa: PLC0415

    monkeypatch.setattr(_execution_mode, "_gate", _execution_mode._ModeGate())
    calls = SimpleNamespace(workers=[], inits=[], prepares=[], closes=[])

    class FakeWorker:
        def __init__(self, config, state):
            calls.workers.append(self)

        def init(self, config):
            calls.inits.append(config)

        def check_context(self):
            pass

        def on_owner_thread(self):
            return False

        def prepare(self, callable_):
            calls.prepares.append(callable_)
            return object()  # Opaque handle; neither a slot nor convertible to int.

        def wait(self, ticket):
            ticket.wait()

        def close(self):
            calls.closes.append(self)

    monkeypatch.setattr(context, "_BorrowedWorkerSession", FakeWorker)
    state = context._ProcessKernelState()
    monkeypatch.setattr(context, "_process", SimpleNamespace(state=state))

    def artifact(key=b"kernel"):
        meta = ArtifactMetadata(
            (),
            (),
            0,
            "a2a3",
            BackendType.Ascend910,
            (),
            ExecutionCapabilities((ArtifactExecutionMode.KERNEL,)),
        )
        result = CompiledArtifact(tmp_path / key.hex(), "a2a3", meta)
        monkeypatch.setattr(
            result,
            "_artifact_runtime",
            SimpleNamespace(
                load=lambda: {".": (key, "tensormap_and_ringbuffer", {})},
                handle=SimpleNamespace(key=key),
            ),
        )
        return result

    return SimpleNamespace(
        state=state,
        config=KernelConfig("a2a3", "tensormap_and_ringbuffer", 0),
        calls=calls,
        worker_cls=FakeWorker,
        artifact=artifact,
    )


@pytest.fixture
def eager_env(monkeypatch):
    """Real initialization/session/admission with public runtime and Torch queue fakes."""
    import threading  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from pypto.runtime import _execution_mode  # noqa: PLC0415
    from pypto.runtime.kernel import arguments, context, session  # noqa: PLC0415
    from pypto.torch import init, interop, launch, shutdown  # noqa: PLC0415

    env = SimpleNamespace(
        events=[],
        queue=[],
        runs=[],
        buffers=[],
        workers=[],
        bridges=[],
        capture=False,
        queued=True,
        stream_id=7,
        rts_stream=12345,
        device=0,
        context=321,
        submit_error=None,
        register_error=None,
        enqueue_error=None,
    )

    class NPUTensor(torch.Tensor):
        def __getattribute__(self, name):
            # Keep real CPU storage operations; only the device label is fake.
            if name == "device":
                return SimpleNamespace(type="npu", index=0)
            return super().__getattribute__(name)

    env.tensor = lambda shape=(16, 16), dtype=torch.float32: torch.empty(shape, dtype=dtype).as_subclass(
        NPUTensor
    )

    def stream():
        return SimpleNamespace(stream_id=env.stream_id, device=SimpleNamespace(type="npu", index=env.device))

    framework = SimpleNamespace(
        __version__="2.6.0.post2",
        npu=SimpleNamespace(current_device=lambda: env.device, current_stream=lambda device: stream()),
        get_npu_format=lambda tensor: 2,
        _C=SimpleNamespace(_npu_shutdown=lambda: None, _npu_shutdown_synchronize=lambda: None),
    )

    class CallConfig:
        def __init__(self):
            self.runtime_env = SimpleNamespace()

        def validate(self):
            pass

    class RunHandle:
        def __init__(self):
            self.complete = False
            self.retired = False
            self.wait_error = None

        @property
        def done(self):
            return self.complete

        def wait(self):
            env.events.append("wait")
            if self.wait_error:
                raise self.wait_error
            self.complete = self.retired = True

    class Worker:
        device_location = property(lambda self: self.location)

        def __init__(self, **options):
            self.location = object()
            self.options = options
            env.workers.append(self)

        def init(self, prewarm_config=None, *, borrowed_context_generation=None):
            env.events.append("init")

        def register(self, callable_):
            env.events.append("register")
            if env.register_error:
                raise env.register_error
            return object()

        def unregister(self, handle):
            pytest.fail("no eviction")

        def submit(self, target, args=None, config=None, *, caller_stream=None):
            env.events.append(("submit", target, args, config, caller_stream, threading.get_ident()))
            if env.submit_error:
                raise env.submit_error
            run = RunHandle()
            env.runs.append(run)
            return run

        def flush_diagnostics(self):
            pytest.fail("no implicit DFX flush")

        def close(self):
            env.events.append("close")

    class Buffer:
        def __init__(self, address, nbytes, location):
            self.address, self.nbytes, self.location = address, nbytes, location
            self.closed = False
            self.close_error = None
            env.buffers.append(self)

        @classmethod
        def wrap(cls, *, address, nbytes, location):
            assert not any(
                not b.closed and b.address < address + nbytes and address < b.address + b.nbytes
                for b in env.buffers
            ), "overlapping live Buffer.wrap"
            return cls(address, nbytes, location)

        def close(self):
            env.events.append("source.close")
            if self.close_error:
                raise self.close_error
            self.closed = True

    class TaskArgs:
        def __init__(self):
            self.tensors, self.scalars = [], []

        def add_tensor(self, tensor, tag, *, transfer):
            assert not self.scalars and transfer == "NONE"
            self.tensors.append((tensor, tag))

        def add_scalar(self, scalar):
            self.scalars.append(scalar)

    api = SimpleNamespace(
        Worker=Worker,
        CallConfig=CallConfig,
        RunHandle=RunHandle,
        Buffer=Buffer,
        TaskArgs=TaskArgs,
        Tensor=lambda buffer, **kw: SimpleNamespace(buffer=buffer, **kw),
        TensorArgType=SimpleNamespace(INPUT="IN", OUTPUT_EXISTING="OUT", INOUT="INOUT"),
        TensorTransfer=SimpleNamespace(NONE="NONE"),
    )

    def run_queue():
        while env.queue:
            env.queue.pop(0)()

    class QueueTicket:
        def __init__(self, tensors, stream_id, device):
            self.tensors, self.stream_id, self.rts_stream = tensors, stream_id, env.rts_stream
            env.bridges.append(self)

        def enqueue(self, callback):
            env.events.append("recordStream")
            if env.enqueue_error:
                raise env.enqueue_error
            if env.queued:
                env.queue.append(lambda: callback(self.rts_stream))
            else:
                callback(self.rts_stream)

        def synchronize(self):
            env.events.append("synchronize")
            run_queue()

        def release(self):
            env.events.append("native.release")
            self.tensors = []

    native = SimpleNamespace(
        prepare=QueueTicket,
        check_call=lambda stream, device: env.capture,
        borrow_context=lambda device: env.context,
        bind_context=lambda context: None,
        framework_alive=lambda: True,
        retain_until_exit=lambda owner: None,
    )
    for module in ("simpler", "simpler.task_interface", "simpler.worker"):
        monkeypatch.setitem(sys.modules, module, api)
    monkeypatch.setattr(session, "check_runtime_pin", lambda: None)
    monkeypatch.setattr(arguments, "check_runtime_pin", lambda: None)
    monkeypatch.setattr(arguments, "_tensor_dtype", str)
    for module in (interop, shutdown):
        monkeypatch.setattr(module, "_load_torch_npu", lambda: framework)
    for module in (launch, shutdown):
        monkeypatch.setattr(module, "_load_native", lambda: native)
    monkeypatch.setattr(_execution_mode, "_gate", _execution_mode._ModeGate())
    env.state = context._ProcessKernelState()
    monkeypatch.setattr(context, "_process", SimpleNamespace(state=env.state))
    env.init, env.run_queue, env.api, env.native, env.framework = init, run_queue, api, native, framework
    yield env
    # Tests of unknown completion intentionally retain owners in production.
    # Drop only the fake state and stop its thread, without pretending it closed.
    owner = env.state._worker
    if owner is not None:
        owner._owner.stop()
