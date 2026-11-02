# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Delayed callback publication and RunHandle retirement before source release."""

import threading
from types import SimpleNamespace

import pytest
from pypto.runtime.kernel.context import KernelState
from pypto.runtime.kernel.pending import PendingTicket


def test_pending_callback_does_not_wait_during_admission(kernel_env):
    env = kernel_env
    env.state.ensure_worker(env.config)
    registration = env.state.ensure_callable(env.artifact(), env.config)
    events = []
    run = SimpleNamespace(done=False, wait=lambda: events.append("wait"))

    def synchronize():
        events.append("sync")
        thread = threading.Thread(target=lambda: ticket.publish(run))
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

    ticket = PendingTicket(
        registration,
        enqueue=lambda t: events.append("enqueue"),
        synchronize=synchronize,
        release=lambda: events.append("release"),
        owners=(object(),),
    )
    env.state.submit(registration, lambda worker: ticket)
    assert events == ["enqueue"] and not ticket.done()
    env.state.close()
    assert events == ["enqueue", "sync", "wait", "release"]
    assert env.state.state is KernelState.CLOSED and ticket._owners == ()


def test_done_retires_run_before_source_release(kernel_env):
    env = kernel_env
    env.state.ensure_worker(env.config)
    registration = env.state.ensure_callable(env.artifact(), env.config)
    events = []
    run = SimpleNamespace(done=True, wait=lambda: events.append("retire"))
    ticket = PendingTicket(
        registration,
        enqueue=lambda t: t.publish(run),
        synchronize=lambda: pytest.fail("ordinary admission must not synchronize"),
        release=lambda: events.append("release"),
        owners=(object(),),
    )
    env.state.submit(registration, lambda worker: ticket)
    assert ticket.done() and ticket.done()
    assert events == ["retire", "release"]
    env.state.close()


@pytest.mark.parametrize("failure", ["enqueue", "callback", "wait", "release", "no_callback"])
def test_failure_retains_sources_registration_and_worker(kernel_env, failure):
    env = kernel_env
    env.state.ensure_worker(env.config)
    registration = env.state.ensure_callable(env.artifact(), env.config)
    source = object()
    events = []
    error = RuntimeError(failure)

    def fail():
        raise error

    def enqueue(ticket):
        if failure == "enqueue":
            fail()
        elif failure == "callback":
            ticket.fail(error)
        elif failure != "no_callback":
            ticket.publish(SimpleNamespace(done=True, wait=fail if failure == "wait" else lambda: None))

    ticket = PendingTicket(
        registration,
        enqueue=enqueue,
        synchronize=lambda: events.append("sync"),
        release=fail if failure == "release" else lambda: events.append("release"),
        owners=(source,),
    )
    if failure == "enqueue":
        with pytest.raises(RuntimeError) as caught:
            env.state.submit(registration, lambda worker: ticket)
        assert caught.value is error
    else:
        env.state.submit(registration, lambda worker: ticket)
    with pytest.raises(RuntimeError):
        env.state.close()
    assert env.state.state is KernelState.FAILED
    assert ticket._owners == (source,) and not env.calls.closes
    assert env.state._submissions == [ticket]
    assert env.state._registrations[registration.identity] is registration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
