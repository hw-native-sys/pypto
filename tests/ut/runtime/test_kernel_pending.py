# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Prepared submission retirement, bounded observation and cancellation."""

from types import SimpleNamespace

import pytest
from pypto.runtime.kernel.callable import KernelRegistration
from pypto.runtime.kernel.pending import PendingTicket


def _ticket(submission, events, *, synchronize=lambda: None):
    def enqueue(ticket):
        ticket.submission = submission
        events.append("enqueue")

    return PendingTicket(
        KernelRegistration("test", 0, 0, None, None, None, None),
        enqueue=enqueue,
        synchronize=synchronize,
        release=lambda: events.append("release"),
        owners=(object(),),
    )


def test_pending_admission_is_nonblocking_and_enqueue_is_one_shot():
    events = []
    submission = SimpleNamespace(done=False, safe_to_release=False)

    def wait(timeout=None):
        events.append(("wait", timeout))
        submission.safe_to_release = True

    submission.wait = wait
    ticket = _ticket(submission, events, synchronize=lambda: events.append("sync"))
    ticket.enqueue()
    assert not ticket.done() and events == ["enqueue"]
    with pytest.raises(RuntimeError, match="once"):
        ticket.enqueue()
    ticket.wait()
    assert events == ["enqueue", "sync", ("wait", None), "release"]
    assert ticket.retired and not ticket._owners
    ticket.wait()
    assert ticket.done() and events.count("release") == 1


def test_completed_submission_uses_bounded_observation():
    events = []
    submission = SimpleNamespace(done=True, safe_to_release=False)

    def wait(timeout=None):
        assert timeout == 0
        events.append("wait")
        submission.safe_to_release = True

    submission.wait = wait
    ticket = _ticket(submission, events, synchronize=lambda: pytest.fail("poll must not synchronize"))
    ticket.enqueue()
    assert ticket.done()
    assert events == ["enqueue", "wait", "release"]


@pytest.mark.parametrize("safe", [False, True])
def test_error_release_follows_native_cleanup_proof(safe):
    events = []
    error = ValueError("original execution error")
    submission = SimpleNamespace(done=True, safe_to_release=safe)

    def wait(timeout=None):
        raise error

    submission.wait = wait
    ticket = _ticket(submission, events)
    ticket.enqueue()
    with pytest.raises(ValueError) as caught:
        ticket.done()
    assert caught.value is error
    assert ticket.retired is safe
    assert bool(ticket._owners) is not safe
    assert ("release" in events) is safe


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
