# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the AllToAllV benchmark harness's pure count helpers.

The harness lives at ``tests/st/distributed/collectives/all_to_all_v_benchmark.py``
and is an explicitly-invoked benchmark, not a pytest test. These tests cover the
pure, device-free helpers it exposes — specifically that the ``asymmetric``
count pattern stays asymmetric when ``routes_per_peer == 1`` (the ``full == 1``
case): the naive ``(3r + 5d + full) % (full + 1)`` formula collapses to the
symmetric ``(r + d + 1) % 2`` there, which leaves only self-copy and zero
remote payload.
"""

import sys
from pathlib import Path

import pytest

# tests/ut/conftest.py intentionally does not add the repo root; add it here so
# we can import the benchmark's pure helpers (whose top-level imports are stdlib
# + torch only, so this is device-free). Mirrors test_phase_fence_dep_compression.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.st.distributed.collectives.all_to_all_v_benchmark import (  # noqa: E402
    make_send_counts,
    shape_for_peer_bytes,
)


@pytest.mark.parametrize("peer_bytes", [32, 64, 1024, 2048])
def test_asymmetric_pattern_full_one_has_remote_asymmetry(peer_bytes: int) -> None:
    """``full == 1`` must still produce an asymmetric, nonzero remote payload.

    These payloads are non-multiples of ``CANONICAL_C``, so they compile a
    ``C' = peer_bytes`` instance with ``routes_per_peer == 1``.
    """
    shape = shape_for_peer_bytes(p=2, peer_bytes=peer_bytes)
    assert shape.routes_per_peer == 1
    counts = make_send_counts("asymmetric", shape)

    remote = [(a, b) for a in range(shape.p) for b in range(shape.p) if a != b]
    assert any(int(counts[a, b, 0].item()) != int(counts[b, a, 0].item()) for a, b in remote)
    assert any(int(counts[a, b, 0].item()) > 0 for a, b in remote)


def test_asymmetric_pattern_full_ge_two_stays_asymmetric() -> None:
    """Control: the existing formula for ``full >= 2`` remains asymmetric."""
    shape = shape_for_peer_bytes(p=4, peer_bytes=2 * 4160)
    assert shape.routes_per_peer == 2
    counts = make_send_counts("asymmetric", shape)

    remote = [(a, b) for a in range(shape.p) for b in range(shape.p) if a != b]
    assert any(int(counts[a, b, 0].item()) != int(counts[b, a, 0].item()) for a, b in remote)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
