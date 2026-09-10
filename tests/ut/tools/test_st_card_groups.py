# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for splitting borrowed cards into per-worker lanes.

The `device_ids` fixture and the distributed device lock both derive a worker's
cards from here, and they must agree: a worker that locks one group while
dispatching on another would drive cards nobody is holding.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cards(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "st" / "harness"
    spec = importlib.util.spec_from_file_location("st_cards", directory / "cards.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_grouping_off_hands_out_the_whole_device_list(cards):
    """The default, and what a single-group run and the non-dist tests expect."""
    assert cards.partition([1, 3, 5, 7], 0) == [[1, 3, 5, 7]]
    assert cards.group_for_worker([1, 3, 5, 7], 0) == [1, 3, 5, 7]


def test_cards_split_into_consecutive_groups(cards):
    assert cards.partition([1, 3, 5, 7], 2) == [[1, 3], [5, 7]]
    assert cards.partition([1, 3, 5, 7, 9, 11], 2) == [[1, 3], [5, 7], [9, 11]]
    assert cards.partition([1, 3, 5, 7], 4) == [[1, 3, 5, 7]]


def test_a_remainder_too_small_to_run_anything_is_dropped(cards):
    """A worker given one card would skip every two-rank test it was handed."""
    assert cards.partition([1, 3, 5], 2) == [[1, 3]]


def test_a_device_list_smaller_than_one_group_is_kept_whole(cards):
    """Better to run one under-sized lane than to hand every worker nothing."""
    assert cards.partition([1], 2) == [[1]]
    assert cards.partition([1, 3], 4) == [[1, 3]]


@pytest.mark.parametrize(
    ("worker", "expected"),
    [("gw0", [1, 3]), ("gw1", [5, 7]), ("gw2", [1, 3]), ("gw3", [5, 7])],
)
def test_workers_are_dealt_to_groups_round_robin(cards, monkeypatch, worker, expected):
    """`-n 4` over two groups gives each group two workers, in order."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", worker)
    assert cards.group_for_worker([1, 3, 5, 7], 2) == expected


@pytest.mark.parametrize("name", ["", "master", "notaworker"])
def test_a_run_without_xdist_takes_the_first_group(cards, monkeypatch, name):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", name)
    assert cards.worker_index() == 0
    assert cards.group_for_worker([1, 3, 5, 7], 2) == [1, 3]


def test_every_worker_lands_on_a_real_group(cards, monkeypatch):
    """The modulo must never index past the groups that exist."""
    for i in range(12):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", f"gw{i}")
        assert cards.group_for_worker([1, 3, 5, 7, 9, 11], 2) in ([1, 3], [5, 7], [9, 11])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
