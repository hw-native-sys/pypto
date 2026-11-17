# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Split the borrowed cards into groups, one per band of xdist workers.

`tests/st/distributed` serializes its device phase per card set (see
``card_lock``), so one card set is one lane: the workers sharing it take turns on
the cards while the rest of them compile. Borrowing two card sets doubles the
lanes, and because the lock is named after the cards it protects, the lanes never
meet.

Assigning them is all this module does. Workers are dealt to groups round-robin,
so ``-n 6`` over two groups gives each group three workers -- the width measured
to keep one set of cards ~95% busy. Which tests land where is left to xdist's own
`--dist load`, which hands the next test to whichever worker is free; splitting
the suite by hand instead cost 25% to imbalance at both two and three ways.

Off by default (`group_size=0`): every caller then sees the whole `--device`
list, which is what a developer running one card set expects, and what the
multi-card tests outside `tests/st/distributed` need.
"""

import os
from collections.abc import Sequence


def partition(device_ids: Sequence[int], group_size: int) -> list[list[int]]:
    """Split *device_ids* into consecutive groups of *group_size*.

    A trailing remainder too small to run anything is dropped rather than handed
    out: a group of one cannot serve a two-rank test, and a worker assigned to it
    would skip every test it was given.
    """
    if group_size <= 0:
        return [list(device_ids)]
    groups = [list(device_ids[i : i + group_size]) for i in range(0, len(device_ids), group_size)]
    full = [group for group in groups if len(group) == group_size]
    return full or [list(device_ids)]


def worker_index() -> int:
    """This xdist worker's ordinal, or 0 when not running under xdist."""
    name = os.environ.get("PYTEST_XDIST_WORKER", "")
    digits = name[2:] if name.startswith("gw") else ""
    return int(digits) if digits.isdigit() else 0


def group_for_worker(device_ids: Sequence[int], group_size: int) -> list[int]:
    """Return the cards this worker owns; the whole list when grouping is off."""
    groups = partition(device_ids, group_size)
    return groups[worker_index() % len(groups)]
