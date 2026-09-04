# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reading a chip-swimlane record.

The on-disk dump is not the view a test wants to assert on. Since runtime JSON
v2 (simpler #985) the host writes raw cycle-domain streams — ``aicore_tasks`` /
``aicpu_tasks`` alongside a ``metadata`` block — and the unified ``tasks`` list,
with cycles converted to microseconds and the AICore/AICPU sides joined, is
rebuilt in Python by the runtime's own converter.

Going through that converter is what keeps a test asserting on the view the
profiling tools present, rather than on whichever shape the host happens to
serialise this month. Tests that read the file directly did not survive the v2
change: they raised ``KeyError: 'tasks'`` the first time they ran against it.
"""

import json
from pathlib import Path
from typing import Any


def read_swimlane(path: "str | Path") -> dict[str, Any]:
    """Return the unified swimlane view for the record at *path*.

    Args:
        path: A ``chip_swimlane_records.json`` written under a run's
            ``dfx_outputs/``.

    Returns:
        The converter's view: ``chip_swimlane_level``, ``metadata``, and a
        ``tasks`` list in the microsecond domain.

    Raises:
        AssertionError: The converter produced no ``tasks`` key, which means the
            record is empty or its schema moved again.
    """
    from simpler_setup.tools.swimlane_converter import read_perf_data  # noqa: PLC0415

    data = read_perf_data(str(path))
    assert "tasks" in data, (
        f"swimlane record {path} produced no 'tasks' view; on-disk keys were "
        f"{sorted(json.loads(Path(path).read_text()))}"
    )
    return data


__all__ = ["read_swimlane"]
