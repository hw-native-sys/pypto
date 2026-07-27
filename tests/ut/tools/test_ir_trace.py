# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for IR pass snapshot discovery."""

from pathlib import Path

import pytest
from pypto.tools.ir_trace.discovery import discover_snapshots
from pypto.tools.ir_trace.model import IRTraceError


def _write_dump(root: Path, files: dict[str, str]) -> Path:
    dump = root / "passes_dump"
    dump.mkdir()
    for name, text in files.items():
        (dump / name).write_text(text, encoding="utf-8")
    return dump


def test_discover_orders_snapshots_and_attaches_warning(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "02_after_UnrollLoops.py": "after two\n",
            "00_frontend.py": "frontend\n",
            "01_after_InlineFunctions.log": "unused variable\n",
            "01_after_InlineFunctions.py": "after one\n",
            "fa_fused_EXTRACT.py": "ignored\n",
        },
    )

    snapshots = discover_snapshots(dump)

    assert [snapshot.index for snapshot in snapshots] == [0, 1, 2]
    assert [snapshot.pass_name for snapshot in snapshots] == [None, "InlineFunctions", "UnrollLoops"]
    assert snapshots[1].warning_text == "unused variable\n"
    assert snapshots[2].warning_text is None


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("missing_directory", "does not exist"),
        ("not_directory", "not a directory"),
        ("missing_frontend", "00_frontend.py"),
        ("no_pass_snapshots", "no pass snapshots"),
        ("starts_at_two", "01"),
        ("index_gap", "02"),
        ("duplicate_index", "01"),
        ("malformed_name", "02_ConvertToSSA.py"),
        ("invalid_snapshot_utf8", "01_after_InlineFunctions.py"),
        ("invalid_warning_utf8", "01_after_InlineFunctions.log"),
    ],
)
def test_discover_rejects_invalid_dump_inputs(tmp_path: Path, case: str, expected_message: str):
    if case == "missing_directory":
        dump = tmp_path / "missing"
    elif case == "not_directory":
        dump = tmp_path / "not-a-directory"
        dump.write_text("not a directory", encoding="utf-8")
    elif case == "missing_frontend":
        dump = _write_dump(tmp_path, {"01_after_InlineFunctions.py": "after one\n"})
    elif case == "no_pass_snapshots":
        dump = _write_dump(tmp_path, {"00_frontend.py": "frontend\n"})
    elif case == "starts_at_two":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "02_after_UnrollLoops.py": "after two\n",
            },
        )
    elif case == "index_gap":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "01_after_InlineFunctions.py": "after one\n",
                "03_after_ConvertToSSA.py": "after three\n",
            },
        )
    elif case == "duplicate_index":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "01_after_InlineFunctions.py": "after one\n",
                "01_after_UnrollLoops.py": "also after one\n",
            },
        )
    elif case == "malformed_name":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "02_ConvertToSSA.py": "malformed\n",
            },
        )
    elif case == "invalid_snapshot_utf8":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "01_after_InlineFunctions.py": "after one\n",
            },
        )
        (dump / "01_after_InlineFunctions.py").write_bytes(b"\xff")
    else:
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "01_after_InlineFunctions.py": "after one\n",
            },
        )
        (dump / "01_after_InlineFunctions.log").write_bytes(b"\xff")

    with pytest.raises(IRTraceError, match=expected_message):
        discover_snapshots(dump)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
