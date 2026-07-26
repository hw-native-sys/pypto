# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for function-aware PTOAS sync-summary comparison."""

import json

import pytest
from pypto.tools import ptoas_sync_summary


def _record(function: str, groups: int, pair_groups: dict[str, int]) -> dict:
    return {
        "schema_version": 1,
        "status": "analyzed",
        "function": function,
        "active_sync_groups": groups,
        "active_sync_operations": groups * 2,
        "pipe_pair_groups": pair_groups,
        "pipe_pair_event_ids": {},
    }


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_load_and_diff_match_functions_instead_of_row_order(tmp_path):
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(
        baseline_path,
        [
            _record("mixed_aic", 10, {"PIPE_M->PIPE_MTE1": 2}),
            _record("mixed_aiv", 20, {"PIPE_V->PIPE_MTE2": 3}),
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            _record("mixed_aiv", 18, {"PIPE_V->PIPE_MTE2": 1}),
            _record("mixed_aic", 11, {"PIPE_M->PIPE_MTE1": 3}),
        ],
    )

    delta = ptoas_sync_summary.diff_sync_summaries(
        ptoas_sync_summary.load_sync_summaries(baseline_path),
        ptoas_sync_summary.load_sync_summaries(candidate_path),
    )

    assert delta["mixed_aic"]["metrics"]["active_sync_groups"] == 1
    assert delta["mixed_aic"]["pipe_pair_groups"]["PIPE_M->PIPE_MTE1"] == 1
    assert delta["mixed_aiv"]["metrics"]["active_sync_groups"] == -2
    assert delta["mixed_aiv"]["pipe_pair_groups"]["PIPE_V->PIPE_MTE2"] == -2


def test_diff_rejects_function_set_mismatch():
    with pytest.raises(ValueError, match="missing=\\['aic'\\].*added=\\['other'\\]"):
        ptoas_sync_summary.diff_sync_summaries(
            {"aic": _record("aic", 1, {})},
            {"other": _record("other", 1, {})},
        )


def test_load_rejects_duplicate_function(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    _write_jsonl(path, [_record("aiv", 1, {}), _record("aiv", 2, {})])
    with pytest.raises(ValueError, match="duplicate function 'aiv'"):
        ptoas_sync_summary.load_sync_summaries(path)


def test_load_rejects_invalid_records(tmp_path):
    bad_json = tmp_path / "bad-json.jsonl"
    bad_json.write_text("{broken\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        ptoas_sync_summary.load_sync_summaries(bad_json)

    missing_function = tmp_path / "missing-function.jsonl"
    missing_function.write_text('{"schema_version": 1}\n')
    with pytest.raises(ValueError, match="missing non-empty 'function'"):
        ptoas_sync_summary.load_sync_summaries(missing_function)


def test_main_writes_function_keyed_json(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "diff.json"
    _write_jsonl(baseline, [_record("aiv", 3, {})])
    _write_jsonl(candidate, [_record("aiv", 2, {})])

    assert ptoas_sync_summary.main([str(baseline), str(candidate), "-o", str(output)]) == 0
    assert json.loads(output.read_text())["aiv"]["metrics"]["active_sync_groups"] == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
