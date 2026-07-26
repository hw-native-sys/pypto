# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for raw DSA reuse-candidate inspection."""

import json

import pytest
from pypto.tools import dsa_reuse_candidates

_PREFIX = "0,1,0->1,ub->external@outbound_dma=>external->ub@inbound_dma,arenas=Vec->Vec,write_after_read"


def _record(ordering: str, hazard: str, path: str) -> str:
    return (
        f"{_PREFIX},{ordering},inter_operation,full_allocation,complete_access_set,"
        "verified_initial_write,in_loop,distance_0,sites=3->7,ranges=0+32->0+32,"
        f"hazard={hazard},dag_path={path}"
    )


def test_parse_ordered_candidate_with_dependency_witness():
    record = dsa_reuse_candidates.parse_candidate_record(
        _record("logical_order", "cross_resource", "r0s3>r0s5>r0s7")
    )
    assert record.first_buffer == 0 and record.second_buffer == 1
    assert record.prior_buffer == 0 and record.next_buffer == 1
    assert record.prior_route == "ub->external@outbound_dma"
    assert record.next_route == "external->ub@inbound_dma"
    assert record.dependence == "write_after_read"
    assert record.ordered_by_logical_dag
    assert record.hazard == "cross_resource"
    assert record.dag_path == ("r0s3", "r0s5", "r0s7")


def test_parse_rejects_inconsistent_ordering_evidence():
    with pytest.raises(ValueError, match="requires a dependency path"):
        dsa_reuse_candidates.parse_candidate_record(_record("logical_order", "cross_resource", "none"))
    with pytest.raises(ValueError, match="must use dag_path=none"):
        dsa_reuse_candidates.parse_candidate_record(
            _record("no_logical_order", "cross_resource", "r0s3>r0s7")
        )


def test_load_checks_candidate_count(tmp_path):
    path = tmp_path / "problem.dsa.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "recognized_reuse_candidates": "2",
                    "recognized_reuse_candidate_records_v4": _record(
                        "no_logical_order", "same_resource", "none"
                    ),
                }
            }
        )
    )
    with pytest.raises(ValueError, match="candidate count mismatch"):
        dsa_reuse_candidates.load_candidate_records(path)


def test_main_filters_raw_candidates(tmp_path, capsys):
    path = tmp_path / "problem.dsa.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "recognized_reuse_candidates": "2",
                    "recognized_reuse_candidate_records_v4": ";".join(
                        [
                            _record("logical_order", "cross_resource", "r0s3>r0s7"),
                            _record("no_logical_order", "same_resource", "none"),
                        ]
                    ),
                }
            }
        )
    )

    assert dsa_reuse_candidates.main([str(path), "--hazard", "cross_resource", "--ordering", "ordered"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result) == 1
    assert result[0]["hazard"] == "cross_resource"
    assert result[0]["ordered_by_logical_dag"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
