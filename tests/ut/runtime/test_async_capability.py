# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Completion descriptors remain validated compiler artifact metadata."""

import pytest
from pypto.runtime._async_capability import (
    async_capability_key,
    validate_async_capability,
)


def _descriptor(**changes):
    return dict(version=1, mode="DEFERRED", max_conditions=64, condition_types=1, **changes)


@pytest.mark.parametrize("count", [1, 63, 64])
def test_artifact_identity_preserves_capacity_and_type(count):
    value = {**_descriptor(), "max_conditions": count}
    assert validate_async_capability(value) == value
    assert async_capability_key(value) == (1, "DEFERRED", count, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 0},
        {"version": True},
        {"mode": "sync"},
        {"mode": []},
        {"max_conditions": 0},
        {"max_conditions": 65},
        {"max_conditions": True},
        {"condition_types": 0},
        {"condition_types": 4},
        {"condition_types": -1},
        {"condition_types": True},
        {"mode": "SYNCHRONOUS"},
        {"mode": "UNKNOWN"},
        {"extra": 1},
    ],
)
def test_invalid_descriptor_is_not_normalized_into_success(changes):
    with pytest.raises(ValueError):
        validate_async_capability({**_descriptor(), **changes})


def test_missing_metadata_stays_unqualified():
    assert validate_async_capability(None) is None
    assert async_capability_key(None) is None


def test_capability_identity_distinguishes_mode_capacity_and_type():
    original = async_capability_key(_descriptor())
    for changes in (
        {"max_conditions": 63},
        {"condition_types": 2},
        {"mode": "SYNCHRONOUS", "max_conditions": 0, "condition_types": 0},
    ):
        assert async_capability_key({**_descriptor(), **changes}) != original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
