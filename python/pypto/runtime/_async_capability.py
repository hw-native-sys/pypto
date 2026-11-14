# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiler completion metadata shared by generated and prebuilt artifacts.

These fields describe compiler-declared waiter capacity, not resource ownership
or cache cleanliness. They remain part of artifact identity; current Simpler
CoreCallable registration accepts only signature and binary. Notification effects
are exported separately by the generated orchestration metadata provider.
"""

from typing import Any, cast


def validate_async_capability(value: Any) -> dict[str, Any] | None:
    """Validate the JSON form of Simpler's version-1 completion descriptor."""
    if value is None:
        return None
    fields = {"version", "mode", "max_conditions", "condition_types"}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid callable async capability fields: {value!r}")
    mapping = cast("dict[str, Any]", value)
    if set(mapping) != fields:
        raise ValueError(f"Invalid callable async capability fields: {mapping!r}")
    version, mode = mapping["version"], mapping["mode"]
    count, types = mapping["max_conditions"], mapping["condition_types"]
    if type(version) is not int or version != 1:
        raise ValueError(f"Unsupported callable async capability version: {version!r}")
    if type(mode) is not str or mode not in {"UNKNOWN", "SYNCHRONOUS", "DEFERRED"}:
        raise ValueError(f"Invalid callable async capability mode: {mode!r}")
    if type(count) is not int or type(types) is not int:
        raise ValueError(f"Callable async capacity/types must be integers: {count!r}, {types!r}")
    valid = (1 <= count <= 64 and 0 < types <= 3) if mode == "DEFERRED" else (count == types == 0)
    if not valid:
        raise ValueError(f"Invalid {mode} callable async capacity/types: {count!r}, {types!r}")
    return {"version": version, "mode": mode, "max_conditions": count, "condition_types": types}


def async_capability_key(value: Any) -> tuple[int, str, int, int] | None:
    """Return a stable identity component without importing native bindings."""
    capability = validate_async_capability(value)
    if capability is None:
        return None
    return (
        capability["version"],
        capability["mode"],
        capability["max_conditions"],
        capability["condition_types"],
    )
