# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared enum parsing utilities for Level and Role extraction from AST."""

import ast
from typing import Any

from pypto.pypto_core import ir

from .diagnostics import ParserSyntaxError

LEVEL_MAP: dict[str, ir.Level] = {
    "AIV": ir.Level.AIV,
    "AIC": ir.Level.AIC,
    "CORE_GROUP": ir.Level.CORE_GROUP,
    "CHIP_DIE": ir.Level.CHIP_DIE,
    "CHIP": ir.Level.CHIP,
    "HOST": ir.Level.HOST,
    "CLUSTER_0": ir.Level.CLUSTER_0,
    "CLUSTER_1": ir.Level.CLUSTER_1,
    "CLUSTER_2": ir.Level.CLUSTER_2,
    "GLOBAL": ir.Level.GLOBAL,
    # Readability aliases
    "L2CACHE": ir.Level.L2CACHE,
    "PROCESSOR": ir.Level.PROCESSOR,
    "UMA": ir.Level.UMA,
    "NODE": ir.Level.NODE,
    "POD": ir.Level.POD,
    "CLOS1": ir.Level.CLOS1,
    "CLOS2": ir.Level.CLOS2,
}

ROLE_MAP: dict[str, ir.Role] = {
    "Orchestrator": ir.Role.Orchestrator,
    "SubWorker": ir.Role.SubWorker,
}

SPLIT_MODE_MAP: dict[str, ir.SplitMode] = {
    "NONE": ir.SplitMode.NONE,
    "UP_DOWN": ir.SplitMode.UP_DOWN,
    "LEFT_RIGHT": ir.SplitMode.LEFT_RIGHT,
}

LOOP_ORIGIN_MAP: dict[str, ir.LoopOrigin] = {
    "Original": ir.LoopOrigin.Original,
    "ChunkOuter": ir.LoopOrigin.ChunkOuter,
    "ChunkInner": ir.LoopOrigin.ChunkInner,
    "ChunkRemainder": ir.LoopOrigin.ChunkRemainder,
}

FUNCTION_TYPE_MAP: dict[str, ir.FunctionType] = {
    "Opaque": ir.FunctionType.Opaque,
    "Orchestration": ir.FunctionType.Orchestration,
    "InCore": ir.FunctionType.InCore,
    "AIC": ir.FunctionType.AIC,
    "AIV": ir.FunctionType.AIV,
    "Group": ir.FunctionType.Group,
    "Spmd": ir.FunctionType.Spmd,
    "Inline": ir.FunctionType.Inline,
}


def extract_enum_value(
    value: ast.expr,
    enum_map: dict[str, Any],
    enum_name: str,
    qualified: str,
) -> Any:
    """Extract enum value from AST: pl.Level.HOST or Level.HOST.

    Args:
        value: AST expression node
        enum_map: Mapping from attribute name to enum value
        enum_name: Enum class name (e.g., "Level")
        qualified: Qualified name for error messages (e.g., "pl.Level")

    Returns:
        Enum value from enum_map
    """
    if not isinstance(value, ast.Attribute):
        raise ParserSyntaxError(
            f"Expected {qualified}.<name>",
            hint=f"Use {qualified}.<name>.",
        )
    if value.attr not in enum_map:
        raise ParserSyntaxError(
            f"Unknown {enum_name} value: {value.attr}",
            hint=f"Valid values: {', '.join(enum_map.keys())}",
        )
    # Check prefix: Level.X
    if isinstance(value.value, ast.Name) and value.value.id == enum_name:
        return enum_map[value.attr]
    # Check prefix: pl.Level.X
    if (
        isinstance(value.value, ast.Attribute)
        and isinstance(value.value.value, ast.Name)
        and value.value.value.id == "pl"
        and value.value.attr == enum_name
    ):
        return enum_map[value.attr]
    raise ParserSyntaxError(f"Expected {qualified}.<name>")


__all__ = [
    "LEVEL_MAP",
    "LOOP_ORIGIN_MAP",
    "ROLE_MAP",
    "SPLIT_MODE_MAP",
    "FUNCTION_TYPE_MAP",
    "extract_enum_value",
]
