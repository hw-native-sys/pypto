# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Composable body AST for fuzz kernel generation.

Represents the kernel body as a tree of composable blocks that can be
arbitrarily nested and sequenced. This enables flexible control flow
generation (for loops, if/else) at any nesting depth.

Example body structures::

    # Plain ops
    [OpBlock(op_chain=[...])]

    # Sequential: ops -> if/else -> ops
    [OpBlock(...), IfElseBlock(...), OpBlock(...)]

    # For loop containing if/else
    [ForBlock(body=[IfElseBlock(...)])]

    # Nested: for -> if/else -> for
    [ForBlock(body=[IfElseBlock(
        then_body=[ForBlock(body=[OpBlock(...)])],
        else_body=[OpBlock(...)],
    )])]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpBlock:
    """A sequence of plain operations (no control flow).

    Attributes:
        op_chain: List of op dicts from OpFuzzer.
        scalar_value_to_param: Mapping from scalar literal values to
            parameter names (e.g., {"2.0": "scalar_0"}).
    """

    op_chain: list[dict[str, Any]]
    scalar_value_to_param: dict[str, str] = field(default_factory=dict)


@dataclass
class ForBlock:
    """A for-loop wrapping a composable body.

    Represents ``for i in pl.range(iterations): body`` with optional
    accumulation (cross-iteration dependency via ``scf.for`` iter_arg).

    Attributes:
        iterations: Number of loop iterations.
        body: Composable body nodes inside the loop.
        accum_var: Variable to accumulate into across iterations (or None).
        last_body_output: Variable name of the body's last output,
            used for the accumulation line ``accum_var = pl.add(last, accum_var)``.
    """

    iterations: int
    body: list[BodyNode]
    accum_var: str | None = None
    last_body_output: str | None = None


@dataclass
class IfElseBlock:
    """An if/else branch with composable bodies.

    Each branch can contain arbitrary body nodes (ops, loops, nested if/else).
    The last output of each branch is renamed to ``branch_out`` for
    function-scoped access after the if/else block.

    Attributes:
        then_body: Body nodes for the then-branch.
        else_body: Body nodes for the else-branch.
    """

    then_body: list[BodyNode]
    else_body: list[BodyNode]


# Union type for composable body nodes
BodyNode = OpBlock | ForBlock | IfElseBlock
