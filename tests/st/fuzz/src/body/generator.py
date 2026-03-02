# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Random body structure generator for fuzz testing.

Generates composable body ASTs with configurable nesting depth and
control flow diversity. Supports:

- Plain ops (no control flow)
- For loops (accumulation mode)
- If/else branches (unequal branch sizes supported)
- Sequential control flows (ops + control flow + ops)
- Nested combinations (for inside if, if inside for, etc.)

The probability of adding control flow **diminishes** with depth:
deeper nesting is less likely, producing a natural distribution
of simple and complex test cases.
"""

from __future__ import annotations

import random
from typing import Any

from ..core.fuzzer import OpChainConfig, OpFuzzer
from .ast import BodyNode, ForBlock, IfElseBlock, OpBlock

# Minimum ops per segment (ensures non-trivial blocks)
MIN_OPS_PER_BLOCK = 1
# Minimum for-loop iterations
MIN_FOR_ITERATIONS = 2
# Maximum for-loop iterations
MAX_FOR_ITERATIONS = 4
# Maximum ops for pre/post segments around control flow
MAX_SEGMENT_OPS = 3


class BodyGenerator:
    """Generates random body ASTs with configurable nesting and control flow.

    The generator creates composable body structures where control flow
    constructs (for loops, if/else) can be nested up to ``max_depth`` levels.
    The probability of adding control flow decreases with depth via
    ``depth_decay``, producing a natural mix of simple and complex cases.
    """

    def __init__(
        self,
        rng: random.Random,
        fuzzer: OpFuzzer,
        max_depth: int = 2,
        depth_decay: float = 0.5,
        for_loop_prob: float = 0.5,
        if_else_prob: float = 0.5,
        min_for_iterations: int = MIN_FOR_ITERATIONS,
        max_for_iterations: int = MAX_FOR_ITERATIONS,
    ):
        """Initialize body generator.

        Args:
            rng: Random number generator for reproducibility.
            fuzzer: OpFuzzer instance for generating op chains.
            max_depth: Maximum nesting depth for control flow.
            depth_decay: Probability multiplier per depth level.
                At depth d, the effective control flow probability is
                ``base_prob * depth_decay^d``. Lower values produce
                shallower structures.
            for_loop_prob: Base probability of generating a for loop.
            if_else_prob: Base probability of generating if/else.
            min_for_iterations: Minimum for-loop iterations.
            max_for_iterations: Maximum for-loop iterations.
        """
        self.rng = rng
        self.fuzzer = fuzzer
        self.max_depth = max_depth
        self.depth_decay = depth_decay
        self.for_loop_prob = for_loop_prob
        self.if_else_prob = if_else_prob
        self.min_for_iterations = min_for_iterations
        self.max_for_iterations = max_for_iterations

    def generate_body(
        self,
        num_ops: int,
        num_inputs: int,
        output_shape: tuple[int, int],
        depth: int = 0,
        basic_ops_only: bool = False,
    ) -> list[BodyNode]:
        """Generate a random body structure.

        At each level, the generator decides whether to:
        1. Produce plain ops (OpBlock)
        2. Wrap ops in a ForBlock
        3. Wrap ops in an IfElseBlock
        4. Produce sequential blocks (pre-ops + control flow + post-ops)

        The probability of choosing control flow diminishes with depth.

        Args:
            num_ops: Total op budget for this body level.
            num_inputs: Number of input tensors available.
            output_shape: Default output shape for ops.
            depth: Current nesting depth (0 = top level).
            basic_ops_only: Restrict to basic vector ops (no matmul/reduction).

        Returns:
            List of BodyNode instances forming the body.
        """
        # Base case: too few ops or max depth reached → plain ops
        if num_ops < 1:
            return [self._make_op_block(1, num_inputs, output_shape, basic_ops_only)]
        if depth >= self.max_depth or num_ops <= 2:
            return [self._make_op_block(num_ops, num_inputs, output_shape, basic_ops_only)]

        # Compute effective probabilities (diminish with depth)
        decay = self.depth_decay**depth
        eff_for_prob = self.for_loop_prob * decay
        eff_if_prob = self.if_else_prob * decay

        # Roll for control flow type
        roll = self.rng.random()

        if roll < eff_for_prob:
            nodes = self._generate_for_body(
                num_ops,
                num_inputs,
                output_shape,
                depth,
                basic_ops_only,
            )
        elif roll < eff_for_prob + eff_if_prob:
            nodes = self._generate_if_else_body(
                num_ops,
                num_inputs,
                output_shape,
                depth,
                basic_ops_only,
            )
        else:
            # Plain ops (no control flow at this level)
            return [self._make_op_block(num_ops, num_inputs, output_shape, basic_ops_only)]

        # Connect control flow outputs to subsequent operations
        _connect_control_flow_outputs(nodes)
        return nodes

    def _generate_for_body(
        self,
        num_ops: int,
        num_inputs: int,
        output_shape: tuple[int, int],
        depth: int,
        basic_ops_only: bool,
    ) -> list[BodyNode]:
        """Generate a body with a ForBlock, optionally with pre/post ops.

        Structure: [pre_ops] + ForBlock([inner_body]) + [post_ops]
        """
        iterations = self.rng.randint(self.min_for_iterations, self.max_for_iterations)

        # Allocate ops: pre + loop body + post
        pre_count, body_count, post_count = self._split_ops_three_way(num_ops)

        nodes: list[BodyNode] = []

        # Pre-loop ops
        if pre_count > 0:
            nodes.append(self._make_op_block(pre_count, num_inputs, output_shape, basic_ops_only))

        # Loop body: recurse to allow nested control flow
        inner_body = self.generate_body(
            body_count,
            num_inputs,
            output_shape,
            depth + 1,
            basic_ops_only,
        )

        # Find accumulation target from pre-loop ops
        accum_var = self._find_accum_var(nodes, output_shape, num_inputs)
        last_body_output = self._get_last_output(inner_body)

        nodes.append(
            ForBlock(
                iterations=iterations,
                body=inner_body,
                accum_var=accum_var,
                last_body_output=last_body_output,
            )
        )

        # Post-loop ops
        if post_count > 0:
            nodes.append(self._make_op_block(post_count, num_inputs, output_shape, basic_ops_only))

        return nodes

    def _generate_if_else_body(
        self,
        num_ops: int,
        num_inputs: int,
        output_shape: tuple[int, int],
        depth: int,
        basic_ops_only: bool,
    ) -> list[BodyNode]:
        """Generate a body with an IfElseBlock, optionally with pre/post ops.

        Structure: [pre_ops] + IfElseBlock(then, else) + [post_ops]
        Branches can have unequal op counts.
        """
        # Allocate ops: pre + branches + post
        pre_count, branch_total, post_count = self._split_ops_three_way(num_ops)

        # Split branch ops unevenly between then and else
        then_count, else_count = self._split_branch_ops(branch_total)

        nodes: list[BodyNode] = []

        # Pre-if ops
        if pre_count > 0:
            nodes.append(
                self._make_op_block(
                    pre_count,
                    num_inputs,
                    output_shape,
                    basic_ops_only=True,
                )
            )

        # Then/else branches: recurse to allow nested control flow
        # Use basic_ops_only=True for branches to avoid matmul complexity
        then_body = self.generate_body(
            then_count,
            num_inputs,
            output_shape,
            depth + 1,
            basic_ops_only=True,
        )
        else_body = self.generate_body(
            else_count,
            num_inputs,
            output_shape,
            depth + 1,
            basic_ops_only=True,
        )

        nodes.append(IfElseBlock(then_body=then_body, else_body=else_body))

        # Post-if ops
        if post_count > 0:
            nodes.append(
                self._make_op_block(
                    post_count,
                    num_inputs,
                    output_shape,
                    basic_ops_only=True,
                )
            )

        return nodes

    def _make_op_block(
        self,
        num_ops: int,
        num_inputs: int,
        output_shape: tuple[int, int],
        basic_ops_only: bool,
    ) -> OpBlock:
        """Generate an OpBlock with a chain of random operations.

        Guarantees at least 1 operation in the chain by retrying with
        basic vector ops if the initial generation returns empty.
        """
        op_chain = self.fuzzer.generate_op_chain(
            OpChainConfig(
                num_ops=max(1, num_ops),
                input_count=num_inputs,
                allow_scalars=True,
                track_shapes=True,
                default_shape=output_shape,
                basic_ops_only=basic_ops_only,
            )
        )

        # Retry with basic_ops_only=True if chain is empty
        if not op_chain:
            op_chain = self.fuzzer.generate_op_chain(
                OpChainConfig(
                    num_ops=1,
                    input_count=num_inputs,
                    allow_scalars=False,
                    track_shapes=True,
                    default_shape=output_shape,
                    basic_ops_only=True,
                )
            )

        # Collect scalar values for parameter mapping
        scalar_values = set()
        for op_dict in op_chain:
            if op_dict.get("scalar_value"):
                scalar_values.add(op_dict["scalar_value"])

        scalar_value_to_param: dict[str, str] = {}
        for idx, value in enumerate(sorted(scalar_values)):
            scalar_value_to_param[value] = f"scalar_{idx}"

        return OpBlock(op_chain=op_chain, scalar_value_to_param=scalar_value_to_param)

    def _split_ops_three_way(self, num_ops: int) -> tuple[int, int, int]:
        """Split total ops into (pre, main, post) segments.

        Pre and post are limited to MAX_SEGMENT_OPS. Main gets the remainder.
        """
        if num_ops <= 2:
            return 0, max(1, num_ops), 0

        max_extra = num_ops - 1  # At least 1 op for main
        pre_count = self.rng.randint(0, min(MAX_SEGMENT_OPS, max_extra // 2))
        remaining = max_extra - pre_count
        post_count = self.rng.randint(0, min(MAX_SEGMENT_OPS, remaining))
        main_count = max(1, num_ops - pre_count - post_count)

        return pre_count, main_count, post_count

    def _split_branch_ops(self, total: int) -> tuple[int, int]:
        """Split ops between then and else branches (unequal allowed)."""
        if total <= 1:
            return max(1, total), max(1, total)

        # Random split: each branch gets at least 1 op
        then_count = self.rng.randint(1, max(1, total - 1))
        else_count = max(1, total - then_count)
        return then_count, else_count

    @staticmethod
    def _find_accum_var(
        nodes: list[BodyNode],
        output_shape: tuple[int, int],
        num_inputs: int,
    ) -> str | None:
        """Find a variable from preceding nodes to use as accumulation target.

        Searches backwards through preceding OpBlocks for a shape-compatible
        output variable. Falls back to tile_a if no OpBlock output matches.
        """
        # Search preceding OpBlocks backwards
        for node in reversed(nodes):
            if isinstance(node, OpBlock) and node.op_chain:
                for op_dict in reversed(node.op_chain):
                    op_shape = op_dict.get("output_shape", output_shape)
                    if op_shape == output_shape:
                        return op_dict["output"]

        # Fall back to first loaded tile
        if num_inputs > 0:
            return "tile_a"

        return None

    @staticmethod
    def _get_last_output(body: list[BodyNode]) -> str | None:
        """Get the last output variable name from a body."""
        if not body:
            return None

        last_node = body[-1]
        if isinstance(last_node, OpBlock) and last_node.op_chain:
            return last_node.op_chain[-1]["output"]
        if isinstance(last_node, IfElseBlock):
            return "branch_out"
        if isinstance(last_node, ForBlock):
            return last_node.accum_var or BodyGenerator._get_last_output(last_node.body)

        return None


def collect_all_scalars(body: list[BodyNode]) -> dict[str, str]:
    """Collect all scalar value-to-param mappings from a body tree.

    Walks the entire AST and merges scalar mappings, re-indexing
    parameter names to avoid collisions.

    Args:
        body: List of BodyNode instances.

    Returns:
        Unified scalar_value_to_param mapping.
    """
    all_values: set[str] = set()
    _collect_scalar_values(body, all_values)

    merged: dict[str, str] = {}
    for idx, value in enumerate(sorted(all_values)):
        merged[value] = f"scalar_{idx}"

    # Update all OpBlocks to use the unified mapping
    _update_scalar_mappings(body, merged)

    return merged


def _collect_scalar_values(body: list[BodyNode], values: set[str]) -> None:
    """Recursively collect scalar values from body tree."""
    for node in body:
        if isinstance(node, OpBlock):
            for op_dict in node.op_chain:
                if op_dict.get("scalar_value"):
                    values.add(op_dict["scalar_value"])
        elif isinstance(node, ForBlock):
            _collect_scalar_values(node.body, values)
        elif isinstance(node, IfElseBlock):
            _collect_scalar_values(node.then_body, values)
            _collect_scalar_values(node.else_body, values)


def _update_scalar_mappings(body: list[BodyNode], mapping: dict[str, str]) -> None:
    """Recursively update OpBlock scalar mappings to use unified mapping."""
    for node in body:
        if isinstance(node, OpBlock):
            node.scalar_value_to_param = mapping
        elif isinstance(node, ForBlock):
            _update_scalar_mappings(node.body, mapping)
        elif isinstance(node, IfElseBlock):
            _update_scalar_mappings(node.then_body, mapping)
            _update_scalar_mappings(node.else_body, mapping)


def _connect_control_flow_outputs(body: list[BodyNode]) -> None:
    """Connect control flow outputs to subsequent OpBlock operations.

    When an IfElseBlock or ForBlock is followed by an OpBlock, replaces one
    tile input of the OpBlock's first operation with the control flow output
    variable. This ensures control flow results are consumed, not dead code.
    """
    for i in range(1, len(body)):
        prev = body[i - 1]
        curr = body[i]

        if not isinstance(curr, OpBlock) or not curr.op_chain:
            continue

        if isinstance(prev, IfElseBlock):
            cf_output = "branch_out"
        elif isinstance(prev, ForBlock):
            cf_output = prev.accum_var
            if not cf_output:
                continue
        else:
            continue

        # Replace one tile input of the first operation
        first_op = curr.op_chain[0]
        inputs = first_op["inputs"]
        for j, inp in enumerate(inputs):
            if inp.startswith("tile_"):
                inputs[j] = cf_output
                break


def body_needs_branch_cond(body: list[BodyNode]) -> bool:
    """Check if any node in the body tree is an IfElseBlock."""
    for node in body:
        if isinstance(node, IfElseBlock):
            return True
        if isinstance(node, ForBlock) and body_needs_branch_cond(node.body):
            return True
    return False


def collect_all_op_chains(body: list[BodyNode]) -> list[dict[str, Any]]:
    """Collect all op chains from the body tree (flattened).

    Used for golden reference validation and metadata.
    """
    chains: list[dict[str, Any]] = []
    for node in body:
        if isinstance(node, OpBlock):
            chains.extend(node.op_chain)
        elif isinstance(node, ForBlock):
            chains.extend(collect_all_op_chains(node.body))
        elif isinstance(node, IfElseBlock):
            chains.extend(collect_all_op_chains(node.then_body))
            chains.extend(collect_all_op_chains(node.else_body))
    return chains
