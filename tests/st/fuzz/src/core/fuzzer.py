# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Operator fuzzer for generating random operator combinations.
"""

import random
from dataclasses import dataclass, field
from typing import Any

from .op_specs import (
    BLOCK_BINARY_OPS,
    BLOCK_COL_EXPAND_OPS,
    BLOCK_MATRIX_OPS,
    BLOCK_REDUCTION_OPS,
    BLOCK_RESHAPE_OPS,
    BLOCK_ROW_EXPAND_OPS,
    BLOCK_UNARY_OPS,
    OpSpec,
    ValueRange,
)
from .shape_utils import generate_aligned_shape, is_shape_aligned

__all__ = ["OpChainConfig", "OpFuzzer", "generate_aligned_shape", "is_shape_aligned"]


@dataclass
class OpChainConfig:
    """Configuration for a single op-chain generation pass."""

    num_ops: int = 5
    input_count: int = 2
    allow_scalars: bool = True
    track_shapes: bool = False
    default_shape: tuple[int, int] = (128, 128)
    prefer_matrix_ops: bool | None = None
    basic_ops_only: bool = False


@dataclass
class _GenState:
    """Mutable generation state shared across op-chain building helpers."""

    available_tiles: list[str]
    available_scalars: list[str]
    initial_inputs: set[str]
    used_inputs: set[str] = field(default_factory=set)
    variable_usage_count: dict[str, int] = field(default_factory=dict)
    variable_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)
    variable_ranges: dict[str, ValueRange] = field(default_factory=dict)
    operations: list[dict[str, Any]] = field(default_factory=list)
    next_tmp_index: int = 0
    track_shapes: bool = False
    default_shape: tuple[int, int] = (128, 128)
    allow_scalars: bool = True
    matmul_outputs: set[str] = field(default_factory=set)


class OpFuzzer:
    """Generates random operator combinations for fuzzing."""

    def __init__(
        self,
        seed: int | None = None,
        enable_advanced_ops: bool = False,
        advanced_ops_probability: float = 0.5,
    ):
        """Initialize fuzzer with optional seed for reproducibility.

        Args:
            seed: Random seed for reproducibility
            enable_advanced_ops: Enable advanced operations (row_expand, row_sum, matmul, etc.)
            advanced_ops_probability: Probability of selecting advanced ops (default: 0.5)
        """
        self.rng = random.Random(seed)

        # Auto-disable advanced ops if probability is 0
        if advanced_ops_probability <= 0.0:
            enable_advanced_ops = False

        # Basic operators (PipeType::V - VECTOR core)
        self.basic_vector_ops = BLOCK_BINARY_OPS + BLOCK_UNARY_OPS
        self.vector_ops = self.basic_vector_ops

        # Advanced operators (PipeType::V - VECTOR core)
        self.advanced_vector_ops = []

        # Matrix operators (PipeType::M - CUBE core)
        self.matrix_ops = []

        # Advanced ops probability control
        self.enable_advanced_ops = enable_advanced_ops
        self.advanced_ops_probability = advanced_ops_probability

        if enable_advanced_ops:
            # Add reduction (row and col), row_expand, col_expand, and reshape operators
            self.advanced_vector_ops = (
                BLOCK_REDUCTION_OPS
                # + BLOCK_COL_REDUCTION_OPS
                + BLOCK_ROW_EXPAND_OPS
                + BLOCK_COL_EXPAND_OPS
                + BLOCK_RESHAPE_OPS
            )
            self.vector_ops = self.basic_vector_ops + self.advanced_vector_ops
            self.matrix_ops = BLOCK_MATRIX_OPS

        # Default to VECTOR operators
        self.ops = self.vector_ops

        # Track operator usage to avoid overuse
        self.op_usage_count = {}
        self.exp_count = 0
        self.div_count = 0
        self.matmul_count = 0  # Track matmul usage
        self.matmul_limit = 1  # Allow at most 1 matmul per chain (set in generate_op_chain)
        # Current kernel's pipe type (None, 'M', 'V')
        self.current_pipe_type = None

    def generate_op_chain(self, config: OpChainConfig | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        """Generate a chain of operator calls.

        All input tensors and intermediate results are guaranteed to contribute
        to the final output through smart generation and post-processing.

        Algorithm:
        1. Select operations from the eligible pool
        2. Assign inputs and track variable shapes
        3. Limit expensive operations (exp, div)
        4. Ensure all input tensors are used
        5. Track value ranges to avoid numerical issues
        6. Route to M-pipe (matmul with memory management) or V-pipe (element-wise)

        Args:
            config: Generation configuration. Individual fields can also be passed
                    as keyword arguments for convenience.
        """
        cfg = OpChainConfig(**kwargs) if config is None else config
        self._select_pipe_type(cfg.prefer_matrix_ops, cfg.basic_ops_only)
        st = self._init_gen_state(cfg.input_count, cfg.allow_scalars, cfg.track_shapes, cfg.default_shape)

        for i in range(cfg.num_ops):
            op = self._pick_next_op(st, cfg.num_ops, i)
            if op is None:
                break

            unused_count = len(st.initial_inputs - st.used_inputs)
            remaining_ops = cfg.num_ops - i
            use_unused_priority = self._compute_unused_priority(unused_count, remaining_ops)

            result = self._select_inputs_for_op(op, st, use_unused_priority)
            if result is None:
                continue
            inputs, scalar_value = result

            self._build_op_dict(op, inputs, scalar_value, st)
            self._auto_expand_vec(st)

        self._finalize_chain(st)
        return st.operations

    def generate_branched_op_chains(
        self,
        config: OpChainConfig,
        num_branches: int = 2,
    ) -> list[list[dict[str, Any]]]:
        """Generate multiple independent op chains sharing the same inputs.

        Each branch is generated independently via generate_op_chain().
        All branches use the same input_count and produce compatible output shapes.

        Args:
            config: Generation configuration (num_ops controls ops per branch, minimum 1).
            num_branches: Number of branches to generate (default 2 for if/else).

        Returns:
            List of op chains, one per branch.
        """
        branch_cfg = OpChainConfig(
            num_ops=max(1, config.num_ops),
            input_count=config.input_count,
            allow_scalars=config.allow_scalars,
            track_shapes=config.track_shapes,
            default_shape=config.default_shape,
            prefer_matrix_ops=config.prefer_matrix_ops,
            basic_ops_only=config.basic_ops_only,
        )
        branches = []
        for _ in range(num_branches):
            chain = self.generate_op_chain(branch_cfg)
            branches.append(chain)
        return branches

    # ------------------------------------------------------------------
    # Private helpers — state initialisation & op-chain building
    # ------------------------------------------------------------------

    def _select_pipe_type(self, prefer_matrix_ops: bool | None, basic_ops_only: bool) -> None:
        """Reset fuzzer counters and select the pipe type for the chain."""
        self.op_usage_count = {}
        self.exp_count = 0
        self.div_count = 0
        self.matmul_count = 0
        self.matmul_limit = 1
        self.current_pipe_type = None

        if basic_ops_only:
            prefer_matrix_ops = False
        elif prefer_matrix_ops is None:
            prefer_matrix_ops = bool(self.matrix_ops and self.rng.random() < 0.05)

        use_matrix = prefer_matrix_ops and bool(self.matrix_ops)
        self.ops = (
            self.matrix_ops if use_matrix else (self.basic_vector_ops if basic_ops_only else self.vector_ops)
        )
        self.current_pipe_type = "M" if use_matrix else "V"

    def _init_gen_state(
        self,
        input_count: int,
        allow_scalars: bool,
        track_shapes: bool,
        default_shape: tuple[int, int],
    ) -> _GenState:
        """Build the initial generation state."""
        available_tiles = [f"tile_{chr(97 + i)}" for i in range(input_count)]
        return _GenState(
            available_tiles=available_tiles,
            available_scalars=["2.0"] if allow_scalars else [],
            initial_inputs=set(available_tiles),
            variable_usage_count={tile: 0 for tile in available_tiles},
            variable_shapes={tile: default_shape for tile in available_tiles} if track_shapes else {},
            variable_ranges={
                tile: ValueRange(can_be_negative=True, can_be_zero=True, can_be_positive=True)
                for tile in available_tiles
            },
            track_shapes=track_shapes,
            default_shape=default_shape,
            allow_scalars=allow_scalars,
        )

    @staticmethod
    def _compute_unused_priority(unused_count: int, remaining_ops: int) -> float:
        """Compute the priority weight for consuming unused input tiles."""
        if unused_count <= 0 or remaining_ops <= 0:
            return 0.7
        if unused_count >= remaining_ops:
            return 1.0
        return min(0.9, 0.7 + 0.3 * (unused_count / remaining_ops))

    def _pick_next_op(self, st: _GenState, num_ops: int, step: int) -> OpSpec | None:
        """Pick the next operator for step *step*, or return None to stop."""
        eligible_ops = self._get_eligible_ops_safe(
            st.available_tiles,
            st.available_scalars,
            st.allow_scalars,
            st.variable_shapes if st.track_shapes else None,
            st.variable_ranges,
        )
        # Ops requiring params (e.g. reshape) need shape info to derive params.
        if not st.track_shapes:
            eligible_ops = [op for op in eligible_ops if not op.requires_params]
        if not eligible_ops:
            return None

        # Prefer binary ops when we urgently need to consume unused inputs
        unused_count = len(st.initial_inputs - st.used_inputs)
        remaining_ops = num_ops - step
        use_unused_priority = self._compute_unused_priority(unused_count, remaining_ops)
        if unused_count > 0 and use_unused_priority >= 0.9:
            binary_ops = [op for op in eligible_ops if sum(1 for t in op.input_types if t == "tile") >= 2]
            if binary_ops:
                eligible_ops = binary_ops

        # Select op with usage-frequency throttling
        op = None
        for _retry in range(3):
            candidate_op = self.rng.choice(eligible_ops)
            if self.op_usage_count.get(candidate_op.name, 0) > num_ops * 0.4:
                if self.rng.random() < 0.3:
                    op = candidate_op
                    break
            else:
                op = candidate_op
                break
        return op if op is not None else eligible_ops[0]

    def _build_op_dict(
        self,
        op: OpSpec,
        inputs: list[str],
        scalar_value: str | None,
        st: _GenState,
    ) -> None:
        """Build an op-result dict, update shapes/ranges/counters, and append to state."""
        output = f"tmp_{st.next_tmp_index}"
        st.next_tmp_index += 1

        params = None
        if op.requires_params:
            input_shapes = [st.variable_shapes[inp] for inp in inputs if inp in st.variable_shapes]
            if input_shapes:
                params = op.generate_params(input_shapes, self.rng)
            else:
                # Cannot derive required params (e.g. reshape without shape tracking).
                # Skip this op to avoid invalid codegen.
                return

        op_dict: dict[str, Any] = {
            "op": op,
            "inputs": inputs,
            "output": output,
            "scalar_value": scalar_value,
            "params": params,
        }

        if st.track_shapes:
            input_shapes = [st.variable_shapes[inp] for inp in inputs if inp in st.variable_shapes]
            output_shape = op.compute_output_shape(input_shapes, params)
            op_dict["output_shape"] = output_shape
            op_dict["input_shapes"] = input_shapes
            st.variable_shapes[output] = output_shape

        input_ranges = [st.variable_ranges[inp] for inp in inputs if inp in st.variable_ranges]
        st.variable_ranges[output] = op.compute_output_range(input_ranges)

        st.operations.append(op_dict)
        st.available_tiles.append(output)
        st.variable_usage_count[output] = 0

        # Track matmul outputs to prevent chaining (Acc type cannot move to Left/Right)
        if "matmul" in op.name:
            st.matmul_outputs.add(output)

        self._update_op_counters(op.name)

    def _update_op_counters(self, op_name: str) -> None:
        """Increment per-op usage counters."""
        self.op_usage_count[op_name] = self.op_usage_count.get(op_name, 0) + 1
        if "exp" in op_name:
            self.exp_count += 1
        if "div" in op_name:
            self.div_count += 1
        if "matmul" in op_name:
            self.matmul_count += 1

    def _auto_expand_vec(self, st: _GenState) -> None:
        """Auto-expand 1-wide vector outputs so they don't get stranded (V-pipe only).

        Optionally inserts reshape before expansion to test cross-dimensional vector usage.
        """
        if not st.track_shapes or self.current_pipe_type != "V" or not st.operations:
            return
        output = st.operations[-1]["output"]
        if output not in st.variable_shapes:
            return
        output_shape = st.variable_shapes[output]

        # Check if this is a vector output from a reduction operation
        is_row_vec = output_shape[1] == 1 and output_shape[0] > 1  # [M, 1]
        is_col_vec = output_shape[0] == 1 and output_shape[1] > 1  # [1, N]

        if is_row_vec or is_col_vec:
            # 30% chance to insert reshape before expansion (cross-dimensional usage)
            if self.enable_advanced_ops and self.rng.random() < 0.3:
                reshape_op = BLOCK_RESHAPE_OPS[0]  # tile.reshape

                # Generate target shape (transpose the vector)
                if is_row_vec:
                    target_shape = (1, output_shape[0])  # [M, 1] -> [1, M]
                else:
                    target_shape = (output_shape[1], 1)  # [1, N] -> [N, 1]

                # Create reshape operation
                reshaped_output = f"tmp_{st.next_tmp_index}"
                st.next_tmp_index += 1

                reshape_dict = {
                    "op": reshape_op,
                    "inputs": [output],
                    "output": reshaped_output,
                    "scalar_value": None,
                    "params": {"target_shape": target_shape},
                    "input_shapes": [output_shape],
                    "output_shape": target_shape,
                }
                st.operations.append(reshape_dict)
                st.variable_shapes[reshaped_output] = target_shape
                st.variable_usage_count[output] = st.variable_usage_count.get(output, 0) + 1
                st.available_tiles.append(reshaped_output)
                st.variable_usage_count[reshaped_output] = 0
                # Reshape preserves value range
                if output in st.variable_ranges:
                    st.variable_ranges[reshaped_output] = st.variable_ranges[output]

                # Update output to the reshaped variable
                output = reshaped_output
                output_shape = target_shape
                is_row_vec, is_col_vec = is_col_vec, is_row_vec  # Swap dimensions

        # Now expand the vector (original or reshaped)
        if output_shape[1] == 1:  # [M, 1] row vector
            self._try_expand_vec(output, vec_dim=1, expand_ops_pool=BLOCK_ROW_EXPAND_OPS, st=st)
        elif output_shape[0] == 1:  # [1, N] col vector
            self._try_expand_vec(output, vec_dim=0, expand_ops_pool=BLOCK_COL_EXPAND_OPS, st=st)

    # ------------------------------------------------------------------
    # Private helpers — input selection
    # ------------------------------------------------------------------

    def _select_inputs_for_op(
        self,
        op: OpSpec,
        st: _GenState,
        use_unused_priority: float,
    ) -> tuple[list[str], str | None] | None:
        """Select concrete inputs for one operation, inserting wrapper ops if needed.

        Returns:
            (inputs, scalar_value) on success.
            None if the operation must be skipped due to unsatisfiable constraints.
        """
        inputs: list[str] = []
        scalar_value: str | None = None
        first_input_shape: tuple[int, int] | None = None

        for input_idx, input_type in enumerate(op.input_types):
            effective_type = self._resolve_effective_type(op, input_type, input_idx, st.allow_scalars)

            if effective_type == "tile":
                result = self._select_tile_input(op, input_idx, st, use_unused_priority, first_input_shape)
                if result is None:
                    return None
                inputs.append(result)
                st.variable_usage_count[result] = st.variable_usage_count.get(result, 0) + 1
                if result in st.initial_inputs:
                    st.used_inputs.add(result)
                if input_idx == 0 and st.track_shapes and result in st.variable_shapes:
                    first_input_shape = st.variable_shapes[result]
            else:  # scalar
                if not st.available_scalars:
                    st.available_scalars.append(f"{self.rng.uniform(0.1, 10.0):.2f}")
                scalar_value = self.rng.choice(st.available_scalars)
                inputs.append(scalar_value)

        if len(inputs) < len(op.input_types):
            return None
        return inputs, scalar_value

    def _resolve_effective_type(
        self, op: OpSpec, input_type: str, input_idx: int, allow_scalars: bool
    ) -> str:
        """Optionally redirect a tile input to scalar."""
        if (
            input_type == "tile"
            and input_idx == 1
            and op.second_can_be_scalar
            and allow_scalars
            and self.rng.random() < 0.5
        ):
            return "scalar"
        return input_type

    def _select_tile_input(
        self,
        op: OpSpec,
        input_idx: int,
        st: _GenState,
        use_unused_priority: float,
        first_input_shape: tuple[int, int] | None = None,
    ) -> str | None:
        """Select a single tile input, applying range/shape/weight filters.

        Returns the selected tile name, or None to skip the whole op.
        """
        candidates = list(st.available_tiles)

        # Matmul inputs must come from original tile_ vars, not from matmul outputs
        if op.constraints.get("requires_memory_management", False):
            candidates = [t for t in candidates if t.startswith("tile_") and t not in st.matmul_outputs]

        # Value-range filtering
        result = self._filter_by_value_range(op, input_idx, candidates, st.variable_ranges)
        if result is None:
            return None
        candidates, needs_abs_wrapper = result

        # Shape filtering
        if st.track_shapes:
            candidates = self._filter_by_shape(
                op, input_idx, candidates, st.variable_shapes, first_input_shape
            )
            if candidates is None:
                return None

        # Weighted selection
        candidates = self._pick_weighted_candidates(candidates, st, use_unused_priority)
        selected = self.rng.choice(candidates)

        # Insert abs() wrapper when no naturally safe tile exists
        if needs_abs_wrapper:
            selected = self._insert_abs_wrapper(selected, st)

        return selected

    def _filter_by_value_range(
        self,
        op: OpSpec,
        input_idx: int,
        candidates: list[str],
        variable_ranges: dict[str, ValueRange],
    ) -> tuple[list[str], bool] | None:
        """Filter candidates by value-range constraints.

        Returns:
            (filtered_candidates, needs_abs_wrapper) or None to skip the op.
        """
        needs_abs_wrapper = False

        if op.constraints.get("positive_only", False):
            is_safe = (
                (lambda r: r.is_safe_for_log()) if "log" in op.name else (lambda r: r.is_safe_for_sqrt())
            )
            safe = [t for t in candidates if t in variable_ranges and is_safe(variable_ranges[t])]
            needs_abs_wrapper = not safe
            if safe:
                candidates = safe

        if op.constraints.get("avoid_zero", False):
            is_divisor = (len(op.input_types) == 2 and input_idx == 1) or len(op.input_types) == 1
            if is_divisor:
                safe = [
                    t for t in candidates if t in variable_ranges and variable_ranges[t].is_safe_for_div()
                ]
                if safe:
                    candidates = safe
                else:
                    return None

        return candidates, needs_abs_wrapper

    def _filter_by_shape(
        self,
        op: OpSpec,
        input_idx: int,
        candidates: list[str],
        variable_shapes: dict[str, tuple[int, int]],
        first_input_shape: tuple[int, int] | None = None,
    ) -> list[str] | None:
        """Filter candidates by shape constraints. Returns None to skip the op."""
        vec_dim = (
            1
            if op.constraints.get("row_vec_required", False)
            else 0
            if op.constraints.get("col_vec_required", False)
            else None
        )
        if vec_dim is not None:
            return self._filter_vec_shape(vec_dim, input_idx, candidates, variable_shapes, first_input_shape)

        # Exclude 1-wide tiles from non-expand operators
        candidates = [t for t in candidates if t not in variable_shapes or variable_shapes[t][1] != 1]
        if op.constraints.get("produces_row_vec", False):
            # Row reduction [M,N]->[M,1]: also exclude [1,N] col vectors to avoid [1,1] output
            candidates = [t for t in candidates if t not in variable_shapes or variable_shapes[t][0] != 1]
        if op.constraints.get("produces_col_vec", False):
            # Col reduction [M,N]->[1,N]: exclude [1,N] col vectors to avoid re-reducing;
            # [M,1] row vectors are already excluded by the general filter above
            candidates = [t for t in candidates if t not in variable_shapes or variable_shapes[t][0] != 1]
        candidates = [t for t in candidates if self._is_shape_compatible(op, t, variable_shapes)]

        # For the second operand of binary ops, enforce broadcast compatibility
        # with the already-selected first operand.
        if input_idx == 1 and first_input_shape is not None:
            if op.constraints.get("matmul_shape", False):
                # Matmul: inner dimensions must match  [M,K] @ [K,N]
                k = first_input_shape[1]
                candidates = [t for t in candidates if t not in variable_shapes or variable_shapes[t][0] == k]
            elif op.constraints.get("exact_shape", False):
                # Ops like minimum/maximum require all operands to have identical shapes
                candidates = [
                    t
                    for t in candidates
                    if t not in variable_shapes or variable_shapes[t] == first_input_shape
                ]
            else:
                # Element-wise / broadcast: second shape must be broadcast-compatible
                candidates = [
                    t
                    for t in candidates
                    if t not in variable_shapes
                    or self._shapes_broadcast_compatible(first_input_shape, variable_shapes[t])
                ]

        return candidates or None

    @staticmethod
    def _filter_vec_shape(
        vec_dim: int,
        input_idx: int,
        candidates: list[str],
        variable_shapes: dict[str, tuple[int, int]],
        first_input_shape: tuple[int, int] | None = None,
    ) -> list[str] | None:
        """Filter candidates for vec-required ops (row_expand / col_expand)."""
        match_dim = 1 - vec_dim
        if input_idx == 0:
            filtered = [t for t in candidates if t not in variable_shapes or variable_shapes[t][vec_dim] != 1]
        elif input_idx == 1:
            filtered = [
                t
                for t in candidates
                if t in variable_shapes
                and variable_shapes[t][vec_dim] == 1
                and (
                    first_input_shape is None or variable_shapes[t][match_dim] == first_input_shape[match_dim]
                )
            ]
        else:
            filtered = candidates
        return filtered or None

    def _pick_weighted_candidates(
        self,
        candidates: list[str],
        st: _GenState,
        use_unused_priority: float,
    ) -> list[str]:
        """Score and filter candidates, preferring unused initial inputs."""
        unused_initial = {t for t in candidates if t in st.initial_inputs and t not in st.used_inputs}
        scores: list[tuple[str, int]] = []
        for t in candidates:
            score = 0
            if t in unused_initial:
                score += 50
                if use_unused_priority >= 0.9:
                    score += 30
            score += max(0, 20 - st.variable_usage_count.get(t, 0) * 5)
            if t.startswith("tmp_"):
                score += 5
            scores.append((t, score))

        max_score = max(s for _, s in scores)
        threshold, prob = (
            (max(max_score * 0.6, 30), 0.85) if max_score >= 40 else (max(max_score * 0.7, 10), 0.75)
        )
        filtered = [t for t, s in scores if s >= threshold]
        if filtered and self.rng.random() < prob:
            return filtered
        return candidates

    def _insert_abs_wrapper(self, selected: str, st: _GenState) -> str:
        """Wrap *selected* in an abs() op so it becomes safe for sqrt/log."""
        abs_op = next((op for op in BLOCK_UNARY_OPS if op.name == "tile.abs"), None)
        if abs_op is None:
            return selected

        abs_output = f"tmp_{st.next_tmp_index}"
        st.next_tmp_index += 1
        abs_dict: dict[str, Any] = {
            "op": abs_op,
            "inputs": [selected],
            "output": abs_output,
            "scalar_value": None,
            "params": None,
        }
        if st.track_shapes:
            abs_dict["output_shape"] = st.variable_shapes.get(selected, st.default_shape)
            st.variable_shapes[abs_output] = abs_dict["output_shape"]
        st.variable_ranges[abs_output] = abs_op.compute_output_range(
            [st.variable_ranges.get(selected, ValueRange())]
        )
        st.operations.append(abs_dict)
        st.available_tiles.append(abs_output)
        st.variable_usage_count[abs_output] = 0
        st.variable_usage_count[selected] = st.variable_usage_count.get(selected, 0) + 1
        return abs_output

    # ------------------------------------------------------------------
    # Private helpers — chain finalisation & vector expansion
    # ------------------------------------------------------------------

    def _finalize_chain(self, st: _GenState) -> None:
        """Merge all unused inputs and intermediates into the final chain output.

        Ensures every initial input tensor and every intermediate result
        contributes to the final output, so no dead variables remain.
        """
        merge_op = (
            self.matrix_ops[0]
            if self.current_pipe_type == "M" and self.matrix_ops
            else next((op for op in BLOCK_BINARY_OPS if op.name == "tile.add"), None)
        )

        # Pass 1: merge unused initial inputs
        for unused in st.initial_inputs - st.used_inputs:
            if not st.operations:
                break
            self._merge_unused(merge_op, unused, st)
            st.used_inputs.add(unused)

        # Pass 2: merge unused intermediate results
        if not st.operations:
            return
        final_output = st.operations[-1]["output"]
        unused_intermediates = [
            var
            for var, count in st.variable_usage_count.items()
            if var.startswith("tmp_") and count == 0 and var != final_output
        ]
        for unused in unused_intermediates:
            self._merge_unused(merge_op, unused, st)

    def _merge_unused(self, merge_op: OpSpec, unused: str, st: _GenState) -> None:
        """Merge a single unused variable into the current chain tail."""
        if not st.operations:
            return
        current_final = st.operations[-1]["output"]
        if st.track_shapes:
            if st.variable_shapes.get(unused, st.default_shape) != st.variable_shapes.get(
                current_final, st.default_shape
            ):
                return
        output = f"tmp_{len(st.operations)}"
        op_dict: dict[str, Any] = {
            "op": merge_op,
            "inputs": [unused, current_final],
            "output": output,
            "scalar_value": None,
            "params": None,
        }
        if st.track_shapes:
            input_shapes = [
                st.variable_shapes.get(unused, st.default_shape),
                st.variable_shapes.get(current_final, st.default_shape),
            ]
            op_dict["output_shape"] = merge_op.compute_output_shape(input_shapes)
            st.variable_shapes[output] = op_dict["output_shape"]
        st.operations.append(op_dict)
        st.available_tiles.append(output)
        st.variable_usage_count[output] = 0
        st.variable_usage_count[unused] = st.variable_usage_count.get(unused, 0) + 1
        st.variable_usage_count[current_final] = st.variable_usage_count.get(current_final, 0) + 1

    def _try_expand_vec(
        self,
        vec_name: str,
        vec_dim: int,
        expand_ops_pool: list[OpSpec],
        st: _GenState,
    ) -> None:
        """Expand a 1-wide vector tile to a full 2-D tile via a broadcast op.

        Args:
            vec_name: Name of the vector tile to expand.
            vec_dim: The dimension that equals 1 in the vector shape.
                     1 -> row vector [M, 1];  0 -> col vector [1, N].
            expand_ops_pool: The broadcast op list to choose from
                             (BLOCK_ROW_EXPAND_OPS or BLOCK_COL_EXPAND_OPS).
        """
        match_dim = 1 - vec_dim
        vec_shape = st.variable_shapes[vec_name]
        match_val = vec_shape[match_dim]

        # Find a full [M, N] tile whose match_dim aligns with the vector
        candidate_tiles = [
            t
            for t in st.available_tiles
            if t in st.variable_shapes
            and st.variable_shapes[t][match_dim] == match_val
            and st.variable_shapes[t][vec_dim] != 1
        ]
        if not candidate_tiles:
            return

        regular_tile = self.rng.choice(candidate_tiles)
        regular_shape = st.variable_shapes[regular_tile]

        # Filter expand ops: limit div usage and check vec safety
        eligible = [op for op in expand_ops_pool if "div" not in op.name or self.div_count < 5]
        vec_range = st.variable_ranges.get(vec_name)
        if vec_range and not vec_range.is_safe_for_div():
            eligible = [op for op in eligible if "div" not in op.name]
        if not eligible:
            return

        expand_op = self.rng.choice(eligible)
        output_name = f"tmp_{st.next_tmp_index}"
        st.next_tmp_index += 1

        op_dict: dict[str, Any] = {
            "op": expand_op,
            "inputs": [regular_tile, vec_name],
            "output": output_name,
            "scalar_value": None,
            "params": None,
            "output_shape": regular_shape,
        }
        st.operations.append(op_dict)
        st.available_tiles.append(output_name)
        st.variable_shapes[output_name] = regular_shape
        st.variable_usage_count[output_name] = 0

        input_ranges = [st.variable_ranges.get(regular_tile), st.variable_ranges.get(vec_name)]
        if all(r is not None for r in input_ranges):
            st.variable_ranges[output_name] = expand_op.compute_output_range(input_ranges)

        st.variable_usage_count[regular_tile] = st.variable_usage_count.get(regular_tile, 0) + 1
        st.variable_usage_count[vec_name] = st.variable_usage_count.get(vec_name, 0) + 1
        self._update_op_counters(expand_op.name)

    # ------------------------------------------------------------------
    # Private helpers — eligibility & compatibility
    # ------------------------------------------------------------------

    # Keyword-based usage limits.  To throttle a new op category, add an
    # entry here — no other code changes needed in the eligibility loop.
    _USAGE_LIMITS: dict[str, str] = {
        "exp": "exp_count",
        "div": "div_count",
        "matmul": "matmul_count",
    }
    _USAGE_CAPS: dict[str, int | str] = {
        "exp": 3,
        "div": 5,
        "matmul": "matmul_limit",  # read from self attribute
    }

    def _is_usage_allowed(self, op: OpSpec) -> bool:
        """Check if op is still within its usage limit."""
        for keyword, count_attr in self._USAGE_LIMITS.items():
            if keyword not in op.name:
                continue
            cap = self._USAGE_CAPS[keyword]
            limit = getattr(self, cap) if isinstance(cap, str) else cap
            if getattr(self, count_attr) >= limit:
                return False
        return True

    def _get_eligible_ops_safe(
        self,
        available_tiles: list[str],
        available_scalars: list[str],
        allow_scalars: bool,
        variable_shapes: dict[str, tuple[int, int]] | None = None,
        variable_ranges: dict[str, ValueRange] | None = None,
    ) -> list[OpSpec]:
        """Get operators that can be applied given the current generation state.

        Each check is delegated to either a table lookup (_is_usage_allowed)
        or to an OpSpec method (has_enough_inputs, is_range_eligible,
        is_shape_eligible), so adding new constraint types only requires
        changes in OpSpec — not here.
        """
        eligible = []
        for op in self.ops:
            if not self._is_usage_allowed(op):
                continue
            if not op.has_enough_inputs(available_tiles, available_scalars, allow_scalars):
                continue
            if variable_ranges is not None and not op.is_range_eligible(
                available_tiles, variable_ranges, allow_scalars
            ):
                continue
            if not op.is_shape_eligible(available_tiles, variable_shapes):
                continue
            eligible.append(op)
        return eligible

    @staticmethod
    def _is_shape_compatible(op: OpSpec, var: str, variable_shapes: dict[str, tuple[int, int]]) -> bool:
        """Check if a variable's shape is compatible with an operator."""
        if var not in variable_shapes:
            return True
        var_shape = variable_shapes[var]
        if op.constraints.get("row_vec_required", False) and var_shape[1] != 1:
            return False
        if op.constraints.get("col_vec_required", False) and var_shape[0] != 1:
            return False
        return True

    @staticmethod
    def _shapes_broadcast_compatible(shape_a: tuple[int, int], shape_b: tuple[int, int]) -> bool:
        """Check if two 2D shapes are broadcast-compatible.

        Two dimensions are compatible when they are equal or one of them is 1.
        """
        return (shape_a[0] == shape_b[0] or shape_b[0] == 1) and (shape_a[1] == shape_b[1] or shape_b[1] == 1)
