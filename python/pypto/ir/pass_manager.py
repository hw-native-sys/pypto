# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pass manager for IR transformations."""

import os
from collections.abc import Callable
from enum import Enum

from pypto.pypto_core import ir as core_ir
from pypto.pypto_core import passes

from .printer import python_print


class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""

    Default = "Default"  # No optimization
    PTOAS = "PTOAS"  # PTO assembly optimization without scheduling and sync


class PassManager:
    """Manager for organizing and executing IR transformation passes.

    PassManager maintains a sequence of Pass instances for different optimization
    strategies and executes them in order on a given Program. It delegates to
    a C++ PassPipeline for execution. Instrumentation (verification, logging)
    is handled by PassContext — see passes.PassContext.

    Usage:
        # Get a pre-configured strategy
        pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
        result = pm.run_passes(program)

        # With property verification via PassContext
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
            result = pm.run_passes(program)
    """

    # Static storage: strategy -> List of (pass_name, pass_factory) tuples
    _strategy_passes: dict[OptimizationStrategy, list[tuple[str, Callable[[], passes.Pass]]]] = {}

    @classmethod
    def _register_passes(cls):
        """Register all strategy Pass configurations."""
        cls._strategy_passes = {
            OptimizationStrategy.Default: [
                ("ConvertToSSA", lambda: passes.convert_to_ssa()),
                ("FlattenCallExpr", lambda: passes.flatten_call_expr()),
                ("RunVerifier", lambda: passes.run_verifier()),
                ("InitMemRef", lambda: passes.init_mem_ref()),
                ("MemoryReuse", lambda: passes.basic_memory_reuse()),
                ("InsertSync", lambda: passes.insert_sync()),
                ("AllocateMemoryAddr", lambda: passes.allocate_memory_addr()),
            ],
            OptimizationStrategy.PTOAS: [
                ("ConvertToSSA", lambda: passes.convert_to_ssa()),
                ("FlattenCallExpr", lambda: passes.flatten_call_expr()),
                ("RunVerifier", lambda: passes.run_verifier()),
                ("InitMemRef", lambda: passes.init_mem_ref()),
                ("MemoryReuse", lambda: passes.basic_memory_reuse()),
                ("AllocateMemoryAddr", lambda: passes.allocate_memory_addr()),
            ],
        }

    @classmethod
    def get_strategy(
        cls,
        strategy: OptimizationStrategy = OptimizationStrategy.Default,
    ) -> "PassManager":
        """Get a PassManager configured for the specified strategy.

        Args:
            strategy: The optimization strategy to use (default: Default)

        Returns:
            A PassManager instance configured with the appropriate passes
        """
        if not cls._strategy_passes:
            cls._register_passes()
        return cls(strategy)

    def __init__(self, strategy: OptimizationStrategy):
        """Initialize PassManager with a specific strategy.

        Args:
            strategy: The optimization strategy to use
        """
        self.strategy = strategy
        self.passes: list[passes.Pass] = []
        self.pass_names: list[str] = []

        # Build pass list
        for pass_name, pass_factory in self._strategy_passes[strategy]:
            self.passes.append(pass_factory())
            self.pass_names.append(pass_name)

        # Build C++ PassPipeline
        self._pipeline = passes.PassPipeline()
        for p in self.passes:
            self._pipeline.add_pass(p)

    def run_passes(
        self,
        input_ir: core_ir.Program,
        dump_ir: bool = False,
        output_dir: str | None = None,
        prefix: str = "pl",
    ) -> core_ir.Program:
        """Execute all passes in sequence on a Program.

        Args:
            input_ir: Input Program to transform
            dump_ir: Whether to dump IR after each pass (default: False)
            output_dir: Directory to dump IR files. Required when dump_ir=True.
            prefix: Module prefix for python_print (default: 'pl')

        Returns:
            Transformed Program after all passes have been applied

        Raises:
            ValueError: If dump_ir=True but output_dir is None
        """
        if not dump_ir:
            # Use C++ PassPipeline for property-tracked execution
            return self._pipeline.run(input_ir)

        # Dump mode: validate parameters, use CallbackInstrument for IR dumping
        if output_dir is None:
            raise ValueError("output_dir is required when dump_ir=True")

        if not isinstance(input_ir, core_ir.Program):
            raise ValueError("dump_ir mode only supports Program input")

        os.makedirs(output_dir, exist_ok=True)

        # Save frontend IR
        frontend_path = os.path.join(output_dir, "00_frontend.py")
        with open(frontend_path, "w") as f:
            f.write(python_print(input_ir, prefix=prefix))

        # Use instrument for IR dumping -- verification handled by C++ pipeline.
        # We index self.pass_names (Python-side names from _register_passes) rather than
        # _pass_obj.get_name() because registered names may differ from C++ names
        # (e.g. "MemoryReuse" vs "BasicMemoryReuse").
        pass_index = 0

        def after_pass(_pass_obj: passes.Pass, program: core_ir.Program) -> None:
            nonlocal pass_index
            pass_name = self.pass_names[pass_index]
            dump_path = os.path.join(output_dir, f"{pass_index + 1:02d}_after_{pass_name}.py")
            with open(dump_path, "w") as f:
                f.write(python_print(program, prefix=prefix))
            pass_index += 1

        dump_instrument = passes.CallbackInstrument(after_pass=after_pass, name="IRDump")

        # Create inner PassContext with the dump instrument. This replaces any outer
        # context's instruments (intentional -- dump mode controls its own instruments)
        # but preserves the outer verification level for the C++ pipeline.
        ctx = passes.PassContext.current()
        level = ctx.get_verification_level() if ctx else passes.get_default_verification_level()

        with passes.PassContext([dump_instrument], level):
            return self._pipeline.run(input_ir)

    def get_pass_names(self) -> list[str]:
        """Get the names of all passes in this manager.

        Returns:
            List of pass names assigned during registration
        """
        return self.pass_names


# Initialize the pass registry when the module is loaded
PassManager._register_passes()
