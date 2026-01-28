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
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from pypto.pypto_core import ir as core_ir
from pypto.pypto_core import passes

from .printer import python_print


class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""

    Default = "Default"  # No optimization
    Custom1 = "Custom1"  # Custom optimization strategy 1
    Custom2 = "Custom2"  # Custom optimization strategy 2
    XPlatform = "XPlatform"  # Cross-platform optimization without scheduling and sync


class PassManager:
    """Manager for organizing and executing IR transformation passes.

    PassManager maintains a sequence of Pass instances for different optimization
    strategies and executes them in order on a given Program. It uses
    a pipeline model where each pass's output becomes the input to the next passes.

    Usage:
        # Get a pre-configured strategy
        pm = PassManager.get_strategy(OptimizationStrategy.Custom2)
        result = pm.run_passes(program)  # For Program

        # Or use the shorthand
        result = PassManager.get_strategy(OptimizationStrategy.Custom2).run_passes(program)
    """

    # Static storage: strategy -> List of (pass_name, pass_factory) tuples
    _strategy_passes: Dict[OptimizationStrategy, List[Tuple[str, Callable[[], passes.Pass]]]] = {}

    @classmethod
    def _register_passes(cls):
        """Register all strategy Pass configurations.

        This method defines the static Pass pipeline for each optimization strategy.
        Each pass is registered with a unique name and a factory function.
        To add a new strategy or modify existing ones, edit this method.
        """
        cls._strategy_passes = {
            OptimizationStrategy.Default: [
                # No passes for Default (no optimization)
                ("InitMemRef", lambda: passes.InitMemRefPass()),
                ("MemoryReuse", lambda: passes.BasicMemoryReusePass()),
                ("InsertSync", lambda: passes.InsertSyncPass()),
            ],
            OptimizationStrategy.Custom1: [
                # Custom optimization strategy 1
                ("IdentityPass_1", lambda: passes.identity()),
            ],
            OptimizationStrategy.Custom2: [
                # Custom optimization strategy 2
                ("IdentityPass_1", lambda: passes.identity()),
                ("IdentityPass_2", lambda: passes.identity()),
            ],
            OptimizationStrategy.XPlatform: [
                ("InitMemRef", lambda: passes.init_mem_ref()),
                ("MemoryReuse", lambda: passes.basic_memory_reuse()),
                ("AddAlloc", lambda: passes.add_alloc()),
            ],
        }

    @classmethod
    def get_strategy(cls, strategy: OptimizationStrategy = OptimizationStrategy.Default) -> "PassManager":
        """Get a PassManager configured for the specified strategy.

        Args:
            strategy: The optimization strategy to use (default: Default)

        Returns:
            A PassManager instance configured with the appropriate passes

        Example:
            pm = PassManager.get_strategy(OptimizationStrategy.Custom2)
            result = pm.run_passes(program)

            pm_default = PassManager.get_strategy()  # Uses default strategy
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
        self.passes = []
        self.pass_names = []

        # Instantiate all passes for this strategy
        for pass_name, pass_factory in self._strategy_passes[strategy]:
            self.passes.append(pass_factory())
            self.pass_names.append(pass_name)

    def run_passes(
        self,
        input_ir: core_ir.Program,
        dump_ir: bool = False,
        output_dir: Optional[str] = None,
        prefix: str = "pl",
    ) -> core_ir.Program:
        """Execute all passes in sequence on a Program.

        Each pass's output becomes the input to the next passes.

        Args:
            input_ir: Input Program to transform
            dump_ir: Whether to dump IR after each pass (default: False)
            output_dir: Directory to dump IR files. Required when dump_ir=True.
            prefix: Module prefix for python_print (default: 'pl')

        Returns:
            Transformed Program after all passes have been applied

        Raises:
            ValueError: If dump_ir=True but output_dir is None
            ValueError: If dump_ir=True but input_ir is not a Program
        """
        if not dump_ir:
            # No dump mode: directly execute all passes using C++ Program interface
            current = input_ir
            for pass_instance in self.passes:
                current = pass_instance(current)
            return current
        else:
            # Dump mode: validate parameters and dump IR after each pass
            if output_dir is None:
                raise ValueError("output_dir is required when dump_ir=True")

            if not isinstance(input_ir, core_ir.Program):
                raise ValueError("dump_ir mode only supports Program input")

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Step 1: Save frontend IR (00_frontend.py)
            frontend_path = os.path.join(output_dir, "00_frontend.py")
            with open(frontend_path, "w") as f:
                f.write(python_print(input_ir, prefix=prefix))

            # Step 2: Execute and dump each pass
            current_program = input_ir
            for i, (pass_instance, pass_name) in enumerate(zip(self.passes, self.pass_names), start=1):
                # Use C++ Program interface directly
                current_program = pass_instance(current_program)

                # Dump IR after this pass
                dump_path = os.path.join(output_dir, f"{i:02d}_after_{pass_name}.py")
                with open(dump_path, "w") as f:
                    f.write(python_print(current_program, prefix=prefix))

            return current_program

    def get_pass_names(self) -> List[str]:
        """Get the names of all passes in this manager.

        Returns:
            List of pass names assigned during registration
        """
        return self.pass_names


# Initialize the pass registry when the module is loaded
PassManager._register_passes()
