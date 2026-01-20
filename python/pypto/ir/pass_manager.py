# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pass manager for IR transformations."""

from enum import Enum
from typing import Callable, Dict, List, Tuple, Union

from pypto.pypto_core import ir as core_ir
from pypto.pypto_core import passes


class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""

    Default = "Default"  # No optimization
    Custom1 = "Custom1"  # Custom optimization strategy 1
    Custom2 = "Custom2"  # Custom optimization strategy 2


class PassManager:
    """Manager for organizing and executing IR transformation passes.

    PassManager maintains a sequence of Pass instances for different optimization
    strategies and executes them in order on a given Function or Program. It uses
    a pipeline model where each pass's output becomes the input to the next pass.

    Usage:
        # Get a pre-configured strategy
        pm = PassManager.get_strategy(OptimizationStrategy.Custom2)
        result = pm.run_passes(func)  # For Function
        result = pm.run_passes(program)  # For Program

        # Or use the shorthand
        result = PassManager.get_strategy(OptimizationStrategy.Custom2).run_passes(func)
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
            ],
            OptimizationStrategy.Custom1: [
                # Custom optimization strategy 1
                ("IdentityPass_1", lambda: passes.IdentityPass()),
            ],
            OptimizationStrategy.Custom2: [
                # Custom optimization strategy 2
                ("IdentityPass_1", lambda: passes.IdentityPass()),
                ("IdentityPass_2", lambda: passes.IdentityPass()),
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
            result = pm.run_passes(func)

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

    def run_passes(self, input_ir: Union[core_ir.Function, core_ir.Program]):
        """Execute all passes in sequence on a Function or Program.

        Each pass's output becomes the input to the next pass.
        For Program inputs, all passes are applied to each function in the program.

        Args:
            input_ir: Input Function or Program to transform

        Returns:
            Transformed Function or Program after all passes have been applied
        """
        if isinstance(input_ir, core_ir.Program):
            # Apply passes to each function in the program
            transformed_functions = []
            for global_var, func in input_ir.functions.items():
                transformed_func = func
                for pass_instance in self.passes:
                    transformed_func = pass_instance.run(transformed_func)
                transformed_functions.append(transformed_func)

            # Create a new Program with the transformed functions
            return core_ir.Program(transformed_functions, input_ir.name, input_ir.span)
        else:
            # For Function input, apply passes in sequence
            current = input_ir
            for pass_instance in self.passes:
                current = pass_instance.run(current)
            return current

    def get_pass_names(self) -> List[str]:
        """Get the names of all passes in this manager.

        Returns:
            List of pass names assigned during registration
        """
        return self.pass_names


# Initialize the pass registry when the module is loaded
PassManager._register_passes()
