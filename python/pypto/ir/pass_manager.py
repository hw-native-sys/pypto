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
from typing import Callable, Dict, List, Tuple

from pypto.pypto_core import passes


class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""

    Default = "Default"  # No optimization
    O1 = "O1"  # Basic optimization
    O2 = "O2"  # Standard optimization
    O3 = "O3"  # Aggressive optimization


class PassManager:
    """Manager for organizing and executing IR transformation passes.

    PassManager maintains a sequence of Pass instances for different optimization
    strategies and executes them in order on a given Function. It uses a pipeline
    model where each pass's output becomes the input to the next pass.

    Usage:
        # Get a pre-configured strategy
        pm = PassManager.get_strategy(OptimizationStrategy.O2)
        result = pm.run(func)

        # Or use the shorthand
        result = PassManager.get_strategy(OptimizationStrategy.O2).run(func)
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
            OptimizationStrategy.O1: [
                # Basic optimization
                ("IdentityPass_1", lambda: passes.IdentityPass()),
            ],
            OptimizationStrategy.O2: [
                # Standard optimization
                ("IdentityPass_1", lambda: passes.IdentityPass()),
                ("IdentityPass_2", lambda: passes.IdentityPass()),
            ],
            OptimizationStrategy.O3: [
                # Aggressive optimization
                ("IdentityPass_1", lambda: passes.IdentityPass()),
                ("IdentityPass_2", lambda: passes.IdentityPass()),
                ("IdentityPass_3", lambda: passes.IdentityPass()),
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
            pm = PassManager.get_strategy(OptimizationStrategy.O2)
            result = pm.run(func)

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

    def run(self, func):
        """Execute all passes in sequence on a function.

        Each pass's output becomes the input to the next pass.

        Args:
            func: Input Function to transform

        Returns:
            Transformed Function after all passes have been applied
        """
        current = func
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
