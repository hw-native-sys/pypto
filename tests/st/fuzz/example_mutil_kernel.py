# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Multi-kernel fuzzing framework usage example

This script demonstrates how to use the multi-kernel test generator to create test cases.
Supports controlling the number and configuration of generated test cases via command-line arguments.

Usage:
    python example_multi_kernel.py --num-cases 5
"""

import argparse
import sys
from pathlib import Path

# Add tests/st to path for importing harness
_SCRIPT_DIR = Path(__file__).parent
_TESTS_ST_DIR = _SCRIPT_DIR.parent
if str(_TESTS_ST_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_ST_DIR))

# noqa: E402 - Import after path modification
from fuzz.src.multi_kernel_test_generator import MultiKernelTestGenerator  # noqa: E402


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate multi-kernel fuzzing test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Generate test cases with default configuration
  python example_multi_kernel.py

  # Specify configuration index (starting from 0)
  python example_multi_kernel.py --config-index 0

  # Specify output file
  python example_multi_kernel.py --output custom_test.py

  # Set error tolerance
  python example_multi_kernel.py --atol 1e-3 --rtol 1e-3

  # Set advanced operators probability (0.0-1.0)
  python example_multi_kernel.py --advanced-ops-prob 0.7

  # Combined usage
  python example_multi_kernel.py --config-index 1 --atol 1e-4 --rtol 1e-4 \\
      --advanced-ops-prob 0.5 --output my_test.py
        """,
    )

    parser.add_argument(
        "--config-index",
        type=int,
        default=0,
        help=(
            "Specify the configuration index to use (starting from 0), "
            "if not specified use all configurations"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: src/fuzzer/generated_tests/test_fuzz_multi_kernel.py)",
    )

    parser.add_argument("--atol", type=float, default=5e-5, help="Absolute error tolerance (default: 1e-4)")

    parser.add_argument("--rtol", type=float, default=5e-5, help="Relative error tolerance (default: 1e-4)")

    parser.add_argument(
        "--advanced-ops-prob",
        type=float,
        default=0.5,
        help="Probability of selecting advanced operators (0.0-1.0, default: 0.5)",
    )

    args = parser.parse_args()

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _SCRIPT_DIR / "generated" / "test_fuzz_multi_kernel.py"

    # Define test cases with different configurations
    # Each configuration can generate multiple test instances (via num_instances control)
    all_configs = [
        {
            "name": "fuzz_sequential_simple",
            "num_instances": 1,  # Generate 1 test case from this configuration
            "seed": 40,
            "enable_advanced_ops": True,  # Enable advanced operators
            "num_kernels": 1,
            "mode": "sequential",
            "shape": (128, 128),
            "num_ops_range": (
                5,
                5,
            ),  # Increase operation count to improve generation probability of row_expand
            "tensor_init_type": "random",
            "input_shapes_list": [
                [(128, 128), (128, 128)],  # kernel_0: 2 inputs with same dimensions
            ],
            "description": "Simple sequential execution: 1 kernel, same dimension inputs",
        },
    ]

    # Select configurations based on config_index
    if args.config_index is not None:
        if args.config_index < 0 or args.config_index >= len(all_configs):
            print(f"Error: configuration index {args.config_index} out of range (0-{len(all_configs) - 1})")
            return
        selected_configs = [all_configs[args.config_index]]
    else:
        selected_configs = all_configs

    # Calculate total test cases
    total_test_cases = sum(config.get("num_instances", 1) for config in selected_configs)

    print("Multi-kernel fuzzing test generator")
    print("=" * 60)
    print(f"Number of configurations: {len(selected_configs)}")
    print(f"Total test cases: {total_test_cases}")
    print(f"Output file: {output_path}")
    print(f"Absolute error tolerance (atol): {args.atol}")
    print(f"Relative error tolerance (rtol): {args.rtol}")
    print(f"Advanced operators probability: {args.advanced_ops_prob}")
    print("=" * 60)
    print()

    print("Will generate the following test cases:")
    print()
    test_case_num = 1
    for config_idx, config in enumerate(selected_configs):
        num_instances = config.get("num_instances", 1)
        print(f"Configuration {config_idx}: {config['name']}")
        print(f"   {config['description']}")
        print(f"   Number of instances: {num_instances}")
        print(f"   Random seed: {config.get('seed', 42)}")
        print(f"   Enable advanced operators: {'Yes' if config.get('enable_advanced_ops', False) else 'No'}")
        print(f"   Tensor initialization: {config.get('tensor_init_type', 'constant')}")
        if num_instances > 1:
            print(f"   Will generate test cases: {test_case_num} - {test_case_num + num_instances - 1}")
        else:
            print(f"   Will generate test case: {test_case_num}")
        test_case_num += num_instances
        print()

    # Expand configurations, create a test case for each instance
    expanded_test_configs = []
    for config in selected_configs:
        num_instances = config.get("num_instances", 1)
        base_seed = config.get("seed")

        for instance_idx in range(num_instances):
            # Create a configuration copy for each instance
            test_config = config.copy()

            # If there are multiple instances, add index after name
            if num_instances > 1:
                test_config["name"] = f"{config['name']}_{instance_idx}"
                # Each instance uses different seed
                if base_seed is not None:
                    test_config["seed"] = base_seed + instance_idx
                else:
                    test_config["seed"] = instance_idx

            expanded_test_configs.append(test_config)

    # Generate test file
    print("Generating test file...")

    # Create independent generator for each configuration (using respective seed and configuration)
    all_test_cases = []
    for test_config in expanded_test_configs:
        generator = MultiKernelTestGenerator(
            seed=test_config.get("seed", 42),
            enable_advanced_ops=test_config.get("enable_advanced_ops", False),
            advanced_ops_probability=args.advanced_ops_prob,
            tensor_init_type=test_config.get("tensor_init_type", "constant"),
        )

        test_code = generator.generate_test_case(
            test_name=test_config["name"],
            num_kernels=test_config.get("num_kernels", 3),
            orchestration_mode=test_config.get("mode", "sequential"),
            shape=test_config.get("shape", (128, 128)),
            num_ops_range=test_config.get("num_ops_range", (3, 7)),
            input_shapes_list=test_config.get("input_shapes_list"),
            tensor_init_type=test_config.get("tensor_init_type"),
            atol=args.atol,
            rtol=args.rtol,
        )
        all_test_cases.append(test_code)

    # Generate file header
    file_header = '''"""
Auto-generated multi-kernel fuzzing test cases

This file is automatically generated by MultiKernelTestGenerator.
Contains multiple test cases, each with multiple InCore kernels and an Orchestration function.
"""

import sys
from pathlib import Path
from typing import Any, List

import torch
import pytest

from harness.core.harness import DataType, PTOTestCase, TensorSpec


'''

    # Generate test suite class
    test_suite = '''
class TestMultiKernelFuzzing:
    """Multi-kernel fuzzing test suite"""

'''

    # Add test method for each test case
    for test_config in expanded_test_configs:
        test_name = test_config["name"]
        test_suite += f'''    def test_{test_name}(self, test_runner):
        """Test {test_name}"""
        test_case = Test{test_name.title().replace("_", "")}()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {{result.error}}"

'''

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_header)
        f.write("\n\n".join(all_test_cases))
        f.write("\n\n")
        f.write(test_suite)

    print()
    print(f"✓ Successfully generated {len(expanded_test_configs)} test case(s)")
    print(f"✓ Output file: {output_path}")
    print()
    print("Run tests:")
    print(f"  pytest {output_path}")
    print()


if __name__ == "__main__":
    main()
