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
from typing import Any

# Add tests/st to path for importing harness
_SCRIPT_DIR = Path(__file__).parent
_TESTS_ST_DIR = _SCRIPT_DIR.parent
if str(_TESTS_ST_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_ST_DIR))

# noqa: E402 - Import after path modification
from fuzz.src.multi_kernel_test_generator import MultiKernelTestGenerator  # noqa: E402


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate multi-kernel fuzzing test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Generate test cases with default configuration
  python example_multi_kernel.py

  # Generate 5 test cases
  python example_multi_kernel.py --num-cases 5

  # Specify configuration index (starting from 0)
  python example_multi_kernel.py --config-index 0

  # Generate 3 test cases from configuration 0
  python example_multi_kernel.py --config-index 0 --num-cases 3

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
        help="Specify the configuration index to use (starting from 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: generated/test_fuzz_multi_kernel.py)",
    )
    parser.add_argument("--atol", type=float, default=5e-5, help="Absolute error tolerance")
    parser.add_argument("--rtol", type=float, default=5e-5, help="Relative error tolerance")
    parser.add_argument(
        "--advanced-ops-prob",
        type=float,
        default=0.5,
        help="Probability of selecting advanced operators (0.0-1.0)",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=None,
        help="Number of test cases to generate",
    )

    return parser


def get_default_configs() -> list[dict[str, Any]]:
    """Get default test configurations."""
    return [
        {
            "name": "fuzz_sequential_simple",
            "num_instances": 1,
            "seed": 40,
            "enable_advanced_ops": False,
            "num_kernels": 2,
            "mode": "sequential",
            "shape": (128, 128),
            "num_ops_range": (7, 10),
            "tensor_init_type": "random",
            "input_shapes_list": [[(128, 128), (128, 128)]],
            "description": "Branching execution: 2 kernels, same dimension inputs",
        },
    ]


def select_configs(all_configs: list[dict[str, Any]], config_index: int | None) -> list[dict[str, Any]]:
    """Select configurations based on config_index."""
    if config_index is None:
        return all_configs
    if config_index < 0 or config_index >= len(all_configs):
        print(f"Error: configuration index {config_index} out of range (0-{len(all_configs) - 1})")
        sys.exit(1)
    return [all_configs[config_index]]


def apply_num_cases_override(configs: list[dict[str, Any]], num_cases: int | None) -> None:
    """Override num_instances if num_cases is specified."""
    if num_cases is None:
        return
    if num_cases <= 0:
        print(f"Error: --num-cases must be positive, got {num_cases}")
        sys.exit(1)
    for config in configs:
        config["num_instances"] = num_cases


def print_summary(configs: list[dict[str, Any]], output_path: Path, args: argparse.Namespace) -> None:
    """Print generation summary."""
    total_test_cases = sum(config.get("num_instances", 1) for config in configs)

    print("Multi-kernel fuzzing test generator")
    print("=" * 60)
    print(f"Number of configurations: {len(configs)}")
    print(f"Total test cases: {total_test_cases}")
    print(f"Output file: {output_path}")
    print(f"Absolute error tolerance (atol): {args.atol}")
    print(f"Relative error tolerance (rtol): {args.rtol}")
    print(f"Advanced operators probability: {args.advanced_ops_prob}")
    print("=" * 60)
    print()


def print_config_details(configs: list[dict[str, Any]]) -> None:
    """Print detailed configuration information."""
    print("Will generate the following test cases:")
    print()
    test_case_num = 1
    for config_idx, config in enumerate(configs):
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


def expand_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand configurations to create individual test configs."""
    expanded = []
    for config in configs:
        num_instances = config.get("num_instances", 1)
        base_seed = config.get("seed")

        for instance_idx in range(num_instances):
            test_config = config.copy()
            if num_instances > 1:
                test_config["name"] = f"{config['name']}_{instance_idx}"
                test_config["seed"] = (base_seed + instance_idx) if base_seed is not None else instance_idx
            expanded.append(test_config)

    return expanded


def generate_test_cases(expanded_configs: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    """Generate test case code for all configurations."""
    all_test_cases = []
    for test_config in expanded_configs:
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

    return all_test_cases


def generate_test_suite(expanded_configs: list[dict[str, Any]]) -> str:
    """Generate test suite class code."""
    test_suite = '''
class TestMultiKernelFuzzing:
    """Multi-kernel fuzzing test suite"""

'''
    for test_config in expanded_configs:
        test_name = test_config["name"]
        test_suite += f'''    def test_{test_name}(self, test_runner):
        """Test {test_name}"""
        test_case = Test{test_name.title().replace("_", "")}()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {{result.error}}"

'''
    return test_suite


def write_test_file(output_path: Path, test_cases: list[str], test_suite: str) -> None:
    """Write generated test cases to file."""
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_header)
        f.write("\n\n".join(test_cases))
        f.write("\n\n")
        f.write(test_suite)


def main() -> None:
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _SCRIPT_DIR / "generated" / "test_fuzz_multi_kernel.py"

    all_configs = get_default_configs()
    selected_configs = select_configs(all_configs, args.config_index)
    apply_num_cases_override(selected_configs, args.num_cases)

    print_summary(selected_configs, output_path, args)
    print_config_details(selected_configs)

    print("Generating test file...")
    expanded_configs = expand_configs(selected_configs)
    test_cases = generate_test_cases(expanded_configs, args)
    test_suite = generate_test_suite(expanded_configs)
    write_test_file(output_path, test_cases, test_suite)

    print()
    print(f"✓ Successfully generated {len(expanded_configs)} test case(s)")
    print(f"✓ Output file: {output_path}")
    print()
    print("Run tests:")
    print(f"  pytest {output_path}")
    print()


if __name__ == "__main__":
    main()
