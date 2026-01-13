# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for basic runtime machine functionality."""

import time

import pytest

import pypto.runtime as rt


def test_runtime_machine_creation():
    """Test creating a runtime machine."""
    machine = rt.RuntimeMachine(num_aicore=4)
    assert machine.get_num_aicore() == 4


def test_simple_program_no_tasks():
    """Test a simple program with arithmetic but no AICORE tasks."""
    machine = rt.RuntimeMachine(num_aicore=2)

    program = rt.RuntimeProgram()
    r0, r1, r2, r3, r4 = (program.reg(i) for i in range(5))
    program.const(r0, 10)
    program.const(r1, 20)
    program.add(r2, r0, r1)  # r2 = 30
    program.sub(r3, r2, r0)  # r3 = 20
    program.mul(r4, r3, r1)  # r4 = 400
    program.halt()

    # Print program
    print("\n" + program.to_python_syntax())

    # Execute synchronously (blocks until completion)
    machine.load_and_run_program(program)
    print("Program execution complete")


def test_program_with_task_dispatch():
    """Test dispatching a task to AICORE."""
    machine = rt.RuntimeMachine(num_aicore=2)

    # Register AICORE task
    def add_task(a, b):
        """Add two numbers on AICORE."""
        return a + b

    machine.register_task("add", add_task)

    # Create program
    program = rt.RuntimeProgram()
    r0, r1, r2 = (program.reg(i) for i in range(3))
    program.const(r0, 100)
    program.const(r1, 200)
    program.dispatch(r2, "add", [r0, r1])  # Async dispatch to AICORE
    program.wait(r2)  # Wait for AICORE task completion
    program.halt()

    print("\n" + program.to_python_syntax())

    # Execute synchronously (blocks until completion)
    machine.load_and_run_program(program)


def test_async_dispatch_multiple_tasks():
    """Test dispatching multiple tasks asynchronously."""
    machine = rt.RuntimeMachine(num_aicore=4)

    # Register tasks
    def multiply(a, b):
        time.sleep(0.01)  # Simulate work
        return a * b

    def add(a, b):
        time.sleep(0.01)  # Simulate work
        return a + b

    machine.register_task("mul", multiply)
    machine.register_task("add", add)

    # Create program
    program = rt.RuntimeProgram()
    r0, r1, r2 = (program.reg(i) for i in range(3))
    r10, r11, r12 = (program.reg(i) for i in range(10, 13))

    program.const(r0, 10)
    program.const(r1, 20)
    program.const(r2, 30)

    # Dispatch multiple tasks (all run in parallel on AICORE workers)
    program.dispatch(r10, "mul", [r0, r1])  # r10 = 10*20 = 200
    program.dispatch(r11, "mul", [r1, r2])  # r11 = 20*30 = 600
    program.dispatch(r12, "add", [r0, r2])  # r12 = 10+30 = 40

    # Wait for all AICORE tasks to complete
    program.wait_all([r10, r11, r12])
    program.halt()

    print("\n" + program.to_python_syntax())

    # Execute synchronously (blocks until completion)
    start_time = time.time()
    machine.load_and_run_program(program)
    elapsed = time.time() - start_time

    # Should be faster than sequential (< 0.04s instead of 0.03s)
    print(f"Elapsed time: {elapsed:.4f}s")
    assert elapsed < 0.04, "Tasks should run in parallel on AICORE workers"


def test_control_flow_jumps():
    """Test control flow with jumps."""
    machine = rt.RuntimeMachine(num_aicore=2)

    program = rt.RuntimeProgram()
    r0, r1, r2, r3 = (program.reg(i) for i in range(4))

    program.const(r0, 10)
    program.const(r1, 10)
    program.cmp_eq(r2, r0, r1)  # r2 = 1 (true)
    program.jump_if_not_zero(r2, "equal_branch")

    # Not equal branch (skipped)
    program.const(r3, 999)
    program.jump("done")

    # Equal branch (executed)
    program.add_label("equal_branch")
    program.const(r3, 42)

    program.add_label("done")
    program.halt()

    print("\n" + program.to_python_syntax())

    # Execute synchronously
    machine.load_and_run_program(program)


def test_memory_operations():
    """Test shared memory operations."""
    machine = rt.RuntimeMachine(num_aicore=2)

    program = rt.RuntimeProgram()
    r0, r1 = (program.reg(i) for i in range(2))

    program.const(r0, 12345)
    program.store_mem(0x1000, r0)  # Write to memory
    program.const(r1, 0)  # Clear r1
    program.load_mem(r1, 0x1000)  # Read from memory
    program.halt()

    print("\n" + program.to_python_syntax())

    # Execute synchronously
    machine.load_and_run_program(program)

    # Check memory
    memory = machine.get_memory()
    assert memory.contains(0x1000)


def test_program_builder_fluent_api():
    """Test the fluent API ProgramBuilder."""
    machine = rt.RuntimeMachine(num_aicore=2)

    # Register task
    machine.register_task("add", lambda a, b: a + b)

    # Build program with fluent API
    builder = rt.ProgramBuilder()
    r0, r1, r2, r3 = (builder.reg(i) for i in range(4))

    program = (
        builder
        .const(r0, 10)
        .const(r1, 20)
        .add(r2, r0, r1)
        .dispatch(r3, "add", [r0, r1])
        .wait(r3)
        .halt()
        .build()
    )

    print("\n" + program.to_python_syntax())

    # Execute synchronously
    machine.load_and_run_program(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
