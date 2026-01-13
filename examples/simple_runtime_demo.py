#!/usr/bin/env python3
"""
Simple Runtime Machine Demo

This example demonstrates the basic functionality of the PyPTO runtime machine.
Note: There may be a thread cleanup warning on exit, but the program executes correctly.
"""

import time
import pypto.runtime as rt

def main():
    print("=" * 60)
    print("PyPTO Runtime Machine - Simple Demo")
    print("=" * 60)

    # Create machine with 2 AICOREs (RuntimeMachine is the AICPU host)
    print("\n1. Creating RuntimeMachine...")
    machine = rt.RuntimeMachine(num_aicore=2)

    # Example 1: Simple arithmetic on AICPU (no AICORE tasks)
    print("\n2. Running simple arithmetic program on AICPU host...")
    prog1 = rt.RuntimeProgram()
    r0, r1, r2, r3 = (prog1.reg(i) for i in range(4))
    prog1.const(r0, 10)
    prog1.const(r1, 20)
    prog1.add(r2, r0, r1)  # r2 = 30
    prog1.mul(r3, r2, r0)  # r3 = 300
    prog1.halt()

    print("\nProgram:")
    print(prog1.to_python_syntax())

    machine.load_and_run_program(prog1)  # Synchronous execution
    print("✓ Program 1 completed successfully!")

    # Example 2: Task dispatch to AICORE
    print("\n3. Dispatching tasks to AICORE workers...")

    # Register AICORE task
    def multiply(a, b):
        """Simulate computation on AICORE"""
        time.sleep(0.01)  # Simulate work
        result = a * b
        print(f"  [AICORE] multiply({a}, {b}) = {result}")
        return result

    machine.register_task("mul_task", multiply)

    prog2 = rt.RuntimeProgram()
    r0, r1, r2 = (prog2.reg(i) for i in range(3))
    prog2.const(r0, 5)
    prog2.const(r1, 7)
    prog2.dispatch(r2, "mul_task", [r0, r1])  # Async dispatch to AICORE
    prog2.wait(r2)  # AICPU host waits for AICORE completion
    prog2.halt()

    print("\nProgram:")
    print(prog2.to_python_syntax())

    machine.load_and_run_program(prog2)  # Synchronous execution
    print("✓ Program 2 completed successfully!")

    # Example 3: Multiple parallel tasks
    print("\n4. Running multiple parallel tasks...")

    def add(a, b):
        time.sleep(0.01)
        return a + b

    machine.register_task("add_task", add)

    prog3 = rt.RuntimeProgram()
    r0, r1, r2 = (prog3.reg(i) for i in range(3))
    r10, r11, r12 = (prog3.reg(i) for i in range(10, 13))
    prog3.const(r0, 10)
    prog3.const(r1, 20)
    prog3.const(r2, 30)

    # Dispatch multiple tasks (run in parallel on AICORE workers)
    prog3.dispatch(r10, "add_task", [r0, r1])
    prog3.dispatch(r11, "mul_task", [r1, r2])
    prog3.dispatch(r12, "add_task", [r0, r2])

    # AICPU host waits for all AICORE tasks
    prog3.wait_all([r10, r11, r12])
    prog3.halt()

    print("\nProgram:")
    print(prog3.to_python_syntax())

    start = time.time()
    machine.load_and_run_program(prog3)  # Synchronous execution
    elapsed = time.time() - start
    print(f"✓ Program 3 completed in {elapsed:.3f}s (parallel execution on AICOREs)")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nNote: AICORE workers are automatically cleaned up when the machine is destroyed.")

if __name__ == "__main__":
    main()
