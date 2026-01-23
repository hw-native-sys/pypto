# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Compile - Complete compilation pipeline from IR to PTO assembly."""

import os
import shutil
import tempfile

from pypto import DataType, ir
from pypto.ir import IRBuilder


def test_compile_basic():
    """Test basic Compile functionality - full pipeline execution."""
    # Create a simple program
    ib = IRBuilder()

    with ib.function("test_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        result = ib.let("result", ir.op.block.muls(x, 2.0))
        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "compile_output")

        # Compile the program
        result_dir = ir.compile(
            program, output_dir=output_dir, strategy=ir.OptimizationStrategy.Default, dump_passes=False
        )

        # Verify output directory was created and returned
        assert result_dir == output_dir
        assert os.path.isdir(result_dir)

        # Verify frontend IR was dumped
        frontend_path = os.path.join(result_dir, "00_frontend.py")
        assert os.path.isfile(frontend_path)

        # Verify PTO assembly was generated
        pto_path = os.path.join(result_dir, "output.pto")
        assert os.path.isfile(pto_path)

        # Verify PTO content
        with open(pto_path, "r") as f:
            pto_code = f.read()
        assert "func @test_func" in pto_code
        assert "tmuls" in pto_code


def test_compile_with_default_output_dir():
    """Test Compile with default output directory (build_output/<program_name>)."""
    # Create a simple program
    ib = IRBuilder()

    with ib.function("simple_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))
        ib.return_stmt(x)

    func = f.get_result()
    program = ir.Program([func], "default_output_test", ir.Span.unknown())

    # Compile with default output directory
    result_dir = ir.compile(program, strategy=ir.OptimizationStrategy.Default, dump_passes=False)

    try:
        # Verify default directory was used with timestamp pattern
        # Directory should be: build_output/default_output_test_YYYYMMDD_HHMMSS
        assert result_dir.startswith(os.path.join("build_output", "default_output_test_"))
        assert os.path.isdir(result_dir)

        # Verify files exist
        assert os.path.isfile(os.path.join(result_dir, "00_frontend.py"))
        assert os.path.isfile(os.path.join(result_dir, "output.pto"))

    finally:
        # Clean up the default output directory
        if os.path.exists("build_output"):
            shutil.rmtree("build_output")


def test_compile_with_passes_dump():
    """Test Compile with pass dumps enabled."""
    # Create a simple program
    ib = IRBuilder()

    with ib.function("test_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        result = ib.let("result", ir.op.block.muls(x, 2.0))
        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "compile_with_dumps")

        # Compile with Custom2 strategy (has 2 IdentityPasses) and dump_passes=True
        result_dir = ir.compile(
            program, output_dir=output_dir, strategy=ir.OptimizationStrategy.Custom2, dump_passes=True
        )

        # Verify output directory
        assert os.path.isdir(result_dir)

        # Verify frontend IR was dumped
        frontend_path = os.path.join(result_dir, "00_frontend.py")
        assert os.path.isfile(frontend_path)

        # Verify pass dumps exist
        # Custom2 has 2 IdentityPasses
        pass1_path = os.path.join(result_dir, "01_after_IdentityPass_1.py")
        pass2_path = os.path.join(result_dir, "02_after_IdentityPass_2.py")
        assert os.path.isfile(pass1_path), f"Expected pass dump at {pass1_path}"
        assert os.path.isfile(pass2_path), f"Expected pass dump at {pass2_path}"

        # Verify PTO assembly was generated
        pto_path = os.path.join(result_dir, "output.pto")
        assert os.path.isfile(pto_path)

        # Verify pass dumps contain valid Python code
        with open(pass1_path, "r") as f:
            pass1_content = f.read()
        assert "test_func_identity" in pass1_content  # Function name should be modified by IdentityPass

        with open(pass2_path, "r") as f:
            pass2_content = f.read()
        assert "test_func_identity_identity" in pass2_content  # Second identity pass


def test_compile_without_passes_dump():
    """Test Compile with pass dumps disabled."""
    # Create a simple program
    ib = IRBuilder()

    with ib.function("test_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        result = ib.let("result", ir.op.block.muls(x, 2.0))
        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "compile_no_dumps")

        # Compile with Custom2 strategy but dump_passes=False
        result_dir = ir.compile(
            program, output_dir=output_dir, strategy=ir.OptimizationStrategy.Custom2, dump_passes=False
        )

        # Verify output directory
        assert os.path.isdir(result_dir)

        # Verify frontend IR was dumped
        frontend_path = os.path.join(result_dir, "00_frontend.py")
        assert os.path.isfile(frontend_path)

        # Verify pass dumps do NOT exist
        pass1_path = os.path.join(result_dir, "01_after_IdentityPass_1.py")
        pass2_path = os.path.join(result_dir, "02_after_IdentityPass_2.py")
        assert not os.path.isfile(pass1_path), f"Pass dump should not exist at {pass1_path}"
        assert not os.path.isfile(pass2_path), f"Pass dump should not exist at {pass2_path}"

        # Verify PTO assembly was generated
        pto_path = os.path.join(result_dir, "output.pto")
        assert os.path.isfile(pto_path)


def test_compile_with_custom1_strategy():
    """Test Compile with Custom1 optimization strategy."""
    # Create a simple program
    ib = IRBuilder()

    with ib.function("test_func") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        result = ib.let("result", ir.op.block.muls(x, 2.0))
        ib.return_stmt(result)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "compile_custom1")

        # Compile with Custom1 strategy (has 1 IdentityPass)
        result_dir = ir.compile(
            program, output_dir=output_dir, strategy=ir.OptimizationStrategy.Custom1, dump_passes=True
        )

        # Verify output directory
        assert os.path.isdir(result_dir)

        # Verify pass dump exists (Custom1 has 1 IdentityPass)
        pass1_path = os.path.join(result_dir, "01_after_IdentityPass_1.py")
        assert os.path.isfile(pass1_path)

        # Verify pass dump contains correct function name
        with open(pass1_path, "r") as f:
            pass1_content = f.read()
        assert "test_func_identity" in pass1_content


def test_compile_original_ir_content():
    """Test that original IR dump contains correct content."""
    # Create a simple program
    ib = IRBuilder()

    with ib.function("my_function") as f:
        x = f.param("x", ir.TileType([16, 16], DataType.FP32))
        f.return_type(ir.TileType([16, 16], DataType.FP32))

        y = ib.let("y", ir.op.block.adds(x, 5.0))
        ib.return_stmt(y)

    func = f.get_result()
    program = ir.Program([func], "original_test", ir.Span.unknown())

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "compile_original")

        # Compile
        result_dir = ir.compile(program, output_dir=output_dir, dump_passes=False)

        # Read frontend IR
        frontend_path = os.path.join(result_dir, "00_frontend.py")
        with open(frontend_path, "r") as f:
            frontend_content = f.read()

        # Verify content contains expected elements
        assert "my_function" in frontend_content
        assert "x" in frontend_content
        assert "y" in frontend_content
        # The python_print should generate valid Python-style IR
        assert "=" in frontend_content  # Assignment statements


def test_compile_pto_output_content():
    """Test that PTO output contains correct assembly."""
    # Create a simple program
    ib = IRBuilder()

    with ib.function("pto_test") as f:
        x = f.param("x", ir.TileType([8, 8], DataType.FP32))
        y = f.param("y", ir.TileType([8, 8], DataType.FP32))
        f.return_type(ir.TileType([8, 8], DataType.FP32))

        z = ib.let("z", ir.op.block.mul(x, y))
        w = ib.let("w", ir.op.block.add(z, x))
        ib.return_stmt(w)

    func = f.get_result()
    program = ir.Program([func], "pto_test_program", ir.Span.unknown())

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "compile_pto")

        # Compile
        result_dir = ir.compile(program, output_dir=output_dir, dump_passes=False)

        # Read PTO assembly
        pto_path = os.path.join(result_dir, "output.pto")
        with open(pto_path, "r") as f:
            pto_content = f.read()

        # Verify PTO assembly content
        assert "func @pto_test" in pto_content
        assert "tmul" in pto_content  # Binary multiply operation
        assert "tadd" in pto_content  # Binary add operation
        assert "return" in pto_content


def test_compile_multiple_functions():
    """Test Compile with a program containing multiple functions."""
    # Create program with two functions
    ib1 = IRBuilder()
    with ib1.function("func1") as f1:
        x = f1.param("x", ir.TileType([8, 8], DataType.FP32))
        f1.return_type(ir.TileType([8, 8], DataType.FP32))
        result = ib1.let("result", ir.op.block.muls(x, 2.0))
        ib1.return_stmt(result)
    func1 = f1.get_result()

    ib2 = IRBuilder()
    with ib2.function("func2") as f2:
        y = f2.param("y", ir.TileType([8, 8], DataType.FP32))
        f2.return_type(ir.TileType([8, 8], DataType.FP32))
        result = ib2.let("result", ir.op.block.adds(y, 1.0))
        ib2.return_stmt(result)
    func2 = f2.get_result()

    program = ir.Program([func1, func2], "multi_func_program", ir.Span.unknown())

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "compile_multi")

        # Compile
        result_dir = ir.compile(
            program, output_dir=output_dir, strategy=ir.OptimizationStrategy.Custom1, dump_passes=True
        )

        # Verify files exist
        assert os.path.isfile(os.path.join(result_dir, "00_frontend.py"))
        assert os.path.isfile(os.path.join(result_dir, "01_after_IdentityPass_1.py"))
        assert os.path.isfile(os.path.join(result_dir, "output.pto"))

        # Verify PTO contains both functions
        pto_path = os.path.join(result_dir, "output.pto")
        with open(pto_path, "r") as f:
            pto_content = f.read()
        assert "func @func1_identity" in pto_content  # Modified by IdentityPass
        assert "func @func2_identity" in pto_content  # Modified by IdentityPass
