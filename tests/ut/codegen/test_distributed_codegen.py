# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for distributed Python code generation."""

import re

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes


class TestDistributedCodegen:
    """Test distributed Python codegen on outlined hierarchy programs."""

    def test_chip_sub_worker_and_orchestrator(self):
        """HOST orchestrator calling CHIP orchestrator → CHIP worker produces submit_next_level."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(x)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_orch(x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Verify imports
        assert "from simpler.task_interface import TaskArgs, TensorArgType" in code
        assert "from simpler_setup.torch_interop import make_tensor_arg" in code

        # Verify function definition
        assert "def host_orch" in code
        assert "orch, _args, config" in code

        # Verify call-site lowering: CHIP orchestrator → submit_next_level
        assert "submit_next_level" in code
        assert 'callables["chip_orch"]' in code
        assert "TaskArgs()" in code

    def test_sub_worker_submit_sub(self):
        """HOST worker (SubWorker) produces submit_sub call."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                self.verify(x)
                return x

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # HOST worker (level 3) → submit_sub
        assert "submit_sub" in code
        assert 'sub_ids["verify"]' in code

    def test_chip_and_sub_worker_combined(self):
        """Program with both CHIP orchestrator (→ chip worker) and HOST SubWorker."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, b)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                f: pl.Tensor[[64], pl.FP32] = self.chip_orch(a, b)
                self.verify(f)
                return f

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "submit_next_level" in code
        assert "submit_sub" in code
        assert "TensorArgType.INPUT" in code

    def test_for_loop_codegen(self):
        """ForStmt in function body produces Python for loop."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.POD, role=pl.Role.Orchestrator)
            def orch_with_loop(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(0, 4):
                    y = pl.add(y, x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "for " in code
        assert "in range(" in code

    def test_python_imports(self):
        """Generated code contains required Python imports."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def simple_worker(x: pl.Tensor[[64], pl.FP32]):
                pass

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "from simpler.task_interface import TaskArgs, TensorArgType" in code
        assert "from simpler_setup.torch_interop import make_tensor_arg" in code

    def test_tensor_arg_type_tags(self):
        """Parameter directions map to correct TensorArgType tags."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, b, f)
                return out

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out: pl.Tensor[[64], pl.FP32] = self.chip_orch(a, b, f)
                return out

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "TensorArgType.INPUT" in code
        assert "TensorArgType.OUTPUT_EXISTING" in code

    def test_bool_constants(self):
        """Boolean constants use Python True/False, not C++ true/false."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def worker(x: pl.Tensor[[64], pl.FP32]):
                pass

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Python uses True/False, not true/false
        assert "true" not in code.lower() or "True" in code or "False" in code

    def test_sub_worker_pure_python_body(self):
        """HOST Worker with pure Python body is captured without DSL parsing."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[128, 128], pl.FP32]):
                import torch  # noqa: PLC0415

                expected = torch.full((128, 128), 5.0, dtype=torch.float32)
                assert torch.allclose(f, expected)  # pyright: ignore[reportArgumentType]

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                self.verify(x)
                return x

        # Should not raise — pure Python body is skipped during DSL parsing
        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "submit_sub" in code
        assert 'sub_ids["verify"]' in code

    def test_sub_worker_body_inlined_in_ir(self):
        """SubWorker body is captured as an InlineStmt on the IR Function."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                self.verify(x)
                return x

        verify_fn = Input.get_function("verify")
        assert verify_fn is not None
        assert isinstance(verify_fn.body, ir.InlineStmt)
        assert verify_fn.body.language == ir.InlineLanguage.Python
        assert isinstance(verify_fn.body.body, str)

    def test_create_tensor_emits_shared_torch_zeros(self):
        """tensor.create in HOST orchestrator emits torch.zeros(...).share_memory_()."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                buf: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                buf: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, buf)
                return result

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                buf: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.chip_orch(a, buf)
                return result

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # torch.zeros with share_memory_() emitted
        assert "torch.zeros(" in code
        assert "torch.float32" in code
        assert ".share_memory_()" in code
        assert "import torch" in code

    def test_create_tensor_shared_zeros_for_multiple_tensors(self):
        """Multiple tensor.create calls each emit torch.zeros(...).share_memory_()."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_add(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_sub(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.sub(a, b)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch_add(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out: pl.Tensor[[64], pl.FP32] = self.chip_add(a, b, f)
                return out

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch_sub(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out: pl.Tensor[[64], pl.FP32] = self.chip_sub(a, b, f)
                return out

            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def reduce_sum(
                sum_ab: pl.Tensor[[64], pl.FP32],
                diff_ab: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return f

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                sum_ab: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                diff_ab: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                out_sum: pl.Tensor[[64], pl.FP32] = self.chip_orch_add(a, b, sum_ab)
                out_diff: pl.Tensor[[64], pl.FP32] = self.chip_orch_sub(a, b, diff_ab)
                out_f: pl.Tensor[[64], pl.FP32] = self.reduce_sum(out_sum, out_diff, f)
                return out_f

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Two torch.zeros().share_memory_() calls
        assert code.count("torch.zeros(") == 2
        assert code.count(".share_memory_()") == 2
        # Parameter tensors still use make_tensor_arg(tensors[...])
        assert 'make_tensor_arg(tensors["a' in code
        assert 'make_tensor_arg(tensors["b' in code

    def test_host_orch_create_tensor_hoisted_to_alloc_intermediates(self):
        """HOST-orch tensor.create lifts to _alloc_intermediates(tensors).

        The simpler L3 runtime forks subworker / chip-worker child processes
        inside w.init(); POSIX shared memory created after fork is invisible
        to inherited children. Intermediate tensors created via
        pl.create_tensor in the HOST orchestrator body must therefore be
        allocated *before* w.init() — codegen splits them into a separate
        _alloc_intermediates(tensors) function that the runtime invokes
        pre-init.
        """

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                buf: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                buf: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, buf)
                return result

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                buf: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.chip_orch(a, buf)
                return result

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        alloc_idx = code.find("def _alloc_intermediates(tensors):")
        host_idx = code.find("def host_orch(")
        assert alloc_idx >= 0, f"Missing _alloc_intermediates in:\n{code}"
        assert host_idx >= 0, f"Missing host_orch in:\n{code}"
        assert alloc_idx < host_idx, "_alloc_intermediates must precede host_orch"

        alloc_block = code[alloc_idx:host_idx]
        host_block = code[host_idx:]

        # Allocation lives in _alloc_intermediates only. SSA renames the local
        # so match by structure rather than the literal source name.
        assert "torch.zeros((64,), dtype=torch.float32).share_memory_()" in alloc_block
        match = re.search(r'tensors\["([^"]+)"\] = torch\.zeros\(', alloc_block)
        assert match is not None, f"No tensors[...] = torch.zeros(...) in:\n{alloc_block}"
        hoisted_name = match.group(1)

        # host_orch must NOT re-allocate the hoisted tensor — but it must
        # still pass it via `tensors["<name>"]` to the chip orchestrator.
        assert "torch.zeros(" not in host_block
        assert f'tensors["{hoisted_name}"]' in host_block

    def test_alloc_intermediates_emitted_when_no_creates(self):
        """HOST orchestrator without tensor.create still gets an empty alloc fn.

        Keeping the symbol present simplifies the runtime contract: it can
        unconditionally call _alloc_intermediates(tensors) before w.init().
        """

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(x)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_orch(x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "def _alloc_intermediates(tensors):" in code
        # Body is just `pass` since there are no allocations to hoist.
        alloc_idx = code.find("def _alloc_intermediates(tensors):")
        host_idx = code.find("def host_orch(")
        alloc_block = code[alloc_idx:host_idx]
        assert "    pass" in alloc_block

    def test_tuple_return_pl_tuple(self):
        """Tuple-return worker (pl.Tuple) populates per-element tensors aliases."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out_s: pl.Out[pl.Tensor[[64], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                return out_s, out_d

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out_s: pl.Out[pl.Tensor[[64], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                s, d = self.chip_orch(a, b, out_s, out_d)
                return s, d

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Each tuple element should get its own tensors[...] alias
        assert code.count('tensors["') >= 2
        # submit_next_level emitted for chip_orch
        assert "submit_next_level" in code
        # Two OUTPUT_EXISTING args for the two Out params
        assert code.count("TensorArgType.OUTPUT_EXISTING") == 2

    def test_tuple_return_builtin_tuple(self):
        """Tuple-return worker (builtin tuple[...]) also produces per-element aliases."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out_s: pl.Out[pl.Tensor[[64], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                return out_s, out_d

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out_s: pl.Out[pl.Tensor[[64], pl.FP32]],
                out_d: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                s, d = self.chip_orch(a, b, out_s, out_d)
                return s, d

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Must produce identical structure as the pl.Tuple variant
        assert code.count('tensors["') >= 2
        assert "submit_next_level" in code
        assert code.count("TensorArgType.OUTPUT_EXISTING") == 2


class TestSubWorkerSourceGeneration:
    """Test _emit_sub_worker_module for correct param names and imports."""

    def test_sub_worker_source_param_names_match_signature(self):
        """_user_* function params come from the IR function params."""
        from pypto.backend.pto_backend import _emit_sub_worker_module  # noqa: PLC0415

        @pl.program
        class P:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[64], pl.FP32]):
                assert f is not None

        verify_fn = P.get_function("verify")
        assert verify_fn is not None
        source = _emit_sub_worker_module(verify_fn)
        param_name = verify_fn.params[0].name_hint
        assert f"def _user_verify({param_name}):" in source
        assert f"{param_name} = _tensor_from_continuous(args.tensor(0))" in source
        assert f"_user_verify({param_name})" in source

    def test_sub_worker_source_imports_torch(self):
        """Generated SubWorker source includes import torch."""
        from pypto.backend.pto_backend import _emit_sub_worker_module  # noqa: PLC0415

        @pl.program
        class P:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def worker(x: pl.Tensor[[64], pl.FP32]):
                pass

        worker_fn = P.get_function("worker")
        assert worker_fn is not None
        source = _emit_sub_worker_module(worker_fn)
        assert "import torch" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
