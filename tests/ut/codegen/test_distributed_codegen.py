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
import time

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes


class TestDistributedCodegen:
    """Test distributed Python codegen on outlined hierarchy programs."""

    def test_chip_sub_worker_and_orchestrator(self):
        """HOST orchestrator calling CHIP orchestrator → CHIP worker dispatches via _submit_chip."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_worker(
                self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return pl.tensor.assemble(out, y, [0])

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(x, out)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_orch(x, out)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Verify imports
        assert "from simpler.task_interface import " in code
        assert "TaskArgs" in code and "TensorArgType" in code
        assert "from pypto.runtime.tensor_arg import make_tensor_arg" in code

        # Verify function definition
        assert "def host_orch" in code
        assert "orch, _args, config" in code

        # Verify call-site lowering: CHIP orchestrator → _submit_chip (the
        # comm-less dispatch passes None and _submit_chip resolves the chip;
        # it forwards to submit_next_level when DFX is off).
        assert "_submit_chip" in code
        assert 'callables["chip_orch"]' in code
        assert "TaskArgs()" in code

    def test_renamed_host_orch_marks_entry(self):
        """Host orchestrator under any name gets the runtime entry marker.

        Regression for issue #1678: the runtime resolves the dispatch entry by
        the ``_pypto_distributed_entry`` marker, not by function name, so a
        renamed ``@pl.jit.host`` orchestrator must carry the marker.
        """

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return pl.tensor.assemble(out, y, [0])

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def moe_ep_l3(
                self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_orch(x, out)
                return y

        program = passes.convert_to_ssa()(Input)
        code = codegen.DistributedCodegen().generate(program)

        assert "def moe_ep_l3(" in code
        assert "moe_ep_l3._pypto_distributed_entry = True" in code
        # The marker must follow the function definition it tags.
        assert code.index("def moe_ep_l3(") < code.index("moe_ep_l3._pypto_distributed_entry")

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
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return pl.tensor.assemble(out, y, [0])

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, b, out)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def verify(f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                f: pl.Tensor[[64], pl.FP32] = self.chip_orch(a, b, out)
                self.verify(f)
                return f

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "_submit_chip" in code
        assert "submit_sub" in code
        assert "TensorArgType.INPUT" in code

    def test_for_loop_codegen(self):
        """ForStmt in function body produces Python for loop."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.POD, role=pl.Role.SubWorker)
            def worker(x: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.POD, role=pl.Role.Orchestrator)
            def orch_with_loop(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 4):
                    self.worker(x)
                return x

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "for " in code
        assert "in range(" in code
        assert "submit_sub" in code
        assert 'sub_ids["worker"]' in code

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

        assert "from simpler.task_interface import " in code
        assert "TaskArgs" in code and "TensorArgType" in code
        assert "from pypto.runtime.tensor_arg import make_tensor_arg" in code

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
                return f

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
        # A concrete (`pass`) body is NOT a runtime-bound callback.
        assert verify_fn.requires_runtime_binding is False

    def test_abstract_sub_worker_is_runtime_bound(self):
        """A SubWorker declared with a `...` body is an abstract callback."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def sample(logits: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8], pl.INT32]: ...

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
                self.sample(x)
                return x

        sample_fn = Input.get_function("sample")
        assert sample_fn is not None
        assert sample_fn.requires_runtime_binding is True
        # Abstract body carries no captured source text.
        assert isinstance(sample_fn.body, ir.InlineStmt)
        assert sample_fn.body.body == ""

    def test_abstract_sub_worker_round_trips_as_ellipsis(self):
        """An abstract SubWorker prints as `...` and reparses to the same flag."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def sample(logits: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8], pl.INT32]: ...

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
                self.sample(x)
                return x

        printed = str(Input)
        assert "def sample(" in printed
        assert "..." in printed

        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Input, reparsed)
        sample_fn = reparsed.get_function("sample")
        assert sample_fn is not None
        assert sample_fn.requires_runtime_binding is True

    def test_abstract_sub_worker_emits_guard_stub(self):
        """Codegen emits a raising guard for an abstract SubWorker module."""
        from pypto.backend.pto_backend import _emit_sub_worker_module  # noqa: PLC0415

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def sample(logits: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8], pl.INT32]: ...

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
                self.sample(x)
                return x

        sample_fn = Input.get_function("sample")
        assert sample_fn is not None
        module = _emit_sub_worker_module(sample_fn)
        assert "def sample(args):" in module
        assert "raise RuntimeError" in module
        assert "prepare(callbacks=" in module
        # Generated stub must be syntactically valid Python.
        compile(module, "<sample>", "exec")

    def test_abstract_sub_worker_survives_pto_serialization(self):
        """requires_runtime_binding round-trips through binary .pto serialization."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
            def sample(logits: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8], pl.INT32]: ...

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[8, 16], pl.FP32]) -> pl.Tensor[[8, 16], pl.FP32]:
                self.sample(x)
                return x

        restored = ir.deserialize(ir.serialize(Input))
        assert isinstance(restored, ir.Program)
        sample_fn = restored.get_function("sample")
        assert sample_fn is not None
        assert sample_fn.requires_runtime_binding is True
        ir.assert_structural_equal(Input, restored)

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
                return buf

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
                return f

            @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
            def chip_sub(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return f

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
        # Plain host tensors use the worker-aware address-free wire helper.
        assert re.search(r'add_chip_tensor_arg\([^,]+, orch\._worker, tensors\["a', code)
        assert re.search(r'add_chip_tensor_arg\([^,]+, orch\._worker, tensors\["b', code)

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
                return buf

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

        alloc_idx = code.find("def _alloc_intermediates(tensors, world_size=1):")
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
            def chip_worker(
                self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return pl.tensor.assemble(out, y, [0])

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(
                self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(x, out)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_orch(x, out)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "def _alloc_intermediates(tensors, world_size=1):" in code
        # Body is just `pass` since there are no allocations to hoist.
        alloc_idx = code.find("def _alloc_intermediates(tensors, world_size=1):")
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
        # _submit_chip dispatch emitted for chip_orch (comm-less, chip resolved at dispatch)
        assert "_submit_chip" in code
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
        assert "_submit_chip" in code
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
        assert f"{param_name} = _tensor_from_continuous(args[0])" in source
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


class TestChipTaskCollection:
    """Next-level program extraction must follow Submit callees, not just Call."""

    def test_collect_chip_funcs_includes_submit_callee(self):
        """Regression for issue #1707: a kernel reached only via ``pl.submit``.

        Orchestration submits ``down_seed`` through ``pl.submit`` inside a
        ``pl.manual_scope``. The submitted InCore callee must appear in the
        collected chip program, or distributed codegen drops it and fails with
        ``function 'down_seed' not found after validation``.
        """
        from pypto.backend.pto_backend import _collect_chip_task_functions  # noqa: PLC0415

        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.InCore)
            def down_seed(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def decode_fwd(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, _a_tid = pl.submit(self.down_seed, x)
                return a

        orch = Input.get_function("decode_fwd")
        assert orch is not None
        chip_funcs = _collect_chip_task_functions(orch, Input)
        names = {f.name for f in chip_funcs}
        assert "down_seed" in names, names


def test_chip_tensor_transfer_slots_ignore_scalar_parameters():
    @pl.program
    class Program:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(
            self,
            data: pl.Tensor[[8], pl.INT32],
            n: pl.Scalar[pl.INT32],
            gate: pl.InOut[pl.Tensor[[2], pl.INT32]],
        ):
            value = pl.tensor.read(gate, [0])
            pl.tensor.write(gate, [1], value + n)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def main(
            self,
            data: pl.Tensor[[8], pl.INT32],
            n: pl.Scalar[pl.INT32],
            gate: pl.InOut[pl.Tensor[[2], pl.INT32]],
        ):
            self.chip(data, n, gate)

    code = codegen.DistributedCodegen().generate(passes.convert_to_ssa()(Program))
    assert re.search(r'add_chip_tensor_arg\([^\n]+TensorArgType.INPUT, callables, "chip", 0\)', code)
    assert re.search(r'add_chip_tensor_arg\([^\n]+TensorArgType.INOUT, callables, "chip", 1\)', code)
    assert code.count(".add_scalar(") == 1
    compile(code, "host_orch.py", "exec")


@pytest.mark.parametrize("via_graph", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("submit", [False, True])
@pytest.mark.parametrize(
    "returned", [("second",), ("second", "first"), ("data", "second", "first"), ("second", "second")]
)
def test_worker_return_aliases_follow_declared_results_not_mutated_parameters(
    normalized, submit, returned, via_graph
):
    """A KV-like InOut before the outputs must not occupy a return position.

    Reordered/repeated results and a returned input also distinguish ReturnStmt
    identity from the list of writable parameters. Submit's completion remains
    separate from these tensor results.
    """
    tensor = "pl.Tensor[[64], pl.FP32]"
    params = f"state: pl.InOut[{tensor}], data: {tensor}, first: pl.Out[{tensor}], second: pl.Out[{tensor}]"
    ret_type = tensor if len(returned) == 1 else "tuple[" + ", ".join([tensor] * len(returned)) + "]"
    values = [f"value{i}" for i in range(len(returned))]
    targets = "(" + ", ".join(values) + "), done" if submit else ", ".join(values)
    call = (
        "pl.submit(self.chip, state, data, first, second)"
        if submit
        else "self.chip(state, data, first, second)"
    )
    graph = ""
    chip_body = "return " + ", ".join(returned)
    if via_graph:
        graph = f"""    @pl.function(type=pl.FunctionType.Graph)
    def graph(self, {params}) -> {ret_type}:
        return {", ".join(returned)}

"""
        chip_body = (
            f"{', '.join(values)} = self.graph(state, data, first, second)\n"
            f"        return {', '.join(values)}"
        )
    text = f"""@pl.program
class ReturnBinding:
{graph}    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, {params}) -> {ret_type}:
        {chip_body}

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, {params}) -> {ret_type}:
        {targets} = {call}
        return {", ".join(values)}
"""
    program = passes.convert_to_ssa()(pl.parse_program(text))
    if normalized:
        program = passes.normalize_return_order()(program)
    code = codegen.DistributedCodegen().generate(program)
    aliases = dict(
        re.findall(r'tensors\["(value\d+)(?:__ssa_v\d+)?"\] = tensors\["(\w+?)(?:__ssa_v\d+)?"\]', code)
    )
    assert aliases == dict(zip(values, returned, strict=True)), code
    assert code.count("TensorArgType.INOUT") == 1, code
    if submit:
        assert re.search(r"_handle = _submit_chip", code), code
    compile(code, "host_orch.py", "exec")


@pytest.mark.parametrize("tail", ["none", "scalar", "tensor"])
def test_known_return_parameter_missing_from_raw_host_call_is_rejected(tail):
    import pypto  # noqa: PLC0415

    extra_param = {
        "none": "",
        "scalar": ", n: pl.Scalar[pl.INT32]",
        "tensor": ", other: pl.Tensor[[64], pl.FP32]",
    }[tail]
    extra_arg = {"none": "", "scalar": ", n", "tensor": ", other"}[tail]
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(
        self, state: pl.InOut[pl.Tensor[[64], pl.FP32]],
        out: pl.Out[pl.Tensor[[64], pl.FP32]]{extra_param},
    ) -> pl.Tensor[[64], pl.FP32]:
        return out

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(
        self, state: pl.InOut[pl.Tensor[[64], pl.FP32]],
        out: pl.Out[pl.Tensor[[64], pl.FP32]]{extra_param},
    ) -> pl.Tensor[[64], pl.FP32]:
        result = self.chip(state, out{extra_arg})
        return result
""")

    class OmitReturnedArgument(ir.IRMutator):
        def visit_call(self, op: ir.Call) -> ir.Expr:
            if op.op.name == "chip":
                return ir.Call(op.op, [op.args[0], *op.args[2:]], op.type, op.span)
            return super().visit_call(op)

    malformed = OmitReturnedArgument().visit_program(program)
    # A middle gap cannot be detected just by checking returned_idx < nargs:
    # the next scalar or tensor shifts into that index. HOST requires full arity.
    with pytest.raises(pypto.InternalError, match="requires every callee argument"):
        codegen.DistributedCodegen().generate(malformed)


@pytest.mark.parametrize("fresh_first", [False, True])
def test_host_mixed_known_and_unresolved_returns_require_explicit_carriers(fresh_first):
    returned = "fresh, x" if fresh_first else "x, fresh"
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
    ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        fresh = pl.add(x, x)
        return {returned}

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
    ) -> pl.Tensor[[64], pl.FP32]:
        first, second = self.chip(x, out)
        return first
""")
    with pytest.raises(ValueError, match="mixed known and unresolved tensor returns"):
        codegen.DistributedCodegen().generate(program)


def test_host_returned_readonly_input_does_not_allocate_output_carrier():
    @pl.program
    class Program:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = self.chip(x)
            return result

    source = codegen.DistributedCodegen().generate(Program)
    assert 'tensors["result"] = tensors["x"]' in source
    assert "torch.zeros(" not in source
    compile(source, "<generated>", "exec")


@pytest.mark.parametrize("normalized", [False, True])
def test_raw_typed_callee_does_not_trust_rebound_formal_identity(normalized):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(
            self, first: pl.Out[pl.Tensor[[64], pl.FP32]], second: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> pl.Tensor[[64], pl.FP32]:
            second = first
            return second

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def main(
            self, first: pl.Out[pl.Tensor[[64], pl.FP32]], second: pl.Out[pl.Tensor[[64], pl.FP32]]
        ) -> pl.Tensor[[64], pl.FP32]:
            result = self.chip(first, second)
            return result

    if not normalized:
        with pytest.raises(ValueError, match="cannot bind unresolved explicit tensor returns"):
            codegen.DistributedCodegen().generate(Program)
        return
    program = passes.normalize_return_order()(passes.convert_to_ssa()(Program))
    source = codegen.DistributedCodegen().generate(program)
    assert re.search(r'tensors\["result[^" ]*"\] = tensors\["first[^" ]*"\]', source)
    assert not re.search(r'tensors\["result[^" ]*"\] = tensors\["second[^" ]*"\]', source)
    compile(source, "<generated>", "exec")


@pytest.mark.parametrize(
    "body",
    [
        "if False:\n            return second\n        return first",
        "if flag:\n            return second\n        return first",
        "if flag:\n            return second",
    ],
)
def test_raw_host_callee_requires_a_single_terminal_return(body):
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]], flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        {body}

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]], flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        result = self.chip(first, second, flag)
        return result
""")
    with pytest.raises(ValueError, match="single terminal return"):
        codegen.DistributedCodegen().generate(program)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("callee_type", ["Opaque", "Orchestration"])
@pytest.mark.parametrize(
    "body",
    [
        "if False:\n            return second\n        return first",
        "if flag:\n            return second\n        return first",
        "if flag:\n            return second",
    ],
)
def test_raw_host_callee_validates_recursive_graph_return_control_flow(body, callee_type, normalized):
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Graph)
    def graph(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
              second: pl.Out[pl.Tensor[[64], pl.FP32]], flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        {body}

    @pl.function(type=pl.FunctionType.{callee_type}, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]], flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        result = self.graph(first, second, flag)
        return result

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]], flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        result = self.chip(first, second, flag)
        return result
""")
    if normalized:
        # Canonicalization must not erase the helper call before HOST validation.
        program = passes.normalize_return_order()(program)
    with pytest.raises(ValueError, match="single terminal return"):
        codegen.DistributedCodegen().generate(program)


@pytest.mark.parametrize("scope", ["manual_scope", "scope"])
@pytest.mark.parametrize("normalized", [False, True])
def test_host_callee_accepts_unconditional_scoped_terminal_return(scope, normalized):
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def chip(self, out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.{scope}():
            return out

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        result = self.chip(out)
        return result
""")
    if normalized:
        program = passes.normalize_return_order()(passes.convert_to_ssa()(program))
    source = codegen.DistributedCodegen().generate(program)
    assert re.search(r'tensors\["result[^" ]*"\] = tensors\["out[^" ]*"\]', source)
    compile(source, "<generated>", "exec")


def test_host_lineage_cache_is_released_after_generate_failure():
    """An exceptional strict query must not affect later non-strict analysis."""
    program = pl.parse_program("""@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Graph)
    def graph(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
              second: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        if False:
            return second
        return first

    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        result = self.graph(first, second)
        return result

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        result = self.chip(first, second)
        return result
""")
    generator = codegen.DistributedCodegen()
    for _ in range(2):
        with pytest.raises(ValueError, match="single terminal return"):
            generator.generate(program)
        # Outside HOST codegen the helper's ambiguous map remains unknown;
        # a leaked strict policy would instead reject this normalization query.
        normalized = passes.normalize_return_order()(program)
        assert normalized is not None


def test_host_callees_reuse_shared_helper_lineage_within_generate():
    """A linear-sized shared helper must not be retraced for every CHIP entry."""
    timings = []
    for size in (128, 2048):
        helper_body = "\n".join(
            f"        value{i} = {'x' if i == 0 else f'value{i - 1}'}" for i in range(size)
        )
        entries = "\n".join(
            f"""    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip{i}(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        result = self.graph(x)
        return result
"""
            for i in range(size)
        )
        calls = "\n".join(f"        result{i} = self.chip{i}(x)" for i in range(size))
        program = pl.parse_program(f"""@pl.program
class SharedHelper:
    @pl.function(type=pl.FunctionType.Graph)
    def graph(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
{helper_body}
        return value{size - 1}

{entries}
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
{calls}
        return result{size - 1}
""")
        samples = []
        generator = codegen.DistributedCodegen()
        for _ in range(3):
            # Parsing is outside the measurement. Thread CPU time excludes
            # scheduling delays; repeating Generate also exercises cache cleanup.
            started = time.thread_time()
            source = generator.generate(program)
            samples.append(time.thread_time() - started)
            aliases = re.findall(r'tensors\["result\d+"\] = tensors\["x"\]', source)
            assert len(aliases) == size
            compile(source, "host_orch.py", "exec")
        timings.append(min(samples))
    # 16x more IR allows generous linear/logarithmic growth and fixed overhead,
    # but rejects the old per-callee retracing of the shared linear alias chain.
    assert timings[1] < 64 * timings[0] + 0.005, timings


def test_host_rejects_unresolved_explicit_tensor_return():
    source = """
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, state: pl.InOut[pl.Tensor[[64], pl.FP32]],
             first: pl.Out[pl.Tensor[[64], pl.FP32]], second: pl.Out[pl.Tensor[[64], pl.FP32]],
             flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        if flag:
            selected = first
        else:
            selected = second
        return selected

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, state: pl.InOut[pl.Tensor[[64], pl.FP32]],
             first: pl.Out[pl.Tensor[[64], pl.FP32]], second: pl.Out[pl.Tensor[[64], pl.FP32]],
             flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        result = self.chip(state, first, second, flag)
        return result
"""
    program = pl.parse_program(source)
    with pytest.raises(ValueError, match="cannot bind unresolved explicit tensor returns"):
        codegen.DistributedCodegen().generate(passes.convert_to_ssa()(program))


@pytest.mark.parametrize("branch_source", [None, "second", "first"])
def test_host_binds_elements_of_explicit_tuple_value(branch_source):
    source = """
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, state: pl.InOut[pl.Tensor[[64], pl.FP32]],
             first: pl.Out[pl.Tensor[[64], pl.FP32]], second: pl.Out[pl.Tensor[[64], pl.FP32]],
             flag: pl.Scalar[pl.BOOL]
    ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        pair = (second, first)
        return pair

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, state: pl.InOut[pl.Tensor[[64], pl.FP32]],
             first: pl.Out[pl.Tensor[[64], pl.FP32]], second: pl.Out[pl.Tensor[[64], pl.FP32]],
             flag: pl.Scalar[pl.BOOL]
    ) -> pl.Tensor[[64], pl.FP32]:
        left, right = self.chip(state, first, second, flag)
        return left
"""
    if branch_source is not None:
        source = source.replace(
            "        pair = (second, first)",
            f"""        if flag:
            selected = second
        else:
            selected = {branch_source}
        pair = (selected, first)""",
        )
    program = pl.parse_program(source)
    if branch_source is not None:
        program = passes.convert_to_ssa()(program)
    if branch_source == "first":
        with pytest.raises(ValueError, match="mixed known and unresolved tensor returns"):
            codegen.DistributedCodegen().generate(program)
        return
    emitted = codegen.DistributedCodegen().generate(program)
    assert re.search(r'tensors\["left(?:__ssa_v\d+)?"\] = tensors\["second(?:__ssa_v\d+)?"\]', emitted)
    assert re.search(r'tensors\["right(?:__ssa_v\d+)?"\] = tensors\["first(?:__ssa_v\d+)?"\]', emitted)


def test_raw_host_binds_directly_returned_call():
    program = pl.parse_program("""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Graph)
    def graph(self, out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        return out

    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, state: pl.InOut[pl.Tensor[[64], pl.FP32]],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        return self.graph(out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, state: pl.InOut[pl.Tensor[[64], pl.FP32]],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        result = self.chip(state, out)
        return result
""")
    source = codegen.DistributedCodegen().generate(program)
    assert 'tensors["result"] = tensors["out"]' in source


def test_host_rejects_unresolved_readonly_return_instead_of_allocating_zeros():
    program = pl.parse_program("""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def pick(self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32],
             flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        if flag:
            selected = a
        else:
            selected = b
        return selected

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32],
             flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        result = self.pick(a, b, flag)
        return result
""")
    with pytest.raises(ValueError, match="cannot bind unresolved explicit tensor returns"):
        codegen.DistributedCodegen().generate(passes.convert_to_ssa()(program))


@pytest.mark.parametrize("tuple_result", [False, True])
def test_host_binds_inferred_call_returns(tuple_result):
    returned = "b, a" if tuple_result else "b"
    targets = "result, other" if tuple_result else "result"
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def pick(self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]):
        return {returned}

    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        {targets} = self.pick(a, b)
        final = self.consume(result)
        return final
""")
    source = codegen.DistributedCodegen().generate(program)
    assert 'tensors["result"] = tensors["b"]' in source
    if tuple_result:
        assert 'tensors["other"] = tensors["a"]' in source
    assert "torch.zeros(" not in source
    compile(source, "host_orch.py", "exec")


@pytest.mark.parametrize("annotated", [True, False])
@pytest.mark.parametrize("tuple_annotation", ["pl.Tuple", "tuple"])
@pytest.mark.parametrize("direct", [False, True])
def test_host_binds_tuple_forwarded_from_graph(annotated, tuple_annotation, direct):
    graph_result = " -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]" if annotated else ""
    forward = (
        "return self.graph(first, second)"
        if direct
        else "pair = self.graph(first, second)\n        return pair"
    )
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Graph)
    def graph(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
              second: pl.Out[pl.Tensor[[64], pl.FP32]]
    ){graph_result}:
        result = (second, first)
        return result

    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]]
    ) -> {tuple_annotation}[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        {forward}

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        left, right = self.chip(first, second)
        return left
""")
    source = codegen.DistributedCodegen().generate(program)
    assert 'tensors["left"] = tensors["second"]' in source
    assert 'tensors["right"] = tensors["first"]' in source


@pytest.mark.parametrize("other", ["second", "first"])
@pytest.mark.parametrize("tuple_annotation", ["pl.Tuple", "tuple"])
def test_host_binds_elements_of_conditional_tuple(other, tuple_annotation):
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]], flag: pl.Scalar[pl.BOOL]
    ) -> {tuple_annotation}[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        if flag:
            pair = (second, first)
        else:
            pair = ({other}, first)
        return pair

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]], flag: pl.Scalar[pl.BOOL]
    ) -> pl.Tensor[[64], pl.FP32]:
        left, right = self.chip(first, second, flag)
        return left
""")
    program = passes.convert_to_ssa()(program)
    if other == "first":
        with pytest.raises(ValueError, match="mixed known and unresolved tensor returns"):
            codegen.DistributedCodegen().generate(program)
        return
    emitted = codegen.DistributedCodegen().generate(program)
    assert re.search(r'tensors\["left(?:__ssa_v\d+)?"\] = tensors\["second(?:__ssa_v\d+)?"\]', emitted)
    assert re.search(r'tensors\["right(?:__ssa_v\d+)?"\] = tensors\["first(?:__ssa_v\d+)?"\]', emitted)


@pytest.mark.parametrize("tensor_position", [0, 1])
@pytest.mark.parametrize("consume_scalar", [False, True])
def test_host_preserves_tensor_position_in_mixed_return(tensor_position, consume_scalar):
    tensor = "pl.Tensor[[64], pl.FP32]"
    scalar = "pl.Scalar[pl.BOOL]"
    types = f"{tensor}, {scalar}" if tensor_position == 0 else f"{scalar}, {tensor}"
    values = "out, flag" if tensor_position == 0 else "flag, out"
    scalar_use = f"selected_flag = pair[{1 - tensor_position}]" if consume_scalar else "selected_flag = flag"
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, state: pl.InOut[{tensor}], out: pl.Out[{tensor}], flag: {scalar}) -> tuple[{types}]:
        return {values}

    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def consume(self, x: {tensor}, condition: {scalar}) -> {tensor}:
        return x

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, state: pl.InOut[{tensor}], out: pl.Out[{tensor}], flag: {scalar}) -> {tensor}:
        pair = self.chip(state, out, flag)
        result = pair[{tensor_position}]
        {scalar_use}
        final = self.consume(result, selected_flag)
        return final
""")
    if consume_scalar:
        with pytest.raises(ValueError, match="cannot consume a scalar"):
            codegen.DistributedCodegen().generate(program)
        return
    emitted = codegen.DistributedCodegen().generate(program)
    assert 'tensors["result"] = tensors["out"]' in emitted
    assert 'tensors["result"] = tensors["state"]' not in emitted


@pytest.mark.parametrize("change", [False, True])
def test_host_binds_invariant_tuple_carry(change):
    yielded = "(first, first)" if change else "carry"
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]]
    ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        pair = (second, first)
        for i, (carry,) in pl.range(3, init_values=(pair,)):
            value = {yielded}
            result = pl.yield_(value)
        return result

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main(self, first: pl.Out[pl.Tensor[[64], pl.FP32]],
             second: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        left, right = self.chip(first, second)
        return left
""")
    program = passes.convert_to_ssa()(program)
    if change:
        with pytest.raises(ValueError, match="mixed known and unresolved tensor returns"):
            codegen.DistributedCodegen().generate(program)
        return
    emitted = codegen.DistributedCodegen().generate(program)
    assert re.search(r'tensors\["left(?:__ssa_v\d+)?"\] = tensors\["second(?:__ssa_v\d+)?"\]', emitted)
    assert re.search(r'tensors\["right(?:__ssa_v\d+)?"\] = tensors\["first(?:__ssa_v\d+)?"\]', emitted)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
