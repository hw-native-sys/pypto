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


def _host_orch_create_code(dtype, init_value):
    """Generate distributed code for a HOST orchestrator that creates one buffer.

    Shared by the ``init_value`` tests below, which differ only in the dtype and
    the fill requested — spelling the three-function hierarchy out once per case
    would bury that difference.
    """

    @pl.program
    class Input:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
        def chip_worker(
            self,
            a: pl.Tensor[[64], dtype],
            buf: pl.Out[pl.Tensor[[64], dtype]],
        ) -> pl.Tensor[[64], dtype]:
            y: pl.Tensor[[64], dtype] = pl.add(a, a)
            return y

        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip_orch(
            self,
            a: pl.Tensor[[64], dtype],
            buf: pl.Out[pl.Tensor[[64], dtype]],
        ) -> pl.Tensor[[64], dtype]:
            result: pl.Tensor[[64], dtype] = self.chip_worker(a, buf)
            return result

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, a: pl.Tensor[[64], dtype]) -> pl.Tensor[[64], dtype]:
            buf: pl.Tensor[[64], dtype] = pl.create_tensor([64], dtype=dtype, init_value=init_value)
            result: pl.Tensor[[64], dtype] = self.chip_orch(a, buf)
            return result

    program = passes.convert_to_ssa()(Input)
    return codegen.DistributedCodegen().generate(program)


class TestDistributedCodegen:
    """Test distributed Python codegen on outlined hierarchy programs."""

    def test_chip_sub_worker_and_orchestrator(self):
        """HOST orchestrator calling CHIP orchestrator → CHIP worker dispatches via _submit_chip."""

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
            def chip_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def moe_ep_l3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_orch(x)
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
        # Plain host tensors use the worker-aware address-free wire helper.
        assert 'make_tensor_arg(orch._worker, tensors["a' in code
        assert 'make_tensor_arg(orch._worker, tensors["b' in code

    def test_create_tensor_init_value_fills_the_buffer(self):
        """A non-zero init_value must reach the HOST-allocated buffer.

        Regression: codegen used to emit ``torch.zeros`` unconditionally, so a
        non-zero ``pl.create_tensor(init_value=...)`` was silently dropped at
        HOST level while the same call was honoured inside a CHIP orchestrator
        (which gets the fill from ``TensorCreateInfo::set_initial_value``). The
        program compiled, built and ran — only the numbers were wrong.
        """
        code = _host_orch_create_code(pl.FP32, init_value=2.5)

        assert "torch.full((64,), 2.5, dtype=torch.float32).share_memory_()" in code
        assert "torch.zeros(" not in code

    def test_create_tensor_init_value_integer_dtype(self):
        """Integer dtypes fill with an integer literal, not a float one."""
        code = _host_orch_create_code(pl.INT32, init_value=7)

        assert "torch.full((64,), 7, dtype=torch.int32).share_memory_()" in code

    def test_create_tensor_init_value_zero_stays_torch_zeros(self):
        """init_value=0 is the allocation default, so nothing changes for it."""
        code = _host_orch_create_code(pl.FP32, init_value=0)

        assert "torch.zeros((64,), dtype=torch.float32).share_memory_()" in code
        assert "torch.full(" not in code

    def test_create_tensor_init_value_rejects_fractional_integer_fill(self):
        """A fractional fill into an integer tensor would truncate silently."""
        with pytest.raises(ValueError, match="whole-number init_value"):
            _host_orch_create_code(pl.INT32, init_value=2.5)

    def test_create_tensor_init_value_rejects_narrow_integer_overflow(self):
        """A fill that does not fit the destination width is a codegen error.

        Exact double representability does not imply the value fits the tensor:
        128 is a fine double and is not an int8. Left to ``torch.full`` the two
        out-of-range directions diverge and both are wrong -- an int8 fill of 128
        raises inside the generated ``_alloc_intermediates`` (a run-time failure
        from a codegen mistake), while a uint8 fill of -1 is accepted and quietly
        becomes 255.
        """
        with pytest.raises(ValueError, match=r"out of range for dtype int8"):
            _host_orch_create_code(pl.INT8, init_value=128)

        with pytest.raises(ValueError, match=r"out of range for dtype uint8"):
            _host_orch_create_code(pl.UINT8, init_value=-1)

    def test_create_tensor_init_value_allows_narrow_integer_extremes(self):
        """The bounds themselves are inclusive -- int8 holds [-128, 127]."""
        assert "torch.full((64,), 127, dtype=torch.int8)" in _host_orch_create_code(pl.INT8, init_value=127)
        assert "torch.full((64,), -128, dtype=torch.int8)" in _host_orch_create_code(pl.INT8, init_value=-128)
        assert "torch.full((64,), 255, dtype=torch.uint8)" in _host_orch_create_code(pl.UINT8, init_value=255)

    def test_create_tensor_init_value_survives_alloc_hoisting(self):
        """The fill must follow the allocation into _alloc_intermediates.

        HOST-orch allocations are hoisted pre-fork so the child processes can
        see the shared mapping; the fill has to be applied there, because the
        hoisted line *is* the allocation.
        """
        code = _host_orch_create_code(pl.FP32, init_value=1.0)

        alloc_idx = code.find("def _alloc_intermediates(tensors, world_size=1):")
        host_idx = code.find("def host_orch(")
        assert alloc_idx >= 0 and alloc_idx < host_idx

        alloc_block = code[alloc_idx:host_idx]
        assert "torch.full((64,), 1, dtype=torch.float32).share_memory_()" in alloc_block

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
