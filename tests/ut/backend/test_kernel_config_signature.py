# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for per-kernel ``signature`` emission in ``kernel_config.py``.

Covers the issue #1458 fix: codegen now emits each kernel's runtime
``ArgDirection`` signature so the tensormap_and_ringbuffer tensor dump builds a
non-empty CoreCallable signature and its per-subtask tensor-arg count matches
the task payload ``tensor_count``. Without this, ``--dump-args`` captured
nothing for a codegen matmul (signature was empty -> count 0 != payload 3).

These tests exercise the codegen-side helper ``_generate_config_file`` directly
and do not require the optional ``simpler`` runtime package. The generated text
is checked for syntactic validity with ``compile`` (which does not execute the
``simpler.task_interface`` import).
"""

from types import SimpleNamespace
from typing import cast

import pytest
from pypto.backend.pto_backend import _generate_config_file
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import passes
from pypto.pypto_core.codegen import KernelRegistration


def _registration(**fields) -> KernelRegistration:
    # Detached metadata fixture; native result records are read-only.
    return cast(KernelRegistration, SimpleNamespace(**fields))


def _base_inputs(signature=None) -> dict:
    return {
        "orch_func_name": "main",
        "kernel_registrations": [
            _registration(
                func_id=0, code_name="matmul_aic", core_type=_ir_core.CoreType.CUBE, signature=signature or []
            )
        ],
    }


def _is_valid_python(text: str) -> bool:
    compile(text, "kernel_config.py", "exec")
    return True


class TestKernelConfigSignature:
    def test_signature_emitted_for_matmul(self) -> None:
        # The matmul AIC kernel takes a, b (inputs) and c (write target).
        text = _generate_config_file(
            **_base_inputs(["IN", "IN", "INOUT"]),
        )
        # ArgDirection import is present and the kernel carries the signature.
        assert "from simpler.task_interface import ArgDirection as _D" in text
        assert '"aicpu_thread_num": 0' in text
        assert '"signature": [_D.IN, _D.IN, _D.INOUT]' in text
        # 3 non-SCALAR entries == payload tensor_count for the matmul (a, b, c).
        assert text.count("_D.") == 3
        assert _is_valid_python(text)

    def test_no_import_without_signatures(self) -> None:
        # Omitting signatures keeps the pre-fix behavior: no import, no key.
        text = _generate_config_file(**_base_inputs())
        assert "ArgDirection" not in text
        assert '"signature"' not in text
        assert _is_valid_python(text)

    def test_empty_signature_map_is_noop(self) -> None:
        text = _generate_config_file(**_base_inputs([]))
        assert "ArgDirection" not in text
        assert '"signature"' not in text
        assert _is_valid_python(text)

    def test_signature_emitted_verbatim_tensor_only(self) -> None:
        # Codegen records only tensor-arg directions (scalars are excluded, as
        # the CoreCallable signature is a per-tensor-arg list); the emitter
        # writes the directions verbatim, preserving tensors-first order.
        text = _generate_config_file(
            **_base_inputs(["IN", "IN", "INOUT", "OUT"]),
        )
        assert '"signature": [_D.IN, _D.IN, _D.INOUT, _D.OUT]' in text
        # No SCALAR members are emitted for codegen-produced signatures.
        assert "_D.SCALAR" not in text
        assert _is_valid_python(text)

    def test_partial_signatures_across_kernels(self) -> None:
        # One kernel has a signature, another does not: import is still emitted,
        # only the kernel with directions gets a "signature" field.
        text = _generate_config_file(
            orch_func_name="main",
            kernel_registrations=[
                _registration(
                    func_id=0, code_name="k_with", core_type=_ir_core.CoreType.VECTOR, signature=["IN", "OUT"]
                ),
                _registration(
                    func_id=1, code_name="k_without", core_type=_ir_core.CoreType.VECTOR, signature=[]
                ),
            ],
        )
        assert "from simpler.task_interface import ArgDirection as _D" in text
        assert text.count('"signature"') == 1
        assert '"signature": [_D.IN, _D.OUT]' in text
        assert _is_valid_python(text)


class TestOrchestrationConfigSignature:
    """Regression tests for the ``ORCHESTRATION`` ``signature`` emission.

    Without this, the ChipCallable signature is empty (``sig_count == 0``) and
    ``bind_callable_to_runtime_impl`` cannot tell which orch tensors are
    read-only inputs, so every tensor is conservatively D2H-copied-back and the
    pure-OUT on-device memset fast path is disabled. The signature is per-tensor
    in ``orch_args`` tensor order (scalars excluded), matching the runtime's
    ``orch_args.tensor(i)`` index.
    """

    def test_orchestration_signature_emitted(self) -> None:
        text = _generate_config_file(
            **_base_inputs(),
            orchestration_signature=["IN", "IN", "OUT"],
        )
        # ArgDirection import is present and the ORCHESTRATION dict carries it.
        assert "from simpler.task_interface import ArgDirection as _D" in text
        assert '"signature": [_D.IN, _D.IN, _D.OUT]' in text
        # The signature sits inside the ORCHESTRATION dict, not KERNELS.
        orch_block = text.split("KERNELS")[0]
        assert '"signature": [_D.IN, _D.IN, _D.OUT]' in orch_block
        assert _is_valid_python(text)

    def test_no_orchestration_signature_is_noop(self) -> None:
        # Omitting the orch signature keeps the pre-fix behavior: no key, and
        # (with no kernel signatures either) no ArgDirection import.
        text = _generate_config_file(**_base_inputs(), orchestration_signature=[])
        assert "ArgDirection" not in text
        assert '"signature"' not in text
        assert _is_valid_python(text)

    def test_orchestration_and_kernel_signatures_coexist(self) -> None:
        text = _generate_config_file(
            **_base_inputs(["IN", "IN", "INOUT"]),
            orchestration_signature=["IN", "OUT"],
        )
        # Both the ORCHESTRATION and the KERNELS entry carry a signature.
        orch_block, kernels_block = text.split("KERNELS", 1)
        assert '"signature": [_D.IN, _D.OUT]' in orch_block
        assert '"signature": [_D.IN, _D.IN, _D.INOUT]' in kernels_block
        assert _is_valid_python(text)

    def test_orchestration_signature_inout(self) -> None:
        text = _generate_config_file(
            **_base_inputs(),
            orchestration_signature=["INOUT", "IN", "OUT"],
        )
        assert '"signature": [_D.INOUT, _D.IN, _D.OUT]' in text
        assert "_D.SCALAR" not in text
        assert _is_valid_python(text)


class TestKernelConfigRuntime:
    """``RUNTIME_CONFIG["runtime"]`` follows the selected runtime ABI."""

    def test_default_runtime_is_tensormap_and_ringbuffer(self) -> None:
        text = _generate_config_file(**_base_inputs())
        assert '"runtime": "tensormap_and_ringbuffer"' in text
        assert "# Runtime configuration for tensormap_and_ringbuffer." in text
        assert _is_valid_python(text)

    def test_host_build_graph_runtime_is_emitted(self) -> None:
        text = _generate_config_file(**_base_inputs(), runtime=passes.RuntimeKind.HOST_BUILD_GRAPH)
        # Both the value the runtime reads and the human-facing header comment
        # must follow, or the file documents a runtime it does not select.
        assert '"runtime": "host_build_graph"' in text
        assert "# Runtime configuration for host_build_graph." in text
        assert "tensormap_and_ringbuffer" not in text
        assert _is_valid_python(text)

    def test_runtime_does_not_disturb_other_keys(self) -> None:
        text = _generate_config_file(
            **_base_inputs(),
            orchestration_signature=["IN", "OUT"],
            runtime=passes.RuntimeKind.HOST_BUILD_GRAPH,
        )
        assert '"aicpu_thread_num": 0' in text
        assert '"signature": [_D.IN, _D.OUT]' in text
        assert _is_valid_python(text)


class TestRuntimeWireNames:
    """The enum is the compile-time contract; the wire name is the ABI boundary."""

    @pytest.mark.parametrize(
        ("kind", "name"),
        [
            (passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, "tensormap_and_ringbuffer"),
            (passes.RuntimeKind.HOST_BUILD_GRAPH, "host_build_graph"),
        ],
    )
    def test_wire_name_round_trips(self, kind, name) -> None:
        # The name is used verbatim as a filesystem path component when
        # compiling kernels, so it is spelled out rather than derived from the
        # enumerator; both directions must agree on that spelling.
        assert passes.runtime_kind_to_name(kind) == name
        assert passes.runtime_kind_from_name(name) == kind

    def test_unknown_wire_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown runtime 'ringbuffer'"):
            passes.runtime_kind_from_name("ringbuffer")


def test_multiple_registrations_reference_one_source():
    registrations = [
        _registration(func_id=i, code_name="shared", core_type=_ir_core.CoreType.VECTOR, signature=signature)
        for i, signature in enumerate([["IN"], ["INOUT"]])
    ]
    text = _generate_config_file("main", registrations)
    assert text.count('"code_name": "shared"') == 2
    assert text.count('"shared.cpp"') == 2
    assert '"func_id": 0' in text and '"func_id": 1' in text
    assert "[_D.IN]" in text and "[_D.INOUT]" in text
    builtin_text = _generate_config_file("main", registrations, unshared_code_names={"shared"})
    assert '"code_name"' not in builtin_text
    assert _is_valid_python(text) and _is_valid_python(builtin_text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
