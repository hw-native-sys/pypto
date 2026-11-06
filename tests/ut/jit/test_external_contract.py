# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Independent providers share a contract without importing any optional DSL."""

from dataclasses import FrozenInstanceError, dataclass, replace
from typing import cast

import pytest
from pypto.backend.external_bridge import RuntimeTarget, validate_target
from pypto.jit.external.contracts import (
    ABIParameter,
    ABIValue,
    Access,
    AliasPermission,
    CompileTarget,
    Completion,
    Core,
    Coverage,
    ExecutionRequirements,
    KernelContract,
    KernelRequest,
    NativeABI,
    ParameterSpec,
    PreparationIdentity,
    PreparedKernel,
    ReturnBinding,
    ScalarSpec,
    SourceArtifact,
    SourceFile,
    TensorEffect,
    TensorSpec,
    Unit,
    WritebackBinding,
)
from pypto.jit.external.identity import preparation_key, prepared_digest
from pypto.jit.external.normalization import plan_call
from pypto.jit.external.protocols import ExternalKernelProvider
from pypto.jit.external.validation import (
    ContractError,
    validate_contract,
    validate_prepared,
    validate_request,
)
from pypto.pypto_core import DataType, ir


@pytest.fixture
def prepared():
    parameters = (
        ParameterSpec("x", TensorSpec((32,), DataType.FP32)),
        ParameterSpec("state", TensorSpec((32,), DataType.FP32)),
        ParameterSpec("out", TensorSpec((32,), DataType.FP32)),
        ParameterSpec("count", ScalarSpec(DataType.INT32)),
    )
    request = KernelRequest(parameters, CompileTarget("a5", "ccec", "test"))
    contract = KernelContract(
        parameters,
        (
            TensorEffect(0, Access.READ),
            TensorEffect(1, Access.READ_WRITE, Coverage.PARTIAL),
            TensorEffect(2, Access.WRITE, Coverage.FULL),
        ),
        public_returns=(ReturnBinding(2),),
        writebacks=(WritebackBinding(2), WritebackBinding(1)),
    )
    # Physical ABI order intentionally differs from logical order.
    abi = NativeABI(
        "fake.descriptor",
        1,
        (
            ABIParameter(3, ABIValue.SCALAR, 4, 4),
            ABIParameter(2, ABIValue.DESCRIPTOR, 32, 8),
            ABIParameter(0, ABIValue.DESCRIPTOR, 32, 8),
            ABIParameter(1, ABIValue.DESCRIPTOR, 32, 8),
        ),
    )
    return PreparedKernel(
        request,
        contract,
        SourceArtifact((SourceFile("entry.cpp", "a" * 64),), "entry.cpp", "kernel_entry"),
        abi,
        ExecutionRequirements((Core.AIV,), 1, Completion.ENTRY_RETURN),
        PreparationIdentity("fake", "1", "1", "b" * 64, True),
    )


def test_public_return_and_hidden_writeback(prepared):
    validate_prepared(prepared)
    plan = plan_call(prepared.contract)
    assert plan.result_parameters == (1, 2)
    assert plan.public_slots == (1,)
    assert plan.writeback_slots == ((1, 0), (2, 1))
    assert plan.parameter_directions == (
        ir.ParamDirection.In,
        ir.ParamDirection.InOut,
        ir.ParamDirection.Out,
        ir.ParamDirection.In,
    )


def test_void_and_repeated_readonly_returns(prepared):
    assert plan_call(replace(prepared.contract, public_returns=())).public_slots == ()
    plan = plan_call(
        replace(prepared.contract, public_returns=(ReturnBinding(0), ReturnBinding(2), ReturnBinding(0)))
    )
    assert plan.result_parameters == (1, 2, 0)
    assert plan.public_slots == (2, 1, 2)
    assert plan.writeback_slots == ((1, 0), (2, 1))


@pytest.mark.parametrize("coverage", [Coverage.PARTIAL, Coverage.UNKNOWN])
def test_write_only_partial_output_keeps_prior_contents(prepared, coverage):
    effects = (*prepared.contract.effects[:2], TensorEffect(2, Access.WRITE, coverage))
    assert (
        plan_call(replace(prepared.contract, effects=effects)).parameter_directions[2]
        == ir.ParamDirection.InOut
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "unsupported schema"),
        ({"required_capabilities": frozenset({"allocate_result"})}, "unsupported required"),
        ({"effects": ()}, "missing tensor effects"),
        ({"writebacks": ()}, "each written tensor"),
        ({"writebacks": (WritebackBinding(1), WritebackBinding(1))}, "each written tensor"),
        ({"public_returns": (ReturnBinding(3),)}, "fresh/scalar results unsupported"),
        ({"public_returns": (ReturnBinding(-1),)}, "not a tensor parameter"),
        ({"alias_permissions": (AliasPermission(0, 0),)}, "distinct ordered"),
        ({"alias_permissions": (AliasPermission(0, 1), AliasPermission(0, 1))}, "duplicate pair"),
    ],
)
def test_invalid_semantics(prepared, change, message):
    with pytest.raises(ContractError, match=message):
        validate_contract(replace(prepared.contract, **change))


@pytest.mark.parametrize(
    ("effect", "message"),
    [
        (TensorEffect(2, Access.WRITE, Coverage.NONE), "write needs coverage"),
        (TensorEffect(2, Access.READ, Coverage.FULL), "non-writing effect"),
        (TensorEffect(2, Access.WRITE, Coverage.PARTIAL, False), "preserve untouched"),
        (TensorEffect(2, cast(Access, "unknown"), Coverage.FULL), "unknown access"),
    ],
)
def test_effects_fail_closed(prepared, effect, message):
    with pytest.raises(ContractError, match=message):
        validate_contract(replace(prepared.contract, effects=(*prepared.contract.effects[:2], effect)))


def test_duplicate_effect_rejected(prepared):
    with pytest.raises(ContractError, match="duplicate tensor"):
        validate_contract(
            replace(prepared.contract, effects=(*prepared.contract.effects, prepared.contract.effects[0]))
        )


def test_request_is_deeply_immutable(prepared):
    with pytest.raises(FrozenInstanceError):
        setattr(prepared.request, "schema_version", 99)
    with pytest.raises(ContractError, match="immutable"):
        validate_request(replace(prepared.request, options=[("opt", 1)]))
    first = replace(prepared.request, constants=(("tile", 16),))
    second = replace(prepared.request, constants=(("tile", 32),))
    validate_request(first)
    validate_request(second)
    assert first != second
    assert prepared.request.constants == ()


def test_mutable_nested_record_is_rejected(prepared):
    @dataclass
    class MutableTarget:
        architecture: str = "a5"
        compiler: str = "ccec"
        compiler_version: str = "test"

    with pytest.raises(ContractError, match="immutable frozen record"):
        validate_request(replace(prepared.request, target=MutableTarget()))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"constants": (("tile", 16), ("tile", 32))}, "duplicate"),
        ({"options": (("limit", float("nan")),)}, "non-finite"),
        ({"schema_version": True}, "unsupported schema"),
    ],
)
def test_invalid_request(prepared, change, message):
    with pytest.raises(ContractError, match=message):
        validate_request(replace(prepared.request, **change))


def test_dynamic_strided_request(prepared):
    parameter = ParameterSpec("view", TensorSpec(("N", 16), DataType.FP32, ("S", 1), "offset"))
    validate_request(replace(prepared.request, parameters=(parameter,)))


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (TensorSpec((32,), DataType.FP32, alignment=3), "power of two"),
        (TensorSpec((-1,), DataType.FP32), "nonnegative"),
        (TensorSpec((32,), DataType.FP32, (32, 1)), "stride rank"),
    ],
)
def test_invalid_tensor_constraints(prepared, spec, message):
    with pytest.raises(ContractError, match=message):
        validate_request(replace(prepared.request, parameters=(ParameterSpec("x", spec),)))


@pytest.mark.parametrize(
    ("physical", "message"),
    [
        (ABIParameter(3, ABIValue.SCALAR, 8, 8), "scalar bit representation"),
        (ABIParameter(7, ABIValue.SCALAR, 4, 4), "invalid logical"),
        (ABIParameter(0, ABIValue.SCALAR, 4, 4), "tensor cannot"),
        (ABIParameter(0, ABIValue.DESCRIPTOR, 32, 3), "power of two"),
        (ABIParameter(0, ABIValue.DIMENSION, 8, 8, axis=1), "invalid tensor axis"),
    ],
)
def test_invalid_abi(prepared, physical, message):
    with pytest.raises(ContractError, match=message):
        validate_prepared(replace(prepared, native_abi=replace(prepared.native_abi, parameters=(physical,))))


def test_abi_metadata_units_and_order_are_retained(prepared):
    parameters = (
        *prepared.native_abi.parameters,
        ABIParameter(0, ABIValue.STRIDE, 8, 8, axis=0, unit=Unit.BYTES),
    )
    modified = replace(prepared, native_abi=replace(prepared.native_abi, parameters=parameters))
    validate_prepared(modified)
    assert modified.native_abi.parameters[0].parameter == 3
    assert modified.native_abi.parameters[-1].unit == Unit.BYTES
    with pytest.raises(ContractError, match="duplicate field"):
        validate_prepared(
            replace(
                modified, native_abi=replace(modified.native_abi, parameters=(*parameters, parameters[0]))
            )
        )


@pytest.mark.parametrize("path", ["/tmp/entry.cpp", "../entry.cpp", "a/../../entry.cpp", "a\\entry.cpp"])
def test_artifact_paths_stay_inside_bundle(prepared, path):
    artifact = SourceArtifact((SourceFile(path, "a" * 64),), path, "entry")
    with pytest.raises(ContractError, match="bundle-relative"):
        validate_prepared(replace(prepared, artifact=artifact))


@pytest.fixture
def target(prepared):
    capabilities = frozenset(
        {
            "core:aiv",
            "aiv_lanes:1",
            "completion:entry_return_visible",
            "tensor_writeback",
            "parameter_returns",
            "partial_write",
        }
    )
    return RuntimeTarget(
        "fake-runtime", "1", prepared.request.target, frozenset({("fake.descriptor", 1)}), capabilities
    )


def test_target_gates_semantics_as_well_as_native_abi(prepared, target):
    validate_target(prepared, target)
    with pytest.raises(ContractError, match="unsupported native ABI"):
        validate_target(prepared, replace(target, supported_abis=frozenset()))
    with pytest.raises(ContractError, match="partial_write"):
        validate_target(prepared, replace(target, capabilities=target.capabilities - {"partial_write"}))
    overlap = replace(
        prepared, contract=replace(prepared.contract, alias_permissions=(AliasPermission(0, 1),))
    )
    with pytest.raises(ContractError, match="aliasing"):
        validate_target(overlap, target)


@pytest.mark.parametrize(
    "execution",
    [
        ExecutionRequirements((Core.AIC,), 1, Completion.ENTRY_RETURN),
        ExecutionRequirements((Core.AIV,), 2, Completion.ENTRY_RETURN),
        ExecutionRequirements((Core.AIV,), 1, Completion.ENTRY_RETURN, full_residency=True),
        ExecutionRequirements(
            (Core.AIC, Core.AIV), 1, Completion.ENTRY_RETURN, full_residency=True, sync_start=True
        ),
    ],
)
def test_execution_requirements_rejected_early(prepared, execution):
    with pytest.raises(ContractError, match="execution:"):
        validate_prepared(replace(prepared, execution=execution))


def test_optional_dsl_is_not_imported(run_without_optional_runtime):
    result = run_without_optional_runtime(
        "import sys; import pypto.jit.external.contracts; import pypto.backend.external_bridge; "
        "assert not any(n.split('.')[0] in {'cannbotdsl', 'tilelang'} for n in sys.modules)"
    )
    assert result.returncode == 0, result.stderr


def test_incomplete_preparation_identity_never_reuses(prepared):
    incomplete = replace(prepared.identity, complete=False)
    assert preparation_key(prepared.request, incomplete) is None
    # Re-preparation can still produce a content-addressable output.
    assert len(prepared_digest(replace(prepared, identity=incomplete))) == 64


def test_preparation_identity_covers_compiler_constants_and_adapter(prepared):
    key = preparation_key(prepared.request, prepared.identity)
    for request in (
        replace(prepared.request, constants=(("tile", 16),)),
        replace(prepared.request, target=replace(prepared.request.target, compiler_version="next")),
        replace(prepared.request, options=(("fast_math", True),)),
    ):
        assert preparation_key(request, prepared.identity) != key
    assert preparation_key(prepared.request, replace(prepared.identity, adapter_version="next")) != key
    first = replace(prepared.request, options=(("a", 1), ("b", 2)))
    second = replace(prepared.request, options=(("b", 2), ("a", 1)))
    assert preparation_key(first, prepared.identity) == preparation_key(second, prepared.identity)


def test_output_identity_covers_abi_effects_and_execution(prepared):
    original = prepared_digest(prepared)
    changes = (
        replace(
            prepared,
            native_abi=replace(
                prepared.native_abi, parameters=tuple(reversed(prepared.native_abi.parameters))
            ),
        ),
        replace(prepared, execution=replace(prepared.execution, logical_launch_context=True)),
        replace(prepared, contract=replace(prepared.contract, public_returns=())),
        replace(prepared, artifact=replace(prepared.artifact, files=(SourceFile("entry.cpp", "c" * 64),))),
    )
    assert all(prepared_digest(change) != original for change in changes)


def test_two_providers_share_normalization_and_bridge_validation(prepared, target):
    class DescriptorProvider:
        def prepare(self, kernel: object, request: KernelRequest) -> PreparedKernel:
            assert kernel == "kernel-ref"
            return replace(prepared, request=request)

    class PointerProvider:
        def prepare(self, kernel: object, request: KernelRequest) -> PreparedKernel:
            assert kernel == "kernel-ref"
            parameters = (
                ABIParameter(0, ABIValue.DATA_POINTER, 8, 8),
                ABIParameter(1, ABIValue.DATA_POINTER, 8, 8),
                ABIParameter(2, ABIValue.DATA_POINTER, 8, 8),
                ABIParameter(3, ABIValue.SCALAR, 4, 4),
            )
            return replace(
                prepared,
                request=request,
                native_abi=NativeABI("fake.pointer", 1, parameters),
                identity=replace(prepared.identity, provider="pointer"),
            )

    target = replace(target, supported_abis=target.supported_abis | {("fake.pointer", 1)})
    providers: tuple[ExternalKernelProvider, ...] = (DescriptorProvider(), PointerProvider())
    results = [provider.prepare("kernel-ref", prepared.request) for provider in providers]
    for result in results:
        validate_target(result, target)
    assert plan_call(results[0].contract) == plan_call(results[1].contract)
    assert prepared_digest(results[0]) != prepared_digest(results[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
