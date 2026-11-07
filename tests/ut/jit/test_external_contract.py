# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Independent providers share a contract without importing any optional DSL."""

import hashlib
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest
from pypto.backend.external_bridge import ExternalTask, RuntimeBridge, RuntimeTarget, validate_target
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


def _source_file(path: str, content: bytes = b"// kernel source\n") -> SourceFile:
    return SourceFile(path, hashlib.sha256(content).hexdigest(), content)


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
        SourceArtifact((_source_file("entry.cpp"),), "entry.cpp", "kernel_entry"),
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
        (TensorEffect(2, cast(Access, "unknown"), Coverage.FULL), "expected immutable .*Access"),
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


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("request", "parameters"),
        ("request", "constants"),
        ("request", "options"),
        ("contract", "parameters"),
        ("contract", "effects"),
        ("contract", "public_returns"),
        ("contract", "writebacks"),
        ("contract", "alias_permissions"),
        ("native_abi", "parameters"),
        ("execution", "cores"),
        ("artifact", "files"),
        ("artifact", "include_dirs"),
    ],
)
def test_tuple_fields_reject_unordered_collections(prepared, section, field):
    record = getattr(prepared, section)
    changed = replace(prepared, **{section: replace(record, **{field: frozenset()})})
    with pytest.raises(ContractError, match=rf"{section}\.{field}: expected immutable tuple"):
        validate_prepared(changed)


def test_native_abi_set_cannot_have_a_validated_digest(prepared):
    # A set of otherwise valid arguments must not let one digest describe
    # multiple physical slot orders across processes.
    abi = replace(prepared.native_abi, parameters=frozenset(prepared.native_abi.parameters))
    with pytest.raises(ContractError, match="native_abi.parameters: expected immutable tuple"):
        prepared_digest(replace(prepared, native_abi=abi))


@pytest.mark.parametrize("field", ["shape", "strides"])
def test_tensor_dimensions_require_ordered_tuples(prepared, field):
    parameter = prepared.request.parameters[0]
    spec = replace(parameter.type, **{field: frozenset({32})})
    request = replace(prepared.request, parameters=(replace(parameter, type=spec),))
    with pytest.raises(ContractError, match=rf"{field}: expected immutable tuple"):
        validate_request(request)


@pytest.mark.parametrize("field", ["constants", "options"])
@pytest.mark.parametrize("entry", [frozenset({"tile", "32"}), "ab", ("tile",)])
def test_named_compile_inputs_require_ordered_pairs(prepared, field, entry):
    request = replace(prepared.request, **{field: (entry,)})
    with pytest.raises(ContractError, match="expected immutable tuple|invalid tuple length"):
        validate_request(request)


def test_capabilities_remain_an_unordered_set(prepared):
    validate_contract(replace(prepared.contract, required_capabilities=frozenset({"tensor_writeback"})))


@pytest.mark.parametrize(
    "capabilities", [(), ("tensor_writeback",), frozenset({b"aliasing"}), frozenset({1})]
)
def test_capability_schema_errors_are_contract_errors(prepared, capabilities):
    with pytest.raises(ContractError, match="required_capabilities.*expected immutable"):
        validate_contract(replace(prepared.contract, required_capabilities=capabilities))


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("artifact", "symbol"),
        ("artifact", "entry"),
        ("native_abi", "name"),
        ("identity", "provider"),
        ("identity", "kernel_digest"),
    ],
)
def test_source_bytes_cannot_leak_into_metadata(prepared, section, field):
    record = getattr(prepared, section)
    changed = replace(prepared, **{section: replace(record, **{field: b"entry"})})
    for validate in (validate_prepared, prepared_digest):
        with pytest.raises(ContractError, match=rf"{section}\.{field}: expected immutable .*str"):
            validate(changed)


def test_source_bytes_are_valid_only_in_the_payload(prepared):
    # Binary content is legal and hashed through SourceFile.sha256, never JSON.
    file = _source_file("entry.cpp", b"\x00\xff\xfe")
    result = replace(prepared, artifact=replace(prepared.artifact, files=(file,)))
    assert len(prepared_digest(result)) == 64
    with pytest.raises(ContractError, match="constants.*expected immutable"):
        validate_request(replace(prepared.request, constants=(("bad", b"bytes"),)))


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
        ({"schema_version": True}, "schema_version: expected immutable .*int"),
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


def _pointer_view(
    prepared: PreparedKernel, spec: TensorSpec, metadata: tuple[ABIParameter, ...] = ()
) -> PreparedKernel:
    parameters = (replace(prepared.request.parameters[0], type=spec), *prepared.request.parameters[1:])
    abi_parameters = tuple(
        ABIParameter(0, ABIValue.DATA_POINTER, 8, 8) if field.parameter == 0 else field
        for field in prepared.native_abi.parameters
    )
    return replace(
        prepared,
        request=replace(prepared.request, parameters=parameters),
        contract=replace(prepared.contract, parameters=parameters),
        native_abi=replace(prepared.native_abi, parameters=(*abi_parameters, *metadata)),
    )


@pytest.mark.parametrize(
    ("spec", "field"),
    [
        (TensorSpec((32,), DataType.FP32, (1,), None), ABIParameter(0, ABIValue.OFFSET, 8, 8)),
        (TensorSpec((32,), DataType.FP32, (1,), "offset"), ABIParameter(0, ABIValue.OFFSET, 8, 8)),
        (TensorSpec(("N",), DataType.FP32, (1,), 0), ABIParameter(0, ABIValue.DIMENSION, 8, 8, axis=0)),
        (TensorSpec((32,), DataType.FP32, (), 0), ABIParameter(0, ABIValue.STRIDE, 8, 8, axis=0)),
        (TensorSpec((32,), DataType.FP32, ("S",), 0), ABIParameter(0, ABIValue.STRIDE, 8, 8, axis=0)),
    ],
)
def test_pointer_abi_requires_runtime_view_metadata(prepared, spec, field):
    missing = _pointer_view(prepared, spec)
    with pytest.raises(ContractError, match=rf"runtime-bound {field.value.name}.*requires a physical field"):
        validate_prepared(missing)
    complete = _pointer_view(prepared, spec, (field,))
    validate_prepared(complete)
    assert len(prepared_digest(complete)) == 64


@pytest.mark.parametrize("offset", [0, 8])
def test_pointer_abi_can_specialize_static_view_metadata(prepared, offset):
    validate_prepared(_pointer_view(prepared, TensorSpec((32,), DataType.FP32, (2,), offset)))


def test_unused_pointer_and_descriptor_can_omit_explicit_metadata(prepared):
    pointer = _pointer_view(prepared, TensorSpec(("N",), DataType.FP32))
    contract = replace(
        pointer.contract, effects=(TensorEffect(0, Access.NONE), *pointer.contract.effects[1:])
    )
    validate_prepared(replace(pointer, contract=contract))
    # A named descriptor ABI carries the view in its descriptor instead.
    validate_prepared(replace(pointer, native_abi=prepared.native_abi))


@pytest.mark.parametrize("path", ["/tmp/entry.cpp", "../entry.cpp", "a/../../entry.cpp", "a\\entry.cpp"])
def test_artifact_paths_stay_inside_bundle(prepared, path):
    artifact = SourceArtifact((_source_file(path),), path, "entry")
    with pytest.raises(ContractError, match="bundle-relative"):
        validate_prepared(replace(prepared, artifact=artifact))


@pytest.mark.parametrize("path", ["artifact_manifest.json", "artifact_manifest.json/entry.cpp"])
def test_artifact_manifest_path_is_reserved(prepared, path):
    artifact = SourceArtifact((_source_file(path),), path, "entry")
    with pytest.raises(ContractError, match="reserved manifest path"):
        validate_prepared(replace(prepared, artifact=artifact))
    with pytest.raises(ContractError, match="reserved manifest path"):
        validate_prepared(replace(prepared, artifact=replace(prepared.artifact, include_dirs=(path,))))


def test_nested_manifest_name_does_not_conflict_with_store(prepared):
    artifact = SourceArtifact(
        (_source_file("include/artifact_manifest.json"),),
        "include/artifact_manifest.json",
        "entry",
        include_dirs=(".", "include"),
    )
    validate_prepared(replace(prepared, artifact=artifact))


@pytest.mark.parametrize("child", ["include/header.h", "include/nested/header.h"])
@pytest.mark.parametrize("reverse", [False, True])
def test_artifact_files_cannot_shadow_directories(prepared, child, reverse):
    # Even when the descendant is enumerated first, its parent must be a directory.
    paths = (child, "include") if reverse else ("include", child)
    artifact = SourceArtifact(tuple(_source_file(path) for path in paths), paths[0], "entry")
    with pytest.raises(ContractError, match="parent of .* conflicts with bundled file 'include'"):
        validate_prepared(replace(prepared, artifact=artifact))


@pytest.mark.parametrize(
    "paths", [("include", "include_extra/header.h"), ("include/first.h", "include/second.h")]
)
def test_artifact_files_can_share_prefixes_and_directories(prepared, paths):
    artifact = SourceArtifact(tuple(_source_file(path) for path in paths), paths[0], "entry")
    validate_prepared(replace(prepared, artifact=artifact))


@pytest.mark.parametrize("directory", ["include", "include/nested", "include/nested/deeper"])
def test_include_directory_cannot_equal_or_descend_from_a_file(prepared, directory):
    artifact = SourceArtifact((_source_file("include"),), "include", "entry", (directory,))
    with pytest.raises(ContractError, match="include directory .* conflicts with bundled file 'include'"):
        validate_prepared(replace(prepared, artifact=artifact))


def test_include_directories_can_contain_headers_and_share_name_prefixes(prepared):
    artifact = SourceArtifact(
        (_source_file("include/header.h"), _source_file("include_extra")),
        "include/header.h",
        "entry",
        (".", "include"),
    )
    validate_prepared(replace(prepared, artifact=artifact))


@pytest.mark.parametrize("content", [b"tampered", bytearray(b"mutable"), "not bytes"])
def test_source_payload_must_be_immutable_and_match_digest(prepared, content):
    file = replace(prepared.artifact.files[0], content=content)
    changed = replace(prepared, artifact=replace(prepared.artifact, files=(file,)))
    with pytest.raises(ContractError, match="immutable|SHA-256"):
        validate_prepared(changed)
    with pytest.raises(ContractError, match="immutable|SHA-256"):
        prepared_digest(changed)


def test_bridge_can_materialize_sources_after_provider_directory_is_removed(prepared, target, tmp_path):
    payloads = {
        "entry.cpp": b'#include "support.h"\nvoid kernel_entry() {}\n',
        "include/support.h": b"// provider header\n",
    }
    provider_directory: Path | None = None

    class TemporaryProvider:
        def prepare(self, kernel: object, request: KernelRequest) -> PreparedKernel:
            nonlocal provider_directory
            with TemporaryDirectory(dir=tmp_path) as directory:
                provider_directory = Path(directory)
                files = []
                for name, content in payloads.items():
                    path = provider_directory / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                    files.append(_source_file(name, path.read_bytes()))
                return replace(
                    prepared,
                    request=request,
                    artifact=SourceArtifact(tuple(files), "entry.cpp", "kernel_entry", ("include",)),
                )

    class MaterializingBridge:
        def lower(self, prepared: PreparedKernel, target: RuntimeTarget) -> ExternalTask:
            validate_target(prepared, target)
            for file in prepared.artifact.files:
                path = tmp_path / "bridge" / file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(file.content)
            return ExternalTask(prepared.artifact, prepared, target, "1")

    provider: ExternalKernelProvider = TemporaryProvider()
    bridge: RuntimeBridge = MaterializingBridge()
    result = provider.prepare("kernel-ref", prepared.request)
    # The bridge receives only the prepared result and target, after cleanup.
    assert provider_directory is not None and not provider_directory.exists()
    task = bridge.lower(result, target)
    assert task.source.include_dirs == ("include",)
    for file in task.source.files:
        assert (tmp_path / "bridge" / file.path).read_bytes() == payloads[file.path]
    # File enumeration order does not change content identity.
    reordered = replace(
        result, artifact=replace(result.artifact, files=tuple(reversed(result.artifact.files)))
    )
    assert prepared_digest(reordered) == prepared_digest(result)


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
    "supported_abis",
    [
        (("fake.descriptor", 1),),
        frozenset({("fake.descriptor", True)}),
        frozenset({("fake.descriptor", 1.0)}),
        frozenset({(b"fake.descriptor", 1)}),
        frozenset({"fake.descriptor"}),
        frozenset({("fake.descriptor",)}),
        frozenset({("", 1)}),
        frozenset({("fake.descriptor", 0)}),
        frozenset({("fake.descriptor", -1)}),
    ],
)
def test_runtime_abi_schema_cannot_authorize_compatible_looking_values(prepared, target, supported_abis):
    # True and 1.0 compare equal to the provider's version 1, so schema checks
    # must run before compatibility membership can accidentally accept them.
    with pytest.raises(ContractError, match="supported_abis|runtime ABI"):
        validate_target(prepared, replace(target, supported_abis=supported_abis))


@pytest.mark.parametrize(
    "change",
    [
        {"capabilities": ("core:aiv",)},
        {"capabilities": frozenset({b"core:aiv"})},
        {"runtime": b"fake-runtime"},
        {"runtime": ""},
        {"version": ""},
    ],
)
def test_runtime_target_metadata_is_validated(prepared, target, change):
    with pytest.raises(ContractError, match="target|runtime name and version"):
        validate_target(prepared, replace(target, **change))


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
        replace(
            prepared, artifact=replace(prepared.artifact, files=(_source_file("entry.cpp", b"changed"),))
        ),
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
                # The request leaves view offsets and strides runtime-bound.
                *(ABIParameter(i, ABIValue.OFFSET, 8, 8) for i in range(3)),
                *(ABIParameter(i, ABIValue.STRIDE, 8, 8, axis=0) for i in range(3)),
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
