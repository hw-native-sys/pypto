# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise internal schema helpers with temporary, test-only dispatcher operators."""

import ctypes
import gc
import importlib
import uuid
from concurrent.futures import ThreadPoolExecutor

import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.ir.param_info import ParamInfo
from pypto.jit.decorator import JITFunction
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime import RunConfig
from pypto.runtime.kernel.context import _ProcessKernelState
from pypto.torch import _registration_state, register, registration
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _param(name="x", shape=(-1, 3), dtype=DataType.FP32, direction=ParamDirection.In):
    """Make shared signature metadata with explicit carrier dimensions."""
    return ParamInfo(name, direction, list(shape) if shape is not None else None, dtype)


@pytest.fixture
def library():
    """Destroy all test definitions and kernels even if an assertion fails."""
    lib = torch.library.Library(f"pypto_schema_test_{uuid.uuid4().hex}", "DEF")
    try:
        yield lib
    finally:
        lib._destroy()


def test_schema_marks_mutation_and_return_aliases():
    signature = registration.RegistrationSignature(
        [
            _param(),
            _param("step", None, DataType.INT32),
            _param("out", direction=ParamDirection.Out),
            _param("cache", direction=ParamDirection.InOut),
        ],
        return_aliases=(2, 3, 2),
    )
    schema = torch._C.parse_schema(signature.schema("update"))
    assert "SymInt step" in str(schema)
    assert [str(a.type) for a in schema.arguments] == ["Tensor", "int", "Tensor", "Tensor"]
    assert [a.alias_info.is_write if a.alias_info else False for a in schema.arguments] == [
        False,
        False,
        True,
        True,
    ]
    for result, expected in zip(schema.returns, ({"a2"}, {"a3"}, {"a2"}), strict=True):
        alias = result.alias_info
        assert alias is not None
        assert alias.before_set == expected and alias.is_write


@pytest.mark.parametrize("aliases", [(), (0,), (0, 0)])
def test_fake_and_meta_preserve_original_views(library, aliases):
    signature = registration.RegistrationSignature(
        [_param(direction=ParamDirection.InOut)], return_aliases=aliases
    )
    signature.define(library, "identity")
    op = getattr(getattr(torch.ops, library.ns), "identity")
    for context in (None, FakeTensorMode()):
        if context is None:
            value = torch.empty((4, 3), device="meta")[1:3]
            result = op(value)
        else:
            with context:
                value = torch.empty((4, 3))[1:3]
                result = op(x=value)
        if not aliases:
            assert result is None
        elif len(aliases) == 1:
            assert result is value
        else:
            assert result[0] is value and result[1] is value
        assert value.shape == (2, 3) and value.storage_offset() == 3


@pytest.mark.parametrize("column_start", [0, 1, 3])
def test_empty_meta_slice_preserves_offset(column_start):
    value = torch.empty((2, 3), device="meta")[2:, column_start:]
    signature = registration.RegistrationSignature(
        [_param(shape=(-1, -1), direction=ParamDirection.InOut)], return_aliases=(0,)
    )
    assert signature.fake(value) is value
    assert value.storage_offset() == 6 + column_start


def test_symbolic_shapes_remain_symbolic():
    mode = FakeTensorMode(shape_env=ShapeEnv())
    value = mode.from_tensor(torch.empty(4, 3), static_shapes=False)
    assert isinstance(value.shape[0], torch.SymInt)
    signature = registration.RegistrationSignature(
        [_param(shape=(-1, -1), direction=ParamDirection.InOut)], return_aliases=(0,)
    )
    assert signature.fake(value) is value
    assert isinstance(value.shape[0], torch.SymInt)


def test_metadata_path_has_no_storage_or_runtime_side_effects(monkeypatch):
    value = torch.empty((2, 3), device="meta")
    signature = registration.RegistrationSignature(
        [_param(direction=ParamDirection.InOut)], return_aliases=(0,)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("metadata helper requested data, allocation or framework runtime")

    monkeypatch.setattr(torch.Tensor, "data_ptr", forbidden)
    monkeypatch.setattr(torch.Tensor, "untyped_storage", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(importlib, "import_module", forbidden)
    assert signature.fake(value) is value


@pytest.mark.parametrize("kind", ["real", "dtype", "rank", "shape", "stride", "grad", "missing"])
def test_invalid_metadata_is_rejected(kind):
    value = torch.empty((2, 3), device="meta")
    args = (value,)
    if kind == "real":
        args = (torch.empty(2, 3),)
    elif kind == "dtype":
        args = (value.to(torch.int32),)
    elif kind == "rank":
        args = (value.reshape(6),)
    elif kind == "shape":
        args = (torch.empty((2, 4), device="meta"),)
    elif kind == "stride":
        args = (torch.empty((3, 2), device="meta").t(),)
    elif kind == "grad":
        value.requires_grad_(True)
    elif kind == "missing":
        args = ()
    error = (
        TypeError if kind in ("real", "dtype", "missing") else RuntimeError if kind == "shape" else ValueError
    )
    with pytest.raises(error):
        registration.RegistrationSignature([_param()]).fake(*args)


def test_inference_allows_gradient_owned_tensors():
    value = torch.empty((2, 3), device="meta", requires_grad=True)
    with torch.no_grad():
        assert (
            registration.RegistrationSignature(
                [_param(direction=ParamDirection.InOut)], return_aliases=(0,)
            ).fake(value)
            is value
        )


@pytest.mark.parametrize("direction", [ParamDirection.Out, ParamDirection.InOut])
def test_outputs_are_mandatory(direction):
    signature = registration.RegistrationSignature([_param(), _param("out", direction=direction)])
    with pytest.raises(TypeError, match="including all Out/InOut"):
        signature.fake(torch.empty((2, 3), device="meta"))


@pytest.mark.parametrize("alias", [-1, 1, True, "0"])
def test_invalid_return_aliases(alias):
    with pytest.raises(ValueError, match="Return alias"):
        registration.RegistrationSignature([_param()], return_aliases=(alias,))


@pytest.mark.parametrize("name", ["", "x, Tensor y", "foo.bar", "ns::op", "1x", "class"])
def test_invalid_names_cannot_inject_schema(name):
    with pytest.raises(ValueError, match="identifier"):
        registration.RegistrationSignature([_param(name)])
    with pytest.raises(ValueError, match="identifier"):
        registration.RegistrationSignature([_param()]).schema(name)


def test_unsupported_contracts():
    with pytest.raises(ValueError, match="read-only identity returns"):
        registration.RegistrationSignature([_param()], return_aliases=(0,))
    with pytest.raises(ValueError, match="Duplicate"):
        registration.RegistrationSignature([_param(), _param()])
    with pytest.raises(ValueError, match="at least one tensor"):
        registration.RegistrationSignature([_param("n", None, DataType.INT32)])
    with pytest.raises(TypeError, match="dispatcher scalar type"):
        registration.RegistrationSignature([_param(), _param("n", None, DataType.UINT64)])
    with pytest.raises(ValueError, match="direction In"):
        registration.RegistrationSignature([_param(), _param("n", None, DataType.INT32, ParamDirection.Out)])
    with pytest.raises(ValueError, match="dimensions"):
        registration.RegistrationSignature([_param(shape=(-2, 3))])
    with pytest.raises(ValueError, match="Return alias"):
        registration.RegistrationSignature([_param(), _param("n", None, DataType.INT32)], return_aliases=(1,))


@pytest.mark.parametrize(
    "dtype,value,error",
    [
        (DataType.INT8, 128, ValueError),
        (DataType.UINT8, -1, ValueError),
        (DataType.INT32, 1.5, TypeError),
        (DataType.INT32, True, TypeError),
        (DataType.BOOL, 1, TypeError),
        (DataType.INT32, ctypes.c_int32(1), TypeError),
    ],
)
def test_invalid_scalar_metadata(dtype, value, error):
    signature = registration.RegistrationSignature([_param(), _param("n", None, dtype)])
    with pytest.raises(error, match="Parameter 'n'"):
        signature.fake(torch.empty((2, 3), device="meta"), value)


def test_signature_owns_copied_metadata():
    param = _param(direction=ParamDirection.InOut)
    signature = registration.RegistrationSignature([param], return_aliases=(0,))
    param.shape[1] = 8
    param.name = "changed"
    assert " x)" in signature.schema("op")
    value = torch.empty((2, 3), device="meta")
    assert signature.fake(value) is value


def test_duplicate_and_conflicting_definitions_fail(library):
    signature = registration.RegistrationSignature(
        [_param(direction=ParamDirection.InOut)], return_aliases=(0,)
    )
    signature.define(library, "identity")
    for candidate in (signature, registration.RegistrationSignature([_param()])):
        with pytest.raises(RuntimeError, match="same name and overload name"):
            candidate.define(library, "identity")
    value = torch.empty((2, 3), device="meta")
    assert getattr(getattr(torch.ops, library.ns), "identity")(value) is value


def test_missing_registration_api_fails_before_definition(library, monkeypatch):
    signature = registration.RegistrationSignature([_param()])
    with monkeypatch.context() as patch:
        patch.delattr(torch.library, "register_fake")
        patch.delattr(torch.library, "impl_abstract", raising=False)
        with pytest.raises(RuntimeError, match="register_fake or impl_abstract"):
            signature.define(library, "unavailable")
    signature.define(library, "unavailable")


@pytest.mark.parametrize("legacy", [False, True])
def test_registration_api_selection_and_library_lifetime(monkeypatch, legacy):
    """Both API names register working kernels owned by the caller's library."""
    register_fake = torch.library.register_fake
    calls = []

    def impl_abstract(qualname, func=None, *, lib=None):
        # Emulate the older API signature while using real dispatcher registration.
        assert legacy, "register_fake must take precedence when available"
        calls.append((qualname, lib))
        return register_fake(qualname, func, lib=lib)

    monkeypatch.setattr(torch.library, "impl_abstract", impl_abstract, raising=False)
    if legacy:
        monkeypatch.delattr(torch.library, "register_fake")
    signature = registration.RegistrationSignature(
        [_param(direction=ParamDirection.InOut)], return_aliases=(0,)
    )
    lib = torch.library.Library(f"pypto_schema_test_{uuid.uuid4().hex}", "DEF")
    qualname = f"{lib.ns}::identity"
    try:
        signature.define(lib, "identity")
        assert calls == ([(qualname, lib)] if legacy else [])
        op = getattr(getattr(torch.ops, lib.ns), "identity")
        value = torch.empty((2, 3), device="meta")
        assert op(value) is value
        with FakeTensorMode():
            value = torch.empty((2, 3))
            assert op(value) is value
        with pytest.raises(RuntimeError, match="same name and overload name"):
            signature.define(lib, "identity")
    finally:
        lib._destroy()
    assert qualname not in torch._C._dispatch_get_all_op_names()
    # Reusing the name also proves that the fake registration handle was removed.
    replacement = torch.library.Library(lib.ns, "DEF")
    try:
        signature.define(replacement, "identity")
    finally:
        replacement._destroy()


def test_mixed_abstract_devices_are_rejected():
    with FakeTensorMode():
        fake = torch.empty((2, 3))
    meta = torch.empty((2, 3), device="meta")
    signature = registration.RegistrationSignature([_param(), _param("out", direction=ParamDirection.Out)])
    with pytest.raises(ValueError, match="has device"):
        signature.fake(fake, meta)


@pytest.mark.parametrize(
    "dtype,value,kind",
    [
        (DataType.FP16, 1.5, "float"),
        (DataType.FP32, 2.0, "float"),
        (DataType.BF16, 1.0, "float"),
        (DataType.BOOL, True, "bool"),
        (DataType.INT8, -128, "SymInt"),
        (DataType.INT16, -32768, "SymInt"),
        (DataType.INT32, -(1 << 31), "SymInt"),
        (DataType.INT64, -(1 << 63), "SymInt"),
        (DataType.UINT8, 255, "SymInt"),
        (DataType.UINT16, 65535, "SymInt"),
        (DataType.UINT32, (1 << 32) - 1, "SymInt"),
        (DataType.INDEX, 3, "SymInt"),
    ],
)
def test_scalar_schema_and_values(dtype, value, kind):
    signature = registration.RegistrationSignature([_param(), _param("value", None, dtype)])
    # Argument.type prints both int and SymInt as "int"; inspect the full schema.
    assert f"{kind} value" in str(torch._C.parse_schema(signature.schema("op")))
    assert signature.fake(torch.empty((2, 3), device="meta"), value) is None


def test_import_and_reload_do_not_register_or_load_optional_runtime(run_without_optional_runtime):
    source = """
import importlib
import torch
before = set(torch._C._dispatch_get_all_op_names())
import pypto.torch.registration
importlib.reload(pypto.torch.registration)
assert set(torch._C._dispatch_get_all_op_names()) == before
"""
    result = run_without_optional_runtime(source)
    assert result.returncode == 0, result.stderr


def test_registered_fake_and_meta_do_not_load_optional_runtime(run_without_optional_runtime):
    """Abstract dispatch stays independent of device runtimes after registration."""
    result = run_without_optional_runtime("""
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from pypto.ir.param_info import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.torch.registration import RegistrationSignature

signature = RegistrationSignature(
    [ParamInfo('out', ParamDirection.Out, [-1, 3], DataType.FP32)], return_aliases=(0,)
)
library = torch.library.Library('pypto_optional_runtime_test', 'DEF')
try:
    signature.define(library, 'identity')
    op = torch.ops.pypto_optional_runtime_test.identity
    value = torch.empty((4, 3), device='meta')[1:]
    assert op(value) is value
    with FakeTensorMode():
        value = torch.empty((4, 3))[1:]
        assert op(value) is value
finally:
    library._destroy()
""")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("aliases", [(2,), (2, 2)])
def test_aliased_schema_matches_cpu_and_fake_metadata(library, aliases):
    signature = registration.RegistrationSignature(
        [_param(), _param("step", None, DataType.INT32), _param("out", direction=ParamDirection.InOut)],
        return_aliases=aliases,
    )
    signature.define(library, "update")

    def update(x, step, out):
        out.add_(x + step)
        args = (x, step, out)
        result = tuple(args[i] for i in aliases)
        return result[0] if len(result) == 1 else result

    library.impl("update", update, "CPU")
    op = getattr(getattr(torch.ops, library.ns), "update")
    result = torch.library.opcheck(
        op,
        (torch.ones(2, 3), 2, torch.zeros(2, 3)),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert all(status == "SUCCESS" for status in result.values())


@pytest.mark.parametrize("unbacked", [False, True])
def test_registered_symbolic_scalar_does_not_specialize_its_value(library, monkeypatch, unbacked):
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    value = mode.from_tensor(torch.empty(4, 3), static_shapes=False)
    scalar = shape_env.create_unbacked_symint() if unbacked else value.shape[0]
    assert isinstance(scalar, torch.SymInt)
    expression = scalar.node.expr
    signature = registration.RegistrationSignature(
        [_param(shape=(-1, -1)), _param("n", None, DataType.INT32)]
    )
    seen = []
    original_fake = signature.fake

    def fake(x, n):
        seen.append(n)
        return original_fake(x, n)

    monkeypatch.setattr(signature, "fake", fake)
    signature.define(library, "symbolic")
    op = getattr(getattr(torch.ops, library.ns), "symbolic")
    with mode:
        assert op(value, scalar) is None
    assert seen
    assert all(isinstance(n, torch.SymInt) and n.node.expr == expression for n in seen)
    assert scalar.node.expr == expression
    # Range validation may add inequalities, but must not specialize to the hint.
    assert all(not guard.expr.is_Equality for guard in shape_env.guards)


def test_shape_scalar_reuses_compiled_graph(library):
    signature = registration.RegistrationSignature(
        [_param(), _param("step", None, DataType.INT32), _param("out", direction=ParamDirection.Out)]
    )
    signature.define(library, "write_shape")

    def write(x, step, out):
        out.copy_(x + step)

    library.impl("write_shape", write, "CPU")
    op = getattr(getattr(torch.ops, library.ns), "write_shape")

    def call(x, out):
        op(x, x.shape[0], out)
        return out

    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(call, backend=counter, fullgraph=True, dynamic=True)
    for rows in (4, 6, 8):
        x, out = torch.ones(rows, 3), torch.empty(rows, 3)
        assert compiled(x, out) is out
        torch.testing.assert_close(out, x + rows)
    assert counter.frame_count == 1


def test_mutation_only_schema_opcheck_and_compile(library):
    signature = registration.RegistrationSignature(
        [_param(), _param("step", None, DataType.INT32), _param("out", direction=ParamDirection.Out)]
    )
    signature.define(library, "write")

    def write(x, step, out):
        out.copy_(x + step)

    library.impl("write", write, "CPU")
    op = getattr(getattr(torch.ops, library.ns), "write")
    result = torch.library.opcheck(op, (torch.ones(2, 3), 2, torch.empty(2, 3)))
    assert all(status == "SUCCESS" for status in result.values())

    def call(x, step, out):
        op(x, step, out)
        return out

    compiled = torch.compile(call, backend="aot_eager", fullgraph=True, dynamic=True)
    for rows, step in ((2, 2), (4, 7)):
        x, out = torch.ones(rows, 3), torch.empty(rows, 3)
        assert compiled(x, step, out) is out
        torch.testing.assert_close(out, x + step)


@pl.jit
def _registered_update(
    x: pl.Tensor[[16, 16], pl.FP32],
    scale: pl.Scalar[pl.FP32],
    out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
):
    a = pl.load(x, [0, 0], [16, 16])
    b = pl.load(out, [0, 0], [16, 16])
    out = pl.store(pl.add(b, pl.mul(a, scale)), [0, 0], out)
    return out, out


@pl.jit
def _registered_constant(
    x: pl.Tensor[[16, 16], pl.FP32],
    out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    value: pl.constexpr = 4,
):
    a = pl.load(x, [0, 0], [16, 16])
    out = pl.store(pl.add(a, value), [0, 0], out)
    return out


@pytest.fixture
def registered_namespace():
    namespace = f"pypto_register_test_{uuid.uuid4().hex}"
    yield namespace
    with _registration_state.lock:
        for name in list(_registration_state.registrations):
            if name.startswith(namespace + "::"):
                entry = _registration_state.registrations.pop(name)
                entry[3]._destroy()


def _cpu_fixture(namespace, name, execute):
    entry = _registration_state.registrations[f"{namespace}::{name}"]
    entry[3].impl(f"_pypto_{name}_mutate", execute, "CPU")


def test_public_registration_fake_meta_and_import_lifetime(registered_namespace, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("registration or Fake/Meta touched binary compilation or a Worker")

    monkeypatch.setattr(JITFunction, "_resolve_compiled", forbidden)
    monkeypatch.setattr(_ProcessKernelState, "ensure_worker", forbidden)
    name = f"{registered_namespace}::update"
    op = register(_registered_update, name)
    assert str(op._schema) == (
        f"{name}(Tensor(a0) x, float scale, Tensor(a2!) out) -> (Tensor(a2!), Tensor(a2!))"
    )
    meta_x, meta_out = torch.empty(16, 16, device="meta"), torch.empty(16, 16, device="meta")
    assert all(value is meta_out for value in op(x=meta_x, scale=2.0, out=meta_out))
    with FakeTensorMode():
        x, out = torch.empty(16, 16), torch.empty(16, 16)
        assert all(value is out for value in op(x, 3.0, out))
    gc.collect()
    importlib.reload(registration)
    assert registration.register(_registered_update, name) is op
    assert all(value is meta_out for value in op(meta_x, 1.0, meta_out))


def test_public_registration_compile_preserves_mutation_and_aliases(registered_namespace):
    op = registration.register(_registered_update, f"{registered_namespace}::update")

    def execute(x, scale, out):
        out.add_(x * scale)

    _cpu_fixture(registered_namespace, "update", execute)
    result = torch.library.opcheck(op, (torch.ones(16, 16), 2.0, torch.zeros(16, 16)))
    assert all(value == "SUCCESS" for value in result.values())

    def call(x, out):
        first, second = op(x, 2.0, out)
        return first, second, second + 1

    compiled = torch.compile(call, backend="aot_eager", fullgraph=True)
    x, out = torch.ones(16, 16), torch.zeros(16, 16)
    for value in (2.0, 4.0):
        first, second, other = compiled(x, out)
        assert first is out and second is out
        torch.testing.assert_close(out, torch.full_like(out, value))
        torch.testing.assert_close(other, torch.full_like(out, value + 1))


def test_public_registration_constexpr_conflicts_and_copies_config(registered_namespace):
    config = RunConfig(platform="a2a3", device_id=0, cache_config=CacheConfig(enabled=False))
    name = f"{registered_namespace}::constant"
    bindings = {"value": 5}
    op = registration.register(_registered_constant, name, constexpr=bindings, config=config)
    bindings["value"] = 6
    assert registration.register(_registered_constant, name, constexpr={"value": 5}, config=config) is op
    config.device_id = 1
    with pytest.raises(ValueError, match="different definition"):
        registration.register(_registered_constant, name, constexpr={"value": 5}, config=config)
    with pytest.raises(ValueError, match="different definition"):
        registration.register(_registered_constant, name, constexpr=bindings)
    with pytest.raises(ValueError, match="different definition"):
        registration.register(_registered_update, name)
    with pytest.raises(ValueError, match="constexpr parameters"):
        registration.register(_registered_update, f"{registered_namespace}::bad", constexpr={"scale": 2})
    assert [arg.name for arg in op._schema.arguments] == ["x", "out"]


def test_public_registration_rejects_grad_but_allows_inference(registered_namespace):
    op = registration.register(_registered_update, f"{registered_namespace}::update")

    def execute(x, scale, out):
        out.add_(x * scale)

    _cpu_fixture(registered_namespace, "update", execute)
    x, out = torch.ones(16, 16, requires_grad=True), torch.zeros(16, 16)
    with pytest.raises(ValueError, match="inference-only"):
        op(x, 1.0, out)
    with torch.no_grad():
        assert all(value is out for value in op(x, 1.0, out))
    torch.testing.assert_close(out, torch.ones_like(out))


def test_public_registration_preserves_foreign_definition(library):
    library.define("occupied(Tensor x) -> Tensor")
    with pytest.raises(ValueError, match="already defined"):
        registration.register(_registered_update, f"{library.ns}::occupied")
    assert str(getattr(torch.ops, library.ns).occupied.default._schema).endswith("(Tensor x) -> Tensor")


def test_public_registration_rolls_back_partial_definition(registered_namespace, monkeypatch):
    original = torch.library.Library.impl

    def fail(library, name, *args, **kwargs):
        if name == "update":
            raise RuntimeError("injected registration failure")
        return original(library, name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.library.Library, "impl", fail)
        with pytest.raises(RuntimeError, match="injected registration failure"):
            registration.register(_registered_update, f"{registered_namespace}::update")
    assert not any(
        name.startswith(registered_namespace + "::") for name in torch._C._dispatch_get_all_op_names()
    )
    op = registration.register(_registered_update, f"{registered_namespace}::update")
    assert op._schema.name == f"{registered_namespace}::update"


def test_public_registration_device_binding_uses_jit_kwargs(registered_namespace, monkeypatch):
    op = registration.register(
        _registered_constant, f"{registered_namespace}::constant", constexpr={"value": 7}
    )
    calls = []

    def execute(kernel, **kwargs):
        assert kernel is _registered_constant
        calls.append(kwargs)
        kwargs["out"].copy_(kwargs["x"] + kwargs["value"])
        return kwargs["out"]

    monkeypatch.setattr(JITFunction, "__call__", execute)
    mutation = getattr(torch.ops, registered_namespace)._pypto_constant_mutate.default
    keys = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)
    x, out = torch.ones(16, 16), torch.empty(16, 16)
    # Force only the dispatcher entry key; use CPU storage and a fixture JIT body.
    assert mutation.redispatch(keys, x, out) is None
    assert calls == [{"x": x, "out": out, "value": 7, "config": None}]
    torch.testing.assert_close(out, torch.full_like(out, 8))
    with pytest.raises(RuntimeError, match="incompatible shape"):
        mutation.redispatch(keys, torch.ones(32, 16), out)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="missing value"):
        op(x)


def test_public_registration_concurrent_requests_share_one_definition(registered_namespace):
    name = f"{registered_namespace}::update"
    with ThreadPoolExecutor(max_workers=4) as pool:
        operators = list(pool.map(lambda _: registration.register(_registered_update, name), range(8)))
    assert all(op is operators[0] for op in operators)
    owned = [name for name in torch._C._dispatch_get_all_op_names() if name.startswith(registered_namespace)]
    assert len(owned) == 2


@pytest.mark.parametrize("name", ["unqualified", "ns::a::b", "ns::a()", "ns::_pypto_reserved"])
def test_public_registration_rejects_invalid_names(name):
    with pytest.raises(ValueError):
        registration.register(_registered_update, name)


@pytest.mark.parametrize("kernel", [lambda x: x])
def test_public_registration_requires_jit_function(registered_namespace, kernel):
    with pytest.raises(TypeError, match="@pl.jit"):
        registration.register(kernel, f"{registered_namespace}::plain")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
