# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""由 PyPTO 驱动 CANNBotDSL 现有的降级流程和 SourceBundle 导出。"""

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# 可选提供方生成的 MLIR 绑定及编译器 API 暴露动态 Python 对象。
# 在此适配模块中明确保留这一边界。
ir: Any = importlib.import_module("cannbotdsl._mlir.ir")
arith: Any = importlib.import_module("cannbotdsl._mlir.dialects.arith")
cannir: Any = importlib.import_module("cannbotdsl._mlir.dialects.cannir")
func: Any = importlib.import_module("cannbotdsl._mlir.dialects.func")
call_lowering: Any = importlib.import_module("cannbotdsl.core.compiler.call_lowering")
host_schema: Any = importlib.import_module("cannbotdsl.core.compiler.host_schema")
kernel_materialize: Any = importlib.import_module("cannbotdsl.core.compiler.kernel_materialize")
passes: Any = importlib.import_module("cannbotdsl.core.compiler.passes")
binding: Any = importlib.import_module("cannbotdsl.core.contracts.binding")
contracts: Any = importlib.import_module("cannbotdsl.core.contracts.model")
frontend: Any = importlib.import_module("cannbotdsl.core.frontend.compiler")
lowering_state: Any = importlib.import_module("cannbotdsl.core.frontend.lowering_state")
ir_transport: Any = importlib.import_module("cannbotdsl.core.ir_transport")
utils_env: Any = importlib.import_module("cannbotdsl.core.utils.env")


@dataclass(frozen=True)
class SourceBundle:
    """设备源码、编译器 ABI，以及 PyPTO 所需的逻辑读写行为。"""

    source: str
    manifest: dict[str, Any]

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "device_source.hpp").write_text(self.source, encoding="utf-8")
        (directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return directory


def _flatten(tree: Any) -> list[int]:
    if isinstance(tree, (tuple, list)):
        return [leaf for child in cast(tuple[Any, ...] | list[Any], tree) for leaf in _flatten(child)]
    return [int(tree)]


def _parameters(device: Any, contract: Any, access: list[int]) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for index, logical in enumerate(contract.runtime_parameters):
        parameter: dict[str, Any] = {"logical_index": index, "name": logical.name}
        if isinstance(logical.argument, contracts.TensorContract):
            tensor = cannir.TensorType(device.type.inputs[index])
            if int(tensor.layout_map_kind) != 0:
                raise ValueError(f"CANNBotDSL parameter {logical.name!r} requires an ND storage view")
            fields: list[dict[str, str | int]] = []
            layout = tensor.logical_layout
            for name, tree in (
                ("shape", layout.shape_type.as_tuple),
                ("stride", layout.stride_type.as_tuple),
            ):
                slot = 0
                for dimension, leaf in enumerate(_flatten(tree)):
                    if leaf < 0:
                        fields.append({"field": name, "slot": slot, "dimension": dimension})
                        slot += 1
            parameter.update(
                kind="tensor",
                dtype=logical.argument.element,
                dynamic_fields=fields,
                access=("none", "read", "write", "read_write")[access[index]],
            )
        else:
            parameter.update(kind="scalar")
        parameters.append(parameter)
    return parameters


def _validate_external_ops(module: Any) -> None:
    """在优化可能消除相关操作之前，拒绝宿主启动语义。"""
    errors: list[str] = []

    def check(operation: Any) -> Any:
        name = operation.name
        unsupported = (
            "debug",
            "dump",
            "hcomm",
            "write_nbi",
            "read_nbi",
            "sync_block",
            "global_sync",
            "get_block_idx",
            "get_block_num",
            "get_subblock_dim",
        )
        if name.startswith("ascsimt.") or any(part in name for part in unsupported):
            errors.append(name)
        return ir.WalkResult.ADVANCE

    module.operation.walk(check)
    if errors:
        raise ValueError(f"Unsupported CANNBotDSL external-call operations: {sorted(set(errors))}")


def _validate_lowered(module: Any) -> str:
    _validate_external_ops(module)
    cores: list[str] = []
    groups: list[str | None] = []
    errors: list[str] = []
    for entry in module.body.operations:
        attrs = entry.attributes
        if "cannir.host" in attrs:
            continue
        if entry.operation.name != "func.func" or "cannir.device" not in attrs:
            raise ValueError("CANNBotDSL external calls require a materialized device entry only")
        if "cannir.pure_simt" in attrs or "cannir.core" not in attrs:
            raise ValueError("CANNBotDSL external calls support AIC/AIV kernels, not SIMT")
        cores.append(ir.StringAttr(attrs["cannir.core"]).value)
        groups.append(str(attrs["cannir.kernel_group"]) if "cannir.kernel_group" in attrs else None)

    def check(operation: Any) -> Any:
        name = operation.name
        if name in ("memref.get_global", "func.call"):
            errors.append(name)
        if name == "memref.alloc" and "cannir.arg_index" not in operation.attributes:
            memref = ir.MemRefType(operation.results[0].type)
            space = memref.memory_space
            if space is None or int(ir.IntegerAttr(space).value) == 0:
                errors.append("implicit GM workspace")
        return ir.WalkResult.ADVANCE

    module.operation.walk(check)
    if errors:
        raise ValueError(f"Unsupported CANNBotDSL external-call operations: {sorted(set(errors))}")
    if len(cores) == 1 and cores[0] in ("aic", "aiv"):
        return cores[0]
    if len(cores) == 2 and set(cores) == {"aic", "aiv"} and groups[0] and groups[0] == groups[1]:
        return "mixed"
    raise ValueError(f"Expected one CANNBotDSL kernel or paired MIX group, got core roles {cores}")


def export_source_bundle(kernel: Any, arguments: list[Any]) -> SourceBundle:
    """追踪核函数，并保留其自身源码生成器报告的 ABI。"""

    kernel._surface.validate_void_return_annotation(path=kernel.fn.__qualname__)
    bound = binding.bind_aot_call(kernel.fn, tuple(arguments), {}, surface_signature=kernel._surface)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        plan = call_lowering.CallLoweringPlan.from_bound_call(kernel.fn, bound)
        # SourceBundle 根据宿主侧操作数推导启动 ABI。这里只追踪该框架；
        # PyPTO 不编译或执行它的宿主入口。
        with ir.InsertionPoint(module.body):
            host = func.FuncOp("__pypto_source_bundle", (list(plan.mlir_types), []))
            host.attributes["cannir.host"] = ir.UnitAttr.get()
            host_schema.apply_physical_shape_upper_bound_attributes(host, bound.contract)
            block = host.add_entry_block()
        state = lowering_state.JitLoweringState(module=module, host_func=host)
        token = lowering_state._CURRENT_LOWERING_STATE.set(state)
        try:
            with (
                ir.InsertionPoint(block),
                frontend.frontend_options_scope(frontend.options_from_snapshot(utils_env.env())),
            ):
                positional, keywords = plan.rebuild(tuple(block.arguments))
                traced = kernel._surface.signature.bind(*positional, **keywords)
                trace_plan = call_lowering._trace_call_plan(
                    kernel.fn, traced.arguments, surface_signature=kernel._surface
                )
                device, debug, _ = kernel_materialize._materialize_device_function(
                    kernel.fn,
                    trace_plan,
                    symbol_name=f"kernel_{kernel.fn.__name__}",
                    materialized_block_dim=1,
                )
                if debug is not None:
                    raise ValueError("CANNBotDSL external calls do not support host-managed debug buffers")
                i64 = ir.IntegerType.get_signless(64)
                one = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 1))
                cannir.KernelLaunchOp(
                    ir.FlatSymbolRefAttr.get(f"kernel_{kernel.fn.__name__}"),
                    one,
                    [ir_transport.unwrap_operand(argument) for argument in block.arguments],
                    thread_dim=0,
                    dyn_ub_buf=0,
                    simt_kernel=False,
                )
                func.ReturnOp([])
            host_schema.apply_physical_shape_upper_bound_attributes(device, bound.contract)
            access: list[int] = list(cannir.infer_kernel_argument_access(device.operation))
            parameters = _parameters(device, bound.contract, access)
            module.operation.verify()
            _validate_external_ops(module)
            passes._verify_array_contracts(module)
            passes._run_cannir_passes(module)
            core = _validate_lowered(module)
            source, abi_json = cannir.translate_to_source_bundle(module.operation)
        finally:
            lowering_state._CURRENT_LOWERING_STATE.reset(token)
    entries = json.loads(abi_json)
    if len(entries) != 1 or entries[0].get("schema") != 1:
        raise ValueError(f"Expected one CANNBotDSL SourceBundle ABI v1 entry, got {entries}")
    abi = entries[0]
    physical = abi["parameters"]
    if sorted(p["logical_index"] for p in physical) != list(range(len(parameters))):
        raise ValueError("CANNBotDSL SourceBundle ABI does not cover every logical argument exactly once")
    return SourceBundle(
        source,
        {
            "export_backend": "source_bundle",
            "symbol": abi["symbol"],
            "core_type": core,
            "arch": "dav-3510",
            "parameters": parameters,
            "device_abi": abi,
        },
    )
