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


def _validate_external_ops(module: Any, *, allow_subblocks: bool = False) -> None:
    """拒绝尚未适配的宿主管理资源和通信操作。"""
    errors: list[str] = []

    def check(operation: Any) -> Any:
        name = operation.name
        if allow_subblocks and name in ("cannir.get_subblock_id", "cannir.get_subblock_dim"):
            return ir.WalkResult.ADVANCE
        unsupported = (
            "debug",
            "dump",
            "hcomm",
            "write_nbi",
            "read_nbi",
            "global_sync",
            "get_block_idx",
            "get_block_num",
            "get_subblock_dim",
            "get_subblock_id",
        )
        if name.startswith("ascsimt.") or any(part in name for part in unsupported):
            errors.append(name)
        return ir.WalkResult.ADVANCE

    module.operation.walk(check)
    if errors:
        raise ValueError(f"Unsupported CANNBotDSL external-call operations: {sorted(set(errors))}")


def _bind_launch_context(devices: list[Any], host: Any, supported: set[str]) -> list[dict[str, Any]]:
    """将硬件查询改为隐藏参数；拆核后的 AIC/AIV 保持相同的设备 ABI。"""
    queries: dict[str, list[tuple[Any, Any]]] = {}
    for device in devices:

        def collect(operation: Any) -> Any:
            dialect, _, name = operation.name.partition(".")
            if dialect in ("cannir", "ascscalar") and name in supported:
                queries.setdefault(name, []).append((device, operation))
            return ir.WalkResult.ADVANCE

        device.operation.walk(collect)
    parameters: list[dict[str, Any]] = []
    for name, operations in sorted(queries.items()):
        dtype = operations[0][1].results[0].type
        index = len(host.type.inputs)
        for function in (*devices, host):
            function.entry_block.add_argument(dtype, ir.Location.unknown())
            function.attributes["function_type"] = ir.TypeAttr.get(
                ir.FunctionType.get([*function.type.inputs, dtype], list(function.type.results))
            )
            if "arg_attrs" in function.attributes:
                function.attributes["arg_attrs"] = ir.ArrayAttr.get(
                    [*ir.ArrayAttr(function.attributes["arg_attrs"]), ir.DictAttr.get()]
                )
        for device, operation in operations:
            argument = device.entry_block.arguments[index]
            operation.results[0].replace_all_uses_with(argument)
            operation.erase()
        parameters.append({"logical_index": index, "name": name, "kind": "context"})
    return parameters


def _bind_lowered_subblocks(module: Any) -> list[dict[str, Any]]:
    """保留 subblock 的范围信息完成布局推导，再接入 simpler 的向量通道编号。"""
    devices = [entry for entry in module.body.operations if "cannir.device" in entry.attributes]
    host = next(entry for entry in module.body.operations if "cannir.host" in entry.attributes)
    parameters = _bind_launch_context(devices, host, {"get_subblock_id", "get_subblock_dim"})
    if parameters:
        # 提供方的 ABI 报告从虚拟宿主入口收集参数；在拆核后同步扩展其唯一启动点。
        for operation in list(host.entry_block.operations):
            if operation.operation.name == "cannir.kernel_launch":
                with ir.InsertionPoint(operation):
                    cannir.KernelLaunchOp(
                        operation.attributes["callee"],
                        operation.operands[0],
                        [ir_transport.unwrap_operand(argument) for argument in host.entry_block.arguments],
                        thread_dim=0,
                        dyn_ub_buf=0,
                        simt_kernel=False,
                    )
                operation.operation.erase()
    return parameters


def _hard_syncall_requirement(module: Any, core: str) -> str | None:
    """从拆核后的同步操作识别实际参与核；核内屏障不要求全设备驻留。"""
    global_barrier = False

    def check(operation: Any) -> Any:
        nonlocal global_barrier
        if "sync_block" in operation.name:
            attrs = operation.attributes
            mode = int(ir.IntegerAttr(attrs["mode"]).value) if "mode" in attrs else 2
            if mode not in (0, 2):
                raise ValueError(f"Unsupported CANNBotDSL external-call sync mode: {mode}")
            global_barrier |= mode == 0
        return ir.WalkResult.ADVANCE

    module.operation.walk(check)
    return {"aic": "aic_only", "aiv": "aiv_only", "mixed": "mix"}[core] if global_barrier else None


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
                host_schema.apply_physical_shape_upper_bound_attributes(device, bound.contract)
                access: list[int] = list(cannir.infer_kernel_argument_access(device.operation))
                parameters = _parameters(device, bound.contract, access)
                # block 数量的占位值为 1，必须在常量折叠之前换成逻辑启动信息。
                parameters.extend(_bind_launch_context([device], host, {"get_block_idx", "get_block_num"}))
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
            module.operation.verify()
            _validate_external_ops(module, allow_subblocks=True)
            passes._verify_array_contracts(module)
            passes._run_cannir_passes(module)
            parameters.extend(_bind_lowered_subblocks(module))
            module.operation.verify()
            core = _validate_lowered(module)
            hard_syncall = _hard_syncall_requirement(module, core)
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
            "hard_syncall": hard_syncall,
            "arch": "dav-3510",
            "parameters": parameters,
            "device_abi": abi,
        },
    )
