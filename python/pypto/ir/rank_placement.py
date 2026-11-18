# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Resolve ordered global ranks and lower a HOST entry onto the assigned topology."""

import sys
from dataclasses import dataclass, field, replace
from typing import cast

from pypto.pypto_core import DataType, ir

from .distributed_compiled_program import (
    ClusterConfig,
    ClusterNodeSpec,
    DistributedConfig,
    _actual_orchestration_entry,
    _validate_cluster_text,
)


@dataclass(frozen=True)
class RankPlacement:
    """The host and physical device assigned to one global rank, in list order."""

    host: str
    device: int

    def __post_init__(self) -> None:
        _validate_cluster_text(self.host, "RankPlacement.host")
        if type(self.device) is not int or self.device < 0:
            raise ValueError("RankPlacement.device must be a nonnegative integer")


@dataclass
class RankConfig:
    """An ordered allocation; construction performs no discovery or allocation.

    The existing L4 ABI groups global ranks contiguously by host. Interleaved
    groups and unequal local counts are rejected instead of renumbering ranks.
    """

    world_size: int
    placements: list[RankPlacement]
    python_executable: str = field(default_factory=lambda: sys.executable)
    comm_profile: str = "a3-fabric-v1"
    runtime: str = "tensormap_and_ringbuffer"
    num_sub_workers: int = 0
    startup_timeout_s: float = 120.0
    mpirun_path: str = "mpirun"
    mpirun_args: tuple[str, ...] = ()
    aicpu_thread_num: int = 0

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size <= 0:
            raise ValueError("RankConfig.world_size must be a positive integer")
        if len(self.placements) != self.world_size:
            raise ValueError("RankConfig requires one placement per global rank")
        self.placements = [RankPlacement(p.host, p.device) for p in self.placements]
        if len({(p.host, p.device) for p in self.placements}) != self.world_size:
            raise ValueError("RankConfig cannot assign the same device to multiple ranks")
        self.mpirun_args = tuple(self.mpirun_args)

    def resolve(self) -> DistributedConfig | ClusterConfig:
        """Produce the existing execution configuration without changing rank order."""

        snapshot = replace(self)
        nodes: list[ClusterNodeSpec] = []
        seen: set[str] = set()
        for placement in snapshot.placements:
            if not nodes or nodes[-1].host != placement.host:
                if placement.host in seen:
                    raise ValueError("Interleaved host rank groups require an extended L4 rank-order ABI")
                seen.add(placement.host)
                nodes.append(ClusterNodeSpec(placement.host, [placement.device]))
            else:
                nodes[-1].devices.append(placement.device)
        if len(nodes) == 1:
            return DistributedConfig(
                device_ids=nodes[0].devices,
                num_sub_workers=snapshot.num_sub_workers,
                runtime=snapshot.runtime,
                aicpu_thread_num=snapshot.aicpu_thread_num,
            )
        return ClusterConfig(
            nodes=nodes,
            python_executable=snapshot.python_executable,
            comm_profile=snapshot.comm_profile,
            runtime=snapshot.runtime,
            num_sub_workers=snapshot.num_sub_workers,
            startup_timeout_s=snapshot.startup_timeout_s,
            mpirun_path=snapshot.mpirun_path,
            mpirun_args=snapshot.mpirun_args,
            aicpu_thread_num=snapshot.aicpu_thread_num,
        )


def resolve_rank_request(
    rank_config: object,
    distributed_config: DistributedConfig | None = None,
    cluster_config: ClusterConfig | None = None,
) -> tuple[RankConfig, DistributedConfig | None, ClusterConfig | None]:
    """Share request validation between direct compilation and JIT compile options."""

    if not isinstance(rank_config, RankConfig):
        raise TypeError("rank_config must be a RankConfig")
    snapshot = replace(rank_config)
    resolved = snapshot.resolve()
    expected_dc: DistributedConfig | None = None
    expected_cc: ClusterConfig | None = None
    if isinstance(resolved, ClusterConfig):
        expected_cc = resolved
    else:
        expected_dc = resolved
    if distributed_config is not None and distributed_config != expected_dc:
        raise ValueError("distributed_config conflicts with rank_config")
    if cluster_config is not None and cluster_config != expected_cc:
        raise ValueError("cluster_config conflicts with rank_config")
    return snapshot, expected_dc, expected_cc


def lower_rank_placement(program: ir.Program, config: RankConfig) -> ir.Program:
    """Insert HOST relays for global-rank CHIP dispatches in a HOST entry.

    Controller-side rank loops are expanded with fresh native variable identities.
    Each relay is shared by calls with the same CHIP signature and local worker,
    independently of the caller's concrete communication window.
    """

    resolved = config.resolve()
    if not isinstance(resolved, ClusterConfig):
        return program
    root = _actual_orchestration_entry(program)
    if root is None or root.level != ir.Level.HOST:
        raise ValueError("Automatic multi-host placement requires one HOST orchestrator entry")
    functions = {func.name: func for func in program.functions.values()}
    relays: dict[tuple[str, int], ir.Function] = {}
    names = set(functions)
    span = root.span

    def integer(value: int) -> ir.ConstInt:
        return ir.ConstInt(value, DataType.INDEX, span)

    class Lower(ir.IRMutator):
        def __init__(self) -> None:
            super().__init__()
            self.constants: dict[ir.Var, int] = {}

        def constant(self, expr: ir.Expr | None) -> int | None:
            if isinstance(expr, ir.ConstInt):
                return expr.value
            if isinstance(expr, ir.Var):
                return self.constants.get(expr)
            if isinstance(expr, ir.Call) and expr.op.name in {
                "pld.system.world_size",
                "pld.system.num_workers",
            }:
                return config.world_size
            return None

        def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
            result = cast(ir.AssignStmt, super().visit_assign_stmt(op))
            value = self.constant(result.value)
            if value is not None:
                self.constants[result.var] = value
            return result

        def visit_call(self, op: ir.Call) -> ir.Expr:
            if op.op.name in {"pld.system.world_size", "pld.system.num_workers"}:
                return ir.ConstInt(config.world_size, cast(ir.ScalarType, op.type).dtype, op.span)
            result = cast(ir.Call, super().visit_call(op))
            callee = functions.get(result.op.name)
            if callee is None:
                return result
            if callee.level != ir.Level.CHIP or callee.role != ir.Role.Orchestrator:
                raise ValueError(
                    "Automatic placement supports CHIP orchestrator dispatch from the HOST entry"
                )
            rank = self.constant(dict(result.attrs).get("worker"))
            if rank is None or not 0 <= rank < config.world_size:
                raise ValueError("Automatic placement requires a statically resolved global-rank worker")
            node, local_worker = divmod(rank, len(resolved.nodes[0].devices))
            key = (callee.name, local_worker)
            if key not in relays:
                name = f"_rank_{callee.name}_{local_worker}"
                while name in names:
                    name += "_"
                names.add(name)
                params = [ir.Var(p.name_hint, p.type, p.span) for p in callee.params]
                dispatched = ir.Call(
                    result.op,
                    params,
                    dict(result.kwargs),
                    {"worker": integer(local_worker)},
                    result.type,
                    result.span,
                )
                if isinstance(result.type, ir.VoidType):
                    body = ir.SeqStmts([ir.EvalStmt(dispatched, span), ir.ReturnStmt([], span)], span)
                    returns = []
                else:
                    value = ir.Var("_rank_result", result.type, span)
                    body = ir.SeqStmts(
                        [ir.AssignStmt(value, dispatched, span), ir.ReturnStmt([value], span)], span
                    )
                    returns = [result.type]
                relay = ir.Function(
                    name,
                    list(zip(params, callee.param_directions, strict=True)),
                    returns,
                    body,
                    span,
                    level=ir.Level.HOST,
                    role=ir.Role.Orchestrator,
                )
                relays[key] = relay
            relay = relays[key]
            attrs = dict(result.attrs)
            attrs["worker"] = integer(node)
            return ir.Call(
                ir.GlobalVar(relay.name), result.args, dict(result.kwargs), attrs, result.type, result.span
            )

        def visit_submit(self, op: ir.Submit) -> ir.Expr:
            raise NotImplementedError(
                "Automatic rank placement of explicit Submit dependencies is not supported"
            )

    lower = Lower()

    def statements(stmt: ir.Stmt) -> list[ir.Stmt]:
        if isinstance(stmt, ir.SeqStmts):
            return [leaf for child in stmt.stmts for leaf in statements(child)]
        if isinstance(stmt, ir.ForStmt):
            start, stop, step = (lower.constant(expr) for expr in (stmt.start, stmt.stop, stmt.step))
            if start is None or stop is None or step is None or stmt.iter_args or stmt.return_vars:
                raise NotImplementedError(
                    "Automatic rank placement requires static loops without carried values"
                )
            if step <= 0:
                raise ValueError("Automatic rank placement requires a positive loop step")
            bodies: list[ir.Stmt] = []
            for rank in range(start, stop, step):
                cloned, _ = ir.deep_clone(stmt.body, [(stmt.loop_var, integer(rank))])
                bodies.extend(statements(cloned))
            return bodies
        return [lower.visit_stmt(stmt)]

    body = ir.SeqStmts(statements(root.body), root.body.span)
    replacement = ir.Function(
        root.name,
        list(zip(root.params, root.param_directions, strict=True)),
        root.return_types,
        body,
        root.span,
        type=root.func_type,
        level=ir.Level.CLUSTER_0,
        role=root.role,
        attrs=root.attrs,
        ir_stage=root.ir_stage,
    )
    return ir.Program(
        [replacement if fn.name == root.name else fn for fn in program.functions.values()]
        + list(relays.values()),
        program.name,
        program.span,
    )
