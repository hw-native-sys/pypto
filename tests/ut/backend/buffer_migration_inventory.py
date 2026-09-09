# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Declared migration audit, not an executable conversion or support registry.

Every legacy backend entry is classified. RESTRICTED means that some forms have
an implementation; it never means every dtype/layout/attribute form is migrated.
Production recipe tables and conversion/native/runtime tests prove support.
"""

from dataclasses import dataclass
from enum import Enum

from pypto import backend


class MigrationStatus(Enum):
    PLANNED = "planned"
    RESTRICTED = "restricted"


_TARGETS = frozenset({backend.BackendType.Ascend910B, backend.BackendType.Ascend950})


@dataclass(frozen=True)
class MigrationFamily:
    name: str
    operations: tuple[str, ...]
    targets: frozenset[backend.BackendType] = _TARGETS
    restricted: frozenset[str] = frozenset()

    def declared_status(self, name: str) -> MigrationStatus:
        """Return audit maturity for a member, without asserting form coverage."""
        if name not in self.operations:
            raise ValueError(f"Operation {name!r} is not classified in {self.name!r}")
        return MigrationStatus.RESTRICTED if name in self.restricted else MigrationStatus.PLANNED


# M0 audited both actual backend registries. Keep all names explicit so a new
# registration cannot silently inherit a family's classification by prefix.
MIGRATION_FAMILIES = (
    MigrationFamily(
        "G01: Allocation declarations",
        (
            "tile.alloc",
            "tile.create",
        ),
        restricted=frozenset({"tile.alloc", "tile.create"}),
    ),
    MigrationFamily(
        "G02: GM tile transfers",
        (
            "tile.load",
            "tile.store",
        ),
        restricted=frozenset({"tile.load", "tile.store"}),
    ),
    MigrationFamily(
        "G03: Binary vector operations",
        (
            "tile.add",
            "tile.sub",
            "tile.mul",
            "tile.maximum",
            "tile.minimum",
            "tile.and",
            "tile.or",
            "tile.shl",
            "tile.shr",
            "tile.part_add",
            "tile.part_mul",
            "tile.part_max",
            "tile.part_min",
        ),
        restricted=frozenset({"tile.add", "tile.mul"}),
    ),
    MigrationFamily(
        "G04: Scalar-vector operations",
        (
            "tile.adds",
            "tile.subs",
            "tile.muls",
            "tile.divs",
            "tile.ands",
            "tile.ors",
            "tile.shls",
            "tile.shrs",
            "tile.maximums",
            "tile.minimums",
            "tile.lrelu",
        ),
    ),
    MigrationFamily(
        "G05: Unary and precision arithmetic",
        (
            "tile.abs",
            "tile.exp",
            "tile.sqrt",
            "tile.neg",
            "tile.not",
            "tile.relu",
            "tile.log",
            "tile.recip",
            "tile.div",
            "tile.rsqrt",
        ),
    ),
    MigrationFamily(
        "G06: Carry and scratch-sensitive arithmetic",
        (
            "tile.addc",
            "tile.subc",
            "tile.addsc",
            "tile.subsc",
            "tile.xor",
            "tile.xors",
            "tile.rem",
            "tile.rems",
            "tile.fmod",
            "tile.fmods",
            "tile.prelu",
        ),
    ),
    MigrationFamily(
        "G07: Comparison and selection",
        (
            "tile.cmp",
            "tile.cmps",
            "tile.sel",
            "tile.sels",
        ),
    ),
    MigrationFamily(
        "G08: Vector generators",
        (
            "tile.full",
            "tile.ci",
            "tile.tri",
            "tile.random",
        ),
    ),
    MigrationFamily(
        "G09: Reductions and indices",
        (
            "tile.row_sum",
            "tile.row_max",
            "tile.row_min",
            "tile.row_prod",
            "tile.col_sum",
            "tile.col_max",
            "tile.col_min",
            "tile.col_prod",
            "tile.row_argmax",
            "tile.row_argmin",
            "tile.col_argmax",
            "tile.col_argmin",
        ),
    ),
    MigrationFamily(
        "G10: Axis expansion and broadcast",
        (
            "tile.col_expand",
            "tile.row_expand",
            "tile.col_expand_add",
            "tile.col_expand_div",
            "tile.col_expand_expdif",
            "tile.col_expand_max",
            "tile.col_expand_min",
            "tile.col_expand_mul",
            "tile.col_expand_sub",
            "tile.row_expand_add",
            "tile.row_expand_div",
            "tile.row_expand_expdif",
            "tile.row_expand_max",
            "tile.row_expand_min",
            "tile.row_expand_mul",
            "tile.row_expand_sub",
        ),
    ),
    MigrationFamily(
        "G11: Padding",
        (
            "tile.fillpad",
            "tile.fillpad_expand",
            "tile.fillpad_inplace",
        ),
    ),
    MigrationFamily(
        "G12: Movement, cast, transpose and assembly",
        (
            "tile.move",
            "tile.cast",
            "tile.transpose",
            "tile.concat",
            "tile.extract",
            "tile.assemble",
        ),
        restricted=frozenset({"tile.move"}),
    ),
    MigrationFamily(
        "G13: Views and valid metadata",
        (
            "tile.reshape",
            "tile.reinterpret_view",
            "tile.transpose_view",
            "tile.slice",
            "tile.set_validshape",
        ),
    ),
    MigrationFamily(
        "G14: Matrix and GEMV",
        (
            "tile.matmul",
            "tile.matmul_acc",
            "tile.matmul_bias",
            "tile.gemv",
            "tile.gemv_acc",
            "tile.gemv_bias",
        ),
    ),
    MigrationFamily(
        "G15: A5 MX and scale binding",
        (
            "tile.matmul_mx",
            "tile.matmul_mx_acc",
            "tile.matmul_mx_bias",
            "tile.tget_scale_addr",
            "tile.tquant_mx_raw",
            "tile.tmov_x2zz",
        ),
        targets=frozenset({backend.BackendType.Ascend950}),
    ),
    MigrationFamily(
        "G16: Gather, scatter and window fill",
        (
            "tile.gather",
            "tile.gatherb",
            "tile.gather_mask",
            "tile.gather_compare",
            "tile.scatter",
            "tile.scatter_mask",
            "tile.gather_row",
            "tile.mgather",
            "tile.mscatter",
        ),
    ),
    MigrationFamily(
        "G17: Sorting",
        (
            "tile.sort32",
            "tile.mrgsort_format1",
            "tile.mrgsort_format2",
        ),
    ),
    MigrationFamily(
        "G18: Scalar memory, local arrays and identities",
        (
            "tensor.dim",
            "tensor.read",
            "tensor.write",
            "tensor.view",
            "tile.read",
            "tile.write",
            "tile.get_block_idx",
            "tile.get_block_num",
            "tile.get_subblock_idx",
            "array.create",
            "array.get_element",
            "array.update_element",
        ),
    ),
    MigrationFamily(
        "G19: Barriers, cache and core synchronization",
        (
            "system.bar_all",
            "system.bar_m",
            "system.bar_v",
            "system.fence",
            "system.cacheinvalid",
            "system.sync_set",
            "system.sync_wait",
            "system.set_ffts",
            "system.syncall",
        ),
    ),
    MigrationFamily(
        "G20: Cross-core pipe ownership",
        (
            "tile.tpush_to_aic",
            "tile.tpush_to_aiv",
            "tile.tpop_from_aic",
            "tile.tpop_from_aiv",
            "system.tfree_to_aic",
            "system.tfree_to_aiv",
            "system.aic_initialize_pipe",
            "system.aiv_initialize_pipe",
            "system.reserve_buffer",
            "system.import_peer_buffer",
        ),
    ),
    MigrationFamily(
        "G21: Distributed windows and completion",
        (
            "pld.tile.get",
            "pld.tile.put",
            "pld.tile.remote_load",
            "pld.tile.remote_store",
            "pld.system.notify",
            "pld.system.wait",
            "pld.system.defer_wait",
            "pld.system.get_comm_ctx",
            "pld.system.rank",
            "pld.system.nranks",
        ),
    ),
    MigrationFamily(
        "G22: Asynchronous prefetch",
        (
            "prefetch.make_context",
            "prefetch.async_prefetch",
            "prefetch.session",
            "prefetch.wait",
        ),
    ),
)

# These callbacks have no current IR registration. They are tracked separately
# from live recipes; a restored IR definition must be reclassified as live.
HISTORICAL_CALLBACKS = (
    "tile.assign",
    "tile.move_fp",
    "tile.partadd",
    "tile.partmax",
    "tile.partmin",
    "tile.print",
    "tile.selc",
    "tile.store_fp",
)
