# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fail on registry drift without mistaking an audit entry for implemented support."""

from collections import Counter
from collections.abc import Callable
from dataclasses import replace

import pytest
from pypto import backend, ir

from .buffer_migration_inventory import HISTORICAL_CALLBACKS, MIGRATION_FAMILIES, MigrationFamily

_TARGETS = tuple(backend.BackendType.__members__.values())


def _audit_issues(
    snapshots: dict[backend.BackendType, set[str]],
    families: tuple[MigrationFamily, ...] = MIGRATION_FAMILIES,
    historical: tuple[str, ...] = HISTORICAL_CALLBACKS,
    is_live: Callable[[str], bool] = ir.is_op_registered,
) -> list[str]:
    """Compare declared migration work against independent runtime registrations."""
    issues = []
    family_ids = [family.name.split(":", 1)[0] for family in families]
    expected_ids = {f"G{i:02}" for i in range(1, 23)}
    if set(family_ids) != expected_ids:
        issues.append(f"Recipe family IDs differ: {sorted(set(family_ids) ^ expected_ids)}")
    if len(family_ids) != len(set(family_ids)):
        issues.append("Duplicate recipe family IDs")

    names = [name for family in families for name in family.operations]
    duplicates = sorted(name for name, count in Counter([*names, *historical]).items() if count > 1)
    if duplicates:
        issues.append(f"Multiply classified entries: {duplicates}")
    live_names = set(names)
    for family in families:
        if not family.operations or not family.targets or not family.targets <= set(_TARGETS):
            issues.append(f"Invalid operation or target declaration: {family.name}")
        unknown_restricted = family.restricted - set(family.operations)
        if unknown_restricted:
            issues.append(f"Restricted status has no family member: {sorted(unknown_restricted)}")

    if set(snapshots) != set(_TARGETS):
        issues.append("Runtime snapshot must include all supported targets")
    declared_targets = {target for family in families for target in family.targets}
    if declared_targets != set(_TARGETS):
        issues.append("Supported backend targets differ from the migration classification targets")
    for target, actual in snapshots.items():
        declared = set(historical) | {
            name for family in families if target in family.targets for name in family.operations
        }
        if unknown := actual - declared:
            issues.append(f"{target.name} has unclassified backend entries: {sorted(unknown)}")
        if stale := declared - actual:
            issues.append(f"{target.name} has stale classifications without callbacks: {sorted(stale)}")

    if orphaned := sorted(name for name in live_names if not is_live(name)):
        issues.append(f"Live classifications have no IR registration: {orphaned}")
    if revived := sorted(name for name in historical if is_live(name)):
        issues.append(f"Historical callbacks now have IR registration; reclassify as live: {revived}")
    return issues


@pytest.fixture(scope="module")
def registry_snapshots():
    return {
        target: set(backend.get_backend_instance(target).get_registered_op_names()) for target in _TARGETS
    }


def test_every_backend_entry_has_one_current_migration_classification(registry_snapshots):
    issues = _audit_issues(registry_snapshots)
    assert not issues, "\n".join(issues)

    # These are audited baseline sizes, not a count of implemented recipes.
    assert len(MIGRATION_FAMILIES) == 22
    assert sum(len(family.operations) for family in MIGRATION_FAMILIES) == 168
    assert len(HISTORICAL_CALLBACKS) == 8


def test_new_backend_entry_cannot_inherit_a_prefix_classification(registry_snapshots):
    changed = {target: names | {"tile.new_unclassified_op"} for target, names in registry_snapshots.items()}
    issues = _audit_issues(changed)
    assert any("unclassified backend entries: ['tile.new_unclassified_op']" in issue for issue in issues)


def test_deleted_callback_leaves_a_detectable_stale_classification(registry_snapshots):
    changed = {target: names - {"tile.add"} for target, names in registry_snapshots.items()}
    issues = _audit_issues(changed)
    assert any("stale classifications without callbacks: ['tile.add']" in issue for issue in issues)


def test_deleted_classification_leaves_a_detectable_unclassified_callback(registry_snapshots):
    families = tuple(
        replace(
            family,
            operations=tuple(name for name in family.operations if name != "tile.add"),
            restricted=family.restricted - {"tile.add"},
        )
        for family in MIGRATION_FAMILIES
    )
    issues = _audit_issues(registry_snapshots, families=families)
    assert any("unclassified backend entries: ['tile.add']" in issue for issue in issues)


def test_deleted_live_ir_definition_is_not_silently_historical(registry_snapshots):
    issues = _audit_issues(
        registry_snapshots, is_live=lambda name: name != "tile.add" and ir.is_op_registered(name)
    )
    assert "Live classifications have no IR registration: ['tile.add']" in issues


def test_restored_historical_ir_definition_requires_a_live_recipe_classification(registry_snapshots):
    issues = _audit_issues(
        registry_snapshots, is_live=lambda name: name == "tile.assign" or ir.is_op_registered(name)
    )
    assert any("reclassify as live: ['tile.assign']" in issue for issue in issues)


def test_target_expansion_is_detected_even_when_union_is_unchanged(registry_snapshots):
    changed = dict(registry_snapshots)
    changed[backend.BackendType.Ascend910B] = changed[backend.BackendType.Ascend910B] | {"tile.matmul_mx"}
    assert set.union(*changed.values()) == set.union(*registry_snapshots.values())
    issues = _audit_issues(changed)
    assert any("Ascend910B has unclassified backend entries: ['tile.matmul_mx']" in issue for issue in issues)


def test_duplicate_family_and_operator_classifications_are_rejected(registry_snapshots):
    issues = _audit_issues(registry_snapshots, families=(*MIGRATION_FAMILIES, MIGRATION_FAMILIES[0]))
    assert "Duplicate recipe family IDs" in issues
    assert any("Multiply classified entries: ['tile.alloc', 'tile.create']" in issue for issue in issues)


def test_stale_maturity_annotation_is_rejected(registry_snapshots):
    first, *remaining = MIGRATION_FAMILIES
    changed = replace(first, restricted=first.restricted | {"tile.no_longer_in_this_family"})
    issues = _audit_issues(registry_snapshots, families=(changed, *remaining))
    assert "Restricted status has no family member: ['tile.no_longer_in_this_family']" in issues


def test_incomplete_target_snapshot_is_rejected(registry_snapshots):
    issues = _audit_issues(
        {backend.BackendType.Ascend910B: registry_snapshots[backend.BackendType.Ascend910B]}
    )
    assert "Runtime snapshot must include all supported targets" in issues
