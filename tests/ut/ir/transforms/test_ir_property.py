# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IRProperty and IRPropertySet."""

from pypto import passes


class TestIRProperty:
    """Test IRProperty enum."""

    def test_property_values_exist(self):
        """Test that all IR properties exist."""
        assert passes.IRProperty.SSAForm is not None
        assert passes.IRProperty.TypeChecked is not None
        assert passes.IRProperty.NoNestedCalls is not None
        assert passes.IRProperty.NormalizedStmtStructure is not None
        assert passes.IRProperty.FlattenedSingleStmt is not None
        assert passes.IRProperty.SplitIncoreOrch is not None
        assert passes.IRProperty.HasMemRefs is not None

    def test_property_values_are_different(self):
        """Test that all property values are distinct."""
        props = [
            passes.IRProperty.SSAForm,
            passes.IRProperty.TypeChecked,
            passes.IRProperty.NoNestedCalls,
            passes.IRProperty.NormalizedStmtStructure,
            passes.IRProperty.FlattenedSingleStmt,
            passes.IRProperty.SplitIncoreOrch,
            passes.IRProperty.HasMemRefs,
        ]
        assert len(props) == len(set(props))


class TestIRPropertySet:
    """Test IRPropertySet operations."""

    def test_empty_set(self):
        """Test creating an empty property set."""
        ps = passes.IRPropertySet()
        assert ps.empty()
        assert ps.to_list() == []

    def test_insert_and_contains(self):
        """Test insert and contains."""
        ps = passes.IRPropertySet()
        ps.insert(passes.IRProperty.SSAForm)
        assert ps.contains(passes.IRProperty.SSAForm)
        assert not ps.contains(passes.IRProperty.TypeChecked)
        assert not ps.empty()

    def test_remove(self):
        """Test removing a property."""
        ps = passes.IRPropertySet()
        ps.insert(passes.IRProperty.SSAForm)
        ps.insert(passes.IRProperty.TypeChecked)
        ps.remove(passes.IRProperty.SSAForm)
        assert not ps.contains(passes.IRProperty.SSAForm)
        assert ps.contains(passes.IRProperty.TypeChecked)

    def test_contains_all(self):
        """Test contains_all check."""
        ps1 = passes.IRPropertySet()
        ps1.insert(passes.IRProperty.SSAForm)
        ps1.insert(passes.IRProperty.TypeChecked)
        ps1.insert(passes.IRProperty.NoNestedCalls)

        ps2 = passes.IRPropertySet()
        ps2.insert(passes.IRProperty.SSAForm)
        ps2.insert(passes.IRProperty.TypeChecked)

        assert ps1.contains_all(ps2)
        assert not ps2.contains_all(ps1)

    def test_union(self):
        """Test union operation."""
        ps1 = passes.IRPropertySet()
        ps1.insert(passes.IRProperty.SSAForm)

        ps2 = passes.IRPropertySet()
        ps2.insert(passes.IRProperty.TypeChecked)

        result = ps1.union_with(ps2)
        assert result.contains(passes.IRProperty.SSAForm)
        assert result.contains(passes.IRProperty.TypeChecked)

    def test_difference(self):
        """Test difference operation."""
        ps1 = passes.IRPropertySet()
        ps1.insert(passes.IRProperty.SSAForm)
        ps1.insert(passes.IRProperty.TypeChecked)

        ps2 = passes.IRPropertySet()
        ps2.insert(passes.IRProperty.SSAForm)

        result = ps1.difference(ps2)
        assert not result.contains(passes.IRProperty.SSAForm)
        assert result.contains(passes.IRProperty.TypeChecked)

    def test_intersection(self):
        """Test intersection operation."""
        ps1 = passes.IRPropertySet()
        ps1.insert(passes.IRProperty.SSAForm)
        ps1.insert(passes.IRProperty.TypeChecked)

        ps2 = passes.IRPropertySet()
        ps2.insert(passes.IRProperty.SSAForm)
        ps2.insert(passes.IRProperty.NoNestedCalls)

        result = ps1.intersection(ps2)
        assert result.contains(passes.IRProperty.SSAForm)
        assert not result.contains(passes.IRProperty.TypeChecked)
        assert not result.contains(passes.IRProperty.NoNestedCalls)

    def test_equality(self):
        """Test equality comparison."""
        ps1 = passes.IRPropertySet()
        ps1.insert(passes.IRProperty.SSAForm)

        ps2 = passes.IRPropertySet()
        ps2.insert(passes.IRProperty.SSAForm)

        ps3 = passes.IRPropertySet()
        ps3.insert(passes.IRProperty.TypeChecked)

        assert ps1 == ps2
        assert ps1 != ps3

    def test_to_list(self):
        """Test converting to a list."""
        ps = passes.IRPropertySet()
        ps.insert(passes.IRProperty.SSAForm)
        ps.insert(passes.IRProperty.TypeChecked)

        props = ps.to_list()
        assert len(props) == 2
        assert passes.IRProperty.SSAForm in props
        assert passes.IRProperty.TypeChecked in props

    def test_to_string(self):
        """Test string representation."""
        ps = passes.IRPropertySet()
        ps.insert(passes.IRProperty.SSAForm)
        s = str(ps)
        assert "SSAForm" in s

    def test_empty_to_string(self):
        """Test string representation of empty set."""
        ps = passes.IRPropertySet()
        assert str(ps) == "{}"


class TestPassPropertyAccessors:
    """Test Pass property accessor methods."""

    def test_convert_to_ssa_properties(self):
        """Test ConvertToSSA requires TypeChecked and produces SSAForm."""
        p = passes.convert_to_ssa()
        assert p.get_name() == "ConvertToSSA"
        assert p.get_required_properties().contains(passes.IRProperty.TypeChecked)
        assert p.get_produced_properties().contains(passes.IRProperty.SSAForm)

    def test_init_memref_properties(self):
        """Test InitMemRef produces HasMemRefs."""
        p = passes.init_mem_ref()
        assert p.get_name() == "InitMemRef"
        assert p.get_produced_properties().contains(passes.IRProperty.HasMemRefs)

    def test_basic_memory_reuse_requires_memrefs(self):
        """Test BasicMemoryReuse requires HasMemRefs."""
        p = passes.basic_memory_reuse()
        assert p.get_required_properties().contains(passes.IRProperty.HasMemRefs)

    def test_flatten_call_expr_produces_no_nested_calls(self):
        """Test FlattenCallExpr produces NoNestedCalls."""
        p = passes.flatten_call_expr()
        assert p.get_produced_properties().contains(passes.IRProperty.NoNestedCalls)

    def test_outline_incore_requires_ssa(self):
        """Test OutlineIncoreScopes requires SSAForm."""
        p = passes.outline_incore_scopes()
        assert p.get_required_properties().contains(passes.IRProperty.SSAForm)
        assert p.get_produced_properties().contains(passes.IRProperty.SplitIncoreOrch)

    def test_run_verifier_no_properties(self):
        """Test RunVerifier has no property declarations."""
        p = passes.run_verifier()
        assert p.get_required_properties().empty()
        assert p.get_produced_properties().empty()
