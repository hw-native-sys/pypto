# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for DAG (Directed Acyclic Graph) operations using PyPTO frontend.

This test validates complex multi-kernel orchestration with mixed operations,
ensuring correct code generation and execution for DAG-structured computations.

The JIT entry is imported from examples/models/vector_dag.py to keep a single
source of truth and ensure examples are guarded by tests.
"""

import pytest
import torch
from examples.models.vector_dag import golden, vector_dag
from harness import st


def _vector_dag_case():
    """Vector DAG computation with 128x128 shape: f = (a + b + 1)(a + b + 2) + (a + b).

    The example's own ``golden(tensors)`` is handed over unchanged: it already
    writes the outputs in place and keys them by parameter name, which is the
    contract a case golden may use. That keeps one source of truth for the
    reference — the example — rather than a copy of it here.
    """
    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    f = torch.zeros((128, 128), dtype=torch.float32)
    return st.case(vector_dag, a, b, f, name="vector_dag_128", golden=golden)


@st.cases(_vector_dag_case())
def test_vector_dag(case_run):
    """The DAG kernel matches the example's reference computation."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
