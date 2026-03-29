# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Model examples — multi-kernel programs with orchestration.

Suggested reading order:
  1. ffn.py                        — FFN modules with shared matmul_kernel
  2. vector_dag.py                 — multi-kernel task DAG
  3. flash_attention.py            — flash attention with loop iter_args
  4. paged_attention.py            — standard paged attention + runtime
  5. paged_attention_batch.py      — batch variant
  6. paged_attention_dynamic.py    — dynamic shapes
  7. paged_attention_multi_config.py — multi-config variant
  8. llama_mini.py                 — single-head LLaMA 7B
"""
