# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Model examples — multi-kernel programs with orchestration, ordered by complexity.

  01_ffn.py                         — FFN modules with shared matmul_kernel
  02_vector_dag.py                  — multi-kernel task DAG
  03_flash_attention.py             — flash attention with loop iter_args
  04_paged_attention.py             — standard paged attention + runtime
  05_paged_attention_batch.py       — batch variant
  06_paged_attention_dynamic.py     — dynamic shapes
  07_paged_attention_multi_config.py — multi-config variant
  08_llama_mini.py                  — single-head LLaMA 7B
"""

import importlib
import sys

_ALIASES = {
    "ffn": "01_ffn",
    "vector_dag": "02_vector_dag",
    "flash_attention": "03_flash_attention",
    "paged_attention": "04_paged_attention",
    "paged_attention_batch": "05_paged_attention_batch",
    "paged_attention_dynamic": "06_paged_attention_dynamic",
    "paged_attention_multi_config": "07_paged_attention_multi_config",
    "llama_mini": "08_llama_mini",
}

for _alias, _numbered in _ALIASES.items():
    _mod = importlib.import_module(f".{_numbered}", __package__)
    globals()[_alias] = _mod
    sys.modules[f"{__package__}.{_alias}"] = _mod
