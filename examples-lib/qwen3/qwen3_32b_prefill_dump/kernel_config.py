# Kernel and Orchestration Configuration

from pathlib import Path

_ROOT_DIR = Path(__file__).parent

# Runtime configuration for tensormap_and_ringbuffer
# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3)
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 4,
	"orch_thread_num": 1,
	"block_dim": 24,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "orchestration" / "qwen3_prefill_layer.cpp"),
	"function_name": "aicpu_orchestration_entry"
}

KERNELS = [
	{"func_id": 0, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_0.cpp"), "core_type": "aiv"},
	{"func_id": 1, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_prefill_layer_incore_1_aic.cpp"), "core_type": "aic"},
	{"func_id": 2, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_1_aiv.cpp"), "core_type": "aiv"},
	{"func_id": 3, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_prefill_layer_incore_2_aic.cpp"), "core_type": "aic"},
	{"func_id": 4, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_2_aiv.cpp"), "core_type": "aiv"},
	{"func_id": 5, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_3.cpp"), "core_type": "aiv"},
	{"func_id": 6, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_4.cpp"), "core_type": "aiv"},
	{"func_id": 7, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_prefill_layer_incore_5_aic.cpp"), "core_type": "aic"},
	{"func_id": 8, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_5_aiv.cpp"), "core_type": "aiv"},
	{"func_id": 9, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_prefill_layer_incore_6_aic.cpp"), "core_type": "aic"},
	{"func_id": 10, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_6_aiv.cpp"), "core_type": "aiv"},
	{"func_id": 11, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_7.cpp"), "core_type": "aiv"},
	{"func_id": 12, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_8.cpp"), "core_type": "aiv"},
	{"func_id": 13, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_prefill_layer_incore_9_aic.cpp"), "core_type": "aic"},
	{"func_id": 14, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_9_aiv.cpp"), "core_type": "aiv"},
	{"func_id": 15, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_prefill_layer_incore_10_aic.cpp"), "core_type": "aic"},
	{"func_id": 16, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_prefill_layer_incore_10_aiv.cpp"), "core_type": "aiv"},
]
