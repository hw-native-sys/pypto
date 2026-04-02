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
	"source": str(_ROOT_DIR / "orchestration" / "qwen3_32b_training_forward_and_backward_layer.cpp"),
	"function_name": "aicpu_orchestration_entry"
}

KERNELS = [
	{"func_id": 0, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_0.cpp"), "core_type": "aiv"},
	{"func_id": 1, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_1.cpp"), "core_type": "aiv"},
	{"func_id": 2, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_32b_training_forward_and_backward_layer_incore_2_aic.cpp"), "core_type": "aic"},
	{"func_id": 3, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_2_aiv.cpp"), "core_type": "aiv"},
	{"func_id": 4, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_3.cpp"), "core_type": "aiv"},
	{"func_id": 5, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_32b_training_forward_and_backward_layer_incore_4_aic.cpp"), "core_type": "aic"},
	{"func_id": 6, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_4_aiv.cpp"), "core_type": "aiv"},
	{"func_id": 7, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_5.cpp"), "core_type": "aiv"},
	{"func_id": 8, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_6.cpp"), "core_type": "aiv"},
	{"func_id": 9, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_32b_training_forward_and_backward_layer_incore_7_aic.cpp"), "core_type": "aic"},
	{"func_id": 10, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_7_aiv.cpp"), "core_type": "aiv"},
	{"func_id": 11, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_8.cpp"), "core_type": "aiv"},
	{"func_id": 12, "source": str(_ROOT_DIR / "kernels" / "aic" / "qwen3_32b_training_forward_and_backward_layer_incore_9_aic.cpp"), "core_type": "aic"},
	{"func_id": 13, "source": str(_ROOT_DIR / "kernels" / "aiv" / "qwen3_32b_training_forward_and_backward_layer_incore_9_aiv.cpp"), "core_type": "aiv"},
]
