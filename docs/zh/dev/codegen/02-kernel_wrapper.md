# PTO 内核包装器生成 (Kernel Wrapper Generation)

通过 `ir.compile()` 使用 PTO 后端编译时, 会自动为每个 InCore 函数生成内核包装器, 以桥接 ptoas 输出到编排调用约定。

## 流水线

```text
InCore Function -> PTOCodegen -> .pto -> ptoas -> .cpp -> kernel_wrapper -> kernels/aiv/<name>.cpp
```

每个 InCore 函数通过 ptoas 独立编译。最终的包装器文件包含:

1. **预处理后的 ptoas 代码** (`__global__ AICORE` 替换为 `static`)
2. **`kernel_entry(__gm__ int64_t* args)`** 包装器, 解包参数数组并转发到 ptoas 函数

## 输出结构

当程序包含编排函数时, PTO 后端生成以下输出结构:

```text
output_dir/
├── passes_dump/                     # IR after each pass
├── ptoas_passes/                    # 可选：每个 ptoas Pass 后的 IR
│   └── <kernel-or-group>/            # 由 ptoas/MLIR 管理的转储树
├── ptoas/                           # Intermediates
│   ├── <func_name>.pto              # MLIR from PTOCodegen
│   └── <func_name>.cpp              # C++ from ptoas
├── kernels/aiv/
│   └── <func_name>.cpp              # Final wrapper
├── orchestration/
│   └── <orch_func_name>.cpp         # simpler runtime orchestration code
└── kernel_config.py                 # Runtime/orchestration/kernel config
```

仅当使用 `ir.compile(..., dump_ptoas_passes=True)` 或
`RunConfig(dump_ptoas_passes=True)` 时才会生成 `ptoas_passes/`。

编排代码生成使用 simpler 运行时 API (`rt_submit_task`, `make_tensor_external` 等) 生成编排 C++ 代码。

## 运行时配置 (`kernel_config.py`)

`kernel_config.py` 提供 `RUNTIME_CONFIG` 字典，派发路径据此启动程序。

| 键 | 何时写入 | 备注 |
| - | ---- | -- |
| `runtime` | 总是 | 目标平台实际使用的 Simpler 实现名。 |
| `aicpu_thread_num` | 总是（`0`） | `0` 选择运行时的架构默认值（a2a3 为 4，a5 为 5）。调用方可以覆盖。 |

通过 `ir.compile(runtime=...)` 或 `PassContext([], runtime=...)` 选择
`RuntimeKind`。需要按运行时做 IR 合法化的 pass 通过
`PassContext::GetRuntime()` 获取该选择。代码生成根据目标平台解析实际实现，
再写入 `RUNTIME_CONFIG["runtime"]`。

| 编译时选择 | 目标平台 | 制品中记录的名称 |
| ----- | ---- | -------- |
| `TENSORMAP_AND_RINGBUFFER` | 受支持的平台 | `tensormap_and_ringbuffer` |
| `HOST_BUILD_GRAPH` | A2/A3 | `host_build_graph` |
| `HOST_BUILD_GRAPH` | A5 和 A5 模拟器 | `percounter` |
| `HOST_BUILD_GRAPH_EXPERIMENTAL` | A5 和 A5 模拟器 | `host_build_graph` |

`passes.runtime_kind_to_name` 和 `passes.runtime_kind_from_name` 转换编译时的
运行时选择。加载制品时，runner 直接读取记录的实现名，并选择对应的 Simpler
运行时。实际实现名也参与构建 `@pl.jit` 缓存键。

## 参数解包

包装器按照标准约定解包 `int64_t* args`:

| 参数类型 | 解包模式 |
| -------- | -------- |
| `TensorType` | `ChipTensor*` -> `buffer.addr` -> 带类型指针 |
| `ScalarType` | `uint64_t` -> 联合体解码 -> 带类型值 |

## SPMD 身份参数

`tile.get_block_idx()`、`tile.get_block_num()` 和 `tile.get_subblock_idx()`
在 codegen 阶段被降阶为合成 `i32` 形参，PTOCodegen 把它们**追加到**
`func.func` 签名末尾，并使用有意义的命名 SSA(`%__pypto_spmd_block_idx`、
`%__pypto_spmd_block_num`、`%__pypto_spmd_subblock_idx`)。这些 op 的 IR
契约不变 -- 合成形参只出现在生成的 MLIR / C++ 中，绝不进入
`Function.params`。追加顺序固定为 `block_idx, block_num, subblock_idx`，
并各自根据函数实际使用的 op 独立决定是否追加。

```mlir
func.func @spmd_kernel(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
                       %__pypto_spmd_block_idx: i32,
                       %__pypto_spmd_block_num: i32)
                       attributes { ... } {
  %0 = arith.index_cast %__pypto_spmd_block_idx : i32 to index
  // ... 把 %0 当作 block 索引使用 ...
}
```

kernel 包装器在 dispatch 时调用 `intrinsic.h::get_block_idx(args)` /
`get_block_num(args)` 一次解析出运行时值，并把它们作为最后两个实参传给
被包装的函数:

```cpp
extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args) {
    // 从运行时 dispatch payload 读取逻辑 SPMD block 身份
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // ... 张量 / 标量 / 动态维参数解包 ...

    // 转发到 ptoas 生成的函数(block 参数追加在末尾)
    spmd_kernel(a, out, __pypto_spmd_block_idx, __pypto_spmd_block_num);
}
```

**subblock_idx(AIV lane)。** `tile.get_subblock_idx()` 走相同的合成形参通道:
wrapper 从 `intrinsic.h::get_sub_block_id(args)`(调度器写入
`GlobalContext.sub_block_id` 的运行时 per-core lane id)解析出值,并把
`__pypto_spmd_subblock_idx` 追加在 block 身份实参之后。它刻意读取运行时
lane id,而非 ccec `get_subblockid()` 寄存器 -- 后者在
`tensormap_and_ringbuffer` 调度下返回过期值。

即使张量程序没有调用 `tile.get_subblock_idx()`，split AIV FIFO 端点也会使用这个
运行时值：PTOAS 下沉 split 端点后，wrapper 后端把 lane 作为 PTO-ISA 显式重载
`TPUSH(pipe, tile, subblock_id)` / `TPOP(...)` 的第三个实参传入。如果函数还没有
合成的 subblock 形参，后端会给生成函数增加一个私有尾随形参。显式重载根据每次调用
的实际 tile 类型推导字节偏移，因此同一个自动 pipe 可以安全承载大小不同的连续传输。
与 block 身份一样，wrapper 会无条件解析运行时 lane，因为
`GlobalContext.sub_block_id` 在每个平台都由调度器填充。端点实参则受构建条件保护：
device 构建会把 lane 传给显式重载，而 CPU simulation 和 in-core cost-model 构建保留
PTO-ISA 的普通双实参端点，因为这些实现已经对 lane context 建模，且不提供仅用于
device 的显式 lane 重载。

**检测范围。** 两层各自基于函数体独立检测 SPMD usage:

- `MemRefCollectorVisitor::UsesSpmdBlockOps` / `UsesSubblockOp`(C++，位于
  `src/codegen/pto/pto_codegen.cpp`)决定 PTOCodegen 是否给该函数签名追加
  block / subblock 形参。
- `_uses_spmd_block_ops` / `_uses_dynamic_subblock_id`(Python，位于
  `python/pypto/backend/pto_backend.py`)决定 wrapper 是否把相应局部变量追加到
  对内函数调用末尾。split `TPUSH` / `TPOP` 端点还由
  `_runtime_split_fifo_endpoint_counts` 检测；它们复用该 subblock 实参，或请求上述
  私有形参。

对于 SPMD 组内自身不调用 `tile.get_block_*` 的 sibling 函数
(`group_uses_spmd=True` 但函数本身不用 SPMD ops)，wrapper 仍会声明这两个
局部变量供 `_generate_arg_unpacking` 中 `__gm_pipe_buffer` 分片逻辑消费，
但**不会**把它们追加到对内调用 -- 这与该函数 MLIR 签名保持一致。

此设计替换了旧的宏 shadow + `[[block_local]] static` /
`static thread_local` 桥接以及 `#pragma push_macro` / `#undef` /
`pop_macro` 舞步。block 和 lane 身份现在与张量指针、标量参数、动态维一样
通过调用图正常传递。

## 实现

**模块**: `python/pypto/backend/pto_backend.py`

关键函数:

- `generate()` -- 入口点: 生成所有 PTO 后端文件 (内核 + 编排 + 配置)
- `_preprocess_ptoas_output()` -- 去除重复包含, 将函数设为静态
- `_generate_arg_unpacking()` -- 根据 IR 参数类型生成 C++ 解包代码
- `_generate_kernel_wrapper()` -- 组装完整的包装器文件

## 另请参阅

- [PTO 代码生成](00-pto_codegen.md)：生成设备内核的 MLIR
- [编排代码生成](01-orchestration_codegen.md)：生成主机侧启动代码
