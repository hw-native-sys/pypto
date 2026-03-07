# 缓冲区管理（Buffer Management）

## 概述

TPUSH/TPOP 的环形缓冲区后端内存因平台而异。在 A2/A3 上位于全局内存（GM），较为简单。在 A5 上位于消费者的片上 SRAM，需要跨核地址解析。

参见 [TPUSH/TPOP 指令](01-tpush_tpop.md) 了解指令规范。

## A2/A3：全局内存布局

在 A2/A3 上，编排层（Orchestration）在 GM 中分配单个 `GM_SLOT_BUFFER`，并作为 INOUT 参数传递给两个 InCore 函数。

```text
GM_SLOT_BUFFER layout (bidirectional, total = 2 * SLOT_NUM * SLOT_SIZE):

┌─────────────────────────────┬─────────────────────────────┐
│  C2V ring buffer            │  V2C ring buffer            │
│  slot[0] .. slot[SLOT_NUM-1]│  slot[0] .. slot[SLOT_NUM-1]│
│  offset: 0                  │  offset: SLOT_NUM*SLOT_SIZE │
└─────────────────────────────┴─────────────────────────────┘
```

```text
Orchestration function (A2A3):
    gm_slot_buf = gm_alloc(2 * SLOT_NUM * SLOT_SIZE)

    for ...:
        cube_kernel(  ..., GM_SLOT_BUFFER=gm_slot_buf, ...)   // INOUT
        vector_kernel(..., GM_SLOT_BUFFER=gm_slot_buf, ...)   // INOUT

Orchestration function (A5):
    // CONSUMER_BUFFER_BASE values are resolved by compiler and passed explicitly
    // GM_SLOT_BUFFER is not used on A5

    for ...:
        cube_kernel(  ..., GM_SLOT_BUFFER=nullptr, ...)
        vector_kernel(..., GM_SLOT_BUFFER=nullptr, ...)
```

## A5：跨核地址问题

在 A5 上，环形缓冲区位于**消费者的本地 SRAM**（UB 或 L1）。这产生了跨核可见性问题：生产者需要消费者的本地 SRAM 地址来进行 DMA，但该地址属于另一个核心的地址空间。

```text
Cube InCore function:                    Vector InCore function:
┌─────────────────────┐                 ┌─────────────────────┐
│  tpush_to_aiv       │   ??? how to   │  consumer_buf =     │
│  DMA to Vector's UB │ ──────────────▶ │  UB[BASE..BASE+SIZE]│
│  at what address?   │   get address?  │  // local segment   │
└─────────────────────┘                 └─────────────────────┘
```

### 解决方案：`CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE`

两个**常量符号**附加到参与 TPUSH/TPOP 的每个 InCore 函数：

| 符号 | 类型 | 说明 |
| ---- | ---- | ---- |
| `{DIR}_CONSUMER_BUFFER_BASE` | `uint32_t` | 消费者 SRAM 中环形缓冲区的基址 |
| `CONSUMER_BUFFER_SIZE` | `uint32_t` | 总预留大小（`SLOT_NUM * SLOT_SIZE`） |

**关键属性：**

1. **按函数、按方向** — 每个消费者函数有自己的基址
2. **跨函数可见** — 生产者以编译时常量导入消费者的基址
3. **分配器预留** — `AllocateMemoryAddr` pass 将 `[BASE, BASE+SIZE)` 标记为已占用

**值的来源：**

| 内核来源 | 值的设置方式 |
| -------- | ------------ |
| `auto_incore` / `ExpandMixedKernel` | Pass 生成值，分配不重叠的 SRAM 区域 |
| 手动编写 | 程序员声明值，须避免冲突 |

解析后的 `CONSUMER_BUFFER_BASE` 值作为**显式参数**（`C2V_CONSUMER_BUF`、`V2C_CONSUMER_BUF`）传递给初始化函数，避免对隐式常量查找的特殊编译器支持。

### 示例：A5 上的单向 C2V

```text
Vector (consumer):
    CONSUMER_BUFFER_BASE = 0x1000
    CONSUMER_BUFFER_SIZE = 8 * TILE_SIZE

    UB layout: [normal tiles] [RESERVED: ring buffer at 0x1000] [normal tiles]
                               ◄─── allocator avoids ───►

Cube (producer):
    CONSUMER_BUFFER_BASE = 0x1000    // imported from Vector
    // Uses as DMA target in tpush_to_aiv
```

### 示例：A5 上的双向通信

每个方向有不同的消费者，各有自己的预留段：

```text
Cube:   V2C_CONSUMER_BUFFER_BASE = 0x2000 (own L1, Cube is consumer)
        C2V_CONSUMER_BUFFER_BASE = 0x1000 (imported from Vector)

Vector: C2V_CONSUMER_BUFFER_BASE = 0x1000 (own UB, Vector is consumer)
        V2C_CONSUMER_BUFFER_BASE = 0x2000 (imported from Cube)
```

### SRAM 布局（A5）

```text
Consumer's SRAM (UB or L1):

┌──────────┬──────────────────────────────┬───────────┐
│ normal   │  CONSUMER_BUFFER segment     │ normal    │
│ tiles    │  [BASE .. BASE+SIZE)         │ tiles     │
│          │  slot[0] .. slot[SLOT_NUM-1] │           │
└──────────┴──────────────────────────────┴───────────┘
             ◄─── allocator avoids ───►
```

## DSL 语法

### `pl.reserve_buffer` — 消费者侧

在当前 InCore 函数中声明一个预留的 SRAM 区域用于环形缓冲区：

```python
@pl.incore
def my_vector_kernel(...):
    pipe_buf = pl.reserve_buffer(
        name="c2v_slot_buffer",
        size=SLOT_NUM * SLOT_SIZE,
        base=pl.AUTO,                  # or literal e.g. 0x1000
    )

    aiv_initialize_pipe(DIR_C2V, SLOT_SIZE, gm_slot_buffer,
                        c2v_consumer_buf=pipe_buf.base,
                        v2c_consumer_buf=0)

    for ...:
        tile = pl.tpop_from_aic(aiv_idx=0)    # zero-copy from pipe_buf on A5
        # ... compute on tile ...
```

### `pl.import_peer_buffer` — 生产者侧

导入消费者的预留缓冲区地址：

```python
@pl.incore
def my_cube_kernel(...):
    peer_buf = pl.import_peer_buffer(
        name="c2v_slot_buffer",
        peer_func=my_vector_kernel,
    )

    aic_initialize_pipe(DIR_C2V, SLOT_SIZE, gm_slot_buffer,
                        c2v_consumer_buf=peer_buf.base,
                        v2c_consumer_buf=0)

    for ...:
        pl.tpush_to_aiv(tile, aiv_idx=0)    # DMA to peer_buf.base on A5
```

### DSL 总结

| 构造 | 用途 | 编写者 |
| ---- | ---- | ------ |
| `pl.reserve_buffer(name, size, base)` | 声明预留的 SRAM 区域 | 编译器（自动）或程序员（手动） |
| `pl.import_peer_buffer(name, peer_func)` | 导入对等函数的缓冲区地址 | 编译器（自动）或程序员（手动） |
| `pl.AUTO` | 请求编译器自动分配地址 | 用于 `base=` 参数 |

## IR 表示

`pl.reserve_buffer` 降级为 `ReserveBuffer` 节点：

```text
func @my_vector_kernel(...) {
    %pipe_buf = reserve_buffer {
        name = "c2v_slot_buffer",
        size = 4096,              // SLOT_NUM * SLOT_SIZE
        base = auto,              // or literal 0x1000
        memory_space = "UB"       // inferred from core type
    }
    ...
}
```

`pl.import_peer_buffer` 降级为 `ImportPeerBuffer` 节点：

```text
func @my_cube_kernel(...) {
    %peer_buf = import_peer_buffer {
        name = "c2v_slot_buffer",
        peer_func = @my_vector_kernel
    }
    ...
}
```

## 分配器处理

`AllocateMemoryAddr` pass 处理 `ReserveBuffer` 节点：

| `base` 值 | 行为 |
| --------- | ---- |
| `auto` | 分配器选择不冲突的地址，回写为已解析的常量 |
| 字面量（如 `0x1000`） | 分配器将区域标记为已占用，重叠时报错 |

分配完成后，`ReserveBuffer.base` 和对应的 `ImportPeerBuffer` 解析为相同的字面量值。

**分配器契约：**

1. 从函数符号表中读取 `{DIR}_CONSUMER_BUFFER_BASE` 和 `CONSUMER_BUFFER_SIZE`
2. 在 SRAM 布局中将 `[BASE, BASE + SIZE)` 标记为已预留
3. 将所有其他符号（tile、临时变量、溢出）分配到该区域之外

## `ExpandMixedKernel` 自动生成

拆分混合 InCore 函数时，该 pass 自动生成：

- 在每个消费者函数中生成 `ReserveBuffer`（`base=auto`）
- 在每个生产者函数中生成 `ImportPeerBuffer`，引用对应消费者

使用 `auto_incore` 内核时无需手动编写 `reserve_buffer` / `import_peer_buffer`。

```text
ExpandMixedKernel pass:

    Input: mixed InCore function with tpush_*/tpop_* ops

    Output:
    ┌───────────────────────────────────┐  ┌───────────────────────────────────┐
    │ Consumer function (e.g. Vector):  │  │ Producer function (e.g. Cube):    │
    │   %buf = reserve_buffer {         │  │   %peer = import_peer_buffer {    │
    │     name = "c2v_slot_buffer",     │  │     name = "c2v_slot_buffer",     │
    │     size = SLOT_NUM * SLOT_SIZE,  │  │     peer_func = @consumer_func    │
    │     base = auto,                  │  │   }                               │
    │     memory_space = "UB"           │  │   ...tpush_to_aiv uses %peer...   │
    │   }                               │  └───────────────────────────────────┘
    │   ...tpop_from_aic uses %buf...   │
    └───────────────────────────────────┘
```

## 编译器工具链要求

1. **DSL 前端** — 支持 `pl.reserve_buffer()` 和 `pl.import_peer_buffer()`
2. **`ExpandMixedKernel` pass** — 拆分时自动生成 `ReserveBuffer` / `ImportPeerBuffer` 节点
3. **`AllocateMemoryAddr` pass** — 预留 `[BASE, BASE+SIZE)`，解析 `auto` 地址，传播到 `ImportPeerBuffer`
4. **跨函数常量传播** — 已解析的 `ReserveBuffer.base` 必须传播到所有引用的 `ImportPeerBuffer` 节点
5. **验证** — 大小不得超过可用 SRAM；每个 `ImportPeerBuffer` 须有匹配的 `ReserveBuffer`；A2A3 上不生成这些节点（存在时忽略）
6. **平台条件代码生成** — A2A3 生成 GM 路径，A5 生成 SRAM 路径
