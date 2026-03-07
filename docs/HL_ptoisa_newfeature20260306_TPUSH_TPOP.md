# Enhanced TPUSH/TPOP ISA Design for Intra-Cluster Function Group Data Communication

## Overview

This document specifies an enhanced ISA design for `TPUSH` and `TPOP` instructions to support **intra-cluster data communication across InCore kernels** within a function group.

### Cluster Architecture

Each cluster contains **1 Cube core** and **2 buddy Vector cores** that share a hardware flag-based synchronization mechanism:

```
┌─────────────────────── Cluster ───────────────────────┐
│                                                       │
│  ┌──────────┐    flags (8 per dir)    ┌──────────┐   │
│  │  Vector 0 │◄══════════════════════►│          │   │
│  └──────────┘   SET/WAIT V→C, C→V    │   Cube   │   │
│                                       │          │   │
│  ┌──────────┐    flags (8 per dir)    │          │   │
│  │  Vector 1 │◄══════════════════════►│          │   │
│  └──────────┘   SET/WAIT V→C, C→V    └──────────┘   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

- A Vector core can **SET** a flag that the Cube core **WAITs** on, and vice versa.
- There are **8 flags per direction per peer** (Vector→Cube: 8 flags, Cube→Vector: 8 flags), for a total of 16 flags per Vector-Cube pair.
- With 2 buddy Vector cores, the cluster has **32 cross-core flags** in total (2 peers × 2 directions × 8 flags).

### Ring Buffer Data Channel

Data is moved between producer and consumer kernels through a **multi-slot ring buffer** with flow control. Each slot holds one fixed-size **Tile**. The ring buffer location depends on the platform:

| Platform | Ring Buffer Location | Description |
|---|---|---|
| **A3** | **Global Memory (GM)** | Ring buffer resides in off-chip DDR/HBM, accessible by all cores in the cluster |
| **A5** | **Consumer's on-chip SRAM** | Ring buffer resides in the consumer core's local memory: **Unified Buffer (UB)** if consumer is a Vector core, or **L1 Buffer** if consumer is a Cube core |

```
A3 Platform:                              A5 Platform:

Producer          GM            Consumer  Producer                    Consumer
┌──────┐   ┌────────────┐   ┌──────┐    ┌──────┐                 ┌──────────────┐
│      │──▶│ slot[0..N-1]│──▶│      │    │      │───────────────▶│ UB / L1      │
│ Cube │   │ (off-chip)  │   │ Vec  │    │ Cube │  DMA to local  │ slot[0..N-1] │
│ /Vec │   └────────────┘   │ /Cube│    │ /Vec │                 │ (on-chip)    │
└──────┘                     └──────┘    └──────┘                 └──────────────┘
```

The A5 placement in consumer-local SRAM eliminates the round-trip to GM, enabling lower-latency data handoff. The consumer can directly operate on tile data in its local buffer without an explicit TLOAD from GM.

The enhanced design extends `TPUSH`/`TPOP` to serve as the primary data communication mechanism between InCore kernels co-scheduled on these cores within the same cluster, enabling:

- **Producer kernel → Ring Buffer → Consumer kernel** tile-level data flow between Cube and buddy Vector cores
- Cross-core synchronization via the hardware flag mechanism (SET/WAIT, 8 flags per direction)
- Multi-slot ring buffer for pipelined execution (`SLOT_NUM` = 8 for unidirectional, 4 for bidirectional)
- Platform-adaptive buffer placement: GM (A3) or consumer-local SRAM (A5)

## Motivation: Intra-Cluster Function Group Communication

When the `ExpandMixedKernel` pass decomposes a mixed InCore function into multiple co-scheduled kernels (e.g., a data-movement kernel on Vector cores and a compute kernel on Cube cores), these kernels need an efficient, synchronized data communication channel within the same cluster.

`TPUSH`/`TPOP` with ring buffer flow control provides exactly this capability — the enhanced design formalizes how the compiler should emit `TPUSH`/`TPOP` pairs to connect the expanded kernel group.

## Enhanced Design: Tag-Based Dual-Channel FIFO Protocol

The enhanced TPUSH/TPOP design uses a **multi-slot ring buffer** with tag-based dual-channel flow control for moving fixed-size Tile data between producer and consumer kernels.

### Producer / Consumer Roles

The terms **producer** and **consumer** are conceptual roles, not bound to a specific core type:

- A **Cube core** can be a producer (e.g., matmul output → Vector for post-processing), or a consumer (e.g., receiving preprocessed data from Vector).
- A **Vector core** can be a producer (e.g., data loading / preprocessing → Cube), or a consumer (e.g., receiving matmul results from Cube).
- In some applications, **both cores are simultaneously producer and consumer** in opposite directions, forming a bidirectional data flow.

### Ring Buffer Structure

Each ring buffer is a **unidirectional** channel from one producer to one consumer. The number of slots is a compile-time constant parameter `SLOT_NUM`, specified during kernel initialization:

| Communication Pattern | SLOT_NUM | Flags Used | Description |
|---|---|---|---|
| **Unidirectional** (one direction only) | **8** | 8 flags for P2C + C2P | All 8 flags per direction dedicated to a single ring buffer |
| **Bidirectional** (both directions simultaneously) | **4** per direction | 4 flags for each of the 2 ring buffers | The 8 available flags are split equally between the two directions |

```
Unidirectional (SLOT_NUM=8):

    Cube (producer)  ──────▶  Vector (consumer)
    Ring Buffer: slot[0..7], using flags 0..7

Bidirectional (SLOT_NUM=4 per direction):

    Cube  ──── Ring Buffer A (slot[0..3], flags 0..3) ────▶  Vector
    Cube  ◀──── Ring Buffer B (slot[0..3], flags 4..7) ────  Vector
```

```
Ring Buffer  —  SLOT_NUM fixed-size Tile slots, indexed by tag

    ┌──────────────────────────────────────────────────────┐
    │  slot[0]    slot[1]    ...    slot[SLOT_NUM-1]       │   A3: Global Memory
    └──────────────────────────────────────────────────────┘   A5: Consumer's UB or L1

Signal Channels (mapped to hardware cross-core flags):
    P2C  —  Producer → Consumer  (data ready signal, indexed by tag)
    C2P  —  Consumer → Producer  (space free signal, indexed by tag)
```

Each ring buffer slot holds exactly one Tile and is identified by a **tag** (0 .. SLOT_NUM-1). The two signal channels `P2C` and `C2P` carry per-tag notifications:
- `SET P2C: tag` — producer signals "data in `slot[tag]` is ready"
- `SET C2P: tag` — consumer signals "`slot[tag]` is free for reuse"
- `WAIT P2C: tag` — consumer blocks until `slot[tag]` is ready
- `WAIT C2P: tag` — producer blocks until `slot[tag]` is free

### API Definition

#### Platform Constant

```cpp
enum PlatformID : uint8_t {
    PLATFORM_A2A3 = 0,   // A2/A3 platform: ring buffer in Global Memory
    PLATFORM_A5   = 1,   // A5 platform: ring buffer in consumer's on-chip SRAM
};
```

`PLATFORM_ID` is a **compile-time constant** generated by the compiler and embedded into the kernel binary. It is used by the initialization APIs and `tpush_*/tpop_*` instructions to select the appropriate behavior:

| PLATFORM_ID | Ring Buffer Location | `tpush_*` Behavior | `tpop_*` Behavior |
|---|---|---|---|
| `PLATFORM_A2A3` | GM (orchestration-allocated `GM_SLOT_BUFFER`) | DMA tile → GM slot | DMA GM slot → local tile |
| `PLATFORM_A5` | Consumer's on-chip SRAM (UB or L1) | DMA tile → consumer's local SRAM slot | Zero-copy: tile references local SRAM directly |

#### Direction Constants

```cpp
enum Direction : uint8_t {
    DIR_C2V = 0,   // Cube → Vector: Cube is producer, Vector is consumer
    DIR_V2C = 1,   // Vector → Cube: Vector is producer, Cube is consumer
};
```

A kernel uses `DIR_C2V` or `DIR_V2C` to specify the data flow direction. For bidirectional communication, both directions are active simultaneously (`DIR_C2V | DIR_V2C`).

#### `DIR_MASK`

A bitmask indicating which directions are active for this kernel:

| DIR_MASK | Value | Meaning | SLOT_NUM per direction |
|---|---|---|---|
| `DIR_C2V` | `0b01` | Unidirectional: Cube → Vector only | 8 |
| `DIR_V2C` | `0b10` | Unidirectional: Vector → Cube only | 8 |
| `DIR_C2V \| DIR_V2C` | `0b11` | Bidirectional: both directions | 4 |

#### `GM_SLOT_BUFFER` and `CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE`

The ring buffer backing memory differs between A2A3 and A5:

| Platform | Ring Buffer Source | Mechanism |
|---|---|---|
| **A2A3** | `GM_SLOT_BUFFER` — orchestration-allocated GM buffer, passed as INOUT argument | Same as before |
| **A5** | `CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE` — compiler-generated constant symbols per InCore function | See "Consumer SRAM Address Problem" section below |

**A2A3**: `GM_SLOT_BUFFER` is allocated in GM by the orchestration and passed to both InCore functions as INOUT.

**A5**: The ring buffer lives in the consumer's local SRAM. Its location is specified by `CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE`, which are **constant symbols attached to each InCore function** (see the detailed design in the "Cross-Core Address Problem on A5" section below). The resolved `CONSUMER_BUFFER_BASE` values are passed as **explicit arguments** (`C2V_CONSUMER_BUF`, `V2C_CONSUMER_BUF`) to the initialization functions, avoiding special compiler requirements for implicit constant lookups.

```
Orchestration function (A2A3):
    gm_slot_buf = gm_alloc(2 * SLOT_NUM * SLOT_SIZE)    // bidirectional

    for ...:
        cube_kernel(  ..., GM_SLOT_BUFFER=gm_slot_buf, ...)   // INOUT
        vector_kernel(..., GM_SLOT_BUFFER=gm_slot_buf, ...)   // INOUT

Orchestration function (A5):
    // CONSUMER_BUFFER_BASE values are resolved by compiler and passed explicitly

    for ...:
        cube_kernel(  ..., GM_SLOT_BUFFER=nullptr, ...)
        vector_kernel(..., GM_SLOT_BUFFER=nullptr, ...)
```

#### `aic_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)`

**Called on the Cube (AIC) core at kernel startup.** Initializes the ring buffer pipe(s) for the specified direction(s).

| Parameter | Type | Description |
|---|---|---|
| `DIR_MASK` | `uint8_t` | Bitmask of active directions (`DIR_C2V`, `DIR_V2C`, or both) |
| `SLOT_SIZE` | `uint32_t` | Size of each ring buffer slot in bytes (= Tile size) |
| `GM_SLOT_BUFFER` | `__gm__ void*` | GM buffer allocated by orchestration (INOUT). Active on A2A3; `nullptr` on A5 |
| `C2V_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base address for C2V direction (Vector's UB). `0` on A2A3; explicit on A5 |
| `V2C_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base address for V2C direction (Cube's own L1). `0` on A2A3; explicit on A5 |

**Description**: Binds the ring buffer pipe(s) to the appropriate backing memory based on `PLATFORM_ID`, computes `SLOT_NUM` from `DIR_MASK` (8 if unidirectional, 4 if bidirectional), and initializes internal state. On A5, the ring buffer base addresses are passed as **explicit arguments** (`C2V_CONSUMER_BUF`, `V2C_CONSUMER_BUF`) — no implicit constant symbol lookup is required. For each direction where the Cube is the **consumer** (`DIR_V2C`), it signals all slots as free to the Vector producer.

- On `PLATFORM_A2A3`: uses `GM_SLOT_BUFFER` in GM for all directions. `C2V_CONSUMER_BUF` and `V2C_CONSUMER_BUF` are ignored.
- On `PLATFORM_A5`:
  - **C2V (Cube is producer)**: uses `C2V_CONSUMER_BUF` — the Vector's UB address, passed explicitly.
  - **V2C (Cube is consumer)**: uses `V2C_CONSUMER_BUF` — Cube's own L1 address, passed explicitly.

**Pseudocode**:

```
function aic_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF):
    if DIR_MASK == (DIR_C2V | DIR_V2C):
        SLOT_NUM = 4
    else:
        SLOT_NUM = 8

    if DIR_MASK & DIR_C2V:
        // Cube is PRODUCER in C2V direction
        if PLATFORM_ID == PLATFORM_A2A3:
            c2v_ring_buf = GM_SLOT_BUFFER                         // GM buffer
        else:  // PLATFORM_A5
            c2v_ring_buf = C2V_CONSUMER_BUF                       // Vector's UB (explicit argument)
        c2v_target_tag = 0

    if DIR_MASK & DIR_V2C:
        // Cube is CONSUMER in V2C direction
        if PLATFORM_ID == PLATFORM_A2A3:
            buf_offset = (DIR_MASK & DIR_C2V) ? SLOT_NUM * SLOT_SIZE : 0
            v2c_ring_buf = GM_SLOT_BUFFER + buf_offset            // GM buffer
        else:  // PLATFORM_A5
            v2c_ring_buf = V2C_CONSUMER_BUF                       // Cube's own L1 (explicit argument)
        v2c_target_tag = 0
        // Signal all slots as free to Vector producer
        for (i = 0; i < SLOT_NUM; i++):
            SET flag_V2C_free: i          // "slot[i] is free, Vector may write"
```

#### `aiv_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)`

**Called on a Vector (AIV) core at kernel startup.** Initializes the ring buffer pipe(s) for the specified direction(s).

| Parameter | Type | Description |
|---|---|---|
| `DIR_MASK` | `uint8_t` | Bitmask of active directions (`DIR_C2V`, `DIR_V2C`, or both) |
| `SLOT_SIZE` | `uint32_t` | Size of each ring buffer slot in bytes (= Tile size) |
| `GM_SLOT_BUFFER` | `__gm__ void*` | GM buffer allocated by orchestration (INOUT). Active on A2A3; `nullptr` on A5 |
| `C2V_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base address for C2V direction (Vector's own UB). `0` on A2A3; explicit on A5 |
| `V2C_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base address for V2C direction (Cube's L1). `0` on A2A3; explicit on A5 |

**Description**: Binds the ring buffer pipe(s) to the appropriate backing memory based on `PLATFORM_ID`, computes `SLOT_NUM`, and initializes internal state. On A5, the ring buffer base addresses are passed as **explicit arguments** (`C2V_CONSUMER_BUF`, `V2C_CONSUMER_BUF`) — no implicit constant symbol lookup is required. For each direction where the Vector is the **consumer** (`DIR_C2V`), it signals all slots as free to the Cube producer.

- On `PLATFORM_A2A3`: uses `GM_SLOT_BUFFER` in GM for all directions. `C2V_CONSUMER_BUF` and `V2C_CONSUMER_BUF` are ignored.
- On `PLATFORM_A5`:
  - **C2V (Vector is consumer)**: uses `C2V_CONSUMER_BUF` — Vector's own UB address, passed explicitly.
  - **V2C (Vector is producer)**: uses `V2C_CONSUMER_BUF` — Cube's L1 address, passed explicitly.

**Pseudocode**:

```
function aiv_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF):
    if DIR_MASK == (DIR_C2V | DIR_V2C):
        SLOT_NUM = 4
    else:
        SLOT_NUM = 8

    if DIR_MASK & DIR_C2V:
        // Vector is CONSUMER in C2V direction
        if PLATFORM_ID == PLATFORM_A2A3:
            c2v_ring_buf = GM_SLOT_BUFFER                         // GM buffer
        else:  // PLATFORM_A5
            c2v_ring_buf = C2V_CONSUMER_BUF                       // Vector's own UB (explicit argument)
        c2v_target_tag = 0
        // Signal all slots as free to Cube producer
        for (i = 0; i < SLOT_NUM; i++):
            SET flag_C2V_free: i          // "slot[i] is free, Cube may write"

    if DIR_MASK & DIR_V2C:
        // Vector is PRODUCER in V2C direction
        if PLATFORM_ID == PLATFORM_A2A3:
            buf_offset = (DIR_MASK & DIR_C2V) ? SLOT_NUM * SLOT_SIZE : 0
            v2c_ring_buf = GM_SLOT_BUFFER + buf_offset            // GM buffer
        else:  // PLATFORM_A5
            v2c_ring_buf = V2C_CONSUMER_BUF                       // Cube's L1 (explicit argument)
        v2c_target_tag = 0
```

#### Buffer Layout and Cross-Core Address Problem on A5

##### A2A3: Straightforward GM Layout

On A2A3, the ring buffer resides in GM. The orchestration allocates a single `GM_SLOT_BUFFER` and passes it to both InCore functions. Both cores access the same physical GM addresses.

```
GM_SLOT_BUFFER (total size = 2 * SLOT_NUM * SLOT_SIZE for bidirectional):

    ┌─────────────────────────────┬─────────────────────────────┐
    │  C2V ring buffer            │  V2C ring buffer            │
    │  slot[0] .. slot[SLOT_NUM-1]│  slot[0] .. slot[SLOT_NUM-1]│
    │  offset: 0                  │  offset: SLOT_NUM*SLOT_SIZE │
    └─────────────────────────────┴─────────────────────────────┘
```

##### A5: Consumer SRAM Address Problem

On A5, the ring buffer for each direction resides in the **consumer's on-chip SRAM** (UB or L1). This creates a fundamental problem:

1. The ring buffer is a **local memory region in the consumer's InCore function**, allocated by the compiler in the consumer's local address space (UB or L1).
2. The **producer** needs to know this address to DMA data into it — but it lives in another core's address space.
3. In standard C/C++ semantics, a local symbol's address from one function cannot be referenced by another function. This violates symbol locality.

```
A5 Problem: C2V direction (Cube produces → Vector consumes)

    Cube InCore function:                    Vector InCore function:
    ┌─────────────────────┐                 ┌─────────────────────┐
    │  tpush_to_aiv    │   ??? how to    │  consumer_buf =     │
    │  DMA to Vector's UB │ ──────────────▶ │  UB[BASE..BASE+SIZE]│
    │  at what address?   │   get address?  │  // local segment   │
    └─────────────────────┘                 └─────────────────────┘
```

##### Solution: `CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE` Constant Symbols

The solution defines two **constant symbols** that are attached to **each InCore function** that participates in TPUSH/TPOP communication:

```cpp
const uint32_t CONSUMER_BUFFER_BASE;   // base address of the consumer's ring buffer in its local SRAM
const uint32_t CONSUMER_BUFFER_SIZE;   // total size in bytes (= SLOT_NUM * SLOT_SIZE)
```

These symbols represent a **reserved memory segment** in the consumer InCore function's local SRAM (UB for Vector, L1 for Cube). The key properties are:

1. **Per-function constants**: Each InCore function that acts as a **consumer** in any TPUSH/TPOP direction has its own `CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE`. Each InCore function that acts as a **producer** also receives the peer consumer's `CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE` so it knows the DMA target.

2. **Value origin**:
   - **Auto-generated kernels** (`auto_incore` / `ExpandMixedKernel`): The values are generated by the `ExpandMixedKernel` pass, which has visibility into both producer and consumer functions' memory layouts and can assign a non-overlapping SRAM region for the ring buffer.
   - **Manually written kernels**: The programmer specifies `CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE` as explicit constant declarations in the InCore function. The values must be chosen to not conflict with other SRAM usage.

3. **Address allocator reservation**: The downstream **memory address allocator** (e.g., `AllocateMemoryAddr` pass) must treat the segment `[CONSUMER_BUFFER_BASE, CONSUMER_BUFFER_BASE + CONSUMER_BUFFER_SIZE)` as **occupied / in-use** in the consumer function's SRAM. It must **not** allocate any other symbols (tiles, temporaries, etc.) into this region. This ensures the ring buffer and the function's normal tile allocations do not overlap.

4. **Cross-function visibility**: The `CONSUMER_BUFFER_BASE` value of a consumer function is visible to its paired producer function as a compile-time constant. The compiler ensures this by:
   - Generating both functions in the same compilation unit (natural for `ExpandMixedKernel`).
   - Emitting the consumer's `CONSUMER_BUFFER_BASE` as a constant in the producer's initialization code.

```
Compiler pipeline (A5):

    ExpandMixedKernel pass:
    ┌──────────────────────────────────────────────────────────┐
    │  1. Identify tpush_*/tpop_* pairs and directions          │
    │  2. For each consumer function:                          │
    │     - Choose CONSUMER_BUFFER_BASE in consumer's SRAM     │
    │     - Set CONSUMER_BUFFER_SIZE = SLOT_NUM * SLOT_SIZE    │
    │     - Attach as constant symbols to the consumer func    │
    │  3. For each producer function:                          │
    │     - Import the consumer's CONSUMER_BUFFER_BASE value   │
    │     - Attach as constant symbol for DMA target address   │
    └──────────────────────────────────────────────────────────┘
                              │
                              ▼
    AllocateMemoryAddr pass:
    ┌──────────────────────────────────────────────────────────┐
    │  For each InCore function with CONSUMER_BUFFER_BASE:     │
    │  - Mark [BASE, BASE+SIZE) as reserved in SRAM layout     │
    │  - Allocate all other tiles/temporaries OUTSIDE this     │
    │    region                                                │
    └──────────────────────────────────────────────────────────┘
```

**Example — C2V unidirectional on A5**:

```
Vector InCore function (consumer):
    CONSUMER_BUFFER_BASE = 0x1000            // compiler-assigned UB address
    CONSUMER_BUFFER_SIZE = 8 * TILE_SIZE     // 8 slots × tile size

    // UB layout after AllocateMemoryAddr:
    //   [0x0000 .. 0x0FFF]  — normal tiles / temporaries
    //   [0x1000 .. 0x1000 + 8*TILE_SIZE)  — RESERVED: ring buffer (CONSUMER_BUFFER segment)
    //   [above  .. UB_END]  — normal tiles / temporaries

Cube InCore function (producer):
    CONSUMER_BUFFER_BASE = 0x1000            // same value, imported from consumer
    CONSUMER_BUFFER_SIZE = 8 * TILE_SIZE     // same value

    // Cube uses CONSUMER_BUFFER_BASE as the DMA target base address
    // in its tpush_to_aiv operations (MTE writes to Vector's UB at this address)
```

**Bidirectional case**: Each direction has a different consumer. The Cube function has its own `CONSUMER_BUFFER_BASE/SIZE` for V2C (ring buffer in Cube's L1), and the Vector function has its own for C2V (ring buffer in Vector's UB). Each function also imports the **peer's** `CONSUMER_BUFFER_BASE` for the direction where it acts as producer.

```
Bidirectional A5:

    Cube InCore function:
        // V2C: Cube is consumer → own L1 segment
        V2C_CONSUMER_BUFFER_BASE = 0x2000    // Cube's L1
        V2C_CONSUMER_BUFFER_SIZE = 4 * TILE_SIZE

        // C2V: Cube is producer → needs Vector's UB address
        C2V_CONSUMER_BUFFER_BASE = 0x1000    // imported from Vector's constant

    Vector InCore function:
        // C2V: Vector is consumer → own UB segment
        C2V_CONSUMER_BUFFER_BASE = 0x1000    // Vector's UB
        C2V_CONSUMER_BUFFER_SIZE = 4 * TILE_SIZE

        // V2C: Vector is producer → needs Cube's L1 address
        V2C_CONSUMER_BUFFER_BASE = 0x2000    // imported from Cube's constant
```

##### Buffer Layout Summary

```
A2A3 (ring buffer in GM, GM_SLOT_BUFFER active):

    GM_SLOT_BUFFER:
    ┌─────────────────────────────┬─────────────────────────────┐
    │  C2V ring buffer            │  V2C ring buffer            │
    │  slot[0] .. slot[SLOT_NUM-1]│  slot[0] .. slot[SLOT_NUM-1]│
    └─────────────────────────────┴─────────────────────────────┘

A5 (ring buffer in consumer SRAM, CONSUMER_BUFFER_BASE/SIZE):

    Vector UB (for C2V, Vector is consumer):
    ┌──────────┬──────────────────────────────┬───────────┐
    │ normal   │  CONSUMER_BUFFER segment     │ normal    │
    │ tiles    │  [BASE .. BASE+SIZE)         │ tiles     │
    │          │  slot[0] .. slot[SLOT_NUM-1] │           │
    └──────────┴──────────────────────────────┴───────────┘
    ◄─── allocator avoids this region ───►

    Cube L1 (for V2C, Cube is consumer):
    ┌──────────┬──────────────────────────────┬───────────┐
    │ normal   │  CONSUMER_BUFFER segment     │ normal    │
    │ tiles    │  [BASE .. BASE+SIZE)         │ tiles     │
    │          │  slot[0] .. slot[SLOT_NUM-1] │           │
    └──────────┴──────────────────────────────┴───────────┘
```

#### Data Transfer Instructions (6 variants: 2 push + 2 pop + 2 free)

Instead of a generic `TPUSH(DIR, ...)` / `TPOP(DIR, ...)` with a direction argument, the ISA defines **six distinct instructions**, each executed on a specific core type with an implicit direction. The `DIR` parameter is removed; the direction is encoded in the instruction opcode itself.

The `tpop` and `tfree` instructions form a **split consumer protocol**: `tpop` acquires a slot (wait-ready + load data) and `tfree` releases it (signal-free + advance tag). This split is essential because the consumer may continue reading from the slot buffer after `tpop` returns — if `tpop` immediately signaled the slot as free, the producer could overwrite the data before the consumer finishes using it. By deferring the release to an explicit `tfree` call, the consumer controls exactly when the slot is recycled.

| Instruction | Executed On | Role | Direction | Description |
|---|---|---|---|---|
| `tpush_to_aiv(TILE, AIV_IDX)` | **Cube** | Producer | C2V | Push tile from Cube to buddy Vector |
| `tpush_to_aic(TILE, AIV_IDX)` | **Vector** | Producer | V2C | Push tile from Vector to buddy Cube |
| `tpop_from_aic(TILE, AIV_IDX)` | **Vector** | Consumer | C2V | Acquire slot: wait ready + load tile (does **not** release slot) |
| `tpop_from_aiv(TILE, AIV_IDX)` | **Cube** | Consumer | V2C | Acquire slot: wait ready + load tile (does **not** release slot) |
| `tfree_to_aic(AIV_IDX)` | **Vector** | Consumer | C2V | Release slot: signal Cube producer that slot is free + advance tag |
| `tfree_to_aiv(AIV_IDX)` | **Cube** | Consumer | V2C | Release slot: signal Vector producer that slot is free + advance tag |

#### `tpush_to_aiv(TILE, AIV_IDX)`

**Executed on Cube (AIC).** Pushes a tile into the C2V ring buffer destined for a buddy Vector core.

| Parameter | Type | Description |
|---|---|---|
| `TILE` | `Tile&` | Source tile data to push |
| `AIV_IDX` | `uint8_t` | Target buddy Vector core index (0 or 1) |

**Pseudocode**:

```
function tpush_to_aiv(TILE, AIV_IDX):
    pipe = get_pipe_state(C2V, AIV_IDX)

    // 1) Wait for slot to be free (Vector consumer has released it)
    WAIT flag_free[C2V, AIV_IDX]: pipe.target_tag

    // 2) DMA tile data into ring buffer slot
    dst_addr = pipe.ring_buf_base + pipe.target_tag * SLOT_SIZE
    if PLATFORM_ID == PLATFORM_A2A3:
        MTE_copy(src=TILE.data, dst=dst_addr, size=SLOT_SIZE)    // tile → GM slot
    else:  // PLATFORM_A5
        MTE_copy(src=TILE.data, dst=dst_addr, size=SLOT_SIZE)    // tile → Vector's UB slot
    SET  mte_flag
    WAIT mte_flag

    // 3) Signal Vector consumer: data in slot is ready
    SET flag_ready[C2V, AIV_IDX]: pipe.target_tag

    // 4) Advance to next slot
    pipe.target_tag = (pipe.target_tag + 1) % SLOT_NUM
```

#### `tpush_to_aic(TILE, AIV_IDX)`

**Executed on Vector (AIV).** Pushes a tile into the V2C ring buffer destined for the buddy Cube core.

| Parameter | Type | Description |
|---|---|---|
| `TILE` | `Tile&` | Source tile data to push |
| `AIV_IDX` | `uint8_t` | This Vector core's own index (0 or 1), identifying which flag pair to use |

**Pseudocode**:

```
function tpush_to_aic(TILE, AIV_IDX):
    pipe = get_pipe_state(V2C, AIV_IDX)

    // 1) Wait for slot to be free (Cube consumer has released it)
    WAIT flag_free[V2C, AIV_IDX]: pipe.target_tag

    // 2) DMA tile data into ring buffer slot
    dst_addr = pipe.ring_buf_base + pipe.target_tag * SLOT_SIZE
    if PLATFORM_ID == PLATFORM_A2A3:
        MTE_copy(src=TILE.data, dst=dst_addr, size=SLOT_SIZE)    // tile → GM slot
    else:  // PLATFORM_A5
        MTE_copy(src=TILE.data, dst=dst_addr, size=SLOT_SIZE)    // tile → Cube's L1 slot
    SET  mte_flag
    WAIT mte_flag

    // 3) Signal Cube consumer: data in slot is ready
    SET flag_ready[V2C, AIV_IDX]: pipe.target_tag

    // 4) Advance to next slot
    pipe.target_tag = (pipe.target_tag + 1) % SLOT_NUM
```

#### `tpop_from_aic(TILE, AIV_IDX)`

**Executed on Vector (AIV).** Acquires the next slot from the C2V ring buffer (data that Cube pushed). The slot remains **held** until the consumer explicitly calls `tfree_to_aic` to release it. This ensures the producer cannot overwrite the slot while the consumer is still reading.

| Parameter | Type | Description |
|---|---|---|
| `TILE` | `Tile&` | Destination tile to receive data |
| `AIV_IDX` | `uint8_t` | This Vector core's own index (0 or 1), identifying which flag pair to use |

**Pseudocode**:

```
function tpop_from_aic(TILE, AIV_IDX):
    pipe = get_pipe_state(C2V, AIV_IDX)

    // 1) Wait for data to be ready (Cube producer has filled the slot)
    WAIT flag_ready[C2V, AIV_IDX]: pipe.target_tag

    // 2) Load tile data from ring buffer slot
    src_addr = pipe.ring_buf_base + pipe.target_tag * SLOT_SIZE
    if PLATFORM_ID == PLATFORM_A2A3:
        MTE_copy(src=src_addr, dst=TILE.data, size=SLOT_SIZE)    // GM slot → local tile
        SET  mte_flag
        WAIT mte_flag
    else:  // PLATFORM_A5
        TILE.data = src_addr             // zero-copy: data already in Vector's UB

    // NOTE: Slot is NOT released here. The consumer must call
    //   tfree_to_aic(AIV_IDX)
    // after it has finished using TILE's data.
```

#### `tpop_from_aiv(TILE, AIV_IDX)`

**Executed on Cube (AIC).** Acquires the next slot from the V2C ring buffer (data that Vector pushed). The slot remains **held** until the consumer explicitly calls `tfree_to_aiv` to release it.

| Parameter | Type | Description |
|---|---|---|
| `TILE` | `Tile&` | Destination tile to receive data |
| `AIV_IDX` | `uint8_t` | Source buddy Vector core index (0 or 1) |

**Pseudocode**:

```
function tpop_from_aiv(TILE, AIV_IDX):
    pipe = get_pipe_state(V2C, AIV_IDX)

    // 1) Wait for data to be ready (Vector producer has filled the slot)
    WAIT flag_ready[V2C, AIV_IDX]: pipe.target_tag

    // 2) Load tile data from ring buffer slot
    src_addr = pipe.ring_buf_base + pipe.target_tag * SLOT_SIZE
    if PLATFORM_ID == PLATFORM_A2A3:
        MTE_copy(src=src_addr, dst=TILE.data, size=SLOT_SIZE)    // GM slot → local tile
        SET  mte_flag
        WAIT mte_flag
    else:  // PLATFORM_A5
        TILE.data = src_addr             // zero-copy: data already in Cube's L1

    // NOTE: Slot is NOT released here. The consumer must call
    //   tfree_to_aiv(AIV_IDX)
    // after it has finished using TILE's data.
```

#### `tfree_to_aic(AIV_IDX)`

**Executed on Vector (AIV).** Releases the currently held C2V slot back to the Cube producer. Must be called **after** the consumer has finished reading all data obtained from the preceding `tpop_from_aic`. This completes the consumer half of the C2V handshake.

| Parameter | Type | Description |
|---|---|---|
| `AIV_IDX` | `uint8_t` | This Vector core's own index (0 or 1), identifying which pipe state to advance |

**Pseudocode**:

```
function tfree_to_aic(AIV_IDX):
    pipe = get_pipe_state(C2V, AIV_IDX)

    // 1) Signal Cube producer: slot is free for reuse
    SET flag_free[C2V, AIV_IDX]: pipe.target_tag

    // 2) Advance to next slot
    pipe.target_tag = (pipe.target_tag + 1) % SLOT_NUM
```

#### `tfree_to_aiv(AIV_IDX)`

**Executed on Cube (AIC).** Releases the currently held V2C slot back to the Vector producer. Must be called **after** the consumer has finished reading all data obtained from the preceding `tpop_from_aiv`. This completes the consumer half of the V2C handshake.

| Parameter | Type | Description |
|---|---|---|
| `AIV_IDX` | `uint8_t` | Target buddy Vector core index (0 or 1), identifying which pipe state to advance |

**Pseudocode**:

```
function tfree_to_aiv(AIV_IDX):
    pipe = get_pipe_state(V2C, AIV_IDX)

    // 1) Signal Vector producer: slot is free for reuse
    SET flag_free[V2C, AIV_IDX]: pipe.target_tag

    // 2) Advance to next slot
    pipe.target_tag = (pipe.target_tag + 1) % SLOT_NUM
```

### Flag Assignment

The 8 hardware flags per direction per peer are mapped as follows:

```
Unidirectional (DIR_C2V only, SLOT_NUM=8):

    flag_ready[C2V, aiv_idx] : flags 0..7   (Cube SETs, Vector WAITs)
    flag_free [C2V, aiv_idx] : flags 0..7   (Vector SETs, Cube WAITs)

Bidirectional (DIR_C2V | DIR_V2C, SLOT_NUM=4):

    flag_ready[C2V, aiv_idx] : flags 0..3   (Cube SETs, Vector WAITs)
    flag_free [C2V, aiv_idx] : flags 0..3   (Vector SETs, Cube WAITs)
    flag_ready[V2C, aiv_idx] : flags 4..7   (Vector SETs, Cube WAITs)
    flag_free [V2C, aiv_idx] : flags 4..7   (Cube SETs, Vector WAITs)
```

### Timing Diagram: Unidirectional C2V (SLOT_NUM=4)

The split `tpop`/`tfree` protocol allows the consumer to hold a slot while computing on the data. The `tfree` may happen any time before the ring buffer wraps back to the same slot.

```
          iter 0              iter 1              iter 2              iter 3              iter 4
tag:        0                   1                   2                   3                   0

AIC (Cube, producer):
          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
          │tpush_to_aiv  │  │tpush_to_aiv  │  │tpush_to_aiv  │  │tpush_to_aiv  │  │tpush_to_aiv  │
          │ WAIT f:0     │  │ WAIT f:1     │  │ WAIT f:2     │  │ WAIT f:3     │  │ WAIT f:0     │
          │ MTE → 0      │  │ MTE → 1      │  │ MTE → 2      │  │ MTE → 3      │  │ MTE → 0      │
          │ SET r:0      │  │ SET r:1      │  │ SET r:2      │  │ SET r:3      │  │ SET r:0      │
          └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                 │                  │                  │                  │                  │
                 ▼ ready            ▼ ready            ▼ ready            ▼ ready            ▼ ready

AIV (Vector, consumer):
                 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                 │tpop_from_aic │  │tpop_from_aic │  │tpop_from_aic │  │tpop_from_aic │
                 │ WAIT r:0     │  │ WAIT r:1     │  │ WAIT r:2     │  │ WAIT r:3     │
                 │ load [0]     │  │ load [1]     │  │ load [2]     │  │ load [3]     │
                 └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                        │                  │                  │                  │
                    (consumer             (consumer          (consumer          (consumer
                     uses data)            uses data)         uses data)         uses data)
                        │                  │                  │                  │
                 ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐
                 │tfree_to_aic  │  │tfree_to_aic  │  │tfree_to_aic  │  │tfree_to_aic  │
                 │ SET f:0      │  │ SET f:1      │  │ SET f:2      │  │ SET f:3      │
                 │ tag++ → 1    │  │ tag++ → 2    │  │ tag++ → 3    │  │ tag++ → 0    │
                 └──────┬───────┘  └──────┴───────┘  └──────┴───────┘  └──────┴───────┘
                        │
                        ▼ free (slot 0 now available for iter 4's tpush)

Legend: r = flag_ready, f = flag_free
        aiv_initialize_pipe pre-SETs f:0..3 so AIC does not block initially
```

### Timing Diagram: Bidirectional (SLOT_NUM=4)

```
          iter 0              iter 1              iter 2              iter 3              iter 4

AIC (Cube):
  C2V:    tpush_to_aiv     tpush_to_aiv     tpush_to_aiv     tpush_to_aiv     tpush_to_aiv
          tag=0               tag=1               tag=2               tag=3               tag=0
  V2C:         tpop_from_aiv     tpop_from_aiv     tpop_from_aiv     tpop_from_aiv
               tag=0               tag=1               tag=2               tag=3
               (use data)         (use data)         (use data)         (use data)
               tfree_to_aiv      tfree_to_aiv      tfree_to_aiv      tfree_to_aiv

AIV (Vector):
  V2C:    tpush_to_aic       tpush_to_aic       tpush_to_aic       tpush_to_aic       tpush_to_aic
          tag=0               tag=1               tag=2               tag=3               tag=0
  C2V:         tpop_from_aic       tpop_from_aic       tpop_from_aic       tpop_from_aic
               tag=0               tag=1               tag=2               tag=3
               (use data)          (use data)          (use data)          (use data)
               tfree_to_aic        tfree_to_aic        tfree_to_aic        tfree_to_aic

Flag usage (per AIV peer):
  flags 0..3 : C2V direction (ready + free)
  flags 4..7 : V2C direction (ready + free)
```

### Key Properties

1. **No deadlock**: The consumer side (`aiv_initialize_pipe` or `aic_initialize_pipe`) pre-signals all SLOT_NUM slots as free before the main loop begins, so the producer can fill up to SLOT_NUM slots before blocking.
2. **Backpressure**: If the producer is faster than the consumer, `tpush_*` blocks at `WAIT flag_free` when all slots are occupied; if the consumer is faster, `tpop_*` blocks at `WAIT flag_ready` when no data is ready.
3. **In-order delivery**: Both sides advance `target_tag` in strict round-robin order `(tag + 1) % SLOT_NUM`, guaranteeing FIFO semantics. The producer advances in `tpush`; the consumer advances in `tfree` (not in `tpop`).
4. **Decoupled DMA**: `tpush_*` uses MTE for async data transfer with an explicit `mte_flag` wait to ensure completion before signaling the consumer.
5. **Buddy core selection**: `AIV_IDX` (0 or 1) selects which of the two buddy Vector cores to communicate with, enabling independent pipes to different Vector cores from the same Cube core.
6. **Direction encoded in opcode**: Each instruction has an implicit, fixed direction — no runtime `DIR` argument is needed. This enables the compiler to statically verify that the correct instruction is used on the correct core type.
7. **Split consumer protocol (tpop/tfree)**: `tpop` only acquires the slot (wait-ready + load); `tfree` releases it (signal-free + advance). This prevents the producer from overwriting a slot while the consumer is still reading from it. The compiler or programmer must ensure every `tpop` is paired with a corresponding `tfree` before the same slot is needed again (i.e., before wrapping around the ring buffer by SLOT_NUM iterations).

### API Summary

| API | Called On | Role | Direction | Description |
|---|---|---|---|---|
| `aic_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)` | Cube (AIC) | Setup | — | Bind ring buffer, init tags, pre-signal free slots for V2C |
| `aiv_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)` | Vector (AIV) | Setup | — | Bind ring buffer, init tags, pre-signal free slots for C2V |
| `tpush_to_aiv(TILE, AIV_IDX)` | Cube (AIC) | Producer | C2V | Wait free → DMA tile to ring buffer → signal ready |
| `tpush_to_aic(TILE, AIV_IDX)` | Vector (AIV) | Producer | V2C | Wait free → DMA tile to ring buffer → signal ready |
| `tpop_from_aic(TILE, AIV_IDX)` | Vector (AIV) | Consumer | C2V | Wait ready → load tile (DMA or zero-copy). Slot remains **held** |
| `tpop_from_aiv(TILE, AIV_IDX)` | Cube (AIC) | Consumer | V2C | Wait ready → load tile (DMA or zero-copy). Slot remains **held** |
| `tfree_to_aic(AIV_IDX)` | Vector (AIV) | Consumer | C2V | Signal Cube producer: slot free → advance tag. **Must follow** `tpop_from_aic` |
| `tfree_to_aiv(AIV_IDX)` | Cube (AIC) | Consumer | V2C | Signal Vector producer: slot free → advance tag. **Must follow** `tpop_from_aiv` |

### `CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE` — Constant Symbols per InCore Function

| Symbol | Type | Scope | Description |
|---|---|---|---|
| `{DIR}_CONSUMER_BUFFER_BASE` | `uint32_t` | Per InCore function, per direction | Base address of the ring buffer in the consumer's local SRAM |
| `CONSUMER_BUFFER_SIZE` | `uint32_t` | Per InCore function | Total reserved size (`SLOT_NUM * SLOT_SIZE`) |

These are constant symbols embedded in each InCore function's symbol table, used for two purposes:

1. **Address allocator reservation**: The `AllocateMemoryAddr` pass reads these symbols and marks the corresponding SRAM region as occupied, preventing other allocations from overlapping.
2. **Explicit argument to initialization**: The resolved `CONSUMER_BUFFER_BASE` values are passed as **explicit arguments** (`C2V_CONSUMER_BUF`, `V2C_CONSUMER_BUF`) to `aic_initialize_pipe` / `aiv_initialize_pipe`. This avoids any special compiler mechanism for implicit constant lookups inside the init function.

Each function that participates in TPUSH/TPOP communication has:

- **As consumer** (owns the buffer): `{DIR}_CONSUMER_BUFFER_BASE` is the base address of the reserved segment in its own SRAM (UB or L1). The value is passed to its own init function and also to the paired producer's init function.
- **As producer** (DMA target): receives the consumer's `{DIR}_CONSUMER_BUFFER_BASE` value via `pl.import_peer_buffer`, and passes it as an explicit argument to its init function.

**Value generation**:

| Kernel Origin | How values are set |
|---|---|
| **`auto_incore` / `ExpandMixedKernel`** | The pass generates both constants when splitting the mixed InCore function. It assigns a non-overlapping SRAM region for the ring buffer. |
| **Manually written** | The programmer declares `CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE` as explicit constants. Values must be chosen to avoid conflict with other SRAM usage. |

**Address allocator contract**: The `AllocateMemoryAddr` pass (or equivalent downstream allocator) must:

1. Read `{DIR}_CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE` from the function's symbol table.
2. Mark `[BASE, BASE + SIZE)` as **reserved / occupied** in the SRAM layout.
3. Allocate all other symbols (tiles, temporaries, spills) **outside** this region.

This ensures the ring buffer segment and normal compute allocations never overlap.

### DSL Grammar: `pl.reserve_buffer` — Reserved Address Space Declaration

The compiler must provide a **DSL-level mechanism** for InCore kernel programs to declare reserved address space for the SLOT_BUFFER. This is necessary because:

1. The address allocator must know which SRAM regions are off-limits **before** it runs.
2. For manually written InCore kernels, the programmer needs an explicit way to express "this region of my local SRAM is reserved for TPUSH/TPOP ring buffer."
3. For compiler-generated kernels (`auto_incore` / `ExpandMixedKernel`), the pass emits the same declaration into the generated IR, so the rest of the pipeline treats it uniformly.

#### Proposed Syntax

**pypto DSL (Python frontend)**:

```python
@pl.incore
def my_vector_kernel(...):
    # Declare a reserved buffer region in this function's local SRAM.
    # The allocator will not place any other symbols in [base, base + size).
    # 'base' can be:
    #   - pl.AUTO: compiler picks the address (typical for auto_incore)
    #   - an integer literal: programmer specifies exact address (manual kernels)
    pipe_buf = pl.reserve_buffer(
        name="c2v_slot_buffer",
        size=SLOT_NUM * SLOT_SIZE,       # total bytes to reserve
        base=pl.AUTO,                    # or e.g. 0x1000 for manual kernels
    )

    # pipe_buf.base is a compile-time constant (resolved by allocator if AUTO)
    # pipe_buf.size is the declared size
    # Pass pipe_buf.base explicitly to the initialization function:
    aiv_initialize_pipe(DIR_C2V, SLOT_SIZE, gm_slot_buffer,
                        c2v_consumer_buf=pipe_buf.base,
                        v2c_consumer_buf=0)

    for ...:
        tile = pl.tpop_from_aic(aiv_idx=0)    # zero-copy from pipe_buf on A5
        # ... compute on tile ...
```

**pypto DSL (producer side)**:

```python
@pl.incore
def my_cube_kernel(...):
    # Producer imports the consumer's reserved buffer address.
    # 'peer_func' identifies the paired consumer InCore function.
    # The compiler resolves peer_buf.base to the consumer's CONSUMER_BUFFER_BASE value.
    peer_buf = pl.import_peer_buffer(
        name="c2v_slot_buffer",
        peer_func=my_vector_kernel,       # reference to paired consumer function
    )

    # Pass peer_buf.base explicitly to the initialization function:
    aic_initialize_pipe(DIR_C2V, SLOT_SIZE, gm_slot_buffer,
                        c2v_consumer_buf=peer_buf.base,
                        v2c_consumer_buf=0)

    for ...:
        pl.tpush_to_aiv(tile, aiv_idx=0)    # DMA to peer_buf.base on A5
```

#### IR Representation

At the IR level, `pl.reserve_buffer` lowers to a **`ReserveBuffer`** node attached to the InCore function:

```
// IR after lowering (conceptual):
func @my_vector_kernel(...) {
    %pipe_buf = reserve_buffer {
        name = "c2v_slot_buffer",
        size = 4096,                    // SLOT_NUM * SLOT_SIZE
        base = auto,                    // or literal 0x1000
        memory_space = "UB"             // inferred from core type (UB for Vector, L1 for Cube)
    }
    ...
}
```

And `pl.import_peer_buffer` lowers to an **`ImportPeerBuffer`** node:

```
func @my_cube_kernel(...) {
    %peer_buf = import_peer_buffer {
        name = "c2v_slot_buffer",
        peer_func = @my_vector_kernel
    }
    ...
}
```

#### Allocator Handling

The `AllocateMemoryAddr` pass processes `ReserveBuffer` nodes as follows:

| `base` value | Allocator behavior |
|---|---|
| `auto` | Allocator picks an address that does not conflict with other allocations. Writes the chosen address back into `%pipe_buf.base`. |
| literal (e.g. `0x1000`) | Allocator marks `[0x1000, 0x1000 + size)` as occupied. Fails with an error if the region overlaps with prior allocations. |

After the allocator runs, `%pipe_buf.base` is a resolved compile-time constant in both the consumer and producer functions. The `ImportPeerBuffer` node resolves to the same literal value as the paired `ReserveBuffer` node.

#### `ExpandMixedKernel` Auto-Generation

When `ExpandMixedKernel` splits a mixed InCore function, it **automatically emits** `ReserveBuffer` and `ImportPeerBuffer` nodes:

```
ExpandMixedKernel pass:

    Input: mixed InCore function with tpush_*/tpop_* ops

    Output:
    ┌───────────────────────────────────┐
    │ Consumer function (e.g. Vector):  │
    │   %buf = reserve_buffer {         │
    │     name = "c2v_slot_buffer",     │
    │     size = SLOT_NUM * SLOT_SIZE,  │
    │     base = auto,                  │  ← allocator will resolve
    │     memory_space = "UB"           │
    │   }                               │
    │   ...tpop_from_aic uses %buf...   │
    └───────────────────────────────────┘

    ┌───────────────────────────────────┐
    │ Producer function (e.g. Cube):    │
    │   %peer = import_peer_buffer {    │
    │     name = "c2v_slot_buffer",     │
    │     peer_func = @consumer_func    │
    │   }                               │
    │   ...tpush_to_aiv uses %peer...│
    └───────────────────────────────────┘
```

The programmer never writes `reserve_buffer` or `import_peer_buffer` when using `auto_incore` — the compiler generates them. These constructs are only explicitly written in **manually authored** InCore kernels.

#### Summary of Grammar Elements

| DSL Construct | Purpose | Who writes it |
|---|---|---|
| `pl.reserve_buffer(name, size, base)` | Declare a reserved SRAM region in the current InCore function for ring buffer use | **Compiler** (auto_incore) or **programmer** (manual kernel) |
| `pl.import_peer_buffer(name, peer_func)` | Import the resolved base address of a peer function's reserved buffer | **Compiler** (auto_incore) or **programmer** (manual kernel) |
| `pl.AUTO` | Sentinel value requesting compiler to auto-assign the base address | Used in `base=` parameter |

These constructs form the **contract** between the InCore kernel program and the address allocator: "this region of my SRAM is spoken for — do not allocate into it."

### Compiler Toolchain Implications

The `CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE` design and the `reserve_buffer` / `import_peer_buffer` grammar require the pypto compiler (or downstream toolchain) to support the following:

1. **DSL frontend — new constructs**: The pypto Python frontend must support `pl.reserve_buffer(...)` and `pl.import_peer_buffer(...)`. These lower to `ReserveBuffer` and `ImportPeerBuffer` IR nodes respectively.

2. **`ExpandMixedKernel` pass — auto-generation**: When splitting a mixed InCore function into Cube and Vector sub-functions, the pass must:
   - Identify `tpush_*/tpop_*` operations and their implied directions.
   - Emit `ReserveBuffer` nodes in consumer functions with `base=auto`.
   - Emit `ImportPeerBuffer` nodes in producer functions referencing the consumer.
   - Set `CONSUMER_BUFFER_SIZE = SLOT_NUM * SLOT_SIZE`.
   - **Insert `tfree_to_aic`/`tfree_to_aiv` calls** in consumer kernels at the point where the consumer has finished reading the data from the popped slot. The pass must analyze the data dependency to determine the earliest safe point for the `tfree` — typically after the last read of the tile variable produced by the corresponding `tpop`.

3. **`AllocateMemoryAddr` pass — reservation and resolution**:
   - For `ReserveBuffer` with `base=auto`: pick a non-conflicting address, write it back as a resolved constant.
   - For `ReserveBuffer` with explicit base: validate no overlap, mark as reserved.
   - For `ImportPeerBuffer`: resolve to the same literal as the paired `ReserveBuffer` in the peer function.
   - All other tile/temporary allocations must avoid `[BASE, BASE + SIZE)`.

4. **Cross-function constant propagation**: The resolved `ReserveBuffer.base` value must be propagated to all `ImportPeerBuffer` nodes that reference it. Since both functions exist in the same compilation unit (generated by `ExpandMixedKernel` or co-compiled manual kernels), this is a straightforward symbol resolution step.

5. **Validation**:
   - The declared `size` must not exceed available SRAM.
   - Every `ImportPeerBuffer` must have a matching `ReserveBuffer` in the referenced peer function.
   - On A2A3, `ReserveBuffer` / `ImportPeerBuffer` nodes are not generated (ring buffer is in GM via `GM_SLOT_BUFFER`). If present, the compiler may emit a warning or ignore them.
   - **Every `tpop_from_aic`/`tpop_from_aiv` must be paired with a corresponding `tfree_to_aic`/`tfree_to_aiv`** in the same kernel. The compiler should verify that no execution path consumes a `tpop` without a matching `tfree` before the ring buffer wraps.

6. **Platform-conditional code generation**: The compiler emits different initialization and data transfer code paths based on `PLATFORM_ID`. On A2A3, the `GM_SLOT_BUFFER` argument path is used; on A5, the `CONSUMER_BUFFER_BASE` constant path is used.

*(To be expanded with future instructions.)*
