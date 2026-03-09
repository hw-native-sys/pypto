# Buffer Management

## Overview

The ring buffer backing memory for TPUSH/TPOP differs between platforms. On A2/A3, it resides in Global Memory (GM) and is straightforward. On A5, it resides in the consumer's on-chip SRAM, requiring cross-core address resolution.

See [TPUSH/TPOP Instructions](01-tpush_tpop.md) for the instruction specification.

## A2/A3: Global Memory Layout

On A2/A3, the orchestration allocates a single `GM_SLOT_BUFFER` in GM and passes it to both InCore functions as an INOUT argument.

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

## A5: Cross-Core Address Problem

On A5, the ring buffer resides in the **consumer's local SRAM** (UB or L1). This creates a cross-core visibility problem: the producer needs the consumer's local SRAM address to DMA into, but that address belongs to another core's address space.

```text
Cube InCore function:                    Vector InCore function:
┌─────────────────────┐                 ┌─────────────────────┐
│  tpush_to_aiv       │   ??? how to   │  consumer_buf =     │
│  DMA to Vector's UB │ ──────────────▶ │  UB[BASE..BASE+SIZE]│
│  at what address?   │   get address?  │  // local segment   │
└─────────────────────┘                 └─────────────────────┘
```

### Solution: `CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE`

Two **constant symbols** are attached to each InCore function participating in TPUSH/TPOP:

| Symbol | Type | Description |
| ------ | ---- | ----------- |
| `{DIR}_CONSUMER_BUFFER_BASE` | `uint32_t` | Base address of ring buffer in consumer's SRAM (`{DIR}` is `C2V` or `V2C`) |
| `CONSUMER_BUFFER_SIZE` | `uint32_t` | Total reserved size (`SLOT_NUM * SLOT_SIZE`) |

**Key properties:**

1. **Per-function, per-direction** — each consumer function has its own base address
2. **Cross-function visible** — producer imports consumer's base as a compile-time constant
3. **Allocator-reserved** — the memory allocator marks `[BASE, BASE+SIZE)` as occupied

**Value origin:**

| Kernel Origin | How Values Are Set |
| ------------- | ------------------ |
| Auto-generated kernels | Compiler generates values, assigns non-overlapping SRAM region |
| Manually written | Programmer declares values, must avoid conflicts |

The resolved `CONSUMER_BUFFER_BASE` values are passed as **explicit arguments** (`C2V_CONSUMER_BUF`, `V2C_CONSUMER_BUF`) to the initialization functions, avoiding special compiler requirements for implicit constant lookups.

### Example: Unidirectional C2V on A5

```text
Vector (consumer):
    C2V_CONSUMER_BUFFER_BASE = 0x1000
    CONSUMER_BUFFER_SIZE = 8 * SLOT_SIZE

    UB layout: [normal tiles] [RESERVED: ring buffer at 0x1000] [normal tiles]
                               ◄─── allocator avoids ───►

Cube (producer):
    C2V_CONSUMER_BUFFER_BASE = 0x1000    // imported from Vector
    // Uses as DMA target in tpush_to_aiv
```

### Example: Bidirectional on A5

Each direction has a different consumer, each with its own reserved segment:

```text
Cube:   V2C_CONSUMER_BUFFER_BASE = 0x2000 (own L1, Cube is consumer)
        C2V_CONSUMER_BUFFER_BASE = 0x1000 (imported from Vector)

Vector: C2V_CONSUMER_BUFFER_BASE = 0x1000 (own UB, Vector is consumer)
        V2C_CONSUMER_BUFFER_BASE = 0x2000 (imported from Cube)
```

### SRAM Layout (A5)

```text
Consumer's SRAM (UB or L1):

┌──────────┬──────────────────────────────┬───────────┐
│ normal   │  CONSUMER_BUFFER segment     │ normal    │
│ tiles    │  [BASE .. BASE+SIZE)         │ tiles     │
│          │  slot[0] .. slot[SLOT_NUM-1] │           │
└──────────┴──────────────────────────────┴───────────┘
             ◄─── allocator avoids ───►
```

## DSL Grammar (Proposed)

> **Note:** The DSL constructs described below (`pl.reserve_buffer`, `pl.import_peer_buffer`, `pl.AUTO`) and the IR nodes (`ReserveBuffer`, `ImportPeerBuffer`) are part of the original ISA design specification. They are **not yet implemented** in the PyPTO codebase and are documented here as a reference for future implementation.

### `pl.reserve_buffer` — Consumer Side

Declares a reserved SRAM region for the ring buffer in the current InCore function:

```python
@pl.incore
def my_vector_kernel(...):
    pipe_buf = pl.reserve_buffer(
        name="c2v_slot_buffer",
        size=SLOT_NUM * SLOT_SIZE,
        base=pl.AUTO,                  # or literal e.g. 0x1000
    )

    aiv_initialize_pipe(DIR_C2V, SLOT_SIZE, None,       # GM_SLOT_BUFFER unused on A5
                        c2v_consumer_buf=pipe_buf.base,
                        v2c_consumer_buf=0)

    for ...:
        tile = pl.tpop_from_aic(aiv_idx=0)    # zero-copy from pipe_buf on A5
        # ... compute on tile ...
```

### `pl.import_peer_buffer` — Producer Side

Imports the consumer's reserved buffer address:

```python
@pl.incore
def my_cube_kernel(...):
    peer_buf = pl.import_peer_buffer(
        name="c2v_slot_buffer",
        peer_func=my_vector_kernel,
    )

    aic_initialize_pipe(DIR_C2V, SLOT_SIZE, None,       # GM_SLOT_BUFFER unused on A5
                        c2v_consumer_buf=peer_buf.base,
                        v2c_consumer_buf=0)

    for ...:
        pl.tpush_to_aiv(tile, aiv_idx=0)    # DMA to peer_buf.base on A5
```

### DSL Summary

| Construct | Purpose | Written By |
| --------- | ------- | ---------- |
| `pl.reserve_buffer(name, size, base)` | Declare reserved SRAM region | Compiler (auto) or programmer (manual) |
| `pl.import_peer_buffer(name, peer_func)` | Import peer's buffer address | Compiler (auto) or programmer (manual) |
| `pl.AUTO` | Request compiler-assigned address | Used in `base=` parameter |

## IR Representation

`pl.reserve_buffer` lowers to a `ReserveBuffer` node:

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

`pl.import_peer_buffer` lowers to an `ImportPeerBuffer` node:

```text
func @my_cube_kernel(...) {
    %peer_buf = import_peer_buffer {
        name = "c2v_slot_buffer",
        peer_func = @my_vector_kernel
    }
    ...
}
```

## Allocator Handling

The memory allocator processes `ReserveBuffer` nodes:

| `base` Value | Behavior |
| ------------ | -------- |
| `auto` | Allocator picks non-conflicting address, writes back as resolved constant |
| Literal (e.g. `0x1000`) | Allocator marks region as occupied, fails on overlap |

After allocation, both `ReserveBuffer.base` and the corresponding `ImportPeerBuffer` resolve to the same literal value.

**Allocator contract:**

1. Read `{DIR}_CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE` from the function's symbol table
2. Mark `[BASE, BASE + SIZE)` as reserved in the SRAM layout
3. Allocate all other symbols (tiles, temporaries, spills) outside this region

## Compiler Requirements

1. **DSL frontend** — support `reserve_buffer` and `import_peer_buffer` constructs
2. **Memory allocator** — reserve `[BASE, BASE+SIZE)`, resolve `auto` addresses, propagate to peer imports
3. **Cross-function constant propagation** — resolved buffer base must propagate to all referencing peer import nodes
4. **Validation** — size must not exceed available SRAM; every peer import must have a matching reservation; on A2A3, these are not needed
5. **Platform-conditional codegen** — emit GM path on A2A3, SRAM path on A5
