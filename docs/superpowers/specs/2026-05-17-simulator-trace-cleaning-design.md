# Simulator Trace Cleaning Tool — Design

- **Date**: 2026-05-17
- **Status**: Approved (brainstorming → ready for implementation plan)
- **Topic**: Convert the simulator's `visualize_data.bin` dump into a clean, Chrome-viewable
  AI-core pipeline trace.

## Problem

The operator simulator emits two profiling artifacts per kernel run, under
`.../OPPROF_*/simulator/`:

1. `trace.json` — a Chrome Trace Event JSON.
2. `visualize_data.bin` — a binary container consumed by MindStudio Insight.

Opening `trace.json` in the Perfetto UI is hard to read: the timeline is dominated
by `SET_FLAG` / `WAIT_FLAG` synchronization slices and by scalar address-arithmetic
instructions, which bury the actual AI-core pipeline (load → compute → store).
MindStudio Insight presents the per-core pipeline clearly, but its format is
specialized for that tool.

**Goal**: a tool that reads the internal `visualize_data.bin` dump and produces a
*simpler*, easy-to-read trace that:

1. is viewable in Chrome (`chrome://tracing`) and the Perfetto UI, and
2. is generated from the `visualize_data.bin` dump.

## Background: the `visualize_data.bin` format

Reverse-engineered from the open-source MindStudio Insight parser
(`gitcode.com/Ascend/msinsight`, `SourceFileParser` / `DataBlock.h`). The file is a
flat sequence of length-prefixed blocks. Each block has a 12-byte header,
4-byte-aligned:

| Offset | Size | Field | Meaning |
| ------ | ---- | ----- | ------- |
| 0 | `uint64` LE | `contentSize` | payload length, including trailing padding |
| 8 | `uint8` | `type` | `DataTypeEnum` |
| 9 | `uint8` | `paddingLength` | 0–3 zero bytes appended for 4-byte alignment |
| 10 | `uint8` | `instrVersion` | only meaningful for `API_INSTR`: `0` = new, `0x5a` = old |
| 11 | `uint8` | `reserve` | always `0x5a` (used as the binary-format magic) |

Payload follows the header: `contentSize` bytes, the real content being the first
`contentSize - paddingLength` bytes.

`DataTypeEnum` (relevant subset):

| `type` | Name | Payload |
| ------ | ---- | ------- |
| 1 | `SOURCE` | source text — body is prefixed by a fixed 4096-byte file-path field |
| **2** | `TRACE` | Chrome Trace Event JSON — identical to the sibling `trace.json` |
| 3 | `API_FILE` | API / source-line mapping JSON |
| **4** | `API_INSTR` | per-instruction stats JSON (`Cores` / `Instructions` / `Instructions Dtype`) |
| 5–14 | `DETAILS_*`, `DISPLAY_CACHE` | base info, compute-load, memory, roofline, cacheline |

Each block payload is itself plain JSON. The tool consumes `TRACE` (type 2) and
`API_INSTR` (type 4); all other block types are skipped for forward-compatibility.

The `API_INSTR` block stores each per-instruction metric as a 3-element array
indexed by the `Cores` list (`["core0.cubecore0","core0.veccore0","core0.veccore1"]`).

## Scope and non-goals

- **In scope**: one invocation processes one `visualize_data.bin` (or one `OPPROF_*`
  directory). Output files are written next to the input.
- **Non-goals**: no tree-walking / batch mode; no Perfetto-native protobuf output;
  API_INSTR metrics are *not* merged into the trace (they go to a sidecar file).

## Approach decision

Output format: **Chrome Trace Event JSON** (rejected alternative: Perfetto-native
protobuf).

- Chrome JSON opens in **both** `chrome://tracing` and the Perfetto UI; protobuf is
  Perfetto-only, which conflicts with the "viewable in Chrome" requirement.
- Chrome JSON needs **no third-party dependency** (stdlib `struct` + `json`).
- Protobuf's only real advantage — native counter tracks — does not apply, because
  metrics live in a separate sidecar file.

## Architecture

A single standalone CLI script, stdlib-only:

- **New file**: `python/pypto/tools/clean_sim_trace.py`
- **New package marker**: `python/pypto/tools/__init__.py`
- Carries the standard CANN Open Software License header.

### CLI

```bash
python -m pypto.tools.clean_sim_trace <path> [-o OUTPUT_DIR] [--keep-scalar] [--raw-metrics]
```

`<path>` accepts either:

- a `visualize_data.bin` file directly, or
- an `OPPROF_*` directory — the script locates `simulator/visualize_data.bin` inside it.

### Outputs

Written next to the input (the `simulator/` directory), or to `-o` if given. The
inputs are never modified — the tool is read-only on `trace.json` /
`visualize_data.bin` and writes only new files.

| File | Content |
| ---- | ------- |
| `trace.clean.json` | Rebuilt Chrome Trace Event JSON — opens in `chrome://tracing` and Perfetto UI |
| `instr_metrics.json` | `API_INSTR` metrics, de-framed to a standalone sidecar |

### Components

The script decomposes into three units with clear boundaries:

1. **Block parser** — `iter_blocks(data: bytes) -> Iterator[tuple[int, bytes]]`.
   Decodes the length-prefixed block container. Pure function of the byte buffer.
2. **Trace rebuilder** — takes the `TRACE` block's `traceEvents`, applies the six
   rebuild rules below, returns the cleaned Chrome-trace dict.
3. **Metrics reshaper** — takes the `API_INSTR` block, returns the per-core sidecar
   structure.

## Block parser

```python
import struct

_HEADER = struct.Struct("<QBBBB")  # contentSize:u64, type:u8, padding:u8, instrVersion:u8, reserve:u8
_MAGIC = 0x5A          # reserve byte
_TYPE_SOURCE, _TYPE_TRACE, _TYPE_API_INSTR = 1, 2, 4
_SOURCE_PATH_LEN = 4096

def iter_blocks(data: bytes):
    """Yield (block_type, payload_bytes) for each block in a visualize_data.bin."""
    off = 0
    while off + _HEADER.size <= len(data):
        size, btype, padding, _instr_ver, reserve = _HEADER.unpack_from(data, off)
        body = off + _HEADER.size
        if reserve != _MAGIC or size > len(data) - body:
            raise ValueError(f"corrupt block at offset {off}: size={size}, reserve={reserve:#x}")
        payload = data[body : body + size]
        if btype == _TYPE_SOURCE:                 # SOURCE blocks prefix a 4096-byte path
            payload = payload[_SOURCE_PATH_LEN:]
        yield btype, payload[: len(payload) - padding]   # strip trailing alignment padding
        off = body + size
```

## Trace rebuild rules

The raw trace uses `pid` = core, `tid` = pipeline lane. Six rules transform it.

### Rule 1 — Lane selection (which `tid`s survive)

| Lane | Action | Why |
| ---- | ------ | --- |
| `MTE2`, `MTE3`, `MTE1`, `VECTOR`, `CUBE`, `FIXPIPE` | keep | the real AI-core pipeline |
| `CACHEMISS` | drop | instruction-cache-miss markers (`thread_state_runnable`) — pure noise |
| `FLOWCTRL`, `ALL` | drop | branch bookkeeping + a kernel-wide `BAR` |
| `SCALAR` | drop by default; `--keep-scalar` restores it | mostly address-arithmetic setup, the dominant clutter; a flag rather than a hard removal because it can carry real scalar stalls |

### Rule 2 — Event filtering on kept lanes

Keep `X` (complete) instruction events. Drop every `SET_FLAG` / `WAIT_FLAG` / `BAR`
slice (the `B`/`E` pairs and any `X`-phase ones).

### Rule 3 — Lane ordering and naming

Emit `process_name` / `process_sort_index` / `thread_name` / `thread_sort_index`
metadata events so the viewer renders a fixed dataflow order:

- Cores: `cubecore0`, `veccore0`, `veccore1` (sort index 0, 1, 2).
- Lanes within a core: `MTE2 · load (GM→UB)` → `MTE1` → `CUBE · matmul` /
  `VECTOR · compute` → `FIXPIPE` → `MTE3 · store (UB→GM)` → `SCALAR` (last, if kept).
- The raw pipe name is kept as the `thread_name` prefix so lanes stay greppable.

### Rule 4 — Sync as dependency arrows

Each `SET_FLAG` (producer pipe) → `WAIT_FLAG` (consumer pipe) pair — matched by
`pid` + `FLAGID` + the `PIPE` / `TRIGGERPIPE` fields parsed from `args.detail` —
becomes one thin flow arrow. Because the flag slices are dropped (Rule 2), the arrow
is **re-anchored** to real instructions:

- arrow **start** = the last surviving instruction slice on the producer pipe at or
  before the `SET_FLAG` timestamp;
- arrow **end** = the first surviving instruction slice on the consumer pipe at or
  after the `WAIT_FLAG` resolves.

Emit a flow event pair (`ph:"s"` and `ph:"f"` with `bp:"e"`) sharing an `id`, with
`cat:"sync"`. A flag pair that cannot be matched to instructions on both sides is
skipped (counted, reported in a one-line summary), never fatal.

### Rule 5 — Coloring

Recolor slices by pipeline via the Chrome-trace `cname` field so each lane is
visually uniform. `args.detail` and `args.pc_addr` are preserved for click-through
detail.

### Rule 6 — Timestamps

Kept verbatim (nanoseconds). The cleaned trace lines up 1:1 with the raw
`trace.json` for cross-checking.

### Before → after example

```jsonc
// BEFORE — raw trace.json (one veccore0 fragment)
{"name":"thread_state_runnable","ph":"X","pid":"core0.veccore0","tid":"CACHEMISS","ts":1.362} // dropped
{"name":"MOV_XD_IMM","ph":"X","pid":"core0.veccore0","tid":"SCALAR","ts":1.557}               // dropped (default)
{"name":"SET_FLAG","ph":"B","pid":"core0.veccore0","tid":"MTE2","ts":3.056,
  "args":{"detail":"PIPE:MTE2,TRIGGERPIPE:VEC,FLAGID:0,"}}                                     // dropped → arrow
{"name":"WAIT_FLAG","ph":"B","pid":"core0.veccore0","tid":"VECTOR","ts":1.769}                 // dropped → arrow
{"name":"MOV_SRC_TO_DST_ALIGN","ph":"X","pid":"core0.veccore0","tid":"MTE2","ts":1.761,"dur":0.416} // kept
{"name":"VADD","ph":"X","pid":"core0.veccore0","tid":"VECTOR","ts":3.058,"dur":0.084}          // kept

// AFTER — trace.clean.json
{"name":"thread_name","ph":"M","pid":"core0.veccore0","tid":"MTE2","args":{"name":"MTE2 · load (GM→UB)"}}
{"name":"thread_sort_index","ph":"M","pid":"core0.veccore0","tid":"MTE2","args":{"sort_index":0}}
{"name":"MOV_SRC_TO_DST_ALIGN","ph":"X","pid":"core0.veccore0","tid":"MTE2","ts":1.761,"dur":0.416,
  "cname":"thread_state_iowait","args":{...}}
{"name":"VADD","ph":"X","pid":"core0.veccore0","tid":"VECTOR","ts":3.058,"dur":0.084,
  "cname":"good","args":{...}}
{"ph":"s","id":42,"cat":"sync","name":"MTE2→VEC flag0","pid":"core0.veccore0","tid":"MTE2","ts":2.177}
{"ph":"f","id":42,"cat":"sync","bp":"e","name":"MTE2→VEC flag0","pid":"core0.veccore0","tid":"VECTOR","ts":3.058}
```

## Metrics sidecar

`instr_metrics.json`. The raw `API_INSTR` per-core 3-element arrays are awkward to
read, so the sidecar is lightly reshaped to one record list per core, with each
per-core array element flattened to a scalar:

```jsonc
{
  "cores": ["core0.cubecore0", "core0.veccore0", "core0.veccore1"],
  "instructions": {
    "core0.veccore0": [
      {"address": "0x10d11050", "source": "LD_XD_XN_IMM ...", "pipe": "SCALAR",
       "cycles": 0, "instructions_executed": 0, "gpr_count": 0,
       "process_bytes": -1, "ub_read_conflict": -1, "ub_write_conflict": -1,
       "vector_utilization_pct": -1.0}
    ]
  },
  "column_types": { ... }   // the "Instructions Dtype" map, passed through verbatim
}
```

A `--raw-metrics` flag dumps the `API_INSTR` block byte-for-byte instead, for anyone
who needs the unmodified form.

## Error handling

Follows the project's blocking / non-blocking convention.

| Situation | Behavior |
| --------- | -------- |
| Path missing / not a file-or-`OPPROF`-dir / no `visualize_data.bin` found | Clear `error:` message, exit 1 (blocking) |
| Bad magic, `contentSize` overflows file | `ValueError` with offset, exit 1 (blocking — file corrupt) |
| No `TRACE` block | Exit 1 — cannot build a trace without it (blocking) |
| No `API_INSTR` block | Warn, skip `instr_metrics.json`, still emit `trace.clean.json` (non-blocking) |
| A flag pair cannot be re-anchored | Skip that arrow, count it, print a one-line summary (non-blocking) |
| Unknown block type | Skipped silently (forward-compatibility) |

## Testing

`tests/ut/tools/test_clean_sim_trace.py` (new `tests/ut/tools/` directory), pytest
style, `assert`-based, ending with the required `pytest.main` block. All fixtures
are synthetic and built in-process — no reliance on `build_output/` — so tests are
hermetic. A helper builds a minimal `visualize_data.bin` in memory
(`_HEADER.pack(...)` + JSON payloads).

| Test | Verifies |
| ---- | -------- |
| `test_iter_blocks_roundtrip` | Two synthetic blocks (TRACE + API_INSTR) decode to exact payloads; padding stripped; magic checked |
| `test_iter_blocks_rejects_corrupt` | Bad `reserve` byte / oversize `contentSize` raise `ValueError` with offset |
| `test_source_block_path_skipped` | A `SOURCE` block's 4096-byte path prefix is dropped |
| `test_rebuild_drops_noise` | `CACHEMISS`/`FLOWCTRL`/`ALL` lanes and `SET_FLAG`/`WAIT_FLAG`/`BAR` slices are gone; `MTE2`/`VECTOR`/`MTE3` `X` events survive |
| `test_rebuild_scalar_flag` | `SCALAR` absent by default, present when `keep_scalar=True` |
| `test_rebuild_lane_order` | `process_*`/`thread_*` metadata emitted with the dataflow `sort_index` order |
| `test_sync_arrows_reanchored` | A SET_FLAG/WAIT_FLAG pair produces one `s`/`f` flow pair anchored to real producer/consumer slices, sharing an `id` |
| `test_unmatchable_flag_skipped` | A flag with no consumer instruction is dropped, not fatal |
| `test_metrics_sidecar_reshape` | 3-element per-core arrays flatten to per-core scalar records keyed by core name |
| `test_missing_api_instr_block` | Trace still emitted, sidecar skipped, warning printed |

## Documentation

A new `docs/en/dev/04-simulator-trace-cleaning.md` (and the `docs/zh-cn/` mirror,
per the documentation rule). The `00`–`03` slots are taken; `04` is the next free
number.
It covers the `visualize_data.bin` block format, the six rebuild rules, the CLI, and
the output files. Kept under the 500-line documentation limit.

## Implementation order

1. `python/pypto/tools/__init__.py` + `clean_sim_trace.py` skeleton with the CANN
   license header and `argparse` CLI.
2. Block parser (`iter_blocks`) — depends on nothing.
3. Trace rebuilder (Rules 1–6) — depends on the block parser.
4. Metrics reshaper — depends on the block parser.
5. Wire CLI → parser → rebuilder + reshaper → file writers; add error handling.
6. Tests in `tests/ut/tools/test_clean_sim_trace.py`.
7. Documentation in `docs/en/dev/` and `docs/zh-cn/`.
