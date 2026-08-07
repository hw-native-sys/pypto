# Memory Map

`memory_map` renders a pass dump into an interactive HTML map of on-chip memory:
**address across, lifetime down**. Every tile becomes a rectangle spanning the
bytes its MemRef owns and the statements over which it is live, so a reuse
decision is one glance instead of a cross-reference between two tables.

It replaces the `report/memory_after_<Pass>.txt` file the compiler used to
write — that report and its `MemoryReport` generator are gone.

## Usage

```bash
python -m pypto.tools.memory_map <path> [-o OUTPUT] [-p PASS_NAME] [-b BACKEND]
```

`<path>` is one of:

| Given | Resolved to |
| ----- | ----------- |
| `.../passes_dump/32_after_AllocateMemoryAddr.py` | that file |
| `.../passes_dump/` | the highest-numbered `*_after_<PASS_NAME>.py` in it |
| `build_output/<case>/` | the same, under its `passes_dump/` |

`-p/--pass-name` selects the pass when `<path>` is a directory (default
`AllocateMemoryAddr`). Output defaults to `<dump>.memory_map.html`.

### Producing the dump

Compile with pass dumping enabled — the dumps land in
`build_output/<case>/passes_dump/`:

```bash
pytest tests/st/<case>.py --dump-passes --save-kernels
```

Pick a dump at or after `AllocateMemoryAddr`. That is the pass that assigns
addresses: before it every MemRef still sits at offset 0, so there is no address
axis to draw and the tool refuses the dump with that explanation. Later dumps
(`33_after_FoldNoOpReshape` onwards) map fine and show the same layout.

### Where the capacities come from

Panel scale is a memory space's capacity, which the dump does not record. The
tool reads it off the backend's own SoC description — walking
`Backend.soc` down to each `Core`'s `Mem` entries — so a backend that carries a
space others do not (Ascend950 has `Bias` / `LeftScale` / `RightScale`) still gets a real capacity instead of
falling back to its own high-water mark and rendering that panel permanently
full.

Which backend is decided in this order:

1. `-b/--backend` — an explicit `BackendType` member name (`Ascend910B`, `Ascend950`).
2. The case's `ptoas/*.pto` output, whose `pto.target_arch` attribute is matched
   against each backend's own `get_pto_target_arch()`. The mapping lives with
   the backends rather than being duplicated here, so a new backend needs no
   change to this tool.
3. Otherwise `Ascend910B` is assumed, and the page labels the chip *assumed* so
   the panel scales are not mistaken for measured ones.

Capacities are read from the **current** source tree, not from whatever the dump
was compiled against. Re-mapping an old `build_output/` after a capacity change
therefore scales its panels by the new value.

## Reading the page

Each compute function (`AIC` / `AIV` / `InCore`) is a collapsible card; the
first is open. `Group` and `Orchestration` functions own no tile memory and are
omitted. The card header carries one pill per memory space with its high-water
mark, capacity and usage — the pill turns red at ≥ 95 %.

| Axis / mark | Meaning |
| ----------- | ------- |
| y, downward | the dump's own line number, which is statement order |
| x | byte address inside one space, on one scale shared by every panel |
| a box | one tile, live over `[first line, last line]` mentioning its name |
| dashed box | a view: a sub-range of a larger allocation on the same base |
| red box | two **different** bases overlap in address *and* lifetime |
| dashed vertical rule | the space's high-water mark |

Each space is a band: a solid divider on its left edge and a faint wash of the
space's own colour behind the lane, with a matching accent bar on its column
header. Both are drawn as insets, so neither consumes lane width.

### One shared byte scale

A byte is the same number of pixels in every panel. Each lane gets an **absolute
pixel width** equal to its span times one px-per-byte factor, so `Mat` at 512 KB
is eight times as wide as `Left` at 64 KB. Box widths therefore mean the same
thing wherever they sit, and a tile in `Acc` can be compared against one in `Mat`
by eye.

Absolute widths — rather than sharing out `fr` units — are what keep the scale
stable when the IR source is shown: the source column is as wide as the longest
line, and with `fr` lanes it would squeeze every panel down to nothing. Widening
the grid and scrolling the pane is the correct outcome instead.

The factor is chosen so the lanes fill the pane, then raised if that would put
the narrowest lane under 72 px. Raising it keeps the factor uniform — the map
simply becomes wider than the pane and scrolls. Boxes also draw at a 2 px floor
instead of vanishing, and the separator between adjacent boxes is painted
*outside* the box, so a box is never narrower than the bytes it owns.

### Zooming

`−` / `+` in the toolbar scale the byte factor by 1.6× per click, from 0.2× to
64×; the middle button shows the current level and resets it. `Ctrl`/`Cmd` +
scroll over the plot does the same and keeps the byte under the pointer in
place, so a tile stays put while it grows. Plain scrolling still pans.

Zoom multiplies the one shared factor, so every lane grows together and box
widths stay comparable across panels — it magnifies the map rather than
re-fitting it. Use it to inspect a tile that is small relative to its space:
a 2 KB buffer in a 512 KB `Mat` panel is legible at 8×.

Two spans are available:

| `x axis:` | Each panel spans | Reads |
| --------- | ---------------- | ----- |
| `limit` (default) | `0 →` the space's capacity | filled fraction = usage; `Right` at its 64 KB cap is obvious |
| `used` | `0 →` the space's high-water mark | the allocated bytes get the whole width, so small tiles are legible |

`used` can make the map much wider than `limit`: when one space peaks at 4 KB
and another at 68 KB, a shared scale that makes the 4 KB lane legible stretches
the 68 KB lane accordingly. Switching spaces off is the counterweight.

### Showing and hiding spaces

The `show` row in the toolbar carries one toggle per memory space. Switching a
space off drops its lane, its column header and its boxes, and the freed width
is redistributed at the same shared factor — useful when only one space is under
investigation, and the way to keep the `used` scale manageable. The last visible
space cannot be switched off. The setting is per function card.

### Selecting a tile

**Click a box** (or focus it and press Enter) to pin it:

- the box gets a halo, and a `pinned <name> ×` chip appears in the toolbar
- its live lines stay highlighted instead of following the mouse
- every mention of the tile's name — and of each alias merged into it — is
  marked in the IR source, the way a search hit reads

Pinning opens the IR source if it was collapsed, since locating the tile there
is the point. Hovering another box still previews its tooltip without disturbing
the pin. Click the pinned box again, press `Esc`, or use the chip's `×`
to clear it. Each function card holds its own selection.

### IR source

**Show IR source** expands the left gutter into the function's IR, aligned line
for line with the plot: a box's top edge sits on the line that defines its tile.
Hovering a box highlights exactly the lines it is live across.

Both states pin the left column so it stays put while the panels scroll:
collapsed it is a thin line-number gutter, expanded it is a source pane of a
width you choose.

**Drag the divider** between the source and the map to trade width between
them. Double-click it to return to the default, or focus it and use the arrow
keys (`Shift` for a bigger step); the width is clamped to 140–2400 px and is
remembered per function card. Long IR lines scroll horizontally inside the pane,
with the line numbers pinned to its left edge.

Resizing only rewrites the grid template — box geometry is a percentage of its
lane and lane widths are absolute — so neither the map nor the shared scale
moves, and dragging stays smooth on a function with thousands of source rows.
The plot pane is capped at 78 % of the viewport height so both scrollbars stay
reachable on a long function.

### Alias merging

An SSA phi/yield chain rebinds one slot under a new name every iteration.
Drawing each rebind separately buries the picture under a stack of identical
rectangles, so tiles sharing `(space, base, offset, size)` whose live ranges
**touch** are merged into one box: the label is the first name and `+N` counts
the rest, with the full list in the tooltip.

Tiles sharing a slot whose live ranges are **disjoint** are never merged — that
is memory reuse, and keeping them apart is the point of the view.

### Live ranges are lexical

A tile's range runs from the first to the last dump line mentioning its name.
That is a lexical approximation, not a dataflow liveness analysis: a name
rebound in both arms of an `if` covers the whole `if`, and a value live across a
loop back-edge shows only its textual extent.

## Related

- `docs/en/dev/passes/35-allocate_memory_addr.md` — the pass whose output is mapped
- `docs/en/dev/passes/34-memory_reuse.md` — the pass that decides which lifetimes may share a slot
- `docs/en/dev/passes/00-pass_manager.md` — `ReportInstrument`, which now only names the artifact directory
- `docs/en/dev/04-simulator-trace-cleaning.md` — the other post-hoc analysis tool in `pypto.tools`
