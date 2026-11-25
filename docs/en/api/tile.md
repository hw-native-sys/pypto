# `pl.tile`

Tile-level operators — the ones that take and return `pl.Tile` values inside an InCore
function. See [Choosing a namespace](../user/ops/00-dispatch.md) for when to reach for these
rather than their `pl.*` or `pl.tensor.*` counterparts.

Replace `pl.tile.create_tile(...)` with `pl.create_tile(...)` or `pl.tile.create(...)`.
Both take the same arguments as the old spelling.

::: pypto.language.tile
    options:
      show_root_heading: false
      filters: ["!^_"]
