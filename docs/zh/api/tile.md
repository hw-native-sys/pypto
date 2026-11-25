# `pl.tile`

Tile 级算子 —— 在 InCore 函数内接受并返回 `pl.Tile` 值的那一族。什么时候该用它们而不是 `pl.*` 或 `pl.tensor.*` 的对应物，见[选择命名空间](../user/ops/00-dispatch.md)。

已有代码可将 `pl.tile.create_tile(...)` 替换为 `pl.create_tile(...)` 或 `pl.tile.create(...)`，参数相同。

::: pypto.language.tile
    options:
      show_root_heading: false
      filters: ["!^_"]
