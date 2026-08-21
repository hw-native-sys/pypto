# `pl`

直接挂在 `pl` 上的名字 —— 装饰器、类型、控制流构造，以及按类型分派的算子包装。这里的算子按你传进去的东西分派：两个 tile 的 `pl.add` 就是 `pl.tile.add`，两个张量的就是 `pl.tensor.add`。见[选择命名空间](../user/ops/00-dispatch.md)。

::: pypto.language
    options:
      show_root_heading: false
      show_submodules: false
      filters:
        - "!^_"
        - "!^(parser|optimizations|adir|array|prefetch|tile|system|tensor)$"
