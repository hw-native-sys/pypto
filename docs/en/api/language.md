# `pl`

The names you reach directly on `pl` — decorators, types, control-flow constructs, and the
type-dispatched operator wrappers. An operator here dispatches on what you hand it: `pl.add`
of two tiles is `pl.tile.add`, of two tensors `pl.tensor.add`. See
[Choosing a namespace](../user/ops/00-dispatch.md).

::: pypto.language
    options:
      show_root_heading: false
      show_submodules: false
      filters:
        - "!^_"
        - "!^(parser|optimizations|adir|array|prefetch|tile|system|tensor)$"
