# `pl.tensor`

张量级算子 —— 接受并返回 `pl.Tensor` 值，因此属于编排层，或任何以整张张量为工作单位的地方。见[选择命名空间](../user/ops/00-dispatch.md)。

创建张量可用 `pl.create_tensor(...)` 或 `pl.tensor.create(...)`，两者参数相同。已有代码中的 `pl.tensor.create_tensor(...)` 可直接替换为其中任一写法。

::: pypto.language.tensor
    options:
      show_root_heading: false
      filters: ["!^_"]
