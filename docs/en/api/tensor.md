# `pl.tensor`

Tensor-level operators — they take and return `pl.Tensor` values, so they belong in
orchestration or wherever a whole tensor is the unit of work. See
[Choosing a namespace](../user/ops/00-dispatch.md).

Use `pl.create_tensor(...)` or `pl.tensor.create(...)` to allocate a tensor.
Both accept the same arguments. Replace older calls to `pl.tensor.create_tensor(...)` with either form.

::: pypto.language.tensor
    options:
      show_root_heading: false
      filters: ["!^_"]
