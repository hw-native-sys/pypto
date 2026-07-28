# 精度定位：选择性张量 Dump

> **状态：** 草稿骨架。捕获并对比某个特定张量的设备值。

## 症状

你怀疑某个中间张量出错，想拿到它的设备实际值，又不想 dump 全部张量。

## 工具

- **前端打标：** `pl.dump_tag(...)` —— 标记一个张量用于选择性 dump。
- **runtime-DFX 选择性 dump** —— 运行时仅写出被打标的张量。
- **L2 swimlane 双跑（double-run）** —— 运行两次以捕获板上值用于对比。

## 步骤

_TODO：_

1. 用 `pl.dump_tag` 给可疑张量打标。
2. 打开 runtime-DFX 选择性 dump 标志。
3. 运行（L2 swimlane 需双跑）并收集 dump。
4. 与黄金值对比。

## 如何读输出

_TODO —— dump 文件格式、命名及对比方法。_

## 参见

- 开发者参考：[`dev/03-runtime-dfx.md`](../../../dev/03-runtime-dfx.md)（选择性张量 dump、L2 swimlane 双跑）
- [DFX 功能](../dfx/00-flag-matrix.md)
