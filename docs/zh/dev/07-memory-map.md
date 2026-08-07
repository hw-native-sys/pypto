# 内存地图（Memory Map）

`memory_map` 将 pass dump 渲染成一张交互式的片上内存 HTML 地图：**横轴是地址，
纵轴向下是生命周期**。每个 tile 画成一个矩形，横向覆盖它的 MemRef 所占的字节，
纵向覆盖它存活的语句区间——复用（reuse）决策一眼可见，不必在两张表之间来回对照。

它取代了编译器过去写出的 `report/memory_after_<Pass>.txt`——该报告及其
`MemoryReport` 生成器已被删除。

## 用法

```bash
python -m pypto.tools.memory_map <path> [-o OUTPUT] [-p PASS_NAME] [-b BACKEND]
```

`<path>` 可以是以下三种之一：

| 传入 | 解析为 |
| ---- | ------ |
| `.../passes_dump/32_after_AllocateMemoryAddr.py` | 该文件本身 |
| `.../passes_dump/` | 其中编号最大的 `*_after_<PASS_NAME>.py` |
| `build_output/<case>/` | 同上，在它的 `passes_dump/` 下查找 |

当 `<path>` 是目录时，用 `-p/--pass-name` 选择 pass（默认 `AllocateMemoryAddr`）。
输出默认写到 `<dump>.memory_map.html`。

### 生成 dump

编译时打开 pass dump —— dump 落在 `build_output/<case>/passes_dump/`：

```bash
pytest tests/st/<case>.py --dump-passes --save-kernels
```

要选 `AllocateMemoryAddr` 或更晚的 dump。地址正是由这个 pass 分配的：在它之前所有
MemRef 的 offset 都还是 0，没有地址轴可画，工具会带着这条说明拒绝处理。更晚的 dump
（`33_after_FoldNoOpReshape` 及以后）都能正常绘制，展示的是同一份布局。

### 容量从哪来

panel 的缩放基准是内存空间的容量，而 dump 里并不记录它。工具从 backend 自己的 SoC
描述里读取——沿 `Backend.soc` 一路走到每个 `Core` 的 `Mem` 条目——因此某个 backend
独有的空间（Ascend950 有 `Bias` / `LeftScale` / `RightScale`）也能拿到真实容量，而不会回退成它自身的 high-water
mark、把该 panel 画成永远满的。

用哪个 backend 按以下顺序决定：

1. `-b/--backend` —— 显式给出 `BackendType` 成员名（`Ascend910B`、`Ascend950`）。
2. 该 case 的 `ptoas/*.pto` 产物，用其中的 `pto.target_arch` 属性去匹配各 backend
   自己的 `get_pto_target_arch()`。这份映射保存在 backend 一侧而不是在本工具里复制
   一份，因此新增 backend 无需改动本工具。
3. 否则假定为 `Ascend910B`，并在页面上标注 *assumed*，以免把 panel 缩放当成实测值。

容量读的是**当前**代码树的值，而不是 dump 当初编译时的值。因此在容量变更之后重新绘制
旧的 `build_output/`，panel 会按新值缩放。

## 如何阅读

每个 compute function（`AIC` / `AIV` / `InCore`）是一张可折叠的卡片，第一张默认展开。
`Group` 和 `Orchestration` 函数不持有 tile 内存，会被跳过。卡片标题栏里每个内存空间
一个 pill，显示 high-water mark、容量和占用率——占用率 ≥ 95 % 时 pill 变红。

| 坐标 / 图元 | 含义 |
| ----------- | ---- |
| y 轴，向下 | dump 文件自身的行号，也就是语句顺序 |
| x 轴 | 某个 space 内的字节地址，所有 panel 共用同一比例尺 |
| 方块 | 一个 tile，存活区间为提及其名字的 `[首行, 末行]` |
| 虚线框 | view：同一 base 上更大分配的一个子区间 |
| 红框 | 两个**不同**的 base 在地址和生命周期上同时重叠 |
| 竖虚线 | 该 space 的 high-water mark |

每个内存空间自成一条色带：左边缘一条实线分隔，lane 背景是该空间颜色的一层淡色，列头上
还有一条同色的强调条。三者都用 inset 绘制，都不占用 lane 宽度。

### 统一的字节比例尺

一个字节在每个 panel 里对应相同的像素数。每条 lane 拿到的是**绝对像素宽度**，等于它的
跨度乘以同一个 px/byte 因子，因此 512 KB 的 `Mat` 正好是 64 KB 的 `Left` 的八倍宽。
方块宽度在任何位置含义一致，`Acc` 里的 tile 与 `Mat` 里的 tile 可以直接目测比较。

用绝对宽度而不是分配 `fr` 份额，正是为了在展开 IR 源码时保住比例尺：源码列的宽度等于
最长的一行，若 lane 用 `fr` 去分剩余空间，就会被挤压到几乎为零。正确的结果是整个网格
变宽、面板横向滚动。

因子先按"铺满面板"来取，如果这会让最窄的 lane 低于 72 px 就整体调大。调大保持因子统一
——只是地图比面板更宽、需要滚动而已。方块同样有 2 px 的最小宽度而不会消失，并且相邻方块
之间的间隔画在方块**外侧**，使方块本身永远不会比它占的字节数更窄。

### 缩放

工具栏里的 `−` / `+` 每次点击把字节因子乘/除 1.6，范围 0.2× 到 64×；中间的按钮显示当前
倍率，点击即复位。在图上按 `Ctrl`/`Cmd` + 滚轮效果相同，并且会**保持指针下的那个字节不动**，
所以被查看的 tile 会原地变大而不是滑走。普通滚动仍然是平移。

缩放乘的是那**同一个**共享因子，因此所有 lane 一起放大、跨 panel 的方块宽度依然可比——
它放大的是整张地图，而不是重新适配。用它来看相对所属空间很小的 tile：512 KB 的 `Mat`
panel 里一个 2 KB 的 buffer，放到 8× 就清晰可辨。

跨度有两种取法：

| `x axis:` | 每个 panel 的跨度 | 读法 |
| --------- | ----------------- | ---- |
| `limit`（默认） | `0 →` 该空间容量 | 填充比例即占用率，`Right` 打满 64 KB 一目了然 |
| `used` | `0 →` 该空间的 high-water mark | 已分配的字节占满整个宽度，小 tile 更易辨认 |

`used` 可能让地图比 `limit` 宽得多：当一个空间峰值只有 4 KB、另一个有 68 KB 时，要让
4 KB 那条 lane 可读，统一比例尺就会把 68 KB 那条拉得很长。关掉不关心的空间是对冲手段。

### 显示 / 隐藏内存空间

工具栏里的 `show` 一行为每个内存空间提供一个开关。关掉某个空间会移除它的 lane、列头和
方块，腾出的宽度按同一个因子重新分配给其余空间——只关注某一个空间时很有用，也是把
`used` 比例尺控制在可用范围内的手段。最后一个可见空间不能被关掉。该设置按函数卡片独立
保存。

### 选中一个 tile

**点击方块**（或聚焦后按 Enter）即可固定选中：

- 方块加上光晕，工具栏出现 `pinned <name> ×` 标记
- 它的存活行保持高亮，不再随鼠标移动而变化
- 该 tile 的名字——以及被合并进来的每个别名——在 IR 源码中的每一处出现都会被标记，
  效果类似搜索命中

若 IR 源码原本是收起的，固定选中会自动展开它，因为在源码中定位正是这个功能的目的。
此时悬停别的方块仍会显示其 tooltip，但不会干扰已固定的选中。再次点击该方块、按
`Esc`、或点击标记上的 `×` 即可取消。每张函数卡片各自维护自己的选中状态。

### IR 源码

**Show IR source** 会把左侧栏展开成该函数的 IR，与图逐行对齐：方块的上边缘正好落在
定义该 tile 的那一行。悬停一个方块会高亮它存活覆盖的那些源码行。

两种状态下左侧列都保持吸附，panel 横向滚动时它始终可见：收起时是一条细的行号栏，
展开后是一个宽度由你决定的源码面板。

**拖拽源码与地图之间的分割条**即可在两者间分配宽度。双击复位为默认宽度，或聚焦后用方向键
微调（按住 `Shift` 步长更大）；宽度限制在 140–2400 px，并按函数卡片各自记忆。过长的 IR
行在面板内横向滚动，行号吸附在其左边缘保持可见。

拖拽只重写 grid 模板——方块几何是其所属 lane 的百分比、lane 宽度是绝对像素——因此地图和
共享比例尺都不会移动，即使函数有数千行源码，拖拽也依然流畅。绘图面板高度上限为视口的
78 %，这样长函数的两个方向滚动条都够得着。

### 别名合并

SSA 的 phi/yield 链每轮迭代都会用一个新名字重新绑定同一块存储。把每次重绑都单独画出来
会让画面被一摞完全相同的矩形淹没，因此共享 `(space, base, offset, size)` 且生命周期
**相接**的 tile 会被合并成一个方块：标签取第一个名字，`+N` 表示合并掉的个数，
完整列表在 tooltip 里。

共享同一 slot 但生命周期**不相交**的 tile 永远不会被合并——那正是内存复用，
把它们分开显示恰恰是这张图的意义所在。

### 生命周期是词法（lexical）意义上的

一个 tile 的区间是从提及其名字的第一行到最后一行。这是一个词法近似，**不是**数据流
活跃性分析：在 `if` 两个分支里都被重新绑定的名字会覆盖整个 `if`；跨循环回边活跃的值
只体现其文本范围。

## 相关文档

- `docs/zh/dev/passes/35-allocate_memory_addr.md` —— 被绘制的那个 pass
- `docs/zh/dev/passes/34-memory_reuse.md` —— 决定哪些生命周期可以共享 slot 的 pass
- `docs/zh/dev/passes/00-pass_manager.md` —— `ReportInstrument`，现在只负责给出产物目录
- `docs/zh/dev/04-simulator-trace-cleaning.md` —— `pypto.tools` 中另一个事后分析工具
