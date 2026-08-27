# 缩小精度差距

结果与 golden 对不上。本页讲的是**按什么顺序去怀疑**，不是 API 参考。

> **前置**：[第一个算子](../tutorials/00-elementwise.md) —— 本页假定你已经有那里的 `allclose` 对比。

## 顺序

```text
结果与 golden 不符
├─ 1. golden 本身对吗？        → write_golden；rtol/atol 合理吗？
├─ 2. 本来就该有差异吗？       → 下面的「可接受差异」表
├─ 3. 编译器已经警告了吗？     → verification level、diagnostics
├─ 4. 是哪个 pass 引入的？     → torch codegen + validate_ir，二分
└─ 5. 是哪个张量错了？         → dump_tag / dumps= + enable_dump_args
```

第 1、2 步花几分钟，就能排除掉大多数报告。第 4、5 步要几小时。**不要从第 4 步开始。**

## 工具

| 步骤 | 层次 | 入口 |
| ---- | ---- | ---- |
| 1 | 端到端 | `pypto.runtime.write_golden` + `RunConfig(rtol=, atol=, golden_data_dir=)` |
| 3 | IR 合法性 | `ir.compile(verification_level=...)` / `PYPTO_VERIFY_LEVEL` |
| 3 | 编译期告警 | `diagnostic_phase` / `disabled_diagnostics` |
| 4 | IR 语义 | `pypto.debug.torch_codegen` |
| 4 | 逐 pass 校验 | `CompiledProgram.validate_ir` |
| 4 | IR 结构 | `ir.compile(dump_passes=PassDumpLevel.EXPLICIT)` |
| 5 | 运行期数据 | `pl.dump_tag(t)` / `dumps=[t]` + `RunConfig(enable_dump_args=1\|2)` |

## 第 1 步：golden 对吗

默认容差是 `rtol=1e-5`，这对 **FP16 输入是错的** —— 它只带约三位十进制有效数字，所以一个正确的 FP16 matmul 拿去和 FP32 参考在 `1e-5` 上比就会挂。在调查 kernel 之前，先确认容差匹配的是**输入**精度。

`write_golden` 把参考记录下来，让后续运行与一份固定产物比较，而不是与一个当场重算的结果比较。当参考本身不确定时，这一点很关键。

## 第 2 步：本来就该有差异吗

有些差异是一个正确编译器的正确行为。在二分任何东西之前，先把它们排除掉。

| 来源 | 差异 | 说明 |
| ---- | ---- | ---- |
| split-K / 原子加 | 末位，逐次运行不同 | 跨核的累加顺序不固定 |
| FP16 / BF16 累加 | 随规约长度增长 | 能宽着累加就用 FP32 |
| 规约形状 | 二叉树 vs 顺序 | `col_sum` 传不传 `tmp_tile` 会改变顺序 |
| backend 差异 | 指令级 | 同一个 op 在不同 backend 上不必逐位相同 |
| 多跳 cast | **通常没有** —— 见下 | `LegalizeTileCast` 展开 ISA 一步做不到的转换 |

**多跳 cast 值得说准确，因为它太容易被赖上。** 在 A5 上 `INT32→FP16` 会展开成 `INT32→FP32→FP16`。这条链与它所替代的那个参考**逐位相同** —— 参考指的是一个假想的、单步完成且采用相同舍入模式与相同溢出行为的 `INT32→FP16`。理由分两半：

- **`|x| ≤ 65504`**（FP16 最大的有限值）：这样的 `x` 远小于 `2^24`，在 FP32 里是精确的。FP32 那一跳不舍入，唯一发生的舍入就是最后一跳 —— 与单步参考所做的舍入相同。
- **`65504 < |x| < 65520`**：这一段**不会**溢出。在就近舍入下它们落到最大的有限 FP16 值 `65504`，因为 `65520` 正是它与该格式本应有的下一个值之间的中点。两种形式的舍入方式相同。
- **`|x| ≥ 65520`**：链式与单步参考都会溢出到无穷。`|x| > 2^24` 时 FP32 那一跳**确实**会舍入，但被舍入的值本就远在 FP16 范围之外，因此改变不了结果。

中间那一段最值得记住：FP16 的「溢出」始于 `65520`，而不是 `65504`。

只有当某个中间类型无法精确表示**确实落在目标范围内**的源值时，链式转换才会引入真正的差异。请去 [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) 查你这条链属于哪一类，而不是假定那一跳就是元凶。

这个断言是可验证的，所以本页就验证它。下面的块用 PyPTO 做 `INT32 → FP16`，并与 torch 对同一批值的转换逐位比较，覆盖论证所依赖的三个区间 —— 可精确表示、超出 FP16 范围、以及超过 `2^24`（此时 FP32 那一跳自身会舍入）：

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
```

<!-- doctest: run -->
```python
ROWS, COLS = 16, 128


@pl.jit
def to_fp16(x: pl.Tensor[[ROWS, COLS], pl.INT32], out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP16]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.cast(x, pl.FP16)
    return out


values = torch.zeros(ROWS, COLS, dtype=torch.int32)
values[0, :4] = torch.tensor([0, 1, 2048, 65504])            # exact in FP16
values[0, 4:8] = torch.tensor([65505, 65519, 65520, 70000])  # the boundary: 65519 -> 65504, 65520 -> inf
values[0, 8:12] = torch.tensor([1 << 20, 1 << 24, (1 << 24) + 1, 1 << 25])   # past 2^24
values[0, 12:16] = torch.tensor([-65504, -65519, -65520, -70000])            # the same boundary, negative

out = torch.zeros(ROWS, COLS, dtype=torch.float16)
to_fp16(values, out, config=CFG)

# Bit-for-bit against torch's conversion of the same integers.
expected = values.to(torch.float16)
assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), (
    f"differs at {(out != expected).nonzero()[:4].tolist()}"
)
```

它跑的是本页 CI 目标所用的 backend，因此确立的是该断言中面向用户的那一半 —— 你拿到的数值与参考一致。A5 特有的**展开**在上文以论证给出，而非在此执行，因为那条链只在该 backend 上出现。

## 第 3 步：编译器是不是已经说了

在做任何二分之前先提高校验级别重编译一次 —— 一份格式错误的 IR 报告会直接点名那个 pass：

```python
prog = ir.compile(program, verification_level=...)   # 或 PYPTO_VERIFY_LEVEL
```

## 第 4 步：是哪个 pass 引入的

这是昂贵的一步，也是值得认真做的一步。

`pypto.debug.torch_codegen` 把 IR 的 `Program` 或 `Function` 变成可执行的 torch，于是 IR 的**语义**可以在主机上运行并与参考对比 —— 不涉及设备：

<!-- doctest: run -->
```python
from pypto.debug import torch_codegen


@pl.jit
def fused(a: pl.Tensor[[64, 128], pl.FP32], b: pl.Tensor[[64, 128], pl.FP32],
          out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.mul(pl.add(a, b), 2.0)
    return out


a = torch.randn(64, 128)
b = torch.randn(64, 128)

src = torch_codegen(fused.lower(a, b, torch.zeros(64, 128)))  # check_shapes=True to assert shapes
assert "def " in src and "torch" in src                       # it is executable python
```

重点不在那个字符串，而在于 IR 的语义可以在 host 上执行并与你的参考对比，从而把「IR 错了」与「设备与 IR 不一致」分开。

`CompiledProgram.validate_ir` 逐 pass 做这件对比。于是二分是机械的：**第一个 IR 不再吻合的 pass** 就是引入差异的那个。用 `ir.compile(dump_passes=PassDumpLevel.EXPLICIT)` dump 出它前后的 IR，读那两份。

> **把你的容差传进去。** `validate_ir` 默认 `rtol=5e-2, atol=5e-2` —— 那是它自己的默认值，不是你在第 1 步定下的那个，而且比多数参考应有的宽松得多。放任不管的话，只要回退小于该值它就报告「对上了」，二分给你的边界便不是真正的边界：
>
> ```python
> compiled.validate_ir(..., rtol=RTOL, atol=ATOL)   # 第 1 步定下的容差
> ```

这一步定位的是**语义**改变。它看不见只在设备上才出现的差异 —— 那要靠第 5 步。

## 第 5 步：是哪个张量错了

当每个 pass 的 IR 都对、而设备结果不对时，去比对真实数据：

```python
pl.dump_tag(t)       # 标记你关心的张量
cfg = RunConfig(platform="a2a3sim", enable_dump_args=1)
```

等级 `1` 只 dump 被标记的张量；等级 `2` dump 每个任务的全部输入输出。用 `python -m simpler_setup.tools.dump_viewer` 查看。

> **致命陷阱：** 在大负载上做全量 dump（`enable_dump_args=2`）会把主机侧收集器打满（约 42 MB/s 的排空速率），并让 AICPU 被 STARS op-execute 超时杀掉。优先用等级 `1` 加上对具体张量的 `pl.dump_tag`。

### 当结果每次运行都不一样

在 dump 任何东西之前，先拿同一份输入多跑几次。值会变说明有东西没有定序 —— 但可能是两件很不一样的事，而且区分起来很便宜：

- **只有末几位在变，且 kernel 用了 split-K 或原子加。** 那是跨核的累加顺序，属于预期之内，上面第 2 步已经覆盖，不需要修。
- **变动比这更大** —— 整块区域不对、数值差很多，或者时对时错。那是任务定序缺陷，pass dump 解释不了它：IR 在每个 pass 上都有权看起来是对的，因为语句顺序并不约束执行顺序。

对于后一种，运行时从缓冲重叠推断 RAW 与 WAW，但 **WAR 不被跟踪**：一个写者去覆盖某个别的任务可能还在读的缓冲，不会产生任何边 —— 因为要找出所有在飞的读者，就得在热路径上为每次写做一趟遍历。那条反依赖得由你来声明。完整规则与代价见[依赖](../performance/03-dependencies.md)。

```python
cfg = RunConfig(platform="a2a3sim", enable_dep_gen=True)   # writes deps.json
```

读这张图，找那条你以为存在的边。修法是把它点名 —— 在写者上写 `deps=[reader_tid]`，读者保持普通输入（`INPUT`，也就是不加标注时本来的样子）。

> **不要靠把读者提升成 `pl.InOut` 来修。** 它确实会产生那条边：`INOUT` 会注册成写者，于是覆盖操作对它取一条 WAW 边。但它同时会让那个缓冲的**其他每一个**读者互相串行 —— 因为它们会轮流成为注册生产者。一个本可被多个任务并发读取的张量会就此完全失去并发，只为买到一条反依赖。

注意 dump 本身会扰动它：`enable_dump_args` 会增加 GM 流量并改变时序，所以打开它之后竞争可能消失或移位。先从图上把定序问题解决掉，再回去比对数值。

### 当你想看的那个值不是张量

dump 只够得着**本来就是张量**的东西。而真正能定案的那个值，往往是 InCore 函数内部的中间结果 —— 最终 store 之前的累加器、融合链里走完一步的 tile —— 它们从不落到 GM，因此也无法被 tag。

给它一个去处：给 kernel 加一个临时的 `pl.Out` 参数，把中间结果 store 进去，再经编排一路带出来。

```python
@pl.jit.incore
def fused(x: pl.Tensor, out: pl.Out[pl.Tensor],
          probe: pl.Out[pl.Tensor]):        # 临时的，只为这次排查
    acc = pl.add(pl.load(x, [0, 0], [64, 128]), 1.0)
    probe = pl.store(acc, [0, 0], probe)    # 这个中间结果现在可被检视
    out = pl.store(pl.exp(acc), [0, 0], out)
    return out, probe
```

这是一次调试改动，不是设计：多出来的参数要付一次 GM 往返，还会改变依赖图，所以问题答完就把它拿掉。它换来的是把中间结果与你的 host 参考直接对比 —— 这通常能一次运行就把「输出错了」变成「错在这一步」。

## 边界情况

| 症状 | 可能原因 | 步骤 |
| ---- | -------- | ---- |
| **正确的 kernel 过不了 `allclose`** | 拿 `rtol=1e-5` 去比 FP16 输入 | 1 |
| **同一输入，多次运行结果不同** | split-K 的原子累加顺序，**或缺一条 WAR 边** | 2，然后 5 |
| **只在长规约时有差异** | FP16/BF16 累加器 | 2 |
| **赖到多跳 cast 头上** | 通常逐位相同 —— 先查它属于哪一类 | 2 |
| **每个 pass 的 IR 都对，设备不对** | 不是语义缺陷 | 5 |
| **行最大值全是 `0.0`** | padding 参与了规约 | 见 [规约与 softmax](../tutorials/01-reduction-softmax.md) |

## 参见

- [实例](01-cases.md) —— 这套顺序的端到端应用。
- [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) —— cast 链何时是精确的。
- [规约与 softmax](../tutorials/01-reduction-softmax.md) —— padding 与规约。
