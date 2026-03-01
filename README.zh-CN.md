# PyPTO

[English](README.md) | **中文**

## 概述

PyPTO（发音：pai p-t-o）是一个面向 AI 加速器的高性能编程框架，旨在简化复杂融合算子和整网模型的开发，同时保持高性能计算能力。该框架采用创新的 **PTO（Parallel Tensor/Tile Operation）编程范式**，以 **Tile 编程模型**为核心设计理念。通过多级中间表示 (IR) 系统，将通过 API 构建的 AI 模型应用从高层 Tensor 图逐步编译为硬件指令，最终生成在目标平台上高效运行的可执行代码。

### 核心特性

- **Tile 编程模型**：所有计算基于 Tile（硬件感知的数据块），充分利用硬件并行计算能力和内存层级结构
- **多级计算图变换**：通过编译 Pass 将 Tensor 图变换为 Tile 图、Block 图和执行图，每一步都包含一系列 Pass 优化流程
- **自动代码生成 (CodeGen)**：编译结果通过 CodeGen 生成底层 PTO 虚拟指令代码，随后编译为目标平台的可执行代码
- **MPMD 执行调度**：可执行代码被加载到设备端，使用 MPMD（多程序多数据）方式调度到设备上的处理器核心
- **完整工具链支持**：完整的编译产物和运行时性能数据可通过 IDE 集成工具链可视化，以识别性能瓶颈；开发者也可通过工具链控制编译和调度行为
- **Python 友好 API**：提供符合算法开发者思维模式的直观 Tensor 级抽象，支持动态形状和符号编程
- **分层抽象设计**：向不同开发者暴露不同抽象层级——算法开发者使用 Tensor 级，性能专家使用 Tile 级，系统开发者使用 Block 级

### 目标用户

- **算法开发者**：主要使用 Tensor 级编程进行快速算法实现和验证，专注于算法逻辑
- **性能优化专家**：可使用 Tile 或 Block 级进行深度性能调优，以实现最佳性能
- **系统开发者**：可在 Tensor/Tile/Block 和 PTO 虚拟指令集层面与第三方框架集成或开发工具链

## 快速开始

### 前置条件

- **Python**：3.10 或更高版本
- **CMake**：3.15 或更高版本
- **C++ 编译器**：支持 C++17 标准（GCC、Clang 或 MSVC）
- **nanobind**：2.0.0 或更高版本（构建时自动安装）
- **scikit-build-core**：0.10.0 或更高版本（构建时自动安装）

### 安装

#### 从源码安装

1. **克隆仓库**：

   ```bash
   git clone https://github.com/hw-native-sys/pypto.git
   cd pypto
   ```

2. **以开发模式安装**（推荐用于开发）：

   ```bash
   pip install -e .
   ```

   或安装开发依赖：

   ```bash
   pip install -e ".[dev]"
   ```

3. **以生产模式安装**：

   ```bash
   pip install .
   ```

构建系统使用 scikit-build-core 自动处理 CMake 配置和 C++ 扩展编译。

#### 构建选项

- **构建类型**：默认构建类型为 `RelWithDebInfo`（带调试符号的优化版本）。可以覆盖此设置：

  ```bash
  CMAKE_BUILD_TYPE=Release pip install .
  ```

- **启用 ccache**（可选，加快重复构建）：

  ```bash
  # ccache 可用时会自动检测并使用
  brew install ccache  # macOS
  sudo apt-get install ccache  # Ubuntu/Debian
  ```

### 运行示例

PyPTO 包含多个演示不同功能的示例：

#### 1. Flash Attention 构建器 (Builder)

```bash
python examples/ir_builder/flash_attention_builder.py
```

#### 2. Block 操作示例

```bash
python examples/ir_builder/block_ops_example.py
```

#### 3. IR 解析器 (Parser) 示例

```bash
python examples/ir_parser/flash_attention_parsing.py
```

### 运行测试

运行测试套件：

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_ir_builder.py

# 以详细输出运行
pytest -v tests/
```

## 许可证

本项目基于 **CANN Open Software License Agreement Version 2.0** 许可。

该许可授予您有限的、全球范围的、免版税的许可，可下载、使用、修改、集成和分发本软件及其衍生作品，用于开发**仅在华为 AI 处理器系统上使用**的软件（包括 Ascend、Kirin、Yueying 及其他华为品牌 AI 芯片组）。

详细许可条款请参阅 [LICENSE](LICENSE) 文件。
