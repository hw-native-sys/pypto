# PyPTO

**English** | [中文](README.zh-CN.md)

## Overview

PyPTO (pronounced: pai p-t-o) is a high-performance programming framework for AI accelerators, designed to simplify the development of complex fused operators and entire model networks while maintaining high-performance computing capabilities. The framework adopts an innovative **PTO (Parallel Tensor/Tile Operation) programming paradigm**, with a **Tile-based programming model** as its core design philosophy. Through a multi-level intermediate representation (IR) system, it compiles AI model applications built via APIs from high-level Tensor graphs step by step into hardware instructions, ultimately generating executable code that runs efficiently on target platforms.

### Core Features

- **Tile-based Programming Model**: All computations are based on Tiles (hardware-aware data blocks), fully leveraging hardware parallel computing capabilities and memory hierarchy
- **Multi-level Computation Graph Transformation**: Transforms Tensor Graphs into Tile Graphs, Block Graphs, and Execution Graphs through compilation passes, with each step including a series of pass optimization workflows
- **Automated Code Generation**: Compilation results generate low-level PTO virtual instruction code through CodeGen, which is then compiled into executable code for target platforms
- **MPMD Execution Scheduling**: Executable code is loaded to the device side and scheduled to processor cores on the device using MPMD (Multiple Program Multiple Data) approach
- **Complete Toolchain Support**: Full compilation artifacts and runtime performance data can be visualized through IDE-integrated toolchains to identify performance bottlenecks; developers can also control compilation and scheduling behavior through the toolchain
- **Python-friendly API**: Provides intuitive Tensor-level abstractions that align with algorithm developers' thinking patterns, supporting dynamic shapes and symbolic programming
- **Layered Abstraction Design**: Exposes different abstraction levels to different developers - algorithm developers use the Tensor level, performance experts use the Tile level, and system developers use the Block level

### Target Users

- **Algorithm Developers**: Primarily use Tensor-level programming for rapid algorithm implementation and validation, focusing on algorithm logic
- **Performance Optimization Experts**: Can use Tile or Block levels for deep performance tuning to achieve optimal performance
- **System Developers**: Can integrate with third-party frameworks or develop toolchains at Tensor/Tile/Block and PTO virtual instruction set levels

## Getting Started

### Prerequisites

- **Python**: Version 3.10 or higher
- **CMake**: Version 3.15 or higher
- **C++ Compiler**: Supporting C++17 standard (GCC, Clang, or MSVC)
- **nanobind**: Version 2.0.0 or higher (automatically installed during build)
- **scikit-build-core**: Version 0.10.0 or higher (automatically installed during build)

### Installation

#### Install from Source

1. **Clone the repository**:

   ```bash
   git clone https://github.com/hw-native-sys/pypto.git
   cd pypto
   ```

2. **Install in development mode** (recommended for development):

   ```bash
   pip install -e .
   ```

   Or with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. **Install in production mode**:

   ```bash
   pip install .
   ```

The build system uses scikit-build-core to automatically handle CMake configuration and C++ extension compilation.

#### Build Options

- **Build type**: The default build type is `RelWithDebInfo` (optimized with debug symbols). You can override this:

  ```bash
  CMAKE_BUILD_TYPE=Release pip install .
  ```

- **Enable ccache** (optional, for faster rebuilds):

  ```bash
  # ccache will be automatically detected and used if available
  brew install ccache  # macOS
  sudo apt-get install ccache  # Ubuntu/Debian
  ```

### Running Examples

PyPTO includes several examples demonstrating different features:

#### 1. Flash Attention Builder

```bash
python examples/ir_builder/flash_attention_builder.py
```

#### 2. Block Operations Example

```bash
python examples/ir_builder/block_ops_example.py
```

#### 3. IR Parser Example

```bash
python examples/ir_parser/flash_attention_parsing.py
```

### Running Tests

To run the test suite:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ir_builder.py

# Run with verbose output
pytest -v tests/
```

## License

This project is licensed under the **CANN Open Software License Agreement Version 2.0**.

This license grants you a limited, worldwide, royalty-free license to download, use, modify, integrate, and distribute the Software or its derivative works for the purpose of developing software **solely for use in systems with Huawei AI Processors** (including Ascend, Kirin, Yueying, and other Huawei-branded AI chipsets).

See the [LICENSE](LICENSE) file for the full license text.
