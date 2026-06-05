# CuTe 入门

CuTe 是一组用于定义和操作线程与数据层次化多维布局的 C++ CUDA 模板抽象。CuTe 提供 Layout（布局）和 Tensor（张量）对象，将数据的类型、Shape（形状）、内存空间和布局紧凑打包，同时代用户完成复杂的索引计算。这样程序员可以专注于算法的逻辑描述，而 CuTe 负责机械性的簿记工作。借助这些工具，我们可以快速设计、实现和修改各类稠密线性代数运算。

CuTe 的核心抽象是层次化多维布局，可与数据数组组合以表示 Tensor。Layout 的表示能力足以涵盖实现高效稠密线性代数所需的大部分场景。Layout 还可以通过函数式组合进行组合和操作，在此基础上我们构建了诸如分块和分区等大量常用操作。

## 系统要求

CuTe 与 CUTLASS 3.x 共享软件依赖，
包括支持 C++17 宿主编译器的 NVCC。

## 前置知识

CuTe 是仅头文件的 CUDA C++库，要求 C++17
（2017 年发布的 C++ 标准修订版）。

本教程假定读者具备中级的 C++经验。
例如，我们假设读者会
阅读和编写模板函数与类，
以及使用 `auto` 关键字推导函数返回类型。
我们会温和对待 C++ 并解释一些
你可能已经了解的内容。

我们也假定读者具备中级的 CUDA 经验。
例如，读者必须了解
设备代码与主机代码的区别，
以及如何启动 kernel。

## 构建测试与示例

CuTe 的测试和示例在 CUTLASS 的正常构建流程中一并构建和运行。

CuTe 的单元测试位于 `[test/unit/cute](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute)` 子目录。

CuTe 的示例位于 `[examples/cute](https://github.com/NVIDIA/cutlass/tree/main/examples/cute)` 子目录。

## 库组织

CuTe 是仅头文件的 C++ 库，因此无需编译源码。库头文件位于顶层 `[include/cute](https://github.com/NVIDIA/cutlass/tree/main/include/cute)` 目录，其中组件按目录分组，以反映其语义。


| Directory                                                                                      | Contents                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[include/cute](https://github.com/NVIDIA/cutlass/tree/main/include/cute)`                     | 顶层每个头文件对应 CuTe 的一个基础构建块，如 `[Layout](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp)` 和 `[Tensor](https://github.com/NVIDIA/cutlass/tree/main/include/cute/tensor.hpp)`。 |
| `[include/cute/container](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container)` | STL 风格对象的实现，如 tuple、array 和 aligned array。                                                                                                                                                   |
| `[include/cute/numeric](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric)`     | 基础数值数据类型，包括非标准浮点类型、非标准整型、复数与整数序列。                                                                                                                                                            |
| `[include/cute/algorithm](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm)` | 工具算法的实现，如 copy、fill、clear，在可用的情况下会自动利用架构特定特性。                                                                                                                                                |
| `[include/cute/arch](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch)`           | 架构特定的矩阵乘法和 copy 指令封装。                                                                                                                                                                        |
| `[include/cute/atom](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom)`           | `arch` 中指令的元信息，以及分区、分块等工具。                                                                                                                                                                   |


## 教程

本目录包含 Markdown 格式的 CuTe 教程。
`[0x_gemm_tutorial.md](./0x_gemm_tutorial.md)`
说明如何使用 CuTe 组件实现稠密矩阵乘法。
它对 CuTe 做了整体概览，是很好的入门材料。

本目录中其他文件讨论 CuTe 的特定部分。

- `[01_layout.md](./01_layout.md)` 描述 Layout，CuTe 的核心抽象。
- `[02_layout_algebra.md](./02_layout_algebra.md)` 描述更高级的 Layout 操作及 CuTe 的 layout 代数。
- `[03_tensor.md](./03_tensor.md)` 描述 Tensor，
一种将 Layout 与数据数组组合而成的多维数组抽象。
- `[04_algorithms.md](./04_algorithms.md)` 概述 CuTe 对 Tensor 的通用算法。
- `[0t_mma_atom.md](./0t_mma_atom.md)` 演示 CuTe 的元信息及其与我们 GPU 架构特定
矩阵乘加（MMA）指令的接口。
- `[0x_gemm_tutorial.md](./0x_gemm_tutorial.md)` 手把手用 CuTe 从零构建 GEMM。
- `[0y_predication.md](./0y_predication.md)` 说明在
分块无法均匀装入矩阵时如何处理。
- `[0z_tma_tensors.md](./0z_tma_tensors.md)` 介绍 CuTe 用于支持 TMA 加载与存储的高级 Tensor 类型。

## 速查技巧

### 如何在主机或设备上打印 CuTe 对象？

`cute::print` 函数为重载了几乎所有 CuTe 类型，包括 Pointer、Integer、Stride、Shape、Layout 和 Tensor。若有疑问，不妨试着对目标对象调用 `print`。

CuTe 的打印函数可在主机或设备上使用。
注意在设备上打印开销较大。
即使只是将打印代码保留在设备上而不被调用
（例如在运行时未执行的 `if` 分支中打印），
也可能生成更慢的代码。
因此，调试完成后请务必移除设备上的打印代码。

你可能也希望只在每个 CTA 的 thread 0 或 grid 的 threadblock 0 上打印。`thread0()` 函数仅在 kernel 的全局 thread 0（即 threadblock 0 的 thread 0）上返回 true。打印 CuTe 对象的常见用法是仅在全局 thread 0 上打印。

```c++
if (thread0()) {
  print(some_cute_object);
}
```

某些算法依赖于某个 thread 或 threadblock，
因此你可能需要在非零的 thread 或 threadblock 上打印。
头文件
`[cute/util/debug.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cute/util/debug.hpp)`
中包含工具函数 `bool thread(int tid, int bid)`，
当运行在 thread `tid` 且 threadblock `bid` 时返回 `true`。

#### 其他输出格式

部分 CuTe 类型有特殊的打印函数，使用不同的输出格式。

`cute::print_layout` 函数将以纯文本表格的形式展示任意秩为 2 的 layout，便于直观理解从坐标到索引的映射。

`cute::print_tensor` 函数将以纯文本多维表格的形式展示任意秩为 1、2、3 或 4 的 tensor，便于在 copy 等操作后验证数据块是否符合预期。

`cute::print_latex` 函数会输出 LaTeX 命令，可通过 `pdflatex` 生成格式良好且带颜色的表格，适用于 Layout、TiledCopy 和 TiledMMA，有助于理解 CuTe 中的布局与分区模式。

