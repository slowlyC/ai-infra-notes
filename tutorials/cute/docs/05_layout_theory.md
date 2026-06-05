# CuTe Layout 表示与代数 —— 理论基础

> 本文翻译自论文 *CuTe Layout Representation and Algebra*（Cris Cecka, NVIDIA Research, [arXiv:2603.02298](https://arxiv.org/abs/2603.02298)）。
> 论文为 CuTe 的 Layout 表示和代数运算提供了完整的数学规范。
> 与本教程系列中 [01_layout](./01_layout.md) 和 [02_layout_algebra](./02_layout_algebra.md) 的工程导向内容互补，
> 本文侧重「为什么这样设计」的理论基础。

---

## 摘要

面向高性能计算和深度学习的现代架构日益引入专用张量指令，包括用于矩阵乘法的 Tensor Core 以及硬件优化的多维数据拷贝指令。这些指令规定了固定且往往复杂的数据布局，必须在整个执行流水线中正确传播，才能保证正确性与最优性能。

本文介绍 CuTe —— 一种表示和操作张量的新型数学规范。CuTe 引入两项关键创新：

1. **层次化 Layout 表示**：直接扩展传统的扁平 Shape + 扁平 Stride 张量表示，使得能够表达现代硬件指令所需的复杂映射。
2. **丰富的 Layout 代数运算**：包括拼接 (concatenation)、合并 (coalescence)、复合 (composition)、补 (complementation)、除法 (division)、分块 (tiling) 和逆 (inversion)，用于实现复杂的布局操作、推导、验证和静态分析。

CuTe 的 Layout 为 GPU 内核中的数据布局和线程安排提供了统一框架，Layout 代数则支持在编译时对布局属性做推理，并以简洁方式表达现代专用张量指令所需的分块和分区模式。

CuTe 已在生产系统中成功部署，是 NVIDIA CUTLASS 库以及 CuTe DSL 等相关项目的基础。

---

## 目录

- [1 引言与动机](#1-引言与动机)
  - [1.1 相关工作](#11-相关工作)
  - [1.2 规范循环与循环变换](#12-规范循环与循环变换)
  - [1.3 张量与折叠](#13-张量与折叠)
- [2 Layout 表示](#2-layout-表示)
  - [2.1 Tuple 与 HTuple](#21-tuple-与-htuple)
  - [2.2 Shape](#22-shape)
  - [2.3 Stride](#23-stride)
  - [2.4 Layout](#24-layout)
  - [2.5 Tensor](#25-tensor)
  - [2.6 应用](#26-应用)
- [3 Layout 代数](#3-layout-代数)
  - [3.1 拼接 (Concatenate)](#31-拼接-concatenate)
  - [3.2 合并 (Coalesce)](#32-合并-coalesce)
  - [3.3 复合 (Composition)](#33-复合-composition)
  - [3.4 逆 (Inverse)](#34-逆-inverse)
  - [3.5 补 (Complement)](#35-补-complement)
- [4 结论](#4-结论)
- [参考文献](#参考文献)

---

## 1 引言与动机

现代 GPU 日益面向以张量为中心的计算做优化，这受深度学习和科学计算需求的驱动。NVIDIA 的 Volta 架构 [1] 引入了 Tensor Core，在硬件中直接支持高效的小矩阵乘法。Turing [2] 和 Ampere [3] 扩展了这一能力，增加了用于 GPU 存储层次内结构化矩阵搬运的专用指令。Hopper [4] 和 Blackwell [5] 架构进一步推进了这一范式，引入了用于在全局内存和共享内存之间高效传输 rank-5 张量的拷贝指令，并进一步扩展了 Tensor Core 的能力。充分利用这些面向张量的硬件特性对于发挥 GPU 峰值性能至关重要，这推动了能够高效表示和操作这些张量的高性能编程模型。

这些已有和新兴硬件高度依赖多维数据在多级层次化内存空间和多级层次化并行度中的存储和访问方式。数据存储布局一直影响性能——它决定了内存访问何时、如何发生——但随着硬件指令变得更大并规定了固定的输入输出布局，其对**正确性**的影响也变得至关重要。这些布局必须贯穿整个执行流水线传播，以保证硬件指令的正确调用和优化的内存访问模式。

本文介绍 CuTe（CUDA Tensors / Compute Unified Tensors）的基础概念，旨在为编写峰值性能线性代数库提供构建模块。CuTe 的两项关键创新：

- **新型张量 Layout 表示**：CuTe 的 Shape、Layout 和 Tensor 天然具有层次性，由更小的嵌套实例构成。这种层次性提供了表达现代张量指令所需复杂映射的手段，同时仍然是 BLAS、`torch.tensor`、`numpy.ndarray`、MATLAB 等库中扁平 Shape + 扁平 Stride 表示的严格扩展。
- **定义在 Layout 上的新型代数运算**：CuTe Layout 支持丰富的运算集，包括拼接、合并、复合、补、除法、分块和逆，运算结果均为新的 CuTe Layout。这些运算支持现代张量指令所要求的复杂分区、操作、验证和推导。

CuTe 的 Layout 表示为编写通用算法中的线程和数据管理提供了直觉化框架。CuTe 的 Layout 代数为开发高性能线性代数内核时操作和生成新 Layout 提供了表达力。它们带来的能力包括：

- **支持复杂布局和分区**：CuTe 支持表达应用特定的复杂数据模式和专用张量指令所需的复杂分区模式。
- **关注点分离**：数据布局独立于算法逻辑声明，促进清晰性和模块性。
- **静态分析与优化**：复杂的代数技术支持根据架构约束对张量参数做检查、重排和分区。

### 1.1 相关工作

CuTe 的动机源于支持高效张量缩并 (tensor contraction) 的需求，张量缩并是许多科学计算和机器学习应用的核心。计算通用张量缩并的传统方法依赖矩阵化 (matricization)——在逻辑上或显式地重组张量数据，以通过一系列 BLAS 库调用来执行计算。

BLAS 提供了核心线性代数运算的高效可移植实现，面向多种架构有高度优化的版本 [6]。在 BLAS 原语中，通用矩阵乘法 (GEMM) 是科学计算和机器学习中优化程度最高、使用最广泛的运算。

BLIS 框架 [7] 扩展了 GEMM，同时支持行列两个模式的非单位步幅，解决了不规则内存布局的部分挑战而无需显式内存拷贝。BLAS 的 strided-batched GEMM 扩展进一步泛化了该原语，使之适用于更多张量缩并 [8]。进一步抽象矩阵布局，CuTe 的一个关键洞察是：利用张量标记法中的多索引 (multi-index)，可以将任意张量缩并转化为规范的 batched-GEMM 原语。

许多现有库依赖张量，且几乎所有库都基于扁平 Shape + 扁平 Stride 表示。Python 中有 `numpy.ndarray` 和 `torch.tensor`：

```python
>>> import numpy
>>> a = numpy.ndarray([3,7,5])
>>> a.dtype
dtype('float64')
>>> a.shape
(3, 7, 5)
>>> a.strides
(280, 40, 8)
```

```python
>>> import torch
>>> a = torch.empty(3,7,5)
>>> a.dtype
torch.float32
>>> a.shape
torch.Size([3, 7, 5])
>>> a.stride()
(35, 5, 1)
```

C++ 中有 `std::mdspan`：

```cpp
std::mdspan a = std::mdspan(data, 3, 7, 5);
a.extent(0); // 3
a.extent(1); // 7
a.extent(2); // 5
a.stride(0); // 35
a.stride(1); // 5
a.stride(2); // 1
```

CuTe 支持这些表示，并通过层次化 Shape 和 Stride、非整数 Stride 以及非整数 Layout 陪域的泛化来严格扩展它们。

对稠密张量表示的独立泛化工作包括 HeLayers [9]、ThunderKittens [10] 和 OpenAI Triton 编译器 [12] 使用的 Linear Layouts [11] 方法。ThunderKittens 为寄存器内存、共享内存、行/列主序 tile、tile-of-tile 以及规定的 warp 和线程访问模式实现了多种专用类型。这些类型在设计时考虑了架构布局要求和内存层次，但其表示范围局限于已有类型和模式。Linear Layouts 基于 F₂ 线性代数，提供了更通用的张量布局表示以及布局分析和生成的途径。然而 Linear Layouts 对 F₂ 的严格依赖使得某些运算符对人类而言难以检查，并将工作限制在 2 的幂次 Shape 和 Stride 上，这对许多应用是不可接受的。

其他启发了本工作部分内容的布局分析方法同样依赖 F₂ 线性代数，如 Edelman 等人 [13]、Cormen 等人 [14] 和 Bouverot 等人 [15] 的工作。这些工作为分析和算法生成提供了基础，但由于所处的应用语境而有所局限。

由于 CuTe 的 C++ 实现在 CUTLASS v3 [16] 中是开源的且附带文档，它已被独立工作引用和使用。Bhaskaracharya 等人 [17] 在整数集关系 (ISL) 语境下分析了 CuTe 和 Linear Layouts [11]，但遗漏了允许将 Linear Layouts 表示为 CuTe Layout 的 Stride 抽象。LEGO [18] 使用了 CuTe Layout 的受限形式和 CuTe 复合的原始形式在代码生成器中生成复杂索引。Colfax Research [19] 在范畴论语境下分析了 CuTe Layout 及其上的部分运算。本文旨在提供 CuTe 概念及其应用的更权威和形式化的处理。

CuTe 的设计和概念已在多个应用中展示了其效用。例如，CuTe 在 Graphene 张量编译器 [20] 中发挥了关键作用。CuTe 的 C++ 实现被用于 Stream-K 算法 [21] 的实现，并且是 NVIDIA CUTLASS v3 库 [22] 开发的核心。CuTe 还是大语言模型领先实现的核心组件，包括 FlashAttention 的各代演进 [23, 24, 25]。此外，CuTe 是 CuTe DSL [26] 等编译器项目的基础——CuTe DSL 是一种基于 Python 的 DSL，用于动态编译面向线性代数应用的 CUDA 软件。

### 1.2 规范循环与循环变换

循环索引的显式计算是高性能线性代数内核开发中的常见挑战。这些计算对程序员来说很难写对，维护起来更是困难。与其将数据访问信息和算法逻辑耦合，不如以矩阵/向量坐标的形式清晰地书写算法逻辑，并将数据访问模式抽象为数据布局。

为说明这一点，首先定义本工作所处理的循环嵌套类别。我们考虑一种**标准循环形式**：单索引循环，从零开始，以常数为界，每次迭代递增 1。

例如，考虑如下循环：

```c
for (int m = 2; m <= 50; m += 3)
    A[m] = e(m);
```

该循环对 A[2], A[5], A[8], ... 赋值。可以变换为规范循环：

```c
for (int i = 0; i < 17; ++i)
    (A + 2)[3*i] = g(i);
```

指针偏移了一个循环不变常量，循环步长归一化为 1，下界变为零，上界为紧且不含端点，纯表达式变换为 g(i) = e(3*i+2)。此时自然可以将上面的例子解释为：遍历一个逻辑上有 17 个元素的向量，逻辑坐标以步幅 3 索引基地址 A + 2 处的数据。该程序可用以下数据表示：

```
Accessor: A + 2
Shape:    17
Stride:   3
```

嵌套循环可做类似处理。考虑如下二维循环嵌套：

```c
for (int n = 3; n < 43; n += 2)
    for (int m = 4; m <= 22; m += 5)
        A[p*m + q*n] = e(m,n);
```

变换为规范循环形式：

```c
for (int j = 0; j < 20; ++j)
    for (int i = 0; i < 4; ++i)
        (A + 4*p + 3*q)[5*p*i + 2*q*j] = g(i,j);
```

此时自然将变换后的循环解释为：遍历一个逻辑上 4x20 的矩阵，行坐标 i 的步幅为 5p，列坐标 j 的步幅为 2q，基地址为 A + 4p + 3q。表示为：

```
Accessor: A + 4p + 3q
Shape:    ( 4, 20)
Stride:   (5p, 2q)
```

一个重要观察是：4x20 矩阵也可以解释为一个 80 元素的向量，具有非均匀的半仿射步幅，等价的规范循环形式为：

```c
for (int k = 0; k < 80; ++k)
    (A + 4*p + 3*q)[5*p*(k%4) + 2*q*(k/4)] = f(k);
```

其中 `%` 是取模，`/` 是整数地板除法。这一变换是反字典序双射 (colexicographical bijection) (i,j) = (k%4, k/4)，在 2D 坐标 (i,j) 和 1D 坐标 k 之间建立对应。该双射等价于前面表示的 Shape，并可直接从 Shape 推导。因此，Shape 表示既可接受 2D 坐标也可接受 1D 坐标，提供了灵活且秩无关的索引框架。

进一步地，规范循环形式还为张量计算优化编译器中常见的可证正确的循环变换提供了指导。考虑最一般的规范循环嵌套：

```c
for (int i0 = 0; i0 < N0; ++i0)
    for (int i1 = 0; i1 < N1; ++i1)
        for (int i2 = 0; i2 < N2; ++i2)
            ...
            A[d0*i0 + d1*i1 + d2*i2 + ...] = e(i0,i1,i2,...);
```

其中 (N₀, N₁, N₂, ...) 是计算的 Shape，(d₀, d₁, d₂, ...) 是访问模式的 Stride：

```
Accessor: A
Shape:    (N0, N1, N2, ...)
Stride:   (d0, d1, d2, ...)
```

由于 Shape:Stride 信息与循环嵌套之间存在一一对应，与其问如何对循环做变换（拆分、转置、拼接、排列、截断、向量化等），不如问：「对 Shape:Stride 表示有哪些合法的变换方式，什么算子提供这些变换？」实际上，如果 **L** = Shape:Stride 表示数据访问和循环嵌套，那么什么函数 P 使得

**L'** = P(**L**) = **L** ∘ P

是 **L** = Shape:Stride（具有某种 Shape 和 Stride）到新循环嵌套 **L'** = Shape':Stride'（具有可能不同的 Shape 和 Stride）的有意义变换？这些变换 P 本质上**重写**了循环嵌套，如果定义得当，它们本身可以是可组合的、可逆的，并提供对命令式循环的函数式编程风格控制。考虑到 1D 坐标与 ND 坐标之间的双射，本文表明这些变换算子 P 的一种非常有效的表示是 **P** = Shape\*:Stride\* —— 与我们用来表示数据访问和循环嵌套的对象是同一类对象。

### 1.3 张量与折叠

为进一步说明可同时用 ND 坐标和 1D 坐标索引的 Shape 的价值，我们推广 [8] 中将张量缩并映射到规范 BLAS 原语的观察。

本文中，张量用粗体字母表示，索引用小写字母，索引的界用对应的大写字母。张量的秩 (rank) 指其维度数，也称模式 (mode) 数。例如：

- 标量 α 是 rank-0 张量
- 向量 **a**ᵢ 是 rank-1 张量，0 ≤ i < I
- 矩阵 **A**ₘₙ 是 rank-2 张量，0 ≤ m < M，0 ≤ n < N
- 三维数组 **A**ₘₙₚ 是 rank-3 张量，0 ≤ m < M，0 ≤ n < N，0 ≤ p < P

对只出现在等式一侧的重复索引求和（Einstein 标记法），一个张量缩并的例子是：

**C**ₛₜqₚ = **A**ₛₜᵤₚᵣ **B**qₜᵣᵤ

表示一个 rank-5 张量与一个 rank-4 张量的缩并，得到一个 rank-4 张量。上述张量缩并可以改写为：

**C**₍ₛₚ₎₍q₎₍ₜ₎ = **A**₍ₛₚ₎₍ᵤᵣ₎₍ₜ₎ **B**₍q₎₍ᵤᵣ₎₍ₜ₎

其中原始张量缩并的模式被分为四类：

- **行模式** m̂：出现在 **A** 和 **C** 中，不出现在 **B** 中
- **列模式** n̂：出现在 **B** 和 **C** 中，不出现在 **A** 中
- **归约模式** k̂：出现在 **A** 和 **B** 中，不出现在 **C** 中
- **批次模式** l̂：出现在 **A**、**B** 和 **C** 中

这称为张量**折叠** (folding)。折叠张量不需要任何显式拷贝，只需改变数据的视图。

以一个 2x2x2 的 8 元素张量为例。扁平表示为 Shape (2,2,2)、Stride (2,1,4)。该张量可以将第三模式折叠进第一模式，得到 4x2 矩阵。此时扁平表示为 Shape (4,2)、Stride (2,1)。原则上也可以将第三模式折叠进第二模式，得到 2x4 矩阵，但此时**不存在**能表示第二模式步幅的单一整数——扁平表示无法表达。

CuTe 表示使用层次化 Shape 和 Stride 解决了这个问题：

| 视图 | 扁平表示 | CuTe 表示 |
|------|----------|-----------|
| 张量 2x2x2 | Shape: (2,2,2) Stride: (2,1,4) | Shape: (2,2,2) Stride: (2,1,4) |
| 矩阵 4x2（模式 2 折入模式 0） | Shape: (4,2) Stride: (2,1) | Shape: ((2,2),2) Stride: ((2,4),1) |
| 矩阵 2x4（模式 2 折入模式 1） | Shape: (2,4) Stride: (2,**X**) ← 不存在 | Shape: (2,(2,2)) Stride: (2,(1,4)) |

对 4x2 矩阵，扁平表示是 CuTe 表示的合并 (coalesced) 版本; 而对 2x4 矩阵，不存在这样的扁平表示。

这种泛化的张量折叠使得所有张量缩并都可以写成单一的规范缩并形式：

**C**_{m̂ n̂ l̂} = **A**_{m̂ k̂ l̂} **B**_{n̂ k̂ l̂}

其中每个模式可以是单个模式或一组模式（称为多模式, multi-mode）。参照 Section 1.2 的规范循环，无论 m̂ 的 Shape 是 M 还是 (M₀, M₁)，都可以用 1D 坐标 m 遍历。因此，任何张量缩并都可以折叠为规范的 batched-GEMM，用四层嵌套循环实现：

```c
for (int l = 0; l < L; ++l)
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                C(m, n, l) += A(m, k, l) * B(n, k, l);
```

该 batched-GEMM 的简单实现可用于计算广泛的兼容张量缩并，包括矩阵乘法 (GEMM)、张量缩并 (GETT) 和卷积 (CONV)，只需构造恰当的折叠布局。

优化可以聚焦于循环重排、分块、向量化等常见优化。这些变换经常可以用 Layout 表示为算法坐标空间上的排列。这些变换 Layout 与数据 Layout 做函数式复合，即可生成与原始问题保持一致的新循环嵌套。详见 Section 3.3 关于 Layout 复合及其在通用分区中的应用。

---

## 2 Layout 表示

CuTe Layout 是多功能对象，能表达广泛的数据和线程安排，在抽象物理地址、分离迭代顺序和存储顺序方面具有极大的效用。本节定义 CuTe 的 Shape、Layout、Tensor 表示及其与坐标的交互。CuTe Layout 支持通用算法——单一实现可应用于任何复杂的数据布局，只要该布局可以折叠为算法的规范形式。Section 2.6 提供了此类算法及其广泛应用的示例。

### 2.1 Tuple 与 HTuple

Tuple 和 HTuple 概念是本工作的基础数据结构。

**定义 2.1 (Tuple)**：Tuple(T) 是从集合 T 中选取元素组成的有限有序列表。对 X = (X₀, X₁, ..., Xₙ₋₁) ∈ Tuple(X)，定义运算：

- **Rank**：rank(X)，元组长度 n
- **Access**：Xᵢ，Tuple X 的第 i 个元素，0 ≤ i < rank(X)

**定义 2.2 (HTuple)**：HTuple(T) 要么是集合 T 的一个元素，要么是 Tuple(HTuple(T))——即「T 的层次化元组」。对 X ∈ HTuple(X)，定义运算：

- **Rank**：rank(X)。若 X ∈ Tuple，则为元组长度; 否则为 1
- **Access**：Xᵢ。HTuple X 的第 i 个元素，0 ≤ i < rank(X)
- **Depth**：depth(X)。若 X ∈ Tuple，则为 1 + max(depth(X₀), depth(X₁), ...); 否则为 0

例如，以下都是 HTuple(Z) 的实例：

```
31    (16,32)    (3,-8,7)    (2,(4,1),-1)    ((4,6),(3,(2,2),8))
```

在推理 HTuple 时，定义**同余** (congruence) 和**弱同余** (weak congruence) 非常有用。

**定义 2.3 (同余)**：同余 ~ 是 HTuple 上的等价关系。对 P ∈ HTuple(P) 和 S ∈ HTuple(S)：

P ~ S 当且仅当：P ∈ P 且 S ∈ S，或 P, S ∈ Tuple 且 rank(P) = rank(S) 且对所有 i，Pᵢ ~ Sᵢ

例如：(4,8) ~ (5,7)，(4,(2,4)) ~ (7,(3,2))，但 (4,8) 与 (4,(2,4)) 不同余。

**定义 2.4 (弱同余)**：弱同余 ≲ 是 HTuple 上的偏序。对 P ∈ HTuple(P) 和 S ∈ HTuple(S)：

P ≲ S 当且仅当：P ∈ P，或 P, S ∈ Tuple 且 rank(P) = rank(S) 且对所有 i，Pᵢ ≲ Sᵢ

即 P 是 S 的 profile 的粗化 (coarsen)，S 是 P 的 profile 的细化 (refine)。

例如：30 ≲ (a,b) ≲ (**v**,(0,α))，30 ≲ (a,b,c) ≲ ((0,0),0,0)，但 (a,b) 与 (a,b,c) 不弱同余。

### 2.2 Shape

多维数组通常由其 Shape 刻画——描述每个模式的范围的正整数序列。MxN 矩阵的 2D Shape 表示为 (M,N)，自然由坐标 (m,n)（0 ≤ m < M，0 ≤ n < N）索引。一个自然的扩展是将 Shape 表示为正整数的层次化元组。

**定义 2.5 (Shape)**：Shape 是 HTuple(Z⁺)，其中 Z⁺ = {1,2,3,...} 是正整数集。Shape S 的 rank 是 HTuple 的 rank。Shape S 的 size 是其元素之积，记为 |S| = ∏ₖ|Sₖ|。

层次化 Shape 的真正价值在于可以用多种坐标系来索引。考虑不超过 N 的非负整数集：

Z_N = {0, 1, 2, ..., N-1}

CuTe 观察到 2D Shape (M,N) 也可以解释为描述 MN 个 1D 元素，由整数坐标 i（0 ≤ i < MN）索引，只需提供一个双射：

S: Z_{MN} ↔ Z_M x Z_N

在 1D 整数坐标 i ∈ Z_{MN} 和 2D 自然坐标 (m,n) ∈ Z_M x Z_N 之间映射。

类似地，2D Shape (M,NP) 可以解释为层次化 Shape (M,(N,P))，由自然坐标 (m,(n,p)) 索引（0 ≤ m < M，0 ≤ n < N，0 ≤ p < P）。

层次化 Shape 和坐标的直接推论是：张量算法可以为最自然的 Shape 编写（Section 2.6）——COPY 用 1D Shape，GEMM 用 2D Shape，batched-GEMM 用 3D Shape 等——同时仍然接受折叠为与算法规格弱同余的层次化 Shape 张量。数据张量的 Shape（通常表示为扁平整数序列）可以任意折叠为通用张量算法所接受的 Shape。而且，由于张量的每个模式都关联一个 Stride（Section 2.3）来索引数据，这种模式折叠允许表达远超简单连续数组（COPY）或行主序/列主序矩阵（BLAS GEMM）的复杂数据布局（Section 2.4）。

#### 2.2.1 坐标集与兼容性

如前所述，层次化 Shape 允许用多种坐标系索引。这里定义特定 Shape 的坐标集以及 Shape 之间共享坐标集的兼容性概念。

**定义 2.6 (坐标集)**：坐标集是非负整数集 Z_N = {0,1,...,N-1} 或坐标集的笛卡尔积 Z_N x Z_M = Z_{(N,M)}。

例如：

```
Z₆ = {0,1,2,3,4,5}
Z₃ x Z₄ = Z_{(3,4)} = {(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)}
(Z₂ x Z₁) x Z₃ = Z_{((2,1),3)} = {((0,0),0),((1,0),0),((0,0),1),((1,0),1),((0,0),2),((1,0),2)}
```

坐标集 Z_S 正是 Shape S 的自然坐标集。Shape S 的其他坐标集是与 S 兼容且粗化 S 的 Shape 的坐标集。

**定义 2.7 (兼容性)**：兼容性 ⪯ 是 Shape 集上的偏序。对 Shape P 和 S：

P ⪯ S 当且仅当：P ∈ Z⁺ 且 P = |S|，或 P, S ∈ Tuple 且 rank(P) = rank(S) 且对所有 i，Pᵢ ⪯ Sᵢ

即 P 粗化 S，S 细化 P。

兼容性要求两个 Shape 的 size 相同。例如：

```
30 ⪯ (2,15) ⪯ (2,(3,5))    且    30 ⪯ (6,5) ⪯ ((3,2),5)
```

但 (2,(3,5)) 和 ((3,2),5) 尽管 size 相同却不兼容。它们共享兼容 Shape 30。

**定义 2.8**：Shape S 定义了一组兼容坐标集 Z(S)，即所有粗化 S 的 Shape 的坐标集：

Z(S) = {Z_{S'} | S' ⪯ S}

每个 Shape 都有一个整数坐标集 {0,1,...,|S|-1} = Z_{|S|} ∈ Z(S)。每个 rank-r Shape 都有一个 rank-r 坐标集 Z_{(|S₀|,|S₁|,...,|S_{r-1}|)} ∈ Z(S)。

如果 Shape P 粗化 Shape S，则 Z(P) ⊆ Z(S)。即 Shape P 内的任何坐标也是 Shape S 的坐标。

#### 2.2.2 坐标

**定义 2.9 (界内坐标)**：Shape S 的界内坐标（简称坐标）是其某个坐标集的元素，c ∈ Z_{S'} ∈ Z(S)。坐标总是 HTuple(N)。

**定义 2.10 (整数坐标)**：Shape S 的整数坐标是 c̄ ∈ Z_{|S|} ∈ Z(S)。整数坐标总是一个整数。

**定义 2.11 (自然坐标)**：Shape S 的自然坐标是 c̃ ∈ Z_S ∈ Z(S)。自然坐标总是与 Shape 同余的 HTuple(N)，即 c̃ ~ S。

为在界内坐标之间变换，我们在 Shape S 的坐标集上构造枚举来定义**坐标列表**。本文采用反字典序：

(a₀,...,aₙ) < (b₀,...,bₙ) 当且仅当 aₙ < bₙ，或 aₙ = bₙ 且 (a₀,...,aₙ₋₁) < (b₀,...,bₙ₋₁)

反字典序枚举在坐标列表上定义双射。函数 **idx2crd**：

```
idx2crd: Z_{|S|} → Z_{(|S₀|,|S₁|,...,|S_{r-1}|)}

i ↦ (i mod |S₀|, ⌊i/|S₀|⌋ mod |S₁|, ..., ⌊i/∏_{k=0}^{r-2}|Sₖ|⌋)
```

将 Z_{|S|} 的第 i 个坐标（Shape S 的第 i 个整数坐标）映射到 Z_{(|S₀|,...,|S_{r-1}|)} 的第 i 个坐标。

其逆函数 **crd2idx**：

```
crd2idx: Z_{(|S₀|,...,|S_{r-1}|)} → Z_{|S|}

(c₀, c₁, ..., c_{r-1}) ↦ c₀ + c₁·|S₀| + ... + c_{r-1}·∏_{k=0}^{r-2}|Sₖ|
```

**越界坐标**

**定义 2.12 (可容纳坐标)**：Shape S 的可容纳坐标 (admissible coordinate) 是任何弱同余于 Shape 的坐标 c ∈ HTuple(Z)，c ≲ S。

**定义 2.13 (越界坐标)**：Shape S 的越界坐标 (out-of-bounds coordinate) 是任何不在界内的可容纳坐标。

**定义 2.14 (同余坐标)**：Shape S 的同余坐标是任何与 Shape 同余的坐标 c ∈ HTuple(Z)，c ~ S。记为 Z^S = {c ∈ HTuple(Z) | c ~ S}。

即 Z_S 是被 Shape S 界定的有限坐标集，Z^S 是所有与 S 同余的无限坐标集。我们有时用 (\*,\*) 作为 profile 占位符，例如：

```
Z^{(*,*)} = {(a,b) | a,b ∈ Z}
Z^{(*,(*,*))} = {(a,(b,c)) | a,b,c ∈ Z}
```

idx2crd 对所有整数（Z^{|S|} 中的坐标，而非仅 Z_{|S|}）都有定义。当 i ≥ |S| 时，结果总是越界坐标。相反，crd2idx 不能保证越界输入产生越界输出。因此 crd2idx 和 idx2crd 仅在界内坐标上互为逆。

### 2.3 Stride

前一节描述了 Shape、Shape 的层次以及 Shape 的坐标。为了构造数据、线程或其他对象的布局，我们定义从 Shape 内坐标到偏移量的映射。

**定义 2.15 (Stride)**：Shape S 的 Stride D 是与 Shape 同余的 HTuple(D)，S ~ D。该 Stride 定义从自然坐标 c̃ ∈ Z_S 到陪域 D 的映射，由 inner_product 给出：

```
inner_product: Z · D → D,     c · d ↦ cd
inner_product: HTuple(Z) · HTuple(D) → D,     c · d ↦ ∑ᵢ inner_product(cᵢ, dᵢ)
```

大多数情况下，Stride 也是 HTuple(Z)，即 D = Z。inner_product 产生的整数通常解释为数据数组中的偏移量。但 Stride 元素的概念泛化到了整数半模 (integer-semimodule) 的任意元素，这为 Layout 可表示的函数集提供了显著的灵活性。

#### 2.3.1 整数半模

**定义 2.16 (整数半模)**：整数半模是一个集合 M，配备结合加法 M + M → M 和标量乘法 Z · M → M。对 a, b ∈ Z 和 m, n, p ∈ M，加法和标量乘法满足：

- 乘法单位元：1 · m = m
- 加法结合律：m + (n + p) = (m + n) + p
- 乘法结合律：a · (b · m) = (ab) · m

不要求加法单位元和加法逆元，因此 (M,+) 是半群。记为 (M,+,·)。

整数 Z 是整数半模。有理数 Q 是整数半模。域 F₂ = ({0,1}, XOR, AND)（模 2 算术）是整数半模。整数半模的任意笛卡尔积或 HTuple 在逐元素加法和标量乘法下也是整数半模。

一个特别有用的整数半模是 (Z^S, +, ·)，其中 Z^S 是所有与 S 同余的 HTuple(Z) 的集合。例如，rank-2 算术元组的基元素构成整数半模：

```
e₀ = (1,0),  e₁ = (0,1),  Z^{(*,*)} = {a·e₀ + b·e₁ | a,b ∈ Z}
```

整数缩放和加法按元素定义：a·e₀ + b·e₁ = (a,b)。因此 e₀、e₁ 及其任意线性组合可用作 Layout 中的 Stride。通过选择 Z^S 中的 Stride 元素，Layout 可以通过 inner_product 运算生成 Shape S 的自然坐标。

### 2.4 Layout

CuTe 使用 Shape S 和 Stride D 定义 Layout 函数（简称 Layout）。Shape S 定义 Layout 函数的定义域，Stride D 定义 Layout 函数的陪域。

Shape 的另一种解释是：它是从所有坐标列表到自然坐标的映射，这个映射是双射的。类似地，Stride 是从 Shape 的自然坐标到某个陪域的映射：

```
S: Z ↔ Z_S,  对所有 Z ∈ Z(S)
D: Z_S → D
```

↔ 是 idx2crd 和 crd2idx 函数; → 是 inner_product。Shape 和 Stride 的复合定义了 Layout 函数。

**定义 2.17 (Layout)**：Layout **L** = D ∘ S 是 Shape S 和 Stride D（S ~ D）的函数式复合，定义了对每个 Z ∈ Z(S) 的映射 Z → D。

#### 2.4.1 标记与运算

Layout **L** 可以用多种标记表示：

```
(4, (3, 2))       S                          S:D 标记             D ∘ S 标记
(2, (8, 1))       D        (4,(3,2)):(2,(8,1)) = S:D     (2,(8,1)) ∘ (4,(3,2)) = D ∘ S
```

最后一种强调 Shape 和 Stride 本身可以解释为函数，复合后定义 Layout 函数。

Layout 的属性与 Shape 属性一致。对 Layout **L** = S:D 和 **U** = X:Y，定义：

- rank(**L**) = rank(S)：Layout 的 rank 是其 Shape 的 rank
- depth(**L**) = depth(S)：Layout 的 depth 是其 Shape 的 depth
- |**L**| = |S|：Layout 的 size 是其 Shape 的 size
- **L**ᵢ = Sᵢ:Dᵢ：第 i 个子 Layout
- Z(**L**) = Z(S)：Layout 的坐标集是其 Shape 的坐标集
- **L** ~ **U** ⟺ S ~ X：两个 Layout 同余当且仅当 Shape 同余
- **L** ⪯ **U** ⟺ S ⪯ X：两个 Layout 兼容当且仅当 Shape 兼容

Layout 求值示例：对 Layout **L** = ((2,2),(4,2)):((1,8),(2,16)) 和整数坐标 22 ∈ Z₃₂ ∈ Z(**L**)：

```
L(22) = L(2,5) = L((0,1),(1,1)) = 26
```

依次展示了整数坐标、等价 2D 坐标、等价自然坐标和计算出的偏移量。

Layout 的界内定义域是所有坐标 c ∈ Z(**L**)。Layout 也可以对越界坐标求值（类比数组的越界访问，行为未定义）。Layout 的（界内）定义域是有限集 Z(**L**)，扩展定义域是所有弱同余于 S 的 HTuple(Z) 的无限集 Z^**L**。

Layout 的**陪域** (codomain) 和**值域** (image) 有区别。陪域是 D（通常是无限的，如 Z 或 Z^D）。值域是有限的，是 Layout 在定义域上所有坐标的求值结果：

```
image(L) = L(Z(L)) = L(Z_{|L|}) ⊆ codomain(L)
```

#### 2.4.2 Layout 示例

定义了 Shape、Stride 和 Layout 后，下面展示 CuTe Layout 是常见扁平 N 维布局的严格泛化。以下所有 Layout 都与 Shape (4,8) 兼容，画为 4x8 矩阵：

| 名称 | Layout |
|------|--------|
| 列主序 Col-Major | (4,8):(1,4) |
| 行主序 Row-Major | (4,8):(8,1) |
| 列主序带填充 Col-Major Padded | (4,8):(1,5) |
| 列主序交错 Col-Major Interleave | (4,(4,2)):(4,(1,16)) |
| 混合 Mixed | ((2,2),(4,2)):((1,8),(2,16)) |
| 分块广播 Blocked Broadcast | ((2,2),(2,4)):((0,2),(0,4)) |

普通的行主序、列主序和带填充布局可以被 CuTe Layout 直接表示，而交错和混合布局则展示了使用嵌套 Shape 和 Stride 对表示集合的严格扩展。特别是，tile-of-tile 的数据网格可以用 CuTe Layout 直接表示。

使用整数半模 Stride（而非整数 Stride）可以构造更多有用的 Layout：

- **坐标 Layout**：(4,8):(e₀,e₁) 和 (4,(4,2)):(e₁,(e₀,6e₁)) 生成坐标。坐标 Layout 与数据 Layout 对称地变换，常用于检测和判断对数据张量的越界访问。对于 Hopper 和 Blackwell 的 TMA 等消费坐标（而非地址）的指令也很有用。

- **二进制 Swizzle Layout**：(4,(4,3)):(f₁,(f₅,f₁₆))，其中 f 使用 F₂ 整数半模（加法为 XOR）。可用于生成 swizzle 模式，防止共享内存的 bank 冲突。

#### 2.4.3 完备性

每个满足 f(0) = 0 且定义域为有限 Z_N 的函数 f 都可以表示为有限序列 CuTe Layout 的函数式复合。这意味着 CuTe Layout 在函数式复合下是生成集。

这样的函数 f 可以表示为：

```
f ≡ (2,2,2,...):(f(1),f(2),f(3),...) ∘ (3,1):(1,4) ∘ (4,1):(1,6) ∘ ... ∘ (N-1,1):(1,2(N-2))
```

对所有 i ∈ Z_N \ {0}，最右侧的 N-3 个 Layout 将 i 映射到 2^{i-1}，最左侧的 Layout 将 2^{i-1} 映射到 f(i)。

#### 2.4.4 半线性性

Shape-Stride 定义 **L** = D ∘ S，结合泛化的整数半模 Stride，给出了 Layout 函数的线性代数视角：

```
L(c) = (D ∘ S)(c) = d · S(c) = d · c̃
```

Shape 函数是到自然坐标 c̃ ∈ Z^S 的半仿射双射，Stride 函数是自然坐标的线性函数。对两个自然坐标 c̃₀, c̃₁ ∈ Z^S，Layout 函数是线性的：

```
L(α c̃₀ + β c̃₁) = d · (α c̃₀ + β c̃₁) = α(d · c̃₀) + β(d · c̃₁) = α L(c̃₀) + β L(c̃₁)
```

但对任意坐标 c₀, c₁ ∈ Z(S)，Layout 不是线性的，因为 Shape 函数不是线性的。

在自然坐标下，d · c̃ 可以解释为广义矩阵-向量积 **L**(c) = **D** c̃，其中 **D** 是元素取自整数半模 D 的矩阵：

- 整数 Stride（D = Z）时：**D** ∈ Z^{1xn}
- 坐标整数半模 Stride（D = Z^S）时：**D** ∈ Z^{mxn}
- 二进制 Stride（D = F₂^m）时：**D** ∈ F₂^{mxn}

| Layout | 线性形式 | 说明 |
|--------|----------|------|
| ((2,2),(4,2)):((1,8),(2,16)) | r = [1 8 2 16] [c₀ c₁ c₂ c₃]ᵀ | 整数 Stride 是 1xn Z-矩阵的列 |
| (4,(4,2)):(e₁,(e₀,6e₁)) | [r₀ r₁]ᵀ = [[0 1 0]; [1 0 6]] [c₀ c₁ c₂]ᵀ | 坐标 Stride 是 mxn Z-矩阵的列 |
| (4,4):(f₁,f₅) | [r₀ r₁ r₂ r₃]ᵀ = [[1 0 1 0]; [0 1 0 1]; ...] [...] | 二进制 Stride 是 mxn F₂-矩阵的列 |

BPC 和 BMMC 变换 [13, 14, 15, 11] 中研究的变换形式为 f(**v**) = **A****v** + **b**，其中 **A** 是 mxn 二进制矩阵，**v** 是长度 n 的二进制向量，**b** 是长度 m 的二进制向量，所有算术在 F₂ 上执行。

CuTe Layout 代数中定义的运算是从线性代数泛化而来的。CuTe Layout 的群复合可解释为矩阵乘法的泛化。CuTe Layout 的右逆和左逆可解释为 Moore-Penrose 伪逆的泛化。CuTe Layout 代数可看作 BPC 和 BMMC 运算超越 F₂ 域的泛化——CuTe Layout 代数主要受张量的异构数据布局中常见的通用整数 Stride 驱动。

### 2.5 Tensor

最后，定义 CuTe 的核心对象——Tensor，通过将 Layout 绑定到 Accessor 来定义。Accessor 是 Layout 的随机访问指针式对象。

**定义 2.18 (Accessor)**：Accessor 是支持偏移和解引用操作的对象：

```
e + d → e'     偏移 accessor e 以 d ∈ D，得到另一个 accessor e'
*e → v         解引用 accessor e，得到值 v
e[d] → *(e+d)  下标运算符
```

D = Z 时，常见的 Accessor 实现包括裸指针（如 T\*）、数组（如 T[N]）和随机访问迭代器（如 `thrust::counting_iterator`、`thrust::transform_iterator` 等）。

**定义 2.19 (Tensor)**：Tensor 由 Accessor e 与 Layout **L** 复合定义，表示为 T = e ∘ **L**。对坐标 c ∈ Z(**L**)，Tensor 先用 Layout 映射坐标到陪域 D，然后偏移 Accessor 并解引用得到值：

```
T(c) = (e ∘ L)(c) = *(e + L(c)) = e[L(c)]
```

大多数 Tensor 是使用内存地址作为 Accessor 的数据布局。例如，内存地址 p 可用作指针 Accessor {p}：

```
{p} + b → {p+b}
*{p} → *p
T = {p} ∘ L
```

所有 Layout 也可以通过与 counting iterator {a}（解引用返回存储的偏移量 a ∈ Z）复合来变成**隐式 Tensor**：

```
{a} + b → {a+b}
*{a} → a
T = {a} ∘ L
```

类似地，坐标 Layout 可以与坐标 Accessor {(a,b)} 复合来变成隐式 Tensor：

```
{(a,b)} + (c,d) → {(a+c, b+d)}
*{(a,b)} → (a,b)
T = {(a,b)} ∘ L
```

#### 2.5.1 切片 (Slicing)

Tensor 可以做**完全求值**或通过**切片**做**部分求值**。CuTe Tensor 可以看作带偏移的 Layout，任意切片可沿自然坐标的任意模式执行。

- **完全求值**：用完整坐标 c 应用时，得到一个值
- **部分求值（切片）**：用不完整坐标 c = c' + c\* 切片时（c\* 是未指定部分），得到新 Tensor：

```
T(c) = (e ∘ L)(c' + c*) = (e + L(c')) ∘ L(c*) = e' ∘ L(c*) = T'(c*)
```

其中 L(c') 可以完全求值并累加到 e 中，L(c\*) 是 L 的未求值子 Layout。切片创建子 Tensor，可进一步求值或操作。

许多张量库（numpy、torch、MATLAB）支持类似切片语法，如 `my_matrix[2,:]` 提取第二行。这些库还支持范围切片，如 `my_matrix[2:4,1:3]`。CuTe **不支持**范围切片，原因如下：

1. 范围切片无法表达所有切片。某些切片（如沿层次化子模式的切片）无法用范围切片表达。

2. 范围切片助长了如下模式：
   ```python
   thr_data = my_data[thr_id*TILE_SIZE:(thr_id+1)*TILE_SIZE]
   ```
   这将 TILE_SIZE（通常是编译时静态常量）和 thr_id（运行时动态索引）混为一谈。CuTe 偏好先排列再切片的两阶段方法：
   ```python
   tiled_data = logical_divide(my_data, TILE_SIZE)  # (TILE_SIZE, NumTiles)
   thr_data = tiled_data[None, thr_id]               # TILE_SIZE
   ```
   分离了 TILE_SIZE 和 thr_id，更易于编译器推理和传播静态信息。

3. 范围切片可以表达无法用 CuTe Layout 表示的切片——由于请求的 tile 与 Tensor 布局不兼容。CuTe 偏好在排列-重塑阶段（复合）而非切片阶段检测和暴露这类错误。

### 2.6 应用

CuTe 提供了紧凑的表示，其可表达的 Layout 集合严格大于传统扁平 Shape + Stride 表示。作为对比，CUTLASS v2 [22] 和 ThunderKittens [10] 等库逐一手动实现每个布局。CUTLASS v2 代码库包含近 300 个独立实现的布局，分布在 87 个文件中，共约 55,000 行代码。而 CuTe 的核心 Layout 表示加上相关代数只需约 **3,000 行代码**，就能表示 CUTLASS v2 中所有 300 个布局及更多。CuTe 中实现的算法可以对输入的 rank 或 Shape 施加约束，但仍兼容满足这些前提条件的任意布局。

CuTe 现在是 CUTLASS v3、CUTLASS v4 和 CuTe DSL 的基础，它们都建立在 CuTe 的核心 Layout 表示和代数之上。

#### 2.6.1 COPY

用 CuTe Tensor 编写的通用 COPY 算法：

```cpp
// @pre size(src) == size(dst)
template <class TS, class SLayout, class TD, class DLayout>
void copy(Tensor<TS,SLayout> const& src,   // N
          Tensor<TD,DLayout>      & dst)   // N
{
    for (int i = 0; i < size(dst); ++i) {
        dst(i) = src(i);
    }
}
```

```python
# @pre size(src) == size(dst)
def copy(src: Tensor,  # N
         dst: Tensor): # N
    for i in range(size(dst)):
        dst[i] = src[i]
```

前置条件指定两个 Tensor 的 size 相等。这个简单的 COPY 实现通过改变源和目标 Tensor 的 Layout 就能涵盖广泛的应用：

| 应用 | 源 Layout | 目标 Layout |
|------|-----------|-------------|
| 1D 数组 | 8:1 | 8:1 |
| ND 数组 | (8,2,3):(1,16,32) | (8,2,3):(1,16,32) |
| Gather | (2,3,2):(42,1,128) | 12:1 |
| Scatter | 12:1 | (2,3,2):(42,1,128) |
| Broadcast | 7:0 | 7:1 |
| Constant | 7:0 | 7:0 |
| Transpose | (8,3):(1,8) | (8,3):(3,1) |
| Tensor Transpose | (8,(3,5)):(1,(57,8)) | (8,15):(1,8) |

任何 rank 的 Tensor 都可以复制到任何其他 rank 的 Tensor。COPY 是一个 rank-1 算法，不受参数 rank 的影响——这是秩无关编程 (rank-agnostic programming) 的体现。

#### 2.6.2 GEMM

用 CuTe Tensor 编写的通用 GEMM 算法：

```cpp
// @pre M: size<0>(A) == size<0>(C)
// @pre N: size<0>(B) == size<1>(C)
// @pre K: size<1>(A) == size<1>(B)
template <class TA, class ALayout, class TB, class BLayout, class TC, class CLayout>
void gemm(Tensor<TA,ALayout> const& A,   // (M,K)
          Tensor<TB,BLayout> const& B,   // (N,K)
          Tensor<TC,CLayout>      & C)   // (M,N)
{
    for (int k = 0; k < size<1>(B); ++k)
        for (int n = 0; n < size<0>(B); ++n)
            for (int m = 0; m < size<0>(A); ++m)
                C(m,n) += A(m,k) * B(n,k);
}
```

```python
# @pre M: size[0](A) == size[0](C)
# @pre N: size[0](B) == size[1](C)
# @pre K: size[1](A) == size[1](B)
def gemm(A: Tensor,  # (M,K)
         B: Tensor,  # (N,K)
         C: Tensor): # (M,N)
    for k in range(size[1](B)):
        for n in range(size[0](B)):
            for m in range(size[0](A)):
                C[m,n] += A[m,k] * B[n,k]
```

这个简单的 GEMM（及 batched-GEMM 扩展）通过改变 Tensor 的 Layout 即可涵盖多种应用：

| 应用 | A-Layout | B-Layout | C-Layout |
|------|----------|----------|----------|
| NT GEMM | (M,K):(1,lda) | (N,K):(1,ldb) | (M,N):(1,ldc) |
| TN GEMM | (M,K):(lda,1) | (N,K):(ldb,1) | (M,N):(1,ldc) |
| NTT GEMM | (N,K):(1,ldb) | (M,K):(1,lda) | (N,M):(1,ldc) |
| BLIS GEMM | (M,K):(dma,dka) | (N,K):(dmb,dkb) | (M,N):(dmc,dnc) |
| GETT | ((M1,M2),K):((1,W),X) | (N,K):(K,1) | ((M1,M2),N):((1,Y),Z) |
| CONV | (K,(C,T,R,S)):DA | ((N,Z,P,Q),(C,T,R,S)):DB | (K,(N,Z,P,Q)):DC |

通过抽象 fused-multiply-add 操作并提供足够强大的分块工具，该算法可以在架构层次的每一级递归地应用。

---

## 3 Layout 代数

CuTe Layout 虽然只是所有可能函数空间的子集，但它能表达的 Layout 函数集合严格大于 BLAS、`torch.tensor`、`numpy.ndarray` 等库中传统的扁平 Shape + Stride 表示。

CuTe Layout 的一个关键效用在于可以通过操作和组合来创建新 Layout。这通过定义在 Layout 上的一组核心代数运算来实现，这些运算进一步可用于构造更高层的运算。

本节定义 Layout 同态 (homomorphism)——接受 CuTe Layout 并产生满足某些函数式性质的新 CuTe Layout 的运算。

### 3.1 拼接 (Concatenate)

Layout 可以表示为其子 Layout 的拼接：

```
L = S:D
  = (S₀,S₁,...,Sₙ):(D₀,D₁,...,Dₙ)
  = (S₀:D₀, S₁:D₁, ..., Sₙ:Dₙ)
  = (L₀, L₁, ..., Lₙ)
```

使得：

对所有 c = (c₀,c₁,...,cₙ) ∈ Z(**L**)，**L**(c) = **L**₀(c₀) + **L**₁(c₁) + ... + **L**ₙ(cₙ)

拼接的可容纳条件是：所有子 Layout 的陪域必须包含在同一个整数半模中。例如，任意两个整数 Stride 的 Layout 可以拼接，但 4:2 和 3:e₀ 不能。

注意到 Layout 的每个子 Layout 也是一个 Layout，任何操作 Layout 的代数运算同样可以应用于单个子 Layout——称为**按模式运算** (by-mode operation)。这用组合子 (combinator) 表示：

```
A ★ ⟨B, C⟩ = (A₀, A₁) ★ ⟨B, C⟩ = (A₀ ★ B, A₁ ★ C)
```

其中 ★ 是某个二元 Layout 运算，⟨⟩ 表示 Layout 的元组（区别于拼接）。

### 3.2 合并 (Coalesce)

给定 Layout **A**，合并 Layout **R** 满足：

- 一致的整数定义域：|**R**| = |**A**|
- 扁平化或整数 Shape：depth(**R**) ≤ 1
- 一致的整数求值：对所有 c̄ ∈ Z_{|**A**|}，**R**(c̄) = **A**(c̄)

coalesce 运算「化简」Layout **A**：将其视为整数上的函数，可能折叠 Shape 为更浅的表示。虽然过程可能移除 rank 和层次信息、修改坐标集、合并多个模式，但保证 Layout 作为整数坐标上的映射保持函数等价。

实践中，引用合并 Layout 通常指达到最小 rank 的合并。

示例：(2,(1,6)):(1,(6,2)) 合并后为 12:1。也可按模式合并：

```
coalesce((2,(1,6)):(1,(6,2)), ⟨*,*⟩) = (coalesce(2:1), coalesce((1,6):(6,2))) = (2:1, 6:2) = (2,6):(1,2)
```

类似地，((4,3),5):((15,1),3) 合并为 (4,15):(15,1)，按模式合并仍为 ((4,3),5):((15,1),3)（因各模式单独无法进一步合并）。

### 3.3 复合 (Composition)

给定 Layout **A** 和 **B**，群复合 Layout **R** = **A** ∘ **B** 满足：

- 定义域兼容性：**B** ⪯ **R**
- 函数式复合：对所有 c ∈ Z(**B**)，**R**(c) = **A**(**B**(c))

**B** 决定结果 Layout 的 Shape 和坐标集（定义了 **R** 的定义域），**A** 决定 **R** 的陪域。兼容性条件保证 **B** 的所有坐标也可用作 **R** 的坐标。可容纳条件要求 **B** 的陪域与 **A** 的定义域兼容。

#### 3.3.1 复合的性质

**单位 Layout**：对任意 Shape S，单位 Layout **I**_S 满足对所有 c ∈ Z_S，**I**_S(c) = c。例如，以下都是单位 Layout **I**₂₄：

```
24:1,    (4,6):(1,4),    (3,(4,2)):(1,(3,12))
```

对陪域为 Z_D 的 Layout **B**，任何 **I**_D 都是群复合的左单位元：**I**_D ∘ **B** = **B**。

对定义域为 Z_S 的 Layout **A**，Shape 为 S 的 **I**_S 是群复合的右单位元：**A** ∘ **I**_S = **A**。

**结合律**：给定 Layout **A**、**B**、**C**，当 image(**C**) ⊆ Z(**B**) 且 image(**B**) ⊆ Z(**A**) 时：

**A** ∘ (**B** ∘ **C**) = (**A** ∘ **B**) ∘ **C**

注意，当上述条件不满足时复合仍然可能，但结合律可能不成立。

#### 3.3.2 求值与约束

群复合的求值可从 Layout 求值运算 idx2crd 和 inner_product 构造性推导。

**基本情形**：设 **B** = s:d（s ∈ Z⁺，d ∈ N），**A** = S:D = (S₀,S₁,...,S_R):(D₀,D₁,...,D_R)（Sᵣ ∈ Z⁺，Dᵣ ∈ D），定义 S 的前缀积 S̄ᵣ = ∏_{k=0}^{r-1} Sₖ。

则对每个 i = 0,1,...,s-1，复合 **R** = **A** ∘ **B** 求值如下：

```
R(i) = (A ∘ B)(i) = ∑_{r=0}^{R-1} (⌊i·d/S̄ᵣ⌋ mod Sᵣ) · Dᵣ + ⌊i·d/S̄_R⌋ · D_R
```

施加**步幅可除条件**：

对每个 r = 0,...,R，S̄ᵣ | d 或 d | S̄ᵣ

定义 δᵣ = ⌈d/S̄ᵣ⌉，ρᵣ = ⌈S̄ᵣ/d⌉，则结果 Shape 和 Stride 为：

```
S'ᵣ = Sᵣ / δᵣ
D'ᵣ = Dᵣ · δᵣ
```

为满足兼容性条件 |S'| = s，还需施加 **Shape 可除条件**：

对每个 r = 0,...,R，⌈S̄ᵣ/d⌉ | s

并修改最后一个 Shape 为 S'_R = s/ρ_R。

**归约情形：分配律**。更一般的复合通过对 **B** 的子 Layout 分配来表达：

```
A ∘ B = A ∘ (B₀, B₁, ...) = (A ∘ B₀, A ∘ B₁, ...)
```

当以下条件之一满足时成立：

1. **B** 的陪域与 Shape S 同余（此时 S 的作用是恒等变换）
2. 所有基本子 Layout 的 Stride 满足步幅可除条件，且所有基本子 Layout 具有互相分离的值域

**归约情形：坐标**。当 **B** 产生坐标时，需将问题归约为基本情形。若 **B** 的 Stride 是坐标整数半模的缩放基元素 d = a·eᵢ，则：

```
A ∘ B = A ∘ (a·eᵢ) ∘ s = Aᵢ ∘ a ∘ s = Aᵢ ∘ B'
```

其中 B' = s:a 回到基本情形。

#### 3.3.3 直觉与可除性

rank-1 左操作数 **A** 的复合是简单的：

```
(S₀):(D₀) ∘ s:d = s:D₀·d
```

无需可除性检查。群复合即使 image(**B**) ⊄ Z(**A**) 也可能进行：

```
7:11 ∘ 3:4 = 3:44
```

**B** 无需互不相交也可满足分配律：

```
7:11 ∘ (3,5):(6,3) = (3,5):(66,33)
```

对更高 rank 的 **A**，直觉策略包含两步：

1. 从 **A** 中「除去」前 d 个元素，确定一个产生 **A** 中每隔 d 个元素的中间 Layout
2. 将中间 Layout 的 size 固定为 s——「保留」前 s 个元素

例如：

```
(4,6,8,10):(2,3,5,7) ∘ 6:12

步骤1: 除去前12个元素 → (1,2,8,10):(X,9,5,7)
步骤2: 保留前6个元素 → (2,3):(9,5)
```

**可除性违反**。某些复合违反步幅可除条件或 Shape 可除条件：

```
(4,6,8):(2,3,5) ∘ 6:3   ← 违反步幅可除 (S̄₁=4 与 d=3 不可除)
(4,6,8):(2,3,5) ∘ 6:1   ← 违反 Shape 可除 (⌈4/1⌉=4 不整除 s=6)
```

这意味着 CuTe Layout 在群复合下不严格封闭。但实践中，可除性违反通常源于概念性应用错误、布局/硬件不兼容或编程错误。这些错误往往可在编译时捕获，从而促进程序安全性和开发效率。

**表面违反**。某些看似违反的情况可通过合并 **A**（Section 3.2）和截断来解决：

```
(4,2,8):(3,12,97) ∘ 3:3 → coalesce → (8,8):(3,97) ∘ 3:3 → truncate → (8):(3) ∘ 3:3 = 3:9
```

#### 3.3.4 应用：分区示例

复合是 CuTe Layout 代数的核心，支持重塑、重步幅、排列、分区、分块和子 Layout 提取等操作。

考虑将 Shape 为 (8,8) 的数据 Layout 用 Ampere Tensor Core 的线程-值模式分区。该指令的线程-值分区模式可表示为 Layout：

```
ThrValLayoutC: ((4,8),2):((16,1),8)
```

将 (thread_idx, value_idx) 映射到 8x8 矩阵中的 1D 坐标。

任何 8x8 数据 Layout 都可以与 ThrValLayoutC 复合来做分区。每次复合产生与 Shape (32,2) 兼容的 Layout，定义 (thread_idx, value_idx) 到数据偏移的映射：

| 数据名称 | 数据 Layout 8x8 (A) | TV Layout 32x2 (B) | 结果 32x2 (R) |
|----------|---------------------|---------------------|---------------|
| ColMajor | (8,8):(1,8) | ((4,8),2):((16,1),8) | ((4,8),2):((16,1),8) |
| RowMajor | (8,8):(8,1) | ((4,8),2):((16,1),8) | ((4,8),2):((2,8),1) |
| Padded | (8,8):(1,9) | ((4,8),2):((16,1),8) | ((4,8),2):((18,1),9) |
| Swizzled | (8,8):(f₁,f₉) | ((4,8),2):((16,1),8) | ((4,8),2):((f₁₈,f₁),f₉) |
| Coordinate | (8,8):(e₀,e₁) | ((4,8),2):((16,1),8) | ((4,8),2):((2e₁,e₀),e₁) |

常见模式代码：

```python
smem_data = Tensor(MyAccessor, MyLayout8x8)          # Tensor: Coord → Offset
tv_layout = Layout(((4,8),2), ((16,1),8))            # TV Layout: (Thr,Val) → Coord
smem_tv   = composition(smem_data, tv_layout)         # Compose: (Thr,Val) → Offset
smem_v    = smem_tv[thr_id, None]                     # 按线程切片得到子张量
copy(smem_v, rmem_data)                               # 拷贝到寄存器张量
```

这种模式在 SIMD 编程中极为常见：每个处理单元获得父数据的对称分区。CuTe 认为，任意分区可以定义为**复合（排列和/或重塑）后加切片**。这种先复合再切片的模式将分区模式（编译时元数据）与实际数据切片（运行时索引）分离，更有利于传播静态信息和减少运行时开销。

#### 3.3.5 按模式复合与 Tiler

群复合可按模式应用，用组合子表示：

```
A ∘ ⟨B, C⟩ = (A₀, A₁) ∘ ⟨B, C⟩ = (A₀ ∘ B, A₁ ∘ C)
```

⟨⟩ 中的 Layout 元组称为 **Tiler**。

**定义 3.1 (Tiler)**：Tiler 是 HTuple(Tile)，其中每个 Tile 是 Layout S:D 或整数 S（等价于 Layout S:1）。

所有 Layout 都是 Tiler，所有整数都是 Tiler。这使得 Shape（如 (4,8)）可以用作 Tiler。常见情形中，Shape 是提取和操作子 Layout 的便捷 Tiler。

以下对象在复合中等价：

```
(4,8) ≡ ⟨4,8⟩ ≡ (4:1, 8:1) ≡ (4,8):(e₀,e₁)
```

### 3.4 逆 (Inverse)

Layout 可以是单射、满射或双射的，相应地有右逆、左逆、完全逆和拟逆。当 Layout 解释为从坐标到偏移的函数时，逆 Layout 可解释为从偏移到坐标的函数。Layout 逆在确定布局中某些偏移出现的位置、提取特定偏移组或确定两个布局的公共子 Layout 时非常有用。

#### 3.4.1 右逆

Layout **L**: Z_{|**L**|} → D 的右（伪）逆是单射 Layout **L**^‡: D_{**L**^‡} → Z_{|**L**|} 满足：

对所有 k ∈ D_{**L**^‡}，**L**^‡(**L**(**L**^‡(k))) = **L**^‡(k)

在常见情形 D = Z 下，回到经典右逆定义：

对所有 k ∈ Z_{|**L**^‡|}，**L**(**L**^‡(k)) = k

如果 Layout **L** 有右逆，则 |**L**^‡| ≤ |**L**|。实践中，引用右逆通常指最大 size 的右逆。示例：

| Layout **L** | 右逆 **L**^‡ | 说明 |
|------------|------------|------|
| (4,8):(1,4) | 32:1 | |
| (4,8):(8,1) | (8,4):(4,1) | |
| (3,7,5):(5,15,1) | (5,21):(21,1) | 非 2 的幂次也可以 |
| (4,8):(1,5) | 4:1 | 非连续值域导致更小的结果 |
| ((2,2),(2,4)):((0,1),(0,2)) | (2,2):(4,8) | stride-0 模式不贡献 |
| (4,8):(e₀,e₁) | (4,8):(1,4) | 结果定义域与陪域 Z^{(\*,\*)} 同余 |

当 Layout **L** 是 Z_{|**L**|} 上的双射时，右逆也是完全逆 **L**^{-1}。具有完全逆的 Layout 称为**紧凑** (compact) Layout。

#### 3.4.2 应用：向量化示例

右逆在检查数据布局、确定连续元素是否存在以及在哪里存在方面极为有用。

直接示例：(4,8):(1,4) 和 (4,8):(8,1) 的右逆分别为 32:1 和 (8,4):(4,1)，因为它们的右逆 size 均为 32，说明两种布局都索引到 32 个连续的物理元素。

更进一步的例子是 CuTe 的**向量化拷贝** (vectorizing-copy)：尝试找出两个 Tensor 之间可同时拷贝的最大元素数。右逆允许 CuTe 确定两个 Layout 之间的最大公共子 Layout，结合硬件能力和指针/步幅的物理对齐信息，可以代数化地确定可安全向量化的元素数量和位置。

一般地，对 Layout **A**: Z(**A**) → Z_α 和 **B**: Z(**B**) → Z_β（|**A**| = |**B**|），希望找到最大整数 K 使得坐标匹配：

```
对所有 k ∈ Z_K，A^‡(k) = B^‡(k)
```

可以通过计算 **A** ∘ **B**^‡ = (**I**_K, **X**) 或 **B** ∘ **A**^‡ = (**I**_K, **Y**) 来高效得到——K 是结果中 stride-1 模式（单位部分）的 size。

互相连续元素的整数坐标由以下 Layout 给出：

```
A^‡ ∘ ⌊B ∘ A^‡⌋_K = ⌊A^‡⌋_K: Z_K → Z_{|A|}
B^‡ ∘ ⌊A ∘ B^‡⌋_K = ⌊B^‡⌋_K: Z_K → Z_{|B|}
```

其中 ⌊·⌋_K 是截断到 size K 的操作。实践中，考虑 Section 2.6.1 的 COPY 参考实现——它可以在内部排列迭代顺序而不改变可观察行为。选择与上述 Layout 匹配的排列，所有连续元素就被排到第一模式，可由向量化指令特殊处理。这为 **AutoVectorization** 优化提供了框架。

#### 3.4.3 左逆

Layout **L**: Z_{|**L**|} → D 的左（伪）逆是 Layout **L**^†: D_{**L**^†} → Z_{|**L**|} 满足：

对所有 k ∈ Z_{|**L**|}，**L**(**L**^†(**L**(k))) = **L**(k)

当 **L** 是单射时，回到经典左逆定义：

对所有 k ∈ Z_{|**L**|}，**L**^†(**L**(k)) = k

示例：

| Layout **L** | 左逆 **L**^† | 说明 |
|------------|------------|------|
| (4,8):(1,4) | 32:1 | |
| (4,8):(8,1) | (8,4):(4,1) | 连续值域时与右逆相同 |
| (4,8):(1,5) | (5,8):(1,4) | 非连续值域导致更大的结果 |
| ((2,2),(2,4)):((0,2),(0,4)) | (2,2,4):(0,2,8) | 结果不唯一（任意 mode-0 Stride 均可） |
| (4,(4,2)):(e₁,(e₀,6e₁)) | (4,(6,2)):(4,(1,16)) | 更大的结果，定义域兼容 |

#### 3.4.4 应用：可容纳性检查示例

左逆用于确定数据 Layout 中特定偏移的存在性和位置。

Blackwell 的 TMEM 加载/存储指令在预定义的特定偏移集处访问 TMEM。TMEM 的物理寻址是最多 512 列 x 128 lane 的 2D 网格。跨 32 位列递增 TMEM 地址 1，跨 lane 递增 16384。

定义记录每条指令访问的所有物理偏移的 Layout：

| 指令 | (InstrCOL, InstrLANE) → TMEM Offset |
|------|--------------------------------------|
| tcgen05.32x32b.x1 | (1,128):(1,16384) |
| tcgen05.32x32b.x2 | (2,128):(1,16384) |
| tcgen05.16x256b.x1 | (8,(16,4)):(1,(16384,32·16384)) |

给定数据 Layout **A**: Z(**A**) → Z_α（逻辑坐标到数据偏移）和指令 Layout **T**: Z(**T**) → Z_β（指令坐标到数据偏移），希望确定 **T**(i) 在 **A** 的值域中的存在性和位置。可通过计算 **A** 的左逆来高效检查：

```
A(A^†(T(i))) = T(i)
```

即：**T**(i) 的所有偏移都在 **A**^† 的定义域中，且所有坐标 **A**^†(**T**(i)) 唯一且在 **A** 的定义域中。Layout **A**^† ∘ **T** 将指令坐标映射到逻辑数据坐标，可结合 `zipped_divide` 等操作将数据 Layout 分区为与指令访问模式对应的子 Layout。

### 3.5 补 (Complement)

Layout **L**: Z(**L**) → D 的补是 Layout **L**\*: Z(**L**\*) → D，满足：

- **弱同余于陪域**：D ≲ **L**\*
- **不相交的值域**：对所有 b ∈ Z(**L**)，对所有 a ∈ Z^{**L**\*} \ {0}，**L**(b) ≠ **L**\*(a)
- **有序值域**：对所有 a ∈ Z_{|**L**\*|} \ {0}，**L**\*(a-1) < **L**\*(a)

补是一个在原 Layout 的陪域中生成元素但不在值域中的 Layout。在复合中，Layout **L** 常用作对另一个 Layout 的间接引用，补 **L**\* 则指向 **L** 未覆盖的元素。补运算支持通过 logical divide 拆分 Layout，以及通过 logical product 重复和扩展 Layout。

示例：

| Layout **L** | 补 **L**\* | 说明 |
|------------|----------|------|
| (4,8):(1,4) | 1:32 | |
| (4,8):(8,1) | 1:32 | 输入顺序无关 |
| (4,8):(1,5) | 1:40 | |
| (4,8):(1,8) | (2,1):(4,64) | 「填补空隙」 |
| ((2,2),(2,4)):((0,2),(0,4)) | (2,1):(1,16) | |
| (4,8):(e₀,e₁) | (1,1):(4e₀,4e₁) | 结果定义域与陪域同余 |

#### 3.5.1 应用：逻辑积 (Logical Product)

两个 Layout **A** 和 **B** 的逻辑积是一个 Layout **R**，其中「Layout **B** 的每个元素被替换为 Layout **A** 的唯一偏移副本」。

定义逻辑积为两个 Layout 产生 rank-2 Layout 的函数：

```
A ⊗ B = (A, A* ∘ B)
```

其中 **A**\* 是 **A** 的补。结果第一模式恰好是输入 **A**，第二模式在 Shape 和元素顺序上与输入 **B** 兼容。第一模式通常称为「tile」，它在「grid」Layout **B** 上重复。

示例：行主序 (3,4):(4,1) tile 在列主序 (2,5):(1,2) grid 上的逻辑积：

```
(3,4):(4,1) ⊗ (2,5):(1,2) = ((3,4),(2,5)):((4,1),(12,24))
```

其中 1:12 是 (3,4):(4,1) 的补。

**相关积**：blocked_product 要求 **A** 和 **B** 的 rank 相同，用 zip 操作组合对应的行模式和列模式。raked_product 类似但 zip 顺序相反。

#### 3.5.2 应用：逻辑除 (Logical Divide)

两个 Layout **A** 和 **B** 的逻辑除是一个 Layout **R**，其中「Layout **A** 被拆分为两部分：**B** 指向的元素和剩余元素」。

定义逻辑除为两个 Layout 产生 rank-2 Layout 的函数：

```
A ⊘ B = A ∘ (B, B*_{|A|})  = A ∘ B★
```

其中 B\*_{|**A**|} 是 **B** 相对于 **A** 的 size 取的补。由于此运算旨在保留 **A** 的所有元素，要求 B★ = (**B**, B\*_{|**A**|}) 对 Z_{|A|} 满射。此外，要求补 B\* 「完备」Layout **B**，即 B★ 有一个满足自反条件的广义逆 B⁺：

```
B★ B⁺ B★ = B★
B⁺ B★ B⁺ = B⁺
```

结果 Layout 的第一模式称为「tile」，第二模式称为「grid」或「tiling」。**B** 参数称为「tiler」。

示例：提取 Layout 中每第三个元素，并保留「剩余」部分：

```
24:3 ⊘ 8:3

即 24:3 ∘ (8:3, 3:1)，其中 3:1 是 8:3 相对于 size 24 的补。

结果: (8,3):(9,3)
第一模式: 每第三个元素（如请求）
第二模式: 包含这些 tile 在原 Layout 中的 3 个副本
```

**相关除法**：逻辑除常按模式应用：

```
A ⊘ ⟨B, C⟩ = (A₀, A₁) ⊘ ⟨B, C⟩ = (A₀ ⊘ B, A₁ ⊘ C)
```

zipped_divide 按模式做逻辑除后将同类模式 zip 在一起。

示例：给定 Layout (8,16):(20,1)，用 tiler ⟨4:1, 8:2⟩ 提取「tile」（每列取 4 个连续元素，每行取隔一的 8 个元素）：

```
(8,16):(20,1) ⊘ ⟨4:1, 8:2⟩ = ((4,2),(8,2)):((20,80),(2,1))
```

蓝色为 tile 模式，黑色为 rest 模式。结果 Layout 仍为 8x16，tiler 所指向的元素被排列到前 4x8 块。

zipped_divide 进一步将 (Tile, Grid) 模式 zip 后可用块标识符切片提取特定 tile：

```
tiled_tensor = zipped_divide(data, tiler)   # (Tile, Grid)
block_data   = tiled_tensor[None, block_id] # Tile
```

这与 Section 3.3.4 的线程-值分区模式很相似，但只需指定 tile 模式，缺失的 Grid 模式由补自动计算。

---

## 4 结论

本文介绍了 CuTe，一种用于表示和操作张量 Layout 的数学框架，以应对现代 GPU 架构日益增长的复杂性。通过层次化 Layout 表示，CuTe 提供了一种通用方法来编写和管理当代硬件专用张量指令所需的复杂数据布局和分区模式。通过丰富的代数运算，CuTe 为开发高性能线性代数内核时操作和生成新 Layout 提供了系统化方法。表示和代数共同支持复杂布局和分区、关注点分离以及静态分析与优化。

CuTe Layout 表示的表达力已在多个关键应用中得到验证。CuTe 支持将通用张量缩并 (GETT) 和卷积 (CONV) 表示为通用矩阵乘法 (GEMM) 的实例，促进代码重用和算法统一。CuTe 使矩阵可以用 (m,n) 坐标索引而不依赖于具体数据布局，通用算法保持与具体布局正交，而要求特定布局的算法或指令可以静态地检查、验证和推理其张量参数。

CuTe Layout 代数支持对张量布局做静态分析和变换，以实现和验证通用算法。例如：

- 为 COPY 操作推导最大向量化机会
- 验证 MMA 和 COPY 指令的输入输出布局
- 推导包含 bank 冲突规避 swizzle 模式的最优共享内存布局

这些能力将布局管理从容易出错的手动过程转变为系统化、可验证的方法论。

CuTe 的实际影响已通过在生产系统中的成功部署得到证明，尤其是作为 NVIDIA CUTLASS v3 和 v4 库的基础。CUTLASS 和 FlashAttention 等应用中的性能评估表明，CuTe 的抽象不引入性能开销，同时显著加速了软件开发。

展望未来，CuTe 的数学基础和代数方法使其成为适应未来架构创新的可持续框架。通过将布局关注点从算法逻辑中分离，并提供强大的编译时推理能力，CuTe 为张量中心的编程建立了一套方法论，能够适应各类计算系统和应用的需求。

---

## 参考文献

- [1] NVIDIA. *Volta Architecture Whitepaper*, 2017.
- [2] NVIDIA. *Turing Architecture Whitepaper*, 2018.
- [3] NVIDIA. *Ampere Architecture Whitepaper*, 2021.
- [4] NVIDIA. *Hopper Architecture Whitepaper*, 2023.
- [5] NVIDIA. *Blackwell Architecture Whitepaper*, 2025.
- [6] Blackford et al. An updated set of basic linear algebra subprograms (BLAS). *ACM TOMS*, 28(2):135–151, 2002.
- [7] Van Zee & van de Geijn. BLIS: A framework for rapidly instantiating BLAS functionality. *ACM TOMS*, 41(3), 2015.
- [8] Shi et al. Tensor Contractions with Extended BLAS Kernels on CPU and GPU. *IEEE HiPC*, 2016.
- [9] Aharoni et al. HeLayers: A tile tensors framework for large neural networks on encrypted data. *PoPETs*, 2023.
- [10] Spector et al. ThunderKittens: Simple, fast, and adorable AI kernels, 2024. arXiv:2410.20399.
- [11] Zhou et al. Linear Layouts: Robust code generation of efficient tensor computation using F₂, 2025. arXiv:2505.23819.
- [12] Tillet et al. Triton: an intermediate language and compiler for tiled neural network computations. *MAPL*, 2019.
- [13] Edelman et al. Index transformation algorithms in a linear algebra framework. *IEEE TPDS*, 5(12), 1994.
- [14] Cormen. Fast permuting on disk arrays. *JPDC*, 17(1), 1993.
- [15] Bouverot-Dupuis & Sheeran. Efficient GPU implementation of affine index permutations on arrays. *FHPNC*, 2023.
- [16] NVIDIA. [CUTLASS v3](https://github.com/NVIDIA/cutlass/tree/v3.0.0), 2023.
- [17] Bhaskaracharya et al. Modeling layout abstractions using integer set relations, 2025. arXiv:2511.10374.
- [18] Tavakkoli et al. LEGO: Layout expression for generating one-to-one mapping, 2025. arXiv:2505.08091.
- [19] Colfax Research. [Categorical foundations for CuTe layouts](https://research.colfax-intl.com/download/categories-of-layouts/), 2025.
- [20] Hagedorn et al. Graphene: An IR for optimized tensor computations on GPUs. *ASPLOS*, 2023.
- [21] Osama et al. Stream-K: Work-centric parallel decomposition for dense matrix-matrix multiplication on the GPU. *PPoPP*, 2023.
- [22] Kerr et al. [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda), 2017.
- [23] Dao et al. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS*, 2022.
- [24] Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning, 2023. arXiv:2307.08691.
- [25] Shah et al. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. *NeurIPS*, 2024.
- [26] NVIDIA. [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html), 2025.
- [27] Chetlur et al. cuDNN: Efficient primitives for deep learning, 2014. arXiv:1410.0759.
- [28] NVIDIA Corporation. PTX ISA 9.0 Documentation, 2025.

---

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
