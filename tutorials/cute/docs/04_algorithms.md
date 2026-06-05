# CuTe Tensor 算法

本节概述在 Tensor 上执行的常见数值算法的接口与实现。

这些算法的实现位于
[include/cute/algorithm/](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/)
目录。

## `copy`

CuTe 的 `copy` 算法将源 Tensor 的元素复制到目标 Tensor 的元素中。
`copy` 的多种重载位于
[`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp)。

### 接口与特化时机

Tensor 封装了数据类型、数据位置，
以及可能还包括其在编译期的 shape 和 stride。
因此，`copy` 可以且会根据其参数类型
分发到多种同步或异步硬件 copy 指令之一。

`copy` 算法有两种主要重载。
第一种仅接受源 Tensor 和目标 Tensor。

```c++
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

第二种除了这两个参数外，还接受一个 Copy_Atom。

```c++
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

两参数 `copy` 重载仅根据两个 Tensor 参数的类型选择默认实现。
Copy_Atom 重载允许调用方通过指定非默认的 copy 实现来覆盖该默认选择。

### 并行度与同步取决于参数类型

默认实现或 Copy_Atom 重载选择的实现
可能完全不使用或使用全部可用并行度，
并可能具有多种同步语义。
具体行为取决于 `copy` 的参数类型。
用户应根据其目标运行架构的知识来理解这些行为。
（开发者通常会为每种 GPU 架构编写各自优化的 kernel。）

`copy` 算法可能是每个 thread 顺序执行，
也可能在若干 thread 的集合（例如 block 或 cluster）上并行执行。

若 `copy` 为并行，
则参与 thread 的集合可能需要同步，
才能保证集合中任意 thread 假定 copy 操作已完成。
例如，若参与 thread 构成一个 CTA，
则用户必须调用 `__syncthreads()`
或 Cooperative Groups 的等价接口，
方可使用 `copy` 的结果。

`copy` 算法可能使用异步 copy 指令，
例如 `cp.async` 或其 C++ 接口 `memcpy_async`。
此时，用户需要执行与该底层实现相适应的
额外同步，
才能使用 `copy` 算法的结果。
[CuTe GEMM 教程示例](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/)
展示了一种这类同步方式。
更优化的 GEMM 实现会使用流水线技术
将异步 `copy` 操作与其他有用工作重叠。

### 通用 copy 实现示例

下面是一个针对任意两个 Tensor 的通用 `copy` 实现示例。

```c++
template <class TA, class ALayout,
          class TB, class BLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<TA, ALayout> const& src,  // Any logical shape
     Tensor<TB, BLayout>      & dst)  // Any logical shape
{
  for (int i = 0; i < size(dst); ++i) {
    dst(i) = src(i);
  }
}
```

该通用 `copy` 算法用一维逻辑坐标寻址两个 Tensor，
从而以逻辑列主序遍历二者。
一些合理的架构无关优化包括以下内容。

1. 若两个 Tensor 有已知且带优化访问指令（如 `cp.async`）的
   内存空间，则分发到该自定义指令。

2. 若两个 Tensor 具有静态 layout 且可证明
   元素向量化合法——例如，四个 `ld.global.b32`
   可合并为一个 `ld.global.b128`——则对源和目的 tensor 进行向量化。

3. 若可能，验证待使用的 copy 指令
   适用于源和目的 tensor。

CuTe 的优化 copy 实现可以完成上述所有优化。

## `copy_if`

CuTe 的 `copy_if` 算法与 `copy` 位于同一头文件，
[`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp)。
该算法与 `copy` 一样接受源和目标 Tensor 参数，
但还接受一个与输入输出同 shape 的「谓词 Tensor」。
源 Tensor 的元素仅当对应的谓词 Tensor 元素非零时才会被复制。

关于为何以及如何使用 `copy_if` 的更多细节，
请参阅教程的
["predication" 章节](./0y_predication.md)。

## `gemm`

### `gemm` 的计算内容

`gemm` 算法接受三个 Tensor：A、B 和 C。
其具体行为取决于参数 Tensor 的模式 (mode) 数量。
我们用字母表示这些 mode。

* V 表示「向量」，即独立元素的 mode。

* M 和 N 分别表示 BLAS GEMM 例程矩阵结果 C 的行数和列数。

* K 表示 GEMM 的归约 mode，
  即 GEMM 沿之求和的 mode。
  详见 [GEMM 教程](./0x_gemm_tutorial.md)。

我们用 `(...) x (...) => (...)` 记法列出
输入 Tensor A 和 B，以及输出 Tensor C 的 mode。
左侧两个 `(...)` 分别描述 A 和 B（按此顺序），
`=>` 右侧的 `(...)` 描述 C。

1. `(V) x (V) => (V)`。向量的逐元素乘积：C<sub>v</sub> += A<sub>v</sub> B<sub>v</sub>。分发到 FMA 或 MMA。

2. `(M) x (N) => (M,N)`。向量的外积：C<sub>mn</sub> += A<sub>m</sub> B<sub>n</sub>。以 V=1 分发到 (4)。

3. `(M,K) x (N,K) => (M,N)`。矩阵乘积：C<sub>mn</sub> += A<sub>mk</sub> B<sub>nk</sub>。对每个 K 分发到 (2)。

4. `(V,M) x (V,N) => (V,M,N)`。批量向量外积：C<sub>vmn</sub> += A<sub>vm</sub> B<sub>vn</sub>。针对寄存器复用优化，对每个 M、N 分发到 (1)。

5. `(V,M,K) x (V,N,K) => (V,M,N)`。批量矩阵乘积：C<sub>vmn</sub> += A<sub>vmk</sub> B<sub>vnk</sub>。对每个 K 分发到 (4)。

关于 CuTe 对 mode 顺序的约定，
请参阅 [GEMM 教程](./0x_gemm_tutorial.md)。
例如，若出现 K，它总是在最右侧（「最外」）。
若出现 V，它总是在最左侧（「最内」）。

### 分发到优化实现

与 `copy` 类似，CuTe 的 `gemm` 实现
会根据其 Tensor 参数的类型分发到
相应的优化实现。
同样与 `copy` 类似，`gemm` 接受可选的 MMA_Atom 参数，
允许调用方覆盖 CuTe 根据 Tensor 参数类型
选取的默认 FMA 指令。

关于 MMA_Atom 以及针对不同架构对 `gemm` 特化的更多信息，
请参阅
[教程的 MMA 章节](./0t_mma_atom.md)。

## `axpby`

`axpby` 算法位于头文件
[`include/cute/algorithm/axpby.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/axpby.hpp)。
它将 $y$ 赋值为 $\alpha x + \beta y$ 的结果，
其中 $\alpha$ 和 $\beta$ 为标量，$x$ 和 $y$ 为 Tensor。
其名称代表 "Alpha times X Plus Beta times Y"，
是原始 BLAS 的 "AXPY" 例程（"Alpha times X Plus Y"）的推广。

## `fill`

`fill` 算法位于头文件
[`include/cute/algorithm/fill.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/fill.hpp)。
它用给定的标量值覆盖其 Tensor 输出参数的元素。

## `clear`

`clear` 算法位于头文件
[`include/cute/algorithm/clear.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/clear.hpp)。
它用零覆盖其 Tensor 输出参数的元素。

## 其他算法

CuTe 还提供其他算法。
其头文件可在
[`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm)
目录中找到。

## Copyright

Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
