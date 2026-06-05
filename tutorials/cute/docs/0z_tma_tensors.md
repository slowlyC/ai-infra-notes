# CuTe TMA Tensor

在使用过程中，你可能会遇到一些看起来奇怪的 CuTe Tensor，打印出来类似：

```
ArithTuple(0,_0,_0,_0) o ((_128,_64),2,3,1):((_1@0,_1@1),_64@1,_1@2,_1@3)
```

什么是 `ArithTuple`？那些是 tensor stride 吗？表示什么？用于什么？

本文旨在回答这些问题，并介绍 CuTe 的一些进阶特性。

## TMA 指令简介

Tensor Memory Accelerator（TMA）是一组在全局内存与共享内存之间拷贝可能为多维数组的指令。TMA 在 Hopper 架构中引入。一条 TMA 指令可以一次拷贝整个 tile 的数据，因此硬件不再需要为 tile 的每个元素分别计算地址、发出拷贝指令。

为实现这一点，TMA 指令接收一个*TMA descriptor*，它是全局内存中多维 tensor 的打包表示，可支持 1、2、3、4 或 5 维。TMA descriptor 包含：

* tensor 的基指针; 
* tensor 元素的数据类型（如 `int`、`float`、`double` 或 `half`）; 
* 各维的大小; 
* 各维的 stride; 
* 以及表示 smem box 大小、smem swizzling 模式和越界访问行为等的其他标志。

该 descriptor 必须在 kernel 执行前于 host 端创建，在所有将发起 TMA 指令的线程块之间共享。进入 kernel 后，TMA 用以下参数执行：

* 指向 TMA descriptor 的指针; 
* 指向共享内存的指针; 
* TMA descriptor 所描述的 GMEM tensor 中的坐标。

例如，带 3-D 坐标的 TMA-store 接口如下。

```cpp
struct SM90_TMA_STORE_3D {
  CUTE_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2) {
    // ... invoke CUDA PTX instruction ...
  }
};
```

可见 TMA 指令并不直接消费指向全局内存的指针。全局内存指针包含在 descriptor 中，被视为常量，并非 TMA 指令的单独参数。TMA 消费的是 TMA descriptor 所定义、TMA 视图下的全局内存坐标。

因此，存储 GMEM 指针并计算偏移、新 GMEM 指针的普通 CuTe Tensor 对 TMA 没有用处。

该怎么办？

## 构建 TMA Tensor

### 隐式 CuTe Tensor

所有 CuTe Tensor 都是 Layout 和 Iterator 的复合 (composition)。普通全局内存 tensor 的迭代器就是其全局内存指针。然而，CuTe Tensor 的迭代器不必是指针，可以是任意随机访问迭代器。

其中一个例子是*计数迭代器* (counting iterator)。它表示以某个值开始的、可能无限的整数序列。我们称该序列的成员为*隐式整数*，因为序列并非显式存储在内存中，迭代器只保存当前值。

可以用计数迭代器创建隐式整数 tensor：

```cpp
Tensor A = make_tensor(counting_iterator<int>(42), make_shape(4,5));
print_tensor(A);
```

输出为：

```
counting_iter(42) o (4,5):(_1,4):
   42   46   50   54   58
   43   47   51   55   59
   44   48   52   56   60
   45   49   53   57   61
```

该 tensor 将逻辑坐标映射到即时计算的整数。因为它仍然是 CuTe Tensor，所以可以和普通 tensor 一样被分块、分区、切片，通过把整数偏移累加到迭代器中实现。

但 TMA 不消费指针或整数，它消费坐标。能否构造 TMA 指令可消费的隐式 TMA 坐标 tensor？若能，我们就可以对该坐标 tensor 做分块、分区、切片，从而始终得到正确的 TMA 坐标传给指令。

### ArithTupleIterator 与 ArithmeticTuple

首先，构造 TMA 坐标的 `counting_iterator` 等价物。它需要支持：

* 解引用得到 TMA 坐标; 
* 按另一个 TMA 坐标偏移。

我们称之为 `ArithmeticTupleIterator`。它保存一个由 `ArithmeticTuple` 表示的坐标（整数 tuple）。`ArithmeticTuple` 是 `cute::tuple` 的（公开子类），重载了 `operator+`，因此可以被另一个 tuple 偏移。两个 tuple 的和是逐元素相加得到的 tuple。

类似 `counting_iterator<int>(42)`，可以创建在 tuple 上的隐式“迭代器”（但不支持 increment 等常见迭代器操作），可被解引用并按其他 tuple 偏移：

```cpp
ArithmeticTupleIterator citer_1 = make_inttuple_iter(42, Int<2>{}, Int<7>{});
ArithmeticTupleIterator citer_2 = citer_1 + make_tuple(Int<0>{}, 5, Int<2>{});
print(*citer_2);
```

输出为：

```
(42,7,_9)
```

TMA Tensor 可以用这样的迭代器保存当前 TMA 坐标“偏移”。这里的“偏移”加引号，因为它显然不是普通的 1-D 数组偏移或指针。

总结：为*整个全局内存 tensor* 创建一个 TMA descriptor。TMA descriptor 定义对该 tensor 的视图，指令接受进入该视图的 TMA 坐标。为生成和跟踪这些 TMA 坐标，我们定义一个可被分块、切片、分区的隐式 CuTe TMA 坐标 tensor，方式与普通 CuTe Tensor 完全一致。

现在可以用该迭代器跟踪和偏移 TMA 坐标，但如何让 CuTe Layout 生成非整数偏移？

### Stride 不仅是整数

普通 tensor 的 layout 将逻辑坐标 `(i,j)` 映射为 1-D 线性索引 `k`。该映射是坐标与 stride 的内积。

TMA Tensor 持有 TMA 坐标的迭代器。因此 TMA Tensor 的 Layout 必须将逻辑坐标映射为 TMA 坐标，而不是 1-D 线性索引。

为此，可以抽象 stride 的含义。Stride 不必是整数，而可以是任意支持与整数（逻辑坐标）做内积的代数对象。显然可以选择此前用过的 `ArithmeticTuple`，它们可以彼此相加，这里再增加 `operator*`，使其也可以被整数缩放。

#### 旁注：整数模 stride

支持元素间加法及与整数乘积的一类对象称为整数模 (integer-module)。

形式上说，整数模是配备 `Z*M -> M` 的阿贝尔群 `(M,+)`，其中 `Z` 为整数。即整数模 `M` 是支持与整数做内积的群。整数本身是整数模。秩为 R 的整数 tuple 是整数模。

原则上，layout stride 可以是任意整数模。

#### 基元素

CuTe 的基元素 (basis element) 定义在头文件 `cute/numeric/arithmetic_tuple.hpp` 中。为便于创建可作为 stride 使用的 `ArithmeticTuple`，CuTe 用 `E` 类型别名定义归一化基元素。“归一化”指基元素的缩放因子为编译期整数 1。

| C++ object | Description             | String representation |
| ---        | ---                     | ---                   |
| `E<>{}`    | `1`                     | `1`                   |
| `E<0>{}`   | `(1,0,...)`             | `1@0`                 |
| `E<1>{}`   | `(0,1,0,...)`           | `1@1`                 |
| `E<0,0>{}` | `((1,0,...),0,...)`     | `1@0@0`               |
| `E<0,1>{}` | `((0,1,0,...),0,...)`   | `1@1@0`               |
| `E<1,0>{}` | `(0,(1,0,...),0,...)`   | `1@0@1`               |
| `E<1,1>{}` | `(0,(0,1,0,...),0,...)` | `1@1@1`               |

上表中“description”列将各基元素解释为无穷 tuple 的整数，其中元素类型未指定的 tuple 位置均为 0。tuple 位置从左到右，从 0 开始计数。例如 `E<1>{}` 在位置 1 为 1：`(0,1,0,...)`。`E<3>{}` 在位置 3 为 1：`(0,0,0,1,0,...)`。

基元素可以*嵌套*。例如上表中，`E<0,1>{}` 表示位置 0 有一个 `E<1>{}`：`((0,1,0,...),0,...)`。类似地，`1@1@0` 表示将 `1` 提升到位置 1 得到 `1@1`：`(0,1,0,...)`，再提升到位置 0。

基元素可以*缩放*，即可乘以整数*缩放因子*。例如在 `5*E<1>{}` 中，缩放因子为 `5`。`5*E<1>{}` 打印为 `5@1`，表示 `(0,5,0,...)`。缩放因子可与任意嵌套交换。例如 `5*E<0,1>{}` 打印为 `5@1@0`，表示 `((0,5,0,...),0,...)`。

只要层级结构兼容，基元素也可以相加。例如 `3*E<0>{} + 4*E<1>{}` 得到 `(3,4,0,...)`。直观上，“兼容”指两个基元素的嵌套结构足以使二者相加。

#### stride 的线性组合

Layout 通过自然坐标与其 stride 的内积工作。对于由整数元素组成的 stride，例如 `(1,100)`，输入坐标 `(i,j)` 与 stride 的内积为 `i + 100j`。用“普通” tensor 的指针加上该索引进行偏移，即得到 `(i,j)` 处元素的指针。

对于基元素组成的 stride，我们仍计算自然坐标与 stride 的内积。例如，若 stride 为 `(1@0,1@1)`，则输入坐标 `(i,j)` 与 stride 的内积为 `i@0 + j@1 = (i,j)`，即 TMA 坐标 `(i,j)`。若要交换坐标顺序，可使用 stride `(1@1,1@0)`，layout 求值得 `i@1 + j@0 = (j,i)`。

基元素的线性组合可解释为可能多维、具有层级结构的坐标。例如 `2*2@1@0 + 3*1@1 + 4*5@1 + 7*1@0@0` 表示 `((0,4,...),0,...) + (0,3,0,...) + (0,20,0,...) + ((7,...),...) = ((7,4,...),23,...)`，可解释为坐标 `((7,4),23)`。

因此，这些 stride 的线性组合可用于生成 TMA 坐标。这些坐标进而可用于偏移 TMA 坐标迭代器。

### 应用于 TMA Tensor

现在可以构造引言中看到的那种 CuTe Tensor。

```cpp
Tensor a = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<0>{}, E<1>{}));
print_tensor(a);

Tensor b = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<1>{}, E<0>{}));
print_tensor(b);
```

输出为：

```
ArithTuple(0,0) o (4,5):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)

ArithTuple(0,0) o (4,5):(_1@1,_1@0):
  (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
  (0,1)  (1,1)  (2,1)  (3,1)  (4,1)
  (0,2)  (1,2)  (2,2)  (3,2)  (4,2)
  (0,3)  (1,3)  (2,3)  (3,3)  (4,3)
```

### Copyright

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
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
