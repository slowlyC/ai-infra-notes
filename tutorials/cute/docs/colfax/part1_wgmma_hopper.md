# CUTLASS 教程：在 NVIDIA Hopper GPU 上使用 WGMMA 实现快速矩阵乘法

**原文**: [Colfax Research - CUTLASS Tutorial Part 1](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)

**翻译说明**: 本文翻译自 Colfax Research 的 CUTLASS Tutorial 系列 Part 1，技术术语保留英文或附中文注释。

---

任何 CUDA 教程系列都少不了 GEMM（通用矩阵乘法，GEneral Matrix Multiplication）这一部分。堪称现代 GPU 上最重要的例程之一，GEMM 构成了神经网络、大语言模型以及许多图形应用中大部分计算的核心; 尽管无处不在，高效实现 GEMM  notoriously 困难。

本系列教程共 3 篇，旨在帮助读者深入理解如何基于 NVIDIA Hopper GPU 使用 CUTLASS 库编写高效的 GEMM kernel。

- [Part 3] 将讨论 persistent kernel 与 [Stream-K](https://arxiv.org/abs/2301.03598)——一种在大量问题规模下达到 state-of-the-art 效率的 GEMM 负载均衡策略。
- [Part 2] 将讨论 [高效 GEMM kernel](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md) 的整体设计，包括 CUTLASS kernel 中使用的 warp-specialization、ping-pong scheduling 等高级技术。
- [Part 1，即本文] 讨论 warpgroup 矩阵乘累加（WGMMA，warpgroup matrix-multiply-accumulate）指令。这些是指向基于 Hopper 架构的 NVIDIA GPU Tensor Core 的原始指令。
- [归档内容](https://colfaxresearch.com/)

**整体概览**。本系列 3 篇文章大致按 GEMM kernel 的完整开发流程展开，但采用「由内向外」的顺序。首先，是调用 Tensor Core 完成计算的 tile 级 GEMM 原语。其次，是「每个 CTA」视角下的 GEMM kernel 设计——包含 prologue、mainloop 和 epilogue——其中主要挑战在于避免快速 Tensor Core 被内存加载瓶颈住。最后，是最外层 grid 层面的 CTA 调度，此时负载均衡成为主要考量。

希望读者在学完本系列后，能对 GEMM 算法有专家级理解，并将其中精妙思路用于设计和实现自己工作领域中的其他 kernel。

### 异步 Warpgroup MMA（WGMMA）

Hopper 引入了异步 warpgroup 级矩阵乘累加（WGMMA）指令。一个 warpgroup 由四个连续的 warp 组成，即 128 个连续线程，其中第一个 warp 的 warp-rank 为 4 的倍数。`wgmma.mma_async` 指令由 warpgroup 内全部 128 个线程共同执行。该操作通常具有以下两种形式，其中矩阵 `C` 充当累加器：

- `C = A * B`，此时累加器 `C` 的输入被禁用。
- `C = A * B + C`

WGMMA 的一项重要约束是：操作数 `B` 必须始终存放在共享内存（SMEM，shared memory）中。相比之下，操作数 `A` 可位于 SMEM 或寄存器（RMEM，register memory）中，累加器 `C` 则始终保存在 RMEM 中。

本文按以下顺序组织。首先，讨论在 CUTLASS 中调用 `wgmma.mma_async` 指令的要点，包括构建相应的 `TiledMMA` 对象，以及创建和划分与 WGMMA 兼容的 SMEM 张量。其次，讨论确保 WGMMA 正确性所需的同步机制。最后，更详细地讨论 WGMMA 中使用的布局，包括核心矩阵（core matrix）概念以及来源于 SMEM 的操作数的矩阵描述符（matrix descriptor）。

为简洁起见，下文中我们将把 `wgmma.mma_async` 简写为 `wgmma`。主要代码参考为 CUTLASS 的 [wgmma 教程](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/examples/cute/tutorial/wgmma_sm90.cu)，由 Pradeep Ramani 贡献，在 3.5.1 版本中加入。

### CUTLASS kernel 中的 WGMMA

本教程的主要目标是解释用于调用 Hopper Tensor Core 执行基于 tile 的 GEMM 的 `wgmma` 原语，以及如何将其作为 `cute::gemm` 调用的一部分进行使用。作为背景，考虑一个标准 GEMM kernel：接收维度为 `MxNxK` 的输入矩阵 `A` 和 `B`，计算 `C=A*B`。为实现并行，kernel 固定静态 tile 大小 `bM`、`bN`、`bK`，并启动 `⌈M/bM⌉x⌈N/bN⌉` 个 CTA 的 grid，每个 CTA 计算输出矩阵的一块 `bMxbN` tile `rC`。该 tile 在写回全局 `C` 矩阵前保存在 CTA 的 RMEM 中。

在单个 CTA 内，kernel 的 mainloop 在 `⌈K/bK⌉` 次迭代中沿内维度循环，依次将 `A` 和 `B` 的 `bMxbK` 与 `bNxbK` tile 从全局内存加载到共享内存，记为 `sA` 和 `sB`; 注意在 CUTLASS 中，我们将 `sB` 的 shape 固定为其数学上的转置形式。（实际中，与常见做法一致，我们将 `A` 和 `B` 的 tile 加载到环形 SMEM buffer 中，stage 数量由编译期整数如 2 或 3 指定。`sA` 和 `sB` 的 shape tuple 的最后一维即为该 stage 数。）`cute::gemm` 调用随后计算（按 stage 切片的）`sA` 与 `sB` 的乘积，并将结果累加到 `rC` 中。mainloop 完成后，epilogue 将 `rC` 写回全局内存。

下面将解释以下 `cute::gemm` 调用及其参数，该代码片段选自 [wgmma 教程](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/wgmma_sm90.cu#L73)，隐去了与当前讨论无关的部分（如流水线 TMA 加载）：

```cpp
template <class TiledMMA, ... >
__global__ device_gemm(TiledMMA tiled_mma, ...) {
  // PROLOGUE
  // ...
  // Define A/B partitioning and C accumulators
  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)

  // Allocate accumulators and clear them
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  // Allocate "fragments"
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)
  
  // PIPELINED MAIN LOOP
  while (k_tile_count > -K_PIPE_MAX) {
    // ...
    // MMAs to cover 1 K_TILE
    cute::warpgroup_arrive();
    // (V,M,K) x (V,N,K) => (V,M,N)
    cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
    cute::warpgroup_commit_batch();
    // Wait for all MMAs in a K_TILE to complete
    cute::warpgroup_wait<0>();
    // ...
  }

  // EPILOGUE
  // ...
}
```

在 CUTLASS 的 [MMA 范式](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0t_mma_atom.md) 中，`cute::gemm` 方法通过统一接口暴露架构特定的 MMA 指令。（若查看 [SM80 教程 GEMM kernel](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu#L275)，会发现其 `cute::gemm` 调用与上述在语法上完全相同。）然而，`cute::gemm` 调用所涉及的参数定义包含诸多 WGMMA 特有内容：

- 若操作数 `A` 来自 SMEM，则 fragment `tCrA` 与 `tCrB` 并非从 SMEM 复制数值的寄存器支撑张量，而是基于 SMEM 构建的矩阵描述符。
- `tCrA`、`tCrB`、`tCrC` 使用 `TiledMMA` 对象按线程进行数据划分，具有程序员需了解的 WGMMA 特有布局。
- SMEM 张量 `sA` 和 `sB` 的布局必须定义为与 `wgmma` 兼容。
- `TiledMMA` 对象 `tiled_mma` 的定义封装了 `cute::gemm` 分发到特定 `wgmma` PTX 指令所需的信息。

此外，围绕 `cute::gemm` 调用还有 warpgroup 同步原语。下面将逐一解释上述概念。

### WGMMA 的 TiledMMA 对象

下文假设数据类型为 FP16，且 `A` 和 `B` 均为 `MN`-major，因此在 BLAS 记法中我们计算的是 NT GEMM。我们使用 `cute::make_tiled_mma` 在 host 上构造 `TiledMMA` 对象：

```cpp
TiledMMA tiled_mma = cute::make_tiled_mma(
  SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});
```

尽管 `cute::make_tiled_mma` 还有若干可选参数，这里先聚焦第一个——MMA Atom。这是一个封装底层 PTX 调用的结构体，本例中对应：

```text
wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16
```

CUTLASS 记法使得封装 PTX 指令与 MMA atom 的对应关系一目了然。首先，SM90 是 Hopper 架构的别称。SM90 MMA atom 命名为 `SM90_MxNxK_XYZ_SS` 或 `SM90_MxNxK_XYZ_RS`，两个模板参数可为 `GMMA::Major::MN` 或 `GMMA::Major::K`，含义如下：

- 两个模板参数表示操作数 `A` 和 `B` 在内存上按 `MN` 模式还是 `K` 模式连续。例如在 BLAS 记法中，两者均为 `K`-major 对应 TN GEMM（参见 [此表](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md#aside-m-major-n-major-k-major)）。注意对 16 位操作数类型，内存布局可以是 `MN`-major 或 `K`-major; 但对非 16 位操作数，布局必须始终为 `K`-major。
- 后缀 `RS` 或 `SS` 表示操作数 `A` 来自寄存器（R）还是共享内存（S）。操作数 `B` 始终来自共享内存，故为 `S`。
- `MxNxK` 是 `wgmma` 指令计算的 tile 尺寸——即「wgmma atom」。并非所有 `MxNxK` 组合都合法，[允许的 shape 列表](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape)：`M` 恒为 64，`N` 为 8 的倍数且从 8 到 256，对 16 位操作数类型 `K` 为 16（更一般地，`K` 固定为 32 字节）。
- `Z` 为累加器的数据类型。
- `X` 和 `Y` 为操作数的数据类型。

MMA Atom 的语法大致如此。我们强调 WGMMA 是 warpgroup 级指令。在代码中，可通过 `size` 获取参与 TiledMMA 对象所定义 MMA 操作的线程数。例如以下 host 代码：

```cpp
dim3 dimBlock(cute::size(tiled_mma));
```

表示 kernel 中每个 CTA 以 1 个 warpgroup（128 个线程）启动。

若希望 2 个 warpgroup 分别执行 WGMMA、各自计算输出 tile 的一半（各自发出各自的 `wgmma` 指令），可将非平凡的布局（`AtomLayoutMNK`）作为第二个参数传给 `make_tiled_mma`。例如：

```cpp
 TiledMMA tiled_mma = make_tiled_mma(
  SM90_64x64x16_F16F16F16_SS{},
  Layout<Shape<_2,_1,_1>>{});
```

定义了一个 WGMMA 操作，其中 warpgroup 1 和 2 分别沿 `M` 模式计算输出 tile 的上半和下半（此时假设 `bM` 为 128 的倍数）。此时 `size(tiled_mma)` 等于 256。

一般而言，`make_tiled_mma` 的两个可选布局参数——`AtomLayoutMNK` 和 `PermutationMNK`——对所有 MMA Atom 的作用相同。关于 `PermutationMNK` 的用法，推荐阅读 Cris Cecka 的 [详解](https://github.com/NVIDIA/cutlass/discussions/1345)。

### WGMMA 的 SMEM 布局约束

下面说明在给定 MMA atom 选择下，SMEM 中操作数矩阵的 tile 尺寸与布局的约束。首先，与所有 MMA 指令一样，MMA atom 的 `MxNxK` 必须能整除操作数及累加器 tile 的尺寸。本例中，即 `bM` 为 64 的倍数、`bN` 为 64 的倍数、`bK` 为 16 的倍数。

其次，WGMMA 对 `sA` 和 `sB` 的 SMEM 布局（shape 与 stride）有额外约束，且该约束随选定的 swizzling 模式变化。具体而言，`sA`（按 stage 切片）的布局一般并非简单的 `(bM,bK):(1,bM)` 或 `(bM,bK):(bK,1)`，`sB` 同理。

要深入理解这些要求，需要引入下文的核心矩阵概念。但在实践中，我们总是可以借助 CUTLASS 提供的若干预定义布局 atom，配合 `cute::tile_to_shape` 方法，构造与 `wgmma` 兼容的布局。在本例中，我们在 host 上按如下方式准备 tile 尺寸及 `sA`、`sB`（其中 `T=cutlass::half_t` 即 CUTLASS 对 FP16 的命名）：

```cpp
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int< 64>{};  
auto bP = Int<  3>{};  // Pipeline

auto sA = cute::tile_to_shape(
	GMMA::Layout_MN_SW128_Atom<T>{},
	cute::make_shape(bM, bK, bP)
);
auto sB = cute::tile_to_shape(
	GMMA::Layout_MN_SW128_Atom<T>{},
	cute::make_shape(bN, bK, bP)
);
```

其中 `MN` 表示该布局 atom 适用于 `MN`-major 操作数，`SW128` 为 128 字节 swizzle 模式。打印 `sA` 或 `sB` 可得：

```text
Sw<3,4,3> o smem_ptr[16b](unset) o ((_64,_2),(_8,_8),_3):((_1,_512),(_64,_1024),_8192)
```

该布局从何而来？`cute::tile_to_shape` 接受一个布局（即 tile）并将其复制以覆盖更大的 shape（类似 `numpy.tile`）。撇开 swizzle 函数 `Sw<3,4,3>`，布局 atom 为 `(64,8):(1,64)`，按列主序在 shape `(128, 64, 3)` 上平铺，因此对 `MxK` 的 shape，较小的外 stride `512` 在 `M` 模式，较大的外 stride `1024` 在 `K` 模式。（最大的 stride `8192` 在 stage 数 `P` 模式，这是合理的，因为 `sA` 或 `sB` 不同 stage 的切片不应在内存中混在一起。）

注意 `64` 乘以 `sizeof(half_t)` 等于 128 字节，即 swizzle 模式名称的由来。这是有意设计的：按核心矩阵的工作方式，我们总是将布局 atom 在连续方向上的长度设为 swizzle 字节数——无 swizzle 时为 `16`，或有 `32`、`64`、`128` 之一。

与之对比，若采用：

```cpp
auto sA = cute::tile_to_shape(
  GMMA::Layout_K_SW128_Atom<T>{},
  cute::make_shape(bM,bK,bP)
);
auto sB = cute::tile_to_shape(
  GMMA::Layout_K_SW128_Atom<T>{},
  cute::make_shape(bN,bK,bP)
);
```

则打印 `sA` 会得到：

```text
Sw<3,4,3> o smem_ptr[16b](unset) o (_128,_64,_3):(_64,_1,_8192)
```

因为此时是在 `(128,64,3)` 上平铺 `(8,64):(64,1)`。（注意布局 `((_8,_16),(_64,_1),_3):((_64,_512),(_1,_0),_8192)` 会合并为 `(_128,_64,_3):(_64,_1,_8192)`。）

一般而言，可在 8 种布局 atom 中选择，对应 `MN` 或 `K`-major 以及四种 swizzle 模式之一：

- 128-byte swizzle：将 8 个连续 16-byte 段进行 swizzle。
- 64-byte swizzle：将 4 个连续 16-byte 段进行 swizzle。
- 32-byte swizzle：将 2 个连续 16-byte 段进行 swizzle。
- No swizzle：不 swizzle，隐含 16-byte 边界。

布局 atom 在 CUTLASS 中的定义位置 [在此](https://github.com/NVIDIA/cutlass/blob/36cbfcf483cc9d2ee65a55c199176ce96da1e33e/include/cute/atom/mma_traits_sm90_gmma.hpp#L66)：

```cpp
GMMA::Layout_MN_INTER_Atom<T>
GMMA::Layout_MN_SW32_Atom<T>
GMMA::Layout_MN_SW64_Atom<T>
GMMA::Layout_MN_SW128_Atom<T>

GMMA::Layout_K_INTER_Atom<T>
GMMA::Layout_K_SW32_Atom<T>
GMMA::Layout_K_SW64_Atom<T>
GMMA::Layout_K_SW128_Atom<T>
```

这些布局 atom 需传入 `tile_to_shape`，其中 `sA` 和 `sB` 的 SMEM shape 由 `make_shape(bM,bK,bP)` 或 `make_shape(bN,bK,bP)` 给出，shape 的模式顺序如上，且布局 atom 的 tile 尺寸需能整除更大的 SMEM shape。这本质上是由 swizzling 模式选择引起的对 SMEM shape 的约束，与 MMA atom shape 带来的另一约束相互独立。

### WGMMA Fragment 与描述符

已创建 `TiledMMA` 对象并在 host 上准备好 SMEM 布局。接下来在 device 上可使用 `tiled_mma` 构造传给 `cute::gemm` 的划分张量。首先，通过 `tiled_mma` 的 `get_thread_slice` 方法传入线程索引，得到 `thr_mma`（`ThrMMA` 对象）; 本例中线程索引从 0 到 127（含）。

接着，参照上方 kernel 代码片段，对任意线程索引打印 `tCsA` 和 `tCsB` 可得：

```text
tCsA: Sw<3,4,3>_smem_ptr[16b](0x7f8800000400) o
	((_64,(_8,_2)),_2,_4,_3):((_1,(_64,_1024)),_512,_2048,_8192)
tCsB: Sw<3,4,3>_smem_ptr[16b](0x7f880000c400) o
	((_64,(_8,_2)),_2,_4,_3):((_1,(_64,_1024)),_512,_2048,_8192)
```

按注释，`tCsA` 的 shape 可理解为 `(MMA,MMA_M,MMA_K,PIPE)`：

- `PIPE` 为 stage 数量。
- `MMA_M` 和 `MMA_K` 为 `sA` 在 `M`、`K` 模式上的平铺范围（故 `MMA_M=bM/64=2`，`MMA_K=bK/16=4`）。
- `MMA` 为 MMA Atom 的 `MxK` shape。

stride 和 swizzle 模式继承自 `sA`。此处需注意的是 `tCsA` 并非 SMEM 的线程级切片，而是以重组布局表示的完整 SMEM 张量。

接下来，对任意线程索引打印「fragment」`tCrA` 和 `tCrB`：

```text
tCrA: GMMA::DescriptorIterator o (_1,_2,_4,_3):(_0,_64,_256,_1024)
tCrB: GMMA::DescriptorIterator o (_1,_2,_4,_3):(_0,_64,_256,_1024)
```

CUTLASS 内部会构造一个「[矩阵描述符](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)」，即保存在寄存器中的 64 位值，用于以 `wgmma` 指令可用的方式描述 SMEM。对程序员而言，关键在于 SMEM 的值并未复制到 RMEM; 访问 `tCrA` 和 `tCrB` 实际上访问的是这些 64 位描述符。此外，这些张量作为「迭代器」，意味着同一时刻仅保存一次 `wgmma` 指令所用的单个 64 位描述符（例如，而非全部 24 个）。

与操作数不同，累加器张量采用更常规方式定义。对线程 0 打印 `tCgC` 和 `tCrC`：

```text
tCgC: gmem_ptr[16b](0x7f877a780000) o ((_2,_2,_8),_2,_2):((512,_8,4096),_64,32768)
tCrC: ptr[16b](0x7feee1fffbe0) o ((_2,_2,_8),_2,_2):((_1,_2,_4),_32,_64)
```

`tCgC` 是 epilogue 中要写入累加器值的输出 GMEM 张量切片，`tCrC` 是在 mainloop 计算过程中保存这些值的寄存器支撑张量。这些张量的 `(MMA,MMA_M,MMA_N)` shape 可理解为：在 MMA atom 的 `MxN=64x64` 输出 tile 中，128 个线程各持有 `32=2*2*8` 个值，`MMA_M=MMA_N=2` 与 `tCsA`、`tCsB` 相同。

每个线程的 32 个 atom 值按需分解为 (2,2,8) 以定义 `tCgC` 布局的对应 stride。具体划分模式可从 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d) 中的图示读出：

（图示说明每个线程 32 个值的重复 Z 形 pattern。例如线程 0 持有 (0,0)、(0,1)、(8,0)、(8,1) 等位置的值，并每 8 列向右重复。）

![WGMMA 64xN 累加器的 Z 形分区模式](https://research.colfax-intl.com/wp-content/uploads/2024/07/wgmma-64N16-D-1.png)

### 再看 gemm 调用

回到上面 kernel 代码片段的第 25 行：

```cpp
// (V,M,K) x (V,N,K) => (V,M,N)
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
```

`cute::gemm` 的各重载首先遍历外层模式 `MMA_M/N` 与 `MMA_K`。选定这些坐标后，即用 MMA atom tile shape 进行计算。换言之，先归约为 [dispatch shape](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/algorithm/gemm.hpp#L178) `(V)x(V)=>(V)` 的 `cute::gemm` 重载。

随后调用 MMA atom 的 [fma 操作](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp#L401)（精确地说在 [mma_unpack](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits.hpp#L112) 方法内）。其中包含内联 PTX 汇编：

```cpp
CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t& d00, uint32_t& d01, uint32_t& d02, uint32_t& d03,
      uint32_t& d04, uint32_t& d05, uint32_t& d06, uint32_t& d07,
      uint32_t& d08, uint32_t& d09, uint32_t& d10, uint32_t& d11,
      uint32_t& d12, uint32_t& d13, uint32_t& d14, uint32_t& d15,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15},"
      " %16,"
      " %17,"
      " p,   %19, %20, %21, %22;\n"
    "}\n"
      : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03),
        "+r"(d04), "+r"(d05), "+r"(d06), "+r"(d07),
        "+r"(d08), "+r"(d09), "+r"(d10), "+r"(d11),
        "+r"(d12), "+r"(d13), "+r"(d14), "+r"(d15)
      : "l"(desc_a),
        "l"(desc_b),
        "r"(int32_t(scale_D)),
        "n"(int32_t(scaleA)),
        "n"(int32_t(scaleB)),
        "n"(int32_t(tnspA)),
        "n"(int32_t(tnspB)));
#else
    CUTE_INVALID_CONTROL_PATH(
    	"Attempting to use SM90_64x64x16_F16F16F16_SS " 
    	"without CUTE_ARCH_MMA_SM90A_ENABLED");
#endif
  }
```

该语法的 PTX 文档在 [此处](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)。与上述 `tCrA`、`tCrB`、`tCrC` 描述一致，可见操作数使用 `uint64` 的 `desc_a` 和 `desc_b`，累加器使用 16 个 `uint32`。`scale_D` 为 0 或 1，控制累加器是否零初始化。

此外，`scaleA`、`scaleB`、`tnspA`、`tnspB` 在 `fma` 外部通过模板参数在编译期确定。`scaleA` 和 `scaleB` 为 1 或 -1（用于操作数取反），`tnspA` 和 `tnspB` 表示是否转置操作数，对 `GMMA::Major::K` 或 `GMMA::Major::MN` 分别为 0 或 1。

### WGMMA 的同步

最后说明围绕 `cute::gemm` 调用的同步原语：

```cpp
cute::warpgroup_arrive();
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
cute::warpgroup_commit_batch();
cute::warpgroup_wait<0>();
```

为何需要这些额外指令？这与 `wgmma` 的异步特性有关。在 Hopper 架构下，「异步」意味着 `wgmma` 可与其它操作并发执行，因此需要依赖步骤之间的同步机制。该机制在 PTX [内存一致性模型](https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html#program-order-async-operations) 中有详细说明。同步不当可能导致：(a) 微妙的竞态和难以定位的 bug; (b) 编译器将 `wgmma` 指令串行化，造成明显性能下降; 或 (c) 未定义行为。

上述 `cute` 方法封装了下列 PTX 指令：

- `cute::warpgroup_arrive()` — `wgmma.fence.sync.aligned`
- `cute::warpgroup_commit_batch()` — `wgmma.commit_group.sync.aligned`
- `cute::warpgroup_wait<0>()` — `wgmma.wait_group.sync.aligned N`

（注意：全文以 `wgmma` 作为 `wgmma.mma_async` 的简写，本小节特意区分。）下面将用法与 PTX 文档中 WGMMA 相关 GEMM 的 [描述](https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions) 对应：

执行以下 `fence` 操作：

- `fence.proxy.async`：使 generic proxy 操作对 async proxy 可见。
- `wgmma.fence`：表示 warpgroup 内的寄存器/共享内存已完成写入。

逐条说明。首先，`wgmma.fence` 确保 `wgmma.mma_async` 仅在所有对该 RMEM 地址的先前访问完成后才访问这些地址。没有 `wgmma.fence` 则行为未定义。例外是 Hopper 允许多条 `wgmma.mma_async` 同时 in-flight; 只要这些指令有相同的累加器 shape，即可共用同一累加器张量（即写同一寄存器地址），此时无需 fence。例如，`cute::gemm` 中遍历 `MMA_K` 的循环内不需插入 `wgmma.fence`。

与 [TMA 操作](https://research.colfax-intl.com/tutorial-hopper-tma/) 类似，`wgmma.mma_async` 在 [async proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy) 中执行。因此，若 generic proxy 中的操作会影响 `wgmma.mma_async` 读取的 SMEM，则需要发出 `fence.proxy.async`。例如，若通过普通 `ld.global`/`st.shared` 将 `A` 和 `B` 拷贝到 SMEM，就需要该 fence。本例使用 TMA 加载，故不需 `fence.proxy.async`，WGMMA 教程代码和 CUTLASS Hopper GEMM kernel 的 mainloop 中也没有出现。（可验证：`fence.proxy.async` 被 `cutlass::arch::fence_view_async_shared()` 封装。）

`wgmma.commit_group` 为每个 warpgroup 创建一个新的 wgmma-group，并将执行 warpgroup 已发起但尚未提交到任何 wgmma-group 的所有先前 `wgmma.mma_async` 指令打包进该新 wgmma-group。本例中，`cute::warpgroup_commit_batch()` 将 `MMA_M*MMA_N*MMA_K` 条 `wgmma.mma_async` 指令打包为一个 wgmma-group。

最后，带参数 `N` 的 `wgmma.wait_group` 会使执行线程等待，直到最近 wgmma-group 中最多仅剩 `N` 个尚未完成，且执行线程提交的所有先前 wgmma-group 均已完成。本例中 `N=0`，故 warpgroup 在继续执行后续指令前会等待整个 wgmma-group 完成。

当 warpgroup 有机会做独立计算时，`N` 的灵活性很有用，例如 [FlashAttention-3](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/) 中的 GEMM-softmax 重叠策略。

### WGMMA 核心矩阵

本节进一步讨论加载到 SMEM 的矩阵 `A` 和 `B` tile 的布局要求，假设 `wgmma` 的两个操作数均来自 SMEM。为简化讨论，先设 `A` 为行主序、`B` 为列主序（即均为 `K`-major）。并回顾：`wgmma` 指令的 tile shape `MxNxK` 满足 `M` 为 64，`K` 乘以数据类型大小为 32 字节，`N` 为 8 的倍数且取 8 到 256。为避免与 `A`/`B` 或 `sA`/`sB` 混淆，将 WGMMA atom tile 记为 `wA` 和 `wB`。

矩阵 `wA` 和 `wB` 被划分为若干更小的矩阵，称为核心矩阵（core matrix）。每个核心矩阵有 strided 方向和 contiguous 方向，其中 strided 方向长度为 8，contiguous 方向为 16 字节。矩阵 `wA` 由 `8x2` 个核心矩阵组成，矩阵 `wB` 由 `2x(N/8)` 个核心矩阵组成。核心矩阵对 `wA` 和 `wB` 的划分如下（图来自 PTX 文档）：

- **wA 在 SMEM 中的布局**（Layout of wA in SMEM）

![wA 在 SMEM 中的布局](https://research.colfax-intl.com/wp-content/uploads/2025/02/wgmma2.png)

- **wB 在 SMEM 中的布局**（Layout of wB in SMEM）

![wB 在 SMEM 中的布局](https://research.colfax-intl.com/wp-content/uploads/2025/02/wgmma3.png)

（图可从 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor) 或 [此推文](https://x.com/hyhieu226/status/1821572717877022876/photo/1) 查看。）

如上所述，SS 模式下的 `wgmma` 需要 `wA`（desc-a）和 `wB`（desc-b）的 [矩阵描述符](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)。该描述符编码五个参数：

- **Matrix base offset**：用于解决 SMEM 地址未按 swizzle 模式重复 pattern 的字节边界对齐时的对齐问题。
- **Swizzling mode**：none、32、64 或 128 字节。
- **SBO（stride dimension byte offset）**：`M` 或 `N` 维度上相邻核心矩阵之间的字节距离。
- **LBO（leading dimension byte offset）**：`K` 维度上相邻核心矩阵之间的字节距离。
- **Start address**：操作数在 SMEM 中的起始基地址。
- 当 wgmma-group 完成后，所有 `wgmma.mma_async` 操作均已执行并完成。

LBO 和 SBO 在上图中标出。

CUTLASS 中的 [make_gmma_desc](https://github.com/NVIDIA/cutlass/blob/06b21349bcf6ddf6a1686a47a137ad1446579db9/include/cute/atom/mma_traits_sm90_gmma.hpp#L194C1-L194C54) 方法根据输入的 SMEM 张量布局构造描述符（作为 [GmmaDescriptor](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/arch/mma_sm90_desc.hpp#L86) 实例）。只要输入张量的布局由上文「WGMMA 的 SMEM 布局约束」中所述的八种规范 GMMA 布局 atom 之一及 `tile_to_shape` 创建，`make_gmma_desc` 就能正确计算 LBO 和 SBO、确定 swizzling 模式并构造描述符。例如，`GmmaDescriptor` 在 `K`-major 情形下（其中 `T*sizeof(dtype)=16`）描述以下合法 WGMMA 布局：

```text
No swizzle       : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2)):((1T,SBO),(1,LBO))
32-byte swizzle  : Swizzle<1,4,3> o smem_ptr o ((8,m),(T,2)):((2T,SBO),(1, T ))
64-byte swizzle  : Swizzle<2,4,3> o smem_ptr o ((8,m),(T,2)):((4T,SBO),(1, T ))
128-byte swizzle : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1, T ))
```

对 GMMA 布局 atom => `tile_to_shape` 模式产生的 [compact](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/layout.hpp#L415) 布局（注意：64 和 128-byte swizzle 时，GMMA 布局 `K` atom 的 `K`-mode 大于 WGMMA atom shape！），LBO 和 SBO 的对应值为：

```text
No swizzle       : LBO = 16x8 = 128 bytes. SBO = 32x8 = 256 bytes.
32-byte swizzle  : SBO = 32x8 = 256 bytes.
64-byte swizzle  : SBO = 64x8 = 512 bytes.
128-byte swizzle : SBO = 128x8 = 1024 bytes.
```

特别地，对 64 和 128-byte swizzle，stride 使得上述合法 WGMMA 布局并非 compact。而是在 `K` 方向上并排放置 2 或 4 个 WGMMA atom 操作数 tile，导致核心矩阵 `M`-mode 的 stride 为 `4T` 和 `8T`。换言之，swizzling 时会在内存中交错 `K`-mode 上逻辑相邻的 2、4 或 8 个核心矩阵，而这些核心矩阵在 64 和 128-byte swizzle 下属于不同的 WGMMA atom。

为完整起见，`MN`-major 情形下的合法 WGMMA 布局如下：

```text
No swizzle       : Swizzle<0,4,3> o smem_ptr o ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
32-byte swizzle  : Swizzle<1,4,3> o smem_ptr o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
64-byte swizzle  : Swizzle<2,4,3> o smem_ptr o ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
128-byte swizzle : Swizzle<3,4,3> o smem_ptr o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
```

### 小结

在 GEMM 系列 [Part 1] 中，我们涵盖了将 WGMMA（warpgroup 矩阵乘累加）作为 Hopper 上 GEMM 原语使用的核心概念。

WGMMA 需要 warpgroup（128 个线程）共同执行矩阵乘法，且只能作用于矩阵的特定 fragment。我们详细介绍了涉及的 shape 与布局，重点是使用规范 GMMA Layout => `tile_to_shape` 模式构造 WGMMA 接受的操作数布局。

WGMMA 的规范使用还需要若干同步机制。本文说明了 `wgmma.fence`、`fence.proxy.async`、`wgmma.commit_group` 和 `wgmma.wait_group` 与 `wgmma.mma_async` 的关系。

最后，我们较详细地介绍了 WGMMA 核心矩阵的内在机制，以及 CUTLASS 如何为来自 SMEM 的操作数构造矩阵描述符。

整体而言，本文应能让程序员在 Hopper 上编写使用 WGMMA 的 CUTLASS kernel。[Part 2] 将在此基础上加入 TMA，讨论如何在 Hopper GEMM kernel 中配合使用 TMA 与 WGMMA，以实现 copy 与 compute 的重叠。

---

*原文发布于 2024 年 8 月 6 日，归类于 Article、Blog、Tutorials。*
