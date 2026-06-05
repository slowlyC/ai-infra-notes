# CuTe 稠密矩阵乘教程

本节将介绍
[这些示例](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/)，
它们展示了若干自包含、单文件形式的稠密矩阵乘实现，仅使用 CuTe。

## `sgemm_1.cu`

教程中最简单的示例，涵盖以下基础内容：将全局内存按 CTA（在 CUDA 中也称线程块）划分成块，将数据块在 CTA 的各线程间划分，以及使用 `cute::copy` 和 `cute::gemm` 编写主循环 (mainloop)。

### 高层接口

我们从文件顶部的内核入口 `gemm_device` 开始。

```c++
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
```

模板参数较多，先快速浏览一下，再深入其用法。

* `ProblemShape`。本次矩阵乘的 MxNxK 问题形状 (Shape)。

* `CtaTiler`。一个 CuTe [tiler 概念](./02_layout_algebra.md#composition-tilers)，用于确定如何从问题形状中提取一块数据。

* `TA const* A`、`TB const* B`、`TC* C`。分别为 A、B、C 数据的类型和指针。

* `AStride`、`BStride`、`CStride`。与 A、B、C 的 `ProblemShape` 对应的 Layout 步幅 (Stride)。

* `ASmemLayout`、`BSmemLayout`、`CSmemLayout`。各 CTA 内用于暂存 A、B、C 的共享内存 Layout（如需要）。

* `AThreadLayout`、`BThreadLayout`、`CThreadLayout`。用于划分各阶段时的线程 Layout。

* `Alpha alpha`、`Beta beta`。用于计算 GEMM 的标量常量类型与值：`C = alpha * A * B + beta * C`。

### 完整 Tensor：Shape、Stride 与数据

多数 GEMM 接口按 M、N、K 顺序列出矩阵维度。CuTe 也采用该约定，但将其打包进单个 IntTuple。本示例中，这些为动态值，定义在调用设备内核的 `gemm_nt` 和 `gemm_tn` 主函数顶部。
```cpp
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);    // (M, N, K)
```

内核中会先检查问题形状是否满足前置条件，再构造各完整矩阵。
```cpp
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                      // (M, N, K)

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));            // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));            // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));            // dC strides for shape MN

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);  // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);  // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);  // (M,N)
```
通过选取 `Shape` 的适当模式 (mode) 来构造各张量 (Tensor)。前置条件保证了 `Shape` 中每个整数都有对应的 `Stride` 中的整数。

注意 B 后注释为 `(N,K)` 而非 `(K,N)`，表示 B 被视为 NxK 矩阵，而不是 BLAS 及多数矩阵乘中常见的 KxN。CuTe 采用约定：`A` 为 `(M,K)`，`B` 为 `(N,K)`，`C` 为 `(M,N)`，我们在各处注释中都尽量体现。

对于 `(M,K)`、`(N,K)` 和 `(M,N)` 张量，`gemm_nt` 和 `gemm_tn` 构造各自使用的步幅。在 `gemm_nt` 中定义为
```cpp
  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);    // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);    // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);    // (dM, dN)
```
在 `gemm_tn` 中定义为
```cpp
  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});    // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});    // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);    // (dM, dN)
```

#### 旁注：M-主序、N-主序、K-主序

我们发现 BLAS 中把“非转置”(N) 与“转置”(T) 标志和 `MxK * KxN` 的模式约定混用，容易模糊核心问题：“矩阵用的是什么布局”以及“在哪个模式下矩阵有 stride-1”。其实这两个问题都可以通过查看 CuTe 的 `Layout` 回答。

相比行主序 (row-major) 或列主序 (column-major)（以及 Transposed 和 Not-Transposed），更方便的说法是：若矩阵在 M 模式下 stride-1 则为 M-主序 (M-major)，在 N 模式下 stride-1 则为 N-主序 (N-major)，在 K 模式下 stride-1 则为 K-主序 (K-major)。此外，由于矩阵乘总是在 K 模式下做归约，从软件角度让 K 模式始终位于同一位置，并采用 `MxK * NxK` 的模式约定会非常方便。实现总是在两个输入矩阵的第二个模式（K 模式）上归约，这样可以对两个输入矩阵一视同仁。

如何与 BLAS 用户的习惯对应？

| BLAS | A Majorness | A Layout        | B Majorness | B Layout        |
| ---  | ---         | ---             | ---         | ---             |
| NT   | M-major     | `(M,K):(1,ldA)` | N-major     | `(N,K):(1,ldB)` |
| TN   | K-major     | `(M,K):(ldA,1)` | K-major     | `(N,K):(ldB,1)` |
| NN   | M-major     | `(M,K):(1,ldA)` | K-major     | `(N,K):(ldB,1)` |
| TT   | K-major     | `(M,K):(ldA,1)` | N-major     | `(N,K):(1,ldB)` |

尽管如此，在合适时我们仍会使用 BLAS 的 NT、TN 等符号来概括描述内核。

### CTA 分区

有了完整矩阵的表示后，就可以对它们分块并分配工作。

最上层是将工作分配到各 CTA。原则上，每个 CTA 的块可以从输入张量中以多种方式得到。可使用多种 [CuTe Tiler](./02_layout_algebra.md#composition-tilers) 对数据分块，本例中只需使用目标 CTA 块形状即可。
```cpp
  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);  // (BLK_M, BLK_N, BLK_K)
```

定义好 tiler 后，可用其将张量在各 CTA 间划分。

```cpp
  // Get the appropriate blocks for this threadblock
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

首先创建 CTA 坐标。
* 该块的 `m` 坐标由 `blockIdx.x` 给出。
* 该块的 `n` 坐标由 `blockIdx.y` 给出。
* 该块的 `k` 坐标未指定——我们要覆盖 `K` 中所有块，故用 `_`（Underscore）占位。

`local_tile` 用于剔除 tiler 与 coord 中对应 `X` 的模式。即 `Step<_1, X,_1>` 等价于
```cpp
  // Use select<0,2> to use only the M- and K-modes of the tiler and coord
  Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord));
```
该 `local_tile` 即是对以下步骤的简写：
1. 通过 [`zipped_divide`](./02_layout_algebra.md#zipped-tiled-flat-divides) 应用 tiler
```cpp
// ((BLK_M,BLK_K),(m,k))
Tensor gA_mk = zipped_divide(mA, select<0,2>(cta_tiler));
```
2. 对第二个模式（Rest 模式）应用 coord，提取该 CTA 对应的块
```cpp
// (BLK_M,BLK_K,k)
Tensor gA = gA_mk(make_coord(_,_), select<0,2>(cta_coord));
```
由于 tiler 与 coord 的投影是对称的，且“应用 tiler 再在 rest 模式上切片以得到分区”这两步很常用，因此被封装成投影式的 `local_tile` 接口。

对张量 `A`，得到秩为 3 的张量，形状为 `(BLK_M,BLK_K,k)`。前两个模式是 CTA 块的模式，最后一个模式索引该 CTA 将参与归约的所有块。在下面的主循环部分，通过 `k_tile` 循环遍历该模式。

### 共享内存张量

用于存放 A、B 数据块的共享内存 Layout，也作为参数 `ASmemLayout sA_layout` 和 `BSmemLayout sB_layout` 传入。

在 `gemm_nt` 中定义为
```c++
  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));   // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));   // (n,k) -> smem_idx; n-major
```
得到简单的 M-主序和 N-主序 Layout。在 `gemm_tn` 中为
```cpp
  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM,bK), LayoutRight{});   // (m,k) -> smem_idx; k-major
  auto sB = make_layout(make_shape(bN,bK), LayoutRight{});   // (n,k) -> smem_idx; k-major
```
得到简单的 K-主序 Layout。

可见，这些 smem Layout 几乎可以是任意的。内核中仅检查两点：共享内存 Layout 为静态，且与 `CtaTiler` 的顶层形状相同。

```cpp
  // Preconditions
  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K
```

使用静态 Layout 有几项好处。
* 可如下所示静态分配共享内存。
* 静态 Layout 往往更高效，便于 CuTe 派发到优化实现。
* 便于证明算法正确性并做如上检查——smem Layout 大小与 CTA 块大小一致。

如前所述，共享内存 Layout 只要满足上述条件即可。对这类内核的优化，常通过找到合适的共享内存 Layout 来获得更好的读写访问模式，包括向量化读写以及避免共享内存 bank 冲突。

有了静态 smem Layout，`gemm_device` 内核即可分配所需共享内存并创建 smem 张量。

```cpp
  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);  // (BLK_N,BLK_K)
```

注意共享内存分配仅依赖数据类型和 Layout。`cosize` 是什么？因为 `Layout` 是一种映射，可以讨论其定义域和值域。Layout 的 `size` 是定义域的大小，`cosize` 是值域的大小。若要为 Layout 产生的所有偏移分配一个数组，可用 Layout 的 `cosize` 作为数组长度（以元素为单位）。

### Copy 分区

内核已通过 `CtaTiler` 得到全局内存块，并通过适当分配得到共享内存块。接下来要建立一种高效方式，把一块全局内存复制到共享内存块。最朴素的方式是单线程逐元素复制。
```cpp
if (thread0()) {
  Tensor gA0 = gA(_,_,0);  // (BLK_M,BLK_K), the 0th tile
  for (int i = 0; i < size(sA); ++i) {
    sA(i) = gA0(i);
  }
}
```
这样可行，但 CTA 内有很多线程，应利用起来。

若把两块数据在 CTA 线程间做分区，每个线程可复制自己那一小份。分区方式很多。

`gemm_nt` 定义了两个*线程* Layout：
```c++
  // Define thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{},Int<8>{}));   // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<32>{},Int<8>{}));   // (n,k) -> thr_idx
```
`gemm_tn` 则定义为：
```c++
  // Define thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{},Int<8>{}), LayoutRight{});  // (m,k) -> thr_idx; k-major
  auto tB = make_layout(make_shape(Int<32>{},Int<8>{}), LayoutRight{});  // (n,k) -> thr_idx; k-major
```
两者都使用 32x8 个线程，用于把 128x8 的 gmem 和 smem 块划分成每线程 4x1 子张量。区别在于 `gemm_nt` 用 M-主序和 N-主序的线程以匹配全局内存数据顺序，`gemm_tn` 用 K-主序线程以匹配全局内存数据顺序。

内核中同样会检查线程 Layout 的约束。
```cpp
  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tB));                          // NumThreads

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});  // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});  // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});  // BLK_N / THR_N
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});  // BLK_K / THR_K
```

这些线程 Layout 用于对全局内存张量和共享内存张量做分区：
```cpp
  Tensor tAgA = local_partition(gA, tA, threadIdx.x);    // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);    // (THR_M,THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);    // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);    // (THR_N,THR_K)

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA));  // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));  // THR_K
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB));  // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));  // THR_K
```
其中 `local_partition` 与 `local_tile` 类似，但坐标是对 `zipped_divide` 的 tile 模式（第一模式）而非 rest 模式（第二模式）做切片。即每个线程在每个 thread tile 上获得一个数据元素，该 thread tile 会重复以覆盖整个数据块。

`tAsA` 这类命名在 CuTe 和 CUTLASS 中很常见，读作“对张量 `sA` 应用分区模式 `tA`”。下一节会看到对 `sA` 应用不同分区器得到 `tCsA`。通过对 `sA` 和 `gA` 应用同一分区模式 `tA`，保持两者的*逻辑一致性*（由上面的断言检查），两张量间的逻辑元素一一对应，即便数据布局不同。例如在 `cute::copy` 中，该命名便于从词法上确认两张量使用同一分区模式。

数据在各线程间分区后，*每个线程*都可参与复制，写法如下：
```cpp
copy(tAgA(_,_,0), tAsA);
```
因为每个线程拥有待复制块中不同的子张量。

### 数学分区

内核已将共享内存中的块从全局内存复制进来。接下来要建立一种高效方式，在该块上计算并累加矩阵乘积。最朴素的方式是单线程直接计算。
```cpp
if (thread0()) {
  for (int m = 0; m < size<0>(gC); ++m) {
    for (int n = 0; n < size<1>(gC); ++n) {
      for (int k = 0; k < size<1>(sA); ++k) {
        gC(m,n) += sA(m,k) * sB(n,k);
      }
    }
  }
}
```
这样可行，但 CTA 内有很多线程，应利用起来。

若把输出块 `gC` 在 CTA 线程间做分区，每个线程可计算自己的子张量。分区方式同样很多。

`gemm_nt` 和 `gemm_tn` 再定义一个*线程* Layout：
```cpp
  // Define thread layouts (static)
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx; m-major
```
这是一个 M-主序的 16x16 线程 Layout，用于将 128x128 的 `C` 数据块分区，使每个线程计算自己的 8x8 子张量。

内核中同样会检查该线程 Layout 的约束。
```cpp
  static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tC) == size(tA));                          // NumThreads

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});  // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});  // BLK_N / THR_N
```

然后使用这些线程 Layout 对全局内存和共享内存中的块做分区：
```cpp
  // Partition sA (M,K) by the rows of tC
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
  // Partition sB (N,K) by the cols of tC
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)

  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));                // THR_M
  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA));                // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));                // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB));                // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));                // BLK_K
```
这里使用投影式接口，避免将 `tC` 的 `N` 模式应用到 `sA` 的 `(BLK_M,BLK_K)` 形状，也避免将 `tC` 的 `M` 模式应用到 `sB` 的 `(BLK_N,BLK_K)` 形状。

![tC_partitioning.png](../../../images/cute/tC_partitioning.png)
该图展示了 `tC` Layout，用绿色和蓝色标出两个线程，给出 `tC` Layout 的投影，并标出 `tCsA`、`tCsB`、`tCgC` 在 `sA`、`sB`、`gC` 中对应的子张量。

数据在各线程间分区后，*每个线程*都可参与计算，写法如下：
```cpp
gemm(tCsA, tCsB, tCrC);
```
因为每个线程拥有待计算数据中不同的子张量。

### 主循环 (Mainloop)

主循环遍历全局内存块，将块读入共享内存，然后执行矩阵乘并累加到累加器 (accumulator)。

```c++
// TUTORIAL: Example of a very simple compute mainloop
//   copy(.) operates on the global and shared memory via the tA|tB partitioning
//   gemm(.) operates on the shared and register memory via the tC partitioning

auto K_TILE_MAX = size<2>(tAgA);

for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
{
  // Copy gmem to smem with tA|tB thread-partitioned tensors
  copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
  copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

  cp_async_fence();        // Label the end of (potential) cp.async instructions
  cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
  __syncthreads();         // Wait for all threads to write to smem

  // Compute gemm on tC thread-partitioned smem
  gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
  __syncthreads();         // Wait for all threads to read from smem
}
```

可以看出，`k_tile` 遍历每一块数据，`cute::copy` 使用 `tA`、`tB` 线程分区张量对当前 `k_tile` 执行，`cute::gemm` 使用 `tC` 线程分区张量进行计算。通过同步保证该内核在任何架构上都能正确运行。

## `sgemm_2.cu`

该示例使用更复杂的 TiledMMA 和 TiledCopy 做分区，取代 `tA`、`tB`、`tC` 线程 Layout。借此强调：共享内存 Layout、分区模式以及各阶段使用的 PTX 指令可以独立指定。

### TiledCopy

首先可用 TiledCopy 分区替代 `tA`、`tB` 分区，以支持更复杂的分区模式，并能派发到具体 copy 指令并做检查。

先看 `gemm_nt` 生成的 TiledCopy。
```cpp
  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},  // Atom: Copy TAs as if they were uint128_t
                                    Layout<Shape<_32,_8>>{},                    // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});                   // Val layout  4x1 m-major
  print_latex(copyA);
```
理解该 TiledCopy 的最直观方式是看 LaTeX 中的分区模式。
![TiledCopyA.png](../../../images/cute/TiledCopyA.png)
左侧为源张量分区，右侧为目标张量分区。本例中分区模式相同，但存在要求源与目标使用不同模式的 PTX 指令。图中显示每个线程读取 4x1 个 `TA` 元素，共有 32x8 个线程。`UniversalCopy<uint128_t>` 强制使用 128 位 copy 指令。若对 `sA` 或 `gA` 的分区得到的 4 个 `TA` 元素无法向量化为 128 位 load/store，CuTe 会静态失败并给出相应错误信息。

使用该 TiledCopy 时，内核这样写：
```cpp
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);            // (CPY,CPY_M,CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tArA = make_fragment_like(tAsA);              // (CPY,CPY_M,CPY_K)
```
通过 `partition_S` 对 `gA` 应用源张量分区，通过 `partition_D` 对 `sA` 应用目标张量分区。结果张量的第一模式 `CPY` 保存单条指令要消费的全部元素。本例中该模式大小应为 4，因为四个 `TA=float` 元素构成一个 128 位 `uint128_t`。

完成分区后，可使用 `copy_a` 中提供的指令，在已分区的张量上执行 `copy`：
```cpp
cute::copy(copy_a, tAgA, tArA);
```

### TiledMMA

接着可用 TiledMMA 分区替代 `tC` 分区，以支持更复杂的分区模式，并能派发到具体 MMA 指令并做检查。

先看 `gemm_nt` 生成的 TiledMMA。
```cpp
  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 UniversalFMA
  print_latex(mmaC);
```
理解该 TiledMMA 的最直观方式是看 LaTeX 中的分区模式。
![TiledMmaC.png](../../../images/cute/TiledMmaC.png)
左侧为 A 张量分区，上方为 B 张量分区，中间为 C 张量分区。由于 `UniversalFMA` 是 1x1x1 MMA 指令，16x16x1 的 tiling 得到 16x16x1 TiledMMA。其他 MMA 指令会涉及不同数量的线程和不同尺寸。本例中所有线程各自从 `A`、`B`、`C` 各读一个元素。

使用该 TiledMMA 时，内核这样写：
```cpp
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);        // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);        // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);        // (MMA,MMA_M,MMA_N)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
```
通过 `partition_A` 对 `sA` 应用 A 张量分区，`partition_B` 对 `sB` 应用 B 张量分区，`partition_C` 对 `gC` 应用 C 张量分区。结果张量的第一模式 `MMA` 保存单条指令要消费的全部元素。本例中该模式大小应为 1，因为 `UniversalFMA` 是 1x1x1 MMA，但一般而言该模式大小可不同，且 `tCsA`、`tCsB`、`tCgC` 中也可能因 MMA 而不同。

完成分区后，可使用 `mma` 中提供的指令，在已分区的张量上执行 `gemm`：
```cpp
cute::gemm(mma, tCsA, tCsB, tCrC);
```

### 其他改动

本版本中，`gemm_tn` 的共享内存 Layout 从 K-主序改为：
```cpp
  // Define the smem layouts (static)
  auto sA = make_layout(make_shape (      bM,          bK),
                        make_stride(Int<1>{}, bM+Int<1>{}));  // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape (      bN,          bK),
                        make_stride(Int<1>{}, bN+Int<1>{}));  // (n,k) -> smem_idx; padded n-major
```
得到 M-主序和 N-主序 Layout，但做了填充以规避共享内存 bank 冲突。这样只是改善了共享内存的访问模式，内核其他部分无需改动。

## `sgemm_sm70.cu`

该示例为 Volta SM70 架构使用优化的主循环，对共享内存和寄存器内存做流水线 (Pipeline)。

## `sgemm_sm80.cu`

该示例为 Ampere SM80 架构使用优化的主循环，通过从全局内存的异步读取，对共享内存做显式流水线。

## 后续

以上示例均假设 CTA 块大小整除问题规模，因而无需对全局内存 load 做谓词 (predicate)。若矩阵分块无法整除矩阵，[教程的谓词部分](./0y_predication.md) 说明了如何处理。

## 将 GETT 视为 GEMM

这里的 “GETT” 表示 “general(ized) tensor times tensor”，即张量收缩。

CuTe 允许矩阵使用嵌套 `Layout`，因而可以按模式类别分组，将 `Tensor` 折叠成“矩阵”。

因此，可以用现有的 GEMM 实现来完成 GETT。下面是一个类似 `gemm_nt` 的启动器，使用 `sgemm_1.cu` 中的同一设备内核，计算具有两个 m 模式的 GETT。
```cpp
// Setup params for a GETT with two m-modes.
// The A and C tensors are assumed to be m0-major.
//   Calls sgemm_1.cu's gemm_device<<<>>> without modification.
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gett(int m0, int m1, int n, int k,
     Alpha alpha,
     TA const* A, int ldAm1, int ldAk,  // m0-major
     TB const* B, int ldBk,
     Beta beta,
     TC      * C, int ldCm1, int ldCn,  // m0-major
     cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = make_shape(m0, m1);                               // (m0,m1)-multimode M
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(make_stride(Int<1>{}, ldAm1), ldAk); // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(make_stride(Int<1>{}, ldCm1), ldCn); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Shape<_64, _2>{};    // Take _64 elements from m0 and _2 elements from m1
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (n,k) -> thr_idx
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}
```
注意，只有 shape `M`、步幅 `dA` 和 `dC`，以及 CTA Tiler `bM` 的定义有变化。上面使用 multimode 问题形状 `M = (m0,m1)` 和 multimode CTA Tiler `bM = <_64,_2>`，以改变每个 CTA 负责计算的全局内存张量 `A` 和 `C` 的部分。

可参考基于 CuTe 的 CUTLASS 3.x 内核中的类似示例，例如 [该 Hopper GETT 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/51_hopper_gett)。

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
