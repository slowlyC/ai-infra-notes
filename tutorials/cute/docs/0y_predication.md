# 谓词保护：当分块不整除时怎么办

[GEMM 教程](./0x_gemm_tutorial.md) 展示了如何通过对输入矩阵和输出矩阵的 tile 迭代来计算矩阵乘法。示例都假设 tile 能整除矩阵，没有余数。若不满足会怎样？例如，要把 41x55 的矩阵分块成 4x8 的 tile，但 41÷4 余 1，55÷8 余 7。这些“多出来的”部分怎么处理？

首先说明，`logical_divide`（CuTe 的分块 layout 方式）会“向上取整”。例如，若 `N` 是 layout `1000:1`，`B` 是 layout `128:1`，则 `logical_divide(N, B)` 是 layout `(128, 8):(1, 128)`。 实质上将原始 shape `N = 1000` 向上舍入成 128x8 的矩阵（如同 `N = 1024`）。那最后 24 个不属于原始数据的元素呢？最后一个 tile 如何正确处理，如何避免越界访问？

和其他 CUDA 编程入门一样，CuTe 惯用的做法是通过“谓词保护”(predication)。与其用“7 个 size-128 的 tile 和 1 个 size-104 的 tile”来表示余数 tile，CuTe 会向上舍入成“8 个 size-128 的 tile”，并构造谓词，使得 kernel 只访问 tile 中落在矩阵范围内的有效数据。这与 GPU 的优化方式一致：无 warp 分支的开销相对较低。也和常见的 CUDA 惯用法一致——在 1-D 上将 N 个工作项均分到 B 个线程块时，先检查“当前线程”是否越界再执行工作。

考虑一个通用分块：将长度为 1000 的向量分成长度为 128 的块。可按如下方式构造谓词 tensor：

```c++
Tensor gmem = ...     // e.g. size 1000
Tensor smem = ...     // e.g. size 128

// Tile the gmem for smem
Tensor gmem_tiled = logical_divide(gmem, size(smem));      // e.g. (128,8)

// Create an identity layout for gmem and tile it similarly
Layout id_layout = make_layout(shape(gmem));               // e.g. 1000:1, explicitly constructed as identity function
Layout id_tiled  = logical_divide(id_layout, size(smem));  // e.g. (128,8):(1,128), but many elements aren't "valid"

// Create a predicate tensor
Tensor pred = make_tensor<bool>(shape(id_tiled));          // e.g. (128,8)
for (int i = 0; i < size(pred); ++i) {
  pred(i) = id_tiled(i) < size(id_layout);  // Predicate: Is the offset within the original shape?
}

// ... intervening code ...

// Note that gmem_tiled, id_tiled, and pred tensors are all congruent
// For tile tile_i, determine if element value_j is in-bounds and copy to smem
if (pred(value_j,tile_i)) { smem(value_j) = gmem_tiled(value_j,tile_i); }
```

通用步骤是：

1. 构造与原始数据相同 shape 的“单位” layout（上例中为 `Layout id_layout = make_layout(shape(gmem))`）; 
2. 对该单位 layout 做同样的分块/分区/切片（可能向上取整），（上例中为 `Layout id_tiled = logical_divide(id_layout, size(smem));`）; 
3. 通过将该参考 layout 的坐标与原始 layout 的边界比较，构造“谓词 tensor”; 
4. 用谓词 tensor 屏蔽对越界元素的访问。

举个相对简单的例子：对 GEMM 的 epilogue 做谓词保护。假设我们已按如下方式将 `mC` 分区到 CTA tile 及 MMA 的线程上。

```cpp
// CTA partitioning
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

// Thread partitioning
auto thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

// ... Compute gemms and accumulate into tCrC ...

// axpby epilogue
for (int i = 0; i < size(tCgC); ++i) {
  tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
}
```

按谓词保护流程做即可：

```cpp
// A coordinate tensor the same shape as mC: (m,n) -> (m,n)
Tensor cC     = make_identity_tensor(shape(mC));

// Repeat partitioning steps applied to mC to our coordinate tensor cC
// CTA partitioning
Tensor cta_cC = local_tile(cC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N) -> (m,n)
// Thread partitioning
Tensor tCcC   = thr_mma.partition_C(cta_cC);                             // (MMA,MMA_M,MMA_N) -> (m,n)

// Predicated axpby epilogue
for (int i = 0; i < size(tCgC); ++i) {
  if (elem_less(tCcC(i), shape(mC))) {  // if coord is in-bounds
    tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
  }
}
```

上例中，CTA 负责对 `mC` 做分块/分区，MMA 负责对 `gC` 做分块/分区，因此这两步也都要应用到单位 tensor 上。坐标 tensor `tCcC` 与寄存器 fragment `tCrC` 以及分区后的全局内存 tensor `tCgC` 是同一 tile 数据中该线程的子张量，结构一致。但 `tCcC` 在求值时保留其原始值域：即原 tensor `mC` 的全局坐标。该全局坐标与 `mC` 的 shape 比较，用于判断操作是否有效。

这种“参考单位 tensor”或“坐标 tensor”做法的优点包括：

1. 不依赖被谓词保护 tensor 的 layout/stride，只依赖逻辑边界。
2. 分区阶段可以是任意操作。CTA 分块、线程分区、TiledMMA、TiledCopy 都可以应用到任意 tensor，包括坐标 tensor。
3. 自然扩展到任意维度的谓词保护。
4. 是常见 CUDA 1-D 并行向量访问模式的自然推广：计算访问索引 `idx`，并谓词保护对向量第 `idx` 个元素的访问，判断 `idx` 是否在范围内。
```cpp
int idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < N)  // idx is a "coord" into gmem and N is the "bound"
  gmem_ptr[idx] = ...;
```

在 SIMT 编程模型中，不应修改 tensor 的范围，以免循环越界。谓词保护是查询原始坐标并判断是否越界的通用方法。这样避免可变/动态循环边界，改为指令级谓词、保持线程一致性并维持负载均衡。该方式足够通用，可扩展到所有秩、所有线程与数据的 layout、以及所有分块/分区模式。特殊情况可以通过在坐标 tensor 或谓词 tensor 中加入假设来处理。

再举一个稍复杂的例子：GEMM 中 A 和 B 加载的 m、n 谓词保护。假设已按如下方式对 A 和 B 的 tile 做了 CTA 和线程分区。

```c++
// CTA partitioning
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

// Thread partitioning
Tensor tAgA = local_partition(gA, tA, thread_idx);                   // (THR_M,THR_K,k)
Tensor tAsA = local_partition(sA, tA, thread_idx);                   // (THR_M,THR_K)

Tensor tBgB = local_partition(gB, tB, thread_idx);                   // (THR_N,THR_K,k)
Tensor tBsB = local_partition(sB, tB, thread_idx);                   // (THR_N,THR_K)
```

`gA` 和 `gB` 是根据 `cta_tiler` 和 `cta_coord` 从 `mA` 和 `mB` 得到的 tile。`tAgA` 和 `tBgB` 是根据线程 layout `tA`、`tB` 以及 `thread_idx` 对 `gA`、`gB` 的分区。

下面的代码创建映射 `(m,k) -> (m,k)` 和 `(n,k) -> (n,k)` 的“单位 tensor”。

```c++
// Coordinate tensors
Tensor cA = make_identity_tensor(shape(mA));   // (m,k) -> (m,k)
Tensor cB = make_identity_tensor(shape(mB));   // (n,k) -> (n,k)
```

然后，对参考 tensor 做完全相同的分块和分区，就像对 `mA`、`mB` 做分块和分区得到 `tAgA`、`tBgB` 一样。

```c++
// CTA partitioning
Tensor cta_cA = local_tile(cA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k) -> (m,k)
Tensor cta_cB = local_tile(cB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k) -> (n,k)

// Thread partitioning
Tensor tAcA = local_partition(cta_cA, tA, thread_idx);                   // (THR_M,THR_K,k) -> (m,k)
Tensor tBcB = local_partition(cta_cB, tB, thread_idx);                   // (THR_N,THR_K,k) -> (m,k)
```

下面的代码创建与 `tAgA`、`tBgB` 对应的谓词 tensor。它们会在 prologue 中计算一次，并在内层循环中用于屏蔽指令。

```c++
Tensor tApA = make_tensor<bool>(make_shape (size<0>(tAcA), size<1>(tAcA)),
                                make_stride(     Int<1>{},      Int<0>{}));
Tensor tBpB = make_tensor<bool>(make_shape (size<0>(tBcB), size<1>(tBcB)),
                                make_stride(     Int<1>{},      Int<0>{}));
```

这里做了几点假设：只关心一次一个数据 tile 的谓词; 只关心 m、n 模式的谓词，k 模式的谓词另行处理。m、n 谓词在每个 tile 内视为常量，会在 mainloop 每次迭代中复用。因此只保存 m、n 模式的谓词，并在 k 模式上广播。填充 tensor 时沿用相同假设：

```c++
// Populate the m- and n-predicates
CUTE_UNROLL
for (int m = 0; m < size<0>(tApA); ++m) {
  tApA(m,0) = elem_less(get<0>(tAcA(m,0,0)), shape<0>(mA));  // Compare the m-coordinate
}
CUTE_UNROLL
for (int n = 0; n < size<0>(tBpB); ++n) {
  tBpB(n,0) = elem_less(get<0>(tBcB(n,0,0)), shape<0>(mB));  // Compare the n-coordinate
}
```

仅比较第 0 个 k-tile 和第 0 个 k-block 的 m、n 坐标。stride 为 0 的广播模式仍允许我们将该数据当作 tile 中每个待加载元素的谓词 tensor 使用。

最后，可以在 `copy_if` 中使用谓词 tensor，只拷贝对应谓词 tensor 元素为 `true` 的元素。

```c++
// Copy a k_tile from global memory to shared memory
copy_if(tApA, tAgA(_,_,k_tile), tAsA);
copy_if(tBpB, tBgB(_,_,k_tile), tBsB);
```
