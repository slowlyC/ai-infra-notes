# CUTLASS 教程：在 NVIDIA Blackwell GPU 上使用硬件支持的 Block-Scaling

**原文**: [Colfax Research - CUTLASS Tutorial Part 4](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/)

**翻译说明**: 本文翻译自 Colfax Research 的 CUTLASS Tutorial 系列 Part 4，技术术语保留英文或附中文注释。

---

欢迎来到我们探究 NVIDIA Blackwell 架构上 GEMM 系列的第 4 部分。此前我们讨论了 Blackwell 新型 Tensor Core UMMA 指令的能力，包括处理亚字节（sub-byte）数据类型，以及如何在 CUTLASS 中使用它们。本文将继续探索低精度计算，讨论如何结合 UMMA 使用 blockscaling 支持。

# Block-Scaling 回顾

上文中我们简要介绍了 blockscaling，简而言之，它是一种反量化技术：在乘加运算之前，将操作数数据乘以一个缩放因子（scale factor）。更精确地讲：

```
D = (A * scale_A) @ (B * scale_B) + C
```

在 AI 应用中，blockscaling 用于弥补低精度数值格式的动态范围不足。做法是：在量化之前，用缩放因子将原本高精度权重或激活张量的所有元素缩放到一个统一的区间。缩放因子的粒度有多种选择：一端可以逐元素缩放，另一端可以为整块矩阵使用一个统一的缩放因子。Blackwell Tensor Core 对介于两者之间的方案提供硬件支持：在稠密 GEMM 中，每个行/列在 K 维度（K-mode）被分为 16 或 32 个元素的分块，每个分块乘以自己的缩放因子。

图 1. Block-Scaled GEMM，来自 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-block-scaling)。

![Figure 1 Block-scaled GEMM](https://research.colfax-intl.com/wp-content/uploads/2026/02/image-4.png)

上图中，A 和 B 的每一行/列被分成两个分块，并分别乘以两个缩放因子。换言之，在 K-mode 下，每个 16 或 32 元素向量可以有一个缩放因子。允许的分块数量和大小取决于数据类型，下一节将讨论。

# Block-Scaled GEMM 的数据类型

在[前文](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)中，我们介绍了 Blackwell 上使用的五种亚字节浮点格式。Block-scaled GEMM 操作数矩阵的基本组成，可以视为一种新数据类型：固定长度的低精度数向量，并伴随每个向量的一个缩放因子。Blackwell blockscaling 支持五种操作数数据类型、向量长度与缩放因子类型的组合：

![Block-scaled GEMM 数据类型表](https://research.colfax-intl.com/wp-content/uploads/2026/02/image-5.png)

| Operand data type | Vector length (elements) | Scale factor data type |
| --- | --- | --- |
| mxf8 | E5M2, E4M3 | 32 | UE8M0 |
| mxf6 | E3M2, E2M3 | 32 | UE8M0 |
| mxf4 | E2M1 | 32 | UE8M0 |
| nvf4 | E2M1 | 16 | UE4M3 |

缩放因子始终是无符号 8 位浮点数。用于 `nvf4` 的 `UE4M3` 类型本质上是非负的 `E4M3` 浮点数（符号位恒为 0）。而 `UE8M0` 类型则用全部 8 位以标准偏置方式表示浮点指数——因此 `UE8M0` 缩放因子的可能值为 2^x，其中 -127 ≤ x ≤ 127。两者都支持 NaN，不支持无穷大。与 `UE8M0` 相比，`UE4M3` 精度更高但范围显著缩小——最大可能值仅为 448，使得 `nvf4` 向量可表示的最大值为 6 × 448 = 2688。

三种 mx 类型在 Open Compute Project 的 Microscaling Format 规范中定义（[PDF](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)），而 `nvf4` 格式为 [NVIDIA 专有](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。与 mx 类型相比，`nvf4` 提供更细粒度的缩放因子和每个缩放因子更少的元素，但相应地会多用一倍的字节用于存储缩放因子。

# Block-Scaled UMMA 的 PTX 语法

带 blockscaling 的 UMMA PTX 指令语法如下：

```
tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
                                        [d-tmem],  a-desc,  b-desc, idesc,
                                        [scale-A-tmem], [scale-B-tmem],enable-input-d;

tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
                                        [d-tmem], [a-tmem], b-desc, idesc,
                                        [scale-A-tmem], [scale-B-tmem],enable-input-d;

.kind = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
.cta_group      = { .cta_group::1,   .cta_group::2 }
.scale_vectorsize = { .scale_vec::1X, .scale_vec::2X, .scale_vec::4X, .block16, .block32 }
```

[Part 1](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) 中已介绍过该语法的多数内容，包括指令描述符、A 和 B 的 SMEM 描述符、从 TMEM 而非 SMEM 读取 A 的能力，以及 `enable-input-d` 标志（用于累加到 D 而非覆盖）。对于 blockscaled 指令，缩放因子必须从 TMEM 读取; `scale-A-tmem` 和 `scale-B-tmem` 参数接收其基地址，即它们 (0, 0) 元素在 TMEM 中的地址。除缩放因子的 TMEM 布局外，还需说明 `.kind` 和 `.scale_vectorsize` 限定符。

## .kind

`.kind` 限定符有三种选项：

- `mxf4nvf4` —— 4 比特输入的更通用指令。
- `mxf4` —— 4 比特输入，使用 `ue8m0` 缩放因子。
- `mxf8f6f4` —— 混合输入，支持 8、6 和 4 比特数据类型。

限定符类型决定了可用的操作数数据类型和缩放因子类型。

`mxf8f6f4` 限定符是前文讨论的 `f8f6f4` 数据类型的 blockscaled 版本，要求与 `f8f6f4` 完全相同：相同的操作数输入类型，以及相同的 16 字节 SMEM/TMEM 填充要求。因此我们参考前文了解 `mxf8f6f4` 的操作数。

`mxf4` 和 `mxf4nvf4` 仅适用于 4 比特输入，具体为 `e2m1`。使用 4 比特专用版本的优点是：与 `mxf8f6f4` 不同，4 比特数据类型无需填充。相反，两个元素可打包进单个 8 比特容器：

图 2. SMEM 中 4 比特值的打包方式，来自 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-packing-formats-mxf4-tmem-dig1)。

![Figure 2 4-bit packing](https://research.colfax-intl.com/wp-content/uploads/2026/02/image-11.png)

这使 SMEM 用量相比使用 `mxf8f6f4` 限定符的 4 比特数据类型减半。因此，若已知工作负载仅使用 fp4，建议使用 `mxf4` 或 `mxf4nvf4`。

`mxf4` 还假定缩放因子类型为 `ue8m0`，而 `mxf4nvf4` 下两种缩放因子类型均可。缩放因子数据类型与操作数类型一样，在运行时通过[指令描述符](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor)指定。

## .scale_vectorsize

将 `.scale_vectorsize` 限定符设为 `.block16` 或 `.block32`，可指定每个缩放因子对应的操作数元素数量：mx 类型为 32，`nvf4` 为 16。

但在内部，Tensor Core 似乎以另一种方式理解向量大小。回顾：A 和 B 的 UMMA 输入原子（atom）在 K-mode 下始终为 32 字节宽（我们将 K-mode 视为两个矩阵的行模式，称之为“UMMA atom rows”）。我们用 `atom_K` 表示 MMA 原子的大小，因此 `mxf8f6f4` 时 `atom_K = 32`，`mxf4` 和 `mxf4nvf4` 时 `atom_K = 64`。与前文一致，主循环 tile 的大小记为 (bM, bN, bK)，通常由若干 UMMA 原子在 K-mode 重复组成。本文中始终令 `bK` 等于 4 个 UMMA 原子（128 字节或 1 缓存行），故 8 比特输入为 `bK = 128`，4 比特输入为 `bK = 256`。

指定 scale vector size 等价于指定一个 UMMA atom row 消耗的缩放因子数量：

```
atom_SFK = atom_K / sf_vec_size
```

`.block16` 和 `.block32` 限定符实际上是 `.scale_vec::1X`、`2X` 和 `4X` 的别名，其中 1、2 或 4 表示每个 UMMA atom row 的缩放因子数量。这会直接影响 UMMA 消耗的缩放因子的形状（见下表），也会影响 TMEM 中的缩放因子布局，下一节会说明。

![scale_vectorsize 表格](https://research.colfax-intl.com/wp-content/uploads/2026/02/image-12.png)

| .scale_vec::1X | .scale_vec::2X | .scale_vec::4X |
| --- | --- | --- |
| Shape of scale_A | M x 1 | M x 2 | M x 4 |
| Shape of scale_B | N x 1 | N x 2 | N x 4 |

并非所有选项都适用于所有数据类型。完整表格见 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-scale-valid-comb-detail)。值得注意的是，block32 是 `mxf8f6f4` 和 `mxf4` 操作数类型唯一支持的选项。前者 `.scale_vec`=1X，因为唯一支持的 `atom_K` 为 32; 后者 `.scale_vec`=2X，因为唯一支持的 `atom_K` 为 64。`mxf4nvf4` 支持 block16（等价于 `.scale_vec`=4X）或 block32（`.scale_vec`=2X）; block32 必须与 `E8M0` 搭配，block16 可与 `E8M0` 或 `E4M3` 搭配。

# 缩放因子布局

最后讨论 UMMA 消费缩放因子时所需的存储方式。UMMA 从 TMEM 读取缩放因子。TMEM 中缩放因子的布局取决于 `.scale_vec` 的取值。本节通过 1X、2X 和 4X 三种情形说明。为简便，我们限定为稠密 MMA，且每个 CTA 的 `bM`=128。`bN` 在 8 到 256 之间可变。

## mxf8f6f4 的 block32/1X（atom_K=32）

该格式是 `mxf8f6f4` 数据类型唯一可用的选项，仅用于该类型。先从 A 矩阵说起。每个 MMA 块的缩放因子向量为 M×1。该向量预期存放在 32 lane × 4 列的 tile 中、1 字节对齐的子列（每列包含 4 个子列）内，子列由 2 位 `SFA_ID` 索引。

图 3. `.scale_vec::1X` 的 TMEM 缩放因子布局，来自 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-scale-factor-a-1x-dig)。

![TMEM scale factor 1X](https://research.colfax-intl.com/wp-content/uploads/2026/02/image-10.png)

该图表示 4 种不同的 UMMA，每种对应一个子列。例如，一个 UMMA 使用子列 `SFA_ID`=00 中 4 列上的缩放因子，另一个使用 `SFA_ID`=01 的值，以此类推。UMMA 使用的子列在指令描述符中设置。参见 [PTX 文档表 43](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instruction-descriptor)（指令描述符第 29–30 位用于 `SFA_ID`）。因假设 `bM`=128，SFA 始终占用 TMEM 的 4 列。

SFB 矩阵的格式与 SFA 完全相同，但列数可变，依选定的 bN 值（8 到 256）在 1 到 8 之间。两种缩放因子最多共需 TMEM 12 列。

注意：虽然这些 sf tile 只使用 TMEM 的 32 个 lane，后面会看到其余 96 个 lane 也被占用，无法另作他用。

## mxf4/mxf4nvf4 的 block32/2X（atom_K=64）

该格式是 `mxf4` 数据类型唯一可用的选项，同时用于 `mxf4` 和 `mxf4nvf4`。再次从 A 矩阵开始。每个 MMA 块的缩放因子向量为 M×2。该向量预期存放在两个相邻的、2 字节对齐的子列中。子列仍由起始子列的 2 位 `SFA_ID` 索引，两种选择为 00 和 10。

图 4. `.scale_vec::2X` 的 TMEM 缩放因子布局，来自 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-scale-factor-a-2x-dig)。

![TMEM scale factor 2X](https://research.colfax-intl.com/wp-content/uploads/2026/03/image-3.png)

该图表示 2 种 UMMA：一种使用 `SFA_ID`=00 中的缩放因子，另一种使用 `SFA_ID`=10。B 的缩放因子格式同样，仅列数可变。由于 4 比特输入使 `bK`=256，缩放因子对 TMEM 的需求翻倍：SFA 8 列，SFB 最多 16 列。

## mxf4nvf4 的 block16/4X（atom_K=64）

该格式仅适用于 `mxf4nvf4` 数据类型。同样从 A 矩阵开始，每个 MMA 块的缩放因子向量为 M×4。此时唯一有效的 SFA_ID 为 00。

图 5. `.scale_vec::4X` 的 TMEM 缩放因子布局，来自 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-scale-factor-a-4x-dig)。

![TMEM scale factor 4X](https://research.colfax-intl.com/wp-content/uploads/2026/02/image-6.png)

该图对应单个 UMMA。虽然只有一个有效的 `SFA_ID`=00，但它仍需要在指令描述符中使用。B 的格式同样，仅列数可变。每个主循环 tile 的缩放因子数量翻倍，因此需要更多 TMEM：SFA 16 列，SFB 最多 32 列，合计最多 48 列。

# CUTLASS Block-Scaling 实现

接下来讨论 CUTLASS 中 blockscaling 的实现，参考 CuTeDSL 示例 [dense_blockscaled_gemm_persistent.py](https://github.com/NVIDIA/cutlass/blob/3476ddb7bd6ca4161a0169103ceaa20ce0eb891f/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py)，重点关注与标准 UMMA 的差异。

## 操作数

首先是操作数。前文已讨论 `f8f6f4` 在 SMEM 中所需的数据格式，以及亚字节类型的特殊 TMA tensor map; `mxf8f6f4` 使用完全相同的条件和指令。Block-scaled GEMM 的 tile 大小约束更严格：1-CTA MMA 要求 `bM`=128，2-CTA MMA 要求 128 或 256。为简化，我们忽略 2-CTA MMA 且 `bM`=128 的情况。

使用 `mxf4` 或 `nvf4` 时，之前看到数据在 SMEM 中被打包成 1 字节。与其他亚字节类型一样，有专门的 TMA tensor map `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B` 用于该 TMA 操作。由于是打包数据类型，无需改动 layout; CUTLASS 在底层抽象了 TMA 的亚字节特性。

## 缩放因子布局

缩放因子始终为 8 比特 dtype，可像其他 8 比特 dtype 一样通过 TMA 加载。但这类加载有一处不同：缩放因子最终必须按前述布局组织在 TMEM 中供 Tensor Core 消费。最直接的加载方式是在 GMEM 中采用相同布局。来自 [CUTLASS 文档](https://docs.nvidia.com/cutlass/4.3.4/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts) 的下图展示了按该布局组织的 SFA GMEM tile：

图 6. 交错布局下 SFA tile 的 GMEM 布局。注意整块 tile 在 GMEM 中应是连续的。来自 [CUTLASS 文档](https://docs.nvidia.com/cutlass/4.3.4/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts)。

![Figure 6 GMEM SFA layout](https://research.colfax-intl.com/wp-content/uploads/2026/03/image.png)

该 512B tile 将在整个 SFA 张量上平铺：

图 7. SFA 的 GMEM 布局，由图 6 的基础 tile 在 SFA 上平铺得到。来自 [CUTLASS 文档](https://docs.nvidia.com/cutlass/4.3.4/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts)。

![Figure 7 GMEM SFA tiled](https://research.colfax-intl.com/wp-content/uploads/2026/03/image-1.png)

在平铺操作中，常将缩放因子向量大小（即 block16 或 block32）作为广播维度（stride 为 0）纳入 shape，并将静态 mode 组合在一起。例如 block16 的广播 SFA layout 为：

```
(((32, 4), REST_M), ((16, 4), REST_K)) : (((16, 4), 512 * REST_K), ((0, 1), 512))
```

采用这种交错布局的 tile 可在向量化、合并、无 bank 冲突的方式下，从 GMEM 透明加载到 SMEM，再从 SMEM 到 TMEM。这与缩放因子向量大小或 MMA k-tile 大小无关——它们只决定需要从 A 加载的对应数据，以及上述 tile 对应的 MMA atom 数量。

朴素量化得到的缩放因子张量可能是简单 K-major。此时需要做置换并使其连续，才能得到交错布局：

```
def interleave_sf_tensor(sf: torch.Tensor) -> torch.Tensor:
    M, SF_K = sf.shape
    REST_M = M // 128
    REST_K = K // 4
    # Reshape M -> (REST_M, 4, 32), SF_K -> (REST_K, 4)
    out = sf.reshape(REST_M, 4, 32, REST_K, 4)
    # Permute to (REST_M, REST_K, 32, 4, 4)
    # and make contiguous to get right strides
    out = out.permute(0, 3, 2, 1, 4).contiguous()
    # Permute to (32, 4, REST_M, 4, REST_K)
    out = out.permute(2, 3, 0, 4, 1)
    return out
```

注意：我们没有 unsqueeze 来得到广播 mode，也没有像 cute layout 那样对 mode 分组，因为 torch 张量不支持; 也可以直接返回 shape 为 `(REST_M, REST_K, 32, 4, 4)` 的连续张量。实际在 kernel 内部，缩放因子张量会被赋予合适的 cute layout。

此外，若量化数据由上游 kernel 生成，该 kernel 也可以直接以这种交错格式写出缩放因子，从而省去额外的内存搬运 kernel。

## Tiled MMA

CuTeDSL 提供辅助函数 `make_blockscaled_trivial_tiled_mma`，用于定义 kernel 使用的 tiled MMA：

```
tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
    self.a_dtype,
    self.a_major_mode,
    self.b_major_mode,
    self.sf_dtype,
    self.sf_vec_size,
    self.cta_group,
    self.mma_inst_shape_mn, 
)
```

在辅助函数内部，可以看到与前述 PTX 指令大致对应的对象：

```
if ab_dtype in {Float8E4M3FN, Float8E5M2}:
    mma_op = MmaMXF8Op(
        ab_dtype,
        (*mma_tiler_mn, 32), # mma instruction shape, e.g. (128, 256, 32)
                             # atom_K must be 32 bytes
        cta_group,           # specifies 1 or 2 CTA UMMA
        a_source,            # can be SMEM or TMEM
        a_leading_mode,      # mxfp8 allows A and B operands to be either major
        b_leading_mode,
    )
elif ab_dtype == Float4E2M1FN:
    # atom_K = 64 for an instruction, and operands must be K-major
    if sf_vec_size == 32:
        mma_op = MmaMXF4Op(
            (*mma_tiler_mn, 64),
            cta_group,
            a_source,)
    elif sf_vec_size == 16:
        mma_op = MmaMXF4NVF4Op(
            sf_dtype,        # can be either E8M0 or E4M3
            (*mma_tiler_mn, 64),
            cta_group,
            a_source,)
return cute.make_tiled_mma(
    cute.make_mma_atom(mma_op, loc=loc, ip=ip), loc=loc, ip=ip)
```

回顾：`MXF8` 时 atom_K 必须为 32，`MXF4/MXF4NVF4` 时为 64。考虑到 TMEM 缩放因子的结构与交错布局，一次性加载足够计算 4 个 MMA atom 的数据是合理的，从而得到如下 mma_tiler（且 `bK = 4 * atom_K`）：

```
mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
mma_inst_tile_k = 4
self.mma_tiler = (
    self.mma_inst_shape_mn[0],
    self.mma_inst_shape_mn[1],
    mma_inst_shape_k * mma_inst_tile_k,
)
```

## 操作数与缩放因子的 TMA 加载

tiled MMA 随后可用于通过更多辅助函数定义 TMA atom：

```
a_op = sm100_utils.cluster_shape_to_tma_atom_A(
    self.cluster_shape_mn, tiled_mma.thr_id
)
a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
    a_op,
    a_tensor,
    a_smem_layout,
    self.mma_tiler,
    tiled_mma,
    self.cluster_layout_vmnk.shape,
)
```

相同方法可用于构建 SFA 的 TMA atom。尽管采用交错布局，SFA 的每个 128×`sf_tile_size_k` 在 GMEM 中仍是连续的，这正是每次 TMA 调用时一个 CTA 要加载的量。

注意：上述 `tma_atom_a` 和 `tma_tensor_a` 在 host 上创建后作为参数传入 device 代码，其中 `tma_tensor_a` 重命名为 `mA_mkl`，一系列操作为每个 CTA 和每次主循环迭代的 `g2s` 加载提供正确信息。B 和 SFA 的操作类似，也与 CUTLASS C++ 中类似。例如，下面跟踪 GMEM SFA 张量的处理序列：

```
# (bM, bK, RestM, RestK, RestL)
gSFA_mkl = cute.local_tile(
    mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
)
...
# (MMA, MMA_M, MMA_SFK, RestM, RestK, RestL)
tCgSFA = thr_mma.partition_A(gSFA_mkl)
# ((atom_v, rest_v), RestM, RestK, RestL)
tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
    tma_atom_sfa,
    block_in_cluster_coord_vmnk[2],
    sfa_cta_layout,
    cute.group_modes(sSFA, 0, 3),
    cute.group_modes(tCgSFA, 0, 3),
)
tAsSFA = cute.filter_zeros(tAsSFA)
tAgSFA = cute.filter_zeros(tAgSFA)
...
# after assignment of worktiles:
# ((atom_v, rest_v), RestK)
tAgSFA_slice = tAgSFA[
    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
]
…
cute.copy(
    tma_atom_sfa,
    tAgSFA_slice[(None, ab_producer_state.count)],
    tAsSFA[(None, ab_producer_state.index)],
    ...
)
```

注意：`gSFA_mkl` 并不是对 `mSFA_mkl` 的实际切片，而只是 rearrangement。由于 kernel 使用持久化 tile 调度器，这分离出与具体 work tile 无关的逻辑——我们保留 `RestM` mode，直到分配 work tile 的代码处，再在 TMA copy 调用前将 `tAgSFA` 切片为 `tAgSFA_slice`。

SFA 和 SFB 在 kernel 中与 A、B 同时需要，因此可通过同一 TMA 流水线加载。

辅助函数 `make_smem_layout_sfa` 和 `make_smem_layout_sfb`（来自 `cutlass.utils.blockscaled_layout`）用于构建适合 GMEM -> SMEM -> TMEM 拷贝的缩放因子 SMEM layout。

对于 `mxf8` 和 128×256 tile，它们如下所示：

```
# sfa_smem_layout_staged:
# (((sf_tile_M, rest_atom_M), (sf_vec_K, rest_atom_K)), MMA_M, MMA_K, STAGE)
((((32,4),1),(32,1)),1,4,4):((((16,4),0),(0,0)),0,1,512)

# sfb_smem_layout_staged:
# (((sf_tile_N, rest_atom_N), (sf_vec_K, rest_atom_K)), MMA_N, MMA_K, STAGE)
((((32,4),2),(32,1)),1,4,4):((((16,4),512),(0,0)),0,1,1024)
```

注意以下几点：

`sfa_smem_layout_staged` 的 layout 与图 4 的 TMEM 图对应：

- (32, 4) : (16, 4) 对应 sf_tile_M —— 32 行缩放因子对应 32 行 MMA A（在 GMEM 和 SMEM 中以 tile 的 1 行，即 16 个值做 stride），接下来 32 行 MMA A 在缩放因子布局中 4 列后重复，依此类推。
- 4 : 1 对应 MMA_K —— 4 个连续 SFA 元素用于 K 方向重复的 4 个 MMA atom。
- 此例中 32 也是 MMA atom 的 K 维度。
- 32 : 0 对应 sf_vec_K —— 一个 SFA 元素应用于 A 在 K 方向的 32 个元素。

若改为 `nvf4` GEMM（对应图 5 的 `.block16/.scale_vec::4X` TMEM 布局），缩放因子 tile 将如下：

```
# sfa_smem_layout_staged: 
# (((sf_tile_M, rest_atom_M), (sf_vec_K, rest_atom_K)), MMA_M, MMA_K, STAGE)
((((32,4),1),(16,4)),1,4,3):((((16,4),0),(0,1)),0,512,2048)

# sfb_smem_layout_staged: 
# (((sf_tile_N, rest_atom_N), (sf_vec_K, rest_atom_K)), MMA_N, MMA_K, STAGE)
((((32,4),2),(16,4)),1,4,3):((((16,4),2048),(0,1)),0,512,4096)
```

此时注意：

- 缩放因子使用的总字节数相比 block32/1X 增加 4 倍。
- `sfb_smem_layout_staged` 的 `rest_atom_N` mode 的 stride 为 2048——因此 UMMA atom 一半的缩放因子在 SMEM 中跨多个 SF tile 存放。后面会看到 TMEM 中两半的缩放因子是相邻的，因此 `s2t` copy 会做一些置换。
- `MMA_K` mode 的 stride 为 512——为容纳 4 个 MMA atom，需要 4 个缩放因子 tile，每个 32×4×4=512 元素。
- `sf_vec_K` 缩小到 16，而 `rest_atom_K`（主循环 tile）增加到 4，因为每个 MMA atom 消耗 64 个 K 值。这也意味着每个 MMA atom 消耗 SFA 和 SFB 的完整交错 tile。
- 每个缩放因子 tile 在 SMEM 中连续，并由 warp 级 `tcgen05.cp` 指令拷贝到 TMEM，因此无需 swizzle。

我们同时给出 block32/2X 的 layout，理解它们留给读者作为练习：

```
# sfa_smem_layout_staged: 
# (((sf_tile_M, rest_atom_M), (sf_vec_K, rest_atom_K)), MMA_M, MMA_K, STAGE)
((((32,4),1),(32,2)),1,(2,2),4):((((16,4),0),(0,1)),0,(2,512),1024)

# sfb_smem_layout_staged: 
# (((sf_tile_N, rest_atom_N), (sf_vec_K, rest_atom_K)), MMA_N, MMA_K, STAGE)
((((32,4),2),(32,2)),1,(2,2),4):((((16,4),1024),(0,1)),0,(2,512),2048)
```

## 将缩放因子数据加载到 TMEM

加载到 SMEM 后，缩放因子还需加载到 TMEM。这通过异步的 [tcgen05.cp 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-cp) 完成。与 `tcgen05.ld` 和 `tcgen05.st` 类似，`tcgen05.cp` 只能在[极有限的若干模式](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-movement-shape)下移动数据，但对这类 kernel 已足够。该操作应由发出 MMA 的 warp 执行，因为 SMEM -> TMEM copy（`tcgen05.cp`）和 MMA 指令（`tcgen05.mma`）都是异步指令，在同一条内部流水线上按序执行。

在 MMA warp 的分支中可以看到：

```
# Accumulator TMEM tensor
acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
# (MMA, MMA_M, MMA_N, STAGE)
# ((128,256),1,1,1):((65536,1),0,0,0)          
tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

# SFA TMEM tensor
sfa_tmem_ptr = cute.recast_ptr(
    acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
    dtype=self.sf_dtype,
)
tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
    tiled_mma,
    self.mma_tiler,
    self.sf_vec_size,
    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
)
tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
# Construction of tCtSFB is similar
```

实用函数 `find_tmem_tensor_col_offset` 如其名，返回输入张量在 TMEM 中占用的列数（以 32 比特 cell 为单位）。对于 mxf8 且 (128, 256) tile 大小：

```
tcgen05.find_tmem_tensor_col_offset(tCtAcc_base) = 256
tcgen05.find_tmem_tensor_col_offset(tCtSFA) = 4
tcgen05.find_tmem_tensor_col_offset(tCtSFB) = 8
```

符合预期。

打印 `tCtSFA` 和 `tCtSFB` 得到：

```
# tCtSFA:
# (((atom_M, multicast_M), (sf_vec_K, rest_atom_K)), MMA_M, MMA_K)
((((32,4),4),(32,1)),1,4):((((262144,4),8388608),(0,0)),0,1)
# tCtSFB: 
# (((atom_N, multicast_N), (sf_vec_K, rest_atom_K)), MMA_N, MMA_K)
((((32,8),4),(32,1)),1,4):((((262144,4),8388608),(0,0)),0,1)
```

形状与 SMEM 中这些张量几乎一致，但有几个数字值得说明：

- MMA_K 的 4 : 1 印证了这一点——这些独立的缩放因子是相邻字节，位于同一 TMEM 列。
- 32 : 262144 对应 SFA 的 32 个 lane。如[本系列 Part 1](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)所见，TMEM 中相邻 lane 的地址 stride 为 65536（参见 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-layout)）。但 TMEM 列为 4 字节宽，CUTLASS 在内部为地址增加两个低位以跟踪字节在列内的位置。因此从 CUTLASS 角度看，字节级数据的 lane 间 stride 为 4 × 65536 = 262144。
- 4 : 8388608，其中 8388608 = 32 × 262144，是我们称为“multicast”的新 mode。如 [Part 1](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) 所述，一个 warp 通常只能从 TMEM 的 32 个 lane 加载或存储，对应其在其 warpgroup 中的位置。但通过 `tcgen05.cp`，一个 warp 可将相同数据拷贝到全部 4 个 32-lane 象限，这正是此处实现的方式。具体来说，kernel 的 [mainloop_s2t_copy_and_partition 方法](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py#L1534) 中构建的 s2t copy 是 CUTLASS 类 `cute.nvgpu.tcgen05.Cp4x32x128bOp` 的实例，使用 `.shape = .32x128b`（即 1 个 SF tile）和 `.multicast = .warpx4` 封装 `tcgen05.cp`。从 MMA 角度看，[PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-block-scaling) 确认：“A 和 B 矩阵的缩放因子需要复制到张量内存的所有 32 个 lane 分区。”

因此，此情形下的 TMEM 布局如下：

图 8. mxf8 GEMM、128×256 tile 下的 TMEM 布局。

![Figure 8 mxf8 GEMM TMEM 布局](https://research.colfax-intl.com/wp-content/uploads/2026/02/image-10.png)

`s2t` copy 的打印输出如下：

```
  Tiled Copy
  Tiler MN:        (512:1,1:0,4:1)
  TV Layout tiled: (1,(32,(4,4),4)):(0,(1,(512,32),128))
Copy Atom
  ThrID:           1:0
  TV Layout Src:   (1,(4,128,4)):(0,(1,4,0))
  TV Layout Dst:   (1,2048):(0,1)
  Value type:      f8E8M0FNU
```

可见 value layout 的大小是单个图示 atom 中 32×16 个 SFA 元素的 4 倍，但 source 有 4:0 的广播 mode，对应 multicast。与 [UMMA 的 tiled MMA](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) 一样，ThrID 索引的是参与 MMA 的 CTA，而非线程。

再对比 `nvf4`（`.block16/.scale_vec::4X`）的打印：

```
# (((atom_M, multicast_M), (sf_vec_K, rest_atom_K)), MMA_M, MMA_K)
((((32,4),4),(16,4)),1,4):((((262144,4),8388608),(0,1)),0,16)
# (((atom_N, multicast_N), (sf_vec_K, rest_atom_K)), MMA_N, MMA_K)
((((32,8),4),(16,4)),1,4):((((262144,4),8388608),(0,1)),0,32)
```

与 SMEM 中一样，`tCtSFA` 中相邻 UMMA atom 的 SF 值相隔 16 列（即一个 SF tile），而 block32/1x 下仅相隔 1 列。与 SMEM 不同，`tCtSFB` 中对应例如 UMMA atom 内 `N`=0 和 `N`=128 的 SF 值位于相邻 SF tile，而非相隔 4 个 SF tile。

block16/4x 下占用 TMEM 的对象如下：

图 9. nvf4 GEMM、128×256 tile 下的 TMEM 布局。

![图 9 nvf4 GEMM TMEM 布局](https://research.colfax-intl.com/wp-content/uploads/2026/02/image-6.png)

## 发出 GEMM

最后看主循环：

```
for k_tile in range(k_tile_cnt):
    if is_leader_cta:
        # Conditionally wait for AB buffer full
        ab_pipeline.consumer_wait(
            ab_consumer_state, peek_ab_full_status
        )
        # Copy SFA/SFB from smem to tmem
        s2t_stage_coord = (
            None,
            None,
            None,
            None,
            ab_consumer_state.index,
        )
        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t_staged,
            tCtSFA_compact_s2t,
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t_staged,
            tCtSFB_compact_s2t,
        )
        # tCtAcc += (tCrA * tCrSFA) @ (tCrB * tCrSFB)
        num_kblocks = cute.size(tCrA, mode=[2])
        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
            kblock_coord = (
                None,
                None,
                kblock_idx,
                ab_consumer_state.index,
            )
            # Set SFA/SFB tensor to tiled_mma
            sf_kblock_coord = (None, None, kblock_idx)
            tiled_mma.set(
                tcgen05.Field.SFA,
                tCtSFA[sf_kblock_coord].iterator,
            )
            tiled_mma.set(
                tcgen05.Field.SFB,
                tCtSFB_mma[sf_kblock_coord].iterator,
            )
            cute.gemm(
                tiled_mma,
                tCtAcc,
                tCrA[kblock_coord],
                tCrB[kblock_coord],
                tCtAcc,
            )
            # Enable accumulate on tCtAcc after first kblock
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        # Async arrive AB buffer empty
```

需要注意几点：

- 根据 PTX 文档，`s2t` copy 是异步的，但我们没有在 `s2t` copy 和 gemm 调用之间看到同步代码。这是因为 `tcgen05.cp` 和 `tcgen05.mma` 形成隐式的 [“tcgen05 pipeline”](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-consistency-model-pipelined-instructions)，保证按指令发出的顺序执行。这也解释了 TMEM 中缩放因子 tile 为何不使用循环缓冲：MMA 会等待最后发出的 `tcgen05.cp` 完成，因此无法重叠两条指令。
- `ab_consumer_state` 流水线状态在两个地方使用：决定将 A 和 B 的哪些 SMEM tile 送入 gemm 调用，以及决定将 SFA 和 SFB 的哪些 SMEM tile 拷贝到 TMEM。TMEM 中的缩放因子 tile 不使用循环缓冲。
- 为保持 `cute.gemm` 的语法，缩放因子 TMEM 张量并非其参数。相反，每次 gemm 调用前需要将 SFA 和 SFB 字段设为 TMEM 中的正确起始地址。

## Pair-UMMA

看一下 2-CTA UMMA、tile 大小 (256, 256) 时的变化。不逐一展开受影响的对象，可以观察到：TMA 拷贝的 SFA 数据像操作数 A 一样在成对 CTA 之间拆分，但每个 CTA 仍然收到 SFB 的两块 tile。打印 kernel 为 SFA 和 SFB 选定的 TMA copy atom 可见：

```
sfa_op: cp.async GMEM -> SMEM bulk tensor copy Operation
  CTA group = 2
sfb_op: cp.async GMEM -> SMEM bulk tensor multicast copy Operation
  CTA group = 2
```

因此 SFB 的 TMA 加载将数据组播（multicast）到两个 CTA。（“CTA group = 2” 表示两个 CTA 都在 leader CTA 的流水线屏障上到达，如 [Part 2](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/) 所述。）

对于 s2t copy，tiled copy 对象与 1-CTA 情况类似，但 `ThrID` 为 2，对应 2 个 CTA：

```
tiled_copy_s2t_sfa: 
Tiled Copy
  Tiler MN:        (512:1,1:0,4:1)
  TV Layout tiled: (2,(32,(4,4),4)):(0,(1,(512,32),128))
Copy Atom
  ThrID:           2:1
  TV Layout Src:   (2,(4,128,4)):(0,(1,4,0))
  TV Layout Dst:   (2,2048):(0,1)
  Value type:      f8E8M0FNU
```

在 PTX 中，这对应 `tcgen05.cp` 上的 `.cta_group::2` 限定符，意味着尽管只有 leader CTA 发出 `s2t` copy，copy 对两个 CTA 都以相同方式执行。

因此，在 `s2t` copy 结束时，pair 中的每个 CTA 在其 TMEM 中拥有 SFA 的不同一半（在其 4 组 32 lane 上各 multicast 4 次），两个 CTA 则拥有相同的 SFB tile（同样 multicast）。

## bN = 64 与 bN = 192

由于每个缩放因子 tile 对应 `M` 或 `N` 方向上的 128 个值，当 UMMA atom 在 `M` 和 `N` 方向上的形状不是 128 的倍数时，会出现额外复杂性。[dense_blockscaled_gemm_persistent.py](https://github.com/NVIDIA/cutlass/blob/3476ddb7bd6ca4161a0169103ceaa20ce0eb891f/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py) 支持 `bN`=64 和 `bN`=192 两种情形，尽管这些思路可推广到所有可能的 `bN`。

理想情况下，`bN`=64 和 `bN`=192 分别只需加载 0.5 和 1.5 个 SFB tile，但若用交错布局做这类加载会是非合并的。因此示例 kernel 的做法是：对 `g2s` 和 `s2t` 都向上取整到最近的整块 tile，并用额外逻辑确保 MMA 期间消费正确的缩放因子。

CuTeDSL 示例通过一个假的 TiledMMA `tiled_mma_sfb` 构造正确的 layout 和 copy（其中 `N` mode 已向上取整到最近的 128，M mode 按 CTA 划分以正确 multicast），使用同一套辅助函数：

```
self.mma_inst_shape_mn_sfb = (
    self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
    cute.round_up(self.mma_inst_shape_mn[1], 128),
)
...
tiled_mma_sfb = sm100_utils.make_tiled_mma(
    ...,
    cute.nvgpu.tcgen05.CtaGroup.ONE,
    self.mma_inst_shape_mn_sfb,
)
```

SFB 的其余对象和方法基于这些取整，使 SFB 的 `g2s` 字节数、SMEM 占用和 `s2t` 字节数：`bN`=192 与 256 相同，`bN`=64 与 128 相同。例如，打印 `bN`=192 时 B 和 SFB 的 SMEM layout：

```
# ((atom_N, atom_K), MMA_N, MMA_K, STAGE)
b_smem_layout_staged: S<3,4,3> o 0 o ((192,32),1,4,5):((128,1),0,32,24576)
# (((sf_tile_N, rest_atom_N), (sf_vec_K, rest_atom_K)), MMA_N, MMA_K, STAGE)
sfb_smem_layout_staged: ((((32,4),2),(32,1)),1,4,5):((((16,4),512),(0,0)),0,1,1024)
```

可见：`b_smem_layout_staged` 第一 mode 的 shape 准确反映了 UMMA atom shape，而 `sfb_smem_layout_staged` 与 bN=256 相同，只是 stage mode 因更小的 B tile 可容纳更多 stage 而增加。

在通用情况定义 `tma_atom_sfb` 和 `tma_tensor_sfb` 后，针对 `bN`=192 有一块 `constexpr` 条件，会修改 `tma_tensor_sfb` 的 `ArithTuple` layout。这些操作得到如下 `tBgSFB`：

```
# ((atom_v, rest_v), RestN, RestK, RestL)
(((16,32,2),1),(2,16),64,(1,1)):(((1@0,1@1,1@2),0),(1@2,3@2),1@3,(0,1@4))
```

与 bN=256 下同一张量对比：

```
# ((atom_v, rest_v), RestN, RestK, RestL)
(((16,32,2),1),32,64,(1,1)):(((1@0,1@1,1@2),0),2@2,1@3,(0,1@4))
```

ArithTuple 有 5 维，依次为 SFB tile 的行、列、`N` 中 SFB tile 坐标、`K` 中 SFB tile 坐标、batch mode `L` 中 SFB tile 坐标（本文假定其平凡）。

回忆：TMA copy 调用加载所提供张量的第一 mode，故 `bN`=192 和 256 都加载两片 SFB tile，但 `bN`=192 时 `RestN` mode 看起来特殊——为 (2,16) : (1@2, 3@2)。即：work tile 的 `N` 坐标步进 1 仅移动 1 个 SFB tile，而 `N` 方向步进 2 则移动 3 个 SFB tile。换言之，`N` 方向上的奇序 work tile 每次步进 1 个 SFB tile，偶序 work tile 每次步进 2 个 SFB tile。示意如下：

图 10. `bN`=192 时 SFB 的 TMA 加载模式。每个偶序 `N` 坐标的 work tile 及紧随其后的奇序 work tile 都会加载中间的 SFB tile，但各只使用一半。

![图 10 bN=192 时 SFB 的 TMA 加载模式](https://research.colfax-intl.com/wp-content/uploads/2026/03/image-2.png)

因此，host SFB 张量中三分之一的数据被加载进通常两倍的 CTA。该设计与 MMA 中的逻辑一致——`bN`=192 时，在每个 `N` 奇序 work tile，TMEM 中的 SFB 指针向前偏移两列：

```
if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
    # If this is an ODD tile, shift the TMEM start address for cta_tile_shape_n=192 case by two words (ignores first 64 columns of SFB)
    offset = cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
    shifted_ptr = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
        + offset,
        dtype=self.sf_dtype,
    )
    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
```

这是因为 `N` 奇序 work tile 的第一个 SFB tile 的前一半实际对应前一个 work tile 的输入。

对于 `bN`=64，`tBgsFB` 与 `bN`=128 相同（每个 CTA 加载所需数据的两倍）。但后面还有针对 `bN`=64 的额外条件块：

```
slice_n = mma_tile_coord_mnl[1]
if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
    slice_n = mma_tile_coord_mnl[1] // 2
    # ((atom_v, rest_v), RestK)
    tBgSFB_slice = tBgSFB[
        (None, slice_n, None, mma_tile_coord_mnl[2])
    ]
```

结果是 N 方向上每个偶序 work tile 及紧随其后的奇序 work tile 都加载同一块 SFB tile，偶序 work tile 使用前半，奇序 work tile 在 MMA 时使用后半。示例中还有另一块 `constexpr` 条件用于偏移 SFB 指针，逻辑与 `bN`=192 情况完全相同——在 N 方向每个奇序 work tile 将 SFB 指针向前偏移两列。

# 结论

本文研究了 Blackwell 硬件支持的 UMMA block-scaling，并梳理了 CuTeDSL 示例 [dense_blockscaled_gemm_persistent.py](https://github.com/NVIDIA/cutlass/blob/3476ddb7bd6ca4161a0169103ceaa20ce0eb891f/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py)。我们说明了消费时在 TMEM 中所需的缩放因子布局，它们被组织为 32×16 字节的 tile，所需 tile 数量取决于 UMMA atom 大小和缩放因子数据类型。我们还 traced 了这些缩放因子从全局内存到共享内存再到张量内存的加载路径。此外，讨论了 block-scaling 在 N 不能被 128 整除时以及 pair-UMMA 情形下的额外复杂性与应对方式。

CuTeDSL 示例 kernel 是一个很好的起点，但远未优化。Blackwell 上 block-scaled GEMM kernel 的优化是近期 [GPU mode 竞赛](https://luma.com/9n27uem4) 的主题——想了解更多，可以从 [获奖作品](https://www.gpumode.com/leaderboard/597?tab=rankings) 中学习。
