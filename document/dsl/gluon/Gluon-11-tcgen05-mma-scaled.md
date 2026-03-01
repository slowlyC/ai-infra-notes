## Gluon-11-tcgen05-mma-scaled

### 前言

本系列是 Gluon 的学习笔记，基于 [Gluon 官方教程](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon) 整理，覆盖了从基础概念到高级优化的完整内容，在原文基础上做了结构重组，并补充了 CUDA 编程模型和 GPU 硬件架构层面的说明。

> 知乎专栏：[DSL教程](https://www.zhihu.com/column/c_1990516179064858333)
> 完整代码：[GitHub](https://github.com/slowlyC/ai-infra-notes/tree/main/tutorials/gluon)

- [Gluon-01-Overview](https://zhuanlan.zhihu.com/p/1990504603008119998)
- [Gluon-02-Layout_Introduction](https://zhuanlan.zhihu.com/p/1990509278801457706)
- [Gluon-03-Async_Copy](https://zhuanlan.zhihu.com/p/1990517098083029502)
- [Gluon-04-TMA](https://zhuanlan.zhihu.com/p/1990517971483906093)
- [Gluon-05-wgmma](https://zhuanlan.zhihu.com/p/1990835358502523949)
- [Gluon-06-tcgen05](https://zhuanlan.zhihu.com/p/1990835546797405057)
- [Gluon-07-Persistent_Kernel](https://zhuanlan.zhihu.com/p/2003592603732563562)
- [Gluon-08-Warp_Specialization](https://zhuanlan.zhihu.com/p/2003602919912649692)
- [Gluon-09-TMA_Gather_Scatter](https://zhuanlan.zhihu.com/p/2003599190585017693)
- [Gluon-10-tcgen05_copy](https://zhuanlan.zhihu.com/p/2003603029975381459)
- **Gluon-11-tcgen05-mma-scaled**（本文）

### 概述

IEEE 754 浮点数结构, 包括符号位、指数位和尾数位。

```
┌──────┬──────────┬──────────┐
│ Sign │ Exponent │ Mantissa │
│  1位 │   E 位    │   M 位   │
└──────┴──────────┴──────────┘
```

Normal情况(E ≠ 0): 值= (-1)^sign × 2^(exp - bias) × (1 + mantissa/2^M)
                        ────────   ────────────────   ─────────────────
                        符号         指数部分            尾数部分
Subnormal情况(E=0): 值 = 2^(1-bias) × (0 + mantissa/2^M)
其中 bias = 2^(E-1) - 1 是让指数位能表示负数的技巧, 指数位是无符号的(如 2 位只能表示 0~3), 但我们需要表示 2^(-1), 2^0, 2^1 等。

fp4_e2m1 的布局为例:

```
┌───┬─────┬─────┐
│ S │  E  │  M  │
│ 1 │ 2位 │ 1位 │ = 4 位
└───┴─────┴─────┘
```

计算 fp4_e2m1: 0 10 1 的值:
bias = 2^(2-1) - 1 = 1
value = (+1) × 2^1 × (1 + 1/2^1) = 1 × 2 × (1 + 0.5) = 3.0

fp4_e2m1可表示的全部值如下:

| E | M | 数值 |
|---|---|------|
| 00 | 0 | 0 |
| 00 | 1 | 0.5 |
| 01 | 0 | 1.0 |
| 01 | 1 | 1.5 |
| 10 | 0 | 2.0 |
| 10 | 1 | 3.0 |
| 11 | 0 | 4.0 |
| 11 | 1 | 6.0 |

加上符号位, fp4_e2m1 可以表示 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} 共 15 个值。

精度(有效位数)的含义:
尾数位数决定精度, 尾数 M 位 → 可以表示 2^M 个不同的小数部分
fp4_e2m1: M=1 位 → 2^1=2 个不同的尾数 (0, 0.5)
          每个指数范围内只有 2 个值
bf16:     M=7 位 → 2^7=128 个不同的尾数
          每个指数范围内有 128 个值
fp16:     M=10 位 → 2^10=1024 个不同的尾数
          每个指数范围内有 1024 个值

精度的直观理解, 想象数轴上的刻度:

fp16 (10位尾数) 在 [0, 1) 区间:

```
|─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─...─┬─|  1024 个刻度
0.0                         1.0
   最小间隔 = 1/1024 ≈ 0.001
```

bf16 (7位尾数) 在 [0, 1) 区间:

```
|──┬──┬──┬──...──┬──|  128 个刻度
0.0                 1.0
   最小间隔 = 1/128 ≈ 0.008
```

fp4_e2m1 (1位尾数) 在 [0, 1) 区间:

```
|────────┬────────|  只有 2 个刻度！
0.0      0.5      1.0
   最小间隔 = 0.5
```

换算成有效十进制位数 ≈ M × log10(2) ≈ M × 0.301
推导过程:
M 位二进制尾数 → 能区分 2^M 个不同的小数部分
n 位十进制数 → 能区分 10^n 个不同的数
要让两者等价: 2^M = 10^n
取对数: n = M × log10(2) = M × 0.301

fp32 (M=23): 23 × 0.301 ≈ 7 位有效十进制数
fp16 (M=10): 10 × 0.301 ≈ 3 位有效十进制数
bf16 (M=7):   7 × 0.301 ≈ 2 位有效十进制数
fp4  (M=1):   1 × 0.301 ≈ 0.3 位(基本没精度！)

各种格式对比

| 格式 | 总位数 | Sign | Exponent | Mantissa | 表示范围 | 有效数字位数 |
|------|--------|------|----------|----------|----------|--------------|
| fp32 | 32 | 1 | 8 | 23 | ±3.4×10^38 | ~7 位 |
| fp16 | 16 | 1 | 5 | 10 | ±65504 | ~3 位 |
| bf16 | 16 | 1 | 8 | 7 | ±3.4×10^38 | ~2 位 |
| fp8_e4m3 | 8 | 1 | 4 | 3 | ±448 | ~1 位 |
| fp8_e5m2 | 8 | 1 | 5 | 2 | ±57344 | ~0.6 位 |
| fp8_e8m0 | 8 | 0 | 8 | 0 | 2^-127 ~ 2^127 | N/A (纯指数) |
| fp4_e2m1 | 4 | 1 | 2 | 1 | ±6 | ~0.3 位 |

理解要点: 
─────────
- 指数位数 (Exponent) → 决定【表示范围】(能表示多大/多小的数, 超过了会有误差)
  例如: bf16 和 fp32 都有 8 位指数, 所以范围相同(±3.4×10^38)

- 尾数位数 (Mantissa) → 决定【精度】(有效数字的个数)
  例如: bf16 只有 7 位尾数, 精度 ~2 位;fp32 有 23 位尾数, 精度 ~7 位

有效数字位数 = M × log10(2) ≈ M × 0.301
注意: 这是【总的有效数字个数】, 是指总共能精确表示的有效数字个数, 不是"小数点后几位"！

- fp32 (~7位): 1234567 或 0.001234567 都能精确表示(7位有效数字), 但 123456789.0 超过了, 后2位会有误差
- fp16 (~3位): 1.23 或 123000 都能精确表示(3位有效数字, 后面会丢失精度)
- fp4  (~0.3位): 连 1 位有效数字都表示不完整！

对有效数字的验证:

```python
import torch
# fp32 可以"表示" 123456789, 但不精确
x = torch.tensor(123456789, dtype=torch.float32)
print(f"存入: 123456789")
print(f"取出: {x.item()}")  # 输出: 123456784.0 ← 最后一位变了！

# fp32 也可以"表示"很大的数, 但同样不精确
y = torch.tensor(1.234567890123e30, dtype=torch.float32)
print(f"存入: 1.234567890123e30")
print(f"取出: {y.item()}")  # 输出: 1.2345678918272927e+30 只有前7位是准的, 后面都是近似
```

### fp4 量化过程示例

```
量化过程:
  原始数据 X (float32): [0.3, 0.9, 1.5, 2.4, 3.6, 4.5, 5.4, 6.0]
  → Step 1: 计算 scale = max_abs / fp4_max = 6.0 / 6.0 = 1.0
  → Step 2: 缩放并量化 X / scale → round_to_fp4 → [0.5, 1.0, 1.5, 2.0, 4.0, 4.0, 6.0, 6.0]
  → 存储: Q (fp4) + S (scale)

反量化过程:
  读取: Q (fp4) + S (scale)
  → X' = Q x S = [0.5, 1.0, 1.5, 2.0, 4.0, 4.0, 6.0, 6.0] x 1.0
  → 反量化值 X' (float32): [0.5, 1.0, 1.5, 2.0, 4.0, 4.0, 6.0, 6.0]
```

假设 VEC_SIZE = 32, K = 128

量化值 (base):  [q0, q1, ..., q31, q32, q33, ..., q63, ..., q96, q97, ..., q127]
               └──── 组 0 ─────┘   └──── 组 1 ─────┘         └──── 组 3 ─────┘
scale 因子:     [     s0       ,         s1        ,   s2    ,     s3        ]
repeat:        [s0, s0, ..., s0, s1, s1, ..., s1, s2, ..., s3, s3, ..., s3]
               └──   32 个   ──┘   └──   32 个   ──┘
反量化值(value):[q0*s0, q1*s0, ..., q31*s0, q32*s1, ..., q127*s3]

为什么乘上 scale 就能得到反量化值？量化的本质是将浮点数映射到更低精度:
量化过程: 原始值 X (float32) → 量化值 Q (fp4) + scale S
反量化过程: X ≈ Q × S

### Block Scaling 简介

Block scaling 是一种量化技术, 它将浮点张量 `X` 量化为: 一个形状相同但数据类型精度更低的张量 `Q`, 以及一个 scale 张量 `S`。
将张量 `X` 划分为大小相等的块来量化为 `Q`, 每个块对应一个 scale 因子。

在对 block-scaled 张量执行矩阵乘法时, 我们从 global memory 将量化后的操作数及其 scales 加载到 SM 上,
然后通过将每个量化值块乘以其对应的 scale 因子来进行反量化。这样 MMA 则是以更高精度进行计算的。

Blackwell GPU 上 `tcgen05_mma_scaled` 指令为 block-scaled MMA 提供了硬件加速支持, 该指令将操作数反量化和 MMA 融合为单条指令。
`tcgen05_mma_scaled` 支持以下特定的 block-scaled 量化方案:

- nvfp4: NVIDIA 专有的 fp4 量化方案, 使用 VEC_SIZE=16 和 float8_e4m3fn 类型的 scales.
  nvfp4 存在是因为 fp4 精度太低, NVIDIA 用不同的 VEC_SIZE 和 scale 格式来优化。而fp8 已经足够精确, 所以NVIDIA 直接使用 OCP 的 mxfp8 标准.
- mxfp4/mxfp6/mxfp8: Open Compute Project (OCP) 的 microscaling 格式(MX),
  用于 fp4/fp6/fp8, 使用 VEC_SIZE=32 和 fp8e8m0 类型的 scales.

Gluon 不支持 mxfp6, 因为 Gluon 没有暴露 fp6 数据类型。
MX scales 是 e8m0 格式, 即 0 位尾数和 8 位指数。换句话说, 它们是从 2**-127 到 2**127 的 2 的幂次, 其中 255 表示 NaN。

nvfp4、mxfp4 和 mxfp8 量化方案使用大小为 `VEC_SIZE` 的一维块, 沿 MMA 的归约维度(即 K 维度)对原始张量进行量化。

例如, 对于如下形式的 block-scale MMA:

```
C = (A * A_scale) @ (B * B_scale)
```

各张量的形状如下:

```
A.shape = (M, K)
B.shape = (N, K)
A_scale.shape = (M, K // VEC_SIZE)
B_scale.shape = (N, K // VEC_SIZE)
```

每个 scale 因子沿 K 维度广播并乘以 A 和 B 张量中 `VEC_SIZE` 个元素组成的向量。

Gluon `tcgen05_mma_scaled`目前仅支持转置的 B 操作数, 也就是 B tile 的形状为 `[BLOCK_N, BLOCK_K]`, 并使用转置的 shared memory 描述符。

在本教程中, 我们将演示如何使用 `tcgen05_mma_scaled` 执行硬件加速的 block-scaled MMA。
然后, 我们将介绍如何使用 `tcgen05_copy` 高效地将 scales 复制到 tensor memory。
之后, 我们还将讲解如何在 global memory 中选择高效的 scale 布局。
最后, 我们将展示如何编写流水线化和 warp-specialized 的 block-scaled MMA。

### 简单的 Block-Scaled 矩阵乘法

首先, 让我们编写一个简单的 block-scaled 矩阵乘法 kernel。
我们假设 scale 因子与其对应的块采用相同的布局。具体来说, A、B、A_scale 和 B_scale 张量将具有以下形状:

```
A.shape = (M, K)
B.shape = (N, K)
A_scale.shape = (M, K // VEC_SIZE)
B_scale.shape = (N, K // VEC_SIZE)
```

注意 Gluon 通过将 2 个 fp4 元素打包到一个 uint8 元素中来表示 fp4 数据类型, 一般沿归约维度(即 K 维度)打包 fp4 元素。
例如, 如果 A 和 B 是 沿 K 维度打包到 uint8 元素中的 fp4e2m1 张量, 它们的形状将是:

```
A.shape = (M, K // 2)
B.shape = (N, K // 2)
A_scale.shape = (M, K // VEC_SIZE)
B_scale.shape = (N, K // VEC_SIZE)
```

```python
@gluon.jit
def simple_mma_scaled_kernel(a_desc, b_desc, c_desc, a_scale_ptr, a_scale_stride_m, a_scale_stride_k, b_scale_ptr,
                             b_scale_stride_n, b_scale_stride_k, VEC_SIZE: gl.constexpr):
    # 因为内存最小单位是 byte, 而fp4 大小为 4 bits(0.5 byte), 无法单独存储, 必须将 2 个 fp4 打包到 1 个 uint8 中。
    # 1 个 uint8: [fp4_1 (高4位) | fp4_0 (低4位)]
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    # fp4 是亚字节数据类型(Sub-byte), 所以从 uint8 tensor 描述符加载操作数时需要考虑这一点。
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1

    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    # BLOCK_K 表示沿 K 维度的实际元素数量。 A.shape = (M, K // 2)  # 存储时 K//2 个 uint8
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    # 为操作数分配 shared memory。
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    # 为 scales 分配 tensor memory。scales 必须使用 `TensorMemoryScalesLayout` 布局。
    # 注意 B 的 scales 始终以 [BLOCK_N, BLOCK_K // VEC_SIZE] 的形状传递给 `tcgen05_mma_scaled`。
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale_ptr.dtype.element_ty, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)

    # 为累加器分配 tensor memory。
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    # 分配 barrier 用于跟踪操作数加载和 MMA。
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0

    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    for k in range(0, K, BLOCK_K):
        # BLOCK_K 是逻辑元素数量, 对于像 fp4 这样的亚字节数据类型, 需要将其转换为 uint8 偏移量。
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE

        # 加载 A 和 B 的 tiles。
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_smem)
        mbarrier.wait(bar, phase)

        # 加载 scales。
        coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])

        # 通过将沿 K 的偏移量除以 VEC_SIZE 来计算正确的偏移量。
        a_scale_offs_m = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, coalesced_2d_layout))
        a_scale_offs_k = k // VEC_SIZE + gl.arange(0, BLOCK_K // VEC_SIZE, layout=gl.SliceLayout(
            0, coalesced_2d_layout))
        a_scale = gl.load(a_scale_ptr + a_scale_offs_m[:, None] * a_scale_stride_m +
                          a_scale_offs_k[None, :] * a_scale_stride_k)

        b_scale_offs_n = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, coalesced_2d_layout))
        b_scale_offs_k = k // VEC_SIZE + gl.arange(0, BLOCK_K // VEC_SIZE, layout=gl.SliceLayout(
            0, coalesced_2d_layout))
        b_scale = gl.load(b_scale_ptr + b_scale_offs_n[:, None] * b_scale_stride_n +
                          b_scale_offs_k[None, :] * b_scale_stride_k)

        # 将 scales 写入 tmem。我们需要先将它们转换为正确的布局, 以便能用 `TensorMemoryScalesLayout` 布局写入 tmem。
        a_scale_layout: gl.constexpr = get_tmem_reg_layout(a_scale.dtype, a_scale.type.shape, scale_layout,
                                                           gl.num_warps())
        b_scale_layout: gl.constexpr = get_tmem_reg_layout(b_scale.dtype, b_scale.type.shape, scale_layout,
                                                           gl.num_warps())
        a_scale = gl.convert_layout(a_scale, a_scale_layout)
        b_scale = gl.convert_layout(b_scale, b_scale_layout)
        a_scale_tmem.store(a_scale)
        b_scale_tmem.store(b_scale)

        # 将操作数和 scale 张量连同正确的操作数格式字符串一起传递给 `tcgen05_mma_scaled`。
        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        # 使用 `use_acc` 进行原地累加, 在第一次迭代时设为 False 以零初始化累加器。B 操作数必须在 shared memory 中进行转置。
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                           use_acc=use_acc)
        # 提交 MMA 并等待其完成。
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    # 使用完 barriers 后务必将其失效, 以避免竞态条件和内存损坏错误。这一点尤为重要, 因为下面几行我们要为累加器分配 acc_smem。
    # 编译器发现 bar 和 mma_bar 在 for 循环后不再使用, 可能会复用这块内存给后面的 acc_smem！
    # 而 mbarrier 内部可能还有硬件状态未完成,  如果复用其内存, 会导致未定义行为！
    # invalidate 做两件事: 硬件层面, 清理 mbarrier 的内部状态；编译器层面, 标记这块 shared memory 可以安全复用。
    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)

    # 从 tensor memory 加载累加器 tile 并将其转换为输出数据类型。
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc = acc_tmem.load(acc_reg_layout)
    acc = acc.to(c_desc.dtype)

    # 通过 TMA store 写入累加器。
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)


def make_operand_descriptor(value: torch.Tensor, BLOCK_MN: int, BLOCK_K: int, MIXED_PREC: bool):
    # 如果操作数的数据类型是 fp4, 它们会被打包到 uint8 中。
    IS_FP4 = value.dtype == torch.uint8
    ELEM_PER_BYTE = 2 if IS_FP4 else 1

    # 当执行混合精度 `tcgen05_mma_scaled` 时(即一个操作数是 mxfp8, 另一个是 mxfp4), fp4 操作数需要在 smem 中进行填充。
    IS_MIXED_PREC_FP4 = MIXED_PREC and IS_FP4
    layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_MN, BLOCK_K // ELEM_PER_BYTE],
        gl.uint8 if IS_FP4 else gl.float8e4nv,
        fp4_padded=IS_MIXED_PREC_FP4,
    )
    return TensorDescriptor.from_tensor(value, [BLOCK_MN, BLOCK_K // ELEM_PER_BYTE], layout)


def make_output_descriptor(M: int, N: int, dtype: torch.dtype, BLOCK_M: int, BLOCK_N: int):
    C = torch.empty(M, N, device="cuda", dtype=dtype)
    C_dtype = getattr(gl, str(dtype).split('.')[1])
    C_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], C_dtype)
    return TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], C_desc_layout)


def simple_mma_scaled(A, B, A_scale, B_scale, VEC_SIZE, out_dtype=torch.float16, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128):
    M, N = A.shape[0], B.shape[0]

    is_nvfp4 = A_scale.dtype == torch.float8_e4m3fn
    assert not is_nvfp4 or B_scale.dtype == A_scale.dtype, "tcgen05_mma_scaled does not support mixing nvfp4 with other microscaled formats"
    # 我们的 MMA block 大小必须至少等于 scale 向量的大小。
    assert BLOCK_K >= VEC_SIZE, f"{BLOCK_K=} must be at least the size of the scale vector {VEC_SIZE=}"
    # TensorMemoryScalesLayout 在写入 tensor memory 时要求至少 32 行, 即 BLOCK_N 必须大于 32。
    # A 的 scales 将有 128 行, 因为使用 `tcgen05_mma_scaled` 时 BLOCK_M 必须是 128。
    assert BLOCK_N >= 32, f"{BLOCK_N=} must be at least 32"
    assert BLOCK_M == 128, f"{BLOCK_M=} must be 128"

    # 混合精度是指一个操作数是 mxfp4, 另一个是 mxfp8。
    MIXED_PREC = A.dtype != B.dtype

    # 混合精度时, fp4 operand 需要在 shared memory 中进行 padding(填充), 而 TMA 的 swizzle 机制要求连续维度至少 128 字节。
    # 实际上 TMA tensor 描述符的 block 形状在连续维度上必须至少是 64, 由于打包存储, 实际要求 BLOCK_K // 2 >= 64, 即 BLOCK_K >= 128
    # 即如果我们使用混合精度, BLOCK_K 必须至少是 128, 以使 fp4 TMA 描述符的内部维度至少为 64。
    assert not MIXED_PREC or BLOCK_K >= 128, f"{BLOCK_K=} must be at least 128 for mixed precision fp4 operands"

    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    # import pdb; pdb.set_trace()
    simple_mma_scaled_kernel[grid](A_desc, B_desc, C_desc, A_scale, *A_scale.stride(), B_scale, *B_scale.stride(),
                                   VEC_SIZE)
    return C_desc.base


# 我们可以使用 `triton.tools.mxfp` 中的通用工具 MXFP4Tensor 和 MXScaleTensor 来初始化量化 Tensor。
# MXFP4Tensor 封装了亚字节 fp4 元素的张量, MXScaleTensor 封装了 e8m0 MX scale 因子的 uint8 张量。
# MXFP4Tensor 是一个高级封装类, 它内部已经用打包的 uint8 存储数据, 但提供了逻辑视图:
# base.shape          # 逻辑形状: (MN, K) - 表示有 MN×K 个 fp4 元素
# base.data.shape     # 实际存储形状: 依赖于内部布局
# base.to(torch.float32)  # 解包为 float32, 形状 (MN, K)
def random_quantized_tensor(MN, K, format):
    assert format in ["mxfp4", "mxfp8", "nvfp4"]
    VEC_SIZE = 16 if format == "nvfp4" else 32

    # 生成一个随机的量化张量, base 内部存储为打包的 uint8, 但逻辑上代表 [MN, K] 个 fp4 值
    base = MXFP4Tensor(size=(MN, K), device="cuda").random()
    # 生成随机的 scale 因子, 沿 K 维度进行缩放, 每 VEC_SIZE 个元素共享一个 scale 因子。
    scale = MXScaleTensor(size=(MN, K // VEC_SIZE), device="cuda").random(low=1 / 128, high=2.0)

    # 计算反量化后的张量用于测试。
    ref = base.to(torch.float32)
    scale_ref = scale.to(torch.float32)
    value = ref * scale_ref.repeat_interleave(VEC_SIZE, dim=1)
    if format == "mxfp8":
        # 对于 mxfp8, 将张量转换为常规的 float8 torch 张量。
        return ref.to(torch.float8_e4m3fn), scale.data, value
    elif format == "mxfp4":
        # 对于 mxfp4, 沿 K 维度打包元素。目的是将 MXFP4Tensor 对象转换为特定维度打包的 torch.uint8 tensor, 以便传给 kernel。
        return base.to_packed_tensor(dim=1), scale.data, value
    else:
        # 对于 nvfp4, 额外将 scale 因子转换为 float8_e4m3fn。
        return base.to_packed_tensor(dim=1), scale_ref.to(torch.float8_e4m3fn), value
```

性能测试结果:

```
|    format     |   tflops/s   |
|---------------|--------------|
| mxfp8 x mxfp8 |    33.41     |
| mxfp4 x mxfp4 |    67.02     |
| mxfp8 x mxfp4 |    34.60     |
| nvfp4 x nvfp4 |    70.84     |
```

性能非常糟糕。然而, 目前还不清楚有多少性能问题是由 scales 引起的。
使用 `ncu --set full --kernel-name simple_mma_scaled_kernel` 对 mxfp8 x mxfp8 情况进行微基准测试, 会在输出中看到:

```
Section: Memory Workload Analysis Tables
OPT   Est. Speedup: 15.72%
      The memory access pattern for global loads from L1TEX might not be optimal. On average, only 4.0 of the 32
      bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between
      threads. Check the Source Counters section for uncoalesced global loads.
----- --------------------------------------------------------------------------------------------------------------
OPT   Est. Speedup: 17.41%
      The memory access pattern for local loads from L1TEX might not be optimal. On average, only 1.0 of the 32
      bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between
      threads. Check the Source Counters section for uncoalesced local loads.
----- --------------------------------------------------------------------------------------------------------------
OPT   Est. Speedup: 17.41%
      The memory access pattern for local stores to L1TEX might not be optimal. On average, only 1.0 of the 32
      bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between
      threads. Check the Source Counters section for uncoalesced local stores.
```

```
部分: 内存工作负载分析表
OPT   预估加速: 15.72%
      从 L1TEX 进行全局加载的内存访问模式可能不是最优的。平均而言, 每个线程仅利用了每个扇区传输的 32 字节中的 4.0 字节。
      这可能是由于线程之间的步长导致的。请检查源计数器部分中的未合并全局加载。
----- --------------------------------------------------------------------------------------------------------------
OPT   预估加速: 17.41%
      从 L1TEX 进行本地加载的内存访问模式可能不是最优的。平均而言, 每个线程仅利用了每个扇区传输的 32 字节中的 1.0 字节。
      这可能是由于线程之间的步长导致的。请检查源计数器部分中的未合并本地加载。
----- --------------------------------------------------------------------------------------------------------------
OPT   预估加速: 17.41%
      向 L1TEX 进行本地存储的内存访问模式可能不是最优的。平均而言, 每个线程仅利用了每个扇区传输的 32 字节中的 1.0 字节。
      这可能是由于线程之间的步长导致的。请检查源计数器部分中的未合并本地存储。
```

下面对 simple_mma_scaled_kernel 中 a_scale 的 load进行分析:
假设: M = 256, K = 512, VEC_SIZE = 32, BLOCK_M = 128, BLOCK_K = 128,
      SCALE_K = K // VEC_SIZE = 16, SCALE_BLOCK_K = BLOCK_K // VEC_SIZE = 4
a_scale 原始形状: [M, SCALE_K] = [256, 16], a_scale_tile 的形状为 [BLOCK_M, SCALE_BLOCK_K] = [128, 4]

矩阵乘法的一次 tile 计算:
         K 轴方向 (总共 16 个 scale 列)
         列0 列1 列2 列3  列4 列5 列6 列7  ... 列15
         ← K 迭代 0 →     ← K 迭代 1 →     ...
         (BLOCK_K=128)   (BLOCK_K=128)
M 轴:
  128 行   [128x4 的 scale tile, 需要连续访问]
  另 128 行 [...]

当 kernel (pid_m=0, k=0) 要加载 [128, 4] 的 scales tile 时:
        列 0   列 1   列 2   列 3
行 0:  [0]    [1]    [2]    [3]     ← 内存地址: 0, 1, 2, 3
行 1:  [16]   [17]   [18]   [19]    ← 内存地址: 16, 17, 18, 19
行 2:  [32]   [33]   [34]   [35]    ← 内存地址: 32, 33, 34, 35
...
行 127: [...]                       ← 内存地址: 127*16, ...
问题: 相邻行之间的地址跳跃了 16 个元素！

线程访问模式分析:
coalesced_2d_layout: [1, 1], [1, 32], [1, num_warps], order=[1, 0]
假设 num_warps = 4, 总共 128 个线程, 当加载 a_scale 时:
  Thread 0: base + 0*stride_m + 0
  Thread 1: base + 0*stride_m + 1
  Thread 2: base + 0*stride_m + 2
  Thread 3: base + 0*stride_m + 3
  Thread 4: base + 1*stride_m + 0  ← 跳跃
  ...
stride_m = SCALE_K = 16
为什么跳跃？因为 SCALE_BLOCK_K = 4, 每行只有 4 个元素, 当 Thread 3 处理完 row0 的最后一列后, Thread 4 必须跳到 row1。

即一个 warp 中的32个线程同时访问的数据为:
   地址: 0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, ...
                     ↑
               地址不连续！需要多次内存事务

内存事务分析:
理想情况: GPU 内存系统一次传输 32 字节 (一个 sector), 32 个线程访问连续的 32 个地址 → 1 次内存事务
实际情况 (simple kernel 布局): 每个线程需要 1 byte (fp8 scale)
理论上 Sector 0 (地址 0-31):
Sector 0 (地址 0-31):
  [0 1 2 3] [4..15 未使用] [16 17 18 19] [20-31 未使用]
   线程0-3                   线程4-7
   4 bytes                   4 bytes
看起来线程 0-3 和 线程 4-7 可以用一个 sector, 但 GPU 的内存系统在合并访问时, 是按照"线程ID → 地址"的映射来判断的,
即使地址在同一个 sector 内, 如果 线程-地址 映射不是连续的, GPU 仍然需要分多次处理。
因此内存事务如下:
  线程 0-3:   访问地址 0-3     → 1 个 sector (但只用了 4/32 字节)
  线程 4-7:   访问地址 16-19   → 另一个 sector (只用了 4/32 字节)
  线程 8-11:  访问地址 32-35   → 又一个 sector (只用了 4/32 字节)
  ...
  每个扇区只利用 4/32 = 12.5% 的带宽！
  这正是 profiler 报告的 "每个线程仅利用了 4.0 字节" 的原因！

这印证了我们的猜测: 从 global memory 加载 scales 的效率很低。
我们可以改变 scales 的布局来解决这个问题, 办法是将 [MN, K // VEC_SIZE] 的布局改为 [M // BLOCK_M, SCALE_K // SCALE_BLOCK_K, BLOCK_M, SCALE_BLOCK_K]
其中 order=[?, ?, 1, 0], 即先沿最后一个维度 dim=3 连续, 然后沿倒数第二个维度 dim=2 连续。
前两个维度分别对应并行的 M 和 K 维度上的 grid 索引, 后两个维度是单个 program 需要 laod 的 scales_tile。

新布局的内存访问模式分析:
新形状: [M//BLOCK_M, SCALE_K//SCALE_BLOCK_K, BLOCK_M, SCALE_BLOCK_K] = [2, 4, 128, 4]
coalesced_1d_layout: [1], [32], [num_warps], order=[0]
128 个线程加载 512 个元素 (每个线程加载 4 个), 线程访问模式:
  Thread 0:  地址 0, 1, 2, 3        (连续)
  Thread 1:  地址 4, 5, 6, 7        (连续)
  Thread 2:  地址 8, 9, 10, 11      (连续)
  ...
  Thread 31: 地址 124, 125, 126, 127 (连续)
  Thread 32: 地址 128, 129, 130, 131 (连续)
  ...
一个 warp 的 32 个线程同时访问地址 0-127 (32线程 x 4元素 = 128 个连续地址)
  → 完美合并! 4 次内存事务即可完成 (每次 32 元素)
  → 100% 带宽利用率!

代码实现上是将原本 scale 的 [MN, K // VEC_SIZE] reshape 为 [M // BLOCK_M, BLOCK_M, SCALE_K // SCALES_BLOCK_K, SCALES_BLOCK_K],
然后用 permute 进行维度置换, 将 [BLOCK_M, SCALES_BLOCK_K] 置换到末尾, 即置换 dim1 和 dim2。

### Scale 布局优化 - Contiguous 布局

```python
def relayout_scales_contiguous(scales: torch.Tensor, BLOCK_MN: int, BLOCK_K: int, VEC_SIZE: int):
    # 假设 scales.siez: [256, 16], SCALES_BLOCK_K = BLOCK_K // VEC_SIZE = 4
    MN, SCALE_K = scales.shape[0], scales.shape[1]
    SCALES_BLOCK_K = BLOCK_K // VEC_SIZE

    # Step 1: reshape
    # [256, 16] → [2, 128, 4, 4]
    #              ↑   ↑   ↑  ↑
    #           M块数 BLOCK_M K块数 SCALE_BLOCK_K
    scales = scales.reshape(MN // BLOCK_MN, BLOCK_MN, SCALE_K // SCALES_BLOCK_K, SCALES_BLOCK_K)
    # Step 2: permute (0, 2, 1, 3), 交换 dim1 和 dim2
    # [2, 128, 4, 4] → [2, 4, 128, 4]
    #                   ↑  ↑   ↑   ↑
    #               M块 K块 BLOCK_M SCALE_BLOCK_K
    scales = scales.permute(0, 2, 1, 3)
    # Step 3: contiguous - 使新布局在内存中连续
    return scales.contiguous()


# 让我们重新实现 contig_kernel 以适应新的 scale 布局。其中只有加载 scales 的方式与 `simple_mma_scaled_kernel` 不同。
@gluon.jit
def mma_scaled_contig_kernel(a_desc, b_desc, c_desc, a_scale_ptr, b_scale_ptr, VEC_SIZE: gl.constexpr):
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale_ptr.dtype.element_ty, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE

        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_smem)
        mbarrier.wait(bar, phase)

        # ======= 以下是与 `simple_mma_scaled_kernel` 不同的部分 =======
        # 加载 scales
        SCALE_K = K // VEC_SIZE  # = 256 // 16 = 16
        SCALE_BLOCK_K: gl.constexpr = BLOCK_K // VEC_SIZE  # = 128 // 32 = 4
        # `a_scale` 形状为 [M // BLOCK_M, SCALE_K // SCALE_BLOCK_K, BLOCK_M, SCALE_BLOCK_K] = [2, 4, 128, 4],
        # 每个内层循环的 tile 将加载 `a_scale[pid_m, k // BLOCK_K, :, :]` 的数据, 即 tile 大小为 [BLOCK_M, SCALE_BLOCK_K]。
        a_stride_k: gl.constexpr = BLOCK_M * SCALE_BLOCK_K  # = 128 * 4 = 512 (一个 scale_tile 的大小, dim1的 stride)
        a_stride_m = SCALE_K // SCALE_BLOCK_K * a_stride_k  # = 16 // 4 * 512 = 4 * 512 = 2048 (一个 m_block 的大小, dim0的 stride)
        b_stride_k: gl.constexpr = BLOCK_N * SCALE_BLOCK_K
        b_stride_n = SCALE_K // SCALE_BLOCK_K * b_stride_k

        # 加载 `a_scale[pid_m, k // BLOCK_K, :, :]`。由于我们知道后两个维度是连续的, 为简单起见可以使用一维加载。
        coalesced_1d: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])

        a_scale_base = a_scale_ptr + pid_m * a_stride_m + k // BLOCK_K * a_stride_k
        b_scale_base = b_scale_ptr + pid_n * b_stride_n + k // BLOCK_K * b_stride_k
        a_scale = gl.load(a_scale_base + gl.arange(0, BLOCK_M * SCALE_BLOCK_K, coalesced_1d))  # 加载 128 * 4 = 512 个连续元素
        b_scale = gl.load(b_scale_base + gl.arange(0, BLOCK_N * SCALE_BLOCK_K, coalesced_1d))
        a_scale = a_scale.reshape(BLOCK_M, SCALE_BLOCK_K)
        b_scale = b_scale.reshape(BLOCK_N, SCALE_BLOCK_K)
        # ======= 以上是与 `simple_mma_scaled_kernel` 不同的部分, 后面完全相同 =======

        a_scale_layout: gl.constexpr = get_tmem_reg_layout(a_scale.dtype, a_scale.type.shape, scale_layout,
                                                           gl.num_warps())
        b_scale_layout: gl.constexpr = get_tmem_reg_layout(b_scale.dtype, b_scale.type.shape, scale_layout,
                                                           gl.num_warps())
        a_scale = gl.convert_layout(a_scale, a_scale_layout)
        b_scale = gl.convert_layout(b_scale, b_scale_layout)
        a_scale_tmem.store(a_scale)
        b_scale_tmem.store(b_scale)

        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                           use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc = acc_tmem.load(acc_reg_layout)
    acc = acc.to(c_desc.dtype)
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)


def mma_scaled_contig(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, out_dtype=torch.float16):
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype
    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mma_scaled_contig_kernel[grid](A_desc, B_desc, C_desc, A_scale, B_scale, VEC_SIZE)
    return C_desc.base
```

性能结果如下:

```
|    format     |   tflops/s   |
|---------------|--------------|
| mxfp8 x mxfp8 |   663.28     |
| mxfp4 x mxfp4 |  1435.05     |
| mxfp8 x mxfp4 |   741.82     |
| nvfp4 x nvfp4 |  1303.69     |
```

这是一个巨大的提速！(relayout_scales_contiguous的时间没有被计算在内)
`simple_mma_scaled` 性能这么差的原因是: 低效的 scale 加载在频繁地冲刷 L2 cache。
通过改变 scales 在 global memory 中的布局, 使 kernel 的内层循环能够更高效地加载, 从而将 kernel 的性能提升了 20 倍。

### Scale 布局优化 - Packed Block 布局

接下来我们可以考虑使用 TMA 来加载 scales。我们将为 scales 选择一个 "packed block" 布局的 5D global memory 布局。
对于 A 矩阵[256, 512], 其布局是

```
[M // (32 * 4), K // (VEC_SIZE * 4), 32, 4, 4]
```

即对于 M = 256, VEC_SIZE = 32，这个 5D 形状是: [256 // 128, 16 // 4, 32, 4, 4] = [2, 4, 32, 4, 4]
5D Packed Block 布局:
  维度 0: M // 128 = 2       M 方向的大块 (大块 0: 行 0-127, 大块 1: 行 128-255)
  维度 1: SCALE_K // 4 = 4   K 方向的大块 (K块0: 列0-3, K块1: 列4-7, ..., K块3: 列12-15)
  维度 2: 32                  每个大块内的行组数量 (128 = 32 x 4)
  维度 3: 4                   每个行组内的行数
  维度 4: 4                   每个 K 大块内的列数 (scale 数量)
  即 128 行 = 32 个行组 x 每组 4 行, 4 列 = 4 个连续的 scale

这样, 在矩阵乘法沿 K blocks 的内层循环中, 每次处理一个 [BLOCK_M, BLOCK_K] 的子 tile, 其中所有 scales 在内存中都是是连续存储的。

稍后在 GPU 上, 我们将在逻辑上 permute 和 reshape scales, 将其恢复为 `tcgen05_mma_scaled` 期望的 2D 布局。

```python
def align_to(a, b):
    # 返回大于或等于 `a` 的最小 `b` 的倍数。
    return triton.cdiv(a, b) * b

def swizzle_scales_packed_block(scales: torch.Tensor, VEC_SIZE: int):
    # 当 scale 张量不是 [128, 4] 的整数倍时, 我们需要填充 scale 张量, 以便它可以使用 packed block 格式。
    PAD_MN = align_to(scales.shape[0], 128) - scales.shape[0]
    PAD_K = align_to(scales.shape[1], 4) - scales.shape[1]
    scales = torch.nn.functional.pad(scales, (0, PAD_K, 0, PAD_MN))

    # reshape: [MN, SCALE_K] -> [REP_MN=2, 4, 32, REP_K=4, 4]
    #            │       │         │       │   │    │      └─ K 方向每组 4 列的大块数量
    #          2*4*32   4*4        │       │   │    └─ K 方向 4 个组
    #                              │       │   └─ M 方向 32 个行组
    #                              │       └─ M 方向每组 4 行
    #                              └─ M 方向 2 个大块
    # [256, 16] → [2, 4, 32, 4, 4]
    MN, SCALE_K = scales.shape[0], scales.shape[1]
    REP_MN = MN // 128  # 2
    REP_K = SCALE_K // 4  # 4
    scales = scales.reshape(REP_MN, 4, 32, REP_K, 4)  # [2, 4, 32, 4, 4]
    # permute: [2, 4, 32, 4, 4] -> [2, 4, 32, 4, 4]
    # [REP_MN, 4, 32, REP_K, 4] -> [REP_MN, REP_K, 32, 4, 4]
    #    ↑     ↑   ↑    ↑    ↑        ↑       ↑    ↑   ↑  ↑
    #    0     1   2    3    4        0       3    2   1  4
    # 交换 dim1 和 dim3, 和 contig 版本类似，都是把 REP_K 置换到前面，只是这里把 128 拆成了 32 * 4。
    scales = scales.permute(0, 3, 2, 1, 4)
    return scales.contiguous()


def make_scales_descriptor(scales: torch.Tensor, BLOCK_MN: int, BLOCK_K: int, VEC_SIZE: int):
    # 注意 5D swizzling 方案有最小 block 要求: BLOCK_N >= 128 且 BLOCK_K >= VEC_SIZE * 4(nvfp4 为 64, MX 为 128)。
    REP_MN = BLOCK_MN // 128  # = 128 // 128 = 1
    REP_K = BLOCK_K // (VEC_SIZE * 4)  # = 128 // (32 * 4) = 1
    # 使用 5D TMA 描述符, block 形状为 [1, REP_MN, REP_K, 2, 256] 的 uint8 元素。
    # 内部维度设为 256 字节, 可以更好地利用 L2 cache, 如果使用 32x16xu8 就会出现 TMA 引擎发出许多小消息(16B)的情况。
    block_shape = [1, REP_MN, REP_K, 2, 256]  # [1, 1, 1, 2, 256]
    scales = scales.reshape(1, scales.shape[0], scales.shape[1], 2, 256)  # [1, 2, 4, 2, 256]
    IS_NVFP4 = scales.dtype == torch.float8_e4m3fn
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float8e4nv if IS_NVFP4 else gl.uint8)
    return TensorDescriptor.from_tensor(scales, block_shape, layout)


@gluon.jit
def unswizzle_scales_packed_block(scales, BLOCK_MN: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    # 从 packed block 布局中 unswizzle scales 子 tile。
    #
    # Step 1: reshape - 将内部的 512 字节展开为逻辑结构, 即把 512 字节重新解释为 32 个行组, 每个行组有 4 行, 每行有 4 列
    # [1, 1, 1, 2, 256] -> [1, 1, 32, 4, 4]
    scales = scales.reshape(scales.shape[1], scales.shape[2], 32, 4, 4)
    # Step 2: permute - 反转 CPU 端的 permute
    # CPU permute 做的是: [REP_MN, 4, 32, REP_K, 4] → [REP_MN, REP_K, 32, 4, 4]
    # GPU permute 要反转: [REP_MN, REP_K, 32, 4, 4] → [REP_MN, 4, 32, REP_K, 4]
    # 即: [1, 1, 32, 4, 4] → [1, 4, 32, 1, 4]
    # 注意: GPU 上 permute 是"索引变换"而非"数据移动"，只是改变了编译器/硬件理解数据布局的方式，内存中的 512 字节始终是连续的。
    scales = scales.permute(0, 3, 2, 1, 4)
    # Step 3: reshape - 合并为 2D
    # [1, 4, 32, 1, 4] -> [128, 4]
    return scales.reshape(BLOCK_MN, BLOCK_K // VEC_SIZE)


@gluon.jit
def mma_scaled_packed_block_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, VEC_SIZE: gl.constexpr):
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale_desc.dtype, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale_desc.dtype, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # ======= 以下是与 `simple_mma_scaled_kernel` 不同的部分=======

    # 分配 shared memory 用于 TMA 加载 scales。
    # a_scale_desc.block_type.shape: [1,1,1,2,256]
    a_scale_smem = gl.allocate_shared_memory(a_scale_desc.dtype, a_scale_desc.block_type.shape, a_scale_desc.layout)
    b_scale_smem = gl.allocate_shared_memory(b_scale_desc.dtype, b_scale_desc.block_type.shape, b_scale_desc.layout)
    REP_M: gl.constexpr = a_scale_desc.block_type.shape[1]  # = 1
    REP_N: gl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_type.shape[2]  # = 1
    B_REP_K: gl.constexpr = b_scale_desc.block_type.shape[2]
    # 沿 REP_M 索引 M 和 N 的子 tiles。
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE
        # 为每个 scale 沿 REP_K 索引 K 子 tile。
        off_k_a_scale = (k // BLOCK_K) * A_REP_K
        off_k_b_scale = (k // BLOCK_K) * B_REP_K

        # gmem -> smem
        mbarrier.expect(
            bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + a_scale_desc.block_type.nbytes +
            b_scale_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_smem)
        tma.async_copy_global_to_shared(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar, a_scale_smem)
        tma.async_copy_global_to_shared(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar, b_scale_smem)
        mbarrier.wait(bar, phase)

        # 我们知道 swizzle_scales 存储到 tensor memory 所需的 2D 布局。
        a_scale_layout: gl.constexpr = get_tmem_reg_layout(a_scale_desc.dtype, [BLOCK_M, BLOCK_K // VEC_SIZE],
                                                           scale_layout, gl.num_warps())
        b_scale_layout: gl.constexpr = get_tmem_reg_layout(b_scale_desc.dtype, [BLOCK_N, BLOCK_K // VEC_SIZE],
                                                           scale_layout, gl.num_warps())

        # 使用 AutoLayout 加载 scales，此时编译器还不知道应该用什么布局加载
        # gl.AutoLayout() 不能单独使用，必须配合 set_auto_layout。原因是 AutoLayout 是一个"占位符"布局，编译器需要一个具体的目标布局来反向推导。
        # AutoLayout 就像代数中的未知数 x, set_auto_layout 就像方程的约束条件, 两者结合才能解出 x。
        a_scale = a_scale_smem.load(gl.AutoLayout())  # 布局 = AutoLayout
        b_scale = b_scale_smem.load(gl.AutoLayout())
        # SMEM 中的数据是 5D packed block 格式，但 tcgen05_mma_scaled 期望 2D 格式，所以需要 unswizzle。
        a_scale = unswizzle_scales_packed_block(a_scale, BLOCK_M, BLOCK_K, VEC_SIZE)  # 布局 = AutoLayout
        b_scale = unswizzle_scales_packed_block(b_scale, BLOCK_N, BLOCK_K, VEC_SIZE)

        # 对 scales 进行 unswizzling 操作后, 不能直接 a_scale_layout，使用可以反向推算出 unswizzling 后, 加载 unswizzle_scales 时应使用的布局。
        # 也可以使用AutoLayout自动推断, 即调用 `set_auto_layout`，让编译器反向传播自动解析 unswizzling 后的 scale 布局。
        # a_scale_smem.load(AutoLayout)    布局=??? (编译器待定)
        #       ↓
        # unswizzle (reshape + permute)    布局=??? (编译器待定)
        #       ↓
        # set_auto_layout(a_scale, a_scale_layout)
        #       ↓                          布局=a_scale_layout (已知)
        #       ↓                          ← 编译器从这里反向传播, 解析所有 AutoLayout
        # a_scale_tmem.store(a_scale)
        a_scale = gl.set_auto_layout(a_scale, a_scale_layout)  # 锚点！
        b_scale = gl.set_auto_layout(b_scale, b_scale_layout)

        # ======= 以上是与 `simple_mma_scaled_kernel` 不同的部分, 后面完全相同 =======
        a_scale_tmem.store(a_scale)
        b_scale_tmem.store(b_scale)

        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                           use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc = acc_tmem.load(acc_reg_layout)
    acc = acc.to(c_desc.dtype)
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)


def mma_scaled_packed_block(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, out_dtype=torch.float16):
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype
    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    A_scale_desc = make_scales_descriptor(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
    B_scale_desc = make_scales_descriptor(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mma_scaled_packed_block_kernel[grid](A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, VEC_SIZE)
    return C_desc.base
```

性能结果:

```
|    format     |   tflops/s   |
|---------------|--------------|
| mxfp8 x mxfp8 |   900.97     |
| mxfp4 x mxfp4 |  2081.76     |
| mxfp8 x mxfp4 |  1000.48     |
| nvfp4 x nvfp4 |  2002.05     |
```

通过使用 TMA, 我们实现了约 35% 的加速。
TMA 能更高效地加载大块连续内存, 而且由于 TMA 将 scales 直接加载到 shared memory, 我们避免了大部分 `convert_layout` 的开销。

### 使用 tcgen05_copy 优化

然而, 我们仍然需要通过寄存器中转 scales, 才能将它们从 shared memory 传输到 tensor memory。
接下来, 我们可以应用在上一个教程中学到的 `tcgen05_copy`, 异步地将 scales 从 shared memory 复制到 tensor memory。

我们需要使用一种新的布局来查看 shared memory, 这种布局会撤销 swizzling。
我们通过 reshape 和 permute shared memory 描述符来实现这一点, 操作顺序与我们生成原始 swizzle 模式的方式相反。

```python
@gluon.jit
def unswizzle_scales_shared_memory(smem, BLOCK_MN: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    smem = smem.reshape((smem.shape[1], smem.shape[2], 32, 4, 4))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((BLOCK_MN, BLOCK_K // VEC_SIZE))

# 但最终的 shared memory 描述符的布局会是什么, 它与 `tcgen05_copy` 兼容吗？
# 为了检查布局, 我们可以编写一个小的 stub kernel 并使用 `gl.static_print` 来打印 constexprs。
@gluon.jit
def scales_layout_test(scales_desc, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    smem = gl.allocate_shared_memory(scales_desc.dtype, scales_desc.block_type.shape, scales_desc.layout)
    gl.static_print(smem.type.layout)
    # 我们不打算执行这个 kernel, 所以可以使用未初始化的 `smem` 来获取前向类型传播以检查布局。
    smem = unswizzle_scales_shared_memory(smem, BLOCK_M, BLOCK_K, VEC_SIZE)
    gl.static_print(smem.type.layout)
```

打印出的布局是

```python
NVMMASharedLayout(
    swizzle_byte_width=0,
    element_bitwidth=8,
    rank=5,
    transposed=False,
    fp4_padded=False,
    cga_layout=[]
)

SharedLinearLayout(
   offset_bases=[[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]],
   block_bases=[],
   alignment=128
)
```

要判断它是否与 `tcgen05_copy` 兼容, 你需要参考 PTX 文档。
Linear layouts 的推理也可能比较棘手,不过我们可以直接尝试用这个布局调用 `tcgen05_copy`, 看看编译器是否会报错。

```python
@gluon.jit
def tcgen05_copy_layout_test(smem_layout: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr,
                             VEC_SIZE: gl.constexpr):
    smem = gl.allocate_shared_memory(gl.uint8, (BLOCK_M, BLOCK_K // VEC_SIZE), smem_layout)
    tmem = allocate_tensor_memory(gl.uint8, (BLOCK_M, BLOCK_K // VEC_SIZE), TensorMemoryScalesLayout())
    tcgen05_copy(smem, tmem)
```

这段代码运行时没有错误, 这意味着该布局与 `tcgen05_copy` 兼容。
如果不兼容, 编译器会抛出类似这样的错误:

```
failed to find valid tcgen05.copy layout from shared memory descriptor
```

例如, `gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2)` 就是不兼容的, 会触发上述错误。
此外, 如果我们将原始 shared memory 布局的 `swizzle_byte_width` 改为非零值, unswizzled 后的布局也会触发同样的错误。
也就是说, 对于 NVMMASharedLayout, 我们必须关闭 swizzling 才能使用 `tcgen05_copy`。

这个 scale 的 packed block 布局是专门设计的, 以便与 TMA 兼容, 并且在 smem 中 unswizzle 后能产生与 `tcgen05_copy` 兼容的布局。

有关 scale 因子布局的更多详细信息, 请参阅:
 1. https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
 2. https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout

有了这些信息, 我们可以重写 kernel 以使用 `tcgen05_copy`。

```python
@gluon.jit
def mma_scaled_tcgen05_copy_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, VEC_SIZE: gl.constexpr):
    # ======= 从 `mma_scaled_packed_block_kernel` 复制的不变代码开始 =======
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale_desc.dtype, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale_desc.dtype, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    use_acc = False

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_scale_smem = gl.allocate_shared_memory(a_scale_desc.dtype, a_scale_desc.block_type.shape, a_scale_desc.layout)
    b_scale_smem = gl.allocate_shared_memory(b_scale_desc.dtype, b_scale_desc.block_type.shape, b_scale_desc.layout)
    REP_M: gl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_type.shape[2]
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE
        off_k_a_scale = (k // BLOCK_K) * A_REP_K
        off_k_b_scale = (k // BLOCK_K) * B_REP_K

        mbarrier.expect(
            bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + a_scale_desc.block_type.nbytes +
            b_scale_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_smem)
        tma.async_copy_global_to_shared(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar, a_scale_smem)
        tma.async_copy_global_to_shared(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar, b_scale_smem)
        mbarrier.wait(bar, phase)

        # ======= 以下是与 `mma_scaled_packed_block_kernel` 不同的部分 =======

        # 在 shared memory 中 unswizzle scales。
        a_scale = unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
        b_scale = unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)
        # 发起异步复制到 tensor memory。`tcgen05_copy` 与 `tcgen05_mma_scaled` 是隐式流水线化的, 所以我们不需要显式地同步它们。
        tcgen05_copy(a_scale, a_scale_tmem)
        tcgen05_copy(b_scale, b_scale_tmem)

        # ======= 以上是与 `mma_scaled_packed_block_kernel` 不同的部分, 后面完全相同 =======

        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                           use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc = acc_tmem.load(acc_reg_layout)
    acc = acc.to(c_desc.dtype)
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)


def mma_scaled_tcgen05_copy(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, out_dtype=torch.float16):
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype
    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    A_scale_desc = make_scales_descriptor(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
    B_scale_desc = make_scales_descriptor(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE)

    # 替换 TMA 描述符布局为不带 swizzling 的布局,
    # 以便 unswizzled 布局与 `tcgen05_copy` 兼容。
    no_swizzle_layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
    A_scale_desc = replace(A_scale_desc, layout=no_swizzle_layout)
    B_scale_desc = replace(B_scale_desc, layout=no_swizzle_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mma_scaled_tcgen05_copy_kernel[grid](A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, VEC_SIZE)
    return C_desc.base
```

性能结果:

```
|    format     |   tflops/s   |
|---------------|--------------|
| mxfp8 x mxfp8 |   929.07     |
| mxfp4 x mxfp4 |  2147.76     |
| mxfp8 x mxfp4 |  1035.60     |
| nvfp4 x nvfp4 |  2092.39     |
```

使用 `tcgen05_copy`, 我们观察到 kernel 有适度的加速。
为了获得剩余的性能提升, 我们将演示一个软件流水线化和 warp-specialized 版本的 block-scaled 矩阵乘法。

将 scales 通过 `tcgen05_copy` 复制到 tmem, 然后执行 `tcgen05_mma_scaled`, 可以抽象为带有 4 个 shared memory 输入的异步 MMA 指令。
我们可以像普通的异步 MMA 一样对其进行流水线化。

### 流水线版本

```python
@gluon.jit
def async_mma_scaled_impl(a_smem, b_smem, a_scale_smem, b_scale_smem, acc_tmem, use_acc, pred):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_smem.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_smem.shape[0]
    BLOCK_N: gl.constexpr = b_smem.shape[0]
    BLOCK_K: gl.constexpr = a_smem.shape[1] * A_ELEM_PER_BYTE
    # 回顾一下, 我们用 `uint8` 来表示 fp4 元素。
    VEC_SIZE: gl.constexpr = 32 if a_scale_smem.dtype == gl.uint8 else 16

    a_scale = unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
    b_scale = unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)

    # 我们不需要将 scales 的 tensor memory 分配提升到循环外部, 所以可以将它们放入这个辅助函数中。
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype, a_scale.type.shape, scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype, b_scale.type.shape, scale_layout)
    tcgen05_copy(a_scale, a_scale_tmem)
    tcgen05_copy(b_scale, b_scale_tmem)

    a_format: gl.constexpr = "e2m1" if a_smem.dtype == gl.uint8 else "e4m3"
    b_format: gl.constexpr = "e2m1" if b_smem.dtype == gl.uint8 else "e4m3"
    tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                       use_acc=use_acc, pred=pred)


# 这个辅助函数根据当前的 `pid_m`、`pid_n` 和 `k` 索引计算所有加载索引并发起异步加载。
# 编译器会执行循环不变量代码外提, 将不依赖于 `k` 的代码(如 `pid_m * BLOCK_M`)
# 提升到内层循环之外, 所以我们可以安全地抽象加载索引而不会损失性能。
#
# 封装加载索引逻辑有助于保持流水线 kernel 代码的整洁, 因为流水线化可能会变得很乱。
@gluon.jit
def issue_loads(producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs,
                b_scale_bufs, bars, pred):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if b_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    REP_M: gl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_type.shape[2]

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N
    off_k_a = k // A_ELEM_PER_BYTE
    off_k_b = k // B_ELEM_PER_BYTE
    off_k_a_scale = (k // BLOCK_K) * A_REP_K
    off_k_b_scale = (k // BLOCK_K) * B_REP_K

    index = producer.index
    bar = bars.index(index)
    mbarrier.expect(
        bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + a_scale_desc.block_type.nbytes +
        b_scale_desc.block_type.nbytes, pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], bar, b_bufs.index(index), pred)
    tma.async_copy_global_to_shared(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar,
                                    a_scale_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar,
                                    b_scale_bufs.index(index), pred)
    return producer.next(pred)


@gluon.jit
def issue_mma(consumer, c_bars, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, producer, p_bars, acc_tmem, use_acc, pred):
    c_index = consumer.index
    mbarrier.wait(c_bars.index(c_index), consumer.phase, pred)
    async_mma_scaled_impl(a_bufs.index(c_index), b_bufs.index(c_index), a_scale_bufs.index(c_index),
                          b_scale_bufs.index(c_index), acc_tmem, use_acc, pred)
    tcgen05_commit(p_bars.index(producer.index), pred)
    return consumer.next(pred), producer.next(pred)


@gluon.jit
def mma_scaled_pipelined_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, num_buffers: gl.constexpr,
                                SchedulerImpl: gl.constexpr):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    # scale 加载比操作数加载小得多(相差 VEC_SIZE 倍)。
    # 我们可以为 scales 使用比操作数更少的缓冲区来节省 shared memory, 因为 scale 加载延迟更低, 但这留给读者作为练习。
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_type.shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_type.shape,
                                             b_scale_desc.layout)
    acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    load_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_bars.index(i), count=1)
    load_producer = t8.Counter.create(0, num_buffers)
    load_consumer = t8.Counter.create(0, num_buffers)

    # 如果 BLOCK_N=256, 双缓冲累加器将使用 tensor memory 的全部 512 列, 这将没有空间留给 scales 的 tensor memory。
    num_acc_buffers: gl.constexpr = 2 if BLOCK_N < 256 else 1
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, gl.num_warps())
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)
    acc_idx = 0

    mma_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(mma_bars.index(i), count=1)
    mma_producer = t8.Counter.create(0, num_acc_buffers)
    mma_consumer = t8.Counter.create(0, num_acc_buffers)

    scheduler = SchedulerImpl.initialize(c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # 剥离的内层循环序言。使用谓词来屏蔽当 K 太小时会越界的剥离迭代, 但假设 K > 0, 即我们至少执行一次内层循环迭代。
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        load_producer = issue_loads(load_producer, pid_m, pid_n, ki, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs,
                                    b_bufs, a_scale_bufs, b_scale_bufs, load_bars, pred=ki < K)
    k = BLOCK_K * (num_buffers - 2)
    load_producer = issue_loads(load_producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs,
                                b_bufs, a_scale_bufs, b_scale_bufs, load_bars, pred=k < K)

    load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                                            mma_producer, mma_bars, acc_bufs.index(acc_idx), use_acc=False, pred=True)
    for _ in range(num_tiles):
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            load_producer = issue_loads(load_producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc,
                                        a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, load_bars, pred=True)
            load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs,
                                                    a_scale_bufs, b_scale_bufs, mma_producer, mma_bars,
                                                    acc_bufs.index(acc_idx), use_acc=True, pred=True)
            # 等待第 N-1 个 MMA 完成, 这样我们才能继续发起加载。
            mbarrier.wait(mma_bars.index(mma_consumer.index), mma_consumer.phase)
            mma_consumer = mma_consumer.next()

        # 剥离下一个序言并将其与流水线排空循环融合。
        epilogue_pid_m, epilogue_pid_n = pid_m, pid_n
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        has_next_tile = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            load_producer = issue_loads(load_producer, pid_m, pid_n, ki, a_desc, b_desc, a_scale_desc, b_scale_desc,
                                        a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, load_bars, has_next_tile and ki < K)

            pred = K > ki + BLOCK_K
            load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs,
                                                    a_scale_bufs, b_scale_bufs, mma_producer, mma_bars,
                                                    acc_bufs.index(acc_idx), use_acc=True, pred=pred)
            mbarrier.wait(mma_bars.index(mma_consumer.index), mma_consumer.phase, pred)
            mma_consumer = mma_consumer.next(pred)

        k = BLOCK_K * (num_buffers - 2)
        load_producer = issue_loads(load_producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs,
                                    b_bufs, a_scale_bufs, b_scale_bufs, load_bars, pred=has_next_tile and k < K)
        cur_acc_buf = acc_bufs.index(acc_idx)

        # 与 Hopper 相比, 我们可以让 Blackwell MMA 重叠得更多一些, 因为累加器存储在 tensor memory 中。
        # 当累加器没有双缓冲时, 我们会在加载完当前 tile 的最终累加器之后、发起 TMA store 之前, 开始下一个 tile 的 MMA。
        # 当累加器双缓冲时, 我们可以在当前 tile 的最后一个 MMA 完成之前就开始下一个 tile 的第一个 MMA。
        if num_acc_buffers == 2:
            acc_idx ^= 1
            load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs,
                                                    a_scale_bufs, b_scale_bufs, mma_producer, mma_bars,
                                                    acc_bufs.index(acc_idx), use_acc=False, pred=has_next_tile)
        mbarrier.wait(mma_bars.index(mma_consumer.index), mma_consumer.phase)
        mma_consumer = mma_consumer.next()
        acc = cur_acc_buf.load(acc_reg_layout)
        if num_acc_buffers == 1:
            load_consumer, mma_producer = issue_mma(load_consumer, load_bars, a_bufs, b_bufs,
                                                    a_scale_bufs, b_scale_bufs, mma_producer, mma_bars,
                                                    acc_bufs.index(acc_idx), use_acc=False, pred=has_next_tile)

        acc = acc.to(c_desc.dtype)
        # 通过等待前一个 store 完成来流水线化 store。
        tma.store_wait(0)
        acc_smem.store(acc)
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [epilogue_pid_m * BLOCK_M, epilogue_pid_n * BLOCK_N], acc_smem)

    # 等待最后一个 store。
    tma.store_wait(0)
    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(load_bars.index(i))
    for i in gl.static_range(num_acc_buffers):
        mbarrier.invalidate(mma_bars.index(i))
```

### Warp-Specialized 版本

我们还提供一个 warp-specialized 实现的示例。我们编写的辅助函数简化了 warp-specialized 代码的编写。

```python
@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_scale_desc: tma.tensor_descriptor
    b_scale_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    a_scale_bufs: gl.shared_memory_descriptor
    b_scale_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SchedulerImpl: gl.constexpr

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    M: gl.tensor
    N: gl.tensor
    K: gl.tensor

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                 load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, SchedulerImpl, BLOCK_M,
                 BLOCK_N, BLOCK_K, M, N, K):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_scale_desc = a_scale_desc
        self.b_scale_desc = b_scale_desc
        self.a_bufs = a_bufs
        self.b_bufs = b_bufs
        self.a_scale_bufs = a_scale_bufs
        self.b_scale_bufs = b_scale_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.SchedulerImpl = gl.constexpr(SchedulerImpl)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.M = M
        self.N = N
        self.K = K


@gluon.jit
def mma_scaled_load_partition(p):
    state = t8.Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.M, p.N, p.BLOCK_M, p.BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        for k in range(0, p.K, p.BLOCK_K):
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            state = issue_loads(state, pid_m, pid_n, k, p.a_desc, p.b_desc, p.a_scale_desc, p.b_scale_desc, p.a_bufs,
                                p.b_bufs, p.a_scale_bufs, p.b_scale_bufs, p.load_ready_bars, pred=True)


@gluon.jit
def mma_scaled_mma_partition(p):
    load_state = t8.Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = t8.Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.M, p.N, p.BLOCK_M, p.BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, p.K, p.BLOCK_K):
            _, load_state = issue_mma(load_state, p.load_ready_bars, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs,
                                      load_state, p.load_empty_bars, acc_buf, use_acc, pred=True)
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def mma_scaled_epilogue_partition(p):
    acc_layout: gl.constexpr = get_tmem_reg_layout(p.c_desc.dtype, (p.BLOCK_M, p.BLOCK_N), p.acc_bufs.type.layout,
                                                   gl.num_warps())
    acc_state = t8.Counter.create(0, p.acc_empty_bars.shape[0])
    acc_smem = gl.allocate_shared_memory(p.c_desc.dtype, p.c_desc.block_type.shape, p.c_desc.layout)
    scheduler = p.SchedulerImpl.initialize(p.M, p.N, p.BLOCK_M, p.BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load(acc_layout)
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()

        tma.store_wait(0)
        acc_smem.store(acc.to(p.c_desc.dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(p.c_desc, [pid_m * p.BLOCK_M, pid_n * p.BLOCK_N], acc_smem)
    tma.store_wait(0)


@gluon.jit
def mma_scaled_warp_specialized_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, num_buffers: gl.constexpr,
                                       SchedulerImpl: gl.constexpr):
    A_ELEM_PER_BYTE: gl.constexpr = 2 if a_desc.dtype == gl.uint8 else 1
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    M = c_desc.shape[0]
    N = c_desc.shape[1]
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_type.shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_type.shape,
                                             b_scale_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    num_acc_buffers: gl.constexpr = 2 if BLOCK_N < 256 else 1
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = PartitionArgs(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                      load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, SchedulerImpl,
                      BLOCK_M, BLOCK_N, BLOCK_K, M, N, K)
    gl.warp_specialize([
        (mma_scaled_epilogue_partition, (p, )),
        (mma_scaled_mma_partition, (p, )),
        (mma_scaled_load_partition, (p, )),
    ], [1, 1], [24, 24])


def mma_scaled(A, B, A_scale, B_scale, VEC_SIZE, impl_kernel, GROUP_SIZE_M=8, out_dtype=torch.float16):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128 if torch.float8_e4m3fn in [A.dtype, B.dtype] else 256
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype
    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N)
    A_scale_desc = make_scales_descriptor(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE)
    B_scale_desc = make_scales_descriptor(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE)

    # 替换 TMA 描述符布局为不带 swizzling 的布局, 以便 unswizzled 布局与 `tcgen05_copy` 兼容。
    no_swizzle_layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
    A_scale_desc = replace(A_scale_desc, layout=no_swizzle_layout)
    B_scale_desc = replace(B_scale_desc, layout=no_swizzle_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    impl_kernel[grid](A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, 3, SchedulerImpl)
    return C_desc.base
```

### 性能测试结果

```
|    format     | pipelined tflops/s | warp-specialized tflops/s |
|---------------|--------------------|---------------------------|
| mxfp8 x mxfp8 |            2018.58 |                   2378.49 |
| mxfp4 x mxfp4 |            3916.62 |                   4870.97 |
| mxfp8 x mxfp4 |            2144.05 |                   2615.73 |
| nvfp4 x nvfp4 |            3842.19 |                   4846.83 |
```

正如预期的那样, 我们获得了巨大的加速。实际上, 我们已经非常接近 NVIDIA 承诺的 5 petaflops 了。

尽管软件流水线版本更慢, 但演示如何实现它仍然很有用, 因为在某些情况下软件流水线会比 warp-specialization 更快。
我们也借此机会展示了与 Hopper MMA 相比, Blackwell MMA 能实现的额外重叠。

我们还展示了如何使用 `tcgen05_copy` 将 MMA scaled 抽象为一个异步 MMA 操作,
并以与 `tcgen05_mma` 相同的方式对其进行流水线化或 warp-specialize。

### 总结

本教程的主要收获:

- scales 在 global memory 中的布局非常重要, 会极大地影响性能。
- `tcgen05_copy` 是将 scales 复制到 tensor memory 的好方法。
- 通过优化 scale 布局和使用 TMA 加载, 可以将性能提升 20 倍以上。
- 流水线化和 warp-specialization 可以进一步提升性能, 接近硬件理论峰值。
