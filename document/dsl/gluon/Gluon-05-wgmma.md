## Gluon-05-wgmma

### 前言

本系列是 Gluon 的学习笔记，基于 [Gluon 官方教程](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon) 整理，覆盖了从基础概念到高级优化的完整内容，在原文基础上做了结构重组，并补充了 CUDA 编程模型和 GPU 硬件架构层面的说明。

> 知乎专栏：[DSL教程](https://www.zhihu.com/column/c_1990516179064858333)
> 完整代码：[GitHub](https://github.com/slowlyC/ai-infra-notes/tree/main/tutorials/gluon)

- [Gluon-01-Overview](https://zhuanlan.zhihu.com/p/1990504603008119998)
- [Gluon-02-Layout_Introduction](https://zhuanlan.zhihu.com/p/1990509278801457706)
- [Gluon-03-Async_Copy](https://zhuanlan.zhihu.com/p/1990517098083029502)
- [Gluon-04-TMA](https://zhuanlan.zhihu.com/p/1990517971483906093)
- **Gluon-05-wgmma**（本文）
- [Gluon-06-tcgen05](https://zhuanlan.zhihu.com/p/1990835546797405057)
- [Gluon-07-Persistent_Kernel](https://zhuanlan.zhihu.com/p/2003592603732563562)
- [Gluon-08-Warp_Specialization](https://zhuanlan.zhihu.com/p/2003602919912649692)
- [Gluon-09-TMA_Gather_Scatter](https://zhuanlan.zhihu.com/p/2003599190585017693)
- [Gluon-10-tcgen05_copy](https://zhuanlan.zhihu.com/p/2003603029975381459)
- [Gluon-11-tcgen05-mma-scaled](https://zhuanlan.zhihu.com/p/2003603119259550000)

Warp Group MMA (也称为 WGMMA 或 MMAv3) 是 Hopper 架构特有的指令, 用于利用 Tensor Cores 执行矩阵
乘累加 (Matrix Multiply-Accumulate, MMA) 运算. WGMMA 指令是异步的, 这意味着它们可以被流水线化 (Pipelined).

在本教程中, 我们将介绍如何在 Gluon 中使用 WGMMA. 我们将构建一个简单的矩阵乘法 Kernel
来演示 WGMMA 的实际用途, 并展示一个对 WGMMA 进行流水线化以获得更好性能的示例.

### 代码

```python
import pytest
import torch
import triton
import itertools
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma_init,
    warpgroup_mma,
    warpgroup_mma_wait,
)


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


if __name__ == "__main__" and not is_hopper():
    raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")

# %%
# 首先，我们通过一个网格大小为 (1,) 的简单 Kernel 来演示 WGMMA. 
#
# `warpgroup_mma` 执行运算 `d = a * b + c`. 
# 其中 `a` 操作数可以来自寄存器或共享内存, `b` 操作数必须来自共享内存, 而 `c` 操作数必须位于寄存器中.
#
# `warpgroup_mma` 本身由许多较小的 `wgmma.mma_async` PTX 指令组成, 这些指令支持有限的指令形状 (Instruction Shapes).
# 指令形状指定为 [m, n, k], 其中:
#
# - `k` 始终为 256 / A.dtype.primitive_bitwidth
# - `m` 始终为 16
# - `n` 的选择规则如下:
#
# 对于浮点 dtype, `n` 必须是 8 的正倍数, 最大可达 256. WGMMA 虽然支持 8 位整数, 但 `n` 必须从以下值中选择:
#
#   224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 24, 16, 8
#
# 此外，`n` 必须能整除 `BLOCK_N` (MMA Tile 的内部维度), 并且必须小于或等于 `maxN`, 
# 其中 `maxN` 计算如下:
#
#     mReps = ceildiv(M, m)
#     nReps = ceildiv(num_warps, mReps)
#     maxN = max(N // nReps, 8)
#
# `warpgroup_mma` 使用 `warps_per_cta` 参数在 Warp 之间分配 MMA 工作, 
# 方式与`BlockedLayout.warps_per_cta` 在 Warp 之间切分 Tensor 类似. 
# `warps_per_cta`的最小不可分单元是 `[4, 1]`. 这意味着 WGMMA 至少需要 4 个 Warp,它们共同组成一个 Warp Group. 
# 要选择正确的 `warps_per_cta`, 可以从原子单元 `[4, 1]` 开始,沿任意维度将其翻倍, 直到它匹配总 Warp 数量.
# 注意由于 `m=16` 且 M 维度至少有 4 个 Warp, 因此 M 维度总大小至少为 64.
#
# 当 `num_warps=8` 时, 我们可以选择 `[4, 2]` 或 `[8, 1]`.
# 回顾 02-layouts 教程, 不同的选择可能会影响例如归约 (Reduction) 等操作的性能.
#
# `warpgroup_mma` 是异步操作, 其完成状态由 Commit Group 跟踪, 这与异步复制和 TMA 存储类似.
# 发出 WGMMA 操作会隐式将其提交到 WGMMA Group, 我们可以等待直到仅剩 N 个未完成的操作.
#
# 由于 `warpgroup_mma` 是异步的, 在操作完成前我们无法访问其结果（即便结果已在寄存器中），也不能修改其输入的共享内存数据. 
# WGMMA 通过异步代理 (Asynchronous Proxy) 访问共享内存. 
# 由于 TMA 也通过异步代理访问共享内存, 因此在 TMA 和 WGMMA 指令之间不需要额外的 Fence.
#
# 但在常规共享内存存储 (Store) 和 `warpgroup_mma` 之间, 需要 Fence 来确保共享内存访问的顺序.
#
# ```python
# b_smem.store(b)
# fence_async_shared()
# warpgroup_mma(a, b_smem, c, is_async=True)
# ```
#
# WGMMA 的完成意味着它从共享内存的读取已结束. 因此, 在等待操作完成后, 写入共享内存
# 输入缓冲区是安全的:
#
# ```python
# d = warpgroup_mma(a, b_smem, c, is_async=True)
# d = warpgroup_mma_wait(num_outstanding=0, deps=(d, ))
# b_smem.store(b)
# ```
#
# 如果 LHS 操作数是从共享内存加载到寄存器中的, WGMMA 的完成意味着该次共享内存加载已结束,
# 随后通过异步代理对该缓冲区的访问不需要 Fence:
#
# ```python
# a = a_smem.load(dot_operand_layout)
# d = warpgroup_mma(a, b_smem, c, is_async=True)
# d = warpgroup_mma_wait(num_outstanding=0, deps=(d, ))
# tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
# ```

# %%
# 让我们实现一个使用 WGMMA 的简单矩阵乘法 Kernel.

@gluon.jit
def small_mma_kernel(a_desc, b_desc, c_desc, d_desc,  #
                     LHS_IN_REG: gl.constexpr, INSTR_SHAPE_N: gl.constexpr, num_warps: gl.constexpr):
    # 加载 A, B 和 C Tile.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # A 的形状为 [M, K].
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    # B 的形状为 [K, N].
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    # C 的形状为 [M, N].
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
    tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # 我们将 Kernel 参数化为 LHS_IN_REG 和 INSTR_SHAPE_N, 观察它们对性能的影响.
    # primitive_bitwidth返回基础数据类型的位宽。例如:
    # float16 (半精度浮点): primitive_bitwidth = 16, float32 (单精度浮点): primitive_bitwidth = 32
    m: gl.constexpr = 16
    k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
    n: gl.constexpr = INSTR_SHAPE_N
    warps_per_cta: gl.constexpr = [num_warps, 1]

    # MMA 形状通过 `c` 的 Layout 传递, 它必须始终是 NVMMADistributedLayout.
    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

    # 当 A 通过寄存器传递时, 它必须具有以下 Layout:
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_desc.dtype.primitive_bitwidth,
    )

    # 当操作数通过共享内存传递时, 它必须具有 NVMMASharedLayout. TMA 也需要使用 NVMMASharedLayout.
    gl.static_assert(isinstance(a_smem.type.layout, gl.NVMMASharedLayout))
    gl.static_assert(isinstance(b_smem.type.layout, gl.NVMMASharedLayout))

    if LHS_IN_REG:
        a = a_smem.load(a_reg_layout)
    else:
        a = a_smem

    c = c_smem.load(c_layout)

    # 发出异步 WGMMA. 
    # 注意 `is_async=False` 是默认值, 即默认同步。在本教程中, 我们将始终使用 `is_async=True`.
    #
    # 另一个需要考虑的重要标志是 `use_acc`.
    #  当 `use_acc=False` 时, `c` 输入会被忽略, 累加器被零初始化. 这是一种高效的零初始化累加器的方式.
    d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)

    # 为了确保 `warpgroup_mma`、等待 (Wait) 和结果使用之间的正确顺序, 
    # 你必须通过 `deps` 参数将 `warpgroup_mma` 的结果传递给等待函数, 并使用 `warpgroup_mma_wait` 的返回值.
    #
    # 等待 0 个未完成操作, 意味着我们确认 WGMMA 已全部完成.
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d, ))

    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    d_smem.store(d)
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG=False, num_warps=4):
    # 计算 `d = a * b + c`.
    a_layout = gl.NVMMASharedLayout.get_default_for(A.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)

    a_desc = TensorDescriptor.from_tensor(A, A.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, cd_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, cd_layout)
    small_mma_kernel[(1, )](
        a_desc, b_desc, c_desc, d_desc,  #
        LHS_IN_REG, INSTR_SHAPE_N, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(64, 32, 32), (64, 256, 128)])
@pytest.mark.parametrize("LHS_IN_REG", [False, True])
@pytest.mark.parametrize("INSTR_SHAPE_N", [16, 64])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_small_mma(M, N, K, LHS_IN_REG, INSTR_SHAPE_N, num_warps):
    maxN = max(N // triton.cdiv(num_warps, triton.cdiv(M, 16)), 8)
    if INSTR_SHAPE_N > maxN:
        pytest.skip(f"INSTR_SHAPE_N={INSTR_SHAPE_N} is too large for M={M}, N={N}, num_warps={num_warps}")

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)
    small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG, num_warps)
    torch.testing.assert_close(A @ B + C, D, atol=1e-3, rtol=1e-1)


# %%
# 让我们研究一下不同参数 (Knobs) 对 WGMMA 性能的影响.

if __name__ == "__main__":
    print("Benchmarking WGMMA")
    print("==================")
    M, N, K = 64, 128, 128
    num_warps = 4
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)

    print("LHS_IN_REG INSTR_SHAPE_N time (us)")
    for LHS_IN_REG, INSTR_SHAPE_N in itertools.product([False, True], [16, 32, 64, 128]):
        fn = lambda: small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG, num_warps)
        ms = triton.testing.do_bench(fn)
        print(f"{LHS_IN_REG!s:>10} {INSTR_SHAPE_N:>13} {ms*1000:>9.2f}")
    print()

# ```
# LHS_IN_REG INSTR_SHAPE_N time (us)
#      False            16      9.47
#      False            32      8.48
#      False            64      8.32
#      False           128      8.32
#       True            16      9.32
#       True            32      8.60
#       True            64      8.37
#       True           128      8.36
# ```
#
# 通常选择最大的 N 会产生最佳性能, 因为每条 `wgmma.mma_async` 指令能处理更多数据.
# 在本例中, 将 LHS 放在寄存器中反而更慢, 因为我们必须先将数据从共享内存加载到寄存器.
# 不过, 如果数据原本就已经在寄存器中, 直接使用寄存器数据通常比先存入共享内存再使用要快.

# %%
# 下面让我们使用WGMMA实现一个分块矩阵乘法Kernel.

# 这个装饰器允许我们从 Gluon constexpr 函数中调用此函数.
@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    """
    计算 warps_per_cta 值. 从原子单元 `[4, 1]` 开始, 沿任意维度将其翻倍, 直到它匹配总 Warp 数量.
    """
    warps_per_cta = [4, 1]
    m = 16
    # 增加 Tile 大小直到用完所有 Warp.
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        # 仅当不会导致广播 (Broadcast) 时才沿 M 轴 Tile.
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    """
    计算最佳的 n 值.
    """
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n


@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    """
    得到 WGMMA Layout.
    """
    m = 16
    k = 256 // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )


@gluon.jit
def blocked_matmul_kernel(a_desc, b_desc, c_desc,  #
                          TRANSPOSE_B: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    # 此 program 负责计算 C 矩阵中 (pid_m, pid_n) 位置的块。
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # 确定 WGMMA Layout.
    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        # 加载 A 和 B 的 Tile.
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_smem)
        if TRANSPOSE_B:
            tma.async_copy_global_to_shared(b_desc, [off_n, k], bar, b_smem)
        else:
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_smem)
        mbarrier.wait(bar, phase=phase)

        # 我们可以通过在共享内存中创建 B Tile 的转置视图来转置 B.
        # WGMMA 会自动处理转置操作.
        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        # 在 0 和 1 之间切换奇偶 Phase
        # 通过轮换 phase，确保每次 wait 等待的是当前循环数据加载的完成信号
        phase ^= 1

        # 发起并等待 WGMMA 完成.
        acc = warpgroup_mma(a_smem, b, acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

    mbarrier.invalidate(bar)

    # 转换精度，并存储 C Tile.
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

    B_BLOCK_SHAPE = [BLOCK_N, BLOCK_K] if TRANSPOSE_B else [BLOCK_K, BLOCK_N]
    b_layout = gl.NVMMASharedLayout.get_default_for(B_BLOCK_SHAPE, gl.float16)
    b_desc = TensorDescriptor.from_tensor(B, B_BLOCK_SHAPE, b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_kernel[grid](a_desc, b_desc, c_desc, TRANSPOSE_B, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_blocked_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn((N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)

    C_ref = A @ (B.T if TRANSPOSE_B else B)
    torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)


# %%
# 让我们对分块矩阵乘法 kernel 进行基准测试。
# 我们需要寻找最佳的块大小,与其盲目地对所有可能性进行自动调优, 不如利用一些原则来缩小搜索空间.
#
# 首先, 尽量为 WGMMA Layout 选择最大的 `n`. 基于 `maxN` 的公式, 这通常需要 `BLOCK_N >= 256`.
# 由于当前 Kernel 未实现 TMA 加载与 WGMMA 的重叠 (流水线), 我们希望每个 SM 能驻留多个 Program
# (即提高 "占用率", Occupancy), 这样当一个 Kernel 停滞时, SM 可以切换到另一个.
#
# 每个 SM 的资源是有限的, Kernel 的资源使用量决定了其最大占用率. 
# SM 使用 Warp 调度器在 Warp 之间调度任务, 它可以极快地切换执行 Warp, 类似于超线程技术.
#
# 基于寄存器和共享内存的约束, 我们可以筛选出满足目标占用率的配置. 
# 请记住, 这些只是经验法则, 很难确定这些配置是否一定能产生最佳性能.

def find_configs(occupancy, dtype, num_buffers=1):
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    # 假设约 1 KB 的 smem 被 mbarrier、编译器生成的代码等占用.
    smem = 228 * 1024 // occupancy - 1024

    configs = []
    BLOCK_MNK = [32, 64, 128, 256]
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps in itertools.product(BLOCK_MNK, BLOCK_MNK, BLOCK_MNK, [4, 8]):
        # 假设每个线程基线使用约 16 个寄存器.
        regs = 64 * 1024 // occupancy - 16 * num_warps * 32

        a_smem = BLOCK_M * BLOCK_K * dtype_bytes
        b_smem = BLOCK_N * BLOCK_K * dtype_bytes
        acc_smem = BLOCK_M * BLOCK_N * dtype_bytes
        # A 和 B 的 SMEM 不与 C 共享生命周期.
        if max((a_smem + b_smem) * num_buffers, acc_smem) > smem:
            continue

        # 累加器是唯一的 f32 内存 Tensor.
        acc_regs = BLOCK_M * BLOCK_N
        # 每个线程的最大寄存器数是 256. 接近此限制也可能导致溢出.
        if acc_regs // num_warps // 32 >= 256:
            continue
        if acc_regs > regs:
            continue

        instr_shape_n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
        configs.append((BLOCK_M, BLOCK_N, BLOCK_K, num_warps, instr_shape_n, occupancy))

    def filter_configs(configs, instr_shape_n):
        max_n_configs = [cfg for cfg in configs if cfg[4] == instr_shape_n]
        # 筛选具有最大 BLOCK_M * BLOCK_K 的配置.
        max_block_mk = max(cfg[0] * cfg[2] for cfg in max_n_configs)
        return [cfg for cfg in max_n_configs if cfg[0] * cfg[2] == max_block_mk]

    top_instr_shape_n = sorted({cfg[4] for cfg in configs}, reverse=True)
    result_configs = filter_configs(configs, top_instr_shape_n[0])
    if len(top_instr_shape_n) > 1:
        result_configs += filter_configs(configs, top_instr_shape_n[1])
    return result_configs


if __name__ == "__main__":
    print("Benchmarking selected configs")
    print("=============================")
    # 以防万一, 检查占用率 1 的配置.
    # [(128, 256, 256, 8, 256, 1), (256, 128, 256, 8, 128, 1)]
    configs = find_configs(occupancy=1, dtype=torch.float16)
    # [(128, 256, 256, 8, 256, 1), (256, 128, 256, 8, 128, 1), (64, 256, 128, 4, 256, 2), \
    # (64, 128, 256, 4, 128, 2), (128, 128, 128, 4, 128, 2), (128, 128, 128, 8, 128, 2)]
    configs += find_configs(occupancy=2, dtype=torch.float16)
    # 在大矩阵乘法上对配置进行基准测试. 请记住, 最佳超参数可能取决于矩阵乘法的形状.
    M, N, K = 8192, 8192, 16 * 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    print("BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n occupancy time (ms) tflops/s")
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps, instr_shape_n, occupancy in configs:
        fn = lambda: blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, False, num_warps)
        ms = triton.testing.do_bench(fn)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"{BLOCK_M:>7} {BLOCK_N:>7} {BLOCK_K:>7} {num_warps:>9} {instr_shape_n:>13} "
              f"{occupancy:>9} {ms:>9.2f} {tflops_per_sec:>8.2f}")
    print()

# 性能结果如下:
# ```
# BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n occupancy time (ms) tflops/s
#     128     256     256         8           256         1      5.34   412.14
#     256     128     256         8           128         1      5.67   387.74
#      64     256     128         4           256         2      4.64   474.03
#      64     128     256         4           128         2      6.18   355.60
#     128     128     128         4           128         2      4.98   441.88
#     128     128     128         8           128         2      5.79   380.08
# ```
#
# 我们的假设得到了验证: 占用率 2 且 `BLOCK_N=256` 的配置确实表现最佳.
# 至于对所有超参数进行全面的自动调优, 就留给读者作为练习了.

# %%
# 466 TFLOPS 是个不错的起点, 我们尚未利用 WGMMA 的异步特性, 也未像之前教程那样对 TMA 加载进行流水线化.
#
# 接下来, 我们保持 TMA 加载同步, 使用WGMMA异步计算, 着重演示 WGMMA 的流水线化. 
# 这需要对操作数使用双缓冲 (Double Buffering),即在 WGMMA 处理前一个缓冲区的数据时, 同时加载下一组数据到备用缓冲区.

# ═══════════════════════════════════════════════════════════════════════════
#                     双缓冲流水线执行时间线
# ═══════════════════════════════════════════════════════════════════════════
# 迭代 0 (k=0, index=0):
#   缓冲区 0: [加载数据块0] → 等待完成
#   缓冲区 1: [闲置]
#
#   操作顺序:
#     1. 加载到缓冲区0
#     2. wait() - 等待加载完成
#     3. warpgroup_mma_wait(0) - 无操作(没有之前的计算)
#     4. warpgroup_mma(缓冲区0) - 启动异步计算
#     5. index ^= 1  →  index = 1
#                                         WGMMA 正在后台计算...
#                                         ↓
# 迭代 1 (k=BLOCK_K, index=1):
#   缓冲区 0: [正在被 WGMMA 使用]
#   缓冲区 1: [加载数据块1] → 等待完成
#
#   操作顺序:
#     1. 加载到缓冲区1(与缓冲区0的计算进行重叠)
#     2. wait() - 等待加载完成
#     3. warpgroup_mma_wait(0) - 等待缓冲区0的计算完成
#     4. warpgroup_mma(缓冲区1) - 启动新计算
#     5. index ^= 1  →  index = 0
#               ↑                                WGMMA 正在后台计算...
#               | 重叠                           ↓
#
# 迭代 2 (k=2*BLOCK_K, index=0):
#   缓冲区 0: [加载数据块2] → 等待完成
#   缓冲区 1: [正在被 WGMMA 使用]
#
#   操作顺序:
#     1. 加载到缓冲区0(与缓冲区1的计算重叠)
#     2. wait() - 等待加载完成
#     3. warpgroup_mma_wait(0) - 等待缓冲区1的计算完成
#     4. warpgroup_mma(缓冲区0) - 启动新计算
#     5. index ^= 1  →  index = 1


@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # 为 A 和 B 各分配 2 个缓冲区.
    a_smem = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)
    index = 0

    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = warpgroup_mma_init(gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout))

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        a = a_smem.index(index)
        b = b_smem.index(index)

        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a)
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1

        # 第一次迭代时，没有进行中的 WGMMA 操作时, `warpgroup_mma_wait` 实际上是无操作 (no-op),
        # 后续迭代时，这行代码等待上一次迭代的 WGMMA 完成, 从而实现访存和计算重叠执行.
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))
        acc = warpgroup_mma(a, b, acc, is_async=True)

        # 切换到下一个缓冲区. 此时 TMA 加载会在 WGMMA 仍在运行时开始.
        index ^= 1

    # 等待最后一个 WGMMA 完成.
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

    mbarrier.invalidate(bar)

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_blocked_matmul_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


# %%
# 让我们对流水线版本进行性能测试。
# 我们之前的最佳块配置使用了 160 KB 的 smem, 这对occupancy=2来说太多了,
# 但如果不利用剩余的 68 KB 空间, 又可能浪费性能.
# 因此, 最佳 Kernel 可能会为了保持 2 的占用率而适当减小 `BLOCK_N`.

if __name__ == "__main__":
    print("Benchmarking pipelined matmul")
    print("=============================")
    configs = find_configs(occupancy=1, dtype=torch.float16, num_buffers=2)
    configs += find_configs(occupancy=2, dtype=torch.float16, num_buffers=2)
    # 添加我们之前的最佳配置, 因为它可能被筛选掉了.
    configs.append([64, 256, 128, 4, 256, 2])

    print("BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n occupancy time (ms) tflops/s")
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps, instr_shape_n, occupancy in configs:
        fn = lambda: blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        ms = triton.testing.do_bench(fn)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"{BLOCK_M:>7} {BLOCK_N:>7} {BLOCK_K:>7} {num_warps:>9} {instr_shape_n:>13} "
              f"{occupancy:>9} {ms:>9.2f} {tflops_per_sec:>8.2f}")
    print()

# %%
# ```
# BLOCK_M BLOCK_N BLOCK_K num_warps instr_shape_n occupancy time (ms) tflops/s
#     128     256     128         8           256         1      5.16   426.06
#     256     128     128         8           128         1      5.70   385.85
#      64     256      64         4           256         2      5.27   417.50
#      64     128     128         4           128         2      5.71   384.98
#     128     128      64         4           128         2      4.44   495.31
#     128     128      64         8           128         2      4.92   446.81
#      64     256     128         4           256         2      6.05   363.36
# ```
#
# 结果显示, `instr_shape_n=128` 确实是最佳配置. 注意之前非流水线版本的最佳配置在这里反而慢了 100 TFLOPS 以上!
# 流水线化 WGMMA 带来了约 5% 的性能提升, 前提是我们必须重新调整超参数.
#
# 将异步 TMA 加载和 WGMMA 双重流水线化留给读者作为练习.
#
# 总结:
#
# - WGMMA 是 Hopper 架构特有的指令, 用于执行块级 MMA.
# - WGMMA 是异步的, 可以与其他操作重叠执行.
# - WGMMA 对其 Layout 有诸多限制.
# - LHS 操作数可以位于共享内存或寄存器中.
# - WGMMA 支持转置输入, 我们可以直接创建转置视图.
# - WGMMA 流水线化通过启用重叠执行来提升性能.
# - 超参数调优对获得最佳性能至关重要.
```

### 总结
- WGMMA 是 Hopper 架构特有的指令, 用于执行块级 MMA.
- WGMMA 是异步的, 可以与其他操作重叠执行.
- WGMMA 对其 Layout 有诸多限制.
- LHS 操作数可以位于共享内存或寄存器中.
- WGMMA 支持转置输入, 我们可以直接创建转置视图.
- WGMMA 流水线化通过启用重叠执行来提升性能.
- 超参数调优对获得最佳性能至关重要.