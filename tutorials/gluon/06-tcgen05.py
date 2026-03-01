"""
第5代 Tensor 核心 (TCGen05)
===========================

在Blackwell架构中, 为Tensor Core引入了一个全新的内存空间—— Tensor Memory, 用于异步 MMA 指令。

在本教程中, 我们会构建一个简单的矩阵乘法 kernel, 来学习如何分配和使用 Tensor memory, 
并演示 `tcgen05` MMA 指令的用法, 以及如何对 MMA 指令进行流水线优化。
.load(layout) 都是 gmem/smem/tmem -> reg
.store() 都是 reg -> gmem/smem/tmem
"""

# %%
# Tensor Memory 介绍:
# Tensor memory 是一个二维内存空间, 在每个 CTA 上有 128 行 × 512 列的单元, 每个单元为 32 bit, 与寄存器文件大小相同。
# 访问 Tensor memory 比共享内存更快, 但也有一些额外的限制:

# - 每个 warp 只能根据其 warp ID 访问其中的 32 行, 因此需要整个 warp 组(4个warp)协同才能访问全部 128 行物理空间。
# - Tensor memory 按列数分配, 分配大小必须在[32, 512] 范围内, 且为 2 的整数次幂。
# - 在 Gluon 中, Tensor memory 的load和存储操作需要 4 或 8 个 warp 参与。
# - 在 Gluon 中, 只有 2D tensor 可以与 Tensor memory 进行交互。
# - 数据可以从共享内存通过 tcgen05_copy 异步复制到 Tensor memory。

# 存储在 Tensor memory 中的数据也需要有layout, 就像共享内存一样。
# 如果寄存器中的Tensor与 Tensor memory 交互, 那么该Tensor的layout会受到 Tensor memory layout 的约束。

# 关于 Tensor memory 的几点补充说明:
# - Tensor memory 本质上是一个额外的寄存器文件。
#   128 × 512 = 65536 个 32 bit单元, 恰好与一个SM上的总寄存器文件大小相同(65536个32位寄存器=256KB)。
# - Tensor memory 可以独立于 MMA 指令使用。
#   只要满足 layout 限制, 它可以作为共享内存的替代方案来传输数据。
# - Tensor memory 在 SM 上动态分配, 虽然不会直接影响占用率, 但如果可用空间不足, 分配操作会被阻塞。

# Tensor memory layout 将数据组织成 2D 块的形式:

# ```python
# TensorMemoryLayout(
#     block=(blockM, blockN), 
#     col_stride=32//primitive_bitwidth, 
# )
# ```

# tensor 被划分为 (blockM, blockN) 的块, 其中:
# - blockM 必须是 64 或 128(此处是逻辑上的, 物理上每个warp只能访问32行)
# - blockN 必须是 [1, 256] 之间的 2 的幂

# col_stride (int): Number of 32-bit columns to advance between logically adjacent columns.
#      Packed layouts use a stride of 1. Unpacked layouts use ``32 / bitwidth``.
# col_stride 表达打包/非打包, 对于小于 32 bit的数据类型, 如bf16类型,
#  - col_stride=1 表示打包, 可以将多个元素打包到同一个 32 位单元中, 如 2 个 bf16 元素打包到 1 个 32-bit 单元。
#  - col_stride=2 表示非打包模式, 即1 个 bf16 元素占用 1 个 32-bit 单元(高 16 位空置).
# 但此时 blockN 必须至少为 `32 // bitwidth`,例如对于 bf16 类型, blockN 至少为`32 // 16 = 2`。

# 注意:当 blockM=64 时, 包含多个块的 tensor 会在 TMEM 中被打包以充分利用物理上的全部 128 行。
# 这可能会使 TMEM 描述符的切片操作变得复杂。

# tcgen05.st 和 tcgen05.ld 是访问 Tensor Memory 的底层 PTX 指令，它们是 warp 级别的指令。
# 即整个 warp 的 32 个 lanes 协同执行一条指令, 每个 lane 负责处理数据的一部分
# 某些 Tensor memory layout 支持多种寄存器 layout, 这会影响原子操作的选择。
# 在本教程中, 我们只使用 `32x32b` 原子: 即每个 lane 存储和load 1 行 TMEM, 每个 lane 访问 32 bit。
# 即假设[BM, BN]=[64, 64], NUM_WARPS=4, 那么只会使用 warp 0,1，访问物理行 0-63。

import itertools
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, 
    allocate_tensor_memory, 
    get_tmem_reg_layout, 
    tma, 
    mbarrier, 
    tcgen05_mma, 
    tcgen05_commit, 
    fence_async_shared,
    tcgen05_copy,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")


#%%
# 下面我们介绍如何使用 Tensor memory 对二维数据进行读写。
# input(global memory) -> 寄存器 -> Tensor memory -> 寄存器 -> output(global memory)

@gluon.jit
def tmem_example_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr, num_warps: gl.constexpr):
    global_memory_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    offs_m = gl.arange(0, M, gl.SliceLayout(1, global_memory_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, global_memory_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    # global memory -> 寄存器
    input = gl.load(in_ptr + offs)

    # 定义 Tensor memory layout
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(64, 64), 
        col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth, 
    )
    # 分配 Tensor memory
    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty, 
        shape=[M, N], 
        layout=tmem_layout, 
    )

    # 使用get_tmem_reg_layout辅助函数, 获取访问 Tensor memory 所需的寄存器 layout。
    tmem_reg_layout: gl.constexpr = get_tmem_reg_layout(
        in_ptr.dtype.element_ty, 
        (M, N), 
        tmem_layout, 
        num_warps=num_warps, 
    )

    # 寄存器layout -> 访问 Tensor memory 所需的寄存器layout。
    input = gl.convert_layout(input, tmem_reg_layout)
    # 寄存器 -> Tensor memory
    tmem.store(input)
    # Tensor memory -> 寄存器
    output = tmem.load(tmem_reg_layout)
    # 将寄存器中的数据转换为 global memory 所需的 layout。
    output = gl.convert_layout(output, global_memory_layout)
    # 寄存器 -> global memory
    gl.store(out_ptr + offs, output)


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_example_kernel(M, N, num_warps):
    input = torch.randn(M, N, dtype=torch.float32, device="cuda")
    output = torch.empty_like(input)

    tmem_example_kernel[(1, )](input, output, M, N, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)

# %%
# tcgen05_mma介绍
# 累加器(accumulator)必须在 TMEM 中, 左操作数(LHS)可以在 SMEM 或 TMEM 中, 右操作数(RHS)必须在 SMEM 中。
# SMEM 操作数必须使用 NVMMASharedLayout。
# tcgen05_mma 是一个异步操作。在操作完成之前, 我们不能读写累加器内存, 也不能写入操作数内存。
#
# tcgen05_mma 通过异步代理访问共享内存:
#
# ```python
# b_smem.store(b)
# fence_async_shared()
# tcgen05_mma(a, b_smem, acc_tmem)
# ```
#
# 在共享内存 store 和 tcgen05_mma 之间需要 fence 来保证它们的内存访问顺序。
# tcgen05_mma 操作完成意味着它对共享内存的读取已经结束, 此时才可以安全地写入共享内存。
#
# tcgen05_mma 的完成由 mbarrier 跟踪, 在 mbarrier 上调用 tcgen05_commit，，
# 会把之前所有已发出的 tcgen05_mma 操作绑定到 mma_bar，当这些 MMA 全部完成时，硬件自动 arrive mma_bar。
# tcgen05_mma 发出时就开始执行了, 而不是由 tcgen05_commit 启动，不 commit 的话 MMA 照样算，只是不知道何时完成。
#
# 关于 mbarrier 工作原理的更多细节, 请参阅 04-tma.py。
#
# 要提交到 mbarrier, 我们可以显式调用 tcgen05_commit, 如 tcgen05_commit(mma_bar)。
# 或者将mbarrier直接传递给tcgen05_mma。如果需要, 还可以有条件地提交 mbarrier。
#
# tcgen05_mma 由多个异步 MMA 指令组成, 每个指令的形状由 TMEM layout 决定,选择更大的指令形状通常能获得更好的性能。
# 注意: tcgen05_mma 只有在仅包含 1 个逻辑块的情况下才支持 blockM=64
# (如果有多个块, 它们会被打包到 128 行中, 此时必须使用 blockM=128)。

# 接下来让我们演示如何在 MMA 操作中使用 TMEM。
# 使用tcgen05_mma api 编写一个简单的矩阵乘法 kernel,: D = A @ B + C
# 使用网格大小 (1, ) 启动, 在单个 tensor 块上执行 MMA。

@gluon.jit
def small_mma_kernel(a_desc, b_desc, c_desc, d_desc, tmem_block: gl.constexpr,  #
                     LHS_IN_TMEM: gl.constexpr, USE_COMMIT: gl.constexpr, num_warps: gl.constexpr):
    # load A、B 和 C 的 tile。
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # A 的形状为 [M, K]。
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    # B 的形状为 [K, N]。
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    # C 的形状为 [M, N]。
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
    tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)

    # 在 TMA 和 tcgen05_mma 之间重用 mbarrier 可能导致未定义行为。
    # 确保 TMA 和 tcgen05_mma 使用单独的 mbarrier, 或在使用前重新初始化。
    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=1)

    # 累加器(accumulator)必须在 TMEM 中, 左操作数(LHS)可以在 SMEM 或 TMEM 中, 右操作数(RHS)必须在 SMEM 中。
    # SMEM 操作数必须使用 NVMMASharedLayout。
    M: gl.constexpr = d_desc.block_type.shape[0]
    N: gl.constexpr = d_desc.block_type.shape[1]
    K: gl.constexpr = a_desc.block_type.shape[1]

    # 先将 c_smem 从 SMEM load 到寄存器, 再从寄存器复制到 acc TMEM 中。
    acc_tmem_layout: gl.constexpr = TensorMemoryLayout(
        tmem_block.value,
        col_stride=32 // d_desc.dtype.primitive_bitwidth, 
    )
    acc_tmem = allocate_tensor_memory(d_desc.dtype, [M, N], acc_tmem_layout)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(
        d_desc.dtype,
        (M, N),
        acc_tmem_layout,
        num_warps,
    )
    # 可以使用 tcgen05_copy 指令将 c_smem 中的数据复制到 acc_tmem 中。
    # 注意: tcgen05_copy 指令要求 TMEM layout 的 M 维度必须是 128。
    if M == 128:
        tcgen05_copy(c_smem, acc_tmem)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        mbarrier.init(bar, count=1)
    else:
        # 指定 acc_reg_layout 加载, 因此不需要再convert了
        acc = c_smem.load(acc_reg_layout)
        acc_tmem.store(acc)

    if LHS_IN_TMEM:
        # 当左操作数是 fp16 或 fp8 类型时, 它会在 TMEM 中以打包形式存储。
        lhs_tmem_layout: gl.constexpr = TensorMemoryLayout(
            tmem_block.value, 
            col_stride=1, 
        )
        lhs_tmem = allocate_tensor_memory(a_desc.dtype, [M, K], lhs_tmem_layout)

        lhs_reg_layout: gl.constexpr = get_tmem_reg_layout(
            a_desc.dtype, 
            (M, K), 
            lhs_tmem_layout, 
            num_warps, 
        )
        # 注意: tcgen05_copy 只支持 32-bit 元素类型, 而 A 矩阵是 float16, 所以 LHS 只能使用寄存器中转方式。
        lhs = a_smem.load(lhs_reg_layout)
        lhs_tmem.store(lhs)
        a = lhs_tmem
    else:
        a = a_smem

    if USE_COMMIT:
        tcgen05_mma(a, b_smem, acc_tmem)
        tcgen05_commit(bar)
    else:
        tcgen05_mma(a, b_smem, acc_tmem, mbarriers=[bar], mbarrier_preds=[True])

    # 等待 MMA 完成。
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # 另一个重要的标志是 `use_acc`。
    # 当 `use_acc=False` 时, 会忽略 TMEM 中累加器的当前值。这是将累加器零初始化的高效方式。
    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    acc = acc_tmem.load(acc_reg_layout)
    d_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def small_mma(A, B, C, D, tmem_block, LHS_IN_TMEM, USE_COMMIT, num_warps):
    a_layout = gl.NVMMASharedLayout.get_default_for(A.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)

    a_desc = TensorDescriptor.from_tensor(A, A.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, cd_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, cd_layout)

    small_mma_kernel[(1, )](
        a_desc, b_desc, c_desc, d_desc, tmem_block,  #
        LHS_IN_TMEM, USE_COMMIT, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (64, 128, 128), (64, 256, 256), (256, 64, 64)])
@pytest.mark.parametrize("LHS_IN_TMEM", [False, True])
@pytest.mark.parametrize("USE_COMMIT", [False, True])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_small_mma(M, N, K, LHS_IN_TMEM, USE_COMMIT, num_warps):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)

    blockM = min(128, M)
    blockN = N

    small_mma(A, B, C, D, (blockM, blockN), LHS_IN_TMEM, USE_COMMIT, num_warps)
    torch.testing.assert_close(A @ B + C, D, atol=1e-3, rtol=1e-1)


# %%
# 下面让我们使用 tcgen05_mma 构建一个简单的分块矩阵乘法 kernel: C = A @ B。
# 每个 program 负责计算输出矩阵的一个块。

@gluon.jit
def blocked_matmul_kernel(a_desc, b_desc, c_desc, TRANSPOSE_B: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # 此 program 负责计算 C 矩阵中 (pid_m, pid_n) 位置的块。
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    phase = 0

    # 确定 TMEM layout。
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    # 我们可以在第一次迭代时设置 `use_acc=False` 来将累加器零初始化。
    use_acc = False
    for k in range(0, K, BLOCK_K):
        # 加载 A 和 B 的 Tile.
        mbarrier.expect(tma_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], tma_bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, k] if TRANSPOSE_B else [k, off_n], tma_bar, b_smem)
        mbarrier.wait(tma_bar, phase=phase)

        # 我们可以通过在共享内存中创建 B tile 的转置视图来转置 B。
        # 这样转置会被转发给 tcgen05_mma, 由硬件自动处理。
        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        # 发起并等待 tcgen05_mma 完成。
        tcgen05_mma(a_smem, b, acc_tmem, use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=phase)
        use_acc = True

        phase ^= 1  # 在 0 和 1 之间切换 phase

    mbarrier.invalidate(tma_bar)
    mbarrier.invalidate(mma_bar)

    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, 
        (BLOCK_M, BLOCK_N), 
        tmem_layout, 
        num_warps, 
    )
    acc = acc_tmem.load(acc_reg_layout)

    # 转换精度, 并存储 C Tile.
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
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_blocked_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn((N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)

    C_ref = A @ (B.T if TRANSPOSE_B else B)
    torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)


# %%
# 让我们对分块矩阵乘法 kernel 进行基准测试。
# 关于超参数选择的更多信息, 请参考前一个教程 05-wgmma.py。

# 一些 tcgen05_mma 特定的注意事项:
# - TMEM 利用率会影响占用率
# - blockN=128 通常是最优的指令形状

if __name__ == "__main__":
    print("Benchmarking selected configs")
    print("=============================")
    M, N, K = 8192, 8192, 16 * 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    print("BLOCK_M BLOCK_N BLOCK_K num_warps time (ms) tflops/s")
    configs = []
    # 选择 BLOCK_M != BLOCK_N 会使一个矩阵load的延迟比另一个矩阵更长。
    # 如果分别对它们进行流水线优化没问题, 但在我们的 kernel 中它们是一起流水线的, 因此使BLOCK_M=BLOCK_N。
    for BLOCK_MN, BLOCK_K, num_warps in itertools.product([64, 128], [64, 128, 256], [4]):
        # SMEM 使用过多, 跳过此配置
        # 只计算了A和B的SMEM使用量, 因为累加器在循环结束后才分配SMEM, 此时 A/B 的 SMEM 已经可以被复用了。
        if (BLOCK_MN * BLOCK_K) * 2 * 2 // 1024 > 224:
            continue
        configs.append((BLOCK_MN, BLOCK_K, num_warps))

        fn = lambda: blocked_matmul(A, B, C, BLOCK_MN, BLOCK_MN, BLOCK_K, False, num_warps)
        # 增加 warmup 和 rep 以获得更稳定的结果。
        ms = triton.testing.do_bench(fn, warmup=100, rep=500)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"{BLOCK_MN:>7} {BLOCK_MN:>7} {BLOCK_K:>7} {num_warps:>9} {ms:>9.2f} {tflops_per_sec:>8.2f}")
    print()

# 性能结果如下:
# ```
# BLOCK_M BLOCK_N BLOCK_K num_warps time (ms) tflops/s
#      64      64      64         4      3.27   671.77
#      64      64     128         4      3.33   660.93
#      64      64     256         4      4.18   526.10
#     128     128      64         4      2.45   898.61
#     128     128     128         4      2.16  1019.46
#     128     128     256         4      3.91   563.13
# ```

# 我们在没有流水线优化的情况下达到了 1020 TFLOPS。
#
# %%
# tcgen05_mma的"隐式流水线"特性
# 由于 tcgen05_mma 是异步的, 我们可以将它与 TMA load 重叠, 从而减少 SM 的空闲时间。

# 虽然 tcgen05 指令是异步的, 但它们具有"隐式流水线"特性, 即在以下情况下,
# 硬件会自动保证指令按发出顺序执行, 无需手动插入同步:

# 1. 两个以上具有相同形状和累加器 dtype 的 tcgen05_mma 指令
#    例如: tcgen05_mma(a1, b1, acc) 后跟 tcgen05_mma(a2, b2, acc)
#    硬件保证第一个 MMA 完成后才开始第二个 MMA

# 2. tcgen05_mma 后跟 tcgen05_commit
#    例如: tcgen05_mma(...) 后跟 tcgen05_commit(bar)
#    硬件保证 commit 会等待 MMA 完成

# 3. tcgen05_copy(复制指令) 和 tcgen05_mma 之间
#    例如: tcgen05_copy(smem, tmem) 后跟 tcgen05_mma(tmem, ...)
#    硬件保证复制完成后才开始 MMA

# 这意味着: 连续发出多个相同类型的 tcgen05_mma 时, 不需要在它们之间显式同步。
# 结合 mbarrier 的完成机制, 可以精确跟踪 MMA 的完成状态, 从而构建细粒度的流水线调度。


# %%
# 接下来, 我们实现一个流水线版本的 blocked_matmul kernel。
# 【kernel 设计】
# 这个 kernel 同时计算输出矩阵 C 的两行块(上下相邻):

#   矩阵 A (M×K)         矩阵 B (K×N)       矩阵 C (M×N)
#   ┌─────────┐          ┌─────┐           ┌─────┐
#   │   U1    │ 128×128  │ B1  │ 128×128   │ UB  │ 128×128  <- 上方输出块
#   │ (upper) │    ×     │     │    =      │     │
#   ├─────────┤          └─────┘           ├─────┤
#   │   V1    │ 128×128                    │ VB  │ 128×128  <- 下方输出块
#   │ (lower) │                            │     │
#   └─────────┘                            └─────┘

#  【符号说明】
# - U = Upper(上方的 A 矩阵块, 位于 [off_m : off_m+128, :])
# - V = lower/Vertical(下方的 A 矩阵块, 位于 [off_m+128 : off_m+256, :])
# - B = B 矩阵块(位于 [:, off_n : off_n+128])
# - UB = 上方块与B的矩阵乘法
# - VB = 下方块与B的矩阵乘法
# - 数字表示第几轮的 K 维度迭代(例如 U1 = 第1轮的U块, 即 A[off_m:off_m+128, 0:128])

# 【BLOCK划分】
# - 沿 M 维度分区: 每个 program 处理 2*BLOCK_M 行(上下两个块)
# - 要求 BLOCK_M = BLOCK_N = 128, 并对所有输入进行双缓冲

# 【SMEM 使用量计算】(当 BLOCK_K = 128 时)
# - u_bufs: [2, 128, 128] fp16 = 2 × 128 × 128 × 2 bytes = 64 KB(双缓冲的上方A块)
# - v_bufs: [2, 128, 128] fp16 = 2 × 128 × 128 × 2 bytes = 64 KB(双缓冲的下方A块)
# - b_bufs: [2, 128, 128] fp16 = 2 × 128 × 128 × 2 bytes = 64 KB(双缓冲的B块)
# - 其他: mbarriers 等 ≈ 几百字节(可忽略)
# - 总计: 64 + 64 + 64 = 192 KB

# 【SMEM 容量限制】
# - Blackwell/Hopper (H100/B100): 每个 SM 的 SMEM 最大为 228 KB(可配置)
# - 192 KB 使用了约 84% 的 SMEM, 留有一定余量
# - 更大的 BLOCK_K (如256) 会超出 SMEM 限制


# 【调度顺序】(假设 K=512, BLOCK_K=128, 则 N=4 轮)
    # Load U1, B1, V1          # 预加载第1轮(k=0:128)
    # Load U2, B2, V2          # 预加载第2轮(k=128:256)
    
    # 循环开始(i=1):
    #     Wait U1,B1 → Compute UB1   # 计算 A[upper, 0:128] × B[0:128, :]
    #     Wait V1    → Compute VB1   # 计算 A[lower, 0:128] × B[0:128, :]
    #     Wait UB1   → Load U3, B3   # load第3轮(k=256:384)
    #     Wait VB1   → Load V3       
    
    # 循环继续(i=2):
    #     Wait U2,B2 → Compute UB2   # 计算 A[upper, 128:256] × B[128:256, :]
    #     Wait V2    → Compute VB2   # 计算 A[lower, 128:256] × B[128:256, :]
    #     Wait UB2   → Load U4, B4   # load第4轮(k=384:512)
    #     Wait VB2   → Load V4
    
    # 收尾(i=3,4):
    #     Compute UB3, VB3
    #     Compute UB4, VB4
    
    # Epilogue(收尾):
    #     Store UB result to C[upper, :]
    #     Store VB result to C[lower, :]

# 【流水线效率】
# - load与MMA比率=3:2 : 每轮迭代进行2次mma计算和3次tmaload
# - 使用双缓冲: 当前轮计算时, 下一轮数据正在load
# - U 和 B 共用一个 mbarrier(load_ub_bars)因为它们总是一起使用

@gluon.jit
def get_and_increment(counter):
    """
    return index, phase, counter.

    index: 0, 1, 0, 1, 0, 1... (每次递增切换)
    phase: 0, 0, 1, 1, 0, 0, 1, 1... (每两次切换一次)
    counter: 0, 1, 2, 3, 4, 5, 6... (每次递增)
    """
    return counter % 2, counter // 2 & 1, counter + 1

@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * (2 * BLOCK_M)
    off_n = pid_n * BLOCK_N

    # u (Upper): 上方的 tile, v(lower/Vertical):下方的 tile 
    u_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [2] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [2] + b_desc.block_type.shape, b_desc.layout)

    # 申请累加器tmem, 使用两个累加器！
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    ub_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    vb_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    # 申请并初始化mbarrier, u和b的load共用一个mbarrier。
    mma_ub_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mma_vb_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    load_ub_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    load_v_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(mma_ub_bars.index(i), count=1)
        mbarrier.init(mma_vb_bars.index(i), count=1)
        mbarrier.init(load_ub_bars.index(i), count=1)
        mbarrier.init(load_v_bars.index(i), count=1)

    load_counter = 0
    mma_counter = 0
    k = 0
    ub_acc = False
    vb_acc = False

    # U1, B1
    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_copy_global_to_shared(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
    # V1
    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
    k += BLOCK_K

    # U2, B2
    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_copy_global_to_shared(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
    # V2
    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
    k += BLOCK_K

    for _ in range(gl.cdiv(K, BLOCK_K) - 2):
        # 等待 Ui 和 Bi load完成, 执行 UBi
        mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
        mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
        tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=ub_acc)
        tcgen05_commit(mma_ub_bars.index(mma_index))
        ub_acc = True
        # 等待 Vi load完成, 执行 VBi
        mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
        tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=vb_acc)
        tcgen05_commit(mma_vb_bars.index(mma_index))
        vb_acc = True

        # 等待 UBi 完成, load U(i+2)
        load_index, load_phase, load_counter = get_and_increment(load_counter)
        mbarrier.wait(mma_ub_bars.index(mma_index), mma_phase)
        load_ub_bar = load_ub_bars.index(load_index)
        mbarrier.expect(load_ub_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))

        # 等待 VBi 完成, load B(i+2) 和 V(i+2)
        mbarrier.wait(mma_vb_bars.index(mma_index), mma_phase)
        tma.async_copy_global_to_shared(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))
        load_v_bar = load_v_bars.index(load_index)
        mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], load_v_bar, v_bufs.index(load_index))
        k += BLOCK_K

    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    ub_bar = mma_ub_bars.index(mma_index)
    vb_bar = mma_vb_bars.index(mma_index)
    epilogue_phase = mma_phase

    # 等待 U(N-1) 和 B(N-1) load完成, 执行 UB(N-1)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=True)
    # 等待 V(N-1) load完成, 执行 VB(N-1)
    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=True)

    # 等待 UN 和 BN load完成, 执行 UBN
    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(u_bufs.index(mma_index), b_bufs.index(mma_index), ub_tmem, use_acc=True)
    tcgen05_commit(ub_bar)
    # 等待 VN load完成, 执行 VBN
    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(v_bufs.index(mma_index), b_bufs.index(mma_index), vb_tmem, use_acc=True)
    tcgen05_commit(vb_bar)

    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, 
        (BLOCK_M, BLOCK_N), 
        tmem_layout, 
        num_warps, 
    )
    # 等待 UBN 完成, 执行 UB epilogue(收尾)
    mbarrier.wait(ub_bar, epilogue_phase)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    ub = ub_tmem.load(acc_reg_layout)
    c_smem.store(ub.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)

    # 等待 VBN 完成, 执行 VB epilogue(收尾)
    mbarrier.wait(vb_bar, epilogue_phase)
    vb = vb_tmem.load(acc_reg_layout)
    tma.store_wait(pendings=0)
    c_smem.store(vb.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m + BLOCK_M, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, 2 * BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_blocked_matmul_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


if __name__ == "__main__":
    print("Benchmarking pipelined matmul")
    print("=============================")
    print("BLOCK_M BLOCK_N BLOCK_K num_warps time (ms) tflops/s")
    # 由于 kernel 是针对特定超参数设计的, 我们只对这些配置进行基准测试。
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps in itertools.product([128], [128], [64, 128], [4, 8]):
        fn = lambda: blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        ms = triton.testing.do_bench(fn, warmup=200, rep=1000)
        flops = 2 * M * N * K
        tflops_per_sec = flops * 1e-12 / (ms * 1e-3)
        print(f"{BLOCK_M:>7} {BLOCK_N:>7} {BLOCK_K:>7} {num_warps:>9} {ms:>9.2f} {tflops_per_sec:>8.2f}")
    print()

# 性能结果:
# ```
# BLOCK_M BLOCK_N BLOCK_K num_warps time (ms) tflops/s
# 128     128      64         4      2.20  1000.51
# 128     128      64         8      1.97  1113.49
# 128     128     128         4      2.21  1040.27
# 128     128     128         8      2.17  1011.47
# ```

# 【性能分析】
# 相比于先前非流水线 kernel 的 1020 TFLOPS (BLOCK_K=128, 4 warps), 
# 流水线 kernel 获得了一定的加速, 最佳配置达到 1113 TFLOPS (BLOCK_K=64, 8 warps)。

# 【占用率分析】
# 占用率(Occupancy) = 实际运行的 block 数 / SM 能同时运行的最大 block 数

# GPU SM 时间线示意:

#   BLOCK_K=128 (低占用率, SMEM=192KB):
#   ╔═══════════════════════════════════╗
#   ║ Block 0: [计算]...[等待]...[计算]   ║  <- 只能运行 1 个 block
#   ╚═══════════════════════════════════╝
#   大量等待时间, SM 利用率低

#   BLOCK_K=64 (高占用率, SMEM=96KB):
#   ╔═══════════════════════════════════╗
#   ║ Block 0: [计算][等待][计算][等待]    ║  <- 可以运行 2 个 block
#   ║ Block 1: [等待][计算][等待][计算]    ║     交替执行!
#   ╚═══════════════════════════════════╝
#   当 Block 0 等待时, Block 1 在计算, SM 利用率高

# BLOCK_K=64 时获得 2 倍占用率.

# 更高的占用率意味着:
# ✓ 更好的延迟隐藏: 当一个 block 等待内存时, 另一个 block 可以计算
# ✓ 但计算量更小: BLOCK_K=64 每次迭代的计算量是 BLOCK_K=128 的一半
# ✗ 更多的迭代次数: K=16384 时, 64 需要 256 次迭代, 128 只需 128 次

# 结果: BLOCK_K=64 凭借更好的占用率, 即使迭代次数翻倍, 仍然获得更好的性能。
# 这说明当前 kernel 受内存延迟限制(memory-bound), 而非计算限制(compute-bound)。

# 具体数字 (假设 K=16384):
#   BLOCK_K=128: 128 次迭代 × 较长的 MMA 时间 × 低占用率(1 block/SM) = 2.16 ms
#   BLOCK_K=64:  256 次迭代 × 较短的 MMA 时间 × 高占用率(2 block/SM) = 1.97 ms ✓
#   虽然迭代次数翻倍, 但更高的占用率带来了 ~9% 的性能提升

# 【num_warps=8 的作用】
# 观察数据: BLOCK_K=64 时, num_warps=8 比 4 快了 10% (1113 vs 1000 TFLOPS)
#          BLOCK_K=128 时, num_warps=8 几乎没有提升 (1011 vs 1040 TFLOPS)

# Kernel 执行时间分解:

#   BLOCK_K=128 (迭代少, MMA 重):
#   ┌──────────────────────────┬────┐
#   │   主循环 (MMA + Load)     │Epi │  Epilogue 占比 ≈ 5%
#   └──────────────────────────┴────┘
#   4 warps 足够, 增加到 8 warps 收益不大

#   BLOCK_K=64 (迭代多, MMA 轻):
#   ┌──────────────┬─────────┐
#   │  主循环       │Epilogue │  Epilogue 占比 ≈ 15%
#   └──────────────┴─────────┘
#   Epilogue 更长, 8 warps 能显著加速

# epilogue 更长时 num_warps=8 更好的原因:
#   - epilogue 包含: 从 TMEM load结果 → 寄存器 layout 转换 → 存储到 SMEM → TMA 写回 GMEM
#   - BLOCK_K=64 时: 
#     - 迭代次数更多 (256 vs 128), 每次 MMA 更快完成
#     - epilogue 需要处理 2 个块 (UB + VB) 的数据 --> 需要更多的寄存器 shuffle
#     - epilogue 占总时间比例更大 (~15% vs ~5%)
#   - 更多 warp (8 vs 4) 的好处:
#     ✓ 并行执行 epilogue 中的 layout 转换 (需要大量寄存器 shuffle)
#     ✓ 并行load两个累加器 (ub_tmem 和 vb_tmem)
#     ✓ 充分利用硬件资源, 减少 epilogue 成为瓶颈
#   - BLOCK_K=128 时: 
#     - 每次 MMA 耗时更长, 主循环占主导
#     - epilogue 占比小, 额外的 warp 作用有限
#     - 反而可能增加调度开销

# 【改进方向】
# 在引入 warp specialization 后, 我们会看到它是精细流水线优化 kernel 的更有效方式:
#   - 可以让不同 warp 专门负责load、计算、存储等不同任务
#   - 进一步提高流水线效率, 减少等待时间
