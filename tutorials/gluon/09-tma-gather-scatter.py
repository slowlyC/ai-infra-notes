"""
原生 TMA Gather 和 Scatter 操作
================================

本教程介绍如何使用 Blackwell GPU 上原生的异步 TMA gather 和 scatter 操作。
这两个操作分别通过 `gl.nvidia.blackwell.tma.async_gather` 和`gl.nvidia.blackwell.tma.async_scatter` 函数实现。

----------------------------------------------------
首先, 用一个简单的 PyTorch 示例来理解 Gather 和 Scatter: 

```python
import torch

# 原始张量 (4行 x 3列)
tensor = torch.tensor([[ 0,  1,  2,  3],   # 第0行
                       [ 4,  5,  6,  7],   # 第1行
                       [ 8,  9, 10, 11],   # 第2行

# Gather: 按行索引"收集"数据
row_indices = [2, 0]   # 要收集的行: 第2、0
col_start = 1          # 从第1列开始
col_offset = 2         # 取2列, col_end = col_start + col_offset = 3, 左闭右开区间

result = tensor[row_indices][:, col_start : col_start + col_offset]
print(result)
# tensor([[ 9, 10],   <- 第2行的 [1:3]
#         [ 1,  2],   <- 第0行的 [1:3]

# Scatter: 按行索引"散布"数据
output = torch.zeros_like(tensor)
data = torch.tensor([[1, 2],
                     [3, 4],

output[row_indices, col_start : col_start + col_offset] = data
print(output)
# tensor([[0, 3, 4, 0],   <- 第0行写入 [3,4]
#         [0, 0, 0, 0],
#         [0, 1, 2, 0],   <- 第2行写入 [1,2]
```

简单来说: 
- **Gather**: 从不同行"收集"连续列 → `data = tensor[行索引][:, 列范围]`
- **Scatter**: 向不同行"散布"连续列 → `tensor[行索引, 列范围] = data`

----------------------------------------------------

TMA Gather/Scatter 是硬件原生的异步 DMA 操作, 比 `gl.load`/`gl.store` 更快。

**限制**: 仅支持 2D tensor descriptor, 块形状必须是 `[1, BLOCK_Y]`。

```python
# Gather - 按行索引批量读取: 
out = tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y]  # out.shape: [x_offsets.shape[0], BLOCK_Y]

# Scatter - 按行索引批量写入: 
tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y] = src  # src.shape:[x_offsets.shape[0], BLOCK_Y]
```

简单说: `x_offsets` 指定要操作哪些行, `y_offset` 指定列的起始位置, `BLOCK_Y` 指定操作多少列。

与 `async_copy_global_to_shared` 和 `async_copy_shared_to_global` 一样, 
`async_gather` 和 `async_scatter` 通过 async proxy 访问共享内存, 所以需要在适当的地方插入 fence。
"""

import sys
import pytest
import torch
import triton
import importlib
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton._C.libtriton import ir, gluon_ir

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (tma, mbarrier, fence_async_shared)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")

# 复用之前教程的工具函数
t7 = importlib.import_module("07-persistence")

# %%
# `async_gather` 和 `async_scatter` 对行偏移量 `x_offsets` 的布局约束
# ====================================
#
# **约束来源**: 底层 PTX 指令 gather4/scatter4 是 warp 级操作, 每次处理 4 行，
# 其对应ptx中的 `cp.async.bulk.tensor.2d.tile::scatter4.global.shared::cta.bulk_group` 指令。
#
# **约束内容**(两条规则): 
# 1. 每 4 个连续元素 → 必须在同一线程的连续寄存器中
# 2. 同一 warp 的所有线程 → 必须持有相同的数据(广播)
#
# **推荐布局(高效)**: 用 SliceLayout 将工作分散到所有 warp
#
# ```python
# # 方式1: 从 dim=0 切片
# gl.SliceLayout(
#     dim=0,
#     parent=gl.BlockedLayout(
#         size_per_thread=[1, 4],         # 每线程持有 4 个连续元素
#         threads_per_warp=[32, 1],       # warp 内线程只分 dim=0
#         warps_per_cta=[1, num_warps],   # 多个 warp 分担 dim=1
#         order=[1, 0],
#     ),
# )
#
# # 方式2: 等价的转置写法, 从 dim=1 切片
# gl.SliceLayout(
#     dim=1,
#     parent=gl.BlockedLayout(
#         size_per_thread=[4, 1],
#         threads_per_warp=[1, 32],
#         warps_per_cta=[num_warps, 1],
#         order=[0, 1],
#     ),
# )
# ```
## 回顾 `02-layouts` 教程, parent `BlockedLayout` 会沿 dim=1 将 4 个连续元素分成一组, 映射到同一线程的 4 个连续寄存器中。
# 当我们沿 dim=0 取 `SliceLayout` 时(删除第0维, 保留第1维), warp 内所有线程就会映射到相同的 4 个连续元素(满足规则2)。
#
# **简单布局(低效但有效)**: 所有元素广播到所有线程
#
# ```python
# gl.BlockedLayout(
#     size_per_thread=[BLOCK_X],           # 每线程持有全部元素
#     threads_per_warp=[num_threads_per_warp],  # num_threads_per_warp=32
#     warps_per_cta=[num_warps],
#     order=[0],
# )
# ```
#
# 这个布局有效是因为所有元素都连续映射到所有线程的寄存器中, 但效率较低;因为所有 warp 持有相同数据, 编译器只让 warp 0 执行所有指令。
# 例如 `BLOCK_X=256`: 
# - 简单布局(BlockedLayout): warp 0 执行 `256//4 = 64` 条 gather4, 其他 warp 空闲
# - 推荐布局(SliceLayout): 会将工作分散到所有 warp, 假设有 4 个 warp, 每个 warp 执行 `256//4//4 = 16` 条 gather4
#
# **linear layout 下的有效条件**: 
# - 前 2 个寄存器基(register bases)必须是 [1] 和 [2](保证 4 元素连续)
# - 所有 lane 基(lane bases)必须是 [0](保证 warp 内广播)

# %%
# 让我们写一个工具函数, 将任意布局转换为线性布局来说明这个概念。

def to_linear_layout(layout, shape):
    context = ir.context()
    ir.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context)
    return builder.to_linear_layout(layout._to_ir(builder), shape)


if __name__ == "__main__":
    num_threads_per_warp = 32
    num_warps = 4
    BLOCK_X = 256

    layout = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[num_threads_per_warp, 1],
            warps_per_cta=[1, num_warps],
            order=[1, 0],
        ),
    )
    # DistributedLinearLayout(
    #     reg_bases=[[1], [2], [16], [32], [64], [128]],
    #     lane_bases=[[0], [0], [0], [0], [0]],
    #     warp_bases=[[4], [8]],
    #     block_bases=[],
    #     shape=[256]
    # )
    print(to_linear_layout(layout, [256]))

    layout = gl.BlockedLayout(
        size_per_thread=[BLOCK_X],
        threads_per_warp=[num_threads_per_warp],
        warps_per_cta=[num_warps],
        order=[0],
    )
    # DistributedLinearLayout(
    #     reg_bases=[[1], [2], [4], [8], [16], [32], [64], [128]],
    #     lane_bases=[[0], [0], [0], [0], [0]],
    #     warp_bases=[[0], [0]],
    #     block_bases=[],
    #     shape=[256]
    # )
    print(to_linear_layout(layout, [256]))

    # 注意上面两个布局中, 前两个寄存器基确实是 [1] 和 [2], 所有 lane 基都是 [0]。
    # 区别在于第二个布局的 warp 基都是 [0], 这会导致 `async_gather` 和
    # `async_scatter` 生成低效的代码。

    # 这是一个无效布局的例子: 
    layout = gl.BlockedLayout(
        size_per_thread=[4],
        threads_per_warp=[num_threads_per_warp],
        warps_per_cta=[num_warps],
        order=[0],
    )
    # DistributedLinearLayout(
    #     reg_bases=[[1], [2]],
    #     lane_bases=[[4], [8], [16], [32], [64]],
    #     warp_bases=[[128], [0]],
    #     block_bases=[],
    #     shape=[256]
    # )
    print(to_linear_layout(layout, [256]))
    # 这个布局是无效的, 因为 lane 基不全是 [0]。

# %%
# 下面我们来编写一个简单的 kernel, 说明如何使用 `async_gather` 和 `async_scatter`。
# 注意这两个操作有一些额外的约束: tensor descriptor 必须是 2D 的, 块形状为 `[1, BLOCK_Y]`。
# 此外还需要注意: 
#
# - 行偏移张量必须至少有 8 个元素, 即 gather 或 scatter 必须处理至少 8 行。
#
# - 根据 dtype 有最小列数限制: `BLOCK_Y >= (32 // tensor_desc.dtype.primitive_bitwidth) * 8`。
#   例如, `float16` 的 tensor descriptor 必须满足 `BLOCK_Y >= 16`。
#
# - `y_offset` 必须对齐到 16 字节, 即 `y_offset % (16 // (tensor_desc.dtype.primitive_bitwidth // 8)) == 0`。
#   例如对于 `float16`, `y_offset` 必须是 8 的倍数。这在运行时由硬件检查, 如果没有对齐到 16 字节, CUDA 驱动会报非法指令错误。
#
# - `x_offsets` 的元素可以越界, 这种情况下 `async_gather` 加载的行会全是零, `async_scatter` 的存储会被忽略。
#
# - `y_offset` 也可以越界。`y_offset:y_offset + BLOCK_Y` 中越界的行元素, 
#   `async_gather` 会加载为零, `async_scatter` 存储时会被忽略。
#
# - 只有 `async_gather` 的 `x_offsets` 元素和 `y_offset` 可以为负数。
#   如果 `async_scatter` 收到负的行或列偏移, CUDA 驱动会报非法指令错误。


# 这个 kernel 计算 `out = tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y]`。
@gluon.jit
def async_gather_kernel(out_ptr, out_stride_x, out_stride_y, tensor_desc, x_offsets_ptr, y_offset,
                        BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    # 使用 coalesced 布局加载 x_offsets 偏移, 以实现高效的向量化加载。
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))

    # 将 x_offsets 偏移的布局转换为满足 `async_gather` 约束的 slice 布局。
    offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    # `async_gather` 从 tensor descriptor 加载行并写入共享内存。
    # 共享内存描述符的布局必须与 tensor descriptor 的共享内存布局匹配。
    smem_dest = gl.allocate_shared_memory(tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout)

    # `async_gather` 是一个异步操作, 使用 mbarrier 来跟踪完成状态。
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    # 在 mbarrier 上调用 `mbarrier.expect`, 传入要加载的字节数。
    mbarrier.expect(bar, BLOCK_X * tensor_desc.block_type.nbytes)

    # 发起 async gather 并等待完成。
    tma.async_gather(tensor_desc, x_offsets, y_offset, barrier=bar, result=smem_dest)  # gmem -> smem
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # 使用 coalesced 布局写入结果。
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    out = smem_dest.load(coalesced_2d_layout)  # smem -> reg

    indices_x = gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None] * out_stride_x
    indices_y = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :] * out_stride_y
    gl.store(out_ptr + indices_x + indices_y, out)  # reg -> gmem


def async_gather(input, x_offsets, y_offset, BLOCK_X, BLOCK_Y):
    gl_dtype = getattr(gl, str(input.dtype).split('.')[1])
    # 选择共享内存布局时, 使用共享内存描述符的维度是Tensor大小 [BLOCK_X, BLOCK_Y]。
    # 但 tensor descriptor 的块形状必须是 [1, BLOCK_Y] 才能用于 async gather。
    layout = gl.NVMMASharedLayout.get_default_for([BLOCK_X, BLOCK_Y], gl_dtype)
    tensor_desc = TensorDescriptor.from_tensor(input, [1, BLOCK_Y], layout)
    out = torch.empty((BLOCK_X, BLOCK_Y), dtype=input.dtype, device="cuda")
    async_gather_kernel[(1, )](out, *out.stride(), tensor_desc, x_offsets, y_offset, BLOCK_X)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("BLOCK_X", [8, 128])
@pytest.mark.parametrize("BLOCK_Y", [16, 128])
@pytest.mark.parametrize("y_offset", [-16, 0, 48, 1000])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_async_gather(BLOCK_X, BLOCK_Y, y_offset, dtype, X_MAX=1024, Y_MAX=1024):
    torch.manual_seed(0)

    input = torch.randn((X_MAX, Y_MAX), dtype=dtype, device="cuda")
    # torch.linspace(start, end, steps): 线性等分, 生成从 start 到 end 的 steps 个等间距的数。
    # torch.linspace(-1024, 2048, 8): tensor([-1024.0, -585.1, -146.3,  292.6,  731.4, 1170.3, 1609.1, 2048.0])
    # 行偏移: 从 -1024 到 2048, 用于测试 masked load 的行为。
    x_offsets = torch.linspace(-X_MAX, 2 * X_MAX, BLOCK_X, dtype=torch.int32, device="cuda")
    # torch.randperm(n): 生成 0 到 n-1 的随机排列(shuffle)。 # torch.randperm(8):tensor([3, 1, 7, 0, 5, 2, 6, 4])
    # tensor([ 292.5714, -1024.0000, -146.2857, 2048.0000, 1170.2856, -585.1428, 1609.1428, 731.4286])
    x_offsets = x_offsets[torch.randperm(BLOCK_X, device="cuda")]

    out = async_gather(input, x_offsets, y_offset, BLOCK_X, BLOCK_Y)

    # 屏蔽越界和负数行偏移。
    x_offsets = torch.where(x_offsets >= X_MAX, -1, x_offsets) # 越界索引标记为 -1
    mask = (x_offsets >= 0).unsqueeze(1)  # [BLOCK_X, 1]

    # 通过填充零来屏蔽越界和负数列偏移。
    y_lo, y_hi = max(0, y_offset), min(y_offset + BLOCK_Y, Y_MAX)  # [0, 112]
    ref = input[x_offsets, y_lo:y_hi] * mask  # [BLOCK_X, 112], 读取有效的 112 列, 乘 mask 将越界的行元素置零
    lo_zeros = torch.zeros(BLOCK_X, y_lo - y_offset, dtype=dtype, device="cuda") # shape: (BLOCK_X, 16)  ← 左侧 16 列填零
    hi_zeros = torch.zeros(BLOCK_X, y_offset + BLOCK_Y - y_hi, dtype=dtype, device="cuda") # shape: (BLOCK_X, 0) ← 右侧 0 列
    ref = torch.cat((lo_zeros, ref, hi_zeros), dim=1)  # [BLOCK_X, BLOCK_Y], (16 列零) + (112 列数据) + (0 列) = 128 列

    torch.testing.assert_close(out, ref, atol=0, rtol=0)


# %%
# 接下来我们看下没有遵循约束会发生什么。
# 对于 `async_gather` 和 `async_scatter`, 如果 `y_offset` 没有对齐到 16 字节, 
# 或者 `async_scatter` 使用了负的行或列偏移, CUDA 驱动会报非法指令错误。

# python 09-tma-gather-scatter.py test_illegal_gather
if __name__ == "__main__":
    # 注意: 非法指令错误会破坏当前 Python 进程中的 CUDA context, 导致退出程序。
    if len(sys.argv) > 1 and sys.argv[1] == "test_illegal_gather":
        try:
            # y_offset=2 对于 bfloat16 来说没有对齐到 16 字节
            test_async_gather(BLOCK_X=128, BLOCK_Y=128, y_offset=2, dtype=torch.bfloat16)
        except RuntimeError as e:
            assert "an illegal instruction was encountered" in str(e)
            raise

# 非法指令错误调试起来比较困难。这类错误通常是因为执行的指令不满足某些运行时的常量约束。
# 要找出是哪条指令导致的错误, 可以在 `cuda-gdb` 调试器中运行程序。例如: 
#
# ```bash
# cuda-gdb --args python 09-tma-gather-scatter.py test_illegal_gather
# ```
#
# 回车后输入 `r` 运行程序, 调试器会在触发非法指令错误的指令处中断: 
#
# ```
# CUDA Exception: Warp Illegal Instruction
# The exception was triggered at PC 0x628fbe590  async_gather_kernel  (09-tma-gather-scatter.py:245)
#
# Thread 1 "python" received signal CUDA_EXCEPTION_4, Warp Illegal Instruction.
# [Switching focus to CUDA kernel 0, grid 9, block (0,0,0), thread (96,0,0), device 0, sm 148, warp 0, lane 0]
# 0x0000000628fbe700 in async_gather_kernel<<<(1,1,1),(128,1,1)>>> () at /root/code/triton/python/tutorials/gluon/09-tma-gather-scatter.py:245
# 245         tma.async_gather(tensor_desc, x_offsets, y_offset, barrier=bar, result=smem_dest)
# ```

# %%
# 接下来我们看下如何使用 `async_scatter`。
# 这个 kernel 计算 `tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y] = src`。

@gluon.jit
def async_scatter_kernel(tensor_desc, x_offsets_ptr, y_offset, src_ptr, src_stride_x, src_stride_y,
                         BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    # 使用 coalesced 布局加载源数据, 以实现高效的向量化加载。
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    indices_x = gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None] * src_stride_x
    indices_y = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :] * src_stride_y
    src = gl.load(src_ptr + indices_x + indices_y)  # gmem -> reg

    # 使用 coalesced 布局加载偏移, 以实现高效的向量化加载。
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))

    # 将偏移的布局转换为满足 `async_scatter` 约束的 slice 布局。
    offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    # `async_scatter` 从共享内存将行存储到 tensor descriptor。
    smem_src = gl.allocate_shared_memory(tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout)
    smem_src.store(src)  # reg -> smem
    # 共享内存存储和 async scatter 之间需要 async fence。
    # 回顾 `04-tma`, 当使用不同的 proxy 访问共享内存时需要 fence
    # (store 使用 generic proxy, `async_scatter` 使用 async proxy)。
    fence_async_shared()
    tma.async_scatter(tensor_desc, x_offsets, y_offset, smem_src)  # smem -> gmem
    # 使用 `store_wait` 等待 async scatter 完成。
    tma.store_wait(0)


def async_scatter(input, x_offsets, y_offset, src, BLOCK_X, BLOCK_Y):
    gl_dtype = getattr(gl, str(input.dtype).split('.')[1])
    # 选择共享内存布局时, 使用共享内存描述符的维度 [BLOCK_X, BLOCK_Y]。
    # 但 tensor descriptor 的块形状仍必须是 [1, BLOCK_Y] 才能用于 async scatter。
    layout = gl.NVMMASharedLayout.get_default_for([BLOCK_X, BLOCK_Y], gl_dtype)
    tensor_desc = TensorDescriptor.from_tensor(input, [1, BLOCK_Y], layout)
    async_scatter_kernel[(1, )](tensor_desc, x_offsets, y_offset, src, *src.stride(), BLOCK_X)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("BLOCK_X", [8, 128])
@pytest.mark.parametrize("BLOCK_Y", [16, 128])
@pytest.mark.parametrize("y_offset", [0, 48, 1000])  # async_scatter的 x_offsets 和 y_offset 不能为负数
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_async_scatter(BLOCK_X, BLOCK_Y, y_offset, dtype, X_MAX=1024, Y_MAX=1024):
    torch.manual_seed(0)

    input = torch.randn((X_MAX, Y_MAX), dtype=dtype, device="cuda")
    input_ref = input.clone()

    # 行偏移从 0 到越界, 用于测试 masked store 的行为。
    # torch.linspace(start, end, steps): 线性等分, 生成从 start 到 end 的 steps 个等间距的数。
    # torch.linspace(0, 2048, 8): tensor([0.0, 292.5, 585.1, 877.7, 1170.2, 1462.8, 1536.0, 1755.4, 2048.0])
    x_offsets = torch.linspace(0, 2 * X_MAX, BLOCK_X, dtype=torch.int32, device="cuda")
    # 随机打乱行偏移。
    # tensor([2048.0, 1536.0, 1755.4, 0.0, 292.5, 585.1, 877.7, 1170.2, 1462.8])
    x_offsets = x_offsets[torch.randperm(BLOCK_X, device="cuda")]

    src = torch.randn((BLOCK_X, BLOCK_Y), dtype=dtype, device="cuda")
    async_scatter(input, x_offsets, y_offset, src, BLOCK_X, BLOCK_Y)

    # 屏蔽越界的行偏移。
    mask = x_offsets < X_MAX
    x_offsets = x_offsets[mask]
    src = src[mask]

    # 屏蔽越界的列偏移。
    y_hi = min(y_offset + BLOCK_Y, Y_MAX)

    input_ref[x_offsets, y_offset:y_hi] = src[:, :y_hi - y_offset]
    torch.testing.assert_close(input, input_ref, atol=0, rtol=0)


# %%
# `async_gather` 和 `async_scatter` 也可以像 `async_copy_global_to_shared` 和`async_copy_shared_to_global` 一样流水线化。
# 为了演示这一点, 我们将编写一个 matmul kernel, 在 M 维度上融合 gather 和 scatter: 
# `out[out_scatter_indx, :] = X[X_gather_indx, :] @ W`。
#
# 回顾 `06-tcgen05-mma`, 我们演示了如何用 `tcgen05_mma` 编写 matmul kernel。
# 本示例将 TMA 加载(包括 `async_gather`)与 `tcgen05_mma` 进行流水线化, 并在持久化外层循环中对 `async_scatter` 进行流水线化。
#
# 在融合了 gather 和 scatter 的分块 matmul kernel 中, 对于每个输出 tile, 
# 我们通过 `gl.load` 加载 X tensor tile 的 M 维度偏移和 W tensor tile 的 N 维度偏移, 并提前调度以考虑全局加载的延迟。

@gluon.jit
def issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs,
                BLOCK_M: gl.constexpr, num_buffers: gl.constexpr, pred=True):
    # 加载 X tensor tile 的 M 维度偏移。我们预期加载足够小(不超过 128 个元素), 
    # 不需要使用 coalesced 布局。直接加载到 `async_gather` 需要的布局以避免布局转换。
    gather_indx_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    offs_x_m = gl.load(X_gather_indx_ptr + off_m + gl.arange(0, BLOCK_M, gather_indx_layout))

    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)

    # W tensor tile 使用常规的 `async_copy_global_to_shared` 加载。
    mbarrier.expect(bar, W_desc.block_type.nbytes + BLOCK_M * X_desc.block_type.nbytes)
    tma.async_gather(X_desc, offs_x_m, k, bar, x_bufs.index(index), pred)
    tma.async_copy_global_to_shared(W_desc, [k, off_n], bar, w_bufs.index(index), pred)
    return producer


@gluon.jit
def issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    b_index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(x_bufs.index(index), w_bufs.index(b_index))
    return consumer, mma


@gluon.jit
def matmul_fused_gather_scatter_kernel(X_desc, W_desc, out_desc, X_gather_indx_ptr, out_scatter_indx_ptr,
                                       BLOCK_M: gl.constexpr, SchedulerImpl: gl.constexpr, num_buffers: gl.constexpr):
    BLOCK_N: gl.constexpr = W_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = W_desc.block_type.shape[0]
    dtype: gl.constexpr = X_desc.dtype
    M = X_desc.shape[0]
    N = W_desc.shape[1]
    K = X_desc.shape[1]

    # 为输入 tile 分配共享内存。
    x_bufs = gl.allocate_shared_memory(dtype, [num_buffers, BLOCK_M, BLOCK_K], X_desc.layout)
    w_bufs = gl.allocate_shared_memory(dtype, [num_buffers, BLOCK_K, BLOCK_N], W_desc.layout)

    # 为输出 tile 分配共享内存。
    out_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], out_desc.layout)

    # 初始化用于多缓冲加载的 barrier。
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = t7.MMAv5.initialize(dtype, BLOCK_M, BLOCK_N, gl.num_warps())
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # 剥离的内层循环 prologue。
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, ki, bars, x_bufs, w_bufs,
                               BLOCK_M, num_buffers)
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs, BLOCK_M,
                           num_buffers)

    for _ in range(num_tiles):
        consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs,
                                   BLOCK_M, num_buffers)
            consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        # 加载输出 tile 的 M 维度偏移。我们预期加载足够小(不超过 128 个元素), 
        # 不需要使用 coalesced 布局。直接加载到 `async_scatter` 需要的布局以避免布局转换。
        scatter_indx_layout: gl.constexpr = gl.SliceLayout(
            0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
        out_offs_m = gl.load(out_scatter_indx_ptr + epilogue_off_m + gl.arange(0, BLOCK_M, scatter_indx_layout))

        # 剥离下一个 prologue 并与流水线 drain 循环融合。
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        # 使用谓词而非条件语句来控制剥离的 prologue。
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, ki, bars, x_bufs, w_bufs,
                                   BLOCK_M, num_buffers, pred)
            consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs,
                               BLOCK_M, num_buffers)

        mma = mma.wait_num_outstanding(0)
        out, mma = mma.take_result()
        out = out.to(dtype)
        # 通过等待上一个存储完成来对 async scatter 进行流水线化。
        tma.store_wait(pendings=0)
        out_smem.store(out)
        fence_async_shared()
        tma.async_scatter(out_desc, out_offs_m, epilogue_off_n, out_smem)
    # 等待最后一个 async scatter 完成。
    tma.store_wait(pendings=0)


# 我们在这里选择合理的默认块大小和加载缓冲区数量。
# 进一步地性能调优留给读者作为练习, 因为本教程的主要目标是演示 async gather 和 async scatter 的用法。
#
# 实现带融合 gather 和 scatter 的 matmul kernel 的唯一替代方式是使用 async_copy(回顾 `03-async-copy`)或 `gl.load` 从全局内存加载, 
# `gl.store` 在 epilogue 中写入输出张量。虽然这些指令提供更灵活的索引, 但比 TMA 和 async gather/scatter 慢得多。
#
# 另外值得注意的是: 也可以在 warp-specialized kernel 中使用 async gather 和 async scatter。
# 只需记住因为行偏移是一个张量, 你可能希望给 load 和 epilogue 分区分配多于 1 个 warp 以提高指令发射吞吐量, 特别是对于处于关键路径上的 load 操作。

def matmul_fused_gather_scatter(X, X_gather_indx, W, out_scatter_indx, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
                                GROUP_SIZE_M=8, num_buffers=3):
    M = X.shape[0]
    N = W.shape[1]
    out = torch.empty((M, N), dtype=X.dtype, device="cuda")

    # 将 torch dtype 转换为 gluon dtype。
    dtype = getattr(gl, str(X.dtype).split('.')[1])
    # 设置输入和输出的 descriptor。
    X_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], dtype)
    W_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], dtype)
    out_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], dtype)

    X_desc = TensorDescriptor.from_tensor(X, [1, BLOCK_K], X_desc_layout)
    W_desc = TensorDescriptor.from_tensor(W, [BLOCK_K, BLOCK_N], W_desc_layout)
    out_desc = TensorDescriptor.from_tensor(out, [1, BLOCK_N], out_desc_layout)

    # 持久化 kernel 的 grid。
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)
    matmul_fused_gather_scatter_kernel[grid](X_desc, W_desc, out_desc, X_gather_indx, out_scatter_indx, BLOCK_M,
                                             SchedulerImpl, num_buffers)
    return out


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 2048), (4096, 4096, 4096)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N", [(128, 128), (128, 64)])
@pytest.mark.parametrize("BLOCK_K, num_buffers", [(128, 2), (64, 3)])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_matmul_fused_gather_scatter(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers):
    torch.manual_seed(0)

    # 用随机排列测试是为了验证 kernel 在任意非连续访问模式下都能正确工作, 也模拟了表查找、稀疏等场景。
    # 随机化 gather 索引。决定从 X 矩阵的哪些行读取数据。
    X_gather_indx = torch.arange(0, M, dtype=torch.int32, device="cuda")
    shfl = torch.randperm(M, device="cuda")
    X_gather_indx = X_gather_indx[shfl]

    # 随机化 scatter 索引。决定把计算结果写到输出矩阵的哪些行。
    out_scatter_indx = torch.arange(0, M, dtype=torch.int32, device="cuda")
    shfl = torch.randperm(M, device="cuda")
    out_scatter_indx = out_scatter_indx[shfl]

    X = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    out = matmul_fused_gather_scatter(X, X_gather_indx, W, out_scatter_indx, BLOCK_M, BLOCK_N, BLOCK_K,
                                      num_buffers=num_buffers)

    out_ref = torch.empty_like(out)
    out_ref[out_scatter_indx, :] = X[X_gather_indx, :] @ W
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-3)

# %%
# 下面让我们对比下 gather/scatter 版本和 07-persistence.py 中的持久化流水线矩阵乘法的性能。
#
# 如果使用顺序索引(不随机化): 
#   X_gather_indx = torch.arange(0, M)
#   out_scatter_indx = torch.arange(0, M)
# 
# 那么计算变成: out[0:M, :] = X[0:M, :] @ W, 等价于普通矩阵乘法 out = X @ W

if __name__ == "__main__":
    persistent_matmul_pipelined = t7.persistent_matmul_pipelined
    GROUP_SIZE_M = 8
    
    M, N, K = 4096, 4096, 4096
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    num_warps = 4
    num_buffers = 4

    X = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    
    # 顺序索引(等价于普通矩阵乘法)
    X_gather_indx_sequential = torch.arange(0, M, dtype=torch.int32, device="cuda")
    out_scatter_indx_sequential = torch.arange(0, M, dtype=torch.int32, device="cuda")
    
    # 随机索引
    X_gather_indx_random = X_gather_indx_sequential[torch.randperm(M, device="cuda")]
    out_scatter_indx_random = out_scatter_indx_sequential[torch.randperm(M, device="cuda")]

    print("=================================")
    print("Benchmarking gather/scatter matmul")
    print(f"args: M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, dtype=bfloat16")
    print(f"{'Kernel 类型':<40} {'时间 (ms)':>12} {'TFLOPS':>12}")
    print("-" * 70)

    flops = 2 * M * N * K

    # 1. 持久化流水线 Matmul (07-persistence.py) - 使用相同的调度器
    C_pipelined = torch.empty(M, N, device="cuda", dtype=torch.float16)
    X_fp16 = X.to(torch.float16)
    W_fp16 = W.to(torch.float16)
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)
    fn_pipelined = lambda: persistent_matmul_pipelined(X_fp16, W_fp16, C_pipelined, BLOCK_M, BLOCK_N, BLOCK_K, 
                                                        num_buffers, num_warps, SchedulerImpl)
    ms_pipelined = triton.testing.do_bench_cudagraph(fn_pipelined)
    tflops_pipelined = flops * 1e-12 / (ms_pipelined * 1e-3)
    print(f"{'持久化流水线 Matmul (07-persistence.py)':<40} {ms_pipelined:>12.3f} {tflops_pipelined:>12.2f}")

    # 2. Gather/Scatter Matmul - 顺序索引
    fn_gs_seq = lambda: matmul_fused_gather_scatter(X, X_gather_indx_sequential, W, out_scatter_indx_sequential, 
                                                     BLOCK_M, BLOCK_N, BLOCK_K, num_buffers=num_buffers)
    ms_gs_seq = triton.testing.do_bench_cudagraph(fn_gs_seq)
    tflops_gs_seq = flops * 1e-12 / (ms_gs_seq * 1e-3)
    print(f"{'Gather/Scatter Matmul (顺序索引)':<40} {ms_gs_seq:>12.3f} {tflops_gs_seq:>12.2f}")

    # 3. Gather/Scatter Matmul - 随机索引
    fn_gs_rand = lambda: matmul_fused_gather_scatter(X, X_gather_indx_random, W, out_scatter_indx_random,
                                                      BLOCK_M, BLOCK_N, BLOCK_K, num_buffers=num_buffers)
    ms_gs_rand = triton.testing.do_bench_cudagraph(fn_gs_rand)
    tflops_gs_rand = flops * 1e-12 / (ms_gs_rand * 1e-3)
    print(f"{'Gather/Scatter Matmul (随机索引)':<40} {ms_gs_rand:>12.3f} {tflops_gs_rand:>12.2f}")

# 性能结果如下(两者都使用持久化 kernel + GroupedPersistentTileScheduler): 
# =================================
# Benchmarking gather/scatter matmul
# args: M=4096, N=4096, K=4096, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, dtype=bfloat16
# Kernel 类型                                     时间 (ms)       TFLOPS
# ----------------------------------------------------------------------
# 持久化流水线 Matmul (07-persistence.py)               0.138       994.06
# Gather/Scatter Matmul (顺序索引)                    0.194       710.00
# Gather/Scatter Matmul (随机索引)                    0.195       706.34
# 
# 可以看出, Gather/Scatter 有额外开销(索引加载、scatter 写入), 性能略低于普通 TMA


# %%
# 本教程的要点是理解如何使用 `async_gather` 和 `async_scatter`。
# 两个指令在块 DMA(如 `async_copy_global_to_shared` 和 `async_copy_shared_to_global`)和常规全局内存 `gl.load` 和 `gl.store` 之间提供了一个中间方案, 
# 允许独立行索引, 同时保持 TMA 的性能。
#
# 请记住以下几点: 
# - 当可以使用时, `async_gather` 和 `async_scatter` 通常比 `gl.load` 和 `gl.store` 更快, 但也不总是如此。另外, TMA 指令会占用共享内存。
#   
# - 有时使用 `async_gather` 或 `async_scatter` 代替块 DMA 指令(如 `async_copy_global_to_shared` 和 `async_copy_shared_to_global`)
#   实际上更快, 但这种情况很少见。
#
# 总的来说, 在编写 kernel 时应该考虑这些指令, 并通过实验找出最佳的 kernel 实现方式。
