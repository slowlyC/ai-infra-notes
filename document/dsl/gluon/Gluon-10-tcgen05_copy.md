## Gluon-10-tcgen05_copy

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
- **Gluon-10-tcgen05_copy**（本文）
- [Gluon-11-tcgen05-mma-scaled](https://zhuanlan.zhihu.com/p/2003603119259550000)

### TCGen05 Copy 指令介绍

本教程介绍如何使用`tcgen05_copy` 指令, 以及其应用场景。

`tcgen05_copy` 是一个异步 tensor core 操作, 用于将数据从共享内存复制到 tensor memory。
和 `tcgen05_mma` 一样, `tcgen05_copy` 的完成状态通过 `tcgen05_commit` 在 mbarrier 上跟踪。
单个或多个 `tcgen05_copy` 操作的完成可以由一个 `tcgen05_commit` 来跟踪:

```python
tcgen05_copy(lhs_smem, lhs_tmem)
tcgen05_copy(acc_smem, acc_tmem)
tcgen05_commit(bar)
mbarrier.wait(bar, phase=phase)
acc = acc_tmem.load(acc_reg_layout)
lhs = lhs_tmem.load(lhs_reg_layout)
```

`tcgen05_copy` 可以将数据复制到 tensor memory 中, 作为 `tcgen05_mma` 指令的输入。
由于 `tcgen05_copy` 与 `tcgen05_mma` 是隐式流水线化的, 因此即使它是异步的, MMA 也保证在 copy 完成后才开始:

```python
tcgen05_copy(smem, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
tcgen05_commit(bar)
mbarrier.wait(bar, phase=phase)
```

隐式流水线化是因为 PTX 级别的 `tcgen05.copy` 和 `tcgen05.mma` 指令由 SM 上的 tensor core pipe 执行。
可以把它想象成一个专门运行 tensor core 指令的单线程, 与 SM 的其余部分异步执行。
换句话说, 所有 `tcgen05_*` 指令都会将一个 tensor core 操作入队到 tensor pipe 中, 按顺序执行。

以下写法也是合法的:

```python
tcgen05_copy(lhs_smem0, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
tcgen05_commit(bar)

tcgen05_copy(lhs_smem1, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
```

因为第二个 `tcgen05_copy` 只会在前面的 `tcgen05_mma` 完成后才执行。
换句话说, `tcgen05_copy`、`tcgen05_mma` 和 `tcgen05_commit` 都是隐式流水线化的, 按顺序执行。

`tcgen05_copy` 也是通过 async proxy 访问共享内存, 因此需要确保在适当的地方插入 fence:

```python
lhs_smem.store(value1)
fence_async_shared()
tcgen05_copy(lhs_smem, lhs_tmem)
tcgen05_commit(bar)

mbarrier.wait(bar, phase=phase)
lhs_smem.store(value0)
```

注意到第二次写入 `lhs_smem` 之前不需要 fence, 因为 mbarrier.wait 操作完成会隐式地 fence generic 和 async proxy。

`tcgen05_copy` 棘手的地方在于选择正确的共享内存和 tensor memory 布局, 因为 `tcgen05_copy` 只支持有限的指令形状.

### tcgen05_copy 基础示例

让我们写一个使用 `tcgen05_copy` 的示例 kernel, 并展示共享内存和 tensor memory 布局的要求。

```python
import itertools
import importlib
import pytest
import triton
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    tensor_memory_descriptor,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    fence_async_shared,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma,
    mbarrier,
    tma,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")

# 复用之前教程的工具函数
t7 = importlib.import_module("07-persistence")
t8 = importlib.import_module("08-warp-specialization")

@gluon.jit
def tcgen05_copy_kernel(in_ptr, in_stride0, in_stride1, out_ptr, out_stride0, out_stride1, M: gl.constexpr,
                        N: gl.constexpr, smem_layout: gl.constexpr, tmem_layout: gl.constexpr):
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    offs_m = gl.arange(0, M, gl.SliceLayout(1, coalesced_2d_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, coalesced_2d_layout))

    input = gl.load(in_ptr + offs_m[:, None] * in_stride0 + offs_n[None, :] * in_stride1)

    # 使用 tile 形状 [M, N] 分配共享内存和 tensor memory。
    smem = gl.allocate_shared_memory(input.dtype, (M, N), smem_layout)
    tmem = allocate_tensor_memory(input.dtype, (M, N), tmem_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
    mbarrier.init(bar, count=1)

    # 从共享内存复制数据到 tensor memory。
    smem.store(input)
    # Fence generic 和 async proxy
    fence_async_shared()
    # 发起异步复制
    tcgen05_copy(smem, tmem)
    # 跟踪异步复制的完成状态
    tcgen05_commit(bar)
    # 等待异步复制完成
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # 从 tensor memory 读取数据。
    tmem_reg_layout: gl.constexpr = get_tmem_reg_layout(input.dtype, (M, N), tmem_layout, gl.num_warps())
    output = tmem.load(tmem_reg_layout)

    # 使用 coalesced 布局写入。
    output = gl.convert_layout(output, coalesced_2d_layout)
    gl.store(out_ptr + offs_m[:, None] * out_stride0 + offs_n[None, :] * out_stride1, output)


def tcgen05_copy_example(M, N, smem_layout, tmem_layout, dtype):
    input = torch.randn(M, N, dtype=dtype, device="cuda")
    output = torch.empty_like(input)
    tcgen05_copy_kernel[(1, )](input, *input.stride(), output, *output.stride(), M, N, smem_layout, tmem_layout)
    # 只检查输入和输出是否相等。
    torch.testing.assert_close(input, output, atol=0, rtol=0)
```

### tcgen05_copy 布局限制

接下来让我们测试编写的`tcgen05_copy` kernel. 首先让我们探索`tcgen05_copy`源共享内存的有效布局。
`tcgen05_copy` 的目标布局是 `TensorMemoryLayout`, 源共享内存布局通常是 `NVMMASharedLayout`。
也支持其他特殊布局, 如 `SharedLinearLayout`, 但本教程不会涉及。

使用`NVMMASharedLayout` 时有以下限制:
- `TensorMemoryLayout` 的 blockM 必须是 128。
- dtype 必须是 32 位(例如 gl.float32)。
- 布局必须是 swizzled(swizzle_byte_width > 0)。
- 布局不能是转置的。

```python
configs = []
TMEM_BLOCK_M = 128
for TMEM_BLOCK_N in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    for M, N in itertools.product([128, 256], [16, 32, 64, 128, 256]):
        if M < TMEM_BLOCK_M or N < TMEM_BLOCK_N or M * N * 4 > 228 * 1024:
            continue
        configs.append((M, N, TMEM_BLOCK_N))

@pytest.mark.parametrize("M, N, TMEM_BLOCK_N", configs)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("swizzle", [32, 64, 128])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tcgen05_copy_nvmma_shared(M, N, TMEM_BLOCK_N, dtype, swizzle):
    bitwidth = dtype.itemsize * 8
    # 不支持某些共享内存布局的实现。
    if M == 256 and swizzle // TMEM_BLOCK_N >= 8:
        pytest.skip("no tcgen05.copy atom exists for codegen")
    # NVMMASharedLayout swizzle 块形状有最小大小限制。
    if N < swizzle // dtype.itemsize:
        pytest.skip("block shape along contiguous dimension is too small for the swizzle byte width")

    bitwidth = dtype.itemsize * 8
    smem_layout = gl.NVMMASharedLayout(swizzle_byte_width=swizzle, element_bitwidth=bitwidth, rank=2)
    tmem_layout = TensorMemoryLayout(block=(TMEM_BLOCK_M, TMEM_BLOCK_N), col_stride=32 // bitwidth)
    tcgen05_copy_example(M, N, smem_layout, tmem_layout, dtype)
```

### tcgen05_copy 在矩阵乘法累加中的应用

虽然 tcgen05_copy 到 TensorMemoryLayout 只支持 32 位 dtype, 但其可以在 mma kernel `D = A @ B + C` 中用到。
具体来说, 我们可以用 `tcgen05_copy` 异步复制`C`到 tensor memory, 然后执行 `tcgen05_mma` 将其累加到 tensor memory。

我们使用 `gl.store` 写回累加器的结果, 直接从寄存器写到全局内存, 跳过共享内存缓冲区, 以节省共享内存。
因为 C 是 float32 类型, 如果用 TMA 写回, 需要先把结果放到共享内存的 float32 缓冲区, 这会占用很多共享内存。
我们继续使用 warp specialization 来高效地重叠 epilogue store 和 kernel 的其余部分,
即 epilogue 的 gl.store 与下一个 tile 的 load/MMA 并行执行, 此时避免在 epilogue 中使用 TMA store 也可以减少对 TMA pipe 的争用。

```python
@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    d_ptr: gl.tensor
    d_stride_m: gl.tensor
    d_stride_n: gl.tensor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    c_buf: gl.shared_memory_descriptor
    c_empty_bar: gl.shared_memory_descriptor
    c_ready_bar: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SchedulerImpl: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, d_ptr, d_stride_m, d_stride_n, a_bufs, b_bufs, load_empty_bars,
                 load_ready_bars, c_buf, c_empty_bar, c_ready_bar, acc_bufs, acc_empty_bars, acc_ready_bars,
                 SchedulerImpl):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.d_ptr = d_ptr
        self.d_stride_m = d_stride_m
        self.d_stride_n = d_stride_n
        self.a_bufs = a_bufs
        self.b_bufs = b_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.c_buf = c_buf
        self.c_empty_bar = c_empty_bar
        self.c_ready_bar = c_ready_bar
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.SchedulerImpl = gl.constexpr(SchedulerImpl)


@gluon.jit
def matmul_accumulate_load_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    c_phase = 1
    state = t8.Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # 发起 C tile 的异步 TMA 加载。
        mbarrier.wait(p.c_empty_bar, c_phase)
        mbarrier.expect(p.c_ready_bar, p.c_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(p.c_desc, [off_m, off_n], p.c_ready_bar, p.c_buf)
        c_phase ^= 1
        # 内层循环加载。
        for k in range(0, K, BLOCK_K):
            bar = p.load_ready_bars.index(state.index)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            mbarrier.expect(bar, p.a_desc.block_type.nbytes + p.b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()


@gluon.jit
def matmul_accmulate_mma_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    c_phase = 0
    load_state = t8.Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = t8.Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for _ in range(scheduler.get_num_tiles()):
        # 我们预期 C 的加载比上一个 epilogue 释放累加器要花更长时间, 所以先获取 c_buf。
        mbarrier.wait(p.c_ready_bar, c_phase)
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        tcgen05_copy(p.c_buf, acc_buf)
        # copy 完成时会释放 c_buf。我们不需要等待 copy 完成, 因为它会与第一个 MMA 隐式流水线化。
        tcgen05_commit(p.c_empty_bar)
        c_phase ^= 1
        for k in range(0, K, BLOCK_K):
            # 等待操作数就绪。
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            # 发起 MMA, 完成后释放加载缓冲区。
            tcgen05_mma(p.a_bufs.index(load_state.index), p.b_bufs.index(load_state.index), acc_buf, use_acc=True)
            tcgen05_commit(p.load_empty_bars.index(load_state.index))
            load_state = load_state.next()
        # 最后一个 MMA 完成时释放累加器。
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def matmul_accumulate_epilogue_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    dtype: gl.constexpr = p.c_desc.dtype

    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    range_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, coalesced_2d_layout))
    range_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, coalesced_2d_layout))

    acc_layout: gl.constexpr = get_tmem_reg_layout(dtype, (BLOCK_M, BLOCK_N), p.acc_bufs.type.layout, gl.num_warps())
    acc_state = t8.Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # 等待累加器。
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load(acc_layout)
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()
        offs_m = (off_m + range_m)
        offs_n = (off_n + range_n)
        # 这个 `convert_layout` 相当昂贵, 会使用很多共享内存,
        # 因为 `acc_layout` 将连续的列分配给同一个线程, 但 coalesced 布局将连续的列分配给不同线程以实现高效的全局写入。
        # 我们可以对存储进行子 tile 化以减少共享内存使用。
        acc = gl.convert_layout(acc, coalesced_2d_layout)
        gl.store(p.d_ptr + offs_m[:, None] * p.d_stride_m + offs_n[None, :] * p.d_stride_n, acc)


@gluon.jit(do_not_specialize=["d_stride_m", "d_stride_n"])
def matmul_accumulate_kernel(a_desc, b_desc, c_desc, d_ptr, d_stride_m, d_stride_n, SchedulerImpl: gl.constexpr,
                             num_buffers: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    c_buf = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    c_empty_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    c_ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(c_empty_bar, count=1)
    mbarrier.init(c_ready_bar, count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_bufs = allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = PartitionArgs(a_desc, b_desc, c_desc, d_ptr, d_stride_m, d_stride_n, a_bufs, b_bufs, load_empty_bars,
                      load_ready_bars, c_buf, c_empty_bar, c_ready_bar, acc_bufs, acc_empty_bars, acc_ready_bars,
                      SchedulerImpl)
    gl.warp_specialize([
        (matmul_accumulate_epilogue_partition, (p, )),
        (matmul_accmulate_mma_partition, (p, )),
        (matmul_accumulate_load_partition, (p, )),
    ], [1, 1], [24, 24])


def matmul_accumulate(A, B, C, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_SIZE_M=8, num_buffers=3):
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)
    M, N = C.shape

    dtype = getattr(gl, str(A.dtype).split('.')[1])
    acc_dtype = getattr(gl, str(C.dtype).split('.')[1])
    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], dtype)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], dtype)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], acc_dtype)

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)
    D = torch.empty((M, N), dtype=C.dtype, device="cuda")

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    matmul_accumulate_kernel[grid](a_desc, b_desc, c_desc, D, *D.stride(), SchedulerImpl, num_buffers)
    return D


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 2048), (4096, 4096, 4096)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N", [(128, 128), (128, 64)])
@pytest.mark.parametrize("BLOCK_K, num_buffers", [(64, 3)])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_matmul_accumulate(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, dtype):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")
    D = matmul_accumulate(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers=num_buffers)
    torch.testing.assert_close(A @ B + C, D, atol=5e-3, rtol=1e-2)
```

### tcgen05_copy 的其他应用场景

`tcgen05_copy` 的另一个重要用途是从共享内存异步复制 tensor scales 到 tensor memory, 用于 `tcgen05_mma_scaled`。
我们将在下一个教程介绍 `tcgen05_mma_scaled`, 现在只需知道 tensor scales 必须通过 tensor memory 传递给 `tcgen05_mma_scaled`,
且 scales 的 tensor memory 布局必须是 `TensorMemoryScalesLayout`。
如果我们通过 TMA 将 scales 加载到共享内存, 就可以用 `tcgen05_copy` 高效地将 scales 复制到 tensor memory,
它可以与 `tcgen05_mma_scaled` 指令隐式流水线化:

```python
tma.async_copy_global_to_shared(a_scale_desc, ..., bar, a_scale_buf)
tma.async_copy_global_to_shared(b_scale_desc, ..., bar, b_scale_buf)
mbarrier.wait(bar, phase)

tcgen05_copy(a_scale_buf, a_scale_tmem)
tcgen05_copy(b_scale_buf, b_scale_tmem)
tcgen05_mma_scaled(a_buf, b_buf, acc_tmem, a_scale_tmem, b_scale_tmem, ...)
tcgen05_commit(mma_bar)
```

本教程的要点是使用 `tcgen05_copy` 从共享内存异步复制数据到 tensor memory。
`tcgen05_copy` 不支持所有布局, 但支持典型的 NVMMASharedLayouts。
这个指令在特定场景下很有用, 可以将数据从共享内存复制到 tensor memory, 而不需要通过寄存器中转, 后者会增加寄存器压力且速度较慢。
`tcgen05_copy`是异步的, 可以与其他 `tcgen05` 指令隐式流水线化。
