## Gluon-08-Warp_Specialization

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
- **Gluon-08-Warp_Specialization**（本文）
- [Gluon-09-TMA_Gather_Scatter](https://zhuanlan.zhihu.com/p/2003599190585017693)
- [Gluon-10-tcgen05_copy](https://zhuanlan.zhihu.com/p/2003603029975381459)
- [Gluon-11-tcgen05-mma-scaled](https://zhuanlan.zhihu.com/p/2003603119259550000)

在典型的 GPU kernel 中, 所有 warp 都并行执行相同的任务, Warp Specialization 是让不同 warp 执行不同的任务。

通过 Warp Specialization, 我们可以将任务分配到不同的 warp 中, 从而重叠执行 kernel 中相互独立的部分。
这缩短了每个 warp 的关键路径, 同时借助 warp 调度器来动态调度这些 warp。
我们还可以重叠那些使用不同硬件单元的非异步操作, 而无需依赖精确的 SASS 级指令交错。

然而, Warp Specialization 也有代价: 额外的同步开销、更高的共享内存使用量(用于分区间的数据通信)、以及更大的总体寄存器压力。

Gluon 中的 Warp Specialization 功能仅支持 Hopper 及更新架构的 GPU。

### 逐元素加法 kernel 的 Warp Specialization 实现

让我们回顾逐元素加法 kernel, 并实现一个 Warp Specialization 版本。
在 Warp Specialization kernel 中, 执行特定任务的一组 warp 称为"分区(partition)", 每个分区可以拥有不同数量的 warp 和寄存器。

首先, 我们需要确定有哪些分区, 以及各分区需要分配多少寄存器。
Warp Specialization 的一个优势在于, 处理标量值的分区仅需 1 个 warp, 可以给其分配较少的寄存器。
例如, 我们可以设置一个只发起 TMA 加载/存储的分区, 该分区可以仅使用 1 个 warp 和 24 个寄存器(24 是 warp 分配的最小寄存器数)。
还可以设置一个计算分区, 包含 4 或 8 个 warp, 用于执行向量加法。准确估算寄存器分配是比较困难的, 通常需要反复试验、性能分析和自动调优。
我们需要使用 mbarrier 在分区之间实现生产者-消费者模式的信号传递。

要编写 Warp Specialization 的 kernel, 需要为每个分区编写独立的函数。
其中一个分区必须指定为"默认"分区, 它的 warp 数量始终等于传递给 kernel 的 `num_warps`。
其他分区称为"工作(worker)"分区, 可以有不同数量的 warp。
所有工作分区函数的签名必须相同, 且只有默认分区可以接受 tensor 参数。

简要说明各分区的职责:
加载分区将输入数据取到共享内存(smem)并通知计算分区, 计算分区消费操作数后, 通过 smem 将结果传递给存储分区。

回顾之前的教程, 我们需要 `fence_async_shared` 来同步异步代理访问和通用代理访问。
这同样适用于不同分区中的缓冲区访问, 即使它们通过 `mbarrier.arrive` 进行了排序:

```python
smem.store(value)  # 在分区 A 中
fence_async_shared()
mbarrier.arrive(bar, count=1)

mbarrier.wait(bar, phase=0)  # 在分区 B 中
tma.async_copy_shared_to_global(desc, [0, 0], smem)
```

共享内存存储和 TMA 存储之间必须插入一个 fence(内存屏障)。

```python
value = smem.load()
mbarrier.arrive(bar, count=1)

mbarrier.wait(bar, phase=0)
fence_async_shared()
tma.async_copy_global_to_shared(desc, [0, 0], bar, smem)
```

共享内存加载和 TMA 加载之间也必须插入一个 fence。

整体架构说明: kernel 实现 C = A + B, 但将工作分成了 3 个独立的分区(partition), 每个分区由不同的 warp 执行

GPU SM 内部三个分区:

```
  Load Partition        Compute Partition       Store Partition
  1 warp, 24 regs       num_warps (4~8), 剩余regs  1 warp, 24 regs
  TMA Load              a + b = c               TMA Store
  Global → SMEM         SMEM 读写                SMEM → Global
        │                  ↑    │                    ↑
        └──────────────────┘    └────────────────────┘
           mbarrier 同步            mbarrier 同步
```

分区间同步时序 (时间 →):

```
Load:     [Load 0] ──arrive load_ready──→ [Load 1] ──arrive──→ [Load 2] ...
                         ↓                     ↓                    ↓
Compute:            wait load_ready       wait load_ready      wait load_ready
                   [Compute 0]            [Compute 1]          [Compute 2] ...
                    arrive load_empty      arrive load_empty    arrive load_empty
                    arrive c_ready         arrive c_ready       arrive c_ready
                         ↓                     ↓                    ↓
Store:              wait c_ready           wait c_ready         wait c_ready
                   [Store 0]              [Store 1]            [Store 2] ...
                    arrive c_empty         arrive c_empty       arrive c_empty
```

双缓冲(num_buffers=2)流水线效果:

```
  时间步   Buffer 0              Buffer 1
  T0       Load[0] 加载中
  T1       Compute[0] 计算中     Load[1] 加载中
  T2       Store[0] 存储中       Compute[1] 计算中
  T3       Load[2] 加载中        Store[1] 存储中
  T4       Compute[2] 计算中     Load[3] 加载中
  ...      ...                  ...

时间 →   T0    T1    T2    T3    T4    T5    T6    T7
Load:   [L0]  [L1]  [L2]  [L3]  [L4]  [L5]  [L6]  [L7]
Compute:      [C0]  [C1]  [C2]  [C3]  [C4]  [C5]  [C6]
Store:              [S0]  [S1]  [S2]  [S3]  [S4]  [S5]
         启动 ← ─ ─ 稳态: 三个分区同时工作 ─ ─ → 收尾
```

```python
import pytest
import torch
import triton
import importlib
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    tensor_memory_descriptor,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

# 重用上一个教程中的实用程序。
t3 = importlib.import_module("03-async-copy")
t4 = importlib.import_module("04-tma")
t7 = importlib.import_module("07-persistence")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")


@gluon.jit
def load_partition(descs, barriers, buffers, xoff, numel, YBLOCK: gl.constexpr):
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    num_buffers: gl.constexpr = a_bufs.type.shape[0]

    # 所有分区的内层循环迭代次数必须相同。
    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        # 每个buffer, 都有对应的 phase, 在这里提前翻转 phase
        # num_buffers = 2 时:
        # i=0 → 0//2=0 → 0&1=0  (phase 0)
        # i=1 → 1//2=0 → 0&1=0  (phase 0)
        # i=2 → 2//2=1 → 1&1=1  (phase 1)  ← 第二轮, phase 翻转
        # i=3 → 3//2=1 → 1&1=1  (phase 1)
        # i=4 → 4//2=2 → 2&1=0  (phase 0)  ← 第三轮, phase 再翻转
        # i=5 → 5//2=2 → 2&1=0  (phase 0)
        # ...
        phase = i // num_buffers & 1
        a_buf = a_bufs.index(index)
        b_buf = b_bufs.index(index)
        load_empty_bar = load_empty_bars.index(index)
        load_ready_bar = load_ready_bars.index(index)

        # 等待当前缓冲区变为空闲。
        # phase 最开始为0, 因此我们从 phase ^ 1 = 1 开始, 让生产者可以立即开始填充流水线。
        mbarrier.wait(load_empty_bar, phase ^ 1)

        # 发起 TMA 加载, 并在加载完成时发出操作数缓冲区就绪的信号。即 load_ready_bar 的计数减 1.
        yoff = i * YBLOCK
        mbarrier.expect(load_ready_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [xoff, yoff], load_ready_bar, a_buf)
        tma.async_copy_global_to_shared(b_desc, [xoff, yoff], load_ready_bar, b_buf)


@gluon.jit
def store_partition(descs, barriers, buffers, xoff, numel, YBLOCK: gl.constexpr):
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    # 此分区消费通过 smem 传递的加法结果, 并将其存储到全局内存。
    num_buffers: gl.constexpr = c_bufs.type.shape[0]
    # 通过软件流水线, 我们会保持 `num_buffers-1` 个存储操作处于进行中(in flight)状态。
    outstanding_stores: gl.constexpr = num_buffers - 1

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        c_buf = c_bufs.index(index)
        c_ready_bar = c_ready_bars.index(index)

        # 等待计算分区生成 c。
        mbarrier.wait(c_ready_bar, phase)
        yoff = i * YBLOCK
        tma.async_copy_shared_to_global(c_desc, [xoff, yoff], c_buf)

        tma.store_wait(outstanding_stores)
        c_empty_bar = c_empty_bars.index((i - outstanding_stores) % num_buffers)
        # 通知计算分区: `outstanding_stores` 次迭代之前的缓冲区已被消费,
        # 前提是已经有足够多的未完成存储。
        mbarrier.arrive(c_empty_bar, count=1, pred=i >= outstanding_stores)

    # 等待完 c 的最后一个值后, 所有其他分区都已退出。
    # 我们只需等待存储操作全部完成。
    tma.store_wait(0)



# 默认分区的签名可以与工作分区函数不同。
@gluon.jit
def compute_partition(barriers, buffers, ynumel, YBLOCK: gl.constexpr, layout: gl.constexpr):
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers

    num_load_buffers: gl.constexpr = a_bufs.type.shape[0]
    num_store_buffers: gl.constexpr = c_bufs.type.shape[0]

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        load_index = i % num_load_buffers
        load_phase = i // num_load_buffers & 1
        a_buf = a_bufs.index(load_index)
        b_buf = b_bufs.index(load_index)
        load_ready_bar = load_ready_bars.index(load_index)
        load_empty_bar = load_empty_bars.index(load_index)

        # 等待操作数就绪, 然后消费它们。
        mbarrier.wait(load_ready_bar, load_phase)
        a_val = a_buf.load(layout)
        b_val = b_buf.load(layout)
        # 在通知加载分区之前插入 fence, 确保共享内存加载和 TMA 加载之间的正确排序。
        fence_async_shared()
        mbarrier.arrive(load_empty_bar, count=1)

        c_val = a_val + b_val

        store_idx = i % num_store_buffers
        store_phase = i // num_store_buffers & 1
        c_buf = c_bufs.index(store_idx)
        c_empty_bar = c_empty_bars.index(store_idx)
        c_ready_bar = c_ready_bars.index(store_idx)

        mbarrier.wait(c_empty_bar, store_phase ^ 1)
        c_buf.store(c_val)
        # 插入 fence 以确保与 TMA 存储的正确排序。
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count=1)


@gluon.jit
def elementwise_add_warp_specialized_kernel(
        a_desc, b_desc, c_desc,
        xnumel, ynumel, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,
        num_load_buffers: gl.constexpr, num_store_buffers: gl.constexpr, num_warps: gl.constexpr):
    # 选择一个便于避免 bank 冲突的布局。
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    # 分配所有 buffer 和 barrier
    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_load_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_load_buffers] + b_desc.block_type.shape, b_desc.layout)
    c_bufs = gl.allocate_shared_memory(c_desc.dtype, [num_store_buffers] + c_desc.block_type.shape, c_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout())
    c_empty_bars = gl.allocate_shared_memory(gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout())
    c_ready_bars = gl.allocate_shared_memory(gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(num_load_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)
    for i in gl.static_range(num_store_buffers):
        mbarrier.init(c_empty_bars.index(i), count=1)
        mbarrier.init(c_ready_bars.index(i), count=1)

    descs = (a_desc, b_desc, c_desc)
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    numel = (xnumel, ynumel)

    pid = gl.program_id(0)
    xoff = pid * XBLOCK

    # `gl.warp_specialize` 用于声明 kernel 的 Warp Specialization 部分。
    # 它接受一个列表, 其中每个元素是 (分区函数, 参数元组) 的形式。
    # 默认分区(列表的第一个元素)的参数可以包含 tensor, 而工作分区的参数则不能包含 tensor。
    # 每个分区的 warp 数量和寄存器预算通过后续的列表参数传递。
    #
    # 注意: NVIDIA GPU 上的 warp 和寄存器分配以 warp group(4 个连续的 warp)为单位进行。
    # Kernel 使用的 warp 数会向上取整到 4 的倍数。
    #
    # 编译器会尝试合理组织 warp 以减少寄存器分配量。默认分区会获得传递给 kernel 的 `maxnreg`中剩余的所有寄存器。
    gl.warp_specialize([
        (compute_partition, (barriers, buffers, ynumel, YBLOCK, layout)),  # 默认分区 - num_warps 个 warp
        (load_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),  # 工作分区1 - 1 warp, 24 寄存器
        (store_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),  # 工作分区2 - 1 warp, 24 寄存器
    ], [1, 1], [24, 24])  # 每个工作分区的 warp 数为1, 每个工作分区的寄容器数为24


def elementwise_add_warp_specialized(a, b, c, XBLOCK=32, YBLOCK=64,  #
                                     num_load_buffers=2, num_store_buffers=2, num_warps=4):
    xnumel, ynumel = a.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )

    block_shape = [XBLOCK, YBLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)

    # 默认情况下, Warp Specialization kernel 假设 maxnreg=256(每线程允许的最大值)来决定寄存器的重新分配。
    # 因此我们需要显式设置寄存器限制, 避免寄存器使用过多。
    # 该 kernel 中总共将有 `num_warps+4` 个 warp, 其中 num_warps 是默认分区的 warp 数,
    # 两个工作分区各使用 1 个 warp, 但由于 warp group 对齐到 4 的倍数, 2 个工作 warp 实际会占用 4 个 warp 的资源槽位。
    # 总寄存器使用量为: maxnreg * (num_warps+4) * 32, maxnreg = 256, num_warps= 4 时共 256 × (4+4) × 32 = 65536 个寄存器, 为单个 SM 的上限。
    # 由于我们设置了 worker_num_regs, 实际使用的寄存器数为(maxnreg × num_warps + 24×4) * 32	= 35824 个
    # 较低的寄存器使用量可以提高 occupancy(每个 SM 能并发运行更多线程块), 从而可能获得更好的性能。在规划 occupancy 时需要考虑这一点。
    elementwise_add_warp_specialized_kernel[grid](  #
        a_desc, b_desc, c_desc, xnumel, ynumel,  #
        XBLOCK, YBLOCK, num_load_buffers, num_store_buffers,  #
        num_warps=num_warps, maxnreg=128)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 120)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 64)])
@pytest.mark.parametrize("num_load_buffers, num_store_buffers", [(1, 1), (2, 2)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_elementwise_add_warp_specialized(xnumel, ynumel, XBLOCK, YBLOCK, num_load_buffers, num_store_buffers,
                                          num_warps):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add_warp_specialized(a, b, c, XBLOCK, YBLOCK, num_load_buffers, num_store_buffers, num_warps)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)
```

### Warp Specialization 版本 elementwise_add kernel 性能测试

```python
if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    XBLOCK = 64
    YBLOCK = 128
    num_load_buffers = 3
    num_store_buffers = 1
    num_warps = 4

    ms = triton.testing.do_bench(lambda: t4.elementwise_add_tma(  #
        A, B, C, XBLOCK, YBLOCK, num_load_buffers))
    print(f"elementwise_add_tma: {t3.get_throughput(ms, C):.2f} TB/s")

    ms = triton.testing.do_bench(lambda: elementwise_add_warp_specialized(  #
        A, B, C, XBLOCK, YBLOCK, num_load_buffers, num_store_buffers, num_warps))
    print(f"elementwise_add_warp_specialized: {t3.get_throughput(ms, C):.2f} TB/s")
    print()
```

GB200 上的性能结果:

```
elementwise_add_tma: 5.89 TB/s
elementwise_add_warp_specialized: 5.98 TB/s
```

Warp Specialization 依靠 warp 调度器来隐藏延迟, 在 04-tma.py 的软件流水线 kernel 之上又获得了一点性能提升。
不过收益比较小, 这是因为 elementwise kernrl 主要是 memory bound的。

在以前的教程中, 我们有时将 kernel 的 occupancy 设为大于 1 运行。这是因为某些kernel可能会停顿 (stall) 或无法充分利用 SM 资源。
在这样做的过程中, 我们依靠 warp 调度来重叠 kernel 实例并隐藏延迟。

然而, 由于程序无法看到 SM 上其他程序在做什么, 它们无法协调 SM 的使用以及共享机器资源。
当用于构建最小化关键路径和最大化硬件利用率的复杂调度时, Warp Specialization 尤其强大。
换句话说, warp Specialization 允许我们将多个程序融合到一个 kernel 中。

### Warp Specialization 版本的 persistent matmul

接下来我们继续对 Blackwell matmul 进行优化, 实现一个 warp Specialization 版本的 persistent matmul: C = A @ B。

- 使用相同的块大小 BLOCK_{M,N,K} = (128, 256, 64)
- 使用 subtile 来减少 epilogue smem
- 对累加器进行双缓冲以完全重叠 epilogue

因为 epilogue 是重叠的, 我们可以通过设置 subtile=4 来允许 4 个缓冲区(对应代码_split_n(acc, p.SUBTILE_FACTOR))。
不过在 K 较小时, 复用 B 可能仍然更好。
不使用 subtile (SUBTILE_FACTOR=1): Epilogue SMEM: [128 × 256] × 2B(fp16) = 64 KB  ← 很大！
使用 subtile (SUBTILE_FACTOR=4): Epilogue SMEM: [128 × 64] × 2B(fp16) = 16 KB  ← 只需要 1/4！
因为 epilogue 分 4 次存储: 
  for i in range(4):  # SUBTILE_FACTOR = 4
      acc_smem.store(accs[i])      # 只需要 [128, 64] 的缓冲区
      tma.async_copy_shared_to_global(...)
复用同一个 [128, 64] 的缓冲区 4 次！
节省的 SMEM = 64KB - 16KB = 48KB → 可以用来增加操作数缓冲区数量, 即num_buffers 可以从 2 增加到 4
SMEM 预算分配(假设 ~228KB 可用):

```
  不用 Subtile (num_buffers=2)          使用 Subtile (num_buffers=4)
    a_bufs[2x128x64]  = 32 KB            a_bufs[4x128x64]  = 64 KB
    b_bufs[2x64x256]  = 64 KB            b_bufs[4x64x256]  = 128 KB
    epilogue[128x256] = 64 KB            epilogue[128x64]  = 16 KB ← 节省 48KB
    总计: ~160 KB, 流水线深度 2           总计: ~208 KB, 流水线深度 4
```

总体架构如下

matmul_warp_specialized_kernel 总体架构:

```
  Load Partition (Worker 1)    MMA Partition (Worker 2)    Epilogue Partition (默认)
  1 warp, 24 regs              1 warp, 24 regs             num_warps, 剩余 regs
  TMA Load A, B                tcgen05_mma()               Load from TMEM
  Global → SMEM                SMEM → TMEM                 Type Convert (fp32→fp16)
  a_bufs[num_buffers]          累加 K 维度                   TMA Store: TMEM → SMEM → Global
         │                          │                             │
         │── load_ready_bars ──→    │── acc_ready_bars ──→        │
         │←─ load_empty_bars ──     │←─ acc_empty_bars ──        │
```

流水线时序图如下

Warp Specialization 双缓冲流水线时序图
                            (num_buffers = 2)

```
  时间步      Buffer 0                          Buffer 1
  ════════════════════════════════════════════════════════════════════════════════════

    T0   Load[0]                 ← Load分区加载第0块数据到Buffer0
         Global → SMEM
                  ↓ load_ready_bar[0]

    T1   Compute[0]                          Load[1]           ← 并行: Load加载下一块
         a + b = c                           Global → SMEM
                  ↓ load_empty_bar[0]                ↓ load_ready_bar[1]
                  ↓ c_ready_bar[0]

    T2   Store[0]                            Compute[1]        ← 并行: Compute计算
         SMEM → Global                       a + b = c
                  ↓ c_empty_bar[0]                   ↓ load_empty_bar[1]
                                                     ↓ c_ready_bar[1]

    T3   Load[2]                             Store[1]          ← 并行: Store存储
         Global → SMEM  ← Buffer0 被复用     SMEM → Global
                                                     ↓ c_empty_bar[1]

    T4   Compute[2]                          Load[3]           ← Buffer1 被复用
         a + b = c                           Global → SMEM

         ...循环继续...
```

```python
# 传递参数的辅助类。
@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SUBTILE_FACTOR: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs,
                 acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_bufs = a_bufs
        self.b_bufs = b_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.SUBTILE_FACTOR = gl.constexpr(SUBTILE_FACTOR)
        self.num_warps = gl.constexpr(num_warps)


# 计数器(环形机制), 用于跟踪 barrier 索引和 phase。
# state.next() 调用序列 (num_buffers=2):
#
# 第一轮 (phase=1):
#   k=0    index=0, phase=1  →  使用 a_bufs[0], b_bufs[0], ready_bars[0]
#   k=64   index=1, phase=1  →  使用 a_bufs[1], b_bufs[1], ready_bars[1]
#          → rollover: index=2 == num_buffers, index 回到 0, phase 翻转为 0
#
# 第二轮 (phase=0):
#   k=128  index=0, phase=0  →  复用 a_bufs[0], b_bufs[0], 但 phase 变了
#   k=172  index=1, phase=0  →  复用 a_bufs[1], b_bufs[1]
#   ...
@aggregate
class Counter:
    index: gl.tensor  # 当前 buffer 索引 (0, 1, ..., num_buffers-1)
    phase: gl.tensor  # 当前 phase (0 或 1), 用于 mbarrier 同步
    num_barriers: gl.constexpr  # buffer 总数

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers  # 是否需要回绕, incr 等于 num_barriers 时为 True
        index = gl.where(rollover, 0, incr)   # rollover=True 时回绕到 0, 否则递增
        phase = gl.where(rollover, self.phase ^ 1, self.phase)  # rollover=True 时翻转 phase
        return Counter(index, phase, self.num_barriers)


@gluon.jit
def matmul_load_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    empty_bars = p.load_empty_bars
    ready_bars = p.load_ready_bars
    state = Counter.create(1, empty_bars.shape[0])

    # 循环遍历所有 Tile 并发出加载指令。
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        for k in range(0, K, BLOCK_K):
            # 获取缓冲区, 发出加载, 并异步完成它们。
            bar = ready_bars.index(state.index)
            mbarrier.wait(empty_bars.index(state.index), state.phase)
            mbarrier.expect(bar, p.a_desc.block_type.nbytes + p.b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()


@gluon.jit
def matmul_mma_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for _ in range(scheduler.get_num_tiles()):
        # 为整个内层循环获取累加器。
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, K, BLOCK_K):
            # 获取操作数, 发出 MMA, 并异步完成。
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(p.a_bufs.index(load_state.index), p.b_bufs.index(load_state.index), acc_buf, use_acc=use_acc)
            tcgen05_commit(p.load_empty_bars.index(load_state.index))
            load_state = load_state.next()
            use_acc = True
        # 异步完成累加器。
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


# 将一个 [M, N] 的 tensor 沿 N 维度分割成 SUBTILE_FACTOR 个子 tensor
@gluon.jit
def _split_n(x, SUBTILE_FACTOR: gl.constexpr):
    split_count: gl.constexpr = SUBTILE_FACTOR.bit_length() - 1  # log2
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            x = xs[j]
            # 重塑为 (M, 2, N//2), 然后进行 permute, 再进行 split, 以便 tensor 元素沿 N 保持连续。
            # [M, N] → [M, 2, N//2] → [M, N//2, 2] → split → 2×[M, N//2]
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs


@gluon.jit
def matmul_epilogue_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    dtype: gl.constexpr = p.c_desc.dtype

    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    acc_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_layout: gl.constexpr = get_tmem_reg_layout(dtype, (BLOCK_M, BLOCK_N), acc_tmem_layout, p.num_warps)
    SPLIT_N: gl.constexpr = BLOCK_N // p.SUBTILE_FACTOR
    acc_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, SPLIT_N], p.c_desc.layout)

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        # 等待累加器。由于 BLOCK_N=256, 我们需要将 TMEM 加载与 SMEM 存储交错以避免溢出 (spilling)。
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load(acc_layout)
        acc_state = acc_state.next()

        # BLOCK_N=256 太大, 一次性处理会导致寄存器溢出 (spilling)。分割成 4 份后, 每次只处理 64 列。
        accs = _split_n(acc, p.SUBTILE_FACTOR)
        for i in gl.static_range(len(accs)):
            acc = accs[i].to(dtype)
            tma.store_wait(pendings=0)  # 与向下转换 (downcast) 重叠
            acc_smem.store(acc.to(dtype))
            # 在第一次 SMEM 存储后到达 (arrive), 并依靠 ptxas 进行交错。
            if i == 0:
                mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
            fence_async_shared()
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + SPLIT_N * i], acc_smem)
    # 将最后一个存储与等待重叠, 然后在这里等待最后一个存储。
    tma.store_wait(pendings=0)


@gluon.jit
def matmul_warp_specialized_kernel(a_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr, num_buffers: gl.constexpr,
                                   SUBTILE_FACTOR: gl.constexpr, num_warps: gl.constexpr):
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

    # acc_bufs=1 时, MMA 必须等待 Epilogue 完成才能开始下一个 Tile, 两者完全串行, 无法重叠！
    # acc_bufs=2 时最优, MMA 使用 acc[1] 时, Epilogue 在处理 acc[0], 两者完全并行！
    # acc_bufs=num_buffers(如 4), 是浪费资源。因为在稳态下, MMA 完成一个 Tile 需要 K/BLOCK_K 次 MMA (比如 32 次),
    # 而 Epilogue 处理一个 Tile 需要 SUBTILE_FACTOR 次 TMA store (比如 4 次) , MMA 时间 >> Epilogue 时间。
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_bufs = allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = PartitionArgs(a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs,
                      acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps)
    gl.warp_specialize([
        (matmul_epilogue_partition, (p, SchedulerImpl)),
        (matmul_load_partition, (p, SchedulerImpl)),
        (matmul_mma_partition, (p, SchedulerImpl)),
    ], [1, 1], [24, 24])


def matmul_warp_specialized(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, SUBTILE_FACTOR, num_warps, SchedulerImpl):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    matmul_warp_specialized_kernel[grid](a_desc, b_desc, c_desc, SchedulerImpl, num_buffers, SUBTILE_FACTOR,
                                         num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("SUBTILE_FACTOR", [4])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("SchedulerImpl", t7.schedulers)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_matmul_warp_specialized(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, SUBTILE_FACTOR, num_warps,
                                 SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    matmul_warp_specialized(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, SUBTILE_FACTOR, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)
```

### matmul_warp_specialized 性能测试

```python
if __name__ == "__main__" and is_blackwell():
    print("Benchmarking matmul_warp_specialized")
    print("====================================")
    args = {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "num_buffers": 4,
        "SUBTILE_FACTOR": 4,
        "num_warps": 4,
        "SchedulerImpl": t7.GroupedPersistentTileScheduler(8),
    }

    M, N = 8192, 8192
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    print("    K  warp-specialized    cublas")
    for K in [2**i for i in range(9, 15)]:
        as_flops = partial(t7.get_flops, M=M, N=N, K=K)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        BT = B.T.contiguous()
        r0 = as_flops(triton.testing.do_bench_cudagraph(lambda: matmul_warp_specialized(A, B, C, **args)))
        r1 = as_flops(triton.testing.do_bench(lambda: cublas.matmul(A, BT, C)))
        print(f"{K:>5} {r0:>17.2f} {r1:>9.2f}")
```

```
    K  warp-specialized    cublas
  512           1160.28   1130.67
 1024           1249.69   1148.52
 2048           1347.18   1261.59
 4096           1390.95   1299.38
 8192           1350.01   1401.10
16384           1448.14   1508.76
```

性能进一步提升！我们在小 K 上击败了 cuBLAS, 即使仍有很多调优工作来进一步提高性能。
在 Blackwell 上, Warp Specialization 对于实现峰值性能很关键。
