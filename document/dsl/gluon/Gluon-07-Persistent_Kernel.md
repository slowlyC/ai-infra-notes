## Gluon-07-Persistent_Kernel

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
- **Gluon-07-Persistent_Kernel**（本文）
- [Gluon-08-Warp_Specialization](https://zhuanlan.zhihu.com/p/2003602919912649692)
- [Gluon-09-TMA_Gather_Scatter](https://zhuanlan.zhihu.com/p/2003599190585017693)
- [Gluon-10-tcgen05_copy](https://zhuanlan.zhihu.com/p/2003603029975381459)
- [Gluon-11-tcgen05-mma-scaled](https://zhuanlan.zhihu.com/p/2003603119259550000)

在之前的教程中, 我们编写 kernel 时采用"一个程序处理一个block"的模式, 通过 grid 来启动所有工作。
这种方式会启动大量程序, 由 GPU 负责调度。主要优点是 GPU 能够在各个 SM (流多处理器)之间自动实现负载均衡。

但这种方法也有明显的缺点: 调度器本身会带来开销, 而且 GPU 对 kernel 的内存访问模式一无所知。
此外, 由于 GPU 需要等待当前 kernel 完全结束后才能分配新的工作, 这也限制了不同工作块之间的计算重叠。

Persistent kernel(持久化内核)是一种不同的编程模式:
每个程序会处理多个工作块, 程序会一直"驻留"在 GPU 上直到所有工作完成。
工作分配通常是静态的, 不过借助更高级的技术或 Cluster Launch Control 等硬件特性, 也可以实现动态调度。

本教程将通过实现一个持久化矩阵乘法来探索 persistent kernel 技术,
并展示如何在外层循环中引入流水线机制, 从而实现更大的计算重叠和更高的吞吐量。

### 统一的 MMA 抽象层

在前两个教程中, 我们分别介绍了 Hopper 和 Blackwell GPU 的 Tensor Core 操作。
为了让本教程更易于理解, 同时展示 Gluon 的特性, 我们将这两代 GPU 的 Tensor Core 构建成统一的抽象层,
使我们的 persistent matmul 能够同时支持 Hopper 和 Blackwell。

我们使用 @aggregate 装饰器来定义一个保存 matmul 状态的类。由于 WGMMA 的限制更多,
我们将 MMA 包装器的 API 设计成与 WGMMA 一致的风格。

```python
import itertools
import pytest
import torch
import triton
import importlib
import sys
from functools import partial
from typing import Union
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
    warpgroup_mma_accumulator,
)
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

t5 = importlib.import_module("05-wgmma")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

profiling_with_ncu = len(sys.argv) > 1 and sys.argv[1] == "profile"


def get_flops(ms, M, N, K):
    flops = 2 * M * N * K
    return flops * 1e-12 / (ms * 1e-3)
```

#### WGMMA 包装器

WGMMA 的包装器, 直接映射到底层 WGMMA 函数。

```python
# WGMMA 的包装器, 直接映射到底层 WGMMA 函数。
@aggregate
class WGMMA:
    acc: Union[warpgroup_mma_accumulator, gl.tensor]
    use_acc: gl.tensor

    @gluon.constexpr_function
    def __init__(self, acc, use_acc):
        self.acc = acc
        self.use_acc = use_acc

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        mma_layout: gl.constexpr = t5.pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
        return WGMMA(acc, gl.to_tensor(False))

    @gluon.jit
    def issue_async_mma(self, a, b):
        acc = warpgroup_mma(a, b, self.acc, is_async=True, use_acc=self.use_acc)
        # 注意: aggregate 类型不支持就地修改, 因此需要返回新实例, 由调用方重新赋值。
        return WGMMA(acc, gl.to_tensor(True))

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        acc = warpgroup_mma_wait(num_outstanding, (self.acc, ))
        return WGMMA(acc, self.use_acc)

    # 获取计算结果并重置累加器状态。
    @gluon.jit
    def take_result(self):
        return self.acc, WGMMA(self.acc, gl.to_tensor(False))
```

#### Blackwell tcgen05 MMA 包装器

Blackwell tcgen05 的 MMA 包装器。
为了实现 `wait_num_outstanding` 功能, 我们需要分配 barrier 并追踪已发出的 MMA 操作数量。

```python
# Blackwell tcgen05 的 MMA 包装器。
# 为了实现 `wait_num_outstanding` 功能, 我们需要分配 barrier 并追踪已发出的 MMA 操作数量。
@aggregate
class MMAv5:
    use_acc: gl.tensor
    acc_tmem: tensor_memory_descriptor
    bar: gl.shared_memory_descriptor
    counter: gl.tensor
    reg_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, use_acc, acc_tmem, bar, counter, reg_layout):
        self.use_acc = use_acc
        self.acc_tmem = acc_tmem
        self.bar = bar
        self.counter = counter
        self.reg_layout = gl.constexpr(reg_layout)

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], layout)
        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), layout, num_warps)
        return MMAv5(gl.to_tensor(False), acc_tmem, bar, gl.to_tensor(0), reg_layout)

    @gluon.jit
    def issue_async_mma(self, a, b):
        tcgen05_mma(a, b, self.acc_tmem, use_acc=self.use_acc)
        tcgen05_commit(self.bar)
        return MMAv5(gl.to_tensor(True), self.acc_tmem, self.bar, self.counter + 1, self.reg_layout)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        mbarrier.wait(self.bar, (self.counter - 1 - num_outstanding) & 1)
        return self

    @gluon.jit
    def take_result(self):
        next = MMAv5(gl.to_tensor(False), self.acc_tmem, self.bar, self.counter, self.reg_layout)
        return self.acc_tmem.load(self.reg_layout), next


def select_mma_impl():
    if torch.cuda.get_device_capability()[0] == 9:
        return WGMMA
    elif torch.cuda.get_device_capability()[0] == 10:
        return MMAv5
    else:
        return None
```

### 流水线化的矩阵乘法

现在让我们实现一个 matmul 来验证上述抽象。我们会同时对load和MMA进行流水线化, 这需要至少两个操作数缓冲区。
流水线化会让 persistent kernel 更有意思, 因为可以重叠的操作更多了。

接下来我们把 kernel 拆分成可复用的组件, 方便在不同实现之间共享。

```python
@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers: gl.constexpr, pred=True):
    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred=pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index), pred)
    return producer


@gluon.jit
def issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))
    return consumer, mma


@gluon.jit
def matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, num_buffers: gl.constexpr,
                            num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    gl.static_assert(num_buffers >= 2, "expected at least 2 buffers")
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    # 分开追踪生产者和消费者索引, 以支持 2 个以上的缓冲区。
    producer = 0
    consumer = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # 使用我们封装的 MMA 抽象
    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)

    # 预取 num_buffers-2 次加载, 为后续 MMA 重叠执行做准备。
    for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)

    for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
        producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)
        consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

    for _ in gl.static_range(num_buffers - 2):
        consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

    mma = mma.wait_num_outstanding(0)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c, mma = mma.take_result()
    c_smem.store(c.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps):
    MMAImpl = select_mma_impl()
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, num_buffers, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_pipelined_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)
```

### 流水线化矩阵乘法性能测试

让我们对组件化的matmul kernel进行性能测试
对于我们的 kernel, 最佳块尺寸是BLOCK_M=128、BLOCK_N=256, 这是在 Blackwell 和 Hopper 上都能使用的最大指令形状。
但在 Hopper 上需要 8 个 warp 才能容纳累加器所需的寄存器。
原因:
累加器大小 = BLOCK_M × BLOCK_N = 128 × 256 = 32,768 个 float32
每个线程最多只能使用 256 个寄存器
4 个 warp(128 线程): 32768 ÷ 128 = 256 寄存器/线程。达到极限, 可能溢出。
为什么 Blackwell 可以用 4 个 warp？
累加器可以存放在Tensor Memory中, 不占用普通寄存器, 所以 Blackwell 的寄存器压力更小。

```python
if __name__ == "__main__":
    M, N, K = 8192, 8192, 16 * 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

if __name__ == "__main__" and not profiling_with_ncu:
    BLOCK_M = 128
    BLOCK_N = 256
    is_hopper = torch.cuda.get_device_capability()[0] == 9
    warps = [8] if is_hopper else [4, 8]
    print("Benchmarking pipelined matmul")
    print("=============================")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print("BLOCK_K num_buffers num_warps tflops/s")
    for (BLOCK_K, num_buffers), num_warps in itertools.product([(128, 2), (64, 3), (64, 4)], warps):
        print(f"{BLOCK_K:>7} {num_buffers:>11} {num_warps:>9}", end=" ")
        fn = lambda: matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps)
        ms = triton.testing.do_bench_cudagraph(fn)
        print(f"{get_flops(ms, M, N, K):8.2f}")
    print()
```

性能结果如下:

```
# BLOCK_K num_buffers num_warps Blackwell  Hopper
#     128           2         4    735.96
#     128           2         8    697.97  489.26
#      64           3         4   1054.00
#      64           3         8    973.94  673.67
#      64           4         4   1175.70
#      64           4         8   1072.83  669.16
#
# Blackwell 的性能与之前教程中的结果一致, Hopper 上也有不错的表现。
# Hopper 在 3 个缓冲区时性能就已趋于饱和, 而 Blackwell 使用 4 个缓冲区还能继续提升。
# 这说明从 Hopper 到 Blackwell, MMA 相对于内存带宽的吞吐量比例有所提高。
# 值得注意的是, 我们的 kernel 占用率(occupancy)为 1。
```

### 持久化内核实现

要让 kernel 变成持久化的, 只需在kernel内部加一层循环, 遍历当前程序需要处理的所有tile。

我们先定义一个 Tile 调度器, 方便切换不同的调度策略。
先从最基础的行优先(row-major)调度器开始。

```python
@aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    @gluon.constexpr_function
    def __init__(self, pid_start, pid_end, num_pid_m):
        self.pid_start = pid_start
        self.pid_end = pid_end
        self.num_pid_m = num_pid_m

    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)
        return PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        # 按列优先顺序线性化 tile ID
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n


# 我们可以在持久性循环之前, 先初始化TMA barrier和MMA状态, 然后在循环内部复用这些状态。
# 注意: 操作数缓冲区的作用域要限制在内层循环内,
# 这样共享内存分配器才能知道它们的生命周期与 TMA 存储缓冲区不重叠, 从而可以复用内存。

@gluon.jit
def persistent_matmul_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, SchedulerImpl: gl.constexpr,
                             num_buffers: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    # 分开追踪生产者和消费者索引, 以支持 2 个以上的缓冲区。
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
        b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
        for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)

        for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
            producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)
            consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        for _ in gl.static_range(num_buffers - 2):
            consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        mma = mma.wait_num_outstanding(0)
        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
        c, mma = mma.take_result()
        c_smem.store(c.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
        tma.store_wait(pendings=0)


def persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    persistent_matmul_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, SchedulerImpl, num_buffers, num_warps=num_warps)


schedulers = [PersistentTileScheduler]


@pytest.mark.parametrize("M, N, K", [(2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("SchedulerImpl", schedulers)
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_persistent_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)
```

### 持久化矩阵乘法性能测试

让我们对持久化 matmul kernel 进行性能测试。

```python
if __name__ == "__main__" and not profiling_with_ncu:
    print("Benchmarking persistent matmul")
    print("==============================")
    print(f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N}")
    print("BLOCK_K num_buffers num_warps tflops/s")
    for (BLOCK_K, num_buffers), num_warps in itertools.product([(128, 2), (64, 3), (64, 4)], warps):
        print(f"{BLOCK_K:>7} {num_buffers:>11} {num_warps:>9}", end=" ")
        fn = lambda: persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps,
                                       PersistentTileScheduler)
        ms = triton.testing.do_bench_cudagraph(fn)
        print(f"{get_flops(ms, M, N, K):8.2f}")
    print()
```

性能结果如下:

```
# BLOCK_K num_buffers num_warps  Blackwell  Hopper
#     128           2         4     712.25
#     128           2         8     686.64  502.84
#      64           3         4    1032.16
#      64           3         8     938.81  661.11
#      64           4         4    1142.26
#      64           4         8    1071.46  658.84
#
# Hopper 上的性能略有提升, 但 Blackwell 上反而略有下降。
# 我们用 ncu 来分析一下 Blackwell 上这两个 kernel 的性能差异。
# 在命令行参数中传入 `profile`, 让每个 kernel 只运行一次, 便于采集性能数据。
# ```
# ncu --set full -o pipelined  --kernel-name matmul_pipelined_kernel  python 07-persistence.py profile
# ncu --set full -o persistent --kernel-name persistent_matmul_kernel python 07-persistence.py profile
# ```
```

```python
if __name__ == "__main__" and profiling_with_ncu:
    matmul_pipelined(A, B, C, 128, 256, 64, 4, 4)
    persistent_matmul(A, B, C, 128, 256, 64, 4, 4, PersistentTileScheduler)
```

### 性能分析

persistent kernel 变慢可能有多种原因。负载不均衡可能源于调度不够优化(工作分配不均匀),
也可能是运行时的波动(比如某些 TMA 访问耗时较长), 而静态调度器无法动态适应这种情况。

另一个可能的原因是全局内存访问模式不够友好:

```
ncu --import  pipelined.ncu-rep | grep "L2 Hit Rate"
    L2 Hit Rate                            %        61.11
ncu --import persistent.ncu-rep | grep "L2 Hit Rate"
    L2 Hit Rate                            %        52.93
```

persistent kernel 的 L2 缓存命中率下降了约 10%。
我们可以通过对 tile 进行"超级分组"(沿 M 维度将多个 tile 分为一组)来改善 L2 局部性。详见 03-matrix-multiplication.py。

### 分组调度器

下面我们实现一个采用分组策略的新调度器。

```
分组调度 vs 行优先调度

行优先调度 (GROUP_SIZE_M=1):
  SM0: [0,0] [0,1] [0,2] ...
  SM1: [1,0] [1,1] [1,2] ...     ← 同一行的 tile 分散在不同 SM
                                    L2 缓存命中率低

分组调度 (GROUP_SIZE_M=8):
  SM0: [0,0] [1,0] ... [7,0]
  SM1: [0,1] [1,1] ... [7,1]     ← 同一列的 tile 在同一 SM
                                    共享 B 矩阵, L2 命中率高
```

```python
def GroupedPersistentTileScheduler(GROUP_SIZE_M):
    # 将参数转为 constexpr, 以便在 kernel 中作为编译时常量使用
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    # 类似 C++ 模板的用法: 通过闭包捕获编译时参数
    @aggregate
    class GroupedPersistentTileSchedulerImpl:
        start_pid: gl.tensor
        num_pid_m: gl.tensor
        num_pid_in_group: gl.tensor
        num_pid: gl.tensor

        @gluon.constexpr_function
        def __init__(self, start_pid, num_pid_m, num_pid_in_group, num_pid):
            self.start_pid = start_pid
            self.num_pid_m = num_pid_m
            self.num_pid_in_group = num_pid_in_group
            self.num_pid = num_pid

        @gluon.jit
        def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
            start_pid = gl.program_id(axis=0)
            num_pid_m = gl.cdiv(M, BLOCK_M)
            num_pid_n = gl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_pid = num_pid_m * num_pid_n
            return GroupedPersistentTileSchedulerImpl(start_pid, num_pid_m, num_pid_in_group, num_pid)

        @gluon.jit
        def get_num_tiles(self):
            return gl.cdiv(self.num_pid - self.start_pid, gl.num_programs(axis=0))

        @gluon.jit
        def get_tile(self, idx):
            tile_id = self.start_pid + idx * gl.num_programs(axis=0)
            group_id = tile_id // self.num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % self.num_pid_in_group) // group_size_m
            return pid_m, pid_n

    GroupedPersistentTileSchedulerImpl.__name__ = f"GroupedPersistentTileScheduler({GROUP_SIZE_M.value})"
    return GroupedPersistentTileSchedulerImpl


# 把新的调度器也加入测试列表
schedulers += [GroupedPersistentTileScheduler(1), GroupedPersistentTileScheduler(8)]

if __name__ == "__main__" and not profiling_with_ncu:
    num_warps = 8 if is_hopper else 4
    num_buffers = 3 if is_hopper else 4
    print("Benchmarking grouped scheduler")
    print("=============================")
    print(f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K}")
    print(f"num_buffers={num_buffers} num_warps={num_warps}")
    print("GROUP_SIZE_M tflops/s")
    for GROUP_SIZE_M in [1, 2, 4, 6, 8]:
        print(f"{GROUP_SIZE_M:>12}", end=" ")
        fn = lambda: persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps,
                                       GroupedPersistentTileScheduler(GROUP_SIZE_M))
        ms = triton.testing.do_bench_cudagraph(fn)
        print(f"{get_flops(ms, M, N, K):8.2f}")
    print()
```

性能结果如下:

```
# 性能结果如下:
# GROUP_SIZE_M Blackwell  Hopper
#            1   1025.11  649.09
#            2   1050.43  651.32
#            4   1032.71  655.51
#            6   1057.27  652.39
#            8   1179.94  648.42
#
# 当 GROUP_SIZE_M=8 时, Blackwell 的性能恢复了。ncu 显示 L2 命中率提升到 70%, 说明调度策略确实影响了缓存效率, 还有进一步优化的空间。
#
# 但在 Hopper 上, 分组调度器反而导致性能下降。persistent kernel 的 L2 命中率为 86%, 而非持久化版本为 89%。
# 可能是负载不均衡问题, 因为分组调度本身不影响命中率, 但会加剧负载不均衡的问题。
```

### 外层循环流水线优化

我们可以通过延后TMA存储的wait, 让当前 tile 的存储操作与下一个 tile 的计算重叠。
这种跨 tile 的外层循环流水线优化方式对 K 较小的情况更有帮助, 因为此时 epilogue 占的比例更大。
tile 的时间 = 内层循环时间 + Epilogue 时间, 内层循环时间 ∝ K / BLOCK_K (迭代次数), Epilogue 时间 ≈ 固定(转换fp32→fp16 + TMA存储),
因此 K 越小, 隐藏 epilogue 开销的收益越大。

但这样做会让 TMA 存储缓冲区的生命周期与操作数缓冲区重叠, 因为原先A/B 和 C 的生命周期不重叠, C_buf可以和 A/B 共享同一块内存。
Hopper和Blackwell的共享内存大小都是228KB, 在配置: BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, dtype=fp16时,
A_buf = BLOCK_M × BLOCK_K × 2 bytes = 128 × 64 × 2  = 16 KB
B_buf = BLOCK_K × BLOCK_N × 2 bytes = 64 × 256 × 2  = 32 KB
C_buf = BLOCK_M × BLOCK_N × 2 bytes = 128 × 256 × 2 = 64 KB
原先(A+B)*4=192KB<228KB, 但是重叠后(A+B)*4+C=256KB>228KB, 因此最多只能用 3 个缓冲区≈208KB。
从前面的性能结果看, Hopper 用 3 个缓冲区问题不大, 但 Blackwell 可能会掉性能, 这是因为Blackwell 的 MMA 吞吐量更高, 需要更多缓冲区来隐藏延迟。

有三种解决方案:

1. 用 gl.store 直接写全局内存(不经过共享内存, 不需要C_buf), 但 gl.store 是同步的, 无法流水线化。而且布局转换仍需要用到共享内存。✗
2. 把 TMA 存储拆成多步, 用更小的缓冲区, 但只能对最后一步做流水线, 重叠程度降低。✗
3. 借用 B 缓冲区的空间。✓

对于 BLOCK_{M,N,K} = (128, 256, 64), 一个 B 缓冲区的大小是累加器的一半。
我们可以分配 5 个 B 缓冲区(内层循环只用 4 个), 多出来的 1 个给 epilogue 用。

```
STEALB 的缓冲区分配策略

目标: 4 个缓冲区用于内层循环 + 1 个缓冲区用于 epilogue

分配:
  A_buf[4]:  A0, A1, A2, A3       ← 4 个, 内层循环用
  B_buf[5]:  B0, B1, B2, B3, B4   ← 5 个！多出 1 个
                                 ↑
                        这个"多出来的"给 epilogue 用
  总计: 64 + 160 = 224 KB < 228 KB

内层循环:
  ─────────────────────────────────────────────────────────────────
  迭代 0: 用 A0, B0
  迭代 1: 用 A1, B1
  迭代 2: 用 A2, B2
  迭代 3: 用 A3, B3
  迭代 4: 用 A0, B4  ← 注意: A 用 %4, B 用 %5, 开始错开
  迭代 5: 用 A1, B0
  ...

Epilogue 时:
  ─────────────────────────────────────────────────────────────────
    producer 已经发完了所有 Load(包括下一个 tile 的预取)
    consumer 已经消费完了所有 MMA
    此时 producer % 5 指向的 B 缓冲区:
    上一次 MMA 已经消费完它了(已经空闲)

假设 epilogue 时 producer = 10

  producer % 5 = 0  →  B0

哪些 B 缓冲区是空闲的？
  - B4 在迭代 4 被用, 现在已经被 MMA 消费完了
  - 或者 B0,B1,B2 中某个已经被消费完的

代码中: c_buf = b_bufs.index(producer % (num_buffers + STEALB))
即: c_buf = b_bufs.index(8 % 5) = B3
但实际上代码会选择一个确保已经空闲的缓冲区
```

#### kernel解析

假设的参数
- num_buffers = 4
- STEALB = 1
- K = 512, BLOCK_K = 64 → 需要 8 次 K 维度迭代
- A 缓冲区: A[0], A[1], A[2], A[3] (索引 % 4)
- B 缓冲区: B[0], B[1], B[2], B[3], B[4] (索引 % 5)

##### 阶段 1: 初始 Prologue

```
# 循环 num_buffers-2 = 2 次
for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):  # ki = 0, 64
    producer = issue_loads_stealb(...)

# 再 Load 一次
k = BLOCK_K * (num_buffers - 2)  # k = 128
producer = issue_loads_stealb(...)
```

执行过程:
初始 Prologue
  操作     producer   A (%4)   B (%5)   说明
  Load     0 → 1      A0       B0       k=0
  Load     1 → 2      A1       B1       k=64
  Load     2 → 3      A2       B2       k=128
Prologue 结束: producer = 3, consumer = 0

##### 阶段 2: 主循环

```
for _ in range(num_tiles):  # 对于每个 tile
    # 先做一个 MMA
    consumer, mma = issue_mma_stealb(consumer, mma, ...)
    
    # 等待上一个 tile 的存储完成
    if STEALB:
        tma.store_wait(pendings=0)
    
    # 内循环主循环, Load 和 MMA 交替
    for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):  # k = 192, 256, 320, 384, 448
        producer = issue_loads_stealb(...)
        consumer, mma = issue_mma_stealb(...)
```

执行过程:
主循环(第一个 tile)
  操作     producer   consumer   A (%4)   B (%5)   k 值
  MMA          3      0 → 1      A0       B0       消费 k=0
  Load     3 → 4          1      A3       B3       k=192
  MMA          4      1 → 2      A1       B1       消费 k=64
  Load     4 → 5          2      A0       B4       k=256 (A/B 开始错开)
  MMA          5      2 → 3      A2       B2       消费 k=128
  Load     5 → 6          3      A1       B0       k=320
  MMA          6      3 → 4      A3       B3       消费 k=192
  Load     6 → 7          4      A2       B1       k=384
  MMA          7      4 → 5      A0       B4       消费 k=256
  Load     7 → 8          5      A3       B2       k=448
  MMA          8      5 → 6      A1       B0       消费 k=320
主循环结束: producer = 8, consumer = 6
注意: 当 producer=4 时, A 用 4%4=0 → A0, 但 B 用 4%5=4 → B4！A 和 B 开始错开了。

##### 阶段 3: Drain + 下一个 Tile 的 Prologue

```
# 融合: 当前 tile 的 drain(剩余 MMA)+ 下一个 tile 的 prologue(预取 Load)
for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):  # ki = 0, 64
    producer = issue_loads_stealb(...)  # 下一个 tile 的数据
    consumer, mma = issue_mma_stealb(...)  # 当前 tile 的剩余 MMA

k = BLOCK_K * (num_buffers - 2)  # k = 128
producer = issue_loads_stealb(...)  # 下一个 tile 的数据
```

执行过程:
Drain + Next Prologue
  操作     producer   consumer   A (%4)   B (%5)   说明
  Load     8 → 9          6      A0       B3       下个 tile k=0
  MMA          9      6 → 7      A2       B1       消费 k=384
  Load     9 → 10         7      A1       B4       下个 tile k=64
  MMA         10      7 → 8      A3       B2       消费 k=448
  Load    10 → 11         8      A2       B0       下个 tile k=128
Drain 结束: producer = 11, consumer = 8
当前 tile 的 8 次 MMA 全部完成

##### 阶段 4: Epilogue

```
mma = mma.wait_num_outstanding(0)  # 等待所有 MMA 完成
c, mma = mma.take_result()         # 取出累加器结果
c = c.to(dtype)                    # 转换为 fp16

# 选择哪个 B 缓冲区来存储 C
c_buf = b_bufs.index(producer % (num_buffers + STEALB))._reinterpret(...)
# producer = 11, 11 % 5 = 1 → 选择 B1
```

此时 producer = 11, 11 % 5 = 1 → 选择 B1

问题: B1 此时是空闲的吗？
B1 的生命周期追踪

B1 被 Load 使用的时机:
  第 1 次: producer = 1 时 (初始 prologue, k=64)
  第 2 次: producer = 6 时 (主循环, k=384)
  第 3 次: producer = 11 时... 还没发生 ← 这正是 epilogue 的时刻

B1 被 MMA 消费的时机:
  第 1 次: consumer = 1 时 (消费 k=64 的数据)
  第 2 次: consumer = 6 时 (消费 k=384 的数据) ← Drain 阶段第一步

Epilogue 时刻 (producer=11, consumer=8):
  B1 上次被 Load: producer = 6 时, 很久以前
  B1 上次被 MMA: consumer = 6 时, 已经消费完了
  B1 下次被 Load: producer = 11 时, 就是现在, 但 Load 还没发出去, 先借用它存储 C
  → B1 现在是空闲的, 可以安全借用

原理:
1. A 用 % 4, B 用 % 5, 循环周期不同
2. producer 总是领先 consumer: Prologue 预取 3 个, 主循环中 Load 和 MMA 交替保持领先
3. 当 epilogue 发生时, producer 指向"下一个要 Load 的位置", 但 Load 还没发出去,
   而该位置对应的 B 缓冲区其 MMA 早就完成了(consumer 已经追上了之前的 Load)
4. 当 producer = P 时, 选择 B[P % 5], 该缓冲区上次被 MMA 消费是在 consumer = P - 5 时,
   而此时 consumer ≈ P - 3, P - 3 > P - 5, 所以 MMA 早就完成了

问题 1: C 是 B 的两倍, 为什么只用 B1？
代码中的断言:
 gl.static_assert(2 * BLOCK_N * BLOCK_K >= BLOCK_M * BLOCK_N, "B tile not large enough to steal")
因此实际上 _reinterpret 会使用连续的两个 B 缓冲区空间！
所以 B1 和 B2 都必须是空闲的！
验证 epilogue 时刻 B1 和 B2 的状态:
Epilogue 时各 B 缓冲区状态 (producer=11):
  缓冲   最后被 Load (producer)   最后被 MMA (consumer)     状态
  B0     producer=10 (下tile)     consumer=5              正在用
  B1     producer=6  (k=384)      consumer=6              空闲 ← 可借用
  B2     producer=7  (k=448)      consumer=7              空闲 ← 可借用
  B3     producer=8  (下tile)     consumer=3 (旧)         正在用
  B4     producer=9  (下tile)     consumer=4 (旧)         正在用
B1 和 B2 都是空闲的, 可以合并使用

问题 2: TMA store 不会和下一个 tile 的 Load 重叠吗？
答案在代码中的 tma.store_wait(pendings=0):

```
for _ in range(num_tiles):
    consumer, mma = issue_mma_stealb(...)  # (1) 先做一个 MMA(消费 B3)
    if STEALB:
        tma.store_wait(pendings=0)  # (2) 等待上一个 tile 的 TMA store 完成
    for k in range(...):
        producer = issue_loads_stealb(...)  # (3) 这里才开始新的 Load
```

时间线:
TMA store (B1+B2 → Global): ========
MMA (消费 B3):                   ====
                                      ↑ store_wait 点
Load (写入 B1):                      ====

#### 实现代码

支持"借用 B 缓冲区"(stealb)的 issue_loads 和 issue_mma 变体版本

```python
@gluon.jit
def issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, stealb: gl.constexpr,
                       num_buffers: gl.constexpr, pred=True):
    index = producer % num_buffers
    b_index = producer % (num_buffers + stealb)
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred=pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(b_index), pred)
    return producer


@gluon.jit
def issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, stealb: gl.constexpr, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    b_index = consumer % (num_buffers + stealb)
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(b_index))
    return consumer, mma


@gluon.jit
def persistent_matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, SchedulerImpl: gl.constexpr,
                                       num_buffers: gl.constexpr, STEALB: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # 所有缓冲区的生命周期相同, 一起分配
    gl.static_assert(num_buffers >= 3, "expected at least 3 buffers")
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    # 如果启用 stealb, 多分配一个 B 缓冲区供 epilogue 使用
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers + STEALB] + b_desc.block_type.shape, b_desc.layout)
    if not STEALB:
        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    else:
        gl.static_assert(2 * BLOCK_N * BLOCK_K >= BLOCK_M * BLOCK_N, "B tile not large enough to steal")
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # 把内层循环的序言(prologue)单独提出来处理
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, ki, bars, a_bufs, b_bufs, STEALB,
                                      num_buffers)
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB, num_buffers)

    for _ in range(num_tiles):
        consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)
        if STEALB:
            # 在发起下一轮 TMA 加载之前, 先等待上一个 tile 的存储完成
            tma.store_wait(pendings=0)
        # steady阶段: Load 和 MMA 交替
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB,
                                          num_buffers)
            consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        # 把下一个 tile 的序言剥离出来, 与当前 tile 的流水线排空(drain)阶段融合
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # 用谓词(predicate)控制是否执行, 避免使用 if 分支
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            # 下一个 tile 的数据
            producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, ki, bars, a_bufs, b_bufs, STEALB,
                                          num_buffers, pred)
            # 当前 tile 的剩余 MMA
            consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)
        k = BLOCK_K * (num_buffers - 2)
        # 下一个 tile 的数据
        producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB,
                                      num_buffers)

        mma = mma.wait_num_outstanding(0)
        c, mma = mma.take_result()
        c = c.to(dtype)
        if not STEALB:
            c_buf = c_smem
            tma.store_wait(pendings=0)
        else:
            # 借用当前空闲的 B 缓冲区作为 C 的存储缓冲区
            c_buf = b_bufs.index(producer % (num_buffers + STEALB))._reinterpret(dtype, c_desc.block_type.shape,
                                                                                 c_desc.layout)
        c_buf.store(c)
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [epilogue_off_m, epilogue_off_n], c_buf)
    tma.store_wait(pendings=0)


def persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    persistent_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, SchedulerImpl, num_buffers,
                                             STEALB=num_buffers == 4, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("SchedulerImpl", schedulers)
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_persistent_matmul_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)
```

### 流水线化持久化矩阵乘法性能测试

```python
if __name__ == "__main__":
    args = {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "num_buffers": 3 if is_hopper else 4,
        "num_warps": 8 if is_hopper else 4,
    }
    scheduler = PersistentTileScheduler if is_hopper else GroupedPersistentTileScheduler(8)
    nonpersistent = partial(matmul_pipelined, **args)
    persistent = partial(persistent_matmul, **args, SchedulerImpl=scheduler)
    persistent_pipelined = partial(persistent_matmul_pipelined, **args, SchedulerImpl=scheduler)

    M, N = 8192, 8192
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    print("Benchmarking pipelined persistent")
    print("=================================")
    print("    K     nonpersistent    persistent   pipelined    cublas")
    for K in [2**i for i in range(9, 15)]:
        as_flops = partial(get_flops, M=M, N=N, K=K)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        BT = B.T.contiguous()
        r0 = as_flops(triton.testing.do_bench_cudagraph(lambda: nonpersistent(A, B, C)))
        r1 = as_flops(triton.testing.do_bench_cudagraph(lambda: persistent(A, B, C)))
        r2 = as_flops(triton.testing.do_bench_cudagraph(lambda: persistent_pipelined(A, B, C)))
        r3 = as_flops(triton.testing.do_bench(lambda: cublas.matmul(A, BT, C)))
        print(f"{K:>5} {r0:>17.2f} {r1:>13.2f} {r2:>11.2f} {r3:>9.2f}")
```

性能结果:

```
# Blackwell 结果:
#
#     K     nonpersistent    persistent   pipelined    cublas
#   512            615.86        828.70      993.50   1108.11
#  1024            997.16       1077.28     1173.31   1347.44
#  2048           1152.74       1190.55     1133.37   1435.01
#  4096           1164.05       1120.92     1143.47   1563.98
#  8192           1160.93       1074.97     1185.40   1491.84
# 16384           1185.62       1096.34     1296.93   1548.42
#
# Hopper 结果:
#
#     K     nonpersistent    persistent   pipelined    cublas
#   512            491.74        485.01      539.88    588.15
#  1024            554.24        575.02      602.52    588.32
#  2048            573.87        594.72      625.91    615.58
#  4096            609.36        630.10      640.48    646.30
#  8192            629.44        646.22      661.57    661.11
# 16384            653.79        660.29      670.00    665.49
#
# 正如预期, K 较小时, 流水线化的 persistent matmul 相比非持久化版本的提升更明显。
# 当 tile 数量不能被 SM 数量整除时, 负载均衡会比较困难。对于 8192x8192 的矩阵,
# 在 Hopper 和 Blackwell 上分别是每个 SM 约 13.5 和 15.5 个 tile, 正好处于尴尬的中间地带。
#
# 在 Hopper 上, 我们的流水线 kernel 与 cuBLAS 性能相当, 中等 K 时甚至略胜一筹。
# 但 cuBLAS 在小 K 时优势明显。在 Blackwell 上差距更大: cuBLAS 快很多。
```

### 总结

关于 matmul 性能的一些观察:

- 在 Hopper 上, 软件流水线足以在中等和较大 K 时达到峰值性能。
- cuBLAS 使用 2-CTA matmul, 借助分布式共享内存实现 256x256 的指令形状,
  能够更高效地给 MMA 喂数据。这一点在 Blackwell 上尤为重要, 因为 MMA 相对于
  TMA 的吞吐量比例更高了。(Gluon 对 2-CTA 的支持还不太稳定。)
- cuBLAS 的 matmul 采用 warp 专业化设计, 这对于在小 K 时完全隐藏 epilogue 开销是必要的。
- 我们的 Blackwell 实现受限于为兼容 Hopper 而设计的统一 API:
  没有对累加器做双缓冲, 也没有用满 TMEM 的 256 列。
- 在 Blackwell 上, 可以使用 `clusterlaunchcontrol` 实现与 GPU 协同的动态调度,
  兼顾静态调度的优化潜力和动态调度的负载均衡能力。

要点:

- Persistent kernel 用静态调度替代了 GPU 原生的块调度, 允许在不同 tile 之间
  更好地协调资源和重叠计算, 代价是失去了动态负载均衡的能力。
- Persistent kernel 对小规模问题尤其有效, 但对大规模问题也有一定收益。
