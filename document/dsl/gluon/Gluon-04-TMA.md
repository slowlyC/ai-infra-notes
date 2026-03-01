## Gluon-04-TMA

### 前言

本系列是 Gluon 的学习笔记，基于 [Gluon 官方教程](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon) 整理，覆盖了从基础概念到高级优化的完整内容，在原文基础上做了结构重组，并补充了 CUDA 编程模型和 GPU 硬件架构层面的说明。

> 知乎专栏：[DSL教程](https://www.zhihu.com/column/c_1990516179064858333)
> 完整代码：[GitHub](https://github.com/slowlyC/ai-infra-notes/tree/main/tutorials/gluon)

- [Gluon-01-Overview](https://zhuanlan.zhihu.com/p/1990504603008119998)
- [Gluon-02-Layout_Introduction](https://zhuanlan.zhihu.com/p/1990509278801457706)
- [Gluon-03-Async_Copy](https://zhuanlan.zhihu.com/p/1990517098083029502)
- **Gluon-04-TMA**（本文）
- [Gluon-05-wgmma](https://zhuanlan.zhihu.com/p/1990835358502523949)
- [Gluon-06-tcgen05](https://zhuanlan.zhihu.com/p/1990835546797405057)
- [Gluon-07-Persistent_Kernel](https://zhuanlan.zhihu.com/p/2003592603732563562)
- [Gluon-08-Warp_Specialization](https://zhuanlan.zhihu.com/p/2003602919912649692)
- [Gluon-09-TMA_Gather_Scatter](https://zhuanlan.zhihu.com/p/2003599190585017693)
- [Gluon-10-tcgen05_copy](https://zhuanlan.zhihu.com/p/2003603029975381459)
- [Gluon-11-tcgen05-mma-scaled](https://zhuanlan.zhihu.com/p/2003603119259550000)

全局内存访问的主要瓶颈通常是寄存器压力. 对于每一条 `LDG.E` 或 `STG.E` 指令, 我们都需要计算一个 64 位地址,

有时还需要计算掩码, 并将结果存储在寄存器中. 虽然向量化可以减轻部分压力, 但并未根除这一问题.

在 Hopper 架构及之后的 GPU 上, TMA (Tensor Memory Accelerator) 专门用于在全局内存中对 N 维数组进行寻址.

TMA 通过使用更紧凑的地址表示形式——"Tensor Descriptor" (Tensor 描述符), 

牺牲了常规全局内存指令的部分寻址灵活性, 换取了更高的访问效率.

此外, TMA 内存事务由一条称为 "Async Proxy" (异步代理) 的独立硬件路径处理. 

这虽然提升了全局内存访问的性能, 但也引入了额外的同步需求.

在本教程中, 我们将介绍如何在 Gluon 中使用 TMA, 演示其如何提升性能, 并讲解如何对 TMA 操作进行流水线化.

### tma kernel示例

TMA 通过"Tensor Descriptor" (Tensor 描述符) 对象来使用. 

Tensor Descriptor 驻留在全局内存中, 包含了 Tensor 的形状, 步幅, 基指针, Layout 以及其他元数据. 

TMA 的读写操作本质上是异步的, 因此我们需要使用 "mbarrier" 对象来进行同步.

使用TMA的Kernel会接收Tensor Descriptor作为参数, 不需要再传递Tensor的步幅, 因为其已经存储在Tensor Descriptor中了.

我们以memcpy_1d为例, 演示如何使用TMA进行内存拷贝.

```
import pytest
import torch
import triton
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

# 复用上一个教程中的部分函数。
t3 = importlib.import_module("03-async-copy")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")


@gluon.jit
def memcpy_1d_tma_kernel(in_desc, out_desc, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)

    # Tensor Descriptor 的 NVMMASharedLayout 中包含了共享内存所需的 SwizzledSharedLayout，直接复用即可。
    smem_layout: gl.constexpr = in_desc.layout
    smem = gl.allocate_shared_memory(in_desc.dtype, [XBLOCK], smem_layout)

    # 异步 TMA 读取的完成状态由 mbarrier 对象跟踪. mbarrier 是一个 64 位的计数器, 驻留在共享内存中.
    #
    # mbarrier 工作原理 (理解为"倒计时闸门"):
    # 1. 初始化: 设置一个计数值 (比如 count=1)
    # 2. Arrive: 每次有操作完成时, 计数减 1
    # 3. 完成: 当计数减到 0 时, 当前阶段完成, 自动进入下一个阶段
    #
    # 重要限制: mbarrier 只能记住 2 个阶段的状态 (当前阶段 + 上一阶段).
    # 如果生产者太快、消费者太慢, 导致阶段号差距超过 1, 就会失去同步.

    # 0. 在共享内存中分配一个 mbarrier 对象, 用于跟踪TMA读取的完成状态.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    # 1. 初始化 mbarrier
    # TMA 操作完成时会自动触发一次 Arrive (计数减 1)，因此我们用 count=1 初始化 mbarrier。
    mbarrier.init(bar, count=1)

    # 检查输入和输出Tensor的块形状和Layout是否一致.
    # 我们不需要再手动计算mask，因为Tensor Descriptor中定义了每次复制的块形状 (Block Shape)，
    # 只需要指定"第几个块" (坐标偏移), 如果超出块的边界, TMA 会根据块形状自动处理掩码, 防止非法访问。
    gl.static_assert(in_desc.block_type == out_desc.block_type)
    gl.static_assert(in_desc.layout == out_desc.layout)

    # 2. 发起 TMA 操作
    # TMA 通过字节计数来跟踪数据传输进度:
    # - mbarrier.expect(bar, N)  - 告诉 barrier: "等待 N 个字节传输完成"
    # - TMA 硬件每传输完一些数据    - 自动递减字节计数
    # - 字节计数减到 0             - 触发一次 Arrive (让 mbarrier 的计数减 1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(in_desc, [pid * XBLOCK], bar, smem)

    # 3. 等待数据读取完成: 通过奇偶校验的阶段号 (phase) 来查询.
    # - mbarrier 初始状态: phase 1 已完成, 正在等待 phase 0 完成
    # - 所以我们 wait(phase=0), 即等待当前正在进行的 phase 0 完成
    mbarrier.wait(bar, phase=0)

    # 当我们使用完 mbarrier 后, 需要将其置为无效.
    mbarrier.invalidate(bar)

    # TMA 可以直接从共享内存存储数据到全局内存, 不需要经过寄存器, 节省了一次寄存器到共享内存的数据搬运.
    tma.async_copy_shared_to_global(out_desc, [pid * XBLOCK], smem)

    # TMA 读取 vs 存储的同步机制不同:
    # - TMA 读取: 用 mbarrier 跟踪完成状态
    # - TMA 存储: 用 "Commit Group" (提交组) 跟踪, 与异步copy的机制类似
    # 注意: 异步复制和 TMA 存储的Commit Group是分开的, 互不影响.
    #
    # store_wait(pendings=0) 的含义: 等待直到"未完成的存储操作数量 ≤ 0", 即全部完成.
    tma.store_wait(pendings=0)


def memcpy_1d_tma(input, output, XBLOCK=8192):
    assert input.shape == output.shape

    # Tensor Descriptor的Layout始终是 NVMMASharedLayout. 
    # 可以使用get_default_for函数获取默认的 NVMMASharedLayout, 但有时可能需要自定义Tensor Descriptor的Layout.
    block_shape = [XBLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    # 将输入和输出Tensor分别包装在 Tensor Descriptor 中.
    in_desc = TensorDescriptor.from_tensor(input, block_shape, layout)
    out_desc = TensorDescriptor.from_tensor(output, block_shape, layout)

    grid = (triton.cdiv(input.numel(), XBLOCK), )
    # num_warps=1, 是因为发起TMA指令只需要一个线程（或一个 Warp）。一旦指令发出，CUDA Core 就可以去休息或者干别的事了。
    # 因为所有繁重工作（数据移动）都卸载给了TMA硬件，剩下的指令发射工作量极小且无法并行化，所以只需要一个Warp来“发号施令”即可。
    memcpy_1d_tma_kernel[grid](in_desc, out_desc, XBLOCK, num_warps=1)


@pytest.mark.parametrize("XBLOCK", [64])
@pytest.mark.parametrize("xnumel", [40, 500])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_memcpy_1d_tma(XBLOCK, xnumel):
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy_1d_tma(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
```

### 逐元素加法-TMA版本

接下来让我们使用 TMA 重写流水线优化的逐元素加法 Kernel。
Kernel 的结构几乎相同，主要变化:
- 为每个缓冲区分配一个 mbarrier 来跟踪读取完成状态。
- 使用 TMA 进行存储, 因此需要为输出c分配共享内存, 因为要先将计算后的c从寄存器存到共享内存, 再通过TMA到全局内存.

GPU 访问共享内存有两条独立的硬件路径
- Async Proxy (异步代理):  TMA 指令专用通道
- Generic Proxy (通用代理): 普通 load/store 指令使用

注意: 这两条通道的操作是**乱序执行**的。 
解决: 使用 `fence_async_shared()` 来强制排序, 确保操作按预期顺序完成.
注: 这里确保的是共享内存的访问顺序, 而不是全局内存的访问顺序。

以下是需要使用 Fence (栅栏) 的危险场景:

【危险场景 1】Generic 读 → Async 写 (RAW 冲突)
```python
value = smem.load()                                       # Generic Proxy: 从 smem 读
fence_async_shared()                                      # ← 必须加 Fence!
tma.async_copy_global_to_shared(desc, [0, 0], bar, smem)  # Async Proxy: 向 smem 写
```
没有 Fence 的后果: TMA 可能在 load 还没读完时就开始写入, 导致读到脏数据.

【危险场景 2】Generic 写 → Async 读 (WAR 冲突)
```python
smem.store(value)                                    # Generic Proxy: 向 smem 写
fence_async_shared()                                 # ← 必须加 Fence!
tma.async_copy_shared_to_global(desc, [0, 0], smem) # Async Proxy: 从 smem 读
```
没有 Fence 的后果: TMA 可能在 store 还没写完时就开始读取, 读到旧数据.

【安全场景】某些操作自带同步保证, 不需要额外 Fence
例如: 等待 TMA 写入共享内存后，再读取共享内存。
```python
tma.async_copy_global_to_shared(desc, [0, 0], bar, smem)  # Async Proxy: 写入 smem
mbarrier.wait(bar, phase=0)                               # 等待 TMA 完成
value = smem.load()                                       # Generic Proxy: 从 smem 读
```
为什么不需要 Fence? 
- `mbarrier.wait` 保证了 TMA 写入已经完成
- 后续的 load 操作在等待之后执行, 顺序已经确定

注意: 这个保证**仅适用于 TMA 读取**。 如果是手动 arrive, 仍需要 Fence:

【仍需 Fence 的场景】手动 arrive + wait 不能保证跨代理同步
```python
smem.store(value)                                    # Generic Proxy: 写入 smem
mbarrier.arrive(bar, count=1)                        # 手动触发 arrive
mbarrier.wait(bar, phase=0)                          # 等待阶段完成
fence_async_shared()                                 # ← 仍然需要 Fence!
tma.async_copy_shared_to_global(desc, [0, 0], smem) # Async Proxy: 从 smem 读
```
原因: `mbarrier.wait` 只保证 mbarrier 本身的阶段完成, 但不保证 Generic Proxy 的 store 一定完成。

```

@gluon.jit
def issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr):
    # 使用同一个 mbarrier 同时跟踪 a 和 b 两个 TMA 读取的完成状态.
    yoff = copy_index * YBLOCK
    bar = bars.index(copy_index % num_buffers)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [xoff, yoff], bar, a_smem.index(copy_index % num_buffers))
    tma.async_copy_global_to_shared(b_desc, [xoff, yoff], bar, b_smem.index(copy_index % num_buffers))
    return copy_index + 1


@gluon.jit
def perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout: gl.constexpr, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr):
    # 等待前面第 `num_buffers-1` 次迭代发起的复制完成.
    read_phase = read_index // num_buffers & 1
    mbarrier.wait(bars.index(read_index % num_buffers), read_phase)
    a_val = a_smem.index(read_index % num_buffers).load(layout)
    b_val = b_smem.index(read_index % num_buffers).load(layout)
    c_val = a_val + b_val
    yoff = read_index * YBLOCK
    # 采用滚动等待策略流水线化存储: 等待上一次存储完成再开始新的存储.
    tma.store_wait(pendings=0)
    c_smem.store(c_val)
    fence_async_shared()
    # 发起异步存储, 无需等待完成, 继续执行后续操作.
    tma.async_copy_shared_to_global(c_desc, [xoff, yoff], c_smem)
    return read_index + 1


@gluon.jit
def elementwise_add_tma_kernel(  #
        a_desc, b_desc, c_desc, xnumel, ynumel,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoff = pid * XBLOCK

    dtype: gl.constexpr = a_desc.type.block_type.element_ty
    # 为输入 a 和 b 分配多重缓冲共享内存, 实现流水线并行.
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], b_desc.layout)

    # 为输出 c 分配共享内存, 用于 TMA 存储. 为什么这里 C 不用多重缓冲?
    # 因为 C 的写入使用了滚动等待策略, 每次写入前等待上一次完成，所以只需要 1 个缓冲区。
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], c_desc.layout)

    # 为每个缓冲区分配一个 mbarrier, 用于跟踪 TMA 读取完成状态.
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)

    copy_index = 0
    read_index = 0

    for _ in gl.static_range(num_buffers - 1):
        copy_index = issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers)

    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        copy_index = issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers)
        read_index = perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout, YBLOCK, num_buffers)

    for _ in gl.static_range(num_buffers - 1):
        read_index = perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout, YBLOCK, num_buffers)

    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(bars.index(i))

    # 确保所有存储操作都已完成.
    tma.store_wait(pendings=0)


def elementwise_add_tma(a, b, c, XBLOCK=32, YBLOCK=64, num_buffers=2):
    assert a.shape == b.shape == c.shape
    xnumel, ynumel = a.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )

    block_shape = [XBLOCK, YBLOCK]
    # TMA Descriptor 必须使用 NVMMASharedLayout.
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    # 创建 TMA Descriptor 时, 步幅 (Stride) 必须满足 16 字节对齐.
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)
    elementwise_add_tma_kernel[grid](a_desc, b_desc, c_desc, xnumel, ynumel, XBLOCK, YBLOCK, num_buffers)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 120)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 64)])
@pytest.mark.parametrize("num_buffers", [1, 2, 3])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_elementwise_add_pipelined(xnumel, ynumel, XBLOCK, YBLOCK, num_buffers):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add_tma(a, b, c, XBLOCK, YBLOCK, num_buffers)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)
```

### tma与异步拷贝版本性能对比

下面将TMA版本的流水线 Kernel, 与上个教程中异步拷贝版本的流水线 Kernel 进行性能对比.

```

if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    XBLOCK = 32
    YBLOCK = 64
    num_buffers = 2

    ms = triton.testing.do_bench(lambda: t3.elementwise_add_pipelined(A, B, C, XBLOCK, YBLOCK, num_buffers))
    print(f"elementwise_add_pipelined: {t3.get_throughput(ms, C):.2f} TB/s")

    ms = triton.testing.do_bench(lambda: elementwise_add_tma(A, B, C, XBLOCK, YBLOCK, num_buffers))
    print(f"elementwise_add_tma: {t3.get_throughput(ms, C):.2f} TB/s")
```

结果如下：

elementwise_add_pipelined: 4.20 TB/s

elementwise_add_tma: 5.50 TB/s

使用 TMA 带来了显著的性能提升.

### TMA kernel调优

接下来我们进一步调优，由于 kernel 的寄存器压力降低, 现在可以增加块大小 (Block Size)了。

实际上, TMA版本的 Kernel 峰值寄存器使用率会保持在较低水平, 因为编译器会在内部循环中交错执行共享内存加载、加法计算和共享内存存储。

此时块大小的主要限制是共享内存容量。

在B200中每个 SM 有 228 KB 的共享内存。如果使用 128x128xf32 的块, 共享内存不足以对输入进行双重缓冲 (Double Buffer)。

而使用 64x128xf32 的块, 三重缓冲 (Triple Buffer) 将占用 224 KB, 再加上一些mbarrier等额外开销, 刚好够用。

详细计算过程如下:

情况 1: 128×128×f32 块 + 双重缓冲

单个块 = 128 × 128 × 4 bytes = 65,536 bytes = 64 KB

总共享内存需求 = A的缓冲 + B的缓冲 + 输出C的缓冲 = (64 KB × 2) + (64 KB × 2) + (64 KB × 1) = 320 KB

情况 2: 64×128×f32 块 + 三重缓冲

单个块 = 64 × 128 × 4 bytes = 32,768 bytes = 32 KB

总共享内存需求 = A的缓冲 + B的缓冲 + 输出C的缓冲 = (32 KB × 3) + (32 KB × 3) + (32 KB × 1) = 224 KB

```
if __name__ == "__main__":
    XBLOCK = 64
    YBLOCK = 128
    num_buffers = 3
    ms = triton.testing.do_bench(lambda: elementwise_add_tma(A, B, C, XBLOCK, YBLOCK, num_buffers))
    print(f"elementwise_add_tma (64x128x3): {t3.get_throughput(ms, C):.2f} TB/s")
```

性能结果如下：

```
elementwise_add_tma (64x128x3): 5.90 TB/s
```

通过增大块大小和流水线深度, 性能得到了进一步提升。

### 总结:

- TMA 使用独立且更快的硬件通道在共享内存和全局内存之间传输数据。
- TMA 指令是异步的: 使用 mbarrier 跟踪读取完成, 使用 Commit Group (提交组) 跟踪存储完成。
- TMA 降低了寄存器压力, 但也限制了寻址灵活性。某些 Tensor Layout 下可能无法使用 TMA。
- TMA 指令支持流水线化, 但需要在异步代理 (Async Proxy) 和通用代理 (Generic Proxy) 之间显式同步。