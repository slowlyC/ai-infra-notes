## Gluon-03-Async_Copy

### 前言

本系列是 Gluon 的学习笔记，基于 [Gluon 官方教程](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon) 整理，覆盖了从基础概念到高级优化的完整内容，在原文基础上做了结构重组，并补充了 CUDA 编程模型和 GPU 硬件架构层面的说明。

> 知乎专栏：[DSL教程](https://www.zhihu.com/column/c_1990516179064858333)
> 完整代码：[GitHub](https://github.com/slowlyC/ai-infra-notes/tree/main/tutorials/gluon)

- [Gluon-01-Overview](https://zhuanlan.zhihu.com/p/1990504603008119998)
- [Gluon-02-Layout_Introduction](https://zhuanlan.zhihu.com/p/1990509278801457706)
- **Gluon-03-Async_Copy**（本文）
- [Gluon-04-TMA](https://zhuanlan.zhihu.com/p/1990517971483906093)
- [Gluon-05-wgmma](https://zhuanlan.zhihu.com/p/1990835358502523949)
- [Gluon-06-tcgen05](https://zhuanlan.zhihu.com/p/1990835546797405057)
- [Gluon-07-Persistent_Kernel](https://zhuanlan.zhihu.com/p/2003592603732563562)
- [Gluon-08-Warp_Specialization](https://zhuanlan.zhihu.com/p/2003602919912649692)
- [Gluon-09-TMA_Gather_Scatter](https://zhuanlan.zhihu.com/p/2003599190585017693)
- [Gluon-10-tcgen05_copy](https://zhuanlan.zhihu.com/p/2003603029975381459)
- [Gluon-11-tcgen05-mma-scaled](https://zhuanlan.zhihu.com/p/2003603119259550000)

现代 GPU 为长延迟操作（如全局内存访问）提供异步指令。异步操作允许内存传输与计算重叠执行，这种技术称为"流水线化"（pipelining）。

不同 GPU 厂商和架构的异步指令各有不同，本教程聚焦于 NVIDIA GPU。在 NVIDIA GPU 上，异步拷贝在全局内存和共享内存之间传输数据，与直接读写寄存器的 `gl.load`/`gl.store` 不同。

### 异步拷贝的基本用法

```python
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp


def is_ampere_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8


if __name__ == "__main__" and not is_ampere_or_newer():
    raise RuntimeError("This tutorial requires Ampere or newer NVIDIA GPU")

# `cp.async`指令用于将数据从全局内存异步拷贝到共享内存, 我们用`cp.async`来重新实现1D memcpy。
# Gluon 中通过gl.allocate_shared_memory分配共享内存，分配时需要指定 layout。
# 共享内存的layout为SwizzledSharedLayout, 主要用于减少 bank 冲突，有时也是某些操作的硬性要求。

@gluon.jit
def memcpy_1d_cpasync_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)

    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel

    # 申请共享内存，layout为SwizzledSharedLayout
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)

    # 发起异步拷贝：全局内存 → 共享内存
    cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
    # commit_group：把之前发出的所有异步拷贝打包成一个"组"
    cp.commit_group()

    # wait_group(0)：阻塞等待，直到未完成的组数为 0
    cp.wait_group(0)

    value = smem.load(layout)
    gl.store(out_ptr + offsets, value, mask=mask)


def memcpy_1d_cpasync(input, output, XBLOCK=8192, num_warps=4):
    grid = (triton.cdiv(input.numel(), XBLOCK), )
    memcpy_1d_cpasync_kernel[grid](input, output, input.numel(), XBLOCK, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, XBLOCK", [(200, 128), (1000, 256)])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_memcpy_1d_cpasync(xnumel, XBLOCK):
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy_1d_cpasync(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
```

### 常规的逐元素加法 kernel 

接下来让我们用逐元素加法 kernel 来逐步进行异步拷贝和流水线优化。

先写一个常规的同步版本kernel作为baseline。

```python
@gluon.jit
def elementwise_add_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
):
    pid = gl.program_id(0)

    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))  # 行索引

    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))  # 列索引
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        b_val = gl.load(b_ptrs + ystride_b * yoffs[None, :], mask=mask)

        c_val = a_val + b_val

        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add(A, B, C, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_add_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 32), (128, 128)])
def test_elementwise_add(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add(a, b, c, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)
```

### 逐元素加法（异步拷贝版）

用`cp.async`异步拷贝重写这个 kernel（暂不做流水线优化）。

```
@gluon.jit
def elementwise_add_cpasync_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    # 为 A、B 分配共享内存
    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        # 异步加载A、B数据到共享内存
        cp.async_copy_global_to_shared(a_smem, a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        cp.async_copy_global_to_shared(b_smem, b_ptrs + ystride_b * yoffs[None, :], mask=mask)
        cp.commit_group()  # 把两个异步拷贝请求操作打包成一组
        cp.wait_group(0)   # 等待该组完成

        # 从共享内存读到寄存器
        a_val = a_smem.load(layout)
        b_val = b_smem.load(layout)

        c_val = a_val + b_val

        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add_cpasync(A, B, C, smem_layout, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_add_cpasync_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, smem_layout)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 32), (128, 128)])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_elementwise_add_cpasync(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    elementwise_add_cpasync(a, b, c, smem_layout, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


def get_throughput(ms, C):
    # 这个 kernel 是 memory-bound 的，用带宽来衡量性能
    tbytes = (3 * C.numel() * C.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    ms = triton.testing.do_bench(lambda: elementwise_add(A, B, C))
    print(f"elementwise_add: {get_throughput(ms, C):.2f} TB/s")

    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    ms = triton.testing.do_bench(lambda: elementwise_add_cpasync(A, B, C, smem_layout))
    print(f"elementwise_add_cpasync: {get_throughput(ms, C):.2f} TB/s")

```

性能测试：

```
elementwise_add: 1.48 TB/s
elementwise_add_cpasync: 3.97 TB/s
```

cpasync 版本快了很多！

注意到这里我们选择了SwizzledSharedLayout的共享内存layout。

为了提高内存读写带宽，共享内存被分割成了32个等大小的内存块，即Bank，每个Bank的大小是4Byte(32bit,等于float类型大小)。

在较新的 GPU 上, bank 是双端口的, 允许在每个周期内为每个线程束提供两个4Byte的请求。任何超过此数量的请求都会导致 bank conflict。

Swizzling 通过位操作（如 XOR）重新映射地址，把原本会冲突的访问分散到不同 bank。

不过在我们的例子中，寄存器layout为BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])，

即32 个线程各处理 1 个 float，正好对应 32 个 bank，所以即使不 swizzle 也没有冲突。

然而在其他情况下, 例如使用bf16时, swizzling 对减少 bank 冲突至关重要。

### 流水线化（Pipelining）优化

基本思路：
- 异步加载下一个数据块的同时，计算当前数据块。这样可以将内存传输（copy）与计算（add）进行重叠。

流水线深度（num_buffers）:
- 表示同时进行的异步拷贝数量. 
- 当 copy 耗时是 add 的 N 倍时，使用 num_buffers = N 个缓冲区，就能达到最优的重叠效果。
  例如：如果 copy 耗时是 add 的 2 倍，用 num_buffers=2 就能让两者完全重叠。
- 代价是需要更多共享内存（每个 buffer 需要 [XBLOCK, YBLOCK] 的空间）

下面我们举例说明:

```
## ══════════════════════════════════════════════════════════════════════════════
# 场景1: copy=1T, add=1T, num_buffers=1(add可以异步执行)
# ═══════════════════════════════════════════════════════════════════════════════
#  时间   0      1      2      3      4      5      6      7      8      9
#        |------|------|------|------|------|------|------|------|------|------|
#  copy  |copy0 |copy1 |copy2 |copy3 |
#  add          |add(0)|add(1)|add(2)|add(3)|
         
#  每个数据块耗时 = 1T(copy) - 1T(add重叠) + 1T(add) ≈ 1T
#  当copy和add时间相同时，一个buffer就能让两者完全重叠, 但实际上往往copy时间更长。
#
## ══════════════════════════════════════════════════════════════════════════════
# 场景2: copy=2T, add=1T, num_buffers=1(add可以异步执行)
# ═══════════════════════════════════════════════════════════════════════════════
#  每次 add 都要等1T个时间！
#
#  时间   0      1      2      3      4      5      6      7      8      9
#        |------|------|------|------|------|------|------|------|------|------|
#  Buf   |===copy(0)===|===copy(1)===|===copy(2)===|===copy(3)===|
#  add                 |add(0)|      |add(1)|      |add(2)|      |add(3)|
#                                 ↑             ↑              ↑             
#                               等待1T         等待1T         等待1T           
#
## ══════════════════════════════════════════════════════════════════════════════
# 场景3: copy=2T, add=1T, num_buffers=2(add可以异步执行)
# ═══════════════════════════════════════════════════════════════════════════════
# 
# 时间    0      1      2      3      4      5      6      7      8      9     
#        |------|------|------|------|------|------|------|------|------|------|  
# Buf0   |===copy(0)===|===copy(2)===|===copy(4)===|===copy(6)===|                  
# Buf1   |===copy(1)===|===copy(3)===|===copy(5)===|===copy(7)===|
# add                  |add(0)|add(1)|add(2)|add(3)|add(4)|add(5)|add(6)|add(7)|
#
## ══════════════════════════════════════════════════════════════════════════════
# 场景4: copy=3T, add=1T, num_buffers=3(add可以异步执行)
# ═══════════════════════════════════════════════════════════════════════════════
# 
# 时间    0      1      2      3      4      5      6      7      8      9     
#        |------|------|------|------|------|------|------|------|------|------|  
# Buf0   |=======copy(0)======|=======copy(3)======|=======copy(6)======|                
# Buf1   |=======copy(1)======|=======copy(4)======|=======copy(7)======|
# Buf2   |=======copy(2)======|=======copy(5)======|=======copy(8)======|
# add                         |add(0)|add(1)|add(2)|add(3)|add(4)|add(5)| 
#
#
# 上面场景假设copy和add都是可以异步执行的，在本节代码中add不能异步执行，因此需要考虑以下情况：
## ══════════════════════════════════════════════════════════════════════════════
# 场景5(本节代码实现): copy=2T, add=1T, num_buffers=2(add不能异步执行)
# ═══════════════════════════════════════════════════════════════════════════════
# 
# 时间    0      1      2      3      4      5      6      7      8      9     
#        |------|------|------|------|------|------|------|------|------|------|------|  
# Buf0   |===copy(0)===|      |===copy(2)===|      |===copy(4)===|      |===copy(6)===|     
#                             |add(1)|             |add(3)|             |add(5)|
# Buf1      |===copy(1)===|          |===copy(3)===|      |===copy(5)===|
#           |...........add(0)|      |.......add(2)|      |.......add(4)|
#
# add                  |add(0)|add(1)|      |add(2)|add(3)|      |add(4)|add(5)|
#
#
## ══════════════════════════════════════════════════════════════════════════════
# 场景6(本节代码实现): copy=2T, add=1T, num_buffers=3(add不能异步执行)
# ═══════════════════════════════════════════════════════════════════════════════
# 
# 时间    0      1      2      3      4      5      6      7      8      9     
#        |------|------|------|------|------|------|------|------|------|------|------|  
# Buf0   |===copy(0)===|      |===copy(3)===|      |===copy(6)===|      |===copy(9)===|     
#                             |add(1)|             |add(4)|             |add(7)|
# Buf1   |===copy(1)===|             |===copy(4)===|      |===copy(7)===|
#                                    |add(2)|             |add(5)|
# Buf2      |===copy(2)===|                 |===copy(5)===|      |===copy(8)===|
#           |...........add(0)|             |add(3)|             |add(6)|
#
# add                  |add(0)|add(1)|add(2)|add(3)|add(4)|add(5)|add(6)|add(7)|
#
#
# 【为什么Warmup/Cooldown阶段是 num_buffers-1？】
# 提前"发射"num_buffers - 1个copy, 这样当 add(i) 完成时，copy(i+1) 刚好也完成，无缝衔接！

# 示例1（num_buffers=2，共10个数据块）：
#   【Warmup】发出 1 个copy:
#     copy(0)
#   【Steady】8次迭代（10-2=8）:
#     迭代0: copy(1) → wait copy(0) → add(0)  # copy(0)和copy(1)在同时进行，等待copy(0)完成后，执行add(0)
#     迭代1: copy(2) → wait copy(1) → add(1)  # copy(1)和copy(2)在同时进行，等待copy(1)完成后，执行add(1)执行
#     ...
#     迭代7: copy(9) → wait copy(8) → add(8)  # copy(8)和copy(9)在同时进行，等待copy(8)完成后，执行add(8)执行
#   【Cooldown】1次迭代（2-1=1）:
#     wait copy(9) → add(9)  # 等待最后的copy(9)完成后，执行add(9)
#
# 示例2（num_buffers=3，共10个数据块）：
#   【Warmup】发出 2 个copy:
#     copy(0)
#     copy(1)
#   【Steady】7次迭代（10-3=7）:
#     迭代0: copy(2) → wait copy(0) → add(0)  # copy(0),copy(1),copy(2)在同时进行，等待copy(0)完成后，执行add(0)
#     迭代1: copy(3) → wait copy(1) → add(1)  # copy(1),copy(2),copy(3)在同时进行，等待copy(1)完成后，执行add(1)
#     迭代2: copy(4) → wait copy(2) → add(2)  # copy(2),copy(3),copy(4)在同时进行，等待copy(2)完成后，执行add(2)
#     ...
#     迭代6: copy(9) → wait copy(7) → add(7)  # copy(7),copy(8),copy(9)在同时进行，等待copy(7)完成后，执行add(7)
#   【Cooldown】2次迭代（3-1=2）:
#     迭代0: wait copy(8) → add(8)  # copy(8),copy(9)在同时进行，等待copy(8)完成后，执行add(8)
#     迭代1: wait copy(9) → add(9)  # 仅剩copy(9)在进行，其完成后，执行add(9)
```

```
【为什么Warmup/Cooldown阶段是 num_buffers-1？】
提前"发射"num_buffers - 1个copy, 这样当 add(i) 完成时，copy(i+1) 刚好也完成，无缝衔接！

示例1（num_buffers=2，共10个数据块）：
  【Warmup】发出 1 个copy:
    copy(0)
  【Steady】8次迭代（10-2=8）:
    迭代0: copy(1) → wait copy(0) → add(0)  # copy(0)和copy(1)在同时进行，等待copy(0)完成后，执行add(0)
    迭代1: copy(2) → wait copy(1) → add(1)  # copy(1)和copy(2)在同时进行，等待copy(1)完成后，执行add(1)执行
    ...
    迭代7: copy(9) → wait copy(8) → add(8)  # copy(8)和copy(9)在同时进行，等待copy(8)完成后，执行add(8)执行
  【Cooldown】1次迭代（2-1=1）:
    wait copy(9) → add(9)  # 等待最后的copy(9)完成后，执行add(9)

示例2（num_buffers=3，共10个数据块）：
  【Warmup】发出 2 个copy:
    copy(0)
    copy(1)
  【Steady】7次迭代（10-3=7）:
    迭代0: copy(2) → wait copy(0) → add(0)  # copy(0),copy(1),copy(2)在同时进行，等待copy(0)完成后，执行add(0)
    迭代1: copy(3) → wait copy(1) → add(1)  # copy(1),copy(2),copy(3)在同时进行，等待copy(1)完成后，执行add(1)
    迭代2: copy(4) → wait copy(2) → add(2)  # copy(2),copy(3),copy(4)在同时进行，等待copy(2)完成后，执行add(2)
    ...
    迭代6: copy(9) → wait copy(7) → add(7)  # copy(7),copy(8),copy(9)在同时进行，等待copy(7)完成后，执行add(7)
  【Cooldown】2次迭代（3-1=2）:
    迭代0: wait copy(8) → add(8)  # copy(8),copy(9)在同时进行，等待copy(8)完成后，执行add(8)
    迭代1: wait copy(9) → add(9)  # 仅剩copy(9)在进行，其完成后，执行add(9)

规律：
- Warmup阶段: 发出 num_buffers-1 个copy，让流水线填满
- Steady阶段: 1个add + (num_buffers-1)个copy 同时进行
- Cooldown阶段: 2次迭代（num_buffers-1），处理剩余的数据块
- 总迭代数 = (num_buffers-1) + 主循环 + (num_buffers-1) = 数据块总数
```

代码实现

```
@gluon.jit
def issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    '''
    提交异步copy操作, 并返回下一个要copy的块的索引.
    '''
    yoffs = copy_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    cp.async_copy_global_to_shared(a_smem.index(copy_idx % num_buffers),  #
                                   a_ptrs + ystride_a * yoffs[None, :], mask)
    cp.async_copy_global_to_shared(b_smem.index(copy_idx % num_buffers),  #
                                   b_ptrs + ystride_b * yoffs[None, :], mask)
    cp.commit_group()
    return copy_idx + 1


@gluon.jit
def perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr, layout: gl.constexpr):
    '''
    处理当前数据块: 从共享内存读取、执行add、写回全局内存.
    返回下一个要处理的块的索引.
    '''
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    c_val = a_val + b_val
    yoffs = read_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)
    return read_idx + 1


@gluon.jit
def elementwise_add_pipelined_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
        smem_layout: gl.constexpr, num_buffers: gl.constexpr,  #
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    y_idx = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    xmask = (xoffs < xnumel)[:, None]

    # 新增: 共享内存大小增加一个num_buffers维度，用于多缓冲。
    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    copy_idx = 0
    read_idx = 0

    # 阶段1：Warmup
    # 目的：在主循环开始之前，先发出 num_buffers-1 个异步copy操作来"预热"流水线
    for _ in gl.static_range(num_buffers - 1):
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                               YBLOCK, num_buffers)

    # 阶段2：Steady State 
    # 目的：实现copy和add的重叠执行，最大化硬件利用率
    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        # 发起下一个数据块的异步copy（预取）
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                               YBLOCK, num_buffers)

        # 等待最老的copy完成, 第一次循环的话就是等待copy(0)完成
        # wait_group(N) 表示"等待, 直到未完成的组数 <= N", 例如：
        #   - num_buffers=3: wait_group(2) 等待, 直到只剩2个未完成.
        #   - 在第一次循环时此处有3次copy在进行中, 只剩2个未完成, 即最早的copy(0)完成了.
        cp.wait_group(num_buffers - 1)
        # 处理就绪的数据块：从共享内存读取、执行add、写回全局内存
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK, num_buffers,
                               layout)

    # 阶段3：Cooldown
    # 目的：处理剩余的数据块，等待所有正在进行的copy完成
    for i in gl.static_range(num_buffers - 1):
        cp.wait_group(num_buffers - 2 - i)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK, num_buffers,
                               layout)


def elementwise_add_pipelined(A, B, C, XBLOCK=32, YBLOCK=64, num_buffers=2):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    return elementwise_add_pipelined_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, smem_layout, num_buffers)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 120)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 64)])
@pytest.mark.parametrize("num_buffers", [1, 2, 3])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_elementwise_add_pipelined(xnumel, ynumel, XBLOCK, YBLOCK, num_buffers):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add_pipelined(a, b, c, XBLOCK, YBLOCK, num_buffers)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)
```

### 流水线kernel性能分析

下面对我们实现的pipeline kernel进行性能测试, 分别测试double buffer和triple buffer的情况.

```
if __name__ == "__main__":
    ms = triton.testing.do_bench(lambda: elementwise_add_pipelined(A, B, C, num_buffers=2))
    print(f"elementwise_add_pipelined (double buffer): {get_throughput(ms, C):.2f} TB/s")
    ms = triton.testing.do_bench(lambda: elementwise_add_pipelined(A, B, C, num_buffers=3))
    print(f"elementwise_add_pipelined (triple buffer): {get_throughput(ms, C):.2f} TB/s")
```


性能测试结果:
```
elementwise_add_pipelined (double buffer): 4.20 TB/s
elementwise_add_pipelined (triple buffer): 4.20 TB/s
```

可以看到, 使用流水线化优化确实产生了一定程度的加速。
但增加流水线深度并不会产生更高的性能, 表明该kernel是memory-bound的。

【性能瓶颈分析】
阻碍性能进一步提升的原因之一是寄存器压力（Register Pressure）。

对于每个线程处理的每个元素，需要占用以下寄存器：

**数据存储：** a_val / b_val / c_val 各 32 bits（1 个 float32 寄存器）

**地址计算**（GPU 上的地址是 64 位的，读写每个元素时都需要知道其在全局内存中的地址）：a_ptrs / b_ptrs / c_ptrs 各 64 bits（2 个 32 位寄存器）

**掩码（Mask）：** mask 用于边界检查，每个元素需要 1 bit，但通常占用完整寄存器

基于layout和分块大小, 分析其寄存器压力:

Layout定义：BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
- Layout块大小 = [1×1×1, 1×32×4] = [1, 128]，即总线程数 = 128线程
- size_per_thread = [1, 1]：在layout块内每个线程拥有1×1个元素的寄存器

而分块大小：[XBLOCK, YBLOCK] = [32, 64]
- 总元素：32 × 64 = 2048个元素
- 相对于layout块[1, 128]：需要在第一维平铺2048/128=16次

因此每个线程实际处理1*16个元素。

每个线程需要的寄存器（简化估算）：
  * 数据: 16个元素 × 3 (a,b,c) = 48个寄存器
  * 地址: 16个元素 × 6 (a,b,c各2个) = 96个寄存器(因为这16个元素不连续，需要存储16个地址)
  * 掩码: 16个元素 ≈ 16个寄存器
  * 其他（临时变量、循环索引等）: ~20个寄存器
  * 总计: ~180个寄存器

GPU限制：每个线程最多256个寄存器
- 当前使用 ~180/256 = 70% 的寄存器
- 如果增大块大小（如64×128），每个线程处理更多元素，寄存器会溢出
- 这就是为什么使用较小的 [32, 64] 块大小

寄存器溢出的后果：
- 寄存器不够时，数据会"溢出"到本地内存（Local Memory）
- 本地内存实际上在全局内存中，访问延迟很高
- 性能会急剧下降

在下一个教程中, 我们将转换tensor描述符和 TMA, 看看它们如何帮助降低寄存器压力, 但代价是寻址灵活性。

### 总结

\- 异步指令允许将访存与计算重叠。

\- 可以通过使用异步全局内存读取来实现异步copy。使用cp.commit_group()一次性提交多个异步操作, cp.wait_group()等待直到所有异步操作完成。

\- 流水线是一种循环优化技术, 用于重叠异步操作。