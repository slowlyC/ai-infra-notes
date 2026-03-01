## Gluon-01-Overview

### 前言

本系列是 Gluon 的学习笔记，基于 [Gluon 官方教程](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon) 整理，覆盖了从基础概念到高级优化的完整内容，在原文基础上做了结构重组，并补充了 CUDA 编程模型和 GPU 硬件架构层面的说明。

> 知乎专栏：[DSL教程](https://www.zhihu.com/column/c_1990516179064858333)
> 完整代码：[GitHub](https://github.com/slowlyC/ai-infra-notes/tree/main/tutorials/gluon)

- **Gluon-01-Overview**（本文）
- [Gluon-02-Layout_Introduction](https://zhuanlan.zhihu.com/p/1990509278801457706)
- [Gluon-03-Async_Copy](https://zhuanlan.zhihu.com/p/1990517098083029502)
- [Gluon-04-TMA](https://zhuanlan.zhihu.com/p/1990517971483906093)
- [Gluon-05-wgmma](https://zhuanlan.zhihu.com/p/1990835358502523949)
- [Gluon-06-tcgen05](https://zhuanlan.zhihu.com/p/1990835546797405057)
- [Gluon-07-Persistent_Kernel](https://zhuanlan.zhihu.com/p/2003592603732563562)
- [Gluon-08-Warp_Specialization](https://zhuanlan.zhihu.com/p/2003602919912649692)
- [Gluon-09-TMA_Gather_Scatter](https://zhuanlan.zhihu.com/p/2003599190585017693)
- [Gluon-10-tcgen05_copy](https://zhuanlan.zhihu.com/p/2003603029975381459)
- [Gluon-11-tcgen05-mma-scaled](https://zhuanlan.zhihu.com/p/2003603119259550000)

### Gluon 概述

Gluon 是一种 GPU 编程语言, 它与 Triton 共享相同的底层MLIR编译器栈。不同的是， Gluon 是一种更底层的语言, 在编写 kernel 时给予开发者更多的控制权, 同时也需要承担更多责任。

本系列教程将从基础概念开始, 逐步深入到Gluon的高级优化技术和现代 GPU 硬件特性, 最终实现一个高性能的 GEMM kernel。本教程假设你已具备 Triton 的基础知识（可参考https://triton-lang.org/main/getting-started/tutorials/index.html）。

从宏观角度看, Gluon 和 Triton 有很多相似之处。一是两者都是 Python DSL, 共享相同的前端和 JIT 接口，二是两者都属于基于分块(tile)的 SPMD 编程模型, 其中块代表分布在多个"程序"上的 N 维数组。

然而在底层细节上，两者却相差较大。Triton 为了方便开发，将许多 kernel 实现和 GPU 硬件的底层细节进行了抽象，分块布局(tile layouts)、内存分配、数据移动和异步操作等都交给了编译器自动管理。虽然 Triton 编译器能为大多数 kernel 生成高效的代码, 但对这些底层细节进行手动调优的仍可能取得更好的性能。当遇到这种情况时, 由于所有细节都被隐藏了, 用户很难进一步优化性能。

Gluon 则将这些细节暴露给用户，这意味着编写 Gluon kernel 需要对 GPU 硬件和编程有更深入的理解, 但也正因如此, 你可以通过精细控制这些底层细节来编写出性能更优的 kernel。

### Gluon kernel示例

下面我们来定义一个 Gluon kernel 并编写其启动器(launcher)，来了解gluon的基本流程。

使用 `@gluon.jit` 装饰器来声明 Gluon kernel, 其调用方式与 Triton kernel 完全相同, 直接从 Python 中调用即可。

```python
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# 定义一个复制标量的简单kernel，将标量从in_ptr复制到标量out_ptr。
@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)

# 在python host侧即可调用我们编写的 Gluon kernel 的代码。
# 当 PyTorch tensor 传递给 Gluon kernel 时, 会自动转换为全局内存指针, 这点与 Triton 一致。
# 网格(grid)的指定方式也完全相同。
def copy_scalar(input, output):
    # 启动单个程序。
    grid = (1, )
    copy_scalar_kernel[grid](input, output, num_warps=1)

# 现在来测试一下 kernel。运行 `pytest 01-intro.py` 即可执行测试。
def test_copy_scalar():
    input = torch.tensor([42.0], device="cuda")
    output = torch.empty_like(input)
    copy_scalar(input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
```

### 带超参数的kernel

接下来编写一个带超参数的 kernel, 超参数以 constexpr 参数的形式传递, 方式与 Triton 相同。

```python
# 一个 memcpy kernel, 将输入的一维tensor根据超参数XBLOCK切分为多个块, 并存储到out_ptr中, 每个程序处理一个块。
@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    # 每个程序处理地址范围 [start, end), 确保不超出 [0, xnumel) 的边界。
    pid = gl.program_id(0)
    start = pid * XBLOCK
    end = min(start + XBLOCK, xnumel)
    # 每个循环处理1个元素，这是因为处理多个元素时需要选择合适的layout。
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)


def memcpy(input, output, XBLOCK):
    # 计算元素总数。例如: input.shape 为 torch.Size([40]) 时, xnumel 为 40
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    memcpy_kernel[grid](input, output, xnumel, XBLOCK, num_warps=1)


@pytest.mark.parametrize("XBLOCK", [64])
@pytest.mark.parametrize("xnumel", [40, 500])
def test_memcpy(XBLOCK, xnumel):
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
```

### 超参数自动调优

Gluon 的超参数同样支持自动调优, 用法与 Triton 一致。下面以自动调优 XBLOCK 为例。

```python
@triton.autotune(
    configs=[triton.Config({"XBLOCK": 2**i}, num_warps=1) for i in range(8, 14)],
    key=["xnumel"],
)
@gluon.jit
def memcpy_kernel_autotune(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK)


def memcpy_autotune(input, output):
    xnumel = input.numel()

    def grid(META):
        return (triton.cdiv(xnumel, META["XBLOCK"]), )

    memcpy_kernel_autotune[grid](input, output, xnumel)


if __name__ == "__main__":
    torch.manual_seed(0)
    xnumel = 2 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    fn = lambda: memcpy_autotune(input, output)
    ms = triton.testing.do_bench(fn)
    gbytes = 2 * xnumel * input.element_size() >> 30
    print("Benchmarking memcpy")
    print("===================")
    print(f"Time:        {ms:.2f} ms")
    print(f"Throughput: {gbytes / (ms * 1e-3):.2f} GB/s")
```

运行 `TRITON_PRINT_AUTOTUNING=1 python 01-intro.py` 可以查看自动调优选择的 XBLOCK 值。

在 GB200 上测试, 最佳 XBLOCK 为 2048, 复制 8 GB 数据的速度约为 666 GB/s, 远低于该 GPU 8 TB/s 的峰值带宽。

```  
Time:        24.00 ms
Throughput: 666.24 GB/s
```



### 总结

因为提高性能是使用 Gluon 编写 kernel 的主要目标, 所以让我们花点时间分析一下。

首先, 我们没有充分利用 GPU 的并行性。每个 Gluon "程序"对应 GPU 上的一个线程块(CTA), 虽然 GPU 可以同时执行多个 CTA, 但在我们的 kernel 中, 每个 CTA 一次只复制 1 个元素。

要想一次复制多个元素, 需要加载和存储分块(tiles), 这就涉及到选择合适的布局(layout), 因为不同的 layout 对性能影响很大。

在下一个教程中, 我们将介绍 Gluon 中 layout 的基础知识, 以及它们如何影响性能。

本教程的要点: 

- 编写 Gluon kernel 的整体流程与 Triton kernel 相同。
- Gluon 采用基于分块的 SPMD 编程模型, 熟悉 Triton 的开发者会觉得很亲切。
- Gluon 主要改变的是device侧的编写方式, host侧代码的差异仅在于 Gluon kernel 可能需要更多的超参数。

### 参考
https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon