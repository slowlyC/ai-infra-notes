"""
Gluon 简介
==========

Gluon 是一种 GPU 编程语言, 它与 Triton 共享相同的编译器栈。
但 Gluon 是一种更底层的语言, 在编写 kernel 时给予开发者更多的控制权, 同时也需要承担更多责任。

从宏观角度看, Gluon 和 Triton 有很多相似之处。两者都是基于 tile 的 SPMD 编程模型, tile 表示分布在多个"程序"上的 N 维数组。
两者都是 Python DSL, 共享相同的前端和 JIT 基础设施。

不过, Triton 将许多 kernel 实现和 GPU 硬件的底层细节进行了抽象。它把分块布局(tile layouts)、内存分配、数据移动和异步操作等
都交给编译器自动管理。这些底层细节的处理直接影响 kernel 的性能。虽然 Triton 编译器能为大多数 kernel 生成高效的代码, 
但手动调优的底层代码仍可能取得更好的性能。当遇到这种情况时, 由于所有细节都被隐藏了, 用户很难进一步优化性能。

Gluon 则将这些细节暴露给用户。这意味着编写 Gluon kernel 需要对 GPU 硬件和编程有更深入的理解。
但也正因如此, 你可以通过精细控制这些底层细节来编写出性能更优的 kernel。

本教程将带你学习 Gluon 中的 GPU kernel 开发, 从基础概念开始, 逐步深入到高级优化技术和现代 GPU 硬件特性。
"""

# %%
# 下面我们来定义一个 Gluon kernel 并编写其启动器(launcher), 来了解gluon的基本流程。
# 使用 `@gluon.jit` 装饰器来声明 Gluon kernel, 其调用方式与 Triton kernel 完全相同, 直接从 Python 中调用即可。

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# 定义一个复制标量的简单 kernel, 将标量从 in_ptr 复制到标量 out_ptr。
@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)

# 在 python host 侧即可调用我们编写的 Gluon kernel 的代码。
# 当 PyTorch tensor 传递给 Gluon kernel 时, 会自动转换为全局内存指针, 这点与 Triton 一致。
def copy_scalar(input, output):
    # 网格(grid)的指定方式也完全相同, 这里启动单个程序。
    grid = (1, )
    copy_scalar_kernel[grid](input, output, num_warps=1)

# 现在来测试一下 kernel。运行 `pytest 01-intro.py` 即可执行测试。
def test_copy_scalar():
    input = torch.tensor([42.0], device="cuda")
    output = torch.empty_like(input)
    copy_scalar(input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# 接下来编写一个带超参数的 kernel, 超参数以 constexpr 参数的形式传递, 方式与 Triton 相同。

# 一个 memcpy kernel, 将输入的一维tensor根据超参数XBLOCK切分为多个块, 并存储到out_ptr中, 每个程序处理一个块。
@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    # 每个程序处理地址范围 [start, end), 确保不超出 [0, xnumel) 的边界。
    pid = gl.program_id(0)
    start = pid * XBLOCK
    end = min(start + XBLOCK, xnumel)
    # 每个循环处理1个元素, 这是因为处理多个元素时需要选择合适的layout,  后面的教程会逐步介绍。
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


# %%
# Gluon 的超参数同样支持自动调优, 用法与 Triton 一致。下面以自动调优 XBLOCK 为例。

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

# 运行 `TRITON_PRINT_AUTOTUNING=1 python 01-intro.py` 可以查看自动调优选择的 XBLOCK 值。
# 在 GB200 上测试, 最佳 XBLOCK 为 2048, 复制 8 GB 数据的速度约为 666 GB/s, 
# 远低于该 GPU 8 TB/s 的峰值带宽。
#   
# ```
# Time:        24.00 ms
# Throughput: 666.24 GB/s
# ```


# %%
# 因为提高性能是使用 Gluon 编写 kernel 的主要目标, 所以让我们花点时间分析一下。
# 首先, 我们没有充分利用 GPU 的并行性。每个 Gluon "程序"对应 GPU 上的一个线程块(CTA), 虽然 GPU 可以同时执行多个 CTA, 
# 但在我们的 kernel 中, 每个 CTA 一次循环只复制 1 个元素。
#
# 要想一次复制多个元素, 需要加载和存储分块(tiles), 这就涉及到选择合适的布局(layout), 因为不同的 layout 对性能影响很大。
#
# 在下一个教程中, 我们将介绍 Gluon 中 layout 的基础知识, 以及它们如何影响性能。

# 本教程的要点: 

# - 编写 Gluon kernel 的整体流程与 Triton kernel 相同。
# - Gluon 采用基于分块的 SPMD 编程模型, 熟悉 Triton 的开发者会觉得很亲切。
# - Gluon 主要改变的是device侧的编写方式, host侧代码的差异仅在于 Gluon kernel 可能需要更多的超参数。
