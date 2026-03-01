"""
Tensor Layout(张量布局)
========================

在 Gluon 中, 每个 tensor 都需要指定 layout。Layout 描述了 tensor 的元素如何分布到线程块中的各个线程上。
这种分布遵循 GPU 的层级结构: 线程块 → warp(线程束)→ lane(通道/线程)→ 寄存器。

Tensor 的元素在线程间均匀分布, 即每个线程处理相同数量的元素。
由于 Triton 要求所有分块维度都是 2 的幂, 因此每个线程处理的元素数量也是 2 的幂。

Layout 本质上定义了一个映射关系: 给定(寄存器编号、lane 编号、warp 编号), 可以确定对应的 tensor 元素。
`BlockedLayout` 是 Gluon 中最常用的 layout 类型, 它将元素组织成与 tensor 维度相同的"layout 块"。

来看一个具体例子: 

```python
gl.BlockedLayout(
    size_per_thread=[2, 4],
    threads_per_warp=[16, 2],
    warps_per_cta=[2, 2],
    order=[1, 0],
)
```

在这个layout块内, layout描述了逻辑上tensor元素与寄存器、线程和线程束分块的层级结构。

这里 `size_per_thread=[2, 4]` 表示每个线程持有一个 2x4 的连续子块, 存储在该线程的寄存器中。
NVIDIA GPU 的物理寄存器是32位的, 一个寄存器能存储多少元素取决于数据类型。
对于fp32数据类型, 一个寄存器能存储1个元素, 而对于fp16数据类型, 一个寄存器能存储2个元素。

我们将 `size_per_thread`、`threads_per_warp` 和 `warps_per_cta` 逐元素相乘, 
得到layout块形状为 [64, 16], 这是一个 CTA 所处理的元素数量。

`order` 指定了tensor维度的分块顺序, 其含义与 triton 的 make_block_ptr 相同。
`order=[1, 0]` 表示先沿第 1 维(列)分块, 再沿第 0 维(行)分块, 即行优先顺序。

假设数据类型为fp32, 对于线程 T, 其持有的元素分布如下(T:n 表示线程T的第n个寄存器): 

```
[[T:0, T:1, T:2, T:3],
 [T:4, T:5, T:6, T:7]]
```

注意寄存器编号沿内层维度(这里是列)递增。

如果 `order` 是 `[0, 1]`(列优先), 则分布变为: 

```
[[T:0, T:2, T:4, T:6],
 [T:1, T:3, T:5, T:7]]
```

接下来看 `threads_per_warp=[16, 2]`, 它描述了一个 warp 内的 32 个线程如何排列。
对于 `order=[1, 0]`, 线程在 warp 内的排列是: 

```
[[ T0,  T1],
 [ T2,  T3],
 ...
 [T28, T29],
 [T30, T31]]
```

把每个线程替换成它持有的 2x4 子块, 就得到整个 warp 持有的 tensor 区域: 

```
[[ T0:0,  T0:1,  T0:2,  T0:3,  T1:0,  T1:1,  T1:2,  T1:3],
 [ T0:4,  T0:5,  T0:6,  T0:7,  T1:4,  T1:5,  T1:6,  T1:7],
 [ T2:0,  T2:1,  T2:2,  T2:3,  T3:0,  T3:1,  T3:2,  T3:3],
 [ T2:4,  T2:5,  T2:6,  T2:7,  T3:4,  T3:5,  T3:6,  T3:7],
 ...
 [T28:0, T28:1, T28:2, T28:3, T29:0, T29:1, T29:2, T29:3],
 [T28:4, T28:5, T28:6, T28:7, T29:4, T29:5, T29:6, T29:7],
 [T30:0, T30:1, T30:2, T30:3, T31:0, T31:1, T31:2, T31:3],
 [T30:4, T30:5, T30:6, T30:7, T31:4, T31:5, T31:6, T31:7]]
```

对 `warps_per_cta=[2, 2]` 重复上述过程, 就能得到整个 layout 块内所有线程的完整映射。

如果tensor的大小与layout块相同, 则元素根据块layout分布。
如果tensor形状不同, 我们需要对layout块进行平铺(tile)或广播(broadcast)。

`平铺(tile)`是指tensor块大于layout块时, 将layout块复制并重复, 以适应tensor的形状, 示例:
考虑一个 `128x64xf32` tensor, 将 tensor 形状除以layout块形状[64, 16], 我们得到layout块的数量为 `[2, 4]`。
根据 `order=[1, 0]` 对layout块进行平铺, 以此来向每个线程添加更多寄存器: 

```
[[B0, B1, B2, B3],
 [B4, B5, B6, B7]]
```

理解寄存器用量对于控制 kernel 的寄存器压力很关键, 计算下`128x64xf32` tensor的寄存器用量:
由于 `size_per_thread=[2, 4]`, 每个线程在一个layout块内持有8个寄存器, 
而覆盖整个tensor需要8个layout块, 因此每个线程共持有8x8=64个寄存器。

`广播(broadcast)`是指tensor块小于layout块时, 将tensor的元素复制并重复, 以适应layout块的形状, 示例:
考虑一个较小的tensor, 比如 `32x8xf32`, layout块形状为`[64, 16]`, 大于tensor的形状。
做法是让tensor沿每个维度广播以适应layout块, 从线程束开始, 然后是线程, 最后是寄存器。

将layout块形状除以tensor形状, 我们得到 `[2, 2]` (即 64÷32=2, 16÷8=2)。
这与 `warps_per_cta=[2, 2]` 完全对应, 也就是说每个线程束正好对应一个完整的tensor。
从tensor的角度来看, 这看起来像: 

```
[[  T0:0| T32:0| T64:0| T96:0, ...,   T1:3| T33:3| T65:3| T97:3],
 [  T0:4| T32:4| T64:4| T96:4, ...,   T1:7| T33:7| T65:7| T97:7],
 ...
 [ T30:0| T62:0| T94:0|T126:0, ...,  T31:3| T63:3| T95:3|T127:3]
 [ T30:4| T62:4| T94:4|T126:4, ...,  T31:7| T63:7| T95:7|T127:7]]
```
位置 [0,0]: 被 T0:0, T32:0, T64:0, T96:0 这4个线程共同持有
位置 [0,1]: 被 T1:3, T33:3, T65:3, T97:3 这4个线程共同持有

即使tensor只有 `32 * 8 = 256` 个元素, 在每个程序中也会用到 `64 * 16 = 1024` 个物理寄存器。


上述介绍的 BlockedLayout 是最常用的 layout, 主要用于: 
- 表示全局内存访问时的合并 layout
- 表示 NVIDIA Blackwell GPU 上 tensor memory 的某些寄存器 layout

Gluon 还提供了多种 layout 类型。有些是特定操作所需的专用 layout, 如 Tensor Core 的 MMA 指令;
有些用于表示 `expand_dims`、`broadcast`、`reshape` 等操作的结果。详见 TritonGPUAttrDefs.td。
"""

# %%
# 现在我们对 BlockedLayout 有了基本了解, 接下来扩展上一教程的 `memcpy` 示例, 看看 layout 如何影响 kernel 性能。
# 我们将让每个程序加载和存储整个数据块, 而不是单个标量。

import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# 辅助函数: 用于选择性运行脚本的某些部分。
# 直接运行 `python 02-layouts.py` 会执行所有内容, 
# 也可以用 `python 02-layouts.py R_vs_throughput,LDG_STG_instructions` 只运行指定部分。
def _enabled(label):
    from sys import argv
    return len(argv) == 1 or label in argv[1].split(",")


# 每个程序负责复制一个数据块, 使用 layout 将工作分配到所有线程, 
# 把 layout 也作为参数传入 kernel, 方便测试不同 layout 的效果。
@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # Gluon 与 Triton 的主要区别: 需要显式指定 tensor 的 layout。
    # 由于 layout 会通过类型推断自动传播, 只需为起始的索引 tensor 指定 layout 即可。
    indices = gl.arange(0, XBLOCK, layout=layout)

    offsets = start + indices
    mask = offsets < xnumel

    value = gl.load(in_ptr + offsets, mask=mask)
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)


def memcpy_1d_impl(input, output, XBLOCK, layout, num_warps):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    compiled_kernel = memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, num_warps=num_warps)
    return compiled_kernel


@pytest.mark.parametrize("XBLOCK", [128, 256])
@pytest.mark.parametrize("xnumel", [200, 1000])
@pytest.mark.parametrize("num_warps", [4])
def test_memcpy_1d(XBLOCK, xnumel, num_warps):
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    layout = gl.BlockedLayout([1], [32], [num_warps], [0])
    memcpy_1d_impl(input, output, XBLOCK, layout, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# 下面我们用不同的 layout 对 kernel 进行性能测试。
# 假设XBLOCK=2048, 这是上一教程中得到的最佳值, 来测试不同 R 值下的性能。
# 对于 1D tensor, BlockedLayout 的选择空间较小, 假设 num_warps=4, 有效的 layout 形式为: 
#
# ```python
# gl.BlockedLayout(
#     size_per_thread=[R],
#     threads_per_warp=[32],
#     warps_per_cta=[4],
#     order=[0],
# )
# ```
#
# 其中 R 必须是 2 的幂。


def get_throughput(input, ms):
    tbytes = (2 * input.numel() * input.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


def bench_memcpy_impl(input, output, impl):
    compiled_kernel = impl(input, output)
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn)
    return compiled_kernel, get_throughput(input, ms)


def bench_memcpy(impl):
    torch.manual_seed(0)
    xnumel = 2 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    return bench_memcpy_impl(input, output, impl)


if __name__ == "__main__" and _enabled("R_vs_throughput"):
    print("R vs. Throughput")
    print("================")
    XBLOCK = 2048
    num_warps = 4
    kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps)
    compiled_kernels = []
    # 当 XBLOCK=2048 时, 由于layout 块大小不能超过 XBLOCK, 所以 Rx32x4 ≤ 2048, 即限制R ≤ 16。
    for i in range(0, 5):
        R = 2**i
        layout = gl.BlockedLayout([R], [32], [num_warps], [0])
        impl = partial(kernel, layout=layout)
        compiled_kernel, throughput = bench_memcpy(impl)
        compiled_kernels.append((R, compiled_kernel))
        print(f"R={R:<3} {throughput:.3f} TB/s")
    print()

# 在 GB200 上的测试结果: 
# ```
# R=1   6.574 TB/s
# R=2   6.476 TB/s
# R=4   6.474 TB/s
# R=8   6.502 TB/s
# R=16  6.214 TB/s
# ```

# %%
# 可以看到 layout 确实影响性能。下面通过查看 SASS 汇编来分析原因。

if __name__ == "__main__" and _enabled("LDG_STG_instructions"):
    print("LDG/STG instructions")
    print("====================")
    for R, compiled_kernel in compiled_kernels:
        print(f"\nR={R}")
        print("==========")
        sass = compiled_kernel.asm["sass"]
        for line in sass.split("\n"):
            if "LDG.E" in line or "STG.E" in line:
                print(line)
    print()

# LDG/STG instructions
# ====================

# R=1
# ==========
# 16:1:2:-:1      @!P0 LDG.E R0, desc[UR4][R8.64];
# --:-:3:-:1      @!P0 LDG.E R15, desc[UR4][R4.64];
# --:-:4:-:1      @!P0 LDG.E R17, desc[UR4][R4.64+0x200];
# --:-:2:-:1      @!P0 LDG.E R19, desc[UR4][R4.64+0x400];
# --:-:2:-:1      @!P0 LDG.E R21, desc[UR4][R4.64+0x600];
# --:-:2:-:1      @!P0 LDG.E R23, desc[UR4][R4.64+0x800];
# --:-:2:-:1      @!P0 LDG.E R25, desc[UR4][R4.64+0xa00];
# --:-:2:-:1      @!P0 LDG.E R27, desc[UR4][R4.64+0xc00];
# --:-:2:-:1      @!P1 LDG.E R33, desc[UR4][R4.64+0x1400];
# --:-:2:-:4      @!P2 LDG.E R35, desc[UR4][R4.64+0x1600];
# --:-:2:-:1      @!P0 LDG.E R29, desc[UR4][R4.64+0x1000];
# --:-:2:-:4      @!P3 LDG.E R37, desc[UR4][R4.64+0x1800];
# --:-:2:-:4      @!P4 LDG.E R10, desc[UR4][R4.64+0x1a00];
# --:-:2:-:4      @!P5 LDG.E R12, desc[UR4][R4.64+0x1c00];
# --:0:2:-:1      @!P0 LDG.E R31, desc[UR4][R4.64+0x1200];
# --:0:5:-:3      @!P6 LDG.E R32, desc[UR4][R8.64];
# 08:0:-:-:1      @!P0 STG.E desc[UR4][R6.64], R15;
# 16:0:-:-:1      @!P0 STG.E desc[UR4][R6.64+0x200], R17;
# 04:0:-:-:1      @!P0 STG.E desc[UR4][R6.64+0x400], R19;
# --:0:-:-:1      @!P0 STG.E desc[UR4][R6.64+0x600], R21;
# --:0:-:-:1      @!P0 STG.E desc[UR4][R6.64+0x800], R23;
# --:0:-:-:1      @!P0 STG.E desc[UR4][R6.64+0xa00], R25;
# --:0:-:-:1      @!P0 STG.E desc[UR4][R6.64+0xc00], R27;
# --:0:-:-:1      @!P0 STG.E desc[UR4][R4.64], R0;
# --:0:-:-:1      @!P0 STG.E desc[UR4][R6.64+0x1000], R29;
# --:0:-:-:1      @!P1 STG.E desc[UR4][R6.64+0x1400], R33;
# --:0:-:-:4      @!P2 STG.E desc[UR4][R6.64+0x1600], R35;
# --:0:-:-:4      @!P3 STG.E desc[UR4][R6.64+0x1800], R37;
# --:0:-:-:4      @!P0 STG.E desc[UR4][R6.64+0x1200], R31;
# --:0:-:-:4      @!P4 STG.E desc[UR4][R6.64+0x1a00], R10;
# --:0:-:-:1      @!P5 STG.E desc[UR4][R6.64+0x1c00], R12;
# 32:-:-:-:1      STG.E desc[UR4][R2.64], R32;

# R=2
# ==========
# 01:-:2:-:1      @!P0 LDG.E.64 R16, desc[UR4][R22.64];
# --:-:3:-:1      @!P3 LDG.E.64 R6, desc[UR4][R22.64+0xc00];
# --:-:4:-:5      @!P4 LDG.E.64 R8, desc[UR4][R22.64+0x1000];
# --:-:4:-:4      @!P1 LDG.E.64 R2, desc[UR4][R22.64+0x400];
# --:-:4:-:4      @!P2 LDG.E.64 R4, desc[UR4][R22.64+0x800];
# --:-:4:-:4      @!P5 LDG.E.64 R10, desc[UR4][R22.64+0x1400];
# --:-:4:-:1      @!P6 LDG.E.64 R12, desc[UR4][R22.64+0x1800];
# --:0:5:-:4      @!P0 LDG.E.64 R14, desc[UR4][R20.64];
# 04:0:-:-:1      @!P1 STG.E.64 desc[UR4][R24.64], R16;
# 08:0:-:-:4      @!P3 STG.E.64 desc[UR4][R24.64+0xc00], R6;
# 16:0:-:-:6      @!P4 STG.E.64 desc[UR4][R24.64+0x1000], R8;
# --:0:-:-:4      @!P1 STG.E.64 desc[UR4][R24.64+0x400], R2;
# --:0:-:-:4      @!P2 STG.E.64 desc[UR4][R24.64+0x800], R4;
# --:0:-:-:4      @!P5 STG.E.64 desc[UR4][R24.64+0x1400], R10;
# --:0:-:-:1      @!P6 STG.E.64 desc[UR4][R24.64+0x1800], R12;
# 32:-:-:-:1      STG.E.64 desc[UR4][R18.64], R14;

# R=4
# ==========
# 16:-:2:-:4      @!P0 LDG.E.128 R8, desc[UR4][R2.64];
# --:-:3:-:4      @!P2 LDG.E.128 R12, desc[UR4][R2.64+0x800];
# --:-:4:-:1      @!P3 LDG.E.128 R16, desc[UR4][R2.64+0x1000];
# --:0:5:-:4      @!P1 LDG.E.128 R4, desc[UR4][R2.64+0x1800];
# 04:0:-:-:4      @!P0 STG.E.128 desc[UR4][R20.64], R8;
# 08:0:-:-:4      @!P2 STG.E.128 desc[UR4][R20.64+0x800], R12;
# 16:0:-:-:1      @!P3 STG.E.128 desc[UR4][R20.64+0x1000], R16;
# 32:-:-:-:1      STG.E.128 desc[UR4][R20.64+0x1800], R4;

# R=8
# ==========
# 01:-:2:-:4      @!P0 LDG.E.128 R4, desc[UR4][R2.64];
# --:-:3:-:4      @!P0 LDG.E.128 R8, desc[UR4][R2.64+0x10];
# --:-:4:-:1      @!P1 LDG.E.128 R12, desc[UR4][R2.64+0x1000];
# --:0:5:-:4      @!P1 LDG.E.128 R16, desc[UR4][R2.64+0x1010];
# 04:0:-:-:4      @!P0 STG.E.128 desc[UR4][R20.64], R4;
# 08:0:-:-:4      @!P0 STG.E.128 desc[UR4][R20.64+0x10], R8;
# 16:0:-:-:1      @!P1 STG.E.128 desc[UR4][R20.64+0x1000], R12;
# 32:-:-:-:1      STG.E.128 desc[UR4][R20.64+0x1010], R16;

# R=16
# ==========
# 01:-:2:-:4      @!P0 LDG.E.128 R4, desc[UR4][R2.64];
# --:-:3:-:4      @!P0 LDG.E.128 R8, desc[UR4][R2.64+0x10];
# --:-:4:-:1      @!P0 LDG.E.128 R12, desc[UR4][R2.64+0x20];
# --:0:5:-:4      @!P0 LDG.E.128 R16, desc[UR4][R2.64+0x30];
# 04:0:-:-:4      @!P0 STG.E.128 desc[UR4][R20.64], R4;
# 08:0:-:-:4      @!P0 STG.E.128 desc[UR4][R20.64+0x10], R8;
# 16:0:-:-:1      @!P0 STG.E.128 desc[UR4][R20.64+0x20], R12;
# 32:-:-:-:1      STG.E.128 desc[UR4][R20.64+0x30], R16;

# 我们看到不同layout会影响读/写向量化和步幅: 

# | R  | width | vec_len | n_loads | stride |
# |----|-------|---------|---------|--------|
# | 1  | 32    | 32      | 1       | 0x00   |
# | 2  | 64    | 64      | 1       | 0x00   |
# | 4  | 128   | 128     | 1       | 0x00   |
# | 8  | 256   | 128     | 2       | 0x10   |
# | 16 | 512   | 128     | 4       | 0x10   |
#
# GPU 具有128字节的缓存行, 分为4个扇区(sectors), 每个sector=32字节, sector是访问全局内存的最小粒度。
# 因此 GPU 试图通过“访存合并”, 对同一扇区的连续访问进行合并, 来最小化扇区访问次数。
#
# 当 R=1 时, 每个线程读取 4 字节(float32), 对应一条 LDG.E 指令读取 32bit。
# 然而内存的最小访问粒度是32字节, 所以最小要读取32字节, 这大于一个线程需要的4字节.
# 而一个线程束(warp)中的 32 个线程访问的是连续的 32*4=128 字节区域，
# 因此硬件会将一个线程束(warp)中的 32 个线程的 LDG.E 访问合并成 128 字节进行访问, 这正好是一个缓存行的大小。
# 注意 PyTorch 分配的tensor是对齐到 256 字节的。
#
# 将 R 增加到 2 或 4 会加宽每个 `LDG.E` 指令, 但会降低kernel速度, 尽管 32B 扇区读取的数量保持不变。
# 性能下降可能有多种硬件原因, 但通过查看汇编指令左侧的调度注释, 我们可以看到其中具体原因, 比如: 
#
# ```
# 16:1:2:-:1	@!P0 LDG.E R0, desc[UR4][R8.64];
# --:-:3:-:1	@!P0 LDG.E R15, desc[UR4][R4.64];
# --:-:4:-:1	@!P0 LDG.E R17, desc[UR4][R4.64+0x200];
# ...
# 08:0:-:-:1	@!P0 STG.E desc[UR4][R6.64], R15;
# 16:0:-:-:1	@!P0 STG.E desc[UR4][R6.64+0x200], R17;
# 04:0:-:-:1	@!P0 STG.E desc[UR4][R6.64+0x400], R19;
# ```
#
# 这种 `16:1:2:-:1` 格式的含义是: 
# ```
# wait_mask : read_barrier : write_barrier : yield : stall
# ```
#
# 具体含义:
# - `wait_mask`: 等待掩码, 表示该指令需要等待哪些屏障被清除才能执行
# - `read_barrier`: 读屏障, 表示该指令读取数据后设置的屏障编号
# - `write_barrier`: 写屏障, 表示该指令写入数据后设置的屏障编号
# - `yield`: 是否让出执行权
# - `stall`: 停顿周期数
#
# 例如 `16:1:2:-:1` 表示:
# - `16`: 等待屏障 16 被清除
# - `1`: 设置读屏障 1
# - `2`: 设置写屏障 2
# - `-`: 不让出执行权
# - `1`: 停顿 1 个周期
#
# 性能影响分析:
# - `LDG.E` 指令设置 `write_barrier`(如 `:2:`), 因为它将数据写入寄存器
# - 后续的 `STG.E` 指令有 `wait_mask`(如 `08:`), 表示它需要等待屏障被清除才能执行
# - 当 R=1 时, 使用更小粒度的 `LDG.E` 指令, 每个指令完成得更快, 屏障清除得更早
# - 这样 `STG.E` 指令可以更早开始执行, 提高了指令级并行度(ILP), 从而提升性能
# - 当 R=2 或 R=4 时, `LDG.E` 指令更宽, 需要更长时间完成, 屏障清除更晚
# - 导致 `STG.E` 指令需要等待更长时间, 降低了指令级并行度, 从而影响性能
#
# 至于为什么 R=8 比 R=2/4 还快, 没有 profiler 很难确定具体原因。


# %%
# 下面我们测试不同 XBLOCK 和 R 值组合下的性能。

if __name__ == "__main__" and _enabled("XBLOCK_R_vs_throughput"):
    print("(XBLOCK, R) vs. Throughput")
    print("==========================")
    num_warps = 4

    print("XBLOCK   ", end=" ")
    for i in range(0, 5):
        print(f"R={2**i:<3}", end=" ")
    print()

    for j in range(10, 15):
        XBLOCK = 2**j
        print(f"{XBLOCK:<8}", end=" ")
        kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps)
        for i in range(0, 5):
            R = 2**i
            layout = gl.BlockedLayout([R], [32], [num_warps], [0])
            impl = partial(kernel, layout=layout)
            compiled_kernel, throughput = bench_memcpy(impl)
            print(f"{throughput:.3f}", end=" ")
        print()
    print()


# 可以看到, 在不同 XBLOCK 下, R=8 并不总是比 R=2/4 快: 
#
# ```
# XBLOCK    R=1   R=2   R=4   R=8   R=16
# 1024     6.566 6.548 6.542 6.550 5.226
# 2048     6.572 6.474 6.474 6.504 6.218
# 4096     6.554 6.492 6.454 6.396 6.182
# 8192     6.606 6.532 6.482 6.478 6.176
# 16384    6.522 6.556 6.486 6.510 6.146
# ```
#
# 从测试结果看, R=1 配合 XBLOCK=8192 可以获得最佳吞吐量。实际应用中可以通过自动调优来搜索最佳参数。


# %%
# 下面我们来写一个 2D memcpy 的Kernel。
# 为高维tensor选择正确的layout比1D tensor要困难得多, 因为高维tensor可能以非连续的方式被访问。

# 在 2D memcpy 中, 我们需要计算每个元素在内存中的位置。对于 2D tensor, 我们需要:
# 1. 计算行的偏移量offset_row(1D)
# 2. 计算列的偏移量offset_col(1D)  
# 3. 将行偏移量乘以行步幅, 列偏移量乘以列步幅
# 4. 将两者相加得到最终的内存地址

# 原始的2D tensor使用2D BlockedLayout, 但行和列的偏移量本身却是1D的, 因此对偏移量使用SliceLayout。
# 例如, 行编译量的layout可以是:
# ```python
# gl.SliceLayout(dim=1, parent=layout)  # 从 2D layout 中切掉第 1 维, 保留第 0 维
# ```

# SliceLayout 的概念很简单: 它从父layout中"切掉"某个维度, 得到一个低维的layout。
# 下面用一个具体例子来说明:

# 假设我们有一个2D BlockedLayout:
# ```python
# layout = gl.BlockedLayout(
#     size_per_thread=[2, 4],       # 每个线程处理 2行x4列 的元素
#     threads_per_warp=[16, 2],     # 每个warp有 16x2 = 32个线程
#     warps_per_cta=[2, 2],         # 每个CTA有 2x2 = 4个warp
#     order=[1, 0],                 # 行主序, 先分块列, 再分块行
# )
# ```

# 这个2D layout的元素分布如下(每个元素标记为"线程号:寄存器号"):
# ```
# [[ T0:0,  T0:1,  T0:2,  T0:3,|  T1:0,  T1:1,  T1:2,  T1:3],  # 第0行
#  [ T0:4,  T0:5,  T0:6,  T0:7,|  T1:4,  T1:5,  T1:6,  T1:7],  # 第1行
#  [ T2:0,  T2:1,  T2:2,  T2:3,|  T3:0,  T3:1,  T3:2,  T3:3],  # 第2行
#  [ T2:4,  T2:5,  T2:6,  T2:7,|  T3:4,  T3:5,  T3:6,  T3:7],  # 第3行
#  ...
#  [T28:0, T28:1, T28:2, T28:3,| T29:0, T29:1, T29:2, T29:3],
#  [T28:4, T28:5, T28:6, T28:7,| T29:4, T29:5, T29:6, T29:7],
#  [T30:0, T30:1, T30:2, T30:3,| T31:0, T31:1, T31:2, T31:3],
#  [T30:4, T30:5, T30:6, T30:7,| T31:4, T31:5, T31:6, T31:7]]  # 第31行
# ```

# 现在, 我们用 gl.SliceLayout(dim=1, parent=layout), 沿 dim=1 创建一个SliceLayout, 过程如下:

# 第一步: 将每一行的所有元素"展平"成一行(用 | 分隔):
# ```
# [ T0:0| T0:1| T0:2| T0:3| T1:0| T1:1| T1:2| T1:3,  # 第0行展平
#   T0:4| T0:5| T0:6| T0:7| T1:4| T1:5| T1:6| T1:7,  # 第1行展平
#   T2:0| T2:1| T2:2| T2:3| T3:0| T3:1| T3:2| T3:3,  # 第2行展平
#   T2:4| T2:5| T2:6| T2:7| T3:4| T3:5| T3:6| T3:7,  # 第3行展平
#  ...
#   T28:0|T28:1|T28:2|T28:3|T29:0|T29:1|T29:2|T29:3,
#   T28:4|T28:5|T28:6|T28:7|T29:4|T29:5|T29:6|T29:7,
#   T30:0|T30:1|T30:2|T30:3|T31:0|T31:1|T31:2|T31:3,
#   T30:4|T30:5|T30:6|T30:7|T31:4|T31:5|T31:6|T31:7]  # 第31行展平
# ```

# 第二步: 删除每行内每个线程的重复寄存器映射, 只保留每个线程的第一个寄存器:
# ```
# [ T0:0| T1:0,  # 第0行: 只保留每个线程的第0个寄存器
#   T0:1| T1:1,  # 第1行: 只保留每个线程的第1个寄存器
#   T2:0| T3:0,  # 第2行: 只保留每个线程的第0个寄存器
#   T2:1| T3:1,  # 第3行: 只保留每个线程的第1个寄存器
#  ...
#   T28:0|T29:0,
#   T28:1|T29:1,
#   T30:0|T31:0,
#   T30:1|T31:1]  # 第31行: 只保留每个线程的第1个寄存器
# ```

# 这样就得到了 1D SliceLayout。它的含义是: 
# 如果对 2D tensor 沿列维度做归约, 结果的每个元素会"广播"到多个线程(这里是 2 个线程)。

# 反过来, 从 1D 扩展到 2D 时, 会将每个元素复制多次以填满整行。
# 由于这些操作只涉及寄存器的逻辑映射(不需要实际的内存读写), 所以广播是零开销的。

@gluon.jit
def memcpy_2d_kernel(in_ptr, out_ptr,  #
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    # 创建行索引和列索引:
    # - indices_x: 行方向的索引, 形状为 [XBLOCK], 例如 [0, 1, 2, ..., XBLOCK-1]
    # - indices_y: 列方向的索引, 形状为 [YBLOCK], 例如 [0, 1, 2, ..., YBLOCK-1]
    #
    # 为什么使用 SliceLayout?
    # 原始的 2D layout 是 [行, 列] 的布局。但我们需要:
    # - indices_x: 只有行维度, 需要从 2D layout 中"切掉"列维度(dim=1), 得到行方向的 1D layout
    # - indices_y: 只有列维度, 需要从 2D layout 中"切掉"行维度(dim=0), 得到列方向的 1D layout
    #
    # 具体做法:
    # - SliceLayout(dim=1, parent=layout): 切掉第1维(列维度), 保留第0维(行维度)
    # - SliceLayout(dim=0, parent=layout): 切掉第0维(行维度), 保留第1维(列维度)
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    # 接下来, 我们需要将行偏移量乘以行步幅, 列偏移量乘以列步幅, 然后相加得到最终的偏移地址。
    # 其中涉及到广播机制, 下面对in_offsets计算过程进行详解:
    # 
    # 假设 XBLOCK=4, YBLOCK=3, start_x=0, start_y=0:
    # 1. indices_x 的形状是 [4], 值为 [0, 1, 2, 3]
    #    indices_x[:, None] 将其扩展为shape:[4, 1]
    #    [[0],
    #     [1],
    #     [2],
    #     [3]]
    #
    # 2. indices_y 的形状是 [3], 值为 [0, 1, 2]
    #    indices_y[None, :] 将其扩展为shape:[1, 3]
    #    [[0, 1, 2]]
    #
    # 3. indices_x[:, None]+indices_y[None, :]会进行广播, 两者都变成 [4, 3]:
    #    indices_x[:, None] 广播为:
    #    [[0, 0, 0],   # 第0行的所有列都使用行索引0
    #     [1, 1, 1],   # 第1行的所有列都使用行索引1
    #     [2, 2, 2],   # 第2行的所有列都使用行索引2
    #     [3, 3, 3]]   # 第3行的所有列都使用行索引3
    #
    #    indices_y[None, :] 广播为:
    #    [[0, 1, 2],   # 第0行的列索引
    #     [0, 1, 2],   # 第1行的列索引
    #     [0, 1, 2],   # 第2行的列索引
    #     [0, 1, 2]]   # 第3行的列索引
    #
    # 4. 计算内存偏移量:
    #    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    #    假设 xstride_in=100, ystride_in=1 (行主序存储):
    #    in_offsets = 100 * [[0,0,0], [1,1,1], [2,2,2], [3,3,3]] 
    #                 + 1 * [[0,1,2], [0,1,2], [0,1,2], [0,1,2]]
    #               = [[0,   1,   2],   # 第0行: 偏移量 0, 1, 2
    #                 [100, 101, 102], # 第1行: 偏移量 100, 101, 102
    #                 [200, 201, 202], # 第2行: 偏移量 200, 201, 202
    #                 [300, 301, 302]] # 第3行: 偏移量 300, 301, 302
    #
    # 这样我们就得到了每个元素在内存中的偏移量, 形状为 [XBLOCK, YBLOCK]。
    # 注意: expand_dims 操作(即 [:, None] 和 [None, :]),会返回具有父layout的tensor,
    # 因此广播后的tensor仍然保持与原始2D tensor 兼容的layout。
    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    # 掩码的计算方式相同: 将1D的边界检查广播到2D
    # 例如, 如果 xnumel=100, ynumel=50:
    # - indices_x[:, None] < xnumel 检查每一行是否在边界内
    # - indices_y[None, :] < ynumel 检查每一列是否在边界内
    # - 两者做 & 运算, 得到 [XBLOCK, YBLOCK] 形状的掩码
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, value, mask=mask)


def memcpy_2d_impl(input, output, XBLOCK, YBLOCK, layout, num_warps):
    xnumel, ynumel = input.shape
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    # 将输入和output tensor的步幅传递给kernel。当步幅为 1 时(这在tensor的内部维度中很常见), 
    # 编译器会针对这种情况生成专门优化的代码, 从而提升性能。
    compiled_kernel = memcpy_2d_kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(), *output.stride(),  #
        layout, XBLOCK, YBLOCK, num_warps=num_warps)
    return compiled_kernel


@pytest.mark.parametrize("XBLOCK, YBLOCK", [(128, 256), (256, 128)])
@pytest.mark.parametrize("xnumel, ynumel", [(100, 2000), (1000, 200)])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("num_warps", [4])
def test_memcpy_2d(XBLOCK, YBLOCK, xnumel, ynumel, transposed, num_warps):
    torch.manual_seed(0)
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty_like(input)
    input = input.T if transposed else input
    output = output.T if transposed else output
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    memcpy_2d_impl(input, output, XBLOCK, YBLOCK, layout, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# 接下来, 我们对2D memcpy进行性能测试。
def bench_memcpy_2d(impl, transposed=False):
    # 8GB 大小的tensor: [32*1024, 64*1024]
    xnumel = 32 * 1024
    ynumel = 64 * 1024
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty_like(input)
    input = input.T if transposed else input
    output = output.T if transposed else output
    return bench_memcpy_impl(input, output, impl)


# 这里我们不进行自动调优, 而是根据1D memcpy时的发现, 选择一个与 R=1 的layout行为相同的 blocked layout。
# 选择 XBLOCK=1 意味着每个程序处理一行数据。
if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    print("Benchmarking 2D memcpy")
    print("======================")
    XBLOCK = 1
    YBLOCK = 2048
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=XBLOCK, YBLOCK=YBLOCK, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_2d(impl)
    print(f"Throughput: {throughput:.3f} TB/s")

# 性能为6.260 TB/s, 比1D memcpy慢5%。有各种原因, 例如更复杂的 2D 算术, 下面让我们深入挖掘一下原因。


# %%
# 我们的 2D memcpy kernel有一个问题: 最佳layout取决于全局内存中tensor的layout。
# 让我们尝试对输入 tensor 和输出 tensor 进行转置, 即设置参数 transposed=True，测试其吞吐量: 
# 注: 转置后tensor内部维度不连续, 即每一行的元素在内存中不是连续的.

if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    _, throughput = bench_memcpy_2d(impl, transposed=True)
    print(f"Transposed throughput: {throughput:.3f} TB/s")

# 性能骤降至 0.774 TB/s。因为内部维度不再连续, 无法进行访存合并。


# 简单地交换XBLOCK和YBLOCK，并转置layout即可恢复性能: 
# gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0]) -> gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])

if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_2d(impl, transposed=True)
    print(f"Fixed throughput: {throughput:.3f} TB/s")
    print()


# 性能提升到6.590 TB/s, 比 1D memcpy 都要快一些。
#
# 在非转置input_tensor及layout, 以及转置input_tensor及layout这两种情况下, 每个程序都访问相同的数据。
# 性能的变化来自于数据局部性, 虽然每个 block 访问的数据量和访问模式相同, 但由于执行顺序不同, 缓存命中率会有差异。
#
# 在后续教程中, 我们将探索实现持久性kernel(persistent kernels)以及如何使用它们来更好地控制调度, 以提高性能。



# %%
# 从上面的数据可以看出, 在输入和output tensor都连续时, 1D memcpy 比 2D memcpy 性能略好.
# 然而, 当输入或输出的layout不连续或更奇形怪状时, 使用 2D memcpy 会有更佳的表现。
#
# 举个例子: 假设我们有一个大小为8GB的tensor, 通过`x[::2]`获取每隔一行的视图, 作为input tensor, 其是非连续的。
# 
# Python 切片语法详解: [start:stop:step]
# - start: 起始位置(默认0)
# - stop: 结束位置(默认到末尾)
# - step: 步长(默认1)
#
# `input[::2]` 的含义:
# - 省略 start 和 stop, 表示从开始到结束
# - step=2 表示每隔一个元素取一个
# - 对于2D tensor, 这表示每隔一行取一行
#
# 具体例子: 假设原始 tensor 形状为 [6, 4]
# ```
# input = [[0,  1,  2,  3],   # 第0行
#          [4,  5,  6,  7],   # 第1行
#          [8,  9,  10, 11],  # 第2行
#          [12, 13, 14, 15],  # 第3行
#          [16, 17, 18, 19],  # 第4行
#          [20, 21, 22, 23]]  # 第5行
# ```
#
# input[::2] 会选取索引为 0, 2, 4 的行(每隔一行):
# ```
# input[::2] = [[0,  1,  2,  3],   # 第0行
#               [8,  9,  10, 11],  # 第2行
#               [16, 17, 18, 19]]  # 第4行
# ```
#
# 为什么这会导致非连续的内存布局?
# 原始 tensor 在内存中是连续存储的:
# ```
# 内存地址: [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23]
#            └─第0行─┘ └─第1行─┘ └─第2行─┘  └─第3行─┘  └─第4行─┘  └─第5行─┘
# ```
#
# input[::2] 在内存中跳过了第1、3、5行:
# ```
# 内存地址: [0,1,2,3, ..., 8,9,10,11, ..., 16,17,18,19, ...]
#            └─第0行─┘      └─第2行─┘       └─第4行─┘
# ```
#
# 因此, 相邻的行在内存中不是连续的, 这导致 tensor 变成非连续的。
#
# 下面我们将这个非连续的input tensor复制到一个连续的output tensor中, 相当于 PyTorch 的 `x.contiguous()` 操作。
# 不过我们手写的2D memcpy 的性能比 PyTorch 的 `x.contiguous()`实现快得多。

if __name__ == "__main__" and _enabled("memcpy_2d_contig"):
    print("Non-contiguous memcpy")
    print("=====================")
    # 8GB大小的input tensor。
    xnumel = 32 * 1024
    ynumel = 64 * 1024
    input = torch.randn((xnumel, ynumel), device="cuda")
    # 获取每隔一行的视图, 即每隔一行取一行。
    input = input[::2]
    output = torch.empty_like(input)
    assert not input.is_contiguous() and output.is_contiguous()

    # 1. 基准测试 2D memcpy。
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=1, YBLOCK=2048, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy: {throughput:.3f} TB/s")

    # 2. 基准测试 PyTorch contiguous。
    fn = lambda: input.contiguous()
    ms = triton.testing.do_bench(fn)
    throughput = get_throughput(input, ms)
    print(f"torch.Tensor.contiguous: {throughput:.3f} TB/s")

    # 3. 利用"转置"技巧来获得更高的性能。
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input.T, output.T, impl)
    print(f"2D memcpy (transposed): {throughput:.3f} TB/s")
    print()

# ```
# 2D memcpy: 6.258 TB/s
# torch.Tensor.contiguous: 2.946 TB/s
# 2D memcpy (transposed): 6.398 TB/s
# ```
#
# 即使输入是非连续的, 我们的 2D memcpy 仍能接近tensor连续时的性能, 且比 PyTorch 的`contiguous()`快2倍以上。


# %%
# 上面的例子说明: layout 选择对全局内存访问性能很关键, 而最佳 layout 取决于 tensor 的内存布局。
# 现在考虑更复杂的情况: 如果输入和输出的连续维度相反怎么办？

if __name__ == "__main__" and _enabled("memcpy_2d_inout"):
    print("2D memcpy in/out layouts")
    print("=========================")

    # input 沿 dim 1 连续。
    input = torch.randn((32 * 1024, 32 * 1024), device="cuda")

    # output 进行了转置操作, 沿 dim 0 连续。
    output = torch.empty((input.shape[1], input.shape[0]), device="cuda").T
    
    # 问题: 输入和输出具有相反的连续方向! 无论我们选择哪种layout, 总有一个方向无法获得合并访问
    #
    # 尝试1: 使用 order=[1, 0], 适合按行访问
    # 这对input tensor很好(按行访问, 获得合并), 但对output tensor很差(按行访问, 无法合并)
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=1, YBLOCK=2048, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy (order=[1, 0]): {throughput:.3f} TB/s")

    # 尝试2: 使用 order=[0, 1], 适合按列访问
    # 这对output tensor很好(按列访问, 获得合并), 但对input tensor很差(按列访问, 无法合并)
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy (order=[0, 1]): {throughput:.3f} TB/s")

# 无论我们选择哪种layout, 性能都很糟糕: 

# ```
# 2D memcpy (order=[1, 0]): 0.978 TB/s  # 输入合并, 输出不合并
# 2D memcpy (order=[0, 1]): 1.674 TB/s  # 输入不合并, 输出合并
# ```
#
# 解决方案: 
# 为 `gl.load` 和 `gl.store` 使用两种不同的layout, 这样两者都能获得合并访问。
#
# 具体实现:
# 1. input tensor 选择适合行访问的layout(优化读取)
# 2. output tensor 选择适合列访问的layout(优化写入)
# 3. 在写入 output tensor 前进行layout转换
# 
# Layout转换的原因和成本:
# 从全局内存加载数据时需要input tensor的layout, 但存储时需要 output tensor 的 layout。
# 因此, 当我们为输入和输出使用不同的 layout 时, 需要在两者之间执行 layout 转换。
#
# Layout转换的成本:
# - 通常需要跨线程和线程束的数据移动
# - 跨线程束的数据移动需要使用共享内存(这是GPU上的宝贵资源)
# - 可能降低占用率和最大流水线深度, 影响性能
#
# 在我们的案例中, Layout转换的成本是不可避免的(因为输入和输出的布局不同).
# 不过这个成本远低于低效全局内存访问的成本(无法合并访问)，因此进行layout转换是值得的。
#


def get_layout_for_gmem_access(tensor, num_warps):
    """
    根据tensor的内存布局返回最适合该tensor的layout, 以优化全局内存访问。
    
    判断逻辑:
    - 如果 tensor.stride(1) == 1: 列方向连续(行主序), 使用 order=[1, 0]
    - 如果 tensor.stride(0) == 1: 行方向连续(列主序), 使用 order=[0, 1]
    """
    # 1D tensor
    if len(tensor.shape) == 1:
        return gl.BlockedLayout([1], [32], [num_warps], [0])

    assert len(tensor.shape) == 2, "only 1D and 2D tensors are supported"
    assert 1 in tensor.stride(), "expected at least 1 contiguous dimension"
    if tensor.stride(1) == 1:
        # 列方向连续(行主序), 按行访问获得合并
        return gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    else:
        # 行方向连续(列主序), 按列访问获得合并
        return gl.BlockedLayout([1, 1], [32, 1], [num_warps, 1], [0, 1])


@gluon.jit
def get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride, ystride,  #
                         XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr):
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)
    offsets = xstride * indices_x[:, None] + ystride * indices_y[None, :]
    return mask, offsets


@gluon.jit
def memcpy_2d_inout_kernel(in_ptr, out_ptr,  #
                           xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                           layout_in: gl.constexpr, layout_out: gl.constexpr,  #
                           XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    # 为输入和输出分别计算索引和掩码(使用各自的layout)。
    mask_in, in_offsets = get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride_in, ystride_in,  #
                                               XBLOCK, YBLOCK, layout_in)
    mask_out, out_offsets = get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride_out, ystride_out,  #
                                                 XBLOCK, YBLOCK, layout_out)

    # 使用input tensor的layout加载数据(获得合并访问)
    value = gl.load(in_ptr + in_offsets, mask=mask_in)

    # 转换数据的layout: 从输入layout转换为输出layout
    # 注意: 如果输入和输出的layout恰好相同, 编译器会自动优化掉多余的代码和layout转换。
    value = gl.convert_layout(value, layout_out)

    # 使用output tensor的layout存储数据(获得合并访问)
    gl.store(out_ptr + out_offsets, value, mask=mask_out)


def memcpy_2d_inout(input, output, num_warps=4):
    assert input.shape == output.shape, "input and output must have the same shape"
    XBLOCK = 64
    YBLOCK = 128
    layout_in = get_layout_for_gmem_access(input, num_warps)
    layout_out = get_layout_for_gmem_access(output, num_warps)
    grid = (triton.cdiv(input.shape[0], XBLOCK), triton.cdiv(input.shape[1], YBLOCK))
    return memcpy_2d_inout_kernel[grid](  #
        input, output,  #
        input.shape[0], input.shape[1],  #
        *input.stride(), *output.stride(),  #
        layout_in, layout_out,  #
        XBLOCK, YBLOCK, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, ynumel", [(300, 400)])
@pytest.mark.parametrize("transpose_in, transpose_out", [(True, False), (False, True)])
def test_memcpy_2d_inout(xnumel, ynumel, transpose_in, transpose_out):
    torch.manual_seed(0)
    if transpose_in:
        input = torch.randn((ynumel, xnumel), device="cuda").T
    else:
        input = torch.randn((xnumel, ynumel), device="cuda")
    if transpose_out:
        output = torch.empty((ynumel, xnumel), device="cuda").T
    else:
        output = torch.empty((xnumel, ynumel), device="cuda")
    memcpy_2d_inout(input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


if __name__ == "__main__" and _enabled("memcpy_2d_inout"):
    _, throughput = bench_memcpy_impl(input, output, memcpy_2d_inout)
    print(f"2D memcpy (in/out layouts): {throughput:.3f} TB/s")

# 性能结果
# ==========================
#
# ```
# 2D memcpy (order=[1, 0]): 0.978 TB/s
# 2D memcpy (order=[0, 1]): 1.674 TB/s
# 2D memcpy (in/out layouts): 4.814 TB/s
# ```

# 这个性能比之前单一layout的方法(0.978 TB/s 或 1.674 TB/s)好得多, 
# 虽然比完全连续的情况(6.258 TB/s)稍慢, 但这是可以接受的, 因为包含了转换成本。
# 我们将在后续教程中学习如何通过流水线优化来隐藏这个成本。


# %%
# Layout对性能的其他影响
# ==========================
#
# 除了全局内存访问, layout还会影响其他操作的性能:

# 1. **归约(Reductions)、扫描(Scans)、收集(Gathers)等操作**
#    这些操作通常需要跨线程和线程束的通信。如果选择输入layout以减少通信量, 效率会更高。

#    例子: 沿列维度对 `128x128` tensor进行归约
   
#    如果layout是 `gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])`:
#    - 一行中的每个元素由不同的线程拥有
#    - 需要线程间通信: 编译器生成蝶形洗牌(butterfly shuffles)在每个线程束内归约
#    - 然后通过共享内存归约每行剩余的4个值
   
#    如果layout是 `gl.BlockedLayout([1, 128], [32, 1], [4, 1], [0, 1])`:
#    - 每个线程正好拥有tensor的一行
#    - 归约不需要线程间通信, 效率更高

#    注意: 编译器在生成高效的归约代码方面做得很好, 通常不需要为了归约而转换layout。
#    但在可以选择的情况下, 选择高效的归约layout仍然是有益的。

# 2. **共享内存访问**
#    共享内存被组织成多个bank, 每个周期每个线程束只能访问一个bank地址。
#    编译器会生成最小化bank冲突的代码, 但bank冲突的数量仍然受layout影响。


# %%
# Layout的等价性和linear layout
# ============================

# **Layout的等价性**
# 在Gluon中, 多个layout可以表示相同的tensor元素映射。例如, 以下两个layout是等效的:

# ```python
# gl.BlockedLayout([1], [32], [4], [0])
# gl.SliceLayout(1, gl.BlockedLayout([1, 1], [32, 1], [4, 1], [1, 0]))
# ```

# 当你在已知等效的layout之间进行转换时, 或者转换只需要在线程内重新排序寄存器(这是免费的)时,
# 可以使用 `gl.convert_layout(x, layout, assert_trivial=True)` 来确保转换是零成本的。

# **linear layout (Linear Layouts)**
# 虽然Gluon没有规范的layout表示, 但所有Gluon layout都可以表示为linear layout。
# linear layout是Gluon中最具表现力和最强大的layout表示:
# - 允许表达零成本的拆分、连接、重塑和排列操作
# - 可以与高维tensor(例如5D或7D)和重塑结合使用
# - 在kernel内执行合并加载和高效的数据转换

# 但是, linear layout相对不常见, 并且可能难以理解。

# 示例: 上述两个等效layout的linear layout表示:
# ```python
# gl.DistributedLinearLayout(
#   reg_bases=[],
#   lane_bases=[[1], [2], [4], [8], [16]],
#   warp_bases=[[32], [64]],
#   block_bases=[],
#   shape=[128],
# )
# ```

# 这个linear layout是1D tensor元素索引位上的7x7单位矩阵,
# 其中低5位解释为通道(lane), 高2位解释为线程束(warp)。

# 更多信息:
# - 数据结构: `include/triton/Tools/LinearLayout.h`
# - 相关论文: https://arxiv.org/abs/2505.23819

# %%
# 总结
# ====

# 本教程的主要收获:

# 1. **显式Layout管理**: Gluon需要显式的layout管理, 有多种类型的layout用于不同的目的
#    - BlockedLayout: 最常用的layout, 用于表示分块的数据分布
#    - SliceLayout: 从高维layout中提取低维layout
#    - LinearLayout: 最强大的layout表示, 支持零成本操作

# 2. **Layout对性能的影响**: Layout影响性能, 有时影响巨大
#    - 全局内存访问: 正确的layout可以获得合并访问, 提升带宽利用率
#    - 线程间通信: 合适的layout可以减少通信开销
#    - 共享内存访问: Layout影响bank冲突的数量

# 3. **Layout是强大的工具**: 通过合理选择layout, 可以编写灵活且高性能的kernel
