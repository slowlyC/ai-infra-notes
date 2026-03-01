"""
融合 Softmax
=============

在本教程中，您将编写一个融合的 softmax 操作，对于特定类别的矩阵（其行可以放入 GPU 的 SRAM 中），
它比 PyTorch 的原生操作快得多。

通过本教程，您将学习到: 

* kernel融合对带宽受限操作的好处。

* Triton 中的归约运算符。

"""

# %%
# 动机
# -----------
#
# 用于逐元素加法的自定义 GPU kernel具有教育价值，但在实践中不会让你走得很远。
# 让我们考虑一个简单的（数值稳定的）softmax 操作的情况: 

import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def naive_softmax(x):
    """使用原生 PyTorch 计算 X 的按行 softmax: e^(x - max(x)) / sum(e^(x - max(x)))

    我们减去最大元素以避免溢出。Softmax 对这种偏移是不变的。
    """
    # 读取 MN 个元素；写入 M 个元素
    x_max = x.max(dim=1)[0]
    # 读取 MN + M 个元素；写入 MN 个元素
    z = x - x_max[:, None]
    # 读取 MN 个元素；写入 MN 个元素
    numerator = torch.exp(z)
    # 读取 MN 个元素；写入 M 个元素
    denominator = numerator.sum(dim=1)
    # 读取 MN + M 个元素；写入 MN 个元素
    ret = numerator / denominator[:, None]
    # 总共: 读取 5MN + 2M 个元素；写入 3MN + 2M 个元素
    return ret

# %%
# 在 PyTorch 中以简单方式实现时，为 :math:`x \in R^{M \times N}` 计算 :code:`y = naive_softmax(x)` 
# 需要从 DRAM 读取 :math:`5MN + 2M` 个元素，并写回 :math:`3MN + 2M` 个元素。
# 这显然是浪费的；我们更希望有一个自定义的"融合"kernel，只读取 X 一次，并在芯片上进行所有必要的计算。
# 这样做只需要读写 :math:`MN` 字节，因此我们可以预期约 4 倍的理论加速（即 :math:`(8MN + 4M) / 2MN`）。
# `torch.jit.script` 标志旨在自动执行这种"kernel融合"，但正如我们稍后将看到的，它仍然远非理想。


# %%
# 计算kernel
# --------------
#
# 我们的 softmax kernel工作如下: 每个程序加载输入矩阵 X 的一组行（按程序数跨步），对其进行归一化，并将结果写回输出 Y。
#
# 请注意，Triton 的一个重要限制是每个块必须具有 2 的幂次方个元素，
# 因此如果我们想要处理任何可能的输入形状，我们需要在内部"填充"每一行并正确保护内存操作: 

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # 程序的起始行
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # stride 表示我们需要增加指针多少才能前进 1 行
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # 块大小是大于 n_cols 的下一个 2 的幂，因此我们可以在单个块中放入每一行
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # 将行加载到 SRAM 中，使用掩码因为 BLOCK_SIZE 可能大于 n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # 为数值稳定性减去最大值
        row_minus_max = row - tl.max(row, axis=0)
        # 注意 Triton 中的指数运算很快但是近似的（即，类似 CUDA 中的 __expf）
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # 将输出写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


# %%
# 我们可以创建一个辅助函数，为任何给定的输入tensor将kernel及其（元）参数加入队列。

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]  # 78
NUM_REGS = properties["max_num_regs"]  # 65536
SIZE_SMEM = properties["max_shared_mem"]  # 232448
WARP_SIZE = properties["warpSize"]  # 32
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape
    # 每次循环迭代的块大小是大于 x 中列数的最小 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # 我们可以使用的另一个技巧是要求编译器通过增加每行分布的 warp 数量（`num_warps`）来使用更多线程。
    # 您将在下一个教程中看到如何以更自然的方式自动调整此值，这样您就不必自己提出手动启发式方法。
    num_warps = 8

    # 软件流水线阶段数。
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # 分配输出
    y = torch.empty_like(x)

    # 预编译kernel以获取寄存器使用情况并计算线程占用率。
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs  # 32
    size_smem = kernel.metadata.shared  # 12320
    if is_hip():
        # for AMD
        # NUM_REGS 表示常规用途寄存器的数量。在 CDNA 架构上，这是所有可用寄存器的一半。
        # 但是，这并非总是如此。在大多数情况下，所有寄存器都可以用作常规用途寄存器。
        # ISA 章节（CDNA3 的 3.6.4）
        # VGPR 从两个池中分配: 常规 VGPR 和累积 VGPR。累积 VGPR 用于矩阵 VALU 指令，也可以直接从内存加载。
        # 一个 wave 最多可以有 512 个总 VGPR，每种类型 256 个。
        # 当一个 wave 少于 512 个总 VGPR 时，每种类型的数量是灵活的 - 不要求两种类型的数量相等。
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS 表示每个多处理器的最大常驻线程数。
        # 当我们用这个数字除以 WARP_SIZE 时，我们得到可以在 CU（多处理器）上并行执行的最大 wave 数。
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)  # 65536 // (32 * 32 * 8) = 8
    occupancy = min(occupancy, SIZE_SMEM // size_smem)  # min(8, 18)
    num_programs = NUM_SM * occupancy  #  78 * 8 = 624

    num_programs = min(num_programs, n_rows)  # min(624, 1823) = 624

    # 创建若干个持久化程序。
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y


# %%
# 单元测试
# ---------

# %%
# 我们确保在具有不规则行数和列数的矩阵上测试我们的kernel。
# 这将使我们能够验证我们的填充机制是否有效。
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

# %%
# 正如预期的那样，结果是相同的。

# %%
# 基准测试
# ---------
#
# 在这里，我们将作为输入矩阵中列数的函数对我们的操作进行基准测试 -- 假设有 4096 行。
# 然后，我们将其性能与 (1) :code:`torch.softmax` 和 (2) 上面定义的 :code:`naive_softmax` 进行比较。


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作绘图 x 轴的参数名称
        x_vals=[128 * i for i in range(2, 100)],  # `x_name` 的不同可能值
        line_arg='provider',  # 其值对应于图中不同线条的参数名称
        line_vals=['triton', 'torch', 'naive_softmax'],  # `line_arg` 的可能值
        line_names=["Triton", "Torch", "Naive Softmax"],  # 线条的标签名称
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # 线条样式
        ylabel="GB/s",  # y 轴的标签名称
        plot_name="softmax-performance",  # 绘图的名称。也用作保存绘图的文件名。
        args={'M': 4096},  # 不在 `x_names` 和 `y_name` 中的函数参数的值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)

# %%
# 在上面的图中，我们可以看到: 
#  - Triton 比 Torch JIT 快 4 倍。这证实了我们的猜测，即 Torch JIT 在这里没有进行任何融合。
#  - Triton 明显快于 :code:`torch.softmax` -- 除了**更易于阅读、理解和维护**。
#    但请注意，PyTorch 的 `softmax` 操作更通用，可以处理任何形状的tensor。
