# copy from https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/softmax.py

import torch
import math
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]


@ct.kernel(occupancy=4)
def softmax_kernel(
    output,
    input,
    n_rows: ConstInt,
    TILE_SIZE: ConstInt,
    DIM_COLS: ConstInt,
):
    # 静态持久调度: 每个 block 处理多行, 通过 stride = num_blocks 循环分配
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)

    for row_idx in range(pid, n_rows, num_programs):
        # 用 gather 加载行数据, 越界位置填 -inf (不影响 softmax 结果)
        row = ct.gather(input, (row_idx, offsets), check_bounds=True, padding_value=-np.inf)
        # 转 float32 保证精度
        row = ct.astype(row, torch.float32)

        # 减去最大值防止 exp 溢出 (数值稳定性)
        row_max = ct.max(row, 0, keepdims=True)
        row_minus_max = ct.sub(row, row_max)

        # 计算 exp 和归一化
        numerator = ct.exp(row_minus_max)
        denominator = ct.sum(numerator, 0, keepdims=True)
        softmax_output = ct.truediv(numerator, denominator)

        # 转回原始 dtype 并存储
        softmax_output = ct.astype(softmax_output, input.dtype)
        ct.scatter(output, (row_idx, offsets), softmax_output, check_bounds=True)

# TMA 版本, 使用静态持久调度
@ct.kernel(occupancy=2)
def softmax_kernel_tma(
    output,
    input,
    n_rows: ConstInt,
    n_cols: ConstInt,
    TILE_SIZE: ConstInt,
):
    # 静态持久调度: 每个 block 处理多行
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    for row_idx in range(pid, n_rows, num_programs):
        # 用 TMA 一次加载整行 (TILE_SIZE >= n_cols), 越界位置填 -inf
        row = ct.load(input, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.NEG_INF)
        row = ct.astype(row, np.float32)

        # 减去最大值防止 exp 溢出
        row_max = ct.max(row, 1, keepdims=True)
        row_minus_max = ct.sub(row, row_max)

        numerator = ct.exp(row_minus_max)
        denominator = ct.sum(numerator, 1, keepdims=True)
        softmax_output = ct.truediv(numerator, denominator)

        softmax_output = ct.astype(softmax_output, input.dtype)
        ct.store(output, index=(row_idx, 0), tile=softmax_output)



def launch_softmax_kernel(input, output, TILE_SIZE=1024):
    """启动基础 cuTile softmax kernel (gather/scatter 版本, 静态持久调度)"""
    n_rows, n_cols = input.shape
    original_n_cols = n_cols

    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count
    occupancy = 4  # 需要与 @ct.kernel(occupancy=4) 保持一致
    num_programs = min(NUM_SM * occupancy, n_rows)
    grid = (num_programs, 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        softmax_kernel,
        (
            output,
            input,
            n_rows,
            TILE_SIZE,
            original_n_cols,
        ),
    )


def next_power_of_2(n: int):
    """返回 >= n 的最小 2 的幂"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def launch_softmax_kernel_tma(
    input,
    output,
):
    """启动 TMA 版本的 cuTile softmax kernel"""
    # 将输入统一为 2D
    original_shape = input.shape
    if input.dim() == 1:
        input = input.unsqueeze(0)
        output = output.unsqueeze(0)
    elif input.dim() > 2:
        input = input.view(-1, input.shape[-1])
        output = output.view(-1, output.shape[-1])

    n_rows, n_cols = input.shape

    TILE_SIZE = next_power_of_2(n_cols)
    original_n_cols = n_cols

    softmax_kernel_forward = softmax_kernel_tma

    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count
    occupancy = 2  # 需要与 @ct.kernel(occupancy=2) 保持一致
    num_programs = min(NUM_SM * occupancy, n_rows)
    grid = (num_programs, 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        softmax_kernel_forward,
        (
            output,
            input,
            n_rows,
            original_n_cols,
            TILE_SIZE,
        ),
    )


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        use_tma=False,
        ):
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)

        if use_tma:
            launch_softmax_kernel_tma(x, y)
        else:
            # 每行一个 tile 保证正确性
            TILE_SIZE = next_power_of_2(n_cols)
            launch_softmax_kernel(x, y, TILE_SIZE=TILE_SIZE)
        return y


def softmax(
    x,
    use_tma=False,
    **kwargs,
):
    """
    使用 cuTile kernel 执行 softmax。

    use_tma=True 时使用 TMA (Tensor Memory Accelerator) 实现,
    需要 H100+ GPU (计算能力 >= 9.0)。
    """
    return Softmax.apply(
        x,
        use_tma,
    )

DEVICE = torch.cuda.current_device()

# %%
# 单元测试
# ---------

# %%
# 使用不规则行列数 (1823, 781) 验证 padding 机制是否正确

torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)

if torch.allclose(softmax(x), torch.softmax(x, axis=1), atol=1e-2, rtol=0):
    print("✅ cuTile and Torch match")
else:
    print("❌ cuTile and Torch differ")

if torch.allclose(softmax(x, use_tma=True), torch.softmax(x, axis=1), atol=1e-2, rtol=0):
    print("✅ cuTile(tma) and Torch match")
else:
    print("❌ cuTile(tma) and Torch differ")

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import triton
import importlib.util
_spec = importlib.util.spec_from_file_location("triton_02_fused_softmax", os.path.join(os.path.dirname(__file__), "../triton/02-fused-softmax.py"))
_triton_softmax_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_triton_softmax_module)
triton_softmax = _triton_softmax_module.softmax
naive_softmax = _triton_softmax_module.naive_softmax

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# %%
# 性能基准测试
# ---------
#
# 以输入矩阵的列数为变量 (固定 4096 行),
# 对比 torch.softmax、naive_softmax、Triton、cuTile (gather) 和 cuTile (TMA) 的性能。


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['cutile', 'cutile-tma', "triton", 'torch', 'naive_softmax'],
        line_names=["cuTile", "cuTile TMA", "Triton", "Torch", "Naive Softmax"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-"), ("pink", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096},
    ))
def benchmark(M, N, provider):

    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    if provider == 'cutile':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'cutile-tma':
        ms = triton.testing.do_bench(lambda: softmax(x, use_tma=True))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)
