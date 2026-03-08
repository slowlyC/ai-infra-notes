# copy from https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/matmul.py

import torch
import cuda.tile as ct
import math


ConstInt = ct.Constant[int]

def swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M):
    """将一维 block ID 映射到二维 (bid_m, bid_n), 同组内的 block 在 M 方向连续以改善 L2 局部性。"""
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(A, B, C,
                  TILE_SIZE_M: ConstInt,   # C 的行方向 tile 大小
                  TILE_SIZE_N: ConstInt,   # C 的列方向 tile 大小
                  TILE_SIZE_K: ConstInt):  # 内积维度 tile 大小
    """
    分块矩阵乘法 C = A @ B。
    每个 CTA 计算输出矩阵 C 中一个 TILE_SIZE_M x TILE_SIZE_N 的 tile,
    沿 K 维度以 TILE_SIZE_K 为步长迭代并用 ct.mma 累加。
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M)

    # ct.num_tiles 根据 A 的形状和 tile shape 计算 K 方向的 tile 数 = ceil(K / TILE_SIZE_K)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_SIZE_M, TILE_SIZE_K))

    # 即使输入是 float16, 也用 float32 累加以保证归约精度
    accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # fp32 转 tf32 以利用 Tensor Core
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K 维度循环: 每次加载 A 的 (TILE_SIZE_M, TILE_SIZE_K) 块和 B 的 (TILE_SIZE_K, TILE_SIZE_N) 块
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(TILE_SIZE_M, TILE_SIZE_K), padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(TILE_SIZE_K, TILE_SIZE_N), padding_mode=zero_pad).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)

    # 转换为输出 dtype 并存储
    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from autotuner import Autotuner, Config, autotune

def _matmul_autotune_configs():

    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        configs = [
            Config(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=32, num_ctas=1, occupancy=2),
        ]
    else:
        # sm100 (Blackwell)
        configs = [
            Config(TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=4, occupancy=1),
            Config(TILE_SIZE_M=512, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1),
        ]
    return configs

@autotune(search_space=_matmul_autotune_configs())
def matmul(a, b, autotuner: Autotuner | None = None):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    from math import ceil
    tuned_result = autotuner(
        torch.cuda.current_stream(),
        grid_fn=lambda named_args, cfg: (
            ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N),
            1,
            1,
        ),
        kernel=matmul_kernel,
        args_fn=lambda cfg: (a, b, c, cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, cfg.TILE_SIZE_K),
    )
    return c

DEVICE = torch.cuda.current_device()

# %%
# 单元测试
# ---------
#
# 将自定义矩阵乘法与 torch 原生实现 (cuBLAS) 对比。

torch.manual_seed(0)
a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
cutile_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"cutile_output_with_fp16_inputs={cutile_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")

if torch.allclose(cutile_output, torch_output, atol=1e-2, rtol=0):
    print("✅ cuTile and Torch match")
else:
    print("❌ cuTile and Torch differ")


TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8:
    torch.manual_seed(0)
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # 预转置 b 提升访存效率
    b = b.T
    b = b.to(torch.float8_e5m2)
    cutile_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"cutile_output_with_fp8_inputs={cutile_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(cutile_output, torch_output, atol=0.125, rtol=0):
        print("✅ cuTile and Torch match")
    else:
        print("❌ cuTile and Torch differ")

import triton
# %%
# 性能基准测试
# ---------
#
# 方阵性能测试: 对比 cuTile / Triton / cuBLAS (fp8 场景跳过 cuBLAS)。

ref_lib = 'cuBLAS'

configs = []
for fp8_inputs in [False, True]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[128 * i for i in range(2, 33)],
            line_arg="provider",
            # fp8 场景不与 cuBLAS 对比 (torch.matmul 暂不支持 fp8)
            line_vals=["cutile", "triton"] if fp8_inputs else ["cutile", "triton", ref_lib.lower()],
            line_names=["cuTile", "Triton"] if fp8_inputs else ["cuTile", "Triton", ref_lib],
            styles=[("orange", "-"), ("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),
            args={"fp8_inputs": fp8_inputs},
        ))

import importlib.util
import os
_spec = importlib.util.spec_from_file_location("triton_03_matrix_multiplication", os.path.join(os.path.dirname(__file__), "../triton/03-matrix-multiplication.py"))
_triton_matmul_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_triton_matmul_module)
triton_mutmul = _triton_matmul_module.matmul

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_mutmul(a, b), quantiles=quantiles)
    if provider == 'cutile':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True)
