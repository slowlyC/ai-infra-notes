## Triton-10-Block-Scaled Matmul

### 前言

本系列教程记录了学习 Triton 的笔记，参考 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)，翻译并添加了一些自己的理解。

- Triton-01-Vector Addition
- Triton-02-Fused Softmax
- Triton-03-Matrix Multiplication
- Triton-04-Low-Memory Dropout
- Triton-05-Layer Normalization
- Triton-06-Fused Attention
- Triton-07-Extern Functions
- Triton-08-Grouped GEMM
- Triton-09-Persistent Matmul
- **Triton-10-Block-Scaled Matmul**（本文）
- Triton-11-Programmatic Dependent Launch

块缩放矩阵乘法
==================================
本教程演示了 Triton 实现的块缩放矩阵乘法，它对 FP4 和 FP8 格式通用, 支持 OCP 格式的 mxfp4 和 mxfp8，以及 NVIDIA 的 nvfp4 格式。

用户可以通过传递 `--format` 参数来运行每种支持的格式的教程，并可以通过指定矩阵维度和迭代步骤来对每种格式的性能进行基准测试。

```bash
# FP4
python 10-block-scaled-matmul.py --format nvfp4
python 10-block-scaled-matmul.py --format mxfp4 --K_range 512 8192 --bench

# FP8
python 10-block-scaled-matmul.py --format mxfp8 --K_range 8192 16384 --K_step 2048 --bench
```

未来计划更新本教程以支持混合精度块缩放矩阵乘法。

### 背景

支持 PTX 8.7 及更高版本的 CUDA 设备可以利用块缩放矩阵乘法指令。
为了在tensor核心 MMA 的快速内循环中低延迟访问这些缩放因子，
重要的是确保块缩放因子按照其访问模式以连续的内存layout存储。

块缩放矩阵乘法tensor核心指令计算以下乘积: 

    C = (A * scale_a) @ (B * scale_b)

其中 scale_a 和 scale_b 是 A 和 B 矩阵的块缩放因子。
在块缩放矩阵乘法下，每个缩放因子在 A 和 B 矩阵的一个向量元素上进行广播和相乘，
通常沿着它们各自的 K 轴。每个缩放因子广播的 A 和 B 元素的数量
在此称为向量大小 (VEC_SIZE)。

在线性行主layout中，缩放因子将采用以下形状

    (M, K // VEC_SIZE) 和 (N, K // VEC_SIZE)   [1]

在全局内存中。然而，为了避免非连续的内存访问，有利于
改为以打包块layout存储缩放因子。对于 LHS 矩阵，此layout为

    (M // 32 // 4, K // VEC_SIZE // 4, 32, 4, 4)   [2]。

通过这种方式，K 块的快速内循环中的每个tensor核心 MMA 可以实现
沿 M 轴的 128 行缩放因子块的连续访问，对于矩阵 A 的每个 BLOCK_M x BLOCK_K 子块。

为了符合 Triton 的 dot_scaled 语言语义，缩放因子
在上述 5D layout [2] 中准备，但随后在逻辑上转置并重塑为
tl.dot_scaled 期望的 2D layout [1]。

有关缩放因子layout的更详细信息，请参见
 1. https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
 2. https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout


```python
import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_desc,  #
        a_scale_desc,  #
        b_desc,  #
        b_scale_desc,  #
        c_desc,  #
        M: tl.constexpr,  #
        N: tl.constexpr,  #
        K: tl.constexpr,  #
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        rep_m: tl.constexpr,  #
        rep_n: tl.constexpr,  #
        rep_k: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
):  #
    # 确定输出数据类型
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    # 获取程序 ID 并计算块位置
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    # 混合精度标志
    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K 维度循环: 执行块缩放的点积
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        # 加载 A 和 B 块
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        # 加载缩放因子
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        # 重塑和转置缩放因子以匹配 dot_scaled 的预期layout
        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        # 执行块缩放的点积
        if MIXED_PREC:
            # 混合精度: FP8 x FP4
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            # FP4 x FP4
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            # FP8 x FP8
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        # 更新偏移量
        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    # 存储结果
    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs):
    """执行块缩放矩阵乘法"""
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    # 将 dtype 转换为整数代码
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 2
    else:
        raise ValueError(f"不支持的数据类型: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m,
        rep_n,
        rep_k,
        configs["num_stages"],
    )
    return output


def initialize_block_scaled(M, N, K, block_scale_type="nvfp4", compute_reference=False):
    """初始化块缩放矩阵乘法的输入和参数"""
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8", "mixed"], f"无效的块缩放类型: {block_scale_type}"
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = "cuda"
    a_ref = MXFP4Tensor(size=(M, K), device=device).random()
    # 类似于 Hopper 的 wgmma 对称 fp8 指令，当使用 fp4 操作数时，Blackwell 的 tcgen05.mma 期望 RHS 采用列主layout。
    # 为了符合 tl.dot_scaled 的预期语义，(M, K) x (K, N)，数据在列主layout中生成，沿 K 打包为 fp4，然后在逻辑上转置。
    # 请注意，如果一个操作数为 fp8 精度，与 Hopper 不同，Blackwell 支持 RHS 矩阵的行主和列主layout。
    # 对于混合精度情况，fp4 RHS 可以是行主或列主layout。但出于性能原因，建议使用列主layout。
    # 如果在混合精度点积中使用 TMA 加载 fp4 RHS 操作数（如本教程中），它必须采用列主layout。
    b_ref = MXFP4Tensor(size=(N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        # 沿 K 打包两个 fp4 元素到一个字节
        a = a_ref.to_packed_tensor(dim=1)

    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=1)

    b_ref = b_ref.to(torch.float32).T

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    # 创建缩放因子
    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]
    epsilon = 1e-8
    a_scale = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale = torch.rand(b_scale_shape, device=device) + epsilon
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    # 使用 5D TMA 描述符 [1, rep_m, rep_k, 2, 256] 与 uint8 元素。
    # 使用 256 个元素，我们可以更好地利用 L2，并且不需要 TMA 引擎发出许多小消息（16B）消息，就像 32x16xu8 一样。
    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]
    a_scale = a_scale.reshape(1, a_scale_shape[0], a_scale.shape[1], 2, 256)
    b_scale = b_scale.reshape(1, b_scale_shape[0], b_scale.shape[1], 2, 256)
    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=b_scale_block_shape)

    reference = None
    if compute_reference:
        # 计算参考结果
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale(packed):
            """解包缩放因子"""
            packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
            num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

        a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }
    return a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference


def validate_block_scaled(M, N, K, block_scale_type="nvfp4"):
    """验证块缩放矩阵乘法的正确性"""
    a_desc, a_scale, b_desc, b_scale, rep_m, rep_n, rep_k, configs, reference = initialize_block_scaled(
        M, N, K, block_scale_type, compute_reference=True)
    output = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    print(f"✅ (通过 {block_scale_type})")


def bench_block_scaled(K, block_scale_type="nvfp4", reps=10):
    """对块缩放矩阵乘法进行基准测试"""
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"问题规模 = {M}x{N}x{K}")

    a_desc, a_scale, b_desc, b_scale, rep_m, rep_n, rep_k, configs, _ = initialize_block_scaled(
        M, N, K, block_scale_type, compute_reference=False)
    _ = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)

    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)
    proton.deactivate(0)
    print("基准测试完成")


def show_profile(profile_name):
    """显示性能分析结果"""
    import triton.profiler.viewer as proton_viewer

    metric_names = ["time/ms"]
    metric_names = ["tflop/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--bench", action="store_true", default=True)
    parser.add_argument("--format", type=str, choices=["mxfp4", "nvfp4", "mxfp8", "mixed"], default="nvfp4")
    args = parser.parse_args()

    if not supports_block_scaling():
        print("⛔ 此示例需要 GPU 支持块缩放矩阵乘法")
    else:
        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # 只要不为 0 就无关紧要

        torch.manual_seed(42)

        validate_block_scaled(8192, 8192, 8192, block_scale_type=args.format)

        if args.bench:
            proton.start("block_scaled_matmul", hook="triton")
            proton.deactivate(0)  # 跳过参数创建
            for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
                bench_block_scaled(K, reps=10000, block_scale_type=args.format)
            proton.finalize()
            show_profile("block_scaled_matmul")
```
