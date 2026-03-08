# copy from https://github.com/NVIDIA/cutile-python/blob/main/samples/VectorAddition.py

# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
import math


ConstInt = ct.Constant[int]


@ct.kernel
def vec_add_kernel_1d(a, b, c, TILE: ConstInt):
    """
    cuTile kernel: 使用 tile-based load/store 实现一维逐元素向量加法。

    每个 block 处理向量中大小为 TILE 的一个分块。
    要求向量长度是 TILE 的整数倍, 或由调用方通过 padding 处理越界。
    """
    # 获取当前 block 在第一个维度上的 ID,
    # 在一维 grid 中直接对应 tile 索引
    bid = ct.bid(0)

    # ct.load 按 tile index 从全局内存搬运数据,
    # 硬件自动将加载操作分配给 block 内各线程
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))

    # 逐元素加法, 在 block 内各线程间并行执行
    sum_tile = a_tile + b_tile

    # 将结果 tile 写回全局内存
    ct.store(c, index=(bid,), tile=sum_tile)


@ct.kernel
def vec_add_kernel_2d(a, b, c, TILE_X: ConstInt, TILE_Y: ConstInt):
    """
    cuTile kernel: 使用 tile-based load/store 实现二维逐元素矩阵加法。

    每个 block 计算矩阵中 TILE_X x TILE_Y 大小的一个分块。
    与一维类似, 要求矩阵维度是 tile 大小的整数倍。
    """
    # bid(0) 对应行方向, bid(1) 对应列方向
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    # index=(bid_x, bid_y) 指定要加载的二维 tile 位置
    a_tile = ct.load(a, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))
    b_tile = ct.load(b, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))

    sum_tile = a_tile + b_tile

    ct.store(c, index=(bid_x, bid_y), tile=sum_tile)


@ct.kernel
def vec_add_kernel_1d_gather(a, b, c, TILE: ConstInt):
    """
    cuTile kernel: 使用 gather/scatter 实现一维逐元素向量加法。

    与 tile-based 版本不同, gather/scatter 自动处理越界访问
    (越界读返回 0, 越界写被忽略), 适用于长度不对齐的情况。
    """
    bid = ct.bid(0)

    # 手动计算全局索引: block 起始偏移 + 局部偏移 [0, 1, ..., TILE-1]
    indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

    # gather 自动将越界元素置零, 无需额外的边界检查
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)

    sum_tile = a_tile + b_tile

    # scatter 只写入边界内的位置, 越界写被忽略
    ct.scatter(c, indices, sum_tile)


@ct.kernel
def vec_add_kernel_2d_gather(
    a, b, c,
    TILE_X: ConstInt, TILE_Y: ConstInt,
):
    """
    cuTile kernel: 使用 gather/scatter 实现二维逐元素矩阵加法。

    自动处理越界访问, 适用于维度不对齐的情况。
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    # 计算当前 tile 内 X 和 Y 方向的全局索引
    x = bid_x * TILE_X + ct.arange(TILE_X, dtype=torch.int32)
    y = bid_y * TILE_Y + ct.arange(TILE_Y, dtype=torch.int32)

    # reshape 为 (TILE_X, 1) 和 (1, TILE_Y), 广播到 (TILE_X, TILE_Y)
    x = x[:, None]
    y = y[None, :]

    a_tile = ct.gather(a, (x, y))
    b_tile = ct.gather(b, (x, y))

    sum_tile = a_tile + b_tile

    ct.scatter(c, (x, y), sum_tile)


def vec_add(a: torch.Tensor, b: torch.Tensor, use_gather: bool = False) -> torch.Tensor:
    """
    对两个张量 (1D/2D) 执行逐元素加法。

    根据维度和 use_gather 选择合适的 kernel:
    - use_gather=False: tile-based load/store, 简单高效, 但要求维度对齐
    - use_gather=True: gather/scatter, 自动处理越界, 推荐用于非对齐维度
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")
    if a.dim() > 2 or b.dim() > 2:
        raise ValueError("This function currently supports only 1D or 2D tensors.")
    if a.device != b.device:
        raise ValueError("Input tensors must be on the same device.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")
    if a.dtype != b.dtype:
        raise ValueError("Input tensors must have the same data type.")

    c = torch.empty_like(a)

    if a.dim() == 1:
        N = a.shape[0]

        # tile 大小取 next power of 2 (上限 1024), 对齐 GPU 访存模式
        TILE = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1

        # ceil(N / TILE) 个 block 覆盖整个向量
        grid = (math.ceil(N / TILE), 1, 1)

        kernel = vec_add_kernel_1d_gather if use_gather else vec_add_kernel_1d
        ct.launch(torch.cuda.current_stream(), grid, kernel, (a, b, c, TILE))
    else:  # 2D
        M, N = a.shape

        # TILE_Y 根据列维度取 next power of 2 (上限 1024)
        TILE_Y = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1
        # TILE_X 使得每个 block 处理的总元素数约为 1024
        TILE_X = max(1, 1024 // TILE_Y)

        if TILE_X * TILE_Y > 1024 and TILE_X > 1:
            TILE_X = 1024 // TILE_Y
            if TILE_X == 0:
                TILE_X = 1

        grid = (math.ceil(M / TILE_X), math.ceil(N / TILE_Y), 1)

        kernel = vec_add_kernel_2d_gather if use_gather else vec_add_kernel_2d
        ct.launch(torch.cuda.current_stream(), grid, kernel, (a, b, c, TILE_X, TILE_Y))

    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    print("--- 运行 cuTile 向量/矩阵加法示例 ---")

    VECTOR_SIZE_1D = 1_000_000
    MATRIX_SHAPE_2D = (2048, 1024)

    # --- 测试用例 1: 一维向量加法 (tile-based) ---
    print("\n--- 测试 1: 一维向量加法 (tile-based) ---")
    a_1d_direct = torch.randn(VECTOR_SIZE_1D, dtype=torch.float32, device='cuda')
    b_1d_direct = torch.randn(VECTOR_SIZE_1D, dtype=torch.float32, device='cuda')
    print(f"Input 1D shape: {a_1d_direct.shape}, dtype: {a_1d_direct.dtype}")

    c_1d_cutile_direct = vec_add(a_1d_direct, b_1d_direct, use_gather=False)
    print(
        f"""cuTile Output 1D shape: {c_1d_cutile_direct.shape},
        dtype: {c_1d_cutile_direct.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_1d_cutile_direct, a_1d_direct + b_1d_direct)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- 测试用例 2: 一维向量加法 (gather/scatter) ---
    # 故意使用不能被 TILE 整除的大小, 验证边界处理
    print("\n--- 测试 2: 一维向量加法 (gather/scatter) ---")
    VECTOR_SIZE_1D_GATHER = 1_000_001
    a_1d_gather = torch.randn(VECTOR_SIZE_1D_GATHER, dtype=torch.float32, device='cuda')
    b_1d_gather = torch.randn(VECTOR_SIZE_1D_GATHER, dtype=torch.float32, device='cuda')
    print(f"Input 1D (gather) shape: {a_1d_gather.shape}, dtype: {a_1d_gather.dtype}")

    c_1d_cutile_gather = vec_add(a_1d_gather, b_1d_gather, use_gather=True)
    print(
        f"""cuTile Output 1D (gather) shape: {c_1d_cutile_gather.shape},
        dtype: {c_1d_cutile_gather.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_1d_cutile_gather, a_1d_gather + b_1d_gather)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- 测试用例 3: 二维矩阵加法 (tile-based) ---
    print("\n--- 测试 3: 二维矩阵加法 (tile-based) ---")
    a_2d_direct = torch.randn(MATRIX_SHAPE_2D, dtype=torch.float32, device='cuda')
    b_2d_direct = torch.randn(MATRIX_SHAPE_2D, dtype=torch.float32, device='cuda')
    print(f"Input 2D shape: {a_2d_direct.shape}, dtype: {a_2d_direct.dtype}")

    c_2d_cutile_direct = vec_add(a_2d_direct, b_2d_direct, use_gather=False)
    print(
        f"""cuTile Output 2D shape: {c_2d_cutile_direct.shape},
        dtype: {c_2d_cutile_direct.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_2d_cutile_direct, a_2d_direct + b_2d_direct)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- 测试用例 4: 二维矩阵加法 (gather/scatter) ---
    # 故意使用不对齐的维度, 验证边界处理
    print("\n--- 测试 4: 二维矩阵加法 (gather/scatter) ---")
    MATRIX_SHAPE_2D_GATHER = (2000, 1000)
    a_2d_gather = torch.randn(MATRIX_SHAPE_2D_GATHER, dtype=torch.float32, device='cuda')
    b_2d_gather = torch.randn(MATRIX_SHAPE_2D_GATHER, dtype=torch.float32, device='cuda')
    print(f"Input 2D (gather) shape: {a_2d_gather.shape}, dtype: {a_2d_gather.dtype}")

    c_2d_cutile_gather = vec_add(a_2d_gather, b_2d_gather, use_gather=True)
    print(
        f"""cuTile Output 2D (gather) shape: {c_2d_cutile_gather.shape},
        dtype: {c_2d_cutile_gather.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_2d_cutile_gather, a_2d_gather + b_2d_gather)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile 向量/矩阵加法示例完成 ---")
