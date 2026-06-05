# Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from typing import Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import torch

"""
TMA V0: 理解 tma_partition, TMA 操作的基础

这个教程用一个简单的 copy kernel 演示 TMA (Tensor Memory Accelerator) 操作。
重点是理解基础的 tma_partition 接口, 它是所有 TMA 操作的入口。

本教程覆盖：
1. TMA Load (Global Memory -> Shared Memory)
2. TMA Store (Shared Memory -> Global Memory)
3. TMA 的 barrier synchronization (elect_one, mbarrier_init, mbarrier_arrive, mbarrier_wait)
4. 结合图示详细解释 tma_partition

学习重点：
- tma_partition: 它怎样为 TMA 操作转换 tensor
- group_modes: 为什么要合并 tensor modes, 以及怎样用它定义 TMA atom shape
- Indexing: 怎样从 partitioned tensor 中选择特定 tile
- Data flow: 从输入 tensor 到 TMA copy 的完整可视化

图示：
约 155 行处有完整图示, 说明：
- 输入 tensor 的 shape 和转换过程
- group_modes 对 tensor layout 的影响
- tma_partition 的输出结构
- 从 global/shared memory 到 TMA copy 的完整数据流
- 针对 CTA tile 的 indexing pattern

示例用法：
```bash
python cutlass_ir/compiler/python/examples/cute/blackwell/tutorial/tutorial_tma/tma_v0.py
```
"""


class Sm100SimpleCopyKernel:
    def __init__(self):
        """
        初始化 Blackwell TMA copy kernel 的配置。
        """
        self.tile_shape = (128, 128)
        self.tile_m, self.tile_n = self.tile_shape
        self.cluster_shape_mn = (1, 1)
        self.threads_per_cta = 32
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, src: cute.Tensor, dst: cute.Tensor):
        if cutlass.const_expr(src.element_type != dst.element_type):
            raise TypeError("Source and destination element types must match")

        self.dtype: Type[cutlass.Numeric] = src.element_type

        # 每个 CTA 使用的 layout: (tile_m, tile_n):(tile_n, 1)
        smem_layout = cute.make_layout(
            (self.tile_m, self.tile_n), stride=(self.tile_n, 1)
        )

        @cute.struct
        class SharedStorage:
            barrier_storage: cute.struct.MemRange[cutlass.Int64, 1]
            smem_data: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_layout)

        # cta_tiler: TMA 使用的 per-CTA tile 范围 (M, N)。
        # 因为 smem_layout 可能包含 swizzle 或 composed layout,
        # 因此用 product_each 沿每个逻辑维度取乘积, 得到 TMA 期望的最终 (tile_m, tile_n) 范围。
        # 在这个简单示例里, product_each(...) 的结果就是 smem_layout.shape = (tile_m, tile_n)。
        cta_tiler = cute.product_each(smem_layout.shape)

        tma_atom_src, tma_tensor_src = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), src, smem_layout, cta_tiler
        )

        tma_atom_dst, tma_tensor_dst = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dst, smem_layout, cta_tiler
        )

        # Grid shape 现在是 (M/TileM, N/TileN)
        grid_shape = cute.ceil_div((*src.layout.shape, 1), self.tile_shape)

        self.kernel(
            tma_atom_src, tma_tensor_src, tma_atom_dst, tma_tensor_dst, smem_layout
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_src: cute.CopyAtom,
        tma_tensor_src: cute.Tensor,
        tma_atom_dst: cute.CopyAtom,
        tma_tensor_dst: cute.Tensor,
        smem_layout: Union[cute.Layout, cute.ComposedLayout],
    ):
        bidx, bidy, _ = cute.arch.block_idx()

        # 分配 Shared Memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # 初始化用于 TMA 同步的 barrier
        barrier_ptr = storage.barrier_storage.data_ptr()

        # 初始化 barrier.
        # v0 的 CTA 只有一个 warp, elect_one() 会在这个 warp 中选出一个线程执行下面的代码,
        # 避免所有线程重复执行 mbarrier_init / mbarrier_expect_tx.
        # 多 warp CTA 中如果所有 warp 都会经过这里, 不能简单用 elect_one() 代替 CTA 唯一的 thread 条件.
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(barrier_ptr, 1)
            cute.arch.mbarrier_expect_tx(barrier_ptr, self.num_tma_load_bytes)

        # fence 保证 init/expect_tx 在继续执行前可见
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # 把 (M, N) tensor 切成 tile: ((TileM, TileN), M/TileM, N/TileN)
        gSrc_tiled = cute.local_tile(
            tma_tensor_src, (self.tile_m, self.tile_n), (None, None)
        )
        gDst_tiled = cute.local_tile(
            tma_tensor_dst, (self.tile_m, self.tile_n), (None, None)
        )

        smem_tensor = storage.smem_data.get_tensor(smem_layout)

        # ======================================================================
        # TMA Partition: 为 TMA 操作准备 tensor
        # ======================================================================
        #
        # tma_partition 会按照 TMA atom 的内部 layout 要求切分 tensor, 
        # 为 TMA copy 准备 source/destination view。
        #
        # 函数签名：
        #   tma_partition(atom, cta_id, cta_layout, smem_tensor, gmem_tensor)
        #     -> (smem_view, gmem_view)
        #
        # 重要要求：两个 tensor 的 Mode 0 都必须表示 TMA atom
        #
        # 示例：M=512, N=128, TileM=128, TileN=64
        #
        # 输入 tensor：
        #   gSrc_tiled:  (128, 64, 4, 2)      # 4 个独立 mode
        #                 └──┬──┘ └──┬──┘
        #                   Tile    Grid
        #
        #   smem_tensor: (128, 64)            # 2 个独立 mode
        #                 └──┬──┘
        #                   Tile
        #
        # 使用 group_modes(tensor, 0, 2) 合并前 2 个 mode：
        #
        #   group_modes(gSrc_tiled, 0, 2)   =>  ((128, 64), 4, 2)
        #                                         └───┬───┘
        #                                         Mode 0 = Atom
        #
        #   group_modes(smem_tensor, 0, 2)  =>  ((128, 64),)
        #                                         └───┬───┘
        #                                         Mode 0 = Atom
        #
        # tma_partition 之后：
        #
        #   tAsA: 带有 TMA 内部 layout 的 SMEM view
        #     Shape: ((TMA_Layout),)
        #     - TMA_Layout: 为高效 SMEM 访问准备的 swizzled/banked layout
        #
        #   tAgA: 保留其余 mode 的 global view
        #     Shape: ((TMA_Layout), 4, 2)
        #            └─────┬─────┘ └──┬──┘
        #            TMA atom      其余 modes (grid)
        #
        # 使用模式：
        #   1. 合并 modes 来定义 atom: group_modes(tensor, 0, 2)
        #   2. 调用 tma_partition: tAsA, tAgA = tma_partition(...)
        #   3. 为 CTA 选择 tile: tAgA_cta = tAgA[(None, bidx, bidy)]
        #      - None: 保留整个 atom
        #      - bidx, bidy: 在其余 modes 上取索引
        #   4. 发起 TMA copy: cute.copy(atom, tAgA_cta, tAsA)
        #
        # 可视化数据流：
        #
        #   Global Memory (512x128)           Shared Memory (128x64)
        #   ┌────────────────────┐            ┌──────────────┐
        #   │  ┌───┬───┐         │            │              │
        #   │  │0,0│0,1│         │            │  smem_tensor │
        #   │  ├───┼───┤         │            │  (128, 64)   │
        #   │  │1,0│1,1│ 4x2     │            │              │
        #   │  ├───┼───┤ tiles   │            └──────────────┘
        #   │  │2,0│2,1│         │                    │
        #   │  ├───┼───┤         │                    │ group_modes
        #   │  │3,0│3,1│         │                    ↓
        #   │  └───┴───┘         │              ((128, 64),)
        #   └────────────────────┘                    │
        #            │                                │
        #            │ gSrc_tiled                     │
        #            │ (128, 64, 4, 2)                │
        #            ↓                                │
        #     group_modes(_, 0, 2)                    │
        #            ↓                                │
        #     ((128, 64), 4, 2)                       │
        #            │                                │
        #            └────────┬───────────────────────┘
        #                     ↓
        #              tma_partition
        #                     ↓
        #      ┌──────────────┴──────────────┐
        #      │                             │
        #    tAgA                          tAsA
        #   ((TMA_Layout), 4, 2)        ((TMA_Layout),)
        #      │
        #      ↓ tAgA[(None, bidx, bidy)]
        #   tAgA_cta
        #   ((TMA_Layout),)
        #
        # ======================================================================

        # TMA Load 划分
        # 这里仅使用 1x1 cluster, 因此 cta_id 是 0, cta_layout 是 (1)。
        # cta_coord 和 cta_layout 的更多设置细节可以参考 tma_v4.py。
        # 注意：Smem 和 gemm 在第一 rank 上应有相同大小, 也就是 atom element size 相同。
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_src,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tensor, 0, 2),
            cute.group_modes(gSrc_tiled, 0, 2),
        )

        # TMA Store 划分
        # 流程与 TMA Load 相同, 只是这里处理 destination tensor。
        # 按照 TMA Store atom 划分 gDst_tiled 和 smem_tensor。
        _, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dst,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tensor, 0, 2),
            cute.group_modes(gDst_tiled, 0, 2),
        )

        # 从 partitioned global views 里选择当前 CTA 对应的 tile。
        # 输入：tAgA, shape 为 ((TMA_Layout), 4, 2)
        # 输出：tAgA_cta, shape 为 ((TMA_Layout),)
        # (None, bidx, bidy) 这个 indexing 的含义：
        #   - None: 保留完整 TMA atom layout, 也就是 mode 0
        #   - bidx: 从 rest mode 1 选择, 表示 M 维度 grid
        #   - bidy: 从 rest mode 2 选择, 表示 N 维度 grid
        tAgA_cta = tAgA[(None, bidx, bidy)]
        tBgB_cta = tBgB[(None, bidx, bidy)]

        # ---------- TMA Load: Global -> Shared ----------
        cute.copy(
            tma_atom_src,
            tAgA_cta,  # source, TMA Tensor View
            tAsA,  # destination, SMEM Tensor View
            tma_bar_ptr=barrier_ptr,
        )

        # 发起 TMA 后在 barrier 上 arrive
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(barrier_ptr)

        # 等待 TMA 完成
        cute.arch.mbarrier_wait(barrier_ptr, 0)

        # ---------- TMA Store: Shared -> Global ----------
        cute.copy(
            tma_atom_dst,
            tAsA,  # source, SMEM Tensor View
            tBgB_cta,  # destination, Global Tensor View
        )


def run_tma_copy(M, N, num_warmup=5, num_iters=20):
    """
    运行 TMA copy kernel, 并测量性能。

    Args:
        M: matrix dimension M
        N: matrix dimension N
        num_warmup: warmup iteration 数
        num_iters: timing iteration 数
    """
    # 创建 shape 为 (M, N) 的 tensor
    a = torch.randn((M, N), dtype=torch.float16, device="cuda")
    b = torch.zeros((M, N), dtype=torch.float16, device="cuda")

    # 这里声明 N-dimension 是 leading dimension, 即 contiguous 维度, 且其能被 16 整除
    a_cute = (
        from_dlpack(a, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )
    b_cute = (
        from_dlpack(b, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )

    copy_kernel = Sm100SimpleCopyKernel()
    compiled_kernel = cute.compile(copy_kernel, a_cute, b_cute)

    # 预热运行
    for _ in range(num_warmup):
        compiled_kernel(a_cute, b_cute)
    torch.cuda.synchronize()

    # 计时运行
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        compiled_kernel(a_cute, b_cute)
    end_event.record()
    torch.cuda.synchronize()

    # 计算性能指标
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_iters

    # 计算 throughput
    # 对 copy 来说：读取 M*N 个元素, 再写入 M*N 个元素
    bytes_per_element = a.element_size()
    total_bytes = 2 * M * N * bytes_per_element  # 读取 + 写入
    throughput_gb_s = (total_bytes / 1e9) / (avg_time_ms / 1000)

    # 打印性能指标
    print(f"Matrix size: {M}×{N}")
    print(f"Tile shape: {copy_kernel.tile_shape}")
    print(f"Average time: {avg_time_ms:.4f} ms")
    print(f"Throughput: {throughput_gb_s:.2f} GB/s")

    # 校验
    if torch.allclose(a, b, atol=1e-3):
        print("Verification: PASSED ✓")
    else:
        print("Verification: FAILED ✗")
        diff = (a - b).abs()
        print(f"Max diff: {diff.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TMA V0: Understanding tma_partition - The Foundation of TMA Operations"
    )
    parser.add_argument("--M", type=int, default=512, help="Matrix dimension M")
    parser.add_argument("--N", type=int, default=128, help="Matrix dimension N")
    parser.add_argument(
        "--num_warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num_iters", type=int, default=20, help="Number of timing iterations"
    )
    args = parser.parse_args()

    run_tma_copy(
        M=args.M,
        N=args.N,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )
