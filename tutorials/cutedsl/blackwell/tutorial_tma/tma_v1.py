# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Tuple, Type
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline
import torch

"""
使用 Producer-Consumer 模式的 TMA 矩阵转置：TMA load -> S2R --> R2S -> TMA store

Warp 角色（Producer-Consumer 模式）：
- Producer: TMA Load Warp (Warp 4), 从 Global 加载到 Shared A
- Consumer: Transpose Warps (Warp 0-3), 等待 load, 然后执行 sA -> sB 转置
- Consumer: TMA Store Warp (Warp 5), 等待 transpose, 然后把 sB 存回 Global

同步：
1. load_mbar_ptr: TMA Load (producer) -> Transpose Warps (consumer)
2. store_mbar_ptr: Transpose Warps (producer) -> TMA Store (consumer)
 
这个示例展示不同 warp 怎样使用 shared memory barrier 协调 producer-consumer 关系。
"""


class Sm100MatrixTransposeKernelV1:
    def __init__(self):
        self.tile_shape = (128, 128)
        self.tile_m, self.tile_n = self.tile_shape
        self.cluster_shape_mn = (1, 1)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)

        # 根据 tile_shape 设置 specialized warp ids
        self.num_trans_warps = 4  # 最大 transpose warp 数
        self.trans_warp_id = tuple(range(self.num_trans_warps)) # 0,1,2,3
        self.tma_load_warp_id = self.num_trans_warps  # warp 4
        self.tma_store_warp_id = self.num_trans_warps + 1  # warp 5
        self.threads_per_cta = 32 * len(
            (self.tma_store_warp_id, self.tma_load_warp_id, *self.trans_warp_id)
        )  # 32 * 6 = 192
        self.num_trans_threads = 32 * len(self.trans_warp_id)  # 32 * 4 = 128

        self.trans_tile = (self.tile_shape[0] // self.num_trans_warps, 8) # (32, 8)

        # 设置 producer-consumer 同步所需的 barriers
        # Barrier 1: Trans warps sync, 用于内部协调, 保证所有 transpose warps 都写完 sB
        self.trans_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.trans_warp_id),
        )
        # Barrier 2: TMA Store warp 等待 Trans warps 完成
        self.store_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32,  # Only TMA store warp
        )
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, src: cute.Tensor, dst: cute.Tensor):
        if cutlass.const_expr(src.element_type != dst.element_type):
            raise TypeError("Source and destination element types must match")

        self.dtype: Type[cutlass.Numeric] = src.element_type

        # 为 TMA descriptor 创建 dst 的转置 view, 这里不会额外创建新的 tensor。
        # dst 是 (N, M):(M, 1), 这里需要把它看作转置后的 (M, N):(1, M)。
        transed_dst = cute.make_tensor(
            dst.iterator,
            cute.make_layout(
                (dst.shape[1], dst.shape[0]), stride=(dst.stride[1], dst.stride[0])
            ),
        )

        # sA 使用 row-major smem layout, shape 为 (tile_m, tile_n)
        smem_layout_sA = sm100_utils.make_smem_layout(
            utils.LayoutEnum.from_tensor(src).mma_major_mode(),
            (self.tile_m, self.tile_n),
            self.dtype,
            1,
        )

        # sB 使用 col-major smem layout, shape 为 (tile_n, tile_m)
        # sB 应当匹配转置后的 destination layout。
        smem_layout_sB = sm100_utils.make_smem_layout(
            utils.LayoutEnum.from_tensor(transed_dst).mma_major_mode(),
            (self.tile_m, self.tile_n),
            self.dtype,
            1,
        )

        @cute.struct
        class SharedStorage:
            # TMA Load 使用的 barrier: producer (TMA) -> consumer (Trans warps)
            load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            # TMA Store 使用的 barrier: producer (Trans warps) -> consumer (TMA Store)
            store_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            # 两个 shared memory buffer: sA 保存 source tile, sB 保存转置后的 destination tile
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_sA)], 128
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_sB)], 128
            ]

        self.shared_storage = SharedStorage
        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_layout_sA)

        # TMA atoms
        # 对 TMA atom 使用 swizzled layout, 在 load 时处理 swizzling。
        tma_atom_src, tma_tensor_src = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            src,
            smem_layout_sA,
            (self.tile_m, self.tile_n),
        )

        tma_atom_dst, tma_tensor_dst = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            transed_dst,
            smem_layout_sB,
            (self.tile_m, self.tile_n),
        )

        grid_shape = cute.ceil_div((*src.layout.shape, 1), self.tile_shape)
        self.kernel(
            tma_atom_src,
            tma_tensor_src,
            tma_atom_dst,
            tma_tensor_dst,
            smem_layout_sA,
            smem_layout_sB,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_load: cute.CopyAtom,
        tma_tensor_src: cute.Tensor,
        tma_atom_store: cute.CopyAtom,
        tma_tensor_dst: cute.Tensor,
        smem_layout_sA: cute.ComposedLayout,
        smem_layout_sB: cute.ComposedLayout,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # 分配 Shared Memory，转置需要两个 buffer：
        # sA: source tile, swizzled
        # sB: TMA Store 使用的 destination tile, swizzled
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(smem_layout_sA.outer, swizzle=smem_layout_sA.inner)
        # sA = cute.make_tensor(storage.sA.iterator, smem_layout_sA)
        sB = storage.sB.get_tensor(smem_layout_sB.outer, swizzle=smem_layout_sB.inner)
        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_layout_sA)
        load_mbar_ptr = storage.load_mbar_ptr.data_ptr()
        store_mbar_ptr = storage.store_mbar_ptr.data_ptr()

        # ------------------------------------------------------------------
        # 初始化 Barriers, 所有 warp 都参与初始化
        # ------------------------------------------------------------------
        if tidx == 0:
            # TMA Load 使用的 barrier: 预期 1 次 arrive, 来自 TMA 完成后的 TMA warp
            cute.arch.mbarrier_init(load_mbar_ptr, 1)
            cute.arch.mbarrier_expect_tx(load_mbar_ptr, self.num_tma_load_bytes)

            # TMA Store 使用的 barrier: 等待 Trans warps arrive
            cute.arch.mbarrier_init(store_mbar_ptr, len(self.trans_warp_id))

        # barrier 初始化后同步所有 warp
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # ------------------------------------------------------------------
        # PRODUCER: TMA Load Warp (G -> sA)
        # ------------------------------------------------------------------
        if warp_idx == self.tma_load_warp_id:
            # 发起 TMA Load
            # ((TileM, TileK), loopM, LoopK)
            gA = cute.local_tile(tma_tensor_src, self.tile_shape, (None, None))
            # ((TileM, TileK), loopM, LoopK)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_load,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2),
            )

            cute.copy(
                tma_atom_load,
                tAgA[(None, bidx, bidy)],
                tAsA[(None, 0)],
                tma_bar_ptr=load_mbar_ptr,
            )

            # 在 mbarrier 上 arrive, 满足初始化时设置的 count 1
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(load_mbar_ptr)

        # ------------------------------------------------------------------
        # CONSUMER: Transpose Warps (sA -> Reg -> sB)
        # ------------------------------------------------------------------
        if warp_idx < self.tma_load_warp_id:
            trans_tid = tidx % self.num_trans_threads

            # 等待 TMA Load 完成, consumer 在 load_mbar 上等待
            cute.arch.mbarrier_wait(load_mbar_ptr, 0)

            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=self.dtype.width,  # 每次复制一个元素
            )

            copy_elems = 1

            # 这里不在 thread code 里手写 (m, n) -> (n, m) 的索引变换.
            # Host 侧已经把 dst 重新 view 成 transed_dst: (M, N):(1, M), 
            # sB 的 smem layout 也是按这个 destination view 来构造的,
            # 所以 transpose warps 可以用同一套 TV mapping 从 sA 读, 再写到 sB.
            # 后续 TMA store 按 transed_dst 的 stride 写回真实 dst, 效果是 dst[n, m] = src[m, n].
            #
            # TV layout 描述的是 "thread/value 到 tile 坐标" 的分配:
            #   T = transpose thread id, range is [0, num_trans_threads)
            #   V = 当前 thread 内的 value-lane.
            # 这里 copy_elems = 1, 所以每次 copy atom 只有一个 value-lane.
            # thread_layout = (num_trans_threads, 1):(1, num_trans_threads) 表示:
            #   T 连续变化时, 线性 id 也连续变化;
            #   V 只有 0, 不参与向量化.
            # 注意这只是单个 copy atom 的 TV mapping.
            # partition_S/partition_D 会把这个 mapping 平铺到整个 128x128 tile 上,
            # 因此每个 thread 最终拿到的是自己的 tile slice, 不是整个 tile 里的单个元素.
            thread_layout = cute.make_layout(
                (self.num_trans_threads, 1),
                stride=(1, self.num_trans_threads),
            )
            value_layout = cute.make_layout((1, copy_elems))
            # 构造一个 TiledCopy, 把上面的 TV mapping 绑定到 CopyUniversalOp.
            # 后面会通过 partition_S/partition_D 分别得到当前 thread 的 source view 和 destination view.
            tiled_copy = cute.make_tiled_copy_tv(atom, thread_layout, value_layout)
            thr_copy = tiled_copy.get_slice(trans_tid)

            # 划分 sA (source), 用于读取
            tCsA = thr_copy.partition_S(sA)
            # 什么时候使用 `tiled_copy.retile(...)`：
            # - 当你自己分配或构造 register tensor, 或者对它做 slice/reshape, 
            #   并且它的内部 layout 不匹配 `tiled_copy` 做 copy-in/out 时期望的 TV layout。
            # 这里为什么不用：
            # - `cute.make_fragment_like(tCsA)` 创建的 rmem fragment 与 `tCsA`
            #   有相同的 per-thread shape/layout, 因此已经匹配 `tiled_copy` 的 view, 可以直接复制进去。
            tCrA = cute.make_fragment_like(tCsA)
            cute.copy(tiled_copy, tCsA, tCrA)

            # 划分 sB, 用于写入
            tCsB = thr_copy.partition_D(sB)

            # 从 register 写入 sB
            cute.copy(tiled_copy, tCrA, tCsB)

            # fence 和 barrier 保证 shared memory store 对 TMA store 可见
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.trans_sync_barrier.arrive_and_wait()

            # Trans warps 通知 TMA Store warp: "sB is ready!"
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(store_mbar_ptr)

        # ------------------------------------------------------------------
        # CONSUMER: TMA Store Warp (sB -> G)
        # ------------------------------------------------------------------
        if warp_idx == self.tma_store_warp_id:
            # 等待 Trans warp 完成, consumer 在 store_mbar 上等待
            cute.arch.mbarrier_wait(store_mbar_ptr, 0)

            gDst_cta = cute.local_tile(
                tma_tensor_dst, (self.tile_m, self.tile_n), (None, None)
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_store,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gDst_cta, 0, 2),
            )
            cute.copy(tma_atom_store, tBsB[(None, 0)], tBgB[(None, bidx, bidy)])


def run_transpose(M, N, num_warmup=5, num_iters=20):
    """
    运行 TMA transpose kernel, 并测量性能。

    Args:
        M: matrix dimension M
        N: matrix dimension N
        num_warmup: warmup iteration 数
        num_iters: timing iteration 数

    Performance Metrics:
        - Throughput: 实际达到的带宽, 单位 GB/s
        - Theoretical BW: 峰值内存带宽 (2048 B/clk × 4000 MHz = 8.192 TB/s)
        - Bandwidth Efficiency: 实际带宽占理论峰值的百分比
    """
    torch.manual_seed(1111)
    # 输入 (M, N)
    input_data = torch.randn((M, N), device="cuda", dtype=torch.float16)
    # 输出 (N, M)
    output_data = torch.zeros((N, M), device="cuda", dtype=torch.float16)

    # CuTe wrappers
    tensor_src = (
        from_dlpack(input_data, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )
    tensor_dst = (
        from_dlpack(output_data, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )

    transpose_kernel = Sm100MatrixTransposeKernelV1()

    print("Start kernel compilation...")

    # 编译并运行
    compiled_kernel = cute.compile(
        transpose_kernel, tensor_src, tensor_dst, options="--generate-line-info"
    )

    print("Start kernel warmup...")
    # 预热运行
    for _ in range(num_warmup):
        compiled_kernel(tensor_src, tensor_dst)
    torch.cuda.synchronize()
    print("Kernel warmup completed.")

    # 计时运行
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        compiled_kernel(tensor_src, tensor_dst)
    end_event.record()
    torch.cuda.synchronize()

    # 计算性能指标
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_iters

    # 计算 throughput
    # 对 transpose 来说：读取 M*N 个元素, 再写入 M*N 个元素
    bytes_per_element = input_data.element_size()
    total_bytes = 2 * M * N * bytes_per_element  # 读取 + 写入
    throughput_gb_s = (total_bytes / 1e9) / (avg_time_ms / 1000)

    # 理论带宽上限
    # Blackwell: 4000 MHz 下为 2048 B/clk
    bytes_per_clk = 2048
    freq_mhz = 4000
    theoretical_bw_gb_s = bytes_per_clk * freq_mhz * 1e6 / 1e9  # 转成 GB/s
    theoretical_bw_tb_s = theoretical_bw_gb_s / 1000  # 转成 TB/s
    bandwidth_efficiency = (throughput_gb_s / theoretical_bw_gb_s) * 100  # 百分比

    # 打印性能指标
    print(f"Matrix size: {M}×{N}")
    print(f"Tile shape: {transpose_kernel.tile_shape}")
    print(f"Average time: {avg_time_ms:.4f} ms")
    print(f"Throughput: {throughput_gb_s:.2f} GB/s")
    print(
        f"Theoretical BW: {theoretical_bw_tb_s:.2f} TB/s ({theoretical_bw_gb_s:.2f} GB/s)"
    )
    print(f"Bandwidth Efficiency: {bandwidth_efficiency:.2f}%")

    # 校验
    expected = input_data.t()
    if torch.allclose(output_data, expected, atol=1e-2):
        print("Verification: PASSED ✓")
    else:
        print("Verification: FAILED ✗")
        print(f"Max diff: {(output_data - expected).abs().max()}")


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="TMA Matrix Transpose with Producer-Consumer Pattern"
    )
    parser.add_argument("--M", type=int, default=128, help="Matrix dimension M")
    parser.add_argument("--N", type=int, default=128, help="Matrix dimension N")
    parser.add_argument(
        "--num_warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num_iters", type=int, default=20, help="Number of timing iterations"
    )
    args = parser.parse_args()

    run_transpose(
        args.M,
        args.N,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )
