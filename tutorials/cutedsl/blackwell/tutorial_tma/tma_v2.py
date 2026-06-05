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
from typing import Tuple, Type, Union
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import torch

"""
使用 Multi-Stage Pipeline 的 TMA 矩阵转置 (v2)

这个版本在 tma_v1.py 的基础上增加：
1. Multi-stage pipeline: 使用多个 buffer 对 TMA loads 和 stores 做 pipelining
2. Pipeline abstraction: 使用 PipelineTmaAsync 做正确的 producer-consumer 协调
3. Persistent tile scheduler: 在 CTA 之间高效分发 work

相比 v1 的改进：
- Multi-stage buffers 可以重叠 TMA loads、computation 和 TMA stores
- Pipeline objects 提供更清晰的同步语义

注意：这里没有使用 TMA multicast, 因为每个 CTA 必须处理不同的 input tile。

Warp 角色：
- Producer: TMA Load Warp, 从 Global 加载到 Shared memory, multi-stage、multi-tile
- Consumer: Transpose Warps, 等待 load, 执行 sA -> sB 转置, multi-tile
- Consumer: TMA Store Warp, 等待 transpose, 把 sB 存回 Global, multi-stage、multi-tile

Pipeline stages：
1. Load Pipeline: TMA Load (producer) -> Transpose Warps (consumer)
2. Store Pipeline: Transpose Warps (producer) -> TMA Store (consumer)
"""


class Sm100MatrixTransposeKernelV2:
    def __init__(
        self,
    ):
        """
        初始化支持 multi-stage pipeline 的 TMA transpose kernel。

        Args:
            tile_shape: tile dimensions (M, N)
            cluster_shape_mn: 用于 CTA 并行执行的 cluster shape (M, N)

        Note:
            - 每个 CTA 独立处理不同 tile
            - stage 数基于可用 shared memory 自动计算
            - persistent scheduler 在 cluster 内的 CTA 之间分发 work
        """
        self.tile_shape = (128, 128)
        self.tile_m, self.tile_n = self.tile_shape
        self.cluster_shape_mn = (1, 1)
        self.cluster_shape_mnl = (*self.cluster_shape_mn, 1)

        # 根据 tile_shape 设置 specialized warp ids
        # 对 128x128 tile, 使用 4 个 transpose warps, 和 v1 相同
        self.max_trans_warps = 4  # 最大 transpose warp 数
        self.num_trans_warps = self.max_trans_warps  # 使用全部 transpose warps
        self.trans_warp_id = tuple(range(self.num_trans_warps))
        self.tma_load_warp_id = self.num_trans_warps
        self.threads_per_cta = 32 * len((self.tma_load_warp_id, *self.trans_warp_id))
        self.num_trans_threads = 32 * len(self.trans_warp_id)  # 128

        # 设置 producer-consumer 同步所需的 barriers
        # Barrier 1: Trans warps sync, 用于内部协调
        self.trans_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.trans_warp_id),
        )
        self.buffer_align_bytes = 128

        # 获取 shared memory capacity, 用于计算 stage 数
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")  # 232448

    @staticmethod
    def _compute_stages(
        tile_m: int,
        tile_n: int,
        dtype: Type[cutlass.Numeric],
        smem_capacity: int,
    ) -> Tuple[int, int]:
        """
        根据 shared memory capacity 计算 load/store stage 数。

        策略：
        1. 计算 load buffer (sA) 和 store buffer (sB) 每个 stage 需要的字节数
        2. 为 barriers 和 alignment 预留空间
        3. 用剩余 smem 除以每个 stage 需要的字节数, 得到最大 stage 数
        4. 把结果限制在合理的最小值和最大值之间

        Args:
            tile_m: tile dimension M
            tile_n: tile dimension N
            dtype: tensor 的 data type
            smem_capacity: shared memory 总容量, 单位 bytes

        Returns:
            (num_load_stages, num_store_stages)
        """
        # 计算每个 tile 的字节数, 这里假设使用 row-major 和 col-major layouts
        bytes_per_element = dtype.width // 8  # 2, for fp16

        # sA (load buffer): tile_m x tile_n 个元素
        sA_bytes_per_stage = tile_m * tile_n * bytes_per_element  # 128*128*2=32768

        # sB (store buffer): tile_m x tile_n 个元素, 表示转置后的 tile
        sB_bytes_per_stage = tile_m * tile_n * bytes_per_element  # 128*128*2=32768

        # 为 barriers 和其他 metadata 预留空间
        # 每个 barrier: 8 bytes (Int64)
        # 估算：最多 16 个 barriers, load + store stages * 2, 再加 alignment
        reserved_bytes = 1024  # 保守估计

        # 可用于 staging buffers 的空间
        available_smem = smem_capacity - reserved_bytes  # 232448 - 1024 = 231424 bytes

        # 计算能够放下的最大 stage 数
        # 需要同时给 load 和 store stages 留空间
        total_bytes_per_stage_pair = sA_bytes_per_stage + sB_bytes_per_stage

        # 最大 stage 数, 为简单起见 load 和 store 使用相同 stage 数
        max_stages = available_smem // total_bytes_per_stage_pair  # 231424//65536 = 3

        # 把 stage 数限制在合理范围
        # 最小值：2 stages, 用于基本 double buffering
        # 最大值：8 stages, 超过后收益通常下降
        num_stages = max(2, min(max_stages, 8))

        return num_stages, num_stages

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """使用 persistent tile scheduler 计算 output tensor C 的 grid size。

        :param c: output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mn: CTA tile 的 shape (M, N)。
        :type cta_tile_shape_mn: tuple[int, int]
        :param cluster_shape_mn: 每个 cluster 在 M、N 维度上的 shape。
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: 最大 active cluster 数。
        :type max_active_clusters: cutlass.Constexpr

        :return: 一个 tuple, 包含：
            - tile_sched_params: persistent tile scheduler 参数。
            - grid: kernel launch 使用的 grid shape。
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mn, (None, None))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mn = gc[(0, (None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        num_ctas_mnl = (*num_ctas_mn, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @cute.jit
    def __call__(
        self, src: cute.Tensor, dst: cute.Tensor, max_active_clusters: cutlass.Constexpr
    ):
        if cutlass.const_expr(src.element_type != dst.element_type):
            raise TypeError("Source and destination element types must match")

        self.dtype: Type[cutlass.Numeric] = src.element_type

        # 根据 tile size 和 dtype 计算较合适的 stage 数
        self.num_load_stages, self.num_store_stages = self._compute_stages(
            self.tile_m,
            self.tile_n,
            self.dtype,
            self.smem_capacity,
        )  # 3, 3

        # 为 TMA descriptor 创建 dst 的转置 view。
        # 和 v1 一样, dst 是 (N, M), 这里需要把它看作转置后的 (M, N)。
        transed_dst = cute.make_tensor(
            dst.iterator,
            cute.make_layout(
                (dst.shape[1], dst.shape[0]), stride=(dst.stride[1], dst.stride[0])
            ),
        )

        # 为 load 和 store buffers 创建 multi-stage layouts
        # sA 使用 row-major smem layout, shape 为 (tile_m, tile_n)
        smem_layout_sA_staged = sm100_utils.make_smem_layout(
            utils.LayoutEnum.from_tensor(src).mma_major_mode(),
            (self.tile_m, self.tile_n),
            self.dtype,
            self.num_load_stages,
        )

        # sB 使用 col-major smem layout, shape 为 (tile_n, tile_m)
        # sB 应当匹配转置后的 destination layout。
        smem_layout_sB_staged = sm100_utils.make_smem_layout(
            utils.LayoutEnum.from_tensor(transed_dst).mma_major_mode(),
            (self.tile_m, self.tile_n),
            self.dtype,
            self.num_store_stages,
        )

        @cute.struct
        class SharedStorage:
            # multi-stage load 使用的 pipeline barriers
            load_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load_stages
            ]
            load_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load_stages
            ]

            # multi-stage store 使用的 pipeline barriers
            store_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_store_stages
            ]
            store_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_store_stages
            ]

            # multi-stage shared memory buffers
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_sA_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_sB_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        a_smem_layout = cute.slice_(smem_layout_sA_staged, (None, None, 0))
        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, a_smem_layout)

        # TMA atoms
        # 每个 CTA 独立加载自己的 tile
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (1,)
        )

        tma_atom_src, tma_tensor_src = cpasync.make_tiled_tma_atom(
            tma_load_op,
            src,
            smem_layout_sA_staged,
            (self.tile_m, self.tile_n),
        )

        tma_atom_dst, tma_tensor_dst = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            transed_dst,
            smem_layout_sB_staged,
            (self.tile_m, self.tile_n),
        )
        tile_sched_params, grid_shape = self._compute_grid(
            transed_dst,
            (self.tile_m, self.tile_n),
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.kernel(
            tma_atom_src,
            tma_tensor_src,
            tma_atom_dst,
            tma_tensor_dst,
            smem_layout_sA_staged,
            smem_layout_sB_staged,
            cluster_layout_vmnk,
            tile_sched_params,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnl,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_load: cute.CopyAtom,
        tma_tensor_src: cute.Tensor,
        tma_atom_store: cute.CopyAtom,
        tma_tensor_dst: cute.Tensor,
        smem_layout_sA_staged: Union[cute.Layout, cute.ComposedLayout],
        smem_layout_sB_staged: Union[cute.Layout, cute.ComposedLayout],
        cluster_layout_vmnk: cute.Layout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # ---------------- Shared mem 与 staged buffers ----------------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA_staged = storage.sA.get_tensor(
            smem_layout_sA_staged.outer, swizzle=smem_layout_sA_staged.inner
        )
        sB_staged = storage.sB.get_tensor(
            smem_layout_sB_staged.outer, swizzle=smem_layout_sB_staged.inner
        )

        load_mbar_ptr = storage.load_full_mbar_ptr.data_ptr()
        _store_mbar_ptr = storage.store_full_mbar_ptr.data_ptr()

        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        load_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_trans_warps
        )

        load_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar_ptr,
            num_stages=self.num_load_stages,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_trans_threads
        )
        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.num_store_stages,
            producer_group=store_producer_group,
        )

        # 重要：跨 cluster 初始化 pipeline barriers。
        # 这必须发生在 pipeline 创建之后、任何 producer/consumer 工作之前。
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        gA = cute.local_tile(tma_tensor_src, self.tile_shape, (None, None))

        _cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, None, 0)).shape
        )
        # ((TileM, TileK), loopM, LoopK)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_load,
            0,
            cute.make_layout(1),
            cute.group_modes(sA_staged, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        gDst_cta = cute.local_tile(tma_tensor_dst, self.tile_shape, (None, None))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_store,
            0,
            cute.make_layout(1),
            cute.group_modes(sB_staged, 0, 2),
            cute.group_modes(gDst_cta, 0, 2),
        )

        # ------------------------------------------------------------------
        # PRODUCER: TMA Load Warp (G -> sA)
        # ------------------------------------------------------------------
        if warp_idx == self.tma_load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            load_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load_stages
            )

            while work_tile.is_valid_tile:
                # 从 tile scheduler 获取 tile coord
                cur_tile_coord = work_tile.tile_idx
                tAgA_slice = tAgA[(None, cur_tile_coord[0], cur_tile_coord[1])]

                load_pipeline.producer_acquire(load_producer_state)

                cute.copy(
                    tma_atom_load,
                    tAgA_slice,
                    tAsA[(None, load_producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(load_producer_state),
                )

                load_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            load_pipeline.producer_tail(load_producer_state)

        # ------------------------------------------------------------------
        # CONSUMER: Transpose Warps (sA -> Reg -> sB)
        # ------------------------------------------------------------------
        if warp_idx < self.tma_load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            trans_tid = tidx % self.num_trans_threads

            load_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load_stages
            )

            while work_tile.is_valid_tile:
                # 从 tile scheduler 获取 tile coord
                cur_tile_coord = work_tile.tile_idx

                # 等待 load pipeline 准备好数据
                load_pipeline.consumer_wait(load_consumer_state)

                atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.dtype,
                    num_bits_per_copy=self.dtype.width,
                )
                copy_elems = 1

                thread_layout = cute.make_layout(
                    (self.num_trans_threads, 1),
                    stride=(1, self.num_trans_threads),
                )
                value_layout = cute.make_layout((1, copy_elems))
                tiled_copy = cute.make_tiled_copy_tv(atom, thread_layout, value_layout)
                thr_copy = tiled_copy.get_slice(trans_tid)

                # sA -> Reg
                tCsA = thr_copy.partition_S(sA_staged)

                tCrA = cute.make_rmem_tensor(
                    tCsA[(None, None, None, 0)].shape, self.dtype
                )
                tCrA = tiled_copy.retile(tCrA)

                cute.copy(
                    tiled_copy,
                    tCsA[(None, None, None, load_consumer_state.index)],
                    tCrA,
                )

                # 释放当前 load stage, 允许 TMA load producer 复用对应的 sA buffer
                load_pipeline.consumer_release(load_consumer_state)
                load_consumer_state.advance()

                index = tile_sched.num_tiles_executed % self.num_store_stages
                # Reg -> sB
                tCsB = thr_copy.partition_D(sB_staged)
                cute.copy(tiled_copy, tCrA, tCsB[(None, None, None, index)])

                # fence 保证 smem 写入可见
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                self.trans_sync_barrier.arrive_and_wait()

                if warp_idx == self.trans_warp_id[0]:
                    cute.copy(
                        tma_atom_store,
                        tBsB[(None, index)],
                        tBgB[(None, cur_tile_coord[0], cur_tile_coord[1])],
                    )
                    store_pipeline.producer_commit()
                    store_pipeline.producer_acquire()
                self.trans_sync_barrier.arrive_and_wait()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            self.trans_sync_barrier.arrive_and_wait()

            store_pipeline.producer_tail()


def run_transpose(M, N, max_active_clusters=0, num_warmup=5, num_iters=20):
    """
    运行 TMA transpose kernel, 自动计算 stage 数并测量性能。

    Args:
        M: matrix dimension M
        N: matrix dimension N
        max_active_clusters: 最大 active cluster 数, 0 表示自动
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

    transpose_kernel = Sm100MatrixTransposeKernelV2()

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(1)  # 148
    # 编译并运行
    compiled_kernel = cute.compile(
        transpose_kernel,
        tensor_src,
        tensor_dst,
        max_active_clusters,
        options="--generate-line-info",
    )

    # 预热运行
    for _ in range(num_warmup):
        compiled_kernel(tensor_src, tensor_dst)
    torch.cuda.synchronize()

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

    # 打印编译后计算出的 stage 数
    print(f"Matrix size: {M}×{N}")
    print(f"Tile shape: {transpose_kernel.tile_shape}")
    print(
        f"Computed stages: Load={transpose_kernel.num_load_stages}, Store={transpose_kernel.num_store_stages}"
    )
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
        description="TMA Matrix Transpose with Multi-Stage Pipeline and Cluster Support (v2)"
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
