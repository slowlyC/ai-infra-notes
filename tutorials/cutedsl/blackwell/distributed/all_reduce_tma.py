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

"""
使用 TMA (Tensor Memory Accelerator) 的分布式 All-Reduce 示例.

这个示例演示如何使用 TMA 搬运数据, 在多 GPU 上执行分布式 all-reduce.
它用于讲解基于 TMA 的分布式操作, 并非面向性能优化的实现.

Tensor 语义:
    - Input: logical shape 为 (world_size, S), S 是每个 rank 的 tensor 大小
    - Output: logical shape 为 (world_size, S), 每个 rank 都得到所有 inputs 之和

Kernel 参数:
    - input: 包含 world_size 个 tensors 的 list, 每个 tensor 的 shape 为 S, 可通过 NVSHMEM 访问
    - output: shape 为 S 的单个 tensor, 使用 multicast address 执行 broadcast

算法 (Two-Shot):
    1. 每个 CTA 从所有 ranks 加载其负责 tile 位置的数据 (TMA Load)
    2. 在 registers 中进行本地累加
    3. 通过 TMA multicast 写出结果, broadcast 到所有 ranks
    4. cross-GPU barrier 确保 kernel 退出前所有操作均已完成

Tile 分配:
    - total_tiles = ceil(S / elems_per_cta)
    - 每个 rank 处理 ceil(total_tiles / world_size) 个 CTAs
    - rank r 上的 CTA i 处理 global_tile_id = r * ctas_per_rank + i

TMA 使用说明 (仅用于教学, 并非性能最优):
    - 使用 1D TMA load, 通过 NVSHMEM addresses 从 remote GPU memory 加载数据
    - 使用 1D TMA store 写入 multicast address, 将结果 broadcast 到所有 ranks
    - 将任意 shape 的 input 展平为 1D 并线性分 tile, 从而支持任意 input shape
    - 使用 2-stage pipeline 重叠来自不同 ranks 的 TMA loads

运行示例:

.. code-block:: bash

    torchrun --nproc-per-node 8 examples/distributed/all_reduce_tma.py --shape 1024,1024
    torchrun --nproc-per-node 8 examples/distributed/all_reduce_tma.py --shape 4,6,8,10,12
"""

import cutlass
import cutlass.utils as utils
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync


class AllReduceTmaKernel:
    """
    基于 TMA 的分布式 All-Reduce kernel.

    这个 kernel 使用 TMA (Tensor Memory Accelerator) 高效搬运数据,
    在多个 GPUs 之间执行 all-reduce.

    算法 (Two-Shot):
        1. 每个 CTA 从所有 ranks 加载其负责 tile 位置的数据
        2. 在 registers 中进行本地累加
        3. 通过 TMA multicast 写出结果, broadcast 到所有 ranks
        4. cross-GPU barrier 确保 kernel 退出前所有操作均已完成

    input/output tensors 可以有任意 rank, 但需要满足:
        - 所有 input tensors 和 output tensor 使用相同 layout
        - layout 是 compact 的, memory 中没有空洞

    我们按照 codomain (physical offset) 顺序线性遍历 tensors,
    从而保证所有 tensors 的 logical coordinate 访问保持一致.

    每个 tile 只由负责它的 rank 归约一次, TMA multicast store 后所有 ranks 都得到完整 output.
    
    数据流示例, world_size=2, total_tiles=4, ctas_per_rank=2:
    
    tile 负责关系:
      rank 0 CTAs -> tile 0, tile 1
      rank 1 CTAs -> tile 2, tile 3
    
    rank 0 input: [A0 A1 | A2 A3]
    rank 1 input: [B0 B1 | B2 B3]
    
    rank 0 CTA 0, tile 0:
      A0 --TMA Load--> stage 0 (ping) --\
      B0 --TMA Load--> stage 1 (pong) ---+--> consumer ADD --> stage 0 = S0
    
    rank 0 CTA 1, tile 1:
      A1 --TMA Load--> stage 0 (ping) --\
      B1 --TMA Load--> stage 1 (pong) ---+--> consumer ADD --> stage 0 = S1
    
    rank 1 CTA 0, tile 2:
      A2 --TMA Load--> stage 0 (ping) --\
      B2 --TMA Load--> stage 1 (pong) ---+--> consumer ADD --> stage 0 = S2
    
    rank 1 CTA 1, tile 3:
      A3 --TMA Load--> stage 0 (ping) --\
      B3 --TMA Load--> stage 1 (pong) ---+--> consumer ADD --> stage 0 = S3
    
    各 CTA 将自己的 stage 0 结果写入所有 ranks 的对应 output tile:
      rank 0 CTA 0: S0 --TMA multicast store--> all ranks output tile 0
      rank 0 CTA 1: S1 --TMA multicast store--> all ranks output tile 1
      rank 1 CTA 0: S2 --TMA multicast store--> all ranks output tile 2
      rank 1 CTA 1: S3 --TMA multicast store--> all ranks output tile 3
    
    rank 0 output: [S0 S1 | S2 S3]
    rank 1 output: [S0 S1 | S2 S3]
    
    world_size > 2 时, 各 rank 的输入 tile 按 stage 0, 1, 0, 1, ... 循环复用.
    
    """

    _elems_per_cta: int = 128 * 128  # 每个 CTA 处理的 elements 数量
    _tma_threads: int = 32
    _consumer_threads: int = 128
    _threads_per_cta: int = _tma_threads + _consumer_threads
    _num_stages: int = 2

    def __init__(self, dtype):
        self.dtype = dtype

        # SMEM layout shape, 会在 JIT context 中转换为 Layout
        self.smem_layout_shape = (self._elems_per_cta,)
        self.tiler = (self._elems_per_cta,)

        # 根据 dtype 大小计算 TMA transaction bytes
        # dtype.width 的单位是 bits, 除以 8 得到 bytes
        self.tma_bytes = (dtype.width // 8) * self._elems_per_cta

        # 根据 dtype 动态创建 SharedStorage type
        elems = self._elems_per_cta
        stages = self._num_stages

        @cute.struct
        class SharedStorage:
            mbar_array: cute.struct.MemRange[cutlass.Int64, stages * 2]
            smem_buffer: cute.struct.Align[
                cute.struct.MemRange[dtype, elems * stages],  # stages 个 tile
                128,
            ]

        self._SharedStorage = SharedStorage

    @cute.jit
    def __call__(
        self,
        input_tensors: list[cute.Tensor],
        output_tensor_mc: cute.Tensor,
        flag: cute.Tensor,
        flag_mc: cute.Tensor,
        local_rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
    ):
        """
        Host 侧 JIT function: 创建 TMA descriptors 并启动 kernel.

        参数:
            input_tensors: 来自各 rank 的 input tensors list, 共 world_size 个 tensors
            output_tensor_mc: 使用 multicast address 的 output tensor
            flag: synchronization flag 的 local view
            flag_mc: synchronization flag 的 multicast view
            local_rank: 当前 rank 的 ID
            world_size: ranks 总数
        """
        # ======================================================================
        # Layout 校验
        # ======================================================================
        ref_layout = input_tensors[0].layout
        ref_size = cute.size(ref_layout)
        ref_cosize = cute.cosize(ref_layout)

        # 检查 compact: size == cosize, memory 中没有空洞
        assert ref_size == ref_cosize, (
            f"Input tensor must be compact: size={ref_size}, cosize={ref_cosize}"
        )
        assert self.tma_bytes % 16 == 0, f"Not aligned to 16B, TMA should not be used."

        # 检查所有 input tensors 是否使用相同 layout
        for i in cutlass.range_constexpr(world_size):
            assert input_tensors[i].layout == ref_layout, (
                f"All input tensors must have the same layout. "
                f"input_tensors[0].layout={ref_layout}, "
                f"input_tensors[{i}].layout={input_tensors[i].layout}"
            )

        # 检查 output tensor 是否使用相同 layout
        assert output_tensor_mc.layout == ref_layout, (
            f"Output tensor must have the same layout as input tensors. "
            f"input layout={ref_layout}, output layout={output_tensor_mc.layout}"
        )

        # ======================================================================
        # 提取 tensor 信息
        # ======================================================================
        # 检查 dtype 是否匹配
        assert input_tensors[0].element_type == self.dtype, (
            f"Input tensor dtype mismatch: expected {self.dtype}, "
            f"got {input_tensors[0].element_type}"
        )

        total_elems = ref_size

        # 展平 layout: 按 codomain 顺序把 tensor 视为 1D
        flat_layout = cute.make_layout((total_elems,))

        # 在 JIT context 中创建 SMEM layout
        smem_layout = cute.make_layout(self.smem_layout_shape)

        # 创建 TMA load descriptors, 每个 rank 一个
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_load_atoms = []
        tma_load_tensors = []

        for i in cutlass.range_constexpr(world_size):
            flat_input = cute.make_tensor(input_tensors[i].iterator, flat_layout)
            tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
                tma_load_op,
                flat_input,
                smem_layout,
                self.tiler,
            )
            tma_load_atoms.append(tma_atom)
            tma_load_tensors.append(tma_tensor)

        # 创建 TMA store descriptor
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        flat_output = cute.make_tensor(output_tensor_mc.iterator, flat_layout)
        tma_store_atom, tma_store_tensor = cpasync.make_tiled_tma_atom(
            tma_store_op,
            flat_output,
            smem_layout,
            self.tiler,
        )

        # 计算 grid
        num_tiles_total = cute.ceil_div(total_elems, self._elems_per_cta)
        ctas_per_rank = cute.ceil_div(num_tiles_total, world_size)

        # 从 SharedStorage 获取 SMEM 大小
        smem_bytes = self._SharedStorage.size_in_bytes()

        # 启动 kernel
        self.kernel(
            tma_load_atoms,
            tma_load_tensors,
            tma_store_atom,
            tma_store_tensor,
            flag,
            flag_mc,
            local_rank,
            world_size,
            num_tiles_total,
            ctas_per_rank,
        ).launch(
            grid=[ctas_per_rank, 1, 1],
            block=[self._threads_per_cta, 1, 1],
            smem=smem_bytes,
        )

    @cute.kernel
    def kernel(
        self,
        # 用于从各 rank 加载数据的 TMA atoms 和 tensors
        tma_load_atoms: list[cute.CopyAtom],
        tma_load_tensors: list[cute.Tensor],
        # 用于写入 multicast address 的 TMA atom 和 tensor
        tma_store_atom: cute.CopyAtom,
        tma_store_tensor: cute.Tensor,
        # Synchronization flags
        flag: cute.Tensor,
        flag_mc: cute.Tensor,
        # Rank 信息
        local_rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        # 用于计算 tile 的 grid 信息
        num_tiles_total: cutlass.Constexpr,
        ctas_per_rank: cutlass.Constexpr,
    ):
        # ======================================================================
        # Thread/Block 索引
        # ======================================================================
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # ======================================================================
        # 分配 SMEM
        # ======================================================================
        staged_smem_layout = cute.make_layout((self._elems_per_cta, self._num_stages))

        smem = utils.SmemAllocator()
        storage = smem.allocate(self._SharedStorage)
        mbar_ptr = storage.mbar_array.data_ptr()
        staged_smem_tensor = storage.smem_buffer.get_tensor(staged_smem_layout)

        # ======================================================================
        # 配置 TMA Pipeline
        # ======================================================================
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self._consumer_threads
        )

        tma_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=self._num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        global_tile_id = local_rank * ctas_per_rank + bidx

        if global_tile_id < num_tiles_total:
            # ======================================================================
            # Warp 0: Producer, 从所有 ranks 执行 TMA Load
            # ======================================================================
            if warp_idx == 0:
                producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self._num_stages
                )

                for rank_i in cutlass.range_constexpr(world_size):
                    tma_pipeline.producer_acquire(producer_state)

                    stage_idx = producer_state.index
                    smem_tile = cute.slice_(staged_smem_tensor, (None, stage_idx))

                    g_tensor_tiled = cute.zipped_divide(
                        tma_load_tensors[rank_i], self.tiler
                    )
                    g_tile = g_tensor_tiled[(None,), global_tile_id]

                    g_tile_flat = cute.group_modes(g_tile, 0, cute.rank(g_tile))
                    s_tile_flat = cute.group_modes(smem_tile, 0, cute.rank(smem_tile))

                    s_part, g_part = cute.nvgpu.cpasync.tma_partition(
                        tma_load_atoms[rank_i],
                        0,
                        cute.make_layout(1),
                        s_tile_flat,
                        g_tile_flat,
                    )

                    cute.copy(
                        tma_load_atoms[rank_i],
                        g_part,
                        s_part,
                        tma_bar_ptr=tma_pipeline.producer_get_barrier(producer_state),
                    )

                    tma_pipeline.producer_commit(producer_state)
                    producer_state.advance()

            # ======================================================================
            # Warp 1-4: Consumer, 从 SMEM 加载、ADD, 再写回 SMEM
            # ======================================================================
            else:
                consumer_tid = tidx - self._tma_threads

                vec_size = 4
                chunk_size = vec_size * self._consumer_threads

                # ------------------------------------------------------------------
                # 使用 stage 0 的 layout 初始化 accumulator
                # ------------------------------------------------------------------
                # (elems, stages) -> (elems,)
                smem_tensor_wo_stage = cute.slice_(staged_smem_tensor, (None, 0))
                # (elems,) -> ((thr_vec,), (num_chunks,))
                smem_tensor_tiled_by_thr_vec = cute.zipped_divide(
                    smem_tensor_wo_stage, (chunk_size,)
                )
                # ((thr_vec,), (num_chunks,)) -> (((vec, threads),), (num_chunks,))
                smem_tensor_tiled_by_thr_vec_tiled_by_vec = cute.logical_divide(
                    smem_tensor_tiled_by_thr_vec, (vec_size,)
                )
                # (((vec, threads),), (num_chunks,)) -> ((vec,), (num_chunks,))
                per_thread_smem_tensor = cute.slice_(
                    smem_tensor_tiled_by_thr_vec_tiled_by_vec,
                    ((None, consumer_tid), None),
                )

                accum = cute.make_rmem_tensor(per_thread_smem_tensor.layout, self.dtype)
                accum.fill(self.dtype(0.0))

                # ------------------------------------------------------------------
                # 主循环: 从 SMEM 加载并累加
                # ------------------------------------------------------------------
                consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self._num_stages
                )

                for rank_i in cutlass.range_constexpr(world_size):
                    tma_pipeline.consumer_wait(consumer_state)

                    stage_idx = consumer_state.index
                    smem_tile = cute.slice_(staged_smem_tensor, (None, stage_idx))

                    # (elems,) -> ((thr_vec,), (num_chunks,))
                    smem_tiled_by_thr_vec = cute.zipped_divide(smem_tile, (chunk_size,))
                    # ((thr_vec,), (num_chunks,)) -> (((vec, threads),), (num_chunks,))
                    smem_tiled_by_thr_vec_tiled_by_vec = cute.logical_divide(
                        smem_tiled_by_thr_vec, (vec_size,)
                    )
                    # (((vec, threads),), (num_chunks,)) -> ((vec,), (num_chunks,))
                    per_thread_smem_view = cute.slice_(
                        smem_tiled_by_thr_vec_tiled_by_vec,
                        ((None, consumer_tid), None),
                    )

                    fragment = per_thread_smem_view.load()
                    accum.store(accum.load() + fragment)

                    tma_pipeline.sync_object_empty.arrive(
                        consumer_state.index, tma_pipeline.consumer_mask
                    )
                    consumer_state.advance()

                # 将累加结果写回 SMEM 的 stage 0
                per_thread_smem_tensor.store(accum.load())

            # ======================================================================
            # 同步点: 所有 warps 在此会合
            # ======================================================================
            cute.arch.sync_threads()

            # ======================================================================
            # Warp 0: 通过 TMA Store 写入 multicast output
            # ======================================================================
            if warp_idx == 0:
                # 使用 fence 确保 SMEM writes 可见
                cute.arch.fence_proxy("async.shared", space="cta")

                smem_tile_out = cute.slice_(staged_smem_tensor, (None, 0))

                g_output_tiled = cute.zipped_divide(tma_store_tensor, self.tiler)
                g_output_tile = g_output_tiled[(None,), global_tile_id]

                g_out_flat = cute.group_modes(
                    g_output_tile, 0, cute.rank(g_output_tile)
                )
                s_out_flat = cute.group_modes(
                    smem_tile_out, 0, cute.rank(smem_tile_out)
                )

                s_part, g_part = cute.nvgpu.cpasync.tma_partition(
                    tma_store_atom,
                    0,
                    cute.make_layout(1),
                    s_out_flat,
                    g_out_flat,
                )

                cute.copy(tma_store_atom, s_part, g_part)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)

        # ==================================================================
        # Cross-GPU barrier 同步, 仅由 thread 0 执行
        # ==================================================================
        if tidx == 0:
            sm_id_linear = (
                cute.arch.block_idx()[0]
                + cute.arch.block_idx()[1] * cute.arch.grid_dim()[0]
                + cute.arch.block_idx()[2]
                * cute.arch.grid_dim()[0]
                * cute.arch.grid_dim()[1]
            )

            # 向所有 ranks 通知当前 CTA 已完成
            utils.distributed.multimem_red_add1(
                flag_mc.iterator + sm_id_linear,
                scope="sys",
                order="release",
            )

            # 相同 idx 的 CTAs 等待所有 peer ranks 上对应 CTA 完成
            utils.distributed.spin_lock_atom_cas_relaxed_wait(
                flag.iterator + sm_id_linear,
                expected_val=world_size,
                reset_val=0,
                scope="sys",
            )


# =============================================================================
# HOST 侧 DRIVER CODE
# =============================================================================

import os
import argparse
import math

import numpy as np
import torch
import torch.distributed as dist
try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device
from cuda.pathfinder import load_nvidia_dynamic_lib

from cutlass.cute.runtime import from_dlpack

try:
    import nvshmem.core
except ImportError as exc:
    raise ImportError(
        "nvshmem4py is required but not installed. Please install it using:\n"
        "  For CUDA 12: pip install nvshmem4py-cu12\n"
        "  For CUDA 13: pip install nvshmem4py-cu13\n"
        "Note: nvshmem4py version >= 0.1.3 is recommended."
    ) from None

try:
    load_nvidia_dynamic_lib("nvshmem_host")
except RuntimeError as exc:
    raise ImportError(
        "nvshmem lib is required but not installed. Please install it using:\n"
        "  For CUDA 12: pip install nvidia-nvshmem-cu12\n"
        "  For CUDA 13: pip install nvidia-nvshmem-cu13\n"
    ) from None


def torchrun_uid_init_bcast():
    """使用 UniqueID 初始化 NVSHMEM, 并以 torchrun 作为 launcher."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dev = Device(local_rank)
    dev.set_current()
    global stream
    stream = dev.create_stream()

    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    num_ranks = dist.get_world_size()

    uid = nvshmem.core.get_unique_id(empty=(local_rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)

    nvshmem.core.init(
        device=dev, uid=uid, rank=local_rank, nranks=num_ranks, initializer_method="uid"
    )


def torchrun_finalize():
    """结束 NVSHMEM 并销毁 process group."""
    nvshmem.core.finalize()
    dist.destroy_process_group()


def run_all_reduce_tma(
    shape: tuple,
    skip_ref_check: bool = False,
):
    """
    运行基于 TMA 的 All-Reduce kernel.

    参数:
        shape: Tensor shape tuple, 例如 (4, 6, 8, 10)
        skip_ref_check: 为 True 时跳过 reference result 校验
    """
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # 计算 elements 总数
    total_elems = math.prod(shape)

    if local_rank == 0:
        print("\nRunning TMA All-Reduce test with:")
        print(f"  Tensor shape: {shape}")
        print(f"  Total elements: {total_elems}")
        print(f"  GPU count: {world_size}")

    # 分配所有 ranks 均可访问的 symmetric input tensor
    local_input_tensor = nvshmem.core.tensor(shape, dtype=torch.float32)
    local_input_tensor.random_(0, 100)

    # 获取 peer tensors, 即各 rank input 的 views
    peer_input_tensors = [
        nvshmem.core.get_peer_tensor(local_input_tensor, r) for r in range(world_size)
    ]

    if local_rank == 0:
        print(f"  Input tensor ptr: {local_input_tensor.data_ptr():#x}")

    # 分配使用 multicast address 的 output tensor
    local_output_tensor = nvshmem.core.tensor(shape, dtype=torch.float32)
    local_output_tensor.fill_(0)
    output_tensor_mc = nvshmem.core.get_multicast_tensor(
        nvshmem.core.Teams.TEAM_NODE, local_output_tensor
    )

    # 分配 synchronization flags
    # Flag 大小为 ctas_per_rank, 与 kernel 的 bidx 索引对应
    elems_per_cta = AllReduceTmaKernel._elems_per_cta
    num_tiles = (total_elems + elems_per_cta - 1) // elems_per_cta
    ctas_per_rank = (num_tiles + world_size - 1) // world_size
    local_flag = nvshmem.core.tensor((ctas_per_rank,), dtype=torch.int32)
    local_flag.fill_(0)
    flag_mc = nvshmem.core.get_multicast_tensor(
        nvshmem.core.Teams.TEAM_NODE, local_flag
    )

    if local_rank == 0:
        print(f"  Number of tiles: {num_tiles}")
        print(f"  CTAs per rank: {ctas_per_rank}")
        print("Compiling kernel...")

    # 创建 kernel instance 并编译
    kernel = AllReduceTmaKernel(cutlass.Float32)

    compiled_func = cute.compile(
        kernel,
        [from_dlpack(t) for t in peer_input_tensors],
        from_dlpack(output_tensor_mc),
        from_dlpack(local_flag),
        from_dlpack(flag_mc),
        local_rank,
        world_size,
    )

    if local_rank == 0:
        print("Compilation successful!")

    if not skip_ref_check:
        if local_rank == 0:
            print("Executing kernel...")

        dist.barrier(device_ids=[local_rank])
        compiled_func(
            [from_dlpack(t) for t in peer_input_tensors],
            from_dlpack(output_tensor_mc),
            from_dlpack(local_flag),
            from_dlpack(flag_mc),
        )
        dist.barrier(device_ids=[local_rank])

        if local_rank == 0:
            print("Verifying results...")

        # 计算期望结果, 即所有 inputs 之和
        expected = sum([t.cpu() for t in peer_input_tensors])

        # 与实际 output 比较
        torch.testing.assert_close(expected, local_output_tensor.cpu())

        if local_rank == 0:
            print("Results verified successfully!")

    # 释放资源
    for i in range(world_size):
        if i != local_rank:
            nvshmem.core.free_tensor(peer_input_tensors[i])

    nvshmem.core.free_tensor(output_tensor_mc)
    nvshmem.core.free_tensor(flag_mc)
    nvshmem.core.free_tensor(local_input_tensor)
    nvshmem.core.free_tensor(local_output_tensor)
    nvshmem.core.free_tensor(local_flag)


def parse_shape(shape_str: str) -> tuple:
    """
    将 shape string 解析为 tuple.
    示例:
        "1024,1024" -> (1024, 1024)
        "2,3,4,5,6,7,8" -> (2, 3, 4, 5, 6, 7, 8)
    """
    return tuple(int(x.strip()) for x in shape_str.split(","))


def main():
    parser = argparse.ArgumentParser(
        description="基于 TMA 的分布式 all-reduce 示例"
    )
    parser.add_argument(
        "--shape",
        default="1024,1024",
        type=str,
        help="以逗号分隔的 Tensor shape, 例如 '1024,1024' 或 '4,6,8,10,12'",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="跳过 reference result 校验",
    )

    args = parser.parse_args()
    shape = parse_shape(args.shape)

    torchrun_uid_init_bcast()
    run_all_reduce_tma(
        shape=shape,
        skip_ref_check=args.skip_ref_check,
    )
    torchrun_finalize()


if __name__ == "__main__":
    main()
