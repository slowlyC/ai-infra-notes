# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# 第五个 GEMM 教程, 在 fp16_gemm_3_1.py 基础上增加 TMA Prefetch 优化.

import argparse
from typing import Tuple, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

"""

TMA Prefetch 通过 cute.prefetch() 在 TMA load 之前将数据从 DRAM 带入 L2 cache,
底层对应 PTX 指令 cp.async.bulk.prefetch.tensor.L2.global.
与 TMA load 不同, prefetch 不写入 SMEM, 不占用 buffer, 不需要 mbarrier 同步, 是 fire-and-forget 的.
对受内存带宽限制的 workload, 这能将后续 TMA load 的延迟从 DRAM 级降到 L2 级.

TMA Prefetch 分两阶段:
1. Initial Prefetch: 在 TMA load 循环开始前, 对前 `prefetch_dist` 个 k-tile 发出 prefetch, 批量预热 L2.
2. Rolling Prefetch: TMA load 循环中, 每次发出当前 k-tile 的 TMA load 后,
   紧跟一条对 k_tile_idx + prefetch_dist 的 prefetch, 滚动维持 L2 预热窗口.

相对 `fp16_gemm_3_1.py` 的配置变化:
* `cluster_shape_mnk`:   (2,1,1) -> (2,2,1), M/N 维各 2 个 CTA
* `mma_inst_shape_mnk`:  (256,256,16) -> (256,64,16), MMA tile N 维缩小
* `mma_tiler_mnk`:       (256,256,64) -> (256,64,64), 对应调整
* `ab_stages`:           6 -> 10, 更多 pipeline buffer
* `prefetch_dist`:       新增, = ab_stages = 10, 预取深度匹配 pipeline 深度

为什么要改变 MMA size 和其他配置?
  Prefetch 需要足够深的 pipeline 才能掩盖 DRAM 延迟, 预取后数据到达 L2 需要时间, 如果 pipeline 太浅 (stage 少), 
  TMA load 会在数据到达 L2 之前就开始, prefetch 形同虚设.因此 `ab_stages` 从 6 增加到 10. 
  但 SMEM 总量固定, 更多 stage 意味着每个 stage 的 buffer 必须更小, 缩小 N 维 tile 是最直接的办法, 
  即 B tile 每 stage 从 256*64*2B = 32KB 降到 64*64*2B = 8KB, 腾出空间容纳额外的 stage.
  同时 `cluster_shape_mnk` N 维从 1 扩到 2, 让 B 的 TMA load 可以 multicast 到 N 维上两个 CTA, 
  用带宽换数据复用, 补偿 per-CTA tile 缩小带来的效率下降.

kernel 内部的改动:
  kernel 函数体相对 fp16_gemm_3_1.py 只在 TMA warp 分支新增了三处 prefetch 代码:
  1. `prefetch_dist = ab_stages` — 定义预取距离, 与 pipeline 深度对齐
  2. Initial Prefetch — 在 TMA load 循环开始前, 一次性对前 `prefetch_dist` 个 k-tile 发出 prefetch, 
     批量预热 L2, 让最初几轮 TMA load 就能命中 L2 而非等 DRAM.
  3. Rolling Prefetch — TMA load 循环内, 每发出一个 k-tile 的 TMA load 后, 紧跟一条
     对 `k_tile_idx + prefetch_dist` 的 prefetch, 滚动维持 L2 预热窗口, 使后续 TMA load 始终能从 L2 取数据.

运行本示例:
.. code-block:: bash
    python examples/blackwell/tutorial_gemm/fp16_gemm_5.py  \
      --mnk 8192,8192,8192

本示例约束:
* m 与 n 的问题规模必须能被 tile 尺寸 m & n (256, 256) 整除
"""

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
use_2cta_instrs = True
# [新增] cluster 形状由 (2,1,1) 改为 (2,2,1), M、N 维各 2 个 CTA
cluster_shape_mnk = (2, 2, 1) if use_2cta_instrs else (1, 1, 1)
# [新增] MMA 指令 tile 由 (256,256,16) 改为 (256,64,16)
mma_inst_shape_mnk = (256, 64, 16)
# [新增] MMA tiler MxN 由 (256,256) 改为 (256,64)
mma_tiler_mnk = (256, 64, 64)
threads_in_epilogue = 128  # 每个 CTA 的 epilogue 线程数

# Pipeline stage 配置
# [新增] ab_stages 由 6 增至 10
ab_stages = 10
epi_stages = 2
acc_stages = 2
num_clc_stage = 1

# 调度器
use_clc_dynamic_scheduler = True
scheduler_type = (
    utils.ClcDynamicPersistentTileScheduler
    if use_clc_dynamic_scheduler
    else utils.StaticPersistentTileScheduler
)
# 应答大小为 4B * 4 个元素
num_clc_response_bytes = 16


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32
    # 仅用于 CLC Dynamic Scheduler
    clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    clc_response: cute.struct.MemRange[cutlass.Int32, 4]


@cute.kernel()
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    c_smem_layout_kind: cutlass.Constexpr,
    epi_smem_layout_staged: cute.ComposedLayout,
    epi_tile: cute.Tile,
    cta_layout_vmnk: cute.Layout,
    tile_sched_params: Union[
        utils.ClcDynamicPersistentTileSchedulerParams,
        utils.PersistentTileSchedulerParams,
    ],
):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

    mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
    is_leader_cta = mma_tile_coord_v == 0

    epilogue_warp_ids = (
        0,
        1,
        2,
        3,
    )
    mma_warp_id = 4
    tma_warp_id = 5
    # 仅 dynamic scheduler 使用:第 7 个 warp (`sched_warp_id = 6`) 专职驱动 CLC tile 调度
    sched_warp_id = 6

    epilog_sync_bar_id = 1
    tmem_alloc_sync_bar_id = 2

    # 预取 TMA descriptor
    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_c)

    # 与同一行、同一列上发射 MMA 的线程数相同; 减 1 避免同一线程被重复计数
    num_mcast_participants = (
        cute.size(cta_layout_vmnk, mode=[1]) + cute.size(cta_layout_vmnk, mode=[2]) - 1
    )

    # Mcast mask 初始化
    tma_mcast_mask_a = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
    )
    tma_mcast_mask_b = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
    )

    # 分配 SMEM
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # Epilogue 同步用的 barrier 1
    epilogue_sync_barrier = pipeline.NamedBarrier(
        barrier_id=epilog_sync_bar_id,
        num_threads=threads_in_epilogue,
    )

    # 仅 MMA warp 与 epilogue warps 参与 TMEM 分配同步; TMA warp 不参与
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=tmem_alloc_sync_bar_id,
        num_threads=32
        * len((mma_warp_id, *epilogue_warp_ids)),  # 5 个 warp = 160 线程
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buffer,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilogue_warp_ids[0],
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
    )

    num_tma_copy_bytes = (
        cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
        + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ) * cute.size(cta_layout_vmnk, mode=[0])

    # 参与 mainloop pipeline 的线程/warp
    mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, size=num_mcast_participants
    )

    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=ab_stages,
        producer_group=mainloop_pipeline_producer_group,
        consumer_group=mainloop_pipeline_consumer_group,
        tx_count=num_tma_copy_bytes,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    # 参与 accumulator pipeline 的线程/warp
    acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    acc_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        size=cute.size(cta_layout_vmnk, mode=[0]) * len(epilogue_warp_ids),
    )

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=acc_stages,
        producer_group=acc_pipeline_producer_group,
        consumer_group=acc_pipeline_consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    # 初始化 `clc_pipeline`(barrier)及 consumer 状态; 仅 CLC Dynamic Scheduler 使用
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_shape_mnk)
        num_clc_consumer_threads = 32 * len(
            (
                sched_warp_id,
                *(
                    cluster_size
                    * (
                        mma_warp_id,
                        tma_warp_id,
                        *epilogue_warp_ids,
                    )
                ),
            )
        )
        clc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_clc_consumer_threads
        )
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=num_clc_stage,
            producer_group=clc_pipeline_producer_group,
            consumer_group=clc_pipeline_consumer_group,
            tx_count=num_clc_response_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # 初始 CLC 应答指针
        clc_response_ptr = storage.clc_response.data_ptr()

        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_clc_stage
        )
    else:
        clc_pipeline = None
        clc_response_ptr = None
        clc_consumer_state = None

    # barrier 初始化后 cluster arrive
    pipeline_init_arrive(cluster_shape_mn=cluster_shape_mnk, is_relaxed=True)

    # 分配 SMEM
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )
    sC = smem.allocate_tensor(
        element_type=io_dtype,
        layout=epi_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=epi_smem_layout_staged.inner,
    )

    # 为 MMA 划分 tensor 并生成 fragments
    # (bM, bK, RestM, RestK)
    gA = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None)
    )
    # (bN, bK, RestN, RestK)
    gB = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None)
    )
    # (bM, bN, RestM, RestN)
    gC = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None)
    )

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    # (MMA, MMA_M, MMA_K, RestM, RestK)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K, RestN, RestK)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N, RestM, RestN)
    tCgC = thr_mma.partition_C(gC)

    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)

    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N, STAGE)
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))

    # 为 TMA 划分 tensor; 需先完成面向 MMA 的划分
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )

    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK)
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None)], epi_tile)

    tCsC, tCgC_tma = cute.nvgpu.cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(gC_epi, 0, 2),
    )

    # 开始计算前 cluster wait
    pipeline_init_wait(cluster_shape_mn=cluster_shape_mnk)

    # 构造 tile 调度器
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        tile_sched = scheduler_type.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
    else:
        tile_sched = scheduler_type.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
    work_tile = tile_sched.initial_work_tile_info()

    #
    # 主循环
    #

    num_k_tiles = cute.size(gA, mode=[3])

    # [新增] 预取距离 `prefetch_dist`, 向前预取多少个 k-tile 到 L2 cache, 本例取 `prefetch_dist = ab_stages`
    # 在 TMA copy 需要之前把数据带到 L2,有助于掩盖 DRAM latency
    prefetch_dist = ab_stages

    # TMA warp(含 Prefetch)
    if warp_idx == tma_warp_id:
        #
        # Persistent tile 调度循环
        #
        while work_tile.is_valid_tile:
            # 从 tile 调度器取当前 tile 坐标
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            # 切到当前 MMA tile 索引
            # ((atom_v, rest_v), RestK)
            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None)]

            # ((atom_v, rest_v), RestK)
            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None)]

            # =========================================================
            # [新增] TMA Prefetch — Initial Prefetch Phase
            # =========================================================
            # 将前 `prefetch_dist` 个 k-tile 预取到 L2 cache(循环预取前 N 个 k-tile 到 L2)
            # 在 TMA copy 开始前先预热 L2
            for pf_k_tile in cutlass.range(
                cutlass.min(prefetch_dist, num_k_tiles), unroll=1
            ):
                cute.prefetch(tma_atom_a, tAgA_slice[(None, pf_k_tile)])
                cute.prefetch(tma_atom_b, tBgB_slice[(None, pf_k_tile)])

            for k_tile_idx in range(num_k_tiles):
                # 等待 A/B buffer 空出后再写入本次 load
                handle = ab_producer.acquire_and_advance()

                # 发起 TMA load(`k_tile_idx` 用法同 fp16_gemm_3_1.py)
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k_tile_idx)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=tma_mcast_mask_a,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k_tile_idx)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=tma_mcast_mask_b,
                )
                # =========================================================
                # [新增] Rolling Prefetch
                # =========================================================
                # 将更靠后的 k-tile 预取进 L2 cache(主循环中滚动预取未来的 k-tile)
                # 沿 K 维推进时持续维持 L2 预热
                if k_tile_idx + prefetch_dist < num_k_tiles:
                    future_k_tile = k_tile_idx + prefetch_dist
                    cute.prefetch(tma_atom_a, tAgA_slice[(None, future_k_tile)])
                    cute.prefetch(tma_atom_b, tBgB_slice[(None, future_k_tile)])

            # 前进到下一个 tile
            if cutlass.const_expr(use_clc_dynamic_scheduler):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # `mbarrier_wait` 防止 cluster 内相互依赖的 thread block(在 TMA/MMA 同步语义下)
        # 过早退出,从而导致过晚的 tcgen05 `commit_arrive` 不合法
        ab_producer.tail()

    # Sched warp(仅 dynamic scheduler)
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        is_first_cta_in_cluster = cta_rank_in_cluster == 0

        if warp_idx == sched_warp_id and is_first_cta_in_cluster:
            # Persistent tile 调度循环
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, num_clc_stage
            )
            while work_tile.is_valid_tile:
                # 前进到下一个 tile
                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

    # MMA warp
    if warp_idx == mma_warp_id:
        # 等待 TMEM 分配完成并取回指针
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        while work_tile.is_valid_tile:
            if is_leader_cta:
                # 等待 accumulator buffer 空出
                acc_empty = acc_producer.acquire_and_advance()

                # 为当前 tile 设置 tensor memory buffer
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile_idx in range(num_k_tiles):
                    # 等待 TMA copy 完成
                    handle = ab_consumer.wait_and_advance()

                    # 执行一个 K-block 规模的 MMA 指令序列
                    num_k_blocks = cute.size(tCrA, mode=[2])
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    # 声明 A/B buffer 已消费完毕,可供下一轮 load
                    handle.release()

                # 声明 accumulator 已全部算完
                acc_empty.commit()

            # 前进到下一个 tile
            if cutlass.const_expr(use_clc_dynamic_scheduler):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # 等待 accumulator pipeline 收尾(buffer 空)
        acc_producer.tail()

    # Epilogue warps
    if warp_idx < mma_warp_id:
        # 分配 TMEM(实际只有 epilogue warp 0 执行分配)
        num_tmem_cols = 512
        tmem.allocate(num_tmem_cols)

        # 等待 TMEM 分配完成并取回指针
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # 初始化 epilogue 的 TMA store pipeline
        epilogue_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=128,
        )
        epilogue_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=epi_stages,
            producer_group=epilogue_pipeline_producer_group,
        )

        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE),
            cutlass.Float32,
        )

        while work_tile.is_valid_tile:
            # 从 tile 调度器取当前 tile 坐标
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            # 等待 accumulator buffer 已满(可读)
            acc_full = acc_consumer.wait_and_advance()

            # 为当前 tile 设置 tensor memory buffer
            # (MMA, MMA_M, MMA_N)
            tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]

            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
            tCtAcc_epi = cute.flat_divide(
                tCtAcc[((None, None), 0, 0)],  # 为何取 0,0？与 MMA 子布局中选定子 tile 一致
                epi_tile,
            )

            mma_tile_coord_mn = cute.slice_(mma_tile_coord_mnl, (None, None, 0))
            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN)
            tCgC_epi = cute.flat_divide(
                tCgC[((None, None), 0, 0, *mma_tile_coord_mn)], epi_tile
            )

            tCgC_tma_cur_tile = tCgC_tma[(None, None, None, *mma_tile_coord_mn)]

            # TMEM -> RMEM 的 tiled copy
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
            # (T2R, T2R_M, T2R_N)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0)].shape, cutlass.Float32
            )
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

            # RMEM -> SMEM 的 copy atom 与 tiled copy
            copy_atom_r2s = cutlass.utils.blackwell_helpers.get_smem_store_op(
                c_smem_layout_kind, cutlass.Float32, cutlass.Float32, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            # (R2S, R2S_M, R2S_N, PIPE_D)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)

            tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
            tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, io_dtype)
            tCgC_grouped = cute.group_modes(
                tCgC_tma_cur_tile, 1, cute.rank(tCgC_tma_cur_tile)
            )

            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

            # Epilogue 子 tile 循环
            for subtile_idx in cutlass.range(subtile_cnt):
                # TMEM -> RMEM
                tTR_tAcc_slice = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_slice, tTR_rAcc)

                # RMEM -> SMEM
                c_buffer = subtile_idx % epi_stages
                tRS_sC_slice = tRS_sC[(None, None, None, c_buffer)]

                # 类型转换
                tRS_rC.store(tRS_rAcc.load().to(io_dtype))

                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC_slice)

                # 内存 fence 与 barrier,保证 SMEM 对后续 TMA store 可见
                cute.arch.fence_view_async_shared()
                epilogue_sync_barrier.arrive_and_wait()
                # SMEM -> GMEM
                if warp_idx == epilogue_warp_ids[0]:
                    cute.copy(
                        tma_atom_c,
                        tCsC[(None, c_buffer)],
                        tCgC_grouped[(None, subtile_idx)],
                    )

                    epilogue_pipeline.producer_commit()
                    epilogue_pipeline.producer_acquire()
                epilogue_sync_barrier.arrive_and_wait()

            # 异步 arrive:accumulator buffer 已空出
            with cute.arch.elect_one():
                acc_full.release()

            # 前进到下一个 tile
            if cutlass.const_expr(use_clc_dynamic_scheduler):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # 等待 C 的 store 全部完成
        epilogue_pipeline.producer_tail()

        # 释放 tensor memory buffer
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


def compute_grid(
    c: cute.Tensor,
    mma_tiler_mnk: Tuple[int, int, int],
    cluster_shape_mnk: Tuple[int, int, int],
    scheduler_type: Union[
        utils.StaticPersistentTileScheduler, utils.ClcDynamicPersistentTileScheduler
    ],
    max_active_clusters: cutlass.Constexpr,
) -> Tuple[
    Union[
        utils.ClcDynamicPersistentTileSchedulerParams,
        utils.PersistentTileSchedulerParams,
    ],
    Tuple[int, int, int],
]:
    c_shape = cute.slice_(mma_tiler_mnk, (None, None, 0))
    gc = cute.zipped_divide(c, tiler=c_shape)
    num_ctas_mn = gc[(0, (None, None))].shape

    if cutlass.const_expr(
        issubclass(scheduler_type, utils.ClcDynamicPersistentTileScheduler)
    ):
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (*num_ctas_mn, 1), cluster_shape_mnk
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
    else:
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (*num_ctas_mn, 1), cluster_shape_mnk
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
    return tile_sched_params, grid


@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
):
    #
    # 构造 tiled MMA
    #

    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    #
    # 为 A、B 构造 SMEM layout
    #

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )

    # `c_smem_layout_kind` 为行/列主序的枚举,不是 CuTe layout 对象
    c_smem_layout_kind = utils.LayoutEnum.from_tensor(c)

    #
    # 构造 VMNK layout
    #

    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

    #
    # 构造 TMA load atom
    #

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
        tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
    )
    a_smem_layout_slice = cute.slice_(a_smem_layout, (None, None, None, 0))
    tma_atom_a, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )
    b_smem_layout_slice = cute.slice_(b_smem_layout, (None, None, None, 0))
    tma_atom_b, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )

    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )

    epi_tile = utils.compute_epilogue_tile_shape(
        cta_tile_shape_mnk,
        use_2cta_instrs,
        c_smem_layout_kind,
        io_dtype,
    )

    epi_smem_layout_staged = cutlass.utils.blackwell_helpers.make_smem_layout_epi(
        io_dtype,
        c_smem_layout_kind,
        epi_tile,
        epi_stages,
    )

    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    tma_atom_c, c_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        c,
        epi_smem_layout,
        epi_tile,
    )

    #
    # 启动 kernel
    #

    tile_sched_params, grid_shape = compute_grid(
        c,
        cta_tile_shape_mnk,
        cluster_shape_mnk,
        scheduler_type,
        max_active_clusters,
    )

    kernel(
        tiled_mma,
        tma_atom_a,
        a_tma_tensor,
        tma_atom_b,
        b_tma_tensor,
        tma_atom_c,
        c_tma_tensor,
        a_smem_layout,
        b_smem_layout,
        c_smem_layout_kind,
        epi_smem_layout_staged,
        epi_tile,
        cta_layout_vmnk,
        tile_sched_params,
    ).launch(
        grid=grid_shape,
        block=[224, 1, 1] if use_clc_dynamic_scheduler else [192, 1, 1],
        cluster=cluster_shape_mnk,
    )


def run_dense_gemm(
    mnk: Tuple[int, int, int],
    tolerance: float,
):
    global torch, cutlass_torch
    import torch
    import cutlass.torch as cutlass_torch

    print("===================================================================")
    print("Running Blackwell fp16 GEMM example 5 (with TMA prefetch):")
    print(f"  mnk:       {mnk}")
    print(f"  tolerance: {tolerance}")
    print("===================================================================")
    print()

    m, n, k = mnk
    torch.manual_seed(1111)

    # 构造 K-major 张量(torch 默认为 row-major,需满足布局约定)
    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        return (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(device="cuda", dtype=dtype)
        )

    a = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
    b = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
    c = make_tensors(m, n, cutlass_torch.dtype(io_dtype))
    a_memref = from_dlpack(a).mark_layout_dynamic()
    b_memref = from_dlpack(b).mark_layout_dynamic()
    c_memref = from_dlpack(c).mark_layout_dynamic()

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mnk[0] * cluster_shape_mnk[1]
    )

    # host JIT 函数入口
    host_function(
        a_memref,
        b_memref,
        c_memref,
        max_active_clusters,
        no_cache=True,
    )

    # 计算参考结果并校验
    ref = (torch.einsum("mk,nk->mn", a, b)).cpu()
    torch.testing.assert_close(
        c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=tolerance, rtol=1e-05
    )


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str):
        try:
            return [int(x.strip()) for x in s.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    from cuda.bindings import driver as cu_driver

    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run this example")

    parser = argparse.ArgumentParser(
        description="Blackwell fp16 GEMM example 5 (with TMA prefetch)"
    )
    parser.add_argument(
        "--mnk",
        type=parse_comma_separated_ints,
        default=(8192, 8192, 8192),
        help="MNK dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    args = parser.parse_args()
    if len(args.mnk) != 3:
        parser.error("--mnk must contain exactly 3 values")

    run_dense_gemm(
        args.mnk,
        args.tolerance,
    )
    print("PASS")
