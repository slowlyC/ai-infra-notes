# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# 第四个 GEMM 教程. 在 fp16_gemm_3_1.py(动态持久 tile 调度器)基础上解决的问题是: cluster 形状固定时 SM 利用率不高。
# 增加了 Preferred Cluster 与 Dynamic Cluster 支持.
# [新增] preferred_cluster_shape_mnk / fallback_cluster_shape_mnk; mega-kernel 内按 is_preferred_cluster 分支; 
# [新增] cluster_specific_kernel 抽取公共逻辑; launch 使用 fallback_cluster= 指定 fallback cluster.


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
第四个 GEMM 教程, 演示 CuTeDSL 中的 Preferred Cluster 与 Dynamic Cluster.

Cluster 是 Hopper 开始引入的 CTA 组织层级, cluster 内的 CTA 可以通过 TMA multicast 共享数据。
更大的 cluster 意味着更高的 multicast 效率, 但可能因量化导致 SM 占用率不佳.
例如 18 个 SM 的 GPU 上使用 2x2 cluster 时可能只用上 16 个 SM, 余下 2 个 SM 空闲.

自 Compute Capability 10.0 起, 模型支持同时指定两种 cluster: preferred cluster 与 fallback cluster.
一次 kernel launch 同时指定两种 cluster 形状, 硬件优先使用大的 (preferred),
SM 不够凑齐一个大 cluster 时自动 fallback 到小的, 把边角 SM 也利用起来。
延续刚才的例子, 可再启动额外的 2x1 cluster 以利用空闲的 2 个 SM.

术语说明
  * Static cluster: cluster 形状在编译期确定.
  * Dynamic Cluster: cluster 形状在运行时由 host 设定.
  * Preferred cluster: 一次 kernel launch 可同时给出 preferred 与 fallback 两种 cluster 形状.

Preferred 与 fallback cluster 形状需满足若干约束.
  * Preferred cluster 的深度(Z 维)必须与 fallback cluster 相同.
  * Fallback cluster 形状必须能整除 preferred cluster 形状.
  * Preferred cluster 形状必须能整除 kernel launch 的 grid 形状.

本示例展示如何在 CuTe DSL Blackwell SM100 kernel 中使用 Dynamic Cluster 与 Preferred Cluster;
可通过 kernel launch 参数指定 preferred 与 fallback cluster 形状.

[新增] 相对 fp16_gemm_3_1.py: mega-kernel 根据运行时 cluster 维度判断 is_preferred_cluster,
分别传入对应 TMA 张量与 tile_sched_params, launch 使用 cluster= 与 fallback_cluster=.

运行本示例:
.. code-block:: bash
    python examples/blackwell/tutorial_gemm/fp16_gemm_4.py  \
      --mnk 8192,8192,8192

本示例约束:
* m 与 n 的问题规模必须能被 tile 尺寸 m & n (256, 256) 整除
"""

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
use_2cta_instrs = True
# [新增] fallback cluster 形状 (与 fp16_gemm_3_1.py 中单一带宽的 cluster 对应)
fallback_cluster_shape_mnk = (2, 1, 1) if use_2cta_instrs else (1, 1, 1)
# [新增] preferred cluster 形状 (更大 N 维 multicast 能力)
preferred_cluster_shape_mnk = (2, 4, 1) if use_2cta_instrs else (1, 1, 1)
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
threads_in_epilogue = 128  # 每个 CTA 的 epilogue 线程数

# Pipeline stage 配置
ab_stages = 6
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
# CLC 响应大小为 4B * 4 个元素
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


# [新增] 抽取各 cluster 形状下的公共 kernel 逻辑 (TMA/MMA/epilogue 与调度).
# 函数体与 fp16_gemm_3_1.py 的 kernel() 完全相同, 仅有两处结构变化:
#   1. 装饰器从 @cute.kernel() 改为 @cute.jit, 因为不再是入口 kernel, 而是被外层 mega-kernel 调用的 device 函数.
#   2. cluster_shape_mnk 从模块级全局变量改为函数参数, 因为 preferred/fallback 两种 cluster 需要传入不同的值.
@cute.jit
def cluster_specific_kernel(
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
    cluster_shape_mnk: Tuple[int, int, int],
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
    # sched_warp_id 仅用于动态调度器
    sched_warp_id = 6

    epilog_sync_bar_id = 1
    tmem_alloc_sync_bar_id = 2

    # Prefetch TMA 描述符
    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_c)

    # 参与者数量与同一行/列上发射 MMA 的线程数一致
    # 减 1 避免同一线程被重复计数
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

    # Barrier 1: epilogue 同步
    epilogue_sync_barrier = pipeline.NamedBarrier(
        barrier_id=epilog_sync_bar_id,
        num_threads=threads_in_epilogue,
    )

    # 仅 MMA warp 与 epilogue warps 参与 TMEM 分配同步
    # TMA warp 不参与
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

    # 参与累加器 pipeline 的线程/warp
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

    # 初始化 clc_pipeline(barrier)与状态
    # 仅用于 CLC Dynamic Scheduler
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
        # 初始 clc 响应指针
        clc_response_ptr = storage.clc_response.data_ptr()

        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_clc_stage
        )
    else:
        clc_pipeline = None
        clc_response_ptr = None
        clc_consumer_state = None

    # barrier 初始化后的 cluster arrive
    pipeline_init_arrive(cluster_shape_mn=cluster_shape_mnk, is_relaxed=True)

    # 分配 SMEM(A/B/C)
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

    # 为 MMA 划分张量并构造 fragment
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

    # 为 TMA 划分张量; 需先完成 MMA 侧划分
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

    # 开始计算前的 cluster wait
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

    # TMA warp
    if warp_idx == tma_warp_id:
        #
        # 持久化 tile 调度循环
        #
        while work_tile.is_valid_tile:
            # 从 tile 调度器取得 tile 坐标
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            # 切片到每个 MMA tile 下标
            # ((atom_v, rest_v), RestK)
            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None)]

            # ((atom_v, rest_v), RestK)
            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None)]

            for k_tile_idx in range(num_k_tiles):
                # 等待 A/B buffer 空闲后再载入
                handle = ab_producer.acquire_and_advance()

                # 发起 TMA load
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

            # 前进到下一个 tile
            if cutlass.const_expr(use_clc_dynamic_scheduler):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # 此处 mbarrier_wait 防止 cluster 内一组相互依赖的 thread block(在 TMA/MMA 同步语义下)
        # 过早退出, 从而导致迟到的 tcgen05 commit_arrive 不合法
        ab_producer.tail()

    # 调度 warp(仅动态调度器)
    if cutlass.const_expr(use_clc_dynamic_scheduler):
        is_first_cta_in_cluster = cta_rank_in_cluster == 0

        if warp_idx == sched_warp_id and is_first_cta_in_cluster:
            # 持久化 tile 调度循环
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

    # MMA warp(矩阵乘累加)
    if warp_idx == mma_warp_id:
        # 等待 TMEM 分配并取回指针
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        while work_tile.is_valid_tile:
            if is_leader_cta:
                # 等待累加器 buffer 空闲
                acc_empty = acc_producer.acquire_and_advance()

                # 为当前 tile 设置 tensor memory buffer
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile_idx in range(num_k_tiles):
                    # 等待 TMA 拷贝完成
                    handle = ab_consumer.wait_and_advance()

                    # 执行一个 K-block 规模的 MMA 指令
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

                    # 表示 A/B buffer 已消费, 可供下一轮 load
                    handle.release()

                # 表示累加器已算完当前 tile
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

        # 等待累加器 buffer 全部排空
        acc_producer.tail()

    # Epilogue warps(写回 C)
    if warp_idx < mma_warp_id:
        # 分配 TMEM(实际由 epilogue warp 0 执行分配)
        num_tmem_cols = 512
        tmem.allocate(num_tmem_cols)

        # 等待 TMEM 分配并取回指针
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
            # 从 tile 调度器取得 tile 坐标
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            # 等待累加器 buffer 就绪
            acc_full = acc_consumer.wait_and_advance()

            # 为当前 tile 设置 tensor memory buffer
            # (MMA, MMA_M, MMA_N)
            tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]

            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
            tCtAcc_epi = cute.flat_divide(
                tCtAcc[((None, None), 0, 0)],  # 为何 0,0 ?
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

                # 内存 fence 与 barrier, 保证 SMEM 对后续 TMA store 可见
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

            # 异步 arrive:累加器 buffer 已空
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

        # 等待 C 写回完成
        epilogue_pipeline.producer_tail()

        # 释放 tensor memory buffer
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


# [新增] 外层 mega-kernel:运行时判定 preferred / fallback cluster 并调用 cluster_specific_kernel.
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a_preferred: cute.CopyAtom,
    mA_mkl_preferred: cute.Tensor,
    tma_atom_b_preferred: cute.CopyAtom,
    mB_nkl_preferred: cute.Tensor,
    tma_atom_a_fallback: cute.CopyAtom,
    mA_mkl_fallback: cute.Tensor,
    tma_atom_b_fallback: cute.CopyAtom,
    mB_nkl_fallback: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    c_smem_layout_kind: cutlass.Constexpr,
    epi_smem_layout_staged: cute.ComposedLayout,
    epi_tile: cute.Tile,
    preferred_cta_layout_vmnk: cute.Layout,
    fallback_cta_layout_vmnk: cute.Layout,
    preferred_tile_sched_params: Union[
        utils.ClcDynamicPersistentTileSchedulerParams,
        utils.PersistentTileSchedulerParams,
    ],
    fallback_tile_sched_params: Union[
        utils.ClcDynamicPersistentTileSchedulerParams,
        utils.PersistentTileSchedulerParams,
    ],
):
    # [新增] 读取 cluster 维度, 判断当前 block 是否处于 preferred cluster
    cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
    is_preferred_cluster = (
        cbdim_x == preferred_cluster_shape_mnk[0]
        and cbdim_y == preferred_cluster_shape_mnk[1]
        and cbdim_z == preferred_cluster_shape_mnk[2]
    )

    # [新增] mega-kernel:在单一 kernel 内按 is_preferred_cluster 分支到对应 TMA 与调度参数
    if is_preferred_cluster:
        cluster_specific_kernel(
            tiled_mma,
            tma_atom_a_preferred,
            mA_mkl_preferred,
            tma_atom_b_preferred,
            mB_nkl_preferred,
            tma_atom_c,
            mC_mnl,
            a_smem_layout,
            b_smem_layout,
            c_smem_layout_kind,
            epi_smem_layout_staged,
            epi_tile,
            preferred_cta_layout_vmnk,
            preferred_cluster_shape_mnk,
            preferred_tile_sched_params,
        )
    else:
        cluster_specific_kernel(
            tiled_mma,
            tma_atom_a_fallback,
            mA_mkl_fallback,
            tma_atom_b_fallback,
            mB_nkl_fallback,
            tma_atom_c,
            mC_mnl,
            a_smem_layout,
            b_smem_layout,
            c_smem_layout_kind,
            epi_smem_layout_staged,
            epi_tile,
            fallback_cta_layout_vmnk,
            fallback_cluster_shape_mnk,
            fallback_tile_sched_params,
        )


# [新增] 相对 fp16_gemm_3_1.py: 分别计算 preferred / fallback 的调度参数与 grid 形状.
def compute_grid(
    c: cute.Tensor,
    cta_tiler_mnk: Tuple[int, int, int],
    preferred_cluster_shape_mnk: Tuple[int, int, int],
    fallback_cluster_shape_mnk: Tuple[int, int, int],
    scheduler_type: Union[
        utils.StaticPersistentTileScheduler, utils.ClcDynamicPersistentTileScheduler
    ],
    preferred_max_active_clusters: cutlass.Constexpr,
    fallback_max_active_clusters: cutlass.Constexpr,
) -> Tuple[
    Union[
        utils.ClcDynamicPersistentTileSchedulerParams,
        utils.PersistentTileSchedulerParams,
    ],
    Tuple[int, int, int],
    Union[
        utils.ClcDynamicPersistentTileSchedulerParams,
        utils.PersistentTileSchedulerParams,
    ],
    Tuple[int, int, int],
]:
    c_shape = cute.slice_(cta_tiler_mnk, (None, None, 0))
    gc = cute.zipped_divide(c, tiler=c_shape)
    num_ctas_mn = gc[(0, (None, None))].shape

    # [新增] 为 preferred cluster 计算调度器参数与 grid.
    # 静态调度器 grid 大小 = max_active_clusters * cluster_size, 即按 GPU 能同时驻留多少个 cluster 来定, 
    # 动态调度器 (CLC) grid 大小由调度器自行计算, 不需要 max_active_clusters.
    if cutlass.const_expr(
        issubclass(scheduler_type, utils.ClcDynamicPersistentTileScheduler)
    ):
        preferred_tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (*num_ctas_mn, 1), preferred_cluster_shape_mnk
        )
        preferred_grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(
            preferred_tile_sched_params
        )
    else:
        preferred_tile_sched_params = utils.PersistentTileSchedulerParams(
            (*num_ctas_mn, 1), preferred_cluster_shape_mnk
        )
        preferred_grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            preferred_tile_sched_params, preferred_max_active_clusters
        )

    # [新增] 为 fallback cluster 计算调度器参数与 grid
    if cutlass.const_expr(
        issubclass(scheduler_type, utils.ClcDynamicPersistentTileScheduler)
    ):
        fallback_tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (*num_ctas_mn, 1), fallback_cluster_shape_mnk
        )
        fallback_grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(
            fallback_tile_sched_params
        )
    else:
        # 静态调度器下 fallback 路径仍使用 preferred 的 tile_sched 参数形状
        fallback_tile_sched_params = utils.PersistentTileSchedulerParams(
            (*num_ctas_mn, 1), preferred_cluster_shape_mnk
        )
        fallback_grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            fallback_tile_sched_params, fallback_max_active_clusters
        )

    return (
        preferred_tile_sched_params,
        preferred_grid,
        fallback_tile_sched_params,
        fallback_grid,
    )


# [新增] 由调用侧传来 preferred / fallback cluster 的 max_active_clusters.
@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    preferred_max_active_clusters: cutlass.Constexpr,
    fallback_max_active_clusters: cutlass.Constexpr,
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
    # 构造 A/B 的 SMEM layout
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

    # c_smem_layout_kind 为行/列主序枚举, 非 CuTe layout 对象
    c_smem_layout_kind = utils.LayoutEnum.from_tensor(c)

    #
    # [新增] 构造 VMNK layout:分别为 fallback 与 preferred cluster
    #

    fallback_cta_layout_mnk = cute.make_layout(fallback_cluster_shape_mnk)
    fallback_cta_layout_vmnk = cute.tiled_divide(
        fallback_cta_layout_mnk, (tiled_mma.thr_id,)
    )

    preferred_cta_layout_mnk = cute.make_layout(preferred_cluster_shape_mnk)
    preferred_cta_layout_vmnk = cute.tiled_divide(
        preferred_cta_layout_mnk, (tiled_mma.thr_id,)
    )

    #
    # [新增] 构造 TMA load atom:为 fallback 与 preferred 各建一套(multicast 形状不同)
    #

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
        tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
    )
    a_smem_layout_slice = cute.slice_(a_smem_layout, (None, None, None, 0))
    tma_atom_a_fallback, a_tma_tensor_fallback = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        fallback_cta_layout_vmnk.shape,
    )
    b_smem_layout_slice = cute.slice_(b_smem_layout, (None, None, None, 0))
    tma_atom_b_fallback, b_tma_tensor_fallback = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        fallback_cta_layout_vmnk.shape,
    )

    tma_atom_a_preferred, a_tma_tensor_preferred = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        preferred_cta_layout_vmnk.shape,
    )
    tma_atom_b_preferred, b_tma_tensor_preferred = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        preferred_cta_layout_vmnk.shape,
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

    (
        tile_sched_params_preferred,
        grid_shape_preferred,
        tile_sched_params_fallback,
        grid_shape_fallback,
    ) = compute_grid(
        c,
        cta_tile_shape_mnk,
        preferred_cluster_shape_mnk,
        fallback_cluster_shape_mnk,
        scheduler_type,
        preferred_max_active_clusters,
        fallback_max_active_clusters,
    )

    kernel(
        tiled_mma,
        tma_atom_a_preferred,
        a_tma_tensor_preferred,
        tma_atom_b_preferred,
        b_tma_tensor_preferred,
        tma_atom_a_fallback,
        a_tma_tensor_fallback,
        tma_atom_b_fallback,
        b_tma_tensor_fallback,
        tma_atom_c,
        c_tma_tensor,
        a_smem_layout,
        b_smem_layout,
        c_smem_layout_kind,
        epi_smem_layout_staged,
        epi_tile,
        preferred_cta_layout_vmnk,
        fallback_cta_layout_vmnk,
        tile_sched_params_preferred,
        tile_sched_params_fallback,
    ).launch(
        grid=grid_shape_preferred,
        block=[224, 1, 1] if use_clc_dynamic_scheduler else [192, 1, 1],
        cluster=preferred_cluster_shape_mnk,
        # [新增] Dynamic Cluster: 指定 fallback cluster 形状
        fallback_cluster=fallback_cluster_shape_mnk,
    )


def run_dense_gemm(
    mnk: Tuple[int, int, int],
    tolerance: float,
):
    global torch, cutlass_torch
    import torch
    import cutlass.torch as cutlass_torch

    print("===================================================================")
    print("Running Blackwell fp16 GEMM example 4 (with MIX cluster size support):")
    print(f"  mnk:                   {mnk}")
    print(f"  tolerance:             {tolerance}")
    print(f"  preferred cluster:     {preferred_cluster_shape_mnk}")
    print(f"  fallback cluster:      {fallback_cluster_shape_mnk}")
    print(f"  CTA tile (MNK):        {mma_tiler_mnk}")
    print("===================================================================")
    print()

    m, n, k = mnk
    torch.manual_seed(1111)

    # 构造 K-major 张量(torch 默认为 row-major)
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

    # [新增] 相对 fp16_gemm_3_1.py: 分别为 preferred / fallback cluster 查询 max_active_clusters.
    preferred_max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        preferred_cluster_shape_mnk[0] * preferred_cluster_shape_mnk[1]
    )
    fallback_max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        fallback_cluster_shape_mnk[0] * fallback_cluster_shape_mnk[1]
    )

    # 进入 host 侧 JIT 函数
    host_function(
        a_memref,
        b_memref,
        c_memref,
        preferred_max_active_clusters,
        fallback_max_active_clusters,
        no_cache=True,
    )

    # 计算参考结果并校验
    print("Validating against reference...")
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
        description="Blackwell fp16 GEMM example 4 (with MIX CGA support)"
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
