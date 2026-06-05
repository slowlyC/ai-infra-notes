# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# 第六个 GEMM 教程. [新增] 在 fp16_gemm_3_1.py 基础上增加 PDL (Programmatic Dependent Launch, 编程式依赖启动) 支持.


import argparse

from typing import Tuple, Union
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


"""

传统 CUDA 中, 同一 stream 上的 kernel 严格串行, kernel A 的所有 thread block 全部退出后, kernel B 才能启动. 
即使 A 已经写完 B 需要的全部数据, B 仍要等 A 的最后一个 thread block.

PDL 是 Blackwell (sm_100) 引入的硬件特性, 允许在 kernel 内部显式控制 "什么时候可以启动下一个kernel", 
把原来的 "整体串行" 拆成可重叠的执行.

PDL 通过两条指令实现:
  - griddepcontrol.launch_dependents — kernel A 内部调用. 当 grid 中所有 thread block 都
    执行到这条指令时, GPU 运行时即可开始调度 kernel B (此时 A 可能还没执行完).
  - griddepcontrol.wait — kernel B 内部调用. 阻塞当前 thread block, 直到 kernel A 的所有
    thread block 都完成且 A 的内存写操作对 B 可见. 这是真正的数据依赖同步点.

时间线示意 (传统 vs PDL):
传统:   A 全部执行 ─────────────> B 执行
PDL:    A [ ... launch_dependents ... ]
        B        [prologue ... wait ... mainloop ...]
                  ^ B 提前启动   ^ B 等 A 完成后再读 A 的输出

本示例的场景: dequantize -> GEMM, GEMM 中的 B 操作数由 dequantize kernel 产生 (INT8 -> FP16).
  - dequantize: 读完量化数据后调用 launch_dependents, 提示运行时可提前启动 GEMM.
  - GEMM: TMA warp 在 mainloop 循环内、首次 load B 之前调用 griddepcontrol_wait, 确保 dequantize 已写完 B.
    GEMM 的 prologue (SMEM 分配、pipeline 初始化、barrier setup) 与 dequantize 的后半段重叠执行.

  对 mnk 256,8192,128 这种小规模问题, prologue/epilogue 时间占比大, PDL 重叠效果更明显 (约 1.16x 加速).
  大矩阵下 mainloop 主导, PDL 收益有限.

cutedsl中 启用 PDL 需两步:
  1. launch 时设置 use_pdl=True.
  2. kernel 中插入 griddepcontrol.launch_dependents 与 griddepcontrol.wait 指令.

kernel 内部的改动 (相对 fp16_gemm_3_1.py):
  - 新增的 dequantize kernel 是独立的简单 kernel, 在反量化完成前调用 launch_dependents.
  - gemm kernel 仅在 TMA warp 分支新增一处 griddepcontrol_wait() 调用 (mainloop 循环内, TMA load 之前),
    其余 MMA/Sched/Epilogue 逻辑完全不变.
  

运行本示例:
.. code-block:: bash
    python examples/blackwell/tutorial_gemm/fp16_gemm_6.py  \
      --mnk 256,8192,128

本示例约束:
* m 与 n 的问题规模必须能被 tile 尺寸 m & n (256, 256) 整除
"""

io_dtype = cutlass.Float16
quant_dtype = cutlass.Int8  # [新增] 量化 B 的元素类型
acc_dtype = cutlass.Float32
use_2cta_instrs = True
cluster_shape_mnk = (2, 1, 1) if use_2cta_instrs else (1, 1, 1)
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
dequant_elements_per_thread = 128  # [新增] 反量化 kernel 每线程处理元素数
threads_in_dequant = 64  # [新增] 反量化 kernel 每 block 线程数
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
# 响应大小为 4B * 4 个元素
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
def dequantize(
    b_quantized: cute.Tensor,
    scale_factor: cutlass.Float32,
    b_dequantized: cute.Tensor,
):
    # [新增] 反量化 B 张量: GMEM 读量化数据, 乘 scale 写回 FP16 B, 供后续 GEMM 使用.
    tiler = (1, dequant_elements_per_thread)
    # (Tile, NumTiles)
    tBQgBQ = cute.zipped_divide(b_quantized, tiler)
    tBQrBQ = cute.make_rmem_tensor(tBQgBQ[None, 0].shape, quant_dtype)

    tBDQgBDQ = cute.zipped_divide(b_dequantized, tiler)
    tidx, _, _ = cute.arch.thread_idx()
    dim_per_blk, _, _ = cute.arch.block_dim()
    bidx, _, _ = cute.arch.block_idx()
    idx = tidx + bidx * dim_per_blk
    # [新增] 在首个 kernel 中插入 griddepcontrol_launch_dependents(), 提示运行时可提前启动后续 kernel.
    # 若后续 kernel 与之并发, 第二个 kernel 可尽早开始读取 (例如 prologue), 获得端到端加速.
    # launch_dependents 的位置不影响正确性, 仅影响性能.
    cute.autovec_copy(tBQgBQ[None, idx], tBQrBQ)
    cute.arch.griddepcontrol_launch_dependents()
    tBDQgBDQ[None, idx].store(
        (tBQrBQ.load() * scale_factor).to(b_dequantized.element_type)
    )


# [新增] 相对 fp16_gemm_3_1.py 的 ``kernel``: 重命名为 ``gemm``, 与 ``dequantize`` 组成 PDL 双-kernel 链路.
@cute.kernel()
def gemm(
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
    # sched_warp_id 仅用于 dynamic scheduler
    sched_warp_id = 6

    epilog_sync_bar_id = 1
    tmem_alloc_sync_bar_id = 2

    # 预取 TMA descriptor
    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_c)

    # 参与者数量等于同一行/列上发射 MMA 的线程数; 减 1 避免同一线程重复计数
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

    # 初始化 clc_pipeline (barrier) 与状态; 仅用于 CLC Dynamic Scheduler
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

    # 为 TMA 划分张量 (需先完成 MMA 划分)
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

    # 构造调度器
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
        # Persistent tile 调度循环
        #
        while work_tile.is_valid_tile:
            # 从 tile 调度器取 tile 坐标
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            # 切到每个 MMA tile 下标
            # ((atom_v, rest_v), RestK)
            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None)]

            # ((atom_v, rest_v), RestK)
            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None)]

            # [新增] GEMM 的第二操作数 gB 由前一 kernel 写入, 须调用 griddepcontrol_wait() 保证正确性.
            # 该指令会阻塞至前驱 kernel 结束且其内存操作对本 kernel 可见; 否则存在数据竞争与未定义行为.
            cute.arch.griddepcontrol_wait()

            # TMA load 循环
            for k_tile_idx in range(num_k_tiles):
                # 等待 A/B buffer 空闲再装入
                handle = ab_producer.acquire_and_advance()

                # 发射 TMA load
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

            # 前进到下一个 k_tile
            if cutlass.const_expr(use_clc_dynamic_scheduler):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # mbarrier_wait 防止 cluster 内一组相互依赖的 thread block (TMA/MMA 同步语义下) 过早退出,
        # 从而避免迟到的 tcgen05 commit_arrive 非法
        ab_producer.tail()

    # Sched warp (仅 dynamic scheduler)
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
                # 等待 accumulator buffer 为空
                acc_empty = acc_producer.acquire_and_advance()

                # 为当前 tile 设置 tensor memory buffer
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile_idx in range(num_k_tiles):
                    # 等待 TMA 拷贝完成
                    handle = ab_consumer.wait_and_advance()

                    # 执行一个 K-block 的 MMA 指令
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

                    # 通知 A/B buffer 已消费, 可装载下一轮
                    handle.release()

                # 通知 accumulator 已全部算完
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

        # 等待 accumulator buffer 排空
        acc_producer.tail()

    # Epilogue warps
    if warp_idx < mma_warp_id:
        # 分配 TMEM (实际只有 epilogue warp 0 执行分配)
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
            # 从 tile 调度器取 tile 坐标
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            # 等待 accumulator buffer 已满
            acc_full = acc_consumer.wait_and_advance()

            # 为当前 tile 设置 tensor memory buffer
            # (MMA, MMA_M, MMA_N)
            tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]

            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
            tCtAcc_epi = cute.flat_divide(
                tCtAcc[((None, None), 0, 0)],  # 为何取 0,0: 对应当前 partition 的单个 accumulator tile
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

            # Epilogue 子块循环
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

                # fence + barrier, 保证 SMEM 写入对后续 TMA store 可见
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

            # 异步 arrive: accumulator buffer 已空
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


# [新增] 同时计算 dequantize 的 grid 与 GEMM 的 grid / tile 调度参数 (相对 fp16_gemm_3_1.py 仅单个 GEMM kernel).
def compute_grid(
    b: cute.Tensor,
    c: cute.Tensor,
    mma_tiler_mnk: Tuple[int, int, int],
    cluster_shape_mnk: Tuple[int, int, int],
    scheduler_type: Union[
        utils.StaticPersistentTileScheduler, utils.ClcDynamicPersistentTileScheduler
    ],
    max_active_clusters: cutlass.Constexpr,
) -> Tuple[
    Tuple[int, int, int],
    Union[
        utils.ClcDynamicPersistentTileSchedulerParams,
        utils.PersistentTileSchedulerParams,
    ],
    Tuple[int, int, int],
]:
    b_size = cute.size(b.shape)
    b_threads = cute.ceil_div(b_size, dequant_elements_per_thread)
    dequant_blks = cute.ceil_div(b_threads, threads_in_dequant)
    dequantize_grid = (dequant_blks, 1, 1)

    c_shape = cute.slice_(mma_tiler_mnk, (None, None, 0))
    gc = cute.zipped_divide(c, tiler=c_shape)
    num_ctas_mn = gc[(0, (None, None))].shape

    if cutlass.const_expr(
        issubclass(scheduler_type, utils.ClcDynamicPersistentTileScheduler)
    ):
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (*num_ctas_mn, 1), cluster_shape_mnk
        )
        gemm_grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(
            tile_sched_params
        )
    else:
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (*num_ctas_mn, 1), cluster_shape_mnk
        )
        gemm_grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
    return dequantize_grid, tile_sched_params, gemm_grid


@cute.jit
def host_function(
    a: cute.Tensor,
    b_quantized: cute.Tensor,
    b_dequantized: cute.Tensor,
    c: cute.Tensor,
    scale_factor: cutlass.Float32,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    # [新增] 相对 fp16_gemm_3_1.py: 增加量化 B / 反量化缓冲与 scale, 拆成 dequantize + gemm 两次 launch, 并启用 use_pdl=True.
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
    # 构造 A、B 的 SMEM layout
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
        b_dequantized.element_type,
        ab_stages,
    )

    # c_smem_layout_kind 为行主序/列主序的枚举, 非 CuTe layout 对象
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
        b_dequantized,
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
    # 启动 kernel: 同一 stream 上先 dequantize 再 gemm, use_pdl=True 配合 PDL 指令
    #

    dequantize_grid, tile_sched_params, gemm_grid = compute_grid(
        b_dequantized,
        c,
        cta_tile_shape_mnk,
        cluster_shape_mnk,
        scheduler_type,
        max_active_clusters,
    )

    dequantize(
        b_quantized,
        scale_factor,
        b_dequantized,
    ).launch(
        grid=dequantize_grid,
        block=[threads_in_dequant, 1, 1],
        stream=stream,
        use_pdl=True,
    )

    gemm(
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
        grid=gemm_grid,
        block=[224, 1, 1] if use_clc_dynamic_scheduler else [192, 1, 1],
        cluster=cluster_shape_mnk,
        stream=stream,
        use_pdl=True,
    )


def run_dense_gemm(
    mnk: Tuple[int, int, int],
    tolerance: float,
):
    global torch, cutlass_torch
    import torch
    import cutlass.torch as cutlass_torch

    print("===================================================================")
    print("Running Blackwell fp16 GEMM example 6 with:")
    print(f"  mnk:       {mnk}")
    print(f"  tolerance: {tolerance}")
    print("===================================================================")
    print()

    m, n, k = mnk
    torch.manual_seed(1111)

    # 构造 K-major 张量 (PyTorch 默认行主序存储)
    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        return (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(device="cuda", dtype=dtype)
        )

    a = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
    b_quantized = make_tensors(n, k, cutlass_torch.dtype(quant_dtype))
    b_dequantized = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
    c = make_tensors(m, n, cutlass_torch.dtype(io_dtype))
    a_memref = from_dlpack(a).mark_layout_dynamic()

    b_quantized_memref = (
        from_dlpack(b_quantized, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    b_dequantized_memref = (
        from_dlpack(b_dequantized, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    c_memref = from_dlpack(c).mark_layout_dynamic()

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mnk[0] * cluster_shape_mnk[1]
    )
    # 随机反量化 scale
    scale_factor = 2.33

    # [新增] PDL 带来的收益在 kernel 本身较短时更明显; 此处用 CUDA Graph 封装以降低 host 侧 launch 开销.
    torch_stream = torch.cuda.Stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled_host_function = cute.compile(
        host_function,
        a_memref,
        b_quantized_memref,
        b_dequantized_memref,
        c_memref,
        scale_factor,
        max_active_clusters,
        current_stream,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch_stream):
        compiled_host_function(
            a_memref,
            b_quantized_memref,
            b_dequantized_memref,
            c_memref,
            scale_factor,
            current_stream,
        )
    graph.replay()

    # 参考结果与校验
    ref_b_dequantized = (b_quantized.to(dtype=torch.float32) * scale_factor).to(
        dtype=cutlass_torch.dtype(io_dtype)
    )
    ref = (torch.einsum("mk,nk->mn", a, ref_b_dequantized)).cpu()
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

    parser = argparse.ArgumentParser(description="Blackwell fp16 GEMM example 6")
    parser.add_argument(
        "--mnk",
        type=parse_comma_separated_ints,
        # 小规模、单 wave 场景更易体现 PDL 优化效果.
        default=(256, 8192, 128),
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
