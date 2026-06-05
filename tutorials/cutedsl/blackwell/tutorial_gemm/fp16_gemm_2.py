# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# 第三个 GEMM 教程. 在第二个教程基础上增加了 Warp Specialization (TMA / MMA / Epilogue warp).

import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

"""
第三个 GEMM 教程, 在 CuTeDSL 中演示简单 Dense GEMM kernel 实现.

相较 `fp16_gemm_1.py`, 本示例增加了 Warp Specialization (WS), 将 TMA、MMA、Epilogue 分配到不同 warp,
并在 epilogue 中使用 TMA Store 代替普通 copy 将结果从寄存器写回全局内存.


性能提升来源:

1. Warp Specialization 重叠数据搬运与计算

WS 的思路是让 CTA 内不同 warp 各司其职 (DMA / MMA / Epilogue), warp 之间通过 pipeline 通信.
好处在于任务级并行: DMA warp 完成当前 K-block 的加载后立即开始下一个 K-block,
同时 MMA warp 正在计算当前 K-block 的结果, 从而隐藏 DRAM 延迟.

非 WS 版本也能通过 prefetch 隐藏延迟, 但 WS 版本的指令级并行更好, 因为不同类型的指令在不同 warp 中发射.
例如非 WS 版本中 TMEM 分配和 TMA load 在同一 warp, TMA load 必须等 TMEM 分配完成; 
WS 版本中它们在不同 warp, 可以重叠执行.

2. 使用 TMA Store 写回结果

通过 TMA 将结果从寄存器写到全局内存实际需要两步:
  1) 寄存器 → 共享内存 (st.shared)
  2) 共享内存 → 全局内存 (TMA store)

这里继续使用 epilogue subtile: 一方面减少 epilogue 的 SMEM 用量,
另一方面可以将下一个 subtile 的 st.shared 与当前 subtile 的 TMA store 重叠, 隐藏 st.shared 延迟.

性能对比:
- 对于大 MMA tile, 如果 ab_stages 足够多, 非 WS 版本也能通过 prefetch 把 TMA load 与 MMA 重叠起来, 
  主循环性能 WS 与非 WS 接近, WS 的提升主要来自 prologue 和 epilogue, 因此 K 维较小时 WS 优势更明显.

- 对于小 MMA tile, 完成同样大小的 output tile 需要更多条 MMA 指令, 每条 MMA 指令前都有 ALU 准备工作,
  包括更新 pipeline 状态、计算 SMEM 偏移、barrier 操作等, 小 tile 意味着 ALU 开销占比更高。
  在非 WS 版本中, 同一个 warp 既要发射 MMA 又要执行这些 ALU 指令, ALU 占比高容易拖慢 MMA 的发射频率。
  WS 版本将它们分到不同 warp, MMA warp 的 ALU 压力更小, MMA 指令发射更高效.


运行本示例:
.. code-block:: bash
    python examples/blackwell/tutorial_gemm/fp16_gemm_2.py  \
      --mnk 8192,8192,8192

本示例约束:
* m 和 n 的问题规模必须能被 tile 尺寸 m & n (256, 256) 整除
"""

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
use_2cta_instrs = True
cluster_shape_mnk = (2, 1, 1) if use_2cta_instrs else (1, 1, 1)
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
threads_in_epilogue = 128  # 每个 CTA 中参与 epilogue 的线程数

# Pipeline stage 配置
ab_stages = 6
epi_stages = 2  # [新增] epilogue SMEM 双缓冲, 重叠 st.shared 与 TMA store
acc_stages = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32


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
):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

    mma_coord_vmnk = (
        bidx % cute.size(cta_layout_vmnk, mode=[0]),
        bidx // cute.size(cta_layout_vmnk, mode=[0]),
        bidy,
        None,
    )
    mma_coord_mnk = mma_coord_vmnk[1:]
    is_leader_cta = mma_coord_vmnk[0] == 0

    # [新增] Warp 角色分配: 6 个 warp, 按功能特化; gemm_1 中只用 warp 0 同时做 TMA load 和 MMA, 其余 warp 空闲
    # warp 0-3: epilogue warp (TMEM→RMEM→SMEM→GMEM)
    # warp 4:   MMA warp (发射 MMA 指令)
    # warp 5:   TMA warp (发射 TMA load)
    epilogue_warp_ids = (
        0,
        1,
        2,
        3,
    )
    mma_warp_id = 4
    tma_warp_id = 5

    #
    # 1. 准备参数
    #

    # 分配 SMEM
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
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
    # [新增] 为 epilogue TMA store 分配 C 的 SMEM (gemm_1 中 epilogue 直接 autovec_copy 到 GMEM, 不需要 sC)
    sC = smem.allocate_tensor(
        element_type=io_dtype,
        layout=epi_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=epi_smem_layout_staged.inner,
    )

    # [新增] Prefetch TMA 描述符在 TMA warp 中执行 (gemm_1 在 warp 0)
    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_c)  # [新增] C 的 TMA store 描述符

    # Multicast 参与者数 = 同行 + 同列 CTA 数 - 1 (避免重复计数)
    num_mcast_participants = (
        cute.size(cta_layout_vmnk, mode=[1]) + cute.size(cta_layout_vmnk, mode=[2]) - 1
    )

    # Multicast mask 初始化
    tma_mcast_mask_a = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
    )
    tma_mcast_mask_b = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
    )

    # 为 MMA 划分 tensor 并生成 fragments
    # (bM, bK, RestK)
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # (bN, bK, RestK)
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # (bM, bN)
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    thr_mma = tiled_mma.get_slice(mma_coord_vmnk[0])
    # (MMA, MMA_M, MMA_K, RestK)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K, RestK)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)

    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)

    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    # [新增] Barrier 1: epilogue 内 4 个 warp 的同步,确保 st.shared 对 TMA store 可见
    epilogue_sync_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_in_epilogue,
    )

    # [新增] TMEM 分配同步: MMA warp + epilogue warps 参与, TMA warp 不参与
    # gemm_1 中所有 128 个线程都参与 (因为没有 WS)
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=2,
        num_threads=32
        * len((mma_warp_id, *epilogue_warp_ids)),  # 5 个 warp = 160 个线程
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buffer,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilogue_warp_ids[0],  # [新增] 由 epilogue warp 0 执行分配 (gemm_1 中任意 warp 0)
        is_two_cta=True if use_2cta_instrs else False,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
    )

    # 为 TMA 划分 tensor; 需要已为 MMA 划分的 tensor
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )

    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    # [新增] 为 epilogue TMA store 划分 C tensor (gemm_1 中没有 TMA store, 直接 autovec_copy)
    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
    tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)

    tCsC, tCgC_tma = cute.nvgpu.cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )

    num_tma_copy_bytes = (
        cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
        + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ) * cute.size(cta_layout_vmnk, mode=[0])

    # 主循环 pipeline 的参与者
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

    # [新增] Accumulator pipeline 的 consumer 改为 epilogue warps (gemm_1 中 consumer 是全部线程)
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

    #
    # 2. 主循环
    #
    # [新增] gemm_1 中 warp 0 串行执行 TMA load → 等待 → MMA → 释放, 所有工作在同一 warp.
    # WS 版本将三类工作分到三组 warp, 各自独立循环, 通过 pipeline barrier 同步.
    num_k_tiles = cute.size(gA, mode=[2])

    # === TMA warp (warp 5): 负责 TMA load ===
    if warp_idx == tma_warp_id:
        for k_tile_idx in range(num_k_tiles):
            # 等待空缓冲区
            handle = ab_producer.acquire_and_advance()

            # 发起 TMA load
            cute.copy(
                tma_atom_a,
                tAgA[(None, k_tile_idx)],
                tAsA[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=tma_mcast_mask_a,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, k_tile_idx)],
                tBsB[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=tma_mcast_mask_b,
            )

        # 等待所有已使用的 buffer 被 consumer 释放, 防止 cluster 内 CTA 过早退出
        ab_producer.tail()

    # === MMA warp (warp 4): 负责发射 MMA 指令 ===
    elif warp_idx == mma_warp_id:
        # 等待 TMEM 分配完成并获取指针
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # 等待空的 accumulator 缓冲区
        if is_leader_cta:
            acc_empty = acc_producer.acquire_and_advance()

            for k_tile_idx in range(num_k_tiles):
                # 等待 TMA 数据就绪
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

                # 通知 A/B 缓冲区已消费完毕, 可进行下一轮 load
                handle.release()

            # 通知 accumulator 已计算完成
            acc_empty.commit()

    # === Epilogue warps (warp 0-3): 负责 TMEM→RMEM→SMEM→GMEM ===
    # [新增] gemm_1 中 epilogue 由所有线程执行 (无 WS), 直接 autovec_copy 到 GMEM.
    # WS 版本 epilogue 独立为 4 个 warp, 使用 TMA store 写回 GMEM.
    elif warp_idx < mma_warp_id:
        # 分配 TMEM (仅 epilogue warp 0 实际执行分配)
        num_tmem_cols = 512
        tmem.allocate(num_tmem_cols)

        # 等待 TMEM 分配完成并获取指针
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # [新增] 初始化 epilogue 的 TMA store pipeline
        epilogue_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=128,
        )
        epilogue_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=epi_stages,
            producer_group=epilogue_pipeline_producer_group,
        )

        # 等待 accumulator 缓冲区填满
        acc_consumer.wait_and_advance()

        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x8)
            if mma_tiler_mnk[0] == 64
            else tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            cutlass.Float32,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        tCtAcc_epi = cute.flat_divide(
            tCtAcc[((None, None), 0, 0)],
            epi_tile,
        )

        # TMEM → RMEM 的 tiled copy
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

        # [新增] RMEM → SMEM 的 copy (gemm_1 中直接 autovec_copy 到 GMEM, 不经过 SMEM)
        copy_atom_r2s = cutlass.utils.blackwell_helpers.get_smem_store_op(
            c_smem_layout_kind, cutlass.Float32, cutlass.Float32, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)

        tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
        tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, io_dtype)

        tCgC_grouped = cute.group_modes(tCgC_tma, 1, cute.rank(tCgC_tma))

        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

        # Epilogue subtile 循环
        # [新增] gemm_1 的 epilogue 流程: TMEM→RMEM (fp32) → 类型转换 → autovec_copy 到 GMEM
        # WS 版本: TMEM→RMEM (fp32) → 类型转换 → SMEM → TMA store 到 GMEM
        for subtile_idx in cutlass.range(subtile_cnt):
            # TMEM → RMEM
            tTR_tAcc_slice = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_slice, tTR_rAcc)

            # RMEM → SMEM
            c_buffer = subtile_idx % epi_stages
            tRS_sC_slice = tRS_sC[(None, None, None, c_buffer)]

            # 类型转换 fp32 → fp16
            tRS_rC.store(tRS_rAcc.load().to(io_dtype))

            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC_slice)

            # [新增] 内存栅栏 + barrier 确保 st.shared 对 TMA store 可见
            cute.arch.fence_view_async_shared()
            epilogue_sync_barrier.arrive_and_wait()
            # [新增] SMEM → GMEM (TMA store), 仅 epilogue warp 0 执行
            if warp_idx == epilogue_warp_ids[0]:
                cute.copy(
                    tma_atom_c,
                    tCsC[(None, c_buffer)],
                    tCgC_grouped[(None, subtile_idx)],
                )

                epilogue_pipeline.producer_commit()
                epilogue_pipeline.producer_acquire()
            epilogue_sync_barrier.arrive_and_wait()

        epilogue_pipeline.producer_tail()

        # 释放 TMEM
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
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
    # 为 A 和 B 构造 SMEM layout
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

    # c_smem_layout_kind 是行/列主序的枚举 (ROW_MAJOR / COL_MAJOR), 不是 CuTe layout.
    # 通过检查 c 的 leading_dim 判断, 传给 helper 函数生成 epilogue tile shape, SMEM layout 和 R2S copy atom.
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
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,  # take the layout and extract the shape internally

    )
    b_smem_layout_slice = cute.slice_(b_smem_layout, (None, None, None, 0))
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )

    # [新增] 计算 epilogue tile 形状, 构造 C 的 SMEM layout 和 TMA store atom
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

    c_tma_atom, c_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        c,
        epi_smem_layout,
        epi_tile,
    )

    #
    # 启动 kernel
    #

    grid_shape = cute.round_up(
        (
            cute.ceil_div(
                c.layout.shape[0], mma_tiler_mnk[0] // (2 if use_2cta_instrs else 1)
            ),
            cute.ceil_div(c.layout.shape[1], mma_tiler_mnk[1]),
            1,
        ),
        cluster_shape_mnk,
    )
    print(f"grid_shape: {grid_shape}")

    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c_tma_atom,
        c_tma_tensor,
        a_smem_layout,
        b_smem_layout,
        c_smem_layout_kind,
        epi_smem_layout_staged,
        epi_tile,
        cta_layout_vmnk,
    ).launch(
        grid=grid_shape,
        block=[192, 1, 1],
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
    print("Running Blackwell fp16 GEMM example 2 with:")
    print(f"  mnk:       {mnk}")
    print(f"  tolerance: {tolerance}")
    print("===================================================================")
    print()

    m, n, k = mnk
    torch.manual_seed(1111)

    # 构造 K-major tensor (torch tensor 为 row-major)
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

    # host JIT 函数入口
    host_function(
        a_memref,
        b_memref,
        c_memref,
        no_cache=True,
    )

    # 计算参考结果并验证
    ref = (torch.einsum("mk,nk->mn", a, b)).cpu()
    torch.testing.assert_close(
        c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=tolerance, rtol=1e-05
    )


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> list[int]:
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

    parser = argparse.ArgumentParser(description="Blackwell fp16 GEMM example 2")
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
