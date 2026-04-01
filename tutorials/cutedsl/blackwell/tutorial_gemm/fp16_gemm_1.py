# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# 这是第二个教程 GEMM. 在第一个教程基础上添加了 2CTA MMA 指令及 2x1 cluster.


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
第二个教程 GEMM, 相较 `fp16_gemm_0.py`, 本示例增加了 2CTA MMA 与 TMA multicast 支持.

当 `fp16_gemm_0.py` 在较高 SM 频率下运行时, dram 延迟会成为潜在性能瓶颈. 本示例的优化:


1. 2CTA MMA 减小 B tensor 的 SMEM, 从而支持更多 ab stages 以掩盖 dram 延迟.

2CTA (CtaGroup.TWO) 是 Blackwell tcgen05 MMA 指令的原生硬件模式.
同一 SM 上的两个 CTA 共享 TMEM, 由 leader CTA (编号 0) 发射 MMA 指令,
follower CTA 不发射 MMA 但参与数据搬运和同步 (代码中 is_leader_cta = mma_coord_vmnk[0] == 0).

两个 CTA 协作计算一个更大的 output tile C[256, 256], 沿 M 维各负责一半:
  CTA_0: C[0:128, 0:256]   需要 A[0:128, K]   + B[0:256, K]
  CTA_1: C[128:256, 0:256] 需要 A[128:256, K] + B[0:256, K]

对 A: 每个 CTA 只需自己那 128 行, SMEM 大小与 1CTA 相同 = 128x64x2B = 16KB.
对 B: 两个 CTA 都需要完整的 N=256 行 B, 但硬件将 B 对半分摊到两个 CTA 的 SMEM 中,
  各存 128 行, 通过 DSMEM (跨 CTA 共享内存) 互相读取对方的一半.
  因此每个 CTA 的 B SMEM = 128x64x2B = 16KB (1CTA 时为 256x64x2B = 32KB).

SMEM 节省带来更多 pipeline stages:
  1CTA: 每 stage 48KB (A 16KB + B 32KB), 最多 227 // 48 = 4 stages, 延迟掩盖 512 * 3 = 1.5K cycles
  2CTA: 每 stage 32KB (A 16KB + B 16KB), 最多 227 // 32 = 7 stages, 延迟掩盖 512 * 6 = 3K cycles


2. TMA multicast 减少 L2 流量.

TMA multicast 是 TMA 的广播模式: 一次 GMEM 读取, 硬件自动将数据写入 cluster 内多个 CTA 的 SMEM, 无需软件额外操作.

使用 TMA multicast 的前提是 Thread Block Cluster, 保证 cluster 内的多个 CTA 被调度到物理相邻的 SM 上, 
可通过 DSMEM 互访 SMEM、通过 TMA multicast 广播数据.
本示例的 cluster_shape_mnk = (2, 1, 1) 即 M 维 2 个 CTA 组成一个 cluster.

以 2x1 cluster 为例, 两个 CTA 沿 M 维排布:
  CTA_0 计算 C[0:128, 0:256],   需要 A[0:128, K]   + B[0:256, K]
  CTA_1 计算 C[128:256, 0:256], 需要 A[128:256, K] + B[0:256, K]
B 被两个 CTA 共用 → TMA 从 L2 读一次完整的 B tile, 将其中各一半分别写入两个 CTA 的 SMEM 中.
A 各不相同 → 无法 multicast, 各自从 L2 读取.

每 tile 的 L2 流量 = A 流量 / (N 维 cluster 数) + B 流量 / (M 维 cluster 数):
  无 multicast:  16KB + 32KB = 48KB
  2x1 cluster:   16KB / 1 + 32KB / 2 = 24KB
  4x4 cluster:   16KB / 4 + 32KB / 4 = 12KB

总结: 
2CTA MMA 是硬件通过 DSMEM 让两个 CTA 共享 B 数据。每个 CTA 只需存 B 的一半到 SMEM, MMA 指令自动跨 CTA 读取另一半。
而 TMA multicast 是一次从 L2 读取 B 数据，分发到两个 CTA 各自需要的那一半。因为 B 已经通过 2CTA 分半了， 实际没有进行multicast。
2CTA 提供更强的延迟掩盖能力, TMA multicast 缩短数据就绪时间. 在延迟/内存带宽受限场景下需综合考虑.


运行本示例:
.. code-block:: bash

    python examples/blackwell/tutorial_gemm/fp16_gemm_1.py  \
      --mnk 8192,8192,8192

本示例约束:
* m 和 n 的问题规模必须能被 tile 尺寸 m & n (256, 256) 整除
"""

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
# [优化] 使用 2x1 cluster: M 维度 2 个 CTA 协作, B tensor 可 multicast
cluster_shape_mnk = (2, 1, 1)
# [优化] MMA tile 从 (128,256,16) 扩大到 (256,256,16), 启用 2CTA 指令
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
threads_per_cta = 128

# [优化] 2CTA 使 B 的 SMEM 减半 (32KB->16KB), AB stage 从 4 增加到 7
# 延迟隐藏窗口: gemm_0 为 512*(4-1)=1.5K cycles, 此处为 512*(7-1)=3K cycles
ab_stages = 7
acc_stage = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    # [优化] 2CTA 共享 TMEM, 释放时需额外 barrier 同步
    tmem_dealloc_mbar_ptr: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


@cute.kernel()
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
):
    # 当前 thread/warp/block 坐标
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    # [优化] 新增 cluster 内坐标计算, gemm_0 中只有简单的 (bidx, bidy)
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
    # mma_coord_vmnk[0] 为 CTA 在 2CTA MMA 中的编号 (0=leader, 1=follower)
    mma_coord_vmnk = (
        bidx % cute.size(cta_layout_vmnk, mode=[0]),
        bidx // cute.size(cta_layout_vmnk, mode=[0]),
        bidy,
        None,
    )
    mma_coord_mnk = mma_coord_vmnk[1:]

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

    # Prefetch TMA descriptor
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # Pipeline 配置
    num_tma_copy_bytes = (
        cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
        + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ) * cute.size(cta_layout_vmnk, mode=[0])
    num_mcast_ctas_a = cute.size(cta_layout_vmnk.shape[2])
    num_mcast_ctas_b = cute.size(cta_layout_vmnk.shape[1])
    num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        ),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            cute.size(cta_layout_vmnk, mode=[0]) * threads_per_cta,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    # 为 MMA 划分tensor 并生成 fragments
    # (bM, bK, RestK)
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # (bN, bK, RestK)
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # (bM, bN)
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(mma_coord_vmnk[0])
    # (MMA, MMA_M, MMA_K)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)
    # (MMA, MMA_M, MMA_K)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    # [优化] TMA partition 使用 cluster 内坐标, A 沿 N 维 multicast (mode=[2]),
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    #  B 沿 M 维 multicast (mode=[1])
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    # [优化] TMA multicast mask: 位掩码, 每位对应 cluster 内一个 CTA,
    # 为 1 表示 TMA 将数据写入该 CTA 的 SMEM.
    # mcast_mode 指定沿 cta_layout_vmnk 的哪个维度广播:
    #   mcast_mode=2 (N 维) → A 的 multicast. 2x1 cluster 的 N 维只有 1 个 CTA, 实际无 multicast.
    #   mcast_mode=1 (M 维) → B 的 multicast. 2x1 cluster 的 M 维有 2 个 CTA, mask=0b11,
    #     TMA 从 L2 读一次 B 同时写入两个 CTA 的 SMEM.
    # 对比 gemm_0: 没有 cluster, TMA copy 不传 mcast_mask, 各 CTA 独立从 L2 读取.
    tma_mcast_mask_a = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
    )
    tma_mcast_mask_b = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
    )

    # 分配 TMEM 并交换 `tCtAcc` 中的指针
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    # [优化] 2CTA 模式下两个 CTA 共享 TMEM, 需传入 is_two_cta 和 dealloc barrier
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    # 在获取已分配 TMEM 起始指针前进行 CTA 内同步
    # 仅 warp 0 执行分配, 故需在获取 TMEM 起始地址前同步
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    # 交换 `tCtAcc` 中的指针
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    # (EpiTile)
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
    )
    # (EpiTile, NumTiles)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    # (EpiTile, NumTiles)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    # 每个 thread 加载 64 x fp32
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    # (TmemCpy, NumTmemCpy, NumTiles)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    # (TmemCpy, NumTmemCpy, NumTiles)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    # (TmemCpy, NumTmemCpy)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    # (TmemCpy, NumTmemCpy)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    #
    # 2. 主循环
    #
    # [优化] 2CTA 中仅 leader CTA (编号 0) 发起 MMA, 硬件自动协调两个 CTA 的 tensor core
    is_leader_cta = mma_coord_vmnk[0] == 0
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        # acc buffer 由共享 TMEM 持有, 仅 leader acquire 一次
        if is_leader_cta:
            acc_producer.acquire_and_advance()
        for _ in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
            # [优化] 发起 TMA loads, 传入 mcast_mask 启用 multicast
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=tma_mcast_mask_a,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, ab_empty.count)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=tma_mcast_mask_b,
            )

            # 执行一个 K-block 的 MMA 指令
            if is_leader_cta:
                ab_full = ab_consumer.wait_and_advance()
                # 执行一个 K-block 的 MMA 指令
                num_k_blocks = cute.size(tCrA, mode=[2])
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, ab_full.index)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_full.release()

        # 发出 accumulator 已计算完成的信号
        if is_leader_cta:
            acc_producer.commit()

    #
    # 3. Epilogue
    #

    # 释放 TMEM 分配锁
    tmem.relinquish_alloc_permit()

    # 等待 accumulator buffer 填满
    acc_full = acc_consumer.wait_and_advance()
    # TMEM -> RMEM -> GEMM
    # Sub-tiling 以获得更好的指令级并行
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    # [优化] 2CTA 新增: 确保 producer 退出前已用 buffer 正确同步
    # 防止 leader CTA 过早退出导致 follower CTA 访问已释放的 dsmem
    if warp_idx == 0:
        ab_producer.tail()
        if is_leader_cta:
            acc_producer.tail()

    # 释放 TMEM
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
):
    # [优化] 构造 tiled MMA, CtaGroup.TWO 启用 2CTA 协作 MMA (gemm_0 用 CtaGroup.ONE)
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.TWO,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    # 为 A 和 B 构造 SMEM layouts
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
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    # 构造 VMNK layout
    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

    # [优化] TMA 使用 Multicast op (gemm_0 用 CopyBulkTensorTileG2SOp)
    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,  # 取 layout 并在内部提取 shape
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )

    grid_shape = cute.round_up(
        cute.ceil_div(
            (*c.layout.shape, 1), (mma_tiler_mnk[0] // 2, *mma_tiler_mnk[1:])
        ),
        cluster_shape_mnk,
    )

    # 打印 kernel 属性, 便于调试
    # print(f"a               = {cute.pretty_str(a)}")
    # print(f"b               = {cute.pretty_str(b)}")
    # print(f"c               = {cute.pretty_str(c)}")
    # print(f"tiled_mma       = {cute.pretty_str(tiled_mma)}")
    # print(f"a_smem_layout   = {cute.pretty_str(a_smem_layout)}")
    # print(f"b_smem_layout   = {cute.pretty_str(b_smem_layout)}")
    # print(f"cta_layout_mnk  = {cute.pretty_str(cta_layout_mnk)}")
    # print(f"cta_layout_vmnk = {cute.pretty_str(cta_layout_vmnk)}")
    # print(f"a_tma_atom   = {cute.pretty_str(a_tma_atom)}")
    # print(f"b_tma_atom   = {cute.pretty_str(b_tma_atom)}")
    # print(f"a_tma_tensor    = {cute.pretty_str(a_tma_tensor)}")
    # print(f"b_tma_tensor    = {cute.pretty_str(b_tma_tensor)}")
    # cute.printf("grid_shape = {}", grid_shape)

    # [优化] 启动 kernel 时传入 cluster shape, gemm_0 没有 cluster 参数
    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
        cta_layout_vmnk,
    ).launch(
        grid=grid_shape,
        block=[threads_per_cta, 1, 1],
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
    print("Running Blackwell fp16 GEMM example 1 with:")
    print(f"  mnk:       {mnk}")
    print(f"  tolerance: {tolerance}")
    print("===================================================================")
    print()

    m, n, k = mnk
    torch.manual_seed(1111)

    # 构造 K-major tensor  (torch tensor 为 row-major)
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
    a_tensor = (
        from_dlpack(a, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    b_tensor = (
        from_dlpack(b, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    c_tensor = (
        from_dlpack(c, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=n)
    )

    # host JIT 函数的入口
    host_function(
        a_tensor,
        b_tensor,
        c_tensor,
        no_cache=True,
    )

    # 计算参考结果并验证
    ref = (torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32))).cpu()
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

    parser = argparse.ArgumentParser(description="Blackwell fp16 GEMM example 1")
    parser.add_argument(
        "--mnk",
        type=parse_comma_separated_ints,
        default=[8192, 8192, 8192],
        help="MNK dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    args = parser.parse_args()
    if len(args.mnk) != 3:
        parser.error("--mnk must contain exactly 3 values")
    if args.mnk[0] % mma_tiler_mnk[0] != 0 or args.mnk[1] % mma_tiler_mnk[1] != 0:
        parser.error("m n must be divisible by mma_tiler_mn")

    run_dense_gemm(
        args.mnk,
        args.tolerance,
    )
    print("PASS")
