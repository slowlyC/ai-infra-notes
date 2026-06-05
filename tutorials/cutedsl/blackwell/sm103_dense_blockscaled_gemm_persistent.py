# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
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
from typing import Optional, Type, Tuple, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm103_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack
from dataclasses import dataclass, field

"""
本示例提供 SM103 批处理 FP4 Ultra blockscaled GEMM 内核的实验性实现; 请注意, 与该内核相关的 API 及实现细节在后续版本中可能会发生变化。

面向 NVIDIA Blackwell SM103 架构、基于 CUTE DSL 的高性能 persistent 批处理 FP4 Ultra blockscaled GEMM 示例。
    - 矩阵 A 为 MxKxL, L 为 batch 维; 对 MXF4/NVF4 输入类型, A 仅可为行主序("K")
    - 矩阵 B 为 NxKxL, L 为 batch 维; 对 MXF4/NVF4 输入类型, B 仅可为行主序("K")
    - 矩阵 C 为 MxNxL, L 为 batch 维; C 可为行主序("N")或列主序("M")
    - 矩阵 SFA 的布局在内部根据 A 的形状与 sm103_BlockScaledBasicChunk 填充, 元素个数为 Mxceil_div(K, sf_vec_size)xL
    - 矩阵 SFB 的布局在内部根据 B 的形状与 sm103_BlockScaledBasicChunk 填充, 元素个数为 Nxceil_div(K, sf_vec_size)xL

该 GEMM 内核支持以下特性: 
    - 使用 Tensor Memory Access(TMA)进行高效访存
    - 使用 Blackwell 的 tcgen05.mma 执行矩阵乘累加(MMA)运算(含 2cta mma 指令)
    - 借助 cluster 实现 TMA multicast, 降低 L2 流量
    - 支持 persistent tile 调度, 在 tile 之间更好地重叠访存与 MMA
    - 支持 warp specialization: A/B 与 scale factor 使用独立的 TMA warp
    - 使用环形缓冲(circular buffer)技术以优化访存与计算重叠

该 GEMM 的执行流程如下: 
    1. TMA A/B warp: 通过 TMA 将 A、B 从全局内存(GMEM)加载到共享内存(SMEM)。
    2. TMA SF warp: 通过 TMA 将 scale factor A/B 从 GMEM 加载到 SMEM。
    3. MMA warp: 
        - 使用 tcgen05.cp 将 scale factor A/B 从 SMEM 拷贝到张量内存(TMEM)。
        - 使用 tcgen05.mma 执行 MMA, 以配合环形缓冲。
    4. Epilogue warp: 
        - 使用 tcgen05.ld 将已完成的累加器从 TMEM 加载到寄存器(RMEM)。
        - 将 C 矩阵类型转换为输出类型。
        - 将 C 从 RMEM 直接写回 GMEM(不使用 TMA store)。
        - 可选地接受逐元素 lambda `epilogue_op` 作用于输出张量: 
            例如 ReLU 可设 epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM103 上 tcgen05.mma.kind.block_scale 指令的行为如下: 
    - 从两个 SMEM 缓冲区(当前与下一个)读取矩阵 A
    - 从两个 SMEM 缓冲区(当前与下一个)读取矩阵 B
    - 从 TMEM 读取 scale factor A
    - 从 TMEM 读取 scale factor B
    - 将累加器写入 TMEM

随后必须将 TMEM 中的累加器加载到寄存器, 再写回 GMEM。

本示例的命令行参数如下所示: 

.. code-block:: bash

    python examples/blackwell/sm103_dense_blockscaled_gemm_persistent.py     \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16      \
      --c_dtype Float16                                                       \
      --mma_tiler_mn 256,256 --cluster_shape_mn 2,4                          \
      --mnkl 4096,4096,6144,1

使用 NCU profiler 采集性能: 

.. code-block:: bash

    ncu python examples/blackwell/sm103_dense_blockscaled_gemm_persistent.py \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16      \
      --c_dtype Float16                                                       \
      --mma_tiler_mn 256,256 --cluster_shape_mn 2,4                          \
      --mnkl 4096,4096,6144,1                                                \
      --warmup_iterations 1 --iterations 10 --skip_ref_check

约束: 
    - 支持的输入数据类型: mxf4、nvf4
        - 有效 dtype 组合的详细说明见下文 Sm103BlockScaledPersistentDenseGemmKernel 类文档
    - A/B 张量必须使用相同数据类型
    - MMA tiler 的 M 必须为 128 或 256(对应 use_2cta_instrs)
    - MMA tiler 的 N 必须为 128 或 256
    - Cluster 的 M/N 必须为正且为 2 的幂, cluster 总大小 ≤ 16
    - 若 MMA tiler 的 M 为 256(use_2cta_instrs), 则 cluster 的 M 必须为 2 的倍数
    - A/B/C 张量的连续维至少 16 字节对齐, 
        即对 MXF4/NVF4, 元素个数为 32 的倍数。
"""


class Sm103BlockScaledPersistentDenseGemmKernel:
    """实现批处理矩阵乘 C = A x SFA x B x SFB, 支持 FP4 数据类型, 
    以及 Blackwell SM103 GPU 上 persistent tile 调度与 warp specialization 等架构特性。

    :param sf_vec_size: Scale factor 向量长度。
    :type sf_vec_size: int
    :param mma_tiler_mn: 矩阵乘累加(MMA)tile 形状 (M, N)。
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: 并行处理用的 cluster 维度 (M, N)。
    :type cluster_shape_mn: Tuple[int, int]


    :note: 当前版本中, A 与 B 张量必须使用相同数据类型
        - 即不支持 A、B 使用不同元素类型(dtype 不一致)

    :note: 支持的 A/B 数据类型、SF 数据类型与 SF 向量长度组合: 
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU / Float8E4M3FN + sf_vec_size: 16

    :note: 支持的累加器数据类型: 
        - Float32

    :note: 支持的 C 数据类型: 
        - Float32
        - Float16 / BFloat16
        - Float8E4M3FN / Float8E5M2
    :note: 约束: 
        - MMA tiler 的 M 必须为 128 或 256(use_2cta_instrs)
        - MMA tiler 的 N 必须为 128 或 256
        - 若 MMA tiler 的 M 为 256, 则 cluster 的 M 必须为 2 的倍数
        - Cluster 的 M/N 必须为正且为 2 的幂, cluster 总大小 ≤ 16
        - 因 scale factor 容量有限, scale factor multicast 要求 cluster 的 M/N ≤ 4

    Example:
        >>> gemm = Sm103BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 256),
        ...     cluster_shape_mn=(2, 4)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
    ):
        """初始化 Blackwell SM103 FP4 Ultra GEMM 内核的配置。

        该配置主要包含: 

        1.  MMA 指令相关设置(tcgen05): 
            - acc_dtype: MMA 累加器数据类型, 恒为 Float32
            - sf_vec_size: Scale factor A/B 的向量长度
            - mma_tiler_mn: MMA 指令 tiler 的 (M, N) 形状

        2.  Cluster 形状: 
            - cluster_shape_mn: CTA cluster 的 (ClusterM, ClusterN) 形状

        :param sf_vec_size: Scale factor 向量长度。
        :type sf_vec_size: int
        :param mma_tiler_mn: MMA 指令的 (M, N) 形状元组。
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Cluster 的 (ClusterM, ClusterN) 形状元组。
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: 是否启用 TMA store。
        :type use_tma_store: bool
        """
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K 维在 _setup_attributes 中再确定
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # 设置专用 warp 编号
        self.epilogue_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_ab_warp_id = 5
        self.tma_sf_warp_id = 6
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_ab_warp_id,
                self.tma_sf_warp_id,
                *self.epilogue_warp_id,
            )
        )
        # 设置 epilogue 同步与 TMEM 指针同步用的 barrier id
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_103")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_103")
        self.sf_buffers_per_tile_k = 4 if self.sf_vec_size == 16 else 2

    def _setup_attributes(self):
        """根据运行时张量输入设置内核相关属性。

        本方法依据输入张量特性(数据类型、leading dimension 等)与内核设置, 配置: 
        - tiled MMA
        - MMA / cluster / tile 形状
        - cluster 布局
        - A/B/SFA/SFB 的 multicast CTA 数量
        - epilogue subtile
        - A/B/SFA/SFB/C 在 SMEM 中的 stage 数
        - A/B/SFA/SFB/C 的 SMEM 布局
        """
        # 计算 MMA 指令形状
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])

        # (CTA tile M, MMA tile N 向上取整到 128, MMA K 指令形状)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        dummy_tiled_mma_sfb = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # 计算 mma / cluster / tile 形状
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            768,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_layout_vmnk.shape[0]),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        blk_mn = 128
        self.cta_n_sf = cute.round_up(cute.size(self.cta_tile_shape_mnk[1]), blk_mn)
        self.mma_sf_tiler = (
            self.cta_tile_shape_mnk[0],
            self.cta_n_sf,
            self.cta_tile_shape_mnk[2] // self.sf_buffers_per_tile_k,
        )

        self.sf_atom = self.Sm103BlockScaledBasicChunk(
            self.sf_vec_size, tiled_mma.op.a_major_mode
        ).layout

        # 计算 cluster 布局
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (dummy_tiled_mma_sfb.thr_id.shape,),
        )

        # 计算 A/B 的 multicast CTA 数量
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # 计算 epilogue subtile
        self.epi_tile = sm103_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        self.num_acc_stage, self.num_ab_stage, self.num_sf_stage, self.num_c_stage = (
            self._compute_stages(
                tiled_mma,
                self.mma_tiler,
                self.epi_tile,
                self.c_dtype,
                self.c_layout,
                self.sf_dtype,
                self.sf_vec_size,
                self.smem_capacity,
                self.occupancy,
                self.use_tma_store,
            )
        )

        # 计算 A/B/SFA/SFB/C 的共享内存布局
        # ((CTA_MMA_M,16bytes),1,8,num_ab_stage)
        self.a_smem_layout_staged = self.sm103_make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.num_ab_stage,
        )

        # ((CTA_MMA_M,16bytes),1,8,3)
        self.a_smem_layout_staged_tma = self.sm103_make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            3,
        )

        # ((CTA_MMA_N,16bytes),1,8,num_ab_stage)
        self.b_smem_layout_staged = self.sm103_make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.num_ab_stage,
        )

        # ((CTA_MMA_N,16bytes),1,8,3)
        self.b_smem_layout_staged_tma = self.sm103_make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            3,
        )

        # (((8,4,4),(sf_vec_size,4)),1,3,num_sf_stage)
        self.sfa_smem_layout_staged = self.sm103_make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_sf_stage,
        )

        # (((32,4,2),(sf_vec_size,4)),1,3,num_sf_stage)
        self.sfb_smem_layout_staged = self.sm103_make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_sf_stage,
        )
        self.c_smem_layout_staged = None
        if self.use_tma_store:
            self.c_smem_layout_staged = sm103_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
            )

        # 当 cta_tile_n = 256 且 num_acc_stage == 1 时, 重叠并双缓冲累加器
        self.overlapping_accum = self.num_acc_stage == 1 and not self.use_tma_store
        self.epi_tile_n = cute.size(self.epi_tile[1])

        if self.overlapping_accum:
            # 由 scale factor 布局计算 SF 在 TMEM 中占用的列数。
            # 列数 = 将布局按 Int32 重解释后的 cosize & 0xFFFF, 
            # 与 find_tmem_tensor_col_offset 中的计算方式一致。
            def _sf_tmem_cols(make_tmem_layout_fn, smem_layout_staged):
                layout = make_tmem_layout_fn(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(smem_layout_staged, (None, None, None, 0)),
                )
                return (
                    cute.cosize(cute.recast_layout(32, self.sf_dtype.width, layout))
                    & 0xFFFF
                )

            self.num_sfa_tmem_cols = _sf_tmem_cols(
                blockscaled_utils.make_tmem_layout_sfa, self.sfa_smem_layout_staged
            )
            self.num_sfb_tmem_cols = _sf_tmem_cols(
                blockscaled_utils.make_tmem_layout_sfb, self.sfb_smem_layout_staged
            )
            self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
            # 在重叠累加模式下于 epilogue 中提前释放累加器缓冲
            self.iter_acc_early_release_in_epilogue = (
                self.num_sf_tmem_cols // self.epi_tile_n
            )

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """按步骤执行 GEMM: 
        - 在 SMEM / grid / TMA 计算之前设置静态属性
        - 设置 TMA load/store 的 atom 与张量
        - 结合硬件约束计算 grid 大小
        - 定义内核的共享存储布局
        - 同步启动内核

        :param a_tensor: 输入张量 A
        :type a_tensor: cute.Tensor
        :param b_tensor: 输入张量 B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor 张量 A(SFA)
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor 张量 B(SFB)
        :type sfb_tensor: cute.Tensor
        :param c_tensor: 输出张量 C
        :type c_tensor: cute.Tensor
        :param max_active_clusters: 最大活跃 cluster 数
        :type max_active_clusters: cutlass.Constexpr
        :param stream: 用于异步执行的 CUDA stream
        :type stream: cuda.CUstream
        :param epilogue_op: 可选的逐元素 lambda, 作用于输出张量
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: 当输入数据类型与 MMA 指令不兼容时抛出。
        """
        # 在 SMEM / grid / TMA 计算前设置静态属性
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)
        # 检查输入数据类型是否与 MMA 指令兼容
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # 设置依赖于 GEMM 输入的属性
        self._setup_attributes()

        # 将 A/B 张量按 scale factor atom 布局填充, 得到 sfa/sfb 张量视图
        sfa_layout = cute.tile_to_shape(self.sf_atom, a_tensor.shape, (2, 1, 3))
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        sfb_layout = cute.tile_to_shape(self.sf_atom, b_tensor.shape, (2, 1, 3))
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        dummy_tiled_mma_sfb = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # 配置 A 的 TMA load
        a_op = sm103_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        # 为 multicast 将布局 cast 为 uint8
        a_smem_layout_tma_ready = self.adapt_layout_for_tma_ab(
            self.a_smem_layout_staged_tma
        )
        a_tensor_uint8 = cute.recast_tensor(a_tensor, cutlass.Uint8)
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            a_op,
            a_tensor_uint8,
            a_smem_layout_tma_ready,
            # 384: 单次 MMA mainloop 迭代沿 K 维处理的 uint8 元素个数。
            (cute.size(tiled_mma.tv_layout_A[1][0]), 384),
            self.cluster_shape_mn[1],
            internal_type=cutlass.Uint8,
        )

        # 配置 B 的 TMA load
        b_op = sm103_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        # 为 multicast 将布局 cast 为 uint8
        b_smem_layout_tma_ready = self.adapt_layout_for_tma_ab(
            self.b_smem_layout_staged_tma
        )
        b_tensor_uint8 = cute.recast_tensor(b_tensor, cutlass.Uint8)
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            b_op,
            b_tensor_uint8,
            b_smem_layout_tma_ready,
            (cute.size(tiled_mma.tv_layout_B[1][0]), 384),
            self.cluster_shape_mn[0] // cute.size(tiled_mma.thr_id.shape),
            internal_type=cutlass.Uint8,
        )

        # 配置 SFA 的 TMA load
        sfa_op = sm103_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        sfa_smem_layout_tma_ready = self.adapt_layout_for_tma_sf(sfa_smem_layout)
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.cpasync.make_tiled_tma_atom(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout_tma_ready,
            (self.mma_sf_tiler[0], self.mma_sf_tiler[2]),
            self.cluster_shape_mn[1],
            internal_type=cutlass.Uint8,
        )

        # 配置 SFB 的 TMA load
        sfb_op = sm103_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        sfb_smem_layout_tma_ready = self.adapt_layout_for_tma_sf(sfb_smem_layout)
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.cpasync.make_tiled_tma_atom(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout_tma_ready,
            (self.mma_sf_tiler[1], self.mma_sf_tiler[2]),
            self.cluster_shape_mn[0] // cute.size(dummy_tiled_mma_sfb.thr_id),
            internal_type=cutlass.Uint8,
        )

        # 配置 C 的 TMA store
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c_tensor,
                epi_smem_layout,
                self.epi_tile,
            )

        a_copy_size = cute.size_in_bytes(
            cutlass.Uint8,
            cute.slice_(self.a_smem_layout_staged_tma, (None, None, None, 0)),
        )
        b_copy_size = cute.size_in_bytes(
            cutlass.Uint8,
            cute.slice_(self.b_smem_layout_staged_tma, (None, None, None, 0)),
        )
        sfa_copy_size = cute.size_in_bytes(
            cutlass.Uint8,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        sfb_copy_size = cute.size_in_bytes(
            cutlass.Uint8,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )
        self.num_tma_load_bytes_ab = (a_copy_size + b_copy_size) * atom_thr_size
        self.num_tma_load_bytes_sf = (sfa_copy_size + sfb_copy_size) * atom_thr_size

        # 计算 grid 大小
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # 定义内核的共享存储结构
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            sf_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_sf_stage]
            sf_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_sf_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # 同步启动内核
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c_tensor,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # GPU device 内核
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        执行 persistent 批处理 GEMM 计算的 GPU device 内核。
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # 预取 TMA 描述符
        #
        if warp_idx == self.tma_ab_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)
        if warp_idx == self.tma_sf_warp_id:
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # 设置 CTA / 线程坐标
        #
        # cluster 内坐标
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # CTA 内坐标
        tidx, _, _ = cute.arch.thread_idx()

        #
        # 分配并初始化: A+B full/empty、SFA+SFB full/empty、累加器 full/empty、TMEM 释放 barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # 初始化 mainloop 的 ab_producer 与 ab_consumer
        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.num_tma_load_bytes_ab,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # 初始化 mainloop 的 sf_producer 与 sf_consumer
        sf_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_sf_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        sf_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_sf_tma_producer
        )
        sf_producer, sf_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.sf_full_mbar_ptr.data_ptr(),
            num_stages=self.num_sf_stage,
            producer_group=sf_producer_group,
            consumer_group=sf_consumer_group,
            tx_count=self.num_tma_load_bytes_sf,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # 初始化 acc_pipeline(barrier)及状态
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
        # TMEM 释放用 barrier 初始化
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        # barrier 初始化后 cluster arrive
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # 设置 SMEM 中的 A/B/SFA/SFB/C 张量
        #
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # 计算 A/B/SFA/SFB buffer full 的 multicast mask
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # 对全局张量做 local_tile 划分
        #
        # (BLK_M, BLK_K, m, k, l)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_((self.mma_tiler[0], self.mma_tiler[1], 384), (None, 0, None)),
            (None, None, None),
        )
        # (BLK_N, BLK_K, n, k, l)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_((self.mma_tiler[0], self.mma_tiler[1], 384), (0, None, None)),
            (None, None, None),
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.mma_sf_tiler, (None, 0, None)),
            (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_sf_tiler, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # 为 TiledMMA 的 A/B/C 划分全局张量
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

        # 构造 tCgA_tmp
        tCgA_mkl_tmp = thr_mma.partition_A(gA_mkl)
        tCgA_layout = self.append_coalesce_layout(tCgA_mkl_tmp.layout)
        cta_tCgA = cute.make_tensor(tCgA_mkl_tmp.iterator, tCgA_layout)
        # ((CTA_MMA_M,256),Rest_MMA_M,Rest_MMA_K, m, k, l)
        tCgA = cute.make_tensor(
            cta_tCgA.iterator,
            cute.tiled_divide(
                cta_tCgA.layout, (cute.size(tiled_mma.tv_layout_A[1][0]), 128)
            ),
        )

        tCgB_nkl_tmp = thr_mma.partition_B(gB_nkl)
        tCgB_layout = self.append_coalesce_layout(tCgB_nkl_tmp.layout)
        cta_tCgB = cute.make_tensor(tCgB_nkl_tmp.iterator, tCgB_layout)
        # ((CTA_MMA_N,256),Rest_MMA_N, Rest_MMA_K, n, k, l)
        tCgB = cute.make_tensor(
            cta_tCgB.iterator,
            cute.tiled_divide(
                cta_tCgB.layout, (cute.size(tiled_mma.tv_layout_B[1][0]), 128)
            ),
        )

        tCgSFA = cute.make_tensor(
            gSFA_mkl.iterator,
            cute.tiled_divide(
                gSFA_mkl.layout, (self.mma_sf_tiler[0], self.mma_sf_tiler[2])
            ),
        )

        tCgSFB = cute.make_tensor(
            gSFB_nkl.iterator,
            cute.tiled_divide(
                gSFB_nkl.layout, (self.mma_sf_tiler[1], self.mma_sf_tiler[2])
            ),
        )
        tCgC = thr_mma.partition_C(gC_mnl)

        # 为 epilogue 谓词构造 C 的恒等张量
        idC = cute.make_identity_tensor(mC_mnl.shape)
        cC_mnl = cute.local_tile(
            idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCcC = thr_mma.partition_C(cC_mnl)

        #
        # 为 TMA load A/B 划分全局 / 共享张量
        #
        # TMA load A 的 partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 1),
        )
        # TMA load B 的 partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 1),
        )

        # scale factor A 的 TMA partition
        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA_compact = cute.filter_zeros(tAsSFA)

        # scale factor B 的 TMA partition
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB_compact = cute.filter_zeros(tBsSFB)

        #
        # 为 TiledMMA 的 A/B/C 划分 SMEM / TMEM 张量
        #
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (self.cta_tile_shape_mnk[1] - self.num_sf_tmem_cols)
                        * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        #
        # TMEM 分配前 cluster wait
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # 构造 tile 调度器
        #
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # 专用于 A/B 的 TMA load warp
        #
        if warp_idx == self.tma_ab_warp_id:
            #
            # AB 加载的 persistent tile 调度循环
            #
            buffers_per_k_tile = 3

            while work_tile.is_valid_tile:
                # 从 tile 调度器获取 tile 坐标
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # 切片到每个 MMA tile 索引
                #
                tAgA_slice = tAgA[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        None,
                        mma_tile_coord_mnl[2],
                    )
                ]
                tBgB_slice = tBgB[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[1],
                        None,
                        mma_tile_coord_mnl[2],
                    )
                ]

                # Peek(try_wait)AB buffer empty, 对应 k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = cutlass.Boolean(1)
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # A/B 张量的 TMA 加载循环
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # 加载 buffers_per_k_tile 个 buffer
                    for buffer in cutlass.range(buffers_per_k_tile, unroll_full=True):
                        # 获取下一个空的 AB buffer
                        ab_empty = ab_producer.acquire_and_advance(peek_ab_empty_status)

                        # TMA 加载 A/B
                        cute.copy(
                            tma_atom_a,
                            cute.group_modes(
                                tAgA_slice[(None, None, buffer, k_tile)], 0, 2
                            ),
                            tAsA[(None, ab_empty.index)],
                            tma_bar_ptr=ab_empty.barrier,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            cute.group_modes(
                                tBgB_slice[(None, None, buffer, k_tile)], 0, 2
                            ),
                            tBsB[(None, ab_empty.index)],
                            tma_bar_ptr=ab_empty.barrier,
                            mcast_mask=b_full_mcast_mask,
                        )

                        # Peek(try_wait)下一个 AB buffer empty
                        peek_ab_empty_status = cutlass.Boolean(1)
                        # 判断是否尚未到达最后一个 k_tile 的最后一个 buffer
                        if not (
                            (k_tile == k_tile_cnt - 1)
                            and (buffer == buffers_per_k_tile - 1)
                        ):
                            peek_ab_empty_status = ab_producer.try_acquire()

                # 前进到下一个 tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # 标记 AB 加载结束
            ab_producer.tail()

        #
        # 专用于 scale factor 张量的 TMA load warp
        #
        if warp_idx == self.tma_sf_warp_id:
            #
            # SF 加载的 persistent tile 调度循环
            #
            while work_tile.is_valid_tile:
                # 从 tile 调度器获取 tile 坐标
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0],
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # 切片到每个 MMA tile 索引
                #
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgSFB_slice = tBgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek(try_wait)SF buffer empty
                sf_producer.reset()
                peek_sf_empty_status = cutlass.Boolean(1)
                peek_sf_empty_status = sf_producer.try_acquire()

                #
                # scale factor 的 TMA 加载循环
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # 按 sf_buffers_per_tile_k 加载 SF 各 stage
                    for sf_stage in cutlass.range(
                        self.sf_buffers_per_tile_k, unroll_full=True
                    ):
                        # 获取下一个空的 SF buffer
                        sf_empty = sf_producer.acquire_and_advance(peek_sf_empty_status)

                        tAgSFA_compact = cute.filter_zeros(
                            tAgSFA_slice[
                                (None, k_tile * self.sf_buffers_per_tile_k + sf_stage)
                            ]
                        )
                        tBgSFB_compact = cute.filter_zeros(
                            tBgSFB_slice[
                                (None, k_tile * self.sf_buffers_per_tile_k + sf_stage)
                            ]
                        )

                        # 本 SF stage 的 TMA 加载 SFA/SFB
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_compact,
                            tAsSFA_compact[(None, sf_empty.index)],
                            tma_bar_ptr=sf_empty.barrier,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_compact,
                            tBsSFB_compact[(None, sf_empty.index)],
                            tma_bar_ptr=sf_empty.barrier,
                            mcast_mask=sfb_full_mcast_mask,
                        )

                        # Peek(try_wait)下一 SF buffer empty
                        peek_sf_empty_status = cutlass.Boolean(1)
                        # 判断是否尚未到达最后一个 k_tile 的最后一个 stage
                        if not (
                            k_tile == k_tile_cnt - 1
                            and sf_stage == self.sf_buffers_per_tile_k - 1
                        ):
                            peek_sf_empty_status = sf_producer.try_acquire()

                # 前进到下一个 tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # 标记 SF 加载结束
            sf_producer.tail()

        #
        # 专用 MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # barrier 同步, 从 SMEM 取回 TMEM 指针
            #
            tmem.wait_for_alloc()

            #
            # 取回 TMEM 指针并构造累加器 / SFA / SFB 张量
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # 构造累加器 TMEM 张量
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # 构造 SFA 的 TMEM 张量
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )

            MMA_M = self.cta_tile_shape_mnk[0]
            MMA_N_SF = self.cta_n_sf
            MMA_K_SF = self.cta_tile_shape_mnk[2] // 2
            mnBasicBlockShape = (32, 4)
            kBasicBlockShape_single = (self.sf_vec_size, 1)
            mma_iter_SFA_shape = (
                (mnBasicBlockShape, MMA_M // 128),
                kBasicBlockShape_single,
            )
            sSFA_iter_shape = (mma_iter_SFA_shape, 1, MMA_K_SF // self.sf_vec_size)
            sSFA_iter_layout = cute.make_layout(sSFA_iter_shape)
            mma_iter_SFB_shape = (
                (mnBasicBlockShape, MMA_N_SF // 128),
                kBasicBlockShape_single,
            )
            sSFB_iter_shape = (mma_iter_SFB_shape, 1, MMA_K_SF // self.sf_vec_size)
            sSFB_iter_layout = cute.make_layout(sSFB_iter_shape)

            tCtSFA_layout_mma = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, self.sf_vec_size, sSFA_iter_layout
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            tCtSFA_mma = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout_mma)

            # 构造 SFB 的 TMEM 张量
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB_layout_mma = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, self.sf_vec_size, sSFB_iter_layout
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            tCtSFB_mma = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout_mma)

            #
            # SFA/SFB 的 S2T 拷贝 partition
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # persistent tile 调度主循环
            #
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            MmasPerSfBuffer = 8 // self.sf_buffers_per_tile_k
            sf_stride = 6 if self.sf_vec_size == 16 else 3

            while work_tile.is_valid_tile:
                # 从 tile 调度器获取 tile 坐标
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # 获取累加器 stage 索引
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                # 为当前 tile 设置 TMEM buffer
                tCtAcc = tCtAcc_base[(None, 0, 0, acc_stage_index)]

                # Peek(try_wait)k_tile = 0 时 AB buffer full
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                # Peek(try_wait)SF buffer full
                sf_consumer.reset()
                peek_sf_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_sf_full_status = sf_consumer.try_wait()

                #
                # 每个 tile 重置 ACCUMULATE 域
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                is_first_iteration = True

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        # 视 sf_vec_size 条件加载 SFA/SFB, 供 MMA0/MMA1 使用
                        if 0 % MmasPerSfBuffer == 0:
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                sf_full.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            peek_sf_full_status = sf_consumer.try_wait()

                        # 等待 A/B 数据就绪(MMA0、MMA1、MMA2 的一部分)
                        ab_full0 = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # peek 下一阶段(MMA2、MMA3、MMA4、MMA5 的一部分)
                        peek_ab_full_status = cutlass.Boolean(1)
                        peek_ab_full_status = ab_consumer.try_wait()

                        # 将 acc acquire 延后到 ublock TMEM
                        if is_first_iteration:
                            acc_pipeline.producer_acquire(acc_producer_state)
                            is_first_iteration = False

                        # MMA0
                        k_block_coord_cur = (None, 0, 0, ab_full0.index)
                        k_block_coord_next = (None, 0, 0, ab_full0.index)
                        sf_kblock_coord = (None, None, 0 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # MMA1
                        k_block_coord_cur = (None, 0, 3, ab_full0.index)
                        k_block_coord_next = (None, 0, 0, ab_full0.index)
                        sf_kblock_coord = (None, None, 1 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # 条件加载 SFA/SFB, 供 MMA2/MMA3 使用
                        if 2 % MmasPerSfBuffer == 0:
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                sf_full.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            peek_sf_full_status = sf_consumer.try_wait()

                        # 等待 A/B 数据就绪(MMA2、MMA3、MMA4、MMA5 的一部分)
                        ab_full1 = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # peek 下一阶段(MMA5 的一部分、MMA6、MMA7)
                        peek_ab_full_status = cutlass.Boolean(1)
                        peek_ab_full_status = ab_consumer.try_wait()

                        # MMA2
                        k_block_coord_cur = (None, 0, 6, ab_full0.index)
                        k_block_coord_next = (None, 0, 0, ab_full1.index)
                        sf_kblock_coord = (None, None, 2 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # 释放已不再需要的 stage_ab_0
                        ab_full0.release()

                        # MMA3
                        k_block_coord_cur = (None, 0, 1, ab_full1.index)
                        k_block_coord_next = (None, 0, 0, ab_full1.index)
                        sf_kblock_coord = (None, None, 3 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # 条件加载 SFA/SFB, 供 MMA4/MMA5 使用
                        if 4 % MmasPerSfBuffer == 0:
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                sf_full.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            peek_sf_full_status = sf_consumer.try_wait()

                        # MMA4
                        k_block_coord_cur = (None, 0, 4, ab_full1.index)
                        k_block_coord_next = (None, 0, 0, ab_full1.index)
                        sf_kblock_coord = (None, None, 4 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # 等待 A/B 数据就绪(MMA5 的一部分、MMA6、MMA7)
                        ab_full2 = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # peek 下一轮首阶段(MMA0、MMA1、MMA2 的一部分)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if k_tile + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                        # MMA5
                        k_block_coord_cur = (None, 0, 7, ab_full1.index)
                        k_block_coord_next = (None, 0, 0, ab_full2.index)
                        sf_kblock_coord = (None, None, 5 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # 条件加载 SFA/SFB, 供 MMA6/MMA7 使用
                        if 6 % MmasPerSfBuffer == 0:
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                sf_full.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            if k_tile + 1 < k_tile_cnt:
                                peek_sf_full_status = sf_consumer.try_wait()

                        ab_full1.release()

                        # MMA6
                        k_block_coord_cur = (None, 0, 2, ab_full2.index)
                        k_block_coord_next = (None, 0, 0, ab_full2.index)
                        sf_kblock_coord = (None, None, 6 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # MMA7
                        k_block_coord_cur = (None, 0, 5, ab_full2.index)
                        k_block_coord_next = (None, 0, 0, ab_full2.index)
                        sf_kblock_coord = (None, None, 7 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        ab_full2.release()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # 前进到下一个 tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 等待累加器 buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        #
        # 专用 epilogue warp
        #
        if warp_idx < self.mma_warp_id:
            #
            # 分配 TMEM buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # barrier 同步, 从 SMEM 取回 TMEM 指针
            #
            tmem.wait_for_alloc()

            #
            # 取回 TMEM 指针并构造累加器张量
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # persistent tile 调度主循环(epilogue)
            #
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            if cutlass.const_expr(self.use_tma_store):
                assert tma_atom_c is not None and sC is not None
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilogue_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage, producer_group=c_producer_group
                )
            while work_tile.is_valid_tile:
                # 从 tile 调度器获取 tile 坐标
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                #
                # 预先前进到下一个 tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_tiles_executed = tile_sched.num_tiles_executed
                if cutlass.const_expr(self.use_tma_store):
                    acc_consumer_state = utils.gemm.sm100.epilogue_tma_store(
                        self,
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc_base,
                        sC,
                        tCgC,
                        epi_tile,
                        num_tiles_executed,
                        epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                    )
                else:
                    acc_consumer_state = utils.gemm.sm100.epilogue(
                        self,
                        tidx,
                        tCtAcc_base,
                        tCgC,
                        epi_tile,
                        epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        tCcC_base=tCcC,
                        mC_mnl=mC_mnl,
                        overlapping_accum=self.overlapping_accum,
                    )

            if cutlass.const_expr(self.use_tma_store):
                # 等待 C 写回完成
                c_pipeline.producer_tail()
            else:
                # TMEM 释放前同步(由调用方完成后续步骤)
                tmem_dealloc_barrier.arrive_and_wait()

            #
            # 释放 TMEM buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)

    @staticmethod
    def make_desc_and_call_mma(
        tiled_mma: cute.TiledMma,
        d: cute.Tensor,
        sA_cur: cute.Tensor,
        sA_next: cute.Tensor,
        sB_cur: cute.Tensor,
        sB_next: cute.Tensor,
        c: cute.Tensor,
    ) -> None:
        """针对 SMEM 中环形缓冲 A/B 的专用 GEMM。

        计算 D <- A * B + C, 其中 A、B 由 (current, next) 缓冲构造的环形 SMEM 描述符表示。C 与 D 可别名。

        部分 tcgen05 MMA 需要在本函数外显式切换 accumulate 域; 由调用方负责。

        所有张量须已按给定的 tiled MMA 完成 partition。

        对需要单线程执行的 MMA Atom, gemm 运算会在内部自动处理线程推选; 此类情形无需手动选线程。

        :param tiled_mma: 用于执行 GEMM 的 tiled MMA 对象。
        :type tiled_mma: cute.TiledMma
        :param d: 目标张量 D
        :type d: cute.Tensor
        :param sA_cur: 操作数 A 的当前 SMEM 张量
        :type sA_cur: cute.Tensor
        :param sA_next: 操作数 A 的下一 SMEM 张量, 用于环形缓冲
        :type sA_next: cute.Tensor
        :param sB_cur: 操作数 B 的当前 SMEM 张量
        :type sB_cur: cute.Tensor
        :param sB_next: 操作数 B 的下一 SMEM 张量, 用于环形缓冲
        :type sB_next: cute.Tensor
        :param c: 第三源操作数张量
        :type c: cute.Tensor
        :return: None
        :rtype: None
        """
        a_desc = tcgen05.make_umma_smem_desc(
            sA_cur.iterator,
            sA_cur.layout,
            "k" if tiled_mma.op.a_major_mode.name == "K" else "mn",
            next_src=sA_next.iterator,
        )
        b_desc = tcgen05.make_umma_smem_desc(
            sB_cur.iterator,
            sB_cur.layout,
            "k" if tiled_mma.op.b_major_mode.name == "K" else "mn",
            next_src=sB_next.iterator,
        )

        view_layout = cute.make_layout(1, stride=0)
        a_tensor = cute.make_tensor(a_desc, view_layout)
        b_tensor = cute.make_tensor(b_desc, view_layout)
        return cute.mma_atom_call(tiled_mma, d, a_tensor, b_tensor, c)

    @staticmethod
    def sm103_make_blockscaled_trivial_tiled_mma(
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        cta_group: tcgen05.CtaGroup,
        mma_tiler_mn: Tuple[int, int],
        a_source: tcgen05.OperandSource = tcgen05.OperandSource.SMEM,
    ) -> cute.TiledMma:
        """创建 SM103(FP4 Ultra)的 blockscaled trivial tiled MMA, K 固定为 96。

        返回按给定 (M, N) tiler 与 CTA group 配置好的 tcgen05 MMA。

        :param sf_dtype: Scale factor 数据类型(通常为 8 位)
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: Scale factor 向量长度
        :type sf_vec_size: int
        :param cta_group: CTA group 配置
        :type cta_group: tcgen05.CtaGroup
        :param mma_tiler_mn: MMA tiler 维度 (M, N)
        :type mma_tiler_mn: Tuple[int, int]
        :param a_source: 操作数 A 的数据来源(默认 SMEM)
        :type a_source: tcgen05.OperandSource

        :return: 面向 SM103 blockscaled 运算配置好的 tiled MMA
        :rtype: cute.TiledMma

        :raises TypeError: 数据类型不受支持时抛出。
        :raises ValueError: sf_vec_size 不受支持时抛出。
        """
        if sf_vec_size == 32:
            mma_op = tcgen05.SM103MmaMXF4Op(
                (*mma_tiler_mn, 96),
                cta_group,
                a_source,
            )
        elif sf_vec_size == 16:
            mma_op = tcgen05.SM103MmaMXF4NVF4Op(
                sf_dtype,
                (*mma_tiler_mn, 96),
                cta_group,
                a_source,
            )
        else:
            raise ValueError(
                f"Unsupported sf_vec_size: {sf_vec_size}. Expected 16 or 32."
            )
        return cute.make_tiled_mma(cute.make_mma_atom(mma_op))

    # 工具函数
    @staticmethod
    def sm103_make_smem_layout_a(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: cute.Tile,
        num_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        """
        使用 K_SW128 与 Uint8 为操作数 A 构造 SMEM 布局。

        通过 make_smem_layout_atom(K_SW128、Uint8)生成 A 的 SMEM 布局。

        :param tiled_mma: 已配置的 tiled MMA 对象
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: MMA tiler 形状 (M, N, K)
        :type mma_tiler_mnk: cute.Tile
        :param num_stages: stage 数量
        :type num_stages: int

        :return: 操作数 A 的 SMEM 布局
        :rtype: cute.Layout
        """
        is_k_major = tiled_mma.op.a_major_mode == tcgen05.OperandMajorMode.K
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            tcgen05.make_smem_layout_atom(
                tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Uint8
            ),
            cute.append(
                (
                    (
                        mma_tiler_mnk[0]
                        // cute.size(tiled_mma.thr_layout_vmnk.shape[0]),
                        16,
                    ),
                    1,
                    8,
                ),
                num_stages,
            ),
            order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        )

        return a_smem_layout_staged

    @staticmethod
    def sm103_make_smem_layout_b(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: cute.Tile,
        num_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        """
        使用 K_SW128 与 Uint8 为操作数 B 构造 SMEM 布局。

        通过 make_smem_layout_atom(K_SW128、Uint8)生成 B 的 SMEM 布局。

        :param tiled_mma: 已配置的 tiled MMA 对象
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: MMA tiler 形状 (M, N, K)
        :type mma_tiler_mnk: cute.Tile
        :param num_stages: stage 数量
        :type num_stages: int

        :return: 操作数 B 的 SMEM 布局
        :rtype: cute.Layout
        """
        is_k_major = tiled_mma.op.b_major_mode == tcgen05.OperandMajorMode.K
        b_smem_layout_staged = tcgen05.tile_to_mma_shape(
            tcgen05.make_smem_layout_atom(
                tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Uint8
            ),
            cute.append(
                ((mma_tiler_mnk[1] // cute.size(tiled_mma.thr_id.shape), 16), 1, 8),
                num_stages,
            ),
            order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        )
        return b_smem_layout_staged

    @dataclass(frozen=True)
    class Sm103BlockScaledBasicChunk:
        """
        由 SM103 上 tcgen05 BlockScaled MMA 指令决定的 scale factor 基本 atom 布局。

        表示 tcgen05 BlockScaled MMA 在 SM103 上使用的固定 scale factor 排布模式; 
        布局由指令规约确定, 不可配置。
        """

        sf_vec_size: int
        major_mode: tcgen05.OperandMajorMode = tcgen05.OperandMajorMode.K
        _layout: cute.Layout = field(init=False, repr=False)

        def __post_init__(self) -> None:
            if self.major_mode == tcgen05.OperandMajorMode.K:
                atom_shape = ((8, 4, 4), (self.sf_vec_size, 4))
                atom_stride = ((16, 128, 4), (0, 1))
            else:
                atom_shape = ((self.sf_vec_size, 4), (8, 4, 4))
                atom_stride = ((0, 1), (16, 128, 4))

            object.__setattr__(
                self, "_layout", cute.make_layout(shape=atom_shape, stride=atom_stride)
            )

        @property
        def layout(self) -> cute.Layout:
            return self._layout

    @staticmethod
    def sm103_make_smem_layout_sfa(
        tiled_mma: cute.TiledMma,
        mma_tiler: cute.Tile,
        sf_vec_size: int,
        num_stages: int,
    ) -> cute.Layout:
        """
        基于以下项构造 SFA 的 SMEM 布局: 
        1) Sm103BlockScaledBasicChunk, 2) MMA tiler, 3) sf_vec_size, 4) stages。

        :param tiled_mma: tiled MMA
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: MMA tiler 形状
        :type mma_tiler: cute.Tile
        :param sf_vec_size: scale factor 向量长度
        :type sf_vec_size: int
        :param num_stages: stage 数量
        :type num_stages: int

        :return: SFA 的 SMEM 布局
        :rtype: cute.Layout
        """
        mma_shape_mk = tiled_mma.partition_shape_A((mma_tiler[0], mma_tiler[2]))
        sf_atom = Sm103BlockScaledPersistentDenseGemmKernel.Sm103BlockScaledBasicChunk(
            sf_vec_size, tiled_mma.op.a_major_mode
        ).layout
        k_divisor = 4 if sf_vec_size == 16 else 2
        mma_sfa_tiler = (
            mma_shape_mk[0][0] * mma_shape_mk[1],
            mma_shape_mk[0][1] * mma_shape_mk[2] // k_divisor,
        )
        sfa_smem_atom_layout = cute.tiled_product(
            sf_atom,
            cute.make_layout(
                cute.shape_div(mma_sfa_tiler, cute.product_each(sf_atom.shape))
            ),
        )
        sfa_smem_layout_staged = cute.make_layout(
            shape=cute.append(sfa_smem_atom_layout.shape, num_stages),
            stride=cute.append(
                sfa_smem_atom_layout.stride,
                cute.size(cute.filter_zeros(sfa_smem_atom_layout)),
            ),
        )
        return sfa_smem_layout_staged

    @staticmethod
    def sm103_make_smem_layout_sfb(
        tiled_mma: cute.TiledMma,
        mma_tiler: cute.Tile,
        sf_vec_size: int,
        num_stages: int,
    ) -> cute.Layout:
        """
        基于 basic chunk、MMA tiler、sf_vec_size、stages 构造 SFB 的 SMEM 布局。

        :param tiled_mma: tiled MMA
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: MMA tiler 形状
        :type mma_tiler: cute.Tile
        :param sf_vec_size: scale factor 向量长度
        :type sf_vec_size: int
        :param num_stages: stage 数量
        :type num_stages: int

        :return: SFB 的 SMEM 布局
        :rtype: cute.Layout
        """
        sf_atom = Sm103BlockScaledPersistentDenseGemmKernel.Sm103BlockScaledBasicChunk(
            sf_vec_size, tiled_mma.op.a_major_mode
        ).layout
        k_divisor = 4 if sf_vec_size == 16 else 2
        mma_sfb_tiler = (mma_tiler[1], mma_tiler[2] // k_divisor)
        if mma_sfb_tiler[0] == 128:
            sfb_smem_atom_layout = cute.tiled_product(
                sf_atom,
                cute.make_layout(
                    cute.shape_div(mma_sfb_tiler, cute.product_each(sf_atom.shape))
                ),
            )
        else:
            sf_k_major_atom256 = cute.make_layout(
                shape=(
                    (32, 4, 2),
                    (sf_vec_size, 4),
                ),
                stride=(
                    (16, 4, mma_sfb_tiler[1] // sf_vec_size // 4 * 512),
                    (0, 1),
                ),
            )
            sfb_smem_atom_layout = cute.tiled_product(
                sf_k_major_atom256,
                cute.make_layout(
                    cute.shape_div(
                        mma_sfb_tiler, cute.product_each(sf_k_major_atom256.shape)
                    )
                ),
            )

        sfb_smem_layout_staged = cute.make_layout(
            shape=cute.append(sfb_smem_atom_layout.shape, num_stages),
            stride=cute.append(
                sfb_smem_atom_layout.stride,
                cute.size(cute.filter_zeros(sfb_smem_atom_layout)),
            ),
        )
        return sfb_smem_layout_staged

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        构造 scale factor 的 SMEM→TMEM 加载用 tiledCopy, 并据此划分 SMEM(源)与 TMEM(目的)张量。

        :param sSF: SMEM 中的 scale factor 张量
        :type sSF: cute.Tensor
        :param tSF: TMEM 中的 scale factor 张量
        :type tSF: cute.Tensor

        :return: 元组 (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t), 其中: 
            - tiled_copy_s2t: scale factor 的 SMEM→TMEM(s2t)tiled 拷贝
            - tCsSF_compact_s2t: 划分后的 SMEM 侧 compact 张量
            - tCtSF_compact_s2t: 划分后的 TMEM 侧 compact 张量
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)
        tCtSF_compact_copy = cute.make_tensor(
            tCtSF_compact.iterator,
            cute.append(
                cute.append(tCtSF_compact[(None, 0, 0)].layout, cute.make_layout((1))),
                cute.make_layout(1),
            ),
        )
        # 构造 S2T CopyAtom 与 tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact_copy)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        use_tma_store: bool,
    ) -> Tuple[int, int, int]:
        """根据启发式计算 A/B 与 SF 操作数的 stage 数量。

        SM103 上 AB 与 SF pipeline 使用相互独立的 stage 计数。

        :param tiled_mma: 定义核心运算的 tiled MMA 对象
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: MMA tiler 形状 (M, N, K)
        :type mma_tiler: tuple[int, int, int]
        :param epi_tile: epilogue tile 形状
        :type epi_tile: cute.Tile
        :param c_dtype: 操作数 C(输出)的数据类型
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: 操作数 C 的布局枚举
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: scale factor 数据类型
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: scale factor 向量长度
        :type sf_vec_size: int
        :param smem_capacity: 可用 SMEM 总容量(字节)
        :type smem_capacity: int
        :param occupancy: 每个 SM 的目标 CTA 数(occupancy)
        :type occupancy: int
        :param use_tma_store: 是否启用 TMA store
        :type use_tma_store: bool

        :return: 计算得到的 stage 数量元组: (ACC stage 数、A/B stage 数、SF stage 数、C/epilogue stage 数)
        :rtype: tuple[int, int, int, int]
        """
        # ACC stage 数: 与 SM100 dense blockscaled gemm 相同
        num_acc_stage = 1 if mma_tiler[1] == 256 else 2

        # 默认 C(epilogue)stage 数
        num_c_stage = 2 if use_tma_store else 0

        # 计算 A、B、SFA、SFB 单 stage 的 SMEM 布局与大小
        a_smem_layout_stage_one = (
            Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_a(
                tiled_mma,
                mma_tiler,
                1,
            )
        )
        b_smem_layout_staged_one = (
            Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_b(
                tiled_mma,
                mma_tiler,
                1,
            )
        )
        sfa_smem_layout_staged_one = (
            Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_sfa(
                tiled_mma,
                mma_tiler,
                sf_vec_size,
                1,
            )
        )
        sfb_smem_layout_staged_one = (
            Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_sfb(
                tiled_mma,
                mma_tiler,
                sf_vec_size,
                1,
            )
        )

        c_smem_layout_staged_one = sm103_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        ab_bytes_per_stage = cute.size_in_bytes(
            cutlass.Uint8, a_smem_layout_stage_one
        ) + cute.size_in_bytes(cutlass.Uint8, b_smem_layout_staged_one)
        sf_bytes_per_stage = cute.size_in_bytes(
            sf_dtype, sfa_smem_layout_staged_one
        ) + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)

        mbar_helpers_bytes = 1024

        num_ab_stage = (
            smem_capacity // occupancy
            - (mbar_helpers_bytes + sf_bytes_per_stage + c_bytes)
        ) // ab_bytes_per_stage

        num_sf_stage = (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * mbar_helpers_bytes
            - occupancy * c_bytes
        ) // (occupancy * sf_bytes_per_stage)

        # 细化 epilogue(C)stage 数: 
        # 在为 A/B stage 与保留字节分配后计算剩余 SMEM
        # 将剩余未用 SMEM 划归 epilogue
        if use_tma_store:
            # xinyu TODO: 是否与 C++ 侧一致待确认
            num_c_stage += (
                smem_capacity
                - occupancy * ab_bytes_per_stage * num_ab_stage
                - occupancy * sf_bytes_per_stage * num_sf_stage
                - occupancy * mbar_helpers_bytes
                - occupancy * c_bytes
            ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_sf_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """使用 persistent tile 调度器计算输出张量 C 对应的 grid 大小。

        :param c: 输出张量 C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: CTA tile 形状 (M, N, K)
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: cluster 在 M、N 维上的形状
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: 最大活跃 cluster 数
        :type max_active_clusters: cutlass.Constexpr

        :return: 元组, 包含: 
            - tile_sched_params: persistent tile 调度器参数
            - grid: 内核启动用的 grid 形状
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        检查 dtype 与 sf_vec_size 是否为受支持的组合。

        :param ab_dtype: 操作数 A、B 的数据类型
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: scale factor 数据类型
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: scale factor 向量长度
        :type sf_vec_size: int
        :param c_dtype: 输出张量 C 的数据类型
        :type c_dtype: Type[cutlass.Numeric]

        :return: 合法为 True, 否则 False
        :rtype: bool
        """
        is_valid = True

        # 校验 ab_dtype
        if ab_dtype != cutlass.Float4E2M1FN:
            is_valid = False

        # 校验 sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # 校验 sf_dtype 是否合法
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # 校验 sf_dtype 与 sf_vec_size 的组合
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False

        # 校验 c_dtype 是否合法
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        检查布局与 dtype 的组合是否合法。

        :param ab_dtype: 操作数 A、B 的数据类型
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: 输出张量 C 的数据类型
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: 张量 A 的主维(连续维)
        :type a_major: str
        :param b_major: 张量 B 的主维(连续维)
        :type b_major: str
        :param c_major: 张量 C 的主维(连续维)
        :type c_major: str

        :return: 布局合法为 True, 否则为 False
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        检查 MMA tiler 与 cluster 形状是否合法。

        :param mma_tiler_mn: MMA 指令 tiler 的 (M, N) 形状
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: CTA cluster 的 (ClusterM, ClusterN) 形状
        :type cluster_shape_mn: Tuple[int, int]

        :return: MMA tiler 与 cluster 形状均合法为 True, 否则为 False
        :rtype: bool
        """
        is_valid = True
        # 跳过无效的 MMA tile 形状
        if not mma_tiler_mn[0] in [128, 256]:
            is_valid = False
        if not mma_tiler_mn[1] in [128, 256]:
            is_valid = False
        # 跳过非法的 cluster 形状(与 2CTA 指令约束相关)
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # 跳过无效的 cluster 形状
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # scale factor multicast 的 cluster 形状特殊校验。
            # 因 scale factor 容量有限, 无法在超过 4 个 CTA 之间做 multicast。
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        检查张量是否满足所需的字节对齐。

        :param m: 张量 A 的行数
        :type m: int
        :param n: 张量 B 的列数
        :type n: int
        :param k: 张量 A 的列数(K 维长度)
        :type k: int
        :param l: batch 维 L 的大小
        :type l: int
        :param ab_dtype: 操作数 A、B 的数据类型
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: 输出张量 C 的数据类型
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: 张量 A 的主轴(连续维)
        :type a_major: str
        :param b_major: 张量 B 的主轴(连续维)
        :type b_major: str
        :param c_major: 张量 C 的主轴(连续维)
        :type c_major: str

        :return: 问题规模与对齐均合法为 True, 否则为 False
        :rtype: bool
        """
        is_valid = True

        def check_contigous_alignment(
            dtype, is_mode0_major, tensor_shape, alignment_bytes
        ):
            """检查张量是否满足所需的字节对齐。

            :param dtype: 张量的数据类型
            :param is_mode0_major: mode 0 是否为主(连续)维
            :param tensor_shape: 张量形状 (mode0, mode1, batch)
            :param alignment_bytes: 所需对齐字节数(例如 16 或 32)
            :return: 满足对齐要求为 True
            """
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            # 计算达到对齐所需的连续元素个数: 
            # alignment_bytes * 8(每字节比特数)/ dtype.width(每元素比特数)
            num_contiguous_elements = alignment_bytes * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        # 检查 A/B 张量是否满足 16 字节对齐
        # 检查 C 张量是否满足 32 字节对齐
        if (
            not check_contigous_alignment(ab_dtype, a_major == "m", (m, k, l), 16)
            or not check_contigous_alignment(ab_dtype, b_major == "n", (n, k, l), 16)
            or not check_contigous_alignment(c_dtype, c_major == "m", (m, n, l), 32)
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        use_tma_store: bool,
    ) -> bool:
        """
        判断当前配置下 GEMM 是否可由本内核实现。

        :param ab_dtype: 操作数 A、B 的数据类型
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: scale factor 张量的数据类型
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: scale factor 向量长度
        :type sf_vec_size: int
        :param c_dtype: 输出张量 C 的数据类型
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: MMA 指令 tiler 的 (M, N) 形状
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: CTA cluster 的 (ClusterM, ClusterN) 形状
        :type cluster_shape_mn: Tuple[int, int]
        :param m: 张量 A 的行数
        :type m: int
        :param n: 张量 B 的列数
        :type n: int
        :param k: 张量 A 的列数(K 维长度)
        :type k: int
        :param l: batch 维 L 的大小
        :type l: int
        :param a_major: 张量 A 的主轴(连续维)
        :type a_major: str
        :param b_major: 张量 B 的主轴(连续维)
        :type b_major: str
        :param c_major: 张量 C 的主轴(连续维)
        :type c_major: str

        :return: 可实现为 True, 否则为 False
        :rtype: bool
        """
        can_implement = True
        # 跳过不支持的类型组合
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # 跳过不支持的布局组合
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # 跳过无效的 MMA tile 形状与 cluster 形状
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # 跳过因加载/存储对齐而不合法的问题规模
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement

    # 对布局做 append 再 coalesce 的辅助函数
    @staticmethod
    def append_coalesce_layout(layout):
        # coalesce 语义可参考 cutlass/python/pycute/layout.py 中的 coalesce
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        result = cute.append(part1, part2)
        result = cute.append(result, layout[3])
        result = cute.append(result, layout[4])
        result = cute.append(result, layout[5])
        return result

    @staticmethod
    def adapt_layout_for_tma_ab(composed_layout):
        # 输入:   S<3,4,3> o 0 o ((128,16),1,8,3):((128,1),0,16,16384)
        # 输出:  S<3,4,3> o 0 o (128,(128,3)):(128,(1,16384))
        # 对应 ctaValueMap: (128,384):(1@0,1@1)
        layout = composed_layout.outer
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        part3 = cute.append(part2, layout[3])
        result = cute.append(part1, part3)
        return cute.make_composed_layout(
            composed_layout.inner, composed_layout.offset, result
        )

    @staticmethod
    def adapt_layout_for_tma_sf(layout):
        # TODO: 待 ethan 核对
        # 输入:  (((8,4,4),(16,4)),1,3):(((16,128,4),(0,1)),0,512)
        # 输出:  ((32,4),(16,4,3)):((16,4),(0,1,512))
        # 对应 ctaValueMap: ((8,4,4),(16,4,3)):((1@0@0@0,1@1@0@0,1@2@0@0),(1@0@0@1,1@1@0@1,1@1@1))
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        result = cute.append(cute.group_modes(part1, 0, cute.rank(part1)), part2)
        return result


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """将 scale factor 张量从 MKL 布局转换为 MMA 规约所需的 M(32x4xrest_m)xK(4xrest_k)xL 布局。"""
    # sf_mma_tensor 展平形状为 (32, 4, rest_m, 4, rest_k, l)
    # 分组为 ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    use_tma_store: bool = True,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    """在 Blackwell 架构上执行带性能测试的 persistent 批处理稠密 blockscaled GEMM。

    本函数准备输入张量, 配置并启动 persistent GEMM 内核, 
    可选地与参考结果做校验, 并对执行耗时进行基准测试。

    :param mnkl: 问题规模 (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: 输入张量 A、B 的数据类型
    :type ab_dtype: Type[cutlass.Numeric]
    :param sf_dtype: scale factor 张量的数据类型
    :type sf_dtype: Type[cutlass.Numeric]
    :param sf_vec_size: scale factor 的向量长度
    :type sf_vec_size: int
    :param c_dtype: 输出张量 C 的数据类型
    :type c_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/c_major: 张量 A/B/C 的内存布局(主维)
    :type a_major/b_major/c_major: str
    :param mma_tiler_mn: MMA 分块(tiling)大小 (M, N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: cluster 在 M、N 维上的形状
    :type cluster_shape_mn: Tuple[int, int]
    :param use_tma_store: 是否使用 TMA store 写回 C
    :type use_tma_store: bool, optional
    :param tolerance: 与参考结果比对时的容差, 默认 1e-01
    :type tolerance: float, optional
    :param warmup_iterations: 正式计时前的预热迭代次数, 默认 0
    :type warmup_iterations: int, optional
    :param iterations: 基准测试重复执行次数, 默认 1
    :type iterations: int, optional
    :param skip_ref_check: 是否跳过与参考结果的校验, 默认 False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: 是否采用环形缓冲等工作集策略以尽量保持「冷」L2, 默认 False
    :type use_cold_l2: bool, optional
    :raises RuntimeError: 当 CUDA GPU 不可用时抛出
    :raises TypeError: 当测试用例或配置不受支持时抛出
    :return: GEMM 内核的执行时间
    :rtype: float
    """
    print(f"Running Sm103 Persistent FP4 Ultra Dense BlockScaled GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}")
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Use TMA Store: {'True' if use_tma_store else 'False'}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    import torch
    import cutlass.torch as cutlass_torch

    # 解包 (M, N, K, L)
    m, n, k, l = mnkl

    # 跳过不受支持的测试用例
    if not Sm103BlockScaledPersistentDenseGemmKernel.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
        use_tma_store,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}, "
            f"use_tma_store: {use_tma_store}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # 创建张量 A/B/C
    a_ref = cutlass_torch.matrix(l, m, k, a_major == "m", cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n, c_major == "m", cutlass.Float32)

    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=32
    )

    # 按字节对齐整除性标记张量紧凑形状(动态)
    a_tensor.mark_compact_shape_dynamic(
        mode=1 if a_major == "k" else 0,
        stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
        divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1 if b_major == "k" else 0,
        stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
        divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
    )
    c_tensor.mark_compact_shape_dynamic(
        mode=1 if c_major == "n" else 0,
        stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
        divisibility=64 if ab_dtype == cutlass.Float4E2M1FN else 32,
    )

    # 创建 scale factor 张量 SFA/SFB
    def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(k, sf_vec_size)
        ref_shape = (l, mn, sf_k)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        ref_permute_order = (1, 2, 0)
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        # 在 CPU 上创建 f32 参考用 torch 张量
        ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            ref_shape,
            torch.float32,
            permute_order=ref_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=1,
                max_val=3,
            ),
        )

        # 在 CPU 上创建 f32 的 cute 布局 torch 张量
        cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            mma_shape,
            torch.float32,
            permute_order=mma_permute_order,
            init_type=cutlass_torch.TensorInitType.SCALAR,
            init_config=cutlass_torch.ScalarInitConfig(value=1.0),
        )

        # 将参考 MKL 布局的 f32 张量写入 cute f32 张量(MMA 布局)
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_f32_torch_tensor_cpu),
            from_dlpack(cute_f32_torch_tensor_cpu),
        )
        cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

        # reshape 以使内存连续
        ref_f32_torch_tensor_cpu = (
            ref_f32_torch_tensor_cpu.permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(l, mn, sf_k, sf_vec_size)
            .reshape(l, mn, sf_k * sf_vec_size)
            .permute(*ref_permute_order)
        )
        # 裁减到实际 K 维长度, 供参考比对
        ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

        # 在 CPU 上创建目标 dtype 的 cute 张量占位
        cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
            cute_f32_torch_tensor_cpu,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )

        # 将 f32 cute 张量转换为目标 dtype 的 cute 张量
        cute_tensor = cutlass_torch.convert_cute_tensor(
            cute_f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=True,
        )
        return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(
        l, m, k, sf_vec_size, sf_dtype
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, sf_dtype
    )

    # 配置 GEMM 内核
    gemm = Sm103BlockScaledPersistentDenseGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
    )

    # 计算当前设备允许的最大活跃 cluster 数
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # 初始化 CUDA stream
    current_stream = cutlass_torch.default_stream()

    # 编译 GEMM 内核
    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        c_tensor,
        max_active_clusters,
        current_stream,
    )
    # 计算参考结果并与内核输出比对
    if not skip_ref_check:
        # 执行一次内核以供参考校验
        compiled_gemm(
            a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, current_stream
        )
        print("Verifying results...")
        res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
        res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
        ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)

        # 将 C 转回 f32 以便与参考逐元素比较
        c_ref_device = c_ref.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(c_ref_device, assumed_align=32).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )
        c_ref = c_ref_device.cpu()

        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            # 参考路径: f32 -> f8 量化 -> 再转回 f32 以对齐低精度输出语义
            ref_f8_ = torch.empty(*(l, m, n), dtype=torch.uint8, device="cuda").permute(
                1, 2, 0
            )
            ref_f8 = from_dlpack(ref_f8_, assumed_align=32).mark_layout_dynamic(
                leading_dim=1
            )
            ref_f8.element_type = c_dtype
            ref_device = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0).cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=32).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f8)
            cute.testing.convert(ref_f8, ref_tensor)
            ref = ref_device.cpu()
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
    def generate_tensors():
        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, _ = cutlass_torch.cute_tensor_like(
            c_ref, c_dtype, is_dynamic_layout=True, assumed_align=32
        )

        # 标记张量满足字节对齐约束
        a_tensor.mark_compact_shape_dynamic(
            mode=1 if a_major == "k" else 0,
            stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
            divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
        )
        b_tensor.mark_compact_shape_dynamic(
            mode=1 if b_major == "k" else 0,
            stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
            divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
        )
        c_tensor.mark_compact_shape_dynamic(
            mode=1 if c_major == "n" else 0,
            stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
            divisibility=64 if ab_dtype == cutlass.Float4E2M1FN else 32,
        )

        _, sfa_tensor, _ = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
        _, sfb_tensor, _ = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)
        return cute.testing.JitArguments(
            a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, current_stream
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + sfa_torch.numel() * sfa_torch.element_size()
            + sfb_torch.numel() * sfb_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = cute.testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time  # 返回内核执行时间(微秒)


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Example of Sm103 FP4 Ultra Dense Persistent BlockScaled GEMM."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 6144, 2),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(256, 256),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(2, 4),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E8M0FNU)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--a_major", choices=["k"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n"], type=str, default="n")
    parser.add_argument(
        "--use_tma_store", action="store_true", help="Use tma store or not"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        args.mnkl,
        args.ab_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.c_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_tma_store,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    m, n, k, l = args.mnkl
    flops = 2.0 * m * n * k * l
    tflops = flops / (exec_time * 1e-6) / 1e12
    print(f"Kernel time: {exec_time:.2f} us, TFLOPS: {tflops:.2f}")
    print("PASS")
