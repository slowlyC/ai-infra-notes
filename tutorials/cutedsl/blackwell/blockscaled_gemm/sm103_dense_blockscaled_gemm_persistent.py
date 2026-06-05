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

"""本示例提供 SM103 批处理 FP4 Ultra blockscaled GEMM kernel 的实验性实现. 请注意, 与该 kernel 相关的 API 和实现细节在后续版本中可能会变化.

这是一个面向 NVIDIA Blackwell SM103 架构, 使用 CUTE DSL 编写的高性能 persistent 批处理 FP4 Ultra blockscaled GEMM 示例.
    - 矩阵 A 为 MxKxL, L 是 batch 维. 对 MXF4/NVF4 输入类型, A 只能为 row-major("K").
    - 矩阵 B 为 NxKxL, L 是 batch 维. 对 MXF4/NVF4 输入类型, B 只能为 row-major("K").
    - 矩阵 C 为 MxNxL, L 是 batch 维, C 可为 row-major("N") 或 column-major("M").
    - 矩阵 SFA 的 layout 会根据 A 的形状和 sm103_BlockScaledBasicChunk 在内部填充, 元素个数为 Mxceil_div(K, sf_vec_size)xL.
    - 矩阵 SFB 的 layout 会根据 B 的形状和 sm103_BlockScaledBasicChunk 在内部填充, 元素个数为 Nxceil_div(K, sf_vec_size)xL.

该 GEMM kernel 支持以下特性:
    - 使用 Tensor Memory Access (TMA) 做高效访存.
    - 使用 Blackwell 的 tcgen05.mma 执行 matrix multiply-accumulate (MMA), 包括 2cta mma 指令.
    - 借助 cluster 实现 TMA multicast, 降低 L2 memory traffic.
    - 支持 persistent tile scheduling, 在 tile 之间更好地重叠 memory load/store 与 MMA.
    - 支持 warp specialization, A/B 与 scale factor 使用独立的 TMA warp.
    - 使用 circular buffer 技术, 优化 memory 与计算的重叠.

该 GEMM 的执行流程如下:
    1. TMA A/B warp: 使用 TMA 将 A 和 B 矩阵从 global memory (GMEM) 加载到 shared memory (SMEM).
    2. TMA SF warp: 使用 TMA 将 scale factor A/B 从 global memory (GMEM) 加载到 shared memory (SMEM).
    3. MMA warp:
        - 使用 tcgen05.cp 指令将 scale factor A/B 从 shared memory (SMEM) 拷贝到 tensor memory (TMEM).
        - 使用 tcgen05.mma 指令执行 matrix multiply-accumulate (MMA), 并配合 circular buffering.
    4. Epilogue warps:
        - 使用 tcgen05.ld 将完成的 accumulator 从 tensor memory (TMEM) 加载到 register (RMEM).
        - 将 C 矩阵转换为输出类型.
        - 不使用 TMA operations, 直接将 C 从 register (RMEM) 写回 global memory (GMEM).
        - 可选择传入逐元素 lambda 函数 epilogue_op 作用于输出张量:
            例如 ReLU 可设置为 epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM103 tcgen05.mma.kind.block_scale 指令的行为如下:
    - 从两个 SMEM buffer 读取矩阵 A, 包括 current buffer 和 next buffer.
    - 从两个 SMEM buffer 读取矩阵 B, 包括 current buffer 和 next buffer.
    - 从 TMEM 读取 scalefactor A.
    - 从 TMEM 读取 scalefactor B.
    - 将 accumulator 写入 TMEM.

随后必须将 TMEM 中的 accumulator 加载到寄存器, 再写回 GMEM.

本示例的输入参数如下:

.. code-block:: bash

    python examples/blackwell/sm103_dense_blockscaled_gemm_persistent.py           --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16            --c_dtype Float16                                                             --mma_tiler_mn 256,256 --cluster_shape_mn 2,4                                --mnkl 4096,4096,6144,1

使用 NCU profiler 采集性能:

.. code-block:: bash

    ncu python examples/blackwell/sm103_dense_blockscaled_gemm_persistent.py       --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16            --c_dtype Float16                                                             --mma_tiler_mn 256,256 --cluster_shape_mn 2,4                                --mnkl 4096,4096,6144,1                                                      --warmup_iterations 1 --iterations 10 --skip_ref_check

约束:
    - 支持的输入数据类型: mxf4, nvf4.
        - 详细的有效 dtype 组合见下文 Sm103BlockScaledPersistentDenseGemmKernel 类文档.
    - A/B tensor 必须使用相同数据类型.
    - MMA tiler M 必须为 128 或 256 (use_2cta_instrs).
    - MMA tiler N 必须为 128 或 256.
    - Cluster shape M/N 必须为正数且为 2 的幂, cluster 总大小 <= 16.
    - 如果 MMA tiler M 为 256 (use_2cta_instrs), cluster shape M 必须为 2 的倍数.
    - A/B/C tensor 的连续维至少需要 16 字节对齐, 即 MXF4/NVF4 的元素个数为 32 的倍数.
"""


class Sm103BlockScaledPersistentDenseGemmKernel:
    """本类实现批处理矩阵乘法 (C = A x SFA x B x SFB), 支持 FP4 数据类型, 并使用 Blackwell SM103 GPU 的 persistent tile scheduling 与 warp specialization 等架构特性.

    :param sf_vec_size: Scale factor 向量长度.
    :type sf_vec_size: int
    :param mma_tiler_mn: Matrix Multiply-Accumulate (MMA) tile 形状 (M,N).
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: 用于并行处理的 cluster 维度 (M,N).
    :type cluster_shape_mn: Tuple[int, int]


    :note: 当前版本中, A 和 B tensor 必须使用相同数据类型.
        - 即不支持 A 与 B 使用不同元素类型.

    :note: 支持的 A/B 数据类型, SF 数据类型和 SF 向量长度组合:
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: 支持的 accumulator 数据类型:
        - Float32

    :note: 支持的 C 数据类型:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: 约束:
        - MMA tiler M 必须为 128 或 256 (use_2cta_instrs).
        - MMA tiler N 必须为 128/256.
        - 如果 MMA tiler M 为 256, cluster shape M 必须为 2 的倍数.
        - Cluster shape M/N 必须为正数且为 2 的幂, cluster 总大小 <= 16.
        - 由于 scale factor 大小有限, scale factor multicast 要求 cluster shape M/N <= 4.

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
        """初始化 Blackwell SM103 FP4 Ultra GEMM kernel 配置.

        该配置包含几个方面:

        1.  MMA 指令设置 (tcgen05):
            - acc_dtype: MMA accumulator 数据类型, 固定为 Float32.
            - sf_vec_size: Scale factor A/B 的向量长度.
            - mma_tiler_mn: MMA 指令 tiler 的 (M, N) 形状.

        2.  Cluster 形状:
            - cluster_shape_mn: CTA cluster 的 (ClusterM, ClusterN) 形状.

        :param sf_vec_size: Scale factor 向量长度.
        :type sf_vec_size: int
        :param mma_tiler_mn: MMA 指令的 (M, N) 形状元组.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: cluster 的 (ClusterM, ClusterN) 形状元组.
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: 是否启用 TMA store.
        :type use_tma_store: bool
        """
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K 维在 _setup_attributes 中确定.
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # 设置专用 warp id.
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
        # 设置 epilogue sync 和 tmem ptr sync 的 barrier id.
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_103")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_103")
        self.sf_buffers_per_tile_k = 4 if self.sf_vec_size == 16 else 2

    def _setup_attributes(self):
        """设置依赖运行时 tensor 输入的 kernel 属性.

        该方法会根据输入 tensor 属性和 kernel 设置来配置若干属性, 包括:
        - 配置 tiled MMA.
        - 计算 MMA/cluster/tile 形状.
        - 计算 cluster layout.
        - 计算 A/B/SFA/SFB 的 multicast CTA 数量.
        - 计算 epilogue subtile.
        - 设置 A/B/SFA/SFB/C 在 shared memory 中的 stage 数量.
        - 计算 A/B/SFA/SFB/C 的 shared memory layout.
        """
        # 计算 mma instruction shapes.
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])

        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
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

        # 计算 mma/cluster/tile shapes.
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

        # 计算 cluster layout.
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (dummy_tiled_mma_sfb.thr_id.shape,),
        )

        # 计算 A/B 的 multicast CTA 数量.
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # 计算 epilogue subtile.
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

        # 计算 A/B/SFA/SFB/C shared memory layout.
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

        # 当 cta_tile_n = 256 且 num_acc_stage == 1 时, 对 accumulator 做 overlap 和 double buffer.
        self.overlapping_accum = self.num_acc_stage == 1 and not self.use_tma_store
        self.epi_tile_n = cute.size(self.epi_tile[1])

        if self.overlapping_accum:
            # 从 scale factor layout 计算 SF TMEM column count.
            # 列数 = Int32-recast layout 的 cosize & 0xFFFF,
            # 与 find_tmem_tensor_col_offset 中的计算保持一致.
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
            # overlapping 时在 epilogue 中提前释放 accumulator buffer.
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
        """按以下步骤执行 GEMM operation:
        - 在 smem/grid/tma 计算前设置静态属性.
        - 设置 TMA load/store atom 和 tensor.
        - 根据硬件约束计算 grid size.
        - 定义 kernel 的 shared storage.
        - 同步启动 kernel.

        :param a_tensor: 输入 tensor A.
        :type a_tensor: cute.Tensor
        :param b_tensor: 输入 tensor B.
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A.
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B.
        :type sfb_tensor: cute.Tensor
        :param c_tensor: 输出 tensor C.
        :type c_tensor: cute.Tensor
        :param max_active_clusters: 最大 active cluster 数量.
        :type max_active_clusters: cutlass.Constexpr
        :param stream: 用于异步执行的 CUDA stream.
        :type stream: cuda.CUstream
        :param epilogue_op: 可选的逐元素 lambda 函数, 作用于输出 tensor.
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: 输入数据类型与 MMA 指令不兼容时抛出.
        """
        # 在 smem/grid/tma 计算前设置静态属性.
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)
        # 检查输入数据类型是否兼容 MMA instruction.
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # 设置依赖 gemm 输入的属性.
        self._setup_attributes()

        # 通过将 A/B tensor 填充到 scale factor atom layout 来设置 sfa/sfb tensor.
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

        # 设置 A 的 TMA load.
        a_op = sm103_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        # 将 layout cast 为 uint8 用于 multicast.
        a_smem_layout_tma_ready = self.adapt_layout_for_tma_ab(
            self.a_smem_layout_staged_tma
        )
        a_tensor_uint8 = cute.recast_tensor(a_tensor, cutlass.Uint8)
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            a_op,
            a_tensor_uint8,
            a_smem_layout_tma_ready,
            # 384 对应单次 MMA mainloop 迭代在 K 维处理的 uint8 元素数量.
            (cute.size(tiled_mma.tv_layout_A[1][0]), 384),
            self.cluster_shape_mn[1],
            internal_type=cutlass.Uint8,
        )

        # 设置 B 的 TMA load.
        b_op = sm103_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        # 将 layout cast 为 uint8 用于 multicast.
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

        # 设置 SFA 的 TMA load.
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

        # 设置 SFB 的 TMA load.
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

        # 设置 C 的 TMA store.
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

        # 计算 grid size.
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # 定义 kernel 的 shared storage.
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

        # 同步启动 kernel.
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

    # GPU device kernel, 入口 kernel.
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
        """执行 persistent batched GEMM 计算的 GPU device kernel."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # 预取 TMA descriptor.
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
        # 设置 cta/thread 坐标.
        #
        # cluster 内坐标.
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
        # CTA 内坐标.
        tidx, _, _ = cute.arch.thread_idx()

        #
        # 分配并初始化: a+b full/empty, sfa+sfb full/empty, accumulator full/empty, tensor memory dealloc barrier.
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # 初始化 mainloop ab_producer 和 ab_consumer.
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

        # 初始化 mainloop sf_producer 和 sf_consumer.
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

        # 初始化 acc_pipeline (barrier) 和 states.
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
        # 初始化 tensor memory dealloc barrier.
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # barrier init 后执行 cluster arrive.
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # 设置 smem tensor A/B/SFA/SFB/C.
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
        # 计算 A/B/SFA/SFB buffer full 的 multicast mask.
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
        # 对 global tensor 做 local_tile partition.
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
        # 为 TiledMMA_A/B/C partition global tensor.
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

        # 创建 tCgA_tmp.
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

        # 创建 C 的 identity tensor, 用于 epilogue predication.
        idC = cute.make_identity_tensor(mC_mnl.shape)
        cC_mnl = cute.local_tile(
            idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCcC = thr_mma.partition_C(cC_mnl)

        #
        # 为 TMA load A/B partition global/shared tensor.
        #
        # TMA load A 的 partition_S/D.
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
        # TMA load B 的 partition_S/D.
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

        # scale factor A 的 TMA partition.
        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA_compact = cute.filter_zeros(tAsSFA)

        # scale factor B 的 TMA partition.
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
        # 为 TiledMMA_A/B/C partition shared/tensor memory tensor.
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
        # tensor memory alloc 前执行 cluster wait.
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # 构造 scheduler.
        #
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # 用于 A/B tensor 的专用 TMA load warp.
        #
        if warp_idx == self.tma_ab_warp_id:
            #
            # AB load 的 persistent tile scheduling loop.
            #
            buffers_per_k_tile = 3

            while work_tile.is_valid_tile:
                # 从 tile scheduler 获取 tile coord.
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # 切片到 per-MMA tile index.
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

                # 对 k_tile = prefetch_k_tile_cnt 的 AB buffer empty 执行 peek (try_wait).
                ab_producer.reset()
                peek_ab_empty_status = cutlass.Boolean(1)
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # A/B tensor 的 TMA load loop.
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # 加载 buffers_per_k_tile 个 buffer.
                    for buffer in cutlass.range(buffers_per_k_tile, unroll_full=True):
                        # 获取下一个 empty AB buffer.
                        ab_empty = ab_producer.acquire_and_advance(peek_ab_empty_status)

                        # 执行 TMA load A/B.
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

                        # 对 next buffer 的 AB buffer empty 执行 peek (try_wait).
                        peek_ab_empty_status = cutlass.Boolean(1)
                        # 检查当前是否不是最后一个 k_tile 的最后一个 buffer.
                        if not (
                            (k_tile == k_tile_cnt - 1)
                            and (buffer == buffers_per_k_tile - 1)
                        ):
                            peek_ab_empty_status = ab_producer.try_acquire()

                # 前进到下一个 tile.
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # 发出 AB loads 结束信号.
            ab_producer.tail()

        #
        # 用于 scale factor tensor 的专用 TMA load warp.
        #
        if warp_idx == self.tma_sf_warp_id:
            #
            # SF load 的 persistent tile scheduling loop.
            #
            while work_tile.is_valid_tile:
                # 从 tile scheduler 获取 tile coord.
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0],
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # 切片到 per-MMA tile index.
                #
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgSFB_slice = tBgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # 对 SF buffer empty 执行 peek (try_wait).
                sf_producer.reset()
                peek_sf_empty_status = cutlass.Boolean(1)
                peek_sf_empty_status = sf_producer.try_acquire()

                #
                # scale factor 的 TMA load loop.
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # 根据 sf_buffers_per_tile_k 加载 SF stages.
                    for sf_stage in cutlass.range(
                        self.sf_buffers_per_tile_k, unroll_full=True
                    ):
                        # 获取下一个 empty SF buffer.
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

                        # 为当前 SF stage 执行 TMA load SFA/SFB.
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

                        # 对 next stage 的 SF buffer empty 执行 peek (try_wait).
                        peek_sf_empty_status = cutlass.Boolean(1)
                        # 检查当前是否不是最后一个 k_tile 的最后一个 stage.
                        if not (
                            k_tile == k_tile_cnt - 1
                            and sf_stage == self.sf_buffers_per_tile_k - 1
                        ):
                            peek_sf_empty_status = sf_producer.try_acquire()

                # 前进到下一个 tile.
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # 发出 SF loads 结束信号.
            sf_producer.tail()

        #
        # 专用 MMA warp.
        #
        if warp_idx == self.mma_warp_id:
            #
            # 通过 barrier 同步, 用于从 shared mem 取回 tensor memory ptr.
            #
            tmem.wait_for_alloc()

            #
            # 取回 tensor memory ptr 并创建 accumulator/SFA/SFB tensor.
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # 创建 accumulator tmem tensor.
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # 创建 SFA tmem tensor.
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

            # 创建 SFB tmem tensor.
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
            # 为 SFA/SFB 的 S2T copy 做 partition.
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
            # persistent tile scheduling loop, 循环调度 tile.
            #
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            MmasPerSfBuffer = 8 // self.sf_buffers_per_tile_k
            sf_stride = 6 if self.sf_vec_size == 16 else 3

            while work_tile.is_valid_tile:
                # 从 tile scheduler 获取 tile coord.
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # 获取 accumulator stage index.
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                # 设置当前 tile 的 tensor memory buffer.
                tCtAcc = tCtAcc_base[(None, 0, 0, acc_stage_index)]

                # 对 k_tile = 0 的 AB buffer full 执行 peek (try_wait).
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                # 对 SF buffer full 执行 peek (try_wait).
                sf_consumer.reset()
                peek_sf_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_sf_full_status = sf_consumer.try_wait()

                #
                # 为每个 tile 重置 ACCUMULATE field.
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                is_first_iteration = True

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        # 根据 sf_vec_size 条件加载 MMA0/MMA1 的 SFA/SFB.
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

                        # 等待 A/B data ready (MMA0, MMA1, part of MMA2).
                        ab_full0 = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # 对 next stage 执行 peek (MMA2, MMA3, MMA4, part of MMA5).
                        peek_ab_full_status = cutlass.Boolean(1)
                        peek_ab_full_status = ab_consumer.try_wait()

                        # 延后 acc acquire, 用于 ublock tmem.
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

                        # 条件加载 MMA2/MMA3 的 SFA/SFB.
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

                        # 等待 A/B data ready (MMA2, MMA3, MMA4, part of MMA5).
                        ab_full1 = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # 对 next stage 执行 peek (part of MMA5, MMA6, MMA7).
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

                        # 释放 stage_ab_0, 因为后续不再需要.
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

                        # 条件加载 MMA4/MMA5 的 SFA/SFB.
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

                        # 等待 A/B data ready (part of MMA5, MMA6, MMA7).
                        ab_full2 = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # 对下一轮 loop 的 first stage 执行 peek (MMA0, MMA1, part of MMA2).
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

                        # 条件加载 MMA6/MMA7 的 SFA/SFB.
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
                # 前进到下一个 tile.
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 等待 accumulator buffer empty.
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
        # 专用 epilogue warps.
        #
        if warp_idx < self.mma_warp_id:
            #
            # 分配 tensor memory buffer.
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # 通过 barrier 同步, 用于从 shared memory 取回 tensor memory ptr.
            #
            tmem.wait_for_alloc()

            #
            # 取回 tensor memory ptr 并创建 accumulator tensor.
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # persistent tile scheduling loop, 循环调度 tile.
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
                # 从 tile scheduler 获取 tile coord.
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                #
                # 预先前进到下一个 tile.
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
                # 等待 C store complete.
                c_pipeline.producer_tail()
            else:
                # 在 TMEM dealloc 前同步, 由 caller 完成.
                tmem_dealloc_barrier.arrive_and_wait()

            #
            # 释放 tensor memory buffer.
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
        """专用于从 SMEM circular-buffered A/B 读取的 GEMM.

        执行 D <- A * B + C, 其中 A 和 B 由 (current, next) buffer 构造出的 circular SMEM descriptor 描述. C 和 D 可以 alias.

        某些 tcgen05 MMA 需要调用方在该 routine 之外显式切换 accumulate field, 这里由调用方负责.

        所有 tensor 都必须已经按提供的 tiled MMA partition.

        对需要 single-threaded execution 的 MMA Atom, gemm op 会在内部自动处理 thread election. 这种情况下不需要手动选择 thread.

        :param atom: MMA atom.
        :type atom: cute.MmaAtom
        :param d: Destination tensor.
        :type d: cute.Tensor
        :param sA_cur: operand A 的 current shared memory tensor.
        :type sA_cur: cute.Tensor
        :param sA_next: operand A 的 next shared memory tensor, 用于 circular buffering.
        :type sA_next: cute.Tensor
        :param sB_cur: operand B 的 current shared memory tensor.
        :type sB_cur: cute.Tensor
        :param sB_next: operand B 的 next shared memory tensor, 用于 circular buffering.
        :type sB_next: cute.Tensor
        :param c: 第三个 source tensor.
        :type c: cute.Tensor
        :return: None.
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
        """为 SM103 (FP4 Ultra) 创建 blockscaled trivial tiled MMA, K 固定为 96.

        返回按给定 (M, N) tiler 和 CTA group 配置的 tcgen05 MMA.

        :param sf_dtype: scale factor 的数据类型, 通常为 8-bit.
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: scale factor 的向量长度.
        :type sf_vec_size: int
        :param cta_group: CTA group 配置.
        :type cta_group: tcgen05.CtaGroup
        :param mma_tiler_mn: MMA tiler 维度 (M, N).
        :type mma_tiler_mn: Tuple[int, int]
        :param a_source: operand A 的 source 位置, 默认是 SMEM.
        :type a_source: tcgen05.OperandSource

        :return: 为 SM103 blockscaled operations 配置好的 tiled MMA atom.
        :rtype: cute.TiledMma

        :raises TypeError: 数据类型不受支持时抛出.
        :raises ValueError: sf_vec_size 不受支持时抛出.
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

    # Utils 辅助方法.
    @staticmethod
    def sm103_make_smem_layout_a(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: cute.Tile,
        num_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        """创建 operand A 的 SM103 shared memory layout."""
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
        """创建 operand B 的 SM103 shared memory layout."""
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
        """创建 SM103 blockscaled basic chunk layout."""

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
        """创建 operand A 的 scale factor shared memory layout."""
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
        """创建 operand B 的 scale factor shared memory layout."""
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
        """创建用于 scale factor tensor 从 smem 加载到 tmem 的 tiledCopy, 并用它划分 smem memory (source) 和 tensor memory (destination).

        :param sSF: smem 中的 scale factor tensor.
        :type sSF: cute.Tensor
        :param tSF: tmem 中的 scale factor tensor.
        :type tSF: cute.Tensor

        :return: 返回 (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t), 其中:
            - tiled_copy_s2t: scale factor tensor 从 smem 到 tmem 加载的 tiled copy operation (s2t).
            - tCsSF_compact_s2t: smem 中已 partition 的 scale factor tensor.
            - tSF_compact_s2t: tmem 中已 partition 的 scale factor tensor.
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
        # 创建 S2T CopyAtom 和 tiledCopy.
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
        """根据 heuristic 计算 A/B 和 SF operands 的 stage 数量.

        :param tiled_mma: 定义主计算的 tiled MMA object.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: MMA tiler 的 (M, N, K) 形状.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: operand A 的数据类型.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: operand B 的数据类型.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: epilogue tile 形状.
        :type epi_tile: cute.Tile
        :param c_dtype: operand C (output) 的数据类型.
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: operand C 的 layout enum.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Scale factor 的数据类型.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor 向量长度.
        :type sf_vec_size: int
        :param smem_capacity: 可用 shared memory 总容量, 单位为 bytes.
        :type smem_capacity: int
        :param occupancy: 每个 SM 目标 CTA 数量 (occupancy).
        :type occupancy: int

        :return: 返回计算出的 stage 数量:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stage 数量, 与 SM100 dense blockscaled gemm 相同.
        num_acc_stage = 1 if mma_tiler[1] == 256 else 2

        # 默认 C stage 数量.
        num_c_stage = 2 if use_tma_store else 0

        # 计算 A, B, SFA, SFB 单个 stage 的 smem layout 和大小.
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

        # 细化 epilogue stage 数量:
        # 计算分配 A/B stage 和 reserved bytes 后剩余的 smem.
        # 将剩余未使用的 smem 分配给 epilogue.
        if use_tma_store:
            # xinyu TODO: 不确定是否与 c++ 对齐.
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
        """使用 persistent tile scheduler 计算输出 tensor C 的 grid size.

        :param c: 输出 tensor C.
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: CTA tile 的 (M, N, K) 形状.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: 每个 cluster 在 M, N 维度上的形状.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: 最大 active cluster 数量.
        :type max_active_clusters: cutlass.Constexpr

        :return: 返回:
            - tile_sched_params: persistent tile scheduler 的参数.
            - grid: kernel launch 的 grid 形状.
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
        """检查 dtypes 和 sf_vec_size 是否构成有效组合.

        :param ab_dtype: A 和 B operands 的数据类型.
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: scale factor 的数据类型.
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: scale factor 向量长度.
        :type sf_vec_size: int
        :param c_dtype: 输出 tensor 的数据类型.
        :type c_dtype: Type[cutlass.Numeric]

        :return: 如果 dtypes 和 sf_vec_size 有效则返回 True, 否则返回 False.
        :rtype: bool
        """
        is_valid = True

        # 检查 ab_dtype 是否有效.
        if ab_dtype != cutlass.Float4E2M1FN:
            is_valid = False

        # 检查 sf_vec_size 是否有效.
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # 检查 sf_dtype 是否有效.
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # 检查 sf_dtype 和 sf_vec_size 组合是否有效.
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False

        # 检查 c_dtype 是否有效.
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
        """检查 layouts 和 dtypes 是否构成有效组合.

        :param ab_dtype: A 和 B operands 的数据类型.
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: 输出 tensor 的数据类型.
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: A tensor 的 major dimension.
        :type a_major: Literal["m", "k"]
        :param b_major: B tensor 的 major dimension.
        :type b_major: Literal["n", "k"]
        :param c_major: C tensor 的 major dimension.
        :type c_major: Literal["m", "n"]

        :return: 如果 layouts 有效则返回 True, 否则返回 False.
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
        """检查 mma tiler 和 cluster shape 是否有效.

        :param mma_tiler_mn: MMA 指令 tiler 的 (M, N) 形状.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: CTA cluster 的 (ClusterM, ClusterN) 形状.
        :type cluster_shape_mn: Tuple[int, int]

        :return: 如果 mma tiler 和 cluster shape 有效则返回 True, 否则返回 False.
        :rtype: bool
        """
        is_valid = True
        # 跳过无效 mma tile shape.
        if not mma_tiler_mn[0] in [128, 256]:
            is_valid = False
        if not mma_tiler_mn[1] in [128, 256]:
            is_valid = False
        # 跳过非法 cluster shape.
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # 跳过无效 cluster shape.
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # 针对 scale factor multicast 的特殊 cluster shape 检查.
            # 由于 scale factor 大小有限, 不能在超过 4 个 CTA 之间 multicast.
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
        """检查 tensor alignment 是否有效.

        :param m: A tensor 的行数.
        :type m: int
        :param n: B tensor 的列数.
        :type n: int
        :param k: A tensor 的列数.
        :type k: int
        :param l: C tensor 的列数.
        :type l: int
        :param ab_dtype: A 和 B operands 的数据类型.
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: 输出 tensor 的数据类型.
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: A tensor 的 major axis.
        :type a_major: Literal["m", "k"]
        :param b_major: B tensor 的 major axis.
        :type b_major: Literal["n", "k"]
        :param c_major: C tensor 的 major axis.
        :type c_major: Literal["m", "n"]

        :return: 如果 problem shape 有效则返回 True, 否则返回 False.
        :rtype: bool
        """
        is_valid = True

        def check_contigous_alignment(
            dtype, is_mode0_major, tensor_shape, alignment_bytes
        ):
            """检查 tensor 是否满足所需的 byte alignment.

            :param dtype: tensor 的数据类型.
            :param is_mode0_major: mode 0 是否为 major (contiguous) mode.
            :param tensor_shape: tensor 形状 (mode0, mode1, batch).
            :param alignment_bytes: 所需 alignment 字节数, 例如 16 或 32.
            :return: 如果满足 alignment 则返回 True.
            """
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            # 计算满足 alignment 所需的连续元素数量.
            # alignment_bytes * 8 (bits per byte) / dtype.width (bits per element).
            num_contiguous_elements = alignment_bytes * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        # 检查 A/B tensor 的 16B alignment.
        # 检查 C tensor 的 32B alignment.
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
        """检查该 gemm 是否可实现.

        :param mnkl: problem size 元组 (M, N, K, L).
        :type mnkl: Tuple[int, int, int, int]
        :param ab_dtype: A 和 B operands 的数据类型.
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: scale factor tensor 的数据类型.
        :type sf_dtype: Type[cutlass.Numeric]
        :param a_major: A tensor 的 major axis.
        :type a_major: Literal["m", "k"]
        :param b_major: B tensor 的 major axis.
        :type b_major: Literal["n", "k"]
        :param c_major: C tensor 的 major axis.
        :type c_major: Literal["m", "n"]
        :param sf_vec_size: 向量长度.
        :type sf_vec_size: int
        :param c_dtype: 输出 tensor 的数据类型.
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: MMA 指令 tiler 的 (M, N) 形状.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: CTA cluster 的 (ClusterM, ClusterN) 形状.
        :type cluster_shape_mn: Tuple[int, int]
        :return: 如果该 gemm 可实现则返回 True, 否则返回 False.
        :rtype: bool
        """
        can_implement = True
        # 跳过不支持的类型.
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # 跳过不支持的 layouts.
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # 跳过无效 mma tile shape 和 cluster shape.
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # 跳过不满足 load/store alignment 的非法 problem shape.
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement

    # 用于 append 和 coalesce layout 的 helper function.
    @staticmethod
    def append_coalesce_layout(layout):
        # coalesce 类似: cutlass/python/pycute/layout.py:coalesce.
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        result = cute.append(part1, part2)
        result = cute.append(result, layout[3])
        result = cute.append(result, layout[4])
        result = cute.append(result, layout[5])
        return result

    @staticmethod
    def adapt_layout_for_tma_ab(composed_layout):
        # 输入: S<3,4,3> o 0 o ((128,16),1,8,3):((128,1),0,16,16384).
        # 输出: S<3,4,3> o 0 o (128,(128,3)):(128,(1,16384)).
        # 用于 ctaValueMap: (128,384):(1@0,1@1).
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
        # TODO: 需要 ethan 确认这里.
        # 输入: (((8,4,4),(16,4)),1,3):(((16,128,4),(0,1)),0,512).
        # 输出: ((32,4),(16,4,3)):((16,4),(0,1,512)).
        # 用于 ctaValueMap: ((8,4,4),(16,4,3)):((1@0@0@0,1@1@0@0,1@2@0@0),(1@0@0@1,1@1@0@1,1@1@1)).
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        result = cute.append(cute.group_modes(part1, 0, cute.rank(part1)), part2)
        return result


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """将 scale factor tensor 从 MKL layout 转换为 MMA 规范的 M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor 的 flatten shape 为 (32, 4, rest_m, 4, rest_k, l).
    # 组合为 ((32, 4, rest_m), (4, rest_k), l).
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
    """在 Blackwell 架构上执行 persistent batched dense blockscaled GEMM operation, 并进行性能 benchmark.

    该函数会准备输入 tensor, 配置并启动 persistent GEMM kernel, 可选执行 reference 校验, 并统计执行性能.

    :param mnkl: problem size (M, N, K, L).
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: 输入 tensor A 和 B 的数据类型.
    :type ab_dtype: Type[cutlass.Numeric]
    :param sf_dtype: scale factor tensor 的数据类型.
    :type sf_dtype: Type[cutlass.Numeric]
    :param sf_vec_size: scale factor tensor 的向量长度.
    :type sf_vec_size: int
    :param c_dtype: 输出 tensor C 的数据类型.
    :type c_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/c_major: tensor A/B/C 的 memory layout.
    :type a_major/b_major/c_major: str
    :param mma_tiler_mn: MMA tiling size.
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster shape.
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: reference 校验比较的 tolerance, 默认 1e-01.
    :type tolerance: float, optional
    :param warmup_iterations: benchmark 前的 warmup 迭代次数, 默认 0.
    :type warmup_iterations: int, optional
    :param iterations: benchmark 迭代次数, 默认 1.
    :type iterations: int, optional
    :param skip_ref_check: 是否跳过 reference result 校验, 默认 False.
    :type skip_ref_check: bool, optional
    :param use_cold_l2: 是否使用 circular buffer 策略确保 cold L2 cache, 默认 False.
    :type use_cold_l2: bool, optional
    :raises RuntimeError: CUDA GPU 不可用时抛出.
    :raises ValueError: 配置无效或 kernel 不支持时抛出.
    :return: GEMM kernel 的执行时间.
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

    # 解包参数.
    m, n, k, l = mnkl

    # 跳过不支持的 testcase.
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

    # 创建 tensor A/B/C.
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

    # 用 byte alignment divisibility 标记 tensor.
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

    # 创建 scale factor tensor SFA/SFB.
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

        # 创建 f32 ref torch tensor (cpu).
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

        # 创建 f32 cute torch tensor (cpu).
        cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            mma_shape,
            torch.float32,
            permute_order=mma_permute_order,
            init_type=cutlass_torch.TensorInitType.SCALAR,
            init_config=cutlass_torch.ScalarInitConfig(value=1.0),
        )

        # 将 ref f32 tensor 转换为 cute f32 tensor.
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_f32_torch_tensor_cpu),
            from_dlpack(cute_f32_torch_tensor_cpu),
        )
        cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

        # reshape 会使 memory contiguous.
        ref_f32_torch_tensor_cpu = (
            ref_f32_torch_tensor_cpu.permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(l, mn, sf_k, sf_vec_size)
            .reshape(l, mn, sf_k * sf_vec_size)
            .permute(*ref_permute_order)
        )
        # 裁剪到 mkl 用于 reference check.
        ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

        # 创建目标 dtype 的 cute torch tensor (cpu).
        cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
            cute_f32_torch_tensor_cpu,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )

        # 将 f32 cute tensor 转换为目标 dtype 的 cute tensor.
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

    # 配置 gemm kernel.
    gemm = Sm103BlockScaledPersistentDenseGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
    )

    # 计算当前 device 上的 max active clusters.
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # 初始化 Stream.
    current_stream = cutlass_torch.default_stream()

    # 编译 gemm kernel.
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
    # 计算 reference result.
    if not skip_ref_check:
        # 执行一次 kernel 用于 reference checking.
        compiled_gemm(
            a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, current_stream
        )
        print("Verifying results...")
        res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
        res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
        ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)

        # 将 c 转回 f32 以便比较.
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
            # 转换 ref: f32 -> f8 -> f32.
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

        # 将 tensor 标记为 byte aligned.
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

    return exec_time  # 返回执行时间, 单位为 microseconds.


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Sm103 FP4 Ultra Dense Persistent BlockScaled GEMM 示例."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 6144, 2),
        help="mnkl 维度, 逗号分隔",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(256, 256),
        help="MMA tile 形状, 逗号分隔",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(2, 4),
        help="cluster shape, 逗号分隔",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E8M0FNU)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--a_major", choices=["k"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n"], type=str, default="n")
    parser.add_argument(
        "--use_tma_store", action="store_true", help="是否使用 TMA store"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="validation 容忍度"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="warmup 迭代次数"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="运行 kernel 的迭代次数",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="跳过 reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="使用 circular buffer tensor sets 确保 L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    run(
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
    print("PASS")
