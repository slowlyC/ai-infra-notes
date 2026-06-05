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
from typing import Type, Tuple, Union, Literal

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

"""
本示例提供 SM100 批处理稠密 blockscaled GEMM 内核的实验性实现; 请注意, 与该内核相关的 API 及实现细节在未来版本中可能会发生变化。

面向 NVIDIA Blackwell SM100 架构、使用 CUTE DSL 的高性能 persistent 批处理稠密 blockscaled GEMM 示例。
- 矩阵 A 为 MxKxL, L 为批维; 对 MXF8 输入类型, A 可为行主("K")或列主("M"); 对 MXF4/NVF4 输入类型, A 仅能为行主("K")
- 矩阵 B 为 NxKxL, L 为批维; 对 MXF8 输入类型, B 可为行主("N")或列主("K"); 对 MXF4/NVF4 输入类型, B 仅能为行主("K")
- 矩阵 C 为 MxNxL, L 为批维; C 可为行主("N")或列主("M")
- 矩阵 SFA 的布局根据 A 的形状与 BlockScaledBasicChunk 在内部填充, 元素个数为 Mxceil_div(K, sf_vec_size)xL
- 矩阵 SFB 的布局根据 B 的形状与 BlockScaledBasicChunk 在内部填充, 元素个数为 Nxceil_div(K, sf_vec_size)xL

本 GEMM 内核支持以下特性: 
    - 使用 Tensor Memory Access(TMA)进行高效访存
    - 使用 Blackwell 的 tcgen05.mma 执行矩阵乘加(MMA)运算(含 2-CTA MMA 指令)
    - 结合 cluster 实现 TMA multicast, 以降低 L2 流量
    - 支持 persistent tile 调度, 在 tile 之间更好地重叠访存与 MMA
    - 支持 warp specialization, 避免 mainloop 加载与 MMA 之间的显式 pipeline 同步

本 GEMM 的工作流程如下: 
1. DMA warp: 使用 TMA 将 A、B 矩阵从全局内存(GMEM)加载到共享内存(SMEM)。
2. MMA warp: 
    - 使用 tcgen05.cp 指令将 scale factor A/B 从 SMEM 拷贝到张量内存(TMEM)。
    - 使用 tcgen05.mma 指令执行矩阵乘加(MMA)。
3. EPILOGUE warp: 
    - 使用 tcgen05.ld 将已完成的累加器从 TMEM 加载到寄存器(RMEM)。
    - 将 C 矩阵类型转换为输出类型。
    - 可选用 TMA 将 C 从 RMEM 经 SMEM 写回 GMEM, 或不经 TMA 直接将 C 从 RMEM 写入 GMEM。
    - 可选用逐元素 lambda 函数 epilogue_op 作用于输出张量: 
      例如 ReLU 可设 epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM100 tcgen05.mma.kind.block_scale 指令的行为如下: 
- 从 SMEM 读取矩阵 A
- 从 SMEM 读取矩阵 B
- 从 TMEM 读取 scalefactor A
- 从 TMEM 读取 scalefactor B
- 将累加器写入 TMEM
随后必须将 TMEM 中的累加器加载到寄存器, 再写回 GMEM。

本示例的输入参数如下所示: 

.. code-block:: bash

    python examples/blackwell/dense_blockscaled_gemm_persistent.py             \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1

使用 NCU 性能分析器采集性能: 

.. code-block:: bash

    ncu python examples/blackwell/dense_blockscaled_gemm_persistent.py         \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1                                                  \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


约束: 
* 支持的输入数据类型: mxf8、mxf4、nvf4
  有效 dtype 组合的详细说明见下文 Sm100BlockScaledPersistentDenseGemmKernel 类文档
* A/B 张量必须使用相同数据类型, 不支持混用(例如 mxf8 x mxf4)
* MMA tiler 的 M 必须为 128 或 256(use_2cta_instrs)
* MMA tiler 的 N 必须为 64/128/192/256
* Cluster 形状的 M/N 必须为正且为 2 的幂, cluster 总大小 ≤ 16
* 若 MMA tiler 的 M 为 256(use_2cta_instrs), 则 cluster 形状的 M 必须为 2 的倍数
* A/B/C 张量的连续维必须至少 16 字节对齐, 
  即对 Float8 与 Float4, 元素个数需分别为 16 与 32 的整数倍。
"""


class Sm100BlockScaledPersistentDenseGemmKernel:
    """本类实现批处理矩阵乘法(C = A x SFA x B x SFB), 支持多种数据类型及
    Blackwell GPU 上特有的架构特性, 并采用 persistent tile 调度与 warp specialization。

    :param sf_vec_size: Scale factor 向量长度。
    :type sf_vec_size: int
    :param mma_tiler_mn: 矩阵乘加(MMA)tile 的形状 (M, N)。
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: 并行处理用的 cluster 维度 (M, N)。
    :type cluster_shape_mn: Tuple[int, int]

    :note: 当前版本中, A 与 B 张量必须使用相同数据类型
        - 例如不支持 A 为 Float8E4M3FN 而 B 为 Float8E5M2

    :note: 支持的 A/B 数据类型、SF 数据类型与 SF 向量长度组合: 
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: 支持的累加器数据类型: 
        - Float32

    :note: 支持的 C 数据类型: 
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: 约束: 
        - MMA tiler 的 M 必须为 128 或 256(use_2cta_instrs)
        - MMA tiler 的 N 必须为 64/128/192/256
        - 若 MMA tiler 的 M 为 256, 则 cluster 形状的 M 必须为 2 的倍数
        - cluster 形状的 M/N 必须为正且为 2 的幂, cluster 总大小 ≤ 16
        - 此外, 因 scale factor 容量有限, 用于 scale factor multicast 时 cluster 形状的 M/N 必须 ≤ 4

    Example:
        >>> gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 128),
        ...     cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        """初始化 Blackwell 稠密 GEMM 内核的配置。

        该配置包含若干关键方面: 

        1.  MMA 指令设置(tcgen05): 
            - acc_dtype: MMA 累加器数据类型, 恒为 Float32
            - sf_vec_size: Scale factor A/B 的向量长度
            - mma_tiler_mn: MMA 指令 tiler 的 (M, N) 形状

        2.  Cluster 形状: 
            - cluster_shape_mn: CTA cluster 的 (ClusterM, ClusterN) 形状

        :param sf_vec_size: Scale factor 向量长度。
        :type sf_vec_size: int
        :param mma_tiler_mn: MMA 指令的 (M, N) 元组形状。
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: cluster 的 (ClusterM, ClusterN) 元组形状。
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K 维在 _setup_attributes 中再确定
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # 设置专用 warp 编号
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # 为 epilogue 同步与 TMEM 指针同步设置 barrier 编号
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

    def _setup_attributes(self):
        """设置依赖于 GEMM 输入的配置。

        本方法根据输入张量属性(数据类型、主维/步幅等)与内核设置配置各项参数: 
        - 配置 tiled MMA
        - 计算 MMA/cluster/tile 形状
        - 计算 cluster 布局
        - 计算 A/B/SFA/SFB 的 multicast CTA
        - 计算 epilogue subtile
        - 设置 A/B/SFA/SFB/C 在 SMEM 中的 stage 数量
        - 计算 A/B/SFA/SFB/C 的 SMEM 布局
        """
        # 计算 MMA 指令形状
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # 计算 MMA/cluster/tile 形状
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # 计算 cluster 布局
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # 计算 A/B 的 multicast CTA 数量
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # 计算 epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # 在 SMEM 中设置 A/B/C 的 stage 数量, 在 TMEM 中设置 ACC 的 stage 数量
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        # 计算 A/B/SFA/SFB/C 的 SMEM 布局
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # 当 num_acc_stage == 1 且 cta_tile_n = 256 时, 重叠并双缓冲累加器
        self.overlapping_accum = self.num_acc_stage == 1

        # 计算 SFA/SFB/累加器占用的 TMEM 列数
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        # 仅在启用 overlapping_accum 时, 需要在 epilogue 中提前释放累加器缓冲区
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_tile_n
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        layouts: cutlass.Constexpr[
            Tuple[tcgen05.OperandMajorMode, tcgen05.OperandMajorMode, utils.LayoutEnum]
        ],
        problem_mnkl: Tuple[int, int, int, int],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """按步骤执行 GEMM: 
        - 在 SMEM/grid/TMA 计算之前设置静态属性
        - 设置 TMA load/store atom 与张量
        - 结合硬件约束计算 grid 大小
        - 定义内核的共享存储结构
        - 同步启动内核

        :param a_tensor: 输入张量 A。
        :type a_tensor: cute.Tensor
        :param b_tensor: 输入张量 B。
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor 张量 A。
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor 张量 B。
        :type sfb_tensor: cute.Tensor
        :param c_tensor: 输出张量 C。
        :type c_tensor: cute.Tensor
        :param max_active_clusters: 最大活跃 cluster 数量。
        :type max_active_clusters: cutlass.Constexpr
        :param stream: 用于异步执行的 CUDA stream。
        :type stream: cuda.CUstream
        :param epilogue_op: 可选的逐元素 lambda, 作用于输出张量。
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: 若输入数据类型与 MMA 指令不兼容。
        """
        # 在 SMEM/grid/TMA 计算之前设置静态属性
        self.a_dtype: Type[cutlass.Numeric] = a_ptr.value_type
        self.b_dtype: Type[cutlass.Numeric] = b_ptr.value_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_ptr.value_type
        self.c_dtype: Type[cutlass.Numeric] = c_ptr.value_type

        m, n, k, l = problem_mnkl
        self.a_major_mode, self.b_major_mode, self.c_layout = layouts

        # 检查输入数据类型是否与 MMA 指令兼容
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # 设置依赖于 GEMM 输入的属性
        self._setup_attributes()

        a_layout = cute.make_ordered_layout((m, cute.assume(k, 32), l), order=(0, 1, 2))
        if cutlass.const_expr(self.a_major_mode == tcgen05.OperandMajorMode.K):
            a_layout = cute.make_ordered_layout(
                (cute.assume(m, 32), k, l), order=(1, 0, 2)
            )
        b_layout = cute.make_ordered_layout((n, cute.assume(k, 32), l), order=(0, 1, 2))
        if cutlass.const_expr(self.b_major_mode == tcgen05.OperandMajorMode.K):
            b_layout = cute.make_ordered_layout(
                (cute.assume(n, 32), k, l), order=(1, 0, 2)
            )
        c_layout = cute.make_ordered_layout((cute.assume(m, 32), n, l), order=(0, 1, 2))
        if cutlass.const_expr(self.c_layout == utils.LayoutEnum.ROW_MAJOR):
            c_layout = cute.make_ordered_layout(
                (m, cute.assume(n, 32), l), order=(1, 0, 2)
            )
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(c_ptr, c_layout)

        # 将 A/B 张量按 scale factor atom 布局填充, 构造 sfa/sfb 张量
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # 为 A 设置 TMA load
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # 为 B 设置 TMA load
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # 为 SFA 设置 TMA load
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # 为 SFB 设置 TMA load
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # ScaledBasis 使用右乘(3 * x 而非 x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # 为 C 设置 TMA store
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # 计算 grid 大小
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # 定义内核的共享存储布局
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # 同步启动内核
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
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
        tiled_mma_sfb: cute.TiledMma,
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
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        执行 Persistent 批处理 GEMM 计算的 GPU device 内核。
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # 预取 TMA 描述符
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # 设置 CTA/线程坐标
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
        # 分配并初始化: A+B full/empty、累加器 full/empty、TMEM 释放 barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # 初始化 mainloop 的 ab_pipeline(barrier)与状态
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # 初始化 acc_pipeline(barrier)与状态
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = self.threads_per_warp * len(self.epilog_warp_id) * (
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

        # TMEM 释放 barrier 初始化
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # barrier 初始化后 cluster arrive
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # 设置 SMEM 张量 A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
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
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # 为 TiledMMA 的 A/B/C 划分全局张量
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # 为 TMA load A/B 划分全局/共享张量
        #
        # TMA load A 的 partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B 的 partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load scale factor A 的 partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load scale factor B 的 partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # 为 TiledMMA 的 A/B/C 划分共享/张量内存张量
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
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
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
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
        # 专用 TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile 调度循环
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
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
                # 切片到每个 MMA tile 索引
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]

                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # 对 k_tile = prefetch_k_tile_cnt: peek(try_wait)AB buffer empty
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # TMA load 循环
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # 条件等待 AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # 对 k_tile = prefetch_k_tile_cnt + k_tile + 1: peek(try_wait)AB buffer empty
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # 前进到下一个 tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 等待 A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # 专用 MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # barrier 同步: 从 SMEM 取回 TMEM 指针
            #
            tmem.wait_for_alloc()

            #
            # 取回 TMEM 指针并构造累加器/SFA/SFB 张量
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # 构造累加器 TMEM 张量
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # 构造 SFA 的 TMEM 张量
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # 构造 SFB 的 TMEM 张量
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # 为 SFA/SFB 的 S2T 拷贝做划分
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
            # Persistent tile 调度循环
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

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

                # 为当前 tile 设置 TMEM 缓冲区
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # 对 k_tile = 0: peek(try_wait)AB buffer full
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # 等待累加器 buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # 对 cta_tile_shape_n=192: 若为奇数 tile, 将 SFB 的 TMEM 起始地址平移两个字(跳过 SFB 前 64 列)
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # 按 SFB 的 64 列为步长平移
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                #
                # 每个 tile 重置 ACCUMULATE 域
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # MMA 主循环(mainloop)
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # 条件等待 AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        # 将 SFA/SFB 从 SMEM 拷贝到 TMEM
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # 将 SFA/SFB 张量设置到 tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # 首个 kblock 之后在 tCtAcc 上启用累加
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # 异步 arrive: AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # 对 k_tile = k_tile + 1: peek(try_wait)AB buffer full
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # 异步 arrive: 累加器 buffer full
                #
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
        #
        # 专用 epilogue warp
        #
        if warp_idx < self.mma_warp_id:
            #
            # 分配 TMEM 缓冲区
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # barrier 同步: 从 SMEM 取回 TMEM 指针
            #
            tmem.wait_for_alloc()

            #
            # 取回 TMEM 指针并构造累加器张量
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # 为 epilogue 做划分
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )

            #
            # Persistent tile 调度循环
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # 参与 TMA store pipeline 的线程/warp
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
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
                # 切片到每个 MMA tile 索引
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # 获取累加器 stage 索引
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = True if acc_stage_index == 0 else False
                else:
                    acc_stage_index = acc_consumer_state.index

                # 为当前 tile 设置 TMEM 缓冲区
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                #
                # 等待累加器 buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # 以 subtile 将累加器写回 GMEM
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n
                                - 1
                                - subtile_idx
                            )
                    #
                    # 将累加器从 TMEM 加载到寄存器
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # 启用 overlapping_accum 时更早异步 arrive 累加器 buffer empty
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            # TMEM load 的 fence
                            cute.arch.fence_view_async_tmem_load()
                            acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    #
                    # 转换为 C 的数据类型
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                    tRS_rC.store(acc_vec)

                    #
                    # 将 C 写入 SMEM
                    #
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # fence 与 barrier, 确保 SMEM 写入对 TMA store 可见
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # TMA 将 C store 到 GMEM
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        # fence 与 barrier, 确保 SMEM 写入对 TMA store 可见
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # 异步 arrive: 累加器 buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # 前进到下一个 tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 释放 TMEM 缓冲区
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            #
            # 等待 C 的 store 完成
            #
            c_pipeline.producer_tail()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        为 scale factor 张量构造 SMEM→TMEM 加载的 tiledCopy, 并据此划分 SMEM(源)与 TMEM(目的)。

        :param sSF: SMEM 中的 scale factor 张量。
        :type sSF: cute.Tensor
        :param tSF: TMEM 中的 scale factor 张量。
        :type tSF: cute.Tensor

        :return: 元组 (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t), 其中: 
            - tiled_copy_s2t: scale factor 的 SMEM→TMEM(s2t)tiled copy 操作
            - tCsSF_compact_s2t: 划分后的 SMEM 中 scale factor 张量
            - tCtSF_compact_s2t: 划分后的 TMEM 中 scale factor 张量
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # 构造 S2T CopyAtom 与 tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        为 TMEM 加载构造 tiledCopy, 并划分 TMEM(源)与寄存器数组(目的)。

        :param tidx: epilogue warp 组中的线程索引。
        :type tidx: cutlass.Int32
        :param tAcc: 待拷贝与划分的累加器张量。
        :type tAcc: cute.Tensor
        :param gC_mnl: 全局张量 C。
        :type gC_mnl: cute.Tensor
        :param epi_tile: epilogue 的 tiler。
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: 是否启用 use_2cta_instrs。
        :type use_2cta_instrs: bool

        :return: 元组 (tiled_copy_t2r, tTR_tAcc, tTR_rAcc), 其中: 
            - tiled_copy_t2r: TMEM→寄存器(t2r)的 tiled copy 操作
            - tTR_tAcc: 划分后的累加器张量
            - tTR_rAcc: 寄存器中用于承接 t2r 结果的累加数据
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # 为 TMEM 加载构造 tiledCopy
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        为 SMEM store 构造 tiledCopy, 并划分寄存器数组(源)与 SMEM(目的)。

        :param tiled_copy_t2r: TMEM→寄存器(t2r)的 tiled copy 操作。
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: 划分后的累加器张量。
        :type tTR_rC: cute.Tensor
        :param tidx: epilogue warp 组中的线程索引。
        :type tidx: cutlass.Int32
        :param sC: 待拷贝与划分的 SMEM 张量。
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: 元组 (tiled_copy_r2s, tRS_rC, tRS_sC), 其中: 
            - tiled_copy_r2s: 寄存器→SMEM(r2s)的 tiled copy 操作
            - tRS_rC: 划分后的张量 C(寄存器源)
            - tRS_sC: 划分后的张量 C(SMEM 目的)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """为 GMEM store 构造 tiledCopy, 并用于 TMA store 版本下
        划分 SMEM(源)与 GMEM(目的)。

        :param tidx: epilogue warp 组中的线程索引。
        :type tidx: cutlass.Int32
        :param atom: TMA store 版本使用的 copy_atom_c, 或非 TMA 版本使用的 tiled_copy_t2r。
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: 全局张量 C。
        :type gC_mnl: cute.Tensor
        :param epi_tile: epilogue 的 tiler。
        :type epi_tile: cute.Tile
        :param sC: 待拷贝与划分的 SMEM 张量。
        :type sC: cute.Tensor

        :return: 元组 (tma_atom_c, bSG_sC, bSG_gC), 其中: 
            - tma_atom_c: TMA copy atom
            - bSG_sC: 划分后的 SMEM 张量 C
            - bSG_gC: 划分后的全局张量 C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """根据启发式计算 A/B/C 操作数的 stage 数量。

        :param tiled_mma: 定义核心计算的 tiled MMA 对象。
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: MMA tiler 的形状 (M, N, K)。
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: 操作数 A 的数据类型。
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: 操作数 B 的数据类型。
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: epilogue tile 形状。
        :type epi_tile: cute.Tile
        :param c_dtype: 操作数 C(输出)的数据类型。
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: 操作数 C 的布局枚举。
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Scale factor 的数据类型。
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor 向量长度。
        :type sf_vec_size: int
        :param smem_capacity: 可用 SMEM 总容量(字节)。
        :type smem_capacity: int
        :param occupancy: 每个 SM 的目标 CTA 数(occupancy)。
        :type occupancy: int

        :return: 计算得到的 stage 数量元组: 
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stage 数量
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # 默认 C 的 stage 数
        num_c_stage = 2

        # 计算 A、B、SFA、SFB、C 单 stage 的 SMEM 布局与大小
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # 临时提供 1 个 stage
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # 临时提供 1 个 stage
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # 临时提供 1 个 stage
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # 临时提供 1 个 stage
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # 计算 A/B/SFA/SFB 的 stage 数: 
        # 从每个 CTA 的 SMEM 总量(capacity / occupancy)出发
        # 减去保留字节与初始 C stage 占用
        # 余量除以每个 A/B/SFA/SFB stage 所需字节数
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # 细化 epilogue 的 stage 数: 
        # 在已为 A/B/SFA/SFB stage 与保留字节分配后计算剩余 SMEM
        # 将剩余未用 SMEM 划归 epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """使用 persistent tile 调度器计算输出张量 C 的 grid 大小。

        :param c: 输出张量 C。
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: CTA tile 的形状 (M, N, K)。
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: 每个 cluster 在 M、N 维上的形状。
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: 最大活跃 cluster 数。
        :type max_active_clusters: cutlass.Constexpr

        :return: 元组, 包含: 
            - tile_sched_params: persistent tile 调度器参数。
            - grid: 内核启动的 grid 形状。
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
        检查 dtype 与 sf_vec_size 是否为有效组合。

        :param ab_dtype: 操作数 A 与 B 的数据类型。
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: scale factor 的数据类型。
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: scale factor 的向量长度。
        :type sf_vec_size: int
        :param c_dtype: 输出张量的数据类型。
        :type c_dtype: Type[cutlass.Numeric]

        :return: 若 dtype 与 sf_vec_size 有效则为 True, 否则为 False。
        :rtype: bool
        """
        is_valid = True

        # 校验 ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # 校验 sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # 校验 sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # 校验 sf_dtype 与 sf_vec_size 的组合
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # 校验 c_dtype
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
        a_major: Literal["m", "k"],
        b_major: Literal["n", "k"],
        c_major: Literal["m", "n"],
    ) -> bool:
        """
        检查布局与 dtype 是否为有效组合。

        :param ab_dtype: 操作数 A 与 B 的数据类型。
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: 输出张量的数据类型。
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: 张量 A 的主维。
        :type a_major: Literal["m", "k"]
        :param b_major: 张量 B 的主维。
        :type b_major: Literal["n", "k"]
        :param c_major: 张量 C 的主维。
        :type c_major: Literal["m", "n"]

        :return: 若布局有效则为 True, 否则为 False。
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
        检查 MMA tiler 与 cluster 形状是否有效。

        :param mma_tiler_mn: MMA 指令 tiler 的 (M, N) 形状。
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: CTA cluster 的 (ClusterM, ClusterN) 形状。
        :type cluster_shape_mn: Tuple[int, int]

        :return: 若 MMA tiler 与 cluster 形状有效则为 True, 否则为 False。
        :rtype: bool
        """
        is_valid = True
        # 跳过无效的 MMA tile 形状
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [64, 128, 192, 256]:
            is_valid = False
        # 跳过非法的 cluster 形状
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # 跳过无效的 cluster 形状
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # scale factor multicast 的 cluster 形状特殊检查: 
            # scale factor 容量有限, 无法在超过 4 个 CTA 之间 multicast。
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
        a_major: Literal["m", "k"],
        b_major: Literal["n", "k"],
        c_major: Literal["m", "n"],
    ) -> bool:
        """
        检查张量对齐是否有效。

        :param m: 张量 A 的行数。
        :type m: int
        :param n: 张量 B 的列数。
        :type n: int
        :param k: 张量 A 的列数。
        :type k: int
        :param l: 批维大小 L(与问题形状一致)。
        :type l: int
        :param ab_dtype: 操作数 A 与 B 的数据类型。
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: 输出张量的数据类型。
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: 张量 A 的主轴。
        :type a_major: Literal["m", "k"]
        :param b_major: 张量 B 的主轴。
        :type b_major: Literal["n", "k"]
        :param c_major: 张量 C 的主轴。
        :type c_major: Literal["m", "n"]

        :return: 若问题形状在对齐约束下有效则为 True, 否则为 False。
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        mnkl: Tuple[int, int, int, int],
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: Literal["m", "k"],
        b_major: Literal["n", "k"],
        c_major: Literal["m", "n"],
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        检查该 GEMM 是否可实现。

        :param mnkl: 问题规模元组 (M, N, K, L)。
        :type mnkl: Tuple[int, int, int, int]
        :param ab_dtype: 操作数 A 与 B 的数据类型。
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: scale factor 张量的数据类型。
        :type sf_dtype: Type[cutlass.Numeric]
        :param a_major: 张量 A 的主轴。
        :type a_major: Literal["m", "k"]
        :param b_major: 张量 B 的主轴。
        :type b_major: Literal["n", "k"]
        :param c_major: 张量 C 的主轴。
        :type c_major: Literal["m", "n"]
        :param sf_vec_size: 向量长度。
        :type sf_vec_size: int
        :param c_dtype: 输出张量的数据类型。
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: MMA 指令 tiler 的 (M, N) 形状。
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: CTA cluster 的 (ClusterM, ClusterN) 形状。
        :type cluster_shape_mn: Tuple[int, int]
        :return: 若可实现则为 True, 否则为 False。
        :rtype: bool
        """
        # 解包参数
        m, n, k, l = mnkl
        can_implement = True
        # 跳过不支持的类型组合
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # 跳过不支持的布局
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # 跳过无效的 MMA tile 形状与 cluster 形状
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # 跳过不满足 load/store 对齐约束的问题形状
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement


# 辅助函数: 将 scale factor 张量从 MKL 布局转换为 (32, 4, restM, 4, restK, l) 格式
@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_ptr: cute.Pointer,
    sf_mma_ptr: cute.Pointer,
    mn: int,
    sf_k: int,
    l: int,
    mma_shape: tuple,
):
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    permuted_shape = tuple(mma_shape[i] for i in mma_permute_order)
    cute_layout = cute.make_ordered_layout(permuted_shape, order=(2, 1, 4, 0, 3, 5))

    sf_ref_tensor = cute.make_tensor(
        sf_ref_ptr, cute.make_layout((mn, sf_k, l), stride=(sf_k, 1, mn * sf_k))
    )
    sf_mma_tensor = cute.make_tensor(sf_mma_ptr, cute_layout)

    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]
    pass


# 辅助函数: 向上取整除法
def ceil_div(a, b):
    return (a + b - 1) // b


# 将 scale factor 张量从 (m, k, l) 转换为 (32, 4, restM, 4, restK, l) 格式
def create_and_reorder_scale_factor_tensor(
    l, mn, k, sf_vec_size, sf_dtype, torch_tensor
):
    """
    在 CUDA 上根据参考张量创建 CUTE 格式的 scale factor 张量。
    """
    sf_k = ceil_div(k, sf_vec_size)
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,  # 批大小
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    # 生成随机 int8 张量, 再转换为 float8_e4m3fn
    cute_tensor = torch.ones(mma_shape, dtype=cutlass_torch.dtype(sf_dtype)).permute(
        3, 4, 1, 5, 2, 0
    )

    # 调用辅助函数完成布局转换
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        make_ptr(
            sf_dtype,
            torch_tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            sf_dtype,
            cute_tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        mn,
        sf_k,
        l,
        mma_shape,
    )
    return cute_tensor.cuda()


# 编译 persistent 稠密 blockscaled GEMM 算子
def scaled_mm(
    gemm_obj: Sm100BlockScaledPersistentDenseGemmKernel,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    a_major: Literal["m", "k"],
    b_major: Literal["n", "k"],
    c_major: Literal["m", "n"],
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
    options: str = "",
):
    # 构造 CuTe 指针
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

    a_major_mode = (
        tcgen05.OperandMajorMode.K if a_major == "k" else tcgen05.OperandMajorMode.MN
    )
    b_major_mode = (
        tcgen05.OperandMajorMode.K if b_major == "k" else tcgen05.OperandMajorMode.MN
    )
    c_layout = (
        utils.LayoutEnum.ROW_MAJOR if c_major == "n" else utils.LayoutEnum.COL_MAJOR
    )
    return cute.compile(
        gemm_obj,
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        (a_major_mode, b_major_mode, c_layout),
        (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
        max_active_clusters,
        stream,
        epilogue_op,
        options=options,
    )


def is_emulated_dtype(
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
) -> bool:
    if c_dtype in {
        cutlass.Float32,
        cutlass.Float16,
        cutlass.BFloat16,
    }:
        if ab_dtype == cutlass.Float4E2M1FN and sf_dtype == cutlass.Float8E4M3FN:
            return False
        if ab_dtype == cutlass.Float8E4M3FN and sf_dtype == cutlass.Float8E8M0FNU:
            return False

    return True


# 将 scale factor 张量从 MKL 布局转为分块(blocked)布局
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape
    # 请确保行数为 128 的倍数、列数为 4 的倍数
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    # 必要时对输入矩阵做填充
    if padded_rows != rows or padded_cols != cols:
        # FP8 类型: 先转为 float32 再填充, 然后视需要转回原 dtype
        original_dtype = input_matrix.dtype
        input_float32 = input_matrix.to(torch.float32)
        padded = torch.nn.functional.pad(
            input_float32,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
        # 若需要则转回原始 dtype
        if original_dtype != input_float32.dtype:
            padded = padded.to(original_dtype)
    else:
        padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


# persistent 稠密 blockscaled GEMM 的参考实现(dtype 模拟路径)
def reference_scaled_mm_emulated(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
    mnkl: Tuple[int, int, int, int],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
):
    m, n, k, l = mnkl
    sfa_expanded = (
        torch.repeat_interleave(sfa, sf_vec_size, dim=1)[:, :k, :]
        .to(dtype=torch.float32)
        .cuda()
    )
    sfb_expanded = (
        torch.repeat_interleave(sfb, sf_vec_size, dim=1)[:, :k, :]
        .to(dtype=torch.float32)
        .cuda()
    )
    res_a = torch.einsum("mkl,mkl->mkl", a, sfa_expanded)
    res_b = torch.einsum("nkl,nkl->nkl", b, sfb_expanded)
    # 将 res_a、res_b 转为 float32 再 einsum, 避免在 'Byte' 上触发 NotImplementedError
    ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)
    c_ref = ref.to(dtype=cutlass_torch.dtype(c_dtype))
    return c_ref


# persistent 稠密 blockscaled GEMM 的参考实现(非模拟 dtype 路径)
def reference_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
    mnkl: Tuple[int, int, int, int],
    c_dtype: Type[cutlass.Numeric],
):
    m, n, k, l = mnkl
    c_ref = torch.clone(c)
    for l_idx in range(l):
        # 将 scale factor 张量转为分块(blocked)格式
        scale_a = to_blocked(sfa[:, :, l_idx])
        scale_b = to_blocked(sfb[:, :, l_idx])
        # 保证 a_slice 为行主 (M, K), 步幅为 (K, 1)
        a_slice = a[:, :, l_idx].contiguous()
        # 保证 b_slice 为行主 (N, K), 转置后得到列主 (K, N)
        b_slice = b[:, :, l_idx].contiguous()
        # (m, k) @ (n, k).T -> (m, n)(GEMM 结果形状)
        res = torch._scaled_mm(
            a_slice,
            b_slice.transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=c_ref.dtype,
        )
        c_ref[:, :, l_idx] = res
    return c_ref


# 为 persistent 稠密 blockscaled GEMM(dtype 模拟路径)构造 CuTe 指针
def construct_cute_pointers_emulated(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
):
    a_cute, _ = cutlass_torch.cute_tensor_like(
        a.cpu(),
        ab_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    a_cute = cutlass_torch.convert_cute_tensor(
        a,
        a_cute,
        ab_dtype,
        is_dynamic_layout=True,
    )
    b_cute, _ = cutlass_torch.cute_tensor_like(
        b.cpu(),
        ab_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    b_cute = cutlass_torch.convert_cute_tensor(
        b,
        b_cute,
        ab_dtype,
        is_dynamic_layout=True,
    )
    a_ptr = a_cute.iterator
    b_ptr = b_cute.iterator

    sfa_ptr = make_ptr(
        sf_dtype, sfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    return a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr, a_cute, b_cute


# 为 persistent 稠密 blockscaled GEMM(非模拟路径)构造 CuTe 指针
def construct_cute_pointers(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
):
    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(
        sf_dtype, sfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    return a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr


# 使用 uint8 / uint32 在 torch 中模拟
# 尚不直接支持的 dtype
def prepare_tensors_emulated(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: Literal["m", "k"],
    b_major: Literal["n", "k"],
    c_major: Literal["m", "n"],
):
    m, n, k, l = mnkl
    sf_k = ceil_div(k, sf_vec_size)

    # 创建 SFA/SFB 张量, 取值于 [1, 3)
    sfa = (
        torch.randint(0, 3, (l, m, sf_k), dtype=torch.uint8)
        .permute(1, 2, 0)
        .to(dtype=cutlass_torch.dtype(sf_dtype))
    )
    sfb = (
        torch.randint(0, 3, (l, n, sf_k), dtype=torch.uint8)
        .permute(1, 2, 0)
        .to(dtype=cutlass_torch.dtype(sf_dtype))
    )

    # 创建 A/B 张量, 取值于 [0, 2)
    if a_major == "k":
        a = torch.randint(-2, 2, (l, m, k), dtype=torch.float32, device="cuda").permute(
            1, 2, 0
        )
    else:
        a = torch.randint(-2, 2, (l, k, m), dtype=torch.float32, device="cuda").permute(
            2, 1, 0
        )
    if b_major == "k":
        b = torch.randint(-2, 2, (l, n, k), dtype=torch.float32, device="cuda").permute(
            1, 2, 0
        )
    else:
        b = torch.randint(-2, 2, (l, k, n), dtype=torch.float32, device="cuda").permute(
            2, 1, 0
        )
    if c_major == "n":
        c = torch.empty(
            (l, m, n), dtype=cutlass_torch.dtype(c_dtype), device="cuda"
        ).permute(1, 2, 0)
    else:
        c = torch.empty(
            (l, n, m), dtype=cutlass_torch.dtype(c_dtype), device="cuda"
        ).permute(2, 1, 0)
    return a, b, c, sfa, sfb


def prepare_tensors(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: Literal["m", "k"],
    b_major: Literal["n", "k"],
    c_major: Literal["m", "n"],
):
    m, n, k, l = mnkl

    if ab_dtype == cutlass.Float4E2M1FN:
        # 使用 int8 为 torch.float4_e2m1fn_x2 分配张量
        # 因此本分支下 K 维元素个数需减半
        k_fct = 2
    else:
        k_fct = 1

    sf_k = ceil_div(k, sf_vec_size)

    # 创建 SFA/SFB 张量
    sfa = (
        torch.randint(0, 3, (l, m, sf_k), dtype=torch.uint8)
        .permute(1, 2, 0)
        .to(dtype=cutlass_torch.dtype(sf_dtype))
    )
    sfb = (
        torch.randint(0, 3, (l, n, sf_k), dtype=torch.uint8)
        .permute(1, 2, 0)
        .to(dtype=cutlass_torch.dtype(sf_dtype))
    )

    # 创建 A/B/C 张量
    if a_major == "k":
        a = torch.randint(
            -2, 2, (l, m, k // k_fct), dtype=torch.int8, device="cuda"
        ).permute(1, 2, 0)
    else:
        a = torch.randint(-2, 2, (l, k, m), dtype=torch.int8, device="cuda").permute(
            2, 1, 0
        )
    if b_major == "k":
        b = torch.randint(
            -2, 2, (l, n, k // k_fct), dtype=torch.int8, device="cuda"
        ).permute(1, 2, 0)
    else:
        b = torch.randint(-2, 2, (l, k, n), dtype=torch.int8, device="cuda").permute(
            2, 1, 0
        )
    if c_major == "n":
        c = torch.randint(
            -2, 2, (l, m, n), dtype=cutlass_torch.dtype(c_dtype), device="cuda"
        ).permute(1, 2, 0)
    else:
        c = torch.randint(
            -2, 2, (l, n, m), dtype=cutlass_torch.dtype(c_dtype), device="cuda"
        ).permute(2, 1, 0)

    if ab_dtype == cutlass.Float4E2M1FN:
        a = a.view(dtype=torch.float4_e2m1fn_x2)
        b = b.view(dtype=torch.float4_e2m1fn_x2)
    else:
        a = a.to(dtype=cutlass_torch.dtype(ab_dtype))
        b = b.to(dtype=cutlass_torch.dtype(ab_dtype))

    c = c.to(dtype=cutlass_torch.dtype(c_dtype))
    return a, b, c, sfa, sfb


# 演示如何将 torch 张量
# 传入 CuTe 内核
def run_scaled_mm(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: Literal["m", "k"],
    b_major: Literal["n", "k"],
    c_major: Literal["m", "n"],
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    """在 Blackwell 架构上执行 persistent 批处理稠密 blockscaled GEMM, 并做性能测试(非模拟 dtype)。

    本函数准备输入张量、配置并启动 persistent GEMM 内核, 可选地与参考结果比对校验, 并对执行耗时做基准测试。

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
    :type a_major/b_major/c_major: Literal["m", "k", "n"]
    :param mma_tiler_mn: MMA tile 尺寸 (M, N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: cluster 在 M、N 维上的形状
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: 与参考结果比对时的容差, 默认 1e-01
    :type tolerance: float, optional
    :param warmup_iterations: 正式计时前的 warmup 次数, 默认 0
    :type warmup_iterations: int, optional
    :param iterations: 基准测试重复执行次数, 默认 1
    :type iterations: int, optional
    :param skip_ref_check: 是否跳过与参考实现的数值校验, 默认 False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: 是否采用环形缓冲等策略尽量得到「冷」L2, 默认 False
    :type use_cold_l2: bool, optional
    :raises RuntimeError: 若当前环境无可用 CUDA GPU
    :raises ValueError: 若配置非法或内核不支持
    :return: GEMM 内核的执行时间
    :rtype: float
    """
    print("Running Sm100 Persistent Dense BlockScaled GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}")
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # 解包参数
    m, n, k, l = mnkl

    # 配置 GEMM 内核
    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
    )

    # 跳过不支持的测试用例
    if not gemm.can_implement(
        mnkl,
        ab_dtype,
        sf_dtype,
        c_dtype,
        a_major,
        b_major,
        c_major,
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # 从 PyTorch 获取当前 CUDA stream
    torch_stream = torch.cuda.current_stream()
    # 取得底层 CUstream 指针
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # 检查当前配置是否可被内核实现
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # 使用占位指针编译 GEMM 内核
    compiled_gemm = scaled_mm(
        gemm,
        ab_dtype,
        c_dtype,
        sf_dtype,
        a_major,
        b_major,
        c_major,
        max_active_clusters,
        current_stream,
        options=f"--opt-level 2",
    )

    # 创建 PyTorch 张量: A、SFA、B、SFB、C
    a, b, c, sfa, sfb = prepare_tensors(
        mnkl, ab_dtype, sf_dtype, sf_vec_size, c_dtype, a_major, b_major, c_major
    )
    # 将 scale factor 张量重排为 (32, 4, restM, 4, restK, l) 格式
    sfa_reordered = create_and_reorder_scale_factor_tensor(
        l, m, k, sf_vec_size, sf_dtype, sfa
    )
    sfb_reordered = create_and_reorder_scale_factor_tensor(
        l, n, k, sf_vec_size, sf_dtype, sfb
    )
    # 构造 CuTe 指针
    a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr = construct_cute_pointers(
        a,
        b,
        sfa_reordered,
        sfb_reordered,
        c,
        ab_dtype,
        sf_dtype,
        c_dtype,
    )

    # 计算参考结果
    if not skip_ref_check:
        # 为参考校验额外执行一次内核
        compiled_gemm(
            a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l), current_stream
        )
        c_ref = reference_scaled_mm(a, b, sfa, sfb, c, (m, n, k, l), c_dtype)
        if c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            # 对低比特宽浮点做按位比较时, 不宜使用 rtol=0.001 与 atol=0.1; 
            # 请改用 rtol=0.0 与 atol=0.0。
            tolerance = 0.0
        torch.testing.assert_close(c, c_ref, atol=tolerance, rtol=tolerance)

    def generate_inputs():
        a, b, c, sfa, sfb = prepare_tensors(
            mnkl,
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            a_major,
            b_major,
            c_major,
        )
        # 将 scale factor 张量重排为 (32, 4, restM, 4, restK, l) 格式
        sfa_reordered = create_and_reorder_scale_factor_tensor(
            l, m, k, sf_vec_size, sf_dtype, sfa
        )
        sfb_reordered = create_and_reorder_scale_factor_tensor(
            l, n, k, sf_vec_size, sf_dtype, sfb
        )
        # 构造 CuTe 指针
        a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr = construct_cute_pointers(
            a,
            b,
            sfa_reordered,
            sfb_reordered,
            c,
            ab_dtype,
            sf_dtype,
            c_dtype,
        )
        jit_args = cute.testing.JitArguments(
            a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l), current_stream
        )
        # 保留对外部对象(例如取 view 时的 Torch 张量)的引用, 防止被提前释放
        jit_args.add_to_scope([a, b, sfa_reordered, sfb_reordered, c])
        return jit_args

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a.numel() * a.element_size()
            + b.numel() * b.element_size()
            + sfa.numel() * sfa.element_size()
            + sfb.numel() * sfb.element_size()
            + c.numel() * c.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = cute.testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_inputs,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    return exec_time  # 返回内核执行时间(微秒)


# 用于兼容 Torch 与 DLPack 均不直接支持的窄精度组合, 
# 例如 Float4E2M1FN 与 Float8E8M0FNU 等。
def run_scaled_mm_with_emulated_dtype(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: Literal["m", "k"],
    b_major: Literal["n", "k"],
    c_major: Literal["m", "n"],
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    """在 Blackwell 架构上执行 persistent 批处理稠密 blockscaled GEMM, 并做性能测试(模拟 dtype)。

    本函数准备输入张量、配置并启动 persistent GEMM 内核, 可选地与参考结果比对校验, 并对执行耗时做基准测试。

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
    :type a_major/b_major/c_major: Literal["m", "n","k"]
    :param mma_tiler_mn: MMA tile 尺寸 (M, N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: cluster 在 M、N 维上的形状
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: 与参考结果比对时的容差, 默认 1e-01
    :type tolerance: float, optional
    :param warmup_iterations: 正式计时前的 warmup 次数, 默认 0
    :type warmup_iterations: int, optional
    :param iterations: 基准测试重复执行次数, 默认 1
    :type iterations: int, optional
    :param skip_ref_check: 是否跳过与参考实现的数值校验, 默认 False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: 是否采用环形缓冲等策略尽量得到「冷」L2, 默认 False
    :type use_cold_l2: bool, optional
    :raises RuntimeError: 若当前环境无可用 CUDA GPU
    :raises ValueError: 若配置非法或内核不支持
    :return: GEMM 内核的执行时间
    :rtype: float
    """
    print("Running Sm100 Persistent Dense BlockScaled GEMM test (Emulated) with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}")
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # 解包参数
    m, n, k, l = mnkl

    # 配置 GEMM 内核
    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
    )

    # 跳过不支持的测试用例
    if not gemm.can_implement(
        mnkl,
        ab_dtype,
        sf_dtype,
        c_dtype,
        a_major,
        b_major,
        c_major,
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # 从 PyTorch 获取当前 CUDA stream
    torch_stream = torch.cuda.current_stream()
    # 取得底层 CUstream 指针
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # 检查当前配置是否可被内核实现
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # 使用占位指针编译 GEMM 内核
    compiled_gemm = scaled_mm(
        gemm,
        ab_dtype,
        c_dtype,
        sf_dtype,
        a_major,
        b_major,
        c_major,
        max_active_clusters,
        current_stream,
        options=f"--opt-level 2",
    )

    # 创建 PyTorch 张量: A、SFA、B、SFB、C
    a, b, c, sfa, sfb = prepare_tensors_emulated(
        mnkl, ab_dtype, sf_dtype, sf_vec_size, c_dtype, a_major, b_major, c_major
    )
    # 将 scale factor 张量重排为 (32, 4, restM, 4, restK, l) 格式
    sfa_reordered = create_and_reorder_scale_factor_tensor(
        l, m, k, sf_vec_size, sf_dtype, sfa
    )
    sfb_reordered = create_and_reorder_scale_factor_tensor(
        l, n, k, sf_vec_size, sf_dtype, sfb
    )
    # 构造 CuTe 指针
    a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr, a_cute, b_cute = (
        construct_cute_pointers_emulated(
            a,
            b,
            sfa_reordered,
            sfb_reordered,
            c,
            ab_dtype,
            sf_dtype,
            c_dtype,
        )
    )

    # 计算参考结果
    if not skip_ref_check:
        # 为参考校验额外执行一次内核
        compiled_gemm(
            a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l), current_stream
        )
        c_ref = reference_scaled_mm_emulated(
            a, b, sfa, sfb, c, (m, n, k, l), sf_vec_size, c_dtype
        )
        if c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            # 对低比特宽浮点做按位比较时, 不宜使用 rtol=0.001 与 atol=0.1; 
            # 请改用 rtol=0.0 与 atol=0.0。
            tolerance = 0.0
        torch.testing.assert_close(c, c_ref, atol=tolerance, rtol=tolerance)

    def generate_inputs():
        a, b, c, sfa, sfb = prepare_tensors_emulated(
            mnkl,
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            a_major,
            b_major,
            c_major,
        )
        # 将 scale factor 张量重排为 (32, 4, restM, 4, restK, l) 格式
        sfa_reordered = create_and_reorder_scale_factor_tensor(
            l, m, k, sf_vec_size, sf_dtype, sfa
        )
        sfb_reordered = create_and_reorder_scale_factor_tensor(
            l, n, k, sf_vec_size, sf_dtype, sfb
        )
        # 构造 CuTe 指针
        a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr, a_cute, b_cute = (
            construct_cute_pointers_emulated(
                a,
                b,
                sfa_reordered,
                sfb_reordered,
                c,
                ab_dtype,
                sf_dtype,
                c_dtype,
            )
        )
        jit_args = cute.testing.JitArguments(
            a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l), current_stream
        )
        # 保留对外部对象(例如取 view 时的 Torch 张量)的引用, 防止被提前释放
        jit_args.add_to_scope([a, b, sfa_reordered, sfb_reordered, c, a_cute, b_cute])
        return jit_args

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a.numel() * a.element_size()
            + b.numel() * b.element_size()
            + sfa.numel() * sfa.element_size()
            + sfb.numel() * sfb.element_size()
            + c.numel() * c.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = cute.testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_inputs,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    return exec_time  # 返回内核执行时间(微秒)


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: Literal["m", "k"],
    b_major: Literal["n", "k"],
    c_major: Literal["m", "n"],
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    """
    根据 dtype 选择并执行合适的 GEMM 入口。

    若需要 dtype 模拟则调用 run_scaled_mm_with_emulated_dtype, 否则调用 run_scaled_mm。
    """
    if is_emulated_dtype(ab_dtype, sf_dtype, c_dtype):
        exec_time = run_scaled_mm_with_emulated_dtype(
            mnkl,
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            a_major,
            b_major,
            c_major,
            mma_tiler_mn,
            cluster_shape_mn,
            tolerance,
            warmup_iterations,
            iterations,
            skip_ref_check,
            use_cold_l2,
        )
    else:
        exec_time = run_scaled_mm(
            mnkl,
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            a_major,
            b_major,
            c_major,
            mma_tiler_mn,
            cluster_shape_mn,
            tolerance,
            warmup_iterations,
            iterations,
            skip_ref_check,
            use_cold_l2,
        )
    return exec_time


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "格式无效, 应为逗号分隔的整数列表。"
            )

    parser = argparse.ArgumentParser(
        description="Sm100 稠密 persistent blockscaled GEMM 示例。"
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(512, 256, 256, 1),
        help="mnkl 各维尺寸(逗号分隔)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="MMA tile 形状 (M, N), 逗号分隔",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="cluster 形状 (M, N), 逗号分隔",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="与参考结果比对时的容差"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="正式计时前的 warmup 次数"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="内核重复执行次数(基准测试)",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="跳过与参考实现的数值校验"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="使用环形缓冲的多组张量, 尽量得到冷的 L2 cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl 必须恰好包含 4 个整数")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn 必须恰好包含 2 个整数")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn 必须恰好包含 2 个整数")

    # 按 dtype 选择对应入口并执行 GEMM
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
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
