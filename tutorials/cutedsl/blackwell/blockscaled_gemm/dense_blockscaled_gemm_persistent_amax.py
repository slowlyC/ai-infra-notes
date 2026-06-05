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
from typing import Type, Tuple, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass._mlir.dialects import math
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack

"""本示例提供 SM100 批处理稠密 blockscaled GEMM kernel 的实验性实现. 请注意, 与该 kernel 相关的 API 和实现细节在后续版本中可能会变化.

这是一个面向 NVIDIA Blackwell SM100 架构, 使用 CUTE DSL 编写的高性能 persistent 批处理稠密 blockscaled GEMM 示例.
- 矩阵 A 为 MxKxL, L 是 batch 维. 对 MXF8 输入类型, A 可为 row-major("K") 或 column-major("M"); 对 MXF4/NVF4 输入类型, A 只能为 row-major("K").
- 矩阵 B 为 NxKxL, L 是 batch 维. 对 MXF8 输入类型, B 可为 row-major("N") 或 column-major("K"); 对 MXF4/NVF4 输入类型, B 只能为 row-major("K").
- 矩阵 C 为 MxNxL, L 是 batch 维, C 可为 row-major("N") 或 column-major("M").
- 矩阵 SFA 的 layout 会根据 A 的形状和 BlockScaledBasicChunk 在内部填充, 元素个数为 Mxceil_div(K, sf_vec_size)xL.
- 矩阵 SFB 的 layout 会根据 B 的形状和 BlockScaledBasicChunk 在内部填充, 元素个数为 Nxceil_div(K, sf_vec_size)xL.

该 GEMM kernel 支持以下特性:
    - 使用 Tensor Memory Access (TMA) 做高效访存.
    - 使用 Blackwell 的 tcgen05.mma 执行 matrix multiply-accumulate (MMA), 包括 2cta mma 指令.
    - 借助 cluster 实现 TMA multicast, 降低 L2 memory traffic.
    - 支持 persistent tile scheduling, 在 tile 之间更好地重叠 memory load/store 与 MMA.
    - 支持 warp specialization, 避免 mainloop load 和 MMA 之间显式 pipeline 同步.

该 GEMM 的执行流程如下:
1. DMA warp: 使用 TMA 将 A 和 B 矩阵从 global memory (GMEM) 加载到 shared memory (SMEM).
2. MMA warp:
    - 使用 tcgen05.cp 指令将 scale factor A/B 从 shared memory (SMEM) 拷贝到 tensor memory (TMEM).
    - 使用 tcgen05.mma 指令执行 matrix multiply-accumulate (MMA).
3. EPILOGUE warp:
    - 使用 tcgen05.ld 将完成的 accumulator 从 tensor memory (TMEM) 加载到 register (RMEM).
    - 将 C 矩阵转换为输出类型.
    - 可选择用 TMA 将 C 从 register (RMEM) 经 shared memory (SMEM) 写回 global memory (GMEM), 也可不经过 TMA 直接从 RMEM 写回 GMEM.
    - 可选择传入逐元素 lambda 函数 epilogue_op 作用于输出张量:
      例如 ReLU 可设置为 epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM100 tcgen05.mma.kind.block_scale 指令的行为如下:
- 从 SMEM 读取矩阵 A.
- 从 SMEM 读取矩阵 B.
- 从 TMEM 读取 scalefactor A.
- 从 TMEM 读取 scalefactor B.
- 将 accumulator 写入 TMEM.
随后必须将 TMEM 中的 accumulator 加载到寄存器, 再写回 GMEM.

本示例的输入参数如下:

.. code-block:: bash

    python examples/blackwell/dense_blockscaled_gemm_persistent.py                  --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16              --c_dtype Float16                                                              --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                                  --mnkl 8192,8192,1024,1

使用 NCU profiler 采集性能:

.. code-block:: bash

    ncu python examples/blackwell/dense_blockscaled_gemm_persistent.py              --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16              --c_dtype Float16                                                              --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                                  --mnkl 8192,8192,1024,1                                                        --warmup_iterations 1 --iterations 10 --skip_ref_check


约束:
* 支持的输入数据类型: mxf8, mxf4, nvf4. 详细的有效 dtype 组合见下文 Sm100BlockScaledPersistentDenseGemmKernel 类文档.
* A/B tensor 必须使用相同数据类型, 不支持混用数据类型, 例如 mxf8 x mxf4.
* MMA tiler M 必须为 128 或 256 (use_2cta_instrs).
* MMA tiler N 必须为 128 或 256.
* Cluster shape M/N 必须为正数且为 2 的幂, cluster 总大小 <= 16.
* 如果 MMA tiler M 为 256 (use_2cta_instrs), cluster shape M 必须为 2 的倍数.
* A/B/C tensor 的连续维至少需要 16 字节对齐, 即 Float8 的元素个数为 16 的倍数, Float4 的元素个数为 32 的倍数.
"""


class Sm100BlockScaledPersistentDenseGemmKernel:
    """本类实现批处理矩阵乘法 (C = A x SFA x B x SFB), 支持多种数据类型, 并使用 Blackwell GPU 的 persistent tile scheduling 与 warp specialization 等架构特性.

    :param sf_vec_size: Scale factor 向量长度.
    :type sf_vec_size: int
    :param mma_tiler_mn: Matrix Multiply-Accumulate (MMA) tile 形状 (M,N).
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: 用于并行处理的 cluster 维度 (M,N).
    :type cluster_shape_mn: Tuple[int, int]

    :note: 当前版本中, A 和 B tensor 必须使用相同数据类型.
        - 例如, 不支持 A 使用 Float8E4M3FN 而 B 使用 Float8E5M2.

    :note: 支持的 A/B 数据类型, SF 数据类型和 SF 向量长度组合:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
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
        - 由于 scale factor 大小有限, 用于 scale factor multicast 时 cluster shape M/N 必须 <= 4.

    Example:
        >>> gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 128),
        ...     cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, amax_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        """初始化 Blackwell 稠密 GEMM kernel 配置.

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
        """

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K 维在 _setup_attributes 中确定.
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # 设置专用 warp id.
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # 设置 cta sync, epilogue sync 和 tmem ptr sync 的 barrier id.
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )

        # Amax reduction 配置.
        self.num_epilog_warps = len(self.epilog_warp_id)

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

    def _setup_attributes(self):
        """设置依赖 GEMM 输入的配置.

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
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )

        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # 这里先构造一次, 用来推导 K tile, cluster layout, smem/tmem layout 等派生属性.
        #  __call__ 中会重新构造并传给 device kernel.
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        # tiled_mma_sfb 是 SFB 专用的辅助 MMA view, 固定用 CtaGroup.ONE.
        # 后续只用于 SFB 的 partition 和 TMA layout, 不代表主 GEMM 多执行一次 MMA.
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # 计算 mma/cluster/tile shapes.
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

        # 计算 cluster layout.
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # 计算 A/B 的 multicast CTA 数量.
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # 计算 epilogue subtile.
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # 设置 shared memory 中的 A/B/C stage 数量和 tensor memory 中的 ACC stage 数量.
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

        # 计算 A/B/SFA/SFB/C shared memory layout.
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

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        amax_tensor: cute.Tensor,
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
        :param amax_tensor: 输出 absolute maximum value 的 tensor.
        :type amax_tensor: cute.Tensor
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
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

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

        # 设置 A 的 TMA load.
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

        # 设置 B 的 TMA load.
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

        # 设置 SFA 的 TMA load.
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

        # 设置 SFB 的 TMA load.
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

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # 设置 C 的 TMA store.
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

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
            # Amax reduction 使用的 shared memory, 每个 epilogue warp 一个 FP32.
            # amax 只有 16 bytes, 因此使用较小 alignment.
            sAmax: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_epilog_warps],
                16,
            ]

        self.shared_storage = SharedStorage
        # 同步启动 kernel.
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
            amax_tensor,
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
        mAmax: cute.Tensor,
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
        """执行 persistent batched GEMM 计算的 GPU device kernel."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # 预取 TMA descriptor.
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

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
        # 分配并初始化: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier.
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # 初始化 mainloop ab_pipeline (barrier) 和 states.
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

        # 初始化 acc_pipeline (barrier) 和 states.
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
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

        # 初始化 tensor memory dealloc barrier.
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # barrier init 后执行 cluster arrive.
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # 设置 smem tensor A/B/SFA/SFB/C.
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

        # 用于 amax reduction 的 shared memory, 每个 epilogue warp 一个 FP32.
        # 简单 1D layout.
        amax_layout = cute.make_layout((self.num_epilog_warps,))
        sAmax = storage.sAmax.get_tensor(amax_layout)

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
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # 为 TiledMMA_A/B/C partition global tensor.
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
        # 为 TMA load A/B partition global/shared tensor.
        #
        # TMA load A 的 partition_S/D.
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
        # TMA load B 的 partition_S/D.
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

        # TMA load scale factor A 的 partition_S/D.
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

        # TMA load scale factor B 的 partition_S/D.
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
        # 为 TiledMMA_A/B/C partition shared/tensor memory tensor.
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # tensor memory alloc 前执行 cluster wait.
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # 专用 TMA load warp.
        #
        if warp_idx == self.tma_warp_id:
            #
            # persistent tile scheduling loop, 循环调度 tile.
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
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
                # 切片到 per-MMA tile index.
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
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # 对 k_tile = prefetch_k_tile_cnt 的 AB buffer empty 执行 peek (try_wait).
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # TMA load loop.
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # 按条件等待 AB buffer empty.
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # 执行 TMA load A/B/SFA/SFB.
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

                    # 对 k_tile = prefetch_k_tile_cnt + k_tile + 1 的 AB buffer empty 执行 peek (try_wait).
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # 前进到下一个 tile.
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 等待 A/B buffer empty.
            #
            ab_pipeline.producer_tail(ab_producer_state)

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
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # 创建 SFB tmem tensor.
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
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
                # 从 tile scheduler 获取 tile coord.
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # 设置当前 tile 的 tensor memory buffer.
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # 对 k_tile = 0 的 AB buffer full 执行 peek (try_wait).
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # 等待 accumulator buffer empty.
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # 为每个 tile 重置 ACCUMULATE field.
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # MMA mainloop.
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # 按条件等待 AB buffer full.
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        # 将 SFA/SFB 从 smem 拷贝到 tmem.
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

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB.
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # 将 SFA/SFB tensor 设置到 tiled_mma.
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # 在第一个 kblock 后对 tCtAcc 启用 accumulate.
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # 异步 arrive AB buffer empty.
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # 对 k_tile = k_tile + 1 的 AB buffer full 执行 peek (try_wait).
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # 异步 arrive accumulator buffer full.
                #
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
            # 为 epilogue 做 partition.
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
            # persistent tile scheduling loop, 循环调度 tile.
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # 参与 tma store pipeline 的 threads/warps.
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
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
                # 切片到 per-MMA tile index.
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

                # 设置当前 tile 的 tensor memory buffer.
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                #
                # 等待 accumulator buffer full.
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # 按 subtile 将 accumulator 存入 global memory.
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                # 为当前 tile 初始化 thread-local amax accumulator.
                # 由于这里计算 absolute maximum, 使用 0.0 作为初始值.
                thread_tile_amax = cutlass.Float32(0.0)

                for subtile_idx in cutlass.range(subtile_cnt):
                    #
                    # 将 accumulator 从 tensor memory buffer 加载到 register.
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # 在当前 tile 的所有 subtile 上累积 thread-level amax.
                    # 注意: 这里需要 absolute value maximum, 因此先取 abs.
                    acc_values = tTR_rAcc.load()
                    # 使用 math.absf 逐元素取绝对值, 支持 vector.
                    abs_acc_values_ir = math.absf(
                        acc_values.ir_value()  # operand, positional 参数.
                    )
                    abs_acc_values = type(acc_values)(
                        abs_acc_values_ir, acc_values.shape, acc_values.dtype
                    )
                    subtile_amax = abs_acc_values.reduce(
                        cute.ReductionOp.MAX,
                        cutlass.Float32(0.0),
                        0,  # 使用 0.0 作为 abs values 的初始值.
                    )
                    thread_tile_amax = cute.arch.fmax(thread_tile_amax, subtile_amax)

                    #
                    # 转换为 C 类型.
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                    tRS_rC.store(acc_vec)

                    #
                    # 将 C 存入 shared memory.
                    #
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # 使用 fence 和 barrier 确保 shared memory store 对 TMA store 可见.
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # 使用 TMA 将 C store 到 global memory.
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, subtile_idx)],
                        )
                        # 使用 fence 和 barrier 确保 shared memory store 对 TMA store 可见.
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                # 所有 subtile 处理完成后执行 amax reduction.
                # 使用 wrapper function 做 warp-level reduction.
                warp_amax = cute.arch.warp_redux_sync(
                    value=thread_tile_amax,
                    kind="fmax",
                    mask_and_clamp=0xFFFFFFFF,
                    nan=True,
                )
                # 每个 epilogue warp 的 lane 0 将 warp amax 写入 shared memory.
                if cute.arch.lane_idx() == 0:
                    sAmax[warp_idx] = cutlass.Float32(warp_amax)

                # 确保所有 epilogue warp 在 block reduction 前完成写入.
                self.epilog_sync_barrier.arrive_and_wait()

                # Block-level reduction: 仅第一个 epilogue warp 的 lane 0 处理.
                if warp_idx == self.epilog_warp_id[0] and cute.arch.lane_idx() == 0:
                    block_amax = cutlass.Float32(
                        0.0
                    )  # absolute maximum 的初始值.
                    for i in cutlass.range(self.num_epilog_warps):
                        warp_amax_val = sAmax[i]
                        block_amax = cute.arch.fmax(block_amax, warp_amax_val)

                    # Global atomic max, 跨所有 tile 累积最终 tensor amax.
                    # 因为这里计算 absolute values, 所有值都是非负的.
                    _ = cute.arch.atomic_fmax(
                        mAmax.iterator, block_amax, sign_bit=False
                    )
                #
                # 异步 arrive accumulator buffer empty.
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # 前进到下一个 tile.
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 释放 tensor memory buffer.
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            #
            # 等待 C store complete.
            #
            c_pipeline.producer_tail()

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

        # 创建 S2T CopyAtom 和 tiledCopy.
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
        """创建用于 tensor memory load 的 tiledCopy, 并用它划分 tensor memory (source) 和 register array (destination).

        :param tidx: epilogue warp group 中的 thread index.
        :type tidx: cutlass.Int32
        :param tAcc: 待拷贝和 partition 的 accumulator tensor.
        :type tAcc: cute.Tensor
        :param gC_mnl: global tensor C.
        :type gC_mnl: cute.Tensor
        :param epi_tile: epilogue tiler.
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: 是否启用 use_2cta_instrs.
        :type use_2cta_instrs: bool

        :return: 返回 (tiled_copy_t2r, tTR_tAcc, tTR_rAcc), 其中:
            - tiled_copy_t2r: tmem 到 register copy (t2r) 的 tiled copy operation.
            - tTR_tAcc: 已 partition 的 accumulator tensor.
            - tTR_rAcc: register 中用于保存 t2r 结果的 accumulated tensor.
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # 创建用于 tensor memory load 的 tiledCopy.
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
        """创建用于 shared memory store 的 tiledCopy, 并用它划分 register array (source) 和 shared memory (destination).

        :param tiled_copy_t2r: tmem 到 register copy (t2r) 的 tiled copy operation.
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: 已 partition 的 accumulator tensor.
        :type tTR_rC: cute.Tensor
        :param tidx: epilogue warp group 中的 thread index.
        :type tidx: cutlass.Int32
        :param sC: 待拷贝和 partition 的 shared memory tensor.
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: 返回 (tiled_copy_r2s, tRS_rC, tRS_sC), 其中:
            - tiled_copy_r2s: register 到 smem copy (r2s) 的 tiled copy operation.
            - tRS_rC: 已 partition 的 tensor C (register source).
            - tRS_sC: 已 partition 的 tensor C (smem destination).
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
        """创建用于 global memory store 的 tiledCopy, 并用它完成:
        在 TMA store 版本中划分 shared memory (source) 和 global memory (destination).

        :param tidx: epilogue warp group 中的 thread index.
        :type tidx: cutlass.Int32
        :param atom: TMA store 版本使用的 copy_atom_c, 或非 TMA store 版本使用的 tiled_copy_t2r.
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: global tensor C.
        :type gC_mnl: cute.Tensor
        :param epi_tile: epilogue tiler.
        :type epi_tile: cute.Tile
        :param sC: 待拷贝和 partition 的 shared memory tensor.
        :type sC: cute.Tensor

        :return: 返回 (tma_atom_c, bSG_sC, bSG_gC), 其中:
            - tma_atom_c: TMA copy atom.
            - bSG_sC: 已 partition 的 shared memory tensor C.
            - bSG_gC: 已 partition 的 global tensor C.
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
        """根据 heuristic 计算 A/B/C operands 的 stage 数量.

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
        # ACC stage 数量.
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # 默认 C stage 数量.
        num_c_stage = 2

        # 计算 A, B, SFA, SFB 和 C 单个 stage 的 smem layout 和大小.
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # 提供一个临时的 1 stage.
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # 提供一个临时的 1 stage.
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # 提供一个临时的 1 stage.
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # 提供一个临时的 1 stage.
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
        amax_bytes = 16

        # 计算 A/B/SFA/SFB stage 数量:
        # 从每个 CTA 的 total smem 开始计算 (capacity / occupancy).
        # 减去 reserved bytes 和初始 C stage bytes.
        # 将剩余空间除以每个 A/B/SFA/SFB stage 所需字节数.
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes + amax_bytes)
        ) // ab_bytes_per_stage

        # 细化 epilogue stage 数量:
        # 计算分配 A/B/SFA/SFB stage 和 reserved bytes 后剩余的 smem.
        # 将剩余未使用的 smem 分配给 epilogue.
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes + amax_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

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
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
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
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
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
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [128, 256]:
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
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # 跳过不支持的 layouts.
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # 跳过无效 mma tile shape 和 cluster shape.
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # 跳过不满足 load/store alignment 的非法 problem shape.
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement


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


def compute_reference_amax(output_tensor) -> float:
    import torch

    """Compute reference amax value on CPU.

    Args:
        output_tensor: torch.Tensor, GEMM output result (CPU tensor)

    Returns:
        float: reference amax value
    """
    # 确保使用 FP32 进行计算.
    if output_tensor.dtype != torch.float32:
        output_fp32 = output_tensor.float()
    else:
        output_fp32 = output_tensor

    # 计算 absolute maximum value.
    reference_amax = torch.amax(torch.abs(output_fp32))

    return reference_amax.item()


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
    import torch
    import cutlass.torch as cutlass_torch

    # 解包参数.
    m, n, k, l = mnkl

    # 跳过不支持的 testcase.
    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
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
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
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
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    # 创建 amax tensor, 单个 FP32 值, 初始化为 -inf.
    amax_ref = cutlass_torch.matrix(
        1,
        1,
        1,
        False,
        cutlass.Float32,
        init_type=cutlass_torch.TensorInitType.SCALAR,
        init_config=cutlass_torch.ScalarInitConfig(-float("inf")),
    )
    amax_tensor, amax_torch = cutlass_torch.cute_tensor_like(
        amax_ref, cutlass.Float32, is_dynamic_layout=True, assumed_align=16
    )

    # 用满足 16B alignment 的 element divisibility 标记 tensor.
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
        divisibility=32 if ab_dtype == cutlass.Float4E2M1FN else 16,
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
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=0,
                max_val=1,
            ),
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
    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
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
        amax_tensor,
        max_active_clusters,
        current_stream,
        options=f"--opt-level 2",
    )

    # 计算 reference result.
    if not skip_ref_check:
        # 执行一次 kernel 用于 reference checking.
        compiled_gemm(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            amax_tensor,
            current_stream,
        )
        print("Verifying results...")
        res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
        res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
        ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)

        # 保存原始 Float32 ref 用于 amax computation (quantization 前).
        ref_for_amax = ref.clone()

        # 将 c 转回 f32 以便比较.
        c_ref_device = c_ref.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(c_ref_device, assumed_align=16).mark_layout_dynamic(
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
            ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            ref_f8.element_type = c_dtype
            ref_device = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0).cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f8)
            cute.testing.convert(ref_f8, ref_tensor)
            ref = ref_device.cpu()
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
        # 校验 amax result.
        device_amax = amax_torch.cpu().squeeze()  # 移除维度, 使其成为 scalar.
        # 使用原始 Float32 ref (quantization 前) 计算 amax.
        reference_amax = torch.tensor(compute_reference_amax(ref_for_amax))

        # AMAX 校验使用与 GEMM result 校验相同的方法.
        torch.testing.assert_close(
            device_amax, reference_amax, atol=tolerance, rtol=1e-02
        )

    def generate_tensors():
        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, _ = cutlass_torch.cute_tensor_like(
            c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
        )

        # 将 tensor 标记为 byte aligned.
        a_tensor.mark_compact_shape_dynamic(
            mode=1 if a_major == "k" else 0,
            stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
            divisibility=2 if ab_dtype == cutlass.Float4E2M1FN else 1,
        )
        b_tensor.mark_compact_shape_dynamic(
            mode=1 if b_major == "k" else 0,
            stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
            divisibility=2 if ab_dtype == cutlass.Float4E2M1FN else 1,
        )
        c_tensor.mark_compact_shape_dynamic(
            mode=1 if c_major == "n" else 0,
            stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
            divisibility=2 if c_dtype == cutlass.Float4E2M1FN else 1,
        )

        _, sfa_tensor, _ = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
        _, sfb_tensor, _ = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)

        # 为 benchmarking 创建 amax tensor, 每次迭代重置为 -inf.
        amax_ref_bench = cutlass_torch.matrix(
            1,
            1,
            1,
            False,
            cutlass.Float32,
            init_type=cutlass_torch.TensorInitType.SCALAR,
            init_config=cutlass_torch.ScalarInitConfig(-float("inf")),
        )
        amax_tensor_bench, _ = cutlass_torch.cute_tensor_like(
            amax_ref_bench, cutlass.Float32, is_dynamic_layout=True, assumed_align=16
        )

        return cute.testing.JitArguments(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            amax_tensor_bench,
            current_stream,
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
        description="Sm100 Dense Persistent BlockScaled GEMM 示例."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(512, 256, 256, 1),
        help="mnkl 维度, 逗号分隔",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="MMA tile 形状, 逗号分隔",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="cluster shape, 逗号分隔",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E8M0FNU)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
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
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
