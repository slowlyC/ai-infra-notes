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

# 本文件为第二个 NVFP4 GEMM 教程, 在第一个教程基础上增加 2CTA MMA 指令及 2x1 cluster.

import argparse
import os
import sys
from typing import Type, Tuple
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack, make_ptr

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(current_dir, "..", "..")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

from blackwell.tutorial_gemm.utils import create_parser, run

# MMA tile 从 (128,256) 扩大到 (256,256), 启用 2CTA 指令
mma_tiler_mn = (256, 256)
mma_inst_shape_k = 64
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16
# 2x1 cluster: M 维度 2 个 CTA 协作, B/SFB tensor 可 multicast
cluster_shape_mnk = (2, 1, 1)

"""
第二个 NVFP4 教程, 相较 `nvfp4_gemm_0.py`, 增加了 2CTA MMA 与 TMA multicast 支持.
优化原理与 `fp16_gemm_1.py` 相同, 此处针对 NVFP4 block-scaled GEMM 的具体参数做说明.


1. 2CTA MMA 减小 B tensor 的 SMEM, 从而支持更多 ab stages 以掩盖 dram 延迟.

同一 SM 上的两个 CTA 共享 TMEM, 由 leader CTA (编号 0) 发射 MMA 指令, follower CTA 不参与 MMA 但参与数据搬运和同步.

硬件在执行 cta_group::2 的 MMA 时, 能通过 DSMEM(Distributed Shared Memory) 直接读取 peer CTA 的 SMEM 中的 B 数据。
所以可以把 B 的 N 维对半拆到两个 CTA 的 SMEM 中, 各存一半, 硬件自动处理跨 CTA 的读取。

scale factor 不支持2CTA 间共享 (CtaGroup.ONE), 因为其数据流经过三级: GMEM → SMEM → TMEM. 
MMA 指令读取 SFA/SFB 时, 操作数是 TMEM, 不是 SMEM, 硬件不会像处理 B 矩阵那样通过 DSMEM 去访问 peer CTA 的 SMEM。
换句话说, B 矩阵享有 "SMEM descriptor + DSMEM" 这套跨 CTA 共享机制, 而 SFB 走的是 SMEM → TMEM → MMA 这条路, 
中间的 S2T copy 和 TMEM 访问都是 CTA-local 的, 没有跨 CTA 共享 SMEM 的能力。

两个 CTA 协作计算更大的 output tile C[256, 256], 沿 M 维各负责一半:
  CTA_0: C[0:128, 0:256]   需要 A[0:128, K]   + B[0:256, K]
  CTA_1: C[128:256, 0:256] 需要 A[128:256, K] + B[0:256, K]

对 A: 每个 CTA 只需自己那 128 行, SMEM 不变.
对 B: 硬件将 B 对半分摊到两个 CTA 的 SMEM 中, 各存 128 行, 通过 DSMEM 互相读取.
对 SFA: 跟随 A, 每个 CTA 只存自己那 128 行, SMEM 不变.
对 SFB: 每个 CTA 仍需完整存储 256 行 SFB, SMEM 不变.

SMEM 每 stage 用量 (NVFP4 有 A + B + SFA + SFB 四份):
  A:   128 x 256 x 0.5B = 16KB (不变)
  SFA: 128 x (256/16) x 1B = 2KB (不变, 跟随 A)
  SFB: 256 x (256/16) x 1B = 4KB (不变, 不支持 2CTA 共享)
  B:   1CTA: 256 x 256 x 0.5B = 32KB → 2CTA: 128 x 256 x 0.5B = 16KB (减半)

SMEM 节省带来更多 pipeline stages:
  1CTA: 每 stage 54KB (16+32+2+4), 最多 227//54 = 4 stages, 延迟掩盖 512*3 = 1.5K cycles
  2CTA: 每 stage 38KB (16+16+2+4), 最多 227//38 = 5 stages, 延迟掩盖 512*4 = 2K cycles


2. TMA multicast 减少 L2 流量.

TMA multicast 是 TMA 的广播模式: 一次 GMEM 读取, 硬件自动将数据同时写入 cluster 内多个 CTA 的 SMEM.

使用前提是 Thread Block Cluster: 保证 cluster 内多个 CTA 被调度到物理相邻的 SM 上,
可通过 DSMEM 互访 SMEM, 通过 TMA multicast 广播数据.
本示例 cluster_shape_mnk = (2, 1, 1), 即 M 维 2 个 CTA 组成一个 cluster.

以 2x1 cluster 为例:
  B:   被两个 CTA 共用, 但这里不是 TMA multicast 广播同一份 B.
       2CTA MMA 使用 CopyBulkTensorTileG2SOp(CtaGroup.TWO), 将 B tile 分摊到两个 CTA 的 SMEM:
       每个 CTA 只存一半 B, MMA 指令可通过 DSMEM 读取 peer CTA 的另一半.
       相比两个独立 CTA 分别加载完整 B, B 的 L2 流量减半.
  SFB: 被两个 CTA 共用, 但 scale factor 不走 B operand 的 2CTA SMEM 共享路径.
       每个 CTA 仍需完整 SFB, 因此使用 TMA multicast 将同一份 SFB 广播到两个 CTA 的 SMEM,
       SFB 的 L2 流量减半.
  A/SFA: 两个 CTA 负责不同 M 行, 数据各不相同, 不做 multicast, 各自从 L2 读取.

每 tile 的 L2 流量 = A 流量 / (N 维 cluster 数) + B 流量 / (M 维 cluster 数):
  无 multicast:  16KB + 32KB = 48KB
  2x1 cluster:   16KB / 1 + 32KB / 2 = 24KB
  4x4 cluster:   16KB / 4 + 32KB / 4 = 12KB

总结: 
2CTA MMA 是硬件通过 DSMEM 让两个 CTA 共享 B 数据。每个 CTA 只需存 B 的一半到 SMEM, MMA 指令自动跨 CTA 读取另一半。
TMA multicast 在本例主要用于 SFB, 因为每个 CTA 都需要完整 SFB, multicast 可以避免从 L2 读两份相同 SFB。
2CTA 提供更强的延迟掩盖能力, TMA multicast 缩短数据就绪时间, 在延迟/内存带宽受限场景下需综合考虑.


运行此示例:
.. code-block:: bash

    python examples/blackwell/tutorial_gemm/nvfp4_gemm_1.py --mnkl 256,256,256,1

    python examples/blackwell/tutorial_gemm/nvfp4_gemm_1.py  \
      --mnkl 8192,8192,8192,1 --do_benchmark

本示例约束:
* m, n, k 的问题规模必须能被 tile size m&n&k (256, 256, 256) 整除
* scale factor 向量大小为 16.
* A/B 矩阵在 k 维度上数据连续.
* C 矩阵在 n 维度上数据连续.
* A/B 矩阵数据类型为 Float4E2M1FN.
* SFA/SFB 矩阵数据类型为 Float8E4M3FN.
"""


class Sm100BlockScaledDenseGemmKernel:
    def __init__(self):
        self.threads_per_cta = 128
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        # 分配 TMEM 的全部 512 列
        self.num_tmem_alloc_cols = 512

        # 2CTA 使 B SMEM 减半 (32KB->16KB), AB stage 从 4 增加到 5
        # 1CTA: 227//(16+32+2+4)=4 stages, 2CTA: 227//(16+16+2+4)=5 stages
        self.num_acc_stage = 1
        self.num_ab_stage = 5

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: tuple,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # 在 smem/grid/tma 计算之前设置静态属性
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        m, n, k, l = problem_size

        # mma_tiler_mn[0]=256 时启用 2CTA MMA (nvfp4_gemm_0 中为 128, 不启用)
        self.use_2cta_instrs = False if mma_tiler_mn[0] == 128 else True

        # 设置 mma_tiler
        mma_inst_tile_k = 4
        self.mma_tiler = (
            mma_tiler_mn[0],  # 256
            mma_tiler_mn[1],  # 256
            mma_inst_shape_k * mma_inst_tile_k,  # 64 * 4 = 256
        )

        # SFB 不支持 2CTA SMEM 共享, 需要单独构建 CtaGroup.ONE 的 tiled_mma_sfb.
        # SFB 是 (N, K/sf_vec_size), 但 CUTLASS 的 tiled_mma 是一个统一的抽象, 因此需要传入 M 维.
        self.mma_inst_shape_sfb = (
            mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1), # 128
            mma_tiler_mn[1],  # 256 
            mma_inst_shape_k,  # 64
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],  # 128
            self.mma_inst_shape_sfb[1],  # 256
            mma_inst_shape_k * mma_inst_tile_k,  # 64 * 4 = 256
        )

        # 创建 CuTe Tensor 对象的描述符
        # cute.assume(k, 32)是编译期提示, 告诉编译器 k 一定是 32 的倍数, 方便后续优化.
        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        # 256bit 对齐, row_major
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout(
                (cute.assume(m, 32), cute.assume(n, 16), l),
                stride=(cute.assume(n, 16), 1, cute.assume(m * n, 512)),
            ),
        )

        # 根据数据 tensor(A 或 B)的 shape, 生成对应 scale factor tensor 的 GMEM layout.
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL): ((32, 4), (16, 4), L)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL): ((32, 4), (16, 4), L)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        # 使用 CtaGroup.TWO 启用 2CTA MMA
        mma_op = tcgen05.MmaMXF4NVF4Op(
            sf_dtype,
            (*mma_tiler_mn, mma_inst_shape_k),  # (256, 256, 64)
            tcgen05.CtaGroup.ONE if not self.use_2cta_instrs else tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
        )
        # tiled_mma: thr_id = 2:1 (size=2, 即 2CTA 协作), shape_mnk = (256, 256, 64)
        tiled_mma = cute.make_tiled_mma(mma_op)

        # 注意: sfB 不支持在 2 CTA 间共享, 需单独处理
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        sfb_mma_op = tcgen05.MmaMXF4NVF4Op(
            sf_dtype,
            self.mma_inst_shape_sfb,  # (128, 256, 64)
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
        )
        # tiled_mma_sfb: thr_id = 1:0 (size=1, 单 CTA), shape_mnk = (128, 256, 64)
        tiled_mma_sfb = cute.make_tiled_mma(sfb_mma_op)

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // (2 if self.use_2cta_instrs else 1), # 128
            self.mma_tiler[1],  # 256
            self.mma_tiler[2],  # 256
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // (2 if self.use_2cta_instrs else 1),  # 64
            self.mma_tiler_sfb[1],  # 256
            self.mma_tiler_sfb[2],  # 256
        )

        # tiled_divide 将 cluster layout (2,1,1) 按 thr_id.shape 分块, 拆出 V (被 MMA 消耗) 和剩余维度.
        # 剩余维度 > 1 表示该方向有多个 CTA 持有相同数据, 可用 TMA multicast.
        #
        # cluster_layout_vmnk (2CTA MMA):
        #   thr_id.shape = 2, M 维被完全消耗 → ((2),1,1,1):((1),0,0,0)
        #   shape = ((2,), 1, 1, 1)  →  V=2, M_rest=1, N_rest=1, K_rest=1
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(cluster_shape_mnk),  # (2, 1, 1)
            (tiled_mma.thr_id.shape,),
        )
        # cluster_layout_sfb_vmnk (1CTA MMA):
        #   thr_id.shape = 1, M 维未消耗 → ((1),2,1,1):((0),1,0,0)
        #   shape = ((1,), 2, 1, 1)  →  V=1, M_rest=2, N_rest=1, K_rest=1
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout(cluster_shape_mnk),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # 计算 A/B/SFA/SFB 的 multicast CTA 数量 (nvfp4_gemm_0 中无 multicast)
        # A 沿 N 维 multicast (shape[2]=1, 无), B 沿 M 维 multicast (shape[1]=1, 无, 靠 DSMEM 共享)
        # SFB 沿 M 维 multicast (sfb shape[1]=2, 有 multicast)
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])    # = 1
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])    # = 1
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])  # = 2
        self.is_a_mcast = self.num_mcast_ctas_a > 1      # False
        self.is_b_mcast = self.num_mcast_ctas_b > 1      # False
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1  # True

        # 计算 A/B/SFA/SFB/C 的 SMEM layout: Swizzle o offset o Layout
        # A: S<3,4,3> o 0 o ((128,64),1,4,5):((256,1),0,64,32768) → 128x256 x 0.5B = 16KB/stage
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            self.num_ab_stage,
        )
        # B: S<3,4,3> o 0 o ((128,64),1,4,5):((256,1),0,64,32768) → 128x256 x 0.5B = 16KB/stage (2CTA 减半)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            self.num_ab_stage,
        )
        # SFA: ((((32,4),1),(16,4)),1,4,5):...  → 128x16 x 1B = 2KB/stage
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            self.num_ab_stage,
        )
        # SFB: ((((32,4),2),(16,4)),1,4,5):...  → 256x16 x 1B = 4KB/stage (完整, 不减半)
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            self.num_ab_stage,
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)  # = 2

        # TMA atom 根据 cluster shape 和 thr_id 自动选择 multicast/non-multicast op.
        # A 的 TMA load: cluster N 维=1, 无 multicast → CopyBulkTensorTileG2SOp (CtaGroup.TWO)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        # 需要去掉 stage 维度
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0)) # ((128, 64), 1, 4)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # B 的 TMA load: CopyBulkTensorTileG2SOp(cta_group=<CtaGroup.TWO>)
        # 不需要显式 multicast 广播, 因为 CopyBulkTensorTileG2SOp(CtaGroup.TWO) 已经在两个 CTA 间分配了 B 数据
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0)) # ((128, 64), 1, 4)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # SFA 的 TMA load: CopyBulkTensorTileG2SOp(cta_group=<CtaGroup.TWO>)
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            cluster_shape_mnk[:2], tiled_mma.thr_id
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

        # SFB 的 TMA load: CopyBulkTensorTileG2SMulticastOp(cta_group=<CtaGroup.TWO>)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            cluster_shape_mnk[:2], tiled_mma.thr_id
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

        # 计算 TMA load 字节数
        a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout) # 16384
        b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout) # 16384
        sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout) # 2048
        sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout) # 4096
        # 每 stage 每 CTA: a=16384 + b=16384 + sfa=2048 + sfb=4096 = 38912 bytes
        # 两个 CTA 共 atom_thr_size=2, 总 TMA load 字节数 = 38912 * 2 = 77824 bytes/stage
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # 计算 grid size
        grid = cute.round_up(
            cute.ceil_div(
                (c_tensor.layout.shape),
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], 1),
            ),
            cluster_shape_mnk,
        )

        # Launch kernel
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
            c_tensor,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=cluster_shape_mnk,
            stream=stream,
        )
        return

    # GPU 设备 kernel
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
        mC_mnl: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        cta_layout_vmnk: cute.Layout,
        cta_layout_sfb_vmnk: cute.Layout,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        执行 batched GEMM 计算的 GPU device kernel.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()

        #
        # 设置 CTA/thread 坐标
        #
        # cluster 内坐标
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        cta_in_cluster_coord_sfb_vmnk = cta_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # cluster 外坐标
        # cta_layout_vmnk = ((2),1,1,1):((1),0,0,0)  →  size([0])=2, V 维含 leader/follower
        mma_tile_coord_vmnk = (
            bidx % cute.size(cta_layout_vmnk, mode=[0]),
            bidx // cute.size(cta_layout_vmnk, mode=[0]),
            bidy,
            bidz,
        )
        mma_tile_coord_mnl = mma_tile_coord_vmnk[1:]
        is_leader_cta = mma_tile_coord_vmnk[0] == 0

        #
        # 定义 kernel 的 shared storage
        #
        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        # 分配 SMEM
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # (MMA, MMA_M, MMA_K, STAGE) = ((128,64),1,4,5) with S<3,4,3>  → 每 stage 128x256 FP4 = 16KB
        sA = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE) = ((128,64),1,4,5) with S<3,4,3>  → 每 stage 128x256 FP4 = 16KB (2CTA 减半)
        sB = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # (MMA, MMA_M, MMA_K, STAGE) = ((((32,4),1),(16,4)),1,4,5)  → 每 stage 128x16 FP8 = 2KB
        sSFA = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        # (MMA, MMA_N, MMA_K, STAGE) = ((((32,4),2),(16,4)),1,4,5)  → 每 stage 256x16 FP8 = 4KB (完整, 不减半)
        sSFB = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        #
        # TMA multicast mask 是 16-bit 位图, bit i 置位表示 TMA 将数据写入 cluster 内 rank=i 的 CTA 的 SMEM.
        # 多个 bit 置位 → 一次 GMEM 读取同时写入多个 CTA (multicast), 减少 L2 流量.
        # 单个 bit 置位 → 仅写入自身 CTA, 等效于无 multicast.
        #
        # create_tma_multicast_mask 的 mcast_mode 指定沿 cta_layout 的哪个维度广播:
        #   mcast_mode=2 (N 维) → A/SFA. cta_layout_vmnk N 维 size=1, 无可广播的 CTA.
        #   mcast_mode=1 (M 维) → B/SFB:
        #     B 用 cta_layout_vmnk, M 维 size=1 (被 2CTA V 消耗) → 无 multicast, B 靠 DSMEM 跨 CTA 共享.
        #     SFB 用 cta_layout_sfb_vmnk, M 维 size=2 → 真正 multicast, 广播到 2 个 CTA.
        #
        # 2x1 cluster 实际 mask 值:
        #   CTA 0: a=0b01(1) b=0b01(1) sfa=0b01(1) sfb=0b11(3)
        #   CTA 1: a=0b10(2) b=0b10(2) sfa=0b10(2) sfb=0b11(3)
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(
            self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs
        ):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
            )  # CTA0=1, CTA1=2, 单 CTA
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
            )  # CTA0=1, CTA1=2, 单 CTA
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
            )  # CTA0=1, CTA1=2, 单 CTA
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_sfb_vmnk, cta_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )  # CTA0=3, CTA1=3, 真正 multicast

        #
        # 初始化 mainloop 的 ab_pipeline, acc_pipeline 及其状态
        #
        # num_tma_producer 表示"有多少个 CTA 的 TMA 会写入本 CTA 的 SMEM", 用于设置
        # consumer 侧 mbarrier 的 arrive_count. 公式: mcast_ctas_a + mcast_ctas_b - 1.
        #   - A 沿 N 维 multicast: mcast_ctas_a 个远程 CTA 的 TMA 会写入本 CTA 的 sA.
        #   - B 沿 M 维 multicast: mcast_ctas_b 个远程 CTA 的 TMA 会写入本 CTA 的 sB.
        #   - -1: 本 CTA 同时负责 A 和 B 的 TMA load, 只 arrive 一次.
        # 本例中 mcast_ctas_a=1, mcast_ctas_b=1, 结果 = 1.
        #
        # B 的 2CTA DSMEM 共享不影响此值: 每个 CTA 独立 TMA load 自己那一半 B,
        # MMA 通过 DSMEM 读取 peer 的 SMEM, 不涉及远程 TMA 写入.
        #
        # SFB 的 TMA multicast 也不影响此值: 远程 CTA 的 TMA 写入本地 SMEM 时,
        # 硬件通过 tx_count (arrive_and_expect_tx) 机制追踪字节完成量, 不额外增加 arrive_count.
        #
        # ab 5 stages, acc 1 stage, tx_count = 77824 bytes/stage
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_cta * (2 if self.use_2cta_instrs else 1),
            ),
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        #
        # Local_tile 切分 global tensors
        #
        # (bM, bK, RestM, RestK, RestL) = (256,256,1,1,1) → 每 tile 256x256 的 A 块
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL) = (256,256,?,?,?) → 每 tile 256x256 的 B 块
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # ((32,4,2),(16,4,4),RestM,RestK,(1,RestL))
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # ((32,4,2),(16,4,4),RestN,RestK,(1,RestL))
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL) = (256,256,?,?,?)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])  # 动态值, K/mma_tiler_K (K=256 时 =1)

        #
        # 为 TiledMMA_A/B/SFA/SFB/C 切分 global tensor
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_vmnk[0])
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_vmnk[0])
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL) = ((128,64),1,4,?,?,?)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL) = ((128,64),1,4,?,?,?)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL) = (((32,4),(16,4)),1,4,?,?,(1,?))
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL) = (((32,4,2),(16,4)),1,4,?,?,(1,?)), 用 thr_mma_sfb
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL) = ((128,256),1,1,?,?,?)  → 每 CTA 128x256
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # 为 TMA load A/B/SFA/SFB 切分 global/shared tensor
        #
        # tAsA: ((atom_v, rest_v), STAGE) = ((32768,1),5)  → SMEM 侧, 每 stage 16KB
        # tAgA: ((atom_v, rest_v), RestM, RestK, RestL) = (((256,128),1),?,?,?)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            # 0,
            # cute.make_layout(1),
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # tBsB: ((atom_v, rest_v), STAGE) = ((32768,1),5)  → SMEM 侧, 每 stage 16KB
        # tBgB: ((atom_v, rest_v), RestN, RestK, RestL) = (((256,128),1),?,?,?)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            # 0,
            # cute.make_layout(1),
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # tAsSFA: ((atom_v, rest_v), STAGE) = ((2048,1),5)  → SMEM 侧, 每 stage 2KB
        # tAgSFA: ((atom_v, rest_v), RestM, RestK, RestL) = (((512,4),1),?,?,(1,?))
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            # 0,
            # cute.make_layout(1),
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # tBsSFB: ((atom_v, rest_v), STAGE) = ((4096,1),5)  → SMEM 侧, 每 stage 4KB
        # tBgSFB: ((atom_v, rest_v), RestN, RestK, RestL) = (((512,4,2),1),?,?,(1,?))
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            cta_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # 为 TiledMMA_A/B/C 切分 shared/tensor memory tensor
        #
        # (MMA, MMA_M, MMA_K, STAGE) = (1,1,4,5):(0,0,2,1024) → SMEM descriptor, 4 K blocks x 5 stages
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE) = (1,1,4,5):(0,0,2,1024)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N) = ((128,256),1,1)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N) = ((128,256),1,1):((65536,1),0,0) → TMEM layout, 128 行 x 256 列
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        #
        # 分配 TMEM buffer
        #
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        # 2CTA 模式下两个 CTA 共享 TMEM, 需传入 is_two_cta 和 dealloc barrier
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        # tCtAcc: TMEM f32, ((128,256),1,1):((65536,1),0,0), 128 行 x 256 列
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        #
        # 构造 SFA/SFB 的 TMEM tensor
        #
        # 获取 SFA tmem ptr
        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
            dtype=sf_dtype,
        )
        # (MMA, MMA_M, MMA_K): tCtSFA_layout = ((((32,4),4),(16,4)),1,4) → 128 行 x 4 K blocks
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        # tCtSFA: TMEM f8E4M3FN, ((((32,4),4),(16,4)),1,4)
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        # 获取 SFB tmem ptr
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=sf_dtype,
        )
        # (MMA, MMA_N, MMA_K): tCtSFB_layout = ((((32,8),4),(16,4)),1,4) → 256 行 x 4 K blocks
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        # tCtSFB: ((((32,8),4),(16,4)),1,4):... → 256 行 (32*8) x 4 K blocks
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        #
        # 为 SFA/SFB 的 S2T copy 做 partition
        #
        # 构造 S2T CopyAtom (SMEM → TMEM), 底层对应 tcgen05.cp PTX 指令
        # 形状 4x32x128b 含义:
        #   4    = 重复 4 次, 覆盖 4 个 K-block (MMA_K=4)
        #   32   = TMEM 行粒度, tcgen05.cp 以 32 行为单位搬运
        #   128b = 每行 16 bytes, 可容纳 16 个 f8 scale factor
        # 单次 atom 搬运量: 4 x 32 x 16B = 2048B, 恰好等于 SFA 每 stage 的数据量
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(
                tcgen05.CtaGroup.ONE
                if not self.use_2cta_instrs
                else tcgen05.CtaGroup.TWO
            ),
            sf_dtype,
        )
        # filter_zeros 压缩 layout, 去除 stride=0 的退化维度, 让后续 copy 分区能正确计算实际数据量
        # (MMA, MMA_MN, MMA_K, STAGE) = ((((32,4),1),(1,4)),1,4,5) → filter 后的 SMEM SFA
        tCsSFA_compact = cute.filter_zeros(sSFA)
        # (MMA, MMA_MN, MMA_K) = ((((32,4),4),(1,4)),1,4) → filter 后的 TMEM SFA
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        # 用 copy atom (4x32x128b) 铺满 TMEM SFA, 构造 TiledCopy
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        # 按 S2T copy atom 视角分区 SMEM 源 / TMEM 目标
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
        # 将 SMEM 分区转为 64-bit SMEM descriptor, tcgen05.cp 指令要求的源操作数格式
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE) = ((((32,1,1),4),1),1,1,4,5)
        tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K) = (((32,16,4),1),1,1,4)
        tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

        # SFB 同理, Rest_Tiler=2 因为 N=256 行需要两轮 atom 才能覆盖 (SFA 的 M=128 只需一轮)
        # (MMA, MMA_MN, MMA_K, STAGE) = ((((32,4),2),(1,4)),1,4,5) → filter 后的 SMEM SFB
        tCsSFB_compact = cute.filter_zeros(sSFB)
        # (MMA, MMA_MN, MMA_K) = ((((32,8),4),(1,4)),1,4) → filter 后的 TMEM SFB
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE) = ((((32,1,1),4),1),2,1,4,5)
        tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K) = (((32,16,4),1),2,1,4)
        tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

        #
        # 按 MMA tile index 进行 slice, 固定 M/N tile + batch, 只留 K 维待迭代
        #
        # ((atom_v, rest_v), RestK) = (((256,128),1),?)
        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        # ((atom_v, rest_v), RestK) = (((256,128),1),?)
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        # ((atom_v, rest_v), RestK) = (((512,4),1),?)
        tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        # ((atom_v, rest_v), RestK) = (((512,4,2),1),?)
        tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        #
        # 在 k_tile 循环中执行 Data copy 与 Math 计算
        #
        if warp_idx == 0:
            # 2CTA 中仅 leader CTA 控制 acc pipeline 和 MMA 发射
            if is_leader_cta:
                acc_producer.acquire_and_advance()

            # 首次 k_tile 迭代时, 将 ACCUMULATE 字段设为 False
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            # 执行 k_tile 循环
            for k_tile in cutlass.range(
                k_tile_cnt, prefetch_stages=self.num_ab_stage - 2
            ):
                # 等待 AB buffer 为空
                ab_empty = ab_producer.acquire_and_advance()

                # A/B/SFA/SFB 的 TMA load, 传入 mcast_mask 启用 multicast
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=b_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, ab_empty.count)],
                    tAsSFA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=sfa_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB[(None, ab_empty.count)],
                    tBsSFB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=sfb_full_mcast_mask,
                )

                if is_leader_cta:
                    # 等待 AB buffer 满
                    ab_full = ab_consumer.wait_and_advance()

                    # 将 SFA/SFB 复制到 TMEM
                    s2t_stage_coord = (None, None, None, None, ab_full.index)
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
                            ab_full.index,
                        )

                        # 将 SFA/SFB tensor 设置到 `tiled_mma`
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
                        # 首个 kblock 之后在 `tCtAcc` 上启用 accumulate
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    # 异步 arrive AB buffer 空
                    ab_full.release()
            if is_leader_cta:
                acc_producer.commit()

        #
        # Epilogue
        # 为 epilogue 做 partition
        #
        # x32 或 x128 均可.
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R_M, T2R_N, EPI_M, EPI_N) = (((128,32),1),2,1,1) → TMEM acc 的线程视图
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
        # (T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL) = ((128,1),2,1,1,?,?,?)
        tTR_gC = thr_copy_t2r.partition_D(tCgC)
        # (T2R_M, T2R_N, EPI_M, EPI_N) = ((128,1),2,1,1) → register buffer (f32)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
        )
        # (T2R_M, T2R_N, EPI_M, EPI_N) = ((128,1),2,1,1) → register buffer (f16)
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, c_dtype
        )
        # STG Atom
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)

        # tTR_gC slice 前: ((128,1),2,1,1,RestM,RestN,RestL), slice 后: ((128,1),2,1,1)
        tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]

        # 等待 accumulator buffer 满
        acc_full = acc_consumer.wait_and_advance()

        # 将 accumulator 复制到 register
        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        acc_vec = epilogue_op(tTR_rAcc.load().to(c_dtype))
        tTR_rC.store(acc_vec)
        # 将 C store 到 global memory
        cute.copy(simt_atom, tTR_rC, tTR_gC)

        acc_full.release()

        # 2CTA 新增: producer 退出前同步, 防止 leader CTA 过早退出
        # 导致 follower CTA 访问已释放的 dsmem
        if warp_idx == 0:
            ab_producer.tail()
            if is_leader_cta:
                acc_producer.tail()

        # 释放 TMEM
        cute.arch.barrier()
        tmem.free(acc_tmem_ptr)

        return


def run_nvfp4_gemm(
    mnkl: Tuple[int, int, int, int],
    tolerance: float,
    warmup_iterations: int = 10,
    iterations: int = 100,
    use_cold_l2: bool = True,
    do_benchmark: bool = False,
):
    run(
        gemm_class=Sm100BlockScaledDenseGemmKernel,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mnk=cluster_shape_mnk,
        mnkl=mnkl,
        tolerance=tolerance,
        do_benchmark=do_benchmark,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cold_l2=use_cold_l2,
    )


if __name__ == "__main__":
    parser = create_parser()
    parser.set_defaults(mnkl=(256, 256, 256, 1))
    args = parser.parse_args()
    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    m, n, k, _ = args.mnkl
    if m % mma_tiler_mn[0] != 0:
        parser.error("M must be multiples of mma_tiler_mn[0] (got m={})".format(m))
    if n % mma_tiler_mn[1] != 0:
        parser.error("N must be multiples of mma_tiler_mn[1] (got n={})".format(n))
    if k % 256 != 0:
        parser.error("k must be a multiple of 256 (got k={})".format(k))

    run_nvfp4_gemm(
        args.mnkl,
        args.tolerance,
        do_benchmark=args.do_benchmark,
    )
    print("PASS")
