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
import os
from typing import Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.distributed as dist
import cuda.bindings.driver as cuda
try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device
from cuda.pathfinder import load_nvidia_dynamic_lib

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Float16, BFloat16, Float32, Float8E4M3FN, Float8E5M2
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm

try:
    import nvshmem.core
except ImportError as exc:
    raise ImportError(
        "nvshmem4py is required but not installed. Please install it using:\n"
        "  For CUDA 12: pip install nvshmem4py-cu12\n"
        "  For CUDA 13: pip install nvshmem4py-cu13\n"
        "Note: nvshmem4py version >= 0.1.3 is recommended."
    ) from None

try:
    load_nvidia_dynamic_lib("nvshmem_host")
except RuntimeError as exc:
    raise ImportError(
        "nvshmem lib is required but not installed. Please install it using:\n"
        "  For CUDA 12: pip install nvidia-nvshmem-cu12\n"
        "  For CUDA 13: pip install nvidia-nvshmem-cu13\n"
    ) from None

"""
使用 CUTE DSL、面向 NVIDIA Blackwell SM100 架构的高性能 distributed persistent batched dense GEMM 示例.
- Matrix A 为 MxKxL, L 是 batch dimension, A 可以是 row-major ("K") 或 column-major ("M")
- Matrix B 为 NxKxL, L 是 batch dimension, B 可以是 row-major ("N") 或 column-major ("K")
- Matrix C 为 MxNxL, L 是 batch dimension, C 可以是 row-major ("N") 或 column-major ("M")
- Matrix C_mc 是 multicast C matrix, 可通过 multimem instructions 将修改 broadcast 到所有 GPUs.

这个 GEMM kernel 支持 TMA、Blackwell tcgen05.mma (包括 2CTA MMA instructions)、
结合 cluster 的 TMA multicast、persistent tile scheduling 和 warp specialization.
它还支持使用 multimem instructions 的 reduce-scatter epilogue, 将工作分配到所有 GPUs.

GEMM 的执行过程如下:
1. DMA warp: 使用 TMA 将 matrices A 和 B 从 global memory (GMEM) 加载到 shared memory (SMEM).
2. MMA warp: 使用 tcgen05.mma instruction 执行 matrix multiply-accumulate (MMA).
3. EPILOGUE warp:
    - 使用 tcgen05.ld 将完成的 accumulator 从 tensor memory (TMEM) 加载到 registers (RMEM).
    - 将 matrix C 转换为 output type.
    - 可以通过 TMA 将 matrix C 从 registers (RMEM) 写入 shared memory (SMEM), 再写入 global memory (GMEM),
      也可以不使用 TMA, 直接将 matrix C 从 RMEM 写入 GMEM.
    - 可以传入 elementwise lambda function `epilogue_op` 作用于 output tensor:
      例如 relu 可设为 `lambda x: cute.where(x > 0, x, cute.full_like(x, 0))`.
4. Reduce-scatter epilogue:
    - 使用 multimem instructions 从所有 ranks 加载并归约 128-bit 数据
    - 使用 multimem instructions 将归约结果写入对应 rank 的 output 分片
    - 当前仅支持 two_shot, 即每个 rank 只计算 output tensor 的一部分
    - 仅当 use_tma_store=True 时支持 reduce-scatter epilogue
    - c_dtype 仅支持 Float16、Float32、BFloat16、Float8E4M3FN、Float8E5M2

SM100 tcgen05.mma instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

本示例的 input arguments 与 dense_gemm.py 相同.

.. code-block:: bash

    torchrun --nproc-per-node 8 examples/distributed/distributed_gemm_reduce_scatter_blackwell.py  \
      --ab_dtype Float8E4M3FN --c_dtype Float16 --acc_dtype Float32                                                 \
      --mma_tiler_mn 256,256 --cluster_shape_mn 2,1                                                            \
      --mnkl 16384,4080,4096,1 --warmup_iterations 3 --iterations 10                                                                               \
      --use_tma_store --use_2cta_instrs --reduce_scatter two_shot 

使用 NSYS profiler 收集性能数据:

.. code-block:: bash

    nsys profile  --gpu-metrics-devices=cuda-visible                                                             \
      torchrun --nproc-per-node 8 examples/distributed/distributed_gemm_reduce_scatter_blackwell.py  \
      --ab_dtype Float8E5M2 --c_dtype Float16 --acc_dtype Float32                                                   \
      --mma_tiler_mn 256,256 --cluster_shape_mn 2,1                                                              \
      --mnkl 16384,4096,4096,1                                                                                    \
      --use_tma_store --use_2cta_instrs --warmup_iterations 3 --iterations 10                                    \
      --reduce_scatter two_shot


约束与 dense_gemm_persistent.py 相同:
* 支持的 input data types: fp16、bf16、tf32、int8、uint8、fp8 (e4m3fn、e5m2),
  有效 dtype 组合详见下方 PersistentDenseGemmKernel class 文档
* A/B tensors 必须使用相同 data type
* MMA tiler M 必须为 64/128 (use_2cta_instrs=False) 或 128/256 (use_2cta_instrs=True)
* MMA tiler N 必须在 32-256 范围内, 步长为 32
* Cluster shape M/N 必须是正的 2 的幂, cluster size 总数不超过 16
* use_2cta_instrs=True 时, Cluster shape M 必须是 2 的倍数
* A/B/C tensors 的 contiguous dimension 必须至少按 16 bytes 对齐,
  即对 TFloat32、Float16/BFloat16、Int8/Uint8/Float8,
  elements 数量必须分别是 4、8、16 的倍数.
* 禁用 TMA store 时不允许 OOB tiles
* 使用 reduce_scatter 时, M 必须是 128 的倍数, world_size 必须为 2、4 或 8

"""


class PersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x B) with support for various data types
    并使用 Blackwell GPU 特有的 persistent tile scheduling 和 warp specialization.

    :param acc_dtype: 计算期间用于 accumulation 的 data type
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: 是否使用 CTA group 2 进行 thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Matrix Multiply-Accumulate (MMA) tile 的 shape (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: 用于 parallel processing 的 cluster dimensions (M,N)
    :type cluster_shape_mn: Tuple[int, int]
    :param use_tma_store: 是否使用 Tensor Memory Access (TMA) 写出结果
    :type use_tma_store: bool
    :param reduce_scatter: reduce scatter mode, can be "two_shot"
    :type reduce_scatter: str

    :note: 当前版本要求 A 和 B tensors 使用相同 data type
        - 例如不支持 A 为 Float8E4M3FN、B 为 Float8E5M2

    :note: 支持的 A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: 支持的 accumulator data types:
        - Float32, 支持所有 floating-point A/B data types
        - Float16, 仅支持 fp16 和 fp8 A/B data types
        - Int32, 仅支持 uint8/int8 A/B data types

    :note: 支持的 C data types:
        - Float32, 用于 float32 和 int32 accumulator data types
        - Int32, 用于 float32 和 int32 accumulator data types
        - Float16/BFloat16, 用于 fp16 和 fp8 accumulator data types
        - Int8/Uint8, 用于 uint8/int8 accumulator data types
        - Float8E4M3FN/Float8E5M2, 用于 float32 accumulator data types

    :note: 约束:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    示例:
        >>> gemm = PersistentDenseGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     use_2cta_instrs=True,
        ...     mma_tiler_mn=(128, 128),
        ...     cluster_shape_mn=(2, 2)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        reduce_scatter="two_shot",
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        配置包括以下部分:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3. Output C tensor store mode:
            - use_tma_store: Boolean indicating whether to use Tensor Memory Access (TMA) for storing results.

        :param acc_dtype: accumulator 的 data type.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: MMA instruction 的 Tuple (M,N) shape.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: cluster 的 Tuple (ClusterM,ClusterN) shape.
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Use Tensor Memory Access (TMA) or normal store for output C tensor.
        :type use_tma_store: bool
        :param reduce_scatter: reduce scatter mode, can be "two_shot"
        :type reduce_scatter: str
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension 延后到 _setup_attributes 中确定
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.reduce_scatter = reduce_scatter

        self.occupancy = 1
        # 设置 specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.reduce_scatter = reduce_scatter
        self.reduce_scatter_warp_id = (6, 7, 8, 9)
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                *self.epilog_warp_id,
                *self.reduce_scatter_warp_id,
            )
        )
        # 设置用于 CTA sync、epilogue sync 和 TMEM pointer sync 的 barrier id
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.reduce_scatter_sync_bar_id = 3
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        self.num_ranks = 1
        self.rank_id = 0
        self.num_ranks = torch.distributed.get_world_size()
        self.rank_id = torch.distributed.get_rank()

    def is_valid(self):
        mma_m, mma_n = self.mma_tile_shape_mn
        if (mma_m // (2 if self.use_2cta_instrs else 1)) not in [64, 128]:
            return False
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            return False
        if self.cluster_shape_mn[0] == 4 and self.cluster_shape_mn[1] == 4:
            return False
        return True

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        这个 method 根据 input tensor 属性
        (data types、leading dimensions) 和 kernel settings 配置以下属性:
        - 配置 tiled MMA
        - 计算 MMA/cluster/tile shapes
        - 计算 cluster layout
        - 计算 A/B 的 multicast CTAs
        - 计算 epilogue subtile
        - 设置 A/B/C 在 SMEM 中的 stage 数量
        - 计算 A/B/C 的 SMEM layout
        - 计算 TMEM allocation columns
        """
        # 配置 tiled MMA
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # 计算 MMA、cluster 和 tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # 计算 cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # 计算 A/B 的 multicast CTA 数量
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # 计算 epilogue subtile
        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

        # 设置 A/B/C 在 SMEM 中的 stage 数量, 以及 ACC 在 TMEM 中的 stage 数量
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
        )

        # 计算 A/B/C 的 SMEM layout
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
        self.c_smem_layout_staged = (
            sm100_utils.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                self.num_c_stage,
            )
            if self.use_tma_store
            else None
        )

        # 计算 TMEM 分配的 columns 数量
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_mc: cute.Tensor = None,
        c_peer_tensors: list = None,
        barrier_flag: cute.Tensor = None,
        barrier_flag_mc: cute.Tensor = None,
    ):
        """Execute the GEMM operation in steps:
        - 在计算 SMEM/grid/TMA 前设置静态属性
        - 设置 TMA load/store atoms 和 tensors
        - 根据 hardware constraints 计算 grid size
        - 定义 kernel 的 shared storage
        - 同步启动 kernel

        :param a: input tensor A
        :type a: cute.Tensor
        :param b: input tensor B
        :type b: cute.Tensor
        :param c: output tensor C
        :type c: cute.Tensor
        :param c_mc: symmetric output tensor C_mc; 对 multicast tensor 的读写会 broadcast 到所有 GPUs
        :type c_mc: cute.Tensor
        :param c_peer_tensors: List of peer tensors for all-reduce
        :type c_peer_tensors: List[cute.Tensor]
        :param barrier_flag: Barrier flag for all-reduce
        :type barrier_flag: cute.Tensor
        :param barrier_flag_mc: Barrier flag for multicast tensor
        :type barrier_flag_mc: cute.Tensor
        :param max_active_clusters: active clusters 的最大数量
        :type max_active_clusters: cutlass.Constexpr
        :param stream: 用于 asynchronous execution 的 CUDA stream
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.
        """
        # 在计算 SMEM、grid 和 TMA 前设置静态属性
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # 检查 input data types 是否与 MMA instruction 兼容
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # 设置依赖 GEMM inputs 的属性
        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # 配置 A 的 TMA load
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        # 配置 B 的 TMA load
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # 配置 C 的 TMA store
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                epi_smem_layout,
                self.epi_tile,
            )

        # 计算 grid size
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.buffer_align_bytes = 1024

        c_smem_size = (
            cute.cosize(self.c_smem_layout_staged.outer) if self.use_tma_store else 0
        )

        # 定义 kernel 的 shared storage
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
                    c_smem_size,
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

        self.shared_storage = SharedStorage

        # 同步启动 kernel
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            c_mc,
            c_peer_tensors,
            barrier_flag,
            barrier_flag_mc,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        c_mc: cute.Tensor,
        c_peer_tensors: list,
        barrier_flag: cute.Tensor,
        barrier_flag_mc: cute.Tensor,
    ):
        """
        执行 persistent batched GEMM 计算的 GPU device kernel.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # 预取 TMA descriptors
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # 设置 CTA/thread coordinates
        #
        # cluster 内的 coordinates
        bidx, bidy, bidz = cute.arch.block_idx()

        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # CTA 内的 coordinate
        tidx, _, _ = cute.arch.thread_idx()

        #
        # 分配并初始化: A+B full/empty、accumulator full/empty 和 TMEM dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar.ptr
        tmem_holding_buf = storage.tmem_holding_buf.ptr

        # 初始化 mainloop ab_pipeline (barrier) 和 states
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
        )

        # 初始化 acc_pipeline (barrier) 和 states
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
        )

        # 初始化 TMEM dealloc barrier
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )
        cute.arch.mbarrier_init_fence()

        # barrier 初始化后执行 cluster arrive
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # 设置 SMEM tensors A/B/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = (
            storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
            if self.use_tma_store
            else None
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        #
        # 计算 A/B buffer full 的 multicast mask
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # 使用 local_tile 划分 global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # 为 TiledMMA_A/B/C 划分 global tensor
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # 为 TMA load A/B 划分 global/shared tensor
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
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # 为 TiledMMA_A/B/C 划分 SMEM/TMEM tensor
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
        # 分配 TMEM 前执行 cluster wait
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling 循环
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # 从 tile scheduler 获取 tile coordinate
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # 按 MMA tile index 切片
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # 通过 try_wait 检查 k_tile=prefetch_k_tile_cnt 的 AB buffer empty
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
                    # 按条件等待 AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # 执行 A/B 的 TMA load
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

                    # 通过 try_wait 检查下一个预取 k_tile 的 AB buffer empty
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # 推进到下一个 tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 等待 A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # 获取 SMEM 中的 TMEM pointer 前执行 barrier sync
            #
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # 获取 TMEM pointer 并创建 accumulator tensor
            #
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling 循环
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
                # 从 tile scheduler 获取 tile coordinate
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # 设置当前 tile 的 TMEM buffer
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # 通过 try_wait 检查 k_tile=0 的 AB buffer full
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # 等待 accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # 为每个 tile 重置 ACCUMULATE field
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # MMA mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # 按条件等待 AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            # 从第一个 kblock 之后启用 tCtAcc accumulate
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # 异步通知 AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # 通过 try_wait 检查下一个 k_tile 的 AB buffer full
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # 异步通知 accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # 推进到下一个 tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 等待 accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # 分配 TMEM buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # 获取 SMEM 中的 TMEM pointer 前执行 barrier sync
            #
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # 获取 TMEM pointer 并创建 accumulator tensor
            #
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # 为 epilogue 执行 partition
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = None
            tiled_copy_r2s = None
            simt_atom = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_gC_partitioned = None
            if cutlass.const_expr(self.use_tma_store):
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
            else:
                (
                    simt_atom,
                    tTR_rC,
                    tTR_gC_partitioned,
                ) = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tiled_copy_t2r, tCgC, epi_tile, sC
                )

            #
            # Persistent tile scheduling 循环
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_pipeline = None
            if cutlass.const_expr(self.use_tma_store):
                # 参与 TMA store pipeline 的 threads/warps
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilog_warp_id)
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )

            while work_tile.is_valid_tile:
                # 从 tile scheduler 获取 tile coordinate
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # 按 MMA tile index 切片
                #
                bSG_gC = None
                tTR_gC = None
                if cutlass.const_expr(self.use_tma_store):
                    # ((ATOM_V, REST_V), EPI_M, EPI_N)
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]
                else:
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                    tTR_gC = tTR_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]

                # 设置当前 tile 的 TMEM buffer
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                #
                # 等待 accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                if cutlass.const_expr(self.use_tma_store):
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                else:
                    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

                #
                # 按 subtiles 将 accumulator 写入 global memory
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    #
                    # 将 accumulator 从 TMEM buffer 加载到 registers
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    if cutlass.const_expr(self.use_tma_store):
                        #
                        # 转换为 C type
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
                        # 使用 fence 和 barrier, 确保 SMEM store 对 TMA store 可见
                        cute.arch.fence_proxy("async.shared", space="cta")
                        epilog_threads = 32 * len(self.epilog_warp_id)
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                        #
                        # 通过 TMA 将 C 写入 global memory
                        #
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            # 使用 fence 和 barrier, 确保 SMEM store 对 TMA store 可见
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )
                    else:
                        #
                        # 转换为 C type
                        #
                        acc_vec = tTR_rAcc.load()
                        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                        tTR_rC.store(acc_vec)

                        #
                        # 将 C 写入 global memory
                        #
                        cute.copy(
                            simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)]
                        )

                #
                # 异步通知 accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # Reduce-Scatter
                if cutlass.const_expr(self.reduce_scatter == "two_shot"):
                    tile_id = Int32(
                        tile_sched._current_work_linear_idx
                        * cute.size(self.cluster_shape_mn) + cute.arch.block_idx_in_cluster()
                    )
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                        # 使用 system barrier, 确保执行 reduce-scatter 前各 GPU 的数据已写入 memory
                        with cute.arch.elect_one():
                            flag = barrier_flag_mc.iterator + tile_id
                            utils.distributed.multimem_red_add1(
                                lock_ptr=flag,
                                scope="gpu",
                                order="release",
                            )

                #
                # 推进到下一个 tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 释放 TMEM buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            #
            # 等待 C store 完成
            #
            if cutlass.const_expr(self.use_tma_store):
                c_pipeline.producer_tail()

        # ///////////////////////////////////////////////////////////////////////////////
        # Reduce-Scatter warps
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.reduce_scatter == "two_shot"):
            if warp_idx >= self.reduce_scatter_warp_id[0]:
                # ///////////////////////////////////////////////////////////////////////////////
                # 添加 persistent tile 循环
                # ///////////////////////////////////////////////////////////////////////////////

                rank_id = self.rank_id
                num_ranks = Int32(self.num_ranks)
                lane_id = cute.arch.lane_idx()

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                # 使用 128-bit load/store 以获得更好性能
                atom_val = 128 // c_mc.element_type.width
                atom_thr_n = self.mma_tiler[1] // atom_val
                atom_thr_m = len(self.reduce_scatter_warp_id) * cute.arch.WARP_SIZE // atom_thr_n
                thr_layout = cute.make_layout((atom_thr_m, atom_thr_n), stride=(atom_thr_n, 1))
                val_layout = cute.make_layout((1, atom_val), stride=(atom_val, 1))

                copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_mc.element_type)
                tiled_copy_fake = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
                thr_copy_fake = tiled_copy_fake.get_slice(tidx-self.reduce_scatter_warp_id[0]*32)

                # predicate tensor
                idC = cute.make_identity_tensor(c_mc.shape)
                
                # 在 tile 级别执行 partition 和 slice
                gC_mc = cute.local_tile(
                    c_mc, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
                )
                cC = cute.local_tile(
                    idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
                )

                m_tiles_in_total = gC_mc.shape[2]
                n_tiles_in_total = gC_mc.shape[3]
                m_tiles_per_rank = m_tiles_in_total // self.num_ranks

                while work_tile.is_valid_tile:

                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_mnl = (
                        ((cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)) % m_tiles_per_rank) + self.rank_id * m_tiles_per_rank,
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )

                    chunk_id = ((cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)) // m_tiles_per_rank)
                    tile_id = mma_tile_coord_mnl[0] + mma_tile_coord_mnl[1] * m_tiles_in_total
                    tile_id = tile_id * cute.size(tiled_mma.thr_id.shape)

                    if not is_leader_cta:
                        tile_id = tile_id + 1

                    # 使用 system barrier, 确保执行 Reduce-Scatter 前各 GPU 的数据已写入 memory
                    flag = barrier_flag.iterator + tile_id
                    
                    if warp_idx == self.reduce_scatter_warp_id[0]:
                        if lane_id == 0:
                            res = 0
                            while res < self.num_ranks:
                                res = cute.arch.load(flag.llvm_ptr, cutlass.Int32, sem="relaxed", scope="gpu")
                    cute.arch.barrier(
                        barrier_id=self.reduce_scatter_sync_bar_id,
                        number_of_threads=32 * len(self.reduce_scatter_warp_id),
                    )
                    if warp_idx == self.reduce_scatter_warp_id[0]:
                        if lane_id == 0:
                            res = cute.arch.atomic_add(
                                flag.llvm_ptr,
                                Int32(1),
                                sem="relaxed",
                                scope="sys",
                            )
                            res = cute.arch.load(flag.llvm_ptr, cutlass.Int32, sem="relaxed", scope="sys")
                            if res == self.num_ranks*2:
                                cute.arch.store(flag.llvm_ptr, Int32(0), sem="relaxed", scope="sys")
                    tCgC_mc = thr_mma.partition_C(gC_mc)
                    tCpC = thr_mma.partition_C(cC)

                    tCgC_mc_slice = tCgC_mc[((None, None), 0, 0, *mma_tile_coord_mnl)]
                    tCpC_slice = tCpC[((None, None), 0, 0, *mma_tile_coord_mnl)]

                    cta_mma_tile_m = self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape)
                    m_local_rank = int(cta_mma_tile_m / self.num_ranks)
                    tCgC_mc_slice_partitioned = cute.zipped_divide(tCgC_mc_slice, (m_local_rank, self.mma_tiler[1]))
                    tCpC_slice_partitioned = cute.zipped_divide(tCpC_slice, (m_local_rank, self.mma_tiler[1]))

                    tCgC_mc_local_rank = cute.slice_(tCgC_mc_slice_partitioned, ((None, None), (chunk_id, 0)))
                    tCpC_local_rank = cute.slice_(tCpC_slice_partitioned, ((None, None), (chunk_id, 0)))
                    frgC_mc = thr_copy_fake.partition_S(tCgC_mc_local_rank)
                    frpC = thr_copy_fake.partition_S(tCpC_local_rank)

                    m_idx = gC_mc.shape[0] * mma_tile_coord_mnl[0]
                    m_per_rank = c_mc.shape[0] // self.num_ranks
                    dst_rank_this_tile = m_idx // m_per_rank

                    peer = c_peer_tensors[rank_id]

                    gC_peer = cute.local_tile(
                      peer, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
                    )
                    tCgC_peer = thr_mma.partition_C(gC_peer)
                    tCgC_peer_slice = tCgC_peer[((None, None), 0, 0, *mma_tile_coord_mnl)]
                    cta_mma_tile_m = self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape)
                    m_local_rank = int(cta_mma_tile_m / self.num_ranks)
                    tCgC_peer_slice_partitioned = cute.zipped_divide(tCgC_peer_slice, (m_local_rank, self.mma_tiler[1]))
                    tCgC_peer_local_rank = cute.slice_(tCgC_peer_slice_partitioned, ((None, None), (chunk_id, 0)))
                    frgC_peer = thr_copy_fake.partition_S(tCgC_peer_local_rank)

                    atom, loop_m, loop_n = frgC_mc.shape
                    tmp_results = cute.make_rmem_tensor((4, loop_m, loop_n), cutlass.Int32)
                    local_chunk_lower_bound = (rank_id * m_per_rank, c_mc.shape[1], c_mc.shape[2])
                    local_chunk_upper_bound = ((rank_id + 1) * m_per_rank, c_mc.shape[1], c_mc.shape[2])

                    for i in cutlass.range_constexpr(loop_m):
                        for j in cutlass.range_constexpr(loop_n):
                            if cute.elem_less(frpC[0, i, j], local_chunk_upper_bound) and not cute.elem_less(frpC[0, i, j], local_chunk_lower_bound): 
                                mc_ptr = frgC_mc[None, i, j].iterator
                                if cutlass.const_expr(self.c_dtype == Float16):
                                    x, y, z, w = utils.distributed.multimem_ld_reduce_8xf16(mc_ptr)
                                elif cutlass.const_expr(self.c_dtype == Float32):
                                    x, y, z, w = utils.distributed.multimem_ld_reduce_4xf32(mc_ptr)
                                elif cutlass.const_expr(self.c_dtype == BFloat16):
                                    x, y, z, w = utils.distributed.multimem_ld_reduce_8xbf16(mc_ptr)
                                elif cutlass.const_expr(self.c_dtype == Float8E4M3FN):
                                    x, y, z, w = utils.distributed.multimem_ld_reduce_16xe4m3(mc_ptr)
                                elif cutlass.const_expr(self.c_dtype == Float8E5M2):
                                    x, y, z, w = utils.distributed.multimem_ld_reduce_16xe5m2(mc_ptr)
                                tmp_results[0, i, j] = x
                                tmp_results[1, i, j] = y
                                tmp_results[2, i, j] = z
                                tmp_results[3, i, j] = w
                            
                    for i in cutlass.range_constexpr(loop_m):
                        for j in cutlass.range_constexpr(loop_n):
                            if cute.elem_less(frpC[0, i, j], local_chunk_upper_bound) and not cute.elem_less(frpC[0, i, j], local_chunk_lower_bound):
                                ptr_int = frgC_peer[None, i, j].iterator.toint().ir_value()
                                x, y, z, w = tmp_results[0, i, j].ir_value(), tmp_results[1, i, j].ir_value(), tmp_results[2, i, j].ir_value(), tmp_results[3, i, j].ir_value()
                                llvm.inline_asm(
                                    T.i32(),
                                    [ptr_int, x, y, z, w],
                                    "st.global.sys.relaxed.v4.f32 [$1], {$2, $3, $4, $5};",
                                    "=r,l,r,r,r,r",
                                    has_side_effects=True,
                                    asm_dialect=0
                                )

                    # 推进到下一个 tile
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                cute.arch.barrier(
                    barrier_id=self.reduce_scatter_sync_bar_id,
                    number_of_threads=32 * len(self.reduce_scatter_warp_id),
                )
                # 使用 system barrier, 确保所有 peer memory transfers 均已完成.
                last_flag_idx = cute.size(
                    tile_sched.params.problem_layout_ncluster_mnl
                ) * cute.size(self.cluster_shape_mn)
                if warp_idx == self.reduce_scatter_warp_id[0]:
                    with cute.arch.elect_one():
                        # 最后一个 tile flag 的 offset
                        last_tile_id_linear = cute.size(
                            tile_sched.params.problem_layout_ncluster_mnl
                        ) * cute.size(self.cluster_shape_mn)
                        # 当前 SM 的 linear id.
                        sm_id_linear = (
                            cute.arch.block_idx()[0]
                            + cute.arch.block_idx()[1] * cute.arch.grid_dim()[0]
                            + cute.arch.block_idx()[2]
                            * cute.arch.grid_dim()[0]
                            * cute.arch.grid_dim()[1]
                        )
                        # 以 sys scope 执行 release flag
                        utils.distributed.multimem_red_add1(
                            lock_ptr=barrier_flag_mc.iterator
                            + last_tile_id_linear
                            + sm_id_linear,
                            scope="sys",
                            order="release",
                        )
                        # 以 sys scope 使用 relaxed spin-lock 等待 flag
                        utils.distributed.spin_lock_atom_cas_relaxed_wait(
                            lock_ptr=barrier_flag.iterator
                            + last_tile_id_linear
                            + sm_id_linear,
                            expected_val=self.num_ranks,
                            reset_val=0,
                            scope="sys",
                        )


    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        为 TMEM load 创建 tiledCopy, 再用它划分 TMEM source 和 register array destination.

        :param tidx: epilogue warp groups 中的 thread index
        :type tidx: cutlass.Int32
        :param tAcc: 待 copy 和 partition 的 accumulator tensor
        :type tAcc: cute.Tensor
        :param gC_mnl: global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: 是否启用 use_2cta_instrs
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # 为 TMEM load 创建 tiledCopy
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
        为 SMEM store 创建 tiledCopy, 再用它划分 register array source 和 SMEM destination.

        :param tiled_copy_t2r: 用于 TMEM 到 register copy (T2R) 的 tiled copy operation
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: partition 后的 accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: epilogue warp groups 中的 thread index
        :type tidx: cutlass.Int32
        :param sC: 待 copy 和 partition 的 SMEM tensor
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
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
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: epilogue warp groups 中的 thread index
        :type tidx: cutlass.Int32
        :param atom: TMA store 版本使用 copy_atom_c, 非 TMA store 版本使用 tiled_copy_t2r
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: 待 copy 和 partition 的 SMEM tensor
        :type sC: cute.Tensor

        :return: A tuple containing either:
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
            - For non-TMA store: (simt_atom, tTR_rC, tTR_gC) where:
                - simt_atom: The SIMT copy atom
                - tTR_rC: The register tensor C
                - tTR_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        if cutlass.const_expr(self.use_tma_store):
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
        else:
            tiled_copy_t2r = atom
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_gC = thr_copy_t2r.partition_D(gC_epi)
            # (T2R, T2R_M, T2R_N)
            tTR_rC = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.c_dtype
            )
            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
            return simt_atom, tTR_rC, tTR_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        smem_capacity: int,
        occupancy: int,
        use_tma_store: bool,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: 定义主要计算的 tiled MMA object.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: MMA tiler 的 shape (M, N, K).
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: operand A 的 data type.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: operand B 的 data type.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: operand C (output) 的 data type.
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: operand C 的 Layout enum.
        :type c_layout: utils.LayoutEnum
        :param smem_capacity: 可用 shared memory 总容量, 单位为 bytes.
        :type smem_capacity: int
        :param occupancy: 每个 SM 的目标 CTA 数量 (occupancy).
        :type occupancy: int
        :param use_tma_store: 是否启用 TMA store.
        :type use_tma_store: bool

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # 默认 ACC stages
        num_acc_stage = 2

        # 默认 C stages
        num_c_stage = 2 if use_tma_store else 0

        # 计算 A、B、C 单个 stage 的 SMEM layout 和大小
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        c_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(
                c_dtype,
                c_layout,
                epi_tile,
                1,
            )
            if use_tma_store
            else None
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = (
            cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            if use_tma_store
            else 0
        )
        c_bytes = c_bytes_per_stage * num_c_stage

        # 计算 A/B stages:
        # 从每个 CTA 可用的 SMEM 总量开始计算, 即 capacity / occupancy
        # 减去 reserved bytes 和初始 C stages bytes
        # 再除以每个 A/B stage 所需的 bytes
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # 调整 epilogue stages:
        # 计算分配 A/B stages 和 reserved bytes 后剩余的 SMEM
        # 把剩余未使用的 SMEM 分配给 epilogue
        if use_tma_store:
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
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: CTA tile 的 shape (M,N,K).
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: 每个 cluster 在 M、N dimensions 上的 shape.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: active clusters 的最大数量.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
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
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        计算 TMEM allocation columns 数量.

        :param tiled_mma: 定义主要计算的 tiled MMA object.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: MMA tile 的 shape (M,N,K).
        :type mma_tiler: tuple[int, int, int]
        :param num_acc_stage: accumulator tensor 的 stage 数量.
        :type num_acc_stage: int

        :return: TMEM allocation columns 数量.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        return num_tmem_alloc_cols

    def is_valid_dtypes(
        self,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        检查 dtypes 是否有效.

        :param ab_dtype: A 和 B operands 的 data type
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: accumulator 的 data type
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: output tensor 的 data type
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.TFloat32,
            cutlass.Uint8,
            cutlass.Int8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }
        if ab_dtype not in valid_ab_dtypes:
            return False

        if self.acc_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.Int32}:
            return False

        # 定义 accumulator type 与 AB type 的兼容关系
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.TFloat32,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },  # Float32 accumulator supports floating point AB types only
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        # 检查 accumulator type 与 AB type 是否兼容
        if ab_dtype not in acc_ab_compatibility[self.acc_dtype]:
            return False

        # 定义 accumulator type 与 C type 的兼容关系
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
            cutlass.Float16: {
                cutlass.BFloat16,
                cutlass.Float16,
            },
            cutlass.Int32: {
                cutlass.BFloat16,
                cutlass.Float16,
                cutlass.Float32,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        # 检查 accumulator type 与 C type 是否兼容
        if c_dtype not in acc_c_compatibility[self.acc_dtype]:
            return False

        # 检查 multimem all-reduce 是否支持 c_dtype
        if cutlass.const_expr(c_dtype not in {cutlass.Float16, cutlass.Float32, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}):
            return False

        return True

    def is_valid_mma_tiler_and_cluster_shape(self) -> bool:
        """Check if the mma tiler and cluster shape are valid.

        :return: MMA tiler 和 cluster shape 有效时返回 True, 否则返回 False
        :rtype: bool
        """
        is_valid = True
        # 排除无效的 MMA tile shape
        if not (
            (not self.use_2cta_instrs and self.mma_tiler_mn[0] in [64, 128])
            or (self.use_2cta_instrs and self.mma_tiler_mn[0] in [128, 256])
        ):
            is_valid = False
        if self.mma_tiler_mn[1] not in range(32, 257, 32):
            is_valid = False
        # 排除不合法的 cluster shape
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            is_valid = False
        # 排除无效的 cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16
            or self.cluster_shape_mn[0] <= 0
            or self.cluster_shape_mn[1] <= 0
            or not is_power_of_2(self.cluster_shape_mn[0])
            or not is_power_of_2(self.cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    def is_valid_tensor_alignment(
        self,
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
        检查 tensor alignment 是否有效.

        :param m: A tensor 的 rows 数量
        :type m: int
        :param n: B tensor 的 columns 数量
        :type n: int
        :param k: A tensor 的 columns 数量
        :type k: int
        :param l: C tensor 的 columns 数量
        :type l: int
        :param ab_dtype: A 和 B operands 的 data type
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: output tensor 的 data type
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: A tensor 的 major axis
        :type a_major: str
        :param b_major: B tensor 的 major axis
        :type b_major: str
        :param c_major: C tensor 的 major axis
        :type c_major: str

        :return: problem shape 有效时返回 True, 否则返回 False
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

    def is_valid_epilog_store_option(
        self,
        m: int,
        n: int,
    ) -> bool:
        """
        检查 epilogue store option 是否有效.

        :param m: A tensor 的 rows 数量
        :type m: int
        :param n: B tensor 的 columns 数量
        :type n: int

        :return: epilogue store option 有效时返回 True, 否则返回 False
        :rtype: bool
        """

        is_valid = True
        # 不使用 TMA store 的版本没有 predication, 因此不支持 OOB tiles
        cta_tile_shape_mn = (
            self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1),
            self.mma_tiler_mn[1],
        )
        if not self.use_tma_store:
            if not (m % cta_tile_shape_mn[0] == 0 and n % cta_tile_shape_mn[1] == 0):
                is_valid = False
        return is_valid

    def can_implement(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor) -> bool:
        """Check if the given tensors can be implemented by this kernel.

        :param a: input tensor A
        :type a: cute.Tensor
        :param b: input tensor B
        :type b: cute.Tensor
        :param c: output tensor C
        :type c: cute.Tensor

        :return: GEMM 支持给定配置时返回 True, 否则返回 False
        :rtype: bool
        """
        m, n, k, l = a.shape[0], b.shape[0], a.shape[1], a.shape[2]

        # 推导 a_major、b_major 和 c_major
        is_m_major_a = utils.LayoutEnum.from_tensor(a).is_m_major_a()
        is_n_major_b = utils.LayoutEnum.from_tensor(b).is_n_major_b()
        is_m_major_c = utils.LayoutEnum.from_tensor(c).is_m_major_c()
        a_major = "m" if is_m_major_a else "k"
        b_major = "n" if is_n_major_b else "k"
        c_major = "m" if is_m_major_c else "n"

        can_implement = True
        # 排除不支持的 types
        if not self.is_valid_dtypes(a.element_type, c.element_type):
            can_implement = False
        # 排除无效的 MMA tile shape 和 cluster shape
        if not self.is_valid_mma_tiler_and_cluster_shape():
            can_implement = False
        # 排除不满足 load/store alignment 的 problem shape
        if not self.is_valid_tensor_alignment(
            m, n, k, l, a.element_type, c.element_type, a_major, b_major, c_major
        ):
            can_implement = False
        # 排除无效的 epilogue store option
        if not self.is_valid_epilog_store_option(m, n):
            can_implement = False
        if dist.get_world_size() not in [2, 4, 8]:
            can_implement = False
        if m % 128 != 0:
            # m 不能被 128 整除时不支持 OOB tiles
            can_implement = False

        return can_implement


def create_mc_tensor(torch_tensor_cpu, dtype, leading_dim, is_dynamic_layout=True):
    torch_tensor_gpu_local = nvshmem.core.tensor(torch_tensor_cpu.shape, dtype=torch_tensor_cpu.dtype)
    torch_tensor_gpu_local.copy_(torch_tensor_cpu)
    torch_tensor_gpu_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, torch_tensor_gpu_local)
    cute_tensor_mc = from_dlpack(
        torch_tensor_gpu_mc,
        assumed_align=16,
    )

    cute_tensor_c_peer_torch_tensors = [nvshmem.core.get_peer_tensor(torch_tensor_gpu_local, rank) for rank in range(dist.get_world_size())]
    cute_tensor_c_peer_tensors = [from_dlpack(t) for t in cute_tensor_c_peer_torch_tensors]

    if is_dynamic_layout:
        cute_tensor_mc = cute_tensor_mc.mark_layout_dynamic(leading_dim=leading_dim)

    cute_tensor = from_dlpack(torch_tensor_gpu_local, assumed_align=16)
    cute_tensor.element_type = dtype
    if is_dynamic_layout:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    cute_tensor = cutlass_torch.convert_cute_tensor(
        torch_tensor_gpu_local,
        cute_tensor,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
    )
    return cute_tensor, cute_tensor_mc, torch_tensor_gpu_local, torch_tensor_gpu_mc, cute_tensor_c_peer_torch_tensors, cute_tensor_c_peer_tensors


def create_tensors(
    l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype
):
    torch.manual_seed(1111)

    a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major == "m", ab_dtype).to(torch.float32).normal_().round_().to(dtype=cutlass_torch.dtype(ab_dtype))
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", ab_dtype).to(torch.float32).normal_().round_().to(dtype=cutlass_torch.dtype(ab_dtype))
    c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major == "m", c_dtype)

    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    c_tensor, c_tensor_mc, c_torch_gpu, c_torch_gpu_mc, c_peer_torch_tensors, c_peer_tensors = create_mc_tensor(
        c_torch_cpu, c_dtype, (1 if c_major == "n" else 0), is_dynamic_layout=True
    )

    return (
        a_tensor,
        b_tensor,
        c_tensor,
        c_tensor_mc,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        c_torch_gpu,
        c_torch_gpu_mc,
        c_peer_torch_tensors,
        c_peer_tensors,
    )


def compare(
    a_torch_cpu, b_torch_cpu, c_torch_gpu, c_dtype, tolerance
):
    # 取回 GPU 计算结果
    kernel_result = c_torch_gpu.cpu()

    # 计算 reference result
    ref = torch.einsum(
        "mkl,nkl->mnl",
        a_torch_cpu.to(dtype=torch.float32),
        b_torch_cpu.to(dtype=torch.float32),
    )

    acc_type = cutlass.Float32
    if c_dtype.width == 16:
        acc_dtype = cutlass.Float32
    elif c_dtype.width == 8:
        acc_dtype = cutlass.Float16
    
    torch_acc_dtype = cutlass_torch.dtype(acc_dtype)
    torch_c_dtype = cutlass_torch.dtype(c_dtype)

    # 本示例中, f8 reduce sum 使用 f16 accumulation precision, f16 则使用 f32
    ref = ref.to(torch_c_dtype).to(torch_acc_dtype).cuda()
    torch.distributed.all_reduce(ref, op=torch.distributed.ReduceOp.SUM)
    ref = ref.to(torch_c_dtype)

    # 转换为更高精度, 以满足 assert_close function 的要求
    ref_result = ref.cpu().to(torch_acc_dtype)
    kernel_result = c_torch_gpu.view(torch_c_dtype).cpu().to(torch_acc_dtype)

    max_val = torch.finfo(kernel_result.dtype).max
    min_val = torch.finfo(kernel_result.dtype).min
    kernel_result = torch.nan_to_num(kernel_result, nan=max_val, posinf=max_val, neginf=min_val)
    ref_result = torch.nan_to_num(ref_result, nan=max_val, posinf=max_val, neginf=min_val)

    rank_id = dist.get_rank()
    chunk_per_rank = kernel_result.shape[0] // dist.get_world_size()
    start_idx = rank_id * chunk_per_rank
    end_idx = start_idx + chunk_per_rank

    # 检查结果是否足够接近
    torch.testing.assert_close(kernel_result[start_idx:end_idx,:], ref_result[start_idx:end_idx,:], atol=tolerance, rtol=1e-05)


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    use_2cta_instrs: bool = True,
    use_tma_store: bool = True,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    reduce_scatter: str = "two_shot",
    **kwargs,
):
    """Execute a persistent batched dense GEMM operation on Blackwell architecture with performance benchmarking.

    This function prepares input tensors, configures and launches the persistent GEMM kernel,
    optionally performs reference validation, and benchmarks the execution performance.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: input tensors A 和 B 的 data type
    :type ab_dtype: Type[cutlass.Numeric]
    :param c_dtype: output tensor C 的 data type
    :type c_dtype: Type[cutlass.Numeric]
    :param acc_dtype: matrix multiplication 期间用于 accumulation 的 data type
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/c_major: tensors A/B/C 的 memory layout
    :type a_major/b_major/c_major: str
    :param mma_tiler_mn: MMA tiling size. If not specified in the decorator parameters, the autotuner will use the
        default value of (256, 256). Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type mma_tiler_mn: Tuple[int, int], optional
    :param cluster_shape_mn: Cluster shape. If not specified in the decorator parameters, the autotuner will use the
        default value of (2, 1). Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type cluster_shape_mn: Tuple[int, int], optional
    :param use_2cta_instrs: Whether to use 2CTA instructions. If not specified in the decorator parameters, the autotuner
        will use the default value of True. Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type use_2cta_instrs: bool, optional
    :param use_tma_store: Whether to use TMA store. If not specified in the decorator parameters, the autotuner will use
        the default value of True. Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type use_tma_store: bool, optional
    :param tolerance: Tolerance value for reference validation comparison, defaults to 1e-01
    :type tolerance: float, optional
    :param warmup_iterations: benchmark 前的 warmup iteration 数量, 默认为 0
    :type warmup_iterations: int, optional
    :param iterations: benchmark iteration 数量, 默认为 1
    :type iterations: int, optional
    :param skip_ref_check: 是否跳过 reference result 校验, 默认为 False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: 是否使用 circular buffer 保证 cold L2 cache, 默认为 False
    :type use_cold_l2: bool, optional
    :param reduce_scatter: reduce scatter mode, can be "two_shot"
    :type reduce_scatter: str, optional
    :raises RuntimeError: CUDA GPU 不可用时抛出
    :raises ValueError: 配置无效或 kernel 不支持时抛出
    :return: GEMM kernel 的执行时间
    :rtype: float
    """
    print(f"Running Blackwell Persistent Dense GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"2CTA MMA instructions: {'True' if use_2cta_instrs else 'False'}")
    print(f"Use TMA Store: {'True' if use_tma_store else 'False'}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")
    print(f"Fused Reduce Scatter Op: {reduce_scatter}")

    # 解包参数
    m, n, k, l = mnkl

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # 从 PyTorch 获取当前 CUDA stream
    torch_stream = torch.cuda.current_stream()
    # 获取 CUstream 类型的原始 stream pointer
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    (
        a_tensor,
        b_tensor,
        c_tensor,
        c_tensor_mc,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        c_torch_gpu,
        c_torch_gpu_mc,
        c_peer_torch_tensors,
        c_peer_tensors,
    ) = create_tensors(
        l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype
    )

    # 创建 GEMM object
    gemm = PersistentDenseGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        reduce_scatter=reduce_scatter,
    )

    # 检查当前配置能否实现
    can_implement = gemm.can_implement(a_tensor, b_tensor, c_tensor)
    if not can_implement:
        raise ValueError(
            f"The current config which is invalid/unsupported: use_2cta_instrs = {use_2cta_instrs}, "
            f"mma_tiler_mn = {mma_tiler_mn}, cluster_shape_mn = {cluster_shape_mn}, "
            f"use_tma_store = {use_tma_store}"
        )
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    def create_barrier_flags():
        cta_tile_shape_mn = (
            mma_tiler_mn[0] // (2 if use_2cta_instrs else 1),
            mma_tiler_mn[1],
        )
        problem_shape_ntile_mn = (m // cta_tile_shape_mn[0], n // cta_tile_shape_mn[1])
        num_tiles = problem_shape_ntile_mn[0] * problem_shape_ntile_mn[1]
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

        barrier_flag_torch = nvshmem.core.tensor(
            (num_tiles + num_sms,), dtype=torch.int32
        )
        barrier_flag_torch.fill_(0)
        barrier_flag_torch_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, barrier_flag_torch)

        barrier_flag = from_dlpack(barrier_flag_torch)
        barrier_flag = barrier_flag.mark_layout_dynamic()
        barrier_flag_mc = from_dlpack(barrier_flag_torch_mc)
        barrier_flag_mc = barrier_flag_mc.mark_layout_dynamic()

        return barrier_flag_torch, barrier_flag_torch_mc, barrier_flag, barrier_flag_mc

    barrier_flag_torch, barrier_flag_torch_mc, barrier_flag, barrier_flag_mc = create_barrier_flags()
    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        max_active_clusters,
        current_stream,
        c_mc=c_tensor_mc,
        c_peer_tensors=c_peer_tensors,
        barrier_flag=barrier_flag,
        barrier_flag_mc=barrier_flag_mc,
    )

    if not skip_ref_check:
        compiled_gemm(
            a_tensor,
            b_tensor,
            c_tensor,
            current_stream,
            c_mc=c_tensor_mc,
            c_peer_tensors=c_peer_tensors,
            barrier_flag=barrier_flag,
            barrier_flag_mc=barrier_flag_mc,
        )
        compare(
            a_torch_cpu,
            b_torch_cpu,
            c_torch_gpu,
            c_dtype,
            tolerance,
        )
    
    nvshmem.core.free_tensor(c_torch_gpu_mc)
    nvshmem.core.free_tensor(c_torch_gpu)
    for i in range(len(c_peer_tensors)):
        if i != dist.get_rank():
            nvshmem.core.free_tensor(c_peer_torch_tensors[i])
    nvshmem.core.free_tensor(barrier_flag_torch_mc)
    nvshmem.core.free_tensor(barrier_flag_torch)

    free_func_and_tensor_pairs = []
    def add_free_func_and_tensor(free_func, tensor):
        free_func_and_tensor_pairs.append((free_func, tensor))

    def generate_tensors():
        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, _ = cutlass_torch.cute_tensor_like(
            c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, c_tensor_mc, c_torch_gpu, c_torch_gpu_mc, c_peer_torch_tensors, c_peer_tensors = create_mc_tensor(
            c_torch_cpu,
            c_dtype,
            (1 if c_major == "n" else 0),
            is_dynamic_layout=True,
        )
        barrier_flag_torch, barrier_flag_torch_mc, barrier_flag, barrier_flag_mc = create_barrier_flags()
        add_free_func_and_tensor(nvshmem.core.free_tensor, c_torch_gpu_mc)
        add_free_func_and_tensor(nvshmem.core.free_tensor, c_torch_gpu)
        for i in range(len(c_peer_torch_tensors)):
            if i != dist.get_rank():
                add_free_func_and_tensor(nvshmem.core.free_tensor, c_peer_torch_tensors[i])
        add_free_func_and_tensor(nvshmem.core.free_tensor, barrier_flag_torch_mc)
        add_free_func_and_tensor(nvshmem.core.free_tensor, barrier_flag_torch)
        return testing.JitArguments(
            a_tensor,
            b_tensor,
            c_tensor,
            current_stream,
            c_mc=c_tensor_mc,
            c_peer_tensors=c_peer_tensors,
            barrier_flag=barrier_flag,
            barrier_flag_mc=barrier_flag_mc,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch_cpu.numel() * a_torch_cpu.element_size()
            + b_torch_cpu.numel() * b_torch_cpu.element_size()
            + c_torch_cpu.numel() * c_torch_cpu.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    for free_func, tensor in free_func_and_tensor_pairs:
        free_func(tensor)
    print(f"exec_time: {exec_time}\n")

    return exec_time  # Return execution time in microseconds


def torchrun_uid_init_bcast():
    """
    使用 UniqueID 初始化 NVSHMEM, 并以 `torchrun` 作为 launcher.

    这里通过 torch.distributed.broadcast 广播 NumPy array.
    """
    # 设置 Torch device
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # nvshmem4py 初始化时需要 cuda.core Device
    dev = Device(local_rank)
    dev.set_current()
    global stream
    stream = dev.create_stream()

    # 初始化 torch.distributed process group
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
    )

    # 从 process group 取 rank 和 nranks
    num_ranks = dist.get_world_size()

    # 为所有 ranks 创建空 uniqueid
    uid = nvshmem.core.get_unique_id(empty=(local_rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)

    nvshmem.core.init(device=dev, uid=uid, rank=local_rank, nranks=num_ranks, initializer_method="uid")


def torchrun_finalize():
    nvshmem.core.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Example of Dense Persistent GEMM on Blackwell."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(256, 256, 512, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.TFloat32)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions feature",
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
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
    parser.add_argument(
        "--reduce_scatter",
        choices=["two_shot"],
        type=str,
        default="two_shot",
        help="Reduce Scatter algorithm to fuse with gemm",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")
    
    if args.c_dtype.width == 8:
        import warnings
        warnings.warn("f8 output is easy to overflow and can be nan here")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torchrun_uid_init_bcast()

    run(
        args.mnkl,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_2cta_instrs,
        args.use_tma_store,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        reduce_scatter=args.reduce_scatter,
    )

    torchrun_finalize()

    
