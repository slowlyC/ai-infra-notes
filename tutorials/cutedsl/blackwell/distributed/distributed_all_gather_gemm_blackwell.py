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
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed as dist
import cuda.bindings.driver as cuda
from cuda.bindings import driver
try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device
from cuda.pathfinder import load_nvidia_dynamic_lib

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.distributed as dist_helpers
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

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
使用 CUTE DSL、面向 NVIDIA Blackwell SM100 架构的高性能 All-Gather + dense GEMM 示例.
本示例假设运行在多 GPU 上, MNKL 表示每个 GPU 的 problem size.
- Matrix A 为 MxKxL, L 是 batch dimension, A 可以是 row-major ("K") 或 column-major ("M")
- Matrix B 为 NxKxL, L 是 batch dimension, B 可以是 row-major ("N") 或 column-major ("K")
- Matrix C 为 MxNxL, L 是 batch dimension, C 可以是 row-major ("N") 或 column-major ("M")

这个 GEMM kernel 支持以下功能:
    - 使用 Tensor Memory Access (TMA) 高效执行 memory operations
    - 使用 Blackwell tcgen05.mma 执行 matrix multiply-accumulate (MMA), 包括 2CTA MMA instructions
    - 结合 cluster 实现 TMA multicast, 减少 L2 memory traffic
    - 使用 persistent tile scheduling, 更好地重叠不同 tiles 间的 memory load/store 和 MMA
    - 使用 warp specialization, 避免在 mainloop load 和 MMA 之间显式构建 pipeline

GEMM 的执行过程如下:
1. DMA warp: 使用 TMA 将 matrices A 和 B 从 global memory (GMEM) 加载到 shared memory (SMEM).
2. MMA warp: 使用 tcgen05.mma instruction 执行 matrix multiply-accumulate (MMA).
3. EPILOGUE warp:
    - 使用 tcgen05.ld 将完成的 accumulator 从 tensor memory (TMEM) 加载到 registers (RMEM).
    - 将 matrix C 转换为 output type.
    - 可以通过 TMA 将 matrix C 从 RMEM 写入 SMEM, 再写入 GMEM;
      也可以不使用 TMA, 直接从 RMEM 写入 GMEM.
    - 可以传入 elementwise lambda function `epilogue_op` 作用于 output tensor.
      例如 relu 可设为 `lambda x: cute.where(x > 0, x, cute.full_like(x, 0))`.

SM100 tcgen05.mma instructions 的操作如下:
- 从 SMEM 读取 matrix A
- 从 SMEM 读取 matrix B
- 将 accumulator 写入 TMEM
随后必须把 TMEM 中的 accumulator 加载到 registers, 再写回 GMEM.

本示例的 input arguments 与 dense_gemm_persistent.py 相同.

.. code-block:: bash

    torchrun --nproc_per_node=8 examples/distributed/distributed_all_gather_gemm_blackwell.py                          \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
      --mnkl 8192,8192,8192,1                                                   \
      --use_tma_store --use_2cta_instrs

使用 NCU profiler 收集性能数据:

.. code-block:: bash

    ncu torchrun --nproc_per_node=8 examples/distributed/distributed_all_gather_gemm_blackwell.py                     \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                 \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,8192,1                                                  \
      --use_tma_store --use_2cta_instrs                                        \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


约束与 dense_gemm_persistent.py 相同:
* A 和 C 必须是 row-major
* 支持的 input data types: fp16、bf16、tf32、int8、uint8、fp8 (e4m3fn、e5m2),
  有效 dtype 组合详见下方 PersistentDenseGemmKernel class 文档
* A/B tensors 必须使用相同 data type
* MMA tiler M 必须为 64/128 (use_2cta_instrs=False) 或 128/256 (use_2cta_instrs=True)
* MMA tiler N 必须在 32-256 范围内, 步长为 32
* Cluster shape M/N 必须是正的 2 的幂, cluster size 总数不超过 16
* use_2cta_instrs=True 时, Cluster shape M 必须是 2 的倍数
* A/B/C tensors 的 contiguous dimension 必须至少按 16 bytes 对齐.
  对 TFloat32、Float16/BFloat16、Int8/Uint8/Float8, elements 数量必须分别是 4、8、16 的倍数
* 禁用 TMA store 时不允许 OOB tiles
"""


class SyncNvlDevices:
    """A class for synchronizing multiple NVIDIA devices using NVL (NVLink) barriers.

    This class implements a barrier synchronization mechanism for distributed computing
    across multiple GPUs, ensuring all participating devices reach a synchronization point
    before proceeding to the next phase of computation.

    :param num_of_parallelism: 参与同步的 devices 总数 (NP)
    :type num_of_parallelism: int
    """

    def __init__(self, num_of_parallelism: int):
        """Initialize the synchronization class.

        :param num_of_parallelism: 参与同步的 devices 总数 (NP)
        :type num_of_parallelism: int
        """
        self.num_of_parallelism = num_of_parallelism

    @cute.kernel
    def kernel(
        self,
        device_idx: cutlass.Int32,
        device_arrival_counters: cute.Tensor,  # Shape: (num_of_parallelism,), dtype: int32
        iteration_flags: cute.Tensor,  # Shape: (num_of_parallelism, 1), dtype: int32 (e.g., input_ready_flags)
    ):
        """
        同步参与计算的 devices, 并重置 iteration flags.

        Args:
            device_idx: The logical ID of the current device.
            device_arrival_counters: Shared counters for barrier synchronization.
            iteration_flags: Flags used by the subsequent kernel (will be reset to 0).
        """
        tidx, _, _ = cute.arch.thread_idx()

        # Barrier 逻辑使用的 constants
        val_one = cutlass.Int32(1)
        val_zero = cutlass.Int32(0)
        max_arrivals = cutlass.Int32(self.num_of_parallelism - 1)

        # --- 阶段 1: 重置 iteration flags, 仅由 thread 0 执行 ---
        # 假设 iteration_flags 的 shape 为 (num_of_parallelism, 1) 或类似形式.
        # 如果 iteration_flags 表示多组 flags, 例如 C++ 中的 Iterations, 则需要调整此循环.
        for i in range(self.num_of_parallelism):
            iteration_flags[i] = val_zero  # 为下一个 kernel 重置 flags

        # --- 阶段 2: 向 peers 通知到达, 仅由 thread 0 执行 ---
        for peer_rank in range(self.num_of_parallelism):
            # 向所有其它 devices 通知到达
            if peer_rank != device_idx:
                counter_ptr = cute.make_ptr(
                    cutlass.Int32,
                    device_arrival_counters[peer_rank],
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                _ = utils.distributed.atomicAdd(counter_ptr, val_one)

        # --- 阶段 3: 等待其它 devices 到达, 仅由 thread 0 执行 ---
        # 轮询 local arrival counter
        local_counter_ptr = cute.make_ptr(
            cutlass.Int32,
            device_arrival_counters[device_idx],
            cute.AddressSpace.gmem,
            assumed_align=4,
        )
        local_counter_tensor = cute.make_tensor(
            local_counter_ptr, cute.make_layout(shape=(1,), stride=(1,))
        )
        current_arrivals = utils.distributed.ld_bypass(local_counter_tensor)[0]
        # 等待 NP-1 个 peers 发出通知
        # 如果环境支持且有必要, 可以加入短暂 sleep/yield, 避免轮询负载过高
        while current_arrivals < max_arrivals:
            # 可以考虑在这里加入短暂延迟, 等价于 __nanosleep
            # 当前仅执行轮询:
            current_arrivals = utils.distributed.ld_bypass(local_counter_tensor)[0]

        # --- 阶段 4: 重置 local counter, 仅由 thread 0 执行 ---
        # 从 local counter 原子减去 NP-1
        reset_val = cutlass.Int32(-(self.num_of_parallelism - 1))
        _ = utils.distributed.atomicAdd(local_counter_ptr, reset_val)

    @cute.jit
    def __call__(
        self,
        device_idx: cutlass.Int32,
        device_arrival_counters: cute.Tensor,
        iteration_flags: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Execute the synchronization kernel.

        :param device_idx: 当前 device 的 logical ID
        :type device_idx: cutlass.Int32
        :param device_arrival_counters: 用于 barrier synchronization 的 shared counters
        :type device_arrival_counters: cute.Tensor
        :param iteration_flags: 后续 kernel 使用的 flags, 随后会重置为 0
        :type iteration_flags: cute.Tensor
        :param stream: 用于执行 kernel 的 CUDA stream
        :type stream: cuda.CUstream
        """
        grid = [1, 1, 1]
        self.kernel(device_idx, device_arrival_counters, iteration_flags).launch(
            grid=grid, block=[1, 1, 1], cluster=(1, 1, 1), smem=0, stream=stream
        )

def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
    use_tma_store: bool,
    c_smem_layout: Union[cute.Layout, None],
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
    :param c_dtype: operand C (output) 的 data type.
    :type c_dtype: type[cutlass.Numeric]
    :param smem_capacity: 可用 shared memory 总容量, 单位为 bytes.
    :type smem_capacity: int
    :param occupancy: 每个 SM 的目标 CTA 数量 (occupancy).
    :type occupancy: int
    :param use_tma_store: 是否启用 TMA store.
    :type use_tma_store: bool
    :param c_smem_layout: C operand 在 SMEM 中的 layout; 不使用 TMA store 时为 None.
    :type c_smem_layout: Union[cute.Layout, None]

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
        tiled_mma, mma_tiler_mnk, a_dtype, 1
    )
    b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b_dtype, 1
    )

    ab_bytes_per_stage = cute.size_in_bytes(
        a_dtype, a_smem_layout_stage_one
    ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
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
        gated_a_load: bool,
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

        4. Gated A load:
            - gated_a_load: Boolean indicating whether to gating A load in the kernel.

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

        self.occupancy = 1
        # 设置 specialized warp ids
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # 设置用于 CTA sync、epilogue sync 和 TMEM pointer sync 的 barrier id
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.gated_a_load = gated_a_load

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

        c_smem_layout = None
        if cutlass.const_expr(self.use_tma_store):
            c_smem_layout = sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, 1
            )

        # 设置 A/B/C 在 SMEM 中的 stage 数量, 以及 ACC 在 TMEM 中的 stage 数量
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
            c_smem_layout,
        )

        # 计算 A/B/C 的 SMEM layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )

        self.c_smem_layout_staged = None
        if self.use_tma_store:
            self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
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
        flag_offset: int,
        gate_a_flags: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
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

        # ring-based all-gather 的特定要求
        a_layout_enum = utils.LayoutEnum.from_tensor(a)
        c_layout_enum = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(a_layout_enum != utils.LayoutEnum.ROW_MAJOR):
            raise TypeError(f"A must be row-major: {a_layout_enum}")
        if cutlass.const_expr(c_layout_enum != utils.LayoutEnum.ROW_MAJOR):
            raise TypeError(f"C must be row-major: {c_layout_enum}")
        self.a_major_mode = a_layout_enum.mma_major_mode()
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
            epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
            )

        # 计算 grid size
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        # 同步启动 kernel
        gate_a_flag_tensor = cute.make_tensor(
            gate_a_flags.iterator + flag_offset,
            cute.make_layout((1,), stride=(1,)),
        )
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
            gate_a_flag_tensor,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
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
        gate_a_flag: cute.Tensor,
        epilogue_op: cutlass.Constexpr,
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
        lane_idx = cute.arch.lane_idx()

        #
        # 分配并初始化: A+B full/empty、accumulator full/empty 和 TMEM dealloc barrier
        #
        # 定义 kernel 的 shared storage
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # 初始化 mainloop ab_pipeline (barrier) 和 states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

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
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilog_warp_id),
            )
        # 初始化 TMEM dealloc barrier
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # barrier 初始化后执行 cluster arrive
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        #
        # 设置 SMEM tensors A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
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
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

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

            #########################################################
            if self.gated_a_load:
                # 等待 flag
                if lane_idx == 0:
                    # 等待 input_ready_flags[iteration_i] 被置位
                    ready_flag = utils.distributed.ld_bypass(gate_a_flag)[0]
                    # 需要使用 volatile load, 防止编译器消除 polling loop
                    while ready_flag == 0:
                        # 使用 volatile load 持续轮询直到数据就绪
                        ready_flag = utils.distributed.ld_bypass(gate_a_flag)[0]
                cute.arch.sync_warp()
            #########################################################
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
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()
                #
                # TMA load 循环
                #

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # 按条件等待 AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # 执行 A/B 的 TMA load
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # 通过 try_wait 检查下一个预取 k_tile 的 AB buffer empty
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                #
                # 推进到下一个 tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # 等待 A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # 获取 TMEM pointer 并创建 accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling 循环
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

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
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

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
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)


                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblk_crd],
                                tCrB[kblk_crd],
                                tCtAcc,
                            )
                            # 从第一个 kblock 之后启用 tCtAcc accumulate
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # 异步通知 AB buffer empty
                        handle.release()

                        # 通过 try_wait 检查下一个 k_tile 的 AB buffer full
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

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
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # 分配 TMEM buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)


            #
            # 获取 TMEM pointer 并创建 accumulator tensor
            #

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Epilogue 的 persistent tile scheduling 循环
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            if cutlass.const_expr(self.use_tma_store):

                assert tma_atom_c is not None and sC is not None
                self.epilogue_tma_store(
                    tidx,
                    warp_idx,
                    acc_pipeline,
                    tiled_mma,
                    tma_atom_c,
                    tCtAcc_base,
                    sC,
                    tCgC,
                    epi_tile,
                    tile_sched,
                    epilogue_op,
                )
            else:
                self.epilogue(
                    tidx,
                    acc_pipeline,
                    tiled_mma,
                    tCtAcc_base,
                    tCgC,
                    epi_tile,
                    tile_sched,
                    epilogue_op,
                    tmem_dealloc_barrier,
                )

            #
            # 释放 TMEM buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def epilogue_tma_store(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        acc_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tma_atom_c: cute.CopyAtom,
        # Epilogue 的 input
        tCtAcc_base: cute.Tensor,
        # Epilogue 的 staging buffer
        sC: cute.Tensor,
        # Epilogue 的 output
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        tile_sched: utils.StaticPersistentTileScheduler,
        epilogue_op: cutlass.Constexpr,
    ) -> None:
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
            epi_tidx, tCtAcc_base, tCgC, epi_tile, self.use_2cta_instrs
        )

        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
            tiled_copy_t2r, tTR_rC, epi_tidx, sC
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )

        # 参与 TMA store pipeline 的 threads/warps
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            32 * len(self.epilog_warp_id),
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.num_c_stage, producer_group=c_producer_group
        )

        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilog_warp_id),
        )

        work_tile = tile_sched.initial_work_tile_info()
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
            # ((ATOM_V, REST_V), EPI_M, EPI_N)
            bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
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
            bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

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
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                # 使用 fence 和 barrier, 确保 SMEM store 对 TMA store 可见
                cute.arch.fence_proxy("async.shared", space="cta")
                epilog_sync_barrier.arrive_and_wait()

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
                epilog_sync_barrier.arrive_and_wait()

            epilog_sync_barrier.arrive_and_wait()

            #
            # 异步通知 accumulator buffer empty
            #
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

            #
            # 推进到下一个 tile
            #
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # 等待 C store 完成
        c_pipeline.producer_tail()

    @cute.jit
    def epilogue(
        self,
        epi_tidx: cutlass.Int32,
        acc_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tCtAcc_base: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        tile_sched: utils.StaticPersistentTileScheduler,
        epilogue_op: cutlass.Constexpr,
        tmem_dealloc_barrier: pipeline.NamedBarrier,
    ) -> None:
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
            epi_tidx, tCtAcc_base, tCgC, epi_tile, self.use_2cta_instrs
        )

        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_gC_partitioned = thr_copy_t2r.partition_D(gC_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].shape, self.c_dtype
        )
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )

        work_tile = tile_sched.initial_work_tile_info()
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
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_gC = tTR_gC_partitioned[
                (None, None, None, None, None, *mma_tile_coord_mnl)
            ]

            # 设置当前 tile 的 TMEM buffer
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_tAcc = tTR_tAcc_base[
                (None, None, None, None, None, acc_consumer_state.index)
            ]

            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

            #
            # 等待 accumulator buffer full
            #
            acc_pipeline.consumer_wait(acc_consumer_state)

            #
            # 按 subtiles 将 accumulator 写入 global memory
            #
            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            for subtile_idx in cutlass.range(subtile_cnt):
                #
                # 将 accumulator 从 TMEM buffer 加载到 registers
                #
                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                #
                # 转换为 C type
                #
                acc_vec = tTR_rAcc.load()
                acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                tTR_rC.store(acc_vec)

                #
                # 将 C 写入 global memory
                #
                cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)])

            #
            # 异步通知 accumulator buffer empty
            #
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

            # 推进到下一个 tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # 释放 TMEM 前同步, 由 caller 执行
        tmem_dealloc_barrier.arrive_and_wait()

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
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
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
        self, ab_dtype: Type[cutlass.Numeric], c_dtype: Type[cutlass.Numeric]
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

        # TODO: 移到 utils
        def check_contiguous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contiguous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contiguous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contiguous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    def is_valid_epilog_store_option(self, m: int, n: int) -> bool:
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

        return can_implement


def create_tensors(
    l,
    m_per_iteration,
    n,
    k,
    a_major,
    b_major,
    c_major,
    ab_dtype,
    c_dtype,
    num_steps,
    local_rank,
):
    torch.manual_seed(1111)
    a_torch_cpu = cutlass_torch.matrix(l, m_per_iteration, k, a_major == "m", ab_dtype)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", ab_dtype)
    c_torch_cpu = cutlass_torch.matrix(l, m_per_iteration, n, c_major == "m", c_dtype)

    # 创建 local buffers
    a_local_list = []
    a_local_torch_list = []
    c_local_list = []
    c_local_torch_list = []
    for m_dim_offset in range(num_steps):
        a_temp = a_torch_cpu.clone()
        a_temp.zero_()
        if m_dim_offset == local_rank:
            a_temp.copy_(a_torch_cpu)
        a_local, a_local_torch = cutlass_torch.cute_tensor_like(
            a_temp, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        a_local_list.append(a_local)
        a_local_torch_list.append(a_local_torch)
        c_temp = c_torch_cpu.clone().zero_()
        c_local, c_local_torch = cutlass_torch.cute_tensor_like(
            c_temp, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_local_list.append(c_local)
        c_local_torch_list.append(c_local_torch)
    a_torch_unique = torch.empty(
        a_torch_cpu.shape, device="cuda", dtype=a_torch_cpu.dtype
    )
    a_torch_unique.copy_(a_torch_cpu)
    a_tensor_symm = nvshmem.core.tensor(a_torch_cpu.shape, dtype=a_torch_cpu.dtype)
    a_tensor_symm.copy_(a_local_torch_list[local_rank])
    a_tensor_peers = [nvshmem.core.get_peer_tensor(a_tensor_symm, rank) for rank in range(world_size)]

    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )

    gate_a_flags = torch.zeros(num_steps, device="cuda", dtype=torch.int32)

    device_arrival_counters = torch.tensor([0], dtype=torch.int32, device="cuda")
    device_arrival_counters_symm = nvshmem.core.tensor(1, dtype=torch.int32)
    device_arrival_counters_symm.copy_(device_arrival_counters)
    device_arrival_counters_peers = [nvshmem.core.get_peer_tensor(device_arrival_counters_symm, rank) for rank in range(world_size)]

    return (
        a_local_list,
        a_local_torch_list,
        a_torch_unique,
        a_tensor_symm,
        a_tensor_peers,
        b_tensor,
        b_torch,
        c_local_list,
        c_local_torch_list,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        gate_a_flags,
        device_arrival_counters_symm,
        device_arrival_counters_peers,
    )


def compare(
    a_torch_unique,
    b_torch_cpu,
    c_torch_local_list,
    c_dtype,
    tolerance,
    rank,
    world_size,
):
    # 取回 GPU 计算结果
    kernel_result = torch.cat(c_torch_local_list, dim=0).cpu()

    all_gather_a = [torch.zeros_like(a_torch_unique) for _ in range(world_size)]
    dist.all_gather(all_gather_a, a_torch_unique)
    a = torch.cat(all_gather_a, dim=0)

    # 在 reference computation 前将 GPU tensors 转到 CPU
    # 这样可确保与 dense_gemm_persistent.py 使用相同的计算路径
    a_cpu = a.cpu()

    # 使用 CPU tensors 计算 reference result, 与 dense_gemm_persistent.py 相同
    num_rows = a_cpu.shape[0]
    ref = torch.einsum(
        "mkl,nkl->mnl",
        a_cpu.to(dtype=torch.float32),
        b_torch_cpu.to(dtype=torch.float32),
    )

    # 将 reference 转换为 c_dtype
    _, ref_torch_gpu = cutlass_torch.cute_tensor_like(
        ref, c_dtype, is_dynamic_layout=True, assumed_align=16
    )
    ref_result = ref_torch_gpu.cpu()
    # 检查结果是否足够接近
    torch.testing.assert_close(kernel_result, ref_result, atol=tolerance, rtol=1e-05)


def run(
    rank,
    world_size,
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
    **kwargs,
):
    """Execute a patthen of All gather + persistent batched dense GEMM operation on Blackwell architecture with performance benchmarking.

    This function prepares input tensors and flags needed for sync between communicaton and gemm, configures and launches the persistent GEMM kernel and Peer mem copys and memset,
    and construt them into a cuda graph.
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
    :raises RuntimeError: CUDA GPU 不可用时抛出
    :raises ValueError: 配置无效或 kernel 不支持时抛出
    :return: GEMM kernel 的执行时间
    :rtype: float
    """
    print("Running Blackwell Persistent Dense GEMM test with:")
    print(f"per GPU mnkl: {mnkl}")
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

    # 解包参数
    m, n, k, l = mnkl
    m_per_step = m // world_size

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # 从 PyTorch 获取当前 CUDA stream
    torch_stream = torch.cuda.current_stream()
    # 获取 CUstream 类型的原始 stream pointer
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    (
        a_local_list,
        a_local_torch_list,
        a_torch_unique,
        a_tensor_symm,
        a_tensor_peers,
        b_tensor,
        b_torch,
        c_local_list,
        c_local_torch_list,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        gate_a_flags,
        device_arrival_counters_symm,
        device_arrival_counters_peers,
    ) = create_tensors(
        l,
        m_per_step,
        n,
        k,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        c_dtype,
        world_size,
        rank,
    )

    # 创建 GEMM object
    gemm = PersistentDenseGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        gated_a_load=False,
    )
    gemm_with_gated_a_load = PersistentDenseGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        gated_a_load=True,
    )

    # 检查当前配置能否实现
    can_implement = gemm.can_implement(a_local_list[0], b_tensor, c_local_list[0])
    if not can_implement:
        raise ValueError(
            f"The current config which is invalid/unsupported: use_2cta_instrs = {use_2cta_instrs}, "
            f"mma_tiler_mn = {mma_tiler_mn}, cluster_shape_mn = {cluster_shape_mn}, "
            f"use_tma_store = {use_tma_store}"
        )
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    compiled_gemm = cute.compile(
        gemm,
        a_local_list[0],
        b_tensor,
        c_local_list[0],
        max_active_clusters,
        current_stream,
        flag_offset=0,
        gate_a_flags=from_dlpack(gate_a_flags),
    )
    compiled_gemm_with_gated_a_load = cute.compile(
        gemm_with_gated_a_load,
        a_local_list[0],
        b_tensor,
        c_local_list[0],
        max_active_clusters,
        current_stream,
        flag_offset=0,
        gate_a_flags=from_dlpack(gate_a_flags),
    )

    device_arrival_counters_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in device_arrival_counters_peers], device="cuda", dtype=torch.int64
    )
    sync_nvl_devices_instance = SyncNvlDevices(world_size)
    compiled_sync_nvl_devices = cute.compile(
        sync_nvl_devices_instance,
        device_idx=rank,
        device_arrival_counters=from_dlpack(device_arrival_counters_ptrs),
        iteration_flags=from_dlpack(gate_a_flags),
        stream=current_stream,
    )

    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream(local_rank)
    copy_stream = torch.cuda.Stream(local_rank)
    gemm_stream = torch.cuda.Stream(local_rank)

    num_steps = world_size
    with torch.cuda.graph(g, stream=capture_stream):
        compiled_sync_nvl_devices(
            rank,
            from_dlpack(device_arrival_counters_ptrs),
            from_dlpack(gate_a_flags),
            cuda.CUstream(capture_stream.cuda_stream),
        )
        gemm_stream.wait_stream(capture_stream)
        for j in range(num_steps):
            m_dim_offset = (local_rank + j) % num_steps
            if j == 0:  # gemm process local data directly
                compiled_gemm(
                    a_local_list[m_dim_offset],
                    b_tensor,
                    c_local_list[m_dim_offset],
                    cuda.CUstream(gemm_stream.cuda_stream),
                    flag_offset=m_dim_offset,
                    gate_a_flags=from_dlpack(gate_a_flags),
                )
            else:  # gemm process remote data with gated a load
                compiled_gemm_with_gated_a_load(
                    a_local_list[m_dim_offset],
                    b_tensor,
                    c_local_list[m_dim_offset],
                    cuda.CUstream(gemm_stream.cuda_stream),
                    flag_offset=m_dim_offset,
                    gate_a_flags=from_dlpack(gate_a_flags),
                )

        copy_stream.wait_stream(capture_stream)
        for j in range(num_steps):
            m_dim_offset = (local_rank + j) % num_steps
            if m_dim_offset != local_rank:
                copy_bytes = (
                    a_local_torch_list[m_dim_offset].element_size()
                    * a_local_torch_list[m_dim_offset].numel()
                )
                dst_address = a_local_torch_list[m_dim_offset].data_ptr()
                src_address = a_tensor_peers[m_dim_offset].data_ptr()
                driver.cuMemcpyDtoDAsync(
                    dst_address,
                    src_address,
                    copy_bytes,
                    copy_stream.cuda_stream,
                )
                memset_address = (
                    gate_a_flags.data_ptr() + m_dim_offset * gate_a_flags.element_size()
                )
                driver.cuMemsetD32Async(
                    memset_address,
                    1,  # value_to_set
                    1,  # size
                    copy_stream.cuda_stream,
                )

        capture_stream.wait_stream(copy_stream)
        capture_stream.wait_stream(gemm_stream)

    replay_stream = torch.cuda.Stream(local_rank)
    # 预热
    with torch.cuda.stream(replay_stream):
        for i in range(warmup_iterations):
            g.replay()
    torch.cuda.synchronize()
    # 性能测试
    with torch.cuda.stream(replay_stream):
        # INSERT_YOUR_CODE
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i in range(iterations):
            g.replay()
        end_event.record()
    torch.cuda.synchronize()
    exec_time = start_event.elapsed_time(end_event)
    # 计算每次 iteration 的平均时间
    avg_time_per_iteration = (exec_time * 1000) / iterations

    # 收集所有 GPUs 的耗时并取最大值
    # 转换为用于 distributed communication 的 tensor
    time_tensor = torch.tensor([avg_time_per_iteration], device="cuda")

    # 将所有耗时收集到 rank 0
    gathered_times = [torch.zeros_like(time_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_times, time_tensor)

    # 获取所有 GPUs 中的最大耗时
    exec_time = max(tensor.cpu().item() for tensor in gathered_times)

    # reference 校验
    if not skip_ref_check:
        # compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
        compare(
            a_torch_unique,
            b_torch_cpu,
            c_local_torch_list,
            c_dtype,
            tolerance,
            rank,
            world_size,
        )

    for i in range(world_size):
        if i != dist.get_rank():
            nvshmem.core.free_tensor(a_tensor_peers[i])
            nvshmem.core.free_tensor(device_arrival_counters_peers[i])
    nvshmem.core.free_tensor(a_tensor_symm)
    nvshmem.core.free_tensor(device_arrival_counters_symm)

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
        description="Example of All Gather + Dense Persistent GEMM on Blackwell."
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

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    torchrun_uid_init_bcast()
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"World size: {world_size}, local rank: {local_rank}")
    torch.cuda.set_device(local_rank)

    run(
        local_rank,
        world_size,
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
    )
    torchrun_finalize()
    print("PASS")
