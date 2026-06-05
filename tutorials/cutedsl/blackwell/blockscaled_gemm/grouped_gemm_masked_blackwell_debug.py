# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
import functools
from cutlass._mlir import ir
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from cutlass.cutlass_dsl import (
    Int32,
    Int64,
    Uint8,
    Uint64,
    T,
    Integer,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass._mlir.dialects import llvm
from flashinfer.utils import get_compute_capability
from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.gemm import grouped_gemm_nt_masked_trace
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    cutlass_to_torch_dtype,
    get_num_sm,
    get_max_active_clusters,
    make_ptr,
)
from typing import Callable, List


sizeof_i32 = 4


# ============================================================================
# 辅助工具函数
# ============================================================================
#
# 这三个函数用于 epilogue 阶段的 dst_signals 机制.
# 在 MoE 场景里, GEMM kernel 按 expert 分批计算.某个 expert 的所有
# output tile 写回完成后, 需要通过原子操作通知下游 combine kernel:
# 这个 expert 的 GEMM 输出已经就绪.
#
# 为了追踪每个 expert 的写回进度, 代码把最多 8 个 expert 的 pending
# 计数打包进一个 Uint64, 每个 expert 占 1 byte.with_byte / read_byte
# 就是在这个 packed u64 上做单字节读写.
@dsl_user_op
def with_byte(obj: Uint64, index: Int32, value: Uint8, *, loc=None, ip=None) -> Uint64:
    # 清零目标 byte, 再写入新的 pending 计数.
    obj &= ~(0xFF << (index * 8))
    obj |= value << (index * 8)
    assert isinstance(obj, Uint64), f"{obj=}"
    return obj


@dsl_user_op
def read_byte(obj: Uint64, index: Int32, *, loc=None, ip=None) -> Uint8:
    # 读取 packed u64 中某个 expert 对应的 pending 计数.
    return ((obj >> (index * 8)) & 0xFF).to(Uint8)


@dsl_user_op
def atomic_add_release_global(addr: Int64, value: Int32, *, loc=None, ip=None) -> Int32:
    # release 语义保证: 此原子加之前的 GMEM 写入对其他 SM 可见.
    # 这里用于 epilogue 写完某个 expert 后递增 dst_signals[expert_idx].
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                addr.ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.add.release.gpu.global.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# ============================================================================
# Masked Tile Scheduler
# ============================================================================
#
# 与普通 dense GEMM 的固定网格不同, 这里的 MoE GEMM 有 L 个 expert,
# 每个 expert 的有效 token 数 masked_m[l] 不同.调度器需要跳过 padding
# 行, 避免为空 token 做无效 GEMM.
#
# 调度的 linear_idx 按 expert 展开后的 (M_tiles, N_tiles) 行优先顺序编号:
#
#   expert 0: [0, M0 * N)
#   expert 1: [M0 * N, (M0 + M1) * N)
#   expert 2: [(M0 + M1) * N, (M0 + M1 + M2) * N)
#
# 其中 Mi = ceil_div(masked_m[i], tile_M).
# persistent cluster 从自己的初始 linear_idx 开始, 每次步进
# num_persistent_clusters, 直到覆盖所有有效 tile.
class MaskedSchedulerParams:
    def __init__(
        self,
        masked_m: cute.Tensor,
        dst_signals: Optional[cute.Pointer],
        c: cute.Tensor,
        c_tiler: Tuple[int, int],
        cluster_shape_mnk: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        if cluster_shape_mnk[2] != 1:
            raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")

        # 将 C 按 c_tiler 分块, 得到完整 tile 网格形状:
        #   (ntile_M, ntile_N, L)
        # 后续 MaskedScheduler 会根据 masked_m 剔除每个 expert 的 padding M tile.
        gc = cute.zipped_divide(c, tiler=c_tiler)
        problem_shape_ntile_mnl = gc[(0, (None, None, None))].shape
        self.masked_m = masked_m
        self.dst_signals = dst_signals
        self.c = c
        self.c_tiler = c_tiler
        self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
        # cluster_shape_mnk 需要保留, 用于 MLIR value reconstruction.
        self._cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]
        self._loc = loc

        # cluster 级别的网格形状: 将 tile 网格再按 cluster_shape_mn 分块.
        self.problem_layout_ncluster_mnl = cute.make_layout(
            cute.ceil_div(
                self.problem_shape_ntile_mnl, cluster_shape_mnk[:2], loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.masked_m,
            self.dst_signals,
            self.c,
            self.c_tiler,
            self._cluster_shape_mnk,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.masked_m,
                self.dst_signals,
                self.c,
                self.c_tiler,
                self._cluster_shape_mnk,
            ],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return MaskedSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    @dsl_user_op
    def get_grid_shape(
        self, max_active_clusters: Int32, *, loc=None, ip=None
    ) -> Tuple[Integer, Integer, Integer]:
        # persistent scheduling 的 grid 不是每个 output tile 一个 CTA.
        # 这里只启动固定数量的 cluster:
        #   grid = (ClusterM, ClusterN, num_persistent_clusters)
        # 每个 cluster 在 while loop 中动态领取后续 work tile.
        num_persistent_clusters = max_active_clusters

        return (*self.cluster_shape_mn, num_persistent_clusters)


class MaskedScheduler:
    # GPU 侧 tile 调度器, 每个专用 warp group 各自持有一份:
    #   TMA warp      负责加载同一个 work_tile 的 A/B/SFA/SFB
    #   MMA warp      负责计算同一个 work_tile
    #   Epilogue warp 负责写回同一个 work_tile
    # 三者从相同 block_idx 初始化并按相同步长 advance, 因此能保持步调一致.
    def __init__(
        self,
        params: MaskedSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        current_batch_idx: Int32,
        accum_tile_m: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
    ):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self._current_batch_idx = current_batch_idx
        self._accum_tile_m = accum_tile_m
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self._current_batch_idx))
        values.extend(extract_mlir_values(self._accum_tile_m))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "MaskedScheduler":
        assert len(values) == 8
        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[0]]
        )
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[1]]
        )
        new_current_batch_idx = new_from_mlir_values(
            self._current_batch_idx, [values[2]]
        )
        new_accum_tile_m = new_from_mlir_values(self._accum_tile_m, [values[3]])
        new_cta_id_in_cluster = new_from_mlir_values(
            self.cta_id_in_cluster, values[4:7]
        )
        new_num_tiles_executed = new_from_mlir_values(
            self._num_tiles_executed, [values[7]]
        )
        return MaskedScheduler(
            self.params,
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_current_batch_idx,
            new_accum_tile_m,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
        )

    # called by host
    @dsl_user_op
    @staticmethod
    def create(
        params: MaskedSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ):
        params = params

        # persistent cluster 总数 = grid 中 CTA 总数 / 每个 cluster 的 CTA 数.
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )

        bidx, bidy, bidz = block_idx

        # bidz 是 persistent grid 中的 cluster id, 也是该 cluster 的初始 work index.
        current_work_linear_idx = Int32(bidz)
        # 从 expert 0 开始扫描.
        current_batch_idx = Int32(0)
        # 已跳过 expert 的累计 M tile 数, 用来把 linear_idx 还原到当前 expert 内坐标.
        accum_tile_m = Int32(0)

        # CTA 在 cluster 内的坐标.
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        num_tiles_executed = Int32(0)
        return MaskedScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            current_batch_idx,
            accum_tile_m,
            cta_id_in_cluster,
            num_tiles_executed,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: MaskedSchedulerParams,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Integer, Integer, Integer]:
        return params.get_grid_shape(max_active_clusters, loc=loc, ip=ip)

    # private method
    @cute.jit
    def _get_current_work_for_linear_idx(
        self,
        current_work_linear_idx: Int32,
        dsm_pending_packed: Optional[Uint64],
        dsm_counter: Optional[Uint8],
        num_c_stage: Optional[int] = None,
    ) -> Tuple[WorkTileInfo, Optional[Uint64]]:
        # 将 linear_idx 解码为 (tile_m, tile_n, expert_idx).
        # 这里不用 problem_layout_ncluster_mnl.get_hier_coord, 因为每个 expert 的
        # M tile 数是 masked_m 决定的动态值, 不能简单按 full padded grid 解码.
        num_tiles_n = self.params.problem_shape_ntile_mnl[1]
        accum_tile_m = self._accum_tile_m
        batch_idx = self._current_batch_idx

        # 跳过所有 linear_idx 已经越过的 expert.
        while (
            (
                accum_tile_m
                + cute.ceil_div(self.params.masked_m[batch_idx], self.params.c_tiler[0])
            )
            * num_tiles_n
            <= current_work_linear_idx
            and batch_idx < self.params.masked_m.shape[0]
        ):
            # dst_signals 模式下, 当 scheduler 跳过一个 expert 时, epilogue 需要记录:
            # dsm_counter 增长到什么值时, 这个 expert 的最后一个 C subtile 已完成.
            if cutlass.const_expr(
                (dsm_pending_packed is not None)
                and (self.params.dst_signals is not None)
            ):
                dsm_pending_packed = with_byte(
                    dsm_pending_packed,
                    index=batch_idx,
                    value=dsm_counter + (num_c_stage - 1),
                )

            accum_tile_m += cute.ceil_div(
                self.params.masked_m[batch_idx], self.params.c_tiler[0]
            )
            batch_idx += Int32(1)

        self._accum_tile_m = accum_tile_m
        self._current_batch_idx = batch_idx

        # 当前 expert 存在, 且 linear_idx 仍位于该 expert 的有效 M tile 范围内,
        # 这个 work tile 才有效.
        is_valid = self._current_batch_idx < self.params.masked_m.shape[0]
        if is_valid:
            is_valid = (
                self._accum_tile_m
                + cute.ceil_div(
                    self.params.masked_m[self._current_batch_idx],
                    self.params.c_tiler[0],
                )
            ) * num_tiles_n > current_work_linear_idx

        # cluster 坐标相对于当前 expert 解码:
        #   M cluster = linear_idx // N - 已跳过的 M tile 数
        #   N tile    = linear_idx % N
        #   L         = current expert
        cur_cluster_coord = (
            current_work_linear_idx // num_tiles_n - self._accum_tile_m,
            current_work_linear_idx % num_tiles_n,
            self._current_batch_idx,
        )

        # 再把 cluster 坐标转换成具体 CTA tile 坐标:
        #   tile_coord = cluster_coord * cluster_shape + cta_id_in_cluster
        cur_tile_coord = tuple(
            Int32(x) * Int32(z) + Int32(y)
            for x, y, z in zip(
                cur_cluster_coord,
                self.cta_id_in_cluster,
                (*self.params.cluster_shape_mn, Int32(1)),
                strict=True,
            )
        )

        return WorkTileInfo(cur_tile_coord, is_valid), dsm_pending_packed

    @dsl_user_op
    def get_current_work(
        self,
        dsm_pending_packed: Optional[Uint64] = None,
        dsm_counter: Optional[Uint8] = None,
        num_c_stage: Optional[int] = None,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[WorkTileInfo, Optional[Uint64]]:
        return self._get_current_work_for_linear_idx(
            self._current_work_linear_idx,
            dsm_pending_packed=dsm_pending_packed,
            dsm_counter=dsm_counter,
            num_c_stage=num_c_stage,
        )

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        tile_info, _ = self.get_current_work(loc=loc, ip=ip)
        return tile_info

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        # 每个 persistent cluster 按 num_persistent_clusters 为步长前进,
        # 保证不同 cluster 覆盖的 linear_idx 不重叠.
        self._current_work_linear_idx += Int32(advance_count) * Int32(
            self.num_persistent_clusters
        )
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


# ============================================================================
# 核心 Kernel: Sm100BlockScaledPersistentDenseGemmKernel
# ============================================================================
#
# 与 fp16_gemm_1.py 这类基础 GEMM 示例相比, 这里主要多了几层机制:
#
# 1. 数据类型: FP16 -> FP4/FP8 + block scale factor
#    FP16 GEMM 直接用 A/B 做 MMA; 本 kernel 的 A/B 是 block-scaled 低精度
#    数据, 每 16 或 32 个 K 元素共享一个 scale factor.MMA 指令为
#    tcgen05.mma.kind.block_scale, 所以除了 A/B, 还要加载 SFA/SFB.
#    SFA/SFB 先由 TMA 从 GMEM 搬到 SMEM, 再由 MMA warp copy 到 TMEM.
#
# 2. 调度方式: 固定 grid -> persistent tile scheduling
#    普通 GEMM 通常 grid = (ceil(M/tile_M), ceil(N/tile_N)), 每个 CTA 负责
#    一个固定 output tile.本 kernel 只启动有限数量的 persistent cluster,
#    每个 cluster 在 while loop 中不断领取下一个有效 tile.MoE 中不同
#    expert 的 token 数差异较大, 这种方式可以跳过 padding 并减少尾部空转.
#
# 3. Warp 特化: 单 warp/CTA 混合工作 -> TMA/MMA/Epilogue 三路并行
#    192 threads = 6 warps:
#      warp 0~3: Epilogue, TMEM -> Reg -> dtype convert/alpha -> SMEM -> TMA store
#      warp 4:   MMA, 等 AB pipeline, copy SF S2T, 发 tcgen05.mma
#      warp 5:   TMA load, GMEM -> SMEM 异步加载 A/B/SFA/SFB
#
# 4. Epilogue: Reg 直写 -> SMEM 中转后 TMA store
#    C 的写回路径是 TMEM accumulator -> register -> SMEM -> GMEM.多一次
#    SMEM 中转, 但可以使用 TMA bulk store, 并通过 c_pipeline 控制 store 深度.
#
# 5. Masked GEMM: 支持每个 expert 的变长 M
#    masked_m[expert] 记录该 expert 的有效 token 数.scheduler 只产生有效
#    M tile 的 work, padding 行不会进入 TMA/MMA/Epilogue 主循环.
"""
This example provides an experimental implementation of the SM100 batched dense blockscaled GEMM kernel, please note that the APIs and implementation details related to this kernel may change in future releases.

A high-performance persistent batched dense blockscaled GEMM example for the NVIDIA Blackwell SM100 architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M") for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K") for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk, which has M×ceil_div(K, sf_vec_size)×L elements respectively
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk, which has N×ceil_div(K, sf_vec_size)×L elements respectively

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp:
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
      or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Input arguments to this example is shown below:

.. code-block:: bash

    python examples/blackwell/dense_blockscaled_gemm_persistent.py            \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_blockscaled_gemm_persistent.py        \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1                                                  \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 128 or 256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.
"""


class Sm100BlockScaledPersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 128/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

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
        sm_version: str,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        """
        supported_sm_versions = ["sm_100", "sm_103"]
        assert sm_version in supported_sm_versions, (
            f"{supported_sm_versions} are the only supported SM versions for cute-dsl backend, but encountered {sm_version}"
        )

        # 累加器固定为 FP32, 低精度 A/B 只影响输入和 scale factor.
        self.acc_dtype = cutlass.Float32
        # 每个 scale factor 覆盖的 K 元素数:
        #   NVFP4 常见为 16
        #   MXFP4/MXFP8 常见为 32
        self.sf_vec_size = sf_vec_size
        # mma_tiler_mn[0] == 256 时启用 two-CTA MMA 指令.
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K 维度依赖 a_dtype.width, 在 _setup_attributes 里根据 dtype 计算.
        self.mma_tiler = (*mma_tiler_mn, 1)

        # CtaGroup.TWO 对应 2CTA 协作 MMA, CtaGroup.ONE 对应普通单 CTA MMA.
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # warp 0~3: epilogue
        # warp 4:   MMA
        # warp 5:   TMA load
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
        # NamedBarrier ID 分配:
        #   0: CTA/cluster 初始化同步
        #   1: epilogue warps 之间同步 SMEM 写入和 TMA store
        #   2: TMEM pointer 广播同步, epilogue 分配后 MMA/epilogue 读取
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.smem_capacity = utils.get_smem_capacity_in_bytes(sm_version)
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # MMA 指令的 K 维度由 256 bits / element_width 决定:
        #   FP4:  256 / 4  = 64
        #   FP8:  256 / 8  = 32
        #   FP16: 256 / 16 = 16
        # 因此 FP4 的单条 MMA 指令在 K 方向比 FP16 覆盖更多逻辑元素.
        # Size example: a/b_dtype=Float4E2M1FN, sf_dtype=Float8E4M3FN,
        # mma_inst_shape_mnk=(128, 128, 64), c_dtype=BFloat16.
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )
        # SFB 有独立的 TiledMMA 配置.2CTA 模式下主 MMA 的 M 维会拆成两个 CTA,
        # 而 SFB 的 partition 需要按 N/K 布局计算, 所以这里单独构造 shape.
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        # 构造 block-scaled TiledMMA:
        #   - A/B 从 SMEM 读取
        #   - SFA/SFB 从 TMEM 读取
        #   - accumulator 写入 TMEM
        # 数学上等价于 Acc += (A * SFA) @ (B * SFB)^T.
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        # SFB 专用 TiledMMA 固定用 CtaGroup.ONE, 主要服务于 SFB 的 TMA partition.
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )

        # 每个 CTA tile 在 K 方向包含 4 个 MMA instruction K slice.
        # FP4 + mma_inst_K=64 时, mma_tiler_K = 64 * 4 = 256.
        # Size example: mma_tiler=(128, 128, 256), cta_tile=(128, 128, 256)
        # in single-CTA mode.
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mnk_sfb[0],
            self.mma_inst_shape_mnk_sfb[1],
            self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # VMNK 中的 V 维是 two-CTA MMA 的 atom_thr 维度.单 CTA 时 V=1.
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # TMA multicast 方向:
        #   A/SFA 沿 N 维复用, 所以 multicast 给同一个 M,不同 N 的 CTA
        #   B/SFB 沿 M 维复用, 所以 multicast 给同一个 N,不同 M 的 CTA
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Epilogue subtile 控制 TMEM -> Reg -> SMEM 的分块大小.
        # subtile 越小, 单次寄存器压力越低; 但 TMA store 次数会增加.
        # Size example: epi_tile=(128, 32), so a CTA tile (128, 128) is written
        # as 4 epilogue subtiles.
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # 动态计算 pipeline stage 数.与简单 FP16 GEMM 不同, 这里 SMEM 里同时有:
        #   A/B staging, SFA/SFB staging, C staging, 以及多组 mbarrier.
        # stage 数取决于 dtype,tile shape,epilogue tile 和当前 SMEM 容量.
        # Size example: num_acc_stage=2, num_ab_stage=5, num_c_stage=5 for
        # FP4 (128,128) tiler on the tested SM100 setup.
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.a_major_mode,
            self.b_dtype,
            self.b_major_mode,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        # 计算 staged SMEM layout:
        #   sA/sB:   TMA load 后供 MMA 从 SMEM 读取
        #   sSFA/SFB:TMA load 后供 MMA warp 做 SMEM -> TMEM copy
        #   sC:      epilogue 先写 SMEM, 再由 TMA store 写 GMEM
        # FP4 size example:
        #   sA/sB:       S<3,4,3> o ((128,64),1,4,5):((256,1),0,64,32768)
        #   sSFA/sSFB:   ((((32,4),1),(16,4)),1,4,5):(((16,4),0),(0,1),0,512,2048)
        #   sC:          S<2,4,3> o ((8,16),(32,1),(1,5)):((32,256),(1,0),(0,4096))
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
        masked_m_tensor: cute.Tensor,
        dst_signals: Optional[cute.Pointer],
        alpha_tensor: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param masked_m_tensor: Masked layout tensor M
        :type masked_m_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param alpha_tensor: Optional 1D tensor of shape (l,) containing per-batch scaling factors.
        :type alpha_tensor: cute.Tensor
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # 先从 CuTe tensor 上读取 dtype 和 major mode.后面的 MMA shape,SMEM layout,
        # TMA descriptor 都依赖这些运行时属性.
        # Size example: a/b_dtype=Float4E2M1FN, sf_dtype=Float8E4M3FN,
        # c_dtype=BFloat16, A/B are K-major.
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # 根据 dtype/layout 设置 MMA,cluster,stage,SMEM/TMEM 相关属性.
        self._setup_attributes()

        # 将 SFA/SFB 解释成 block_scale MMA 需要的 atom layout:
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
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
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

        # Setup TMA load for B
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

        # Setup TMA load for SFA
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

        # Setup TMA load for SFB
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
        # Size example: FP4 (128,128,256) stage has 36864 bytes of TMA load
        # traffic per AB pipeline stage, counting A + B + SFA + SFB.
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size
        # 上面这个 tx_count 是一个 AB pipeline stage 的总 TMA 字节数.
        # TMA load warp 对每个 K block 发 4 笔 copy: A + B + SFA + SFB.
        # barrier 只有在这 4 笔 copy 对应的字节数都到齐后才放行 MMA warp.

        # C 的写回走 TMA store: Reg -> SMEM -> GMEM.
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # 计算 persistent grid.grid 的 z 维是可同时活跃的 persistent cluster 数,
        # 不是 output tile 总数.
        self.tile_sched_params, grid = self._compute_grid(
            masked_m_tensor,  # add masked layout
            dst_signals,
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # SharedStorage 包含:
        #   - AB pipeline full/empty barriers
        #   - Acc pipeline full/empty barriers
        #   - 2CTA TMEM dealloc barrier
        #   - TMEM pointer holding buffer
        #   - staged SMEM: sC, sA, sB, sSFA, sSFB
        # Size example: FP4 total SharedStorage is 226304 bytes (~221 KB) on
        # the tested configuration, including AB/SF/C stages and barriers.
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
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

        # Launch the kernel synchronously
        # Launch size example: grid=(1,1,148), block=(192,1,1),
        # cluster=(1,1,1), smem=226304 for FP4 single-CTA cluster on the tested
        # SM100 setup. 148 is the active SM count in that run.
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
            alpha_tensor,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
        )
        return

    # =========================================================================
    # GPU device kernel
    # =========================================================================
    #
    # 6 个 warp 分三组并行工作:
    #
    #   TMA warp (warp 5):
    #       persistent loop { scheduler 取 tile -> K-loop TMA load A/B/SFA/SFB }
    #
    #   MMA warp (warp 4):
    #       persistent loop { 等 AB full -> SF S2T -> tcgen05.mma -> Acc full }
    #
    #   Epilogue warps (warp 0~3):
    #       persistent loop { 等 Acc full -> T2R -> convert/alpha -> R2S -> TMA store C }
    #
    # Pipeline 关系:
    #
    #   TMA warp -- ab_pipeline --> MMA warp -- acc_pipeline --> Epilogue warps
    #                                                            |
    #                                                        c_pipeline
    #                                                            |
    #                                                     TMA store C -> GMEM
    #
    # Launch size example for FP4 debug input (L=4, M=256, K=512, N=256):
    #   grid    = (1, 1, 148), persistent grid, z = num_SM
    #   block   = (192, 1, 1), 6 warps
    #   cluster = (1, 1, 1), single CTA cluster
    #   smem    = 226304 bytes, AB*5 + SF*5 + Acc*2 + C*5 + barriers
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
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        alpha: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: MaskedSchedulerParams,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # 预取 TMA descriptor.只有 TMA warp 会发起 GMEM<->SMEM TMA 操作,
        # 所以 descriptor prefetch 也放在 TMA warp 中.
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # CTA / thread 坐标
        #
        # bidx/bidy: CTA 在 cluster 内的 M/N 方向坐标
        # bidz: persistent cluster id, 也是 scheduler 的初始 linear_idx
        bidx, bidy, bidz = cute.arch.block_idx()
        # two-CTA MMA 中的 V 坐标, 0 是 leader CTA, 1 是 follower CTA.
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
        # CTA 内 thread id.
        tidx, _, _ = cute.arch.thread_idx()

        #
        # 分配 SMEM 并初始化各类 barrier / pipeline 状态.
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # AB pipeline: TMA warp producer -> MMA warp consumer.
        # 一个 stage 表示 A/B/SFA/SFB 当前 K block 都已从 GMEM 到达 SMEM.
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

        # Acc pipeline: MMA warp producer -> Epilogue warps consumer.
        # 一个 stage 表示当前 output tile 的 TMEM accumulator 已经计算完成.
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = (
            self.threads_per_warp
            * len(self.epilog_warp_id)
            * (2 if use_2cta_instrs else 1)
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

        # 2CTA 模式下两个 CTA 共享 TMEM allocation, 释放前需要额外同步.
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )
        cute.arch.mbarrier_init_fence()

        # barrier 初始化完成后, cluster 内 CTA 再继续进入数据搬运和 TMEM 分配.
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # 构造 SMEM tensor 视图.
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
        # 计算 TMA multicast mask.
        #   A/SFA: 沿 N 维 multicast
        #   B/SFB: 沿 M 维 multicast
        # SFB 使用 cluster_layout_sfb_vmnk, 因为它有独立的 TiledMMA partition.
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
        # 对 global tensor 做 local_tile 分块, 但暂不选择具体 RestM/RestN/RestL.
        # persistent scheduler 后面会动态选择当前 tile 坐标.
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
        k_block_cnt = cute.size(gA_mkl, mode=[3])
        # Size example: total_K=512 and mma_tiler_K=256 gives k_block_cnt=2.

        #
        # 按当前 CTA 的 MMA slice 对 global tile 做 partition.
        # 命名约定:
        #   tCgA = thread/MMA view of Cta/global A
        #   tCgB = thread/MMA view of Cta/global B
        #   tCgC = thread/MMA view of Cta/global C
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
        # TMA load partition.
        # 这里保留 RestM/RestN/RestL 维度, 因为每个 persistent iteration
        # 都要根据 scheduler 的 tile 坐标切出当前 GMEM slice.
        #
        # TMA load A partition_S/D
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
        # TMA load B partition_S/D
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

        #  TMA load SFA partition_S/D
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

        # TMA load SFB partition_S/D
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
        # 构造 MMA fragment:
        #   tCrA/tCrB: MMA 从 SMEM 读取 A/B 的视图
        #   tCtAcc_fake: 只有 layout 正确的占位 accumulator, 指针稍后由 TMEM alloc 填入
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
        # 所有 barrier 初始化完成后, 才能开始 TMEM alloc 和数据搬运.
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        #
        # Specialized TMA load warp (warp 5)
        #
        # 职责:
        #   1. 从 MaskedScheduler 获取当前 tile
        #   2. 对每个 K block 发起 A/B/SFA/SFB 四笔 TMA load
        #   3. 通过 ab_pipeline 通知 MMA warp 数据已到达 SMEM
        # Size example: each K block issues 4 TMA copies, 36864 bytes/stage for
        # the FP4 (128,128,256) debug configuration.
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = MaskedScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # 当前 work_tile 的坐标为 (tile_m, tile_n, expert_idx).
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # 根据当前 tile 坐标切出 A/B/SFA/SFB 的 GMEM slice.
                # K 维 RestK 保留给下面的 K-loop.
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

                # 非阻塞预探测第一个 AB empty stage.若成功, 下面 acquire 可直接跳过等待.
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_block_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # K-loop: 每个 k_block 对应一个 mma_tiler_K 范围.
                #
                for k_block in cutlass.range(0, k_block_cnt, 1, unroll=1):  # noqa: B007
                    # 如果预探测没成功, 这里阻塞等待对应 AB stage 可写.
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # 发起四笔异步 TMA copy.它们共享同一个 ab_pipeline barrier,
                    # barrier 的 tx_count 是 A+B+SFA+SFB 的总字节数.
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

                    # 推进 stage 指针, 并预探测下一个 AB empty stage.
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_block_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # 当前 output tile 的所有 K block 都已发起 TMA load, 转向下一个 tile.
                #
                tile_sched.advance_to_next_work()
                work_tile, _ = tile_sched.get_current_work()

            #
            # 等待所有 TMA producer 信号被 consumer 正确接收.
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp (warp 4)
        #
        # 职责:
        #   1. 等待 epilogue 分配 TMEM 并广播指针
        #   2. 等待 TMA warp 将 A/B/SFA/SFB 加载到 SMEM
        #   3. 将 SFA/SFB 从 SMEM copy 到 TMEM
        #   4. 设置 tcgen05.Field.SFA/SFB, 发 block_scale MMA
        #   5. 通过 acc_pipeline 通知 epilogue accumulator 已就绪
        # Size example: K=512 with mma_tiler_K=256 means each output tile has
        # 2 K-blocks in the MMA mainloop.
        #
        if warp_idx == self.mma_warp_id:
            #
            # 等待 epilogue warp 0 完成 TMEM alloc 并写入 tmem_holding_buf.
            #
            tmem_ptr_read_threads = self.threads_per_warp * len(
                (self.mma_warp_id, *self.epilog_warp_id)
            )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # 取回 TMEM 指针并构造 Acc/SFA/SFB tensor.
            #
            # TMEM layout:
            #   [Accumulator columns] [SFA columns] [SFB columns]
            #
            # acc_tmem_ptr 指向 accumulator 开始, SFA/SFB 紧随其后.
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA 紧跟在 Acc 之后, find_tmem_tensor_col_offset 计算前一个 tensor
            # 占用的 TMEM column 数.
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

            # SFB 紧跟在 SFA 之后.
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
            # 配置 scale factor 的 SMEM -> TMEM copy.
            # tcgen05.Cp4x32x128bOp 是 block_scale MMA 特有的 SF copy 路径.
            #
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = MaskedScheduler.create(
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
                # 与 TMA warp 使用相同 scheduler 逻辑, 得到同一个 output tile.
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # 选择当前 output tile 对应的 accumulator stage.
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # 非阻塞预探测第一个 AB full stage.
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # 等待 epilogue 释放当前 accumulator stage, 确保不会覆盖仍在写回的 TMEM.
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # 每个新 tile 的第一条 MMA 需要覆盖 accumulator, 后续 kphase 才累加.
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # MMA K-loop.
                # 每个 k_block:
                #   1. 等待 A/B/SFA/SFB 对应 SMEM stage 就绪
                #   2. copy SFA/SFB from SMEM to TMEM
                #   3. 对该 K block 内的每个 kphase 发 MMA
                #
                for k_block in cutlass.range_constexpr(k_block_cnt):  # noqa: B007
                    if is_leader_cta:
                        # 如果预探测没成功, 这里阻塞等待 TMA producer 完成.
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        # block_scale 特有: 当前 stage 的 SFA/SFB 从 SMEM copy 到 TMEM.
                        # MMA 指令随后会从 TMEM 读取 SF.
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

                        # 一个 k_block 内可能有多个 kphase.每个 kphase 对应一条
                        # tcgen05.mma.kind.block_scale:
                        #   Acc += (A[kphase] * SFA[kphase]) @ (B[kphase] * SFB[kphase])^T
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (
                                None,
                                None,
                                kphase_idx,
                                ab_consumer_state.index,
                            )

                            # 发射 MMA 前, 先把当前 kphase 的 SFA/SFB TMEM 指针写入
                            # tiled_mma 的指令字段.
                            sf_kphase_coord = (None, None, kphase_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kphase_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB[sf_kphase_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kphase_coord],
                                tCrB[kphase_coord],
                                tCtAcc,
                            )

                            # 第一条 MMA 后开启累加模式.
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # 释放当前 AB stage 给 TMA warp 复用.
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # 推进 AB consumer 指针, 并预探测下一个 full stage.
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # 当前 output tile 的 K-loop 完成, 通知 epilogue 可以读取 accumulator.
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # 前进到下一个 output tile.
                #
                tile_sched.advance_to_next_work()
                work_tile, _ = tile_sched.get_current_work()

            #
            # 等待最后的 acc commit 被 epilogue 消费.
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps (warp 0~3)
        #
        # 职责:
        #   1. 分配 TMEM, 并把指针广播给 MMA warp
        #   2. 等待 acc_pipeline full
        #   3. 将 accumulator 从 TMEM load 到 register
        #   4. 做 optional alpha 和 dtype conversion
        #   5. register -> SMEM, 再 TMA store 到 GMEM
        #   6. 如果启用 dst_signals, 在 C store 可见后通知 combine
        # Size example: epi_tile=(128,32), so one CTA tile (128,128) is split
        # into 4 subtile stores.
        #
        if warp_idx < self.mma_warp_id:
            #
            # Epilogue warp 0 负责分配 TMEM.MMA warp 不分配, 只通过 barrier
            # 读取 epilogue 写入 shared buffer 的 TMEM pointer.
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # 同步 MMA warp 和所有 epilogue warp, 确保 TMEM pointer 已写入.
            #
            tmem_ptr_read_threads = self.threads_per_warp * len(
                (self.mma_warp_id, *self.epilog_warp_id)
            )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # 取回 accumulator 的 TMEM pointer, 构造 epilogue 读取视图.
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Epilogue 的三段 copy chain:
            #   Step 1: TMEM -> Register (T2R)
            #   Step 2: Register -> SMEM (R2S)
            #   Step 3: SMEM -> GMEM (TMA store)
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )

            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = MaskedScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # C store pipeline 控制 SMEM -> GMEM 的 TMA store 深度.
            # 这是 store pipeline, 没有像 AB/Acc 那样的独立 consumer warp.
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            if cutlass.const_expr(tile_sched_params.dst_signals is not None):
                assert self.num_c_stage < 256, "must be representable in 1 byte"
                num_experts = tile_sched_params.masked_m.shape[0]
                assert num_experts <= 8, "need to be packable into a u64"
            # dst_signals 追踪状态:
            #   dsm_counter: 已提交的 C subtile 计数
            #   dsm_pending_packed: 每个 expert 期望看到的 dsm_counter 值
            #   dsm_pending_idx: 下一个待发送 signal 的 expert
            dsm_pending_packed = Uint64(0)
            dsm_pending_idx = Int32(0)
            dsm_counter = Uint8(0)

            while work_tile.is_valid_tile:
                # 当前 output tile 坐标.
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # 切出当前 tile 对应的 GMEM C partition.
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

                # 选择当前 accumulator stage.
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                #
                # 等待 MMA warp 完成当前 tile 的 accumulator.
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # 按 subtile 写回 accumulator.
                # 一个 CTA tile 可能被拆成多个 epilogue subtile, 用来降低寄存器压力,
                # 同时把 C store 分散到 c_pipeline 的多个 stage.
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    #
                    # Step 1: TMEM -> Register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Step 2: optional alpha, C[:, :, expert] = alpha[expert] * Acc
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cutlass.const_expr(alpha is not None):
                        acc_vec = acc_vec * alpha[work_tile.tile_idx[2]]

                    acc_vec = acc_vec.to(self.c_dtype)
                    tRS_rC.store(acc_vec)

                    #
                    # Step 3: Register -> SMEM.
                    # TMA store 要从 SMEM 读 C, 所以不能直接从 register 写 GMEM.
                    #
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # 确保 SMEM 写入对 TMA store 单元可见.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilog_threads = self.threads_per_warp * len(self.epilog_warp_id)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    #
                    # Step 4: SMEM -> GMEM.只有 epilogue warp 0 发起 TMA store.
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, subtile_idx)],
                        )

                        c_pipeline.producer_commit()

                        if cutlass.const_expr(
                            tile_sched_params.dst_signals is not None
                        ):
                            dsm_counter = (dsm_counter + 1).to(Uint8)
                            will_write_signals = (
                                read_byte(dsm_pending_packed, dsm_pending_idx)
                                == dsm_counter
                            )

                            if will_write_signals:
                                # The original c_pipeline.producer_acquire()
                                #   := PipelineTmaStore.producer_acquire()
                                #   := TmaStoreFence.wait()
                                #   := cute.arch.cp_async_bulk_wait_group(self.num_stages - 1, read=True)
                                cute.arch.cp_async_bulk_wait_group(
                                    self.num_c_stage - 1,
                                    # read=False 不只等 read group, 也等 TMA writes 完成.
                                    # 这里必须确保 C 已写到 GMEM, 再发送 dst_signal.
                                    read=False,
                                )
                            else:
                                c_pipeline.producer_acquire()

                        else:
                            c_pipeline.producer_acquire()

                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    if cutlass.const_expr(tile_sched_params.dst_signals is not None):
                        lane_id = tidx % 32
                        if warp_idx == self.epilog_warp_id[0] and lane_id == 0:
                            while (dsm_pending_idx < num_experts) and (
                                read_byte(dsm_pending_packed, dsm_pending_idx)
                                == dsm_counter
                            ):
                                # 发送当前 expert 的完成信号.release 原子保证 signal 前的
                                # C store 对下游 combine 可见.
                                atomic_add_release_global(
                                    tile_sched_params.dst_signals.toint()
                                    + sizeof_i32 * dsm_pending_idx,
                                    value=1,
                                )
                                dsm_pending_idx += 1

                #
                # 释放 accumulator stage 给 MMA warp, 允许后续 tile 复用这段 TMEM.
                #
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # 前进到下一个 tile, 并在跳过 expert 时更新 dsm_pending_packed.
                #
                tile_sched.advance_to_next_work()
                work_tile, dsm_pending_packed = tile_sched.get_current_work(
                    dsm_pending_packed=dsm_pending_packed,
                    dsm_counter=dsm_counter,
                    num_c_stage=self.num_c_stage,
                )

            #
            # 释放 TMEM.
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = self.threads_per_warp * len(self.epilog_warp_id)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                # 2CTA 模式下, 两个 CTA 都不再使用 TMEM 后才能 dealloc.
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            #
            # 等待所有 C store 完成, 并补发剩余 dst_signals.
            #
            if cutlass.const_expr(tile_sched_params.dst_signals is not None):
                # The original c_pipeline.producer_tail()
                #   := PipelineTmaStore.producer_tail()
                #   := TmaStoreFence.tail()
                #   := cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.cp_async_bulk_wait_group(
                    0,
                    # read=False 保证 TMA writes 完成后再发送最终信号.
                    read=False,
                )

                lane_id = tidx % 32
                if warp_idx == self.epilog_warp_id[0] and lane_id == 0:
                    while dsm_pending_idx < num_experts:
                        # 某些 expert 的 signal 可能在主循环内没有刚好触发,
                        # tail 阶段统一补发.
                        atomic_add_release_global(
                            tile_sched_params.dst_signals.toint()
                            + sizeof_i32 * dsm_pending_idx,
                            value=1,
                        )
                        dsm_pending_idx += 1

            else:
                c_pipeline.producer_tail()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        # Scale Factor 的 SMEM -> TMEM copy 配置.
        #
        # block_scale MMA 要求 SFA/SFB 放在 TMEM 中, 但 TMA load 只能先把它们
        # 搬到 SMEM.这里用 tcgen05.Cp4x32x128bOp 配置 S2T copy, 把 SMEM 中
        # staged 的 SF 搬到 TMEM, 供后续 tcgen05.mma.kind.block_scale 读取.
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
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
        # Epilogue Step 1: TMEM -> Register.
        # accumulator 在 TMEM 中按 tiled_mma 的 C layout 存放, 这里根据 epi_tile
        # 拆成 epilogue subtile, 为每个 epilogue thread 生成 T2R partition.
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
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
        tTR_rAcc = cute.make_fragment(
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
        # Epilogue Step 2: Register -> SMEM.
        # T2R 得到的是 FP32 accumulator register fragment.做完 alpha 和 dtype
        # conversion 后, 需要先写到 sC, 再由 TMA store 写 GMEM.
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
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
        # Epilogue Step 3: SMEM -> GMEM.
        # 这里配置 TMA store 的 source/destination partition.bSG_sC 是 SMEM
        # 视图, bSG_gC 是 GMEM C tile 视图, 后续 persistent loop 再按当前 tile
        # 坐标切片.
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
            - tma_atom_c: The TMA copy atom
            - bSG_sC: The partitioned shared memory tensor C
            - bSG_gC: The partitioned global tensor C
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
        a_major_mode: tcgen05.OperandMajorMode,
        b_dtype: Type[cutlass.Numeric],
        b_major_mode: tcgen05.OperandMajorMode,
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param a_major_mode: Major mode of operand A.
        :type a_major_mode: tcgen05.OperandMajorMode
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param b_major_mode: Major mode of operand B.
        :type b_major_mode: tcgen05.OperandMajorMode
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Acc stage 数:
        #   N tile = 256 时用 1 stage
        #   N tile = 128 时用 2 stage
        # 这样可以在较窄 N tile 下更好地重叠 MMA 和 epilogue.
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # C store pipeline 至少保留 2 个 stage, 后面若 SMEM 有剩余会继续增加.
        num_c_stage = 2

        # 先计算单 stage 的 SMEM footprint:
        #   AB stage = A + B + SFA + SFB
        #   C stage  = epilogue TMA store 的 SMEM staging
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
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
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

        # 先为 AB pipeline 分配尽可能多的 stage:
        #   可用 SMEM = smem_capacity / occupancy - barrier 预留 - 初始 C stages
        #   num_ab_stage = 可用 SMEM / 单个 AB stage bytes
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # 剩余 SMEM 分给 C pipeline, 提高 epilogue store 的缓冲深度.
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        masked_m_tensor: cute.Tensor,
        dst_signals: Optional[cute.Pointer],
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[MaskedSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[MaskedSchedulerParams, tuple[int, int, int]]
        """
        # c_tiler 只取 CTA tile 的 M/N 维, K 维对输出 tile 网格无意义.
        c_tiler = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        # MaskedSchedulerParams 会同时保存 masked_m,dst_signals,C tensor 和
        # tile/cluster shape, kernel 内 TMA/MMA/Epilogue 三条路径共用这套调度参数.
        tile_sched_params = MaskedSchedulerParams(
            masked_m_tensor, dst_signals, c, c_tiler, cluster_shape_mnl
        )
        # grid = (ClusterM, ClusterN, max_active_clusters).
        grid = MaskedScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
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
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
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
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [128, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
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
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
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
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement


# ============================================================================
# Scale Factor 布局转换工具
# ============================================================================
#
# tcgen05.mma.kind.block_scale 要求 SFA/SFB 按 MMA atom layout 排列.
# 逻辑上 SF 是 (M, ceil_div(K, sf_vec_size), L), 但 MMA 需要:
#
#   (m32, m4, rest_m, k4, rest_k, L)
#
# 其中:
#   m32 * m4 = 32 * 4 = 128, 对应一个 MMA tile 在 M/N 方向的 SF atom
#   k4 是一个 SF atom 覆盖的 K 方向 group
#   rest_m/rest_k 是剩余 tile 维度
#
# create_scale_factor_tensor 是测试/样例辅助函数, 用来构造 reference SF 和
# CuTeDSL 需要的 6D layout.生产路径中 grouped_gemm_nt_masked 直接接收已经
# 准备好的 SFA/SFB tensor.
@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L_mma_spec(
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)


# Create scale factor tensor SFA/SFB
def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype, device):
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

    # Create f32 ref torch tensor
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

    # Create f32 cute torch tensor
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

    # convert ref f32 tensor to cute f32 tensor
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )
    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.to(device, non_blocking=True)

    # reshape makes memory contiguous
    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    # prune to mkl for reference check.
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]
    ref_f32_torch_tensor = ref_f32_torch_tensor_cpu.to(device, non_blocking=True)

    # Create dtype cute torch tensor (cpu)
    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    # Convert f32 cute tensor to dtype cute tensor
    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=True,
    )
    return ref_f32_torch_tensor, cute_tensor, cute_torch_tensor


# ============================================================================
# Host 侧 CuTeDSL JIT 入口
# ============================================================================
#
# MaskedBatchedMatmulCuteDSL 是 @cute.jit 编译入口.它接收 raw pointer,
# 在 JIT 内部重建 CuTe tensor layout, 然后调用 device kernel.
#
# 为什么用 raw pointer:
#   get_cute_dsl_compiled_masked_gemm_kernel 用 functools.cache 缓存编译结果.
#   编译 cache key 由 shape,dtype,layout,tile shape,cluster shape,
#   sm_count,sm_version,dst_signals 是否启用等静态参数组成, tensor 地址在调用时传入.
class MaskedBatchedMatmulCuteDSL:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        ab_dtype: torch.dtype,
        sf_dtype: torch.dtype,
        c_dtype: torch.dtype,
        alpha_dtype: torch.dtype,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        sm_count: int,
        sm_version: str,
    ):
        self._m = m
        self._n = n
        self._k = k
        self._l = l
        self._a_major = a_major
        self._b_major = b_major
        self._c_major = c_major
        self._ab_dtype = ab_dtype
        self._sf_dtype = sf_dtype
        self._c_dtype = c_dtype
        self._alpha_dtype = alpha_dtype
        self._sf_vec_size = sf_vec_size
        self._mma_tiler_mn = mma_tiler_mn
        self._cluster_shape_mn = cluster_shape_mn

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
                f"MaskedBatchedMatmulCuteDSL: Unsupported with {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
            )

        # 计算 persistent scheduler 可用的最大 cluster 数.
        # sm_count 可以小于物理 SM 数, 用来给其他 kernel 或 combine 阶段预留 SM.
        self._max_active_clusters = min(
            get_max_active_clusters(
                self._cluster_shape_mn[0] * self._cluster_shape_mn[1]
            ),
            sm_count,
        )
        self._sm_version = sm_version

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        masked_m_ptr: cute.Pointer,
        dst_signals_ptr: Optional[cute.Pointer],
        alpha_ptr: cute.Pointer,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout(
                (self._m, self._k, self._l),
                order=(0, 1, 2) if self._a_major == "m" else (1, 0, 2),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._n, self._k, self._l),
                order=(0, 1, 2) if self._b_major == "n" else (1, 0, 2),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (self._m, self._n, self._l),
                order=(0, 1, 2) if self._c_major == "m" else (1, 0, 2),
            ),
        )

        # 计算 SF tensor 的 6D atom layout.
        # 逻辑 SF K 维是 ceil_div(K, sf_vec_size), 再按 k4=4 分组.
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(self._k, self._sf_vec_size)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape_a = (
            self._l,
            ceil_div(self._m, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )
        mma_shape_b = (
            self._l,
            ceil_div(self._n, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sfa_tensor = cute.make_tensor(
            sfa_ptr,
            layout=cute.make_ordered_layout(
                mma_shape_a,
                order=mma_permute_order,
            ),
        )
        sfb_tensor = cute.make_tensor(
            sfb_ptr,
            layout=cute.make_ordered_layout(
                mma_shape_b,
                order=mma_permute_order,
            ),
        )
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L_mma_spec(sfa_tensor)
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L_mma_spec(sfb_tensor)

        masked_m_tensor = cute.make_tensor(
            masked_m_ptr,
            layout=cute.make_ordered_layout((self._l,), order=(0,)),
        )

        # Use const_expr for compile-time conditional
        alpha_tensor = (
            cute.make_tensor(
                alpha_ptr,
                layout=cute.make_ordered_layout((self._l,), order=(0,)),
            )
            if cutlass.const_expr(alpha_ptr is not None)
            else None
        )

        Sm100BlockScaledPersistentDenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            sm_version=self._sm_version,
        )(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            masked_m_tensor,
            dst_signals_ptr,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
        )


@functools.cache
def get_cute_dsl_compiled_masked_gemm_kernel(
    m: int,              # M 维度 (各 expert 最大 token 数, 如 2048)
    n: int,              # N 维度 (输出特征数, Gemm1 为 4096, Gemm2 为 7168)
    k: int,              # K 维度 (输入特征数, 逻辑值非 packed, 如 7168 或 2048)
    l: int,              # L 维度 (batch = num_experts, 如 64)
    a_major: str,        # "k" (A 为 K-major, 当前唯一支持的布局)
    b_major: str,        # "k" (B 为 K-major, 当前唯一支持的布局)
    c_major: str,        # "n" (C 输出为 N-major)
    ab_dtype: Type[cutlass.Numeric],     # NVFP4 时为 cutlass.Float4E2M1FN
    sf_dtype: Type[cutlass.Numeric],     # NVFP4 block scale 为 cutlass.Float8E4M3FN
    c_dtype: Type[cutlass.Numeric],      # 输出为 cutlass.BFloat16
    alpha_dtype: Optional[Type[cutlass.Numeric]],  # cutlass.Float32 或 None
    sf_vec_size: int,    # NVFP4 为 16 (每个 scale factor block 的元素数)
    mma_tiler_mn: Tuple[int, int],       # (128, 128) - MMA tile shape
    cluster_shape_mn: Tuple[int, int],   # (1, 1) - CTA cluster shape
    sm_count: int,       # persistent scheduler 使用的 SM 数 (如 148)
    sm_version: str,     # Blackwell 为 "sm_100"
    enable_dst_signals: bool,  # True 时启用 epilogue -> combine 信号机制
) -> Callable:
    # functools.cache 会按完整参数 tuple 缓存编译好的 CuTeDSL kernel.
    # DeepSeek MoE 常见两类 GEMM:
    #   Gemm1: gate/up projection, K 大,N 相对小
    #   Gemm2: down projection, K 相对小,N 大
    # 两类 shape 会分别触发一次 JIT 编译, 后续调用复用 kernel.
    #
    # Size examples:
    #   Gemm1: m=2048, n=4096, k=7168, l=64
    #          M-tiles=2048/128=16, N-tiles=4096/128=32, K-blocks=7168/256=28
    #   Gemm2: m=2048, n=7168, k=2048, l=64
    #          M-tiles=2048/128=16, N-tiles=7168/128=56, K-blocks=2048/256=8
    #   dtypes: ab_dtype=Float4E2M1FN, sf_dtype=Float8E4M3FN, c_dtype=BFloat16
    #   sf_vec_size=16, mma_tiler_mn=(128,128), cluster_shape_mn=(1,1)
    def get_cute_pointers(
        input_tensors: Optional[List[torch.tensor]],
    ) -> List[cute.Pointer]:
        if input_tensors is None:
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                masked_m_data_ptr,
                dst_signals_data_ptr,
                alpha_data_ptr,
            ) = [16 for _ in range(8)]

            if not enable_dst_signals:
                dst_signals_data_ptr = None

        else:
            (
                a_tensor_gpu,
                b_tensor_gpu,
                sfa_tensor_gpu,
                sfb_tensor_gpu,
                c_tensor_gpu,
                masked_m_tensor_gpu,
                dst_signals_tensor_gpu,
                alpha_tensor_gpu,
            ) = input_tensors

            assert enable_dst_signals == (dst_signals_tensor_gpu is not None)

            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                masked_m_data_ptr,
                dst_signals_data_ptr,
                alpha_data_ptr,
            ) = (
                a_tensor_gpu.data_ptr(),
                b_tensor_gpu.data_ptr(),
                sfa_tensor_gpu.data_ptr(),
                sfb_tensor_gpu.data_ptr(),
                c_tensor_gpu.data_ptr(),
                masked_m_tensor_gpu.data_ptr(),
                dst_signals_tensor_gpu.data_ptr()
                if dst_signals_tensor_gpu is not None
                else None,
                alpha_tensor_gpu.data_ptr() if alpha_tensor_gpu is not None else None,
            )

        a_ptr = make_ptr(
            ab_dtype,
            a_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        b_ptr = make_ptr(
            ab_dtype,
            b_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        sfa_ptr = make_ptr(
            sf_dtype,
            sfa_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        sfb_ptr = make_ptr(
            sf_dtype,
            sfb_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_ptr = make_ptr(
            c_dtype,
            c_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        masked_m_ptr = make_ptr(
            cutlass.Int32,
            masked_m_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        dst_signals_ptr = (
            make_ptr(
                cutlass.Uint32,
                dst_signals_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if dst_signals_data_ptr is not None
            else None
        )
        alpha_ptr = (
            make_ptr(
                alpha_dtype,
                alpha_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if alpha_data_ptr is not None and alpha_dtype is not None
            else None
        )

        return [
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            c_ptr,
            masked_m_ptr,
            dst_signals_ptr,
            alpha_ptr,
        ]

    kernel = cute.compile(
        MaskedBatchedMatmulCuteDSL(
            m=m,
            n=n,
            k=k,
            l=l,
            a_major=a_major,
            b_major=b_major,
            c_major=c_major,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            alpha_dtype=alpha_dtype,
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sm_count=sm_count,
            sm_version=sm_version,
        ),
        *get_cute_pointers(None),
        cutlass_torch.current_stream(),
    )

    def tensor_api(
        a_tensor_gpu: torch.Tensor,
        b_tensor_gpu: torch.Tensor,
        sfa_tensor_gpu: torch.Tensor,
        sfb_tensor_gpu: torch.Tensor,
        masked_m_tensor_gpu: torch.Tensor,
        dst_signals_tensor_gpu: torch.Tensor,
        c_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_tensor_gpu: Optional[torch.Tensor] = None,
    ):
        # Tensor API size examples:
        #   a_tensor_gpu:        Gemm1 (2048,3584,64), Gemm2 (2048,1024,64), uint8 FP4 packed
        #   b_tensor_gpu:        Gemm1 (4096,3584,64), Gemm2 (7168,1024,64), uint8 FP4 packed
        #   sfa_tensor_gpu:      Gemm1 (32,4,16,4,112,64), Gemm2 (32,4,16,4,32,64)
        #   sfb_tensor_gpu:      Gemm1 logical scale (64,4096,448), Gemm2 (64,7168,128)
        #   c_tensor_gpu:        Gemm1 (2048,4096,64), Gemm2 (2048,7168,64), bf16
        #   masked_m_tensor_gpu: (64,), int32, valid token count per expert
        #   alpha_tensor_gpu:    (1,1,64) or equivalent per-expert alpha, float32
        if c_tensor_gpu is None:
            # fp4 gemm output is not supported
            c_tensor_gpu = torch.empty(
                (l, m, n),
                dtype=cutlass_to_torch_dtype(c_dtype),
                device="cuda",
            )

        # fp4 or fp8 torch tensor to cute tensor
        current_stream = cutlass_torch.current_stream()

        nonlocal kernel
        kernel(
            *get_cute_pointers(
                [
                    a_tensor_gpu,
                    b_tensor_gpu,
                    sfa_tensor_gpu,
                    sfb_tensor_gpu,
                    c_tensor_gpu,
                    masked_m_tensor_gpu,
                    dst_signals_tensor_gpu,
                    alpha_tensor_gpu,
                ]
            ),
            current_stream,
        )

        return c_tensor_gpu

    return tensor_api


# ============================================================================
# Public API: grouped_gemm_nt_masked
# ============================================================================
#
# Blackwell SM100+ 的 masked batched GEMM 入口.
# 输入约定:
#   lhs = (A, SFA)
#   rhs = (B, SFB)
#   A:   逻辑 (M, K, L), FP4 时实际 storage shape 是 (M, K/2, L)
#   B:   逻辑 (N, K, L), FP4 时实际 storage shape 是 (N, K/2, L)
#   out: 逻辑 (M, N, L)
#   masked_m: (L,), 每个 expert 的有效 M 行数
#
# DeepSeek-style size examples:
#   Gemm1 lhs:
#     A   = (2048, 3584, 64) uint8 FP4 packed, logical K=7168
#     SFA = (32, 4, 16, 4, 112, 64) float8 scale
#   Gemm1 rhs:
#     B   = (4096, 3584, 64) uint8 FP4 packed
#     SFB = (64, 4096, 448) float8 scale
#   Gemm1 out = (2048, 4096, 64) bf16
#
#   Gemm2 lhs:
#     A   = (2048, 1024, 64) uint8 FP4 packed, logical K=2048
#     SFA = (32, 4, 16, 4, 32, 64) float8 scale
#   Gemm2 rhs:
#     B   = (7168, 1024, 64) uint8 FP4 packed
#     SFB = (64, 7168, 128) float8 scale
#   Gemm2 out = (2048, 7168, 64) bf16
#
# Runtime size examples:
#   masked_m = (64,) int32, sum~1100, avg~17, max~192 in the debug sample
#   alpha    = (1, 1, 64) float32 when per-expert epilogue scaling is enabled
#   sm_count = 148 on the L20C/SM100 debug run; Gemm2 may reserve SMs for combine
#
@flashinfer_api(trace=grouped_gemm_nt_masked_trace)
def grouped_gemm_nt_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],  # activation (A, SFA) 对
    rhs: Tuple[torch.Tensor, torch.Tensor],  # 权重 (B, SFB) 对
    out: torch.Tensor,  # 预分配输出 tensor, 调用方 permute 为 (M,N,L); (M,N,L) bf16
    masked_m: torch.Tensor,  # 每 expert 有效 token 数, 超出部分被 masked scheduler 跳过; (L,) int32; (64,) sum~1100, avg~17, max~192
    *,
    ab_dtype: str,  # A/B 元素类型; NVFP4 用 "float4_e2m1fn", 也支持 "float8_e4m3fn"/"float8_e5m2"
    sf_dtype: str,  # scale factor 类型; NVFP4 用 "float8_e4m3fn", MXFP4 用 "float8_e8m0fnu"
    c_dtype: str,  # 输出元素类型; DeepSeek MoE 用 "bfloat16"
    sf_vec_size: int,  # 每个 scale factor block 覆盖的元素数; NVFP4=16, MXFP4=32
    dst_signals: Optional[torch.Tensor] = None,  # epilogue-to-combine 信号, kernel 写完 expert 后通知 combine; (L,) uint32
    sm_count: Optional[int] = None,  # 限制使用 SM 数, 默认全部可用; Gemm2 可减少以留 SM 给 combine; 148 (L20C)
    **kwargs,  # alpha: 每 expert 输出 scale, epilogue 融合 C=alpha*(A@B^T); (1,1,L) float32; (1,1,64); alpha_dtype: "float32"; mma_tiler_mn: MMA tile (128,128); cluster_shape_mn: (1,1)
):
    """
    Executes a masked, batched matrix multiplication (GEMM) with scale factors and optional alpha scaling at output.

    Args:
        lhs (Tuple[torch.Tensor, torch.Tensor]): Tuple containing the left-hand side input tensor (A) and its scale factor tensor (SFA).
            - A should be in (m, k, l) order, but physically (l, m, k). For fp4 tensor with 8-bit storage, we expect the shape to be (m, k/2, l).
            - SFA should be in (m32, m4, rm, k4, rk, l) order, but physically (l, rm, rk, m32, m4, k4)
        rhs (Tuple[torch.Tensor, torch.Tensor]): Tuple containing the right-hand side input tensor (B) and its scale factor tensor (SFB).
            - B should be in (n, k, l) order, but physically (l, n, k). For fp4 tensor with 8-bit storage, we expect the shape to be (n, k/2, l).
            - SFB should be in (n32, n4, rn, k4, rk, l) order, but physically (l, rn, rk, n32, n4, k4)
        out (torch.Tensor): Output tensor to store the result, with shape (l, m, n).
        masked_m (torch.Tensor): 1D tensor of shape (l,) specifying the valid row count for each batch (used for masking).
        ab_dtype (str): Data type for A and B matrices. Supported: "float4_e2m1fn", "float8_e4m3fn", "float8_e5m2".
        sf_dtype (str): Data type for scale factors. Supported: "float8_e8m0fnu", "float8_e4m3fn".
        c_dtype (str): Data type for output matrix C. Supported: "float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2".
        sf_vec_size (int): Vector size for scale factors. Typically 16 or 32.
        sm_count (int, optional): Number of SMs to use. Default: max available SMs under the CTA configuration.
        mma_tiler_mn (Tuple[int, int], optional): Shape of the MMA tiler (M, N). Default: (128, 128).
        cluster_shape_mn (Tuple[int, int], optional): Shape of the CTA cluster (ClusterM, ClusterN). Default: (1, 1).
        alpha_dtype (str, optional): Data type for alpha scaling factors.
        alpha (torch.Tensor, optional): Optional 1D tensor of shape (l,) containing per-batch scaling factors. Perform per-batch scaling out = alpha * out.

    Notes:
        - 维度约定: L=batch/experts, M=tokens, N=output_features, K=input_features.
          FP4 时 tensor shape 中的 k 是 K/2 (packed).
        - SFA/SFB 6D 布局: (m32=32, m4=4, rm=ceil(M/128), k4=4, rk=ceil(K/64), L).
        - masked tile scheduler 跳过 padding 行, 使稀疏 expert 负载高效.
        - kernel 按 (M, N, K, L, dtypes, tiler) 缓存, 首次调用触发 JIT 编译.
    """

    a_torch, sfa_torch = lhs
    b_torch, sfb_torch = rhs
    c_torch = out

    m, k, l = a_torch.shape
    n, _, _ = b_torch.shape

    if ab_dtype == "float4_e2m1fn":
        # todo(yingyi): update mnk based on a_major and b_major, and support more major.
        # Note: only support deepgemm-like shape for now
        k = k * 2

    mma_tiler_mn = kwargs.pop("mma_tiler_mn", (128, 128))
    cluster_shape_mn = kwargs.pop("cluster_shape_mn", (1, 1))
    if sm_count is None:
        sm_count = get_num_sm(a_torch.device)

    alpha = kwargs.pop("alpha", None)
    alpha_dtype = kwargs.pop("alpha_dtype", None)

    assert len(kwargs) == 0, f"Unsupported kwargs: {kwargs}"

    major, minor = get_compute_capability(a_torch.device)
    if major == 11 and minor == 0:
        raise ValueError("SM110 is not supported for cute-dsl backend.")

    # JIT 编译 (或获取缓存的) CuTeDSL kernel, 然后调用.
    # get_cute_dsl_compiled_masked_gemm_kernel 返回一个 callable (tensor_api),
    # 接受 GPU tensor 指针并启动 kernel.
    # kernel 按完整参数元组缓存; 首次调用触发编译 (~数秒), 后续复用编译结果.
    # Gemm1: m=2048, n=4096, k=7168, l=64, Float4E2M1FN, Float8E4M3FN, BFloat16
    # Gemm2: m=2048, n=7168, k=2048, l=64, 相同 dtypes
    return get_cute_dsl_compiled_masked_gemm_kernel(
        m=m,  # M 维度 (最大 token 数, 如 2048)
        n=n,  # N 维度 (输出特征数, 如 4096 或 7168)
        k=k,  # K 维度 (输入特征数, 逻辑值, 如 7168 或 2048)
        l=l,  # L 维度 (expert 数, 如 64)
        a_major="k",  # A 为 K-major (K 维度行主序)
        b_major="k",  # B 为 K-major (权重存储为 [N, K])
        c_major="n",  # C 为 N-major (输出存储为 [M, N])
        ab_dtype=get_cutlass_dtype(ab_dtype),  # Float4E2M1FN
        sf_dtype=get_cutlass_dtype(sf_dtype),  # Float8E4M3FN
        c_dtype=get_cutlass_dtype(c_dtype),  # BFloat16
        alpha_dtype=None if alpha is None else get_cutlass_dtype(alpha_dtype),
        sf_vec_size=sf_vec_size,  # 16 (每个 scale factor block 的元素数)
        mma_tiler_mn=mma_tiler_mn,  # (128, 128)
        cluster_shape_mn=cluster_shape_mn,  # (1, 1)
        sm_count=sm_count,  # 如 148
        sm_version=f"sm_{major}{minor}",  # "sm_100"
        enable_dst_signals=dst_signals is not None,
    )(
        a_tensor_gpu=a_torch,  # (M, K/2, L) uint8
        b_tensor_gpu=b_torch,  # (N, K/2, L) uint8
        sfa_tensor_gpu=sfa_torch,  # A 的 6D float8_e4m3fn scale factor
        sfb_tensor_gpu=sfb_torch,  # B 的 (L, N, sf_k) float8_e4m3fn scale factor
        c_tensor_gpu=c_torch,  # (M, N, L) bf16 输出
        masked_m_tensor_gpu=masked_m,  # (L,) int32 有效行数
        dst_signals_tensor_gpu=dst_signals,  # (L,) uint32 或 None
        alpha_tensor_gpu=alpha,  # (1, 1, L) float32 或 None
    )
