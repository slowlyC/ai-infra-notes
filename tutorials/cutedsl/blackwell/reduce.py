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

"""
CuTe-DSL Kernel 的层级归约工具
==============================

本模块提供可复用的 GPU 归约原语, 支持跨 warp、block 和 cluster (SM90+) 的归约.

概述
----
GPU 归约通常遵循层级模式:

1. **Warp 归约**: warp 内的线程通过 shuffle 指令归约.
   使用 CuTe-DSL 库的 `cute.arch.warp_reduction()`.

2. **Block 归约**: block 内的多个 warp 通过共享内存归约.
   使用本模块的 `block_reduce()`.

3. **Cluster 归约** (SM90+): cluster 内的多个 CTA 通过分布式共享内存
   和 mbarrier 同步进行归约.
   使用本模块的 `cluster_reduce()`.

4. **行归约**: 根据问题配置编排所有归约层级.
   使用本模块的 `row_reduce()`.

共享内存缓冲区 Layout 约定
---------------------------

`block_reduce` 的缓冲区:
    - 形状: (rows_per_block, warps_per_row)
    - 每个 warp 的 lane 0 将归约值写入 buffer[row_idx, col_idx]
    - 线程映射: row_idx = warp_idx // warps_per_row
                col_idx = warp_idx % warps_per_row

    示例 (8 个 warp, 2 行, 每行 4 个 warp):
        Warp 0 -> buffer[0, 0]    Warp 4 -> buffer[1, 0]
        Warp 1 -> buffer[0, 1]    Warp 5 -> buffer[1, 1]
        Warp 2 -> buffer[0, 2]    Warp 6 -> buffer[1, 2]
        Warp 3 -> buffer[0, 3]    Warp 7 -> buffer[1, 3]

`cluster_reduce` 的缓冲区:
    - 形状: (rows_per_block, (warps_per_row, cluster_n))
    - 第二维是层级式的: (本地 warp 槽位, CTA 编号)
    - 每个 CTA 写入 cluster 维度中自己的槽位

    示例 (cluster_n=4, 每行 2 个 warp):
        CTA 0, Warp 0 -> buffer[row, (0, 0)]
        CTA 0, Warp 1 -> buffer[row, (1, 0)]
        CTA 1, Warp 0 -> buffer[row, (0, 1)]
        CTA 1, Warp 1 -> buffer[row, (1, 1)]
        ... CTA 2, 3 同理

Mbarrier 要求 (Cluster 归约)
-----------------------------
调用 cluster 归约时, 调用方需要:
1. 在共享内存中分配 mbarrier
2. 用 `cute.arch.mbarrier_init(mbar_ptr, thread_count)` 初始化
3. 将 mbarrier 指针传给 `cluster_reduce()`

cluster_reduce 函数内部处理:
- 设置预期事务字节数
- 执行异步跨 CTA 写入
- 等待所有写入完成

使用示例
--------

.. code-block:: python

    from reduce import row_reduce, block_reduce, cluster_reduce

    @cute.jit
    def my_kernel(...):
        # 为归约分配共享内存, 形状取决于 warps_per_row 和 cluster_n
        if cluster_n > 1:
            reduction_buffer = cute.make_smem_tensor(
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                Float32
            )
        else:
            reduction_buffer = cute.make_smem_tensor(
                cute.make_layout((rows_per_block, warps_per_row)),
                Float32
            )

        # 执行行归约
        result = row_reduce(
            tensor_ssa,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            mbar_ptr,
            cluster_n,
            init_val=Float32(0.0)
        )

参考
----
cluster 同步原语 (set_block_rank, store_shared_remote)
参考了 Quack: https://github.com/Dao-AILab/quack
"""

import operator
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


# =============================================================================
# 用于 Cluster 通信的内联 PTX 操作
# =============================================================================
#
# 这些操作实现 cluster 内跨 CTA 通信 (SM90+).
# 使用内联 PTX 汇编, 因为这些功能尚未在 MLIR 中暴露.
# =============================================================================


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """
    将本地共享内存指针映射到同一 cluster 中另一个 CTA 的等效地址.

    使用 PTX `mapa.shared::cluster` 指令, 将本地共享内存地址
    翻译为对端 CTA 共享内存空间中的对应地址.

    Args:
        smem_ptr: 本地共享内存指针
        peer_cta_rank_in_cluster: 目标 CTA 在 cluster 中的编号 (0 到 cluster_size-1)

    Returns:
        Int32, 表示对端 CTA 共享内存中的映射地址

    Note:
        需要 SM90+ 且启用了 cluster 支持.
        kernel 必须以适当的 cluster 维度启动.
    """
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    异步地将 Float32 值写入 cluster 中远程 CTA 的共享内存,
    并通过 mbarrier 跟踪完成状态.

    使用 PTX `st.async.shared::cluster` 指令, 该指令会:
    1. 将本地 smem 地址翻译到对端 CTA 的地址空间
    2. 异步写入远程共享内存
    3. 写入完成后通知 mbarrier

    Args:
        val: 要写入的 Float32 值
        smem_ptr: 目标地址 (本地共享内存坐标)
        mbar_ptr: 跟踪完成状态的 mbarrier 指针
        peer_cta_rank_in_cluster: 目标 CTA 在 cluster 中的编号

    Note:
        - mbarrier 必须已设置预期事务字节数
        - 使用 `cute.arch.mbarrier_arrive_and_expect_tx()` 设置事务
        - 使用 `cute.arch.mbarrier_wait()` 等待所有写入完成
        - 需要 SM90+ 且启用了 cluster 支持
    """
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """
    获取 tensor 中指定坐标处元素的指针.

    在 cluster 归约中执行跨 CTA 写入时, 需要获取共享内存中
    特定元素的地址, 此函数用于该场景.

    Args:
        x: tensor (通常是共享内存 tensor)
        coord: 坐标元组, 可以是层级式的, 如 (row, (col, cluster_idx))

    Returns:
        指向指定坐标元素的指针
    """
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


# =============================================================================
# Block 级归约
# =============================================================================


@cute.jit
def block_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """
    通过共享内存在 block 内所有 warp 间归约.

    前提: 每个 warp 已完成 warp 级归约, 贡献一个值 (来自 lane 0).
    本函数的步骤:
    1. 每个 warp 的值写入共享内存
    2. block 同步
    3. 对收集到的值做最终 warp 归约

    Args:
        val: warp 归约后的值 (仅 lane 0 的值有效)
        op: 二元归约算子, 如 `operator.add` 或 `cute.arch.fmax`
        reduction_buffer: 共享内存 tensor, 形状 (rows_per_block, warps_per_row)
        init_val: 归约的单位元 (求和用 0, 求最大值用 -inf)

    Returns:
        block 级归约结果 (所有线程得到相同的值)

    缓冲区 Layout:
        - 形状: (rows_per_block, warps_per_row)
        - warps_per_row 从 reduction_buffer.shape[1] 推断
        - 线程映射:
            row_idx = warp_idx // warps_per_row
            col_idx = warp_idx % warps_per_row

    示例:
        8 个 warp 处理 2 行 (每行 4 个 warp):

        .. code-block:: python

            reduction_buffer = cute.make_smem_tensor(
                cute.make_layout((2, 4)),  # 2 行, 每行 4 个 warp
                Float32
            )
            result = block_reduce(warp_val, operator.add, reduction_buffer, Float32(0.0))
    """
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    # 每个 warp 的 lane 0 将值写入共享内存
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    # 所有 lane 参与读取和归约, 但只有 lane < warps_per_row 的数据有效
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(block_reduce_val, op)


# =============================================================================
# Cluster 级归约 (SM90+)
# =============================================================================


@cute.jit
def cluster_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
) -> Float32:
    """
    通过分布式共享内存在 cluster 内所有 CTA 间归约.

    将 block 归约扩展到 cluster 内多个 CTA, 使用异步跨 CTA 写入
    和 mbarrier 同步. 步骤:
    1. 设置 mbarrier 的预期事务字节数
    2. 每个 warp 的值异步写入所有对端 CTA
    3. 等待所有写入完成
    4. 对收集到的所有值做归约

    Args:
        val: warp 归约后的值 (仅 lane 0 的值用于写入)
        op: 二元归约算子, 如 `operator.add` 或 `cute.arch.fmax`
        reduction_buffer: 共享内存 tensor, 层级式形状
                          (rows_per_block, (warps_per_row, cluster_n))
        mbar_ptr: 已初始化的 mbarrier 的共享内存指针
        cluster_n: cluster 中的 CTA 数量 (编译期常量)
        init_val: 归约的单位元 (求和用 0, 求最大值用 -inf)

    Returns:
        cluster 级归约结果 (所有 CTA 的所有线程得到相同的值)

    缓冲区 Layout:
        - 形状: (rows_per_block, (warps_per_row, cluster_n))
        - 第二维是层级式的:
          - 第一层: warps_per_row (本地 warp 槽位)
          - 第二层: cluster_n (每个 CTA 一个槽位)
        - 访问模式: buffer[row_idx, (col_idx, cta_rank)]

    前置条件:
        - SM90+ 且支持 cluster
        - 调用前 mbarrier 必须已初始化
        - kernel 必须以适当的 cluster 维度启动

    示例:
        4 个 CTA 的 cluster, 每个有 2 个 warp 处理一行:

        .. code-block:: python

            # 分配带 cluster 维度的缓冲区
            reduction_buffer = cute.make_smem_tensor(
                cute.make_layout((rows_per_block, (2, 4))),  # 2 warp, 4 CTA
                Float32
            )

            # 初始化 mbarrier (每个 kernel 一次)
            mbar = cute.make_smem_tensor(cute.make_layout((1,)), cute.arch.Mbarrier)
            cute.arch.mbarrier_init(mbar.iterator, thread_count)

            # 执行 cluster 归约
            result = cluster_reduce(
                warp_val, operator.add, reduction_buffer,
                mbar.iterator, cluster_n=4, init_val=Float32(0.0)
            )
    """
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    rows_per_block = reduction_buffer.shape[0]
    warps_per_row = reduction_buffer.shape[1][0]

    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    # Warp 0 的 lane 0 设置 mbarrier 的预期事务字节数
    # 每个 warp 发送 cluster_n 次写入 (每个 CTA 一次), 每次 4 字节
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            expected_bytes = num_warps * cluster_n * 4  # 4 bytes per Float32
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expected_bytes)

    # lane < cluster_n 的线程各写入不同 CTA 的共享内存
    # 将本 warp 的值分发到 cluster 中所有 CTA
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )

    # 等待所有跨 CTA 写入完成
    cute.arch.mbarrier_wait(mbar_ptr, phase=0)

    # 此时每个 CTA 都拥有 cluster 中所有 CTA 的值, 做本地归约
    num_total = warps_per_row * cluster_n
    num_iter = cute.ceil_div(num_total, 32)

    block_reduce_val = init_val
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * 32
        if idx < num_total:
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])

    return cute.arch.warp_reduction(block_reduce_val, op)


# =============================================================================
# 行归约 (编排函数)
# =============================================================================


@cute.jit
def row_reduce(
    x: cute.TensorSSA,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
):
    """
    层级行归约, 自动选择归约策略.

    编排完整的归约流水线:
    1. 本地归约: 每个线程归约自己负责的部分
    2. Warp 归约: warp 内线程通过 shuffle 归约
    3. Block 归约: 如需要, 多个 warp 通过共享内存归约
    4. Cluster 归约: 如需要, 多个 CTA 通过分布式共享内存归约

    根据 `threads_per_row` 和 `cluster_n` 自动选择合适的归约层级.

    Args:
        x: 包含待归约值的 TensorSSA (在寄存器中)
        op: 归约操作 (cute.ReductionOp.ADD 或 cute.ReductionOp.MAX)
        threads_per_row: 协作处理每行的线程数 (编译期)
        reduction_buffer: 用于 block/cluster 归约的共享内存 tensor
        mbar_ptr: mbarrier 指针 (仅 cluster_n > 1 时使用)
        cluster_n: cluster 中的 CTA 数量 (1 表示单 CTA 归约)
        init_val: 归约的单位元

    Returns:
        每行的完整归约结果

    归约策略选择:
        - threads_per_row <= 32, cluster_n == 1: 仅 warp 归约
        - threads_per_row > 32, cluster_n == 1: warp + block 归约
        - cluster_n > 1: warp + cluster 归约 (处理所有情况)

    示例:
        .. code-block:: python

            # 128 线程/行的求和归约, 单 CTA
            result = row_reduce(
                tensor_ssa,
                cute.ReductionOp.ADD,
                threads_per_row=128,
                reduction_buffer=smem_buffer,
                mbar_ptr=None,
                cluster_n=1,
                init_val=Float32(0.0)
            )

            # 256 线程/行的求最大值归约, 4 CTA cluster
            result = row_reduce(
                tensor_ssa,
                cute.ReductionOp.MAX,
                threads_per_row=256,
                reduction_buffer=smem_buffer,
                mbar_ptr=mbar.iterator,
                cluster_n=4,
                init_val=Float32.neg_inf
            )
    """
    # 第 1 步: 本地归约 - 每个线程归约自己寄存器中的值
    local_val = x.reduce(op, init_val=init_val, reduction_profile=0)

    # 将 ReductionOp 枚举映射为 warp/block 归约用的二元算子
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax,
    }[op]

    # 第 2 步: Warp 归约
    # 若 threads_per_row < 32, 只使用对应数量的线程参与归约
    warp_width = min(threads_per_row, 32)
    warp_val = cute.arch.warp_reduction(local_val, warp_op, threads_in_group=warp_width)

    # 判断是否需要更高层级的归约
    warps_per_row = max(threads_per_row // 32, 1)

    # 第 3 & 4 步: Block 或 cluster 归约 (如需要)
    if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
        if cutlass.const_expr(cluster_n == 1):
            # 单 CTA: 使用 block 归约
            return block_reduce(warp_val, warp_op, reduction_buffer, init_val)
        else:
            # 多 CTA: 使用 cluster 归约
            return cluster_reduce(
                warp_val, warp_op, reduction_buffer, mbar_ptr, cluster_n, init_val
            )
    else:
        # 单个 warp 处理整行: warp 归约即可
        return warp_val

