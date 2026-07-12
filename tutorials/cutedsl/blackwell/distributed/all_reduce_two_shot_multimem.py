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


import os
import time
import argparse

import numpy as np
import torch
import torch.distributed as dist
try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device
from cuda.pathfinder import load_nvidia_dynamic_lib

import cutlass
import cutlass.utils as utils
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

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
使用 CuTe DSL 和 PyTorch Symmetric Memory 的分布式 Two-Shot All-Reduce 示例.

这个示例 kernel 演示如何利用 multimem 功能实现 two-shot all-reduce.
multimem instruction 操作 symmetric memory, 可以把 broadcast 和 reduce 卸载到 NVLink Switch,
从而减少 NVLink traffic.

调用 `multimem.ld_reduce addrA` 时, 各 device 上对应的数据会被发送到 NVLS, 并将归约结果返回.
对于 `multimem.st dataA addrA`, 数据只需发送到 NVLS 一次, 再由 NVLS broadcast 到各 device.
因此, multimem 可以将 memory traffic 和 instruction count 降低 8 倍.

本示例使用 two-shot 风格的 all-reduce, 即每个 device 计算一部分数据, 再把结果写到所有 devices.
与 one-shot all-reduce 相比, two-shot 可以获得更高的吞吐性能.
input 和 output 都是 symmetric memory, 因此这里不需要额外的 communication buffers.
使用 `sm_wise_inter_gpu_multimem_barrier` 同步各 device 间的数据,
以确保每个 device 都已完成 data transfer.

运行示例:

.. code-block:: bash

    torchrun --nproc-per-node 8  examples/distributed/all_reduce_two_shot_multimem.py --M 1024 --N 512
    torchrun --nproc-per-node 8  examples/distributed/all_reduce_two_shot_multimem.py \
        --M 1024 --N 1024 --benchmark --warmup_iterations 2 --iterations 100
"""

# 每个 tile 只由负责它的 rank 归约一次, broadcast 后所有 ranks 都得到完整 output.
#
# 数据流示例, world_size=2, total_tiles=4, chunk_size=2:
#
# tile 负责关系:
#   rank 0 CTAs -> tile 0, tile 1
#   rank 1 CTAs -> tile 2, tile 3
#
# rank 0 input: [A0 A1 | A2 A3]
# rank 1 input: [B0 B1 | B2 B3]
#                  |      |
#                  |      \-> rank 1 CTAs: (A2,B2)->S2, (A3,B3)->S3
#                  \--------> rank 0 CTAs: (A0,B0)->S0, (A1,B1)->S1
#
# rank 0 CTAs: [S0 S1] --multimem.st--> rank 0 output: [S0 S1 | -- --]
#                                    \-> rank 1 output: [S0 S1 | -- --]
# rank 1 CTAs: [S2 S3] --multimem.st--> rank 0 output: [-- -- | S2 S3]
#                                    \-> rank 1 output: [-- -- | S2 S3]
#
# rank 0 output: [S0 S1 | S2 S3]
# rank 1 output: [S0 S1 | S2 S3]
#
@cute.kernel
def all_reduce_multimem_kernel(
    gIn: cute.Tensor,
    gOut: cute.Tensor,
    flag: cute.Tensor,
    flag_mc: cute.Tensor,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    local_rank: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # 为 CTAs 切分 tile
    # logical id -> address
    num_ctas = cute.size(gIn, mode=[1])
    chunk_size = num_ctas // world_size
    blk_idx = local_rank * chunk_size + bidx

    blk_coord = ((None, None), blk_idx)
    local_tile_out = gOut[blk_coord]
    local_tile_in = gIn[blk_coord]

    assert gIn.element_type == gOut.element_type

    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gIn.element_type,
        num_bits_per_copy=128,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    thr_in = thr_copy.partition_S(local_tile_in)
    thr_out = thr_copy.partition_D(local_tile_out)

    (_, rest_m), _, _ = thr_in.shape
    (_, rest_m_stride), _, _ = thr_in.stride

    for i in cutlass.range_constexpr(rest_m):
        # Shot 1 (reduce): 从所有 ranks 的 multicast input 归约对应的 4 个 float32.
        x, y, z, w = utils.distributed.multimem_ld_reduce_4xf32(
            thr_in[(None, i), 0, 0].iterator
        )
        # Shot 2 (broadcast): 将归约结果写入所有 ranks 的 multicast output.
        utils.distributed.multimem_st_4xb32(
            thr_out[(None, i), 0, 0].iterator, x, y, z, w
        )

    # 确保 CTA 中所有 threads 都已发出 multimem.ld_reduce 和 multimem.st instructions
    cute.arch.sync_threads()

    if tidx == 0:
        # 当前 CTA 的 linear id.
        sm_id_linear = (
            cute.arch.block_idx()[0]
            + cute.arch.block_idx()[1] * cute.arch.grid_dim()[0]
            + cute.arch.block_idx()[2]
            * cute.arch.grid_dim()[0]
            * cute.arch.grid_dim()[1]
        )
        # 向所有 Ranks 发送到达通知,
        # 该操作的语义是对所有 ranks 的 local_flag[sm_id_linear] 加 1
        utils.distributed.multimem_red_add1(
            flag_mc.iterator + sm_id_linear,
            scope="sys",
            order="release",
        )
        # 以 sys scope 使用 relaxed spin-lock 等待 flag
        # 每个 rank 轮询自己的 local_flag[sm_id_linear], 直到 flag == world_size
        utils.distributed.spin_lock_atom_cas_relaxed_wait(
            flag.iterator + sm_id_linear,
            expected_val=world_size,
            reset_val=0,
            scope="sys",
        )

@cute.jit
def all_reduce_multimem(
    mIn: cute.Tensor,
    mOut: cute.Tensor,
    flag: cute.Tensor,
    flag_mc: cute.Tensor,
    local_rank: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    copy_bits: cutlass.Constexpr = 128,
):
    dtype = mIn.element_type
    vector_size = copy_bits // dtype.width

    # 为每个 CTA 选择 128x128 tile
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((32, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gIn = cute.zipped_divide(mIn, tiler_mn)
    gOut = cute.zipped_divide(mOut, tiler_mn)

    all_reduce_multimem_kernel(
        gIn,
        gOut,
        flag,
        flag_mc,
        thr_layout,
        val_layout,
        local_rank,
        world_size,
    ).launch(
        grid=[cute.size(gOut, mode=[1]) // world_size, 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def run_all_reduce_multimem(
    M,
    N,
    warmup_iterations=2,
    iterations=10,
    skip_ref_check=False,
    benchmark=True,
):
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    tile_m = 128
    tile_n = 128

    if local_rank == 0:
        print("\nRunning Elementwise Add test with:")
        print(f"Tensor dimensions: [{M}, {N}]")
        print(f"GPU count: {world_size}")

    local_input_tensor = nvshmem.core.tensor((M, N), dtype=torch.float32)
    input_tensor = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, local_input_tensor)

    local_output_tensor = nvshmem.core.tensor((M, N), dtype=torch.float32)
    output_tensor = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, local_output_tensor)

    local_flag = nvshmem.core.tensor((M*N//(tile_m*tile_n)), dtype=torch.int32)
    flag_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, local_flag)
    
    if local_rank == 0:
        print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    compiled_func = cute.compile(
        all_reduce_multimem,
        from_dlpack(input_tensor),
        from_dlpack(output_tensor),
        from_dlpack(local_flag),
        from_dlpack(flag_mc),
        local_rank,
        world_size,
    )
    compilation_time = time.time() - start_time
    if local_rank == 0:
        print(f"Compilation time: {compilation_time:.4f} seconds")
        print("Executing all-reduce two shot multimem kernel...")

    if not skip_ref_check:
        dist.barrier(device_ids=[local_rank])
        compiled_func(
            from_dlpack(input_tensor),
            from_dlpack(output_tensor),
            from_dlpack(local_flag),
            from_dlpack(flag_mc),
        )
        dist.barrier(device_ids=[local_rank])
        if local_rank == 0:
            print("Verifying results...")

        local_buffers = [nvshmem.core.get_peer_tensor(local_input_tensor, local_rank) for local_rank in range(world_size)]
        torch.testing.assert_close(sum([buffer.cpu() for buffer in local_buffers]), local_output_tensor.cpu())
        if local_rank == 0:
            print("Results verified successfully!")
        for i in range(world_size):
            if i != local_rank:
                nvshmem.core.free_tensor(local_buffers[i])

    # 始终先释放 multicast tensors
    nvshmem.core.free_tensor(input_tensor)
    nvshmem.core.free_tensor(output_tensor)
    nvshmem.core.free_tensor(flag_mc)
    nvshmem.core.free_tensor(local_input_tensor)
    nvshmem.core.free_tensor(local_output_tensor)
    nvshmem.core.free_tensor(local_flag)

    if not benchmark:
        return
    
    free_func_and_tensor_pairs = []
    def add_free_func_and_tensor(free_func, tensor):
        free_func_and_tensor_pairs.append((free_func, tensor))

    def generate_tensors():
        local_input_tensor = nvshmem.core.tensor((M, N), dtype=torch.float32)
        input_tensor_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, local_input_tensor)

        local_output_tensor = nvshmem.core.tensor((M, N), dtype=torch.float32)
        output_tensor_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, local_output_tensor)

        local_flag = nvshmem.core.tensor((M*N//(tile_m*tile_n)), dtype=torch.int32)
        flag_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, local_flag)

        ja = testing.JitArguments(
            from_dlpack(input_tensor_mc),
            from_dlpack(output_tensor_mc),
            from_dlpack(local_flag),
            from_dlpack(flag_mc),
        )
        tensors_to_free = [input_tensor_mc, output_tensor_mc, flag_mc, local_input_tensor, local_output_tensor, local_flag]
        for tensor in tensors_to_free:
            add_free_func_and_tensor(nvshmem.core.free_tensor, tensor)
        return ja

    dist.barrier(device_ids=[local_rank])
    avg_time_us = testing.benchmark(
        compiled_func,
        workspace_generator=generate_tensors,
        workspace_count=10,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    dist.barrier(device_ids=[local_rank])
    torch.cuda.synchronize()

    # 打印执行结果
    if local_rank == 0:
        print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")
        print(
            f"Achieved memory throughput: {((world_size + 1) * output_tensor.numel() * 32 // 8) / (avg_time_us / 1e6) / 1e9:.2f} GB/s"
        )
    
    for free_func, tensor in free_func_and_tensor_pairs:
        free_func(tensor)
    return


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


def main():
    parser = argparse.ArgumentParser(
        description="elementwise add 示例, 用于演示 kernel 如何接收 numpy/pytorch 输入"
    )
    parser.add_argument("--M", default=1024, type=int)
    parser.add_argument("--N", default=1024, type=int)
    parser.add_argument("--warmup_iterations", default=2, type=int)
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()

    torchrun_uid_init_bcast()

    run_all_reduce_multimem(args.M, args.N, args.warmup_iterations, args.iterations, args.skip_ref_check, args.benchmark)

    torchrun_finalize()

    return


if __name__ == "__main__":
    main()
