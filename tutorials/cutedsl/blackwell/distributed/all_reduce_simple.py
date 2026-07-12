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
import importlib
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
import cutlass.cute as cute
import cutlass.cute.testing as testing
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
使用 CuTe DSL 和 PyTorch Symmetric Memory 的分布式 All-Reduce 加法示例.

这个示例 kernel 演示如何结合 CuTe DSL 的 SIMT copy 和 PyTorch symmetric memory,
在多 GPU 上执行分布式 all-reduce. 基础 CuTe layout 计算来自 elementwise_add.py 示例.

这是一个简化版 all-reduce kernel. 它会直接把 remote memory 中的数据 copy 到 registers,
然后累加数据, 最后把累加结果写回 local global memory.
如果每个 device 上的 input tensors 都可以被远端访问, 就可以用此 kernel 执行 all-reduce.

在 host 侧, 可以用 `torch.distributed._symmetric_memory` 管理 symmetric memory.
通过 `symm_mem.empty` 和 `symm_mem.rendezvous` 创建 symmetric tensor,
再用 `get_buffer` 获取所有 devices 都可访问的 tensors.
这样可以隐藏启用 remote memory access 所需的 CUDA driver API 调用细节.

.. code-block:: python

    t = symm_mem.empty((M, N), device=torch.device(f"cuda:{rank}"))
    hdl = symm_mem.rendezvous(t, dist.group.WORLD)
    # 从 symmetric memory 获取其它 devices 上的 tensors
    tensor_list = [hdl.get_buffer(rank, t.shape, t.dtype) for rank in range(world_size)]

运行示例:

.. code-block:: bash

    torchrun --nproc-per-node 8 distributed/all_reduce_simple.py --M 1024 --N 512
    torchrun --nproc-per-node 8 distributed/all_reduce_simple.py \
        --M 1024 --N 1024 --benchmark --warmup_iterations 2 --iterations 100
"""


@cute.kernel
def all_reduce_simple_kernel(
    inputs: list[cute.Tensor],
    gOut: cute.Tensor,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # 为 CTAs 切分 tile
    # logical id -> address
    blk_coord = ((None, None), bidx)
    local_tile_out = gOut[blk_coord]
    local_tile_list = [t[blk_coord] for t in inputs]

    assert all(t.element_type == inputs[0].element_type for t in inputs)

    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        inputs[0].element_type,
    )
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        inputs[0].element_type,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    thr_tensor_list = [thr_copy.partition_S(tensor) for tensor in local_tile_list]
    thr_out = thr_copy.partition_D(local_tile_out)
    frg_tensor_list = [cute.make_fragment_like(tensor) for tensor in thr_tensor_list]
    frg_acc = cute.make_fragment_like(thr_out)
    frg_acc.fill(0.0)

    # 从所有 devices 的相同 offset 加载 frg, 并累加到 frg_acc
    # 即 frg_acc = 0.0 + rank0_frg + rank1_frg + ...
    for thr, frg in zip(thr_tensor_list, frg_tensor_list):
        cute.copy(copy_atom_load, thr, frg)
        tmp = frg.load() + frg_acc.load()
        frg_acc.store(tmp)

    # 从 register memory copy 到 global memory
    cute.copy(copy_atom_store, frg_acc, thr_out)


@cute.jit
def all_reduce_simple(
    inputs: list[cute.Tensor], output: cute.Tensor, copy_bits: cutlass.Constexpr = 128
):
    dtype = inputs[0].element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    divided_inputs = [cute.zipped_divide(tensor, tiler_mn) for tensor in inputs]
    gOut = cute.zipped_divide(output, tiler_mn)  # ((Tile),(Rest))
    all_reduce_simple_kernel(
        divided_inputs,
        gOut,
        thr_layout,
        val_layout,
    ).launch(
        grid=[cute.size(gOut, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def run_all_reduce_simple(
    M,
    N,
    warmup_iterations=2,
    iterations=10,
    skip_ref_check=False,
    benchmark=True,
):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        print("\nRunning Elementwise Add test with:")
        print(f"Tensor dimensions: [{M}, {N}]")
        print(f"GPU count: {world_size}")

    local_tensor = nvshmem.core.tensor((M, N), dtype=torch.float32)
    local_tensor.random_(0, 100)
    # peer tensors 是 remote views, 不会在当前 GPU 上复制远端 tensor 数据.
    tensor_list = [nvshmem.core.get_peer_tensor(local_tensor, rank) for rank in range(world_size)]
    output = torch.zeros((M, N), device=f"cuda:{rank}")

    if rank == 0:
        print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    compiled_func = cute.compile(all_reduce_simple, [from_dlpack(t) for t in tensor_list], from_dlpack(output))
    compilation_time = time.time() - start_time
    if rank == 0:
        print(f"Compilation time: {compilation_time:.4f} seconds")
        print("Executing vector add kernel...")

    if not skip_ref_check:
        dist.barrier(device_ids=[rank])
        compiled_func([from_dlpack(t) for t in tensor_list], from_dlpack(output))
        if rank == 0:
            print("Verifying results...")
        dist.barrier(device_ids=[rank])
        torch.testing.assert_close(sum([t.cpu() for t in tensor_list]), output.cpu())
        if rank == 0:
            print("Results verified successfully!")
    
    for t in tensor_list:
        nvshmem.core.free_tensor(t)

    if not benchmark:
        return

    free_func_and_tensor_pairs = []
    def add_free_func_and_tensor(free_func, tensor):
        free_func_and_tensor_pairs.append((free_func, tensor))

    def generate_tensors():
        local_tensor = nvshmem.core.tensor((M, N), dtype=torch.float32)
        local_tensor.random_(0, 100)
        tensor_list = [nvshmem.core.get_peer_tensor(local_tensor, rank) for rank in range(world_size)]
        output = torch.zeros((M, N), device=f"cuda:{rank}")

        ja = testing.JitArguments(
            [from_dlpack(t) for t in tensor_list],
            from_dlpack(output),
        )
        for tensor in tensor_list:
            add_free_func_and_tensor(nvshmem.core.free_tensor, tensor)
        return ja
   
    avg_time_us = testing.benchmark(
        compiled_func,
        workspace_generator=generate_tensors,
        workspace_count=10,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    # 打印执行结果
    if rank == 0:
        print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")
        print(
            f"Achieved memory throughput: {((world_size + 1) * output.numel() * 32 // 8) / (avg_time_us / 1e6) / 1e9:.2f} GB/s"
        )
        print(f"First few elements of result: \n{output[:3, :3]}")

    for free_func, tensor in free_func_and_tensor_pairs:
        free_func(tensor)


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
    if dist.get_rank() == 0:
        import pdb; pdb.set_trace()

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

    run_all_reduce_simple(args.M, args.N, args.warmup_iterations, args.iterations, args.skip_ref_check, args.benchmark)

    torchrun_finalize()

    return


if __name__ == "__main__":
    main()
