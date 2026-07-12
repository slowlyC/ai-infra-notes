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
import argparse
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import vector

"""
使用 CuTe DSL 和细粒度内存控制实现的分布式 One-Shot All-Reduce 示例.
这是现有 tensorrt_llm kernel 的镜像版本:
https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu

这个示例 kernel 演示如何用 CuTe DSL 和细粒度内存控制执行 one-shot all-reduce.
它使用专用 communication buffers 交换数据, 这些 buffers 作为 ping-pong buffers.
执行过程中, kernel 使用一个 buffer 做通信, 并把下一个 buffer 初始化为 negative zero.

在这个 kernel 中, 每个 thread 只负责 128bits 数据.
kernel 会把本地数据写到不同 ranks 的每个 buffer, 然后从 local rank buffer 读取数据.
buffer 本身起到 barrier 的作用, 如果 kernel 读到 negative zero, 表示数据还没 ready 或还不可见,
kernel 会重新读取该数据.

如果每个 device 上的 input tensors 不能被远端访问, 可以使用这个 kernel 执行 one-shot all-reduce,
因为它通过 communication buffers 交换数据.

这里使用 .SYS memory scope 和 .VOLATILE memory order, 确保数据在 system scope 可见.

.. code-block:: bash

    torchrun --nproc-per-node 8  examples/cute/blackwell/distributed/all_reduce_one_shot_lamport.py --M 8192 --N 8192
    torchrun --nproc-per-node 8  examples/cute/blackwell/distributed/all_reduce_one_shot_lamport.py \
        --M 8192 --N 8192 --benchmark --warmup_iterations 2 --iterations 10
"""


PING_PONG_SIZE = 3


class AllReduceOneShotLamportKernel:
    @cute.jit
    def __call__(
        self,
        rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        signal: cutlass.Int32,
        local_input: cute.Tensor,
        local_output: cute.Tensor,
        buffers: list[cute.Tensor],
        stream: cuda.CUstream,
    ):
        copy_bits = 128
        dtype = local_input.element_type  # 32
        vector_size = copy_bits // dtype.width  # 128 / 32 = 4

        thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
        val_layout = cute.make_ordered_layout((1, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        grouped_buffers = [cute.group_modes(buffer, 0, 2) for buffer in buffers]
        tiled_buffers = [
            cute.zipped_divide(buffer, (tiler_mn, world_size, PING_PONG_SIZE))
            for buffer in grouped_buffers
        ]
        tiled_input = cute.zipped_divide(local_input, tiler_mn)
        tiled_output = cute.zipped_divide(local_output, tiler_mn)

        self.kernel(
            tiled_buffers,
            tiled_input,
            tiled_output,
            thr_layout,
            val_layout,
            signal,
            rank,
        ).launch(
            grid=[cute.size(tiled_input, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
            stream=stream,
        )

    # 每个 CTA tile 的通信流程:
    #
    # signal -> ping, 本轮通信 slot
    #        -> pong, 下一轮清理 slot
    #
    # local buffer[dst=rank][src=all][pong] <- -0.0
    #
    # local_input -> frg_in(register)
    #                     |
    #                     v
    #     fan-out R2G 写入各 rank 的 symmetric buffers
    #                     |
    #                     +-> buffer[dst=0][src=rank][ping]
    #                     +-> ...
    #                     +-> buffer[dst=world_size-1][src=rank][ping]
    #
    # 当前 rank 遍历所有的 symmetric buffers, 并累加数据
    # for i in range_constexpr(len(buffers)):
    #     buffer[dst=rank][src=i][ping]
    #                      |
    #                      v
    #            轮询直到不再是 -0.0
    #                      |
    #                      v
    #           frg_acc += src payload
    #
    # frg_acc(register) -> local_output
    #
    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        buffers: list[cute.Tensor],
        local_input: cute.Tensor,
        local_output: cute.Tensor,
        thr_layout: cute.Layout,
        val_layout: cute.Layout,
        signal: cutlass.Int32,
        rank: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        ping = signal % 3
        pong = (signal + 1) % 3

        buffer_local = buffers[rank]
        cta_coord = ((None, None), bidx)
        local_tile_in = local_input[cta_coord]
        local_tile_out = local_output[cta_coord]

        ping_coord = (((None, None), None, ping), bidx)
        pong_coord = (((None, None), None, pong), bidx)

        read_buffer = buffer_local[ping_coord]
        clear_buffer = buffer_local[pong_coord]

        write_coord = (((None, None), rank, ping), bidx)
        write_buffers = [buffer[write_coord] for buffer in buffers]

        # 假设所有 buffers 和 input 有相同 element type
        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            buffers[0].element_type,
            num_bits_per_copy=128,
            memory_scope=cute.nvgpu.common.MemoryScope.SYS,
            memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            buffers[0].element_type,
            num_bits_per_copy=128,
            memory_scope=cute.nvgpu.common.MemoryScope.SYS,
            memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
        )
        tiled_copy = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)

        thr_write_buffer_list = [
            thr_copy.partition_D(tensor) for tensor in write_buffers
        ]
        thr_read_buffer = thr_copy.partition_S(read_buffer)

        thr_clear_buffer = thr_copy.partition_D(clear_buffer)

        thr_in = thr_copy.partition_S(local_tile_in)
        thr_out = thr_copy.partition_D(local_tile_out)

        frg_in = cute.make_fragment_like(thr_in)
        frg_clear = cute.make_fragment_like(thr_clear_buffer)
        frg_acc = cute.make_fragment_like(thr_out)
        frg_acc.fill(0.0)

        # 把下一个 buffer 清成 negative zero
        clear_tensor = frg_clear.load()
        frg_size = cute.size(clear_tensor.shape)
        neg0_i32_vec = cute.full_like(clear_tensor, 0x80000000, cutlass.Int32)
        neg0_f32_vec = vector.bitcast(T.vector(frg_size, T.f32()), neg0_i32_vec)
        neg0_f32_tensor = cute.TensorSSA(
            neg0_f32_vec, clear_tensor.shape, cutlass.Float32
        )
        frg_clear.store(neg0_f32_tensor)
        cute.copy(copy_atom_store, frg_clear, thr_clear_buffer)

        # 读取 local data 到 register
        cute.copy(copy_atom_load, thr_in, frg_in)

        # 把 local data 写到不同 ranks 的每个 symmetric buffer
        for thr_write_buffer in thr_write_buffer_list:
            cute.copy(copy_atom_store, frg_in, thr_write_buffer)

        frg_in_vector_neg0_i32 = cute.full_like(
            frg_in, cutlass.Int32(0x80000000), cutlass.Int32
        )
        frg_in_size = cute.size(frg_in.shape)

        # 遍历每个 buffer 并累加数据
        for i in cutlass.range_constexpr(len(buffers)):
            read_coord = (None, 0, 0, i)
            cute.copy(copy_atom_load, thr_read_buffer[read_coord], frg_in[None, 0, 0])
            frg_vector = frg_in.load()
            frg_vector_i32 = cute.TensorSSA(
                vector.bitcast(T.vector(frg_in_size, T.i32()), frg_vector),
                frg_in.shape,
                cutlass.Int32,
            )
            isNotNeg0 = cute.all_(
                cute.TensorSSA(
                    frg_vector_i32 != frg_in_vector_neg0_i32,
                    frg_in.shape,
                    cutlass.Boolean,
                )
            )
            # 如果数据是 negative zero, 表示数据还没 ready 或还不可见, 需要重新读取
            while not isNotNeg0:
                cute.copy(
                    copy_atom_load, thr_read_buffer[read_coord], frg_in[None, 0, 0]
                )
                frg_vector = frg_in.load()
                frg_vector_i32 = cute.TensorSSA(
                    vector.bitcast(T.vector(frg_in_size, T.i32()), frg_vector),
                    frg_in.shape,
                    cutlass.Int32,
                )
                isNotNeg0 = cute.all_(
                    cute.TensorSSA(
                        frg_vector_i32 != frg_in_vector_neg0_i32,
                        frg_in.shape,
                        cutlass.Boolean,
                    )
                )
            frg_acc.store(frg_in.load() + frg_acc.load())

        cute.copy(copy_atom_store, frg_acc, thr_out)


def run_all_reduce_one_shot(
    M,
    N,
    warmup_iterations=2,
    iterations=10,
    skip_ref_check=False,
    benchmark=True,
):
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        print("\nRunning Elementwise Add test with:")
        print(f"Tensor dimensions: [{M}, {N}]")
        print(f"GPU count: {world_size}")

    # 每个 rank 额外显存开销为 PING_PONG_SIZE * world_size * M * N * dtype_size.
    t = symm_mem.empty(
        [
            PING_PONG_SIZE,
            world_size,
            M,
            N,
        ],
        device="cuda",
    ).neg_()
    hdl = symm_mem.rendezvous(t, dist.group.WORLD)
    buffer_tensor_list = [
        hdl.get_buffer(rank, t.shape, t.dtype).permute(2, 3, 1, 0)
        for rank in range(world_size)
    ]  # buffer[M, N, src_rank, slot]
    signal = cutlass.Int32(0)
    input_tensor = torch.randn([M, N], device=f"cuda:{rank}")
    output_tensor = torch.zeros([M, N], device=f"cuda:{rank}")
    stream = cutlass.cuda.default_stream()
    all_reduce_one_shot_lamport_kernel = AllReduceOneShotLamportKernel()

    compiled_func = cute.compile(
        all_reduce_one_shot_lamport_kernel,
        rank,
        world_size,
        signal,
        from_dlpack(input_tensor, assumed_align=32),
        from_dlpack(output_tensor, assumed_align=32),
        [from_dlpack(t, assumed_align=32) for t in buffer_tensor_list],
        stream=stream,
    )

    if not skip_ref_check:
        compiled_func(
            signal,
            from_dlpack(input_tensor, assumed_align=32),
            from_dlpack(output_tensor, assumed_align=32),
            [from_dlpack(t, assumed_align=32) for t in buffer_tensor_list],
            stream,
        )
        if rank == 0:
            print("Verifying results...")
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
        dist.barrier(device_ids=[rank])
        torch.testing.assert_close(input_tensor.cpu(), output_tensor.cpu())
        if rank == 0:
            print("Results verified successfully!")

    if not benchmark:
        return

    def generate_tensors():
        t = symm_mem.empty(
            [
                PING_PONG_SIZE,
                world_size,
                M * N,
            ],
            device="cuda",
        ).neg_()
        hdl = symm_mem.rendezvous(t, group=dist.group.WORLD.group_name)
        # 从 symmetric memory 获取其它 devices 上的 tensors
        buffers = [
            hdl.get_buffer(rank, t.shape, t.dtype).permute(2, 1, 0)
            for rank in range(world_size)
        ]
        input_tensor = torch.randn(M * N, device=f"cuda:{rank}")
        output_tensor = torch.zeros(M * N, device=f"cuda:{rank}")

        ja = testing.JitArguments(
            cutlass.Int32(0),
            from_dlpack(input_tensor, assumed_align=32),
            from_dlpack(output_tensor, assumed_align=32),
            [from_dlpack(t, assumed_align=32) for t in buffers],
            stream=stream,
        )
        ja._hdl = (
            hdl  # 延长 hdl 生命周期, 覆盖 kernel 执行期间
        )
        ja._t = t  # 同上
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
            f"Achieved memory throughput: {((world_size + 1) * output_tensor.numel() * 32 // 8) / (avg_time_us / 1e6) / 1e9:.2f} GB/s"
        )


def run(
    M,
    N,
    warmup_iterations=2,
    iterations=10,
    skip_ref_check=False,
    benchmark=True,
):
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    globals()["torch"] = torch
    globals()["dist"] = dist
    globals()["symm_mem"] = symm_mem

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    try:
        run_all_reduce_one_shot(
            M,
            N,
            warmup_iterations,
            iterations,
            skip_ref_check,
            benchmark,
        )
    finally:
        if dist.is_initialized():
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

    run(
        args.M,
        args.N,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.benchmark,
    )

    return


if __name__ == "__main__":
    main()
