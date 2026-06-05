/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// 本示例演示如何将张量分块 (tile) 并进行高效、合并的拷贝. 同时展示如何做向量化访问,
// 这对某些工作负载可能是有用优化或必要条件.
//
// `copy_kernel()` 和 `copy_kernel_vectorized()` 均假定一对维度为 (m, n) 的张量已通过
// `tiled_divide()` 完成分块.
//
// 结果是一组维度为 ((M, N), m', n') 的相容张量, 其中 (M, N) 表示静态尺寸的 tile,
// m' 和 n' 表示张量中此类 tile 的数量.
//
// 每个静态 tile 映射到一个 CUDA 线程块, 负责对全局内存做高效 load/store.
//
// `copy_kernel()` 使用 `cute::local_partition()` 对张量分块, 并用条带状索引将结果
// 映射到线程. 线程以 (ThreadShape_M, ThreadShape_N) 方式排布, 并在 tile 上复制.
//
// `copy_kernel_vectorized()` 使用 `cute::make_tiled_copy()` 做类似分块, 通过
// `cute::Copy_Atom` 实现向量化. 实际向量宽度由 `ThreadShape` 定义.
//
// 本示例假定整体张量形状可被 tile 尺寸整除, 不做谓词判断.


/// 简单拷贝核函数.
//
// 使用 local_partition() 将 tile 按 (THR_M, THR_N) 排布的线程进行划分.
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout)
{
  using namespace cute;

  // 对分块后的张量切片
  Tensor tile_S = S(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)

  // 按给定线程布局, 将 tile 在线程间划分.

  // 概念:                            张量    线程布局        线程索引
  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)

  // 构造与每个线程分区同形状的寄存器张量.
  // 使用 make_tensor 尝试匹配 thr_tile_S 的布局.
  Tensor fragment = make_tensor_like(thr_tile_S);               // (ThrValM, ThrValN)

  // 从 GMEM 拷到 RMEM, 再从 RMEM 拷到 GMEM.
  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}

/// 向量化拷贝核函数.
///
/// 使用 `make_tiled_copy()` 通过向量指令完成拷贝. 前置条件: 指针需按向量尺寸对齐.
///
template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy)
{
  using namespace cute;

  // 对张量切片, 得到每个 tile 的视图.
  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)

  // 构造对应每个线程切片的张量.
  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)

  // 构造与每个线程分区同形状的寄存器张量.
  // 因首模式为指令局部模式, 使用 make_fragment.
  Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)

  // 从 GMEM 拷到 RMEM, 再从 RMEM 拷到 GMEM.
  copy(tiled_copy, thr_tile_S, fragment);
  copy(tiled_copy, fragment, thr_tile_D);
}

/// 主函数
int main(int argc, char** argv)
{
  //
  // 给定二维形状, 执行高效拷贝
  //

  using namespace cute;
  using Element = float;

  // 定义动态维度 (m, n) 的张量形状
  auto tensor_shape = make_shape(256, 512);

  //
  // 分配并初始化
  //

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  //
  // 构造张量
  //

  Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));

  //
  // 对张量分块
  //

  // 定义静态尺寸的 block (M, N).
  // 约定: 大写字母表示静态模式.
  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
    return -1;
  }
  // 与上述判断等价
  if (not evenly_divides(tensor_shape, block_shape)) {
    std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
    return -1;
  }

  // 对张量分块: (m, n) ==> ((M, N), m', n'), 其中 (M, N) 为静态 tile 形状,
  // 模式 (m', n') 为 tile 数量.
  //
  // 将用于确定 CUDA 核函数的 grid 维度.
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')

  // 构造具有特定访问模式的 TiledCopy.
  //   该版本使用:
  //   (1) 线程布局描述线程数量与排布 (如行主序、列主序等),
  //   (2) 每个线程将访问的值布局.

  // 线程排布
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32,8) -> thr_idx

  // 每个线程的值排布
  Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));   // (4,1) -> val_idx

  // 定义 `AccessType`, 控制实际内存访问指令的宽度.
  using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;     // 特定访问宽度的拷贝指令
  //using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // 支持多种拷贝策略的通用类型
  //using CopyOp = AutoVectorizingCopy;                                              // 假定输入最大对齐的自适应宽度指令

  // Copy_Atom 对应于施加于 Element 类型张量的一次 CopyOperation.
  using Atom = Copy_Atom<CopyOp, Element>;

  // 构造 tiled copy: 对 copy atoms 做分块.
  //
  // 注意: 假定向量和线程布局与 GMEM 中的连续数据对齐. 其他线程布局可行,
  // 但可能导致非合并读取. 其他值布局也可行, 但不相容布局会在编译期报错.
  TiledCopy tiled_copy = make_tiled_copy(Atom{},             // 访问策略
                                         thr_layout,         // 线程布局 (如 32x4 列主序)
                                         val_layout);        // 值布局 (如 4x1)

  //
  // 确定 grid 与 block 维度
  //

  dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid 形状对应模式 m' 与 n'
  dim3 blockDim(size(thr_layout));

  //
  // 启动核函数
  //
  copy_kernel_vectorized<<< gridDim, blockDim >>>(
    tiled_tensor_S,
    tiled_tensor_D,
    tiled_copy);

  cudaError result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  //
  // 校验
  //

  h_D = d_D;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  return 0;
}
