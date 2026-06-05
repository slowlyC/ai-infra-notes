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

// 本示例在 `tiled_copy` 基础上, 使用谓词张量保护 `cute::copy_if()` 的内存访问.
// 从而支持形状不是 block 尺寸整数倍的张量.
//
// 实现方式: 实例化一个坐标张量以对应待访问张量元素, 再计算用于掩蔽访问的谓词张量.
// 示例展示如何用与 GMEM 张量分块相同的 CuTe 操作, 构造包含坐标的恒等张量和包含
// 掩码位的谓词张量.
//
// 本示例实现两种变体:
//  - copy_if_kernel() 使用 `cute::local_partition()` 构造每个线程的切片
//  - copy_if_kernel_vectorized() 使用 `make_tiled_copy()` 实现向量化内存访问.
//
// 张量形状和步长必须能被向量访问的形状整除.
//

/// 简单拷贝核函数.
//
// 使用 local_partition() 将 tile 按 (THR_M, THR_N) 排布的线程进行划分.
template <class TensorS, class TensorD, class BlockShape, class ThreadLayout>
__global__ void copy_if_kernel(TensorS S, TensorD D, BlockShape block_shape, ThreadLayout)
{
  using namespace cute;

  // 构造坐标张量, 其元素为访问张量 S 和 D 时所用的坐标.
  auto shape_S = shape(S);
  Tensor C = make_identity_tensor(shape_S);
  // 构造谓词张量: 将坐标与原形状比较.
  Tensor P = cute::lazy::transform(C, [&](auto c) { return elem_less(c, shape_S); });

  // 将输入张量分块
  auto block_coord = make_coord(blockIdx.x, blockIdx.y);
  Tensor tile_S = local_tile(S, block_shape, block_coord);   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = local_tile(D, block_shape, block_coord);   // (BlockShape_M, BlockShape_N)
  Tensor tile_P = local_tile(P, block_shape, block_coord);   // (BlockShape_M, BlockShape_N)

  // 按给定线程布局, 将 tile 在线程间划分.

  // 概念:                            张量    线程布局        线程索引
  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);
  Tensor thr_tile_P = local_partition(tile_P, ThreadLayout{}, threadIdx.x);

  // 使用 `thr_tile_P` 保护访问, 从 GMEM 拷到 GMEM.
  copy_if(thr_tile_P, thr_tile_S, thr_tile_D);
}

/// 向量化拷贝核函数.
///
/// 使用 `make_tiled_copy()` 通过向量指令完成拷贝. 前置条件: 指针需按向量尺寸对齐.
///
template <class TensorS, class TensorD, class BlockShape, class Tiled_Copy>
__global__ void copy_if_kernel_vectorized(TensorS S, TensorD D, BlockShape block_shape, Tiled_Copy tiled_copy)
{
  using namespace cute;

  // 构造坐标张量, 其元素为访问张量 S 和 D 时所用的坐标.
  auto shape_S = shape(S);
  Tensor C = make_identity_tensor(shape_S);
  // 构造谓词张量: 将坐标与原形状比较.
  Tensor P = cute::lazy::transform(C, [&](auto c) { return elem_less(c, shape_S); });

  // 将输入张量分块
  auto block_coord = make_coord(blockIdx.x, blockIdx.y);
  Tensor tile_S = local_tile(S, block_shape, block_coord);       // (BlockShape_M, BlockShape_N)
  Tensor tile_D = local_tile(D, block_shape, block_coord);       // (BlockShape_M, BlockShape_N)
  Tensor tile_P = local_tile(P, block_shape, block_coord);       // (BlockShape_M, BlockShape_N)

  //
  // 构造对应每个线程切片的张量.
  //
  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  Tensor thr_tile_S = thr_copy.partition_S(tile_S);              // (CPY, CPY_M, CPY_N)
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);              // (CPY, CPY_M, CPY_N)
  Tensor thr_tile_P = thr_copy.partition_S(tile_P);              // (CPY, CPY_M, CPY_N)

#if 0
  // 从 GMEM 拷到 GMEM
  copy_if(tiled_copy, thr_tile_P, thr_tile_S, thr_tile_D);
#else
  // make_fragment_like() 在 RMEM 中构造与 thr_tile_S 同形状的张量.
  Tensor frag = make_fragment_like(thr_tile_S);

  // 从 GMEM 拷到 RMEM, 再从 RMEM 拷到 GMEM
  copy_if(tiled_copy, thr_tile_P, thr_tile_S, frag);
  copy_if(tiled_copy, thr_tile_P, frag,       thr_tile_D);
#endif
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
  auto tensor_shape = make_shape(528, 300);

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  //
  // 初始化
  //

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;
  thrust::device_vector<Element> d_Zero = h_D;

  //
  // 构造张量
  //

  Tensor tensor_S = make_tensor(make_gmem_ptr(d_S.data().get()), make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(make_gmem_ptr(d_D.data().get()), make_layout(tensor_shape));

  //
  // 分块
  //

  // 定义静态尺寸的 block (M, N).
  //
  // 约定: 大写字母表示静态模式.
  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  // 对张量分块: (m, n) ==> ((M, N), m', n'), 其中 (M, N) 为静态 tile 形状,
  // 模式 (m', n') 为 tile 数量.
  //
  // 将用于确定 CUDA 核函数的 grid 维度.
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);        // ((M, N), m', n')

  // 描述线程布局, 并在 tile 'block_shape' 上复制.
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int< 8>{}));  // (ThrM, ThrN)

  //
  // 确定 grid 与 block 维度
  //

  dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid 形状对应模式 m' 与 n'
  dim3 blockDim(size(thr_layout));

  //
  // 启动核函数
  //

  // copy_if()
  copy_if_kernel<<< gridDim, blockDim >>>(
    tensor_S,
    tensor_D,
    block_shape,
    thr_layout);

  cudaError result = cudaDeviceSynchronize();

  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  h_D = d_D;

  //
  // 校验
  //

  auto verify = [](thrust::host_vector<Element> const &S, thrust::host_vector<Element> const &D){

    int32_t errors = 0;
    int32_t const kErrorLimit = 10;

    if (S.size() != D.size()) {
      return 1;
    }

    for (size_t i = 0; i < D.size(); ++i) {
      if (S[i] != D[i]) {
        std::cerr << "Error. S[" << i << "]: " << S[i] << ",   D[" << i << "]: " << D[i] << std::endl;

        if (++errors >= kErrorLimit) {
          std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
          return errors;
        }
      }
    }

    return errors;
  };

  if (verify(h_D, h_S)) {
    return -1;
  } else {
    std::cout << "Success." << std::endl;
  }

  thrust::copy(d_Zero.begin(), d_Zero.end(), d_D.begin());

  // 构造具有特定访问模式的 TiledCopy.
  //   该版本使用:
  //   (1) 线程布局描述线程数量与排布 (如行主序、列主序等),
  //   (2) 每个线程将访问的值布局.

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

  // 带向量化的 copy_if()
  copy_if_kernel_vectorized<<< gridDim, blockDim >>>(
    tensor_S,
    tensor_D,
    block_shape,
    tiled_copy);

  result = cudaDeviceSynchronize();

  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  h_D = d_D;

  if (verify(h_D, h_S)) {
    return -1;
  } else {
    std::cout << "Success." << std::endl;
  }
  return 0;
}
