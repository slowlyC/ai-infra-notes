/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//                             CuTe SM100 编程教程
// 本教程系列演示 CUTLASS 中常用的 CuTe Blackwell 能力.
// 目标是让开发者熟悉 CuTe SM100 接口.
//
// 教程系列分为五个阶段:
// * 01_mma_sm100.cu: 使用 tcgen05.mma 指令的简单 Blackwell SM100 GEMM.
// * 02_mma_tma_sm100.cu: 使用 tcgen05.mma 和 TMA 指令的简单 Blackwell SM100 GEMM.
// * 03_mma_tma_multicast_sm100.cu: 使用 tcgen05.mma 和 Multicast TMA 的 Blackwell SM100 GEMM.
// * 04_mma_tma_2sm_sm100.cu: 使用 2SM tcgen05.mma 和 2SM Multicast TMA 的 Blackwell SM100 GEMM.
// * 05_mma_tma_epi_sm100.cu: 使用 2SM tcgen05.mma, 2SM TMA mainloop 和 TMA epilogue 的 Blackwell SM100 GEMM.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cstdio>

// 使用 Thrust 处理 host/device 分配
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// CUTLASS 头文件
#include <cutlass/half.h>                       // F16 数据类型
#include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

// CuTe 头文件
#include <cute/tensor.hpp>                      // CuTe tensor 实现
#include <cute/arch/cluster_sm90.hpp>           // 查询 cluster launch 细节的 CuTe 函数
#include <cute/numeric/integral_constant.hpp>   // _1, _256 等编译期常量
#include <cute/algorithm/cooperative_copy.hpp>  // 自动向量化 copy 操作
#include <cute/arch/tmem_allocator_sm100.hpp>   // SM100 TMEM 分配器

// 教程辅助函数
#include "example_utils.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// 教程 03: 使用 tcgen05.mma 和 Multicast TMA 的 Blackwell SM100 GEMM
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// 这里实现 GEMM: D (f32) = beta * C (F32) + alpha * A (F16) * B (F16), 其中:
// - 矩阵 A 为 MxK, K-major (BLAS transpose T, row-major)
// - 矩阵 B 为 NxK, K-major (BLAS transpose N, column-major)
// - 矩阵 C 和 D 为 MxN, N-major (BLAS row-major)
//
// 相对 02_mma_tma_sm100.cu, 本教程新增:
// 1. 引入 ClusterShape, 用于 thread block 之间的协同执行
// 2. 引入 TMA multicast
// 3. 加强 TMA <-> MMA 同步, 支持 cluster-wide 操作
//
// 本 GEMM kernel 执行以下步骤:
// 1. 使用 Multicasted TMA load 操作将 A 和 B 矩阵从 GMEM 加载到 SMEM.
// 2. 使用 tcgen05.mma 指令执行 matrix multiply-accumulate (MMA).
// 3. 使用 tcgen05.ld 将完成的累加器从 TMEM 加载到 RMEM.
// 4. 从 GMEM 将 C 矩阵读入 RMEM.
// 5. 对 MMA 累加器和 C 矩阵应用 alpha 和 beta 缩放.
// 6. 将 D 矩阵从 RMEM 存储到 GMEM.
//
// SM100 tcgen05.mma 指令的行为如下:
// - 从 SMEM 或 TMEM 读取矩阵 A
// - 从 SMEM 读取矩阵 B
// - 将累加器写入 TMEM
// TMEM 中的累加器必须先加载到寄存器, 然后才能写回 GMEM.
//
// tcgen05.mma 指令需要一个 Instruction Descriptor, 用于编码 A, B 和累加器类型,
//   以及 MMA 的 M 和 N 维度.
// 从 SMEM 读取的 A 和 B 矩阵需要以 SMEM Descriptor 的形式提供给 MMA 指令.
//   在 CuTe 术语中, 它们是 tcgen05.mma 的 A 和 B fragment.
// 本教程会展示 CuTe 如何在指令和 fragment 中透明地提供这些 descriptor.
//
// MMA 细节:
// 本例使用 tcgen05.mma.f16 指令 (F16xF16 = F32), 执行 128x256x16 MMA.
// 因为 C 和 D 矩阵都使用 F32, 所以累加器类型也选择 F32.
// 本例的 F16xF16 = F32 MMA 类型如下:
// TypeA = cutlass::half_t;  // MMA A 数据类型
// TypeB = cutlass::half_t;  // MMA B 数据类型
// TypeC = float;            // MMA C 数据类型
// TypeD = float;            // MMA D 数据类型
// TypeAccumulator = float;  // TypeC 和 TypeD 都是 float, 因此使用 float 累加器

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// A 和 B 矩阵的共享内存缓冲区.
template <class TypeA,           // Tensor A 数据类型
          class TypeB,           // Tensor B 数据类型
          class ASmemLayout,     // (MmaA, NumMma_M, NumMma_K, ...)
          class BSmemLayout>     // (MmaB, NumMma_N, NumMma_K, ...)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t mma_barrier;  // 跟踪 SMEM 上 MMA 计算的 barrier
  alignas(16) cute::uint64_t tma_barrier;  // 跟踪 TMA 到 SMEM 传输的 barrier

  alignas(16) cute::uint32_t tmem_base_ptr; // TMEM 分配的基址指针

  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};

// Device kernel
template <class SharedStorage,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomB,
          class Alpha, class Beta>
__global__ static
void
gemm_device(ATensor mA,                      // (Gemm_M, Gemm_K)
            BTensor mB,                      // (Gemm_N, Gemm_K)
            CTensor mC,                      // (Gemm_M, Gemm_N)
            DTensor mD,                      // (Gemm_M, Gemm_N)
            MmaTiler_MNK mma_tiler,          // <MmaTile_M, MmaTile_N, MmaTile_K>
            TiledMMA tiled_mma,              // <    Mma_M,     Mma_N,     Mma_K>
            ClusterShape_MNK cluster_shape,  // (ClusterM, ClusterN, ClusterK)
            CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
            CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
            Alpha alpha, Beta beta)
{
  // 步骤 1: Prologue.

  // Cluster 内的 CTA layout: (V,M,N,K) -> CTA idx
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));

  // 根据 CTA grid 坐标构造 MMA grid 坐标
  auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk), // Peer CTA 坐标
                                   blockIdx.x / size<0>(cluster_layout_vmnk), //    MMA-M 坐标
                                   blockIdx.y,                                //    MMA-N 坐标
                                   _);                                        //    MMA-K 坐标

  // 使用 mma_tiler 和 mma_coord 对 GMEM tensor 分块, 得到本 mma tile 处理的切片.
  // CuTe 提供 local_tile 分块函数. local_tile 接受 4 个参数:
  //   * 要分块的 tensor
  //   * 用于分块的 tiler
  //   * 用于切分分块后 tensor 的坐标
  //   * projection, 用于忽略 Tiler 和 Coordinate 中不需要的 mode
  auto mma_coord = select<1,2,3>(mma_coord_vmnk);
  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X,_1>{});  // (MmaTile_M, MmaTile_K, Tiles_K)
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step< X,_1,_1>{});  // (MmaTile_N, MmaTile_K, Tiles_K)
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1,_1, X>{});  // (MmaTile_M, MmaTile_N)
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1,_1, X>{});  // (MmaTile_M, MmaTile_N)

  if (thread0()) {
    print("mA:\t"); print(mA); print("\n");   // mA:   ArithTuple(_0,_0) o (512,256):(_1@1,_1@0)
    print("mB:\t"); print(mB); print("\n");   // mB:   ArithTuple(_0,_0) o (1024,256):(_1@1,_1@0)
    print("mC:\t"); print(mC); print("\n");   // mC:   gmem_ptr[32b](GMEM_ADDR_C) o (512,1024):(1024,_1)
    print("mD:\t"); print(mD); print("\n");   // mD:   gmem_ptr[32b](GMEM_ADDR_D) o (512,1024):(1024,_1)

    print("gA:\t"); print(gA); print("\n");   // gA:   ArithTuple(_0,0) o (_128,_64,4):(_1@1,_1@0,_64@0)
    print("gB:\t"); print(gB); print("\n");   // gB:   ArithTuple(_0,0) o (_256,_64,4):(_1@1,_1@0,_64@0)
    print("gC:\t"); print(gC); print("\n");   // gC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile) o (_128,_256):(256,_1)
    print("gD:\t"); print(gD); print("\n");   // gD:   gmem_ptr[32b](GMEM_ADDR_D + offset_for_mma_tile) o (_128,_256):(256,_1)
  } __syncthreads();

  // SMEM tensor

  // 分配 SMEM
  extern __shared__ char shared_memory[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // 表示 A 和 B 的 SMEM 缓冲区
  Tensor tCsA = shared_storage.tensor_sA();         // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCsB = shared_storage.tensor_sB();         // (MmaB, NumMma_M, NumMma_K, Tiles_K)

  //
  // A 和 B 的 MMA 分块
  //

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);   // 使用 Peer CTA 坐标
  Tensor tCgA = cta_mma.partition_A(gA);         // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCgB = cta_mma.partition_B(gB);         // (MmaB, NumMma_N, NumMma_K, Tiles_K)
  Tensor tCgC = cta_mma.partition_C(gC);         // (MmaC, NumMma_M, NumMma_N)
  Tensor tCgD = cta_mma.partition_C(gD);         // (MmaC, NumMma_M, NumMma_N)

  if (thread0()) {
    print("tCgA:\t"); print(tCgA); print("\n");  // tCgA:   ArithTuple(_0,0) o ((_128,_16),_1,_4,4):((_1@1,_1@0),_0,_16@0,_64@0)
    print("tCgB:\t"); print(tCgB); print("\n");  // tCgB:   ArithTuple(_0,0) o ((_256,_16),_1,_4,4):((_1@1,_1@0),_0,_16@0,_64@0)
    print("tCgC:\t"); print(tCgC); print("\n");  // tCgC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile + offset_for_mma) o ((_128,_256),_1,_1):((256,_1),_0,_0)
    print("tCgD:\t"); print(tCgD); print("\n");  // tCgD:   gmem_ptr[32b](GMEM_ADDR_D + offset_for_mma_tile + offset_for_mma) o ((_128,_256),_1,_1):((256,_1),_0,_0)
  } __syncthreads();

  // MMA Fragment 分配
  // 这里分配 "fragment", 它们是作为 cute::gemm 输入的 SMEM descriptor.
  // 对 tcgen05.mma 操作:
  // - 矩阵 A 和 B 来自 SMEM
  // - tCrA 和 tCrB 分别提供 tCsA 和 tCsB 的 descriptor 视图
  // - 每个 descriptor 的第一个 mode 表示单次 MMA 操作使用的 SMEM
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);      // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);      // (MmaB, NumMma_M, NumMma_K, Tiles_K)

  // TMEM 分配
  // 在 SM100 架构上, 累加器只存放在 tensor memory (TMEM) 中.
  // ThrMma 的 make_fragment_C() 会创建一个 layout 适合累加器的 TMEM tensor.
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);    // (MmaC, NumMma_M, NumMma_N)

  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
  }
  __syncthreads(); // 等待所有线程, 直到 warp0 完成 TMEM 分配
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  if (thread0()) {
    print("tCsA:\t"); print(tCsA); print("\n");     // tCsA:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_A) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
    print("tCsB:\t"); print(tCsB); print("\n");     // tCsB:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_256,_16),_1,_4):((_64,_1),_0,_16)
    print("tCrA:\t"); print(tCrA); print("\n");     // tCrA:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
    print("tCrB:\t"); print(tCrB); print("\n");     // tCrB:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
    print("tCtAcc:\t"); print(tCtAcc); print("\n"); // tCtAcc: tmem_[32b](TMEM_ADDR) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
  } __syncthreads();

  // TMA 设置
  //
  //   这里是 TMA 分块, 它们使用专用的自定义 partitioner.
  //   本例中, TMA 会把 load multicast 到多个 CTA.
  //   A 的 load 沿 cluster_shape_MNK 的 N 维 multicast,
  //   B 的 load 沿 cluster_shape_MNK 的 M 维 multicast.
  //      所有 multicast 都必须与 host 端通过 make_tma_atom 构造的 tma_x 保持一致.
  //   对 A tensor: group_modes<0,3> 将形状为 (MmaA, NumMma_M, NumMma_K, Tiles_K) 的 tensor
  //      转换为 ((MmaA, NumMma_M, NumMma_K), Tiles_K). 分块只关注 mode-0, 即 MMA Tile MK.
  //   对 B tensor: group_modes<0,3> 将形状为 (MmaB, NumMma_M, NumMma_K, Tiles_K) 的 tensor
  //      转换为 ((MmaB, NumMma_M, NumMma_K), Tiles_K). 分块只关注 mode-0, 即 MMA Tile NK.
  //   简单说, TMA 会通过一次 cute::copy 负责 mode-0 中的所有内容.
  //   tma_partition 会根据 tma_x atom 和 multicast 信息对 mode-0 重排并添加 offset.

  // m 坐标相同的每个 CTA 都会加载 A 的一部分
  // n 坐标相同的每个 CTA 都会加载 B 的一部分
  // cluster 中 CTA 1,2 的 multicast 行为
  //   A multicast            B multicast
  //    0  1  2  3             0  1  2  3
  // 0  -  -  -  -          0  -  -  X  -
  // 1  X  X  X  X          1  -  -  X  -
  // 2  -  -  -  -          2  -  -  X  -
  // 3  -  -  -  -          3  -  -  X  -
  // tma_multicast_mask_A = 0x2222
  // tma_multicast_mask_B = 0x0F00
  // mma_multicast_mask_C = 0x2F22

  // 构造用于 multicast 的 CTA-in-Cluster 坐标
  auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));

  // 沿 N-mode 投影 tma_A 的 cluster_layout
  auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                    get<2>(cta_in_cluster_coord_vmnk),          // CTA 在 cluster N mode 上的坐标
                                    make_layout(size<2>(cluster_layout_vmnk)),  // CTA 在 cluster N mode 上的 layout
                                    group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));

  // 沿 M-mode 投影 tma_B 的 cluster_layout
  auto [tBgB, tBsB] = tma_partition(tma_atom_B,
                                    get<1>(cta_in_cluster_coord_vmnk),          // CTA 在 cluster M mode 上的坐标
                                    make_layout(size<1>(cluster_layout_vmnk)),  // CTA 在 cluster M mode 上的 layout
                                    group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));

  // 沿 N-mode 投影 cluster_layout 和 cta_coord, 得到 A 的 multicast mask
  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  // 沿 M-mode 投影 cluster_layout 和 cta_coord, 得到 B 的 multicast mask
  uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  // 沿 VM + VN mode 投影 cluster_layout 和 cta_coord, 得到 C 的 multicast mask
  uint16_t mma_mcast_mask_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                              create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

  // 计算 TMA 每个 tile 传输的总字节数, 用于跟踪完成状态
  int tma_transaction_bytes = sizeof(make_tensor_like(tAsA))
                            + sizeof(make_tensor_like(tBsB));

  if (thread0()) {
    print("tAgA:\t"); print(tAgA); print("\n");  // tAgA:   ArithTuple(_0,0) o (((_64,_128),_1),4):(((_1@0,_1@1),_0),_64@0)
    print("tAsA:\t"); print(tAsA); print("\n");  // tAsA:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_A) o ((_8192,_1)):((_1,_0))
    print("tBgB:\t"); print(tBgB); print("\n");  // tBgB:   ArithTuple(_0,0) o (((_64,_256),_1),4):(((_1@0,_1@1),_0),_64@0)
    print("tBsB:\t"); print(tBsB); print("\n");  // tBsB:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_16384,_1)):((_1,_0))
    printf("tma_transaction_bytes: %d\n", tma_transaction_bytes);
    printf("tma_mcast_mask_a: %x\n", tma_mcast_mask_a);
    printf("tma_mcast_mask_b: %x\n", tma_mcast_mask_b);
    printf("mma_mcast_mask_c: %x\n", mma_mcast_mask_c);
  } __syncthreads();

  // Barrier 初始化
  // SMEM 中的 barrier 由单个线程初始化.
  if (elect_one_warp && elect_one_thr) {
    // 与当前 CTA 一起参与 multicast 的 CTA 数量 (对 A 和 B 矩阵都是如此)
    int num_mcast_participants = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
    cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */ num_mcast_participants);
    cute::initialize_barrier(shared_storage.tma_barrier, /* num_threads */ 1);
  }
  int mma_barrier_phase_bit = 0;  // 每个 barrier 都有一个关联的 phase_bit.
  int tma_barrier_phase_bit = 0;  // 每个 barrier 都有一个关联的 phase_bit.
  cute::cluster_sync();           // 确保 Cluster 内所有 CTA 的所有线程都能看到 barrier 初始化.

  // 步骤 2: Mainloop.

  // 将 mma accumulate 选项设为 zero, 使第一条 MMA 指令清空 TMEM 累加器.
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // 执行 MmaTile_M x MmaTile_N x GEMM_K GEMM
  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile)
  {
    // 步骤 2a: 加载 A 和 B tile

    // TMA load 操作:
    // - 使用单个线程执行异步 TMA load
    // - 设置 transaction bytes, 并通过 barrier 执行
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
      copy(tma_atom_A.with(shared_storage.tma_barrier,tma_mcast_mask_a), tAgA(_,k_tile), tAsA); // 加载 MmaTile_M x MmaTile_K A tile
      copy(tma_atom_B.with(shared_storage.tma_barrier,tma_mcast_mask_b), tBgB(_,k_tile), tBsB); // 加载 MmaTile_N x MmaTile_K B tile
    }

    // 步骤 2b: 执行本 tile 的 MMA

    // 等待 TMA load 到 SMEM 完成
    cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;

    // tcgen05.mma 指令要求单线程执行:
    // - 只有一个 warp 执行 MMA 相关循环操作
    // - CuTe 操作在内部管理 tcgen05.mma 和 tcgen05.cp 的单线程执行
    // - 用户不需要显式写 elect_one_sync 区域
    if (elect_one_warp) {
      // 执行 MmaTile_M x MmaTile_N x MmaTile_K GEMM
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      // 确保 MMA 已完成, 之后才能复用 A 和 B 的 SMEM.
      cutlass::arch::umma_arrive_multicast(&shared_storage.mma_barrier, mma_mcast_mask_c); // 所有 multicast CTA 都编码在 mask 中.
    }
    // 等待 MMA 完成, 避免覆盖 A 和 B 的 SMEM.
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }

  // 步骤 3: Epilogue.

  // 为累加器创建 tiled copy 操作 (TMEM -> RMEM)
  TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy   thr_t2r_copy   = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDgC = thr_t2r_copy.partition_D(tCgC);                   // (CpyD, NumCpy_M, NumCpy_N)
  Tensor tDrC = make_fragment_like(tDgC);                         // (CpyD, NumCpy_M, NumCpy_N)
  // 加载 C tensor, GMEM -> RMEM
  copy(tDgC, tDrC);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);               // (CpyS, NumCpy_M, NumCpy_N)
  Tensor tDgD   = thr_t2r_copy.partition_D(tCgD);                 // (CpyD, NumCpy_M, NumCpy_N)
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));              // (CpyD, NumCpy_M, NumCpy_N)
  // 加载 TMEM -> RMEM
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  // AXPBY RMEM -> RMEM: tDrC = alpha * tDrAcc + beta * tDrC
  axpby(alpha, tDrAcc, beta, tDrC);
  // 存储 RMEM -> GMEM
  copy(tDrC, tDgD);

  __syncthreads();

  // 先释放分配权, 让下一个 CTA 可以被调度.
  // 然后释放 TMEM.
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class TypeA, class LayoutA,
          class TypeB, class LayoutB,
          class TypeC, class LayoutC,
          class TypeD, class LayoutD,
          class Alpha, class Beta>
void gemm_host_f16xf16_f32_f32_tnt(TypeA const* device_ptr_A, LayoutA layout_A,
                                   TypeB const* device_ptr_B, LayoutB layout_B,
                                   TypeC const* device_ptr_C, LayoutC layout_C,
                                   TypeD      * device_ptr_D, LayoutD layout_D,
                                   Alpha const alpha, Beta const beta)
{
  assert(shape<0>(layout_A) == shape<0>(layout_C));  // Gemm_M
  assert(shape<0>(layout_A) == shape<0>(layout_D));  // Gemm_M
  assert(shape<0>(layout_B) == shape<1>(layout_C));  // Gemm_N
  assert(shape<0>(layout_B) == shape<1>(layout_D));  // Gemm_N
  assert(shape<1>(layout_A) == shape<1>(layout_B));  // Gemm_K

  // 在 global memory 中表示完整 tensor
  Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);      // (Gemm_M, Gemm_K)
  Tensor mB = make_tensor(make_gmem_ptr(device_ptr_B), layout_B);      // (Gemm_N, Gemm_K)
  Tensor mC = make_tensor(make_gmem_ptr(device_ptr_C), layout_C);      // (Gemm_M, Gemm_N)
  Tensor mD = make_tensor(make_gmem_ptr(device_ptr_D), layout_D);      // (Gemm_M, Gemm_N)

  // 获取当前 GEMM 的 M, N, K 维度
  auto Gemm_M = shape<0>(layout_A);
  auto Gemm_N = shape<0>(layout_B);
  auto Gemm_K = shape<1>(layout_A);
  std::cout << "Running for problem shape (MxNxK): " << Gemm_M << "x" << Gemm_N << "x" << Gemm_K << std::endl;

  ////////////////////////////////////////////////////////////
  //
  // 初始化 GEMM kernel 参数
  //
  ////////////////////////////////////////////////////////////

  // 创建 TiledMma. make_tiled_mma 以目标指令和可选的指令 layout 为参数,
  // 基于给定 mma 指令创建更大的 TiledMma.
  // 查看 cute/arch/mma_sm100_umma.hpp 可了解所有 tcgen05.mma 指令.
  TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC,                 // Mma 的 A, B 和累加器类型
                                                           128, 256,                            // Mma 的 M 和 N 维度
                                                           UMMA::Major::K, UMMA::Major::K>{});  // A 和 B layout

  // 也可以打印并检查 tiled_mma
  print(tiled_mma);
  // TiledMMA
  //   ThrLayoutVMNK:  (_1,_1,_1,_1):(_0,_0,_0,_0)
  //   PermutationMNK: (_,_,_)
  // MMA_Atom
  //   ThrID:      _1:_0
  //   Shape_MNK:  (_128,_256,_16)                      // MmaM, MmaN, MmaK 指令尺寸
  //   LayoutA_TV: (_1,(_128,_16)):(_0,(_1,_128))       // TV -> A 矩阵的 MmaCoordinate 映射
  //   LayoutB_TV: (_1,(_256,_16)):(_0,(_1,_256))       // TV -> B 矩阵的 MmaCoordinate 映射
  //   LayoutC_TV: (_1,(_128,_256)):(_0,(_1,_128))      // TV -> C 矩阵的 MmaCoordinate 映射

  // 定义 MMA tiler 尺寸 (静态)
  auto bM = tile_size<0>(tiled_mma);             // MMA Tile M. 每个 MMA Tile M 使用 1 个 MMA.
  auto bN = tile_size<1>(tiled_mma);             // MMA Tile N. 每个 MMA Tile N 使用 1 个 MMA.
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};  // MMA Tile K. 每个 MMA Tile K 使用 4 个 MMA. 对 16b 类型, tcgen05.mma 为 K16.
  auto mma_tiler = make_shape(bM, bN, bK);       // (MMA_M, MMA_N, MMA_K)

  // 在 SM90 中, MMA 是 CTA-local 的, 并执行线程级分块.
  // 在 SM100 中, MMA 是 Cluster-local 的, 并执行 CTA 级分块.
  // 因此, SM90 使用 cta_tiler 从 Problem 中取出 CTA 对应部分,
  // 而 SM100 使用 mma_tiler 从 Problem 中取出 MMA 对应部分.
  // 随后 MMA 的分块会给出 CTA-local 的工作.

  if (not evenly_divides(shape(mma_tiler), tile_shape(tiled_mma))) {
    std::cerr << "The MMA Shape should evenly divide the MMA Tiler." << std::endl;
    return;
  }

  if (not evenly_divides(make_shape(Gemm_M, Gemm_N, Gemm_K), mma_tiler)) {
    std::cerr << "OOB accesses are not supported. MmaTiler_MNK should evenly divide ProblemShape_MNK." << std::endl;
    return;
  }

  //
  // 确定 SMEM layout:
  //

  //  * A 和 B 的 SMEM layout 必须匹配 MMA 指令期望的后分块 (CTA-local) shape.
  //  * CuTe 提供 partition_shape_[A|B] 函数来确定后分块 shape.
  //    这些函数以 TiledMma 和 MMA Tile Shape 为输入, 返回至少 rank-3 的 shape.
  //    其中第一个 mode 与 MMA 指令具有相同 shape, 第二和第三个 mode 分别表示
  //    MMA 指令在 MMA tile 的 M/N mode 和 K mode 上重复的次数.
  //  * 注意, kernel launch 时需要 SMEM layout 来确定 SMEM 分配大小.

  // 分块前 Tile Shape (MmaTile_M, MmaTile_K) 到分块后 (MmaA, NumMma_M, NumMma_K)
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  // 分块前 Tile Shape (MmaTile_N, MmaTile_K) 到分块后 (MmaB, NumMma_N, NumMma_K)
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  // 打印并检查本例的 mma_shape_A 和 mma_shape_B.
  print("mma_shape_A:\t"); print(mma_shape_A); print("\n");  // mma_shape_A:  ((_128,_16),_1,_4)
  print("mma_shape_B:\t"); print(mma_shape_B); print("\n");  // mma_shape_B:  ((_256,_16),_1,_4)

  // A 和 B tensor 在 SMEM 中使用 swizzle 以提升 MMA 性能.
  //  * 但表达 swizzled layout 很困难.
  //  * CuTe 为 SM100 提供 tile_to_mma_shape 函数, 用于为后分块 Mma Shape 创建 swizzled layout.
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

  // 打印并检查本例的 sA_layout 和 sB_layout.
  print("sA_layout:\t"); print(sA_layout); print("\n");      // sA_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
  print("sB_layout:\t"); print(sB_layout); print("\n");      // sB_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_256,_16),_1,_4):((_64,_1),_0,_16)

  // 现在可以得到 SMEM 分配大小
  using SMEMStorage = SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  //
  // TMA Descriptor 创建 (host 侧)
  //

  // Cluster shape 和 layout
  auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  Copy_Atom tma_atom_A = make_tma_atom(
      SM90_TMA_LOAD_MULTICAST{},       // 带 multicast 的 TMA load 操作
      mA,                              // 源 GMEM tensor
      sA_layout,                       // 目标 SMEM layout
      select<0,2>(mma_tiler),          // TMA 操作使用的 MK Tiler
      size<2>(cluster_layout_vmnk)     // N-mode 上参与 multicast 的 CTA 数量
    );
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));   // (Gemm_M, Gemm_K)

  print("tma_atom_A:\t"); print(tma_atom_A); print("\n");
  // tma_atom_A:     Copy_Atom
  //  ThrID:        _1:_0
  //  ValLayoutSrc: (_1,_8192):(_0,_1)
  //  ValLayoutDst: (_1,_8192):(_0,_1)
  //  ValLayoutRef: (_1,_8192):(_0,_1)
  //  ValueType:    16b

  Copy_Atom tma_atom_B = make_tma_atom(
      SM90_TMA_LOAD_MULTICAST{},      // 带 multicast 的 TMA load 操作
      mB,                             // 源 GMEM tensor
      sB_layout,                      // 目标 SMEM layout
      select<1,2>(mma_tiler),         // TMA 操作使用的 NK Tiler
      size<1>(cluster_layout_vmnk)    // M-mode 上参与 multicast 的 CTA 数量
    );
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));   // (Gemm_N, Gemm_K)

  print("tma_atom_B:\t"); print(tma_atom_B); print("\n");
  // tma_atom_B:     Copy_Atom
  //  ThrID:        _1:_0
  //  ValLayoutSrc: (_1,_16384):(_0,_1)
  //  ValLayoutDst: (_1,_16384):(_0,_1)
  //  ValLayoutRef: (_1,_16384):(_0,_1)
  //  ValueType:    16b

  ////////////////////////////////////////////////////////////
  //
  // 启动 GEMM kernel
  //
  ////////////////////////////////////////////////////////////

  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
  dim3 dimGrid(size(ceil_div(Gemm_M, bM * size<1>(cluster_layout_vmnk))) * dimCluster.x,
               size(ceil_div(Gemm_N, bN * size<2>(cluster_layout_vmnk))) * dimCluster.y);
  int  smemBytes = sizeof(SMEMStorage);

  auto* kernel_ptr = &gemm_device<SMEMStorage,
                                  decltype(mA_tma), decltype(mB_tma), decltype(mC), decltype(mD),
                                  decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                                  decltype(tma_atom_A), decltype(tma_atom_B), // 包含 TMA descriptor.
                                  Alpha, Beta>;

  // 设置 kernel 属性 (设置 SMEM)
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes));

  printf("Grid launched: %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  printf("Cluster launched: %d, %d, %d\n", dimCluster.x, dimCluster.y, dimCluster.z);

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                             mA_tma, mB_tma, mC, mD,
                                                             mma_tiler, tiled_mma, cluster_shape,
                                                             tma_atom_A, tma_atom_B,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

int main(int argc, char** argv)
{
  cudaDeviceProp props;
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaGetDeviceProperties(&props, current_device_id);
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if ((props.major != 10) || (props.major == 10 && props.minor > 1)) {
    std::cerr << "This example requires NVIDIA's Blackwell Architecture GPU with compute capability 100a." << std::endl;
    std::cerr << "  Found " << props.major << "." << props.minor << std::endl;
    return -1;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  int Gemm_M = 512;
  if (argc >= 2)
    sscanf(argv[1], "%d", &Gemm_M);

  int Gemm_N = 1024;
  if (argc >= 3)
    sscanf(argv[2], "%d", &Gemm_N);

  int Gemm_K = 256;
  if (argc >= 4)
    sscanf(argv[3], "%d", &Gemm_K);

  ////////////////////////////////////////////////////////////
  //
  // 创建 A, B, C 和 D tensor
  //
  ////////////////////////////////////////////////////////////
  // 定义数据类型. A 和 B 的类型与 MMA 指令一致.
  using TypeA = cutlass::half_t; // MMA A 数据类型
  auto type_str_a = "half_t";
  using TypeB = cutlass::half_t; // MMA B 数据类型
  auto type_str_b = "half_t";
  using TypeC = float;           // MMA C 数据类型
  [[maybe_unused]] auto type_str_c = "float";
  using TypeD = float;           // MMA D 数据类型
  auto type_str_d = "float";
  using TypeAccumulator = float; // TypeC 和 TypeD 都是 float, 因此使用 float 累加器.

  // A tensor MxK K-major (Layout T = Row-Major)
  Layout layout_A = make_layout(make_shape (Gemm_M,   Gemm_K),
                                make_stride(Gemm_K, Int<1>{}));   // (Gemm_M,Gemm_K):(Gemm_K,_1)
  // B tensor NxK K-major (Layout N = Column-Major)
  Layout layout_B = make_layout(make_shape (Gemm_N,   Gemm_K),
                                make_stride(Gemm_K, Int<1>{}));   // (Gemm_N,Gemm_K):(Gemm_K,_1)
  // C tensor MxN N-major (Layout T = Row-Major)
  Layout layout_C = make_layout(make_shape (Gemm_M,   Gemm_N),
                                make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
  // D tensor MxN N-major (Layout T = Row-Major)
  Layout layout_D = make_layout(make_shape (Gemm_M,   Gemm_N),
                                make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)

  // 为 A, B 和 C tensor 创建 host 端分配及 host CuTe tensor.
  thrust::host_vector<TypeA>   host_A(Gemm_M * Gemm_K);
  Tensor host_tensor_A = make_tensor(host_A.data(), layout_A);
  print("host_tensor_A:\t"); print(host_tensor_A); print("\n"); // host_tensor_A:	ptr[16b](ADDR_A) o (512,256):(256,_1)

  thrust::host_vector<TypeB>   host_B(Gemm_N * Gemm_K);
  Tensor host_tensor_B = make_tensor(host_B.data(), layout_B);
  print("host_tensor_B:\t"); print(host_tensor_B); print("\n"); // host_tensor_B:	ptr[16b](ADDR_B) o (1024,256):(256,_1)

  thrust::host_vector<TypeC>   host_C(Gemm_M * Gemm_N);
  Tensor host_tensor_C = make_tensor(host_C.data(), layout_C);
  print("host_tensor_C:\t"); print(host_tensor_C); print("\n"); // host_tensor_C:	ptr[32b](ADDR_C) o (512,1024):(1024,_1)

  // 这里暂时不需要为 D 创建 host_tensor.
  thrust::device_vector<TypeD> device_D(Gemm_M * Gemm_N);

  // 使用随机值初始化 A, B 和 C tensor.
  initialize_tensor(host_tensor_A);
  initialize_tensor(host_tensor_B);
  initialize_tensor(host_tensor_C);

  // 将 A, B 和 C tensor 从 host memory 拷贝到 device memory
  thrust::device_vector<TypeA> device_A = host_A;
  thrust::device_vector<TypeB> device_B = host_B;
  thrust::device_vector<TypeC> device_C = host_C;

  using Alpha = float;
  using Beta = float;
  Alpha alpha = 1.0f;
  Beta beta = 0.0f;
  // 设置输入/输出 tensor 和 kernel 参数, 并在 device 上执行 kernel
  gemm_host_f16xf16_f32_f32_tnt(device_A.data().get(), layout_A,
                                device_B.data().get(), layout_B,
                                device_C.data().get(), layout_C,
                                device_D.data().get(), layout_D,
                                alpha, beta);
  // 为 D tensor 创建 host 端分配, 并将 D tensor 从 device 传回 host
  thrust::host_vector<TypeD> host_D = device_D;
  // 为 D tensor 创建 non-owning CuTe tensor
  Tensor host_tensor_D = make_tensor(host_D.data(), layout_D);

  ////////////////////////////////////////////////////////////
  //
  // 执行 reference GEMM kernel
  //
  ////////////////////////////////////////////////////////////

  thrust::host_vector<TypeD> host_reference_D(Gemm_M*Gemm_N);
  auto host_reference_tensor_D = make_tensor(host_reference_D.data(), layout_D);
  reference_gemm<TypeAccumulator>(host_tensor_A, host_tensor_B, host_tensor_C, host_reference_tensor_D, alpha, beta);

  ////////////////////////////////////////////////////////////
  //
  // 比较结果
  //
  ////////////////////////////////////////////////////////////

  auto relative_error = print_matrix_multiply_mollified_relative_error(type_str_a, host_tensor_A,
                                                                       type_str_b, host_tensor_B,
                                                                       type_str_d, host_tensor_D, host_reference_tensor_D);
  bool success = relative_error <= 0.0;
  std::cout << "Execution is " << ((success) ? "successful." : "failed.") << std::endl;
#else
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED must be enabled, but it is not. Test is waived \n" << std::endl;
#endif

  return 0;
}
