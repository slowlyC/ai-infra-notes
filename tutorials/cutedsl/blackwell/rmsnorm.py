# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import ctypes
import functools
import math
from typing import Optional, Tuple, Union

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass import Boolean, Float32, Int32, Int64
from cutlass.cute.runtime import make_ptr

# 支持直接运行和模块导入两种方式
try:
    from .reduce import row_reduce
except ImportError:
    from reduce import row_reduce

"""
RMSNorm: Root Mean Square 层归一化, 适用于 Hopper & Blackwell (SM90+)
====================================================================

基于 CuTe DSL 的高性能 RMSNorm 实现, 通过 cluster 级归约支持大隐藏维度.

RMSNorm 计算公式: y = x / sqrt(mean(x^2) + eps) * weight

各变量维度:
- x:      (M, N)  输入, M 为 token 数, N 为隐藏维度
- y:      (M, N)  输出, 与 x 同形
- x^2:    (M, N)  逐元素平方
- mean(): 沿 N 维求均值, 结果 (M, 1), 即每行一个标量
- rms:    (M, 1)  sqrt(mean(x^2) + eps)
- weight: (N,)    可学习的 per-channel 缩放因子, 沿 M 维广播


核心特性:
---------
1. CLUSTER 同步 (SM90+)
   - 多个 CTA 协作处理大 N 维度
   - 每个 CTA 处理 N/cluster_n 个元素, 然后跨 cluster 归约
   - 使用 mbarrier 实现高效的跨 CTA 同步

2. 架构特化调优
   - SM80 (Ampere): 单 CTA 执行 (cluster_n=1)
   - SM90 (Hopper): 大 N 时启用 cluster 支持
   - SM100 (Blackwell): 与 SM90 相同

3. 向量化内存访问
   - 128-bit 向量化 load/store, 最大化访存吞吐
   - TiledCopy 抽象管理 gmem <-> smem <-> rmem 的数据搬运

Cluster 大小选择 (FP16):
------------------------
- N <= 16K: cluster_n = 1 (单 CTA)
- N <= 32K: cluster_n = 2
- N <= 64K: cluster_n = 4
- N <= 128K: cluster_n = 8
- 更大: cluster_n = 16

运行示例:

.. code-block:: bash

    python examples/python/CuTeDSL/blackwell/rmsnorm.py --M 2048 --N 4096 --dtype BFloat16
    python examples/python/CuTeDSL/blackwell/rmsnorm.py --M 2048 --N 4096 --dtype BFloat16 --benchmark
    python examples/python/CuTeDSL/blackwell/rmsnorm.py --M 2048 --N 32768 --dtype BFloat16 --benchmark

使用 NCU 性能分析器采集性能数据:

.. code-block:: bash

    ncu python examples/python/CuTeDSL/blackwell/rmsnorm.py --M 2048 --N 4096 --dtype BFloat16 --skip_ref_check
"""

# =============================================================================
# 架构检测
# =============================================================================


@functools.lru_cache(maxsize=16)
def get_sm_version(device: Optional[Union[int, torch.device, str]] = None) -> int:
    """获取 CUDA 设备的 SM (计算能力) 版本号."""
    if not torch.cuda.is_available():
        return 80  # 默认回退值
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def supports_cluster() -> bool:
    """检查当前设备是否支持 cluster 操作 (SM90+)."""
    return get_sm_version() >= 90


# =============================================================================
# 谓词工具
# =============================================================================


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """创建用于边界检查的谓词 tensor, 防止越界访问."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tXcX, mode=[0, 1]), cute.size(tXcX, mode=[1]), cute.size(tXcX, mode=[2])),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(tXcX[(0, rest_v), 0, rest_k][1], limit)
    return tXpX


# =============================================================================
# RMSNorm 配置类
# =============================================================================


class RMSNormConfig:
    """
    RMSNorm kernel 的配置.

    封装所有在初始化时一次性计算好的 kernel 参数, 遵循 CUTLASS 官方示例的 CuTe-DSL 惯例.
    """

    COPY_BITS = 128  # 128-bit 向量化加载

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        N: int,
        has_weight: bool = True,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.N = N
        self.has_weight = has_weight
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        # 128-bit 加载对应的向量大小 (元素个数)
        self.vec_size = self.COPY_BITS // dtype.width  # 128 / 16 = 8

        # 计算 cluster 大小 (仅 SM90+)
        self.cluster_n = self._compute_cluster_n(N, dtype, self.sm_version) # 1

        # cluster 模式下每个 CTA 负责的 N
        self.N_per_cta = N // self.cluster_n  # 4096

        # 线程配置
        self.threads_per_row = self._compute_threads_per_row(self.N_per_cta)  # 64
        self.num_threads = self._compute_num_threads(self.N_per_cta)  # 128

        # 派生值
        self.num_vec_blocks = max(
            1, (self.N_per_cta // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
        )  # max(1, (4096 // 8 + 64 - 1) // 64) = 8
        self.rows_per_block = self.num_threads // self.threads_per_row  # 2
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row  # 4096
        self.warps_per_row = max(self.threads_per_row // 32, 1)  # 2

    @staticmethod
    def _compute_cluster_n(N: int, dtype: type[cutlass.Numeric], sm_version: int) -> int:
        """根据 N 和架构计算最优 cluster 大小."""
        if sm_version < 90:
            return 1

        if dtype.width == 16:  # FP16/BF16
            if N <= 16 * 1024:
                return 1
            elif N <= 32 * 1024:
                return 2
            elif N <= 64 * 1024:
                return 4
            elif N <= 128 * 1024:
                return 8
            else:
                return 16
        else:  # FP32
            if N <= 32 * 1024:
                return 1
            elif N <= 64 * 1024:
                return 2
            elif N <= 128 * 1024:
                return 4
            elif N <= 256 * 1024:
                return 8
            else:
                return 16

    @staticmethod
    def _compute_threads_per_row(N_per_cta: int) -> int:
        """根据每个 CTA 的 N 计算每行最优线程数."""
        if N_per_cta <= 64:
            return 8
        elif N_per_cta <= 128:
            return 16
        elif N_per_cta <= 3072:
            return 32
        elif N_per_cta <= 6144:
            return 64
        elif N_per_cta <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(N_per_cta: int) -> int:
        """计算每个 block 的总线程数."""
        return 128 if N_per_cta <= 16384 else 256

    @staticmethod
    def _make_tv_layout(
        threads_per_row: int,
        rows_per_block: int,
        vec_size: int,
        num_vec_blocks: int,
    ) -> tuple:
        """创建 Thread-Value layout, 保证合并的向量化内存访问."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def smem_size_in_bytes(self) -> int:
        """计算共享内存需求 (字节)."""
        tile_bytes = self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
        reduction_bytes = self.rows_per_block * self.warps_per_row * self.cluster_n * 4
        mbar_bytes = 8 if self.cluster_n > 1 else 0
        return tile_bytes + reduction_bytes + mbar_bytes


# =============================================================================
# RMSNorm Kernel 类
# =============================================================================


class RMSNormKernel:
    """
    支持 cluster 同步的 RMSNorm kernel, 处理大 N.

    特性:
    - 基于 cluster 的归约, 适用于大 N (SM90+)
    - 多个 CTA 通过 mbarrier 协作
    - 单次归约 (平方和) + cluster 级聚合

    示例:
        >>> kernel = RMSNormKernel(cutlass.Float16, N=4096)
        >>> kernel(x_ptr, w_ptr, o_ptr, M, eps, stream)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        N: int,
        has_weight: bool = True,
        config: RMSNormConfig | None = None,
    ):
        # 使用提供的配置或创建新配置
        if config is not None:
            self.cfg = config
        else:
            self.cfg = RMSNormConfig(dtype, N, has_weight)

        # 暴露常用属性以方便访问
        self.dtype = self.cfg.dtype
        self.N = self.cfg.N
        self.has_weight = self.cfg.has_weight
        self.cluster_n = self.cfg.cluster_n

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer | None,
        o_ptr: cute.Pointer,
        M: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        """Host 端函数, 启动 RMSNorm kernel."""
        cfg = self.cfg

        # 从原始指针创建 CuTe tensor
        mX = cute.make_tensor(
            x_ptr,
            cute.make_layout((M, cfg.N), stride=(cfg.N, 1)),
        )
        mO = cute.make_tensor(
            o_ptr,
            cute.make_layout((M, cfg.N), stride=(cfg.N, 1)),
        )

        if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
            mW = cute.make_tensor(
                w_ptr,
                cute.make_layout((cfg.N,), stride=(1,)),
            )
        else:
            mW = None

        # 使用静态方法创建 TV layout
        tv_shape, tv_stride = RMSNormConfig._make_tv_layout(
            cfg.threads_per_row,
            cfg.rows_per_block,
            cfg.vec_size,
            cfg.num_vec_blocks,
        )  # ((64,2),(8,8)), ((16,1),(2,1024))
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (cfg.rows_per_block, cfg.cols_per_tile)  # (2, 4096)

        self.kernel(mX, mW, mO, eps, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, cfg.rows_per_block), cfg.cluster_n, 1],
            block=[cfg.num_threads, 1, 1],
            cluster=[1, cfg.cluster_n, 1] if cutlass.const_expr(cfg.cluster_n > 1) else None,
            smem=cfg.smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mO: cute.Tensor,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        """Device kernel, 实现支持 cluster 的 RMSNorm."""
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if cutlass.const_expr(cfg.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]  # 多 CTA 时, 读取当前 CTA 在 cluster 内的编号
        else:
            cluster_y = cutlass.const_expr(0)  # 单 CTA 时, cluster_y 为 0

        M = mX.shape[0]
        threads_per_row = tv_layout.shape[0][0]
        warps_per_row = max(threads_per_row // 32, 1)
        rows_per_block = tiler_mn[0]

        # =====================================================================
        # 分配共享内存
        # =====================================================================
        smem = utils.SmemAllocator()

        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        if cutlass.const_expr(cfg.cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cfg.cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(Int64, num_elems=1)

        # =====================================================================
        # 初始化 cluster
        # =====================================================================
        if cutlass.const_expr(cfg.cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # =====================================================================
        # 创建 identity tensor 并划分 tile
        # =====================================================================
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))  # (2,4096):(4096,1)
        gO = cute.local_tile(mO, tiler_mn, (bidx, cluster_y))  # (2,4096):(4096,1)
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))  # (2,4096):(1@0,1@1)

        # 把 1D 的 weight 向量广播成 2D tensor, 使其能和输入 X 做相同的 tile/partition 操作
        if cutlass.const_expr(cfg.has_weight and mW is not None):
            # (N,):(1,) -> (rows_per_block,):(0,)
            # stride=0 意味着沿行方向移动时地址不变, 所有行看到的都是同一份 weight 数据。
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
            gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))
            if tidx == 0:
                cute.printf(gW)

        # =====================================================================
        # 创建 TiledCopy 操作
        # =====================================================================
        copy_atom_load_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )

        copy_atom_load_W = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )

        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
        tiled_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # 对 tensor 进行 partition (按线程切分)
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)

        # 寄存器片段 (存放从 smem 加载到寄存器的数据)
        tXrX = cute.make_fragment_like(tXgX)
        tXrO = cute.make_fragment_like(tXgO)

        if cutlass.const_expr(cfg.has_weight and mW is not None):
            tWgW = thr_copy_W.partition_S(gW)
            tWrW = cute.make_fragment_like(tWgW)
            tXrW = thr_copy_X.retile(tWrW)

        # =====================================================================
        # 边界检查
        # =====================================================================
        tXpX = predicate_k(tXcX, limit=cfg.N)

        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # =====================================================================
        # 异步拷贝: global -> shared
        # =====================================================================
        if row_in_bounds:
            cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)

        cute.arch.cp_async_commit_group()

        # 等待异步拷贝期间同步加载 weight (延迟隐藏)
        if cutlass.const_expr(cfg.has_weight and mW is not None):
            tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=cfg.N)
            cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)

        cute.arch.cp_async_wait_group(0)

        # =====================================================================
        # Pass 1: 计算平方和, 通过 cluster 归约
        # =====================================================================
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)
        if tidx == 0 and bidx == 0:
            cute.print_tensor(x)

        x_sq = x * x
        sum_sq = row_reduce(
            x_sq,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            mbar_ptr,
            cfg.cluster_n,
            Float32(0.0),
        )

        # rstd = 1 / sqrt(mean(x²) + eps)
        mean_sq = sum_sq / cfg.N
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # 归约后同步 (确保 reduction_buffer 可被安全复用)
        if cutlass.const_expr(cfg.cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # =====================================================================
        # Pass 2: 归一化并输出
        # =====================================================================
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        y = x * rstd

        # 若有 weight 则乘以缩放权重
        if cutlass.const_expr(cfg.has_weight and mW is not None):
            w = tXrW.load().to(Float32)
            y = y * w

        # 写回 global memory
        tXrO.store(y.to(cfg.dtype))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)


# =============================================================================
# Kernel 编译与缓存
# =============================================================================

# torch dtype -> cutlass dtype 映射
_torch_to_cutlass_dtype = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

# 已编译 kernel 缓存
_compile_cache: dict = {}


def get_compiled_kernel(
    dtype: type[cutlass.Numeric],
    N: int,
    has_weight: bool,
    stream: cuda.CUstream,
):
    """
    获取或编译指定配置的 RMSNorm kernel.

    使用编译缓存, 对相同的 (dtype, N, has_weight) 组合避免重复编译.

    :param dtype: 数据类型 (Float16, BFloat16, Float32)
    :param N: 隐藏维度大小
    :param has_weight: 是否应用 weight
    :param stream: CUDA stream
    :return: 已编译的 kernel 函数
    """
    key = (dtype, N, has_weight)
    if key not in _compile_cache:
        kernel_obj = RMSNormKernel(dtype, N, has_weight)

        # 使用代表性参数进行编译 (实际值在运行时传入)
        compiled_kernel = cute.compile(
            kernel_obj,
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),  # x_ptr
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
            if has_weight
            else None,  # w_ptr
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),  # o_ptr
            Int32(1),  # M (dummy)
            Float32(1e-6),  # eps (dummy)
            stream,
        )

        _compile_cache[key] = compiled_kernel

    return _compile_cache[key]


# =============================================================================
# Tensor Creation Utilities
# =============================================================================


def create_tensors(
    M: int,
    N: int,
    dtype: type[cutlass.Numeric],
    has_weight: bool,
) -> Tuple:
    """创建 RMSNorm 的输入、权重和输出 tensor."""
    torch.manual_seed(42)
    torch_dtype = cutlass_torch.dtype(dtype)

    x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
    weight = torch.randn(N, device="cuda", dtype=torch_dtype) if has_weight else None
    out = torch.empty_like(x)

    return x, weight, out


def rmsnorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PyTorch 参考实现的 RMSNorm, 用于正确性验证."""
    x_f32 = x.float()
    rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + eps)
    x_norm = x_f32 / rms
    if weight is not None:
        x_norm = x_norm * weight.float()
    return x_norm.to(x.dtype)


# =============================================================================
# Run Function
# =============================================================================


def run(
    M: int,
    N: int,
    dtype: type[cutlass.Numeric],
    has_weight: bool = True,
    eps: float = 1e-6,
    tolerance: float = 1e-2,
    warmup_iterations: int = 2,
    iterations: int = 100,
    skip_ref_check: bool = False,
    benchmark: bool = False,
) -> float:
    """
    执行 RMSNorm 并可选地进行性能基准测试.

    :param M: 行数 (batch_size * seq_len)
    :param N: 隐藏维度大小
    :param dtype: 数据类型 (Float16, BFloat16, Float32)
    :param has_weight: 是否应用可学习权重
    :param eps: 数值稳定性的 epsilon
    :param tolerance: 正确性检查的容差
    :param warmup_iterations: 基准测试的预热迭代次数
    :param iterations: 基准测试的迭代次数
    :param skip_ref_check: 跳过参考实现的正确性检查
    :param benchmark: 启用基准测试
    :return: 执行时间 (微秒), benchmark=False 时返回 0
    """
    print("Running RMSNorm test with:")
    print(f"  M: {M}, N: {N}")
    print(f"  dtype: {dtype}")
    print(f"  has_weight: {has_weight}")
    print(f"  eps: {eps}")
    print(f"  SM version: {get_sm_version()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this example!")
    # 获取 CUDA stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # 创建 tensor
    x, weight, out = create_tensors(M, N, dtype, has_weight)

    # 获取配置信息
    config = RMSNormConfig(dtype, N, has_weight)
    print(f"  cluster_n: {config.cluster_n}")
    print(f"  threads_per_row: {config.threads_per_row}")
    print(f"  rows_per_block: {config.rows_per_block}")

    # 获取已编译 kernel
    compiled_kernel = get_compiled_kernel(dtype, N, has_weight, stream)

    # 创建 kernel 调用所需的指针
    x_ptr = make_ptr(dtype, x.data_ptr())
    w_ptr = make_ptr(dtype, weight.data_ptr()) if weight is not None else None
    out_ptr = make_ptr(dtype, out.data_ptr())

    # 运行 kernel 并验证正确性
    if not skip_ref_check:
        compiled_kernel(x_ptr, w_ptr, out_ptr, Int32(M), Float32(eps), stream)
        torch.cuda.synchronize()

        ref = rmsnorm_ref(x, weight, eps)
        torch.testing.assert_close(out, ref, atol=tolerance, rtol=tolerance)
        print("Correctness check passed!")

    if not benchmark:
        return 0.0

    # Benchmark
    print(f"\nBenchmarking with {warmup_iterations} warmup, {iterations} iterations...")

    def generate_tensors():
        x, weight, out = create_tensors(M, N, dtype, has_weight)
        x_ptr = make_ptr(dtype, x.data_ptr())
        w_ptr = make_ptr(dtype, weight.data_ptr()) if weight is not None else None
        out_ptr = make_ptr(dtype, out.data_ptr())
        return testing.JitArguments(x_ptr, w_ptr, out_ptr, Int32(M), Float32(eps), stream)

    exec_time_us = testing.benchmark(
        compiled_kernel,
        workspace_generator=generate_tensors,
        workspace_count=10,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        stream=stream,
    )

    # 计算吞吐量
    torch_dtype = cutlass_torch.dtype(dtype)
    bytes_per_elem = torch.tensor([], dtype=torch_dtype).element_size()
    total_bytes = M * N * bytes_per_elem * 2  # 读 x + 写 out
    if has_weight:
        total_bytes += N * bytes_per_elem  # 读 weight (均摊到 M 行)

    throughput_gbps = (total_bytes / (exec_time_us / 1e6)) / 1e9

    print(f"Kernel execution time: {exec_time_us:.2f} us")
    print(f"Memory throughput: {throughput_gbps:.2f} GB/s")

    return exec_time_us


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RMSNorm kernel example for Blackwell (SM100)"
    )

    parser.add_argument("--M", type=int, default=2048, help="Number of rows")
    parser.add_argument("--N", type=int, default=4096, help="Hidden dimension size")
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
        help="Data type (Float16, BFloat16, Float32)",
    )
    parser.add_argument(
        "--has_weight",
        action="store_true",
        default=True,
        help="Apply learnable weight",
    )
    parser.add_argument(
        "--no_weight",
        action="store_true",
        help="Disable learnable weight",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Epsilon for numerical stability",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="Tolerance for correctness check",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=2,
        help="Warmup iterations for benchmarking",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference correctness check",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmarking",
    )

    args = parser.parse_args()

    # 处理 weight 开关
    has_weight = args.has_weight and not args.no_weight

    run(
        M=args.M,
        N=args.N,
        dtype=args.dtype,
        has_weight=has_weight,
        eps=args.eps,
        tolerance=args.tolerance,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
    )

    print("PASS")

