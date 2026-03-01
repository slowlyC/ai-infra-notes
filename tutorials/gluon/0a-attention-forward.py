import copy
import math
import torch
import triton
import pytest
import itertools

from triton.language.core import _aggregate as aggregate

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tensor_memory_descriptor,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
    float2,
)
from triton.experimental.gluon.language.nvidia.blackwell.float2 import Float2Tensor


# ===-----------------------------------------------------------------------===#
# Layout Utilities
# ===-----------------------------------------------------------------------===#

@gluon.constexpr_function
def get_mma_instr_shape(shape, element_ty):
    """
    根据矩阵形状和数据类型, 确定 MMA 指令的 tile 形状。
    TensorMemoryLayout 是 2D 的, 所以 k 维度不会被用到, 当调用 tcgen05_mma 时, K 由硬件和输入数据的布局自动处理 
    
    Args:
        shape: 矩阵形状 [M, N]
        element_ty: 数据类型 (决定 K 的大小)
    
    Returns:
        (m, n, k): MMA 指令的 tile 形状
    """
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    # fp32: 256/32 = 8
    # bf16: 256/16 = 16
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


# ===-----------------------------------------------------------------------===#
# Data Abstractions
# ===-----------------------------------------------------------------------===#

# 管理循环缓冲区的索引和 mbarrier 的 phase
@aggregate
class BarrierCounter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def increment(self):
        if self.num_barriers == 1:
            return BarrierCounter(gl.to_tensor(0), self.phase ^ 1, self.num_barriers)
        next_index = self.index + 1
        rollover = next_index == self.num_barriers  # 到头了
        index = gl.where(rollover, 0, next_index)   # index 回到 0
        phase = gl.where(rollover, self.phase ^ 1, self.phase)  # 翻转 phase (0→1 或 1→0)
        return BarrierCounter(index, phase, self.num_barriers)


def Channel(T, alloc_fn):
    @aggregate
    class ChannelType:
        mem: T  # 数据缓冲区 (SMEM 或 TMEM), shape = [num_buffers, ...]   
        ready_bars: gl.shared_memory_descriptor  # "数据已就绪" 信号 (Producer → Consumer) 
        empty_bars: gl.shared_memory_descriptor  # "缓冲区已空" 信号 (Consumer → Producer)
        num_buffers: gl.constexpr  # 缓冲区数量 (流水线深度)
        num_consumers: gl.constexpr  # 消费者数量 (empty_bar 需要等多少个 arrive) 

        @gluon.constexpr_function
        def __init__(self, mem, ready_bars, empty_bars, num_buffers, num_consumers):
            self.mem = mem
            self.ready_bars = ready_bars
            self.empty_bars = empty_bars
            self.num_buffers = gl.constexpr(num_buffers)
            self.num_consumers = gl.constexpr(num_consumers)

        @gluon.jit
        def alloc(shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr, num_buffers: gl.constexpr,
                  num_consumers: gl.constexpr = 1):
            mem = alloc_fn(dtype, [num_buffers] + shape, layout)
            ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            for i in gl.static_range(num_buffers):
                mbarrier.init(ready_bars.index(i), count=1)
                mbarrier.init(empty_bars.index(i), count=num_consumers)
                mbarrier.arrive(empty_bars.index(i), count=num_consumers)
            return ChannelType(mem, ready_bars, empty_bars, num_buffers, num_consumers)

        @gluon.jit
        def acquire_producer(self, counter):
            index, phase = counter.index, counter.phase
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(empty_bar, phase)
            return mem, ready_bar

        @gluon.jit
        def acquire_consumer(self, counter):
            index, phase = counter.index, counter.phase
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(ready_bar, phase)
            return mem, empty_bar

        @gluon.jit
        def create_counter(self):
            return BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), self.num_buffers)

        @gluon.jit
        def create_producer(self):
            return Producer(self, self.create_counter())

        @gluon.jit
        def create_consumer(self):
            return Consumer(self, self.create_counter())

        @gluon.jit
        def release(self):
            if isinstance(self.mem, gl.shared_memory_descriptor):
                self.mem._keep_alive()
            for i in gl.static_range(self.num_buffers):
                mbarrier.invalidate(self.ready_bars.index(i))
                mbarrier.invalidate(self.empty_bars.index(i))

    @aggregate
    class Producer:
        channel: ChannelType
        counter: BarrierCounter

        @gluon.constexpr_function
        def __init__(self, channel, counter):
            self.channel = channel
            self.counter = counter

        @gluon.jit
        def acquire(self):
            mem, ready_bar = self.channel.acquire_producer(self.counter)
            next = Producer(self.channel, self.counter.increment())
            return mem, ready_bar, next

    @aggregate
    class Consumer:
        channel: ChannelType
        counter: BarrierCounter

        @gluon.constexpr_function
        def __init__(self, channel, counter):
            self.channel = channel
            self.counter = counter

        @gluon.jit
        def acquire(self):
            mem, empty_bar = self.channel.acquire_consumer(self.counter)
            next = Consumer(self.channel, self.counter.increment())
            return mem, empty_bar, next

    return ChannelType, Producer, Consumer


SharedMemoryChannel, SharedMemoryProducer, SharedMemoryConsumer = Channel(gl.shared_memory_descriptor,
                                                                          gl.allocate_shared_memory)
TensorMemoryChannel, TensorMemoryProducer, TensorMemoryConsumer = Channel(tensor_memory_descriptor,
                                                                          allocate_tensor_memory)


@gluon.jit
def get_desc_channel(desc, num_buffers: gl.constexpr, num_consumers: gl.constexpr = 1):
    shape: gl.constexpr = desc.block_type.shape
    layout: gl.constexpr = desc.layout
    return SharedMemoryChannel.alloc(shape, desc.dtype, layout, num_buffers, num_consumers)


@gluon.jit
def issue_async_tma_load(smem, bar, desc, offset):
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_copy_global_to_shared(desc, [offset, 0], bar, smem)


# ===-----------------------------------------------------------------------===#
# Gluon Attention
# ===-----------------------------------------------------------------------===#

@aggregate
class AttentionConfig:
    qk_scale: gl.tensor
    Z: gl.tensor
    H: gl.tensor
    N_CTX: gl.tensor

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    HEAD_DIM: gl.constexpr
    GROUP_SIZE_N: gl.constexpr
    NUM_SMS: gl.constexpr
    dtype: gl.constexpr
    num_warps: gl.constexpr

    # 分割因子 (减少寄存器压力)
    SPLIT_D_FACTOR: gl.constexpr
    SPLIT_EXP_FACTOR: gl.constexpr
    SPLIT_QK_LOAD_FACTOR: gl.constexpr
    SPLIT_M: gl.constexpr
    SPLIT_D: gl.constexpr

    q_shape: gl.constexpr
    k_shape: gl.constexpr
    v_shape: gl.constexpr
    qk_shape: gl.constexpr
    o_shape: gl.constexpr

    qk_tmem_layout: gl.constexpr
    o_tmem_layout: gl.constexpr
    p_tmem_layout: gl.constexpr

    qk_layout: gl.constexpr
    o_splitn_layout: gl.constexpr
    alpha_2d_layout: gl.constexpr

    num_kv_buffers: gl.constexpr
    use_exp2_turnstile: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, qk_scale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, GROUP_SIZE_N, NUM_SMS, STAGE, dtype,
                 num_warps):
        self.qk_scale = qk_scale
        self.Z = Z
        self.H = H
        self.N_CTX = N_CTX

        self.BLOCK_M = gl.constexpr(BLOCK_M)  # 256
        self.BLOCK_N = gl.constexpr(BLOCK_N)  # 128
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.GROUP_SIZE_N = gl.constexpr(GROUP_SIZE_N)  # 4 if causal else 1
        self.NUM_SMS = gl.constexpr(NUM_SMS)  # 148
        self.dtype = gl.constexpr(dtype)
        self.num_warps = gl.constexpr(num_warps)

        self.SPLIT_D_FACTOR = gl.constexpr(2)
        self.SPLIT_EXP_FACTOR = gl.constexpr(256 // HEAD_DIM)
        self.SPLIT_QK_LOAD_FACTOR = gl.constexpr(2 if STAGE == 1 else 1)
        self.SPLIT_M = gl.constexpr(self.BLOCK_M // 2)
        self.SPLIT_D = gl.constexpr(self.HEAD_DIM // self.SPLIT_D_FACTOR)

        self.q_shape = gl.constexpr([self.SPLIT_M, self.HEAD_DIM])  # [128, 128]
        self.k_shape = gl.constexpr([self.BLOCK_N, self.HEAD_DIM])  # [128, 128]
        self.qk_shape = gl.constexpr([self.SPLIT_M, self.BLOCK_N])  # [128, 128]
        self.v_shape = gl.constexpr([self.BLOCK_N, self.HEAD_DIM])  # [128, 128]
        self.o_shape = gl.constexpr([self.SPLIT_M, self.HEAD_DIM])  # [128, 128]

        qk_instr_shape = get_mma_instr_shape(self.qk_shape, gl.float32)  # (128, 128, 8) 
        o_instr_shape = get_mma_instr_shape(self.o_shape, gl.float32)  # (128, 128, 8) 
        self.qk_tmem_layout = gl.constexpr(TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), col_stride=1)) # (128, 128) 
        self.o_tmem_layout = gl.constexpr(TensorMemoryLayout((o_instr_shape[0], o_instr_shape[1]), col_stride=1)) # (128, 128) 
        self.p_tmem_layout = gl.constexpr(TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), col_stride=1)) # (128, 128) 
        o_splitn_tmem_layout: gl.constexpr = TensorMemoryLayout(
            (o_instr_shape[0], o_instr_shape[1] // self.SPLIT_D_FACTOR), col_stride=1)

        self.qk_layout = gl.constexpr(
            get_tmem_reg_layout(gl.float32, self.qk_shape, self.qk_tmem_layout, self.num_warps,
                                instr_variant="32x32b_splitn"))
        self.o_splitn_layout = gl.constexpr(
            get_tmem_reg_layout(gl.float32, (self.o_shape[0], self.o_shape[1] // self.SPLIT_D_FACTOR),
                                o_splitn_tmem_layout, self.num_warps))
        self.alpha_2d_layout = gl.constexpr(gl.BlockedLayout([1, 1], [32, 1], [self.num_warps, 1], [0, 1]))

        is_fp16 = self.dtype.value in [gl.float16, gl.bfloat16]
        if is_fp16:
            self.num_kv_buffers = gl.constexpr(3 if HEAD_DIM == 128 else 6)
        else:
            self.num_kv_buffers = gl.constexpr(4 if HEAD_DIM == 128 else 8)

        self.use_exp2_turnstile = gl.constexpr(HEAD_DIM == 64)

    @gluon.jit
    def get_program(self, pid_m, pid_n):
        start_m = pid_m
        off_hz = pid_n
        off_z = off_hz // self.H
        off_h = off_hz % self.H

        offset_y = off_z * (self.N_CTX * self.H) + off_h * self.N_CTX
        qo_offset_y = offset_y + start_m * self.BLOCK_M

        return AttentionProgram(self, start_m, off_hz, offset_y, qo_offset_y)


@aggregate
class ProgramScheduler:
    config: AttentionConfig
    start_pid: gl.tensor
    num_pid_n: gl.tensor
    num_pid_in_group: gl.tensor
    num_tiles: gl.tensor

    @gluon.constexpr_function
    def __init__(self, config, start_pid, num_pid_n, num_pid_in_group, num_tiles):
        self.config = config
        self.start_pid = start_pid
        self.num_pid_n = num_pid_n
        self.num_pid_in_group = num_pid_in_group
        self.num_tiles = num_tiles

    @gluon.jit
    def create(config):
        start_pid = gl.program_id(0)
        num_pid_m = gl.cdiv(config.N_CTX, config.BLOCK_M)
        num_pid_n = config.Z * config.H
        num_pid_in_group = num_pid_m * config.GROUP_SIZE_N
        num_tiles = num_pid_m * num_pid_n  # 总 tile 数: (seq_len 方向的 tile 数) × (batch × heads) 
        return ProgramScheduler(config, start_pid, num_pid_n, num_pid_in_group, num_tiles)

    @gluon.jit
    def get_program(self, tile_id):
        # Swizzle 调度
        # 普通调度 (row-major):            Swizzle 调度 (更好的 L2 局部性):
        #  0  1  2  3                     0  1  4  5
        #  4  5  6  7                     2  3  6  7
        #  8  9 ... ...                   8  9 ... ...
        #                                 GROUP_SIZE_N = 2
        #                                 同一组内的 tile 共享 K/V cache
        group_id = tile_id // self.num_pid_in_group
        first_pid_n = group_id * self.config.GROUP_SIZE_N
        group_size_n = min(self.num_pid_n - first_pid_n, self.config.GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_m = (tile_id % self.num_pid_in_group) // group_size_n
        return self.config.get_program(pid_m, pid_n)


# 单个 Tile 的参数配置
@aggregate
class AttentionProgram:
    config: AttentionConfig
    start_m: gl.tensor
    off_hz: gl.tensor
    offset_y: gl.tensor
    qo_offset_y: gl.tensor

    @gluon.constexpr_function
    def __init__(self, config, start_m, off_hz, offset_y, qo_offset_y):
        self.config = config
        self.start_m = start_m  # pid_m, Q 的起始行 tile 索引
        self.off_hz = off_hz  # batch_idx × H + head_idx, batch-head 的线性索引
        self.offset_y = offset_y  # off_z × (N_CTX × H) + off_h × N_CTX, K/V 在 global mem 的 offset
        self.qo_offset_y = qo_offset_y  # offset_y + start_m × BLOCK_M, Q/O 在 global mem 的 offset

    @gluon.jit
    def get_fused_loop_bounds(self, STAGE: gl.constexpr):
        # 在 _attn_fwd_load, _attn_fwd_mma, _attn_fwd_correction 中调用, 用于确定 K/V 的加载范围
        BLOCK_M: gl.constexpr = self.config.BLOCK_M
        # stage = 3 if causal else 1
        if STAGE == 1:  # 全范围 (causal = False)
            return 0, self.config.N_CTX
        elif STAGE == 2:  # 只处理对角块
            return self.start_m * BLOCK_M, (self.start_m + 1) * BLOCK_M
        elif STAGE == 3:  # 从 0 到对角块结束 (causal = True)  
            return 0, (self.start_m + 1) * BLOCK_M
        else:
            return 0, 0

    @gluon.jit
    def get_loop_bounds(self, STAGE: gl.constexpr):
        # 在 _softmax_inner_loop 中调用, 用于确定 QK 的加载范围
        # if STAGE & 1: _softmax_inner_loop(STAGE=4-STAGE)
        # if STAGE & 2: _softmax_inner_loop(STAGE=2)
        BLOCK_M: gl.constexpr = self.config.BLOCK_M
        if STAGE == 1:  # 返回非 causal 部分 
            lo, hi = 0, self.start_m * BLOCK_M
        elif STAGE == 2:  # 返回对角块
            lo, hi = self.start_m * BLOCK_M, (self.start_m + 1) * BLOCK_M
        else:
            lo, hi = 0, self.config.N_CTX  # 返回全范围
        return lo, hi


# ===-----------------------------------------------------------------------===#
# _gluon_attn
# ===-----------------------------------------------------------------------===#

@gluon.jit
def _borrow_s_as_p(config, s_tmem):
    p_tmem = s_tmem.slice(0, config.BLOCK_N // 2)
    return p_tmem._reinterpret(config.dtype, config.qk_shape, config.p_tmem_layout)


@gluon.jit
def _borrow_s_as_alpha(config, s_tmem):
    alpha_tmem = s_tmem.slice(config.BLOCK_N // 2, 1)
    alpha_layout: gl.constexpr = TensorMemoryLayout([config.SPLIT_M, 1], col_stride=1)
    return alpha_tmem._reinterpret(gl.float32, [config.SPLIT_M, 1], alpha_layout)


@gluon.jit
def _borrow_s_for_epilogue(config, s_tmem):
    m_i_tmem = s_tmem.slice(config.BLOCK_N // 2 + 1, 1)
    l_i_tmem = s_tmem.slice(config.BLOCK_N // 2 + 2, 1)
    layout: gl.constexpr = TensorMemoryLayout([config.SPLIT_M, 1], col_stride=1)
    m_i_tmem = m_i_tmem._reinterpret(gl.float32, [config.SPLIT_M, 1], layout)
    l_i_tmem = l_i_tmem._reinterpret(gl.float32, [config.SPLIT_M, 1], layout)
    return m_i_tmem, l_i_tmem


@gluon.constexpr_function
def _get_split_n_layout(layout: gl.constexpr, SPLIT_FACTOR: gl.constexpr = 2):
    """
    调整 Layout 以支持 split_n 操作(将 tensor 沿 N 维度分成两半)。
    目标: 让 tensor 的最后一个 register basis 是 [0, shape[1]//2] 
    例如 shape = [128, 128], target = [0, 128 // 2] = [0, 64] 
    """
    assert isinstance(layout, gl.DistributedLinearLayout), "split_n requires a distributed layout"
    assert SPLIT_FACTOR == 1 or SPLIT_FACTOR == 2, "split_n requires a split factor of 1 or 2"
    if SPLIT_FACTOR == 1:
        return layout
    else:
        target = [0, layout.shape[1] // 2]  # [0, 2^{m-1}]
        last_reg_idx = len(layout.reg_bases) - 1
        reg_last = layout.reg_bases[last_reg_idx]

        if reg_last == target:
            return layout

        ret = copy.deepcopy(layout)

        # 在列表中查找 [0, 2^{m-1}] 并将其与最后一个寄存器交换
        # Layout 的 basis 解释: 
        #    DistributedLinearLayout 用 basis 向量描述数据分布:                                   
        #    - reg_bases:   寄存器内的数据位置                                                    
        #    - lane_bases:  32 个 lane 的数据位置                                                 
        #    - warp_bases:  warp 间的数据位置                                                     
        #  basis [0, 64] 表示这个维度跨越 N 方向的 64 个元素
        #      N 方向: 0        64       128
        #              |        |         |
        #              |─上半部 ─|─ 下半部 ─|
        #              | reg=0  |  reg=1  |
        #  为什么要把 [0, 64] 放到最后一个 reg_basis?                                              
        #  这样 _split_n 操作可以简单地通过 reshape 来分割:                                    
        #  tensor[128, 128] → tensor[128, 2, 64] → 取 [:, 0, :] 和 [:, 1, :]  
        for L in (ret.reg_bases, ret.lane_bases, ret.warp_bases, ret.block_bases):
            for i, b in enumerate(L):
                if b == target:
                    L[i], ret.reg_bases[last_reg_idx] = reg_last, target
                    return ret
        assert False, f"split_n requires having a basis {target}. Got\n{layout}"

   
@gluon.jit
def _split_n(x, SPLIT_FACTOR: gl.constexpr = 2):
    """
    将 tensor 沿着 N 维度(列) 平均分割成多个部分
    输入: x [M, N] = [4, 8], SPLIT_FACTOR = 2
      a0 a1 a2 a3 | a4 a5 a6 a7
      b0 b1 b2 b3 | b4 b5 b6 b7
      c0 c1 c2 c3 | c4 c5 c6 c7
      d0 d1 d2 d3 | d4 d5 d6 d7
       前半部分        后半部分
    """
    if SPLIT_FACTOR == 1:
        return (x, )
    else:
        layout: gl.constexpr = _get_split_n_layout(x.type.layout)
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        x0 = gl.convert_layout(x0, layout, assert_trivial=True)
        x1 = gl.convert_layout(x1, layout, assert_trivial=True)
        return _split_n(x0, SPLIT_FACTOR // 2) + _split_n(x1, SPLIT_FACTOR // 2)


@gluon.constexpr_function
def _get_join_n_layout(layout, SPLIT_FACTOR: gl.constexpr = 2):
    assert isinstance(layout, gl.DistributedLinearLayout), "join_n requires a Linear layout"
    shape = list(layout.shape)
    regs = [[0, shape[1] * (1 << i)] for i in range(int(math.log2(SPLIT_FACTOR)))]
    shape[1] *= SPLIT_FACTOR
    return gl.DistributedLinearLayout(
        layout.reg_bases + regs,
        layout.lane_bases,
        layout.warp_bases,
        layout.block_bases,
        shape,
    )


@gluon.jit
def _join_n(xs):
    """
    _split_n 的逆操作, 将多个分割的 tensor 沿 N 方向合并。
    """
    if len(xs) == 1:
        return xs[0]
    else:
        x0 = _join_n(xs[:len(xs) // 2])
        x1 = _join_n(xs[len(xs) // 2:])
        layout: gl.constexpr = _get_join_n_layout(x0.type.layout)
        x = gl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] * 2])
        return gl.convert_layout(x, layout, assert_trivial=True)


@gluon.jit
def _attn_fwd_load(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    q_producer = q_chnl.create_producer()
    kv_producer = kv_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_fused_loop_bounds(STAGE)

        q0_offset = prog.qo_offset_y + config.SPLIT_M * 0
        q0_smem, q0_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(q0_smem, q0_bar, desc_q, q0_offset)

        offsetkv_y = prog.offset_y + lo
        k_smem, k_bar, kv_producer = kv_producer.acquire()
        issue_async_tma_load(k_smem, k_bar, desc_k, offsetkv_y)

        q1_offset = prog.qo_offset_y + config.SPLIT_M * 1
        q1_smem, q1_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(q1_smem, q1_bar, desc_q, q1_offset)

        v_smem, v_bar, kv_producer = kv_producer.acquire()
        issue_async_tma_load(v_smem, v_bar, desc_v, offsetkv_y)

        for start_n in range(lo + config.BLOCK_N, hi, config.BLOCK_N):
            offsetkv_y = prog.offset_y + start_n
            k_smem, k_bar, kv_producer = kv_producer.acquire()
            issue_async_tma_load(k_smem, k_bar, desc_k, offsetkv_y)
            v_smem, v_bar, kv_producer = kv_producer.acquire()
            issue_async_tma_load(v_smem, v_bar, desc_v, offsetkv_y)

# _attn_fwd_mma流程:
# ----------------------
# # Prologue: 第一个 K/V 块
# S0 = Q0 @ K[0].T
# S1 = Q1 @ K[0].T
# # (等待 Softmax0 把 S0 → P0)
# O0 = P0 @ V[0]          # 计算O0, 第一次不累加
# # Main Loop: 剩余 K/V 块
# for i in range(1, num_mmas):
#     S0 = Q0 @ K[i].T
#     # 用上一轮的 V 和刚计算好的 P1
#     O1 += P1 @ V[i-1]   # 计算O1,累加 (第一次时不累加)
#     S1 = Q1 @ K[i].T
#     # 用当前的 V 和 P0
#     O0 += P0 @ V[i]     # 计算O0,累加
# # Epilogue: 最后一次 P1 @ V
# O1 += P1 @ V[num_mmas-1]  # 计算O1,累加
@gluon.jit
def _attn_fwd_mma(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    q_consumer = q_chnl.create_consumer()
    kv_consumer = kv_chnl.create_consumer()
    o_producer = o_chnl.create_producer()

    s0_producer = s0_chnl.create_producer()
    s1_producer = s1_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_fused_loop_bounds(STAGE)
        num_mmas = (hi - lo) // config.BLOCK_N

        q0_smem, q0_bar, q_consumer = q_consumer.acquire()
        k_smem, k_bar, kv_consumer = kv_consumer.acquire()
        s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
        tcgen05_mma(q0_smem, k_smem.permute((1, 0)), s0_tmem, use_acc=False, mbarriers=[s0_bar])

        q1_smem, q1_bar, q_consumer = q_consumer.acquire()
        s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
        tcgen05_mma(q1_smem, k_smem.permute((1, 0)), s1_tmem, use_acc=False, mbarriers=[s1_bar, k_bar])

        v_smem, v_bar, kv_consumer = kv_consumer.acquire()
        o0_tmem, o0_bar, o_producer = o_producer.acquire()
        s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
        p0_tmem = _borrow_s_as_p(config, s0_tmem)
        tcgen05_mma(p0_tmem, v_smem, o0_tmem, use_acc=False, mbarriers=[o0_bar])
        o1_init = False

        for _ in range(num_mmas - 1):
            k_smem, k_bar, kv_consumer = kv_consumer.acquire()
            tcgen05_mma(q0_smem, k_smem.permute((1, 0)), s0_tmem, use_acc=False, mbarriers=[s0_bar])

            o1_tmem, o1_bar, o_producer = o_producer.acquire()
            s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
            p1_tmem = _borrow_s_as_p(config, s1_tmem)
            tcgen05_mma(p1_tmem, v_smem, o1_tmem, use_acc=o1_init, mbarriers=[o1_bar, v_bar])
            o1_init = True

            tcgen05_mma(q1_smem, k_smem.permute((1, 0)), s1_tmem, use_acc=False, mbarriers=[s1_bar, k_bar])

            v_smem, v_bar, kv_consumer = kv_consumer.acquire()
            o0_tmem, o0_bar, o_producer = o_producer.acquire()
            s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
            p0_tmem = _borrow_s_as_p(config, s0_tmem)
            tcgen05_mma(p0_tmem, v_smem, o0_tmem, mbarriers=[o0_bar])

        tcgen05_commit(q0_bar)
        tcgen05_commit(q1_bar)

        o1_tmem, o1_bar, o_producer = o_producer.acquire()
        s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
        p1_tmem = _borrow_s_as_p(config, s1_tmem)
        tcgen05_mma(p1_tmem, v_smem, o1_tmem, use_acc=o1_init, mbarriers=[o1_bar, v_bar, s0_bar, s1_bar])

# 位掩码构造示例
# 假设21行 col_limit_right = 21 (可以看到前 21 列) 
# 对于第一组 (s=0, 列 0-15):
# ────────────────────────────
# col_lim_right_s = 21 - 0 = 21, col_lim_right_cur = max(21,0) = 21
# mask = -1 << 21  # mask 左移超过 16 位, 低 16 位全为 0, 所以组内全部可见
#
# 对于第二组 (s=16, 列 16-31):
# ────────────────────────────
# col_lim_right_s = 21 - 16 = 5, col_lim_right_cur = max(5,0) = 5
# mask = -1 << 5 = 0xFFFFFFE0
#      = 1111 1111 1111 1111 1111 1111 1110 0000 (二进制)  # 低 5 位是 0, 表示可见                                   
#
# 之后检查第二组内每个位置的 i (0-15):
#   i=0: mask & (1<<0) = 0xFFE0 & 0x0001 = 0 → mask_i_bit=True  → 保留 (可见)
#   i=1: mask & (1<<1) = 0xFFE0 & 0x0002 = 0 → mask_i_bit=True  → 保留 (可见)
#   i=2: mask & (1<<2) = 0xFFE0 & 0x0004 = 0 → mask_i_bit=True  → 保留 (可见)
#   i=3: mask & (1<<3) = 0xFFE0 & 0x0008 = 0 → mask_i_bit=True  → 保留 (可见)
#   i=4: mask & (1<<4) = 0xFFE0 & 0x0010 = 0 → mask_i_bit=True  → 保留 (可见)
#   ──────────────────────────────────────────────────────────────────────────
#   i=5: mask & (1<<5) = 0xFFE0 & 0x0020 = 0x0020 ≠ 0 → mask_i_bit=False → -∞
#   i=6: mask & (1<<6) = 0xFFE0 & 0x0040 = 0x0040 ≠ 0 → mask_i_bit=False → -∞
#   ... (i=7 到 i=15 都被 mask)
# 结果: 列 16,17,18,19,20 可见; 列 21,22,...,31 被 mask
# 即列 0-20 可见, 其余都被 mask
@gluon.jit
def _mask_scalar(qk, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s        # 当前s组内可见的列数
    col_lim_right_cur = max(col_lim_right_s, 0)  # 至少为 0
    mask = -1 << col_lim_right_cur               # 构造位掩码
    mask_i_bit = (mask & (1 << i)) == 0          # 检查第 i 位是否为 0
    return gl.where(mask_i_bit, qk, -float("inf"))


# _apply_causal_mask 完整流程
#
# 输入: qk [SPLIT_M, BLOCK_N], col_limit_right [SPLIT_M, 1]
#
# 列索引:   0   1   2   ...  15 | 16  17  18  ...  31 | 32  ...
#           ← ─ ─ s=0 ─ ─ ─ ─ → ← ─ ─ s=16 ─ ─ ─ ─ →
#               i=0,1,...,15          i=0,1,...,15
#
# 输出:
# Row 0: col_limit_right=1
#   V  -∞  -∞  -∞  ...  -∞  -∞  -∞  -∞  ...  -∞  ...
#
# Row 5: col_limit_right=6
#   V   V   V   V   V   V  -∞  -∞  ...  -∞  -∞  ...
#   0   1   2   3   4   5   6   7  ...  15   16  ...
#
# Row 20: col_limit_right=21
#   V   V  ...  V   V   V   V   V   V  -∞  -∞  ...
#   0   1      14  15  16  17  18  19  20  21  22
#   ← ─  s=0  ─ → ← ─ ─ ─ s=16 ─ ─ ─ ─ →
#
# 为什么使用位掩码？
# -----------------
# 朴素方法:
#   mask = offs_n >= col_limit_right     # 逐元素比较
#   qk = where(mask, -inf, qk)           # 逐元素选择 
#   需要 BLOCK_N 次比较, 每次生成一个 predicate
# 位掩码方法:
#   mask = -1 << col_lim_right_cur       # 一次移位生成 16 位掩码
#   mask_i_bit = (mask & (1 << i)) == 0  # 使用 R2P 指令提取 predicate
#   1. R2P (Register to Predicate) 指令可以从寄存器一次提取多个 predicate
#   2. 减少分支/比较指令数量
#   3. 利用 GPU 的位操作单元 
@gluon.jit
def _apply_causal_mask(qk, col_limit_right):
    # 对每 16 个元素的块计算的位掩码应用因果掩码。这在 SASS 级别可以高效使用 R2P(寄存器到谓词)指令。
    # Tri Dao https://github.com/Dao-AILab/flash-attention/commit/bac1001e4f6caa09d70537495d6746a685a2fa78
    offs_n = gl.arange(0, qk.shape[1])[None, :]  # [0, 1, 2, ..., 127]
    # s = offs_n & ~0xf                     # 16 元素分组的起始索引           
    #   = offs_n & 0xFFFFFFF0               # 清除低 4 位                     
    #   = [0,0,0,...,0, 16,16,...,16, 32,32,...,32, ...]
    #     |─  16个  ─|   |─  16个  ─|  |─  16个  ─|
    s = offs_n & ~0xf
    # i = offs_n & 0xf                      # 组内的偏移量 (0-15)
    #   = [0,1,2,...,15, 0,1,2,...,15, 0,1,2,...,15, ...]   
    i = offs_n & 0xf
    # NOTE: 我们在这里使用 map_elementwise 是为了生成一个交错的指令序列, 该序列一次处理 qk 的一个元素。
    # 这可以让 ptxas 更好地优化 SASS, 而不是先算完所有 mask, 再做所有 where
    # 交错顺序: mask[0], where[0], mask[1], where[1], ... 
    # 而不是:   mask[0..N], where[0..N] 
    return gl.map_elementwise(_mask_scalar, qk, col_limit_right, s, i)


@gluon.jit
def _compute_and_store_exp2(config, qk, p_tmem):
    SIZE: gl.constexpr = p_tmem.shape[1] // config.SPLIT_EXP_FACTOR
    qks = _split_n(qk, config.SPLIT_EXP_FACTOR)
    ps = ()
    for i in gl.static_range(config.SPLIT_EXP_FACTOR):
        p = gl.exp2(qks[i])
        p_tmem.slice(i * SIZE, SIZE).store(p.to(config.dtype))
        ps = ps + (p, )
    return _join_n(ps)


@gluon.jit
def _subtiled_qk_load(config, s_tmem):
    SIZE: gl.constexpr = s_tmem.shape[1] // config.SPLIT_QK_LOAD_FACTOR
    s = s_tmem.slice(0, SIZE)
    layout: gl.constexpr = get_tmem_reg_layout(gl.float32, s.shape, s.layout, config.num_warps)
    qks = ()
    for i in gl.static_range(config.SPLIT_QK_LOAD_FACTOR):
        qks = qks + (s_tmem.slice(i * SIZE, SIZE).load(layout), )
    return _join_n(qks)


@gluon.jit
def _softmax_inner_loop(tile_id: gl.constexpr, config, prog,  #
                        s_consumer, corr_producer, exp_turnstile, corr_bar,  #
                        offs_m, m_i, l_i, STAGE: gl.constexpr):
    lo, hi = prog.get_loop_bounds(STAGE)

    for start_n in range(lo, hi, config.BLOCK_N):
        # Step 1: 从 MMA partition 获取 QK 结果
        s_tmem, s_bar, s_consumer = s_consumer.acquire()
        qk = _subtiled_qk_load(config, s_tmem)

        # Step 2: 应用 Causal Mask (仅在 STAGE==2 对角线上的块时需要)
        if STAGE == 2:
            # 含义: 对于当前块的第 i 行, 其可以看到的列数是 col_limit_right[i](就是对角线及之前的元素)
            # 假设 offs_m=[128, 129, ..., 255], start_n=128, 得 col_limit_right=[1, 2, 3, ..., 128]
            # 那么行 m=130 (全局), 在当前块中是第 2 行, col_limit_right[2] = 3, 即其可以看到前 3 列
            col_limit_right = (offs_m - start_n + 1)[:, None]
            qk = _apply_causal_mask(qk, col_limit_right)

        # Step 3: 计算新的行最大值 m_ij 和校正系数 alpha
        m_ij = gl.maximum(m_i, gl.max(qk, 1) * config.qk_scale)
        alpha = gl.exp2(m_i - m_ij)

        # Step 4: 将 alpha 传递给 Correction partition (用于校正 O), 这里同样借用 s_tmem
        alpha_tmem = _borrow_s_as_alpha(config, s_tmem)
        alpha_tmem.store(gl.convert_layout(alpha.expand_dims(1), config.alpha_2d_layout))
        mbarrier.arrive(corr_bar, count=1)

        # Step 5: 计算 P = exp2(qk * scale - m_ij)
        rowmax = float2.pack(-m_ij[:, None].broadcast_to(qk.shape), axis=1)
        # qk = qk * scale - m_ij (使用 float2 FMA 优化)
        qk = float2.pack(qk, axis=1)
        qk = float2.fma(qk, float2.full_like(qk, config.qk_scale), rowmax)
        qk = float2.unpack(qk, axis=1)

        # P = exp2(qk)  (使用 turnstile 避免 EX2 竞争)
        if config.use_exp2_turnstile:
            _, exp_bar, exp_turnstile = exp_turnstile.acquire()

        # FIXME: 当使用 FADD2 reductions 时, ptxas 行为异常, 在进行 FADD2、FMUL2、EX2 时, 即使远低于寄存器限制也会发生溢出。
        # 子块 SPLIT_EXP_FACTOR 大小设置为 4 以尽量减少溢出。
        p_tmem = _borrow_s_as_p(config, s_tmem)
        # 将 P 存入 p_tmem, 供 MMA partition 计算 P @ V
        p = _compute_and_store_exp2(config, qk, p_tmem)

        mbarrier.arrive(s_bar, count=1)
        _, corr_bar, corr_producer = corr_producer.acquire()

        if config.use_exp2_turnstile:
            mbarrier.arrive(exp_bar, count=1)

        # Step 6: 更新 l_i 和 m_i
        l_ij = float2.pack2(*_split_n(p)).sum(axis=1)
        l_ij = Float2Tensor(gl.convert_layout(l_ij.value, l_i.value.type.layout, assert_trivial=True))
        alpha = gl.convert_layout(alpha, l_i.value.type.layout, assert_trivial=True)
        l_i = float2.fma(l_i, float2.pack2(alpha, alpha), l_ij)
        m_i = m_ij

    return m_i, l_i, corr_bar, s_consumer, corr_producer, exp_turnstile


@gluon.jit
def _softmax_tile(tile_id: gl.constexpr, config, M, desc_o, STAGE: gl.constexpr,  #
                  s_chnl, corr_chnl, exp_turnstile):
    qk_slice_dim1: gl.constexpr = gl.SliceLayout(1, config.qk_layout)
    sum_layout: gl.constexpr = _get_split_n_layout(config.qk_layout)

    s_consumer = s_chnl.create_consumer()
    corr_producer = corr_chnl.create_producer()
    _, corr_bar, corr_producer = corr_producer.acquire()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)

        offs_m = prog.start_m * config.BLOCK_M
        offs_m += gl.arange(tile_id * config.SPLIT_M, (1 + tile_id) * config.SPLIT_M)

        m_i = gl.full([config.SPLIT_M], -float("inf"), gl.float32, qk_slice_dim1)
        # Accumulate into 2 row-sums so the reduction can be performed with FADD2.
        l_i = gl.full([config.SPLIT_M], 0.0, gl.float32, gl.SliceLayout(1, sum_layout))
        # GPU 有 FADD2 指令, 可以一次操作两个 float32
        # 原始方式: l_i += sum(P)
        # 优化方式: l_i_packed = FADD2(l_i_packed, P_packed)  // 一次处理两个值
        # pack2(l, l) 创建 [l, l] 的结构, 循环中使用 float2 运算, 最后 unpack 并相加得到真正的结果
        l_i = float2.pack2(l_i, l_i)

        # causal_mask 的索引方式与 triton 中的 06-fused-attention.py 一致
        # 阶段 1: off-band(对角线之前的块, 全部处理, 无需mask)
        if STAGE & 1:
            m_i, l_i, corr_bar, s_consumer, corr_producer, exp_turnstile = _softmax_inner_loop(  #
                tile_id, config, prog, s_consumer, corr_producer, exp_turnstile, corr_bar,  #
                offs_m, m_i, l_i, STAGE=4 - STAGE)
        # 阶段 2: on-band(对角线上的块, 部分处理, 需要causal mask)
        if STAGE & 2:
            m_i, l_i, corr_bar, s_consumer, corr_producer, exp_turnstile = _softmax_inner_loop(  #
                tile_id, config, prog, s_consumer, corr_producer, exp_turnstile, corr_bar,  #
                offs_m, m_i, l_i, STAGE=2)
        l_i0, l_i1 = float2.unpack2(l_i)
        l_i = l_i0 + l_i1

        s_tmem, s_bar, s_consumer = s_consumer.acquire()
        # s_tmem 原本存储 QK 结果 [SPLIT_M, BLOCK_N], 借用它的一部分空间来存储 m_i 和 l_i
        # 为什么可以复用？ 当需要存储 m_i, l_i 时, 该迭代的 QK 计算已完成,结果已经转换成 P (softmax 概率) 并传递出去
        # 复用的目的: 节省 Tensor Memory 空间, m_i, l_i 需要传递给 Correction partition 用于最终 rescale 
        m_i_tmem, l_i_tmem = _borrow_s_for_epilogue(config, s_tmem)
        m_i_tmem.store(gl.convert_layout(m_i.expand_dims(1), config.alpha_2d_layout))
        l_i_tmem.store(gl.convert_layout(l_i.expand_dims(1), config.alpha_2d_layout))

        mbarrier.arrive(corr_bar, count=1)
        _, corr_bar, corr_producer = corr_producer.acquire()

        mbarrier.arrive(s_bar, count=1)


@gluon.jit
def _attn_fwd_softmax0(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs
    _softmax_tile(0, config, M, desc_o, STAGE, s0_chnl, c0_chnl, exp_turnstile.create_producer())


@gluon.jit
def _attn_fwd_softmax1(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs
    _softmax_tile(1, config, M, desc_o, STAGE, s1_chnl, c1_chnl, exp_turnstile.create_consumer())


@gluon.jit
def _attn_fwd_epilogue(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    epi_consumer = epi_chnl.create_consumer()
    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)

        # 从 Correction partition 等待 o0_smem, 写回 gmem
        o0_smem, o0_bar, epi_consumer = epi_consumer.acquire()
        tma.async_copy_shared_to_global(desc_o, [prog.qo_offset_y + config.SPLIT_M * 0, 0], o0_smem)

        # 从 Correction partition 等待 o1_smem, 写回 gmem
        o1_smem, o1_bar, epi_consumer = epi_consumer.acquire()
        tma.async_copy_shared_to_global(desc_o, [prog.qo_offset_y + config.SPLIT_M * 1, 0], o1_smem)

        tma.store_wait(1)
        mbarrier.arrive(o0_bar, count=1)
        tma.store_wait(0)
        mbarrier.arrive(o1_bar, count=1)


@gluon.jit
def _attn_fwd_correction_rescale(config, s_tmem, corr_consumer, o_consumer):
    """
    对累积的 O 进行 rescale: O = O * alpha
    
    alpha 来自 Softmax partition, alpha = exp(m_old - m_new)
    """
    alpha_layout: gl.constexpr = gl.SliceLayout(1, config.o_splitn_layout)

    # 等待 MMA partition 当前累积的 o_tmem
    o_tmem, o_bar, o_consumer = o_consumer.acquire()
    # 等待 Softmax partition 的 alpha
    _, corr_bar, corr_consumer = corr_consumer.acquire()

    # alpha 借用的 s_tmem
    alpha = _borrow_s_as_alpha(config, s_tmem).load(config.alpha_2d_layout)
    # 通知 Softmax partition 可以继续进行
    mbarrier.arrive(corr_bar, count=1)

    alpha = gl.convert_layout(alpha.reshape([config.SPLIT_M]), alpha_layout)
    # 广播 alpha 到 O 的形状 [SPLIT_M, SPLIT_D], SPLIT_M=BLOCK_M//2, SPLIT_D=HEAD_DIM//SPLIT_D_FACTOR
    # float2.pack 是 SIMD 优化技术, 用于利用 GPU 的 双精度浮点指令 (FADD2/FMUL2) 一次处理两个 float32 值。
    #
    #   原始数据 (axis=1 方向):
    #     a0   a1   a2   a3   a4   a5   a6   a7     shape: [M, 8]
    #
    #   float2.pack(tensor, axis=1):
    #     (a0, a1)  (a2, a3)  (a4, a5)  (a6, a7)    shape: [M, 4], 元素类型: float2
    #
    #   float2.unpack(tensor, axis=1):
    #     a0   a1   a2   a3   a4   a5   a6   a7     shape: [M, 8], 元素类型: float32
    #
    # 普通 float32 乘法:
    #   FMUL a0, alpha    # 1 条指令, 1 次乘法                                               
    #   FMUL a1, alpha    # 1 条指令, 1 次乘法   
    # float2 乘法 (FMUL2):
    #   FMUL2 (a0,a1), (alpha,alpha)  # 1 条指令, 2 次乘法, 吞吐量翻倍
    alpha = float2.pack(alpha[:, None].broadcast_to(config.o_shape[0], config.SPLIT_D), axis=1)
    # 分块处理 (减少寄存器压力), SPLIT_D_FACTOR=2
    for i in gl.static_range(config.SPLIT_D_FACTOR):
        o_ref = o_tmem.slice(i * config.SPLIT_D, config.SPLIT_D)
        o = float2.pack(o_ref.load(config.o_splitn_layout), axis=1)
        o = o * alpha  # rescale! float2 × float2 (使用 FMUL2 指令)
        o_ref.store(float2.unpack(o, axis=1))
    # 通知 MMA partition 可以继续进行
    mbarrier.arrive(o_bar, count=1)
    return corr_consumer, o_consumer


@gluon.jit
def _attn_fwd_correction_epilogue(config, prog, s_tmem, M, corr_consumer, epi_producer, o_consumer):
    """
    遍历完KV后, 进行最后一次校正(V2论文12-15行):
    1. 用 l_i 归一化: O = O / l_i
    2. 将 o 转换为 bf16 并写入 o_smem, 交由 _attn_fwd_epilogue 写回 global memory
    3. 保存 m_i 用于反向传播
    """
    alpha_layout: gl.constexpr = gl.SliceLayout(1, config.o_splitn_layout)

    # 从 Softmax partition 获取最终的 m_i 和 l_i, 因为借用的 s_tmem, 所以需要调用 _borrow_s_for_epilogue
    _, corr_bar, corr_consumer = corr_consumer.acquire()
    m_i_tmem, l_i_tmem = _borrow_s_for_epilogue(config, s_tmem)
    m_i = m_i_tmem.load(config.alpha_2d_layout).reshape([config.SPLIT_M])
    m_i = gl.convert_layout(m_i, alpha_layout)
    l_i = l_i_tmem.load(config.alpha_2d_layout).reshape([config.SPLIT_M])
    l_i = gl.convert_layout(l_i, alpha_layout)
    mbarrier.arrive(corr_bar, count=1)

    # 获取输出缓冲区
    o_smem, epi_bar, epi_producer = epi_producer.acquire()
    o_tmem, o_bar, o_consumer = o_consumer.acquire()

    # 共享内存子块大小受字节大小的限制。
    contigDimSize: gl.constexpr = o_smem.type.layout.swizzle_byte_width * 8 // o_smem.type.element_ty.primitive_bitwidth
    if o_smem.type.shape[1] // config.SPLIT_D_FACTOR >= contigDimSize:
        SPLIT_N_FACTOR: gl.constexpr = config.SPLIT_D_FACTOR
    else:
        SPLIT_N_FACTOR: gl.constexpr = 1
    gl.static_assert(o_smem.type.shape[1] // SPLIT_N_FACTOR >= contigDimSize,
                     "Block shape is too small for the swizzle byte size in NVMMA Shared Layout")
    SPLIT_N: gl.constexpr = o_smem.type.shape[1] // SPLIT_N_FACTOR

    # 归一化: O = O / l_i, 使用 pack 优化
    # scale = 1/l_i, 需要将其广播到 [SPLIT_M, SPLIT_N]
    scale = float2.pack((1 / l_i)[:, None].broadcast_to(config.o_shape[0], SPLIT_N), axis=1)
    for i in gl.static_range(SPLIT_N_FACTOR):
        o_ref = o_tmem.slice(i * SPLIT_N, SPLIT_N)
        o = float2.pack(o_ref.load(config.o_splitn_layout), axis=1)
        o = o * scale  # 归一化: O = O * (1/l_i) = O / l_i
        # 转换为 bf16 并存入 SMEM
        o_smem.slice(i * SPLIT_N, SPLIT_N, dim=1).store(float2.unpack(o, axis=1).to(config.dtype))

    # 同步
    fence_async_shared()
    mbarrier.arrive(epi_bar, count=1)  # 通知 Epilogue 数据已就绪
    mbarrier.arrive(o_bar, count=1)  # 释放 o_tmem

    #  计算 mi = logsumexp (m_i + log(l_i)) 用于反向传播
    m_i += gl.log2(l_i)
    coalesced: gl.constexpr = gl.BlockedLayout([1], [32], [config.num_warps], [0])
    offs_m = prog.start_m * config.BLOCK_M
    offs_m += gl.arange(0 * config.SPLIT_M, 1 * config.SPLIT_M, coalesced)
    m_ptrs = M + prog.off_hz * config.N_CTX + offs_m
    gl.store(m_ptrs, gl.convert_layout(m_i, coalesced))  # 写回 mi 到 global memory

    return corr_consumer, epi_producer, o_consumer

# O_new = O_old * alpha + P_new @ V_new
@gluon.jit
def _attn_fwd_correction(config, chnls, descs, M, STAGE: gl.constexpr):
    q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile = chnls

    s0_tmem = s0_chnl.mem.index(0)
    s1_tmem = s1_chnl.mem.index(0)
    # 分别从 Softmax0 和 Softmax1 接收 alpha
    corr0_consumer = c0_chnl.create_consumer()
    corr1_consumer = c1_chnl.create_consumer()
    # 从 MMA 接收累积的 O
    o_consumer = o_chnl.create_consumer()

    epi_producer = epi_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_fused_loop_bounds(STAGE)
        num_corrections = (hi - lo) // config.BLOCK_N  # 需要校正的次数

        # 跳过第一次校正 (因为第一次没有"之前的结果"需要校正)
        _, corr0_bar, corr0_consumer = corr0_consumer.acquire()
        mbarrier.arrive(corr0_bar, count=1)  # 只消费, 不计算
        _, corr1_bar, corr1_consumer = corr1_consumer.acquire()
        mbarrier.arrive(corr1_bar, count=1)  # 只消费, 不计算

        # 遍历剩下的 KV 块进行 rescale O, 论文第10行
        for i in range(num_corrections - 1):
            corr0_consumer, o_consumer = _attn_fwd_correction_rescale(config, s0_tmem, corr0_consumer, o_consumer)
            corr1_consumer, o_consumer = _attn_fwd_correction_rescale(config, s1_tmem, corr1_consumer, o_consumer)

        # 遍历完所有的KV块后, 再对 O 进行 scale, O = O / l_i, 论文第12-15行
        corr0_consumer, epi_producer, o_consumer = _attn_fwd_correction_epilogue(  #
            config, prog, s0_tmem, M, corr0_consumer, epi_producer, o_consumer)
        corr1_consumer, epi_producer, o_consumer = _attn_fwd_correction_epilogue(  #
            config, prog, s1_tmem, M, corr1_consumer, epi_producer, o_consumer)


def attention_repr(specialization):
    name = "gluon_attention"
    # Up to 150 TFLOPS faster for fp8!
    if specialization.constants["dtype"] == gl.float8e5:
        name = "cutlass_" + name
    return name


@gluon.jit(do_not_specialize=["Z"], repr=attention_repr)
def attention_kernel(  #
        sm_scale, M, Z, H, N_CTX, desc_q, desc_k, desc_v, desc_o,  #
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
        GROUP_SIZE_N: gl.constexpr, NUM_SMS: gl.constexpr, STAGE: gl.constexpr, dtype: gl.constexpr,  #
        num_warps: gl.constexpr):
    qk_scale = sm_scale * 1.44269504
    config = AttentionConfig(qk_scale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, GROUP_SIZE_N, NUM_SMS, STAGE,  #
                             dtype, num_warps)

    # num_buffers 固定为2, 对应 SPLIT_M = BLOCK_M // 2
    q_chnl = get_desc_channel(desc_q, num_buffers=2)
    # K 和 V 的形状完全相同, 共享同一个 Channel, 交替使用缓冲区
    kv_chnl = get_desc_channel(desc_k, num_buffers=config.num_kv_buffers)
    o_chnl = TensorMemoryChannel.alloc(config.o_shape, gl.float32, config.o_tmem_layout, num_buffers=2)
    epi_chnl = SharedMemoryChannel.alloc(config.o_shape, config.dtype, gl.constexpr(desc_o.layout), num_buffers=2)
    s0_chnl = TensorMemoryChannel.alloc(config.qk_shape, gl.float32, config.qk_tmem_layout, num_buffers=1)
    s1_chnl = TensorMemoryChannel.alloc(config.qk_shape, gl.float32, config.qk_tmem_layout, num_buffers=1)
    c0_chnl = SharedMemoryChannel.alloc([1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1)
    c1_chnl = SharedMemoryChannel.alloc([1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1)
    exp_turnstile = SharedMemoryChannel.alloc([1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1)

    chnls = (q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl, exp_turnstile)
    descs = (desc_q, desc_k, desc_v, desc_o)
    # 
    # 分区	               Warps Regs/Thread  Threads	该分区总寄存器
    # Correction (默认)	     4	    128	     4×32=128	128 × 128 = 16,384(maxnreg)
    # Softmax0 (Worker 0)	4	   192	    4×32=128   128 × 192 = 24,576
    # Softmax1 (Worker 1)	4	   192	    4×32=128   128 × 192 = 24,576
    # MMA (Worker 2)	    1	   24	    1×32=32	   32 × 24 = 768
    # Load (Worker 3)	    1	   24	    1×32=32	   32 × 24 = 768
    # Epilogue (Worker 4)	1	   24	    1×32=32	   32 × 24 = 768
    # 总寄存器需求: 16,384 + 24,576 + 24,576 + 768 + 768 + 768 = 67,840 个寄存器 > 65,536, 所以默认分区寄存器数会小于128
    #
    # 为什么 Softmax 需要 192 个寄存器/线程？
    # Softmax Warp 寄存器使用分析
    #
    #   每个线程需要存储:
    #
    #   1. QK 矩阵块的一部分
    #      - qk: [SPLIT_M, BLOCK_N] = [128, 128] 分布在 128 个线程上
    #      - 每线程约 128 个 float32 值 = 128 regs
    #
    #   2. 统计量
    #      - m_i (row max): 每线程若干个值
    #      - l_i (row sum): 每线程若干个值
    #      - alpha: 校正因子
    #
    #   3. 中间计算
    #      - exp2 结果, float2 打包/解包的临时值, 循环变量、地址计算等
    #
    #   4. 控制流
    #      - barrier 指针, channel 状态, 循环计数器
    #
    #   总计约 150-192 个寄存器/线程, 编译器会根据代码自动分配, 192 是上限约束
    #
    # 为什么 MMA/Load/Epilogue 只需要 24 个寄存器？
    # MMA/Load/Epilogue Warp 寄存器使用分析
    #
    #   这些 Warp 主要是"指令发射器", 不保存大量数据:
    #
    #   MMA Warp:
    #     tcgen05_mma(smem_ptr, smem_ptr, tmem_ptr, ...)
    #                 ↑          ↑          ↑
    #              指针        指针        指针
    #
    #     只需要存储: 几个指针、barrier handle、循环计数器
    #     实际数据在 SMEM 和 TMEM 中, 不在寄存器里
    #
    #   Load Warp:
    #     tma.async_copy_global_to_shared(desc, [offset, 0], bar, smem)
    #                                      ↑        ↑        ↑     ↑
    #                                   指针     offset     bar   指针
    #
    #     只需要存储: TMA descriptor 指针、offset、barrier handle
    #     数据直接从 Global → SMEM, 不经过寄存器
    #
    #   24 regs/thread x 32 threads = 768 regs/warp 已经足够
    #
    # 为什么是24?(fa3也是24)
    # 硬件最小值: 理论上每线程可以只用 8 个寄存器 (1 个分配单元), 使用24可以防止寄存器不足。
    # 编译器会将超出的变量 "spill" 到 Local Memory (实际是 DRAM), 导致性能严重下降。
    # MMA/Load/Epilogue Warp 的最小需求
    #
    #   这些 Warp 的典型寄存器使用:
    #
    #   Load Warp 示例:
    #     pid (循环计数器)           ~2 regs
    #     scheduler 状态             ~4 regs
    #     offset_y, qo_offset_y     ~4 regs
    #     smem 指针                  ~2 regs
    #     barrier handle            ~2 regs
    #     producer 状态              ~4 regs
    #     TMA descriptor 相关       ~4 regs
    #     临时变量                   ~2 regs
    #     ─────────────────────────────────
    #     总计约                    ~24 regs
    #
    #   24 是一个"刚好够用"的选择, 既能完成任务, 又节省寄存器给其他分区
    gl.warp_specialize([
        (_attn_fwd_correction, (config, chnls, descs, M, STAGE)),  # 默认分区: 4 warps(num_warps=4), 128 reg(maxnreg=128)
        (_attn_fwd_softmax0, (config, chnls, descs, M, STAGE)),  # 4 warps, 192 reg
        (_attn_fwd_softmax1, (config, chnls, descs, M, STAGE)),  # 4 warps, 192 reg
        (_attn_fwd_mma, (config, chnls, descs, M, STAGE)),  # 1 warp, 24 reg
        (_attn_fwd_load, (config, chnls, descs, M, STAGE)),  # 1 warp, 24 reg
        (_attn_fwd_epilogue, (config, chnls, descs, M, STAGE)),  # 1 warp, 24 reg
    ], [4, 4, 1, 1, 1], [192, 192, 24, 24, 24])

    q_chnl.release()
    kv_chnl.release()
    o_chnl.release()
    epi_chnl.release()
    s0_chnl.release()
    s1_chnl.release()
    c0_chnl.release()
    c1_chnl.release()
    exp_turnstile.release()


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


def torch_dtype_to_triton(dtype):
    if dtype == torch.float8_e5m2:
        return gl.float8e5
    return getattr(gl, str(dtype).split('.')[1])


def make_tensor_desc(x, shape, strides, block_shape):
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, torch_dtype_to_triton(x.dtype))
    return TensorDescriptor(x, shape=shape, strides=strides, block_shape=block_shape, layout=layout)


def attention_forward(q, k, v, causal, sm_scale):
    # 输入张量形状: Q, K, V 都是 [Z, H, N_CTX, HEAD_DIM]
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    stage = 3 if causal else 1

    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    # The kernel will split BLOCK_M into two subtiles.
    BLOCK_M = 256
    BLOCK_N = 128
    SPLIT_M = BLOCK_M // 2
    GROUP_SIZE_N = 4 if causal else 1
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    desc_q = make_tensor_desc(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[SPLIT_M, HEAD_DIM_K])
    desc_v = make_tensor_desc(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_k = make_tensor_desc(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_o = make_tensor_desc(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[SPLIT_M, HEAD_DIM_K])

    num_pid_m = triton.cdiv(q.shape[2], BLOCK_M)  # seq
    num_pid_n = q.shape[0] * q.shape[1]  # Batch * Heads
    # 使用 Persistent Kernel
    grid = min(NUM_SMS, num_pid_m * num_pid_n)

    attention_kernel[(grid, )](
        sm_scale, M, q.shape[0], q.shape[1], q.shape[2],  #
        desc_q, desc_k, desc_v, desc_o,  #
        BLOCK_M, BLOCK_N, HEAD_DIM_K, GROUP_SIZE_N, NUM_SMS,  #
        stage, torch_dtype_to_triton(q.dtype),  #
        num_warps=4, maxnreg=128)  # maxnreg只约束默认分区

    return o, M


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [2, 48])
@pytest.mark.parametrize("N_CTX", [256, 1024, 4 * 1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not is_blackwell(), reason="Gluon attention is only supported on Blackwell GPUs")
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, profile=False):
    device = "cuda"

    torch.manual_seed(42)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5

    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)

    tri_out, _ = attention_forward(q, k, v, causal, sm_scale)
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)


# ===-----------------------------------------------------------------------===#
# Benchmarking
# ===-----------------------------------------------------------------------===#

BATCH = [4]
N_HEADS = [32]
HEAD_DIM = [64, 128]
causal = [False, True]
providers = ["triton-bf16", "cudnn-bf16", "triton-fp8"]
N_CTX = [2**i for i in range(10, 17)]

bench_configs = []
for Z, H, D, is_causal in itertools.product(BATCH, N_HEADS, HEAD_DIM, causal):
    config = triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=N_CTX,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")],
        ylabel="TFLOPS",
        plot_name=f"Attention Z={Z} H={H} D={D} causal={is_causal}",
        args={
            "Z": Z,
            "H": H,
            "HEAD_DIM": D,
            "causal": is_causal,
        },
    )
    bench_configs.append(config)


@triton.testing.perf_report(bench_configs)
def bench(Z, H, N_CTX, HEAD_DIM, causal, provider):
    provider, dtype = provider.split("-")
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp8":
        dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    device = "cuda"

    torch.manual_seed(42)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), device=device).normal_(mean=0.0, std=0.5).requires_grad_()).to(dtype)
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), device=device).normal_(mean=0.0, std=0.5).requires_grad_()).to(dtype)
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), device=device).normal_(mean=0.0, std=0.5).requires_grad_()).to(dtype)
    sm_scale = 1.3

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
        if provider == "triton":
            fn = lambda: attention_forward(q, k, v, causal, sm_scale)
        elif provider == "cudnn":
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        ms = triton.testing.do_bench(fn)
        flops_per_matmul = 2.0 * Z * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench.run(print_data=True)
