"""
Tri Dao Flash Attention v2 的 Triton 实现 (https://tridao.me/publications/flash2/flash2.pdf)

致谢: OpenAI kernel 团队

额外致谢:

* 原始 Flash Attention 论文 (https://arxiv.org/abs/2205.14135)
* Rabe & Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9

# Online Softmax 数学原理
# ──────────────────────
#
# 标准 Softmax: P = softmax(QK^T) = exp(QK^T) / sum(exp(QK^T))
#
# 问题: 需要先算完所有 QK^T 才能得到 sum, 无法流式处理
# Online Softmax 解决方案(假设当前为Qi):
# ─────────────────────────────────────
# 维护两个状态:
# - m_i: 当前已见元素的行最大值
# - l_i: 当前已见元素的行和 (经过 max 校正后)
#
# 固定Qi, 遍历K_j/V_j:
#   1. m_ij = max(m_i, max(QiK_j))           # 更新最大值
#   2. alpha = exp(m_i - m_ij)              # 旧状态的校正系数
#   3. P_j = exp(QiK_j - m_ij)               # 新块的 softmax 概率
#   4. l_i = l_i * alpha + sum(P_j)         # 更新行和
#   5. m_i = m_ij                           # 更新最大值状态
#
#   同时，之前累积的 O 也需要用 alpha 校正:
#   O = O * alpha + P_j @ V_j
# 在更新所有的K/V块后，再对O进行rescale:
# O = O / li
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # 假设:  BLOCK_M = 4, BLOCK_N = 4, N_CTX = 8, causal = True (STAGE = 3)  
    #            K 块 →      j=0          j=1
    #                    ┌──────────┬──────────┐
    #                    │  K[0:4]  │  K[4:8]  │
    #     Q 块 ↓         │          │          │
    #  ┌─────────────────┼──────────┼──────────┤
    #  │                 │ ✓✗✗✗     │ ✗✗✗✗     │
    #  │  start_m=0      │ ✓✓✗✗     │ ✗✗✗✗     │
    #  │  Q[0:4]         │ ✓✓✓✗     │ ✗✗✗✗     │
    #  │                 │ ✓✓✓✓     │ ✗✗✗✗     │
    #  ├─────────────────┼──────────┼──────────┤
    #  │                 │ ✓✓✓✓     │ ✓✗✗✗     │
    #  │  start_m=1      │ ✓✓✓✓     │ ✓✓✗✗     │
    #  │  Q[4:8]         │ ✓✓✓✓     │ ✓✓✓✗     │
    #  │                 │ ✓✓✓✓     │ ✓✓✓✓     │
    #  └─────────────────┴──────────┴──────────┘
    #                       ↑          ↑
    #                    全有效     部分有效/全无效
    # lo (low) 和 hi (high) 定义了当前阶段需要遍历的 K/V 块的范围。
    # - 对于 start_m = 0 (处理 Q[0:4])
    #   第一次调用 _attn_fwd_inner (STAGE & 1 = True, 传入 STAGE = 4-3 = 1)
    #   if STAGE == 1: 
    #     lo, hi = 0, 0  # for start_n in range(0, 0, BLOCK_N) -> 空循环！不执行任何操作
    #   第二次调用 _attn_fwd_inner (STAGE & 2 = True, 传入 STAGE = 2):
    #   elif STAGE == 2:
    #     lo, hi = 0, 4  # for start_n in range(0, 4, BLOCK_N) -> 处理 K[0:4]，on-band，causal mask
    # - 对于 start_m = 1 (处理 Q[4:8])
    #  第一次调用 _attn_fwd_inner (STAGE & 1 = True, 传入 STAGE = 4-3 = 1):
    #  if STAGE == 1: 
    #     lo, hi = 0, 4 # for start_n in range(0, 4, BLOCK_N) -> 处理 K[0:4]，off-band，无需 mask
    #  第二次调用 _attn_fwd_inner (STAGE & 2 = True, 传入 STAGE = 2): 
    #  elif STAGE == 2:
    #     lo, hi = 4, 8 # for start_n in range(4, 8, BLOCK_N) -> 处理 K[4:8]，on-band，需要 causal mask
    # 可以看出只有当 STAGE == 2 时，才需要处理 causal mask

    # causal = True: 处理对角线之前的块
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    # causal = True: 处理对角线上的块，需要causal mask
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M) # tl.multiple_of 告诉编译器 "lo 一定是 BLOCK_M 的倍数"，以使用对齐的内存访问指令
    # causal = False
    else:
        lo, hi = 0, N_CTX  # 遍历所有K块
    # 计算 K 的起始偏移
    offsetk_y = offset_y + lo
    # 计算 V 的起始偏移, fp8时V进行了转置，因此需要乘以HEAD_DIM
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # 循环遍历 k、v 并更新累加器
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- 计算 qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- 计算修正因子
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- 更新输出累加器 --
        # 优化 Blackwell 上 warp specialization 模式的乘法效率
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # 准备 p 和 v 以进行点积
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # 注意，这种非转置 v 仅在 Blackwell 上受 FP8 支持
        acc = tl.dot(p, v, acc)
        # 更新 m_i 和 l_i
        # 将其放在循环的末尾以减少寄存器压力
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]
if "PYTEST_VERSION" in os.environ:
    # 在测试中使用单一配置以确保可重复性
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]

    # 过滤掉 BLOCK_M > N_CTX 的配置
    # 过滤掉 causal 模式下 BLOCK_M < BLOCK_N 的配置
    return [
        conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX and (
            conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    # 定位到当前block的(batch, head) 的起始位置 
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    # 定位当前q块的起始位置(在offset_y的基础上，再加上seq_len的偏移)
    qo_offset_y = offset_y + start_m * BLOCK_M
    # 初始化块的偏移量: m维度和n维度的偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # 初始化 m 和 l 的指针
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # 加载缩放因子
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # 加载 q: 它将在整个过程中保留在 SRAM 中
    q = desc_q.load([qo_offset_y, 0])

    # 对于 causal = True，STAGE = 3，_attn_fwd_inner 得到 1 作为其 STAGE
    # 对于 causal = False，STAGE = 1，_attn_fwd_inner 得到 3 作为其 STAGE

    # STAGE & 1 检查最低位 (是否处理 off-band)
    # STAGE & 2 检查次低位 (是否处理 on-band)
    # causal=False 时, STAGE=1 (二进制 01)  
    #   STAGE & 1 = 01 & 01 = 01 = True  → 执行 (传入 4-1=3，处理全部)
    #   STAGE & 2 = 01 & 10 = 00 = False → 不执行
    # causal=True 时, STAGE=3 (二进制 11)
    #   STAGE & 1 = 11 & 01 = 01 = True  → 执行 (传入 4-3=1, off-band)
    #   STAGE & 2 = 11 & 10 = 10 = True  → 执行 (传入 2, on-band)


    # 阶段 1: off-band(对角线之前的块, 全部处理，无需mask)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # 阶段 2: on-band(对角线上的块, 部分处理，需要causal mask)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # 尾声
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # 加载
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # 写回
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(dk, dv,  # 输出：累加的 dK, dV
                   Q, k, v, sm_scale,   # 输入：Q指针，已加载的k,v
                   DO,  # 输入：dO指针
                   M, D,  # 输入：logsumexp M，预计算的 Delta
                   # 由 Q/K/V/DO 共享。
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # 由包装器填充。
                   start_n, start_m, num_steps,  # 循环参数
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 必须是 BLOCK_M1 的倍数，否则代码将无法工作。
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # 在计算 qk 之前加载 m 以减少流水线停顿。
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        # 重计算 S, P
        qkT = tl.dot(k, qT)  # 论文10行: S = QK^T
        pT = tl.math.exp2(qkT - m[None, :])  # 论文11行: P = exp(S-L)
        # 自回归掩码。
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # 计算 dV。
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)  # 论文12行: dV = P^T @ dO
        # D (= delta) 被 ds_scale 预除。
        Di = tl.load(D + offs_m)
        # 计算 dP 和 dS。
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)  # 论文13行: dP = dO @ V^T
        dsT = pT * (dpT - Di[None, :])  # 论文14行: dS = P ⊙ (dP - D)
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))  # 论文16行: dK = dS^T @ Q
        # 增加指针。
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # 由 Q/K/V/DO 共享。
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # 由包装器填充。
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) 被 ds_scale 预除。
    Di = tl.load(D + offs_m)
    # BLOCK_M2 必须是 BLOCK_N2 的倍数，否则代码将无法工作。
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        # 重计算 S, P
        qk = tl.dot(q, kT)  # 论文10行: S = QK^T
        p = tl.math.exp2(qk - m)  # 论文11行: P = exp(S-L)
        # 自回归掩码。
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # 计算 dP 和 dS。
        dp = tl.dot(do, vT).to(tl.float32)  # 论文13行: dP = dO @ V^T
        ds = p * (dp - Di[:, None])  # 论文14行: dS = P ⊙ (dP - D)
        ds = ds.to(tl.float16)
        # 计算 dQ。
        # 注意: 我们需要在最后对 dq 进行缩放，因为 kT 是预缩放的。
        dq += tl.dot(ds, tl.trans(kT))  # 论文15行: dQ = dS @ K
        # 增加指针。
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq

# _attn_bwd 完整流程 (以 pid=2 为例)
# ────────────────────────────────────
#
# 假设: N_CTX=5 块, BLOCK_M1=BLOCK_N1=BLOCK_M2=BLOCK_N2=1
#       BLK_SLICE_FACTOR=1 (简化)
#
# Causal Attention 矩阵 (pid=2 的 thread block):
#
#            K,V 块 →    0     1     2     3     4
#                     ┌─────┬─────┬─────┬─────┬─────┐
#     Q 块  0         │  ◉  │  ×  │  ×  │  ×  │  ×  │
#            1        │  √  │  ◉  │  ×  │  ×  │  ×  │
#            2  ←pid  │  √  │  √  │  ◉  │  ×  │  ×  │  ← dQ 处理这行(从右往左)
#            3        │  √  │  √  │  √  │  ◉  │  ×  │
#            4        │  √  │  √  │  √  │  √  │  ◉  │
#                     └─────┴─────┴─────┴─────┴─────┘
#                                    ↑
#                           dK,dV 处理这列(从上往下)
#     ◉ = on-band (需要 mask)
#     √ = off-band (不需要 mask)
#     × = 跳过 (causal 外)
#
# ═══════════════════════════════════════════════════
# 【调用1】_attn_bwd_dkdv(MASK=True)
#   处理 K,V 块 2 对应的 on-band: Q 块 2
#   累加: (Q2, K2, V2) → dK2, dV2
# ───────────────────────────────────
# 【调用2】_attn_bwd_dkdv(MASK=False)
#   处理 K,V 块 2 对应的 off-band: Q 块 3, 4
#   累加: (Q3, K2, V2) → dK2, dV2
#         (Q4, K2, V2) → dK2, dV2
#   写回: dK2, dV2
#
# ═══════════════════════════════════════════════════
# 【调用3】_attn_bwd_dq(MASK=True)
#   处理 Q 块 2 对应的 on-band: K,V 块 2
#   累加: (Q2, K2, V2) → dQ2
# ─────────────────────────────────
# 【调用4】_attn_bwd_dq(MASK=False)
#   处理 Q 块 2 对应的 off-band: K,V 块 0, 1
#   累加: (Q2, K0, V0) → dQ2
#         (Q2, K1, V1) → dQ2
#   写回: dQ2

@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # 由 Q/K/V/DO 共享。
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              CAUSAL: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # batch/head 的偏移指针
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # 加载缩放因子
    offs_k = tl.arange(0, HEAD_DIM)

    # 确定当前处理的 K,V 块
    start_n = pid * BLOCK_N1  # K,V 块的起始位置
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # 加载 K 和 V: 它们在整个内部循环中保留在 SRAM 中。
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        # 第一次调用 _attn_bwd_dkdv, 计算 on-band(对角线块) 的 dK 和 dV
        #   start_m = start_n = pid * BLOCK_N1 = 2 * 128 = 256
        #   num_steps = BLOCK_N1 // MASK_BLOCK_M1 = 128 // 16 = 8
        #   得到 Q 块范围: [start_m, start_m + num_steps * MASK_BLOCK_M1) = [256, 256+8*16) = [256, 384)
        # K,V 块 pid=2 的范围是 [256, 384)，共 128 个 token (BLOCK_N1=128), 所以对角线的 Q 块的范围是 [256, 384)
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(dk, dv,  #
                                Q, k, v, sm_scale,  #
                                DO,  #
                                M, D,  #
                                stride_tok, stride_d,  #
                                H, N_CTX,  #
                                MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                                start_n, start_m, num_steps,  #
                                MASK=True,  #
                                )

        start_m += num_steps * MASK_BLOCK_M1  # 移动到对角线之后

    # 计算 non-masked blocks 的 dK 和 dV
    # 第二次调用 _attn_bwd_dkdv, 计算 off-band 的 dK 和 dV
    #    start_m += num_steps * MASK_BLOCK_M1 = 256 + 8 * 16 = 384
    #    num_steps = (N_CTX - start_m) // BLOCK_M1
    #    Q 块范围: [start_m, N_CTX)
    num_steps = (N_CTX - start_m) // BLOCK_M1  # 剩余的 Q 块数, 不需要再用MASK_BLOCK_M1精细分块了
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # 写回 dK。
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # 开始计算 dQ
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        # 第一次调用 _attn_bwd_dq, 计算 on-band(对角线块)的 dQ。
        # 注意: 按从右到左扫描 QK^T 的每一行，但在_attn_bwd_dq 的内部，还是从左到右。这不是什么优化，只是为了复用代码结构。
        #   start_m = pid * BLOCK_M2 = 2 * 128 = 256
        #   end_n = start_m + BLOCK_M2 = 256 + 128 = 384 (causal 边界)
        #   num_steps = BLOCK_M2 // MASK_BLOCK_N2 = 128 // 16 = 8
        #   start_n = end_n - num_steps * MASK_BLOCK_N2 = 384 - 8 * 16 = 256 # 从右往左传入kernel
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V,  #
                          do, m, D,  #
                          stride_tok, stride_d,  #
                          H, N_CTX,  #
                          BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                          start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                          MASK=True,  #
                          )
        # 第二次调用 _attn_bwd_dq, 计算 off-band 的 dQ。
        #   end_n -= num_steps * MASK_BLOCK_N2 = 384 - 8 * 16 = 256
        #   num_steps = end_n // BLOCK_N2 = 256 // 32 = 8
        #   start_n = end_n - num_steps * BLOCK_N2 = 256 - 8 * 32 = 0 # 通常为0
        end_n -= num_steps * MASK_BLOCK_N2  # 更新右边界
        # stage 2
        num_steps = end_n // BLOCK_N2  # 剩余的 K,V 块数
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, start_n, num_steps,  #
                      MASK=False,  #
                      )
    # 写回 dQ。
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        # 形状约束: q.shape = [Z, H, N_CTX, HEAD_DIM] # [batch, head, seq_len, head_dim]
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # Hopper 上无法对非转置的第二个 tensor 执行 FP8 点积,
        # 当 v 为 float8_e5m2 时需要转置。
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # AMD 设备的调优参数
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        # 全 0 tensor 存储 LSE (logsumexp), 反向传播时用于重建 P 矩阵
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # 对 Hopper + warpspec 使用 device_descriptor。
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # y_dim 不是 head dimension，而是用于 TensorDescriptor 的展平维度大小。
            y_dim = q.shape[0] * q.shape[1] * q.shape[2] # batch * head * seq_len

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1],
                                          block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                                          block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        _attn_fwd[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    # Flash Attention v2 反向传播完整流程
    # ----------
    # 输入: Q, K, V, O, dO (来自上游的梯度), L (前向保存的 logsumexp)  
    # 输出: dQ, dK, dV
    # ----------
    # 预处理D = rowsum(dO ⊙ O)  # 论文4行
    # 外层循环: for j = 1 to Tc (遍历 K,V 块) 
    #     加载 K_j, V_j 到 SRAM
    #     初始化 dK_j = 0, dV_j = 0
    #     内层循环: for i = 1 to Tr (遍历 Q 块)
    #         重计算 S,P: S=QK^T, P=exp(S-L)  # 论文10、11行
    #         dV = P^T @ dO  # 论文12行
    #         计算 dP,dS: dP=dO@V^T, dS=P⊙(dP-D)  # 论文13、14行
    #         dQ = dS @ K, 写回dQ  # 论文15行
    #         dK = dS^T @ Q  # 论文16行
    #     写回dK, dV
    # 返回 dQ, dK, dV

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128  # 预处理阶段的块大小 
        NUM_WARPS, NUM_STAGES = 4, 5
        # BLOCK_M1: dK, dV 计算时 Q 块的大小
        # BLOCK_N1: dK, dV 计算时 K,V 块的大小 
        # BLOCK_M2: dQ 计算时 Q 块的大小
        # BLOCK_N2: dQ 计算时 K,V 块的大小
        # 为什么 BLOCK_N1 > BLOCK_M1?                                               
        # ─────────────────────────────                                             
        #    计算 dK, dV 时, K,V 块常驻 SRAM，需要足够大以分摊加载开销                               
        #    Q 块流式加载，小一些可以减少 SRAM 压力                                                                                                      
        #             Q 方向 →                                                        
        #         ┌────┬────┬────┬────┬────┬────┬────┬────┐                          
        #         │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │  BLOCK_M1=32            
        #     K,V ├────┴────┴────┴────┴────┴────┴────┴────┤                          
        #     方  │                                       │                          
        #     向  │            BLOCK_N1 = 128             │  ← 一个 K,V 块          
        #     ↓   │            常驻 SRAM                   │                          
        #         │                                       │                          
        #         └───────────────────────────────────────┘                          
        # 为什么 BLOCK_N1 > BLOCK_M1?
        # ─────────────────────────────    
        #   计算 dQ 时, Q 块常驻 SRAM，需要足够大                               
        #   K,V 块流式加载，小一些减少压力    
        #             K,V 方向 →                                                         
        #         ┌────┬────┬────┬────┬────┬────┬────┬────┐                          
        #         │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │  BLOCK_N2=32           
        #     Q   ├────┴────┴────┴────┴────┴────┴────┴────┤                          
        #     方  │                                       │                          
        #     向  │            BLOCK_M2 = 128             │  ← 一个 Q 块          
        #     ↓   │            常驻 SRAM                   │                          
        #         │                                       │                          
        #         └───────────────────────────────────────┘                  
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        # BLK_SLICE_FACTOR: 在处理对角线块 (on-band) 时，使用更小的块
        # 在bwd kernel中会用到:
        #   MASK_BLOCK_M1 = BLOCK_M1 // BLK_SLICE_FACTOR = 32 // 2 = 16
        #   MASK_BLOCK_N2 = BLOCK_N2 // BLK_SLICE_FACTOR = 32 // 2 = 16
        # 由于对角线块(on-band)有一半是无效的，用更小的块 (16×16) 可以更精细地处理对角线块
        #  ┌──┬──┬──┬──┐                                                      
        #  │ ✓│ ×│  │  │  ← 块大小 16×16                                     
        #  │ ✓│ ✓│  │  │    更精细地处理边界                                 
        #  ├──┼──┼──┼──┤                                                      
        #  │  │  │ ✓│ ×│                                                      
        #  │  │  │ ✓│ ✓│                                                      
        #  └──┴──┴──┴──┘               
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)  # [batch, head, seq_len]
        # 预处理阶段, 计算delta = tl.sum(o * do, axis=1) # 对应论文第4行: D = rowsum(do ⊙ o)
        # 为什么需要 D? 
        # 在计算 dS 时: dS = P ⊙ (dP - D)  # 对应论文第14行
        # 推导: dS_ij = P_ij × (dP_ij - Σ_k P_ik × dP_ik) = P_ij × (dP_ij - D_i) 
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
            CAUSAL=ctx.causal,  #
        )

        return dq, dk, dv, None, None, None, None


attention = _attention.apply

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')


@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [2, 48])
@pytest.mark.parametrize("N_CTX", [128, 1024, (2 if is_hip() else 4) * 1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("warp_specialize", [False, True] if is_blackwell() else [False])
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []))
def test_op(Z, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, dtype=torch.float16):
    if mode == "fwd" and "fp16" in provider:
        pytest.skip("避免运行两次前向计算。")
    if mode == "bwd" and "fp8" in provider:
        pytest.skip("不支持使用 FP8 的反向传播。")
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # 参考实现
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    if mode == "bwd":
        dout = torch.randn_like(q)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    # triton 实现
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, causal, sm_scale, warp_specialize).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # 比较
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # AMD Instinct MI200 的已知硬件限制, 需放宽相对容差。
    # 详见 https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS = 4, 32
# 对于固定头部和批次=4，改变序列长度
configs = []
for HEAD_DIM in [64, 128]:
    for mode in ["fwd", "bwd"]:
        for causal in [True, False]:
            # 在 Hopper 上为因果前向启用 warpspec
            enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
            for warp_specialize in [False, True] if enable_ws else [False]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals=[2**i for i in range(10, 15)],
                        line_arg="provider",
                        line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                        (["flash"] if HAS_FLASH else []),
                        line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                        (["Flash-2"] if HAS_FLASH else []),
                        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                        ylabel="TFLOPS",
                        plot_name=
                        f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                        args={
                            "H": N_HEADS,
                            "BATCH": BATCH,
                            "HEAD_DIM": HEAD_DIM,
                            "mode": mode,
                            "causal": causal,
                            "warp_specialize": warp_specialize,
                        },
                    ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, warp_specialize)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # 目前仅适用于 Ampere 之后的 GPU
    bench_flash_attention.run(save_path=".", print_data=True)

