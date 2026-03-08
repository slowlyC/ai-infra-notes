# 代码来源：https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/attention.py
# 反向传播参考: Flash Attention v2 (https://tridao.me/publications/flash2/flash2.pdf)

import torch
import cuda.tile as ct
from cuda.tile import RoundingMode as RMd
import math

INV_LOG_2 = 1.0 / math.log(2)  # 用于 exp2 替代 exp 的常量: exp(x) = exp2(x / ln2)
LN2 = 0.6931471824645996       # ln(2), 将 log2 值转回自然对数时使用

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

@ct.kernel()
def fmha_kernel(Q, K, V, Out, LSE,
                qk_scale: float,
                input_pos: int,
                TILE_D: ConstInt,
                H: ConstInt,
                TILE_M: ConstInt,
                TILE_N: ConstInt,
                QUERY_GROUP_SIZE: ConstInt,
                CAUSAL: ConstBool,
                EVEN_K: ConstBool):
    """
    Fused Multi-Head Attention 前向 kernel, 采用 online softmax 实现。

    同时输出注意力结果 O 和 LSE (log-sum-exp), LSE 在反向传播中用于
    从 Q, K 重建 softmax 概率 P, 避免存储完整的 attention matrix。

    参数:
      Q: [batch, num_heads, q_len, head_dim]
      K: [batch, num_kv_heads, kv_len, head_dim]
      V: [batch, num_kv_heads, kv_len, head_dim]
      Out: [batch, num_heads, q_len, head_dim]   -- 输出
      LSE: [batch, num_heads, q_len]              -- log-sum-exp
      qk_scale: softmax 缩放因子, 一般为 1/sqrt(d)
      input_pos: 序列偏移 (用于 decode 阶段的 causal mask 计算)
      QUERY_GROUP_SIZE: GQA 中每组 query head 共享的 KV head 数
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    # GQA: 多个 query head 共享同一组 KV head
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # 将 qk_scale 转换为 log2 域: scale * (1/ln2)
    # 后续 softmax 统一使用 exp2 替代 exp, 减少指令开销
    qk_scale = qk_scale * INV_LOG_2

    # 当前 block 处理的 Q 行的全局位置
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1], 用于与 [1, TILE_N] 广播比较

    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Online softmax 累加器 (float32 保证数值稳定性)
    # m_i: 当前 row-wise 最大值 (初始化为 -inf)
    # l_i: 指数和 (初始化为 0)
    # acc: 加权 V 的累加
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # 加载当前 block 负责的 Q tile
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)
    ).reshape((TILE_M, TILE_D))

    # --- Causal mask 优化: 跳过完全被 mask 的 K block ---
    #
    # 示例: input_pos=0, TILE_M=4, TILE_N=4, k_seqlen=8
    #
    #              K block    j=0          j=1
    #                   ┌──────────┬──────────┐
    #                   │  K[0:4]  │  K[4:8]  │
    #   Q block         │          │          │
    #   ┌───────────────┼──────────┼──────────┤
    #   │  bid_x=0      │ ✓✗✗✗     │ ✗✗✗✗     │  ← j=1 全部被 mask,
    #   │  Q[0:4]       │ ✓✓✗✗     │ ✗✗✗✗     │    直接跳过
    #   │               │ ✓✓✓✗     │ ✗✗✗✗     │
    #   │               │ ✓✓✓✓     │ ✗✗✗✗     │
    #   ├───────────────┼──────────┼──────────┤
    #   │  bid_x=1      │ ✓✓✓✓     │ ✓✗✗✗     │  ← j=1 部分有效,
    #   │  Q[4:8]       │ ✓✓✓✓     │ ✓✓✗✗     │    需要应用 mask
    #   │               │ ✓✓✓✓     │ ✓✓✓✗     │
    #   │               │ ✓✓✓✓     │ ✓✓✓✓     │
    #   └───────────────┴──────────┴──────────┘
    #
    # bid_x=0: Tc=1 (仅 j=0), mask_start=0 (所有 K block 都需要 mask)
    # bid_x=1: Tc=2 (j=0 和 j=1), mask_start=1 (j=0 全部有效, j=1 需要 mask)
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        # mask_start: 从这个 K block 开始才需要应用 causal mask
        # (之前的 K block 位置 < Q 位置, 全部有效)
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        # Tc: 只迭代至少有一个有效位置的 K block 数量
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # --- 主循环: 遍历 K/V tile ---
    for j in range(0, Tc):
        # 通过 order=(0,1,3,2) 在 TMA 搬运时转置 K
        # TMA 的硬件转置避免了寄存器级别的 permute 开销
        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))

        # QK^T = Q @ K^T
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # 仅对 j >= mask_start 的 block 应用 causal mask (更早的 block 全部有效)
        # offs_m [TILE_M, 1] >= offs_n [1, TILE_N] 广播为下三角 mask
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            # bool mask → additive mask: False 位置加 -inf
            mask = ct.where(mask, 0.0, -math.inf)
            qk += mask

        # --- Online softmax 更新 ---
        # 优化: 先对 qk 做 reduce_max, 再乘 scale, 减少 O(M*N) 次乘法到 O(M)
        #   标准写法: m = max(m, rowmax(qk * scale));  p = exp(qk * scale - m)
        #   优化写法: m = max(m, rowmax(qk) * scale);  p = exp2(qk * scale - m)
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)

        # 用新旧最大值之差修正之前的累加量
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # 加载 V 并累加 P @ V
        v = ct.load(
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # LSE = m + log2(l), 与 exp2 域的 softmax 一致;
    # 反向时通过 P = exp2(QK * scale_log2 - LSE) 重建 softmax 概率
    lse = m_i + ct.log2(l_i)
    lse = lse.reshape((1, 1, TILE_M)).astype(ct.float32)
    ct.store(LSE, index=(batch_idx, head_idx, bid_x), tile=lse)

    # 最终输出 O = acc / l_i
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


@ct.kernel(occupancy=4)
def fmha_bwd_preprocess_kernel(
    O, DO, Delta,
    TILE_D: ConstInt, H: ConstInt, N_CTX: ConstInt, TILE_M: ConstInt,
):
    """
    反向预处理: Delta[i] = rowsum(O[i] * dO[i])。

    对应 Flash Attention v2 论文 Algorithm 2, line 4。
    Delta 用于后续 dK/dV/dQ 计算中的 dS = P * (dP - Delta) 项。
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    o = ct.load(
        O, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D), latency=2
    ).reshape((TILE_M, TILE_D)).astype(ct.float32)

    do = ct.load(
        DO, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D), latency=2
    ).reshape((TILE_M, TILE_D)).astype(ct.float32)

    delta = ct.sum(o * do, axis=-1, keepdims=False)
    delta = delta.reshape((1, 1, TILE_M))
    ct.store(Delta, index=(batch_idx, head_idx, bid_x), tile=delta)


@ct.kernel(occupancy=2)
def fmha_bwd_dkdv_kernel(
    Q, K, V, DO, LSE, Delta, DK, DV,
    qk_scale: float,
    TILE_D: ConstInt, H: ConstInt, N_CTX: ConstInt,
    TILE_M: ConstInt, TILE_N: ConstInt,
    CAUSAL: ConstBool,
):
    """
    计算 dK 和 dV。

    每个 block 拥有一个 K/V tile (TILE_N 个 token), 遍历所有 Q tile 累加梯度。
    Causal 模式下只遍历 q_pos >= k_pos 的 Q tile。

    数学公式:
      P = softmax(QK^T * scale)    -- 从 LSE 重建, 不存储完整 P
      dV = P^T @ dO
      dS = P * (dO @ V^T - Delta)
      dK = dS^T @ Q * scale
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    start_n = bid_x * TILE_N
    qk_scale_log2 = qk_scale * INV_LOG_2

    # K 通过 order= 在 TMA 搬运时转置; V 需要显式 permute 得到 V^T
    kT = ct.load(
        K, index=(batch_idx, head_idx, 0, bid_x), shape=(1, 1, TILE_D, TILE_N),
        order=(0, 1, 3, 2), latency=2
    ).reshape((TILE_D, TILE_N))
    v = ct.load(
        V, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_N, TILE_D), latency=2
    ).reshape((TILE_N, TILE_D))
    vT = ct.permute(v, (1, 0))

    dk = ct.full((TILE_N, TILE_D), 0.0, dtype=ct.float32)
    dv = ct.full((TILE_N, TILE_D), 0.0, dtype=ct.float32)

    # Causal: 只遍历 q_pos >= start_n 的 Q block
    if CAUSAL:
        start_m_idx = start_n // TILE_M
    else:
        start_m_idx = 0
    Tr = ct.cdiv(N_CTX, TILE_M)

    for i in range(start_m_idx, Tr):
        start_m = i * TILE_M
        offs_m = start_m + ct.arange(TILE_M, dtype=ct.int32)[:, None]

        q = ct.load(Q, index=(batch_idx, head_idx, i, 0), shape=(1, 1, TILE_M, TILE_D), latency=2).reshape((TILE_M, TILE_D))
        do = ct.load(DO, index=(batch_idx, head_idx, i, 0), shape=(1, 1, TILE_M, TILE_D), latency=4).reshape((TILE_M, TILE_D))
        lse = ct.load(LSE, index=(batch_idx, head_idx, i), shape=(1, 1, TILE_M)).reshape((TILE_M, 1))
        Di = ct.load(Delta, index=(batch_idx, head_idx, i), shape=(1, 1, TILE_M)).reshape((TILE_M, 1))

        # 从 LSE 重建 softmax 概率 P, 避免存储完整 attention matrix
        # P = exp2(QK^T * scale_log2 - LSE)
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, kT, qk)
        p = ct.exp2(qk * qk_scale_log2 - lse, flush_to_zero=True)

        if CAUSAL:
            offs_n_row = start_n + ct.arange(TILE_N, dtype=ct.int32)[None, :]
            mask = offs_m >= offs_n_row
            p = ct.where(mask, p, 0.0)

        # dV += P^T @ dO
        pT = ct.permute(p, (1, 0)).astype(DO.dtype)
        dv = ct.mma(pT, do, dv)

        # dS = P * (dO @ V^T - Delta)
        dp = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        dp = ct.mma(do, vT, dp)
        ds = p * (dp - Di)

        # dK += dS^T @ Q
        dsT = ct.permute(ds, (1, 0)).astype(Q.dtype)
        dk = ct.mma(dsT, q, dk)

    # 最终 dK 需乘 scale
    dk = (dk * qk_scale).reshape((1, 1, TILE_N, TILE_D)).astype(DK.dtype)
    dv = dv.reshape((1, 1, TILE_N, TILE_D)).astype(DV.dtype)
    ct.store(DK, index=(batch_idx, head_idx, bid_x, 0), tile=dk)
    ct.store(DV, index=(batch_idx, head_idx, bid_x, 0), tile=dv)


@ct.kernel(occupancy=2)
def fmha_bwd_dq_kernel(
    Q, K, V, DO, LSE, Delta, DQ,
    qk_scale: float,
    TILE_D: ConstInt, H: ConstInt, N_CTX: ConstInt,
    TILE_M: ConstInt, TILE_N: ConstInt,
    CAUSAL: ConstBool,
):
    """
    计算 dQ。

    每个 block 拥有一个 Q tile (TILE_M 个 token), 遍历所有 K/V tile 累加梯度。
    Causal 模式下只遍历 k_pos <= q_pos 的 K/V tile。

    数学公式:
      P = exp2(QK^T * scale_log2 - LSE)   -- 从 LSE 重建
      dS = P * (dO @ V^T - Delta)
      dQ = dS @ K * scale
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    qk_scale_log2 = qk_scale * INV_LOG_2

    start_m = bid_x * TILE_M
    offs_m = start_m + ct.arange(TILE_M, dtype=ct.int32)[:, None]

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D), latency=2).reshape((TILE_M, TILE_D))
    do = ct.load(DO, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D), latency=2).reshape((TILE_M, TILE_D))
    lse = ct.load(LSE, index=(batch_idx, head_idx, bid_x), shape=(1, 1, TILE_M)).reshape((TILE_M, 1))
    Di = ct.load(Delta, index=(batch_idx, head_idx, bid_x), shape=(1, 1, TILE_M)).reshape((TILE_M, 1))

    dq = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Causal: 只遍历 k_pos <= start_m + TILE_M 的 K/V block
    if CAUSAL:
        end_n = ct.cdiv(start_m + TILE_M, TILE_N)
        Tc = min(end_n, ct.cdiv(N_CTX, TILE_N))
    else:
        Tc = ct.cdiv(N_CTX, TILE_N)

    for j in range(0, Tc):
        start_n = j * TILE_N
        offs_n = start_n + ct.arange(TILE_N, dtype=ct.int32)[None, :]

        # K 和 V 都通过 TMA order= 转置加载, 避免寄存器级 permute
        kT = ct.load(K, index=(batch_idx, head_idx, 0, j), shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2), latency=2).reshape((TILE_D, TILE_N))
        vT = ct.load(V, index=(batch_idx, head_idx, 0, j), shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2), latency=4).reshape((TILE_D, TILE_N))

        # 从 LSE 重建 P
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, kT, qk)
        p = ct.exp2(qk * qk_scale_log2 - lse, flush_to_zero=True)

        if CAUSAL:
            mask = offs_m >= offs_n
            p = ct.where(mask, p, 0.0)

        # dS = P * (dO @ V^T - Delta)
        dp = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        dp = ct.mma(do, vT, dp)
        ds = p * (dp - Di)

        # dQ += dS @ K
        k = ct.permute(kT, (1, 0))
        ds_casted = ds.astype(K.dtype)
        dq = ct.mma(ds_casted, k, dq)

    # 最终 dQ 需乘 scale
    dq = (dq * qk_scale).reshape((1, 1, TILE_M, TILE_D)).astype(DQ.dtype)
    ct.store(DQ, index=(batch_idx, head_idx, bid_x, 0), tile=dq)


import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from autotuner import Autotuner, Config, autotune

def _fmha_autotune_configs():
    """前向 kernel 的自动调优配置。"""
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        configs = [
            Config(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2),
        ]
    else:
        # sm100 (Blackwell)
        configs = [
            Config(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=1),
            Config(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2),
            Config(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=2),
            Config(TILE_M=128, TILE_N=128, num_ctas=2, occupancy=2),
            Config(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2),
            Config(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=1),
            Config(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=4),
            Config(TILE_M=64, TILE_N=64, num_ctas=2, occupancy=1),
            Config(TILE_M=64, TILE_N=32, num_ctas=1, occupancy=2),
            Config(TILE_M=256, TILE_N=32, num_ctas=2, occupancy=2),
            Config(TILE_M=32, TILE_N=32, num_ctas=1, occupancy=1),
        ]
    return configs


def _fmha_bwd_autotune_configs():
    """反向 dK/dV 和 dQ kernel 共用的自动调优配置。"""
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        configs = [
            Config(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2),
        ]
    else:
        # sm100 (Blackwell)
        configs = [
            Config(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2),
            Config(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=1),
            Config(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2),
            Config(TILE_M=64, TILE_N=128, num_ctas=1, occupancy=2),
            Config(TILE_M=128, TILE_N=64, num_ctas=1, occupancy=2),
            Config(TILE_M=32, TILE_N=128, num_ctas=1, occupancy=2),
            Config(TILE_M=128, TILE_N=32, num_ctas=1, occupancy=2),
            Config(TILE_M=64, TILE_N=64, num_ctas=2, occupancy=2),
            Config(TILE_M=128, TILE_N=64, num_ctas=2, occupancy=2),
        ]
    return configs


@autotune(search_space=_fmha_autotune_configs())
def cutile_autotune_fmha(
    q,
    k,
    v,
    o,
    sm_scale,
    input_pos,
    hidden_size,
    num_heads,
    query_group_size,
    is_causal,
    EVEN_K,
    autotuner: Autotuner | None = None,
):
    batch_size, _, q_len, _ = q.shape
    # LSE 用于反向传播中重建 softmax, 避免保存完整 attention matrix
    lse = torch.empty((batch_size, num_heads, q_len), dtype=torch.float32, device=q.device)
    tuned_result = autotuner(
        torch.cuda.current_stream(),
        grid_fn=lambda named_args, cfg: (
            math.ceil(q_len / cfg.TILE_M),
            batch_size * num_heads,
            1,
        ),
        kernel=fmha_kernel,
        args_fn=lambda cfg: (
            q,
            k,
            v,
            o,
            lse,
            sm_scale,
            input_pos,
            hidden_size,
            num_heads,
            cfg.TILE_M,
            cfg.TILE_N,
            query_group_size,
            is_causal,
            EVEN_K,
        ),
    )
    return o, lse


def _fmha_bwd_preprocess_configs():
    return [
        Config(TILE_M=128, num_ctas=1, occupancy=2),
    ]


@autotune(search_space=_fmha_bwd_preprocess_configs())
def cutile_fmha_bwd_preprocess(
    o, do, delta,
    autotuner: Autotuner | None = None,
):
    batch_size, num_heads, seq_len, hidden_size = o.shape
    tuned_result = autotuner(
        torch.cuda.current_stream(),
        grid_fn=lambda named_args, cfg: (
            math.ceil(seq_len / cfg.TILE_M),
            batch_size * num_heads,
            1,
        ),
        kernel=fmha_bwd_preprocess_kernel,
        args_fn=lambda cfg: (
            o, do, delta, hidden_size, num_heads, seq_len, cfg.TILE_M,
        ),
    )
    return delta


@autotune(search_space=_fmha_bwd_autotune_configs())
def cutile_autotune_fmha_bwd_dkdv(
    q, k, v, do, lse, delta,
    dk, dv,
    sm_scale,
    hidden_size,
    num_heads,
    seq_len,
    is_causal,
    autotuner: Autotuner | None = None,
):
    batch_size = q.shape[0]
    tuned_result = autotuner(
        torch.cuda.current_stream(),
        grid_fn=lambda named_args, cfg: (
            math.ceil(seq_len / cfg.TILE_N),
            batch_size * num_heads,
            1,
        ),
        kernel=fmha_bwd_dkdv_kernel,
        args_fn=lambda cfg: (
            q, k, v, do, lse, delta, dk, dv,
            sm_scale,
            hidden_size,
            num_heads,
            seq_len,
            cfg.TILE_M,
            cfg.TILE_N,
            is_causal,
        ),
    )
    return dk, dv


@autotune(search_space=_fmha_bwd_autotune_configs())
def cutile_autotune_fmha_bwd_dq(
    q, k, v, do, lse, delta,
    dq,
    sm_scale,
    hidden_size,
    num_heads,
    seq_len,
    is_causal,
    autotuner: Autotuner | None = None,
):
    batch_size = q.shape[0]
    tuned_result = autotuner(
        torch.cuda.current_stream(),
        grid_fn=lambda named_args, cfg: (
            math.ceil(seq_len / cfg.TILE_M),
            batch_size * num_heads,
            1,
        ),
        kernel=fmha_bwd_dq_kernel,
        args_fn=lambda cfg: (
            q, k, v, do, lse, delta, dq,
            sm_scale,
            hidden_size,
            num_heads,
            seq_len,
            cfg.TILE_M,
            cfg.TILE_N,
            is_causal,
        ),
    )
    return dq



def tile_prefill_fmha(q, k, v, sm_scale, is_causal=True, kernel_configs=None):
    """Prefill 阶段的 FMHA 入口, 处理 GQA 和 EVEN_K 逻辑。"""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    batch_size, num_heads, q_len, hidden_size = q.shape
    _, num_head_kv, k_len, _ = k.shape

    assert num_heads % num_head_kv == 0
    query_group_size = num_heads // num_head_kv

    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v
    o = torch.empty_like(q)

    input_pos = 0  # prefill 阶段从位置 0 开始

    max_tile_n = max(cfg.kwargs['TILE_N'] for cfg in _fmha_autotune_configs())
    EVEN_K = (k_len % max_tile_n) == 0
    o, lse = cutile_autotune_fmha(
        q, k, v, o, sm_scale, input_pos, hidden_size, num_heads, query_group_size, is_causal, EVEN_K
    )
    return o, lse


class CuTileFlashAttention(torch.autograd.Function):
    """
    PyTorch autograd 包装: cuTile Flash Attention v2 前向 + 反向。

    前向保存 Q, K, V, O, LSE 供反向使用;
    反向分三步: preprocess (计算 Delta) → dK/dV → dQ。
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal, sm_scale):
        o, lse = tile_prefill_fmha(q, k, v, sm_scale, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        _, num_heads, seq_len, hidden_size = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(lse)

        # Step 1: Delta[i] = rowsum(O[i] * dO[i])
        cutile_fmha_bwd_preprocess(o, do, delta)

        # Step 2: 计算 dK, dV (每个 block 拥有一个 K/V tile, 遍历所有 Q)
        cutile_autotune_fmha_bwd_dkdv(
            q, k, v, do, lse, delta,
            dk, dv,
            sm_scale, hidden_size, num_heads, seq_len, is_causal,
        )

        # Step 3: 计算 dQ (每个 block 拥有一个 Q tile, 遍历所有 K/V)
        cutile_autotune_fmha_bwd_dq(
            q, k, v, do, lse, delta,
            dq,
            sm_scale, hidden_size, num_heads, seq_len, is_causal,
        )

        return dq, dk, dv, None, None


def cutile_fmha(
    q,
    k,
    v,
    scaling=None,
    is_causal=True,
    **kwargs,
):
    """用户接口: cuTile Flash Attention。"""
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))
    return CuTileFlashAttention.apply(q, k, v, is_causal, scaling)


def reference_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float = None,
    is_causal: bool = True,
):
    """参考实现: 使用 PyTorch 原生 scaled_dot_product_attention。"""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=scaling
    )


def test_accuracy():
    """正确性测试: 比较 cuTile FMHA 与 PyTorch 原生实现的前向和反向结果。"""
    DEVICE = torch.cuda.current_device()
    dtype = torch.float16
    torch.manual_seed(42)

    BATCH, H, N_CTX, HEAD_DIM = 2, 32, 1024, 128
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)

    # PyTorch 参考
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out = reference_fmha(q_ref, k_ref, v_ref, scaling=sm_scale, is_causal=True)
    dout = torch.randn_like(ref_out)
    ref_out.backward(dout)
    ref_dq, ref_dk, ref_dv = q_ref.grad.clone(), k_ref.grad.clone(), v_ref.grad.clone()

    # cuTile
    cutile_out = cutile_fmha(q, k, v, scaling=sm_scale, is_causal=True)
    cutile_out.backward(dout)
    cutile_dq, cutile_dk, cutile_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # 前向对比
    fwd_match = torch.allclose(cutile_out, ref_out, atol=1e-2, rtol=0)
    print(f"test cutile fwd kernel accuracy: {'success' if fwd_match else '❌failed'}")
    if not fwd_match:
        print(f"max diff: {(cutile_out - ref_out).abs().max().item()}")

    # 反向对比
    dq_match = torch.allclose(cutile_dq, ref_dq, atol=1e-2, rtol=1e-2)
    dk_match = torch.allclose(cutile_dk, ref_dk, atol=1e-2, rtol=1e-2)
    dv_match = torch.allclose(cutile_dv, ref_dv, atol=1e-2, rtol=1e-2)
    print(f"test cutile bwd kernel accuracy:")
    print(f"dQ: {'success' if dq_match else '❌failed'}")
    if not dq_match:
        print(f"max diff: {(cutile_dq - ref_dq).abs().max().item()}")

    print(f"dK: {'success' if dk_match else '❌failed'}")
    if not dk_match:
        print(f"max diff: {(cutile_dk - ref_dk).abs().max().item()}")

    print(f"dV: {'success' if dv_match else '❌failed'}")
    if not dv_match:
        print(f"max diff: {(cutile_dv - ref_dv).abs().max().item()}")

    if fwd_match and dq_match and dk_match and dv_match:
        print("all tests passed!")
        return True
    else:
        print("some tests failed, please check the implementation!")
        return False

import triton

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

import importlib.util
import os
_spec = importlib.util.spec_from_file_location("triton_06_fused_attention", os.path.join(os.path.dirname(__file__), "../triton/06-fused-attention.py"))
_triton_attention_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_triton_attention_module)
is_blackwell = _triton_attention_module.is_blackwell
is_hopper = _triton_attention_module.is_hopper

DEVICE = torch.cuda.current_device()
TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS = 4, 32

configs = []
for HEAD_DIM in [256]:
    for mode in ["fwd", "bwd"]:
        for causal in [True,]:
            enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
            for warp_specialize in [False,] if enable_ws else [False]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals=[2**i for i in range(10, 15)],
                        line_arg="provider",
                        line_vals=["cutile-fp16"] + ["triton-fp16"] +
                        (["cutile-fp8"] if TORCH_HAS_FP8 else []) +
                        (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                        (["flash"] if HAS_FLASH else []),
                        line_names=["cuTile [FP16]"] + ["Triton [fp16]"] +
                        (["cuTile [FP8]"] if TORCH_HAS_FP8 else []) +
                        (["Triton [fp8]"] if TORCH_HAS_FP8 else []) +
                        (["Flash-2"] if HAS_FLASH else []),
                        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-"), ("pink", "-")],
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

triton_attention = _triton_attention_module.attention

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    sm_scale = 1.3

    if "triton" in provider:
        # Triton 反向 kernel 不支持 HEAD_DIM > 128 (shared memory 限制)
        if mode == "bwd" and HEAD_DIM > 128:
            return float('nan')
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        fn = lambda: triton_attention(q, k, v, causal, sm_scale, warp_specialize)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if "cutile" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            # v = v.permute(0, 1, 3, 2).contiguous()
            # v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        fn = lambda: cutile_fmha(q, k, v, scaling=sm_scale, is_causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=200) if mode == "fwd" and "fp8" in provider else triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn) if mode == "fwd" else triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # test_accuracy()
    bench_flash_attention.run(print_data=True)
