# FlashAttention 代码分析：CuTile 实现

上篇文章 [FlashAttention 代码分析: Triton 实现](https://zhuanlan.zhihu.com/p/2011582362362864169) 分析了 Triton 的FlashAttention 实现。本文接着来分析 CuTile 的 FlashAttention 实现。其 forward 基于 [TileGym](https://github.com/NVIDIA/TileGym)，backward 是笔者自己实现，用以性能测试，完整代码见 [06-fused-attention.py](https://github.com/slowlyC/ai-infra-notes/blob/main/tutorials/cutile/06-fused-attention.py)

## 1. 算法原理

CuTile 版本的算法和 Triton 版本相同，完整推导参考 [Triton 篇](FlashAttention2_Triton.md) 的「1. 算法原理」 部分，这里仅保留必要原理于流程。

forward 过程不计算完整的 $N \times N$ attention matrix，而是固定一个 Q block，流式遍历 K/V blocks，在 SRAM 中维护每行的最大值、指数和与输出累加器：

```
for each Q block:
  m_i = -∞,  l_i = 0,  acc = 0
  for each visible K/V block:
    S = Q × K^T
    m_new = max(m_i, rowmax(S))
    alpha = exp(m_i - m_new)
    P = exp(S - m_new)
    l_i = alpha * l_i + rowsum(P)
    acc = alpha * acc + P × V
    m_i = m_new
  O = acc / l_i
  LSE = m_i + log(l_i)
```

实际实现里用 `exp2` 替代 `exp`，所以 scale 会先转到 log2 域，后面代码里的 `qk_scale * INV_LOG_2` 就对应这一步。

backward 不保存 P，只保存 LSE。backward 时重新计算 `QK^T`，用 LSE 恢复 softmax 概率，再按 FA2 的公式计算 dV、dK、dQ：

```
Delta = rowsum(O * dO)
P = exp(QK^T * scale - LSE)
dV = P^T × dO
dS = P * (dO × V^T - Delta)
dQ = dS × K
dK = dS^T × Q
```

## 2. CuTile 实现分析

### 2.1 Forward：`fmha_kernel`

CuTile 的前向只有一个 kernel 函数，结构更扁平。同时输出注意力结果 O 和 LSE (log-sum-exp)，LSE 在反向传播中用于从 Q、K 重建 softmax 概率 P，避免存储完整的 attention matrix。

```python
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
```

参数说明：

- `Q`: [batch, num_heads, q_len, head_dim]
- `K`: [batch, num_kv_heads, kv_len, head_dim]（GQA 时 num_kv_heads < num_heads）
- `V`: [batch, num_kv_heads, kv_len, head_dim]
- `Out`: [batch, num_heads, q_len, head_dim]
- `LSE`: [batch, num_heads, q_len]，反向传播用
- `input_pos`: 序列偏移，decode 阶段 causal mask 的起始位置（prefill 时为 0）
- `QUERY_GROUP_SIZE`: GQA 中每组 query head 共享的 KV head 数
- `EVEN_K`: K 序列长度能否被最大 TILE_N 整除，决定是否需要边界检查

**Block 索引和 GQA**

```python
    bid_x = ct.bid(0)                              # Q block index
    bid_y = ct.bid(1)                              # batch × heads
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE        # GQA: 映射到 KV head

    qk_scale = qk_scale * INV_LOG_2                # 转到 log2
```

CuTile 版本原生支持 GQA：Q 的 head index 除以 `QUERY_GROUP_SIZE` 得到 KV head index，多个 query head 共享同一组 KV。后续加载 K/V 时用 `off_kv_h` 而非 `head_idx`。Triton 版本不支持 GQA。

**初始化**

```python
    # 当前 block 处理的 Q 行的全局位置
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m += input_pos                            # decode 阶段需要加偏移
    offs_m = offs_m[:, None]                       # [TILE_M, 1], 用于与 [1, TILE_N] 广播比较

    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)[None, :]  # [1, TILE_N]

    # Online softmax 累加器 (float32 保证数值稳定性)
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)   # 行最大值
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)         # 指数和
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)    # 加权 V 累加

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
```

`ct.load` 的 index 是 4D tile 索引（不是元素偏移），shape 指定 tile 大小，底层通过 TMA descriptor 完成地址计算和数据搬运。

**Causal Mask 的循环范围优化**

与 Triton 的 STAGE 两次调用方案不同，CuTile 用一个统一循环 + `mask_start` 条件判断。`mask_start` 之前的 K block 全部有效（不需要 mask），之后才需要应用 causal mask。

```python
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
        mask_start = k_seqlen // TILE_N              # 永远不触发 mask
```

**主循环**

```python
    for j in range(0, Tc):
        # K 通过 TMA order=(0,1,3,2) 在搬运时转置
        # 原始布局 [batch, head, N, D], order 交换最后两维 → 加载为 [D, N]
        # TMA 的硬件转置避免了寄存器级别的 permute 开销
        k = ct.load(K, index=(batch_idx, off_kv_h, 0, j),
                    shape=(1, 1, TILE_D, TILE_N),
                    order=(0, 1, 3, 2),
                    latency=2).reshape((TILE_D, TILE_N))

        # QK GEMM: qk = Q × K^T, shape [TILE_M, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # 仅对 j >= mask_start 的 block 应用 causal mask (更早的 block 全部有效)
        # offs_m [TILE_M, 1] >= offs_n [1, TILE_N] 广播为下三角 mask
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)    # 序列长度边界检查
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)      # 下三角
            mask = ct.where(mask, 0.0, -math.inf)    # bool → additive mask
            qk += mask

        # --- Online Softmax ---
        # 优化: 先对 qk 做 reduce_max, 再乘 scale, 减少乘法次数
        #   标准写法: m = max(m, rowmax(qk * scale))  → O(M*N) 次乘法
        #   优化写法: m = max(m, rowmax(qk) * scale)  → O(M) 次乘法
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij
        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)

        # 用新旧最大值之差修正之前的累加量
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # PV GEMM: acc += P × V
        v = ct.load(V, index=(batch_idx, off_kv_h, j, 0),
                    shape=(1, 1, TILE_N, TILE_D),
                    latency=4).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij
```

几个实现细节：

- **TMA `order` 转置**：`order=(0,1,3,2)` 交换 K 的最后两维，在 TMA 搬运阶段就完成转置（HBM [N,D] → SRAM [D,N]），避免了像 Triton 那样在寄存器中做 `.T`
- `**latency` 参数**：控制 TMA prefetch 距离。K 用 `latency=2`（很快就要用），V 用 `latency=4`（QK GEMM 之后才用，有更多时间预取）
- `**flush_to_zero=True`**：exp2 和 alpha 都启用了 denormal flush，极小的非规格化浮点数直接变 0，避免 denormal 处理导致的性能下降
- `**EVEN_K` 边界检查**：K 序列长度不能被 TILE_N 整除时，最后一个 block 需要检查越界。`EVEN_K` 在入口函数中根据 `k_len % max_tile_n == 0` 计算，整除时可以跳过检查

**Epilogue**

```python
    # LSE = m + log2(ℓ), 与 exp2 域的 softmax 一致
    # 反向时通过 P = exp2(QK * scale_log2 - LSE) 重建 softmax 概率
    lse = m_i + ct.log2(l_i)
    lse = lse.reshape((1, 1, TILE_M)).astype(ct.float32)
    ct.store(LSE, index=(batch_idx, head_idx, bid_x), tile=lse)

    # O = acc / ℓ, 使用近似除法 (rcp.approx + 乘法)
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)
```

`ct.truediv` 的 `rounding_mode=RMd.APPROX` 使用硬件的 `rcp.approx` 指令计算 1/ℓ 再乘以 acc，比精确除法快。LSE 存储后在反向传播中复用：`P = exp2(QK * scale_log2 - LSE)` 可以直接从 Q、K 和 LSE 重建完整的 softmax 概率，无需保存 N×N 的 P 矩阵。

### 2.2 Backward：预处理 `fmha_bwd_preprocess_kernel`

对应论文 Algorithm 2 的 line 4，计算 Delta[i] = rowsum(O[i] * dO[i])。Delta 在后续 dK/dV/dQ 计算中用于 dS = P * (dP - Delta) 项，预计算避免在内层循环中重复求和。

```python
@ct.kernel(occupancy=4)
def fmha_bwd_preprocess_kernel(O, DO, Delta,
                               TILE_D: ConstInt, H: ConstInt, N_CTX: ConstInt,
                               TILE_M: ConstInt):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    o = ct.load(O, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D), latency=2
                ).reshape((TILE_M, TILE_D)).astype(ct.float32)

    do = ct.load(DO, index=(batch_idx, head_idx, bid_x, 0),
                 shape=(1, 1, TILE_M, TILE_D), latency=2
                 ).reshape((TILE_M, TILE_D)).astype(ct.float32)

    # D_i = rowsum(O_i ⊙ dO_i)
    delta = ct.sum(o * do, axis=-1, keepdims=False)
    delta = delta.reshape((1, 1, TILE_M))
    ct.store(Delta, index=(batch_idx, head_idx, bid_x), tile=delta)
```

`occupancy=4` 提示编译器每个 SM 调度 4 个 block。这个 kernel 计算量很小（只做逐元素乘法 + reduce），瓶颈在访存，高 occupancy 可以掩盖访存延迟。

### 2.3 Backward：dK/dV 计算 `fmha_bwd_dkdv_kernel`

每个 block 拥有一个 K/V tile (TILE_N 个 token)，遍历所有（causal 允许范围内的）Q tile 来累加 dK 和 dV。从 LSE 重建 softmax 概率 P，避免存储完整的 attention matrix。

数学公式：

```
P = softmax(QK^T × scale)    — 从 LSE 重建
dV = P^T × dO
dS = P ⊙ (dO × V^T - Delta)
dK = dS^T × Q × scale
```

```python
@ct.kernel(occupancy=2)
def fmha_bwd_dkdv_kernel(Q, K, V, DO, LSE, Delta, DK, DV,
                         qk_scale: float,
                         TILE_D: ConstInt, H: ConstInt, N_CTX: ConstInt,
                         TILE_M: ConstInt, TILE_N: ConstInt,
                         CAUSAL: ConstBool):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    start_n = bid_x * TILE_N
    qk_scale_log2 = qk_scale * INV_LOG_2

    # K 通过 TMA order= 在搬运时转置为 kT [TILE_D, TILE_N]
    kT = ct.load(K, index=(batch_idx, head_idx, 0, bid_x),
                 shape=(1, 1, TILE_D, TILE_N),
                 order=(0, 1, 3, 2), latency=2).reshape((TILE_D, TILE_N))
    # V 需要两种形式: v [TILE_N, TILE_D] 用于 dV = P^T @ dO,
    #                vT [TILE_D, TILE_N] 用于 dP = dO @ V^T
    # 所以先正常加载 v, 再 permute 得到 vT
    v = ct.load(V, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_N, TILE_D), latency=2).reshape((TILE_N, TILE_D))
    vT = ct.permute(v, (1, 0))

    dk = ct.full((TILE_N, TILE_D), 0.0, dtype=ct.float32)
    dv = ct.full((TILE_N, TILE_D), 0.0, dtype=ct.float32)

    # Causal: 只遍历 q_pos >= start_n 的 Q blocks
    if CAUSAL:
        start_m_idx = start_n // TILE_M
    else:
        start_m_idx = 0
    Tr = ct.cdiv(N_CTX, TILE_M)

    for i in range(start_m_idx, Tr):
        start_m = i * TILE_M
        offs_m = start_m + ct.arange(TILE_M, dtype=ct.int32)[:, None]

        q = ct.load(Q, index=(batch_idx, head_idx, i, 0),
                    shape=(1, 1, TILE_M, TILE_D), latency=2).reshape((TILE_M, TILE_D))
        do = ct.load(DO, index=(batch_idx, head_idx, i, 0),
                     shape=(1, 1, TILE_M, TILE_D), latency=4).reshape((TILE_M, TILE_D))
        lse = ct.load(LSE, index=(batch_idx, head_idx, i),
                      shape=(1, 1, TILE_M)).reshape((TILE_M, 1))
        Di = ct.load(Delta, index=(batch_idx, head_idx, i),
                     shape=(1, 1, TILE_M)).reshape((TILE_M, 1))

        # 从 LSE 重建 softmax 概率: P = exp2(QK^T × scale_log2 - LSE)
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, kT, qk)
        p = ct.exp2(qk * qk_scale_log2 - lse, flush_to_zero=True)

        if CAUSAL:
            offs_n_row = start_n + ct.arange(TILE_N, dtype=ct.int32)[None, :]
            mask = offs_m >= offs_n_row
            p = ct.where(mask, p, 0.0)

        # dV += P^T × dO
        pT = ct.permute(p, (1, 0)).astype(DO.dtype)
        dv = ct.mma(pT, do, dv)

        # dS = P ⊙ (dO × V^T - Delta)
        dp = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        dp = ct.mma(do, vT, dp)
        ds = p * (dp - Di)

        # dK += dS^T × Q
        dsT = ct.permute(ds, (1, 0)).astype(Q.dtype)
        dk = ct.mma(dsT, q, dk)

    # 最终 dK 需乘 scale
    dk = (dk * qk_scale).reshape((1, 1, TILE_N, TILE_D)).astype(DK.dtype)
    dv = dv.reshape((1, 1, TILE_N, TILE_D)).astype(DV.dtype)
    ct.store(DK, index=(batch_idx, head_idx, bid_x, 0), tile=dk)
    ct.store(DV, index=(batch_idx, head_idx, bid_x, 0), tile=dv)
```

与 Triton 版本的差异：

- 独立 kernel，不和 dQ 共享 thread block
- 没有 on-band/off-band 拆分，统一循环内条件判断
- `V^T` 通过 `ct.permute` 显式转置。这里不能像前向 kernel 那样用 TMA `order` 直接转置加载，因为 dV = P^T × dO 需要原始形式的 V（实际上 dV 的计算不直接用 V，但 dP = dO × V^T 需要 V^T）——同一个 V 数据需要两种布局
- `latency` 参数：do 使用 `latency=4`（更大的预取距离），因为 do 在 QK GEMM 和 P 重建之后才用到

### 2.4 Backward：dQ 计算 `fmha_bwd_dq_kernel`

每个 block 拥有一个 Q tile (TILE_M 个 token)，遍历所有（causal 允许范围内的）K/V tile 来累加 dQ。

数学公式：

```
P = exp2(QK^T × scale_log2 - LSE)   — 从 LSE 重建
dS = P ⊙ (dO × V^T - Delta)
dQ = dS × K × scale
```

```python
@ct.kernel(occupancy=2)
def fmha_bwd_dq_kernel(Q, K, V, DO, LSE, Delta, DQ,
                       qk_scale: float,
                       TILE_D: ConstInt, H: ConstInt, N_CTX: ConstInt,
                       TILE_M: ConstInt, TILE_N: ConstInt,
                       CAUSAL: ConstBool):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    qk_scale_log2 = qk_scale * INV_LOG_2

    start_m = bid_x * TILE_M
    offs_m = start_m + ct.arange(TILE_M, dtype=ct.int32)[:, None]

    # Q, dO, LSE, Delta 常驻 SRAM, 整个循环中不变
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D), latency=2).reshape((TILE_M, TILE_D))
    do = ct.load(DO, index=(batch_idx, head_idx, bid_x, 0),
                 shape=(1, 1, TILE_M, TILE_D), latency=2).reshape((TILE_M, TILE_D))
    lse = ct.load(LSE, index=(batch_idx, head_idx, bid_x),
                  shape=(1, 1, TILE_M)).reshape((TILE_M, 1))
    Di = ct.load(Delta, index=(batch_idx, head_idx, bid_x),
                 shape=(1, 1, TILE_M)).reshape((TILE_M, 1))

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

        # K 和 V 都通过 TMA order=(0,1,3,2) 转置加载
        # 这里 V 也可以用 TMA 直接加载为 vT, 因为 dQ kernel 只需要 V^T
        # (对比 dK/dV kernel 中 V 需要两种形式, 所以那里必须 load + permute)
        kT = ct.load(K, index=(batch_idx, head_idx, 0, j),
                     shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2), latency=2).reshape((TILE_D, TILE_N))
        vT = ct.load(V, index=(batch_idx, head_idx, 0, j),
                     shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2), latency=4).reshape((TILE_D, TILE_N))

        # 从 LSE 重建 P
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, kT, qk)
        p = ct.exp2(qk * qk_scale_log2 - lse, flush_to_zero=True)

        if CAUSAL:
            mask = offs_m >= offs_n
            p = ct.where(mask, p, 0.0)

        # dS = P ⊙ (dO × V^T - Delta)
        dp = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        dp = ct.mma(do, vT, dp)
        ds = p * (dp - Di)

        # dQ += dS × K (先 permute kT 回 k, 再做 mma)
        k = ct.permute(kT, (1, 0))
        ds_casted = ds.astype(K.dtype)
        dq = ct.mma(ds_casted, k, dq)

    # 最终 dQ 需乘 scale
    dq = (dq * qk_scale).reshape((1, 1, TILE_M, TILE_D)).astype(DQ.dtype)
    ct.store(DQ, index=(batch_idx, head_idx, bid_x, 0), tile=dq)
```

dQ kernel 中 V 也通过 TMA `order` 参数直接转置加载为 `vT`，不需要先加载 V 再 `ct.permute`。这比 dK/dV kernel 中对 V 的处理更高效——dK/dV kernel 需要 V 的原始形式（dV 的形状是 [TILE_N, TILE_D]），同时需要 V^T 来做 `dP = dO × V^T`，所以那里必须两种形式都有，只能 load 原始 V + permute 得到 V^T。

### 2.5 入口函数与 Autotune

`**tile_prefill_fmha` 入口**

前向 kernel 通过 `tile_prefill_fmha` 入口调用，处理 GQA 和 EVEN_K 逻辑：

```python
def tile_prefill_fmha(q, k, v, sm_scale, is_causal=True):
    batch_size, num_heads, q_len, hidden_size = q.shape
    _, num_head_kv, k_len, _ = k.shape
    query_group_size = num_heads // num_head_kv       # GQA group size

    max_tile_n = max(cfg.kwargs['TILE_N'] for cfg in _fmha_autotune_configs())
    EVEN_K = (k_len % max_tile_n) == 0               # 是否需要边界检查
    o, lse = cutile_autotune_fmha(q, k, v, o, sm_scale, input_pos=0, ...)
    return o, lse
```

`EVEN_K` 根据 K 序列长度是否能被所有候选 TILE_N 中的最大值整除来决定。整除时 kernel 内部可以跳过边界检查。

**Autotune 配置**

CuTile 使用自定义 Autotuner（见 `autotuner.py`），为不同 GPU 架构提供独立配置：

```python
# 前向 kernel 配置 (sm100/Blackwell)
configs = [
    Config(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=1),
    Config(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2),
    Config(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=2),
    Config(TILE_M=128, TILE_N=128, num_ctas=2, occupancy=2),
    # ... 共 11 种配置
    Config(TILE_M=32, TILE_N=32, num_ctas=1, occupancy=1),
]

Config(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2)
```

与 Triton autotune 的区别：CuTile 的 Config 额外包含 `num_ctas`（CTA cluster 大小，Hopper+）和 `occupancy`（每 SM 的 block 数），这些参数给编译器提供了更多调度信息。反向 dK/dV 和 dQ kernel 共用一套配置，各自独立 autotune。

**autograd 集成**

```python
class CuTileFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal, sm_scale):
        o, lse = tile_prefill_fmha(q, k, v, sm_scale, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, o, lse)     # 保存 Q,K,V,O,LSE
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal
        _, num_heads, seq_len, hidden_size = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(lse)

        # Step 1: Delta[i] = rowsum(O[i] * dO[i])
        cutile_fmha_bwd_preprocess(o, do, delta)

        # Step 2: dK, dV (每个 block 拥有一个 K/V tile, 遍历 Q)
        cutile_autotune_fmha_bwd_dkdv(
            q, k, v, do, lse, delta, dk, dv,
            sm_scale, hidden_size, num_heads, seq_len, is_causal)

        # Step 3: dQ (每个 block 拥有一个 Q tile, 遍历 K/V)
        cutile_autotune_fmha_bwd_dq(
            q, k, v, do, lse, delta, dq,
            sm_scale, hidden_size, num_heads, seq_len, is_causal)

        return dq, dk, dv, None, None
```

反向分三步独立 kernel launch，每步只计算一种梯度，各自有独立的 grid 和 autotune 配置。对比 Triton 把 dK/dV 和 dQ 合进同一个 kernel（同一个 thread block 分两阶段执行），CuTile 的方式更直观，代价是多了 kernel launch 开销（实际影响不大，因为每个 kernel 的计算量足够）。

## 3. 总结

CuTile 版本实现的算法仍然是 FA2 Online Softmax + backward 重计算，其思路与 Triton 版本相同。实测下来 CuTile 版本的 forward 和 backward 在 Blackwell 上的性能都优于 Triton 版本，甚至在某些 size 下逼近 Gluon 的性能，尤其在 head_dim=256 和较大序列长度时差距更明显。这可能与 CuTile 编译器对 TMA 和 tile 调度的底层优化有关。
