# FlashAttention 2 代码分析：Triton 实现

之前工作中对比了 FlashAttention 的 Triton/CuTile/FA3/Gluon/FA4 的性能，本文主要梳理 FlashAttention 2 在 Triton 官方教程中的实现，主要代码源自 Triton 官方教程 (OpenAI kernel team)，完整代码见 [06-fused-attention.py](https://github.com/slowlyC/ai-infra-notes/blob/main/tutorials/triton/06-fused-attention.py)

> 论文：Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023.
>
> 原理图解推荐：[图解大模型计算加速系列：FlashAttention V1/V2/V3](https://zhuanlan.zhihu.com/p/668888063)

## 1. 算法原理

### 1.1 标准 Attention 的内存问题

标准 Attention 需要完整的 N \times N 注意力矩阵：

```
S = Q × K^T          ← O(N²d) 计算, O(N²) 存储
P = softmax(S)       ← 需要两次遍历 (max + sum)
O = P × V            ← O(N²d) 计算
```

当序列长度 N 较大时，S 和 P 占用的 HBM 成为瓶颈。FlashAttention 的思路是分块（tiling）计算——只在 SRAM 中保留当前 block 的中间结果，不把 N \times N 矩阵写回 HBM。

### 1.2 Forward 原理

分块计算的难点在 Softmax：标准 Softmax 需要先遍历一次拿到全局 max 和 sum，再遍历一次计算 exp(x-max)/sum。Online Softmax 通过维护两个行统计量，让 Softmax 可以边遍历边更新。

固定 Q_i (一个 Q block)，依次遍历 K_j, V_j：

```
初始化: m_i = -∞,  ℓ_i = 0,  O_i = 0

对每个 K/V block j:
  1. S_ij = Q_i × K_j^T                    // QK GEMM
  2. m_new = max(m_i, rowmax(S_ij))        // 更新行最大值
  3. α = exp(m_i - m_new)                  // 修正因子
  4. P̃_j = exp(S_ij - m_new)               // 当前 block 的 softmax 分子
  5. ℓ_i = α × ℓ_i + rowsum(P̃_j)           // 更新行和
  6. O_i = α × O_i + P̃_j × V_j             // rescale + PV GEMM
  7. m_i = m_new

最终: O_i = O_i / ℓ_i                       // 归一化
```

修正因子 α 的作用：处理新 block 时如果出现更大的 score，之前所有 block 累加的 O_i 和 ℓ_i 都需要等比缩小。α = exp(m_old - m_new) 在 m_new > m_old 时小于 1，执行缩小操作。

**exp2 替代 exp**：代码中都用到了 `exp2`  替代 `exp` (自然底数)。GPU 硬件的 SFU 原生支持 `ex2.approx.f32` 指令，比 exp 更快。代价是 softmax_scale 需要预先转换到 log2 域：`qk_scale_log2 = qk_scale / ln(2)`。同时，编译器可以把 `x * scale - max * scale` 融合为一条 `ffma` 指令。

### 1.3 Backward Pass 原理

FA2 的反向传播不保存完整的 P 矩阵（N \times N），而是只保存一个每行一个标量的 LSE (log-sum-exp)，反向时重建 P：

```
前向保存: Q, K, V, O, LSE    其中 LSE_i = m_i + log(ℓ_i)

反向重建: P_ij = exp(S_ij × scale - LSE_i)
  其中 S_ij = Q_i × K_j^T
```

反向的梯度公式 (对应论文 Algorithm 2)：

```
预处理:    D_i = rowsum(O_i ⊙ dO_i)              // line 4
重计算:    S = Q × K^T,  P = exp(S×scale - LSE)  // line 10-11
dV = P^T × dO                                   // line 12
dP = dO × V^T                                   // line 13
dS = P ⊙ (dP - D)                               // line 14
dQ = dS × K                                     // line 15
dK = dS^T × Q                                   // line 16
```

D_i 的推导：标准 softmax 的反向为 dS_{ij} = P_{ij}(dP_{ij} - \sum_k P_{ik} dP_{ik})。展开 \sum_k P_{ik} dP_{ik}，其中 dP = dO \times V^T，利用 O = PV 可以化简为 \sum_k O_{ik} \cdot dO_{ik} = D_i。预计算 D_i 避免在内层循环中重复求和。

## 2. Triton 实现分析

### 2.1 Forward：外层 `_attn_fwd`

```python
@triton.autotune(configs=list(filter(keep, configs)),
                 key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              FP8_OUTPUT: tl.constexpr,
              STAGE: tl.constexpr,
              warp_specialize: tl.constexpr,
              IS_HOPPER: tl.constexpr):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    start_m = tl.program_id(0)               # Q block index
    off_hz = tl.program_id(1)                 # batch × heads
    off_z = off_hz // H                       # batch index
    off_h = off_hz % H                        # head index
```

**TensorDescriptor 构造**

```python
    y_dim = Z * H * N_CTX
    # 把 [batch, head, seq, dim] 视为 [y_dim, HEAD_DIM] 的 2D tensor
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM],
                                     strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM],
                                     strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    # V 的 descriptor 取决于是否 FP8 (FP8 需要转置)
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim],
                                         strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM],
                                     strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
```

Hopper (SM90+) 上使用 TensorDescriptor 走 TMA 路径，否则 fallback 到 `tl.make_tensor_descriptor` (kernel 内创建)。

**初始化和 Q 加载**

```python
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504         # sm_scale / ln(2), 转到 log2 域

    q = desc_q.load([qo_offset_y, 0])        # 加载 Q block, 整个循环中常驻 SRAM
```

Q 只加载一次，在遍历所有 K/V block 的过程中一直保留在寄存器中。

**调用内层循环与 STAGE 设计**

外层通过 STAGE 位编码把 causal mask 拆成两次调用 `_attn_fwd_inner`：

```python
# causal=True:  STAGE=3 (二进制 11)
#   STAGE & 1 → True  → 调用 STAGE=4-3=1: off-band, 全部有效
#   STAGE & 2 → True  → 调用 STAGE=2:     on-band, 需要 causal mask
#
# causal=False: STAGE=1 (二进制 01)
#   STAGE & 1 → True  → 调用 STAGE=4-1=3: 全部 K blocks, 无 mask
#   STAGE & 2 → False → 不执行
```

以一个 causal 的例子说明（BLOCK_M=4, BLOCK_N=4, N_CTX=8）:

```
         K block →   j=0        j=1
              ┌──────────┬──────────┐
  start_m=0   │ ✓✗✗✗     │ ✗✗✗✗     │  STAGE=1: lo=0, hi=0 (空)
  Q[0:4]      │ ✓✓✗✗     │          │  STAGE=2: lo=0, hi=4 (on-band)
              │ ✓✓✓✗     │          │
              │ ✓✓✓✓     │          │
              ├──────────┼──────────┤
  start_m=1   │ ✓✓✓✓     │ ✓✗✗✗     │  STAGE=1: lo=0, hi=4 (off-band)
  Q[4:8]      │ ✓✓✓✓     │ ✓✓✗✗     │  STAGE=2: lo=4, hi=8 (on-band)
              │ ✓✓✓✓     │ ✓✓✓✗     │
              │ ✓✓✓✓     │ ✓✓✓✓     │
              └──────────┴──────────┘
```

好处：off-band 路径中编译器可以完全消除 mask 相关的分支和 `tl.where` 操作。

```python
    # STAGE & 1: off-band (或 non-causal 时处理全部)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX,
                                        warp_specialize, IS_HOPPER)
    # STAGE & 2: on-band (causal mask)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX,
                                        warp_specialize, IS_HOPPER)
```

**Epilogue: 归一化 + 存储**

```python
    # LSE = m + log2(ℓ), 与 exp2 域一致
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # 存储 LSE (backward 用于重建 P)
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)

    # 存储 O
    desc_o.store([qo_offset_y, 0], acc.to(dtype))
```

**Autotune 配置**

```python
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w,
                  pre_hook=_host_descriptor_pre_hook)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in NUM_STAGES_OPTIONS    # [2, 3, 4] on Hopper
    for w in [4, 8]
]
```

搜索空间：2 × 3 × 3 × 2 = 36 种配置。`prune_invalid_configs` 过滤掉 BLOCK_M > N_CTX 和 causal 下 BLOCK_M < BLOCK_N 的无效配置。

### 2.2 Forward：内层循环 `_attn_fwd_inner`

由 `_attn_fwd` 调用，执行对一个 Q block 的所有 K/V block 遍历，是 Online Softmax 的实际计算逻辑。

```python
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    desc_k, desc_v,
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
```

参数说明：

- `acc`: 输出累加器 O_i，[BLOCK_M, HEAD_DIM]，FP32
- `l_i`, `m_i`: Online Softmax 的行和与行最大值，[BLOCK_M]
- `q`: 当前 Q block，已加载到寄存器
- `desc_k`, `desc_v`: K 和 V 的 TensorDescriptor (Hopper TMA) 或指针
- `STAGE`: 由外层传入，控制 causal mask 行为（1=off-band, 2=on-band, 3=全部）

**循环范围计算**

STAGE 值决定了 lo/hi，以上面的示意图为例展开：

```
对于 start_m=0 (处理 Q[0:4]):
  第一次调用 (STAGE=4-3=1): lo=0, hi=0×4=0 → 空循环, 不执行
  第二次调用 (STAGE=2):     lo=0×4=0, hi=(0+1)×4=4 → 处理 K[0:4], on-band

对于 start_m=1 (处理 Q[4:8]):
  第一次调用 (STAGE=4-3=1): lo=0, hi=1×4=4 → 处理 K[0:4], off-band, 无需 mask
  第二次调用 (STAGE=2):     lo=1×4=4, hi=(1+1)×4=8 → 处理 K[4:8], on-band
```

```python
    # STAGE=1: off-band, 从 0 到当前 Q block 起始位置
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    # STAGE=2: on-band, 只处理对角线上的 K block
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)  # 对齐 hint, 让编译器使用对齐内存指令
    # STAGE=3: non-causal, 遍历全部
    else:
        lo, hi = 0, N_CTX
```

**主循环**

```python
    offsetk_y = offset_y + lo
    # FP8 时 V 是转置存储的, 偏移计算不同
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo

    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # --- QK GEMM ---
        k = desc_k.load([offsetk_y, 0]).T       # TMA 加载 K block, .T 转置
        qk = tl.dot(q, k)                       # S = Q × K^T

        # --- Softmax ---
        if STAGE == 2:
            # on-band: 先乘 scale 再 mask (mask 值 -1e6 需要和 scaled score 同量级)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            # off-band / non-causal: 可以延迟乘 scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)                    # P̃ = exp2(S×scale - m)
        alpha = tl.math.exp2(m_i - m_ij)        # 修正因子
        l_ij = tl.sum(p, 1)                      # rowsum(P̃)

        # --- Rescale O 并累加 PV ---
        # Blackwell warp_specialize 模式下的特殊乘法优化
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]           # O_i = α × O_i

        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T    # FP8 V 转置存储
        else:
            v = desc_v.load([offsetv_y, 0])

        p = p.to(dtype)
        acc = tl.dot(p, v, acc)                  # O_i += P̃ × V

        # --- 更新状态 ---
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N

    return acc, l_i, m_i
```

on-band (STAGE==2) 和 off-band 的 softmax 计算顺序不同：on-band 先乘 scale 再做 max（因为 mask 值 -1e6 需要和 scaled 后的值比较），off-band 先取 max 再乘 scale（少一次 O(M×N) 乘法，只做 O(M) 乘法）。

### 2.3 Backward：预处理 `_attn_bwd_preprocess`

```python
@triton.jit
def _attn_bwd_preprocess(O, DO, Delta,
                         Z, H, N_CTX,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)

    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)

    # D_i = rowsum(O_i ⊙ dO_i)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)
```

### 2.4 Backward：dK/dV 计算 `_attn_bwd_dkdv`

固定一个 K/V block，遍历所有（causal 允许范围内的）Q blocks 来累加 dK 和 dV。

```python
@triton.jit
def _attn_bwd_dkdv(dk, dv,
                   Q, k, v, sm_scale,
                   DO, M, D,
                   stride_tok, stride_d,
                   H, N_CTX, BLOCK_M1: tl.constexpr,
                   BLOCK_N1: tl.constexpr,
                   HEAD_DIM: tl.constexpr,
                   start_n, start_m, num_steps,
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)

    # Q 以转置形式加载 (Q^T), 因为要做 K × Q^T = (Q × K^T)^T
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1

    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # 在 dot 之前加载 LSE, 让访存和计算流水化
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)

        # [论文 Line 10-11] 重计算 S, 重建 P
        qkT = tl.dot(k, qT)                         # S^T = K × Q^T
        pT = tl.math.exp2(qkT - m[None, :])         # P^T = exp2(S^T - LSE)

        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)

        do = tl.load(do_ptrs)

        # [论文 Line 12] dV += P^T × dO
        ppT = pT.to(tl.float16)
        dv += tl.dot(ppT, do)

        # [论文 Line 13] dP^T = V × dO^T
        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)

        # [论文 Line 14] dS^T = P^T ⊙ (dP^T - D)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)

        # [论文 Line 16] dK += dS^T × Q
        dk += tl.dot(dsT, tl.trans(qT))

        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok

    return dk, dv
```

整个函数中 S 以转置形式 `K × Q^T` 计算，因为 K 是常驻 SRAM 的那一端（外层循环按列遍历），这样避免了对 K 的转置。

### 2.5 Backward：dQ 计算 `_attn_bwd_dq`

固定一个 Q block，遍历所有 K/V blocks 来累加 dQ。

```python
@triton.jit
def _attn_bwd_dq(dq, q, K, V,
                 do, m, D,
                 stride_tok, stride_d,
                 H, N_CTX,
                 BLOCK_M2: tl.constexpr,
                 BLOCK_N2: tl.constexpr,
                 HEAD_DIM: tl.constexpr,
                 start_m, start_n, num_steps,
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2

    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)                        # K^T
        vT = tl.load(vT_ptrs)                        # V^T

        # [论文 Line 10-11] 重计算 S, 重建 P
        qk = tl.dot(q, kT)                           # S = Q × K^T
        p = tl.math.exp2(qk - m)                     # P = exp2(S - LSE)

        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)

        # [论文 Line 13-14] dP = dO × V^T,  dS = P ⊙ (dP - D)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)

        # [论文 Line 15] dQ += dS × K
        # 注意: K 已被预乘 sm_scale * RCP_LN2, 所以最终需要乘回 LN2
        dq += tl.dot(ds, tl.trans(kT))

        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok

    return dq
```

### 2.6 Backward：主 kernel `_attn_bwd`

把 dK/dV 和 dQ 合并到一个 kernel 中，同一个 thread block (pid) 分两阶段执行。以 pid=2 为例展示完整流程：

```
Causal Attention 矩阵 (N_CTX=5 块, pid=2):

         K,V 块 →    0     1     2     3     4
                  ┌─────┬─────┬─────┬─────┬─────┐
  Q 块  0         │  ◉  │  ×  │  ×  │  ×  │  ×  │
         1        │  √  │  ◉  │  ×  │  ×  │  ×  │
         2  ←pid  │  √  │  √  │  ◉  │  ×  │  ×  │  ← dQ 处理这行
         3        │  √  │  √  │  √  │  ◉  │  ×  │
         4        │  √  │  √  │  √  │  √  │  ◉  │
                  └─────┴─────┴─────┴─────┴─────┘
                                 ↑
                        dK,dV 处理这列
  ◉ = on-band (需要 mask)  √ = off-band  × = 跳过

调用 1: _attn_bwd_dkdv(MASK=True)  → K,V块2 的 on-band: Q块2 → 累加 dK2, dV2
调用 2: _attn_bwd_dkdv(MASK=False) → K,V块2 的 off-band: Q块3,4 → 累加 dK2, dV2, 写回
调用 3: _attn_bwd_dq(MASK=True)    → Q块2 的 on-band: K,V块2 → 累加 dQ2
调用 4: _attn_bwd_dq(MASK=False)   → Q块2 的 off-band: K,V块0,1 → 累加 dQ2, 写回
```

```python
@triton.jit
def _attn_bwd(Q, K, V, sm_scale, DO, DQ, DK, DV,
              M, D,
              stride_z, stride_h, stride_tok, stride_d,
              H, N_CTX,
              BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
              BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
              BLK_SLICE_FACTOR: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              CAUSAL: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996

    bhid = tl.program_id(2)                   # batch × head
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)                    # block index

    # 指针偏移到当前 batch/head
    Q += adj; K += adj; V += adj; DO += adj
    DQ += adj; DK += adj; DV += adj
    M += off_chz; D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    start_m = 0

    # on-band 用更小的块: MASK_BLOCK_M1 = BLOCK_M1 / BLK_SLICE_FACTOR
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # K, V 常驻 SRAM
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
```

**BLOCK 尺寸的不对称设计**

dK/dV 和 dQ 使用不同的 BLOCK 尺寸，常驻端用大块（分摊加载开销），流式端用小块（减少 SRAM 压力）：

```python
#    计算 dK, dV 时, K,V 块常驻 SRAM，需要足够大以分摊加载开销.Q 块流式加载，小一些可以减少 SRAM 压力
#             Q 方向 →
#         ┌────┬────┬────┬────┬────┬────┬────┬────┐
#         │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │  BLOCK_M1=32
#     K,V ├────┴────┴────┴────┴────┴────┴────┴────┤
#      ↓  │     BLOCK_N1 = 128，常驻 SRAM          │  ← 一个 K,V 块
#         └───────────────────────────────────────┘
#
#    计算 dQ 时, Q 块常驻 SRAM，需要足够大。K,V 块流式加载，小一些减少压力
#           K,V 方向 →
#         ┌────┬────┬────┬────┬────┬────┬────┬────┐
#         │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │ 32 │  BLOCK_N2=32
#       Q ├────┴────┴────┴────┴────┴────┴────┴────┤
#       ↓ │      BLOCK_M2 = 128，常驻 SRAM         │  ← 一个 Q 块
#         └───────────────────────────────────────┘
```

`BLK_SLICE_FACTOR=2` 控制 on-band 的精细分块：对角线块约一半元素被 mask 掉，用更小的块（MASK_BLOCK_M1 = 32/2 = 16）减少无效计算：

**阶段一：dK/dV**

```python
    if CAUSAL:
        # 调用 1: on-band (对角线块, 需要 mask, 用 MASK_BLOCK_M1)
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D,
                                stride_tok, stride_d, H, N_CTX,
                                MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
                                start_n, start_m, num_steps,
                                MASK=True)
        start_m += num_steps * MASK_BLOCK_M1

    # 调用 2: off-band (对角线以下, 无 mask, 用 BLOCK_M1)
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D,
                            stride_tok, stride_d, H, N_CTX,
                            BLOCK_M1, BLOCK_N1, HEAD_DIM,
                            start_n, start_m, num_steps,
                            MASK=False)

    # 写回 dK, dV
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)
```

**阶段二：dQ**

```python
    # 同一个 pid 还负责计算一个 Q block 的 dQ
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m = tl.load(M + offs_m)[:, None]

    if CAUSAL:
        # on-band: 先从右往左确定 start_n, kernel 内部还是从左到右遍历
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V, do, m, D,
                          stride_tok, stride_d, H, N_CTX,
                          BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
                          start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
                          MASK=True)
        end_n -= num_steps * MASK_BLOCK_N2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    # off-band
    dq = _attn_bwd_dq(dq, q, K, V, do, m, D,
                      stride_tok, stride_d, H, N_CTX,
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,
                      start_m, start_n, num_steps,
                      MASK=False)

    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2                                 # 补偿 exp2 域的 scale
    tl.store(dq_ptrs, dq)
```

dQ 最终乘 `LN2`：backward 入口处 K 被预乘了 `sm_scale × RCP_LN2`（`arg_k = k * sm_scale * RCP_LN2`），这样 kernel 内的 QK dot 直接得到 exp2 域的 score，省去 scale 乘法。但 dQ 的公式是 `dQ = dS × K × sm_scale`，预乘把 `sm_scale / ln(2)` 融进了 K，所以最终要乘 `LN2 = ln(2)` 来补偿。

### 2.7 autograd 集成：`_attention`

**Forward**

```python
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        HEAD_DIM_K = q.shape[-1]
        o = torch.empty_like(q)
        stage = 3 if causal else 1                 # STAGE 编码

        # Hopper + warpspec 用 device_descriptor, 否则用 host_descriptor
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], ...)
            # FP8 时 V 转置存储: shape=[HEAD_DIM, y_dim]
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[N_CTX, 1], ...)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], ...)
            ...
        else:
            desc_q, desc_k, desc_v, desc_o = q, k, v, o   # fallback

        # Blackwell warp_specialize 模式限制寄存器数量
        if is_blackwell() and warp_specialize:
            extra_kern_args["maxnreg"] = 168 if HEAD_DIM_K == 128 else 80

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), ...)  # LSE
        _attn_fwd[grid](sm_scale, M, Z, H, desc_q, desc_k, desc_v, desc_o, ...)

        ctx.save_for_backward(q, k, v, o, M)
        return o
```

TensorDescriptor 的两种模式：SM90+ 支持 host descriptor（在 CPU 端创建，通过参数传入），否则 fallback 到 kernel 内用 `tl.make_tensor_descriptor` 创建。Blackwell 上 `maxnreg` 限制每个 thread 的寄存器数量，防止 warp_specialize 模式下寄存器溢出。

**Backward**

```python
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors

        # K 预乘 sm_scale * RCP_LN2, 让 kernel 内 QK dot 直接在 exp2 域
        RCP_LN2 = 1.4426950408889634
        arg_k = k * (ctx.sm_scale * RCP_LN2)

        # 预处理: Delta = rowsum(O ⊙ dO)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](o, do, delta, ...)

        # 主 kernel: dK/dV + dQ (同一个 kernel, 分两阶段)
        _attn_bwd[grid](q, arg_k, v, ctx.sm_scale, do, dq, dk, dv, M, delta,
                        ..., BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32,
                        BLK_SLICE_FACTOR=2, ...)
        return dq, dk, dv, None, None, None, None
```

反向传播的完整流程：

```
Step 1: _attn_bwd_preprocess → Delta[i] = rowsum(O[i] ⊙ dO[i])      [论文 Line 4]
Step 2: _attn_bwd (每个 pid 执行两阶段):
  阶段一: _attn_bwd_dkdv → 固定 K/V, 遍历 Q, 累加 dK/dV              [论文 Line 10-16]
  阶段二: _attn_bwd_dq   → 固定 Q, 遍历 K/V, 累加 dQ                 [论文 Line 10-15]
```

K 预乘 `sm_scale * RCP_LN2` 的好处：kernel 内 `exp2(QK^T - LSE)` 中的 QK^T 已经是 exp2 域的 scaled score，不需要再单独乘 scale，减少了一次 O(M×N) 的乘法。

## 3. 总结

Triton 版本实现的是标准 FA2 路线：forward 用 Online Softmax 避免一次性计算完整的 N×N 的 attention matrix，backward 只保存 LSE，再通过 QK 重计算恢复 P。代码里比较值得关注的是两处工程化处理：forward 用 STAGE 把 causal mask 拆成 off-band 和 on-band 两段，让无 mask 路径在编译期变得更干净；backward 把 dK/dV 和 dQ 放进同一个主 kernel，同一个 pid 先处理列方向的 dK/dV，再处理行方向的 dQ，减少 kernel launch。
