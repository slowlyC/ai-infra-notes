# Native Sparse Attention & MoBA解读

# Native Sparse Attention

论文地址：[https://arxiv.org/pdf/2502.11089](https://arxiv.org/pdf/2502.11089)

## 背景

随着LLM上下文长度的增长，注意力所需要的计算复杂度是平方复杂度的，Flash Attention实现了O(N)的访存复杂度，但是并未减少注意力的计算量，因此如何减少计算量成为关键。DeepSeek近期发布了`NSA`(Native Sparse Attention)，目的就是进一步减少训练和推理的成本。

NSA主要解决的是如何**高效进行长文本建模**（Long-Context Modeling）。NSA希望能够在超长上下文的场景下，相较于原生Full Attention，在训练和推理阶段都具有速度上的显著优势。其提出两项创新：

1. **硬件对齐系统**：优化分块稀疏注意力以提升Tensor Core利用率和内存访问效率，实现算术强度的均衡；

2. **训练感知设计**：通过高效算法和反向传播算子实现稳定的端到端训练，使NSA能够同时支持端到端训练和高效部署。

   ### Sparse Attention

   NSA主要受实验中人们观察到的**Full Attention的天然稀疏性（Sparsity）**现象启发。

   原始的Full Attention，计算的时空复杂度随着序列长度呈现**平方级别**的增长，导致整体的prefill和decoding时间严重受限于attention的计算。

   self-attention的计算公式如下：

   $Q=xW\_Q,K=xW\_K,V=xW\_V\\x\_{out}=softmax(\frac{QK^{T}}{\sqrt{h}})\cdot V\cdot W\_o + x\\$

   假设输入数据的形状为 \[b,s\] ，hidden\_size为h。

   1.  计算 Q,K,V 矩阵乘，输入和输出为 $\[b,s,h\] \times \[h,h\]\rightarrow \[b,s,h\]$，计算量为$3\*2bsh^2=6bsh^2 $

   2.  计算$QK^{T}$矩阵乘，输入和输出为 $\[b, head\\_num, s,per\\_head\\_hidden\\_size\]\times \[b,head\\_num,per\\_head\\_hidden\\_size,s\]\rightarrow \[b,head\\_num,s,s\]$，计算量为$ 2bs^2h$

   3.  计算$score\cdot V $矩阵乘，输入和形状为 $\[b,head\\_num,s,s\]\times\[b,head\\_num,s,per\\_head\\_hidden\\_size\]\rightarrow\[b,head\\_num,s,per\\_head\\_hidden\\_size\] $，计算量为$2bs^2h$

人们在实验中关注到，**softmax后的注意力分数具有非常稀疏的特性**，因为attention score的和为1，那么context 越长越 sparse。也就是说在每个位置，当前token需要聚合的信息其实很多时候只是过去有限的token集合而非全集。这就启发了去设计一些稀疏化的算法，充分利用attention稀疏的现象降低计算的复杂度。

### Native的含义

为什么要从原生开始预训练，关键的是后期应用的稀疏注意力性能退化严重, 总结以下结论是稀疏化处理后20%只能恢复70%的分数，并且稀疏化的注意力不稳定，非常脆弱。

1.  **性能退化**：后期应用稀疏性导致模型偏离预训练的优化轨迹。研究表明，即使选取最重要的20%注意力也仅能覆盖70%的总注意力分数，这使得预训练模型中的检索头等结构在推理阶段的剪枝过程中容易受损。

2.  **训练效率需求**：在现代LLM开发中，高效处理长序列训练至关重要。这包括在更长文档上进行预训练以增强模型能力，以及后续的长文本微调和强化学习等适配阶段。然而，现有稀疏注意力方法主要针对推理阶段，未能有效解决训练阶段的计算挑战。这一局限性阻碍了通过高效训练开发更强大长文本模型的进程。

## 效果

在一个27B大小的Transformer基座上使用了NSA机制，并预训练了200B token。实验结果证明了NSA设计的效率提升，同时在推理和传统任务的benchmark效果上也是无损的，甚至有些还提升了。并且在长文检索任务上，NSA也是无损的，说明NSA机制并不会显著丢失上下文内部的信息。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/eYVOL5jV1drbzlpz/img/2c2db192-9246-4955-a076-71ee248ea461.png)

**提升的原理（猜测）**：以往的Full Attention内部可能会有一些冗余的Attention，强迫模型必须attend到一些无关信息。NSA或许让模型有能力学到更好的sparse pattern，将这些无关信息丢失掉，可能能帮助模型更focus到重要的信息上。

## 具体原理

下图描述了NSA算法的几个细节，主要包括了三种稀疏化注意力操作，分别为**压缩注意力**（把控全局信息）、**选择注意力**（把控局部信息）、**滑窗注意力**（把控关联紧密的上下文信息）。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/eYVOL5jV1drbzlpz/img/3428cda7-c5bf-4f67-a0b8-52a315b24ee6.png)

假设序列 t 为32，即图中左上方顶部蓝色方格长条，每一格代表第 i 时刻的$\textbf{k}\_{i},\textbf{v}\_{i}$，将$\textbf{k}\_{:32},\textbf{v}\_{:32} $按照长度为 8 划分为4块 $\textbf{k}^{(1)}\_{1:8},\textbf{v}^{(1)}\_{1:8} ,\textbf{k}^{(2)}\_{8:16},\textbf{v}^{(2)}\_{8:16}....$，即 block\_size=8。

1.  **压缩**： 将每个KV块压成一个向量即$ \textbf{k}^{(1)}\_{1:8}\rightarrow \tilde{\textbf{k}}^{\text{(1)cmp}}\in \mathcal{R}^{d \times c} , \text{v}^{(1)} $同理， 压缩的目的是把一段长度为8的序列压成c个序列, 并且有关系c<8，为了便于理解我们令c=1，即 1 个token代表了一序列。至于怎么压缩我们后面探讨。 这里的压缩注意力是$\textbf{q}\_{32}$和4个段落的$ \tilde{\textbf{k}}^{\text{(b)cmp}},b={1,2,3,4} $进行计算得到4个注意力分数，所以这里做将v进行计算后，就得到压缩的注意力输出了，代表了全局信息$o^{(\text{cmp})}\_{32}\in \mathcal{R}^{d \times 1}$

2.  **选择**：在压缩时得到的段落注意力分数，选择**top-2,** 即第2和第4的绿色块。，那么我们就找到对应的段落块当成局部信息$\textbf{k}^{(2)}\_{8:16},\textbf{v}^{(2)}\_{8:16} ,\textbf{k}^{(4)}\_{24:32},\textbf{v}^{(4)}\_{24:32}$， 获取到KV后我们也可以计算选择注意力，最终为$o^{(\text{sle})}\_{32}\in \mathcal{R}^{d \times 1}$

3.  **滑窗**： 在原$\textbf{k}\_{:32},\textbf{v}\_{:32} $序列里取就近的8 个键和值$\textbf{k}\_{24:32},\textbf{v}\_{24:32}$， 同样可以得到滑窗注意力为$o^{(\text{win})}\_{32}\in \mathcal{R}^{d \times 1}$

4.  **门控**： 汇聚三种注意力$g^{(cmp)}\_{32}o^{(\text{cmp})}\_{32}+g^{(sle)}o^{(\text{sle})}\_{32}+g^{(win)}o^{(\text{win})}\_{32}$, 其中门控g 为输入$x\_{32} $进行线性层变换并加入sigmoid激活得到。门控网络也是可学习的。

进一步分析，假设原来有t=32个上下文KV，在压缩/选择/滑窗里分别有$N\_t=4+8\*2+8=28$个上下文KV，以此实现了注意力的减少。

那么当上下文扩展为64k时, 如果取128个全局压缩KV，8个512选择块KV和滑动窗口4096个KV, 那么我们得到了压缩倍数7.88：

$65536/(128+8\times 512+4096)=65536/8320 \approx 7.88 \\$

### Compressed Attention（压缩注意力）

比喻：把一本书的每一个章节用一段话总结，以快速阅读和决策。

**实现原理**：以block\_size为单位，将previous kv 压缩为 block\_num 个 dense tensor，再与当前的q进行attention计算。

特别要注意：Compression Attention每个头会独立计算attention score，即每个head会注意到不同的片段。

假设参数配置：

```python
bs = 1
t = 32 # tokens
block_size = 8  # block_size
block_nums = t // block_size = 4
dim = 256
heads = 4
head_dim = dim//heads = 64

q = torch.randn(bs, seq_len, heads, head_dim) # [1, 32, 4, 64]
k = torch.randn(bs, seq_len, heads, head_dim) # [1, 32, 4, 64]
v = torch.randn(bs, seq_len, heads, head_dim)
```

pytorch实现，并与full-attention对比

```python
# compression
k_cmp, v_cmp = compression(k, v, block_size)  # [bs, block_nums, head, head_dim]: [1, 4, 4, 64]
Q_mha = Q.view(bs, t, heads, head_dim).transpose(1,2) # [bs, head, block_nums, head_dim]
K_cmp_mha = K_cmp.view(bs, block_nums, heads, head_dim).transpose(1,2)  # [bs, head, block_nums, head_dim]
V_cmp_mha = V_cmp.view(bs, block_nums, heads, head_dim).transpose(1,2)

# (bs, head, q_len, head_dim) @ (bs, head, head_dim, block_nums) --> [1, 4, 32, 64] @ [1, 4, 64, 4] -> [1, 4, 32, 4]
score_cmp = Q_mha @ K_cmp_mha.transpose(2,3) # [bs, head, q_len, block_nums]: [1, 4, 32, 4]
p_cmp = F.softmax(score_cmp, dim = -1)
###### 32*4  ######
# 0.1  0.3  0.2  0.4
# 0.2  0.5  0.1  0.2
# ...
# ...
###################
# ([1, 4, 32, 4]) @ ([1, 4, 4, 64]) --> torch.Size([1, 4, 32, 64])
o_cmp = p_cmp @ V_cmp_mha



# raw attn
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#  (bs, head, q_len, head_dim) @ (bs, head, head_dim, q_len) --> [1, 4, 32, 64] @ [1, 4, 64, 32] --> [1, 4, 32, 32]
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) # [1, 4, 32, 32]
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
###### 32*4  ######
# 0.1  0.3  0.2  0.2 0.1 0.1 ...
# 0.2  0.1  0.3  0.2 0.1 0.1 ...
# ...
# ...
###################
# ([1, 4, 32, 32]) @ ([1, 4, 32, 64]) --> ([1, 4, 32, 64]) 
attn_output = torch.matmul(attn_weights, value_states)
```

其中压缩算法具体实现为：

```python
def compression(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int
) -> torch.Tensor:
    # Currently, we set mean pooling as our basic compression function.
    B, T, H = k.shape[:3]
    num_block = math.ceil(T / block_size)
    if k.shape[1] % block_size != 0:
        k = F.pad(k, (0, 0, 0, 0, 0, num_block * block_size - T))
        v = F.pad(v, (0, 0, 0, 0, 0, num_block * block_size - T))
    k_cmp = k.view(B, num_block, block_size, H, -1).mean(dim=2)
    v_cmp = v.view(B, num_block, block_size, H, -1).mean(dim=2)
    return k_cmp, v_cmp
```

### Selected Attention（选择注意力）：

根据每本书的章节总结，选出最感兴趣的几章，然后再逐字仔细阅读。

**实现原理**：以block为单位，利用压缩注意力计算出的表示，先和当前q计算注意力分数，然后取出注意力分数top-k的block，针对这些block内部的token进行进一步的attention 计算

**硬件高效设计**：由于每个q中的token关注到的kv block不一致，因此使用fa的方式并不高效（对于full-attention，每个token都需要和全部的kv计算，因此fa可以分批加载qkv到SRAM上）。nsa基于GQA，限制Group内部的Query head只能选择相同的kv-block，这样可以在head 维度凑矩阵乘法，从而利用GPU的tensor core加速运算，并且不同Group间自由选择仍然保持了一定的自由程度。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/eYVOL5jV1drbzlpz/img/adcd02eb-c630-4df2-a98a-f0448c9ceb26.png)

N：序列长度，d\_k：q和k向量的维度，d\_v：v向量的维度，h：group内head数目

torch 代码使用topk 选取K个片段

```python
# mha
print(p_cmp.shape) # [bs, head, q_len, block_nums]
# 在head维度上进行合并
p_slc = p_cmp.sum(dim = 1)
print(p_slc.shape) # [bs, q_len, block_nums]

# 对于GQA，每个组选取不同的KV
p_slc = p_slc.view(B, H, G, q_len, block_nums).sum(2) 

# top-K
select_top_k = 2
_, idx = torch.topk(p_slc, dim = 2, k = select_top_k)

idx_slc_start = idx * d
idx_slc_end = idx * d + l
K_slc = torch.randn(batch_size, t, d * select_top_k, dim)
V_slc = torch.randn(batch_size, t, d * select_top_k, dim)
for i in range(batch_size):
    for j in range(t):
        for k in range(select_top_k):
            K_slc[i, j, k * d : k * d + l, :] = K[i, idx_slc_start[i, j, k ] :  idx_slc_end[i, j, k ] , :]
            V_slc[i, j, k * d : k * d + l, :] = V[i, idx_slc_start[i, j, k ] :  idx_slc_end[i, j, k ] , :]

      # shared head KV
# IN GQA Group: [1-head KV & N-head Q] ----repeat kv-head---> [N-head KV & N-head Q]

V_slc_mha = V_slc.view(batch_size, t, select_top_k * d, heads, head_dim).transpose(2,3)
V_slc = V_slc_mha.sum(dim = 2, keepdim = True)
print(V_slc.shape) # bs, seq_len, head, select_seq_len, head_dim

K_slc_mha = K_slc.view(batch_size, t, select_top_k * d, heads, head_dim).transpose(2,3)
K_slc = K_slc_mha.sum(dim = 2, keepdim = True)
print(V_slc.shape) # bs, seq_len, head, select_seq_len, head_dim

o_slc = torch.zeros(batch_size, t, dim)
for j in range(t):
    Q_slc_j = Q_mha[:, :, j, :].unsqueeze(dim = 2)
    K_slc_j = K_slc[:, j, :, :, :].repeat(1, heads, 1, 1)
    V_slc_j = V_slc[:, j, :, :, :].repeat(1, heads, 1, 1)
    
    attn_score_j = Q_slc_j @ K_slc_j.transpose(2,3)
    p_slc_j = F.softmax(attn_score_j, dim = -1) 
    # print(p_slc.shape)

    o_slc_j = p_slc_j @ V_slc_j # bs, seq, dim   
    # print(o_slc_j.shape)

    o_slc_j = o_slc_j.transpose(1,2).view(batch_size, 1, dim)
    o_slc[:, j, :] = o_slc_j
```

triton kernel调用侧逻辑

```python
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    grid = (T, NV, B * H)

parallel_nsa_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        block_counts=block_counts,
        offsets=offsets,
        token_indices=token_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )
```

triton kernel具体实现如下代码。由于每个token需要attend的KV block是不同的，所以NSA用Group Size维度去凑tensor core调用，对应上图中的h。具体到代码为第79行的b\_s = tl.dot(b\_q, b\_k)，此处运算的计算维度是\[G, BS\] = \[G, BK\] @ \[BK, BS\]，这里的G是group\_size，由此凑出了tensor core运算，实现了高效计算。  
需要注意的是由于用group维度去凑tensor core，因此nsa的group\_size需要比较大，在triton中mnk的size最小是16\*16\*16，因此group\_size最小为16，这样在模型设计时需要更多的head数。

```python
@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    block_counts,
    offsets,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H*S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BS, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float('-inf'))

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = tl.exp(b_mp - b_m)
            # [G, BS]
            b_p = tl.exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            # [G, BV]
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

            b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += tl.log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))

```

### Sliding Attention（滑动窗口注意力）

逐字仔细阅读当前token最邻近的内容

**实现原理**：选取一个邻近的window size，只在这个window内部进行attention的计算；

可以直接复用fa的实现。

```python
o_swa = flash_attn_func(
    q, k, v,
    causal=True,
    window_size=(window_size-1, 0)
)
```

### Gated Aggregation（门控网络）

上述三种注意力分别会得到一个output，NSA预先将输入特征通过一个**门控层**，计算出三个output各自的权重，最后加权求和得到最终的输出。

```python
gate = torch.nn.Sequential(
        torch.nn.Linear(HQ*D, HQ * 3, bias=False, dtype=dtype),
        torch.nn.Sigmoid(),
    ).to(device)
g = rearrange(gate(hidden_states), 's b (h d) -> b s h d', d=3)

o = (
        gate[..., 0:1] * o_cmp
        + gate[..., 1:2] * o_slc
        + gate[..., 2:3] * o_swa
    )
```

## 总结

有效提升训练和推理速度，同时能够提升训练效果。  
对Selected Attention进行了设计，硬件优化驱动模型设计。

# MoBA

**把 MoE 的稀疏计算思想引入 attention 中。**

相比于nsa，只保留了selected sttention。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/eYVOL5jV1drbzlpz/img/30b20b53-9e3f-4f75-84fc-599b3a2e044a.png)

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/eYVOL5jV1drbzlpz/img/b1b85251-3036-4305-8cae-a64c19d3094f.png)

```python
# 输入：
#   Q: 查询向量序列 [seq_len, d]
#   K, V: key/value 向量序列 [seq_len, d]
# 参数：
#   block_size: 每个 block 的长度
#   top_k: 每个 query 选择的 top-k blocks

def moba_attention(Q, K, V, block_size, top_k):
    seq_len, d = Q.shape
    num_blocks = seq_len // block_size

    # Step 1: 块划分（均匀划分 K/V）
    blocks_K = K.view(num_blocks, block_size, d)
    blocks_V = V.view(num_blocks, block_size, d)

    # Step 2: 计算每个 block 的代表 key（平均池化）
    block_keys = blocks_K.mean(dim=1)  # [num_blocks, d]

    # Step 3: 对每个 query 计算与所有 block 的相似度
    relevance_scores = Q @ block_keys.T  # [seq_len, num_blocks]

    # Step 4: 选择 top-k 最相关的 block（稀疏选择）
    topk_indices = relevance_scores.topk(k=top_k, dim=-1).indices  # [seq_len, top_k]

    # Step 5: 聚合选中的 K/V（稀疏 attention）
    output = torch.zeros_like(Q)
    for i in range(seq_len):
        qi = Q[i]  # 当前 query
        selected_ks = []
        selected_vs = []
        for j in topk_indices[i]:
            selected_ks.append(blocks_K[j])
            selected_vs.append(blocks_V[j])
        selected_K = torch.cat(selected_ks, dim=0)  # [top_k * block_size, d]
        selected_V = torch.cat(selected_vs, dim=0)

        # 执行标准 attention
        attn_weights = (qi @ selected_K.T) / (d ** 0.5)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        output[i] = attn_probs @ selected_V

    return output
```

注意：block\_size需要比较大

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/eYVOL5jV1drbzlpz/img/e8a6fe14-2f66-44ae-bd89-22a836fc4f24.png)

而nsa

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/eYVOL5jV1drbzlpz/img/45d5cfce-6961-4ef8-a3ab-231daa2bb6da.png)

# 总结：

MoBA 将 MoE 用在了长上下文注意力中，**采用门控机制将查询标记选择性地路由到最相关的块。同时**MoBA的灵活性使其能够与现有模型无缝集成。

_**“**__我们相信模型自己是能学会 Sparsity 的__**”**_

nsa则更加硬件高效，在4k-8k下就有速度和精度提升。