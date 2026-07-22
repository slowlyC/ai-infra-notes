# DeepSeek DSpark 原理与 SGLang 实现分析

本文从投机解码和 MTP 开始，介绍 DSpark 为什么采用“并行 draft backbone + 串行 Markov head”，以及 confidence-scheduled verification 如何根据请求置信度和引擎负载分配 Target verify 预算。后半部分基于 SGLang 的 DeepSeek V4 DSpark 实现，按实际调用链展开代码。

参考资料：

- 论文：[DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)
- 训练代码：[DeepSpec](https://github.com/deepseek-ai/DeepSpec)
- SGLang 代码基线`43124cdd90e543ea79b63800390e917063809510`

文中的 SGLang 路径省略前缀 `python/sglang/srt/`。当前 checkout 是 DSpark 开发版本，文件组织和配置项后续可能继续调整。

## 目录

- [1. 背景：自回归解码、投机解码与 MTP](#1-背景自回归解码投机解码与-mtp)
- [2. 为什么还需要 DSpark](#2-为什么还需要-dspark)
- [3. DSpark 整体结构](#3-dspark-整体结构)
- [4. 从训练理解 DSpark](#4-从训练理解-dspark)
- [5. SGLang 中的 DSpark 推理流程](#5-sglang-中的-dspark-推理流程)
- [6. Markov head 的结构和开销](#6-markov-head-的结构和开销)
- [7. Confidence-Scheduled Verification](#7-confidence-scheduled-verification)
- [8. Target verify、bonus token 与 acc_len](#8-target-verifybonus-token-与-acc_len)
- [9. DSpark 为什么能加速](#9-dspark-为什么能加速)
- [10. DSpark 与常规 MTP 的区别](#10-dspark-与常规-mtp-的区别)
- [11. 当前 DeepSeek V4 Pro 配置](#11-当前-deepseek-v4-pro-配置)
- [12. 常见问题](#12-常见问题)
- [13. 源码阅读顺序](#13-源码阅读顺序)

## 1. 背景：自回归解码、投机解码与 MTP

### 1.1 自回归解码的串行瓶颈

普通语言模型按下面的条件概率生成 token：

$$
p(x_{t+1}\mid x_{\le t})
$$

每轮只能确定一个新 token：

```text
Target(context)              → token₁
Target(context, token₁)      → token₂
Target(context, token₁,₂)    → token₃
```

虽然历史 token 的 KV cache 可以复用，新 token 仍然要经过完整的 Target 模型。生成 $N$ 个 token 至少需要 $N$ 次串行 Target forward。Decode 阶段每轮 token 数较少，GEMM 和 MoE kernel 往往没有充分占满 GPU，串行 launch 延迟也无法跨 token 摊薄。

### 1.2 投机解码是一套推理框架

投机解码引入较轻的 Draft 模型。Draft 先提出一段候选，Target 再用一次 forward 并行验证整段候选：

```text
Draft 提出 token₁ token₂ token₃ token₄
                    ↓
Target 一次并行计算四个位置的分布
                    ↓
接受最长正确前缀，并产生一个 bonus token
```

以 greedy decoding 为例，假设 Draft 和 Target 的结果是：

```text
位置            1       2       3       4
Draft          今天    天气    不错    呢
Target         今天    天气    很好    ...
比较结果         ✓       ✓       ✗
```

本轮接受 `今天 天气`。位置 3 使用 Target 给出的 `很好` 作为 bonus token，位置 3 之后的 Draft token 全部丢弃。下一轮从 `很好` 继续 draft。

Sampling 模式不能简单比较 argmax，而要使用 rejection sampling。设 Draft 和 Target 在当前位置的分布分别为 $q$ 和 $p$，Draft token $y$ 的接受概率为：

$$
\alpha(y)=\min\left(1,\frac{p(y)}{q(y)}\right)
$$

如果拒绝，则从经过修正的 Target 残差分布采样。正确的接受规则可以保持最终输出仍服从 Target 分布，Draft 只影响速度，不改变模型质量。

如果一轮最终提交的 token 数为 $C$，可以用下面的简化式理解平均单 token 延迟：

$$
T_{token}\approx\frac{T_{draft}+T_{verify}}{\mathbb{E}[C]}
$$

投机解码有三个优化方向：

- 降低 Draft 延迟 $T_{draft}$；
- 提高候选质量，让一次 verify 提交更多 token；
- 避免验证大概率会被拒绝的 suffix，降低有效 $T_{verify}$。

MTP、EAGLE、DFlash 和 DSpark 的差别，主要在于如何平衡这三个方向。

### 1.3 MTP 是训练能力，不是完整的投机解码流程

MTP（Multi-Token Prediction）在训练阶段让模型预测多个未来 token。普通 next-token loss 只监督 $x_{t+1}$：

$$
\mathcal{L}_{NTP}=-\log p(x_{t+1}\mid x_{\le t})
$$

MTP 还增加更远位置的监督。下面是顺序式 MTP 的示意写法：

$$
\mathcal{L}_{MTP}
=\sum_{k=1}^{K}\lambda_k
\left[-\log p_k(x_{t+k}\mid x_{\le t},x_{t+1:t+k-1})\right]
$$

DeepSeek 使用顺序连接的 MTP modules，而不是让多个独立 head 完全并行地猜未来位置。训练时可以使用 shifted ground-truth token，推理时则把上一 MTP step 的 sampled token 传给下一 step。

DeepSeek 的 MTP 模块通常包含 token embedding、主模型 hidden state 的融合投影、一个额外 Transformer layer，以及与主模型共享的 LM head。SGLang 中 DeepSeek V4 NextN 的入口是：

```text
models/deepseek_v4_nextn.py
└── DeepseekV4ModelNextN
    ├── embed_tokens
    ├── e_proj / h_proj
    ├── DeepseekV4DecoderLayer(is_nextn=True)
    └── shared_head
```

推理时可以连续调用这个 MTP 层来提出候选：

```text
Target hidden + 当前 token
    ↓
MTP step 1 → draft token₁
    ↓
MTP step 2 → draft token₂
    ↓
MTP step 3 → draft token₃
```

这只是 Draft 阶段。候选仍要交给 Target verify，才能组成保持 Target 分布的完整投机解码。

因此两者的关系是：

```text
投机解码 = Draft proposal + Target verify + Accept/Reject

MTP      = 可用来实现 Draft proposal 的一种模型能力
```

## 2. 为什么还需要 DSpark

### 2.1 自回归 Draft 的问题

MTP/EAGLE 一类自回归 Draft 显式依赖前面已经采样的 token，候选通常比较连贯，但产生长度为 $\gamma$ 的候选需要执行 $\gamma$ 个串行 draft step：

```text
draft₁ → sample₁ → draft₂ → sample₂ → ... → draftγ
```

Draft 模型必须很浅，block 也不能太长，否则 Draft 本身的延迟会抵消 Target verify 的收益。

### 2.2 完全并行 Draft 的问题

DFlash 一类并行 Draft 把未知位置替换为 MASK，一次 forward 同时产生整个 block 的 logits：

```text
[anchor, MASK, MASK, MASK, MASK]
                ↓ one forward
[logits₁, logits₂, logits₃, logits₄, logits₅]
```

所有输入在 forward 前已经确定，因此 Attention、MLP 和 LM head 可以把各位置作为一个大 tensor 并行计算。Draft latency 不再随 block size 线性增长。

代价是各位置在计算 base logits 时不知道其他位置最后采样出了什么。假设上下文存在两种自然续写：

```text
of course
no problem
```

完全独立的位置预测可能得到：

```text
位置 1 采样：of
位置 2 采样：problem
结果：of problem
```

位置越靠后，缺失的 block 内因果信息越多，接受率通常衰减得越明显，这就是 suffix decay。

### 2.3 固定长度 verify 的问题

并行 Draft 很容易一次生成较长的候选，但“生成出来”不代表“值得验证”。高并发时，如果每条请求都固定验证完整 block，大量低置信度 suffix 会占用 Target batch token 预算，却在前面的错误出现后全部作废。

DSpark 分别处理这两个问题：

| 问题 | DSpark 的处理方式 |
|------|------------------|
| 并行 Draft 缺少 block 内 token 依赖 | 用轻量串行 Markov/RNN head 修正 base logits |
| 高并发下固定验证长 suffix 浪费算力 | 用 confidence head 和硬件感知调度器选择每条请求的 verify 长度 |

## 3. DSpark 整体结构

DSpark 的完整名称是 Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation。一次解码循环可以概括为：

```text
Target 上轮产生 anchor token，并提供目标层 hidden features
    ↓
并行 Draft backbone
输入 [anchor, MASK, MASK, MASK, MASK]
一次得到整个 block 的 hidden states 和 base logits
    ↓
轻量串行 head
Markov(anchor) → 采样 draft₁
Markov(draft₁) → 采样 draft₂
Markov(draft₂) → 采样 draft₃
Markov(draft₃) → 采样 draft₄
Markov(draft₄) → 采样 draft₅
    ↓
Confidence head
估计每个 draft 位置的条件存活概率
    ↓
Hardware-aware prefix scheduler
根据置信度和 Target SPS 曲线决定各请求 verify_len
    ↓
Target verify
接受正确前缀，产生 bonus token
```

“Semi-Autoregressive”指的是大部分 Draft 计算并行完成，只有一个很轻的输出 head 按 token 串行执行。它既不是 MTP 式的全 Draft backbone 自回归，也不是 DFlash 式的完全位置独立。

## 4. 从训练理解 DSpark

### 4.1 训练数据和冻结策略

DSpark 从 Target 生成的序列中随机采样 anchor position，并以 anchor 后面的 $\gamma$ 个 token 作为一个训练 block。

训练期间：

- Target 模型保持冻结；
- Draft 与 Target 共享并冻结 token embedding 和 LM head；
- 只更新并行 Draft backbone、串行 Markov/RNN head 和 confidence head。

共享 embedding 和 LM head 可以让 Draft 与 Target 使用同一个 token 表示空间和词表投影。Draft 需要学习的是如何用较少层数和 Target hidden features 近似 Target 的未来分布。

### 4.2 并行 backbone 学什么

设上轮 Target 产生的 anchor 为 $a$，block size 为 $\gamma$。DSpark 构造：

$$
[a,m,m,\ldots,m]
$$

其中 $m$ 是 MASK/noise token。并行 backbone 一次输出：

$$
H=(h_1,h_2,\ldots,h_\gamma)
$$

并通过共享 LM head 得到 base logits：

$$
Z^{base}=(z^{base}_1,z^{base}_2,\ldots,z^{base}_\gamma)
$$

这些位置共享 Target 提供的上下文特征，但在 backbone forward 时还没有看到 block 内实际采样的 token。它们学习的是每个位置在上下文条件下的强先验分布。

DSpark 沿用了 DFlash 的 Target feature/KV injection 思路。Target 指定层的 hidden states 会被投影到 Draft hidden space，并作为 Draft attention 可访问的上下文。这样 Draft 不需要重新编码完整 Target 历史。

### 4.3 Markov head 学什么

并行 backbone 给出长程上下文和位置先验，Markov head 补充相邻 token 的转移关系。第 $i$ 个位置的最终 logits 为：

$$
z_i=z_i^{base}+b(y_{i-1})
$$

其中 $y_{i-1}$ 是前一个实际采样 token，$b$ 是对整个词表的 bias。

如果直接保存所有 token pair 的转移矩阵，需要：

$$
M\in\mathbb{R}^{V\times V}
$$

参数量是 $V^2$。DSpark 用 rank 为 $r$ 的低秩分解：

$$
E\in\mathbb{R}^{V\times r},\qquad
W\in\mathbb{R}^{V\times r}
$$

前一个 token $y_{i-1}$ 的 bias 为：

$$
b(y_{i-1})=E[y_{i-1}]W^T
$$

实现上就是：

```text
previous token id
    ↓
Embedding(V, r)
    ↓
Linear(r, V, bias=False)
    ↓
vocabulary bias
```

它叫 Markov head，不是因为使用了某种特殊的“Markov layer”，而是因为修正项只依赖前一个 token：

$$
b_i=b(y_{i-1})
$$

这对应一阶马尔可夫依赖。完整 DSpark 并没有忘记更早的历史，因为 $z_i^{base}$ 已经包含 Target context 和 Draft backbone 信息。

### 4.4 Confidence head 学什么

Confidence head 为每个位置输出一个标量：

$$
c_i=P(\text{位置 }i\text{ 被 Target 接受}\mid
\text{位置 }1\ldots i-1\text{ 已接受})
$$

它的输入包含当前位置的 Draft hidden state，以及前一个 token 的 Markov embedding：

$$
c_i=\sigma\left(w^T[h_i;E[y_{i-1}]]\right)
$$

位置 $i$ 能进入最终接受前缀，需要前面所有位置都存活，因此 prefix survival probability 是：

$$
s_i=\prod_{j=1}^{i}c_j
$$

Confidence head 预测的是条件概率，scheduler 真正关心的是累积存活概率 $s_i$。

### 4.5 训练损失

论文中的训练目标由三部分组成：

$$
\mathcal{L}
=\lambda_{ce}\mathcal{L}_{ce}
+\lambda_{dist}\mathcal{L}_{dist}
+\lambda_{conf}\mathcal{L}_{conf}
$$

交叉熵损失让 Draft 预测 ground-truth token：

$$
\mathcal{L}_{ce}=-\sum_i w_i\log q_i(x_i)
$$

分布匹配损失最小化 Draft 分布 $q_i$ 与 Target 分布 $p_i$ 的 total variation distance：

$$
TV(p_i,q_i)=\frac{1}{2}\sum_{v\in\mathcal{V}}|p_i(v)-q_i(v)|
$$

理想 speculative sampling 的单步期望接受率与 $1-TV(p_i,q_i)$ 直接相关，因此这个损失比只对齐 ground-truth token 更贴近投机解码目标。

Confidence loss 使用 soft acceptance label：

$$
\hat{c}_i=1-TV(p_i,q_i)
$$

再用 binary cross-entropy 训练 confidence head。三项损失都使用位置权重 $w_i$，让靠前位置获得更高权重，因为 prefix verify 中前面位置的错误会让整个 suffix 失效。

## 5. SGLang 中的 DSpark 推理流程

### 5.1 模块关系

当前 SGLang 实现的主要文件如下：

```text
speculative/dspark_components/dspark_worker_v2.py
    DSparkWorkerV2，组织整个解码循环

speculative/dspark_components/dspark_draft.py
    构造 anchor + MASK block，运行并行 Draft，调用 Markov sampling

models/deepseek_v4_dspark.py
    DeepSeek V4 Draft stages、Markov head、confidence head、Target hidden KV 写入

models/dspark.py
    通用 Markov/RNN/confidence head 和串行 sample loop

speculative/dspark_components/dspark_planner.py
    Confidence 计算、SPS cost table、verify budget 和 ragged layout

speculative/dspark_components/dspark_verify.py
    Target verify、accept/reject、hidden/KV commit

speculative/dspark_components/kernels/dspark_accept.py
    greedy/sampling accept 和 commit_lens 计算
```

`DSparkWorkerV2` 的 decode 主路径是：

```text
alloc_verify_window()
    ↓
DraftBlockProposer.propose()
    ↓
DSparkVerifyPlanner.compute_confidence_tensor()
    ↓
resolve_verify_token_budget() + schedule_layout()
    ↓
Target verify
    ↓
accept_and_finalize()
    ↓
commit_hidden()
```

对应代码在 `speculative/dspark_components/dspark_worker_v2.py:507-630`。

### 5.2 构造并行 Draft 输入

`DraftBlockProposer._run_forward()` 先用 MASK 填满 `[bs, gamma]`，再把第一列替换成上轮 bonus/anchor token：

```python
draft_block_ids = torch.full(
    (bs, gamma),
    int(self._mask_token_id),
    dtype=torch.long,
    device=device,
)
draft_block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
```

代码位置：`speculative/dspark_components/dspark_draft.py:352-355`。

当 `gamma=5` 时，每条请求的 Draft 输入为：

```text
[anchor, MASK, MASK, MASK, MASK]
```

随后把整个 block 展平放入一个 `ForwardBatch`：

```python
draft_forward_batch = ForwardBatch(
    forward_mode=ForwardMode.TARGET_VERIFY,
    batch_size=bs,
    input_ids=draft_block_ids.flatten(),
    ...
)
```

Draft model runner 只执行一次 forward。所有请求、所有 block 位置一起进入 GPU，得到 `[bs * gamma, hidden]`。

### 5.3 Draft backbone 一次计算整个 block

DeepSeek V4 的 Draft 模型入口是 `DeepseekV4ForCausalLMDSpark.forward()`：

```python
if input_embeds is None:
    input_embeds = self.forward_embed(input_ids)
x = input_embeds
for stage in self.stages:
    x = stage(positions, x, forward_batch)

return LogitsProcessorOutput(
    next_token_logits=None,
    hidden_states=x,
)
```

代码位置：`models/deepseek_v4_dspark.py:690-706`。

这就是“Draft 可以并行”的直接原因。它没有等待位置 1 采样完再构造位置 2 的输入，而是提前用 MASK 占位。所有位置的 embedding 已知，可以作为同一个 tensor 通过 Attention、MoE 和 GEMM。

并行的代价也很明确：backbone 产生位置 2 hidden state 时还不知道位置 1 最后采样出的 token，所以这里只能得到 base logits。

### 5.4 共享 LM head 产生 base logits

Draft hidden 经 mHC collapse 和 norm 后，使用 Target 的共享 LM head：

```python
def compute_base_logits(self, x):
    x_post_hc = self.collapse_hc_head(x)
    return self._logits_from_x_post_hc(x_post_hc), x_post_hc

def _logits_from_x_post_hc(self, x_post_hc):
    x = self.stages[-1].norm(x_post_hc)
    local_logits = torch.matmul(x.to(weight.dtype), weight.T)
    return local_logits
```

代码位置：`models/deepseek_v4_dspark.py:719-739`。

`DraftBlockProposer.propose()` 再把 logits 恢复成三维：

```python
base_logits = base_logits.view(bs, self.gamma, -1)
```

其形状为：

```text
[batch_size, gamma, vocab_size]
```

### 5.5 Markov head 串行采样

`sample_draft_block()` 调用：

```python
draft_tokens, corrected_logits = markov_head.sample_block(
    base_logits,
    first_prev_tokens=anchor_tokens,
    hidden_states=draft_hidden,
    sampler=sampler,
)
```

代码位置：`speculative/dspark_components/dspark_draft.py:203-208`。

最终进入 `models/dspark.py` 中的 `run_markov_block()`：

```python
prev_tokens = first_prev_tokens.long()

for step_idx in range(proposal_len):
    step_logits = head.apply_step_logits(
        base_logits[:, step_idx, :],
        token_ids=prev_tokens,
        hidden_states=step_hidden,
    )
    next_tokens = sampler(step_logits, step_idx)
    sampled_tokens.append(next_tokens)
    prev_tokens = next_tokens
```

代码位置：`models/dspark.py:32-62`。

它表达的依赖关系是：

```text
anchor
  ↓ Markov bias + base_logits₁
draft₁
  ↓ Markov bias + base_logits₂
draft₂
  ↓ Markov bias + base_logits₃
draft₃
```

Markov head 在当前位置采样之前校正 logits，不会事后修改已经采样的 token。

### 5.6 DeepSeek V4 的 Markov head

DeepSeek V4 使用 `DSparkV4MarkovHead`：

```python
self.markov_w1 = VocabParallelEmbedding(
    self.vocab_size,
    self.markov_rank,
    enable_tp=False,
)
self.markov_w2 = nn.Linear(
    self.markov_rank,
    self.vocab_size,
    bias=False,
)
```

应用 bias 的逻辑为：

```python
def compute_step_bias(self, token_ids, hidden_states):
    del hidden_states
    return self.project_bias(self.get_prev_embeddings(token_ids))

def apply_step_logits(self, logits, token_ids, hidden_states):
    return logits + self.compute_step_bias(token_ids, hidden_states)
```

代码位置：`models/deepseek_v4_dspark.py:287-374`。

这里明确丢弃了 `hidden_states`，因此当前 V4 checkpoint 使用的是 vanilla 一阶 Markov head。通用 `models/dspark.py` 还实现了 gated 和 RNN 变体，但不能在没有对应训练权重的情况下直接切换。

### 5.7 Confidence 计算

如果开启 confidence head，DeepSeek V4 会构造每个位置的前驱 token 序列：

```python
prev_seq = torch.cat(
    [anchor_tokens.view(-1, 1),
     sampled_tokens[:, : self.gamma - 1]],
    dim=1,
)
markov_embed_stack = self.markov_head.get_prev_embeddings(prev_seq)
confidence_raw = confidence_head(x_post_hc, markov_embed_stack)
confidence = confidence_head.apply_sts(confidence_raw)
```

代码位置：`models/deepseek_v4_dspark.py:741-765`。

Confidence head 本身是一次轻量 projection：

```python
features = torch.cat([hidden_states, markov_embed_stack], dim=-1)
confidence_raw = self.proj(features).squeeze(-1)
confidence = torch.sigmoid(confidence_raw / sts_temperatures)
```

`sts_temperatures` 是 Sequential Temperature Scaling 的逐位置温度。它用于校准置信度的绝对数值，不改变 speculative accept 规则。

### 5.8 Target verify 和 finalize

Draft 完成后，worker 拼出 Target verify 输入：

```python
verify_ids_2d = torch.cat(
    [draft_block_ids[:, :1], draft_tokens],
    dim=1,
).contiguous()
```

代码位置：`speculative/dspark_components/dspark_worker_v2.py:565-567`。

当 `gamma=5` 时：

```text
Draft proposal 数：5
Target verify window：[anchor, draft₁, draft₂, draft₃, draft₄, draft₅]
verify width：gamma + 1 = 6
```

Target 一次 forward 输出每个 verify 位置的 logits 和 hidden states。随后 `accept_draft_tokens()` 分别处理 greedy、sampling 或混合 sampling 请求：

```text
speculative/dspark_components/dspark_verify.py:653-716
```

Greedy 路径在 `kernels/dspark_accept.py:598-620` 中比较 Draft candidate 与 Target argmax，得到：

- `correct_len`，Target 接受的 Draft token 数；
- `bonus`，第一个不匹配位置或完整接受后的 Target token。

最终提交长度为：

```python
commit_lens = correct_len.to(torch.int32) + 1
```

代码位置：`kernels/dspark_accept.py:716-728`。

这里的 `+1` 就是 bonus token。

## 6. Markov head 的结构和开销

### 6.1 它比 Draft backbone 小多少

当前 DeepSeek V4 Pro DSpark checkpoint 的关键尺寸为：

```text
vocab_size  = 129280
hidden_size = 7168
markov_rank = 512
gamma       = 5
Draft stage = 3
```

Markov embedding 是查表，主要计算来自：

```text
Linear(512, 129280)
```

单位置约需要：

$$
512\times129280\approx66.2\text{M MACs}
$$

Draft 共享 LM head 为：

```text
Linear(7168, 129280)
```

单位置约需要：

$$
7168\times129280\approx926.7\text{M MACs}
$$

两者计算量之比约为：

$$
\frac{512}{7168}=\frac{1}{14}
$$

Markov projection 对相同数量位置的理论乘加量约为 LM head 的十四分之一。Draft backbone 还包含三个 stage 的 Attention 和 MoE，因此 Markov head 明显更轻。

### 6.2 为什么仍然不能把它当成免费操作

Markov head 有两个不利因素：

- 五个位置必须串行执行，不能合成一个跨位置的大 GEMM；
- 每一步都输出完整词表，并执行 sampling。

小 batch 下，五次小 GEMM 的 GPU 利用率和 kernel launch 开销可能比 FLOPs 比例看起来更明显。SGLang 因此默认启用：

```python
SGLANG_DSPARK_OPT_MARKOV_W2_BF16 = True
SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD = True
SGLANG_DSPARK_FAST_SAMPLING = True
```

代码位置：`environ.py:288-292`。

TP shard 路径只计算本 rank 的词表分片，再进行 all-gather，见 `models/deepseek_v4_dspark.py:376-393`。

## 7. Confidence-Scheduled Verification

### 7.1 为什么置信度不能只看当前位置

假设 confidence head 给出：

```text
c₁ = 0.95
c₂ = 0.90
c₃ = 0.70
c₄ = 0.30
```

它们是“前缀已经存活条件下，当前位置继续存活”的概率。各位置真正进入接受前缀的概率为：

```text
s₁ = 0.95
s₂ = 0.95 × 0.90 = 0.855
s₃ = 0.95 × 0.90 × 0.70 = 0.5985
s₄ = 0.95 × 0.90 × 0.70 × 0.30 = 0.17955
```

位置 4 即使自己的条件概率不算极低，也必须建立在前三个位置全部正确的基础上。Scheduler 应按 cumulative survival，而不是单点 confidence 分配 verify 预算。

### 7.2 STS 校准

Hardware-aware scheduler 需要置信度的绝对数值来估算期望接受 token 数。神经网络 confidence 往往存在过度自信，因此论文使用 Sequential Temperature Scaling，从前往后校准每个位置的 cumulative survival probability。

SGLang 通过：

```text
--speculative-dspark-confidence-sts-path <calibration.json>
```

加载逐位置温度，入口在 `speculative/dspark_components/dspark_planner.py:83-112`。STS 只影响 verify length 的选择，不参与 Target accept/reject，因此校准误差影响性能决策，不应改变输出分布。

### 7.3 Hardware-aware prefix scheduler

同一个额外 verify token 在不同负载下的成本不同：

```text
低负载：Target batch 还有空余，增加 verify token 可能几乎不增加 step time
高负载：Target batch 已接近饱和，低置信度 token 会挤占其他请求容量
```

SGLang 先离线 profile Target 在不同 verify token batch size 下的 steps per second，得到 SPS cost table：

```text
--speculative-dspark-sps-table-path <sps.json>
```

设当前 batch 有 $B$ 条请求，从所有请求的候选 prefix 中按 survival probability 从高到低取前 $k$ 个额外 verify token。代码计算：

$$
\tau(k)=B+\sum_{j=1}^{k}s_{(j)}
$$

其中 $B$ 表示每条请求至少提交的一个 Target token，$s_{(j)}$ 是排序后的 survival probability。再结合该 batch token 数对应的 steps per second：

$$
\theta(k)=\tau(k)\cdot SPS(B+k)
$$

选择让期望 token throughput 最大的 $k$。对应代码是：

```text
speculative/dspark_components/dspark_planner.py
└── compute_verify_token_budget():926-963
```

GPU 端再根据 budget 选择每条请求的 `verify_lens`：

```python
verify_lens = ScheduleVerifyLensTopk.execute(
    confidence=confidence,
    budget=budget,
    cfg=self._schedule_cfg,
)
```

代码位置：`dspark_planner.py:550-590`。

由于 speculative decoding 只能接受连续前缀，调度也必须保持 prefix 约束，不能验证某条请求的位置 4 却跳过位置 2 和 3。

### 7.4 SGLang 的三种 ragged verify 模式

当前实现定义了：

```text
static
cap-accept
compact
```

代码位置：`speculative/ragged_verify.py:13-27`。

`static` 使用固定完整 verify window，不创建 confidence head，也不做 confidence scheduling。`compact` 会把每条请求选中的不同长度压紧成 ragged Target batch，实际减少送入 Target 的 verify token。`cap-accept` 会计算调度长度并限制本轮最多接受到哪里，但 Target 仍走 non-compact 的完整窗口，因此它本身不等价于节省 Target verify 计算。

如果没有提供有效 SPS table，当前 compact planner 会退化为 verify-all。`server_args.py:1715-1721` 明确说明，未初始化的常数 SPS 表本身不会带来 throughput 收益。

## 8. Target verify、bonus token 与 acc_len

### 8.1 Target verify 输出什么

Target verify 对 `[anchor, draft₁, ..., draftγ]` 做一次前向，主要输出：

- 每个 verify 位置的 Target logits，用于 greedy 比较或 rejection sampling；
- 对应 hidden states，用于提交被接受位置的 Target 状态，并注入下一轮 Draft KV。

Target verify 不是返回一个“这段对不对”的布尔值，而是返回每个位置的完整 Target 分布。Accept kernel 再从左到右确定最长接受前缀。

### 8.2 第二个 Draft token 错了会发生什么

假设：

```text
Draft：  token₁ token₂ token₃ token₄ token₅
Target： token₁ token₂* ...
```

结果是：

```text
接受 token₁
丢弃 token₂、token₃、token₄、token₅
提交 Target 产生的 token₂* 作为 bonus
下一轮从 token₂* 继续
```

这可以恢复到 Target 的生成路径，但不是把原 Draft token₂ “改写后继续保留 suffix”。因为 token₃ 以后都是基于错误 token₂ 采样的，必须一起丢弃。

Markov head 与这里的 Target 修正也不是一回事：

```text
Markov head：在 Draft token 采样前调整 logits
Target verify：在 Draft 完成后决定接受、拒绝并给出 bonus
```

### 8.3 `correct_len`、`commit_lens` 和 `acc_len`

SGLang 中：

```text
correct_len = 本轮被 Target 接受的 Draft token 数
commit_lens  = correct_len + 1
```

`commit_lens` 包含 bonus token，因此即使第一个 Draft token 就错了：

```text
correct_len = 0
commit_lens = 1
```

系统仍然前进一个由 Target 产生的 token。

服务指标中的 `acc_len` 或 `spec_accept_length` 通常表示每次 verify 平均让请求前进多少 token，口径包含 bonus。它不等于“Draft 平均猜对几个”。阅读 benchmark 时要先确认统计的是 `correct_len` 还是 `commit_lens`。

## 9. DSpark 为什么能加速

### 9.1 Target 仍然计算 verify block

DSpark 没有让 Target 跳过所有候选 token 的计算。Target 确实需要为进入 verify window 的 token 计算 logits。收益来自执行方式变化：

```text
普通解码：5 个 token = 5 次串行 Target forward

DSpark：一次较轻的 Draft block
        + 一次 Target block verify
        → 一轮可能提交多个 token
```

Target 的 block verify 把多个 query position 合到一个 forward 中，GEMM、Attention 和 MoE 获得更大的 token 维度，GPU 利用率通常高于单 token decode。系统减少的是串行 Target 轮数和每个输出 token 分摊的 launch/调度成本，不是凭空消除所有 token FLOPs。

### 9.2 DSpark 相对 MTP 的收益来源

MTP 产生 $\gamma$ 个候选，需要多次串行运行额外 Transformer layer。DSpark 把较重的 backbone 计算改为一次并行 block forward，仅保留低秩 Markov projection 和 sampling 串行执行。因此它可以使用更深的 Draft backbone 和更长的 block，同时保持 Draft latency 可控。

更高的模型容量提高 base logits 质量，Markov head 又缓解 suffix decay。两者共同提高每次 Target verify 的期望提交长度。

### 9.3 Confidence scheduler 的收益来源

并发升高后，Target verify token 会直接占用 decode batch 容量。Confidence scheduler 不再对每条请求固定验证完整 block，而是优先验证高 survival probability 的 prefix，把有限的 Target token budget 分配给预期收益较高的位置。

[DSpark 论文](https://arxiv.org/abs/2607.05147)报告的线上收益来自特定 DeepSeek V4 模型、硬件、流量和调度配置。论文给出的 matched-throughput 结果是，DSpark 相对 MTP-1 将 V4-Flash 的 per-user generation speed 提高 60%–85%，V4-Pro 提高 57%–78%。这些数字不能直接当作当前本地 SGLang 脚本的预期收益，实际结果仍取决于 batch、请求分布、SPS 表、verify mode、Draft kernel 和 Target backend。

### 9.4 什么时候可能不加速

以下情况会压低 DSpark 收益：

- Draft 与 Target 分布差异大，`correct_len` 长期接近 0；
- block 太长，大量 suffix 永远到不了 verify 接受位置；
- batch 很小，Markov 的多次小 GEMM 和 sampling launch 占比高；
- 高并发下仍使用 static verify-all；
- ragged verify 的实际 shape 没有命中合适的 CUDA Graph tier；
- Draft/Target 的 Attention、MoE、通信后端没有为当前硬件和 shape 优化。

是否加速要看完整 serving path 的 TPOT、ITL、output TPS 和平均 `commit_lens`，不能只看 Draft kernel 或平均接受长度。

## 10. DSpark 与常规 MTP 的区别

| 对比项 | 常规 MTP / NEXTN | DSpark |
|--------|------------------|--------|
| 定位 | 训练出的 NextN 预测模块，可作为自回归 Draft | 半自回归 Draft + confidence-scheduled verify 框架 |
| 重计算方式 | 多次串行运行轻量 Transformer layer | 重 backbone 一次并行，轻 head 串行 |
| block 内依赖 | 每一步显式看到前面 sampled token | base logits 并行，Markov/RNN head 补局部依赖 |
| Draft block 长度 | 受串行 Draft latency 限制 | 并行 backbone 使长 block 更可行 |
| verify 长度 | 通常固定，或使用通用 adaptive 策略 | 原生设计了 confidence + hardware-aware scheduler |
| Target verify | 需要 | 需要 |
| 输出分布 | 正确 accept 时保持 Target 分布 | 正确 accept 和非前视调度时保持 Target 分布 |

在当前 SGLang 中，`NEXTN` 和 `DSPARK` 是 `--speculative-algo` 的两个可选值。同一个 server 只有一个 `speculative_algorithm`，worker factory 也只创建一套 speculative worker。因此不能在同一条生成链中先跑常规 MTP，再叠加 DSpark。

这不表示二者没有训练或结构关联。DSpark 的 Draft stage 可以复用 Target 特征、共享 embedding/LM head，并采用与 MTP 类似的轻量附加模块思路。但运行时它们是互斥的 Draft 方案。

## 11. 当前 DeepSeek V4 Pro 配置

### 11.1 Checkpoint 参数

`/cpfs/models/DeepSeek-V4-Pro-DSpark/config.json` 中的 DSpark 参数为：

```json
{
  "vocab_size": 129280,
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "dspark_block_size": 5,
  "dspark_noise_token_id": 128799,
  "dspark_target_layer_ids": [58, 59, 60],
  "dspark_markov_rank": 512
}
```

对应关系如下：

| 配置 | SGLang 含义 |
|------|-------------|
| `dspark_block_size=5` | `gamma=5`，每轮提出 5 个 Draft token |
| `speculative_num_draft_tokens=6` | Target verify width 为 `gamma+1` |
| `dspark_noise_token_id` | 构造并行 block 时使用的 MASK/noise token |
| `dspark_target_layer_ids=[58,59,60]` | 注入三个 Target feature，并构造三个 Draft stage |
| `dspark_markov_rank=512` | Markov transition 的低秩维度 |

`DeepseekV4ForCausalLMDSpark.__init__()` 根据 `target_layer_ids` 的数量创建 Draft stages：

```python
self.num_stages = len(dspark_config.target_layer_ids)
self.stages = nn.ModuleList(
    DSparkV4Stage(...) for stage_id in range(self.num_stages)
)
```

代码位置：`models/deepseek_v4_dspark.py:572-625`。

### 11.2 当前启动脚本

`script/dsv4-pro/launch_server_dsv4_pro_decode_highspeed_dspark.sh` 中的相关配置为：

```bash
export SGLANG_RAGGED_VERIFY_MODE=static

sglang serve \
  --model-path /cpfs/models/DeepSeek-V4-Pro-DSpark/ \
  --tp 8 \
  --speculative-algo DSPARK \
  --speculative-dspark-block-size 5 \
  --speculative-num-draft-tokens 6
```

这里启用了 DSpark 的并行 backbone 和 Markov head，但 `SGLANG_RAGGED_VERIFY_MODE=static` 会在模型构造阶段直接禁用 confidence head：

```python
if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
    return None
```

代码位置：`models/deepseek_v4_dspark.py:417-421`。

因此当前脚本运行的是：

```text
Semi-autoregressive Draft
+ 固定长度 Target verify
```

而不是论文中完整的：

```text
Semi-autoregressive Draft
+ Confidence head
+ STS calibration
+ Hardware-aware dynamic verify scheduling
```

如果后续评估 confidence-scheduled verification，需要同时确认 checkpoint 包含已训练的 confidence head、选择非 static ragged verify mode、生成当前硬件/后端对应的 SPS table，并按需要拟合 STS calibration。缺少其中任何一项，都不能把结果归因于完整 DSpark scheduler。

## 12. 常见问题

### 12.1 为什么 Draft backbone 可以并行，Target 却不能直接并行生成

Draft backbone 使用提前确定的 MASK 输入，计算的是近似 base logits。Target 需要保持真实自回归分布，位置 2 的输入必须包含实际采样的 token₁，不能提前用未知值替代。

Target 可以并行做 verify，是因为候选 token 已经由 Draft 给出。此时 Target 的整段输入确定，可以并行计算每个候选位置的条件分布。

### 12.2 Markov head 为什么不直接使用 Target

如果每采样一个 Draft token 都重新调用 Target，就回到了普通自回归解码，失去投机收益。Markov head 的目的就是用低秩 `Embedding + Linear` 近似局部 token 转移，以远低于 Target forward 的成本补充 block 内依赖。

### 12.3 Markov head 能改回已经采样错的 token 吗

不能。它只在当前位置采样前调整 logits。已经采样的前一个 token 会作为下一位置的输入，但不会被回写。真正发现 Draft 与 Target 不一致并恢复 Target 路径的是 Target verify。

### 12.4 `base_logits + Markov bias` 能改变什么

Sampling 和 argmax 都直接作用在最终 logits 上。即使 base logits 已经计算完成，加上 Markov bias 仍会改变词表排序和概率分布，从而改变当前位置即将采样的 token。

例如 base logits 更偏向 `problem`，前一个 token 为 `of` 时，Markov bias 可以提高 `course` 并降低 `problem`，使最终采样从 `problem` 变成 `course`。

### 12.5 DSpark 是否一定比 MTP 快

不一定。DSpark 用更重的并行 backbone 换取更高候选质量和更长 block，还增加了 Markov sampling 和可选 confidence scheduling。只有当一次 verify 平均提交的 token 增长足以覆盖这些额外开销时，端到端才会加速。

### 12.6 使用 DSpark 后还能同时使用常规 MTP 吗

当前 SGLang 不能在同一个 server 的同一条生成链中叠加二者。需要在 `--speculative-algo NEXTN` 和 `--speculative-algo DSPARK` 之间选择。可以分别启动两个服务做 A/B benchmark。
