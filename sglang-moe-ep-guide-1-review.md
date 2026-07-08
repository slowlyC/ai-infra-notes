# SGLang GLM-5.1 MoE & Expert Parallelism 原理详解

文档基于 SGLang 代码库中 GLM-5.1 的 MoE 实现。模型入口是 `GlmMoeDsaForCausalLM`，MoE 主体复用 `DeepseekV2MoE` 类，按代码调用链逐层展开。每一章对应调用链的一层，读者从第一章开始读，相当于在逐层 step-into 代码。

重点覆盖 EP（Expert Parallelism）的两种 dispatch 模式：normal（高吞吐，常用于 prefill）和 low_latency（低延迟，常用于 decode，但也可用于 prefill），以及 TP（Tensor Parallelism）模式的 FP4 MoE 计算。通过具体的 tensor shape 示例说明数据流向。

代码基线：`sglang/python/sglang/srt/`（以下路径省略此前缀）

## 1. GLM-5.1 MoE 概览

> **代码定位**：`models/deepseek_v2.py` → `DeepseekV2MoE.__init__()` / `forward()`

### 1.1 MoE 在 Transformer Block 中的位置

以 GLM-5.1 单个 transformer block 为例，MoE 替代了标准 FFN 的位置：

```
DeepseekV2DecoderLayer.forward()
    │
    ├── RMSNorm
    ├── Attention (MLA: Multi-head Latent Attention)
    │     └── q/k/v_proj → FlashAttention → o_proj
    ├── residual connection
    │
    ├── RMSNorm
    └── MoE                              ← 本文档的重点
          ├── Gate
          ├── TopK
          ├── Shared Expert
          └── Routed Experts
```

GLM-5.1 有 75 个 MoE 层（另有 3 个 dense FFN 层，不走 MoE）。每个 MoE 层包含 256 个 routed expert 和 1 个 shared expert。

Mixture of Experts 是一种条件计算架构：每个 token 不经过全部参数，而是通过路由机制选出若干"专家"（expert），只激活被选中的 expert 做计算。这样模型参数量很大，但每个 token 的实际计算量不变。

### 1.2 MoE 模块介绍

MoE 层的入口是 `models/deepseek_v2.py` → `DeepseekV2MoE`，它包含四个子模块：

`**self.gate**` (`MoEGate`) — 一个线性层 `(hidden_size, n_experts)` = `(6144, 256)`，把 token 的隐藏状态映射成 256 个分数，表示"这个 token 应该被哪些 expert 处理"。

`**self.topk**` (`TopK`) — 从 Gate 输出的 256 个分数中选出最高的 8 个，得到 `topk_ids`（选中哪 8 个 expert）和 `topk_weights`（对应的权重，归一化后和为 1）。GLM-5.1 配置为 `n_group=1, topk_group=1`，底层仍调用 `layers/moe/topk.py` → `biased_grouped_topk_impl()`，在第 2 章详述。

`**self.experts**` (`DeepEPMoE` 或 `FusedMoE`) — 256 个结构相同、参数独立的 SwiGLU FFN（见第 3 章）。每个 token 只经过被 TopK 选中的 8 个，结果按 topk_weights 加权求和。这是 MoE 计算量最大的部分，也是优化的重点。具体使用哪个实现类由 `get_moe_impl_class()` 根据配置决定（EP 模式下用 `DeepEPMoE`，否则用 `FusedMoE`）。

`**self.shared_experts**` (`DeepseekV2MLP`) — 一个所有 token 都无条件经过的 FFN，结构和 routed expert 相同（SwiGLU FFN：gate_up → SiLU(gate) ⊙ up → down）。GLM-5.1 有 1 个 shared expert，`intermediate_size = 2048`（和 routed expert 一样大）。Shared expert 的作用是提供"通用基线"计算，即无论 token 被路由到哪些 routed expert，shared expert 的输出都会加进去。Shared expert 和 routed experts 是**并行**执行的（不存在先后依赖），SGLang 利用这一点做 overlap。实现上，shared expert 可以 fuse 到 routed expert 中（作为第 257 个 expert 一起计算，由 `num_fused_shared_experts` 控制），也可以独立计算。

最终输出：

```
output = shared_expert(x) + routed_scaling_factor × Σ(topk_weight_i × routed_expert_i(x))    (i = 1..8)
```

其中 `routed_scaling_factor` 是一个配置常量，可以融合到 topk_weights 中（`should_fuse_routed_scaling_factor_in_topk`），也可以在最后单独乘。源码中通过 `x.add_(final_hidden_states, alpha=self.routed_scaling_factor)` 实现。

### 1.3 forward() 逻辑

`DeepseekV2MoE.forward()` 根据是否启用 all-to-all MoE（EP / Mooncake / Mori 等后端）选择不同的前向路径：

```
DeepseekV2MoE.forward(hidden_states, forward_batch)
  │
  ├─ _enable_a2a_moe == False
  │    ├─ alt_stream 可用 → forward_normal_dual_stream()   # shared expert 和 routed expert 双流并行 (alt = alternative)
  │    └─ 否则            → forward_normal()               # 非 EP 的标准 MoE
  │
  └─ _enable_a2a_moe == True
       └─ forward_deepep()                                # EP 模式（本文重点）
```

本文重点关注 EP 模式，即走到 `forward_deepep()`。其完整调用链如下：

```
DeepseekV2MoE.forward_deepep(hidden_states, forward_batch)  # models/deepseek_v2.py
  │
  ├─ self.gate(hidden_states) → router_logits           # MoEGate.forward() → (num_tokens, 256)
  │
  ├─ self._forward_shared_experts(hidden_states)        # DeepseekV2MLP (shared expert 并行计算)
  │    → shared_output                                  # (num_tokens, 6144)
  │
  ├─ self.topk(hidden_states, router_logits)            # TopK → biased_grouped_topk_impl()
  │    → TopKOutput(topk_ids, topk_weights)             # (num_tokens, 8)
  │
  └─ self.experts(hidden_states, topk_output)           # self.experts = DeepEPMoE
       │                                                # layers/moe/ep_moe/layer.py
       └─ DeepEPMoE.forward_impl()
            ├─ dispatcher.dispatch(hidden_states, topk_output)   # DeepEPDispatcher
            │    → dispatch_output
            ├─ run_moe_core(dispatch_output)
            │    → combine_input
            └─ dispatcher.combine(combine_input)
                 → final_hidden_states                  # (num_tokens, 6144)
```

### 1.4 GLM-5.1 MoE 配置

本文后续章节中出现的 tensor shape、expert 数量等具体数值均基于 GLM-5.1 的模型配置，其来自 `/local8/GLM-5.1-NVFP4-TensorRT/config.json`：


| 参数                          | 值        | 说明                           |
| --------------------------- | -------- | ---------------------------- |
| hidden_size                 | 6144     | token 的隐藏维度                  |
| intermediate_size           | 12288    | dense FFN 层的中间维度（非 MoE 层）    |
| moe_intermediate_size       | 2048     | 每个 MoE expert 的中间维度          |
| n_routed_experts            | 256      | 路由专家总数                       |
| n_shared_experts            | 1        | shared expert 数量             |
| num_experts_per_tok (top-k) | 8        | 每 token 激活 8 个 routed expert |
| n_group                     | 1        | 将 256 expert 分为 1 组          |
| topk_group                  | 1        | 每次选 1 个组                     |
| topk_method                 | noaux_tc | 使用 sigmoid + bias 路由         |


注意 `intermediate_size`（12288）是 dense FFN 层用的，MoE expert 用的是 `moe_intermediate_size`（2048），两者差 6 倍。这就是 MoE 的设计理念：单个 expert 很小（2048），但有 256 个，总参数量大但每 token 只激活 8 个。

## 2. Gate 与 Routing

> **代码定位**：`DeepseekV2MoE.forward_deepep()` → `router_logits = self.gate(hidden_states)` →`topk_output = self.topk(hidden_states, router_logits)`
> 文件：`models/deepseek_v2.py`（调用端），`layers/moe/topk.py`（路由实现）

### 2.1 Gate 计算（MoEGate）

Gate 是一个线性层，将 hidden_states 映射到 expert 维度上的得分。代码：`models/deepseek_v2.py` → `MoEGate`。

`MoEGate.__init__()` 定义两个可学习参数：

- `self.weight`: `nn.Parameter((n_routed_experts, hidden_size))` = `(256, 6144)` — gate 线性层的权重
- `self.e_score_correction_bias`: `nn.Parameter((n_routed_experts,))` = `(256,)` — 仅当 `topk_method == "noaux_tc"` 时存在，用于 GLM-5.1 的 bias-based 路由（见 2.2 节）。这个 bias 在 `DeepseekV2MoE.__init__()` 中通过 `correction_bias=self.gate.e_score_correction_bias` 传给 `TopK`

`MoEGate.forward()` 的计算本质就是一次矩阵乘法：

```
router_logits = hidden_states @ gate_weight.T
```

- `hidden_states`: `(num_tokens, 6144)` — 输入
- `gate_weight`: `(256, 6144)` — 可学习参数
- `router_logits`: `(num_tokens, 256)` — 每 token 对 256 expert 的原始得分

实现上有一个针对小 batch 的优化（源码 `MoEGate.forward()`，主要受益场景是 decode）：源码里的专用 router GEMM 条件不匹配 GLM-5.1 的 hidden_size=6144，通常走 `F.linear`。小 batch 下标准 GEMM 效率不高，专用 kernel 可以减少 launch overhead。

### 2.2 GLM-5.1 Grouped TopK 路由

GLM-5.1 使用 biased grouped topk（而不是简单的 top-k）。代码：`layers/moe/topk.py` → `biased_grouped_topk_impl()`，由 `TopK.forward_cuda()` → `select_experts()` 调用。具体步骤：

GLM-5.1 用 sigmoid 而非 softmax 来计算 expert 分数（`topk_method = "noaux_tc"`，其中 noaux = No Auxiliary loss，tc = Token Choice）。softmax 会让 256 个 expert 的分数互相竞争（和为 1），sigmoid 则让每个 expert 的分数独立（互不影响）。独立分数配合可学习的 `correction_bias`，可以通过调高冷门 expert 的 bias 来引导负载均衡，而不需要传统 MoE 的辅助平衡损失函数（aux loss）。

```python
# ① sigmoid 计分
scores = gating_output.sigmoid()                       # (num_token, 256), 每个值在 [0, 1]

# ② 加 bias（只影响选择，不影响最终权重）
scores_for_choice = scores + correction_bias.unsqueeze(0)  # (num_token, 256)

# ③ 分组：GLM-5.1 配置为 256 experts → 1 group，取组内 top-2 分数之和
group_scores = (scores_for_choice
    .view(num_token, num_expert_group, -1)             # (num_token, 1, 256)
    .topk(2, dim=-1)[0]                                # 每组 top-2
    .sum(dim=-1))                                      # (num_token, 1)

# ④ 选 top-1 group
group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # (num_token, 1)

# ⑤ 掩码：只保留被选中 group 内的 expert，其余置 -inf
group_mask = torch.zeros_like(group_scores)
group_mask.scatter_(1, group_idx, 1)                   # (num_token, 1) one-hot
score_mask = group_mask.unsqueeze(-1).expand(...).reshape(num_token, -1)  # (num_token, 256)
tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

# ⑥ 在未掩码的 expert 中选 top-8
_, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1)  # (num_token, 8)

# ⑦ 权重取原始 sigmoid 分数（不含 bias），然后归一化
topk_weights = scores.gather(1, topk_ids)              # (num_token, 8)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renormalize
```

**为什么要保留 grouped topk 代码路径**：通用实现支持多 group 负载均衡；GLM-5.1 当前 `n_group=1, topk_group=1`，等价于在单组 256 experts 内选 top-8。

### 2.3 示例

假设 batch_size=4（4 个 token），简化为 8 expert / 1 group / top-2 expert：

```
Token 0 的 routing 结果:
  sigmoid scores: [0.8, 0.3, 0.1, 0.7, 0.2, 0.9, 0.4, 0.5]
                   └────────────── Group 0 ──────────────┘
  Group 0 top-2 之和: 0.9+0.8 = 1.7
  选 top-1 group: Group 0
  在 Group 0 中选 top-2 expert: expert 5 (0.9), expert 0 (0.8)

最终: topk_ids[0] = [5, 0], topk_weights[0] = [0.53, 0.47] (renormalized)
  注: topk_weights 来自所有被选中 expert 的原始 sigmoid 分数 (0.9, 0.8)，归一化后得到 (0.53, 0.47)
```

## 3. 单个 Expert 的结构（SwiGLU FFN）

> **代码定位**：`DeepseekV2MoE.__init__()` → `self.experts`（MoE layer，由 `get_moe_impl_class()` 返回，EP 模式下为 `DeepEPMoE`，内部包含多个 SwiGLU FFN expert）/ `self.shared_experts`（`DeepseekV2MLP`，独立的 SwiGLU FFN）。
> 文件：`models/deepseek_v2.py`（`DeepseekV2MLP`），`layers/moe/ep_moe/layer.py`（`DeepEPMoE`），`layers/activation.py`（`SiluAndMul`）

在进入 EP 通信之前，先了解每个 expert 内部算什么。`self.experts` 是一个 MoE layer 封装（通过 `get_moe_impl_class()` 选择具体实现：EP 模式下为 `DeepEPMoE`，非 EP 下为 `FusedMoE` / `FlashInferFP4MoE` 等），其内部的每个 routed expert 和 shared expert 的**计算结构**相同，都是 SwiGLU FFN。一个 expert 的完整计算是：

`output = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down`

其中 `@` 表示矩阵乘，`⊙` 表示逐元素乘。SwiGLU 的名字来自 **Swi**sh（SiLU）与 **GLU**（Gated Linear Unit）的组合：

**SiLU**（也叫 Swish）是一个激活函数：`SiLU(x) = x ⊙ σ(x)`，其中 σ 是 sigmoid。σ(x) 输出范围 (0, 1)，相当于一个"软开关"，即让输入自己决定自己被保留多少：x 为正时近乎原样通过，x 为负时被抑制趋近于 0，过渡区间平滑连续。相比 ReLU 的硬截断（x < 0 直接归零），SiLU 在负值区域仍有梯度流动，避免了死神经元问题。实现：`layers/activation.py` → `SiluAndMul`，底层调用 `sgl_kernel.silu_and_mul`，接收 gate_up 拼接结果，内部完成 SiLU + 逐元素乘。

**GLU**（Gated Linear Unit）是一种门控机制：输入 x 同时经过两路并行的线性变换，即 gate 分支 `x @ W_gate` 和 up 分支 `x @ W_up`，gate 分支过激活函数后作为"门"与 up 分支逐元素相乘。这让网络可以选择性地放大或抑制 up 分支各维度的信息，比单路 FFN 有更强的表达能力。

SiLU 提供平滑非线性，GLU 提供维度级别的门控选择，两者结合使得 SwiGLU FFN 在同等参数量下比标准 ReLU FFN 有更好的训练效果（Shazeer, 2020）。代价是多了一路线性变换（gate 和 up 两个权重矩阵而非一个），但 SGLang 通过将两者合并为一次 GEMM 来消除这个开销（见下文）。SwiGLU 的输出维度是 intermediate_size（2048），**down_proj**（`W_down`）将其投射回 hidden_size（6144）。

### 3.1 权重矩阵与数据流

三个权重矩阵的大小：

- `W_gate`: (2048, 6144) — 从 hidden 映射到 intermediate
- `W_up`:   (2048, 6144) — 同上，和 gate 并行
- `W_down`: (6144, 2048) — 从 intermediate 映射回 hidden

对一个 token（形状 `(1, 6144)`）来说：

```
x: (1, 6144)
    │
    ├─ gate_proj: x @ W_gate.T ─→ gate: (1, 2048) ──┐
    │                                               ├─ SiLU(gate) ⊙ up → (1, 2048)
    ├─ up_proj:   x @ W_up.T   ─→ up:   (1, 2048) ──┘       │
    │   （gate_proj 和 up_proj 是并行的两个线性层）             │
    │                                                       ↓
    └────────────────────────────────────── down_proj: (1, 2048) @ W_down.T → (1, 6144)
```

gate_proj 和 up_proj 是并行计算的（输入相同，权重不同），然后 gate 过 SiLU 激活后与 up 逐元素相乘（这就是 SwiGLU），最后经过 down_proj 投射回 hidden 维度。

### 3.2 gate_up 合并优化

gate_proj 和 up_proj 合并为一个矩阵 **w1**（shape `[2*moe_intermediate_size, hidden_size]` = `[4096, 6144]`），一次 GEMM 同时算出 gate 和 up。down_proj 对应 **w2**（shape `[hidden_size, moe_intermediate_size]` = `[6144, 2048]`）。

因此每个 expert 的计算简化为两次矩阵乘法：

- **Gemm1** (gate_up): `(m, 6144) @ (4096, 6144)^T → (m, 4096)` — 前 2048 列是 gate，后 2048 列是 up
- SiLU 激活 + 逐元素乘：`SiLU(前2048列) ⊙ 后2048列 → (m, 2048)`
- **Gemm2** (down): `(m, 2048) @ (6144, 2048)^T → (m, 6144)`

### 3.3 代码路径

- **Shared expert FFN**：`models/deepseek_v2.py` → `DeepseekV2MLP`，包含 `gate_up_proj`（MergedColumnParallelLinear）和 `down_proj`（RowParallelLinear），激活函数为 `SiluAndMul()`
- **Routed expert 权重**：在 MoE layer 中以 `w13_weight`（gate_up 合并）和 `w2_weight`（down）存储，由 fused MoE kernel 统一计算
- **SiLU ⊙ mul 激活**：独立 expert 走 `layers/activation.py` → `SiluAndMul`；MoE fused kernel 内部直接调用 `sgl_kernel.silu_and_mul`（`layers/moe/moe_runner/triton.py`、`deep_gemm.py`）；low-latency mode 还有和 FP4 量化融合的版本 `silu_and_mul_masked_post_quant_fwd`（`layers/moe/ep_moe/kernels.py`）

### 3.4 Shared Expert vs Routed Expert 的实现差异

数学上每个 expert（包括 shared expert）都在做同样的 SwiGLU 计算，但实现方式不同：

- **Shared expert**（`DeepseekV2MLP`）：使用标准线性层（`MergedColumnParallelLinear` + `RowParallelLinear`），每次处理所有 token，走 dense GEMM（FP4 模型下是 Dense FP4 GEMM，和 Attention 的 q/k/v/o_proj 一样）。只有 1 个 expert 且所有 token 都经过，不需要路由和分组，标准 dense GEMM 已经是最优选择
- **Routed expert**（`DeepEPMoE` / `FusedMoE` 等）：256 个 expert 的权重合并存储为 `w13_weight`（gate_up，w1 和 w3 拼接，沿用 Mixtral 命名）和 `w2_weight`（down），通过 fused MoE kernel 一次处理多个 expert 的不同 token 子集。EP8 时每卡 32 个 local expert，MoE kernel 解决的是"多个小 expert 并行计算不同 token 子集"的调度问题，需要 `GroupProblemShape` 或 `masked_m` 来处理不等长的 per-expert batch

## 4. Expert Parallelism — DeepEPMoE

> **代码定位**：`DeepseekV2MoE.__init__()` → `self.experts = get_moe_impl_class()(...)` — EP 模式下返回 `DeepEPMoE`，其 `self.dispatcher` 为 `DeepEPDispatcher`，`dispatcher`内部持有 DeepEP `Buffer` 对象。
> 文件：`models/deepseek_v2.py`（MoE 实现选择），`layers/moe/ep_moe/layer.py`（`DeepEPMoE`），`layers/moe/token_dispatcher/deepep.py`（`DeepEPDispatcher`）

### 4.1 为什么需要 EP

GLM-5.1 有 256 个 routed expert。在 NVFP4 量化下：

```
每 expert 参数量:
  w1: 4096 × 6144 × 0.5 byte (FP4)  ≈ 12.6 MB
  w2: 6144 × 2048 × 0.5 byte (FP4)  ≈  6.3 MB
  blockscale + alpha 等辅助参数       ≈  2.4 MB（粗略估算）
  合计 ≈ 21 MB / expert

256 experts × 21 MB ≈ 5.4 GB
```

以上仅为单层的 routed expert 参数量，GLM-5.1 共有 75 层 MoE，全部 expert 参数远超单卡显存。EP 将 expert 分散到多张卡上：

- 每卡只需存储部分 expert 的参数，释放显存给 KV cache 和 activation
- 配合 DP，多个 rank 的 token 汇聚到 expert 所在卡，每个 expert 处理的 batch 更大，GPU 利用率更高
- 代价是需要 GPU 间通信交换 token

### 4.2 EP 分配与 Token 交换

以 **EP=8**（8 张 GPU，每卡 32 expert）为例：

```
GPU-0: Expert 0-31       GPU-1: Expert 32-63
GPU-2: Expert 64-95      GPU-3: Expert 96-127
GPU-4: Expert 128-159    GPU-5: Expert 160-191
GPU-6: Expert 192-223    GPU-7: Expert 224-255
```

每卡的参数量：32 × 21 MB ≈ 0.7 GB（vs 不用 EP 时的 5.4 GB）。

假设 GPU-0 上有一个 token，routing 结果选中了 expert [3, 35, 67, 99, 130, 162, 195, 230]：

```
GPU-0 的 token:
  expert 3   → 留在 GPU-0 本地计算
  expert 35  → 发送到 GPU-1
  expert 67  → 发送到 GPU-2
  expert 99  → 发送到 GPU-3
  expert 130 → 发送到 GPU-4
  expert 162 → 发送到 GPU-5
  expert 195 → 发送到 GPU-6
  expert 230 → 发送到 GPU-7
```

每张 GPU 上的每个 token 都可能需要发送到任意一张持有对应 expert 的卡，同时也会接收来自其他卡的 token。这是一个 all-to-all 通信模式 — 标准 all-to-all 等价于分布式矩阵转置：把 N 个 GPU 各自的发送数据看作矩阵的一行，通信完成后每个 GPU 拿到对应的一列。MoE 场景下属于不等长的 all-to-all，每对 GPU 之间交换的 token 数量取决于 routing 结果。

### 4.3 执行流程：dispatch → compute → combine

`DeepEPMoE`（`layers/moe/ep_moe/layer.py`）继承自 `FusedMoE`，它的 `forward_impl()` 分为三步：dispatch → compute → combine。

```python
def forward_impl(self, hidden_states, topk_output):
    # dispatch: 将本 GPU 的 token 发送到目标 expert 所在的 GPU（all-to-all 通信）
    dispatch_output = self.dispatcher.dispatch(
        hidden_states=hidden_states, topk_output=topk_output
    )
    # compute: 在本 GPU 上对收到的 token 执行 SwiGLU FFN（CUTLASS 或 CuTeDSL GEMM）
    combine_input = self.run_moe_core(dispatch_output)
    # combine: 反向 all-to-all 通信，将计算结果发回 token 原来所在的 GPU，按 routing weight 加权汇总
    hidden_states = self.dispatcher.combine(combine_input=combine_input)
    return hidden_states
```

沿用 4.2 的例子（EP=8，token 路由到 expert [3, 35, 67, 99, 130, 162, 195, 230]），三步的数据流：

**Dispatch**

GPU-0 把这个 token 的 hidden_states 发送到 GPU-1/2/3/4/5/6/7（本地的 expert 3 不需要发送）。每张目标 GPU 都收到同一份 hidden_states 副本，但只会用本卡上的 expert 来处理。

**Compute**

各 GPU 独立执行 SwiGLU FFN。以 GPU-1 为例，它收到了这个 token，需要经过 expert 35 前向计算。计算完成后，GPU-1 上得到结果向量 `out_35`。其他 GPU 同理各自处理本卡上的 expert。

**Combine**

反向 all-to-all 通信，各 GPU 把 expert 的计算结果发回 token 原来所在的 GPU（GPU-0）。GPU-0 需要收集全部 8 个 expert 的结果，按 routing weight 做加权求和：

```
final_output = out_3×w_3 + out_35×w_35 + out_67×w_67 + out_99×w_99
             + out_130×w_130 + out_162×w_162 + out_195×w_195 + out_230×w_230
```

即 `Σ_k (topk_weight_k × expert_k)`。combine 不只是 "搬数据"，还需要对多个 expert 的结果做归约（reduce），这是 EP 模式下的核心操作。

补充：
Normal mode 和 Low-latency mode 在加权求和的位置上有差异，这个选择直接影响了通信量和 kernel 开销。

Normal mode 在 compute 阶段多一步 `post_reorder` kernel：它在本卡上把同一 token 的多个 expert 结果按 `topk_weight` 加权合并成一个 partial sum，combine 阶段只需跨卡做不带权的 sum。这样每个源 token 从每张远端 GPU 只收一个合并后的向量，通信量较小。代价是多一个 kernel launch。

LL mode 跳过 `post_reorder`，把逐 expert 的原始结果直接发回源 GPU，在 combine 接收侧一次性完成 weighted sum。通信量更大（每个 expert 各一个向量），但省掉了额外 kernel 的开销，也避免打断 combine 与 shared expert 计算的 overlap 流水线。

两种策略的最终数学结果完全相同，具体实现在后续的 combine 章节中展开。
