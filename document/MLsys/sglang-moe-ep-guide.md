# SGLang DeepSeek MoE & Expert Parallelism 原理详解

本文档基于 SGLang 代码库中 DeepSeek-V3 的 MoE 实现，以 `DeepseekV2MoE` 类为入口，按代码调用链逐层展开。每一章对应调用链的一层，读者从第一章开始读，相当于在逐层 step-into 代码。

重点覆盖 EP（Expert Parallelism）的两种 dispatch 模式：normal（高吞吐，常用于 prefill）和 low_latency（低延迟，常用于 decode，但也可用于 prefill），以及 TP（Tensor Parallelism）模式的 FP4 MoE 计算。通过具体的 tensor shape 示例说明数据流向。

代码基线：`sglang/python/sglang/srt/`（以下路径省略此前缀）

## 1. DeepseekV3 MoE 概览

> **代码定位**：`models/deepseek_v2.py` → `DeepseekV2MoE.__init__()` / `forward()`

### 1.1 MoE 在 Transformer Block 中的位置

以 DeepSeek-V3 单个 transformer block 为例，MoE 替代了标准 FFN 的位置：

```
DeepseekV2Block.forward()
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

DeepSeek-V3 有 61 个 MoE 层（另有 3 个 dense FFN 层，不走 MoE）。每个 MoE 层包含 256 个 routed expert 和 1 个 shared expert。

Mixture of Experts 是一种条件计算架构：每个 token 不经过全部参数，而是通过路由机制选出若干"专家"（expert），只激活被选中的 expert 做计算。这样模型参数量很大，但每个 token 的实际计算量不变。

### 1.2 MoE 模块介绍

MoE 层的入口是 `models/deepseek_v2.py` → `DeepseekV2MoE`，它包含四个子模块：

`**self.gate**` (`MoEGate`) — 一个线性层 `(hidden_size, n_experts)` = `(7168, 256)`，把 token 的隐藏状态映射成 256 个分数，表示"这个 token 应该被哪些 expert 处理"。

`**self.topk**` (`TopK`) — 从 Gate 输出的 256 个分数中选出最高的 8 个，得到 `topk_ids`（选中哪 8 个 expert）和 `topk_weights`（对应的权重，归一化后和为 1）。DeepSeek-V3 用分组路由（先选组再选 expert），底层调用 `layers/moe/topk.py` → `biased_grouped_topk_impl()`，在第 2 章详述。

`**self.experts**` (`DeepEPMoE` 或 `FusedMoE`) — 256 个结构相同、参数独立的 SwiGLU FFN（见第 3 章）。每个 token 只经过被 TopK 选中的 8 个，结果按 topk_weights 加权求和。这是 MoE 计算量最大的部分，也是优化的重点。具体使用哪个实现类由 `get_moe_impl_class()` 根据配置决定（EP 模式下用 `DeepEPMoE`，否则用 `FusedMoE`）。

`**self.shared_experts**` (`DeepseekV2MLP`) — 一个所有 token 都无条件经过的 FFN，结构和 routed expert 相同（SwiGLU FFN：gate_up → SiLU(gate) ⊙ up → down）。DeepSeek-V3 有 1 个 shared expert，`intermediate_size = 2048`（和 routed expert 一样大）。Shared expert 的作用是提供"通用基线"计算，即无论 token 被路由到哪些 routed expert，shared expert 的输出都会加进去。Shared expert 和 routed experts 是**并行**执行的（不存在先后依赖），SGLang 利用这一点做 overlap（详见第 11 章）。实现上，shared expert 可以 fuse 到 routed expert 中（作为第 257 个 expert 一起计算，由 `num_fused_shared_experts` 控制），也可以独立计算。

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
  │    → shared_output                                  # (num_tokens, 7168)
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
                 → final_hidden_states                  # (num_tokens, 7168)
```

### 1.4 DeepSeek-V3 MoE 配置

本文后续章节中出现的 tensor shape、expert 数量等具体数值均基于 DeepSeek-V3 的模型配置, 其来自 checkpoint 的 `config.json`：


| 参数                          | 值        | 说明                           |
| --------------------------- | -------- | ---------------------------- |
| hidden_size                 | 7168     | token 的隐藏维度                  |
| intermediate_size           | 18432    | dense FFN 层的中间维度（非 MoE 层）    |
| moe_intermediate_size       | 2048     | 每个 MoE expert 的中间维度          |
| n_routed_experts            | 256      | 路由专家总数                       |
| n_shared_experts            | 1        | shared expert 数量             |
| num_experts_per_tok (top-k) | 8        | 每 token 激活 8 个 routed expert |
| n_group                     | 8        | 将 256 expert 分为 8 组          |
| topk_group                  | 4        | 每次选 4 个组                     |
| topk_method                 | noaux_tc | 使用 sigmoid + bias 路由         |


注意 `intermediate_size`（18432）是 dense FFN 层用的，MoE expert 用的是 `moe_intermediate_size`（2048），两者差 9 倍。这就是 MoE 的设计理念：单个 expert 很小（2048），但有 256 个，总参数量大但每 token 只激活 8 个。

## 2. Gate 与 Routing

> **代码定位**：`DeepseekV2MoE.forward_deepep()` → `router_logits = self.gate(hidden_states)` →`topk_output = self.topk(hidden_states, router_logits)`
> 文件：`models/deepseek_v2.py`（调用端），`layers/moe/topk.py`（路由实现）

### 2.1 Gate 计算（MoEGate）

Gate 是一个线性层，将 hidden_states 映射到 expert 维度上的得分。代码：`models/deepseek_v2.py` → `MoEGate`。

`MoEGate.__init__()` 定义两个可学习参数：

- `self.weight`: `nn.Parameter((n_routed_experts, hidden_size))` = `(256, 7168)` — gate 线性层的权重
- `self.e_score_correction_bias`: `nn.Parameter((n_routed_experts,))` = `(256,)` — 仅当 `topk_method == "noaux_tc"` 时存在，用于 DeepSeek-V3 的 bias-based 路由（见 2.2 节）。这个 bias 在 `DeepseekV2MoE.__init__()` 中通过 `correction_bias=self.gate.e_score_correction_bias` 传给 `TopK`

`MoEGate.forward()` 的计算本质就是一次矩阵乘法：

```
router_logits = hidden_states @ gate_weight.T
```

- `hidden_states`: `(num_tokens, 7168)` — 输入
- `gate_weight`: `(256, 7168)` — 可学习参数
- `router_logits`: `(num_tokens, 256)` — 每 token 对 256 expert 的原始得分

实现上有一个针对小 batch 的优化（源码 `MoEGate.forward()`，主要受益场景是 decode）：当 token 数 ≤ 16 且在 SM90+ GPU 上时，使用专用的 `dsv3_router_gemm`（输出 float32）；否则走 `F.linear`。小 batch 下标准 GEMM 效率不高，专用 kernel 可以减少 launch overhead。

### 2.2 DeepSeek-V3 Grouped TopK 路由

DeepSeek-V3 使用 biased grouped topk（而不是简单的 top-k）。代码：`layers/moe/topk.py` → `biased_grouped_topk_impl()`，由 `TopK.forward_cuda()` → `select_experts()` 调用。具体步骤：

DeepSeek-V3 用 sigmoid 而非 softmax 来计算 expert 分数（`topk_method = "noaux_tc"`，其中 noaux = No Auxiliary loss，tc = Token Choice）。softmax 会让 256 个 expert 的分数互相竞争（和为 1），sigmoid 则让每个 expert 的分数独立（互不影响）。独立分数配合可学习的 `correction_bias`，可以通过调高冷门 expert 的 bias 来引导负载均衡，而不需要传统 MoE 的辅助平衡损失函数（aux loss）。这就是 DeepSeek-V3 论文中 "Auxiliary-Loss-Free Load Balancing" 的实现方式。

```python
# ① sigmoid 计分
scores = gating_output.sigmoid()                       # (num_token, 256), 每个值在 [0, 1]

# ② 加 bias（只影响选择，不影响最终权重）
scores_for_choice = scores + correction_bias.unsqueeze(0)  # (num_token, 256)

# ③ 分组：256 experts → 8 groups, 每组 32 experts，取每组内 top-2 分数之和
group_scores = (scores_for_choice
    .view(num_token, num_expert_group, -1)             # (n, 8, 32)
    .topk(2, dim=-1)[0]                                # 每组 top-2
    .sum(dim=-1))                                      # (n, 8)

# ④ 选 top-4 groups
group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # (n, 4)

# ⑤ 掩码：只保留被选中 group 内的 expert，其余置 -inf
group_mask = torch.zeros_like(group_scores)
group_mask.scatter_(1, group_idx, 1)                   # (n, 8) one-hot
score_mask = group_mask.unsqueeze(-1).expand(...).reshape(num_token, -1)  # (n, 256)
tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

# ⑥ 在未掩码的 expert 中选 top-8
_, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1)  # (n, 8)

# ⑦ 权重取原始 sigmoid 分数（不含 bias），然后归一化
topk_weights = scores.gather(1, topk_ids)              # (n, 8)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renormalize
```

**为什么要分组路由**：直接从 256 expert 选 8 个容易扎堆（部分 expert 被高频选中，其余空闲）。先选 group 再选 expert，保证激活的 expert 分布在不同区域，负载更均衡。

### 2.3 示例

假设 batch_size=4（4 个 token），简化为 16 expert / 2 groups / top-1 group / top-2 expert：

```
Token 0 的 routing 结果:
  sigmoid scores: [0.8, 0.3, 0.1, 0.7, | 0.2, 0.9, 0.4, 0.5, | ...]
                   └── Group 0 ──────┘   └── Group 1 ──────┘
  Group 0 top-2 之和: 0.8+0.7 = 1.5
  Group 1 top-2 之和: 0.9+0.5 = 1.4
  选 top-1 group: Group 0
  在 Group 0 中选 top-2 expert: expert 0 (0.8), expert 3 (0.7)

最终: topk_ids[0] = [0, 3], topk_weights[0] = [0.53, 0.47] (renormalized)
  注: topk_weights 来自所有被选中 expert 的原始 sigmoid 分数 (0.8, 0.7)，归一化后得到 (0.53, 0.47)
```

## 3. 单个 Expert 的结构（SwiGLU FFN）

> **代码定位**：`DeepseekV2MoE.__init__()` → `self.experts`（MoE layer，由 `get_moe_impl_class()` 返回，EP 模式下为 `DeepEPMoE`，内部包含多个 SwiGLU FFN expert）/ `self.shared_experts`（`DeepseekV2MLP`，独立的 SwiGLU FFN）。
> 文件：`models/deepseek_v2.py`（`DeepseekV2MLP`），`layers/moe/ep_moe/layer.py`（`DeepEPMoE`），`layers/activation.py`（`SiluAndMul`）

在进入 EP 通信之前，先了解每个 expert 内部算什么。`self.experts` 是一个 MoE layer 封装（通过 `get_moe_impl_class()` 选择具体实现：EP 模式下为 `DeepEPMoE`，非 EP 下为 `FusedMoE` / `FlashInferFP4MoE` 等），其内部的每个 routed expert 和 shared expert 的**计算结构**相同，都是 SwiGLU FFN。一个 expert 的完整计算是：

`output = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down`

其中 `@` 表示矩阵乘，`⊙` 表示逐元素乘。SwiGLU 的名字来自 **Swi**sh（SiLU）与 **GLU**（Gated Linear Unit）的组合：

**SiLU**（也叫 Swish）是一个激活函数：`SiLU(x) = x ⊙ σ(x)`，其中 σ 是 sigmoid。σ(x) 输出范围 (0, 1)，相当于一个"软开关"，即让输入自己决定自己被保留多少：x 为正时近乎原样通过，x 为负时被抑制趋近于 0，过渡区间平滑连续。相比 ReLU 的硬截断（x < 0 直接归零），SiLU 在负值区域仍有梯度流动，避免了死神经元问题。实现：`layers/activation.py` → `SiluAndMul`，底层调用 `sgl_kernel.silu_and_mul`，接收 gate_up 拼接结果，内部完成 SiLU + 逐元素乘。

**GLU**（Gated Linear Unit）是一种门控机制：输入 x 同时经过两路并行的线性变换，即 gate 分支 `x @ W_gate` 和 up 分支 `x @ W_up`，gate 分支过激活函数后作为"门"与 up 分支逐元素相乘。这让网络可以选择性地放大或抑制 up 分支各维度的信息，比单路 FFN 有更强的表达能力。

SiLU 提供平滑非线性，GLU 提供维度级别的门控选择，两者结合使得 SwiGLU FFN 在同等参数量下比标准 ReLU FFN 有更好的训练效果（Shazeer, 2020）。代价是多了一路线性变换（gate 和 up 两个权重矩阵而非一个），但 DeepSeek 通过将两者合并为一次 GEMM 来消除这个开销（见下文）。SwiGLU 的输出维度是 intermediate_size（2048），**down_proj**（`W_down`）将其投射回 hidden_size（7168）。

### 3.1 权重矩阵与数据流

三个权重矩阵的大小：

- `W_gate`: (2048, 7168) — 从 hidden 映射到 intermediate
- `W_up`:   (2048, 7168) — 同上，和 gate 并行
- `W_down`: (7168, 2048) — 从 intermediate 映射回 hidden

对一个 token（形状 `(1, 7168)`）来说：

```
x: (1, 7168)
    │
    ├─ gate_proj: x @ W_gate.T ─→ gate: (1, 2048) ──┐
    │                                               ├─ SiLU(gate) ⊙ up → (1, 2048)
    ├─ up_proj:   x @ W_up.T   ─→ up:   (1, 2048) ──┘       │
    │   （gate_proj 和 up_proj 是并行的两个线性层）             │
    │                                                       ↓
    └────────────────────────────────────── down_proj: (1, 2048) @ W_down.T → (1, 7168)
```

gate_proj 和 up_proj 是并行计算的（输入相同，权重不同），然后 gate 过 SiLU 激活后与 up 逐元素相乘（这就是 SwiGLU），最后经过 down_proj 投射回 hidden 维度。

### 3.2 gate_up 合并优化

gate_proj 和 up_proj 合并为一个矩阵 **w1**（shape `[2*moe_intermediate_size, hidden_size]` = `[4096, 7168]`），一次 GEMM 同时算出 gate 和 up。down_proj 对应 **w2**（shape `[hidden_size, moe_intermediate_size]` = `[7168, 2048]`）。

因此每个 expert 的计算简化为两次矩阵乘法：

- **Gemm1** (gate_up): `(m, 7168) @ (4096, 7168)^T → (m, 4096)` — 前 2048 列是 gate，后 2048 列是 up
- SiLU 激活 + 逐元素乘：`SiLU(前2048列) ⊙ 后2048列 → (m, 2048)`
- **Gemm2** (down): `(m, 2048) @ (7168, 2048)^T → (m, 7168)`

### 3.3 代码路径

- **Shared expert FFN**：`models/deepseek_v2.py` → `DeepseekV2MLP`，包含 `gate_up_proj`（MergedColumnParallelLinear）和 `down_proj`（RowParallelLinear），激活函数为 `SiluAndMul()`
- **Routed expert 权重**：在 MoE layer 中以 `w13_weight`（gate_up 合并）和 `w2_weight`（down）存储，由 fused MoE kernel 统一计算
- **SiLU ⊙ mul 激活**：独立 expert 走 `layers/activation.py` → `SiluAndMul`；MoE fused kernel 内部直接调用 `sgl_kernel.silu_and_mul`（`layers/moe/moe_runner/triton.py`、`deep_gemm.py`）；low-latency mode 还有和 FP4 量化融合的版本 `silu_and_mul_masked_post_quant_fwd`（`layers/moe/ep_moe/kernels.py`）

### 3.4 Shared Expert vs Routed Expert 的实现差异

数学上每个 expert（包括 shared expert）都在做同样的 SwiGLU 计算，但实现方式不同：

- **Shared expert**（`DeepseekV2MLP`）：使用标准线性层（`MergedColumnParallelLinear` + `RowParallelLinear`），每次处理所有 token，走 dense GEMM（FP4 模型下是 Dense FP4 GEMM，和 Attention 的 q/k/v/o_proj 一样）。只有 1 个 expert 且所有 token 都经过，不需要路由和分组，标准 dense GEMM 已经是最优选择
- **Routed expert**（`DeepEPMoE` / `FusedMoE` 等）：256 个 expert 的权重合并存储为 `w13_weight`（gate_up，w1 和 w3 拼接，沿用 Mixtral 命名）和 `w2_weight`（down），通过 fused MoE kernel 一次处理多个 expert 的不同 token 子集。MoE kernel 解决的是"64 个小 expert 并行计算不同 token 子集"的调度问题，需要 `GroupProblemShape` 或 `masked_m` 来处理不等长的 per-expert batch

## 4. Expert Parallelism — DeepEPMoE

> **代码定位**：`DeepseekV2MoE.__init__()` → `self.experts = get_moe_impl_class()(...)` — EP 模式下返回 `DeepEPMoE`，其 `self.dispatcher` 为 `DeepEPDispatcher`，`dispatcher`内部持有 DeepEP `Buffer` 对象。
> 文件：`models/deepseek_v2.py`（MoE 实现选择），`layers/moe/ep_moe/layer.py`（`DeepEPMoE`），`layers/moe/token_dispatcher/deepep.py`（`DeepEPDispatcher`）

### 4.1 为什么需要 EP

DeepSeek-V3 有 256 个 expert。在 NVFP4 量化下：

```
每 expert 参数量:
  w1: 4096 × 7168 × 0.5 byte (FP4)  = 14.7 MB
  w2: 7168 × 2048 × 0.5 byte (FP4)  =  7.3 MB
  blockscale + alpha 等辅助参数       ≈  1.0 MB
  合计 ≈ 23 MB / expert

256 experts × 23 MB ≈ 5.9 GB
```

以上仅为单层的 expert 参数量，DeepSeek-V3 共有约 60 层 MoE，全部 expert 参数远超单卡显存。EP 将 expert 分散到多张卡上：

- 每卡只需存储部分 expert 的参数，释放显存给 KV cache 和 activation
- 配合 DP，多个 rank 的 token 汇聚到 expert 所在卡，每个 expert 处理的 batch 更大，GPU 利用率更高
- 代价是需要 GPU 间通信交换 token

### 4.2 EP 分配与 Token 交换

以 **EP=4**（4 张 GPU，每卡 64 expert）为例：

```
GPU-0: Expert 0-63       GPU-1: Expert 64-127
GPU-2: Expert 128-191    GPU-3: Expert 192-255
```

每卡的参数量：64 × 23 MB ≈ 1.5 GB（vs 不用 EP 时的 5.9 GB）。

假设 GPU-0 上有一个 token，routing 结果选中了 expert [3, 5, 67, 70, 130, 133, 195, 198]：

```
GPU-0 的 token:
  expert 3   → 留在 GPU-0 本地计算
  expert 5   → 留在 GPU-0 本地计算
  expert 67  → 发送到 GPU-1
  expert 70  → 发送到 GPU-1
  expert 130 → 发送到 GPU-2
  expert 133 → 发送到 GPU-2
  expert 195 → 发送到 GPU-3
  expert 198 → 发送到 GPU-3
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

## 5. Dispatch 阶段

> **代码定位**：`DeepseekV2MoE.forward_deepep()` → `self.experts()` → `DeepEPMoE.forward_impl()` → `self.dispatcher.dispatch()`
> 文件：`layers/moe/token_dispatcher/deepep.py`（`DeepEPDispatcher` → `_DeepEPDispatcherImplNormal` / `_DeepEPDispatcherImplLowLatency`），DeepEP 库 `deep_ep/buffer.py`（`Buffer`）

Dispatch 将本 GPU 的 token 发送到 expert 所在的 GPU。SGLang 使用 DeepEP 库（DeepSeek 开源）实现 GPU 间 token 交换，核心接口是 `Buffer` 类，提供两种模式。SGLang 侧的封装：`layers/moe/token_dispatcher/deepep.py` → `DeepEPDispatcher`（统一入口），内部分为 `_DeepEPDispatcherImplNormal` 和 `_DeepEPDispatcherImplLowLatency`。

### 5.1 两种模式：Normal 和 Low Latency

DeepEP 提供两种通信模式，SGLang 根据场景自动选择：


| 模式          | DeepEP API                                           | 适用场景                             | 特点              |
| ----------- | ---------------------------------------------------- | -------------------------------- | --------------- |
| Normal      | `buffer.get_dispatch_layout()` + `buffer.dispatch()` | 大 batch（常用于 prefill）             | 高吞吐, all-to-all |
| Low Latency | `buffer.low_latency_dispatch()`                      | 小 batch（常用于 decode，也可用于 prefill） | 低延迟, RDMA 直写    |


**模式选择逻辑**（`layers/moe/utils.py` → `DeepEPMode.resolve()`）：

- AUTO（默认）：根据 batch 大小自动选择（大 batch → Normal, 小 batch → Low Latency）；实际部署中 prefill 阶段倾向 Normal, decode 阶段倾向 Low Latency, 但不是严格绑定
- 可通过 `--deepep-mode normal/low_latency` 强制指定（此时 prefill 和 decode 都用同一模式）

通信模式决定了后续的计算路径（`run_moe_core()` 根据 dispatch 输出格式分支）：

```
DeepEPMoE.forward_impl(dispatch_output)
  │
  ├─ deprecate_flag == True (FP8 + deep_gemm)
  │    └─ super().forward_impl()               # 委托给 FusedMoE 通用框架
  │
  └─ deprecate_flag == False → run_moe_core(dispatch_output)
       │
       ├─ format == DEEPEP_NORMAL
       │    ├─ use_w4afp8 → forward_cutlass_w4afp8()
       │    ├─ modelopt_fp4 → assert False     # 开源无实现
       │    │   (lingjun 分支: forward_cutlass_nvfp4_normal → cutlass_moe_ep_fp4)
       │    └─ else → assert False
       │
       └─ format == DEEPEP_LL
            ├─ flashinfer_cutedsl + modelopt_fp4
            │    → forward_flashinfer_cutedsl()
            │         → flashinfer_cutedsl_moe_masked()  # CuTeDSL (3D masked)
            ├─ use_w4afp8 → forward_cutlass_w4afp8_masked()
            └─ else → assert False
```

**两种模式全面对比**：


| 维度            | Normal                                      | Low-latency                                   |
| ------------- | ------------------------------------------- | --------------------------------------------- |
| 通信方式          | all-to-all (NVLink/RDMA)                    | RDMA 直写 (IBGDA)                               |
| 布局预计算         | 需要 `get_dispatch_layout`                    | 不需要, 预分配固定 buffer                             |
| 输出格式          | 2D `(total_recv, hidden)` 无 padding         | 3D `(num_experts, max_buf, hidden)` 有 padding |
| 计算 kernel     | `cutlass_moe_ep_fp4` (CUTLASS grouped GEMM) | `flashinfer_cutedsl_moe_masked` (CuTeDSL)     |
| 量化            | FP8 通信 + NVFP4 计算                           | FP8 通信 + NVFP4 计算                             |
| 适合 batch size | 大 (数百~数千 token)                             | 小 (数十~数百 token)                               |
| CUDA Graph    | 不支持 (动态 shape)                              | 支持 (固定 buffer size)                           |


### 5.2 dispatch_a / dispatch_b 分阶段设计

SGLang 将 dispatch 和 combine 各拆为 a/b 两阶段，目的是在 a 和 b 之间插入 hook 实现 overlap：

```python
# DeepEPDispatcher.dispatch() 的实际流程
def dispatch(self, hidden_states, topk_output):
    self.dispatch_a(hidden_states, topk_output)   # 启动异步通信（非阻塞）
    if self._deepep_dispatch_hooks is not None:
        self._deepep_dispatch_hooks(self)         # 通信进行中，执行 hook（如 shared expert 计算）
    ret = self.dispatch_b()                       # 等待通信完成，构造输出
    return ret
```

combine 同理（`combine_a()` → hook → `combine_b()`）。

两种模式下 a/b 之间 hook 的 overlap 效果不同：

- **Normal mode**：`dispatch_a()` 只做 FP8 量化和 event capture，实际通信在 `dispatch_b()` 才发生。a/b 之间的 hook 不与通信并行。Normal mode 的 overlap 主要靠 CUDA stream 级别的并行：shared expert 在 `alt_stream` 上执行，与主流上的 gate → topk → dispatch 通信并行（源码 `models/deepseek_v2.py` → `DeepseekV2MoE.forward_deepep()` 中 `with torch.cuda.stream(self.alt_stream): shared_output = self._forward_shared_experts(...)`）。
- **Low-latency mode**：`dispatch_a()` 内部直接启动异步 RDMA 通信，`dispatch_b()` 只等待完成。a/b 之间的 hook 与 RDMA 通信真正并行。

这种 a/b 分离还服务于：

- **TBO (Two-Batch Overlap)**：将 decode batch 拆为两半交替执行，通信和计算重叠
- **CUDA Graph**：a/b 分界作为 graph capture 的边界

### 5.3 Normal mode dispatch

代码：`_DeepEPDispatcherImplNormal.dispatch_a()` → `dispatch_b()` → `_dispatch_core()`。以 num_tokens=1024, top-8, EP=4, 64 experts/GPU 为例。

**dispatch_a()** — 准备阶段（源码 `_DeepEPDispatcherImplNormal.dispatch_a()`）：

提取 topk 路由结果，并对 hidden_states 做 FP8 量化以减少后续通信数据量。

```python
# 以 num_tokens=1024, top-8, EP=4, 64 experts/GPU 为例
def dispatch_a(self, hidden_states, topk_output):
    # hidden_states: (1024, 7168) bf16
    # topk_output: topk_ids (1024, 8) 和 topk_weights (1024, 8)
    topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
    topk_ids = topk_ids.to(torch.int64)
    if ENABLE_JIT_DEEPGEMM and not SGLANG_DEEPEP_BF16_DISPATCH:
        hidden_states = sglang_per_token_group_quant_fp8(hidden_states, 128, ...)  # FP8 量化减少通信量
    previous_event = Buffer.capture() if self.async_finish else None
    return hidden_states, topk_ids, topk_weights, previous_event
```

**dispatch_b()** → `_dispatch_core()` — 通信阶段（源码 `_DeepEPDispatcherImplNormal.dispatch_b()` / `_dispatch_core()`）：

先调用 `get_dispatch_layout` 计算通信布局，再通过 `buffer.dispatch` 执行 all-to-all 通信，将 token 发送到对应 expert 所在的 GPU，最后封装为 `DeepEPNormalDispatchOutput` 返回。

```python
def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
    (hidden_states, topk_ids, topk_weights,
     num_recv_tokens_per_expert, event) = self._dispatch_core(...)
    event.current_stream_wait() if self.async_finish else ()
    return DeepEPNormalDispatchOutput(
        hidden_states, hidden_states_scale,
        topk_ids, topk_weights, num_recv_tokens_per_expert)

def _dispatch_core(self, x, topk_ids, topk_weights, previous_event):
    buffer = self._get_buffer()

    # get_dispatch_layout: 分析通信布局（见下文详解）
    (num_tokens_per_rank, num_tokens_per_rdma_rank,
     num_tokens_per_expert, is_token_in_rank, previous_event
    ) = buffer.get_dispatch_layout(topk_ids, self.num_experts, ...)

    # buffer.dispatch: 执行 all-to-all 通信（见下文详解）
    (recv_x, recv_topk_ids, recv_topk_weights,
     num_recv_tokens_per_expert, self.handle, event
    ) = buffer.dispatch(x, topk_idx=topk_ids, topk_weights=topk_weights,
                        num_tokens_per_rank=num_tokens_per_rank,
                        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                        is_token_in_rank=is_token_in_rank,
                        num_tokens_per_expert=num_tokens_per_expert,
                        expert_alignment=128, ...)  # DeepGEMM 要求 128 对齐

    return recv_x, recv_topk_ids, recv_topk_weights, num_recv_tokens_per_expert, event
```

### 5.3.1 get_dispatch_layout 详解

`get_dispatch_layout` 是 DeepEP `Buffer` 类的方法，在 GPU 上根据 `topk_ids` 统计通信布局，即每对 GPU 之间需要交换多少 token、每个 expert 需要接收多少 token。这些元数据是后续 `buffer.dispatch()` 的前置依赖。

**调用链**：`deep_ep/buffer.py` → `Buffer.get_dispatch_layout()` → `self.runtime.get_dispatch_layout()` → C++ binding `deep_ep.cpp` → `layout::get_dispatch_layout()` → CUDA kernel `csrc/kernels/layout.cu`

**CUDA kernel 实现**（`DeepEP/csrc/kernels/layout.cu`）：

kernel 使用 256 threads/block，通过 `blockIdx.x`（即 SM 编号）分区：前 `ceil(num_experts / 4)` 个 SM（如 256 experts → SM 0-63）负责统计 per-expert token 数，剩余 SM 统计 per-rank token 数和 `is_token_in_rank`。核心逻辑如下：

```cpp
// DeepEP/csrc/kernels/layout.cu
template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(
    const topk_idx_t* topk_idx,        // (num_tokens, num_topk)
    int* num_tokens_per_rank,          // (num_ranks,) output
    int* num_tokens_per_rdma_rank,     // (num_rdma_ranks,) output
    int* num_tokens_per_expert,        // (num_experts,) output
    bool* is_token_in_rank,            // (num_tokens, num_ranks) output
    int num_tokens, int num_topk,
    int num_ranks, int num_experts)
{
// SM 分区: 前 ceil(num_experts/kNumExpertsPerSM) 个 SM 处理 per-expert 统计,
//         剩余 SM 处理 per-rank 统计 (通过 blockIdx.x 天然分区)
// 以 256 experts, kNumExpertsPerSM=4 为例: SM 0-63 处理 expert, SM 64+ 处理 rank

// SM 0 ~ ceil(num_experts/4)-1: 统计 per-expert token 数 
__shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
int expert_begin_idx = sm_id * kNumExpertsPerSM;
int expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);

// 每个线程遍历自己负责的 token, 统计落在本 SM 负责的 expert 范围内的计数
for (int i = thread_id; i < num_tokens; i += kNumThreads) {
    auto shifted_topk_idx = topk_idx + i * num_topk;
    for (int j = 0; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin_idx <= expert_idx && expert_idx < expert_end_idx)
            ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
    }
}
__syncthreads();
// 线程间 reduce: 每个线程累加所有其他线程的计数
if (expert_begin_idx + thread_id < expert_end_idx) {
    int sum = 0;
    for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_expert_per_thread[i][thread_id];
    num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
}

// SM ceil(num_experts/4) 之后: 统计 per-rank token 数 + is_token_in_rank 
// (expert_begin_idx >= num_experts, 不满足上面的 if 条件, 执行到这里)
const auto num_expert_per_rank = num_experts / num_ranks;
for (int i = thread_id; i < num_tokens; i += kNumThreads) {
    auto shifted_topk_idx = topk_idx + i * num_topk;
    int is_in_rank[kNumRanksPerSM] = {0};
    for (int j = 0; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        // expert_id → rank: rank = expert_id / num_expert_per_rank
        rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
        is_in_rank[rank_idx]++;
    }
    // 写 is_token_in_rank 并累加 per-rank 计数
    for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
        is_token_in_rank[i * num_ranks + j + rank_begin_idx] = (is_in_rank[j] > 0);
        num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
    }
}
```

以 1024 tokens, top-8, EP=4 为例，假设**负载均衡**（路由结果在 4 卡间均匀分布）：

```
topk_ids: (1024, 8)  # 每 token 选 8 个 expert, 共 1024×8 = 8192 个 token-expert pair

get_dispatch_layout 返回 5 个值:
  num_tokens_per_rank:       [1024, 1024, 1024, 1024]   # 当前卡向各 rank 发送的去重 token 数
  num_tokens_per_rdma_rank:  None                       # EP=4 单机内, 无跨机 RDMA; 多机时为 [num_rdma_ranks] int
  num_tokens_per_expert:     [32, 32, 32, ...]          # 理想均衡下每 expert = 8192/256 = 32 token
  is_token_in_rank:          (1024, 4) bool             # 标记每个 token 是否需要发到对应 rank
  event:                     EventOverlap               # async_finish=True 时的同步事件
```

各字段说明：

- `**num_tokens_per_rank**` `[num_ranks]`：当前卡需要发送到各目标 rank 的**去重 token 数**。和 `is_token_in_rank` 一样做了去重，即同一 token 即使有多个 expert 在同一 rank，也只计 1 次。统计包含自身 rank ，即 rank 0 发给 rank 0 的 local dispatch 也走同一条路径（内存拷贝而非 NVLink）。均衡时 top-8 分 4 卡，每个 token 平均有 2 个 expert 在每张卡上，所以每个 token 几乎都需要发到每张卡，`num_tokens_per_rank ≈ [1024, 1024, 1024, 1024]`。通信时同一份 hidden_states 只发一次，接收端根据 `topk_ids` 分给各 expert。
- `**num_tokens_per_rdma_rank`** `[num_rdma_ranks]` 或 `None`：跨机场景下，需要通过 RDMA 发送到各远端机器的 pair 数。`num_rdma_ranks = max(1, num_ranks / NUM_MAX_NVL_PEERS)`，即机器数（`NUM_MAX_NVL_PEERS` 通常为 8，一个 NVLink domain 内的 GPU 数）。单机内通信时 `num_rdma_ranks == 1`，走 intranode 路径，此值为 `None`。多机时（如 2 机 × 8 GPU = EP=16，`num_rdma_ranks=2`），DeepEP 采用两级路由（RDMA 到远端机器 → NVLink 转发到目标 GPU），这个值告诉 RDMA sender 每台远端机器需要发送多少数据。
- `**num_tokens_per_expert`** `[num_experts]`：**本卡**的 token 中路由到每个全局 expert 的 pair 数（不去重）。长度 256（全局 expert 总数）。注意这只是发送端的单卡统计，不是接收端某个 expert 实际收到的总数。
- `**is_token_in_rank`** `[num_tokens, num_ranks]` bool：shape 为 `(1024, 4)`，其中 4 = EP_SIZE。标记每个 token 是否需要发到对应 rank。这是一个去重视图，即同一 token 即使有多个 expert 落在同一 rank，也只标记一次 `True`。`buffer.dispatch` 用它决定是否把 token 的 hidden_states 数据拷贝到目标 rank 的发送 buffer 中（同一份数据只发一次，接收端根据 `topk_ids` 再分给各 expert）。
- `**event`**：`async_finish=True` 时返回的 CUDA event，用于流间同步。

上面示例中"每 expert = 32 token"是理想均衡假设。实际 routing 结果受输入 token 的语义分布影响，即使 DeepSeek V3 使用了 auxiliary-loss-free 的负载均衡策略（通过动态 bias term 调节），每个 expert 收到的 token 数仍会围绕均值波动，部分 hot expert 可能收到显著多于均值的 token。不均衡时输出类似：

```
num_tokens_per_rank:      [980, 1024, 1010, 1000]   # 各卡去重 token 数略有差异
num_tokens_per_expert:    [25, 42, 18, 38, ...]     # 每 expert 的 token 数各不相同
```

这种不均衡直接影响后续 expert 计算的效率：收到 token 多的卡计算量更大，成为瓶颈；收到少的卡则浪费算力在等待上。

### 5.3.2 buffer.dispatch 详解

`buffer.dispatch()` 是 DeepEP 的 all-to-all 通信入口。它接收 `get_dispatch_layout` 的输出作为通信计划，执行实际的 GPU 间数据交换。

**调用链**：`deep_ep/buffer.py` → `Buffer.dispatch()` → 根据拓扑分支 → `Buffer.intranode_dispatch()` 或 `Buffer.internode_dispatch()` → C++ runtime → CUDA kernel

```python
# DeepEP Buffer.dispatch (deep_ep/buffer.py, 简化)
def dispatch(self, x, topk_idx=None, topk_weights=None,
             num_tokens_per_rank=None, num_tokens_per_rdma_rank=None,
             is_token_in_rank=None, num_tokens_per_expert=None,
             expert_alignment=1, config=None, ...):

    if self.runtime.get_num_rdma_ranks() > 1:
        # 跨机: RDMA + NVLink 混合通信
        return self.internode_dispatch(x, ...)
    else:
        # 机内: 纯 NVLink P2P 通信
        return self.intranode_dispatch(x, ...)
```

**机内 dispatch**（`DeepEP/csrc/kernels/intranode.cu`）

所有 GPU 通过 NVLink 互联。C++ 运行时 (`deep_ep.cpp`) 依次调用两个 kernel：

1. `**intranode::notify_dispatch`** — 各 rank 通过 NVLink 交换 metadata（`num_tokens_per_rank`、`num_tokens_per_expert`、prefix sum），填充 `rank_prefix_matrix` 和 `channel_prefix_matrix`。同时更新 CPU 侧的 `moe_recv_counter` 和 `moe_recv_expert_counter`（通过 mapped memory），CPU 轮询这些计数器确认总接收 token 数后分配 `recv_x` 等接收 buffer。
2. `**intranode::dispatch`** — 根据 `channel_prefix_matrix` 的通信计划，通过 NVLink P2P 执行实际数据传输。每个 SM pair（偶数发送、奇数接收）对应一个 channel，使用环形 buffer（head/tail 游标 + acquire/release 语义）实现流控。

```cpp
// DeepEP/csrc/kernels/intranode.cu (简化, 保留核心流程)
template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch(
    int4* recv_x, float* recv_x_scales,
    int* recv_src_idx, topk_idx_t* recv_topk_idx, float* recv_topk_weights,
    int* recv_channel_offset, int* send_head,
    const int4* x, const float* x_scales,
    const topk_idx_t* topk_idx, const float* topk_weights,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    int num_tokens, int hidden_int4, int num_topk, int num_experts,
    void** buffer_ptrs, int rank, int num_max_send_tokens, int num_recv_buffer_tokens)
{
    // SM 奇偶分工: 偶数 SM 发送, 奇数 SM 接收
    const bool is_sender = (blockIdx.x % 2 == 0);
    const auto responsible_channel = blockIdx.x / 2;
    // 每个线程组负责一个 rank 的数据
    const auto responsible_rank = thread_id / (kNumThreads / kNumRanks);

    // channel buffer 布局 (存储在接收端):
    //   channel_start/end_offset: sender 写入 token 范围的编码 (负数编码避免与 0 混淆)
    //   channel_head/tail_idx: 环形 buffer 的读写游标, head 由 receiver 推进, tail 由 sender 推进
    //   channel_x_buffers: 存放 hidden_states 数据
    //   channel_src_idx_buffers / topk_idx_buffers / topk_weights_buffers / x_scales_buffers

    if (is_sender) {
        // 1. 写入本 channel 负责的 token 范围 (通过 channel_prefix_matrix 划分)
        // 2. 遍历 token, 检查 is_token_in_rank[token_idx * kNumRanks + responsible_rank]
        //    跳过不需要发到 responsible_rank 的 token
        // 3. 获取环形 buffer 空槽 (tail 递增), 检查 head 确保不溢出
        // 4. 通过 NVLink P2P 写入对端 buffer:
        //    - hidden_states: UNROLLED_WARP_COPY 按 int4 粒度写入 channel_x_buffers
        //    - topk_idx: 转换为 local expert index (减去 responsible_rank * num_experts_per_rank)
        //      不属于该 rank 的 expert 标为 -1
        //    - topk_weights: 对应 expert 不在该 rank 的置 0
        //    - x_scales: FP8 时的 per-group scale
        // 5. 所有 warp 同步后, 更新 tail_idx (release 语义, 对 receiver 可见)
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // 等待环形 buffer 有空间 (head/tail 差值 < buffer 容量)
            // ...
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++; continue;
                }
                int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;
                // NVLink P2P 写入 x, src_idx, topk_idx, topk_weights, x_scales
                // SM90 可用 TMA (Tensor Memory Accelerator) 加速大块传输
            }
            // 更新 tail_idx, release 语义
            st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
        }
    } else {
        // 1. 轮询 channel_start/end_offset 获取要接收的 token 数
        // 2. 轮询 channel_tail_idx (acquire 语义), 检测新 token 到达
        // 3. 从 channel buffer 拷贝到 recv_x/recv_topk_idx/recv_topk_weights
        //    SM90 使用 TMA load → shared memory → TMA store 提升带宽
        //    非 SM90 回退到 UNROLLED_WARP_COPY (ld_nc_global + st_na_global)
        // 4. 推进 head_idx, 释放环形 buffer 空间
        while (num_tokens_to_recv > 0) {
            cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());
            // 拷贝 cached_channel_tail_idx - cached_channel_head_idx 个 token
            // 推进 head_idx
        }
    }
}
```

**跨机 dispatch**（`DeepEP/csrc/kernels/internode.cu`）

GPU 在同机内通过 NVLink 互联，跨机的同编号 GPU 之间通过 RDMA 互联。C++ 运行时同样依次调用两个 kernel：

1. `**internode::notify_dispatch`** — 通过 NVSHMEM 在所有 rank 间同步 `num_tokens_per_rank`、`num_tokens_per_rdma_rank` 等元数据，填充 `rdma_channel_prefix_matrix` 和 `gbl_channel_prefix_matrix`，CPU 侧轮询计数器后分配接收 buffer。
2. `internode::dispatch` — 执行实际跨机数据传输，dispatch kernel 内部 warp 按 5 种角色分工，详细代码为

```cpp
// DeepEP/csrc/kernels/internode.cu (简化, 保留核心流程)
template <bool kLowLatencyMode, int kNumRDMARanks, bool kCachedMode,
          int kNumTMABytesPerWarp, int kNumDispatchRDMASenderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32, 1)
    dispatch(int4* recv_x, float* recv_x_scales,
             topk_idx_t* recv_topk_idx, float* recv_topk_weights,
             SourceMeta* recv_src_meta,
             const int4* x, const float* x_scales,
             const topk_idx_t* topk_idx, const float* topk_weights,
             int* send_rdma_head, int* send_nvl_head,
             const int* rdma_channel_prefix_matrix,
             const int* gbl_channel_prefix_matrix,
             const bool* is_token_in_rank,
             int num_tokens, int hidden_int4, int num_topk, int num_experts,
             void* rdma_buffer_ptr, void** buffer_ptrs,
             int rank, int num_ranks)
{
    // 5 种 warp 角色
    enum class WarpRole {
        kRDMASender,              // RDMA 发送: 遍历 token, 写入 RDMA symmetric buffer
        kRDMASenderCoordinator,   // 协调: 监控 sender 进度, 批量发起 IBGDA RDMA PUT
        kRDMAAndNVLForwarder,     // 转发: 从 RDMA buffer 取数据, NVLink 转发到机内目标 GPU
        kForwarderCoordinator,    // 协调 forwarder 的转发进度
        kNVLReceivers             // 接收: 从 NVLink buffer 拷贝到 recv_x
    };

    // SM 奇偶分工: 偶数 SM 做 forwarder, 奇数 SM 做 sender/receiver
    const bool is_forwarder = (sm_id % 2 == 0);

    // warp 角色分配 (奇数 SM 为例):
    //   warp 0 ~ kNumDispatchRDMASenderWarps-1: RDMA sender
    //   warp kNumDispatchRDMASenderWarps: coordinator
    //   warp kNumDispatchRDMASenderWarps+1 ~ +8: NVLink receiver (每个负责一个 NVLink peer)

    if (warp_role == WarpRole::kRDMASender) {
        // 1. 遍历 channel 负责的 token, 检查 is_token_in_rank 确定目标 RDMA rank
        // 2. 将 x, x_scales, topk_idx, topk_weights, src_meta 打包写入
        //    rdma_channel_data 的 symmetric send buffer (对端 GPU 的对称地址)
        // 3. 等待环形 buffer 有空间 (rdma_channel_head vs tail)
        // 4. 通过 shared memory lock + window bitmap 跟踪发送进度
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            // 读取 is_token_in_rank, 判断需要发到哪些 RDMA rank
            uint64_t is_token_in_rank_uint64 = __ldg(...);
            // 将数据写入各 dst_rdma_rank 的 send buffer slot
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ...);
            // 更新 window bitmap, release lock
        }

    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        // 监控 rdma_send_channel_tail 进度, 按 chunk 批量发起 IBGDA RDMA PUT
        // 为缓解 incast 拥塞, 按 (channel_id + rdma_rank) 轮转目标 rank 顺序
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            for (int i = 0; i < kNumRDMARanks; ++i) {
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                // 检查 sender 已处理的 token 数, 凑够一批后:
                nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg, ...);
                // 通过 atomic add 更新对端 rdma_channel_tail
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), ...);
            }
        }

    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        // 1. 轮询 rdma_channel_meta 等待 RDMA sender 写入的 metadata 到达
        //    (包含 NVL prefix start/end 和 RDMA channel prefix)
        // 2. 从 rdma_channel_data round-robin 各 src_rdma_rank 取 token
        //    轮询 rdma_channel_tail 检测新 token 到达
        // 3. TMA load 到 shared memory, 再 TMA store 写入 nvl_channel_x
        //    (NVLink P2P 到 dst_nvl_rank 的 buffer)
        // 4. 更新 nvl_channel_tail, 推进 rdma_channel_head 释放 RDMA buffer
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            // round-robin 选择 src_rdma_rank
            src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
            // TMA load from RDMA buffer → smem → TMA store to NVLink buffer
            tma_load_1d(tma_buffer, src_ptr, tma_mbarrier, num_bytes_per_token);
            mbarrier_wait(tma_mbarrier, tma_phase);
            tma_store_1d(tma_buffer, dst_ptr, num_bytes_per_token, false);
        }

    } else if (warp_role == WarpRole::kNVLReceivers) {
        // 1. 轮询 nvl_channel_prefix_start/end 获取要接收的 token 范围
        // 2. 轮询 nvl_channel_tail 检测 forwarder 写入的 NVLink 数据到达
        // 3. TMA load → smem → TMA store 拷贝到 recv_x 的 expert-contiguous 位置
        //    同时拷贝 topk_idx (转为 local expert index), topk_weights, src_meta
        // 4. 推进 nvl_channel_head 释放 NVLink buffer
        while (num_tokens_to_recv > 0) {
            cached_nvl_channel_tail = ld_acquire_sys_global(nvl_channel_tail.buffer());
            // TMA 拷贝 + 推进 head
        }
    }
}
```

通信完成后，`buffer.dispatch()` 返回通信结果。注意 dispatch 返回的是**接收端聚合**结果，而 `get_dispatch_layout` 返回的是**单卡发送端**统计，因此均衡场景下 dispatch 各数值是 layout 的 4 倍（EP=4，4 张卡各发等量 token 到本卡）。以**负载均衡**场景为例，GPU-0 上：

```python
# buffer.dispatch() 返回值 (均衡场景, GPU-0, EP=4)
# 每张卡发 1024 个去重 token 到本卡, 4 张卡共 4096 个去重 token
recv_x                = (4096, 7168) bf16      # 从 4 张卡接收到的去重 token, 按 expert 0-63 连续排列
                                               # FP8 dispatch 时为 tuple: (recv_x_fp8, recv_x_scales)
recv_topk_idx         = (4096, 8) int64        # 接收到的 topk_ids, 已转为 local expert index (0-63)
recv_topk_weights     = (4096, 8) float32      # 接收到的 topk_weights, 对应 -1 位置置 0
num_recv_tokens_per_expert = [128, ..., 128]   # list[int], 长度 64, 每 expert 从 4 张卡共收 32×4=128 token                                           
handle                = (rank_prefix_matrix,   # 通信 layout 信息, combine 阶段需要
                          channel_prefix_matrix,
                          recv_channel_prefix_matrix,
                          recv_src_idx,        # (4096,) int, 每个 recv token 在源卡的原始 index
                          is_token_in_rank,
                          send_head)
event                 = EventOverlap           # async_finish=True 时的同步事件
```

以**不均衡**场景为例，GPU-0 上：

```
recv_x:                      (3920, 7168) bf16
recv_topk_idx:               (3920, 8) int64
recv_topk_weights:           (3920, 8) float32
num_recv_tokens_per_expert:  [100, 168, 72, 152, ...]  # 各 expert 不等
```

`_dispatch_core` 接收到上述结果后，返回`dispatch_b` 等待通信完成并封装为 `DeepEPNormalDispatchOutput` 

```python
# dispatch_b → _dispatch_core → C++ runtime → kernel 完成后返回
return DeepEPNormalDispatchOutput(
    hidden_states=recv_x,              # (num_recv_tokens, 7168), 去重 token, 按 expert 连续排列
    hidden_states_scale=recv_x_scales, # FP8 dispatch 时的 per-group scale, 否则为 None
    topk_ids=recv_topk_ids,            # (num_recv_tokens, 8), 已转为 local expert index
    topk_weights=recv_topk_weights,    # (num_recv_tokens, 8)
    num_recv_tokens_per_expert=num_recv_tokens_per_expert  # list[int], 长度 64 (num_local_experts)
)
```

注意 `DeepEPNormalDispatchOutput` 仅用于向后续 expert 计算传递数据。combine（归约）阶段不依赖这个封装，而是依赖 `_dispatch_core` 另外保存在 `self.handle` 中的通信 layout 信息（prefix matrix、channel offset、recv 索引、send_head 等）。

### 5.4 Low-latency mode dispatch

代码：`_DeepEPDispatcherImplLowLatency.dispatch_a()` → `dispatch_b()`。

**dispatch_a()** — 准备，并且执行通信（源码 `_DeepEPDispatcherImplLowLatency.dispatch_a()`）

Low-latency mode 与 normal mode 的根本区别：不需要 `get_dispatch_layout` 预计算布局，而是直接通过 RDMA（IBGDA）将 token 写入目标 GPU 的预分配 buffer 中。

```python
# 以 batch=128, EP=4, top-8, 64 experts/GPU 为例 (decode 场景, num_max_dispatch_tokens_per_rank=128)
def dispatch_a(self, hidden_states, topk_output):
    # hidden_states: (128, 7168) bf16
    # topk_output 含 topk_ids (128, 8) + topk_weights (128, 8)
    buffer = self._get_buffer()
    topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
    topk_ids = topk_ids.to(torch.int64)
    # expected_m: 公式 = (hidden.shape[0] × group_size × top_k + num_experts) // num_experts
    # = (128 × 4 × 8 + 256) // 256 = 17
    # 含义: 4 张卡各发 128 tokens × top-8 = 4096 token-expert pairs / 256 experts ≈ 16, 向上取整 17
    # avg(masked_m) ≈ 16, 与 expected_m 吻合; expected_m 用于 GEMM kernel 配置选择
    expected_m = (
        hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
        + self.num_experts
    ) // self.num_experts
    # RDMA 直写到目标 GPU, 输出 3D tensor
    hidden_states, masked_m, event, hook = self._dispatch_core(hidden_states, topk_ids)
    return hidden_states, topk_ids, topk_weights, masked_m, expected_m, event, hook
```

`dispatch_a` 中调用 `_dispatch_core`（源码 `_DeepEPDispatcherImplLowLatency._dispatch_core()`）：

```python
def _dispatch_core(self, hidden_states, topk_ids):
    # 根据配置选择量化方式, 默认 FP8
    use_nvfp4 = (input_global_scale is not None)
    use_fp8 = (not use_nvfp4 and not SGLANG_DEEPEP_BF16_DISPATCH)

    buffer = self._get_buffer()
    # 直接调用 buffer.low_latency_dispatch, 不需要 get_dispatch_layout
    packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
        buffer.low_latency_dispatch(
            hidden_states, topk_ids,
            self.num_max_dispatch_tokens_per_rank,  # 环境变量 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, 默认 128, 上限 1024
            self.num_experts,                       # 256
            use_fp8=use_fp8,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        )
    )
    # 返回: packed_recv_hidden 是 3D tensor (或 FP8 tuple),
    #       packed_recv_count 即 masked_m
    return packed_recv_hidden, self.packed_recv_count, event, hook
```

与 Normal mode 的 `_dispatch_core` 对比：不调用 `get_dispatch_layout`，不传入 `num_tokens_per_rank` 等布局参数，而是依赖预分配的固定大小 RDMA buffer 直接通信。所谓 "固定大小" 体现在 `DeepEP/csrc/config.hpp` 的 `LowLatencyLayout` 中：Buffer 创建时（服务启动阶段），RDMA recv buffer 按 `num_experts × num_max_dispatch_tokens_per_rank × num_bytes_per_msg` 一次性分配，这些参数都是配置项，运行时不随实际 token 数变化。因此无需像 normal mode 那样每次 dispatch 前动态计算 `num_tokens_per_rank`。

### 5.4.1 buffer.low_latency_dispatch 详解

`low_latency_dispatch` 是 DeepEP 的低延迟通信实现，使用 NVSHMEM IBGDA（InfiniBand GPU Direct Async）绕过 CPU，GPU 直接发起 RDMA 写请求。

**调用链**：`deep_ep/buffer.py` → `Buffer.low_latency_dispatch()` → C++ `Buffer::low_latency_dispatch` (`DeepEP/csrc/deep_ep.cpp`) → `internode_ll::dispatch()` → CUDA kernel `DeepEP/csrc/kernels/internode_ll.cu`

```python
# DeepEP Buffer.low_latency_dispatch (deep_ep/buffer.py, 简化)
def low_latency_dispatch(self, x, topk_idx,
                         num_max_dispatch_tokens_per_rank, num_experts,
                         use_fp8=True, use_nvfp4=False, x_global_scale=None,
                         async_finish=False, return_recv_hook=False, ...):
    # 输入:
    #   x: (128, 7168) bf16 # 本 rank 的 128 tokens
    #   topk_idx: (128, 8) int64 # routing 结果
    #   num_max_dispatch_tokens_per_rank: 环境变量控制, 默认 128, 硬上限 1024
    #   num_experts: 256
    #   use_fp8: True 时在发送前量化为 FP8 减少通信量
    # 输出:
    #   recv_x: 3D tensor 或 FP8 tuple (见下)
    #   recv_count: (num_local_experts,) int32
    #   handle, event, hook
```

**CUDA kernel 实现**（`DeepEP/csrc/kernels/internode_ll.cu`）：

kernel 使用 1024 threads/block，通过 `phases` 位标志控制 send/recv 两个阶段（可分别或同时执行）。每个 SM 的 warp 分为两组：前 `num_warps - 1` 个 warp 负责 FP8 量化和 RDMA 发送，最后一个 warp 负责读取 `topk_idx` 统计 per-expert 发送计数。

**Send phase 核心逻辑**：

```cpp
// DeepEP/csrc/kernels/internode_ll.cu
template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(
    void* packed_recv_x, void* packed_recv_x_scales,
    int* packed_recv_src_info, int64_t* packed_recv_layout_range,
    int* packed_recv_count, int* mask_buffer_ptr,
    void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
    const void* x, const topk_idx_t* topk_idx,
    int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
    int num_tokens, int num_max_dispatch_tokens_per_rank,
    int num_topk, int num_experts, int rank, int num_ranks,
    int phases)  // 位标志: LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE
{
// ==== Send phase ====
// 每个 SM 循环处理多个 token (token_idx = sm_id, sm_id + num_sms, ...)
for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
    // 读取 topk_idx, 确定目标 expert
    auto dst_expert_idx = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id));

    // FP8 量化 (如果 kUseFP8): 读 BF16 → 计算 per-128-channel amax → cast to FP8
    for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
        auto int4_value = __ldg(x_int4 + i);     // 读 BF16 数据
        // ... warp_reduce_max 计算 128-channel amax, 转 FP8, 写入 rdma_x staging buffer
    }

    // IBGDA RDMA 写入: 直接把数据写到目标 rank 的 recv buffer 对应位置
    if (dst_expert_idx >= 0) {
        int slot_idx = atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1);
        auto dst_rank = dst_expert_idx / num_local_experts;
        auto dst_ptr = rdma_recv_x + dst_expert_local_idx * num_ranks * max_tokens * msg_size + rank * max_tokens * msg_size + slot_idx * msg_size;
        // P2P (NVLink) 或 RDMA 写入
        if (nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank) != 0)
            UNROLLED_WARP_COPY(8, lane_id, ...);  // NVLink: warp-level 批量拷贝
        else
            nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, ...);  // RDMA: IBGDA 非阻塞写入
        // 完成后原子递增 finish counter
        atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1);
    }
}
// 最后一个 warp: 等所有发送完成 → 通过 RDMA/P2P 将 per-expert count 写到对端 rdma_recv_count
```

**Recv phase 核心逻辑**：

```cpp
// ==== Recv phase (同一 kernel 的下半段) ====

// 每个 warp group 负责一个 responsible_expert_idx
// 轮询 rdma_recv_count[local_expert, src_rank] 等待数据到达
while ((num_recv_tokens = ld_acquire_sys_global(
            rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0
       && (clock64() - start_time) <= NUM_TIMEOUT_CYCLES)
    ;

// 数据到达后: 从 rdma_recv_x 拷贝到 packed_recv_x (连续布局)
// 同时记录 packed_recv_src_info (来源 rank + token_idx, combine 时原路返回)
for (int i = 0; i < num_recv_tokens; ++i) {
    // 从 rdma buffer 读 token, 写入 packed buffer 的对应 expert slot
    // packed_recv_count[local_expert_idx] += num_recv_tokens (所有 src_rank 累加)
}
```

与 normal mode 的区别：不需要 `get_dispatch_layout` 预计算布局（没有 `num_tokens_per_rank` 等输入），RDMA buffer 是预分配的固定大小 `(num_local_experts, num_max_tokens, hidden)`，不需要 CPU 参与调度。代价是输出是固定大小的 3D tensor，有 padding 浪费。

**dispatch_b()** — 等待通信完成：

等待 RDMA 写入完成，将接收到的 3D tensor 和 `masked_m`（每个 expert 实际收到的 token 数）封装为 `DeepEPLLDispatchOutput` 返回。

其中 hidden_states 是 3D tensor，具体维度为

```
hidden_states (3D): (64, 512, 7168)
                     │    │    └── hidden_size
                     │    └── max_buf = num_max_dispatch_tokens_per_rank × group_size = 128 × 4
                     └── num_local_experts
```

`expected_m` 是 CPU 在 `dispatch_a` 阶段静态计算的值，`masked_m` 是 GPU 通信完成后的实际值（`packed_recv_count`）。`masked_m[i]` 统计 expert i 从**所有 4 张卡**接收到的 token 数，不只是本卡发的。本例中 4 张卡各发 128 tokens × top-8 = 4096 token-expert pairs / 256 experts，avg(masked_m) ≈ 16，与 expected_m=17 吻合。不均衡时个别 expert 的 `masked_m[i]` 可能超过均值，但不会超过 `max_buf`（= `num_max_dispatch_tokens_per_rank × group_size = 512`）。`expected_m` 在 DeepGEMM 中仅用于 `get_best_config` 选择 kernel tile 配置，不影响计算正确性；CuTeDSL 的 masked GEMM 不接受 `expected_m` 参数，直接以 `masked_m` 控制每个 expert 的有效计算行数。所有 expert 的 token padding 到相同长度：

```
masked_m: (64,), 例如 [14, 18, 12, 20, ...]  # avg ≈ 16 (batch=128, EP=4)

Expert 0 (masked_m=14):  [ tok_0  tok_1  ... tok_13  ░░░░  ... ░░░░ ]  ← 前 14 行有效, 后 498 行 padding
Expert 1 (masked_m=18):  [ tok_14 tok_15 ... tok_31  ░░░░  ... ░░░░ ]  ← 前 18 行有效
Expert 2 (masked_m=12):  [ tok_32 tok_33 ... tok_43  ░░░░  ... ░░░░ ]  ← 前 12 行有效
Expert 3 (masked_m=20):  [ tok_44 tok_45 ... tok_63  ░░░░  ... ░░░░ ]
...
                          ░░░░ = padding (无效数据)
```

最终返回 DeepEPLLDispatchOutput

```python
def dispatch_b(self, hidden_states, topk_ids, topk_weights,
               masked_m, expected_m, event, hook):
    hook() if self.return_recv_hook else event.current_stream_wait()
    # 输出 DeepEPLLDispatchOutput:
    #   hidden_states:       (64, 512, 7168) bf16  # 3D, max_buf = num_max_dispatch_tokens_per_rank × group_size
    #                        # 本例 num_max_dispatch_tokens_per_rank=128, group_size=4 → max_buf=512
    #   hidden_states_scale: (64, 512, 448) float8  # blockscale, FP8/NVFP4 dispatch 时
    #   topk_ids:            (128, 8) int64
    #   topk_weights:        (128, 8) float32
    #   masked_m:            (64,) int32  # 每 expert 从全部 4 卡收到的 token 数, 如 [14, 18, 12, 20, ...]
    #   expected_m:          17  # 公式计算值, 与 avg(masked_m) ≈ 16 吻合
    return DeepEPLLDispatchOutput(
        hidden_states, hidden_states_scale,
        topk_ids, topk_weights, masked_m, expected_m)
```

## 6. Expert 计算 — Low-latency Mode

> **代码定位**：`DeepEPMoE.forward_impl()` → `run_moe_core()` → `forward_flashinfer_cutedsl()` → `flashinfer_cutedsl_moe_masked()`
> 文件：`layers/moe/ep_moe/layer.py`（路径选择），`layers/moe/flashinfer_cutedsl_moe.py`（`flashinfer_cutedsl_moe_masked()`），`layers/quantization/modelopt_quant.py`（权重定义）

本章覆盖 LL mode 下 MoE expert 计算的完整流程，包括 NVFP4 量化体系、各 kernel 的源码级分析、以及 layout/SMEM/TMEM 等硬件层面的实现细节。

### 6.1 NVFP4 量化体系

理解 expert 计算之前，需要先了解 NVFP4 的量化机制，因为每一步的输入输出 dtype、scale 参数都和这个体系有关。

#### 6.1.1 两级缩放

NVFP4 使用 block scale + global scale 两级缩放：

```
实际浮点值 = FP4_raw × block_scale × alpha

FP4_raw:     4-bit 浮点值 (E2M1), 两个 packed 在一个 uint8 中
             可表示范围: {0, 0.5, 1, 1.5, 2, 3, 4, 6} 及其负值
block_scale: 每 16 个元素共享一个 float8_e4m3 值
             存储为 swizzled 格式（硬件访问优化）
alpha:       每个 expert 一个 float32 值
             alpha = weight_global_scale / input_global_scale
```

#### 6.1.2 权重 tensor 布局

以 EP=4（每卡 64 expert）为例，每卡存储的权重：

| 张量 | shape | dtype | 说明 |
| --- | ----- | ----- | ---- |
| `w13_weight` (w1) | `(64, 4096, 3584)` | uint8 | gate_up, 7168/2=3584 |
| `w13_blockscale_swizzled` | `(64, 4096, 448)` | float8_e4m3 | 7168/16=448, swizzled |
| `g1_alphas` (w1_alpha) | `(64,)` | float32 | Gemm1 的 per-expert scale |
| `w2_weight` (w2) | `(64, 7168, 1024)` | uint8 | down, 2048/2=1024 |
| `w2_blockscale_swizzled` | `(64, 7168, 128)` | float8_e4m3 | 2048/16=128, swizzled |
| `g2_alphas` (w2_alpha) | `(64,)` | float32 | Gemm2 的 per-expert scale |
| `w13_input_scale_quant` | `(64,)` | float32 | 输入 global scale |
| `w2_input_scale_quant` | `(64,)` | float32 | 中间激活 global scale |
| `w2_weight_scale_2` | `(64,)` | float32 | w2 的 weight global scale |

权重由 `ModelOptNvFp4FusedMoEMethod.create_weights()`（`layers/quantization/modelopt_quant.py`）创建。

#### 6.1.3 Dynamic vs Static activation scale

Gemm2 输入的量化 scale 有两种计算方式：

**Static scale**（开源 SGLang 默认路径）使用离线标定的固定值 `a2_global_scale`（即 `w2_input_scale_quant`），每个 expert 一个固定值。Gemm2 的 alpha 固定使用 `w2_alpha`（即 `g2_alphas`）。

**Dynamic scale**（lingjun 分支特有，`SGLANG_NVFP4_DYNAMIC_ACT_SCALE=1`）在运行时对 Gemm1 输出的有效行计算 absmax，推导实际 scale。公式：

```
absmax = masked_absmax(gateup_output, masked_m)  → (L,), float32
a2_scale = (448 × 6) / absmax                    # FP8_MAX × FP4_MAX / absmax
alpha2 = w2_weight_scale_2 / a2_scale
```

Dynamic scale 能自适应运行时 activation 的数值范围，减少 FP4 量化精度损失，但引入额外 kernel launch 开销（~18 us，详见 6.7 节）。

### 6.2 入口与 dispatch_output 解包

调用链：

```
DeepEPMoE.forward_flashinfer_cutedsl(dispatch_output) # layers/moe/ep_moe/layer.py
  │
  ├─ hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
  │
  └─ quant_method.apply_without_routing_weights() # layers/quantization/modelopt_quant.py
       │
       └─ flashinfer_cutedsl_moe_masked(    # layers/moe/flashinfer_cutedsl_moe.py
              hidden_states = (data, scale),       # 如果 NVFP4 dispatch
              input_global_scale = per-expert scale,
              w1, w1_blockscale, w1_alpha,
              w2, a2_global_scale, w2_blockscale, w2_alpha,
              masked_m,
              down_sm_count, down_signals, down_start_event  # 可选 overlap 参数
          )
```

Low-latency mode 下，`dispatch_output` 是 `DeepEPLLDispatchOutput`（见 5.4）：

```python
def forward_flashinfer_cutedsl(self, dispatch_output):
    hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
    # hidden_states:       (64, 512, 7168) bf16  # max_buf = 128 × 4 = 512
    # hidden_states_scale: (64, 512, 448) 或 None (BF16 dispatch 时)
    # masked_m:            (64,) int32  # 每 expert 从全部 4 卡收到的 token 数, avg ≈ 16
    output = self.quant_method.apply_without_routing_weights(
        layer=self, x=(hidden_states, hidden_states_scale), masked_m=masked_m, ...)
    return output
```

`masked_m` 是贯穿整个计算流程的参数，控制每一步只处理有效 token。`topk_ids` 和 `topk_weights` 在 LL mode 中由 dispatcher 在 combine 阶段使用，此处丢弃。

### 6.3 执行步骤总览与数据流

源码：`layers/moe/flashinfer_cutedsl_moe.py` → `flashinfer_cutedsl_moe_masked()`。

以 decode batch=128, EP=4（无 DP attention）为例（trace 实测），完整的 tensor shape 变换和各 kernel 耗时：

```
hidden_states: (128, 7168) bf16
    │
    ├─ gate → topk → topk_ids: (128, 8)
    │
    ▼ DeepEP low_latency_dispatch (RDMA)                        ~13 us
recv_hidden: (64, 512, 7168) bf16 (3D, padded, max_buf=512)
masked_m: (64,) int32, avg ≈ 16
    │
    ├─ Step 1: FP4 输入量化 (仅 BF16 dispatch 时)               ~6 us      → 6.4 节
    │    scaled_fp4_grouped_quantize(hidden, masked_m, input_global_scale)
    │    → a_q: (64, 512, 3584) uint8, a_q_sf: (64, 512, 448) e4m3
    │    permute → (512, 3584, 64)                              (zero-copy, 见 6.10 节)
    │
    ├─ Step 2: Gemm1 — gate_up projection (CuTeDSL)            ~165 us    → 6.5 节
    │    grouped_gemm_nt_masked: (512,3584,64) FP4 × (4096,3584,64) FP4
    │    → gateup: (512, 4096, 64) bf16
    │
    ├─ Step 3: Dynamic scale (仅 lingjun 分支)                  ~18 us     → 6.6 节
    │    masked_absmax(gateup, masked_m) → per-expert absmax
    │    → a2_scale, alpha2
    │
    ├─ Step 4: SiLU + FP4 量化                                  ~6 us      → 6.7 节
    │    silu_and_mul_scaled_nvfp4_experts_quantize(gateup, masked_m, a2_scale)
    │    → diq: (512, 1024, 64) uint8, diq_sf
    │
    ├─ Step 5: Gemm2 — down projection (CuTeDSL)               ~89 us     → 6.8 节
    │    grouped_gemm_nt_masked: (512,1024,64) FP4 × (7168,1024,64) FP4
    │    → out: (512, 7168, 64) bf16
    │
    ├─ permute back: (512, 7168, 64) → (64, 512, 7168)
    │
    ▼ DeepEP low_latency_combine (RDMA + topk_weights)          ~18 us
final: (128, 7168) bf16
```

**耗时分布（static scale, 单层）**：

| 阶段 | 耗时 | 占比 |
| ---- | ---: | ---: |
| dispatch | 13 us | 4.5% |
| FP4 量化 | 6 us | 2.1% |
| Gemm1 | 165 us | 57.3% |
| SiLU+量化 | 6 us | 2.1% |
| Gemm2 | 89 us | 30.9% |
| combine | 18 us | 6.3% |
| **总计** | **~288 us** | |

Dynamic scale 时 Gemm1 和 SiLU+量化之间插入 ~18 us 的 absmax + scale 推导，总计 ~306 us。

### 6.4 Step 1: `scaled_fp4_grouped_quantize` — 输入量化

> **源码定位**：
>
> - SGLang 调用端：`layers/moe/flashinfer_cutedsl_moe.py` L100-104
> - FlashInfer Python API：`flashinfer/fp4_quantization.py` → `scaled_fp4_grouped_quantize()`
> - 后端实现：`flashinfer/fp4_quantization.py` → `scaled_fp4_grouped_quant_sm100()`
> - C++ kernel：`flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp` → `silu_and_mul_scaled_nvfp4_experts_quantize()` with `use_silu_and_mul=False`

此步仅在 BF16 dispatch 时执行（`hidden_states[1] is None`），NVFP4 dispatch 时（`SGLANG_MOE_NVFP4_DISPATCH=1`）输入已量化，跳过。

**调用链**：

```
flashinfer_cutedsl_moe_masked()                        # SGLang
  │  hidden_states[1] is None → 需要量化
  │
  └─ flashinfer.scaled_fp4_grouped_quantize(            # flashinfer/fp4_quantization.py L977
         a = hidden_states[0],   # (64, 512, 7168) bf16
         mask = masked_m,        # (64,) int32
         a_global_sf = input_global_scale,  # (64,) float32, per-expert
     )
       │
       └─ scaled_fp4_grouped_quant_sm100(input_tensor, input_global_scale, mask)
            │  l, m, k = 64, 512, 7168
            │  output = empty(64, 512, 3584) uint8     # k // 2, 两个 FP4 packed 一个 byte
            │  output_scales = empty(64, 512, 112) int32  # padded layout
            │
            └─ module.silu_and_mul_scaled_nvfp4_experts_quantize(
                   output, output_scales, input_tensor,
                   input_global_scale, mask,
                   False,  # use_silu_and_mul = False → 纯量化
               )
```

C++ 侧（`fp4Quantize.cpp` L190-253）通过 `use_silu_and_mul` 标志区分模式。`False` 时直接做 NVFP4 量化，底层调用 TensorRT-LLM 的 `invokeSiluAndMulNVFP4Quantization<__nv_bfloat16>()`，完成：

1. 读取 `mask[expert_id]`，只处理前 `mask[expert_id]` 行（跳过 padding）
2. 对每 16 个连续元素（`sf_vec_size=16`）计算 block-level absmax
3. 用 `input_global_scale[expert_id]` 和 block absmax 推导 block scale（E4M3 格式）
4. BF16 值除以 `(block_scale × input_global_scale)`，round 到 FP4（E2M1），两个 FP4 pack 进一个 uint8

**输出 layout 变换**（Python 侧 permute + swizzle，permute 原因见 6.10 节）：

```python
# 量化完成后的 layout 变换 (scaled_fp4_grouped_quant_sm100 L520-529)

output = output.permute(1, 2, 0)
# (l, m, k//2) = (64, 512, 3584) → 逻辑 (512, 3584, 64), 物理不变

# block scale 的 swizzle layout:
output_scales = output_scales.view(torch.float8_e4m3fn)  # (64, 512, 448) e4m3
output_scales = output_scales.view(l, padded_m//128, padded_k//4, 32, 4, 4)
output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
# → (32, 4, 4, 4, 112, 64) — Blackwell tcgen05.mma.block_scale 要求的 TMEM swizzle 格式
```

block scale swizzle 由 `cutlass.utils.blockscaled_layout` 中的 `BlockScaledBasicChunk` 定义，每 128 行为一组，每组 32×4=128 个 scale，逻辑 permute 成 `(m32, m4, rm, k4, rk, l)` 匹配 MMA 指令读取模式。

### 6.5 Step 2: `grouped_gemm_nt_masked` — Gemm1 (gate_up projection)

> **源码定位**：
>
> - SGLang 调用端：`layers/moe/flashinfer_cutedsl_moe.py` L137-148
> - FlashInfer API：`flashinfer/cute_dsl/blockscaled_gemm.py` → `grouped_gemm_nt_masked()` L2947-3048
> - GPU kernel：同文件 → `Sm100BlockScaledPersistentDenseGemmKernel.kernel()` L976-1825

这是 LL mode 计算量最大的 kernel（trace ~165 us EP=4 / ~101 us EP=8），也是优化的首要目标。

#### 6.5.1 为什么用 "Dense" kernel 而不是 "Grouped" kernel

API 名叫 `grouped_gemm_nt_masked`，底层调用的是 `Sm100BlockScaledPersistentDenseGemmKernel`。这不是 bug，而是有意为之：

- **Grouped GEMM**：每个 group 的 M/N/K 可以不同，A/B/C 通过指针数组传入，kernel 需要在切换 group 时更新 TMA descriptor。group 数多（64）但每个 M 很小时开销明显。
- **Batched Dense GEMM**：所有 batch 共享相同的 M/N/K，A/B/C 是一个大 3D tensor。TMA descriptor 只创建一次，kernel 通过 L 维索引定位不同 expert。

DeepEP 的 LL mode 满足 batched dense 的前提：dispatch 后所有 expert 数据 pad 到统一 `(max_tokens, hidden_dim)` shape，放进 `(L, M, K)` 的 3D tensor。每个 expert 的有效行数由 `masked_m[i]` 指定，通过 `MaskedScheduler` 跳过 padding tile。64 个 expert 平均 `masked_m≈16`，每个 expert 只有 1 个 M-tile，如果用 Grouped GEMM 会频繁切换 group，batched dense + mask 方案更高效。

#### 6.5.2 API 调用

```python
grouped_gemm_nt_masked(
    (a_q, a_q_sf),                                # lhs: A + SFA
    (w1.permute(1, 2, 0), w1_blockscale),         # rhs: B + SFB
    gateup_output,                                # out: C
    masked_m,                                     # (64,) int32
    ab_dtype="float4_e2m1fn",                     # FP4 (E2M1)
    sf_dtype="float8_e4m3fn",                     # block scale 类型
    c_dtype="bfloat16",                           # 输出类型
    sf_vec_size=16,                               # 每 16 元素一个 block scale (NVF4)
    alpha=w1_alpha.view(1, 1, num_experts),       # (1, 1, 64), per-expert scale
    alpha_dtype="float32",
)
```

参数推导：`a_torch.shape = (512, 3584, 64)` → m=512, k=7168 (FP4 packed), l=64；`b_torch.shape = (4096, 3584, 64)` → n=4096。对 64 个 expert，每个做 `masked_m[i] × 4096 × 7168` 的 GEMM。kernel 按 `(m, n, k, l, dtypes, tile, cluster, sm_count)` 做编译缓存（`@functools.lru_cache`）。

#### 6.5.3 Kernel 架构：Warp 分工与 Persistent Loop

`Sm100BlockScaledPersistentDenseGemmKernel` 是 Blackwell (SM100) 专用的 persistent warp-specialized GEMM kernel，6 个 warp group（192 threads/CTA）：

| Warp ID | 角色 | 职责 |
| ------- | ---- | ---- |
| 5 | TMA warp | 从 GMEM 加载 A/B/SFA/SFB 到 SMEM，TMA prefetch descriptor |
| 4 | MMA warp | SFA/SFB 从 SMEM 拷贝到 TMEM，执行 `tcgen05.mma` block-scaled MMA |
| 0-3 | Epilogue warps | TMEM → Register → 类型转换 + alpha scaling → SMEM → TMA store |

三种 warp 角色各自运行独立的 persistent loop，通过 `MaskedScheduler` 协调 tile 分配：

```python
tile_sched = MaskedScheduler.create(tile_sched_params, block_idx(), grid_dim())
work_tile = tile_sched.initial_work_tile_info()
while work_tile.is_valid_tile:
    # ... 加载/计算/写出当前 tile
    tile_sched.advance_to_next_work()
    work_tile, _ = tile_sched.get_current_work()
```

`MaskedScheduler` 根据 `masked_m[batch_idx]` 计算每个 expert 需要多少个 M-tile，将这些 tile 按 `(M-tile, N-tile)` 二维编号为连续 `linear_idx`，每个 cluster 按步幅跳跃取 tile。当某个 expert 的 tile 取完后自动跳到下一个。padding 行不产生 tile。

以 Gemm1 为例：MMA tiler M=128，N-tile 数 = 4096/128 = 32。`masked_m[i]=18` → M-tile 数 = ceil(18/128) = 1，该 expert 1×32 = 32 个 tile。64 个 expert 总 tile 数约 2048 个。

#### 6.5.4 TMA Warp (Warp 5)

TMA warp 从 GMEM 加载数据到 SMEM，使用 Blackwell 的 TMA 硬件单元：

```python
# blockscaled_gemm.py L1262-1368 — 简化
if warp_idx == self.tma_warp_id:
    cpasync.prefetch_descriptor(tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb, tma_atom_c)
    while work_tile.is_valid_tile:
        for k_block in range(k_block_cnt):
            ab_pipeline.producer_acquire(ab_producer_state, ...)
            cute.copy(tma_atom_a, tAgA_slice[k], tAsA[stage], mcast_mask=...)   # A
            cute.copy(tma_atom_b, tBgB_slice[k], tBsB[stage], mcast_mask=...)   # B
            cute.copy(tma_atom_sfa, tAgSFA_slice[k], tAsSFA[stage], ...)         # SFA
            cute.copy(tma_atom_sfb, tBgSFB_slice[k], tBsSFB[stage], ...)         # SFB
            ab_producer_state.advance()
        tile_sched.advance_to_next_work()
```

K 维分 `k_block_cnt` 块。Gemm1：K=7168，`mma_tiler_k = 256`（FP4: 256bit/4bit=64 × inst_tile_k=4），`k_block_cnt = 7168/256 = 28`。TMA 支持 multicast：`cluster_shape_mn` 不为 `(1,1)` 时，A 沿 N 维 multicast，B 沿 M 维 multicast，减少 L2 流量。TMA 和 MMA 之间通过 `PipelineTmaUmma` 做 producer-consumer 同步，`num_ab_stage` 个 SMEM buffer。

#### 6.5.5 MMA Warp (Warp 4)

MMA warp 将 SFA/SFB 从 SMEM 拷贝到 TMEM（via `tcgen05.cp`），然后执行 `tcgen05.mma` block-scaled MMA：

```python
# blockscaled_gemm.py L1373-1572 — 简化
if warp_idx == self.mma_warp_id:
    while work_tile.is_valid_tile:
        acc_pipeline.producer_acquire(acc_producer_state)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in range(k_block_cnt):
            ab_pipeline.consumer_wait(ab_consumer_state, ...)
            cute.copy(tiled_copy_s2t_sfa, tCsSFA[stage], tCtSFA_s2t)  # SMEM → TMEM
            cute.copy(tiled_copy_s2t_sfb, tCsSFB[stage], tCtSFB_s2t)
            for kphase_idx in range(num_kphases):
                tiled_mma.set(tcgen05.Field.SFA, tCtSFA[kphase_idx].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tCtSFB[kphase_idx].iterator)
                cute.gemm(tiled_mma, tCtAcc, tCrA[kphase], tCrB[kphase], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_pipeline.consumer_release(ab_consumer_state)
        acc_pipeline.producer_commit(acc_producer_state)
```

`tcgen05.mma.block_scale` 指令读 SMEM 中的 A/B 和 TMEM 中的 SFA/SFB，执行 `A × SFA × B × SFB` 并累加到 TMEM 中的 Float32 accumulator。block scale 在 MMA 指令级别原生支持。MMA 和 Epilogue 之间通过 `PipelineUmmaAsync` 同步。

#### 6.5.6 Epilogue Warps (Warp 0-3)

Epilogue warps 将 TMEM 中的 Float32 accumulator 转换为输出类型并写回 GMEM：

```python
# blockscaled_gemm.py L1576-1825 — 简化
if warp_idx < self.mma_warp_id:
    while work_tile.is_valid_tile:
        acc_pipeline.consumer_wait(acc_consumer_state)
        for subtile_idx in range(subtile_cnt):
            cute.copy(tiled_copy_t2r, tTR_tAcc[(subtile)], tTR_rAcc)   # TMEM → Register
            acc_vec = acc_vec * alpha[work_tile.tile_idx[2]]             # per-expert alpha
            acc_vec = acc_vec.to(self.c_dtype)                           # Float32 → BFloat16
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(buffer)])         # Register → SMEM
            cute.copy(tma_atom_c, bSG_sC[(buffer)], bSG_gC[(subtile)])  # TMA Store: SMEM → GMEM
```

alpha scaling 在 epilogue 应用：`alpha = w1_alpha[expert_id]`，是 NVFP4 两级量化中 `weight_global_scale / input_global_scale` 的合并缩放因子。

#### 6.5.7 `dst_signals` 机制

`dst_signals` 是可选的 `(num_experts,)` int32 tensor，用于 Gemm2 和 combine 的 overlap。Epilogue warp 在写完一个 expert 的所有 tile 后，通过 `atomic_add_release_global` 递增该 expert 的 signal counter。combine 端轮询 counter，一旦某 expert 的 signal 到达就立即开始该 expert 的 combine，不需要等所有 expert 算完。

`dsm_pending_packed`（u64，每 byte 代表一个 expert 的 pending count）和 `dsm_counter` 跟踪完成状态。Gemm1 不使用（传 None），只有 Gemm2 在 SBO combine overlap 模式下使用。

#### 6.5.8 当前配置与优化空间

| 参数 | Gemm1 当前值 | 支持的选项 | 说明 |
| --- | ---------- | -------- | ---- |
| `mma_tiler_mn` | `(128, 128)` | `(128,128)`, `(256,128)`, `(128,256)`, `(256,256)` | 256 时启用 2-CTA MMA |
| `cluster_shape_mn` | `(1, 1)` | power-of-2, 总积 ≤ 16 | 影响 TMA multicast |
| `sf_vec_size` | 16 | 16 (NVF4), 32 (MXF4/MXF8) | NVF4 固定 16 |
| `num_ab_stage` | 自动 | 取决于 SMEM 容量 | 多级 buffering depth |
| `num_acc_stage` | 自动 | TMEM 容量约束 | accumulator pipeline depth |
| `sm_count` | 全部 SM | 可指定 | Gemm1 用全部, Gemm2 可限制 |

每个 expert M 很小（avg 16），只有 1 个 M-tile，MaskedScheduler 的 persistent loop 以 N-tile 为主要并行度。`(256, 128)` tile 在 M < tile_M 时浪费更多，但 `cluster_shape_mn=(1,2)` 可以让 B 做 multicast（N=4096 较大），减少 L2 流量。

### 6.6 Step 3: `masked_absmax` — Dynamic Scale 计算

> **源码定位**：`layers/moe/flashinfer_cutedsl_moe.py` L26-155（lingjun 分支）
> **调用位置**：`flashinfer_cutedsl_moe_masked()` L287-294，在 Gemm1 输出和 SiLU+FP4 量化之间

此步仅在 `SGLANG_NVFP4_DYNAMIC_ACT_SCALE=1` 时执行（lingjun 分支），开源 SGLang 跳过。

Gemm1 输出是 3D padded tensor `(L, M, 2*intermediate)`，每个 expert 只有前 `masked_m[i]` 行有效。标准 `torch.abs().max()` 不支持 per-expert masked reduction，因此需要自定义 Triton kernel。

#### 6.6.1 两阶段 Reduce

```
输入: x (L, M, N), mask_m (L,)    # L=experts, M=max_buf, N=2*intermediate
  │
  ├─ Stage 1: absmax_partial_kernel
  │    grid: (L, ceil(M/32), ceil(N/2048))
  │    每个 thread block 处理一个 (expert, M-block, N-block) tile
  │    检查 mask_m[expert] 跳过无效行 → 对有效元素求 abs + max
  │    → partial_output: (L, num_m_blocks, num_n_blocks) float32
  │
  └─ Stage 2: absmax_final_reduce_kernel
       grid: (L,)
       每个 thread block 对一个 expert 的所有 partial max 求 final max
       → output: (L,) float32
```

Stage 1 的 tile 大小 `BLOCK_SIZE_M=32, BLOCK_SIZE_N=2048`。以 EP=8 `(32, 512, 4096)` 为例：grid = `(32, 16, 2)` = 1024 thread blocks，但大部分 M-block 的 `m_start >= masked_m[i]`（avg ~19 << 512）会 early return，实际有效 tile 约 `32 × 1 × 2 = 64` 个。

Stage 2 的 `BLOCK_SIZE_REDUCE = next_power_of_2(16 × 2) = 32`，grid = `(32,)`。

Stage 1 核心逻辑（简化）：

```python
@triton.jit
def absmax_partial_kernel(x_ptr, mask_ptr, partial_out_ptr, ...,
                          BLOCK_SIZE_M: tl.constexpr = 32,
                          BLOCK_SIZE_N: tl.constexpr = 2048):
    pid_b, pid_m, pid_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m_limit = tl.load(mask_ptr + pid_b)
    if pid_m * BLOCK_SIZE_M >= m_limit:
        tl.store(partial_out_ptr + ..., 0.0); return
    mask = (offs_m[:, None] < m_limit) & (offs_n[None, :] < N)
    vals = tl.load(ptr, mask=mask, other=0.0)
    tl.store(partial_out_ptr + ..., tl.max(tl.abs(vals.to(tl.float32))))
```

#### 6.6.2 Scale 推导

```python
if envs.SGLANG_NVFP4_DYNAMIC_ACT_SCALE.get():
    a2_scale = (448 * 6) / masked_absmax(gateup_output.permute(2, 0, 1), masked_m)
    alpha2 = w2_weight_scale_2 / a2_scale
else:
    a2_scale = a2_global_scale
    alpha2 = w2_alpha
```

`448 × 6 = 2688` 是 NVFP4 两级量化中 block scale × FP4 数值的理论最大乘积。

#### 6.6.3 Trace 数据

以 `8k_2k_con1536_D_EP8` trace（L20C Blackwell, EP=8 LL Decode）5742 层统计：

**Gemm1 → Gemm2 之间的 6 个 kernel**：

| # | Kernel | median (us) | p90 (us) |
| - | ------ | ----------: | -------: |
| 1 | `absmax_partial_kernel` | 11.17 | 12.16 |
| 2 | `absmax_final_reduce_kernel` | 1.63 | 1.82 |
| 3 | `reciprocal_kernel` | 1.57 | 1.76 |
| 4 | `AUnaryFunctor` (mul) | 1.22 | 1.44 |
| 5 | `BinaryFunctor` (div) | 2.50 | 2.88 |
| 6 | `cvt_fp16_to_fp4_expert` | 9.79 | 13.09 |

Dynamic scale 总开销（kernel 1-5）：median 18.02 us, p90 19.71 us，占 MoE 计算的 9.9%。开销主要来自 6 次 kernel launch 的固定 overhead（每次 ~2-3 us），实际计算量很小。

**完整 MoE 层耗时（EP=8 LL Decode, 5742 层）**：

| 阶段 | median (us) | 占比 |
| ---- | ----------: | ---: |
| Gemm1 (CuTeDSL) | 101.06 | 55.6% |
| Gemm2 (CuTeDSL) | 53.12 | 29.2% |
| Dynamic scale (5 kernel) | 18.02 | 9.9% |
| SiLU + FP4 量化 | 9.79 | 5.4% |
| **MoE 计算总计** | **181.85** | **100%** |

#### 6.6.4 优化方向

- **融合 absmax + scale 推导**：6 个 kernel 合并为 1 个，消除 5 次 launch overhead
- **融合进 SiLU+FP4 量化**：先扫描求 absmax → 推导 scale → SiLU+量化，一次 launch。难度高，absmax 是全局 reduction，需跨 thread block 同步
- **融合进 Gemm1 epilogue**：epilogue warp 写出 C 前顺带计算 tile-level absmax，通过 atomic max 汇总到 per-expert max。需修改 FlashInfer CuTeDSL 模板
- **端到端融合**：Gemm1 + SiLU + 量化 + Gemm2 融合为一个 megakernel（Alpha-MoE 方案），absmax 成为 kernel 内部的 intermediate reduction

### 6.7 Step 4: `silu_and_mul_scaled_nvfp4_experts_quantize` — 激活 + 量化

> **源码定位**：
>
> - SGLang 调用端：`layers/moe/flashinfer_cutedsl_moe.py` L151-155
> - C++ kernel：`flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp` → `silu_and_mul_scaled_nvfp4_experts_quantize()` with `use_silu_and_mul=True`
> - CUDA kernel：`tensorrt_llm::kernels::invokeSiluAndMulNVFP4Quantization<__nv_bfloat16>()`

这是 Gemm1 和 Gemm2 之间的融合 kernel：SiLU 激活 + element-wise multiply + NVFP4 量化，一次 kernel launch 完成。

**调用链**：

```
flashinfer_cutedsl_moe_masked()
  └─ silu_and_mul_scaled_nvfp4_experts_quantize(
         a = gateup_output.permute(2, 0, 1),  # (64, 512, 4096) bf16
         mask = masked_m,
         a_global_sf = a2_scale,               # (64,) float32, static 或 dynamic
     )
       └─ silu_and_mul_scaled_nvfp4_experts_quantize_sm100(input, mask, global_scale)
            │  l=64, m=512, k_by_2=4096 → k=2048
            │  output: (64, 512, 1024) uint8, output_scales: (64, 512, 32) int32
            └─ C++ kernel(output, output_scales, input, global_scale, mask, True)
```

**C++ kernel 内部逻辑**：输入 `(l×m, 2k) = (32768, 4096)`，前 2048 列是 gate，后 2048 列是 up。计算 `SiLU(gate) ⊙ up = (gate × sigmoid(gate)) ⊙ up`，然后对 activated 的每 16 个元素做 NVFP4 量化（block absmax → block scale E4M3 → round FP4 → pack uint8），mask 控制只处理每 expert 前 `masked_m[i]` 行。输出同样经过 permute + swizzle（同 6.4 节）。

与 `kernels.py` 中 Triton 实现的对比：`_silu_and_mul_post_quant_kernel` 做 SiLU + FP8 量化（DeepGemm 路径），`_silu_and_mul_post_per_tensor_quant_kernel` 做 SiLU + per-tensor FP8（W4AFP8 路径）。CuTeDSL 路径使用 TRT-LLM C++ kernel 做 SiLU+FP4 融合量化，因为 FP4 的 pack/swizzle 逻辑在 C++ 中实现更高效。

### 6.8 Step 5: `grouped_gemm_nt_masked` — Gemm2 (down projection)

> **源码定位**：同 6.5 节，共享 `Sm100BlockScaledPersistentDenseGemmKernel` kernel，仅参数不同。

#### 6.8.1 与 Gemm1 的参数差异

```python
grouped_gemm_nt_masked(
    (diq, diq_sf),                                   # A: (512, 1024, 64) FP4 packed
    (w2.permute(1, 2, 0), w2_blockscale),            # B: (7168, 1024, 64) FP4
    out,                                              # C: (512, 7168, 64) bf16
    masked_m,
    alpha=w2_alpha.view(1, 1, num_experts),           # (1, 1, 64), dynamic 时为 alpha2
    sm_count=down_sm_count,                           # 可选, 限制 SM 数
    dst_signals=down_signals,                         # 可选, signal combine 启动
)
```

| 维度 | Gemm1 | Gemm2 |
| --- | ----- | ----- |
| M (per expert) | masked_m[i], avg ≈ 16 | 同左 |
| N | 4096 (= 2 × intermediate) | 7168 (= hidden) |
| K | 7168 (= hidden) | 2048 (= intermediate) |
| K-block 数 | 7168/256 = 28 | 2048/256 = 8 |
| trace 耗时 (EP=4) | ~165 us | ~89 us |

Gemm2 K 更小所以 K-loop 更短，耗时更少。N 更大（7168 vs 4096）但 K-loop 优势主导。

#### 6.8.2 `sm_count` — SM 数量限制

当 SBO combine overlap 启用时，`down_sm_count` 被设为比全部 SM 数小（如 120），剩余 SM 留给 combine 通信 kernel。Gemm2 的 grid 中 `max_active_clusters` 受 `sm_count` 约束。

#### 6.8.3 `dst_signals` — combine overlap

Gemm2 在 SBO combine overlap 模式下传入 `dst_signals`。epilogue 写完一个 expert 的所有输出 tile 后递增该 expert 的 signal，combine 端监听实现 expert 级别的 pipeline overlap。`down_gemm_overlap_args` 由 `ModelOptNvFp4FusedMoEMethod.apply_without_routing_weights` L1316-1333 提供。

### 6.9 附录：Layout 与 Permute

整个 expert 计算中反复出现 `permute(1,2,0)` 和 `permute(2,0,1)` 操作，这一节集中解释。

#### 6.9.1 本质

SGLang/DeepEP 约定 expert-first：`(L, M, K)`。CuTeDSL 的 `grouped_gemm_nt_masked` 要求 `(M, K, L)` — L 在最后一维。所以：

- 进 CuTeDSL 前：`permute(1,2,0)` → `(L,M,K) → (M,K,L)`
- 出 CuTeDSL 后：`permute(2,0,1)` → `(M,N,L) → (L,M,N)`

所有 permute 都是 zero-copy（只改 stride 元数据），物理内存中 K 维始终连续（stride=1）。

#### 6.9.2 为什么 CuTeDSL 要求 L 在最后一维

CUTE 的 `from_dlpack` 从 stride 推断 tensor layout。`(L=64, M=512, K=3584)` 的 stride 是 `(1835008, 3584, 1)`。CUTE 判断 batch 维的方式是**最后一个维度的 stride 最大**。不 permute 时 stride[0]=1835008 最大但在第 0 维，CUTE 会把 L 和 M 合并成一个大矩阵。`permute(1,2,0)` 后 stride 变为 `(3584, 1, 1835008)`，stride[2] 最大 → CUTE 正确识别 L 为 batch 维。

#### 6.9.3 各步骤 permute 汇总

```
Step 1 量化输出: a_q (64,512,3584) → permute(1,2,0) → (512,3584,64)     # 函数内部做
Step 2 Gemm1 输出: (64,512,4096) → permute(1,2,0) → (512,4096,64)       # SGLang 分配后手动
Step 3 Gemm1 权重: w1 (64,4096,3584) → permute(1,2,0) → (4096,3584,64)
Step 4 SiLU+量化: gateup (512,4096,64) → permute(2,0,1) → (64,512,4096) # TRT-LLM 要求 expert-first
     输出 diq 内部 permute(1,2,0) → (512,1024,64)                         # 给 Gemm2
Step 5 Gemm2: w2 (64,7168,1024) → permute(1,2,0) → (7168,1024,64)
     out (64,512,7168) → permute(1,2,0) → (512,7168,64)
Step 6: out (512,7168,64) → permute(2,0,1) → (64,512,7168)               # 恢复 expert-first 给 combine
```

Step 4 的来回 permute 是因为 CuTeDSL 和 TRT-LLM kernel 的 layout 约定不同。如果将 SiLU+量化迁移到 CuTeDSL 可以消除这次来回。

### 6.10 附录：CuTeDSL Kernel 配置与 SMEM/TMEM

#### 6.10.1 运行时配置

kernel 的 tile/pipeline/smem 配置在 `_setup_attributes()`（L570-710）动态计算。Gemm1 的 FP4 下 `mma_tiler_k = 64 × 4 = 256`（`mma_inst_bits_k=256`, FP4 width=4bit）。pipeline stage 数由 SMEM 容量和 occupancy 约束自动推导。

#### 6.10.2 Shared Memory 结构

```python
class SharedStorage:
    ab_full_mbar_ptr / ab_empty_mbar_ptr   # TMA↔MMA pipeline barriers
    acc_full_mbar_ptr / acc_empty_mbar_ptr # MMA↔Epilogue pipeline barriers
    tmem_dealloc_mbar_ptr / tmem_holding_buf
    sC: Align[c_dtype, 1024]               # Epilogue buffer, num_c_stage 份
    sA: Align[a_dtype, 1024]               # A tiles, num_ab_stage 份
    sB: Align[b_dtype, 1024]               # B tiles, num_ab_stage 份
    sSFA: Align[sf_dtype, 1024]            # SFA, num_ab_stage 份
    sSFB: Align[sf_dtype, 1024]            # SFB, num_ab_stage 份
```

总 SMEM 使用约 ~200KB+（Blackwell 单 SM 最大 228KB shared memory）。

#### 6.10.3 Tensor Memory (TMEM)

Blackwell 的 512 列 Tensor Memory 存放三类数据：Accumulator（Float32）、SFA（从 SMEM 通过 `tcgen05.cp` 拷贝）、SFB。TMEM 在 epilogue warp 0 中通过 `cute.arch.alloc_tmem(512, ...)` 分配，MMA warp 每个 K-block 将当前 stage 的 SFA/SFB 拷贝到 TMEM，`tcgen05.mma` 直接从 TMEM 读取 scale factor 和写入 accumulator。

## 7. Combine 阶段

> **代码定位**：`DeepEPMoE.forward_impl()` → `self.dispatcher.combine()`
> 文件：`layers/moe/token_dispatcher/deepep.py`（`_DeepEPDispatcherImplNormal` / `_DeepEPDispatcherImplLowLatency`）

Combine 是 dispatch 的逆操作：将计算结果发送回原始 GPU 并加权求和。代码统一入口：`layers/moe/token_dispatcher/deepep.py` → `DeepEPDispatcher.combine()`。

### 7.1 Normal mode combine

代码：`_DeepEPDispatcherImplNormal.combine_a()` → `combine_b()`。

```

combine_a():
  expert 计算结果: (4096, 7168)
  → 在 cutlass_moe_ep_fp4 内部已通过 deepep_post_reorder_triton_kernel 应用了 topk_weights
  → 直接传递给 buffer.combine()

combine_b():
  buffer.combine(output, handle, ...)
    → all-to-all 逆操作
    → 每个 token 收集来自各 expert 的结果
    → combined_hidden_states: (1024, 7168), bf16

```

routing weight 在 normal mode 中是在 scatter（Step 9）时应用的。

### 7.2 Low-latency mode combine

代码：`_DeepEPDispatcherImplLowLatency.combine_a()` → `combine_b()`。

```

combine_a():
  expert 计算结果: (64, 512, 7168), bf16
  buffer.low_latency_combine(
      x = hidden_states,         # (64, 512, 7168)
      topk_idx = topk_ids,       # (128, 8)
      topk_weights = topk_weights, # (128, 8)
      handle = self.handle,      # dispatch 时保存的元数据
  )
    → RDMA 发送结果回源 GPU
    → 同时按 topk_weights 加权求和

combine_b():
  → 等待完成
  → combined_hidden_states: (128, 7168), bf16

```

routing weight 在 low-latency mode 中是在 combine 时应用的（不在 MoE kernel 内部）。

## 8. Shared Expert 与 Overlap

> **代码定位**：`DeepseekV2MoE.forward_deepep()` → `self._forward_shared_experts(hidden_states)` — 在 dispatch 通信期间并行计算 shared expert。
> 文件：`models/deepseek_v2.py`（`_forward_shared_experts()`、`forward_deepep()` 中的 overlap 逻辑）

### 8.1 Shared Expert 结构

Shared expert 和 routed expert 结构相同（SwiGLU FFN），但**所有 token 都无条件经过**。DeepSeek-V3 有 1 个 shared expert，`intermediate_size = 2048`。

代码路径：`models/deepseek_v2.py` → `DeepseekV2MLP`，包含 `gate_up_proj`（MergedColumnParallelLinear）和 `down_proj`（RowParallelLinear）。EP 模式下 shared expert 不做 TP 切分（`tp_rank=0, tp_size=1`），每卡独立持有完整参数。

Shared expert 的 `_forward_shared_experts()` 实现（源码）：当 `num_fused_shared_experts == 0` 且 token 数 > 0 时，调用 `self.shared_experts(hidden_states)` 计算；`num_fused_shared_experts > 0` 时 shared expert 被 fuse 为第 257 个 expert 和 routed experts 一起计算，此方法返回 None。

### 8.2 与 Dispatch 的 Overlap

Shared expert 和 routed experts 之间不存在数据依赖（两者的输入都是同一个 hidden_states，输出最后相加），因此可以并行执行。SGLang 利用这一点做 overlap：在 DeepEP dispatch 通信期间同时计算 shared expert，隐藏通信延迟。

`forward_deepep()` 中有多种 overlap 策略（源码 `models/deepseek_v2.py`）：

**默认策略（alt_stream overlap）**：shared expert 在 `alt_stream` 上计算，和 dispatch/expert compute 重叠执行。

```

主 stream:      gate → topk → dispatch_a → dispatch_b → expert compute → combine
alt_stream:     ──────────── shared_expert ──────────────────────────────────────
                (alt_stream.wait_stream 同步)           (current_stream.wait_event 同步)

```

**SBO dispatch overlap**（`SboFlags.enable_dispatch_shared_one_stream_overlap()`）：通过 hook 在 dispatch_a 通信启动后、dispatch_b 等待前，在同一个 stream 上插入 shared expert 计算。

```

dispatch_a (RDMA 启动) → _deepep_dispatch_hook: shared_expert → dispatch_b (等待通信完成)

```

**SBO combine overlap**（`SboFlags.enable_combine_shared_two_stream_overlap()`）：shared expert 推迟到 Gemm2 完成后、combine 启动前执行，通过限制 SM 数量（`configure_deep_gemm_num_sms`）与 combine 通信并行。

**Shared expert fusion**（`num_fused_shared_experts > 0`）：将 shared expert 融合为第 257 个 expert，和 routed experts 一起 dispatch、计算、combine。此时 `_forward_shared_experts()` 返回 None，不需要单独的 overlap。

### 8.3 最终输出合并

```python
if shared_output is not None:
    x = shared_output
    x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
    final_hidden_states = x
else:
    final_hidden_states *= self.routed_scaling_factor
```

`routed_scaling_factor` 可以融合到 topk_weights 中（`should_fuse_routed_scaling_factor_in_topk`），此时不再额外乘。



