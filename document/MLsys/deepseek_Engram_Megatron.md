# Engram 条件记忆：原理与 Megatron-LM 接入分析

本文从 DeepSeek 官方实现出发解释 Engram 的数据流，再用 Megatron-LM 中的 `EngramModule`、`EngramGPTModel` 和 `EngramTransformerLayer` 串起一次 forward。Megatron 代码只用于帮助定位模块，不展开工程化细节。

## 1. 先用一句话理解 Engram

Engram 是一张由局部 N-gram 直接寻址的大型记忆表。它先根据输入 token 的局部组合取出一个静态向量，再让当前层的隐藏状态决定这个向量该保留多少，最后把结果加回 Transformer 残差流。

可以把 Transformer 和 Engram 分别想成：

- Transformer：现场分析上下文，擅长动态计算和组合推理。
- Engram：根据局部 token 组合查表，擅长记住固定名称、术语和常见短语。

例如模型读到：

```text
... New York Stock Exchange ...
```

在处理 `York` 时，普通 Transformer 需要通过若干层 Attention 和 MLP，逐渐把 `New` 与 `York` 组合成一个实体概念。Engram 可以直接对后缀二元组 `(New, York)` 做哈希并查出一个记忆向量。

但查表结果并不总是可靠。例如 `New York` 是稳定短语，而某个哈希槽也可能因为碰撞存入了不相关模式。所以 Engram 不会无条件采用查表结果，而是用当前隐藏状态算一个 gate：

```text
局部 token 组合 → 哈希查表 → 静态记忆
                              ↓
当前隐藏状态 ───────────────→ gate → 加回残差流
```

这也是论文所说的 conditional memory：查哪一行由 token ID 决定，是否采用以及采用多少由当前上下文决定。

## 2. Engram 不会替代普通 Embedding

普通 token Embedding 仍然需要。二者作用的位置不同：

```text
input_ids
   ↓
普通 token Embedding，只在模型入口执行
   ↓
Transformer 第 0 层
   ↓
Transformer 第 1 层 + 可选 Engram
   ↓
...
   ↓
Transformer 第 L 层 + 可选 Engram
   ↓
LM Head
```

普通 Embedding 回答的是“单个 token 一开始用什么向量表示”。Engram 回答的是“运行到某一层时，这段局部 token 组合能否唤起一条有用的静态记忆”。

官方演示也同时保留了两者：模型入口仍然是 `nn.Embedding`，只有配置在 `layer_ids` 中的 Transformer 层才实例化 `Engram`。论文也明确说明标准 input embedding 和 un-embedding/LM Head 保持不变。

在 Megatron-LM 中，普通 Embedding 的现有路径是：

- `megatron/core/models/gpt/gpt_model.py:156`：`GPTModel` 创建 `LanguageModelEmbedding`。
- `megatron/core/models/gpt/gpt_model.py:334`：`_preprocess()` 使用 `input_ids` 产生 `decoder_input`。
- `megatron/core/models/common/embeddings/language_model_embedding.py:55`：创建 `VocabParallelEmbedding`。
- `megatron/core/models/common/embeddings/language_model_embedding.py:111`：执行 word embedding lookup。
- `megatron/core/models/common/embeddings/language_model_embedding.py:118`：把 `[B, S, H]` 转为 Megatron Transformer 使用的 `[S, B, H]`。

引入 Engram 时，这条入口路径应保留。

## 3. Engram 的完整数据流

官方实现可以拆成六步：token 压缩、N-gram 构造、多头哈希、Embedding 查表、上下文门控、短卷积。

### 3.1 Tokenizer compression

原始 tokenizer 中，语义等价或近似等价的 token 可能拥有不同 ID，例如大小写、前导空格或 Unicode 形式不同。官方 `CompressedTokenizer` 会解码 token，再执行 Unicode 归一化、去重音符号、小写化和空白归一化，最后预计算：

```text
原始 token ID → canonical token ID
```

这不是重新训练一个 tokenizer，而是给 N-gram 查表使用的一层 ID 映射。普通 token Embedding 仍然使用模型原来的 token ID。

### 3.2 构造后缀 N-gram

假设压缩后的 token 序列为：

```text
[Only, Alexander, the, Great]
```

当当前位置是 `Great`，最大 N-gram 阶数为 3 时，Engram 关注：

```text
2-gram: (the, Great)
3-gram: (Alexander, the, Great)
```

它使用的是以当前位置结尾的 suffix N-gram。序列开头缺少的历史位置由 `pad_id` 补齐。

### 3.3 多头哈希

所有可能的 N-gram 数量太大，无法为每种组合分配唯一表项，所以 Engram 使用哈希表。对层号 \(l\)、N-gram 阶数 \(n\) 和哈希头 \(j\)，可以抽象成：

\[
i_{l,n,j,t}=H_{l,n,j}(x_{t-n+1},\ldots,x_t) \bmod N_{l,n,j}
\]

官方演示中的 `NgramHashMapping._get_ngram_hashes()` 使用层相关的随机奇数乘子，把多个 token ID 做乘法和 XOR 混合，再对每个哈希头对应的质数表长取模。

同一个 N-gram 使用多个哈希头的目的不是 Attention 式的多头并行，而是减轻单一哈希碰撞。各头查出的短向量最后拼接起来：

\[
e_t=\operatorname{Concat}_{n,j} E_{l,n,j}[i_{l,n,j,t}]
\]

这里有两个容易混淆的点：

- N-gram 是查表的 key，不是最终输出向量。
- 哈希表存在碰撞，Engram 依靠多头哈希和后面的上下文 gate 降低碰撞带来的噪声。

### 3.4 Key/Value 投影

查出的静态记忆 \(e_t\) 经过两个投影：

\[
k_t=W_k e_t, \qquad v_t=W_v e_t
\]

其中 \(k_t\) 用于判断记忆与当前上下文是否匹配，\(v_t\) 是真正准备注入残差流的内容。

这与 Attention 的 Query/Key/Value 有相似的命名，但不是一轮 token-to-token Attention：这里只有当前位置隐藏状态与当前位置查表结果之间的点积，最终得到一个标量 gate，没有构造 \(S\times S\) 注意力矩阵。

### 3.5 Context-aware gate

当前位置的隐藏状态 \(h_t\) 作为动态 Query。官方实现先分别归一化 Query 和 Key，再计算：

\[
s_t=\frac{\langle \operatorname{RMSNorm}(h_t),
\operatorname{RMSNorm}(k_t)\rangle}{\sqrt{d}}
\]

代码还会对分数应用 signed square root：

\[
\hat{s}_t=\operatorname{sign}(s_t)\sqrt{|s_t|}, \qquad
g_t=\sigma(\hat{s}_t)
\]

最终的门控记忆为：

\[
m_t=g_t\,v_t
\]

可以把它理解成：“查到了这条记忆，但当前句子是否真的需要它”。查表只看局部 token ID，gate 则能利用前面 Transformer 层已经汇总的全局上下文。

### 3.6 ShortConv 与残差注入

官方 Engram 对门控后的记忆序列执行 depthwise causal convolution：

\[
o_t=m_t+\operatorname{ShortConv}(m)_t
\]

ShortConv 的 kernel size 为 4，dilation 取最大 N-gram 阶数。它让相邻位置查出的记忆发生轻量交互，并补充非线性。随后 Engram 输出在 Attention 之前注入：

```text
h = h + Engram(h, input_ids)
h = h + Attention(h)
h = h + MoE(h)
```

官方 `TransformerBlock.forward()` 就是这个顺序。因此 Engram 不是 Attention 的一个分支，也不是用来选择 KV 的索引器，而是标准 Transformer block 之前额外的一条条件记忆残差支路。

## 4. 附带实现与官方 Engram 的对应关系

从附带代码的 forward 结构看，它是 Engram 原理的一种 Megatron 风格实现，而不是普通 Embedding 的替代品。

对应关系如下：

| 官方 Engram 演示 | 附带 Engram 实现 | 含义 |
|---|---|---|
| `NgramHashMapping` + `MultiHeadEmbedding` | `NGramEmbedding` | 生成层相关 N-gram 哈希并查表 |
| 未单独提供 unigram 后端 | unigram 分片 Embedding | 只按单 token ID 查每层独立表，可用于退化或对照模式 |
| `key_projs` | `key_proj` | 把查表向量投影成门控用 Key |
| `value_proj` | `value_proj` | 把查表向量投影成待注入的 Value |
| 每个分支独立 RMSNorm | `key_norm`、`query_norm` | 稳定点积 gate |
| signed sqrt + sigmoid | 同样的 gate 公式 | 让隐藏状态控制记忆强度 |
| `ShortConv` | `CanonLayer` | 对门控记忆做短程因果卷积 |
| `h + Engram(...)` | 层级 Engram 模块只返回贡献量 | 外层调用者负责把 Engram 输出加回残差流 |

### 4.1 两种 Embedding backend

附带实现可以选择 unigram 和 N-gram 两种后端。下面用统一后的 Engram 命名表示其语义，不对应原代码中的精确符号名：

```python
if config.engram_embedding_backend == "unigram":
    self.engram_embedding = LayerVocabParallelEmbedding(...)
elif config.engram_embedding_backend == "ngram":
    self.engram_embedding = NGramEmbedding(...)
```

`unigram` 模式只根据当前位置 token ID 查表，更接近“每层再做一次 token Embedding”。`ngram` 模式把局部 token 组合当作 key，才对应论文中的 Engram 主体设计。

所以更准确的说法是：

```text
EngramLayer = 每层条件记忆外壳
NGramEmbedding backend = Engram 式记忆检索
unigram backend = 更简单的单 token 对照或退化形式
```

### 4.2 单残差流 gate

非 Hyper-Connection 路径中，代码直接令：

```python
query = hidden_states
key = self.key_proj(embedding)
value = self.value_proj(embedding)
```

随后对 Query 和 Key 归一化，做逐元素乘积与 hidden 维求和，得到每个 token 一个标量 gate。这与官方公式一致。

### 4.3 多分支 Hyper-Connection gate

当 `hc_mult > 1` 时，隐藏状态被看作：

```text
[S, B, HC × D] → [S, B, HC, D]
```

Value 在不同分支间共享，Key 则一次投影出 `HC × D`，每个分支使用自己的 Key 和 Query 算 gate：

\[
m_{t,r}=g_{t,r}\,v_t
\]

其中 \(r\) 是残差分支编号。这与论文的多分支策略一致：所有分支共享一张记忆表和 Value 投影，但允许各分支形成不同的门控行为。

## 5. 在 Megatron-LM 中看一次 Engram forward

相关代码集中在：

```text
megatron/core/models/engram/
   ├─ engram_module.py
   │    CompressedTokenizer
   │    NgramHashMapping
   │    MultiHeadEmbedding
   │    ShortConv
   │    EngramModule
   │
   ├─ engram_model.py
   │    EngramTransformerLayer
   │    EngramGPTModel
   │
   └─ test_engram.py
```

不用先研究 Megatron 的所有抽象，只看下面这条调用链即可：

```text
EngramGPTModel.forward(input_ids)
   ↓
NgramHashMapping 为所有 Engram layer 生成 hash IDs
   ↓
各层 EngramModule 预先完成 Embedding lookup
   ↓
EngramTransformerLayer.forward(hidden_states)
   ↓
hidden_states += EngramModule(hidden_states)
   ↓
进入原有 self-attention 和 MLP
```

`EngramGPTModel` 负责第一阶段。它根据 `input_ids` 为配置中的 Engram 层生成 N-gram hash IDs，并让各层的 `EngramModule` 完成查表。这里对应前文的“静态记忆检索”。

`EngramTransformerLayer` 负责第二阶段。运行到指定层时，它把当前 hidden states 交给 `EngramModule`，计算 Query/Key gate、门控 Value 和 ShortConv，然后在 self-attention 之前把结果加回残差流：

```python
hidden_states = hidden_states + self.engram(hidden_states)
hidden_states = self_attention(hidden_states) + hidden_states
hidden_states = mlp(hidden_states) + hidden_states
```

因此，Megatron 代码体现的仍然是同一件事：`input_ids` 决定查哪条记忆，`hidden_states` 决定是否采用这条记忆。

## 6. 参数量、计算量和显存该怎样理解

假设第 \(l\) 个 Engram 层共有若干张表，每张表大小为 \(N_{l,n,j}\)，单头 embedding 维度为 \(d_e\)，记忆表参数量约为：

\[
P_{memory}=\sum_l\sum_n\sum_j N_{l,n,j}d_e
\]

扩大 \(N_{l,n,j}\) 会增加总参数和存储容量，但每个 token 仍然只在每个 head 查一行，所以 active lookup 数量不随表容量增长。这就是论文把 Engram 称为另一种稀疏轴的原因。

每个 token 的额外计算主要来自：

- 固定数量的哈希与 embedding lookup。
- Key/Value 线性投影。
- Query/Key 归一化和点积 gate。
- depthwise ShortConv。

因此“表参数可以很大”和“计算开销很小”可以同时成立。但大表并不是免费参数，训练时仍需要存权重、梯度和优化器状态，推理时也需要解决 host 容量与带宽。若预取无法被前层计算隐藏，延迟仍会暴露出来。

## 7. Engram、MoE 与 Attention 的关系

| 模块 | 条件依据 | 每个 token 激活什么 | 主要用途 |
|---|---|---|---|
| 普通 Embedding | 单个 token ID | 一行词表向量 | 构造模型入口表示 |
| Attention | 当前 Q/K 相似度 | 序列中的若干 token/value | 动态聚合上下文 |
| MoE | 当前 hidden state 的 router 分数 | 少量 Expert | 条件计算 |
| Engram | 局部 N-gram 的确定性哈希，外加 hidden gate | 少量记忆表行 | 条件记忆 |

Engram 与 MoE 都能扩大稀疏参数量，但方向不同。MoE 的地址在运行时由 hidden-state router 动态决定，Engram 的地址在进入模型前就能由 token IDs 确定。后者因此更适合提前预取。

Engram 也不是 N-gram 语言模型本身。传统 N-gram 模型通常直接根据局部 token 组合估计下一个 token 概率；Engram 只把 N-gram 当作查记忆的地址，取出的向量仍要经过 gate 并注入深层 Transformer。

## 8. 推荐的源码阅读顺序

如果想从代码重新走一遍 Engram，按下面顺序看即可：

```text
1. N-gram 哈希、查表、gate 与 ShortConv
   megatron/core/models/engram/engram_module.py
   ↓
2. input_ids 触发哈希与查表
   megatron/core/models/engram/engram_model.py::EngramGPTModel
   ↓
3. Attention 前的 Engram 残差注入
   megatron/core/models/engram/engram_model.py::EngramTransformerLayer
```

阅读时要一直保留这条主线：

```text
input_ids 决定查哪条静态记忆
hidden_states 决定是否相信这条记忆
Engram 输出作为额外残差进入指定 Transformer 层
```

## 9. 当前结论

Engram 的主体不是“再加一个更大的普通 Embedding”，而是“由 N-gram 确定性寻址、由当前 hidden state 动态过滤的 per-layer memory”。附带代码中的 N-gram backend、Key/Value 投影、signed-sqrt gate、Hyper-Connection 分支和 ShortConv，与官方 Engram 演示的数据流高度对应，因此可以把它理解成 Engram 原理的 Megatron 风格落地。

在 Megatron-LM 中，`EngramGPTModel` 负责根据输入 token 准备查表结果，`EngramModule` 负责检索、门控和 ShortConv，`EngramTransformerLayer` 负责在 Attention 前完成残差注入。源码只是把前面的算法流程拆成了三个类。

## 参考资料

- [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)
- [DeepSeek Engram 官方仓库](https://github.com/deepseek-ai/Engram)
- [Engram 官方演示代码](https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py)
- [Megatron-LM Engram 实现](https://github.com/NVIDIA/Megatron-LM/pull/3689)
