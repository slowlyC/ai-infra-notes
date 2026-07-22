# SGLang 的 KV Cache、MambaPool 与 Prefix Cache

在普通 Transformer 中，大家习惯把推理缓存统称为 KV Cache。到了 KDA、GDN、Mamba 这类带线性注意力或状态空间层的混合模型，同一句话就不够准确了：Full Attention 层仍然保存逐 token 的 KV，线性注意力层保存的是一个递推状态。SGLang 需要同时管理这两类形状、生命周期都不同的数据，并让它们一起参与 prefix cache。

本文基于当前 SGLang 源码，说明这些缓存分别保存什么、一次 prefix cache 命中实际复用了什么，以及 Mamba checkpoint、ping-pong buffer 和 Radix Tree 之间的关系。文中的 “Mamba cache” 是源码里的通用命名，它也用于 KDA、GDN 等线性注意力层，并不只表示原始 Mamba 模型。

本文关注 serving runtime，不讨论 KDA/GDN 的训练算法和 kernel 推导。线性注意力为什么能写成 recurrent state、GDN 的 state 如何更新，可以参考同目录下的 [GDN_Analysis.md](./GDN_Analysis.md)；这里从“状态已经定义好”开始，讨论 SGLang 如何分配、复用和淘汰它。

## 目录

- [1. 先区分三件事](#1-先区分三件事)
- [2. 为什么混合模型需要两种 pool](#2-为什么混合模型需要两种-pool)
- [3. Radix prefix cache 管理什么](#3-radix-prefix-cache-管理什么)
- [4. 混合模型如何把 KV 和 Mamba 状态放进同一棵树](#4-混合模型如何把-kv-和-mamba-状态放进同一棵树)
- [5. Mamba checkpoint 为什么不能像 KV 一样随意切分](#5-mamba-checkpoint-为什么不能像-kv-一样随意切分)
- [6. ping-pong buffer 与 checkpoint 的关系](#6-ping-pong-buffer-与-checkpoint-的关系)
- [7. 一次混合模型的多轮 cache hit](#7-一次混合模型的多轮-cache-hit)
- [8. 100K 上下文会不会存下大量 Mamba state](#8-100k-上下文会不会存下大量-mamba-state)
- [9. 状态默认保存在哪里](#9-状态默认保存在哪里)
- [10. 淘汰时发生什么](#10-淘汰时发生什么)
- [11. 源码阅读地图](#11-源码阅读地图)
- [12. 总结](#12-总结)

## 1. 先区分三件事

| 名称 | 保存的内容 | 主要作用 |
|---|---|---|
| KV Cache | Full Attention 层每个历史 token 的 K/V，MLA 模型则是压缩后的 latent KV | 让后续 token 不必重新计算历史注意力状态 |
| MambaPool | KDA、GDN、Mamba 等层的递推状态，例如 convolution state 和 temporal state | 让模型从某个序列位置继续递推 |
| Prefix Cache | token 前缀到上述内部状态的索引和所有权管理 | 在新请求中找到并复用已经计算过的公共前缀 |

Prefix Cache 并不是第三份模型状态，也不是把回答文本缓存起来。它更像一个目录：key 是 token 序列，value 指向 KV pool 和 MambaPool 中已经存在的物理槽位。

所以 cache hit 的含义是：新请求的输入 token 前缀已经有可复用的模型内部状态。这段前缀可以跳过 prefill，未命中的输入尾部仍要 prefill，新的回答仍要 decode。

### 1.1 数据缓存和索引缓存不是一回事

把 Prefix Cache 直接等同于 KV Cache，容易混淆“数据在哪里”和“怎样找到数据”。在当前实现里，两者分工如下：

```text
模型状态数据
  Full Attention KV / MLA latent KV  → TokenToKVPool
  KDA/GDN/Mamba recurrent state      → MambaPool

查找与所有权
  token 前缀 → Radix Tree 节点 → pool slot index
```

Radix Tree 的节点不会再存一份完整 KV tensor。它保存物理 slot 的索引，并通过 lock、LRU 和插入/淘汰逻辑决定这些 slot 还归谁使用。正因为 Prefix Cache 管的是索引和所有权，同一份 KV 可以由多个后续请求共享，而不必为每个请求复制一遍。

Mamba 状态稍有不同。它是可变的递推状态，新请求命中后不能直接在共享 checkpoint 上继续写，所以需要先 copy-on-write 到新的活动 slot。这里共享的是“恢复起点”，不是后续可原地修改的工作区。

### 1.2 hit 和 miss 分别省掉什么

假设请求输入长度是 `N`，其中前 `H` 个 token 命中 Prefix Cache，新生成 `M` 个 token：

```text
cache miss：prefill N 个输入 token → decode M 个输出 token
cache hit： prefill N-H 个输入 token → decode M 个输出 token
```

命中不会减少新回答的 decode token 数，只减少输入上下文的重复计算。对 100K 上下文只追加几十个 token 的场景，`H/N` 可以非常接近 100%，因此 token 口径的命中率达到 99% 完全合理。

如果 `H = 0`，输入走完整 prefill，而不是“全部走 decode”。Prefill 和 decode 是两个执行阶段，是否命中只决定 prefill 从哪里开始。

### 1.3 什么内容必须完全相同

Radix Cache 比较的是 token IDs，不是原始字符串的语义。两段文字看起来一样，但只要 chat template、special token、tool result、图片占位 token 或 tokenizer 不同，token 序列就可能在中间分叉。

此外，`RadixKey` 还带有 `extra_key`。当前注释列出的典型用途是 LoRA ID 和 cache salt。同一串 token 如果属于不同 LoRA 或不同隔离命名空间，也不会共享缓存。这个设计保证了 prefix reuse 不会跨越本来不兼容的模型状态。

## 2. 为什么混合模型需要两种 pool

### 2.1 Full Attention 的状态随 token 数增长

对标准自回归注意力，第 `t` 个 token 生成后，每层都要留下对应的 K/V。缓存长度与上下文长度一起增长：

```text
token 位置       0        1        2       ...       t
KV 物理槽位    slot_8   slot_21  slot_4    ...    slot_103
```

`ReqToTokenPool` 保存“请求内的逻辑 token 位置 → 物理 KV 槽位”的映射，`MHATokenToKVPool` 或 `MLATokenToKVPool` 才保存真正的 KV tensor。相关实现位于：

- `python/sglang/srt/mem_cache/memory_pool.py` 中的 `ReqToTokenPool`
- 同一文件中的 `MHATokenToKVPool` 和 `MLATokenToKVPool`

KV 槽位按 token 或 page 分配，因此 100K 上下文确实会占用约 100K 个 token 对应的 KV 空间。

#### 三层索引关系

SGLang 没有用一个连续 tensor 直接表示“某个请求的所有 KV”，而是把请求、逻辑 token 位置和物理存储拆开。`ReqToTokenPool` 的主体是一个二维 int32 tensor：

```python
self.req_to_token = torch.zeros(
    (max_running_requests + 1, max_context_len),
    dtype=torch.int32,
    device=device,
)
```

它可以按下面的方式理解：

```text
req_pool_idx = 7

req_to_token[7, 0] = 31
req_to_token[7, 1] = 96
req_to_token[7, 2] = 12

逻辑 token 0 的 KV 在物理 slot 31
逻辑 token 1 的 KV 在物理 slot 96
逻辑 token 2 的 KV 在物理 slot 12
```

所以 `req_pool_idx` 不是 KV slot，它只选择映射表的一行；表中每个元素才指向 TokenToKVPool。这样做允许物理 KV 非连续分配，也允许 Radix Cache 把已存在的物理索引写回新请求的映射行。

请求完成后，`ReqToTokenPool.free(req)` 释放的是请求行。KV slot 是否释放，要看它已经移交给 Radix Cache、仍被活动请求保护，还是可以交还给 token allocator。这两个生命周期不能混为一谈。

#### MHA KV 和 MLA latent KV

`MHATokenToKVPool` 为每层保存 K/V；从概念上看，一个 token、一个层的占用近似为：

$$
B_{\text{MHA, token, layer}}
\approx 2 \times H_{kv} \times D_{head} \times B_{dtype}
$$

前面的 2 来自 K 和 V。实际布局还受 TP 切分、page padding、dtype 和 backend 影响。

MLA 不保存传统的完整多头 K/V，而由 `MLATokenToKVPool` 保存压缩后的 latent KV。其主要存储维度是 `kv_cache_dim`，因此占用更接近：

$$
B_{\text{MLA}}
\approx N_{token} \times L_{MLA} \times D_{cache} \times B_{dtype}
$$

两者的内容不同，但对 Radix Cache 来说都表现为“每个 token 对应一个物理 cache index”。Prefix Cache 不需要理解 attention kernel 如何解释该 index。

#### page size 带来的对齐

当 `page_size > 1` 时，Radix key 和可提交 KV 长度会向下对齐到完整 page：

```python
aligned_len = len(key) // page_size * page_size
```

假设输入有 103 个 token、`page_size=16`，树中可直接索引的部分是前 96 个 token。尾部 7 个 token 可能仍在当前请求的映射中，但不能作为完整 Radix page 共享。源码用 `cache_protected_len` 区分“已经由树保护的长度”和“请求当前拥有的 prefix_indices 长度”，避免下一次 chunk 提交时重复释放或泄漏这段 partial page。

### 2.2 Mamba 状态是序列位置上的快照

线性注意力或状态空间层可以抽象为：

```text
state_t = F(state_{t-1}, token_t)
```

模型向前走到位置 `t` 后，只要保留 `state_t`，就能从 `t + 1` 继续递推。它不需要像 Full Attention 那样，为同一请求长期保留每个历史 token 的完整状态。

`MambaPool` 因此按固定大小的 state slot 分配，而不是按 token 分 page。一个活动请求通常占一个工作 slot，里面包含所有相关线性注意力层的 convolution state、temporal state 等 tensor。`MambaSlotAllocator` 的文件注释也明确写着：它分配的是 request-level fixed-size slot，不是 token KV index。

相关实现位于：

- `python/sglang/srt/mem_cache/memory_pool.py` 中的 `MambaPool`、`HybridReqToTokenPool`
- `python/sglang/srt/mem_cache/allocator/mamba.py` 中的 `MambaSlotAllocator`

两条访问链可以简化为：

```text
请求 req
  req_pool_idx → ReqToTokenPool[请求, token 位置] → KV 物理槽位 → TokenToKVPool
  mamba_pool_idx                                      → MambaPool state slot
```

前一条链是一串逐 token 的索引，后一条链通常只指向一个当前递推状态。

#### MambaPool 的实际 tensor 形状

`MambaPool` 从模型的 `BaseLinearStateParams` 读取 conv state 和 temporal state 的 shape。忽略 speculative decoding 的中间状态后，普通布局可简化为：

```python
conv_state[i].shape = (
    num_mamba_layers,
    num_slots + 1,
    *conv_shape_i,
)

temporal_state.shape = (
    num_mamba_layers,
    num_slots + 1,
    *temporal_state_shape,
)
```

第二维就是 slot 维。给一个请求分配 `mamba_pool_idx=17`，表示所有相关层都使用各自 tensor 中的 slot 17：

```text
conv_state[0][:, 17, ...]
conv_state[1][:, 17, ...]
temporal_state[:, 17, ...]
```

这也解释了为什么一个 Mamba slot 可能比“单层 KV”大：一个 slot 横跨所有 KDA/GDN/Mamba 层，并同时包含 temporal state 和短卷积窗口。但它的大小不随该请求已经走过多少 token 增长。

如果 temporal state 的 shape 是 `(H, D_v, D_k)`，忽略 conv 和对齐，一个全精度 slot 的主要占用可以写成：

$$
B_{\text{state slot}}
\approx L_{linear} \times H \times D_v \times D_k \times B_{dtype}
$$

这只是便于理解的估算。真实值应再加所有 conv tensor、可选 ReplaySSM ring、speculative intermediate state 和各硬件布局的 padding。

#### 为什么 KDA/GDN 也叫 Mamba cache

当前代码把多种 recurrent linear attention 的 runtime state 统一放进 `MambaPool`，这是基础设施命名，不是算法分类。`MambaPool.State` 只要求状态能表示为 conv tensor 列表和 temporal tensor，具体递推由对应 attention backend 解释。

因此看到下面这些字段时，可以把 `mamba` 读成“linear recurrent state”：

```text
req.mamba_pool_idx
req.mamba_last_track_seqlen
ComponentType.MAMBA
mamba_radix_cache_strategy
```

KDA 和 GDN 的 temporal shape、gate 语义可以不同，但它们都需要“从某个序列位置恢复固定大小状态”，所以能复用同一套 slot allocator、Radix component 和 HiCache transfer hook。

### 2.3 两类 pool 的差异总结

| 维度 | TokenToKVPool | MambaPool |
|---|---|---|
| 分配粒度 | token 或 page | fixed-size state slot |
| 单请求活动占用 | 随上下文长度增长 | 通常固定为 1 个 running slot，策略可能再加 tracking slot |
| 状态是否可按 token 切分 | 可以 | 不可以 |
| Radix 节点保存什么 | 一段 KV 物理索引 | 某个序列位置的 checkpoint slot 索引 |
| cache hit 后能否直接共享 | KV 可只读共享 | checkpoint 需 COW 到新活动 slot |
| 内存压力来源 | 长上下文 token 数 | 并发活动请求数和保留 checkpoint 数 |

## 3. Radix prefix cache 管理什么

普通模型默认使用 `RadixCache`。它是一棵压缩前缀树，逻辑 key 由 token IDs 和 `extra_key` 组成。`extra_key` 可用于隔离 LoRA、cache salt 等不同命名空间，避免只看 token 相同就误复用状态。

树节点主要保存：

- 一段压缩后的 token key；
- 这段 token 对应的物理 KV 索引；
- 父子关系、访问时间和 lock reference；
- 启用 HiCache 时的 host-side 索引。

`match_prefix()` 沿树查找最长公共 token 前缀，并受 `page_size` 对齐限制。它返回的 `device_indices` 不是 KV tensor 本身，而是 TokenToKVPool 的物理索引。

`RadixKey`、`TreeNode` 和 `RadixCache` 位于 `python/sglang/srt/mem_cache/radix_cache.py`。

### 3.1 `RadixKey`：token 序列加命名空间

`RadixKey` 的主要字段可以压缩成下面四个：

```python
class RadixKey:
    token_ids: array[int]
    extra_key: str | None
    is_bigram: bool
    limit: int | None
```

`limit` 让调用方可以只匹配原始 token 数组的前一部分，而不先复制长数组。`is_bigram` 用于 EAGLE 等特殊视图，普通请求可以先忽略。对 Prefix Cache 语义最重要的是 `token_ids` 和 `extra_key`。

`RadixKey.match()` 返回两个 key 的最长公共前缀，并把结果向下对齐到 `page_size`。它首先检查 `extra_key` 是否兼容，再比较 token。也就是说，树的逻辑 key 实际上是：

```text
(extra_key, token_0, token_1, ..., token_n)
```

cache hit 不是模糊匹配，也不是 embedding 相似度搜索。第一个不同 token 出现后，后面的内容即使重新相同，也不能继续命中，因为后续 hidden state 已经依赖不同的历史。

### 3.2 压缩 Radix Tree 如何表示公共前缀

Radix Tree 的一条边可以保存多个 token，而不是每个 token 建一个节点。假设系统先后处理三段输入：

```text
S + Q1 + A1 + Q2
S + Q1 + A1 + Q3
S + Q4
```

树可以近似表示为：

```text
root
  [S]
    [Q1, A1]
      [Q2]
      [Q3]
    [Q4]
```

这里的 `[Q1, A1]` 是一条压缩边。若后来插入的 key 在这条边中间分叉，`_split_node()` 会把它拆成新的父子节点。Full KV 的 value 可以按 split 位置切成前后两段；Mamba checkpoint 不能这样切，第 5 节会单独解释。

压缩边减少了 Python 节点数，但不改变匹配语义。树上的逻辑深度由 token 长度决定，不等于 Python 对象层数。

### 3.3 `match_prefix()` 返回的是可直接复用的物理索引

请求进入新一轮调度时，`Req.init_next_round_input()` 会重新组合完整输入，并调用：

```python
match_result = tree_cache.match_prefix(
    MatchPrefixParams(
        key=RadixKey(
            token_ids=self.full_untruncated_fill_ids,
            extra_key=self.extra_key,
            limit=key_limit,
        ),
        req=self,
        cow_mamba=cow_mamba,
    )
)
```

普通 `RadixCache` 的结果里，`device_indices` 是命中前缀所有 KV slot index 拼接成的一维 tensor。请求保存它之后，`ScheduleBatch.prepare_for_extend()` 只取未命中尾部：

```python
input_ids = [
    req.get_fill_ids()[len(req.prefix_indices):]
    for req in reqs
]
```

这两处代码把 Prefix Cache 与 prefill 计算直接连起来：`len(prefix_indices)` 决定 extend 从哪个 token 开始。

`MatchResult` 还包含多个节点锚点：

| 字段 | 含义 |
|---|---|
| `last_device_node` | GPU 上可直接复用部分的最后节点 |
| `last_host_node` | HiCache 开启后，host 上可命中的最后节点 |
| `best_match_node` | 所有组件共同接受的最深节点 |
| `host_hit_length` | Full KV 需要从 host load back 的 token 数 |
| `mamba_host_hit_length` | 需要 load back 的 Mamba slot 数，通常是 0 或 1 |
| `mamba_branching_seqlen` | token 树可继续命中、但缺少 Mamba checkpoint 时的潜在分叉位置 |

普通 GPU-only `RadixCache` 中，前三个节点通常相同。到了 UnifiedRadixCache + HiCache，它们必须分开，因为“token 路径存在”“GPU 数据存在”“host 备份存在”是三种状态。

### 3.4 第一次请求：计算并插入

假设第一轮上下文为：

```text
S + Q1 → A1
```

这里的 `S` 是 system prompt。第一次请求没有可复用前缀，SGLang 对 `S + Q1` 做 prefill，再 decode 出 `A1`。请求完成或执行 chunked prefill 的中间提交时，`cache_finished_req()` 或 `cache_unfinished_req()` 把 token key 和物理 KV 索引插入 Radix Tree。

插入时，如果树上已经有相同前缀，新请求刚计算出的重复 KV 槽位会被释放，树上原来的物理槽位继续作为共享版本。

### 3.5 finished 和 unfinished 为什么都要提交

SGLang 有两个重要提交入口：

- `cache_finished_req()` 处理请求结束后的最终提交；
- `cache_unfinished_req()` 处理 chunked prefill 或仍在运行请求的中间提交。

最终提交的普通 KV 路径可简化为：

```text
收集 token_ids 和 req_to_token 中的 KV indices
  → 按 page_size 截断 RadixKey
  → insert(key, values)
  → 释放与旧树前缀重复的新 KV slot
  → 释放不足一页的尾部
  → 解除活动请求对旧 last_node 的锁
```

`insert()` 返回 `prefix_len`，表示本次 key 中已有多少 token 已经存在于树上。新请求在计算前并不知道另一个并发请求是否刚插入了相同前缀，所以它可能已经生成一份重复 KV。插入时释放 `[cache_protected_len, prefix_len)` 对应的新 slot，保留树中的共享版本，解决了并发重复计算后的所有权合并问题。

`cache_unfinished_req()` 多一个“重新 match 并改写请求映射”的过程：

```text
insert 当前已完成部分
  → 重新 match 得到树中规范的 KV indices
  → 写回 req_to_token 当前请求行
  → 更新 prefix_indices / cache_protected_len / last_node
  → 请求下个 chunk 继续使用同一份前缀
```

这不是把每个 prefill chunk 都复制到另一块缓存。大部分情况下只是把已计算物理 slot 的所有权交给树，并让当前请求也引用树中的规范索引。

### 3.6 第二次请求：查找并复用

第二轮发送给模型的输入通常是完整消息历史：

```text
S + Q1 + A1 + Q2
```

它在 API 层是一个新请求，并不是把上一轮执行循环原地改成 decode。不过 `S + Q1 + A1` 与已缓存 token 前缀相同，`match_prefix()` 可以返回这段前缀的 KV 索引。调度器只把 `Q2` 等未命中尾部放进 extend/prefill，随后再 decode 新回答。

因此，多轮对话很容易获得高 token 命中率，但并非必然命中。状态可能已被淘汰，请求也可能路由到没有这份缓存的实例；模板、工具结果、代码内容、tokenization、LoRA 或 cache salt 变化也会让前缀在某处开始分叉。

常见的“99% cache hit”更可能表示输入 token 中约 99% 命中了前缀缓存，而不是 100 个请求中直接返回了 99 个旧答案。具体仍要以指标的定义为准。

### 3.7 lock reference 保护活动请求

树节点一旦交给 Prefix Cache，并不代表可以随时淘汰。请求命中某个 `last_node` 后，`inc_lock_ref()` 会保护它依赖的缓存；请求完成、切换到新的规范节点或被撤回时，再由 `dec_lock_ref()` 释放保护。

普通 Full KV 的锁覆盖从命中节点到根的路径，因为完整 prefix 由沿途所有边共同组成。被锁住的 token 会从 evictable 统计移动到 protected 统计。只有 lock reference 归零的缓存才可参与正常淘汰。

因此同一个节点同时存在三个概念：

```text
逻辑存在：token key 仍在树上
物理存在：value 仍指向有效 pool slot
可淘汰：  lock_ref == 0，并满足组件的淘汰条件
```

后面讨论 tombstone 和 HiCache 时，这三个维度会继续出现。

## 4. 混合模型如何把 KV 和 Mamba 状态放进同一棵树

普通 Transformer 只需要 token KV，一棵 `RadixCache` 足够。混合模型的问题在于，同一段 token 前缀要同时满足多个执行层的恢复条件：Full Attention 层需要逐 token KV，线性注意力层需要某个位置的 recurrent state，SWA 模型还可能有窗口状态。

如果每类状态各建一棵树，它们的分叉位置、锁和淘汰可能不一致。当前 SGLang 用一棵逻辑树加多个 component 解决这个问题。

### 4.1 从模型配置到缓存类型

缓存创建入口位于 `python/sglang/srt/mem_cache/kv_cache_builder.py` 和 `registry.py`。构建阶段先根据模型配置判断：

```text
is_hybrid_swa：是否混合 Full Attention 与 Sliding Window Attention
is_hybrid_ssm：是否包含 Mamba/KDA/GDN 等 recurrent linear attention
```

默认选择链可以概括为：

```text
普通 Full Attention / MLA
  → RadixCache

Hybrid SWA 或 Hybrid SSM
  → UnifiedRadixCache

Hybrid + --enable-hierarchical-cache
  → UnifiedRadixCache.init_hicache()
```

这里还有自定义 radix backend、LMCache、FlexKV 等旁路，本文只讲内置默认实现。对当前的 KDA/GDN/Mamba 混合模型，不能再沿旧的 `MambaRadixCache` 类名推断真实入口，应先看 `registry.default_radix_cache_factory()`。

当前代码对 hybrid SSM 模型默认创建 `UnifiedRadixCache`，并注册两个组件：

```text
UnifiedTreeNode
  component_data[FULL]  → TokenToKVPool 中的一段 KV 索引
  component_data[MAMBA] → MambaPool 中的一个 state slot 索引
```

如果模型还有 Sliding Window Attention，则会再加入 `SWA` 组件。选择逻辑位于 `python/sglang/srt/mem_cache/registry.py`，统一树和组件实现位于：

- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/mem_cache/unified_cache_components/full_component.py`
- `python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py`

同一逻辑前缀只建一套树节点，但不同组件有独立的 value、lock、LRU 和容量统计。Full 组件要复用匹配路径上所有 token 的 KV；Mamba 组件只需要匹配终点处最新的有效状态快照。

早期的专用实现 `MambaRadixCache` 仍很适合观察这套机制，但不应把它当作当前 hybrid SSM 的默认选择。当前入口已经统一到组件化的 `UnifiedRadixCache`。

### 4.2 `UnifiedTreeNode` 的 component data

统一树节点不再只有一个 `value`。每个组件拥有一份 `ComponentData`：

```python
@dataclasses.dataclass
class ComponentData:
    value: torch.Tensor | None = None
    lock_ref: int = 0
    metadata: dict = field(default_factory=dict)
    host_value: torch.Tensor | None = None
    host_lock_ref: int = 0
```

对混合 Full + Mamba 模型，同一节点可以是：

```text
node.key = [一段 token]

node.component_data[FULL].value
  = [41, 77, 12, ...]       # 一段 TokenToKVPool 索引

node.component_data[MAMBA].value
  = [9]                     # 一个 Mamba checkpoint slot
```

`value=None` 不一定表示节点不存在。它可能表示该组件的数据已经从 device 淘汰，但树结构仍保留；如果 `host_value` 还存在，HiCache 仍可把它 load back。这种“逻辑节点存在、某个组件为空”的状态称为 tombstone。

### 4.3 同一棵树，不同资源账本

`UnifiedRadixCache` 负责逻辑遍历，每个 `TreeComponent` 负责自己的物理资源。当前内置 component 包括：

| Component | pool | 命中后复用方式 | 资源统计 |
|---|---|---|---|
| `FULL` | TokenToKVPool | 复用整条匹配路径的 KV indices | token 数、路径锁、device/host leaf |
| `SWA` | SWA pool 或窗口映射 | 按滑动窗口边界恢复 | 独立 lock/LRU/metadata |
| `MAMBA` | MambaPool 或 checkpoint pool | 取终点 checkpoint，COW 到活动 slot | slot 数、终点锁、独立 LRU |

组件化的价值不是代码形式更整齐，而是资源单位确实不同。Full 组件释放 1000 表示释放 1000 个 token slot；Mamba 组件释放 1 表示释放一个跨多层的 state slot。统一树必须分别统计：

```python
EvictParams(
    num_tokens=full_need,
    swa_num_tokens=swa_need,
    mamba_num=mamba_slots_need,
)
```

### 4.4 最长 token 前缀不一定是有效混合前缀

`UnifiedRadixCache.match_prefix()` 在遍历节点时，会为每个 component 创建 validator。只有所有 validator 都接受某个节点，`best_match_node` 才会向前推进。

对 GPU-only 的 Full + Mamba 模型，条件近似为：

```text
FULL.value 不是 None
并且
MAMBA.value 不是 None
```

如果树的 Full KV 能匹配到 4096 token，但最深 Mamba checkpoint 只在 3840，那么直接恢复边界只能落在 3840。3840 到 4096 之间的 Full KV 即使还在，也不能单独把线性注意力层带到正确状态。

这不是浪费 Full KV。`mamba_branching_seqlen` 可以记录更深的对齐分叉位置，后续 extend 过程中重新计算这段 linear state，并在合适位置生成新 checkpoint。统一树保留了“逻辑上还能走多远”和“当前能直接恢复多远”两个信息。

### 4.5 Full 组件与 Mamba 组件的 LRU 触摸方式不同

一次 Full prefix hit 会消费根到 `last_node` 的全部 KV，所以沿途节点都与当前请求相关。Mamba 只消费 `best_match_node` 的最新 checkpoint，不会读取沿途每一个历史 checkpoint。

`MambaComponent.refresh_lru()` 因此只在 `MATCH_END` 时把真正使用的终点状态提升为 MRU，不把所有祖先状态一起刷新。如果每次多轮对话都把整条会话的所有 Mamba checkpoint 变热，旧会话会成组占据 LRU 前端，淘汰粒度反而变差。

### 4.6 为什么还保留 `MambaRadixCache`

源码中仍有专用 `mamba_radix_cache.py` 和 `hi_mamba_radix_cache.py`。它们保留了早期实现及部分专用路径，很多 checkpoint 语义写得更直接，例如：

- `_match_prefix_helper()` 只把真正带 Mamba state 的节点作为恢复点；
- `_split_node()` 明确不给拆出的父节点伪造 Mamba state；
- `_match_post_processor()` 计算 branching point 并安排 COW。

阅读时可以用它们帮助理解算法，但判断当前运行时行为时，应以 `registry.py` 选择到的 `UnifiedRadixCache + MambaComponent` 为准。

## 5. Mamba checkpoint 为什么不能像 KV 一样随意切分

KV 是逐 token 保存的。假设已经缓存 1000 个 token，那么取前 600 个 token 的 KV 索引即可得到位置 600 的注意力前缀。

对 Full Attention，可以写成：

```text
KV[0:1000] = KV[0:600] + KV[600:1000]
```

Radix 节点在 600 处分裂时，只需切两个 index slice，不需要运行模型。

Mamba state 不具备这种可切分性。位置 1000 的 `state_1000` 是前 1000 个 token 递推后的结果，无法从它直接还原 `state_600`。因此，当 Radix 节点因为前缀分叉而在中间拆分时，新的父节点可以切分 KV 索引，却不能凭空得到对应位置的 Mamba state。`MambaComponent.redistribute_on_node_split()` 会让新父节点的 Mamba value 保持为空。

### 5.1 checkpoint 是某个精确位置的函数值

将 recurrent update 写为：

$$
S_t = F(S_{t-1}, x_t)
$$

那么：

$$
S_{1000} = F(F(\cdots F(S_0, x_1), \cdots), x_{1000})
$$

`S_1000` 包含前 1000 个 token 递推后的压缩结果，但没有保存一条可逆轨迹。通常不存在函数 `G`，使系统只拿 `S_1000` 就能得到 `S_600`。要获得 `S_600`，必须当时保存它，或者从更早 checkpoint 重新执行 token 递推到 600。

这与“state 大小固定”并不矛盾。固定大小只是说明 `S_t` 的 tensor shape 不随 `t` 增长，不表示它可以表示所有历史时刻的独立快照。

### 5.2 分叉发生在 checkpoint 中间时

假设已有缓存：

```text
token 路径：0 ---------------- 1024
Mamba checkpoint：             S_1024
```

新请求只与它共享前 768 个 token：

```text
公共前缀：0 -------- 768
旧分支：              769 -------- 1024
新分支：              769 -------- ...
```

Full KV 可以在 768 处拆开。Mamba 只有 `S_1024`，它属于旧分支，不能挂到新建的 768 父节点上。统一组件的 split hook 因而执行：

```python
new_parent.component_data[MAMBA].value = None
child.component_data[MAMBA].value = old_checkpoint
```

如果 768 之前还有一个 `S_512`，新请求可以从 512 恢复，再对 token 513 到 768 做一次 prefill/recurrent replay。若没有更早 checkpoint，就只能从根状态开始重新计算线性注意力部分。

这带来一个重要限制：混合模型能够复用到的位置，必须同时存在有效的 Full KV 和 Mamba checkpoint。即使 token 树还能匹配得更深，Mamba 状态只保存到较早的对齐点，真正可直接恢复的 prefix 也要以有效 checkpoint 为准；剩余 token 需要重新递推。

命中 checkpoint 后，SGLang 不会让新请求直接改写树上的共享 Mamba slot。`MambaComponent.finalize_match_result()` 会为请求分配活动 slot，并记录 copy-on-write 的源 slot。实际执行前再把缓存状态复制到新 slot，随后这个请求只修改自己的活动状态。

### 5.3 copy-on-write 为什么延迟执行

命中阶段运行在 scheduler 的缓存管理路径，模型 forward 使用自己的 CUDA stream 和批量 tensor。`finalize_match_result()` 不立刻逐请求复制所有 conv/temporal tensor，而是先记录：

```text
req.mamba_cow_src_index = cached_slot
req.mamba_pool_idx      = new_active_slot
```

随后 `ScheduleBatch._collect_deferred_mamba_cow_and_clear()` 把一批请求的源、目标 slot 合并成 tensor：

```text
mamba_cow_src_indices
mamba_cow_dst_indices
mamba_clear_indices
```

真正的 `MambaPool.copy_from(src, dst)` 或 clear 在 forward stream 上批量执行。这样既避免 scheduler 侧零散同步，也保证模型开始读活动状态前，COW 已在正确 stream 顺序中完成。

`copy_from()` 会复制所有 conv state 和 temporal state。若启用了 ReplaySSM，它还要求源 checkpoint 已经 flush，复制后把目标 ring cursor 重置为 0，因为 Radix checkpoint 表示一个没有未提交 ring update 的完整状态。

### 5.4 running state、tracking state、Radix checkpoint

这三种状态 tensor shape 可以相同，但所有权不同：

| 状态 | 谁拥有 | 是否会继续被模型写 | 生命周期 |
|---|---|---|---|
| running slot | 活动请求 | 会 | 请求执行期间 |
| tracking slot | 活动请求的 ping-pong buffer | 到达下一跟踪边界时会覆盖或交换 | 请求执行期间 |
| Radix checkpoint | Prefix Cache 节点 | 不会，作为只读恢复点 | 到被淘汰或 cache reset |

donate 操作改变的是第三列前的所有权：某个 tracking slot 从“请求暂存”变成“Radix 只读 checkpoint”。COW 则执行反方向的数据复制，从只读 checkpoint 生成新的 running slot。

## 6. ping-pong buffer 与 checkpoint 的关系

活动请求的 Mamba state 一直在变化。为了在不中断正常计算的情况下拿到稳定快照，SGLang 提供三种策略：

| 策略 | 活动请求持有的状态 | 特点 |
|---|---|---|
| `no_buffer` | 1 个活动 slot | 保存 prefix 时复制或转交活动状态；要求更严格，当前组件在关闭 extra buffer 时要求 `page_size=1` |
| `extra_buffer` | 1 个活动 slot，加 1 或 2 个跟踪 slot | overlap schedule 下通常使用两份 ping-pong 跟踪槽，计算和快照可以错开 |
| `extra_buffer_lazy` | 1 个活动 slot，加 1 个常驻跟踪 slot | 第二个跟踪 slot 只在边界按需分配，使用后释放 |

### 6.1 `no_buffer`：保存时复制当前状态

`no_buffer` 只有 running slot。请求仍在运行、但需要把当前前缀提交给 Radix 时，不能把 running slot 直接拿走，否则后续 forward 没有工作状态。`MambaComponent.prepare_for_caching_req()` 会：

```text
分配一个新 checkpoint slot
  → MambaPool.copy_from(running, checkpoint)
  → 把 checkpoint slot 插入 Radix
  → running slot 继续归请求使用
```

请求已经结束时，running slot 不再需要继续写，可以直接把它作为候选 `mamba_value` 交给树；如果相同 checkpoint 已存在，cleanup 再释放这份未采用的 slot。

这一策略逻辑直接，但状态复制可能与请求执行路径形成同步点。当前 `MambaComponent` 还要求关闭 extra buffer 时 `page_size=1`，否则 token page 对齐与精确 state 位置难以统一。

### 6.2 `extra_buffer`：运行和快照分离

开启 overlap schedule 时，`HybridReqToTokenPool` 为每个活动请求准备两个 tracking slot：

```text
running slot：模型正常递推
track[0]：    某个稳定边界的快照
track[1]：    下一次快照目标
```

`mamba_next_track_idx` 指示下一次 forward 应写哪个 tracking slot。到达边界后，普通模式把 index 切换到另一侧。保存 prefix 时，`get_mamba_ping_pong_keep_idx()` 找出最近完成的那一侧。

中间提交采用 donate，而不是再复制一份：

```text
track[0] = slot 31，已经是稳定快照

donate slot 31 给 Radix
为请求分配 slot 52
track[0] = slot 52，后续继续使用
```

slot 31 和 slot 52 都来自同一个 MambaPool。所谓 extra buffer 是额外 slot，不是另一套 tensor pool。

### 6.3 `extra_buffer_lazy`：减少常驻 tracking slot

lazy 模式初始只分配一个 tracking slot，另一个位置先写成 `-1`。请求真正到达 track boundary 前，再尝试为第二侧分配 slot。边界处理完成后，临时 slot 可以释放。

它降低了很多短请求永远用不到第二个 tracking slot 时的常驻占用，但状态机更严格。若请求在需要 checkpoint 时没有成功准备有效另一侧，代码会跳过不安全的 cache insert，不能把被覆盖或不完整的状态挂进 Radix。

`auto` 会结合模型/后端是否支持 extra buffer、是否启用 overlap schedule、`page_size` 等条件，在 `extra_buffer` 和 `no_buffer` 之间选择。它不是所有模型都固定使用同一种方案。

当前 override 逻辑的主干近似为：

```text
如果策略是 auto：
  wants_overlap = overlap schedule 没有关闭
  wants_paging  = page_size > 1

  如果 (wants_overlap 或 wants_paging) 且模型/后端支持 extra buffer：
    strategy = extra_buffer
  否则：
    strategy = no_buffer
    disable_overlap_schedule = True
```

因此看到默认参数 `auto` 时，不能只读 dataclass 默认值判断最终运行模式；应查看 server args post-process 后的 resolved value。

源码中的三个容量常量用于新请求入场时的 Mamba slot 预留和淘汰余量：

```text
支持 Mamba prefix cache，非 lazy：3
支持 Mamba prefix cache，lazy：   2
不由 prefix cache 管理 Mamba：    1
```

对应常量位于 `python/sglang/srt/mem_cache/common.py`，`alloc_req_slots()` 会用它们计算保守的 eviction headroom。对启用 overlap 的 `extra_buffer`，3 个 slot 正好对应 1 个 running 加 2 个 ping-pong；`no_buffer` 虽然不持有两份 ping-pong，非 lazy prefix cache 的入场检查仍按 3 倍预留，为 checkpoint 和 copy-on-write 留出空间。

物理上也不是“一个完整 MambaPool，再复制两个同样大的 MambaPool”。系统只有按容量配置的大 pool，running、tracking 和 Radix checkpoint 都从中占用固定大小的 slot。这里的 3 表示一个新请求需要准备的 slot 余量，不表示整池复制三遍，也不表示每个 Radix checkpoint 都附带两份 ping-pong。

一次典型的状态流转是：

```text
活动 slot 继续计算
  → 到达跟踪边界，把稳定状态写入 ping-pong slot
  → 请求需要提交 prefix 时，把最新有效 slot donate 给 Radix 节点
  → 给活动请求补一个新 slot，继续后续跟踪
```

`donate` 是转移槽位所有权，不是复制整个大 tensor。旧 slot 归 Radix 节点，新 slot 替代它进入 ping-pong buffer。相关代码是 `HybridReqToTokenPool.donate_mamba_ping_pong_slot()` 和 `MambaComponent.prepare_for_caching_req()`。

### 6.4 插入完成后的 slot 清理

一次 insert 可能遇到两种结果：

```text
新 checkpoint：目标节点原来没有 Mamba value
重复 checkpoint：目标节点已经有等价 Mamba value
```

`InsertResult.mamba_exist` 用来区分它们。如果插入成功，新 donate 的 slot 留在 Radix；如果节点已经有 checkpoint，新分配或复制的候选 slot 必须立即释放，否则每次重复请求都会泄漏一份 state。

请求完成时还要释放 running slot 和未被保留的 ping-pong slot。`free_mamba_cache()` 接收可选的 `mamba_ping_pong_track_buffer_to_keep`，只留下已经转交给 Radix 的那一侧，其余 slot 交还 `MambaSlotAllocator`。

### 6.5 256 token 到底控制什么

当前 `mamba_track_interval` 默认是 256，它控制 decode 期间多久刷新一次可供 Radix 使用的跟踪状态。`batch_result_processor.py` 中的 `_mamba_check_track_boundary()` 判断请求是否跨过该边界。

非 speculative decode 的判断直接使用已提交 KV 长度：

```python
if req.kv_committed_len % mamba_track_interval == 0:
    track_seqlen = req.kv_committed_len
```

例如请求从长度 511 decode 到 512，且 interval 为 256，forward 会在 512 位置生成稳定跟踪状态，result processor 再记录 `mamba_last_track_seqlen=512` 并切换 ping-pong index。

speculative decode 一次可能接受多个 token，不能只判断最终长度取模。代码比较接受前后是否跨过 interval bucket，并把 checkpoint 位置取到最近跨过的完整边界。

这不等于“每生成 256 token，就永久保存一个 checkpoint”。ping-pong slot 会被反复覆盖或交换。只有请求在 chunk 边界被提交、完成，或缓存逻辑把最新稳定 slot 交给某个 Radix 节点时，它才成为树上可复用的 checkpoint。不同会话分支持续进入树后，才可能形成多个持久到淘汰为止的状态槽。

prefill 还有另一个对齐量 `mamba_cache_chunk_size`。它由模型的 `mamba_chunk_size`（没有时使用线性注意力的默认 chunk）与 `page_size` 共同决定，用来选择 prefill 过程中可以提取状态的序列位置。两者不要混为一谈：

- `mamba_track_interval`：decode 期间跟踪状态的刷新间隔；
- `mamba_cache_chunk_size`：prefill 状态 checkpoint 的计算和对齐粒度。

相关逻辑位于 `python/sglang/srt/server_args.py`、`python/sglang/srt/managers/schedule_batch.py` 和 `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`。

### 6.6 prefill 为什么使用另一个 chunk 对齐

线性注意力 prefill 通常按 chunk 计算中间 state。`ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend()` 根据本次 extend 长度选择可提取的 state：

```text
prefix_len = P
extend_len = E
chunk_size = C

tracked_len = P + floor(E / C) * C
```

假设已经命中 1024 token，本轮 prefill 300 token，`mamba_cache_chunk_size=128`，那么本轮能稳定跟踪到：

```text
1024 + floor(300 / 128) * 128 = 1280
```

最后 44 个 token 已经参与本轮计算，但如果 backend 没有在那个任意位置输出可保存 state，它们不能算进该 checkpoint。请求完成时 UnifiedRadixCache 会把可提交 key/KV 长度截断到 component 返回的最小 `effective_cache_len`，保证 token key 与 Mamba state 表示同一个序列位置。

代码还要区分 state 来自 chunk 的 `h` 中间结果，还是最后一个位置的 `last_recurrent_state`。某些对齐位置不是本次 extend 的最后位置时，会用 `_force_track_h()` 调整索引，让 backend 从正确的中间 state 取快照。

### 6.7 ReplaySSM 与 256-token checkpoint 不是一回事

当前 GDN/KDA 路径还可选 ReplaySSM。它用一个较短 ring 缓冲最近的 recurrent update，减少每个 decode step 对完整 temporal state 的 HBM 写回；ring 满或遇到强制边界时再 flush。

ReplaySSM 的 `linear_replayssm_cache_len` 控制内部 ring 长度，`mamba_track_interval` 控制 Radix tracking 边界。前者是 kernel/HBM 优化，后者是 Prefix Cache 快照节奏。为了让 Radix checkpoint 自洽，`MambaPool.copy_from()` 要求源 ring 已经 flush，`write_pos==0`；否则只复制 temporal state 会丢掉 ring 中尚未折叠的更新。

## 7. 一次混合模型的多轮 cache hit

这一节把前面的对象放回一个完整请求。为了便于追踪，先列出 `Req` 上与缓存相关的主要字段：

| 字段 | 用途 |
|---|---|
| `req_pool_idx` | 选择 ReqToTokenPool 的请求行 |
| `prefix_indices` | 当前可复用的 device KV indices |
| `cache_protected_len` | 已由 Prefix Cache 锁保护的 token 长度 |
| `last_node` | device prefix 的锁锚点 |
| `best_match_node` | 所有组件共同接受的最深节点 |
| `mamba_pool_idx` | 本请求的 running state slot |
| `mamba_ping_pong_track_buffer` | tracking slot index 数组 |
| `mamba_last_track_seqlen` | 最近有效 checkpoint 对应的序列长度 |
| `mamba_cow_src_index` | prefix hit 后等待复制的 checkpoint slot |
| `mamba_branching_seqlen` | 需要重新构建 state 的潜在分叉位置 |

这些字段分成两类：`prefix_indices/last_node` 描述共享缓存，`req_pool_idx/mamba_pool_idx` 描述当前活动请求。多轮对话每轮都会有新的请求对象和活动 slot，但可以指向相同的共享 Prefix Cache 节点。

### 7.1 第一轮开始：没有 prefix hit

仍以两轮对话为例。第一轮处理 `S + Q1` 并生成 `A1` 时：

```text
Full Attention 层：为 S + Q1 + A1 保存逐 token KV
KDA/GDN/Mamba 层：维护当前递推状态，并在有效对齐位置留下 checkpoint
Prefix Cache：用 S + Q1 + A1 的 token 路径索引上述两类状态
```

运行时过程更详细地展开如下：

```text
1. init_next_round_input()
   Radix Tree 为空，prefix_indices=[]

2. 调度 prefill
   extend_input = S + Q1
   为请求分配 req_pool_idx、KV slots、mamba_pool_idx

3. 模型 forward
   Full 层写逐 token KV
   KDA/GDN/Mamba 层更新 running state
   extra_buffer 模式在可用位置写 tracking state

4. decode A1
   每步继续追加 KV、更新 running state
   跨过 track interval 时刷新 ping-pong state

5. release_kv_cache()
   根据 effective_kv_committed_len 提交已完成部分
   UnifiedRadixCache.cache_finished_req() 让各 component 准备 insert data
   token key、Full KV indices、Mamba checkpoint 进入同一叶节点

6. 清理活动资源
   释放请求行、running slot、未保留 tracking slot
   Radix 持有的 KV/checkpoint 继续留在 pool
```

需要注意最后一个 token 的 KV 是否已经 committed，由 `kv_committed_len` 和具体 decode/speculative 路径决定。Prefix Cache 插入的不是“字符串意义上的完整回答”，而是实际已经形成有效模型状态的 token 前缀。

### 7.2 第二轮开始：匹配完整历史

第二轮输入 `S + Q1 + A1 + Q2` 时：

```text
Radix 匹配公共 token 前缀
  → Full 组件返回公共前缀的 KV 物理索引
  → Mamba 组件找到最新有效 checkpoint
  → copy-on-write 到本请求的活动 Mamba slot
  → 仅对 checkpoint 之后的未复用尾部做 prefill/递推
  → decode 新回答 A2
```

对应代码链可以写成：

```text
Req.init_next_round_input()
  → RadixKey(full_untruncated_fill_ids, extra_key)
  → UnifiedRadixCache.match_prefix()
    → FullComponent validator
    → MambaComponent validator
    → MambaComponent.finalize_match_result()
      记录 checkpoint → active slot 的 deferred COW
  → req.prefix_indices = match_result.device_indices

ScheduleBatch.prepare_for_extend()
  → input_ids = full_input[len(prefix_indices):]
  → 收集 mamba_cow_src_indices / mamba_cow_dst_indices

model forward
  → 先恢复/复制 Mamba state
  → 只计算未命中尾部
  → 进入正常 decode
```

从这里可以看出，第二轮不是第一轮 decode loop 的延续。请求调度、sampling params、输出流和活动 memory slot 都是新的；只是输入历史的模型状态通过 Prefix Cache 复用。

### 7.3 一个带 checkpoint 回退的具体例子

假设第二轮输入共有 4200 个 token：

```text
Full KV 最长匹配：          4096
最近有效 Mamba checkpoint： 3840
新问题尾部结束位置：         4200
```

混合模型不能直接从 Full KV 的 4096 开始，因为 Mamba state 只到 3840。有效执行可以理解为：

```text
0    -------- 3840：Full KV + Mamba state 一起复用
3840 -------- 4096：已有 Full KV 不能单独完成恢复，重新 prefill/recurrent
4096 -------- 4200：新输入尾部正常 prefill
4200 -------- ... ：decode 新回答
```

统一实现会用共同有效的 `device_indices` 和 branching 信息组织这段计算，而不是让 Full 与 Mamba 各自从不同长度直接进入 forward。

如果 Mamba checkpoint 比 Full KV 的最长匹配位置更早，`A1` 尾部的一部分也可能需要重新计算。它仍然是 cache hit，只是有效命中长度受 Mamba checkpoint 边界约束。

如果相关树节点已经被淘汰，则第二轮从仍然存在的更短前缀继续；完全没有匹配状态时，对完整输入重新 prefill，然后照常 decode。cache miss 从来不等于“完整输入全部走 decode”。Prefill 负责读取输入上下文，decode 负责一个个生成新 token。

### 7.4 代码修改场景为什么通常仍能命中大部分

代码 Agent 的第二轮 prompt 常包含固定 system prompt、仓库摘要、旧对话、上轮工具结果和新的 diff。只要序列组织方式是 append-only，变化发生在输入尾部，旧前缀仍然可以命中：

```text
第一轮：S + repo_snapshot_v1 + Q1 + A1
第二轮：S + repo_snapshot_v1 + Q1 + A1 + tool_diff + Q2
```

这里 `tool_diff + Q2` 是未命中尾部。如果客户端每轮都把“最新完整代码库快照”插回 system prompt 中间，情况会不同：

```text
第一轮：S + repo_snapshot_v1 + Q1
第二轮：S + repo_snapshot_v2 + Q1 + ...
```

Radix 在 `repo_snapshot_v1` 与 `repo_snapshot_v2` 第一个不同 token 处停止，后面的旧对话即使文本相同也无法跨过分叉继续命中。Prefix Cache 只复用连续前缀，不复用任意公共子串。

### 7.5 为什么多轮对话也不是百分之百保证命中

即使历史消息没有被用户修改，仍可能出现这些情况：

- GPU/host cache 因容量压力淘汰了旧节点；
- 负载均衡把新请求路由到另一个没有共享缓存的实例；
- server 重启，默认内存缓存全部消失；
- chat template、system prompt、tool schema 或 tokenizer 版本变化；
- LoRA、cache salt、位置 embedding override 等隔离条件变化；
- 上一轮状态没有到达安全 checkpoint，或完成时选择跳过 insert；
- page/chunk 对齐让最后一小段不能进入可共享 prefix。

所以“历史固定”只满足 token key 稳定这一项。物理状态仍在、路由命中同一缓存域、组件恢复条件有效，三者同时成立才会产生完整的 cache hit。

## 8. 100K 上下文会不会存下大量 Mamba state

需要把活动状态和 prefix checkpoint 分开计算。

对单个正在运行的 100K 请求：

- Full KV 仍然随 token 数增长，100K token 就需要对应规模的 KV 或 latent KV；
- Mamba 活动状态的大小主要由层数和 state shape 决定，不随这 100K token 线性增长；
- 开启 extra buffer 后，再增加跟踪用的 1 或 2 个同形 slot。

### 8.1 用公式拆开 Mamba 占用

设一个全精度 Mamba slot 占用 `B_slot`，活动请求数为 `R`，每个请求常驻 tracking slot 数为 `T`，Radix 当前保留 checkpoint 数为 `C`。忽略 speculative state、allocator padding 和 HiCache host pool，Mamba HBM 占用可以近似写成：

$$
B_{\text{Mamba}}
\approx
R \times (1 + T) \times B_{slot}
+ C \times B_{slot}
$$

其中：

- `1` 是每个活动请求的 running slot；
- `T` 在 no_buffer 为 0，普通 overlap extra_buffer 通常为 2，lazy 稳态通常为 1；
- `C` 是 Radix 真正保留的 checkpoint 数，不是 token 数。

在 donate 瞬间，请求可能先申请替换 slot，再把旧 tracking slot 交给 Radix；在 COW 命中时，也可能先申请 destination slot。这就是调度器需要保守 headroom 的原因。它不能只按最终稳态公式把显存塞满，否则状态所有权切换的瞬时分配会失败。

### 8.2 100K token 与 390 个 checkpoint 没有直接等号

`100000 / 256 ≈ 390` 只能说明 decode 长度跨过了约 390 个 tracking boundary。普通 ping-pong 行为是反复复用 1 或 2 个 tracking slot：

```text
256：  track[0] 写 S_256
512：  track[1] 写 S_512
768：  track[0] 覆盖为 S_768
1024： track[1] 覆盖为 S_1024
```

如果请求一直运行、没有在每个边界生成独立 Radix 分支，前面的 `S_256` 和 `S_512` 已被覆盖，不会形成 390 个常驻 checkpoint。

真正让 `C` 增长的是独立前缀被提交并继续留在树上。例如 100 个会话分别在不同 token 长度完成，或者同一公共前缀分出大量后续分支，每个可复用叶/内部点都可能持有一个 checkpoint。此时容量与会话工作集相关，而不是与某个单请求的上下文长度简单线性相关。

Prefix Cache 会改变第三点：每个被保留的会话分支或 Radix 节点可以拥有一个 Mamba checkpoint。因此，Mamba prefix cache 的总占用与“保留了多少个有效 checkpoint”相关，而不是简单等于“上下文 token 数 ÷ 256”。活跃分支很多时，checkpoint pool 仍可能很大，SGLang 会通过独立的 Mamba LRU、lock reference 和容量预算淘汰旧状态。

### 8.3 与 Full KV 的增长方式对照

同一个 100K 混合模型请求可以同时具有两种不同的增长曲线：

```text
Full Attention KV
  与 Full 层数 × 已缓存 token 数近似线性增长

Mamba running state
  与 linear 层数 × state shape × 活动/跟踪 slot 数相关
  对单请求不随 100K token 线性增长

Mamba Radix checkpoints
  与仍保留的可复用前缀节点数相关
```

因此“单个 Mamba state 比单层 KV 大”不能直接推出“100K Mamba cache 一定比 100K KV 大”。正确比较应把 KV 的 token 维乘进去，同时把 Mamba 的层维和 checkpoint 数乘进去。

实际容量由 `kv_cache_configurator.py` 统一预算。`mamba_full_memory_ratio` 用于平衡 Mamba state memory 与 Full KV memory，`max_mamba_cache_size` 可以限制 Mamba slot 数。最终值还会受最大并发请求、speculative decoding、page size 和剩余 HBM 约束。

当前代码还提供可选的 int8 Mamba checkpoint pool。活动状态仍使用原精度 MambaPool，交给 Radix 的 checkpoint 可量化后存入单独 pool，命中时再恢复到活动 slot。它是显存内的容量优化，不是落盘机制；默认不开启。实现位于 `python/sglang/srt/mem_cache/mamba_checkpoint_pool.py`。

### 8.4 int8 checkpoint pool 保存什么

开启 `enable_int8_mamba_checkpoint` 后，running state 仍在全精度 `MambaPool`，只有 Radix-owned checkpoint 进入 `MambaCheckpointPool`：

```text
temporal state：int8 qdata + per-(head, k-channel) scale
conv window：   保持原 dtype
```

插入时 `store_from_active()` 量化 temporal 并复制 conv；命中时 `load_to_active()` 一次性反量化到新的全精度活动 slot，后续 recurrence 不在 int8 上运行。默认 checkpoint slot 数是 active Mamba pool size 的 2 倍，可由 `int8_mamba_ckpt_size` 调整。

“约 2 倍容量”是 temporal state 从 bf16/fp16 变成 int8 后的近似说法。scale 和未量化 conv 仍占空间，所以不能把整池字节数机械除以 2。

当前 server args 明确拒绝 int8 Mamba checkpoint 与 `enable_hierarchical_cache` 同时开启，也拒绝不理解该 pool 的自定义 radix backend。虽然 `mamba_checkpoint_pool.py` 的模块说明提到 host-offload 组合，运行时校验才是当前 checkout 的实际边界，文档应以后者为准。

## 9. 状态默认保存在哪里

在 GPU serving 的默认配置下，KV tensor、Mamba 活动状态和 Radix checkpoint 指向的状态都在 GPU pool 中。Radix Tree 的 Python 节点和索引管理在 CPU 侧，但节点的 device value 指向 GPU 槽位。普通 prefix cache 不会自动把缓存写入 SSD，也不会在服务重启后保留。

### 9.1 默认 Radix Cache 的 CPU/GPU 边界

默认结构可以画成：

```text
CPU Python objects
  Radix Tree 拓扑
  token edge key
  parent / children
  lock_ref / LRU metadata

GPU tensors
  ReqToTokenPool 映射
  TokenToKVPool 中的 KV
  MambaPool 中的 running/tracking/checkpoint state
```

树节点在 CPU 不代表 KV 已经 offload 到 CPU。`node.value=[31, 96, ...]` 只是几个索引，真正的大 tensor 仍位于 device pool。进程退出后，Python 树和 GPU pool 一起消失。

启用 HiCache 后，`UnifiedRadixCache` 可以为 Full、SWA、Mamba 组件附加 host-side pool，并按配置继续接入 L3 storage。此时命中结果可能先从 host 或外部存储 load back。HiCache 是额外的分层缓存能力，不应与默认 Radix prefix cache 淵称为“自动落盘”。

相关入口位于 `UnifiedRadixCache.init_hicache()` 和 `MambaComponent` 的 HiCache hooks。

### 9.2 HiCache 的三层数据流

HiCache 的层次可简化为：

```text
L1 Device：GPU KV pool / MambaPool
    ↕ backup / load back
L2 Host：CPU pinned-memory pool
    ↕ storage write / prefetch
L3 Storage：配置的外部存储 backend
```

`UnifiedRadixCache.init_hicache()` 创建 controller 并把对应 host pool 挂到各 component。Mamba component 通过 `PoolName.MAMBA` 构造 transfer：

```text
BACKUP_HOST：   Mamba device slot → host Mamba slot
LOAD_BACK：     host Mamba slot → 新 device slot
BACKUP_STORAGE：host data → L3 backend
PREFETCH：      L3 backend → host pool
```

Full KV 传输的单位是 token/page，Mamba 传输的单位是整个 state slot。统一树仍用同一 token key 定位节点，但每个 component 提交自己的 transfer 描述。

### 9.3 device hit、host hit 和 storage hit

一次匹配可能出现：

```text
前 2048 token：Full KV 和 Mamba state 都在 GPU
2048-4096：    Full KV 在 host，Mamba checkpoint 也在 host
4096 之后：    只有逻辑 key 或 L3 记录
```

`MatchResult` 分别记录 device anchor、host anchor 和 host hit length。scheduler 不能立刻把 host hit 当作 `prefix_indices` 使用，必须先申请 device slot 并完成 H2D load back。对 Mamba state，`mamba_host_hit_length` 通常只需表示一个 checkpoint slot；对 Full KV，则可能是成百上千个 token page。

HiCache 的收益和代价都很明确：它减少 GPU 淘汰后的完整 re-prefill，但增加 host/storage 容量、数据传输、超时处理和一致性管理。是否比重新计算划算，取决于前缀长度、互联带宽、模型 prefill 吞吐和命中工作集。

### 9.4 一个月后还能不能命中

默认 Radix Cache 不能保证。进程重启、GPU 淘汰或请求路由变化都会让状态消失。

即使服务商实现了持久化 Prefix Cache，也要同时保留 token key、模型/版本命名空间、KV/Mamba 状态格式以及可用的路由索引，才可能在一个月后恢复。模型权重、tokenizer、位置编码策略或 cache layout 变化，都可能让旧状态失效。

对话历史长期保存在数据库，只代表服务端能够重新构造 prompt。它不等于模型内部 KV/state 也长期保存。最稳妥的语义仍是：历史文本可以重新 prefill；Prefix Cache 是性能优化，命中与否不应影响正确性。

## 10. 淘汰时发生什么

统一树中的各组件拥有独立的资源统计和 LRU。活动请求通过 lock reference 保护正在使用的节点；内存不足时，未锁定的旧状态可以被淘汰。

### 10.1 为什么不能只用一个“缓存大小”

Full KV 和 Mamba state 的分配单位不同，UnifiedRadixCache 分别维护：

```text
component_evictable_size_[FULL]   # token slots
component_protected_size_[FULL]

component_evictable_size_[MAMBA]  # state slots
component_protected_size_[MAMBA]
```

token allocator 申请失败时，可以请求 `EvictParams(num_tokens=N)`；Mamba slot allocator 申请失败时，请求 `EvictParams(mamba_num=M)`。`UnifiedRadixCache.evict()` 让每个 component 按自己的 target 驱动淘汰，而不是把一个 Mamba slot 粗暴换算成若干 token 后共用同一 LRU。

### 10.2 叶节点删除和内部 tombstone

Full 组件释放的是一段 token KV 槽位，Mamba 组件释放的是一个 state/checkpoint slot。内部节点的某个组件被淘汰后，树的逻辑 token 结构可以暂时保留为 tombstone，其他仍存在的组件不一定同时消失；叶节点淘汰则可能触发整节点清理。统一缓存还定义了组件间的级联淘汰规则，以维持 Full、SWA、Mamba 数据的一致性。

内部节点可能仍是其他分支的公共 token 路径，所以淘汰 Mamba checkpoint 时可以只做：

```text
free(mamba_value)
node.component_data[MAMBA].value = None
```

Full KV 或子节点仍在，逻辑节点继续存在。以后请求走到这里时，validator 发现 Mamba value 是 tombstone，不把它当作完整混合恢复点。

叶节点没有子分支继续依赖。统一实现会把 device leaf 的组件作为一个一致性单元处理，释放资源后从父节点移除，并向上清理已经没有 component data、也没有子节点的 tombstone 链。

### 10.3 级联淘汰优先级

统一组件给内部节点定义了大致优先级：

```text
Full：  2
SWA：   1
Mamba： 0
```

低优先级辅助状态可以单独变成 tombstone。例如淘汰一个内部 Mamba checkpoint，不要求同时删除仍可服务其他用途的 Full KV。反过来，如果内部 Full 基础数据被淘汰，同节点上依赖它组织恢复语义的低优先级组件也会级联处理。

叶节点各组件优先级都视为 0，任何组件驱动叶删除时都要保持节点整体一致，不能留下一个没有基础 token path 的孤立 checkpoint。

### 10.4 lock 如何改变 LRU 账本

当 lock reference 从 0 变为 1，组件把对应资源从 evictable 计数转入 protected 计数；最后一个请求释放锁时，再移回 evictable。Full 组件通常锁整条祖先路径，Mamba 组件只锁实际使用的 checkpoint 节点。

这也是 cache hit 后不能立即把命中节点淘汰的原因。虽然请求已经把 Full indices 写入自己的 `ReqToTokenPool` 行，indices 指向的物理 slot 仍由树共享；树必须知道当前有请求依赖它。Mamba COW 完成后，活动请求拥有自己的 state，checkpoint 锁可以按生命周期释放，但恢复动作完成前仍需防止源 slot 被回收。

被淘汰只会降低下一次命中长度，不会改变模型语义。新请求重新 prefill 缺失的那部分即可恢复相同位置的内部状态。

### 10.5 重新 prefill 与旧 checkpoint 的结果是否相同

在相同模型权重、输入 token、位置、dtype 和确定性 kernel 条件下，从头 prefill 与从等价 checkpoint 恢复应表示同一模型状态。两条路径的差别是计算是否重复，不是模型“有没有再次读书”：

```text
从头 prefill 一本书
  每个 token 重新经过模型，生成最终 KV/state

checkpoint hit
  直接装载上次读完相同前缀后留下的 KV/state
```

如果 checkpoint 被淘汰，模型必须重新读取书的 token，也就是重新 prefill；它不会因为数据库里还存着回答文本就自动知道书的内部表示。

浮点 kernel 调度、量化 checkpoint 或非确定性实现可能产生数值级微小差异，但 Prefix Cache 的设计要求恢复状态在模型语义上等价。若 token key 与 state 序列位置不一致，那是缓存正确性错误，而不是允许的 cache miss 行为。

### 10.6 cache reset 与进程重启

树 reset 不只清 Python 节点，还要清对应 allocator 的 slot 所有权。`HybridReqToTokenPool.clear()` 会重置 Mamba allocator；如果启用了独立 int8 checkpoint pool，也要同时 clear，否则树已经丢掉索引，checkpoint slot 却仍被标记为占用，会形成不可达的内存泄漏。

## 11. 源码阅读地图

| 路径 | 建议关注的内容 |
|---|---|
| `python/sglang/srt/mem_cache/memory_pool.py` | `ReqToTokenPool`、KV pool、`MambaPool`、`HybridReqToTokenPool` |
| `python/sglang/srt/mem_cache/allocator/mamba.py` | request-level Mamba slot 分配 |
| `python/sglang/srt/mem_cache/base_prefix_cache.py` | `MatchResult`、`InsertParams`、`EvictParams` 等统一接口 |
| `python/sglang/srt/mem_cache/common.py` | cache release 与 Mamba admission headroom 常量 |
| `python/sglang/srt/mem_cache/allocation.py` | 请求入场前的 KV/Mamba slot 分配和淘汰余量 |
| `python/sglang/srt/mem_cache/radix_cache.py` | 普通 KV prefix cache 的匹配、插入和淘汰 |
| `python/sglang/srt/mem_cache/registry.py` | 普通模型与 hybrid SSM 的缓存实现选择 |
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | 统一树的请求提交、匹配、锁和淘汰 |
| `python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py` | Mamba checkpoint、copy-on-write、donate 和 HiCache hooks |
| `python/sglang/srt/mem_cache/unified_cache_components/README.md` | UnifiedRadixCache 组件接口和级联淘汰设计 |
| `python/sglang/srt/mem_cache/mamba_radix_cache.py` | 专用 Mamba Radix 实现，可用于理解 checkpoint 不可切分的细节 |
| `python/sglang/srt/mem_cache/mamba_checkpoint_pool.py` | int8 checkpoint 的量化、反量化和独立 allocator |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | Full KV 与 Mamba state 的显存预算 |
| `python/sglang/srt/managers/schedule_batch.py` | 请求匹配结果如何转成 extend/prefill 范围 |
| `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` | decode 跟踪边界与 ping-pong 更新 |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | prefill/extend 中 state track index 的计算 |
| `python/sglang/srt/server_args.py` | cache strategy、track interval、chunk size 等配置 |

### 11.1 推荐阅读顺序

如果目的是先建立整体心智模型，建议按下面的顺序读：

```text
1. memory_pool.py
   分清 req row、token KV slot、Mamba state slot

2. radix_cache.py
   先理解普通 KV 的 match / insert / lock / evict

3. registry.py
   确认当前模型最终实例化哪个 tree cache

4. unified_cache_components/README.md
   理解一棵树、多组件的设计

5. mamba_component.py
   看 checkpoint validator、COW、donate、cleanup

6. schedule_batch.py + batch_result_processor.py
   把缓存对象接回 prefill/decode 请求生命周期

7. kv_cache_configurator.py + HiCache
   最后再看容量预算和分层存储
```

直接从 `mamba_radix_cache.py` 一路向下读，容易把旧专用类误认成当前默认入口；直接从 kernel 读，又容易看懂 state 更新却不知道 checkpoint 的所有权何时变化。

### 11.2 主要配置项

| ServerArgs 字段 | 默认值 | 作用 |
|---|---:|---|
| `disable_radix_cache` | `False` | 关闭 Prefix Cache；不等于关闭请求内 KV Cache |
| `mamba_radix_cache_strategy` | `auto` | 在 `no_buffer`、`extra_buffer`、`extra_buffer_lazy` 中解析最终策略 |
| `mamba_track_interval` | `256` | decode 阶段跟踪 Mamba state 的边界间隔 |
| `mamba_full_memory_ratio` | `0.9` | Mamba state 与 Full KV 的内存预算比例参数 |
| `max_mamba_cache_size` | `None` | 可选的 Mamba cache slot 数上限 |
| `enable_int8_mamba_checkpoint` | `False` | 用独立 int8 pool 保存 Radix-owned checkpoint |
| `int8_mamba_ckpt_size` | `None` | int8 checkpoint slot 数，未设置时使用 active pool size 的 2 倍 |
| `enable_hierarchical_cache` | `False` | 启用 device/host/可选 storage 分层缓存 |

`mamba_cache_chunk_size` 不是直接保存的普通 dataclass 参数，而是 property。它根据模型 `mamba_chunk_size` 和 resolved `page_size` 计算，并要求较大者能被较小者整除。

关闭 Radix Cache 后，模型仍然必须有请求内 KV Cache 和 running Mamba state，否则同一请求无法正常 decode。`disable_radix_cache` 关闭的是跨请求 prefix reuse，不是把所有推理状态都关闭。

### 11.3 调试时先观察什么

出现命中长度不符合预期时，可以按下面的顺序排查：

```text
token key
  完整 token IDs 是否真的是 append-only
  extra_key / LoRA / salt 是否相同

page 和 chunk
  page_size 向下对齐后还剩多长
  mamba_cache_chunk_size 对齐到哪里
  mamba_last_track_seqlen 是多少

tree match
  prefix_indices 长度
  last_device_node / best_match_node
  mamba_branching_seqlen

slot ownership
  mamba_pool_idx
  mamba_cow_src_index
  ping-pong buffer 与 next_track_idx

capacity
  Full/Mamba available、evictable、protected 分别是多少
  是否发生过 device/host 淘汰
```

只看一个汇总 cache hit rate 很难区分“token 在第 10 个位置就不同”和“Full 能命中很深，但 Mamba checkpoint 较早”。混合模型调试需要同时看 token prefix length 和 state checkpoint length。

### 11.4 常见误解

**“关闭 Prefix Cache 就不需要 KV Cache。”** 不对。请求内部 decode 仍然依赖历史 KV；关闭的是跨请求复用和 Radix 所有权。

**“命中 99% 表示 99% 请求不运行模型。”** 不对。通常表示输入 token 的大部分 prefill 被跳过，新 token 仍要 decode。

**“多轮对话的第二轮是上一轮继续 decode。”** API serving 中通常不是。第二轮是带完整历史的新请求，通过 Prefix Cache 恢复旧状态。

**“每 256 token 永久保存一个 Mamba checkpoint。”** 不对。256 是默认 decode tracking interval，ping-pong slot 会复用；只有交给 Radix 的状态才长期保留到淘汰。

**“ping-pong 意味着复制两份完整 MambaPool。”** 不对。每个活动请求多占若干 state slot，所有 slot 来自同一个按容量配置的 pool。

**“有 Full KV 就能命中混合模型前缀。”** 不够。恢复点还需要有效 Mamba checkpoint，UnifiedRadixCache 取所有组件共同接受的边界。

**“把聊天记录存在数据库，就等于 KV cache 持久化。”** 不对。数据库保存文本只能支持重新构造 prompt，模型内部 tensor 是否持久化是另一套系统。

## 12. 总结

SGLang 中的 KV Cache 和 MambaPool 都是模型计算状态，区别在于前者按 token/page 增长，后者按序列位置保存固定形状的递推状态。Prefix Cache 则用 token Radix Tree 找到这些状态，让另一个请求从相同前缀继续计算。

对混合 KDA/GDN/Mamba 模型，当前 `UnifiedRadixCache` 在同一逻辑树上分别管理 Full KV 和 Mamba checkpoint。KV 可以按 token 切分，Mamba checkpoint 只能在确切的序列位置恢复，因此最终命中长度必须同时满足 token/page、线性注意力 chunk 和有效状态快照的约束。

ping-pong buffer 解决的是运行过程中如何稳定地捕获状态，Radix checkpoint 解决的是状态如何跨请求复用。默认 256 token 的跟踪间隔只是 decode 快照刷新节奏，不代表系统会为 100K 上下文永久保存近 400 份 Mamba state。真正占用 prefix cache 容量的是仍留在 Radix 节点上、尚未被淘汰的 checkpoint。
