# KDA、Gated DeltaNet-2 与 EDA：从耦合 Delta 更新到独立擦除

本文梳理 Kimi Delta Attention（KDA）、Gated DeltaNet-2（GDN2）和 Erase-then-Delta Attention（EDA）的递推公式、直觉差异与训练实现。KDA 和 GDN2 的代码分析基于 [flash-linear-attention][fla-commit] 的 `24ed9b6902990e3b18b39320f165d7395b9ebd7b` 版本；该版本没有 EDA/EKDA operator，因此 EDA 部分只分析论文公式与 chunkwise 思路，不补写不存在的代码。

相关论文：

- [Kimi Linear: An Expressive, Efficient Attention Architecture][kda-paper]
- [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention][gdn2-paper]
- [Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention][eda-paper]

如果需要先了解 Gated DeltaNet、Delta Rule 和 WY 表示，可以阅读同目录的 [GDN 原理与代码分析](./GDN_Analysis.md)。本文不重复完整的 GDN 推导，而是集中回答三个问题：KDA 改了什么，GDN2 解开了哪一层耦合，EDA 又为什么还需要一个独立擦除地址。

## 先看结论

三种方法都把历史压缩到固定大小的矩阵状态，并用 query 读取。区别在于它们如何控制旧状态的衰减、定向擦除和新内容写入。

| 方法 | 被动衰减 | 主动擦除 | 写入 | 擦除与写入的关系 |
| --- | --- | --- | --- | --- |
| KDA | key 轴逐通道的 $D_t$ | 标量 $\beta_t$，地址为 $k_t$ | 同一个 $\beta_t$，地址为 $k_t$ | 擦除强度、写入强度和地址都绑定在当前 $k_t$ |
| GDN2 | 保留 KDA 的 $D_t$ | key 轴向量 $b_t$，擦除读方向为 $b_t\odot k_t$ | value 轴向量 $w_t$，写入内容为 $w_t\odot v_t$ | 解开“擦多强”和“写多强”，主动更新仍围绕当前 $k_t$ |
| EDA | 保留逐通道的 $D_t$ | 独立地址 $e_t$ 与标量 $\gamma_t$ | 标准 Delta 更新 $k_t,v_t,\beta_t$ | 解开“在哪里擦”和“在哪里写” |

最短的理解是：

```text
KDA   ：每个 key channel 可以有不同保留率，但一次 Delta 更新仍由标量 beta 统一控制。
GDN2  ：擦除使用 b[K]，写入使用 w[V]，两边的通道强度不再相同。
EDA   ：先在独立地址 e 擦除，再在地址 k 执行标准 Delta 更新。
```

GDN2 与 EDA 不是同一种“解耦”。GDN2 解开 gate-level coupling，控制擦除和写入各有多强；EDA 解开 address-level coupling，控制擦除和写入各发生在哪里。二者在思路上可以互补，但本文讨论的论文公式和参考代码没有把它们组合成一个 operator。

## 统一记号：状态矩阵是一张线性映射

本文采用参考代码的状态布局。对单个 head：

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $q_t,k_t$ | $\mathbb{R}^{K}$ | query 和当前写入地址 |
| $v_t$ | $\mathbb{R}^{V}$ | 当前写入内容 |
| $S_t$ | $\mathbb{R}^{K\times V}$ | 固定大小的关联记忆 |
| $g_t$ | $\mathbb{R}^{K}$ | log-space 逐通道衰减 |
| $D_t=\operatorname{Diag}(\exp(g_t))$ | $\mathbb{R}^{K\times K}$ | key 轴衰减矩阵 |

输出是：

$$
o_t=q_t^\top S_t\in\mathbb{R}^{V}.
$$

将外积 $k_tv_t^\top$ 写入状态后，任意 query 的读取结果为：

$$
q_t^\top(k_tv_t^\top)=(q_t^\top k_t)v_t^\top.
$$

因此 $k_t$ 决定“写到 key 空间的哪个方向”，$v_t$ 决定“从这个方向读出什么内容”。$S_t$ 可以看成从 key/query 空间到 value 空间的线性映射。

有些论文和已有文档使用转置布局 $S_t\in\mathbb{R}^{V\times K}$，对应读取写成 $S_tq_t$、写入写成 $v_tk_t^\top$。两种记号完全等价，只要整篇推导保持一致即可。本文选择 $K\times V$，因为它与参考代码默认的 `[K, V]` state 一致。

## Delta Rule：三种方法共同的起点

先对旧状态应用衰减：

$$
\bar S_t=D_tS_{t-1}.
$$

在当前地址 $k_t$ 读取到的旧内容是：

$$
r_t=\bar S_t^\top k_t\in\mathbb{R}^{V}.
$$

标准 Delta Rule 将目标值与旧读取结果的误差写回同一个地址：

$$
S_t=\bar S_t+\beta_tk_t(v_t-r_t)^\top.
$$

展开后得到：

$$
S_t=(I-\beta_tk_tk_t^\top)\bar S_t+\beta_tk_tv_t^\top.
$$

其中：

$$
-\beta_tk_tk_t^\top\bar S_t
$$

是对当前地址旧内容的定向擦除，

$$
+\beta_tk_tv_t^\top
$$

是在同一地址写入新内容。若 $k_t$ 已 L2 归一化且 $\beta_t=1$，更新后满足 $S_t^\top k_t=v_t$。这也可以看成对局部回归损失

$$
\frac{1}{2}\lVert \bar S_t^\top k_t-v_t\rVert_2^2
$$

执行一步梯度下降。

后面三种方法都保留“状态是关联记忆”这一视角，但分别改动 $D_t$、Delta residual 内的 gate，或者擦除地址。

## KDA：把衰减从 head-wise 标量变成 key-channel 向量

### 更新公式

KDA 的单步更新为：

$$
\bar S_t=D_tS_{t-1},
$$

$$
S_t=\bar S_t+\beta_tk_t(v_t-\bar S_t^\top k_t)^\top,
$$

其中 $D_t=\operatorname{Diag}(\exp(g_t))$，$g_t\in\mathbb{R}^{K}$；$\beta_t$ 仍是每个 head 的标量。

与原始 GDN 相比，KDA 的主要变化在衰减项。原始 GDN 通常用一个 head-wise 标量同时缩放整个状态，KDA 允许 key 轴每个 channel 使用不同保留率：

```text
GDN decay：alpha_t * S
KDA decay：Diag(alpha_t[0], ..., alpha_t[K-1]) * S
```

这让不同 key feature 可以拥有不同时间尺度。某些 channel 可以快速遗忘，另一些 channel 可以长时间保留。

KDA 没有改变标准 Delta residual 的结构。$\beta_t$ 同时乘在擦除项和写入项上：

$$
S_t=\bar S_t
-\beta_tk_t(k_t^\top\bar S_t)
+\beta_tk_tv_t^\top.
$$

因此模型不能在一次更新中表达“强擦旧内容，但只弱写新内容”，也不能让不同 value channel 使用不同写入强度。

### Recurrent 代码

最直接的入口是 [`naive_recurrent_kda`][kda-naive]。忽略 batch/head 维度后，核心代码等价于：

```python
S = S * exp(g)[..., None]
old = (k[..., None] * S).sum(-2)
S = S + beta * k[..., None] * (v - old)[..., None, :]
o = (q[..., None] * S).sum(-2)
```

参考实现中的实际顺序是：

1. `S *= exp(g)`，逐 key channel 衰减。
2. `old = k^T S`，读取当前地址的旧 value。
3. `v - old`，形成 Delta residual。
4. `beta * k * (v - old)^T`，擦除并写回同一个地址。
5. `q^T S`，从更新后的状态读取输出。

推理使用的 Triton kernel 是 [`fused_recurrent_kda_fwd_kernel`][kda-recurrent]。其中第 174～177 行衰减 running state，第 179～191 行计算旧读取、应用 $\beta$，第 192～198 行完成 rank-1 更新和输出读取。

### Layer 如何生成参数

[`KimiDeltaAttention`][kda-layer] 中：

- `q_proj`、`k_proj`、`v_proj` 生成 $q,k,v$。
- `f_proj` 生成逐 key channel 的原始 decay logit。
- `b_proj` 生成每个 value head 的标量 $\beta$。
- `A_log` 与 `dt_bias` 参与 decay parameterization。

默认非 bounded gate 的 log-decay 形式为：

$$
g_t=-\exp(A_{\log})\odot\operatorname{softplus}(f_t+\text{dt\_bias}),
$$

所以 $g_t\le 0$，实际保留率 $\exp(g_t)\in(0,1]$。Layer 将 raw `g` 和 raw `beta` 传入 op，并启用 kernel 内 gate activation、$q/k$ L2 normalization 与 beta sigmoid，调用位置见 [`KimiDeltaAttention.forward`][kda-layer-forward]。

### Chunkwise 训练

逐 token recurrence 沿时间维串行，训练路径使用 chunkwise 算法。KDA 的单步转移矩阵是：

$$
A_t=(I-\beta_tk_tk_t^\top)D_t,
$$

它属于 diagonal-plus-low-rank（DPLR）结构。chunk 内多个低秩更新可以压缩成 WY 表示，再用固定大小 GEMM 和小型下三角求解替代逐 token 外积循环。

[`chunk_kda_fwd`][kda-chunk-forward] 的高层流程是：

```text
g 的 chunk-local cumsum
    → 构造 intra-chunk Aqk、Akk 和 WY 辅助量 w/u
    → chunk 间状态递推
    → 组合 inter-chunk 与 intra-chunk 输出
```

代码对应：

- `kda_gate_chunk_cumsum` 或 `chunk_local_cumsum` 计算 chunk 内累计衰减。
- `chunk_kda_fwd_intra` 构造 `Aqk`、`Akk` 与 WY 辅助量。
- `chunk_gated_delta_rule_fwd_h` 完成 chunk 间状态递推。
- `chunk_gla_fwd_o_gk` 组合历史状态贡献和 chunk 内 causal 贡献。

这里的局部变量 `w` 是 **WY 表示中的辅助量**，不是 GDN2 的 write gate，也不是 EDA 的 erase address。后文会再次区分这三个含义。

## GDN2：把标量 beta 拆成 key-side erase gate 和 value-side write gate

### KDA 还保留了什么耦合

KDA 中同一个标量 $\beta_t$ 控制：

$$
\text{erase strength}=\beta_t,
$$

$$
\text{write strength}=\beta_t.
$$

但擦除与写入作用在状态矩阵的不同轴上：

- 擦除通过 key 轴读取旧内容。
- 写入决定 value 轴的哪些内容进入状态。

使用同一个标量意味着所有 key/value channel 只能一起强或一起弱。GDN2 将它们改成两个向量 gate。

### 更新公式

GDN2 保留 KDA 的逐通道 decay：

$$
\bar S_t=D_tS_{t-1}.
$$

然后定义：

$$
r_t=\bar S_t^\top(b_t\odot k_t),\qquad b_t\in[0,1]^K,
$$

$$
u_t=w_t\odot v_t,\qquad w_t\in[0,1]^V.
$$

状态更新为：

$$
S_t=\bar S_t+k_t(u_t-r_t)^\top.
$$

展开后是论文和参考代码使用的形式：

$$
S_t=(I-k_t(b_t\odot k_t)^\top)D_tS_{t-1}
+k_t(w_t\odot v_t)^\top.
$$

$b_t$ 对 $k_t$ 的每个 key coordinate 施加不同擦除权重；$w_t$ 对 $v_t$ 的每个 value coordinate 施加不同写入权重。两边不再共享同一个标量。

当：

$$
b_t=\beta_t\mathbf{1}_K,
\qquad
w_t=\beta_t\mathbf{1}_V,
$$

GDN2 精确退化为 KDA：

$$
S_t=\bar S_t+\beta_tk_t(v_t-\bar S_t^\top k_t)^\top.
$$

若再把 $D_t$ 从逐通道对角矩阵退化为 head-wise 标量 $\alpha_tI$，就得到原始 GDN 的更新形式。

### GDN2 没有引入独立擦除地址

$b_t\odot k_t$ 可以改变读取旧内容时各 key coordinate 的权重，但它仍由当前写入 key $k_t$ 构造，rank-1 更新的左因子也仍是 $k_t$。因此 GDN2 解开的是擦除和写入的**通道强度**，不是两个完全独立的地址。

这一区别可以压缩成：

```text
KDA  ：beta 标量决定擦除和写入。
GDN2 ：b[K] 决定如何读取并擦除旧内容，w[V] 决定写入 v 的哪些通道。
       两个操作仍围绕当前 k 组织。
```

### Recurrent 代码

[`naive_recurrent_gdn2`][gdn2-naive] 是公式的逐行翻译：

```python
S = S * exp(g)[..., None]
erase = ((b * k)[..., None] * S).sum(-2)
v_new = w * v - erase
S = S + k[..., None] * v_new[..., None, :]
o = (q[..., None] * S).sum(-2)
```

与 KDA recurrent 对照，状态 shape、衰减、rank-1 写回和 query 读取都没有变化。变化集中在 residual：

```python
# KDA
v_new = beta * (v - k.T @ S)

# GDN2
v_new = w * v - (b * k).T @ S
```

Triton decode kernel 中的对应代码位于 [`fused_recurrent_gdn2_fwd_kernel`][gdn2-recurrent]：第 199～206 行加载 `b` 并计算 `(b * k)^T S`，第 208～209 行计算 `w * v - erase`，第 211～217 行沿 `k` 写回状态并读取输出。

### Layer 如何生成 b 和 w

[`GatedDeltaNet2`][gdn2-layer] 保留 KDA 的 $q/k/v$ 与 decay 分支，并新增两个独立线性投影：

```python
self.b_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
self.w_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
```

forward 中分别经过 sigmoid：

```python
b = self.b_proj(hidden_states).sigmoid()  # [..., K]
w = self.w_proj(hidden_states).sigmoid()  # [..., V]
```

然后 reshape 成 per-head 张量并传给 `chunk_gdn2` 或 `fused_recurrent_gdn2`，见 [`GatedDeltaNet2.forward`][gdn2-layer-forward]。

### GDN2 的 chunkwise 改动

GDN2 的转移矩阵是非对称 rank-1 修正：

$$
A_t=(I-k_t(b_t\odot k_t)^\top)D_t.
$$

它仍可使用与 KDA 相同的高层 WY/chunk 结构，但 erase-side 和 write-side 辅助量不再共享标量 $\beta_t$。

参考 PyTorch chunk 实现 [`naive_chunk_gdn2`][gdn2-naive-chunk] 给出了最清楚的对应关系：

$$
K'_t=(b_t\odot k_t)\odot\exp(g_t),
$$

$$
U=A(w\odot V),
\qquad
W_{\mathrm{WY}}=AK'.
$$

代码中：

```python
k_g_b = k * exp(g) * b
u_wy = A_inv @ (w * v)
w_wy = A_inv @ k_g_b
```

训练主流程 [`chunk_gdn2_fwd`][gdn2-chunk-forward] 仍然是：

```text
decay cumsum
    → GDN2 intra/WY
    → 共用的 chunk_gated_delta_rule_fwd_h
    → 共用的 chunk_gla_fwd_o_gk
```

真正不同的部分在 intra/WY 构造：

- [`chunk_gdn2_fwd_kernel_intra_sub_chunk`][gdn2-chunk-intra] 使用 `b * k` 构造 `Akk`。
- [`recompute_w_u_fwd_gdn2_kernel`][gdn2-wy] 使用 `w_gate * v` 构造 write-side `u`，使用 `b * k * exp(g)` 构造 erase-side `w_wy`。
- backward 不能再把标量 $\beta$ 提到 reduction 外部，必须在 key/value 对应的梯度累积位置保留 $b$ 和 $w$。

为避免命名冲突，GDN2 的 chunk forward 将 value-side write gate 命名为 `w_gate`，WY 辅助量命名为 `w_wy`。

## EDA：增加独立擦除地址，再执行标准 Delta 更新

### GDN2 之后仍存在的限制

GDN2 可以分别控制 key-side 擦除通道与 value-side 写入通道，但主动编辑仍围绕当前 $k_t$。如果过期信息位于另一个地址，模型只能等待逐通道 decay 慢慢削弱它，或者通过后续与该地址对齐的写入间接覆盖。

EDA 处理的是这一层 address coupling。它不修改标准 Delta 写入，而是在 Delta 之前增加一次独立地址的定向擦除。

### 更新公式

EDA 先应用逐通道衰减：

$$
\bar S_t=D_tS_{t-1}.
$$

然后使用独立的 erase address $e_t\in\mathbb{R}^{K}$ 和 erase gate $\gamma_t$：

$$
\widetilde S_t=(I-\gamma_te_te_t^\top)\bar S_t.
$$

最后在擦除后的状态上执行不变的标准 Delta Rule：

$$
S_t=(I-\beta_tk_tk_t^\top)\widetilde S_t+\beta_tk_tv_t^\top.
$$

合并为：

$$
S_t=(I-\beta_tk_tk_t^\top)
(I-\gamma_te_te_t^\top)
D_tS_{t-1}
+\beta_tk_tv_t^\top.
$$

执行顺序从右向左：

```text
逐通道 decay
    → 在 e 地址擦除
    → 在 k 地址做 Delta correction/write
    → 用 q 读取
```

写成接近 recurrent kernel 的形式：

```python
S *= exp(g)[:, None]
S -= gamma * e[:, None] * (e @ S)[None, :]
S += beta * k[:, None] * (v - k @ S)[None, :]
o = q @ S
```

这里的第二个更新仍包含标准 Delta Rule 自带的“在 $k$ 地址擦旧值再写新值”。EDA 不是用独立擦除替换 Delta 擦除，而是在它之前增加另一条 cleanup path。

### e 和 gamma 从哪里来

$e_t$ 和 $\gamma_t$ 不是存放在 recurrence 里的静态参数，而是由当前 token 的 hidden state 经过可训练投影生成的逐 token 激活。论文中的结构可以概括为：

$$
e_t=\operatorname{L2Norm}(\operatorname{Proj}_e(x_t)),
\qquad
\gamma_t=\sigma(\operatorname{Proj}_\gamma(x_t)).
$$

`Proj_e` 的投影矩阵和 `Proj_gamma` 的投影矩阵是模型参数；$e_t$ 与 $\gamma_t$ 是每个 token 动态计算的结果。论文对 erase address 使用低秩投影以控制参数量，最终仍生成 $K$ 维并经过 L2 normalization 的地址向量。

因此 EDA 学到的是两件事：当前 token 应该指向哪个旧记忆方向，以及这次清理应该有多强。$\gamma_t\approx0$ 时，额外擦除路径基本关闭。

### 独立擦除如何影响读取

只看 EDA 的额外擦除步骤，任意 query $q$ 的读出变成：

$$
q^\top\widetilde S
=q^\top\bar S
-\gamma(q^\top e)e^\top\bar S.
$$

若 $e$ 已归一化：

- 当 $q=e$ 时，沿 erase address 读出的旧内容缩放为原来的 $1-\gamma$。
- 当 $q\perp e$ 时，这一步在后续 Delta 更新之前不会改变该 query 的读出。

因此它是一个 rank-1 的软擦除，不是从离散字典中无条件删除一条记录。若多个语义关联在向量空间中发生叠加，错误选择 $e$ 仍可能损伤有用内容；$\gamma$ 允许模型把这条路径关小。

[EDA 论文][eda-paper]的状态分析中，erase address 与 write key 没有退化成同一个方向，报告的平均 $|\cos(e_t,k_t)|$ 约为 0.105。论文同时观察到独立擦除会降低 raw write-key recall，因此将它定位为按需启用的 cleanup mechanism，而不是无代价地提高所有历史记忆的保真度。

### 一个更合适的直觉例子

离散字典只用于帮助理解，不能等同于神经网络内部状态。假设记忆以“位置”为地址：

```text
树上 → 鸟
屋顶 → 空
```

当前信息是“鸟从树上飞到了屋顶”，需要：

```text
擦除：树上 → 鸟
写入：屋顶 → 鸟
```

EDA 可以令 $e_t$ 对应“树上”，$k_t$ 对应“屋顶”，$v_t$ 对应“鸟”。KDA 和 GDN2 的单个 head、单个 token 更新都围绕当前 $k_t$ 组织，没有直接的“在 $e_t$ 删除，同时在另一个 $k_t$ 写入”操作。

如果字典改为以“动物”为地址：

```text
鸟 → 在树上
```

更新成“鸟 → 飞走了”仍是同一地址替换，标准 Delta Rule 已经可以完成。这种例子不能说明 EDA 的独立地址能力。

### EDA 的 chunkwise 思路

EDA 每个原始 token 有两个连续的 rank-1 修正：erase factor 和 delta factor。论文为了复用已有 DPLR chunk 算法，将长度 $T$ 的序列展开成长度 $2T$ 的伪序列：

$$
(q'_{2t-1},k'_{2t-1},v'_{2t-1},\beta'_{2t-1},D'_{2t-1})
=(0,e_t,0,\gamma_t,D_t),
$$

$$
(q'_{2t},k'_{2t},v'_{2t},\beta'_{2t},D'_{2t})
=(q_t,k_t,v_t,\beta_t,I).
$$

奇数子步只执行 decay 和 erase，偶数子步执行标准 Delta 更新并输出。这样 EDA 可以归约为已有 DPLR recurrence，但每个原始 token 对应两个低秩子步。

本文参考的 `flash-linear-attention` 版本没有 `fla/ops/eda` 或 `fla/ops/ekda`，因此这里不把论文伪序列写成仓库代码，也不借用其他代码库的实现。

## 三者放在同一条演进线上

### 从 GDN 到 KDA

原始 GDN 的衰减通常是每个 head 一个标量。KDA 将它变成 key 轴逐通道向量：

$$
\alpha_tI
\quad\longrightarrow\quad
D_t=\operatorname{Diag}(\alpha_t).
$$

改动发生在“旧状态如何自然遗忘”，Delta residual 仍由标量 $\beta_t$ 控制。

### 从 KDA 到 GDN2

KDA 的 $\beta_t$ 同时控制旧读取的擦除和新 value 的写入。GDN2 将它拆到矩阵的两个轴：

$$
\beta_t
\quad\longrightarrow\quad
b_t\in\mathbb{R}^{K},\quad w_t\in\mathbb{R}^{V}.
$$

改动发生在“擦多少、写多少”，当前 key $k_t$ 仍是主动编辑的中心。

### 从同地址编辑到 EDA

EDA 保留标准 Delta 写入，并增加独立擦除地址：

$$
\text{same-address correction at }k_t
\quad\longrightarrow\quad
\text{erase at }e_t\text{, then correct/write at }k_t.
$$

改动发生在“在哪里擦”。它与 GDN2 的 channel-wise erase/write gate 属于不同维度的扩展。

可以用下面的关系概括：

```text
GDN
  └─ 衰减从 head-wise scalar 变成 key-channel vector → KDA
       ├─ beta 拆成 b[K] 与 w[V]                  → GDN2
       └─ 增加独立 erase address e，再做 Delta    → EDA
```

这张关系图表达的是改动位置，不表示 GDN2 和 EDA 互相包含。GDN2 可以退化到 KDA；EDA 在关闭额外擦除路径（$\gamma_t=0$）时回到其 gated-delta baseline。

## 代码中的符号冲突

三套推导和实现都可能出现字母 `w`，但含义不同。

| 出现场景 | `w` 的含义 | 形状 |
| --- | --- | --- |
| KDA chunk/WY | WY 紧凑表示中的 erase-side 辅助矩阵 | `[B, T, H, K]` 或 chunked 等价布局 |
| GDN2 公式与 op 输入 | value-side write gate，代码中部分位置写作 `w_gate` | `[B, T, H, V]` |
| EDA 的部分工程实现习惯 | 可能用 `w` 表示论文中的 erase address $e$ | `[B, T, H, K]` |

阅读代码时应根据 shape 判断语义。GDN2 的 `w_gate * v` 是逐 value channel 写入门，不是擦除地址；`w_wy` 则是 WY 辅助量，也不是模型投影产生的新 gate。

## 参考代码阅读顺序

建议按下面顺序阅读，每一步只增加一种复杂度。

### KDA

```text
fla/ops/kda/naive.py
    → fla/ops/kda/fused_recurrent.py
    → fla/layers/kda.py
    → fla/ops/kda/chunk_fwd.py
    → fla/ops/kda/chunk_intra.py / wy_fast.py
    → tests/ops/test_kda.py
```

- [`naive_recurrent_kda`][kda-naive]：确认公式和 state shape。
- [`fused_recurrent_kda_fwd_kernel`][kda-recurrent]：确认 decode 中的实际更新顺序。
- [`KimiDeltaAttention`][kda-layer]：确认 $q/k/v/g/\beta$ 从哪里生成。
- [`chunk_kda_fwd`][kda-chunk-forward]：理解训练路径如何拆成 gate、intra/WY、state、output。
- [`test_kda.py`][kda-tests]：查看 naive、chunk、recurrent 的一致性覆盖。

### GDN2

```text
fla/ops/gdn2/naive.py
    → fla/ops/gdn2/fused_recurrent.py
    → fla/layers/gdn2.py
    → fla/ops/gdn2/chunk_fwd.py
    → fla/ops/gdn2/chunk_intra.py / wy_fast.py
    → tests/ops/test_gdn2.py
```

- [`naive_recurrent_gdn2`][gdn2-naive]：观察 residual 如何从标量 $\beta$ 变成 $b/w$ 两个向量。
- [`fused_recurrent_gdn2_fwd_kernel`][gdn2-recurrent]：观察 `(b*k)^T S` 与 `w*v`。
- [`GatedDeltaNet2`][gdn2-layer]：确认独立 `b_proj`、`w_proj`。
- [`chunk_gdn2_fwd`][gdn2-chunk-forward]：确认高层训练结构与 KDA 相同。
- [`wy_fast.py`][gdn2-wy]：确认 `w_gate*v` 和 `b*k*exp(g)` 分别进入 write/erase 辅助量。
- [`test_gdn2.py`][gdn2-tests]：查看 forward、backward、varlen、gate-in-kernel 和 layer 级覆盖。

## 总结

KDA、GDN2 和 EDA 都在解决固定大小线性记忆中的干扰，但扩展位置不同。

KDA 改进长期保留策略。$D_t$ 在 key 轴逐通道衰减，使不同 feature 拥有不同时间尺度；当前地址的主动编辑仍是标量 $\beta_t$ 控制的标准 Delta Rule。

GDN2 改进同一次主动编辑内部的自由度。$b_t$ 决定 key-side 如何读取并擦除旧内容，$w_t$ 决定 value-side 写入哪些新内容；它解开强度与通道结构，但没有引入完全独立于 $k_t$ 的擦除地址。

EDA 改进地址管理。它先用 $e_t,\gamma_t$ 清理一个独立方向，再在 $k_t$ 上执行原有 Delta correction/write，因此可以表达“在一处清理，在另一处写入”。代价是每个 token 多一个 rank-1 状态修正，错误擦除也可能损伤叠加在同一向量空间中的有用信息。

从代码角度，理解三者最可靠的入口仍是 recurrent reference。先看每个 token 如何更新 $S$，再看 chunkwise 算法如何用累计 decay、WY 表示和 GEMM 重排同一组数学操作，避免一开始被 `Aqk`、`Akk`、`w_wy`、`u_wy` 等辅助量遮住主线。
