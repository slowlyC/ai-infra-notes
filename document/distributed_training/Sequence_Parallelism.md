# 大模型分布式训练：序列并行详解

## 概述
近期长序列模型逐渐成为大模型领域的研究热点，然而序列长度的不断增长会导致大量的 Activation 显存占用，对现有的分布式训练系统提出了新的挑战。**序列并行（Context Parallelism，在一些论文中也称为 Sequence Parallelism，本文按照 Megatron-LM 的说法称为 Context Parallelism）** 是一种将输入数据按序列维度进行切分的技术，目前已成为长序列场景下训练和推理的一种有效方法。

本文将对序列并行的原理、通信开销进行详细分析，并对比 TP-SP 与 CP 的适用场景。

## 序列并行原理
当前较为主流的序列并行技术为 **Ring-Attention**[1] 和 **DeepSpeed-Ulysses**[2]，分别集成到了 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 和 [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) 框架中，本节将详细介绍这两种技术原理。

### Ring-Attention
Megatron-LM 中的 Sequence Parallelism（以下简称 SP）在张量并行的基础上，进一步将 Transformer 层中的 LayerNorm 以及 Dropout 的输入沿着序列维度进行了切分，使得各个设备上面只需要做一部分的 Dropout 和 LayerNorm 以减少激活所需的内存，但 SP 并不能切分 self-attention 模块，其需要在进行 Attention 计算前通过 all-gather 操作聚合输入序列的完整信息。

为了进一步优化长序列场景下的训练性能，Megatron-LM 提出了新的序列并行方法，称为 Context Parallelism（以下简称 CP），其主要参考了 Ring-Attention 的原理进行实现。与 SP 不同，CP 在数据输入时就沿着序列维度进行了分割，然后输入给不同的 GPU 设备，这样在 Attention 计算时每个设备也仅计算一部分的 Attention。
![图1 Ring-Attention原理示意图](https://pic3.zhimg.com/v2-23f23fbdddf08c34c42bd26e0912df98_r.jpg)
Ring-Attention 具体原理示意如图1所示。图中为 4 个 GPU 设备使用 Ring-Attention 序列并行技术计算一次 Attention 的过程，并行度大小为 4，其中关键步骤已使用红色标记标出，各步骤详细解释如下：

1. 输入数据切分：根据序列并行大小（图中为 4）对输入数据进行切分，每个设备拿到对应分片数据，此时计算得到的 QKV 矩阵亦为分片后的数据。
2. 通信 KV 矩阵数据：每个设备间通过 P2P 通信，与相邻设备交换 KV 矩阵数据，形成 Ring 状网络拓扑。以设备 1 为例，其向设备 2 发送自己的 K0V0 矩阵数据，同时接收设备 0 的 K3V3 矩阵数据。
3. 进行分块 Attention 计算：各设备分别并行计算各自分块数据的 self-attention（图中使用了 Flash-Attention 进行计算），获得单个分块的输出矩阵。
4. 循环通信 KV 矩阵与分块 Attention 计算：每个设备将接收到的 KV 矩阵发送给下一个设备，同时根据接收到的 KV 矩阵继续进行分块 Attention 计算。总共需要进行序列并行度减 1 次循环，图中序列并行度为 4，因此需要 3 次循环。
5. 分块输出矩阵计算修正：在第一次分块 Attention 计算之后，每次分块 Attention 计算都需要对输出中间值 L 进行修正，保证输出正确。这是因为 max 值都是各自分块计算的结果，需要根据前一个分块的结果进行修正，以维护全局最大的 max 值，具体原理可参考图 2 的 Flash-Attention 算法流程图或 Flash-Attention 论文[3]。
6. 计算最终输出：每个设备算完所有的分块 Attention 后，对最终结果 O 进行修正、合并。
![图2 Flash-Attention2计算流程(forward部分)，其中第3行外层循环i遍历Q矩阵，第6行内层循环j遍历KV矩阵，第9行计算全局max值，因为使用序列并行后KV矩阵是分块的，因此此处需要进行修正。](https://pic1.zhimg.com/v2-849fee7d1b24b46f1b08260e295bb410_r.jpg)
可以看出 Ring-Attention 从原理上看就是分布式的 Flash-Attention，其在实现上通过 P2P 通信与计算重叠的方式隐藏了通信开销。

### DeepSpeed-Ulysses
DeepSpeed-Ulysses 同样对输入数据按序列维度进行划分，然后输入给不同的 GPU 设备，这样每个设备生成 QKV 矩阵也相当于按序列维度进行了分割。其具体原理如图 3 所示，在 Attention 计算之前，需要对已分割的 QKV 矩阵执行 all2all 通信操作（all2all 通信等价于矩阵的分布式转置操作），收集序列维度上的信息，使得每个 GPU 设备上的 QKV 矩阵在序列维度上都是完整的，转而对 attention_head 维度根据并行度重新进行划分。之后每个设备各自进行自己的 Attention 计算，在得到输出矩阵后，再进行一次 all2all 通信，以收集 attention_head 维度上的计算结果，转而在序列维度上根据并行度再次进行划分。
![图3 DeepSpeed-Ulysses原理示意图，其对序列划分后对 QKV 矩阵进行 all2all 通信，以聚合序列维度信息；在设备各自计算 Attention 后，再次进行 all2all 通信，以聚合attention_head 维度结果，同时重新划分序列维度。](https://pic2.zhimg.com/v2-ba8ca1bfddf7e91cbd952acddd0380b3_r.jpg)
其中第一次 all2all 通信的具体内容为序列并行划分后的 QKV 矩阵，矩阵大小为 $[N/P, d]$，需要注意的是每个划分后的 QKV 矩阵都需要调用一次 all2all 通信，即总共 3 次通信。以 4 个设备，并且序列并行度为 4（即图中 P=4）为例，对 Q 矩阵进行 all2all 通信，使得每个设备得到完整的序列信息，Q 矩阵 all2all 操作前后变化如图 4 所示。
![图4 Q矩阵all2all操作前后变化示意](https://picx.zhimg.com/v2-7e7f066d01122e453c83d52b446abedb_r.jpg)
在每个设备进行 Attention 计算得到输出矩阵后，需要调用第二次 all2all 通信，从而使得每个设备得到按序列维度划分的 N/P 的结果，如图 5 所示。
![图5 Attention计算后对输出矩阵O调用all2all通信，通信后每个设备得到按序列维度划分的N/P份的结果](https://pic3.zhimg.com/v2-9ffe6101eb9f4f1c1f96daba0aa20e6a_r.jpg)
DeepSpeed-Ulysses 原理并不复杂，主要是利用 all2all 通信将输入序列与 attention_head 在各个设备间进行聚合与分散。DeepSpeed 中的代码实现也很简洁，主要是重新定义了 DistributedAttention 类，以实现对原有 Attention 模块（代码中的 local_attn）的替换，主要是在 local_attn 前调用 3 次 all2all 通信，在之后调用 1 次 all2all 通信。

```python
class DistributedAttention(torch.nn.Module):
    ...
    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
    #in shape : e.g.,  [s/p:h:]
    query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
    key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
    value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)

    #out shape : e.g., [s:h/p:]
    context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)
    output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

    #out e.g., [s/p::h]
    return output
```

每次 all2all 通信具体是调用了 PyTorch 的 all_to_all_single 通信算子。

```python
def single_all_to_all(input, scatter_idx, gather_idx, group):
    ...
    seq_world_size = dist.get_world_size(group)
    input_t = input.reshape(
        [seq_world_size, inp_shape[scatter_idx]] + \
        inp_shape[scatter_idx + 1:]
    ).contiguous()

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)
    ...
    return output.reshape(
        inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:]).contiguous()
```

DeepSpeed-Ulysses 实现起来并不复杂，但需要注意序列并行度大小必须能整除 head_num，且不能超过 head_num。

## 通信范式与 overlap
### Ring-Attention
Ring-Attention 需要使用 P2P 通信，Megatron-LM 中的主要实现是封装了 PyTorch 的 isend 和 irecv 通信算子，提供了各个 device 之间的 P2P 通信方式。接下来分析每个 device 的通信内容和通信量：

对于 GPT 模型，假设序列并行度大小为 $p$，模型 hidden_size 为 $h$，输入数据的形状为 $[b, s]$，其中 $b$ 是 batch size，$s$ 是 sequence length，则模型 Attention 层构造的 QKV 矩阵大小为 $[b, s/p, h]$。

在 Ring-Attention 中，每个 device 需要通信的内容是各自的 K 矩阵和 V 矩阵，其大小均为 $[b, s/p, h]$，则每次 P2P 通信的数据量为 $\frac{2bsh}{p}$。对于每一个 Attention 层的 Forward，总共需要进行 $p - 1$ 次 P2P 通信，总通信量为 $\frac{2bsh(p-1)}{p}$。

### DeepSpeed-Ulysses
DeepSpeed-Ulysses 需要使用 all2all 通信，DeepSpeed 中的主要实现是封装了 PyTorch 的 all_to_all_single 通信算子。接下来分析每次 all2all 的通信内容和通信量：

基于 Ring-Attention 中的 GPT 模型，进一步假设模型的 head_num 为 $a$，per_head_hidden_size 为 $d$，即 $h = a \times d$，则模型 Attention 层构造的 QKV 矩阵大小为 $[b, s/p, a, d]$。

在进行 Attention 计算前需要对 QKV 矩阵进行 3 次 all2all 通信，通信后每个矩阵的维度变为 $[b, s, a/p, d]$；Attention 计算后输出矩阵大小为 $[b, s, a/p, d]$，再次进行 all2all 通信，输出矩阵维度变为 $[b, s/p, a, d]$。因此每一层的 Forward 共需要进行 4 次 all2all 通信。对于单次 all2all 操作，每个设备发送 $p-1$ 份大小为 $\frac{bsh}{p}$ 的数据，因此单个设备单次 all2all 通信量为 $\frac{(p-1)bsh}{p}$，4 次 all2all 的总通信量为 $\frac{4(p-1)bsh}{p}$。

### overlap 优化
Ring-Attention 的每个设备在计算 local Attention 时，可以同时向下一个设备发送自己的 KV 矩阵以及接收上一个设备的 KV 矩阵，因此可以重叠计算与通信。每个设备一次 P2P 通信的数据量为 $\frac{2bsh}{p}$，而同时进行 Attention 计算的计算量为 $\frac{4bs^2h}{p^2}$ FLOPS，因此只要满足通信传输时间小于计算处理时间，完全可以通过计算与通信的 overlap，使得通信时间完全隐藏。但需要注意，在 Ring-Attention 的过程中需要分配额外的显存以缓存 P2P 通信接收到的 KV 矩阵。

DeepSpeed-Ulysses 目前没有实现 all2all 通信与计算的 overlap，因此在训练过程中会有额外的通信开销。但可以通过优化 all2all 通信算子的方式来减少这一开销。

## 框架侧如何开启序列并行
### Megatron-LM
Megatron-LM 开启序列并行需要确保 Megatron-Core (>=0.5.0) 以及 Transformer Engine (>=1.1)，在训练配置中添加以下参数，其中 cp_size 为序列并行大小：

```shell
--context-parallel-size $cp_size
```

> 注意：1）使用序列并行需要使用 megatron-core 定义的 model，即添加训练参数 `--use-mcore-models`。2）开启序列并行后数据并行的大小会相应变化，即 `data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)`。

以 llama7B 模型为例，使用一台阿里云 ecs.embgn7vx.32xlarge 机型进行单机四卡实验，设置序列并行大小为 4，即参数配置为 `context-parallel-size=4`，其中一个层的 Forward 实际 timeline 如图 6 所示：
![图6 Megatron-LM开启序列并行，一个层的Forward实际timeline，通信部分调用了p2p，且可以overlap](https://pic1.zhimg.com/v2-24c7dd6d1ad0f2c8ae74264dcbc6e1d8_r.jpg)
从图中可以看出，每一层的 Forward 总共调用了 cp_size - 1 = 3 次 P2P 通信，与理论分析相符。其中每个 P2P 通信都与计算进行了重叠，减少了通信开销。同时为了提高计算效率，Megatron-LM 创建了两个 CUDA stream 交替进行 Attention 的计算，即图中的计算流 0 和计算流 1。

### DeepSpeed
使用 DeepSpeed-Ulysses 需要 DeepSpeed 版本大于等于 v0.10.2，同时替换原模型的 Attention 模块为 DeepSpeed 的 DistributedAttention，示例如下：

```python
from deepspeed.sequence.layer import DistributedAttention

def __init__():
    ...
    self.local_attn = CoreAttention(self.layer_number, config, self.attn_mask_type)
    self.dist_attn = DistributedAttention(self.local_attn, parallel_state.get_sequence_parallel_group())
    ...

def forward():
    ...
    context_layer = self.dist_attn(query_layer, key_layer, value_layer, attention_mask)
```

Megatron-DeepSpeed 框架中提供了示例用以快速使用，只需在训练配置中添加以下参数，其中 cp_size 为序列并行大小：

```shell
ds-sequence-parallel-size $cp_size
```

以 GPT 模型为例，使用一台阿里云 ecs.embgn7vx.32xlarge 机型进行单机四卡实验，设置序列并行大小为 4，即参数配置为 `ds-sequence-parallel-size=4`，其中一个层的 Forward 实际 timeline 如图 7 所示：
![图7 Megatron-DeepSpeed开启序列并行，一个层的Forward实际timeline，通信部分调用了all2all](https://pica.zhimg.com/v2-cf0077fdad5b4e6cf535600f9d1757d8_r.jpg)
从图中可以看出，每一层的 Forward 总共调用了 4 次 all2all 通信，分别为 3 次 QKV 矩阵的通信与 1 次输出矩阵的通信，与理论分析相符。同时这 4 次 all2all 通信均没有与计算进行重叠，会产生额外的通信开销，尚存在优化空间。

### 总结
Ring-Attention 需要侵入式修改 Attention 的计算流程，集成到自定义模型中实现起来较为复杂，同时要保证每个设备的计算量负载均衡，因此对变长输入、mask 矩阵计算等问题处理起来工程量较大。而 DeepSpeed-Ulysses 集成到自定义模型中仅需对原有 Attention 模块进行封装即可，工程量较小，但其序列并行度受 head_num 约束，存在一定的限制。

假设序列并行度大小为 $p$，模型 hidden_size 为 $h$，输入数据的形状为 $[b, s]$，其中 $b$ 是 batch size，$s$ 是 sequence length，原始模型前向传播的 Activation 显存占用为 $M_b$（均使用 Flash-Attention 2），则 Ring-Attention 和 DeepSpeed-Ulysses 开销总结如表 1 所示。

| 序列并行模式 | 通信 op | 单次通信量 | 每层通信次数 | 节省 Activation 显存 | 是否可以 overlap | 是否受限 head_num | 是否支持 Zero |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ring-Attention | P2P | 2bsh/p | p-1 | (p-1)Mb/p | Y | N | 支持 Zero1、Zero2 |
| DeepSpeed-Ulysses | all2all | (p-1)bsh/p | 4 | (p-1)Mb/p | N | Y | 支持 Zero1、Zero2、Zero3 |

表1 Ring-Attention 和 DeepSpeed-Ulysses 开销总结

## TP-SP 与 CP 的对比
在 Megatron-LM 中，Sequence Parallelism（TP-SP）和 Context Parallelism（CP）虽然都涉及序列维度的切分，但设计目标和适用场景有本质区别。

**TP-SP**（Tensor Parallelism with Sequence Parallelism）：在张量并行（TP）的基础上，将 LayerNorm 和 Dropout 等非张量并行区域的输入沿序列维度切分，以减少这些区域的 Activation 显存冗余。TP-SP 并不切分 Attention 的计算，Attention 计算前仍需通过 all-gather 聚合完整序列。因此 TP-SP 的主要目的是**在 TP 已有的通信组内，进一步节省 Activation 显存**，而非解决长序列问题。

**CP**（Context Parallelism）：在数据输入时就沿序列维度进行切分，各设备仅持有部分序列，Attention 计算也以分块方式进行（Ring-Attention）或通过 all2all 聚合头维度（DeepSpeed-Ulysses）。CP 的主要目的是**在序列维度上进行并行切分，使长序列训练成为可能**。

两者的核心区别总结如下：

| 对比维度 | TP-SP | CP |
| :---: | :---: | :---: |
| 切分目标 | 在 TP 基础上切分非并行区域的 Activation | 独立切分序列维度，包括 Attention |
| 是否切分 Attention | 否（需 all-gather 聚合后完整计算） | 是（Ring-Attention 分块计算或 Ulysses 切分 head） |
| 通信组 | 复用 TP 通信组 | 独立的 CP 通信组 |
| 通信算子 | all-gather / reduce-scatter | P2P（Ring-Attention）或 all2all（Ulysses） |
| 主要收益 | 节省 Activation 显存 | 支持长序列训练，降低单卡序列长度 |
| 适用场景 | 模型参数大、需要 TP 的场景 | 序列长度大、单卡显存无法容纳完整序列的场景 |
| 是否可与对方组合 | 可以，两者正交 | 可以，两者正交 |

**如何选择 TP 和 CP：**

- **仅需 TP（无需 CP）**：序列长度适中（如 2K\~8K），单卡显存足以容纳完整序列的 Activation，但模型参数量大需要切分。此时 TP-SP 已能有效减少 Activation 显存，无需引入 CP 的额外通信开销。
- **需要 CP（可选 TP）**：序列长度很长（如 32K\~1M+），单卡显存无法容纳完整序列的 Activation（即使使用了 TP-SP），此时必须通过 CP 在序列维度上切分 Attention 计算。
- **TP + CP 组合**：大模型 + 长序列场景下的典型配置。TP 负责切分模型参数和非 Attention 区域的 Activation，CP 负责切分序列维度上的 Attention 计算。两者作用于不同的通信组，互不冲突。

### TP 与 CP 通信量分析

上面的对比更多是定性的，下面从通信量角度做定量分析，帮助判断在给定模型和序列长度下，TP 和 CP 各自的通信代价。

**符号约定**：沿用前文，$b$ 为 batch size，$s$ 为 sequence length，$h$ 为 hidden_size，$L$ 为 Transformer 层数，$t$ 为 TP 并行度，$c$ 为 CP 并行度（Ring-Attention 模式）。

**TP / TP-SP 每层通信量**

先看纯 TP（不开 SP）的情况。在 Megatron-LM 中，每个 Transformer 层的 Forward 包含 2 次 all-reduce（Attention 块的 RowParallelLinear 输出聚合 + MLP 块的 RowParallelLinear 输出聚合）。all-reduce 的消息大小为 $bsh$（完整的 $[b, s, h]$ 张量），单次 all-reduce 每个设备的通信量为 $\frac{2(t-1) \cdot bsh}{t}$（ring all-reduce 的标准公式），2 次 all-reduce 的总量为：

$$V_{TP} = \frac{4(t-1) \cdot bsh}{t}$$

开启 SP 后，all-reduce 被拆分为 all-gather + reduce-scatter（$\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}$）。每层 Forward 变为 2 次 all-gather（ColumnParallelLinear 前聚合序列维度）+ 2 次 reduce-scatter（RowParallelLinear 后分散序列维度），共 4 次操作。对于 all-gather 和 reduce-scatter，每个设备的通信量均为 $\frac{(t-1) \cdot bsh}{t}$，4 次通信的总量仍然是：

$$V_{TP\text{-}SP} = \frac{4(t-1) \cdot bsh}{t}$$

**TP-SP 的通信量与纯 TP 完全相同**，它的收益不在于减少通信量，而在于将 all-reduce 拆开后，LayerNorm 和 Dropout 可以在序列切分状态（$[b, s/t, h]$）下执行，从而节省这些区域的 Activation 显存。

当 $t$ 较大时，$V_{TP\text{-}SP}$ 近似为 $4bsh$，通信量与 $s$ 和 $h$ 均成正比，但不随并行度增加而显著增长。

**CP（Ring-Attention）每层通信量**

前文已推导，Ring-Attention 每层 Forward 共 $c-1$ 次 P2P 通信，每次通信量为 $\frac{2bsh}{c}$（发送 KV 矩阵），总量为：

$$V_{CP} = \frac{2(c-1) \cdot bsh}{c}$$

当 $c$ 较大时近似为 $2bsh$。Ring-Attention 可以将通信与计算重叠，因此实际暴露的通信时间取决于能否被计算掩盖。

**通信量对比**

| 并行方式 | 每层 Forward 通信量 | 通信算子 | 能否 overlap |
| :---: | :---: | :---: | :---: |
| TP-SP ($t$ 路) | $\frac{4(t-1) \cdot bsh}{t}$ | all-gather + reduce-scatter | 部分可以（与计算重叠有限） |
| CP-Ring ($c$ 路) | $\frac{2(c-1) \cdot bsh}{c}$ | P2P | 可以（计算与通信重叠） |

需要注意，两者的通信覆盖范围不同：TP-SP 的 4 次集合通信覆盖的是 Attention 和 MLP 的所有 Linear 层（QKV proj、O proj、MLP up/down proj），而 CP-Ring 的 P2P 通信仅覆盖 Attention 部分的 KV 矩阵传递，MLP 部分无需额外通信（每张卡独立计算自己的序列片段）。因此两者并非同一范围内的通信量对比，**当 TP 和 CP 同时使用时，两组通信是叠加关系而非替代关系**。

**GQA 对通信量的影响**

上述公式假设标准 MHA（Multi-Head Attention）。在使用 GQA（Grouped-Query Attention）的模型中，KV head 数量从 $a$ 减少为 $a_k$（通常 $a_k = a/g$，$g$ 为 group 数），这对不同并行方式的通信量影响不同：

- **TP-SP 不受 GQA 影响**：TP-SP 的 all-gather/reduce-scatter 操作的数据维度是 $[b, s, h]$，通信量由 hidden_size $h$ 决定，和 KV head 数无关。无论是 MHA 还是 GQA，TP-SP 的通信量都是 $\frac{4(t-1) \cdot bsh}{t}$。
- **CP-Ring 通信量随 GQA 显著减少**：Ring-Attention 只传递 KV 矩阵（不传 Q），GQA 下 KV 矩阵大小从 $[b, s/c, a, d]$ 变为 $[b, s/c, a_k, d]$，即 KV 的 hidden 维度从 $h$ 缩为 $h/g$。单次 P2P 通信量从 $\frac{2bsh}{c}$（K+V 各 $h$）降为 $\frac{2bsh}{gc}$（K+V 各 $h/g$），通信量缩为原来的 $1/g$。
- **CP-Ulysses 通信量部分减少**：Ulysses 的 all2all 涉及 Q、K、V 三个矩阵加输出 O，GQA 下 K 和 V 变小但 Q 不变。总通信量从 4 份 $bsh$ 降为 $2bsh + 2bsh/g$（Q 和 O 不变，K 和 V 各缩为 $1/g$）。

GQA 使得 CP（尤其是 Ring-Attention）相比 TP-SP 的通信优势进一步扩大。在 GQA 模型中（如 LLaMA-2/3、Qwen 等），CP 的通信代价更低，这也是选择 CP 而非扩大 TP 的一个有利因素。

在相同并行度下（$t = c = p$），TP-SP 的绝对通信量约为 CP-Ring 的 2 倍（MHA 下），使用 GQA 后差距更大。但考虑到 TP-SP 覆盖整层 Linear 而 CP-Ring 只需传递 KV，两者的通信效率不能简单用倍数比较。实际通信开销还取决于以下因素：

- **通信带宽差异**：TP 通常部署在机内 NVLink 互联（单向带宽 300\~450 GB/s），而 CP 如果跨机部署则受限于 RDMA 网络带宽（通常 25\~100 GB/s）。即使 TP 通信量更大，高带宽也能使实际通信时间更短。以 8 卡 NVLink（$t=8$）和 4 机 RDMA（$c=4$）为例，TP 通信时间 $\propto \frac{4bsh}{450}$，CP 通信时间 $\propto \frac{2bsh}{100}$，TP 实际更快。
- **overlap 能力**：CP（Ring-Attention）的 P2P 通信可以与 Attention 计算完全重叠。单次 P2P 传输 $\frac{2bsh}{c}$ 的数据，同时进行 $\frac{4bs^2h}{c^2}$ FLOPS 的 Attention 计算，当 $s$ 足够大时计算时间远大于通信时间，通信几乎被完全隐藏。TP 的 all-gather / reduce-scatter 则位于 Linear 层的计算路径上（all-gather 在 GEMM 前、reduce-scatter 在 GEMM 后），overlap 空间有限。
- **计算效率**：TP 将矩阵乘法沿 hidden_size 维度切分，每张卡的 GEMM 规模变为原来的 $1/t$，当 $t$ 过大时单卡矩阵变小，GPU 利用率下降。CP 切分序列维度，每张卡的 Attention 计算量变为原来的 $1/c^2$（因为 QK^T 的计算量与序列长度的平方成正比），长序列场景下计算量下降更快，同时 MLP 的计算量按 $1/c$ 下降，单卡 GEMM 规模不受影响。

### 显存切分的本质差异

通信量之外，TP 和 CP 对显存的影响有根本区别，这也是选型时最容易被忽视的因素。

TP 在切分 Activation 的同时，也将**模型权重和优化器状态**切分到了 TP 组内的各个设备上。以 $t$ 路 TP 为例，每张卡只存储 $1/t$ 的权重参数和对应的优化器状态（如 Adam 的一阶/二阶矩），权重显存直接降为 $1/t$。TP-SP 在此基础上进一步将非并行区域的 Activation（LayerNorm、Dropout 的输入输出）沿序列维度切分为 $1/t$，使得整个 Transformer 层的 Activation 显存也按 $1/t$ 缩减（参见 Megatron-LM SP 论文[5] 公式 4：$\frac{sbh}{t}(34 + 5\frac{as}{h})$）。

相比之下，CP 只切分序列维度的 Activation——每张卡持有 $s/c$ 长度的序列片段，Attention 和 MLP 的 Activation 显存按 $1/c$ 缩减。但 CP **不切分模型权重和优化器状态**，每张卡仍然持有完整的模型副本。这意味着纯 CP（不搭配 TP 或 ZeRO）的显存节省仅限于 Activation 部分。如果模型本身很大（如 70B+），单靠 CP 无法将权重和优化器放进单卡显存，必须搭配 TP 切分权重或使用 ZeRO-3 分片优化器。

这也解释了 Ulysses 论文[2]中的一个看似反直觉的实验结果：CP + ZeRO-1 的内存效率反而**低于** TP-SP，因为 ZeRO-1 只分片优化器状态而不分片权重，叠加 CP 后虽然 Activation 显存下降了，但权重显存没有变化。只有 CP + ZeRO-3（同时分片权重、梯度和优化器状态）才能达到与 TP-SP 相当的内存效率。

总结为一句话：**TP 同时切分权重和 Activation，CP 只切分 Activation**。这决定了大模型场景下 TP 几乎是必选项，而 CP 是在 TP 之上为长序列追加的扩展手段。

### TP 与 CP 选择策略

综合通信量、显存切分和 overlap 能力，实际选择时可以参考以下思路：

- **序列长度适中（$s$ ≤ 8K）**：Attention 计算量相对有限，CP 的 overlap 收益不大，而 TP 可以在机内 NVLink 高带宽下高效运行，同时切分权重和 Activation。此时优先使用 TP（开启 SP 以节省 Activation 显存），无需引入 CP。
- **序列长度较长（$s$ > 32K）且机内卡数已被 TP 占满**：单卡 Activation 显存不足，需要引入 CP。此时 Attention 计算量大，Ring-Attention 的通信可以被计算掩盖，CP 的实际通信开销很小。权重已经由 TP 切分，CP 进一步切分序列维度的 Activation。
- **超长序列（$s$ > 128K）**：CP 几乎是必选项。此时可以适当减小 TP 并行度（减少 TP 通信），将更多设备分配给 CP，因为长序列下 CP 的通信完全可以被计算隐藏。
- **跨机场景**：TP 应限制在机内（NVLink），CP 可以跨机部署。如果需要更大的并行度，优先扩展 CP 到跨机节点，而非将 TP 扩展到跨机。
- **小模型 + 长序列**（如 Diffusion 模型）：模型权重不大，单卡放得下，此时可以直接使用 CP 而不需要 TP，必要时搭配 ZeRO-3 来分片优化器状态。

总结来说，TP 同时解决权重和 Activation 的显存问题，通信在计算的关键路径上，适合在机内高带宽互联下使用；CP 只解决 Activation 的显存问题，通信可以通过计算重叠隐藏，适合在长序列场景下扩展。选择时的核心原则是：**TP 解决模型大的问题，CP 解决序列长的问题，两者在通信拓扑和显存切分维度上正交互补**。

## 扩展阅读
DeepSpeed-Ulysses 和 Ring-Attention 并不是二选一的方案，二者可以混合组成共同切分序列维度的方案，避免了 Ulysses 序列并行度必须小于 attention head 数量的限制，且混合并行的通信模式对异构网络更友好，算法流程如图 8 所示。
![图8 混合序列并行算法流程图，分为ulysses_pg和ring_pg两个序列并行通信组，流程为先对QKV矩阵进行all2all通信，之后进行RingAttention计算，再对最后的结果O进行all2all通信](https://pic1.zhimg.com/v2-1a86a82e7793ce17390b828561f2ecb6_r.jpg)
具体实验结果可以参考论文[4]。

## 参考
1. [https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)
2. [https://arxiv.org/abs/2309.14509](https://arxiv.org/abs/2309.14509)
3. [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
4. [https://arxiv.org/abs/2405.07719](https://arxiv.org/abs/2405.07719)
5. [https://arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198)
