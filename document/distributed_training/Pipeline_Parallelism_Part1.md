# 大模型分布式训练：流水线并行详解（一）

## 概述
流水线并行的朴素思想是将模型的不同层放置到不同的计算设备，降低单个计算设备的显存消耗，从而实现超大规模的模型训练，但同时流水线并行会引入额外的通信开销和设备空闲时间（Pipeline Bubble）。

本文将详细剖析 GPipe、1F1B、Interleaved 1F1B 等经典流水线并行调度方式的原理、通信范式与开销。

## 流水线并行原理
### 基础原理
流水线并行（Pipeline Parallelism）也叫 Inter-Layer Parallelism。不同于张量并行中把参数 tensor 划分到不同的设备，流水线并行是把模型按 layer 划分到不同的设备，以达到切分模型的目的。

使用流水线并行方式对模型进行切分的示例如图 1 所示，图中模型的 32 个 Layer 被切分到四个 GPU 设备上，形成流水线并行的 4 个 Stage（流水线并行中一个 Stage 对应一个 GPU 设备），对应着图中四个黄色框。
![图1. 流水线并行切分示意图](https://pic2.zhimg.com/v2-7f373a13d73bd2db67fbc84cf09693b5_r.jpg)
为了形成流水线执行的效果，即不同的设备同时执行不同的 Layer，需要把一个 batch 的数据进一步切分为 micro batch。每个设备计算一个 micro batch 后，将结果传递给下一个设备，同时开始继续计算下一个 micro batch，每个 micro batch 以近似流水线的形式在各个 Stage 间传递。流水线并行需要执行严格的 Optimizer 语义，即一个 batch 内的所有 micro batch 遇到的模型都是同一个版本。为了达到这个效果，Megatron-LM 在一个 batch 的结尾引入了流水线 flush，即做一次跨设备的同步，然后再用 Optimizer 更新参数，这种参数更新的方式也叫做 Gradient Accumulation。

### GPipe
GPipe[1] 提出了一种朴素的流水线并行实现，首先执行一个 batch 中所有 micro batch 的前向传播，然后执行所有 micro batch 的反向传播。图 2 所示为 GPipe 流水线并行调度示意图，横向对应时间，纵向代表设备（Stage）。前向传播以蓝色矩形表示，反向传播以绿色矩形表示，矩形中的数字代表 micro batch 的序号，黑色的竖线对应一个 batch 数据的 Pipeline flush。
![图2. GPipe 流水线并行调度方式](https://pic4.zhimg.com/v2-8f65c2d918187d7dc6d3e81c21b91b8d_r.jpg)
具体说明如下，图中代表一个 4 级（Stage）流水线，每个 Stage 即一个设备。一个 batch 被拆分为 8 个 micro batch，可以看到每一个 Stage 是先执行完所有 micro batch 的前向传播，然后再执行所有 micro batch 的反向传播，最后所有 Stage 在 pipeline flush 处进行一次同步。这里假设反向传播的计算量是前向传播的 2 倍，因此反向传播的耗时也假定为前向传播的两倍。

图中灰色的部分表示没有执行任何计算，即设备处于空闲状态，在流水线并行中通常被叫做 **Bubble**（流水线气泡），这些 Bubble 会降低设备的利用率。

接下来分析 GPipe 流水线并行的开销。假设一个 batch 中 micro batch 的数量为 $m$，流水线并行的 Stage 个数为 $p$，一个 micro batch 的前向传播和反向传播的执行时间分别为 $t_f$ 和 $t_b$。则上图中在前向阶段存在 $(p-1)t_f$ 的 Bubble，在反向阶段存在 $(p-1)t_b$ 的 Bubble，因此一个迭代的 Bubble 时间 $t_{pb}$ 为：

$$t_{pb} = (p-1) \cdot (t_f + t_b)$$

假设一个迭代的理想时间（即没有 Bubble 的时间）$t_{id} = m \cdot (t_f + t_b)$，则 GPipe 流水线并行中 Bubble 占比 $P_{bubble}$ 为：

$$P_{bubble} = \frac{t_{pb}}{t_{id}} = \frac{p-1}{m}$$

接下来分析 GPipe 流水线并行的显存占用。假设每个 micro batch 前向传播的 Activation 占用的显存为 $M_b$，由于每个 micro batch 前向传播的 Activation 都需要暂存到依赖它的反向计算完成才会释放，因此每个 Stage 的峰值显存 $M_{peak} = m \cdot M_b$。

在实际训练中，为了减少流水线并行的 Bubble 占比 $P_{bubble}$，往往需要 $m \gg p$，但这样会使 micro batch 数量过大，从而导致显存占用过多。

### 1F1B
为了降低 GPipe 流水线并行的显存占用，PipeDream-Flush[2] 提出了一种缩短流水线并行中 Activation 生命周期的方法，叫做 one forward pass one backward pass（简称 1F1B）。

与 GPipe 相同，假设一个 batch 中 micro batch 的数量为 $m$，流水线并行的 Stage 个数为 $p$，一个 micro batch 的前向传播和反向传播的执行时间分别为 $t_f$ 和 $t_b$。1F1B 流水线并行的调度方式如图 3 所示。与 GPipe 中每个 Stage 都驻留了 $m$ 个未完成的 micro batch 不同，1F1B 限制了每个 Stage 最多驻留 $p$ 个未完成的 micro batch，使反向传播与前向传播交替进行，从而减少了每个 Stage 驻留的前向传播 Activation 数目。
![图3 1F1B流水线并行调度方式](https://picx.zhimg.com/v2-26534e76aaa7a7bf2f2b0d64e2d72185_r.jpg)
1F1B 模式的特点是一个迭代的时间和 Bubble 占比 $P_{bubble}$ 都没有变化，即 $P_{bubble} = \frac{p-1}{m}$。但减少了每个 Stage 的峰值显存，假设当前 Stage 编号为 $i$（从 0 开始），则：

$$M_{peak} = (p - i) \cdot M_b$$

由于在实际训练中往往 $p \ll m$，所以每个 Stage 的峰值显存明显减少。

### Interleaved 1F1B
1F1B 模式降低了流水线并行的显存，但没有解决 Bubble 降低设备利用率的问题。Megatron-LM 提出了 Interleaved 1F1B[3]，可以减少流水线并行 Bubble 数目，从而用更少的时间完成一轮迭代。
![图4 Interleaved 1F1B流水线并行调度方式](https://pic4.zhimg.com/v2-56f08f8083276acb1a1adef93a664647_r.jpg)
图 4 为 Interleaved 1F1B 流水线并行调度的示意图。Interleaved 1F1B 模式是让一个 Stage 虚拟成 $v$ 个 Stage（Stage 即图中的 Device），从原本每个 Stage 计算 $l$ 个连续的 layer 段，变成计算 $v$ 个不连续的 layer 段，每段 layer 的数量为 $l/v$。如图 4 上半部分为 1F1B 调度模式，Stage 1 负责 layer 1\~4，Stage 2 负责 layer 5\~8，以此类推。而在图 4 下半部分 Interleaved 1F1B 模式中，Stage 1 负责 layer 1\~2 和 9\~10，Stage 2 负责 layer 3\~4 和 11\~12，以此类推。这样可以让流水线中每个 Stage 的单次计算时间更短，缩短下一个 Stage 的等待时间，从而减少流水线并行中的 Bubble 数量。

依然假设一个 batch 中 micro batch 的数量为 $m$，流水线并行的 Stage 个数为 $p$，一个 micro batch 的前向传播和反向传播的执行时间分别为 $t_f$ 和 $t_b$。另外假设 Interleaved 1F1B 模式的 virtual Stage 数为 $v$，则每个 Stage 流水线的 Bubble 时间为：

$$t_{pb}^{int.} = \frac{(p-1) \cdot (t_f + t_b)}{v}$$

进而得到 Interleaved 1F1B 模式的 Bubble 占比 $P_{bubble}$ 为：

$$P_{bubble} = \frac{t_{pb}^{int.}}{t_{id}} = \frac{1}{v} \cdot \frac{p-1}{m}$$

相比于 1F1B，其在每次迭代中的 Bubble 占比减少了 $v$ 倍。但需要注意，由于虚拟出了 $v$ 个 Stage，每个 Stage 上的计算量虽然没有增加，但通信量却增加了 $v$ 倍。

### BREADTH-FIRST PP（Looped BFS）
除了上述 Interleaved 1F1B（也称 Looped DFS）之外，还有一种广度优先的调度方式 Looped BFS（BREADTH-FIRST PP），其在 virtual stage 的基础上优先把所有 micro batch 的 forward 都计算完毕再计算 backward，可以更好地与数据并行梯度同步 overlap。由于篇幅原因，Looped BFS 的详细原理将在下篇文章中介绍。

## 通信范式与 overlap
### 通信范式
流水线并行需要使用 P2P 通信，Megatron-LM 中的主要实现是封装了 PyTorch 的 send 和 recv 通信算子，提供了每个 Stage 之间的双向 P2P 通信。接下来分析每个 Stage 间的通信内容和通信量：

对于 GPT 模型，假设 hidden_size 为 $h$，输入数据的形状为 $[b, s]$，其中 $b$ 是 batch size，$s$ 是 sequence length，则模型每一层输入到输出的映射为 $[b, s, h] \rightarrow [b, s, h]$。

在流水线并行中，每个 Stage 间通信的内容为 Forward 计算的中间激活以及 Backward 计算的中间激活的梯度，其 tensor 大小均为 $[b, s, h]$。若使用混合精度进行训练，通信格式为 FP16（每个元素 2 Bytes），则单次 P2P 通信的数据量为 $2bsh$ Bytes。

假设 micro batch 的数量为 $m$。在 1F1B 模式中，以中间 Stage 为例：每个 micro batch 的 Forward 阶段需要一次 recv（接收上游 Stage 的中间激活）和一次 send（发送中间激活给下游 Stage），Backward 阶段需要一次 recv（接收下游 Stage 的梯度）和一次 send（发送梯度给上游 Stage）。因此中间 Stage 每 micro batch 有 4 次 P2P 通信，每 batch 共 $4m$ 次，总通信量为 $4m \times 2bsh = 8mbsh$ Bytes。首尾 Stage 仅与一侧通信，每 batch 为 $2m$ 次，总通信量为 $4mbsh$ Bytes。Interleaved 1F1B 模式的总通信量为 1F1B 模式的 $v$ 倍。

### 计算与通信 overlap
在流水线并行中，overlap 策略主要用于减少 P2P 通信带来的计算延迟。每个 Stage 在进入下一个计算阶段之前通常需要等待前一个阶段完成其通信任务，即发送（send）与接收（receive）操作。这就导致有时候虽然计算资源可用，但因为需要等待通信完成而无法进行计算，从而造成资源浪费。
![图5 p2p通信带来的计算延迟，上半部分为理想情况下的1F1B调度，下半部分为额外的p2p通信时间带来的计算延迟](https://pic4.zhimg.com/v2-7955046d2a5f2342c66f7ba006330f8d_r.jpg)
图 5 为 P2P 通信带来的计算延迟示意图，上半部分为理想情况下的 1F1B 调度，下半部分红色箭头为假设的 P2P 通信时间。可以看出由于每个 Stage 都会使用 P2P 通信向上下游收发数据，通信延迟会大量阻塞计算，在带宽受限的场景下影响会更为严重。

以 Interleaved 1F1B 模式为例，分析每个 Stage 间的具体 P2P 通信流程与 overlap 方法，如图 6 所示：
![图6 Interleaved 1F1B模式通信流程与overlap方法](https://pic3.zhimg.com/v2-364ecda2c0aa705bf3814aa98caa42ee_r.jpg)
图 6 中上半部分为 Interleaved 1F1B 模式的原始调度方式，下半部分为 overlap 后的调度方式。在原始调度中，warmup 阶段每个 Forward 计算都需要启动 send/recv 操作，但是每个 Forward 计算实际只依赖于 recv 操作去获取前一个 Stage 的中间激活，因此可以将 send 操作解耦，让其与计算进行重叠。而在 steady 阶段，前向和后向计算都不依赖于相邻的通信操作：以 Backward 计算为例，前一个 recv 操作是为了下一个 Forward 计算而进行的，而 send 操作是为了前一个 Stage 的 Backward 计算而进行的，因此 send 操作和 recv 操作可以都异步启动，从而与计算重叠进行。

通过这种 overlap 方法可以提高流水线并行的训练效率，降低 P2P 通信引起的计算延迟问题。

## 框架侧如何开启流水线并行
目前支持流水线并行的主流大模型训练框架有英伟达的 Megatron-LM、微软的 DeepSpeed 等，本节主要介绍如何使用 Megatron-LM 开启流水线并行并展示其效果。

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 支持 1F1B 和 Interleaved 1F1B 两种流水线并行模式。Interleaved 1F1B 模式相比 1F1B 模式减少了 Bubble 占比，但总通信量增加，可以通过 overlap 方法减少通信延迟带来的影响。

### 开启 1F1B 模式
Megatron-LM 开启 1F1B 流水线并行只需在训练启动脚本加入以下参数即可，其中 PP_SIZE 为流水线并行的 Stage 数：

```shell
--pipeline-model-parallel-size $PP_SIZE
```

以 llama7B 模型为例，使用一台阿里云 ecs.embgn7vx.32xlarge 机型进行单机四卡实验，设置流水线并行数目为 4，即 `pipeline-model-parallel-size=4`，实际 timeline 如图 7 所示：
![图7 1F1B流水线并行模式4个Stage的实际timeline](https://picx.zhimg.com/v2-2919a12abbf1be45e8aba9db00fa14d1_r.jpg)
分析可知，由于实验环境为单机，因此在机内 NVLink 的高带宽场景下，P2P 通信时间很短暂，图中大段的设备空闲时间是由流水线 Bubble 引起的，这也说明了流水线并行需要尽量减少 Bubble 的占比。

### 开启 Interleaved 1F1B 模式
Megatron-LM 开启 Interleaved 1F1B 模式，需要在 1F1B 模式的基础上设置 `num-layers-per-virtual-pipeline-stage` 参数，该参数的含义是每个 virtual stage 需要处理的模型层数。PP_SIZE 为流水线并行的 Stage 数，完整参数设置如下：

```shell
--pipeline-model-parallel-size $PP_SIZE
--num-layers-per-virtual-pipeline-stage $num_layers_per_virtual_pipeline_stage
```

> 注意：开启 Interleaved 1F1B 模式，PP_SIZE 需要大于 2，同时每个 Stage 需要处理的层数必须能整除每个 virtual stage 需要处理的层数，即 `(num_layers // pipeline_model_parallel_size) % num_layers_per_virtual_pipeline_stage == 0`。

以 llama7B 模型（共有 32 个 Layer）为例，使用一台阿里云 ecs.embgn7vx.32xlarge 机型进行单机四卡实验，设置流水线并行数目为 4，每个 virtual stage 处理 2 个 Layer（即 virtual_stage = 32 / 4 / 2 = 4），则参数配置为 `pipeline-model-parallel-size=4`，`num-layers-per-virtual-pipeline-stage=2`，实际 timeline 如图 8 所示：
![图8 Interleaved 1F1B流水线并行模式4个Stage的实际timeline](https://pic1.zhimg.com/v2-1e02029994d944dfdf511656fbc67230_r.jpg)
从图中可以看出相比 1F1B 模式，Interleaved 1F1B 模式的 Bubble 占比明显减少，每个迭代的训练时间也明显缩短。但观察 timeline 也可以看出 P2P 通信的数量增多，由于实验设置了 virtual_stage=4，因此通信量增加约 4 倍，与理论分析相符，这说明 Interleaved 1F1B 模式在带宽瓶颈的训练场景下需要谨慎使用。

目前版本的 Megatron-LM 开启 Interleaved 1F1B 可以使用 overlap 方法来重叠计算与通信（1F1B 模式无法启用 overlap），效果如图 9 所示：
![图9 Interleaved 1F1B模式overlap后timeline](https://pic1.zhimg.com/v2-8c4dab1d763f8f97e4c054b58f5cfcea_r.jpg)
从图中可以看出开启 overlap 后，新增了一个 CUDA stream 用以解耦 send 操作与 recv 操作，每个 Stage 的计算与通信有所重叠。进一步放大其中一段以观察细节，如图 10 所示：
![图10 Interleaved 1F1B模式overlap细节](https://pica.zhimg.com/v2-976198e9669d7919a703e8358c374602_r.jpg)
开启 overlap 后可以重叠大部分的计算与通信，不过仍存在一小部分无法 overlap 的通信，即 warmup 阶段每个 Stage 的 recv 操作，这是因为该操作具有强依赖性。同时流水线中的 Bubble 并没有减少，因此目前业内流水线并行主要研究方向在如何减少 Bubble 占比上。

## 总结
假设一个 batch 中 micro batch 的数量为 $m$，流水线并行的 Stage 个数为 $p$，一个 micro batch 的前向传播和反向传播的执行时间分别为 $t_f$ 和 $t_b$，每个 micro batch 前向传播的 Activation 占用的显存为 $M_b$，Interleaved 1F1B 模式的 virtual stage 数为 $v$。则各个流水线并行模式的开销如表 1 所示：

| 流水线并行模式 | 显存占用 | 通信 op | 单次 P2P 通信量 (FP16) | 中间 Stage 每 batch 通信量 (FP16) | Bubble 占比 | 是否可以 overlap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPipe | m*Mb | P2P | 2bsh Bytes | 8mbsh Bytes | (p-1)/m | N |
| 1F1B | (p-i)*Mb | P2P | 2bsh Bytes | 8mbsh Bytes | (p-1)/m | N |
| Interleaved 1F1B | (1+1/v)(p-i)*Mb | P2P | 2bsh Bytes | 8vmbsh Bytes | (p-1)/(vm) | Y |

表1：各个流水线并行模式的开销汇总

由上述分析以及表 1 可知，流水线并行建议使用 Megatron-LM 框架，可以进一步结合数据并行、张量并行从而组成 3D 并行，使用 Interleaved 1F1B 模式降低 Bubble 占比，同时开启 overlap 以减少 P2P 通信时间对计算延迟的影响。

## 扩展阅读
流水线并行是大规模分布式训练的关键技术之一，从 GPipe 到 Megatron-LM 中的 Interleaved 1F1B 模式，探索了多种方法来减少流水线并行的 Bubble 占比，但 Bubble 的存在仍影响着流水线并行的训练效率。论文[4]提出反向计算可以分成两部分，一部分计算输入的梯度，另一部分计算参数的梯度。将反向梯度计算拆分为两部分后，可以分开调度，从而提出了一种新的流水线并行范式，理论上可以将 Bubble 占比降为 0。

## 参考
1. [https://arxiv.org/abs/1811.06965](https://arxiv.org/abs/1811.06965)
2. [https://arxiv.org/abs/2006.09503](https://arxiv.org/abs/2006.09503)
3. [https://arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473)
4. [https://arxiv.org/abs/2401.10241](https://arxiv.org/abs/2401.10241)
