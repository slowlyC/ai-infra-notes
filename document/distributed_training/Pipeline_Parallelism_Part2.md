# 大模型分布式训练：流水线并行详解（二）

## 概述
前文已介绍 GPipe、1F1B、Interleaved 1F1B 等经典流水线并行调度方式，本文在此基础上重点分析 Looped BFS、Zero Bubble、DeepSeek v3 的 DualPipe 等前沿流水线并行方案的原理与开销。

## Looped BFS
在 LLama 3.1 中提到了深度优先调度（DFS）流水线和广度优先调度（BFS）流水线，这个概念首先在论文[1]中提出，如图 1 所示。图中（a）是 GPipe 调度，（b）是 1F1B 调度，（c）是 Looped DFS 流水线（即上篇文章介绍的 Interleaved 1F1B），（d）是 Looped BFS 流水线。
![](https://pica.zhimg.com/v2-699ed6b3bc875ae802a754373365c7be_r.jpg)
图1 GPipe、1F1B、Looped DFS、Looped BFS 流水线示意

观察发现，Looped DFS 流水线其实就是 1F1B 加上了 virtual Stage，而 Looped BFS 则是 GPipe 加上了 virtual Stage。上篇文章提到，由于要计算完所有 micro batch 才开始进行反向传播，GPipe 的峰值显存占用高于 1F1B。类似地，Looped BFS 流水线的峰值显存也高于 Looped DFS。

那么 Looped BFS 流水线的优势是什么？其主要优势体现在数据并行场景下。假设 PP 设备数为 4，micro batch 数量为 8，virtual Stage 为 4，在 Looped DFS 和 Looped BFS 的每个 virtual Stage 的最后一个 micro batch 结束时，需要进行数据并行上的梯度同步，总共需要进行 virtual Stage 次梯度同步。由于 Looped BFS 每次先计算完每个 virtual Stage 所有的 micro batch，因此其**开启梯度同步的时间要早于 Looped DFS**，以此类推最后一次梯度同步结束的时间也早于 Looped DFS。在数据并行 Zero-2 的场景下，由于梯度累积的原因，每个 micro batch 在完成反向传播后都要进行一次梯度同步，通信量会大大增加，此时 Looped BFS 的优势会进一步放大。

接下来介绍 LLama 3.1 中的流水线并行方案，如图 2 所示。假设流水线设备数是 P，微批次的总数是 M，同一阶段连续前向或反向微批次的数量是 N。在 Looped DFS 流水线中需要 N = P = 4，而 Looped BFS 流水线中需要 N = M。然而在预训练中通常需要灵活调整批处理大小，因此 LLama 3.1 将 N 设置为一个可调变量，在 Looped DFS 和 Looped BFS 之间寻找通信效率和显存效率的最佳平衡点。
![](https://pic2.zhimg.com/v2-34209606001d63ddd17d3150c9856851_r.jpg)
图2 LLama 3.1 流水线示意

## Zero Bubble
### B/W 分割
在一个训练流程中，分为前向传播 F 和反向传播 B。在大部分训练框架中，这两者都是作为单一的函数提供给上层接口使用的。以下是 Megatron-LM 中 backward 方法的关键代码：

```python

    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

...

        if ctx.gradient_accumulation_fusion:
            ...
        else:
            grad_weight = grad_output.t().matmul(total_input)
```

代码中 `grad_input` 和 `grad_weight` 分别是计算输入的梯度和参数的梯度。由于在反向传播中要计算两个矩阵乘法，因此反向传播的时间通常约等于前向传播的两倍。从代码中可以看出两者间没有相互依赖，可以独立计算。这种设计在数据并行时工作良好，因为在进行第 i 层参数梯度的 allreduce 通信时可以与第 i-1 层输入的梯度计算重叠。但是在流水线并行中，却增加了计算的顺序依赖，因为第 i-1 层的反向传播需要等待第 i 层的反向传播结束后才可以进行，即第 i-1 层的 grad_input 隐性依赖于第 i 层的 grad_weight，这通常对流水线并行的效率非常不利。

因此可以考虑对 grad_input 和 grad_weight 分开计算，如图 3 所示。从左到右，使用 F 表示前向传播，B 和 W 分别表示 grad_input 和 grad_weight 的计算。将反向传播分割为 B 和 W 两个阶段后，仍然需要确保同一个 micro batch 中的 F 和 B 保持顺序依赖（因为 B 依赖于 F），然而同一阶段的 W 可以在对应的 B 之后的任何位置灵活安排。这就允许策略性地放置 W 来填充 Pipeline 的气泡，从而大幅提高流水线并行的效率。
![](https://pic2.zhimg.com/v2-09bde341695a5cc4fd1aad3f3c8e1ec5_r.jpg)
图3 B/W 分割示意

需要注意的一个细节是，**在实际训练中，B 和 W 的耗时并不完全相同**，表 1 所示为三个算子的具体 FLOPs。因此 B/W 分割会带来额外的流水线负载不均衡。
![](https://picx.zhimg.com/v2-77120703adabc2a869c12d2375cc48bf_r.jpg)
表1 F、B、W 的 FLOPs 与激活显存需求

### ZB-H1/H2
基于 B/W 分割，论文[3]提出的第一个流水线并行调度为 ZB-H1。图 4 为 1F1B 流水线，图 5 为 ZB-H1 流水线。ZB-H1 在调度上总体遵循 1F1B 的调度方式，每个设备 micro batch 的数量与 1F1B 相同，不同的是其根据 warmup 的 micro batch 数量调整了 W 的开始点：在最后一个设备计算完 B 后，不是继续计算 W，而是计算下一个 micro batch 的 F。对比第一个 Stage 即可发现，**ZB-H1 的 Bubble 减少到了 1F1B 的三分之一**，这是因为从倒数第二个设备开始更早地启动了 B，并且每个设备尾端的 Bubble 被较晚启动的 W 填充。由于流水线并行的峰值显存占用主要取决于第一个设备，而 ZB-H1 和 1F1B 第一个设备的 warmup 数量以及 B/W 的计算顺序完全相同，因此两个流水线的峰值显存占用完全一致。
![](https://pic2.zhimg.com/v2-a42c2ffb9b7efd86e5dfdcc863413375_r.jpg)
图4 1F1B 流水线
![](https://pica.zhimg.com/v2-af0d8815217b35db15a9e58b4b24acec_r.jpg)
图5 ZB-H1 流水线

> 需要注意的是，在第一个设备上显存占用与 1F1B 一致，但在其他设备上显存占用会增加。这是因为 1F1B 计算反向传播是一个整体方法，计算完成后前向传播的激活就被释放了；但 ZB-H1 由于分割了 B/W，在进行 B 计算后需要保留一部分激活用于计算 W。不过由于流水线的峰值显存占用仅取决于第一个设备，因此对训练过程影响不大。

当允许比 1F1B 更大的内存占用，并且有足够数量的 micro batch 时，可以实现一个近乎零气泡的流水线调度，称为 ZB-H2。如图 6 所示，在 steady 阶段添加更多的 F 来填补第一个 B 之前的气泡，同时在流水线尾部重新排序 W，将整个流水线的形状变为平行四边形，从而消除流水线中的所有气泡。ZB-H2 中 steady 阶段填充的气泡会在最后设备的 cooldown 阶段出现，因此需要移除优化器步骤之间的同步，详细介绍可以参考论文[3]第 4 小节，本文不再赘述。
![](https://picx.zhimg.com/v2-e4513ec943aeff675664531be8a4ac2d_r.jpg)
图6 ZB-H2 流水线

可以看出 ZB-H2 虽然可以实现零气泡，但流水线的显存占用却翻倍了，这是因为第一个设备需要更多的 warmup micro batch 来填充气泡。

### ZB-V
那么如何在与 1F1B 相同显存的情况下实现零气泡？论文提出了一种新的 micro batch 排布方式，如图 7 所示。左边为 virtual stage 数量为 2 的 Interleaved 1F1B 排布方式（称为 Parallel），右边为提出的 V-Shape 排布方式。观察两种排布中第一个设备的显存使用情况：在 Parallel 中，virtual stage 1 的 mbs0 的激活在显存中的生命周期为 l1，virtual stage 2 的 mbs0 的激活在显存中的生命周期为 l4，之后被逐渐消耗，因此显存峰值约等于 l1 + l4；同理，在 V-Shape 中，显存峰值约等于 l1 + l6。由于在流水线并行中，显存的最大限制往往出现在第一个设备，因此 V-Shape 可以有效降低显存占用。另外观察 V-Shape 中不同设备的显存占用，可以发现各设备间显存占用基本一致，因此理论上 V-Shape 还可以实现各设备的峰值显存均衡。
![](https://pica.zhimg.com/v2-8b5d1d01979de625938f3c5d4181e2b6_r.jpg)
图7 V-Shape 调度示意

基于 V-Shape 的排布方式，对 ZB-H2 重新调度排布，可以得到 ZB-V 的流水线调度，如图 8 所示。ZB-V 在实现了 ZB-H2 零气泡的同时，仅使用了与 ZB-H1 相当（相对 ZB-H2 减半）的显存占用。
![](https://pic2.zhimg.com/v2-3a5b2613a80a8ceaf45b00c3aaf33f49_r.jpg)
图8 ZB-V 流水线

## Chimera
DeepSeek v3 的 DualPipe 流水线使用了双向流水线调度的形式，其借鉴了 Chimera[4] 流水线调度。如图 9 所示，Chimera 将模型参数复制为两份 replica0 和 replica1，分别从第一个设备和最后一个设备同时开始进行流水线调度，即 down pipeline 和 up pipeline，由此构成了双向的流水线调度。

Chimera 流水线相比于 1F1B 流水线，Bubble 的比例更小，但代价是需要保存两份模型参数，因此峰值显存需求大幅增加，同时还需要额外的 allreduce 来同步双向 pipeline 中的梯度。
![](https://picx.zhimg.com/v2-8e98311adb04e0a05e560dd2d12f79f7_r.jpg)
图9 Chimera 流水线

## DualPipe
DeepSeek v3 是一个 MoE 模型，其计算过程主要为：1）attention 层的计算；2）all-to-all dispatch；3）MoE 计算（即 MLP 层的计算）；4）all-to-all combine。其中 all-to-all dispatch 和 all-to-all combine 通信会耗费大量的时间，那么如何实现 all-to-all 通信与计算的 overlap？DeepSeek v3 巧妙利用了 Chimera 流水线双向 pipeline 的思路，将模型参数复制为两份，让**一个 micro batch 在进行计算的同时，另一个 micro batch 进行通信**，从而实现计算与通信的 overlap。

计算流与通信流的具体调度细节如图 10 所示。图中实心三角形和空心三角形分别代表两个模型 replica，橘色方块代表 forward，绿色方块代表输入的梯度 grad_input，蓝色方块代表参数的梯度 grad_weight，这与 Zero Bubble 的定义形式相同。红色箭头为 Forward 的数据流向，绿色箭头为 Backward 的数据流向。
![](https://pic2.zhimg.com/v2-1d63bf21c0ab275752cf33ca0ef5443f_r.jpg)
图10 DeepSeek v3 计算与通信 overlap 示意

根据上述思路，DeepSeek v3 设计了 DualPipe 流水线。为了说明 DualPipe 流水线的具体设计思路，本文将其调度原理一步步进行拆解。

首先加入 micro batch 0 和 10 的双向 pipeline，在最后一个 Stage 立即开始反向传播。
![](https://pic3.zhimg.com/v2-f2221371bd687464081889dce4baeb36_r.jpg)
图11

同理，加入 1 和 11：
![](https://pic4.zhimg.com/v2-81bb06d8b9b897163bb0ff920b47fc59_r.jpg)
图12

之后加入 2 和 12：
![](https://pic1.zhimg.com/v2-0680eb8b6b8c0f799368bda3ca27c902_r.jpg)
图13

加入 3 和 13：
![](https://pic4.zhimg.com/v2-96acbc4eb2d203def47b1f0d22946749_r.jpg)
图14

加入 4 和 14。重点来了：在 dev3 加入 4 时，正好同时开始计算另一条 pipeline 的 10B，此时可以进行 overlap，如图中红色方框所示，合并为 4|10 计算；dev4 合并 14|0 计算。
![](https://pic1.zhimg.com/v2-b75548ddc92651882ab754a2b99413a6_r.jpg)
图15

补充完 4 和 14：
![](https://pic3.zhimg.com/v2-c5edd2edcf4cbc92e780cc691c39826a_r.jpg)
图16

同样加入 5 和 15，dev2 合并 5|11，dev5 合并 15|1，灰色区域是流水线 Bubble。
![](https://pic4.zhimg.com/v2-a295ed58dd93b6cc4f05dc8fc2876b45_r.jpg)
图17

加入 6 和 16：
![](https://picx.zhimg.com/v2-86134a79d973174e329dec1693332cbd_r.jpg)
图18

注意之前的 micro batch 13 没有立即开始 backward，原因就在这里——它可以和 7 合并，组成 7|13。
![](https://pica.zhimg.com/v2-a557b06597de9e139b6081689ed88cc0_r.jpg)
图19

接着加入 8 和 18。这里 8 没有立即放到 13B 的后面，因为可以在后面和 14B 合并，因此多出来的是 Bubble。
![](https://pic2.zhimg.com/v2-307233d7db811152f133f69743b83ddf_r.jpg)
图20

同理加入 9 和 19：
![](https://pic4.zhimg.com/v2-d44b29076e9805f549ec9ecd6a1c5241_r.jpg)
图21

在 9 和 19 的 forward 完成的同时，5 和 15 的 backward 也完成了。之后需要完成 6\~9 以及 16\~19 的 backward，完整的 DualPipe 流水线参考图 22。需要注意，在结束的时候把 B 和 W 拆开来调度了，思路和 Zero Bubble 类似——让 B 尽早调度，用 W 来填充 Bubble。
![](https://pica.zhimg.com/v2-6eec47eab7a0747f87fc4cb58f6a0918_r.jpg)
图22 完整的 DualPipe 流水线

可以看出 DualPipe 流水线既能 overlap all-to-all 以及 PP 的通信，还能减少 Bubble，但缺点是需要双倍的模型参数以及更加复杂的系统设计。

## 总结
假设一个 batch 中 micro batch 的数量为 $m$，流水线并行的 Stage 个数为 $p$，一个 micro batch 的前向传播和反向传播的执行时间分别为 $F$ 和 $B$，参数梯度计算时间为 $W$，F 与 B overlap 执行的时间为 $FB$，每个 micro batch 前向传播的 Activation 的激活显存为 $M_b$，Interleaved 1F1B 模式的 virtual stage 数为 $v$。则各个流水线并行模式的开销如表 2 所示：

| Pipeline | Bubble 数量 | Activation | Parameter | PP communication | 优点 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| GPipe | (p-1)(F+B) | m*Mb | 1x | 1x |  |
| 1F1B | (p-1)(F+B) | (p-i)*Mb | 1x | 1x |  |
| Interleaved 1F1B | (p-1)(F+B)/v | (1+1/v)(p-i)*Mb | 1x | vx | 较为均衡 |
| Looped BFS | (p-1)(F+B) | m*Mb/v | 1x | 1x | 数据并行友好 |
| ZB-H1 | (p-1)(F+B-2W) | (p-i)*Mb | 1x | 1x |  |
| ZB-H2 | (p-1)(F+B-3W) | 2(p-i)*Mb | 1x | 1x |  |
| ZB-V | (p-1)(F+B-3W)/2 | (p-i)*Mb | 1x | 2x | Bubble 最低 |
| DualPipe | (p/2-1)(FB+B-3W) | (p-i)*Mb | 2x | 2x | overlap all-to-all |

表2 各流水线开销总结

## 思考与延伸
由于 DualPipe 的 Dual 部分会导致 2 倍的参数冗余，因此可以参考 V-Shape 的思想，使用另一个 virtual Stage 来替代双向的 pipeline，称为 Cut-in-half（[https://hackmd.io/@ufotalent/S1N_ay0ckx](https://hackmd.io/@ufotalent/S1N_ay0ckx)）。这是因为 DualPipe 流水线可以上下分为两个完全相同的镜像，因此用 virtual stage 来替代另一个 pipeline 流仍然可以掩盖计算与通信，具体调度如图 23 所示。
![](https://picx.zhimg.com/v2-b2319da54c4a917efbc72b45f53134c3_r.jpg)
图23 Cut-in-half 流水线

由于设备数量减少了一半，每个设备的参数内存占用仍然保持不变，因此可以通过增加设备数量并减少每个阶段的层数，设计一个与 DualPipe 设备数量一致的 Cut-in-half 调度方案。由于层数减少，每个设备的参数量降至原来的 50%。这样做的缺点是增加了 virtual stage 后，会引发更严重的流水线 imbalance。

## 参考
1. [https://arxiv.org/abs/2211.05953](https://arxiv.org/abs/2211.05953)
2. [https://arxiv.org/abs/2405.15362](https://arxiv.org/abs/2405.15362)
3. [https://arxiv.org/abs/2401.10241](https://arxiv.org/abs/2401.10241)
4. [https://arxiv.org/abs/2107.06925](https://arxiv.org/abs/2107.06925)
5. [https://zhuanlan.zhihu.com/p/22681871459](https://zhuanlan.zhihu.com/p/22681871459)
