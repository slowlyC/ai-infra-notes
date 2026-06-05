# CUTLASS 教程：持久化 Kernel 与 Stream-K

**原文**: [Colfax Research - CUTLASS Tutorial Part 3](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)

**翻译说明**: 本文翻译自 Colfax Research 的 CUTLASS Tutorial 系列 Part 3，技术术语保留英文或附中文注释。

---

欢迎来到 GEMM（通用矩阵乘法）教程系列的第三部分。在 [Part 1](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) 和 [Part 2](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) 中，我们从单个 threadblock 的视角详细讨论了 GEMM，介绍了 WGMMA 矩阵乘法原语、流水线以及 warp  specialization（warp 专业化）。在本部分中，我们将从整个 grid 的视角审视 GEMM。在此层面，主要有两类优化：(1) 利用 threadblock swizzling 和 cluster 来最大化 L2 cache 命中; (2) 在 threadblock 之间更合理地划分工作，以饱和 GPU 的计算资源并实现良好的负载均衡。本文重点讨论后者（我们也在附录中讨论前者）。

具体而言，我们将讨论名为 [Stream-K](https://arxiv.org/abs/2301.03598) 的划分策略，它解决的是 **wave quantization（波次量化）** 问题——当工作 tile 数量无法被 streaming multiprocessor（SM）数量整除时就会出现这一问题。当标准的基于 tile 的输出划分无法占满 GPU 时（例如 M 和 N 较小而 K 较大），Stream-K 同样有用。

本文结构如下。我们首先描述 wave quantization 问题以及持久化 kernel 的概念。接着回顾在 threadblock 之间划分 GEMM 工作负载的多种策略，包括 Stream-K 及其前身 [Split-K](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#parallelized-reductions)，重点说明它们如何应对 wave quantization。然后解释 kernel 作者如何编写自己的 tile scheduler; 作为示例，我们在本系列 Part 2 的 GEMM kernel 上添加了 Stream-K 实现，[代码托管于 Github](https://github.com/ColfaxResearch/cfx-article-src/tree/master/streamk)。最后，附录深入介绍 CUTLASS 中的 Stream-K 实现。

## 概览：Wave Quantization 问题

NVIDIA GPU 由若干 streaming multiprocessor（SM）组成：每个 SM 拥有自己的 shared memory、register file、Tensor Core 等，彼此独立运行。理想的工作负载会通过均匀分配工作充分利用 SM 之间的并行，使所有 SM 在整个 kernel 执行期间保持繁忙。然而，若部分 SM 比其余 SM 更快完成其任务，它们将空闲等待其他 SM 完成。这就是负载不均（load imbalance）的典型表现。

考虑可将计算划分为等大同等工作单元的情况，每个工作单元可由单个 SM 在相同时间内完成。例如，GEMM 通常被划分为每个计算一个 bM x bN 输出 tile 的工作单元。这些工作单元再分配给 threadblock（CTA），每个 CTA 在可用 SM 上计算其分配的工作单元。我们将工作单元到 SM 的分配称为 **scheduling（调度）**。

若工作单元数量超过可用 SM 数量，工作单元将以多波（wave）方式处理，其中 1 wave 表示每个可用 SM 完成一个工作单元。

**Wave quantization** 在工作单元数量无法被可用 SM 数量整除时出现。例如，假设有 10 个工作单元和 4 个 SM，则工作单元执行时间线大致如下：

![Wave quantization 时间线](https://research.colfax-intl.com/wp-content/uploads/2024/12/quantization.png)

此时前两波为全波，所有 SM 都在使用。最后一波为部分波，仅有半数 SM 在工作。

当工作项数量相对 SM 数量较小时，wave quantization 会严重拖累性能。例如，在拥有 114 个 SM 的 H100 PCIe GPU 上，115 个工作单元需要 2 波——与 228 个工作单元完全相同！换言之，增加第 115 个工作单元会使设备利用率大约减半。另一方面，虽然 114,001 个工作单元也会受到同样的量化效应，其相对 kernel 总代价的影响很小。更多信息可见 [NVIDIA 深度学习性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quant)。

为观察 wave quantization 的影响，我们使用本系列 Part 2 中构造的 GEMM kernel，在不同 wave 数量下测量性能。设 MxK 矩阵 A 与 KxN 矩阵 B 的 GEMM，`bM` 和 `bN` 为工作 tile 的维度，并假设它们能整除 M 和 N。则总 wave 数为 `ceil((M/bM * N/bN)/num_SMs)`。为研究量化效果，我们需变化 **tiles-per-SM**，即 `(M/bM * N/bN)/num_SMs`; 小数部分表示最后一波的“满度”。因此，固定 `M=1024`、`K=4096`，以 `bN` 的步长变化 `N`（此处为 192）。

左图显示 TFLOPs/s 性能，右图显示耗时，测试基于 H100 PCIe GPU。竖直虚线表示 wave 边界（tiles-per-SM 跨越整数值处）。左图可见 wave quantization 效应——越过 wave 边界时性能骤降。右图表明耗时主要由总 wave 数这一离散参数决定（x 在 (0,1] 时为 1，在 (1,2] 时为 2，依此类推）。

![TFLOPs/s 和 elapsed time 随 tiles-per-SM 变化的性能曲线](https://research.colfax-intl.com/wp-content/uploads/2024/12/m1024-wave-quantization-time-tflops.png)

需注意第二次量化效应小于第一次——wave quantization 的影响随 wave 数量增加而减弱。然而，增加 wave 数量并不容易，尤其考虑到 NVIDIA GPU 的 SM 数量随新架构持续增长。因此，在不依赖问题规模假设的前提下，设计缓解 wave quantization 的策略很重要。

#### 持久化 Kernel（Persistent Kernels）

要应对 wave quantization，需要设计更好的划分与调度方案。此前本文中的 kernel 使用与问题维度相关的 grid，每个 CTA 只处理一个工作单元。例如在 GEMM 中，工作单元是 `MxN` 输出矩阵的 `bMxbN` tile，`bM`、`bN` 在编译期固定。每个工作单元由单个 CTA 在 `M/bM x N/bN` 的 grid 中计算。因此 launch 参数如下：

```
dim3 dimGrid(ceil_div(M, bM), ceil_div(M, bN));
```

该方式的问题在于，尽管对 threadblock 如何分配到 SM 有一定控制，却难以实现更复杂的调度策略。因此我们将采用另一种设计：**持久化 kernel**。在持久化 kernel 中，grid 大小为固定值。通常该值等于可用 SM 数量，使每个 CTA 独占一个 SM。可用如下 CUDA 代码获取 `dimGrid` 所需的 SM 数量：

```
int num_SMs;
cudaGetDeviceAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

dim3 dimGrid(num_SMs);
```

每个 CTA 在其 SM 上持续运行，处理多个工作单元直至全部完成。这种设计让程序员能更精细地控制调度，告知每个 CTA 如何遍历工作单元。凭借这一灵活性，我们可以以最小化 wave quantization 和负载不均的方式分配工作。

实践中，工作单元到 CTA 的分配通常交由 **tile scheduler（tile 调度器）** 完成，本质上是一个高级迭代器，告诉每个 CTA 下一个工作单元的位置以及何时停止。虽然每个输出 tile 的总工作量不变，但更换 tile scheduler 可以探索更复杂的策略以减轻负载不均，例如 Stream-K。

## 用持久化 Kernel 应对 Wave Quantization

为循序渐进地理解 Stream-K，先回顾一些更简单却低效的方案是很有帮助的。[Stream-K 论文](https://arxiv.org/abs/2301.03598) 对此有深入讨论，建议阅读。为方便读者，此处给出摘要。

为使本节数字更易理解，我们假设一个虚构的 GPU：[Hipparchus](https://en.wikipedia.org/wiki/Hipparchus) H10，仅有 4 个 SM。

#### Data Parallel（数据并行）

从最基础的版本开始：在 M 和 N 方向上均匀划分 tile，并以轮询方式分配。注意这与使用非持久化、按工作 tile grid launch 的 kernel 几乎相同，唯一区别是执行顺序可被保证。研究这一情况仍有助于理解 wave quantization 成为问题的场景。由于工作单元之间无依赖，这称为 **data-parallel work schedule（数据并行工作调度）**。

图 1：Data-parallel 划分。

![图 1 Data-parallel 分区](https://research.colfax-intl.com/wp-content/uploads/2024/12/split-mn.png)

图 1 为示例。GEMM 工作负载被划分为 9 个工作 tile。由于工作项相同，tile 按 wave 处理。具体地，9 个 tile 在 H10 的 4 个 SM 上以 3 wave 完成：2 个全波，以及 1 个部分波（4 个 SM 中仅 1 个在使用）。若每个 tile 在其 SM 上达到 100% 利用率，则整体利用率为 2.25/3 = 75%。

最直接的思路是：wave quantization 在工作单元更多时影响更小——可通过缩小每个工作单元来增加工作单元数量。

图 2：将 bN 减半后的 data-parallel 划分。

![图 2 减半 bN 后的分区](https://research.colfax-intl.com/wp-content/uploads/2024/12/split-mn-more-tiles.png)

图 2 在 N 方向将 bN 减半。现共有 18 个 tile，需 5 wave：4 个全波和 1 个部分波（4 个 SM 中有 2 个在用）。若仍假设每个 tile 100% 利用率，整体利用率为 4.5/5 = 90%。此外，图 2 中每个 tile 的 FLOP 约为图 1 的一半，粗略估计每波时间也应减半。因此，尽管图 2 有 5 波而图 1 有 3 波，图 2 的总耗时仅为图 1 的 (5*0.5)/3 = 83%！那问题出在哪？

遗憾的是，我们做了过多简化假设，已无法正确建模 Hipparchus H10 的行为。核心问题是：随 tile 尺寸减小，单个 tile 的计算可能变得更低效。因此，假设将 tile 减半会使计算时间减半或维持单 CTA 利用率不变，可能是错误的。

主要代价之一是 [arithmetic intensity（算术强度）](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem) 降低。由于访存耗时，我们希望用大量算术运算遮蔽访存延迟。在 GEMM 中，计算 matmul tile 的 CTA 会做算术和 GMEM 访问。注意减半只会减半前者，不会减半后者。例如 128 x 128 x 128 的工作 tile 对每次 GMEM 传输约有 85.3 次运算，而 128 x 64 x 128 则只有 64 次。

另一个问题：假设 CTA 大小不变， tile 减半意味着 CTA 中每个 warp 处理的指令减半，从而减少 warp scheduler 可用于隐藏延迟的机会，这对流水线 GEMM 的性能至关重要。

最后，tile 大小可能受到 MMA atom 选择的约束。例如 H10 可能需使用 128 x 128 x 16 的 WGMMA atom 才能获得最大吞吐。这进一步限制了 tile 的最小尺寸。

这些因素的权衡并不直观，为特定问题找到合适的 tile 大小可能需要试错——例如使用 [CUTLASS Profiler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/profiler.md)。

#### Split-K

此前我们只在 M 和 N 方向划分，还有另一个维度可划分：**K 方向**。当 K 很大时效果最好; 同样，bK 过小会对 arithmetic intensity 和 latency-hiding 造成损失。

Split-K 调度沿 K 方向将 tile 切为固定数量的片段。例如图 3 中沿 K 方向切为 2 个工件。

图 3：Split-K 划分。

![图 3 Split-K 分区](https://research.colfax-intl.com/wp-content/uploads/2024/12/split-k.png)

该策略带来新问题：每个 CTA 仅对其 bM x bN 输出 tile 积累部分结果。为完成计算，参与该输出 tile 的 CTA 需要合并结果。常见做法是在辅助 GMEM workspace 中进行 **turnstile reduction（转闸式归约）**。协作同一 tile 的各个 CTA 会等待处理更早 K 索引的 CTA 到达 barrier，然后将自己的部分结果归约写入 workspace 并到达 barrier。最后一个 CTA 不归约到 workspace，而是从 workspace 归约到自己的累加器并执行 epilogue。注意额外的 GMEM 访问和 barrier 同步会带来额外开销，图 3 中以 “arrive” 和 “reduce” 块表示。

Split-K 引入新超参：split 数量，伴随多种权衡：

- 引入同步与归约开销，这是 Split-MN 所没有的额外代价。split 越多，同步越贵。
- 增加 split 数量会减少每个 CTA 的指令数，从而削弱 latency-hiding 机会。
- 增加 split 会减小 K 方向的 tile 尺寸，可能提高 GMEM 访问与计算之比。
- 增加 split 能减弱 wave quantization，可能提升整体 SM 利用率。

#### Stream-K

前述策略改善了 wave quantization，但未消除它。回到 9 个 tile 分布在 4 个 SM 的例子，理想情况是每个 SM 运行 2.25 波。这正是 Stream-K 的动机。

Stream-K 策略为每个 SM 分配一个持久的 CTA。每个 CTA 被分配**分数数量**的工作 tile，被切分的 tile 均沿 K 方向切分。与 Split-K 一样，对每个被切分的 tile，协作的 CTA 可通过 GMEM workspace 中的 turnstile reduction 合并结果。

图 4：Stream-K 划分。

![图 4 Stream-K 分区](https://research.colfax-intl.com/wp-content/uploads/2024/12/stream-k.png)

例如图 4 中，SM0 上的持久 CTA 计算 work tile 0 全部、work tile 1 全部，以及 work tile 2 的 1/4。SM1 的 CTA 计算 work tile 2 剩余部分、work tile 3 全部，以及 work tile 4 的一半，依此类推。部分 tile 的调度使 worktile 的首段远早于末段完成，以最小化同步开销（注意，对于 K 方向极长的 tile，这并不总能实现）。

与之前策略相比：

- 需要额外的 GMEM 传输，使部分 tile 数据能在 CTA 间共享。
- 多数情况下，可将输出 tile 前段的计算安排在末段之前完成，使负责 epilogue 的 CTA 不必长时间等待 barrier。
- 大量原始 128 x 128 x 128 工作 tile 由单个 CTA 完整处理，因此部分保留大 tile 的优势：高计算/访存比、长指令序列、可使用大 WGMMA 指令。若原始 kernel 可达到每 CTA 100% 利用率，此 kernel 也可以。
- 通过消除 wave 消除了 quantization。每个 CTA 计算 2.25 个 tile。除同步与归约的额外时间外，总计算约为 2.25 单位，而原 kernel 需要 3 单位。

#### 混合 Stream-K（Hybrid Stream-K）

最后一类改进与 cache 性能有关。分 tile GEMM kernel 的特点之一是，每个操作数 tile 会被多个输出 tile 共用。例如在 split-MN 情形中，tile B0 用于计算输出 tile 0、1、2。

图 5：wave 下的数据复用。

![图 5 数据复用](https://research.colfax-intl.com/wp-content/uploads/2024/12/dp-reuse.png)

此处输出 tile 0、1、2 同时计算。当某个 CTA 从全局内存取 tile B0 时，它也会进入 L2 cache。其他同样请求 B0 的 CTA 将 cache 命中，从而更快加载。cache 有限，旧数据可能被淘汰，因此这些请求需在相近时间发生。

更具体地，操作数 tile 也沿 K 方向划分，每个 CTA 对其操作数 tile 的 K-block 做内层循环。wave 0 开始时，SM 0、1、2 会同时请求 tile B0 的第 0 个 K-block，其中两个会 cache 命中。下一轮循环中，它们会请求第 1 个 K-block，依此类推。

然而，stream-K kernel 引入了**偏移（skew）**：由于各 SM 先从不同大小的部分 tile 开始，它们倾向于同时处于不同的 K-offset。回到图 4，wave 0 开始时 SM0 和 SM1 都在用 B0 的数据——但 SM0 需要第 0 个 K-block，SM1 需要中间段。该调度下 K-offset 几乎从不对齐，cache 命中更难。简言之，消除 “wave” 并使各 SM 不同步调度，会带来 cache 性能变差的隐藏代价。

解决办法是将计算重新调度为持久化 kernel 与普通 data-parallel kernel 的**混合**。由于 data-parallel 调度没有 skew，应尽可能使用该调度，仅保留足够多的 tile 用 Stream-K 处理 wave quantization。为在 Stream-K 阶段均衡 SM 负载，需将 1 个全波和 1 个部分波分配给该阶段。

该调度如图 6 所示。初始 Stream-K 阶段处理 1–2 个全波的计算。每个 SM 最多获得 2 个部分 worktile。设计上，这些 tile 的总大小与 CTA 无关，因此所有 CTA 预期在此阶段大约同时完成。该阶段结束后，仅剩完整 work tile，且数量能被 SM 数整除。因此可用非持久化、data-parallel 策略计算这些 tile，既无 wave quantization，又享有更好 cache 性能。见图 6：

图 6：混合 Stream-K 划分。

![图 6 混合 Stream-K](https://research.colfax-intl.com/wp-content/uploads/2024/12/hybrid.png)

在此可以预期 work tile 6、7、8 几乎同时计算，并带来操作数 tile B2 的 cache 命中。类似地，work tile 5 和 8 可共享 A tile 的 cache。本例中 data-parallel 阶段仅 1 波，但 tile 更多的更大 GEMM 会有更长的 data-parallel 阶段和更多 cache 利用。

## Tile Scheduler 抽象

由于划分与调度工作的问题与 per-CTA 的内存和计算 largely 分离，GEMM 实现（如 CUTLASS）常将它们封装为 **tile scheduler** 抽象。（这比 GEMM 更通用——例如 [FlashAttention-3 也支持带 tile scheduler 类的持久化 kernel](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/tile_scheduler.hpp)。）下一节我们具体考察 CUTLASS 的实现; 这里先概括 tile scheduler 的职责。

首先，kernel 的 grid 形状取决于 tile 调度。tile scheduler 负责确定 kernel 的 grid 大小。对非持久化 kernel，这与逻辑 grid 相同且依赖问题规模; 对持久化 kernel，则固定，通常等于 SM 数量。我们在启动时向 tile scheduler 查询 grid 大小并用其 launch kernel。

在 kernel 内，每个线程会构造 tile scheduler 的实例。mainloop 和 epilogue 被包裹在 scheduler 提供的 tile 工作循环中，形式类似：

```cpp
for (auto worktile = scheduler.get_initial_tile();
    scheduler.is_valid(worktile);
    worktile = scheduler.get_next_tile(worktile)) {
        auto [m_block, n_block, k_block_start, k_block_stop] = worktile.get_block_coord();
        for (k_block = k_block_start; k_block < k_block_stop; ++k_block) {
            // mainloop
        }
        // epilogue
}
```

实现这些迭代原语的一种简单方式是让 scheduler 维护 worktile 的线性下标。对持久化 kernel，每个 CTA 初始获得下标为 `blockIdx.x` 的 worktile（即底层 SM 的线性下标）; 通过每次前移 `gridDim.x`（SM 数量）得到下一个 tile; 当下标不超过总 tile 数时 tile 有效。将线性下标映射到实际 (M, N) tile 坐标的工作委托给 `worktile` 对象。

这对持久化 data-parallel 调度已足够，但 Stream-K 等更复杂调度需要更多能力。对 Stream-K，K 方向的工作 assignment 大小与 tile 相关，因此 worktile 实际应向 kernel 提供四个坐标，如代码所示。

对 Stream-K 和 Split-K，部分或全部 CTA 会输出需要聚合的部分结果，由此带来：

- 只有 1 个 CTA 负责该输出 tile 的 epilogue。该 CTA 需从 workspace 归约到自己的累加器并执行 epilogue，而非归约进 workspace。scheduler 需告知每个 CTA 在其参与的每个 tile 上是否负责 epilogue。
- 开始新 worktile 时，每个 CTA 需知晓这是完整输出 tile（结果写入输出 tensor）还是部分 tile（结果写入 workspace）。
- 需要额外的 GMEM workspace，既用于部分结果，也用于协作同一 tile 的 CTA 间同步的 barrier 数组。所需空间依赖问题规模，需在 kernel launch 前动态分配。kernel 内，scheduler 需向 CTA 提供 workspace 的合适指针。

正如 CUTLASS 实现所示，可在上述简单框架上做诸多改进，包括让 scheduler 决定 tile 启动顺序、用启发式在 Stream-K、Split-K 和 data-parallel 间回退，以及在 Hopper 上正确使用 cluster。下文将展开说明。

[我们在 GitHub 上的示例](https://github.com/ColfaxResearch/cfx-article-src/tree/master/streamk) 提供三种 scheduler： trivial 非持久化 scheduler（按问题形状的 grid 为每个 CTA 分配 1 个 worktile）; data-parallel 持久化 scheduler; 以及融合部分（非全部）CUTLASS 优化的 Stream-K 混合 scheduler。实践中我们发现，很多 CUTLASS 的优化对获得合理性能是必要的：归约带来的额外 GMEM 访问和更小 tile 是真实代价，Stream-K 工作 assignment 的边界需精心调整以最小化该代价。

下方展示 Stream-K tile scheduler 的部分性能指标。相对于 data-parallel scheduler，我们的 Stream-K 实现每波早期表现更好，减轻了 wave quantization，但部分尾波开始填满时性能有所下降。“Heuristic” 曲线采用 CUTLASS 的启发式：当尾波至少半满时从 Stream-K 切回 data-parallel。这显然是明智的选择。

![无启发式时的性能曲线](https://research.colfax-intl.com/wp-content/uploads/2024/12/m1024-no-heuristic.png)

![有启发式时的性能曲线](https://research.colfax-intl.com/wp-content/uploads/2024/12/m1024-w-heuristic.png)

## 总结

本文讨论了 wave quantization 及其对 GEMM 性能的影响。我们在 Part 2 构建的 GEMM 实现中观察到 wave quantization 造成的明显性能波动。随后讨论了应对 wave quantization 的多种策略，重点介绍 Stream-K。最后给出了 Stream-K tile scheduler 的一版实现，以消除 GEMM 实现中的 wave quantization 效应。本文是本系列用 CUTLASS/CuTe  abstraction 实现高性能 Hopper GEMM 的第三部分总结。

## 附录：CUTLASS 中的 Stream-K

本附录探讨 CUTLASS 中 Stream-K 的细节：使用方法、相对其他 scheduler 的性能以及部分实现优化。

#### 在 GEMM API 中使用 Stream-K

首先介绍如何在 CUTLASS 3.X GEMM API 中使用 Stream-K scheduler。我们先简要回顾 CUTLASS 3.X GEMM API，讨论限于与 Stream-K 相关的部分; 更多 [细节](https://github.com/NVIDIA/cutlass/blob/main/media/docs/gemm_api_3x.md) 与 [示例](https://github.com/NVIDIA/cutlass/tree/main/examples) 见 CUTLASS 仓库。本文代码基于 CUTLASS [example 48](https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu)。

CUTLASS GEMM API 分为三部分：

- **Kernel**：对 epilogue 和 mainloop 的封装
- **Mainloop**：定义单个 worktile 如何计算
- **Epilogue**：定义部分结果如何合并及可能修改

它们由各自的 CollectiveBuilder 创建，为开发者配置 GEMM kernel 提供能力。开发者也可让 CUTLASS 根据内部 heuristic 自动选择配置。以下为使用 auto 的 GEMM kernel：

```cpp
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue
>;
```

要让 GEMM kernel 使用 Stream-K，需让 `GemmKernel` 使用 `StreamKScheduler`：

```cpp
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler
>;
```

此外，仅特定 mainloop 和 epilogue schedule 支持 Stream-K。Mainloop 和 Epilogue 都需使用 `TmaWarpSpecializedCooperative`：

```cpp
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    // ..... //
    cutlass::epilogue::TmaWarpSpecializedCooperative
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    // ..... //
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;
```

此时 GEMM kernel 已配置为使用 Stream-K scheduler。需注意 Stream-K scheduler 并非总是采用 Stream-K 划分。默认情况下会根据内部 heuristic 选择最佳划分。CUTLASS scheduler 为其 DecompositionMode 提供四种选项：

- `Heuristic`：CUTLASS 根据问题选择模式
- `StreamK`：实现 Stream-K 划分
- `SplitK`：实现 SplitK，用户指定 split 数
- `DataParallel`：K 方向无 split

稍后会更深入讨论 decomposition 模式。目前可通过在 scheduler 参数中设置，强制使用 Stream-K decomposition。可在 `Gemm` 的 arguments 中完成：

```cpp
using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
DecompositionMode decomp = DecompositionMode::StreamK;

int splits=1;
typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
scheduler_args = { splits, static_cast<int>(options.swizzle), options.raster, decomp};

typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
    hw_info,
    scheduler_args
};
```

除 `DecompositionMode` 外，scheduler 参数还包括与 Split-K 和 threadblock rasterization 相关的选项（附录后文也会讨论）。准备好 arguments 和 `GemmKernel` 后，即可用 Stream-K 划分运行 GEMM：

```cpp
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
Gemm gemm;

size_t workspace_size = Gemm::get_workspace_size(arguments);

cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
CUTLASS_CHECK(gemm.can_implement(arguments));
CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
CUTLASS_CHECK(gemm.run());
```

#### Stream-K 性能

既然已介绍如何用特定 scheduler 运行 GEMM，下面看不同输入规模下的表现。同样固定 M、K，以 tile 尺寸的步长变化 N，横轴为 tiles-per-SM：`(M/bM * N/bN)/num_SMs`。我们比较 Stream-K、Split-K 和 DataParallel 三种模式，并针对不同 K 值重复测试。基准在 H100 PCIe GPU 上进行。

竖直虚线表示 wave 边界。DataParallel 在跨越 wave 边界时如预期出现明显性能下降，即 wave quantization 效应。DataParallel 在最后一波接近满（tiles-per-SM 略小于整数）时优于或持平其他模式，在接近空（略大于整数）时落后。wave quantization 在总 wave 数较少时最为明显。

Split-K 下 wave quantization 有所缓解。Split-K  effectively 将 worktile 数量乘上 K 倍，wave 数也相应增加。图中可见 2 split 的 Split-K 性能振荡频率是 DataParallel 的两倍。遗憾的是，对多数情况归约的额外开销超过收益，Split-K 很少优于另外两者（通常只在 tile 极少、不 split 会导致 GPU 严重欠利用时）。图中仅展示 K=2 的 Split-K 以保持清晰; 除极小的 X 外，更大 K 通常不如 K=2。

Stream-K 则几乎不表现 wave quantization，随 wave 数变化波动很小。Stream-K 划分通常优于或持平 Split-K，在 K 较大且最后一波接近空时优于 DataParallel。DataParallel 与 Stream-K 在 N=7296 处（对应 X=1024*7296/114=4）结果相同，因 tile 可均匀分配给 CTA，无需部分 tile 或归约，故两者等价。

除三种显式 decomposition 模式外，CUTLASS 还有 Heuristic 模式。具体 heuristic 见后文，但可看出它相对 Stream-K 和 DataParallel（split-K 已剔除）的表现。

可见 CUTLASS Heuristic 模式能很好地预测最佳 decomposition 模式。量化效应低时选 DataParallel，高时选 Stream-K。由于 Heuristic 为默认，一般不需指定 decomposition 模式，让 CUTLASS 决定即可。

#### CUTLASS 实现细节

下面讨论 CUTLASS 中 stream-K scheduler 的实现细节（以 CUTLASS 3.6 为准）。

**Schedule**。CUTLASS 实现上文介绍的混合 schedule 的一个版本：scheduler 最多将 2 个 wave 分配给 Stream-K 工作，剩余工作以 data-parallel 方式组织。由于 data-parallel wave 倾向于同时工作在相同 K offset，L2 cache 性能应得到改善。

**Reduction**。默认情况下，协作同一输出 tile 的 CTA 以 “turnstile” 方式工作。设某输出 tile 由 CTA 0、1、…、n 处理，按分配到的 K 索引范围递增排序。首先 CTA 0 计算并写入全局内存 workspace。CTA 1 在 barrier 等待 CTA 0 写完，然后将自己的输出归约进同一 workspace。CTA 2 等待 CTA 1 后归约，依此类推。最后 CTA n 等待 CTA n-1，但不归约进 workspace，而是从 workspace 归约到自己的累加器，再计算 epilogue 并写入输出 tensor。

在可选的 “nondeterministic 模式”（用户通过参数 `ReductionMode::Nondeterministic` 指定）下，CTA 1、…、n-1 不再相互等待，而是原子归约进 workspace。所有 CTA 仍须等待 CTA 0 初始化 workspace; CTA n 仍须等待 CTA 0、…、n-1。非确定性源于归约 1、…、n-1 可以任意顺序发生（且浮点加法非结合）。

**Decomposition 模式**。CUTLASS stream-K scheduler 也支持 Split-K 和 data-parallel 持久化 schedule，用户可通过 `decomposition_mode` 参数选择。（传入 `splits` 不等于 1 会强制 scheduler 以给定 split 数运行 split-K。）用户也可选择 `DecompositionMode::Heuristic`，scheduler 会从 stream-K 回退到更简单 schedule：若无 wave quantization，或尾波至少半满，则回退到 data-parallel; 若被分配 stream-K 工作的 CTA 数能被它们应处理的 stream-K tile 数整除，则回退到 split-K。由于 Stream-K 带有归约与同步的额外开销，在 wave quantization 不构成问题时回退到 data-parallel 合理。我们的测试表明，该 heuristic 在多种问题规模下几乎总能做出最优选择。

**Threadblock rasterization（线程块光栅化）**。持久化 kernel 的另一优势（与 wave quantization 无关）是可以选择 worktile 的启动顺序。对 GEMM 而言，这主要影响 cache：若输出矩阵同行或同列（相同 M 或 N 索引）的 worktile 在相近时间被处理，它们会同时从 GMEM 加载某一操作数矩阵的数据，有利于 L2 cache 命中。

因此，改进持久化 kernel cache 性能的简单做法是沿 M 或 N 方向按序启动 worktile。例如沿 N 方向、尽量保持 M 不变地 launch，操作数矩阵 A 的数据常会 cache 命中。在 CUTLASS 中，可向 scheduler 传入 `raster_order`，`RasterOrderOptions::AlongM` 和 `AlongN` 提供该行为。通常希望沿 worktile 单位下较短的方向 raster; `RasterOrderOptions::Heuristic` 会自动选择。

图 7：沿 M 方向的光栅化。

![图 7 Threadblock rasterization](https://research.colfax-intl.com/wp-content/uploads/2024/12/rasterization-2.png)

图 7 展示当 M < N 时沿 M 方向的 thread block rasterization 情形。

![Swizzle 示意图 1](https://research.colfax-intl.com/wp-content/uploads/2024/12/swizzle-2.png)

![Swizzle 示意图 2](https://research.colfax-intl.com/wp-content/uploads/2024/12/swizzle-3.png)
