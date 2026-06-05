# CUTLASS 教程：基于流水线的高效 GEMM Kernel 设计

**原文**: [Colfax Research - CUTLASS Tutorial Part 2](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)

**翻译说明**: 本文翻译自 Colfax Research 的 CUTLASS Tutorial 系列 Part 2，技术术语保留英文或附中文注释。

---

欢迎来到 GEMM（通用矩阵乘法）教程系列的第二部分。在 [Part 1](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) 中，我们通过 WGMMA 讨论了 GEMM 的计算层面——这是基于 NVIDIA® Hopper™ 架构 GPU 上用于乘小矩阵块的原始指令。在本部分中，我们将聚焦 GEMM 的内存层面，具体讲解如何高效地将操作数张量的小块从 GPU 的全局内存（global memory）搬入其片内内存（on-chip memory），以便传入 WGMMA（或其他原始 MMA 指令）。

要解释的核心概念是：如何编排数据流水线，以高效地“喂饱” Tensor Core。在 GEMM kernel 设计语境下，**流水线（pipelining）** 指通过维护多个数据缓冲区，将 copy 操作与 MMA 操作在时间上重叠执行。本文将介绍在 Hopper 架构上有效的两种流水线策略：

- **多阶段（Multistage）**：通过异步 copy（Hopper 上的 TMA 或 Ampere 上的 `cp.async`）加载下一批数据，同时在当前批数据上计算，从而掩盖数据传输。各 warp 兼具 producer 和 consumer 角色。
- **Warp 特化（warp-specialization）**：将 warp 划分为 producer（数据传输）和 consumer（计算），并让二者并发执行。

为保证 kernel 的正确性，需要仔细处理数据依赖：它们决定了 buffer 何时可被 MMA 指令读取，以及何时可被 copy 操作填充。我们将详细说明如何借助 CUTLASS 库中的工具——尤其是 CUTLASS Pipeline 类——编写流水线 GEMM kernel 所需的同步逻辑。

文中还会给出流水线的性能评估，展示仅利用这一种优化思路，即可在半精度下使 Hopper GEMM kernel 达到约 65% 的利用率。附录中我们会说明如何为基于 NVIDIA Ampere 架构的 GPU 编写流水线 GEMM kernel。

## 整体思路："喂饱猛兽"

GEMM kernel 主要有两类操作：把数据 copy 到正确的内存地址，以及进行乘加运算。前者由 copy 指令完成：[Hopper 上的 TMA](https://research.colfax-intl.com/tutorial-hopper-tma/)、[Ampere 上的 cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)，以及更早架构上的普通 copy。后者自 2017 年 [Volta 架构](https://en.wikipedia.org/wiki/Volta_(microarchitecture)) 起，已完全由 Tensor Core 负责。

经过多代演进，Tensor Core 已成为“吞数”猛兽。例如 H200 SXM GPU 的 Tensor Core 可达 [3,958 TFLOPS](https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/hpc-datasheet-sc23-h200)（每秒万亿次浮点运算），而同款 H200 SXM 的内存带宽仅为 4.8 TB/s。数据搬运速度远慢于 Tensor Core，要充分利用并不容易。因此 CUDA 编程——尤其是 GEMM kernel 设计——的一个常见主题是：如何足够快地 copy 数据，让 Tensor Core 保持忙碌。我们称之为“喂饱猛兽”。

一般而言，有两种互补策略可用于“喂饱猛兽”，它们作用于不同范围（grid vs block）。第一种是**高效的 threadblock 调度**，即在 CTA 间分配计算以获得良好的负载均衡和更高的 L2 cache 命中率。我们将在后续博客中讨论，目前可参考 [threadblock rasterization](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#threadblock-rasterization) 和 persistent kernel 等技术，例如 CUTLASS 中的实现。本教程关注的是第二种：**将 copy 与计算重叠**。具体而言，在 Tensor Core 忙于对当前一批数据进行乘加时，应让 copy 单元去 copy 下一批数据，从而有效掩盖部分 copy 延迟。这正是流水线的目标。

### 延迟、Warp 与 Warp 特化

在介绍流水线机制前，先简单回顾引言中提到的两种重叠策略：多阶段（multistage）与 warp 特化（warp-specialization）。

首先，将内存 copy 与计算重叠的想法并非 GPU 独有。熟悉 CPU 的读者会想到 [cache prefetching](https://en.wikipedia.org/wiki/Cache_prefetching)：在数据被需要之前发起异步预取请求。本文讨论的流水线在概念上与 CPU cache prefetching 一致。不过，GPU 上的硬件预取 [在芯片面积上代价较高](https://developer.nvidia.com/blog/boosting-application-performance-with-gpu-memory-prefetching/)，因此实现方式不同。

GPU 程序员实现重叠的最基本方式是利用“富余的 warps”。NVIDIA GPU 允许每个 SM（[streaming multiprocessor，流多处理器](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)#Streaming_multiprocessors)）上驻留大量 warps，并可以极低开销在它们之间切换。当某个 warp 因慢速访存而停滞时，warp scheduler 可以直接切到另一个 warp。为给 scheduler 提供更多掩盖延迟的机会，大约在 2011 年引入了 **warp 特化** [1, 2]：一部分 warp 专门负责内存 fetch（producer），另一部分专门负责计算（consumer），并通过命名 barrier 进行同步。这样 warp scheduler 可以更轻松地将 copy 延迟隐藏在计算中（反之亦然）。

自 Ampere 架构起，Nvidia 引入 `cp.async`，使得在同一 warp 内既可 copy 又可做计算，且 copy 为异步。也就是说，warp 可以发出 `cp.async` 将数据加载到下一个 buffer，然后立即在当前 buffer 上执行运算，而无需等待异步 load 完成。这使“用 warp 特化来掩盖数据传输”变得不再必要。**多阶段 kernel 设计**正是利用这一点。最快的 Ampere GEMM kernel 以及著名的 FlashAttention-2 都采用多阶段设计。

最新的 Hopper 架构则引入了 TMA 异步 copy、warpgroup 级寄存器重分配等新特性，使得 warp 特化在 Hopper 上非常有效（详见后文）。最快的 CUTLASS Hopper GEMM kernel 采用 warp 特化。

### 流水线示意图

图 1 展示了 `LOAD` 与 `MMA` 的理论流水线。其中 `LOAD` 表示将操作数矩阵块从 GMEM 复制到 SMEM，`MMA` 表示对 SMEM 中操作数块做 Tensor Core 乘加。如图所示，通过将两次 `LOAD` 与两次 `MMA` 重叠，可节省 2 个单位时间。

**图 1.** 3 次 load 与 3 次 MMA 的流水线示意图。

![Figure 1. 3 个 load 和 3 个 MMA 步骤的流水线示意](https://research.colfax-intl.com/wp-content/uploads/2024/08/Pipeline-illustration.png)

由图 1 引出的问题是：`LOAD_1` 和 `LOAD_2` 把数据 copy 到哪里？显然，不希望后续 load 在 MMA 完成计算之前覆盖此前 load 的数据，也不希望因等待 SMEM 有空位可写而导致不必要 stall。否则，预期的 2 个单位时间收益就无法实现。

一个简单办法是：在 SMEM 中预留 MMA 所需两倍的内存，并交替使用。这种策略称为**双缓冲（double buffering）**，见图 2。当然，可以推广到两个以上的 buffer，以创造更多重叠机会，更高效利用硬件，代价是占用更多 SMEM。

**图 2.** 使用两个交替 SMEM 阶段 `S_0` 和 `S_1` 的流水线。矩阵块依次加载到 `S_0` 和 `S_1`，与 Tensor Core 运算重叠。注意全局 tile 记为 `G_1`、`G_2`、`G_3`、`G_4` 等，是递增而非像 SMEM 阶段那样交替，因此每一步都操作新的 tile。

![Figure 2. 双阶段 SMEM 交替流水线](https://research.colfax-intl.com/wp-content/uploads/2024/08/Pipeline-2-stages-1.png)

正确且高效地实现流水线并不 trivial。程序员需要处理多 buffer 以及跨多线程的异步 load 调用。下一节说明如何通过 CUTLASS 抽象——`Pipeline` 类——实现流水线。

### CUTLASS Pipeline 抽象

CUTLASS 的异步 [Pipeline 类](https://github.com/NVIDIA/cutlass/blob/main/media/docs/pipeline.md) 是管理多 buffer、多参与线程下 copy 与 compute 的有效抽象，包括 `PipelineAsync`、`PipelineTmaAsync` 和 `PipelineTransactionAsync`，本文用 “`Pipeline`” 泛指。

先在高层次说明 CUTLASS `Pipeline` 如何编排数据流水线。设 `buffers` 为含 `N` 个 stage 的共享内存 buffer，需要在向 buffer 写数据的 producer（如 TMA）与使用这些数据的 consumer（如 WGMMA）之间做同步。

![CUTLASS 软件流水线](https://github.com/NVIDIA/cutlass/raw/main/media/images/software-pipeline.png)

**Barrier。** 为使 producer 与 consumer 之间在 buffer stage 上同步，Pipeline 遵循标准的 acquire-release 模型，用 lock 管理对 buffer 的访问。为此，设 `full_barrier` 和 `empty_barrier` 为两个长度为 `N` 的 barrier 对象数组，这些 barrier 对象具有 phase bit，初始为 0，在 0 与 1 之间翻转。

具体而言，这些 barrier 对象是驻留在 SMEM 的 [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)。mbarrier 除 phase bit 外还带有 arrival count，支持 arrive-on 与 wait 操作，并在达到 arrival count 阈值时翻转 phase。重要的是，这些 barrier 的值需对所有线程可见。

**线程本地流水线状态。** `PipelineState` 类是一个线程本地的枚举器，用于跟踪当前 stage 的 index 和 phase，其中 stage 数量 `N` 作为模板参数传入。index 取模 `N` 的整数值，phase 为 0 或 1。`PipelineState` 的 `++` 运算符被 [重载](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cutlass/pipeline/sm90_pipeline.hpp#L140)，使 index 按模 `N` 递增，且当 index 递增回 0 时 phase 翻转。

**同步。** 下面说明 barrier 与 PipelineState 如何用于 producer 与 consumer 的同步。为避免歧义，需区分“producer 动作”和“发起该动作的 producer 线程”，二者可能解耦（如 TMA）。首先，producer 动作将翻转 `full_barrier[i]` 的 phase，表示已填满第 `i` 个 stage，consumer 可以从中读取。类似地，consumer 线程翻转 `empty_barrier[i]` 的 phase，表示已消费完第 `i` 个 stage，producer 可以写入。

我们对 producer 动作或 consumer 线程如何通过 arrival count 机制翻转 SMEM 中的 phase bit 不做具体规定。例如，所有 consumer 线程可以共同增加 arrival count，或每 warp 选一个 consumer 线程来执行。

最后，每个线程（无论 producer 还是 consumer）都维护一个 phase 以与 barrier 的 phase 匹配; 若线程同时扮演 producer 和 consumer，则需要跟踪两个 phase。这些线程的“内部” phase 会随 mainloop 迭代而翻转。

**四种 pipeline 方法。** 设 `pipeline` 为由指向 `full_barrier` 和 `empty_barrier` 的指针初始化的 `Pipeline` 实例，`pipe_state` 为 `PipelineState` 实例。`pipeline` 可调用以下四种方法：

- `pipeline.consumer_release(pipe_state)`：向 `empty_barrier[pipe_state.index()]` 发出 arrival count 增加信号。
- `pipeline.consumer_wait(pipe_state)`：阻塞调用线程，直到 `full_barrier[pipe_state.index()]` 的 phase 相对于 `pipe_state.phase()` 翻转。
- `pipeline.producer_commit(pipe_state)`：向 `full_barrier[pipe_state.index()]` 发出 arrival count 增加信号。
- `pipeline.producer_acquire(pipe_state)`：阻塞调用线程，直到 `empty_barrier[pipe_state.index()]` 的 phase 相对于 `pipe_state.phase()` 翻转。

在 `producer_acquire` 和 `consumer_wait` 的说明中，“相对于 `pipe_state` 的 phase 翻转”意指：若 barrier 当前 phase 为 0，当 `pipe_state` 的 phase 为 0 时阻塞，为 1 时不阻塞。

按此描述，(`producer_acquire`, `consumer_release`) 与 (`producer_commit`, `consumer_wait`) 在功能上完全对称。但若 `Pipeline` 是 `PipelineTmaAsync`，则 `full_barrier` 被封装为 `cutlass::arch::ClusterTransactionBarrier` 实例，其 signaling 由 TMA load 方法通过增加 transaction count 完成。此时 `producer_commit` 实际上是 no-op; 后文会再说明。在伪代码中若未显式写出 TMA copy，我们仍会插入 `producer_commit`。

综合以上，下面的伪代码展示了四种 pipeline 方法的使用：

```cpp
using PipelineState = typename cutlass::PipelineState<N>;
// 我们将 smem_pipe_write 初始化为相反 phase（即 1 而非 0），
// 因为 buffer 初始为空。
PipelineState smem_pipe_write = cutlass::make_producer_start_state<Pipeline>();
PipelineState smem_pipe_read;
for (int i = 0; i < total_steps; ++i) {
  pipeline.producer_acquire(smem_pipe_write);
  // 获取数据（如 TMA、cp.async 等）
  pipeline.producer_commit(smem_pipe_write);
  ++smem_pipe_write;

  pipeline.consumer_wait(smem_pipe_read);
  // 计算负载（如 WGMMA）
  pipeline.consumer_release(smem_pipe_read);
  ++smem_pipe_read;
}
```

上述代码有助于理解 producer/consumer 的 acquire-release 模式，建议读者手动追踪几次循环中各状态，并与前面的同步描述对应。

但该 snippet 中 producer 与 consumer 串行执行，从未并发，因此并不实用。有效的流水线必须让 producer 与 consumer 重叠。接下来介绍**多阶段 kernel 设计**，给出一种实现方式。

### 多阶段 Kernel 设计

下面使用 Pipeline 的 TMA 专用版本 `PipelineTmaAsync`，在 Hopper GEMM kernel 中构建 2-stage 流水线，使 TMA 与 WGMMA 重叠。kernel 启动 128 个线程（即 1 个 warpgroup）。假设读者熟悉 CUTLASS 中 TMA 与 WGMMA 的用法（我们曾在两篇 [博客](https://research.colfax-intl.com/tutorial-hopper-tma/) [文章](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) 中详细讨论），此处略去传入 `cute::copy` 和 `cute::gemm` 的 tensor 准备过程。

```cpp
using MainloopPipeline = typename cutlass::PipelineTmaAsync<2>;
using PipelineState = typename cutlass::PipelineState<2>;

typename MainloopPipeline::Params params;
// 每 stage TMA load 传输的字节数（A 和 B）
params.transaction_bytes = TmaTransactionBytes;
params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
params.is_leader = threadIdx.x == 0;
params.num_consumers = 128;

// 本示例忽略 cluster
auto cluster_shape = Shape<_1,_1,_1>{};

// pipeline_storage 是 cutlass::PipelineTmaAsync<2>::SharedStorage 实例
// 成员包含 full_barrier 和 empty_barrier
// 位于管理 smem 对象的 SharedStorage 结构体中
MainloopPipeline pipeline(shared_storage.pipeline_storage, params, cluster_shape);

__syncthreads();

PipelineState smem_pipe_write = 
    cutlass::make_producer_start_state<MainloopPipeline>();
PipelineState smem_pipe_read;

// 准备 GEMM 所需的 tensor
// ...

// 由 leader 线程发起第一次 TMA load
if(threadIdx.x == 0) {
  pipeline.producer_acquire(smem_pipe_write);
  BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
  // smem_pipe_write.index() == 0
  copy(tma_load_a.with(*tmaBar, 0), tAgA(_,0), tAsA(_,0));
  copy(tma_load_b.with(*tmaBar, 0), tBgB(_,0), tBsB(_,0));
  ++smem_pipe_write;
}

for (int i = 0; i < k_tile_count - 1; ++i) {
  // 仅 leader 线程发起 TMA load
  if(threadIdx.x == 0) {
    pipeline.producer_acquire(smem_pipe_write);
    BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
    auto write_stage = smem_pipe_write.index();
    copy(tma_load_a.with(*tmaBar, 0), tAgA(_,i+1), tAsA(_,write_stage));
    copy(tma_load_b.with(*tmaBar, 0), tBgB(_,i+1), tBsB(_,write_stage));
    ++smem_pipe_write;
  }

  // 对上一轮已完成的 load 进行计算
  pipeline.consumer_wait(smem_pipe_read);
  auto read_stage = smem_pipe_read.index();
  // WGMMA
  warpgroup_arrive();
  gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), tCrC);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  pipeline.consumer_release(smem_pipe_read);
  ++smem_pipe_read;
}

// 处理最后一次计算迭代
pipeline.consumer_wait(smem_pipe_read);
auto read_stage = smem_pipe_read.index();
warpgroup_arrive();
gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), tCrC);
warpgroup_commit_batch();
warpgroup_wait<0>();
pipeline.consumer_release(smem_pipe_read);

// 写出累加器的 epilogue
axpby(alpha, tCrC, beta, tCgC);
```

这里，主循环的每次迭代会异步发起第 `(i+1)` 次 TMA load，并执行第 `i` 次 WGMMA 计算，`smem_pipe_write` 与 `smem_pipe_read` 相差一个 stage。

注意：TMA 博客中使用的 `cute::set_barrier_transaction_bytes`（或其等价物 `cutlass::arch::arrive_and_expect_tx`）在此未出现，其职责由 `PipelineTmaAsync` 的 `producer_acquire` 承担。该方法内部 [执行](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cutlass/pipeline/sm90_pipeline.hpp#L401)（其中 `stage` 和 `phase` 为其 `PipelineState` 参数的 index 与 phase）：

```cpp
if (barrier_token != BarrierStatus::WaitDone) {
   empty_barrier_ptr_[stage].wait(phase);
}

if (params_.is_leader) {
   full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
}
```

此外，我们通过 `producer_get_barrier(smem_pipe_write)` 获取 `full_barrier[smem_pipe_write.index()]` 的指针，供 `cute::copy` 中 TMA `TiledCopy` 对象 `tma_load_a` 和 `tma_load_b` 使用。

当 `cute::copy` 与 pipeline 的 `full_barrier` mbarrier 关联后，可利用 TMA 基于 transaction count 的完成机制通知 consumer buffer 已就绪，因此 pipeline 本身无需调用 `producer_commit`。这就是 CUTLASS 将 `PipelineTmaAsync` 的 `producer_commit` 做成 no-op 的原因。

通过这种方式组织流水线，可实现数据传输与计算的重叠，充分发挥异步操作掩盖延迟的潜力。本例虽用 TMA，但 Ampere 架构也可用 `cp.async` 实现类似效果，附录会进一步说明。不过，在 Hopper 上有时更适合采用 warp 特化设计而非多阶段，下面进行介绍。

### Warp 特化

在多阶段 kernel 中，每个 warp 同时担任 producer 和 consumer，角色切换由 `PipelineState` 抽象完成，TMA load 的异步性使两类操作可以重叠。**Warp 特化**是另一种策略：将不同 warp 分配不同角色，部分 warp 专门做 memory copy（producer），部分专门做计算（consumer）。如上所述，warp scheduler 可通过在两类 warp 间切换来掩盖延迟。注意：与多阶段不同，warp 特化本身不依赖异步执行，但在实际中仍能从中获益。

在我们的 GEMM 中，producer warp 用 TMA 将数据从 global memory 加载到 shared memory，consumer warp 用 WGMMA 进行 tile-wise GEMM。在简化 setting 下，两类 warp 内部执行是串行的，即 TMA 与 WGMMA 指令并未在 warpgroup 内部重叠。更精细的 kernel schedule 会利用 TMA 与 WGMMA 的异步性，实现 warpgroup 内与其他指令的重叠，例如 [FlashAttention-3](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/)。

Warp 特化在 Hopper 上特别有吸引力，原因有三：

- Hopper 支持通过 [setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) 指令在 warpgroup 粒度上手动进行寄存器（去）分配，因而可将更多寄存器分配给通常需求更大的 consumer warp。
- WGMMA 可从 shared memory 取操作数，consumer warp 无需自行做 load。
- TMA 相比早期 copy 指令寄存器占用更少。

关于最后一点：每个 SM 的寄存器有限，在 Hopper 之前的架构中，每个 warp 在 kernel 启动时被分配固定且相等的寄存器数。对多阶段 pipeline（各 warp 做相同工作）没问题，但对 warp 特化而言往往浪费：producer warp（只 load 数据）通常比 consumer warp（做运算）需要的寄存器少，尤其使用 TMA 时。对寄存器密集型 workload，利用这些“浪费”的寄存器，可能意味着让每个 SM 容纳更多 warp 或避免 register spilling。

下面给出 warp 特化代码 snippet。同前，`Pipeline` 类抽象了 warp 特化 kernel 的复杂配置。

```cpp
// 创建 pipeline 和 stage 的迭代器
using MainloopPipeline = typename cutlass::PipelineAsync<2>;
using PipelineState = typename cutlass::PipelineState<2>;

// Producer warps
if (isProducerWarp(threadIdx.x)) {
  // 只有一个线程应调用 TMA
  if(isTMAThread(threadIdx.x)) {
    PipelineState smem_pipe_write = 
      cutlass::make_producer_start_state<MainloopPipeline>();
    for (...) {
      pipeline.producer_acquire(smem_pipe_write);
      copy(...); // TMA
      ++smem_pipe_write;
    }
  }
}
// Consumer warps
else {
  PipelineState smem_pipe_read;
  for (...) {
    pipeline.consumer_wait(smem_pipe_read);
    // WGMMA
    pipeline.consumer_release(smem_pipe_read);
    ++smem_pipe_read;
  }
  // Epilogue
}
```

结构与前面的基本 pipeline 类似，但增加了外层条件分支，将 workload 拆给 producer warp 和 consumer warp。Epilogue 放在 consumer warp 中，因为需写出 consumer 线程寄存器中的累加器。

要判断线程所属 warp 与 warpgroup，可这样做：

```cpp
int warp_group_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
int warp_group_thread_idx = threadIdx.x % 128;
```

上述 snippet 使用 `__shfl_sync`，即 warp 内的值广播（更多信息见[此处](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)），以保证 warp 内所有线程得到相同值。

下面看其如何应用到 GEMM。本系列 [Part 1](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) 中，WGMMA 指令按 warpgroup 组织，因此 producer 与 consumer 也按 warpgroup 组织。使用 TMA pipeline，以便 producer 侧可用 TMA。

对于 2 stage、2 warpgroup 的 WS kernel，先修改 pipeline 初始化：

```cpp
using MainloopPipeline = typename cutlass::PipelineTmaAsync<2>;
using PipelineState = typename cutlass::PipelineState<2>;

typename MainloopPipeline::Params params;
params.transaction_bytes = TmaTransactionBytes;
const int producerWarpGroupId = 0;
if (warp_group_idx == producerWarpGroupId)
  params.role = MainloopPipeline::ThreadCategory::Producer;
else
  params.role = MainloopPipeline::ThreadCategory::Consumer;
params.is_leader = warp_group_thread_idx == 0;
params.num_consumers = 128;

auto cluster_shape = make_shape(Int<1>{},Int<1>{},Int<1>{});

// 创建 pipeline
MainloopPipeline pipeline(shared_storage.pipeline_storage, params, cluster_shape);
```

需强调第 12 行：虽然 `params.num_consumers` 仍为 128，但这里只统计 consumer warpgroup 的 128 个线程，而非全部 256 个。

主循环部分，总体结构与最初的示例一致，但 producer 侧有几处不同：

```cpp
// Hopper GEMM 1 个 consumer warpgroup 时的示例值
using LowerRegisterCount = Int<40>;
using HigherRegisterCount = Int<256>;

if (warp_group_idx == producerWarpGroupId) {
  cutlass::arch::warpgroup_reg_dealloc<LowerRegisterCount{}>();
  int lane_predicate = cute::elect_one_sync();
  if (warp_idx_in_warpgroup == 0 && lane_predicate) {
    PipelineState smem_pipe_write = 
      cutlass::make_producer_start_state<MainloopPipeline>();
    for (...) {
      pipeline.producer_acquire(smem_pipe_write);
      copy(...); // TMA
      ++smem_pipe_write;
    }
  }
} else { // consumer warpgroup
  cutlass::arch::warpgroup_reg_alloc<HigherRegisterCount{}>();
  PipelineState smem_pipe_read;
  for (...) {
    pipeline.consumer_wait(smem_pipe_read);
    gemm(...); // WGMMA
    pipeline.consumer_release(smem_pipe_read);
    ++smem_pipe_read;
  }
  // 写出累加器的 epilogue
  axpby(...);
}
```

第 6 行与第 18 行中，我们通过 [CUTLASS 调用](https://github.com/NVIDIA/cutlass/blob/3a8c01a18b24c35b216922481ac762496720a99d/include/cutlass/arch/reg_reconfig.h) 手动（去）分配多余寄存器，该调用进而使用 PTX 原语 [setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) 调整 warpgroup 内线程的寄存器分配。按文档说明，`warpgroup_reg_dealloc()` 释放多余寄存器，将每线程最大寄存器数降至 `M`; `warpgroup_reg_alloc()` 则请求额外寄存器，将每线程最大寄存器数提升至 `N`。

具体数值取决于算法和硬件约束。在 Hopper 上，单线程最多 255 个寄存器，`setmaxnreg` 可设为 24 至 256（含）范围内 8 的倍数。一般而言，Hopper GEMM WS kernel 建议让一个 CTA 占满一个 SM。因此应尽量满足：(a) 分配给发起 TMA 的 producer warpgroup 的寄存器尽可能少; (b) 用满 [每个 SM 64K](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy) 的寄存器文件。例如，1 个 producer warpgroup + 2 个 consumer warpgroup 时，24/240/240 划分通常有效（504 < 512，512*128 = 64*1024）; 1 个 producer + 3 个 consumer 时可用 32/160/160/160。注意：若分配总寄存器数超出寄存器文件大小，程序会崩溃。

此外，必须保证每个 warpgroup 中只有一个线程调用 TMA。在示例中，只有第一个 warp 参与，并通过 `elect_one_sync` 选出负责 TMA 调用的线程。上述代码针对 2 个 warpgroup，稍作修改即可用于更多 warpgroup 和 stage。

warpgroup 和 stage 的数量应结合 kernel profiling 谨慎选择。一般而言，更多 stage 和 warpgroup 带来更多并行与重叠机会，但也消耗更多资源：更多 stage 需要更多 SMEM，更多 warpgroup 会增加 register pressure。

## 性能

我们以 CUTLASS [Hopper GEMM 教程代码](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/wgmma_sm90.cu) 为基础，实现了多阶段与 warp 特化的半精度（FP16）GEMM kernel，并增加了 FP32 累加与 TMA store 输出。对 MxNxK = 8192×8192×8192 进行调优，FP16 与 FP32 累加采用不同 tile 尺寸。所选 tile 尺寸与 stage 数如下（bMxbNxbK 能整除 MxNxK）：

- **FP32 累加**：bM = 256，bN = 192，bK = 128，2 stages，2 个 MMA warpgroup，cluster size (1, 2, 1)。
- **FP16 累加**：bM = 256，bN = 256，bK = 96，2 stages，4 个 MMA warpgroup，cluster size (1, 2, 1)。

矩阵用随机浮点数转 FP16 初始化，测得 TFLOP/s 如下（10 次迭代，5 次测量取平均）：

- **FP32 累加**：Multistage 477，WS 485。
- **FP16 累加**：Multistage 531，WS 536。

H100 PCIe GPU 上半精度 dense MMA 理论峰值为 750 TFLOP/s，因此在 FP32 累加的标准 setting 下，我们达到约 65% 的理论峰值。Multistage 与 WS kernel 均可在 [Colfax 的 github](https://github.com/ColfaxResearch/cfx-article-src/tree/master/pipeline-gemm) 获取。

需注意：CUTLASS Hopper GEMM 教程代码用随机 ±1 初始化矩阵，会导致性能虚高，参见 [此文](https://www.thonking.ai/p/strangely-matrix-multiplications)。例如，用 ±1 初始化时，我们 multistage kernel 的 FP16 累加性能会从约 530 膨胀到约 630 TFLOP/s。

作为对比，我们使用 CUTLASS profiler、10 次 profiling 迭代测得的 fastest CUTLASS FP16 Hopper GEMM kernel 为 630 TFLOP/s（约 84% 利用率）。（注：本文早期版本报告约 74%，因 profiling 迭代次数过多导致 H100 PCIe GPU 在 350W TDP 下热限速。）该数值对应以下 kernel：

```
cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_void_f16_128x256x64_2x1x1_0_tnn_align8_warpspecialized_cooperative_epi_tma
```

该 CUTLASS kernel 采用 [此处](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#warp-specialization) 描述的 “Warp-Specialized Persistent Cooperative” 设计。我们预计，实现 threadblock rasterization 以及能在 CTA 间重叠 prologue 与 epilogue 的 persistent kernel 后，当前流水线 GEMM 与 fastest GEMM 的差距会显著缩小。在更非常规的问题尺寸下，Stream-K 的负载均衡也会起作用。本算例中，Stream-K CUTLASS kernel 性能接近（625 TFLOP/s）。

关于 warpgroup 级寄存器重分配对 WS kernel 的作用：可用编译选项 `-Xptxas=--verbose` 查看寄存器使用。（注：该选项与 `--generate-code` 不兼容，需用 `--gencode`。）启用寄存器重分配时，寄存器使用量会固定为 warpgroup 数量的函数。例如 3 个 warpgroup 时：

```
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers
```

4 个 warpgroup 时：

```
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 128 registers
```

注意 168×3 = 504，128×4 = 512，即 producer 与 consumer 寄存器数之和需小于等于这些值（这也是 3 个 warpgroup 时 32/240/240 划分不可行的原因）。

另一方面，若原本寄存器用量就很低，重分配可能无实际影响。例如 FP16 累加时，去掉寄存器重分配后：

```
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers
```

且实测时间无明显变化。但 FP32 累加时：

```
    2784 bytes stack frame, 4764 bytes spill stores, 4760 bytes spill loads
ptxas info    : Used 168 registers
```

重测时间约为 21 TFLOP/s，性能急剧下降。不过，将 tuning 参数改为 (bM = 128, bN = 256, bK = 128, 2 stages, 2 MMA warpgroups, cluster (2,1,1))，可在无 spilling、无寄存器重分配的情况下获得几乎同样好的性能（460 TFLOP/s）。

最后，在 FlashAttention-3 等带有多个寄存器累加器的 fused WS kernel 中，寄存器重分配成为避免过多 spilling 的必要手段。

## 小结

本文介绍了流水线技术的全貌：其目标是通过 overlapping 内存 copy 与计算来掩盖延迟，以及为何这对性能至关重要。随后介绍了两种流水线设计：

- **Warp 特化**：将 warp 划分为 producer 与 consumer，并并发执行。producer 或 consumer 操作也可为异步（如 Hopper 上的 TMA 与 WGMMA）。
- **多阶段**：通过异步 copy（Hopper 上的 TMA 或 Ampere 上的 `cp.async`）加载下一批数据，同时计算当前批，从而掩盖数据传输。各 warp 兼具 producer 与 consumer 角色。

我们详细说明了如何使用 CUTLASS Pipeline 类管理 Hopper GEMM kernel 中实现这两种流水线策略所需的同步逻辑，并对两种 pipeline 在 GEMM 示例上做了对比。在简化 setting 下二者表现相当，实践中表现最好的 Hopper GEMM kernel 则采用 warp 特化（例如 [CUTLASS profiler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/profiler.md) 所示）。

本教程 Part 3 将讨论整体 kernel 的调度策略，包括 threadblock rasterization、persistent kernel，以及近年提出的 [Stream-K GEMM](https://arxiv.org/abs/2301.03598)。

## 附录：Ampere GEMM 的流水线

正文中讨论了使用 TMA 做内存搬运、WGMMA 做计算的流水线。二者均为 Hopper 架构（`sm90`）引入，在更早架构上不可用。要在旧架构上实现类似范式需额外步骤。为完整性，我们补充 Ampere 架构（`sm80`）上 GEMM 的流水线实现，参考 [CUTLASS 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu) 的 `sm80` 实现。与正文中 `sm90` 代码相比，Ampere 有两个主要区别：

- WGMMA 的两个操作数均可直接从 SMEM 读取，而此处 MMA 操作数需从寄存器（RMEM）加载。因此 MMA 运行前需从 SMEM load 到 RMEM 的指令。此外，可将 SMEM→RMEM 的 load 也做成流水线以进一步提升性能，这增加了整体设计复杂度。
- Ampere 有从 GMEM 到 SMEM 的异步 load 指令 [cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)，但无 warp 级别的寄存器分配控制。这使 warp 特化不太合适，更倾向于编写多阶段 pipeline，让每个 warp 兼具 producer 与 consumer 角色。

**图 3.** Ampere GEMM 通过两层嵌套流水线掩盖延迟。图来自 [CUTLASS 文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)。

图 3 展示了 kernel 的整体结构。（该图早于 Ampere，有一处与 Ampere 不符：使用 `cp.async` 时，“load global” 与 “store shared” 并非两个独立阶段，而是单条机器指令。）主循环每次迭代用 Ampere 的 `cp_async` 指令异步 load 后续 tile 从 GMEM 到 SMEM，并与当前 tile 上的计算重叠。此外层流水线与 Hopper 上的多阶段 pipeline 类似。内层 unroll 循环从 SMEM 将 tile 的连续片段 load 到 RMEM 并做运算。虽为同步操作，仍可通过 CPU 中称为（在此语境下有些混淆）[软件流水线（software pipelining）](https://en.wikipedia.org/wiki/Software_pipelining) 的技术减少延迟。

先看主循环前的 pre-fetch 阶段：

```cpp
TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},
                                  Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                  Layout<Shape< _1,_1>>{});              // Val layout  1x1
TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TB>, TB>{},
                                  Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                  Layout<Shape< _1,_1>>{});              // Val layout  1x1

// 待 copy 的 tile 数量
int k_tile_count = size<3>(tAgA);
// 当前要从 gmem 读取的 tile 索引
int k_tile_next = 0;

// 初始 load。为除最后一个以外的所有 pipe 发起异步 load
for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
  copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
  copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
  cp_async_fence();
  --k_tile_count;
  if (k_tile_count > 0) { ++k_tile_next; }
}

// 在继续前等待第一个 tile 就绪
cp_async_wait<K_PIPE_MAX-2>();
__syncthreads();
```

copy 通过 [CUTLASS cp_async API](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/memory_sm80.h)（封装 [cp.async PTX 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)）异步发起。相关方法说明：

- `cp_async_wait<N>()` 使 CTA 等待，直到最近发起的 commit group 中至多 N 个仍在执行。本例共发起 `K_PIPE_MAX-1` 个 commit group，故 `cp_async_wait<K_PIPE_MAX-2>()` 等价于等待最老的一组完成（即 A、B 的第 0 个 tile 完成 copy）。其他 commit group 可能更早完成，但该调用会一直等到最老的。
- 使用 `cp.async` 的 copy 通过 `cp_async_fence()` 划分为 “commit group”。本例中每个 commit group copy A 和 B 各一个 tile。

以下是 kernel 主循环，略去 SMEM→RMEM load 与计算，只保留 GMEM→SMEM pipeline：

```cpp
while (k_tile_count > -(K_PIPE_MAX-1)) {
  // 处理 tiled gemm 中的一个 block
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
    // 启动下一 tile 的异步 copy
    if (k_block == 0) {
      copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
      copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
      cp_async_fence();

      --k_tile_count;
      if (k_tile_count > 0) { ++k_tile_next; }
    }

    // 从 SMEM load block 到 RMEM（省略）

    if (k_block == K_BLOCK_MAX-1) {
      // 等待之前的 copy 完成
      cp_async_wait<K_PIPE_MAX-2>();
      __syncthreads();
    }

    // 在 block 上计算（省略）
  }
}
```

流程与 pre-fetch 阶段类似：每次外层循环开始时发起新一轮异步 copy，在循环末尾 CTA 等待下一所需 tile 的 copy。接近计算尾声时，GMEM 已无新 tile 可 copy，代码用 `k_tile_count <= 0` 表示，并发出未使用的 dummy copy。

注意该示例未使用 CUTLASS `Pipeline` 类，因为不需要 mbarrier 管理同步，而是手动在数据 buffer 间切换。内层循环遍历 buffer 大小，以跟踪使用哪个 buffer。细节虽不同，整体结构与正文中的简单 pipeline 一致。

最后看包含 SMEM→RMEM load 与 MMA 的内层循环。SMEM→RMEM 传输比 GMEM→SMEM 快得多，但访问延迟仍足以让将 load 与计算 overlapping 有利可图。概念与 GMEM→SMEM 相同：我们有额外 buffer（寄存器），对它们发起 load 指令，而计算在其它寄存器上进行。但与显式异步调用不同，这里依赖**软件流水线**。

**软件流水线**是通过去除连续高延迟指令之间的依赖来最大化硬件利用的优化技术。对我们而言，若 SMEM→RMEM load 与计算在硬件与数据上均独立，则可并发执行。SMEM 到 RMEM 的 load 由 LSU（Load/Store Unit）处理，计算由计算单元（如 Tensor Core）处理。虽无公开文档，普遍认为 [这些硬件单元可并发执行](https://forums.developer.nvidia.com/t/how-does-the-lsu-load-store-unit-execute-load-store-instructions-in-the-ampere-architecture/273699)，故硬件依赖不是问题。数据依赖则可能成问题。

考虑：

```cpp
for (i=0; i<N-1; i++) {
  load2rmem(i);
  compute(i);
}
```

问题在于 `compute(i)` 需要 load 的数据，因此要等 `load2rmem(i)` 完成才能开始，数据依赖使两者串行。与 GMEM→SMEM pipeline 类似，改为 load 下一个 buffer：

```cpp
load2rmem(0);
for (i=0;i<N-1; i++) {
  load2rmem(i+1);
  compute(i);
}
compute(N-1);
```

此时 load 与 compute 在数据和硬件上均无依赖，可并发执行。sm80 CUTLASS 示例中对应代码如下：

```cpp
CUTE_UNROLL
for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
  // Load A, B shmem->regs for k_block+1
  auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
  copy(tCsA_p(_,_,k_block_next), tCrA(_,_,k_block_next));
  copy(tCsB_p(_,_,k_block_next), tCrB(_,_,k_block_next));

  // Thread-level register gemm for k_block
  gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
}
```

其中 `tCrA` 与 `tCrB` 为由 CUTLASS `make_fragment` 创建的 RMEM 引用。copy 与 GEMM 可并发执行，因为访问的是不同的 `k_block`。

## 参考文献

[1] Michael Bauer, Henry Cook, and Bruce Khailany. 2011. "CudaDMA: optimizing GPU memory bandwidth via warp specialization." In Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis (SC '11). Association for Computing Machinery, New York, NY, USA, Article 12, 1–11. [https://doi.org/10.1145/2063384.2063400](https://doi.org/10.1145/2063384.2063400)

[2] Michael Bauer, Sean Treichler, and Alex Aiken. 2014. "Singe: leveraging warp specialization for high performance on GPUs". In Proceedings of the 19th ACM SIGPLAN symposium on Principles and practice of parallel programming (PPoPP '14). Association for Computing Machinery, New York, NY, USA, 119–130. [https://doi.org/10.1145/2555243.2555258](https://doi.org/10.1145/2555243.2555258)
