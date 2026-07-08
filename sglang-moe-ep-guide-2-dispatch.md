# SGLang DeepSeek MoE & Expert Parallelism 原理详解

文档基于 SGLang 代码库中 DeepSeek-V3 的 MoE 实现，以 `DeepseekV2MoE` 类为入口，按代码调用链逐层展开。每一章对应调用链的一层，读者从第一章开始读，相当于在逐层 step-into 代码。

重点覆盖 EP（Expert Parallelism）的两种 dispatch 模式：normal（高吞吐，常用于 prefill）和 low_latency（低延迟，常用于 decode，但也可用于 prefill），以及 TP（Tensor Parallelism）模式的 FP4 MoE 计算。通过具体的 tensor shape 示例说明数据流向。

代码基线：`sglang/python/sglang/srt/`（以下路径省略此前缀）

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
