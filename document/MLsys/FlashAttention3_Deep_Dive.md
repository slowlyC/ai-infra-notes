# FlashAttention-3 代码分析

前几篇文章[Gluon-0a-Attention-Forward](https://zhuanlan.zhihu.com/p/2011582362362864169) ，[FlashAttention 2 代码分析：Triton 与 CuTile 实现](https://zhuanlan.zhihu.com/p/2014101400808862224) 分析了 FA 的 Triton/CuTile/Gluon 实现，本文来分析 FlashAttention-3 的 CUTLASS 实现，主要基于 [flash-attention](https://github.com/Dao-AILab/flash-attention) commit `eacbc560`，以代码分析为主。原理部分可以见论文

> Jay Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision*, 2024.

## 1. FA3 简述

### 1.1 传统执行模式的问题

FlashAttention-2 中 GEMM 和 Softmax 串行执行：

```
时间 →

Tensor Core: ==== QK GEMM ====                    ==== PV GEMM ====
                              ↓                   ↑
SFU:                          ---- Softmax ----

Tensor Core 在 Softmax 期间完全空闲
```

### 1.2 FA3 的思路

让 Softmax 和 GEMM 重叠执行，用 Tensor Core 的计算时间掩盖 SFU 的低吞吐：

```
时间 →

Tensor Core: ==== QK_j ==== ==== PV_{j-1} ==== ==== QK_{j+1} ====
                                 ↑
SFU:              ------- Softmax_j -------
                  (与 PV_{j-1} 并行)
```

## 2. Hopper 架构特性

FA3 深度利用了 Hopper 的三个硬件特性：

### 2.1 TMA (Tensor Memory Accelerator)

硬件级异步内存搬运引擎，独立于 CUDA Core 完成多维地址计算和数据搬运。

```cpp
// 传统方式：CUDA Core 逐元素计算地址
for (int i = 0; i < N; i++) {
    int addr = base_ptr + i * stride_row + j * stride_col;
    smem[i][j] = gmem[addr];
}

// TMA：一条指令，硬件处理地址计算和搬运
copy(tma_load_K.with(barrier), gK, sK);
// CUDA Core 可以去做其他计算
```

TMA 的价值不仅是解放 CUDA Core，还在于它能直接配合 Pipeline barrier 机制——Producer 发起 TMA 后可以立即 commit，Consumer 通过 barrier 等待数据就绪。

### 2.2 WGMMA (Warpgroup Matrix Multiply-Accumulate)

以 Warpgroup (128 线程) 为单位的异步矩阵乘法指令。

WGMMA 的三个特性使 FA3 的 overlap 成为可能：

- **异步发射**：指令发出后立即返回，不阻塞后续指令
- **直接从 SMEM 读取**：一个操作数可以直接来自 Shared Memory（SS-GEMM），无需先加载到寄存器
- **细粒度等待控制**：`wgmma.wait_group N` 允许指定还有多少个 WGMMA 可以未完成

```
wgmma.wait_group 语义:
  wait_group 0  →  等待所有之前的 WGMMA 完成
  wait_group 1  →  最多允许 1 个 WGMMA 仍在执行
  wait_group 2  →  最多允许 2 个 WGMMA 仍在执行
```

在 CUTLASS 的 `flash::gemm<zero_init, wg_wait>` 封装中，`wg_wait = -1` 表示在 gemm 内部不调用 `warpgroup_wait`，由调用者自行选择等待时机。这是实现 QK/PV 两个 GEMM 重叠的基础。

### 2.3 动态寄存器重分配

运行时调整不同 Warpgroup 的寄存器配额。Hopper 引入了 `setmaxnreg` 指令，允许 warpgroup 粒度的寄存器重分配。

```cpp
// flash_fwd_kernel_sm90.h
if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();  // 保留 ~24 个
}
else {  // Consumer
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();     // 分配 ~240 个
}
```

```
重分配前后对比:

              默认          →  重分配后
  WG0 (Producer):  128 regs →   24 regs   (只管 TMA 调度)
  WG1 (Consumer):  128 regs →  240 regs   (存 FP32 累加器)
```

Producer 的任务是发起 TMA 搬运和管理 pipeline 状态，寄存器需求很小。Consumer 需要在寄存器中保存完整的 O_i 累加器（FP32 精度），以及 Softmax 的行统计量 (row_max, row_sum)，寄存器压力大。这种不对称的分配正好匹配两者的角色。

## 3. 整体架构：Producer-Consumer 模型

### 3.1 Thread Block 内部分工

一个 Thread Block 包含 2-3 个 Warpgroups，典型配置：

```
Warpgroup 0: Producer
  - 使用 TMA 从 HBM 加载 Q, K, V 到 Shared Memory
  - 管理 Pipeline 状态 (acquire/commit)
  - 实际可能只用 1 个 Warp (32 线程, SingleProducerWarp 模式)
  - 寄存器: ~24 个

Warpgroup 1 (2): Consumer(s)
  - 执行 QK^T GEMM (SS-GEMM: Shared × Shared)
  - 计算 Online Softmax
  - 执行 PV GEMM (RS-GEMM: Register × Shared)
  - 累加结果到寄存器中的 O_i
  - 寄存器: ~240 个 (存储 tOrO 累加器)
```

注：代码中 `SingleProducerWarp` 是编译期开关。启用时 Producer 只用 1 个 Warp，其余 3 个 Warp 提前退出，Thread Block 实际活跃线程为 128×NumConsumerWG + 32。

### 3.2 数据流与内存层次

```
数据流向

  HBM (Global Memory)
    Q: [batch, heads, seq_len, head_dim]
    K: [batch, heads, seq_len, head_dim]
    V: [batch, heads, seq_len, head_dim]
                           │
                           │ TMA (异步搬运)
                           ↓
  Shared Memory (~160 KB, 仅 Q/K/V 数据部分)
    sQ: [128 × 128] = 32 KB
    sK: [128 × 128] × 2 stages = 64 KB
    sV: [128 × 128] × 2 stages = 64 KB
    (另有 pipeline barrier、scheduler 等元数据)
                           │
                           │ WGMMA
                           ↓
  Registers
    tSrS: [128 × 128] FP32 = 64 KB (Scores, 临时)
    tOrO: [128 × 128] FP32 = 64 KB (Output 累加器)
    m_i:  [128] FP32 (每行最大值)
    ℓ_i:  [128] FP32 (Softmax 分母)
```

### 3.3 双缓冲 Pipeline 机制

```
2-Stage 循环缓冲

  Shared Memory 中的 K/V 使用双缓冲:

    Stage 0                    Stage 1
    ┌──────────────┐           ┌──────────────┐
    │ K_j (128×128)│           │K_{j+1}(128×128)│
    │ V_j (128×128)│           │V_{j+1}(128×128)│
    └──────────────┘           └──────────────┘
          ↑                          ↑
     Producer                     Producer
     写入偶数迭代                  写入奇数迭代
          ↓                          ↓
     Consumer                     Consumer
     读取偶数迭代                  读取奇数迭代

  Pipeline 同步:
    • Producer: acquire → 等待 stage 空闲
    • Producer: commit  → 通知 Consumer 数据就绪
    • Consumer: wait    → 等待数据就绪
    • Consumer: release → 通知 Producer 可以重用
```

对应代码中的 Pipeline API：

```cpp
// mainloop_fwd_sm90_tma_gmma_ws.hpp

// Pipeline 类型定义
using MainloopPipelineK = cutlass::PipelineTmaAsync<kStages>;
using MainloopPipelineV = cutlass::PipelineTmaAsync<kStages>;
using PipelineState = cutlass::PipelineState<kStages>;

// Producer 端
pipeline_k.producer_acquire(smem_pipe_write);  // 等待 stage 空闲
copy(tma_load_K.with(barrier), gK, sK);        // TMA 加载
pipeline_k.producer_commit(smem_pipe_write);   // 通知 Consumer

// Consumer 端
pipeline_k.consumer_wait(smem_pipe_read);      // 等待数据就绪
// ... 使用数据 ...
pipeline_k.consumer_release(smem_pipe_read);   // 通知 Producer 可以重用
```

K 和 V 使用独立的 pipeline（`pipeline_k`、`pipeline_v`），这样 Consumer 可以在用完 K 之后立即 release K 的 stage，而不必等 V 也用完。Producer 就能尽早开始加载下一轮的 K。

## 4. Algorithm 1：CTA 视角的完整流程

### 4.1 算法伪代码

```
Algorithm 1: FlashAttention-3 forward pass (CTA view)

Require: Q_i ∈ R^{B_r×d}, K,V ∈ R^{N×d}, block size B_c, T_c = ⌈N/B_c⌉

1:  Initialize pipeline with s-stage circular SMEM buffer
2:  if in producer warpgroup then
3:      Deallocate registers
4:      Issue load Q_i from HBM to SMEM
5:      Commit to notify consumer
6:      for j = 0 to T_c-1 do
7:          Wait for (j%s)th stage to be consumed
8:          Issue loads K_j, V_j to (j%s)th stage
9:          Commit to notify consumers
10:     end for
11: else  // consumer warpgroup
12:     Reallocate registers
13:     Initialize O_i = 0, ℓ_i = 0, m_i = -∞
14:     Wait for Q_i
15:     for j = 0 to T_c-1 do
16:         Wait for K_j
17:         Compute S_i^(j) = Q_i × K_j^T (SS-GEMM)
18:         Update m_i, compute P̃_i^(j), update ℓ_i
19:         Wait for V_j
20:         Update O_i = rescale(O_i) + P̃_i^(j) × V_j (RS-GEMM)
21:         Release (j%s)th stage
22:     end for
23:     Compute O_i = O_i / ℓ_i, L_i = m_i + log(ℓ_i)
24:     Write O_i, L_i to HBM
25: end if
```

### 4.2 代码对照：Producer 部分

以下是简化后的 Producer 逻辑骨架（省略了 AppendKV、SingleProducerWarp、rotary embedding 等分支）：

```cpp
// flash_fwd_kernel_sm90.h Line 308-359 (简化)
if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

    for (auto work_tile_info = scheduler.get_initial_work();
         work_tile_info.is_valid();
         work_tile_info = scheduler.get_next_work()) {

        // mainloop.load() 内部执行:
        //   pipeline_k.producer_acquire()  → 等待 stage 空闲
        //   copy(tma_load_K, gK, sK)       → TMA 异步搬运
        //   pipeline_k.producer_commit()   → 通知 Consumer 数据就绪
        //   (对 V 同理)
        mainloop.load(params, pipeline_k, pipeline_v, pipeline_vt,
                      smem_pipe_write, shared_storage, ...);
    }
    mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, ...);
}
```

### 4.3 代码对照：Consumer 部分

```cpp
// flash_fwd_kernel_sm90.h Line 360-452 (简化)
else {  // Consumer
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

    TiledMmaPV tiled_mma_pv;
    Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0,1>(TileShape_MNK_PV{}));
    // Softmax 构造时 row_max 默认 -inf, row_sum 默认 0
    flash::Softmax softmax(softmax_scale_log2);

    for (auto work_tile_info = scheduler.get_initial_work();
         work_tile_info.is_valid(); ) {

        // mainloop.mma() 内部执行完整的 K/V 遍历:
        //   等待 K_j → QK GEMM → Softmax → 等待 V_j → PV GEMM → release buffer
        tile_valid = mainloop.mma(params, pipeline_k, pipeline_v,
                                  smem_pipe_read, tOrO, softmax, ...);

        // get_next_work 在 epilogue 之前调用，让 scheduler prefetch 下一个 tile
        work_tile_info = scheduler.get_next_work(params.scheduler, work_tile_info);

        // 归一化 O_i 并写回 HBM (包括 LSE = m_i + log(ℓ_i) 用于 backward)
        epilogue.store(params, tOrO, softmax.row_sum, ...);
    }
}
```

## 5. Algorithm 2：Consumer Warpgroup 计算流程

### 5.1 算法伪代码

```
Algorithm 2: FlashAttention-3 consumer warpgroup forward pass

1:  Reallocate registers
2:  Initialize O_i = 0 ∈ R^{B_r×d}, ℓ_i = 0, m_i = -∞ ∈ R^{B_r}
3:  Wait for Q_i and K_0
4:  Compute S_cur = Q_i × K_0^T using WGMMA. Commit and wait.
5:  Release the 0th stage buffer for K.
6:  Compute m_i, P̃_cur, ℓ_i based on S_cur, and rescale O_i.

7:  for 1 ≤ j < T_c - 1 do
8:      Wait for K_j
9:      Compute S_next = Q_i × K_j^T using WGMMA. Commit but do not wait.
10:     Wait for V_{j-1}
11:     Compute O_i = O_i + P̃_cur × V_{j-1} using WGMMA. Commit but do not wait.
12:     Wait for the WGMMA Q_i × K_j^T.
13:     Compute m_i, P̃_next, ℓ_i based on S_next.
14:     Wait for the WGMMA P̃_cur × V_{j-1} and then rescale O_i.
15:     Release the (j%s)th stage for K, (j-1%s)th stage for V.
16:     Copy S_next to S_cur.
17: end for

18: Wait for V_{T_c-1}
19: Compute O_i = O_i + P̃_last × V_{T_c-1} using WGMMA. Commit and wait.
20: Epilogue: Rescale O_i, compute L_i, write O_i and L_i to HBM.
```

### 5.2 代码对照

以下是 `mainloop_fwd_sm90_tma_gmma_ws.hpp` 中 `IntraWGOverlap` 分支的代码骨架，注释中标注了对应 Algorithm 2 的行号。

**Prologue（第 0 次迭代，Line 1094-1166）**

```cpp
// mainloop_fwd_sm90_tma_gmma_ws.hpp, mma() 函数内

// [Algo Line 1-2] 寄存器重分配在 flash_fwd_kernel_sm90.h; 初始化 O_i
clear(tOrO);

// [Algo Line 3] 等待 Q 和 K_0
barrier_Q.wait(work_idx % 2);
consumer_wait(pipeline_k, smem_pipe_read);

// [Algo Line 4] 第一次 QK GEMM, 同步等待
flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(...), tSrS);
warpgroup_wait<0>();

// [Algo Line 5] 释放 K buffer
pipeline_k.consumer_release(smem_pipe_read);

// [Algo Line 6] 第一次 Softmax
scoremod_premask_fn(tSrS);
mask.apply(tSrS, m_block, n_block);
scores_scale = softmax.max_get_scale<true, true>(tSrS);
softmax.online_softmax<true, true>(tSrS);
```

**Mainloop（fwd_step lambda, Line 1170-1207, 对应 Algorithm 2 Line 7-17）**

```cpp
auto fwd_step = [&](int n_block, auto mask_fn, auto check_inf_type) {
    PipelineState smem_pipe_read_v(smem_pipe_read.index(), ...);
    ++smem_pipe_read;
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, ...);

    // [Algo Line 8] 等待 K_j
    consumer_wait(pipeline_k, smem_pipe_read);

    // [Algo Line 9] QK GEMM (异步, 不等待)
    flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(...), tSrS);

    // [Algo Line 10] 等待 V_{j-1}
    consumer_wait(pipeline_v, smem_pipe_read_v);

    // [Algo Line 11] PV GEMM (异步, 不等待)
    flash::gemm<false, -1>(tiled_mma_pv, tOrP, tOrV(...), tOrO);

    // [Algo Line 12] 等待 QK 完成 (PV 继续执行)
    warpgroup_wait<1>();
    pipeline_k.consumer_release(smem_pipe_read);

    // [Algo Line 13] Softmax (与 PV GEMM 并行执行)
    scoremod_premask_fn(tSrS);
    mask_fn(tSrS, n_block);
    softmax.max_get_scale<false, Check_inf>(tSrS);
    softmax.online_softmax<false, Check_inf>(tSrS);

    // [Algo Line 14] 等待 PV 完成
    warpgroup_wait<0>();
    // [Algo Line 15] 释放 V buffer
    pipeline_v.consumer_release(smem_pipe_read_v);

    // [Algo Line 16] P 类型转换 + rescale O
    convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
    softmax.rescale_o(tOrO, scores_scale);
};
```

**Epilogue（最后一次迭代，Line 1239-1250）**

```cpp
// [Algo Line 18] 等待最后的 V
consumer_wait(pipeline_v, smem_pipe_read);

// [Algo Line 19] 最后的 PV GEMM
flash::gemm<false, -1>(tiled_mma_pv, tOrP, tOrV(...), tOrO);

// [Algo Line 20] 归一化 + 写回
scores_scale = softmax.finalize(v_descale);
warpgroup_wait<0>();
pipeline_v.consumer_release(smem_pipe_read);
softmax.rescale_o(tOrO, scores_scale);
// 随后 epilogue.store() 写回 O 和 LSE 到 HBM
```

## 6. 优化一：Intra-warpgroup Overlap

### 6.1 原理：利用 warpgroup_wait1 实现重叠

在一个 fwd_step 迭代内，依次发射两个 WGMMA（QK 和 PV），然后用 `warpgroup_wait<1>` 只等前一个完成：

```
发射 QK GEMM (wg_wait=-1) → 发射 PV GEMM (wg_wait=-1) → warpgroup_wait<1>
                                                                ↓
                                                          QK 完成, PV 仍在执行
                                                                ↓
                                                    此时 SFU 可以做 Softmax, 与 PV 并行
```

### 6.2 代码实现

```cpp
// mainloop_fwd_sm90_tma_gmma_ws.hpp Line 1170-1207

auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
    
    PipelineState smem_pipe_read_v(smem_pipe_read.index(), ...);
    ++smem_pipe_read;
    
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, ...);
    
    // Step 1: 发起当前迭代的 QK GEMM (异步)
    if (!UseSchedulerBarrier || warp_group_idx == 0) {
        consumer_wait(pipeline_k, smem_pipe_read);
    }
    warp_scheduler_barrier_sync();

    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
        tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

    // Step 2: 可选的提前 rescale (RescaleOBeforeGemm 路径)
    if constexpr (RescaleOBeforeGemm) {
        softmax.rescale_o(tOrO, scores_scale);
    }

    // Step 3: 发起上一迭代的 PV GEMM (异步)
    if (!UseSchedulerBarrier || warp_group_idx == 0) {
        consumer_wait(pipeline_v, smem_pipe_read_v);
    }

    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
        tiled_mma_pv,
        cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP),
        tOrV(_, _, _, smem_pipe_read_v.index()),
        tOrO);
    // 此时 QK 和 PV 两个 WGMMA 都已发射到 Tensor Core

    warp_scheduler_barrier_arrive();

    // Step 4: 等待 QK 完成，PV 继续执行
    warpgroup_wait<1>();
    pipeline_k.consumer_release(smem_pipe_read);

    // Step 5: Softmax (SFU 计算 exp2f, 与 PV GEMM 并行)
    scoremod_premask_fn(tSrS);
    mask_fn(tSrS, n_block);

    cute::copy(softmax.template max_get_scale<false, Check_inf>(tSrS),
               scores_scale);
    softmax.template online_softmax<false, Check_inf>(tSrS);

    // Step 6: 等待 PV 完成, 释放 V buffer
    warpgroup_wait<0>();
    pipeline_v.consumer_release(smem_pipe_read_v);

    // Step 7: 类型转换 + rescale, 准备下一轮
    convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
    if constexpr (!RescaleOBeforeGemm) {
        softmax.rescale_o(tOrO, scores_scale);
    }
};
```

### 6.3 时序图

```
Intra-warpgroup Overlap 时序

 === Prologue (第 0 次迭代, 无重叠) ===

 Tensor Core: ==== QK_0 ====
                        ↓ wg_wait<0>
 SFU:                   --- Softmax_0 ---

 === Mainloop (第 j 次迭代, j >= 1) ===

 Tensor Core: ==== QK_j ==== ==== PV_{j-1} ====
                         ↓                  ↓
                   wg_wait<1>          wg_wait<0>
                   (QK 完成)           (PV 完成)
                         ↓
 SFU:               --- Softmax_j ---
                    (与 PV_{j-1} 并行)
```

要点：

- Prologue 只做 QK_0 + Softmax_0，此时没有上一轮的 PV 可以重叠
- Mainloop 中 QK_j 先发射，紧接着发射 PV_{j-1}，两者在 Tensor Core 上流水
- `warpgroup_wait<1>` 等 QK_j 完成后，开始 Softmax_j；PV_{j-1} 仍在 Tensor Core 执行
- SFU (Softmax) 和 Tensor Core (PV) 并行，SFU 的低吞吐被 Tensor Core 的计算时间掩盖

### 6.4 RescaleOBeforeGemm 优化

代码中有一个 `RescaleOBeforeGemm` 编译期开关，控制 `rescale_o` 的执行时机：

```
默认路径 (!RescaleOBeforeGemm):
  QK GEMM → PV GEMM → wait<1> → Softmax → wait<0> → rescale_o → ...

RescaleOBeforeGemm 路径:
  QK GEMM → rescale_o → PV GEMM → wait<1> → Softmax → wait<0> → ...
```

把 rescale_o 提前到 PV GEMM 之前有个好处：rescale_o 只是逐元素乘法（CUDA Core），可以在等待 V 数据就绪的间隙执行。PV GEMM 的 `zero_init=false` 意味着它会累加到 tOrO 上，如果 tOrO 在 GEMM 期间被 rescale 修改就会出错。提前 rescale 避免了这个 data hazard。

## 7. 优化二：Inter-warpgroup Overlap

### 7.1 问题：多个 Consumer WG 的 Tensor Core 争抢

当有 2 个 Consumer Warpgroups 时，各自独立处理不同的 K/V block。如果两个 WG 同时发射 WGMMA，会争抢 Tensor Core 资源，导致两者都变慢。

### 7.2 解决方案：Pingpong 调度

```
Inter-warpgroup Pingpong 调度

 时间 →   T0    T1    T2    T3    T4    T5    T6    T7    T8    T9

 WG1:    ==== QK_0 ==== → arrive(bar2)
                    │
                    │     --- Softmax_0 ---
                    │            │
                    │      sync(bar1) ← 等待 WG2 的 QK_1 完成
                    │            │
                    │            └→ ==== PV_0 ==== → arrive(bar2)
                    │                        │
                    │                        │     ==== QK_2 ==== → ...
                    │                        │
 WG2:              sync(bar2) ← 等待 WG1 的 QK_0 完成
                         │
                         └→ ==== QK_1 ==== → arrive(bar1)
                                       │
                                       │     --- Softmax_1 ---
                                       │            │
                                       │      sync(bar2) ← 等待 WG1 的 QK_2
                                       │            │
                                       │            └→ ==== PV_1 ====

 TC 使用: ========      ========      ========      ========
          (WG1 QK)      (WG2 QK)      (WG1 PV)      (WG2 PV)

 SFU 使用:        --------        --------
                  (WG1 SM)        (WG2 SM)

 效果: TC 和 SFU 始终被充分利用，没有空闲！
```

### 7.3 同步代码实现

```cpp
// mainloop_fwd_sm90_tma_gmma_ws.hpp Line 914-931

CUTLASS_DEVICE void
warp_scheduler_barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
        // 等待另一个 WG 发来的信号
        // WG1 等待 bar1, WG2 等待 bar2
        cutlass::arch::NamedBarrier::sync(
            2 * cutlass::NumThreadsPerWarpGroup, 
            static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 
            flash::canonical_warp_group_idx_nosync());
    }
}

CUTLASS_DEVICE void
warp_scheduler_barrier_arrive() {
    if constexpr (UseSchedulerBarrier) {
        // 向另一个 WG 发送信号
        // WG1 发送到 bar2 (给 WG2), WG2 发送到 bar1 (给 WG1)
        int const cur_WG = flash::canonical_warp_group_idx_nosync() - 1;
        int const next_WG = NumMmaWarpGroups == 2 
            ? 1 - cur_WG 
            : (cur_WG < NumMmaWarpGroups - 1 ? cur_WG + 1 : 0);
        cutlass::arch::NamedBarrier::arrive(
            2 * cutlass::NumThreadsPerWarpGroup, 
            static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + next_WG);
    }
}
```

### 7.4 为什么 WG1 的 PV 要等待 WG2 的 QK？

```
Pingpong 同步逻辑:

  WG1 执行完 QK_0 后:
    → arrive(bar_WG2)          // 通知 WG2 可以开始 GEMM
    → Softmax_0 (SFU)          // 自己去做 Softmax
    → sync(bar_WG1)            // 等 WG2 的信号, 再开始 PV_0

  WG2 收到 WG1 的 arrive 后:
    → QK_1 (Tensor Core)
    → arrive(bar_WG1)          // 通知 WG1 可以开始 PV_0
    → Softmax_1 (SFU)
    → sync(bar_WG2)            // 等 WG1 的信号
```

任何时刻只有一个 WG 在使用 Tensor Core，另一个 WG 在做 Softmax。Tensor Core 和 SFU 始终各被一个 WG 占用，没有空闲也没有争抢。

注意 Intra-WG Overlap 和 Inter-WG Overlap 并不互斥——代码中 `UseSchedulerBarrier` 控制是否启用 Inter-WG Pingpong，而 `IntraWGOverlap` 控制单个 WG 内部的 QK/PV 流水。两者可以同时生效。

## 8. Online Softmax 数学原理

### 8.1 标准 Softmax 需要两次遍历

标准 Softmax 需要先遍历一次算 max 和 sum，再遍历一次算输出。对 FlashAttention 的分块场景来说，这意味着需要把所有 K block 过两遍——显然不可行。

### 8.2 Online Softmax：边算边更新

Online Softmax 每处理一个新的 K block 就增量更新统计量：

```
第 j 个 block 的更新:
  m_new = max(m_old, max(S_block))           // 更新行最大值
  α = exp(m_old - m_new)                     // 修正因子
  ℓ_new = α × ℓ_old + Σ exp(S_block - m_new) // 更新分母
  O_new = α × O_old + P̃_block × V_block      // 更新累加器
  其中 P̃_block = exp(S_block - m_new)
```

修正因子 α 的作用：当新 block 出现更大的值时，之前累加的 O_old 和 ℓ_old 都需要乘以 α 来"补偿"——因为之前是在旧的 max 下算的 exp，现在 max 变大了，之前所有的 exp 值都需要等比缩小。

### 8.3 代码实现

实际代码将 Online Softmax 拆成两步调用：`max_get_scale` 计算新 max 和修正因子，`online_softmax` 应用 exp 并累加 row_sum。以下是简化后的逻辑（完整代码见 `hopper/softmax.h`）：

```cpp
// softmax.h: max_get_scale — 更新 row_max, 计算 scores_scale
template<bool Is_first, bool Check_inf>
TensorT max_get_scale(Tensor& acc_s) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    TensorT scores_scale;
    if constexpr (Is_first) {
        reduce_max<true>(scores, row_max);
        fill(scores_scale, 1.f);  // 第一次不需要 rescale
    } else {
        Tensor scores_max_prev = make_fragment_like(row_max);
        copy(row_max, scores_max_prev);
        reduce_max<false>(scores, row_max);  // 与当前 block 取 max
        for (int mi = 0; mi < size(row_max); ++mi) {
            // α = exp2(( m_old - m_new ) * scale_log2)
            scores_scale(mi) = exp2f((scores_max_prev(mi) - row_max(mi)) * softmax_scale_log2);
            row_sum(mi) *= scores_scale(mi);  // ℓ_new = α × ℓ_old
        }
    }
    return scores_scale;
}
```

```cpp
// softmax.h: online_softmax — 计算 exp2 并累加 row_sum
template<bool Is_first, bool Check_inf>
void online_softmax(Tensor& acc_s) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    // P̃ = exp2( (S - m) * scale_log2 ), 原地写回 acc_s
    scale_apply_exp2(scores, row_max, softmax_scale_log2);
    // row_sum += Σ P̃ (thread-local reduce, warp reduce 延迟到 finalize)
    reduce_sum<Is_first, /*warp_reduce=*/false>(scores, row_sum);
}
```

```cpp
// softmax.h: rescale_o — O_new = α × O_old
void rescale_o(Tensor& acc_o, TensorT const& scores_scale) {
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi)
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
            acc_o_rowcol(mi, ni) *= scores_scale(mi);
}
```

**为什么用 exp2f 而不是 expf？** 代码中统一使用以 2 为底的 `exp2f`，配合预先计算好的 `softmax_scale_log2 = softmax_scale × log₂(e)`。这是因为 GPU 的 SFU 原生支持 `ex2.approx.f32` 指令（以 2 为底的快速近似），而 `expf` 需要额外的乘法来转换底数。用 exp2f 还有一个好处：编译器可以把 `x * scale - max * scale` 融合成一条 `ffma` 指令。

### 8.4 finalize：最终归一化

当所有 K/V block 处理完后，`finalize` 完成最终的归一化：

```cpp
// softmax.h: finalize
TensorT finalize(float const final_scale = 1.f) {
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);  // warp 内 reduce
    TensorT scores_scale;
    for (int mi = 0; mi < size(row_sum); ++mi) {
        float sum = row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
        scores_scale(mi) = inv_sum * final_scale;
        // 存储 LSE = m + ln(ℓ), 用于 backward pass
        row_sum(mi) = (sum == 0.f || sum != sum)
            ? -INFINITY
            : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
    }
    return scores_scale;
}
```

这里有几个细节值得注意：

- `row_sum` 的 warp reduce（`quad_allreduce_`）被延迟到 finalize，而不是每个 block 都做。这是因为 Online Softmax 只需要 thread-local 的 row_sum 来累加，跨线程的 reduce 只在最终归一化时才需要
- `sum != sum` 是 NaN 检测（IEEE 754 中 NaN 不等于自身），处理全 mask 场景
- `row_sum` 被复用：finalize 之后存的不再是 Σexp，而是 LSE (log-sum-exp)，backward pass 需要这个值

## 9. 补充

### 9.1 张量命名约定

CuTe/CUTLASS 中的变量命名有约定：


| 前缀  | 含义                   | 示例                             |
| --- | -------------------- | ------------------------------ |
| `t` | Tensor (partitioned) | `tOrO` = tensor O, partitioned |
| `s` | Shared Memory        | `sQ` = Q in smem               |
| `g` | Global Memory        | `gK` = K in gmem               |
| `r` | Register             | `tSrS` 中的 `r`                  |


变量名读法举例：`tSrS` = **t**ensor **S**cores in **r**egisters for **S**;  `tOrO` = **t**ensor **O**utput in **r**egisters for **O**。第一个字母 `t` 表示这是经过 `partition_fragment_C` 等函数分区后的 tensor，每个线程只持有其中一部分。 
`tOrO` 累加器为例进行详解

```cpp
// flash_fwd_kernel_sm90.h Line 418

Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));
```

```
tOrO 结构:

  名称: tOrO = Tensor O in Registers for Output

  逻辑形状: [128 × 128] (B_r × d)
  数据类型: FP32 (高精度累加)
  存储位置: 寄存器 (分布在 256 个 Consumer 线程中)

  生命周期:
    1. 初始化为 0
    2. 每次迭代: O_new = α × O_old + P̃ × V
    3. 最终: O = O / ℓ (归一化)
    4. 写回 HBM

  分布方式:
    每个线程持有若干个 8×8 的小块 (由 WGMMA 指令决定)
```

### 9.2 FP8 低精度支持

FA3 的一个特性是支持 FP8 精度的 QK/PV GEMM。代码中通过 `Is_FP8` 编译期开关控制，涉及以下处理：

- **Descale**：FP8 量化后需要乘以 descale 因子恢复精度。代码中 `q_descale × k_descale` 被乘到 `softmax_scale_log2` 中，统一在 Softmax 阶段处理，避免额外的 rescale 步骤
- **Max_offset**：`softmax.h` 中的 `Max_offset` 模板参数（FP8 时为 8）。计算 exp2f 时减去 `max_offset=8`，使输出范围从 [0, 1] 扩展到 [0, 256]，更好地利用 FP8 的表示范围，减少 underflow
- **Permute**：FP8 WGMMA 对寄存器布局有特殊要求，`permute_Cregs_fp8` 和 `permute_Aregs_fp8` 负责在 GEMM 前后调整寄存器排列

