# Blackwell TMA 的 CUTLASS 教程示例

## TMA V0: 理解 tma_partition, TMA 操作的基础

这个示例用 CuTe DSL 在 NVIDIA Blackwell (SM100) 架构上演示 Tensor Memory Accelerator (TMA) 操作的基本组成部分。重点是理解 `tma_partition` 接口, 它是所有 TMA 操作都会遇到的基础接口。

### 基本概念

*   **TMA Load (Global → Shared)**: 使用 TMA 硬件把 Global Memory 中的大块数据异步搬到 Shared Memory。
*   **TMA Store (Shared → Global)**: 把 Shared Memory 中的大块数据异步写回 Global Memory。
*   **tma_partition**: 按照 TMA atom 的布局要求重新划分 tensor, 为 TMA 操作准备 source/destination view。
*   **group_modes**: 把多个 tensor mode 合并成一个 mode, 用来定义 TMA atom 的形状。正确使用 `tma_partition` 时这一步很重要。
*   **mbarrier Synchronization**: 用硬件 barrier（`mbarrier_init`、`mbarrier_arrive`、`mbarrier_wait`）同步异步 TMA 操作。

### Kernel 架构

`Sm100SimpleCopyKernel` 执行一个简单的 tile-based copy, 用来说明 TMA 的基本流程。

#### 配置

*   **tile_shape**: 固定为 (128, 128), 表示 tile 的 (M, N) 尺寸。
*   **cluster_shape_mn**: 固定为 (1, 1), 表示单 CTA 执行, 不使用 cluster 并行。
*   **Shared Memory**: 单个 buffer, 大小正好容纳一个 tile（`tile_m × tile_n` 个元素）。
*   **Synchronization**: 一个 mbarrier, 用于等待 TMA load 完成。

#### 主要组件

1.  **TMA Descriptor Creation**: 创建封装 TMA 硬件指令的 TMA atoms（`tma_atom_src`、`tma_atom_dst`）。
2.  **Shared Memory Layout**: 为了简化, 使用 row-major layout `(tile_m, tile_n):(tile_n, 1)`。
3.  **Barrier Management**: 用一个 barrier 协调 TMA load 完成后再继续处理。

### 执行流程

1.  **Initialization**:
    *   为一个 tile 分配 shared memory buffer。
    *   使用 `elect_one()` 初始化 mbarrier, 保证同步语义正确。
    *   设置 barrier 预期的 TMA transaction 字节数（`tile_m × tile_n × element_size`）。
    *   barrier 初始化完成后同步所有线程。

2.  **Tensor Preparation**:
    ```
    把 global tensor 切成 (tile_m, tile_n) blocks
    使用 group_modes 把 tile 维度合并为 Mode 0, 也就是 TMA atom
    
    示例: gSrc_tiled 的 shape 为 (128, 128, 4, 2)
    group_modes(_, 0, 2) 之后: ((128, 128), 4, 2)
                                  └───┬────┘  └─┬─┘
                                   Mode 0    其余 modes
    ```

3.  **TMA Partition**:
    ```python
    tAsA, tAgA = tma_partition(
        tma_atom_src,          # TMA 操作 atom
        cta_id=0,              # cluster 内的 CTA ID
        cta_layout,            # cluster layout
        group_modes(smem_tensor, 0, 2),   # SMEM view, Mode 0 = atom
        group_modes(gSrc_tiled, 0, 2)     # Global view, Mode 0 = atom
    )
    # 返回:
    # tAsA: 带有 TMA 内部 layout 的 SMEM view
    # tAgA: Global view, shape 为 ((TMA_Layout), grid_m, grid_n)
    ```

4.  **Tile Selection**:
    ```python
    # 为当前 CTA 选择对应 tile
    tAgA_cta = tAgA[(None, bidx, bidy)]
    # None: 保留完整 TMA atom, 也就是 Mode 0
    # bidx, bidy: grid 维度上的索引
    ```

5.  **TMA Load** (Global → Shared):
    ```python
    cute.copy(tma_atom_src, tAgA_cta, tAsA, tma_bar_ptr=barrier_ptr)
    # 在 barrier 上 arrive, producer 表示已发起操作
    # 等待 barrier, 所有线程等待 TMA 完成
    ```

6.  **TMA Store** (Shared → Global):
    ```python
    cute.copy(tma_atom_dst, tAsA, tBgB_cta)
    # 同步 store 会在 kernel 退出前完成
    ```

### 理解 tma_partition

`tma_partition` 是 TMA 操作的入口。这个教程在代码里加入了比较完整的内联图示, 用来说明：

*   **Input Tensor Shapes**: 划分前 tensor 是怎样组织的。
*   **group_modes Effect**: mode grouping 怎样构造 TMA atom 需要的结构。
*   **Partition Output**: TMA 操作拿到的 partitioned tensor 结构。
*   **Indexing Pattern**: 怎样从 partitioned view 中选择当前 CTA 对应的 tile。

详细图示见 `tma_v0.py` 约 155-255 行。

### 配置参数

*   `tile_shape`: 固定为 (128, 128), 表示 tile 的 (M, N) 尺寸。
*   `cluster_shape_mn`: 固定为 (1, 1), 表示单 CTA 执行。
*   `threads_per_cta`: 32, 一个 warp, 所有线程都参与 barrier。
*   `buffer_align_bytes`: 1024, shared memory 对齐字节数。

### 用法

用自定义矩阵尺寸运行 copy kernel：

```bash
# 基本用法, 默认 512×128 matrix
python tma_v0.py

# 自定义尺寸
python tma_v0.py --M 1024 --N 2048

# 自定义 benchmark 迭代次数
python tma_v0.py --M 4096 --N 4096 --num_warmup 10 --num_iters 50
```

#### 示例代码

```python
from tma_v0 import run_tma_copy

# 在 1024×2048 matrix 上运行 copy
run_tma_copy(M=1024, N=2048)
# 输出性能指标和校验结果
```

### 性能注意点

*   **Single-stage operation**: 没有 pipeline, 只是简单的 load → store 顺序执行。
*   **Educational focus**: 目标是理解 TMA 基础, 不追求峰值性能。
*   **Barrier synchronization**: 演示 mbarrier 的正确使用模式。
*   **V1/V2 基础**: 这里的概念是理解 V1/V2 multi-stage pipeline 和 warp specialization 的基础。

## TMA V1: 使用 Producer-Consumer 模式做矩阵转置

这个示例在 NVIDIA Blackwell (SM100) 架构上实现基于 TMA 的矩阵转置, 并使用 mbarrier 做 producer-consumer 同步。

### 基本概念

*   **Producer-Consumer 模式**: 不同 warp 通过 mbarrier 协调工作。
*   **TMA Operations**: 在 Global Memory 和 Shared Memory 之间异步搬运大块数据。
*   **Shared Memory Swizzle**: 使用优化后的 shared memory layout, 避免转置时的 bank conflict。
*   **Warp Specialization**: 把加载、转置、存储分别交给不同 warp。
*   **mbarrier Synchronization**: 使用硬件 barrier 协调异步操作。

### Kernel 架构

`Sm100MatrixTransposeKernel` 执行 tiled matrix transpose（M×N → N×M）, 整体设计如下。

#### Warp 角色

1.  **TMA Load Warp**（Warp 4, Producer）: 发起 TMA load, 把 Global Memory 中的数据加载到 Shared Memory buffer `sA`。
2.  **Transpose Warps**（Warps 0-3, 4 个 warp, Consumer/Producer）:
    *   等待 TMA load 填充 `sA`（作为 `load_mbar` 的 consumer）。
    *   执行 `sA` → Registers → `sB` 的转置搬运。
    *   向 TMA Store warp 发信号, 表示 `sB` 已准备好（作为 `store_mbar` 的 producer）。
3.  **TMA Store Warp**（Warp 5, Consumer）:
    *   等待 `sB` 准备完成（作为 `store_mbar` 的 consumer）。
    *   发起 TMA store, 把 `sB` 写回 Global Memory。

#### 同步 barrier

kernel 使用两个 mbarrier 完成 producer-consumer 协调。

1.  **load_mbar_ptr**: 同步 TMA Load → Transpose Warps。
    *   Producer: TMA Load Warp, 在 TMA 完成后 arrive。
    *   Consumer: Transpose Warps, 读取 `sA` 前等待。
    *   Expected arrivals: 1, 来自 TMA load warp。
    *   Expected transactions: `tile_m × tile_n × element_size` 字节。

2.  **store_mbar_ptr**: 同步 Transpose Warps → TMA Store。
    *   Producer: Transpose Warps, 每个 warp 写完 `sB` 后 arrive。
    *   Consumer: TMA Store Warp, 发起 TMA store 前等待。
    *   Expected arrivals: 4, 每个 transpose warp 一次。

#### 执行流程

1.  **Initialization**:
    *   所有 warp 参与 barrier 初始化, 由 thread 0 完成实际初始化。
    *   分配两个 shared memory buffer: `sA`（row-major）和 `sB`（column-major）。
    *   为 source 和转置后的 destination 创建 TMA descriptor。
    *   初始化 `load_mbar`, 设置 expected count 为 1, 并设置 transaction bytes。
    *   初始化 `store_mbar`, expected count 为 4, 也就是 transpose warp 的数量。

2.  **TMA Load Warp**（load pipeline 的 producer）:
    ```
    按 tile shape 划分 source tensor
    发起 TMA load: Global[block_tile] → sA
    在 load_mbar 上 arrive, 表示数据可用
    ```

3.  **Transpose Warps**（load 的 consumer, store 的 producer）:
    ```
    等待 load_mbar, 直到 TMA load 完成
    
    划分 sA 供读取, 每个线程处理其中一部分
    copy data: sA → Registers
    划分 sB 供写入
    copy data: Registers → sB, 转置由 layout 完成
    
    fence, 保证 smem 写入对后续操作可见
    使用 trans_sync_barrier 同步
    [选出一个线程] 在 store_mbar 上 arrive, 表示完成
    ```

4.  **TMA Store Warp**（store pipeline 的 consumer）:
    ```
    等待 store_mbar, 直到转置完成
    按 tile shape 划分 destination tensor
    发起 TMA store: sB → Global[block_tile]
    ```

### 主要特性

*   **Simple Producer-Consumer Model**: 不同 warp 分工清晰, 逻辑容易跟踪。
*   **Efficient Synchronization**: 使用硬件 mbarrier 降低同步开销。
*   **Memory Layout Optimization**: swizzled layout 用于避免 bank conflict。
*   **Transposition via Layouts**: 通过 `sA` 和 `sB` 的不同 memory layout 完成转置。

### 配置参数

*   `tile_shape`: 固定为 (128, 128), 表示 tile 的 (M, N) 尺寸。
*   `cluster_shape_mn`: 固定为 (1, 1), 表示单 CTA 执行。
*   Warp count: 6 个 warp（1 个 TMA Load + 4 个 Transpose + 1 个 TMA Store）。

### 用法

用自定义矩阵尺寸运行 transpose kernel：

```bash
# 基本用法, 128×128 matrix
python tma_v1.py

# 自定义尺寸
python tma_v1.py --M 1024 --N 2048

# 自定义 benchmark 迭代次数
python tma_v1.py --M 4096 --N 4096 --num_warmup 10 --num_iters 50
```

#### 示例代码

```python
from tma_v1 import run_transpose

# 在 1024×2048 matrix 上运行 transpose
run_transpose(M=1024, N=2048)
# 输出性能指标和校验结果
```

### 性能注意点

*   **Single-stage pipeline**: 比 multi-stage 更简单, 但可能让部分 warp 空闲。
*   **Warp specialization**: 明确的角色划分可以降低同步复杂度。
*   **Good for learning**: 适合用来理解 TMA 和 mbarrier 的基本概念。

## TMA V2: 使用 Multi-Stage Pipeline 做矩阵转置

这个示例在 NVIDIA Blackwell (SM100) 架构上展示带 multi-stage pipelining 的 TMA 矩阵转置实现, 用 pipeline 重叠 load、transpose 和 store。

### 基本概念

*   **Multi-Stage Pipeline**: 使用多个 buffer 重叠 TMA loads、计算（transpose）和 TMA stores, 以隐藏内存延迟。
*   **Pipeline Abstractions**: `PipelineTmaAsync` 和 `PipelineTmaStore` 提供 producer-consumer 同步抽象。
*   **Persistent Tile Scheduler**: 高效地把 work 分发给 CTA, 用于动态负载均衡。
*   **Shared Memory Swizzle**: 使用优化后的 shared memory layout 避免 bank conflict。
*   **Warp Specialization**: 不同 warp 负责 loading 和 transposing；第一个 transpose warp 也负责 storing。

### Kernel 架构

`Sm100MatrixTransposeKernelV2` 执行 tiled matrix transpose（M×N → N×M）, 整体设计如下。

#### Warp 角色

1.  **TMA Load Warp**（Producer）: 使用 TMA load 把 tile 从 Global Memory 加载到 Shared Memory buffer `sA`。
2.  **Transpose Warps**（4 个 warp, Consumer/Producer）:
    *   等待 load pipeline 中的 `sA` 数据。
    *   执行 `sA` → Registers → `sB` 的转置搬运。
    *   通过 named barrier 同步, 保证所有 transpose warp 都完成。
    *   第一个 transpose warp（`trans_warp_id[0]`）发起从 `sB` 到 Global Memory 的 TMA store。

#### Pipeline stages

kernel 使用两个独立 pipeline, 提升并行度。

1.  **Load Pipeline**: `TMA Load Warp`（producer）→ `Transpose Warps`（consumer）。
    *   multi-stage buffer `sA`, stage 数基于可用 SMEM 自动计算。
    *   处理当前 tile 时可以预取后续 tile。
2.  **Store Pipeline**: `Transpose Warps`（producer）→ `First Transpose Warp`（consumer, 发起 TMA store）。
    *   multi-stage buffer `sB`, stage 数基于可用 SMEM 自动计算。
    *   所有 transpose warp 写完 `sB` 后, 第一个 transpose warp 负责 TMA store。

#### Stage 计算

kernel 用 `_compute_stages()` 自动计算 pipeline stage 数。

*   计算 `sA`（`tile_m × tile_n`）和 `sB`（`tile_m × tile_n`）每个 stage 需要的字节数。
*   为 pipeline barrier 和 metadata 预留空间, 约 1KB。
*   用剩余 shared memory 除以每个 stage pair 需要的字节数。
*   把 stage 数限制在 2-8 之间, 2 表示至少 double buffering, 超过 8 收益通常会下降。

#### 执行流程

1.  **Initialization**:
    *   分配 multi-stage shared memory buffers（`sA_staged`、`sB_staged`）。
    *   为 source 和转置后的 destination 创建 TMA descriptor。
    *   初始化 load 和 store pipeline 及其 barrier 同步。
    *   用 persistent tile scheduler 把 work tiles 分发给 CTA。

2.  **TMA Load Warp**（load pipeline 的 producer）:
    ```
    for each tile assigned by scheduler:
        acquire load pipeline 中下一个可用 stage
        发起 TMA load: Global[tile] → sA[stage]
        advance 到下一个 stage
    ```

3.  **Transpose Warps**（load 的 consumer, store 的 producer）:
    ```
    for each tile assigned by scheduler:
        等待 load pipeline 填充当前 stage
        copy data: sA[load_stage] → Registers
        release load pipeline stage
        
        transpose and write: Registers → sB[store_stage]
        fence, 保证 smem 写入可见
        用 barrier 同步所有 transpose warps
        
        [仅第一个 transpose warp] 发起 TMA store: sB[stage] → Global[tile]
        [仅第一个 transpose warp] commit 到 store pipeline
        [仅第一个 transpose warp] acquire 下一个 store pipeline stage
        用 barrier 同步所有 transpose warps
    ```

4.  **Pipeline Teardown**:
    *   producer/consumer tail 操作确保所有 in-flight 操作都完成。

### 主要特性

*   **Automatic Stage Optimization**: kernel 根据 tile size、data type 和可用 shared memory 自动计算 stage 数。
*   **Persistent Tile Scheduler**: 用 persistent scheduling 高效分发 work, 改善不规则矩阵尺寸下的负载均衡。
*   **Memory Layout Optimization**: 对 row-major input 和 column-major transposed output 使用合适的 swizzling。
*   **Efficient Synchronization**: CTA 内协调使用 named barrier, pipeline stage 同步使用 mbarrier。

### 配置参数

*   `tile_shape`: 固定为 (128, 128), 表示 tile 的 (M, N) 尺寸。
*   `cluster_shape_mn`: 固定为 (1, 1), 表示单 CTA 执行。
*   Number of pipeline stages: 根据可用 SMEM 自动计算。

### 用法

用自定义矩阵尺寸运行 transpose kernel：

```bash
# 基本用法, 128×128 matrix
python tma_v2.py

# 自定义尺寸
python tma_v2.py --M 1024 --N 2048

python tma_v2.py --M 1024 --N 2048 --num_warmup 10 --num_iters 50
```

#### 示例代码

```python
from tma_v2 import run_transpose

# 在 1024×2048 matrix 上运行 transpose
run_transpose(M=1024, N=2048)
# 校验通过时输出 "TransposeSuccess!"
```

### 性能注意点

*   **Multi-stage pipelining** 通过重叠 loads、computation 和 stores 隐藏内存延迟。
*   **Persistent scheduling** 对不规则矩阵尺寸提供更好的负载均衡。
*   **Warp specialization** 提高吞吐：TMA load warp 负责所有 load, transpose warps 负责计算, 第一个 transpose warp 负责 store。

## TMA V3: TMA With MMA (Tensor Cores)

待补充。
