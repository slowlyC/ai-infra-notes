## H100 与 B200 架构对比

简单整理了下 Hopper 和 Blackwell 架构上的异同点。

### H100 GPU 多级缓存结构

```
SM (Streaming Multiprocessor)
  寄存器文件 (Register File)
    容量: ~256KB per SM, 共 132 个 SM
    延迟: 0 cycles
    每个线程私有，存储局部变量

  L1 Cache / 共享内存 (Shared Memory)
    容量: 256KB per SM (L1+SMEM 共享池, SMEM 最大可配 228KB)
    延迟: ~30 cycles
    SM 内所有线程共享
    缓存行大小: 128 bytes

    合并访存: 同一 warp 的 32 线程访问连续地址 → 1 次 128B 加载
    分散访存: 32 线程访问随机地址 → 多次加载，带宽浪费

  TLB (地址转换缓存)
    缓存虚拟地址 → 物理地址的映射
    TLB Miss 代价: ~数百 cycles
                    ↓ (片上 → 片外)
L2 Cache (全局共享)
  容量: 50MB
  延迟: ~200 cycles
  所有 SM 共享，用于跨 SM 数据交换

  L2 分区结构:
    [Partition 0] [Partition 1] [Partition 2] ... [Partition N]
    地址按哈希映射到不同分区，分区内访问 < 跨分区访问
                    ↓ (片外高带宽总线)
HBM3 (全局显存)
  容量: 80GB (H100 SXM)
  带宽: 3.35 TB/s
  延迟: ~400-600 cycles

  HBM 多通道结构:
    [Stack 0] [Stack 1] [Stack 2] [Stack 3] [Stack 4]
    5 个 HBM3 Stack，地址交错分布以最大化带宽
```

访问延迟与容量对比:

| 层级 | 延迟 | 容量 | 特点 |
|------|------|------|------|
| 寄存器 | 0 cycles | ~256KB/SM | 线程私有 |
| L1/共享内存 | ~30 cycles | 256KB/SM | SM 内共享 |
| L2 Cache | ~200 cycles | 50MB | 全局共享，分区式 |
| HBM3 全局内存 | ~400-600 cycles | 80GB | 高带宽，高延迟 |

### B200 (Blackwell) 内存层次结构

```
SM (Streaming Multiprocessor)
  寄存器文件 (Register File)
    容量: 256KB per SM (~64K × 32-bit 寄存器), 共 148 个 SM
    延迟: 0 cycles
    每个线程私有

  [新增] Tensor Memory (TMEM)
    容量: 128 行 × 512 列 × 32bit = 256KB per CTA
    专为 TCGen05 (第五代 Tensor Core) MMA 指令设计
    本质上是一个额外的寄存器文件，由 warp group 协同使用

    TMEM 2D 结构:
         列 0    列 1    ...   列 511
    行0  [32bit] [32bit] ...  [32bit]  ← Warp 0 访问行 0-31
    行32 [32bit] [32bit] ...  [32bit]  ← Warp 1 访问行 32-63
    行64 [32bit] [32bit] ...  [32bit]  ← Warp 2 访问行 64-95
    行96 [32bit] [32bit] ...  [32bit]  ← Warp 3 访问行 96-127

    每个 Warp 只能访问 32 行 (按 Warp ID 分配)
    需要 Warp Group (4 个 Warp) 协同才能访问全部 128 行
    按列分配，大小 ∈ [32, 512]，且为 2 的幂

  L1 Cache / 共享内存 (Shared Memory)
    共享内存: 最大可配 228KB per SM
    L1 Cache: 与 SMEM 共用 256KB 池
    延迟: ~28 cycles
    缓存行大小: 128 bytes

    数据流向:
      Global Memory → Shared Memory (TMA 异步复制)
      Shared Memory → Tensor Memory (tcgen05_copy 异步复制) [新增]
      Tensor Memory → Tensor Core (tcgen05_mma 指令)       [新增]

  TLB (地址转换缓存)
    缓存虚拟地址 → 物理地址的映射
    TLB Miss 代价: ~数百 cycles
                    ↓ (片上 → 片外)
L2 Cache (全局共享)
  容量: 50MB
  延迟: 相比 H100 降低约 58% (微基准测试数据)
  所有 SM 共享
                    ↓ (片外高带宽总线)
HBM3e (全局显存)
  容量: 192GB (H100 的 2.4 倍)
  带宽: 8 TB/s (H100 的 2.4 倍)
  延迟: ~500 cycles

  HBM3e 通道结构:
    [Stack 0: 48GB] [Stack 1: 48GB] [Stack 2: 48GB] [Stack 3: 48GB]
    4 个 HBM3e Stack，每 Stack 带宽 2 TB/s，总计 8 TB/s
```

### 内存带宽数据对比

| 层级 | H100 (Hopper) | B200 (Blackwell) | 变化 |
|------|---------------|-------------------|------|
| 寄存器文件 | 256KB/SM, 132 SM, 0 cycles | 256KB/SM, 148 SM, 0 cycles | 不变 |
| Tensor Memory | 无 | 256KB/CTA, tcgen05 专用 | 新增 |
| L1 + 共享内存 | 256KB/SM 共享池, ~30 cycles | 256KB/SM 共享池, ~28 cycles | 延迟略降 |
| TLB | Miss ~数百 cycles | Miss ~数百 cycles | 不变 |
| L2 Cache | 50MB, ~200 cycles | 50MB, 延迟降低约 58% | 延迟大幅降低 |
| HBM | 80GB, 3.35 TB/s, ~500 cycles | 192GB, 8 TB/s, ~500 cycles | 容量和带宽约 2.4 倍 |
| NVLink | 4.0, 900 GB/s | 5.0, 1.8 TB/s | 带宽翻倍 |

### 浮点计算性能对比

| 计算精度 | H100 SXM | B200 SXM | 提升 |
|----------|----------|----------|------|
| FP64 向量 (CUDA Core) | 34 TFLOPS | 40 TFLOPS | +18% |
| FP64 Tensor Core | 67 TFLOPS | 80 TFLOPS | +19% |
| FP32 向量 (CUDA Core) | 67 TFLOPS | 80 TFLOPS | +19% |
| TF32 Tensor Core | 989 TFLOPS | 1,100-1,200 TFLOPS | +11-21% |
| BF16 Tensor Core | 1,979 TFLOPS | 2,250-4,500 TFLOPS | +14-127% |
| FP16 Tensor Core | 1,979 TFLOPS | 2,250-4,500 TFLOPS | +14-127% |
| FP8 Tensor Core | 3,958 TFLOPS | 4,500-9,000 TFLOPS | +14-127% |
| FP4 Tensor Core (新增) | 不支持 | 9,000-18,000 TFLOPS | — |
| INT8 Tensor Core | 3,958 TOPS | 4,500-9,000 TOPS | +14-127% |

注: B200 性能数据存在多个来源，表中给出范围值。实际性能因工作负载而异。

### 稀疏计算性能对比

| 稀疏精度 (2:4 Sparsity) | H100 SXM | B200 SXM | 提升 |
|--------------------------|----------|----------|------|
| TF32 Sparse | 1,979 TFLOPS | 2,200-2,400 TFLOPS | +11-21% |
| BF16/FP16 Sparse | 3,958 TFLOPS | 4,500-9,000 TFLOPS | +14-127% |
| FP8 Sparse | 7,916 TFLOPS | 9,000-18,000 TFLOPS | +14-127% |
| FP4 Sparse (新增) | 不支持 | 18,000-36,000 TFLOPS | — |
| INT8 Sparse | 7,916 TOPS | 9,000-18,000 TOPS | +14-127% |

注: 稀疏性能 = 稠密性能 × 2，利用 2:4 结构化稀疏。

### 互连与 I/O 对比

| 参数 | H100 SXM | B200 SXM | 变化 |
|------|----------|----------|------|
| NVLink 版本 | 4.0 | 5.0 | 新一代 |
| NVLink 总带宽 (双向) | 900 GB/s | 1.8 TB/s | 2 倍 |
| NVLink 链路数 | 18 | 18 | 不变 |
| 单链路带宽 | 50 GB/s | 100 GB/s | 2 倍 |
| PCIe | 5.0 x16, 128 GB/s | 5.0 x16, 128 GB/s | 不变 |
| NVSwitch | 第三代 | 第四代 | 新一代 |
| MIG (多实例 GPU) | 最多 7 实例 | 支持 (待确认) | — |

### 功耗与能效对比

| 参数 | H100 SXM | B200 SXM | 变化 |
|------|----------|----------|------|
| TDP | 700 W | 1,000 W | +43% |
| FP32 能效 | 0.096 TFLOPS/W | ~0.08 TFLOPS/W | 略降 |
| BF16/FP16 Tensor 能效 | 2.83 TFLOPS/W | 2.25-4.5 TFLOPS/W | 视规格 |
| FP8 Tensor 能效 | 5.65 TFLOPS/W | 4.5-9.0 TFLOPS/W | 视规格 |
| 推理吞吐量/W (LLM) | 基准 | ~2.5x 基准 | 显著提升 |
| 冷却方式 | 液冷/风冷 | 主要液冷 | 液冷为主 |

### AI/ML 特性对比

| 特性 | H100 SXM | B200 SXM | 变化 |
|------|----------|----------|------|
| Tensor Core 代数 | 第四代 (wgmma) | 第五代 (tcgen05) | 新一代 |
| 支持精度 | FP64/TF32/BF16/FP16/FP8/INT8 | FP64/TF32/BF16/FP16/FP8/FP4/INT8 | 新增 FP4 |
| Tensor Memory | 无 | 256KB/CTA | 新增 |
| 异步数据路径 | HBM → SMEM → TC | HBM → SMEM → TMEM → TC | 三级流水线 |
| Transformer Engine | v1 | v2 | 增强 |
| FP8 自动混合精度 | 支持 | 支持 | 不变 |
| FP4 支持 | 不支持 | 支持 | 新增 |
| 结构化稀疏 (2:4) | 支持, 2x | 支持, 2x | 不变 |
| TMA | 支持 | 支持 | 不变 |
| cp.async | 支持 | 支持 | 不变 |
| TMEM 异步复制 | 不支持 | 支持 (tcgen05_copy) | 新增 |

### 总结

B200 相比 H100 的核心变化:

**显著提升 (>50%)**
- 显存带宽: 3.35 → 8 TB/s (+139%)
- 显存容量: 80 → 192 GB (+140%)
- L2 访问延迟降低约 58%
- NVLink 带宽: 900 GB/s → 1.8 TB/s (+100%)
- 晶体管数: 800 亿 → 2080 亿 (+160%)

**全新特性**
- Tensor Memory (TMEM): 256KB/CTA，专为第五代 Tensor Core 设计
- FP4 精度支持
- 第五代 Tensor Core (tcgen05): 异步 MMA + TMEM 流水线
- 三级异步数据流: HBM → SMEM → TMEM → Tensor Core

**中等提升 (10-50%)**
- FP64/FP32 性能: +18-19%
- L1 延迟: 30 → 28 cycles

**需要注意**
- 功耗增加: 700W → 1,000W (+43%)
- 散热需求增加，主要依赖液冷
