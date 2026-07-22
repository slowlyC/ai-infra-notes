# NVIDIA Rubin GPU 架构：原文中文概述

## 从对话式 AI 到持续运行的 Agentic AI

AI 系统正在从一次提问、一次回答的交互方式，转向持续执行的 Agentic workflow。Agent 会在较长时间内反复推理、规划、调用工具、校验中间结果，并在大规模上下文中完成多步任务。

这类负载要求系统同时具备较低的单步延迟、较高的 decode 吞吐、长上下文 Attention 效率、足够大的 KV cache，以及在紧耦合 GPU 域中扩展大模型的能力。NVIDIA 因而把数据中心视为一个整体计算单元，并以 Vera Rubin 平台承载这类工作负载。

Rubin GPU 是该平台的计算基础。NVIDIA 表示，在其内部 2T 参数 MoE workload 上，Vera Rubin 平台相对 Blackwell 可提供最高 10 倍的单位能耗 Agentic throughput。这个结果来自 GPU、CPU、内存、互连和机架系统的协同设计。

[![Figure 1：Hopper、Blackwell 与 Rubin 的 Agent throughput 和 interactivity](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/hopper-blackwell-rubin-throughput-interactivity-comparison.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/hopper-blackwell-rubin-throughput-interactivity-comparison.webp)

*Figure 1：Vera Rubin 平台在 NVIDIA 内部 2T MoE workload 上的 Agentic inference Pareto frontier。图片来源：NVIDIA。*

## Rubin GPU 如何支撑 Agentic workload

Rubin GPU 由两颗接近光罩面积上限的 compute die 组成，两颗 die 通过同一封装内的 NV-HBI（NVIDIA High-Bandwidth Interface）连接。原文给出的主要规格如下。

| 项目 | Rubin GPU 规格 |
|---|---:|
| 晶体管数量 | 3360 亿 |
| SM 数量 | 224 |
| Tensor Core 数量 | 896 |
| Transformer Engine | 第三代 |
| NVFP4 推理性能 | 最高 50 PFLOPS |
| HBM | 最高 288 GB HBM4，12-Hi stack |
| HBM 峰值带宽 | 最高 22 TB/s |
| NVLink 6 scale-up 带宽 | 3600 GB/s |
| NVLink-C2C 带宽 | 1800 GB/s |
| Host I/O | PCIe Gen 6 x16，最高 256 GB/s |

[![Figure 2：NVIDIA Rubin GPU 芯片结构](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/nvidia-rubin-gpu-chip-architecture-1.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/nvidia-rubin-gpu-chip-architecture-1.webp)

*Figure 2：Rubin GPU 芯片结构。图片来源：NVIDIA。*

3360 亿晶体管、224 个 SM 和 896 个 Tensor Core 提供计算密度。第三代 Transformer Engine 可以在多种数值格式之间调整精度，使 Rubin 的 NVFP4 推理性能最高达到 50 PFLOPS，同时兼顾模型精度。

GPU 内部以 GPC 组织计算资源，并配置集中式 L2 cache。GigaThread Engine 负责工作调度，MIG Control 支持一张 GPU 上的多工作负载分区，NV-DEC 负责解码处理。这些单元共同提升不同 Agentic 阶段之间切换时的计算利用率。

数据路径方面，Rubin 配置最高 288 GB HBM4 和 22 TB/s 峰值带宽。增强的 TMA 负责复杂布局下的数据搬运；NVLink 6 连接 NVLink Switch，提供 GPU-GPU all-to-all 通信；NVLink-C2C 提供 coherent CPU-GPU 通信；PCIe Gen 6 x16 则提供 Host 连接。

Rubin 还将 Confidential Computing 与 TEE-I/O 纳入平台设计，用于保护静态存储、链路传输和计算使用过程中的数据。

## Rubin 如何加速推理中的主要路径

原文认为，峰值计算能力并不足以决定 Agentic inference 性能。数据搬运、矩阵计算、长上下文 Attention，以及相互依赖的 kernel 如何衔接，都会影响端到端效率。

### 加速机架级 MoE 权重和 token 搬运

MoE 会把 token 动态路由到不同 expert。Expert 数量增加后，查找和移动 expert 权重产生的开销也随之上升。

不同 expert 的 tensor 往往具有相同 layout，只是位于不同内存地址。Rubin 为 TMA 增加 inline descriptor update 支持。Kernel 可以复用一个描述公共 layout 的 descriptor，并在执行 TMA 指令时覆盖 memory pointer、stride 等字段，不必在内存中为每个 expert 修改或维护一份 descriptor。

[![Figure 3：Rubin 简化 MoE descriptor 共享](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/rubin-gpu-moe-descriptor-sharing-1.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/rubin-gpu-moe-descriptor-sharing-1.webp)

*Figure 3：Blackwell 为不同 expert 使用不同 descriptor，Rubin 允许多个 expert 共享 descriptor。图片来源：NVIDIA。*

这样可以减少 metadata 管理和数据搬运开销，让更多 GPU 时间用于实际推理计算，并改善大规模 MoE 的扩展效率。

### 提高机架级矩阵运算效率

Rubin 将 Tensor Core 每个时钟周期沿 K 维处理的数据量提高到 Blackwell 的两倍。模型分布到更多 GPU 后，每张 GPU 负责的输出通常变小，但 reduction 维 K 依然较长，因此减少 K-loop 次数对高 Tensor Parallel 场景尤其有利。

原文示例中，Blackwell 需要四轮 K 迭代的 GEMM，Rubin 用两轮即可完成。更少的迭代可以降低循环开销，提高 Tensor Core 利用率，并改善 context 和 decode 阶段 GEMM 的执行效率。

[![Figure 4：Rubin 将 K 维指令吞吐提高一倍](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/rubin-k-dimension-instruction-throughput-1.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/rubin-k-dimension-instruction-throughput-1.webp)

*Figure 4：Rubin 用两轮 K-loop 完成 Blackwell 四轮所覆盖的工作。图片来源：NVIDIA。*

### 处理长上下文 Attention

上下文增长后，Attention 需要处理更大的 score matrix，执行更多 Softmax normalization，并将结果与 V 相乘。Rubin 通过 activation sparsity、adaptive compression 和更高的 Softmax 吞吐来缩短这条路径。

原文给出的处理方式从 dense `QK^T` 开始。Rubin 从 Tensor Memory 读取中间 attention score 时，可将其转换成结构化 2:4 稀疏表示，同时生成非零值和对应 metadata。Softmax 随后只处理保留值，第二个 Attention GEMM 再使用 sparse MMA 将结果与 dense V 相乘。最终输出仍保持下游模型所需的 dense layout。

[![Figure 5：Rubin activation sparsity 与 adaptive compression](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/nvidia-rubin-adaptive-compression-sparsity-1.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/nvidia-rubin-adaptive-compression-sparsity-1.webp)

*Figure 5：Activation 被压缩为非零值和 metadata，并用于 Attention 或 MLP。图片来源：NVIDIA。*

这种中间表示减少了 attention score 的写入量、存储量、Softmax 工作量和第二次 GEMM 的计算量。相同机制也可以应用于 MLP activation。

随着 Tensor Core 变快，依赖 exponential 和 reduction 的 Softmax 更容易成为瓶颈。原文给出的每 SM、每 clock 相对 exponential 吞吐如下。

| GPU 平台 | FP32 | BF16/FP16 |
|---|---:|---:|
| Blackwell | 1x | 1x |
| Blackwell Ultra | 2x | 2x |
| Rubin | 2x | 4x |

Rubin 的 activation sparsity 减少中间数据量，更高的 exponential throughput 则让 Softmax 更好地跟上矩阵计算速度。

### 改善依赖 kernel 的执行效率

推理过程中，一个 kernel 生成 activation 后，后继 kernel 才能继续执行。传统 producer-consumer 调度可能要求 consumer 等待较大范围的 producer 工作完成，从而在 GPU timeline 中留下空闲区间。

Blackwell 的 Programmatic Dependent Launch 已允许 consumer kernel 提前启动，但依赖数据尚未就绪时仍需等待。Rubin 提供更细粒度的 dependent kernel coordination，使 consumer 可以在所需输入 tile 可用后开始对应工作，而不必等待更大批次的 producer 工作全部完成。

[![Figure 6：Blackwell 与 Rubin 的 producer-consumer timeline](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/blackwell-rubin-timelines-producer-consumer-thread-blocks-1.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/blackwell-rubin-timelines-producer-consumer-thread-blocks-1.webp)

*Figure 6：Blackwell 使用较粗粒度触发，Rubin 按数据 tile 让 consumer 更早开始。图片来源：NVIDIA。*

更早的 consumer 进度可以压缩 kernel 之间的空洞，并提高相互依赖 kernel 的重叠程度。对于 activation 串行流经模型的 Agentic inference，这会直接影响单用户 token throughput。

## 内存与通信如何维持推理吞吐

模型、上下文和 GPU 域持续增大后，权重、activation、KV cache 和通信数据的流动效率与计算能力同样重要。

### 加速 scale-up 通信

通信融合进 GPU kernel 后，kernel 可以在计算进行期间直接通过 NVLink 向远端 GPU 写入数据或执行 reduction，不必每次返回 CPU。传统做法除了搬运 payload，还需要 barrier、acknowledgment 或 atomic flag 等同步操作。

Rubin 为 device-initiated NVLink communication 引入 counted writes。接收 GPU 可以通过 counter 更高效地判断传输完成状态，减少同步相关的延迟和互连流量。

[![Figure 7：Rubin 使用 counted writes 加速 NVLink 通信](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/rubin-nvlink-communication-acceleration.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/rubin-nvlink-communication-acceleration.webp)

*Figure 7：Rubin 用 counted writes 简化 GPU-GPU 数据传输的完成通知。图片来源：NVIDIA。*

### 为 decode 共同设计内存子系统

推理的 decode 阶段通常受内存子系统限制。Agentic workload 会在较长上下文、大 KV cache 和交互式 token generation 上花费更多时间，因此 achieved memory bandwidth 会显著影响端到端性能。

Rubin 使用 HBM4，接口宽度相对 HBM3e 翻倍。配合新的 memory controller、内存生态协同设计和更紧密的 compute-memory 集成，其峰值带宽达到 22 TB/s，是 Blackwell 和 Blackwell Ultra 的 2.8 倍。

[![Figure 8：从 Blackwell 到 Rubin 的内存带宽](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/memory-bandwidth-nvidia-rubin-gpu.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/memory-bandwidth-nvidia-rubin-gpu.webp)

*Figure 8：Rubin 的 HBM4 峰值带宽提高到 22 TB/s。图片来源：NVIDIA。*

最高 288 GB HBM4 容量用于容纳模型、扩大上下文和 KV cache、提高并发，并减少 KV cache offload。更高带宽则服务 token-by-token generation，在 decode 期间快速搬运权重和 KV state。TMA 与 memory-locality 策略用于提高复杂 layout 下的内存子系统利用率。

## 面向能效、扩展性和可靠性的系统设计

Vera Rubin NVL72 将 Rubin GPU 扩展成一个机架级执行域，使计算、网络、液冷、供电和运维共同服务于固定功率预算下的有效 token 产出。

### 在相同功率预算内容纳更多 GPU

AI workload 的功率需求会快速变化。短时功率尖峰会迫使数据中心预留供电能力，导致一部分容量长期不能用于计算。Vera Rubin 电源采用带 state-of-charge 管理的 Intelligent Power Smoothing，通过储能吸收峰值并平滑输入功率。

[![Figure 9：Intelligent Power Smoothing](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/gpu-power-chart.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/gpu-power-chart.webp)

*Figure 9：储能覆盖短时功率尖峰，使持续 AC 输入更加平稳。图片来源：NVIDIA。*

相较上一代 power-smoothing 技术，原文称该方案可将平均功耗降低约 10%，并将 50 ms 峰值功耗降低约 20%。更平稳的功率曲线可以降低持续最大功率需求，在相同 AI factory 功率预算下配置更多计算资源。

DSX MaxLPS 将功率控制扩展到 GPU、机架、45°C 液冷和 workload，DSX OS 负责调度、生命周期管理和健康自动化。NVIDIA 表示，在 energy-efficient operating point 上，DSX MaxLPS 最多可以让同一功率预算容纳 40% 更多 GPU，同时只对 workload 性能产生较小影响。

[![Figure 10：DSX MaxLPS 固定功率预算下的 GPU 数量](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/power-budget-comparison-dsx-maxlps-1.webp)](https://developer-blogs.nvidia.com/wp-content/uploads/2026/07/power-budget-comparison-dsx-maxlps-1.webp)

*Figure 10：DSX MaxLPS 利用原本被功率峰值占用的容量部署更多 GPU。图片来源：NVIDIA。*

### 为可靠性和数据中心扩展设计机架

Vera Rubin NVL72 的第三代 MGX 机架采用 cable-free compute 和 switch tray、45°C 液冷、动态机架级 power steering 与 Intelligent Power Smoothing，使计算、网络、散热和供电作为一个系统运行。

可热插拔 NVLink switch tray 和增强的 RAS 能力用于提高大规模运行的可靠性。NVLink 提供机架内的紧耦合 scale-up 连接，Spectrum-X Ethernet 则支持 pod 和多机架系统之间的灵活连接。

## 原文结论

Rubin 面向的是长上下文推理、多步生成、分布式 MoE decode 和低延迟交互持续运行的场景。NVIDIA 将计算、内存、网络、机架供电、散热和软件共同设计，目标是减少数据、通信和功率约束造成的等待，并在固定功率预算内完成更多有效 token 和 Agentic work。

## 参考资料

- [Inside NVIDIA Rubin GPU Architecture: Powering the Era of Agentic AI](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/)
- [NVIDIA Rubin GPU 架构：注解导读](./NVIDIA_Rubin_GPU_Architecture_Annotated.md)
