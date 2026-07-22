# CuTeDSL 分布式示例

这个目录包含使用 CuTeDSL 和 NVSHMEM 实现多 GPU 通信的分布式示例。目前还不支持在 device 侧通过 NVSHMEM 实现 copy/put/get，这里只使用 host 侧初始化和内存分配。

## 阅读顺序

本目录建议按下面顺序阅读。

1. `all_reduce_simple.py`，最小 all-reduce 示例。它通过 `nvshmem.core.get_peer_tensor()` 取 peer tensors，在 kernel 里逐 rank load、register accumulation，再写回 local output。
2. `all_reduce_one_shot_lamport.py`，通信 buffer 风格的 one-shot all-reduce。重点看 ping-pong buffer、negative zero sentinel、`.SYS` memory scope 和 `.VOLATILE` memory order。
3. `all_reduce_two_shot_multimem.py`，基于 `multimem` 的 two-shot all-reduce。重点看 `multimem.ld_reduce`、`multimem.st`、`multimem.red` 以及 flag 同步。
4. `all_reduce_tma.py`，把 two-shot all-reduce 和 TMA pipeline 结合起来。重点看每个 CTA 从所有 rank TMA load、register accumulation、TMA multicast store 和 cross-GPU barrier。
5. `distributed_gemm_all_reduce_blackwell.py`，接着看 GEMM + all-reduce epilogue。它延续前面的 two-shot tile ownership，重点看 `use_tma_store=True` 时如何在 epilogue 里接入 `LDMCxSTMC`，让每个 rank 只计算一部分 output，再将归约结果 broadcast 到所有 ranks。
6. `distributed_all_gather_gemm_blackwell.py`，再看 all-gather + dense GEMM。这里通信从 output epilogue 转到 GEMM input 一侧，适合在理解 GEMM + all-reduce 后对比两种融合位置。
7. `distributed_gemm_reduce_scatter_blackwell.py`，最后看 reduce-scatter 版本。它和 all-reduce GEMM 很接近，但多了按 rank 分片输出和 tile ownership 的理解负担。

## All-Reduce 实现对比

`one-shot` 和 `two-shot` 描述的是 collective algorithm 的阶段与工作划分，`TMA` 和 `multimem` 描述的是 data movement primitive。因此，`all_reduce_tma.py` 在算法上也是 two-shot，只是它用逐 rank TMA load、SMEM pipeline 和显式加法替代了 `multimem.ld_reduce`。

| 实现 | 数据流与工作划分 | 每个 rank 的额外通信存储 | 适用场景 |
|------|------------------|--------------------------|----------|
| `all_reduce_simple.py` | 每个 rank 直接读取所有 peer inputs，并重复计算完整 output | 无 tensor-sized communication buffer | 教学和 correctness baseline；world size 较小，且 input 可被所有 peers 直接访问 |
| `all_reduce_one_shot_lamport.py` | 每个 rank 将本地 input fan-out 到所有 ranks 的 inbox，再从本地 inbox 归约完整 output | 当前实现约为 `PING_PONG_SIZE * world_size * tensor_size`，其中 `PING_PONG_SIZE=3` | 小消息、低延迟；原始 input 不能被远端直接访问，但可以使用 symmetric communication buffer |
| `all_reduce_two_shot_multimem.py` | 每个 rank 只归约 `1 / world_size` 的 output，使用 `multimem.ld_reduce` reduce，再用 `multimem.st` broadcast | 无 tensor-sized communication buffer，只有同步 flags | 支持 NVLS/NVSwitch 的单机中大消息和高吞吐场景 |
| `all_reduce_tma.py` | 使用 two-shot tile ownership；逐 rank TMA load 到两级 SMEM，consumer 显式累加，再 TMA multicast store | 无 tensor-sized global communication buffer；每个 CTA 使用两级 SMEM | 学习 distributed TMA pipeline，或需要在 SMEM/RMEM 中插入自定义转换和归约逻辑 |

四种实现的数据流可以简化为：

```text
simple:
  all peer GMEM -> per-thread RMEM -> ADD -> local full output

one-shot Lamport:
  local input -> fan-out to all symmetric inboxes -> local ADD -> local full output

two-shot multimem:
  assigned output slice -> multimem.ld_reduce -> multimem.st -> all ranks

two-shot TMA:
  assigned output tile -> peer TMA loads -> SMEM ping-pong -> RMEM ADD
                       -> TMA multicast store -> all ranks
```

性能没有脱离 message size、world size、dtype 和 GPU topology 的固定排序。一般情况下：

- 小消息和较少 ranks 下，`simple` 与 one-shot 都应以实测延迟为准；one-shot 用额外显存换取 communication buffer 和单阶段 fan-out。
- 支持 NVLS 时，two-shot multimem 通常更适合中大消息和吞吐优先的场景，因为每个输出元素只归约一次，并把 reduce/broadcast 卸载到 NVLink Switch。
- 当前 TMA 版本用于教学，并非面向 standalone All-Reduce 性能优化；它的价值主要是 pipeline 和自定义融合能力。
- `distributed_gemm_all_reduce_blackwell.py` 使用的是 two-shot multimem 路径，即 `LDMCxSTMC`，不是 `all_reduce_tma.py` 的软件累加路径。

## NVSHMEM 依赖

这些示例需要两个组件。

1. **NVSHMEM4Py** (`nvshmem4py-cu12` / `nvshmem4py-cu13`): 提供 NVIDIA NVSHMEM 官方 Python binding 的 Python 包。参见 [NVSHMEM4Py Documentation](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/index.html)。

2. **NVSHMEM Library** (`nvidia-nvshmem-cu12` / `nvidia-nvshmem-cu13`): 底层 native library，包含实际的 NVSHMEM 实现。

### 概览

**NVSHMEM4Py** (`nvshmem4py-cu12` / `nvshmem4py-cu13`) 是一个 Python binding library，为 NVSHMEM 功能提供 Pythonic 接口。在这些示例中，它主要用于：

- 分配支持跨 GPU peer-to-peer (P2P) 通信的 tensor
- 分配 multicast (MC) tensor，以便利用 `multimem` 指令执行高效 collective operations

**nvidia-nvshmem** (`nvidia-nvshmem-cu12` / `nvidia-nvshmem-cu13`) 是底层 library，它把 NVSHMEM functions 封装到 dynamic libraries (`.so` files) 中。NVSHMEM4Py 会在运行时动态加载并调用这些 library。

### 安装

CUDA 12:
```bash
pip install nvshmem4py-cu12 nvidia-nvshmem-cu12
```

CUDA 13:
```bash
pip install nvshmem4py-cu13 nvidia-nvshmem-cu13
```

> **注意:** 推荐使用 `nvshmem4py` version >= 0.1.3。

### 使用的主要 API

这里主要使用 `nvshmem.core` 中的以下 API。

| API | 说明 |
|-----|-------------|
| `nvshmem.core.tensor(shape, dtype)` | 分配支持 P2P communication 的 symmetric tensor |
| `nvshmem.core.get_peer_tensor(tensor, pe)` | 返回 tensor handle，用于访问 remote PE (processing element) 上的指定 tensor |
| `nvshmem.core.get_multicast_tensor(tensor)` | 返回可通过 `multimem` 指令访问的 tensor，用于高效 multicast operations |
| `nvshmem.core.free_tensor(tensor)` | 显式释放已分配的 symmetric memory |

### 内存管理

NVSHMEM 需要**手动管理内存**。PyTorch tensor 会自动 garbage-collect，而 NVSHMEM symmetric memory 必须通过 `nvshmem.core.free_tensor()` 显式释放，否则可能造成 memory leak。

示例：
```python
import nvshmem.core

# 初始化环境
# 参考示例中的 torchrun_uid_init_bcast()

# 分配 symmetric tensor
local_tensor = nvshmem.core.tensor((M, N), dtype=torch.float32)

# 获取用于 P2P 访问的 peer tensors
tensor_list = [nvshmem.core.get_peer_tensor(local_tensor, rank) for rank in range(world_size)]

# ... 使用 tensors ...

# 使用完成后显式释放内存
for t in tensor_list:
    nvshmem.core.free_tensor(t)

# 结束环境
# 参考示例中的 torchrun_finalize()

```

## Multimem 指令

这些示例展示如何使用 NVIDIA `multimem` PTX instructions 执行高效的多 GPU collective operations。`multimem` 指令作用于通过 `nvshmem.core.get_multicast_tensor()` 获取的 multicast (MC) addresses，从而实现跨多 GPU 的硬件加速通信。

### 为什么 Multimem 快: NVLS (NVLink SHARP)

`multimem` 指令利用 **NVLS (NVLink SHARP)** 技术执行 **in-network computation**。当多个 GPU 映射同一块 symmetric memory region 时，`multimem` 指令可以作用于 multicast address，直接在 NVLink/NVSwitch fabric 中执行硬件加速的 reduction 或 broadcast operations，而不需要先把数据传回 GPU memory。

**主要收益：**
- **In-network computation**: Reduction 和 broadcast operations 在 NVSwitch hardware 中完成，而不是在 GPU compute units 中完成
- **Reduced memory traffic**: 数据在 interconnect 中 in-flight 处理，减少 HBM bandwidth 消耗
- **Lower latency**: 单条指令替代多次 loads/stores 和 arithmetic operations

### 指令类别

这些示例使用三类 `multimem` 指令。

#### 1. `multimem.ld_reduce` - Reduction

从 multicast address 读取数据，并返回所有 GPU 上的 **reduced result**（例如 sum）：

```
multimem.ld_reduce.sys.relaxed.global.add.v4.f32 {$0, $1, $2, $3}, [$4];
```

这条指令从 multicast address 读取数据，并对所有通过 NVLS 映射该地址的 GPU 执行 sum reduction (`.add`)。

**Accumulator Precision**: 对于 lower-precision data types，可以指定更高的 accumulator precision 来提升数值精度：
- **FP16 / BF16**: 可以使用 FP32 accumulator (`.acc::f32`)
- **FP8 (E4M3 / E5M2)**: 可以使用 FP16 accumulator (`.acc::f16`)

FP16 使用 FP32 accumulator 的示例：
```
multimem.ld_reduce.sys.relaxed.global.add.acc::f32.v4.f16x2 {$0, $1, $2, $3}, [$4];
```

#### 2. `multimem.st` - Broadcast via Store

把数据存到 multicast address，从而将数据 **broadcast** 到所有参与的 GPU：

```
multimem.st.sys.relaxed.global.v4.f32 [$1], {$2, $3, $4, $5};
```

这会把数据写入 multicast address，使数据对所有通过 NVLS 映射该地址的 GPU 可见。

#### 3. `multimem.red` - Broadcast via Atomic Reduction

在 multicast address 上执行 atomic reduction operation。这通常用于跨 GPU 的 **signaling/synchronization**：

```
multimem.red.release.sys.global.add.u32 [$0], 1;
```

这条指令会对 multicast address 原子加上一个值。配合同步模式（例如 spin locks）使用时，它可以实现高效的 inter-GPU barrier，让所有 GPU 都能观察到更新后的值。

## 后续工作

`nvidia-nvshmem-cu12/cu13` packages 包含 LLVM IR bitcode libraries，未来可能可以集成到 CuTeDSL 中。这样就能在 CuTeDSL kernels 内直接调用 NVSHMEM functions，从 kernel 层面对 communication patterns 做更细粒度的控制。

## 参考资料

- [NVSHMEM4Py Documentation](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/index.html)
- [NVSHMEM API Reference](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/index.html)
- [multimem PTX instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem)
