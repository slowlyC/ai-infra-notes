# CuTeDSL 分布式示例

这个目录包含使用 CuTeDSL 和 NVSHMEM 实现多 GPU 通信的分布式示例。目前还不支持在 device 侧通过 NVSHMEM 实现 copy/put/get，这里只使用 host 侧初始化和内存分配。

## 阅读顺序

本目录建议按下面顺序阅读。

1. `all_reduce_simple.py`，最小 all-reduce 示例。它通过 `nvshmem.core.get_peer_tensor()` 取 peer tensors，在 kernel 里逐 rank load、register accumulation，再写回 local output。
2. `all_reduce_one_shot_lamport.py`，通信 buffer 风格的 one-shot all-reduce。重点看 ping-pong buffer、negative zero sentinel、`.SYS` memory scope 和 `.VOLATILE` memory order。
3. `all_reduce_two_shot_multimem.py`，基于 `multimem` 的 two-shot all-reduce。重点看 `multimem.ld_reduce`、`multimem.st`、`multimem.red` 以及 flag 同步。
4. `all_reduce_tma.py`，把 two-shot all-reduce 和 TMA pipeline 结合起来。重点看每个 CTA 从所有 rank TMA load、register accumulation、TMA multicast store 和 cross-GPU barrier。
5. `distributed_all_gather_gemm_blackwell.py`，先看这个 fused GEMM 示例。它是 all-gather + dense GEMM，通信和 GEMM 的耦合相对清楚，适合作为进入大文件的入口。
6. `distributed_gemm_all_reduce_blackwell.py`，再看 GEMM + all-reduce epilogue。重点看 `use_tma_store=True` 时如何在 epilogue 里接入 `LDMCxSTMC`。
7. `distributed_gemm_reduce_scatter_blackwell.py`，最后看 reduce-scatter 版本。它和 all-reduce GEMM 很接近，但多了按 rank 分片输出和 tile ownership 的理解负担。

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
