## Triton-04-Low-Memory Dropout

### 前言

本系列教程记录了学习 Triton 的笔记，参考 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)，翻译并添加了一些自己的理解。

- Triton-01-Vector Addition
- Triton-02-Fused Softmax
- Triton-03-Matrix Multiplication
- **Triton-04-Low-Memory Dropout**（本文）
- Triton-05-Layer Normalization
- Triton-06-Fused Attention
- Triton-07-Extern Functions
- Triton-08-Grouped GEMM
- Triton-09-Persistent Matmul
- Triton-10-Block-Scaled Matmul
- Triton-11-Programmatic Dependent Launch

低内存 Dropout
==================

在本教程中，你将编写一个内存高效的 dropout 实现，其状态仅由一个 int32 种子组成。
这与更传统的 dropout 实现不同，传统实现的状态通常由一个与输入形状相同的位掩码tensor组成。

在此过程中，你将学习到: 

* PyTorch 的朴素 Dropout 实现的局限性。

* Triton 中的并行伪随机数生成。

### 基线实现

*dropout* 算子最初在 [SRIVASTAVA2014]_ 中引入，作为在低数据环境下（即正则化）
提高深度神经网络性能的一种方法。

它接受一个向量作为输入，并产生一个与输出形状相同的向量。输出中的每个标量都有 :math:`p` 
的概率被改为零，否则从输入中复制。这迫使网络即使只有 :math:`1 - p` 的标量可用时也能表现良好。

在评估时，我们想使用网络的全部能力，因此我们设置 :math:`p=0`。简单地说，这会增加输出的范数
（这可能是一件坏事，例如，它可能导致输出 softmax 温度的人为降低）。为了防止这种情况，
我们将输出乘以 :math:`\frac{1}{1 - p}`，这使得无论 dropout 概率如何，范数保持一致。

让我们首先看一下基线实现。

```python
import tabulate
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(
    x_ptr,  # 输入指针
    x_keep_ptr,  # 0 和 1 的掩码指针
    output_ptr,  # 输出指针
    n_elements,  # `x` tensor中的元素数量
    p,  # `x` 中的元素被改为零的概率
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # 下面这一行是关键部分，在上面的段落中描述！
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # 写回输出
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


# 输入tensor
x = torch.randn(size=(10, ), device=DEVICE)
# Dropout 掩码
p = 0.5
# tensor([1, 1, 0, 0, 0, 0, 1, 1, 1, 0], device='cuda:0', dtype=torch.int32)
x_keep = (torch.rand(size=(10, ), device=DEVICE) > p).to(torch.int32)
output = dropout(x, x_keep=x_keep, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))
```

### 带种子的 Dropout

上面的 dropout 实现工作正常，但处理起来可能有点麻烦。首先，我们需要为反向传播存储 dropout 掩码。
其次，在使用重计算/检查点时，dropout 状态管理会变得非常棘手（例如，参见 
https://pytorch.org/docs/stable/checkpoint.html 中关于 `preserve_rng_state` 的所有注释）。
在本教程中，我们将描述一种替代实现，它具有以下优点: 
(1) 内存占用更小；(2) 需要更少的数据移动；(3) 简化了在kernel的多次调用中持久化随机性的管理。

Triton 中的伪随机数生成很简单。在本教程中，我们将使用 :code:`triton.language.rand` 函数，
该函数在给定种子和 :code:`int32` 偏移量块的情况下，生成一个在 [0, 1) 中均匀分布的 
:code:`float32` 值块。但如果需要，Triton 还提供其他 :ref:`随机数生成策略<Random Number Generation>`。

.. note::
   Triton 的 PRNG 实现基于 Philox 算法（在 [SALMON2011]_ 中描述）。

让我们把它们放在一起。

```python
@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # 计算此实例处理的元素的内存偏移量
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 从 x 加载数据
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # 随机修剪它
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # 写回
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10, ), device=DEVICE)
# 与基线相比 - dropout 掩码从未实例化！
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))
```

我们有了一个 triton kernel，只要种子相同，就会应用相同的 dropout 掩码。
如需探索 GPU 编程中伪随机性的应用，可参考 `python/triton/language/random.py`。

### 练习

1. 扩展kernel以在矩阵上操作，并使用种子向量 - 每行一个种子。
2. 添加对跨步的支持。
3. （挑战）实现一个稀疏 Johnson-Lindenstrauss 变换的kernel，每次使用种子动态生成投影矩阵。

### 参考文献

.. [SALMON2011] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", 2011
.. [SRIVASTAVA2014] Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014
